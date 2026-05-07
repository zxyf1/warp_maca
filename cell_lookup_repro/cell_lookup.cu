/*
 * Minimal standalone CUDA C++ reproduction of test_cell_lookup.
 *
 * Stripped of Warp framework. Compiled offline with nvcc (not NVRTC).
 * NOTE: If the bug does not reproduce here, try run.py which uses NVRTC,
 * the same JIT compiler that Warp uses. The MetaX predicated-write bug
 * may be specific to NVRTC's optimizer.
 *
 * Source mapping:
 *   Vec2 / Coords         ← wp.vec2 / wp.vec3 (Coords)
 *   seg_closest()         ← closest_point.py project_on_seg_at_origin
 *   tri_closest()         ← closest_point.py project_on_tri_at_origin (2D)
 *   cell_lookup_buggy()   ← geometry.py make_filtered_cell_lookup (before fix)
 *   cell_lookup_fixed()   ← geometry.py make_filtered_cell_lookup (after fix)
 *   gen_trimesh()         ← test_fem.py _gen_trimesh(3,3) + utils.py grid_to_tris
 *
 * Build:
 *   make
 * Run:
 *   ./cell_lookup
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/* ── Basic types ──────────────────────────────────────────────────────────── */

struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };   /* barycentric coords (≡ Warp Coords) */

static __host__ __device__ Vec2 v2(float x, float y) {
    Vec2 v; v.x = x; v.y = y; return v;
}
static __host__ __device__ Vec3 v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

static __host__ __device__ float dot2(Vec2 a, Vec2 b) { return a.x*b.x + a.y*b.y; }
static __host__ __device__ Vec2  sub2(Vec2 a, Vec2 b) { return v2(a.x-b.x, a.y-b.y); }
static __host__ __device__ Vec2  mul2(float s, Vec2 a) { return v2(s*a.x, s*a.y); }

static __host__ __device__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ── Geometry ─────────────────────────────────────────────────────────────── */
/* Direct translation of closest_point.py                                      */

static __device__ void seg_closest(Vec2 q, Vec2 seg, float len_sq,
                                    float *out_dist, float *out_t)
{
    float t = clampf(dot2(q, seg) / len_sq, 0.0f, 1.0f);
    Vec2 diff = sub2(q, mul2(t, seg));
    *out_dist = dot2(diff, diff);
    *out_t    = t;
}

/*
 * project_on_tri_at_origin (2D variant of closest_point.py).
 * Returns squared distance and barycentric coords via out pointers.
 */
static __device__ void tri_closest(Vec2 q, Vec2 e1, Vec2 e2,
                                    float *out_dist, Vec3 *out_coords)
{
    float e1e1 = dot2(e1, e1);
    float e1e2 = dot2(e1, e2);
    float e2e2 = dot2(e2, e2);
    float det  = e1e1 * e2e2 - e1e2 * e1e2;

    if (det > e1e1 * e2e2 * 1.0e-6f) {
        float e1p = dot2(e1, q);
        float e2p = dot2(e2, q);
        float s   = (e2e2 * e1p - e1e2 * e2p) / det;
        float t   = (e1e1 * e2p - e1e2 * e1p) / det;
        if (s >= 0.0f && t >= 0.0f && s + t <= 1.0f) {
            Vec2 diff = sub2(q, v2(s*e1.x + t*e2.x, s*e1.y + t*e2.y));
            *out_dist   = dot2(diff, diff);
            *out_coords = v3(1.0f - s - t, s, t);
            return;
        }
    }

    float d1, t1, d2, t2, d12, t12;
    seg_closest(q, e1, e1e1, &d1, &t1);
    seg_closest(q, e2, e2e2, &d2, &t2);
    Vec2 e12 = sub2(e2, e1);
    seg_closest(sub2(q, e1), e12, dot2(e12, e12), &d12, &t12);

    if (d1 <= d2) {
        if (d1 <= d12) {
            *out_dist   = d1;
            *out_coords = v3(1.0f - t1, t1, 0.0f);
            return;
        }
    } else if (d2 <= d12) {
        *out_dist   = d2;
        *out_coords = v3(1.0f - t2, 0.0f, t2);
        return;
    }

    *out_dist   = d12;
    *out_coords = v3(0.0f, 1.0f - t12, t12);
}

/* ── Buggy kernel ─────────────────────────────────────────────────────────── */
/*
 * Mirrors make_filtered_cell_lookup in geometry.py BEFORE the wp.where fix.
 *
 * The `if (dist <= closest_dist) { closest_coords = coords; }` block causes
 * a predicated write to each float component of the Vec3 struct.
 * On MetaX, the NVRTC compiler incorrectly handles this: some components
 * may be silently skipped, leaving closest_coords as the sentinel (-1e8,...).
 */
__global__ void cell_lookup_buggy(
    const Vec2* positions,
    const int3* tri_indices,
    int         n_tris,
    const Vec2* query_pos,
    int*        out_cell,
    Vec3*       out_coords,
    int         n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec2 pos          = query_pos[i];
    int  closest_cell = -1;
    Vec3 closest_coords = v3(-1.0e8f, -1.0e8f, -1.0e8f);  /* OUTSIDE sentinel */

    float pad = 1.0e-5f;

    /* Outer: expand search radius — mirrors `while closest_cell == NULL` */
    while (closest_cell == -1) {
        float closest_dist = pad * pad;

        /* Inner: test each candidate — mirrors `while bvh_query_next(...)` */
        for (int cell = 0; cell < n_tris; cell++) {
            int3 vidx = tri_indices[cell];
            Vec2 p0   = positions[vidx.x];
            Vec2 p1   = positions[vidx.y];
            Vec2 p2   = positions[vidx.z];

            /* Cheap AABB filter substituting BVH candidate enumeration */
            float lo_x = fminf(fminf(p0.x, p1.x), p2.x) - pad;
            float lo_y = fminf(fminf(p0.y, p1.y), p2.y) - pad;
            float hi_x = fmaxf(fmaxf(p0.x, p1.x), p2.x) + pad;
            float hi_y = fmaxf(fmaxf(p0.y, p1.y), p2.y) + pad;
            if (pos.x < lo_x || pos.y < lo_y || pos.x > hi_x || pos.y > hi_y)
                continue;

            Vec2  q  = sub2(pos, p0);
            Vec2  e1 = sub2(p1,  p0);
            Vec2  e2 = sub2(p2,  p0);
            float dist;
            Vec3  coords;
            tri_closest(q, e1, e2, &dist, &coords);

            /*
             * BUG: MetaX NVRTC compiler incorrectly handles the predicated
             * write to closest_coords.{x,y,z} inside this if block.
             * The compiler generates `@p st` or `@p mov` instructions for
             * each component, but some may be silently skipped.
             */
            if (dist <= closest_dist) {
                closest_dist   = dist;
                closest_cell   = cell;
                closest_coords = coords;   /* ← predicated Vec3 write */
            }
        }

        if (pad >= 1.0e6f) break;
        pad = fminf(4.0f * pad, 1.0e6f);
    }

    out_cell[i]   = closest_cell;
    out_coords[i] = closest_coords;
}

/* ── Fixed kernel ─────────────────────────────────────────────────────────── */
/*
 * Mirrors geometry.py AFTER the wp.where fix.
 *
 * Each component is updated with a ternary `? :` expression.
 * In PTX this compiles to individual `selp.f32` instructions — unconditional
 * selects that read both operands before choosing. No predicated block is
 * emitted, so the MetaX bug is avoided.
 *
 * Equivalent to:
 *   closest_coords[k] = wp.where(is_closer, coords[k], closest_coords[k])
 */
__global__ void cell_lookup_fixed(
    const Vec2* positions,
    const int3* tri_indices,
    int         n_tris,
    const Vec2* query_pos,
    int*        out_cell,
    Vec3*       out_coords,
    int         n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec2 pos            = query_pos[i];
    int  closest_cell   = -1;
    Vec3 closest_coords = v3(-1.0e8f, -1.0e8f, -1.0e8f);

    float pad = 1.0e-5f;

    while (closest_cell == -1) {
        float closest_dist = pad * pad;

        for (int cell = 0; cell < n_tris; cell++) {
            int3 vidx = tri_indices[cell];
            Vec2 p0   = positions[vidx.x];
            Vec2 p1   = positions[vidx.y];
            Vec2 p2   = positions[vidx.z];

            float lo_x = fminf(fminf(p0.x, p1.x), p2.x) - pad;
            float lo_y = fminf(fminf(p0.y, p1.y), p2.y) - pad;
            float hi_x = fmaxf(fmaxf(p0.x, p1.x), p2.x) + pad;
            float hi_y = fmaxf(fmaxf(p0.y, p1.y), p2.y) + pad;
            if (pos.x < lo_x || pos.y < lo_y || pos.x > hi_x || pos.y > hi_y)
                continue;

            Vec2  q  = sub2(pos, p0);
            Vec2  e1 = sub2(p1,  p0);
            Vec2  e2 = sub2(p2,  p0);
            float dist;
            Vec3  coords;
            tri_closest(q, e1, e2, &dist, &coords);

            /*
             * Fix: ternary per component → `selp.f32` in PTX.
             * Unconditionally reads coords.{x,y,z} and closest_coords.{x,y,z}
             * before selecting — no predicated block.
             */
            int is_closer   = (dist <= closest_dist) ? 1 : 0;
            closest_dist    = is_closer ? dist      : closest_dist;
            closest_cell    = is_closer ? cell      : closest_cell;
            closest_coords.x = is_closer ? coords.x : closest_coords.x;
            closest_coords.y = is_closer ? coords.y : closest_coords.y;
            closest_coords.z = is_closer ? coords.z : closest_coords.z;
        }

        if (pad >= 1.0e6f) break;
        pad = fminf(4.0f * pad, 1.0e6f);
    }

    out_cell[i]   = closest_cell;
    out_coords[i] = closest_coords;
}

/* ── Mesh generation ──────────────────────────────────────────────────────── */
/*
 * Mirrors _gen_trimesh(N=3, N=3) from test_fem.py and grid_to_tris(3,3)
 * from warp/_src/fem/utils.py. Triangle index ordering is identical.
 *
 * Vertices: positions[i*(N+1)+j] = (i/N, j/N)
 * Triangles: for each grid cell (cx,cy), two triangles:
 *   lower = [cx*(N+1)+cy,  (cx+1)*(N+1)+cy,  (cx+1)*(N+1)+(cy+1)]
 *   upper = [cx*(N+1)+cy,  (cx+1)*(N+1)+(cy+1),  cx*(N+1)+(cy+1)]
 * Outer loop: cy (matches grid_to_tris transpose ordering)
 */
#define N    3
#define NV   ((N+1)*(N+1))   /* 16 */
#define NTRI (2*N*N)          /* 18 */

static void gen_trimesh(Vec2 pos[NV], int3 tri[NTRI])
{
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= N; j++)
            pos[i*(N+1)+j] = v2((float)i/N, (float)j/N);

    int t = 0;
    for (int cy = 0; cy < N; cy++) {           /* cy outer matches Python */
        for (int cx = 0; cx < N; cx++) {
            int a = (N+1)*cx + cy;
            int b = (N+1)*(cx+1) + cy;
            int c = (N+1)*(cx+1) + (cy+1);
            int d = (N+1)*cx + (cy+1);
            tri[t].x = a; tri[t].y = b; tri[t].z = c; t++;  /* lower */
            tri[t].x = a; tri[t].y = c; tri[t].z = d; t++;  /* upper */
        }
    }
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */

#define CUDA_CHECK(x) do {                                                    \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                cudaGetErrorString(_e), __FILE__, __LINE__);                  \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static Vec2 reconstruct(const Vec2 *pos, const int3 *tri, int cell, Vec3 coords)
{
    if (cell < 0) return v2(-1e9f, -1e9f);
    Vec2 p0 = pos[tri[cell].x];
    Vec2 p1 = pos[tri[cell].y];
    Vec2 p2 = pos[tri[cell].z];
    return v2(coords.x*p0.x + coords.y*p1.x + coords.z*p2.x,
              coords.x*p0.y + coords.y*p1.y + coords.z*p2.y);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* Build mesh on host */
    Vec2 h_pos[NV];
    int3 h_tri[NTRI];
    gen_trimesh(h_pos, h_tri);

    /* Query = centroid of each triangle */
    Vec2 h_query[NTRI];
    for (int t = 0; t < NTRI; t++) {
        Vec2 p0 = h_pos[h_tri[t].x];
        Vec2 p1 = h_pos[h_tri[t].y];
        Vec2 p2 = h_pos[h_tri[t].z];
        h_query[t] = v2((p0.x+p1.x+p2.x)/3.0f, (p0.y+p1.y+p2.y)/3.0f);
    }

    /* Allocate device memory */
    Vec2 *d_pos, *d_query;
    int3 *d_tri;
    int  *d_cell_b,   *d_cell_f;
    Vec3 *d_coords_b, *d_coords_f;

    CUDA_CHECK(cudaMalloc(&d_pos,      NV   * sizeof(Vec2)));
    CUDA_CHECK(cudaMalloc(&d_tri,      NTRI * sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_query,    NTRI * sizeof(Vec2)));
    CUDA_CHECK(cudaMalloc(&d_cell_b,   NTRI * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_f,   NTRI * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coords_b, NTRI * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_coords_f, NTRI * sizeof(Vec3)));

    CUDA_CHECK(cudaMemcpy(d_pos,   h_pos,   NV   * sizeof(Vec2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri,   h_tri,   NTRI * sizeof(int3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, NTRI * sizeof(Vec2), cudaMemcpyHostToDevice));

    /* Launch: one thread per query, one block */
    cell_lookup_buggy<<<1, NTRI>>>(d_pos, d_tri, NTRI, d_query, d_cell_b, d_coords_b, NTRI);
    cell_lookup_fixed<<<1, NTRI>>>(d_pos, d_tri, NTRI, d_query, d_cell_f, d_coords_f, NTRI);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Download */
    int  h_cell_b[NTRI],   h_cell_f[NTRI];
    Vec3 h_coords_b[NTRI], h_coords_f[NTRI];
    CUDA_CHECK(cudaMemcpy(h_cell_b,   d_cell_b,   NTRI*sizeof(int),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_f,   d_cell_f,   NTRI*sizeof(int),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_coords_b, d_coords_b, NTRI*sizeof(Vec3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_coords_f, d_coords_f, NTRI*sizeof(Vec3), cudaMemcpyDeviceToHost));

    /* Verify */
    float tol = 0.001f;
    int buggy_fails = 0, fixed_fails = 0;

    printf("%-6s  %-14s  %-16s  %-16s\n",
           "tri", "query(x,y)", "buggy result", "fixed result");
    printf("%s\n", "-----------------------------------------------------------");

    for (int t = 0; t < NTRI; t++) {
        Vec2 q  = h_query[t];
        Vec2 rb = reconstruct(h_pos, h_tri, h_cell_b[t], h_coords_b[t]);
        Vec2 rf = reconstruct(h_pos, h_tri, h_cell_f[t], h_coords_f[t]);

        int ok_b = fabsf(rb.x - q.x) <= tol && fabsf(rb.y - q.y) <= tol;
        int ok_f = fabsf(rf.x - q.x) <= tol && fabsf(rf.y - q.y) <= tol;
        if (!ok_b) buggy_fails++;
        if (!ok_f) fixed_fails++;

        printf("tri[%2d]  (%.3f,%.3f)  (%.3f,%.3f)[%s]  (%.3f,%.3f)[%s]\n",
               t, q.x, q.y,
               rb.x, rb.y, ok_b ? "OK  " : "FAIL",
               rf.x, rf.y, ok_f ? "OK  " : "FAIL");
    }

    printf("%s\n", "-----------------------------------------------------------");
    printf("buggy: %d/%d FAIL  |  fixed: %d/%d FAIL\n",
           buggy_fails, NTRI, fixed_fails, NTRI);

    if (buggy_fails > 0 && fixed_fails == 0)
        printf("=> MetaX predicated Vec3 write bug confirmed.\n");
    else if (buggy_fails == 0)
        printf("=> No bug detected (NVIDIA or patched compiler).\n");
    else
        printf("=> UNEXPECTED: fixed kernel also fails.\n");

    cudaFree(d_pos); cudaFree(d_tri);   cudaFree(d_query);
    cudaFree(d_cell_b);   cudaFree(d_cell_f);
    cudaFree(d_coords_b); cudaFree(d_coords_f);
    return buggy_fails > 0 ? 1 : 0;
}
