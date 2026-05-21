/*
 * Standalone NVRTC reproduction of the MetaX predicated vec3 write bug.
 *
 * Compile (offline with nvcc — offline compiler has no bug):
 *   nvcc -O2 repro_nvrtc.cu -lcuda -lnvrtc -o repro_nvrtc
 *
 * Run:
 *   ./repro_nvrtc
 *   ./repro_nvrtc --dump-bc   # also write cell_lookup.bc (NVIDIA only)
 *
 * How it works:
 *   The buggy and fixed device kernels live in the KERNEL_SRC string below.
 *   They are JIT-compiled at runtime via the NVRTC C API — the exact same
 *   compiler path that Warp uses — which is where MetaX has the bug.
 *   The offline nvcc compilation of this host file does NOT trigger the bug.
 *   All BVH types and traversal code are inlined in KERNEL_SRC; no external
 *   headers are needed at runtime.
 *
 * Dependencies: CUDA SDK only (cuda.h, nvrtc.h, cuda_runtime.h).
 *   No Warp Python runtime needed. No warp/native headers needed.
 *
 * Bug explanation:
 *   bvh_query_t (BVH traversal state) contains a shared-memory stack pointer,
 *   traversal counters, and 2×vec3 query bounds — many live registers.
 *   Under that register pressure, NVRTC code-sinks the closest_coords read
 *   into the `if` block, producing @p mov.f32 / @p st.f32 (predicated writes).
 *   MetaX NVRTC incorrectly handles predicated float writes: the update to
 *   closest_coords is silently discarded even when the predicate is TRUE.
 *   Fix: replace the if block with ternary ? : per component → PTX selp.f32
 *   (unconditional select), which avoids any predicated block.
 */

/* ── Standard includes ──────────────────────────────────────────────────────── */
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

/* ── Error checking ─────────────────────────────────────────────────────────── */

#define CUDA_CHECK(x) do {                                                      \
    cudaError_t _e = (x);                                                       \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
                cudaGetErrorString(_e), __FILE__, __LINE__);                    \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define CU_CHECK(x) do {                                                        \
    CUresult _r = (x);                                                          \
    if (_r != CUDA_SUCCESS) {                                                   \
        const char* _s = "unknown";                                             \
        cuGetErrorString(_r, &_s);                                              \
        fprintf(stderr, "CU driver error %s at %s:%d\n",                       \
                _s, __FILE__, __LINE__);                                        \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define NVRTC_CHECK(x) do {                                                     \
    nvrtcResult _r = (x);                                                       \
    if (_r != NVRTC_SUCCESS) {                                                  \
        fprintf(stderr, "NVRTC error %s at %s:%d\n",                           \
                nvrtcGetErrorString(_r), __FILE__, __LINE__);                   \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* ── Host-side BVH struct definitions ──────────────────────────────────────── */
/*
 * Copied from warp/native/bvh.h — field order and types must match exactly
 * so that cudaMemcpy of this struct produces a valid wp::BVH on the device.
 *
 * wp::vec3 = vec_t<3,float> = struct { float c[3]; } — same layout as float3.
 * All pointer fields are 8 bytes on 64-bit; int fields are 4 bytes.
 * The struct layout (offsets) is identical to wp::BVH.
 */

struct HostPackedNode {          /* mirrors BVHPackedNodeHalf in bvh.h */
    float        x, y, z;
    unsigned int i : 31;         /* left-child index (internal) or start (leaf) */
    unsigned int b : 1;          /* leaf flag */
};

struct HostBVH {                 /* mirrors wp::BVH in bvh.h — 104 bytes on 64-bit */
    HostPackedNode* node_lowers;
    HostPackedNode* node_uppers;
    int*            node_parents;
    int*            node_counts;        /* nullptr for host-built BVH */
    int*            primitive_indices;
    int             max_depth;
    int             max_nodes;
    int             num_nodes;
    int             num_leaf_nodes;
    int*            root;
    float3*         item_lowers;        /* same 8-byte pointer as wp::vec3* */
    float3*         item_uppers;
    int*            item_groups;        /* nullptr */
    int             num_items;
    int             leaf_size;
    void*           context;            /* nullptr */
};

/* ── Minimal BVH builder ────────────────────────────────────────────────────── */
/*
 * Reimplements warp/native/bvh.cpp TopDownBVHBuilder with median split.
 * For 18 triangles this produces a small valid BVH tree in a few microseconds.
 */

static HostPackedNode make_node_h(float3 b, int child, bool leaf)
{
    HostPackedNode n;
    n.x = b.x; n.y = b.y; n.z = b.z;
    n.i = (unsigned int)child;
    n.b = leaf ? 1u : 0u;
    return n;
}

static void calc_bounds(const float3* lo, const float3* hi, const int* prim,
                        int s, int e, float3& out_lo, float3& out_hi)
{
    out_lo = make_float3( FLT_MAX,  FLT_MAX,  FLT_MAX);
    out_hi = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int k = s; k < e; ++k) {
        int p = prim[k];
        out_lo.x = fminf(out_lo.x, lo[p].x);
        out_lo.y = fminf(out_lo.y, lo[p].y);
        out_lo.z = fminf(out_lo.z, lo[p].z);
        out_hi.x = fmaxf(out_hi.x, hi[p].x);
        out_hi.y = fmaxf(out_hi.y, hi[p].y);
        out_hi.z = fmaxf(out_hi.z, hi[p].z);
    }
}

static int build_rec(HostBVH& bvh, const float3* lo, const float3* hi,
                     int s, int e, int depth, int par)
{
    int ni = bvh.num_nodes++;
    if (depth > bvh.max_depth) bvh.max_depth = depth;

    float3 blo, bhi;
    calc_bounds(lo, hi, bvh.primitive_indices, s, e, blo, bhi);

    int n = e - s;
    if (n <= bvh.leaf_size || depth >= 32) {   /* 32 = BVH_QUERY_STACK_SIZE */
        bvh.node_lowers[ni] = make_node_h(blo, s, true);
        bvh.node_uppers[ni] = make_node_h(bhi, e, false);
        bvh.num_leaf_nodes++;
    } else {
        /* Median split along longest axis */
        float ex = bhi.x - blo.x, ey = bhi.y - blo.y, ez = bhi.z - blo.z;
        int   ax = (ey > ex) ? 1 : 0;
        if (ez > (ax == 0 ? ex : ey)) ax = 2;
        float cen = ax == 0 ? 0.5f * (blo.x + bhi.x) :
                    ax == 1 ? 0.5f * (blo.y + bhi.y) : 0.5f * (blo.z + bhi.z);

        int mid = s;
        for (int k = s; k < e; ++k) {
            int   p = bvh.primitive_indices[k];
            float c = ax == 0 ? 0.5f * (lo[p].x + hi[p].x) :
                      ax == 1 ? 0.5f * (lo[p].y + hi[p].y) :
                                0.5f * (lo[p].z + hi[p].z);
            if (c < cen)
                std::swap(bvh.primitive_indices[mid++], bvh.primitive_indices[k]);
        }
        if (mid == s || mid == e) mid = s + n / 2;   /* degenerate fallback */

        int left  = build_rec(bvh, lo, hi, s,   mid, depth + 1, ni);
        int right = build_rec(bvh, lo, hi, mid, e,   depth + 1, ni);
        bvh.node_lowers[ni] = make_node_h(blo, left,  false);
        bvh.node_uppers[ni] = make_node_h(bhi, right, false);
    }
    bvh.node_parents[ni] = par;
    return ni;
}

static HostBVH build_bvh_cpu(float3* item_lo, float3* item_hi, int n)
{
    HostBVH bvh = {};
    bvh.num_items        = n;
    bvh.leaf_size        = 1;
    bvh.max_nodes        = 2 * n;
    bvh.node_lowers      = new HostPackedNode[bvh.max_nodes];
    bvh.node_uppers      = new HostPackedNode[bvh.max_nodes];
    bvh.node_parents     = new int[bvh.max_nodes];
    bvh.primitive_indices = new int[n];
    bvh.root             = new int[1];
    bvh.item_lowers      = item_lo;
    bvh.item_uppers      = item_hi;
    for (int i = 0; i < n; ++i) bvh.primitive_indices[i] = i;
    build_rec(bvh, item_lo, item_hi, 0, n, 0, -1);
    bvh.root[0] = 0;     /* top-down builder: root always at node index 0 */
    return bvh;
}

/*
 * Upload the host BVH to device memory.
 * Returns uint64_t id = device pointer to the BVH struct,
 * which is passed directly to bvh_query_aabb() inside the kernel.
 */
static uint64_t upload_bvh(HostBVH& h)
{
    HostPackedNode *d_nl, *d_nu;
    int            *d_par, *d_prim, *d_root;
    float3         *d_il, *d_iu;

    CUDA_CHECK(cudaMalloc(&d_nl,   h.num_nodes * sizeof(HostPackedNode)));
    CUDA_CHECK(cudaMalloc(&d_nu,   h.num_nodes * sizeof(HostPackedNode)));
    CUDA_CHECK(cudaMalloc(&d_par,  h.num_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prim, h.num_items  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_root, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_il,   h.num_items  * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_iu,   h.num_items  * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(d_nl,   h.node_lowers,       h.num_nodes * sizeof(HostPackedNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nu,   h.node_uppers,       h.num_nodes * sizeof(HostPackedNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_par,  h.node_parents,      h.num_nodes * sizeof(int),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prim, h.primitive_indices, h.num_items  * sizeof(int),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_root, h.root,              sizeof(int),                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_il,   h.item_lowers,       h.num_items  * sizeof(float3),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iu,   h.item_uppers,       h.num_items  * sizeof(float3),         cudaMemcpyHostToDevice));

    /* Build a HostBVH struct that holds device pointers (same layout as wp::BVH) */
    HostBVH d_bvh     = h;
    d_bvh.node_lowers      = d_nl;
    d_bvh.node_uppers      = d_nu;
    d_bvh.node_parents     = d_par;
    d_bvh.node_counts      = nullptr;
    d_bvh.primitive_indices = d_prim;
    d_bvh.root             = d_root;
    d_bvh.item_lowers      = d_il;
    d_bvh.item_uppers      = d_iu;
    d_bvh.item_groups      = nullptr;
    d_bvh.context          = nullptr;

    /* Copy the struct itself to device memory so bvh_get(id) can dereference it */
    HostBVH* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(HostBVH)));
    CUDA_CHECK(cudaMemcpy(d_ptr, &d_bvh, sizeof(HostBVH), cudaMemcpyHostToDevice));
    return (uint64_t)(uintptr_t)d_ptr;
}

/* ── Mesh generation ────────────────────────────────────────────────────────── */
/*
 * Mirrors _gen_trimesh(N=3) from test_fem.py and grid_to_tris(3,3) from utils.py.
 * Triangle index ordering is identical (outer loop cy, inner cx).
 */

#define N_GRID 3
#define N_VERTS ((N_GRID + 1) * (N_GRID + 1))   /* 16 */
#define N_TRIS  (2 * N_GRID * N_GRID)            /* 18 */

static void gen_trimesh(float2 pos[N_VERTS], int3 tri[N_TRIS])
{
    for (int i = 0; i <= N_GRID; ++i)
        for (int j = 0; j <= N_GRID; ++j)
            pos[i * (N_GRID + 1) + j] = make_float2((float)i / N_GRID,
                                                     (float)j / N_GRID);
    int t = 0;
    for (int cy = 0; cy < N_GRID; ++cy) {
        for (int cx = 0; cx < N_GRID; ++cx) {
            int a = (N_GRID + 1) * cx + cy;
            int b = (N_GRID + 1) * (cx + 1) + cy;
            int c = (N_GRID + 1) * (cx + 1) + (cy + 1);
            int d = (N_GRID + 1) * cx + (cy + 1);
            tri[t].x = a; tri[t].y = b; tri[t].z = c; ++t;  /* lower triangle */
            tri[t].x = a; tri[t].y = c; tri[t].z = d; ++t;  /* upper triangle */
        }
    }
}

static void build_aabb(const float2* pos, const int3* tri, int n,
                       float3* lo, float3* hi)
{
    for (int t = 0; t < n; ++t) {
        float2 p0 = pos[tri[t].x], p1 = pos[tri[t].y], p2 = pos[tri[t].z];
        lo[t] = make_float3(fminf(fminf(p0.x, p1.x), p2.x),
                            fminf(fminf(p0.y, p1.y), p2.y), 0.f);
        hi[t] = make_float3(fmaxf(fmaxf(p0.x, p1.x), p2.x),
                            fmaxf(fmaxf(p0.y, p1.y), p2.y), 0.f);
    }
}

/* ── Device kernel source ───────────────────────────────────────────────────── */
/*
 * Self-contained: all BVH types and traversal code are inlined here.
 * Extracted from warp/native/bvh.h + intersect.h.  No external headers needed.
 * Uses float2 / float3 / int3 (CUDA built-ins, same layout as wp::vec2/3/3i).
 *
 * BUG LOCATION (cell_lookup_buggy):
 *   After tri_closest_2d(), the block:
 *       if (dist <= closest_dist) { closest_coords = coords; }
 *   causes NVRTC to generate @p mov.f32 for each of the 3 float components.
 *   On MetaX the predicated write is incorrectly discarded even when TRUE.
 *
 * FIX (cell_lookup_fixed):
 *   Ternary ? : per component → PTX selp.f32 (unconditional select, no bug).
 */
static const char KERNEL_SRC[] = R"KERNEL(
/* ══ BVH types and traversal — inlined from warp/native/bvh.h ═══════════════
   Layout of BVH / BVHPackedNodeHalf must match warp/native/bvh.h exactly so
   that the device pointer uploaded by the host (HostBVH struct) is valid.    */

#define BVH_QUERY_STACK_SIZE 32
#define WP_TILE_BLOCK_DIM    256

struct BVHPackedNodeHalf {
    float        x, y, z;
    unsigned int i : 31;   /* left-child index (internal) or range start (leaf) */
    unsigned int b : 1;    /* 1 = leaf node */
};

/* Same field order/types as wp::BVH in bvh.h (104 bytes on 64-bit) */
struct BVH {
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;
    int*               node_parents;
    int*               node_counts;
    int*               primitive_indices;
    int max_depth, max_nodes, num_nodes, num_leaf_nodes;
    int*    root;
    float3* item_lowers;    /* wp::vec3 has identical memory layout to float3 */
    float3* item_uppers;
    int*    item_groups;
    int     num_items, leaf_size;
    void*   context;
};

/* Strided shared-memory stack (one slot per depth per thread) */
struct bvh_stack_t {
    __device__ inline int  operator[](int d) const { return ptr[d * WP_TILE_BLOCK_DIM]; }
    __device__ inline int& operator[](int d)       { return ptr[d * WP_TILE_BLOCK_DIM]; }
    int* ptr;
};

/* Per-thread traversal state — large struct = high register pressure */
struct bvh_query_t {
    __device__ bvh_query_t()
        : count(0), primitive_counter(0), bounds_nr(-1), is_ray(false)
    { input_lower = make_float3(0,0,0); input_upper = make_float3(0,0,0); }
    __device__ bvh_query_t& operator+=(const bvh_query_t&) { return *this; }

    BVH         bvh;         /* copy of BVH struct: many pointer+int fields */
    bvh_stack_t stack;       /* ptr into shared memory */
    int         count;
    int         primitive_counter;
    float3      input_lower;
    float3      input_upper;
    int         bounds_nr;
    bool        is_ray;
};

__device__ inline BVH bvh_get(unsigned long long id)
{
    return *(BVH*)(id);
}

/* Texture-cache load: BVHPackedNodeHalf is 16 bytes = float4 */
__device__ inline BVHPackedNodeHalf bvh_load_node(const BVHPackedNodeHalf* p, int i)
{
    float4 f = __ldg((const float4*)p + i);
    return (const BVHPackedNodeHalf&)f;
}

__device__ inline bool aabb_overlap(float3 lo_a, float3 hi_a,
                                     float3 lo_b, float3 hi_b)
{
    return !(lo_a.x > hi_b.x || hi_a.x < lo_b.x ||
             lo_a.y > hi_b.y || hi_a.y < lo_b.y ||
             lo_a.z > hi_b.z || hi_a.z < lo_b.z);
}

__device__ inline bvh_query_t bvh_query_aabb(unsigned long long id,
                                              float3 lower, float3 upper)
{
    bvh_query_t q;
    /* Static shared memory for the traversal stack.
       Sized for BVH_QUERY_STACK_SIZE × WP_TILE_BLOCK_DIM = 32 KB.
       Each thread uses its own strided slice via threadIdx.x.               */
    __shared__ int _stack[BVH_QUERY_STACK_SIZE * WP_TILE_BLOCK_DIM];
    q.stack.ptr       = &_stack[threadIdx.x];
    q.bvh             = bvh_get(id);
    q.input_lower     = lower;
    q.input_upper     = upper;
    q.is_ray          = false;
    q.stack[0]        = *q.bvh.root;
    q.count           = 1;
    q.primitive_counter = 0;
    q.bounds_nr       = -1;
    return q;
}

__device__ inline bool bvh_query_next(bvh_query_t& q, int& index)
{
    BVH bvh = q.bvh;
    while (q.count) {
        int ni = q.stack[--q.count];
        BVHPackedNodeHalf nl = bvh_load_node(bvh.node_lowers, ni);
        BVHPackedNodeHalf nu = bvh_load_node(bvh.node_uppers, ni);

        if (q.primitive_counter == 0) {   /* node-level AABB test */
            if (!aabb_overlap(q.input_lower, q.input_upper,
                              make_float3(nl.x, nl.y, nl.z),
                              make_float3(nu.x, nu.y, nu.z)))
                continue;
        }

        if (nl.b) {                              /* leaf */
            int start = (int)nl.i, end = (int)nu.i;
            if (end - start == 1) {              /* fast path: single prim */
                index = bvh.primitive_indices[start];
                q.bounds_nr = index;
                q.primitive_counter = 0;
                return true;
            }
            /* multi-prim leaf: one primitive per call */
            int prim = bvh.primitive_indices[start + q.primitive_counter++];
            if (start + q.primitive_counter == end)
                q.primitive_counter = 0;
            else
                q.stack[q.count++] = ni;
            if (!aabb_overlap(q.input_lower, q.input_upper,
                              bvh.item_lowers[prim], bvh.item_uppers[prim]))
                continue;
            index = prim;
            q.bounds_nr = prim;
            return true;
        } else {                                 /* internal node */
            q.primitive_counter = 0;
            q.stack[q.count++] = (int)nl.i;     /* left child  */
            q.stack[q.count++] = (int)nu.i;     /* right child */
        }
    }
    return false;
}

/* ══ 2D geometry ═════════════════════════════════════════════════════════════
   Direct translation of project_on_tri_at_origin from closest_point.py      */

__device__ static float2 seg_closest_2d(float2 q, float2 seg, float len_sq,
                                         float& out_t)
{
    float t = fmaxf(0.f, fminf(1.f, (q.x*seg.x + q.y*seg.y) / len_sq));
    out_t = t;
    return make_float2(q.x - t*seg.x, q.y - t*seg.y);
}

__device__ static void tri_closest_2d(float2 q, float2 e1, float2 e2,
                                       float& out_dist, float3& out_coords)
{
    float e1e1 = e1.x*e1.x + e1.y*e1.y;
    float e1e2 = e1.x*e2.x + e1.y*e2.y;
    float e2e2 = e2.x*e2.x + e2.y*e2.y;
    float det  = e1e1*e2e2 - e1e2*e1e2;

    if (det > e1e1*e2e2*1.e-6f) {
        float e1p = e1.x*q.x + e1.y*q.y;
        float e2p = e2.x*q.x + e2.y*q.y;
        float s   = (e2e2*e1p - e1e2*e2p) / det;
        float t   = (e1e1*e2p - e1e2*e1p) / det;
        if (s >= 0.f && t >= 0.f && s+t <= 1.f) {
            float dx = q.x - s*e1.x - t*e2.x;
            float dy = q.y - s*e1.y - t*e2.y;
            out_dist   = dx*dx + dy*dy;
            out_coords = make_float3(1.f-s-t, s, t);
            return;
        }
    }

    float t1, t2, t12;
    float2 r1  = seg_closest_2d(q,  e1, e1e1, t1);
    float2 r2  = seg_closest_2d(q,  e2, e2e2, t2);
    float2 e12 = make_float2(e2.x-e1.x, e2.y-e1.y);
    float  e12e12 = e12.x*e12.x + e12.y*e12.y;
    float2 r12 = seg_closest_2d(make_float2(q.x-e1.x, q.y-e1.y),
                                 e12, e12e12, t12);

    float d1  = r1.x*r1.x  + r1.y*r1.y;
    float d2  = r2.x*r2.x  + r2.y*r2.y;
    float d12 = r12.x*r12.x + r12.y*r12.y;

    if (d1 <= d2) {
        if (d1 <= d12) { out_dist = d1;  out_coords = make_float3(1.f-t1, t1, 0.f); return; }
    } else if (d2 <= d12) {
        out_dist = d2;  out_coords = make_float3(1.f-t2, 0.f, t2); return;
    }
    out_dist = d12; out_coords = make_float3(0.f, 1.f-t12, t12);
}

/* ══ Buggy kernel ════════════════════════════════════════════════════════════
   if (dist <= closest_dist) { closest_coords = coords; }
   → NVRTC code-sinks closest_coords read into the if block (register pressure
     from bvh_query_t), then emits @p mov.f32 for each component.
   → MetaX incorrectly discards the write even when predicate is TRUE.        */

extern "C" __global__ void cell_lookup_buggy(
    unsigned long long bvh_id,
    const float2*      positions,
    const int3*        tri_indices,
    const float2*      query_pos,
    int*               out_cell,
    float3*            out_coords,
    int                n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 pos          = query_pos[i];
    int    closest_cell = -1;
    float3 closest_coords = make_float3(-1.e8f, -1.e8f, -1.e8f);
    float  pad          = 1.e-5f;

    while (closest_cell == -1) {
        float closest_dist = pad * pad;

        bvh_query_t query = bvh_query_aabb(bvh_id,
            make_float3(pos.x - pad, pos.y - pad, -pad),
            make_float3(pos.x + pad, pos.y + pad,  pad));
        int cell_index = 0;

        while (bvh_query_next(query, cell_index)) {
            int3   vidx = tri_indices[cell_index];
            float2 p0   = positions[vidx.x];
            float2 p1   = positions[vidx.y];
            float2 p2   = positions[vidx.z];

            float2 q  = make_float2(pos.x - p0.x, pos.y - p0.y);
            float2 e1 = make_float2(p1.x  - p0.x, p1.y  - p0.y);
            float2 e2 = make_float2(p2.x  - p0.x, p2.y  - p0.y);

            float  dist; float3 coords;
            tri_closest_2d(q, e1, e2, dist, coords);

            /* BUG: @p mov.f32 per component; MetaX discards when predicate TRUE */
            if (dist <= closest_dist) {
                closest_dist   = dist;
                closest_cell   = cell_index;
                closest_coords = coords;
            }
        }

        if (pad >= 1.e6f) break;
        pad = fminf(4.f * pad, 1.e6f);
    }

    out_cell[i]   = closest_cell;
    out_coords[i] = closest_coords;
}

/* ══ Fixed kernel ═════════════════════════════════════════════════════════════
   Ternary ? : per component → PTX selp.f32 (unconditional select, no bug).   */

extern "C" __global__ void cell_lookup_fixed(
    unsigned long long bvh_id,
    const float2*      positions,
    const int3*        tri_indices,
    const float2*      query_pos,
    int*               out_cell,
    float3*            out_coords,
    int                n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float2 pos          = query_pos[i];
    int    closest_cell = -1;
    float3 closest_coords = make_float3(-1.e8f, -1.e8f, -1.e8f);
    float  pad          = 1.e-5f;

    while (closest_cell == -1) {
        float closest_dist = pad * pad;

        bvh_query_t query = bvh_query_aabb(bvh_id,
            make_float3(pos.x - pad, pos.y - pad, -pad),
            make_float3(pos.x + pad, pos.y + pad,  pad));
        int cell_index = 0;

        while (bvh_query_next(query, cell_index)) {
            int3   vidx = tri_indices[cell_index];
            float2 p0   = positions[vidx.x];
            float2 p1   = positions[vidx.y];
            float2 p2   = positions[vidx.z];

            float2 q  = make_float2(pos.x - p0.x, pos.y - p0.y);
            float2 e1 = make_float2(p1.x  - p0.x, p1.y  - p0.y);
            float2 e2 = make_float2(p2.x  - p0.x, p2.y  - p0.y);

            float  dist; float3 coords;
            tri_closest_2d(q, e1, e2, dist, coords);

            /* FIX: selp.f32 per component — unconditional, no predicated block */
            int closer     = (dist <= closest_dist);
            closest_dist   = closer ? dist       : closest_dist;
            closest_cell   = closer ? cell_index : closest_cell;
            closest_coords = make_float3(
                closer ? coords.x : closest_coords.x,
                closer ? coords.y : closest_coords.y,
                closer ? coords.z : closest_coords.z);
        }

        if (pad >= 1.e6f) break;
        pad = fminf(4.f * pad, 1.e6f);
    }

    out_cell[i]   = closest_cell;
    out_coords[i] = closest_coords;
}
)KERNEL";

/* ── NVRTC compilation ──────────────────────────────────────────────────────── */

static std::vector<char> compile_kernels(bool dump_bc)
{
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, KERNEL_SRC, "cell_lookup_kernel.cu",
                                   0, nullptr, nullptr));

    const char* opts[] = { "-arch=compute_80" };
    nvrtcResult res = nvrtcCompileProgram(prog, 1, opts);
    if (res != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        std::vector<char> log(log_sz);
        nvrtcGetProgramLog(prog, log.data());
        fprintf(stderr, "NVRTC compile failed:\n%s\n", log.data());
        exit(1);
    }

    /* Dump .bc bitcode (NVIDIA only; skip gracefully on MetaX if unsupported) */
    if (dump_bc) {
        size_t bc_sz = 0;
        nvrtcResult br = nvrtcGetBitcodeSize(prog, &bc_sz);
        if (br == NVRTC_SUCCESS && bc_sz > 0) {
            std::vector<char> bc(bc_sz);
            NVRTC_CHECK(nvrtcGetBitcode(prog, bc.data()));
            FILE* f = fopen("cell_lookup.bc", "wb");
            if (f) { fwrite(bc.data(), 1, bc_sz, f); fclose(f); }
            printf("[--dump-bc] %zu bytes → cell_lookup.bc\n", bc_sz);
            printf("  llvm-dis cell_lookup.bc | grep -A 40 cell_lookup_buggy\n");
            printf("  Look for: @p mov.f32 (buggy) vs selp.f32 (fixed)\n");
        } else {
            printf("[--dump-bc] nvrtcGetBitcode not supported on this platform"
                   " — skipping.\n");
        }
    }

    size_t ptx_sz;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_sz));
    std::vector<char> ptx(ptx_sz);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);
    return ptx;
}

/* ── Verification helper ────────────────────────────────────────────────────── */

static float2 reconstruct(const float2* pos, const int3* tri, int cell,
                           const float3& coords)
{
    if (cell < 0) return make_float2(-1e9f, -1e9f);
    float2 p0 = pos[tri[cell].x], p1 = pos[tri[cell].y], p2 = pos[tri[cell].z];
    return make_float2(coords.x * p0.x + coords.y * p1.x + coords.z * p2.x,
                       coords.x * p0.y + coords.y * p1.y + coords.z * p2.y);
}

/* ── Main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char** argv)
{
    bool dump_bc = (argc >= 2 && strcmp(argv[1], "--dump-bc") == 0);

    /* ── Build mesh (CPU) ─────────────────────────────────────────────────── */
    float2 h_pos[N_VERTS];
    int3   h_tri[N_TRIS];
    gen_trimesh(h_pos, h_tri);

    /* Query = centroid of each triangle */
    float2 h_query[N_TRIS];
    for (int t = 0; t < N_TRIS; ++t) {
        float2 p0 = h_pos[h_tri[t].x];
        float2 p1 = h_pos[h_tri[t].y];
        float2 p2 = h_pos[h_tri[t].z];
        h_query[t] = make_float2((p0.x + p1.x + p2.x) / 3.f,
                                 (p0.y + p1.y + p2.y) / 3.f);
    }

    /* Per-triangle 3-D AABBs (z extended to ±0 to form a thin slab) */
    float3 h_lo[N_TRIS], h_hi[N_TRIS];
    build_aabb(h_pos, h_tri, N_TRIS, h_lo, h_hi);

    /* ── Build BVH (CPU) then upload to device ────────────────────────────── */
    HostBVH h_bvh  = build_bvh_cpu(h_lo, h_hi, N_TRIS);
    uint64_t bvh_id = upload_bvh(h_bvh);
    printf("BVH built: %d nodes, %d leaf nodes.\n",
           h_bvh.num_nodes, h_bvh.num_leaf_nodes);

    /* ── Allocate device buffers ──────────────────────────────────────────── */
    float2 *d_pos, *d_query;
    int3   *d_tri;
    int    *d_cell_b, *d_cell_f;
    float3 *d_coords_b, *d_coords_f;

    CUDA_CHECK(cudaMalloc(&d_pos,      N_VERTS * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_tri,      N_TRIS  * sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_query,    N_TRIS  * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_cell_b,   N_TRIS  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_f,   N_TRIS  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coords_b, N_TRIS  * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_coords_f, N_TRIS  * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(d_pos,   h_pos,   N_VERTS * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri,   h_tri,   N_TRIS  * sizeof(int3),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, N_TRIS  * sizeof(float2), cudaMemcpyHostToDevice));

    /* ── NVRTC compile (triggers MetaX bug in the JIT path) ─────────────── */
    printf("Compiling kernels via NVRTC...\n");
    std::vector<char> ptx = compile_kernels(dump_bc);
    printf("NVRTC compilation OK (%zu bytes PTX).\n", ptx.size());

    /* ── Load PTX via CUDA driver API ────────────────────────────────────── */
    CU_CHECK(cuInit(0));

    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        CUdevice dev;
        CU_CHECK(cuDeviceGet(&dev, 0));
        CU_CHECK(cuCtxCreate(&ctx, 0, dev));
    }

    CUmodule   module;
    CUfunction fn_buggy, fn_fixed;
    CU_CHECK(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&fn_buggy, module, "cell_lookup_buggy"));
    CU_CHECK(cuModuleGetFunction(&fn_fixed,  module, "cell_lookup_fixed"));

    /* ── Launch kernels: 1 block × N_TRIS threads ────────────────────────── */
    /*
     * sharedMemBytes = 0: the BVH stack is declared as a static __shared__
     * array inside bvh_query() (BVH_QUERY_STACK_SIZE * WP_TILE_BLOCK_DIM ints
     * = 32 KB) — CUDA accounts for static shared memory automatically.
     */
    int n = N_TRIS;
    auto launch = [&](CUfunction fn, CUdeviceptr d_cell, CUdeviceptr d_coords) {
        void* args[] = { &bvh_id,
                         &d_pos, &d_tri, &d_query,
                         &d_cell, &d_coords, &n };
        CU_CHECK(cuLaunchKernel(fn,
                                1, 1, 1,   /* grid  */
                                n, 1, 1,   /* block */
                                0,         /* sharedMemBytes (static __shared__ handled automatically) */
                                0,         /* stream */
                                args, nullptr));
    };

    launch(fn_buggy, (CUdeviceptr)d_cell_b, (CUdeviceptr)d_coords_b);
    launch(fn_fixed,  (CUdeviceptr)d_cell_f, (CUdeviceptr)d_coords_f);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Download results ────────────────────────────────────────────────── */
    int    h_cell_b[N_TRIS], h_cell_f[N_TRIS];
    float3 h_cb[N_TRIS],     h_cf[N_TRIS];
    CUDA_CHECK(cudaMemcpy(h_cell_b, d_cell_b, N_TRIS*sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cell_f, d_cell_f, N_TRIS*sizeof(int),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cb, d_coords_b,   N_TRIS*sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cf, d_coords_f,   N_TRIS*sizeof(float3), cudaMemcpyDeviceToHost));

    /* ── Verify and report ───────────────────────────────────────────────── */
    const float TOL = 0.001f;
    int buggy_fails = 0, fixed_fails = 0;

    printf("\n%-6s  %-14s  %-16s  %-16s\n",
           "tri", "query(x,y)", "buggy result", "fixed result");
    printf("%.62s\n",
           "------------------------------------------------------------------");
    for (int t = 0; t < N_TRIS; ++t) {
        float2 q  = h_query[t];
        float2 rb = reconstruct(h_pos, h_tri, h_cell_b[t], h_cb[t]);
        float2 rf = reconstruct(h_pos, h_tri, h_cell_f[t], h_cf[t]);

        int ok_b = fabsf(rb.x - q.x) <= TOL && fabsf(rb.y - q.y) <= TOL;
        int ok_f = fabsf(rf.x - q.x) <= TOL && fabsf(rf.y - q.y) <= TOL;
        if (!ok_b) ++buggy_fails;
        if (!ok_f) ++fixed_fails;

        printf("tri[%2d]  (%.3f,%.3f)  (%.3f,%.3f)[%s]  (%.3f,%.3f)[%s]\n",
               t, q.x, q.y,
               rb.x, rb.y, ok_b ? "OK  " : "FAIL",
               rf.x, rf.y, ok_f ? "OK  " : "FAIL");
    }
    printf("%.62s\n",
           "------------------------------------------------------------------");
    printf("buggy: %d/%d FAIL  |  fixed: %d/%d FAIL\n",
           buggy_fails, N_TRIS, fixed_fails, N_TRIS);

    if (buggy_fails > 0 && fixed_fails == 0)
        printf("=> MetaX predicated vec3 write bug CONFIRMED.\n");
    else if (buggy_fails == 0)
        printf("=> No bug detected (NVIDIA GPU or patched compiler).\n");
    else
        printf("=> UNEXPECTED: fixed kernel also fails — check BVH construction.\n");

    CU_CHECK(cuModuleUnload(module));
    return (buggy_fails > 0 && fixed_fails == 0) ? 0 : 1;
}
