"""
Minimal standalone reproduction of test_cell_lookup from warp/tests/test_fem.py.

Stripped of all warp.fem abstractions. Uses warp core (@wp.func / @wp.kernel)
directly so the kernel goes through the exact same NVRTC compilation path as
the original failing test.

Source mapping:
  project_on_tri_at_origin_2d ← warp/_src/fem/geometry/closest_point.py
  cell_lookup_buggy kernel     ← warp/_src/fem/geometry/geometry.py
                                  make_filtered_cell_lookup (before fix)
  cell_lookup_fixed kernel     ← geometry.py after wp.where fix
  gen_trimesh_np               ← warp/tests/test_fem.py _gen_trimesh(3,3)
                                  + warp/_src/fem/utils.py grid_to_tris(3,3)

Usage:
  python run.py

Expected output on MetaX C500 (before geometry.py fix):
  buggy: N/18 fail  |  fixed: 0/18 fail
"""

import numpy as np
import warp as wp

# ─── Geometry functions ───────────────────────────────────────────────────────
# Direct translation of closest_point.py into @wp.func (2D variant).


@wp.func
def seg_closest(q: wp.vec2, seg: wp.vec2, len_sq: float):
    """project_on_seg_at_origin for 2D vectors."""
    t = wp.clamp(wp.dot(q, seg) / len_sq, 0.0, 1.0)
    diff = q - t * seg
    return wp.dot(diff, diff), t


@wp.func
def tri_closest(q: wp.vec2, e1: wp.vec2, e2: wp.vec2):
    """project_on_tri_at_origin from closest_point.py, 2D embedding.

    Returns (sq_distance, barycentric_coords) where barycentric_coords
    is wp.vec3 = Coords(w0, w1, w2) as in the FEM implementation.
    """
    e1e1 = wp.dot(e1, e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.dot(e2, e2)
    det = e1e1 * e2e2 - e1e2 * e1e2

    if det > e1e1 * e2e2 * 1.0e-6:
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)
        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det
        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            diff = q - s * e1 - t * e2
            return wp.dot(diff, diff), wp.vec3(1.0 - s - t, s, t)

    d1, t1 = seg_closest(q, e1, e1e1)
    d2, t2 = seg_closest(q, e2, e2e2)
    e12 = e2 - e1
    d12, t12 = seg_closest(q - e1, e12, wp.dot(e12, e12))

    if d1 <= d2:
        if d1 <= d12:
            return d1, wp.vec3(1.0 - t1, t1, 0.0)
    elif d2 <= d12:
        return d2, wp.vec3(1.0 - t2, 0.0, t2)

    return d12, wp.vec3(0.0, 1.0 - t12, t12)


# ─── Buggy kernel ─────────────────────────────────────────────────────────────
# Mirrors make_filtered_cell_lookup in geometry.py BEFORE the wp.where fix.
# The `if dist <= closest_dist: closest_coords = coords` block causes a
# predicated write to a wp.vec3 on MetaX. The MetaX NVRTC compiler incorrectly
# handles this: some float components may not be written, leaving closest_coords
# as its sentinel value (-1e8, -1e8, -1e8) instead of the true barycentric
# coordinates.


@wp.kernel
def cell_lookup_buggy(
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.vec3i),
    n_tris: int,
    query_pos: wp.array(dtype=wp.vec2),
    out_cell: wp.array(dtype=int),
    out_coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pos = query_pos[i]

    closest_cell = int(-1)
    closest_coords = wp.vec3(-1.0e8, -1.0e8, -1.0e8)  # OUTSIDE sentinel

    pad = float(1.0e-5)

    # Outer loop: expand AABB pad until at least one triangle is found.
    # Mirrors: while closest_cell == NULL_ELEMENT_INDEX in geometry.py
    while closest_cell == -1:
        closest_dist = pad * pad

        # Inner loop: test each triangle candidate.
        # In the original: while wp.bvh_query_next(query, cell_index).
        # Here: brute-force with cheap AABB pre-filter (same structure).
        for cell in range(n_tris):
            vidx = tri_indices[cell]
            p0 = positions[vidx[0]]
            p1 = positions[vidx[1]]
            p2 = positions[vidx[2]]

            # AABB filter substituting BVH candidate enumeration.
            lo_x = wp.min(wp.min(p0[0], p1[0]), p2[0]) - pad
            lo_y = wp.min(wp.min(p0[1], p1[1]), p2[1]) - pad
            hi_x = wp.max(wp.max(p0[0], p1[0]), p2[0]) + pad
            hi_y = wp.max(wp.max(p0[1], p1[1]), p2[1]) + pad

            if pos[0] >= lo_x and pos[1] >= lo_y and pos[0] <= hi_x and pos[1] <= hi_y:
                q = pos - p0
                e1 = p1 - p0
                e2 = p2 - p0
                dist, coords = tri_closest(q, e1, e2)

                # ← Bug target: MetaX compiler incorrectly handles predicated
                # write to closest_coords (wp.vec3) inside this if block.
                # Symptom: closest_coords remains (-1e8, -1e8, -1e8) even
                # when closest_cell is correctly updated.
                if dist <= closest_dist:
                    closest_dist = dist
                    closest_cell = cell
                    closest_coords = coords

        if pad >= 1.0e6:
            break
        pad = wp.min(4.0 * pad, 1.0e6)

    out_cell[i] = closest_cell
    out_coords[i] = closest_coords


# ─── Fixed kernel ─────────────────────────────────────────────────────────────
# Mirrors geometry.py AFTER the wp.where fix.
# wp.where compiles to PTX `selp` (select-with-predicate) — an unconditional
# instruction that reads both operands before selecting. This eliminates the
# predicated block entirely, avoiding the MetaX compiler bug.


@wp.kernel
def cell_lookup_fixed(
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.vec3i),
    n_tris: int,
    query_pos: wp.array(dtype=wp.vec2),
    out_cell: wp.array(dtype=int),
    out_coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pos = query_pos[i]

    closest_cell = int(-1)
    closest_coords = wp.vec3(-1.0e8, -1.0e8, -1.0e8)

    pad = float(1.0e-5)

    while closest_cell == -1:
        closest_dist = pad * pad

        for cell in range(n_tris):
            vidx = tri_indices[cell]
            p0 = positions[vidx[0]]
            p1 = positions[vidx[1]]
            p2 = positions[vidx[2]]

            lo_x = wp.min(wp.min(p0[0], p1[0]), p2[0]) - pad
            lo_y = wp.min(wp.min(p0[1], p1[1]), p2[1]) - pad
            hi_x = wp.max(wp.max(p0[0], p1[0]), p2[0]) + pad
            hi_y = wp.max(wp.max(p0[1], p1[1]), p2[1]) + pad

            if pos[0] >= lo_x and pos[1] >= lo_y and pos[0] <= hi_x and pos[1] <= hi_y:
                q = pos - p0
                e1 = p1 - p0
                e2 = p2 - p0
                dist, coords = tri_closest(q, e1, e2)

                # Fix: wp.where per component → each compiles to `selp`.
                # Reads coords[0/1/2] and closest_coords[0/1/2] unconditionally
                # before the select — no predicated block emitted.
                is_closer = dist <= closest_dist
                closest_dist = wp.where(is_closer, dist, closest_dist)
                closest_cell = wp.where(is_closer, cell, closest_cell)
                closest_coords = wp.vec3(
                    wp.where(is_closer, coords[0], closest_coords[0]),
                    wp.where(is_closer, coords[1], closest_coords[1]),
                    wp.where(is_closer, coords[2], closest_coords[2]),
                )

        if pad >= 1.0e6:
            break
        pad = wp.min(4.0 * pad, 1.0e6)

    out_cell[i] = closest_cell
    out_coords[i] = closest_coords


# ─── Mesh generation ──────────────────────────────────────────────────────────
# Exact translation of _gen_trimesh(N, N) + grid_to_tris(N, N) from test_fem.py
# and fem/utils.py. Triangle index ordering is identical.


def gen_trimesh_np(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    # positions[i*(N+1)+j] = (x[i], y[j])
    positions = np.transpose(
        np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)
    ).reshape(-1, 2).astype(np.float32)

    # grid_to_tris(N, N): for each grid cell (cx,cy) two triangles.
    # Transposed meshgrid → cy outer, cx inner in flattened output.
    cx, cy = np.meshgrid(np.arange(N, dtype=np.int32), np.arange(N, dtype=np.int32), indexing="ij")
    tris = np.transpose(
        np.array([
            (N + 1) * cx + cy,
            (N + 1) * (cx + 1) + cy,
            (N + 1) * (cx + 1) + (cy + 1),
            (N + 1) * cx + cy,
            (N + 1) * (cx + 1) + (cy + 1),
            (N + 1) * cx + (cy + 1),
        ])
    ).reshape(-1, 3).astype(np.int32)

    return positions, tris


# ─── Verification helper ──────────────────────────────────────────────────────


def reconstruct(pos_np, tri_np, cell, coords):
    """Barycentric reconstruction: w0*p0 + w1*p1 + w2*p2."""
    if cell < 0:
        return np.array([-1e9, -1e9], dtype=np.float32)
    v = tri_np[cell]
    return coords[0] * pos_np[v[0]] + coords[1] * pos_np[v[1]] + coords[2] * pos_np[v[2]]


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    device = "cuda:0"
    wp.init()

    N = 3
    pos_np, tri_np = gen_trimesh_np(N)
    n_tris = tri_np.shape[0]  # 18

    # Query point for each triangle = centroid
    query_np = np.array(
        [(pos_np[tri_np[t, 0]] + pos_np[tri_np[t, 1]] + pos_np[tri_np[t, 2]]) / 3.0
         for t in range(n_tris)],
        dtype=np.float32,
    )

    with wp.ScopedDevice(device):
        positions   = wp.array(pos_np,  dtype=wp.vec2)
        tri_indices = wp.array(tri_np,  dtype=wp.vec3i)
        query_pos   = wp.array(query_np, dtype=wp.vec2)

        out_cell_b   = wp.zeros(n_tris, dtype=int)
        out_coords_b = wp.zeros(n_tris, dtype=wp.vec3)
        out_cell_f   = wp.zeros(n_tris, dtype=int)
        out_coords_f = wp.zeros(n_tris, dtype=wp.vec3)

        wp.launch(cell_lookup_buggy, dim=n_tris,
                  inputs=[positions, tri_indices, n_tris, query_pos,
                          out_cell_b, out_coords_b])
        wp.launch(cell_lookup_fixed, dim=n_tris,
                  inputs=[positions, tri_indices, n_tris, query_pos,
                          out_cell_f, out_coords_f])
        wp.synchronize()

        cell_b   = out_cell_b.numpy()
        coords_b = out_coords_b.numpy()
        cell_f   = out_cell_f.numpy()
        coords_f = out_coords_f.numpy()

    tol = 0.001
    buggy_fails = fixed_fails = 0

    print(f"{'tri':>4}  {'query (x,y)':>14}  "
          f"{'buggy result':>14}  "
          f"{'fixed result':>14}")
    print("-" * 62)

    for t in range(n_tris):
        q  = query_np[t]
        rb = reconstruct(pos_np, tri_np, cell_b[t], coords_b[t])
        rf = reconstruct(pos_np, tri_np, cell_f[t], coords_f[t])

        ok_b = abs(rb[0] - q[0]) <= tol and abs(rb[1] - q[1]) <= tol
        ok_f = abs(rf[0] - q[0]) <= tol and abs(rf[1] - q[1]) <= tol
        if not ok_b: buggy_fails += 1
        if not ok_f: fixed_fails += 1

        tag_b = "OK  " if ok_b else "FAIL"
        tag_f = "OK  " if ok_f else "FAIL"
        print(f"tri[{t:2d}]  ({q[0]:.3f},{q[1]:.3f})  "
              f"({rb[0]:.3f},{rb[1]:.3f})[{tag_b}]  "
              f"({rf[0]:.3f},{rf[1]:.3f})[{tag_f}]")

    print("-" * 62)
    print(f"buggy: {buggy_fails}/{n_tris} FAIL  |  fixed: {fixed_fails}/{n_tris} FAIL")

    if buggy_fails > 0 and fixed_fails == 0:
        print("=> MetaX predicated vec3 write bug confirmed.")
    elif buggy_fails == 0:
        print("=> No bug detected on this platform (may be NVIDIA or patched).")
    else:
        print("=> UNEXPECTED: fixed kernel also fails — investigate.")


if __name__ == "__main__":
    main()
