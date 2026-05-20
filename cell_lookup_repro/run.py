"""
Minimal standalone reproduction of test_cell_lookup from warp/tests/test_fem.py.

Stripped of all warp.fem abstractions. Uses warp core (@wp.func / @wp.kernel)
directly so the kernel goes through the exact same NVRTC compilation path as
the original failing test.

Source mapping:
  seg_closest / tri_closest    ← warp/_src/fem/geometry/closest_point.py
  cell_lookup_bvh_buggy        ← geometry.py make_filtered_cell_lookup (before fix)
  cell_lookup_bvh_fixed        ← geometry.py make_filtered_cell_lookup (after fix)
  gen_trimesh_np               ← test_fem.py _gen_trimesh(3,3)
                                  + warp/_src/fem/utils.py grid_to_tris(3,3)

Two kernels are provided:
  bvh_buggy: uses real wp.Bvh + wp.bvh_query_next, if-block for closest_coords
  bvh_fixed: same BVH structure, wp.where per component instead of if-block

Usage:
  python run.py
"""

import numpy as np
import warp as wp

# ─── Geometry functions ───────────────────────────────────────────────────────
# Direct translation of closest_point.py into @wp.func (2D variant).


@wp.func
def seg_closest(q: wp.vec2, seg: wp.vec2, len_sq: float):
    t = wp.clamp(wp.dot(q, seg) / len_sq, 0.0, 1.0)
    diff = q - t * seg
    return wp.dot(diff, diff), t


@wp.func
def tri_closest(q: wp.vec2, e1: wp.vec2, e2: wp.vec2):
    """project_on_tri_at_origin from closest_point.py, 2D embedding."""
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


# ─── BVH-based buggy kernel ───────────────────────────────────────────────────
# Faithful reproduction of make_filtered_cell_lookup in geometry.py BEFORE fix.
#
# Uses real wp.bvh_query_aabb + wp.bvh_query_next which compiles to the full
# bvh_query_t struct traversal (shared memory stack, BVH node pointers, etc.).
# This is what creates the register pressure that triggers the NVRTC optimizer
# to do code sinking, causing the predicated vec3 write bug on MetaX.


@wp.kernel
def cell_lookup_bvh_buggy(
    bvh_id: wp.uint64,
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.vec3i),
    query_pos: wp.array(dtype=wp.vec2),
    out_cell: wp.array(dtype=int),
    out_coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pos = query_pos[i]

    closest_cell = int(-1)
    closest_coords = wp.vec3(-1.0e8, -1.0e8, -1.0e8)  # OUTSIDE sentinel

    pad = float(1.0e-5)

    # Outer loop: expand AABB pad until a cell is found.
    # Mirrors: while closest_cell == NULL_ELEMENT_INDEX in geometry.py
    while closest_cell == -1:
        closest_dist = pad * pad

        # Real BVH query — identical to geometry.py:
        #   query = wp.bvh_query_aabb(bvh_id, _bvh_vec(pos) - wp.vec3(pad), ...)
        query = wp.bvh_query_aabb(
            bvh_id,
            wp.vec3(pos[0] - pad, pos[1] - pad, -pad),
            wp.vec3(pos[0] + pad, pos[1] + pad,  pad),
        )
        cell_index = int(0)

        # Inner loop: BVH traversal — identical to geometry.py:
        #   while wp.bvh_query_next(query, cell_index):
        while wp.bvh_query_next(query, cell_index):
            vidx = tri_indices[cell_index]
            p0 = positions[vidx[0]]
            p1 = positions[vidx[1]]
            p2 = positions[vidx[2]]

            q = pos - p0
            e1 = p1 - p0
            e2 = p2 - p0
            dist, coords = tri_closest(q, e1, e2)

            # Bug target: MetaX NVRTC incorrectly handles the predicated write
            # to closest_coords (wp.vec3) inside this if block.
            # Symptom: closest_cell is correct but closest_coords stays at
            # the sentinel (-1e8, -1e8, -1e8), so domain(s_guess) is wrong.
            if dist <= closest_dist:
                closest_dist = dist
                closest_cell = cell_index
                closest_coords = coords

        if pad >= 1.0e6:
            break
        pad = wp.min(4.0 * pad, 1.0e6)

    out_cell[i] = closest_cell
    out_coords[i] = closest_coords


# ─── BVH-based fixed kernel ───────────────────────────────────────────────────
# Mirrors geometry.py AFTER the wp.where fix.
# wp.where compiles to PTX `selp` — unconditional select, no predicated block.


@wp.kernel
def cell_lookup_bvh_fixed(
    bvh_id: wp.uint64,
    positions: wp.array(dtype=wp.vec2),
    tri_indices: wp.array(dtype=wp.vec3i),
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

        query = wp.bvh_query_aabb(
            bvh_id,
            wp.vec3(pos[0] - pad, pos[1] - pad, -pad),
            wp.vec3(pos[0] + pad, pos[1] + pad,  pad),
        )
        cell_index = int(0)

        while wp.bvh_query_next(query, cell_index):
            vidx = tri_indices[cell_index]
            p0 = positions[vidx[0]]
            p1 = positions[vidx[1]]
            p2 = positions[vidx[2]]

            q = pos - p0
            e1 = p1 - p0
            e2 = p2 - p0
            dist, coords = tri_closest(q, e1, e2)

            # Fix: wp.where per component → PTX `selp`, no predicated block.
            is_closer = dist <= closest_dist
            closest_dist = wp.where(is_closer, dist, closest_dist)
            closest_cell = wp.where(is_closer, cell_index, closest_cell)
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
# Exact translation of _gen_trimesh(N, N) + grid_to_tris(N, N).


def gen_trimesh_np(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    positions = np.transpose(
        np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)
    ).reshape(-1, 2).astype(np.float32)

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


def build_bvh(pos_np, tri_np, device):
    """Build wp.Bvh from per-triangle 3D AABB (z extended to cover z=0 plane)."""
    n_tris = tri_np.shape[0]
    lowers = np.zeros((n_tris, 3), dtype=np.float32)
    uppers = np.zeros((n_tris, 3), dtype=np.float32)
    for t in range(n_tris):
        v = tri_np[t]
        pts = pos_np[v]                        # (3, 2)
        lowers[t] = [pts[:, 0].min(), pts[:, 1].min(), 0.0]
        uppers[t] = [pts[:, 0].max(), pts[:, 1].max(), 0.0]
    return wp.Bvh(
        wp.array(lowers, dtype=wp.vec3, device=device),
        wp.array(uppers, dtype=wp.vec3, device=device),
    )


# ─── Verification helper ──────────────────────────────────────────────────────


def reconstruct(pos_np, tri_np, cell, coords):
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

    query_np = np.array(
        [(pos_np[tri_np[t, 0]] + pos_np[tri_np[t, 1]] + pos_np[tri_np[t, 2]]) / 3.0
         for t in range(n_tris)],
        dtype=np.float32,
    )

    with wp.ScopedDevice(device):
        positions   = wp.array(pos_np,   dtype=wp.vec2)
        tri_indices = wp.array(tri_np,   dtype=wp.vec3i)
        query_pos   = wp.array(query_np, dtype=wp.vec2)

        bvh = build_bvh(pos_np, tri_np, device)

        out_cell_b   = wp.zeros(n_tris, dtype=int)
        out_coords_b = wp.zeros(n_tris, dtype=wp.vec3)
        out_cell_f   = wp.zeros(n_tris, dtype=int)
        out_coords_f = wp.zeros(n_tris, dtype=wp.vec3)

        wp.launch(cell_lookup_bvh_buggy, dim=n_tris,
                  inputs=[bvh.id, positions, tri_indices, query_pos,
                          out_cell_b, out_coords_b])
        wp.launch(cell_lookup_bvh_fixed, dim=n_tris,
                  inputs=[bvh.id, positions, tri_indices, query_pos,
                          out_cell_f, out_coords_f])
        wp.synchronize()

        cell_b   = out_cell_b.numpy()
        coords_b = out_coords_b.numpy()
        cell_f   = out_cell_f.numpy()
        coords_f = out_coords_f.numpy()

    tol = 0.001
    buggy_fails = fixed_fails = 0

    print(f"{'tri':>4}  {'query (x,y)':>14}  "
          f"{'bvh_buggy':>14}  {'bvh_fixed':>14}")
    print("-" * 60)

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

    print("-" * 60)
    print(f"bvh_buggy: {buggy_fails}/{n_tris} FAIL  |  bvh_fixed: {fixed_fails}/{n_tris} FAIL")

    if buggy_fails > 0 and fixed_fails == 0:
        print("=> MetaX predicated vec3 write bug confirmed.")
    elif buggy_fails == 0:
        print("=> No bug detected (NVIDIA, patched compiler, or insufficient register pressure).")
    else:
        print("=> UNEXPECTED: fixed kernel also fails.")


if __name__ == "__main__":
    main()
