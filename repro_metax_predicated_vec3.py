"""
Minimal reproduction for MetaX NVRTC compiler bug:
  vec3 assignment inside a predicated (if) block partially fails.

Root cause analysis note
------------------------
During debugging of test_triangle_mesh, elements 2 and 4 failed while element 0 passed.
A preliminary hypothesis labelled 2 and 4 as "Type A triangles (lower-left→lower-right→
upper-right)". This hypothesis is INCORRECT — element 0 has the exact same vertex
arrangement and passes. The real differentiator is likely thread index / BVH traversal
depth, which affects the compiler's register-allocation decisions for each thread.

Actual bug
----------
MetaX NVRTC incorrectly compiles predicated vec3 assignments inside `if` blocks.
Some component writes are silently skipped.  The bug only triggers when the `if` block
is inside a nested loop structure (outer while-sentinel + inner for/while), matching the
pattern in `make_filtered_cell_lookup` in geometry.py.

A flat kernel with a single `if` is too simple — the MetaX optimizer does not apply the
buggy transformation there.

Expected behavior (NVIDIA / CPU)
---------------------------------
  Both test_nested_if_block and test_nested_wp_where produce identical results.

MetaX bug
---------
  test_nested_if_block  → some threads give closest_coords = (0,0,0)   [WRONG]
  test_nested_wp_where  → all threads give correct results              [OK]

Fix
---
Replace `if dist <= closest_dist: closest_coords = coords` with `wp.where` per
component, which compiles to PTX `selp` (unconditional select) instead of predicated
`mov` instructions.  Applied in geometry.py:make_filtered_cell_lookup.
"""

import warp as wp
import numpy as np

wp.init()


@wp.func
def compute_coords(x: float, branch: int) -> wp.vec3:
    """
    Multi-branch function returning vec3, simulating project_on_tri_at_origin.
    Multiple internal branches force the compiler to handle divergent return paths.
    """
    if branch == 0:
        return wp.vec3(1.0 - x, x, x * 0.5)
    elif branch == 1:
        return wp.vec3(x * 0.5, 1.0 - x, x)
    else:
        return wp.vec3(x, x * 0.5, 1.0 - x)


@wp.kernel
def test_nested_if_block(
    values: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    """
    Buggy pattern on MetaX:
      vec3 assigned inside if-block (predicated region) inside nested loops.
      MetaX compiler may skip some component writes for certain threads.

    Mirrors make_filtered_cell_lookup structure:
      outer while (sentinel) → inner for (BVH traversal) → if dist <= threshold
    """
    i = wp.tid()
    NULL = int(-1)

    closest_cell   = int(NULL)
    closest_coords = wp.vec3(-1.0e8, -1.0e8, -1.0e8)  # OUTSIDE sentinel
    closest_dist   = float(0.0)
    pad            = float(1.0e-5)

    # Outer loop: mirrors `while closest_cell == NULL_ELEMENT_INDEX`
    while closest_cell == NULL:
        closest_dist = pad * pad

        # Inner loop: mirrors `while bvh_query_next(..., cell_index)`
        for j in range(5):
            coords = compute_coords(values[i], j % 3)
            dist   = wp.length_sq(coords - wp.vec3(values[i], values[i], values[i]))

            if dist <= closest_dist:        # ← predicated block
                closest_dist   = dist
                closest_cell   = j
                closest_coords = coords     # ← MetaX bug may trigger here

        if pad > float(1.0):
            break
        pad = pad * float(4.0)

    result[i] = closest_coords


@wp.kernel
def test_nested_wp_where(
    values: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    """
    Fixed pattern using wp.where (compiles to PTX selp, not predicated mov).
    Both operands are read unconditionally before selection — no predicated block.
    """
    i = wp.tid()
    NULL = int(-1)

    closest_cell   = int(NULL)
    closest_coords = wp.vec3(-1.0e8, -1.0e8, -1.0e8)
    closest_dist   = float(0.0)
    pad            = float(1.0e-5)

    while closest_cell == NULL:
        closest_dist = pad * pad
        for j in range(5):
            coords    = compute_coords(values[i], j % 3)
            dist      = wp.length_sq(coords - wp.vec3(values[i], values[i], values[i]))
            is_closer = dist <= closest_dist
            closest_dist   = wp.where(is_closer, dist, closest_dist)
            closest_cell   = wp.where(is_closer, j, closest_cell)
            closest_coords = wp.vec3(
                wp.where(is_closer, coords[0], closest_coords[0]),
                wp.where(is_closer, coords[1], closest_coords[1]),
                wp.where(is_closer, coords[2], closest_coords[2]),
            )
        if pad > float(1.0):
            break
        pad = pad * float(4.0)

    result[i] = closest_coords


def run_test(device: str):
    N = 18  # same as test_triangle_mesh cell count

    values_np = np.linspace(0.1, 0.9, N, dtype=np.float32)
    values    = wp.array(values_np, dtype=float, device=device)

    result_if    = wp.zeros(N, dtype=wp.vec3, device=device)
    result_where = wp.zeros(N, dtype=wp.vec3, device=device)

    wp.launch(test_nested_if_block,  dim=N, inputs=[values, result_if],    device=device)
    wp.launch(test_nested_wp_where,  dim=N, inputs=[values, result_where],  device=device)
    wp.synchronize()

    r_if    = result_if.numpy()
    r_where = result_where.numpy()

    print(f"\n=== device: {device} ===")
    print(f"{'i':>3}  {'if-block':>36}  {'wp.where':>36}  status")
    all_match = True
    for i in range(N):
        match = np.allclose(r_if[i], r_where[i], atol=1e-4)
        if not match:
            all_match = False
        flag = "OK" if match else "FAIL <<<"
        print(f"{i:>3}  {str(r_if[i]):>36}  {str(r_where[i]):>36}  {flag}")

    print()
    if all_match:
        print("RESULT: PASS — if-block and wp.where produce identical results.")
    else:
        print("RESULT: FAIL — if-block produces wrong results on this platform.")
        print("              This confirms the MetaX predicated vec3 assignment bug.")


if __name__ == "__main__":
    run_test("cpu")
    run_test("cuda:0")
