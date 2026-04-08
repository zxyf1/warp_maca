"""
Minimal reproduction for MetaX NVRTC compiler bug:
  vec3 assignment inside a predicated (if) block partially fails.

Expected behavior (NVIDIA):
  Both kernels produce the same result.

MetaX bug:
  test_if_block  -> closest = (0.0, 0.0, 0.0)  [wrong]
  test_wp_where  -> closest = correct values    [correct]
"""

import warp as wp
import numpy as np

wp.init()


@wp.func
def compute_coords(x: float) -> wp.vec3:
    """Simulate cell_closest_point: returns a computed vec3."""
    return wp.vec3(1.0 - x, x, x * 0.5)


@wp.kernel
def test_if_block(
    values: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    """
    Buggy pattern on MetaX:
      vec3 assigned inside if-block (predicated region).
      MetaX compiler may skip some component writes.
    """
    i = wp.tid()

    dist = float(0.0)           # always 0 (inside element)
    threshold = float(1.0e-9)   # dist < threshold → condition is TRUE
    closest = wp.vec3(-1.0e8, -1.0e8, -1.0e8)  # initial "not found" value

    coords = compute_coords(values[i])  # returns a real vec3

    if dist <= threshold:       # TRUE: 0 <= 1e-9
        closest = coords        # MetaX bug: may not write all 3 components

    result[i] = closest


@wp.kernel
def test_wp_where(
    values: wp.array(dtype=float),
    result: wp.array(dtype=wp.vec3),
):
    """
    Fixed pattern using wp.where (compiles to PTX selp, not predicated mov).
    Both operands are read unconditionally before selection.
    """
    i = wp.tid()

    dist = float(0.0)
    threshold = float(1.0e-9)
    closest = wp.vec3(-1.0e8, -1.0e8, -1.0e8)

    coords = compute_coords(values[i])

    is_closer = dist <= threshold
    closest = wp.vec3(
        wp.where(is_closer, coords[0], closest[0]),
        wp.where(is_closer, coords[1], closest[1]),
        wp.where(is_closer, coords[2], closest[2]),
    )

    result[i] = closest


def run_test(device: str):
    N = 18  # same as test_triangle_mesh cell count

    values_np = np.linspace(0.1, 0.9, N, dtype=np.float32)
    values = wp.array(values_np, dtype=float, device=device)

    result_if    = wp.zeros(N, dtype=wp.vec3, device=device)
    result_where = wp.zeros(N, dtype=wp.vec3, device=device)

    wp.launch(test_if_block, dim=N, inputs=[values, result_if],    device=device)
    wp.launch(test_wp_where,  dim=N, inputs=[values, result_where], device=device)
    wp.synchronize()

    r_if    = result_if.numpy()
    r_where = result_where.numpy()

    print(f"\n=== device: {device} ===")
    print(f"{'i':>3}  {'if-block':>30}  {'wp.where':>30}  {'match':>6}")
    all_match = True
    for i in range(N):
        match = np.allclose(r_if[i], r_where[i], atol=1e-4)
        if not match:
            all_match = False
        flag = "OK" if match else "FAIL <<<"
        print(f"{i:>3}  {str(r_if[i]):>30}  {str(r_where[i]):>30}  {flag}")

    print()
    if all_match:
        print("RESULT: PASS — if-block and wp.where produce identical results.")
    else:
        print("RESULT: FAIL — if-block produces wrong results on this platform.")
        print("              This confirms the MetaX predicated vec3 assignment bug.")


if __name__ == "__main__":
    run_test("cpu")
    run_test("cuda:0")
