"""MLX capability probe — run on a Mac with Apple Silicon.

Usage:
    cd /path/to/convexkernels
    uv sync --extra mac --extra dev
    python docs/probes/mac_probe.py | tee tasks/mac_probe_output.txt

Paste the resulting text into tasks/results.md under "P0 -> MLX probe".
"""

import time

import mlx.core as mx


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def probe_metal_kernel() -> None:
    section("Probe 1: mx.fast.metal_kernel smoke test")
    source = """
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem] * inp[elem];
    """
    kernel = mx.fast.metal_kernel(
        name="square",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )
    a = mx.arange(1_000_000, dtype=mx.float32)
    (out,) = kernel(
        inputs=[a],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )
    mx.eval(out)
    print(f"out[42] = {float(out[42])} (expect 1764.0)")
    print(f"out[1000] = {float(out[1000])} (expect 1000000.0)")


def probe_linalg_surface() -> None:
    section("Probe 2: mlx.core.linalg surface inventory")
    m = mx.linalg
    print(f"dir: {[x for x in dir(m) if not x.startswith('_')]}")
    for name in ["cholesky", "solve_triangular", "solve", "lu", "qr",
                 "svd", "norm", "inv", "tri", "cross"]:
        present = hasattr(m, name)
        print(f"  {name}: {'present' if present else 'MISSING'}")


def time_matvec(m: int, n: int, dtype, reps: int = 100) -> tuple[float, float]:
    A = mx.random.normal(shape=(m, n), dtype=dtype)
    x = mx.random.normal(shape=(n,), dtype=dtype)
    mx.eval(A, x)
    for _ in range(5):
        y = A @ x
        z = A.T @ y
        mx.eval(z)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = A @ x
        z = A.T @ y
        mx.eval(z)
    elapsed = (time.perf_counter() - t0) / reps
    bytes_per_iter = 2 * m * n * dtype.size
    bw = bytes_per_iter / elapsed / 1e9
    return elapsed * 1e3, bw


SHAPES = [(2000, 500), (5000, 2000), (500, 2000), (2000, 10000)]


def probe_matvec_baseline() -> None:
    section("Probe 3: mx.matmul baseline timing on bench shapes (fp32)")
    print(f"{'(m,n)':<14} {'ms/iter':>10} {'GB/s':>10}")
    for m, n in SHAPES:
        t_ms, bw = time_matvec(m, n, mx.float32)
        print(f"{f'({m},{n})':<14} {t_ms:>10.3f} {bw:>10.1f}")


def probe_precision_axis() -> None:
    section("Probe 4: precision axis dry run")
    for dtype in [mx.float32, mx.float16, mx.bfloat16]:
        print(f"\n  dtype = {dtype}")
        print(f"  {'(m,n)':<14} {'ms/iter':>10} {'GB/s':>10}")
        for m, n in [(2000, 500), (5000, 2000)]:
            t_ms, bw = time_matvec(m, n, dtype)
            print(f"  {f'({m},{n})':<14} {t_ms:>10.3f} {bw:>10.1f}")


if __name__ == "__main__":
    print(f"mlx version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"default device: {mx.default_device()}")
    probe_metal_kernel()
    probe_linalg_surface()
    probe_matvec_baseline()
    probe_precision_axis()
