"""Analytical roofline estimates for dense LASSO/FISTA runs.

This is intentionally source-independent for the MVP: the dominant work is the
two dense matvecs in the host algorithm, while synthesized Metal kernels mutate
the O(n) tail. The estimate gives the proposer a scale-aware signal for whether
a candidate is bandwidth-bound or mostly paying launch/framework overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


DEFAULT_APPLE_SILICON_PEAK_GB_S = 150.0


@dataclass(frozen=True)
class RooflineEstimate:
    m: int
    n: int
    dtype: str
    bytes_per_iter: int
    flops_per_iter: int
    arithmetic_intensity: float
    roofline_floor_ms_per_iter: float
    measured_ms_per_iter: float
    achieved_bandwidth_gb_s: float
    roofline_pct: float
    peak_bandwidth_gb_s: float
    peak_mem_mb: float
    strategy: str = "direct"


def dtype_storage_bytes(dtype_name: str) -> int:
    """Return storage bytes for the named dtype used by eval configs."""
    dtype = dtype_name.lower()
    if dtype in {"fp16", "float16", "bf16", "bfloat16"}:
        return 2
    if dtype in {"fp64", "float64"}:
        return 8
    return 4


def dense_fista_bytes_per_iter(m: int, n: int, dtype_name: str = "fp32") -> int:
    """Approximate dense FISTA bytes moved per iteration.

    The model counts two full reads of dense `A` plus O(m+n) vector traffic.
    It matches `docs/roofline.md` for fp32:
    `8*m*n + 32*(m+n)` bytes.
    """
    scalar_bytes = dtype_storage_bytes(dtype_name)
    matrix_bytes = 2 * m * n * scalar_bytes
    vector_bytes = 8 * (m + n) * scalar_bytes
    return int(matrix_bytes + vector_bytes)


def dense_fista_flops_per_iter(m: int, n: int) -> int:
    """Approximate floating-point ops per dense FISTA iteration."""
    matvec_flops = 4 * m * n
    residual_flops = m
    tail_flops = 8 * n
    return int(matvec_flops + residual_flops + tail_flops)


def dense_fista_peak_mem_mb(m: int, n: int, dtype_name: str = "fp32") -> float:
    """Coarse resident artifact size for dense LASSO/FISTA evaluation."""
    scalar_bytes = dtype_storage_bytes(dtype_name)
    n_scalars = m * n + 4 * n + m
    return n_scalars * scalar_bytes / (1024 * 1024)


def roofline_floor_ms_per_iter(
    m: int,
    n: int,
    *,
    dtype_name: str = "fp32",
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> float:
    bytes_per_iter = dense_fista_bytes_per_iter(m, n, dtype_name)
    return bytes_per_iter / (peak_bandwidth_gb_s * 1e9) * 1e3


def estimate_dense_fista_roofline(
    *,
    m: int,
    n: int,
    dtype_name: str = "fp32",
    wall_time_ms: float,
    n_iters: int,
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> RooflineEstimate:
    """Estimate roofline utilization from total wall time and iteration count."""
    bytes_per_iter = dense_fista_bytes_per_iter(m, n, dtype_name)
    flops_per_iter = dense_fista_flops_per_iter(m, n)
    floor_ms = roofline_floor_ms_per_iter(
        m,
        n,
        dtype_name=dtype_name,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
    )
    measured_ms_per_iter = _measured_ms_per_iter(wall_time_ms, n_iters)
    achieved_bandwidth_gb_s = (
        bytes_per_iter / (measured_ms_per_iter / 1e3) / 1e9
        if measured_ms_per_iter > 0 else 0.0
    )
    roofline_pct = (
        min(100.0, floor_ms / measured_ms_per_iter * 100.0)
        if measured_ms_per_iter > 0 else 0.0
    )
    return RooflineEstimate(
        m=m,
        n=n,
        dtype=dtype_name,
        bytes_per_iter=bytes_per_iter,
        flops_per_iter=flops_per_iter,
        arithmetic_intensity=flops_per_iter / bytes_per_iter,
        roofline_floor_ms_per_iter=floor_ms,
        measured_ms_per_iter=measured_ms_per_iter,
        achieved_bandwidth_gb_s=achieved_bandwidth_gb_s,
        roofline_pct=roofline_pct,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
        peak_mem_mb=dense_fista_peak_mem_mb(m, n, dtype_name),
        strategy="direct",
    )


# ---------------------------------------------------------------------------
# Gram-path model
#
# The direct gradient `g = A^T(Ay - b)` reads the (m, n) matrix `A` twice per
# iteration. The Gram path precomputes `G = A^T A` (n, n) and `c = A^T b` once,
# then each iteration is a single dense matvec `g = G y - c`. For tall problems
# (m >> n) this trades an expensive one-time setup for far less per-iteration
# traffic, which is exactly why the current `tall_medium` champion is a Gram
# variant. The direct model in this file does NOT describe that champion, so the
# functions below give the proposer a strategy-correct bandwidth/AI signal and
# an analytical setup-amortization crossover.
# ---------------------------------------------------------------------------


def gram_fista_bytes_per_iter(
    n: int,
    dtype_name: str = "fp32",
    *,
    symmetric: bool = False,
) -> int:
    """Approximate dense Gram-FISTA bytes moved per iteration.

    The hot path is one matvec `g = G y - c`. `G` is (n, n) and is read once;
    `c` and the O(n) FISTA vectors (y, x, x_prev, g, z) add linear traffic.

    `G = A^T A` is symmetric, so a triangular/`symv`-style kernel reads only the
    lower triangle: `symmetric=True` models that ~2x bandwidth lever. This is a
    real kernel option that the matvec FLOP count alone does not expose.
    """
    scalar_bytes = dtype_storage_bytes(dtype_name)
    if symmetric:
        matrix_elems = n * (n + 1) // 2
    else:
        matrix_elems = n * n
    matrix_bytes = matrix_elems * scalar_bytes
    vector_bytes = 8 * n * scalar_bytes
    return int(matrix_bytes + vector_bytes)


def gram_fista_flops_per_iter(n: int) -> int:
    """Approximate FLOPs per dense Gram-FISTA iteration.

    The matvec `G y` is `2 n^2` FLOPs (n^2 FMAs); the `- c`, prox, and momentum
    tail are O(n). A symmetric kernel does the same arithmetic with half the
    reads, so FLOPs are storage-independent here.
    """
    matvec_flops = 2 * n * n
    tail_flops = 9 * n
    return int(matvec_flops + tail_flops)


def gram_setup_flops(m: int, n: int) -> int:
    """FLOPs to form the Gram data once: `G = A^T A` and `c = A^T b`.

    `A^T A` is an (n, n) output whose entries are length-m dot products
    (`m n^2` FMAs = `2 m n^2` FLOPs); `A^T b` adds `2 m n`. This one-time cost is
    what the amortization crossover has to pay back.
    """
    return int(2 * m * n * n + 2 * m * n)


def gram_setup_bytes(m: int, n: int, dtype_name: str = "fp32") -> int:
    """Lower-bound bytes for Gram setup: read `A` once, write `G`.

    A single streamed pass over `A` is `m n` reads; the `G` result is `n^2`
    writes. A real GEMM rereads tiles of `A`, so this is an optimistic floor
    used only to characterize setup as bandwidth- vs compute-leaning.
    """
    scalar_bytes = dtype_storage_bytes(dtype_name)
    return int((m * n + n * n) * scalar_bytes)


def gram_fista_peak_mem_mb(m: int, n: int, dtype_name: str = "fp32") -> float:
    """Resident artifact size for Gram FISTA: `G` (n, n) dominates `A` once freed."""
    scalar_bytes = dtype_storage_bytes(dtype_name)
    n_scalars = n * n + 6 * n
    return n_scalars * scalar_bytes / (1024 * 1024)


def gram_roofline_floor_ms_per_iter(
    n: int,
    *,
    dtype_name: str = "fp32",
    symmetric: bool = False,
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> float:
    bytes_per_iter = gram_fista_bytes_per_iter(n, dtype_name, symmetric=symmetric)
    return bytes_per_iter / (peak_bandwidth_gb_s * 1e9) * 1e3


def estimate_gram_fista_roofline(
    *,
    m: int,
    n: int,
    dtype_name: str = "fp32",
    wall_time_ms: float,
    n_iters: int,
    symmetric: bool = False,
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> RooflineEstimate:
    """Estimate Gram-path roofline utilization from solve wall time and iters.

    `wall_time_ms` should be the setup-excluded solve time (the amortized cost
    model): setup is a one-time cost characterized separately by
    `amortization_crossover`.
    """
    bytes_per_iter = gram_fista_bytes_per_iter(n, dtype_name, symmetric=symmetric)
    flops_per_iter = gram_fista_flops_per_iter(n)
    floor_ms = gram_roofline_floor_ms_per_iter(
        n,
        dtype_name=dtype_name,
        symmetric=symmetric,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
    )
    measured_ms_per_iter = _measured_ms_per_iter(wall_time_ms, n_iters)
    achieved_bandwidth_gb_s = (
        bytes_per_iter / (measured_ms_per_iter / 1e3) / 1e9
        if measured_ms_per_iter > 0 else 0.0
    )
    roofline_pct = (
        min(100.0, floor_ms / measured_ms_per_iter * 100.0)
        if measured_ms_per_iter > 0 else 0.0
    )
    return RooflineEstimate(
        m=m,
        n=n,
        dtype=dtype_name,
        bytes_per_iter=bytes_per_iter,
        flops_per_iter=flops_per_iter,
        arithmetic_intensity=flops_per_iter / bytes_per_iter,
        roofline_floor_ms_per_iter=floor_ms,
        measured_ms_per_iter=measured_ms_per_iter,
        achieved_bandwidth_gb_s=achieved_bandwidth_gb_s,
        roofline_pct=roofline_pct,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
        peak_mem_mb=gram_fista_peak_mem_mb(m, n, dtype_name),
        strategy="gram_symmetric" if symmetric else "gram",
    )


def estimate_fista_roofline(
    *,
    m: int,
    n: int,
    dtype_name: str = "fp32",
    wall_time_ms: float,
    n_iters: int,
    gradient_strategy: str = "direct",
    symmetric: bool = False,
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> RooflineEstimate:
    """Dispatch to the direct or Gram roofline model by gradient strategy.

    `gradient_strategy="gram"` selects the Gram per-iter traffic model so the
    reported bandwidth/AI/utilization describe the Gram champion rather than the
    direct path it replaced.
    """
    if gradient_strategy == "gram":
        return estimate_gram_fista_roofline(
            m=m,
            n=n,
            dtype_name=dtype_name,
            wall_time_ms=wall_time_ms,
            n_iters=n_iters,
            symmetric=symmetric,
            peak_bandwidth_gb_s=peak_bandwidth_gb_s,
        )
    return estimate_dense_fista_roofline(
        m=m,
        n=n,
        dtype_name=dtype_name,
        wall_time_ms=wall_time_ms,
        n_iters=n_iters,
        peak_bandwidth_gb_s=peak_bandwidth_gb_s,
    )


@dataclass(frozen=True)
class AmortizationCrossover:
    """Setup-amortization analysis for switching direct -> Gram on fixed `A`.

    With `A` fixed across repeated solves, Gram pays `setup_ms` once and saves
    `per_solve_saving_ms` on every solve. `crossover_solves` is the break-even
    number of solves; below it, direct is cheaper end-to-end even though Gram has
    the faster per-iteration solve.
    """
    setup_ms: float
    direct_solve_ms: float
    gram_solve_ms: float
    per_solve_saving_ms: float
    crossover_solves: float
    gram_ever_wins: bool


def amortization_crossover(
    *,
    setup_ms: float,
    direct_solve_ms: float,
    gram_solve_ms: float,
) -> AmortizationCrossover:
    """Break-even solve count for amortizing Gram setup over repeated solves.

    `direct total(N) = N * direct_solve_ms`
    `gram total(N)   = setup_ms + N * gram_solve_ms`
    Equal at `N* = setup_ms / (direct_solve_ms - gram_solve_ms)`.

    If Gram's per-solve time is not faster, setup can never be amortized and
    `crossover_solves` is returned as `inf`.
    """
    saving = direct_solve_ms - gram_solve_ms
    if saving <= 0.0:
        crossover = float("inf")
    else:
        crossover = setup_ms / saving
    return AmortizationCrossover(
        setup_ms=float(setup_ms),
        direct_solve_ms=float(direct_solve_ms),
        gram_solve_ms=float(gram_solve_ms),
        per_solve_saving_ms=float(saving),
        crossover_solves=float(crossover),
        gram_ever_wins=saving > 0.0,
    )


def _measured_ms_per_iter(wall_time_ms: float, n_iters: int) -> float:
    if n_iters <= 0 or not isfinite(wall_time_ms):
        return 0.0
    return max(0.0, float(wall_time_ms)) / n_iters
