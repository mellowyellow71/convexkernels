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
    )


# ---------------------------------------------------------------------------
# Gram-path model + proposer cost-model hint
#
# The direct gradient `g = A^T(Ay - b)` reads the (m, n) matrix `A` twice per
# iteration. The Gram path precomputes `G = A^T A` (n, n) and `c = A^T b` once,
# then each iteration is a single dense matvec `g = G y - c`. For tall problems
# (m >> n) this trades an expensive one-time setup for far less per-iteration
# traffic. With the algorithm now an open search dimension, the loop hands the
# proposer a `roofline_hint(problem)` so it can reason about which gradient form
# / precision / layout is bandwidth-favourable for the active shape, instead of
# rediscovering it by trial and error.
# ---------------------------------------------------------------------------


def gram_fista_bytes_per_iter(
    n: int,
    dtype_name: str = "fp32",
    *,
    symmetric: bool = False,
) -> int:
    """Approximate dense Gram-FISTA bytes moved per iteration.

    The hot path is one matvec `g = G y - c`: `G` (n, n) read once plus O(n)
    vector traffic. `G = A^T A` is symmetric, so a triangular/`symv`-style
    kernel reads only the lower triangle (`symmetric=True`) — a ~2x bandwidth
    lever the FLOP count alone does not expose.
    """
    scalar_bytes = dtype_storage_bytes(dtype_name)
    matrix_elems = n * (n + 1) // 2 if symmetric else n * n
    return int(matrix_elems * scalar_bytes + 8 * n * scalar_bytes)


def gram_fista_flops_per_iter(n: int) -> int:
    """Approximate FLOPs per dense Gram-FISTA iteration (`2 n^2` matvec + O(n))."""
    return int(2 * n * n + 9 * n)


def gram_setup_flops(m: int, n: int) -> int:
    """FLOPs to form Gram data once: `G = A^T A` (`2 m n^2`) + `A^T b` (`2 m n`)."""
    return int(2 * m * n * n + 2 * m * n)


def gram_fista_peak_mem_mb(m: int, n: int, dtype_name: str = "fp32") -> float:
    """Resident artifact size for Gram FISTA: `G` (n, n) dominates."""
    scalar_bytes = dtype_storage_bytes(dtype_name)
    return (n * n + 6 * n) * scalar_bytes / (1024 * 1024)


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
    """Estimate Gram-path roofline utilization from solve wall time and iters."""
    bytes_per_iter = gram_fista_bytes_per_iter(n, dtype_name, symmetric=symmetric)
    flops_per_iter = gram_fista_flops_per_iter(n)
    floor_ms = gram_roofline_floor_ms_per_iter(
        n, dtype_name=dtype_name, symmetric=symmetric,
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
    )


def amortization_crossover_solves(
    setup_ms: float, direct_solve_ms: float, gram_solve_ms: float
) -> float:
    """Break-even number of repeated (fixed-`A`) solves for Gram setup to pay off.

    `direct(N) = N * direct_solve_ms`; `gram(N) = setup_ms + N * gram_solve_ms`;
    equal at `N* = setup_ms / (direct_solve_ms - gram_solve_ms)`. Returns `inf`
    when Gram's per-solve time is not faster (setup never amortizes).
    """
    saving = direct_solve_ms - gram_solve_ms
    if saving <= 0.0:
        return float("inf")
    return setup_ms / saving


def _per_iter_view(
    *, label: str, bytes_per_iter: int, flops_per_iter: int,
    floor_ms: float,
) -> dict:
    return {
        "strategy": label,
        "bytes_per_iter_mb": round(bytes_per_iter / 1e6, 4),
        "arithmetic_intensity_ops_per_byte": round(flops_per_iter / bytes_per_iter, 4),
        "roofline_floor_ms_per_iter": round(floor_ms, 6),
    }


def roofline_hint(
    problem,
    *,
    dtype_name: str = "fp32",
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S,
) -> dict:
    """Compact, proposer-facing bandwidth/AI hint for the active problem shape.

    Pure analytical: compares the per-iteration memory traffic of the direct,
    Gram, and symmetric-Gram gradient paths at the target peak bandwidth, plus a
    qualitative regime read and concrete levers. The goal is to steer the open
    algorithm search toward the bandwidth-favourable form for this shape rather
    than have the proposer rediscover it.
    """
    m = int(getattr(problem, "m"))
    n = int(getattr(problem, "n"))
    direct_bytes = dense_fista_bytes_per_iter(m, n, dtype_name)
    gram_bytes = gram_fista_bytes_per_iter(n, dtype_name)
    gram_sym_bytes = gram_fista_bytes_per_iter(n, dtype_name, symmetric=True)

    per_iter = {
        "direct": _per_iter_view(
            label="direct",
            bytes_per_iter=direct_bytes,
            flops_per_iter=dense_fista_flops_per_iter(m, n),
            floor_ms=roofline_floor_ms_per_iter(
                m, n, dtype_name=dtype_name, peak_bandwidth_gb_s=peak_bandwidth_gb_s),
        ),
        "gram": _per_iter_view(
            label="gram",
            bytes_per_iter=gram_bytes,
            flops_per_iter=gram_fista_flops_per_iter(n),
            floor_ms=gram_roofline_floor_ms_per_iter(
                n, dtype_name=dtype_name, peak_bandwidth_gb_s=peak_bandwidth_gb_s),
        ),
        "gram_symmetric": _per_iter_view(
            label="gram_symmetric",
            bytes_per_iter=gram_sym_bytes,
            flops_per_iter=gram_fista_flops_per_iter(n),
            floor_ms=gram_roofline_floor_ms_per_iter(
                n, dtype_name=dtype_name, symmetric=True,
                peak_bandwidth_gb_s=peak_bandwidth_gb_s),
        ),
    }

    tall_ratio = m / n if n else 1.0
    if tall_ratio >= 2.0:
        regime = "tall"
        gram_note = (
            f"m/n={tall_ratio:.1f}: Gram moves "
            f"{direct_bytes / gram_bytes:.1f}x fewer bytes/iter than direct; "
            f"favourable once setup amortizes (gram_setup≈{gram_setup_flops(m, n):.2e} FLOPs)."
        )
    elif tall_ratio <= 0.5:
        regime = "wide"
        gram_note = (
            f"m/n={tall_ratio:.2f}: wide; Gram G is (n,n) with n>m, so the "
            "direct path moves fewer bytes/iter — prefer direct here."
        )
    else:
        regime = "square"
        gram_note = (
            f"m/n={tall_ratio:.2f}: near-square; direct vs Gram per-iter traffic "
            "is comparable — decide on setup amortization and iteration count."
        )

    return {
        "shape": {"m": m, "n": n, "regime": regime},
        "peak_bandwidth_gb_s": peak_bandwidth_gb_s,
        "dtype": dtype_name,
        "per_iter": per_iter,
        "gram_vs_direct_byte_ratio": round(direct_bytes / gram_bytes, 3),
        "amortization": gram_note,
        "levers": [
            "FISTA per-iter is bandwidth-bound (AI≈0.5): minimize bytes/iter, "
            "not FLOPs.",
            "Symmetric (symv) Gram reads only the lower triangle of G = A^T A "
            f"(~{gram_bytes / gram_sym_bytes:.1f}x fewer bytes than dense Gram).",
            "KKT-per-iter is a second matvec; checking KKT every k iters ~halves "
            "Gram per-iter traffic.",
            "fp16/bf16 or quantized G halves bytes again, but must keep the "
            "trusted fp32 KKT check (squared condition number is the risk).",
        ],
    }


def _measured_ms_per_iter(wall_time_ms: float, n_iters: int) -> float:
    if n_iters <= 0 or not isfinite(wall_time_ms):
        return 0.0
    return max(0.0, float(wall_time_ms)) / n_iters
