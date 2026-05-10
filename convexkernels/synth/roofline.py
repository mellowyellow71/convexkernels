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


def _measured_ms_per_iter(wall_time_ms: float, n_iters: int) -> float:
    if n_iters <= 0 or not isfinite(wall_time_ms):
        return 0.0
    return max(0.0, float(wall_time_ms)) / n_iters
