"""Tests for analytical roofline estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from convexkernels.synth.roofline import (
    dense_fista_bytes_per_iter,
    dense_fista_flops_per_iter,
    estimate_dense_fista_roofline,
    roofline_floor_ms_per_iter,
)


def test_dense_fista_roofline_matches_documented_fp32_formula():
    m, n = 5000, 2000

    assert dense_fista_bytes_per_iter(m, n, "fp32") == (
        8 * m * n + 32 * (m + n)
    )
    assert dense_fista_flops_per_iter(m, n) == 4 * m * n + m + 8 * n
    assert roofline_floor_ms_per_iter(
        m,
        n,
        dtype_name="fp32",
        peak_bandwidth_gb_s=150.0,
    ) == pytest.approx(0.5348266667)


def test_estimate_dense_fista_roofline_from_total_wall_time():
    estimate = estimate_dense_fista_roofline(
        m=5000,
        n=2000,
        dtype_name="fp32",
        wall_time_ms=32.868,
        n_iters=44,
        peak_bandwidth_gb_s=150.0,
    )

    assert estimate.measured_ms_per_iter == pytest.approx(0.747)
    assert estimate.roofline_pct == pytest.approx(71.5966, rel=1e-4)
    assert estimate.achieved_bandwidth_gb_s == pytest.approx(107.395, rel=1e-3)
    assert estimate.arithmetic_intensity == pytest.approx(
        estimate.flops_per_iter / estimate.bytes_per_iter
    )
