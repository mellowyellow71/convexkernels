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


# --- Gram-path cost model + proposer hint -----------------------------------

import numpy as np  # noqa: E402

from convexkernels.frontend.lasso import Lasso  # noqa: E402
from convexkernels.synth.roofline import (  # noqa: E402
    amortization_crossover_solves,
    gram_fista_bytes_per_iter,
    gram_fista_flops_per_iter,
    gram_setup_flops,
    roofline_hint,
)


def _lasso(m, n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    return Lasso(A, b, 1.0)


def test_gram_bytes_flops_and_symmetric_halving():
    n = 2000
    assert gram_fista_bytes_per_iter(n, "fp32") == n * n * 4 + 8 * n * 4
    assert gram_fista_bytes_per_iter(n, "fp32", symmetric=True) == (
        (n * (n + 1) // 2) * 4 + 8 * n * 4
    )
    assert gram_fista_flops_per_iter(n) == 2 * n * n + 9 * n
    assert gram_setup_flops(5000, n) == 2 * 5000 * n * n + 2 * 5000 * n


def test_roofline_hint_tall_regime_prefers_gram():
    hint = roofline_hint(_lasso(5000, 2000))
    assert hint["shape"]["regime"] == "tall"
    # direct reads A twice (2mn), gram reads G once (n^2) -> ~5x on tall shape
    assert hint["gram_vs_direct_byte_ratio"] == pytest.approx(5.0, rel=2e-3)
    pi = hint["per_iter"]
    assert pi["gram"]["bytes_per_iter_mb"] < pi["direct"]["bytes_per_iter_mb"]
    assert pi["gram_symmetric"]["bytes_per_iter_mb"] < pi["gram"]["bytes_per_iter_mb"]
    assert any("symv" in lever or "symmetric" in lever for lever in hint["levers"])


def test_roofline_hint_wide_regime_prefers_direct():
    hint = roofline_hint(_lasso(500, 4000))
    assert hint["shape"]["regime"] == "wide"
    # G is (n,n) with n>>m, so direct moves fewer bytes/iter
    pi = hint["per_iter"]
    assert pi["direct"]["bytes_per_iter_mb"] < pi["gram"]["bytes_per_iter_mb"]


def test_amortization_crossover_solves():
    assert amortization_crossover_solves(3414.0, 53.6, 20.6) == pytest.approx(
        3414.0 / (53.6 - 20.6)
    )
    assert amortization_crossover_solves(1000.0, 20.0, 25.0) == float("inf")
