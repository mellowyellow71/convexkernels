"""Tests for analytical roofline estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from convexkernels.bench.shapes import ShapeSpec
from convexkernels.synth import tiers as tiers_mod
from convexkernels.synth.roofline import (
    amortization_crossover,
    dense_fista_bytes_per_iter,
    dense_fista_flops_per_iter,
    estimate_dense_fista_roofline,
    estimate_fista_roofline,
    estimate_gram_fista_roofline,
    gram_fista_bytes_per_iter,
    gram_fista_flops_per_iter,
    gram_setup_flops,
    roofline_floor_ms_per_iter,
)
from convexkernels.synth.sandbox import SandboxResult


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


def test_run_tier3_records_roofline_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results = [
        SandboxResult(
            status="completed",
            kkt_final=1e-7,
            wall_time_s=0.002,
            converged=True,
            iters=10,
        )
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    spec = ShapeSpec("tiny", m=10, n=20)
    tier = tiers_mod.run_tier3(
        run_dir=tmp_path / "tier3",
        shapes=(spec,),
        kernel_module="convexkernels.kernels.numpy_ref",
        config=tiers_mod.EvalConfig(
            seed_kernel={
                "module": "convexkernels.kernels.numpy_ref",
                "step": "fista_step",
                "init": "init_state",
            },
            peak_bandwidth_gb_s=100.0,
        ),
        max_iters=100,
        tol=1e-6,
        reps=1,
    )

    per_shape = tier.per_shape[0]
    assert tier.passed
    assert per_shape.bytes_per_iter == dense_fista_bytes_per_iter(10, 20, "fp32")
    assert per_shape.measured_ms_per_iter == pytest.approx(0.2)
    assert per_shape.roofline_floor_ms_per_iter > 0.0
    assert per_shape.roofline_pct_med > 0.0
    assert tier.rank_summary["median_roofline_pct"] == per_shape.roofline_pct_med
    assert tier.rank_summary["peak_bandwidth_gb_s"] == 100.0


# ---------------------------------------------------------------------------
# Gram-path model
# ---------------------------------------------------------------------------


def test_gram_fista_bytes_and_flops_match_single_matvec():
    n = 2000

    # Dense Gram per-iter traffic is one read of the (n, n) matrix plus O(n).
    assert gram_fista_bytes_per_iter(n, "fp32") == n * n * 4 + 8 * n * 4
    # Symmetric (symv) kernel reads only the lower triangle: ~half the matrix.
    assert gram_fista_bytes_per_iter(n, "fp32", symmetric=True) == (
        (n * (n + 1) // 2) * 4 + 8 * n * 4
    )
    assert gram_fista_flops_per_iter(n) == 2 * n * n + 9 * n


def test_gram_moves_far_less_per_iter_than_direct_for_tall_shapes():
    # tall_medium: the Gram champion's regime. Per-iteration, Gram reads only G
    # (n^2) while direct reads A twice (2mn), so Gram moves ~5x fewer bytes.
    m, n = 5000, 2000
    direct = dense_fista_bytes_per_iter(m, n, "fp32")
    gram = gram_fista_bytes_per_iter(n, "fp32")
    assert direct / gram == pytest.approx(5.0, rel=2e-3)
    # Symmetric storage roughly halves Gram traffic again.
    gram_sym = gram_fista_bytes_per_iter(n, "fp32", symmetric=True)
    assert gram / gram_sym == pytest.approx(2.0, rel=1e-2)


def test_estimate_gram_roofline_labels_strategy_and_uses_gram_bytes():
    est = estimate_gram_fista_roofline(
        m=5000,
        n=2000,
        dtype_name="fp32",
        wall_time_ms=20.6,
        n_iters=22,
        peak_bandwidth_gb_s=150.0,
    )
    assert est.strategy == "gram"
    assert est.bytes_per_iter == gram_fista_bytes_per_iter(2000, "fp32")
    assert est.measured_ms_per_iter == pytest.approx(20.6 / 22)
    assert est.arithmetic_intensity == pytest.approx(
        est.flops_per_iter / est.bytes_per_iter
    )
    sym = estimate_gram_fista_roofline(
        m=5000, n=2000, dtype_name="fp32",
        wall_time_ms=20.6, n_iters=22, symmetric=True,
    )
    assert sym.strategy == "gram_symmetric"
    assert sym.bytes_per_iter < est.bytes_per_iter


def test_estimate_fista_roofline_dispatches_by_strategy():
    common = dict(
        m=5000, n=2000, dtype_name="fp32", wall_time_ms=30.0,
        n_iters=22, peak_bandwidth_gb_s=150.0,
    )
    direct = estimate_fista_roofline(gradient_strategy="direct", **common)
    gram = estimate_fista_roofline(gradient_strategy="gram", **common)
    assert direct.strategy == "direct"
    assert direct.bytes_per_iter == dense_fista_bytes_per_iter(5000, 2000, "fp32")
    assert gram.strategy == "gram"
    assert gram.bytes_per_iter == gram_fista_bytes_per_iter(2000, "fp32")


def test_gram_setup_flops_dominated_by_gram_matrix():
    m, n = 5000, 2000
    assert gram_setup_flops(m, n) == 2 * m * n * n + 2 * m * n


def test_amortization_crossover_breakeven_solves():
    # Gram pays a one-time setup but each solve is cheaper; the crossover is the
    # number of repeated solves needed before Gram wins end-to-end.
    cross = amortization_crossover(
        setup_ms=3414.0, direct_solve_ms=53.6, gram_solve_ms=20.6
    )
    assert cross.gram_ever_wins
    assert cross.per_solve_saving_ms == pytest.approx(33.0)
    assert cross.crossover_solves == pytest.approx(3414.0 / 33.0, rel=1e-6)


def test_amortization_crossover_infinite_when_gram_not_faster():
    cross = amortization_crossover(
        setup_ms=1000.0, direct_solve_ms=20.0, gram_solve_ms=25.0
    )
    assert not cross.gram_ever_wins
    assert cross.crossover_solves == float("inf")


def test_tier3_roofline_uses_gram_model_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results = [
        SandboxResult(
            status="completed",
            kkt_final=1e-7,
            wall_time_s=0.002,
            converged=True,
            iters=10,
        )
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    spec = ShapeSpec("tiny", m=40, n=20)
    tier = tiers_mod.run_tier3(
        run_dir=tmp_path / "tier3",
        shapes=(spec,),
        kernel_module="convexkernels.kernels.numpy_ref",
        config=tiers_mod.EvalConfig(
            seed_kernel={
                "module": "convexkernels.kernels.numpy_ref",
                "step": "fista_step",
                "init": "init_state",
            },
            gradient_strategy="gram",
            peak_bandwidth_gb_s=100.0,
        ),
        max_iters=100,
        tol=1e-6,
        reps=1,
    )

    per_shape = tier.per_shape[0]
    assert tier.passed
    # Gram model keys per-iter traffic on n only, not 2mn.
    assert per_shape.bytes_per_iter == gram_fista_bytes_per_iter(20, "fp32")
