"""Tests for analytical roofline estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from convexkernels.bench.shapes import ShapeSpec
from convexkernels.synth import tiers as tiers_mod
from convexkernels.synth.roofline import (
    dense_fista_bytes_per_iter,
    dense_fista_flops_per_iter,
    estimate_dense_fista_roofline,
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
