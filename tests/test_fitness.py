"""Tests for structured fitness summaries."""

from __future__ import annotations

import json
from pathlib import Path

from convexkernels.synth.fitness import (
    build_fitness_report,
    fitness_from_record,
    summarize_fitness_report,
    write_fitness_report,
)


def _record(
    reason: str,
    *,
    accepted: bool = False,
    tier2_wall_ms: float = 102.0,
    speed_ref_ms: float = 104.0,
    speed_margin: float = 0.97,
    roofline_pct: float | None = None,
) -> dict:
    record = {
        "id": "rec-1",
        "slot": {
            "problem_family": "lasso",
            "algorithm": "fista",
            "hardware": "apple_silicon",
            "dtype": "fp32",
        },
        "edit": {"type": "vectorize", "payload": {}, "rationale": "test"},
        "tier1": {
            "passed": True,
            "wall_time_ms": 45.0,
            "wall_time_std_ms": 2.25,
        },
        "tier2": {
            "passed": True,
            "converged": True,
            "n_iters": 93,
            "wall_time_ms": tier2_wall_ms,
            "wall_time_std_ms": 2.0,
            "speed_ref_wall_time_ms": speed_ref_ms,
            "speed_ref_margin": speed_margin,
            "kkt_final": 8.8e-7,
        },
        "decision": {"accepted": accepted, "reason": reason},
    }
    if roofline_pct is not None:
        record["tier3"] = {
            "passed": accepted,
            "per_shape": [
                {
                    "roofline_pct_med": roofline_pct,
                }
            ],
            "rank_summary": {
                "median_wall_time_ms": 20.0,
                "median_roofline_pct": roofline_pct,
            },
        }
    return record


def test_fitness_from_record_diagnoses_tier2_near_miss():
    fitness = fitness_from_record(_record("tier_failed:2_speed"))

    assert fitness["performance_class"] == "tier2_speed_near_miss"
    assert fitness["tier2_speed_ratio"] == 0.9808
    assert fitness["tier2_speed_target_ratio"] == 1.0111
    assert fitness["bottleneck_hint"] == "full_convergence_speed_limited"
    assert "near miss" in fitness["recommendation"]


def test_fitness_from_record_uses_roofline_bottleneck_hint():
    low = fitness_from_record(
        _record("keep:tier3_passed", accepted=True, roofline_pct=4.6)
    )
    high = fitness_from_record(
        _record("tier_failed:3", roofline_pct=82.0)
    )

    assert low["bottleneck_hint"] == "overhead_or_algorithm_limited"
    assert high["bottleneck_hint"] == "bandwidth_or_dtype_limited"


def test_write_and_summarize_fitness_report(tmp_path: Path):
    path = tmp_path / "synth_state" / "fitness.json"

    report = write_fitness_report(
        [
            _record("tier_failed:2_speed"),
            _record("discard:duplicate_source", tier2_wall_ms=0.0, speed_ref_ms=0.0),
            _record("keep:tier3_passed", accepted=True, roofline_pct=4.6),
        ],
        path,
    )

    loaded = json.loads(path.read_text())
    summary = summarize_fitness_report(
        report,
        "lasso/fista/apple_silicon/fp32",
    )

    assert loaded["version"] == 1
    assert summary["n_records"] == 3
    assert summary["performance_classes"]["tier2_speed_near_miss"] == 1
    assert summary["bottleneck_hints"]["no_behavior_change"] == 1
    assert summary["near_misses"][0]["tier2_speed_ratio"] == 0.9808
    assert summary["overhead_limited"][0]["bottleneck_hint"] == (
        "overhead_or_algorithm_limited"
    )


def test_build_fitness_report_global_fallback():
    report = build_fitness_report([_record("tier_failed:2_speed")])
    summary = summarize_fitness_report(report, "missing/slot")

    assert summary["n_records"] == 1
    assert summary["near_misses"][0]["edit_type"] == "vectorize"
