"""Tests for cross-slot edit transfer seeds."""

from __future__ import annotations

from convexkernels.synth.lineage import Slot, seed_from_neighbors


def _record(
    *,
    problem_family: str,
    accepted: bool = True,
    payload: dict | None = None,
    record_id: str = "rec",
    tier2_wall_ms: float = 90.0,
    tier2_ref_ms: float = 100.0,
) -> dict:
    return {
        "id": record_id,
        "slot": {
            "problem_family": problem_family,
            "algorithm": "fista",
            "hardware": "apple_silicon",
            "dtype": "fp32",
        },
        "edit": {
            "type": "tile_change",
            "payload": payload or {"threadgroup_size": 128},
            "rationale": "source edit",
            "proposer_role": "impl",
            "proposer_model": "test",
            "source": "structured_grid",
        },
        "tier1": {"passed": True, "wall_time_ms": 10.0},
        "tier2": {
            "passed": accepted,
            "wall_time_ms": tier2_wall_ms,
            "speed_ref_wall_time_ms": tier2_ref_ms,
        },
        "decision": {"accepted": accepted, "reason": "keep:tier2_passed"},
        "source": {"hash": f"sha256:{record_id}"},
    }


def test_seed_from_neighbors_returns_ranked_transfer_edits():
    target = Slot("nonnegative_lasso", "fista", "apple_silicon", "fp32")
    records = [
        _record(
            problem_family="lasso",
            payload={"threadgroup_size": 512, "kernel_name_suffix": "slow"},
            record_id="slow",
            tier2_wall_ms=98.0,
            tier2_ref_ms=100.0,
        ),
        _record(
            problem_family="lasso",
            payload={"threadgroup_size": 128, "kernel_name_suffix": "fast"},
            record_id="fast",
            tier2_wall_ms=88.0,
            tier2_ref_ms=100.0,
        ),
    ]

    seeds = seed_from_neighbors(records, target, k=1)

    assert len(seeds) == 1
    assert seeds[0].payload["threadgroup_size"] == 128
    assert seeds[0].source == "transfer:lasso/fista/apple_silicon/fp32"


def test_seed_from_neighbors_skips_full_source_across_problem_family():
    target = Slot("nonnegative_lasso", "fista", "apple_silicon", "fp32")
    records = [
        _record(
            problem_family="lasso",
            payload={"full_source": "from lasso_only import fista_step\n"},
        )
    ]

    assert seed_from_neighbors(records, target, k=3) == []


def test_seed_from_neighbors_skips_payload_already_tried_in_target():
    target = Slot("nonnegative_lasso", "fista", "apple_silicon", "fp32")
    records = [
        _record(
            problem_family="lasso",
            payload={"threadgroup_size": 128, "kernel_name_suffix": "source"},
        ),
        _record(
            problem_family="nonnegative_lasso",
            accepted=False,
            payload={"threadgroup_size": 128, "kernel_name_suffix": "target"},
        ),
    ]

    assert seed_from_neighbors(records, target, k=3) == []


def test_seed_from_neighbors_adapts_lasso_branchless_for_nonnegative_target():
    target = Slot("nonnegative_lasso", "fista", "apple_silicon", "fp32")
    records = [
        _record(
            problem_family="lasso",
            payload={
                "branchless_soft_threshold": True,
                "kernel_name_suffix": "branchless",
            },
            tier2_wall_ms=80.0,
            tier2_ref_ms=100.0,
        ),
        _record(
            problem_family="lasso",
            payload={
                "threadgroup_size": 128,
                "branchless_soft_threshold": True,
                "kernel_name_suffix": "tg128_branchless",
            },
            tier2_wall_ms=90.0,
            tier2_ref_ms=100.0,
        ),
    ]

    seeds = seed_from_neighbors(records, target, k=3)

    assert len(seeds) == 1
    assert seeds[0].payload == {
        "threadgroup_size": 128,
        "kernel_name_suffix": "tg128_branchless",
    }
