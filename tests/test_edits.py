"""Tests for edit outcome priors."""

from __future__ import annotations

import json
from pathlib import Path

from convexkernels.synth.edits import (
    build_edit_priors,
    structured_payload_signature,
    summarize_edit_priors,
    write_edit_priors,
)
from convexkernels.synth.lineage import Slot


def _record(
    edit_type: str,
    reason: str,
    *,
    accepted: bool = False,
    payload: dict | None = None,
) -> dict:
    return {
        "slot": {
            "problem_family": "lasso",
            "algorithm": "fista",
            "hardware": "apple_silicon",
            "dtype": "fp32",
        },
        "edit": {"type": edit_type, "rationale": "test", "payload": payload or {}},
        "tier1": {"passed": True, "wall_time_ms": 10.0},
        "tier2": {
            "passed": accepted,
            "wall_time_ms": 20.0,
            "speed_ref_wall_time_ms": 19.0,
        },
        "decision": {"accepted": accepted, "reason": reason},
    }


def test_build_edit_priors_counts_outcomes():
    records = [
        _record("fuse_op", "keep:tier2_passed", accepted=True),
        _record("dtype_swap", "invalid:kkt_above_tier1_tol"),
        _record("dtype_swap", "tier_failed:2_speed"),
    ]

    priors = build_edit_priors(records)

    assert priors["global"]["fuse_op"]["n_accepted"] == 1
    assert priors["global"]["fuse_op"]["accept_rate"] == 1.0
    assert priors["global"]["fuse_op"]["median_tier1_wall_ms"] == 10.0
    assert priors["global"]["fuse_op"]["median_tier2_wall_ms"] == 20.0
    assert priors["global"]["dtype_swap"]["n_proposed"] == 2
    assert priors["global"]["dtype_swap"]["n_tier2_speed_failed"] == 1
    assert priors["global"]["dtype_swap"]["n_invalid"] == 1


def test_write_and_summarize_edit_priors(tmp_path: Path):
    slot = Slot("lasso", "fista", "apple_silicon", "fp32")
    path = tmp_path / "synth_state" / "edits.json"

    priors = write_edit_priors(
        [
            _record("fuse_op", "keep:tier2_passed", accepted=True),
            _record(
                "vectorize",
                "tier_failed:2_speed",
                payload={
                    "items_per_thread": 4,
                    "threadgroup_size": 128,
                    "kernel_name_suffix": "vec4_tg128",
                },
            ),
            _record(
                "vectorize",
                "tier_failed:2_speed",
                payload={
                    "items_per_thread": 4,
                    "threadgroup_size": 128,
                    "kernel_name_suffix": "different_suffix",
                },
            ),
        ],
        path,
    )
    loaded = json.loads(path.read_text())
    summary = summarize_edit_priors(priors, slot, min_attempts=1)

    assert loaded["version"] == 2
    assert summary["top_accepted"][0]["edit_type"] == "fuse_op"
    assert summary["avoid_until_changed"][0]["edit_type"] == "vectorize"
    assert summary["avoid_structured_payloads"][0]["payload"] == {
        "items_per_thread": 4,
        "threadgroup_size": 128,
    }
    assert summary["avoid_structured_payloads"][0]["n"] == 2
    assert summary["avoid_structured_payloads"][0]["median_tier2_ref_ms"] == 19.0
    assert summary["near_miss_structured_payloads"][0]["tier2_speed_ratio"] == 1.0526


def test_structured_payload_signature_ignores_suffix_and_empty_fields():
    assert structured_payload_signature(
        {
            "items_per_thread": 2,
            "remove_bounds_check": False,
            "kernel_name_suffix": "vec2_a",
        }
    ) == structured_payload_signature(
        {
            "items_per_thread": 2,
            "kernel_name_suffix": "vec2_b",
        }
    )
