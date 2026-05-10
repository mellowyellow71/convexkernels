"""Tests for the deterministic structured proposer."""

from __future__ import annotations

from convexkernels.synth.lineage import Slot
from convexkernels.synth.proposers.structured import StructuredGridProposer


def test_structured_grid_proposer_emits_structured_payloads():
    proposer = StructuredGridProposer()
    edit = proposer.propose(Slot("lasso", "fista", "hw", "fp32"), None, [])

    assert edit.type == "algo_variant"
    assert edit.payload == {
        "gradient_strategy": "gram",
        "dtype_strategy": "fp32",
        "kernel_name_suffix": "gram_fp32",
    }
    assert edit.source == "structured_grid"


def test_structured_grid_proposer_skips_seen_payloads():
    proposer = StructuredGridProposer(
        payloads=(
            {"threadgroup_size": 128, "kernel_name_suffix": "first"},
            {"items_per_thread": 2, "kernel_name_suffix": "second"},
        )
    )
    history = [
        {
            "edit": {
                "payload": {
                    "threadgroup_size": 128,
                    "kernel_name_suffix": "old_suffix",
                }
            }
        }
    ]

    edit = proposer.propose(Slot("lasso", "fista", "hw", "fp32"), None, history)

    assert edit.type == "vectorize"
    assert edit.payload["items_per_thread"] == 2


def test_structured_grid_proposer_adapts_defaults_for_nonnegative_lasso():
    proposer = StructuredGridProposer()
    edit = proposer.propose(
        Slot("nonnegative_lasso", "fista", "hw", "fp32"),
        None,
        [],
    )

    assert edit.type == "hoist_to_threadgroup"
    assert edit.payload == {
        "remove_bounds_check": True,
        "kernel_name_suffix": "nobounds",
    }
