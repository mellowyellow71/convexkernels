"""Test the OpenAI proposer plumbing without making real API calls.

Uses an injected fake client; verifies prompt formatting, structured-response
parsing, validation, and Edit construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from convexkernels.synth.lineage import Slot
from convexkernels.synth.proposers.openai import OpenAIProposer


_MOCK_SOURCE = """import mlx.core as mx

def init_state(problem):
    pass

def fista_step(state, problem, t):
    pass
"""


@dataclass
class FakeResponse:
    output_text: str


@pytest.fixture
def fake_client():
    client = MagicMock()
    client.responses.create.return_value = FakeResponse(
        output_text=json.dumps(
            {
                "edit_type": "tile_change",
                "rationale": (
                    "Increased threadgroup from 256 to 512 to amortize launch overhead."
                ),
                "source": _MOCK_SOURCE,
                "structured_edit": {
                    "threadgroup_size": None,
                    "items_per_thread": None,
                    "remove_bounds_check": None,
                    "branchless_soft_threshold": None,
                    "gradient_strategy": None,
                    "dtype_strategy": None,
                    "kernel_name_suffix": None,
                },
            }
        )
    )
    return client


def test_openai_proposer_parses_structured_response(fake_client):
    proposer = OpenAIProposer(client=fake_client)
    slot = Slot("lasso", "fista", "m3_pro", "fp32")
    edit = proposer.propose(slot, parent_id=None, history=[])

    assert edit.type == "tile_change"
    assert "256 to 512" in edit.rationale
    assert "import mlx.core" in edit.payload["full_source"]
    assert edit.payload["full_source"].rstrip().endswith("pass")
    assert edit.proposer_role == "impl"
    assert edit.proposer_model == "gpt-5.5"
    assert edit.source == "openai_responses"


def test_openai_proposer_includes_champion_source_in_prompt(fake_client):
    proposer = OpenAIProposer(client=fake_client)
    proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])

    call_args = fake_client.responses.create.call_args
    prompt = call_args.kwargs["input"][0]["content"]
    assert "fista_step" in prompt
    assert "init_state" in prompt
    assert "{{champion_source}}" not in prompt
    assert "{{recent_history}}" not in prompt
    assert "{{runtime_context}}" not in prompt


def test_openai_proposer_requests_structured_output(fake_client):
    proposer = OpenAIProposer(
        client=fake_client, reasoning_effort="high", api_timeout_s=123.0
    )
    proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])

    kwargs = fake_client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert kwargs["text"]["format"]["name"] == "kernel_edit"
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["timeout"] == 123.0
    assert "temperature" not in kwargs


def test_openai_proposer_raises_on_missing_source(fake_client):
    fake_client.responses.create.return_value = FakeResponse(
        output_text=json.dumps(
            {
                "edit_type": "tile_change",
                "rationale": "Forgot the source.",
                "source": "",
            }
        )
    )
    proposer = OpenAIProposer(client=fake_client)
    with pytest.raises(ValueError, match="non-empty `source` or a non-empty `structured_edit`"):
        proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])


def test_openai_proposer_raises_on_invalid_source(fake_client):
    fake_client.responses.create.return_value = FakeResponse(
        output_text=json.dumps(
            {
                "edit_type": "tile_change",
                "rationale": "Returned an incomplete module.",
                "source": "def init_state(problem):\n    pass\n",
            }
        )
    )
    proposer = OpenAIProposer(client=fake_client)
    with pytest.raises(ValueError, match="missing required `fista_step`"):
        proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])


def test_openai_proposer_parses_structured_edit(fake_client):
    fake_client.responses.create.return_value = FakeResponse(
        output_text=json.dumps(
            {
                "edit_type": "tile_change",
                "rationale": "Try a smaller threadgroup for launch-bound shapes.",
                "source": "",
                "structured_edit": {
                    "threadgroup_size": 128,
                    "items_per_thread": 2,
                    "remove_bounds_check": False,
                    "branchless_soft_threshold": None,
                    "gradient_strategy": "gram",
                    "dtype_strategy": "fp32",
                    "kernel_name_suffix": "tg128",
                },
            }
        )
    )
    proposer = OpenAIProposer(client=fake_client)
    edit = proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])

    assert edit.payload == {
        "threadgroup_size": 128,
        "items_per_thread": 2,
        "remove_bounds_check": False,
        "gradient_strategy": "gram",
        "dtype_strategy": "fp32",
        "kernel_name_suffix": "tg128",
    }


def test_openai_proposer_history_formatting(fake_client):
    proposer = OpenAIProposer(client=fake_client)
    history = [
        {
            "edit": {"type": "tile_change", "rationale": "tried bigger tile"},
            "tier1": {"wall_time_ms": 12.3},
            "decision": {"reason": "tier1_passed"},
        },
    ]
    proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=history)
    prompt = fake_client.responses.create.call_args.kwargs["input"][0]["content"]
    assert "tile_change" in prompt
    assert "tried bigger tile" in prompt


def test_openai_proposer_runtime_context(fake_client):
    proposer = OpenAIProposer(client=fake_client)
    proposer.set_runtime_context(
        {
            "baseline_wall_ms": 5.5,
            "target_wall_ms": 5.225,
            "tier1_reps": 3,
            "tier1_gate_role": "tier2_escalation",
            "tier2_baseline_wall_ms": 20.0,
            "tier2_target_wall_ms": 19.4,
            "tier2_reps": 2,
            "confirm_tier2_speed": True,
            "algorithm_variant": "restart",
            "current_kernel": {
                "gradient_strategy": "gram",
                "dtype_strategy": "fp32",
                "focus": (
                    "Current champion uses Gram-precomputed gradients; "
                    "prioritize G @ y - c."
                ),
            },
            "shape": {
                "name": "wide_small",
                "m": 500,
                "n": 2000,
                "regime": "small_dense_launch_overhead_bound",
                "guidance": "avoid cosmetic rewrites",
            },
            "history_summary": {
                "n_attempts": 3,
                "decision_counts": {"discard:not_faster_than_baseline": 2},
                "edit_outcomes": {"other -> discard:not_faster_than_baseline": 2},
                "fastest_rejected": {
                    "edit_type": "other",
                    "reason": "tier_failed:2_speed",
                    "tier1_wall_time_ms": 5.1,
                    "tier2_wall_time_ms": 21.0,
                },
            },
            "fitness_summary": {
                "n_records": 4,
                "performance_classes": {"tier2_speed_near_miss": 2},
                "bottleneck_hints": {"overhead_or_algorithm_limited": 1},
                "near_misses": [
                    {
                        "edit_type": "tile_change",
                        "tier2_speed_ratio": 0.9788,
                        "tier2_speed_target_ratio": 1.0091,
                        "recommendation": "near miss: build around this only with a larger lever",
                    },
                ],
                "overhead_limited": [
                    {
                        "edit_type": "fuse_op",
                        "tier3_median_roofline_pct": 4.59,
                        "recommendation": "try reducing launches/setup or iteration count",
                    },
                ],
                "high_noise": [
                    {
                        "edit_type": "vectorize",
                        "tier1_timing_cv": 0.12,
                    },
                ],
            },
            "edit_priors_summary": {
                "top_accepted": [
                    {"edit_type": "fuse_op", "n": 3, "accepted": 1},
                ],
                "avoid_until_changed": [
                    {"edit_type": "dtype_swap", "n": 4, "accepted": 0},
                ],
                "top_structured_payloads": [
                    {
                        "edit_type": "fuse_op",
                        "payload": {"branchless_soft_threshold": True},
                        "n": 2,
                        "accepted": 1,
                    },
                ],
                "near_miss_structured_payloads": [
                    {
                        "edit_type": "tile_change",
                        "payload": {
                            "threadgroup_size": 512,
                            "branchless_soft_threshold": True,
                        },
                        "tier2_speed_ratio": 0.979,
                    },
                ],
                "avoid_structured_payloads": [
                    {
                        "edit_type": "vectorize",
                        "payload": {
                            "items_per_thread": 2,
                            "threadgroup_size": 128,
                        },
                        "n": 1,
                        "accepted": 0,
                    },
                ],
            },
        }
    )
    proposer.propose(Slot("lasso", "fista", "m3_pro", "fp32"), parent_id=None, history=[])
    prompt = fake_client.responses.create.call_args.kwargs["input"][0]["content"]
    assert "Tier-1 median baseline (3 reps): 5.500 ms" in prompt
    assert "Tier-1 escalation target: pass KKT and run below 5.225 ms" in prompt
    assert "Tier-2 median convergence baseline (2 reps): 20.000 ms" in prompt
    assert "Tier-2 promotion target: converge and run below 19.400 ms" in prompt
    assert "paired remeasurement" in prompt
    assert "Active shape: wide_small (m=500, n=2000)" in prompt
    assert "Host algorithm variant: FISTA restart" in prompt
    assert "Current champion kernel strategy: gradient_strategy=gram" in prompt
    assert "Current champion search focus" in prompt
    assert "G @ y - c" in prompt
    assert "Current-run outcome counts" in prompt
    assert "Fastest rejected Tier-1 pass so far" in prompt
    assert "Structured fitness class counts" in prompt
    assert "Structured fitness bottleneck hints" in prompt
    assert "Fitness near misses with diagnosis" in prompt
    assert "Low-roofline overhead/algorithm-limited examples" in prompt
    assert "High timing-noise examples" in prompt
    assert "Historical edit types with accepted wins" in prompt
    assert "Historical edit types to avoid" in prompt
    assert "Historical structured payloads with accepted wins" in prompt
    assert "Historical structured payload near misses" in prompt
    assert "Historical structured payloads to avoid as exact repeats" in prompt
