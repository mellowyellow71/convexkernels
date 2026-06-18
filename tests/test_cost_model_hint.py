"""Cost-model hint reaches the research state and the proposer prompt."""

from __future__ import annotations

import numpy as np

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.proposers.openai import _format_prompt
from convexkernels.synth.research_state import build_research_state
from convexkernels.synth.roofline import roofline_hint


def _hint():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4000, 1500))
    b = rng.standard_normal(4000)
    return roofline_hint(Lasso(A, b, 1.0))


def test_research_state_includes_cost_model_when_given():
    st = build_research_state(
        lineage_rows=[], baseline_times={"sklearn": 0.01},
        champion=None, kkt_tol=1e-6, cost_model=_hint(),
    )
    assert "hardware_cost_model" in st
    assert st["hardware_cost_model"]["shape"]["regime"] == "tall"


def test_research_state_omits_cost_model_when_absent():
    st = build_research_state(
        lineage_rows=[], baseline_times={}, champion=None, kkt_tol=1e-6,
    )
    assert "hardware_cost_model" not in st


def test_prompt_surfaces_cost_model_section():
    state = build_research_state(
        lineage_rows=[], baseline_times={"sklearn": 0.01},
        champion=None, kkt_tol=1e-6, cost_model=_hint(),
    )
    ctx = {
        "slot": {"problem_family": "lasso", "algorithm": "open",
                 "hardware": "apple_silicon", "dtype": "open"},
        "kkt_tol": 1e-6, "margin": 0.97, "program_md": "",
        "current_source": "", "current_source_path": None,
        "current_score": {
            "reached_target": False, "kkt_final": 1.0, "time_to_kkt_s": float("inf"),
            "total_time_s": float("inf"), "setup_s": 0.0, "n_reps": 0, "trajectory": [],
        },
        "research_state": state,
    }
    prompt = _format_prompt(ctx)
    assert "hardware cost model" in prompt
    assert "gram_symmetric" in prompt
    assert "lever:" in prompt
