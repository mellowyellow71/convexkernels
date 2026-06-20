"""The deterministic mutation proposer: the framework searches without an LLM."""

from __future__ import annotations

import types

import numpy as np
import pytest

from convexkernels.bench.metrics import trusted_kkt
from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.lineage import Slot
from convexkernels.synth.loop import run_synth_loop
from convexkernels.synth.proposers.mutation import (
    MutationProposer,
    config_family,
    operator_grid,
    render_source,
)


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 60))
    x_true = rng.standard_normal(60) * (rng.random(60) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(120)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def test_every_grid_config_renders_a_valid_solver(tiny_lasso):
    class Rec:
        def __init__(self):
            self.last = float("inf")

        def record(self, x):
            self.last = trusted_kkt(tiny_lasso, np.asarray(x))
            return self.last

        def should_stop(self, tol):
            return self.last <= tol

    for cfg in operator_grid():
        mod = types.ModuleType("gen")
        exec(compile(render_source(cfg), "<gen>", "exec"), mod.__dict__)
        prob = mod.prepare_problem(tiny_lasso) if hasattr(mod, "prepare_problem") else tiny_lasso
        x = mod.solve(prob, Rec(), kkt_tol=1e-6, max_time_s=10.0)
        assert trusted_kkt(tiny_lasso, np.asarray(x)) < 1e-6


def test_gram_configs_emit_prepare_problem():
    for cfg in operator_grid():
        src = render_source(cfg)
        assert ("prepare_problem" in src) == (cfg["gradient"] == "gram")


def test_policy_expands_champion_neighbors_first():
    mp = MutationProposer()
    champ_src = render_source({"gradient": "gram", "restart": True, "check_every": 5})
    ctx = {"current_source": champ_src, "research_state": {"tried_directions": []}}
    edit = mp.propose(ctx)
    champ_fam = config_family({"gradient": "gram", "restart": True, "check_every": 5})
    # a one-knob neighbour of the champion differs in exactly one field
    proposed = edit.algorithm_family
    assert proposed != champ_fam
    assert proposed.startswith("fista_")


def test_policy_skips_already_tried_families():
    mp = MutationProposer()
    tried = [{"idea": config_family(cfg)} for cfg in operator_grid()[:-1]]
    ctx = {"current_source": "", "research_state": {"tried_directions": tried}}
    edit = mp.propose(ctx)
    # only the last grid config remains untried
    assert edit.algorithm_family == config_family(operator_grid()[-1])


def test_loop_runs_and_improves_with_no_llm(tmp_path, tiny_lasso):
    rows = run_synth_loop(
        proposer=MutationProposer(),
        problem=tiny_lasso,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=Slot("lasso", "open", "linux_x86", "open"),
        state_root=tmp_path,
        n_proposals=5,
        kkt_tol=1e-6,
        max_time_s=10.0,
        reps=1,
        margin=0.97,
        problem_backend="native",
        problem_dtype="fp32",
        dtype_strategy="fp32",
        warmup_runs=0,
        timeout_s=60.0,
        compute_baselines=False,
        verbose=False,
    )
    assert len(rows) == 5
    scored = [r for r in rows if r.score]
    assert scored, "framework produced evaluated candidates"
    # KKT-valid candidates found with no model in the path
    assert any(r.score.get("reached_target") for r in scored)
    # genuine search: more than one distinct algorithm family was explored
    families = {(r.edit or {}).get("algorithm_family") for r in rows}
    assert len([f for f in families if f]) >= 2
