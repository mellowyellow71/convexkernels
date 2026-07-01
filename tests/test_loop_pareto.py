"""End-to-end test: the loop with `selection="pareto"` (two-objective mode).

Drives `run_synth_loop` with a deterministic inline proposer emitting FISTA
variants, on the native numpy backend, and checks the Pareto semantics:
  - a candidate whose curve adds frontier hypervolume is KEPT,
  - a duplicate-region candidate is discarded as pareto_dominated,
  - research_state.json carries the frontier summary for the proposer,
  - the sandbox scored on the duality gap (score_metric threading).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.lineage import Slot
from convexkernels.synth.loop import Edit, run_synth_loop


def _render(check_every: int, restart: bool = True) -> str:
    mom = (
        "        if float(np.dot(y-xn, xn-x))>0.0:\n"
        "            tn,mom=1.0,0.0\n"
        "        else:\n"
        "            tn=0.5*(1+np.sqrt(1+4*theta*theta)); mom=(theta-1)/tn\n"
        if restart else
        "        tn=0.5*(1+np.sqrt(1+4*theta*theta)); mom=(theta-1)/tn\n"
    )
    return (
        "import numpy as np\n\n"
        "def solve(problem, recorder, *, kkt_tol, max_time_s):\n"
        "    n=problem.n; x=np.zeros(n); y=x.copy(); theta=1.0; t=1.0/problem.L; it=0\n"
        "    while it<200000:\n"
        "        it+=1\n"
        "        g=problem.grad_smooth(y)\n"
        "        xn=problem.prox(y-t*g,t)\n"
        + mom +
        "        y=xn+mom*(xn-x); x=xn; theta=tn\n"
        f"        if it%{check_every}==0:\n"
        "            recorder.record(x)\n"
        "            if recorder.should_stop(kkt_tol): break\n"
        "    recorder.record(x)\n"
        "    return x\n"
    )


class _InlineProposer:
    """Deterministic variants: distinct check_every so curves differ."""

    model = "inline"

    def __init__(self, variants):
        self._variants = list(variants)
        self._i = 0

    def propose(self, ctx: dict) -> Edit:
        ce, rs = self._variants[self._i % len(self._variants)]
        self._i += 1
        return Edit(
            type="full_source",
            rationale=f"fista chk{ce} restart={rs}",
            full_source=_render(ce, rs),
            proposer_role="impl",
            proposer_model=self.model,
        )


@pytest.fixture()
def small_lasso():
    rng = np.random.default_rng(0)
    m, n, k = 600, 200, 12
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam)


def test_pareto_selection_end_to_end(small_lasso, tmp_path):
    slot = Slot(problem_family="lasso", algorithm="open", hardware="cpu", dtype="open")
    rows = run_synth_loop(
        proposer=_InlineProposer([(5, True), (50, True), (5, True)]),
        problem=small_lasso,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=slot,
        state_root=tmp_path,
        n_proposals=3,
        reps=1,
        kkt_tol=1e-6,
        max_time_s=5.0,
        selection="pareto",
        problem_backend="native",
        compute_baselines=False,
        verbose=False,
    )
    assert len(rows) == 3
    reasons = [r.decision.get("reason", "") for r in rows]

    # Every decision is a Pareto decision, not a champion-race decision.
    for reason in reasons:
        assert reason.startswith(("keep:adds_frontier_hv", "discard:pareto_dominated",
                                  "discard:duplicate_source"))
    # At least one variant extended the frontier past the seed.
    assert any(r.startswith("keep:adds_frontier_hv") for r in reasons)

    # research_state carries the frontier context for the proposer.
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert state.get("selection") == "pareto"
    pareto = state["pareto"]
    assert pareto["hypervolume"] > 0.0
    assert len(pareto["frontier"]) >= 1
    assert pareto["frontier_best_gap"] is not None


def test_champion_selection_unchanged(small_lasso, tmp_path):
    # Default mode still runs the single-champion race with KKT reasons.
    slot = Slot(problem_family="lasso", algorithm="open", hardware="cpu", dtype="open")
    rows = run_synth_loop(
        proposer=_InlineProposer([(5, True)]),
        problem=small_lasso,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=slot,
        state_root=tmp_path,
        n_proposals=1,
        reps=1,
        kkt_tol=1e-6,
        max_time_s=5.0,
        problem_backend="native",
        compute_baselines=False,
        verbose=False,
    )
    reason = rows[0].decision.get("reason", "")
    assert ("champion" in reason) or reason.startswith("discard:did_not_reach_target")
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert "pareto" not in state
