"""End-to-end tests for the autoresearch loop (new KKT-time-to-target API).

Linux-runnable via the native numpy `solve()` contract — no LLM, no MLX. Drives
`run_synth_loop` with a stub proposer so the loop's scoring, lineage/tree
write, checkpoint, and research-state behaviour are unit-testable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.lineage import Slot
from convexkernels.synth.loop import run_synth_loop
from convexkernels.synth.proposers.stub import StubProposer


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 60))
    x_true = rng.standard_normal(60) * (rng.random(60) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(120)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def _slot() -> Slot:
    return Slot("lasso", "open", "linux_x86", "open")


# A candidate that converges (a valid solve), used to exercise acceptance.
_GOOD_SOLVE = '''
import numpy as np
def solve(problem, recorder, *, kkt_tol, max_time_s):
    x = np.zeros(problem.n); y = x.copy(); theta = 1.0
    t = 1.0 / problem.L
    for it in range(1, 100000):
        g = problem.grad_smooth(y)
        xn = problem.prox(y - t*g, t)
        if float(np.dot(y - xn, xn - x)) > 0.0:
            tn, mom = 1.0, 0.0
        else:
            tn = 0.5*(1.0 + np.sqrt(1.0 + 4.0*theta*theta)); mom = (theta-1.0)/tn
        y = xn + mom*(xn - x); x = xn; theta = tn
        if it % 5 == 0:
            recorder.record(x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(x)
    return x
'''

# A candidate that never reaches the target (returns zeros).
_BAD_SOLVE = '''
import numpy as np
def solve(problem, recorder, *, kkt_tol, max_time_s):
    x = np.zeros(problem.n)
    recorder.record(x)
    return x
'''


def _common_kwargs(tmp_path: Path, proposer):
    return dict(
        proposer=proposer,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=_slot(),
        state_root=tmp_path,
        problem_backend="native",
        kkt_tol=1e-6,
        max_time_s=20.0,
        reps=1,
        warmup_runs=0,
        compute_baselines=False,
        verbose=False,
    )


def test_loop_logs_and_roots_tree(tmp_path: Path, tiny_lasso: Lasso):
    proposer = StubProposer([("noop reject", _BAD_SOLVE)])
    rows = run_synth_loop(problem=tiny_lasso, n_proposals=1,
                          **_common_kwargs(tmp_path, proposer))
    assert len(rows) == 1
    assert rows[0].decision["accepted"] is False
    assert "did_not_reach_target" in rows[0].decision["reason"]
    # seed checkpoint roots the tree; rejected experiment links to it
    ckpts = list((tmp_path / "checkpoints").iterdir())
    assert len(ckpts) == 1  # only the seed (rejected candidate is not a node)
    assert rows[0].parent_id == ckpts[0].name
    assert (tmp_path / "research_state.json").exists()
    assert (tmp_path / "lineage.jsonl").exists()


def test_loop_rejects_slower_valid_candidate(tmp_path: Path, tiny_lasso: Lasso):
    # The good solve is the same algorithm as the seed → not faster → reject.
    proposer = StubProposer([("same algo", _GOOD_SOLVE)])
    rows = run_synth_loop(problem=tiny_lasso, n_proposals=1,
                          **_common_kwargs(tmp_path, proposer))
    assert rows[0].score is not None
    assert rows[0].score["reached_target"] is True
    # valid but not faster than the seed champion
    assert not rows[0].decision["accepted"]
    assert "not_faster_than_champion" in rows[0].decision["reason"]


def test_loop_records_proposer_crash(tmp_path: Path, tiny_lasso: Lasso):
    proposer = StubProposer([])  # exhausted → raises → recorded as crash
    rows = run_synth_loop(problem=tiny_lasso, n_proposals=1,
                          **_common_kwargs(tmp_path, proposer))
    assert len(rows) == 1
    assert not rows[0].decision["accepted"]
    assert rows[0].decision["reason"].startswith("crash:proposer_error")


def test_research_state_tracks_baseline_bar(tmp_path: Path, tiny_lasso: Lasso):
    proposer = StubProposer([("noop", _BAD_SOLVE)])
    kwargs = _common_kwargs(tmp_path, proposer)
    kwargs["compute_baselines"] = True
    kwargs["baseline_solvers"] = ("CLARABEL", "SCS")
    run_synth_loop(problem=tiny_lasso, n_proposals=1, **kwargs)
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert "bar_to_beat" in state
    assert set(state["bar_to_beat"]["all_baselines_time_to_kkt_s"]) == {"CLARABEL", "SCS"}
    assert state["champion"] is not None
