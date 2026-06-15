"""Checkpoint store + experiment-tree + resume/branch tests (Linux, native)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.checkpoints import CheckpointStore
from convexkernels.synth.lineage import Slot
from convexkernels.synth.loop import run_synth_loop
from convexkernels.synth.proposers.stub import StubProposer


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

_BAD_SOLVE = '''
import numpy as np
def solve(problem, recorder, *, kkt_tol, max_time_s):
    x = np.zeros(problem.n)
    recorder.record(x)
    return x
'''


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 60))
    x_true = rng.standard_normal(60) * (rng.random(60) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(120)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def _kwargs(tmp_path: Path, proposer, **over):
    base = dict(
        proposer=proposer,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=Slot("lasso", "open", "linux_x86", "open"),
        state_root=tmp_path,
        problem_backend="native",
        kkt_tol=1e-6,
        max_time_s=20.0,
        reps=1,
        warmup_runs=0,
        compute_baselines=False,
        verbose=False,
    )
    base.update(over)
    return base


# ---------- unit: store ----------

def test_store_roundtrip_and_tree(tmp_path: Path):
    store = CheckpointStore(tmp_path)
    sc_a = {"total_time_s": 1.0, "time_to_kkt_s": 1.0, "kkt_final": 1e-8}
    sc_b = {"total_time_s": 0.5, "time_to_kkt_s": 0.5, "kkt_final": 1e-8}
    store.save(id="a", parent_id=None, source="x=1", score=sc_a,
               trajectory=[[0.0, 1.0]], algorithm_tag="seed", problem_hash="h")
    store.save(id="b", parent_id="a", source="x=2", score=sc_b,
               trajectory=[], algorithm_tag="fista", problem_hash="h")

    assert store.get("a").parent_id is None
    assert store.get("b").parent_id == "a"
    assert {c.id for c in store.all()} == {"a", "b"}
    assert store.best("h").id == "b"          # min total_time_s
    assert store.get("b").source() == "x=2"
    assert store.get("missing") is None


# ---------- loop: accept creates a child node ----------

def test_loop_accepts_and_branches(tmp_path: Path, tiny_lasso: Lasso):
    proposer = StubProposer([("faster", _GOOD_SOLVE)])
    rows = run_synth_loop(problem=tiny_lasso, n_proposals=1,
                          **_kwargs(tmp_path, proposer, margin=2.0))
    assert rows[0].decision["accepted"] is True

    store = CheckpointStore(tmp_path)
    seeds = [c for c in store.all() if c.algorithm_tag == "seed"]
    children = [c for c in store.all() if c.algorithm_tag != "seed"]
    assert len(seeds) == 1
    assert len(children) == 1
    assert children[0].parent_id == seeds[0].id
    assert rows[0].parent_id == seeds[0].id  # experiment linked to seed node


# ---------- resume/branch from an arbitrary node ----------

def test_resume_from_branches_from_node(tmp_path: Path, tiny_lasso: Lasso):
    run_synth_loop(problem=tiny_lasso, n_proposals=1,
                   **_kwargs(tmp_path, StubProposer([("faster", _GOOD_SOLVE)]), margin=2.0))
    store = CheckpointStore(tmp_path)
    node = [c for c in store.all() if c.algorithm_tag != "seed"][0]

    rows = run_synth_loop(problem=tiny_lasso, n_proposals=1, resume_from=node.id,
                          **_kwargs(tmp_path, StubProposer([("noop", _BAD_SOLVE)])))
    # Resumed run does not create a new seed node; the experiment branches off
    # the resumed checkpoint.
    assert rows[0].parent_id == node.id
    seeds = [c for c in store.all() if c.algorithm_tag == "seed"]
    assert len(seeds) == 1  # still just the one from the first run
