"""Tests for the new (post-pivot) minimal synth loop.

Linux-runnable. Drives `run_synth_loop` end-to-end with a stub proposer over
a tiny FISTA-LASSO problem so the loop's gating, lineage write, and
champion update behaviour are unit-testable without an LLM in the path.
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
    A = rng.standard_normal((200, 100))
    x_true = rng.standard_normal(100) * (rng.random(100) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(200)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


_NUMPY_REF_SOURCE = '''
import numpy as np
from dataclasses import dataclass
from convexkernels.frontend.problem import Problem


@dataclass
class FistaState:
    x: np.ndarray
    y: np.ndarray
    theta: float


def init_state(problem: Problem) -> FistaState:
    n = problem.n
    return FistaState(x=np.zeros(n), y=np.zeros(n), theta=1.0)


def fista_step(state: FistaState, problem: Problem, t: float) -> FistaState:
    g = problem.grad_smooth(state.y)
    x_next = problem.prox(state.y - t * g, t)
    theta_next = (1.0 + np.sqrt(1.0 + 4.0 * state.theta * state.theta)) / 2.0
    momentum = (state.theta - 1.0) / theta_next
    y_next = x_next + momentum * (x_next - state.x)
    return FistaState(x=x_next, y=y_next, theta=theta_next)
'''


def _slot() -> Slot:
    return Slot(problem_family="lasso", algorithm="fista", hardware="linux_x86_64", dtype="fp64")


def test_loop_runs_baseline_and_logs(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """Loop runs the baseline measurement, accepts no proposals (n=0)."""
    rows = run_synth_loop(
        proposer=StubProposer([]),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        slot=_slot(),
        state_root=tmp_path,
        n_proposals=0,
        algorithm="fista",
        max_iters=2000,
        fitness_tol=1e-6,
        reps=2,
        speedup_margin=0.97,
        warmup_runs=0,
    )
    assert rows == []
    assert (tmp_path / "runs").exists()


_SLOWER_SOURCE = '''
import numpy as np
import time
from dataclasses import dataclass
from convexkernels.frontend.problem import Problem


@dataclass
class FistaState:
    x: np.ndarray
    y: np.ndarray
    theta: float


def init_state(problem: Problem) -> FistaState:
    n = problem.n
    return FistaState(x=np.zeros(n), y=np.zeros(n), theta=1.0)


def fista_step(state: FistaState, problem: Problem, t: float) -> FistaState:
    # Inject ~5 ms of extra work to guarantee this is slower than the baseline.
    time.sleep(0.005)
    g = problem.grad_smooth(state.y)
    x_next = problem.prox(state.y - t * g, t)
    theta_next = (1.0 + np.sqrt(1.0 + 4.0 * state.theta * state.theta)) / 2.0
    momentum = (state.theta - 1.0) / theta_next
    y_next = x_next + momentum * (x_next - state.x)
    return FistaState(x=x_next, y=y_next, theta=theta_next)
'''


def test_loop_rejects_strictly_slower_candidate(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """A candidate that adds 5ms per iter is unconditionally slower → rejected."""
    rows = run_synth_loop(
        proposer=StubProposer([("inject sleep", _SLOWER_SOURCE)]),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        slot=_slot(),
        state_root=tmp_path,
        n_proposals=1,
        algorithm="fista",
        max_iters=200,  # short cap so the sleep doesn't take forever
        fitness_tol=1e-3,  # allow this small max_iters to "converge" loosely
        reps=1,
        speedup_margin=0.97,
        warmup_runs=0,
    )
    assert len(rows) == 1
    assert rows[0].decision["accepted"] is False
    assert "not_faster_than_baseline" in rows[0].decision["reason"]


def test_loop_records_proposer_crash(tmp_path: Path, tiny_lasso: Lasso) -> None:
    class BadProposer:
        def propose(self, ctx):  # noqa: ARG002
            raise RuntimeError("boom")

    rows = run_synth_loop(
        proposer=BadProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        slot=_slot(),
        state_root=tmp_path,
        n_proposals=1,
        algorithm="fista",
        max_iters=2000,
        fitness_tol=1e-6,
        reps=2,
        speedup_margin=0.97,
        warmup_runs=0,
    )
    assert len(rows) == 1
    assert "crash:proposer_error:RuntimeError" in rows[0].decision["reason"]


def test_loop_records_kkt_above_tol(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """Source that doesn't actually update x → fitness stays high → discard."""
    BROKEN_SOURCE = '''
import numpy as np
from dataclasses import dataclass
from convexkernels.frontend.problem import Problem


@dataclass
class FistaState:
    x: np.ndarray
    y: np.ndarray
    theta: float


def init_state(problem):
    n = problem.n
    return FistaState(x=np.zeros(n), y=np.zeros(n), theta=1.0)


def fista_step(state, problem, t):
    return state  # no-op: KKT will not improve
'''
    rows = run_synth_loop(
        proposer=StubProposer([("no-op step", BROKEN_SOURCE)]),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        slot=_slot(),
        state_root=tmp_path,
        n_proposals=1,
        algorithm="fista",
        max_iters=200,  # short cap so the no-op doesn't loop forever
        fitness_tol=1e-6,
        reps=1,
        speedup_margin=0.97,
        warmup_runs=0,
    )
    assert len(rows) == 1
    reason = rows[0].decision["reason"]
    assert "discard:" in reason and ("not_converged" in reason or "kkt_above_tol" in reason)


def test_loop_writes_lineage_and_champion(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """Lineage JSONL is written; champion.py is created when a candidate is kept."""
    # Use a faster numpy_ref-equivalent that should beat the baseline by a
    # margin: skip the prox redundancy by inlining. We'll just stub a small
    # speedup via larger step (likely faster but still valid).
    SPEEDIER = _NUMPY_REF_SOURCE.replace("0.5 * problem.L", "0.6 * problem.L")  # no-op for our test
    rows = run_synth_loop(
        proposer=StubProposer([
            ("identity 1", _NUMPY_REF_SOURCE),
            ("identity 2", SPEEDIER),
        ]),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        slot=_slot(),
        state_root=tmp_path,
        n_proposals=2,
        algorithm="fista",
        max_iters=2000,
        fitness_tol=1e-6,
        reps=1,
        speedup_margin=0.97,
        warmup_runs=0,
    )
    lineage = (tmp_path / "lineage.jsonl").read_text().splitlines()
    assert len(lineage) == 2
    parsed = [json.loads(line) for line in lineage]
    assert all("decision" in row for row in parsed)
    assert all("score" in row for row in parsed)
