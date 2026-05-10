"""PDHG correctness tests on basis pursuit.

  min ||x||_1  s.t.  A x = b

PDHG solves this via the saddle-point form. Test vs CVXPY-CLARABEL on a
synthetic problem with known sparse ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from convexkernels.algorithms.pdhg import pdhg
from convexkernels.bench.eq_qp_shapes import (
    DEFAULT_BP_SHAPES,
    make_synthetic_basis_pursuit,
)
from convexkernels.frontend.basis_pursuit import BasisPursuit


def _cvxpy_basis_pursuit(prob: BasisPursuit, tol: float = 1e-10):
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(prob.n)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [prob.A @ x == prob.b]
    cp.Problem(objective, constraints).solve(
        solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
    )
    return np.asarray(x.value)


def test_pdhg_basis_pursuit_matches_cvxpy_small() -> None:
    prob = make_synthetic_basis_pursuit(DEFAULT_BP_SHAPES[0])
    x_cvx = _cvxpy_basis_pursuit(prob)
    res = pdhg(prob, max_iters=20000, tol=1e-7, variant="basic")
    assert res.converged, f"PDHG did not converge: gap={res.gap_final:.3e}"
    # L1 norm match: PDHG and CVXPY should land at the same primal objective.
    l1_pdhg = prob.primal_objective(res.x)
    l1_cvx = prob.primal_objective(x_cvx)
    assert abs(l1_pdhg - l1_cvx) / max(abs(l1_cvx), 1.0) < 1e-3, (
        f"L1 mismatch: pdhg={l1_pdhg:.6f} cvx={l1_cvx:.6f}"
    )
    # Constraint feasibility under tol.
    assert prob.primal_residual(res.x) < 1e-4


def test_pdhg_bp_constraint_feasibility() -> None:
    """At convergence, A x ≈ b within feasibility tolerance."""
    prob = make_synthetic_basis_pursuit(DEFAULT_BP_SHAPES[0])
    res = pdhg(prob, max_iters=20000, tol=1e-7)
    assert res.converged
    assert prob.primal_residual(res.x) < 1e-4


def test_pdhg_bp_gap_decreases() -> None:
    prob = make_synthetic_basis_pursuit(DEFAULT_BP_SHAPES[0])
    res = pdhg(prob, max_iters=10000, tol=1e-9, record_history=True)
    gaps = res.history["gap"]
    assert len(gaps) > 0
    assert gaps[-1] < gaps[0]
    assert res.gap_final < gaps[0]


def test_bp_recovers_sparse_signal() -> None:
    """Standard BP recovery: under RIP, x_true is the L1-min solution."""
    rng = np.random.default_rng(0)
    m, n = 80, 200
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    support = rng.choice(n, size=10, replace=False)
    x_true[support] = rng.standard_normal(10)
    b = A @ x_true
    prob = BasisPursuit(A, b)

    res = pdhg(prob, max_iters=30000, tol=1e-8)
    assert res.converged
    # Recovered x should be close to x_true on the support.
    rel_err = float(np.linalg.norm(res.x - x_true) / max(np.linalg.norm(x_true), 1.0))
    assert rel_err < 1e-2, f"recovery error {rel_err:.3e} too large"
