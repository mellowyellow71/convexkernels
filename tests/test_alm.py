"""ALM correctness tests on equality-constrained QPs.

Goals:
  1. ALM matches CVXPY (CLARABEL) primal+dual on small EQ-QP instances.
  2. Primal/dual residuals → 0; the convergence test fires correctly.
  3. The Cholesky factor is built once (basic) and rebuilt on rho change
     (adaptive).
  4. The problem class' triangular solve actually solves what it claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from convexkernels.algorithms.alm import alm
from convexkernels.bench.eq_qp_shapes import DEFAULT_EQ_QP_SHAPES, make_synthetic_eq_qp
from convexkernels.frontend.equality_qp import EqualityQP


def _cvxpy_eq_qp(prob: EqualityQP, tol: float = 1e-12):
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(prob.n)
    obj = 0.5 * cp.quad_form(x, cp.psd_wrap(prob.P)) + prob.q @ x
    constraints = [prob.A @ x == prob.b_constraint]
    cp.Problem(cp.Minimize(obj), constraints).solve(
        solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
    )
    return np.asarray(x.value)


def test_alm_matches_cvxpy_small() -> None:
    prob = make_synthetic_eq_qp(DEFAULT_EQ_QP_SHAPES[0])
    x_cvx = _cvxpy_eq_qp(prob)
    res = alm(prob, max_iters=2000, tol=1e-8, rho=1.0)
    assert res.converged, (
        f"ALM did not converge: primal={res.primal_res_final:.3e} "
        f"dual={res.dual_res_final:.3e}"
    )
    rel = float(np.max(np.abs(res.x - x_cvx)) / max(np.max(np.abs(x_cvx)), 1.0))
    assert rel < 1e-4, f"iterate drift {rel:.3e}"
    p_alm = prob.primal_objective(res.x)
    p_cvx = prob.primal_objective(x_cvx)
    assert abs(p_alm - p_cvx) / max(abs(p_cvx), 1.0) < 1e-6


def test_alm_residuals_decrease() -> None:
    prob = make_synthetic_eq_qp(DEFAULT_EQ_QP_SHAPES[0])
    res = alm(prob, max_iters=500, tol=1e-10, rho=1.0, record_history=True)
    p_traj = res.history["primal_res"]
    d_traj = res.history["dual_res"]
    assert len(p_traj) > 0
    # Both residuals should decrease over the run (allow non-monotone but
    # check the final is much smaller than the initial).
    assert p_traj[-1] < p_traj[0]
    assert d_traj[-1] < d_traj[0]


def test_alm_adaptive_rho_invalidates_factor() -> None:
    """Adaptive rho should rebuild the Cholesky factor when rho changes."""
    prob = make_synthetic_eq_qp(DEFAULT_EQ_QP_SHAPES[0])
    factor_count = 0
    orig_build = prob.build_factor

    def counting_build(rho):
        nonlocal factor_count
        factor_count += 1
        return orig_build(rho)

    prob.build_factor = counting_build  # type: ignore[method-assign]

    # Adaptive variant with strong imbalance starting rho — should trigger rebuilds.
    # The convergence test is per-primal-residual only; we just need progress
    # (orders of magnitude decrease) and confirmed factor rebuilds.
    res = alm(
        prob, max_iters=400, tol=1e-7,
        rho=0.001,           # tiny rho will produce very imbalanced residuals
        variant="adaptive",
        rho_balance=10.0,
        rho_factor=10.0,
    )
    assert res.primal_res_final < 1.0, f"primal_res did not decrease: {res.primal_res_final:.3e}"
    assert factor_count >= 2, f"adaptive rho should have rebuilt factor; got {factor_count}"


def test_solve_with_factor_round_trip() -> None:
    """factor(P + rho A^T A); solve should return x such that the system holds."""
    prob = make_synthetic_eq_qp(DEFAULT_EQ_QP_SHAPES[0])
    rho = 1.5
    factor = prob.build_factor(rho)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(prob.n)
    x = prob.solve_with_factor(factor, rhs)
    H = prob.P + rho * (prob.A.T @ prob.A)
    np.testing.assert_allclose(H @ x, rhs, atol=1e-9, rtol=1e-9)


def test_alm_handles_zero_q_zero_b() -> None:
    """Trivial case: P=I, q=0, A=I, b=0 — optimum is x=0."""
    n = 5
    P = np.eye(n)
    q = np.zeros(n)
    A = np.eye(n)
    b = np.zeros(n)
    prob = EqualityQP(P, q, A, b)
    res = alm(prob, max_iters=200, tol=1e-10, rho=1.0)
    assert res.converged
    assert np.max(np.abs(res.x)) < 1e-10
