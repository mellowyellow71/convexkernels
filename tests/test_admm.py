"""ADMM correctness tests on LASSO via x=z splitting."""
from __future__ import annotations

import numpy as np
import pytest

from convexkernels.algorithms.admm import admm
from convexkernels.algorithms.kkt import lasso_kkt_residual
from convexkernels.bench.shapes import DEFAULT_SHAPES, make_synthetic_lasso
from convexkernels.frontend.lasso_admm import LassoAdmm


def _cvxpy_lasso(A, b, lam, tol: float = 1e-12):
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(A.shape[1])
    obj = 0.5 * cp.sum_squares(A @ x - b) + lam * cp.norm(x, 1)
    cp.Problem(cp.Minimize(obj)).solve(
        solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
    )
    return np.asarray(x.value)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_admm_lasso_matches_cvxpy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    m, n = 200, 100
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))

    prob = LassoAdmm(A, b, lam)
    # Use adaptive rho to converge faster on these random instances.
    res = admm(prob, max_iters=5000, tol=1e-5, rho=1.0, variant="adaptive")
    assert res.converged, f"ADMM did not converge: primal={res.primal_res_final:.3e} dual={res.dual_res_final:.3e}"

    x_cvx = _cvxpy_lasso(A, b, lam)
    p_admm = prob.primal_objective(res.z)
    p_cvx = prob.primal_objective(x_cvx)
    assert abs(p_admm - p_cvx) / max(abs(p_cvx), 1.0) < 1e-3
    rel = float(np.max(np.abs(res.z - x_cvx)) / max(np.max(np.abs(x_cvx)), 1.0))
    assert rel < 5e-2, f"iterate drift {rel:.3e}"


def test_admm_residuals_decrease() -> None:
    """Both residuals should be small at the end. (Not strictly monotone:
    when lam is large enough, z stays at zero for the first iter and the
    initial dual residual is zero; the assertion is just final-near-zero.)
    """
    rng = np.random.default_rng(0)
    m, n = 100, 50
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ x_true
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    prob = LassoAdmm(A, b, lam)
    res = admm(prob, max_iters=2000, tol=1e-7, rho=1.0, record_history=True)
    assert res.primal_res_final < 1e-5, f"final primal_res {res.primal_res_final:.3e}"
    assert res.dual_res_final < 1e-5, f"final dual_res {res.dual_res_final:.3e}"
    p_traj = res.history["primal_res"]
    assert p_traj[-1] <= p_traj[0]


def test_admm_kkt_at_optimum() -> None:
    """ADMM-converged z should satisfy the LASSO KKT residual."""
    rng = np.random.default_rng(7)
    m, n = 150, 80
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))

    prob = LassoAdmm(A, b, lam)
    res = admm(prob, max_iters=10000, tol=1e-9, rho=1.0)
    assert res.converged
    kkt = lasso_kkt_residual(A, b, lam, res.z)
    assert kkt < 1e-4, f"KKT residual {kkt:.3e} too large at ADMM optimum"


def test_admm_adaptive_rho_invalidates_factor() -> None:
    rng = np.random.default_rng(2)
    m, n = 100, 50
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ x_true
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    prob = LassoAdmm(A, b, lam)

    factor_count = 0
    orig_build = prob.build_factor

    def counting_build(rho):
        nonlocal factor_count
        factor_count += 1
        return orig_build(rho)

    prob.build_factor = counting_build  # type: ignore[method-assign]

    res = admm(
        prob, max_iters=1000, tol=1e-7,
        rho=0.001, variant="adaptive",
        rho_balance=10.0, rho_factor=10.0,
    )
    assert factor_count >= 2, f"adaptive rho should rebuild factor; got {factor_count}"
    assert res.primal_res_final < 1.0, f"primal_res did not decrease: {res.primal_res_final:.3e}"
