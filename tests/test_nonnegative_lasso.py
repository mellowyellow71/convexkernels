"""Nonnegative LASSO frontend, KKT, and FISTA tests."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from convexkernels.algorithms.fista import fista
from convexkernels.algorithms.kkt import nonnegative_lasso_kkt_residual
from convexkernels.frontend.nonnegative_lasso import NonnegativeLasso


def make_problem(
    m: int = 120,
    n: int = 80,
    sparsity: float = 0.15,
    noise: float = 1e-2,
    lam_frac: float = 0.1,
    seed: int = 0,
) -> NonnegativeLasso:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.abs(rng.standard_normal(n)) * (rng.random(n) < sparsity)
    b = A @ x_true + noise * rng.standard_normal(m)
    lam_max = float(max(np.max(A.T @ b), 0.0))
    return NonnegativeLasso(A, b, lam=lam_frac * lam_max)


def cvxpy_solve(
    prob: NonnegativeLasso,
    eps: float = 1e-11,
    max_iter: int = 50000,
) -> np.ndarray:
    x = cp.Variable(prob.n, nonneg=True)
    obj = 0.5 * cp.sum_squares(prob.A @ x - prob.b) + prob.lam * cp.sum(x)
    cp.Problem(cp.Minimize(obj)).solve(
        solver=cp.CLARABEL,
        tol_gap_abs=eps,
        tol_gap_rel=eps,
        tol_feas=eps,
        max_iter=max_iter,
    )
    return np.asarray(x.value)


def test_nonnegative_lasso_prox_projects_to_nonnegative():
    prob = make_problem()
    v = np.array([-2.0, -0.1, 0.05, 1.0])

    out = prob.prox(v, t=0.5)

    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out, np.maximum(v - 0.5 * prob.lam, 0.0))


def test_nonnegative_lasso_kkt_at_cvxpy_optimum_is_near_zero():
    prob = make_problem(seed=1)
    x_star = cvxpy_solve(prob)

    assert prob.kkt_residual(x_star) < 1e-6


def test_nonnegative_lasso_free_function_matches_method():
    prob = make_problem(seed=2)
    x_star = cvxpy_solve(prob)

    via_method = prob.kkt_residual(x_star)
    via_function = nonnegative_lasso_kkt_residual(
        prob.A,
        prob.b,
        prob.lam,
        x_star,
        L=prob.L,
        lambda_max=prob.lambda_max,
    )

    assert abs(via_method - via_function) < 1e-12


def test_nonnegative_lasso_zero_solution_when_lam_above_max():
    prob = make_problem(seed=3)
    prob_big = NonnegativeLasso(prob.A, prob.b, lam=1.5 * prob.lambda_max)

    assert prob_big.kkt_residual(np.zeros(prob_big.n)) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonnegative_lasso_fista_converges(seed: int):
    prob = make_problem(seed=seed)

    result = fista(prob, max_iters=8000, tol=1e-6, variant="restart")

    assert result.converged, (
        f"seed={seed}: KKT={result.kkt_final:.2e} after {result.n_iters} iters"
    )
    assert np.min(result.x) >= -1e-10


def test_nonnegative_lasso_fista_matches_cvxpy():
    prob = make_problem(seed=4)
    x_cvxpy = cvxpy_solve(prob)
    result = fista(prob, max_iters=10000, tol=1e-7, variant="restart")

    assert result.converged
    assert result.kkt_final < 1e-6
    rel_drift = float(
        np.max(np.abs(result.x - x_cvxpy)) / max(np.max(np.abs(x_cvxpy)), 1.0)
    )
    assert rel_drift < 1e-4
