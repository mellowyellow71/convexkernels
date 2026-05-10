"""KKT residual is the fitness function — these tests are non-negotiable."""

import cvxpy as cp
import numpy as np
import pytest

from convexkernels.algorithms.kkt import lasso_kkt_residual
from convexkernels.frontend.lasso import Lasso


def make_problem(m: int = 100, n: int = 50, sparsity: float = 0.2,
                 noise: float = 1e-2, lam_frac: float = 0.1, seed: int = 42) -> Lasso:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < sparsity)
    b = A @ x_true + noise * rng.standard_normal(m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam=lam_frac * lam_max)


def cvxpy_solve(prob: Lasso) -> np.ndarray:
    x = cp.Variable(prob.n)
    obj = 0.5 * cp.sum_squares(prob.A @ x - prob.b) + prob.lam * cp.norm1(x)
    cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
    return np.asarray(x.value)


def test_kkt_at_cvxpy_optimum_is_near_zero():
    prob = make_problem()
    x_star = cvxpy_solve(prob)
    residual = prob.kkt_residual(x_star)
    assert residual < 1e-6, f"KKT residual at CVXPY optimum = {residual:.2e}, expected < 1e-6"


def test_free_function_matches_method():
    prob = make_problem()
    x_star = cvxpy_solve(prob)
    via_method = prob.kkt_residual(x_star)
    via_function = lasso_kkt_residual(prob.A, prob.b, prob.lam, x_star)
    assert abs(via_method - via_function) < 1e-12


def test_kkt_at_random_point_is_large():
    prob = make_problem()
    rng = np.random.default_rng(0)
    x_random = rng.standard_normal(prob.n)
    assert prob.kkt_residual(x_random) > 0.01


def test_kkt_at_zero_when_lam_above_max_is_zero():
    prob_base = make_problem()
    prob_big = Lasso(prob_base.A, prob_base.b, lam=prob_base.lambda_max * 1.5)
    assert prob_big.kkt_residual(np.zeros(prob_big.n)) < 1e-12


def test_kkt_at_zero_when_lam_below_max_is_positive():
    prob = make_problem(lam_frac=0.1)
    residual_at_zero = prob.kkt_residual(np.zeros(prob.n))
    assert residual_at_zero > 0.1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kkt_at_optimum_across_seeds(seed: int):
    prob = make_problem(seed=seed)
    x_star = cvxpy_solve(prob)
    residual = prob.kkt_residual(x_star)
    assert residual < 1e-5, f"seed={seed}: KKT={residual:.2e}"
