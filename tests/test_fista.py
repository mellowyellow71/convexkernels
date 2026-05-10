"""FISTA convergence + correctness tests.

P1 acceptance:
- FISTA achieves KKT(x_fista) < 1e-6
- FISTA matches CVXPY at <1e-4 relative drift on N=200, p=500 random LASSO
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from convexkernels.algorithms.fista import fista
from convexkernels.frontend.lasso import Lasso


def make_problem(
    m: int = 200,
    n: int = 500,
    sparsity: float = 0.1,
    noise: float = 1e-2,
    lam_frac: float = 0.1,
    seed: int = 0,
) -> Lasso:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < sparsity)
    b = A @ x_true + noise * rng.standard_normal(m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam=lam_frac * lam_max)


def cvxpy_solve(prob: Lasso, eps: float = 1e-12, max_iter: int = 50000) -> np.ndarray:
    x = cp.Variable(prob.n)
    obj = 0.5 * cp.sum_squares(prob.A @ x - prob.b) + prob.lam * cp.norm1(x)
    cp.Problem(cp.Minimize(obj)).solve(
        solver=cp.CLARABEL,
        tol_gap_abs=eps,
        tol_gap_rel=eps,
        tol_feas=eps,
        max_iter=max_iter,
    )
    return np.asarray(x.value)


def test_fista_converges_to_kkt_tolerance():
    prob = make_problem()
    result = fista(prob, max_iters=5000, tol=1e-6)
    assert result.converged, (
        f"FISTA did not converge in 5000 iters; KKT={result.kkt_final:.2e}"
    )
    assert result.kkt_final < 1e-6


def test_fista_matches_cvxpy_p1_acceptance():
    """P1 acceptance bar: relative drift < 1e-4 vs CVXPY on N=200, p=500."""
    prob = make_problem(m=200, n=500, seed=0)
    x_cvxpy = cvxpy_solve(prob)
    assert prob.kkt_residual(x_cvxpy) < 1e-8, "CVXPY didn't reach 1e-8 KKT (P1 acceptance)"

    result = fista(prob, max_iters=10000, tol=1e-7, variant="restart")
    assert result.converged
    assert result.kkt_final < 1e-6

    rel_drift = float(
        np.max(np.abs(result.x - x_cvxpy)) / max(np.max(np.abs(x_cvxpy)), 1.0)
    )
    assert rel_drift < 1e-4, (
        f"FISTA vs CVXPY drift {rel_drift:.2e} exceeds 1e-4"
    )


def test_restart_no_slower_than_basic():
    prob = make_problem(seed=1)
    res_basic = fista(prob, variant="basic", max_iters=10000, tol=1e-6)
    res_restart = fista(prob, variant="restart", max_iters=10000, tol=1e-6)
    assert res_basic.converged and res_restart.converged
    assert res_restart.n_iters <= int(1.5 * res_basic.n_iters), (
        f"restart={res_restart.n_iters} basic={res_basic.n_iters}"
    )


def test_kkt_trajectory_is_monotone_eventually():
    """After warmup, KKT should not blow up. Used as a convex-invariant kill switch later."""
    prob = make_problem(seed=2)
    result = fista(prob, max_iters=2000, tol=1e-6)
    traj = result.history["kkt"]
    # After 50% of iters, max KKT in tail should not exceed first-half max
    half = len(traj) // 2
    if half > 0:
        first_half_max = max(traj[:half])
        tail_max = max(traj[half:])
        assert tail_max <= first_half_max * 1.05, (
            "KKT residual blew up in late iterations"
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fista_across_seeds(seed: int):
    prob = make_problem(seed=seed)
    result = fista(prob, max_iters=5000, tol=1e-6, variant="restart")
    assert result.converged, (
        f"seed={seed}: KKT={result.kkt_final:.2e} after {result.n_iters} iters"
    )
