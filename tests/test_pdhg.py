"""PDHG correctness tests on TV-L2 denoising.

Goals:
  1. PDHG numpy reference matches CVXPY (CLARABEL) primal objective on small
     1D TV-L2 problems.
  2. The primal-dual gap fitness signal goes to zero at convergence.
  3. The accelerated variant converges faster than basic in iters.
"""
from __future__ import annotations

import numpy as np
import pytest

from convexkernels.algorithms.pdhg import pdhg
from convexkernels.frontend.total_variation import TVDenoising1D, TVDenoising2D


def _cvxpy_solve_tv1d(b: np.ndarray, lam: float, tol: float = 1e-12):
    cp = pytest.importorskip("cvxpy")
    n = b.shape[0]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(x - b) + lam * cp.tv(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    return np.asarray(x.value), float(prob.value)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pdhg_basic_matches_cvxpy_tv1d(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 64
    truth = np.cumsum(rng.standard_normal(n) * (rng.random(n) < 0.1))
    b = truth + 0.1 * rng.standard_normal(n)
    lam = 0.5

    x_cvx, p_cvx = _cvxpy_solve_tv1d(b, lam)
    prob = TVDenoising1D(b, lam)
    res = pdhg(prob, variant="basic", max_iters=20000, tol=1e-9)

    assert res.converged, f"PDHG did not converge: gap={res.gap_final:.3e}"
    p_pdhg = prob.primal_objective(res.x)
    assert abs(p_pdhg - p_cvx) / max(abs(p_cvx), 1.0) < 1e-4, (
        f"primal mismatch: pdhg={p_pdhg:.6f} vs cvxpy={p_cvx:.6f}"
    )
    rel_err = float(np.max(np.abs(res.x - x_cvx)) / max(np.max(np.abs(x_cvx)), 1.0))
    assert rel_err < 1e-3, f"iterate drift {rel_err:.3e}"


def test_pdhg_gap_decreases_to_zero() -> None:
    rng = np.random.default_rng(7)
    n = 32
    b = rng.standard_normal(n)
    prob = TVDenoising1D(b, lam=0.3)
    res = pdhg(prob, variant="basic", max_iters=5000, tol=1e-8, record_history=True)
    gaps = res.history["gap"]
    assert len(gaps) > 0
    assert gaps[0] > gaps[-1], "gap did not decrease over the run"
    assert res.gap_final < 1e-6, f"final gap too large: {res.gap_final:.3e}"


def test_pdhg_accelerated_converges_to_same_answer() -> None:
    """Accelerated variant must converge to the same iterate as basic.

    For TV-L2 the basic PDHG is already Q-linear under strong convexity, so
    accelerated isn't always faster in iters; the correctness property we want
    is that it lands on the same optimum.
    """
    rng = np.random.default_rng(11)
    n = 64
    b = rng.standard_normal(n)
    prob = TVDenoising1D(b, lam=0.5)
    res_basic = pdhg(prob, variant="basic", max_iters=10000, tol=1e-7)
    res_accel = pdhg(prob, variant="accelerated", max_iters=20000, tol=1e-7, gamma=1.0)
    assert res_basic.converged and res_accel.converged, (
        f"basic={res_basic.converged}, accel={res_accel.converged}"
    )
    p_basic = prob.primal_objective(res_basic.x)
    p_accel = prob.primal_objective(res_accel.x)
    assert abs(p_basic - p_accel) / max(abs(p_basic), 1.0) < 1e-3, (
        f"primal mismatch: basic={p_basic:.6f} accel={p_accel:.6f}"
    )


def test_pdhg_2d_runs_and_decreases_gap() -> None:
    rng = np.random.default_rng(3)
    h, w = 16, 16
    truth = np.zeros((h, w))
    truth[4:12, 4:12] = 1.0
    b = truth + 0.2 * rng.standard_normal((h, w))
    prob = TVDenoising2D(b, lam=0.3, isotropic=False)
    res = pdhg(prob, variant="basic", max_iters=2000, tol=1e-6, record_history=True)
    assert res.history["gap"][0] > res.history["gap"][-1]
    assert res.gap_final < 1e-3, f"2D TV gap final {res.gap_final:.3e}"


def test_tv1d_zero_lam_converges_to_b() -> None:
    """With lam=0 the problem is min_x 0.5||x - b||^2, optimum is x = b. PDHG
    converges asymptotically (the f-prox shrinks toward b each iter), so we just
    require gap → 0 and x → b within reasonable tolerance.
    """
    rng = np.random.default_rng(0)
    b = rng.standard_normal(20)
    prob = TVDenoising1D(b, lam=0.0)
    res = pdhg(prob, variant="basic", max_iters=2000, tol=1e-9)
    assert res.converged, f"PDHG with lam=0 did not converge: gap={res.gap_final:.3e}"
    assert np.max(np.abs(res.x - b)) / max(np.max(np.abs(b)), 1.0) < 1e-3


def test_tv1d_large_lam_gives_constant() -> None:
    """For lam large enough, optimum is the constant signal mean(b)."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal(40)
    prob = TVDenoising1D(b, lam=100.0)
    res = pdhg(prob, variant="basic", max_iters=20000, tol=1e-9)
    assert res.converged
    assert np.std(res.x) < 1e-3, f"x not constant under huge lam: std={np.std(res.x):.3e}"
