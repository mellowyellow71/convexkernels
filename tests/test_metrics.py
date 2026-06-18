import math

import numpy as np

from convexkernels.bench.metrics import time_to_target, trusted_kkt
from convexkernels.frontend.lasso import Lasso
from convexkernels.frontend.lasso_path import LassoPath


def _small_lasso(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((40, 20))
    x_true = np.zeros(20)
    x_true[:3] = rng.standard_normal(3)
    b = A @ x_true + 0.01 * rng.standard_normal(40)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def test_trusted_kkt_matches_frontend_single():
    prob = _small_lasso()
    x = np.zeros(prob.n)
    assert trusted_kkt(prob, x) == prob.kkt_residual(x)


def test_trusted_kkt_path_is_scalar_max():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((30, 12))
    b = rng.standard_normal(30)
    lmax = float(np.max(np.abs(A.T @ b)))
    lambdas = np.array([0.5, 0.2, 0.05]) * lmax
    prob = LassoPath(A, b, lambdas)
    X = np.zeros((prob.n, prob.K))
    val = trusted_kkt(prob, X)
    assert val == prob.kkt_residual_max(X)
    assert np.isscalar(val) or np.ndim(val) == 0


def test_trusted_kkt_small_near_optimum():
    # Solve to high accuracy with our own fista, KKT should be ~0.
    from convexkernels.algorithms.fista import fista

    prob = _small_lasso()
    res = fista(prob, max_iters=20000, tol=1e-10, variant="restart")
    assert trusted_kkt(prob, res.x) < 1e-6


def test_time_to_target_never_reached():
    pts = [(0.0, 1.0), (1.0, 0.5), (2.0, 0.2)]
    assert time_to_target(pts, 1e-3) == math.inf


def test_time_to_target_interpolates_loglinear():
    # kkt drops 1.0 -> 0.01 over t in [0,2]; tol=0.1 should land at t=1.0
    pts = [(0.0, 1.0), (2.0, 0.01)]
    t = time_to_target(pts, 0.1)
    assert abs(t - 1.0) < 1e-9


def test_time_to_target_first_point_already_below():
    pts = [(0.3, 1e-8), (0.6, 1e-9)]
    assert time_to_target(pts, 1e-6) == 0.3
