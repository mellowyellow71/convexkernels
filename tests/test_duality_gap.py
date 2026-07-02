"""Tests for the oracle-free LASSO duality gap (single + batched path)."""
from __future__ import annotations

import numpy as np

from convexkernels.algorithms.gap import lasso_duality_gap, lasso_duality_gap_batched
from convexkernels.bench.metrics import trusted_gap, trusted_kkt
from convexkernels.frontend.lasso import Lasso
from convexkernels.frontend.lasso_path import LassoPath


def _make(m=400, n=150, k=10, lam_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam = lam_frac * float(np.max(np.abs(A.T @ b)))
    return A, b, lam


def _fista(prob, iters=4000):
    x = np.zeros(prob.n); y = x.copy(); t = 1.0 / prob.L; th = 1.0
    for _ in range(iters):
        g = prob.grad_smooth(y); xn = prob.prox(y - t * g, t)
        if np.dot(y - xn, xn - x) > 0:
            tn, mo = 1.0, 0.0
        else:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * th * th)); mo = (th - 1) / tn
        y = xn + mo * (xn - x); x = xn; th = tn
    return x


def test_gap_closed_form_at_zero():
    A, b, lam = _make()
    p = Lasso(A, b, lam)
    r = lam / p.lambda_max
    halfb = 0.5 * float(b @ b)
    expect = halfb * (1 - r) ** 2 / (halfb + 1.0)
    assert abs(p.duality_gap(np.zeros(p.n)) - expect) < 1e-9


def test_gap_zero_at_optimum_and_nonnegative():
    A, b, lam = _make()
    p = Lasso(A, b, lam)
    x = _fista(p)
    assert p.duality_gap(x) < 1e-8
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert p.duality_gap(rng.standard_normal(p.n)) >= 0.0


def test_gap_vanishes_when_lambda_exceeds_lambda_max():
    # For lam >= lambda_max, x=0 is optimal, so the gap at 0 must be ~0.
    A, b, _ = _make()
    p = Lasso(A, b, 1.01 * float(np.max(np.abs(A.T @ b))))
    assert p.duality_gap(np.zeros(p.n)) < 1e-12


def test_trusted_gap_dispatches_to_duality_gap():
    A, b, lam = _make()
    p = Lasso(A, b, lam)
    x = _fista(p, iters=500)
    assert abs(trusted_gap(p, x) - p.duality_gap(x)) < 1e-15
    # and it is a different ruler than the KKT residual (both oracle-free)
    assert trusted_gap(p, x) >= 0.0 and trusted_kkt(p, x) >= 0.0


def test_batched_path_gap():
    A, b, _ = _make(m=500, n=120)
    lmax = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(0.6 * lmax, 0.03 * lmax, 8)
    p = LassoPath(A, b, lambdas)
    X0 = np.zeros((p.n, p.K))
    per_col = p.duality_gap_per_col(X0)
    assert per_col.shape == (p.K,)
    assert np.all(per_col >= 0.0)
    # column with the largest lambda has the smallest gap at x=0 (closest to 0-opt)
    assert per_col[0] < per_col[-1]
    assert abs(p.duality_gap_max(X0) - float(np.max(per_col))) < 1e-15
    # one column matches the single-LASSO gap with the same lambda
    single = Lasso(A, b, float(lambdas[3])).duality_gap(X0[:, 3])
    assert abs(per_col[3] - single) < 1e-12
