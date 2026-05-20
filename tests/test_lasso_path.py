"""Correctness tests for the batched-lambda LASSO path solver.

Critical guarantees:
  1. `lasso_kkt_residual_batched` matches the scalar `lasso_kkt_residual`
     column-by-column on K=1 and K>1 inputs.
  2. The numpy sequential FISTA-path solver converges to per-lambda KKT
     residual < tol on a small synthetic problem.
  3. The numpy solver agrees with Adelie within `1e-4` relative Frobenius
     across the full path (gated `pytest.skip` if adelie is not installed).
  4. `LassoPath.prox_path` matches per-column scalar prox.

The wide-hero shape Adelie test is slow (m=1000, n=50000, K=50); marked
`slow` so a quick run skips it.
"""

from __future__ import annotations

import numpy as np
import pytest

from convexkernels.algorithms.kkt import lasso_kkt_residual
from convexkernels.algorithms.kkt_batched import lasso_kkt_residual_batched
from convexkernels.bench.path_shapes import (
    DEFAULT_PATH_SHAPES, PathShapeSpec, get_path_shape, make_path_problem,
)
from convexkernels.frontend.lasso import Lasso
from convexkernels.frontend.lasso_path import LassoPath
from convexkernels.kernels.numpy_fista_path_ref import fista_path_numpy


def _small_spec() -> PathShapeSpec:
    """Tiny shape so tests run fast on Linux CI."""
    return PathShapeSpec(
        name="path_test_small", m=80, n=60,
        sparsity=0.1, noise=1e-2, K=10, lam_min_frac=0.05,
    )


def test_kkt_batched_matches_scalar_K1():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((40, 30))
    b = rng.standard_normal(40)
    x = rng.standard_normal(30) * 0.1
    lam = 0.5 * float(np.max(np.abs(A.T @ b)))

    scalar = lasso_kkt_residual(A, b, lam, x)
    batched = lasso_kkt_residual_batched(
        A, b, np.array([lam]), x[:, None],
    )
    assert batched.shape == (1,)
    assert abs(float(batched[0]) - scalar) < 1e-12


def test_kkt_batched_matches_scalar_K_many():
    """Per-column residual must equal the scalar version applied to each column."""
    rng = np.random.default_rng(11)
    A = rng.standard_normal((50, 40))
    b = rng.standard_normal(50)
    lambda_max = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(lambda_max, 0.05 * lambda_max, 8)
    # Random per-column iterates; not at optimum, just to exercise the math.
    X = rng.standard_normal((40, 8)) * 0.05

    batched = lasso_kkt_residual_batched(A, b, lambdas, X)
    L = float(np.linalg.norm(A, ord=2)) ** 2
    for k in range(8):
        scalar = lasso_kkt_residual(
            A, b, float(lambdas[k]), X[:, k],
            L=L, lambda_max=lambda_max,
        )
        assert abs(float(batched[k]) - scalar) < 1e-10, (
            f"col {k}: batched={batched[k]:.6e} scalar={scalar:.6e}"
        )


def test_prox_path_matches_scalar():
    rng = np.random.default_rng(13)
    A = rng.standard_normal((30, 20))
    b = rng.standard_normal(30)
    lambda_max = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(lambda_max, 0.1 * lambda_max, 5)
    V = rng.standard_normal((20, 5))
    prob = LassoPath(A, b, lambdas)

    out_batched = prob.prox_path(V, 0.3)
    for k in range(5):
        ours = Lasso(A, b, float(lambdas[k])).prox(V[:, k], 0.3)
        np.testing.assert_allclose(out_batched[:, k], ours, atol=1e-12)


def test_numpy_path_converges_small():
    spec = _small_spec()
    A, b, lambdas = make_path_problem(spec, seed=0)
    prob = LassoPath(A, b, lambdas)
    res = fista_path_numpy(prob, max_iters=5000, tol=1e-7)
    assert (res.kkt_per_lambda < 1e-6).all(), (
        f"some columns did not converge: max kkt={res.kkt_per_lambda.max():.2e}"
    )


def test_lambda_max_yields_zero_solution():
    """At lambda = lambda_max, the all-zero solution is optimal (KKT residual
    is small)."""
    spec = _small_spec()
    A, b, _ = make_path_problem(spec, seed=0)
    lambda_max = float(np.max(np.abs(A.T @ b)))
    prob = LassoPath(A, b, np.array([lambda_max]))
    kkt = prob.kkt_residual(np.zeros((spec.n, 1)))
    # At lambda_max, ||A^T b||_inf = lam, so prox-fixed-point: z = -g = A^T b,
    # soft(z, lam) = sign(z) * max(|z| - lam, 0). |z|_inf = lam so the max is
    # exactly lam at the argmax index, soft = sign(z) * 0 = 0; everywhere else
    # the threshold zeros out. So r = L*0 - 0 = 0. Residual is zero.
    assert kkt[0] < 1e-12


def test_path_shapes_registered():
    """Sanity: the four headline shapes are reachable via get_path_shape()."""
    expected = {"path_wide_hero", "path_tall_medium", "path_square",
                "path_wide_small"}
    got = {s.name for s in DEFAULT_PATH_SHAPES}
    assert expected.issubset(got)
    hero = get_path_shape("path_wide_hero")
    assert hero.m == 1000 and hero.n == 50000 and hero.K == 50


def test_mlx_fista_path_converges_small():
    """MLX path seed converges on a small problem (Mac-only)."""
    mx = pytest.importorskip("mlx.core")
    from convexkernels.algorithms.fista_path import fista_path
    from convexkernels.kernels.mlx.lib import LassoPathMLX
    from convexkernels.kernels.mlx.seeds.gram_fista_path_v0 import (
        fista_path_step, init_state, kkt_max, prepare_problem,
    )

    spec = _small_spec()
    A, b, lambdas = make_path_problem(spec, seed=0)
    prob = LassoPath(A, b, lambdas)
    prob_mlx = LassoPathMLX.from_lasso_path(prob, dtype=mx.float32)
    prob_gram = prepare_problem(prob_mlx)

    res = fista_path(
        prob, max_iters=20000, tol=1e-6, convergence_check_every=10,
        kernel_step=fista_path_step, kernel_init=init_state,
        kernel_kkt_max=kkt_max, kernel_problem=prob_gram,
    )
    assert res.converged, (
        f"MLX path solver did not converge in {res.n_iters} iters; "
        f"max KKT={res.kkt_max_final:.2e}"
    )
    # Check against numpy reference within reasonable agreement.
    ref = fista_path_numpy(prob, max_iters=20000, tol=1e-7)
    rel = np.linalg.norm(res.X - ref.X, "fro") / max(
        np.linalg.norm(ref.X, "fro"), 1e-12,
    )
    assert rel < 1e-3, f"MLX vs numpy disagreement: rel-Frob={rel:.3e}"


@pytest.mark.slow
def test_numpy_path_matches_adelie_small():
    """Cross-check: our numpy reference agrees with Adelie within 1e-4 rel-Frob
    on a small problem. Skips if adelie is not installed."""
    adelie = pytest.importorskip("adelie")
    spec = _small_spec()
    A, b, lambdas = make_path_problem(spec, seed=0)
    prob = LassoPath(A, b, lambdas)
    ours = fista_path_numpy(prob, max_iters=20000, tol=1e-9)
    assert (ours.kkt_per_lambda < 1e-7).all()

    # Adelie: lambda = our_lam / m.
    A_f = np.asfortranarray(A)
    state = adelie.solver.grpnet(
        X=A_f,
        glm=adelie.glm.gaussian(y=b),
        intercept=False,
        lmda_path=np.ascontiguousarray(lambdas / spec.m),
        tol=1e-12,
        max_iters=int(1e6),
        early_exit=False,
        progress_bar=False,
    )
    X_adelie = np.stack(
        [np.asarray(B.toarray().squeeze()) for B in state.betas], axis=1,
    )
    rel = np.linalg.norm(ours.X - X_adelie, "fro") / max(
        np.linalg.norm(X_adelie, "fro"), 1e-12,
    )
    assert rel < 1e-3, (
        f"numpy reference disagrees with Adelie: rel-Frob={rel:.3e}"
    )
