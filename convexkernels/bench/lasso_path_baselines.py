"""Full-regularization-path LASSO baseline harness.

Each baseline takes `(A, b, lambdas)` (or a `LassoPath`), returns a
`PathBaselineResult` with the (n, K) solution matrix, wall time, and the
per-lambda KKT residual under our scale-free formulation.

Multi-rep median, configurable warmups. Mirrors the Sakana-hardened harness
discipline used in `synth/_eval_kernel.py`.

Lambda convention: input `lambdas` is decreasing (high-to-low, glmnet/Adelie
standard). Adelie's `grpnet` parameterizes its loss as `(1/m)||Xb - y||^2/2`,
so we map `our_lam` -> `adelie_lam = our_lam / m`. sklearn maps the same way.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..algorithms.kkt_batched import lasso_kkt_residual_batched
from ..frontend.lasso_path import LassoPath


@dataclass
class PathBaselineResult:
    name: str
    X: np.ndarray            # (n, K)
    wall_time_s: float       # median across reps
    wall_times_s: list[float]  # all rep times
    kkt_per_lambda: np.ndarray  # (K,) under our scale-free formulation
    primal_obj_per_lambda: np.ndarray  # (K,)
    extra: dict = field(default_factory=dict)


def _path_primal_obj(prob: LassoPath, X: np.ndarray) -> np.ndarray:
    """Per-column primal objective: 0.5||AX - b||^2 + lam_k * ||X[:, k]||_1."""
    R = prob.A @ X - prob.b[:, None]
    sq = 0.5 * np.sum(R * R, axis=0)
    l1 = np.sum(np.abs(X), axis=0)
    return sq + prob.lambdas * l1


def _multirep_median(
    fn, *, reps: int, warmup: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Run `fn()` warmup+reps times; discard warmups; return (X_last, median_s, times)."""
    for _ in range(warmup):
        _ = fn()
    times = []
    X_last = None
    for _ in range(reps):
        t0 = time.perf_counter()
        X_last = fn()
        times.append(time.perf_counter() - t0)
    return X_last, float(np.median(times)), times


def _gate(name: str, prob: LassoPath, X: np.ndarray,
          wall_s: float, times: list[float],
          extra: Optional[dict] = None) -> PathBaselineResult:
    kkt = lasso_kkt_residual_batched(
        prob.A, prob.b, prob.lambdas, X,
        L=prob.L, lambda_max=prob.lambda_max,
    )
    obj = _path_primal_obj(prob, X)
    return PathBaselineResult(
        name=name, X=X, wall_time_s=wall_s, wall_times_s=times,
        kkt_per_lambda=kkt, primal_obj_per_lambda=obj,
        extra=extra or {},
    )


def run_numpy_path(
    prob: LassoPath, *, reps: int = 5, warmup: int = 2,
    max_iters: int = 20000, tol: float = 1e-7,
) -> PathBaselineResult:
    """Sequential FISTA-Gram numpy reference with warm-starting along the path."""
    from ..kernels.numpy_fista_path_ref import fista_path_numpy

    def go():
        return fista_path_numpy(prob, max_iters=max_iters, tol=tol).X

    X, wall, times = _multirep_median(go, reps=reps, warmup=warmup)
    return _gate("numpy_fista_path", prob, X, wall, times,
                 extra={"max_iters": max_iters, "tol": tol})


def run_sklearn_path(
    prob: LassoPath, *, reps: int = 5, warmup: int = 2,
    max_iter: int = 20000, tol: float = 1e-8,
) -> PathBaselineResult:
    """sklearn `lasso_path` with the same lambda grid.

    sklearn parameterizes as `(1/(2*m))||Ax-y||^2 + alpha||x||_1`, so the
    mapping is `alpha = lam / m`. Returns `(alphas, coefs, _)` with coefs
    shape `(n, K)`. Reportedly slow on wide problems; baseline only.
    """
    from sklearn.linear_model import lasso_path

    alphas = prob.lambdas / prob.m

    def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, coefs, _ = lasso_path(
                prob.A, prob.b, alphas=alphas,
                max_iter=max_iter, tol=tol, fit_intercept=False,
            )
        return np.ascontiguousarray(coefs, dtype=np.float64)

    X, wall, times = _multirep_median(go, reps=reps, warmup=warmup)
    return _gate("sklearn_path", prob, X, wall, times,
                 extra={"alphas": alphas, "max_iter": max_iter, "tol": tol})


def run_adelie_path(
    prob: LassoPath, *, reps: int = 5, warmup: int = 2,
    tol: float = 1e-12, max_iters: int = int(1e6),
) -> PathBaselineResult:
    """Adelie full-path solve via `grpnet(lmda_path=...)`.

    Adelie's loss is `(1/m)||Xb - y||^2/2 + lambda||b||_1`, so the mapping
    is `adelie_lambda = our_lambda / m`. The full lambda path is passed in
    one call; Adelie internally warm-starts along the path with screening.

    This is the *headline number* we are trying to beat on `path_wide_hero`.
    """
    import adelie as ad
    A_f = np.asfortranarray(prob.A)
    adelie_lmdas = np.ascontiguousarray(prob.lambdas / prob.m)

    def go():
        state = ad.solver.grpnet(
            X=A_f,
            glm=ad.glm.gaussian(y=prob.b),
            intercept=False,
            lmda_path=adelie_lmdas,
            tol=tol,
            max_iters=max_iters,
            early_exit=False,  # honor every lambda in lmda_path
            progress_bar=False,
        )
        Xs = [np.asarray(b.toarray().squeeze()) for b in state.betas]
        return np.ascontiguousarray(np.stack(Xs, axis=1), dtype=np.float64)

    X, wall, times = _multirep_median(go, reps=reps, warmup=warmup)
    return _gate("adelie_path", prob, X, wall, times,
                 extra={"tol": tol, "max_iters": max_iters})


def run_mlx_fista_path(
    prob: LassoPath, *, reps: int = 5, warmup: int = 2,
    max_iters: int = 10000, tol: float = 1e-6,
    convergence_check_every: int = 10,
    dtype: str = "fp32",
) -> PathBaselineResult:
    """Batched FISTA-Gram path solver on MLX (Apple Silicon).

    This is the **seed** that Phase 3 autoresearch will mutate. Each rep
    rebuilds the MLX problem view and the Gram precompute under the timing
    contract — `prepare_problem` cost is included in wall_ms so the
    candidate cannot move per-iter work into setup without paying for it.

    Convergence: max-over-columns KKT residual gated every
    `convergence_check_every` iters on device (no per-iter host transfer).
    """
    import mlx.core as mx

    from ..algorithms.fista_path import fista_path
    from ..kernels.mlx.lib import LassoPathMLX
    from ..kernels.mlx.seeds.gram_fista_path_v0 import (
        fista_path_step, init_state, kkt_max, prepare_problem,
    )

    mlx_dtype = {"fp32": mx.float32, "fp16": mx.float16,
                 "bf16": mx.bfloat16}[dtype]

    # The MLX problem view holds A, b, lambdas on device. Building it is
    # part of setup (path-independent across reps after the first), so we
    # build it once outside the timed region.
    prob_mlx = LassoPathMLX.from_lasso_path(prob, dtype=mlx_dtype)
    # Force-materialize so transfer cost is amortized before warmups.
    mx.eval(prob_mlx.A, prob_mlx.b, prob_mlx.lambdas)

    def go():
        # prepare_problem builds the kernel-side LassoPathGramMLX (Gram or
        # direct mode); both setup and solve are inside the timing window.
        prob_kernel = prepare_problem(prob_mlx)
        res = fista_path(
            prob_kernel,
            max_iters=max_iters,
            tol=tol,
            convergence_check_every=convergence_check_every,
            kernel_step=fista_path_step,
            kernel_init=init_state,
            kernel_kkt_max=kkt_max,
            record_history=False,
        )
        return res.X

    X, wall, times = _multirep_median(go, reps=reps, warmup=warmup)
    return _gate(
        f"mlx_fista_path_{dtype}", prob, X, wall, times,
        extra={"max_iters": max_iters, "tol": tol,
               "convergence_check_every": convergence_check_every,
               "dtype": dtype},
    )


ALL_PATH_BASELINES = (run_numpy_path, run_sklearn_path, run_adelie_path)
