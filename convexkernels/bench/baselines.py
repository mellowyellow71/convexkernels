"""Baseline solver adapters.

Each baseline takes a `Lasso` problem, returns a `BaselineResult`. The harness
in `run.py` runs all baselines on each shape and logs to `tasks/results.md`.

Solvers covered:
  - numpy_fista (our reference, restart variant)
  - sklearn   (coordinate descent, sklearn.linear_model.Lasso)
  - adelie    (prox-Newton + CD with screening, adelie.solver.grpnet)
  - cvxpy     (interior-point oracle, CLARABEL)

alpaqa (PANOC) is deferred — its Python API requires CasADi/JAX bindings to
construct a problem object, which is a large detour for a single baseline.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..frontend.lasso import Lasso


@dataclass
class BaselineResult:
    name: str
    x: np.ndarray
    n_iters: Optional[int]
    wall_time_s: float
    kkt_final: float
    primal_obj: float
    extra: dict = field(default_factory=dict)


def _primal_objective(prob: Lasso, x: np.ndarray) -> float:
    r = prob.matvec(x) - prob.b
    return 0.5 * float(r @ r) + prob.lam * float(np.sum(np.abs(x)))


def _result(name: str, x: np.ndarray, n_iters: Optional[int],
            wall: float, prob: Lasso, extra: Optional[dict] = None) -> BaselineResult:
    return BaselineResult(
        name=name,
        x=x,
        n_iters=n_iters,
        wall_time_s=wall,
        kkt_final=prob.kkt_residual(x),
        primal_obj=_primal_objective(prob, x),
        extra=extra or {},
    )


def run_numpy_fista(prob: Lasso, *, max_iters: int = 20000,
                    tol: float = 1e-7, variant: str = "restart") -> BaselineResult:
    from ..algorithms.fista import fista
    t0 = time.perf_counter()
    res = fista(
        prob, max_iters=max_iters, tol=tol, variant=variant,
        record_history=False,
    )
    return _result(
        f"numpy_fista_{variant}", res.x, res.n_iters,
        time.perf_counter() - t0, prob,
        extra={"converged": res.converged},
    )


def run_sklearn(prob: Lasso, *, max_iter: int = 20000,
                tol: float = 1e-8) -> BaselineResult:
    """sklearn `Lasso` minimizes (1/2n)||Ax-y||^2 + alpha||x||_1.

    Map to our parameterization: alpha = lam / m.
    """
    from sklearn.linear_model import Lasso as SkLasso
    alpha = prob.lam / prob.m
    model = SkLasso(
        alpha=alpha, fit_intercept=False, max_iter=max_iter,
        tol=tol, selection="cyclic",
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prob.A, prob.b)
    return _result(
        "sklearn", model.coef_, int(model.n_iter_),
        time.perf_counter() - t0, prob,
        extra={"alpha": alpha},
    )


def run_adelie(prob: Lasso, *, tol: float = 1e-12,
               max_iters: int = int(1e6)) -> BaselineResult:
    """adelie's `grpnet` minimizes (1/2n)||Xb - y||^2 + lambda||b||_1
    for single-feature groups (i.e. plain LASSO).

    Map: lambda = lam / m. Adelie expects F-contiguous matrices.
    """
    import adelie as ad
    A_f = np.asfortranarray(prob.A)
    t0 = time.perf_counter()
    state = ad.solver.grpnet(
        X=A_f,
        glm=ad.glm.gaussian(y=prob.b),
        intercept=False,
        lmda_path=[prob.lam / prob.m],
        tol=tol,
        max_iters=max_iters,
        progress_bar=False,
    )
    wall = time.perf_counter() - t0
    x = np.asarray(state.betas[-1].toarray().squeeze())
    return _result("adelie", x, None, wall, prob)


def run_cvxpy(prob: Lasso, *, eps: float = 1e-12,
              max_iter: int = 50000) -> BaselineResult:
    """CVXPY/CLARABEL — interior-point oracle for ground truth."""
    import cvxpy as cp
    x = cp.Variable(prob.n)
    obj = 0.5 * cp.sum_squares(prob.A @ x - prob.b) + prob.lam * cp.norm1(x)
    cvxprob = cp.Problem(cp.Minimize(obj))
    t0 = time.perf_counter()
    cvxprob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=eps, tol_gap_rel=eps, tol_feas=eps,
        max_iter=max_iter,
    )
    wall = time.perf_counter() - t0
    return _result("cvxpy", np.asarray(x.value), None, wall, prob)


ALL_BASELINES = (run_numpy_fista, run_sklearn, run_adelie, run_cvxpy)
