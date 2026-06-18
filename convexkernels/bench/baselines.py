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


# Per-solver kwarg for the max-iteration cap (used to build anytime curves).
_MAXITER_KW = {
    "CLARABEL": "max_iter",
    "SCS": "max_iters",
    "OSQP": "max_iter",
    "ECOS": "max_iters",
}


def build_cvxpy_lasso(prob: Lasso):
    """Build a (reusable) cvxpy LASSO problem; returns `(cvxprob, x_var)`.

    Reusing one `cp.Problem` across an iteration-cap sweep means cvxpy
    canonicalizes once (amortized), so each anytime-curve point reflects solver
    convergence rather than repeated Python modeling overhead.
    """
    import cvxpy as cp

    x = cp.Variable(prob.n)
    obj = 0.5 * cp.sum_squares(prob.A @ x - prob.b) + prob.lam * cp.norm1(x)
    return cp.Problem(cp.Minimize(obj)), x


def _solver_kwargs(solver: str, max_iter: Optional[int], eps: float) -> dict:
    kwargs: dict = {}
    if max_iter is not None:
        kwargs[_MAXITER_KW[solver]] = int(max_iter)
    if solver == "CLARABEL":
        kwargs.update(tol_gap_abs=eps, tol_gap_rel=eps, tol_feas=eps)
    elif solver == "SCS":
        kwargs.update(eps=eps)
    elif solver == "OSQP":
        kwargs.update(eps_abs=eps, eps_rel=eps)
    elif solver == "ECOS":
        kwargs.update(abstol=eps, reltol=eps, feastol=eps)
    return kwargs


def solve_existing_cvxpy(
    cvxprob, x_var, solver: str, *,
    max_iter: Optional[int] = None, eps: float = 1e-12,
) -> tuple[Optional[np.ndarray], float]:
    """Solve an already-built cvxpy problem; returns `(x, wall_time_s)`."""
    import cvxpy as cp

    kwargs = _solver_kwargs(solver, max_iter, eps)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cvxprob.solve(solver=getattr(cp, solver), verbose=False, **kwargs)
    except Exception:  # noqa: BLE001 — capped/failed solves are expected on the sweep
        return None, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    val = None if x_var.value is None else np.asarray(x_var.value)
    return val, wall


def solve_cvxpy_lasso(
    prob: Lasso, *, solver: str = "CLARABEL",
    max_iter: Optional[int] = None, eps: float = 1e-12,
) -> tuple[Optional[np.ndarray], float]:
    """One-shot: build + solve the LASSO with a chosen cvxpy backend."""
    cvxprob, x = build_cvxpy_lasso(prob)
    return solve_existing_cvxpy(cvxprob, x, solver, max_iter=max_iter, eps=eps)


def run_cvxpy(prob: Lasso, *, solver: str = "CLARABEL", eps: float = 1e-12,
              max_iter: int = 50000) -> BaselineResult:
    """CVXPY baseline. `solver` ∈ {CLARABEL (IPM), SCS, OSQP, ECOS}."""
    x, wall = solve_cvxpy_lasso(prob, solver=solver, max_iter=max_iter, eps=eps)
    if x is None:
        x = np.zeros(prob.n)
    return _result(solver.lower(), x, None, wall, prob)


def run_scs(prob: Lasso, **kw) -> BaselineResult:
    return run_cvxpy(prob, solver="SCS", **kw)


def run_osqp(prob: Lasso, **kw) -> BaselineResult:
    return run_cvxpy(prob, solver="OSQP", **kw)


def run_ecos(prob: Lasso, **kw) -> BaselineResult:
    return run_cvxpy(prob, solver="ECOS", **kw)


# Names accepted by the cvxpy-backed anytime-curve machinery in `curves.py`.
CVXPY_SOLVERS = ("CLARABEL", "SCS", "OSQP", "ECOS")

ALL_BASELINES = (
    run_numpy_fista, run_sklearn, run_adelie,
    run_cvxpy, run_scs, run_osqp, run_ecos,
)
