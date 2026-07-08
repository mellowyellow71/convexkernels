"""Anytime KKT-residual-vs-time curves for the classical baseline panel.

The autoresearch headline is "reach a target optimality faster than the
classical solvers." To compare apples-to-apples, every baseline is reduced to
the same anytime curve the candidate produces: a list of `(wall_time_s, kkt)`
points, with `kkt = metric(problem, x)` — the identical ruler used for the
candidate (see `bench/metrics.py`).

cvxpy exposes no portable per-iteration callback, so the curve is traced by
solving the baseline at a sweep of iteration caps (`max_iter ∈ sweep`); a larger
cap costs more time and reaches a smaller residual. This is the standard
anytime-benchmark construction when callbacks are unavailable. Curves are
cached per problem on disk (baselines do not change between sessions).

The panel also includes the *native fast-LASSO* solvers — scikit-learn
(coordinate descent) and adelie (CD + screening) — traced the same way via each
solver's own iteration cap. They are the real bar for LASSO: on typical shapes
their time-to-target is one-to-two orders of magnitude below the generic conic
panel, so omitting them lets a candidate "beat the baselines" while being far
slower than a trivial fast-LASSO call. These are optional deps (`.[baselines]`);
a missing one degrades to an empty curve (time_to_kkt = inf), never a crash.
For a path, both fast-LASSO baselines use their native warm-started path solve —
adelie a single multi-lambda `grpnet`, sklearn a single `lasso_path` (coordinate
descent warm-started down the decreasing-lambda grid) — not K independent
cold per-column solves, which understated the fast-LASSO bar.

For a `LassoPath`, the path is scored as a whole: at iteration budget `N`, each
column is solved (as a single LASSO) at cap `N`, the per-column solve times are
summed (sequential path solve), and the trusted *max-over-columns* KKT of the
assembled `(n, K)` matrix is the curve's `kkt` value at that total time.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..frontend.lasso import Lasso
from ..frontend.lasso_path import LassoPath
from .baselines import build_cvxpy_lasso, solve_existing_cvxpy
from .metrics import time_to_target, trusted_kkt

DEFAULT_SWEEP: tuple[int, ...] = (
    5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400,
)

# cvxpy-backed conic/QP solvers vs. native fast-LASSO solvers. The latter are
# the real bar to beat for LASSO (coordinate descent + screening), and are
# optional deps (`.[baselines]`) — their curve builders degrade to an empty
# curve (time_to_kkt = inf) when the package is not importable.
CVXPY_CURVE_SOLVERS: tuple[str, ...] = ("CLARABEL", "SCS", "OSQP", "ECOS")
NATIVE_CURVE_SOLVERS: tuple[str, ...] = ("sklearn", "adelie")

# Default panel: generic convex solvers PLUS the fast LASSO solvers, so the
# proposer's "bar to beat" includes adelie/sklearn, not only the conic panel.
DEFAULT_PANEL_SOLVERS: tuple[str, ...] = CVXPY_CURVE_SOLVERS + NATIVE_CURVE_SOLVERS


def problem_hash(problem) -> str:
    """Stable hash of the problem data (A, b, lambdas/lam)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(problem.A, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(problem.b, dtype=np.float64).tobytes())
    if isinstance(problem, LassoPath):
        h.update(np.ascontiguousarray(problem.lambdas, dtype=np.float64).tobytes())
    else:
        h.update(np.float64(problem.lam).tobytes())
    return h.hexdigest()[:16]


def _single_curve(prob: Lasso, solver: str, sweep: Sequence[int], metric=trusted_kkt) -> list[tuple[float, float]]:
    cvxprob, x_var = build_cvxpy_lasso(prob)  # canonicalize once, sweep on it
    pts: list[tuple[float, float]] = []
    for cap in sweep:
        x, wall = solve_existing_cvxpy(cvxprob, x_var, solver, max_iter=int(cap))
        if x is None:
            continue
        pts.append((wall, metric(prob, x)))
    pts.sort(key=lambda p: p[0])
    return pts


def _path_curve(prob: LassoPath, solver: str, sweep: Sequence[int], metric=trusted_kkt) -> list[tuple[float, float]]:
    # Build one reusable cvxpy problem per column (canonicalized once each).
    cols = []
    for k in range(prob.K):
        col_prob = Lasso(prob.A, prob.b, float(prob.lambdas[k]))
        cvxprob, x_var = build_cvxpy_lasso(col_prob)
        cols.append((cvxprob, x_var))

    pts: list[tuple[float, float]] = []
    for cap in sweep:
        # A column that fails / hasn't produced an iterate at this cap keeps its
        # zero vector — an honest "not yet converged" reading for that column —
        # rather than voiding the whole path point (one hard small-lambda column
        # must not collapse the entire cap's curve sample).
        X = np.zeros((prob.n, prob.K))
        total_t = 0.0
        for k, (cvxprob, x_var) in enumerate(cols):
            x, wall = solve_existing_cvxpy(cvxprob, x_var, solver, max_iter=int(cap))
            total_t += wall
            if x is not None:
                X[:, k] = x
        pts.append((total_t, metric(prob, X)))
    pts.sort(key=lambda p: p[0])
    return pts


# --- native fast-LASSO solvers (coordinate descent / screening) -------------
#
# Unlike the conic solvers, these have no per-iteration callback either, so we
# trace the same anytime curve by sweeping each solver's own iteration cap. They
# solve in the (1/2m) scikit/glmnet parameterization, so lambda maps as
# `alpha = lam / m`; the trusted KKT is always recomputed on the canonical
# problem, so the comparison stays on the one ruler. Import is lazy and guarded:
# a missing optional dep yields an empty curve (never crashes the panel).


def _sklearn_lasso_coef(prob: Lasso, cap: int) -> tuple[Optional[np.ndarray], float]:
    import warnings

    from sklearn.linear_model import Lasso as SkLasso

    model = SkLasso(
        alpha=prob.lam / prob.m, fit_intercept=False,
        max_iter=int(cap), tol=1e-14, selection="cyclic",
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # capped fits warn on non-convergence
        model.fit(prob.A, prob.b)
    return np.asarray(model.coef_), time.perf_counter() - t0


def _sklearn_path_coefs(problem: LassoPath, cap: int) -> tuple[Optional[np.ndarray], float]:
    """One warm-started `lasso_path` solve over the whole lambda path.

    This is sklearn's actual fast-path solver: coordinate descent that
    warm-starts each lambda from the previous (decreasing) lambda's solution —
    the same edge Adelie's native path solve has. Scoring the path with K cold
    per-column fits understated the fast-LASSO bar; this restores it.
    """
    import warnings

    from sklearn.linear_model import lasso_path

    # `problem.lambdas` are decreasing (glmnet convention); alpha = lam / m.
    # lasso_path warm-starts down a descending alpha grid and returns
    # coefs (n_features, n_alphas) aligned to that same descending order.
    alphas = np.asarray(problem.lambdas, dtype=float) / problem.m
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # capped path warns on non-convergence
        _, coefs, _ = lasso_path(
            problem.A, problem.b, alphas=alphas,
            max_iter=int(cap), tol=1e-14,
        )
    wall = time.perf_counter() - t0
    X = np.asarray(coefs)
    if X.shape != (problem.n, problem.K):
        return None, wall
    return X, wall


def _sklearn_curve(problem, sweep: Sequence[int], metric=trusted_kkt) -> list[tuple[float, float]]:
    try:
        import sklearn  # noqa: F401
    except Exception:  # noqa: BLE001 — optional dep absent
        return []
    pts: list[tuple[float, float]] = []
    if isinstance(problem, LassoPath):
        for cap in sweep:
            X, wall = _sklearn_path_coefs(problem, cap)
            if X is None:
                continue
            pts.append((wall, metric(problem, X)))
    else:
        for cap in sweep:
            x, wall = _sklearn_lasso_coef(problem, cap)
            pts.append((wall, metric(problem, x)))
    pts.sort(key=lambda p: p[0])
    return pts


def _adelie_curve(problem, sweep: Sequence[int], metric=trusted_kkt) -> list[tuple[float, float]]:
    """adelie grpnet anytime curve.

    For a path, adelie solves *all* lambdas in one `grpnet` call (screening +
    warm-start across decreasing lambdas — its actual edge), so the path curve
    is one native path solve per cap, not K independent column solves.
    """
    try:
        import adelie as ad
    except Exception:  # noqa: BLE001 — optional dep absent
        return []

    pts: list[tuple[float, float]] = []
    if isinstance(problem, LassoPath):
        A_f = np.asfortranarray(problem.A)
        lmda_path = [float(v) for v in (problem.lambdas / problem.m)]
        for cap in sweep:
            t0 = time.perf_counter()
            try:
                state = ad.solver.grpnet(
                    X=A_f, glm=ad.glm.gaussian(y=problem.b), intercept=False,
                    lmda_path=lmda_path, tol=1e-14, max_iters=int(cap),
                    progress_bar=False,
                )
            except Exception:  # noqa: BLE001 — capped solves may error
                continue
            wall = time.perf_counter() - t0
            betas = np.asarray(state.betas.toarray())  # (K, n)
            if betas.shape != (problem.K, problem.n):
                continue
            pts.append((wall, metric(problem, betas.T)))
    else:
        A_f = np.asfortranarray(problem.A)
        for cap in sweep:
            t0 = time.perf_counter()
            try:
                state = ad.solver.grpnet(
                    X=A_f, glm=ad.glm.gaussian(y=problem.b), intercept=False,
                    lmda_path=[problem.lam / problem.m], tol=1e-14,
                    max_iters=int(cap), progress_bar=False,
                )
            except Exception:  # noqa: BLE001
                continue
            wall = time.perf_counter() - t0
            # betas is a (n_lambda, n) scipy sparse matrix; negative-indexing a
            # sparse matrix raises, so densify first. At a low iteration cap
            # grpnet may return zero rows (didn't finish lambda 0) — skip those.
            betas = np.asarray(state.betas.toarray())
            if betas.shape != (1, problem.n):
                continue
            x = betas[-1]
            pts.append((wall, metric(problem, x)))
    pts.sort(key=lambda p: p[0])
    return pts


def baseline_kkt_time_curve(
    problem, solver: str, sweep: Optional[Sequence[int]] = None, metric=trusted_kkt,
) -> list[tuple[float, float]]:
    """Trace `(wall_time_s, metric)` for `solver` on `problem` across a cap sweep.

    `metric(problem, x)` is the optimality ruler for the curve's y-axis —
    `trusted_kkt` by default, `trusted_gap` for a duality-gap panel. One curve
    must use one ruler end to end; never mix curves built with different
    metrics on the same axis (their units and convergence scales differ).
    """
    sweep = tuple(sweep) if sweep is not None else DEFAULT_SWEEP
    if solver == "sklearn":
        return _sklearn_curve(problem, sweep, metric)
    if solver == "adelie":
        return _adelie_curve(problem, sweep, metric)
    if isinstance(problem, LassoPath):
        return _path_curve(problem, solver, sweep, metric)
    return _single_curve(problem, solver, sweep, metric)


def time_to_kkt(curve: Sequence[Sequence[float]], tol: float) -> float:
    """Smallest time on `curve` with kkt <= tol (log-linear interpolation)."""
    return time_to_target(curve, tol)


def baseline_panel(
    problem,
    *,
    solvers: Sequence[str] = DEFAULT_PANEL_SOLVERS,
    sweep: Optional[Sequence[int]] = None,
    cache_dir: Optional[Path] = None,
    metric=trusted_kkt,
    metric_tag: str = "kkt",
) -> dict[str, list[tuple[float, float]]]:
    """Curves for each solver, cached under `cache_dir/<hash>/<solver>[.<tag>].json`.

    `metric`/`metric_tag` select the y-axis ruler for the whole panel (default
    KKT residual; pass `trusted_gap`/"gap" for a duality-gap panel). The tag
    namespaces the on-disk cache so a gap panel never reuses a cached KKT panel.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    phash = problem_hash(problem)
    cdir = Path(cache_dir) / phash if cache_dir is not None else None
    suffix = "" if metric_tag == "kkt" else f".{metric_tag}"
    for solver in solvers:
        cpath = (cdir / f"{solver}{suffix}.json") if cdir is not None else None
        if cpath is not None and cpath.exists():
            out[solver] = [tuple(p) for p in json.loads(cpath.read_text())]
            continue
        curve = baseline_kkt_time_curve(problem, solver, sweep, metric)
        out[solver] = curve
        # Don't cache an empty curve: it means an optional solver (adelie/sklearn)
        # was absent, and we want a later run with it installed to retry.
        if cpath is not None and curve:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(curve))
    return out


def cached_adelie_curve(problem, npz_path) -> list[tuple[float, float]]:
    """One-point anytime curve for the cached Adelie reference solution.

    Adelie (glmnet-family coordinate descent) is the strong CPU baseline and the
    headline target on the wide hero shape, where the cvxpy interior-point panel
    is intractable. We cache its full-path solve `(X, wall_ms)` offline; here we
    score that single converged endpoint on the *same* trusted KKT ruler so it
    drops onto the gap-vs-time plot and into the `bar_to_beat` as one point
    `(wall_s, trusted_kkt(problem, X_adelie))`.
    """
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float64)
    if X.shape != (problem.n, problem.K):
        raise ValueError(
            f"adelie cache shape {X.shape} != problem (n,K)=({problem.n},{problem.K})"
        )
    wall_s = float(d["wall_ms"]) / 1000.0
    return [(wall_s, metric(problem, X))]


def best_baseline_time_to_kkt(
    panel: dict[str, Sequence[Sequence[float]]], tol: float,
) -> tuple[Optional[str], float]:
    """The fastest baseline (name, time) to reach `tol`. (None, inf) if none."""
    best_name: Optional[str] = None
    best_t = float("inf")
    for name, curve in panel.items():
        t = time_to_kkt(curve, tol)
        if t < best_t:
            best_name, best_t = name, t
    return best_name, best_t
