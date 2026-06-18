"""Anytime KKT-residual-vs-time curves for the classical baseline panel.

The autoresearch headline is "reach a target optimality faster than the
classical solvers." To compare apples-to-apples, every baseline is reduced to
the same anytime curve the candidate produces: a list of `(wall_time_s, kkt)`
points, with `kkt = trusted_kkt(problem, x)` — the identical ruler used for the
candidate (see `bench/metrics.py`).

cvxpy exposes no portable per-iteration callback, so the curve is traced by
solving the baseline at a sweep of iteration caps (`max_iter ∈ sweep`); a larger
cap costs more time and reaches a smaller residual. This is the standard
anytime-benchmark construction when callbacks are unavailable. Curves are
cached per problem on disk (baselines do not change between sessions).

For a `LassoPath`, the path is scored as a whole: at iteration budget `N`, each
column is solved (as a single LASSO) at cap `N`, the per-column solve times are
summed (sequential path solve), and the trusted *max-over-columns* KKT of the
assembled `(n, K)` matrix is the curve's `kkt` value at that total time.
"""

from __future__ import annotations

import hashlib
import json
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


def _single_curve(prob: Lasso, solver: str, sweep: Sequence[int]) -> list[tuple[float, float]]:
    cvxprob, x_var = build_cvxpy_lasso(prob)  # canonicalize once, sweep on it
    pts: list[tuple[float, float]] = []
    for cap in sweep:
        x, wall = solve_existing_cvxpy(cvxprob, x_var, solver, max_iter=int(cap))
        if x is None:
            continue
        pts.append((wall, trusted_kkt(prob, x)))
    pts.sort(key=lambda p: p[0])
    return pts


def _path_curve(prob: LassoPath, solver: str, sweep: Sequence[int]) -> list[tuple[float, float]]:
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
        pts.append((total_t, trusted_kkt(prob, X)))
    pts.sort(key=lambda p: p[0])
    return pts


def baseline_kkt_time_curve(
    problem, solver: str, sweep: Optional[Sequence[int]] = None,
) -> list[tuple[float, float]]:
    """Trace `(wall_time_s, kkt)` for `solver` on `problem` across a cap sweep."""
    sweep = tuple(sweep) if sweep is not None else DEFAULT_SWEEP
    if isinstance(problem, LassoPath):
        return _path_curve(problem, solver, sweep)
    return _single_curve(problem, solver, sweep)


def time_to_kkt(curve: Sequence[Sequence[float]], tol: float) -> float:
    """Smallest time on `curve` with kkt <= tol (log-linear interpolation)."""
    return time_to_target(curve, tol)


def baseline_panel(
    problem,
    *,
    solvers: Sequence[str] = ("CLARABEL", "SCS", "OSQP", "ECOS"),
    sweep: Optional[Sequence[int]] = None,
    cache_dir: Optional[Path] = None,
) -> dict[str, list[tuple[float, float]]]:
    """Curves for each solver, cached under `cache_dir/<hash>/<solver>.json`."""
    out: dict[str, list[tuple[float, float]]] = {}
    phash = problem_hash(problem)
    cdir = Path(cache_dir) / phash if cache_dir is not None else None
    for solver in solvers:
        cpath = (cdir / f"{solver}.json") if cdir is not None else None
        if cpath is not None and cpath.exists():
            out[solver] = [tuple(p) for p in json.loads(cpath.read_text())]
            continue
        curve = baseline_kkt_time_curve(problem, solver, sweep)
        out[solver] = curve
        if cpath is not None:
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
    return [(wall_s, trusted_kkt(problem, X))]


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
