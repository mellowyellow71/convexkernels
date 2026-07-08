"""The single trusted optimality ruler for the autoresearch loop.

Every method — the candidate kernel and every classical baseline — is scored
by the *same* function computed on its iterate: the scale-free KKT residual of
the canonical numpy (fp64) frontend problem. No `f*` reference solve is needed
(a convex KKT residual is a self-certifying optimality certificate), so there
is no circularity between "the target" and "the baselines we want to beat".

`trusted_kkt(problem, x)` must always be called on the canonical frontend
problem (`Lasso`, `LassoPath`, ...), never on candidate-supplied code, so a
candidate cannot fake the metric.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def trusted_kkt(problem, x) -> float:
    """Scalar, scale-free KKT residual of `x` for `problem` (max over columns).

    Dispatches by the trusted frontend interface:
      - path problems expose `kkt_residual_max(X) -> float`,
      - single problems expose `kkt_residual(x) -> float`.
    """
    x = np.asarray(x, dtype=np.float64)
    if hasattr(problem, "kkt_residual_max"):
        return float(problem.kkt_residual_max(x))
    r = problem.kkt_residual(x)
    r_arr = np.asarray(r, dtype=np.float64)
    if r_arr.ndim == 0:
        return float(abs(r_arr))
    return float(np.max(np.abs(r_arr)))


def trusted_gap(problem, x) -> float:
    """Scalar oracle-free duality gap of `x` for `problem` (max over columns).

    The literal optimality gap Adelie/glmnet report, so a candidate and the
    baselines are compared on the *same* y-axis. Like `trusted_kkt`, must be
    called on the canonical frontend problem.

    Raises for problem types without a dual helper rather than silently falling
    back to the KKT residual: the two rulers have different units and
    convergence scales, and mixing them on one axis corrupts every comparison
    made against the curve (a gap panel must be gap end to end).
    """
    x = np.asarray(x, dtype=np.float64)
    if hasattr(problem, "duality_gap_max"):
        return float(problem.duality_gap_max(x))
    if hasattr(problem, "duality_gap"):
        return float(problem.duality_gap(x))
    raise TypeError(
        f"{type(problem).__name__} exposes no duality gap; refusing to fall "
        "back to the KKT residual (mixing rulers on one axis). Use trusted_kkt "
        "for this problem type, or add a duality_gap helper to its frontend."
    )


def time_to_target(points: Iterable[Sequence[float]], tol: float) -> float:
    """First wall-clock time at which KKT <= `tol`, log-linearly interpolated.

    `points` is an iterable of `(t, kkt)` in increasing `t`. Returns the
    interpolated crossing time, or `math.inf` if the curve never reaches `tol`.
    Interpolation is linear in `t` against `log(kkt)` between the last point
    above `tol` and the first point at/below it (the natural scale for a
    geometrically-converging residual).
    """
    prev: tuple[float, float] | None = None
    for pt in points:
        t = float(pt[0])
        k = float(pt[1])
        if k <= tol:
            if prev is None:
                return t
            t0, k0 = prev
            if k0 <= tol or k0 <= 0.0 or k <= 0.0:
                return t
            frac = (math.log(k0) - math.log(tol)) / (math.log(k0) - math.log(k))
            frac = min(max(frac, 0.0), 1.0)
            return t0 + frac * (t - t0)
        prev = (t, k)
    return math.inf
