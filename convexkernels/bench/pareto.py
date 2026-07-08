"""Two-objective scoring for the autoresearch loop: the (wall-clock, gap) plane.

Unlike a single-objective kernel search (throughput at fixed correctness) or the
earlier time-to-fixed-tolerance score, a solver is judged here on **two
independent axes at once**: wall-clock time (x) and optimality gap (y, the
oracle-free duality gap — see `algorithms.gap.lasso_duality_gap`). Each solver
run is an *anytime curve* of ``(time_s, gap)`` points, and the object of
interest is the lower-left **Pareto frontier** over all curves.

Keep rule (the archive): a candidate is admitted iff it is **not strictly
dominated** by any existing point — i.e. it is the best solver for *some* region
of the plane. A fast-to-1e-3 low-precision kernel and a slow-to-1e-12 fp64
kernel can therefore both belong on the frontier; they win different regimes.

Search reward (where to look next): **dominated hypervolume relative to a
baseline curve** (e.g. Adelie's). It is the area of the (time, log10-gap) plane
the candidate dominates that the baseline does not — the multi-objective analog
of a single speedup number. Maximizing it drives proposals into the baseline's
territory. Gaps span decades, so the y-axis is measured in ``log10(gap)``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

Point = Sequence[float]          # (time_s, gap)
Curve = Iterable[Point]

_GAP_FLOOR = 1e-16               # clip gap before log10 (gap==0 at the optimum)


def _clean(points: Curve) -> list[tuple[float, float]]:
    """Finite (time>=0) points as (time, log10(gap)), gap clamped to a 1e-16 floor."""
    out: list[tuple[float, float]] = []
    for p in points:
        t, g = float(p[0]), float(p[1])
        if not (math.isfinite(t) and math.isfinite(g)) or t < 0.0:
            continue
        out.append((t, math.log10(max(g, _GAP_FLOOR))))
    return out


def _front(tl_points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated set for minimization on (time, log-gap), as a staircase.

    Sorted by time ascending; ties keep the smaller log-gap. A point survives
    iff its log-gap is strictly below the running minimum.
    """
    if not tl_points:
        return []
    pts = sorted(tl_points, key=lambda p: (p[0], p[1]))
    front: list[tuple[float, float]] = []
    best_l = math.inf
    for t, l in pts:
        if l < best_l:
            front.append((t, l))
            best_l = l
    return front


def pareto_front(points: Curve) -> list[tuple[float, float]]:
    """Public: non-dominated ``(time_s, gap)`` points (minimize time and gap)."""
    front_tl = _front(_clean(points))
    return [(t, 10.0 ** l) for t, l in front_tl]


def is_dominated(point: Point, by: Curve) -> bool:
    """True iff some point in `by` is strictly better on BOTH time and gap."""
    t, g = float(point[0]), float(point[1])
    for q in by:
        tq, gq = float(q[0]), float(q[1])
        if tq < t and gq < g:
            return True
    return False


def auto_nadir(
    *curves: Curve, time: Optional[float] = None, gap: Optional[float] = None,
) -> tuple[float, float]:
    """Reference (worst) point for hypervolume: max time, max gap over all curves.

    A common nadir across the candidate and the whole panel is what makes
    per-solver hypervolume advantages comparable. Pass `time`/`gap` to pin it.
    """
    ts: list[float] = []
    gs: list[float] = []
    for c in curves:
        for p in c:
            t, g = float(p[0]), float(p[1])
            if math.isfinite(t) and math.isfinite(g) and g > 0.0:
                ts.append(t)
                gs.append(g)
    T = time if time is not None else (max(ts) if ts else 1.0)
    G = gap if gap is not None else (max(gs) if gs else 1.0)
    return float(T), float(G)


def dominated_hypervolume(points: Curve, nadir: tuple[float, float]) -> float:
    """Area of the (time, log10-gap) plane dominated by `points` toward `nadir`.

    `nadir = (T, G)` is the worst reference point (larger time and gap are
    worse). Units: seconds × log10-gap-decades. Monotone: adding a
    non-dominated point never decreases it.
    """
    T, Ln = float(nadir[0]), math.log10(max(float(nadir[1]), _GAP_FLOOR))
    # keep only points strictly better than the nadir on both axes
    inside = [(t, l) for t, l in _clean(points) if t < T and l < Ln]
    front = _front(inside)
    if not front:
        return 0.0
    hv = 0.0
    for i, (t, l) in enumerate(front):
        t_next = front[i + 1][0] if i + 1 < len(front) else T
        hv += (t_next - t) * (Ln - l)
    return float(hv)


def hypervolume_advantage(
    candidate: Curve, baseline: Curve, nadir: tuple[float, float],
) -> float:
    """Hypervolume the candidate adds beyond the baseline: HV(cand∪base) − HV(base).

    >0 iff the candidate is non-dominated (better) somewhere the baseline is the
    incumbent — the scalar reward for pushing the frontier into the baseline's
    territory. Always >= 0 by monotonicity of the union.
    """
    cand = list(candidate)
    base = list(baseline)
    hv_base = dominated_hypervolume(base, nadir)
    hv_union = dominated_hypervolume(cand + base, nadir)
    return float(hv_union - hv_base)


def score_against_panel(
    candidate: Curve,
    panel: dict[str, Curve],
    *,
    nadir: Optional[tuple[float, float]] = None,
) -> dict:
    """Score a candidate curve against a panel of baseline solver curves.

    Returns per-solver hypervolume advantage, the advantage vs the *combined*
    panel frontier (the real "how do we do against all of them"), the candidate
    and panel hypervolumes, and a `dominates_panel` flag (candidate claims area
    no panel solver reaches). All HVs use one shared `nadir` (auto by default).
    """
    cand = list(candidate)
    panel_lists = {name: list(c) for name, c in panel.items()}
    if nadir is None:
        nadir = auto_nadir(cand, *panel_lists.values())

    union_panel: list = []
    for c in panel_lists.values():
        union_panel.extend(c)

    per_solver = {
        name: hypervolume_advantage(cand, c, nadir)
        for name, c in panel_lists.items()
    }
    hv_candidate = dominated_hypervolume(cand, nadir)
    hv_panel = dominated_hypervolume(union_panel, nadir)
    adv_vs_panel = hypervolume_advantage(cand, union_panel, nadir)
    return {
        "nadir": nadir,
        "hv_candidate": hv_candidate,
        "hv_panel": hv_panel,
        "advantage_vs_solver": per_solver,
        "advantage_vs_panel": adv_vs_panel,
        "dominates_panel": adv_vs_panel > 0.0,
        "beats_solver": {name: adv > 0.0 for name, adv in per_solver.items()},
    }


__all__ = [
    "pareto_front",
    "is_dominated",
    "auto_nadir",
    "dominated_hypervolume",
    "hypervolume_advantage",
    "score_against_panel",
]
