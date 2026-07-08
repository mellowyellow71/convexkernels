"""Pareto archive: the two-objective keep-rule + hypervolume reward for the loop.

In the single-champion loop a candidate is kept iff it reaches a fixed tolerance
faster than the incumbent — a *scalar* race. Here the loop instead maintains the
**Pareto frontier** of anytime ``(wall_time_s, optimality_gap)`` curves: a
candidate is admitted iff its curve is **non-dominated** — i.e. it is the best
solver for *some* region of the (time, gap) plane, so it adds dominated
hypervolume over what we already keep. A fast-to-1e-3 kernel and a slow-to-1e-12
kernel can both belong on the frontier.

The same hypervolume machinery yields the **search reward** handed to the
proposer: ``advantage_vs_panel`` is the area the candidate claims that the whole
classical baseline panel (Adelie/SCS/ECOS/...) does not — the multi-objective
analog of a single speedup number, and the signal to push proposals into the
baselines' territory.

All geometry lives in `bench.pareto`; this class is the stateful session archive
(fixed reference nadir, accumulated frontier points, baseline panel for reward).
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..bench.pareto import (
    auto_nadir,
    dominated_hypervolume,
    pareto_front,
    score_against_panel,
)

Point = Sequence[float]          # (wall_time_s, gap)


class ParetoArchive:
    """Stateful (time, gap) Pareto frontier with a fixed hypervolume reference."""

    def __init__(
        self,
        *,
        baselines: Optional[dict[str, list[Point]]] = None,
        nadir: Optional[tuple[float, float]] = None,
        min_advantage: float = 1e-9,
    ):
        self.baselines = {k: list(v) for k, v in (baselines or {}).items() if v}
        self.min_advantage = float(min_advantage)
        self._points: list[tuple[float, float]] = []
        # Reference (worst) point. Derived from the baseline panel, widened ONCE
        # by the first seed() to also bound the starting champion's curve, then
        # FROZEN: every hypervolume/advantage after that is measured against the
        # same reference, so session-wide numbers are comparable and the reward
        # cannot be inflated by a later, worse-but-nadir-widening curve.
        self._frozen = False
        if nadir is not None:
            self._nadir = (float(nadir[0]), float(nadir[1]))
        else:
            self._nadir = auto_nadir(*self.baselines.values()) if self.baselines else (1.0, 1.0)

    # ---- reference / state ----
    @property
    def nadir(self) -> tuple[float, float]:
        return self._nadir

    def _widen_nadir(self, points: list[Point]) -> None:
        """Grow the reference point so it stays worse than everything seen.

        No-op once the reference is frozen (after the first seed): the whole
        point of the fixed reference is that it does not move under later curves.
        """
        if self._frozen:
            return
        T, G = self._nadir
        nt, ng = auto_nadir(points, time=None, gap=None)
        self._nadir = (max(T, nt), max(G, ng))

    def seed(self, points: list[Point]) -> None:
        """Initialise the frontier (e.g. with the starting champion's curve).

        This is the last time the reference moves: it widens to bound the seed
        curve, then freezes for the rest of the session.
        """
        pts = _clean_points(points)
        self._widen_nadir(pts)
        self._points.extend(pts)
        self._frozen = True

    def frontier(self) -> list[tuple[float, float]]:
        return pareto_front(self._points)

    def hypervolume(self) -> float:
        return dominated_hypervolume(self._points, self._nadir)

    # ---- the keep-rule + reward ----
    def consider(self, points: list[Point]) -> dict:
        """Evaluate a candidate curve WITHOUT mutating the archive.

        Returns the keep decision (does it add frontier hypervolume?), the
        archive-relative advantage, and the reward signal vs the baseline panel.
        """
        pts = _clean_points(points)
        hv_before = dominated_hypervolume(self._points, self._nadir)
        hv_after = dominated_hypervolume(self._points + pts, self._nadir)
        advantage = hv_after - hv_before
        panel = score_against_panel(pts, self.baselines, nadir=self._nadir) if self.baselines else None
        return {
            "accepted": advantage > self.min_advantage,
            "nadir": self._nadir,               # logged per decision (now fixed)
            "advantage_over_archive": float(advantage),
            "hv_archive_before": float(hv_before),
            "hv_archive_after": float(hv_after),
            "advantage_vs_panel": (panel["advantage_vs_panel"] if panel else None),
            "dominates_panel": (panel["dominates_panel"] if panel else None),
            "beats_solver": (panel["beats_solver"] if panel else None),
        }

    def accept(self, points: list[Point]) -> None:
        """Commit a candidate curve to the frontier (call after a keep).

        Does NOT move the reference — it was frozen at seed, so accepting a
        curve cannot retroactively change the hypervolume of earlier decisions.
        A curve extending past the frozen nadir simply claims no area out there.
        """
        pts = _clean_points(points)
        self._points.extend(pts)

    def summary(self) -> dict:
        front = self.frontier()
        return {
            "nadir": self._nadir,
            "hypervolume": self.hypervolume(),
            "frontier": front,
            "frontier_best_gap": (min(g for _, g in front) if front else None),
            "frontier_fastest_time": (min(t for t, _ in front) if front else None),
            "n_baselines": len(self.baselines),
        }


def _clean_points(points: list[Point]) -> list[tuple[float, float]]:
    import math
    out: list[tuple[float, float]] = []
    for p in points:
        try:
            t, g = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(t) and math.isfinite(g) and t >= 0.0 and g > 0.0:
            out.append((t, g))
    return out


__all__ = ["ParetoArchive"]
