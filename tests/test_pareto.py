"""Tests for the (time, gap) Pareto / dominated-hypervolume scoring."""
from __future__ import annotations

import math

from convexkernels.bench.pareto import (
    auto_nadir,
    dominated_hypervolume,
    hypervolume_advantage,
    is_dominated,
    pareto_front,
    score_against_panel,
)

NADIR = (10.0, 1.0)   # log10(1) = 0 -> simple hand-computable areas


def test_pareto_front_drops_dominated_keeps_tradeoffs():
    # (2,0.1) fast/loose and (5,0.01) slow/tight are both non-dominated;
    # (6,0.5) is dominated by both.
    pts = [(2.0, 0.1), (5.0, 0.01), (6.0, 0.5)]
    front = sorted(pareto_front(pts))
    assert front == [(2.0, 0.1), (5.0, 0.01)]


def test_is_dominated():
    base = [(2.0, 0.01)]
    assert is_dominated((5.0, 0.1), base)        # worse on both
    assert not is_dominated((1.0, 0.1), base)    # faster
    assert not is_dominated((5.0, 0.001), base)  # tighter


def test_hypervolume_single_and_two_points():
    # one point (2, 0.1): (T-t)*(Ln - l) = (10-2)*(0-(-1)) = 8
    assert math.isclose(dominated_hypervolume([(2.0, 0.1)], NADIR), 8.0)
    # two points (2,0.1),(5,0.01): 3*1 + 5*2 = 13
    assert math.isclose(dominated_hypervolume([(2.0, 0.1), (5.0, 0.01)], NADIR), 13.0)


def test_hypervolume_ignores_dominated_and_out_of_range():
    # (6,0.5) is dominated; (12,...) is past the time nadir -> neither adds area.
    base = [(2.0, 0.1), (5.0, 0.01)]
    assert math.isclose(
        dominated_hypervolume(base + [(6.0, 0.5), (12.0, 0.001)], NADIR), 13.0
    )


def test_hypervolume_advantage_rewards_the_fast_region():
    # baseline only has the tight/slow point; candidate adds the fast/loose one.
    baseline = [(5.0, 0.01)]                       # HV = (10-5)*2 = 10
    candidate = [(2.0, 0.1)]                       # union HV = 13 -> adv = 3
    assert math.isclose(hypervolume_advantage(candidate, baseline, NADIR), 3.0)


def test_dominated_candidate_has_zero_advantage():
    baseline = [(2.0, 0.01)]                       # faster AND tighter
    candidate = [(5.0, 0.1)]                       # dominated -> adds nothing
    assert math.isclose(hypervolume_advantage(candidate, baseline, NADIR), 0.0)


def test_score_against_panel():
    # adelie-like: tight but slow; scs-like: fast but loose.
    panel = {
        "adelie": [(8.0, 1e-6)],
        "scs": [(1.0, 1e-1)],
    }
    # candidate dominates adelie's tight region faster, and is not as fast as scs
    candidate = [(4.0, 1e-8), (4.5, 1e-9)]
    s = score_against_panel(candidate, panel, nadir=(20.0, 1.0))
    assert s["advantage_vs_solver"]["adelie"] > 0.0     # beats adelie somewhere
    assert s["beats_solver"]["adelie"] is True
    assert s["advantage_vs_panel"] > 0.0                # adds area beyond the whole panel
    assert s["dominates_panel"] is True
    assert s["hv_candidate"] > 0.0


def test_auto_nadir_from_union():
    T, G = auto_nadir([(1.0, 0.5)], [(3.0, 0.2)])
    assert T == 3.0 and G == 0.5
