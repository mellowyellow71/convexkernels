"""Tests for the loop's Pareto archive (two-objective keep-rule + HV reward)."""
from __future__ import annotations

import math

from convexkernels.synth.pareto_archive import ParetoArchive


def _archive(**kw):
    # Fixed nadir keeps hand-computed areas stable across tests.
    return ParetoArchive(nadir=(10.0, 1.0), **kw)


def test_seed_then_dominated_candidate_rejected():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    res = a.consider([(5.0, 1e-1)])          # slower AND looser
    assert res["accepted"] is False
    assert res["advantage_over_archive"] == 0.0


def test_faster_looser_candidate_accepted():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    res = a.consider([(0.5, 1e-1)])          # faster, looser -> non-dominated
    assert res["accepted"] is True
    assert res["advantage_over_archive"] > 0.0


def test_slower_tighter_candidate_accepted():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    res = a.consider([(5.0, 1e-6)])          # slower, tighter -> non-dominated
    assert res["accepted"] is True


def test_consider_does_not_mutate_accept_does():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    hv0 = a.hypervolume()
    cand = [(5.0, 1e-6)]
    a.consider(cand)
    assert a.hypervolume() == hv0            # consider is a pure query
    a.accept(cand)
    assert a.hypervolume() > hv0
    assert len(a.frontier()) == 2


def test_curve_partially_dominated_still_accepted():
    # A curve whose tail is dominated but whose head is a new fast region.
    a = _archive()
    a.seed([(2.0, 1e-2)])
    res = a.consider([(0.5, 5e-1), (3.0, 1e-2)])
    assert res["accepted"] is True           # the (0.5, .5) head adds area


def test_reward_vs_panel():
    panel = {"adelie": [(8.0, 1e-6)], "scs": [(1.0, 1e-1)]}
    a = ParetoArchive(baselines=panel, nadir=(20.0, 1.0))
    a.seed([(6.0, 1e-3)])
    res = a.consider([(4.0, 1e-8)])          # tighter than adelie, faster
    assert res["accepted"] is True
    assert res["advantage_vs_panel"] is not None and res["advantage_vs_panel"] > 0.0
    assert res["dominates_panel"] is True
    assert res["beats_solver"]["adelie"] is True


def test_no_panel_reward_is_none():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    res = a.consider([(1.0, 1e-3)])
    assert res["advantage_vs_panel"] is None
    assert res["dominates_panel"] is None


def test_nadir_widens_with_accepted_points():
    a = _archive()
    a.seed([(2.0, 1e-2)])
    a.accept([(50.0, 2.0)])                  # worse than the initial nadir
    T, G = a.nadir
    assert T >= 50.0 and G >= 2.0


def test_nonfinite_and_invalid_points_ignored():
    a = _archive()
    a.seed([(2.0, 1e-2), (float("inf"), 1e-3), (1.0, 0.0), (-1.0, 1e-3)])
    assert len(a.frontier()) == 1            # only the valid point survives


def test_summary_fields():
    a = _archive()
    a.seed([(2.0, 1e-2), (5.0, 1e-6)])
    s = a.summary()
    assert s["hypervolume"] > 0.0
    assert math.isclose(s["frontier_best_gap"], 1e-6)
    assert math.isclose(s["frontier_fastest_time"], 2.0)
    assert len(s["frontier"]) == 2
