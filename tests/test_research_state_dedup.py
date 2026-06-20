"""Curated-history dedup keys on algorithm_family, not the constant edit.type."""

from __future__ import annotations

from convexkernels.synth.research_state import _idea_key, build_research_state


def _row(fam, reason="discard:not_faster_than_champion", rationale="x", accepted=False):
    return {
        "id": "abcd1234",
        "edit": {"type": "full_source", "rationale": rationale, "algorithm_family": fam},
        "decision": {"accepted": accepted, "reason": reason},
        "score": {"time_to_kkt_s": 1.0, "kkt_final": 1e-7},
    }


def test_idea_key_uses_algorithm_family():
    assert _idea_key(_row("fista_gram")) == "fista_gram"


def test_idea_key_falls_back_to_normalized_rationale():
    row = {"edit": {"type": "full_source", "rationale": "Switch to ADMM   splitting"}}
    assert _idea_key(row) == "switch to admm splitting"


def test_idea_key_final_fallback_to_type():
    assert _idea_key({"edit": {"type": "full_source"}}) == "full_source"


def test_digest_keeps_distinct_families():
    rows = [_row(f"family_{i}") for i in range(6)]
    st = build_research_state(lineage_rows=rows, baseline_times={}, champion=None, kkt_tol=1e-6)
    ideas = {d["idea"] for d in st["tried_directions"]}
    assert len(ideas) == 6  # previously all 6 collapsed to a single "full_source" bucket


def test_digest_collapses_same_family_same_outcome():
    rows = [_row("fista_gram") for _ in range(5)]
    st = build_research_state(lineage_rows=rows, baseline_times={}, champion=None, kkt_tol=1e-6)
    fista = [d for d in st["tried_directions"] if d["idea"] == "fista_gram"]
    assert len(fista) == 1  # same family + same outcome prefix dedups
