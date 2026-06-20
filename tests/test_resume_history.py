"""On resume, history is repopulated from the durable lineage log (per slot)."""

from __future__ import annotations

import json
from pathlib import Path

from convexkernels.synth.loop import _resume_history, _slot_to_dict
from convexkernels.synth.lineage import Slot


def _write_lineage(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _slot():
    return Slot(problem_family="lasso_path", algorithm="open", hardware="apple_silicon", dtype="open")


def test_fresh_run_returns_empty(tmp_path):
    lp = tmp_path / "lineage.jsonl"
    _write_lineage(lp, [{"slot": _slot_to_dict(_slot()), "source": {"hash": "h1"}}])
    assert _resume_history(lp, _slot(), resume_from=None) == []


def test_missing_log_returns_empty(tmp_path):
    assert _resume_history(tmp_path / "nope.jsonl", _slot(), resume_from="ck1") == []


def test_resume_reloads_matching_slot_rows(tmp_path):
    lp = tmp_path / "lineage.jsonl"
    other = _slot_to_dict(Slot("nonneg_lasso", "open", "apple_silicon", "open"))
    rows = [
        {"slot": _slot_to_dict(_slot()), "source": {"hash": "h1"}, "edit": {"algorithm_family": "fista_gram"}},
        {"slot": other, "source": {"hash": "h2"}},  # different slot -> excluded
        {"slot": _slot_to_dict(_slot()), "source": {"hash": "h3"}, "edit": {"algorithm_family": "admm"}},
    ]
    _write_lineage(lp, rows)
    out = _resume_history(lp, _slot(), resume_from="ck1")
    assert len(out) == 2
    hashes = {r["source"]["hash"] for r in out}
    assert hashes == {"h1", "h3"}  # the other slot's h2 is filtered out


def test_reloaded_history_enables_duplicate_guard(tmp_path):
    # The dup-guard scans history for matching source hashes; reloaded rows
    # therefore make a previously-tried source a duplicate on resume.
    lp = tmp_path / "lineage.jsonl"
    _write_lineage(lp, [{"slot": _slot_to_dict(_slot()), "source": {"hash": "sha256:dead"}}])
    history = _resume_history(lp, _slot(), resume_from="ck1")
    seen = any((r.get("source") or {}).get("hash") == "sha256:dead" for r in history)
    assert seen
