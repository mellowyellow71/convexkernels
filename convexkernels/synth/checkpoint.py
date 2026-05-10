"""Write-ahead log for synth-loop crash recovery.

Before each evaluation, the synth loop calls `mark_started(run_dir, ...)`
which writes `runs/<id>/started.json`. After evaluation, the lineage record
is appended to `lineage.jsonl`. On startup, `find_orphans()` returns the IDs
of run dirs that have `started.json` but no corresponding lineage record —
those are crashed evaluations.

Default policy: log orphans, do not requeue. The synth loop documents this
choice in its run header.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lineage import Edit, Slot, load_records, now_iso


def mark_started(
    run_dir: Path, slot: Slot, edit: Edit, source_path: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "slot": {
            "problem_family": slot.problem_family,
            "algorithm": slot.algorithm,
            "hardware": slot.hardware,
            "dtype": slot.dtype,
        },
        "edit": {
            "type": edit.type,
            "payload": edit.payload,
            "rationale": edit.rationale,
            "proposer_role": edit.proposer_role,
            "proposer_model": edit.proposer_model,
            "source": edit.source,
        },
        "source_path": source_path,
        "marked_at": now_iso(),
    }
    (run_dir / "started.json").write_text(json.dumps(payload, indent=2))


def find_orphans(runs_root: Path, lineage_path: Path) -> list[str]:
    """Return run-dir IDs with `started.json` but no corresponding lineage record."""
    completed_ids: set[str] = {rec["id"] for rec in load_records(lineage_path)}
    if not runs_root.exists():
        return []
    orphans: list[str] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "started.json").exists() and child.name not in completed_ids:
            orphans.append(child.name)
    return sorted(orphans)
