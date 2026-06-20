"""Curated research state — the anti-context-rot mechanism.

Instead of replaying a growing raw lineage into the proposer prompt (which
saturates and rots), the loop rebuilds a compact, bounded summary from the
durable experiment tree each iteration:

  - the current champion (algorithm tag, time-to-target, kkt),
  - the bar to beat: ranked baseline times-to-target,
  - a deduplicated digest of tried directions (algorithm family + one-line
    outcome + why), capped so the prompt stays small.

This summary, plus the current checkpoint's source, is all the proposer sees of
history — durable, progress-aware, and bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _idea_key(row: dict) -> str:
    """Coarse dedup signature for a tried direction.

    Keys on the proposer's `algorithm_family` tag. `edit.type` is always
    "full_source" in the open-search loop, so keying on it collapsed the entire
    discard history into one bucket and starved the proposer of negative signal.
    Falls back to a normalized rationale prefix for older/untagged rows.
    """
    edit = row.get("edit") or {}
    fam = str(edit.get("algorithm_family") or "").strip().lower()
    if fam:
        return fam
    rationale = " ".join(str(edit.get("rationale") or "").lower().split())
    if rationale:
        return rationale[:48]
    return str(edit.get("type") or "other").strip().lower()


def build_tree_summary(
    checkpoints: list,
    *,
    problem_hash: str,
    max_nodes: int = 20,
) -> dict:
    """Bounded summary of the branchable checkpoint tree for the Director.

    One row per checkpoint (filtered to `problem_hash`): the full `id` (the
    Director must return a real id to branch from), `parent_id`, `algorithm_tag`,
    the score triple, and `n_children` (so the Director can see which lines are
    saturated/already-explored). Capped at `max_nodes` — keep the root plus the
    best-N by `total_time_s` — to preserve the anti-context-rot bound. This is the
    ONLY view of history the Director gets beyond the curated digest, and it is
    still derived from the durable tree, never raw lineage.
    """
    nodes = [c for c in checkpoints if getattr(c, "problem_hash", None) == problem_hash]
    child_count: dict[str, int] = {}
    for c in nodes:
        pid = getattr(c, "parent_id", None)
        if pid:
            child_count[pid] = child_count.get(pid, 0) + 1

    def _row(c) -> dict:
        return {
            "id": c.id,
            "parent_id": c.parent_id,
            "algorithm_tag": c.algorithm_tag,
            "total_time_s": c.total_time_s,
            "time_to_kkt_s": c.time_to_kkt_s,
            "kkt_final": c.kkt_final,
            "n_children": child_count.get(c.id, 0),
        }

    if len(nodes) > max_nodes:
        roots = [c for c in nodes if not c.parent_id]
        rest = [c for c in nodes if c.parent_id]
        rest.sort(key=lambda c: (c.total_time_s if c.total_time_s is not None else float("inf")))
        kept = roots + rest[: max(0, max_nodes - len(roots))]
        keep_ids = {c.id for c in kept}
        nodes = [c for c in nodes if c.id in keep_ids]

    return {"nodes": [_row(c) for c in nodes], "n_total": len(checkpoints)}


def build_research_state(
    *,
    lineage_rows: list[dict],
    baseline_times: dict[str, float],
    champion: Optional[dict],
    kkt_tol: float,
    max_ideas: int = 12,
    cost_model: Optional[dict] = None,
    analyst_notes: Optional[list] = None,
    max_notes: int = 8,
) -> dict:
    """Assemble the compact state dict from the durable tree + baselines.

    `cost_model` is an optional analytical bandwidth/AI hint for the active
    problem shape (see `synth.roofline.roofline_hint`); it steers the open
    algorithm search toward the bandwidth-favourable gradient form.

    `analyst_notes` is an optional bounded list of advisory one-line summaries of
    why recent candidates failed (see `synth.analyst`); the newest `max_notes`
    are surfaced to the Director. Advisory only — never affects the gate.
    """
    ranked_baselines = sorted(
        ((name, t) for name, t in baseline_times.items()),
        key=lambda kv: kv[1],
    )
    best_baseline = ranked_baselines[0] if ranked_baselines else (None, float("inf"))

    # Deduplicated digest: keep all accepted, plus the most recent distinct
    # discarded ideas, newest first, capped at max_ideas.
    accepted: list[dict] = []
    discarded: list[dict] = []
    seen: set[str] = set()
    for row in reversed(lineage_rows):
        decision = row.get("decision") or {}
        score = row.get("score") or {}
        entry = {
            "id": str(row.get("id", ""))[:8],
            "idea": _idea_key(row),
            "rationale": (row.get("edit") or {}).get("rationale", "")[:160],
            "outcome": decision.get("reason", ""),
            "time_to_kkt_s": score.get("time_to_kkt_s"),
            "kkt_final": score.get("kkt_final"),
        }
        if decision.get("accepted"):
            accepted.append(entry)
            continue
        key = entry["idea"] + "|" + entry["outcome"].split(":", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        discarded.append(entry)

    digest = accepted + discarded[: max(0, max_ideas - len(accepted))]

    state = {
        "kkt_tol": kkt_tol,
        "champion": champion,
        "bar_to_beat": {
            "best_baseline": best_baseline[0],
            "best_baseline_time_to_kkt_s": best_baseline[1],
            "all_baselines_time_to_kkt_s": dict(ranked_baselines),
        },
        "tried_directions": digest,
        "n_experiments": len(lineage_rows),
        "n_accepted": sum(
            1 for r in lineage_rows if (r.get("decision") or {}).get("accepted")
        ),
    }
    if cost_model is not None:
        state["hardware_cost_model"] = cost_model
    if analyst_notes:
        state["analyst_notes"] = list(analyst_notes)[-max_notes:]
    return state


def write_research_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))
