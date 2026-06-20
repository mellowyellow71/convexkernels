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


def build_research_state(
    *,
    lineage_rows: list[dict],
    baseline_times: dict[str, float],
    champion: Optional[dict],
    kkt_tol: float,
    max_ideas: int = 12,
    cost_model: Optional[dict] = None,
) -> dict:
    """Assemble the compact state dict from the durable tree + baselines.

    `cost_model` is an optional analytical bandwidth/AI hint for the active
    problem shape (see `synth.roofline.roofline_hint`); it steers the open
    algorithm search toward the bandwidth-favourable gradient form.
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
    return state


def write_research_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str))
