"""Strategic Director agent — the layer above the Proposer.

The Proposer writes a faster `solve()` from whatever branch point and direction
it is handed. It has no view of the *search* — it always worked from the current
champion and chose its own direction, which on a saturated shape collapses into a
rut of near-identical micro-edits.

The Director sits above it. Each step it reads the **bounded** curated state
(`research_state`) plus a **bounded** summary of the durable checkpoint tree
(`research_state.build_tree_summary`) — never raw lineage, so the anti-context-rot
property is preserved — and emits a structured `Directive`:

  - which checkpoint to branch from (enabling non-greedy / backtracking search
    over the existing `CheckpointStore` tree, not just champion-chaining),
  - a concrete search direction / lever to pursue,
  - a strategic signal (explore / exploit / backtrack / pivot / stop).

The Director never evaluates anything: the trusted-KKT sandbox stays the sole,
deterministic accept/reject gate. The directive changes *where* and *what* we
search, never the bar.

`--director off` uses `StubDirector`, whose default `Directive()` reproduces the
loop's previous greedy-from-champion behavior exactly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_REASONING_EFFORT = "medium"
_DEFAULT_API_TIMEOUT_S = 240.0

CHAMPION = "champion"  # sentinel: branch from the current global champion
VALID_SIGNALS = ("explore", "exploit", "backtrack", "pivot", "stop")


@dataclass
class Directive:
    """A strategic instruction from the Director for the next proposal(s)."""
    branch_from: str = CHAMPION          # checkpoint id, or the CHAMPION sentinel
    direction: str = ""                  # concrete lever/hypothesis to pursue
    algorithm_family_hint: str = ""      # snake_case tag the proposer should aim for
    rationale: str = ""
    signal: str = "exploit"              # explore|exploit|backtrack|pivot|stop
    saturated: bool = False

    def is_default(self) -> bool:
        """True when this is the no-op directive (== today's greedy behavior)."""
        return (
            self.branch_from in (CHAMPION, "")
            and not self.direction.strip()
            and self.signal in ("exploit", "")
        )


class Director(Protocol):
    def direct(self, state: dict) -> Directive: ...


class StubDirector:
    """Always 'branch from champion, free choice' — i.e. today's behavior.

    Used by `--director off`/`stub` and as the fallback when an LLM Director
    errors, so a Director failure degrades to the greedy loop rather than
    aborting the run.
    """

    def direct(self, state: dict) -> Directive:  # noqa: ARG002 — state unused by design
        return Directive()


def _known_ids(state: dict) -> set[str]:
    tree = state.get("tree_summary") or {}
    return {str(n.get("id")) for n in (tree.get("nodes") or []) if n.get("id")}


def coerce_directive(raw: dict, state: dict) -> Directive:
    """Build a validated Directive from a parsed dict.

    Defensive: an unknown `branch_from` is coerced to the CHAMPION sentinel (the
    loop must never branch from a hallucinated id), and an unknown `signal`
    falls back to "exploit".
    """
    branch = str(raw.get("branch_from") or CHAMPION).strip()
    if branch != CHAMPION and branch not in _known_ids(state):
        branch = CHAMPION
    signal = str(raw.get("signal") or "exploit").strip().lower()
    if signal not in VALID_SIGNALS:
        signal = "exploit"
    return Directive(
        branch_from=branch,
        direction=str(raw.get("direction") or "").strip(),
        algorithm_family_hint=str(raw.get("algorithm_family_hint") or "").strip(),
        rationale=str(raw.get("rationale") or "").strip(),
        signal=signal,
        saturated=bool(raw.get("saturated", False)),
    )


class OpenAIDirector:
    """Calls the OpenAI Responses API and parses a structured `Directive`.

    Mirrors `proposers.openai.OpenAIProposer`: one stateless call per step, a
    strict json_schema response, no persistent memory. The only history it sees
    is the bounded `tree_summary` + `research_state` handed in via `state`.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
        api_timeout_s: float = _DEFAULT_API_TIMEOUT_S,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_timeout_s = float(api_timeout_s)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client  # tests inject a mock

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed; install with `pip install -e .[ai]`"
            ) from exc
        self._client = OpenAI(api_key=self._api_key, timeout=self.api_timeout_s)
        return self._client

    def direct(self, state: dict) -> Directive:
        # Lazy import avoids a loop->director->proposers.openai->loop cycle.
        from .proposers.openai import _extract_response_text

        prompt = _format_director_prompt(state)
        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "director_directive",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "branch_from", "direction", "algorithm_family_hint",
                            "rationale", "signal", "saturated",
                        ],
                        "properties": {
                            "branch_from": {
                                "type": "string",
                                "description": (
                                    "The checkpoint id (from the experiment tree) to branch the "
                                    "next proposal from, or the literal 'champion' to use the "
                                    "current global champion. Use a non-champion node to backtrack "
                                    "to an earlier, more promising line."
                                ),
                            },
                            "direction": {
                                "type": "string",
                                "description": (
                                    "One concrete lever / hypothesis the proposer should pursue "
                                    "next, e.g. 'warm-start across the decreasing-lambda path', "
                                    "'fp16 inner matmul with fp32 KKT check', '8-bit quantized "
                                    "full-gradient violation check inside the active set'. Be "
                                    "specific; avoid directions already shown failing in the digest."
                                ),
                            },
                            "algorithm_family_hint": {
                                "type": "string",
                                "description": (
                                    "Short snake_case tag for the approach you want tried, e.g. "
                                    "'screening_fista', 'admm', 'fp16_inner'. Steers the dedup key."
                                ),
                            },
                            "rationale": {
                                "type": "string",
                                "description": "One paragraph: why this branch + direction now.",
                            },
                            "signal": {
                                "type": "string",
                                "enum": list(VALID_SIGNALS),
                                "description": (
                                    "Strategic mode: 'exploit' (refine the champion line), "
                                    "'explore' (a new lever), 'backtrack' (branch from an earlier "
                                    "node), 'pivot' (change algorithm family), or 'stop' (the "
                                    "search is saturated and further proposals are not worthwhile)."
                                ),
                            },
                            "saturated": {
                                "type": "boolean",
                                "description": (
                                    "True if you judge the current line saturated (recent "
                                    "proposals are near-identical micro-edits with no gain)."
                                ),
                            },
                        },
                    },
                    "strict": True,
                },
            },
        )
        text = _extract_response_text(response)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"director response not valid JSON: {exc}: {text[:500]}")
        return coerce_directive(parsed, state)


# ---------- prompt assembly ----------


def _format_director_prompt(state: dict) -> str:
    rs = state.get("research_state") or {}
    tree = state.get("tree_summary") or {}
    champ = rs.get("champion") or {}
    bar = rs.get("bar_to_beat") or {}
    tried = rs.get("tried_directions") or []
    notes = rs.get("analyst_notes") or []
    kkt_tol = state.get("kkt_tol", rs.get("kkt_tol", 1e-6))

    sections: list[str] = []
    sections.append(
        "You are the STRATEGIST (Director) of a KKT-verified kernel-autoresearch "
        "loop. You do NOT write code. Each step you decide, from the bounded "
        "experiment-tree summary below, (1) which checkpoint to branch the next "
        "proposal from and (2) one concrete direction for the implementer to "
        "pursue. A separate deterministic evaluator verifies optimality, so you "
        "cannot fake progress; your only job is to steer the search well — avoid "
        "repeating directions the digest shows failing, and backtrack to an "
        "earlier node or declare saturation when a line stops improving."
    )

    sections.append("--- current champion ---")
    sections.append(
        "total_time_s={} time_to_kkt_s={} kkt_final={}".format(
            champ.get("total_time_s"), champ.get("time_to_kkt_s"), champ.get("kkt_final"),
        )
    )

    sections.append("--- bar to beat (classical baselines, time_to_kkt_s) ---")
    sections.append(
        "best: {} @ {}; all: {}".format(
            bar.get("best_baseline"),
            bar.get("best_baseline_time_to_kkt_s"),
            json.dumps(bar.get("all_baselines_time_to_kkt_s") or {}, default=str),
        )
    )

    sections.append("--- experiment tree (checkpoint nodes you may branch from) ---")
    nodes = tree.get("nodes") or []
    if nodes:
        for n in nodes:
            sections.append(json.dumps(n, default=str))
    else:
        sections.append("(only the seed so far)")

    sections.append("--- tried directions (curated; do not repeat failures) ---")
    if tried:
        for t in tried:
            sections.append(json.dumps(t, default=str))
    else:
        sections.append("(none yet)")

    if notes:
        sections.append("--- analyst notes (advisory: why recent candidates failed) ---")
        for nt in notes:
            sections.append(str(nt))

    cm = rs.get("hardware_cost_model") or {}
    if cm:
        shape = cm.get("shape") or {}
        sections.append("--- hardware cost model ---")
        sections.append(
            "shape m={} n={} regime={}; levers: {}".format(
                shape.get("m"), shape.get("n"), shape.get("regime"),
                json.dumps(cm.get("levers") or [], default=str),
            )
        )

    sections.append("--- output ---")
    sections.append(
        "Return JSON with: branch_from (a node id above, or 'champion'), "
        "direction, algorithm_family_hint, rationale, signal "
        "({}), saturated. Target trusted KKT < {:.1e}. "
        "No markdown, no text outside the JSON.".format(
            "|".join(VALID_SIGNALS), kkt_tol,
        )
    )
    return "\n\n".join(sections)


__all__ = [
    "Directive", "Director", "StubDirector", "OpenAIDirector",
    "coerce_directive", "CHAMPION", "VALID_SIGNALS",
]
