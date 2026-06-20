"""Advisory Analyst agent — explains failures to the Director.

After a candidate is discarded, the Analyst produces ONE short line on *why*
(e.g. "plateaued at kkt~2e-3 — quantization floor; needs fp32 polish", "slower
than champion: extra full-gradient scan dominates"). These notes accumulate
(bounded) into `research_state["analyst_notes"]` and are surfaced to the Director.

It is **strictly advisory**: the trusted-KKT sandbox remains the sole accept/
reject gate, and an Analyst note can never flip a decision. `--analyst off` uses
`StubAnalyst`, which returns "" (no notes), reproducing the prior behavior.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol

_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_REASONING_EFFORT = "low"
_DEFAULT_API_TIMEOUT_S = 120.0


class Analyst(Protocol):
    def analyze(self, candidate_ctx: dict) -> str: ...


class StubAnalyst:
    """No-op analyst (--analyst off). Returns "" so no notes are produced."""

    def analyze(self, candidate_ctx: dict) -> str:  # noqa: ARG002 — unused by design
        return ""


class OpenAIAnalyst:
    """One cheap Responses-API call summarizing why a candidate failed.

    Stateless and advisory; mirrors the proposer/director client surface.
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
        self._client = client

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

    def analyze(self, candidate_ctx: dict) -> str:
        from .proposers.openai import _extract_response_text

        prompt = _format_analyst_prompt(candidate_ctx)
        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analyst_note",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["summary"],
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": (
                                    "ONE short line (<= 25 words): the concrete reason this "
                                    "candidate failed and the lever it implies for next time, "
                                    "e.g. 'plateaued at kkt~2e-3 (fp32 floor) — needs fp64 "
                                    "polish' or 'valid but slower: redundant full-gradient scan'."
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
        except json.JSONDecodeError:
            return ""
        return str(parsed.get("summary", "")).strip()


def _format_analyst_prompt(ctx: dict) -> str:
    score = ctx.get("score") or {}
    traj = score.get("trajectory") or []
    pts = ", ".join(f"({t:.3f}s,{k:.1e})" for t, k in traj[:12])
    return "\n\n".join([
        "You are the ANALYST of a KKT-verified kernel-autoresearch loop. A "
        "candidate was just rejected by the deterministic gate. In ONE short "
        "line, say why it failed and what lever it implies — for the strategist "
        "to use. Do not propose code.",
        "--- candidate ---",
        "rationale: {}".format(ctx.get("rationale", "")[:300]),
        "algorithm_family: {}".format(ctx.get("algorithm_family", "")),
        "decision: {}".format(ctx.get("reason", "")),
        "reached_target={} kkt_final={} time_to_kkt_s={} total_time_s={}".format(
            score.get("reached_target"), score.get("kkt_final"),
            score.get("time_to_kkt_s"), score.get("total_time_s"),
        ),
        "champion total_time_s={}".format(ctx.get("champion_total_time_s")),
        "KKT-vs-time: {}".format(pts or "(none)"),
        "--- output ---",
        "Return JSON {\"summary\": \"<one line>\"}. No markdown.",
    ])


__all__ = ["Analyst", "StubAnalyst", "OpenAIAnalyst"]
