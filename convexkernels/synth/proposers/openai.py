"""OpenAI-backed kernel proposer for the autoresearch loop.

Slimmed for the 2026-05-10 pivot. The loop now feeds a rich proposal context
(current source, multi-rep score, fitness trajectory, history with
rationales, program.md instruction layer); this proposer formats it into a
single Responses-API call and parses a structured `{rationale, full_source}`
back. No structured edit grammar, no edit priors, no fitness diagnostics —
those proved to be observation-only machinery.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..loop import Edit


_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_REASONING_EFFORT = "medium"
_DEFAULT_API_TIMEOUT_S = 240.0


class OpenAIProposer:
    """Calls the OpenAI Responses API and parses a structured kernel rewrite.

    The caller (`run_synth_loop`) hands us a ``ctx`` dict each iteration
    containing the active source, current score, history, and program.md.
    We turn that into a single prompt; the model is asked for JSON
    ``{rationale: str, full_source: str}``.

    Failures are surfaced as raised exceptions; the loop logs them as
    ``crash:proposer_error:<type>`` and continues.
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

    def propose(self, ctx: dict) -> Edit:
        prompt = _format_prompt(ctx)
        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "kernel_proposal",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["rationale", "full_source"],
                        "properties": {
                            "rationale": {
                                "type": "string",
                                "description": (
                                    "One paragraph: what is being changed, why this should "
                                    "be faster, and which constraint (KKT/gap, semantics, "
                                    "interface) it preserves."
                                ),
                            },
                            "full_source": {
                                "type": "string",
                                "description": (
                                    "The complete replacement Python source for the kernel "
                                    "module. Must define `solve(problem, recorder, *, "
                                    "kkt_tol, max_time_s) -> X` (and optionally "
                                    "`prepare_problem`) plus any helpers it needs. The "
                                    "algorithm is yours to choose."
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
            raise RuntimeError(f"openai response not valid JSON: {exc}: {text[:500]}")

        rationale = str(parsed.get("rationale", "")).strip()
        source = str(parsed.get("full_source", ""))
        if not source.strip():
            raise RuntimeError("openai returned empty full_source")
        return Edit(
            type="full_source",
            rationale=rationale,
            full_source=source,
            proposer_role="impl",
            proposer_model=self.model,
        )


# ---------- prompt assembly ----------


def _format_prompt(ctx: dict) -> str:
    """Build the prompt from the loop's proposal context.

    Sections (Karpathy-program.md style):
      1. SYSTEM/ROLE
      2. PROGRAM.MD: research org code, what to optimize, what NOT to change
      3. CURRENT SOURCE: the file being rewritten
      4. CURRENT SCORE: multi-rep timing + fitness with trajectory
      5. HISTORY: last N attempts with rationale + outcome
      6. TARGET METRIC: hard gate + speedup margin
      7. OUTPUT: JSON contract
    """
    slot = ctx["slot"]
    score = ctx["current_score"]
    state = ctx.get("research_state") or {}
    traj = score.get("trajectory") or []
    program = (ctx.get("program_md") or "").strip()
    src_path = ctx.get("current_source_path") or "<unknown>"
    kkt_tol = ctx.get("kkt_tol", 1e-6)
    margin = ctx.get("margin", 0.97)

    sections: list[str] = []
    sections.append(
        "You are a solver-synthesis proposer for a KKT-verified autoresearch loop. "
        "You rewrite a single Python module that defines "
        "`solve(problem, recorder, *, kkt_tol, max_time_s) -> X` (and optionally "
        "`prepare_problem`). The algorithm is YOURS to choose — FISTA, ADMM, PDHG, "
        "coordinate descent, screening, reduced precision, quantization, custom "
        "Metal kernels — anything convex-correct. The harness times your solve and "
        "verifies optimality with a TRUSTED KKT residual it computes itself (you "
        "cannot fake it). Keep the candidate iff it reaches the target faster than "
        "the current champion. Be concrete about the algorithm and the kernels."
    )

    if program:
        sections.append("--- program.md (research org code) ---")
        sections.append(program)

    sections.append("--- spec (problem / hardware) ---")
    sections.append(json.dumps(slot, indent=2))

    sections.append("--- current source ---")
    sections.append(f"path: {src_path}")
    sections.append("```python")
    sections.append(ctx.get("current_source", ""))
    sections.append("```")

    sections.append("--- current champion score (median over {} reps) ---".format(score.get("n_reps", 1)))
    sections.append(
        "reached_target={}; kkt_final={:.3e}; time_to_kkt_s={}; total_time_s={} "
        "(setup_s={:.4f})".format(
            score.get("reached_target"),
            score.get("kkt_final", 0.0),
            _fmt_s(score.get("time_to_kkt_s")),
            _fmt_s(score.get("total_time_s")),
            score.get("setup_s", 0.0),
        )
    )

    if traj:
        pts = ", ".join(f"({t:.3f}s,{k:.1e})" for t, k in traj[:12])
        sections.append(f"KKT-vs-time trajectory (t, kkt): {pts}")

    bar = state.get("bar_to_beat") or {}
    sections.append("--- bar to beat (classical baselines, time_to_kkt_s) ---")
    sections.append(
        "best baseline: {} @ {}\nall: {}".format(
            bar.get("best_baseline"),
            _fmt_s(bar.get("best_baseline_time_to_kkt_s")),
            json.dumps(bar.get("all_baselines_time_to_kkt_s") or {}, default=str),
        )
    )

    tried = state.get("tried_directions") or []
    sections.append("--- tried directions (curated; don't repeat failures) ---")
    if tried:
        for t in tried:
            sections.append(json.dumps(t, default=str))
    else:
        sections.append("(none yet — first proposal)")

    sections.append("--- target metric ---")
    sections.append(
        "Reach trusted KKT < {tol:.1e} (the hard correctness gate). Your "
        "total_time_s (setup + time_to_kkt) must be < {margin} * the champion's "
        "(i.e. ≥{pct:.1f}% faster) to be kept. Beating the baseline bar above is "
        "the real goal.".format(
            tol=kkt_tol, margin=margin, pct=(1.0 - margin) * 100,
        )
    )

    sections.append("--- output ---")
    sections.append(
        "Return JSON with two fields:\n"
        "  rationale: one paragraph naming the algorithm + the concrete change and "
        "why it should reach the KKT target sooner.\n"
        "  full_source: the complete replacement Python module (defines `solve`).\n"
        "Do not wrap in markdown code fences. Do not include explanatory text outside the JSON."
    )
    return "\n\n".join(sections)


def _fmt_s(v) -> str:
    import math as _math

    if isinstance(v, (int, float)) and _math.isfinite(v):
        return f"{v:.4f}s"
    return "—"


def _extract_response_text(response: Any) -> str:
    """Pull the text payload out of an OpenAI Responses-API response."""
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text)
    if hasattr(response, "output") and response.output:
        chunks: list[str] = []
        for item in response.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                t = getattr(c, "text", None) or getattr(c, "value", None)
                if t:
                    chunks.append(str(t))
        if chunks:
            return "\n".join(chunks)
    if isinstance(response, dict):
        return str(response.get("output_text") or response.get("text") or "")
    return str(response)
