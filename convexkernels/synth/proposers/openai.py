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
                                    "module. Must define both the step function "
                                    "(fista_step / pdhg_step) and init_state and any "
                                    "helpers it needs."
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
    fitness_kind = score.get("fitness_kind", "kkt")
    traj = score.get("fitness_trajectory") or []
    history = ctx.get("history", [])
    program = (ctx.get("program_md") or "").strip()
    src_path = ctx.get("current_source_path") or "<unknown>"

    sections: list[str] = []
    sections.append(
        "You are a kernel-synthesis proposer for a KKT/gap-gated autoresearch loop. "
        "You rewrite a single Python file that defines `pdhg_step` (or `fista_step`) "
        "and `init_state`. The harness evaluates your candidate in a sandbox, gates "
        "on convergence and speedup, and either keeps your version as the new best "
        "or discards it. Be concrete: name the operations you fuse, the launches "
        "you eliminate, the dtype/storage choice. Do not propose changes that break "
        "the public interface or violate the convergence contract."
    )

    if program:
        sections.append("--- program.md (research org code) ---")
        sections.append(program)

    sections.append("--- slot ---")
    sections.append(json.dumps(slot, indent=2))

    sections.append("--- current source ---")
    sections.append(f"path: {src_path}")
    sections.append("```python")
    sections.append(ctx.get("current_source", ""))
    sections.append("```")

    sections.append("--- current score (multi-rep, n_reps={}) ---".format(score.get("n_reps", 1)))
    sections.append(
        "converged={}; {} median={:.3e}; iters={};\n"
        "solve_ms median={:.3f}, min={:.3f}, max={:.3f}, std={:.3f}\n"
        "setup_ms median={:.3f}".format(
            score.get("converged"),
            fitness_kind,
            score.get("fitness_final", 0.0),
            score.get("iters", 0),
            score.get("solve_ms_median", 0.0),
            score.get("solve_ms_min", 0.0),
            score.get("solve_ms_max", 0.0),
            score.get("solve_ms_std", 0.0),
            score.get("setup_ms_median", 0.0),
        )
    )

    if traj:
        formatted = ", ".join(f"{v:.2e}" for v in traj)
        sections.append(f"fitness trajectory (compressed, ~9 points across {score.get('iters', 0)} iters): {formatted}")
        if len(traj) >= 2 and traj[0] > 0 and traj[-1] / max(traj[0], 1e-30) < 1e-3:
            sections.append("(monotone descent — no obvious stall)")
        elif len(traj) >= 2 and traj[-1] > traj[0]:
            sections.append("(WARNING: trajectory is increasing — divergence)")

    sections.append("--- history (last {} proposals on this slot) ---".format(len(history)))
    if history:
        for h in history:
            sections.append(json.dumps(h, default=str))
    else:
        sections.append("(empty — first proposal on this slot)")

    sections.append("--- target metric ---")
    sections.append(
        "fitness ({fk}) must be < {tol:.1e}. solve_ms median must be < "
        "{margin} * current best (i.e. >= {pct:.1f}% faster). cost model: {cm}.".format(
            fk=fitness_kind,
            tol=ctx.get("fitness_tol", 1e-6),
            margin=ctx.get("speedup_margin", 0.97),
            pct=(1.0 - ctx.get("speedup_margin", 0.97)) * 100,
            cm=ctx.get("cost_model", "single"),
        )
    )

    sections.append("--- output ---")
    sections.append(
        "Return JSON with two fields:\n"
        "  rationale: one paragraph stating what changed and why it should be faster.\n"
        "  full_source: the complete replacement Python module source.\n"
        "Do not wrap in markdown code fences. Do not include explanatory text outside the JSON."
    )
    return "\n\n".join(sections)


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
