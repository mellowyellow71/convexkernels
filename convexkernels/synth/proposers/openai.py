"""OpenAI proposer: calls the Responses API for kernel synthesis.

Reads the champion source + recent lineage history, formats the synth prompt
(`convexkernels/synth/prompts/impl_openai.md`), calls the model, and parses a
structured JSON response into an `Edit`.

Requires `OPENAI_API_KEY` env var (or pass `api_key=...`) unless a test client
is injected.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import convexkernels
from ..edits import EDIT_TYPES
from ..lineage import Edit, Slot

_PKG_ROOT = Path(convexkernels.__file__).parent
_PROMPT_PATH = _PKG_ROOT / "synth" / "prompts" / "impl_openai.md"
_DEFAULT_SEED_PATH = _PKG_ROOT / "kernels" / "mlx" / "seeds" / "fista_step_v0.py"

_EDIT_TYPES = set(EDIT_TYPES)

_STRUCTURED_EDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "threadgroup_size",
        "items_per_thread",
        "remove_bounds_check",
        "branchless_soft_threshold",
        "gradient_strategy",
        "dtype_strategy",
        "kernel_name_suffix",
    ],
    "properties": {
        "threadgroup_size": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 1024,
            "description": "New Metal threadgroup size, or null.",
        },
        "items_per_thread": {
            "type": ["integer", "null"],
            "enum": [2, 4, None],
            "description": "Have each Metal thread process 2 or 4 contiguous coefficients, or null.",
        },
        "remove_bounds_check": {
            "type": ["boolean", "null"],
            "description": "Remove the per-element i >= n guard, or null.",
        },
        "branchless_soft_threshold": {
            "type": ["boolean", "null"],
            "description": "Use max(z-thresh,0)-max(-z-thresh,0), or null.",
        },
        "gradient_strategy": {
            "type": ["string", "null"],
            "enum": ["direct", "gram", None],
            "description": "Switch the FISTA gradient path, or null.",
        },
        "dtype_strategy": {
            "type": ["string", "null"],
            "enum": ["fp32", "fp16_storage", "mixed_gram", None],
            "description": "Dtype/search strategy for the candidate, or null.",
        },
        "kernel_name_suffix": {
            "type": ["string", "null"],
            "description": "Optional suffix for the generated Metal kernel name.",
        },
    },
}

_RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "kernel_edit",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["edit_type", "rationale", "source", "structured_edit"],
            "properties": {
                "edit_type": {
                    "type": "string",
                    "enum": sorted(_EDIT_TYPES),
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence describing the proposed mutation.",
                },
                "source": {
                    "type": "string",
                    "description": "Complete Python source, or empty string when using structured_edit.",
                },
                "structured_edit": _STRUCTURED_EDIT_SCHEMA,
            },
        },
    }
}


class OpenAIProposer:
    """LLM proposer that emits a full kernel-module source via OpenAI."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        api_key: Optional[str] = None,
        max_tokens: int = 6000,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = "medium",
        api_timeout_s: Optional[float] = 180.0,
        seed_source_path: Optional[Path] = None,
        prompt_template: Optional[str] = None,
        client: Any | None = None,
    ):
        if client is None:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Export it in the shell that "
                    "runs the synth loop, or pass api_key=..."
                )
            # Lazy import: openai is an optional `[ai]` extra. Importing here
            # lets the rest of the synth module be usable without it.
            from openai import OpenAI

            client = OpenAI(api_key=resolved_key)

        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api_timeout_s = api_timeout_s
        self.seed_source_path = Path(seed_source_path or _DEFAULT_SEED_PATH)
        self.prompt_template = prompt_template or _PROMPT_PATH.read_text()
        self.runtime_context: dict[str, Any] = {}

    def set_runtime_context(self, context: dict[str, Any]) -> None:
        """Update loop-provided measurement context for the next proposal."""
        self.runtime_context = context

    def propose(
        self, slot: Slot, parent_id: Optional[str], history: list[dict]
    ) -> Edit:
        champion_source = self.seed_source_path.read_text()
        recent_history = self._format_history(history[-5:])
        runtime_context = self._format_runtime_context(self.runtime_context)

        prompt = self.prompt_template
        prompt = prompt.replace("{{champion_source}}", champion_source)
        prompt = prompt.replace("{{recent_history}}", recent_history)
        prompt = prompt.replace("{{runtime_context}}", runtime_context)

        request: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": self.max_tokens,
            "text": _RESPONSE_FORMAT,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.api_timeout_s is not None:
            request["timeout"] = self.api_timeout_s

        response = self._client.responses.create(**request)
        text = self._response_text(response)
        data = self._parse_json(text)

        edit_type = data.get("edit_type") or "other"
        if edit_type not in _EDIT_TYPES:
            edit_type = "other"
        rationale = data.get("rationale") or "no rationale provided"
        source = data.get("source") or ""
        structured_payload = self._structured_payload(data.get("structured_edit"))
        if source.strip():
            self._validate_source(source)
            payload = {"full_source": source}
        elif structured_payload:
            payload = structured_payload
        else:
            raise ValueError(
                "OpenAI response must provide non-empty `source` or a "
                "non-empty `structured_edit`"
            )

        return Edit(
            type=edit_type,
            payload=payload,
            rationale=rationale,
            proposer_role="impl",
            proposer_model=self.model,
            source="openai_responses",
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        text = "".join(chunks)
        if not text.strip():
            raise ValueError("OpenAI response did not contain output text")
        return text

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Structured outputs should be plain JSON, but this keeps the error
            # path useful if a mock or older model wraps it in a code fence.
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if not match:
                raise ValueError(
                    "OpenAI response was not valid JSON. "
                    f"First 500 chars:\n{text[:500]}"
                ) from None
            data = json.loads(match.group(1))

        if not isinstance(data, dict):
            raise ValueError("OpenAI response JSON must be an object")
        return data

    @staticmethod
    def _validate_source(source: str) -> None:
        if not source.strip():
            raise ValueError("OpenAI response missing non-empty `source`")
        required = {
            "init_state": r"def\s+init_state\s*\(",
            "fista_step": r"def\s+fista_step\s*\(",
        }
        for name, pattern in required.items():
            if not re.search(pattern, source):
                raise ValueError(
                    f"OpenAI response source missing required `{name}` function"
                )

    @staticmethod
    def _structured_payload(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload: dict[str, Any] = {}
        if value.get("threadgroup_size") is not None:
            payload["threadgroup_size"] = int(value["threadgroup_size"])
        if value.get("items_per_thread") is not None:
            payload["items_per_thread"] = int(value["items_per_thread"])
        for key in ("remove_bounds_check", "branchless_soft_threshold"):
            if value.get(key) is not None:
                payload[key] = bool(value[key])
        if value.get("gradient_strategy") is not None:
            payload["gradient_strategy"] = str(value["gradient_strategy"])
        if value.get("dtype_strategy") is not None:
            payload["dtype_strategy"] = str(value["dtype_strategy"])
        if value.get("kernel_name_suffix"):
            payload["kernel_name_suffix"] = str(value["kernel_name_suffix"])
        return payload

    @staticmethod
    def _format_history(recent: list[dict]) -> str:
        if not recent:
            return "(no prior attempts)"
        lines = []
        for r in recent:
            if r.get("kind") == "baseline":
                tier1 = r.get("tier1", {})
                line = (
                    "- baseline: "
                    f"wall_time_ms={tier1.get('wall_time_ms', '?')}, "
                    f"n_reps={tier1.get('n_reps', 1)}, "
                    f"kkt_final={tier1.get('kkt_final', '?')}"
                )
                tier2 = r.get("tier2")
                if tier2:
                    line += (
                        f"; tier2_wall_time_ms={tier2.get('wall_time_ms', '?')}, "
                        f"tier2_n_reps={tier2.get('n_reps', 1)}"
                    )
                lines.append(line)
                continue
            edit = r.get("edit", {})
            tier1 = r.get("tier1", {})
            tier2 = r.get("tier2", {})
            decision = r.get("decision", {})
            wall = tier1.get("wall_time_ms", "?")
            reps = tier1.get("n_reps", 1)
            outcome = decision.get("reason", "?")
            line = (
                f"- type={edit.get('type', '?')}, "
                f"rationale={edit.get('rationale', '')[:80]!r}\n"
                f"  outcome: {outcome}, tier1_wall_time_ms={wall}, "
                f"tier1_n_reps={reps}"
            )
            if tier2:
                line += f", tier2_wall_time_ms={tier2.get('wall_time_ms', '?')}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_runtime_context(context: dict[str, Any]) -> str:
        baseline = context.get("baseline_wall_ms")
        target = context.get("target_wall_ms")
        if baseline is None:
            return "No timed baseline is available for this run."
        tier1_reps = context.get("tier1_reps", 1)
        lines = []

        shape = context.get("shape") or {}
        if shape:
            lines.append(
                "Active shape: "
                f"{shape.get('name', 'unknown')} "
                f"(m={shape.get('m', '?')}, n={shape.get('n', '?')}), "
                f"regime={shape.get('regime', 'unknown')}."
            )
            guidance = shape.get("guidance")
            if guidance:
                lines.append(f"Shape guidance: {guidance}")

        variant = context.get("algorithm_variant")
        if variant:
            lines.append(f"Host algorithm variant: FISTA {variant}.")
        dtype_strategy = context.get("dtype_strategy")
        problem_dtype = context.get("problem_dtype")
        if dtype_strategy or problem_dtype:
            lines.append(
                "Dtype search context: "
                f"problem_dtype={problem_dtype}, dtype_strategy={dtype_strategy}."
            )
        cost_model = context.get("cost_model")
        if cost_model:
            lines.append(
                f"Speed gate cost model: {cost_model} "
                "(single includes setup; amortized uses solve-only time)."
            )
        current_kernel = context.get("current_kernel") or {}
        if current_kernel:
            gradient_strategy = current_kernel.get("gradient_strategy")
            dtype_strategy_current = current_kernel.get("dtype_strategy")
            if gradient_strategy or dtype_strategy_current:
                lines.append(
                    "Current champion kernel strategy: "
                    f"gradient_strategy={gradient_strategy}, "
                    f"dtype_strategy={dtype_strategy_current}."
                )
            focus = current_kernel.get("focus")
            if focus:
                lines.append(f"Current champion search focus: {focus}")

        lines.append(
            f"Current warmed champion Tier-1 median baseline "
            f"({tier1_reps} reps): {baseline:.3f} ms."
        )
        if target is not None:
            gate_role = context.get("tier1_gate_role", "promotion")
            if gate_role == "tier2_escalation":
                lines.append(
                    f"Tier-1 escalation target: pass KKT and run below "
                    f"{target:.3f} ms to earn Tier-2 evaluation."
                )
            else:
                lines.append(
                    f"Tier-1 keep target: pass KKT and run below {target:.3f} ms."
                )

        tier2_baseline = context.get("tier2_baseline_wall_ms")
        tier2_target = context.get("tier2_target_wall_ms")
        tier2_reps = context.get("tier2_reps", 1)
        if tier2_baseline is not None:
            lines.append(
                f"Current warmed champion Tier-2 median convergence baseline "
                f"({tier2_reps} reps): {tier2_baseline:.3f} ms."
            )
        if tier2_target is not None:
            lines.append(
                f"Tier-2 promotion target: converge and run below "
                f"{tier2_target:.3f} ms."
            )
        if context.get("confirm_tier2_speed"):
            lines.append(
                "Tier-2 speed wins are confirmed against a paired remeasurement "
                "of the current champion before promotion."
            )

        history_summary = context.get("history_summary") or {}
        if history_summary.get("n_attempts"):
            lines.append(
                "Current-run outcome counts: "
                f"{history_summary.get('decision_counts', {})}."
            )
            lines.append(
                "Most common edit outcomes: "
                f"{history_summary.get('edit_outcomes', {})}."
            )
            fastest = history_summary.get("fastest_rejected")
            if fastest:
                lines.append(
                    "Fastest rejected Tier-1 pass so far: "
                    f"edit={fastest.get('edit_type')}, "
                    f"reason={fastest.get('reason')}, "
                    f"tier1={fastest.get('tier1_wall_time_ms')} ms, "
                    f"tier2={fastest.get('tier2_wall_time_ms')} ms."
                )

        fitness = context.get("fitness_summary") or {}
        if fitness and fitness.get("n_records"):
            lines.append(
                "Structured fitness class counts: "
                f"{fitness.get('performance_classes', {})}."
            )
            lines.append(
                "Structured fitness bottleneck hints: "
                f"{fitness.get('bottleneck_hints', {})}."
            )
            near_misses = fitness.get("near_misses") or []
            if near_misses:
                lines.append(
                    "Fitness near misses with diagnosis: "
                    f"{near_misses}."
                )
            overhead_limited = fitness.get("overhead_limited") or []
            if overhead_limited:
                lines.append(
                    "Low-roofline overhead/algorithm-limited examples: "
                    f"{overhead_limited}."
                )
            roofline_limited = fitness.get("roofline_limited") or []
            if roofline_limited:
                lines.append(
                    "High-roofline bandwidth/dtype-limited examples: "
                    f"{roofline_limited}."
                )
            high_noise = fitness.get("high_noise") or []
            if high_noise:
                lines.append(
                    "High timing-noise examples; avoid overfitting these: "
                    f"{high_noise}."
                )
        edit_priors = context.get("edit_priors_summary") or {}
        if edit_priors:
            top_accepted = edit_priors.get("top_accepted") or []
            avoid = edit_priors.get("avoid_until_changed") or []
            top_payloads = edit_priors.get("top_structured_payloads") or []
            near_miss_payloads = (
                edit_priors.get("near_miss_structured_payloads") or []
            )
            avoid_payloads = edit_priors.get("avoid_structured_payloads") or []
            if top_accepted:
                lines.append(f"Historical edit types with accepted wins: {top_accepted}.")
            if avoid:
                lines.append(
                    "Historical edit types to avoid unless materially changed: "
                    f"{avoid}."
                )
            if top_payloads:
                lines.append(
                    "Historical structured payloads with accepted wins: "
                    f"{top_payloads}."
                )
            if near_miss_payloads:
                lines.append(
                    "Historical structured payload near misses, ranked by "
                    f"Tier-2 speed ratio: {near_miss_payloads}."
                )
            if avoid_payloads:
                lines.append(
                    "Historical structured payloads to avoid as exact repeats: "
                    f"{avoid_payloads}."
                )
        return "\n".join(lines)
