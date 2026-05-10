"""Claude proposer: calls the Anthropic API for kernel synthesis.

Reads the champion source + recent lineage history, formats the synth prompt
(`convexkernels/synth/prompts/impl.md`), calls the model, parses the
`<edit_type>`/`<rationale>`/`<source>` tags from the response.

Requires `ANTHROPIC_API_KEY` env var (or pass `api_key=...`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import convexkernels
from ..lineage import Edit, Slot

_PKG_ROOT = Path(convexkernels.__file__).parent
_PROMPT_PATH = _PKG_ROOT / "synth" / "prompts" / "impl.md"
_DEFAULT_SEED_PATH = _PKG_ROOT / "kernels" / "mlx" / "seeds" / "fista_step_v0.py"


class ClaudeProposer:
    """LLM proposer that emits a full kernel-module source via the Anthropic API."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        max_tokens: int = 6000,
        temperature: float = 1.0,
        seed_source_path: Optional[Path] = None,
        prompt_template: Optional[str] = None,
    ):
        # Lazy import: anthropic is an optional `[ai]` extra. Importing here lets
        # the rest of the synth module be usable without it.
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed_source_path = Path(seed_source_path or _DEFAULT_SEED_PATH)
        self.prompt_template = prompt_template or _PROMPT_PATH.read_text()

    def propose(
        self, slot: Slot, parent_id: Optional[str], history: list[dict]
    ) -> Edit:
        champion_source = self.seed_source_path.read_text()
        recent_history = self._format_history(history[-5:])

        prompt = self.prompt_template
        prompt = prompt.replace("{{champion_source}}", champion_source)
        prompt = prompt.replace("{{recent_history}}", recent_history)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        edit_type = self._extract_tag(text, "edit_type") or "other"
        rationale = self._extract_tag(text, "rationale") or "no rationale provided"
        source = self._extract_tag(text, "source")

        if not source:
            raise ValueError(
                f"Claude response missing <source> tag. "
                f"First 500 chars of response:\n{text[:500]}"
            )

        return Edit(
            type=edit_type,
            payload={"full_source": source},
            rationale=rationale,
            proposer_role="impl",
            proposer_model=self.model,
            source="claude_subagent",
        )

    @staticmethod
    def _extract_tag(text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _format_history(recent: list[dict]) -> str:
        if not recent:
            return "(no prior attempts)"
        lines = []
        for r in recent:
            edit = r.get("edit", {})
            tier1 = r.get("tier1", {})
            decision = r.get("decision", {})
            kkt = "?"
            wall = tier1.get("wall_time_ms", "?")
            outcome = decision.get("reason", "?")
            lines.append(
                f"- type={edit.get('type', '?')}, "
                f"rationale={edit.get('rationale', '')[:80]!r}\n"
                f"  outcome: {outcome}, wall_time_ms={wall}"
            )
        return "\n".join(lines)
