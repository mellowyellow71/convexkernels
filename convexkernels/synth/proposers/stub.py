"""Deterministic stub proposer for tests and regression smokes.

Returns a sequence of pre-canned full-source edits. Used to drive the new
loop without an LLM in the path so loop semantics can be unit-tested.
"""
from __future__ import annotations

from typing import Sequence

from ..loop import Edit


class StubProposer:
    """Plays back a fixed sequence of (rationale, full_source) pairs."""

    def __init__(self, sources: Sequence[tuple[str, str]]):
        self._sources = list(sources)
        self._idx = 0

    def propose(self, ctx: dict) -> Edit:
        if self._idx >= len(self._sources):
            raise StopIteration("stub exhausted")
        rationale, src = self._sources[self._idx]
        self._idx += 1
        return Edit(
            type="full_source",
            rationale=rationale,
            full_source=src,
            proposer_role="impl",
            proposer_model="stub",
        )
