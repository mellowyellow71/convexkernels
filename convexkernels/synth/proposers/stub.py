"""Deterministic stub proposer for testing the synth-loop machinery.

P3.3 uses this to validate the loop end-to-end without an LLM in the critical
path. API-backed proposers come in P3.4.
"""

from __future__ import annotations

from typing import Any

from ..lineage import Edit


class DeterministicStubProposer:
    """Cycles through a fixed set of edit types with synthetic payloads.

    Does NOT actually mutate kernel source — the loop pairs this proposer with
    the seed kernel and just runs the seed each iteration. This validates the
    loop, lineage, sandbox, and gate plumbing while being deterministic.
    """

    def __init__(self, edit_types: tuple[str, ...] = ("tile_change", "dtype_swap")):
        self.edit_types = edit_types
        self.counter = 0

    def propose(self, slot: Any, parent_id: Any, history: list[dict]) -> Edit:
        edit_type = self.edit_types[self.counter % len(self.edit_types)]
        self.counter += 1
        return Edit(
            type=edit_type,
            payload={"variant_index": self.counter},
            rationale=f"deterministic stub #{self.counter}",
            proposer_role="impl",
            proposer_model="stub",
            source="manual",
        )
