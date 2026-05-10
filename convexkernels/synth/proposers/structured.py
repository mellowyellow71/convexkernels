"""Deterministic structured proposer.

This proposer is deliberately non-LLM: it sweeps the structured edit fields the
applier knows how to transform. It is useful for cheap local exploration and for
populating edit priors before asking the API for more creative mutations.
"""

from __future__ import annotations

from typing import Any

from ..edits import structured_payload_signature
from ..lineage import Edit


def default_structured_payloads() -> tuple[dict[str, Any], ...]:
    """Return the default deterministic structured mutation grid."""
    return (
        {
            "gradient_strategy": "gram",
            "dtype_strategy": "fp32",
            "kernel_name_suffix": "gram_fp32",
        },
        {
            "gradient_strategy": "gram",
            "dtype_strategy": "mixed_gram",
            "kernel_name_suffix": "gram_mixed",
        },
        {
            "dtype_strategy": "fp16_storage",
            "kernel_name_suffix": "fp16_storage",
        },
        {"branchless_soft_threshold": True, "kernel_name_suffix": "branchless"},
        {"remove_bounds_check": True, "kernel_name_suffix": "nobounds"},
        {"threadgroup_size": 128, "kernel_name_suffix": "tg128"},
        {"threadgroup_size": 512, "kernel_name_suffix": "tg512"},
        {"items_per_thread": 2, "kernel_name_suffix": "vec2"},
        {"items_per_thread": 4, "kernel_name_suffix": "vec4"},
        {
            "threadgroup_size": 128,
            "branchless_soft_threshold": True,
            "kernel_name_suffix": "tg128_branchless",
        },
        {
            "threadgroup_size": 512,
            "branchless_soft_threshold": True,
            "kernel_name_suffix": "tg512_branchless",
        },
        {
            "items_per_thread": 2,
            "threadgroup_size": 128,
            "kernel_name_suffix": "vec2_tg128",
        },
        {
            "items_per_thread": 2,
            "threadgroup_size": 512,
            "kernel_name_suffix": "vec2_tg512",
        },
        {
            "items_per_thread": 4,
            "threadgroup_size": 128,
            "kernel_name_suffix": "vec4_tg128",
        },
        {
            "items_per_thread": 4,
            "threadgroup_size": 512,
            "kernel_name_suffix": "vec4_tg512",
        },
    )


class StructuredGridProposer:
    """Cycle through deterministic structured edits, skipping seen payloads."""

    model = "structured-grid"

    def __init__(self, payloads: tuple[dict[str, Any], ...] | None = None):
        self.payloads = payloads or default_structured_payloads()
        self._uses_default_payloads = payloads is None
        self.counter = 0

    def propose(self, slot: Any, parent_id: Any, history: list[dict]) -> Edit:
        payloads = self._payloads_for_slot(slot)
        seen = {
            _payload_signature((record.get("edit") or {}).get("payload") or {})
            for record in history
            if record.get("edit")
        }
        for _ in range(len(payloads)):
            payload = dict(payloads[self.counter % len(payloads)])
            self.counter += 1
            if _payload_signature(payload) not in seen:
                return _edit_from_payload(payload, self.counter)

        payload = dict(payloads[self.counter % len(payloads)])
        self.counter += 1
        return _edit_from_payload(payload, self.counter)

    def _payloads_for_slot(self, slot: Any) -> tuple[dict[str, Any], ...]:
        if (
            not self._uses_default_payloads
            or getattr(slot, "problem_family", "") != "nonnegative_lasso"
        ):
            return self.payloads

        adapted: list[dict[str, Any]] = []
        seen: set[str] = set()
        for payload in self.payloads:
            candidate = dict(payload)
            candidate.pop("branchless_soft_threshold", None)
            if not any(
                key in candidate and candidate[key] not in (None, False, "")
                for key in ("threadgroup_size", "items_per_thread", "remove_bounds_check")
            ):
                continue
            signature = _payload_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            adapted.append(candidate)
        return tuple(adapted) or self.payloads


def _edit_from_payload(payload: dict[str, Any], counter: int) -> Edit:
    return Edit(
        type=_edit_type(payload),
        payload=payload,
        rationale=_rationale(payload),
        proposer_role="impl",
        proposer_model="structured-grid",
        source="structured_grid",
    )


def _edit_type(payload: dict[str, Any]) -> str:
    if payload.get("items_per_thread"):
        return "vectorize"
    if payload.get("gradient_strategy"):
        return "algo_variant"
    if payload.get("dtype_strategy"):
        return "dtype_swap"
    if payload.get("threadgroup_size"):
        return "tile_change"
    if payload.get("branchless_soft_threshold"):
        return "fuse_op"
    if payload.get("remove_bounds_check"):
        return "hoist_to_threadgroup"
    return "other"


def _rationale(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    if payload.get("items_per_thread"):
        parts.append(f"{payload['items_per_thread']} items per Metal thread")
    if payload.get("gradient_strategy"):
        parts.append(f"gradient strategy {payload['gradient_strategy']}")
    if payload.get("dtype_strategy"):
        parts.append(f"dtype strategy {payload['dtype_strategy']}")
    if payload.get("threadgroup_size"):
        parts.append(f"threadgroup {payload['threadgroup_size']}")
    if payload.get("branchless_soft_threshold"):
        parts.append("branchless soft-threshold")
    if payload.get("remove_bounds_check"):
        parts.append("remove exact-grid bounds check")
    return "Structured sweep: " + ", ".join(parts)


def _payload_signature(payload: dict[str, Any]) -> str:
    return structured_payload_signature(payload)
