"""Edit grammar helpers and outcome priors.

The OpenAI proposer can emit either complete source files or a small structured
payload. This module keeps outcome accounting separate from proposers so the
loop can learn both coarse edit-type priors and exact structured-payload priors.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .lineage import Slot

EDIT_TYPES: tuple[str, ...] = (
    "tile_change",
    "dtype_swap",
    "fuse_op",
    "hoist_to_threadgroup",
    "vectorize",
    "algo_variant",
    "swap_layout",
    "other",
)

STRUCTURED_PAYLOAD_KEYS: tuple[str, ...] = (
    "threadgroup_size",
    "items_per_thread",
    "remove_bounds_check",
    "branchless_soft_threshold",
    "gradient_strategy",
    "dtype_strategy",
)


def build_edit_priors(records: Iterable[dict]) -> dict[str, Any]:
    """Build JSON-serializable edit outcome priors from lineage records."""
    global_stats: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    by_slot: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(_empty_stats)
    )
    global_payloads: dict[str, dict[str, Any]] = {}
    by_slot_payloads: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in records:
        edit = record.get("edit") or {}
        edit_type = edit.get("type")
        if not edit_type:
            continue

        slot_key = _slot_key(record.get("slot") or {})
        for stats in (global_stats[edit_type], by_slot[slot_key][edit_type]):
            _add_record(stats, record)

        payload = edit.get("payload") or {}
        payload_signature = structured_payload_signature(payload)
        if payload_signature:
            compact_payload = compact_structured_payload(payload)
            for entries in (global_payloads, by_slot_payloads[slot_key]):
                entry = entries.setdefault(
                    payload_signature,
                    {
                        "edit_type": edit_type,
                        "payload": compact_payload,
                        "stats": _empty_stats(),
                    },
                )
                _add_record(entry["stats"], record)

    global_final = {
        edit_type: _finalize_stats(stats)
        for edit_type, stats in sorted(global_stats.items())
    }
    slot_final = {
        slot_key: {
            edit_type: _finalize_stats(stats)
            for edit_type, stats in sorted(slot_stats.items())
        }
        for slot_key, slot_stats in sorted(by_slot.items())
    }
    global_payload_final = _finalize_payload_entries(global_payloads)
    slot_payload_final = {
        slot_key: _finalize_payload_entries(payload_stats)
        for slot_key, payload_stats in sorted(by_slot_payloads.items())
    }
    return {
        "version": 2,
        "global": global_final,
        "by_slot": slot_final,
        "global_payloads": global_payload_final,
        "payloads_by_slot": slot_payload_final,
    }


def write_edit_priors(records: Iterable[dict], path: Path) -> dict[str, Any]:
    """Regenerate `edits.json` from lineage records and write it."""
    priors = build_edit_priors(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(priors, indent=2, sort_keys=True))
    return priors


def summarize_edit_priors(
    priors: dict[str, Any],
    slot: Slot,
    *,
    min_attempts: int = 2,
    payload_min_attempts: int = 1,
    max_items: int = 4,
) -> dict[str, Any]:
    """Return a compact proposer-facing summary for this slot."""
    slot_stats = priors.get("by_slot", {}).get(slot.key(), {})
    global_stats = priors.get("global", {})
    slot_payloads = priors.get("payloads_by_slot", {}).get(slot.key(), {})
    global_payloads = priors.get("global_payloads", {})
    merged: dict[str, dict[str, Any]] = {}
    for edit_type, stats in global_stats.items():
        merged[edit_type] = dict(stats)
        merged[edit_type]["scope"] = "global"
    for edit_type, stats in slot_stats.items():
        merged[edit_type] = dict(stats)
        merged[edit_type]["scope"] = "slot"

    attempted = [
        (edit_type, stats)
        for edit_type, stats in merged.items()
        if stats.get("n_proposed", 0) >= min_attempts
    ]
    accepted = sorted(
        (
            (edit_type, stats)
            for edit_type, stats in attempted
            if stats.get("n_accepted", 0) > 0
        ),
        key=lambda item: (
            item[1].get("accept_rate", 0.0),
            item[1].get("n_accepted", 0),
        ),
        reverse=True,
    )
    avoid = sorted(
        (
            (edit_type, stats)
            for edit_type, stats in attempted
            if stats.get("n_accepted", 0) == 0
        ),
        key=lambda item: (
            item[1].get("n_tier2_speed_failed", 0),
            item[1].get("n_invalid", 0) + item[1].get("n_crashed", 0),
            item[1].get("n_proposed", 0),
        ),
        reverse=True,
    )
    payload_attempted = _merge_payload_stats(
        global_payloads,
        slot_payloads,
        min_attempts=payload_min_attempts,
    )
    payload_accepted = sorted(
        (
            stats for stats in payload_attempted
            if stats.get("n_accepted", 0) > 0
        ),
        key=lambda stats: (
            stats.get("accept_rate", 0.0),
            stats.get("n_accepted", 0),
            -float(stats.get("median_tier2_wall_ms") or float("inf")),
        ),
        reverse=True,
    )
    payload_avoid = sorted(
        (
            stats for stats in payload_attempted
            if stats.get("n_accepted", 0) == 0
        ),
        key=lambda stats: (
            stats.get("n_tier2_speed_failed", 0),
            stats.get("n_invalid", 0) + stats.get("n_crashed", 0),
            stats.get("n_proposed", 0),
        ),
        reverse=True,
    )
    payload_near_misses = sorted(
        (
            stats for stats in payload_attempted
            if stats.get("n_accepted", 0) == 0
            and stats.get("n_tier2_speed_failed", 0) > 0
            and _tier2_speed_ratio(stats) is not None
        ),
        key=lambda stats: (
            _tier2_speed_ratio(stats) or float("inf"),
            -stats.get("n_proposed", 0),
        ),
    )
    return {
        "slot": slot.key(),
        "top_accepted": [
            _compact_stats(edit_type, stats) for edit_type, stats in accepted[:max_items]
        ],
        "avoid_until_changed": [
            _compact_stats(edit_type, stats) for edit_type, stats in avoid[:max_items]
        ],
        "top_structured_payloads": [
            _compact_payload_stats(stats)
            for stats in payload_accepted[:max_items]
        ],
        "near_miss_structured_payloads": [
            _compact_payload_stats(stats)
            for stats in payload_near_misses[:max_items]
        ],
        "avoid_structured_payloads": [
            _compact_payload_stats(stats)
            for stats in payload_avoid[:max_items]
        ],
    }


def compact_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the behaviorally meaningful structured payload fields."""
    return {
        key: payload[key]
        for key in STRUCTURED_PAYLOAD_KEYS
        if key in payload and payload[key] not in (None, False, "")
    }


def structured_payload_signature(payload: dict[str, Any]) -> str:
    """Return a stable signature for a structured payload, or empty string.

    `kernel_name_suffix` is intentionally ignored because it does not change
    behavior. Full-source payloads do not get a structured signature.
    """
    compact = compact_structured_payload(payload)
    if not compact or "full_source" in payload:
        return ""
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _empty_stats() -> dict[str, Any]:
    return {
        "n_proposed": 0,
        "n_accepted": 0,
        "n_tier1_passed": 0,
        "n_tier2_passed": 0,
        "n_tier2_speed_failed": 0,
        "n_invalid": 0,
        "n_crashed": 0,
        "tier1_wall_ms": [],
        "tier2_wall_ms": [],
        "tier2_speed_ref_wall_ms": [],
        "accepted_tier1_wall_ms": [],
        "accepted_tier2_wall_ms": [],
    }


def _add_record(stats: dict[str, Any], record: dict) -> None:
    decision = record.get("decision") or {}
    reason = str(decision.get("reason", ""))
    tier1 = record.get("tier1") or {}
    tier2 = record.get("tier2") or {}

    stats["n_proposed"] += 1
    if tier1.get("passed"):
        stats["n_tier1_passed"] += 1
    if tier1.get("wall_time_ms") is not None:
        stats["tier1_wall_ms"].append(float(tier1["wall_time_ms"]))
    if tier2.get("passed"):
        stats["n_tier2_passed"] += 1
    if tier2.get("wall_time_ms") is not None:
        stats["tier2_wall_ms"].append(float(tier2["wall_time_ms"]))
    if tier2.get("speed_ref_wall_time_ms"):
        stats["tier2_speed_ref_wall_ms"].append(
            float(tier2["speed_ref_wall_time_ms"])
        )
    if reason == "tier_failed:2_speed":
        stats["n_tier2_speed_failed"] += 1
    if reason.startswith("invalid:"):
        stats["n_invalid"] += 1
    if reason.startswith("crash:"):
        stats["n_crashed"] += 1
    if decision.get("accepted"):
        stats["n_accepted"] += 1
        if tier1.get("wall_time_ms") is not None:
            stats["accepted_tier1_wall_ms"].append(float(tier1["wall_time_ms"]))
        if tier2.get("wall_time_ms") is not None:
            stats["accepted_tier2_wall_ms"].append(float(tier2["wall_time_ms"]))


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    proposed = stats["n_proposed"]
    tier1_wall = stats.pop("tier1_wall_ms")
    tier2_wall = stats.pop("tier2_wall_ms")
    tier2_speed_ref_wall = stats.pop("tier2_speed_ref_wall_ms")
    accepted_tier1 = stats.pop("accepted_tier1_wall_ms")
    accepted_tier2 = stats.pop("accepted_tier2_wall_ms")
    out = dict(stats)
    out["accept_rate"] = stats["n_accepted"] / proposed if proposed else 0.0
    out["tier1_pass_rate"] = stats["n_tier1_passed"] / proposed if proposed else 0.0
    out["tier2_pass_rate"] = stats["n_tier2_passed"] / proposed if proposed else 0.0
    out["median_tier1_wall_ms"] = (
        float(statistics.median(tier1_wall)) if tier1_wall else None
    )
    out["median_tier2_wall_ms"] = (
        float(statistics.median(tier2_wall)) if tier2_wall else None
    )
    out["median_tier2_speed_ref_wall_ms"] = (
        float(statistics.median(tier2_speed_ref_wall))
        if tier2_speed_ref_wall else None
    )
    out["median_tier1_wall_ms_when_accepted"] = (
        float(statistics.median(accepted_tier1)) if accepted_tier1 else None
    )
    out["median_tier2_wall_ms_when_accepted"] = (
        float(statistics.median(accepted_tier2)) if accepted_tier2 else None
    )
    return out


def _compact_stats(edit_type: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "edit_type": edit_type,
        "scope": stats.get("scope", "global"),
        "n": stats.get("n_proposed", 0),
        "accepted": stats.get("n_accepted", 0),
        "accept_rate": round(float(stats.get("accept_rate", 0.0)), 3),
        "tier2_speed_failed": stats.get("n_tier2_speed_failed", 0),
        "invalid_or_crashed": stats.get("n_invalid", 0) + stats.get("n_crashed", 0),
    }


def _finalize_payload_entries(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        signature: {
            "signature": signature,
            "edit_type": entry["edit_type"],
            "payload": entry["payload"],
            **_finalize_stats(entry["stats"]),
        }
        for signature, entry in sorted(entries.items())
    }


def _merge_payload_stats(
    global_payloads: dict[str, Any],
    slot_payloads: dict[str, Any],
    *,
    min_attempts: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for signature, stats in global_payloads.items():
        merged[signature] = dict(stats)
        merged[signature]["scope"] = "global"
    for signature, stats in slot_payloads.items():
        merged[signature] = dict(stats)
        merged[signature]["scope"] = "slot"
    return [
        stats for stats in merged.values()
        if stats.get("n_proposed", 0) >= min_attempts
    ]


def _compact_payload_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "signature": stats.get("signature", ""),
        "edit_type": stats.get("edit_type", "other"),
        "payload": stats.get("payload", {}),
        "scope": stats.get("scope", "global"),
        "n": stats.get("n_proposed", 0),
        "accepted": stats.get("n_accepted", 0),
        "accept_rate": round(float(stats.get("accept_rate", 0.0)), 3),
        "tier2_speed_failed": stats.get("n_tier2_speed_failed", 0),
        "invalid_or_crashed": stats.get("n_invalid", 0) + stats.get("n_crashed", 0),
        "median_tier1_ms": _round_optional(stats.get("median_tier1_wall_ms")),
        "median_tier2_ms": _round_optional(stats.get("median_tier2_wall_ms")),
        "median_tier2_ref_ms": _round_optional(
            stats.get("median_tier2_speed_ref_wall_ms")
        ),
        "tier2_speed_ratio": _round_optional(_tier2_speed_ratio(stats), ndigits=4),
    }


def _tier2_speed_ratio(stats: dict[str, Any]) -> float | None:
    wall = stats.get("median_tier2_wall_ms")
    ref = stats.get("median_tier2_speed_ref_wall_ms")
    if wall is None or ref in (None, 0):
        return None
    return float(wall) / float(ref)


def _round_optional(value: Any, *, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _slot_key(slot: dict[str, Any]) -> str:
    return "/".join(
        str(slot.get(key, "unknown"))
        for key in ("problem_family", "algorithm", "hardware", "dtype")
    )
