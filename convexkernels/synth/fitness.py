"""Structured fitness summaries derived from lineage records.

The evaluator's hard gates still live in `loop.py` and `tiers.py`. This module
turns the raw gate outputs into a compact diagnostic layer for proposers:
correctness status, speed ratios, timing noise, roofline signal, bottleneck
hint, and a concrete recommendation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def fitness_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable fitness summary for one lineage record."""
    edit = record.get("edit") or {}
    decision = record.get("decision") or {}
    tier1 = record.get("tier1") or {}
    tier2 = record.get("tier2") or {}
    tier3 = record.get("tier3") or {}

    tier1_cv = _timing_cv(tier1.get("wall_time_ms"), tier1.get("wall_time_std_ms"))
    tier2_cv = _timing_cv(tier2.get("wall_time_ms"), tier2.get("wall_time_std_ms"))
    tier2_speed_ratio = _ratio(
        tier2.get("wall_time_ms"),
        tier2.get("speed_ref_wall_time_ms"),
    )
    tier2_speed_margin = _optional_float(tier2.get("speed_ref_margin"))
    tier2_speed_target_ratio = (
        tier2_speed_ratio / tier2_speed_margin
        if tier2_speed_ratio is not None and tier2_speed_margin
        else None
    )
    tier3_rank = tier3.get("rank_summary") or {}
    roofline_values = [
        float(shape["roofline_pct_med"])
        for shape in tier3.get("per_shape", []) or []
        if shape.get("roofline_pct_med") is not None
    ]

    performance_class = _performance_class(
        decision=decision,
        tier2_speed_ratio=tier2_speed_ratio,
    )
    bottleneck_hint = _bottleneck_hint(
        performance_class=performance_class,
        decision=decision,
        tier3_median_roofline_pct=tier3_rank.get("median_roofline_pct"),
        roofline_values=roofline_values,
        tier2_speed_ratio=tier2_speed_ratio,
    )
    recommendation = _recommendation(
        performance_class=performance_class,
        bottleneck_hint=bottleneck_hint,
        tier2_speed_ratio=tier2_speed_ratio,
        tier2_speed_margin=tier2_speed_margin,
    )

    return {
        "record_id": record.get("id", ""),
        "slot": _slot_key(record.get("slot") or {}),
        "edit_type": edit.get("type", "unknown"),
        "accepted": bool(decision.get("accepted")),
        "decision_reason": decision.get("reason", ""),
        "performance_class": performance_class,
        "bottleneck_hint": bottleneck_hint,
        "recommendation": recommendation,
        "tier1_passed": bool(tier1.get("passed")),
        "tier1_wall_ms": _round_optional(tier1.get("wall_time_ms")),
        "tier1_setup_ms": _round_optional(tier1.get("setup_time_ms")),
        "tier1_solve_ms": _round_optional(tier1.get("solve_time_ms")),
        "tier1_single_solve_ms": _round_optional(
            tier1.get("single_solve_wall_time_ms")
        ),
        "tier1_amortized_ms": _round_optional(
            tier1.get("amortized_wall_time_ms")
        ),
        "cost_model": tier1.get("cost_model") or tier2.get("cost_model") or "single",
        "tier1_timing_cv": _round_optional(tier1_cv, ndigits=4),
        "tier2_passed": bool(tier2.get("passed")) if tier2 else False,
        "tier2_wall_ms": _round_optional(tier2.get("wall_time_ms")),
        "tier2_setup_ms": _round_optional(tier2.get("setup_time_ms")),
        "tier2_solve_ms": _round_optional(tier2.get("solve_time_ms")),
        "tier2_single_solve_ms": _round_optional(
            tier2.get("single_solve_wall_time_ms")
        ),
        "tier2_amortized_ms": _round_optional(
            tier2.get("amortized_wall_time_ms")
        ),
        "tier2_ref_wall_ms": _round_optional(tier2.get("speed_ref_wall_time_ms")),
        "tier2_speed_ratio": _round_optional(tier2_speed_ratio, ndigits=4),
        "tier2_speed_margin": _round_optional(tier2_speed_margin, ndigits=4),
        "tier2_speed_target_ratio": _round_optional(
            tier2_speed_target_ratio,
            ndigits=4,
        ),
        "tier2_timing_cv": _round_optional(tier2_cv, ndigits=4),
        "n_iters": _optional_int(tier2.get("n_iters")),
        "kkt_final": _round_optional(tier2.get("kkt_final"), ndigits=8),
        "tier3_passed": bool(tier3.get("passed")) if tier3 else False,
        "tier3_median_wall_ms": _round_optional(
            tier3_rank.get("median_wall_time_ms")
        ),
        "tier3_median_roofline_pct": _round_optional(
            tier3_rank.get("median_roofline_pct")
        ),
        "tier3_min_roofline_pct": _round_optional(
            min(roofline_values) if roofline_values else None
        ),
        "tier3_max_roofline_pct": _round_optional(
            max(roofline_values) if roofline_values else None
        ),
    }


def build_fitness_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a proposer-facing structured fitness report from lineage."""
    vectors = [fitness_from_record(record) for record in records if record.get("edit")]
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vector in vectors:
        by_slot[vector["slot"]].append(vector)

    return {
        "version": 1,
        "global": _summarize_vectors(vectors),
        "by_slot": {
            slot: _summarize_vectors(slot_vectors)
            for slot, slot_vectors in sorted(by_slot.items())
        },
    }


def write_fitness_report(
    records: Iterable[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    """Regenerate `fitness.json` from lineage records and write it."""
    report = build_fitness_report(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def summarize_fitness_report(
    report: dict[str, Any],
    slot_key: str,
    *,
    max_items: int = 4,
) -> dict[str, Any]:
    """Return a compact summary suitable for prompt runtime context."""
    summary = report.get("by_slot", {}).get(slot_key) or report.get("global") or {}
    if not summary:
        return {}
    return {
        "slot": slot_key,
        "n_records": summary.get("n_records", 0),
        "performance_classes": summary.get("performance_classes", {}),
        "bottleneck_hints": summary.get("bottleneck_hints", {}),
        "accepted": (summary.get("accepted") or [])[:max_items],
        "near_misses": (summary.get("near_misses") or [])[:max_items],
        "high_noise": (summary.get("high_noise") or [])[:max_items],
        "roofline_limited": (summary.get("roofline_limited") or [])[:max_items],
        "overhead_limited": (summary.get("overhead_limited") or [])[:max_items],
    }


def _summarize_vectors(vectors: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(v["performance_class"] for v in vectors)
    bottleneck_counts = Counter(v["bottleneck_hint"] for v in vectors)
    accepted = [v for v in vectors if v["accepted"]]
    near_misses = [
        v for v in vectors
        if v["performance_class"] == "tier2_speed_near_miss"
    ]
    high_noise = [
        v for v in vectors
        if (v.get("tier1_timing_cv") or 0.0) >= 0.05
        or (v.get("tier2_timing_cv") or 0.0) >= 0.05
    ]
    roofline_limited = [
        v for v in vectors
        if v["bottleneck_hint"] == "bandwidth_or_dtype_limited"
    ]
    overhead_limited = [
        v for v in vectors
        if v["bottleneck_hint"] == "overhead_or_algorithm_limited"
    ]

    return {
        "n_records": len(vectors),
        "performance_classes": dict(class_counts.most_common()),
        "bottleneck_hints": dict(bottleneck_counts.most_common()),
        "accepted": [_compact_vector(v) for v in accepted[-6:]],
        "near_misses": [
            _compact_vector(v)
            for v in sorted(
                near_misses,
                key=lambda item: item.get("tier2_speed_target_ratio") or float("inf"),
            )[:6]
        ],
        "high_noise": [
            _compact_vector(v)
            for v in sorted(
                high_noise,
                key=lambda item: max(
                    item.get("tier1_timing_cv") or 0.0,
                    item.get("tier2_timing_cv") or 0.0,
                ),
                reverse=True,
            )[:6]
        ],
        "roofline_limited": [
            _compact_vector(v)
            for v in sorted(
                roofline_limited,
                key=lambda item: item.get("tier3_median_roofline_pct") or 0.0,
                reverse=True,
            )[:6]
        ],
        "overhead_limited": [
            _compact_vector(v)
            for v in sorted(
                overhead_limited,
                key=lambda item: item.get("tier3_median_roofline_pct") or float("inf"),
            )[:6]
        ],
    }


def _compact_vector(vector: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "record_id",
        "edit_type",
        "decision_reason",
        "performance_class",
        "bottleneck_hint",
        "recommendation",
        "tier1_wall_ms",
        "tier1_timing_cv",
        "tier2_wall_ms",
        "tier2_speed_ratio",
        "tier2_speed_target_ratio",
        "tier2_timing_cv",
        "tier3_median_roofline_pct",
    )
    return {
        key: vector[key]
        for key in keys
        if vector.get(key) not in (None, "", False)
    }


def _performance_class(
    *,
    decision: dict[str, Any],
    tier2_speed_ratio: float | None,
) -> str:
    reason = str(decision.get("reason", ""))
    if decision.get("accepted"):
        return "accepted"
    if reason == "tier_failed:2_speed" and tier2_speed_ratio is not None:
        if tier2_speed_ratio <= 1.03:
            return "tier2_speed_near_miss"
        return "tier2_speed_loss"
    if reason == "discard:not_faster_than_baseline":
        return "tier1_speed_loss"
    if reason == "discard:duplicate_source":
        return "duplicate"
    if reason.startswith("invalid:"):
        return "invalid"
    if reason.startswith("crash:"):
        return "crash"
    if reason.startswith("tier_failed:"):
        return "higher_tier_failed"
    return "rejected"


def _bottleneck_hint(
    *,
    performance_class: str,
    decision: dict[str, Any],
    tier3_median_roofline_pct: Any,
    roofline_values: list[float],
    tier2_speed_ratio: float | None,
) -> str:
    reason = str(decision.get("reason", ""))
    if performance_class in {"invalid", "crash"}:
        return "correctness_or_runtime"
    if performance_class == "duplicate":
        return "no_behavior_change"

    roofline_pct = _optional_float(tier3_median_roofline_pct)
    if roofline_pct is None and roofline_values:
        roofline_pct = sum(roofline_values) / len(roofline_values)
    if roofline_pct is not None:
        if roofline_pct >= 70.0:
            return "bandwidth_or_dtype_limited"
        if roofline_pct <= 40.0:
            return "overhead_or_algorithm_limited"
        return "mixed_roofline"

    if reason == "tier_failed:2_speed" and tier2_speed_ratio is not None:
        return "full_convergence_speed_limited"
    if reason == "discard:not_faster_than_baseline":
        return "tier1_tail_or_launch_limited"
    return "unknown"


def _recommendation(
    *,
    performance_class: str,
    bottleneck_hint: str,
    tier2_speed_ratio: float | None,
    tier2_speed_margin: float | None,
) -> str:
    if performance_class == "accepted":
        return "use as parent or transfer seed"
    if performance_class == "duplicate":
        return "do not retry exact source or payload; change a semantic field"
    if performance_class == "invalid":
        return "fix correctness before measuring speed"
    if performance_class == "crash":
        return "fix runtime/import/applier failure before measuring speed"
    if bottleneck_hint == "bandwidth_or_dtype_limited":
        return "try dtype/layout/A-memory changes rather than cosmetic tail edits"
    if bottleneck_hint == "overhead_or_algorithm_limited":
        return "try reducing launches/setup or iteration count; tail bandwidth is not the bottleneck"
    if performance_class == "tier2_speed_near_miss":
        ratio = f"{tier2_speed_ratio:.4f}" if tier2_speed_ratio is not None else "?"
        margin = f"{tier2_speed_margin:.4f}" if tier2_speed_margin else "?"
        return (
            "near miss: build around this only with a larger lever "
            f"(ratio={ratio}, target={margin})"
        )
    if performance_class == "tier1_speed_loss":
        return "avoid this edit unless combined with a clear launch-count or algorithm change"
    return "needs larger semantic change before another retry"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _optional_float(numerator)
    den = _optional_float(denominator)
    if num is None or den in (None, 0.0):
        return None
    return num / den


def _timing_cv(wall_time_ms: Any, std_ms: Any) -> float | None:
    wall = _optional_float(wall_time_ms)
    std = _optional_float(std_ms)
    if wall in (None, 0.0) or std is None:
        return None
    return std / wall


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: Any, *, ndigits: int = 3) -> float | None:
    value = _optional_float(value)
    if value is None:
        return None
    return round(value, ndigits)


def _slot_key(slot: dict[str, Any]) -> str:
    return "/".join(
        str(slot.get(key, "unknown"))
        for key in ("problem_family", "algorithm", "hardware", "dtype")
    )
