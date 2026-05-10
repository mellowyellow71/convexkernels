"""Lineage record schema + JSONL writer.

Implements `docs/schema.md`. The synth loop appends one record per proposal
to `synth_state/lineage.jsonl` (relative to repo root by default).

Records have all fields populated up to the highest tier the proposal reached;
later tiers are absent if a tier failed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Slot:
    problem_family: str
    algorithm: str
    hardware: str
    dtype: str

    def key(self) -> str:
        return f"{self.problem_family}/{self.algorithm}/{self.hardware}/{self.dtype}"


@dataclass
class Edit:
    type: str
    payload: dict
    rationale: str
    proposer_role: Literal["algorithm", "kernel", "impl"]
    proposer_model: str
    source: str  # "claude_subagent" | "manual" | "stub" | "transfer:<src_slot>"


@dataclass
class SourceInfo:
    path: str
    hash: str
    diff_from_parent: str = ""


@dataclass
class Tier1Result:
    passed: bool
    reject_reason: Optional[str] = None  # compile_failed | kkt_increased | nan | timeout
    wall_time_ms: float = 0.0
    setup_time_ms: float = 0.0
    solve_time_ms: float = 0.0
    single_solve_wall_time_ms: float = 0.0
    amortized_wall_time_ms: float = 0.0
    cost_model: str = "single"
    n_reps: int = 1
    wall_time_min_ms: float = 0.0
    wall_time_max_ms: float = 0.0
    wall_time_std_ms: float = 0.0


@dataclass
class Tier2Result:
    passed: bool
    converged: bool
    n_iters: int
    wall_time_ms: float
    kkt_final: float
    setup_time_ms: float = 0.0
    solve_time_ms: float = 0.0
    single_solve_wall_time_ms: float = 0.0
    amortized_wall_time_ms: float = 0.0
    cost_model: str = "single"
    kkt_trajectory_downsampled: list[float] = field(default_factory=list)
    primal_obj_first_last: list[float] = field(default_factory=list)
    n_reps: int = 1
    wall_time_min_ms: float = 0.0
    wall_time_max_ms: float = 0.0
    wall_time_std_ms: float = 0.0
    speed_ref_wall_time_ms: float = 0.0
    speed_ref_margin: float = 0.0
    speed_ref_source: str = ""


@dataclass
class Tier3PerShape:
    m: int
    n: int
    n_iters_med: int
    wall_time_med_ms: float
    kkt_final_med: float
    roofline_pct_med: float
    peak_mem_mb: float
    setup_time_med_ms: float = 0.0
    solve_time_med_ms: float = 0.0
    single_solve_wall_time_med_ms: float = 0.0
    amortized_wall_time_med_ms: float = 0.0
    cost_model: str = "single"
    bytes_per_iter: int = 0
    flops_per_iter: int = 0
    arithmetic_intensity: float = 0.0
    roofline_floor_ms_per_iter: float = 0.0
    measured_ms_per_iter: float = 0.0
    achieved_bandwidth_gb_s: float = 0.0


@dataclass
class Tier3Result:
    passed: bool
    per_shape: list[Tier3PerShape] = field(default_factory=list)
    rank_summary: dict = field(default_factory=dict)


@dataclass
class Decision:
    accepted: bool
    reason: str
    pareto_dominates: list[str] = field(default_factory=list)
    champion_for_slot: bool = False


@dataclass
class LineageRecord:
    id: str
    parent_id: Optional[str]
    generation: int
    created_at: str
    evaluated_at: str
    slot: Slot
    edit: Edit
    source: SourceInfo
    tier1: Tier1Result
    decision: Decision
    tier2: Optional[Tier2Result] = None
    tier3: Optional[Tier3Result] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove tier2/tier3 if absent (schema convention)
        if d["tier2"] is None:
            d.pop("tier2")
        if d["tier3"] is None:
            d.pop("tier3")
        return d


def append_record(record: LineageRecord, jsonl_path: Path) -> None:
    """Append `record` as a JSON line. Uses O_APPEND for atomicity within a single
    process; for multi-process writers, a flock-based sidecar would be needed."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), default=str) + "\n"
    with jsonl_path.open("a") as f:
        f.write(line)


def load_records(jsonl_path: Path) -> list[dict]:
    """Read all lineage records as dicts (each line a JSON object).
    Tolerates partial last-line (truncates)."""
    if not jsonl_path.exists():
        return []
    out: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # Last line may be partial after a crash; ignore
            continue
    return out


def seed_from_neighbors(
    records: Iterable[dict],
    slot: Slot,
    *,
    k: int = 3,
) -> list[Edit]:
    """Return top transferable edits from neighboring slots.

    Neighbor slots follow `docs/schema.md`: same problem family with different
    hardware/dtype, or same algorithm with a different problem family. For
    cross-problem transfer, structured payloads are preferred; full-source
    kernels are skipped because they usually bake in a problem-specific backend
    class and prox.
    """
    records_list = list(records)
    if k <= 0:
        return []

    from .edits import structured_payload_signature

    seen_in_target = {
        structured_payload_signature(
            _adapt_payload_for_target(
                (record.get("edit") or {}).get("payload") or {},
                slot,
            )
        )
        for record in records_list
        if _record_slot(record) == slot
    }
    seen_in_target.discard("")

    candidates: list[tuple[tuple[float, float, str], dict, Slot]] = []
    best_by_signature: dict[str, tuple[tuple[float, float, str], dict, Slot]] = {}
    for record in records_list:
        src_slot = _record_slot(record)
        if src_slot is None or not _is_neighbor_slot(src_slot, slot):
            continue
        if not (record.get("decision") or {}).get("accepted"):
            continue

        edit = record.get("edit") or {}
        payload = edit.get("payload") or {}
        if not payload:
            continue
        if "full_source" in payload and src_slot.problem_family != slot.problem_family:
            continue
        payload = _adapt_payload_for_target(payload, slot)
        if not payload:
            continue

        signature = structured_payload_signature(payload)
        if signature and signature in seen_in_target:
            continue
        if not signature:
            signature = f"full_source:{(record.get('source') or {}).get('hash', '')}"
        if signature == "full_source:":
            continue

        rank = _transfer_rank(record)
        current = best_by_signature.get(signature)
        if current is None or rank < current[0]:
            best_by_signature[signature] = (rank, record, src_slot)

    candidates = sorted(best_by_signature.values(), key=lambda item: item[0])
    transferred: list[Edit] = []
    for _rank, record, src_slot in candidates[:k]:
        edit = record.get("edit") or {}
        payload = _adapt_payload_for_target(edit.get("payload") or {}, slot)
        transferred.append(
            Edit(
                type=str(edit.get("type") or "other"),
                payload=payload,
                rationale=(
                    "Transferred from "
                    f"{src_slot.key()}: {str(edit.get('rationale') or '')}"
                ).strip(),
                proposer_role=edit.get("proposer_role") or "impl",
                proposer_model=edit.get("proposer_model") or "transfer",
                source=f"transfer:{src_slot.key()}",
            )
        )
    return transferred


def _record_slot(record: dict) -> Slot | None:
    raw = record.get("slot") or {}
    try:
        return Slot(
            problem_family=str(raw["problem_family"]),
            algorithm=str(raw["algorithm"]),
            hardware=str(raw["hardware"]),
            dtype=str(raw["dtype"]),
        )
    except KeyError:
        return None


def _is_neighbor_slot(src: Slot, target: Slot) -> bool:
    if src.key() == target.key():
        return False
    same_problem_other_target = (
        src.problem_family == target.problem_family
        and src.algorithm == target.algorithm
        and (src.hardware != target.hardware or src.dtype != target.dtype)
    )
    same_algorithm_other_problem = (
        src.algorithm == target.algorithm
        and src.problem_family != target.problem_family
    )
    return same_problem_other_target or same_algorithm_other_problem


def _transfer_rank(record: dict) -> tuple[float, float, str]:
    tier2 = record.get("tier2") or {}
    ref = float(tier2.get("speed_ref_wall_time_ms") or 0.0)
    wall = float(tier2.get("wall_time_ms") or 0.0)
    if ref > 0.0 and wall > 0.0:
        return (wall / ref, wall, str(record.get("id") or ""))

    tier1 = record.get("tier1") or {}
    wall = float(tier1.get("wall_time_ms") or 0.0)
    if wall > 0.0:
        return (1.0, wall, str(record.get("id") or ""))
    return (float("inf"), float("inf"), str(record.get("id") or ""))


def _adapt_payload_for_target(payload: dict, target: Slot) -> dict:
    adapted = dict(payload)
    if target.problem_family == "nonnegative_lasso":
        adapted.pop("branchless_soft_threshold", None)
        has_behavioral_structured_edit = any(
            key in adapted and adapted[key] not in (None, False, "")
            for key in ("threadgroup_size", "items_per_thread", "remove_bounds_check")
        )
        if not has_behavioral_structured_edit and "full_source" not in adapted:
            return {}
    return adapted
