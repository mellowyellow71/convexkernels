"""Minimal synth-loop driver.

A compact Karpathy-style ratchet loop that:
  1. Detects orphans from prior crashes.
  2. Measures the current seed/champion with the same Tier-1 evaluator.
  2. For each proposal:
     - Asks the proposer for an `Edit` (stub for P3.3, API-backed for P3.4).
     - Allocates a fresh `runs/<id>/` and writes a `started.json` WAL marker.
     - Writes `eval_config.json` + pickled problem.
     - Runs sandbox eval against either the proposed source or the seed kernel.
     - Tier-1 gate: completed + KKT < tier1_kkt_tol + faster than baseline.
     - Appends a lineage record per `docs/schema.md`.
     - Keeps/promotes only faster KKT-valid source.

Tier 2/3 are absent in P3.3 records; they come online in P4.
"""

from __future__ import annotations

import importlib.util
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

from ..bench.shapes import DEFAULT_SHAPES, ShapeSpec
from .applier import (
    apply_config_edit,
    apply_edit,
    has_applicable_payload,
    has_config_payload,
)
from .champion_store import ChampionStore
from .checkpoint import find_orphans, mark_started
from .edits import summarize_edit_priors, write_edit_priors
from .fitness import summarize_fitness_report, write_fitness_report
from .lineage import (
    Decision,
    Edit,
    LineageRecord,
    Slot,
    SourceInfo,
    Tier1Result,
    append_record,
    load_records,
    new_id,
    now_iso,
    seed_from_neighbors,
)
from .tiers import EvalConfig, run_tier1, run_tier2, run_tier3


def detect_hardware() -> str:
    """Coarse hardware tag for slot keys."""
    sys = platform.system().lower()
    machine = platform.machine().lower()
    if sys == "darwin" and "arm" in machine:
        return "apple_silicon"
    return f"{sys}_{machine}"


class Proposer(Protocol):
    def propose(
        self, slot: Slot, parent_id: str | None, history: list[dict]
    ) -> Edit: ...


def _source_path_for_module(module_spec: str) -> Path | None:
    if module_spec.endswith(".py"):
        return Path(module_spec)
    spec = importlib.util.find_spec(module_spec)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _set_proposer_context(
    proposer: Proposer,
    *,
    current_source_path: Path | None,
    current_kernel_summary: dict[str, Any],
    baseline_wall_ms: float | None,
    target_wall_ms: float | None,
    tier1_reps: int,
    tier1_gate_role: str,
    tier2_baseline_wall_ms: float | None,
    tier2_target_wall_ms: float | None,
    tier2_reps: int,
    confirm_tier2_speed: bool,
    algorithm_variant: str,
    problem_dtype: str,
    dtype_strategy: str,
    cost_model: str,
    shape_context: dict[str, Any],
    history_summary: dict[str, Any],
    edit_priors_summary: dict[str, Any],
    fitness_summary: dict[str, Any],
) -> None:
    if current_source_path is not None and hasattr(proposer, "seed_source_path"):
        setattr(proposer, "seed_source_path", Path(current_source_path))
    if hasattr(proposer, "set_runtime_context"):
        proposer.set_runtime_context(
            {
                "baseline_wall_ms": baseline_wall_ms,
                "target_wall_ms": target_wall_ms,
                "tier1_reps": tier1_reps,
                "tier1_gate_role": tier1_gate_role,
                "tier2_baseline_wall_ms": tier2_baseline_wall_ms,
                "tier2_target_wall_ms": tier2_target_wall_ms,
                "tier2_reps": tier2_reps,
                "confirm_tier2_speed": confirm_tier2_speed,
                "algorithm_variant": algorithm_variant,
                "problem_dtype": problem_dtype,
                "dtype_strategy": dtype_strategy,
                "cost_model": cost_model,
                "current_kernel": current_kernel_summary,
                "shape": shape_context,
                "history_summary": history_summary,
                "edit_priors_summary": edit_priors_summary,
                "fitness_summary": fitness_summary,
            }
        )


def _compact_history_record(record: dict) -> dict:
    """Keep prompt history lightweight and avoid sending full prior sources."""
    edit = dict(record.get("edit", {}))
    payload = dict(edit.get("payload", {}))
    if "full_source" in payload:
        payload = {
            key: value
            for key, value in payload.items()
            if key != "full_source"
        }
        payload["full_source_bytes"] = len(record["edit"]["payload"]["full_source"])
    if payload:
        edit["payload"] = payload
    else:
        edit.pop("payload", None)

    compact: dict[str, Any] = {
        "id": record.get("id"),
        "parent_id": record.get("parent_id"),
        "generation": record.get("generation"),
        "edit": edit,
        "tier1": record.get("tier1", {}),
        "decision": record.get("decision", {}),
    }
    if "tier2" in record:
        compact["tier2"] = record["tier2"]
    if "tier3" in record:
        compact["tier3"] = {
            "passed": record["tier3"].get("passed"),
            "rank_summary": record["tier3"].get("rank_summary", {}),
        }
    return compact


def _record_matches_slot(record: dict, slot: Slot) -> bool:
    record_slot = record.get("slot") or {}
    return (
        record_slot.get("problem_family") == slot.problem_family
        and record_slot.get("algorithm") == slot.algorithm
        and record_slot.get("hardware") == slot.hardware
        and record_slot.get("dtype") == slot.dtype
    )


def _gradient_strategy_for_path(
    source_path: Path | None, kernel_module_spec: str
) -> str:
    """Detect the gradient strategy of a kernel given its source file path."""
    source = ""
    if source_path is not None and Path(source_path).exists():
        source = Path(source_path).read_text(errors="replace")
    return _detect_gradient_strategy(source, kernel_module_spec)


def _detect_gradient_strategy(source: str, kernel_module_spec: str) -> str:
    """Classify a kernel source/module as the Gram or direct gradient path.

    Used both for the proposer's runtime focus and to label the roofline cost
    model so Tier-3 reports Gram-path bandwidth/AI for Gram champions. This only
    selects a reporting/metadata strategy; it never changes the compute path.
    """
    is_gram = (
        "LassoGramMLX" in source
        or 'GRADIENT_STRATEGY = "gram"' in source
        or "gram_fista" in kernel_module_spec
    )
    return "gram" if is_gram else "direct"


def _kernel_source_summary(
    source_path: Path | None,
    kernel_module_spec: str,
    *,
    dtype_strategy: str,
    cost_model: str,
) -> dict[str, Any]:
    source = ""
    if source_path is not None and source_path.exists():
        source = source_path.read_text(errors="replace")
    is_gram = _detect_gradient_strategy(source, kernel_module_spec) == "gram"
    if is_gram:
        focus = (
            "Current champion uses Gram-precomputed gradients. Tail-kernel "
            "edits have limited leverage unless paired with a change to the "
            "Gram gradient path; prioritize LassoGramMLX.grad_smooth, "
            "G @ y - c, Gram storage dtype/layout, and KKT-safe precision."
        )
        if cost_model == "single":
            focus += (
                " The active gate charges setup, so any Gram edit must also "
                "reduce or amortize precompute setup."
            )
        elif cost_model == "amortized":
            focus += (
                " The active gate is solve-only, so setup is recorded but "
                "promotion is decided by repeated-solve iteration time."
            )
        gradient_strategy = "gram"
    else:
        focus = (
            "Current champion uses the direct A.T@(A@y-b) gradient path. "
            "Large dense/tall regimes may benefit from proposing Gram "
            "precompute or dtype/layout changes, but KKT remains the gate."
        )
        gradient_strategy = "direct"

    return {
        "gradient_strategy": gradient_strategy,
        "dtype_strategy": dtype_strategy,
        "cost_model": cost_model,
        "focus": focus,
    }


def _problem_shape(problem: Any) -> tuple[int | None, int | None]:
    m = getattr(problem, "m", None)
    n = getattr(problem, "n", None)
    if m is not None and n is not None:
        return int(m), int(n)
    A = getattr(problem, "A", None)
    shape = getattr(A, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None, None


def _shape_context(problem: Any, shape_name: str | None) -> dict[str, Any]:
    m, n = _problem_shape(problem)
    inferred_name = shape_name
    if inferred_name is None and m is not None and n is not None:
        for spec in DEFAULT_SHAPES:
            if spec.m == m and spec.n == n:
                inferred_name = spec.name
                break

    mn = (m or 0) * (n or 0)
    if mn and mn <= 1_500_000:
        regime = "small_dense_launch_overhead_bound"
        guidance = (
            "Small dense shape: do not spend rounds on cosmetic imports, "
            "dataclass-only rewrites, scalar-buffer caching, or fp16 state "
            "storage unless the edit changes convergence or launch count."
        )
    else:
        regime = "large_dense_bandwidth_bound"
        guidance = (
            "Large dense shape: bandwidth and dtype/layout may matter, but "
            "promotion still requires KKT-safe full-convergence speed."
        )

    return {
        "name": inferred_name or "unknown",
        "m": m,
        "n": n,
        "regime": regime,
        "guidance": guidance,
    }


def _summarize_history(history: list[dict]) -> dict[str, Any]:
    attempts = [h for h in history if h.get("edit")]
    if not attempts:
        return {"n_attempts": 0}

    decision_counts = Counter(
        str(h.get("decision", {}).get("reason", "?")) for h in attempts
    )
    edit_counts = Counter(
        f"{h.get('edit', {}).get('type', '?')} -> "
        f"{h.get('decision', {}).get('reason', '?')}"
        for h in attempts
    )
    tier1_passed = [h for h in attempts if h.get("tier1", {}).get("passed")]
    fastest_rejected = None
    rejected_passes = [
        h for h in tier1_passed
        if not h.get("decision", {}).get("accepted")
    ]
    if rejected_passes:
        fastest = min(
            rejected_passes,
            key=lambda h: h.get("tier1", {}).get("wall_time_ms", float("inf")),
        )
        fastest_rejected = {
            "edit_type": fastest.get("edit", {}).get("type", "?"),
            "reason": fastest.get("decision", {}).get("reason", "?"),
            "tier1_wall_time_ms": fastest.get("tier1", {}).get("wall_time_ms"),
            "tier2_wall_time_ms": fastest.get("tier2", {}).get("wall_time_ms"),
            "rationale": fastest.get("edit", {}).get("rationale", "")[:160],
        }

    return {
        "n_attempts": len(attempts),
        "decision_counts": dict(decision_counts.most_common(5)),
        "edit_outcomes": dict(edit_counts.most_common(5)),
        "fastest_rejected": fastest_rejected,
    }


def run_synth_loop(
    *,
    proposer: Proposer,
    problem: Any,
    seed_kernel: dict,  # {"module": str, "step": str, "init": str}
    n_proposals: int = 5,
    state_root: Path,
    slot: Slot | None = None,
    shape_name: str | None = None,
    tier1_kkt_tol: float = 1e-3,
    tier1_max_iters: int = 200,
    tier1_reps: int = 1,
    tier1_escalation_margin: float = 1.0,
    algorithm_variant: Literal["basic", "restart"] = "basic",
    problem_backend: str = "native",
    problem_dtype: str = "fp32",
    dtype_strategy: str = "fp32",
    cost_model: str = "single",
    warmup_runs: int = 1,
    require_speedup: bool = True,
    speedup_margin: float = 0.95,
    promotion_tier: Literal["tier1", "tier2", "tier3"] = "tier1",
    tier2_problem: Any | None = None,
    tier2_kkt_tol: float = 1e-6,
    tier2_max_iters: int = 5000,
    tier2_reps: int = 1,
    require_tier2_speed: bool = True,
    tier2_speed_margin: float = 0.97,
    confirm_tier2_speed: bool = True,
    tier3_shapes: tuple[ShapeSpec, ...] = (),
    tier3_kkt_tol: float = 1e-6,
    tier3_max_iters: int = 5000,
    tier3_reps: int = 3,
    tier3_seed: int = 0,
    transfer_seed_k: int = 0,
    timeout_s: float = 10.0,
    verbose: bool = False,
) -> list[dict]:
    """Run `n_proposals` iterations of the minimal synth loop."""
    state_root = Path(state_root)
    tier1_reps = max(1, int(tier1_reps))
    tier2_reps = max(1, int(tier2_reps))
    tier3_reps = max(1, int(tier3_reps))
    runs_root = state_root / "runs"
    lineage_path = state_root / "synth_state" / "lineage.jsonl"
    edits_path = state_root / "synth_state" / "edits.json"
    fitness_path = state_root / "synth_state" / "fitness.json"

    if slot is None:
        slot = Slot(
            problem_family="lasso",
            algorithm="fista",
            hardware=detect_hardware(),
            dtype="fp64",
        )
    if promotion_tier == "tier3" and not tier3_shapes:
        raise ValueError("promotion_tier='tier3' requires at least one tier3 shape")

    orphans = find_orphans(runs_root, lineage_path)
    if orphans and verbose:
        print(f"[synth] found {len(orphans)} orphaned runs from prior crash: "
              f"{orphans[:3]}{'...' if len(orphans) > 3 else ''}")

    persisted_records = load_records(lineage_path)
    appended: list[dict] = []
    history: list[dict] = [
        _compact_history_record(record)
        for record in persisted_records
        if _record_matches_slot(record, slot)
    ][-25:]
    edit_priors = write_edit_priors(persisted_records, edits_path)
    fitness_report = write_fitness_report(persisted_records, fitness_path)
    transfer_queue = seed_from_neighbors(
        persisted_records,
        slot,
        k=transfer_seed_k,
    )
    seen_source_hashes = {
        record.get("source", {}).get("hash")
        for record in persisted_records
        if record.get("source", {}).get("hash")
        and record.get("source", {}).get("hash") != "seed"
    }
    champion_store = ChampionStore(state_root)
    champion_path, champion_metadata = champion_store.current_source_for_workload(
        slot,
        cost_model,
    )
    champion_summary = champion_metadata.get("summary", {})
    current_kernel_module_spec = (
        str(champion_path) if champion_path is not None else seed_kernel["module"]
    )
    current_source_path = (
        champion_path if champion_path is not None
        else _source_path_for_module(seed_kernel["module"])
    )
    current_problem_dtype = str(
        champion_summary.get("candidate_problem_dtype") or problem_dtype
    )
    current_dtype_strategy = str(
        champion_summary.get("candidate_dtype_strategy") or dtype_strategy
    )
    parent_id: str | None = None
    generation = 1
    baseline_wall_ms: float | None = None
    tier2_baseline_wall_ms: float | None = None
    current_gradient_strategy = _gradient_strategy_for_path(
        current_source_path, current_kernel_module_spec
    )
    eval_config = EvalConfig(
        seed_kernel=seed_kernel,
        variant=algorithm_variant,
        problem_backend=problem_backend,
        problem_dtype=current_problem_dtype,
        dtype_strategy=current_dtype_strategy,
        gradient_strategy=current_gradient_strategy,
        cost_model=cost_model,
        warmup_runs=warmup_runs,
        timeout_s=timeout_s,
    )
    tier2_speed_gate = (
        require_speedup
        and require_tier2_speed
        and promotion_tier in {"tier2", "tier3"}
    )
    tier1_gate_margin = (
        speedup_margin if promotion_tier == "tier1"
        else tier1_escalation_margin
    )
    tier1_gate_role = (
        "promotion" if promotion_tier == "tier1"
        else "tier2_escalation"
    )

    if require_speedup:
        baseline_run_dir = runs_root / f"_baseline_{new_id()}"
        baseline_tier1, baseline = run_tier1(
            run_dir=baseline_run_dir,
            problem=problem,
            kernel_module=current_kernel_module_spec,
            config=eval_config,
            max_iters=tier1_max_iters,
            tol=tier1_kkt_tol,
            reps=tier1_reps,
        )
        if not (
            baseline_tier1.passed
            and baseline.wall_time_s is not None
        ):
            raise RuntimeError(
                "Current seed/champion failed Tier-1 baseline evaluation: "
                f"status={baseline.status}, kkt={baseline.kkt_final}, "
                f"error={baseline.error_message}"
            )
        baseline_wall_ms = baseline_tier1.wall_time_ms
        baseline_history = {
            "kind": "baseline",
            "source": {"path": str(current_source_path or current_kernel_module_spec)},
            "tier1": {
                "wall_time_ms": baseline_wall_ms,
                "n_reps": baseline_tier1.n_reps,
                "wall_time_min_ms": baseline_tier1.wall_time_min_ms,
                "wall_time_max_ms": baseline_tier1.wall_time_max_ms,
                "wall_time_std_ms": baseline_tier1.wall_time_std_ms,
                "kkt_final": baseline.kkt_final,
            },
            "decision": {"reason": "baseline"},
        }
        if tier2_speed_gate:
            tier2_baseline_dir = runs_root / f"_baseline_tier2_{new_id()}"
            baseline_tier2, baseline_tier2_result = run_tier2(
                run_dir=tier2_baseline_dir,
                problem=tier2_problem or problem,
                kernel_module=current_kernel_module_spec,
                config=eval_config,
                max_iters=tier2_max_iters,
                tol=tier2_kkt_tol,
                reps=tier2_reps,
            )
            if not baseline_tier2.passed:
                raise RuntimeError(
                    "Current seed/champion failed Tier-2 baseline evaluation: "
                    f"status={baseline_tier2_result.status}, "
                    f"kkt={baseline_tier2_result.kkt_final}, "
                    f"error={baseline_tier2_result.error_message}"
                )
            tier2_baseline_wall_ms = baseline_tier2.wall_time_ms
            baseline_history["tier2"] = {
                "wall_time_ms": tier2_baseline_wall_ms,
                "n_reps": baseline_tier2.n_reps,
                "wall_time_min_ms": baseline_tier2.wall_time_min_ms,
                "wall_time_max_ms": baseline_tier2.wall_time_max_ms,
                "wall_time_std_ms": baseline_tier2.wall_time_std_ms,
                "kkt_final": baseline_tier2.kkt_final,
                "n_iters": baseline_tier2.n_iters,
            }
        history.append(baseline_history)
        if verbose:
            print(
                f"[synth] baseline kkt={baseline.kkt_final:.2e} "
                f"wall={baseline_wall_ms:.1f}ms reps={baseline_tier1.n_reps} "
                f"target<{baseline_wall_ms * tier1_gate_margin:.1f}ms"
            )
            if tier2_baseline_wall_ms is not None:
                print(
                    f"[synth] tier2 baseline wall={tier2_baseline_wall_ms:.1f}ms "
                    f"reps={tier2_reps} target<"
                    f"{tier2_baseline_wall_ms * tier2_speed_margin:.1f}ms"
                )

    if transfer_queue and verbose:
        print(
            f"[synth] queued {len(transfer_queue)} transfer seed edits "
            f"for {slot.key()}"
        )

    for i in range(n_proposals):
        target_wall_ms = (
            baseline_wall_ms * speedup_margin
            if baseline_wall_ms is not None and promotion_tier == "tier1"
            else (
                baseline_wall_ms * tier1_escalation_margin
                if baseline_wall_ms is not None else None
            )
        )
        tier2_target_wall_ms = (
            tier2_baseline_wall_ms * tier2_speed_margin
            if tier2_baseline_wall_ms is not None else None
        )
        _set_proposer_context(
            proposer,
            current_source_path=current_source_path,
            current_kernel_summary=_kernel_source_summary(
                current_source_path,
                current_kernel_module_spec,
                dtype_strategy=current_dtype_strategy,
                cost_model=cost_model,
            ),
            baseline_wall_ms=baseline_wall_ms,
            target_wall_ms=target_wall_ms,
            tier1_reps=tier1_reps,
            tier1_gate_role=tier1_gate_role,
            tier2_baseline_wall_ms=tier2_baseline_wall_ms,
            tier2_target_wall_ms=tier2_target_wall_ms,
            tier2_reps=tier2_reps,
            confirm_tier2_speed=confirm_tier2_speed,
            algorithm_variant=algorithm_variant,
            problem_dtype=current_problem_dtype,
            dtype_strategy=current_dtype_strategy,
            cost_model=cost_model,
            shape_context=_shape_context(problem, shape_name),
            history_summary=_summarize_history(history),
            edit_priors_summary=summarize_edit_priors(edit_priors, slot),
            fitness_summary=summarize_fitness_report(fitness_report, slot.key()),
        )
        rec_id = new_id()
        record_parent_id = parent_id
        record_generation = generation
        run_dir = runs_root / rec_id
        try:
            if transfer_queue:
                edit = transfer_queue.pop(0)
            else:
                edit = proposer.propose(slot, parent_id=parent_id, history=history)
        except Exception as exc:  # noqa: BLE001
            edit = Edit(
                type="proposer_error",
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                rationale=f"proposer failed before emitting an edit: {exc}",
                proposer_role="impl",
                proposer_model=getattr(proposer, "model", type(proposer).__name__),
                source="proposer_error",
            )
            record = LineageRecord(
                id=rec_id,
                parent_id=record_parent_id,
                generation=record_generation,
                created_at=now_iso(),
                evaluated_at=now_iso(),
                slot=slot,
                edit=edit,
                source=SourceInfo(path="", hash=""),
                tier1=Tier1Result(
                    passed=False,
                    reject_reason="proposer_error",
                    wall_time_ms=0.0,
                ),
                decision=Decision(
                    accepted=False,
                    reason=f"crash:proposer_error:{type(exc).__name__}",
                ),
            )
            append_record(record, lineage_path)
            record_dict = record.to_dict()
            persisted_records.append(record_dict)
            edit_priors = write_edit_priors(persisted_records, edits_path)
            fitness_report = write_fitness_report(persisted_records, fitness_path)
            history.append(_compact_history_record(record_dict))
            appended.append(record_dict)
            if verbose:
                print(
                    f"[synth {i+1}/{n_proposals}] {record.decision.reason} "
                    f"edit=proposer_error"
                )
            continue

        candidate_eval_config = eval_config

        # If the edit can produce source, apply it and use the file path.
        # Config-only edits materialize a source wrapper with prepare_problem().
        # Otherwise the loop falls back to the seed kernel (used by stub/MVP).
        try:
            if has_applicable_payload(edit):
                parent_source_path = (
                    current_source_path
                    or _source_path_for_module(current_kernel_module_spec)
                )
                source_info = apply_edit(
                    edit,
                    run_dir,
                    parent_source_path=parent_source_path,
                )
                kernel_module_spec = source_info.path
                if has_config_payload(edit):
                    (
                        source_info,
                        kernel_module_spec,
                        next_problem_dtype,
                        next_dtype_strategy,
                    ) = apply_config_edit(
                        edit,
                        run_dir,
                        parent_source_path=Path(source_info.path),
                        default_kernel_module=current_kernel_module_spec,
                        default_problem_dtype=current_problem_dtype,
                        default_dtype_strategy=current_dtype_strategy,
                    )
                    candidate_eval_config = EvalConfig(
                        seed_kernel=seed_kernel,
                        variant=algorithm_variant,
                        problem_backend=problem_backend,
                        problem_dtype=next_problem_dtype,
                        dtype_strategy=next_dtype_strategy,
                        gradient_strategy=_gradient_strategy_for_path(
                            Path(source_info.path), kernel_module_spec
                        ),
                        cost_model=cost_model,
                        warmup_runs=warmup_runs,
                        timeout_s=timeout_s,
                    )
            elif has_config_payload(edit):
                parent_source_path = (
                    current_source_path
                    or _source_path_for_module(current_kernel_module_spec)
                )
                (
                    source_info,
                    kernel_module_spec,
                    next_problem_dtype,
                    next_dtype_strategy,
                ) = apply_config_edit(
                    edit,
                    run_dir,
                    parent_source_path=parent_source_path,
                    default_kernel_module=current_kernel_module_spec,
                    default_problem_dtype=current_problem_dtype,
                    default_dtype_strategy=current_dtype_strategy,
                )
                candidate_eval_config = EvalConfig(
                    seed_kernel=seed_kernel,
                    variant=algorithm_variant,
                    problem_backend=problem_backend,
                    problem_dtype=next_problem_dtype,
                    dtype_strategy=next_dtype_strategy,
                    gradient_strategy=_gradient_strategy_for_path(
                        Path(source_info.path), kernel_module_spec
                    ),
                    cost_model=cost_model,
                    warmup_runs=warmup_runs,
                    timeout_s=timeout_s,
                )
            else:
                source_info = SourceInfo(path=seed_kernel["module"], hash="seed")
                kernel_module_spec = seed_kernel["module"]
        except Exception as exc:  # noqa: BLE001
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "applier_error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n"
            )
            record = LineageRecord(
                id=rec_id,
                parent_id=record_parent_id,
                generation=record_generation,
                created_at=now_iso(),
                evaluated_at=now_iso(),
                slot=slot,
                edit=edit,
                source=SourceInfo(path="", hash=""),
                tier1=Tier1Result(
                    passed=False,
                    reject_reason="applier_error",
                    wall_time_ms=0.0,
                ),
                decision=Decision(
                    accepted=False,
                    reason=f"invalid:applier_error:{type(exc).__name__}",
                ),
            )
            append_record(record, lineage_path)
            record_dict = record.to_dict()
            persisted_records.append(record_dict)
            edit_priors = write_edit_priors(persisted_records, edits_path)
            fitness_report = write_fitness_report(persisted_records, fitness_path)
            history.append(_compact_history_record(record_dict))
            appended.append(record_dict)
            if verbose:
                print(
                    f"[synth {i+1}/{n_proposals}] {record.decision.reason} "
                    f"edit={edit.type}"
                )
            continue

        # WAL: write `started.json` BEFORE evaluation begins
        mark_started(run_dir, slot, edit, source_path=source_info.path)

        if source_info.hash in seen_source_hashes:
            tier1 = Tier1Result(
                passed=False,
                reject_reason="duplicate_source",
                wall_time_ms=0.0,
            )
            record = LineageRecord(
                id=rec_id,
                parent_id=record_parent_id,
                generation=record_generation,
                created_at=now_iso(),
                evaluated_at=now_iso(),
                slot=slot,
                edit=edit,
                source=source_info,
                tier1=tier1,
                decision=Decision(
                    accepted=False,
                    reason="discard:duplicate_source",
                ),
            )
            append_record(record, lineage_path)
            record_dict = record.to_dict()
            persisted_records.append(record_dict)
            edit_priors = write_edit_priors(persisted_records, edits_path)
            fitness_report = write_fitness_report(persisted_records, fitness_path)
            history.append(_compact_history_record(record_dict))
            appended.append(record_dict)
            if verbose:
                print(f"[synth {i+1}/{n_proposals}] "
                      f"discard:duplicate_source edit={edit.type}")
            continue

        tier1, result = run_tier1(
            run_dir=run_dir,
            problem=problem,
            kernel_module=kernel_module_spec,
            config=candidate_eval_config,
            max_iters=tier1_max_iters,
            tol=tier1_kkt_tol,
            reps=tier1_reps,
        )

        # Tier-1 correctness gate, then Karpathy-style performance ratchet.
        proposal_wall_ms = tier1.wall_time_ms
        tier2 = None
        tier3 = None
        if tier1.passed:
            faster = (
                not require_speedup
                or baseline_wall_ms is None
                or proposal_wall_ms < baseline_wall_ms * tier1_gate_margin
            )
            if faster:
                tier_passed = True
                decision_reason = (
                    "keep:tier1_speedup" if require_speedup else "tier1_passed"
                )

                if promotion_tier in {"tier2", "tier3"}:
                    tier2, _ = run_tier2(
                        run_dir=run_dir,
                        problem=tier2_problem or problem,
                        kernel_module=kernel_module_spec,
                        config=candidate_eval_config,
                        max_iters=tier2_max_iters,
                        tol=tier2_kkt_tol,
                        reps=tier2_reps,
                    )
                    if not tier2.passed:
                        tier_passed = False
                        decision_reason = "tier_failed:2"
                    elif tier2_speed_gate and tier2_baseline_wall_ms is not None:
                        speed_ref_wall_ms = tier2_baseline_wall_ms
                        speed_ref_source = "startup"
                        if (
                            confirm_tier2_speed
                            and tier2.wall_time_ms
                            < tier2_baseline_wall_ms * tier2_speed_margin
                        ):
                            confirm_tier2, _ = run_tier2(
                                run_dir=run_dir / "tier2_speed_ref",
                                problem=tier2_problem or problem,
                                kernel_module=current_kernel_module_spec,
                                config=eval_config,
                                max_iters=tier2_max_iters,
                                tol=tier2_kkt_tol,
                                reps=tier2_reps,
                            )
                            if not confirm_tier2.passed:
                                tier_passed = False
                                decision_reason = "tier_failed:2_speed_ref"
                            else:
                                speed_ref_wall_ms = confirm_tier2.wall_time_ms
                                speed_ref_source = "paired"

                        tier2.speed_ref_wall_time_ms = speed_ref_wall_ms
                        tier2.speed_ref_margin = tier2_speed_margin
                        tier2.speed_ref_source = speed_ref_source
                        if (
                            tier_passed
                            and tier2.wall_time_ms
                            >= speed_ref_wall_ms * tier2_speed_margin
                        ):
                            tier_passed = False
                            decision_reason = "tier_failed:2_speed"

                if tier_passed and promotion_tier == "tier3":
                    tier3 = run_tier3(
                        run_dir=run_dir,
                        shapes=tier3_shapes,
                        kernel_module=kernel_module_spec,
                        config=candidate_eval_config,
                        max_iters=tier3_max_iters,
                        tol=tier3_kkt_tol,
                        reps=tier3_reps,
                        seed=tier3_seed,
                    )
                    if not tier3.passed:
                        tier_passed = False
                        decision_reason = "tier_failed:3"

                if tier_passed:
                    if promotion_tier == "tier2":
                        decision_reason = "keep:tier2_passed"
                    elif promotion_tier == "tier3":
                        decision_reason = "keep:tier3_passed"
                    promoted = False
                    if has_applicable_payload(edit) or has_config_payload(edit):
                        promoted_path = champion_store.promote(
                            slot=slot,
                            source_path=Path(source_info.path),
                            record_id=rec_id,
                            source_hash=source_info.hash,
                            workload_key=cost_model,
                            summary={
                                "promotion_tier": promotion_tier,
                                "cost_model": cost_model,
                                "workload_key": cost_model,
                                "problem_dtype": problem_dtype,
                                "candidate_problem_dtype": (
                                    candidate_eval_config.problem_dtype
                                ),
                                "candidate_dtype_strategy": (
                                    candidate_eval_config.dtype_strategy
                                ),
                                "tier1_wall_time_ms": proposal_wall_ms,
                                "tier1_setup_time_ms": tier1.setup_time_ms,
                                "tier1_solve_time_ms": tier1.solve_time_ms,
                                "tier1_single_solve_wall_time_ms": (
                                    tier1.single_solve_wall_time_ms
                                ),
                                "tier1_amortized_wall_time_ms": (
                                    tier1.amortized_wall_time_ms
                                ),
                                "tier1_reps": tier1.n_reps,
                                "tier1_wall_time_min_ms": tier1.wall_time_min_ms,
                                "tier1_wall_time_max_ms": tier1.wall_time_max_ms,
                                "tier1_wall_time_std_ms": tier1.wall_time_std_ms,
                                "tier1_kkt_final": result.kkt_final,
                                "tier2_wall_time_ms": (
                                    tier2.wall_time_ms if tier2 else None
                                ),
                                "tier2_setup_time_ms": (
                                    tier2.setup_time_ms if tier2 else None
                                ),
                                "tier2_solve_time_ms": (
                                    tier2.solve_time_ms if tier2 else None
                                ),
                                "tier2_single_solve_wall_time_ms": (
                                    tier2.single_solve_wall_time_ms if tier2 else None
                                ),
                                "tier2_amortized_wall_time_ms": (
                                    tier2.amortized_wall_time_ms if tier2 else None
                                ),
                                "tier2_reps": tier2.n_reps if tier2 else None,
                                "tier2_speed_margin": (
                                    tier2_speed_margin if tier2_speed_gate else None
                                ),
                                "tier2_speed_ref_wall_time_ms": (
                                    tier2.speed_ref_wall_time_ms if tier2 else None
                                ),
                                "tier2_speed_ref_source": (
                                    tier2.speed_ref_source if tier2 else None
                                ),
                                "tier3_rank_summary": (
                                    tier3.rank_summary if tier3 else None
                                ),
                            },
                        )
                        current_kernel_module_spec = str(promoted_path)
                        current_source_path = promoted_path
                        current_problem_dtype = candidate_eval_config.problem_dtype
                        current_dtype_strategy = candidate_eval_config.dtype_strategy
                        baseline_wall_ms = proposal_wall_ms
                        if tier2 is not None and tier2_speed_gate:
                            tier2_baseline_wall_ms = tier2.wall_time_ms
                        parent_id = rec_id
                        generation += 1
                        promoted = True
                    decision = Decision(
                        accepted=True,
                        reason=decision_reason,
                        pareto_dominates=[],
                        champion_for_slot=promoted,
                    )
                else:
                    decision = Decision(
                        accepted=False,
                        reason=decision_reason,
                    )
            else:
                decision = Decision(
                    accepted=False,
                    reason="discard:not_faster_than_baseline",
                )
        else:
            reject_reason = (
                result.status if result.status != "completed"
                else "kkt_above_tier1_tol"
            )
            reason_prefix = "crash" if result.status != "completed" else "invalid"
            decision = Decision(
                accepted=False,
                reason=f"{reason_prefix}:{reject_reason}",
            )

        record = LineageRecord(
            id=rec_id,
            parent_id=record_parent_id,
            generation=record_generation,
            created_at=now_iso(),
            evaluated_at=now_iso(),
            slot=slot,
            edit=edit,
            source=source_info,
            tier1=tier1,
            decision=decision,
            tier2=tier2,
            tier3=tier3,
        )
        append_record(record, lineage_path)
        record_dict = record.to_dict()
        if source_info.hash and source_info.hash != "seed":
            seen_source_hashes.add(source_info.hash)
        persisted_records.append(record_dict)
        edit_priors = write_edit_priors(persisted_records, edits_path)
        fitness_report = write_fitness_report(persisted_records, fitness_path)
        history.append(_compact_history_record(record_dict))
        appended.append(record_dict)

        if verbose:
            kkt_str = f"{result.kkt_final:.2e}" if result.kkt_final else "n/a"
            wall_str = f"{proposal_wall_ms:.1f}ms"
            base_str = (
                f" baseline={baseline_wall_ms:.1f}ms"
                if baseline_wall_ms is not None else ""
            )
            print(f"[synth {i+1}/{n_proposals}] {decision.reason} "
                  f"kkt={kkt_str} wall={wall_str}{base_str} edit={edit.type}")

    return appended
