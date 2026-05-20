"""Karpathy-style autoresearch driver for KKT/gap-gated kernel synthesis.

The job of this loop:
  1. Detect orphans from prior crashes (write-ahead `started.json`).
  2. Measure the current seed / persisted champion under multi-rep timing.
  3. For each proposal:
     - Build a rich context (current source, fitness trajectory, history with
       rationales, target metric, hard gate) and ask the proposer for a full-
       source rewrite.
     - Materialize the candidate, eval in the subprocess sandbox.
     - Gate: converged AND fitness < tol AND solve_ms < best * margin.
     - Append a lineage row, promote champion if kept.

Algorithm-agnostic: dispatches to `fista` or `pdhg` via the `algorithm` field
on the eval config. Fitness is `kkt` for FISTA, `gap` for PDHG; both stored
in the same `fitness_final` field so the gating logic stays uniform.

Intentionally absent (per the 2026-05-10 pivot): structured edit grammar,
edit priors, fitness diagnostics, slot-keyed champion store with symlinks,
tier-3 evaluator, transfer seeds. Those proved to be observation-only
machinery that didn't steer behavior. The signal channel is what matters.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from .checkpoint import find_orphans, mark_started
from .lineage import Edit as _LineageEdit
from .lineage import Slot
from .sandbox import SandboxResult, run_kernel, write_eval_config


# ---------- types ----------


@dataclass
class Edit:
    """Slim Edit for the new loop. Mostly a full-source rewrite from the LLM.

    Adapted to the legacy `lineage.Edit` schema for `mark_started` compatibility
    via `_to_lineage_edit`.
    """
    type: str
    rationale: str
    full_source: str = ""
    proposer_role: str = "impl"
    proposer_model: str = ""

    def _to_lineage_edit(self) -> _LineageEdit:
        return _LineageEdit(
            type=self.type,
            payload={"full_source_bytes": len(self.full_source)},
            rationale=self.rationale,
            proposer_role=self.proposer_role,  # type: ignore[arg-type]
            proposer_model=self.proposer_model,
            source="openai",
        )


@dataclass
class EvalScore:
    converged: bool
    fitness_final: float           # KKT (FISTA) or primal-dual gap (PDHG)
    fitness_kind: str              # "kkt" | "gap"
    iters: int
    solve_ms_median: float
    solve_ms_min: float
    solve_ms_max: float
    solve_ms_std: float
    setup_ms_median: float = 0.0
    fitness_trajectory: list[float] = field(default_factory=list)  # compressed
    n_reps: int = 1


@dataclass
class LineageRow:
    id: str
    parent_id: Optional[str]
    generation: int
    created_at: str
    slot: dict
    edit: dict
    source: dict
    score: Optional[dict]
    decision: dict


class Proposer(Protocol):
    def propose(self, ctx: dict) -> Edit: ...


# ---------- helpers ----------


def detect_hardware() -> str:
    s = platform.system().lower()
    m = platform.machine().lower()
    if s == "darwin" and "arm" in m:
        return "apple_silicon"
    return f"{s}_{m}"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _resolve_seed_path(seed_kernel: dict) -> Optional[Path]:
    spec = seed_kernel["module"]
    if spec.endswith(".py"):
        return Path(spec)
    s = importlib.util.find_spec(spec)
    if s is None or s.origin is None:
        return None
    return Path(s.origin)


def _slot_to_dict(slot: Slot) -> dict:
    return {
        "problem_family": slot.problem_family,
        "algorithm": slot.algorithm,
        "hardware": slot.hardware,
        "dtype": slot.dtype,
    }


def _edit_to_dict(edit: Edit) -> dict:
    d = asdict(edit)
    if d["full_source"]:
        d["source_bytes"] = len(d["full_source"])
        d["full_source"] = d["full_source"][:0]  # don't repeat in lineage
    return d


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _load_slot_history(lineage_path: Path, slot: Slot, *, last: int) -> list[dict]:
    if not lineage_path.exists():
        return []
    rows: list[dict] = []
    for line in lineage_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rs = row.get("slot") or {}
        if (
            rs.get("problem_family") == slot.problem_family
            and rs.get("algorithm") == slot.algorithm
            and rs.get("hardware") == slot.hardware
            and rs.get("dtype") == slot.dtype
        ):
            rows.append(row)
    return rows[-last:]


def _compress_trajectory(traj: list[float]) -> list[float]:
    """Down-sample a trajectory to ~9 anchor points: start, quartiles, end,
    plus the iteration of the largest decrease and the largest non-decrease.
    Keeps the prompt small while preserving stall/divergence signal."""
    if not traj:
        return []
    n = len(traj)
    if n <= 9:
        return [float(v) for v in traj]
    quartile_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    diffs = [traj[k + 1] - traj[k] for k in range(n - 1)]
    if diffs:
        worst = int(max(range(len(diffs)), key=lambda k: diffs[k]))
        best = int(min(range(len(diffs)), key=lambda k: diffs[k]))
        idx = sorted(set(quartile_idx + [worst, best, max(0, worst - 1), min(n - 1, best + 1)]))
    else:
        idx = quartile_idx
    return [float(traj[i]) for i in idx]


# ---------- evaluation ----------


def _stats(values: list[float]) -> dict:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _evaluate(
    *,
    problem: Any,
    kernel_module_spec: str,
    kernel_step: str,
    kernel_init: str,
    algorithm: str,
    run_dir: Path,
    max_iters: int,
    tol: float,
    reps: int,
    variant: str,
    problem_backend: str,
    problem_dtype: str,
    cost_model: str,
    warmup_runs: int,
    timeout_s: float,
) -> tuple[Optional[EvalScore], Optional[str]]:
    """Run the kernel `reps` times and aggregate. Returns (score, error)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    solve_times_ms: list[float] = []
    setup_times_ms: list[float] = []
    iters: list[int] = []
    fitness: list[float] = []
    last_traj: list[float] = []
    last_converged = True

    for rep in range(max(1, reps)):
        rep_dir = run_dir / (f"rep_{rep}" if reps > 1 else "")
        rep_dir.mkdir(parents=True, exist_ok=True)
        write_eval_config(
            rep_dir,
            problem,
            kernel_module=kernel_module_spec,
            kernel_step=kernel_step,
            kernel_init=kernel_init,
            variant=variant,
            problem_backend=problem_backend,
            problem_dtype=problem_dtype,
            warmup_runs=warmup_runs,
            max_iters=max_iters,
            tol=tol,
            algorithm=algorithm,
        )
        result = run_kernel(rep_dir, timeout_s=timeout_s)
        if result.status != "completed":
            return None, f"{result.status}:{(result.error_type or '')}:{(result.error_message or '')[:200]}"
        # Cost-model gate: under "single", the candidate is timed including
        # subprocess `prepare_problem` + driver wall_time; under "amortized",
        # only the driver's repeated-solve wall_time. This blocks stuffing
        # arbitrary work into prepare_problem to game the speedup gate.
        if cost_model == "amortized":
            timed_ms = (result.solve_time_s or 0.0) * 1000.0
        else:
            timed_ms = (result.single_solve_time_s
                        or ((result.setup_time_s or 0.0) + (result.solve_time_s or 0.0))) * 1000.0
        solve_times_ms.append(timed_ms)
        setup_times_ms.append((result.setup_time_s or 0.0) * 1000.0)
        iters.append(int(result.iters or 0))
        fitness.append(float(result.kkt_final or 0.0))
        last_converged = bool(result.converged)
        # Trajectory persisted alongside the result, if we extended the eval to write it.
        traj_path = rep_dir / "trajectory.json"
        if traj_path.exists():
            try:
                last_traj = list(json.loads(traj_path.read_text()))
            except json.JSONDecodeError:
                last_traj = []

    s_solve = _stats(solve_times_ms)
    s_setup = _stats(setup_times_ms)
    fitness_kind = "gap" if algorithm == "pdhg" else "kkt"
    score = EvalScore(
        converged=last_converged,
        fitness_final=float(statistics.median(fitness)) if fitness else 0.0,
        fitness_kind=fitness_kind,
        iters=int(statistics.median(iters)) if iters else 0,
        solve_ms_median=s_solve["median"],
        solve_ms_min=s_solve["min"],
        solve_ms_max=s_solve["max"],
        solve_ms_std=s_solve["std"],
        setup_ms_median=s_setup["median"],
        fitness_trajectory=_compress_trajectory(last_traj),
        n_reps=len(solve_times_ms),
    )
    return score, None


# ---------- gating ----------


def _decide(
    cand: EvalScore,
    best: EvalScore,
    fitness_tol: float,
    speedup_margin: float,
) -> tuple[bool, str]:
    if not cand.converged:
        return False, f"discard:not_converged"
    if cand.fitness_final >= fitness_tol:
        return False, f"discard:{cand.fitness_kind}_above_tol"
    if cand.solve_ms_median >= best.solve_ms_median * speedup_margin:
        return False, "discard:not_faster_than_baseline"
    return True, "keep:passed"


# ---------- proposal context ----------


def _compact_history_for_prompt(history: list[dict], last: int = 10) -> list[dict]:
    out: list[dict] = []
    for row in history[-last:]:
        compact = {
            "id": row.get("id", "")[:8],
            "edit_type": (row.get("edit") or {}).get("type"),
            "edit_rationale": (row.get("edit") or {}).get("rationale", "")[:200],
            "decision": (row.get("decision") or {}).get("reason"),
        }
        s = row.get("score") or {}
        if s:
            compact["fitness"] = s.get("fitness_final")
            compact["fitness_kind"] = s.get("fitness_kind")
            compact["solve_ms"] = s.get("solve_ms_median")
            compact["iters"] = s.get("iters")
        out.append(compact)
    return out


def _build_proposal_ctx(
    *,
    slot: Slot,
    current_source_path: Optional[Path],
    current_score: EvalScore,
    history: list[dict],
    algorithm: str,
    fitness_tol: float,
    speedup_margin: float,
    cost_model: str,
    program_md: str,
) -> dict:
    src = ""
    if current_source_path is not None and current_source_path.exists():
        src = current_source_path.read_text(errors="replace")
    return {
        "slot": _slot_to_dict(slot),
        "algorithm": algorithm,
        "fitness_kind": "gap" if algorithm == "pdhg" else "kkt",
        "fitness_tol": fitness_tol,
        "speedup_margin": speedup_margin,
        "cost_model": cost_model,
        "program_md": program_md,
        "current_source": src,
        "current_source_path": str(current_source_path) if current_source_path else None,
        "current_score": asdict(current_score),
        "history": _compact_history_for_prompt(history, last=10),
    }


# ---------- main loop ----------


def run_synth_loop(
    *,
    proposer: Proposer,
    problem: Any,
    seed_kernel: dict,
    slot: Slot,
    state_root: Path,
    n_proposals: int = 50,
    algorithm: str = "fista",
    variant: str = "basic",
    fitness_tol: float = 1e-6,
    max_iters: int = 5000,
    reps: int = 5,
    speedup_margin: float = 0.97,
    problem_backend: str = "native",
    problem_dtype: str = "fp32",
    cost_model: str = "single",
    warmup_runs: int = 1,
    timeout_s: float = 60.0,
    program_md: str = "",
    verbose: bool = False,
) -> list[LineageRow]:
    """Run a full Karpathy-style autoresearch session.

    Returns the lineage rows appended this session (excluding the baseline).
    All durable state lives under `state_root`:
        runs/                 per-proposal artifacts
        lineage.jsonl         append-only experiment log
        program.md            (caller-managed) instruction layer

    The loop is single-threaded by design — Mac MLX can't share a GPU
    cleanly across processes. Cluster parallelism is one slot per node.
    """
    state_root = Path(state_root)
    runs_root = state_root / "runs"
    lineage_path = state_root / "lineage.jsonl"
    runs_root.mkdir(parents=True, exist_ok=True)

    orphans = find_orphans(runs_root, lineage_path)
    if orphans and verbose:
        print(f"[synth] {len(orphans)} orphan(s) from prior crash")

    history = _load_slot_history(lineage_path, slot, last=20)

    # Baseline measurement
    seed_path = _resolve_seed_path(seed_kernel)
    base_dir = runs_root / f"_baseline_{_new_id()}"
    base_score, base_err = _evaluate(
        problem=problem,
        kernel_module_spec=seed_kernel["module"],
        kernel_step=seed_kernel["step"],
        kernel_init=seed_kernel["init"],
        algorithm=algorithm,
        run_dir=base_dir,
        max_iters=max_iters,
        tol=fitness_tol,
        reps=reps,
        variant=variant,
        problem_backend=problem_backend,
        problem_dtype=problem_dtype,
        cost_model=cost_model,
        warmup_runs=warmup_runs,
        timeout_s=timeout_s,
    )
    if base_score is None or not base_score.converged or base_score.fitness_final >= fitness_tol:
        raise RuntimeError(
            f"baseline did not converge: err={base_err} "
            f"score={base_score}"
        )
    if verbose:
        print(f"[synth] baseline {base_score.fitness_kind}={base_score.fitness_final:.2e} "
              f"solve={base_score.solve_ms_median:.1f}±{base_score.solve_ms_std:.1f}ms "
              f"iters={base_score.iters} (n_reps={reps})")

    current_score = base_score
    current_source_path: Optional[Path] = seed_path
    current_kernel_module = seed_kernel["module"]

    appended: list[LineageRow] = []
    for i in range(n_proposals):
        ctx = _build_proposal_ctx(
            slot=slot,
            current_source_path=current_source_path,
            current_score=current_score,
            history=history,
            algorithm=algorithm,
            fitness_tol=fitness_tol,
            speedup_margin=speedup_margin,
            cost_model=cost_model,
            program_md=program_md,
        )

        # Ask proposer
        try:
            edit = proposer.propose(ctx)
        except Exception as exc:  # noqa: BLE001
            row = LineageRow(
                id=_new_id(), parent_id=None, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit={"type": "n/a", "rationale": "", "proposer_role": "impl"},
                source={"path": "", "hash": ""},
                score=None,
                decision={"accepted": False, "reason": f"crash:proposer_error:{type(exc).__name__}"},
            )
            history.append(asdict(row))
            appended.append(row)
            _append_jsonl(lineage_path, asdict(row))
            if verbose:
                print(f"[synth] proposal {i+1}/{n_proposals}: proposer crash: {exc}")
            continue

        if not edit.full_source.strip():
            row = LineageRow(
                id=_new_id(), parent_id=None, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit={"type": edit.type, "rationale": edit.rationale, "proposer_role": edit.proposer_role},
                source={"path": "", "hash": ""},
                score=None,
                decision={"accepted": False, "reason": "discard:empty_source"},
            )
            history.append(asdict(row))
            appended.append(row)
            _append_jsonl(lineage_path, asdict(row))
            if verbose:
                print(f"[synth] proposal {i+1}/{n_proposals}: empty source")
            continue

        source_hash = _hash(edit.full_source)
        if any((row.get("source") or {}).get("hash") == source_hash for row in history):
            row = LineageRow(
                id=_new_id(), parent_id=None, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit={"type": edit.type, "rationale": edit.rationale, "proposer_role": edit.proposer_role},
                source={"path": "", "hash": source_hash},
                score=None,
                decision={"accepted": False, "reason": "discard:duplicate_source"},
            )
            history.append(asdict(row))
            appended.append(row)
            _append_jsonl(lineage_path, asdict(row))
            if verbose:
                print(f"[synth] proposal {i+1}/{n_proposals}: duplicate source")
            continue

        run_id = _new_id()
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = run_dir / "source.py"
        candidate_path.write_text(edit.full_source)

        mark_started(
            run_dir,
            slot,
            edit._to_lineage_edit(),
            source_path=str(candidate_path),
        )

        cand_score, cand_err = _evaluate(
            problem=problem,
            kernel_module_spec=str(candidate_path),
            kernel_step=seed_kernel["step"],
            kernel_init=seed_kernel["init"],
            algorithm=algorithm,
            run_dir=run_dir,
            max_iters=max_iters,
            tol=fitness_tol,
            reps=reps,
            variant=variant,
            problem_backend=problem_backend,
            problem_dtype=problem_dtype,
            cost_model=cost_model,
            warmup_runs=warmup_runs,
            timeout_s=timeout_s,
        )

        if cand_score is None:
            row = LineageRow(
                id=run_id, parent_id=None, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit={"type": edit.type, "rationale": edit.rationale, "proposer_role": edit.proposer_role},
                source={"path": str(candidate_path), "hash": source_hash},
                score=None,
                decision={"accepted": False, "reason": f"crash:{cand_err}"},
            )
            history.append(asdict(row))
            appended.append(row)
            _append_jsonl(lineage_path, asdict(row))
            if verbose:
                print(f"[synth] proposal {i+1}/{n_proposals}: crash {cand_err}")
            continue

        kept, reason = _decide(cand_score, current_score, fitness_tol, speedup_margin)
        row = LineageRow(
            id=run_id, parent_id=None, generation=len(appended) + 1,
            created_at=_now_iso(), slot=_slot_to_dict(slot),
            edit={"type": edit.type, "rationale": edit.rationale, "proposer_role": edit.proposer_role},
            source={"path": str(candidate_path), "hash": source_hash},
            score=asdict(cand_score),
            decision={"accepted": kept, "reason": reason},
        )
        history.append(asdict(row))
        appended.append(row)
        _append_jsonl(lineage_path, asdict(row))

        if verbose:
            tag = "KEEP" if kept else f"discard ({reason.split(':',1)[-1]})"
            print(
                f"[synth] proposal {i+1}/{n_proposals}: {tag} "
                f"{cand_score.fitness_kind}={cand_score.fitness_final:.2e} "
                f"solve={cand_score.solve_ms_median:.1f}±{cand_score.solve_ms_std:.1f}ms "
                f"iters={cand_score.iters}"
            )

        if kept:
            current_source_path = candidate_path
            current_kernel_module = str(candidate_path)
            current_score = cand_score
            (state_root / "champion.py").unlink(missing_ok=True)
            (state_root / "champion.py").write_text(edit.full_source)

    return appended
