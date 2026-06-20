"""Autoresearch driver: KKT-verified time-to-target, open algorithm search.

The job of this loop:
  1. Measure the baseline panel (classical solvers) → the bar to beat, as
     each solver's time to reach the trusted KKT target.
  2. Measure the seed (or a resumed checkpoint) under the same ruler.
  3. For each proposal:
     - Build a bounded context from the curated research state (rebuilt from
       the durable experiment tree, never from raw chat history) plus the
       current checkpoint's source, and ask the proposer for a full-source
       `solve()` rewrite. Algorithm choice is part of the search space.
     - Materialize the candidate, eval in the subprocess sandbox.
     - Score: reached target AND total_time_s < champion * margin.
     - Append a lineage row (linked to its parent checkpoint → experiment
       tree). If kept, write a durable checkpoint and advance the champion.

The single optimality ruler is `bench.metrics.trusted_kkt`, computed by the
harness on every iterate — candidate and baseline alike.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from ..bench.curves import baseline_panel, problem_hash, time_to_kkt
from .analyst import Analyst, StubAnalyst
from .checkpoints import CheckpointStore
from .director import CHAMPION, Director, Directive, StubDirector
from .lineage import Edit as _LineageEdit
from .lineage import Slot, load_records
from .research_state import (
    build_research_state,
    build_tree_summary,
    write_research_state,
)
from .sandbox import run_kernel, write_eval_config


# ---------- types ----------


@dataclass
class Edit:
    type: str
    rationale: str
    full_source: str = ""
    proposer_role: str = "impl"
    proposer_model: str = ""
    # Coarse algorithm-family tag (e.g. "fista_gram", "admm", "coordinate_descent").
    # `type` is always "full_source", so the curated-history dedup keys on this
    # instead — otherwise every discarded direction collapses into one bucket.
    algorithm_family: str = ""

    def _to_lineage_edit(self) -> _LineageEdit:
        return _LineageEdit(
            type=self.type,
            payload={
                "full_source_bytes": len(self.full_source),
                "algorithm_family": self.algorithm_family,
            },
            rationale=self.rationale,
            proposer_role=self.proposer_role,  # type: ignore[arg-type]
            proposer_model=self.proposer_model,
            source="openai",
        )


@dataclass
class EvalScore:
    reached_target: bool
    kkt_final: float
    time_to_kkt_s: float       # median over reps (solve-only; inf if not reached)
    total_time_s: float        # median (setup + time_to_kkt); cold-start bar
    setup_s: float
    n_reps: int = 1
    trajectory: list = field(default_factory=list)  # representative rep (t, kkt)


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


def _resume_history(lineage_path: Path, slot: Slot, resume_from: Optional[str]) -> list[dict]:
    """Prior lineage rows for this slot, for repopulating `history` on resume.

    Returns [] for a fresh run (no `resume_from`) or a missing log. Filtering by
    slot keeps a multi-slot state_root from leaking another slot's directions.
    """
    if resume_from is None or not lineage_path.exists():
        return []
    slot_dict = _slot_to_dict(slot)
    return [r for r in load_records(lineage_path) if (r.get("slot") or {}) == slot_dict]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("inf")


# ---------- evaluation ----------


def _evaluate(
    *,
    problem: Any,
    kernel_module_spec: str,
    run_dir: Path,
    kkt_tol: float,
    max_time_s: float,
    reps: int,
    problem_backend: str,
    problem_dtype: str,
    dtype_strategy: str,
    warmup_runs: int,
    timeout_s: float,
) -> tuple[Optional[EvalScore], Optional[str]]:
    """Run the candidate `reps` times and aggregate time-to-target."""
    run_dir.mkdir(parents=True, exist_ok=True)
    times: list[float] = []
    setups: list[float] = []
    kkts: list[float] = []
    reached_all = True
    last_traj: list = []

    for rep in range(max(1, reps)):
        rep_dir = run_dir / (f"rep_{rep}" if reps > 1 else "")
        rep_dir.mkdir(parents=True, exist_ok=True)
        write_eval_config(
            rep_dir,
            problem,
            kernel_module=kernel_module_spec,
            kernel_solve="solve",
            problem_backend=problem_backend,
            problem_dtype=problem_dtype,
            dtype_strategy=dtype_strategy,
            warmup_runs=warmup_runs,
            kkt_tol=kkt_tol,
            max_time_s=max_time_s,
        )
        result = run_kernel(rep_dir, timeout_s=timeout_s)
        if result.status != "completed":
            return None, f"{result.status}:{(result.error_type or '')}:{(result.error_message or '')[:200]}"
        reached_all = reached_all and bool(result.reached_target)
        kkts.append(float(result.kkt_final if result.kkt_final is not None else float("inf")))
        setups.append(float(result.setup_time_s or 0.0))
        ttk = result.time_to_kkt_s
        times.append(float(ttk) if ttk is not None else float("inf"))
        if result.trajectory:
            last_traj = result.trajectory

    time_med = _median(times)
    setup_med = _median(setups)
    total = setup_med + time_med  # inf-safe: inf + finite = inf
    score = EvalScore(
        reached_target=reached_all,
        kkt_final=_median(kkts),
        time_to_kkt_s=time_med,
        total_time_s=total,
        setup_s=setup_med,
        n_reps=len(times),
        trajectory=last_traj,
    )
    return score, None


# ---------- gating ----------


def _decide(
    cand: EvalScore,
    best: EvalScore,
    kkt_tol: float,
    margin: float,
    best_baseline: tuple[Optional[str], float],
) -> tuple[bool, str]:
    if not cand.reached_target or cand.kkt_final > kkt_tol:
        return False, "discard:did_not_reach_target"
    bl_name, bl_time = best_baseline
    bl_tag = ""
    if bl_name is not None and cand.total_time_s > 0:
        bl_tag = f";vs_best_baseline={bl_name}({bl_time / cand.total_time_s:.2f}x)"
    if best.total_time_s == float("inf") or not best.reached_target:
        return True, f"keep:first_valid_champion{bl_tag}"
    if cand.total_time_s >= best.total_time_s * margin:
        return False, f"discard:not_faster_than_champion{bl_tag}"
    ratio = cand.total_time_s / best.total_time_s if best.total_time_s else 0.0
    return True, f"keep:beats_champion({ratio:.2f}x){bl_tag}"


# ---------- strategy (Director) ----------


def _maybe_direct(
    director: Director,
    director_state: dict,
    i: int,
    every_k: int,
    last: Optional[Directive],
    verbose: bool,
) -> Directive:
    """Call the Director on steps where i % every_k == 0, else replay `last`.

    A Director exception degrades to the default (greedy-from-champion) directive
    so a strategy failure never aborts the run — parity with proposer-crash
    resilience.
    """
    if every_k > 1 and i % every_k != 0 and last is not None:
        return last
    try:
        return director.direct(director_state)
    except Exception as exc:  # noqa: BLE001 — strategy is advisory, never fatal
        if verbose:
            print(f"[synth] director failed ({type(exc).__name__}: {exc}); using champion")
        return Directive()


def _resolve_branch(
    directive: Directive,
    store: CheckpointStore,
    champion_id: Optional[str],
    champion_source_path: Optional[Path],
) -> tuple[Optional[str], Optional[Path]]:
    """Resolve the directive's branch point to (parent_id, source_path).

    The CHAMPION sentinel (or an unknown/missing checkpoint id) falls back to the
    current global champion — i.e. today's behavior. A valid non-champion id
    routes the next proposal to branch from that earlier node (backtracking),
    built entirely on the existing CheckpointStore tree.
    """
    if not directive.branch_from or directive.branch_from == CHAMPION:
        return champion_id, champion_source_path
    ck = store.get(directive.branch_from)
    if ck is None:
        return champion_id, champion_source_path
    return ck.id, ck.source_path


def _safe_analyze(analyst: Analyst, candidate_ctx: dict, verbose: bool) -> str:
    """Advisory analyst call; a failure yields no note (never fatal)."""
    try:
        return analyst.analyze(candidate_ctx) or ""
    except Exception as exc:  # noqa: BLE001 — advisory, never fatal
        if verbose:
            print(f"[synth] analyst failed ({type(exc).__name__}: {exc})")
        return ""


# ---------- proposal context ----------


def _build_proposal_ctx(
    *,
    slot: Slot,
    current_source_path: Optional[Path],
    current_score: EvalScore,
    research_state: dict,
    kkt_tol: float,
    margin: float,
    program_md: str,
    directive: Optional[Directive] = None,
) -> dict:
    src = ""
    if current_source_path is not None and current_source_path.exists():
        src = current_source_path.read_text(errors="replace")
    return {
        "slot": _slot_to_dict(slot),
        "kkt_tol": kkt_tol,
        "margin": margin,
        "program_md": program_md,
        "current_source": src,
        "current_source_path": str(current_source_path) if current_source_path else None,
        "current_score": asdict(current_score),
        "research_state": research_state,
        "directive": asdict(directive) if directive is not None else None,
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
    kkt_tol: float = 1e-6,
    max_time_s: float = 60.0,
    reps: int = 3,
    margin: float = 0.97,
    problem_backend: str = "mlx",
    problem_dtype: str = "fp32",
    dtype_strategy: str = "fp32",
    warmup_runs: int = 1,
    timeout_s: float = 120.0,
    program_md: str = "",
    resume_from: Optional[str] = None,
    compute_baselines: bool = True,
    baseline_solvers: tuple[str, ...] = (
        "CLARABEL", "SCS", "OSQP", "ECOS", "sklearn", "adelie",
    ),
    extra_baseline_curves: Optional[dict] = None,
    director: Optional[Director] = None,
    analyst: Optional[Analyst] = None,
    director_every_k: int = 1,
    verbose: bool = False,
) -> list[LineageRow]:
    """Run a full autoresearch session. Durable state under `state_root`:
        runs/                 per-proposal artifacts
        checkpoints/<id>/     durable champion nodes (the experiment tree)
        baselines/<hash>/     cached baseline KKT-vs-time curves
        lineage.jsonl         append-only experiment log (with parent_id)
        research_state.json   curated state rebuilt from the tree each iter
    """
    state_root = Path(state_root)
    runs_root = state_root / "runs"
    lineage_path = state_root / "lineage.jsonl"
    directives_path = state_root / "directives.jsonl"
    runs_root.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(state_root)
    phash = problem_hash(problem)

    # Strategy layer. Defaults (StubDirector/StubAnalyst) reproduce the prior
    # greedy-from-champion behavior exactly: the directive is the no-op sentinel,
    # branches always resolve to the champion, and no analyst notes are produced.
    director = director or StubDirector()
    analyst = analyst or StubAnalyst()
    analyst_notes: list[str] = []

    # ---- baseline panel: the bar to beat ----
    baseline_times: dict[str, float] = {}
    panel: dict[str, list] = {}
    if compute_baselines:
        try:
            panel = baseline_panel(
                problem, solvers=baseline_solvers,
                cache_dir=state_root / "baselines",
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[synth] baseline panel failed: {exc}")
    # Injected curves (e.g. cached Adelie on the hero shape, where the cvxpy
    # interior-point panel is intractable). Persist them under the baselines
    # cache dir so the end-of-run plot overlays them too.
    if extra_baseline_curves:
        bdir = state_root / "baselines" / phash
        bdir.mkdir(parents=True, exist_ok=True)
        for name, curve in extra_baseline_curves.items():
            panel[name] = curve
            (bdir / f"{name}.json").write_text(json.dumps(curve))
    baseline_times = {
        name: time_to_kkt(curve, kkt_tol) for name, curve in panel.items()
    }
    best_baseline = (None, float("inf"))
    if baseline_times:
        bl = min(baseline_times.items(), key=lambda kv: kv[1])
        best_baseline = bl
    if verbose:
        print(f"[synth] baselines time_to_kkt: {baseline_times}")

    # ---- resolve the starting point (resume checkpoint or seed) ----
    # `champion_id` is the current global-champion checkpoint node; per-proposal
    # branch points are resolved from the Director's directive each iteration.
    champion_id: Optional[str] = None
    if resume_from is not None:
        ck = store.get(resume_from)
        if ck is None:
            raise RuntimeError(f"resume checkpoint not found: {resume_from}")
        current_source_path: Optional[Path] = ck.source_path
        current_kernel_module = str(ck.source_path)
        champion_id = ck.id
        if verbose:
            print(f"[synth] resuming from checkpoint {ck.id}")
    else:
        current_source_path = _resolve_seed_path(seed_kernel)
        current_kernel_module = seed_kernel["module"]

    # ---- measure the starting champion ----
    base_dir = runs_root / f"_baseline_{_new_id()}"
    base_score, base_err = _evaluate(
        problem=problem,
        kernel_module_spec=current_kernel_module,
        run_dir=base_dir,
        kkt_tol=kkt_tol,
        max_time_s=max_time_s,
        reps=reps,
        problem_backend=problem_backend,
        problem_dtype=problem_dtype,
        dtype_strategy=dtype_strategy,
        warmup_runs=warmup_runs,
        timeout_s=timeout_s,
    )
    if base_score is None or not base_score.reached_target:
        raise RuntimeError(f"seed did not reach target: err={base_err} score={base_score}")
    if verbose:
        print(f"[synth] seed total={base_score.total_time_s:.3f}s "
              f"(setup={base_score.setup_s:.3f} + ttk={base_score.time_to_kkt_s:.3f}) "
              f"kkt={base_score.kkt_final:.2e}")

    current_score = base_score

    # Root the experiment tree: the seed itself is a durable checkpoint node.
    if resume_from is None:
        seed_src = ""
        if current_source_path is not None and current_source_path.exists():
            seed_src = current_source_path.read_text(errors="replace")
        seed_ck = store.save(
            id=_new_id(), parent_id=None, source=seed_src,
            score=asdict(base_score), trajectory=base_score.trajectory,
            algorithm_tag="seed", problem_hash=phash,
        )
        champion_id = seed_ck.id

    # On resume, repopulate history from the durable lineage log so the curated
    # research state (tried directions) and the duplicate-source guard — both of
    # which scan `history` — survive across sessions.
    history: list[dict] = _resume_history(lineage_path, slot, resume_from)
    if resume_from is not None and verbose:
        print(f"[synth] resume: reloaded {len(history)} prior lineage rows for this slot")

    # One-time analytical bandwidth/AI hint for this problem shape; steers the
    # open algorithm search toward the bandwidth-favourable gradient form.
    cost_model_hint: Optional[dict] = None
    try:
        from .roofline import roofline_hint

        cost_model_hint = roofline_hint(problem)
    except Exception as exc:  # noqa: BLE001 — hint is advisory, never fatal
        if verbose:
            print(f"[synth] roofline hint unavailable: {exc}")

    def _refresh_state() -> dict:
        champ = {
            "source_path": str(current_source_path) if current_source_path else None,
            "total_time_s": current_score.total_time_s,
            "time_to_kkt_s": current_score.time_to_kkt_s,
            "kkt_final": current_score.kkt_final,
        }
        state = build_research_state(
            lineage_rows=history,
            baseline_times=baseline_times,
            champion=champ,
            kkt_tol=kkt_tol,
            cost_model=cost_model_hint,
            analyst_notes=analyst_notes,
        )
        write_research_state(state_root / "research_state.json", state)
        return state

    appended: list[LineageRow] = []
    last_directive: Optional[Directive] = None
    for i in range(n_proposals):
        research_state = _refresh_state()

        # --- strategy: Director picks the branch point + search direction ---
        director_state = {
            "research_state": research_state,
            "tree_summary": build_tree_summary(store.all(), problem_hash=phash),
            "kkt_tol": kkt_tol,
            "champion_id": champion_id,
        }
        directive = _maybe_direct(
            director, director_state, i, director_every_k, last_directive, verbose,
        )
        last_directive = directive
        if not directive.is_default():
            _append_jsonl(directives_path, {"i": i, **asdict(directive)})
        if directive.signal == "stop":
            if verbose:
                print(f"[synth] {i+1}/{n_proposals}: director signalled stop")
            break
        # Resolve the branch point through the existing CheckpointStore tree.
        parent_id, branch_source_path = _resolve_branch(
            directive, store, champion_id, current_source_path,
        )

        ctx = _build_proposal_ctx(
            slot=slot,
            current_source_path=branch_source_path,  # source the proposer rewrites
            current_score=current_score,             # the bar (global champion)
            research_state=research_state,
            kkt_tol=kkt_tol,
            margin=margin,
            program_md=program_md,
            directive=directive,
        )

        def _edit_dict(e) -> dict:
            return {
                "type": e.type, "rationale": e.rationale,
                "proposer_role": e.proposer_role,
                "algorithm_family": e.algorithm_family,
                "directive_signal": directive.signal,
                "branch_from": parent_id,
            }

        def _record(row: LineageRow) -> None:
            history.append(asdict(row))
            appended.append(row)
            _append_jsonl(lineage_path, asdict(row))

        try:
            edit = proposer.propose(ctx)
        except Exception as exc:  # noqa: BLE001
            _record(LineageRow(
                id=_new_id(), parent_id=parent_id, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit={"type": "n/a", "rationale": "", "proposer_role": "impl",
                      "directive_signal": directive.signal, "branch_from": parent_id},
                source={"path": "", "hash": ""}, score=None,
                decision={"accepted": False, "reason": f"crash:proposer_error:{type(exc).__name__}"},
            ))
            if verbose:
                print(f"[synth] {i+1}/{n_proposals}: proposer crash: {exc}")
            continue

        if not edit.full_source.strip():
            _record(LineageRow(
                id=_new_id(), parent_id=parent_id, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit=_edit_dict(edit),
                source={"path": "", "hash": ""}, score=None,
                decision={"accepted": False, "reason": "discard:empty_source"},
            ))
            continue

        source_hash = _hash(edit.full_source)
        if any((row.get("source") or {}).get("hash") == source_hash for row in history):
            _record(LineageRow(
                id=_new_id(), parent_id=parent_id, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit=_edit_dict(edit),
                source={"path": "", "hash": source_hash}, score=None,
                decision={"accepted": False, "reason": "discard:duplicate_source"},
            ))
            continue

        run_id = _new_id()
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = run_dir / "source.py"
        candidate_path.write_text(edit.full_source)

        cand_score, cand_err = _evaluate(
            problem=problem,
            kernel_module_spec=str(candidate_path),
            run_dir=run_dir,
            kkt_tol=kkt_tol,
            max_time_s=max_time_s,
            reps=reps,
            problem_backend=problem_backend,
            problem_dtype=problem_dtype,
            dtype_strategy=dtype_strategy,
            warmup_runs=warmup_runs,
            timeout_s=timeout_s,
        )

        if cand_score is None:
            _record(LineageRow(
                id=run_id, parent_id=parent_id, generation=len(appended) + 1,
                created_at=_now_iso(), slot=_slot_to_dict(slot),
                edit=_edit_dict(edit),
                source={"path": str(candidate_path), "hash": source_hash}, score=None,
                decision={"accepted": False, "reason": f"crash:{cand_err}"},
            ))
            if verbose:
                print(f"[synth] {i+1}/{n_proposals}: crash {cand_err}")
            continue

        kept, reason = _decide(cand_score, current_score, kkt_tol, margin, best_baseline)
        row = LineageRow(
            id=run_id, parent_id=parent_id, generation=len(appended) + 1,
            created_at=_now_iso(), slot=_slot_to_dict(slot),
            edit=_edit_dict(edit),
            source={"path": str(candidate_path), "hash": source_hash},
            score=asdict(cand_score),
            decision={"accepted": kept, "reason": reason},
        )
        _record(row)

        if verbose:
            tag = "KEEP" if kept else f"discard ({reason.split(':', 1)[-1]})"
            print(f"[synth] {i+1}/{n_proposals}: {tag} "
                  f"total={cand_score.total_time_s:.3f}s kkt={cand_score.kkt_final:.2e}")

        if kept:
            # The accepted candidate branches from `parent_id` (the directed
            # branch point) and becomes the new global champion.
            store.save(
                id=run_id, parent_id=parent_id,
                source=edit.full_source,
                score=asdict(cand_score),
                trajectory=cand_score.trajectory,
                algorithm_tag=edit.type,
                problem_hash=phash,
            )
            current_source_path = candidate_path
            current_kernel_module = str(candidate_path)
            current_score = cand_score
            champion_id = run_id
        else:
            # Advisory only: the Analyst explains the failure to the Director on
            # the next step. It never touches the accept/reject gate above.
            note = _safe_analyze(analyst, {
                "rationale": edit.rationale,
                "algorithm_family": edit.algorithm_family,
                "reason": reason,
                "score": asdict(cand_score),
                "champion_total_time_s": current_score.total_time_s,
            }, verbose)
            if note:
                analyst_notes.append(note)

    _refresh_state()

    # Best-effort plot of the final champion vs the baseline panel.
    try:
        from ..bench.plotting import plot_state_root

        plot_state_root(state_root, problem, kkt_tol=kkt_tol)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[synth] plot failed: {exc}")

    return appended
