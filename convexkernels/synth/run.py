"""CLI driver for the autoresearch loop.

Usage:
  python -m convexkernels.synth.run \\
    --slot lasso/fista/apple_silicon/fp32 \\
    --shape tall_medium \\
    --seed convexkernels.kernels.mlx.seeds.fista_step_v0 \\
    --proposer openai \\
    --n-proposals 50 \\
    --reps 5 \\
    --state-root ./synth_run_$(date +%Y%m%d) \\
    --program-md convexkernels/synth/program.md

Slot dispatch: `<problem_family>/<algorithm>/<hardware>/<dtype>`. Problem
families: `lasso`, `nonneg_lasso`, `total_variation_1d`. Algorithms:
`fista`, `pdhg`. Hardware tag is a free-form string (default = auto).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .lineage import Slot
from .loop import detect_hardware, run_synth_loop
from .proposers.openai import OpenAIProposer
from .proposers.stub import StubProposer


def _make_problem(problem_family: str, shape_name: str):
    """Construct a problem instance for the chosen family + bench shape."""
    from ..bench.shapes import (
        DEFAULT_SHAPES,
        DEFAULT_TV1D_SHAPES,
        make_synthetic_lasso,
        make_synthetic_nonnegative_lasso,
        make_synthetic_tv_1d,
    )
    from ..bench.eq_qp_shapes import (
        DEFAULT_BP_SHAPES,
        DEFAULT_EQ_QP_SHAPES,
        make_synthetic_basis_pursuit,
        make_synthetic_eq_qp,
    )

    if problem_family in {"lasso", "nonneg_lasso", "lasso_admm"}:
        spec = next((s for s in DEFAULT_SHAPES if s.name == shape_name), None)
        if spec is None:
            raise ValueError(f"unknown lasso shape {shape_name!r}; pick from {[s.name for s in DEFAULT_SHAPES]}")
        if problem_family == "lasso":
            return make_synthetic_lasso(spec)
        if problem_family == "nonneg_lasso":
            return make_synthetic_nonnegative_lasso(spec)
        # lasso_admm: build the regular Lasso then wrap.
        from ..frontend.lasso_admm import LassoAdmm
        base = make_synthetic_lasso(spec)
        return LassoAdmm(base.A, base.b, base.lam)
    if problem_family == "total_variation_1d":
        spec = next((s for s in DEFAULT_TV1D_SHAPES if s.name == shape_name), None)
        if spec is None:
            raise ValueError(f"unknown tv1d shape {shape_name!r}; pick from {[s.name for s in DEFAULT_TV1D_SHAPES]}")
        return make_synthetic_tv_1d(spec)
    if problem_family == "equality_qp":
        spec = next((s for s in DEFAULT_EQ_QP_SHAPES if s.name == shape_name), None)
        if spec is None:
            raise ValueError(f"unknown equality_qp shape {shape_name!r}; pick from {[s.name for s in DEFAULT_EQ_QP_SHAPES]}")
        return make_synthetic_eq_qp(spec)
    if problem_family == "basis_pursuit":
        spec = next((s for s in DEFAULT_BP_SHAPES if s.name == shape_name), None)
        if spec is None:
            raise ValueError(f"unknown basis_pursuit shape {shape_name!r}; pick from {[s.name for s in DEFAULT_BP_SHAPES]}")
        return make_synthetic_basis_pursuit(spec)
    if problem_family == "lasso_path":
        from ..bench.path_shapes import (
            DEFAULT_PATH_SHAPES, make_path_problem,
        )
        from ..frontend.lasso_path import LassoPath
        spec = next((s for s in DEFAULT_PATH_SHAPES if s.name == shape_name), None)
        if spec is None:
            raise ValueError(
                f"unknown lasso_path shape {shape_name!r}; pick from "
                f"{[s.name for s in DEFAULT_PATH_SHAPES]}"
            )
        A, b, lambdas = make_path_problem(spec)
        return LassoPath(A, b, lambdas)
    raise ValueError(f"unknown problem_family {problem_family!r}")


# Public alias (used by scripts/plot_gap_time.py).
make_problem = _make_problem


def _default_seed_kernel(problem_family: str, backend: str) -> dict:
    """Resolve a seed kernel exposing `solve()` by (problem_family, backend).

    Algorithm is no longer part of the spec — the seed is just the starting
    point for the open search. Each seed module exposes `solve(problem,
    recorder, *, kkt_tol, max_time_s)`.
    """
    if backend == "mlx":
        if problem_family == "lasso_path":
            return {"module": "convexkernels.kernels.mlx.seeds.gram_fista_path_v0"}
        if problem_family in {"lasso", "nonneg_lasso"}:
            return {"module": "convexkernels.kernels.mlx.seeds.fista_step_v0"}
    elif backend == "native":
        if problem_family in {"lasso", "nonneg_lasso", "lasso_path"}:
            return {"module": "convexkernels.kernels.numpy_solve_ref"}
    raise ValueError(f"no default solve-seed for ({problem_family}, backend={backend})")


def _parse_slot_spec(slot_str: str) -> Slot:
    """Spec is `problem_family/hardware`. Algorithm/dtype are open (search
    space), kept on the Slot as the constant tag "open" for telemetry."""
    parts = slot_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"slot must be problem_family/hardware, got {slot_str!r}")
    pf, hw = parts
    algo, dt = "open", "open"
    if hw == "auto":
        hw = detect_hardware()
    return Slot(problem_family=pf, algorithm=algo, hardware=hw, dtype=dt)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, help="problem_family/hardware (e.g. lasso_path/apple_silicon)")
    p.add_argument("--shape", required=True, help="bench shape name (e.g. path_tall_medium)")
    p.add_argument("--seed", default=None, help="seed kernel module dotted-path (auto if omitted)")
    p.add_argument("--proposer", choices=["openai", "stub", "algopool"], default="openai")
    p.add_argument("--model", default="gpt-5.5", help="OpenAI model name")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--api-timeout-s", type=float, default=240.0)
    p.add_argument("--n-proposals", type=int, default=50)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--kkt-tol", type=float, default=1e-6, help="trusted KKT target")
    p.add_argument("--max-time-s", type=float, default=60.0, help="per-solve wall budget")
    p.add_argument("--margin", type=float, default=0.97, help="must be this much faster than champion")
    p.add_argument("--problem-backend", choices=["native", "mlx"], default="mlx")
    p.add_argument("--problem-dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--dtype-strategy", default="fp32")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--state-root", required=True, type=Path)
    p.add_argument("--program-md", default="convexkernels/synth/program.md", type=Path)
    p.add_argument("--resume-from", default=None, help="checkpoint id to branch from")
    p.add_argument("--no-baselines", action="store_true", help="skip the baseline panel")
    p.add_argument("--baseline-solvers", default="CLARABEL,SCS,OSQP,ECOS,sklearn,adelie")
    p.add_argument("--adelie-cache", default="auto",
                   help="path to a cached Adelie .npz to inject as the bar-to-beat "
                        "('auto' = convexkernels/bench/cache/adelie_path_<shape>.npz; "
                        "'none' to disable)")
    p.add_argument("--director", choices=["openai", "stub", "off"], default="off",
                   help="strategist that picks branch point + direction; 'off'/'stub' "
                        "= greedy-from-champion (today's behavior)")
    p.add_argument("--director-model", default="gpt-5.5")
    p.add_argument("--director-reasoning-effort", default="medium")
    p.add_argument("--director-every-k", type=int, default=1,
                   help="call the director every K proposals (cost control)")
    p.add_argument("--analyst", choices=["openai", "off"], default="off",
                   help="advisory LLM that explains failures to the director (never gates)")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args(argv)

    slot = _parse_slot_spec(args.slot)
    problem = _make_problem(slot.problem_family, args.shape)
    if args.seed is None:
        seed_kernel = _default_seed_kernel(slot.problem_family, args.problem_backend)
    else:
        seed_kernel = {"module": args.seed}

    program_md = ""
    if args.program_md and args.program_md.exists():
        program_md = args.program_md.read_text()

    # Resolve an optional cached Adelie reference to inject as the bar-to-beat.
    extra_baseline_curves = None
    if args.adelie_cache != "none":
        if args.adelie_cache == "auto":
            cache_path = Path("convexkernels/bench/cache") / f"adelie_path_{args.shape}.npz"
        else:
            cache_path = Path(args.adelie_cache)
        if cache_path.exists():
            from ..bench.curves import cached_adelie_curve
            try:
                extra_baseline_curves = {"ADELIE": cached_adelie_curve(problem, cache_path)}
                print(f"[run] injected Adelie bar from {cache_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[run] could not load Adelie cache {cache_path}: {exc}")
        elif args.adelie_cache != "auto":
            print(f"[run] adelie cache not found: {cache_path}")

    if args.proposer == "openai":
        proposer = OpenAIProposer(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            api_timeout_s=args.api_timeout_s,
        )
    elif args.proposer == "algopool":
        from .proposers.algopool import AlgoPoolProposer
        proposer = AlgoPoolProposer()
    else:
        proposer = StubProposer([])

    # Strategy layer (default off == greedy-from-champion, today's behavior).
    if args.director == "openai":
        from .director import OpenAIDirector
        director = OpenAIDirector(
            model=args.director_model,
            reasoning_effort=args.director_reasoning_effort,
            api_timeout_s=args.api_timeout_s,
        )
    else:
        from .director import StubDirector
        director = StubDirector()
    if args.analyst == "openai":
        from .analyst import OpenAIAnalyst
        analyst = OpenAIAnalyst(api_timeout_s=args.api_timeout_s)
    else:
        from .analyst import StubAnalyst
        analyst = StubAnalyst()

    rows = run_synth_loop(
        proposer=proposer,
        problem=problem,
        seed_kernel=seed_kernel,
        slot=slot,
        state_root=args.state_root,
        n_proposals=args.n_proposals,
        kkt_tol=args.kkt_tol,
        max_time_s=args.max_time_s,
        reps=args.reps,
        margin=args.margin,
        problem_backend=args.problem_backend,
        problem_dtype=args.problem_dtype,
        dtype_strategy=args.dtype_strategy,
        warmup_runs=args.warmup_runs,
        timeout_s=args.timeout_s,
        program_md=program_md,
        resume_from=args.resume_from,
        compute_baselines=not args.no_baselines,
        baseline_solvers=tuple(s.strip() for s in args.baseline_solvers.split(",") if s.strip()),
        extra_baseline_curves=extra_baseline_curves,
        director=director,
        analyst=analyst,
        director_every_k=args.director_every_k,
        verbose=args.verbose,
    )

    accepted = sum(1 for r in rows if r.decision.get("accepted"))
    discarded = len(rows) - accepted
    print(f"\nsession complete: {accepted} kept / {discarded} discarded across {len(rows)} proposals")
    print(f"lineage: {args.state_root}/lineage.jsonl")
    print(f"checkpoints: {args.state_root}/checkpoints/")
    print(f"research state: {args.state_root}/research_state.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
