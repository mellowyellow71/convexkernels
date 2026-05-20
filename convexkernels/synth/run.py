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


def _default_seed_kernel(problem_family: str, algorithm: str, backend: str) -> dict:
    """Resolve a seed kernel module/step/init by (problem_family, algorithm, backend)."""
    if backend == "mlx":
        if problem_family == "lasso" and algorithm == "fista":
            return {"module": "convexkernels.kernels.mlx.seeds.fista_step_v0",
                    "step": "fista_step", "init": "init_state"}
        if problem_family == "nonneg_lasso" and algorithm == "fista":
            return {"module": "convexkernels.kernels.mlx.seeds.nonnegative_fista_step_v0",
                    "step": "fista_step", "init": "init_state"}
        if problem_family == "total_variation_1d" and algorithm == "pdhg":
            return {"module": "convexkernels.kernels.mlx.seeds.pdhg_step_v0",
                    "step": "pdhg_step", "init": "init_state"}
        if problem_family == "basis_pursuit" and algorithm == "pdhg":
            return {"module": "convexkernels.kernels.mlx.seeds.pdhg_bp_step_v0",
                    "step": "pdhg_step", "init": "init_state"}
        if problem_family == "equality_qp" and algorithm == "alm":
            return {"module": "convexkernels.kernels.mlx.seeds.alm_step_v0",
                    "step": "alm_step", "init": "init_state"}
        if problem_family == "lasso_admm" and algorithm == "admm":
            return {"module": "convexkernels.kernels.mlx.seeds.admm_lasso_step_v0",
                    "step": "admm_step", "init": "init_state"}
        if problem_family == "lasso_path" and algorithm == "fista_gram":
            return {"module": "convexkernels.kernels.mlx.seeds.gram_fista_path_v0",
                    "step": "fista_path_step", "init": "init_state"}
    elif backend == "native":
        if algorithm == "fista":
            return {"module": "convexkernels.kernels.numpy_ref",
                    "step": "fista_step", "init": "init_state"}
        if algorithm == "pdhg":
            return {"module": "convexkernels.kernels.numpy_pdhg_ref",
                    "step": "pdhg_step", "init": "init_state"}
        if algorithm == "alm":
            return {"module": "convexkernels.kernels.numpy_alm_ref",
                    "step": "alm_step", "init": "init_state"}
        if algorithm == "admm":
            return {"module": "convexkernels.kernels.numpy_admm_ref",
                    "step": "admm_step", "init": "init_state"}
    raise ValueError(f"no default seed for ({problem_family}, {algorithm}, backend={backend})")


def _parse_slot_spec(slot_str: str) -> Slot:
    parts = slot_str.split("/")
    if len(parts) != 4:
        raise ValueError(f"slot must be problem_family/algorithm/hardware/dtype, got {slot_str!r}")
    pf, algo, hw, dt = parts
    if hw == "auto":
        hw = detect_hardware()
    return Slot(problem_family=pf, algorithm=algo, hardware=hw, dtype=dt)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, help="problem_family/algorithm/hardware/dtype")
    p.add_argument("--shape", required=True, help="bench shape name (e.g. tall_medium, tv1d_medium)")
    p.add_argument("--seed", default=None, help="seed kernel module dotted-path (auto if omitted)")
    p.add_argument("--seed-step", default=None, help="seed kernel step function name")
    p.add_argument("--seed-init", default=None, help="seed kernel init function name")
    p.add_argument("--default-step-name", default=None, help=argparse.SUPPRESS)
    p.add_argument("--proposer", choices=["openai", "stub"], default="openai")
    p.add_argument("--model", default="gpt-5.5", help="OpenAI model name")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--api-timeout-s", type=float, default=240.0)
    p.add_argument("--n-proposals", type=int, default=50)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--fitness-tol", type=float, default=1e-6)
    p.add_argument("--speedup-margin", type=float, default=0.97)
    p.add_argument("--variant", default="basic")
    p.add_argument("--problem-backend", choices=["native", "mlx"], default="mlx")
    p.add_argument("--problem-dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--cost-model", choices=["single", "amortized"], default="single")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--state-root", required=True, type=Path)
    p.add_argument("--program-md", default="convexkernels/synth/program.md", type=Path)
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args(argv)

    slot = _parse_slot_spec(args.slot)
    problem = _make_problem(slot.problem_family, args.shape)
    if args.seed is None:
        seed_kernel = _default_seed_kernel(
            slot.problem_family, slot.algorithm, args.problem_backend,
        )
    else:
        default_step = {"pdhg": "pdhg_step", "alm": "alm_step",
                        "admm": "admm_step", "fista_gram": "fista_path_step",
                        "fista_path": "fista_path_step"}.get(slot.algorithm, "fista_step")
        seed_kernel = {
            "module": args.seed,
            "step": args.seed_step or default_step,
            "init": args.seed_init or "init_state",
        }
    program_md = ""
    if args.program_md and args.program_md.exists():
        program_md = args.program_md.read_text()

    if args.proposer == "openai":
        proposer = OpenAIProposer(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            api_timeout_s=args.api_timeout_s,
        )
    else:
        proposer = StubProposer([])

    rows = run_synth_loop(
        proposer=proposer,
        problem=problem,
        seed_kernel=seed_kernel,
        slot=slot,
        state_root=args.state_root,
        n_proposals=args.n_proposals,
        algorithm=slot.algorithm,
        variant=args.variant,
        fitness_tol=args.fitness_tol,
        max_iters=args.max_iters,
        reps=args.reps,
        speedup_margin=args.speedup_margin,
        problem_backend=args.problem_backend,
        problem_dtype=args.problem_dtype,
        cost_model=args.cost_model,
        warmup_runs=args.warmup_runs,
        timeout_s=args.timeout_s,
        program_md=program_md,
        verbose=args.verbose,
    )

    accepted = sum(1 for r in rows if r.decision.get("accepted"))
    discarded = len(rows) - accepted
    print(f"\nsession complete: {accepted} kept / {discarded} discarded across {len(rows)} proposals")
    print(f"lineage: {args.state_root}/lineage.jsonl")
    print(f"champion source: {args.state_root}/champion.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
