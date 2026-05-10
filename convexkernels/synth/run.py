"""Standalone synth-loop driver.

Usage:
    python -m convexkernels.synth.run --n-proposals 3 --shape tall_small

Reads OPENAI_API_KEY from env. Writes lineage + run artifacts under
`./synth_run/` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bench.shapes import (
    DEFAULT_SHAPES,
    make_synthetic_lasso,
    make_synthetic_nonnegative_lasso,
)
from .lineage import Slot
from .loop import detect_hardware


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proposals", type=int, default=3)
    parser.add_argument("--shape", type=str, default="tall_small",
                        choices=[s.name for s in DEFAULT_SHAPES])
    parser.add_argument("--problem-family", type=str, default="lasso",
                        choices=["lasso", "nonnegative_lasso"],
                        help="Convex problem family to synthesize for.")
    parser.add_argument("--state-root", type=str, default="./synth_run")
    parser.add_argument("--tier1-tol", type=float, default=1e-3)
    parser.add_argument("--tier1-max-iters", type=int, default=200)
    parser.add_argument("--tier1-reps", type=int, default=3,
                        help="Timed Tier-1 repetitions; median time is gated.")
    parser.add_argument("--tier1-escalation-margin", type=float, default=1.0,
                        help="For Tier-2/3 promotion, run Tier-2 when Tier-1 is below this fraction of champion Tier-1 time.")
    parser.add_argument("--variant", type=str, default="restart",
                        choices=["basic", "restart"],
                        help="Host-side FISTA variant used during evaluation.")
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Optional sampling temperature. Omitted by default.")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for OpenAI reasoning models.")
    parser.add_argument("--api-timeout-s", type=float, default=180.0,
                        help="Per-proposal OpenAI request timeout.")
    parser.add_argument("--seed-kernel", type=str,
                        default=None,
                        help="Dotted module path of the seed kernel. Defaults by problem family.")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--proposer", type=str, default="openai",
                        choices=["openai", "stub", "structured"])
    parser.add_argument("--transfer-seed-k", type=int, default=0,
                        help="Evaluate up to k accepted edits transferred from neighbor slots before asking the proposer.")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Problem dtype for MLX-backed kernel evaluation.")
    parser.add_argument("--dtype-strategy", type=str, default="fp32",
                        choices=["fp32", "fp16_storage", "mixed_gram"],
                        help="Dtype strategy label for dtype-search experiments.")
    parser.add_argument("--gradient-strategy", type=str, default="direct",
                        choices=["direct", "gram"],
                        help="FISTA gradient path to use when choosing the default seed.")
    parser.add_argument("--cost-model", type=str, default="single",
                        choices=["single", "amortized", "both"],
                        help="Timing metric used for speed gates; 'both' logs both and gates on single-solve time.")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Untimed warmup evaluations before the timed Tier-1 run.")
    parser.add_argument("--speedup-margin", type=float, default=0.95,
                        help="Keep only proposals below this fraction of champion wall time.")
    parser.add_argument("--no-speed-gate", action="store_true",
                        help="Debug mode: accept KKT-valid proposals even if slower.")
    parser.add_argument("--promotion-tier", type=str, default="tier1",
                        choices=["tier1", "tier2", "tier3"],
                        help="Highest tier required before champion promotion.")
    parser.add_argument("--tier2-shape", type=str, default="tall_medium",
                        choices=[s.name for s in DEFAULT_SHAPES])
    parser.add_argument("--tier2-tol", type=float, default=1e-6)
    parser.add_argument("--tier2-max-iters", type=int, default=5000)
    parser.add_argument("--tier2-reps", type=int, default=3,
                        help="Timed Tier-2 repetitions; median time is gated.")
    parser.add_argument("--tier2-speed-margin", type=float, default=0.97,
                        help="For Tier-2/3 promotion, require convergence time below this fraction of champion Tier-2 time.")
    parser.add_argument("--no-tier2-speed-gate", action="store_true",
                        help="Debug mode: Tier-2 checks convergence only, not speed.")
    parser.add_argument("--no-tier2-confirm-speed", action="store_true",
                        help="Debug mode: skip paired champion remeasurement before Tier-2 speed promotion.")
    parser.add_argument("--tier3-shapes", type=str, default="tall_small,wide_small",
                        help="Comma-separated shape names for Tier-3.")
    parser.add_argument("--tier3-tol", type=float, default=1e-6)
    parser.add_argument("--tier3-max-iters", type=int, default=5000)
    parser.add_argument("--tier3-reps", type=int, default=3)
    args = parser.parse_args()

    spec = next(s for s in DEFAULT_SHAPES if s.name == args.shape)
    shape_by_name = {s.name: s for s in DEFAULT_SHAPES}
    problem_factory = (
        make_synthetic_nonnegative_lasso
        if args.problem_family == "nonnegative_lasso"
        else make_synthetic_lasso
    )
    problem = problem_factory(spec, seed=0)
    tier2_problem = problem_factory(shape_by_name[args.tier2_shape], seed=0)
    tier3_shapes = tuple(
        shape_by_name[name.strip()]
        for name in args.tier3_shapes.split(",")
        if name.strip()
    )
    problem_backend = "native"
    slot_dtype = "fp64"
    problem_dtype = args.dtype
    if args.dtype_strategy == "fp16_storage":
        problem_dtype = "fp16"
    elif args.dtype_strategy == "mixed_gram":
        problem_dtype = "fp32"

    seed_kernel_module = args.seed_kernel
    if seed_kernel_module is None:
        if args.problem_family == "nonnegative_lasso":
            seed_kernel_module = "convexkernels.kernels.mlx.seeds.nonnegative_fista_step_v0"
        elif args.gradient_strategy == "gram":
            seed_kernel_module = "convexkernels.kernels.mlx.seeds.gram_fista_step_v0"
        else:
            seed_kernel_module = "convexkernels.kernels.mlx.seeds.fista_step_v0"

    if ".mlx." in seed_kernel_module:
        problem_backend = "mlx"
        slot_dtype = args.dtype_strategy

    if args.proposer == "openai":
        from .proposers.openai import OpenAIProposer
        proposer = OpenAIProposer(
            model=args.model,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            api_timeout_s=args.api_timeout_s,
        )
    elif args.proposer == "structured":
        from .proposers.structured import StructuredGridProposer
        proposer = StructuredGridProposer()
    else:
        from .proposers.stub import DeterministicStubProposer
        proposer = DeterministicStubProposer()

    from .loop import run_synth_loop
    appended = run_synth_loop(
        proposer=proposer,
        problem=problem,
        seed_kernel={
            "module": seed_kernel_module,
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=args.n_proposals,
        state_root=Path(args.state_root),
        shape_name=args.shape,
        slot=Slot(
            problem_family=args.problem_family,
            algorithm="fista",
            hardware=detect_hardware(),
            dtype=slot_dtype,
        ) if ".mlx." in seed_kernel_module else None,
        problem_backend=problem_backend,
        problem_dtype=problem_dtype,
        dtype_strategy=args.dtype_strategy,
        cost_model=args.cost_model,
        algorithm_variant=args.variant,
        warmup_runs=args.warmup_runs,
        require_speedup=not args.no_speed_gate,
        speedup_margin=args.speedup_margin,
        promotion_tier=args.promotion_tier,
        tier2_problem=tier2_problem,
        tier2_kkt_tol=args.tier2_tol,
        tier2_max_iters=args.tier2_max_iters,
        tier2_reps=args.tier2_reps,
        require_tier2_speed=not args.no_tier2_speed_gate,
        tier2_speed_margin=args.tier2_speed_margin,
        confirm_tier2_speed=not args.no_tier2_confirm_speed,
        tier3_shapes=tier3_shapes,
        tier3_kkt_tol=args.tier3_tol,
        tier3_max_iters=args.tier3_max_iters,
        tier3_reps=args.tier3_reps,
        transfer_seed_k=args.transfer_seed_k,
        tier1_kkt_tol=args.tier1_tol,
        tier1_max_iters=args.tier1_max_iters,
        tier1_reps=args.tier1_reps,
        tier1_escalation_margin=args.tier1_escalation_margin,
        timeout_s=args.timeout_s,
        verbose=True,
    )

    accepted = [r for r in appended if r["decision"]["accepted"]]
    print()
    print("=== summary ===")
    print(f"problem: {args.problem_family}")
    print(f"shape: {spec.name} (m={spec.m}, n={spec.n})")
    print(f"proposer: {args.proposer} (model: {args.model})")
    print(f"proposals: {len(appended)}")
    print(f"accepted: {len(accepted)}")
    if accepted:
        for r in accepted:
            print(f"  - {r['edit']['type']}: {r['tier1']['wall_time_ms']:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
