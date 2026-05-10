"""Confirm the FISTA Gram champion at tall_medium amortized with multi-rep timing.

P-pivot.0 of the autoresearch pivot. Exists to lock down the regression baseline
before we delete the old synth scaffolding. Read the latest amortized champion
out of the existing state-root, run Tier-2 with reps=5 (no proposals, no
promotion), and report median/min/max/std of the solve time. If the recorded
16.4 ms holds within ~5 percent we keep going; if it doesn't we re-baseline.
"""
from __future__ import annotations

import json
import statistics
import sys
import uuid
from pathlib import Path

import numpy as np

from convexkernels.bench.shapes import DEFAULT_SHAPES, make_synthetic_lasso
from convexkernels.synth.champion_store import ChampionStore
from convexkernels.synth.lineage import Slot
from convexkernels.synth.tiers import EvalConfig, run_tier2


def main() -> int:
    state_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "synth_run_p4_strategy_tall_medium_amortized_20260509"
    )
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    shape = "tall_medium"
    cost_model = "amortized"
    variant = "restart"
    tol = 1e-6
    max_iters = 5000

    slot = Slot("lasso", "fista", "apple_silicon", "fp32")
    store = ChampionStore(state_root)
    source_path, meta = store.current_source_for_workload(slot, cost_model)
    if source_path is None:
        print(f"no {cost_model} champion found under {state_root}", file=sys.stderr)
        return 2

    summary = (meta or {}).get("summary") or {}
    recorded_tier2 = summary.get("tier2_wall_time_ms")
    print(f"champion id   : {(meta or {}).get('id')}")
    print(f"source path   : {source_path}")
    print(f"recorded tier2: {recorded_tier2} ms (single rep, possibly noisy)")
    print(f"now confirming with reps={reps} on {shape}/{cost_model}")

    spec = next(s for s in DEFAULT_SHAPES if s.name == shape)
    problem = make_synthetic_lasso(spec, seed=0)

    run_dir = state_root / "runs" / f"confirm-{uuid.uuid4()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        seed_kernel={
            "module": str(source_path),
            "step": "fista_step",
            "init": "init_state",
        },
        variant=variant,
        problem_backend="mlx",
        problem_dtype="fp32",
        dtype_strategy=summary.get("dtype_strategy") or "fp32",
        cost_model=cost_model,
        warmup_runs=1,
        timeout_s=240.0,
    )

    tier, _result = run_tier2(
        run_dir=run_dir,
        problem=problem,
        kernel_module=str(source_path),
        config=config,
        max_iters=max_iters,
        tol=tol,
        reps=reps,
    )

    print()
    print(f"passed         : {tier.passed}")
    print(f"converged      : {tier.converged}")
    print(f"iters          : {tier.n_iters}")
    print(f"kkt_final      : {tier.kkt_final}")
    print(f"solve median ms: {tier.solve_time_ms}")
    print(f"solve min ms   : {tier.wall_time_min_ms}")
    print(f"solve max ms   : {tier.wall_time_max_ms}")
    print(f"solve std ms   : {tier.wall_time_std_ms}")
    print(f"n_reps         : {tier.n_reps}")

    if recorded_tier2 is not None and tier.solve_time_ms is not None:
        ratio = tier.solve_time_ms / recorded_tier2
        verdict = "HOLDS" if 0.85 <= ratio <= 1.15 else "DRIFTED"
        print(f"\nratio vs recorded: {ratio:.3f} — {verdict}")

    out_path = state_root / "confirm_gram_champion.json"
    out_path.write_text(json.dumps({
        "champion_id": (meta or {}).get("id"),
        "recorded_tier2_ms": recorded_tier2,
        "confirm_solve_median_ms": tier.solve_time_ms,
        "confirm_solve_min_ms": tier.wall_time_min_ms,
        "confirm_solve_max_ms": tier.wall_time_max_ms,
        "confirm_solve_std_ms": tier.wall_time_std_ms,
        "n_reps": tier.n_reps,
        "kkt_final": tier.kkt_final,
        "passed": tier.passed,
        "converged": tier.converged,
        "iters": tier.n_iters,
        "shape": shape,
        "cost_model": cost_model,
        "variant": variant,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0 if tier.passed else 1


if __name__ == "__main__":
    sys.exit(main())
