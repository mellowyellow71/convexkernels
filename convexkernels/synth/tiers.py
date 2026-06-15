"""Multi-tier evaluator for synthesized kernels.

Tier 1 is the cheap ratchet gate. Tier 2 verifies convergence at a tighter
tolerance on a selected problem. Tier 3 runs the same kernel over a shape suite
and records median per-shape timing. All tiers execute through the existing
subprocess sandbox.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..bench.shapes import ShapeSpec, make_synthetic_lasso
from .lineage import Tier1Result, Tier2Result, Tier3PerShape, Tier3Result
from .roofline import DEFAULT_APPLE_SILICON_PEAK_GB_S, estimate_fista_roofline
from .sandbox import SandboxResult, run_kernel, write_eval_config


@dataclass(frozen=True)
class EvalConfig:
    seed_kernel: dict
    variant: str = "basic"
    problem_backend: str = "native"
    problem_dtype: str = "fp32"
    dtype_strategy: str = "fp32"
    gradient_strategy: str = "direct"
    gram_symmetric: bool = False
    cost_model: str = "single"
    warmup_runs: int = 1
    timeout_s: float = 60.0
    peak_bandwidth_gb_s: float = DEFAULT_APPLE_SILICON_PEAK_GB_S


def run_tier1(
    *,
    run_dir: Path,
    problem: Any,
    kernel_module: str,
    config: EvalConfig,
    max_iters: int,
    tol: float,
    reps: int = 1,
) -> tuple[Tier1Result, SandboxResult]:
    results = _run_eval_reps(
        run_dir=run_dir,
        tier_name="tier1",
        reps=reps,
        problem=problem,
        kernel_module=kernel_module,
        config=config,
        max_iters=max_iters,
        tol=tol,
    )
    result = _aggregate_sandbox_results(results)
    passed = all(_passes_tier1(r, tol) for r in results)
    stats = _wall_time_stats(results, cost_model=config.cost_model)
    if passed:
        tier = Tier1Result(
            passed=True,
            wall_time_ms=stats["median_ms"],
            setup_time_ms=stats["setup_median_ms"],
            solve_time_ms=stats["solve_median_ms"],
            single_solve_wall_time_ms=stats["single_median_ms"],
            amortized_wall_time_ms=stats["amortized_median_ms"],
            cost_model=config.cost_model,
            n_reps=len(results),
            wall_time_min_ms=stats["min_ms"],
            wall_time_max_ms=stats["max_ms"],
            wall_time_std_ms=stats["std_ms"],
        )
    else:
        reject_reason = _first_reject_reason(results, tol=tol, require_converged=False)
        tier = Tier1Result(
            passed=False,
            reject_reason=reject_reason,
            wall_time_ms=stats["median_ms"],
            setup_time_ms=stats["setup_median_ms"],
            solve_time_ms=stats["solve_median_ms"],
            single_solve_wall_time_ms=stats["single_median_ms"],
            amortized_wall_time_ms=stats["amortized_median_ms"],
            cost_model=config.cost_model,
            n_reps=len(results),
            wall_time_min_ms=stats["min_ms"],
            wall_time_max_ms=stats["max_ms"],
            wall_time_std_ms=stats["std_ms"],
        )
    _write_tier_json(run_dir / "tier1.json", tier)
    return tier, result


def run_tier2(
    *,
    run_dir: Path,
    problem: Any,
    kernel_module: str,
    config: EvalConfig,
    max_iters: int,
    tol: float,
    reps: int = 1,
) -> tuple[Tier2Result, SandboxResult]:
    results = _run_eval_reps(
        run_dir=run_dir / "tier2_eval",
        tier_name="tier2",
        reps=reps,
        problem=problem,
        kernel_module=kernel_module,
        config=config,
        max_iters=max_iters,
        tol=tol,
    )
    result = _aggregate_sandbox_results(results)
    passed = all(_passes_tier2(r, tol) for r in results)
    stats = _wall_time_stats(results, cost_model=config.cost_model)
    tier = Tier2Result(
        passed=passed,
        converged=all(bool(r.converged) for r in results),
        n_iters=int(statistics.median([int(r.iters or 0) for r in results])),
        wall_time_ms=stats["median_ms"],
        kkt_final=_max_kkt(results),
        setup_time_ms=stats["setup_median_ms"],
        solve_time_ms=stats["solve_median_ms"],
        single_solve_wall_time_ms=stats["single_median_ms"],
        amortized_wall_time_ms=stats["amortized_median_ms"],
        cost_model=config.cost_model,
        n_reps=len(results),
        wall_time_min_ms=stats["min_ms"],
        wall_time_max_ms=stats["max_ms"],
        wall_time_std_ms=stats["std_ms"],
    )
    _write_tier_json(run_dir / "tier2.json", tier)
    return tier, result


def run_tier3(
    *,
    run_dir: Path,
    shapes: Iterable[ShapeSpec],
    kernel_module: str,
    config: EvalConfig,
    max_iters: int,
    tol: float,
    reps: int = 3,
    seed: int = 0,
) -> Tier3Result:
    per_shape: list[Tier3PerShape] = []
    all_passed = True

    for spec in shapes:
        times: list[float] = []
        setup_times: list[float] = []
        solve_times: list[float] = []
        single_times: list[float] = []
        iters: list[int] = []
        kkts: list[float] = []
        shape_passed = True
        for rep in range(reps):
            problem = make_synthetic_lasso(spec, seed=seed)
            eval_dir = run_dir / "tier3_eval" / spec.name / f"rep_{rep}"
            result = _run_eval(
                run_dir=eval_dir,
                problem=problem,
                kernel_module=kernel_module,
                config=config,
                max_iters=max_iters,
                tol=tol,
            )
            if not (
                result.status == "completed"
                and bool(result.converged)
                and result.kkt_final is not None
                and result.kkt_final < tol
            ):
                shape_passed = False
                all_passed = False
                continue
            times.append(_selected_time_s(result, config.cost_model) * 1000)
            setup_times.append((result.setup_time_s or 0.0) * 1000)
            solve_times.append((result.solve_time_s or result.wall_time_s or 0.0) * 1000)
            single_times.append(
                (result.single_solve_time_s or result.wall_time_s or 0.0) * 1000
            )
            iters.append(int(result.iters or 0))
            kkts.append(float(result.kkt_final))

        if times:
            wall_time_med_ms = float(statistics.median(times))
            n_iters_med = int(statistics.median(iters))
            roofline = estimate_fista_roofline(
                m=spec.m,
                n=spec.n,
                dtype_name=config.problem_dtype,
                wall_time_ms=wall_time_med_ms,
                n_iters=n_iters_med,
                gradient_strategy=config.gradient_strategy,
                symmetric=config.gram_symmetric,
                peak_bandwidth_gb_s=config.peak_bandwidth_gb_s,
            )
            per_shape.append(
                Tier3PerShape(
                    m=spec.m,
                    n=spec.n,
                    n_iters_med=n_iters_med,
                    wall_time_med_ms=wall_time_med_ms,
                    kkt_final_med=float(statistics.median(kkts)),
                    roofline_pct_med=roofline.roofline_pct,
                    peak_mem_mb=roofline.peak_mem_mb,
                    setup_time_med_ms=float(statistics.median(setup_times)),
                    solve_time_med_ms=float(statistics.median(solve_times)),
                    single_solve_wall_time_med_ms=float(statistics.median(single_times)),
                    amortized_wall_time_med_ms=float(statistics.median(solve_times)),
                    cost_model=config.cost_model,
                    bytes_per_iter=roofline.bytes_per_iter,
                    flops_per_iter=roofline.flops_per_iter,
                    arithmetic_intensity=roofline.arithmetic_intensity,
                    roofline_floor_ms_per_iter=roofline.roofline_floor_ms_per_iter,
                    measured_ms_per_iter=roofline.measured_ms_per_iter,
                    achieved_bandwidth_gb_s=roofline.achieved_bandwidth_gb_s,
                )
            )
        else:
            roofline = estimate_fista_roofline(
                m=spec.m,
                n=spec.n,
                dtype_name=config.problem_dtype,
                wall_time_ms=float("inf"),
                n_iters=0,
                gradient_strategy=config.gradient_strategy,
                symmetric=config.gram_symmetric,
                peak_bandwidth_gb_s=config.peak_bandwidth_gb_s,
            )
            per_shape.append(
                Tier3PerShape(
                    m=spec.m,
                    n=spec.n,
                    n_iters_med=0,
                    wall_time_med_ms=float("inf"),
                    kkt_final_med=float("inf"),
                    roofline_pct_med=0.0,
                    peak_mem_mb=roofline.peak_mem_mb,
                    cost_model=config.cost_model,
                    bytes_per_iter=roofline.bytes_per_iter,
                    flops_per_iter=roofline.flops_per_iter,
                    arithmetic_intensity=roofline.arithmetic_intensity,
                    roofline_floor_ms_per_iter=roofline.roofline_floor_ms_per_iter,
                    measured_ms_per_iter=0.0,
                    achieved_bandwidth_gb_s=0.0,
                )
            )
        if not shape_passed:
            all_passed = False

    wall_times = [p.wall_time_med_ms for p in per_shape]
    n_iters = [p.n_iters_med for p in per_shape]
    roofline_pcts = [p.roofline_pct_med for p in per_shape if p.roofline_pct_med > 0]
    tier = Tier3Result(
        passed=all_passed,
        per_shape=per_shape,
        rank_summary={
            "median_wall_time_ms": float(statistics.median(wall_times)) if wall_times else 0.0,
            "median_n_iters": int(statistics.median(n_iters)) if n_iters else 0,
            "median_roofline_pct": (
                float(statistics.median(roofline_pcts)) if roofline_pcts else 0.0
            ),
            "peak_bandwidth_gb_s": config.peak_bandwidth_gb_s,
            "n_shapes": len(per_shape),
            "reps": reps,
        },
    )
    _write_tier_json(run_dir / "tier3.json", tier)
    return tier


def _run_eval(
    *,
    run_dir: Path,
    problem: Any,
    kernel_module: str,
    config: EvalConfig,
    max_iters: int,
    tol: float,
) -> SandboxResult:
    write_eval_config(
        run_dir,
        problem,
        kernel_module=kernel_module,
        kernel_step=config.seed_kernel["step"],
        kernel_init=config.seed_kernel["init"],
        variant=config.variant,
        problem_backend=config.problem_backend,
        problem_dtype=config.problem_dtype,
        dtype_strategy=config.dtype_strategy,
        warmup_runs=config.warmup_runs,
        max_iters=max_iters,
        tol=tol,
    )
    return run_kernel(run_dir, timeout_s=config.timeout_s)


def _run_eval_reps(
    *,
    run_dir: Path,
    tier_name: str,
    reps: int,
    problem: Any,
    kernel_module: str,
    config: EvalConfig,
    max_iters: int,
    tol: float,
) -> list[SandboxResult]:
    reps = max(1, int(reps))
    results: list[SandboxResult] = []
    for rep in range(reps):
        eval_dir = run_dir if reps == 1 else run_dir / f"{tier_name}_reps" / f"rep_{rep}"
        results.append(
            _run_eval(
                run_dir=eval_dir,
                problem=problem,
                kernel_module=kernel_module,
                config=config,
                max_iters=max_iters,
                tol=tol,
            )
        )
    return results


def _passes_tier1(result: SandboxResult, tol: float) -> bool:
    return (
        result.status == "completed"
        and result.kkt_final is not None
        and result.kkt_final < tol
    )


def _passes_tier2(result: SandboxResult, tol: float) -> bool:
    return (
        result.status == "completed"
        and bool(result.converged)
        and result.kkt_final is not None
        and result.kkt_final < tol
    )


def _first_reject_reason(
    results: list[SandboxResult],
    *,
    tol: float,
    require_converged: bool,
) -> str:
    for result in results:
        if result.status != "completed":
            return result.status
        if require_converged and not bool(result.converged):
            return "not_converged"
        if result.kkt_final is None or result.kkt_final >= tol:
            return "kkt_above_tier1_tol"
    return "unknown"


def _selected_time_s(result: SandboxResult, cost_model: str) -> float:
    if cost_model == "amortized" and result.solve_time_s is not None:
        return float(result.solve_time_s)
    return float(result.wall_time_s or 0.0)


def _wall_time_stats(
    results: list[SandboxResult],
    *,
    cost_model: str,
) -> dict[str, float]:
    times = [
        _selected_time_s(result, cost_model) * 1000
        for result in results
        if result.wall_time_s is not None or result.solve_time_s is not None
    ]
    setup_times = [
        (result.setup_time_s or 0.0) * 1000
        for result in results
        if result.setup_time_s is not None
    ]
    solve_times = [
        (result.solve_time_s or result.wall_time_s or 0.0) * 1000
        for result in results
        if result.solve_time_s is not None or result.wall_time_s is not None
    ]
    single_times = [
        (result.single_solve_time_s or result.wall_time_s or 0.0) * 1000
        for result in results
        if result.single_solve_time_s is not None or result.wall_time_s is not None
    ]
    if not times:
        return {
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
            "setup_median_ms": 0.0,
            "solve_median_ms": 0.0,
            "single_median_ms": 0.0,
            "amortized_median_ms": 0.0,
        }
    return {
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "setup_median_ms": (
            float(statistics.median(setup_times)) if setup_times else 0.0
        ),
        "solve_median_ms": (
            float(statistics.median(solve_times)) if solve_times else 0.0
        ),
        "single_median_ms": (
            float(statistics.median(single_times)) if single_times else 0.0
        ),
        "amortized_median_ms": (
            float(statistics.median(solve_times)) if solve_times else 0.0
        ),
    }


def _aggregate_sandbox_results(results: list[SandboxResult]) -> SandboxResult:
    if not results:
        return SandboxResult(status="no_result")

    stats = _wall_time_stats(results, cost_model="single")
    first_failed = next((r for r in results if r.status != "completed"), None)
    status = "completed" if first_failed is None else first_failed.status
    first_error = first_failed or next(
        (r for r in results if r.error_type or r.error_message),
        None,
    )
    return SandboxResult(
        status=status,
        iters=int(statistics.median([int(r.iters or 0) for r in results])),
        kkt_final=_max_kkt(results),
        wall_time_s=stats["median_ms"] / 1000,
        setup_time_s=stats["setup_median_ms"] / 1000,
        solve_time_s=stats["solve_median_ms"] / 1000,
        single_solve_time_s=stats["single_median_ms"] / 1000,
        converged=all(bool(r.converged) for r in results),
        error_type=first_error.error_type if first_error else None,
        error_message=first_error.error_message if first_error else None,
        stderr_tail=first_error.stderr_tail if first_error else "",
    )


def _max_kkt(results: list[SandboxResult]) -> float:
    kkts = [float(r.kkt_final) for r in results if r.kkt_final is not None]
    return max(kkts) if kkts else float("inf")


def _write_tier_json(path: Path, tier: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(tier), indent=2))
