"""Baseline timing for the PDHG-TV-1D seed kernel.

Establishes the regression floor for the new specimen-2 slot before any
autoresearch is run on it. Mirrors `confirm_gram_champion.py` in spirit:
no proposals, just multi-rep median/min/max/std of the seed solve time at
gap < 1e-6, plus the primal objective for cross-validation against CVXPY.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from convexkernels.bench.shapes import (
    DEFAULT_TV1D_SHAPES,
    TvShapeSpec,
    make_synthetic_tv_1d,
)


def _try_cvxpy_solve(b: np.ndarray, lam: float, *, max_n: int = 1024):
    """Solve the small TV-1D problem with CVXPY for cross-validation. Returns
    (primal_objective, x*) or None for problems too large for the solver."""
    if b.shape[0] > max_n:
        return None
    try:
        import cvxpy as cp
    except ImportError:
        return None
    n = b.shape[0]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(x - b) + lam * cp.tv(x)
    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    except Exception:
        return None
    return float(prob.value), np.asarray(x.value)


def main() -> int:
    shape_name = sys.argv[1] if len(sys.argv) > 1 else "tv1d_medium"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    backend = sys.argv[3] if len(sys.argv) > 3 else "mlx"
    tol = 1e-6
    max_iters = 5000
    warmup = 1

    spec = next(s for s in DEFAULT_TV1D_SHAPES if s.name == shape_name)
    print(f"shape: {spec}")
    print(f"backend: {backend}, reps: {reps}, tol: {tol}, warmup: {warmup}")

    prob_np = make_synthetic_tv_1d(spec, seed=0)

    # CVXPY oracle (small problems only)
    oracle = _try_cvxpy_solve(prob_np.b, prob_np.lam)
    if oracle is not None:
        p_oracle, x_oracle = oracle
        print(f"CVXPY primal: {p_oracle:.6f}")
    else:
        print("CVXPY oracle skipped (problem too large or solver unavailable)")
        p_oracle = None

    # Build the per-iter problem and load the seed kernel
    if backend == "mlx":
        import mlx.core as mx
        from convexkernels.kernels.mlx.lib import TVDenoising1DMLX
        from convexkernels.kernels.mlx.seeds.pdhg_step_v0 import (
            init_state as kernel_init,
            pdhg_step as kernel_step,
        )
        problem = TVDenoising1DMLX.from_problem(prob_np, dtype=mx.float32)
    elif backend == "numpy":
        from convexkernels.kernels.numpy_pdhg_ref import (
            init_state as kernel_init,
            pdhg_step as kernel_step,
        )
        problem = prob_np
    else:
        raise ValueError(f"unknown backend {backend}")

    from convexkernels.algorithms.pdhg import pdhg

    # Warmup (Metal compile, allocator paths, etc.)
    for _ in range(warmup):
        pdhg(problem, max_iters=max_iters, tol=tol, kernel_step=kernel_step,
             kernel_init=kernel_init, record_history=False)

    solve_times_ms = []
    iters_list = []
    gap_finals = []
    for rep in range(reps):
        t0 = time.perf_counter()
        res = pdhg(problem, max_iters=max_iters, tol=tol,
                   kernel_step=kernel_step, kernel_init=kernel_init,
                   record_history=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not res.converged:
            print(f"rep {rep}: did not converge, gap={res.gap_final:.3e}", file=sys.stderr)
            return 1
        solve_times_ms.append(elapsed_ms)
        iters_list.append(int(res.n_iters))
        gap_finals.append(float(res.gap_final))

    median_ms = statistics.median(solve_times_ms)
    min_ms = min(solve_times_ms)
    max_ms = max(solve_times_ms)
    std_ms = statistics.stdev(solve_times_ms) if len(solve_times_ms) > 1 else 0.0

    print()
    print(f"converged       : True (all {reps} reps)")
    print(f"iters median    : {statistics.median(iters_list)}")
    print(f"gap median      : {statistics.median(gap_finals):.3e}")
    print(f"solve median ms : {median_ms:.3f}")
    print(f"solve min ms    : {min_ms:.3f}")
    print(f"solve max ms    : {max_ms:.3f}")
    print(f"solve std ms    : {std_ms:.3f}")
    print(f"solve CV %      : {(std_ms / median_ms * 100):.1f}")

    if p_oracle is not None:
        # Run one more measured solve to capture a final x for primal comparison
        res = pdhg(problem, max_iters=max_iters, tol=tol,
                   kernel_step=kernel_step, kernel_init=kernel_init,
                   record_history=False)
        x_final = np.asarray(res.x)
        p_seed = prob_np.primal_objective(x_final)
        print(f"seed primal     : {p_seed:.6f}")
        print(f"primal vs CVXPY : rel err {abs(p_seed - p_oracle) / max(abs(p_oracle), 1.0):.3e}")

    out = {
        "shape": shape_name,
        "backend": backend,
        "tol": tol,
        "reps": reps,
        "iters_median": float(statistics.median(iters_list)),
        "gap_median": float(statistics.median(gap_finals)),
        "solve_median_ms": median_ms,
        "solve_min_ms": min_ms,
        "solve_max_ms": max_ms,
        "solve_std_ms": std_ms,
        "primal_cvxpy": p_oracle,
    }
    out_path = Path("baseline_pdhg_tv.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
