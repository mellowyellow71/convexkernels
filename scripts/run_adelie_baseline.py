#!/usr/bin/env python3
"""Pre-compute Adelie's full-path solution for each path shape; cache to disk.

These are the headline numbers the pivot is trying to beat. The cached
(X_adelie, lambdas, wall_ms) tuples become the ground-truth references that
all later solvers (numpy, MLX seed, autoresearch champions) are compared
against.

Run on the M3 Pro (Adelie is CPU-only):

    .venv/bin/python scripts/run_adelie_baseline.py \
        --shapes path_wide_hero,path_tall_medium,path_square,path_wide_small \
        --reps 5 --warmup 2 \
        --out convexkernels/bench/cache/

Output: one `adelie_path_{shape}.npz` per shape, containing:
    X       : (n, K) float64
    lambdas : (K,) float64 (decreasing)
    wall_ms : float (median across reps)
    times_ms: (reps,) float
    kkt     : (K,) float (per-lambda residual under our scale-free formulation)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="path_wide_hero")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--max-iters", type=int, default=int(1e6))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="convexkernels/bench/cache")
    args = ap.parse_args()

    # Defer imports so --help works even without adelie installed.
    from convexkernels.bench.lasso_path_baselines import run_adelie_path
    from convexkernels.bench.path_shapes import get_path_shape, make_path_problem
    from convexkernels.frontend.lasso_path import LassoPath

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    print(f"running adelie baseline on shapes: {shape_names}")
    print(f"reps={args.reps} warmup={args.warmup} tol={args.tol}")
    print(f"out_dir={out_dir.resolve()}")
    print()

    for name in shape_names:
        spec = get_path_shape(name)
        print(f"[{name}] m={spec.m} n={spec.n} K={spec.K} "
              f"sparsity={spec.sparsity} seed={args.seed}")
        t_gen = time.perf_counter()
        A, b, lambdas = make_path_problem(spec, seed=args.seed)
        prob = LassoPath(A, b, lambdas)
        print(f"[{name}]   generated in {time.perf_counter() - t_gen:.2f}s; "
              f"lambda_max={prob.lambda_max:.3e}")
        t_run = time.perf_counter()
        res = run_adelie_path(
            prob, reps=args.reps, warmup=args.warmup,
            tol=args.tol, max_iters=args.max_iters,
        )
        print(f"[{name}]   adelie path: wall_median={res.wall_time_s*1000:.1f} ms "
              f"(times={[f'{t*1000:.1f}' for t in res.wall_times_s]}); "
              f"max KKT={res.kkt_per_lambda.max():.2e}")
        print(f"[{name}]   total Adelie+rep harness: "
              f"{time.perf_counter() - t_run:.1f}s")

        out_path = out_dir / f"adelie_path_{name}.npz"
        np.savez(
            out_path,
            X=res.X,
            lambdas=lambdas,
            wall_ms=np.float64(res.wall_time_s * 1000.0),
            times_ms=np.array([t * 1000.0 for t in res.wall_times_s]),
            kkt=res.kkt_per_lambda,
            shape_name=np.array(name),
            spec_m=np.int64(spec.m),
            spec_n=np.int64(spec.n),
            spec_K=np.int64(spec.K),
            spec_sparsity=np.float64(spec.sparsity),
            spec_seed=np.int64(args.seed),
        )
        print(f"[{name}]   saved -> {out_path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
