#!/usr/bin/env python3
"""Benchmark the hand-written MLX batched-FISTA-Gram path seed against Adelie.

Loads cached Adelie reference per shape, runs the MLX seed under the same
multi-rep harness, reports wall_ms + correctness (Frobenius distance to
Adelie's X cache + per-lambda KKT). This is the Phase 2 acceptance check.

Run on the M3 Pro:
    .venv/bin/python scripts/run_mlx_seed_benchmark.py \
        --shapes path_wide_hero,path_tall_medium,path_square,path_wide_small \
        --reps 5 --warmup 2

If MLX seed wall_ms beats Adelie's on any shape, the pivot's first
sentence is true *before* Phase 3 autoresearch even runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="path_wide_hero")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--convergence-check-every", type=int, default=10)
    ap.add_argument("--dtype", default="fp32",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--cache-dir", default="convexkernels/bench/cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from convexkernels.bench.lasso_path_baselines import run_mlx_fista_path
    from convexkernels.bench.path_shapes import (
        get_path_shape, make_path_problem,
    )
    from convexkernels.frontend.lasso_path import LassoPath

    cache_dir = Path(args.cache_dir)
    shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]
    print(f"benching MLX seed on shapes: {shape_names}")
    print(f"reps={args.reps} warmup={args.warmup} dtype={args.dtype} "
          f"tol={args.tol} max_iters={args.max_iters}")
    print()

    rows = []
    for name in shape_names:
        spec = get_path_shape(name)
        print(f"[{name}] m={spec.m} n={spec.n} K={spec.K}")

        # Load Adelie cache.
        adelie_npz = cache_dir / f"adelie_path_{name}.npz"
        if not adelie_npz.exists():
            print(f"[{name}]   NO Adelie cache at {adelie_npz} — skipping")
            continue
        adelie = np.load(adelie_npz)
        X_adelie = adelie["X"]
        adelie_ms = float(adelie["wall_ms"])

        # Build problem.
        t_gen = time.perf_counter()
        A, b, lambdas = make_path_problem(spec, seed=args.seed)
        prob = LassoPath(A, b, lambdas)
        print(f"[{name}]   problem generated in "
              f"{time.perf_counter() - t_gen:.2f}s")

        # Run MLX seed.
        t_run = time.perf_counter()
        res = run_mlx_fista_path(
            prob, reps=args.reps, warmup=args.warmup,
            max_iters=args.max_iters, tol=args.tol,
            convergence_check_every=args.convergence_check_every,
            dtype=args.dtype,
        )
        run_total = time.perf_counter() - t_run

        # Correctness check vs Adelie cache.
        rel = float(np.linalg.norm(res.X - X_adelie, "fro") / max(
            np.linalg.norm(X_adelie, "fro"), 1e-12,
        ))
        speedup = adelie_ms / (res.wall_time_s * 1000.0)
        seed_ms = res.wall_time_s * 1000.0

        print(f"[{name}]   MLX seed: wall={seed_ms:.1f} ms median "
              f"(reps={[f'{t*1000:.1f}' for t in res.wall_times_s]})")
        print(f"[{name}]   max per-lambda KKT (ours)={res.kkt_per_lambda.max():.2e}")
        print(f"[{name}]   adelie ref: {adelie_ms:.1f} ms; speedup = "
              f"{speedup:.2f}x")
        print(f"[{name}]   rel Frobenius vs Adelie: {rel:.3e} "
              f"(< 1e-4 target: {'PASS' if rel < 1e-4 else 'FAIL'})")
        print(f"[{name}]   total harness: {run_total:.1f}s")
        print()

        rows.append({
            "shape": name, "adelie_ms": adelie_ms, "seed_ms": seed_ms,
            "speedup": speedup, "rel_frob": rel,
            "max_kkt": float(res.kkt_per_lambda.max()),
            "reps_ms": [t * 1000.0 for t in res.wall_times_s],
        })

    if rows:
        print("=" * 78)
        print(f"{'shape':<20} {'adelie ms':>10} {'seed ms':>10} {'speedup':>8} "
              f"{'rel-Frob':>10} {'max KKT':>10}")
        print("-" * 78)
        for r in rows:
            print(f"{r['shape']:<20} {r['adelie_ms']:>10.1f} {r['seed_ms']:>10.1f} "
                  f"{r['speedup']:>8.2f} {r['rel_frob']:>10.2e} {r['max_kkt']:>10.2e}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
