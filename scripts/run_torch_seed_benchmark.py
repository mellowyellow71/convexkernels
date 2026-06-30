#!/usr/bin/env python3
"""Benchmark the torch/CUDA batched-FISTA-Gram path seed: precision ladder.

Measures *real* time-to-trusted-KKT on the GPU for the path workload (G@Y is a
throughput-bound GEMM, where reduced precision actually buys wall-clock), under
the same trusted Recorder the autoresearch loop uses:

    fp32           : reference
    bf16_switch    : bf16 G@Y until the KKT plateaus, then fp32 endgame
    fp16_switch    : same with fp16

The win grows with iteration count, so the harder (ill-conditioned / low-reg)
regimes are where reduced precision pays. Setup (Gram upload/precompute) is
timed separately; the headline metric is time_to_kkt (solve only), which the
Recorder reports excluding its own trusted-KKT eval cost.

    python scripts/run_torch_seed_benchmark.py --n 2048 --K 128 --reps 3
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def make_path(m, n, k, K, rho, seed):
    from convexkernels.frontend.lasso_path import LassoPath
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((m, n))
    if rho > 0:
        Z = np.sqrt(1 - rho) * Z + np.sqrt(rho) * rng.standard_normal((m, 1))
    A = Z / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lmax = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(0.9 * lmax, 0.02 * lmax, K)
    return LassoPath(A, b, lambdas)


def bench_strategy(prob, strat, *, kkt_tol, max_time_s, reps, check_every, device):
    from convexkernels.bench.metrics import trusted_kkt
    from convexkernels.synth.recorder import Recorder
    from convexkernels.kernels.torch.seeds import gram_fista_path_torch as seed
    kkt_fn = functools.partial(trusted_kkt, prob)
    t2k, setups, reached, kfin = [], [], [], []
    for r in range(reps + 1):  # first rep is a discarded warmup
        ts = time.perf_counter()
        view = seed.prepare_problem(prob, {"dtype_strategy": strat, "device": device})
        setup = time.perf_counter() - ts
        rec = Recorder(kkt_fn, max_time_s=max_time_s)
        X = seed.solve(view, rec, kkt_tol=kkt_tol, max_time_s=max_time_s, check_every=check_every)
        if r == 0:
            continue
        tk = rec.time_to_kkt(kkt_tol)
        t2k.append(tk)
        setups.append(setup)
        kf = trusted_kkt(prob, np.asarray(X, dtype=np.float64))
        kfin.append(kf)
        reached.append(kf < kkt_tol)
    return {
        "time_to_kkt_ms": median(t2k) * 1e3 if t2k else float("inf"),
        "setup_ms": median(setups) * 1e3,
        "reached": all(reached),
        "kkt_final": median(kfin),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=120, help="true nnz")
    ap.add_argument("--K", type=int, default=128, help="path length (lambdas)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if dev == "cuda":
        torch.cuda.synchronize()
        _ = (torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda"))
        torch.cuda.synchronize()  # warm context before any timed setup
    print(f"device={dev} torch={torch.__version__} "
          f"{'('+torch.cuda.get_device_name(0)+')' if dev=='cuda' else ''}")

    regimes = [
        ("well_cond", dict(m=args.m, n=args.n, k=args.k, K=args.K, rho=0.0, seed=0)),
        ("ill_cond",  dict(m=args.m, n=args.n, k=args.k, K=args.K, rho=0.9, seed=1)),
    ]
    strategies = ["fp64", "fp32_switch", "bf16_switch", "fp16_switch"]
    for label, kw in regimes:
        prob = make_path(**kw)
        print(f"\n=== {label}: m={prob.m} n={prob.n} K={prob.K} ===")
        ref = None
        for strat in strategies:
            r = bench_strategy(prob, strat, kkt_tol=args.tol, max_time_s=args.max_time_s,
                               reps=args.reps, check_every=args.check_every, device=dev)
            if strat == "fp64":
                ref = r["time_to_kkt_ms"]
            spd = (ref / r["time_to_kkt_ms"]) if (ref and r["reached"]) else float("nan")
            print(f"  {strat:13s} time_to_kkt={r['time_to_kkt_ms']:8.2f} ms  "
                  f"setup={r['setup_ms']:7.2f} ms  reached={r['reached']}  "
                  f"kkt={r['kkt_final']:.1e}  {spd:5.2f}x vs fp64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
