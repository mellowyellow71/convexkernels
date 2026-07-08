#!/usr/bin/env python3
"""Benchmark the algopool families on the LASSO-path shapes.

Runs each generated `solve()` module against a numpy `LassoPath` under the same
Recorder contract the synth harness uses (record(X) -> trusted KKT), and reports
time-to-KKT and the anytime (time, KKT) trajectory. The plain FISTA-path
candidate (the mutation-layer base, no screening) is the in-family baseline, so
the speedup here is *screening vs. no screening* on the identical harness — the
evidence for the PR's claim, committed as an .npz alongside it.

    python scripts/run_algopool_bench.py --shapes path_tall_medium,path_wide_small \
        --out convexkernels/bench/cache/algopool_bench.npz

`path_wide_hero` (m=1000, n=50000, K=50) is the hero shape; it is slow in pure
numpy (no MLX here), so it is opt-in via --shapes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from convexkernels.bench.metrics import trusted_kkt
from convexkernels.bench.path_shapes import get_path_shape, make_path_problem
from convexkernels.frontend.lasso_path import LassoPath
from convexkernels.synth.proposers.algopool import SCREENING_FAMILIES, render_fista_path


class Recorder:
    def __init__(self, problem, max_time_s):
        self.p = problem
        self.max_time_s = max_time_s
        self._t0 = time.perf_counter()
        self._overhead = 0.0
        self.last = float("inf")
        self.traj: list[tuple[float, float]] = []

    def record(self, x):
        o0 = time.perf_counter()
        self.last = trusted_kkt(self.p, np.asarray(x, dtype=np.float64))
        self._overhead += time.perf_counter() - o0
        self.traj.append((time.perf_counter() - self._t0 - self._overhead, self.last))
        return self.last

    def should_stop(self, tol):
        return self.last <= tol or (time.perf_counter() - self._t0) >= self.max_time_s


def _load(source: str):
    mod = types.ModuleType("cand")
    exec(compile(source, "<algopool-candidate>", "exec"), mod.__dict__)
    return mod


def time_to_kkt(traj, tol):
    for t, k in traj:
        if k <= tol:
            return t
    return float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="path_tall_medium,path_wide_small")
    ap.add_argument("--kkt-tol", type=float, default=1e-6)
    ap.add_argument("--max-time-s", type=float, default=120.0)
    ap.add_argument("--out", default="convexkernels/bench/artifacts/algopool_bench.npz")
    args = ap.parse_args()

    candidates = {
        **SCREENING_FAMILIES,
        "fista_path_baseline": render_fista_path({"restart": True, "check_every": 10}),
    }
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    saved: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    print(f"algopool bench  tol={args.kkt_tol:.0e}  shapes={shapes}")
    for shape in shapes:
        spec = get_path_shape(shape)
        A, b, lambdas = make_path_problem(spec)
        prob = LassoPath(A, b, lambdas)
        print(f"\n[{shape}] m={prob.m} n={prob.n} K={prob.K}")
        summary[shape] = {"m": prob.m, "n": prob.n, "K": prob.K, "results": {}}
        for name, source in candidates.items():
            mod = _load(source)
            rec = Recorder(prob, args.max_time_s)
            t0 = time.perf_counter()
            X = mod.solve(prob, rec, kkt_tol=args.kkt_tol, max_time_s=args.max_time_s)
            wall = time.perf_counter() - t0
            kkt = trusted_kkt(prob, np.asarray(X, dtype=np.float64))
            ttk = time_to_kkt(rec.traj, args.kkt_tol)
            ok = "reached" if kkt <= args.kkt_tol else "MISSED "
            print(f"  {name:22s} {ok}  time_to_kkt={ttk:7.3f}s  "
                  f"wall={wall:7.3f}s  kkt={kkt:.2e}  nnz={int(np.count_nonzero(X))}")
            summary[shape]["results"][name] = {
                "reached": bool(kkt <= args.kkt_tol),
                "time_to_kkt_s": None if ttk == float("inf") else round(ttk, 4),
                "wall_s": round(wall, 4), "kkt_final": kkt, "nnz": int(np.count_nonzero(X)),
            }
            if rec.traj:
                arr = np.asarray(rec.traj, dtype=np.float64)
                saved[f"{shape}__{name}__t"] = arr[:, 0]
                saved[f"{shape}__{name}__kkt"] = arr[:, 1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **saved)
    meta = {"kkt_tol": args.kkt_tol, "max_time_s": args.max_time_s, "shapes": summary}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nartifact -> {out}  ({len(saved)//2} trajectories)")
    print(f"summary  -> {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
