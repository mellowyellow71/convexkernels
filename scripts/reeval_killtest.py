"""Re-evaluate selected kill-test champions under the corrected timing contract.

Validates that:
  - the honest 256-step champion (gen 10) holds up under the new eval
  - the gamed 30x256-warm-start champion (gen 23) is now correctly billed
    for its setup work and no longer beats the seed

Usage:
  PYTHONPATH=. .venv/bin/python scripts/reeval_killtest.py
"""
from __future__ import annotations

import json
import statistics
import time
import sys
from pathlib import Path

import numpy as np

from convexkernels.algorithms.pdhg import pdhg
from convexkernels.bench.shapes import DEFAULT_TV1D_SHAPES, make_synthetic_tv_1d
from convexkernels.kernels.mlx.lib import TVDenoising1DMLX
import mlx.core as mx


CANDIDATES = [
    ("seed", "convexkernels.kernels.mlx.seeds.pdhg_step_v0"),
    ("gen10_honest_256step", "synth_run_pdhg_tv_killtest/runs/fefd55af-cdfb-47b5-9f1f-995df2ff1e6c/source.py"),
    ("gen23_gamed_warmstart", "synth_run_pdhg_tv_killtest/runs/08dd1879-1e71-46b0-ba14-db5b45d57cce/source.py"),
]


def _load_module(spec):
    if spec.endswith(".py"):
        import importlib.util
        m = importlib.util.spec_from_file_location("cand", spec)
        mod = importlib.util.module_from_spec(m)
        sys.modules["cand"] = mod
        m.loader.exec_module(mod)
        return mod
    import importlib
    return importlib.import_module(spec)


def main():
    spec = next(s for s in DEFAULT_TV1D_SHAPES if s.name == "tv1d_medium")
    prob_np = make_synthetic_tv_1d(spec, seed=0)

    for label, source_spec in CANDIDATES:
        print(f"\n=== {label}  ({source_spec}) ===")
        try:
            mod = _load_module(source_spec)
        except Exception as exc:
            print(f"  load failed: {exc}")
            continue

        prob_mlx = TVDenoising1DMLX.from_problem(prob_np, dtype=mx.float32)
        if hasattr(mod, "prepare_problem"):
            try:
                prob_mlx = mod.prepare_problem(prob_mlx)
            except TypeError:
                prob_mlx = mod.prepare_problem(prob_mlx, {})

        kstep = getattr(mod, "pdhg_step")
        kinit = getattr(mod, "init_state")

        # Warmup
        try:
            pdhg(prob_mlx, max_iters=5000, tol=1e-6,
                 kernel_step=kstep, kernel_init=kinit,
                 record_history=False, convergence_check_every=10)
        except Exception as exc:
            print(f"  warmup failed: {exc}")
            continue

        times_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = pdhg(prob_mlx, max_iters=5000, tol=1e-6,
                       kernel_step=kstep, kernel_init=kinit,
                       record_history=False, convergence_check_every=10)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(times_ms)
        std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        print(f"  iters={res.n_iters} gap={res.gap_final:.2e} "
              f"wall_ms median={med:.2f} std={std:.2f} (5 reps, includes kernel_init)")


if __name__ == "__main__":
    main()
