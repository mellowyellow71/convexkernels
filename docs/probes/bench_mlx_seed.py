"""Benchmark the MLX seed kernel vs numpy_ref on all bench shapes (Mac only)."""

from __future__ import annotations

import time

import mlx.core as mx

from convexkernels.algorithms.fista import fista
from convexkernels.bench.shapes import DEFAULT_SHAPES, make_synthetic_lasso
from convexkernels.kernels.mlx.lib import LassoMLX
from convexkernels.kernels.mlx.seeds.fista_step_v0 import (
    fista_step as mlx_step,
    init_state as mlx_init,
)


def main() -> None:
    # tol=1e-6 is realistic for fp32; below that you hit the precision floor.
    TOL = 1e-6
    print(f"{'shape':<14} {'backend':<14} {'iters':>6} {'wall_s':>9} {'ms/iter':>9} {'kkt':>10}")
    for spec in DEFAULT_SHAPES:
        prob = make_synthetic_lasso(spec, seed=0)

        t0 = time.perf_counter()
        r = fista(prob, max_iters=5000, tol=TOL, variant="basic", record_history=False)
        wall = time.perf_counter() - t0
        ms_per_iter = wall * 1000 / max(r.n_iters, 1)
        print(f"{spec.name:<14} {'numpy_ref':<14} {r.n_iters:>6} "
              f"{wall:>9.3f} {ms_per_iter:>9.2f} {r.kkt_final:>10.2e}")

        prob_mlx = LassoMLX.from_lasso(prob, dtype=mx.float32)
        t0 = time.perf_counter()
        r = fista(prob_mlx, max_iters=5000, tol=TOL, variant="basic", record_history=False,
                  kernel_step=mlx_step, kernel_init=mlx_init)
        wall = time.perf_counter() - t0
        ms_per_iter = wall * 1000 / max(r.n_iters, 1)
        print(f"{spec.name:<14} {'mlx_seed_fp32':<14} {r.n_iters:>6} "
              f"{wall:>9.3f} {ms_per_iter:>9.2f} {r.kkt_final:>10.2e}")


if __name__ == "__main__":
    main()
