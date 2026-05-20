"""Batched FISTA-Gram path seed — v0 (hand-written reference).

Per-iter on a (n, K) batched iterate:

  g = grad_path_smooth(Y)                       # one matmul (Gram or direct)
  Z = Y - g / L                                 # elementwise (n, K)
  X_new = soft(Z, lambdas / L  per col)         # per-column threshold
  restart_k = sum((Y - X_new) * (X_new - X), axis=0) > 0   # per col
  theta_new = where(restart, 1, (1 + sqrt(1+4 theta^2))/2)
  mom        = where(restart, 0, (theta - 1) / theta_new)
  Y_new = X_new + mom[None, :] * (X_new - X)

The Gram precompute `G = A^T A` is computed once in `prepare_problem` and
reused across all K lambdas and all iterations. For wide p>>n where G
doesn't fit in a single Metal buffer (e.g. path_wide_hero: G would be
10 GB fp32), `LassoPathGramMLX.from_lasso_path_mlx` automatically falls
back to the direct form `A.T @ (A @ Y - b[:, None])`. The dispatch
happens once in `prepare_problem`; the inner loop is shape-agnostic.

**Per-column gradient restart** (O'Donoghue & Candes 2012) is in the
seed because vanilla FISTA on path_wide_hero needs >10000 iters cold-
started, blowing past any reasonable time budget. Restart converges
geometrically in practice (~100s of iters). Per-column lets each
lambda's iteration use its own restart schedule — important when
lambdas span 100x in magnitude and convergence speeds differ a lot
across columns.

This is the seed. Phase 3 autoresearch will mutate it: per-column
convergence masking (freeze columns that converged), fp16 inner with
fp32 KKT, fused Metal kernel for the elementwise tail (the current
seed lets MLX's lazy graph fuse on its own), tiling, SAFE/STRONG
screening rules, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from convexkernels.algorithms.fista_path import FistaPathState
from convexkernels.kernels.mlx.lib import LassoPathGramMLX, LassoPathMLX


@dataclass
class FistaPathStateMLX:
    """MLX-backed batched FISTA state. X, Y are (n, K); theta is (K,)."""
    X: mx.array
    Y: mx.array
    theta: mx.array

    @property
    def K(self) -> int:
        return int(self.X.shape[1])


def prepare_problem(
    problem: LassoPathMLX, config: dict | None = None,
) -> LassoPathGramMLX:
    """Build the per-iter problem view (Gram or direct), under the timing contract.

    Decided based on whether `n^2 * elem_size` fits in `gram_budget_bytes`.
    For path_wide_hero, falls back to direct form (two-matmul gradient)
    because Gram is 10 GB fp32.
    """
    config = config or {}
    dtype_strategy = config.get("dtype_strategy", "fp32")
    gradient_dtype = {
        "fp32": mx.float32,
        "fp16_storage": mx.float16,
        "mixed_gram": mx.float16,
    }.get(dtype_strategy, problem.dtype)
    kkt_dtype = mx.float32 if gradient_dtype != mx.float32 else None
    gram_budget = int(config.get("gram_budget_bytes", 8 * 1024 ** 3))
    return LassoPathGramMLX.from_lasso_path_mlx(
        problem,
        gradient_dtype=gradient_dtype,
        kkt_dtype=kkt_dtype,
        gram_budget_bytes=gram_budget,
    )


def init_state(problem: LassoPathGramMLX) -> FistaPathStateMLX:
    n = problem.n
    K = problem.K
    return FistaPathStateMLX(
        X=mx.zeros((n, K), dtype=problem.dtype),
        Y=mx.zeros((n, K), dtype=problem.dtype),
        theta=mx.ones((K,), dtype=problem.dtype),
    )


def fista_path_step(
    state: FistaPathStateMLX, problem: LassoPathGramMLX, t: float,
) -> FistaPathStateMLX:
    """One batched FISTA-Gram iter with per-column gradient restart.

    All elementwise ops are MLX; the lazy graph fuses the (n, K) tail
    into a few Metal kernels under the hood. We don't write the fused
    Metal kernel by hand here — that's a Phase 3 autoresearch lever.
    """
    g = problem.grad_path_smooth(state.Y)
    inv_L = problem.dtype.__class__  # placeholder; use scalar below

    # Z = Y - t * g; threshold = t * lambdas (with t = 1/L)
    Z = state.Y - t * g
    kappa = (t * problem.lambdas).astype(state.Y.dtype)        # (K,)
    X_new = mx.sign(Z) * mx.maximum(mx.abs(Z) - kappa[None, :], 0.0)

    # Per-column gradient-restart indicator:
    # restart_k = ((Y - X_new) * (X_new - X_prev)).sum(axis=0) > 0
    diff_y = state.Y - X_new                    # (n, K)
    diff_x = X_new - state.X                    # (n, K)
    restart = mx.sum(diff_y * diff_x, axis=0) > 0.0   # (K,) bool

    theta_advance = 0.5 * (
        1.0 + mx.sqrt(1.0 + 4.0 * state.theta * state.theta)
    )
    theta_new = mx.where(
        restart, mx.ones_like(state.theta), theta_advance,
    )
    mom = mx.where(
        restart,
        mx.zeros_like(state.theta),
        (state.theta - 1.0) / theta_new,
    )
    Y_new = X_new + mom[None, :] * diff_x       # = X_new + mom * (X_new - X_prev)

    return FistaPathStateMLX(X=X_new, Y=Y_new, theta=theta_new)


def kkt_max(
    state: FistaPathStateMLX, problem: LassoPathGramMLX,
) -> float:
    """Max-over-columns KKT residual, on device.

    Used by the host driver as the convergence gate every
    `convergence_check_every` iterations. Returns a Python float
    (forces a host sync — but only every N iters, not every iter).
    """
    return problem.kkt_residual_max(state.X)


def to_host_state(state: FistaPathStateMLX) -> FistaPathState:
    return FistaPathState(X=state.X, Y=state.Y, theta=state.theta)


__all__ = [
    "FistaPathStateMLX", "prepare_problem", "init_state",
    "fista_path_step", "kkt_max", "to_host_state",
]
