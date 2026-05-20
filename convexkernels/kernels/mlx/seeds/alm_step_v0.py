"""First seed kernel for ALM on equality-constrained QP, MLX backend.

The seed deliberately keeps the per-iter compute readable so the synthesis
loop has clean lever points. Per ALM iteration:

  1. rhs = -q + A^T (rho b - lam)         (linear assembly, MLX)
  2. y   = L^{-1} rhs                     (triangular solve, mx.linalg)
  3. x   = (L^T)^{-1} y                   (triangular solve, mx.linalg)
  4. lam = lam + rho (A x - b)            (multiplier update, MLX)

The Cholesky factor of (P + rho A^T A) is built once in `init_state` (or on
the first step if rho changes). The two triangular solves dominate per-iter
time. The multiplier update is fused with the constraint evaluation.

Autoresearch lever points the loop should be able to find:
- Reuse the rhs and constraint evaluation in a single Metal pass.
- Specialized trisolve for small/medium m (replace mx.linalg.solve_triangular
  with a hand-written kernel when n is small enough that launch overhead
  dominates the lapack call).
- fp16 storage on the factor with periodic fp32 correction.
- For fixed P, q, A, b, run k ALM iterations per public step (analogous to
  PDHG temporal fusion — but the linear solve makes the dependency cone
  global, not local, so this only helps if k is small).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

try:
    from ..lib import EqualityQPMLX
except ImportError:
    from convexkernels.kernels.mlx.lib import EqualityQPMLX


@dataclass
class AlmStateMLX:
    """MLX-backed ALM state. `factor` stores the cached Cholesky."""
    x: mx.array
    lam: mx.array
    rho: float
    factor: Any = None


def init_state(
    problem: EqualityQPMLX,
    *,
    rho: float = 1.0,
) -> AlmStateMLX:
    n = problem.n
    m = problem.m_constraints
    state = AlmStateMLX(
        x=mx.zeros(n, dtype=problem.dtype),
        lam=mx.zeros(m, dtype=problem.dtype),
        rho=float(rho),
        factor=None,
    )
    # Build the Cholesky factor eagerly so the first step doesn't pay the
    # build cost. The factor is cached on the state for all subsequent steps
    # at this rho.
    state.factor = problem.build_factor(state.rho)
    return state


def alm_step(state: AlmStateMLX, problem: EqualityQPMLX) -> AlmStateMLX:
    if state.factor is None:
        factor = problem.build_factor(state.rho)
    else:
        factor = state.factor
    rhs = problem.x_rhs(state.lam, state.rho)
    x_new = problem.solve_with_factor(factor, rhs)
    Ax_b = problem.A_apply(x_new) - problem.b_constraint
    lam_new = state.lam + state.rho * Ax_b
    return AlmStateMLX(x=x_new, lam=lam_new, rho=state.rho, factor=factor)
