"""First seed kernel for ADMM-on-LASSO, MLX backend.

Per ADMM iteration on LASSO with x = z splitting:

  1. rhs = A^T b + rho (z - u)         (matvec; A^T b is cached)
  2. x   = (A^T A + rho I)^{-1} rhs    (trisolve, mx.linalg, CPU stream)
  3. z   = soft(x + u, lam / rho)
  4. u   += x - z

Factor is built once in init_state. Per-iter cost is dominated by the
trisolve on n x n. Same lever as ALM-equality-QP: replace mx.linalg trisolve
with custom Metal kernel when launch overhead is the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

try:
    from ..lib import LassoAdmmMLX
except ImportError:
    from convexkernels.kernels.mlx.lib import LassoAdmmMLX


@dataclass
class AdmmStateMLX:
    x: mx.array
    z: mx.array
    u: mx.array
    rho: float
    factor: Any = None


def init_state(
    problem: LassoAdmmMLX,
    *,
    rho: float | None = None,
) -> AdmmStateMLX:
    """rho default = problem.default_rho (Boyd's lambda_max heuristic).
    The autoresearch loop can override via a kernel_step / driver kwarg if
    the model wants to experiment with different penalties.
    """
    n = problem.n
    if rho is None:
        rho = float(getattr(problem, "default_rho", 1.0))
    state = AdmmStateMLX(
        x=mx.zeros(n, dtype=problem.dtype),
        z=mx.zeros(n, dtype=problem.dtype),
        u=mx.zeros(n, dtype=problem.dtype),
        rho=float(rho),
        factor=None,
    )
    state.factor = problem.build_factor(state.rho)
    return state


def admm_step(state: AdmmStateMLX, problem: LassoAdmmMLX) -> AdmmStateMLX:
    if state.factor is None:
        factor = problem.build_factor(state.rho)
    else:
        factor = state.factor
    rhs = problem.x_rhs_admm(state.z, state.u, state.rho)
    x_new = problem.solve_with_factor(factor, rhs)
    z_new = problem.prox_g(x_new + state.u, 1.0 / state.rho)
    u_new = state.u + x_new - z_new
    return AdmmStateMLX(x=x_new, z=z_new, u=u_new, rho=state.rho, factor=factor)
