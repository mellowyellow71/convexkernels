"""Seed MLX kernel for nonnegative LASSO FISTA.

This mirrors `fista_step_v0` for LASSO but swaps soft-thresholding for the
nonnegative LASSO prox:

    x_next = max(y - t*g - t*lam, 0)
"""

from __future__ import annotations

import math

import mlx.core as mx

from convexkernels.kernels.mlx.lib import NonnegativeLassoMLX
from convexkernels.kernels.mlx.seeds.fista_step_v0 import FistaStateMLX


_FUSED_STEP_KERNEL = mx.fast.metal_kernel(
    name="nn_lasso_fused_zpos_momentum",
    input_names=["y", "g", "x_prev", "scalars"],
    output_names=["x_next", "y_next"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i >= y_shape[0]) return;
        T t      = scalars[0];
        T lam    = scalars[1];
        T mom    = scalars[2];
        T thresh = t * lam;

        T zi = y[i] - t * g[i];
        T xi_new = metal::max(zi - thresh, T(0));

        x_next[i] = xi_new;
        y_next[i] = xi_new + mom * (xi_new - x_prev[i]);
    """,
    ensure_row_contiguous=True,
)


def init_state(problem: NonnegativeLassoMLX) -> FistaStateMLX:
    n = problem.n
    return FistaStateMLX(
        x=mx.zeros(n, dtype=problem.dtype),
        y=mx.zeros(n, dtype=problem.dtype),
        theta=1.0,
    )


def fista_step(
    state: FistaStateMLX,
    problem: NonnegativeLassoMLX,
    t: float,
) -> FistaStateMLX:
    """One nonnegative LASSO FISTA iteration with fused prox/momentum tail."""
    g = problem.grad_smooth(state.y)

    theta_next = (1.0 + math.sqrt(1.0 + 4.0 * state.theta * state.theta)) / 2.0
    momentum = (state.theta - 1.0) / theta_next

    scalars = mx.array([t, problem.lam, momentum], dtype=problem.dtype)

    x_next, y_next = _FUSED_STEP_KERNEL(
        inputs=[state.y, g, state.x, scalars],
        template=[("T", problem.dtype)],
        grid=(state.x.shape[0], 1, 1),
        threadgroup=(min(256, state.x.shape[0]), 1, 1),
        output_shapes=[state.x.shape, state.x.shape],
        output_dtypes=[problem.dtype, problem.dtype],
    )

    return FistaStateMLX(x=x_next, y=y_next, theta=theta_next)
