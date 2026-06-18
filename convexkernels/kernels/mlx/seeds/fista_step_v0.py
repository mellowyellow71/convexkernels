"""First seed kernel: fused soft-threshold + axpy + momentum step.

The matvecs `A @ y` and `A.T @ r` stay as `mx.matmul` (already bandwidth-tuned).
This kernel fuses the cheap O(n) tail of one FISTA iteration into a single
Metal pass:

  1. z = y - t * g
  2. x_next = soft(z, t * lam)
  3. y_next = x_next + momentum * (x_next - x_prev)

Done as one kernel that reads y[i], g[i], x_prev[i] and writes x_next[i],
y_next[i] in one pass over the n elements. Saves 2 reads + 2 writes of HBM
relative to three separate ops; meaningful only for moderate-to-large n.

This is the **seed** the synthesis loop will mutate from in P3+. The point is
not that it's fast — it's that it's a clean, easy-to-mutate starting point
that proves the harness works end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from ..lib import LassoMLX


@dataclass
class FistaStateMLX:
    """MLX-backed FISTA state. Mirrors the numpy_ref `FistaState` shape."""
    x: mx.array
    y: mx.array
    theta: float


# Build the kernel once at module import time. mx.fast.metal_kernel returns a
# callable that the kernel body specializes via templates per call.
_FUSED_STEP_KERNEL = mx.fast.metal_kernel(
    name="fista_fused_zsoft_momentum",
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
        T abs_zi = metal::abs(zi);
        T sign_zi = (zi > T(0)) ? T(1) : ((zi < T(0)) ? T(-1) : T(0));
        T xi_new = (abs_zi > thresh)
            ? sign_zi * (abs_zi - thresh)
            : T(0);

        x_next[i] = xi_new;
        y_next[i] = xi_new + mom * (xi_new - x_prev[i]);
    """,
    ensure_row_contiguous=True,
)


def init_state(problem: LassoMLX) -> FistaStateMLX:
    n = problem.n
    return FistaStateMLX(
        x=mx.zeros(n, dtype=problem.dtype),
        y=mx.zeros(n, dtype=problem.dtype),
        theta=1.0,
    )


def fista_step(state: FistaStateMLX, problem: LassoMLX, t: float) -> FistaStateMLX:
    """One FISTA iteration with a fused MLX kernel for the O(n) tail."""
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


def solve(problem, recorder, *, kkt_tol, max_time_s, check_every: int = 25):
    """Algorithm-open seed entry point: plain FISTA owning its own loop.

    Reports progress through the trusted `Recorder` (which evaluates the KKT
    on the canonical numpy problem) and stops when the target is reached or the
    wall budget is spent.
    """
    state = init_state(problem)
    t = 1.0 / problem.L
    it = 0
    while it < 100000:
        it += 1
        state = fista_step(state, problem, t)
        if it % check_every == 0:
            recorder.record(state.x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(state.x)
    return state.x


__all__ = ["FistaStateMLX", "init_state", "fista_step", "solve"]
