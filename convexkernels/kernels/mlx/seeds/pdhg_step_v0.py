"""First seed kernel for PDHG / Chambolle-Pock on TV-L2 1D denoising.

The seed deliberately keeps the per-iter compute readable and modular so the
synthesis loop has clean lever points. Per iteration:

  1. Kx_bar = grad(x_bar)           # length n-1
  2. y_pre  = y + sigma * Kx_bar
  3. y_new  = clip(y_pre, -lam, lam)            # prox of g* (indicator)
  4. KTy    = grad_T(y_new)         # length n
  5. x_pre  = x - tau * KTy
  6. x_new  = (x_pre + tau * b) / (1 + tau)     # prox of tau f, f = (1/2)||x-b||^2
  7. x_bar_new = x_new + theta * (x_new - x)

The cheap O(n) tail (steps 5-7) is fused into one Metal pass; steps 1-4 stay
as MLX ops. The autoresearch loop's job is to find tighter fusions: e.g.
fuse the tail with steps 1-3, or rewrite K_T as a custom kernel that doesn't
materialize the padded buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ..lib import TVDenoising1DMLX


@dataclass
class PdhgStateMLX:
    """MLX-backed PDHG state. Mirrors the numpy reference state shape."""
    x: mx.array
    y: mx.array
    x_bar: mx.array
    tau: float
    sigma: float
    theta: float


_FUSED_TAIL_KERNEL = mx.fast.metal_kernel(
    name="pdhg_tv_fused_tail",
    input_names=["x_old", "KTy", "b", "scalars"],
    output_names=["x_next", "x_bar_next"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i >= x_old_shape[0]) return;
        T tau   = scalars[0];
        T theta = scalars[1];

        T x_pre  = x_old[i] - tau * KTy[i] + tau * b[i];
        T x_new  = x_pre / (T(1) + tau);

        x_next[i]     = x_new;
        x_bar_next[i] = x_new + theta * (x_new - x_old[i]);
    """,
    ensure_row_contiguous=True,
)


def init_state(
    problem: TVDenoising1DMLX,
    *,
    tau: float | None = None,
    sigma: float | None = None,
    theta: float = 1.0,
) -> PdhgStateMLX:
    n = problem.n
    m = problem.m
    L_K = float(problem.L_K)
    if tau is None:
        tau = 1.0 / L_K
    if sigma is None:
        sigma = 1.0 / L_K
    return PdhgStateMLX(
        x=mx.zeros(n, dtype=problem.dtype),
        y=mx.zeros(m, dtype=problem.dtype),
        x_bar=mx.zeros(n, dtype=problem.dtype),
        tau=float(tau),
        sigma=float(sigma),
        theta=float(theta),
    )


def pdhg_step(state: PdhgStateMLX, problem: TVDenoising1DMLX) -> PdhgStateMLX:
    """One PDHG/Chambolle-Pock iteration with a fused MLX kernel for the O(n) tail."""
    Kx_bar = problem.K_apply(state.x_bar)              # length n-1
    y_pre = state.y + state.sigma * Kx_bar
    y_new = mx.clip(y_pre, -problem.lam, problem.lam)  # prox g*

    KTy = problem.K_T_apply(y_new)                     # length n

    scalars = mx.array([state.tau, state.theta], dtype=problem.dtype)

    x_next, x_bar_next = _FUSED_TAIL_KERNEL(
        inputs=[state.x, KTy, problem.b, scalars],
        template=[("T", problem.dtype)],
        grid=(state.x.shape[0], 1, 1),
        threadgroup=(min(256, state.x.shape[0]), 1, 1),
        output_shapes=[state.x.shape, state.x.shape],
        output_dtypes=[problem.dtype, problem.dtype],
    )

    return PdhgStateMLX(
        x=x_next,
        y=y_new,
        x_bar=x_bar_next,
        tau=state.tau,
        sigma=state.sigma,
        theta=state.theta,
    )
