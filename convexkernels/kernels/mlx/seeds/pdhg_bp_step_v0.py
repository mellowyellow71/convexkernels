"""First seed kernel for PDHG on basis pursuit, MLX backend.

Same algorithm template as `pdhg_step_v0` (TV) but K is dense A. Per iter:

  1. Kx_bar = A @ x_bar          (matvec, MLX)
  2. y_pre  = y + sigma * Kx_bar
  3. y_new  = y_pre - sigma * b  (prox of g*(y) = <b, y>)
  4. KTy    = A^T @ y_new        (matvec, MLX)
  5. x_pre  = x - tau * KTy
  6. x_new  = soft(x_pre, tau)   (prox of f(x) = ||x||_1)
  7. x_bar_new = x_new + theta * (x_new - x)

The matvecs (1) and (4) dominate per-iter cost on dense BP. Steps 5-7 are
fused into one Metal pass; everything else stays as MLX ops.

Cross-problem-transfer note (vs `pdhg_step_v0` for TV): the autoresearch
"recompute adjacent duals" trick from TV does NOT apply here because A is
dense — every output of K^T y depends on every entry of y. Temporal fusion
also doesn't work for the same reason. The autoresearch lever for BP is
genuinely different: matvec fusion, dtype experiments, and possibly
algorithmic restart variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

try:
    from ..lib import BasisPursuitMLX
except ImportError:
    from convexkernels.kernels.mlx.lib import BasisPursuitMLX


@dataclass
class PdhgStateMLX:
    x: mx.array
    y: mx.array
    x_bar: mx.array
    tau: float
    sigma: float
    theta: float


_FUSED_TAIL_KERNEL = mx.fast.metal_kernel(
    name="pdhg_bp_fused_tail",
    input_names=["x_old", "KTy", "scalars"],
    output_names=["x_next", "x_bar_next"],
    source="""
        uint i = thread_position_in_grid.x;
        if (i >= x_old_shape[0]) return;
        T tau   = scalars[0];
        T theta = scalars[1];

        T x_pre = x_old[i] - tau * KTy[i];
        T abs_xp = metal::abs(x_pre);
        T sign_xp = (x_pre > T(0)) ? T(1) : ((x_pre < T(0)) ? T(-1) : T(0));
        T x_new = (abs_xp > tau) ? sign_xp * (abs_xp - tau) : T(0);

        x_next[i]     = x_new;
        x_bar_next[i] = x_new + theta * (x_new - x_old[i]);
    """,
    ensure_row_contiguous=True,
)


def init_state(
    problem: BasisPursuitMLX,
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


def pdhg_step(state: PdhgStateMLX, problem: BasisPursuitMLX) -> PdhgStateMLX:
    Kx_bar = problem.K_apply(state.x_bar)              # m
    y_pre = state.y + state.sigma * Kx_bar             # m
    y_new = y_pre - state.sigma * problem.b            # m, prox of <b, y>

    KTy = problem.K_T_apply(y_new)                     # n

    scalars = mx.array([state.tau, state.theta], dtype=problem.dtype)
    x_next, x_bar_next = _FUSED_TAIL_KERNEL(
        inputs=[state.x, KTy, scalars],
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
