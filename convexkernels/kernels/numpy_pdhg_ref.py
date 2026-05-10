"""Reference numpy kernel for PDHG / Chambolle-Pock.

Sister of `numpy_ref.py` (FISTA) — the correctness oracle for MLX PDHG kernels.
Every PDHG variant the synth loop discovers is tested against this via
``assert_pdhg_equivalent`` / primal-dual-gap-gated equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PdhgState:
    x: Any
    y: Any
    x_bar: Any
    tau: float
    sigma: float
    theta: float


def init_state(
    problem: Any,
    *,
    tau: float | None = None,
    sigma: float | None = None,
    theta: float = 1.0,
) -> PdhgState:
    """Zero-init PDHG iterates. Defaults to ``tau = sigma = 1 / L_K`` so
    ``tau * sigma * L_K^2 = 1`` (boundary feasibility).
    """
    L_K = float(problem.L_K)
    if tau is None:
        tau = 1.0 / L_K
    if sigma is None:
        sigma = 1.0 / L_K
    x0 = np.zeros(problem.n) if not hasattr(problem, "shape") else np.zeros(problem.shape)
    y0 = problem.K_apply(x0)
    if isinstance(y0, tuple):
        y0 = tuple(np.zeros_like(yi) for yi in y0)
    else:
        y0 = np.zeros_like(y0)
    x_bar0 = x0.copy() if hasattr(x0, "copy") else x0
    return PdhgState(
        x=x0, y=y0, x_bar=x_bar0,
        tau=float(tau), sigma=float(sigma), theta=float(theta),
    )


def pdhg_step(state: PdhgState, problem: Any) -> PdhgState:
    """One PDHG iteration in pure numpy (Chambolle-Pock 2011, Algorithm 1)."""
    Kx_bar = problem.K_apply(state.x_bar)
    if isinstance(state.y, tuple):
        y_new = tuple(yi + state.sigma * kxi for yi, kxi in zip(state.y, Kx_bar))
    else:
        y_new = state.y + state.sigma * Kx_bar
    y_new = problem.prox_g_conjugate(y_new, state.sigma)

    KTy = problem.K_T_apply(y_new)
    x_new = problem.prox_f(state.x - state.tau * KTy, state.tau)

    x_bar_new = x_new + state.theta * (x_new - state.x)
    return PdhgState(
        x=x_new, y=y_new, x_bar=x_bar_new,
        tau=state.tau, sigma=state.sigma, theta=state.theta,
    )
