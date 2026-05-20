"""Primal-Dual Hybrid Gradient (Chambolle-Pock 2011) driver.

Specimen-2 algorithm for the synthesis loop. Solves saddle-point problems

    min_x max_y  <K x, y> + f(x) - g*(y)

via the iterates

    y_{k+1}     = prox_{sigma g*}(y_k + sigma K x_bar_k)
    x_{k+1}     = prox_{tau f}(x_k - tau K^T y_{k+1})
    x_bar_{k+1} = x_{k+1} + theta (x_{k+1} - x_k)

Step convergence requires ``tau * sigma * ||K||^2 <= 1``. The accelerated
variant (Algorithm 2 of CP 2011) requires f uniformly convex with modulus
``gamma > 0`` and adapts (tau, sigma, theta) per iter; convergence rate
improves from O(1/k) to O(1/k^2).

The host owns the outer loop, step adaptation, gap convergence test, and
history. The per-iter compute is a swappable ``kernel_step``; default is a
pure-numpy reference. MLX kernels swap it via ``kernel_step`` /
``kernel_init`` keyword args, mirroring `fista.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Optional

import numpy as np


@dataclass
class PdhgState:
    """Numpy reference state for PDHG. MLX kernels carry their own version."""
    x: Any
    y: Any
    x_bar: Any
    tau: float
    sigma: float
    theta: float


@dataclass
class PdhgResult:
    x: Any
    y: Any
    n_iters: int
    converged: bool
    gap_final: float
    wall_time_s: float
    history: dict = field(default_factory=dict)


def _default_numpy_init(problem: Any, *, tau: float, sigma: float, theta: float) -> PdhgState:
    """Numpy zero-init for PDHG iterates.

    Works for both 1D problems (TVDenoising1D: x is length n, y is length m)
    and 2D problems (TVDenoising2D: y is a tuple). The ABC isn't formalized;
    we duck-type via ``problem.K_apply(zeros)`` to size y.
    """
    x0 = np.zeros(problem.n) if not hasattr(problem, "shape") else np.zeros(problem.shape)
    y0 = problem.K_apply(x0)
    if isinstance(y0, tuple):
        y0 = tuple(np.zeros_like(yi) for yi in y0)
    else:
        y0 = np.zeros_like(y0)
    x_bar0 = x0.copy() if hasattr(x0, "copy") else x0
    return PdhgState(x=x0, y=y0, x_bar=x_bar0, tau=tau, sigma=sigma, theta=theta)


def _numpy_pdhg_step(state: PdhgState, problem: Any) -> PdhgState:
    """One PDHG iteration in pure numpy. The reference correctness oracle."""
    Kx_bar = problem.K_apply(state.x_bar)
    if isinstance(state.y, tuple):
        y_new = tuple(yi + state.sigma * kxi for yi, kxi in zip(state.y, Kx_bar))
    else:
        y_new = state.y + state.sigma * Kx_bar
    y_new = problem.prox_g_conjugate(y_new, state.sigma)

    KTy = problem.K_T_apply(y_new)
    x_new = problem.prox_f(state.x - state.tau * KTy, state.tau)

    if isinstance(x_new, np.ndarray):
        x_bar_new = x_new + state.theta * (x_new - state.x)
    else:
        x_bar_new = x_new + state.theta * (x_new - state.x)
    return PdhgState(
        x=x_new,
        y=y_new,
        x_bar=x_bar_new,
        tau=state.tau,
        sigma=state.sigma,
        theta=state.theta,
    )


def pdhg(
    problem: Any,
    *,
    max_iters: int = 5000,
    tol: float = 1e-6,
    variant: Literal["basic", "accelerated", "restart"] = "basic",
    tau: Optional[float] = None,
    sigma: Optional[float] = None,
    theta: float = 1.0,
    gamma: Optional[float] = None,
    kernel_step: Callable[[PdhgState, Any], PdhgState] = _numpy_pdhg_step,
    kernel_init: Callable[..., PdhgState] = _default_numpy_init,
    record_history: bool = True,
    convergence_check_every: int = 1,
    callback: Optional[Callable[[int, PdhgState, float], None]] = None,
) -> PdhgResult:
    """Run PDHG / Chambolle-Pock on `problem` until gap < tol or max_iters.

    Variants:
      - ``basic``: Algorithm 1 of CP 2011. Constant (tau, sigma, theta).
      - ``accelerated``: Algorithm 2 of CP 2011. Requires ``gamma > 0``
        (strong-convexity modulus of f). Adapts (tau, sigma, theta) per iter.
      - ``restart``: O'Donoghue-Candes-style gradient restart. Resets the
        extrapolation when (y_new - y_old) . (x_new - x_old) > 0 plus the
        analogous primal indicator. Conservative — falls back to basic when
        either term is missing.

    If ``tau`` / ``sigma`` are not supplied, defaults are ``tau = sigma = 1 / L_K``
    so ``tau * sigma * L_K^2 = 1`` (boundary feasibility). `problem.L_K` is
    expected; ``L_K = ||K||_2``.
    """
    L_K = float(problem.L_K)
    if tau is None:
        tau = 1.0 / L_K
    if sigma is None:
        sigma = 1.0 / L_K
    if variant == "accelerated" and gamma is None:
        gamma = 1.0  # f(x) = (1/2)||x-b||^2 has modulus 1; default works for TV-L2

    # Time `kernel_init` as part of `wall_time_s` so a candidate cannot game
    # the eval by stuffing per-step compute into init_state. Without this an
    # accepted variant can move N solver iterations into init_state, report
    # iters=k where k << N, and pass the speedup gate while doing more total
    # work than the seed. The Sakana CUDA Engineer benchmark-cheating mode.
    t0 = perf_counter()
    state = kernel_init(problem, tau=tau, sigma=sigma, theta=theta)

    history: dict = {"gap": [], "wall_time": []} if record_history else {}
    gap = float("inf")
    converged = False
    k = -1
    check_every = max(1, int(convergence_check_every))

    for k in range(max_iters):
        if variant == "accelerated":
            # CP 2011 Algorithm 2: update (tau, sigma, theta) BEFORE the prox
            # steps so the new step sizes are used in the next iterate.
            theta_k = 1.0 / float(np.sqrt(1.0 + 2.0 * gamma * state.tau))
            tau_k = theta_k * state.tau
            sigma_k = state.sigma / theta_k
            state = type(state)(
                x=state.x,
                y=state.y,
                x_bar=state.x_bar,
                tau=tau_k,
                sigma=sigma_k,
                theta=theta_k,
            )

        state_new = kernel_step(state, problem)

        if variant == "restart":
            # Restart when the extrapolation step opposes the primal motion.
            try:
                primal_indicator = float(
                    np.sum((np.asarray(state.x) - np.asarray(state_new.x))
                           * (np.asarray(state_new.x) - np.asarray(state.x)))
                )
            except Exception:
                primal_indicator = 0.0
            if primal_indicator > 0.0:
                state_new = type(state_new)(
                    x=state_new.x,
                    y=state_new.y if not isinstance(state_new.y, tuple) else state_new.y,
                    x_bar=state_new.x,
                    tau=state.tau,
                    sigma=state.sigma,
                    theta=state.theta,
                )

        state = state_new

        # Convergence check + trajectory recording every `check_every` iters.
        # On MLX, computing `gap` forces a full GPU sync (gap calls float() on
        # several mx.sum reductions); checking every iter on small problems is
        # the dominant cost. check_every=1 preserves prior behavior for numpy
        # reference and small-problem tests.
        if (k + 1) % check_every == 0 or k == max_iters - 1:
            gap = problem.primal_dual_gap(state.x, state.y)
            if record_history:
                history["gap"].append(gap)
                history["wall_time"].append(perf_counter() - t0)
            if callback is not None:
                callback(k, state, gap)
            if gap < tol:
                converged = True
                break

    final_x = state.x if isinstance(state.x, np.ndarray) else np.asarray(state.x)
    return PdhgResult(
        x=final_x,
        y=state.y,
        n_iters=k + 1,
        converged=converged,
        gap_final=gap,
        wall_time_s=perf_counter() - t0,
        history=history,
    )
