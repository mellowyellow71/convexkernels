"""FISTA driver.

The host owns the outer loop, step-size, restart policy, KKT convergence test,
and history. The per-iter compute is a swappable `KernelStep` (default is the
numpy reference; P3+ ships MLX kernels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Optional

import numpy as np

from ..frontend.problem import Problem
from ..kernels.numpy_ref import FistaState, fista_step as _numpy_fista_step


@dataclass
class FistaResult:
    x: Any  # numpy or mlx array; converted to numpy at the boundary
    n_iters: int
    converged: bool
    kkt_final: float
    wall_time_s: float
    history: dict = field(default_factory=dict)


def _restart_indicator(state_prev: FistaState, state_new: FistaState) -> float:
    # O'Donoghue & Candes 2012: restart when (y_k - x_{k+1}) . (x_{k+1} - x_k) > 0.
    # Backend-agnostic: works for numpy and mlx arrays via .sum() and float().
    diff_y = state_prev.y - state_new.x
    diff_x = state_new.x - state_prev.x
    return float((diff_y * diff_x).sum())


def _default_numpy_init(problem: Problem) -> FistaState:
    n = problem.n
    return FistaState(x=np.zeros(n), y=np.zeros(n), theta=1.0)


def fista(
    problem: Problem,
    *,
    max_iters: int = 1000,
    tol: float = 1e-6,
    variant: Literal["basic", "restart"] = "basic",
    kernel_step: Callable[[FistaState, Problem, float], FistaState] = _numpy_fista_step,
    kernel_init: Callable[[Problem], FistaState] = _default_numpy_init,
    record_history: bool = True,
    callback: Optional[Callable[[int, FistaState, float], None]] = None,
) -> FistaResult:
    """Run FISTA on `problem` until KKT < tol or max_iters.

    `variant`:
      - "basic": vanilla FISTA, Beck & Teboulle 2009.
      - "restart": gradient-restart FISTA, O'Donoghue & Candes 2012. Resets
        theta and momentum when the last step opposed the descent direction.

    `kernel_step` is the per-iter compute; `kernel_init` builds the initial
    state. Both default to the numpy reference. MLX kernels override both.
    """
    state = kernel_init(problem)
    t = 1.0 / problem.L

    history: dict = {"kkt": [], "wall_time": []} if record_history else {}
    t0 = perf_counter()
    kkt = float("inf")
    converged = False
    k = -1

    for k in range(max_iters):
        state_new = kernel_step(state, problem, t)

        if variant == "restart" and _restart_indicator(state, state_new) > 0.0:
            # Reset momentum: theta=1 and y=x. Use the kernel's array module via
            # state_new.x.copy() (both numpy and mlx implement .copy() on arrays
            # but for mlx, x is already immutable, so a fresh reference is fine).
            state_new = type(state_new)(x=state_new.x, y=state_new.x, theta=1.0)

        state = state_new
        kkt = problem.kkt_residual(state.x)

        if record_history:
            history["kkt"].append(kkt)
            history["wall_time"].append(perf_counter() - t0)
        if callback is not None:
            callback(k, state, kkt)
        if kkt < tol:
            converged = True
            break

    # Convert mlx array to numpy at the boundary for FistaResult.x.
    final_x = state.x if isinstance(state.x, np.ndarray) else np.asarray(state.x)
    return FistaResult(
        x=final_x,
        n_iters=k + 1,
        converged=converged,
        kkt_final=kkt,
        wall_time_s=perf_counter() - t0,
        history=history,
    )
