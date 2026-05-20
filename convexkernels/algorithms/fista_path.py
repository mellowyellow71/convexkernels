"""Batched-lambda FISTA driver for full-regularization-path LASSO.

Host owns the outer loop, step size, and KKT-batched gating. Per-iter
compute is a swappable `kernel_step`. The driver is *backend-agnostic*:
it works with either the numpy host `LassoPath` (slow reference) or the
MLX `LassoPathGramMLX` view (fast path). Both must expose:

  problem.n            : int       (n features)
  problem.K            : int       (K lambdas)
  problem.L            : float     (Lipschitz constant of the smooth part)
  problem.dtype        : (optional, kernel-side only)
  problem.kkt_residual_max(X) -> float  (max-over-cols scale-free residual)

State shape: `FistaPathState(X, Y, theta)` where X, Y are (n, K) and
theta is (K,) per-column momentum.

Timing contract: `t0 = perf_counter()` is set BEFORE `kernel_init`. The
candidate cannot hide per-iter compute in init_state without paying for
it in the headline wall_ms (Sakana protection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Optional

import numpy as np


@dataclass
class FistaPathState:
    X: Any
    Y: Any
    theta: Any

    @property
    def K(self) -> int:
        return int(self.X.shape[1])


@dataclass
class FistaPathResult:
    x: np.ndarray                  # alias for X; numpy at the boundary
    X: np.ndarray
    n_iters: int
    converged: bool
    kkt_final: float               # scalar, max over columns; same name as fista.FistaResult
    kkt_max_final: float           # explicit alias
    kkt_per_col_final: np.ndarray  # (K,)
    wall_time_s: float
    history: dict = field(default_factory=dict)


def _numpy_path_init(problem) -> FistaPathState:
    n, K = problem.n, problem.K
    return FistaPathState(
        X=np.zeros((n, K), dtype=np.float64),
        Y=np.zeros((n, K), dtype=np.float64),
        theta=np.ones(K, dtype=np.float64),
    )


def _numpy_path_step(state: FistaPathState, problem, t: float) -> FistaPathState:
    """One vanilla-FISTA step on the path, using the Gram form.

    Expects the problem to expose `prepared.G`, `prepared.c` (path-independent
    Gram precompute on the host). For LassoPath this lives at
    `problem.prepared`; for kernel-side LassoPathGramMLX a different path is
    used (the MLX kernel overrides `kernel_step`).
    """
    prep = problem.prepared
    G, c, L = prep.G, prep.c, prep.L
    g = G @ state.Y - c[:, None]
    Z = state.Y - g / L
    X_new = np.sign(Z) * np.maximum(
        np.abs(Z) - problem.lambdas[None, :] / L, 0.0,
    )
    theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * state.theta * state.theta))
    diff_y = state.Y - X_new
    diff_x = X_new - state.X
    restart = (diff_y * diff_x).sum(axis=0) > 0
    theta_new = np.where(restart, 1.0, theta_new)
    mom = np.where(restart, 0.0, (state.theta - 1.0) / theta_new)
    Y_new = X_new + mom[None, :] * diff_x
    return FistaPathState(X=X_new, Y=Y_new, theta=theta_new)


def fista_path(
    problem,
    *,
    max_iters: int = 10000,
    tol: float = 1e-6,
    convergence_check_every: int = 10,
    variant: Literal["basic"] = "basic",
    kernel_step: Callable[[FistaPathState, Any, float], FistaPathState]
        = _numpy_path_step,
    kernel_init: Callable[[Any], FistaPathState] = _numpy_path_init,
    kernel_kkt_max: Optional[Callable[[FistaPathState, Any], float]] = None,
    record_history: bool = True,
) -> FistaPathResult:
    """Run batched FISTA on a LASSO path.

    `kernel_step`, `kernel_init`, `kernel_kkt_max` default to the numpy
    reference. MLX kernels override all three.

    `kernel_kkt_max(state, problem) -> float` short-circuits the host-side
    KKT computation when the kernel can compute it on-device cheaply. This
    is critical for performance: a per-iter device-to-host transfer would
    dwarf the actual work on path_wide_hero.
    """
    t0 = perf_counter()

    state = kernel_init(problem)
    t = 1.0 / problem.L

    history: dict = {"kkt": [], "wall_time": []} if record_history else {}
    converged = False
    k = -1
    kkt_max = float("inf")

    for k in range(max_iters):
        state = kernel_step(state, problem, t)
        if (k + 1) % convergence_check_every == 0:
            if kernel_kkt_max is not None:
                kkt_max = float(kernel_kkt_max(state, problem))
            else:
                kkt_max = float(problem.kkt_residual_max(state.X))
            if record_history:
                history["kkt"].append(kkt_max)
                history["wall_time"].append(perf_counter() - t0)
            if kkt_max < tol:
                converged = True
                break

    X_final = state.X if isinstance(state.X, np.ndarray) else np.asarray(state.X)
    # Per-column residual at the end, via numpy KKT (independent of kernel).
    # This is cheap (one matmul on host) and gives the eval the per-column
    # info even when the kernel only returned the max.
    if hasattr(problem, "kkt_residual"):
        kkt_per_col = np.asarray(problem.kkt_residual(X_final))
        if kkt_per_col.ndim == 0:
            kkt_per_col = np.array([float(kkt_per_col)])
    elif hasattr(problem, "kkt_residual_per_col"):
        kkt_per_col = np.asarray(problem.kkt_residual_per_col(state.X))
    else:
        kkt_per_col = np.array([kkt_max])
    kkt_max_final = float(kkt_per_col.max())
    return FistaPathResult(
        x=X_final,
        X=X_final,
        n_iters=k + 1,
        converged=converged,
        kkt_final=kkt_max_final,
        kkt_max_final=kkt_max_final,
        kkt_per_col_final=kkt_per_col,
        wall_time_s=perf_counter() - t0,
        history=history,
    )
