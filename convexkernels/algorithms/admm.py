"""ADMM (Alternating Direction Method of Multipliers) driver — Boyd 2011.

Solves problems of the form

    min  f(x) + g(z)  s.t.  x = z

via the iterates (Boyd §3.1):

    x_{k+1} = prox_{(1/rho) f}(z_k - u_k)         # x-update
    z_{k+1} = prox_{(1/rho) g}(x_{k+1} + u_k)     # z-update
    u_{k+1} = u_k + x_{k+1} - z_{k+1}             # multiplier update

Stopping (Boyd §3.3.1):

    primal_res = ||x - z||
    dual_res   = ||rho (z - z_prev)||

Specialization for LASSO (problem class `LassoAdmm`):
    f(x) = (1/2)||A x - b||^2   ->   prox_f(v, 1/rho) = (A^T A + rho I)^{-1} (A^T b + rho v)
    g(z) = lam ||z||_1          ->   prox_g(v, 1/rho) = soft(v, lam/rho)

The expensive (A^T A + rho I) Cholesky factor is built once in init_state and
reused. Same precompute-and-cache lever as FISTA-Gram and ALM-equality-QP,
but for a problem with explicit splitting and no extrapolation step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Optional

import numpy as np


@dataclass
class AdmmState:
    """Numpy reference state for ADMM."""
    x: Any
    z: Any
    u: Any
    rho: float
    factor: Any = None  # cached (A^T A + rho I) factor


@dataclass
class AdmmResult:
    x: Any
    z: Any
    u: Any
    n_iters: int
    converged: bool
    primal_res_final: float
    dual_res_final: float
    wall_time_s: float
    history: dict = field(default_factory=dict)


def _default_numpy_init(problem: Any, *, rho: float) -> AdmmState:
    n = int(problem.n)
    state = AdmmState(
        x=np.zeros(n), z=np.zeros(n), u=np.zeros(n),
        rho=float(rho), factor=None,
    )
    state.factor = problem.build_factor(state.rho)
    return state


def _numpy_admm_step(state: AdmmState, problem: Any) -> AdmmState:
    """One ADMM iteration in pure numpy.

    Calls into the problem's prox_f (the expensive linear solve, cached factor)
    and prox_g (cheap soft-threshold or similar).
    """
    if state.factor is None:
        factor = problem.build_factor(state.rho)
    else:
        factor = state.factor
    # x = prox_f(z - u, 1/rho) using the cached factor.
    rhs = problem.x_rhs_admm(state.z, state.u, state.rho)
    x_new = problem.solve_with_factor(factor, rhs)
    # z = prox_g(x + u, 1/rho)
    z_new = problem.prox_g(x_new + state.u, 1.0 / state.rho)
    # u += x - z
    u_new = state.u + x_new - z_new
    return AdmmState(x=x_new, z=z_new, u=u_new, rho=state.rho, factor=factor)


def admm(
    problem: Any,
    *,
    max_iters: int = 2000,
    tol: float = 1e-6,
    rho: Optional[float] = None,
    rho_adapt: bool = False,
    rho_balance: float = 10.0,
    rho_factor: float = 2.0,
    variant: Literal["basic", "adaptive"] = "basic",
    kernel_step: Callable[[AdmmState, Any], AdmmState] = _numpy_admm_step,
    kernel_init: Callable[..., AdmmState] = _default_numpy_init,
    record_history: bool = True,
    convergence_check_every: int = 1,
    callback: Optional[Callable[[int, AdmmState, float, float], None]] = None,
) -> AdmmResult:
    """Run ADMM on `problem` until both primal and dual residuals < tol.

    Variants:
      - ``basic``: constant rho, factor cached once, no rebalancing.
      - ``adaptive``: Boyd §3.4.1 adaptive rho with rebalancing. Factor is
        invalidated on rho changes; the kernel rebuilds it lazily.
    """
    if variant == "adaptive":
        rho_adapt = True

    # Time `kernel_init` as part of `wall_time_s` (timing-contract fix).
    # If rho is None, defer to the seed/init_state's default (typically a
    # problem-aware Boyd lambda_max heuristic).
    t0 = perf_counter()
    if rho is None:
        state = kernel_init(problem)
    else:
        state = kernel_init(problem, rho=rho)
    history: dict = {"primal_res": [], "dual_res": [], "wall_time": []} if record_history else {}
    primal_res = float("inf")
    dual_res = float("inf")
    converged = False
    k = -1
    check_every = max(1, int(convergence_check_every))

    z_prev = state.z

    for k in range(max_iters):
        state_new = kernel_step(state, problem)

        if (k + 1) % check_every == 0 or k == max_iters - 1:
            primal_res = float(np.linalg.norm(np.asarray(state_new.x - state_new.z)))
            dual_res = state.rho * float(
                np.linalg.norm(np.asarray(state_new.z - z_prev))
            )
            if record_history:
                history["primal_res"].append(primal_res)
                history["dual_res"].append(dual_res)
                history["wall_time"].append(perf_counter() - t0)
            if callback is not None:
                callback(k, state_new, primal_res, dual_res)
            if primal_res < tol and dual_res < tol:
                converged = True
                state = state_new
                break

            if rho_adapt:
                ratio = primal_res / max(dual_res, 1e-30)
                new_rho = state_new.rho
                if ratio > rho_balance:
                    new_rho = state_new.rho * rho_factor
                elif ratio < 1.0 / rho_balance:
                    new_rho = state_new.rho / rho_factor
                if new_rho != state_new.rho:
                    # Invalidate factor cache; kernel rebuilds lazily.
                    # u must be rescaled: u' = u * rho / new_rho (Boyd §3.4.1).
                    scale = state_new.rho / new_rho
                    state_new = type(state_new)(
                        x=state_new.x, z=state_new.z, u=state_new.u * scale,
                        rho=float(new_rho), factor=None,
                    )

        state = state_new
        z_prev = state_new.z

    final_x = state.x if isinstance(state.x, np.ndarray) else np.asarray(state.x)
    final_z = state.z if isinstance(state.z, np.ndarray) else np.asarray(state.z)
    final_u = state.u if isinstance(state.u, np.ndarray) else np.asarray(state.u)
    return AdmmResult(
        x=final_x, z=final_z, u=final_u,
        n_iters=k + 1,
        converged=converged,
        primal_res_final=primal_res,
        dual_res_final=dual_res,
        wall_time_s=perf_counter() - t0,
        history=history,
    )
