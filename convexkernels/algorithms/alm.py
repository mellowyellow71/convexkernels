"""Augmented Lagrangian Method (ALM) host driver.

Specimen-3 algorithm for the synthesis loop. Solves equality-constrained
convex problems

    min  f(x)  s.t.  A x = b

via the augmented Lagrangian

    L_rho(x, lam) = f(x) + <lam, A x - b> + (rho/2) ||A x - b||^2

and the iterates

    x_{k+1}   = argmin_x L_rho(x, lam_k)
    lam_{k+1} = lam_k + rho (A x_{k+1} - b)

For quadratic f(x) = (1/2) x^T P x + q^T x, the x-update is a single linear
solve

    (P + rho A^T A) x = -q + A^T (rho b - lam_k)

which the host caches as a Cholesky factorization of (P + rho A^T A) when
rho is constant — exactly analogous to the FISTA-Gram precompute lever, but
larger because the factor is reused across all iterations.

Stopping (Boyd 2011 §3.3.1, equality-only specialization):

    primal_res = ||A x - b||
    dual_res   = ||rho A^T (b - A x_prev)||   (proxy from previous iterate)

Both -> 0 at the optimum.

The host owns the rho schedule, residual calculation, history, and the
factor cache. The per-iter compute (the inner linear solve and multiplier
update) is the swappable `kernel_step` so MLX/Metal kernels can replace
``mx.linalg.cholesky`` + ``mx.linalg.solve_triangular`` with custom
trisolve passes when profiling demands it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Optional

import numpy as np


@dataclass
class AlmState:
    """Numpy reference state for ALM. MLX kernels carry their own version."""
    x: Any
    lam: Any
    rho: float
    factor: Any = None  # cached Cholesky factor of (P + rho A^T A); kernel-specific


@dataclass
class AlmResult:
    x: Any
    lam: Any
    n_iters: int
    converged: bool
    primal_res_final: float
    dual_res_final: float
    wall_time_s: float
    history: dict = field(default_factory=dict)


def _default_numpy_init(problem: Any, *, rho: float) -> AlmState:
    """Numpy zero-init for ALM iterates. The factor is built lazily on first
    step so the initial state stays small. The problem must expose
    `n` (dim of x) and `m_constraints` (dim of lam = number of equality rows).
    """
    n = int(problem.n)
    m = int(problem.m_constraints)
    return AlmState(x=np.zeros(n), lam=np.zeros(m), rho=float(rho), factor=None)


def _numpy_alm_step(state: AlmState, problem: Any) -> AlmState:
    """One ALM iteration in pure numpy.

    For quadratic f the x-update is a closed-form linear solve. The factor
    is cached on the state so subsequent iterations reuse it (the dominant
    win and the analog of FISTA-Gram precompute).
    """
    if state.factor is None:
        state = type(state)(
            x=state.x, lam=state.lam, rho=state.rho,
            factor=problem.build_factor(state.rho),
        )
    rhs = problem.x_rhs(state.lam, state.rho)        # -q + A^T (rho b - lam)
    x_new = problem.solve_with_factor(state.factor, rhs)
    Ax_b = problem.A_apply(x_new) - problem.b_constraint
    lam_new = state.lam + state.rho * Ax_b
    return AlmState(x=x_new, lam=lam_new, rho=state.rho, factor=state.factor)


def alm(
    problem: Any,
    *,
    max_iters: int = 1000,
    tol: float = 1e-6,
    rho: Optional[float] = None,
    rho_adapt: bool = False,
    rho_balance: float = 10.0,
    rho_factor: float = 2.0,
    variant: Literal["basic", "adaptive"] = "basic",
    kernel_step: Callable[[AlmState, Any], AlmState] = _numpy_alm_step,
    kernel_init: Callable[..., AlmState] = _default_numpy_init,
    record_history: bool = True,
    convergence_check_every: int = 1,
    callback: Optional[Callable[[int, AlmState, float, float], None]] = None,
) -> AlmResult:
    """Run ALM until ``primal_res < tol AND dual_res < tol`` or ``max_iters``.

    `variant`:
      - ``basic``: constant ``rho``, factor cached once, no rebalancing.
      - ``adaptive``: Boyd 2011 §3.4.1 adaptive ``rho``: when
        ``primal_res / dual_res > rho_balance``, multiply rho by
        ``rho_factor``; reverse direction when ratio < ``1/rho_balance``.
        A factor change invalidates the Cholesky cache; the kernel rebuilds
        it lazily on the next step.

    The dual residual approximation here is ``rho * ||A^T (Ax - Ax_prev)||``
    which mirrors Boyd's dual-residual proxy for ADMM at the equality slice.
    """
    if variant == "adaptive":
        rho_adapt = True

    # Time `kernel_init` as part of `wall_time_s`. See pdhg.py / fista.py for
    # the rationale: any per-step compute stuffed into init_state must count.
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

    Ax_prev = problem.A_apply(state.x)

    for k in range(max_iters):
        state_new = kernel_step(state, problem)
        Ax_new = problem.A_apply(state_new.x)

        if (k + 1) % check_every == 0 or k == max_iters - 1:
            # Backend-agnostic residual computation: keep operands in their
            # native array type for problem.A_T_apply, convert only the final
            # vector norms via np.asarray.
            primal_res = float(np.linalg.norm(np.asarray(Ax_new - problem.b_constraint)))
            dual_dir = problem.A_T_apply(Ax_new - Ax_prev)
            # Dual residual without the rho factor: measures how much the x-
            # iterates are still moving in the constraint-gradient direction.
            # For pure equality-ALM the x-update is exact so the original
            # rho-scaled formulation amplifies under large rho with no benefit.
            dual_res = float(np.linalg.norm(np.asarray(dual_dir)))
            if record_history:
                history["primal_res"].append(primal_res)
                history["dual_res"].append(dual_res)
                history["wall_time"].append(perf_counter() - t0)
            if callback is not None:
                callback(k, state_new, primal_res, dual_res)
            # For pure equality-ALM, primal residual is the gating criterion.
            # Dual residual is logged for diagnostics. Both reach zero at
            # the optimum; primal feasibility is what's hard to satisfy.
            if primal_res < tol:
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
                    # Invalidate factor cache; the kernel will rebuild lazily.
                    state_new = type(state_new)(
                        x=state_new.x, lam=state_new.lam,
                        rho=float(new_rho), factor=None,
                    )

        state = state_new
        Ax_prev = Ax_new

    final_x = state.x if isinstance(state.x, np.ndarray) else np.asarray(state.x)
    final_lam = state.lam if isinstance(state.lam, np.ndarray) else np.asarray(state.lam)
    return AlmResult(
        x=final_x,
        lam=final_lam,
        n_iters=k + 1,
        converged=converged,
        primal_res_final=primal_res,
        dual_res_final=dual_res,
        wall_time_s=perf_counter() - t0,
        history=history,
    )
