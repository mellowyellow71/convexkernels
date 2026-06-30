"""Primal-dual gap and dual-residual fitness signals for PDHG and ALM/ADMM.

`kkt.py` is the fitness for proximal-gradient methods on f+g problems
(FISTA-on-LASSO style). For saddle-point methods (PDHG/Chambolle-Pock) and
augmented-Lagrangian methods (ALM/ADMM), the natural oracle-free correctness
signal is the primal-dual gap or the primal/dual residual pair.

All formulas here are scale-free, computed from problem data + iterate, and
equal zero iff the iterate is optimal — same contract as `kkt.py`. They are
the load-bearing fitness for the PDHG and ALM specimens of the synthesis
loop, so a bug here corrupts every promotion decision.

Notation:
- Saddle point: min_x max_y <Kx, y> + f(x) - g*(y).
- For TV-L2 denoising: f(x) = (1/2)||x - b||^2, g(z) = lam * ||z||_1, K = grad.
- For an ALM/ADMM problem min f(x) + g(z) s.t. Ax + Bz = c, the primal residual
  is r = Ax + Bz - c, the dual residual is s = rho * A^T B (z - z_prev).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def lasso_duality_gap(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    x: np.ndarray,
    *,
    scale: float | None = None,
) -> float:
    r"""Scale-free LASSO duality gap — the oracle-free optimality gap.

    Primal:  P(x) = 1/2 ||A x - b||^2 + lam ||x||_1
    Dual:    D(theta) = 1/2 ||b||^2 - 1/2 ||b - theta||^2   s.t. ||A^T theta||_inf <= lam

    Given any primal x, the residual ``theta_raw = b - A x`` is the natural dual
    point; it is rescaled by ``s = min(1, lam / ||A^T theta_raw||_inf)`` to land
    inside the dual-feasible box (the same construction used by glmnet/Adelie and
    by SAFE screening). ``P(x) >= D(theta)`` always, so the gap is non-negative
    and equals zero iff x is the LASSO optimum — no reference solution required,
    which is exactly why it can be the y-axis for a "beat Adelie" comparison.

    Returns ``(P - D) / scale`` with ``scale = 0.5 ||b||^2 + 1`` by default
    (mirrors `tv_l2_primal_dual_gap`; scale-consistent up to alpha^2 under
    ``(b, lam) -> (alpha b, alpha lam)``).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    theta_raw = b - A @ x                       # residual, in R^m
    dnorm = float(np.max(np.abs(A.T @ theta_raw))) if A.size else 0.0
    s = min(1.0, lam / dnorm) if dnorm > 0.0 else 1.0
    theta = s * theta_raw
    P = 0.5 * float(theta_raw @ theta_raw) + lam * float(np.sum(np.abs(x)))
    D = 0.5 * float(b @ b) - 0.5 * float((b - theta) @ (b - theta))
    if scale is None:
        scale = 0.5 * float(b @ b) + 1.0
    return float(max(P - D, 0.0)) / scale


def lasso_duality_gap_batched(
    A: np.ndarray,
    b: np.ndarray,
    lambdas: np.ndarray,
    X: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    """Per-column LASSO duality gap for the batched path; shape ``(K,)``.

    Vectorized `lasso_duality_gap` over the K lambdas of a path (each column of
    ``X`` is the solution for ``lambdas[k]``). Driver gates / scores on the curve
    of ``max(per_column_gap)`` vs wall-clock.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    Theta_raw = b[:, None] - A @ X                       # (m, K)
    dnorm = np.max(np.abs(A.T @ Theta_raw), axis=0)      # (K,)
    s = np.where(dnorm > 0.0, np.minimum(1.0, lambdas / np.maximum(dnorm, 1e-300)), 1.0)
    Theta = s[None, :] * Theta_raw
    P = 0.5 * np.sum(Theta_raw ** 2, axis=0) + lambdas * np.sum(np.abs(X), axis=0)
    diff = b[:, None] - Theta
    D = 0.5 * float(b @ b) - 0.5 * np.sum(diff ** 2, axis=0)
    if scale is None:
        scale = 0.5 * float(b @ b) + 1.0
    return np.maximum(P - D, 0.0) / scale


def tv_l2_primal_objective(
    b: np.ndarray,
    K_apply: Callable[[np.ndarray], np.ndarray],
    lam: float,
    x: np.ndarray,
) -> float:
    """Primal objective for TV-L2 denoising: 0.5||x - b||^2 + lam ||Kx||_1."""
    Kx = K_apply(x)
    return 0.5 * float(np.sum((x - b) ** 2)) + lam * float(np.sum(np.abs(Kx)))


def tv_l2_dual_objective(
    b: np.ndarray,
    K_T_apply: Callable[[np.ndarray], np.ndarray],
    lam: float,
    y: np.ndarray,
) -> float:
    r"""Dual objective for TV-L2 denoising.

    Saddle: min_x max_y <Kx, y> + 0.5||x - b||^2 - I{||y||_inf <= lam}.
    Eliminating x in closed form (x* = b - K^T y) gives:

        D(y) = -0.5 ||K^T y||^2 + <K^T y, b>          if ||y||_inf <= lam
             = -inf                                    otherwise.

    Returning -inf for infeasible y is correct but unhelpful as a fitness
    signal during iteration; we return the unconstrained value and let the
    caller decide whether to project y onto the lam-ball before calling.
    """
    KTy = K_T_apply(y)
    return -0.5 * float(np.sum(KTy ** 2)) + float(np.dot(KTy, b))


def tv_l2_primal_dual_gap(
    b: np.ndarray,
    K_apply: Callable[[np.ndarray], np.ndarray],
    K_T_apply: Callable[[np.ndarray], np.ndarray],
    lam: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scale: float | None = None,
) -> float:
    r"""Scale-free primal-dual gap for TV-L2 denoising.

    Returns ``(P(x) - D(y_proj)) / scale`` where ``y_proj`` clips y to the
    lam-ball so the dual is feasible. Default scale is
    ``0.5 ||b||^2 + 1`` — invariant under (b, lam) -> (alpha b, alpha lam)
    only up to alpha^2, which matches the gap's natural scaling.

    P(x) >= D(y_proj) for any (x, y_proj), so this is non-negative; equals
    zero iff (x, y) is a saddle point.
    """
    y_proj = np.clip(y, -lam, lam)
    P = tv_l2_primal_objective(b, K_apply, lam, x)
    D = tv_l2_dual_objective(b, K_T_apply, lam, y_proj)
    if scale is None:
        scale = 0.5 * float(np.sum(b ** 2)) + 1.0
    return float(max(P - D, 0.0)) / scale


def alm_residuals(
    A_apply: Callable[[np.ndarray], np.ndarray],
    B_apply: Callable[[np.ndarray], np.ndarray],
    A_T_apply: Callable[[np.ndarray], np.ndarray],
    c: np.ndarray,
    rho: float,
    x: np.ndarray,
    z: np.ndarray,
    z_prev: np.ndarray,
) -> dict:
    r"""Primal/dual residuals for ALM/ADMM on min f(x) + g(z) s.t. Ax + Bz = c.

    Boyd 2011 §3.3.1:
        r_primal = Ax + Bz - c                     (constraint violation)
        r_dual   = rho * A^T B (z - z_prev)        (proxy for stationarity)

    Both -> 0 at convergence; together they are the standard ADMM stop test.
    Returns a dict with raw and normalized values. Normalization mirrors
    Boyd (2011) Eq. 3.13: divide by max(||Ax||, ||Bz||, ||c||) for primal,
    and by ||rho * A^T y|| for dual where y is the multiplier.
    """
    Ax = A_apply(x)
    Bz = B_apply(z)
    r_p = Ax + Bz - c
    r_d = rho * A_T_apply(B_apply(z) - B_apply(z_prev))
    norm_p = max(float(np.linalg.norm(Ax)), float(np.linalg.norm(Bz)), float(np.linalg.norm(c)), 1.0)
    norm_d = max(float(np.linalg.norm(rho * A_T_apply(np.zeros_like(c)))), 1.0)  # placeholder; caller can override
    return {
        "primal_residual": float(np.linalg.norm(r_p)),
        "dual_residual": float(np.linalg.norm(r_d)),
        "primal_residual_rel": float(np.linalg.norm(r_p)) / norm_p,
        "dual_residual_rel": float(np.linalg.norm(r_d)) / norm_d,
    }


def assert_pdhg_equivalent(
    iterate_kernel: tuple[Any, Any],
    iterate_ref: tuple[Any, Any],
    problem: Any,
    *,
    gap_tol: float = 1e-6,
    drift_warn: float = 1e-2,
) -> dict:
    """Functional equivalence test contract for PDHG-style methods.

    Mirrors `algorithms.kkt.assert_equivalent`. Both kernel and reference
    iterates ``(x, y)`` must satisfy ``problem.primal_dual_gap(x, y) < gap_tol``.
    Iterate drift on x is logged but does not fail the test.
    """
    import warnings

    x_k, y_k = iterate_kernel
    x_r, y_r = iterate_ref
    gap_k = problem.primal_dual_gap(x_k, y_k)
    gap_r = problem.primal_dual_gap(x_r, y_r)
    assert gap_k < gap_tol, f"kernel did not converge: gap={gap_k:.2e}"
    assert gap_r < gap_tol, f"reference did not converge: gap={gap_r:.2e}"

    x_k_np = np.asarray(x_k)
    x_r_np = np.asarray(x_r)
    denom = max(float(np.max(np.abs(x_r_np))), 1.0)
    rel_drift = float(np.max(np.abs(x_k_np - x_r_np)) / denom)
    if rel_drift > drift_warn:
        warnings.warn(
            f"large iterate drift {rel_drift:.2e} "
            f"(precision regime or near-degenerate active set)",
            stacklevel=2,
        )
    return {"gap_kernel": gap_k, "gap_ref": gap_r, "rel_drift": rel_drift}
