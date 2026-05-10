"""KKT residual implementations + functional-equivalence test contract.

Each `Problem` subclass owns a `kkt_residual(x)` method, but the actual math
lives here as free functions for testability and reuse from contexts where a
`Problem` object isn't available.

The fitness function for the synthesis loop. A bug here corrupts every
promotion decision, so this module is in the critical-files list.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def lasso_kkt_residual(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    x: np.ndarray,
    *,
    L: float | None = None,
    lambda_max: float | None = None,
) -> float:
    r"""Scale-free KKT residual for LASSO via the fixed-point reformulation.

    The textbook stationarity condition for LASSO ``min 0.5||Ax - b||^2 + lam ||x||_1``
    is the case-split:

        v_i = |g_i + lam * sign(x_i)|       if x_i != 0
              max(|g_i| - lam, 0)           if x_i == 0

    where g = A^T(Ax - b). The case-split is mathematically clean but breaks under
    floating-point: solver outputs have entries like 1e-12 that should count as zero
    but pass `x != 0`.

    The equivalent (and standard) prox-residual is robust:

        r = L*x - soft(L*x - g, lam)
        KKT(x) = ||r||_inf / (lam + ||A^T b||_inf)

    where soft(z, t) = sign(z) * max(|z| - t, 0). This is zero iff x is optimal,
    matches the case-split formula at non-degenerate points, and degrades gracefully
    when |x_i| is below the solver noise floor (the residual scales with L * |x_i|
    rather than producing a spurious O(lam) violation).

    The normalization makes the residual invariant under (A, b, lam) -> (alpha*A,
    alpha*b, alpha^2*lam).
    """
    g = A.T @ (A @ x - b)
    if L is None:
        L = float(np.linalg.norm(A, ord=2)) ** 2
    z = L * x - g
    soft_z = np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)
    r = L * x - soft_z
    if lambda_max is None:
        lambda_max = float(np.max(np.abs(A.T @ b)))
    denom = lam + lambda_max
    if denom == 0.0:
        return float(np.max(np.abs(r)))
    return float(np.max(np.abs(r)) / denom)


def nonnegative_lasso_kkt_residual(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    x: np.ndarray,
    *,
    L: float | None = None,
    lambda_max: float | None = None,
) -> float:
    r"""Scale-free KKT residual for nonnegative LASSO.

    Problem:

        min_x 0.5||Ax - b||^2 + lam * 1^T x
        s.t.  x >= 0

    KKT conditions are:

        x_i >= 0
        g_i + lam >= 0
        x_i * (g_i + lam) = 0

    The prox fixed-point form is more robust numerically:

        x = max(x - g/L - lam/L, 0)

    Multiplying by L gives the residual used here:

        r = L*x - max(L*x - g - lam, 0)
    """
    g = A.T @ (A @ x - b)
    if L is None:
        L = float(np.linalg.norm(A, ord=2)) ** 2
    z = L * x - g
    prox_z = np.maximum(z - lam, 0.0)
    r = L * x - prox_z
    if lambda_max is None:
        lambda_max = float(max(np.max(A.T @ b), 0.0))
    denom = lam + max(lambda_max, float(np.max(np.abs(A.T @ b))))
    if denom == 0.0:
        return float(np.max(np.abs(r)))
    return float(np.max(np.abs(r)) / denom)


def assert_equivalent(
    x_kernel: Any,
    x_ref: Any,
    problem: Any,
    *,
    kkt_tol: float = 1e-6,
    drift_warn: float = 1e-2,
) -> dict:
    """Functional equivalence test contract.

    Both `x_kernel` and `x_ref` must satisfy ``problem.kkt_residual(.) < kkt_tol``.
    Iterate drift ``||x_kernel - x_ref||_inf / max(||x_ref||_inf, 1)`` is
    computed and warned-on (above `drift_warn`), but does NOT fail the test.

    Rationale (see `tasks/todo.md` "Interface contracts"): low-precision
    kernels can land at slightly different points within the KKT-optimal set
    yet remain optimal. Numerical thresholds spuriously fail those. KKT-based
    gating is precision-agnostic and uses the same epsilon as the synthesis
    loop's convergence test.

    Inputs may be numpy arrays or any array type whose subtraction returns
    something supporting ``np.asarray``. The kernel/ref iterates are
    converted to numpy here for the drift calculation.
    """
    kkt_k = problem.kkt_residual(x_kernel)
    kkt_r = problem.kkt_residual(x_ref)
    assert kkt_k < kkt_tol, f"kernel did not converge: KKT={kkt_k:.2e}"
    assert kkt_r < kkt_tol, f"reference did not converge: KKT={kkt_r:.2e}"

    x_k_np = np.asarray(x_kernel)
    x_r_np = np.asarray(x_ref)
    denom = max(float(np.max(np.abs(x_r_np))), 1.0)
    rel_drift = float(np.max(np.abs(x_k_np - x_r_np)) / denom)
    if rel_drift > drift_warn:
        warnings.warn(
            f"large iterate drift {rel_drift:.2e} "
            f"(precision regime or near-degenerate active set)",
            stacklevel=2,
        )
    return {"kkt_kernel": kkt_k, "kkt_ref": kkt_r, "rel_drift": rel_drift}
