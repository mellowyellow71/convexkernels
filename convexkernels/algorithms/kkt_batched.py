"""Batched (per-column) LASSO KKT residual.

Vectorized version of `lasso_kkt_residual` over K lambdas. Each column of
X corresponds to a different lambda; the residual is computed independently
per column with the same scale-free prox-fixed-point formulation as the
scalar version. The driver gates on `max(per_column_residual) < tol`.

Sister to `algorithms/kkt.py`. Critical-files list — bug here corrupts
every gating decision in the path-batched loop.
"""

from __future__ import annotations

import numpy as np


def lasso_kkt_residual_batched(
    A: np.ndarray,
    b: np.ndarray,
    lambdas: np.ndarray,
    X: np.ndarray,
    *,
    L: float | None = None,
    lambda_max: float | None = None,
) -> np.ndarray:
    r"""Per-column scale-free KKT residual for batched LASSO.

    For X of shape (p, K) and lambdas of shape (K,), returns a (K,)-vector
    of residuals. Per-column formulation matches `lasso_kkt_residual`:

        g[:, k] = A^T (A X[:, k] - b)
        z[:, k] = L * X[:, k] - g[:, k]
        soft_z[:, k] = sign(z[:, k]) * max(|z[:, k]| - lambdas[k], 0)
        r[:, k] = L * X[:, k] - soft_z[:, k]
        KKT[k] = ||r[:, k]||_inf / (lambdas[k] + ||A^T b||_inf)

    `lambda_max` is the *data-side* scale ||A^T b||_inf, shared across the
    path (depends on (A, b) only, not on lambdas[k]).

    Implementation note: AX is one (m, K) matmul; A^T (AX - b) is one
    (p, K) matmul. This is the same matrix-matrix structure the batched
    FISTA-Gram solver exploits.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (p, K), got shape {X.shape}")
    if lambdas.ndim != 1 or lambdas.shape[0] != X.shape[1]:
        raise ValueError(
            f"lambdas must be 1D with len = X.shape[1]={X.shape[1]}, "
            f"got shape {lambdas.shape}"
        )
    if L is None:
        L = float(np.linalg.norm(A, ord=2)) ** 2
    if lambda_max is None:
        lambda_max = float(np.max(np.abs(A.T @ b)))

    # g shape (p, K). AX - b[:, None] broadcasts. b is (m,), AX is (m, K).
    AX = A @ X
    g = A.T @ (AX - b[:, None])
    z = L * X - g
    soft_z = np.sign(z) * np.maximum(np.abs(z) - lambdas[None, :], 0.0)
    r = L * X - soft_z
    # Per-column inf-norm.
    r_inf = np.max(np.abs(r), axis=0)
    denom = lambdas + lambda_max
    # Where denom == 0 (only if lam=0 AND lambda_max=0), fall back to bare r_inf.
    out = np.where(denom > 0.0, r_inf / np.maximum(denom, 1e-300), r_inf)
    return out.astype(np.float64, copy=False)
