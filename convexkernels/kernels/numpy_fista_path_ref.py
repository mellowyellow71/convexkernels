"""Numpy sequential FISTA over a lambda path. Correctness oracle.

For each lambda in the path (high-to-low), runs FISTA-Gram with warm-start
from the previous lambda's solution. This is the standard glmnet/Adelie
strategy: warm-starting along the path is dramatically cheaper than
cold-starting at each lambda.

No speed target. This solver exists to (a) verify Adelie agrees with our
KKT formulation across the path and (b) provide a known-good X_path to
seed test fixtures.

Uses the Gram form `G y - c` for the gradient so the same precompute the
batched MLX solver will share is exercised here at single-column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..algorithms.kkt import lasso_kkt_residual
from ..frontend.lasso_path import LassoPath


@dataclass
class FistaPathResult:
    X: np.ndarray         # (n, K)
    kkt_per_lambda: np.ndarray  # (K,)
    iters_per_lambda: np.ndarray  # (K,) ints
    converged_per_lambda: np.ndarray  # (K,) bools


def fista_path_numpy(
    prob: LassoPath,
    *,
    max_iters: int = 10000,
    tol: float = 1e-6,
    convergence_check_every: int = 10,
) -> FistaPathResult:
    """Solve LASSO at each lambda in `prob.lambdas`, warm-starting along the path.

    Iterates from the largest lambda (where the all-zero solution is at or
    near optimal) down to the smallest. For each lambda, runs FISTA-Gram
    with restart and warm-starts the next lambda from the current solution.
    """
    n = prob.n
    K = prob.K
    prep = prob.prepared
    G, c, L = prep.G, prep.c, prep.L
    lambda_max_data = prep.lambda_max

    X = np.zeros((n, K), dtype=np.float64)
    kkt_out = np.zeros(K, dtype=np.float64)
    iters_out = np.zeros(K, dtype=np.int64)
    conv_out = np.zeros(K, dtype=bool)

    x = np.zeros(n, dtype=np.float64)  # warm-start

    inv_L = 1.0 / L

    for k in range(K):
        lam = float(prob.lambdas[k])
        # FISTA-Gram with restart, warm-started at x.
        y = x.copy()
        x_prev = x.copy()
        theta = 1.0
        prev_obj = float("inf")
        converged = False
        iters = max_iters
        for it in range(max_iters):
            g = G @ y - c
            x_new = np.sign(y - inv_L * g) * np.maximum(
                np.abs(y - inv_L * g) - lam * inv_L, 0.0
            )
            theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
            y = x_new + ((theta - 1.0) / theta_new) * (x_new - x_prev)
            # Gradient-restart (O'Donoghue & Candes 2012): if momentum points
            # uphill in the smooth objective, reset.
            obj = 0.5 * float(x_new @ (G @ x_new)) - float(c @ x_new) \
                + lam * float(np.sum(np.abs(x_new)))
            if obj > prev_obj:
                y = x_new.copy()
                theta_new = 1.0
            prev_obj = obj
            x_prev = x_new
            theta = theta_new

            if (it + 1) % convergence_check_every == 0:
                kkt = lasso_kkt_residual(
                    prob.A, prob.b, lam, x_new,
                    L=L, lambda_max=lambda_max_data,
                )
                if kkt < tol:
                    iters = it + 1
                    converged = True
                    break

        x = x_new
        X[:, k] = x
        iters_out[k] = iters
        conv_out[k] = converged
        kkt_out[k] = lasso_kkt_residual(
            prob.A, prob.b, lam, x,
            L=L, lambda_max=lambda_max_data,
        )

    return FistaPathResult(
        X=X, kkt_per_lambda=kkt_out,
        iters_per_lambda=iters_out, converged_per_lambda=conv_out,
    )
