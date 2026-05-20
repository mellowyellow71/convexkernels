"""Batched-lambda LASSO frontend for the full-regularization-path pivot.

`LassoPath(A, b, lambdas)` is the path analog of `Lasso(A, b, lam)`. State
shape: x lives in R^{n x K} where each column is the solution for one lambda
in the path. Lambdas are stored in *decreasing* order (glmnet/Adelie
convention).

The Gram precompute G = A^T A is path-independent: it depends only on (A, b),
not on lambdas[k]. This is the amortization lever that makes batched
FISTA-Gram a single matrix-matrix problem instead of K matrix-vector problems.

Hardware fit on Apple Silicon M3 Pro: G @ Y is a single (n,n) x (n,K) gemm,
soft-threshold is a per-column elementwise op (lambdas broadcast across rows),
and the whole thing stays in unified memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class PreparedPath:
    """Precomputed quantities shared across the entire lambda path."""
    G: np.ndarray         # (n, n) = A.T @ A
    c: np.ndarray         # (n,)   = A.T @ b
    L: float              # spectral norm squared of A
    lambda_max: float     # ||A.T b||_inf (data-side scale, path-independent)


class LassoPath:
    def __init__(self, A: np.ndarray, b: np.ndarray, lambdas: np.ndarray):
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(
                f"b must be 1D with len = A.shape[0]={A.shape[0]}, got shape {b.shape}"
            )
        if lambdas.ndim != 1 or lambdas.shape[0] == 0:
            raise ValueError(
                f"lambdas must be 1D non-empty, got shape {lambdas.shape}"
            )
        if np.any(lambdas < 0):
            raise ValueError("all lambdas must be non-negative")
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def K(self) -> int:
        return int(self.lambdas.shape[0])

    @cached_property
    def lambda_max(self) -> float:
        """Smallest lambda for which the all-zero solution is optimal.

        Path-independent: depends only on (A, b). Used as the scale-free
        denominator in KKT residual computations.
        """
        return float(np.max(np.abs(self.A.T @ self.b)))

    @cached_property
    def L(self) -> float:
        """Spectral norm squared of A; the FISTA step size is 1/L."""
        return float(np.linalg.norm(self.A, ord=2) ** 2)

    @cached_property
    def prepared(self) -> PreparedPath:
        """Compute the Gram matrix and related quantities once for the whole path."""
        G = self.A.T @ self.A
        c = self.A.T @ self.b
        return PreparedPath(G=G, c=c, L=self.L, lambda_max=self.lambda_max)

    @property
    def default_rho(self) -> float:
        """Boyd 2011 lambda_max heuristic for ADMM. Path-independent (does
        not depend on lambdas[k])."""
        return float(self.lambda_max)

    def matvec_path(self, X: np.ndarray) -> np.ndarray:
        """A @ X where X is (n, K). Returns (m, K)."""
        return self.A @ X

    def rmatvec_path(self, Y: np.ndarray) -> np.ndarray:
        """A.T @ Y where Y is (m, K). Returns (n, K)."""
        return self.A.T @ Y

    def grad_smooth_path(self, X: np.ndarray) -> np.ndarray:
        """Per-column gradient of 0.5||AX - b||^2: A.T @ (A X - b).

        Returns (n, K). The b broadcast happens column-wise so each column's
        gradient is computed independently against the same data vector b.
        """
        return self.A.T @ (self.A @ X - self.b[:, None])

    def prox_path(self, V: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Per-column soft-threshold with per-column threshold.

        For column k, applies soft-threshold with parameter `t * lambdas[k]`
        (t is the same step size for all columns; lambda varies per column).
        Returns the proximal operator of t * lam * ||.||_1 column-wise.

        If t is a scalar, the per-column threshold is t * lambdas. If t is
        (K,), the threshold is t * lambdas elementwise.
        """
        if V.ndim != 2 or V.shape[1] != self.K:
            raise ValueError(
                f"V must be (n, K={self.K}), got shape {V.shape}"
            )
        t_arr = np.asarray(t, dtype=np.float64)
        if t_arr.ndim == 0:
            kappa = t_arr * self.lambdas  # (K,)
        elif t_arr.shape == (self.K,):
            kappa = t_arr * self.lambdas
        else:
            raise ValueError(
                f"t must be scalar or shape (K={self.K},), got shape {t_arr.shape}"
            )
        return np.sign(V) * np.maximum(np.abs(V) - kappa[None, :], 0.0)

    def kkt_residual(self, X: np.ndarray) -> np.ndarray:
        """Per-column scale-free KKT residual; shape (K,).

        Driver gates on `max(kkt_residual(X)) < tol`.
        """
        from ..algorithms.kkt_batched import lasso_kkt_residual_batched
        return lasso_kkt_residual_batched(
            self.A, self.b, self.lambdas, X,
            L=self.L, lambda_max=self.lambda_max,
        )

    def kkt_residual_max(self, X: np.ndarray) -> float:
        """Convenience: scalar max-over-columns of the per-column residual."""
        return float(np.max(self.kkt_residual(X)))
