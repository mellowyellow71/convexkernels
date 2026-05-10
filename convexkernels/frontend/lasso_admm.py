"""LASSO via ADMM with x = z splitting.

  min  (1/2)||A x - b||^2 + lam ||z||_1   s.t.  x = z

Provides the ADMM-specific interface on top of the existing `Lasso` data
(A, b, lam):

  build_factor(rho)        -> Cholesky of (A^T A + rho I)
  solve_with_factor(F, v)  -> (A^T A + rho I)^{-1} v
  x_rhs_admm(z, u, rho)    -> A^T b + rho (z - u)
  prox_g(v, t)             -> soft-threshold(v, t * lam)

The factor is rebuilt only on rho changes. For tall A (m > n) the factor is
n x n; for wide A (n > m) the matrix-inversion lemma form would be cheaper
(m x m factor) but that variant is left for the autoresearch loop to
discover.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .lasso import Lasso


class LassoAdmm:
    """LASSO problem viewed through the ADMM x=z splitting."""

    def __init__(self, A: np.ndarray, b: np.ndarray, lam: float):
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got {A.shape}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(f"b must be length {A.shape[0]}, got {b.shape}")
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.lam = float(lam)

    @classmethod
    def from_lasso(cls, lasso: Lasso) -> "LassoAdmm":
        return cls(lasso.A, lasso.b, lasso.lam)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @cached_property
    def A_T_b(self) -> np.ndarray:
        """Cached A^T b — used in every x-update."""
        return self.A.T @ self.b

    def x_rhs_admm(self, z: np.ndarray, u: np.ndarray, rho: float) -> np.ndarray:
        """RHS of the ADMM x-update: A^T b + rho (z - u)."""
        return self.A_T_b + rho * (z - u)

    def build_factor(self, rho: float) -> dict:
        """Cholesky of (A^T A + rho I). Cached once per rho."""
        H = self.A.T @ self.A + rho * np.eye(self.n)
        try:
            from scipy.linalg import cho_factor
            c, lower = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
            return {"cho_factor": (c, lower), "kind": "scipy_cho_factor"}
        except ImportError:
            L = np.linalg.cholesky(H)
            return {"L": L, "kind": "numpy_cholesky"}

    def solve_with_factor(self, factor: dict, rhs: np.ndarray) -> np.ndarray:
        if factor["kind"] == "scipy_cho_factor":
            from scipy.linalg import cho_solve
            return cho_solve(factor["cho_factor"], rhs, check_finite=False)
        L = factor["L"]
        try:
            from scipy.linalg import solve_triangular
            y = solve_triangular(L, rhs, lower=True, check_finite=False)
            return solve_triangular(L.T, y, lower=False, check_finite=False)
        except ImportError:
            return np.linalg.solve(L @ L.T, rhs)

    def prox_g(self, v: np.ndarray, t: float) -> np.ndarray:
        """Soft-thresholding: prox of t * lam * ||z||_1 at v."""
        kappa = t * self.lam
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    def primal_objective(self, x: np.ndarray) -> float:
        residual = self.A @ x - self.b
        return 0.5 * float(np.dot(residual, residual)) + self.lam * float(np.sum(np.abs(x)))

    def kkt_residual(self, x: np.ndarray) -> float:
        """KKT residual reusing the canonical Lasso formulation."""
        from ..algorithms.kkt import lasso_kkt_residual
        L = float(np.linalg.norm(self.A, ord=2)) ** 2
        return lasso_kkt_residual(self.A, self.b, self.lam, x, L=L)
