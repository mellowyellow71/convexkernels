"""LASSO problem: min_x (1/2)||Ax - b||^2 + lam * ||x||_1.

Closed-form KKT residual (Beck & Teboulle 2009; Boyd et al. 2011 §6.4):
    g = A^T(Ax - b)
    v_i = |g_i + lam * sign(x_i)|    if x_i != 0
          max(|g_i| - lam, 0)         if x_i == 0
    KKT(x) = ||v||_inf / (lam + ||A^T b||_inf)

KKT(x) = 0 iff x is the LASSO optimum. Scale-free.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .problem import Problem


class Lasso(Problem):
    def __init__(self, A: np.ndarray, b: np.ndarray, lam: float):
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(
                f"b must be 1D with len = A.shape[0]={A.shape[0]}, got shape {b.shape}"
            )
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.lam = float(lam)

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        return self.A.T @ y

    def grad_smooth(self, x: np.ndarray) -> np.ndarray:
        return self.rmatvec(self.matvec(x) - self.b)

    def prox(self, v: np.ndarray, t: float) -> np.ndarray:
        kappa = t * self.lam
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    @cached_property
    def lambda_max(self) -> float:
        return float(np.max(np.abs(self.rmatvec(self.b))))

    @cached_property
    def L(self) -> float:
        return float(np.linalg.norm(self.A, ord=2) ** 2)

    def kkt_residual(self, x: np.ndarray) -> float:
        from ..algorithms.kkt import lasso_kkt_residual
        return lasso_kkt_residual(
            self.A, self.b, self.lam, x, L=self.L, lambda_max=self.lambda_max
        )
