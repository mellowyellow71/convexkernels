"""Nonnegative LASSO problem.

    min_x 0.5 ||Ax - b||^2 + lam * 1^T x
    s.t.  x >= 0

This is the first problem-family variant used to test transfer from the LASSO
kernel-search lineage. It keeps the same smooth gradient and Lipschitz
constant as LASSO, but swaps the nonsmooth prox and KKT residual.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from .problem import Problem


class NonnegativeLasso(Problem):
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
        return np.maximum(v - t * self.lam, 0.0)

    @cached_property
    def lambda_max(self) -> float:
        return float(max(np.max(self.rmatvec(self.b)), 0.0))

    @cached_property
    def L(self) -> float:
        return float(np.linalg.norm(self.A, ord=2) ** 2)

    def kkt_residual(self, x: np.ndarray) -> float:
        from ..algorithms.kkt import nonnegative_lasso_kkt_residual

        return nonnegative_lasso_kkt_residual(
            self.A,
            self.b,
            self.lam,
            x,
            L=self.L,
            lambda_max=self.lambda_max,
        )
