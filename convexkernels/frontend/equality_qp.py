"""Equality-constrained convex QP: min (1/2) x^T P x + q^T x  s.t.  A x = b.

Specimen-3 frontend, the canonical ALM target. P is symmetric positive
semi-definite; the augmented Hessian (P + rho A^T A) is symmetric positive
definite for any rho > 0 (assuming A has full row rank), so Cholesky is the
natural inner solver.

The class exposes the operations the ALM driver needs:

- ``A_apply(x) -> A x``           (forward op)
- ``A_T_apply(y) -> A^T y``       (adjoint)
- ``b_constraint``                 the equality RHS
- ``x_rhs(lam, rho)``              -q + A^T (rho b - lam)
- ``build_factor(rho)``            cached Cholesky of (P + rho A^T A)
- ``solve_with_factor(factor, rhs)`` triangular solve(s)

`n` is the primal dim; `m_constraints` is the dual dim (number of equality
rows). Both are required by the ALM kernel_init.
"""

from __future__ import annotations

from functools import cached_property
from typing import Tuple

import numpy as np


class EqualityQP:
    """min (1/2) x^T P x + q^T x  s.t.  A x = b.

    `P` is a symmetric PSD matrix (n x n).
    `q` is a length-n vector.
    `A` is an m x n matrix (full row rank).
    `b` is a length-m vector.
    """

    def __init__(self, P: np.ndarray, q: np.ndarray, A: np.ndarray, b: np.ndarray):
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P must be square, got {P.shape}")
        if q.ndim != 1 or q.shape[0] != P.shape[0]:
            raise ValueError(f"q must be 1D length {P.shape[0]}, got {q.shape}")
        if A.ndim != 2 or A.shape[1] != P.shape[0]:
            raise ValueError(f"A must be m x {P.shape[0]}, got {A.shape}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(f"b must be length {A.shape[0]}, got {b.shape}")
        self.P = np.ascontiguousarray(P, dtype=np.float64)
        self.q = np.ascontiguousarray(q, dtype=np.float64)
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.b_constraint = np.ascontiguousarray(b, dtype=np.float64)

    @property
    def n(self) -> int:
        return int(self.P.shape[0])

    @property
    def m_constraints(self) -> int:
        return int(self.A.shape[0])

    def A_apply(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def A_T_apply(self, y: np.ndarray) -> np.ndarray:
        return self.A.T @ y

    def primal_objective(self, x: np.ndarray) -> float:
        return 0.5 * float(x @ self.P @ x) + float(self.q @ x)

    def x_rhs(self, lam: np.ndarray, rho: float) -> np.ndarray:
        """RHS of the augmented x-subproblem: -q + A^T (rho b - lam)."""
        return -self.q + self.A.T @ (rho * self.b_constraint - lam)

    def build_factor(self, rho: float) -> dict:
        """Cholesky-factor (P + rho A^T A); returned as a dict the kernel
        can store on its state. Numpy reference uses scipy.linalg.cho_factor
        if available, falls back to numpy's Cholesky otherwise."""
        H = self.P + rho * (self.A.T @ self.A)
        try:
            from scipy.linalg import cho_factor
            c, lower = cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
            return {"cho_factor": (c, lower), "kind": "scipy_cho_factor"}
        except ImportError:
            L = np.linalg.cholesky(H)
            return {"L": L, "kind": "numpy_cholesky"}

    def solve_with_factor(self, factor: dict, rhs: np.ndarray) -> np.ndarray:
        """Solve (P + rho A^T A) x = rhs given the cached factor."""
        if factor["kind"] == "scipy_cho_factor":
            from scipy.linalg import cho_solve
            return cho_solve(factor["cho_factor"], rhs, check_finite=False)
        L = factor["L"]
        # solve L y = rhs, then L.T x = y
        y = np.linalg.solve_triangular(L, rhs, lower=True) if hasattr(np.linalg, "solve_triangular") else _np_trisolve(L, rhs, lower=True)
        return _np_trisolve(L.T, y, lower=False)

    def primal_residual(self, x: np.ndarray) -> float:
        """||A x - b||_2."""
        return float(np.linalg.norm(self.A @ x - self.b_constraint))


def _np_trisolve(T: np.ndarray, b: np.ndarray, *, lower: bool) -> np.ndarray:
    """Triangular solve fallback for numpy without scipy (rare).

    Uses scipy if present (much faster), otherwise dense solve.
    """
    try:
        from scipy.linalg import solve_triangular
        return solve_triangular(T, b, lower=lower, check_finite=False)
    except ImportError:
        return np.linalg.solve(T, b)
