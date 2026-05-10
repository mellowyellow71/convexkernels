"""Base abstraction for convex problems with a closed-form KKT residual.

Subclasses (`Lasso`, `NonnegLasso`, ...) provide problem-specific implementations
of `prox`, `grad_smooth`, `kkt_residual`, etc. The synthesis loop dispatches over
slot keys `(problem_family, algorithm, hardware, dtype)` and only sees the
abstract interface below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    @abstractmethod
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute A @ x."""

    @abstractmethod
    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """Compute A.T @ y."""

    @abstractmethod
    def grad_smooth(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the smooth part of the objective at x."""

    @abstractmethod
    def prox(self, v: np.ndarray, t: float) -> np.ndarray:
        """prox_{t * g}(v) where g is the non-smooth part of the objective."""

    @abstractmethod
    def kkt_residual(self, x: np.ndarray) -> float:
        """Scale-free scalar KKT residual; equals 0 iff x is optimal."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Dimension of the decision variable x."""

    @property
    @abstractmethod
    def L(self) -> float:
        """Lipschitz constant of the smooth gradient (for FISTA step size)."""

    @property
    @abstractmethod
    def lambda_max(self) -> float:
        """Critical regularization above which x* = 0."""
