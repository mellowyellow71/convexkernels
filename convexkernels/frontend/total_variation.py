"""Total-variation L2 denoising: min_x (1/2)||x - b||^2 + lam ||K x||_1.

Specimen #2 for the synthesis loop, the canonical Chambolle-Pock target.
``K`` is the discrete forward-difference gradient. The saddle-point form is

    min_x max_y  <Kx, y> + (1/2)||x - b||^2 - I{||y||_inf <= lam}

with f(x) = (1/2)||x - b||^2 and g*(y) = I{||y||_inf <= lam}.

``L_K^2 = ||K||_2^2 <= 4`` (sharp for the no-boundary forward-difference
gradient on a length-n signal as n -> infinity), so any ``tau, sigma`` with
``tau * sigma * 4 <= 1`` is a valid PDHG step pair.

Why TV (not basis pursuit) as PDHG specimen 2:
- ``K = grad`` has known sparse structure → real fusion lever for kernels.
- 1D and 2D variants share the host driver but differ in the K kernel,
  giving the synthesis loop two natural slots that share an algorithm.
- A CVXPY oracle (CLARABEL on ``cp.tv``) gives a ground truth for tests.
"""

from __future__ import annotations

from functools import cached_property
from typing import Tuple

import numpy as np

from ..algorithms.gap import (
    tv_l2_dual_objective,
    tv_l2_primal_dual_gap,
    tv_l2_primal_objective,
)


def grad_1d(x: np.ndarray) -> np.ndarray:
    """Forward-difference gradient with no boundary: y[i] = x[i+1] - x[i]. Output length n-1."""
    return x[1:] - x[:-1]


def grad_T_1d(y: np.ndarray, n: int) -> np.ndarray:
    """Adjoint of `grad_1d`. Output length n; y has length n-1."""
    out = np.zeros(n, dtype=y.dtype)
    out[:-1] -= y
    out[1:] += y
    return out


def grad_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2D forward-difference gradient. ``x`` shape (h, w) → (dy, dx) each shape (h-1, w) and (h, w-1)."""
    dy = x[1:, :] - x[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    return dy, dx


def grad_T_2d(dy: np.ndarray, dx: np.ndarray, h: int, w: int) -> np.ndarray:
    """Adjoint of `grad_2d`. Returns shape (h, w)."""
    out = np.zeros((h, w), dtype=dy.dtype)
    out[:-1, :] -= dy
    out[1:, :] += dy
    out[:, :-1] -= dx
    out[:, 1:] += dx
    return out


class TVDenoising1D:
    """1D total-variation L2 denoising problem.

    Saddle-point form for PDHG:
        min_x max_y  <K x, y> + (1/2)||x - b||^2 - I{||y||_inf <= lam}

    where K is the forward-difference operator (length n -> length n-1).
    """

    def __init__(self, b: np.ndarray, lam: float):
        if b.ndim != 1:
            raise ValueError(f"b must be 1D, got shape {b.shape}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.lam = float(lam)

    @property
    def n(self) -> int:
        return self.b.shape[0]

    @property
    def m(self) -> int:
        """Length of the dual variable: n - 1 for forward-difference."""
        return self.b.shape[0] - 1

    @cached_property
    def L_K(self) -> float:
        """Spectral norm of K. ||grad||_2 < 2 for any finite n; we use 2 as upper bound."""
        return 2.0

    def K_apply(self, x: np.ndarray) -> np.ndarray:
        return grad_1d(x)

    def K_T_apply(self, y: np.ndarray) -> np.ndarray:
        return grad_T_1d(y, self.n)

    def prox_f(self, v: np.ndarray, tau: float) -> np.ndarray:
        """prox of tau * f at v, f(x) = (1/2)||x - b||^2.  Closed form: (v + tau b) / (1 + tau)."""
        return (v + tau * self.b) / (1.0 + tau)

    def prox_g_conjugate(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """prox of sigma * g* at z, g*(y) = I{||y||_inf <= lam}.  Closed form: clip to lam-ball."""
        del sigma  # indicator function: prox is just projection, sigma drops out
        return np.clip(z, -self.lam, self.lam)

    def primal_objective(self, x: np.ndarray) -> float:
        return tv_l2_primal_objective(self.b, self.K_apply, self.lam, x)

    def dual_objective(self, y: np.ndarray) -> float:
        return tv_l2_dual_objective(self.b, self.K_T_apply, self.lam, y)

    def primal_dual_gap(self, x: np.ndarray, y: np.ndarray) -> float:
        return tv_l2_primal_dual_gap(
            self.b, self.K_apply, self.K_T_apply, self.lam, x, y
        )


class TVDenoising2D:
    """2D total-variation L2 denoising on an h x w image.

    Same saddle-point structure as 1D; the dual y is a pair (dy, dx) where
    ``dy`` has shape (h-1, w) and ``dx`` has shape (h, w-1). The L1 norm of
    Kx is the *isotropic* TV used by Chambolle-Pock when summed over pixels
    via ``sqrt(dy^2 + dx^2)``; the *anisotropic* TV used here for simplicity
    sums |dy| + |dx|. The synthesis loop's specimen is anisotropic by default;
    isotropic is a flag that flips the prox of g*.
    """

    def __init__(self, b: np.ndarray, lam: float, *, isotropic: bool = False):
        if b.ndim != 2:
            raise ValueError(f"b must be 2D, got shape {b.shape}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        self.b = np.ascontiguousarray(b, dtype=np.float64)
        self.lam = float(lam)
        self.isotropic = bool(isotropic)

    @property
    def shape(self) -> tuple[int, int]:
        return self.b.shape

    @property
    def n(self) -> int:
        return int(np.prod(self.b.shape))

    @cached_property
    def L_K(self) -> float:
        return float(np.sqrt(8.0))  # ||grad||_2^2 < 8 in 2D forward-difference

    def K_apply(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_2d(x)

    def K_T_apply(self, y: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        dy, dx = y
        h, w = self.b.shape
        return grad_T_2d(dy, dx, h, w)

    def prox_f(self, v: np.ndarray, tau: float) -> np.ndarray:
        return (v + tau * self.b) / (1.0 + tau)

    def prox_g_conjugate(
        self, z: Tuple[np.ndarray, np.ndarray], sigma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        del sigma
        dy, dx = z
        if self.isotropic:
            mag = np.sqrt(dy ** 2 + dx ** 2)
            scale = np.minimum(self.lam / np.maximum(mag, 1e-30), 1.0)
            return dy * scale, dx * scale
        return np.clip(dy, -self.lam, self.lam), np.clip(dx, -self.lam, self.lam)

    def primal_objective(self, x: np.ndarray) -> float:
        dy, dx = self.K_apply(x)
        if self.isotropic:
            tv = float(np.sum(np.sqrt(dy ** 2 + dx ** 2)))
        else:
            tv = float(np.sum(np.abs(dy)) + np.sum(np.abs(dx)))
        return 0.5 * float(np.sum((x - self.b) ** 2)) + self.lam * tv

    def dual_objective(self, y: Tuple[np.ndarray, np.ndarray]) -> float:
        KTy = self.K_T_apply(y)
        return -0.5 * float(np.sum(KTy ** 2)) + float(np.sum(KTy * self.b))

    def primal_dual_gap(
        self, x: np.ndarray, y: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        dy, dx = y
        # Project y onto the lam-ball (anisotropic) or lam-circle (isotropic)
        # before computing the dual, to keep gap >= 0 for any (x, y).
        if self.isotropic:
            mag = np.sqrt(dy ** 2 + dx ** 2)
            scale = np.minimum(self.lam / np.maximum(mag, 1e-30), 1.0)
            y_proj = (dy * scale, dx * scale)
        else:
            y_proj = (np.clip(dy, -self.lam, self.lam), np.clip(dx, -self.lam, self.lam))
        P = self.primal_objective(x)
        D = self.dual_objective(y_proj)
        scale_norm = 0.5 * float(np.sum(self.b ** 2)) + 1.0
        return float(max(P - D, 0.0)) / scale_norm
