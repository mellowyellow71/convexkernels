"""Basis pursuit denoising (BP): min_x ||x||_1  s.t.  A x = b.

Specimen for testing cross-algorithm / cross-problem transfer (PDHG-TV ->
PDHG-BP). Saddle-point form for PDHG / Chambolle-Pock:

    min_x max_y  <A x, y> + ||x||_1 - <b, y>

with f(x) = ||x||_1 and g*(y) = <b, y> (linear). Then:

  prox_{tau f}(v) = sign(v) * max(|v| - tau, 0)            (soft-threshold)
  prox_{sigma g*}(z) = z - sigma * b                       (linear function prox)

Step constraint: tau * sigma * ||A||_2^2 <= 1.

Convergence is tracked via the primal residual ||A x - b||_2 since the gap
formulation involves an indicator that's zero only at exact feasibility;
this matches Chambolle-Pock's convention for constrained problems.

Why BP for transfer testing: same algorithm (PDHG) as TV, different
operator (dense A vs forward-difference). An accepted edit on TV-1D
(e.g. the recompute-adjacent-duals fusion) should *not* directly transfer
because A is dense; this gives a clean negative result for the cross-
problem transfer machinery. A successful transfer would target PDHG-level
mutations (step adaptation, restart) rather than TV-stencil-specific ones.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np


class BasisPursuit:
    """min ||x||_1 s.t. A x = b. PDHG is the natural solver."""

    def __init__(self, A: np.ndarray, b: np.ndarray):
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got {A.shape}")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError(f"b must be 1D length {A.shape[0]}, got {b.shape}")
        self.A = np.ascontiguousarray(A, dtype=np.float64)
        self.b = np.ascontiguousarray(b, dtype=np.float64)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @cached_property
    def L_K(self) -> float:
        """Spectral norm ||A||_2."""
        return float(np.linalg.norm(self.A, ord=2))

    def K_apply(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def K_T_apply(self, y: np.ndarray) -> np.ndarray:
        return self.A.T @ y

    def prox_f(self, v: np.ndarray, tau: float) -> np.ndarray:
        """Soft-thresholding: prox of tau * ||x||_1 at v."""
        return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

    def prox_g_conjugate(self, z: np.ndarray, sigma: float) -> np.ndarray:
        """Prox of sigma * <b, y> at z. Closed form: z - sigma * b."""
        return z - sigma * self.b

    def primal_objective(self, x: np.ndarray) -> float:
        """L1 norm of x (does not enforce feasibility)."""
        return float(np.sum(np.abs(x)))

    def primal_residual(self, x: np.ndarray) -> float:
        """||A x - b||_2 — feasibility violation."""
        return float(np.linalg.norm(self.A @ x - self.b))

    def dual_objective(self, y: np.ndarray) -> float:
        """Dual is -<b, y>. Y feasibility requires ||A^T y||_inf <= 1
        (the dual of the L1 norm); we don't project here, so the value is
        meaningful only when y is feasible. The PDHG iterates respect this
        by construction of prox_f and prox_g*."""
        return -float(np.dot(self.b, y))

    def primal_dual_gap(self, x: np.ndarray, y: np.ndarray) -> float:
        """Scale-free convergence indicator: combines L1 gap and feasibility.

        For BP the natural saddle-point gap is:
            gap = ||x||_1 + <b, y> + penalty * ||Ax - b||

        with penalty large enough to enforce feasibility at the optimum.
        Normalize by max(||b||, 1) so it's scale-free in b.
        """
        residual = self.primal_residual(x)
        l1 = self.primal_objective(x)
        # Feasibility-enforced gap. The first term is the standard
        # saddle-point gap (zero at the optimum when y is feasible);
        # adding the residual ensures we don't accept infeasible iterates.
        gap = abs(l1 + float(np.dot(self.b, y))) + residual
        scale = max(float(np.linalg.norm(self.b)), 1.0)
        return gap / scale
