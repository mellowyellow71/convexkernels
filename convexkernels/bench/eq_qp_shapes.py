"""Synthetic equality-QP problems for the specimen-3 bench."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.equality_qp import EqualityQP


@dataclass(frozen=True)
class EqQpShapeSpec:
    name: str
    n: int            # primal dimension
    m: int            # number of equality constraints
    cond_P: float = 50.0   # condition number of P
    seed: int = 0


DEFAULT_EQ_QP_SHAPES: tuple[EqQpShapeSpec, ...] = (
    EqQpShapeSpec("eqqp_small",  n=64,   m=8),
    EqQpShapeSpec("eqqp_medium", n=512,  m=64),
    EqQpShapeSpec("eqqp_large",  n=2048, m=256),
)


@dataclass(frozen=True)
class BasisPursuitShapeSpec:
    """min ||x||_1 s.t. A x = b. m << n (under-determined system)."""
    name: str
    m: int       # number of constraints
    n: int       # primal dimension (n > m for sparsity recovery to make sense)
    sparsity: float = 0.05   # fraction of nonzero entries in true x
    seed: int = 0


DEFAULT_BP_SHAPES: tuple[BasisPursuitShapeSpec, ...] = (
    BasisPursuitShapeSpec("bp_small",  m=64,  n=256),
    BasisPursuitShapeSpec("bp_medium", m=256, n=1024),
    BasisPursuitShapeSpec("bp_large",  m=512, n=4096),
)


def make_synthetic_basis_pursuit(spec: BasisPursuitShapeSpec, seed: int | None = None):
    """Synthetic basis-pursuit instance with sparse ground truth.

    A is m x n Gaussian (m < n). x_true is sparsity-fraction sparse with
    N(0,1) nonzeros. b = A @ x_true. The recovered x* ought to coincide
    with x_true under standard RIP conditions.
    """
    from ..frontend.basis_pursuit import BasisPursuit

    rng = np.random.default_rng(spec.seed if seed is None else seed)
    A = rng.standard_normal((spec.m, spec.n)) / np.sqrt(spec.m)
    x_true = rng.standard_normal(spec.n) * (rng.random(spec.n) < spec.sparsity)
    b = A @ x_true
    return BasisPursuit(A, b)


def make_synthetic_eq_qp(spec: EqQpShapeSpec, seed: int | None = None) -> EqualityQP:
    """Generate a synthetic equality-constrained QP.

    P is a positive-definite matrix with controlled condition number; q and
    A are dense Gaussian; b is chosen so that the problem has a feasible
    interior (i.e. b is in range(A) by construction).
    """
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    # Build P with controlled condition number via random orthogonal U:
    raw = rng.standard_normal((spec.n, spec.n))
    Q, _ = np.linalg.qr(raw)
    eigs = np.linspace(1.0, spec.cond_P, spec.n)
    P = Q @ np.diag(eigs) @ Q.T
    P = 0.5 * (P + P.T)  # numerical symmetry

    q = rng.standard_normal(spec.n)
    A = rng.standard_normal((spec.m, spec.n))
    # Ensure b is in range(A) so the constraint is feasible.
    x_feas = rng.standard_normal(spec.n)
    b = A @ x_feas
    return EqualityQP(P, q, A, b)
