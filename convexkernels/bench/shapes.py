"""Bench problem generators.

Deterministic synthetic LASSO instances. Real datasets (rcv1, news20) live
in `datasets.py` (added later if/when needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.lasso import Lasso
from ..frontend.nonnegative_lasso import NonnegativeLasso


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    m: int
    n: int
    sparsity: float = 0.05
    noise: float = 1e-2
    lam_frac: float = 0.1


DEFAULT_SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec("tall_small",  m=2000, n=500),
    ShapeSpec("tall_medium", m=5000, n=2000),
    ShapeSpec("wide_small",  m=500,  n=2000),
    ShapeSpec("wide_large",  m=2000, n=10000),
)


def make_synthetic_lasso(spec: ShapeSpec, seed: int = 0) -> Lasso:
    """Generate a synthetic LASSO instance.

    A entries are iid N(0,1), x_true is sparsity-fraction sparse with N(0,1)
    nonzeros, b = A x_true + noise * N(0,1). Lambda set as `lam_frac * lam_max`
    so the optimum is sparse but non-trivial.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((spec.m, spec.n))
    x_true = rng.standard_normal(spec.n) * (rng.random(spec.n) < spec.sparsity)
    b = A @ x_true + spec.noise * rng.standard_normal(spec.m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam=spec.lam_frac * lam_max)


def make_synthetic_nonnegative_lasso(
    spec: ShapeSpec,
    seed: int = 0,
) -> NonnegativeLasso:
    """Generate a synthetic nonnegative LASSO instance."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((spec.m, spec.n))
    x_true = np.abs(rng.standard_normal(spec.n)) * (
        rng.random(spec.n) < spec.sparsity
    )
    b = A @ x_true + spec.noise * rng.standard_normal(spec.m)
    lam_max = float(max(np.max(A.T @ b), 0.0))
    return NonnegativeLasso(A, b, lam=spec.lam_frac * lam_max)
