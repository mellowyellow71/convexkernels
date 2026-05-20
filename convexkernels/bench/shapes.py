"""Bench problem generators.

Deterministic synthetic LASSO instances. Real datasets (rcv1, news20) live
in `datasets.py` (added later if/when needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.lasso import Lasso
from ..frontend.nonnegative_lasso import NonnegativeLasso
from ..frontend.total_variation import TVDenoising1D, TVDenoising2D


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


@dataclass(frozen=True)
class TvShapeSpec:
    """Total-variation problem shape: piecewise-constant truth + Gaussian noise."""
    name: str
    n: int
    n_pieces: int = 8
    noise: float = 0.1
    lam: float = 0.5


DEFAULT_TV1D_SHAPES: tuple[TvShapeSpec, ...] = (
    TvShapeSpec("tv1d_small",  n=256,  n_pieces=4, noise=0.1, lam=0.5),
    TvShapeSpec("tv1d_medium", n=2048, n_pieces=16, noise=0.1, lam=0.5),
    TvShapeSpec("tv1d_large",  n=16384, n_pieces=64, noise=0.1, lam=0.5),
)


def make_synthetic_tv_1d(spec: TvShapeSpec, seed: int = 0) -> TVDenoising1D:
    """Generate a piecewise-constant truth + Gaussian noise 1D TV problem.

    The autoresearch loop uses this as a fixed-shape evaluator for PDHG seeds.
    Number of pieces and noise level set the difficulty; lam set near 0.5 for
    a recognizable-but-non-trivial recovery target.
    """
    rng = np.random.default_rng(seed)
    breakpoints = np.sort(rng.choice(spec.n, size=spec.n_pieces - 1, replace=False))
    levels = rng.standard_normal(spec.n_pieces)
    truth = np.zeros(spec.n)
    edges = np.concatenate([[0], breakpoints, [spec.n]])
    for i in range(spec.n_pieces):
        truth[edges[i]:edges[i + 1]] = levels[i]
    b = truth + spec.noise * rng.standard_normal(spec.n)
    return TVDenoising1D(b, lam=spec.lam)
