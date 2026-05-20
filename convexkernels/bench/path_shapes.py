"""Path-aware bench shape specs for the full-regularization-path LASSO pivot.

Each spec generates `(A, b, lambdas)` where `lambdas` is a log-spaced grid
from `lambda_max` down to `lam_min_frac * lambda_max` (standard glmnet/Adelie
convention: decreasing).

The hero shape (`path_wide_hero`) is where Adelie's coordinate descent hurts
worst: p >> n, full path of K lambdas. We commit to beating Adelie here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathShapeSpec:
    name: str
    m: int           # samples (rows of A)
    n: int           # features (cols of A)
    sparsity: float
    noise: float
    K: int           # number of lambdas in the path
    lam_min_frac: float  # smallest lambda = lam_min_frac * lambda_max


DEFAULT_PATH_SHAPES: tuple[PathShapeSpec, ...] = (
    # Hero: wide p >> n. Adelie's per-cycle CD cost scales with n (features),
    # so this is the structural-weakness regime. Matrix-matrix path-batching
    # on M-series unified memory should win here.
    PathShapeSpec(
        "path_wide_hero", m=1000, n=50000,
        sparsity=0.01, noise=1e-2, K=50, lam_min_frac=0.01,
    ),
    # Regression: medium tall. The original Gram-FISTA champion shape.
    PathShapeSpec(
        "path_tall_medium", m=5000, n=2000,
        sparsity=0.05, noise=1e-2, K=50, lam_min_frac=0.01,
    ),
    # Regression: square.
    PathShapeSpec(
        "path_square", m=10000, n=10000,
        sparsity=0.02, noise=1e-2, K=50, lam_min_frac=0.01,
    ),
    # Regression: small wide.
    PathShapeSpec(
        "path_wide_small", m=500, n=2000,
        sparsity=0.05, noise=1e-2, K=50, lam_min_frac=0.01,
    ),
)


def get_path_shape(name: str) -> PathShapeSpec:
    for spec in DEFAULT_PATH_SHAPES:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown path shape {name!r}; available: "
                   f"{[s.name for s in DEFAULT_PATH_SHAPES]}")


def make_path_problem(
    spec: PathShapeSpec, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `(A, b, lambdas)` for a path shape spec.

    Returns lambdas in decreasing order (high-to-low) to match glmnet/Adelie
    convention. The path is log-spaced from `lambda_max` (smallest lam with
    all-zero solution) down to `lam_min_frac * lambda_max`.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((spec.m, spec.n))
    x_true = rng.standard_normal(spec.n) * (rng.random(spec.n) < spec.sparsity)
    b = A @ x_true + spec.noise * rng.standard_normal(spec.m)
    lambda_max = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(
        lambda_max, spec.lam_min_frac * lambda_max, spec.K,
    )
    return (
        np.ascontiguousarray(A, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(lambdas, dtype=np.float64),
    )
