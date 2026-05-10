"""FISTA seed that precomputes dense Gram data before iteration.

The per-iteration tail kernel is the same as `fista_step_v0`; the difference is
the problem preparation hook. The sandbox calls `prepare_problem()` once before
warmup/timing so the FISTA gradient becomes:

    g = (A.T @ A) @ y - A.T @ b

instead of:

    g = A.T @ (A @ y - b)

This is a larger search dimension for tall dense LASSO: setup is more
expensive, but solve iterations can be faster when the Gram data is amortized.
"""

from __future__ import annotations

import mlx.core as mx

from convexkernels.kernels.mlx.lib import LassoGramMLX, LassoMLX
from convexkernels.kernels.mlx.seeds.fista_step_v0 import (
    FistaStateMLX,
    fista_step,
    init_state,
)


def prepare_problem(problem: LassoMLX, config: dict | None = None) -> LassoGramMLX:
    """Precompute `A.T @ A` and `A.T @ b` for dense LASSO/FISTA."""
    config = config or {}
    dtype_strategy = config.get("dtype_strategy", "fp32")
    gradient_dtype = {
        "fp32": mx.float32,
        "fp16_storage": mx.float16,
        "mixed_gram": mx.float16,
    }.get(dtype_strategy, problem.dtype)
    kkt_dtype = mx.float32 if gradient_dtype != mx.float32 else None
    return LassoGramMLX.from_lasso_mlx(
        problem,
        gradient_dtype=gradient_dtype,
        kkt_dtype=kkt_dtype,
    )


__all__ = ["FistaStateMLX", "prepare_problem", "init_state", "fista_step"]
