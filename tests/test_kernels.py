"""Functional-equivalence tests for kernels.

Skipped on Linux (no MLX). On Mac, runs the seed MLX kernel on a small problem
and asserts both numpy_ref and the MLX kernel reach KKT-optimum (per the
`assert_equivalent` contract); iterate drift is logged but does not fail.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from convexkernels.algorithms.fista import fista
from convexkernels.algorithms.kkt import assert_equivalent
from convexkernels.frontend.lasso import Lasso
from convexkernels.frontend.nonnegative_lasso import NonnegativeLasso

_HAS_MLX = importlib.util.find_spec("mlx") is not None
mlx_only = pytest.mark.skipif(not _HAS_MLX, reason="MLX not available on this host")


def make_tiny_lasso(seed: int = 0) -> Lasso:
    rng = np.random.default_rng(seed)
    m, n = 200, 100
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n) * (rng.random(n) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam=0.1 * lam_max)


def make_tiny_nonnegative_lasso(seed: int = 0) -> NonnegativeLasso:
    rng = np.random.default_rng(seed)
    m, n = 200, 100
    A = rng.standard_normal((m, n))
    x_true = np.abs(rng.standard_normal(n)) * (rng.random(n) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(m)
    lam_max = float(max(np.max(A.T @ b), 0.0))
    return NonnegativeLasso(A, b, lam=0.1 * lam_max)


@mlx_only
def test_mlx_seed_kernel_functional_equivalence_fp32():
    """fista_step_v0 (mlx fp32) and numpy_ref both reach KKT < 1e-6 on the same problem."""
    from convexkernels.kernels.mlx.lib import LassoMLX
    from convexkernels.kernels.mlx.seeds.fista_step_v0 import (
        fista_step as mlx_step,
        init_state as mlx_init,
    )
    import mlx.core as mx

    prob_np = make_tiny_lasso()
    prob_mlx = LassoMLX.from_lasso(prob_np, dtype=mx.float32)

    res_np = fista(prob_np, max_iters=5000, tol=1e-7, variant="basic")
    res_mlx = fista(
        prob_mlx, max_iters=5000, tol=1e-7, variant="basic",
        kernel_step=mlx_step, kernel_init=mlx_init,
    )

    assert res_np.converged
    assert res_mlx.converged

    # Functional equivalence: KKT must be small for both; drift logged.
    diag = assert_equivalent(
        res_mlx.x, res_np.x, prob_np, kkt_tol=1e-6, drift_warn=1e-3
    )
    print(f"fp32 mlx vs np: drift={diag['rel_drift']:.2e}, "
          f"kkt_mlx={diag['kkt_kernel']:.2e}, kkt_np={diag['kkt_ref']:.2e}")


@mlx_only
def test_mlx_seed_kernel_functional_equivalence_fp16_storage():
    """Same kernel at fp16 storage. Drift may be larger; KKT must still pass."""
    from convexkernels.kernels.mlx.lib import LassoMLX
    from convexkernels.kernels.mlx.seeds.fista_step_v0 import (
        fista_step as mlx_step,
        init_state as mlx_init,
    )
    import mlx.core as mx

    prob_np = make_tiny_lasso()
    prob_mlx16 = LassoMLX.from_lasso(prob_np, dtype=mx.float16)

    res_np = fista(prob_np, max_iters=5000, tol=1e-7, variant="basic")
    res_mlx = fista(
        prob_mlx16, max_iters=5000, tol=1e-3, variant="basic",
        kernel_step=mlx_step, kernel_init=mlx_init,
    )

    assert res_np.converged

    # fp16 end-to-end (no fp32 accumulator in this seed kernel) hits a precision
    # floor around 1e-3. This is the loosest dtype variant; the synth loop will
    # explore fp16-storage/fp32-accum as a separate slot. Drift is unbounded
    # here (different optima within the precision noise floor).
    diag = assert_equivalent(
        res_mlx.x, res_np.x, prob_np, kkt_tol=1e-2, drift_warn=2.0
    )
    print(f"fp16 mlx vs np fp64: drift={diag['rel_drift']:.2e}, "
          f"kkt_mlx={diag['kkt_kernel']:.2e}, kkt_np={diag['kkt_ref']:.2e}")


@mlx_only
def test_mlx_gram_seed_kernel_functional_equivalence_fp32():
    from convexkernels.kernels.mlx.lib import LassoGramMLX, LassoMLX
    from convexkernels.kernels.mlx.seeds.gram_fista_step_v0 import (
        fista_step as gram_step,
        init_state as gram_init,
        prepare_problem,
    )
    import mlx.core as mx

    prob_np = make_tiny_lasso(seed=2)
    prob_mlx = LassoMLX.from_lasso(prob_np, dtype=mx.float32)
    prob_gram = prepare_problem(prob_mlx, {"dtype_strategy": "fp32"})

    assert isinstance(prob_gram, LassoGramMLX)
    probe = mx.array(np.random.default_rng(0).standard_normal(prob_np.n), dtype=mx.float32)
    direct_g = np.asarray(prob_mlx.grad_smooth(probe))
    gram_g = np.asarray(prob_gram.grad_smooth(probe))
    np.testing.assert_allclose(gram_g, direct_g, rtol=3e-4, atol=3e-3)

    res_np = fista(prob_np, max_iters=5000, tol=1e-7, variant="restart")
    res_gram = fista(
        prob_gram,
        max_iters=5000,
        tol=1e-6,
        variant="restart",
        kernel_step=gram_step,
        kernel_init=gram_init,
    )

    assert res_np.converged
    assert res_gram.converged
    diag = assert_equivalent(
        res_gram.x, res_np.x, prob_np, kkt_tol=1e-6, drift_warn=1e-3
    )
    print(f"gram fp32 mlx vs np: drift={diag['rel_drift']:.2e}, "
          f"kkt_gram={diag['kkt_kernel']:.2e}, kkt_np={diag['kkt_ref']:.2e}")


@mlx_only
def test_mlx_nonnegative_seed_kernel_functional_equivalence_fp32():
    from convexkernels.kernels.mlx.lib import NonnegativeLassoMLX
    from convexkernels.kernels.mlx.seeds.nonnegative_fista_step_v0 import (
        fista_step as mlx_step,
        init_state as mlx_init,
    )
    import mlx.core as mx

    prob_np = make_tiny_nonnegative_lasso()
    prob_mlx = NonnegativeLassoMLX.from_problem(prob_np, dtype=mx.float32)

    res_np = fista(prob_np, max_iters=5000, tol=1e-7, variant="restart")
    res_mlx = fista(
        prob_mlx,
        max_iters=5000,
        tol=1e-6,
        variant="restart",
        kernel_step=mlx_step,
        kernel_init=mlx_init,
    )

    assert res_np.converged
    assert res_mlx.converged
    assert np.min(res_mlx.x) >= -1e-8

    diag = assert_equivalent(
        res_mlx.x,
        res_np.x,
        prob_np,
        kkt_tol=1e-6,
        drift_warn=1e-3,
    )
    print(f"nn fp32 mlx vs np: drift={diag['rel_drift']:.2e}, "
          f"kkt_mlx={diag['kkt_kernel']:.2e}, kkt_np={diag['kkt_ref']:.2e}")
