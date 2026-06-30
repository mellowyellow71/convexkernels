"""Tests for the torch/CUDA batched-FISTA-Gram path seed.

Run on CPU by default so they pass in GPU-less CI (the switch logic and bf16
numerics are device-independent); the CUDA-only path is exercised when a GPU is
present. torch is an optional dep, so the whole module skips without it.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from convexkernels.bench.metrics import trusted_kkt
from convexkernels.frontend.lasso_path import LassoPath
from convexkernels.synth.recorder import Recorder
from convexkernels.kernels.torch.seeds import gram_fista_path_torch as seed


def _problem(m=500, n=200, k=12, K=12, rho=0.0, seed_val=0):
    rng = np.random.default_rng(seed_val)
    Z = rng.standard_normal((m, n))
    if rho > 0:
        Z = np.sqrt(1 - rho) * Z + np.sqrt(rho) * rng.standard_normal((m, 1))
    A = Z / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lmax = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(0.6 * lmax, 0.03 * lmax, K)
    return LassoPath(A, b, lambdas)


def _solve(prob, strat, device="cpu", tol=1e-6, max_time_s=30.0, check_every=20):
    kkt_fn = functools.partial(trusted_kkt, prob)
    view = seed.prepare_problem(prob, {"dtype_strategy": strat, "device": device})
    rec = Recorder(kkt_fn, max_time_s=max_time_s)
    X = seed.solve(view, rec, kkt_tol=tol, max_time_s=max_time_s, check_every=check_every)
    return X, view, rec


def test_fp64_reaches_gate_and_returns_host_array():
    prob = _problem(rho=0.6)                                # fp32 would floor here
    X, _, _ = _solve(prob, "fp64")
    assert isinstance(X, np.ndarray)                        # host array, not a CUDA tensor
    assert X.shape == (prob.n, prob.K)
    assert trusted_kkt(prob, X.astype(np.float64)) < 1e-6


def test_on_device_kkt_matches_trusted():
    prob = _problem()
    X, view, _ = _solve(prob, "fp64")
    Xt = torch.as_tensor(X, dtype=torch.float64, device=view.device)
    kdev = float(view.kkt_device(Xt))
    ktrust = trusted_kkt(prob, X.astype(np.float64))
    # both are tiny near the optimum; agree to a loose absolute tolerance
    assert abs(kdev - ktrust) < 1e-5


def test_bf16_switch_reaches_exact_gate():
    # Switch ends in an fp64 endgame, so it reaches the gate even on a problem
    # whose fp32 floor sits above 1e-6.
    prob = _problem(rho=0.6)
    X, _, _ = _solve(prob, "bf16_switch")
    assert trusted_kkt(prob, X.astype(np.float64)) < 1e-6


def test_pure_bf16_floors_above_gate():
    # The whole point of the switch: pure low precision is biased and cannot
    # reach the exact gate, so it must be caught (never silently "passes").
    prob = _problem()
    X, _, _ = _solve(prob, "bf16", max_time_s=10.0)
    assert trusted_kkt(prob, X.astype(np.float64)) > 1e-6


def test_unknown_strategy_falls_back_to_fp64_exact():
    prob = _problem()
    view = seed.prepare_problem(prob, {"dtype_strategy": "nonsense", "device": "cpu"})
    assert view.mode == "exact" and view.Glow is None
    X, _, _ = _solve(prob, "nonsense")
    assert trusted_kkt(prob, X.astype(np.float64)) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cuda_fp16_switch_reaches_gate():
    prob = _problem(rho=0.6)
    X, _, _ = _solve(prob, "fp16_switch", device="cuda")
    assert trusted_kkt(prob, X.astype(np.float64)) < 1e-6
