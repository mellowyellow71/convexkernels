"""End-to-end check of the algorithm-open solve contract via the sandbox."""

import json

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.sandbox import run_kernel, write_eval_config


def _small_lasso(seed=0, m=60, n=25):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[:4] = rng.standard_normal(4)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def test_numpy_solve_reaches_target(tmp_path):
    prob = _small_lasso()
    write_eval_config(
        tmp_path, prob,
        kernel_module="convexkernels.kernels.numpy_solve_ref",
        problem_backend="native",
        kkt_tol=1e-6, max_time_s=30.0,
    )
    res = run_kernel(tmp_path, timeout_s=60)
    assert res.status == "completed", res.error_message
    assert res.reached_target is True
    assert res.kkt_final <= 1e-6
    assert res.time_to_kkt_s is not None and res.time_to_kkt_s >= 0.0
    # trajectory recorded and ends below tol
    traj = json.loads((tmp_path / "trajectory.json").read_text())
    assert len(traj) >= 1
    assert traj[-1][1] <= 1e-6


def test_wrong_solve_is_rejected(tmp_path):
    # A candidate that returns zeros never reaches the target.
    bad = tmp_path / "bad_kernel.py"
    bad.write_text(
        "import numpy as np\n"
        "def solve(problem, recorder, *, kkt_tol, max_time_s):\n"
        "    x = np.zeros(problem.n)\n"
        "    recorder.record(x)\n"
        "    return x\n"
    )
    prob = _small_lasso()
    write_eval_config(
        tmp_path, prob,
        kernel_module=str(bad),
        problem_backend="native",
        kkt_tol=1e-6, max_time_s=10.0,
    )
    res = run_kernel(tmp_path, timeout_s=30)
    assert res.status == "completed"
    assert res.reached_target is False
    assert res.time_to_kkt_s is None


def test_candidate_cannot_fake_metric(tmp_path):
    # A candidate that lies about convergence (records nothing, returns zeros)
    # is still judged by the trusted KKT on the returned iterate.
    liar = tmp_path / "liar.py"
    liar.write_text(
        "import numpy as np\n"
        "def solve(problem, recorder, *, kkt_tol, max_time_s):\n"
        "    return np.zeros(problem.n)\n"
    )
    prob = _small_lasso()
    write_eval_config(
        tmp_path, prob,
        kernel_module=str(liar),
        problem_backend="native",
        kkt_tol=1e-6, max_time_s=10.0,
    )
    res = run_kernel(tmp_path, timeout_s=30)
    assert res.status == "completed"
    assert res.reached_target is False
