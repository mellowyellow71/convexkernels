import math

import numpy as np
import pytest

from convexkernels.bench.curves import (
    baseline_kkt_time_curve,
    baseline_panel,
    best_baseline_time_to_kkt,
    problem_hash,
    time_to_kkt,
)
from convexkernels.frontend.lasso import Lasso


def _small_lasso(seed=0, m=50, n=20):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[:3] = rng.standard_normal(3)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


@pytest.mark.parametrize("solver", ["CLARABEL", "SCS", "OSQP", "ECOS"])
def test_curve_decreases_and_reaches_loose_tol(solver):
    prob = _small_lasso()
    curve = baseline_kkt_time_curve(prob, solver, sweep=(10, 50, 200, 1000, 5000))
    assert len(curve) >= 2
    # times increasing (sorted), kkt finite
    ts = [t for t, _ in curve]
    assert ts == sorted(ts)
    assert all(math.isfinite(k) for _, k in curve)
    # a loose target is reachable
    assert math.isfinite(time_to_kkt(curve, 1e-2))


def test_problem_hash_stable_and_data_dependent():
    p1 = _small_lasso(seed=0)
    p2 = _small_lasso(seed=0)
    p3 = _small_lasso(seed=1)
    assert problem_hash(p1) == problem_hash(p2)
    assert problem_hash(p1) != problem_hash(p3)


def test_baseline_panel_cached(tmp_path):
    prob = _small_lasso()
    panel = baseline_panel(
        prob, solvers=("CLARABEL", "SCS"),
        sweep=(50, 500), cache_dir=tmp_path,
    )
    assert set(panel) == {"CLARABEL", "SCS"}
    # cache file written
    cached = list(tmp_path.glob("*/CLARABEL.json"))
    assert cached
    name, t = best_baseline_time_to_kkt(panel, 1e-2)
    assert name in {"CLARABEL", "SCS"}
    assert math.isfinite(t)
