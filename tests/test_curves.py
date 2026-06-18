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


# --- native fast-LASSO baselines (sklearn / adelie) -------------------------

import importlib.util  # noqa: E402

from convexkernels.bench.curves import (  # noqa: E402
    DEFAULT_PANEL_SOLVERS,
    baseline_kkt_time_curve,
)
from convexkernels.frontend.lasso_path import LassoPath  # noqa: E402

_HAS_ADELIE = importlib.util.find_spec("adelie") is not None


def _small_path(seed=0, m=60, n=24, K=5):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[:3] = rng.standard_normal(3)
    b = A @ x_true + 0.01 * rng.standard_normal(m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(0.2 * lam_max, 0.02 * lam_max, K)
    return LassoPath(A, b, lambdas)


def test_sklearn_curve_single_decreases_and_reaches_tol():
    prob = _small_lasso()
    curve = baseline_kkt_time_curve(prob, "sklearn", sweep=(5, 25, 100, 400, 2000))
    assert len(curve) >= 2
    ts = [t for t, _ in curve]
    assert ts == sorted(ts)
    assert all(math.isfinite(k) for _, k in curve)
    # coordinate descent reaches a tight target fast on a well-conditioned shape
    assert math.isfinite(time_to_kkt(curve, 1e-2))


def test_sklearn_curve_handles_path():
    path = _small_path()
    curve = baseline_kkt_time_curve(path, "sklearn", sweep=(5, 50, 500))
    assert len(curve) >= 2
    assert all(math.isfinite(k) and t >= 0.0 for t, k in curve)


def test_adelie_curve_never_raises_and_is_wellformed():
    # adelie is an optional dep: absent -> empty curve; present -> valid curve.
    prob = _small_lasso()
    curve = baseline_kkt_time_curve(prob, "adelie", sweep=(5, 50, 500))
    assert isinstance(curve, list)
    if not _HAS_ADELIE:
        assert curve == []
    else:
        ts = [t for t, _ in curve]
        assert ts == sorted(ts)
        assert all(math.isfinite(k) for _, k in curve)


def test_default_panel_includes_fast_lasso_solvers():
    assert "sklearn" in DEFAULT_PANEL_SOLVERS
    assert "adelie" in DEFAULT_PANEL_SOLVERS


def test_panel_does_not_cache_empty_curves(tmp_path):
    # Requesting only adelie: when absent the curve is empty and must NOT be
    # cached, so a later run with adelie installed re-attempts it.
    prob = _small_lasso()
    panel = baseline_panel(prob, solvers=("adelie",), sweep=(50, 500), cache_dir=tmp_path)
    assert set(panel) == {"adelie"}
    cached = list(tmp_path.glob("*/adelie.json"))
    if panel["adelie"] == []:
        assert cached == []
    else:
        assert cached  # present + non-empty -> cached


def test_fast_lasso_beats_conic_panel_on_small_lasso():
    # The reason this baseline matters: the fast-LASSO bar is far tighter than
    # the conic panel, so it must be in the bar-to-beat.
    prob = _small_lasso(m=200, n=80)
    panel = baseline_panel(
        prob, solvers=("CLARABEL", "sklearn"),
        sweep=(5, 25, 100, 400, 1600), cache_dir=None,
    )
    t_sklearn = time_to_kkt(panel["sklearn"], 1e-4)
    t_clarabel = time_to_kkt(panel["CLARABEL"], 1e-4)
    assert math.isfinite(t_sklearn)
    assert t_sklearn < t_clarabel
