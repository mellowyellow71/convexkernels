"""One panel, one ruler: the gap-panel assembly must use trusted_gap end to end.

Regression tests for the bug where the panel script built the sklearn curve on
the KKT-residual ruler while every other member (and the candidate) used the
duality gap — two different units on one axis, silently corrupting the
"dominated by sklearn" comparison.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from convexkernels.bench.curves import baseline_kkt_time_curve
from convexkernels.bench.metrics import trusted_gap, trusted_kkt
from convexkernels.frontend.lasso import Lasso

REPO = Path(__file__).resolve().parent.parent


def _problem(m=300, n=100, k=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam)


def _sklearn_coef(prob, cap):
    import warnings

    from sklearn.linear_model import Lasso as SkLasso

    model = SkLasso(alpha=prob.lam / prob.m, fit_intercept=False,
                    max_iter=int(cap), tol=1e-14, selection="cyclic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prob.A, prob.b)
    return np.asarray(model.coef_)


def test_sklearn_curve_metric_is_actually_applied():
    # The curve's y-values must be the requested metric of sklearn's iterates —
    # cyclic CD is deterministic, so an independent fit reproduces them exactly.
    prob = _problem()
    sweep = (5, 2000)
    curve = baseline_kkt_time_curve(prob, "sklearn", sweep, metric=trusted_gap)
    assert len(curve) == len(sweep)
    # Curves are sorted by wall time, not sweep order — compare as value sets.
    got = sorted(y for _, y in curve)
    want_gap = sorted(trusted_gap(prob, _sklearn_coef(prob, cap)) for cap in sweep)
    want_kkt = sorted(trusted_kkt(prob, _sklearn_coef(prob, cap)) for cap in sweep)
    assert got == pytest.approx(want_gap, rel=1e-12)
    # and it is NOT the KKT ruler (the two genuinely differ on these iterates)
    assert got != pytest.approx(want_kkt, rel=1e-3)


def test_panel_script_assembly_uses_one_ruler():
    # Load the actual script and check build_gap_panel puts sklearn on the gap
    # ruler — this is the assembly path where the KKT curve leaked in.
    spec = importlib.util.spec_from_file_location(
        "run_pareto_panel", REPO / "scripts" / "run_pareto_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    prob = _problem()
    sweep = (5, 2000)
    panel = mod.build_gap_panel(prob, ["sklearn"], sweep)
    expected = baseline_kkt_time_curve(prob, "sklearn", sweep, metric=trusted_gap)
    got = sorted(y for _, y in panel["sklearn"])
    want = sorted(y for _, y in expected)
    assert got == pytest.approx(want, rel=1e-9)
