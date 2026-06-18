"""Panel summary + plot artifact (the empirical headline)."""

from __future__ import annotations

import json
import math

import numpy as np

from convexkernels.bench.plotting import (
    panel_summary,
    plot_state_root,
    render_summary_md,
)
from convexkernels.bench.curves import problem_hash
from convexkernels.frontend.lasso import Lasso


def _curve_reaching(t_cross, tol=1e-6):
    # Two points straddling the target so time_to_kkt interpolates ~ t_cross.
    return [(t_cross * 0.5, tol * 10), (t_cross, tol)]


def test_panel_summary_ranks_and_computes_speedup():
    baselines = {
        "CLARABEL": _curve_reaching(0.090),
        "sklearn": _curve_reaching(0.010),
        "adelie": [],  # absent -> never reaches -> inf
    }
    champion = {"tag": "gram_fista", "time_to_kkt_s": 0.004}
    s = panel_summary(baselines, 1e-6, champion=champion)
    # champion fastest, then sklearn, then CLARABEL, adelie last (inf)
    order = [r["method"] for r in s["ranking"]]
    assert order[0] == "gram_fista"
    assert order.index("sklearn") < order.index("CLARABEL")
    assert order[-1] == "adelie"
    assert s["best_baseline"]["method"] == "sklearn"
    assert math.isclose(s["champion_speedup_vs_best_baseline"], 0.010 / 0.004, rel_tol=1e-9)
    assert s["champion_beats_baselines"] is True


def test_panel_summary_marks_champion_slower():
    baselines = {"sklearn": _curve_reaching(0.001)}
    s = panel_summary(baselines, 1e-6, champion={"tag": "x", "time_to_kkt_s": 0.05})
    assert s["champion_beats_baselines"] is False
    assert s["champion_speedup_vs_best_baseline"] < 1.0


def test_render_summary_md_has_table_and_verdict():
    s = panel_summary(
        {"sklearn": _curve_reaching(0.01)}, 1e-6,
        champion={"tag": "cand", "time_to_kkt_s": 0.005},
    )
    md = render_summary_md(s)
    assert "| rank | method | kind | time to target |" in md
    assert "faster than" in md
    assert "cand" in md and "sklearn" in md


def test_plot_state_root_writes_png_and_summary(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 25))
    b = rng.standard_normal(60)
    prob = Lasso(A, b, 0.1 * float(np.max(np.abs(A.T @ b))))

    # Pre-seed the baseline cache so the plot doesn't re-solve anything.
    phash = problem_hash(prob)
    bdir = tmp_path / "baselines" / phash
    bdir.mkdir(parents=True)
    (bdir / "sklearn.json").write_text(json.dumps([[0.005, 1e-5], [0.01, 1e-7]]))
    (bdir / "CLARABEL.json").write_text(json.dumps([[0.05, 1e-4], [0.09, 1e-7]]))

    out = plot_state_root(tmp_path, prob, kkt_tol=1e-6)
    assert out is not None and out.exists()
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["best_baseline"]["method"] == "sklearn"
    assert out.with_suffix(".summary.md").exists()
