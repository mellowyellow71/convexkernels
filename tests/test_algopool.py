"""Tests for the CD + gap-safe-screening algorithm-pool proposer.

Every family template must (a) compile as a standalone module, (b) reach a
tight duality gap on the trusted ruler, and (c) return a full-length iterate —
screening families must reassemble the full vector, not the reduced one. The
screening families are additionally checked on a problem whose true support
is known, and end-to-end through the Pareto loop.
"""
from __future__ import annotations

import functools
import types

import numpy as np
import pytest

from convexkernels.bench.metrics import trusted_gap
from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.recorder import Recorder
from convexkernels.synth.proposers.algopool import FAMILIES, AlgoPoolProposer


def _problem(m=500, n=200, k=12, lam_frac=0.08, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    sup = rng.choice(n, k, replace=False)
    xt[sup] = rng.standard_normal(k) + np.sign(rng.standard_normal(k))
    b = A @ xt + 0.005 * rng.standard_normal(m)
    lam = lam_frac * float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam)


def _run_family(name: str, prob, tol=1e-10, max_time_s=20.0):
    mod = types.ModuleType(f"cand_{name}")
    exec(compile(FAMILIES[name], name, "exec"), mod.__dict__)
    p = mod.prepare_problem(prob) if hasattr(mod, "prepare_problem") else prob
    rec = Recorder(functools.partial(trusted_gap, prob), max_time_s=max_time_s)
    x = mod.solve(p, rec, kkt_tol=tol, max_time_s=max_time_s)
    return np.asarray(x, dtype=np.float64), rec


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_family_reaches_tight_gap(name):
    prob = _problem()
    x, _ = _run_family(name, prob)
    assert x.shape == (prob.n,)                       # full-length iterate
    assert trusted_gap(prob, x) < 1e-10


def test_screening_recovers_known_support():
    # Screening families must not zero out true-support coefficients.
    prob = _problem(k=8, lam_frac=0.05, seed=3)
    x_ref, _ = _run_family("fista_gram", prob, tol=1e-12)
    for name in ("cd_screen", "fista_screen", "gram_screen"):
        x, _ = _run_family(name, prob, tol=1e-12)
        assert np.max(np.abs(x - x_ref)) < 1e-6, name


def test_proposer_emits_each_family_once_then_repeats():
    p = AlgoPoolProposer()
    seen = []
    for _ in range(len(FAMILIES)):
        e = p.propose({"research_state": {"tried_directions": []}})
        fam = e.rationale.split(":", 1)[0]
        assert fam in FAMILIES and fam not in seen
        seen.append(fam)
    # pool exhausted -> re-emits (the loop's dup-source guard absorbs it)
    e = p.propose({"research_state": {"tried_directions": []}})
    assert e.rationale.split(":", 1)[0] in FAMILIES


def test_proposer_skips_families_in_research_state():
    tried = [{"idea": "cd_screen: generated algorithm-family candidate"}]
    p = AlgoPoolProposer()
    e = p.propose({"research_state": {"tried_directions": tried}})
    assert not e.rationale.startswith("cd_screen")


def test_pool_through_pareto_loop(tmp_path):
    from convexkernels.synth.lineage import Slot
    from convexkernels.synth.loop import run_synth_loop

    prob = _problem(m=400, n=150, k=8)
    rows = run_synth_loop(
        proposer=AlgoPoolProposer(),
        problem=prob,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=Slot(problem_family="lasso", algorithm="open", hardware="cpu", dtype="open"),
        state_root=tmp_path,
        n_proposals=3,
        reps=1,
        kkt_tol=1e-8,
        max_time_s=10.0,
        selection="pareto",
        problem_backend="native",
        compute_baselines=False,
        verbose=False,
    )
    assert len(rows) == 3
    for r in rows:
        assert r.decision.get("reason", "").startswith(
            ("keep:adds_frontier_hv", "discard:pareto_dominated", "discard:duplicate_source")
        )
    # the pool's families all pass the gate, so at least one must extend the frontier
    assert any(r.decision.get("accepted") for r in rows)
