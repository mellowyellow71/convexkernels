"""Tests for the path-native algorithm-pool proposer.

Covers the load-bearing behaviour, not just that strings render:
  - every family execs to an importable `solve()` that reaches the *gated*
    trusted KKT on a real numpy LassoPath (and on a single Lasso via the
    one-column-path adapter);
  - dedup runs against the REAL research_state shape the loop produces
    (`algorithm_family` -> `_idea_key` -> `tried_directions[i]["idea"]`), not a
    fabricated one;
  - the proposer emits a distinct `algorithm_family` per candidate (the actual
    dedup key) and walks screening families then the FISTA mutation layer;
  - the mutation layer's champion-centred search moves exactly one knob.
"""
from __future__ import annotations

import time
import types

import numpy as np
import pytest

from convexkernels.bench.metrics import trusted_kkt
from convexkernels.frontend.lasso import Lasso
from convexkernels.frontend.lasso_path import LassoPath
from convexkernels.synth.proposers import algopool
from convexkernels.synth.proposers.algopool import (
    AlgoPoolProposer,
    SCREENING_FAMILIES,
    fista_config_family,
    fista_operator_grid,
    render_fista_path,
)
from convexkernels.synth.research_state import _idea_key, build_research_state


# --- fixtures ---------------------------------------------------------------
def _path(m=200, n=400, k=10, K=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam_max = float(np.max(np.abs(A.T @ b)))
    lambdas = np.geomspace(0.5 * lam_max, 0.02 * lam_max, K)
    return LassoPath(A, b, lambdas)


def _single(m=200, n=400, k=10, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam = 0.1 * float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam)


class _Recorder:
    """Mirror of the harness Recorder contract: record(x) -> trusted kkt."""

    def __init__(self, problem, max_time_s=30.0):
        self.p = problem
        self.max_time_s = max_time_s
        self._t0 = time.perf_counter()
        self.last = float("inf")
        self.n = 0

    def record(self, x):
        self.last = trusted_kkt(self.p, np.asarray(x, dtype=np.float64))
        self.n += 1
        return self.last

    def should_stop(self, tol):
        return self.last <= tol or (time.perf_counter() - self._t0) >= self.max_time_s


def _load(source: str):
    mod = types.ModuleType("cand")
    exec(compile(source, "<algopool-candidate>", "exec"), mod.__dict__)
    return mod


ALL_SOURCES = {
    **SCREENING_FAMILIES,
    "fista_path_restart_chk10": render_fista_path({"restart": True, "check_every": 10}),
}


# --- render / import --------------------------------------------------------
@pytest.mark.parametrize("name,source", list(ALL_SOURCES.items()))
def test_family_renders_to_importable_solve(name, source):
    mod = _load(source)
    assert callable(getattr(mod, "solve", None)), f"{name} has no solve()"


# --- the load-bearing gate: reaches the trusted KKT on a real path ----------
TOL = 1e-6


@pytest.mark.parametrize("name,source", list(ALL_SOURCES.items()))
def test_family_reaches_trusted_kkt_on_path(name, source):
    prob = _path()
    mod = _load(source)
    rec = _Recorder(prob)
    X = mod.solve(prob, rec, kkt_tol=TOL, max_time_s=30.0)
    X = np.asarray(X, dtype=np.float64)
    assert X.shape == (prob.n, prob.K)
    assert trusted_kkt(prob, X) <= TOL


@pytest.mark.parametrize("name,source", list(SCREENING_FAMILIES.items()))
def test_screening_family_handles_single_lasso(name, source):
    # The one-column-path adapter: a single Lasso must yield an (n,) iterate.
    prob = _single()
    mod = _load(source)
    rec = _Recorder(prob)
    x = mod.solve(prob, rec, kkt_tol=TOL, max_time_s=30.0)
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (prob.n,)
    assert trusted_kkt(prob, x) <= TOL


# --- dedup against the REAL research_state shape ----------------------------
def _lineage_row(fam, reason="discard:not_faster_than_champion"):
    # Exactly the shape loop.py._record writes (see _edit_dict / LineageRow).
    return {
        "id": "abcd1234",
        "edit": {"type": "full_source", "rationale": f"{fam}: ...",
                 "algorithm_family": fam},
        "decision": {"accepted": False, "reason": reason},
        "score": {"time_to_kkt_s": 1.0, "kkt_final": 1e-7},
    }


def test_algorithm_family_is_the_real_dedup_key():
    # A family emitted as algorithm_family round-trips through _idea_key into
    # research_state's tried_directions[i]["idea"].
    row = _lineage_row("path_cd_screen")
    assert _idea_key(row) == "path_cd_screen"


def test_proposer_skips_families_present_in_research_state():
    rows = [_lineage_row("path_cd_screen")]
    state = build_research_state(
        lineage_rows=rows, baseline_times={}, champion=None, kkt_tol=TOL,
    )
    ideas = {d["idea"] for d in state["tried_directions"]}
    assert "path_cd_screen" in ideas  # the real produced shape, not fabricated

    prop = AlgoPoolProposer()
    edit = prop.propose({"research_state": state, "current_source": ""})
    assert edit.algorithm_family != "path_cd_screen"
    assert edit.type == "full_source"


def test_proposer_emits_distinct_families_then_reemits():
    prop = AlgoPoolProposer()
    ctx = {"research_state": {}, "current_source": ""}
    n_families = len(SCREENING_FAMILIES) + 1 + len(fista_operator_grid())  # +champion base
    seen = []
    for _ in range(n_families + 2):
        e = prop.propose(ctx)
        assert e.algorithm_family, "every candidate must carry a dedup key"
        seen.append(e.algorithm_family)
    # screening families come first, each exactly once and distinct
    assert seen[: len(SCREENING_FAMILIES)] == list(SCREENING_FAMILIES)
    # the pool eventually re-emits (exhaustion) -> loop's duplicate-source guard
    assert len(seen) != len(set(seen))


# --- mutation layer: champion-centred one-knob search (folds PR #11) ---------
def test_fista_neighbors_move_exactly_one_knob():
    base = {"restart": True, "check_every": 10}
    for nb in algopool._fista_neighbors(base):
        diff = [k for k in base if base[k] != nb[k]]
        assert len(diff) == 1, f"{nb} differs from champion in {diff}, not one knob"


def test_proposer_expands_champion_neighbor_after_screening_exhausted():
    # With all screening families tried, and a champion FISTA config embedded in
    # current_source, the next proposal is a ONE-KNOB neighbour of that config.
    champ = {"restart": True, "check_every": 10}
    champ_src = render_fista_path(champ)
    rows = [_lineage_row(f) for f in SCREENING_FAMILIES]
    state = build_research_state(
        lineage_rows=rows, baseline_times={}, champion=None, kkt_tol=TOL,
    )
    prop = AlgoPoolProposer()
    edit = prop.propose({"research_state": state, "current_source": champ_src})
    assert edit.algorithm_family.startswith("fista_path_")
    # decode the emitted family back to a config and check one-knob adjacency
    emitted = {fista_config_family(c): c for c in fista_operator_grid()}[edit.algorithm_family]
    diff = [k for k in champ if champ[k] != emitted[k]]
    assert len(diff) == 1
