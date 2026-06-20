"""Tests for the Director/Analyst strategy layer over the experiment tree.

Stub-driven and Linux-runnable (native numpy `solve()` contract, no LLM, no MLX).
Covers: backward-compat (--director off == greedy), branch-point selection via the
CheckpointStore tree, the directive never weakening the gate, stop/invalid-branch
handling, director-failure degradation, the bounded tree summary, and the advisory
(never-gating) Analyst. Plus parse tests for OpenAIDirector/OpenAIAnalyst with
injected fake clients.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.lineage import Slot
from convexkernels.synth.loop import run_synth_loop
from convexkernels.synth.proposers.stub import StubProposer
from convexkernels.synth.checkpoints import CheckpointStore
from convexkernels.synth.director import (
    CHAMPION, Directive, StubDirector, OpenAIDirector, coerce_directive,
)
from convexkernels.synth.analyst import StubAnalyst, OpenAIAnalyst
from convexkernels.synth.research_state import build_tree_summary


# ---- fixtures / helpers ----


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((120, 60))
    x_true = rng.standard_normal(60) * (rng.random(60) < 0.1)
    b = A @ x_true + 1e-2 * rng.standard_normal(120)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def _slot() -> Slot:
    return Slot("lasso", "open", "linux_x86", "open")


def _good(tag: str = "") -> str:
    # A valid FISTA solve; `tag` only perturbs the source so hashes differ.
    return f'''
import numpy as np
def solve(problem, recorder, *, kkt_tol, max_time_s):
    # variant: {tag}
    x = np.zeros(problem.n); y = x.copy(); theta = 1.0
    t = 1.0 / problem.L
    for it in range(1, 100000):
        g = problem.grad_smooth(y)
        xn = problem.prox(y - t*g, t)
        if float(np.dot(y - xn, xn - x)) > 0.0:
            tn, mom = 1.0, 0.0
        else:
            tn = 0.5*(1.0 + np.sqrt(1.0 + 4.0*theta*theta)); mom = (theta-1.0)/tn
        y = xn + mom*(xn - x); x = xn; theta = tn
        if it % 5 == 0:
            recorder.record(x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(x)
    return x
'''


_SLOW = '''
import time
import numpy as np
def solve(problem, recorder, *, kkt_tol, max_time_s):
    time.sleep(0.3)             # >> any tiny-problem champion, even at a 10x bar
    x = np.zeros(problem.n); y = x.copy(); theta = 1.0
    t = 1.0 / problem.L
    for it in range(1, 100000):
        g = problem.grad_smooth(y)
        xn = problem.prox(y - t*g, t)
        if float(np.dot(y - xn, xn - x)) > 0.0:
            tn, mom = 1.0, 0.0
        else:
            tn = 0.5*(1.0 + np.sqrt(1.0 + 4.0*theta*theta)); mom = (theta-1.0)/tn
        y = xn + mom*(xn - x); x = xn; theta = tn
        if it % 5 == 0:
            recorder.record(x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(x)
    return x
'''


def _kwargs(tmp_path: Path, proposer, **over):
    base = dict(
        proposer=proposer,
        seed_kernel={"module": "convexkernels.kernels.numpy_solve_ref"},
        slot=_slot(),
        state_root=tmp_path,
        problem_backend="native",
        kkt_tol=1e-6,
        max_time_s=20.0,
        reps=1,
        warmup_runs=0,
        compute_baselines=False,
        verbose=False,
    )
    base.update(over)
    return base


class ScriptedDirector:
    """Returns directives from a callable `fn(call_idx, state) -> Directive`."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def direct(self, state: dict) -> Directive:
        d = self._fn(self.calls, state)
        self.calls += 1
        return d


class ScriptedAnalyst:
    def __init__(self, note: str):
        self._note = note

    def analyze(self, candidate_ctx: dict) -> str:
        return self._note


def _seed_id(store: CheckpointStore) -> str:
    return next(c.id for c in store.all() if c.algorithm_tag == "seed")


# ---- 1. backward compat: director off == greedy champion-chain ----


def test_director_off_reproduces_greedy(tmp_path, tiny_lasso):
    proposer = StubProposer([("a", _good("a")), ("b", _good("b"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=2, margin=10.0,
        director=StubDirector(), **_kwargs(tmp_path, proposer),
    )
    store = CheckpointStore(tmp_path)
    seed = _seed_id(store)
    assert all(r.decision["accepted"] for r in rows)
    # greedy chain: first branches from seed, second from the first accept.
    assert rows[0].parent_id == seed
    assert rows[1].parent_id == rows[0].id


# ---- 2. branch selection routes through CheckpointStore ----


def test_director_branch_selection(tmp_path, tiny_lasso):
    # On the 3rd proposal, branch from the seed (not the champion) → a non-linear
    # tree where the seed has two children.
    def fn(call_idx, state):
        if call_idx == 2:
            sid = next(n["id"] for n in state["tree_summary"]["nodes"]
                       if n["algorithm_tag"] == "seed")
            return Directive(branch_from=sid, direction="x", signal="backtrack")
        return Directive()

    proposer = StubProposer([("a", _good("a")), ("b", _good("b")), ("c", _good("c"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=3, margin=10.0,
        director=ScriptedDirector(fn), **_kwargs(tmp_path, proposer),
    )
    store = CheckpointStore(tmp_path)
    seed = _seed_id(store)
    assert rows[2].parent_id == seed          # branched from seed, not champion
    assert rows[2].parent_id != rows[1].id    # genuinely non-greedy
    # seed now has >= 2 children in the tree (rows[0] and rows[2]).
    summ = build_tree_summary(store.all(), problem_hash=store.all()[0].problem_hash)
    seed_row = next(n for n in summ["nodes"] if n["id"] == seed)
    assert seed_row["n_children"] >= 2


# ---- 3. directive cannot weaken the gate ----


def test_directive_does_not_weaken_gate(tmp_path, tiny_lasso):
    # Accept a fast champion, then direct a branch from the seed with a SLOW
    # candidate — it must still be rejected for not beating the champion.
    def fn(call_idx, state):
        if call_idx == 1:
            sid = next(n["id"] for n in state["tree_summary"]["nodes"]
                       if n["algorithm_tag"] == "seed")
            return Directive(branch_from=sid, direction="slow", signal="backtrack")
        return Directive()

    # margin=10.0 makes proposal-1 reliably accept (a same-speed solve clears a
    # 10x bar despite sub-ms timing noise), while the +300ms _SLOW candidate is
    # far over even a 10x bar — so the rejection is the gate's doing, not luck.
    proposer = StubProposer([("fast", _good("fast")), ("slow", _SLOW)])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=2, margin=10.0,
        director=ScriptedDirector(fn), **_kwargs(tmp_path, proposer),
    )
    store = CheckpointStore(tmp_path)
    assert rows[0].decision["accepted"] is True
    assert rows[1].decision["accepted"] is False
    assert "not_faster_than_champion" in rows[1].decision["reason"]
    assert rows[1].parent_id == _seed_id(store)   # the branch DID happen


# ---- 4. stop signal halts early ----


def test_director_stop_signal(tmp_path, tiny_lasso):
    director = ScriptedDirector(lambda i, s: Directive(signal="stop"))
    proposer = StubProposer([("a", _good("a"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=5,
        director=director, **_kwargs(tmp_path, proposer),
    )
    assert rows == []   # stopped before any proposal


# ---- 5. invalid branch id coerces to champion (no crash) ----


def test_director_invalid_branch_coerces_to_champion(tmp_path, tiny_lasso):
    director = ScriptedDirector(
        lambda i, s: Directive(branch_from="does-not-exist", direction="x"))
    proposer = StubProposer([("a", _good("a"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=1, margin=10.0,
        director=director, **_kwargs(tmp_path, proposer),
    )
    store = CheckpointStore(tmp_path)
    assert len(rows) == 1
    assert rows[0].parent_id == _seed_id(store)   # fell back to champion(seed)


# ---- 6. director failure degrades to greedy ----


def test_director_failure_degrades(tmp_path, tiny_lasso):
    class Boom:
        def direct(self, state):
            raise RuntimeError("director down")

    proposer = StubProposer([("a", _good("a")), ("b", _good("b"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=2, margin=10.0,
        director=Boom(), **_kwargs(tmp_path, proposer),
    )
    store = CheckpointStore(tmp_path)
    seed = _seed_id(store)
    assert rows[0].parent_id == seed             # degraded to champion chain
    assert rows[1].parent_id == rows[0].id


# ---- 7. tree summary is bounded and well-formed ----


def test_build_tree_summary_bounded():
    cps = []
    for i in range(40):
        cps.append(SimpleNamespace(
            id=f"n{i}", parent_id=(None if i == 0 else "n0"),
            algorithm_tag=("seed" if i == 0 else "full_source"),
            total_time_s=float(40 - i), time_to_kkt_s=1.0, kkt_final=1e-7,
            problem_hash="H",
        ))
    cps.append(SimpleNamespace(id="other", parent_id=None, algorithm_tag="seed",
                               total_time_s=1.0, time_to_kkt_s=1.0, kkt_final=1e-7,
                               problem_hash="OTHER"))
    summ = build_tree_summary(cps, problem_hash="H", max_nodes=10)
    nodes = summ["nodes"]
    assert len(nodes) <= 10
    ids = {n["id"] for n in nodes}
    assert "n0" in ids                 # root always kept
    assert "other" not in ids          # filtered by problem_hash
    assert all({"id", "parent_id", "algorithm_tag", "n_children"} <= set(n) for n in nodes)
    root = next(n for n in nodes if n["id"] == "n0")
    assert root["n_children"] >= 1


# ---- 8. analyst is advisory only ----


def test_analyst_note_is_advisory(tmp_path, tiny_lasso):
    # proposal1 accept; proposal2 SLOW -> rejected -> analyst note; proposal3 accept.
    # The note must reach research_state but must NOT change any decision.
    proposer = StubProposer([("a", _good("a")), ("slow", _SLOW), ("c", _good("c"))])
    rows = run_synth_loop(
        problem=tiny_lasso, n_proposals=3, margin=10.0,
        director=StubDirector(), analyst=ScriptedAnalyst("ADVISORY: reject everything"),
        **_kwargs(tmp_path, proposer),
    )
    assert rows[0].decision["accepted"] is True
    assert rows[1].decision["accepted"] is False    # SLOW rejected by the gate
    assert rows[2].decision["accepted"] is True      # gate unaffected by the note
    state = json.loads((tmp_path / "research_state.json").read_text())
    assert any("ADVISORY" in n for n in state.get("analyst_notes", []))


# ---- 9. OpenAIDirector / OpenAIAnalyst parse with injected clients ----


def _fake_client(payload: dict):
    resp = SimpleNamespace(output_text=json.dumps(payload))
    create = lambda **kw: resp  # noqa: E731
    return SimpleNamespace(responses=SimpleNamespace(create=create))


def test_openai_director_parses_directive():
    client = _fake_client({
        "branch_from": "champion", "direction": "warm-start across path",
        "algorithm_family_hint": "warmstart", "rationale": "because",
        "signal": "explore", "saturated": False,
    })
    d = OpenAIDirector(client=client).direct({"research_state": {}, "tree_summary": {"nodes": []}})
    assert isinstance(d, Directive)
    assert d.direction == "warm-start across path"
    assert d.signal == "explore"
    assert d.branch_from == CHAMPION


def test_openai_director_coerces_unknown_branch_and_signal():
    # unknown branch id (not in tree) and bad signal -> coerced
    state = {"research_state": {}, "tree_summary": {"nodes": [{"id": "real1"}]}}
    raw = {"branch_from": "ghost", "direction": "d", "algorithm_family_hint": "",
           "rationale": "", "signal": "teleport", "saturated": False}
    d = coerce_directive(raw, state)
    assert d.branch_from == CHAMPION
    assert d.signal == "exploit"


def test_openai_analyst_parses_summary():
    client = _fake_client({"summary": "plateaued at kkt~2e-3 (fp32 floor)"})
    note = OpenAIAnalyst(client=client).analyze({"score": {}})
    assert "plateaued" in note


def test_stub_director_and_analyst_are_noops():
    assert StubDirector().direct({}).is_default()
    assert StubAnalyst().analyze({}) == ""
