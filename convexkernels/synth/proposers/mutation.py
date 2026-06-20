"""Deterministic mutation-operator proposer — the framework's LLM-free search.

This is what makes the loop a *framework* rather than a wrapper around repeated
LLM calls: it generates complete `solve()` candidates programmatically, by
rendering a parametric numpy-FISTA template over a grid of real algorithmic
operators, and drives them through the *same* trusted-KKT evaluation/gating/
checkpoint machinery the LLM proposers use. The loop therefore makes measurable,
KKT-gated progress with no model in the path.

Operators (the search space):
  - gradient    : "direct" (Aᵀ(Ay−b)) or "gram" (precompute G=AᵀA, c=Aᵀb; use
                  Gy−c — fewer bytes/iter on tall shapes; KKT-safe, exact in
                  fp64). The gram form is materialized as a `prepare_problem`
                  hook, timed as setup.
  - restart     : O'Donoghue–Candès gradient restart on/off (convergence rate).
  - check_every : how often to call `recorder.record`/`should_stop` (the
                  overhead vs. stop-granularity trade-off).

Search policy — champion-centred coordinate descent: expand the current
champion's one-knob neighbours first (read back from the `MUTATION_CONFIG`
marker the template embeds), then breadth over the untried grid. Configurations
already tried are skipped via the curated `research_state` family digest, so the
policy never repeats a direction.

Correctness: the candidate may transform `problem` however it likes (e.g. the
gram view), but the harness always recomputes the trusted KKT on the canonical
problem and re-checks the returned iterate — so every operator here is gated the
same way an LLM proposal is.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Optional

from ..loop import Edit

# The operator grid. Small on purpose: it is a real, enumerable search space the
# framework owns, not a stand-in for the open-ended LLM search.
_GRADIENTS: tuple[str, ...] = ("direct", "gram")
_RESTARTS: tuple[bool, ...] = (True, False)
_CHECK_EVERY: tuple[int, ...] = (5, 20)

_BASE_CONFIG = {"gradient": "direct", "restart": True, "check_every": 5}

_CONFIG_MARKER = "# MUTATION_CONFIG:"


def operator_grid() -> list[dict]:
    """All configurations in the operator space (cartesian product)."""
    grid = []
    for gradient, restart, check_every in itertools.product(
        _GRADIENTS, _RESTARTS, _CHECK_EVERY
    ):
        grid.append(
            {"gradient": gradient, "restart": restart, "check_every": check_every}
        )
    return grid


def config_family(cfg: dict) -> str:
    """Stable algorithm_family tag for a config (keys the dedup digest)."""
    return "fista_{}_{}_chk{}".format(
        cfg["gradient"],
        "restart" if cfg["restart"] else "norestart",
        cfg["check_every"],
    )


def render_source(cfg: dict) -> str:
    """Render a complete, importable `solve()` module for `cfg`."""
    gradient = cfg["gradient"]
    restart = bool(cfg["restart"])
    check_every = int(cfg["check_every"])

    if restart:
        momentum = (
            "        if float(np.dot(y - xn, xn - x)) > 0.0:\n"
            "            tn, mom = 1.0, 0.0\n"
            "        else:\n"
            "            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))\n"
            "            mom = (theta - 1.0) / tn"
        )
    else:
        momentum = (
            "        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))\n"
            "        mom = (theta - 1.0) / tn"
        )

    prepare = ""
    if gradient == "gram":
        prepare = (
            "\n\n"
            "def prepare_problem(problem):\n"
            "    # One-time Gram precompute (timed as setup): iterate on G y - c\n"
            "    # instead of Aᵀ(Ay-b). Exact in fp64, so KKT-equivalent.\n"
            "    A = np.asarray(problem.A); b = np.asarray(problem.b)\n"
            "    return _GramView(problem, A.T @ A, A.T @ b)\n\n\n"
            "class _GramView:\n"
            "    def __init__(self, base, G, c):\n"
            "        self.n = int(base.n); self.L = float(base.L)\n"
            "        self._base = base; self.G = G; self.c = c\n"
            "    def grad_smooth(self, y):\n"
            "        return self.G @ y - self.c\n"
            "    def prox(self, v, t):\n"
            "        return self._base.prox(v, t)\n"
        )

    header = f"{_CONFIG_MARKER} {json.dumps(cfg, sort_keys=True)}\n"
    solve = (
        "import numpy as np\n\n\n"
        "def solve(problem, recorder, *, kkt_tol, max_time_s):\n"
        "    n = problem.n\n"
        "    x = np.zeros(n); y = x.copy(); theta = 1.0\n"
        "    t = 1.0 / problem.L\n"
        f"    check_every = {check_every}\n"
        "    it = 0\n"
        "    while it < 200000:\n"
        "        it += 1\n"
        "        g = problem.grad_smooth(y)\n"
        "        xn = problem.prox(y - t * g, t)\n"
        f"{momentum}\n"
        "        y = xn + mom * (xn - x); x = xn; theta = tn\n"
        "        if it % check_every == 0:\n"
        "            recorder.record(x)\n"
        "            if recorder.should_stop(kkt_tol):\n"
        "                break\n"
        "    recorder.record(x)\n"
        "    return x\n"
    )
    return header + solve + prepare + "\n__all__ = ['solve']\n"


def _champion_config(ctx: dict) -> Optional[dict]:
    """Recover the champion's config from the MUTATION_CONFIG marker, if any."""
    src = ctx.get("current_source") or ""
    m = re.search(re.escape(_CONFIG_MARKER) + r"\s*(\{.*\})", src)
    if not m:
        return None
    try:
        cfg = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    if {"gradient", "restart", "check_every"} <= set(cfg):
        return cfg
    return None


def _tried_families(ctx: dict) -> set[str]:
    rs = ctx.get("research_state") or {}
    fams: set[str] = set()
    for d in rs.get("tried_directions") or []:
        fam = str(d.get("idea") or "").strip().lower()
        if fam:
            fams.add(fam)
    return fams


def _neighbors(cfg: dict) -> list[dict]:
    """One-knob mutations of `cfg` (coordinate neighbours)."""
    out: list[dict] = []
    for knob, alternatives in (
        ("gradient", _GRADIENTS),
        ("restart", _RESTARTS),
        ("check_every", _CHECK_EVERY),
    ):
        for value in alternatives:
            if cfg.get(knob) != value:
                nb = dict(cfg)
                nb[knob] = value
                out.append(nb)
    return out


class MutationProposer:
    """LLM-free proposer: programmatic operators + champion-centred search."""

    model = "mutation-grid"

    def __init__(self, grid: Optional[list[dict]] = None, base_config: Optional[dict] = None):
        self._grid = grid if grid is not None else operator_grid()
        self._base = dict(base_config) if base_config else dict(_BASE_CONFIG)

    def propose(self, ctx: dict) -> Edit:
        tried = _tried_families(ctx)
        champion = _champion_config(ctx) or self._base

        # 1) champion's one-knob neighbours first (coordinate descent)
        for cfg in _neighbors(champion):
            if config_family(cfg) not in tried:
                return self._edit(cfg)
        # 2) breadth over the untried remainder of the grid
        for cfg in self._grid:
            if config_family(cfg) not in tried:
                return self._edit(cfg)
        # 3) space exhausted — re-emit the champion; the loop's duplicate-source
        #    guard records it as discard:duplicate_source and moves on.
        return self._edit(champion)

    def _edit(self, cfg: dict) -> Edit:
        return Edit(
            type="full_source",
            rationale=(
                "deterministic mutation: gradient={gradient}, restart={restart}, "
                "check_every={check_every}".format(**cfg)
            ),
            full_source=render_source(cfg),
            proposer_role="impl",
            proposer_model=self.model,
            algorithm_family=config_family(cfg),
        )


__all__ = [
    "MutationProposer",
    "operator_grid",
    "config_family",
    "render_source",
]
