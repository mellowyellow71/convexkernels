"""Algorithm-pool proposer — path-native gap-safe screening + CD/FISTA, LLM-free.

The panel evidence (PR #14) says the binding constraint on the (time, gap)
plane is the *algorithm family*, not FISTA tuning: coordinate descent with
active-set screening owns the lower-left corner that the conic solvers and a
plain FISTA-Gram cannot reach. This proposer generates complete `solve()`
modules from the families that live there, and runs them under the standard
time-to-KKT champion gate (no Pareto machinery) — the fastest route to value.

The target specimen is the hero **`LassoPath`** (`m=1000, n=50000, K=50`): a
full regularization path over `K` decreasing lambdas, iterate `X` of shape
`(n, K)`. Every family here is path-native — it warm-starts down the path
(glmnet/Adelie homotopy) and screens per column — and also accepts a single
`Lasso` (viewed as a one-column path), so the same pool serves both specimens.

Families:
  path_cd_screen    : homotopy CD + strong-rule seed + gap-safe screening
                      (the Adelie-shaped candidate; CD pays O(m) per *surviving*
                      coordinate, and the working set stays near the support).
  path_fista_screen : union working-set + vectorized reduced FISTA + gap-safe
                      re-screen (no Python coordinate loop; the reduced step
                      bound is the full L, so no per-block SVD).
  fista_path_*      : parametric FISTA-path mutation layer (restart / check_every
                      knobs), champion-centered one-knob search. This folds in
                      the deterministic MutationProposer (PR #11) as a layer, so
                      there is one deterministic-proposer framework, not two.

Gap-safe screening (Fercoq-Gramfort-Salmon 2015): from any primal x for a
single lambda, the rescaled residual theta = s*(b - A x), s = min(1, lam /
||A^T(b - A x)||_inf), is dual-feasible, and the dual optimum lies in a ball of
radius rho = sqrt(2*gap) around theta (1-strong concavity of the dual). Any
feature with |A_j^T theta| + ||A_j|| * rho < lam is provably inactive and can be
dropped; the reduced problem has the same solution. The harness re-verifies the
trusted KKT on the full canonical problem, so a screening bug degrades the curve
(rejected candidate), never a silent wrong answer.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Optional

from ..loop import Edit

_HEADER = "import numpy as np\n\n\n"

# Shared path adapter + gap-safe screening, embedded verbatim in each candidate.
# `_as_path(problem)` returns (A, b, lambdas, is_single): a single `Lasso` is
# viewed as a one-column path so the same solver serves both specimens.
_PATH_PRELUDE = '''
def _as_path(problem):
    A = np.ascontiguousarray(np.asarray(problem.A, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(problem.b, dtype=np.float64))
    if hasattr(problem, "lambdas"):
        lambdas = np.asarray(problem.lambdas, dtype=np.float64)
        return A, b, lambdas, False
    lambdas = np.asarray([float(problem.lam)], dtype=np.float64)
    return A, b, lambdas, True


def _finish(X, is_single):
    return X[:, 0] if is_single else X


def _column_gap(r, b, x_l1, lam, abs_ATr):
    """Exact single-lambda duality gap at x (residual r = b - A x) and the dual
    rescale s for theta = s * r. `abs_ATr` = |A^T r| is reused from screening."""
    ATr_inf = float(np.max(abs_ATr)) if abs_ATr.size else 0.0
    s = 1.0 if ATr_inf <= lam else lam / ATr_inf
    P = 0.5 * float(r @ r) + lam * x_l1
    bmt = b - s * r
    D = 0.5 * float(b @ b) - 0.5 * float(bmt @ bmt)
    return max(P - D, 0.0), s


def _kkt_col(x, ATr, lam, L, lam_max):
    """Scale-free single-column KKT residual (the exact metric the loop gates).

    `ATr = A^T r` with r = b - A x, so g = A^T(A x - b) = -ATr. Matches
    `lasso_kkt_residual`: r_kkt = L x - soft(L x - g, lam), normalized by
    lam + ||A^T b||_inf. Stopping on this (not the duality gap) is what makes a
    candidate actually reach the gated tol — gap and KKT are different rulers.
    """
    z = L * x + ATr
    soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)
    return float(np.max(np.abs(L * x - soft))) / (lam + lam_max)
'''


# ---- path_cd_screen ---------------------------------------------------------
_PATH_CD_SCREEN = _HEADER + _PATH_PRELUDE + '''

def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Homotopy CD down the lambda path with gap-safe screening (Adelie-shaped).

    For each lambda (decreasing), warm-start from the previous column's support,
    seed a working set with the strong rule, and run cyclic CD (residual
    updates, O(m) per active coordinate) with a gap-safe re-screen each sweep.
    The working set contracts toward the true support, so CD never pays the
    O(n) per-sweep cost that kills coordinate descent on wide p >> n.
    """
    A, b, lambdas, is_single = _as_path(problem)
    n = A.shape[1]; K = int(lambdas.shape[0])
    AT = np.ascontiguousarray(A.T)                      # (n, m) rows = features
    col_sq = np.einsum("ij,ij->i", AT, AT)              # ||A_j||^2
    col_norm = np.sqrt(col_sq)
    valid = col_sq > 0.0
    inv_sq = np.where(valid, 1.0 / np.where(valid, col_sq, 1.0), 0.0)
    L = float(problem.L)
    lam_max = float(np.max(np.abs(AT @ b)))
    X = np.zeros((n, K))
    x = np.zeros(n)                                     # warm-start carrier
    supp = np.zeros(0, dtype=int)
    for k in range(K):
        lam = float(lambdas[k])
        r = b - A[:, supp] @ x[supp] if supp.size else b.copy()
        ATr = AT @ r
        lam_prev = float(lambdas[k - 1]) if k > 0 else float(np.max(np.abs(ATr)))
        keep = np.flatnonzero(valid & (np.abs(ATr) >= 2.0 * lam - lam_prev))
        if keep.size == 0:
            keep = np.flatnonzero(valid)
        # Each pass runs several cheap residual-based CD sweeps (only touch the
        # working set) then ONE full A^T r gemv for the KKT check + gap-safe
        # re-screen — instead of paying the O(n m) gemv every single sweep.
        for _ in range(600):
            active = keep[col_sq[keep] > 0.0]
            for _ in range(3):
                for j in active:
                    xj = x[j]
                    rho = float(AT[j] @ r) + col_sq[j] * xj
                    xn = np.sign(rho) * max(abs(rho) - lam, 0.0) * inv_sq[j]
                    if xn != xj:
                        r += AT[j] * (xj - xn)
                        x[j] = xn
            ATr = AT @ r                                # full gemv: KKT + screen
            if _kkt_col(x, ATr, lam, L, lam_max) <= kkt_tol:
                break
            # gap-safe screen: contract the working set toward the support
            abs_ATr = np.abs(ATr)
            gap, s = _column_gap(r, b, float(np.sum(np.abs(x))), lam, abs_ATr)
            new_keep = np.flatnonzero(s * abs_ATr + col_norm * np.sqrt(2.0 * gap) >= lam)
            if new_keep.size:
                keep = new_keep
        X[:, k] = x
        supp = np.flatnonzero(x != 0.0)
        recorder.record(_finish(X, is_single))
        if recorder.should_stop(kkt_tol):
            break
    recorder.record(_finish(X, is_single))
    return _finish(X, is_single)
'''


# ---- path_fista_screen ------------------------------------------------------
_PATH_FISTA_SCREEN = _HEADER + _PATH_PRELUDE + '''

def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Union working-set FISTA over the whole path (vectorized, gap-safe).

    Maintain one working set W = union of the per-column active features. Run
    batched FISTA on the reduced A_W over all K columns at once (per-column
    soft-threshold with lambda broadcast), then a gap-safe re-screen per column
    grows/shrinks W. No Python coordinate loop; the reduced FISTA step bound is
    the full problem's L (||A_W||_2 <= ||A||_2), so no per-block SVD.
    """
    A, b, lambdas, is_single = _as_path(problem)
    n = A.shape[1]; K = int(lambdas.shape[0])
    col_norm = np.sqrt(np.einsum("ij,ij->j", A, A))
    ATb = A.T @ b
    lam_max = float(np.max(np.abs(ATb)))
    L = float(problem.L); t = 1.0 / L
    # strong-rule seed per column, unioned
    strong = np.abs(ATb)[:, None] >= (2.0 * lambdas[None, :] - lam_max)
    W = np.flatnonzero(np.any(strong, axis=1))
    if W.size == 0:
        W = np.arange(n)
    X = np.zeros((n, K))
    for _ in range(2000):
        Aw = np.ascontiguousarray(A[:, W])
        Xw = X[W, :].copy(); Yw = Xw.copy(); theta = 1.0
        for _ in range(20):
            G = Aw.T @ (Aw @ Yw - b[:, None])           # (|W|, K) gradient
            V = Yw - t * G
            Xn = np.sign(V) * np.maximum(np.abs(V) - t * lambdas[None, :], 0.0)
            if float(np.sum((Yw - Xn) * (Xn - Xw))) > 0.0:
                tn, mom = 1.0, 0.0
            else:
                tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
                mom = (theta - 1.0) / tn
            Yw = Xn + mom * (Xn - Xw); Xw = Xn; theta = tn
        X = np.zeros((n, K)); X[W, :] = Xw
        recorder.record(_finish(X, is_single))
        # ---- gap-safe re-screen per column on the FULL problem ----
        R = b[:, None] - Aw @ Xw                         # (m, K) residuals
        absA_TR = np.abs(A.T @ R)                        # (n, K)
        ATr_inf = np.max(absA_TR, axis=0)                # (K,)
        s = np.where(ATr_inf <= lambdas, 1.0, lambdas / np.maximum(ATr_inf, 1e-300))
        P = 0.5 * np.einsum("ij,ij->j", R, R) + lambdas * np.sum(np.abs(Xw), axis=0)
        BmT = b[:, None] - s[None, :] * R
        D = 0.5 * float(b @ b) - 0.5 * np.einsum("ij,ij->j", BmT, BmT)
        gaps = np.maximum(P - D, 0.0)
        rho = np.sqrt(2.0 * gaps)
        survive = s[None, :] * absA_TR + col_norm[:, None] * rho[None, :] >= lambdas[None, :]
        new_W = np.flatnonzero(np.any(survive, axis=1))
        if new_W.size:
            W = new_W
        if recorder.should_stop(kkt_tol):
            break
    recorder.record(_finish(X, is_single))
    return _finish(X, is_single)
'''


# ---- FISTA-path mutation layer (folds in PR #11 MutationProposer) ------------
_CONFIG_MARKER = "# MUTATION_CONFIG:"
_RESTARTS: tuple[bool, ...] = (True, False)
_CHECK_EVERY: tuple[int, ...] = (10, 40)
_BASE_CONFIG = {"restart": True, "check_every": 10}


def fista_config_family(cfg: dict) -> str:
    return "fista_path_{}_chk{}".format(
        "restart" if cfg["restart"] else "norestart", cfg["check_every"]
    )


def render_fista_path(cfg: dict) -> str:
    """Render a complete batched FISTA-path `solve()` module for `cfg`.

    Wide p >> n makes the n x n Gram infeasible, so the gradient stays the
    matrix form A^T(A Y - b) (via `grad_smooth_path`) — no Gram knob. The
    searchable knobs are O'Donoghue-Candes restart and the record cadence.
    """
    restart = bool(cfg["restart"]); check_every = int(cfg["check_every"])
    if restart:
        momentum = (
            "        if float(np.sum((Y - Xn) * (Xn - X))) > 0.0:\n"
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
    header = f"{_CONFIG_MARKER} {json.dumps(cfg, sort_keys=True)}\n"
    body = (
        "import numpy as np\n\n\n"
        + _PATH_PRELUDE
        + "\n\ndef solve(problem, recorder, *, kkt_tol, max_time_s):\n"
        "    A, b, lambdas, is_single = _as_path(problem)\n"
        "    n = A.shape[1]; K = int(lambdas.shape[0])\n"
        "    L = float(problem.L); t = 1.0 / L\n"
        "    X = np.zeros((n, K)); Y = X.copy(); theta = 1.0\n"
        f"    check_every = {check_every}\n"
        "    it = 0\n"
        "    while it < 200000:\n"
        "        it += 1\n"
        "        G = A.T @ (A @ Y - b[:, None])\n"
        "        V = Y - t * G\n"
        "        Xn = np.sign(V) * np.maximum(np.abs(V) - t * lambdas[None, :], 0.0)\n"
        f"{momentum}\n"
        "        Y = Xn + mom * (Xn - X); X = Xn; theta = tn\n"
        "        if it % check_every == 0:\n"
        "            recorder.record(_finish(X, is_single))\n"
        "            if recorder.should_stop(kkt_tol):\n"
        "                break\n"
        "    recorder.record(_finish(X, is_single))\n"
        "    return _finish(X, is_single)\n"
    )
    return header + body + "\n__all__ = ['solve']\n"


def fista_operator_grid() -> list[dict]:
    return [
        {"restart": r, "check_every": c}
        for r, c in itertools.product(_RESTARTS, _CHECK_EVERY)
    ]


def _fista_neighbors(cfg: dict) -> list[dict]:
    out: list[dict] = []
    for knob, alts in (("restart", _RESTARTS), ("check_every", _CHECK_EVERY)):
        for v in alts:
            if cfg.get(knob) != v:
                nb = dict(cfg); nb[knob] = v
                out.append(nb)
    return out


def _champion_fista_config(ctx: dict) -> Optional[dict]:
    src = ctx.get("current_source") or ""
    m = re.search(re.escape(_CONFIG_MARKER) + r"\s*(\{.*\})", src)
    if not m:
        return None
    try:
        cfg = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    return cfg if {"restart", "check_every"} <= set(cfg) else None


# ---- the pool ---------------------------------------------------------------
# family name (== algorithm_family dedup key) -> source
SCREENING_FAMILIES: dict[str, str] = {
    "path_cd_screen": _PATH_CD_SCREEN,
    "path_fista_screen": _PATH_FISTA_SCREEN,
}


def _tried(ctx: dict) -> set[str]:
    """Families already tried, read from research_state's dedup digest.

    `research_state["tried_directions"][i]["idea"]` is `_idea_key(row)` which on
    master keys on `edit.algorithm_family` — so a family name emitted as
    `algorithm_family` round-trips here exactly.
    """
    rs = ctx.get("research_state") or {}
    out: set[str] = set()
    for d in rs.get("tried_directions") or []:
        idea = str(d.get("idea") or "").strip().lower()
        if idea:
            out.add(idea)
    return out


class AlgoPoolProposer:
    """Deterministic proposer over the path-native screening + FISTA pool.

    One framework, two layers: the screening families (breadth over algorithm
    space, ordered by expected plane-area gain) and the FISTA-path mutation
    layer folded in from the MutationProposer (champion-centered one-knob
    search). Each candidate carries a distinct `algorithm_family` — the real
    dedup key — so re-proposals are absorbed by research_state and, failing
    that, the loop's duplicate-source guard.
    """

    model = "algo-pool"

    def __init__(self, families: Optional[dict[str, str]] = None,
                 fista_base: Optional[dict] = None):
        self._families = dict(families) if families is not None else dict(SCREENING_FAMILIES)
        self._fista_base = dict(fista_base) if fista_base else dict(_BASE_CONFIG)
        self._emitted: set[str] = set()

    def propose(self, ctx: dict) -> Edit:
        tried = _tried(ctx) | self._emitted

        # 1) breadth over the screening families (the panel-pointed frontier)
        for name, source in self._families.items():
            if name not in tried:
                self._emitted.add(name)
                return self._edit_source(name, source)

        # 2) FISTA-path mutation layer: champion neighbours first, then breadth
        champion = _champion_fista_config(ctx) or self._fista_base
        for cfg in _fista_neighbors(champion) + fista_operator_grid():
            fam = fista_config_family(cfg)
            if fam not in tried:
                self._emitted.add(fam)
                return self._edit_source(fam, render_fista_path(cfg))

        # 3) pool exhausted — re-emit the first family; the loop's duplicate-
        #    source guard records it as a discard and the session winds down.
        first = next(iter(self._families))
        return self._edit_source(first, self._families[first])

    def _edit_source(self, family: str, source: str) -> Edit:
        return Edit(
            type="full_source",
            rationale=f"{family}: deterministic path-native algorithm candidate "
                      f"(gap-safe screening / FISTA pool)",
            full_source=source,
            proposer_role="impl",
            proposer_model=self.model,
            algorithm_family=family,
        )


__all__ = [
    "AlgoPoolProposer",
    "SCREENING_FAMILIES",
    "render_fista_path",
    "fista_config_family",
    "fista_operator_grid",
]
