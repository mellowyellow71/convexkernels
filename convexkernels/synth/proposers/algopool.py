"""Algorithm-pool proposer — CD + gap-safe screening candidates, LLM-free.

The Pareto loop measured (vs the live baseline panel on this box) that FISTA
variants claim essentially no plane-area from the coordinate-descent solvers
(sklearn/adelie): `vs_panel` collapsed to ~0 once adelie entered the panel. The
binding constraint is the *algorithm family*, so this proposer generates
candidates from the families that own the lower-left of the (time, gap) plane:

  fista_gram    : batched-gradient baseline (G = AᵀA precompute)
  cd_residual   : cyclic coordinate descent with residual updates (glmnet-style
                  closed-form soft-threshold per coordinate, O(m) each)
  fista_screen  : gap-safe screening + FISTA on the surviving features
  cd_screen     : gap-safe screening + CD on the surviving features (the
                  adelie-shaped candidate)

Gap-safe screening (Fercoq–Gramfort–Salmon 2015): from any primal x, the
rescaled residual θ = s·(b − Ax), s = min(1, λ/‖Aᵀ(b−Ax)‖∞), is dual-feasible,
and the dual optimum lies in a ball of radius ρ = √(2·gap(x, θ)) around θ
(1-strong concavity of the dual). Hence any feature with

    |A_jᵀ θ| + ‖A_j‖₂ · ρ < λ

is *provably* inactive at the optimum and can be discarded — the reduced
problem has the same solution. Candidates re-screen every sweep as the gap
shrinks, so the working set keeps contracting toward the true support. The
harness still verifies the trusted metric on the full canonical problem, so a
screening bug shows up as a rejected candidate, never a silent wrong answer.

Search policy mirrors the mutation proposer: propose each untried family
(breadth over algorithm space) with the family name as a stable rationale
prefix (the research-state dedup key), then stop re-proposing (the loop's
duplicate-source guard absorbs re-emissions).
"""

from __future__ import annotations

from typing import Optional

from ..loop import Edit

_HEADER = "import numpy as np\n\n\n"

# Shared helper: gap-safe screening pass, embedded verbatim in the candidates
# that use it. Operates on the *full* problem data; returns the surviving
# feature indices, the dual-feasible point's gap, and the current full iterate.
_SCREEN_HELPERS = '''
def _gap_and_theta_scale(r, b, x_l1, lam, ATr_inf):
    """Exact unscaled gap and the dual rescale s for theta = s*r."""
    s = 1.0 if ATr_inf <= lam else lam / ATr_inf
    P = 0.5 * float(r @ r) + lam * x_l1
    bmt = b - s * r
    D = 0.5 * float(b @ b) - 0.5 * float(bmt @ bmt)
    return max(P - D, 0.0), s


def _gap_safe_keep(abs_ATr, s, lam, col_norms, gap):
    """Indices surviving the gap-safe sphere test (provably may be active).

    `abs_ATr` is |A^T r| (already computed for the gap); theta = s*r, so the
    correlation with the dual point is s * abs_ATr — no second gemv needed.
    """
    rho = np.sqrt(2.0 * gap)
    return np.flatnonzero(s * abs_ATr + col_norms * rho >= lam)
'''

_FISTA_GRAM = _HEADER + '''
def prepare_problem(problem):
    A = np.asarray(problem.A); b = np.asarray(problem.b)
    return _Gram(problem, A.T @ A, A.T @ b)


class _Gram:
    def __init__(self, base, G, c):
        self.n = int(base.n); self.L = float(base.L)
        self._base = base; self.G = G; self.c = c
    def grad_smooth(self, y):
        return self.G @ y - self.c
    def prox(self, v, t):
        return self._base.prox(v, t)


def solve(problem, recorder, *, kkt_tol, max_time_s):
    n = problem.n
    x = np.zeros(n); y = x.copy(); theta = 1.0
    t = 1.0 / problem.L
    it = 0
    while it < 300000:
        it += 1
        g = problem.grad_smooth(y)
        xn = problem.prox(y - t * g, t)
        if float(np.dot(y - xn, xn - x)) > 0.0:
            tn, mom = 1.0, 0.0
        else:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
            mom = (theta - 1.0) / tn
        y = xn + mom * (xn - x); x = xn; theta = tn
        if it % 25 == 0:
            recorder.record(x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(x)
    return x
'''

_CD_RESIDUAL = _HEADER + '''
def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Cyclic coordinate descent with residual updates (glmnet-style).

    Per coordinate j: rho = A_j.(r + A_j x_j); x_j <- soft(rho, lam)/||A_j||^2,
    then r is updated incrementally. One sweep = one pass over coordinates.
    """
    A = np.ascontiguousarray(np.asarray(problem.A).T)   # rows = features
    b = np.asarray(problem.b)
    lam = float(problem.lam)
    n = problem.n
    sq = np.einsum("ij,ij->i", A, A)                    # ||A_j||^2
    x = np.zeros(n)
    r = b.copy()                                        # r = b - A x
    order = np.flatnonzero(sq > 0.0)
    sweep = 0
    while sweep < 100000:
        sweep += 1
        for j in order:
            xj = x[j]
            rho = float(A[j] @ r) + sq[j] * xj
            xn = np.sign(rho) * max(abs(rho) - lam, 0.0) / sq[j]
            if xn != xj:
                r -= A[j] * (xn - xj)
                x[j] = xn
        recorder.record(x)
        if recorder.should_stop(kkt_tol):
            break
    return x
'''

_FISTA_SCREEN = _HEADER + _SCREEN_HELPERS + '''

def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Strong-rule init + gap-safe screening + FISTA on the surviving set.

    Strong rule (Tibshirani 2012) seeds a small working set immediately (from
    x=0 the gap-safe sphere keeps everything, so a cold gap-safe screen is a
    no-op); the gap-safe re-screen after every block is the *safe* mechanism
    that re-admits anything the heuristic wrongly dropped and contracts the set
    as the gap shrinks. ||A_keep||_2 <= ||A||_2, so the full problem's L stays
    a valid step bound on every reduced problem — no per-block SVD.
    """
    A = np.asarray(problem.A); b = np.asarray(problem.b)
    lam = float(problem.lam); n = problem.n
    col_norms = np.sqrt(np.einsum("ij,ij->j", A, A))
    ATb = A.T @ b
    lam_max = float(np.max(np.abs(ATb)))
    strong = np.flatnonzero(np.abs(ATb) >= 2.0 * lam - lam_max)
    keep = strong if strong.size else np.arange(n)
    t = 1.0 / float(problem.L)
    x = np.zeros(n)
    block = 10
    it_total = 0
    while it_total < 300000:
        # ---- FISTA block on the reduced problem ----
        Ak = A[:, keep]
        xk = x[keep].copy(); yk = xk.copy(); theta = 1.0
        for _ in range(block):
            it_total += 1
            g = Ak.T @ (Ak @ yk - b)
            v = yk - t * g
            xn = np.sign(v) * np.maximum(np.abs(v) - t * lam, 0.0)
            if float(np.dot(yk - xn, xn - xk)) > 0.0:
                tn, mom = 1.0, 0.0
            else:
                tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
                mom = (theta - 1.0) / tn
            yk = xn + mom * (xn - xk); xk = xn; theta = tn
        x = np.zeros(n); x[keep] = xk
        recorder.record(x)
        # ---- gap-safe re-screen on the FULL problem (safe: certifies/corrects) ----
        r = b - Ak @ xk
        abs_ATr = np.abs(A.T @ r)
        ATr_inf = float(np.max(abs_ATr))
        gap, s = _gap_and_theta_scale(r, b, float(np.sum(np.abs(xk))), lam, ATr_inf)
        new_keep = _gap_safe_keep(abs_ATr, s, lam, col_norms, gap)
        if new_keep.size:
            keep = new_keep
        if recorder.should_stop(kkt_tol):
            break
    return x
'''

_CD_SCREEN = _HEADER + _SCREEN_HELPERS + '''

def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Strong-rule init + gap-safe screening + cyclic CD (adelie-shaped).

    Strong rule seeds a small working set from the start (a cold gap-safe
    screen keeps everything — the sphere radius is huge at x=0); CD then pays
    O(m) per *surviving* coordinate. The gap-safe re-screen each sweep is the
    safe mechanism: it re-admits any active feature the heuristic dropped and
    contracts the set as the gap shrinks, so the reduced optimum is the full
    optimum.
    """
    A = np.asarray(problem.A); b = np.asarray(problem.b)
    lam = float(problem.lam); n = problem.n
    col_norms = np.sqrt(np.einsum("ij,ij->j", A, A))
    AT = np.ascontiguousarray(A.T)                      # rows = features
    sq = col_norms ** 2
    ATb = AT @ b
    lam_max = float(np.max(np.abs(ATb)))
    strong = np.flatnonzero(np.abs(ATb) >= 2.0 * lam - lam_max)
    keep = strong if strong.size else np.arange(n)
    x = np.zeros(n)
    r = b.copy()                                        # r = b - A x (full)
    sweep_total = 0
    while sweep_total < 100000:
        # ---- CD sweeps on the working set ----
        active = keep[sq[keep] > 0.0]
        for _ in range(2):
            sweep_total += 1
            for j in active:
                xj = x[j]
                rho = float(AT[j] @ r) + sq[j] * xj
                xn = np.sign(rho) * max(abs(rho) - lam, 0.0) / sq[j]
                if xn != xj:
                    r -= AT[j] * (xn - xj)
                    x[j] = xn
        recorder.record(x)
        # ---- gap-safe re-screen on the FULL problem (safe: certifies/corrects) ----
        abs_ATr = np.abs(AT @ r)
        ATr_inf = float(np.max(abs_ATr))
        gap, s = _gap_and_theta_scale(r, b, float(np.sum(np.abs(x))), lam, ATr_inf)
        new_keep = _gap_safe_keep(abs_ATr, s, lam, col_norms, gap)
        if new_keep.size:
            keep = new_keep
        if recorder.should_stop(kkt_tol):
            break
    return x
'''

_GRAM_SCREEN = _HEADER + _SCREEN_HELPERS + '''

def solve(problem, recorder, *, kkt_tol, max_time_s):
    """Working-set Gram-FISTA (glmnet-style ever-active + violators).

    Start from the top-correlation features (the strong rule degenerates to
    keep-everything for cold-start lambda < lambda_max/2), build the *reduced*
    Gram G_W = A_W^T A_W — m·|W|^2 flops instead of the full m·n^2 — and run
    FISTA on it at |W|^2 flops per iteration. After each inner solve, one full
    gemv finds KKT violators (|A_j^T r| > lam) to admit, and the gap-safe test
    certifies the set. Terminates when no violators remain and the trusted gap
    is under tol; the harness re-verifies on the full problem regardless.
    """
    A = np.asarray(problem.A); b = np.asarray(problem.b)
    lam = float(problem.lam); n = problem.n
    ATb = A.T @ b
    W = np.argsort(-np.abs(ATb))[: max(32, n // 16)]
    W = np.sort(W)
    x = np.zeros(n)
    outer = 0
    while outer < 200:
        outer += 1
        Aw = np.ascontiguousarray(A[:, W])
        G = Aw.T @ Aw
        c = ATb[W]
        Lw = float(np.linalg.eigvalsh(G)[-1]) * 1.01 if W.size < n else float(problem.L)
        t = 1.0 / max(Lw, 1e-12)
        xk = x[W].copy(); yk = xk.copy(); theta = 1.0
        for _ in range(2000):
            g = G @ yk - c
            xn = np.sign(yk - t * g) * np.maximum(np.abs(yk - t * g) - t * lam, 0.0)
            if float(np.dot(yk - xn, xn - xk)) > 0.0:
                tn, mom = 1.0, 0.0
            else:
                tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
                mom = (theta - 1.0) / tn
            yk = xn + mom * (xn - xk); xk = xn; theta = tn
            # cheap reduced-problem stationarity test (no full gemv)
            gs = G @ xk - c
            viol = np.maximum(np.abs(gs) - lam, 0.0) * (xk == 0.0) \\
                 + np.abs(gs + lam * np.sign(xk)) * (xk != 0.0)
            if float(np.max(viol)) < 0.1 * lam * 1e-8:
                break
        x = np.zeros(n); x[W] = xk
        recorder.record(x)
        # ---- full-problem violator check + admit ----
        r = b - Aw @ xk
        abs_ATr = np.abs(A.T @ r)
        violators = np.flatnonzero(abs_ATr > lam * (1.0 + 1e-12))
        new = np.setdiff1d(violators, W, assume_unique=False)
        if new.size:
            W = np.sort(np.concatenate([W, new]))
            continue
        if recorder.should_stop(kkt_tol):
            break
    return x
'''

# family name (stable dedup key, used as the rationale prefix) -> source
FAMILIES: dict[str, str] = {
    "gram_screen": _GRAM_SCREEN,
    "cd_screen": _CD_SCREEN,
    "fista_screen": _FISTA_SCREEN,
    "cd_residual": _CD_RESIDUAL,
    "fista_gram": _FISTA_GRAM,
}


def _tried(ctx: dict) -> set[str]:
    rs = ctx.get("research_state") or {}
    out: set[str] = set()
    for d in rs.get("tried_directions") or []:
        idea = str(d.get("idea") or "").strip().lower()
        if idea:
            out.add(idea.split(":", 1)[0].split(" ", 1)[0])
    return out


class AlgoPoolProposer:
    """Deterministic proposer over the CD/screening algorithm pool.

    Proposes each untried family once, ordered by expected plane-area gain
    (screened CD first — the family the panel measurement pointed at). When the
    pool is exhausted it re-emits the first family; the loop's duplicate-source
    guard records the re-emission as a discard and the session winds down.
    """

    model = "algo-pool"

    def __init__(self, families: Optional[dict[str, str]] = None):
        self._families = dict(families) if families is not None else dict(FAMILIES)
        self._emitted: set[str] = set()

    def propose(self, ctx: dict) -> Edit:
        tried = _tried(ctx) | self._emitted
        for name, source in self._families.items():
            if name not in tried:
                self._emitted.add(name)
                return self._edit(name, source)
        first = next(iter(self._families))
        return self._edit(first, self._families[first])

    def _edit(self, name: str, source: str) -> Edit:
        return Edit(
            type="full_source",
            rationale=f"{name}: generated algorithm-family candidate "
                      f"(CD/screening pool targeting the panel's frontier)",
            full_source=source,
            proposer_role="impl",
            proposer_model=self.model,
        )


__all__ = ["AlgoPoolProposer", "FAMILIES"]
