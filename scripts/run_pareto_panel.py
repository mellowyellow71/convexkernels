#!/usr/bin/env python3
"""Score a candidate solver against a baseline panel on the (time, gap) plane.

Builds anytime ``(wall_time_s, duality_gap)`` curves for each baseline solver
(cap-sweep — solve at growing iteration caps) and for a candidate FISTA-Gram
solver, then reports the **dominated-hypervolume advantage** of the candidate
over each baseline and over the combined panel (the multi-objective "how do we
do"). Optionally writes a log-scale plot (gap vs wall-clock).

    python scripts/run_pareto_panel.py --solvers SCS,ECOS --n 600 --plot out.png

Adelie is the headline bar but has no py3.13 wheel on Linux; inject a cached
Adelie curve with --adelie-npz, or run this on the Mac where adelie builds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from convexkernels.bench.baselines import build_cvxpy_lasso, solve_existing_cvxpy
from convexkernels.bench.metrics import trusted_gap
from convexkernels.bench.pareto import auto_nadir, score_against_panel
from convexkernels.frontend.lasso import Lasso


def make_lasso(m, n, k, lam_frac, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    xt = np.zeros(n)
    xt[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xt + 0.01 * rng.standard_normal(m)
    lam = lam_frac * float(np.max(np.abs(A.T @ b)))
    return Lasso(A, b, lam)


def cvxpy_gap_curve(prob, solver, sweep):
    try:
        cvxprob, xvar = build_cvxpy_lasso(prob)
    except Exception as e:
        print(f"  [{solver}] cvxpy build failed: {e}")
        return []
    pts = []
    for cap in sweep:
        try:
            x, wall = solve_existing_cvxpy(cvxprob, xvar, solver, max_iter=int(cap))
        except Exception:
            continue
        pts.append((wall, trusted_gap(prob, x)))
    return pts


def build_gap_panel(prob, solvers, sweep):
    """Anytime (wall_time_s, duality_gap) curves for each panel solver.

    Every member — cvxpy-backed and native alike — is built with the SAME
    ruler (`trusted_gap`); a panel must never mix rulers on one axis. The
    native solvers go through `baseline_kkt_time_curve(metric=trusted_gap)`,
    which recomputes the trusted gap on each capped solve's iterate.
    """
    from convexkernels.bench.curves import baseline_kkt_time_curve

    panel = {}
    for s in solvers:
        if s in ("sklearn", "adelie"):
            panel[s] = baseline_kkt_time_curve(prob, s, sweep, metric=trusted_gap)
        else:
            panel[s] = cvxpy_gap_curve(prob, s, sweep)
    return panel


def candidate_gram_fista_curve(prob, *, max_iters, check_every):
    """numpy FISTA-Gram candidate; returns (cumulative_time_s, gap) checkpoints."""
    A, b = prob.A, prob.b
    t0 = time.perf_counter()
    G = A.T @ A
    c = A.T @ b
    L = prob.L
    t = 1.0 / L
    n = prob.n
    x = np.zeros(n); y = x.copy(); th = 1.0
    overhead = 0.0
    pts = []
    for it in range(1, max_iters + 1):
        g = G @ y - c
        xn = prob.prox(y - t * g, t)
        if float(np.dot(y - xn, xn - x)) > 0.0:
            tn, mo = 1.0, 0.0
        else:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * th * th)); mo = (th - 1.0) / tn
        y = xn + mo * (xn - x); x = xn; th = tn
        if it % check_every == 0:
            o0 = time.perf_counter()
            gap = trusted_gap(prob, x)             # measurement, excluded from time
            overhead += time.perf_counter() - o0
            pts.append((time.perf_counter() - t0 - overhead, gap))
            if gap < 1e-12:
                break
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1500)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--lam-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solvers", default="SCS,ECOS,sklearn")
    ap.add_argument("--caps", default="50,100,200,500,1000,2000,5000")
    ap.add_argument("--cand-iters", type=int, default=5000)
    ap.add_argument("--cand-check-every", type=int, default=20)
    ap.add_argument("--adelie-npz", default=None, help="cached adelie (t,gap) curve .npz with arrays t,gap")
    ap.add_argument("--plot", default=None, help="path to write a gap-vs-time PNG")
    args = ap.parse_args()

    prob = make_lasso(args.m, args.n, args.k, args.lam_frac, args.seed)
    sweep = [int(c) for c in args.caps.split(",") if c]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    print(f"LASSO m={prob.m} n={prob.n} lam_frac={args.lam_frac}  panel={solvers}")

    panel = build_gap_panel(prob, solvers, sweep)
    for s, curve in panel.items():
        n_ok = len(curve)
        print(f"  baseline {s:9s}: {n_ok} curve points"
              + (f"  best_gap={min(g for _,g in curve):.1e}" if n_ok else "  (empty)"))
    if args.adelie_npz and Path(args.adelie_npz).exists():
        d = np.load(args.adelie_npz)
        panel["adelie"] = list(zip(d["t"].tolist(), d["gap"].tolist()))
        print(f"  baseline adelie   : {len(panel['adelie'])} cached points")

    cand = candidate_gram_fista_curve(prob, max_iters=args.cand_iters, check_every=args.cand_check_every)
    print(f"  candidate gram-fista: {len(cand)} points  best_gap={min(g for _,g in cand):.1e}")

    panel = {k: v for k, v in panel.items() if v}
    if not panel:
        print("no baseline curves available (install scs/ecos or pass --adelie-npz)")
        return 1

    s = score_against_panel(cand, panel)
    print(f"\nnadir (time_s, gap) = ({s['nadir'][0]:.3f}, {s['nadir'][1]:.2e})")
    print(f"hypervolume  candidate={s['hv_candidate']:.3f}  panel={s['hv_panel']:.3f}")
    print("dominated-hypervolume advantage of candidate over each baseline:")
    for name, adv in sorted(s["advantage_vs_solver"].items(), key=lambda kv: -kv[1]):
        print(f"  vs {name:9s}: {adv:+.3f}   {'WIN (claims area it does not)' if adv > 0 else '—'}")
    print(f"advantage vs COMBINED panel: {s['advantage_vs_panel']:+.3f}  "
          f"-> dominates_panel={s['dominates_panel']}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for name, c in panel.items():
                if c:
                    tt, gg = zip(*sorted(c))
                    ax.plot(tt, gg, "o-", label=name, alpha=0.8)
            tt, gg = zip(*sorted(cand))
            ax.plot(tt, gg, "s-", color="black", lw=2, label="candidate (gram-fista)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("wall-clock time (s)"); ax.set_ylabel("duality gap")
            ax.set_title(f"(time, gap) panel — advantage vs panel {s['advantage_vs_panel']:+.2f}")
            ax.legend(); ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout(); fig.savefig(args.plot, dpi=120)
            print(f"\nplot -> {args.plot}")
        except Exception as e:
            print(f"plot skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
