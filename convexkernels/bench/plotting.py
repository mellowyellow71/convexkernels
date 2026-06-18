"""KKT-residual-vs-time plot — the core empirical artifact.

Renders one log-y figure overlaying the autoresearch champion's trajectory and
every classical baseline's anytime curve on the *same* trusted-KKT ruler, with
a horizontal line at the target tolerance. A champion whose curve crosses the
target line to the *left* of the baselines is "converging faster than the
classical solvers" — the headline claim.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from .curves import baseline_panel, problem_hash, time_to_kkt
from .metrics import time_to_target


def panel_summary(
    baselines: dict[str, list],
    kkt_tol: float,
    *,
    champion: Optional[dict] = None,
) -> dict:
    """Rank champion + every baseline by time-to-target on the one ruler.

    `champion` (optional) is `{"tag", "trajectory": [(t, kkt)], "setup_s"}` or
    `{"tag", "time_to_kkt_s"}`. Returns a ranking plus the champion's speedup
    over the *fastest baseline* — the headline number ("reach the target faster
    than the classical solvers"), made explicit instead of eyeballed off a plot.
    """
    rows: list[dict] = []
    for name, curve in baselines.items():
        rows.append({
            "method": name, "kind": "baseline",
            "time_to_kkt_s": time_to_kkt(curve, kkt_tol),
        })

    champ_time = float("inf")
    champ_tag = None
    if champion is not None:
        champ_tag = champion.get("tag", "champion")
        if champion.get("time_to_kkt_s") is not None:
            champ_time = float(champion["time_to_kkt_s"])
        elif champion.get("trajectory"):
            setup = float(champion.get("setup_s") or 0.0)
            traj = [(setup + float(t), float(k)) for t, k in champion["trajectory"]]
            champ_time = time_to_target(traj, kkt_tol)
        rows.append({
            "method": champ_tag, "kind": "champion", "time_to_kkt_s": champ_time,
        })

    rows.sort(key=lambda r: r["time_to_kkt_s"])

    best_baseline = min(
        (r for r in rows if r["kind"] == "baseline" and math.isfinite(r["time_to_kkt_s"])),
        key=lambda r: r["time_to_kkt_s"], default=None,
    )
    speedup = None
    beats = None
    if champion is not None and best_baseline is not None and math.isfinite(champ_time) and champ_time > 0:
        speedup = best_baseline["time_to_kkt_s"] / champ_time
        beats = champ_time < best_baseline["time_to_kkt_s"]

    return {
        "kkt_tol": kkt_tol,
        "ranking": rows,
        "champion": None if champion is None else {
            "tag": champ_tag, "time_to_kkt_s": champ_time,
        },
        "best_baseline": None if best_baseline is None else {
            "method": best_baseline["method"],
            "time_to_kkt_s": best_baseline["time_to_kkt_s"],
        },
        "champion_speedup_vs_best_baseline": speedup,
        "champion_beats_baselines": beats,
    }


def _fmt_t(t: float) -> str:
    return "inf" if not math.isfinite(t) else (f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.3f} s")


def render_summary_md(summary: dict) -> str:
    """One-screen markdown table of the ranking + the headline speedup line."""
    lines = [
        f"# Time-to-KKT panel (target {summary['kkt_tol']:g})",
        "",
        "| rank | method | kind | time to target |",
        "|---:|---|---|---:|",
    ]
    for i, r in enumerate(summary["ranking"], 1):
        lines.append(f"| {i} | {r['method']} | {r['kind']} | {_fmt_t(r['time_to_kkt_s'])} |")
    sp = summary.get("champion_speedup_vs_best_baseline")
    bb = summary.get("best_baseline") or {}
    if sp is not None:
        verdict = "faster than" if summary.get("champion_beats_baselines") else "slower than"
        lines += [
            "",
            f"**Champion is {sp:.2f}× {verdict} the best baseline "
            f"({bb.get('method')} @ {_fmt_t(bb.get('time_to_kkt_s', float('inf')))}).**",
        ]
    return "\n".join(lines) + "\n"


def _load_cached_baselines(state_root: Path, phash: str) -> dict[str, list]:
    bdir = state_root / "baselines" / phash
    out: dict[str, list] = {}
    if bdir.exists():
        for f in sorted(bdir.glob("*.json")):
            out[f.stem] = [tuple(p) for p in json.loads(f.read_text())]
    return out


def plot_state_root(
    state_root: Path,
    problem,
    *,
    kkt_tol: float = 1e-6,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Plot the best champion vs the baseline panel for `problem`.

    Returns the written PNG path, or None if there was nothing to plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..synth.checkpoints import CheckpointStore

    state_root = Path(state_root)
    phash = problem_hash(problem)

    baselines = _load_cached_baselines(state_root, phash)
    if not baselines:
        baselines = baseline_panel(problem, cache_dir=state_root / "baselines")

    store = CheckpointStore(state_root)
    champ = store.best(phash)

    fig, ax = plt.subplots(figsize=(7, 5))

    def _mark_crossing(curve, color):
        # Put a marker where the curve crosses the target line — the metric.
        tc = time_to_kkt(curve, kkt_tol)
        if math.isfinite(tc):
            ax.plot([tc], [kkt_tol], marker="v", ms=8, color=color, zorder=5)

    for name, curve in baselines.items():
        if not curve:
            continue
        ts = [t for t, _ in curve]
        ks = [k for _, k in curve]
        line, = ax.plot(ts, ks, marker="o", ms=3, alpha=0.8, label=name)
        _mark_crossing(curve, line.get_color())

    champion = None
    if champ is not None:
        traj = json.loads((champ.dir / "trajectory.json").read_text())
        score = json.loads((champ.dir / "score.json").read_text())
        setup = float(score.get("setup_s") or 0.0)
        champion = {"tag": champ.algorithm_tag, "trajectory": traj, "setup_s": setup}
        if traj:
            ts = [setup + float(t) for t, _ in traj]
            ks = [float(k) for _, k in traj]
            ax.plot(ts, ks, color="black", lw=2.2, marker="s", ms=3,
                    label=f"champion ({champ.algorithm_tag})")
            _mark_crossing([(setup + float(t), float(k)) for t, k in traj], "black")

    summary = panel_summary(baselines, kkt_tol, champion=champion)

    ax.axhline(kkt_tol, color="red", ls="--", lw=1, label=f"target {kkt_tol:g}")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("trusted KKT residual")
    sp = summary.get("champion_speedup_vs_best_baseline")
    bb = summary.get("best_baseline") or {}
    if sp is not None:
        verdict = "faster" if summary.get("champion_beats_baselines") else "SLOWER"
        ax.set_title(
            f"KKT vs time — champion {sp:.2f}× {verdict} than best baseline "
            f"({bb.get('method')})"
        )
    else:
        ax.set_title("Optimality (KKT) vs time")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if out_path is None:
        plots = state_root / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = plots / f"{phash}_{ts_tag}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # Machine- and human-readable headline next to the figure.
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    out_path.with_suffix(".summary.md").write_text(render_summary_md(summary))
    return out_path
