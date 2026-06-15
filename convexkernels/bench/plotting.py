"""KKT-residual-vs-time plot — the core empirical artifact.

Renders one log-y figure overlaying the autoresearch champion's trajectory and
every classical baseline's anytime curve on the *same* trusted-KKT ruler, with
a horizontal line at the target tolerance. A champion whose curve crosses the
target line to the *left* of the baselines is "converging faster than the
classical solvers" — the headline claim.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .curves import baseline_panel, problem_hash


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

    for name, curve in baselines.items():
        if not curve:
            continue
        ts = [t for t, _ in curve]
        ks = [k for _, k in curve]
        ax.plot(ts, ks, marker="o", ms=3, alpha=0.8, label=name)

    if champ is not None:
        traj = json.loads((champ.dir / "trajectory.json").read_text())
        score = json.loads((champ.dir / "score.json").read_text())
        setup = float(score.get("setup_s") or 0.0)
        if traj:
            ts = [setup + float(t) for t, _ in traj]
            ks = [float(k) for _, k in traj]
            ax.plot(ts, ks, color="black", lw=2.2, marker="s", ms=3,
                    label=f"champion ({champ.algorithm_tag})")

    ax.axhline(kkt_tol, color="red", ls="--", lw=1, label=f"target {kkt_tol:g}")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("trusted KKT residual")
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
    return out_path
