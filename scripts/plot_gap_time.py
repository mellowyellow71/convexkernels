"""CLI: render the KKT-vs-time figure for a finished/in-progress state root.

Usage:
    python scripts/plot_gap_time.py <state_root> \
        --problem-family lasso_path --shape path_tall_medium [--kkt-tol 1e-6]

The problem is reconstructed the same way the run did (so the problem hash, and
thus the cached baseline curves, match).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("state_root", type=Path)
    p.add_argument("--problem-family", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--kkt-tol", type=float, default=1e-6)
    args = p.parse_args(argv)

    # Import lazily so `--help` works without the full stack.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from convexkernels.bench.plotting import plot_state_root
    from convexkernels.synth.run import make_problem

    problem = make_problem(args.problem_family, args.shape)
    out = plot_state_root(args.state_root, problem, kkt_tol=args.kkt_tol)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
