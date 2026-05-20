"""Analyze a lineage.jsonl and emit a Karpathy-style results summary.

Reads one or more `lineage.jsonl` files and produces:
  - Header with slot + counts (total / kept / discarded / crashed)
  - Trajectory of accepted champions (id, gen, wall_ms, fitness, brief rationale)
  - Discard breakdown by reason
  - Best champion summary

Usage:
  python scripts/lineage_summary.py PATH [PATH ...]

Each PATH is either a `lineage.jsonl` file or a directory containing one.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _summarize(path: Path) -> None:
    rows = _load(path)
    if not rows:
        print(f"\n=== {path}\n(empty)")
        return

    slot = rows[0].get("slot", {})
    total = len(rows)
    kept = sum(1 for r in rows if (r.get("decision") or {}).get("accepted"))
    crash = sum(
        1 for r in rows if "crash" in (r.get("decision") or {}).get("reason", "")
    )
    discard = total - kept - crash
    reasons = Counter((r.get("decision") or {}).get("reason", "?") for r in rows)

    print(f"\n=== {path}")
    print(
        f"slot: {slot.get('problem_family')}/{slot.get('algorithm')} "
        f"on {slot.get('hardware')}/{slot.get('dtype')}"
    )
    print(f"total={total}  kept={kept}  discarded={discard}  crashed={crash}")
    print()

    print("trajectory of accepted champions (gen / iters / fitness / solve_ms / id):")
    print(f"  {'gen':>4}  {'iters':>5}  {'fitness':>10}  {'solve_ms':>8}  id")
    accepted = [r for r in rows if (r.get("decision") or {}).get("accepted")]
    best = None
    best_ms = float("inf")
    for r in accepted:
        s = r.get("score") or {}
        rid = (r.get("id") or "")[:8]
        gen = r.get("generation", "?")
        iters = s.get("iters", "?")
        fit = s.get("fitness_final")
        ms = s.get("solve_ms_median")
        fit_str = f"{fit:.2e}" if isinstance(fit, (int, float)) else "?"
        ms_str = f"{ms:.2f}" if isinstance(ms, (int, float)) else "?"
        print(f"  {gen:>4}  {iters:>5}  {fit_str:>10}  {ms_str:>8}  {rid}")
        if isinstance(ms, (int, float)) and ms < best_ms:
            best, best_ms = r, ms

    print()
    print("discard reasons:")
    for reason, cnt in reasons.most_common():
        print(f"  {cnt:>4}  {reason}")

    if best is not None:
        s = best.get("score") or {}
        rat = (best.get("edit") or {}).get("rationale", "")[:200].replace("\n", " ")
        print()
        print(f"best champion: id={best['id'][:8]} gen={best['generation']}")
        print(
            f"  wall_ms={s.get('solve_ms_median'):.2f} +/- {s.get('solve_ms_std'):.2f}  "
            f"iters={s.get('iters')}  fitness={s.get('fitness_final'):.2e}"
        )
        print(f"  rationale: {rat}")

    if accepted:
        first = accepted[0].get("score") or {}
        first_ms = first.get("solve_ms_median")
        if isinstance(first_ms, (int, float)) and isinstance(best_ms, (int, float)):
            print()
            print(
                f"speedup over first accepted candidate: "
                f"{first_ms / best_ms:.2f}x (first={first_ms:.2f} -> best={best_ms:.2f})"
            )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in argv[1:]:
        path = Path(arg)
        if path.is_dir():
            path = path / "lineage.jsonl"
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        _summarize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
