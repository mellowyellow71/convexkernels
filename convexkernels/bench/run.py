"""Bench runner — runs all baselines on all shapes, writes results."""

from __future__ import annotations

import argparse
from typing import Iterable

import polars as pl

from .baselines import ALL_BASELINES, BaselineResult
from .shapes import DEFAULT_SHAPES, ShapeSpec, make_synthetic_lasso


def run_one(spec: ShapeSpec, seed: int = 0) -> list[BaselineResult]:
    prob = make_synthetic_lasso(spec, seed=seed)
    results = []
    for runner in ALL_BASELINES:
        try:
            results.append(runner(prob))
        except Exception as e:
            print(f"  [error] {runner.__name__} on {spec.name}: {type(e).__name__}: {e}")
    return results


def to_dataframe(rows: Iterable[tuple[ShapeSpec, BaselineResult]]) -> pl.DataFrame:
    records = []
    for spec, r in rows:
        records.append({
            "shape": spec.name,
            "m": spec.m,
            "n": spec.n,
            "solver": r.name,
            "n_iters": r.n_iters,
            "wall_s": r.wall_time_s,
            "kkt_final": r.kkt_final,
            "primal_obj": r.primal_obj,
        })
    return pl.DataFrame(records)


def run_all(shapes: tuple[ShapeSpec, ...] = DEFAULT_SHAPES,
            seed: int = 0) -> pl.DataFrame:
    rows: list[tuple[ShapeSpec, BaselineResult]] = []
    for spec in shapes:
        print(f"\n=== {spec.name} (m={spec.m}, n={spec.n}) ===")
        for r in run_one(spec, seed=seed):
            print(f"  {r.name:>20} iters={str(r.n_iters):>6} "
                  f"wall={r.wall_time_s:.3f}s kkt={r.kkt_final:.2e} "
                  f"obj={r.primal_obj:.4f}")
            rows.append((spec, r))
    return to_dataframe(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape", type=str, default=None,
                        help="Run only one shape by name")
    args = parser.parse_args()

    shapes = DEFAULT_SHAPES
    if args.shape:
        shapes = tuple(s for s in DEFAULT_SHAPES if s.name == args.shape)
        if not shapes:
            print(f"unknown shape: {args.shape}; available: {[s.name for s in DEFAULT_SHAPES]}")
            return 2

    df = run_all(shapes, seed=args.seed)
    print("\n=== summary ===")
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
