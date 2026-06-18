"""Subprocess entry point for kernel evaluation (algorithm-open contract).

Invoked by `synth/sandbox.py` as:
    python -m convexkernels.synth._eval_kernel <run_dir>

The candidate kernel owns the *entire* solve and chooses its own algorithm. It
exposes a single entry point:

    def solve(problem, recorder, *, kkt_tol, max_time_s) -> X

and (optionally) `prepare_problem(problem[, config])` for one-time setup
(Gram precompute, device upload, ...). The harness:

  1. times setup (`_prepare_problem` + the candidate's `prepare_problem`),
  2. builds a `Recorder` bound to the *trusted* numpy frontend problem so the
     KKT metric cannot be faked,
  3. runs `solve(...)`,
  4. independently recomputes the trusted KKT on the returned iterate (the
     final anti-gaming gate),
  5. writes `result.json` with `time_to_kkt_s`, `kkt_final`, `reached_target`,
     the setup/solve split, and the full `(t, kkt)` trajectory.

There is no fixed-driver dispatch any more — algorithm choice is part of the
search space.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import json
import math
import pickle
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

from convexkernels.bench.metrics import trusted_kkt
from convexkernels.synth.recorder import Recorder


def _load_kernel_module(spec: str):
    """Load a kernel module by either dotted module path or .py file path."""
    if spec.endswith(".py"):
        loaded_spec = importlib.util.spec_from_file_location("synth_kernel", spec)
        if loaded_spec is None or loaded_spec.loader is None:
            raise ImportError(f"could not load kernel from path: {spec}")
        module = importlib.util.module_from_spec(loaded_spec)
        sys.modules[loaded_spec.name] = module
        loaded_spec.loader.exec_module(module)
        return module
    return importlib.import_module(spec)


def _prepare_problem(prob, config: dict):
    """Convert the canonical pickled problem to the requested eval backend."""
    if config.get("problem_backend", "native") != "mlx":
        return prob

    import mlx.core as mx

    from convexkernels.frontend.basis_pursuit import BasisPursuit
    from convexkernels.frontend.equality_qp import EqualityQP
    from convexkernels.frontend.lasso_admm import LassoAdmm
    from convexkernels.frontend.lasso_path import LassoPath
    from convexkernels.frontend.nonnegative_lasso import NonnegativeLasso
    from convexkernels.frontend.total_variation import TVDenoising1D
    from convexkernels.kernels.mlx.lib import (
        BasisPursuitMLX,
        EqualityQPMLX,
        LassoAdmmMLX,
        LassoMLX,
        LassoPathMLX,
        NonnegativeLassoMLX,
        TVDenoising1DMLX,
    )

    dtype_name = config.get("problem_dtype", "fp32")
    dtype = {"fp32": mx.float32, "fp16": mx.float16}[dtype_name]
    if isinstance(prob, NonnegativeLasso):
        return NonnegativeLassoMLX.from_problem(prob, dtype=dtype)
    if isinstance(prob, TVDenoising1D):
        return TVDenoising1DMLX.from_problem(prob, dtype=dtype)
    if isinstance(prob, EqualityQP):
        return EqualityQPMLX.from_problem(prob, dtype=dtype)
    if isinstance(prob, BasisPursuit):
        return BasisPursuitMLX.from_problem(prob, dtype=dtype)
    if isinstance(prob, LassoAdmm):
        return LassoAdmmMLX.from_problem(prob, dtype=dtype)
    if isinstance(prob, LassoPath):
        return LassoPathMLX.from_lasso_path(prob, dtype=dtype)
    return LassoMLX.from_lasso(prob, dtype=dtype)


def _call_prepare(kernel_module, prob, config):
    if not hasattr(kernel_module, "prepare_problem"):
        return prob
    fn = kernel_module.prepare_problem
    if len(inspect.signature(fn).parameters) >= 2:
        return fn(prob, config)
    return fn(prob)


def main(run_dir: str) -> int:
    run_path = Path(run_dir)
    config_path = run_path / "eval_config.json"
    result_path = run_path / "result.json"

    try:
        config = json.loads(config_path.read_text())
        with open(config["problem_pickle_path"], "rb") as f:
            trusted_problem = pickle.load(f)  # canonical numpy frontend problem

        kkt_tol = float(config.get("kkt_tol", config.get("tol", 1e-6)))
        max_time_s = float(config.get("max_time_s", 60.0))
        solve_name = config.get("kernel_solve", "solve")

        # ---- warm the accelerator before timing setup ----
        # The first mx.eval in a fresh subprocess pays a fixed ~1.5-2s Metal
        # context cold-init. That is a per-process artifact, identical for every
        # candidate and NOT paid by the in-process cvxpy baselines, so charging
        # it to a candidate's setup would dwarf the real algorithmic setup
        # (data upload, Gram precompute) on small/medium shapes. Force context
        # init here so the timed setup measures solver work, not device boot.
        if config.get("problem_backend") == "mlx":
            try:
                import mlx.core as mx  # type: ignore

                mx.eval(mx.array([0.0]) + 1.0)
            except Exception:
                pass

        # ---- setup (timed) ----
        setup_t0 = perf_counter()
        kernel_module = _load_kernel_module(config["kernel_module"])
        prob = _prepare_problem(trusted_problem, config)
        prob = _call_prepare(kernel_module, prob, config)
        setup_time_s = perf_counter() - setup_t0

        solve = getattr(kernel_module, solve_name)
        kkt_fn = functools.partial(trusted_kkt, trusted_problem)

        # ---- warmups (discarded) ----
        for _ in range(int(config.get("warmup_runs", 0))):
            w = Recorder(kkt_fn, max_time_s=max_time_s)
            solve(prob, w, kkt_tol=kkt_tol, max_time_s=max_time_s)

        # ---- timed solve ----
        rec = Recorder(kkt_fn, max_time_s=max_time_s)
        solve_t0 = perf_counter()
        X = solve(prob, rec, kkt_tol=kkt_tol, max_time_s=max_time_s)
        solve_time_s = perf_counter() - solve_t0

        # ---- trusted final verification (anti-gaming gate) ----
        X_host = Recorder._materialize(X)
        kkt_final = trusted_kkt(trusted_problem, X_host)
        np.save(run_path / "x.npy", X_host)

        trajectory = [[float(t), float(k)] for (t, k) in rec.trajectory]
        (run_path / "trajectory.json").write_text(json.dumps(trajectory))

        reached = bool(kkt_final <= kkt_tol)
        ttk = rec.time_to_kkt(kkt_tol)
        if reached and not math.isfinite(ttk):
            # converged at the final iterate but never sampled below tol
            ttk = solve_time_s

        result = {
            "status": "completed",
            "kkt_final": kkt_final,
            "reached_target": reached,
            "time_to_kkt_s": (ttk if math.isfinite(ttk) else None),
            "setup_time_s": float(setup_time_s),
            "solve_time_s": float(solve_time_s),
            "n_records": int(rec.n_records),
            "trajectory": trajectory,
        }
        result_path.write_text(json.dumps(result))
        return 0

    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "runtime_error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
        result_path.write_text(json.dumps(result))
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m convexkernels.synth._eval_kernel <run_dir>",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
