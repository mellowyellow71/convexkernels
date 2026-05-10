"""Subprocess entry point for kernel evaluation.

Invoked by `synth/sandbox.py` as:
    python -m convexkernels.synth._eval_kernel <run_dir>

Reads:
  <run_dir>/eval_config.json   # {kernel_module, kernel_step, kernel_init,
                                  variant, max_iters, tol, problem_pickle_path}
  <problem_pickle_path>        # pickled Lasso (or compatible Problem)

Writes (on success):
  <run_dir>/result.json        # {status: 'completed', iters, kkt_final,
                                  wall_time_s, converged}
  <run_dir>/x.npy              # final iterate

On error: writes <run_dir>/result.json with status != 'completed' and stderr-style detail.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import pickle
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np


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
    from convexkernels.frontend.nonnegative_lasso import NonnegativeLasso
    from convexkernels.frontend.total_variation import TVDenoising1D
    from convexkernels.kernels.mlx.lib import (
        BasisPursuitMLX,
        EqualityQPMLX,
        LassoAdmmMLX,
        LassoMLX,
        NonnegativeLassoMLX,
        TVDenoising1DMLX,
    )

    dtype_name = config.get("problem_dtype", "fp32")
    dtype = {
        "fp32": mx.float32,
        "fp16": mx.float16,
    }[dtype_name]
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
    return LassoMLX.from_lasso(prob, dtype=dtype)


def main(run_dir: str) -> int:
    run_path = Path(run_dir)
    config_path = run_path / "eval_config.json"
    result_path = run_path / "result.json"

    try:
        config = json.loads(config_path.read_text())
        with open(config["problem_pickle_path"], "rb") as f:
            prob = pickle.load(f)
        setup_t0 = perf_counter()
        kernel_module = _load_kernel_module(config["kernel_module"])
        prob = _prepare_problem(prob, config)
        if hasattr(kernel_module, "prepare_problem"):
            prepare_problem = kernel_module.prepare_problem
            if len(inspect.signature(prepare_problem).parameters) >= 2:
                prob = prepare_problem(prob, config)
            else:
                prob = prepare_problem(prob)
        setup_time_s = perf_counter() - setup_t0

        kernel_step = getattr(kernel_module, config["kernel_step"])
        kernel_init = getattr(kernel_module, config["kernel_init"])

        algorithm = config.get("algorithm", "fista")
        # Record history (= trajectory of KKT or gap) for the final timed run
        # so the proposer can see *where* convergence stalled or diverged.
        # convergence_check_every reduces per-iter GPU sync overhead from
        # checking the convergence criterion every iteration; defaults to 10
        # for sandbox runs, can be overridden via eval_config.
        algo_kwargs = {
            "max_iters": config.get("max_iters", 200),
            "tol": config.get("tol", 1e-6),
            "variant": config.get("variant", "basic"),
            "kernel_step": kernel_step,
            "kernel_init": kernel_init,
            "record_history": True,
            "convergence_check_every": int(config.get("convergence_check_every", 10)),
        }
        if algorithm == "fista":
            from convexkernels.algorithms.fista import fista as _driver
            fitness_attr = "kkt_final"
        elif algorithm == "pdhg":
            from convexkernels.algorithms.pdhg import pdhg as _driver
            fitness_attr = "gap_final"
        elif algorithm == "alm":
            from convexkernels.algorithms.alm import alm as _driver
            # ALM reports primal+dual residuals; we surface primal_res_final
            # as the gating fitness (it is the constraint-violation magnitude
            # that goes to zero at the optimum and serves as the
            # KKT-equivalent oracle-free signal for equality-constrained QP).
            fitness_attr = "primal_res_final"
        elif algorithm == "admm":
            from convexkernels.algorithms.admm import admm as _driver
            # ADMM gates on max(primal_res, dual_res) — both must be small.
            fitness_attr = "primal_res_final"
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}; expected fista, pdhg, alm, or admm")

        for _ in range(config.get("warmup_runs", 0)):
            _driver(prob, **algo_kwargs)

        res = _driver(prob, **algo_kwargs)
        solve_time_s = float(res.wall_time_s)
        single_solve_time_s = float(setup_time_s + solve_time_s)

        np.save(run_path / "x.npy", np.asarray(res.x))

        # Compress the per-iter fitness trajectory and write alongside the
        # result so the synth loop's proposer-context builder can read it
        # cheaply. The full history can be huge (thousands of iters); only
        # the compressed series goes to the LLM prompt.
        history = getattr(res, "history", {}) or {}
        traj_key = {"fista": "kkt", "pdhg": "gap", "alm": "primal_res", "admm": "primal_res"}.get(algorithm, "kkt")
        traj = list(history.get(traj_key, []))
        if traj:
            (run_path / "trajectory.json").write_text(json.dumps(traj))

        # KKT/gap reported as kkt_final regardless of algorithm so the synth
        # loop's gating logic stays uniform. Algorithms that produce a primal-
        # dual gap rather than a KKT residual write it to the same field;
        # acceptance gates apply identically.
        fitness_value = float(getattr(res, fitness_attr))
        result = {
            "status": "completed",
            "iters": int(res.n_iters),
            "kkt_final": fitness_value,
            "fitness_kind": "gap" if algorithm == "pdhg" else "kkt",
            "wall_time_s": single_solve_time_s,
            "setup_time_s": float(setup_time_s),
            "solve_time_s": solve_time_s,
            "single_solve_time_s": single_solve_time_s,
            "converged": bool(res.converged),
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
