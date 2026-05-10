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

    from convexkernels.frontend.nonnegative_lasso import NonnegativeLasso
    from convexkernels.kernels.mlx.lib import LassoMLX, NonnegativeLassoMLX

    dtype_name = config.get("problem_dtype", "fp32")
    dtype = {
        "fp32": mx.float32,
        "fp16": mx.float16,
    }[dtype_name]
    if isinstance(prob, NonnegativeLasso):
        return NonnegativeLassoMLX.from_problem(prob, dtype=dtype)
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

        from convexkernels.algorithms.fista import fista
        fista_kwargs = {
            "max_iters": config.get("max_iters", 200),
            "tol": config.get("tol", 1e-6),
            "variant": config.get("variant", "basic"),
            "kernel_step": kernel_step,
            "kernel_init": kernel_init,
            "record_history": False,
        }
        for _ in range(config.get("warmup_runs", 0)):
            fista(prob, **fista_kwargs)

        res = fista(
            prob,
            **fista_kwargs,
        )
        solve_time_s = float(res.wall_time_s)
        single_solve_time_s = float(setup_time_s + solve_time_s)

        np.save(run_path / "x.npy", np.asarray(res.x))
        result = {
            "status": "completed",
            "iters": int(res.n_iters),
            "kkt_final": float(res.kkt_final),
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
