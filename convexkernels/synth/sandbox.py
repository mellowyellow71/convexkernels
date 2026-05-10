"""Subprocess sandbox for kernel evaluation.

Spawns `python -m convexkernels.synth._eval_kernel <run_dir>` with a timeout.
The eval script reads `<run_dir>/eval_config.json` and `<run_dir>/<problem.pkl>`,
writes `<run_dir>/result.json` (and `x.npy` on success).

Memory cap: we attempt to set `RLIMIT_AS` via `preexec_fn`. This works on
Linux but is unreliable on macOS — Mac users should rely on the timeout as
the primary guard rail. Documented in `tasks/results.md`.
"""

from __future__ import annotations

import json
import pickle
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SandboxResult:
    status: str  # completed | timeout | runtime_error | no_result
    iters: Optional[int] = None
    kkt_final: Optional[float] = None
    wall_time_s: Optional[float] = None
    setup_time_s: Optional[float] = None
    solve_time_s: Optional[float] = None
    single_solve_time_s: Optional[float] = None
    converged: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stderr_tail: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SandboxResult":
        return cls(
            status=d.get("status", "no_result"),
            iters=d.get("iters"),
            kkt_final=d.get("kkt_final"),
            wall_time_s=d.get("wall_time_s"),
            setup_time_s=d.get("setup_time_s"),
            solve_time_s=d.get("solve_time_s"),
            single_solve_time_s=d.get("single_solve_time_s"),
            converged=d.get("converged"),
            error_type=d.get("error_type"),
            error_message=d.get("error_message"),
            stderr_tail=d.get("traceback", ""),
        )


def _make_preexec(memlimit_mb: int):
    if sys.platform == "darwin":
        # Mac: RLIMIT_AS is unreliable; skip
        return None

    def _preexec() -> None:
        memlimit_bytes = memlimit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memlimit_bytes, memlimit_bytes))

    return _preexec


def write_eval_config(
    run_dir: Path,
    problem: Any,
    *,
    kernel_module: str,
    kernel_step: str = "fista_step",
    kernel_init: str = "init_state",
    variant: str = "basic",
    problem_backend: str = "native",
    problem_dtype: str = "fp32",
    dtype_strategy: str = "fp32",
    warmup_runs: int = 0,
    max_iters: int = 200,
    tol: float = 1e-6,
    algorithm: str = "fista",
) -> Path:
    """Pickle the problem and write the eval config to `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    problem_path = run_dir / "problem.pkl"
    with problem_path.open("wb") as f:
        pickle.dump(problem, f)

    config = {
        "kernel_module": kernel_module,
        "kernel_step": kernel_step,
        "kernel_init": kernel_init,
        "variant": variant,
        "problem_backend": problem_backend,
        "problem_dtype": problem_dtype,
        "dtype_strategy": dtype_strategy,
        "warmup_runs": warmup_runs,
        "max_iters": max_iters,
        "tol": tol,
        "algorithm": algorithm,
        "problem_pickle_path": str(problem_path),
    }
    config_path = run_dir / "eval_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def run_kernel(
    run_dir: Path,
    *,
    timeout_s: float = 30.0,
    memlimit_mb: int = 4096,
) -> SandboxResult:
    """Run `_eval_kernel` in a subprocess against `run_dir`'s config.

    Caller is responsible for writing `eval_config.json` and the problem pickle
    via `write_eval_config()` before calling.
    """
    run_dir = Path(run_dir)
    cmd = [
        sys.executable,
        "-m",
        "convexkernels.synth._eval_kernel",
        str(run_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            preexec_fn=_make_preexec(memlimit_mb),
        )
    except subprocess.TimeoutExpired as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        return SandboxResult(status="timeout", stderr_tail=stderr)

    result_path = run_dir / "result.json"
    if not result_path.exists():
        stderr = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        return SandboxResult(
            status="no_result",
            error_message=f"subprocess returncode={proc.returncode}",
            stderr_tail=stderr,
        )

    return SandboxResult.from_dict(json.loads(result_path.read_text()))
