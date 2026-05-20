"""Reference numpy kernel for ALM / Augmented Lagrangian.

Sister of `numpy_ref.py` (FISTA) and `numpy_pdhg_ref.py` (PDHG) — the
correctness oracle for MLX ALM kernels. Re-exports the default numpy ALM
state and step from `convexkernels.algorithms.alm`.
"""

from __future__ import annotations

from typing import Any

from ..algorithms.alm import AlmState as _AlmState
from ..algorithms.alm import _default_numpy_init, _numpy_alm_step

AlmState = _AlmState


def init_state(problem: Any, **kwargs) -> AlmState:
    """ALM init. Accepts an optional ``rho`` kwarg from the host driver."""
    return _default_numpy_init(problem, rho=float(kwargs.get("rho", 1.0)))


def alm_step(state: AlmState, problem: Any) -> AlmState:
    """One ALM iteration (cached-Cholesky linear solve + multiplier update)."""
    return _numpy_alm_step(state, problem)
