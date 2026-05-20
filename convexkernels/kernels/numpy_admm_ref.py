"""Reference numpy kernel for ADMM. Sister of `numpy_alm_ref.py`."""

from __future__ import annotations

from typing import Any

from ..algorithms.admm import AdmmState as _AdmmState
from ..algorithms.admm import _default_numpy_init, _numpy_admm_step

AdmmState = _AdmmState


def init_state(problem: Any, **kwargs) -> AdmmState:
    return _default_numpy_init(problem, rho=float(kwargs.get("rho", 1.0)))


def admm_step(state: AdmmState, problem: Any) -> AdmmState:
    return _numpy_admm_step(state, problem)
