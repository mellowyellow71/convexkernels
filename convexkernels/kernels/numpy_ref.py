"""Reference NumPy kernel for FISTA.

The correctness oracle. Every future MLX kernel variant in P3+ is tested
against this via `assert_equivalent` (functional equivalence, KKT-gated).

The synth loop never benchmarks against this directly — it benchmarks against
`mx.matmul` baselines on the Mac.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frontend.problem import Problem


@dataclass
class FistaState:
    x: np.ndarray
    y: np.ndarray
    theta: float


def init_state(problem: Problem) -> FistaState:
    n = problem.n
    return FistaState(x=np.zeros(n), y=np.zeros(n), theta=1.0)


def fista_step(state: FistaState, problem: Problem, t: float) -> FistaState:
    """One FISTA iteration (Beck & Teboulle 2009)."""
    g = problem.grad_smooth(state.y)
    x_next = problem.prox(state.y - t * g, t)
    theta_next = (1.0 + np.sqrt(1.0 + 4.0 * state.theta * state.theta)) / 2.0
    momentum = (state.theta - 1.0) / theta_next
    y_next = x_next + momentum * (x_next - state.x)
    return FistaState(x=x_next, y=y_next, theta=theta_next)
