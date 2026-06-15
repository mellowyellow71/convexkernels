"""Reference numpy `solve()` candidate under the algorithm-open contract.

This is the native-backend analog of the MLX seeds: a self-contained FISTA
(with O'Donoghue-Candes gradient restart) that owns its whole loop and reports
progress through the trusted `Recorder`. Used as a seed for `problem_backend=
native` runs and as the Linux contract-test candidate.

Contract:
    solve(problem, recorder, *, kkt_tol, max_time_s) -> x
"""

from __future__ import annotations

import numpy as np


def solve(problem, recorder, *, kkt_tol: float, max_time_s: float, check_every: int = 10):
    n = problem.n
    x = np.zeros(n)
    y = x.copy()
    theta = 1.0
    t = 1.0 / problem.L

    it = 0
    max_iters = 200000
    while it < max_iters:
        it += 1
        g = problem.grad_smooth(y)
        x_new = problem.prox(y - t * g, t)
        if float(np.dot(y - x_new, x_new - x)) > 0.0:
            theta_new = 1.0
            mom = 0.0
        else:
            theta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta * theta))
            mom = (theta - 1.0) / theta_new
        y = x_new + mom * (x_new - x)
        x = x_new
        theta = theta_new
        if it % check_every == 0:
            recorder.record(x)
            if recorder.should_stop(kkt_tol):
                break
    recorder.record(x)
    return x


__all__ = ["solve"]
