"""Trusted trajectory recorder handed to every candidate `solve()`.

The candidate owns the algorithm and decides *when* to checkpoint, but it never
computes the optimality metric itself: `Recorder.record(x)` materializes the
iterate and evaluates the *trusted* KKT function (bound by the harness to the
canonical numpy frontend problem). The recorder produces the `(t, kkt)`
trajectory the loop scores `time_to_kkt` against.

Timing contract:
  - `t0` is set at the start of the solve (excludes one-time setup, which the
    harness times separately as `setup_time_s`).
  - The cost of the trusted KKT evaluation itself is *excluded* from the
    reported timestamps (it is measurement instrumentation, not algorithm
    work) by accumulating it into `_overhead`. Forcing the iterate to
    materialize (an MLX GPU sync) *is* counted — that work is the algorithm's.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from ..bench.metrics import time_to_target


class Recorder:
    def __init__(
        self,
        kkt_fn: Callable[[np.ndarray], float],
        *,
        max_time_s: float = float("inf"),
        t0: float | None = None,
    ):
        self._kkt_fn = kkt_fn
        self.max_time_s = float(max_time_s)
        self._t0 = perf_counter() if t0 is None else float(t0)
        self._overhead = 0.0
        self.trajectory: list[tuple[float, float]] = []
        self.last_kkt = float("inf")
        self.n_records = 0

    @staticmethod
    def _materialize(x) -> np.ndarray:
        """Force compute (counts as solve time) and return a host fp64 array."""
        try:
            import mlx.core as mx  # type: ignore

            if isinstance(x, mx.array):
                mx.eval(x)
        except Exception:
            pass
        return np.asarray(x, dtype=np.float64)

    def record(self, x) -> float:
        """Snapshot iterate `x`: stamp the time and evaluate the trusted KKT."""
        x_host = self._materialize(x)
        t = perf_counter() - self._t0 - self._overhead
        ov0 = perf_counter()
        k = float(self._kkt_fn(x_host))
        self._overhead += perf_counter() - ov0
        self.trajectory.append((float(t), k))
        self.last_kkt = k
        self.n_records += 1
        return k

    @property
    def elapsed(self) -> float:
        return perf_counter() - self._t0 - self._overhead

    def should_stop(self, kkt_tol: float) -> bool:
        """Convergence/budget check the candidate's loop should poll."""
        return self.last_kkt <= kkt_tol or self.elapsed >= self.max_time_s

    def time_to_kkt(self, tol: float) -> float:
        return time_to_target(self.trajectory, tol)
