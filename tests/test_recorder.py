import time

import numpy as np

from convexkernels.synth.recorder import Recorder


def test_recorder_uses_injected_trusted_fn_not_candidate():
    # The recorder must report the trusted value regardless of what the
    # candidate "thinks" the iterate is worth.
    calls = []

    def trusted(x):
        calls.append(np.asarray(x).copy())
        return 0.5  # trusted verdict

    rec = Recorder(trusted)
    k = rec.record(np.ones(5))
    assert k == 0.5
    assert rec.last_kkt == 0.5
    assert len(calls) == 1
    assert len(rec.trajectory) == 1


def test_recorder_excludes_kkt_cost_from_timestamps():
    def slow_trusted(x):
        time.sleep(0.02)  # expensive "measurement"
        return 1.0

    rec = Recorder(slow_trusted)
    rec.record(np.zeros(3))
    rec.record(np.zeros(3))
    # Two records each sleeping 20ms; elapsed (which subtracts overhead)
    # should stay well under the ~40ms of pure measurement cost.
    assert rec.elapsed < 0.02
    # And recorded timestamps should not accumulate the measurement cost.
    assert rec.trajectory[1][0] < 0.02


def test_recorder_should_stop_on_tol_and_budget():
    rec = Recorder(lambda x: 1e-9)
    rec.record(np.zeros(2))
    assert rec.should_stop(1e-6) is True

    rec2 = Recorder(lambda x: 1.0, max_time_s=0.0)
    assert rec2.should_stop(1e-6) is True  # budget exhausted


def test_recorder_time_to_kkt():
    rec = Recorder(lambda x: 1.0)
    rec.trajectory = [(0.0, 1.0), (2.0, 0.01)]
    assert abs(rec.time_to_kkt(0.1) - 1.0) < 1e-9
