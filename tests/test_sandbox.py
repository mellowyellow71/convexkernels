"""End-to-end test for the sandbox + checkpoint infra.

Uses the numpy_ref kernel (Linux-runnable). Validates:
  - run_kernel runs FISTA in a subprocess and returns a completed result
  - the result matches in-process FISTA to functional equivalence
  - mark_started + find_orphans round-trips correctly
  - lineage records can be written and read back
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from convexkernels.algorithms.fista import fista
from convexkernels.algorithms.kkt import assert_equivalent
from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.checkpoint import find_orphans, mark_started
from convexkernels.synth._eval_kernel import _load_kernel_module
from convexkernels.synth.lineage import (
    Decision,
    Edit,
    LineageRecord,
    Slot,
    SourceInfo,
    Tier1Result,
    append_record,
    load_records,
    new_id,
    now_iso,
)
from convexkernels.synth.sandbox import run_kernel, write_eval_config


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((100, 50))
    x_true = rng.standard_normal(50) * (rng.random(50) < 0.2)
    b = A @ x_true + 1e-2 * rng.standard_normal(100)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def test_sandbox_completes_end_to_end(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """End-to-end: write config, run_kernel, parse result, compare to in-process."""
    run_dir = tmp_path / "run_e2e"
    write_eval_config(
        run_dir,
        tiny_lasso,
        kernel_module="convexkernels.kernels.numpy_ref",
        kernel_step="fista_step",
        kernel_init="init_state",
        max_iters=2000,
        tol=1e-6,
    )
    result = run_kernel(run_dir, timeout_s=30.0)

    assert result.status == "completed", (
        f"sandbox failed: {result.status} / {result.error_message} / "
        f"stderr_tail={result.stderr_tail[-500:]}"
    )
    assert result.converged
    assert result.kkt_final < 1e-6
    assert result.iters is not None and result.iters < 2000

    # Compare to in-process FISTA
    res_inproc = fista(tiny_lasso, max_iters=2000, tol=1e-6, variant="basic")
    x_sandbox = np.load(run_dir / "x.npy")
    assert_equivalent(x_sandbox, res_inproc.x, tiny_lasso, kkt_tol=1e-6, drift_warn=1e-2)


def test_sandbox_runtime_error_returns_status(tmp_path: Path, tiny_lasso: Lasso) -> None:
    """Pointing at a non-existent kernel module should return runtime_error, not crash."""
    run_dir = tmp_path / "run_bad"
    write_eval_config(
        run_dir,
        tiny_lasso,
        kernel_module="convexkernels.kernels.does_not_exist",
        kernel_step="fista_step",
        kernel_init="init_state",
    )
    result = run_kernel(run_dir, timeout_s=10.0)
    assert result.status == "runtime_error"
    assert "does_not_exist" in (result.error_message or "")


def test_eval_path_loader_supports_dataclasses(tmp_path: Path) -> None:
    """Path-loaded proposal modules must be in sys.modules before dataclass decoration."""
    source_path = tmp_path / "proposal.py"
    source_path.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class ProposalState:\n"
        "    x: int\n"
    )

    module = _load_kernel_module(str(source_path))

    assert module.ProposalState(3).x == 3


def test_sandbox_prepare_problem_hook_records_timing(
    tmp_path: Path,
    tiny_lasso: Lasso,
) -> None:
    source_path = tmp_path / "prepared_kernel.py"
    source_path.write_text(
        "from pathlib import Path\n"
        "from convexkernels.kernels.numpy_ref import init_state, fista_step\n\n"
        "def prepare_problem(problem, config):\n"
        "    Path(config['problem_pickle_path']).with_name('prepared.txt').write_text(\n"
        "        config.get('dtype_strategy', '')\n"
        "    )\n"
        "    return problem\n"
    )
    run_dir = tmp_path / "run_prepared"
    write_eval_config(
        run_dir,
        tiny_lasso,
        kernel_module=str(source_path),
        kernel_step="fista_step",
        kernel_init="init_state",
        dtype_strategy="mixed_gram",
        max_iters=2000,
        tol=1e-6,
    )

    result = run_kernel(run_dir, timeout_s=30.0)

    assert result.status == "completed"
    assert (run_dir / "prepared.txt").read_text() == "mixed_gram"
    assert result.setup_time_s is not None and result.setup_time_s >= 0.0
    assert result.solve_time_s is not None and result.solve_time_s > 0.0
    assert result.single_solve_time_s == pytest.approx(
        result.setup_time_s + result.solve_time_s
    )
    assert result.wall_time_s == pytest.approx(result.single_solve_time_s)


def test_checkpoint_orphan_detection(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    lineage_path = tmp_path / "synth_state" / "lineage.jsonl"

    slot = Slot("lasso", "fista", "linux_x86", "fp64")
    edit = Edit(
        type="tile_change", payload={"from": 128, "to": 256},
        rationale="test", proposer_role="impl",
        proposer_model="stub", source="manual",
    )

    # Started but no lineage → orphan
    rec_id_1 = new_id()
    mark_started(runs_root / rec_id_1, slot, edit, source_path="seed.py")

    # Started + lineage → not an orphan
    rec_id_2 = new_id()
    mark_started(runs_root / rec_id_2, slot, edit, source_path="seed.py")
    record = LineageRecord(
        id=rec_id_2, parent_id=None, generation=0,
        created_at=now_iso(), evaluated_at=now_iso(),
        slot=slot, edit=edit,
        source=SourceInfo(path="seed.py", hash="sha256:..."),
        tier1=Tier1Result(passed=True, wall_time_ms=12.0),
        decision=Decision(accepted=True, reason="seed"),
    )
    append_record(record, lineage_path)

    orphans = find_orphans(runs_root, lineage_path)
    assert orphans == [rec_id_1]


def test_lineage_round_trip(tmp_path: Path) -> None:
    lineage_path = tmp_path / "lineage.jsonl"
    slot = Slot("lasso", "fista", "m3_pro", "fp32")
    edit = Edit("tile_change", {"from": 128, "to": 256}, "rationale",
                "impl", "stub", "manual")

    record = LineageRecord(
        id=new_id(), parent_id=None, generation=0,
        created_at=now_iso(), evaluated_at=now_iso(),
        slot=slot, edit=edit,
        source=SourceInfo(path="x.py", hash="sha256:abc"),
        tier1=Tier1Result(passed=True, wall_time_ms=10.0),
        decision=Decision(accepted=True, reason="seed"),
    )
    append_record(record, lineage_path)

    records = load_records(lineage_path)
    assert len(records) == 1
    assert records[0]["id"] == record.id
    assert records[0]["slot"]["problem_family"] == "lasso"
    assert records[0]["edit"]["type"] == "tile_change"
    # tier2/tier3 absent (not populated)
    assert "tier2" not in records[0]
    assert "tier3" not in records[0]
