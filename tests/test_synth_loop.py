"""Test the minimal synth loop end-to-end.

Uses the deterministic stub proposer + numpy_ref kernel; runs on Linux.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from convexkernels.frontend.lasso import Lasso
from convexkernels.synth.lineage import Slot, load_records
from convexkernels.synth import tiers as tiers_mod
from convexkernels.synth.loop import detect_hardware, run_synth_loop
from convexkernels.synth.proposers.structured import StructuredGridProposer
from convexkernels.synth.proposers.stub import DeterministicStubProposer
from convexkernels.synth.sandbox import SandboxResult


@pytest.fixture
def tiny_lasso() -> Lasso:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((100, 50))
    x_true = rng.standard_normal(50) * (rng.random(50) < 0.2)
    b = A @ x_true + 1e-2 * rng.standard_normal(100)
    return Lasso(A, b, lam=0.1 * float(np.max(np.abs(A.T @ b))))


def test_synth_loop_runs_and_writes_lineage(tmp_path: Path, tiny_lasso: Lasso):
    appended = run_synth_loop(
        proposer=DeterministicStubProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=3,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        tier1_max_iters=500,
        require_speedup=False,
    )
    assert len(appended) == 3

    records = load_records(tmp_path / "synth_state" / "lineage.jsonl")
    assert len(records) == 3
    assert (tmp_path / "synth_state" / "edits.json").exists()
    assert (tmp_path / "synth_state" / "fitness.json").exists()

    # All proposals should pass tier-1 (numpy_ref converges easily on tiny problem)
    assert all(r["tier1"]["passed"] for r in records), [r["tier1"] for r in records]
    assert all(r["decision"]["accepted"] for r in records)

    # Edit types cycle through tile_change, dtype_swap
    assert records[0]["edit"]["type"] == "tile_change"
    assert records[1]["edit"]["type"] == "dtype_swap"
    assert records[2]["edit"]["type"] == "tile_change"

    # Lineage records contain only tier1 (tier2/tier3 absent at P3.3)
    for r in records:
        assert "tier2" not in r
        assert "tier3" not in r


def test_synth_loop_tier1_failure_recorded(tmp_path: Path, tiny_lasso: Lasso):
    """Hard tier1 tol that the seed can't meet → all proposals should fail tier-1."""
    appended = run_synth_loop(
        proposer=DeterministicStubProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=2,
        state_root=tmp_path,
        tier1_kkt_tol=1e-12,  # impossible to reach
        tier1_max_iters=10,   # not enough iters even for an easy bar
        require_speedup=False,
    )

    records = load_records(tmp_path / "synth_state" / "lineage.jsonl")
    assert len(records) == 2
    assert all(not r["tier1"]["passed"] for r in records)
    assert all(not r["decision"]["accepted"] for r in records)
    assert all(r["tier1"]["reject_reason"] == "kkt_above_tier1_tol" for r in records)


def test_synth_loop_orphan_detection_after_crash(tmp_path: Path, tiny_lasso: Lasso):
    """Run a loop, create a fake started.json without a lineage record (simulating crash),
    re-run; verify the orphan is detected (logged but not requeued)."""
    # First run
    run_synth_loop(
        proposer=DeterministicStubProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        require_speedup=False,
    )

    # Simulate a crash: started.json with no lineage row
    fake_id = "0123abcd-fake-orphan-aaaa-aaaaaaaaaaaa"
    (tmp_path / "runs" / fake_id).mkdir(parents=True)
    (tmp_path / "runs" / fake_id / "started.json").write_text("{}")

    # Re-run; the loop should detect 1 orphan and continue
    run_synth_loop(
        proposer=DeterministicStubProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        require_speedup=False,
    )

    records = load_records(tmp_path / "synth_state" / "lineage.jsonl")
    assert len(records) == 2  # 1 from each run; orphan NOT auto-requeued


def test_synth_loop_speed_ratchet_keeps_only_faster(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """With speed gating enabled, KKT-valid but slower proposals are discarded."""
    class FullSourceStubProposer:
        def __init__(self):
            self.counter = 0

        def propose(self, slot, parent_id, history):
            from convexkernels.synth.lineage import Edit

            self.counter += 1
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        f"# proposal {self.counter}\n"
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale=f"proposal {self.counter}",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.008,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.009,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=FullSourceStubProposer(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=2,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=True,
        speedup_margin=0.95,
    )

    assert appended[0]["decision"]["accepted"]
    assert appended[0]["decision"]["reason"] == "keep:tier1_speedup"
    assert not appended[1]["decision"]["accepted"]
    assert appended[1]["decision"]["reason"] == "discard:not_faster_than_baseline"


def test_tier1_repeated_timing_uses_median(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.030,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.020,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    tier, result = tiers_mod.run_tier1(
        run_dir=tmp_path / "run",
        problem=tiny_lasso,
        kernel_module="convexkernels.kernels.numpy_ref",
        config=tiers_mod.EvalConfig(
            seed_kernel={
                "module": "convexkernels.kernels.numpy_ref",
                "step": "fista_step",
                "init": "init_state",
            }
        ),
        max_iters=500,
        tol=1e-3,
        reps=3,
    )

    assert tier.passed
    assert tier.n_reps == 3
    assert tier.wall_time_ms == pytest.approx(20.0)
    assert tier.wall_time_min_ms == pytest.approx(10.0)
    assert tier.wall_time_max_ms == pytest.approx(30.0)
    assert result.wall_time_s == pytest.approx(0.020)


def test_tier1_amortized_cost_model_uses_solve_time(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.110,
            setup_time_s=0.100, solve_time_s=0.010,
            single_solve_time_s=0.110, converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.220,
            setup_time_s=0.200, solve_time_s=0.020,
            single_solve_time_s=0.220, converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.330,
            setup_time_s=0.300, solve_time_s=0.030,
            single_solve_time_s=0.330, converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    tier, _ = tiers_mod.run_tier1(
        run_dir=tmp_path / "run",
        problem=tiny_lasso,
        kernel_module="convexkernels.kernels.numpy_ref",
        config=tiers_mod.EvalConfig(
            seed_kernel={
                "module": "convexkernels.kernels.numpy_ref",
                "step": "fista_step",
                "init": "init_state",
            },
            cost_model="amortized",
        ),
        max_iters=500,
        tol=1e-3,
        reps=3,
    )

    assert tier.wall_time_ms == pytest.approx(20.0)
    assert tier.setup_time_ms == pytest.approx(200.0)
    assert tier.solve_time_ms == pytest.approx(20.0)
    assert tier.single_solve_wall_time_ms == pytest.approx(220.0)
    assert tier.amortized_wall_time_ms == pytest.approx(20.0)
    assert tier.cost_model == "amortized"


def test_synth_loop_records_proposer_errors_and_continues(
    tmp_path: Path, tiny_lasso: Lasso
):
    """A transient provider/proposer failure should not abort the whole run."""
    from convexkernels.synth.lineage import Edit

    class FailsThenSucceeds:
        model = "test-model"

        def __init__(self):
            self.counter = 0

        def propose(self, slot, parent_id, history):
            self.counter += 1
            if self.counter == 1:
                raise RuntimeError("provider unavailable")
            return Edit(
                type="other",
                payload={},
                rationale="second attempt",
                proposer_role="impl",
                proposer_model="test-model",
                source="manual",
            )

    appended = run_synth_loop(
        proposer=FailsThenSucceeds(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=2,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        tier1_max_iters=500,
        require_speedup=False,
    )

    assert len(appended) == 2
    assert appended[0]["edit"]["type"] == "proposer_error"
    assert appended[0]["decision"]["reason"] == "crash:proposer_error:RuntimeError"
    assert appended[1]["decision"]["accepted"]


def test_synth_loop_skips_duplicate_full_source(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Exact source repeats should be recorded without another sandbox run."""
    from convexkernels.synth.lineage import Edit

    class SameSourceTwice:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="repeat source",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=SameSourceTwice(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=2,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    assert appended[0]["decision"]["accepted"]
    assert not appended[1]["decision"]["accepted"]
    assert appended[1]["decision"]["reason"] == "discard:duplicate_source"
    assert appended[1]["tier1"]["reject_reason"] == "duplicate_source"
    assert results == []


def test_synth_loop_applies_structured_edit(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Structured payloads should be applied to the current source and evaluated."""
    from convexkernels.synth.lineage import Edit

    class StructuredOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="tile_change",
                payload={"threadgroup_size": 128, "kernel_name_suffix": "tg128"},
                rationale="structured tile edit",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=StructuredOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.mlx.seeds.fista_step_v0",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    source = Path(appended[0]["source"]["path"]).read_text()
    assert appended[0]["decision"]["accepted"]
    assert "threadgroup=(min(128, state.x.shape[0]), 1, 1)" in source
    assert "from convexkernels.kernels.mlx.lib import LassoMLX" in source


def test_synth_loop_materializes_config_edit(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Gradient/dtype strategy payloads should promote as replayable source."""
    from convexkernels.synth.lineage import Edit

    class ConfigOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="algo_variant",
                payload={
                    "gradient_strategy": "gram",
                    "dtype_strategy": "mixed_gram",
                    "kernel_name_suffix": "gram_mixed",
                },
                rationale="try Gram gradient with fp16 solve path",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    seen_configs = []

    def fake_run_kernel(run_dir, **kwargs):
        seen_configs.append(json.loads((Path(run_dir) / "eval_config.json").read_text()))
        return SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        )

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=ConfigOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.mlx.seeds.fista_step_v0",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        problem_backend="mlx",
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    source = Path(appended[0]["source"]["path"]).read_text()
    assert appended[0]["decision"]["accepted"]
    assert appended[0]["decision"]["champion_for_slot"]
    assert seen_configs[0]["kernel_module"] == appended[0]["source"]["path"]
    assert seen_configs[0]["problem_dtype"] == "fp32"
    assert seen_configs[0]["dtype_strategy"] == "mixed_gram"
    assert 'GRADIENT_STRATEGY = "gram"' in source
    assert 'DTYPE_STRATEGY = "mixed_gram"' in source
    assert "LassoGramMLX.from_lasso_mlx" in source

    metadata_path = (
        tmp_path
        / "synth_state"
        / "champions"
        / "lasso"
        / "fista"
        / detect_hardware()
        / "fp64"
        / "workloads"
        / "single"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["summary"]["candidate_dtype_strategy"] == "mixed_gram"


def test_synth_loop_runs_structured_grid_proposer(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=StructuredGridProposer(
            payloads=(
                {
                    "branchless_soft_threshold": True,
                    "kernel_name_suffix": "branchless",
                },
            )
        ),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.mlx.seeds.fista_step_v0",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    assert appended[0]["edit"]["source"] == "structured_grid"
    assert appended[0]["decision"]["accepted"]
    assert appended[0]["edit"]["payload"]["branchless_soft_threshold"] is True


def test_synth_loop_structured_proposer_skips_persisted_payloads(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """A restarted structured sweep should continue past same-slot history."""
    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.011,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)
    payloads = (
        {"threadgroup_size": 128, "kernel_name_suffix": "first"},
        {"threadgroup_size": 512, "kernel_name_suffix": "second"},
    )
    seed_kernel = {
        "module": "convexkernels.kernels.mlx.seeds.fista_step_v0",
        "step": "fista_step",
        "init": "init_state",
    }

    first = run_synth_loop(
        proposer=StructuredGridProposer(payloads=payloads),
        problem=tiny_lasso,
        seed_kernel=seed_kernel,
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )
    second = run_synth_loop(
        proposer=StructuredGridProposer(payloads=payloads),
        problem=tiny_lasso,
        seed_kernel=seed_kernel,
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    assert first[0]["edit"]["payload"]["kernel_name_suffix"] == "first"
    assert second[0]["edit"]["payload"]["kernel_name_suffix"] == "second"
    assert results == []


def test_synth_loop_records_applier_errors_and_continues(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """A bad structured payload should not abort later proposals."""
    from convexkernels.synth.lineage import Edit

    class BadThenGood:
        def __init__(self):
            self.counter = 0

        def propose(self, slot, parent_id, history):
            self.counter += 1
            if self.counter == 1:
                return Edit(
                    type="vectorize",
                    payload={"items_per_thread": 3},
                    rationale="bad structured value",
                    proposer_role="impl",
                    proposer_model="test",
                    source="manual",
                )
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="valid fallback",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=BadThenGood(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=2,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=False,
    )

    assert appended[0]["decision"]["reason"] == "invalid:applier_error:ValueError"
    assert appended[0]["tier1"]["reject_reason"] == "applier_error"
    assert (tmp_path / "runs" / appended[0]["id"] / "applier_error.txt").exists()
    assert appended[1]["decision"]["accepted"]
    assert results == []


def test_synth_loop_evaluates_transfer_seed_before_proposer(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Transfer seeds should be evaluated before the fallback proposer is asked."""
    lineage_path = tmp_path / "synth_state" / "lineage.jsonl"
    lineage_path.parent.mkdir(parents=True)
    lineage_path.write_text(
        json.dumps(
            {
                "id": "source-record",
                "slot": {
                    "problem_family": "lasso",
                    "algorithm": "fista",
                    "hardware": detect_hardware(),
                    "dtype": "fp64",
                },
                "edit": {
                    "type": "tile_change",
                    "payload": {
                        "threadgroup_size": 128,
                        "kernel_name_suffix": "xfer_tg128",
                    },
                    "rationale": "accepted source edit",
                    "proposer_role": "impl",
                    "proposer_model": "test",
                    "source": "structured_grid",
                },
                "source": {"hash": "sha256:source-record"},
                "tier1": {"passed": True, "wall_time_ms": 10.0},
                "tier2": {
                    "passed": True,
                    "wall_time_ms": 18.0,
                    "speed_ref_wall_time_ms": 20.0,
                },
                "decision": {"accepted": True, "reason": "keep:tier2_passed"},
            }
        )
        + "\n"
    )

    class ShouldNotBeAsked:
        def propose(self, slot, parent_id, history):
            raise AssertionError("fallback proposer should not be called")

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=ShouldNotBeAsked(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.mlx.seeds.nonnegative_fista_step_v0",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        slot=Slot("nonnegative_lasso", "fista", detect_hardware(), "fp64"),
        tier1_kkt_tol=1e-3,
        require_speedup=False,
        transfer_seed_k=1,
    )

    assert appended[0]["edit"]["source"].startswith("transfer:lasso/fista/")
    assert appended[0]["edit"]["payload"]["threadgroup_size"] == 128
    source = Path(appended[0]["source"]["path"]).read_text()
    assert "threadgroup=(min(128, state.x.shape[0]), 1, 1)" in source
    assert "NonnegativeLassoMLX" in source


def test_synth_loop_tier2_promotion_records_tier2(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """A full-source speedup can require Tier-2 before champion promotion."""
    from convexkernels.synth.lineage import Edit

    class FullSourceOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="tier2 candidate",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.008,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.020,
            converged=True, iters=20,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=FullSourceOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=True,
        speedup_margin=0.95,
        promotion_tier="tier2",
        tier2_problem=tiny_lasso,
        tier2_kkt_tol=1e-6,
        require_tier2_speed=False,
    )

    record = appended[0]
    assert record["decision"]["accepted"]
    assert record["decision"]["reason"] == "keep:tier2_passed"
    assert record["decision"]["champion_for_slot"]
    assert record["tier2"]["passed"]

    champion = (
        tmp_path / "synth_state" / "champions" / "lasso" / "fista"
        / detect_hardware() / "fp64" / "workloads" / "single" / "champion.py"
    )
    assert champion.is_symlink()


def test_synth_loop_tier2_speed_gate_rejects_slow_convergence(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Tier-1 wins are rejected if full convergence is slower than champion."""
    from convexkernels.synth.lineage import Edit

    class FullSourceOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="fast smoke, slow convergence",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.020,
            converged=True, iters=20,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.008,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.021,
            converged=True, iters=20,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=FullSourceOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=True,
        speedup_margin=0.95,
        promotion_tier="tier2",
        tier2_problem=tiny_lasso,
        tier2_kkt_tol=1e-6,
        tier2_speed_margin=0.95,
    )

    record = appended[0]
    assert not record["decision"]["accepted"]
    assert record["decision"]["reason"] == "tier_failed:2_speed"
    assert record["tier2"]["passed"]


def test_synth_loop_tier2_escalates_tier1_near_miss(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """Tier-2 promotion uses Tier-1 as an escalation filter, not final speed."""
    from convexkernels.synth.lineage import Edit

    class FullSourceOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="near miss in tier1, real win in tier2",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.020,
            converged=True, iters=20,
        ),
        # Faster than baseline, but not by the 0.95 Tier-1 promotion margin.
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.0098,
            converged=True, iters=10,
        ),
        # Full convergence clears the Tier-2 speed gate.
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.018,
            converged=True, iters=20,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=FullSourceOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=True,
        speedup_margin=0.95,
        promotion_tier="tier2",
        tier2_problem=tiny_lasso,
        tier2_kkt_tol=1e-6,
        tier2_speed_margin=0.97,
        tier1_escalation_margin=1.0,
        confirm_tier2_speed=False,
    )

    record = appended[0]
    assert record["decision"]["accepted"]
    assert record["decision"]["reason"] == "keep:tier2_passed"
    assert record["tier1"]["wall_time_ms"] == pytest.approx(9.8)
    assert record["tier2"]["wall_time_ms"] == pytest.approx(18.0)


def test_synth_loop_tier2_paired_confirm_blocks_timing_drift(
    tmp_path: Path, tiny_lasso: Lasso, monkeypatch: pytest.MonkeyPatch
):
    """A startup-baseline win must also beat a paired champion remeasurement."""
    from convexkernels.synth.lineage import Edit

    class FullSourceOnce:
        def propose(self, slot, parent_id, history):
            return Edit(
                type="other",
                payload={
                    "full_source": (
                        "from convexkernels.kernels.numpy_ref import "
                        "init_state, fista_step\n"
                    )
                },
                rationale="apparent tier2 win",
                proposer_role="impl",
                proposer_model="test",
                source="manual",
            )

    results = [
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.010,
            converged=True, iters=10,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.020,
            converged=True, iters=20,
        ),
        SandboxResult(
            status="completed", kkt_final=1e-4, wall_time_s=0.009,
            converged=True, iters=10,
        ),
        # Looks faster than the startup Tier-2 baseline.
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.018,
            converged=True, iters=20,
        ),
        # Paired remeasurement is faster still, so promotion is blocked.
        SandboxResult(
            status="completed", kkt_final=1e-7, wall_time_s=0.017,
            converged=True, iters=20,
        ),
    ]

    def fake_run_kernel(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tiers_mod, "run_kernel", fake_run_kernel)

    appended = run_synth_loop(
        proposer=FullSourceOnce(),
        problem=tiny_lasso,
        seed_kernel={
            "module": "convexkernels.kernels.numpy_ref",
            "step": "fista_step",
            "init": "init_state",
        },
        n_proposals=1,
        state_root=tmp_path,
        tier1_kkt_tol=1e-3,
        require_speedup=True,
        speedup_margin=0.95,
        promotion_tier="tier2",
        tier2_problem=tiny_lasso,
        tier2_kkt_tol=1e-6,
        tier2_speed_margin=0.97,
        tier1_escalation_margin=1.0,
        confirm_tier2_speed=True,
    )

    record = appended[0]
    assert not record["decision"]["accepted"]
    assert record["decision"]["reason"] == "tier_failed:2_speed"
    assert record["tier2"]["speed_ref_source"] == "paired"
    assert record["tier2"]["speed_ref_wall_time_ms"] == pytest.approx(17.0)
