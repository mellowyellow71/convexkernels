"""Tests for the P4 champion store."""

from __future__ import annotations

import json
from pathlib import Path

from convexkernels.synth.champion_store import ChampionStore
from convexkernels.synth.lineage import Slot


def test_champion_store_promotes_symlink_and_index(tmp_path: Path):
    store = ChampionStore(tmp_path)
    slot = Slot("lasso", "fista", "apple_silicon", "fp32")
    source = tmp_path / "runs" / "abc" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("def fista_step():\n    pass\n")

    champion = store.promote(
        slot=slot,
        source_path=source,
        record_id="abc",
        source_hash="sha256:abc",
        summary={"tier1_wall_time_ms": 4.2},
    )

    assert champion.is_symlink()
    assert champion.resolve() == source.resolve()
    assert store.current_source(slot) == champion

    index = json.loads((tmp_path / "synth_state" / "champions" / "index.json").read_text())
    assert index[slot.key()] == "abc"

    metadata = json.loads((champion.parent / "metadata.json").read_text())
    assert metadata["id"] == "abc"
    assert metadata["summary"]["tier1_wall_time_ms"] == 4.2
    assert (champion.parent / "pareto.jsonl").read_text().count("\n") == 1


def test_champion_store_isolates_workload_champions(tmp_path: Path):
    store = ChampionStore(tmp_path)
    slot = Slot("lasso", "fista", "apple_silicon", "fp32")
    source = tmp_path / "runs" / "gram" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("def fista_step():\n    pass\n")

    champion = store.promote(
        slot=slot,
        source_path=source,
        record_id="gram",
        source_hash="sha256:gram",
        workload_key="amortized",
        summary={"cost_model": "amortized"},
    )

    assert champion.is_symlink()
    assert champion.parent.name == "amortized"
    assert store.current_source(slot, workload_key="amortized") == champion
    assert store.current_source(slot, workload_key="single") is None

    selected, metadata = store.current_source_for_workload(slot, "amortized")
    assert selected == champion
    assert metadata["workload_key"] == "amortized"

    selected, metadata = store.current_source_for_workload(slot, "single")
    assert selected is None
    assert metadata == {}

    index = json.loads((tmp_path / "synth_state" / "champions" / "index.json").read_text())
    assert index[f"{slot.key()}::amortized"] == "gram"


def test_champion_store_selects_fastest_matching_workload(
    tmp_path: Path,
):
    store = ChampionStore(tmp_path)
    slot = Slot("lasso", "fista", "apple_silicon", "fp32")
    legacy_source = tmp_path / "runs" / "legacy" / "source.py"
    workload_source = tmp_path / "runs" / "workload" / "source.py"
    legacy_source.parent.mkdir(parents=True)
    workload_source.parent.mkdir(parents=True)
    legacy_source.write_text("def fista_step():\n    pass\n")
    workload_source.write_text("def fista_step():\n    pass\n")

    legacy = store.promote(
        slot=slot,
        source_path=legacy_source,
        record_id="legacy",
        source_hash="sha256:legacy",
        summary={"cost_model": "amortized", "tier2_wall_time_ms": 10.0},
    )
    store.promote(
        slot=slot,
        source_path=workload_source,
        record_id="workload",
        source_hash="sha256:workload",
        workload_key="amortized",
        summary={"cost_model": "amortized", "tier2_wall_time_ms": 20.0},
    )

    selected, metadata = store.current_source_for_workload(slot, "amortized")

    assert selected == legacy
    assert metadata["id"] == "legacy"
