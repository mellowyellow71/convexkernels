"""Slot-keyed champion store.

P4 implementation of the schema in `docs/schema.md`: a per-slot champion
symlink, metadata, pareto history, and a root index mapping slot keys to the
current champion lineage id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .lineage import Slot, now_iso


class ChampionStore:
    """Manage current champions under `state_root/synth_state/champions`."""

    def __init__(self, state_root: Path):
        self.state_root = Path(state_root)
        self.root = self.state_root / "synth_state" / "champions"
        self.index_path = self.root / "index.json"

    def slot_dir(self, slot: Slot) -> Path:
        return (
            self.root
            / slot.problem_family
            / slot.algorithm
            / slot.hardware
            / slot.dtype
        )

    def workload_dir(self, slot: Slot, workload_key: str) -> Path:
        return self.slot_dir(slot) / "workloads" / _safe_workload_key(workload_key)

    def champion_path(
        self,
        slot: Slot,
        *,
        workload_key: str | None = None,
    ) -> Path:
        if workload_key:
            return self.workload_dir(slot, workload_key) / "champion.py"
        return self.slot_dir(slot) / "champion.py"

    def current_source(
        self,
        slot: Slot,
        *,
        workload_key: str | None = None,
    ) -> Path | None:
        path = self.champion_path(slot, workload_key=workload_key)
        return path if path.exists() else None

    def current_metadata(
        self,
        slot: Slot,
        *,
        workload_key: str | None = None,
    ) -> dict[str, Any]:
        path = self.champion_path(slot, workload_key=workload_key).parent / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def current_source_for_workload(
        self,
        slot: Slot,
        workload_key: str,
    ) -> tuple[Path | None, dict[str, Any]]:
        """Return the champion for a cost/workload, with safe legacy fallback."""
        candidates: list[tuple[float, Path, dict[str, Any]]] = []

        metadata = self.current_metadata(slot, workload_key=workload_key)
        source = self.current_source(slot, workload_key=workload_key)
        if source is not None:
            candidates.append((_champion_metric_ms(metadata), source, metadata))

        legacy_metadata = self.current_metadata(slot)
        legacy_cost_model = (legacy_metadata.get("summary") or {}).get("cost_model")
        legacy_source = self.current_source(slot)
        if legacy_source is not None and legacy_cost_model == workload_key:
            candidates.append(
                (_champion_metric_ms(legacy_metadata), legacy_source, legacy_metadata)
            )

        if not candidates:
            return None, {}
        _, selected_source, selected_metadata = min(candidates, key=lambda item: item[0])
        return selected_source, selected_metadata

    def promote(
        self,
        *,
        slot: Slot,
        source_path: Path,
        record_id: str,
        source_hash: str,
        summary: dict[str, Any],
        workload_key: str | None = None,
    ) -> Path:
        """Atomically promote `source_path` as the slot champion.

        Uses a symlink per `docs/schema.md`; if an old champion exists, it is
        replaced by `os.replace()` of a temporary symlink.
        """
        champion_path = self.champion_path(slot, workload_key=workload_key)
        champion_dir = champion_path.parent
        champion_dir.mkdir(parents=True, exist_ok=True)
        tmp_link = champion_dir / "champion.py.tmp"

        source_abs = Path(source_path).resolve()
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(source_abs, tmp_link)
        os.replace(tmp_link, champion_path)

        metadata = {
            "id": record_id,
            "accepted_at": now_iso(),
            "source_hash": source_hash,
            "source_path": str(source_abs),
            "champion_path": str(champion_path),
            "workload_key": workload_key,
            "summary": summary,
        }
        self._atomic_write_json(champion_dir / "metadata.json", metadata)
        self._append_jsonl(champion_dir / "pareto.jsonl", metadata)
        self._update_index(slot, record_id, workload_key=workload_key)
        return champion_path

    def _update_index(
        self,
        slot: Slot,
        record_id: str,
        *,
        workload_key: str | None = None,
    ) -> None:
        index = {}
        if self.index_path.exists():
            index = json.loads(self.index_path.read_text())
        key = slot.key() if not workload_key else f"{slot.key()}::{workload_key}"
        index[key] = record_id
        self._atomic_write_json(self.index_path, index)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)

    @staticmethod
    def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(data, sort_keys=True) + "\n")


def _safe_workload_key(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)
    return safe.strip("_") or "default"


def _champion_metric_ms(metadata: dict[str, Any]) -> float:
    summary = metadata.get("summary") or {}
    for key in ("tier2_wall_time_ms", "tier1_wall_time_ms"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    return float("inf")
