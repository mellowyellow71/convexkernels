"""Durable checkpoint store + branchable experiment tree.

The autoresearch loop's memory must survive a rotting context window, so every
accepted champion is written as a durable node on disk:

    synth_state/checkpoints/<id>/
        source.py          the kernel source that produced this champion
        score.json         the EvalScore dict
        trajectory.json    the representative (t, kkt) trajectory
        meta.json          {id, parent_id, created_at, algorithm_tag,
                            problem_hash, time_to_kkt_s, total_time_s, kkt_final}

`parent_id` links each checkpoint to the node it was derived from, forming the
experiment tree. A session can resume/branch from *any* checkpoint id (not only
the latest), which lets a stalled line of search be abandoned for an earlier,
more promising node.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Checkpoint:
    id: str
    parent_id: Optional[str]
    created_at: str
    algorithm_tag: str
    problem_hash: str
    time_to_kkt_s: Optional[float]
    total_time_s: Optional[float]
    kkt_final: Optional[float]
    dir: Path

    @property
    def source_path(self) -> Path:
        return self.dir / "source.py"

    def source(self) -> str:
        return self.source_path.read_text()


class CheckpointStore:
    def __init__(self, state_root: Path):
        self.root = Path(state_root) / "checkpoints"

    def save(
        self,
        *,
        id: str,
        parent_id: Optional[str],
        source: str,
        score: dict,
        trajectory: list,
        algorithm_tag: str,
        problem_hash: str,
    ) -> Checkpoint:
        d = self.root / id
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.py").write_text(source)
        (d / "score.json").write_text(json.dumps(score, default=str))
        (d / "trajectory.json").write_text(json.dumps(trajectory))
        meta = {
            "id": id,
            "parent_id": parent_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm_tag": algorithm_tag,
            "problem_hash": problem_hash,
            "time_to_kkt_s": score.get("time_to_kkt_s"),
            "total_time_s": score.get("total_time_s"),
            "kkt_final": score.get("kkt_final"),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return self._from_meta(meta, d)

    def _from_meta(self, meta: dict, d: Path) -> Checkpoint:
        return Checkpoint(
            id=meta["id"],
            parent_id=meta.get("parent_id"),
            created_at=meta.get("created_at", ""),
            algorithm_tag=meta.get("algorithm_tag", ""),
            problem_hash=meta.get("problem_hash", ""),
            time_to_kkt_s=meta.get("time_to_kkt_s"),
            total_time_s=meta.get("total_time_s"),
            kkt_final=meta.get("kkt_final"),
            dir=d,
        )

    def get(self, id: str) -> Optional[Checkpoint]:
        d = self.root / id
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        return self._from_meta(json.loads(meta_path.read_text()), d)

    def all(self) -> list[Checkpoint]:
        if not self.root.exists():
            return []
        out: list[Checkpoint] = []
        for d in sorted(self.root.iterdir()):
            mp = d / "meta.json"
            if mp.exists():
                out.append(self._from_meta(json.loads(mp.read_text()), d))
        return out

    def best(self, problem_hash: str) -> Optional[Checkpoint]:
        """Fastest checkpoint (min total_time_s) for the given problem."""
        best: Optional[Checkpoint] = None
        for c in self.all():
            if c.problem_hash != problem_hash:
                continue
            if c.total_time_s is None:
                continue
            if best is None or c.total_time_s < (best.total_time_s or float("inf")):
                best = c
        return best
