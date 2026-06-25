from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .index_format import PROTOCOL_IDS
from .index_loader import Stage22Index


@dataclass
class Stage22Traversal:
    index: Stage22Index
    scene_order: str = "shuffle_per_epoch"
    segment_order: str = "shuffle_per_epoch"
    forbid_consecutive_same_scene: bool = True
    seed: int = 0
    scene_cursor: int = 0
    segment_cursors: Dict[Tuple[int, str], int] = field(default_factory=dict)
    window_cursors: Dict[Tuple[int, str], int] = field(default_factory=dict)
    bootstrap_cursors: Dict[int, int] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)
    _scene_queues: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _segment_queues: Dict[str, List[int]] = field(default_factory=dict, init=False, repr=False)
    _last_scene: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def _protocol_segment_rows(self, protocol: str) -> Dict[int, List[int]]:
        pid = int(PROTOCOL_IDS[str(protocol)])
        rows = np.unique(self.index.windows[self.index.windows["protocol_id"] == pid]["segment_row"]).astype(np.int64)
        by_scene: Dict[int, List[int]] = {}
        for row in rows.tolist():
            seg = self.index.segments[int(row)]
            by_scene.setdefault(int(seg["scene_id"]), []).append(int(row))
        return {int(k): sorted(int(x) for x in v) for k, v in by_scene.items()}

    def available_protocols(self) -> Sequence[str]:
        out: List[str] = []
        for name, pid in PROTOCOL_IDS.items():
            if int(np.count_nonzero(self.index.windows["protocol_id"] == int(pid))) > 0:
                out.append(str(name))
        return tuple(out)

    def _ordered(self, values: Iterable[int], *, mode: str) -> List[int]:
        out = sorted(int(x) for x in values)
        if str(mode) == "shuffle_per_epoch":
            self._rng.shuffle(out)
        return out

    def _scene_queue_key(self, protocol: str) -> str:
        return str(protocol)

    def _next_scene(self, protocol: str, scenes: Sequence[int]) -> int:
        key = self._scene_queue_key(protocol)
        queue = self._scene_queues.get(key, [])
        scene_set = {int(x) for x in scenes}
        queue = [int(x) for x in queue if int(x) in scene_set]
        if not queue:
            queue = self._ordered(scenes, mode=str(self.scene_order))
            if (
                bool(self.forbid_consecutive_same_scene)
                and len(queue) > 1
                and int(queue[0]) == int(self._last_scene)
            ):
                queue.append(queue.pop(0))
        scene = int(queue.pop(0))
        self._scene_queues[key] = queue
        self._last_scene = int(scene)
        self.scene_cursor = int(self.scene_cursor) + 1
        return int(scene)

    def _next_segment_row(self, *, scene: int, protocol: str, rows: Sequence[int]) -> int:
        key = f"{int(scene)}::{str(protocol)}"
        row_set = {int(x) for x in rows}
        queue = [int(x) for x in self._segment_queues.get(key, []) if int(x) in row_set]
        if not queue:
            queue = self._ordered(rows, mode=str(self.segment_order))
        row = int(queue.pop(0))
        self._segment_queues[key] = queue
        self.segment_cursors[(int(scene), str(protocol))] = int(self.segment_cursors.get((int(scene), str(protocol)), 0)) + 1
        return int(row)

    def next_window(self, protocol: str) -> np.void:
        by_scene = self._protocol_segment_rows(str(protocol))
        scenes = sorted(int(x) for x in by_scene.keys())
        if not scenes:
            raise ValueError(f"Stage2_2 index has no windows for protocol {protocol!r}")
        scene = self._next_scene(str(protocol), scenes)
        rows = by_scene[int(scene)]
        segment_row = self._next_segment_row(scene=int(scene), protocol=str(protocol), rows=rows)
        pid = int(PROTOCOL_IDS[str(protocol)])
        candidates = self.index.windows[
            (self.index.windows["protocol_id"] == pid) & (self.index.windows["segment_row"] == int(segment_row))
        ]
        if int(candidates.shape[0]) == 0:
            raise ValueError(f"Stage2_2 no candidate windows for segment_row={segment_row} protocol={protocol}")
        win_key = (int(segment_row), str(protocol))
        win_pos = int(self.window_cursors.get(win_key, 0)) % int(candidates.shape[0])
        self.window_cursors[win_key] = win_pos + 1
        return candidates[int(win_pos)]

    def next_bootstrap_frame(self) -> np.void:
        if int(self.index.bootstrap_frames.shape[0]) == 0:
            raise ValueError("Stage2_2 index has no bootstrap frames")
        by_scene: Dict[int, List[int]] = {}
        for row, seg in enumerate(self.index.segments):
            by_scene.setdefault(int(seg["scene_id"]), []).append(int(row))
        scenes = sorted(int(x) for x in by_scene)
        scene = self._next_scene("bootstrap", scenes)
        rows = by_scene[int(scene)]
        segment_row = self._next_segment_row(scene=int(scene), protocol="bootstrap", rows=rows)
        candidates = self.index.bootstrap_frames[self.index.bootstrap_frames["segment_row"] == int(segment_row)]
        if int(candidates.shape[0]) == 0:
            raise ValueError(f"Stage2_2 bootstrap has no candidates for segment_row={segment_row}")
        pos = int(self.bootstrap_cursors.get(int(segment_row), 0)) % int(candidates.shape[0])
        self.bootstrap_cursors[int(segment_row)] = int(self.bootstrap_cursors.get(int(segment_row), 0)) + 1
        return candidates[int(pos)]

    def state_dict(self) -> Dict[str, object]:
        return {
            "scene_cursor": int(self.scene_cursor),
            "segment_cursors": {f"{k[0]}::{k[1]}": int(v) for k, v in self.segment_cursors.items()},
            "window_cursors": {f"{k[0]}::{k[1]}": int(v) for k, v in self.window_cursors.items()},
            "bootstrap_cursors": {str(k): int(v) for k, v in self.bootstrap_cursors.items()},
            "rng_state": self._rng.getstate(),
            "scene_queues": {str(k): [int(x) for x in v] for k, v in self._scene_queues.items()},
            "segment_queues": {str(k): [int(x) for x in v] for k, v in self._segment_queues.items()},
            "last_scene": int(self._last_scene),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.scene_cursor = int(state.get("scene_cursor", 0))
        self.segment_cursors = {}
        for raw_key, value in dict(state.get("segment_cursors", {}) or {}).items():
            scene, protocol = str(raw_key).split("::", 1)
            self.segment_cursors[(int(scene), str(protocol))] = int(value)
        self.window_cursors = {}
        for raw_key, value in dict(state.get("window_cursors", {}) or {}).items():
            row, protocol = str(raw_key).split("::", 1)
            self.window_cursors[(int(row), str(protocol))] = int(value)
        self.bootstrap_cursors = {int(k): int(v) for k, v in dict(state.get("bootstrap_cursors", {}) or {}).items()}
        if state.get("rng_state", None) is not None:
            self._rng.setstate(state["rng_state"])
        self._scene_queues = {str(k): [int(x) for x in v] for k, v in dict(state.get("scene_queues", {}) or {}).items()}
        self._segment_queues = {str(k): [int(x) for x in v] for k, v in dict(state.get("segment_queues", {}) or {}).items()}
        self._last_scene = int(state.get("last_scene", -1))


__all__ = ["Stage22Traversal"]
