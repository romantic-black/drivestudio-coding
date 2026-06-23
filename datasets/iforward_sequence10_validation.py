from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SEQUENCE10_MANIFEST_VERSION = "sequence10_manifest_v1"
SEQUENCE10_VALIDATION_PROTOCOLS = (
    "SingleFrame-K8",
    "S10-D1-Causal",
    "S10-D2-Causal",
    "S10-D1-Repair",
    "S10-D2-Repair",
    "Repeat Stability",
)


@dataclass(frozen=True)
class Sequence10ManifestEntry:
    scene_id: int
    segment_id: int
    stride: int
    start_block_pos: int
    keyframe_indices: Tuple[int, ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "scene_id": int(self.scene_id),
            "segment_id": int(self.segment_id),
            "stride": int(self.stride),
            "start_block_pos": int(self.start_block_pos),
            "keyframe_indices": [int(x) for x in self.keyframe_indices],
        }


def build_sequence10_manifest(
    *,
    dataset: Any,
    scene_segment_pairs: Iterable[Tuple[int, int]],
    strides: Iterable[int] = (1, 2),
    max_entries: int = 32,
) -> Dict[str, Any]:
    entries: List[Sequence10ManifestEntry] = []
    for scene_id, segment_id in sorted((int(s), int(g)) for s, g in scene_segment_pairs):
        sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
        keyframes = sorted(int(x) for x in list(getattr(sidx, "keyframe_indices", []) or []))
        for stride in sorted(int(x) for x in strides):
            if stride not in (1, 2):
                continue
            limit = int(len(keyframes) - 9 * int(stride))
            for start in range(max(0, limit)):
                selected = tuple(int(keyframes[start + i * int(stride)]) for i in range(10))
                if len(set(selected)) != 10:
                    continue
                entries.append(
                    Sequence10ManifestEntry(
                        scene_id=int(scene_id),
                        segment_id=int(segment_id),
                        stride=int(stride),
                        start_block_pos=int(start),
                        keyframe_indices=selected,
                    )
                )
                if len(entries) >= int(max_entries):
                    break
            if len(entries) >= int(max_entries):
                break
        if len(entries) >= int(max_entries):
            break
    return {
        "version": SEQUENCE10_MANIFEST_VERSION,
        "protocols": list(SEQUENCE10_VALIDATION_PROTOCOLS),
        "entries": [entry.to_json() for entry in entries],
    }


def write_sequence10_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SEQUENCE10_MANIFEST_VERSION",
    "SEQUENCE10_VALIDATION_PROTOCOLS",
    "Sequence10ManifestEntry",
    "build_sequence10_manifest",
    "write_sequence10_manifest",
]
