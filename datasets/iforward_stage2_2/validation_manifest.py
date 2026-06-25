from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .index_format import IFORWARD_STAGE2_2_SCHEDULER_VERSION, PROTOCOL_PATTERNS
from .index_loader import Stage22Index, load_stage2_2_index


STAGE22_VALIDATION_PROTOCOLS = (
    "S10-D1-Causal",
    "S10-D2-Causal",
    "S10-I123-Causal",
    "S10-D1-Repair",
    "S10-D2-Repair",
    "Repeat Stability",
    "Order Robustness",
)


def protocol_to_window_protocol(protocol: str) -> str:
    text = str(protocol)
    if "D2" in text:
        return "D2"
    if "I123" in text:
        return "I123"
    return "D1"


def build_stage2_2_validation_manifest(
    *,
    index: Stage22Index,
    protocols: Optional[Sequence[str]] = None,
    max_entries: int = 8,
) -> Dict[str, Any]:
    protocols = tuple(str(x) for x in (protocols or STAGE22_VALIDATION_PROTOCOLS))
    entries: List[Dict[str, Any]] = []
    for protocol in protocols:
        window_protocol = protocol_to_window_protocol(str(protocol))
        windows = index.windows_for_protocol(window_protocol)
        count = 0
        for raw in windows:
            seg = index.segments[int(raw["segment_row"])]
            entries.append(
                {
                    "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
                    "index_fingerprint": index.fingerprint,
                    "protocol": str(protocol),
                    "window_protocol": str(window_protocol),
                    "scene_id": int(seg["scene_id"]),
                    "segment_id": int(seg["segment_id"]),
                    "segment_row": int(raw["segment_row"]),
                    "start_local_frame": int(raw["start_local_frame"]),
                    "pattern_id": int(raw["pattern_id"]),
                }
            )
            count += 1
            if count >= max(1, int(max_entries)):
                break
    if not entries:
        raise ValueError("Stage2_2 validation manifest is empty")
    return {
        "scheduler_version": IFORWARD_STAGE2_2_SCHEDULER_VERSION,
        "index_fingerprint": index.fingerprint,
        "protocol_patterns": {
            str(k): [[int(x) for x in pattern] for pattern in patterns]
            for k, patterns in PROTOCOL_PATTERNS.items()
        },
        "entries": entries,
    }


def write_stage2_2_validation_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def load_or_build_stage2_2_validation_manifest(
    *,
    index_dir: str | Path,
    manifest_path: str | Path = "",
    protocols: Optional[Sequence[str]] = None,
    max_entries: int = 8,
) -> Dict[str, Any]:
    if str(manifest_path):
        path = Path(manifest_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not list(data.get("entries", []) or []):
                raise ValueError("Stage2_2 validation manifest is empty")
            return dict(data)
    index = load_stage2_2_index(index_dir)
    manifest = build_stage2_2_validation_manifest(index=index, protocols=protocols, max_entries=max_entries)
    if str(manifest_path):
        write_stage2_2_validation_manifest(manifest_path, manifest)
    return manifest


__all__ = [
    "STAGE22_VALIDATION_PROTOCOLS",
    "build_stage2_2_validation_manifest",
    "load_or_build_stage2_2_validation_manifest",
    "protocol_to_window_protocol",
    "write_stage2_2_validation_manifest",
]
