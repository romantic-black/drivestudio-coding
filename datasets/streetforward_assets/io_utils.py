from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_parquet_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as exc:
        raise ValueError(
            "Writing parquet requires pandas/pyarrow. Please install them in the current environment."
        ) from exc
    df = pd.DataFrame(rows)
    df.to_parquet(str(path), index=False)
    _fsync_file(path)


def read_parquet_table(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise ValueError(
            "Reading parquet requires pandas/pyarrow. Please install them in the current environment."
        ) from exc
    df = pd.read_parquet(str(path))
    return df.to_dict(orient="records")


def flatten_keyframe_to_frames(keyframe_to_frames: Dict[int, List[int]]) -> Dict[str, np.ndarray]:
    keys = sorted(int(k) for k in keyframe_to_frames.keys())
    flat: List[int] = []
    offsets: List[int] = [0]
    for k in keys:
        vals = [int(x) for x in keyframe_to_frames[int(k)]]
        flat.extend(vals)
        offsets.append(len(flat))
    return {
        "keyframe_indices_sorted": np.asarray(keys, dtype=np.int32),
        "keyframe_to_frames_flat": np.asarray(flat, dtype=np.int32),
        "keyframe_to_frames_offsets": np.asarray(offsets, dtype=np.int64),
    }


def restore_keyframe_to_frames(
    keyframe_indices_sorted: np.ndarray,
    keyframe_to_frames_flat: np.ndarray,
    keyframe_to_frames_offsets: np.ndarray,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    keys = [int(x) for x in np.asarray(keyframe_indices_sorted).tolist()]
    flat = np.asarray(keyframe_to_frames_flat).astype(np.int64, copy=False)
    offs = np.asarray(keyframe_to_frames_offsets).astype(np.int64, copy=False)
    if offs.ndim != 1 or len(offs) != len(keys) + 1:
        raise ValueError("invalid keyframe_to_frames_offsets shape")
    for i, k in enumerate(keys):
        lo = int(offs[i])
        hi = int(offs[i + 1])
        out[k] = [int(v) for v in flat[lo:hi].tolist()]
    return out


def atomic_write_asset_dir(
    final_dir: Path,
    writer: Callable[[Path], None],
    *,
    tmp_root: Path,
) -> None:
    final_dir = final_dir.resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        raise ValueError(f"asset directory already exists: {final_dir}")

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{final_dir.name}.partial.",
            dir=str(tmp_root.resolve()),
        )
    )
    try:
        writer(tmp_dir)
        ready_path = tmp_dir / "READY"
        ready_path.write_text("ready\n", encoding="utf-8")
        _fsync_file(ready_path)

        for p in sorted(tmp_dir.rglob("*")):
            if p.is_file():
                _fsync_file(p)
        _fsync_dir(tmp_dir)
        os.rename(str(tmp_dir), str(final_dir))
        _fsync_dir(final_dir.parent)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def append_registry_row(registry_path: Path, row: Dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception:
        fallback = registry_path.with_suffix(".jsonl")
        with fallback.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(fallback.parent)
        return

    try:
        if registry_path.exists():
            df = pd.read_parquet(str(registry_path))
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_parquet(str(registry_path), index=False)
        _fsync_file(registry_path)
        _fsync_dir(registry_path.parent)
    except Exception:
        fallback = registry_path.with_suffix(".jsonl")
        with fallback.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(fallback.parent)


def list_asset_dirs_by_prefix(base_dir: Path, prefix: str) -> List[Path]:
    if not base_dir.exists():
        return []
    out: List[Path] = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            out.append(p)
    return sorted(out)
