from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.streetforward_assets.io_utils import (
    append_registry_row,
    atomic_write_asset_dir,
    flatten_keyframe_to_frames,
    list_asset_dirs_by_prefix,
    read_parquet_table,
    read_json,
    read_npz,
    restore_keyframe_to_frames,
    write_parquet_table,
    write_json,
    write_npz,
)
from datasets.streetforward_assets.schema import (
    SCHEMA_VERSION,
    normalize_missing_policy,
    require_manifest_fields,
)


def _fingerprint_placeholder(name: str) -> str:
    return f"sha256:{name}"


def _update_hash_ndarray(h: Any, name: str, arr: np.ndarray) -> None:
    """Incorporate array bytes in a process-independent way (no Python hash())."""
    h.update(name.encode("utf-8"))
    a = np.ascontiguousarray(arr)
    h.update(str(a.dtype.str).encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes(order="C"))


def stable_scene_asset_id_suffix(
    *,
    dataset: str,
    scene_id: int,
    num_frames: int,
    num_cams: int,
    split_config: Dict[str, Any],
    scene_index_arrays: Dict[str, Any],
    image_table_rows: List[Dict[str, Any]],
) -> str:
    """
    Stable 8-hex suffix for scene asset directory names.
    Same inputs -> same suffix across processes / PYTHONHASHSEED.
    """
    h = hashlib.sha256()
    h.update(b"v1/streetforward_scene_asset_id\n")
    h.update(str(dataset).encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(scene_id)).encode("ascii"))
    h.update(b"\0")
    h.update(str(int(num_frames)).encode("ascii"))
    h.update(b"\0")
    h.update(str(int(num_cams)).encode("ascii"))
    h.update(b"\0")
    h.update(json.dumps(split_config, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    for key in sorted(scene_index_arrays.keys()):
        _update_hash_ndarray(h, key, np.asarray(scene_index_arrays[key]))
    h.update(
        json.dumps(
            sorted(
                [
                    {
                        "frame_idx": int(r["frame_idx"]),
                        "cam_id": int(r["cam_id"]),
                        "img_idx": int(r["img_idx"]),
                        "height": int(r["height"]),
                        "width": int(r["width"]),
                        "image_path": str(r.get("image_path", "")),
                        "depth_path": str(r.get("depth_path", "")),
                        "sky_mask_path": str(r.get("sky_mask_path", "")),
                        "dynamic_mask_path": str(r.get("dynamic_mask_path", "")),
                    }
                    for r in image_table_rows
                ],
                key=lambda x: (x["frame_idx"], x["cam_id"]),
            ),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    )
    return h.hexdigest()[:8]


def stable_segment_asset_id_suffix(
    *,
    dataset: str,
    scene_id: int,
    segment_id: int,
    parent_scene_asset_id: str,
    segment_index_payload: Dict[str, Any],
    segment_pose_payload: Dict[str, Any],
    pointcloud_payload: Dict[str, Any],
    dynamic_tracks_payload: Dict[str, Any],
    segment_aabb: Any,
    pointcloud_config_normalized: Dict[str, Any],
    dynamic_concat: np.ndarray,
    dynamic_intids: Sequence[int],
    dyn_orig_ids: Sequence[int],
    offsets: Sequence[int],
) -> str:
    """
    Stable 8-hex suffix for segment asset directory names from export payloads
    (aligned with bytes written to pointcloud / index / tracks files).
    """
    h = hashlib.sha256()
    h.update(b"v1/streetforward_segment_asset_id\n")
    h.update(str(dataset).encode("utf-8"))
    h.update(b"\0")
    h.update(str(int(scene_id)).encode("ascii"))
    h.update(b"\0")
    h.update(str(int(segment_id)).encode("ascii"))
    h.update(b"\0")
    h.update(str(parent_scene_asset_id).encode("utf-8"))
    h.update(b"\0")
    h.update(
        json.dumps(pointcloud_config_normalized, sort_keys=True, ensure_ascii=True).encode("utf-8")
    )
    _update_hash_ndarray(h, "segment_aabb", np.asarray(segment_aabb, dtype=np.float32))

    sip = segment_index_payload
    k2f = sip["keyframe_to_frames"]
    f2k = sip["frame_to_keyframe"]
    h.update(
        json.dumps(
            {
                "num_cams": int(sip["num_cams"]),
                "frame_indices": [int(x) for x in sip["frame_indices"]],
                "test_frame_indices": [int(x) for x in sip["test_frame_indices"]],
                "keyframe_indices": [int(x) for x in sip["keyframe_indices"]],
                "keyframe_to_frames": {
                    str(int(k)): [int(x) for x in k2f[k]]
                    for k in sorted(k2f.keys(), key=lambda x: int(x))
                },
                "frame_to_keyframe": {
                    str(int(k)): int(f2k[k]) for k in sorted(f2k.keys(), key=lambda x: int(x))
                },
                "segment_first_frame_idx": int(sip["segment_first_frame_idx"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    _update_hash_ndarray(h, "train_image_refs", np.asarray(sip["train_image_refs"], dtype=np.int32))
    _update_hash_ndarray(h, "test_image_refs", np.asarray(sip["test_image_refs"], dtype=np.int32))

    _update_hash_ndarray(
        h,
        "segment_first_pose_world",
        np.asarray(segment_pose_payload["segment_first_pose_world"], dtype=np.float32),
    )
    _update_hash_ndarray(
        h,
        "world_to_seg0",
        np.asarray(segment_pose_payload["world_to_seg0"], dtype=np.float32),
    )
    h.update(str(segment_pose_payload["segment_pose_source"]).encode("utf-8"))

    _update_hash_ndarray(
        h, "background", np.asarray(pointcloud_payload["background"], dtype=np.float32)
    )
    _update_hash_ndarray(h, "dynamic_concat", np.asarray(dynamic_concat, dtype=np.float32))
    off = np.asarray(list(offsets), dtype=np.int64)
    _update_hash_ndarray(h, "dynamic_points_offsets", off)
    di = np.asarray(list(dynamic_intids), dtype=np.int32)
    _update_hash_ndarray(h, "dynamic_instance_intids", di)
    dor = np.asarray(list(dyn_orig_ids), dtype=np.int32)
    _update_hash_ndarray(h, "dynamic_instance_original_ids", dor)
    im = pointcloud_payload.get("instance_mapping", {})
    h.update(
        json.dumps({str(int(k)): int(im[k]) for k in sorted(im.keys())}, sort_keys=True).encode("utf-8")
    )
    h.update(
        json.dumps(pointcloud_payload.get("metadata", {}), sort_keys=True, ensure_ascii=True).encode(
            "utf-8"
        )
    )

    for key in sorted(dynamic_tracks_payload.keys()):
        _update_hash_ndarray(h, f"dynamic_tracks/{key}", np.asarray(dynamic_tracks_payload[key]))
    return h.hexdigest()[:8]


def _serialize_knn_init_payload(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    bg_map_raw = payload.get("background_avg_dist_by_k", {})
    dyn_map_raw = payload.get("dynamic_avg_dist_by_k", {})
    if not isinstance(bg_map_raw, dict):
        raise ValueError("knn payload field background_avg_dist_by_k must be a dict")
    if not isinstance(dyn_map_raw, dict):
        raise ValueError("knn payload field dynamic_avg_dist_by_k must be a dict")

    bg_ks = sorted(int(k) for k in bg_map_raw.keys())
    dyn_ks = sorted(int(k) for k in dyn_map_raw.keys())

    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray([1], dtype=np.int32),
        "background_ks": np.asarray(bg_ks, dtype=np.int32),
        "dynamic_ks": np.asarray(dyn_ks, dtype=np.int32),
    }

    bg_count: Optional[int] = None
    for k in bg_ks:
        arr = np.asarray(bg_map_raw[k]).astype(np.float32, copy=False).reshape(-1)
        if bg_count is None:
            bg_count = int(arr.shape[0])
        elif int(arr.shape[0]) != int(bg_count):
            raise ValueError(
                "all background kNN arrays must share the same length; "
                f"got {arr.shape[0]} vs {bg_count} for k={k}"
            )
        arrays[f"background_avg_dist_k{k}"] = arr

    dynamic_instance_intids: List[int] = []
    for k in dyn_ks:
        m = dyn_map_raw[k]
        if not isinstance(m, dict):
            raise ValueError(f"dynamic_avg_dist_by_k[{k}] must be a dict[intid -> np.ndarray]")
        ids = sorted(int(x) for x in m.keys())
        if not dynamic_instance_intids:
            dynamic_instance_intids = ids
        elif ids != dynamic_instance_intids:
            raise ValueError(
                f"dynamic instance id sets must match across k values; expected={dynamic_instance_intids}, got={ids}"
            )

    offsets: List[int] = [0]
    for intid in dynamic_instance_intids:
        first_arr: Optional[np.ndarray] = None
        for k in dyn_ks:
            arr = np.asarray(dyn_map_raw[k][int(intid)]).astype(np.float32, copy=False).reshape(-1)
            if first_arr is None:
                first_arr = arr
            elif int(arr.shape[0]) != int(first_arr.shape[0]):
                raise ValueError(
                    f"dynamic instance {intid} has inconsistent point count across k values: "
                    f"{arr.shape[0]} vs {first_arr.shape[0]}"
                )
        offsets.append(offsets[-1] + (int(first_arr.shape[0]) if first_arr is not None else 0))

    arrays["dynamic_instance_intids"] = np.asarray(dynamic_instance_intids, dtype=np.int32)
    arrays["dynamic_offsets"] = np.asarray(offsets, dtype=np.int64)

    for k in dyn_ks:
        parts: List[np.ndarray] = []
        for intid in dynamic_instance_intids:
            arr = np.asarray(dyn_map_raw[k][int(intid)]).astype(np.float32, copy=False).reshape(-1)
            parts.append(arr)
        concat = (
            np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            if parts
            else np.zeros((0,), dtype=np.float32)
        )
        arrays[f"dynamic_avg_dist_k{k}"] = concat
    return arrays


def _deserialize_knn_init_payload(raw: Dict[str, np.ndarray]) -> Dict[str, Any]:
    version_raw = np.asarray(raw.get("schema_version", np.asarray([1], dtype=np.int32))).reshape(-1)
    if int(version_raw[0]) != 1:
        raise ValueError(f"Unsupported knn_init schema_version={int(version_raw[0])}, expected 1")

    bg_ks = [int(x) for x in np.asarray(raw.get("background_ks", np.zeros((0,), dtype=np.int32))).tolist()]
    dyn_ks = [int(x) for x in np.asarray(raw.get("dynamic_ks", np.zeros((0,), dtype=np.int32))).tolist()]

    background_avg_dist_by_k: Dict[int, np.ndarray] = {}
    bg_count: Optional[int] = None
    for k in bg_ks:
        key = f"background_avg_dist_k{k}"
        if key not in raw:
            raise ValueError(f"knn_init missing required key: {key}")
        arr = np.asarray(raw[key]).astype(np.float32, copy=False).reshape(-1)
        if bg_count is None:
            bg_count = int(arr.shape[0])
        elif int(arr.shape[0]) != int(bg_count):
            raise ValueError(
                f"background kNN array length mismatch for k={k}: {arr.shape[0]} vs {bg_count}"
            )
        background_avg_dist_by_k[int(k)] = arr

    intids = np.asarray(raw.get("dynamic_instance_intids", np.zeros((0,), dtype=np.int32))).astype(np.int32, copy=False)
    offsets = np.asarray(raw.get("dynamic_offsets", np.zeros((1,), dtype=np.int64))).astype(np.int64, copy=False)
    if offsets.ndim != 1 or int(offsets.shape[0]) != int(intids.shape[0]) + 1:
        raise ValueError(
            "knn_init dynamic_offsets shape mismatch: "
            f"expected len={int(intids.shape[0]) + 1}, got {tuple(offsets.shape)}"
        )
    if int(offsets[0]) != 0:
        raise ValueError("knn_init dynamic_offsets must start at 0")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("knn_init dynamic_offsets must be non-decreasing")

    dynamic_avg_dist_by_k: Dict[int, Dict[int, np.ndarray]] = {}
    total = int(offsets[-1])
    for k in dyn_ks:
        key = f"dynamic_avg_dist_k{k}"
        if key not in raw:
            raise ValueError(f"knn_init missing required key: {key}")
        concat = np.asarray(raw[key]).astype(np.float32, copy=False).reshape(-1)
        if int(concat.shape[0]) != int(total):
            raise ValueError(
                f"knn_init {key} length mismatch: expected {total}, got {concat.shape[0]}"
            )
        per_instance: Dict[int, np.ndarray] = {}
        for i, intid in enumerate(intids.tolist()):
            lo = int(offsets[i])
            hi = int(offsets[i + 1])
            per_instance[int(intid)] = concat[lo:hi]
        dynamic_avg_dist_by_k[int(k)] = per_instance

    return {
        "background_avg_dist_by_k": background_avg_dist_by_k,
        "dynamic_avg_dist_by_k": dynamic_avg_dist_by_k,
    }


@dataclass(frozen=True)
class SegmentAssetHandle:
    asset_dir: Path

    def _require_ready(self) -> None:
        if not (self.asset_dir / "READY").exists():
            raise ValueError(f"asset missing READY marker: {self.asset_dir}")

    def load_manifest(self) -> Dict[str, Any]:
        self._require_ready()
        manifest = read_json(self.asset_dir / "manifest.json")
        require_manifest_fields(manifest, asset_type="streetforward_segment_init_asset")
        if "segment_id" not in manifest:
            raise ValueError("segment manifest missing segment_id")
        return manifest

    def load_segment_index(self) -> Dict[str, Any]:
        self._require_ready()
        raw = read_npz(self.asset_dir / "segment_index.npz")
        keyframe_to_frames = restore_keyframe_to_frames(
            raw["keyframe_indices_sorted"],
            raw["keyframe_to_frames_flat"],
            raw["keyframe_to_frames_offsets"],
        )
        frame_to_keyframe_dense = np.asarray(raw["frame_to_keyframe_dense"]).astype(np.int32, copy=False)
        frame_indices = [int(x) for x in np.asarray(raw["train_frame_indices"]).tolist()]
        frame_to_keyframe = {
            int(frame_indices[i]): int(frame_to_keyframe_dense[i])
            for i in range(len(frame_indices))
        }
        out = {
            "scene_id": int(raw["scene_id"][0]),
            "segment_id": int(raw["segment_id"][0]),
            "num_cams": int(raw["num_cams"][0]),
            "frame_indices": frame_indices,
            "test_frame_indices": [int(x) for x in np.asarray(raw["test_frame_indices"]).tolist()],
            "keyframe_indices": [int(x) for x in np.asarray(raw["segment_keyframe_indices"]).tolist()],
            "keyframe_to_frames": keyframe_to_frames,
            "frame_to_keyframe": frame_to_keyframe,
            "segment_first_frame_idx": int(raw["segment_first_frame_idx"][0]),
            "train_image_refs": np.asarray(raw["train_image_refs"]).astype(np.int32, copy=False),
            "test_image_refs": np.asarray(raw["test_image_refs"]).astype(np.int32, copy=False),
        }
        return out

    def load_segment_pose(self) -> Dict[str, Any]:
        self._require_ready()
        raw = read_npz(self.asset_dir / "segment_pose.npz")
        return {
            "segment_first_pose_world": torch.from_numpy(
                np.asarray(raw["segment_first_pose_world"]).astype(np.float32, copy=False)
            ),
            "world_to_seg0": torch.from_numpy(
                np.asarray(raw["world_to_seg0"]).astype(np.float32, copy=False)
            ),
            "segment_first_frame_idx": int(raw["segment_first_frame_idx"][0]),
            "segment_pose_source": str(raw["segment_pose_source"][0]),
        }

    def load_pointcloud(self) -> Dict[str, Any]:
        self._require_ready()
        static_npz = read_npz(self.asset_dir / "pointcloud_static.npz")
        dyn_npz = read_npz(self.asset_dir / "pointcloud_dynamic.npz")

        background = np.asarray(static_npz["background"]).astype(np.float32, copy=False)
        metadata_json = str(static_npz["metadata_json"][0])
        metadata = json.loads(metadata_json) if metadata_json else {}

        concat = np.asarray(dyn_npz["dynamic_points_concat"]).astype(np.float32, copy=False)
        offsets = np.asarray(dyn_npz["dynamic_points_offsets"]).astype(np.int64, copy=False)
        intids = np.asarray(dyn_npz["dynamic_instance_intids"]).astype(np.int32, copy=False)
        orig_ids = np.asarray(dyn_npz["dynamic_instance_original_ids"]).astype(np.int32, copy=False)
        mapping_keys = np.asarray(dyn_npz["instance_mapping_keys"]).astype(np.int32, copy=False)
        mapping_vals = np.asarray(dyn_npz["instance_mapping_values"]).astype(np.int32, copy=False)

        dynamic: Dict[int, np.ndarray] = {}
        for i, intid in enumerate(intids.tolist()):
            lo = int(offsets[i])
            hi = int(offsets[i + 1])
            dynamic[int(intid)] = concat[lo:hi]
        instance_mapping = {int(k): int(v) for k, v in zip(mapping_keys.tolist(), mapping_vals.tolist())}
        metadata["dynamic_instance_original_ids"] = [int(x) for x in orig_ids.tolist()]
        return {
            "background": background,
            "dynamic": dynamic,
            "instance_mapping": instance_mapping,
            "metadata": metadata,
        }

    def load_dynamic_tracks(self) -> Dict[str, Any]:
        self._require_ready()
        raw = read_npz(self.asset_dir / "dynamic_tracks.npz")
        return {
            "frame_indices": np.asarray(raw["frame_indices"]).astype(np.int32, copy=False),
            "instance_intids": np.asarray(raw["instance_intids"]).astype(np.int32, copy=False),
            "instances_quats": np.asarray(raw["instances_quats"]).astype(np.float32, copy=False),
            "instances_trans": np.asarray(raw["instances_trans"]).astype(np.float32, copy=False),
            "instances_fv": np.asarray(raw["instances_fv"]).astype(np.uint8, copy=False),
            "static_instance_intids": np.asarray(raw["static_instance_intids"]).astype(np.int32, copy=False),
        }

    def has_knn_init(self) -> bool:
        self._require_ready()
        return (self.asset_dir / "knn_init.npz").exists()

    def load_knn_init(self) -> Optional[Dict[str, Any]]:
        self._require_ready()
        path = self.asset_dir / "knn_init.npz"
        if not path.exists():
            return None
        raw = read_npz(path)
        return _deserialize_knn_init_payload(raw)


@dataclass(frozen=True)
class SceneAssetHandle:
    asset_dir: Path
    store: "StreetForwardAssetStore"
    dataset: str
    scene_id: int

    def _require_ready(self) -> None:
        if not (self.asset_dir / "READY").exists():
            raise ValueError(f"scene asset missing READY marker: {self.asset_dir}")

    def load_manifest(self) -> Dict[str, Any]:
        self._require_ready()
        manifest = read_json(self.asset_dir / "manifest.json")
        require_manifest_fields(manifest, asset_type="streetforward_scene_asset")
        return manifest

    def load_image_meta(self, refs: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
        self._require_ready()
        index = self.store._get_scene_image_table_index(
            dataset=self.dataset,
            scene_id=int(self.scene_id),
            asset_dir=self.asset_dir,
        )
        out: List[Dict[str, Any]] = []
        for ref in refs:
            key = (int(ref[0]), int(ref[1]))
            if key not in index:
                raise ValueError(
                    f"image_ref {key} not found in scene image table (dataset={self.dataset} scene={self.scene_id})"
                )
            out.append(dict(index[key]))
        return out


class StreetForwardAssetStore:
    def __init__(self, root: str, *, missing_policy: str = "error") -> None:
        self.root = Path(root).resolve()
        self.missing_policy = normalize_missing_policy(missing_policy)
        self.scene_pool_dir = self.root / "scene_pool"
        self.segment_pool_dir = self.root / "segment_pool"
        self.tmp_dir = self.root / "tmp"
        self.registries_dir = self.root / "registries"
        self._scene_image_table_cache: Dict[Tuple[str, int, str], Dict[Tuple[int, int], Dict[str, Any]]] = {}
        self._registry_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _resolve_segment_dir(self, dataset: str, scene_id: int, segment_id: int) -> Optional[Path]:
        prefix = f"seg-{dataset}-{int(scene_id):06d}-{int(segment_id):06d}-"
        matches = list_asset_dirs_by_prefix(self.segment_pool_dir, prefix)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def _resolve_scene_dir(self, dataset: str, scene_id: int) -> Optional[Path]:
        prefix = f"scene-{dataset}-{int(scene_id):06d}-"
        matches = list_asset_dirs_by_prefix(self.scene_pool_dir, prefix)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def has_segment_asset(self, dataset: str, scene_id: int, segment_id: int) -> bool:
        if self._resolve_segment_asset_id_from_registry(dataset, scene_id, segment_id) is not None:
            return True
        return self._resolve_segment_dir(dataset, scene_id, segment_id) is not None

    def has_segment_knn_init_asset(self, dataset: str, scene_id: int, segment_id: int) -> bool:
        try:
            handle = self.get_segment_asset_registry_first(dataset, scene_id, segment_id)
        except ValueError:
            return False
        return handle.has_knn_init()

    def get_scene_asset(self, dataset: str, scene_id: int) -> SceneAssetHandle:
        scene_asset_id = self._resolve_scene_asset_id_from_scene_registry(dataset, scene_id)
        if scene_asset_id is not None:
            return self.get_scene_asset_by_asset_id(
                scene_asset_id, dataset=str(dataset), scene_id=int(scene_id)
            )
        p = self._resolve_scene_dir(dataset, scene_id)
        if p is None:
            raise ValueError(f"scene asset not found: dataset={dataset} scene={scene_id}")
        handle = SceneAssetHandle(
            asset_dir=p,
            store=self,
            dataset=str(dataset),
            scene_id=int(scene_id),
        )
        handle._require_ready()
        return handle

    def get_segment_asset(self, dataset: str, scene_id: int, segment_id: int) -> SegmentAssetHandle:
        seg_asset_id = self._resolve_segment_asset_id_from_registry(dataset, scene_id, segment_id)
        if seg_asset_id is not None:
            return self._get_segment_handle_for_registry_asset_id(
                dataset, scene_id, segment_id, seg_asset_id
            )
        p = self._resolve_segment_dir(dataset, scene_id, segment_id)
        if p is None:
            raise ValueError(
                f"segment asset not found: dataset={dataset} scene={scene_id} segment={segment_id}"
            )
        return SegmentAssetHandle(asset_dir=p)

    def _load_registry_rows(self, registry_name: str) -> List[Dict[str, Any]]:
        if registry_name in self._registry_cache:
            return list(self._registry_cache[registry_name])
        path = self.registries_dir / registry_name
        if not path.exists():
            self._registry_cache[registry_name] = []
            return []
        rows = read_parquet_table(path)
        self._registry_cache[registry_name] = list(rows)
        return rows

    def get_scene_asset_by_asset_id(
        self,
        scene_asset_id: str,
        *,
        dataset: Optional[str] = None,
        scene_id: Optional[int] = None,
    ) -> SceneAssetHandle:
        asset_dir = self.scene_pool_dir / str(scene_asset_id)
        if not asset_dir.exists():
            raise ValueError(f"scene asset_id not found in scene_pool: {scene_asset_id}")
        handle = SceneAssetHandle(
            asset_dir=asset_dir,
            store=self,
            dataset=str(dataset) if dataset is not None else "",
            scene_id=int(scene_id) if scene_id is not None else -1,
        )
        manifest = handle.load_manifest()
        if str(manifest["asset_id"]) != str(scene_asset_id):
            raise ValueError(
                f"scene asset_id mismatch: requested={scene_asset_id} got={manifest.get('asset_id')}"
            )
        if dataset is not None and str(manifest["dataset"]) != str(dataset):
            raise ValueError(
                f"scene asset dataset mismatch: expected={dataset} got={manifest['dataset']}"
            )
        if scene_id is not None and int(manifest["scene_id"]) != int(scene_id):
            raise ValueError(
                f"scene asset scene_id mismatch: expected={scene_id} got={manifest['scene_id']}"
            )
        return SceneAssetHandle(
            asset_dir=asset_dir,
            store=self,
            dataset=str(manifest["dataset"]),
            scene_id=int(manifest["scene_id"]),
        )

    def _resolve_scene_asset_id_from_scene_registry(self, dataset: str, scene_id: int) -> Optional[str]:
        rows = self._load_registry_rows("scene_registry.parquet")
        matches = [
            r
            for r in rows
            if str(r.get("dataset")) == str(dataset) and int(r.get("scene_id")) == int(scene_id)
        ]
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda r: int(r.get("created_at_unix", 0)), reverse=True)
        return str(matches[0]["scene_asset_id"])

    def _get_segment_handle_for_registry_asset_id(
        self,
        dataset: str,
        scene_id: int,
        segment_id: int,
        seg_asset_id: str,
    ) -> SegmentAssetHandle:
        asset_dir = self.segment_pool_dir / str(seg_asset_id)
        if not asset_dir.exists():
            raise ValueError(
                f"segment registry points to missing asset directory: {seg_asset_id}"
            )
        handle = SegmentAssetHandle(asset_dir=asset_dir)
        manifest = handle.load_manifest()
        if str(manifest["dataset"]) != str(dataset):
            raise ValueError(
                f"segment asset dataset mismatch: expected={dataset} got={manifest['dataset']}"
            )
        if int(manifest["scene_id"]) != int(scene_id) or int(manifest["segment_id"]) != int(segment_id):
            raise ValueError(
                "segment asset scene/segment mismatch: "
                f"expected=({scene_id},{segment_id}) got=({manifest['scene_id']},{manifest['segment_id']})"
            )
        return handle

    def _resolve_segment_asset_id_from_registry(
        self,
        dataset: str,
        scene_id: int,
        segment_id: int,
    ) -> Optional[str]:
        rows = self._load_registry_rows("segment_registry.parquet")
        matches = [
            r
            for r in rows
            if str(r.get("dataset")) == str(dataset)
            and int(r.get("scene_id")) == int(scene_id)
            and int(r.get("segment_id")) == int(segment_id)
        ]
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda r: int(r.get("created_at_unix", 0)), reverse=True)
        return str(matches[0]["segment_asset_id"])

    def list_registered_scene_ids(self, dataset: str) -> List[int]:
        rows = self._load_registry_rows("segment_registry.parquet")
        out = sorted(
            {
                int(r.get("scene_id"))
                for r in rows
                if str(r.get("dataset")) == str(dataset)
            }
        )
        return out

    def list_registered_segment_ids(self, dataset: str, scene_id: int) -> List[int]:
        rows = self._load_registry_rows("segment_registry.parquet")
        out = sorted(
            {
                int(r.get("segment_id"))
                for r in rows
                if str(r.get("dataset")) == str(dataset)
                and int(r.get("scene_id")) == int(scene_id)
            }
        )
        return out

    def get_segment_asset_registry_first(
        self,
        dataset: str,
        scene_id: int,
        segment_id: int,
    ) -> SegmentAssetHandle:
        seg_asset_id = self._resolve_segment_asset_id_from_registry(dataset, scene_id, segment_id)
        if seg_asset_id is None:
            raise ValueError(
                "segment is not registered in segment_registry.parquet (registry-first resolution only): "
                f"dataset={dataset} scene_id={scene_id} segment_id={segment_id}"
            )
        return self._get_segment_handle_for_registry_asset_id(
            dataset, scene_id, segment_id, seg_asset_id
        )

    def resolve_segment_scene_assets_registry_first(
        self,
        dataset: str,
        scene_id: int,
        segment_id: int,
    ) -> Dict[str, Any]:
        segment_handle = self.get_segment_asset_registry_first(dataset, scene_id, segment_id)
        segment_manifest = segment_handle.load_manifest()
        parent_scene_asset_id = str(segment_manifest["parent_scene_asset_id"])
        scene_handle = self.get_scene_asset_by_asset_id(
            parent_scene_asset_id,
            dataset=str(dataset),
            scene_id=int(scene_id),
        )
        return {
            "segment_handle": segment_handle,
            "segment_manifest": segment_manifest,
            "scene_handle": scene_handle,
            "parent_scene_asset_id": parent_scene_asset_id,
        }

    def verify_segment_asset(
        self,
        dataset: str,
        scene_id: int,
        segment_id: int,
        *,
        expected_fingerprints: Optional[Dict[str, str]] = None,
    ) -> SegmentAssetHandle:
        handle = self.get_segment_asset(dataset, scene_id, segment_id)
        manifest = handle.load_manifest()
        if expected_fingerprints:
            for key, expected in expected_fingerprints.items():
                got = str(manifest.get(key))
                if got != str(expected):
                    raise ValueError(
                        f"segment asset fingerprint mismatch for {key}: expected {expected!r}, got {got!r}"
                    )
        return handle

    def _get_scene_image_table_index(
        self,
        *,
        dataset: str,
        scene_id: int,
        asset_dir: Path,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        cache_key = (str(dataset), int(scene_id), str(asset_dir.resolve()))
        cached = self._scene_image_table_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = read_parquet_table(asset_dir / "image_table.parquet")
        idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for row in rows:
            key = (int(row["frame_idx"]), int(row["cam_id"]))
            if key in idx:
                raise ValueError(f"duplicate image_table key (frame_idx,cam_id)={key} in {asset_dir}")
            idx[key] = row
        self._scene_image_table_cache[cache_key] = idx
        return idx

    def export_scene_asset(
        self,
        *,
        dataset: str,
        scene_id: int,
        scene_name: str,
        num_frames: int,
        num_cams: int,
        split_config: Dict[str, Any],
        scene_index_arrays: Dict[str, Any],
        image_table_rows: List[Dict[str, Any]],
    ) -> str:
        fp_short = stable_scene_asset_id_suffix(
            dataset=str(dataset),
            scene_id=int(scene_id),
            num_frames=int(num_frames),
            num_cams=int(num_cams),
            split_config=split_config,
            scene_index_arrays=scene_index_arrays,
            image_table_rows=image_table_rows,
        )
        asset_id = f"scene-{dataset}-{int(scene_id):06d}-{fp_short}"
        final_dir = self.scene_pool_dir / asset_id
        now = int(time.time())
        manifest = {
            "asset_type": "streetforward_scene_asset",
            "schema_version": SCHEMA_VERSION,
            "asset_id": asset_id,
            "dataset": str(dataset),
            "scene_id": int(scene_id),
            "scene_name": str(scene_name),
            "num_frames": int(num_frames),
            "num_cams": int(num_cams),
            "source_data_fingerprint": _fingerprint_placeholder(f"scene-source-{scene_id}"),
            "config_fingerprint": _fingerprint_placeholder(f"scene-config-{scene_id}"),
            "implementation_fingerprint": _fingerprint_placeholder("streetforward-v3"),
            "image_table_version": 1,
            "split_config": split_config,
            "created_at_unix": now,
        }
        require_manifest_fields(manifest, asset_type="streetforward_scene_asset")

        def _writer(tmp_dir: Path) -> None:
            write_npz(tmp_dir / "scene_index.npz", scene_index_arrays)
            write_parquet_table(tmp_dir / "image_table.parquet", image_table_rows)
            write_json(tmp_dir / "split_summary.json", {"scene_id": int(scene_id)})
            write_json(tmp_dir / "manifest.json", manifest)

        atomic_write_asset_dir(final_dir=final_dir, writer=_writer, tmp_root=self.tmp_dir)
        append_registry_row(
            self.registries_dir / "scene_registry.parquet",
            {
                "dataset": str(dataset),
                "scene_id": int(scene_id),
                "scene_asset_id": asset_id,
                "created_at_unix": now,
            },
        )
        self._registry_cache.pop("scene_registry.parquet", None)
        return asset_id

    def export_segment_asset(
        self,
        *,
        dataset: str,
        scene_id: int,
        segment_id: int,
        parent_scene_asset_id: str,
        segment_index_payload: Dict[str, Any],
        segment_pose_payload: Dict[str, Any],
        pointcloud_payload: Dict[str, Any],
        dynamic_tracks_payload: Dict[str, Any],
        segment_aabb: Any,
        pointcloud_config_normalized: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> str:
        dynamic = pointcloud_payload.get("dynamic", {})
        instance_mapping = pointcloud_payload.get("instance_mapping", {})
        metadata = pointcloud_payload.get("metadata", {})
        dynamic_intids = sorted(int(k) for k in dynamic.keys())
        concat_parts = []
        offsets: List[int] = [0]
        for intid in dynamic_intids:
            pts = np.asarray(dynamic[intid], dtype=np.float32)
            concat_parts.append(pts)
            offsets.append(offsets[-1] + int(pts.shape[0]))
        if concat_parts:
            dynamic_concat = np.concatenate(concat_parts, axis=0).astype(np.float32, copy=False)
        else:
            dynamic_concat = np.zeros((0, 6), dtype=np.float32)
        dyn_orig_ids: List[int] = []
        for intid in dynamic_intids:
            found_orig = None
            for k, v in instance_mapping.items():
                if int(v) == int(intid):
                    found_orig = int(k)
                    break
            dyn_orig_ids.append(-1 if found_orig is None else int(found_orig))

        fp_short = stable_segment_asset_id_suffix(
            dataset=str(dataset),
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            parent_scene_asset_id=str(parent_scene_asset_id),
            segment_index_payload=segment_index_payload,
            segment_pose_payload=segment_pose_payload,
            pointcloud_payload=pointcloud_payload,
            dynamic_tracks_payload=dynamic_tracks_payload,
            segment_aabb=segment_aabb,
            pointcloud_config_normalized=pointcloud_config_normalized,
            dynamic_concat=dynamic_concat,
            dynamic_intids=dynamic_intids,
            dyn_orig_ids=dyn_orig_ids,
            offsets=offsets,
        )
        asset_id = f"seg-{dataset}-{int(scene_id):06d}-{int(segment_id):06d}-{fp_short}"
        final_dir = self.segment_pool_dir / asset_id
        now = int(time.time())
        manifest = {
            "asset_type": "streetforward_segment_init_asset",
            "schema_version": SCHEMA_VERSION,
            "asset_id": asset_id,
            "dataset": str(dataset),
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "parent_scene_asset_id": str(parent_scene_asset_id),
            "segment_first_frame_idx": int(segment_pose_payload["segment_first_frame_idx"]),
            "segment_pose_source": str(segment_pose_payload["segment_pose_source"]),
            "seg0_camera_id": 0,
            "segment_aabb": np.asarray(segment_aabb, dtype=np.float32).tolist(),
            "pointcloud_config_normalized": pointcloud_config_normalized,
            "stats": stats,
            "source_data_fingerprint": _fingerprint_placeholder(
                f"segment-source-{scene_id}-{segment_id}"
            ),
            "config_fingerprint": _fingerprint_placeholder(f"segment-config-{scene_id}-{segment_id}"),
            "implementation_fingerprint": _fingerprint_placeholder("streetforward-v3"),
            "created_at_unix": now,
        }
        require_manifest_fields(manifest, asset_type="streetforward_segment_init_asset")

        def _writer(tmp_dir: Path) -> None:
            kf_flat = flatten_keyframe_to_frames(segment_index_payload["keyframe_to_frames"])
            frame_indices = [int(x) for x in segment_index_payload["frame_indices"]]
            frame_to_key = segment_index_payload["frame_to_keyframe"]
            frame_to_key_dense = np.asarray([int(frame_to_key[int(f)]) for f in frame_indices], dtype=np.int32)
            write_npz(
                tmp_dir / "segment_index.npz",
                {
                    "scene_id": np.asarray([int(scene_id)], dtype=np.int32),
                    "segment_id": np.asarray([int(segment_id)], dtype=np.int32),
                    "num_cams": np.asarray([int(segment_index_payload["num_cams"])], dtype=np.int32),
                    "train_frame_indices": np.asarray(frame_indices, dtype=np.int32),
                    "test_frame_indices": np.asarray(
                        [int(x) for x in segment_index_payload["test_frame_indices"]], dtype=np.int32
                    ),
                    "segment_keyframe_indices": np.asarray(
                        [int(x) for x in segment_index_payload["keyframe_indices"]], dtype=np.int32
                    ),
                    "segment_first_frame_idx": np.asarray(
                        [int(segment_index_payload["segment_first_frame_idx"])], dtype=np.int32
                    ),
                    "frame_to_keyframe_dense": frame_to_key_dense,
                    "train_image_refs": np.asarray(segment_index_payload["train_image_refs"], dtype=np.int32),
                    "test_image_refs": np.asarray(segment_index_payload["test_image_refs"], dtype=np.int32),
                    **kf_flat,
                },
            )
            write_npz(
                tmp_dir / "segment_pose.npz",
                {
                    "segment_first_pose_world": np.asarray(
                        segment_pose_payload["segment_first_pose_world"], dtype=np.float32
                    ),
                    "world_to_seg0": np.asarray(segment_pose_payload["world_to_seg0"], dtype=np.float32),
                    "segment_first_frame_idx": np.asarray(
                        [int(segment_pose_payload["segment_first_frame_idx"])], dtype=np.int32
                    ),
                    "segment_pose_source": np.asarray(
                        [str(segment_pose_payload["segment_pose_source"])], dtype="<U32"
                    ),
                },
            )
            write_npz(
                tmp_dir / "pointcloud_static.npz",
                {
                    "background": np.asarray(pointcloud_payload["background"], dtype=np.float32),
                    "metadata_json": np.asarray([json.dumps(metadata, ensure_ascii=True)], dtype="<U65535"),
                },
            )
            write_npz(
                tmp_dir / "pointcloud_dynamic.npz",
                {
                    "dynamic_points_concat": dynamic_concat,
                    "dynamic_points_offsets": np.asarray(offsets, dtype=np.int64),
                    "dynamic_instance_intids": np.asarray(dynamic_intids, dtype=np.int32),
                    "dynamic_instance_original_ids": np.asarray(dyn_orig_ids, dtype=np.int32),
                    "instance_mapping_keys": np.asarray(
                        [int(k) for k in sorted(instance_mapping.keys())], dtype=np.int32
                    ),
                    "instance_mapping_values": np.asarray(
                        [int(instance_mapping[k]) for k in sorted(instance_mapping.keys())], dtype=np.int32
                    ),
                },
            )
            write_npz(tmp_dir / "dynamic_tracks.npz", dynamic_tracks_payload)
            write_json(tmp_dir / "stats.json", stats)
            write_json(tmp_dir / "manifest.json", manifest)

        atomic_write_asset_dir(final_dir=final_dir, writer=_writer, tmp_root=self.tmp_dir)
        append_registry_row(
            self.registries_dir / "segment_registry.parquet",
            {
                "dataset": str(dataset),
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "scene_asset_id": str(parent_scene_asset_id),
                "segment_asset_id": asset_id,
                "created_at_unix": now,
            },
        )
        self._registry_cache.pop("segment_registry.parquet", None)
        return asset_id

    def export_segment_knn_init_asset(
        self,
        *,
        dataset: str,
        scene_id: int,
        segment_id: int,
        knn_payload: Dict[str, Any],
        overwrite: bool = False,
    ) -> bool:
        """
        Export precomputed kNN init payload into an existing segment asset directory.

        Returns True when a new file is written, False when skipped due to existing file and overwrite=False.
        """
        handle = self.get_segment_asset_registry_first(str(dataset), int(scene_id), int(segment_id))
        final_path = handle.asset_dir / "knn_init.npz"
        if final_path.exists() and not bool(overwrite):
            return False

        arrays = _serialize_knn_init_payload(knn_payload)
        tmp_name = f"{final_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        tmp_path = self.tmp_dir / tmp_name
        try:
            write_npz(tmp_path, arrays)
            os.replace(str(tmp_path), str(final_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return True
