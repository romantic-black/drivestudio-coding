from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor

from datasets.asset_preload_manager_v2 import (
    PRIORITY_EPISODE_SUPERSET,
    PRIORITY_NEXT_BLOCK_EXACT,
    PRIORITY_SEGMENT_STATIC,
    PRIORITY_TEST_REFS,
    AssetPreloadManagerV2,
    LoadedViewPackV2,
    coerce_preload_cfg_dict_v2,
    dict_to_loaded_view_pack_v2,
    loaded_view_pack_to_device_v2,
    parse_preload_cfg_v2,
)
from datasets.sky_mask_semantics import normalize_sky_mask_to_one_is_sky
from datasets.streetforward_assets import StreetForwardAssetStore
from datasets.train_scheduler_v6 import TrainSchedulerV6
from datasets.train_scheduler_v7 import TrainSchedulerV7

ImageRef = Tuple[int, int]

logger = logging.getLogger(__name__)

_POINTCLOUD_CAP_KEYS = (
    "near_max_points",
    "distant_max_points",
    "monocular_dynamic_recovery_max_points_per_instance",
)

_V4_CACHE_MAX_ITEM_KEYS = (
    "scene_asset_cache_max_items",
    "segment_static_cache_max_items",
    "image_meta_cache_max_items",
    "view_pack_cache_max_items",
)


@dataclass(frozen=True)
class SegmentIndexV4:
    scene_id: int
    segment_id: int
    num_cams: int
    frame_indices: List[int]
    test_frame_indices: List[int]
    train_frame_set: FrozenSet[int]
    test_frame_set: FrozenSet[int]
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    frame_to_keyframe: Dict[int, int]
    segment_first_frame_idx: int
    train_image_refs: Tuple[ImageRef, ...]
    test_image_refs: Tuple[ImageRef, ...]


@dataclass(frozen=True)
class BatchRequestV4:
    scene_id: int
    segment_id: int
    source_image_ref: ImageRef
    target_image_refs: List[ImageRef]
    source_image_refs: Optional[List[ImageRef]] = None
    include_test: bool = False
    test_image_refs: Optional[List[ImageRef]] = None


@dataclass(frozen=True)
class EvalRequestV4:
    scene_id: int
    segment_id: int
    source_image_ref: ImageRef
    eval_image_refs: List[ImageRef]


@dataclass(frozen=True)
class SegmentStaticBundle:
    segment_asset_id: str
    parent_scene_asset_id: str
    segment_index: SegmentIndexV4
    segment_aabb: Tensor
    segment_pose: Dict[str, Any]
    pointcloud: Dict[str, Any]
    dynamic_tracks: Dict[str, Any]


def _cap_int_or_none(d: Dict[str, Any], k: str) -> Optional[int]:
    v = d.get(k)
    if v is None:
        return None
    i = int(v)
    if i <= 0:
        raise ValueError(f"dataset.pointcloud.{k} must be > 0 when set, got {v!r}")
    return i


def _pointcloud_cap_triplet_differs(asset_pc: Dict[str, Any], runtime_pc: Dict[str, Any]) -> bool:
    for key in _POINTCLOUD_CAP_KEYS:
        if _cap_int_or_none(asset_pc, key) != _cap_int_or_none(runtime_pc, key):
            return True
    return False


def _make_cap_downsample_rng(segment_manifest: Dict[str, Any], runtime_pc: Dict[str, Any]) -> np.random.Generator:
    payload = {
        "asset_id": str(segment_manifest["asset_id"]),
        "near_max_points": _cap_int_or_none(runtime_pc, "near_max_points"),
        "distant_max_points": _cap_int_or_none(runtime_pc, "distant_max_points"),
        "monocular_dynamic_recovery_max_points_per_instance": _cap_int_or_none(
            runtime_pc, "monocular_dynamic_recovery_max_points_per_instance"
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(seed)


def _random_subset_rows(arr: np.ndarray, max_count: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_count is None:
        return arr
    n = int(arr.shape[0])
    if n == 0:
        return arr
    m = int(max_count)
    if m <= 0:
        raise ValueError("max_count must be > 0 when set")
    if n <= m:
        return arr
    idx = rng.choice(n, size=m, replace=False)
    idx.sort()
    return arr[idx].astype(np.float32, copy=False)


def _split_background_near_distant(background: np.ndarray, segment_aabb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg_aabb = np.asarray(segment_aabb, dtype=np.float32).reshape(2, 3)
    crop_min = seg_aabb[0]
    crop_max = seg_aabb[1]
    xyz = background[:, :3].astype(np.float32, copy=False)
    in_crop = ((xyz >= crop_min[None, :]) & (xyz <= crop_max[None, :])).all(axis=1)
    near = background[in_crop]
    distant = background[~in_crop]
    return near, distant


def _apply_runtime_random_pointcloud_downsample(
    *,
    pointcloud: Dict[str, Any],
    segment_manifest: Dict[str, Any],
    segment_aabb: Tensor,
    runtime_pc: Dict[str, Any],
) -> Dict[str, Any]:
    asset_pc = segment_manifest.get("pointcloud_config_normalized")
    if not isinstance(asset_pc, dict):
        return pointcloud
    if not _pointcloud_cap_triplet_differs(asset_pc, runtime_pc):
        return pointcloud

    r_near = _cap_int_or_none(runtime_pc, "near_max_points")
    r_distant = _cap_int_or_none(runtime_pc, "distant_max_points")
    r_mono = _cap_int_or_none(runtime_pc, "monocular_dynamic_recovery_max_points_per_instance")
    if r_near is None and r_distant is None and r_mono is None:
        return pointcloud

    rng = _make_cap_downsample_rng(segment_manifest, runtime_pc)
    out = dict(pointcloud)
    aabb_np = segment_aabb.detach().cpu().numpy() if torch.is_tensor(segment_aabb) else np.asarray(segment_aabb)

    bg = np.asarray(out.get("background"), dtype=np.float32)
    if bg.size > 0:
        near, distant = _split_background_near_distant(bg, aabb_np)
        near = _random_subset_rows(near, r_near, rng)
        distant = _random_subset_rows(distant, r_distant, rng)
        if near.size == 0 and distant.size == 0:
            out["background"] = np.zeros((0, 6), dtype=np.float32)
        else:
            out["background"] = np.concatenate([near, distant], axis=0).astype(np.float32, copy=False)

    dyn = out.get("dynamic")
    if isinstance(dyn, dict) and len(dyn) > 0 and r_mono is not None:
        new_dyn: Dict[int, np.ndarray] = {}
        for intid, pts in dyn.items():
            arr = np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                new_dyn[int(intid)] = arr
            else:
                new_dyn[int(intid)] = _random_subset_rows(arr, r_mono, rng)
        out["dynamic"] = new_dyn

    ds_name = segment_manifest.get("dataset")
    scene_id = segment_manifest.get("scene_id")
    seg_id = segment_manifest.get("segment_id")
    logger.debug(
        "Runtime pointcloud caps differ from segment asset manifest; applied random subsample to runtime caps "
        "(dataset=%s scene_id=%s segment_id=%s asset_id=%s)",
        ds_name,
        scene_id,
        seg_id,
        segment_manifest.get("asset_id"),
    )
    return out


class MultiSceneDatasetV4:
    def __init__(
        self,
        *,
        dataset_cfg: Any,
        data_cfg: Any,
        device: torch.device,
        asset_store: Optional[StreetForwardAssetStore] = None,
        preload_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.dataset_cfg = dataset_cfg
        self.device = device
        self._lock = threading.RLock()
        self._initialized = False

        assets_cfg = self._cfg_get(self.data_cfg, "assets")
        if asset_store is not None:
            self.asset_store = asset_store
        else:
            assets_root = self._cfg_get(assets_cfg, "root")
            if assets_root is None:
                raise ValueError("MultiSceneDatasetV4 requires data.assets.root")
            self.asset_store = StreetForwardAssetStore(str(assets_root), missing_policy="error")
        self.asset_missing_policy = "error"

        seg_aabb = self._cfg_get(self.dataset_cfg, "segment_aabb")
        if seg_aabb is None:
            raise ValueError("MultiSceneDatasetV4 requires dataset.segment_aabb")
        segment_aabb = torch.as_tensor(seg_aabb, dtype=torch.float32)
        if segment_aabb.shape != (2, 3):
            raise ValueError(f"dataset.segment_aabb must have shape [2,3], got {tuple(segment_aabb.shape)}")
        if not torch.all(segment_aabb[0] < segment_aabb[1]):
            raise ValueError("dataset.segment_aabb min must be strictly less than max for all axes")
        self.segment_aabb = segment_aabb

        pixel_source_cfg = self._cfg_get(self.data_cfg, "pixel_source", {}) or {}
        self._load_sky_mask = bool(self._cfg_get(pixel_source_cfg, "load_sky_mask", False))
        self._load_dynamic_mask = bool(self._cfg_get(pixel_source_cfg, "load_dynamic_mask", False))
        self._sky_mask_loader_semantics = self._parse_sky_mask_semantics()

        (
            self._scene_asset_cache_max_items,
            self._segment_static_cache_max_items,
            self._image_meta_cache_max_items,
            self._view_pack_cache_max_items,
        ) = self._parse_required_cache_max_items()

        self._scene_asset_cache: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()
        self._segment_static_cache: "OrderedDict[Tuple[str, int, int], SegmentStaticBundle]" = OrderedDict()
        self._image_meta_cache: "OrderedDict[Tuple[str, int, int, int], Dict[str, Any]]" = OrderedDict()
        self._view_pack_cache: "OrderedDict[Tuple[str, int, int, int, int], LoadedViewPackV2]" = OrderedDict()
        self._pair_score_cache: Dict[Tuple[str, ImageRef, ImageRef, str], float] = {}

        self._segment_bundle_inflight: Dict[Tuple[str, int, int], threading.Event] = {}
        self._segment_bundle_inflight_lock = threading.Lock()
        self._image_meta_inflight: Dict[Tuple[str, int, int, int], threading.Event] = {}
        self._image_meta_inflight_lock = threading.Lock()
        self._view_pack_inflight: Dict[Tuple[str, int, int, int, int], threading.Event] = {}
        self._view_pack_inflight_lock = threading.Lock()

        self._preload_active_scene_id: Optional[int] = None
        self._preload_active_segment_id: Optional[int] = None
        self._preload_training_scene_id: Optional[int] = None
        self._preload_training_segment_id: Optional[int] = None

        effective_preload: Optional[Dict[str, Any]]
        if preload_cfg is None:
            effective_preload = coerce_preload_cfg_dict_v2(self._cfg_get(self.data_cfg, "preload"))
        elif isinstance(preload_cfg, dict):
            effective_preload = dict(preload_cfg)
        else:
            effective_preload = coerce_preload_cfg_dict_v2(preload_cfg)
        self._preload_rtcfg = parse_preload_cfg_v2(effective_preload)
        self._preload_manager: Optional[AssetPreloadManagerV2] = None
        if self._preload_rtcfg is not None:
            self._preload_manager = AssetPreloadManagerV2(self, self._preload_rtcfg)
        self._enable_view_pack_cache = True
        if self._preload_rtcfg is not None:
            self._enable_view_pack_cache = bool(self._preload_rtcfg.enable_view_pack_cache)

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, default)
            except TypeError:
                pass
        return getattr(cfg, key, default)

    def _parse_required_cache_max_items(self) -> Tuple[int, int, int, int]:
        missing = [k for k in _V4_CACHE_MAX_ITEM_KEYS if self._cfg_get(self.data_cfg, k) is None]
        if missing:
            raise ValueError(
                "MultiSceneDatasetV4 requires explicit cache size limits on data_cfg (no implicit unbounded "
                f"caches). Missing keys: {missing}"
            )
        out: List[int] = []
        for key in _V4_CACHE_MAX_ITEM_KEYS:
            val = int(self._cfg_get(self.data_cfg, key))
            if val < 0:
                raise ValueError(f"data.{key} must be >= 0, got {val}")
            out.append(val)
        return int(out[0]), int(out[1]), int(out[2]), int(out[3])

    def _parse_sky_mask_semantics(self) -> Optional[str]:
        if not self._load_sky_mask:
            return None
        raw = self._cfg_get(self.data_cfg, "sky_mask_semantics")
        if raw is None:
            raise ValueError(
                "data.sky_mask_semantics is required when pixel_source.load_sky_mask is true. "
                "Use one_is_non_sky if PNG nonzero means non-sky, or one_is_sky if nonzero means sky."
            )
        s = str(raw).strip()
        if s not in ("one_is_sky", "one_is_non_sky"):
            raise ValueError(
                f"data.sky_mask_semantics must be one_is_sky or one_is_non_sky, got {raw!r}"
            )
        return s

    def _asset_dataset_name(self) -> str:
        ds = self._cfg_get(self.data_cfg, "dataset")
        if ds is None:
            raise ValueError("data.dataset is required for MultiSceneDatasetV4")
        return str(ds)

    def list_training_scene_ids(self) -> List[int]:
        ds_name = self._asset_dataset_name()
        configured = [int(x) for x in list(self._cfg_get(self.data_cfg, "train_scene_ids", []))]
        registered = self.asset_store.list_registered_scene_ids(ds_name)
        if len(configured) == 0:
            if len(registered) == 0:
                raise ValueError("No training scenes: train_scene_ids empty and segment registry has no rows")
            return registered
        reg_set = set(registered)
        out = [sid for sid in configured if sid in reg_set] if len(reg_set) > 0 else list(configured)
        if len(out) == 0:
            raise ValueError(
                f"No train scenes from config exist in segment registry (dataset={ds_name}, configured={configured})"
            )
        return out

    def list_segment_ids(self, scene_id: int) -> List[int]:
        ds_name = self._asset_dataset_name()
        seg_ids = self.asset_store.list_registered_segment_ids(ds_name, int(scene_id))
        if len(seg_ids) == 0:
            raise ValueError(
                f"No registered segments for dataset={ds_name} scene_id={int(scene_id)} in segment_registry"
            )
        return seg_ids

    def _cache_get(
        self,
        cache: "OrderedDict[Any, Any]",
        key: Any,
    ) -> Any:
        val = cache.get(key)
        if val is not None:
            cache.move_to_end(key)
        return val

    def _cache_set(
        self,
        cache: "OrderedDict[Any, Any]",
        key: Any,
        value: Any,
        *,
        max_items: Optional[int],
    ) -> bool:
        if max_items == 0:
            return False
        if key not in cache:
            cache[key] = value
        cache.move_to_end(key)
        if max_items is not None:
            while len(cache) > max_items:
                cache.popitem(last=False)
        return True

    def _wait_on_inflight(
        self,
        inflight: Dict[Any, threading.Event],
        lock: threading.Lock,
        key: Any,
    ) -> Optional[threading.Event]:
        with lock:
            existing = inflight.get(key)
            if existing is not None:
                return existing
            ev = threading.Event()
            inflight[key] = ev
            return None

    def _finish_inflight(
        self,
        inflight: Dict[Any, threading.Event],
        lock: threading.Lock,
        key: Any,
    ) -> None:
        with lock:
            ev = inflight.pop(key, None)
        if ev is not None:
            ev.set()

    def _validate_training_assets(self) -> None:
        ds_name = self._asset_dataset_name()
        for scene_id in self.list_training_scene_ids():
            for segment_id in self.list_segment_ids(scene_id):
                resolved = self.asset_store.resolve_segment_scene_assets_registry_first(
                    ds_name, int(scene_id), int(segment_id)
                )
                segment_manifest = resolved["segment_manifest"]
                asset_aabb = segment_manifest.get("segment_aabb")
                if asset_aabb is None:
                    raise ValueError(
                        f"segment manifest missing segment_aabb (dataset={ds_name} scene={scene_id} seg={segment_id})"
                    )
                segment_aabb = torch.as_tensor(asset_aabb, dtype=torch.float32)
                if segment_aabb.shape != (2, 3):
                    raise ValueError(
                        f"segment manifest segment_aabb must have shape [2,3] "
                        f"(dataset={ds_name} scene={scene_id} seg={segment_id}), got {tuple(segment_aabb.shape)}"
                    )
                if not torch.allclose(segment_aabb, self.segment_aabb, atol=1e-6, rtol=1e-6):
                    raise ValueError(
                        "segment_aabb mismatch between dataset config and segment manifest: "
                        f"config={self.segment_aabb.tolist()} asset={segment_aabb.tolist()} "
                        f"(dataset={ds_name} scene={scene_id} seg={segment_id})"
                    )
                parent_scene_asset_id = str(segment_manifest["parent_scene_asset_id"])
                scene_manifest = resolved["scene_handle"].load_manifest()
                if str(scene_manifest["asset_id"]) != parent_scene_asset_id:
                    raise ValueError(
                        "parent_scene_asset_id does not match linked scene manifest asset_id: "
                        f"segment declares {parent_scene_asset_id!r}, scene manifest has {scene_manifest['asset_id']!r} "
                        f"(dataset={ds_name} scene={scene_id} seg={segment_id})"
                    )
                seg_idx = resolved["segment_handle"].load_segment_index()
                if int(seg_idx["scene_id"]) != int(scene_id) or int(seg_idx["segment_id"]) != int(segment_id):
                    raise ValueError(
                        "segment_index scene/segment_id mismatch vs registry: "
                        f"expected=({scene_id},{segment_id}) got=({seg_idx['scene_id']},{seg_idx['segment_id']})"
                    )

    def initialize(self) -> None:
        self._validate_training_assets()
        self._initialized = True
        if self._preload_manager is not None:
            self._preload_manager.start()

    def shutdown_preload(self) -> None:
        if self._preload_manager is not None:
            self._preload_manager.stop()

    def __del__(self) -> None:
        try:
            self.shutdown_preload()
        except Exception:
            pass

    def set_preload_active_scope(self, scene_id: int, segment_id: int) -> None:
        self._preload_active_scene_id = int(scene_id)
        self._preload_active_segment_id = int(segment_id)

    def clear_preload_active_scope(self) -> None:
        self._preload_active_scene_id = None
        self._preload_active_segment_id = None

    def set_preload_training_scope(self, scene_id: int, segment_id: int) -> None:
        self._preload_training_scene_id = int(scene_id)
        self._preload_training_segment_id = int(segment_id)

    def clear_preload_training_scope(self) -> None:
        self._preload_training_scene_id = None
        self._preload_training_segment_id = None

    def clear_preload_scheduler_scope(self) -> None:
        self.clear_preload_active_scope()
        self.clear_preload_training_scope()

    def _resolve_segment_bundle(self, scene_id: int, segment_id: int) -> SegmentStaticBundle:
        ds_name = self._asset_dataset_name()
        key = (ds_name, int(scene_id), int(segment_id))
        with self._lock:
            cached = self._cache_get(self._segment_static_cache, key)
        if cached is not None:
            return cached

        inflight = self._wait_on_inflight(self._segment_bundle_inflight, self._segment_bundle_inflight_lock, key)
        if inflight is not None:
            inflight.wait()
            with self._lock:
                cached = self._cache_get(self._segment_static_cache, key)
            if cached is None:
                raise ValueError(f"segment bundle inflight missing after wait for {key}")
            return cached

        try:
            resolved = self.asset_store.resolve_segment_scene_assets_registry_first(
                ds_name, int(scene_id), int(segment_id)
            )
            segment_handle = resolved["segment_handle"]
            scene_handle = resolved["scene_handle"]
            segment_manifest = resolved["segment_manifest"]
            parent_scene_asset_id = str(segment_manifest["parent_scene_asset_id"])

            segment_payload = segment_handle.load_segment_index()
            segment_pose = segment_handle.load_segment_pose()
            pointcloud = segment_handle.load_pointcloud()
            dynamic_tracks = segment_handle.load_dynamic_tracks()
            pointcloud, dynamic_tracks = self._reconcile_dynamic_payloads(
                pointcloud=pointcloud,
                dynamic_tracks=dynamic_tracks,
            )
            sidx = self._build_segment_index_from_asset_payload(segment_payload)

            asset_aabb = segment_manifest.get("segment_aabb")
            if asset_aabb is None:
                raise ValueError("segment manifest missing segment_aabb")
            segment_aabb = torch.as_tensor(asset_aabb, dtype=torch.float32)
            if segment_aabb.shape != (2, 3):
                raise ValueError(
                    f"segment manifest segment_aabb must have shape [2,3], got {tuple(segment_aabb.shape)}"
                )
            if not torch.allclose(segment_aabb, self.segment_aabb, atol=1e-6, rtol=1e-6):
                raise ValueError(
                    "segment_aabb mismatch between dataset config and segment manifest: "
                    f"config={self.segment_aabb.tolist()} asset={segment_aabb.tolist()}"
                )

            pc_rt = self._cfg_get(self.dataset_cfg, "pointcloud")
            if pc_rt is not None and OmegaConf.is_config(pc_rt):
                pc_rt = OmegaConf.to_container(pc_rt, resolve=True)
            if isinstance(pc_rt, dict):
                pointcloud = _apply_runtime_random_pointcloud_downsample(
                    pointcloud=pointcloud,
                    segment_manifest=segment_manifest,
                    segment_aabb=segment_aabb,
                    runtime_pc=pc_rt,
                )

            bundle = SegmentStaticBundle(
                segment_asset_id=str(segment_manifest["asset_id"]),
                parent_scene_asset_id=parent_scene_asset_id,
                segment_index=sidx,
                segment_aabb=segment_aabb,
                segment_pose=segment_pose,
                pointcloud=pointcloud,
                dynamic_tracks=dynamic_tracks,
            )
            with self._lock:
                cached = self._cache_set(
                    self._segment_static_cache,
                    key,
                    bundle,
                    max_items=self._segment_static_cache_max_items,
                )
                self._cache_set(
                    self._scene_asset_cache,
                    (ds_name, int(scene_id)),
                    scene_handle,
                    max_items=self._scene_asset_cache_max_items,
                )
                return self._segment_static_cache[key] if cached else bundle
        finally:
            self._finish_inflight(self._segment_bundle_inflight, self._segment_bundle_inflight_lock, key)

    @staticmethod
    def _reconcile_dynamic_payloads(
        *,
        pointcloud: Dict[str, Any],
        dynamic_tracks: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dynamic = pointcloud.get("dynamic")
        if not isinstance(dynamic, dict) or len(dynamic) == 0:
            return pointcloud, dynamic_tracks

        instance_intids = np.asarray(dynamic_tracks["instance_intids"]).astype(np.int32, copy=False)
        quats = np.asarray(dynamic_tracks["instances_quats"]).astype(np.float32, copy=False)
        trans = np.asarray(dynamic_tracks["instances_trans"]).astype(np.float32, copy=False)
        fv = np.asarray(dynamic_tracks["instances_fv"]).astype(np.uint8, copy=False)
        if quats.shape[:2] != fv.shape or trans.shape[:2] != fv.shape:
            raise ValueError("dynamic_tracks arrays shape mismatch")
        if quats.shape[1] != len(instance_intids):
            raise ValueError("dynamic_tracks instance axis mismatch")

        visible_mask = (fv > 0).any(axis=0)
        visible_intids = {int(instance_intids[i]) for i in np.where(visible_mask)[0].tolist()}
        pointcloud_intids = {int(k) for k in dynamic.keys()}
        keep_intids = sorted(pointcloud_intids & visible_intids)

        reconciled_pointcloud = dict(pointcloud)
        reconciled_tracks = dict(dynamic_tracks)
        reconciled_pointcloud["dynamic"] = {int(i): dynamic[int(i)] for i in keep_intids}

        if len(keep_intids) == 0:
            frame_count = int(fv.shape[0])
            reconciled_tracks["instance_intids"] = np.zeros((0,), dtype=np.int32)
            reconciled_tracks["instances_quats"] = np.zeros((frame_count, 0, 4), dtype=np.float32)
            reconciled_tracks["instances_trans"] = np.zeros((frame_count, 0, 3), dtype=np.float32)
            reconciled_tracks["instances_fv"] = np.zeros((frame_count, 0), dtype=np.uint8)
            return reconciled_pointcloud, reconciled_tracks

        intid_to_col = {int(v): i for i, v in enumerate(instance_intids.tolist())}
        cols = [intid_to_col[i] for i in keep_intids if i in intid_to_col]
        if len(cols) != len(keep_intids):
            raise ValueError("dynamic pointcloud instance ids are not covered by dynamic_tracks")

        reconciled_tracks["instance_intids"] = np.asarray(keep_intids, dtype=np.int32)
        reconciled_tracks["instances_quats"] = quats[:, cols, :]
        reconciled_tracks["instances_trans"] = trans[:, cols, :]
        reconciled_tracks["instances_fv"] = fv[:, cols]
        return reconciled_pointcloud, reconciled_tracks

    def _build_segment_index_from_asset_payload(self, payload: Dict[str, Any]) -> SegmentIndexV4:
        train_frames = [int(x) for x in payload["frame_indices"]]
        test_frames = [int(x) for x in payload["test_frame_indices"]]
        train_refs = tuple((int(x[0]), int(x[1])) for x in np.asarray(payload["train_image_refs"]).tolist())
        test_refs = tuple((int(x[0]), int(x[1])) for x in np.asarray(payload["test_image_refs"]).tolist())
        return SegmentIndexV4(
            scene_id=int(payload["scene_id"]),
            segment_id=int(payload["segment_id"]),
            num_cams=int(payload["num_cams"]),
            frame_indices=train_frames,
            test_frame_indices=test_frames,
            train_frame_set=frozenset(train_frames),
            test_frame_set=frozenset(test_frames),
            keyframe_indices=[int(x) for x in payload["keyframe_indices"]],
            keyframe_to_frames={int(k): [int(x) for x in v] for k, v in payload["keyframe_to_frames"].items()},
            frame_to_keyframe={int(k): int(v) for k, v in payload["frame_to_keyframe"].items()},
            segment_first_frame_idx=int(payload["segment_first_frame_idx"]),
            train_image_refs=train_refs,
            test_image_refs=test_refs,
        )

    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexV4:
        return self._resolve_segment_bundle(scene_id, segment_id).segment_index

    def validate_image_ref(
        self,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
        purpose: Literal["train", "test"],
    ) -> None:
        sidx = self.get_segment_index(scene_id, segment_id)
        frame_idx = int(image_ref[0])
        cam_id = int(image_ref[1])
        if cam_id < 0 or cam_id >= int(sidx.num_cams):
            raise ValueError(
                f"cam_id={cam_id} out of range for scene={scene_id} segment={segment_id}, num_cams={sidx.num_cams}"
            )
        if purpose == "train" and frame_idx not in sidx.train_frame_set:
            raise ValueError(
                f"train image_ref frame_idx={frame_idx} not in train_frame_indices "
                f"(scene={scene_id} segment={segment_id})"
            )
        if purpose == "test" and frame_idx not in sidx.test_frame_set:
            raise ValueError(
                f"test image_ref frame_idx={frame_idx} not in test_frame_indices "
                f"(scene={scene_id} segment={segment_id})"
            )

    @staticmethod
    def _resize_2d_tensor_to_hw(x: Tensor, height: int, width: int, *, mode: str) -> Tensor:
        if int(x.shape[0]) == int(height) and int(x.shape[1]) == int(width):
            return x
        if x.dim() == 2:
            y = x.unsqueeze(0).unsqueeze(0).float()
            if mode == "bilinear":
                y = F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
            else:
                y = F.interpolate(y, size=(height, width), mode="nearest")
            return y[0, 0].to(dtype=x.dtype)
        if x.dim() == 3:
            y = x.permute(2, 0, 1).unsqueeze(0).float()
            if mode == "bilinear":
                y = F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
            else:
                y = F.interpolate(y, size=(height, width), mode="nearest")
            return y[0].permute(1, 2, 0).to(dtype=x.dtype)
        raise ValueError(f"_resize_2d_tensor_to_hw expects 2D/3D tensor, got shape={tuple(x.shape)}")

    def _load_image_meta(self, scene_id: int, segment_id: int, image_ref: ImageRef) -> Dict[str, Any]:
        ds_name = self._asset_dataset_name()
        k = (ds_name, int(scene_id), int(image_ref[0]), int(image_ref[1]))
        with self._lock:
            cached = self._cache_get(self._image_meta_cache, k)
        if cached is not None:
            return dict(cached)

        inflight = self._wait_on_inflight(self._image_meta_inflight, self._image_meta_inflight_lock, k)
        if inflight is not None:
            inflight.wait()
            with self._lock:
                cached = self._cache_get(self._image_meta_cache, k)
            if cached is None:
                raise ValueError(f"image meta inflight missing after wait for {k}")
            return dict(cached)

        try:
            bundle = self._resolve_segment_bundle(scene_id, segment_id)
            scene_handle = self.asset_store.get_scene_asset_by_asset_id(
                bundle.parent_scene_asset_id,
                dataset=ds_name,
                scene_id=int(scene_id),
            )
            rows = scene_handle.load_image_meta([(int(image_ref[0]), int(image_ref[1]))])
            if len(rows) != 1:
                raise ValueError(f"Expected one image meta row for scene={scene_id} ref={image_ref}")
            row = dict(rows[0])
            row["intrinsic_4x4"] = np.asarray(row["intrinsic_4x4_flat"], dtype=np.float32).reshape(4, 4)
            row["camera_to_world"] = np.asarray(row["camera_to_world_flat"], dtype=np.float32).reshape(4, 4)
            with self._lock:
                cached = self._cache_set(
                    self._image_meta_cache,
                    k,
                    dict(row),
                    max_items=self._image_meta_cache_max_items,
                )
                return dict(self._image_meta_cache[k]) if cached else dict(row)
        finally:
            self._finish_inflight(self._image_meta_inflight, self._image_meta_inflight_lock, k)

    def _load_depth_from_asset_path(self, depth_path: str, height: int, width: int) -> Tensor:
        path = Path(depth_path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(str(path), allow_pickle=False)
        elif suffix == ".npz":
            z = np.load(str(path), allow_pickle=False)
            keys = list(z.keys())
            if len(keys) != 1:
                raise ValueError(f"depth npz must have exactly one array, got keys={keys}")
            arr = z[keys[0]]
        else:
            arr = np.asarray(Image.open(str(path)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        t = torch.as_tensor(arr, dtype=torch.float32)
        if t.shape[0] != int(height) or t.shape[1] != int(width):
            t = self._resize_2d_tensor_to_hw(t, int(height), int(width), mode="bilinear")
        return t.to(device=self.device)

    def _load_mask_from_asset_path(self, path_str: str, height: int, width: int) -> Tensor:
        arr = np.asarray(Image.open(path_str))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.as_tensor(arr, dtype=torch.float32)
        if mask.shape[0] != int(height) or mask.shape[1] != int(width):
            mask = self._resize_2d_tensor_to_hw(mask, int(height), int(width), mode="nearest")
        mask = mask.to(device=self.device)
        if mask.max().item() > 1.0:
            mask = (mask > 0.0).float()
        return mask

    @staticmethod
    def _normalize_sky_mask(mask: Tensor, semantics: str) -> Tensor:
        return normalize_sky_mask_to_one_is_sky((mask > 0.5).float(), semantics)

    def _compute_viewdirs(self, height: int, width: int, intrinsic: Tensor, camera_to_world: Tensor) -> Tensor:
        intr = intrinsic[:3, :3]
        c2w = camera_to_world[:3, :3]
        xs = torch.arange(int(width), device=self.device, dtype=torch.float32)
        ys = torch.arange(int(height), device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x = grid_x.reshape(-1)
        y = grid_y.reshape(-1)
        camera_dirs = torch.stack(
            [
                (x - intr[0, 2] + 0.5) / intr[0, 0],
                (y - intr[1, 2] + 0.5) / intr[1, 1],
                torch.ones_like(x),
            ],
            dim=-1,
        )
        directions = camera_dirs @ c2w.T
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdim=True)
        viewdirs = directions / (direction_norm + 1e-8)
        return viewdirs.reshape(int(height), int(width), 3)

    def _load_view_from_asset_paths(
        self,
        scene_id: int,
        image_ref: ImageRef,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_path = str(meta.get("image_path", ""))
        if not image_path:
            raise ValueError(f"Missing image_path for scene={scene_id} image_ref={image_ref}")
        depth_path = str(meta.get("depth_path", ""))
        sky_mask_path = str(meta.get("sky_mask_path", ""))
        dynamic_mask_path = str(meta.get("dynamic_mask_path", ""))

        height = int(meta["height"])
        width = int(meta["width"])
        pil_img = Image.open(image_path).convert("RGB")
        if pil_img.size != (width, height):
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR  # type: ignore[attr-defined]
            pil_img = pil_img.resize((width, height), resample=resample)
        image_arr = np.asarray(pil_img, dtype=np.float32)
        image = torch.as_tensor(image_arr / 255.0, dtype=torch.float32, device=self.device)

        if depth_path:
            depth = self._load_depth_from_asset_path(depth_path, height, width)
        else:
            depth = torch.ones((height, width), dtype=torch.float32, device=self.device) * 10.0

        sky_mask = None
        if self._load_sky_mask and sky_mask_path:
            if self._sky_mask_loader_semantics is None:
                raise ValueError(
                    "Sky mask tensor is present but data.sky_mask_semantics is not configured. "
                    "Set pixel_source.load_sky_mask: true and data.sky_mask_semantics."
                )
            raw_sky = self._load_mask_from_asset_path(sky_mask_path, height, width)
            sky_mask = self._normalize_sky_mask(raw_sky, self._sky_mask_loader_semantics)

        dynamic_mask = None
        if self._load_dynamic_mask and dynamic_mask_path:
            dynamic_mask = self._load_mask_from_asset_path(dynamic_mask_path, height, width)

        viewdirs = self._compute_viewdirs(
            height,
            width,
            torch.as_tensor(meta["intrinsic_4x4"], dtype=torch.float32, device=self.device),
            torch.as_tensor(meta["camera_to_world"], dtype=torch.float32, device=self.device),
        )

        return {
            "image": image,
            "extrinsic": torch.as_tensor(meta["camera_to_world"], dtype=torch.float32, device=self.device),
            "intrinsic": torch.as_tensor(meta["intrinsic_4x4"], dtype=torch.float32, device=self.device),
            "depth": depth,
            "sky_mask": sky_mask,
            "viewdirs": viewdirs,
            "dynamic_mask": dynamic_mask,
            "egocar_mask": None,
            "frame_idx": int(image_ref[0]),
            "cam_idx": int(image_ref[1]),
        }

    def _preload_view_key_is_cached(self, key: Tuple[str, int, int, int, int]) -> bool:
        if not self._enable_view_pack_cache:
            return False
        with self._lock:
            return key in self._view_pack_cache

    def _get_cached_or_load_view(self, scene_id: int, segment_id: int, image_ref: ImageRef) -> Dict[str, Any]:
        ds_name = self._asset_dataset_name()
        key = (ds_name, int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
        created_inflight = False
        if self._enable_view_pack_cache:
            with self._lock:
                cached = self._cache_get(self._view_pack_cache, key)
                if cached is not None:
                    return loaded_view_pack_to_device_v2(cached, self.device)
            inflight = self._wait_on_inflight(self._view_pack_inflight, self._view_pack_inflight_lock, key)
            if inflight is not None:
                inflight.wait()
                with self._lock:
                    cached = self._cache_get(self._view_pack_cache, key)
                    if cached is not None:
                        return loaded_view_pack_to_device_v2(cached, self.device)
                raise ValueError(f"view pack inflight missing after wait for {key}")
            created_inflight = True
        try:
            meta = self._load_image_meta(scene_id, segment_id, image_ref)
            pack = self._load_view_from_asset_paths(scene_id, image_ref, meta)
            lvp = dict_to_loaded_view_pack_v2(pack)
            if not self._enable_view_pack_cache:
                return loaded_view_pack_to_device_v2(lvp, self.device)
            with self._lock:
                cached = self._cache_set(
                    self._view_pack_cache,
                    key,
                    lvp,
                    max_items=self._view_pack_cache_max_items,
                )
                if cached:
                    return loaded_view_pack_to_device_v2(self._view_pack_cache[key], self.device)
                return loaded_view_pack_to_device_v2(lvp, self.device)
        finally:
            if created_inflight:
                self._finish_inflight(self._view_pack_inflight, self._view_pack_inflight_lock, key)

    def _build_dynamic_info_from_asset_tracks(
        self,
        tracks: Dict[str, Any],
        *,
        frame_indices: Sequence[int],
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        frames = np.asarray(tracks["frame_indices"]).astype(np.int32, copy=False)
        intids = np.asarray(tracks["instance_intids"]).astype(np.int32, copy=False)
        quats = np.asarray(tracks["instances_quats"]).astype(np.float32, copy=False)
        trans = np.asarray(tracks["instances_trans"]).astype(np.float32, copy=False)
        fv = np.asarray(tracks["instances_fv"]).astype(np.uint8, copy=False)
        if quats.shape[:2] != fv.shape or trans.shape[:2] != fv.shape:
            raise ValueError("dynamic_tracks arrays shape mismatch")
        if quats.shape[0] != len(frames) or quats.shape[1] != len(intids):
            raise ValueError("dynamic_tracks frame/instance axes mismatch")
        frame_to_row = {int(f): i for i, f in enumerate(frames.tolist())}
        out: Dict[int, Dict[str, Any]] = {}
        for fidx in sorted(set(int(x) for x in frame_indices)):
            row = frame_to_row.get(int(fidx))
            if row is None:
                continue
            inst: Dict[int, Dict[str, Any]] = {}
            for col, intid in enumerate(intids.tolist()):
                if int(fv[row, col]) == 0:
                    continue
                inst[int(intid)] = {
                    "quat": quats[row, col].tolist(),
                    "trans": trans[row, col].tolist(),
                }
            out[int(fidx)] = {"instances": inst}
        if not out:
            return None
        if not any(len(v.get("instances", {})) > 0 for v in out.values()):
            return None
        return out

    @staticmethod
    def _stack_optional_masks(
        items: List[Optional[Tensor]],
        fallback_images: List[Tensor],
        *,
        as_viewdirs: bool,
        device: torch.device,
    ) -> Optional[Tensor]:
        if not any(x is not None for x in items):
            return None
        out: List[Tensor] = []
        for val, image in zip(items, fallback_images):
            if val is not None:
                out.append(val.to(device=device, dtype=torch.float32))
                continue
            h, w = int(image.shape[0]), int(image.shape[1])
            if as_viewdirs:
                out.append(torch.zeros((h, w, 3), dtype=torch.float32, device=device))
            else:
                out.append(torch.zeros((h, w), dtype=torch.float32, device=device))
        return torch.stack(out, dim=0)

    def _resolve_test_image_refs(self, sidx: SegmentIndexV4) -> List[ImageRef]:
        pixel_source_cfg = self._cfg_get(self.data_cfg, "pixel_source", {}) or {}
        max_test = int(self._cfg_get(pixel_source_cfg, "max_test_images", 0))
        frames = sorted(int(x) for x in sidx.test_frame_set)
        if max_test > 0 and len(frames) > max_test:
            frames = frames[:max_test]
        refs: List[ImageRef] = []
        for frame_idx in frames:
            for cam_id in range(int(sidx.num_cams)):
                refs.append((int(frame_idx), int(cam_id)))
        return refs

    def _assemble_segment_batch_from_image_refs(
        self,
        scene_id: int,
        segment_id: int,
        source_image_refs: Sequence[ImageRef],
        target_image_refs: Sequence[ImageRef],
        *,
        include_test: bool,
        test_image_refs: Optional[Sequence[ImageRef]],
        enforce_target0_equals_source: bool,
        target_ref_purpose: Literal["train", "test"] = "train",
    ) -> Dict[str, Any]:
        if len(source_image_refs) == 0:
            raise ValueError("source_image_refs must not be empty")
        if len(target_image_refs) == 0:
            raise ValueError("target_image_refs must not be empty")
        sidx = self.get_segment_index(scene_id, segment_id)
        for ref in source_image_refs:
            self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="train")
        for ref in target_image_refs:
            self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose=target_ref_purpose)
        if enforce_target0_equals_source and tuple(target_image_refs[0]) != tuple(source_image_refs[0]):
            raise ValueError("target_image_refs[0] must equal source_image_ref when enforce_target0_equals_source=True")

        bundle = self._resolve_segment_bundle(scene_id, segment_id)
        world_to_seg0 = bundle.segment_pose["world_to_seg0"].to(device=self.device, dtype=torch.float32)
        segment_first_pose = bundle.segment_pose["segment_first_pose_world"].to(device=self.device, dtype=torch.float32)
        segment_first_frame_idx = int(bundle.segment_pose["segment_first_frame_idx"])
        segment_pose_source = str(bundle.segment_pose["segment_pose_source"])

        def _load_role(
            refs: Sequence[ImageRef],
            *,
            allow_missing_keyframe: bool,
        ) -> Dict[str, Any]:
            images: List[Tensor] = []
            extrinsics: List[Tensor] = []
            intrinsics: List[Tensor] = []
            depths: List[Tensor] = []
            frame_indices: List[int] = []
            cam_indices: List[int] = []
            keyframe_indices: List[int] = []
            sky_masks: List[Optional[Tensor]] = []
            viewdirs: List[Optional[Tensor]] = []
            dynamic_masks: List[Optional[Tensor]] = []
            egocar_masks: List[Optional[Tensor]] = []
            for ref in refs:
                pack = self._get_cached_or_load_view(scene_id, segment_id, tuple(ref))
                fidx = int(pack["frame_idx"])
                images.append(pack["image"])
                ext = pack["extrinsic"].to(device=self.device, dtype=torch.float32)
                extrinsics.append(world_to_seg0 @ ext)
                intrinsics.append(pack["intrinsic"].to(device=self.device, dtype=torch.float32))
                depths.append(pack["depth"].to(device=self.device, dtype=torch.float32))
                frame_indices.append(fidx)
                cam_indices.append(int(pack["cam_idx"]))
                kf = sidx.frame_to_keyframe.get(fidx)
                if kf is None and not allow_missing_keyframe:
                    raise ValueError(
                        f"frame_idx={fidx} has no keyframe mapping (scene={scene_id} segment={segment_id})"
                    )
                keyframe_indices.append(-1 if kf is None else int(kf))
                sky_masks.append(pack.get("sky_mask"))
                viewdirs.append(pack.get("viewdirs"))
                dynamic_masks.append(pack.get("dynamic_mask"))
                egocar_masks.append(pack.get("egocar_mask"))
            out: Dict[str, Any] = {
                "image": torch.stack(images, dim=0),
                "extrinsics": torch.stack(extrinsics, dim=0),
                "intrinsics": torch.stack(intrinsics, dim=0),
                "depth": torch.stack(depths, dim=0),
                "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
                "cam_indices": torch.tensor(cam_indices, dtype=torch.long),
                "keyframe_indices": torch.tensor(keyframe_indices, dtype=torch.long),
            }
            sky = self._stack_optional_masks(
                sky_masks, images, as_viewdirs=False, device=self.device
            )
            if sky is not None:
                out["sky_mask"] = sky
            vd = self._stack_optional_masks(
                viewdirs, images, as_viewdirs=True, device=self.device
            )
            if vd is not None:
                out["viewdirs"] = vd
            dyn = self._stack_optional_masks(
                dynamic_masks, images, as_viewdirs=False, device=self.device
            )
            if dyn is not None:
                out["dynamic_mask"] = dyn
            ego = self._stack_optional_masks(
                egocar_masks, images, as_viewdirs=False, device=self.device
            )
            if ego is not None:
                out["egocar_mask"] = ego
            return out

        source = _load_role(source_image_refs, allow_missing_keyframe=False)
        target = _load_role(
            target_image_refs,
            allow_missing_keyframe=(target_ref_purpose == "test"),
        )
        all_frames = set(source["frame_indices"].tolist()) | set(target["frame_indices"].tolist())

        pointcloud = bundle.pointcloud
        dynamic_info = None
        dynamic_points = pointcloud.get("dynamic")
        if isinstance(dynamic_points, dict) and len(dynamic_points) > 0:
            if not bundle.dynamic_tracks:
                raise ValueError(
                    "dynamic pointcloud is non-empty but dynamic_tracks is missing in strict asset mode"
                )
            dynamic_info = self._build_dynamic_info_from_asset_tracks(
                bundle.dynamic_tracks, frame_indices=sorted(int(x) for x in all_frames)
            )
            # Keep pointcloud.dynamic aligned with the current batch frame window.
            # Some segments contain dynamic instances that are only visible in other frames.
            visible_intids_in_batch: set[int] = set()
            if dynamic_info is not None:
                for frame_obj in dynamic_info.values():
                    instances = frame_obj.get("instances", {})
                    for intid in instances.keys():
                        visible_intids_in_batch.add(int(intid))
            pointcloud = dict(pointcloud)
            if len(visible_intids_in_batch) == 0:
                pointcloud["dynamic"] = {}
            else:
                pointcloud["dynamic"] = {
                    int(intid): dynamic_points[int(intid)]
                    for intid in sorted(visible_intids_in_batch)
                    if int(intid) in dynamic_points
                }

        batch: Dict[str, Any] = {
            "scene_id": torch.tensor([int(scene_id)], dtype=torch.long),
            "scene_folder_name": f"{int(scene_id):03d}",
            "segment_id": int(segment_id),
            "aabb": bundle.segment_aabb.to(device=self.device),
            "segment_first_pose": segment_first_pose,
            "segment_first_frame_idx": segment_first_frame_idx,
            "segment_first_pose_source": segment_pose_source,
            "request_meta": {
                "source_image_refs": [tuple(r) for r in source_image_refs],
                "target_image_refs": [tuple(r) for r in target_image_refs],
                "test_image_refs": None,
                "assembly_mode": "image_ref_v4",
            },
            "source": source,
            "target": target,
            "pointcloud": pointcloud,
        }
        if dynamic_info is not None:
            batch["dynamic_info"] = dynamic_info

        if include_test:
            resolved = [tuple(r) for r in (test_image_refs or self._resolve_test_image_refs(sidx))]
            for ref in resolved:
                self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="test")
            if len(resolved) > 0:
                test = _load_role(resolved, allow_missing_keyframe=True)
                test.pop("keyframe_indices", None)
                batch["test"] = test
                batch["request_meta"]["test_image_refs"] = resolved
        return batch

    def get_segment_batch_from_image_refs(
        self,
        request: BatchRequestV4,
        *,
        enforce_target0_equals_source: bool = True,
    ) -> Dict[str, Any]:
        include_test = bool(request.include_test)
        source_refs = (
            [tuple(x) for x in request.source_image_refs]
            if request.source_image_refs is not None
            else [tuple(request.source_image_ref)]
        )
        return self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            source_refs,
            request.target_image_refs,
            include_test=include_test,
            test_image_refs=request.test_image_refs if include_test else None,
            enforce_target0_equals_source=enforce_target0_equals_source,
            target_ref_purpose="train",
        )

    def get_segment_eval_batch_from_image_refs(self, request: EvalRequestV4) -> Dict[str, Any]:
        raw = self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            [request.source_image_ref],
            request.eval_image_refs,
            include_test=False,
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="test",
        )
        out: Dict[str, Any] = dict(raw)
        out["eval"] = out.pop("target")
        out["request_meta"]["eval_image_refs"] = [tuple(r) for r in request.eval_image_refs]
        out["request_meta"]["assembly_mode"] = "eval_image_ref_v4"
        return out

    def build_preload_hint(
        self,
        *,
        scene_id: int,
        segment_id: int,
        future_image_refs: Sequence[ImageRef],
        scope: str = "next_block_exact",
    ) -> Dict[str, Any]:
        bundle = self._resolve_segment_bundle(scene_id, segment_id)
        refs = [(int(x[0]), int(x[1])) for x in future_image_refs]
        unique_frames = sorted({int(x[0]) for x in refs})
        unique_cams = sorted({int(x[1]) for x in refs})
        return {
            "hint_version": 3,
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "segment_asset_id": str(bundle.segment_asset_id),
            "future_image_refs": refs,
            "unique_frame_indices": unique_frames,
            "unique_cam_indices": unique_cams,
            "required_static": {
                "segment_bundle": True,
                "test_refs": False,
            },
            "scope": str(scope),
        }

    def submit_preload_hint(
        self,
        *,
        hint: Dict[str, Any],
        hint_scope: str,
        epoch_idx: int,
        global_step: int,
        block_idx_global: int,
        include_test: bool,
    ) -> None:
        _ = (epoch_idx, global_step, block_idx_global)
        mgr = self._preload_manager
        if mgr is None:
            return
        mgr.submit_preload_hint(
            hint=hint,
            hint_scope=str(hint_scope),
            include_test=bool(include_test),
        )

    def get_or_compute_pair_score(
        self,
        scene_id: int,
        segment_id: int,
        src: ImageRef,
        tgt: ImageRef,
        *,
        mode: str = "none",
    ) -> Optional[float]:
        _ = (scene_id, segment_id, src, tgt)
        if mode == "none":
            return None
        raise ValueError(f"MultiSceneDatasetV4 unsupported pair score mode={mode!r}")

    def create_train_scheduler_v6(
        self,
        *,
        state_write_interval_steps: int,
        updates_per_block: int,
        keyframes_per_episode: int,
        episodes_per_segment: int,
        total_target_frames: int,
        include_source_frame: bool,
        neighbor_ring: int,
        prefer_nearby_keyframes: bool,
        fallback_expand_to_segment: bool,
        with_replacement: bool,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
    ) -> TrainSchedulerV6:
        return TrainSchedulerV6(
            dataset=self,
            state_write_interval_steps=state_write_interval_steps,
            updates_per_block=updates_per_block,
            keyframes_per_episode=keyframes_per_episode,
            episodes_per_segment=episodes_per_segment,
            total_target_frames=total_target_frames,
            include_source_frame=include_source_frame,
            neighbor_ring=neighbor_ring,
            prefer_nearby_keyframes=prefer_nearby_keyframes,
            fallback_expand_to_segment=fallback_expand_to_segment,
            with_replacement=with_replacement,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
        )

    def create_train_scheduler_v7(
        self,
        *,
        steps_per_block: int,
        blocks_per_episode: int,
        total_target_frames: int,
        include_source_frame: bool,
        frame_within_keyframe_policy: str,
        min_keyframes_required_policy: str,
        traversal_mode: str,
        switch_after_episode: bool,
        segment_order: str,
        scene_order: str,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
        emit_preload_hints: bool,
        warm_next_block_exact: bool,
        warm_next_episode_chain: bool,
    ) -> TrainSchedulerV7:
        return TrainSchedulerV7(
            dataset=self,
            steps_per_block=steps_per_block,
            blocks_per_episode=blocks_per_episode,
            total_target_frames=total_target_frames,
            include_source_frame=include_source_frame,
            frame_within_keyframe_policy=frame_within_keyframe_policy,
            min_keyframes_required_policy=min_keyframes_required_policy,
            traversal_mode=traversal_mode,
            switch_after_episode=switch_after_episode,
            segment_order=segment_order,
            scene_order=scene_order,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            emit_preload_hints=emit_preload_hints,
            warm_next_block_exact=warm_next_block_exact,
            warm_next_episode_chain=warm_next_episode_chain,
        )

    # Preload worker hooks
    def _preload_worker_scene_meta(self, scene_id: int, segment_id: int, meta: Dict[str, Any]) -> None:
        _ = (meta,)
        self._resolve_segment_bundle(scene_id, segment_id)

    def _preload_worker_segment_static(self, scene_id: int, segment_id: int, meta: Dict[str, Any]) -> None:
        _ = (meta,)
        self._resolve_segment_bundle(scene_id, segment_id)

    def _preload_worker_view_meta(
        self, scene_id: int, segment_id: int, image_ref: ImageRef, meta: Dict[str, Any]
    ) -> None:
        _ = (meta,)
        self._load_image_meta(scene_id, segment_id, image_ref)

    def _preload_worker_view_pack(
        self, scene_id: int, segment_id: int, image_ref: ImageRef, meta: Dict[str, Any]
    ) -> None:
        _ = (meta,)
        self._get_cached_or_load_view(scene_id, segment_id, image_ref)
