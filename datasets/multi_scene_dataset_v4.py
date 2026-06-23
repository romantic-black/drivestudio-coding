from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
import dataclasses
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
from datasets.pointcloud_generators.dynamic_balance import (
    compute_volume_balanced_point_caps,
    dynamic_point_balance_enabled,
    normalize_dynamic_point_balance_cfg,
    volume_map_from_metadata,
)
from datasets.streetforward_assets import StreetForwardAssetStore
from datasets.train_scheduler_v6 import TrainSchedulerV6
from datasets.train_scheduler_v7 import TrainSchedulerV7
from datasets.train_scheduler_v8 import TrainSchedulerV8
from datasets.train_scheduler_v9 import TrainSchedulerV9, ViewSetRolloutBatchV9
from streetforward_core.data.schedulers.legacy_v9_phase_a_adapter import convert_v9_phase_a_plan

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

_EGO_MASK_MISSING = object()

_EXPECTED_ASSET_COORDINATE_FRAME = "seg0_camera_opencv"
_IFORWARD_STAGE2_1_SCHEDULER_VERSION = "iforward_stage2_1_parent_temporal"
_IFORWARD_SEQUENCE10_SCHEDULER_VERSION = "iforward_sequence10_v1"


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
    knn_init: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class KNNValidationRequirementsV4:
    enabled: bool
    background_ks: Tuple[int, ...]
    dynamic_ks: Tuple[int, ...]
    required_branches: Tuple[str, ...]
    fixed_neighbor_enabled: bool
    neighbor_k_store: int


def _cap_int_or_none(d: Dict[str, Any], k: str) -> Optional[int]:
    v = d.get(k)
    if v is None:
        return None
    i = int(v)
    if i <= 0:
        raise ValueError(f"dataset.pointcloud.{k} must be > 0 when set, got {v!r}")
    return i


def _extract_pointcloud_caps(d: Dict[str, Any]) -> Dict[str, Optional[int]]:
    return {k: _cap_int_or_none(d, k) for k in _POINTCLOUD_CAP_KEYS}


def _pointcloud_cap_keys_differ(
    asset_pc: Dict[str, Any],
    runtime_pc: Dict[str, Any],
    keys: Sequence[str],
) -> bool:
    for key in keys:
        if _cap_int_or_none(asset_pc, key) != _cap_int_or_none(runtime_pc, key):
            return True
    return False


def _pointcloud_dynamic_balance_differ(asset_pc: Dict[str, Any], runtime_pc: Dict[str, Any]) -> bool:
    asset_balance = normalize_dynamic_point_balance_cfg(asset_pc.get("dynamic_point_balance"))
    runtime_balance = normalize_dynamic_point_balance_cfg(runtime_pc.get("dynamic_point_balance"))
    return asset_balance != runtime_balance


def _parse_knn_validation_requirements(raw: Any) -> KNNValidationRequirementsV4:
    if raw is None:
        return KNNValidationRequirementsV4(
            enabled=False,
            background_ks=tuple(),
            dynamic_ks=tuple(),
            required_branches=tuple(),
            fixed_neighbor_enabled=False,
            neighbor_k_store=0,
        )
    payload = raw
    if OmegaConf.is_config(payload):
        payload = OmegaConf.to_container(payload, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"knn_requirements must be a dict when provided, got {type(raw)}")

    bg_ks = sorted({int(x) for x in list(payload.get("background_ks", []) or [])})
    dyn_ks = sorted({int(x) for x in list(payload.get("dynamic_ks", []) or [])})
    for k in bg_ks + dyn_ks:
        if int(k) <= 0:
            raise ValueError(f"knn_requirements k must be > 0, got {k}")

    branches = sorted({str(x).strip() for x in list(payload.get("required_branches", []) or []) if str(x).strip()})
    invalid_branches = [b for b in branches if b not in {"bg", "distant", "rigid"}]
    if invalid_branches:
        raise ValueError(
            "knn_requirements.required_branches contains unsupported entries: "
            f"{invalid_branches}"
        )

    fixed_neighbor_enabled = bool(payload.get("fixed_neighbor_enabled", False))
    neighbor_k_store = int(payload.get("neighbor_k_store", 0))
    if neighbor_k_store < 0:
        raise ValueError(f"knn_requirements.neighbor_k_store must be >= 0, got {neighbor_k_store}")
    if neighbor_k_store > 0:
        fixed_neighbor_enabled = True
    if fixed_neighbor_enabled and neighbor_k_store <= 0:
        raise ValueError("knn_requirements.fixed_neighbor_enabled=true requires neighbor_k_store > 0")

    enabled = bool(payload.get("enabled", False)) or len(bg_ks) > 0 or len(dyn_ks) > 0 or fixed_neighbor_enabled
    if enabled and len(bg_ks) == 0 and len(dyn_ks) == 0 and not fixed_neighbor_enabled:
        raise ValueError(
            "knn_requirements.enabled=true requires at least one of background_ks/dynamic_ks"
        )

    return KNNValidationRequirementsV4(
        enabled=bool(enabled),
        background_ks=tuple(int(x) for x in bg_ks),
        dynamic_ks=tuple(int(x) for x in dyn_ks),
        required_branches=tuple(branches),
        fixed_neighbor_enabled=bool(fixed_neighbor_enabled),
        neighbor_k_store=int(neighbor_k_store),
    )


class MultiSceneDatasetV4:
    def __init__(
        self,
        *,
        dataset_cfg: Any,
        data_cfg: Any,
        device: torch.device,
        asset_store: Optional[StreetForwardAssetStore] = None,
        preload_cfg: Optional[Dict[str, Any]] = None,
        knn_requirements: Optional[Dict[str, Any]] = None,
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
        runtime_pc = self._cfg_get(self.dataset_cfg, "pointcloud")
        if runtime_pc is not None and OmegaConf.is_config(runtime_pc):
            runtime_pc = OmegaConf.to_container(runtime_pc, resolve=True)
        if runtime_pc is not None and not isinstance(runtime_pc, dict):
            raise ValueError(f"dataset.pointcloud must be a mapping when provided, got {type(runtime_pc)}")
        self._runtime_pointcloud_cfg: Optional[Dict[str, Any]] = (
            dict(runtime_pc) if isinstance(runtime_pc, dict) else None
        )
        self._max_dynamic_points_per_segment = self._parse_max_dynamic_points_per_segment()
        self._segment_dynamic_point_count_cache: Dict[Tuple[str, int, int], int] = {}
        self._dynamic_point_filter_logged_scenes: set[Tuple[str, int]] = set()
        self._knn_requirements = _parse_knn_validation_requirements(knn_requirements)

        pixel_source_cfg = self._cfg_get(self.data_cfg, "pixel_source", {}) or {}
        self._load_sky_mask = bool(self._cfg_get(pixel_source_cfg, "load_sky_mask", False))
        self._load_dynamic_mask = bool(self._cfg_get(pixel_source_cfg, "load_dynamic_mask", False))
        self._load_egocar_mask = bool(self._cfg_get(pixel_source_cfg, "load_egocar_mask", True))
        self._sky_mask_loader_semantics = self._parse_sky_mask_semantics()
        self._pixel_source_cameras: List[int] = [int(x) for x in list(self._cfg_get(pixel_source_cfg, "cameras", []) or [])]
        self._egocar_mask_cache: "OrderedDict[Tuple[str, int, int, int], Any]" = OrderedDict()
        self._egocar_mask_cache_max_items = 64
        self._egocar_missing_warned: set[Tuple[str, int]] = set()

        (
            self._scene_asset_cache_max_items,
            self._segment_static_cache_max_items,
            self._image_meta_cache_max_items,
            self._view_pack_cache_max_items,
        ) = self._parse_required_cache_max_items()

        self._scene_asset_cache: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()
        self._segment_static_cache: "OrderedDict[Tuple[str, int, int], SegmentStaticBundle]" = OrderedDict()
        self._segment_index_cache: Dict[Tuple[str, int, int], SegmentIndexV4] = {}
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

    def _knn_required_branches_label(self) -> str:
        if len(self._knn_requirements.required_branches) == 0:
            return "fixed_cached" if self._knn_requirements.fixed_neighbor_enabled else "unknown"
        return ",".join(self._knn_requirements.required_branches)

    def _knn_strict_pointcloud_cap_keys(self) -> Tuple[str, ...]:
        branches = set(str(x) for x in self._knn_requirements.required_branches)
        if len(branches) == 0:
            return tuple(_POINTCLOUD_CAP_KEYS)

        keys: List[str] = []
        if "bg" in branches:
            keys.append("near_max_points")
        if "distant" in branches:
            keys.append("distant_max_points")
        if "rigid" in branches:
            keys.append("monocular_dynamic_recovery_max_points_per_instance")
        return tuple(k for k in _POINTCLOUD_CAP_KEYS if k in set(keys))

    def _knn_strict_dynamic_balance(self) -> bool:
        branches = set(str(x) for x in self._knn_requirements.required_branches)
        return len(branches) == 0 or "rigid" in branches

    @staticmethod
    def _stride_keep_indices(num_points: int, max_count: Optional[int]) -> np.ndarray:
        n = int(num_points)
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        if max_count is None or int(max_count) <= 0 or n <= int(max_count):
            return np.arange(n, dtype=np.int64)
        step = max(1, n // int(max_count))
        idx = np.arange(0, n, step, dtype=np.int64)
        if int(idx.shape[0]) > int(max_count):
            idx = idx[: int(max_count)]
        return idx

    def _runtime_cap_or_none(self, key: str) -> Optional[int]:
        runtime_pc = self._runtime_pointcloud_cfg
        if not isinstance(runtime_pc, dict):
            return None
        return _cap_int_or_none(runtime_pc, key)

    def _apply_runtime_pointcloud_caps(
        self,
        *,
        pointcloud: Dict[str, Any],
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> Dict[str, Any]:
        near_cap = self._runtime_cap_or_none("near_max_points")
        distant_cap = self._runtime_cap_or_none("distant_max_points")
        dynamic_cap = self._runtime_cap_or_none("monocular_dynamic_recovery_max_points_per_instance")
        runtime_pc = self._runtime_pointcloud_cfg if isinstance(self._runtime_pointcloud_cfg, dict) else {}
        dynamic_balance_cfg = normalize_dynamic_point_balance_cfg(
            runtime_pc.get("dynamic_point_balance")
        )
        dynamic_balance_on = dynamic_point_balance_enabled(dynamic_balance_cfg)
        if dynamic_balance_on and dynamic_cap is None:
            raise ValueError(
                "dataset.pointcloud.dynamic_point_balance.enable=true requires "
                "dataset.pointcloud.monocular_dynamic_recovery_max_points_per_instance for runtime caps."
            )
        if near_cap is None and distant_cap is None and dynamic_cap is None and not dynamic_balance_on:
            return pointcloud

        background = np.asarray(
            pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)),
            dtype=np.float32,
        )
        if background.ndim != 2 or background.shape[1] < 3:
            raise ValueError(
                "pointcloud.background must have shape [N,>=3] for runtime cap downsample, "
                f"got {tuple(background.shape)} (context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )

        xyz = np.asarray(background[:, :3], dtype=np.float32)
        crop_min = self.segment_aabb[0].detach().cpu().numpy()
        crop_max = self.segment_aabb[1].detach().cpu().numpy()
        in_crop = ((xyz >= crop_min[None, :]) & (xyz <= crop_max[None, :])).all(axis=1)
        near_idx = np.nonzero(in_crop)[0].astype(np.int64, copy=False)
        distant_idx = np.nonzero(~in_crop)[0].astype(np.int64, copy=False)
        near_keep = self._stride_keep_indices(int(near_idx.shape[0]), near_cap)
        distant_keep = self._stride_keep_indices(int(distant_idx.shape[0]), distant_cap)
        keep_bg_idx = np.concatenate([near_idx[near_keep], distant_idx[distant_keep]], axis=0)
        background_after = np.ascontiguousarray(background[keep_bg_idx], dtype=np.float32)

        dynamic_raw = pointcloud.get("dynamic", {})
        dynamic_after: Any = dynamic_raw
        dynamic_before_total = 0
        dynamic_after_total = 0
        dynamic_point_caps: Dict[int, int] = {}
        if dynamic_balance_on:
            dynamic_point_caps = compute_volume_balanced_point_caps(
                dynamic_cap,
                volume_map_from_metadata(pointcloud.get("metadata")),
                dynamic_balance_cfg,
            )
        if isinstance(dynamic_raw, dict):
            dyn_out: Dict[int, np.ndarray] = {}
            for intid_raw in sorted(dynamic_raw.keys(), key=lambda x: int(x)):
                intid = int(intid_raw)
                pts = np.asarray(dynamic_raw[intid_raw], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    raise ValueError(
                        f"pointcloud.dynamic[{intid}] must have shape [N,>=3] for runtime cap downsample, "
                        f"got {tuple(pts.shape)} (context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
                    )
                cap = dynamic_point_caps.get(intid, dynamic_cap) if dynamic_balance_on else dynamic_cap
                keep_dyn = self._stride_keep_indices(int(pts.shape[0]), cap)
                pts_after = np.ascontiguousarray(pts[keep_dyn], dtype=np.float32)
                dyn_out[intid] = pts_after
                dynamic_before_total += int(pts.shape[0])
                dynamic_after_total += int(pts_after.shape[0])
            dynamic_after = dyn_out
        elif dynamic_raw is None:
            dynamic_before_total = 0
            dynamic_after_total = 0

        if (
            int(background_after.shape[0]) != int(background.shape[0])
            or dynamic_before_total != dynamic_after_total
        ):
            logger.info(
                "Runtime pointcloud caps applied (KNN init disabled): "
                "context=%s scene_id=%d segment_id=%d "
                "near_before=%d near_after=%d near_cap=%s "
                "distant_before=%d distant_after=%d distant_cap=%s "
                "dynamic_before=%d dynamic_after=%d dynamic_cap=%s",
                context,
                int(scene_id),
                int(segment_id),
                int(near_idx.shape[0]),
                int(near_keep.shape[0]),
                str(near_cap),
                int(distant_idx.shape[0]),
                int(distant_keep.shape[0]),
                str(distant_cap),
                int(dynamic_before_total),
                int(dynamic_after_total),
                str(dynamic_point_caps if dynamic_balance_on else dynamic_cap),
            )

        out = dict(pointcloud)
        out["background"] = background_after
        if isinstance(dynamic_raw, dict):
            out["dynamic"] = dynamic_after
        return out

    def _assert_knn_runtime_caps_match(
        self,
        *,
        segment_manifest: Dict[str, Any],
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> None:
        if not self._knn_requirements.enabled:
            return
        runtime_pc = self._runtime_pointcloud_cfg
        if not isinstance(runtime_pc, dict):
            return
        asset_pc = segment_manifest.get("pointcloud_config_normalized")
        if not isinstance(asset_pc, dict):
            raise ValueError(
                "Strict asset alignment requires segment manifest pointcloud_config_normalized for cap checks, "
                f"but it is missing/invalid (context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        strict_cap_keys = self._knn_strict_pointcloud_cap_keys()
        cap_mismatch = _pointcloud_cap_keys_differ(asset_pc, runtime_pc, strict_cap_keys)
        balance_mismatch = (
            self._knn_strict_dynamic_balance()
            and _pointcloud_dynamic_balance_differ(asset_pc, runtime_pc)
        )
        if not cap_mismatch and not balance_mismatch:
            return
        if balance_mismatch:
            raise ValueError(
                "Runtime dynamic_point_balance for KNN-backed rigid branches must match exported "
                "asset config; runtime dynamic downsample is disabled for KNN-backed branches in "
                "strict mode. "
                f"(context={context} branches={self._knn_required_branches_label()} "
                f"scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"asset_dynamic_point_balance={asset_pc.get('dynamic_point_balance')} "
                f"runtime_dynamic_point_balance={runtime_pc.get('dynamic_point_balance')}). "
                "Re-export assets (segment + segment_knn) with the current pointcloud config."
            )
        raise ValueError(
            "Runtime pointcloud caps for KNN-backed branches must match exported asset caps; "
            "runtime cap downsample is disabled for KNN-backed branches in strict mode. "
            f"(context={context} branches={self._knn_required_branches_label()} "
            f"scene_id={int(scene_id)} segment_id={int(segment_id)} "
            f"checked_cap_keys={list(strict_cap_keys)} "
            f"asset_caps={_extract_pointcloud_caps(asset_pc)} runtime_caps={_extract_pointcloud_caps(runtime_pc)}). "
            "Re-export assets (segment + segment_knn) with the current pointcloud config."
        )

    @staticmethod
    def _validate_segment_coordinate_metadata(
        segment_manifest: Dict[str, Any],
        *,
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> None:
        raw_frame = segment_manifest.get("asset_coordinate_frame")
        coord_meta = segment_manifest.get("coordinate_metadata")
        if raw_frame is None and isinstance(coord_meta, dict):
            raw_frame = coord_meta.get("asset_coordinate_frame")
        if raw_frame is None:
            return
        frame = str(raw_frame)
        if frame != _EXPECTED_ASSET_COORDINATE_FRAME:
            raise ValueError(
                "Unsupported StreetForward segment asset coordinate frame: "
                f"{frame!r} (expected {_EXPECTED_ASSET_COORDINATE_FRAME!r}; "
                f"context={context} scene_id={int(scene_id)} segment_id={int(segment_id)}). "
                "Re-export assets with the standard seg0 camera/OpenCV coordinate contract."
            )

    def _validate_required_knn_payload(
        self,
        *,
        pointcloud: Dict[str, Any],
        knn_init: Dict[str, Any],
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> None:
        if not self._knn_requirements.enabled:
            return
        bg_map, dyn_map = self._parse_and_validate_required_knn_maps(
            knn_init=knn_init,
            scene_id=scene_id,
            segment_id=segment_id,
            context=context,
        )

        background = np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)), dtype=np.float32)
        if background.ndim != 2 or background.shape[1] < 3:
            raise ValueError(
                "pointcloud.background must have shape [N,>=3] for KNN validation, "
                f"got {tuple(background.shape)} (scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        bg_count = int(background.shape[0])
        for k in self._knn_requirements.background_ks:
            arr_np = np.asarray(bg_map[int(k)], dtype=np.float32).reshape(-1)
            if int(arr_np.shape[0]) != int(bg_count):
                raise ValueError(
                    "knn_init background length mismatch with pointcloud background during KNN validation: "
                    f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                    f"k={int(k)} knn_len={int(arr_np.shape[0])} background_points={int(bg_count)})"
                )

        dynamic = pointcloud.get("dynamic", {})
        if not isinstance(dynamic, dict):
            raise ValueError(
                f"pointcloud.dynamic must be a dict for KNN validation, got {type(dynamic)} "
                f"(scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        dynamic_intids = sorted(int(x) for x in dynamic.keys())
        for k in self._knn_requirements.dynamic_ks:
            per_instance_raw = dyn_map[int(k)]
            if not isinstance(per_instance_raw, dict):
                raise ValueError(
                    f"knn_init.dynamic_avg_dist_by_k[{int(k)}] must be a dict[intid -> np.ndarray] "
                    f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
                )
            per_instance = {int(intid): v for intid, v in per_instance_raw.items()}
            missing_intids = [int(i) for i in dynamic_intids if int(i) not in per_instance]
            if missing_intids:
                raise ValueError(
                    "knn_init dynamic KNN payload missing required instances: "
                    f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                    f"k={int(k)} missing_intids={missing_intids}). "
                    "Re-export segment KNN assets with the current config."
                )
            for intid in dynamic_intids:
                pts = np.asarray(dynamic[int(intid)], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    raise ValueError(
                        f"pointcloud.dynamic[{int(intid)}] must have shape [N,>=3], got {tuple(pts.shape)} "
                        f"(scene_id={int(scene_id)} segment_id={int(segment_id)})"
                    )
                arr_np = np.asarray(per_instance[int(intid)], dtype=np.float32).reshape(-1)
                if int(arr_np.shape[0]) != int(pts.shape[0]):
                    raise ValueError(
                        "knn_init dynamic length mismatch with pointcloud dynamic during KNN validation: "
                        f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                        f"k={int(k)} intid={int(intid)} knn_len={int(arr_np.shape[0])} "
                        f"dynamic_points={int(pts.shape[0])})"
                    )

        if self._knn_requirements.fixed_neighbor_enabled:
            self._validate_required_knn_neighbors(
                pointcloud=pointcloud,
                knn_init=knn_init,
                scene_id=scene_id,
                segment_id=segment_id,
                context=context,
            )

    def _validate_required_knn_neighbors(
        self,
        *,
        pointcloud: Dict[str, Any],
        knn_init: Dict[str, Any],
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> None:
        required_k_store = int(self._knn_requirements.neighbor_k_store)
        if required_k_store <= 0:
            raise ValueError(
                "Internal error: fixed_neighbor_enabled requires neighbor_k_store > 0, "
                f"got {required_k_store}"
            )

        bg_knn_raw = knn_init.get("bg_knn_idx")
        rigid_knn_raw = knn_init.get("rigid_knn_idx")
        if bg_knn_raw is None or rigid_knn_raw is None:
            raise ValueError(
                "Stage5_1 fixed cached KNN is required but segment knn_init lacks bg_knn_idx/rigid_knn_idx: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)}). "
                "Run tools/build_streetforward_segment_knn_assets.py before training."
            )

        bg_knn = np.asarray(bg_knn_raw, dtype=np.int64)
        rigid_knn = np.asarray(rigid_knn_raw, dtype=np.int64)
        if bg_knn.ndim != 2:
            raise ValueError(
                f"knn_init.bg_knn_idx must be rank-2 (context={context} scene_id={int(scene_id)} "
                f"segment_id={int(segment_id)}), got {tuple(bg_knn.shape)}"
            )
        if rigid_knn.ndim != 2:
            raise ValueError(
                f"knn_init.rigid_knn_idx must be rank-2 (context={context} scene_id={int(scene_id)} "
                f"segment_id={int(segment_id)}), got {tuple(rigid_knn.shape)}"
            )
        if int(bg_knn.shape[1]) < int(required_k_store):
            raise ValueError(
                "knn_init.bg_knn_idx neighbor_k_store must be >= required value: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"required={required_k_store} got={int(bg_knn.shape[1])})"
            )
        if int(rigid_knn.shape[1]) < int(required_k_store):
            raise ValueError(
                "knn_init.rigid_knn_idx neighbor_k_store must be >= required value: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"required={required_k_store} got={int(rigid_knn.shape[1])})"
            )

        meta_k_store = int(knn_init.get("knn_neighbor_k_store", 0) or 0)
        if int(meta_k_store) > 0 and int(meta_k_store) < int(required_k_store):
            raise ValueError(
                "knn_init.knn_neighbor_k_store must be >= required neighbor_k_store when provided: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"required={required_k_store} got={meta_k_store})"
            )

        background = np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)), dtype=np.float32)
        if background.ndim != 2 or background.shape[1] < 3:
            raise ValueError(
                "pointcloud.background must have shape [N,>=3] for fixed neighbor validation, "
                f"got {tuple(background.shape)} (scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        bg_count = int(background.shape[0])
        if int(bg_knn.shape[0]) != int(bg_count):
            raise ValueError(
                "knn_init.bg_knn_idx row count mismatch with pointcloud background: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"knn_rows={int(bg_knn.shape[0])} background_points={bg_count})"
            )
        if int(bg_count) > 0 and (
            np.any(bg_knn < 0) or np.any(bg_knn >= int(bg_count))
        ):
            raise ValueError(
                "knn_init.bg_knn_idx contains out-of-range values; expected [0, N_bg) "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} N_bg={bg_count})"
            )

        dynamic = pointcloud.get("dynamic", {})
        if not isinstance(dynamic, dict):
            raise ValueError(
                f"pointcloud.dynamic must be a dict for fixed neighbor validation, got {type(dynamic)} "
                f"(scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        rigid_total = 0
        for intid in sorted(int(x) for x in dynamic.keys()):
            pts = np.asarray(dynamic[int(intid)], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(
                    f"pointcloud.dynamic[{int(intid)}] must have shape [N,>=3], got {tuple(pts.shape)} "
                    f"(scene_id={int(scene_id)} segment_id={int(segment_id)})"
                )
            rigid_total += int(pts.shape[0])
        if int(rigid_knn.shape[0]) != int(rigid_total):
            raise ValueError(
                "knn_init.rigid_knn_idx row count mismatch with pointcloud dynamic total: "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"knn_rows={int(rigid_knn.shape[0])} dynamic_points={rigid_total})"
            )
        if int(rigid_total) > 0 and (
            np.any(rigid_knn < 0) or np.any(rigid_knn >= int(rigid_total))
        ):
            raise ValueError(
                "knn_init.rigid_knn_idx contains out-of-range values; expected [0, N_rigid) "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"N_rigid={rigid_total})"
            )

    @staticmethod
    def _sample_knn_neighbor_columns(
        *,
        available_k_store: int,
        required_k_store: int,
    ) -> Optional[np.ndarray]:
        available = int(available_k_store)
        required = int(required_k_store)
        if available < required:
            raise ValueError(f"available_k_store must be >= required_k_store, got {available} < {required}")
        if required <= 0:
            raise ValueError(f"required_k_store must be > 0 for fixed cached neighbors, got {required}")
        if available == required:
            return None
        sampled = np.random.choice(available, size=required, replace=False)
        sampled = np.sort(np.asarray(sampled, dtype=np.int64))
        return sampled

    def _parse_and_validate_required_knn_maps(
        self,
        *,
        knn_init: Dict[str, Any],
        scene_id: int,
        segment_id: int,
        context: str,
    ) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        bg_map_raw = knn_init.get("background_avg_dist_by_k", {})
        dyn_map_raw = knn_init.get("dynamic_avg_dist_by_k", {})
        if not isinstance(bg_map_raw, dict):
            raise ValueError(
                "knn_init.background_avg_dist_by_k must be a dict "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        if not isinstance(dyn_map_raw, dict):
            raise ValueError(
                "knn_init.dynamic_avg_dist_by_k must be a dict "
                f"(context={context} scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        bg_map = {int(k): v for k, v in bg_map_raw.items()}
        dyn_map = {int(k): v for k, v in dyn_map_raw.items()}

        missing_bg_ks = [int(k) for k in self._knn_requirements.background_ks if int(k) not in bg_map]
        missing_dyn_ks = [int(k) for k in self._knn_requirements.dynamic_ks if int(k) not in dyn_map]
        if missing_bg_ks or missing_dyn_ks:
            raise ValueError(
                "KNN scale init is required by model config but segment knn_init is missing required k values: "
                f"(context={context} branches={self._knn_required_branches_label()} "
                f"scene_id={int(scene_id)} segment_id={int(segment_id)} "
                f"missing_background_ks={missing_bg_ks} missing_dynamic_ks={missing_dyn_ks}). "
                "Re-export segment KNN assets with the current config."
            )
        return bg_map, dyn_map

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

    def _parse_max_dynamic_points_per_segment(self) -> Optional[int]:
        pointcloud_cfg = self._cfg_get(self.dataset_cfg, "pointcloud", {}) or {}
        raw = self._cfg_get(pointcloud_cfg, "max_dynamic_points_per_segment")
        if raw is None:
            segment_filter_cfg = self._cfg_get(self.dataset_cfg, "segment_filter", {}) or {}
            raw = self._cfg_get(segment_filter_cfg, "max_dynamic_points")
        if raw is None:
            raw = self._cfg_get(self.dataset_cfg, "max_dynamic_points_per_segment")
        if raw is None:
            return None
        limit = int(raw)
        if limit <= 0:
            raise ValueError(
                "dataset.pointcloud.max_dynamic_points_per_segment must be > 0 when set, "
                f"got {raw!r}"
            )
        return int(limit)

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
            out = registered
        else:
            reg_set = set(registered)
            out = [sid for sid in configured if sid in reg_set] if len(reg_set) > 0 else list(configured)
            if len(out) == 0:
                raise ValueError(
                    f"No train scenes from config exist in segment registry (dataset={ds_name}, configured={configured})"
                )
        if self._max_dynamic_points_per_segment is not None:
            out = [int(sid) for sid in out if len(self.list_segment_ids(int(sid))) > 0]
            if len(out) == 0:
                raise ValueError(
                    "No training scenes remain after filtering segments by "
                    "dataset.pointcloud.max_dynamic_points_per_segment="
                    f"{int(self._max_dynamic_points_per_segment)}"
                )
        return out

    def list_segment_ids(self, scene_id: int) -> List[int]:
        ds_name = self._asset_dataset_name()
        seg_ids = self.asset_store.list_registered_segment_ids(ds_name, int(scene_id))
        if len(seg_ids) == 0:
            raise ValueError(
                f"No registered segments for dataset={ds_name} scene_id={int(scene_id)} in segment_registry"
            )
        if self._max_dynamic_points_per_segment is not None:
            seg_ids = self._filter_segment_ids_by_dynamic_points(
                ds_name=ds_name,
                scene_id=int(scene_id),
                segment_ids=seg_ids,
            )
        return seg_ids

    @staticmethod
    def _dynamic_point_count_from_stats(stats: Any) -> Optional[int]:
        if not isinstance(stats, dict):
            return None
        for key in ("dynamic_points", "dynamic_point_count", "num_dynamic_points"):
            if stats.get(key) is not None:
                return int(stats[key])
        return None

    def _load_segment_dynamic_point_count(self, ds_name: str, scene_id: int, segment_id: int) -> int:
        key = (str(ds_name), int(scene_id), int(segment_id))
        cached = self._segment_dynamic_point_count_cache.get(key)
        if cached is not None:
            return int(cached)

        handle = self.asset_store.get_segment_asset_registry_first(str(ds_name), int(scene_id), int(segment_id))
        manifest = handle.load_manifest()
        count = self._dynamic_point_count_from_stats(manifest.get("stats"))

        if count is None:
            stats_path = handle.asset_dir / "stats.json"
            if stats_path.exists():
                with stats_path.open("r", encoding="utf-8") as f:
                    count = self._dynamic_point_count_from_stats(json.load(f))

        if count is None:
            dyn_path = handle.asset_dir / "pointcloud_dynamic.npz"
            if dyn_path.exists():
                with np.load(str(dyn_path), allow_pickle=False) as z:
                    offsets = np.asarray(z["dynamic_points_offsets"], dtype=np.int64).reshape(-1)
                    count = int(offsets[-1]) if int(offsets.shape[0]) > 0 else 0

        if count is None:
            raise ValueError(
                "dataset.pointcloud.max_dynamic_points_per_segment requires segment dynamic point counts, "
                "but none were found in manifest stats, stats.json, or pointcloud_dynamic.npz "
                f"(dataset={ds_name} scene_id={int(scene_id)} segment_id={int(segment_id)})"
            )
        self._segment_dynamic_point_count_cache[key] = int(count)
        return int(count)

    def _filter_segment_ids_by_dynamic_points(
        self,
        *,
        ds_name: str,
        scene_id: int,
        segment_ids: Sequence[int],
    ) -> List[int]:
        limit = self._max_dynamic_points_per_segment
        if limit is None:
            return [int(x) for x in segment_ids]
        kept: List[int] = []
        skipped: List[Tuple[int, int]] = []
        for segment_id in segment_ids:
            count = self._load_segment_dynamic_point_count(
                str(ds_name),
                int(scene_id),
                int(segment_id),
            )
            if int(count) > int(limit):
                skipped.append((int(segment_id), int(count)))
                continue
            kept.append(int(segment_id))

        if skipped:
            log_key = (str(ds_name), int(scene_id))
            if log_key not in self._dynamic_point_filter_logged_scenes:
                self._dynamic_point_filter_logged_scenes.add(log_key)
                logger.warning(
                    "Skipping StreetForward segments with too many dynamic points: "
                    "dataset=%s scene_id=%d max_dynamic_points_per_segment=%d skipped_count=%d "
                    "skipped_preview=%s kept=%d",
                    str(ds_name),
                    int(scene_id),
                    int(limit),
                    int(len(skipped)),
                    skipped[:20],
                    int(len(kept)),
                )
        return kept

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
        scene_ids = self.list_training_scene_ids()
        segments_by_scene: Dict[int, List[int]] = {}
        total_segments = 0
        for scene_id in scene_ids:
            seg_ids = self.list_segment_ids(scene_id)
            segments_by_scene[int(scene_id)] = seg_ids
            total_segments += len(seg_ids)
        logger.info(
            "Asset validation begin: dataset=%s scenes=%d segments=%d knn_required=%s",
            ds_name,
            int(len(scene_ids)),
            int(total_segments),
            bool(self._knn_requirements.enabled),
        )
        checked = 0
        t_start = time.monotonic()
        for scene_id in scene_ids:
            seg_ids = segments_by_scene[int(scene_id)]
            logger.info(
                "Asset validation scene begin: dataset=%s scene_id=%s segments=%d",
                ds_name,
                int(scene_id),
                int(len(seg_ids)),
            )
            for segment_id in seg_ids:
                resolved = self.asset_store.resolve_segment_scene_assets_registry_first(
                    ds_name, int(scene_id), int(segment_id)
                )
                segment_manifest = resolved["segment_manifest"]
                self._validate_segment_coordinate_metadata(
                    segment_manifest,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    context="initialize/_validate_training_assets",
                )
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
                self._cache_segment_index_from_payload(ds_name, int(scene_id), int(segment_id), seg_idx)
                self._assert_knn_runtime_caps_match(
                    segment_manifest=segment_manifest,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    context="initialize/_validate_training_assets/pre_pointcloud",
                )
                if self._knn_requirements.enabled:
                    knn_init = resolved["segment_handle"].load_knn_init()
                    if not isinstance(knn_init, dict):
                        raise ValueError(
                            "KNN assets are required by model config, but segment knn_init asset is missing: "
                            f"(dataset={ds_name} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                            f"branches={self._knn_required_branches_label()}). "
                            "Run tools/build_streetforward_segment_knn_assets.py before training."
                        )
                    self._parse_and_validate_required_knn_maps(
                        knn_init=knn_init,
                        scene_id=int(scene_id),
                        segment_id=int(segment_id),
                        context="initialize/_validate_training_assets/pre_pointcloud",
                    )
                    pointcloud = resolved["segment_handle"].load_pointcloud()
                    self._validate_required_knn_payload(
                        pointcloud=pointcloud,
                        knn_init=knn_init,
                        scene_id=int(scene_id),
                        segment_id=int(segment_id),
                        context="initialize/_validate_training_assets/strict_runtime_caps",
                    )
                checked += 1
                if checked == 1 or checked == total_segments or (checked % 50) == 0:
                    logger.info(
                        "Asset validation progress: %d/%d (scene_id=%s segment_id=%s elapsed=%.1fs)",
                        int(checked),
                        int(total_segments),
                        int(scene_id),
                        int(segment_id),
                        float(time.monotonic() - t_start),
                    )
        logger.info(
            "Asset validation done: dataset=%s scenes=%d segments=%d elapsed=%.1fs",
            ds_name,
            int(len(scene_ids)),
            int(total_segments),
            float(time.monotonic() - t_start),
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
            self._validate_segment_coordinate_metadata(
                segment_manifest,
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                context="_resolve_segment_bundle",
            )

            segment_payload = segment_handle.load_segment_index()
            segment_pose = segment_handle.load_segment_pose()
            pointcloud = segment_handle.load_pointcloud()
            dynamic_tracks = segment_handle.load_dynamic_tracks()
            knn_init = segment_handle.load_knn_init()
            if not self._knn_requirements.fixed_neighbor_enabled:
                pointcloud, dynamic_tracks = self._reconcile_dynamic_payloads(
                    pointcloud=pointcloud,
                    dynamic_tracks=dynamic_tracks,
                )
            if not self._knn_requirements.enabled:
                pointcloud = self._apply_runtime_pointcloud_caps(
                    pointcloud=pointcloud,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    context="_resolve_segment_bundle/runtime_caps_no_knn_init",
                )
            if self._knn_requirements.enabled:
                if not isinstance(knn_init, dict):
                    raise ValueError(
                        "KNN assets are required by model config, but segment knn_init asset is missing: "
                        f"(dataset={ds_name} scene_id={int(scene_id)} segment_id={int(segment_id)} "
                        f"branches={self._knn_required_branches_label()}). "
                        "Run tools/build_streetforward_segment_knn_assets.py before training."
                    )
                self._validate_required_knn_payload(
                    pointcloud=pointcloud,
                    knn_init=knn_init,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    context="_resolve_segment_bundle/pre_runtime_cap_check",
                )
            sidx = self._build_segment_index_from_asset_payload(segment_payload)
            self._cache_segment_index(ds_name, int(scene_id), int(segment_id), sidx)

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
            self._assert_knn_runtime_caps_match(
                segment_manifest=segment_manifest,
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                context="_resolve_segment_bundle/pre_pointcloud",
            )

            if self._knn_requirements.enabled and isinstance(knn_init, dict):
                self._validate_required_knn_payload(
                    pointcloud=pointcloud,
                    knn_init=knn_init,
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    context="_resolve_segment_bundle/strict_runtime_caps",
                )

            if self._knn_requirements.enabled and knn_init is not None:
                bg_knn_map = knn_init.get("background_avg_dist_by_k", {})
                if isinstance(bg_knn_map, dict) and len(bg_knn_map) > 0:
                    any_bg_knn = next(iter(bg_knn_map.values()))
                    bg_count_knn = int(np.asarray(any_bg_knn).reshape(-1).shape[0])
                    bg_count_pc = int(np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)).shape[0]))
                    if bg_count_knn != bg_count_pc:
                        raise ValueError(
                            "KNN init background length mismatch with runtime pointcloud. "
                            f"(scene_id={int(scene_id)} segment_id={int(segment_id)} "
                            f"knn_background_points={int(bg_count_knn)} "
                            f"runtime_background_points={int(bg_count_pc)}). "
                            "KNN row alignment is required; re-export segment KNN assets with the current pointcloud config."
                        )

            bundle = SegmentStaticBundle(
                segment_asset_id=str(segment_manifest["asset_id"]),
                parent_scene_asset_id=parent_scene_asset_id,
                segment_index=sidx,
                segment_aabb=segment_aabb,
                segment_pose=segment_pose,
                pointcloud=pointcloud,
                dynamic_tracks=dynamic_tracks,
                knn_init=knn_init if self._knn_requirements.enabled else None,
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

    @staticmethod
    def _dynamic_point_counts_by_instance(dynamic_points: Any) -> Dict[int, int]:
        if not isinstance(dynamic_points, dict):
            return {}
        out: Dict[int, int] = {}
        for intid_raw, pts_raw in dynamic_points.items():
            intid = int(intid_raw)
            pts = np.asarray(pts_raw, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(
                    f"pointcloud.dynamic[{intid}] must have shape [N,>=3], got {tuple(pts.shape)}"
                )
            out[intid] = int(pts.shape[0])
        return out

    @staticmethod
    def _build_rigid_instance_layout(
        *,
        instance_intids: Sequence[int],
        point_counts_by_instance: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        intids = [int(x) for x in instance_intids]
        offsets: List[int] = [0]
        for intid in intids:
            n = int(point_counts_by_instance.get(intid, 0))
            if n < 0:
                raise ValueError(f"dynamic point count must be >= 0 for intid={intid}, got {n}")
            offsets.append(offsets[-1] + n)
        return (
            np.asarray(intids, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
        )

    @staticmethod
    def _validate_runtime_dynamic_layout_against_knn_init(
        *,
        knn_init: Dict[str, Any],
        runtime_instance_intids: Sequence[int],
        runtime_instance_offsets: np.ndarray,
    ) -> None:
        asset_intids_raw = knn_init.get("dynamic_instance_intids")
        asset_offsets_raw = knn_init.get("dynamic_offsets")
        if asset_intids_raw is None and asset_offsets_raw is None:
            return
        if asset_intids_raw is None or asset_offsets_raw is None:
            raise ValueError(
                "knn_init dynamic_instance_intids and dynamic_offsets must both be present when either is set."
            )

        asset_intids = [int(x) for x in np.asarray(asset_intids_raw, dtype=np.int64).reshape(-1).tolist()]
        asset_offsets = np.asarray(asset_offsets_raw, dtype=np.int64).reshape(-1)
        if asset_offsets.ndim != 1 or int(asset_offsets.shape[0]) != int(len(asset_intids)) + 1:
            raise ValueError(
                "knn_init dynamic_offsets shape mismatch: "
                f"expected len={len(asset_intids) + 1}, got {tuple(asset_offsets.shape)}"
            )
        if int(asset_offsets[0]) != 0:
            raise ValueError("knn_init dynamic_offsets must start at 0")
        if np.any(asset_offsets[1:] < asset_offsets[:-1]):
            raise ValueError("knn_init dynamic_offsets must be non-decreasing")

        runtime_intids = [int(x) for x in runtime_instance_intids]
        runtime_offsets = np.asarray(runtime_instance_offsets, dtype=np.int64).reshape(-1)
        if runtime_offsets.ndim != 1 or int(runtime_offsets.shape[0]) != int(len(runtime_intids)) + 1:
            raise ValueError(
                "runtime dynamic_offsets shape mismatch: "
                f"expected len={len(runtime_intids) + 1}, got {tuple(runtime_offsets.shape)}"
            )
        if int(runtime_offsets[0]) != 0:
            raise ValueError("runtime dynamic_offsets must start at 0")
        if np.any(runtime_offsets[1:] < runtime_offsets[:-1]):
            raise ValueError("runtime dynamic_offsets must be non-decreasing")

        asset_count_by_intid: Dict[int, int] = {}
        for i, intid in enumerate(asset_intids):
            cnt = int(asset_offsets[i + 1] - asset_offsets[i])
            if cnt < 0:
                raise ValueError(f"knn_init dynamic_offsets invalid count for intid={intid}: {cnt}")
            asset_count_by_intid[int(intid)] = cnt

        if len(asset_count_by_intid) == 0 and len(runtime_intids) > 0:
            # Backward compatibility for older fixed-KNN assets/tests that stored
            # rigid_knn_idx rows but no explicit dynamic_instance_intids layout.
            # Row-count alignment is validated before this call; keep the legacy
            # runtime dynamic ordering assumption instead of failing here.
            return

        for i, intid in enumerate(runtime_intids):
            if intid not in asset_count_by_intid:
                raise ValueError(
                    f"runtime dynamic instance {intid} is missing in knn_init.dynamic_instance_intids"
                )
            runtime_cnt = int(runtime_offsets[i + 1] - runtime_offsets[i])
            if runtime_cnt != int(asset_count_by_intid[int(intid)]):
                raise ValueError(
                    "runtime dynamic point count mismatches knn_init layout: "
                    f"intid={intid} runtime={runtime_cnt} asset={asset_count_by_intid[int(intid)]}"
                )

    def _build_segment_index_from_asset_payload(self, payload: Dict[str, Any]) -> SegmentIndexV4:
        asset_num_cams = int(payload["num_cams"])
        configured_cams = [int(x) for x in self._pixel_source_cameras]
        if len(configured_cams) > 0:
            expected_prefix = list(range(len(configured_cams)))
            if configured_cams != expected_prefix:
                raise ValueError(
                    "MultiSceneDatasetV4 currently expects data.pixel_source.cameras to be a zero-based "
                    f"contiguous prefix when narrowing pre-exported assets; got {configured_cams}. "
                    "Re-export assets with the desired camera order or use a prefix such as [0, 1, 2]."
                )
            if len(configured_cams) > asset_num_cams:
                raise ValueError(
                    "data.pixel_source.cameras requests more cameras than the asset contains: "
                    f"configured={configured_cams}, asset_num_cams={asset_num_cams}"
                )
            active_cam_ids = set(configured_cams)
            effective_num_cams = int(len(configured_cams))
        else:
            active_cam_ids = set(range(asset_num_cams))
            effective_num_cams = int(asset_num_cams)

        train_frames = [int(x) for x in payload["frame_indices"]]
        test_frames = [int(x) for x in payload["test_frame_indices"]]
        train_refs = tuple(
            (int(x[0]), int(x[1]))
            for x in np.asarray(payload["train_image_refs"]).tolist()
            if int(x[1]) in active_cam_ids
        )
        test_refs = tuple(
            (int(x[0]), int(x[1]))
            for x in np.asarray(payload["test_image_refs"]).tolist()
            if int(x[1]) in active_cam_ids
        )
        return SegmentIndexV4(
            scene_id=int(payload["scene_id"]),
            segment_id=int(payload["segment_id"]),
            num_cams=effective_num_cams,
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

    def _cache_segment_index(
        self,
        ds_name: str,
        scene_id: int,
        segment_id: int,
        sidx: SegmentIndexV4,
    ) -> SegmentIndexV4:
        key = (str(ds_name), int(scene_id), int(segment_id))
        with self._lock:
            self._segment_index_cache[key] = sidx
        return sidx

    def _cache_segment_index_from_payload(
        self,
        ds_name: str,
        scene_id: int,
        segment_id: int,
        payload: Dict[str, Any],
    ) -> SegmentIndexV4:
        return self._cache_segment_index(
            str(ds_name),
            int(scene_id),
            int(segment_id),
            self._build_segment_index_from_asset_payload(payload),
        )

    def _load_segment_index_light(self, scene_id: int, segment_id: int) -> SegmentIndexV4:
        ds_name = self._asset_dataset_name()
        key = (ds_name, int(scene_id), int(segment_id))
        with self._lock:
            bundle = self._cache_get(self._segment_static_cache, key)
            if bundle is not None:
                self._segment_index_cache[key] = bundle.segment_index
                return bundle.segment_index
            cached = self._segment_index_cache.get(key)
        if cached is not None:
            return cached

        resolved = self.asset_store.resolve_segment_scene_assets_registry_first(
            ds_name, int(scene_id), int(segment_id)
        )
        segment_manifest = resolved["segment_manifest"]
        self._validate_segment_coordinate_metadata(
            segment_manifest,
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            context="get_segment_index",
        )
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
        segment_payload = resolved["segment_handle"].load_segment_index()
        if int(segment_payload["scene_id"]) != int(scene_id) or int(segment_payload["segment_id"]) != int(segment_id):
            raise ValueError(
                "segment_index scene/segment_id mismatch vs registry: "
                f"expected=({scene_id},{segment_id}) "
                f"got=({segment_payload['scene_id']},{segment_payload['segment_id']})"
            )
        return self._cache_segment_index_from_payload(ds_name, int(scene_id), int(segment_id), segment_payload)

    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexV4:
        return self._load_segment_index_light(scene_id, segment_id)

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
        return t

    def _load_mask_from_asset_path(self, path_str: str, height: int, width: int) -> Tensor:
        arr = np.asarray(Image.open(path_str))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.as_tensor(arr, dtype=torch.float32)
        if mask.shape[0] != int(height) or mask.shape[1] != int(width):
            mask = self._resize_2d_tensor_to_hw(mask, int(height), int(width), mode="nearest")
        if mask.max().item() > 1.0:
            mask = (mask > 0.0).float()
        return mask

    def _resolve_egocar_mask_path(self, cam_id: int) -> Optional[Path]:
        ds_name = self._asset_dataset_name()
        candidates: List[int] = [int(cam_id)]
        # Some pipelines store cam indices as slot indices into pixel_source.cameras.
        if 0 <= int(cam_id) < len(self._pixel_source_cameras):
            mapped_cam_id = int(self._pixel_source_cameras[int(cam_id)])
            if mapped_cam_id not in candidates:
                candidates.append(mapped_cam_id)
        for cid in candidates:
            p = Path("data") / "ego_masks" / ds_name / f"{int(cid)}.png"
            if p.exists():
                return p
        return None

    def _load_egocar_mask_for_view(self, cam_id: int, height: int, width: int) -> Optional[Tensor]:
        if not self._load_egocar_mask:
            return None
        key = (self._asset_dataset_name(), int(cam_id), int(height), int(width))
        with self._lock:
            cached = self._cache_get(self._egocar_mask_cache, key)
        if cached is _EGO_MASK_MISSING:
            return None
        if torch.is_tensor(cached):
            return cached

        path = self._resolve_egocar_mask_path(int(cam_id))
        if path is None:
            ds_name = self._asset_dataset_name()
            warn_key = (ds_name, int(cam_id))
            with self._lock:
                if warn_key not in self._egocar_missing_warned:
                    self._egocar_missing_warned.add(warn_key)
                    logger.warning(
                        "No egocar mask template for dataset=%s cam_id=%d. "
                        "Expected file under data/ego_masks/%s/{cam_id}.png; "
                        "ego suppression for this camera will be disabled.",
                        ds_name,
                        int(cam_id),
                        ds_name,
                    )
            with self._lock:
                self._cache_set(
                    self._egocar_mask_cache,
                    key,
                    _EGO_MASK_MISSING,
                    max_items=self._egocar_mask_cache_max_items,
                )
            return None

        mask = self._load_mask_from_asset_path(str(path), int(height), int(width))
        mask = (mask > 0.5).float()
        with self._lock:
            self._cache_set(
                self._egocar_mask_cache,
                key,
                mask,
                max_items=self._egocar_mask_cache_max_items,
            )
        return mask

    @staticmethod
    def _normalize_sky_mask(mask: Tensor, semantics: str) -> Tensor:
        return normalize_sky_mask_to_one_is_sky((mask > 0.5).float(), semantics)

    def _compute_viewdirs(self, height: int, width: int, intrinsic: Tensor, camera_to_world: Tensor) -> Tensor:
        device = intrinsic.device
        intr = intrinsic[:3, :3]
        c2w = camera_to_world[:3, :3]
        xs = torch.arange(int(width), device=device, dtype=torch.float32)
        ys = torch.arange(int(height), device=device, dtype=torch.float32)
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
        image = torch.as_tensor(image_arr / 255.0, dtype=torch.float32)

        if depth_path:
            depth = self._load_depth_from_asset_path(depth_path, height, width)
        else:
            depth = torch.ones((height, width), dtype=torch.float32) * 10.0

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
        egocar_mask = self._load_egocar_mask_for_view(int(image_ref[1]), height, width)

        viewdirs = self._compute_viewdirs(
            height,
            width,
            torch.as_tensor(meta["intrinsic_4x4"], dtype=torch.float32),
            torch.as_tensor(meta["camera_to_world"], dtype=torch.float32),
        )

        return {
            "image": image,
            "extrinsic": torch.as_tensor(meta["camera_to_world"], dtype=torch.float32),
            "intrinsic": torch.as_tensor(meta["intrinsic_4x4"], dtype=torch.float32),
            "depth": depth,
            "sky_mask": sky_mask,
            "viewdirs": viewdirs,
            "dynamic_mask": dynamic_mask,
            "egocar_mask": egocar_mask,
            "frame_idx": int(image_ref[0]),
            "cam_idx": int(image_ref[1]),
        }

    def _preload_view_key_is_cached(self, key: Tuple[str, int, int, int, int]) -> bool:
        if not self._enable_view_pack_cache:
            return False
        with self._lock:
            return key in self._view_pack_cache

    def _get_cached_or_load_view(
        self,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
        *,
        materialize: bool = True,
    ) -> Dict[str, Any]:
        ds_name = self._asset_dataset_name()
        key = (ds_name, int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
        created_inflight = False
        if self._enable_view_pack_cache:
            with self._lock:
                cached = self._cache_get(self._view_pack_cache, key)
                if cached is not None:
                    if not materialize:
                        return {}
                    return loaded_view_pack_to_device_v2(cached, self.device)
            inflight = self._wait_on_inflight(self._view_pack_inflight, self._view_pack_inflight_lock, key)
            if inflight is not None:
                inflight.wait()
                with self._lock:
                    cached = self._cache_get(self._view_pack_cache, key)
                    if cached is not None:
                        if not materialize:
                            return {}
                        return loaded_view_pack_to_device_v2(cached, self.device)
                raise ValueError(f"view pack inflight missing after wait for {key}")
            created_inflight = True
        try:
            meta = self._load_image_meta(scene_id, segment_id, image_ref)
            pack = self._load_view_from_asset_paths(scene_id, image_ref, meta)
            lvp = dict_to_loaded_view_pack_v2(pack)
            if not self._enable_view_pack_cache:
                if not materialize:
                    return {}
                return loaded_view_pack_to_device_v2(lvp, self.device)
            with self._lock:
                cached = self._cache_set(
                    self._view_pack_cache,
                    key,
                    lvp,
                    max_items=self._view_pack_cache_max_items,
                )
                if not materialize:
                    return {}
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
                # Some frames in a training window can be absent from asset dynamic_tracks
                # (e.g. no annotated dynamic pose at that frame). Keep an explicit empty
                # frame entry so downstream rigid frame-index resolution remains stable.
                out[int(fidx)] = {"instances": {}}
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
        return out

    @staticmethod
    def _stack_optional_masks(
        items: List[Optional[Tensor]],
        fallback_images: List[Tensor],
        *,
        as_viewdirs: bool,
        device: torch.device,
        force_zeros_if_all_missing: bool = False,
    ) -> Optional[Tensor]:
        if not any(x is not None for x in items) and not force_zeros_if_all_missing:
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
        aux_image_refs: Optional[Sequence[ImageRef]] = None,
        query_label_image_refs: Optional[Sequence[ImageRef]] = None,
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
        if aux_image_refs is not None:
            for ref in aux_image_refs:
                self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="train")
        if query_label_image_refs is not None:
            for ref in query_label_image_refs:
                self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="train")
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
                egocar_masks,
                images,
                as_viewdirs=False,
                device=self.device,
                force_zeros_if_all_missing=bool(self._load_egocar_mask),
            )
            if ego is not None:
                out["egocar_mask"] = ego
            return out

        source = _load_role(source_image_refs, allow_missing_keyframe=False)
        target = _load_role(
            target_image_refs,
            allow_missing_keyframe=(target_ref_purpose == "test"),
        )
        aux_target = None
        if aux_image_refs is not None and len(aux_image_refs) > 0:
            aux_target = _load_role(
                aux_image_refs,
                allow_missing_keyframe=False,
            )
        query_label = None
        if query_label_image_refs is not None and len(query_label_image_refs) > 0:
            query_label = _load_role(
                query_label_image_refs,
                allow_missing_keyframe=False,
            )
        all_frames = set(source["frame_indices"].tolist()) | set(target["frame_indices"].tolist())
        if aux_target is not None:
            all_frames = all_frames | set(aux_target["frame_indices"].tolist())
        if query_label is not None:
            all_frames = all_frames | set(query_label["frame_indices"].tolist())

        pointcloud = bundle.pointcloud
        knn_init = bundle.knn_init
        needs_fixed_neighbors = bool(self._knn_requirements.fixed_neighbor_enabled)
        dynamic_info = None
        visible_intids_in_batch: set[int] = set()
        dynamic_points_full = pointcloud.get("dynamic")
        if isinstance(dynamic_points_full, dict) and len(dynamic_points_full) > 0:
            if not bundle.dynamic_tracks:
                raise ValueError(
                    "dynamic pointcloud is non-empty but dynamic_tracks is missing in strict asset mode"
                )
            dynamic_info = self._build_dynamic_info_from_asset_tracks(
                bundle.dynamic_tracks, frame_indices=sorted(int(x) for x in all_frames)
            )
            if needs_fixed_neighbors and not dynamic_info:
                # Fixed cached KNN keeps full-segment dynamic row-space in pointcloud.dynamic.
                # Some frame windows can have no visible dynamic instances; provide empty per-frame
                # entries so trainer rigid initialization can proceed without requiring visible poses.
                dynamic_info = {
                    int(fid): {"instances": {}}
                    for fid in sorted(int(x) for x in all_frames)
                }
            if dynamic_info is not None:
                for frame_obj in dynamic_info.values():
                    instances = frame_obj.get("instances", {})
                    for intid in instances.keys():
                        visible_intids_in_batch.add(int(intid))
            # Non-fixed KNN path can window dynamic points to current batch-visible instances.
            # Fixed cached KNN must keep full-segment rigid row-space stable across batches,
            # otherwise cached node_state_rigid rows will drift from rigid_knn_idx rows.
            if not needs_fixed_neighbors:
                pointcloud = dict(pointcloud)
                pointcloud["dynamic"] = {
                    int(intid): dynamic_points_full[int(intid)]
                    for intid in sorted(visible_intids_in_batch)
                    if int(intid) in dynamic_points_full
                }
                if len(visible_intids_in_batch) == 0:
                    dynamic_info = None

        knn_init_batch: Optional[Dict[str, Any]] = None
        if self._knn_requirements.enabled and knn_init is not None:
            bg_map_raw = knn_init.get("background_avg_dist_by_k", {})
            if not isinstance(bg_map_raw, dict):
                raise ValueError("knn_init.background_avg_dist_by_k must be a dict")
            bg_count = int(np.asarray(pointcloud.get("background", np.zeros((0, 6), dtype=np.float32)).shape[0]))
            bg_map: Dict[int, np.ndarray] = {}
            for k, arr in bg_map_raw.items():
                k_i = int(k)
                arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
                if int(arr_np.shape[0]) != int(bg_count):
                    raise ValueError(
                        "knn_init background length mismatch with runtime pointcloud: "
                        f"k={k_i} knn_len={arr_np.shape[0]} bg_count={bg_count}"
                    )
                bg_map[k_i] = arr_np

            dyn_map_raw = knn_init.get("dynamic_avg_dist_by_k", {})
            if not isinstance(dyn_map_raw, dict):
                raise ValueError("knn_init.dynamic_avg_dist_by_k must be a dict")
            dyn_points_now = pointcloud.get("dynamic", {})
            dyn_ids_now = (
                sorted(int(x) for x in dyn_points_now.keys())
                if isinstance(dyn_points_now, dict)
                else []
            )
            dyn_point_counts: Dict[int, int] = {}
            for intid in dyn_ids_now:
                pts = np.asarray(dyn_points_now[intid], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    raise ValueError(
                        f"pointcloud.dynamic[{intid}] must have shape [N,>=3], got {tuple(pts.shape)}"
                    )
                dyn_point_counts[int(intid)] = int(pts.shape[0])
            dyn_map: Dict[int, Dict[int, np.ndarray]] = {}
            for k, per_instance in dyn_map_raw.items():
                k_i = int(k)
                if not isinstance(per_instance, dict):
                    raise ValueError(f"knn_init.dynamic_avg_dist_by_k[{k_i}] must be a dict[intid -> np.ndarray]")
                out_per: Dict[int, np.ndarray] = {}
                for intid in dyn_ids_now:
                    if intid not in per_instance:
                        raise ValueError(
                            f"knn_init missing dynamic instance {intid} for k={k_i}"
                        )
                    arr_np = np.asarray(per_instance[intid], dtype=np.float32).reshape(-1)
                    if int(arr_np.shape[0]) != int(dyn_point_counts[int(intid)]):
                        raise ValueError(
                            "knn_init dynamic length mismatch with runtime pointcloud: "
                            f"k={k_i} intid={intid} knn_len={arr_np.shape[0]} pts={dyn_point_counts[int(intid)]}"
                        )
                    out_per[int(intid)] = arr_np
                dyn_map[k_i] = out_per
            rigid_total_now = int(sum(int(x) for x in dyn_point_counts.values()))
            rigid_instance_intids_now, rigid_instance_offsets_now = self._build_rigid_instance_layout(
                instance_intids=dyn_ids_now,
                point_counts_by_instance=dyn_point_counts,
            )

            bg_knn_batch: Optional[np.ndarray] = None
            rigid_knn_batch: Optional[np.ndarray] = None
            knn_neighbor_k_store_batch: Optional[int] = None
            rigid_knn_row_ids_batch: Optional[np.ndarray] = None
            rigid_instance_intids_batch: Optional[np.ndarray] = None
            rigid_instance_offsets_batch: Optional[np.ndarray] = None
            if needs_fixed_neighbors:
                bg_knn_raw = knn_init.get("bg_knn_idx")
                rigid_knn_raw = knn_init.get("rigid_knn_idx")
                if bg_knn_raw is None or rigid_knn_raw is None:
                    raise ValueError(
                        "Stage5_1 fixed cached KNN requires knn_init.bg_knn_idx and knn_init.rigid_knn_idx."
                    )
                bg_knn_np = np.asarray(bg_knn_raw)
                rigid_knn_np = np.asarray(rigid_knn_raw)
                if bg_knn_np.ndim != 2 or rigid_knn_np.ndim != 2:
                    raise ValueError(
                        "knn_init bg_knn_idx/rigid_knn_idx must both be rank-2 "
                        f"(got bg={tuple(bg_knn_np.shape)} rigid={tuple(rigid_knn_np.shape)})"
                    )
                if int(bg_knn_np.shape[1]) != int(rigid_knn_np.shape[1]):
                    raise ValueError(
                        "knn_init bg_knn_idx/rigid_knn_idx K_store mismatch: "
                        f"{bg_knn_np.shape[1]} vs {rigid_knn_np.shape[1]}"
                    )
                if int(bg_knn_np.shape[0]) != int(bg_count):
                    raise ValueError(
                        "knn_init bg_knn_idx row mismatch with runtime background pointcloud: "
                        f"knn_rows={bg_knn_np.shape[0]} bg_count={bg_count}"
                    )
                if int(rigid_knn_np.shape[0]) != int(rigid_total_now):
                    rigid_counts_full = self._dynamic_point_counts_by_instance(dynamic_points_full)
                    rigid_total_full = int(sum(int(x) for x in rigid_counts_full.values()))
                    raise ValueError(
                        "fixed_cached rigid KNN requires runtime rigid row-space == full-segment row-space: "
                        f"knn_rows={rigid_knn_np.shape[0]} rigid_total_now={rigid_total_now} "
                        f"full_segment_rigid_total={rigid_total_full} selected_instances={dyn_ids_now}"
                    )
                if not np.issubdtype(bg_knn_np.dtype, np.integer):
                    bg_knn_np = bg_knn_np.astype(np.int64, copy=False)
                if not np.issubdtype(rigid_knn_np.dtype, np.integer):
                    rigid_knn_np = rigid_knn_np.astype(np.int64, copy=False)

                required_k_store = int(self._knn_requirements.neighbor_k_store) if needs_fixed_neighbors else 0
                if int(required_k_store) <= 0:
                    raise ValueError(
                        "Internal error: fixed cached KNN requires neighbor_k_store > 0, "
                        f"got {required_k_store}"
                    )
                available_k_store = int(bg_knn_np.shape[1])
                if int(available_k_store) < int(required_k_store):
                    raise ValueError(
                        "knn_init bg_knn_idx neighbor_k_store is smaller than runtime requirement: "
                        f"required={required_k_store} got={available_k_store}"
                    )
                if int(rigid_knn_np.shape[1]) < int(required_k_store):
                    raise ValueError(
                        "knn_init rigid_knn_idx neighbor_k_store is smaller than runtime requirement: "
                        f"required={required_k_store} got={rigid_knn_np.shape[1]}"
                    )
                meta_k_store = int(knn_init.get("knn_neighbor_k_store", 0) or 0)
                if int(meta_k_store) > 0 and int(meta_k_store) < int(required_k_store):
                    raise ValueError(
                        "knn_init.knn_neighbor_k_store is smaller than runtime requirement: "
                        f"required={required_k_store} got={meta_k_store}"
                    )
                sampled_cols = self._sample_knn_neighbor_columns(
                    available_k_store=int(available_k_store),
                    required_k_store=int(required_k_store),
                )
                if sampled_cols is not None:
                    bg_knn_np = np.ascontiguousarray(bg_knn_np[:, sampled_cols], dtype=np.int64)
                    rigid_knn_np = np.ascontiguousarray(rigid_knn_np[:, sampled_cols], dtype=np.int64)
                else:
                    bg_knn_np = np.ascontiguousarray(bg_knn_np, dtype=np.int64)
                    rigid_knn_np = np.ascontiguousarray(rigid_knn_np, dtype=np.int64)
                bg_knn_batch = bg_knn_np
                rigid_knn_batch = rigid_knn_np
                knn_neighbor_k_store_batch = int(required_k_store)
                self._validate_runtime_dynamic_layout_against_knn_init(
                    knn_init=knn_init,
                    runtime_instance_intids=rigid_instance_intids_now.tolist(),
                    runtime_instance_offsets=rigid_instance_offsets_now,
                )
                rigid_knn_row_ids_batch = np.arange(int(rigid_knn_np.shape[0]), dtype=np.int64)
                rigid_instance_intids_batch = np.ascontiguousarray(rigid_instance_intids_now, dtype=np.int64)
                rigid_instance_offsets_batch = np.ascontiguousarray(rigid_instance_offsets_now, dtype=np.int64)

            knn_init_batch = {
                "background_avg_dist_by_k": bg_map,
                "dynamic_avg_dist_by_k": dyn_map,
            }
            if bg_knn_batch is not None and rigid_knn_batch is not None:
                knn_init_batch["bg_knn_idx"] = bg_knn_batch
                knn_init_batch["rigid_knn_idx"] = rigid_knn_batch
                knn_init_batch["knn_neighbor_k_store"] = int(knn_neighbor_k_store_batch or 0)
                if rigid_knn_row_ids_batch is not None:
                    knn_init_batch["rigid_knn_row_ids"] = rigid_knn_row_ids_batch
                if rigid_instance_intids_batch is not None and rigid_instance_offsets_batch is not None:
                    knn_init_batch["rigid_instance_intids"] = rigid_instance_intids_batch
                    knn_init_batch["rigid_instance_offsets"] = rigid_instance_offsets_batch

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
                "aux_image_refs": [tuple(r) for r in aux_image_refs] if aux_image_refs is not None else [],
                "query_label_refs": [tuple(r) for r in query_label_image_refs]
                if query_label_image_refs is not None
                else [],
                "test_image_refs": None,
                "assembly_mode": "image_ref_v4",
            },
            "source": source,
            "target": target,
            "pointcloud": pointcloud,
        }
        if aux_target is not None:
            batch["aux_target"] = aux_target
        if query_label is not None:
            batch["query_label"] = query_label
        if dynamic_info is not None:
            batch["dynamic_info"] = dynamic_info
        if knn_init_batch is not None:
            batch["knn_init"] = knn_init_batch
            if "bg_knn_idx" in knn_init_batch and "rigid_knn_idx" in knn_init_batch:
                batch["knn_struct_neighbors"] = {
                    "bg_knn_idx": knn_init_batch["bg_knn_idx"],
                    "rigid_knn_idx": knn_init_batch["rigid_knn_idx"],
                    "knn_neighbor_k_store": int(knn_init_batch.get("knn_neighbor_k_store", 0)),
                }
                if "rigid_knn_row_ids" in knn_init_batch:
                    batch["knn_struct_neighbors"]["rigid_knn_row_ids"] = knn_init_batch["rigid_knn_row_ids"]
                if "rigid_instance_intids" in knn_init_batch and "rigid_instance_offsets" in knn_init_batch:
                    batch["knn_struct_neighbors"]["rigid_instance_intids"] = knn_init_batch["rigid_instance_intids"]
                    batch["knn_struct_neighbors"]["rigid_instance_offsets"] = knn_init_batch["rigid_instance_offsets"]

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
        source_image_refs = getattr(request, "source_image_refs", None)
        source_refs = (
            [tuple(x) for x in source_image_refs]
            if source_image_refs is not None
            else [tuple(request.source_image_ref)]
        )
        return self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            source_refs,
            request.target_image_refs,
            aux_image_refs=None,
            include_test=include_test,
            test_image_refs=getattr(request, "test_image_refs", None) if include_test else None,
            enforce_target0_equals_source=enforce_target0_equals_source,
            target_ref_purpose="train",
        )

    @staticmethod
    def _flatten_v9_ref_groups(ref_groups: Sequence[Sequence[ImageRef]]) -> List[ImageRef]:
        out: List[ImageRef] = []
        for group in ref_groups:
            for ref in group:
                out.append((int(ref[0]), int(ref[1])))
        return out

    @staticmethod
    def _dedupe_v9_refs_keep_order(refs: Sequence[ImageRef]) -> List[ImageRef]:
        seen: set[ImageRef] = set()
        out: List[ImageRef] = []
        for ref in refs:
            r = (int(ref[0]), int(ref[1]))
            if r in seen:
                continue
            seen.add(r)
            out.append(r)
        return out

    @staticmethod
    def _dedupe_v9_refs_roles_keep_order(
        refs: Sequence[ImageRef],
        roles: Sequence[str],
    ) -> tuple[List[ImageRef], List[str]]:
        if len(refs) != len(roles):
            raise ValueError(f"V9 refs/roles length mismatch: {len(refs)} vs {len(roles)}")
        seen: set[ImageRef] = set()
        out_refs: List[ImageRef] = []
        out_roles: List[str] = []
        for ref, role in zip(refs, roles):
            r = (int(ref[0]), int(ref[1]))
            if r in seen:
                continue
            seen.add(r)
            out_refs.append(r)
            out_roles.append(str(role))
        return out_refs, out_roles

    @staticmethod
    def _assert_no_v9_ref_role_conflicts(refs: Sequence[ImageRef], roles: Sequence[str]) -> None:
        if len(refs) != len(roles):
            raise ValueError(f"V9 refs/roles length mismatch: {len(refs)} vs {len(roles)}")
        role_by_ref: Dict[ImageRef, str] = {}
        for ref, role in zip(refs, roles):
            r = (int(ref[0]), int(ref[1]))
            role_s = str(role)
            prev = role_by_ref.get(r)
            if prev is not None and prev != role_s:
                raise ValueError(f"V9 final supervision ref {r} has conflicting roles: {prev} vs {role_s}")
            role_by_ref[r] = role_s

    @staticmethod
    def _assert_no_iforward_ref_role_conflicts(refs: Sequence[ImageRef], roles: Sequence[str]) -> None:
        if len(refs) != len(roles):
            raise ValueError(f"IForward refs/roles length mismatch: {len(refs)} vs {len(roles)}")
        role_by_ref: Dict[ImageRef, str] = {}
        for ref, role in zip(refs, roles):
            r = (int(ref[0]), int(ref[1]))
            role_s = str(role)
            prev = role_by_ref.get(r)
            if prev is not None and prev != role_s:
                raise ValueError(f"IForward final supervision ref {r} has conflicting roles: {prev} vs {role_s}")
            role_by_ref[r] = role_s

    @staticmethod
    def _iforward_ref_key(ref: ImageRef) -> str:
        return f"{int(ref[0])}:{int(ref[1])}"

    def _assemble_segment_batch_from_iforward_request(
        self,
        *,
        scene_id: int,
        segment_id: int,
        plan: Any,
        include_test: bool = False,
    ) -> Dict[str, Any]:
        plan_scheduler_version = str(getattr(plan, "scheduler_version", ""))
        if plan_scheduler_version not in {
            "iforward_v1",
            "iforward_v3_random_window",
            "iforward_v4_coverage_ordered",
            _IFORWARD_STAGE2_1_SCHEDULER_VERSION,
            _IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
        }:
            raise ValueError(
                "expected IForwardRolloutPlan.scheduler_version == 'iforward_v1' "
                "or 'iforward_v3_random_window' or 'iforward_v4_coverage_ordered' "
                f"or '{_IFORWARD_STAGE2_1_SCHEDULER_VERSION}' or '{_IFORWARD_SEQUENCE10_SCHEDULER_VERSION}'"
            )
        if int(scene_id) != int(getattr(plan, "scene_id")) or int(segment_id) != int(getattr(plan, "segment_id")):
            raise ValueError(
                "IForward request scene/segment mismatch: "
                f"request=({int(scene_id)},{int(segment_id)}) "
                f"plan=({int(getattr(plan, 'scene_id'))},{int(getattr(plan, 'segment_id'))})"
            )

        evidence_refs = self._dedupe_v9_refs_keep_order(
            [(int(ref[0]), int(ref[1])) for ref in list(getattr(plan, "evidence_refs_flat", []) or [])]
        )
        target_refs_raw = [(int(ref[0]), int(ref[1])) for ref in list(getattr(plan, "target_refs_flat", []) or [])]
        target_roles_raw = [str(x) for x in list(getattr(plan, "target_roles_flat", []) or [])]
        if plan_scheduler_version in {
            "iforward_v4_coverage_ordered",
            _IFORWARD_STAGE2_1_SCHEDULER_VERSION,
            _IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
        }:
            # V4 final eval intentionally allows one image ref to appear under
            # both rollout-supervision and eval-only roles on the last rollout.
            target_refs, target_roles = [tuple(x) for x in target_refs_raw], [str(x) for x in target_roles_raw]
        else:
            self._assert_no_iforward_ref_role_conflicts(target_refs_raw, target_roles_raw)
            target_refs, target_roles = self._dedupe_v9_refs_roles_keep_order(target_refs_raw, target_roles_raw)
        if len(evidence_refs) == 0:
            raise ValueError("IForward request requires non-empty evidence refs")
        if len(target_refs) == 0:
            raise ValueError("IForward request requires non-empty target refs")
        if len(target_refs) != len(target_roles):
            raise ValueError("IForward target refs/roles length mismatch after dedupe")
        for ref in evidence_refs + target_refs:
            self.validate_image_ref(int(scene_id), int(segment_id), tuple(ref), purpose="train")

        bundle = self._resolve_segment_bundle(int(scene_id), int(segment_id))
        dynamic_points = bundle.pointcloud.get("dynamic") if isinstance(bundle.pointcloud, dict) else None
        if isinstance(dynamic_points, dict) and len(dynamic_points) > 0:
            if not bool(self._knn_requirements.fixed_neighbor_enabled):
                raise ValueError(
                    "IForward state carry requires stable full-segment rigid row-space. "
                    "Enable fixed cached KNN / fixed_neighbor_enabled, or add an "
                    "IForward full-dynamic-row-space assembly path."
                )

        batch = self._assemble_segment_batch_from_image_refs(
            int(scene_id),
            int(segment_id),
            evidence_refs,
            target_refs,
            aux_image_refs=None,
            query_label_image_refs=None,
            include_test=bool(include_test),
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )

        source_ref_to_index: Dict[ImageRef, int] = {tuple(ref): int(idx) for idx, ref in enumerate(evidence_refs)}
        target_ref_to_index: Dict[ImageRef, int] = {tuple(ref): int(idx) for idx, ref in enumerate(target_refs)}
        target_indices_by_role: Dict[str, List[int]] = {}
        target_refs_by_role: Dict[str, List[ImageRef]] = {}
        for idx, (ref, role) in enumerate(zip(target_refs, target_roles)):
            role_s = str(role)
            target_indices_by_role.setdefault(role_s, []).append(int(idx))
            target_refs_by_role.setdefault(role_s, []).append(tuple(ref))

        plan_dict = dataclasses.asdict(plan)
        steps_meta: List[Dict[str, Any]] = []
        for step in list(getattr(plan, "steps", []) or []):
            step_dict = dataclasses.asdict(step)
            step_refs = [(int(ref[0]), int(ref[1])) for ref in list(getattr(step, "evidence_refs", []) or [])]
            step_dict["evidence_refs"] = [tuple(x) for x in step_refs]
            step_dict["source_indices"] = [int(source_ref_to_index[tuple(ref)]) for ref in step_refs]
            steps_meta.append(step_dict)

        final_plan = getattr(plan, "final_supervision", None)
        final_meta = dataclasses.asdict(final_plan) if final_plan is not None else {}
        final_meta.update(
            {
                "refs": [tuple(x) for x in target_refs],
                "roles": [str(x) for x in target_roles],
                "target_indices_by_role": {
                    str(role): [int(x) for x in indices]
                    for role, indices in target_indices_by_role.items()
                },
                "target_refs_by_role": {
                    str(role): [tuple(x) for x in refs]
                    for role, refs in target_refs_by_role.items()
                },
            }
        )

        role_groups: List[Dict[str, Any]] = [
            {
                "role": "evidence_input",
                "refs": [tuple(x) for x in evidence_refs],
                "image_roles": ["evidence_input" for _ in evidence_refs],
                "allow_update_evidence": True,
                "allow_render_loss": False,
                "allow_memory_write": True,
                "mask_policy": str(
                    ((getattr(plan, "request_meta", None) or {}).get("evidence_mask_policy", "non_sky_non_egocar"))
                ),
            }
        ]
        for role, refs in target_refs_by_role.items():
            role_groups.append(
                {
                    "role": str(role),
                    "refs": [tuple(x) for x in refs],
                    "image_roles": [str(role) for _ in refs],
                    "allow_update_evidence": False,
                    "allow_render_loss": True,
                    "allow_memory_write": False,
                    "mask_policy": "non_sky_non_egocar",
                }
            )

        iforward_meta = {
            "scheduler_version": plan_scheduler_version,
            "model_family": "IForward",
            "scene_id": int(getattr(plan, "scene_id")),
            "segment_id": int(getattr(plan, "segment_id")),
            "episode_id": int(getattr(plan, "episode_id")),
            "rollout_id_global": int(getattr(plan, "rollout_id_global")),
            "rollout_idx_in_episode": int(getattr(plan, "rollout_idx_in_episode")),
            "rollouts_per_episode": int(getattr(plan, "rollouts_per_episode", 1)),
            "inner_K": int(getattr(plan, "inner_K")),
            "shape_name": str(getattr(plan, "shape_name")),
            "requested_blocks_per_rollout": int(getattr(plan, "requested_blocks_per_rollout")),
            "actual_blocks_per_rollout": int(getattr(plan, "actual_blocks_per_rollout")),
            "repeats_per_block": int(getattr(plan, "repeats_per_block")),
            "requested_inner_K": int(getattr(plan, "requested_inner_K")),
            "actual_inner_K": int(getattr(plan, "actual_inner_K")),
            "short_rollout": bool(getattr(plan, "short_rollout")),
            "short_rollout_reason": str(getattr(plan, "short_rollout_reason")),
            "input_frame_indices": [int(x) for x in list(getattr(plan, "input_frame_indices", []) or [])],
            "input_keyframe_indices": [int(x) for x in list(getattr(plan, "input_keyframe_indices", []) or [])],
            "delivery_frame_indices": [int(x) for x in list(getattr(plan, "delivery_frame_indices", []) or [])],
            "episode_num_blocks": int(getattr(plan, "episode_num_blocks", len(getattr(plan, "keyframe_window", []) or []))),
            "window_policy": str(getattr(plan, "window_policy", "")),
            "window_start": int(getattr(plan, "window_start", -1)),
            "window_end": int(getattr(plan, "window_end", -1)),
            "window_block_ids": [int(x) for x in list(getattr(plan, "window_block_ids", []) or [])],
            "window_keyframe_indices": [int(x) for x in list(getattr(plan, "window_keyframe_indices", []) or [])],
            "window_frame_indices": [int(x) for x in list(getattr(plan, "window_frame_indices", []) or [])],
            "window_hash": int(getattr(plan, "window_hash", -1)),
            "window_revisit_count": int(getattr(plan, "window_revisit_count", 0)),
            "unique_windows_seen": int(getattr(plan, "unique_windows_seen", 0)),
            "is_repeated_window": bool(getattr(plan, "is_repeated_window", False)),
            "sequence_id": int(getattr(plan, "sequence_id", -1)),
            "sequence_length": int(getattr(plan, "sequence_length", 0)),
            "sequence_stride": int(getattr(plan, "sequence_stride", 0)),
            "sequence_start_block_pos": int(getattr(plan, "sequence_start_block_pos", -1)),
            "sequence_block_ids": [int(x) for x in list(getattr(plan, "sequence_block_ids", []) or [])],
            "sequence_keyframe_indices": [int(x) for x in list(getattr(plan, "sequence_keyframe_indices", []) or [])],
            "sequence_source_frame_indices": [
                int(x) for x in list(getattr(plan, "sequence_source_frame_indices", []) or [])
            ],
            "sequence_positions": [int(x) for x in list(getattr(plan, "sequence_positions", []) or [])],
            "history_positions": [int(x) for x in list(getattr(plan, "history_positions", []) or [])],
            "repair_positions": [int(x) for x in list(getattr(plan, "repair_positions", []) or [])],
            "scheduler_phase": str(getattr(plan, "scheduler_phase", "")),
            "rollout_phase": str(getattr(plan, "rollout_phase", "")),
            "repair_enabled": bool(getattr(plan, "repair_enabled", False)),
            "repair_permutation_hash": int(getattr(plan, "repair_permutation_hash", -1)),
            "temporal_read_count": int(getattr(plan, "temporal_read_count", 0)),
            "temporal_commit_count": int(getattr(plan, "temporal_commit_count", 0)),
            "observation_commit_count": int(getattr(plan, "observation_commit_count", 0)),
            "optimizer_memory_update_count": int(getattr(plan, "optimizer_memory_update_count", 0)),
            "evidence_refs_flat": [tuple(x) for x in evidence_refs],
            "target_refs_flat": [tuple(x) for x in target_refs],
            "target_roles_flat": [str(x) for x in target_roles],
            "steps": steps_meta,
            "source_ref_to_index_keyed": {
                self._iforward_ref_key(ref): int(idx)
                for ref, idx in source_ref_to_index.items()
            },
            "target_ref_to_index_keyed": {
                self._iforward_ref_key(ref): int(idx)
                for ref, idx in target_ref_to_index.items()
            },
            "final_supervision": final_meta,
            "reset_scene_state_before_rollout": bool(getattr(plan, "reset_scene_state_before_rollout")),
            "carry_scene_state_after_rollout": bool(getattr(plan, "carry_scene_state_after_rollout")),
            "episode_end_after_rollout": bool(getattr(plan, "episode_end_after_rollout")),
            "discard_scene_state_after_rollout": bool(getattr(plan, "episode_end_after_rollout")),
            "detach_graph_after_rollout": bool(getattr(plan, "detach_graph_after_rollout")),
            "leakage_check": dict(getattr(plan, "leakage_check", {}) or {}),
        }

        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(dict(getattr(plan, "request_meta", {}) or {}))
        request_meta["scheduler_version"] = plan_scheduler_version
        request_meta["model_family"] = "IForward"
        request_meta["scene_id"] = int(getattr(plan, "scene_id"))
        request_meta["segment_id"] = int(getattr(plan, "segment_id"))
        request_meta["episode_id"] = int(getattr(plan, "episode_id"))
        request_meta["episode_idx_global"] = int(getattr(plan, "episode_id"))
        request_meta["rollout_id_global"] = int(getattr(plan, "rollout_id_global"))
        request_meta["rollout_idx_in_episode"] = int(getattr(plan, "rollout_idx_in_episode"))
        request_meta["inner_K"] = int(getattr(plan, "inner_K"))
        request_meta["source_image_refs"] = [tuple(x) for x in evidence_refs]
        request_meta["source_image_ref"] = tuple(evidence_refs[0])
        request_meta["target_image_refs"] = [tuple(x) for x in target_refs]
        request_meta["target_image_roles"] = [str(x) for x in target_roles]
        request_meta["role_groups"] = role_groups
        request_meta["iforward"] = iforward_meta
        request_meta["assembly_mode"] = "image_ref_iforward_v1"
        if len(request_meta["target_image_refs"]) != len(request_meta["target_image_roles"]):
            raise ValueError(
                "IForward target_image_refs/target_image_roles mismatch after assembly: "
                f"{len(request_meta['target_image_refs'])} vs {len(request_meta['target_image_roles'])}"
            )
        batch["request_meta"] = request_meta
        batch["_iforward"] = iforward_meta
        batch["_iforward_plan"] = plan_dict
        batch["_iforward_runtime_maps"] = {
            "source_ref_to_index": dict(source_ref_to_index),
            "target_ref_to_index": dict(target_ref_to_index),
        }
        return batch

    def _assemble_segment_batch_from_iforward_random_window_request(
        self,
        *,
        scene_id: int,
        segment_id: int,
        plan: Any,
        include_test: bool = False,
    ) -> Dict[str, Any]:
        from models.iforward.random_window_batch import (
            RANDOM_WINDOW_ASSEMBLY_MODE,
            RANDOM_WINDOW_MODEL_FAMILY,
            RANDOM_WINDOW_SCHEDULER_VERSION,
        )

        if str(getattr(plan, "scheduler_version", "")) != RANDOM_WINDOW_SCHEDULER_VERSION:
            raise ValueError("expected IForwardRandomWindowPlan.scheduler_version == 'random_window_v1'")
        if int(scene_id) != int(getattr(plan, "scene_id")) or int(segment_id) != int(getattr(plan, "segment_id")):
            raise ValueError(
                "IForward random-window request scene/segment mismatch: "
                f"request=({int(scene_id)},{int(segment_id)}) "
                f"plan=({int(getattr(plan, 'scene_id'))},{int(getattr(plan, 'segment_id'))})"
            )

        evidence_refs = self._dedupe_v9_refs_keep_order(
            [(int(ref[0]), int(ref[1])) for ref in list(getattr(plan, "evidence_refs_flat", []) or [])]
        )
        target_refs_raw = [(int(ref[0]), int(ref[1])) for ref in list(getattr(plan, "target_refs_flat", []) or [])]
        target_roles_raw = [str(x) for x in list(getattr(plan, "target_roles_flat", []) or [])]
        self._assert_no_iforward_ref_role_conflicts(target_refs_raw, target_roles_raw)
        target_refs, target_roles = self._dedupe_v9_refs_roles_keep_order(target_refs_raw, target_roles_raw)
        if not evidence_refs:
            raise ValueError("IForward random-window request requires non-empty evidence refs")
        if not target_refs:
            raise ValueError("IForward random-window request requires non-empty target refs")
        if len(target_refs) != len(target_roles):
            raise ValueError("IForward random-window target refs/roles length mismatch after dedupe")
        for ref in evidence_refs + target_refs:
            self.validate_image_ref(int(scene_id), int(segment_id), tuple(ref), purpose="train")

        bundle = self._resolve_segment_bundle(int(scene_id), int(segment_id))
        dynamic_points = bundle.pointcloud.get("dynamic") if isinstance(bundle.pointcloud, dict) else None
        if isinstance(dynamic_points, dict) and len(dynamic_points) > 0:
            if not bool(self._knn_requirements.fixed_neighbor_enabled):
                raise ValueError(
                    "IForward random-window state carry requires stable full-segment rigid row-space. "
                    "Enable fixed cached KNN / fixed_neighbor_enabled."
                )

        batch = self._assemble_segment_batch_from_image_refs(
            int(scene_id),
            int(segment_id),
            evidence_refs,
            target_refs,
            aux_image_refs=None,
            query_label_image_refs=None,
            include_test=bool(include_test),
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )

        source_ref_to_index: Dict[ImageRef, int] = {tuple(ref): int(idx) for idx, ref in enumerate(evidence_refs)}
        target_ref_to_index: Dict[ImageRef, int] = {tuple(ref): int(idx) for idx, ref in enumerate(target_refs)}
        target_indices_by_role: Dict[str, List[int]] = {}
        target_refs_by_role: Dict[str, List[ImageRef]] = {}
        for idx, (ref, role) in enumerate(zip(target_refs, target_roles)):
            role_s = str(role)
            target_indices_by_role.setdefault(role_s, []).append(int(idx))
            target_refs_by_role.setdefault(role_s, []).append(tuple(ref))

        plan_dict = dataclasses.asdict(plan)
        steps_meta: List[Dict[str, Any]] = []
        for step in list(getattr(plan, "steps", []) or []):
            step_dict = dataclasses.asdict(step)
            step_refs = [(int(ref[0]), int(ref[1])) for ref in list(getattr(step, "evidence_refs", []) or [])]
            step_dict["evidence_refs"] = [tuple(x) for x in step_refs]
            step_dict["source_indices"] = [int(source_ref_to_index[tuple(ref)]) for ref in step_refs]
            steps_meta.append(step_dict)

        role_groups: List[Dict[str, Any]] = [
            {
                "role": "evidence_input",
                "refs": [tuple(x) for x in evidence_refs],
                "image_roles": ["evidence_input" for _ in evidence_refs],
                "allow_update_evidence": True,
                "allow_render_loss": False,
                "allow_memory_write": True,
                "mask_policy": str((getattr(plan, "request_meta", None) or {}).get("evidence_mask_policy", "non_sky_non_egocar")),
            }
        ]
        for role, refs in target_refs_by_role.items():
            role_groups.append(
                {
                    "role": str(role),
                    "refs": [tuple(x) for x in refs],
                    "image_roles": [str(role) for _ in refs],
                    "allow_update_evidence": False,
                    "allow_render_loss": True,
                    "allow_memory_write": False,
                    "mask_policy": "non_sky_non_egocar",
                }
            )

        iforward_meta = {
            "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
            "model_family": RANDOM_WINDOW_MODEL_FAMILY,
            "scene_id": int(getattr(plan, "scene_id")),
            "segment_id": int(getattr(plan, "segment_id")),
            "episode_id": int(getattr(plan, "episode_id")),
            "rollout_id_global": int(getattr(plan, "rollout_id_global")),
            "rollout_idx_in_episode": int(getattr(plan, "rollout_idx_in_episode")),
            "rollouts_per_episode": int(getattr(plan, "rollouts_per_episode")),
            "window_start": int(getattr(plan, "window_start")),
            "window_end": int(getattr(plan, "window_end")),
            "window_block_ids": [int(x) for x in list(getattr(plan, "window_block_ids", []) or [])],
            "window_keyframe_indices": [int(x) for x in list(getattr(plan, "window_keyframe_indices", []) or [])],
            "window_frame_indices": [int(x) for x in list(getattr(plan, "window_frame_indices", []) or [])],
            "window_hash": int(getattr(plan, "window_hash")),
            "window_revisit_count": int(getattr(plan, "window_revisit_count")),
            "unique_windows_seen": int(getattr(plan, "unique_windows_seen")),
            "is_repeated_window": bool(getattr(plan, "is_repeated_window")),
            "blocks_per_rollout": int(getattr(plan, "blocks_per_rollout")),
            "repeats_per_block": int(getattr(plan, "repeats_per_block")),
            "inner_K": int(getattr(plan, "inner_K")),
            "input_frame_indices": [int(x) for x in list(getattr(plan, "input_frame_indices", []) or [])],
            "input_keyframe_indices": [int(x) for x in list(getattr(plan, "input_keyframe_indices", []) or [])],
            "nearby_frame_indices": [int(x) for x in list(getattr(plan, "nearby_frame_indices", []) or [])],
            "evidence_refs_flat": [tuple(x) for x in evidence_refs],
            "target_refs_flat": [tuple(x) for x in target_refs],
            "target_roles_flat": [str(x) for x in target_roles],
            "current_latest_refs": [tuple(x) for x in list(getattr(plan, "current_latest_refs", []) or [])],
            "in_rollout_history_refs": [tuple(x) for x in list(getattr(plan, "in_rollout_history_refs", []) or [])],
            "short_window_history_refs": [tuple(x) for x in list(getattr(plan, "short_window_history_refs", []) or [])],
            "nearby_refs": [tuple(x) for x in list(getattr(plan, "nearby_refs", []) or [])],
            "steps": steps_meta,
            "source_ref_to_index_keyed": {
                self._iforward_ref_key(ref): int(idx)
                for ref, idx in source_ref_to_index.items()
            },
            "target_ref_to_index_keyed": {
                self._iforward_ref_key(ref): int(idx)
                for ref, idx in target_ref_to_index.items()
            },
            "final_supervision": {
                "target_indices_by_role": {
                    str(role): [int(x) for x in indices]
                    for role, indices in target_indices_by_role.items()
                },
                "target_refs_by_role": {
                    str(role): [tuple(x) for x in refs]
                    for role, refs in target_refs_by_role.items()
                },
            },
            "reset_scene_state_before_rollout": bool(getattr(plan, "reset_scene_state_before_rollout")),
            "carry_scene_state_after_rollout": bool(getattr(plan, "carry_scene_state_after_rollout")),
            "episode_end_after_rollout": bool(getattr(plan, "episode_end_after_rollout")),
            "discard_scene_state_after_rollout": bool(getattr(plan, "episode_end_after_rollout")),
            "detach_graph_after_rollout": bool(getattr(plan, "detach_graph_after_rollout")),
            "leakage_check": dict(getattr(plan, "leakage_check", {}) or {}),
        }

        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(dict(getattr(plan, "request_meta", {}) or {}))
        request_meta["scheduler_version"] = RANDOM_WINDOW_SCHEDULER_VERSION
        request_meta["model_family"] = RANDOM_WINDOW_MODEL_FAMILY
        request_meta["scene_id"] = int(getattr(plan, "scene_id"))
        request_meta["segment_id"] = int(getattr(plan, "segment_id"))
        request_meta["episode_id"] = int(getattr(plan, "episode_id"))
        request_meta["episode_idx_global"] = int(getattr(plan, "episode_id"))
        request_meta["rollout_id_global"] = int(getattr(plan, "rollout_id_global"))
        request_meta["rollout_idx_in_episode"] = int(getattr(plan, "rollout_idx_in_episode"))
        request_meta["rollouts_per_episode"] = int(getattr(plan, "rollouts_per_episode"))
        request_meta["inner_K"] = int(getattr(plan, "inner_K"))
        request_meta["source_image_refs"] = [tuple(x) for x in evidence_refs]
        request_meta["source_image_ref"] = tuple(evidence_refs[0])
        request_meta["target_image_refs"] = [tuple(x) for x in target_refs]
        request_meta["target_image_roles"] = [str(x) for x in target_roles]
        request_meta["role_groups"] = role_groups
        request_meta["iforward"] = iforward_meta
        request_meta["assembly_mode"] = RANDOM_WINDOW_ASSEMBLY_MODE
        batch["request_meta"] = request_meta
        batch["_iforward"] = iforward_meta
        batch["_iforward_plan"] = plan_dict
        batch["_iforward_runtime_maps"] = {
            "source_ref_to_index": dict(source_ref_to_index),
            "target_ref_to_index": dict(target_ref_to_index),
        }
        return batch

    def _assemble_segment_batch_from_v9_request(
        self,
        *,
        scene_id: int,
        segment_id: int,
        v9_plan: ViewSetRolloutBatchV9,
        include_test: bool = False,
    ) -> Dict[str, Any]:
        if str(v9_plan.scheduler_version) != "v9":
            raise ValueError("expected ViewSetRolloutBatchV9.scheduler_version == 'v9'")
        if int(scene_id) != int(v9_plan.scene_id) or int(segment_id) != int(v9_plan.segment_id):
            raise ValueError(
                "V9 request scene/segment mismatch: "
                f"request=({int(scene_id)},{int(segment_id)}) "
                f"plan=({int(v9_plan.scene_id)},{int(v9_plan.segment_id)})"
            )

        evidence_refs = self._dedupe_v9_refs_keep_order(
            self._flatten_v9_ref_groups(v9_plan.evidence_refs_by_step)
        )
        block_refs = self._flatten_v9_ref_groups(v9_plan.block_loss_refs_by_step)
        nearby_refs = self._flatten_v9_ref_groups(v9_plan.nearby_loss_refs_by_step)
        prefix_refs = self._flatten_v9_ref_groups(v9_plan.prefix_loss_refs_by_step)
        raw_loss_refs = block_refs + nearby_refs + prefix_refs
        raw_loss_roles = (
            ["block_loss" for _ in block_refs]
            + ["nearby_loss" for _ in nearby_refs]
            + ["prefix_loss" for _ in prefix_refs]
        )
        final_meta = dict((v9_plan.request_meta or {}).get("phase_b_final_supervision") or {})
        if str((v9_plan.request_meta or {}).get("phase_b_loss_timing", "")) == "rollout_final_only":
            final_refs = [(int(ref[0]), int(ref[1])) for ref in list(final_meta.get("refs", []) or [])]
            final_roles = [str(x) for x in list(final_meta.get("roles", []) or [])]
            if len(final_refs) != len(final_roles) or not final_refs:
                raise ValueError("V9 final rollout request requires phase_b_final_supervision refs/roles")
            self._assert_no_v9_ref_role_conflicts(final_refs, final_roles)
            raw_loss_refs = [tuple(x) for x in final_refs]
            raw_loss_roles = [str(x) for x in final_roles]
        loss_refs, loss_roles = self._dedupe_v9_refs_roles_keep_order(raw_loss_refs, raw_loss_roles)
        query_refs = self._dedupe_v9_refs_keep_order(v9_plan.query_label_refs)
        aux_refs = self._dedupe_v9_refs_keep_order(v9_plan.aux_loss_refs)
        if len(evidence_refs) == 0:
            raise ValueError("V9 request requires non-empty evidence refs")
        if len(loss_refs) == 0:
            raise ValueError("V9 request requires non-empty render loss refs for batch['target']")
        for ref in evidence_refs + loss_refs + query_refs + aux_refs:
            self.validate_image_ref(int(scene_id), int(segment_id), tuple(ref), purpose="train")

        batch = self._assemble_segment_batch_from_image_refs(
            int(scene_id),
            int(segment_id),
            evidence_refs,
            loss_refs,
            aux_image_refs=aux_refs,
            query_label_image_refs=query_refs,
            include_test=bool(include_test),
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )
        plan_dict = dataclasses.asdict(v9_plan)
        request_meta = dict(batch.get("request_meta") or {})
        request_meta.update(dict(v9_plan.request_meta or {}))
        request_meta["source_image_refs"] = [tuple(x) for x in evidence_refs]
        request_meta["source_image_ref"] = tuple(evidence_refs[0])
        request_meta["target_image_refs"] = [tuple(x) for x in loss_refs]
        request_meta["target_image_roles"] = [str(x) for x in loss_roles]
        request_meta["query_label_refs"] = [tuple(x) for x in query_refs]
        request_meta["aux_image_refs"] = [tuple(x) for x in aux_refs]
        request_meta["assembly_mode"] = "image_ref_v9"
        if len(request_meta["target_image_refs"]) != len(request_meta["target_image_roles"]):
            raise ValueError(
                "V9 target_image_refs/target_image_roles mismatch after assembly: "
                f"{len(request_meta['target_image_refs'])} vs {len(request_meta['target_image_roles'])}"
            )
        batch["request_meta"] = request_meta
        batch["_scheduler_v9"] = plan_dict
        if str(v9_plan.phase) == "phase_A_block_local_unroll":
            batch["rollout_plan"] = convert_v9_phase_a_plan(v9_plan)
        return batch

    def get_segment_eval_batch_from_image_refs(self, request: EvalRequestV4) -> Dict[str, Any]:
        raw = self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            [request.source_image_ref],
            request.eval_image_refs,
            aux_image_refs=None,
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

    def build_preload_hint_light(
        self,
        *,
        scene_id: int,
        segment_id: int,
        future_image_refs: Sequence[ImageRef],
        scope: str = "next_block_exact",
    ) -> Dict[str, Any]:
        refs = [(int(x[0]), int(x[1])) for x in future_image_refs]
        unique_frames = sorted({int(x[0]) for x in refs})
        unique_cams = sorted({int(x[1]) for x in refs})
        return {
            "hint_version": 4,
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "segment_asset_id": "",
            "future_image_refs": refs,
            "unique_frame_indices": unique_frames,
            "unique_cam_indices": unique_cams,
            "required_static": {
                "segment_bundle": True,
                "test_refs": False,
            },
            "scope": str(scope),
            "lightweight": True,
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
        block_order: str = "block_major",
        step_major_switch_interval_steps: int = 1,
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
            block_order=str(block_order),
            step_major_switch_interval_steps=int(step_major_switch_interval_steps),
        )

    def create_train_scheduler_v8(
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
        block_order: str = "block_major",
        step_major_switch_interval_steps: int = 1,
        target_policy: str = "visited_episode_frames",
        history_target_policy: str = "nearest_visited",
        reset_policy: str = "episode_end",
        near_random_supervision_cfg: Optional[Any] = None,
        aux_feature_splat_targets_cfg: Optional[Any] = None,
        block_source_frame_policy: str = "fixed_once_per_episode",
        episode_source_mode: str = "keyframes",
    ) -> TrainSchedulerV8:
        return TrainSchedulerV8(
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
            block_order=str(block_order),
            step_major_switch_interval_steps=int(step_major_switch_interval_steps),
            target_policy=str(target_policy),
            history_target_policy=str(history_target_policy),
            reset_policy=str(reset_policy),
            near_random_supervision_cfg=near_random_supervision_cfg,
            aux_feature_splat_targets_cfg=aux_feature_splat_targets_cfg,
            block_source_frame_policy=str(block_source_frame_policy),
            episode_source_mode=str(episode_source_mode),
        )

    def create_train_scheduler_long_phase_b(
        self,
        *,
        episode_window_cfg: Optional[Any] = None,
        rollout_shapes: Optional[Sequence[Any]] = None,
        rollout_shapes_schedule: Optional[Sequence[Any]] = None,
        anchor_sampling_cfg: Optional[Any] = None,
        traversal_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
        evidence_cfg: Optional[Any] = None,
        final_supervision_cfg: Optional[Any] = None,
        rigid_meta_cfg: Optional[Any] = None,
        distant_meta_cfg: Optional[Any] = None,
        fail_fast: bool = True,
    ) -> TrainSchedulerLongPhaseB:
        from datasets.train_scheduler_long_phase_b import TrainSchedulerLongPhaseB

        return TrainSchedulerLongPhaseB(
            dataset=self,
            episode_window_cfg=episode_window_cfg,
            rollout_shapes=rollout_shapes,
            rollout_shapes_schedule=rollout_shapes_schedule,
            anchor_sampling_cfg=anchor_sampling_cfg,
            traversal_cfg=traversal_cfg,
            preload_cfg=preload_cfg,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            evidence_cfg=evidence_cfg,
            final_supervision_cfg=final_supervision_cfg,
            rigid_meta_cfg=rigid_meta_cfg,
            distant_meta_cfg=distant_meta_cfg,
            fail_fast=bool(fail_fast),
        )

    def create_train_scheduler_iforward(
        self,
        *,
        episode_cfg: Optional[Any] = None,
        rollout_cfg: Optional[Any] = None,
        traversal_cfg: Optional[Any] = None,
        evidence_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        memory_cfg: Optional[Any] = None,
        loss_timing_cfg: Optional[Any] = None,
        leakage_check_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        version: str = "iforward_v1",
        fail_fast: bool = True,
    ) -> Any:
        from datasets.train_scheduler_iforward import TrainSchedulerIForward

        return TrainSchedulerIForward(
            dataset=self,
            episode_cfg=episode_cfg,
            rollout_cfg=rollout_cfg,
            traversal_cfg=traversal_cfg,
            evidence_cfg=evidence_cfg,
            supervision_cfg=supervision_cfg,
            memory_cfg=memory_cfg,
            loss_timing_cfg=loss_timing_cfg,
            leakage_check_cfg=leakage_check_cfg,
            preload_cfg=preload_cfg,
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            seed=seed,
            version=str(version),
            fail_fast=bool(fail_fast),
        )

    def create_train_scheduler_iforward_random_window(
        self,
        *,
        traversal_cfg: Optional[Any] = None,
        segment_cfg: Optional[Any] = None,
        episode_cfg: Optional[Any] = None,
        rollout_cfg: Optional[Any] = None,
        evidence_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        memory_cfg: Optional[Any] = None,
        loss_timing_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        fail_fast: bool = True,
        fixed_window_starts: Optional[Sequence[int]] = None,
    ) -> Any:
        from datasets.iforward_random_window_scheduler import IForwardRandomWindowScheduler

        return IForwardRandomWindowScheduler(
            dataset=self,
            traversal_cfg=traversal_cfg,
            segment_cfg=segment_cfg,
            episode_cfg=episode_cfg,
            rollout_cfg=rollout_cfg,
            evidence_cfg=evidence_cfg,
            supervision_cfg=supervision_cfg,
            memory_cfg=memory_cfg,
            loss_timing_cfg=loss_timing_cfg,
            preload_cfg=preload_cfg,
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            seed=seed,
            fail_fast=bool(fail_fast),
            fixed_window_starts=fixed_window_starts,
        )

    def create_train_scheduler_v9(
        self,
        *,
        phase: str,
        steps_per_block: int,
        blocks_per_episode: int,
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
        warm_v9_role_refs: bool = True,
        block_order: str = "block_major",
        step_major_switch_interval_steps: int = 1,
        target_policy: str = "visited_episode_frames",
        reset_policy: str = "episode_end",
        block_source_frame_policy: str = "random_within_keyframe_per_visit",
        episode_source_mode: str = "keyframes",
        phase_a_cfg: Optional[Any] = None,
        phase_b_cfg: Optional[Any] = None,
        leakage_check_cfg: Optional[Any] = None,
        fail_fast: bool = True,
    ) -> TrainSchedulerV9:
        return TrainSchedulerV9(
            dataset=self,
            phase=phase,  # type: ignore[arg-type]
            steps_per_block=int(steps_per_block),
            blocks_per_episode=int(blocks_per_episode),
            include_source_frame=bool(include_source_frame),
            frame_within_keyframe_policy=str(frame_within_keyframe_policy),
            min_keyframes_required_policy=str(min_keyframes_required_policy),
            traversal_mode=str(traversal_mode),
            switch_after_episode=bool(switch_after_episode),
            segment_order=str(segment_order),
            scene_order=str(scene_order),
            include_test=bool(include_test),
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
            emit_preload_hints=bool(emit_preload_hints),
            warm_next_block_exact=bool(warm_next_block_exact),
            warm_next_episode_chain=bool(warm_next_episode_chain),
            warm_v9_role_refs=bool(warm_v9_role_refs),
            block_order=str(block_order),
            step_major_switch_interval_steps=int(step_major_switch_interval_steps),
            target_policy=str(target_policy),
            reset_policy=str(reset_policy),
            block_source_frame_policy=str(block_source_frame_policy),
            episode_source_mode=str(episode_source_mode),
            phase_a_cfg=phase_a_cfg,
            phase_b_cfg=phase_b_cfg,
            leakage_check_cfg=leakage_check_cfg,
            fail_fast=bool(fail_fast),
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
        self._get_cached_or_load_view(scene_id, segment_id, image_ref, materialize=False)
