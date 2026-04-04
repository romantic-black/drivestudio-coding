"""
MultiSceneDatasetV3: image-ref (frame_idx, cam_id) driven batch assembly for StreetForward.

Canonical API: get_segment_batch_from_image_refs(BatchRequestV3).

Legacy frame API: get_segment_batch_from_frames expands each training frame to all cameras
(same tensor layout as scheduler v2), but batch keyframe_indices are **image-level aligned**
(len == number of source/target images), not one entry per selected keyframe like
get_segment_batch(). Do not assume len(keyframe_indices) == num_source_keyframes.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

from datasets.dataset_preload_manager import (
    PRIORITY_EPISODE_SUPERSET,
    PRIORITY_NEXT_BLOCK_EXACT,
    PRIORITY_TEST_REFS,
    DatasetPreloadManager,
    LoadedViewPack,
    dict_to_loaded_view_pack,
    loaded_view_pack_to_device,
    parse_preload_cfg,
    pin_memory_from_cfg,
)
from datasets.driving_dataset import DrivingDataset
from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)

ImageRef = Tuple[int, int]


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


@dataclass(frozen=True)
class SegmentIndex:
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


@dataclass(frozen=True)
class BatchRequestV3:
    scene_id: int
    segment_id: int
    source_image_ref: ImageRef
    target_image_refs: List[ImageRef]
    include_test: bool = False
    test_image_refs: Optional[List[ImageRef]] = None


def _build_segment_index_dict(
    scene_id: int,
    segment_id: int,
    scene_data: Dict[str, Any],
) -> SegmentIndex:
    segment = scene_data["segments"][segment_id]
    scene_dataset = scene_data["dataset"]
    num_cams = int(scene_dataset.num_cams)
    keyframe_segments: List[List[int]] = scene_data["keyframe_segments"]
    seg_keyframes: List[int] = list(segment.get("keyframe_indices", []))
    train_frames = sorted(set(segment.get("frame_indices", [])))
    if not train_frames:
        raise ValueError(f"Segment has no frame_indices (scene={scene_id} segment={segment_id})")
    test_frames = list(segment.get("test_frame_indices", []))
    train_frame_set = frozenset(train_frames)
    test_frame_set = frozenset(int(x) for x in test_frames)

    keyframe_to_frames: Dict[int, List[int]] = {}
    for kf_idx in seg_keyframes:
        if kf_idx < 0 or kf_idx >= len(keyframe_segments):
            raise ValueError(
                f"Invalid keyframe index {kf_idx} in segment keyframe_indices "
                f"(scene={scene_id} segment={segment_id}, len(keyframe_segments)={len(keyframe_segments)})"
            )
        frames_in_kf = sorted(f for f in keyframe_segments[kf_idx] if f in train_frame_set)
        if len(frames_in_kf) == 0:
            raise ValueError(
                f"Keyframe {kf_idx} has no train frames overlapping segment frame_indices "
                f"(scene={scene_id} segment={segment_id})"
            )
        keyframe_to_frames[int(kf_idx)] = frames_in_kf

    frame_to_keyframe: Dict[int, int] = {}
    for kf_idx, frames in keyframe_to_frames.items():
        for f in frames:
            if f in frame_to_keyframe:
                raise ValueError(
                    f"Frame {f} belongs to multiple keyframes in scene={scene_id} segment={segment_id}"
                )
            frame_to_keyframe[int(f)] = int(kf_idx)

    for f in train_frames:
        if f not in frame_to_keyframe:
            raise ValueError(
                f"Frame {f} in segment frame_indices has no keyframe mapping "
                f"(scene={scene_id} segment={segment_id})"
            )

    first_frame = train_frames[0]
    return SegmentIndex(
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        num_cams=num_cams,
        frame_indices=train_frames,
        test_frame_indices=test_frames,
        train_frame_set=train_frame_set,
        test_frame_set=test_frame_set,
        keyframe_indices=list(seg_keyframes),
        keyframe_to_frames=keyframe_to_frames,
        frame_to_keyframe=frame_to_keyframe,
        segment_first_frame_idx=int(first_frame),
    )


class MultiSceneDatasetV3(MultiSceneDataset):
    """
    Extends MultiSceneDataset with SegmentIndex cache and image-ref batch assembly.
    """

    def __init__(self, *args: Any, preload_cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._segment_index_cache: Dict[Tuple[int, int], SegmentIndex] = {}
        self._pair_score_cache: Dict[Tuple[int, int, Tuple[int, int], Tuple[int, int], str], float] = {}
        self._pixel_source_io_lock = threading.RLock()
        self._preload_rtcfg = parse_preload_cfg(preload_cfg)
        self._preload_manager: Optional[DatasetPreloadManager] = None
        if self._preload_rtcfg is not None:
            self._preload_manager = DatasetPreloadManager(self, self._preload_rtcfg)
        self._view_pack_cache: "OrderedDict[Tuple[int, int, int, int], LoadedViewPack]" = OrderedDict()
        self._view_pack_lock = threading.RLock()
        self._preload_active_scene_id: Optional[int] = None
        self._preload_active_segment_id: Optional[int] = None
        self._preload_training_scene_id: Optional[int] = None
        self._preload_training_segment_id: Optional[int] = None
        self._scene_preload_inflight: Dict[int, int] = {}
        self._scene_preload_inflight_lock = threading.Lock()
        self._scene_unloading: Set[int] = set()
        self._scene_unloading_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self.shutdown_preload()
        except Exception:
            pass

    def initialize(self) -> None:
        super().initialize()
        if self._preload_manager is not None:
            self._preload_manager.start()

    def shutdown_preload(self) -> None:
        if self._preload_manager is not None:
            self._preload_manager.stop()

    def set_preload_active_scope(self, scene_id: int, segment_id: int) -> None:
        """Scheduler-declared (scene, segment) for preload stale checks; call before emit/submit preload hints."""
        self._preload_active_scene_id = int(scene_id)
        self._preload_active_segment_id = int(segment_id)

    def _scene_cache_protected_train_scene_ids_unlocked(self) -> Set[int]:
        """Scenes TrainSchedulerV4 + batch assembly care about; do not prefer evicting these (see _ensure_scene_loaded)."""
        out: Set[int] = set()
        if self._preload_training_scene_id is not None:
            out.add(int(self._preload_training_scene_id))
        if self._preload_active_scene_id is not None:
            out.add(int(self._preload_active_scene_id))
        return out

    def _ensure_scene_loaded(self, scene_id: int) -> Optional[Dict]:
        """
        Like MultiSceneDataset._ensure_scene_loaded, but cache eviction protects V4 preload/training
        scope instead of legacy scene_training_queue / current_scene_index.

        Eviction calls ``_unload_scene`` without holding ``self._lock``: V3 unload waits on preload
        workers, which may need the lock inside this same method.
        """
        with self._lock:
            if scene_id in self.train_scenes_cache:
                return self.train_scenes_cache[scene_id]
            if scene_id in self.eval_scene_ids and scene_id in self.eval_scenes:
                return self.eval_scenes[scene_id]

        max_cache_size = self.preload_scene_count + 1
        while True:
            with self._lock:
                if scene_id in self.train_scenes_cache:
                    return self.train_scenes_cache[scene_id]
                if scene_id in self.eval_scene_ids and scene_id in self.eval_scenes:
                    return self.eval_scenes[scene_id]
                if len(self.train_scenes_cache) < max_cache_size:
                    break
                protected = self._scene_cache_protected_train_scene_ids_unlocked()
                victim: Optional[int] = None
                for cached_scene_id in list(self.train_scenes_cache.keys()):
                    if cached_scene_id not in protected:
                        victim = cached_scene_id
                        break
                if victim is None and self.train_scenes_cache:
                    victim = next(iter(self.train_scenes_cache.keys()))
            if victim is None:
                break
            self._unload_scene(victim)

        if scene_id in self.eval_scene_ids and scene_id not in self.eval_scenes:
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                with self._lock:
                    self.eval_scenes[scene_id] = scene_data
                return scene_data
            return None

        scene_data = self._load_and_prepare_scene(scene_id)
        if scene_data is not None:
            with self._lock:
                self.train_scenes_cache[scene_id] = scene_data
            return scene_data
        return None

    def _preload_should_abort_for_unload(self, scene_id: int) -> bool:
        sid = int(scene_id)
        with self._scene_unloading_lock:
            return sid in self._scene_unloading

    def _preload_begin_scene_work(self, scene_id: int) -> None:
        sid = int(scene_id)
        with self._scene_preload_inflight_lock:
            self._scene_preload_inflight[sid] = self._scene_preload_inflight.get(sid, 0) + 1

    def _preload_end_scene_work(self, scene_id: int) -> None:
        sid = int(scene_id)
        with self._scene_preload_inflight_lock:
            c = self._scene_preload_inflight.get(sid, 0) - 1
            if c <= 0:
                self._scene_preload_inflight.pop(sid, None)
            else:
                self._scene_preload_inflight[sid] = c

    def _wait_scene_preload_quiescent(self, scene_id: int, timeout: float = 120.0) -> None:
        sid = int(scene_id)
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._scene_preload_inflight_lock:
                if self._scene_preload_inflight.get(sid, 0) == 0:
                    return
            time.sleep(0.005)
        raise RuntimeError(
            f"Timeout waiting for preload workers to finish scene_id={sid} (inflight preload I/O)"
        )

    def _preload_view_key_is_cached(self, key: Tuple[int, int, int, int]) -> bool:
        if self._preload_rtcfg is None or not self._preload_rtcfg.enable_view_pack_cache:
            return False
        with self._view_pack_lock:
            return key in self._view_pack_cache

    def _unload_scene(self, scene_id: int) -> None:
        sid = int(scene_id)
        with self._scene_unloading_lock:
            self._scene_unloading.add(sid)
        try:
            if self._preload_manager is not None:
                self._preload_manager.clear_pending_for_scene(sid)
            self._wait_scene_preload_quiescent(sid)
            super()._unload_scene(sid)
            with self._view_pack_lock:
                stale_view_keys = [k for k in self._view_pack_cache if k[0] == sid]
                for k in stale_view_keys:
                    del self._view_pack_cache[k]
            stale_index_keys = [k for k in self._segment_index_cache if k[0] == scene_id]
            for k in stale_index_keys:
                del self._segment_index_cache[k]
            stale_pair_keys = [k for k in self._pair_score_cache if k[0] == scene_id]
            for k in stale_pair_keys:
                del self._pair_score_cache[k]
        finally:
            with self._scene_unloading_lock:
                self._scene_unloading.discard(sid)

    def _pick_view_cache_eviction_victim(
        self,
        cur_sc: Optional[int],
        cur_seg: Optional[int],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not self._view_pack_cache:
            return None
        for k in self._view_pack_cache.keys():
            if cur_sc is not None and int(k[0]) != int(cur_sc):
                return k
        for k in self._view_pack_cache.keys():
            if (
                cur_sc is not None
                and cur_seg is not None
                and int(k[0]) == int(cur_sc)
                and int(k[1]) != int(cur_seg)
            ):
                return k
        return next(iter(self._view_pack_cache))

    def _evict_view_cache_if_needed_unlocked(self, new_key: Tuple[int, int, int, int]) -> None:
        cfg = self._preload_rtcfg
        if cfg is None or not cfg.enable_view_pack_cache:
            return
        cur_sc = self._preload_training_scene_id
        cur_seg = self._preload_training_segment_id
        max_total = int(cfg.view_cache_max_items_total)
        while len(self._view_pack_cache) >= max_total:
            vk = self._pick_view_cache_eviction_victim(cur_sc, cur_seg)
            if vk is None:
                break
            del self._view_pack_cache[vk]

    def _trim_view_cache_per_scene_cap_unlocked(self, scene_id: int) -> None:
        cfg = self._preload_rtcfg
        if cfg is None or not cfg.enable_view_pack_cache:
            return
        max_per_scene = int(cfg.view_cache_max_items_per_scene)
        sid = int(scene_id)
        while sum(1 for k in self._view_pack_cache if int(k[0]) == sid) > max_per_scene:
            drop: Optional[Tuple[int, int, int, int]] = None
            for k in self._view_pack_cache.keys():
                if int(k[0]) == sid:
                    drop = k
                    break
            if drop is None:
                break
            del self._view_pack_cache[drop]

    def _preload_worker_load_view_pack(
        self,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
    ) -> str:
        cfg = self._preload_rtcfg
        if cfg is None or not cfg.enable_view_pack_cache:
            return "skipped"
        key = (int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
        with self._view_pack_lock:
            if key in self._view_pack_cache:
                return "cache_hit"
        try:
            if self._preload_should_abort_for_unload(int(scene_id)):
                return "failed"
            scene_data = self._ensure_scene_loaded(int(scene_id))
            if scene_data is None:
                return "failed"
            scene_dataset = scene_data["dataset"]
            raw = self._load_view_from_image_ref(scene_dataset, tuple(image_ref))
            pin = pin_memory_from_cfg(cfg)
            lvp = dict_to_loaded_view_pack(raw, pin_memory=pin)
            with self._view_pack_lock:
                if key in self._view_pack_cache:
                    return "cache_hit"
                self._evict_view_cache_if_needed_unlocked(key)
                self._view_pack_cache[key] = lvp
                self._view_pack_cache.move_to_end(key)
                self._trim_view_cache_per_scene_cap_unlocked(key[0])
            return "loaded"
        except Exception as exc:
            logger.debug("preload worker _preload_worker_load_view_pack: %s", exc, exc_info=True)
            return "failed"

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
        if self._preload_manager is None or self._preload_rtcfg is None:
            return
        scene_id = int(hint["scene_id"])
        segment_id = int(hint["segment_id"])
        refs: List[ImageRef] = [(int(r[0]), int(r[1])) for r in hint["future_image_refs"]]
        if hint_scope == "next_block_exact":
            if not self._preload_rtcfg.warm_next_block_exact:
                return
            for ref in refs:
                self._preload_manager.submit_image_ref(
                    PRIORITY_NEXT_BLOCK_EXACT, scene_id, segment_id, ref
                )
            if include_test and self._preload_rtcfg.warm_test_refs:
                for ref in self.resolve_test_image_refs_deterministic(scene_id, segment_id):
                    self._preload_manager.submit_image_ref(
                        PRIORITY_TEST_REFS, scene_id, segment_id, ref
                    )
        elif hint_scope == "episode_source_superset":
            if not self._preload_rtcfg.warm_episode_source_superset:
                return
            for ref in refs:
                self._preload_manager.submit_image_ref(
                    PRIORITY_EPISODE_SUPERSET, scene_id, segment_id, ref
                )
        else:
            return

    def _get_cached_or_load_view_from_image_ref(
        self,
        scene_id: int,
        segment_id: int,
        scene_dataset: DrivingDataset,
        image_ref: ImageRef,
    ) -> Dict[str, Any]:
        ref_t = (int(image_ref[0]), int(image_ref[1]))
        key = (int(scene_id), int(segment_id), ref_t[0], ref_t[1])
        if self._preload_rtcfg is None or not self._preload_rtcfg.enable_view_pack_cache:
            return self._load_view_from_image_ref(scene_dataset, ref_t)

        with self._view_pack_lock:
            if key in self._view_pack_cache:
                self._view_pack_cache.move_to_end(key)
                return loaded_view_pack_to_device(self._view_pack_cache[key], self.device)

        raw = self._load_view_from_image_ref(scene_dataset, ref_t)
        pin = pin_memory_from_cfg(self._preload_rtcfg)
        lvp = dict_to_loaded_view_pack(raw, pin_memory=pin)
        with self._view_pack_lock:
            if key in self._view_pack_cache:
                self._view_pack_cache.move_to_end(key)
                return loaded_view_pack_to_device(self._view_pack_cache[key], self.device)
            self._evict_view_cache_if_needed_unlocked(key)
            self._view_pack_cache[key] = lvp
            self._view_pack_cache.move_to_end(key)
            self._trim_view_cache_per_scene_cap_unlocked(key[0])
            return loaded_view_pack_to_device(lvp, self.device)

    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndex:
        key = (int(scene_id), int(segment_id))
        if key in self._segment_index_cache:
            return self._segment_index_cache[key]
        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segments = scene_data.get("segments", [])
        if int(segment_id) < 0 or int(segment_id) >= len(segments):
            raise ValueError(f"segment_id={segment_id} out of range for scene {scene_id}")
        idx = _build_segment_index_dict(int(scene_id), int(segment_id), scene_data)
        self._segment_index_cache[key] = idx
        return idx

    def validate_image_ref(
        self,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
        *,
        purpose: Literal["train", "test"] = "train",
    ) -> None:
        frame_idx, cam_id = int(image_ref[0]), int(image_ref[1])
        sidx = self.get_segment_index(scene_id, segment_id)
        if cam_id < 0 or cam_id >= sidx.num_cams:
            raise ValueError(
                f"cam_id={cam_id} out of range for num_cams={sidx.num_cams} "
                f"(scene={scene_id} segment={segment_id})"
            )
        if purpose == "train":
            if frame_idx not in sidx.train_frame_set:
                raise ValueError(
                    f"frame_idx={frame_idx} not in segment train frame_indices "
                    f"(scene={scene_id} segment={segment_id})"
                )
        else:
            if frame_idx not in sidx.test_frame_set:
                raise ValueError(
                    f"frame_idx={frame_idx} not in segment test_frame_indices "
                    f"(scene={scene_id} segment={segment_id})"
                )

    def get_or_compute_pair_score(
        self,
        scene_id: int,
        segment_id: int,
        src: ImageRef,
        tgt: ImageRef,
        mode: str = "none",
    ) -> Optional[float]:
        if mode == "none":
            return None
        raise NotImplementedError(
            f"get_or_compute_pair_score with mode={mode!r} is not implemented in MultiSceneDatasetV3 v1"
        )

    def build_preload_hint(
        self,
        scene_id: int,
        segment_id: int,
        future_image_refs: List[ImageRef],
    ) -> Dict[str, Any]:
        frames = sorted({int(r[0]) for r in future_image_refs})
        cams = sorted({int(r[1]) for r in future_image_refs})
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "future_image_refs": list(future_image_refs),
            "unique_frame_indices": frames,
            "unique_cam_indices": cams,
            "hint_version": 1,
        }

    def _load_view_from_image_ref(
        self,
        scene_dataset: DrivingDataset,
        image_ref: ImageRef,
    ) -> Dict[str, Any]:
        frame_idx, cam_idx = int(image_ref[0]), int(image_ref[1])
        num_cams = scene_dataset.num_cams
        if cam_idx < 0 or cam_idx >= num_cams:
            raise ValueError(f"cam_idx={cam_idx} out of range (num_cams={num_cams})")
        img_idx = frame_idx * num_cams + cam_idx
        try:
            with self._pixel_source_io_lock:
                image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
                depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
        except Exception as e:
            raise ValueError(f"Failed to load image img_idx={img_idx}: {e}") from e

        if depth is None:
            H, W = image_infos["pixels"].shape[:2]
            depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0

        intrinsic_3x3 = cam_infos["intrinsics"]
        intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)

        sky_mask = image_infos.get("sky_masks")
        if sky_mask is not None:
            sky_mask = self._normalize_sky_mask(sky_mask)

        viewdirs = image_infos.get("viewdirs")
        egocar_mask = image_infos.get("egocar_masks")

        return {
            "image": image_infos["pixels"],
            "extrinsic": cam_infos["camera_to_world"],
            "intrinsic": intrinsic_4x4,
            "depth": depth,
            "sky_mask": sky_mask,
            "viewdirs": viewdirs,
            "egocar_mask": egocar_mask,
            "frame_idx": frame_idx,
            "cam_idx": cam_idx,
        }

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
    ) -> Dict[str, Any]:
        if len(source_image_refs) < 1:
            raise ValueError("source_image_refs must be non-empty")
        if len(target_image_refs) < 1:
            raise ValueError("target_image_refs must be non-empty")

        sidx = self.get_segment_index(scene_id, segment_id)
        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment = scene_data["segments"][int(sidx.segment_id)]
        scene_dataset = scene_data["dataset"]

        for ref in source_image_refs:
            self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="train")
        for ref in target_image_refs:
            self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="train")

        self._preload_training_scene_id = int(scene_id)
        self._preload_training_segment_id = int(segment_id)

        if enforce_target0_equals_source:
            if len(source_image_refs) == 1:
                if tuple(target_image_refs[0]) != tuple(source_image_refs[0]):
                    raise ValueError(
                        "target_image_refs[0] must equal source_image_ref when enforce_target0_equals_source=True "
                        f"(got {target_image_refs[0]} vs {source_image_refs[0]})"
                    )
            else:
                n_src = len(source_image_refs)
                if len(target_image_refs) < n_src:
                    raise ValueError(
                        "target_image_refs must be at least as long as source_image_refs when "
                        "enforce_target0_equals_source=True and multiple source images are used "
                        f"(len(target)={len(target_image_refs)}, len(source)={n_src})"
                    )
                prefix = [tuple(target_image_refs[i]) for i in range(n_src)]
                srcs = [tuple(r) for r in source_image_refs]
                if prefix != srcs:
                    raise ValueError(
                        "For multi-source assembly, target_image_refs must start with the full source_image_refs "
                        f"prefix (expected {srcs}, got target prefix {prefix})"
                    )

        segment_first_pose, segment_first_frame_idx, segment_pose_source = self._get_segment_first_pose(
            scene_dataset=scene_dataset,
            segment=segment,
            segment_id=int(segment_id),
        )
        segment_first_pose = segment_first_pose.to(device=self.device, dtype=torch.float32)
        try:
            world_to_seg0 = torch.linalg.inv(segment_first_pose)
        except RuntimeError as exc:
            raise ValueError(
                f"Segment {segment_id} first pose is non-invertible; cannot build segment coordinate transform."
            ) from exc

        def _transform_extrinsics_list(extrinsics_list: List[Tensor]) -> List[Tensor]:
            transformed: List[Tensor] = []
            for ext in extrinsics_list:
                ext_tensor = self._to_4x4_tensor(ext).to(device=self.device, dtype=torch.float32)
                transformed.append(world_to_seg0 @ ext_tensor)
            return transformed

        def _load_stack_role(
            refs: Sequence[ImageRef],
        ) -> Tuple[
            List[Tensor],
            List[Tensor],
            List[Tensor],
            List[Tensor],
            List[int],
            List[int],
            List[int],
            List[Optional[Tensor]],
            bool,
            List[Optional[Tensor]],
            bool,
            List[Optional[Tensor]],
            bool,
        ]:
            images: List[Tensor] = []
            extrinsics: List[Tensor] = []
            intrinsics: List[Tensor] = []
            depths: List[Tensor] = []
            frame_idxs: List[int] = []
            cam_idxs: List[int] = []
            kf_idxs: List[int] = []
            sky_masks: List[Optional[Tensor]] = []
            has_sky = False
            viewdirs_list: List[Optional[Tensor]] = []
            has_vd = False
            egocar_list: List[Optional[Tensor]] = []
            has_ego = False

            for ref in refs:
                pack = self._get_cached_or_load_view_from_image_ref(
                    scene_id, segment_id, scene_dataset, tuple(ref)
                )
                fidx = int(pack["frame_idx"])
                images.append(pack["image"])
                extrinsics.append(pack["extrinsic"])
                intrinsics.append(pack["intrinsic"])
                depths.append(pack["depth"])
                frame_idxs.append(fidx)
                cam_idxs.append(int(pack["cam_idx"]))
                kf_idxs.append(int(sidx.frame_to_keyframe[fidx]))
                sm = pack.get("sky_mask")
                if sm is not None:
                    has_sky = True
                sky_masks.append(sm)
                vd = pack.get("viewdirs")
                if vd is not None:
                    has_vd = True
                viewdirs_list.append(vd)
                em = pack.get("egocar_mask")
                if em is not None:
                    has_ego = True
                egocar_list.append(em)

            return (
                images,
                extrinsics,
                intrinsics,
                depths,
                frame_idxs,
                cam_idxs,
                kf_idxs,
                sky_masks,
                has_sky,
                viewdirs_list,
                has_vd,
                egocar_list,
                has_ego,
            )

        src_pack = _load_stack_role(source_image_refs)
        tgt_pack = _load_stack_role(target_image_refs)

        source_images = src_pack[0]
        source_extrinsics = _transform_extrinsics_list(src_pack[1])
        source_intrinsics = src_pack[2]
        source_depths = src_pack[3]
        source_frame_idxs = src_pack[4]
        source_cam_idxs = src_pack[5]
        source_kf_idxs = src_pack[6]
        source_sky_masks = src_pack[7]
        has_source_sky_mask = src_pack[8]
        source_viewdirs_list = src_pack[9]
        has_source_viewdirs = src_pack[10]
        source_egocar_masks = src_pack[11]
        has_source_egocar_mask = src_pack[12]

        target_images = tgt_pack[0]
        target_extrinsics = _transform_extrinsics_list(tgt_pack[1])
        target_intrinsics = tgt_pack[2]
        target_depths = tgt_pack[3]
        target_frame_idxs = tgt_pack[4]
        target_cam_idxs = tgt_pack[5]
        target_kf_idxs = tgt_pack[6]
        target_sky_masks = tgt_pack[7]
        has_target_sky_mask = tgt_pack[8]
        target_viewdirs_list = tgt_pack[9]
        has_target_viewdirs = tgt_pack[10]
        target_egocar_masks = tgt_pack[11]
        has_target_egocar_mask = tgt_pack[12]

        pointcloud = None
        if self.pointcloud_generator is not None:
            pc_key = (int(scene_id), int(segment_id))
            pointcloud = self._segment_pointcloud_cache.get(pc_key)
            if pointcloud is None:
                pointcloud = self.pointcloud_generator.generate_pointcloud(
                    dataset=self,
                    scene_id=scene_id,
                    segment_id=segment_id,
                    segment_first_pose=segment_first_pose,
                )
                self._segment_pointcloud_cache[pc_key] = pointcloud

        all_frame_indices: Set[int] = set(source_frame_idxs) | set(target_frame_idxs)
        if include_test:
            if test_image_refs is not None:
                for ref in test_image_refs:
                    if int(ref[0]) not in sidx.test_frame_set:
                        raise ValueError(
                            f"test_image_ref {ref} frame not in segment test_frame_indices "
                            f"(scene={scene_id} segment={segment_id})"
                        )
                    self.validate_image_ref(scene_id, segment_id, tuple(ref), purpose="test")
                    all_frame_indices.add(int(ref[0]))
            else:
                all_frame_indices.update(segment.get("test_frame_indices", []))

        dynamic_info = None
        if pointcloud is not None and isinstance(pointcloud, dict) and "dynamic" in pointcloud:
            dynamic_pcd = pointcloud.get("dynamic")
            if isinstance(dynamic_pcd, dict) and len(dynamic_pcd) > 0 and pointcloud.get("instance_mapping") is None:
                raise ValueError(
                    "Dynamic pointcloud provided but instance_mapping is missing; "
                    "cannot build dynamic_info without mapping original IDs to pointcloud intids."
                )
            instance_mapping = pointcloud.get("instance_mapping")
            exclude_instance_intids: Optional[Set[int]] = None
            meta = pointcloud.get("metadata") if isinstance(pointcloud, dict) else None
            if meta:
                raw = meta.get("static_instance_intids")
                if raw:
                    exclude_instance_intids = {int(x) for x in raw}
            if scene_dataset.pixel_source is not None and scene_dataset.pixel_source.instances_pose is not None:
                dynamic_info = self._build_dynamic_info(
                    scene_dataset=scene_dataset,
                    frame_indices=sorted(all_frame_indices),
                    instance_mapping=instance_mapping,
                    world_to_seg0=world_to_seg0,
                    exclude_instance_intids=exclude_instance_intids,
                )

        test_images: List[Tensor] = []
        test_extrinsics: List[Tensor] = []
        test_intrinsics: List[Tensor] = []
        test_depths: List[Tensor] = []
        test_frame_idxs: List[int] = []
        test_cam_idxs: List[int] = []
        test_sky_masks: List[Optional[Tensor]] = []
        has_test_sky_mask = False
        test_viewdirs_list: List[Optional[Tensor]] = []
        has_test_viewdirs = False
        test_egocar_masks: List[Optional[Tensor]] = []
        has_test_egocar_mask = False

        num_cams = int(scene_dataset.num_cams)

        resolved_test_image_refs: Optional[List[Tuple[int, int]]] = None

        def _append_test_pack(pack: Dict[str, Any]) -> None:
            nonlocal has_test_sky_mask, has_test_viewdirs, has_test_egocar_mask
            test_images.append(pack["image"])
            test_extrinsics.append(pack["extrinsic"])
            test_intrinsics.append(pack["intrinsic"])
            test_depths.append(pack["depth"])
            test_frame_idxs.append(int(pack["frame_idx"]))
            test_cam_idxs.append(int(pack["cam_idx"]))
            sm = pack.get("sky_mask")
            if sm is not None:
                has_test_sky_mask = True
            test_sky_masks.append(sm)
            vd = pack.get("viewdirs")
            if vd is not None:
                has_test_viewdirs = True
            test_viewdirs_list.append(vd)
            em = pack.get("egocar_mask")
            if em is not None:
                has_test_egocar_mask = True
            test_egocar_masks.append(em)

        if include_test:
            resolved_test_image_refs = []
            if test_image_refs is not None:
                for ref in test_image_refs:
                    ref_t = (int(ref[0]), int(ref[1]))
                    pack = self._get_cached_or_load_view_from_image_ref(
                        scene_id, segment_id, scene_dataset, ref_t
                    )
                    _append_test_pack(pack)
                    resolved_test_image_refs.append(ref_t)
            else:
                segment_test_frames = segment.get("test_frame_indices", [])
                if len(segment_test_frames) > 0:
                    pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
                    try:
                        max_test_images = pixel_source_cfg.get("max_test_images", 0)
                    except Exception:
                        max_test_images = 0
                    if max_test_images > 0 and len(segment_test_frames) > max_test_images:
                        selected_test_frames = random.sample(segment_test_frames, max_test_images)
                    else:
                        selected_test_frames = segment_test_frames
                    for frame_idx in selected_test_frames:
                        for cam_idx in range(num_cams):
                            ref = (int(frame_idx), int(cam_idx))
                            pack = self._get_cached_or_load_view_from_image_ref(
                                scene_id, segment_id, scene_dataset, ref
                            )
                            _append_test_pack(pack)
                            resolved_test_image_refs.append(ref)

        if include_test and len(test_extrinsics) > 0:
            test_extrinsics = _transform_extrinsics_list(test_extrinsics)

        scene_folder_name = (
            f"{int(scene_id):03d}" if self.data_cfg.get("dataset") not in ["kitti", "nuplan"] else str(scene_id)
        )
        batch_aabb = self.segment_aabb.to(device=self.device)

        source_keyframe_indices_tensor = torch.tensor(source_kf_idxs, dtype=torch.long)
        target_keyframe_indices_tensor = torch.tensor(target_kf_idxs, dtype=torch.long)

        batch: Dict[str, Any] = {
            "scene_id": torch.tensor([scene_id], dtype=torch.long),
            "scene_folder_name": scene_folder_name,
            "segment_id": segment_id,
            "aabb": batch_aabb,
            "segment_first_pose": segment_first_pose,
            "segment_first_frame_idx": segment_first_frame_idx,
            "segment_first_pose_source": segment_pose_source,
            "request_meta": {
                "source_image_refs": [tuple(r) for r in source_image_refs],
                "target_image_refs": [tuple(r) for r in target_image_refs],
                "test_image_refs": resolved_test_image_refs,
                "assembly_mode": "image_ref",
            },
            "index_meta": {
                "source_keyframe_indices": source_kf_idxs,
                "target_keyframe_indices": target_kf_idxs,
            },
            "keyframe_info": {
                "segment_keyframes": segment["keyframe_indices"],
                "source_keyframes": list(dict.fromkeys(source_kf_idxs)),
                "target_keyframes": list(dict.fromkeys(target_kf_idxs)),
            },
            "source": {
                "image": torch.stack(source_images, dim=0),
                "extrinsics": torch.stack(source_extrinsics, dim=0),
                "intrinsics": torch.stack(source_intrinsics, dim=0),
                "depth": torch.stack(source_depths, dim=0),
                "frame_indices": torch.tensor(source_frame_idxs, dtype=torch.long),
                "cam_indices": torch.tensor(source_cam_idxs, dtype=torch.long),
                "keyframe_indices": source_keyframe_indices_tensor,
            },
            "target": {
                "image": torch.stack(target_images, dim=0),
                "extrinsics": torch.stack(target_extrinsics, dim=0),
                "intrinsics": torch.stack(target_intrinsics, dim=0),
                "depth": torch.stack(target_depths, dim=0),
                "frame_indices": torch.tensor(target_frame_idxs, dtype=torch.long),
                "cam_indices": torch.tensor(target_cam_idxs, dtype=torch.long),
                "keyframe_indices": target_keyframe_indices_tensor,
            },
        }

        if len(source_image_refs) == 1:
            batch["index_meta"]["source_keyframe_idx"] = source_kf_idxs[0]

        if has_source_sky_mask:
            source_sky_mask_stack = []
            for mask, img in zip(source_sky_masks, source_images):
                if mask is None:
                    H, W = img.shape[:2]
                    source_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    source_sky_mask_stack.append(mask.to(self.device).float())
            batch["source"]["sky_mask"] = torch.stack(source_sky_mask_stack, dim=0)

        if has_source_viewdirs:
            source_viewdirs_stack = []
            for vd, img in zip(source_viewdirs_list, source_images):
                if vd is None:
                    H, W = img.shape[:2]
                    source_viewdirs_stack.append(torch.zeros((H, W, 3), dtype=torch.float32, device=self.device))
                else:
                    source_viewdirs_stack.append(vd.to(self.device).float())
            batch["source"]["viewdirs"] = torch.stack(source_viewdirs_stack, dim=0)

        if has_source_egocar_mask:
            source_egocar_mask_stack = []
            for mask, img in zip(source_egocar_masks, source_images):
                if mask is None:
                    H, W = img.shape[:2]
                    source_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    source_egocar_mask_stack.append(mask.to(self.device).float())
            batch["source"]["egocar_mask"] = torch.stack(source_egocar_mask_stack, dim=0)

        if has_target_sky_mask:
            target_sky_mask_stack = []
            for mask, img in zip(target_sky_masks, target_images):
                if mask is None:
                    H, W = img.shape[:2]
                    target_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    target_sky_mask_stack.append(mask.to(self.device).float())
            batch["target"]["sky_mask"] = torch.stack(target_sky_mask_stack, dim=0)

        if has_target_viewdirs:
            target_viewdirs_stack = []
            for vd, img in zip(target_viewdirs_list, target_images):
                if vd is None:
                    H, W = img.shape[:2]
                    target_viewdirs_stack.append(torch.zeros((H, W, 3), dtype=torch.float32, device=self.device))
                else:
                    target_viewdirs_stack.append(vd.to(self.device).float())
            batch["target"]["viewdirs"] = torch.stack(target_viewdirs_stack, dim=0)

        if has_target_egocar_mask:
            target_egocar_mask_stack = []
            for mask, img in zip(target_egocar_masks, target_images):
                if mask is None:
                    H, W = img.shape[:2]
                    target_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    target_egocar_mask_stack.append(mask.to(self.device).float())
            batch["target"]["egocar_mask"] = torch.stack(target_egocar_mask_stack, dim=0)

        if pointcloud is not None:
            batch["pointcloud"] = pointcloud
        if dynamic_info is not None:
            batch["dynamic_info"] = dynamic_info

        if include_test and len(test_images) > 0:
            batch["test"] = {
                "image": torch.stack(test_images, dim=0),
                "extrinsics": torch.stack(test_extrinsics, dim=0),
                "intrinsics": torch.stack(test_intrinsics, dim=0),
                "depth": torch.stack(test_depths, dim=0),
                "frame_indices": torch.tensor(test_frame_idxs, dtype=torch.long),
                "cam_indices": torch.tensor(test_cam_idxs, dtype=torch.long),
            }
            if has_test_sky_mask:
                test_sky_mask_stack = []
                for mask, img in zip(test_sky_masks, test_images):
                    if mask is None:
                        H, W = img.shape[:2]
                        test_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                    else:
                        test_sky_mask_stack.append(mask.to(self.device).float())
                batch["test"]["sky_mask"] = torch.stack(test_sky_mask_stack, dim=0)
            if has_test_viewdirs:
                test_viewdirs_stack = []
                for vd, img in zip(test_viewdirs_list, test_images):
                    if vd is None:
                        H, W = img.shape[:2]
                        test_viewdirs_stack.append(torch.zeros((H, W, 3), dtype=torch.float32, device=self.device))
                    else:
                        test_viewdirs_stack.append(vd.to(self.device).float())
                batch["test"]["viewdirs"] = torch.stack(test_viewdirs_stack, dim=0)
            if has_test_egocar_mask:
                test_egocar_mask_stack = []
                for mask, img in zip(test_egocar_masks, test_images):
                    if mask is None:
                        H, W = img.shape[:2]
                        test_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                    else:
                        test_egocar_mask_stack.append(mask.to(self.device).float())
                batch["test"]["egocar_mask"] = torch.stack(test_egocar_mask_stack, dim=0)

        return batch

    def resolve_test_image_refs_deterministic(self, scene_id: int, segment_id: int) -> List[ImageRef]:
        """
        Expand segment test frames to per-camera image refs in a fixed order (sorted frames, then cams).
        Matches the default test path in _assemble_segment_batch_from_image_refs when test_image_refs is None,
        but without random subsampling so TrainSchedulerV4 can pin the same refs for every step in a block.
        """
        sidx = self.get_segment_index(scene_id, segment_id)
        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment = scene_data["segments"][int(sidx.segment_id)]
        segment_test_frames = sorted(int(f) for f in segment.get("test_frame_indices", []))
        if len(segment_test_frames) == 0:
            return []
        pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
        max_test_images = int(pixel_source_cfg.get("max_test_images", 0))
        if max_test_images > 0 and len(segment_test_frames) > max_test_images:
            selected = segment_test_frames[:max_test_images]
        else:
            selected = list(segment_test_frames)
        num_cams = int(scene_data["dataset"].num_cams)
        refs: List[ImageRef] = []
        for frame_idx in selected:
            for cam_idx in range(num_cams):
                ref: ImageRef = (int(frame_idx), int(cam_idx))
                self.validate_image_ref(scene_id, segment_id, ref, purpose="test")
                refs.append(ref)
        return refs

    def get_segment_batch_from_image_refs(
        self,
        request: BatchRequestV3,
        *,
        enforce_target0_equals_source: bool = True,
    ) -> Dict[str, Any]:
        if len(request.target_image_refs) == 0:
            raise ValueError("target_image_refs must not be empty")
        if request.test_image_refs is not None and not request.include_test:
            raise ValueError("test_image_refs is set but include_test=False")
        return self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            [request.source_image_ref],
            request.target_image_refs,
            include_test=request.include_test,
            test_image_refs=request.test_image_refs,
            enforce_target0_equals_source=enforce_target0_equals_source,
        )

    def create_train_scheduler_v4(
        self,
        *,
        state_write_interval_steps: int,
        updates_per_block: int,
        keyframes_per_episode: int,
        episodes_per_segment: int,
        keyframe_window_policy: str,
        pair_order_policy: str,
        total_target_images: int,
        include_source: bool,
        extra_target_policy: str,
        prefer_nearby_keyframes: bool,
        fallback_expand_to_segment: bool,
        fallback_with_replacement: bool,
        overlap_mode: str,
        emit_preload_hints: bool,
        execute_preload_hints: bool,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
    ) -> "TrainSchedulerV4":
        return TrainSchedulerV4(
            dataset=self,
            state_write_interval_steps=state_write_interval_steps,
            updates_per_block=updates_per_block,
            keyframes_per_episode=keyframes_per_episode,
            episodes_per_segment=episodes_per_segment,
            keyframe_window_policy=keyframe_window_policy,
            pair_order_policy=pair_order_policy,
            total_target_images=total_target_images,
            include_source=include_source,
            extra_target_policy=extra_target_policy,
            prefer_nearby_keyframes=prefer_nearby_keyframes,
            fallback_expand_to_segment=fallback_expand_to_segment,
            fallback_with_replacement=fallback_with_replacement,
            overlap_mode=overlap_mode,
            emit_preload_hints=emit_preload_hints,
            execute_preload_hints=execute_preload_hints,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
        )

    def get_segment_batch_from_frames(
        self,
        scene_id: int,
        segment_id: int,
        source_frame_idx: int,
        target_frame_indices: List[int],
        include_test: bool = False,
        test_frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Legacy frame-level API: expands each training frame to all cameras (scheduler v2 tensor layout).

        Unlike ``get_segment_batch()``, ``keyframe_indices`` in the returned batch follow **image-level**
        alignment (one per source/target image). Legacy code that assumes
        ``len(source['keyframe_indices']) == num_source_keyframes`` will not hold here.
        """
        if len(target_frame_indices) == 0:
            raise ValueError("target_frame_indices must not be empty")
        if target_frame_indices[0] != source_frame_idx:
            raise ValueError(
                "target_frame_indices[0] must equal source_frame_idx for scheduler v2 semantics"
            )

        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segments = scene_data.get("segments", [])
        if int(segment_id) < 0 or int(segment_id) >= len(segments):
            raise ValueError(f"segment_id={segment_id} out of range for scene {scene_id}")
        segment = segments[int(segment_id)]
        segment_frames = set(segment.get("frame_indices", []))
        if int(source_frame_idx) not in segment_frames:
            raise ValueError(
                f"source_frame_idx={source_frame_idx} is not in scene={scene_id} segment={segment_id}"
            )
        for frame_idx in target_frame_indices:
            if int(frame_idx) not in segment_frames:
                raise ValueError(
                    f"target frame {frame_idx} is not in scene={scene_id} segment={segment_id}"
                )

        num_cams = int(scene_data["dataset"].num_cams)
        source_image_refs = [(int(source_frame_idx), c) for c in range(num_cams)]
        target_image_refs = [
            (int(f), c) for f in target_frame_indices for c in range(num_cams)
        ]

        test_refs: Optional[List[ImageRef]] = None
        if include_test and test_frame_indices is not None:
            test_refs = [
                (int(f), c) for f in test_frame_indices for c in range(num_cams)
            ]

        return self._assemble_segment_batch_from_image_refs(
            int(scene_id),
            int(segment_id),
            source_image_refs,
            target_image_refs,
            include_test=include_test,
            test_image_refs=test_refs,
            enforce_target0_equals_source=True,
        )


class TrainSchedulerV4:
    """
    Image-ref StreetForward scheduler: single source image + multi-target images per step,
    U / block / episode / segment hierarchy. Uses MultiSceneDatasetV3.get_segment_batch_from_image_refs only.
    """

    def __init__(
        self,
        *,
        dataset: MultiSceneDatasetV3,
        state_write_interval_steps: int,
        updates_per_block: int,
        keyframes_per_episode: int,
        episodes_per_segment: int,
        keyframe_window_policy: str,
        pair_order_policy: str,
        total_target_images: int,
        include_source: bool,
        extra_target_policy: str,
        prefer_nearby_keyframes: bool,
        fallback_expand_to_segment: bool,
        fallback_with_replacement: bool,
        overlap_mode: str,
        emit_preload_hints: bool,
        execute_preload_hints: bool,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
    ) -> None:
        if state_write_interval_steps < 1:
            raise ValueError("state_write_interval_steps must be >= 1")
        if updates_per_block < 1:
            raise ValueError("updates_per_block must be >= 1")
        if keyframes_per_episode < 1:
            raise ValueError("keyframes_per_episode must be >= 1")
        if episodes_per_segment < 1:
            raise ValueError("episodes_per_segment must be >= 1")
        if total_target_images < 1:
            raise ValueError("total_target_images must be >= 1")
        if not include_source:
            raise ValueError("include_source must be true in TrainSchedulerV4")
        if extra_target_policy != "same_cam_different_keyframe":
            raise ValueError(f"Unsupported extra_target_policy={extra_target_policy!r}")
        if keyframe_window_policy != "random_contiguous_window":
            raise ValueError(f"Unsupported keyframe_window_policy={keyframe_window_policy!r}")
        if pair_order_policy != "shuffle_without_replacement":
            raise ValueError(f"Unsupported pair_order_policy={pair_order_policy!r}")
        if overlap_mode != "none":
            raise ValueError(
                f"TrainSchedulerV4 v1 only supports overlap_mode='none', got {overlap_mode!r}"
            )

        self.dataset = dataset
        self.U = int(state_write_interval_steps)
        self.updates_per_block = int(updates_per_block)
        self.keyframes_per_episode = int(keyframes_per_episode)
        self.episodes_per_segment = int(episodes_per_segment)
        self.keyframe_window_policy = str(keyframe_window_policy)
        self.pair_order_policy = str(pair_order_policy)
        self.total_target_images = int(total_target_images)
        self.include_source = bool(include_source)
        self.extra_target_policy = str(extra_target_policy)
        self.prefer_nearby_keyframes = bool(prefer_nearby_keyframes)
        self.fallback_expand_to_segment = bool(fallback_expand_to_segment)
        self.fallback_with_replacement = bool(fallback_with_replacement)
        self.overlap_mode = str(overlap_mode)
        self.emit_preload_hints = bool(emit_preload_hints)
        self.execute_preload_hints = bool(execute_preload_hints)
        self.include_test = bool(include_test)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None

        self.epoch_idx = 0
        self.global_step = 0
        self.epoch_plan: List[Dict[str, Any]] = []
        self.plan_cursor = 0
        self.current_segment_state: Optional[Dict[str, Any]] = None
        self._pending_events: List[Dict[str, Any]] = []
        self._block_idx_global = 0
        self._reset_episode_idx = 0

        if not self.dataset._initialized:
            self.dataset.initialize()
        self.start_new_epoch()

    def pop_events(self) -> List[Dict[str, Any]]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(event)

    def _init_epoch_segment_pair_iterator(self) -> None:
        """Order scenes for this epoch without loading them; segments enumerated lazily via get_scene."""
        if self.fixed_scene_id is not None:
            self._epoch_scene_queue = [int(self.fixed_scene_id)]
        else:
            if len(getattr(self.dataset, "scene_training_queue", [])) == 0:
                self.dataset._initialize_training_queue()
            q = list(getattr(self.dataset, "scene_training_queue", []))
            if len(q) == 0:
                raise ValueError("No valid scenes in dataset.scene_training_queue")
            random.shuffle(q)
            self._epoch_scene_queue = q
        self._epoch_scene_q_idx = 0
        self._epoch_current_scene_id = None
        self._epoch_segment_ids = []
        self._epoch_segment_pos = 0

    def _next_scene_segment_pair(self) -> Optional[Tuple[int, int]]:
        """Next (scene_id, segment_id). Loads one scene at a time when moving to that scene's segments."""
        while True:
            if self._epoch_segment_pos >= len(self._epoch_segment_ids):
                if self._epoch_scene_q_idx >= len(self._epoch_scene_queue):
                    return None
                sid = int(self._epoch_scene_queue[self._epoch_scene_q_idx])
                self._epoch_scene_q_idx += 1
                scene_data = self.dataset.get_scene(sid)
                if scene_data is None:
                    raise ValueError(f"Scene {sid} cannot be loaded")
                nseg = len(scene_data["segments"])
                if self.fixed_segment_id is not None:
                    if self.fixed_segment_id < 0 or self.fixed_segment_id >= nseg:
                        raise ValueError(f"fixed_segment_id={self.fixed_segment_id} out of range in scene={sid}")
                    self._epoch_segment_ids = [int(self.fixed_segment_id)]
                else:
                    self._epoch_segment_ids = list(range(nseg))
                    random.shuffle(self._epoch_segment_ids)
                self._epoch_segment_pos = 0
                self._epoch_current_scene_id = sid
            seg_id = int(self._epoch_segment_ids[self._epoch_segment_pos])
            self._epoch_segment_pos += 1
            assert self._epoch_current_scene_id is not None
            return (int(self._epoch_current_scene_id), seg_id)

    def _ensure_epoch_plan_index(self, idx: int) -> None:
        """Append plan stubs (scene_id, segment_id only) until len(epoch_plan) > idx or epoch exhausted."""
        while len(self.epoch_plan) <= idx:
            p = self._next_scene_segment_pair()
            if p is None:
                break
            self.epoch_plan.append({"scene_id": int(p[0]), "segment_id": int(p[1])})

    def _hydrate_plan_item_budget(self, idx: int) -> None:
        """Fill segment_budget_u / step budget from episode × window × block (see TrainSchedulerV4)."""
        self._ensure_epoch_plan_index(idx)
        if idx >= len(self.epoch_plan):
            return
        it = self.epoch_plan[idx]
        if "segment_budget_u" in it:
            return
        sidx = self.dataset.get_segment_index(int(it["scene_id"]), int(it["segment_id"]))
        num_keyframes = len(sidx.keyframe_indices)
        num_cams = int(sidx.num_cams)
        w_eff = int(min(self.keyframes_per_episode, num_keyframes))
        b_seg = int(self.episodes_per_segment * w_eff * num_cams)
        segment_budget_u = int(b_seg * self.updates_per_block)
        it["num_keyframes"] = int(num_keyframes)
        it["num_cams"] = num_cams
        it["w_eff"] = w_eff
        it["b_seg"] = b_seg
        it["segment_budget_u"] = segment_budget_u
        it["segment_step_budget"] = int(segment_budget_u * self.U)
        it["U"] = int(self.U)

    def build_epoch_plan(self) -> None:
        self._init_epoch_segment_pair_iterator()
        self.epoch_plan = []
        self.plan_cursor = 0

    def start_new_epoch(self) -> None:
        self.epoch_idx += 1
        self.build_epoch_plan()
        self.current_segment_state = None

    def _validate_target_sampling_feasible(self, sidx: SegmentIndex) -> None:
        nk = len(sidx.keyframe_indices)
        extra_needed = self.total_target_images - 1
        if extra_needed <= 0:
            return
        if nk < 2:
            raise ValueError(
                "TrainSchedulerV4: total_target_images > 1 requires at least 2 keyframes in the segment "
                f"(scene={sidx.scene_id} segment={sidx.segment_id}, got {nk})"
            )
        if not self.fallback_with_replacement:
            if nk - 1 < extra_needed:
                raise ValueError(
                    "TrainSchedulerV4: fallback_with_replacement=false requires "
                    f"len(keyframe_indices)-1 >= total_target_images-1 "
                    f"(scene={sidx.scene_id} segment={sidx.segment_id}, "
                    f"got nk={nk}, need distinct extras={extra_needed})"
                )
        window_extra_cap = min(self.keyframes_per_episode, nk) - 1
        if (
            not self.fallback_expand_to_segment
            and not self.fallback_with_replacement
            and window_extra_cap < extra_needed
        ):
            raise ValueError(
                "TrainSchedulerV4: without fallback_expand_to_segment or fallback_with_replacement, "
                "each episode window must contain enough keyframes distinct from the source for extra targets: "
                f"need window_extra_cap=min(keyframes_per_episode,nk)-1 >= total_target_images-1 "
                f"(scene={sidx.scene_id} segment={sidx.segment_id}, "
                f"keyframes_per_episode={self.keyframes_per_episode}, nk={nk}, "
                f"window_extra_cap={window_extra_cap}, extra_needed={extra_needed})"
            )

    def _sample_contiguous_window(self, sidx: SegmentIndex) -> List[int]:
        seg_kfs = list(sidx.keyframe_indices)
        R_kf = self.keyframes_per_episode
        if len(seg_kfs) > R_kf:
            start = random.randint(0, len(seg_kfs) - R_kf)
            return list(seg_kfs[start : start + R_kf])
        return list(seg_kfs)

    def _build_pair_list(self, window: List[int], num_cams: int) -> List[Tuple[int, int]]:
        pairs = [(int(kf), cam) for kf in window for cam in range(num_cams)]
        if self.pair_order_policy == "shuffle_without_replacement":
            random.shuffle(pairs)
        return pairs

    def _kf_positions(self, sidx: SegmentIndex) -> Dict[int, int]:
        return {int(k): i for i, k in enumerate(sidx.keyframe_indices)}

    def _sample_target_image_refs(
        self,
        sidx: SegmentIndex,
        source_image_ref: ImageRef,
        episode_window: List[int],
    ) -> List[ImageRef]:
        f_src, cam_src = int(source_image_ref[0]), int(source_image_ref[1])
        kf_src = int(sidx.frame_to_keyframe[f_src])
        pos = self._kf_positions(sidx)
        if kf_src not in pos:
            raise ValueError(f"source keyframe {kf_src} not in segment keyframe_indices")
        pos_src = int(pos[kf_src])

        refs: List[ImageRef] = [(f_src, cam_src)]
        extra_needed = self.total_target_images - 1
        if extra_needed <= 0:
            return refs

        def _sorted_window_others() -> List[int]:
            others = [int(k) for k in episode_window if int(k) != kf_src]
            if self.prefer_nearby_keyframes:
                others.sort(key=lambda k: abs(int(pos[k]) - pos_src))
            return others

        def _sorted_segment_others() -> List[int]:
            others = [int(k) for k in sidx.keyframe_indices if int(k) != kf_src]
            if self.prefer_nearby_keyframes:
                others.sort(key=lambda k: abs(int(pos[k]) - pos_src))
            return others

        picked_kfs: List[int] = []
        for kf in _sorted_window_others():
            if len(picked_kfs) >= extra_needed:
                break
            picked_kfs.append(kf)

        if len(picked_kfs) < extra_needed and self.fallback_expand_to_segment:
            for kf in _sorted_segment_others():
                if len(picked_kfs) >= extra_needed:
                    break
                if kf in picked_kfs:
                    continue
                picked_kfs.append(kf)

        if len(picked_kfs) < extra_needed:
            if not self.fallback_with_replacement:
                raise ValueError(
                    f"Not enough distinct keyframes for {extra_needed} extra targets "
                    f"(scene={sidx.scene_id} segment={sidx.segment_id})"
                )
            pool = [int(k) for k in sidx.keyframe_indices if int(k) != kf_src]
            if len(pool) == 0:
                raise ValueError(
                    f"No non-source keyframes for extra targets (scene={sidx.scene_id} segment={sidx.segment_id})"
                )
            while len(picked_kfs) < extra_needed:
                picked_kfs.append(int(random.choice(pool)))

        for kf in picked_kfs[:extra_needed]:
            frame_tgt = int(random.choice(sidx.keyframe_to_frames[int(kf)]))
            refs.append((frame_tgt, cam_src))

        return refs

    def _refs_for_pair(
        self,
        sidx: SegmentIndex,
        kf: int,
        cam: int,
        episode_window: List[int],
    ) -> Tuple[ImageRef, List[ImageRef]]:
        frame_src = int(random.choice(sidx.keyframe_to_frames[int(kf)]))
        source_ref: ImageRef = (frame_src, int(cam))
        target_refs = self._sample_target_image_refs(sidx, source_ref, episode_window)
        return source_ref, target_refs

    def _emit_preload_hint_episode_superset(
        self, scene_id: int, segment_id: int, pair_list: List[Tuple[int, int]]
    ) -> None:
        """All (frame, cam) that any block in this episode might use as source — no random frame guess."""
        if not self.emit_preload_hints and not self.execute_preload_hints:
            return
        sidx = self.dataset.get_segment_index(scene_id, segment_id)
        seen: Set[Tuple[int, int]] = set()
        ordered: List[ImageRef] = []
        for kf, cam in pair_list:
            for f in sidx.keyframe_to_frames[int(kf)]:
                t = (int(f), int(cam))
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
        hint = self.dataset.build_preload_hint(scene_id, segment_id, future_image_refs=ordered)
        if self.emit_preload_hints:
            self._emit(
                {
                    "type": "preload_hint",
                    "epoch_idx": int(self.epoch_idx),
                    "global_step": int(self.global_step),
                    "scene_id": int(scene_id),
                    "segment_id": int(segment_id),
                    "hint": hint,
                    "hint_scope": "episode_source_superset",
                }
            )
        if self.execute_preload_hints:
            self.dataset.submit_preload_hint(
                hint=hint,
                hint_scope="episode_source_superset",
                epoch_idx=int(self.epoch_idx),
                global_step=int(self.global_step),
                block_idx_global=int(self._block_idx_global),
                include_test=bool(self.include_test),
            )

    def _emit_preload_hint_next_block_exact(self, st: Dict[str, Any]) -> None:
        """Next block's source + target refs using the same sampling as _start_block (RNG state preserved)."""
        if not self.emit_preload_hints and not self.execute_preload_hints:
            return
        pc = int(st["pair_cursor"])
        pair_list: List[Tuple[int, int]] = list(st["pair_list"])
        if pc >= len(pair_list):
            return
        scene_id = int(st["scene_id"])
        segment_id = int(st["segment_id"])
        rng_state = random.getstate()
        try:
            kf, cam = pair_list[pc]
            sidx = self.dataset.get_segment_index(scene_id, segment_id)
            src, tgts = self._refs_for_pair(sidx, int(kf), int(cam), list(st["episode_window_keyframes"]))
            future = list(dict.fromkeys(list([src]) + list(tgts)))
            hint = self.dataset.build_preload_hint(scene_id, segment_id, future_image_refs=future)
            if self.emit_preload_hints:
                self._emit(
                    {
                        "type": "preload_hint",
                        "epoch_idx": int(self.epoch_idx),
                        "global_step": int(self.global_step),
                        "scene_id": scene_id,
                        "segment_id": segment_id,
                        "hint": hint,
                        "hint_scope": "next_block_exact",
                    }
                )
            if self.execute_preload_hints:
                self.dataset.submit_preload_hint(
                    hint=hint,
                    hint_scope="next_block_exact",
                    epoch_idx=int(self.epoch_idx),
                    global_step=int(self.global_step),
                    block_idx_global=int(st.get("block_idx_global", self._block_idx_global)),
                    include_test=bool(self.include_test),
                )
        finally:
            random.setstate(rng_state)

    def _start_episode(self) -> None:
        st = self.current_segment_state
        if st is None:
            raise ValueError("TrainSchedulerV4 internal state is not initialized")
        if int(st["episodes_started"]) >= self.episodes_per_segment:
            raise ValueError("TrainSchedulerV4: _start_episode called when episode quota is exhausted")
        scene_id = int(st["scene_id"])
        segment_id = int(st["segment_id"])
        sidx = self.dataset.get_segment_index(scene_id, segment_id)

        st["episodes_started"] = int(st["episodes_started"]) + 1
        window = self._sample_contiguous_window(sidx)
        pair_list = self._build_pair_list(window, sidx.num_cams)
        st["episode_window_keyframes"] = list(window)
        st["pair_list"] = pair_list
        st["pair_cursor"] = 0
        st["block_idx_in_episode"] = -1

        self._reset_episode_idx += 1
        self._emit(
            {
                "type": "reset_event",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": scene_id,
                "segment_id": segment_id,
                "reset_episode_idx": int(self._reset_episode_idx),
                "reason": "episode_begin",
                "window_keyframes": list(window),
                "num_pairs": int(len(pair_list)),
            }
        )
        self._emit_preload_hint_episode_superset(scene_id, segment_id, pair_list)

    def _start_block(self) -> bool:
        st = self.current_segment_state
        if st is None:
            raise ValueError("TrainSchedulerV4 internal state is not initialized")
        scene_id = int(st["scene_id"])
        segment_id = int(st["segment_id"])
        sidx = self.dataset.get_segment_index(scene_id, segment_id)

        while True:
            if int(st["segment_local_u"]) >= int(st["segment_budget_u"]):
                self._end_segment()
                return False
            if int(st["pair_cursor"]) < len(st["pair_list"]):
                break
            if int(st["segment_local_u"]) >= int(st["segment_budget_u"]):
                self._end_segment()
                return False
            if int(st["episodes_started"]) < self.episodes_per_segment:
                self._start_episode()
                continue
            raise ValueError(
                "TrainSchedulerV4: exhausted episode quota before reaching segment budget; "
                "check scheduler_v4 configuration vs segment_budget_u."
            )

        kf_src, cam_src = st["pair_list"][int(st["pair_cursor"])]
        st["pair_cursor"] = int(st["pair_cursor"]) + 1
        source_ref, target_refs = self._refs_for_pair(
            sidx, int(kf_src), int(cam_src), list(st["episode_window_keyframes"])
        )
        frame_src = int(source_ref[0])

        st["source_keyframe_idx"] = int(kf_src)
        st["source_frame_idx"] = int(frame_src)
        st["source_cam_idx"] = int(cam_src)
        st["source_image_ref"] = source_ref
        st["target_image_refs"] = list(target_refs)

        remaining_u = int(st["segment_budget_u"]) - int(st["segment_local_u"])
        st["effective_u_this_block"] = int(min(self.updates_per_block, remaining_u))
        if st["effective_u_this_block"] < 1:
            raise ValueError("TrainSchedulerV4: effective_u_this_block < 1 (internal error)")
        st["u_in_block"] = 0

        st["block_idx_in_episode"] = int(st["block_idx_in_episode"]) + 1
        st["block_idx_in_segment"] = int(st.get("block_idx_in_segment", -1)) + 1
        self._block_idx_global += 1
        st["block_idx_global"] = int(self._block_idx_global)

        eff_u = int(st["effective_u_this_block"])
        K_steps_effective = int(eff_u * self.U)
        if self.include_test:
            st["block_test_image_refs"] = self.dataset.resolve_test_image_refs_deterministic(scene_id, segment_id)
        else:
            st["block_test_image_refs"] = None

        self._emit(
            {
                "type": "block_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": scene_id,
                "segment_id": segment_id,
                "reset_episode_idx": int(self._reset_episode_idx),
                "block_idx_in_episode": int(st["block_idx_in_episode"]),
                "block_idx_in_segment": int(st["block_idx_in_segment"]),
                "block_idx_global": int(st["block_idx_global"]),
                "source_keyframe_idx": int(kf_src),
                "source_frame_idx": int(frame_src),
                "source_cam_idx": int(cam_src),
                "source_image_ref": tuple(source_ref),
                "target_image_refs": [tuple(x) for x in target_refs],
                "U": int(self.U),
                "updates_per_block": int(self.updates_per_block),
                "effective_u_this_block": eff_u,
                "K_u_nominal": int(self.updates_per_block),
                "K_u_effective": eff_u,
                "K_steps_effective": int(K_steps_effective),
                "K_steps": int(K_steps_effective),
            }
        )
        self._emit_preload_hint_next_block_exact(st)
        return True

    def _end_segment(self) -> None:
        st = self.current_segment_state
        if st is None:
            return
        self._emit(
            {
                "type": "segment_end",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(st["scene_id"]),
                "segment_id": int(st["segment_id"]),
                "segment_local_u": int(st["segment_local_u"]),
                "source_image_ref": tuple(st.get("source_image_ref", (0, 0))),
            }
        )
        self.plan_cursor += 1
        self.current_segment_state = None

    def _enter_segment(self) -> None:
        self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            raise ValueError("TrainSchedulerV4: no segment available in epoch plan")
        self._hydrate_plan_item_budget(self.plan_cursor)
        item = self.epoch_plan[self.plan_cursor]
        scene_id = int(item["scene_id"])
        segment_id = int(item["segment_id"])
        sidx = self.dataset.get_segment_index(scene_id, segment_id)
        segment_budget_u = int(item["segment_budget_u"])
        self._validate_target_sampling_feasible(sidx)

        self.current_segment_state = {
            "scene_id": scene_id,
            "segment_id": segment_id,
            "segment_local_step": 0,
            "segment_local_u": 0,
            "segment_budget_u": segment_budget_u,
            "segment_step_budget": int(item["segment_step_budget"]),
            "episodes_started": 0,
            "block_idx_in_segment": -1,
            "pair_list": [],
            "pair_cursor": 0,
            "episode_window_keyframes": [],
            "u_in_block": 0,
            "effective_u_this_block": 0,
            "source_image_ref": (-1, -1),
            "target_image_refs": [],
            "source_keyframe_idx": -1,
            "source_frame_idx": -1,
            "source_cam_idx": -1,
        }
        self.dataset.set_preload_active_scope(scene_id, segment_id)
        self._emit(
            {
                "type": "segment_begin",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": scene_id,
                "segment_id": segment_id,
                "num_keyframes": int(item["num_keyframes"]),
                "num_cams": int(item["num_cams"]),
                "w_eff": int(item["w_eff"]),
                "b_seg": int(item["b_seg"]),
                "segment_budget_u": int(segment_budget_u),
                "segment_step_budget": int(item["segment_step_budget"]),
                "updates_per_block": int(self.updates_per_block),
                "keyframes_per_episode": int(self.keyframes_per_episode),
                "episodes_per_segment": int(self.episodes_per_segment),
                "total_target_images": int(self.total_target_images),
                "U": int(self.U),
            }
        )
        self._start_episode()
        if not self._start_block():
            raise ValueError("TrainSchedulerV4: could not start first block in segment (configuration error)")

    def next_batch(self) -> Dict[str, Any]:
        self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            self.start_new_epoch()
            self._ensure_epoch_plan_index(self.plan_cursor)
        if self.plan_cursor >= len(self.epoch_plan):
            raise ValueError(
                "TrainSchedulerV4: epoch has no (scene, segment) pairs; check dataset.scene_training_queue"
            )
        if self.current_segment_state is None:
            self._enter_segment()

        st = self.current_segment_state
        if st is None:
            raise ValueError("TrainSchedulerV4 internal state is not initialized")

        scene_id = int(st["scene_id"])
        segment_id = int(st["segment_id"])
        block_test = st.get("block_test_image_refs") if self.include_test else None
        req = BatchRequestV3(
            scene_id=scene_id,
            segment_id=segment_id,
            source_image_ref=tuple(st["source_image_ref"]),
            target_image_refs=[tuple(x) for x in st["target_image_refs"]],
            include_test=self.include_test,
            test_image_refs=block_test,
        )
        batch = self.dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)

        st["segment_local_step"] = int(st["segment_local_step"]) + 1
        self.global_step += 1

        if int(st["segment_local_step"]) % self.U == 0:
            st["segment_local_u"] = int(st["segment_local_u"]) + 1
            st["u_in_block"] = int(st["u_in_block"]) + 1

            if int(st["u_in_block"]) >= int(st["effective_u_this_block"]):
                eff_u_end = int(st["effective_u_this_block"])
                self._emit(
                    {
                        "type": "block_end",
                        "epoch_idx": int(self.epoch_idx),
                        "global_step": int(self.global_step),
                        "scene_id": scene_id,
                        "segment_id": segment_id,
                        "block_idx_in_segment": int(st["block_idx_in_segment"]),
                        "block_idx_global": int(st.get("block_idx_global", 0)),
                        "source_image_ref": tuple(st["source_image_ref"]),
                        "num_updates_in_block": eff_u_end,
                        "K_u_nominal": int(self.updates_per_block),
                        "K_u_effective": eff_u_end,
                        "K_steps_effective": int(eff_u_end * self.U),
                        "U": int(self.U),
                    }
                )
                done_seg = int(st["segment_local_u"]) >= int(st["segment_budget_u"])
                if done_seg:
                    self._end_segment()
                else:
                    self._start_block()

        return batch

    def get_current_info(self) -> Dict[str, Any]:
        st = self.current_segment_state
        if st is None:
            self._ensure_epoch_plan_index(self.plan_cursor)
        if st is None and self.plan_cursor < len(self.epoch_plan):
            self._hydrate_plan_item_budget(self.plan_cursor)
            item = self.epoch_plan[self.plan_cursor]
            K_u_nominal = int(self.updates_per_block)
            K_u_effective = K_u_nominal
            K_steps_effective = int(K_u_effective * self.U)
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(item["scene_id"]),
                "segment_id": int(item["segment_id"]),
                "segment_local_step": 0,
                "segment_step_budget": int(item["segment_step_budget"]),
                "segment_local_u": 0,
                "segment_budget_u": int(item["segment_budget_u"]),
                "block_idx_in_segment": 0,
                "block_idx_global": int(self._block_idx_global),
                "source_frame_idx": -1,
                "source_keyframe_idx": -1,
                "source_cam_idx": -1,
                "source_image_ref": (-1, -1),
                "target_image_refs": [],
                "U": int(self.U),
                "K_u_nominal": K_u_nominal,
                "K_u_effective": K_u_effective,
                "K_steps_effective": int(K_steps_effective),
                "K_steps": int(K_steps_effective),
                "R_steps": 0,
                "T_steps": int(K_steps_effective),
            }
        if st is None:
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": -1,
                "segment_id": -1,
                "segment_local_step": 0,
                "segment_step_budget": 0,
                "segment_local_u": 0,
                "segment_budget_u": 0,
                "block_idx_in_segment": 0,
                "block_idx_global": int(self._block_idx_global),
                "source_frame_idx": -1,
                "source_keyframe_idx": -1,
                "source_cam_idx": -1,
                "source_image_ref": (-1, -1),
                "target_image_refs": [],
                "U": int(self.U),
                "K_u_nominal": 0,
                "K_u_effective": 0,
                "K_steps_effective": 0,
                "K_steps": 0,
                "R_steps": 0,
                "T_steps": 0,
            }
        K_u_nominal = int(self.updates_per_block)
        K_u_effective = int(st.get("effective_u_this_block", K_u_nominal))
        K_steps_effective = int(K_u_effective * self.U)
        return {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(st["scene_id"]),
            "segment_id": int(st["segment_id"]),
            "segment_local_step": int(st["segment_local_step"]),
            "segment_step_budget": int(st["segment_step_budget"]),
            "segment_local_u": int(st["segment_local_u"]),
            "segment_budget_u": int(st["segment_budget_u"]),
            "block_idx_in_segment": int(st["block_idx_in_segment"]),
            "block_idx_global": int(st.get("block_idx_global", self._block_idx_global)),
            "source_frame_idx": int(st.get("source_frame_idx", -1)),
            "source_keyframe_idx": int(st.get("source_keyframe_idx", -1)),
            "source_cam_idx": int(st.get("source_cam_idx", -1)),
            "source_image_ref": tuple(st.get("source_image_ref", (-1, -1))),
            "target_image_refs": [tuple(x) for x in st.get("target_image_refs", [])],
            "U": int(self.U),
            "K_u_nominal": K_u_nominal,
            "K_u_effective": K_u_effective,
            "K_steps_effective": int(K_steps_effective),
            "K_steps": int(K_steps_effective),
            "R_steps": 0,
            "T_steps": int(K_steps_effective),
        }
