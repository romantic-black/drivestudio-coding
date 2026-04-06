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

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

from datasets.dataset_preload_manager import (
    PRIORITY_EPISODE_SUPERSET,
    PRIORITY_NEXT_BLOCK_EXACT,
    PRIORITY_SEGMENT_STATIC,
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


def _visibility_mask_seg0(
    points_xyz: np.ndarray,
    c2w_seg0: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """points_xyz (M,3) in seg0; c2w_seg0 (4,4); K (3,3). Returns visibility bool (M,)."""
    m = int(points_xyz.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=bool)
    w2c = np.linalg.inv(c2w_seg0.astype(np.float64))
    ph = np.concatenate([points_xyz.astype(np.float64), np.ones((m, 1), dtype=np.float64)], axis=1)
    pc = (w2c @ ph.T).T[:, :3]
    proj = (K.astype(np.float64) @ pc.T).T
    zp = proj[:, 2]
    valid_z = zp > 1e-8
    u = np.zeros(m, dtype=np.float64)
    v = np.zeros(m, dtype=np.float64)
    u[valid_z] = proj[valid_z, 0] / zp[valid_z]
    v[valid_z] = proj[valid_z, 1] / zp[valid_z]
    vis = valid_z & (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
    return vis.astype(bool)


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


@dataclass(frozen=True)
class EvalRequestV3:
    scene_id: int
    segment_id: int
    source_image_ref: ImageRef
    eval_image_refs: List[ImageRef]


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


def representative_frame_for_keyframe(sidx: SegmentIndex, keyframe_idx: int) -> int:
    frames = sidx.keyframe_to_frames[int(keyframe_idx)]
    if len(frames) == 0:
        raise ValueError(
            f"keyframe {keyframe_idx} has no frames (scene={sidx.scene_id} segment={sidx.segment_id})"
        )
    return int(frames[len(frames) // 2])


class MultiSceneDatasetV3(MultiSceneDataset):
    """
    Extends MultiSceneDataset with SegmentIndex cache and image-ref batch assembly.
    """

    def __init__(
        self,
        *args: Any,
        preload_cfg: Optional[Dict[str, Any]] = None,
        overlap_stats_log_interval_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._segment_index_cache: Dict[Tuple[int, int], SegmentIndex] = {}
        # (score, n_a, n_b, n_ab) for pointcloud_topk
        self._pair_score_cache: Dict[
            Tuple[int, int, Tuple[int, int], Tuple[int, int], str, int],
            Tuple[float, int, int, int],
        ] = {}
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
        self._scene_load_coord_lock = threading.Lock()
        self._scene_load_inflight: Dict[int, threading.Event] = {}
        self._view_load_coord_lock = threading.Lock()
        self._view_load_inflight: Dict[Tuple[int, int, int, int], threading.Event] = {}
        self._segment_pointcloud_coord_lock = threading.Lock()
        self._segment_pointcloud_inflight: Dict[Tuple[int, int], threading.Event] = {}
        self._segment_index_coord_lock = threading.Lock()
        self._segment_index_inflight: Dict[Tuple[int, int], threading.Event] = {}
        self._segment_pose_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._test_image_refs_cache: Dict[Tuple[int, int, int], List[ImageRef]] = {}
        self._overlap_stats_log_interval_steps = int(overlap_stats_log_interval_steps)
        self._overlap_stats: Dict[str, float] = {
            "pair_queries": 0.0,
            "pair_cache_hits": 0.0,
            "pair_cache_misses": 0.0,
            "pair_compute_miss_ms_sum": 0.0,
            "pair_eval_wall_ms_sum": 0.0,
            "src_rep_no_visible": 0.0,
            "candidate_eval_count": 0.0,
        }

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

    def clear_preload_active_scope(self) -> None:
        self._preload_active_scene_id = None
        self._preload_active_segment_id = None

    def set_preload_training_scope(self, scene_id: int, segment_id: int) -> None:
        """Training segment for scene-cache eviction protection; pair with clear on segment/epoch end."""
        self._preload_training_scene_id = int(scene_id)
        self._preload_training_segment_id = int(segment_id)

    def clear_preload_training_scope(self) -> None:
        self._preload_training_scene_id = None
        self._preload_training_segment_id = None

    def clear_preload_scheduler_scope(self) -> None:
        self.clear_preload_active_scope()
        self.clear_preload_training_scope()

    def maybe_log_preload_stats(self, global_step: int) -> None:
        cfg = self._preload_rtcfg
        mgr = self._preload_manager
        if cfg is None or mgr is None:
            return
        interval = int(cfg.stats_log_interval_steps)
        if interval <= 0:
            return
        gs = int(global_step)
        if gs % interval != 0:
            return
        stats = mgr.pop_stats()
        if not stats:
            return
        completed = float(stats.get("tasks_completed", 0))
        lat = float(stats.get("total_latency_ms", 0.0))
        avg_ms = lat / max(completed, 1.0)
        ol = int(stats.get("overlap_pairs_loaded", 0))
        och = int(stats.get("overlap_pair_cache_hits_worker", 0))
        olat = float(stats.get("overlap_pair_total_latency_ms", 0.0))
        ol_done = max(ol + och, 1)
        avg_ol_ms = olat / float(ol_done)
        logger.info(
            "preload_stats global_step=%s tasks_completed=%s views_loaded=%s cache_hits_worker=%s "
            "segment_static_completed=%s tasks_failed=%s tasks_dropped_stale=%s tasks_dropped_queue_full=%s "
            "avg_task_latency_ms=%.3f overlap_pairs_loaded=%s overlap_pair_cache_hits_worker=%s "
            "overlap_pairs_failed=%s avg_overlap_pair_latency_ms=%.3f",
            gs,
            int(stats.get("tasks_completed", 0)),
            int(stats.get("views_loaded", 0)),
            int(stats.get("cache_hits_worker", 0)),
            int(stats.get("segment_static_completed", 0)),
            int(stats.get("tasks_failed", 0)),
            int(stats.get("tasks_dropped_stale", 0)),
            int(stats.get("tasks_dropped_queue_full", 0)),
            avg_ms,
            ol,
            och,
            int(stats.get("overlap_pairs_failed", 0)),
            avg_ol_ms,
        )

    def maybe_log_overlap_stats(self, global_step: int) -> None:
        interval = int(self._overlap_stats_log_interval_steps)
        if interval <= 0:
            return
        gs = int(global_step)
        if gs % interval != 0:
            return
        st = self._overlap_stats
        pq = float(st.get("pair_queries", 0.0))
        if pq <= 0:
            return
        hits = float(st.get("pair_cache_hits", 0.0))
        misses = float(st.get("pair_cache_misses", 0.0))
        miss_ms = float(st.get("pair_compute_miss_ms_sum", 0.0))
        wall_ms = float(st.get("pair_eval_wall_ms_sum", 0.0))
        avg_miss_ms = miss_ms / max(misses, 1.0)
        avg_wall_ms = wall_ms / max(pq, 1.0)
        logger.info(
            "overlap_stats global_step=%s pair_queries=%s pair_cache_hits=%s pair_cache_misses=%s "
            "avg_pair_compute_miss_ms=%.4f avg_pair_eval_wall_ms=%.4f src_rep_no_visible=%s candidate_eval_count=%s",
            gs,
            int(pq),
            int(hits),
            int(misses),
            avg_miss_ms,
            avg_wall_ms,
            int(st.get("src_rep_no_visible", 0.0)),
            int(st.get("candidate_eval_count", 0.0)),
        )
        for k in st:
            st[k] = 0.0

    def _preload_segment_static_redundant(self, scene_id: int, segment_id: int) -> bool:
        key = (int(scene_id), int(segment_id))
        cfg = self._preload_rtcfg
        pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
        max_test_cap = int(pixel_source_cfg.get("max_test_images", 0))
        tr_key = (int(scene_id), int(segment_id), max_test_cap)
        with self._lock:
            has_idx = key in self._segment_index_cache
            has_pc = key in self._segment_pointcloud_cache
            has_pose = key in self._segment_pose_cache
            has_tr = tr_key in self._test_image_refs_cache
        if cfg is not None and cfg.warm_segment_pointcloud and self.pointcloud_generator is not None:
            return bool(has_idx and has_pc and has_pose and has_tr)
        return bool(has_idx and has_pose and has_tr)

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

        with self._scene_load_coord_lock:
            if scene_id in self._scene_load_inflight:
                ev = self._scene_load_inflight[scene_id]
                is_owner = False
            else:
                ev = threading.Event()
                self._scene_load_inflight[scene_id] = ev
                is_owner = True
        if not is_owner:
            ev.wait(timeout=600.0)
            return self._ensure_scene_loaded(scene_id)
        try:
            with self._lock:
                if scene_id in self.train_scenes_cache:
                    return self.train_scenes_cache[scene_id]
                if scene_id in self.eval_scene_ids and scene_id in self.eval_scenes:
                    return self.eval_scenes[scene_id]
            if scene_id in self.eval_scene_ids:
                scene_data = self._load_and_prepare_scene(scene_id)
                if scene_data is not None:
                    with self._lock:
                        if scene_id in self.eval_scenes:
                            return self.eval_scenes[scene_id]
                        self.eval_scenes[scene_id] = scene_data
                return scene_data
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                with self._lock:
                    if scene_id in self.train_scenes_cache:
                        return self.train_scenes_cache[scene_id]
                    self.train_scenes_cache[scene_id] = scene_data
            return scene_data
        finally:
            with self._scene_load_coord_lock:
                self._scene_load_inflight.pop(scene_id, None)
            ev.set()

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
            # Clear inflight coordination without holding the main lock to avoid lock-order deadlocks.
            with self._segment_index_coord_lock:
                for k in list(self._segment_index_inflight.keys()):
                    if int(k[0]) == sid:
                        ev = self._segment_index_inflight.pop(k, None)
                        if ev is not None:
                            ev.set()
            with self._segment_pointcloud_coord_lock:
                for k in list(self._segment_pointcloud_inflight.keys()):
                    if int(k[0]) == sid:
                        ev = self._segment_pointcloud_inflight.pop(k, None)
                        if ev is not None:
                            ev.set()
            with self._view_load_coord_lock:
                for k in list(self._view_load_inflight.keys()):
                    if int(k[0]) == sid:
                        ev = self._view_load_inflight.pop(k, None)
                        if ev is not None:
                            ev.set()
            with self._view_pack_lock:
                stale_view_keys = [k for k in self._view_pack_cache if k[0] == sid]
                for k in stale_view_keys:
                    del self._view_pack_cache[k]
            with self._lock:
                super()._unload_scene(sid)
                stale_index_keys = [k for k in self._segment_index_cache if k[0] == sid]
                for k in stale_index_keys:
                    del self._segment_index_cache[k]
                stale_pair_keys = [k for k in self._pair_score_cache if k[0] == sid]
                for k in stale_pair_keys:
                    del self._pair_score_cache[k]
                for k in list(self._segment_pose_cache.keys()):
                    if int(k[0]) == sid:
                        del self._segment_pose_cache[k]
                for k in list(self._test_image_refs_cache.keys()):
                    if int(k[0]) == sid:
                        del self._test_image_refs_cache[k]
        finally:
            with self._scene_unloading_lock:
                self._scene_unloading.discard(sid)

    def _switch_to_next_scene(self) -> None:
        """Override base to avoid holding self._lock across unloads (preload can deadlock)."""
        with self._lock:
            if self.current_scene_index >= len(self.scene_training_queue):
                logger.warning("No more scenes in training queue")
                return
            current_scene_id = self.scene_training_queue[self.current_scene_index]

        self._unload_scene(current_scene_id)

        with self._lock:
            self.current_scene_index += 1
            if self.current_scene_index >= len(self.scene_training_queue):
                logger.info("All scenes in training queue have been processed")
                self._ensure_training_queue_ready()
                if self.current_scene_index >= len(self.scene_training_queue):
                    return
            self._ensure_training_queue_ready()

        self._preload_scenes()

    def mark_scene_completed(self, scene_id: int) -> None:
        with self._lock:
            if self.current_scene_index >= len(self.scene_training_queue) or len(self.scene_training_queue) == 0:
                logger.warning("No current scene to mark as completed")
                return
            current_scene_id = self.scene_training_queue[self.current_scene_index]
            if int(scene_id) != int(current_scene_id):
                logger.warning("Scene %s does not match current scene %s. Ignoring.", scene_id, current_scene_id)
                return
        self._switch_to_next_scene()

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

    def _materialize_view_pack_cache(
        self,
        key: Tuple[int, int, int, int],
        scene_dataset: DrivingDataset,
        ref_t: Tuple[int, int],
    ) -> None:
        cfg = self._preload_rtcfg
        if cfg is None or not cfg.enable_view_pack_cache:
            return
        with self._view_load_coord_lock:
            with self._view_pack_lock:
                if key in self._view_pack_cache:
                    return
            if key in self._view_load_inflight:
                ev = self._view_load_inflight[key]
                is_waiter = True
            else:
                ev = threading.Event()
                self._view_load_inflight[key] = ev
                is_waiter = False
        if is_waiter:
            ev.wait(timeout=600.0)
            return
        try:
            with self._view_pack_lock:
                if key in self._view_pack_cache:
                    return
            raw = self._load_view_from_image_ref(scene_dataset, ref_t)
            pin = pin_memory_from_cfg(cfg)
            lvp = dict_to_loaded_view_pack(raw, pin_memory=pin)
            with self._view_pack_lock:
                if key in self._view_pack_cache:
                    return
                self._evict_view_cache_if_needed_unlocked(key)
                self._view_pack_cache[key] = lvp
                self._view_pack_cache.move_to_end(key)
                self._trim_view_cache_per_scene_cap_unlocked(key[0])
        finally:
            with self._view_load_coord_lock:
                self._view_load_inflight.pop(key, None)
            ev.set()

    def _ensure_segment_pose_cached(
        self,
        scene_id: int,
        segment_id: int,
        scene_dataset: DrivingDataset,
        segment: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, int, str]:
        key = (int(scene_id), int(segment_id))
        with self._lock:
            ent = self._segment_pose_cache.get(key)
        if ent is not None:
            return (
                ent["segment_first_pose"],
                ent["world_to_seg0"],
                int(ent["segment_first_frame_idx"]),
                str(ent["segment_pose_source"]),
            )
        segment_first_pose, segment_first_frame_idx, segment_pose_source = self._get_segment_first_pose(
            scene_dataset=scene_dataset,
            segment=segment,
            segment_id=int(segment_id),
        )
        segment_first_pose = segment_first_pose.to(device=self.device, dtype=torch.float32)
        # Materialize: cam_to_world slices can be lazy/subclass tensors; linalg.inv may error with
        # "lazy wrapper should be called at most once" if the input is not a plain dense tensor.
        segment_first_pose = segment_first_pose.contiguous().clone()
        try:
            world_to_seg0 = torch.linalg.inv(segment_first_pose)
        except RuntimeError as exc:
            raise ValueError(
                f"Segment {segment_id} first pose is non-invertible; cannot build segment coordinate transform."
            ) from exc
        with self._lock:
            if key not in self._segment_pose_cache:
                self._segment_pose_cache[key] = {
                    "segment_first_pose": segment_first_pose,
                    "world_to_seg0": world_to_seg0,
                    "segment_first_frame_idx": int(segment_first_frame_idx),
                    "segment_pose_source": str(segment_pose_source),
                }
        return segment_first_pose, world_to_seg0, int(segment_first_frame_idx), str(segment_pose_source)

    def _ensure_segment_pointcloud_cached(
        self,
        scene_id: int,
        segment_id: int,
        segment_first_pose: Tensor,
    ) -> Any:
        if self.pointcloud_generator is None:
            return None
        pc_key = (int(scene_id), int(segment_id))
        with self._lock:
            if pc_key in self._segment_pointcloud_cache:
                return self._segment_pointcloud_cache[pc_key]
        with self._segment_pointcloud_coord_lock:
            with self._lock:
                if pc_key in self._segment_pointcloud_cache:
                    return self._segment_pointcloud_cache[pc_key]
            if pc_key in self._segment_pointcloud_inflight:
                ev = self._segment_pointcloud_inflight[pc_key]
                is_waiter = True
            else:
                ev = threading.Event()
                self._segment_pointcloud_inflight[pc_key] = ev
                is_waiter = False
        if is_waiter:
            ev.wait(timeout=600.0)
            with self._lock:
                return self._segment_pointcloud_cache.get(pc_key)
        try:
            pc = self.pointcloud_generator.generate_pointcloud(
                dataset=self,
                scene_id=scene_id,
                segment_id=segment_id,
                segment_first_pose=segment_first_pose,
            )
            with self._lock:
                if pc_key not in self._segment_pointcloud_cache:
                    self._segment_pointcloud_cache[pc_key] = pc
            with self._lock:
                return self._segment_pointcloud_cache.get(pc_key)
        finally:
            with self._segment_pointcloud_coord_lock:
                self._segment_pointcloud_inflight.pop(pc_key, None)
            ev.set()

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
            self._materialize_view_pack_cache(key, scene_dataset, tuple(image_ref))
            with self._view_pack_lock:
                if key in self._view_pack_cache:
                    return "loaded"
            return "failed"
        except Exception as exc:
            logger.debug("preload worker _preload_worker_load_view_pack: %s", exc, exc_info=True)
            return "failed"

    def _preload_worker_segment_static(self, scene_id: int, segment_id: int, meta: Dict[str, Any]) -> str:
        cfg = self._preload_rtcfg
        if cfg is None or not cfg.warm_segment_static:
            return "skipped"
        sid, seg = int(scene_id), int(segment_id)
        try:
            if self._preload_should_abort_for_unload(sid):
                return "failed"
            self.get_segment_index(sid, seg)
            self.resolve_test_image_refs_deterministic(sid, seg)
            scene_data = self._ensure_scene_loaded(sid)
            if scene_data is None:
                return "failed"
            sidx = self.get_segment_index(sid, seg)
            segment = scene_data["segments"][int(sidx.segment_id)]
            scene_dataset = scene_data["dataset"]
            segment_first_pose, _, _, _ = self._ensure_segment_pose_cached(sid, seg, scene_dataset, segment)
            if cfg.warm_segment_pointcloud and self.pointcloud_generator is not None:
                self._ensure_segment_pointcloud_cached(sid, seg, segment_first_pose)
            return "loaded"
        except Exception as exc:
            logger.debug("_preload_worker_segment_static: %s", exc, exc_info=True)
            return "failed"

    def _preload_worker_overlap_pair(
        self,
        scene_id: int,
        segment_id: int,
        src_rep_image_ref: ImageRef,
        tgt_rep_image_ref: ImageRef,
        *,
        mode: str,
        point_sample_size: int,
        meta: Dict[str, Any],
    ) -> str:
        sid, seg = int(scene_id), int(segment_id)
        src_t = (int(src_rep_image_ref[0]), int(src_rep_image_ref[1]))
        tgt_t = (int(tgt_rep_image_ref[0]), int(tgt_rep_image_ref[1]))
        mode_s = str(mode)
        pss = int(point_sample_size)
        try:
            if self._preload_should_abort_for_unload(sid):
                return "failed"
            if self.is_pair_score_cached(sid, seg, src_t, tgt_t, mode_s, pss):
                return "cache_hit"
            self.get_segment_index(sid, seg)
            scene_data = self._ensure_scene_loaded(sid)
            if scene_data is None:
                return "failed"
            sidx = self.get_segment_index(sid, seg)
            segment = scene_data["segments"][int(sidx.segment_id)]
            scene_dataset = scene_data["dataset"]
            segment_first_pose, _, _, _ = self._ensure_segment_pose_cached(sid, seg, scene_dataset, segment)
            if self.pointcloud_generator is not None:
                self._ensure_segment_pointcloud_cached(sid, seg, segment_first_pose)
            self.get_or_compute_pair_score(
                sid,
                seg,
                src_t,
                tgt_t,
                mode=mode_s,
                point_sample_size=pss,
                account_runtime_stats=False,
            )
            return "loaded"
        except Exception as exc:
            logger.debug("_preload_worker_overlap_pair: %s", exc, exc_info=True)
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
        base_meta: Dict[str, Any] = {
            "epoch_idx": int(epoch_idx),
            "global_step": int(global_step),
            "block_idx_global": int(block_idx_global),
            "hint_scope": str(hint_scope),
        }
        overlap_pairs_raw = hint.get("future_overlap_pairs")
        overlap_pairs: List[Dict[str, Any]] = list(overlap_pairs_raw) if overlap_pairs_raw else []
        overlap_meta = hint.get("overlap_meta")
        if hint_scope == "next_block_exact":
            if not self._preload_rtcfg.warm_next_block_exact:
                return
            if self._preload_rtcfg.warm_segment_static:
                self._preload_manager.submit_segment_static(
                    PRIORITY_SEGMENT_STATIC, scene_id, segment_id, meta=base_meta
                )
            for ref in refs:
                self._preload_manager.submit_image_ref(
                    PRIORITY_NEXT_BLOCK_EXACT, scene_id, segment_id, ref, meta=base_meta
                )
            if include_test and self._preload_rtcfg.warm_test_refs:
                for ref in self.resolve_test_image_refs_deterministic(scene_id, segment_id):
                    self._preload_manager.submit_image_ref(
                        PRIORITY_TEST_REFS, scene_id, segment_id, ref, meta=base_meta
                    )
            if (
                self._preload_rtcfg.warm_overlap_pairs_next_block_exact
                and overlap_meta is not None
                and str(overlap_meta.get("mode")) == "pointcloud_topk"
                and overlap_pairs
            ):
                pss = int(overlap_meta["point_sample_size"])
                for p in overlap_pairs:
                    sr = p["src_rep_image_ref"]
                    tr = p["tgt_rep_image_ref"]
                    src_ref = (int(sr[0]), int(sr[1]))
                    tgt_ref = (int(tr[0]), int(tr[1]))
                    self._preload_manager.submit_overlap_pair(
                        PRIORITY_NEXT_BLOCK_EXACT,
                        scene_id,
                        segment_id,
                        src_ref,
                        tgt_ref,
                        mode="pointcloud_topk",
                        point_sample_size=pss,
                        meta=dict(base_meta),
                    )
        elif hint_scope == "episode_source_superset":
            if not self._preload_rtcfg.warm_episode_source_superset:
                return
            if self._preload_rtcfg.warm_segment_static:
                self._preload_manager.submit_segment_static(
                    PRIORITY_SEGMENT_STATIC, scene_id, segment_id, meta=base_meta
                )
            for ref in refs:
                self._preload_manager.submit_image_ref(
                    PRIORITY_EPISODE_SUPERSET, scene_id, segment_id, ref, meta=base_meta
                )
            if (
                self._preload_rtcfg.warm_overlap_pairs_episode_superset
                and overlap_meta is not None
                and str(overlap_meta.get("mode")) == "pointcloud_topk"
                and overlap_pairs
            ):
                pss = int(overlap_meta["point_sample_size"])
                for p in overlap_pairs:
                    sr = p["src_rep_image_ref"]
                    tr = p["tgt_rep_image_ref"]
                    src_ref = (int(sr[0]), int(sr[1]))
                    tgt_ref = (int(tr[0]), int(tr[1]))
                    self._preload_manager.submit_overlap_pair(
                        PRIORITY_EPISODE_SUPERSET,
                        scene_id,
                        segment_id,
                        src_ref,
                        tgt_ref,
                        mode="pointcloud_topk",
                        point_sample_size=pss,
                        meta=dict(base_meta),
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

        self._materialize_view_pack_cache(key, scene_dataset, ref_t)
        with self._view_pack_lock:
            if key not in self._view_pack_cache:
                raise RuntimeError(f"view pack cache still empty after materialize: key={key!r}")
            return loaded_view_pack_to_device(self._view_pack_cache[key], self.device)

    def _get_view_geometry_from_image_ref(
        self,
        scene_id: int,
        segment_id: int,
        scene_dataset: DrivingDataset,
        image_ref: ImageRef,
        *,
        world_to_seg0_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Extrinsic / intrinsic / image size for overlap scoring only.
        Reuses ``_view_pack_cache`` without ``loaded_view_pack_to_device`` (no GPU pack materialization).
        """
        ref_t = (int(image_ref[0]), int(image_ref[1]))
        key = (int(scene_id), int(segment_id), ref_t[0], ref_t[1])
        cfg = self._preload_rtcfg

        def _from_pack_tensors(ext_t: Tensor, intr_t: Tensor, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
            ext_np = ext_t.detach().cpu().numpy().astype(np.float64)
            c2w_seg0 = world_to_seg0_np @ ext_np
            intr_cpu = intr_t.detach().cpu().numpy().astype(np.float64)
            if intr_cpu.shape == (4, 4):
                K = intr_cpu[:3, :3]
            elif intr_cpu.shape == (3, 3):
                K = intr_cpu
            else:
                raise ValueError(f"unexpected intrinsic shape {intr_cpu.shape}")
            return c2w_seg0, K, int(h), int(w)

        if cfg is None or not cfg.enable_view_pack_cache:
            pack = self._load_view_from_image_ref(scene_dataset, ref_t)
            img = pack["image"]
            H, Wim = int(img.shape[0]), int(img.shape[1])
            ext_t = self._to_4x4_tensor(pack["extrinsic"]).to(device=self.device, dtype=torch.float64)
            intr = pack["intrinsic"]
            intr_t = intr if isinstance(intr, torch.Tensor) else torch.as_tensor(intr, device=self.device)
            return _from_pack_tensors(ext_t, intr_t, H, Wim)

        lvp: Optional[LoadedViewPack] = None
        with self._view_pack_lock:
            ent = self._view_pack_cache.get(key)
            if ent is not None:
                self._view_pack_cache.move_to_end(key)
                lvp = ent

        if lvp is None:
            self._materialize_view_pack_cache(key, scene_dataset, ref_t)
            with self._view_pack_lock:
                lvp = self._view_pack_cache.get(key)
            if lvp is None:
                raise RuntimeError(f"view pack cache still empty after materialize: key={key!r}")

        H, Wim = int(lvp.image.shape[0]), int(lvp.image.shape[1])
        ext_t = lvp.extrinsic
        if not isinstance(ext_t, torch.Tensor):
            ext_t = torch.as_tensor(ext_t)
        ext_t = ext_t.to(dtype=torch.float64)
        intr_t = lvp.intrinsic
        if not isinstance(intr_t, torch.Tensor):
            intr_t = torch.as_tensor(intr_t)
        intr_t = intr_t.to(dtype=torch.float64)
        return _from_pack_tensors(ext_t, intr_t, H, Wim)

    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndex:
        key = (int(scene_id), int(segment_id))
        with self._lock:
            if key in self._segment_index_cache:
                return self._segment_index_cache[key]
        with self._segment_index_coord_lock:
            if key in self._segment_index_inflight:
                ev = self._segment_index_inflight[key]
                is_waiter = True
            else:
                ev = threading.Event()
                self._segment_index_inflight[key] = ev
                is_waiter = False
        if is_waiter:
            ev.wait(timeout=600.0)
            return self.get_segment_index(scene_id, segment_id)
        try:
            with self._lock:
                if key in self._segment_index_cache:
                    return self._segment_index_cache[key]
            scene_data = self._ensure_scene_loaded(int(scene_id))
            if scene_data is None:
                raise ValueError(f"Scene {scene_id} cannot be loaded")
            segments = scene_data.get("segments", [])
            if int(segment_id) < 0 or int(segment_id) >= len(segments):
                raise ValueError(f"segment_id={segment_id} out of range for scene {scene_id}")
            idx = _build_segment_index_dict(int(scene_id), int(segment_id), scene_data)
            with self._lock:
                if key in self._segment_index_cache:
                    return self._segment_index_cache[key]
                self._segment_index_cache[key] = idx
            return idx
        finally:
            with self._segment_index_coord_lock:
                self._segment_index_inflight.pop(key, None)
            ev.set()

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
        *,
        point_sample_size: Optional[int] = None,
        counts_out: Optional[Dict[str, int]] = None,
        account_runtime_stats: bool = True,
    ) -> Optional[float]:
        if mode == "none":
            return None
        if mode != "pointcloud_topk":
            raise ValueError(f"get_or_compute_pair_score: unsupported mode={mode!r}")
        if point_sample_size is None:
            raise ValueError("get_or_compute_pair_score(mode='pointcloud_topk') requires point_sample_size")
        if int(point_sample_size) < 1:
            raise ValueError(f"point_sample_size must be >= 1, got {point_sample_size}")

        t_wall0 = time.perf_counter()
        sid, seg = int(scene_id), int(segment_id)
        src_t = (int(src[0]), int(src[1]))
        tgt_t = (int(tgt[0]), int(tgt[1]))
        pss = int(point_sample_size)
        cache_key = (sid, seg, src_t, tgt_t, str(mode), pss)
        with self._lock:
            cached = self._pair_score_cache.get(cache_key)
        if cached is not None:
            score, n_a, n_b, n_ab = cached
            wall_ms = (time.perf_counter() - t_wall0) * 1000.0
            if account_runtime_stats:
                self._overlap_stats["pair_queries"] += 1.0
                self._overlap_stats["pair_cache_hits"] += 1.0
                self._overlap_stats["pair_eval_wall_ms_sum"] += wall_ms
                self._overlap_stats["candidate_eval_count"] += 1.0
            if counts_out is not None:
                counts_out.clear()
                counts_out.update({"n_a": int(n_a), "n_b": int(n_b), "n_ab": int(n_ab)})
            return float(score)

        if self.pointcloud_generator is None:
            raise ValueError("get_or_compute_pair_score(pointcloud_topk) requires dataset.pointcloud_generator")

        t_miss0 = time.perf_counter()
        scene_data = self._ensure_scene_loaded(sid)
        if scene_data is None:
            raise ValueError(f"Scene {sid} cannot be loaded for pair score")
        segments = scene_data.get("segments", [])
        if seg < 0 or seg >= len(segments):
            raise ValueError(f"segment_id={seg} out of range for scene {sid}")
        segment = segments[seg]
        scene_dataset = scene_data["dataset"]

        segment_first_pose, world_to_seg0, _, _ = self._ensure_segment_pose_cached(
            sid, seg, scene_dataset, segment
        )
        world_to_seg0_np = world_to_seg0.detach().cpu().numpy().astype(np.float64)

        pc_any = self._ensure_segment_pointcloud_cached(sid, seg, segment_first_pose)
        if pc_any is None or not isinstance(pc_any, dict):
            raise ValueError(f"segment pointcloud missing for scene={sid} segment={seg}")
        bg = pc_any.get("background")
        if bg is None:
            raise ValueError(f"pointcloud has no 'background' for scene={sid} segment={seg}")
        bg_np = np.asarray(bg, dtype=np.float64)
        if bg_np.ndim != 2 or bg_np.shape[1] < 3:
            raise ValueError(
                f"background pointcloud must be 2D with >=3 columns, got shape={getattr(bg_np, 'shape', None)}"
            )
        xyz = bg_np[:, :3]
        n_pts = int(xyz.shape[0])
        if n_pts == 0:
            score = 0.0
            n_a = n_b = n_ab = 0
            with self._lock:
                self._pair_score_cache[cache_key] = (float(score), n_a, n_b, n_ab)
            miss_ms = (time.perf_counter() - t_miss0) * 1000.0
            wall_ms = (time.perf_counter() - t_wall0) * 1000.0
            if account_runtime_stats:
                self._overlap_stats["pair_queries"] += 1.0
                self._overlap_stats["pair_cache_misses"] += 1.0
                self._overlap_stats["pair_compute_miss_ms_sum"] += miss_ms
                self._overlap_stats["pair_eval_wall_ms_sum"] += wall_ms
                self._overlap_stats["candidate_eval_count"] += 1.0
            if counts_out is not None:
                counts_out.clear()
                counts_out.update({"n_a": n_a, "n_b": n_b, "n_ab": n_ab})
            return float(score)

        m_take = min(pss, n_pts)
        seed = (sid * 1_000_003 + seg) * 1_000_003 + pss
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        if m_take < n_pts:
            idx = rng.choice(n_pts, size=m_take, replace=False)
            pts = xyz[idx].astype(np.float64, copy=False)
        else:
            pts = xyz.astype(np.float64, copy=False)

        self.validate_image_ref(sid, seg, src_t, purpose="train")
        self.validate_image_ref(sid, seg, tgt_t, purpose="train")

        c2w_a, Ka, Ha, Wa = self._get_view_geometry_from_image_ref(
            sid, seg, scene_dataset, src_t, world_to_seg0_np=world_to_seg0_np
        )
        c2w_b, Kb, Hb, Wb = self._get_view_geometry_from_image_ref(
            sid, seg, scene_dataset, tgt_t, world_to_seg0_np=world_to_seg0_np
        )

        va = _visibility_mask_seg0(pts, c2w_a, Ka, Ha, Wa)
        vb = _visibility_mask_seg0(pts, c2w_b, Kb, Hb, Wb)
        n_a = int(np.sum(va))
        n_b = int(np.sum(vb))
        n_ab = int(np.sum(va & vb))
        if n_a == 0:
            score = 0.0
            if account_runtime_stats:
                self._overlap_stats["src_rep_no_visible"] += 1.0
        else:
            score = float(n_ab) / float(n_a)

        with self._lock:
            self._pair_score_cache[cache_key] = (float(score), n_a, n_b, n_ab)

        miss_ms = (time.perf_counter() - t_miss0) * 1000.0
        wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        if account_runtime_stats:
            self._overlap_stats["pair_queries"] += 1.0
            self._overlap_stats["pair_cache_misses"] += 1.0
            self._overlap_stats["pair_compute_miss_ms_sum"] += miss_ms
            self._overlap_stats["pair_eval_wall_ms_sum"] += wall_ms
            self._overlap_stats["candidate_eval_count"] += 1.0

        if counts_out is not None:
            counts_out.clear()
            counts_out.update({"n_a": n_a, "n_b": n_b, "n_ab": n_ab})
        return float(score)

    def is_pair_score_cached(
        self,
        scene_id: int,
        segment_id: int,
        src: ImageRef,
        tgt: ImageRef,
        mode: str,
        point_sample_size: int,
    ) -> bool:
        cache_key = (
            int(scene_id),
            int(segment_id),
            (int(src[0]), int(src[1])),
            (int(tgt[0]), int(tgt[1])),
            str(mode),
            int(point_sample_size),
        )
        with self._lock:
            return cache_key in self._pair_score_cache

    def build_preload_hint(
        self,
        scene_id: int,
        segment_id: int,
        future_image_refs: List[ImageRef],
        future_overlap_pairs: Optional[List[Dict[str, Any]]] = None,
        overlap_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        frames = sorted({int(r[0]) for r in future_image_refs})
        cams = sorted({int(r[1]) for r in future_image_refs})
        hint_version = 2 if (
            (future_overlap_pairs is not None and len(future_overlap_pairs) > 0)
            or (overlap_meta is not None)
        ) else 1
        out: Dict[str, Any] = {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "future_image_refs": list(future_image_refs),
            "unique_frame_indices": frames,
            "unique_cam_indices": cams,
            "hint_version": int(hint_version),
        }
        if future_overlap_pairs is not None:
            out["future_overlap_pairs"] = list(future_overlap_pairs)
        if overlap_meta is not None:
            out["overlap_meta"] = dict(overlap_meta)
        return out

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

        segment_first_pose, world_to_seg0, segment_first_frame_idx, segment_pose_source = self._ensure_segment_pose_cached(
            scene_id, segment_id, scene_dataset, segment
        )

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
            pointcloud = self._ensure_segment_pointcloud_cached(
                int(scene_id), int(segment_id), segment_first_pose
            )

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

        Note: ``data.pixel_source.max_test_images`` caps the number of **test frame indices** selected from
        ``test_frame_indices``; total image refs are ``len(selected_frames) * num_cams`` (not ``max_test_images``).
        """
        pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
        max_test_cap = int(pixel_source_cfg.get("max_test_images", 0))
        cache_key = (int(scene_id), int(segment_id), max_test_cap)
        with self._lock:
            cached = self._test_image_refs_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        sidx = self.get_segment_index(scene_id, segment_id)
        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment = scene_data["segments"][int(sidx.segment_id)]
        segment_test_frames = sorted(int(f) for f in segment.get("test_frame_indices", []))
        if len(segment_test_frames) == 0:
            with self._lock:
                if cache_key not in self._test_image_refs_cache:
                    self._test_image_refs_cache[cache_key] = []
            return []
        if max_test_cap > 0 and len(segment_test_frames) > max_test_cap:
            selected = segment_test_frames[:max_test_cap]
        else:
            selected = list(segment_test_frames)
        num_cams = int(scene_data["dataset"].num_cams)
        refs: List[ImageRef] = []
        for frame_idx in selected:
            for cam_idx in range(num_cams):
                ref: ImageRef = (int(frame_idx), int(cam_idx))
                self.validate_image_ref(scene_id, segment_id, ref, purpose="test")
                refs.append(ref)
        with self._lock:
            if cache_key not in self._test_image_refs_cache:
                self._test_image_refs_cache[cache_key] = list(refs)
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

    def get_segment_eval_batch_from_image_refs(
        self,
        request: EvalRequestV3,
    ) -> Dict[str, Any]:
        if len(request.eval_image_refs) == 0:
            raise ValueError("eval_image_refs must not be empty")
        raw = self._assemble_segment_batch_from_image_refs(
            request.scene_id,
            request.segment_id,
            [request.source_image_ref],
            request.eval_image_refs,
            include_test=False,
            test_image_refs=None,
            enforce_target0_equals_source=False,
        )
        out: Dict[str, Any] = {
            "scene_id": raw["scene_id"],
            "scene_folder_name": raw["scene_folder_name"],
            "segment_id": raw["segment_id"],
            "aabb": raw["aabb"],
            "segment_first_pose": raw["segment_first_pose"],
            "segment_first_frame_idx": raw["segment_first_frame_idx"],
            "segment_first_pose_source": raw["segment_first_pose_source"],
            "source": raw["source"],
            "eval": raw["target"],
            "request_meta": dict(raw.get("request_meta") or {}),
            "index_meta": dict(raw.get("index_meta") or {}),
            "keyframe_info": dict(raw.get("keyframe_info") or {}),
        }
        out["request_meta"]["eval_image_refs"] = [tuple(r) for r in request.eval_image_refs]
        out["request_meta"]["assembly_mode"] = "eval_image_ref"
        out["request_meta"].pop("target_image_refs", None)
        if "pointcloud" in raw:
            out["pointcloud"] = raw["pointcloud"]
        if "dynamic_info" in raw:
            out["dynamic_info"] = raw["dynamic_info"]
        return out

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
        overlap_point_sample_size: Optional[int] = None,
        overlap_candidate_frame_policy: Optional[str] = None,
        overlap_score_type: Optional[str] = None,
        overlap_min: Optional[float] = None,
        temporal_neighbor_pool: str = "none",
        temporal_neighbor_max_ring: Optional[int] = None,
        temporal_neighbor_cams: Optional[List[int]] = None,
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
            overlap_point_sample_size=overlap_point_sample_size,
            overlap_candidate_frame_policy=overlap_candidate_frame_policy,
            overlap_score_type=overlap_score_type,
            overlap_min=overlap_min,
            temporal_neighbor_pool=temporal_neighbor_pool,
            temporal_neighbor_max_ring=temporal_neighbor_max_ring,
            temporal_neighbor_cams=temporal_neighbor_cams,
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
        overlap_point_sample_size: Optional[int] = None,
        overlap_candidate_frame_policy: Optional[str] = None,
        overlap_score_type: Optional[str] = None,
        overlap_min: Optional[float] = None,
        temporal_neighbor_pool: str = "none",
        temporal_neighbor_max_ring: Optional[int] = None,
        temporal_neighbor_cams: Optional[List[int]] = None,
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
        om = str(overlap_mode)
        if om not in ("none", "pointcloud_topk"):
            raise ValueError(f"TrainSchedulerV4: unsupported overlap_mode={overlap_mode!r}")
        if om == "pointcloud_topk":
            if overlap_point_sample_size is None:
                raise ValueError("TrainSchedulerV4: overlap_point_sample_size is required when overlap_mode=pointcloud_topk")
            if int(overlap_point_sample_size) < 1:
                raise ValueError(f"overlap_point_sample_size must be >= 1, got {overlap_point_sample_size}")
            if overlap_candidate_frame_policy is None:
                raise ValueError("overlap_candidate_frame_policy is required when overlap_mode=pointcloud_topk")
            if str(overlap_candidate_frame_policy) != "middle":
                raise ValueError(
                    f"overlap_candidate_frame_policy must be 'middle', got {overlap_candidate_frame_policy!r}"
                )
            if overlap_score_type is None:
                raise ValueError("overlap_score_type is required when overlap_mode=pointcloud_topk")
            if str(overlap_score_type) != "nab_over_na":
                raise ValueError(f"overlap_score_type must be 'nab_over_na', got {overlap_score_type!r}")
            if overlap_min is None:
                raise ValueError("overlap_min is required when overlap_mode=pointcloud_topk")
            if getattr(dataset, "pointcloud_generator", None) is None:
                raise ValueError("TrainSchedulerV4: overlap_mode=pointcloud_topk requires dataset.pointcloud_generator")

        tnp = str(temporal_neighbor_pool)
        if tnp not in ("none", "ring"):
            raise ValueError(f"TrainSchedulerV4: temporal_neighbor_pool must be 'none' or 'ring', got {temporal_neighbor_pool!r}")
        if tnp == "ring":
            if temporal_neighbor_max_ring is None:
                raise ValueError("TrainSchedulerV4: temporal_neighbor_max_ring is required when temporal_neighbor_pool=ring")
            if int(temporal_neighbor_max_ring) < 1:
                raise ValueError(f"temporal_neighbor_max_ring must be >= 1, got {temporal_neighbor_max_ring}")
        elif temporal_neighbor_max_ring is not None:
            raise ValueError("TrainSchedulerV4: temporal_neighbor_max_ring must be omitted when temporal_neighbor_pool=none")
        if temporal_neighbor_cams is not None:
            if len(temporal_neighbor_cams) == 0:
                raise ValueError("TrainSchedulerV4: temporal_neighbor_cams must be null or a non-empty list of ints")

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
        self.overlap_point_sample_size = (
            int(overlap_point_sample_size) if overlap_point_sample_size is not None else None
        )
        self.overlap_candidate_frame_policy = overlap_candidate_frame_policy
        self.overlap_score_type = overlap_score_type
        self.overlap_min = float(overlap_min) if overlap_min is not None else None
        self.temporal_neighbor_pool = tnp
        self.temporal_neighbor_max_ring = (
            int(temporal_neighbor_max_ring) if temporal_neighbor_max_ring is not None else None
        )
        self.temporal_neighbor_cams = (
            [int(x) for x in temporal_neighbor_cams] if temporal_neighbor_cams is not None else None
        )

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
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

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

    def _extra_target_keyframe_pool(
        self,
        sidx: SegmentIndex,
        kf_src: int,
        episode_window: List[int],
        pos: Dict[int, int],
        pos_src: int,
    ) -> List[int]:
        """Ordered unique keyframes != kf_src: episode window first, then segment-only (if expand)."""
        window_kfs = {int(k) for k in episode_window}
        part1 = [int(k) for k in episode_window if int(k) != int(kf_src)]
        if self.prefer_nearby_keyframes:
            part1.sort(key=lambda k: abs(int(pos[k]) - pos_src))
        part2: List[int] = []
        if self.fallback_expand_to_segment:
            part2 = [
                int(k)
                for k in sidx.keyframe_indices
                if int(k) != int(kf_src) and int(k) not in window_kfs
            ]
            if self.prefer_nearby_keyframes:
                part2.sort(key=lambda k: abs(int(pos[k]) - pos_src))
        seen: Set[int] = set()
        out: List[int] = []
        for k in part1 + part2:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _use_temporal_neighbor_ring(self, cam_src: int) -> bool:
        if self.temporal_neighbor_pool != "ring":
            return False
        if self.temporal_neighbor_cams is None:
            return True
        return int(cam_src) in self.temporal_neighbor_cams

    def _ring_keyframes_in_pool(
        self,
        full_pool: List[int],
        pos: Dict[int, int],
        pos_src: int,
        R: int,
    ) -> Set[int]:
        r = int(R)
        pos_s = int(pos_src)
        out: Set[int] = set()
        for kf in full_pool:
            if abs(int(pos[int(kf)]) - pos_s) <= r:
                out.add(int(kf))
        return out

    def _pointcloud_topk_pick_from_scored_rows(
        self,
        scored_rows: List[Tuple[int, int, float]],
        extra_needed: int,
        thr: float,
    ) -> Optional[Tuple[List[int], str, List[int]]]:
        """Returns (picked_kfs, policy, above_threshold_keyframes) or None if pick is impossible."""
        if not scored_rows:
            return None
        rows = sorted(scored_rows, key=lambda r: (-r[2], r[0]))
        ranked_kfs = [int(r[1]) for r in rows]
        above_kfs = [int(r[1]) for r in rows if float(r[2]) > thr]
        if len(above_kfs) >= extra_needed:
            return (
                list(random.sample(above_kfs, int(extra_needed))),
                "random_above_min",
                above_kfs,
            )
        if len(ranked_kfs) >= extra_needed:
            return ranked_kfs[:extra_needed], "topk_fallback", above_kfs
        if not self.fallback_with_replacement:
            return None
        if len(ranked_kfs) == 0:
            return None
        picked: List[int] = []
        for j in range(extra_needed):
            picked.append(ranked_kfs[j % len(ranked_kfs)])
        return picked, "topk_fallback", above_kfs

    def _sample_target_image_refs(
        self,
        sidx: SegmentIndex,
        source_image_ref: ImageRef,
        episode_window: List[int],
    ) -> Tuple[List[ImageRef], Optional[Dict[str, Any]]]:
        f_src, cam_src = int(source_image_ref[0]), int(source_image_ref[1])
        kf_src = int(sidx.frame_to_keyframe[f_src])
        pos = self._kf_positions(sidx)
        if kf_src not in pos:
            raise ValueError(f"source keyframe {kf_src} not in segment keyframe_indices")
        pos_src = int(pos[kf_src])

        refs: List[ImageRef] = [(f_src, cam_src)]
        extra_needed = self.total_target_images - 1
        if extra_needed <= 0:
            return refs, None

        if self.overlap_mode == "none":

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

            def _legacy_none_pick() -> List[int]:
                picked: List[int] = []
                for kf in _sorted_window_others():
                    if len(picked) >= extra_needed:
                        break
                    picked.append(kf)

                if len(picked) < extra_needed and self.fallback_expand_to_segment:
                    for kf in _sorted_segment_others():
                        if len(picked) >= extra_needed:
                            break
                        if kf in picked:
                            continue
                        picked.append(kf)

                if len(picked) < extra_needed:
                    if not self.fallback_with_replacement:
                        raise ValueError(
                            f"Not enough distinct keyframes for {extra_needed} extra targets "
                            f"(scene={sidx.scene_id} segment={sidx.segment_id})"
                        )
                    pool_kf = [int(k) for k in sidx.keyframe_indices if int(k) != kf_src]
                    if len(pool_kf) == 0:
                        raise ValueError(
                            f"No non-source keyframes for extra targets (scene={sidx.scene_id} segment={sidx.segment_id})"
                        )
                    while len(picked) < extra_needed:
                        picked.append(int(random.choice(pool_kf)))
                return picked[:extra_needed]

            temporal_meta: Optional[Dict[str, Any]] = None
            if self._use_temporal_neighbor_ring(cam_src) and self.temporal_neighbor_max_ring is not None:
                full_pool = self._extra_target_keyframe_pool(sidx, kf_src, episode_window, pos, pos_src)
                picked_kfs: Optional[List[int]] = None
                ring_eff: Optional[int] = None
                for R in range(1, int(self.temporal_neighbor_max_ring) + 1):
                    ring_set = self._ring_keyframes_in_pool(full_pool, pos, pos_src, R)
                    ring_ordered = [int(kf) for kf in full_pool if int(kf) in ring_set]
                    picked_try: List[int] = []
                    for kf in ring_ordered:
                        if len(picked_try) >= extra_needed:
                            break
                        if kf not in picked_try:
                            picked_try.append(kf)
                    if len(picked_try) >= extra_needed:
                        picked_kfs = picked_try[:extra_needed]
                        ring_eff = R
                        break
                    if len(ring_ordered) < extra_needed and not self.fallback_with_replacement:
                        continue
                    if len(ring_ordered) >= 1 and self.fallback_with_replacement:
                        picked_rep: List[int] = []
                        while len(picked_rep) < extra_needed:
                            picked_rep.append(int(random.choice(ring_ordered)))
                        picked_kfs = picked_rep
                        ring_eff = R
                        break
                if picked_kfs is None:
                    picked_kfs = _legacy_none_pick()
                    temporal_meta = {
                        "temporal_neighbor_pool": "ring",
                        "temporal_neighbor_ring_effective": None,
                        "temporal_neighbor_fallback_full_pool": True,
                    }
                else:
                    temporal_meta = {
                        "temporal_neighbor_pool": "ring",
                        "temporal_neighbor_ring_effective": int(ring_eff) if ring_eff is not None else None,
                        "temporal_neighbor_fallback_full_pool": False,
                    }
            else:
                picked_kfs = _legacy_none_pick()

            for kf in picked_kfs:
                frame_tgt = int(random.choice(sidx.keyframe_to_frames[int(kf)]))
                refs.append((frame_tgt, cam_src))

            return refs, temporal_meta

        # overlap_mode == "pointcloud_topk"
        assert self.overlap_point_sample_size is not None
        full_pool = self._extra_target_keyframe_pool(sidx, kf_src, episode_window, pos, pos_src)
        if len(full_pool) == 0:
            raise ValueError(
                f"No candidate keyframes for overlap (scene={sidx.scene_id} segment={sidx.segment_id})"
            )

        rep_src_frame = representative_frame_for_keyframe(sidx, kf_src)
        rep_src: ImageRef = (rep_src_frame, cam_src)
        pss = int(self.overlap_point_sample_size)
        sid, seg = int(sidx.scene_id), int(sidx.segment_id)
        assert self.overlap_min is not None
        thr = float(self.overlap_min)

        scored_by_kf: Dict[int, Tuple[int, float, Dict[str, int]]] = {}
        cache_hits = 0
        cache_misses = 0
        pair_compute_miss_time_ms_total = 0.0
        pair_eval_wall_time_ms_total = 0.0

        def score_kf(pi: int, kf: int) -> None:
            nonlocal cache_hits, cache_misses, pair_compute_miss_time_ms_total, pair_eval_wall_time_ms_total
            rep_tgt = (representative_frame_for_keyframe(sidx, int(kf)), cam_src)
            was_cached = self.dataset.is_pair_score_cached(
                sid, seg, rep_src, rep_tgt, "pointcloud_topk", pss
            )
            t0 = time.perf_counter()
            cnts: Dict[str, int] = {}
            score = self.dataset.get_or_compute_pair_score(
                sid,
                seg,
                rep_src,
                rep_tgt,
                mode="pointcloud_topk",
                point_sample_size=pss,
                counts_out=cnts,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            pair_eval_wall_time_ms_total += dt_ms
            if score is None:
                raise ValueError("get_or_compute_pair_score(pointcloud_topk) returned None")
            scored_by_kf[int(kf)] = (int(pi), float(score), dict(cnts))
            if was_cached:
                cache_hits += 1
            else:
                cache_misses += 1
                pair_compute_miss_time_ms_total += dt_ms

        ring_effective: Optional[int] = None
        fallback_full = False
        picked_kfs: List[int]
        extra_target_pick_policy: str
        above_kfs: List[int]

        use_ring = self._use_temporal_neighbor_ring(cam_src) and self.temporal_neighbor_max_ring is not None
        pick_result: Optional[Tuple[List[int], str, List[int]]] = None
        if use_ring:
            for R in range(1, int(self.temporal_neighbor_max_ring) + 1):
                ring_set = self._ring_keyframes_in_pool(full_pool, pos, pos_src, R)
                for pi, kf in enumerate(full_pool):
                    if int(kf) not in ring_set or int(kf) in scored_by_kf:
                        continue
                    score_kf(pi, int(kf))
                scored_rows_ring = [
                    (scored_by_kf[int(kf)][0], int(kf), scored_by_kf[int(kf)][1])
                    for kf in full_pool
                    if int(kf) in ring_set and int(kf) in scored_by_kf
                ]
                pick_result = self._pointcloud_topk_pick_from_scored_rows(scored_rows_ring, extra_needed, thr)
                if pick_result is not None:
                    picked_kfs, extra_target_pick_policy, above_kfs = pick_result
                    ring_effective = R
                    break
            if pick_result is None:
                for pi, kf in enumerate(full_pool):
                    if int(kf) not in scored_by_kf:
                        score_kf(pi, int(kf))
                scored_rows_full = [
                    (scored_by_kf[int(kf)][0], int(kf), scored_by_kf[int(kf)][1])
                    for kf in full_pool
                    if int(kf) in scored_by_kf
                ]
                pick_result = self._pointcloud_topk_pick_from_scored_rows(scored_rows_full, extra_needed, thr)
                if pick_result is None:
                    raise ValueError(
                        f"pointcloud_topk temporal ring fallback: could not pick {extra_needed} extra targets "
                        f"(scene={sidx.scene_id} segment={sidx.segment_id})"
                    )
                picked_kfs, extra_target_pick_policy, above_kfs = pick_result
                ring_effective = None
                fallback_full = True
        else:
            for pi, kf in enumerate(full_pool):
                score_kf(pi, int(kf))
            scored_rows_full = [
                (scored_by_kf[int(kf)][0], int(kf), scored_by_kf[int(kf)][1])
                for kf in full_pool
                if int(kf) in scored_by_kf
            ]
            pick_result = self._pointcloud_topk_pick_from_scored_rows(scored_rows_full, extra_needed, thr)
            if pick_result is None:
                raise ValueError("ranked_kfs is empty (internal error)")
            picked_kfs, extra_target_pick_policy, above_kfs = pick_result

        pool_final = [int(kf) for kf in full_pool if int(kf) in scored_by_kf]
        candidate_rep_image_refs: List[ImageRef] = [
            (representative_frame_for_keyframe(sidx, int(kf)), cam_src) for kf in pool_final
        ]
        candidate_keyframe_indices = [int(k) for k in pool_final]
        candidate_scores = [float(scored_by_kf[int(kf)][1]) for kf in pool_final]
        candidate_pair_counts = [dict(scored_by_kf[int(kf)][2]) for kf in pool_final]
        candidate_target_image_ref_lists: List[List[List[int]]] = [
            [[int(f), int(cam_src)] for f in sidx.keyframe_to_frames[int(kf)]] for kf in pool_final
        ]

        scored_rows = [
            (scored_by_kf[int(kf)][0], int(kf), scored_by_kf[int(kf)][1]) for kf in pool_final
        ]

        selected_target_scores: List[float] = []
        for kf in picked_kfs:
            for r in scored_rows:
                if r[1] == kf:
                    selected_target_scores.append(float(r[2]))
                    break

        for kf in picked_kfs:
            frame_tgt = int(random.choice(sidx.keyframe_to_frames[int(kf)]))
            refs.append((frame_tgt, cam_src))

        overlap_payload: Dict[str, Any] = {
            "scene_id": sid,
            "segment_id": seg,
            "overlap_mode": "pointcloud_topk",
            "overlap_score_type": str(self.overlap_score_type),
            "overlap_min": float(thr),
            "extra_target_pick_policy": extra_target_pick_policy,
            "candidates_above_min_count": int(len(above_kfs)),
            "extra_target_count": int(extra_needed),
            "overlap_point_sample_size": pss,
            "source_image_ref": [int(source_image_ref[0]), int(source_image_ref[1])],
            "source_keyframe_idx": int(kf_src),
            "source_rep_image_ref": [int(rep_src[0]), int(rep_src[1])],
            "candidate_rep_image_refs": [[int(a[0]), int(a[1])] for a in candidate_rep_image_refs],
            "candidate_target_image_ref_lists": candidate_target_image_ref_lists,
            "candidate_keyframe_indices": candidate_keyframe_indices,
            "candidate_scores": candidate_scores,
            "candidate_pair_counts": candidate_pair_counts,
            "selected_target_image_refs": [[int(x[0]), int(x[1])] for x in refs[1:]],
            "selected_target_scores": selected_target_scores,
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "pair_compute_miss_time_ms_total": float(pair_compute_miss_time_ms_total),
            "pair_eval_wall_time_ms_total": float(pair_eval_wall_time_ms_total),
            "temporal_neighbor_pool": self.temporal_neighbor_pool,
            "temporal_neighbor_ring_effective": ring_effective,
            "temporal_neighbor_fallback_full_pool": fallback_full,
        }
        return refs, overlap_payload

    def _refs_for_pair(
        self,
        sidx: SegmentIndex,
        kf: int,
        cam: int,
        episode_window: List[int],
    ) -> Tuple[ImageRef, List[ImageRef], Optional[Dict[str, Any]]]:
        frame_src = int(random.choice(sidx.keyframe_to_frames[int(kf)]))
        source_ref: ImageRef = (frame_src, int(cam))
        target_refs, overlap_payload = self._sample_target_image_refs(sidx, source_ref, episode_window)
        return source_ref, target_refs, overlap_payload

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
        if self.overlap_mode == "pointcloud_topk" and self.overlap_point_sample_size is not None:
            st_ep = self.current_segment_state
            if st_ep is None:
                raise ValueError("TrainSchedulerV4: current_segment_state is None for episode superset overlap pairs")
            episode_window = list(st_ep["episode_window_keyframes"])
            pos = self._kf_positions(sidx)
            seen_pairs: Set[Tuple[int, int, int, int]] = set()
            pair_dicts: List[Dict[str, Any]] = []
            # Use the full extra-target pool (same as _extra_target_keyframe_pool), not the temporal ring
            # subset, so DatasetPreloadManager overlap tasks still warm pairs used if ring expands to full.
            for kf_src, cam_src in pair_list:
                pos_src = int(pos[int(kf_src)])
                pool = self._extra_target_keyframe_pool(
                    sidx, int(kf_src), episode_window, pos, pos_src
                )
                rf_src = int(representative_frame_for_keyframe(sidx, int(kf_src)))
                for kf_tgt in pool:
                    rf_tgt = int(representative_frame_for_keyframe(sidx, int(kf_tgt)))
                    key = (rf_src, int(cam_src), rf_tgt, int(cam_src))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    pair_dicts.append(
                        {
                            "src_rep_image_ref": [key[0], key[1]],
                            "tgt_rep_image_ref": [key[2], key[3]],
                        }
                    )
            hint = self.dataset.build_preload_hint(
                scene_id,
                segment_id,
                future_image_refs=ordered,
                future_overlap_pairs=pair_dicts,
                overlap_meta={
                    "mode": "pointcloud_topk",
                    "point_sample_size": int(self.overlap_point_sample_size),
                },
            )
        else:
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
            src, tgts, overlap_pl = self._refs_for_pair(
                sidx, int(kf), int(cam), list(st["episode_window_keyframes"])
            )
            future = list(dict.fromkeys(list([src]) + list(tgts)))
            if overlap_pl is not None:
                sr = overlap_pl["source_rep_image_ref"]
                cands = overlap_pl["candidate_rep_image_refs"]
                pss = int(overlap_pl["overlap_point_sample_size"])
                overlap_pairs: List[Dict[str, Any]] = [
                    {
                        "src_rep_image_ref": [int(sr[0]), int(sr[1])],
                        "tgt_rep_image_ref": [int(cr[0]), int(cr[1])],
                    }
                    for cr in cands
                ]
                hint = self.dataset.build_preload_hint(
                    scene_id,
                    segment_id,
                    future_image_refs=future,
                    future_overlap_pairs=overlap_pairs,
                    overlap_meta={"mode": "pointcloud_topk", "point_sample_size": pss},
                )
            else:
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
        source_ref, target_refs, overlap_payload = self._refs_for_pair(
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

        if overlap_payload is not None and overlap_payload.get("overlap_mode") == "pointcloud_topk":
            ev_os = {
                "type": "overlap_select",
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "reset_episode_idx": int(self._reset_episode_idx),
                "block_idx_in_episode": int(st["block_idx_in_episode"]),
                "block_idx_in_segment": int(st["block_idx_in_segment"]),
                "block_idx_global": int(st["block_idx_global"]),
                **overlap_payload,
            }
            self._emit(ev_os)

        bb: Dict[str, Any] = {
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
            "overlap_mode": str(self.overlap_mode),
        }
        if overlap_payload is not None and overlap_payload.get("selected_target_scores") is not None:
            bb["selected_target_scores"] = list(overlap_payload["selected_target_scores"])
        if overlap_payload is not None:
            for k in (
                "temporal_neighbor_pool",
                "temporal_neighbor_ring_effective",
                "temporal_neighbor_fallback_full_pool",
            ):
                if k in overlap_payload:
                    bb[k] = overlap_payload[k]
        self._emit(bb)
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
        if hasattr(self.dataset, "clear_preload_scheduler_scope"):
            self.dataset.clear_preload_scheduler_scope()

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
        if hasattr(self.dataset, "set_preload_training_scope"):
            self.dataset.set_preload_training_scope(scene_id, segment_id)
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

        # Aligns with this `batch` before block_end / _start_block mutates `st` (get_current_info() would
        # otherwise describe the *next* block after a completed block).
        batch["_scheduler_v4_aligned_info"] = {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(st["scene_id"]),
            "segment_id": int(st["segment_id"]),
            "segment_local_step": int(st["segment_local_step"]),
            "block_idx_in_segment": int(st.get("block_idx_in_segment", -1)),
            "block_idx_global": int(st.get("block_idx_global", 0)),
            "source_image_ref": tuple(st["source_image_ref"]),
            "target_image_refs": [tuple(x) for x in st["target_image_refs"]],
            "U": int(self.U),
        }

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

        if hasattr(self.dataset, "maybe_log_preload_stats"):
            self.dataset.maybe_log_preload_stats(int(self.global_step))
        if hasattr(self.dataset, "maybe_log_overlap_stats"):
            self.dataset.maybe_log_overlap_stats(int(self.global_step))
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
