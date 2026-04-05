from __future__ import annotations

import threading
import time
from collections import OrderedDict
from types import MethodType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from datasets.dataset_preload_manager import DatasetPreloadManager, parse_preload_cfg
from datasets.multi_scene_dataset import MultiSceneDataset
from datasets.multi_scene_dataset_v3 import (
    BatchRequestV3,
    MultiSceneDatasetV3,
    SegmentIndex,
    _build_segment_index_dict,
    _visibility_mask_seg0,
)


def test_build_segment_index_dict_maps_frames_to_keyframes():
    scene_data = {
        "dataset": MagicMock(num_cams=3),
        "keyframe_segments": [[0, 1], [2, 3, 4]],
        "segments": [
            {
                "frame_indices": [0, 1, 2, 3, 4],
                "test_frame_indices": [20],
                "keyframe_indices": [0, 1],
            }
        ],
    }
    idx = _build_segment_index_dict(7, 0, scene_data)
    assert idx.scene_id == 7
    assert idx.segment_id == 0
    assert idx.num_cams == 3
    assert idx.segment_first_frame_idx == 0
    assert idx.keyframe_to_frames[0] == [0, 1]
    assert idx.keyframe_to_frames[1] == [2, 3, 4]
    assert idx.frame_to_keyframe[0] == 0
    assert idx.frame_to_keyframe[4] == 1
    assert idx.train_frame_set == frozenset([0, 1, 2, 3, 4])
    assert idx.test_frame_set == frozenset([20])


def test_build_segment_index_dict_invalid_keyframe_index_raises():
    scene_data = {
        "dataset": MagicMock(num_cams=1),
        "keyframe_segments": [[0, 1]],
        "segments": [
            {
                "frame_indices": [0, 1],
                "test_frame_indices": [],
                "keyframe_indices": [0, 99],
            }
        ],
    }
    with pytest.raises(ValueError, match="Invalid keyframe index"):
        _build_segment_index_dict(0, 0, scene_data)


def test_build_segment_index_dict_keyframe_empty_train_overlap_raises():
    scene_data = {
        "dataset": MagicMock(num_cams=1),
        "keyframe_segments": [[0, 1], [100, 101]],
        "segments": [
            {
                "frame_indices": [0, 1],
                "test_frame_indices": [],
                "keyframe_indices": [0, 1],
            }
        ],
    }
    with pytest.raises(ValueError, match="no train frames"):
        _build_segment_index_dict(0, 0, scene_data)


def test_build_segment_index_dict_conflicting_frame_raises():
    scene_data = {
        "dataset": MagicMock(num_cams=1),
        "keyframe_segments": [[0], [0]],
        "segments": [
            {
                "frame_indices": [0],
                "test_frame_indices": [],
                "keyframe_indices": [0, 1],
            }
        ],
    }
    with pytest.raises(ValueError, match="multiple keyframes"):
        _build_segment_index_dict(0, 0, scene_data)


def test_validate_image_ref_train_vs_test():
    sidx = SegmentIndex(
        scene_id=0,
        segment_id=0,
        num_cams=2,
        frame_indices=[1, 2, 3],
        test_frame_indices=[10],
        train_frame_set=frozenset([1, 2, 3]),
        test_frame_set=frozenset([10]),
        keyframe_indices=[0],
        keyframe_to_frames={0: [1, 2, 3]},
        frame_to_keyframe={1: 0, 2: 0, 3: 0},
        segment_first_frame_idx=1,
    )
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.get_segment_index = MagicMock(return_value=sidx)

    MultiSceneDatasetV3.validate_image_ref(v3, 0, 0, (2, 0), purpose="train")
    with pytest.raises(ValueError, match="train"):
        MultiSceneDatasetV3.validate_image_ref(v3, 0, 0, (10, 0), purpose="train")

    MultiSceneDatasetV3.validate_image_ref(v3, 0, 0, (10, 0), purpose="test")
    with pytest.raises(ValueError, match="test"):
        MultiSceneDatasetV3.validate_image_ref(v3, 0, 0, (2, 0), purpose="test")

    with pytest.raises(ValueError, match="cam_id"):
        MultiSceneDatasetV3.validate_image_ref(v3, 0, 0, (2, 5), purpose="train")


def test_get_or_compute_pair_score_none_mode():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    assert MultiSceneDatasetV3.get_or_compute_pair_score(v3, 0, 0, (0, 0), (1, 0), mode="none") is None
    with pytest.raises(ValueError, match="unsupported mode"):
        MultiSceneDatasetV3.get_or_compute_pair_score(v3, 0, 0, (0, 0), (1, 0), mode="overlap")


def test_get_segment_batch_from_image_refs_does_not_touch_random_keyframe_selectors():
    """V3 image-ref path must not call _select_source_and_target_keyframes / _select_frame_from_keyframe."""
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._assemble_segment_batch_from_image_refs = MagicMock(return_value={"assembled": True})
    select_kf = MagicMock()
    select_frame = MagicMock()
    v3._select_source_and_target_keyframes = select_kf
    v3._select_frame_from_keyframe = select_frame
    v3.get_segment_batch_from_image_refs = MethodType(MultiSceneDatasetV3.get_segment_batch_from_image_refs, v3)

    req = BatchRequestV3(
        scene_id=0,
        segment_id=0,
        source_image_ref=(1, 0),
        target_image_refs=[(1, 0), (2, 0)],
        include_test=False,
    )
    out = v3.get_segment_batch_from_image_refs(req)
    assert out == {"assembled": True}
    select_kf.assert_not_called()
    select_frame.assert_not_called()
    v3._assemble_segment_batch_from_image_refs.assert_called_once()


def test_build_preload_hint_structure():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    hint = MultiSceneDatasetV3.build_preload_hint(
        v3,
        scene_id=1,
        segment_id=2,
        future_image_refs=[(3, 0), (3, 1), (5, 0)],
    )
    assert hint["scene_id"] == 1
    assert hint["segment_id"] == 2
    assert hint["unique_frame_indices"] == [3, 5]
    assert hint["unique_cam_indices"] == [0, 1]
    assert hint["hint_version"] == 1


def test_build_preload_hint_v2_with_overlap_pairs():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    pairs = [{"src_rep_image_ref": [0, 0], "tgt_rep_image_ref": [1, 0]}]
    hint = MultiSceneDatasetV3.build_preload_hint(
        v3,
        scene_id=0,
        segment_id=0,
        future_image_refs=[(0, 0)],
        future_overlap_pairs=pairs,
        overlap_meta={"mode": "pointcloud_topk", "point_sample_size": 4096},
    )
    assert hint["hint_version"] == 2
    assert hint["future_overlap_pairs"] == pairs
    assert hint["overlap_meta"]["mode"] == "pointcloud_topk"


def test_get_or_compute_pair_score_respects_account_runtime_stats_false_on_hit():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._lock = threading.RLock()
    v3._pair_score_cache = {}
    v3._overlap_stats = {
        "pair_queries": 0.0,
        "pair_cache_hits": 0.0,
        "pair_cache_misses": 0.0,
        "pair_compute_miss_ms_sum": 0.0,
        "pair_eval_wall_ms_sum": 0.0,
        "src_rep_no_visible": 0.0,
        "candidate_eval_count": 0.0,
    }
    v3._pair_score_cache[(0, 0, (0, 0), (1, 0), "pointcloud_topk", 1024)] = (0.3, 1, 1, 1)
    out = MultiSceneDatasetV3.get_or_compute_pair_score(
        v3,
        0,
        0,
        (0, 0),
        (1, 0),
        mode="pointcloud_topk",
        point_sample_size=1024,
        account_runtime_stats=False,
    )
    assert out == pytest.approx(0.3)
    assert v3._overlap_stats["pair_queries"] == 0.0


def test_submit_overlap_pair_skips_when_pair_score_cached():
    cfg = {
        "enable": True,
        "num_workers": 1,
        "max_pending_tasks": 64,
        "enable_view_pack_cache": True,
        "view_cache_max_items_total": 32,
        "view_cache_max_items_per_scene": 16,
        "view_cache_device": "cpu",
        "drop_stale_hints": True,
        "dedupe_tasks": True,
        "warm_segment_static": False,
        "warm_segment_pointcloud": False,
        "warm_next_block_exact": True,
        "warm_test_refs": False,
        "warm_episode_source_superset": False,
        "warm_overlap_pairs_episode_superset": False,
        "warm_overlap_pairs_next_block_exact": False,
        "stats_log_interval_steps": 0,
    }
    ds = MagicMock()
    ds.is_pair_score_cached = MagicMock(return_value=True)
    mgr = DatasetPreloadManager(ds, parse_preload_cfg(cfg))
    mgr.submit_overlap_pair(
        0,
        1,
        0,
        (0, 0),
        (1, 0),
        mode="pointcloud_topk",
        point_sample_size=1024,
        meta={},
    )
    assert mgr._heap == []
    ds.is_pair_score_cached.assert_called_once()


def test_batch_request_test_refs_require_include_test():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._assemble_segment_batch_from_image_refs = MagicMock(return_value={})
    v3.get_segment_batch_from_image_refs = MethodType(MultiSceneDatasetV3.get_segment_batch_from_image_refs, v3)

    req = BatchRequestV3(
        scene_id=0,
        segment_id=0,
        source_image_ref=(1, 0),
        target_image_refs=[(1, 0)],
        include_test=False,
        test_image_refs=[(10, 0)],
    )
    with pytest.raises(ValueError, match="include_test"):
        v3.get_segment_batch_from_image_refs(req)


def test_unload_scene_clears_v3_caches():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._lock = threading.RLock()
    v3._segment_index_cache = {(1, 0): MagicMock(), (2, 0): MagicMock()}
    v3._pair_score_cache = {(1, 0, (0, 0), (1, 0), "m", 1024): (0.25, 1, 1, 1)}
    v3._view_pack_cache = OrderedDict({(1, 0, 0, 0): MagicMock(), (2, 0, 0, 0): MagicMock()})
    v3._view_pack_lock = threading.RLock()
    v3._segment_index_coord_lock = threading.Lock()
    v3._segment_pointcloud_coord_lock = threading.Lock()
    v3._view_load_coord_lock = threading.Lock()
    v3._preload_manager = MagicMock()
    v3._scene_preload_inflight = {}
    v3._scene_preload_inflight_lock = threading.Lock()
    v3._scene_unloading = set()
    v3._scene_unloading_lock = threading.Lock()
    v3._segment_pose_cache = {}
    v3._test_image_refs_cache = {}
    v3._segment_index_inflight = {}
    v3._segment_pointcloud_inflight = {}
    v3._view_load_inflight = {}
    with patch.object(MultiSceneDataset, "_unload_scene", lambda self, sid: None):
        MultiSceneDatasetV3._unload_scene(v3, 1)
    assert (1, 0) not in v3._segment_index_cache
    assert (2, 0) in v3._segment_index_cache
    assert len(v3._pair_score_cache) == 0
    assert (1, 0, 0, 0) not in v3._view_pack_cache
    assert (2, 0, 0, 0) in v3._view_pack_cache
    v3._preload_manager.clear_pending_for_scene.assert_called_once_with(1)


def test_mark_scene_completed_unloads_without_lock():
    class TrackingLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self.owner = False

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.release()

        def acquire(self, *args, **kwargs):
            out = self._lock.acquire(*args, **kwargs)
            if out:
                self.owner = True
            return out

        def release(self) -> None:
            self.owner = False
            self._lock.release()

    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._lock = TrackingLock()
    v3.scene_training_queue = [1, 2]
    v3.current_scene_index = 0
    v3._ensure_training_queue_ready = MagicMock()
    v3._preload_scenes = MagicMock()
    calls = []

    def _unload_scene(scene_id: int) -> None:
        assert not v3._lock.owner
        calls.append(int(scene_id))

    v3._unload_scene = _unload_scene
    MultiSceneDatasetV3.mark_scene_completed(v3, 1)
    assert calls == [1]
    assert v3.current_scene_index == 1


def test_preload_manager_stop_preserves_thread_on_timeout():
    cfg = {
        "enable": True,
        "num_workers": 1,
        "max_pending_tasks": 64,
        "enable_view_pack_cache": True,
        "view_cache_max_items_total": 32,
        "view_cache_max_items_per_scene": 16,
        "view_cache_device": "cpu",
        "drop_stale_hints": True,
        "dedupe_tasks": True,
        "warm_segment_static": False,
        "warm_segment_pointcloud": False,
        "warm_next_block_exact": True,
        "warm_test_refs": False,
        "warm_episode_source_superset": False,
        "warm_overlap_pairs_episode_superset": False,
        "warm_overlap_pairs_next_block_exact": False,
        "stats_log_interval_steps": 0,
    }
    mgr = DatasetPreloadManager(MagicMock(), parse_preload_cfg(cfg))
    block = threading.Event()

    def _block_run(self) -> None:
        block.wait()

    mgr._run = MethodType(_block_run, mgr)
    mgr.start()

    for _ in range(50):
        if mgr._thread is not None and mgr._thread.is_alive():
            break
        time.sleep(0.01)

    assert mgr._thread is not None and mgr._thread.is_alive()
    t = mgr._thread
    mgr.stop(timeout=0.01)
    assert mgr._thread is t
    assert mgr._thread.is_alive()

    mgr.start()
    assert mgr._thread is t

    block.set()
    mgr.stop(timeout=1.0)
    assert mgr._thread is None


def test_get_cached_or_load_view_single_underlying_load():
    cfg = {
        "enable": True,
        "num_workers": 1,
        "max_pending_tasks": 64,
        "enable_view_pack_cache": True,
        "view_cache_max_items_total": 32,
        "view_cache_max_items_per_scene": 16,
        "view_cache_device": "cpu",
        "drop_stale_hints": True,
        "dedupe_tasks": True,
        "warm_segment_static": False,
        "warm_segment_pointcloud": False,
        "warm_next_block_exact": True,
        "warm_test_refs": False,
        "warm_episode_source_superset": False,
        "warm_overlap_pairs_episode_superset": False,
        "warm_overlap_pairs_next_block_exact": False,
        "stats_log_interval_steps": 0,
    }
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    v3._preload_rtcfg = parse_preload_cfg(cfg)
    v3._view_pack_cache = OrderedDict()
    v3._view_pack_lock = threading.RLock()
    v3._view_load_coord_lock = threading.Lock()
    v3._view_load_inflight = {}
    v3._preload_training_scene_id = 1
    v3._preload_training_segment_id = 0
    loads = []

    def _fake_load(sd, ref):
        loads.append(ref)
        return {
            "image": torch.zeros(2, 2, 3),
            "extrinsic": torch.eye(4),
            "intrinsic": torch.eye(4),
            "depth": torch.ones(2, 2),
            "sky_mask": None,
            "viewdirs": None,
            "egocar_mask": None,
            "frame_idx": int(ref[0]),
            "cam_idx": int(ref[1]),
        }

    v3._load_view_from_image_ref = _fake_load
    sd = MagicMock()
    MultiSceneDatasetV3._get_cached_or_load_view_from_image_ref(v3, 1, 0, sd, (0, 0))
    MultiSceneDatasetV3._get_cached_or_load_view_from_image_ref(v3, 1, 0, sd, (0, 0))
    assert len(loads) == 1


def test_visibility_mask_seg0_front_point():
    pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    vis = _visibility_mask_seg0(pts, c2w, K, 480, 640)
    assert vis.shape == (1,) and bool(vis[0])


def test_visibility_mask_seg0_behind_camera():
    pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    vis = _visibility_mask_seg0(pts, c2w, K, 480, 640)
    assert not bool(vis[0])


def test_get_segment_index_invalid_segment_id_valueerror():
    """Out-of-range segment_id must raise ValueError (not IndexError)."""
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._ensure_scene_loaded = MagicMock(
        return_value={
            "segments": [{"frame_indices": [0], "keyframe_indices": [0], "test_frame_indices": []}],
            "keyframe_segments": [[0]],
            "dataset": MagicMock(num_cams=1),
        }
    )
    v3._segment_index_cache = {}
    v3._lock = threading.Lock()
    v3._segment_index_coord_lock = threading.Lock()
    v3._segment_index_inflight = {}
    v3.get_segment_index = MethodType(MultiSceneDatasetV3.get_segment_index, v3)
    with pytest.raises(ValueError, match="segment_id"):
        v3.get_segment_index(0, 99)
