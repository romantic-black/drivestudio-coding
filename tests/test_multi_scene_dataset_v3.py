from __future__ import annotations

import threading
import time
from collections import OrderedDict
from types import MethodType
from unittest.mock import MagicMock, patch
from PIL import Image

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


def test_batch_request_disables_test_refs_by_default_policy():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._test_refs_enabled = MagicMock(return_value=False)
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
    v3.get_segment_batch_from_image_refs(req)
    kwargs = v3._assemble_segment_batch_from_image_refs.call_args.kwargs
    assert kwargs["include_test"] is False
    assert kwargs["test_image_refs"] is None


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
    v3._load_view_meta_from_asset = MagicMock(return_value=None)
    v3._overlay_pack_geometry_from_asset = MagicMock(side_effect=lambda **kw: kw["pack"])
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
    MultiSceneDatasetV3._get_cached_or_load_view_from_image_ref(
        v3, 1, 0, (0, 0), scene_dataset_opt=sd
    )
    MultiSceneDatasetV3._get_cached_or_load_view_from_image_ref(
        v3, 1, 0, (0, 0), scene_dataset_opt=sd
    )
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


def test_get_segment_index_allows_runtime_when_prebuilt_assets_disabled_even_if_policy_error():
    """Runtime-only dataset: asset_missing_policy default/error must not block get_segment_index."""
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._ensure_scene_loaded = MagicMock(
        return_value={
            "segments": [{"frame_indices": [0], "keyframe_indices": [0], "test_frame_indices": []}],
            "keyframe_segments": [[0]],
            "dataset": MagicMock(num_cams=1),
        }
    )
    v3._segment_index_cache = {}
    v3.use_prebuilt_assets = False
    v3.asset_store = None
    v3.asset_missing_policy = "error"
    v3._lock = threading.Lock()
    v3._segment_index_coord_lock = threading.Lock()
    v3._segment_index_inflight = {}
    v3._asset_handle_or_raise = MagicMock(return_value=None)
    v3.get_segment_index = MethodType(MultiSceneDatasetV3.get_segment_index, v3)
    idx = v3.get_segment_index(0, 0)
    assert idx.segment_id == 0
    v3._ensure_scene_loaded.assert_called()


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
    v3.asset_missing_policy = "ignore"
    v3._lock = threading.Lock()
    v3._segment_index_coord_lock = threading.Lock()
    v3._segment_index_inflight = {}
    v3.get_segment_index = MethodType(MultiSceneDatasetV3.get_segment_index, v3)
    with pytest.raises(ValueError, match="segment_id"):
        v3.get_segment_index(0, 99)


def test_get_segment_index_prefers_asset_payload():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._segment_index_cache = {}
    v3._lock = threading.Lock()
    v3._segment_index_coord_lock = threading.Lock()
    v3._segment_index_inflight = {}
    v3.use_prebuilt_assets = True
    v3.asset_missing_policy = "error"
    v3.data_cfg = MagicMock(get=MagicMock(return_value="nuscenes"))
    v3._ensure_scene_loaded = MagicMock(side_effect=AssertionError("fallback should not run"))
    payload = {
        "scene_id": 2,
        "segment_id": 1,
        "num_cams": 2,
        "frame_indices": [10, 11],
        "test_frame_indices": [20],
        "keyframe_indices": [0],
        "keyframe_to_frames": {0: [10, 11]},
        "frame_to_keyframe": {10: 0, 11: 0},
        "segment_first_frame_idx": 10,
        "train_image_refs": np.asarray([[10, 0], [10, 1], [11, 0], [11, 1]], dtype=np.int32),
        "test_image_refs": np.asarray([[20, 0], [20, 1]], dtype=np.int32),
    }
    handle = MagicMock(load_segment_index=MagicMock(return_value=payload))
    v3._asset_handle_or_raise = MagicMock(return_value=handle)
    v3._build_segment_index_from_asset_payload = MethodType(
        MultiSceneDatasetV3._build_segment_index_from_asset_payload, v3
    )
    v3.get_segment_index = MethodType(MultiSceneDatasetV3.get_segment_index, v3)

    idx = v3.get_segment_index(2, 1)
    assert idx.scene_id == 2
    assert idx.segment_id == 1
    assert idx.test_image_refs == ((20, 0), (20, 1))
    v3._ensure_scene_loaded.assert_not_called()


def test_get_view_geometry_prefers_asset_metadata_without_view_load():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3._preload_rtcfg = None
    v3._view_pack_lock = threading.RLock()
    v3._view_pack_cache = OrderedDict()
    v3._materialize_view_pack_cache = MagicMock(side_effect=AssertionError("should not materialize"))
    v3._load_view_from_image_ref = MagicMock(side_effect=AssertionError("should not load view pack"))
    v3._load_view_meta_from_asset = MagicMock(
        return_value={
            "camera_to_world": np.eye(4, dtype=np.float32),
            "intrinsic_4x4": np.eye(4, dtype=np.float32),
            "height": 120,
            "width": 200,
        }
    )
    c2w_seg0, K, H, W = MultiSceneDatasetV3._get_view_geometry_from_image_ref(
        v3,
        scene_id=1,
        segment_id=0,
        image_ref=(3, 0),
        world_to_seg0_np=np.eye(4, dtype=np.float64),
        scene_dataset_opt=MagicMock(),
    )
    assert H == 120 and W == 200
    assert np.allclose(c2w_seg0, np.eye(4))
    assert np.allclose(K, np.eye(3))


def test_load_view_from_asset_paths_resizes_when_file_resolution_differs_from_meta(tmp_path):
    """Image table may record training H,W (e.g. after downscale) while paths point at full-res files."""
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.full((6, 8, 3), 200, dtype=np.uint8)).save(rgb)
    dep = tmp_path / "d.npy"
    np.save(str(dep), np.ones((6, 8), dtype=np.float32) * 4.0)
    dyn = tmp_path / "dyn.png"
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(dyn)

    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    meta = {
        "image_path": str(rgb),
        "depth_path": str(dep),
        "sky_mask_path": "",
        "dynamic_mask_path": str(dyn),
        "camera_to_world": np.eye(4, dtype=np.float32),
        "intrinsic_4x4": np.eye(4, dtype=np.float32),
        "height": 2,
        "width": 3,
    }
    pack = MultiSceneDatasetV3._load_view_from_asset_paths(v3, 0, (0, 0), meta)
    assert tuple(pack["image"].shape) == (2, 3, 3)
    assert tuple(pack["depth"].shape) == (2, 3)
    assert pack["sky_mask"] is None
    assert tuple(pack["egocar_mask"].shape) == (2, 3)


def test_error_mode_get_cached_or_load_view_does_not_call_runtime_loader():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    v3._preload_rtcfg = None
    v3.asset_missing_policy = "error"
    v3._load_view_meta_from_asset = MagicMock(
        return_value={
            "image_path": "unused",
            "depth_path": "",
            "sky_mask_path": "",
            "dynamic_mask_path": "",
            "camera_to_world": np.eye(4, dtype=np.float32),
            "intrinsic_4x4": np.eye(4, dtype=np.float32),
            "height": 2,
            "width": 2,
        }
    )
    v3._load_view_from_asset_paths = MagicMock(
        return_value={
            "image": torch.zeros(2, 2, 3),
            "extrinsic": torch.eye(4),
            "intrinsic": torch.eye(4),
            "depth": torch.ones(2, 2),
            "sky_mask": None,
            "viewdirs": None,
            "egocar_mask": None,
            "frame_idx": 0,
            "cam_idx": 0,
        }
    )
    v3._load_view_from_image_ref = MagicMock(side_effect=AssertionError("runtime loader should not run"))
    out = MultiSceneDatasetV3._get_cached_or_load_view_from_image_ref(v3, 1, 0, (0, 0), None)
    assert int(out["frame_idx"]) == 0
    v3._load_view_from_image_ref.assert_not_called()


def test_error_mode_materialize_cache_does_not_call_runtime_loader():
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
    v3.asset_missing_policy = "error"
    v3._preload_rtcfg = parse_preload_cfg(cfg)
    v3._view_pack_cache = OrderedDict()
    v3._view_pack_lock = threading.RLock()
    v3._view_load_coord_lock = threading.Lock()
    v3._view_load_inflight = {}
    v3._preload_training_scene_id = 1
    v3._preload_training_segment_id = 0
    v3._load_view_meta_from_asset = MagicMock(
        return_value={
            "image_path": "unused",
            "depth_path": "",
            "sky_mask_path": "",
            "dynamic_mask_path": "",
            "camera_to_world": np.eye(4, dtype=np.float32),
            "intrinsic_4x4": np.eye(4, dtype=np.float32),
            "height": 2,
            "width": 2,
        }
    )
    v3._load_view_from_asset_paths = MagicMock(
        return_value={
            "image": torch.zeros(2, 2, 3),
            "extrinsic": torch.eye(4),
            "intrinsic": torch.eye(4),
            "depth": torch.ones(2, 2),
            "sky_mask": None,
            "viewdirs": None,
            "egocar_mask": None,
            "frame_idx": 0,
            "cam_idx": 0,
        }
    )
    v3._load_view_from_image_ref = MagicMock(side_effect=AssertionError("runtime loader should not run"))
    MultiSceneDatasetV3._materialize_view_pack_cache(v3, (1, 0, 0, 0), (0, 0), None)
    assert (1, 0, 0, 0) in v3._view_pack_cache
    v3._load_view_from_image_ref.assert_not_called()


def test_resolve_test_refs_helper_kept_but_main_path_disabled():
    sidx = SegmentIndex(
        scene_id=0,
        segment_id=0,
        num_cams=2,
        frame_indices=[1, 2],
        test_frame_indices=[10, 11],
        train_frame_set=frozenset([1, 2]),
        test_frame_set=frozenset([10, 11]),
        keyframe_indices=[0],
        keyframe_to_frames={0: [1, 2]},
        frame_to_keyframe={1: 0, 2: 0},
        segment_first_frame_idx=1,
    )
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.data_cfg = MagicMock(pixel_source={"max_test_images": 1})
    v3._lock = threading.RLock()
    v3._test_image_refs_cache = {}
    v3.get_segment_index = MagicMock(return_value=sidx)
    helper = MultiSceneDatasetV3.resolve_test_image_refs_deterministic_from_sidx(v3, sidx)
    main_refs = MultiSceneDatasetV3.resolve_test_image_refs_deterministic(v3, 0, 0)
    assert helper == [(10, 0), (10, 1)]
    assert main_refs == []


def test_dynamic_empty_pointcloud_allows_missing_tracks():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    v3.use_prebuilt_assets = True
    v3.asset_store = MagicMock()
    v3.asset_missing_policy = "error"
    v3.pointcloud_generator = object()
    v3.data_cfg = {"dataset": "nuscenes"}
    v3.segment_aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    sidx = SegmentIndex(
        scene_id=0,
        segment_id=0,
        num_cams=1,
        frame_indices=[1],
        test_frame_indices=[],
        train_frame_set=frozenset([1]),
        test_frame_set=frozenset(),
        keyframe_indices=[0],
        keyframe_to_frames={0: [1]},
        frame_to_keyframe={1: 0},
        segment_first_frame_idx=1,
    )
    v3.get_segment_index = MagicMock(return_value=sidx)
    v3.validate_image_ref = MagicMock()
    v3._ensure_scene_loaded = MagicMock(side_effect=AssertionError("runtime should not run"))
    v3._ensure_segment_pose_cached_from_assets_only = MagicMock(
        return_value=(torch.eye(4), torch.eye(4), 1, "asset")
    )
    pack = {
        "image": torch.zeros(2, 2, 3),
        "extrinsic": torch.eye(4),
        "intrinsic": torch.eye(4),
        "depth": torch.ones(2, 2),
        "sky_mask": None,
        "viewdirs": None,
        "egocar_mask": None,
        "frame_idx": 1,
        "cam_idx": 0,
    }
    v3._get_cached_or_load_view_from_image_ref = MagicMock(return_value=pack)
    v3._load_view_from_image_ref = MagicMock(side_effect=AssertionError("runtime view load should not run"))
    v3._ensure_segment_pointcloud_cached = MagicMock(
        return_value={"background": np.zeros((0, 3), dtype=np.float32), "dynamic": {}, "metadata": {}}
    )
    v3._load_segment_dynamic_tracks_cached = MagicMock(
        side_effect=AssertionError("dynamic tracks should not be loaded when dynamic pointcloud is empty")
    )
    batch = MultiSceneDatasetV3._assemble_segment_batch_from_image_refs(
        v3,
        scene_id=0,
        segment_id=0,
        source_image_refs=[(1, 0)],
        target_image_refs=[(1, 0)],
        include_test=False,
        test_image_refs=None,
        enforce_target0_equals_source=True,
    )
    assert "dynamic_info" not in batch


def test_train_mainline_no_scene_runtime_load_in_error_mode():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    v3.use_prebuilt_assets = True
    v3.asset_store = MagicMock()
    v3.asset_missing_policy = "error"
    v3.pointcloud_generator = object()
    v3.data_cfg = {"dataset": "nuscenes"}
    v3.segment_aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    sidx = SegmentIndex(
        scene_id=0,
        segment_id=0,
        num_cams=1,
        frame_indices=[1],
        test_frame_indices=[],
        train_frame_set=frozenset([1]),
        test_frame_set=frozenset(),
        keyframe_indices=[0],
        keyframe_to_frames={0: [1]},
        frame_to_keyframe={1: 0},
        segment_first_frame_idx=1,
    )
    v3.get_segment_index = MagicMock(return_value=sidx)
    v3.validate_image_ref = MagicMock()
    v3._ensure_scene_loaded = MagicMock(side_effect=AssertionError("runtime should not run"))
    v3._ensure_segment_pose_cached_from_assets_only = MagicMock(
        return_value=(torch.eye(4), torch.eye(4), 1, "asset")
    )
    pack = {
        "image": torch.zeros(2, 2, 3),
        "extrinsic": torch.eye(4),
        "intrinsic": torch.eye(4),
        "depth": torch.ones(2, 2),
        "sky_mask": None,
        "viewdirs": None,
        "egocar_mask": None,
        "frame_idx": 1,
        "cam_idx": 0,
    }
    v3._get_cached_or_load_view_from_image_ref = MagicMock(return_value=pack)
    v3._load_view_from_image_ref = MagicMock(side_effect=AssertionError("runtime view load should not run"))
    v3._ensure_segment_pointcloud_cached = MagicMock(
        return_value={"background": np.zeros((0, 3), dtype=np.float32), "dynamic": {}, "metadata": {}}
    )
    v3._load_segment_dynamic_tracks_cached = MagicMock(
        side_effect=AssertionError("tracks should not be loaded for empty dynamic pointcloud")
    )

    MultiSceneDatasetV3._assemble_segment_batch_from_image_refs(
        v3,
        scene_id=0,
        segment_id=0,
        source_image_refs=[(1, 0)],
        target_image_refs=[(1, 0)],
        include_test=False,
        test_image_refs=None,
        enforce_target0_equals_source=True,
    )
    v3._ensure_scene_loaded.assert_not_called()
    v3._load_view_from_image_ref.assert_not_called()


def test_train_mainline_dynamic_tracks_required_when_dynamic_non_empty():
    v3 = MultiSceneDatasetV3.__new__(MultiSceneDatasetV3)
    v3.device = torch.device("cpu")
    v3.use_prebuilt_assets = True
    v3.asset_store = MagicMock()
    v3.asset_missing_policy = "error"
    v3.pointcloud_generator = object()
    v3.data_cfg = {"dataset": "nuscenes"}
    v3.segment_aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    sidx = SegmentIndex(
        scene_id=0,
        segment_id=0,
        num_cams=1,
        frame_indices=[1],
        test_frame_indices=[],
        train_frame_set=frozenset([1]),
        test_frame_set=frozenset(),
        keyframe_indices=[0],
        keyframe_to_frames={0: [1]},
        frame_to_keyframe={1: 0},
        segment_first_frame_idx=1,
    )
    v3.get_segment_index = MagicMock(return_value=sidx)
    v3.validate_image_ref = MagicMock()
    v3._ensure_scene_loaded = MagicMock(side_effect=AssertionError("runtime should not run"))
    v3._ensure_segment_pose_cached_from_assets_only = MagicMock(
        return_value=(torch.eye(4), torch.eye(4), 1, "asset")
    )
    pack = {
        "image": torch.zeros(2, 2, 3),
        "extrinsic": torch.eye(4),
        "intrinsic": torch.eye(4),
        "depth": torch.ones(2, 2),
        "sky_mask": None,
        "viewdirs": None,
        "egocar_mask": None,
        "frame_idx": 1,
        "cam_idx": 0,
    }
    v3._get_cached_or_load_view_from_image_ref = MagicMock(return_value=pack)
    v3._ensure_segment_pointcloud_cached = MagicMock(
        return_value={
            "background": np.zeros((0, 3), dtype=np.float32),
            "dynamic": {7: np.zeros((1, 3), dtype=np.float32)},
            "instance_mapping": {1007: 7},
            "metadata": {},
        }
    )
    v3._load_segment_dynamic_tracks_cached = MagicMock(return_value=None)

    with pytest.raises(ValueError, match="forbids runtime fallback"):
        MultiSceneDatasetV3._assemble_segment_batch_from_image_refs(
            v3,
            scene_id=0,
            segment_id=0,
            source_image_refs=[(1, 0)],
            target_image_refs=[(1, 0)],
            include_test=False,
            test_image_refs=None,
            enforce_target0_equals_source=True,
        )
