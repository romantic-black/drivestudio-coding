from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.multi_scene_dataset_v4 import BatchRequestV4, MultiSceneDatasetV4
from datasets.streetforward_assets import StreetForwardAssetStore


def _prepare_demo_assets(
    tmp_path,
    *,
    tracks_all_invisible: bool = False,
    instances_fv_override: np.ndarray | None = None,
):
    image0 = tmp_path / "im0.png"
    image1 = tmp_path / "im1.png"
    sky0 = tmp_path / "sky0.png"
    sky1 = tmp_path / "sky1.png"
    dyn0 = tmp_path / "dyn0.png"
    dyn1 = tmp_path / "dyn1.png"
    depth0 = tmp_path / "d0.npy"
    depth1 = tmp_path / "d1.npy"

    Image.fromarray(np.full((4, 5, 3), 64, dtype=np.uint8)).save(image0)
    Image.fromarray(np.full((4, 5, 3), 192, dtype=np.uint8)).save(image1)
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(sky0)
    Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(sky1)
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(dyn0)
    Image.fromarray(np.ones((4, 5), dtype=np.uint8) * 255).save(dyn1)
    np.save(depth0, np.full((4, 5), 1.0, dtype=np.float32))
    np.save(depth1, np.full((4, 5), 2.0, dtype=np.float32))

    store = StreetForwardAssetStore(str(tmp_path), missing_policy="error")
    scene_asset_id = store.export_scene_asset(
        dataset="nuscenes",
        scene_id=1,
        scene_name="000001",
        num_frames=3,
        num_cams=1,
        split_config={"test_image_stride": 0},
        scene_index_arrays={
            "scene_id": np.asarray([1], dtype=np.int32),
            "train_frame_indices": np.asarray([0, 1], dtype=np.int32),
            "test_frame_indices": np.asarray([2], dtype=np.int32),
            "keyframe_indices": np.asarray([0, 1], dtype=np.int32),
            "keyframe_to_frames_flat": np.asarray([0, 1], dtype=np.int32),
            "keyframe_to_frames_offsets": np.asarray([0, 1, 2], dtype=np.int64),
            "segment_ids": np.asarray([0], dtype=np.int32),
            "segment_frame_indices_flat": np.asarray([0, 1], dtype=np.int32),
            "segment_frame_offsets": np.asarray([0, 2], dtype=np.int64),
            "segment_keyframe_indices_flat": np.asarray([0, 1], dtype=np.int32),
            "segment_keyframe_offsets": np.asarray([0, 2], dtype=np.int64),
        },
        image_table_rows=[
            {
                "frame_idx": 0,
                "cam_id": 0,
                "img_idx": 0,
                "is_train": True,
                "is_test": False,
                "image_path": str(image0),
                "depth_path": str(depth0),
                "sky_mask_path": str(sky0),
                "dynamic_mask_path": str(dyn0),
                "height": 4,
                "width": 5,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            },
            {
                "frame_idx": 1,
                "cam_id": 0,
                "img_idx": 1,
                "is_train": True,
                "is_test": False,
                "image_path": str(image1),
                "depth_path": str(depth1),
                "sky_mask_path": str(sky1),
                "dynamic_mask_path": str(dyn1),
                "height": 4,
                "width": 5,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            },
            {
                "frame_idx": 2,
                "cam_id": 0,
                "img_idx": 2,
                "is_train": False,
                "is_test": True,
                "image_path": str(image1),
                "depth_path": str(depth1),
                "sky_mask_path": str(sky1),
                "dynamic_mask_path": str(dyn1),
                "height": 4,
                "width": 5,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            },
        ],
    )
    instances_fv = np.asarray([[1], [1]], dtype=np.uint8)
    if tracks_all_invisible:
        instances_fv = np.asarray([[0], [0]], dtype=np.uint8)
    if instances_fv_override is not None:
        instances_fv = np.asarray(instances_fv_override, dtype=np.uint8)

    store.export_segment_asset(
        dataset="nuscenes",
        scene_id=1,
        segment_id=0,
        parent_scene_asset_id=scene_asset_id,
        segment_index_payload={
            "num_cams": 1,
            "frame_indices": [0, 1],
            "test_frame_indices": [2],
            "keyframe_indices": [0, 1],
            "keyframe_to_frames": {0: [0], 1: [1]},
            "frame_to_keyframe": {0: 0, 1: 1},
            "segment_first_frame_idx": 0,
            "train_image_refs": np.asarray([[0, 0], [1, 0]], dtype=np.int32),
            "test_image_refs": np.asarray([[2, 0]], dtype=np.int32),
        },
        segment_pose_payload={
            "segment_first_pose_world": np.eye(4, dtype=np.float32),
            "world_to_seg0": np.eye(4, dtype=np.float32),
            "segment_first_frame_idx": 0,
            "segment_pose_source": "camera",
        },
        pointcloud_payload={
            "background": np.zeros((2, 6), dtype=np.float32),
            "dynamic": {9: np.ones((1, 6), dtype=np.float32)},
            "instance_mapping": {1009: 9},
            "metadata": {"static_instance_intids": []},
        },
        dynamic_tracks_payload={
            "frame_indices": np.asarray([0, 1], dtype=np.int32),
            "instance_intids": np.asarray([9], dtype=np.int32),
            "instances_quats": np.asarray([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=np.float32),
            "instances_trans": np.asarray([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
            "instances_fv": instances_fv,
            "static_instance_intids": np.asarray([], dtype=np.int32),
        },
        segment_aabb=np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float32),
        pointcloud_config_normalized={"type": "hybrid"},
        stats={"background_points": 2},
    )
    return store


def _build_cfg(tmp_path):
    data_cfg = OmegaConf.create(
        {
            "dataset": "nuscenes",
            "assets": {"root": str(tmp_path)},
            "scene_asset_cache_max_items": 32,
            "segment_static_cache_max_items": 32,
            "image_meta_cache_max_items": 128,
            "view_pack_cache_max_items": 128,
            "pixel_source": {
                "max_test_images": 0,
                "load_sky_mask": True,
                "load_dynamic_mask": True,
            },
            "sky_mask_semantics": "one_is_sky",
        }
    )
    dataset_cfg = OmegaConf.create(
        {
            "segment_aabb": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        }
    )
    return data_cfg, dataset_cfg


def test_v4_batch_from_assets_strict_success(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(0, 0), (1, 0)],
        include_test=False,
    )
    batch = ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
    assert batch["source"]["image"].shape == (1, 4, 5, 3)
    assert batch["target"]["image"].shape == (2, 4, 5, 3)
    assert batch["source"]["viewdirs"].shape == (1, 4, 5, 3)
    assert batch["source"]["dynamic_mask"].shape == (1, 4, 5)
    assert torch.allclose(batch["aabb"], torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))
    assert "pointcloud" in batch
    assert "dynamic_info" in batch


def test_v4_reconcile_drops_invisible_dynamic_instances(tmp_path):
    store = _prepare_demo_assets(tmp_path, tracks_all_invisible=True)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(0, 0), (1, 0)],
        include_test=False,
    )
    batch = ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
    assert "dynamic_info" not in batch
    assert batch["pointcloud"]["dynamic"] == {}


def test_v4_batch_window_drops_non_visible_dynamic_instances(tmp_path):
    store = _prepare_demo_assets(
        tmp_path,
        instances_fv_override=np.asarray([[1], [0]], dtype=np.uint8),
    )
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(1, 0),
        target_image_refs=[(1, 0)],
        include_test=False,
    )
    batch = ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
    assert "dynamic_info" not in batch
    assert batch["pointcloud"]["dynamic"] == {}


def test_v4_enforce_target0_equals_source(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(1, 0)],
        include_test=False,
    )
    with pytest.raises(ValueError, match="target_image_refs\\[0\\] must equal source_image_ref"):
        ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)


def test_v4_never_uses_runtime_scene_loader(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds._ensure_scene_loaded = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime should not run"))
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(0, 0), (1, 0)],
    )
    batch = ds.get_segment_batch_from_image_refs(req)
    assert batch["target"]["frame_indices"].tolist() == [0, 1]


def test_v4_segment_aabb_mismatch_raises(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    dataset_cfg.segment_aabb = [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    with pytest.raises(ValueError, match="segment_aabb mismatch"):
        ds.initialize()


def test_v4_requires_explicit_cache_limits(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    del data_cfg["scene_asset_cache_max_items"]
    with pytest.raises(ValueError, match="explicit cache size limits"):
        MultiSceneDatasetV4(
            dataset_cfg=dataset_cfg,
            data_cfg=data_cfg,
            device=torch.device("cpu"),
            asset_store=store,
        )


def test_v4_scheduler_v6_factory_and_next_batch(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.get_scene = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("get_scene should not be called"))
    sch = ds.create_train_scheduler_v6(
        state_write_interval_steps=1,
        updates_per_block=1,
        keyframes_per_episode=2,
        episodes_per_segment=1,
        total_target_frames=2,
        include_source_frame=True,
        neighbor_ring=1,
        prefer_nearby_keyframes=True,
        fallback_expand_to_segment=True,
        with_replacement=True,
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    batch = sch.next_batch()
    assert "_scheduler_v6_aligned_info" in batch
