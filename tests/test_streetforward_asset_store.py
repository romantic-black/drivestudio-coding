from __future__ import annotations

import numpy as np
import pytest

from datasets.streetforward_assets import StreetForwardAssetStore
from datasets.streetforward_assets.asset_store import stable_scene_asset_id_suffix


def test_stable_scene_asset_id_suffix_is_process_independent():
    scene_index_arrays = {
        "scene_id": np.asarray([1], dtype=np.int32),
        "train_frame_indices": np.asarray([0, 1], dtype=np.int32),
        "test_frame_indices": np.asarray([2], dtype=np.int32),
        "keyframe_indices": np.asarray([0], dtype=np.int32),
        "keyframe_to_frames_flat": np.asarray([0, 1], dtype=np.int32),
        "keyframe_to_frames_offsets": np.asarray([0, 2], dtype=np.int64),
        "segment_ids": np.asarray([0], dtype=np.int32),
        "segment_frame_indices_flat": np.asarray([0, 1], dtype=np.int32),
        "segment_frame_offsets": np.asarray([0, 2], dtype=np.int64),
        "segment_keyframe_indices_flat": np.asarray([0], dtype=np.int32),
        "segment_keyframe_offsets": np.asarray([0, 1], dtype=np.int64),
    }
    a = stable_scene_asset_id_suffix(
        dataset="nuscenes",
        scene_id=1,
        num_frames=10,
        num_cams=2,
        split_config={"test_image_stride": 0},
        scene_index_arrays=scene_index_arrays,
        image_table_rows=[
            {
                "frame_idx": 0,
                "cam_id": 0,
                "img_idx": 0,
                "height": 10,
                "width": 20,
                "image_path": "/tmp/a.jpg",
                "depth_path": "/tmp/a.npy",
                "sky_mask_path": "/tmp/a.png",
                "dynamic_mask_path": "/tmp/a_dyn.png",
            }
        ],
    )
    b = stable_scene_asset_id_suffix(
        dataset="nuscenes",
        scene_id=1,
        num_frames=10,
        num_cams=2,
        split_config={"test_image_stride": 0},
        scene_index_arrays=scene_index_arrays,
        image_table_rows=[
            {
                "frame_idx": 0,
                "cam_id": 0,
                "img_idx": 0,
                "height": 10,
                "width": 20,
                "image_path": "/tmp/a.jpg",
                "depth_path": "/tmp/a.npy",
                "sky_mask_path": "/tmp/a.png",
                "dynamic_mask_path": "/tmp/a_dyn.png",
            }
        ],
    )
    assert a == b
    assert len(a) == 8


def test_segment_asset_roundtrip(tmp_path):
    store = StreetForwardAssetStore(str(tmp_path), missing_policy="error")
    scene_asset_id = store.export_scene_asset(
        dataset="nuscenes",
        scene_id=1,
        scene_name="000001",
        num_frames=10,
        num_cams=2,
        split_config={"test_image_stride": 0},
        scene_index_arrays={
            "scene_id": np.asarray([1], dtype=np.int32),
            "train_frame_indices": np.asarray([0, 1], dtype=np.int32),
            "test_frame_indices": np.asarray([2], dtype=np.int32),
            "keyframe_indices": np.asarray([0], dtype=np.int32),
            "keyframe_to_frames_flat": np.asarray([0, 1], dtype=np.int32),
            "keyframe_to_frames_offsets": np.asarray([0, 2], dtype=np.int64),
            "segment_ids": np.asarray([0], dtype=np.int32),
            "segment_frame_indices_flat": np.asarray([0, 1], dtype=np.int32),
            "segment_frame_offsets": np.asarray([0, 2], dtype=np.int64),
            "segment_keyframe_indices_flat": np.asarray([0], dtype=np.int32),
            "segment_keyframe_offsets": np.asarray([0, 1], dtype=np.int64),
        },
        image_table_rows=[
            {
                "frame_idx": 0,
                "cam_id": 0,
                "img_idx": 0,
                "is_train": True,
                "is_test": False,
                "image_path": "/tmp/im0.jpg",
                "depth_path": "/tmp/d0.npy",
                "sky_mask_path": "/tmp/s0.png",
                "dynamic_mask_path": "/tmp/m0.png",
                "height": 10,
                "width": 20,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            },
            {
                "frame_idx": 0,
                "cam_id": 1,
                "img_idx": 1,
                "is_train": True,
                "is_test": False,
                "image_path": "/tmp/im1.jpg",
                "depth_path": "/tmp/d1.npy",
                "sky_mask_path": "/tmp/s1.png",
                "dynamic_mask_path": "/tmp/m1.png",
                "height": 10,
                "width": 20,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            },
        ],
    )
    seg_asset_id = store.export_segment_asset(
        dataset="nuscenes",
        scene_id=1,
        segment_id=0,
        parent_scene_asset_id=scene_asset_id,
        segment_index_payload={
            "num_cams": 2,
            "frame_indices": [0, 1],
            "test_frame_indices": [2],
            "keyframe_indices": [0],
            "keyframe_to_frames": {0: [0, 1]},
            "frame_to_keyframe": {0: 0, 1: 0},
            "segment_first_frame_idx": 0,
            "train_image_refs": np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
            "test_image_refs": np.asarray([[2, 0], [2, 1]], dtype=np.int32),
        },
        segment_pose_payload={
            "segment_first_pose_world": np.eye(4, dtype=np.float32),
            "world_to_seg0": np.eye(4, dtype=np.float32),
            "segment_first_frame_idx": 0,
            "segment_pose_source": "camera",
        },
        pointcloud_payload={
            "background": np.zeros((2, 6), dtype=np.float32),
            "dynamic": {5: np.ones((1, 6), dtype=np.float32)},
            "instance_mapping": {1005: 5},
            "metadata": {"static_instance_intids": [7]},
        },
        dynamic_tracks_payload={
            "frame_indices": np.asarray([0, 1], dtype=np.int32),
            "instance_intids": np.asarray([5], dtype=np.int32),
            "instances_quats": np.asarray([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=np.float32),
            "instances_trans": np.asarray([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
            "instances_fv": np.asarray([[1], [1]], dtype=np.uint8),
            "static_instance_intids": np.asarray([7], dtype=np.int32),
        },
        segment_aabb=np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float32),
        pointcloud_config_normalized={"type": "hybrid"},
        stats={"background_points": 2},
    )

    assert seg_asset_id.startswith("seg-nuscenes-")
    handle = store.verify_segment_asset("nuscenes", 1, 0)
    manifest = handle.load_manifest()
    assert manifest["asset_id"] == seg_asset_id
    sidx = handle.load_segment_index()
    assert sidx["num_cams"] == 2
    pc = handle.load_pointcloud()
    assert "background" in pc
    assert 5 in pc["dynamic"]
    tracks = handle.load_dynamic_tracks()
    assert tracks["instances_quats"].shape == (2, 1, 4)
    scene_handle = store.get_scene_asset("nuscenes", 1)
    rows = scene_handle.load_image_meta([(0, 0), (0, 1)])
    assert len(rows) == 2
    assert rows[0]["img_idx"] == 0
    with pytest.raises(ValueError, match="not found"):
        scene_handle.load_image_meta([(99, 0)])


def test_registry_first_resolve_segment_and_parent_scene(tmp_path):
    store = StreetForwardAssetStore(str(tmp_path), missing_policy="error")
    scene_asset_id = store.export_scene_asset(
        dataset="nuscenes",
        scene_id=3,
        scene_name="000003",
        num_frames=2,
        num_cams=1,
        split_config={"test_image_stride": 0},
        scene_index_arrays={
            "scene_id": np.asarray([3], dtype=np.int32),
            "train_frame_indices": np.asarray([0], dtype=np.int32),
            "test_frame_indices": np.asarray([1], dtype=np.int32),
            "keyframe_indices": np.asarray([0], dtype=np.int32),
            "keyframe_to_frames_flat": np.asarray([0], dtype=np.int32),
            "keyframe_to_frames_offsets": np.asarray([0, 1], dtype=np.int64),
            "segment_ids": np.asarray([0], dtype=np.int32),
            "segment_frame_indices_flat": np.asarray([0], dtype=np.int32),
            "segment_frame_offsets": np.asarray([0, 1], dtype=np.int64),
            "segment_keyframe_indices_flat": np.asarray([0], dtype=np.int32),
            "segment_keyframe_offsets": np.asarray([0, 1], dtype=np.int64),
        },
        image_table_rows=[
            {
                "frame_idx": 0,
                "cam_id": 0,
                "img_idx": 0,
                "is_train": True,
                "is_test": False,
                "image_path": "/tmp/im0.jpg",
                "depth_path": "/tmp/d0.npy",
                "sky_mask_path": "/tmp/s0.png",
                "dynamic_mask_path": "/tmp/m0.png",
                "height": 10,
                "width": 20,
                "intrinsic_4x4_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
                "camera_to_world_flat": np.eye(4, dtype=np.float32).reshape(-1).tolist(),
            }
        ],
    )
    store.export_segment_asset(
        dataset="nuscenes",
        scene_id=3,
        segment_id=0,
        parent_scene_asset_id=scene_asset_id,
        segment_index_payload={
            "num_cams": 1,
            "frame_indices": [0],
            "test_frame_indices": [1],
            "keyframe_indices": [0],
            "keyframe_to_frames": {0: [0]},
            "frame_to_keyframe": {0: 0},
            "segment_first_frame_idx": 0,
            "train_image_refs": np.asarray([[0, 0]], dtype=np.int32),
            "test_image_refs": np.asarray([[1, 0]], dtype=np.int32),
        },
        segment_pose_payload={
            "segment_first_pose_world": np.eye(4, dtype=np.float32),
            "world_to_seg0": np.eye(4, dtype=np.float32),
            "segment_first_frame_idx": 0,
            "segment_pose_source": "camera",
        },
        pointcloud_payload={
            "background": np.zeros((1, 6), dtype=np.float32),
            "dynamic": {},
            "instance_mapping": {},
            "metadata": {},
        },
        dynamic_tracks_payload={
            "frame_indices": np.asarray([0], dtype=np.int32),
            "instance_intids": np.asarray([], dtype=np.int32),
            "instances_quats": np.zeros((1, 0, 4), dtype=np.float32),
            "instances_trans": np.zeros((1, 0, 3), dtype=np.float32),
            "instances_fv": np.zeros((1, 0), dtype=np.uint8),
            "static_instance_intids": np.asarray([], dtype=np.int32),
        },
        segment_aabb=np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float32),
        pointcloud_config_normalized={"type": "hybrid"},
        stats={"background_points": 1},
    )
    resolved = store.resolve_segment_scene_assets_registry_first("nuscenes", 3, 0)
    assert str(resolved["segment_manifest"]["parent_scene_asset_id"]) == scene_asset_id
    scene_manifest = resolved["scene_handle"].load_manifest()
    assert scene_manifest["asset_id"] == scene_asset_id


def test_get_scene_asset_by_asset_id_validates_scene_id(tmp_path):
    store = StreetForwardAssetStore(str(tmp_path), missing_policy="error")
    scene_asset_id = store.export_scene_asset(
        dataset="nuscenes",
        scene_id=7,
        scene_name="000007",
        num_frames=1,
        num_cams=1,
        split_config={"test_image_stride": 0},
        scene_index_arrays={
            "scene_id": np.asarray([7], dtype=np.int32),
            "train_frame_indices": np.asarray([0], dtype=np.int32),
            "test_frame_indices": np.asarray([], dtype=np.int32),
            "keyframe_indices": np.asarray([0], dtype=np.int32),
            "keyframe_to_frames_flat": np.asarray([0], dtype=np.int32),
            "keyframe_to_frames_offsets": np.asarray([0, 1], dtype=np.int64),
            "segment_ids": np.asarray([0], dtype=np.int32),
            "segment_frame_indices_flat": np.asarray([0], dtype=np.int32),
            "segment_frame_offsets": np.asarray([0, 1], dtype=np.int64),
            "segment_keyframe_indices_flat": np.asarray([0], dtype=np.int32),
            "segment_keyframe_offsets": np.asarray([0, 1], dtype=np.int64),
        },
        image_table_rows=[],
    )
    with pytest.raises(ValueError, match="scene_id mismatch"):
        store.get_scene_asset_by_asset_id(scene_asset_id, dataset="nuscenes", scene_id=8)
