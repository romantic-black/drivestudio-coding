from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.multi_scene_dataset_v4 import BatchRequestV4, MultiSceneDatasetV4
from datasets.streetforward_assets import StreetForwardAssetStore
from datasets.train_scheduler_v9 import StepPlanV9, ViewSetRolloutBatchV9


def _prepare_demo_assets(
    tmp_path,
    *,
    tracks_all_invisible: bool = False,
    instances_fv_override: np.ndarray | None = None,
    pointcloud_config_normalized: dict | None = None,
    pointcloud_payload: dict | None = None,
    knn_payload: dict | None = None,
    coordinate_metadata: dict | None = None,
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

    if pointcloud_payload is None:
        pointcloud_payload = {
            "background": np.zeros((2, 6), dtype=np.float32),
            "dynamic": {9: np.ones((1, 6), dtype=np.float32)},
            "instance_mapping": {1009: 9},
            "metadata": {"static_instance_intids": []},
        }
    if pointcloud_config_normalized is None:
        pointcloud_config_normalized = {"type": "hybrid"}
    bg_n = int(np.asarray(pointcloud_payload["background"]).shape[0])
    dyn_n = sum(int(np.asarray(v).shape[0]) for v in pointcloud_payload.get("dynamic", {}).values())
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
        pointcloud_payload=pointcloud_payload,
        dynamic_tracks_payload={
            "frame_indices": np.asarray([0, 1], dtype=np.int32),
            "instance_intids": np.asarray([9], dtype=np.int32),
            "instances_quats": np.asarray([[[1, 0, 0, 0]], [[1, 0, 0, 0]]], dtype=np.float32),
            "instances_trans": np.asarray([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.float32),
            "instances_fv": instances_fv,
            "static_instance_intids": np.asarray([], dtype=np.int32),
        },
        segment_aabb=np.asarray([[-1, -1, -1], [1, 1, 1]], dtype=np.float32),
        pointcloud_config_normalized=pointcloud_config_normalized,
        stats={"background_points": bg_n, "dynamic_points": dyn_n},
        coordinate_metadata=coordinate_metadata,
    )
    if knn_payload is not None:
        store.export_segment_knn_init_asset(
            dataset="nuscenes",
            scene_id=1,
            segment_id=0,
            knn_payload=knn_payload,
            overwrite=True,
        )
    return store


def _build_cfg(tmp_path, *, pointcloud: dict | None = None):
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
    ds: dict = {
        "segment_aabb": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    }
    if pointcloud is not None:
        ds["pointcloud"] = pointcloud
    dataset_cfg = OmegaConf.create(ds)
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


def test_v4_filters_segments_with_too_many_dynamic_points(tmp_path):
    store = _prepare_demo_assets(
        tmp_path,
        pointcloud_payload={
            "background": np.zeros((2, 6), dtype=np.float32),
            "dynamic": {9: np.ones((3, 6), dtype=np.float32)},
            "instance_mapping": {1009: 9},
            "metadata": {"static_instance_intids": []},
        },
    )
    data_cfg, dataset_cfg = _build_cfg(
        tmp_path,
        pointcloud={"max_dynamic_points_per_segment": 2},
    )
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    assert ds.list_segment_ids(1) == []
    with pytest.raises(ValueError, match="No training scenes remain after filtering segments"):
        ds.list_training_scene_ids()


def test_v4_preload_worker_warms_cpu_view_cache_without_materializing(tmp_path, monkeypatch):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()

    def _fail_materialize(*_args, **_kwargs):
        raise AssertionError("preload worker should not materialize cached view packs")

    monkeypatch.setattr(
        "datasets.multi_scene_dataset_v4.loaded_view_pack_to_device_v2",
        _fail_materialize,
    )
    ds._preload_worker_view_pack(1, 0, (0, 0), {})
    assert len(ds._view_pack_cache) == 1
    cached = next(iter(ds._view_pack_cache.values()))
    assert cached.image.device.type == "cpu"
    assert cached.depth.device.type == "cpu"
    assert cached.viewdirs is not None
    assert cached.viewdirs.device.type == "cpu"


def test_v4_loads_egocar_mask_from_static_template(tmp_path, monkeypatch):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    data_cfg["pixel_source"]["load_egocar_mask"] = True

    ego_root = tmp_path / "data" / "ego_masks" / "nuscenes"
    ego_root.mkdir(parents=True, exist_ok=True)
    ego_np = np.zeros((4, 5), dtype=np.uint8)
    ego_np[0, 0] = 255
    ego_np[1, 2] = 255
    Image.fromarray(ego_np).save(ego_root / "0.png")

    monkeypatch.chdir(tmp_path)
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
    assert "egocar_mask" in batch["source"]
    assert "egocar_mask" in batch["target"]
    assert batch["source"]["egocar_mask"].shape == (1, 4, 5)
    assert batch["target"]["egocar_mask"].shape == (2, 4, 5)
    expected = torch.from_numpy((ego_np > 0).astype(np.float32))
    assert torch.allclose(batch["source"]["egocar_mask"][0], expected)
    assert torch.allclose(batch["target"]["egocar_mask"][0], expected)
    assert torch.allclose(batch["target"]["egocar_mask"][1], expected)


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


def test_v4_v9_request_materializes_role_batches(tmp_path):
    store = _prepare_demo_assets(tmp_path)
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    step = StepPlanV9(
        step_idx=0,
        source_keyframe_idx=0,
        source_frame_idx=0,
        block_idx=0,
        evidence_refs=[(0, 0)],
        block_loss_refs=[(0, 0)],
        nearby_loss_refs=[],
        prefix_loss_refs=[],
        query_label_refs=[(1, 0)],
        aux_loss_refs=[],
        evidence_frame_indices=[0],
        loss_frame_indices=[0],
        nearby_frame_indices=[],
        query_frame_indices=[1],
    )
    plan = ViewSetRolloutBatchV9(
        scheduler_version="v9",
        phase="phase_B_viewset_rollout",
        scene_id=1,
        segment_id=0,
        episode_id=0,
        episode_start_keyframe_pos=0,
        keyframe_window=[0, 1],
        frame_chain=[0, 1],
        num_cams=1,
        inner_K=1,
        steps=[step],
        evidence_refs_by_step=[[(0, 0)]],
        block_loss_refs_by_step=[[(0, 0)]],
        nearby_loss_refs_by_step=[[]],
        prefix_loss_refs_by_step=[[]],
        query_label_refs=[(1, 0)],
        aux_loss_refs=[],
        request_meta={
            "scheduler_version": "v9",
            "scheduler_phase": "phase_B_viewset_rollout",
            "target_image_refs": [(0, 0)],
            "target_image_roles": ["block_loss"],
            "query_label_refs": [(1, 0)],
        },
    )
    batch = ds._assemble_segment_batch_from_v9_request(
        scene_id=1,
        segment_id=0,
        v9_plan=plan,
        include_test=False,
    )
    assert batch["source"]["image"].shape == (1, 4, 5, 3)
    assert batch["target"]["image"].shape == (1, 4, 5, 3)
    assert batch["query_label"]["image"].shape == (1, 4, 5, 3)
    assert batch["_scheduler_v9"]["scheduler_version"] == "v9"
    assert batch["request_meta"]["assembly_mode"] == "image_ref_v9"
    assert batch["request_meta"]["source_image_refs"] == [(0, 0)]
    assert batch["request_meta"]["target_image_refs"] == [(0, 0)]
    assert batch["request_meta"]["target_image_roles"] == ["block_loss"]
    assert batch["request_meta"]["query_label_refs"] == [(1, 0)]


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


def test_v4_accepts_standard_coordinate_metadata(tmp_path):
    MultiSceneDatasetV4._validate_segment_coordinate_metadata(
        {"asset_coordinate_frame": "seg0_camera_opencv"},
        scene_id=1,
        segment_id=0,
        context="test",
    )


def test_v4_rejects_unknown_coordinate_metadata(tmp_path):
    with pytest.raises(ValueError, match="Unsupported StreetForward segment asset coordinate frame"):
        MultiSceneDatasetV4._validate_segment_coordinate_metadata(
            {"coordinate_metadata": {"asset_coordinate_frame": "waymo_native_ego"}},
            scene_id=1,
            segment_id=0,
            context="test",
        )


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


def test_v4_runtime_cap_helper_downsamples_without_knn():
    ds = MultiSceneDatasetV4.__new__(MultiSceneDatasetV4)
    ds._runtime_pointcloud_cfg = {
        "near_max_points": 3,
        "distant_max_points": 2,
        "monocular_dynamic_recovery_max_points_per_instance": 2,
    }
    ds.segment_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

    near = np.zeros((5, 6), dtype=np.float32)
    distant = np.zeros((4, 6), dtype=np.float32)
    distant[:, 0] = 3.0
    pointcloud = {
        "background": np.concatenate([near, distant], axis=0),
        "dynamic": {
            9: np.ones((4, 6), dtype=np.float32),
            10: np.ones((1, 6), dtype=np.float32),
        },
    }

    out = ds._apply_runtime_pointcloud_caps(
        pointcloud=pointcloud,
        scene_id=1,
        segment_id=0,
        context="test",
    )

    assert int(out["background"].shape[0]) == 5
    assert int(out["dynamic"][9].shape[0]) == 2
    assert int(out["dynamic"][10].shape[0]) == 1


def test_v4_runtime_cap_mismatch_random_downsamples(tmp_path):
    inside_pts = np.zeros((60, 6), dtype=np.float32)
    outside_pts = np.zeros((80, 6), dtype=np.float32)
    outside_pts[:, 0] = 3.0
    background = np.concatenate([inside_pts, outside_pts], axis=0)
    store = _prepare_demo_assets(
        tmp_path,
        pointcloud_config_normalized={
            "type": "hybrid",
            "near_max_points": 1000,
            "distant_max_points": 1000,
            "monocular_dynamic_recovery_max_points_per_instance": 1000,
        },
        pointcloud_payload={
            "background": background,
            "dynamic": {9: np.ones((20, 6), dtype=np.float32)},
            "instance_mapping": {1009: 9},
            "metadata": {"static_instance_intids": []},
        },
    )
    data_cfg, dataset_cfg = _build_cfg(
        tmp_path,
        pointcloud={
            "type": "hybrid",
            "near_max_points": 30,
            "distant_max_points": 40,
            "monocular_dynamic_recovery_max_points_per_instance": 5,
        },
    )
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    bundle = ds._resolve_segment_bundle(1, 0)
    assert bundle.pointcloud["background"].shape[0] == 70
    assert bundle.pointcloud["dynamic"][9].shape[0] == 5


def test_v4_matching_pointcloud_caps_skips_runtime_downsample(tmp_path):
    inside_pts = np.zeros((60, 6), dtype=np.float32)
    outside_pts = np.zeros((80, 6), dtype=np.float32)
    outside_pts[:, 0] = 3.0
    background = np.concatenate([inside_pts, outside_pts], axis=0)
    store = _prepare_demo_assets(
        tmp_path,
        pointcloud_config_normalized={
            "type": "hybrid",
            "near_max_points": 1000,
            "distant_max_points": 1000,
            "monocular_dynamic_recovery_max_points_per_instance": 1000,
        },
        pointcloud_payload={
            "background": background,
            "dynamic": {9: np.ones((20, 6), dtype=np.float32)},
            "instance_mapping": {1009: 9},
            "metadata": {"static_instance_intids": []},
        },
    )
    data_cfg, dataset_cfg = _build_cfg(
        tmp_path,
        pointcloud={
            "type": "hybrid",
            "near_max_points": 1000,
            "distant_max_points": 1000,
            "monocular_dynamic_recovery_max_points_per_instance": 1000,
        },
    )
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
    )
    ds.initialize()
    bundle = ds._resolve_segment_bundle(1, 0)
    assert bundle.pointcloud["background"].shape[0] == 140
    assert bundle.pointcloud["dynamic"][9].shape[0] == 20


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


def test_v4_fixed_knn_overprovisioned_random_subsampled_to_required(tmp_path, monkeypatch):
    pointcloud_payload = {
        "background": np.zeros((2, 6), dtype=np.float32),
        "dynamic": {9: np.ones((3, 6), dtype=np.float32)},
        "instance_mapping": {1009: 9},
        "metadata": {"static_instance_intids": []},
    }
    bg_knn_src = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    rigid_knn_src = np.asarray(
        [
            [0, 1, 2, 1],
            [1, 0, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    store = _prepare_demo_assets(
        tmp_path,
        pointcloud_payload=pointcloud_payload,
        knn_payload={
            "background_avg_dist_by_k": {},
            "dynamic_avg_dist_by_k": {},
            "bg_knn_idx": bg_knn_src,
            "rigid_knn_idx": rigid_knn_src,
            "knn_neighbor_k_store": 4,
        },
    )
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
        knn_requirements={
            "enabled": True,
            "fixed_neighbor_enabled": True,
            "neighbor_k_store": 2,
        },
    )
    ds.initialize()

    def _mock_choice(a, size, replace=False):
        assert int(a) == 4
        assert int(size) == 2
        assert not bool(replace)
        return np.asarray([3, 1], dtype=np.int64)

    monkeypatch.setattr(np.random, "choice", _mock_choice)
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(0, 0), (1, 0)],
        include_test=False,
    )
    batch = ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
    knn_struct = batch["knn_struct_neighbors"]
    assert int(knn_struct["knn_neighbor_k_store"]) == 2
    expected_cols = np.asarray([1, 3], dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(knn_struct["bg_knn_idx"]), bg_knn_src[:, expected_cols])
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_knn_idx"]), rigid_knn_src[:, expected_cols])
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_knn_row_ids"]), np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_instance_intids"]), np.asarray([9], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_instance_offsets"]), np.asarray([0, 3], dtype=np.int64))


def test_v4_fixed_knn_invisible_dynamic_window_slices_rigid_knn_rows(tmp_path, monkeypatch):
    pointcloud_payload = {
        "background": np.zeros((2, 6), dtype=np.float32),
        "dynamic": {9: np.ones((3, 6), dtype=np.float32)},
        "instance_mapping": {1009: 9},
        "metadata": {"static_instance_intids": []},
    }
    bg_knn_src = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int64,
    )
    rigid_knn_src = np.asarray(
        [
            [0, 1, 2, 1],
            [1, 0, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    store = _prepare_demo_assets(
        tmp_path,
        tracks_all_invisible=True,
        pointcloud_payload=pointcloud_payload,
        knn_payload={
            "background_avg_dist_by_k": {},
            "dynamic_avg_dist_by_k": {},
            "bg_knn_idx": bg_knn_src,
            "rigid_knn_idx": rigid_knn_src,
            "knn_neighbor_k_store": 4,
        },
    )
    data_cfg, dataset_cfg = _build_cfg(tmp_path)
    ds = MultiSceneDatasetV4(
        dataset_cfg=dataset_cfg,
        data_cfg=data_cfg,
        device=torch.device("cpu"),
        asset_store=store,
        knn_requirements={
            "enabled": True,
            "fixed_neighbor_enabled": True,
            "neighbor_k_store": 2,
        },
    )
    ds.initialize()

    def _mock_choice(a, size, replace=False):
        assert int(a) == 4
        assert int(size) == 2
        assert not bool(replace)
        return np.asarray([3, 1], dtype=np.int64)

    monkeypatch.setattr(np.random, "choice", _mock_choice)
    req = BatchRequestV4(
        scene_id=1,
        segment_id=0,
        source_image_ref=(0, 0),
        target_image_refs=[(0, 0), (1, 0)],
        include_test=False,
    )
    batch = ds.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
    assert "dynamic_info" in batch
    assert isinstance(batch["dynamic_info"], dict)
    assert sorted(int(x) for x in batch["dynamic_info"].keys()) == [0, 1]
    assert all(len(v.get("instances", {})) == 0 for v in batch["dynamic_info"].values())
    assert set(int(x) for x in batch["pointcloud"]["dynamic"].keys()) == {9}
    knn_struct = batch["knn_struct_neighbors"]
    assert int(knn_struct["knn_neighbor_k_store"]) == 2
    assert np.asarray(knn_struct["bg_knn_idx"]).shape == (2, 2)
    assert np.asarray(knn_struct["rigid_knn_idx"]).shape == (3, 2)
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_knn_row_ids"]), np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_instance_intids"]), np.asarray([9], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(knn_struct["rigid_instance_offsets"]), np.asarray([0, 3], dtype=np.int64))
