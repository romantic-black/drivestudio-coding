from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tools.streetforward_asset_export_preflight import (
    WAYMO_DEFAULT_CAMERAS,
    prepare_streetforward_asset_export_config,
)


def _write_text(path: Path, text: str = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_waymo_scene(
    root: Path,
    *,
    scene_id: int = 546,
    frames: int = 2,
    depth_cameras: tuple[int, ...] = WAYMO_DEFAULT_CAMERAS,
    layout_cameras: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> None:
    scene = root / f"{scene_id:03d}"
    for frame_idx in range(frames):
        _write_text(scene / "ego_pose" / f"{frame_idx:03d}.txt")
        _touch(scene / "lidar" / f"{frame_idx:03d}.bin")
        for cam_id in layout_cameras:
            _touch(scene / "images" / f"{frame_idx:03d}_{cam_id}.jpg")
        for cam_id in depth_cameras:
            _touch(scene / "depth" / f"{frame_idx:03d}_{cam_id}.npy")
    for cam_id in layout_cameras:
        _write_text(scene / "extrinsics" / f"{cam_id}.txt")
        _write_text(scene / "intrinsics" / f"{cam_id}.txt", "1 1 0 0 0 0 0 0 0\n")


def _waymo_cfg(
    root: Path,
    *,
    cameras: list[int] | None = None,
    load_depth_maps: bool | None = True,
    monocular_chosen_cam_ids: list[int] | None = None,
) -> OmegaConf:
    pixel_source = {
        "type": "datasets.waymo.waymo_sourceloader.WaymoPixelSource",
        "downscale_when_loading": [2, 2, 2, 2, 2],
        "undistort": False,
        "load_sky_mask": True,
        "load_dynamic_mask": True,
        "load_objects": True,
        "load_smpl": False,
    }
    if cameras is not None:
        pixel_source["cameras"] = cameras
        pixel_source["downscale_when_loading"] = [2 for _ in cameras]
    if load_depth_maps is not None:
        pixel_source["load_depth_maps"] = load_depth_maps

    pointcloud = {
        "type": "hybrid",
        "lidar_sparsity": "full",
        "monocular_sparsity": "full",
        "monocular_filter_sky": True,
        "monocular_depth_consistency": False,
        "monocular_downscale": 1,
        "near_max_points": 100,
        "distant_max_points": 100,
        "monocular_dynamic_recovery_max_points_per_instance": 10,
    }
    if monocular_chosen_cam_ids is not None:
        pointcloud["monocular_chosen_cam_ids"] = monocular_chosen_cam_ids

    return OmegaConf.create(
        {
            "data": {
                "data_root": str(root),
                "dataset": "waymo",
                "start_timestep": 0,
                "end_timestep": 1,
                "train_scene_ids": [546],
                "pixel_source": pixel_source,
                "lidar_source": {
                    "type": "datasets.waymo.waymo_sourceloader.WaymoLiDARSource",
                    "load_lidar": True,
                    "only_use_top_lidar": False,
                    "truncated_max_range": 80,
                    "truncated_min_range": -2,
                    "lidar_downsample_factor": 4,
                    "lidar_percentile": 0.02,
                },
            },
            "dataset": {
                "segment_aabb": [[-20.0, -10.0, -5.0], [20.0, 4.8, 80.0]],
                "pointcloud": pointcloud,
            },
        }
    )


def test_waymo_preflight_defaults_to_front_three_cameras(tmp_path):
    _make_waymo_scene(tmp_path, depth_cameras=WAYMO_DEFAULT_CAMERAS)
    cfg = _waymo_cfg(
        tmp_path,
        cameras=None,
        load_depth_maps=None,
        monocular_chosen_cam_ids=None,
    )

    prepare_streetforward_asset_export_config(cfg)

    assert list(cfg.data.pixel_source.cameras) == [0, 1, 2]
    assert bool(cfg.data.pixel_source.load_depth_maps) is True
    assert list(cfg.dataset.pointcloud.monocular_chosen_cam_ids) == [0, 1, 2]


def test_waymo_preflight_rejects_side_cameras_when_depth_is_missing(tmp_path):
    _make_waymo_scene(tmp_path, depth_cameras=WAYMO_DEFAULT_CAMERAS)
    cfg = _waymo_cfg(tmp_path, cameras=[0, 1, 2, 3, 4], load_depth_maps=True)

    with pytest.raises(ValueError, match="side cameras 3/4"):
        prepare_streetforward_asset_export_config(cfg)


def test_waymo_preflight_rejects_monocular_cameras_not_loaded(tmp_path):
    _make_waymo_scene(tmp_path, depth_cameras=WAYMO_DEFAULT_CAMERAS)
    cfg = _waymo_cfg(
        tmp_path,
        cameras=[0, 1, 2],
        load_depth_maps=True,
        monocular_chosen_cam_ids=[0, 3],
    )

    with pytest.raises(ValueError, match="monocular_chosen_cam_ids must be a subset"):
        prepare_streetforward_asset_export_config(cfg)


def test_nuscenes_preflight_is_noop(tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset": "nuscenes",
                "data_root": str(tmp_path),
                "pixel_source": {"cameras": [0, 1, 2, 3, 4, 5]},
            },
            "dataset": {
                "pointcloud": {"monocular_chosen_cam_ids": [0, 1, 2, 3, 4, 5]},
            },
        }
    )

    prepare_streetforward_asset_export_config(cfg)

    assert list(cfg.data.pixel_source.cameras) == [0, 1, 2, 3, 4, 5]
