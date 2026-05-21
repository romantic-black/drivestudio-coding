"""Tests for dynamic/static point cloud split policy."""

import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "open3d" not in sys.modules:
    fake_open3d = types.ModuleType("open3d")
    fake_open3d.geometry = types.SimpleNamespace(PointCloud=object)
    fake_open3d.utility = types.SimpleNamespace(Vector3dVector=lambda x: x)
    sys.modules["open3d"] = fake_open3d

from datasets.pointcloud_generators.lidar import LiDARRGBPointCloudGenerator
from datasets.pointcloud_generators.monocular import MonocularRGBPointCloudGenerator


def _instance(intid: int, center_x: float = 0.0):
    T_ow = np.eye(4, dtype=np.float32)
    T_ow[0, 3] = float(center_x)
    return {
        "intid": int(intid),
        "T_ow": T_ow,
        "size_lwh": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    }


def _mono_generator():
    return MonocularRGBPointCloudGenerator(
        chosen_cam_ids=[0],
        dynamic_recovery_enable=True,
        dynamic_recovery_bbox_expand_xyz_m=[0.0, 0.0, 0.0],
        dynamic_recovery_max_points_per_instance=10,
    )


def test_monocular_background_removes_moving_bbox_when_dynamic_mask_misses():
    gen = _mono_generator()
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]], dtype=np.float32)
    pixels_yx = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    frame_data = {"dynamic_mask": np.zeros((1, 2), dtype=np.float32)}

    bg_points, bg_colors = gen._filter_background_points_with_instance_policy(
        points,
        colors,
        pixels_yx,
        frame_data,
        [_instance(0)],
        set(),
    )

    assert bg_points.shape == (1, 3)
    assert np.allclose(bg_points[0], [2.0, 0.0, 0.0])
    assert np.allclose(bg_colors[0], [0.0, 255.0, 0.0])


def test_monocular_background_keeps_static_bbox_when_dynamic_mask_marks_it():
    gen = _mono_generator()
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[255.0, 0.0, 0.0]], dtype=np.float32)
    pixels_yx = np.asarray([[0, 0]], dtype=np.int64)
    frame_data = {"dynamic_mask": np.ones((1, 1), dtype=np.float32)}

    bg_points, bg_colors = gen._filter_background_points_with_instance_policy(
        points,
        colors,
        pixels_yx,
        frame_data,
        [_instance(0)],
        {0},
    )

    assert bg_points.shape == (1, 3)
    assert np.allclose(bg_points[0], [0.0, 0.0, 0.0])
    assert np.allclose(bg_colors[0], [255.0, 0.0, 0.0])


def test_monocular_dynamic_recovery_skips_static_instances():
    gen = _mono_generator()
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]], dtype=np.float32)
    instances = [_instance(0), _instance(1, center_x=2.0)]
    moving_instances, stationary_instances = gen._split_instances_by_static_intids(instances, {1})

    dynamic, recovered = gen._recover_dynamic_points_by_3d_bbox(
        points,
        colors,
        moving_instances,
    )

    assert [int(x["intid"]) for x in stationary_instances] == [1]
    assert recovered == 1
    assert sorted(dynamic.keys()) == [0]


def test_lidar_static_instance_stays_background():
    gen = LiDARRGBPointCloudGenerator()
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[255.0, 0.0, 0.0]], dtype=np.float32)

    background, dynamic = gen._separate_static_dynamic(
        points,
        colors,
        [_instance(0)],
        skip_instance_intids={0},
    )

    assert dynamic == {}
    assert background.shape == (1, 6)
    assert np.allclose(background[0, :3], [0.0, 0.0, 0.0])
