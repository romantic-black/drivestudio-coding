"""Tests for pointcloud_generators.motion_utils (load module without datasets/__init__ chain)."""

import importlib.util
import os

import torch

_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "pointcloud_generators")
_spec = importlib.util.spec_from_file_location(
    "motion_utils",
    os.path.join(_pkg_dir, "motion_utils.py"),
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
compute_static_instance_intids = _mod.compute_static_instance_intids


class _FakePixelSource:
    def __init__(self, instances_pose, per_frame_instance_mask):
        self.instances_pose = instances_pose
        self.per_frame_instance_mask = per_frame_instance_mask


def test_static_instance_all_stationary():
    # 2 frames, 2 instances: no motion -> both static at thresh 0.5
    pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
    mask = torch.ones(2, 2, dtype=torch.bool)
    ps = _FakePixelSource(pose, mask)
    static_ids = compute_static_instance_intids(ps, [0, 1], 0.5)
    assert static_ids == {0, 1}


def test_static_instance_one_moving():
    pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
    pose[1, 0, 0, 3] = 1.0  # instance 0 moves 1m between frames
    mask = torch.ones(2, 2, dtype=torch.bool)
    ps = _FakePixelSource(pose, mask)
    static_ids = compute_static_instance_intids(ps, [0, 1], 0.5)
    assert static_ids == {1}


def test_single_visible_frame_is_static():
    pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    mask[0, 0] = True
    ps = _FakePixelSource(pose, mask)
    static_ids = compute_static_instance_intids(ps, [0, 1], 0.5)
    assert static_ids == {0}
