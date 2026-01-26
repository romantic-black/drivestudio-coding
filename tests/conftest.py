import os
import sys
from unittest.mock import Mock
import types

import numpy as np
import pytest

# Ensure project root is on sys.path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Mock heavy optional dependencies before importing project modules
sys.modules.setdefault("cv2", Mock())
open3d_module = types.ModuleType("open3d")
open3d_module.geometry = Mock()
open3d_module.utility = Mock()
open3d_module.io = Mock()
sys.modules.setdefault("open3d", open3d_module)
nvdiffrast_module = types.ModuleType("nvdiffrast")
nvdiffrast_torch = types.ModuleType("nvdiffrast.torch")
nvdiffrast_torch.rasterize = lambda *args, **kwargs: None
nvdiffrast_module.torch = nvdiffrast_torch
sys.modules.setdefault("nvdiffrast", nvdiffrast_module)
sys.modules.setdefault("nvdiffrast.torch", nvdiffrast_torch)
neighbors_module = types.ModuleType("sklearn.neighbors")
neighbors_module.NearestNeighbors = Mock()
sklearn_module = types.ModuleType("sklearn")
sklearn_module.neighbors = neighbors_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules.setdefault("sklearn.neighbors", neighbors_module)

pytorch3d_module = types.ModuleType("pytorch3d")
pytorch3d_transforms = types.ModuleType("pytorch3d.transforms")
pytorch3d_transforms.matrix_to_quaternion = lambda x: x
pytorch3d_ops = types.ModuleType("pytorch3d.ops")
pytorch3d_ops.knn_points = lambda *args, **kwargs: None
sys.modules.setdefault("pytorch3d", pytorch3d_module)
sys.modules.setdefault("pytorch3d.transforms", pytorch3d_transforms)
sys.modules.setdefault("pytorch3d.ops", pytorch3d_ops)
pytorch3d_module.transforms = pytorch3d_transforms
pytorch3d_module.ops = pytorch3d_ops
scipy_module = types.ModuleType("scipy")
scipy_spatial = types.ModuleType("scipy.spatial")
scipy_spatial_transform = types.ModuleType("scipy.spatial.transform")


class _FakeSlerp:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


class _FakeRotation:
    @classmethod
    def from_matrix(cls, *args, **kwargs):
        return cls()

    def as_matrix(self):
        return np.eye(3)


scipy_spatial_transform.Slerp = _FakeSlerp
scipy_spatial_transform.Rotation = _FakeRotation
scipy_spatial.transform = scipy_spatial_transform
scipy_module.spatial = scipy_spatial
sys.modules.setdefault("scipy", scipy_module)
sys.modules.setdefault("scipy.spatial", scipy_spatial)
sys.modules.setdefault("scipy.spatial.transform", scipy_spatial_transform)

from datasets.multi_scene_dataset import MultiSceneDataset


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_pointcloud_patch: disable automatic pointcloud generator mocking",
    )


@pytest.fixture(autouse=True)
def _mock_pointcloud_generator(monkeypatch, request):
    """
    Provide a lightweight mock pointcloud generator for tests by default.
    
    Tests that need to exercise the real pointcloud_config requirement can opt out
    with the marker ``@pytest.mark.no_pointcloud_patch``.
    """
    if "no_pointcloud_patch" in request.keywords:
        return

    mock_generator = Mock()
    mock_generator.generate_pointcloud.return_value = {
        "background": np.zeros((0, 6), dtype=np.float32),
        "metadata": {},
    }

    # Patch generator creation to avoid heavy dependencies in tests
    def _create_pointcloud_generator(self, pointcloud_config, data_cfg, device):
        return mock_generator

    monkeypatch.setattr(
        MultiSceneDataset, "_create_pointcloud_generator", _create_pointcloud_generator
    )

    # Inject a default pointcloud_config if tests do not provide one explicitly
    orig_init = MultiSceneDataset.__init__

    def _init(self, *args, **kwargs):
        kwargs.setdefault("pointcloud_config", {"type": "mock"})
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(MultiSceneDataset, "__init__", _init)

    return mock_generator
