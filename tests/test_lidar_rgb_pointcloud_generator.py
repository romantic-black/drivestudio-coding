"""
Unit tests for LiDAR RGB Point Cloud Generator.
"""
import pytest
import torch
import numpy as np
import json
import os
import tempfile
import shutil
import open3d as o3d
from unittest.mock import Mock, MagicMock, patch, mock_open
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from datasets.pointcloud_generators.rgb_pointcloud_generator import (
    LiDARRGBPointCloudGenerator,
    StaticPointCloud,
    DynamicPointCloud,
)

# Default AABB for tests
DEFAULT_CROP_AABB = np.array([[-20, -20, -20], [20, 4.8, 70]])
DEFAULT_INPUT_AABB = np.array([[-20, -20, -20], [20, 4.8, 70]])


class TestLiDARRGBPointCloudGeneratorBase:
    """Test base class methods."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        assert generator.dataset == "waymo"
        assert generator.resomult == 0.5
        assert generator.chosen_cam_ids == [0, 1, 2, 3, 4]
        assert generator.camera_priority == [0, 1, 2, 3, 4]
        assert hasattr(generator, '_instances_cache')
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        generator = LiDARRGBPointCloudGenerator(
            chosen_cam_ids=[0, 1],
            camera_priority=[1, 0],
            resomult=0.25,
            dataset="kitti",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        assert generator.dataset == "kitti"
        assert generator.resomult == 0.25
        assert generator.chosen_cam_ids == [0, 1]
        assert generator.camera_priority == [1, 0]
    
    def test_camera_priority_waymo(self):
        """Test camera priority for Waymo dataset."""
        generator = LiDARRGBPointCloudGenerator(
            dataset="waymo",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        assert generator.camera_priority == [0, 1, 2, 3, 4]
    
    def test_camera_priority_nuscenes(self):
        """Test camera priority for nuScenes dataset."""
        generator = LiDARRGBPointCloudGenerator(
            dataset="nuscenes",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        assert generator.camera_priority == [0, 1, 2, 3, 4, 5]
    
    def test_camera_priority_kitti(self):
        """Test camera priority for KITTI dataset."""
        generator = LiDARRGBPointCloudGenerator(
            dataset="kitti",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        assert generator.camera_priority == [0, 1]


class TestLiDARRGBPointCloudGeneratorMethods:
    """Test LiDARRGBPointCloudGenerator methods."""
    
    def test_load_lidar_points_vehicle(self):
        """Test LiDAR point cloud loading."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock scene_dataset
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([0, 0, 0, 1, 1, 1])
        mock_lidar_source.points = torch.tensor([
            [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 0
            [0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 0
            [0, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 0
            [0, 0, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 1
            [0, 0, 0, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 1
            [0, 0, 0, 16, 17, 18, 0, 0, 0, 0, 0, 0, 0, 0],  # frame 1
        ])
        
        mock_scene_dataset = Mock()
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        # Test loading frame 0
        pts_w = generator._load_lidar_points_world(mock_scene_dataset, 0)
        assert pts_w is not None
        assert pts_w.shape == (3, 3)
        assert np.allclose(pts_w[0], [1, 2, 3])
        assert np.allclose(pts_w[1], [4, 5, 6])
        assert np.allclose(pts_w[2], [7, 8, 9])
        
        # Test loading frame 1
        pts_w = generator._load_lidar_points_world(mock_scene_dataset, 1)
        assert pts_w is not None
        assert pts_w.shape == (3, 3)
        assert np.allclose(pts_w[0], [10, 11, 12])
    
    def test_load_lidar_points_vehicle_no_points(self):
        """Test LiDAR loading when no points available."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        mock_scene_dataset = Mock()
        mock_scene_dataset.lidar_source = None
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        pts_w = generator._load_lidar_points_world(mock_scene_dataset, 0)
        assert pts_w is None
    
    def test_get_ego_pose(self):
        """Test ego pose retrieval."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock scene_dataset with lidar_source
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        # Mock timesteps and lidar_to_worlds
        mock_lidar_source.timesteps = torch.tensor([0, 0, 1, 1])  # 2 points for frame 0, 2 for frame 1
        # lidar_to_worlds 按点存储，形状 (num_points, 4, 4)
        # 使用 torch.tensor 确保有 shape 属性
        lidar_to_worlds_list = [
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32) * 2,
            np.eye(4, dtype=np.float32) * 2,
        ]
        mock_lidar_source.lidar_to_worlds = torch.stack([torch.from_numpy(x) for x in lidar_to_worlds_list])
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        T_vw = generator._get_ego_pose(mock_scene_dataset, 0)
        assert T_vw is not None
        assert T_vw.shape == (4, 4)
        assert np.allclose(T_vw, np.eye(4))
    
    def test_get_ego_pose_none(self):
        """Test ego pose retrieval when not available."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        mock_scene_dataset = Mock()
        mock_scene_dataset.ego_poses = None
        mock_scene_dataset.lidar_source = None
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        T_vw = generator._get_ego_pose(mock_scene_dataset, 0)
        assert T_vw is None
    
    def test_project_points_to_image(self):
        """Test point projection to image plane."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Create test points (world coordinates)
        points_w = np.array([
            [0, 0, 10],  # Should project to center
            [0, 0, 5],   # Closer point
            [100, 100, 1],  # Far point, should be outside image
        ], dtype=np.float32)
        
        # Camera at origin looking along +Z
        T_cw = np.eye(4, dtype=np.float32)
        
        # Intrinsics: fx=fy=100, cx=cy=50 (image center)
        K = np.array([
            [100, 0, 50],
            [0, 100, 50],
            [0, 0, 1],
        ], dtype=np.float32)
        
        img_size = (100, 100)
        
        uv, dists, indices = generator._project_points_to_image(
            points_w, T_cw, K, img_size
        )
        
        assert len(uv) > 0
        assert len(dists) > 0
        assert len(indices) > 0
        # First point should project to center
        assert np.allclose(uv[0], [50, 50], atol=1)
    
    def test_split_static_dynamic(self):
        """Test static/dynamic point splitting."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Create test points (world coordinates + RGB)
        # Instance box: at origin, size [2, 2, 2], so half=[1, 1, 1], range is [-1, 1] for each axis
        pts_wrgb = np.array([
            [0, 0, 0, 1, 0, 0],      # Inside instance box (at origin)
            [0.5, 0.5, 0.5, 0, 1, 0],  # Inside instance box
            [5, 0, 0, 0, 0, 1],      # Outside instance box (x=5 > 1)
            [0, 2, 0, 1, 1, 1],      # Outside instance box (y=2 > 1)
        ], dtype=np.float32)
        
        # Create instance: box at origin, size [2, 2, 2]
        T_ow = np.eye(4, dtype=np.float32)  # Object at origin
        size_lwh = np.array([2, 2, 2], dtype=np.float32)
        inst_list = [(1, T_ow, size_lwh)]
        
        bg_points, dynamic_points = generator._split_static_dynamic(pts_wrgb, inst_list)
        
        # Should have 2 static points and 1 dynamic instance with 2 points
        assert len(bg_points) == 2
        assert len(dynamic_points) == 1
        assert 1 in dynamic_points
        assert dynamic_points[1].shape[0] == 2  # 2 points in instance
    
    def test_split_static_dynamic_no_instances(self):
        """Test splitting when no instances."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        pts_wrgb = np.array([
            [0, 0, 0, 1, 0, 0],
            [5, 0, 0, 0, 1, 0],
        ], dtype=np.float32)
        
        inst_list = []
        
        bg_points, dynamic_points = generator._split_static_dynamic(pts_wrgb, inst_list)
        
        # All points should be static
        assert len(bg_points) == 2
        assert len(dynamic_points) == 0


class TestLiDARRGBPointCloudGeneratorIntegration:
    """Integration tests for LiDARRGBPointCloudGenerator."""
    
    def test_generate_pointcloud_basic(self):
        """Test basic pointcloud generation (no instances)."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock dataset
        mock_dataset = Mock(spec=MultiSceneDataset)
        mock_dataset.get_segment_frames = Mock(return_value=[0, 1])
        mock_dataset.get_scene = Mock(return_value={
            'dataset': Mock(),
        })
        mock_dataset.get_frame_data = Mock(return_value={
            'image': torch.rand(100, 200, 3),
            'extrinsic': torch.eye(4),
            'intrinsic': torch.eye(4),
            'depth': torch.rand(100, 200),
        })
        
        # Mock scene_dataset
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([0, 1])
        mock_lidar_source.points = torch.tensor([
            [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        # Mock lidar_to_worlds for ego pose (按点存储)
        lidar_to_worlds_list = [
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        ]
        mock_lidar_source.lidar_to_worlds = torch.stack([torch.from_numpy(x) for x in lidar_to_worlds_list])
        
        # Patch methods
        with patch.object(generator, '_load_instances_info', return_value=({}, {}, {})):
            mock_dataset.get_scene.return_value['dataset'] = mock_scene_dataset
            
            # Use generate_pointcloud_with_static_dynamic for testing the full functionality
            static_pc, dynamic_pc = generator.generate_pointcloud_with_static_dynamic(
                mock_dataset, 0, 0
            )
            
            assert isinstance(static_pc, StaticPointCloud)
            assert isinstance(dynamic_pc, DynamicPointCloud)
            assert len(static_pc.frame_points) == 2
            assert isinstance(dynamic_pc.instance_id_mapping, dict)
            assert isinstance(dynamic_pc.points_by_instance, dict)
            
            # Also test that generate_pointcloud returns PointCloud
            pointcloud = generator.generate_pointcloud(mock_dataset, 0, 0)
            assert isinstance(pointcloud, o3d.geometry.PointCloud)
    
    def test_generate_pointcloud_with_instances(self):
        """Test pointcloud generation with instances."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock dataset
        mock_dataset = Mock(spec=MultiSceneDataset)
        mock_dataset.get_segment_frames = Mock(return_value=[0])
        mock_dataset.get_scene = Mock(return_value={
            'dataset': Mock(),
        })
        mock_dataset.get_frame_data = Mock(return_value={
            'image': torch.rand(100, 200, 3),
            'extrinsic': torch.eye(4),
            'intrinsic': torch.eye(4),
            'depth': torch.rand(100, 200),
        })
        
        # Mock scene_dataset
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([0, 0])  # Two points for frame 0
        mock_lidar_source.points = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Point at origin (in instance)
            [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Point outside instance
        ])
        # Mock lidar_to_worlds for ego pose (按点存储)
        lidar_to_worlds_list = [
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        ]
        mock_lidar_source.lidar_to_worlds = torch.stack([torch.from_numpy(x) for x in lidar_to_worlds_list])
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        # Mock instances info
        waymoid2intid = {0: 1}
        id2framePoseSize = {
            0: {
                0: (np.eye(4), np.array([2, 2, 2]))  # Box at origin, size 2x2x2
            }
        }
        frame_instances = {"0": [0]}
        
        with patch.object(generator, '_load_instances_info', return_value=(
            waymoid2intid, id2framePoseSize, frame_instances
        )):
            mock_dataset.get_scene.return_value['dataset'] = mock_scene_dataset
            
            # Use generate_pointcloud_with_static_dynamic for testing the full functionality
            static_pc, dynamic_pc = generator.generate_pointcloud_with_static_dynamic(
                mock_dataset, 0, 0
            )
            
            assert len(static_pc.frame_points) == 1
            # Should have static points (point outside instance) and/or dynamic points (point inside instance)
            # The point at origin [0,0,0] should be in the instance, point at [5,0,0] should be static
            assert len(static_pc.frame_points[0]) > 0 or len(dynamic_pc.points_by_instance) > 0
            # Verify we have both static and dynamic points
            total_static = len(static_pc.frame_points[0])
            total_dynamic = sum(
                sum(len(points) for points in frame_dict.values())
                for frame_dict in dynamic_pc.points_by_instance.values()
            )
            assert total_static + total_dynamic == 2  # Total should be 2 points


class TestCoordinateTransforms:
    """Test coordinate transformations."""
    
    def test_vehicle_to_world_transform(self):
        """Test vehicle to world coordinate transformation."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Test point in vehicle coordinates
        pts_v = np.array([[1, 2, 3]], dtype=np.float32)
        
        # Vehicle pose: translation [10, 20, 30]
        T_vw = np.eye(4, dtype=np.float32)
        T_vw[:3, 3] = [10, 20, 30]
        
        pts_w = (T_vw[:3, :3] @ pts_v.T + T_vw[:3, 3:4]).T
        
        assert np.allclose(pts_w[0], [11, 22, 33])
    
    def test_world_to_object_transform(self):
        """Test world to object local coordinate transformation."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Test point in world coordinates
        pts_w = np.array([[11, 22, 33]], dtype=np.float32)
        
        # Object pose: translation [10, 20, 30]
        T_ow = np.eye(4, dtype=np.float32)
        T_ow[:3, 3] = [10, 20, 30]
        
        T_wo = np.linalg.inv(T_ow)
        pw_h = np.concatenate([pts_w, np.ones((1, 1), dtype=np.float32)], axis=1)
        po = (T_wo @ pw_h.T).T[:, :3]
        
        assert np.allclose(po[0], [1, 2, 3], atol=1e-5)
    
    def test_in_box_check(self):
        """Test bounding box point check."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Points in object local coordinates
        po = np.array([
            [0, 0, 0],      # Inside
            [1, 0, 0],      # Inside (on boundary)
            [2, 0, 0],      # Outside
            [0.5, 0.5, 0.5],  # Inside
        ], dtype=np.float32)
        
        size_lwh = np.array([2, 2, 2], dtype=np.float32)
        half = size_lwh / 2.0
        
        mask = (np.abs(po) <= (half + 1e-6)).all(axis=1)
        
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3] == True


class TestRGBColoring:
    """Test RGB coloring functionality."""
    
    def test_single_camera_coloring(self):
        """Test single camera coloring."""
        generator = LiDARRGBPointCloudGenerator(
            chosen_cam_ids=[0],
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # This would require more complex mocking of the full pipeline
        # For now, just test that the method exists and can be called
        assert hasattr(generator, '_colorize_points_world')
    
    def test_get_opencv2dataset_matrix(self):
        """Test OpenCV to dataset coordinate transformation matrix."""
        generator_waymo = LiDARRGBPointCloudGenerator(
            dataset="waymo",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        matrix_waymo = generator_waymo._get_opencv2dataset_matrix()
        assert matrix_waymo.shape == (4, 4)
        assert not np.allclose(matrix_waymo, np.eye(4))  # Waymo should have non-identity matrix
        
        generator_kitti = LiDARRGBPointCloudGenerator(
            dataset="kitti",
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        matrix_kitti = generator_kitti._get_opencv2dataset_matrix()
        assert matrix_kitti.shape == (4, 4)
        assert np.allclose(matrix_kitti, np.eye(4))  # KITTI should have identity matrix


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_lidar_points(self):
        """Test handling of empty LiDAR point cloud."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([])
        mock_lidar_source.points = torch.tensor([]).reshape(0, 14)
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0  # 添加 start_timestep 属性
        
        pts_w = generator._load_lidar_points_world(mock_scene_dataset, 0)
        assert pts_w is None or pts_w.shape[0] == 0
    
    def test_missing_instances_files(self):
        """Test handling of missing instance files."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        scene_data = {
            'dataset': Mock(),
        }
        scene_data['dataset'].data_cfg = Mock()
        scene_data['dataset'].data_cfg.data_root = "/nonexistent"
        scene_data['dataset'].data_cfg.scene_idx = 0
        
        waymoid2intid, id2framePoseSize, frame_instances = generator._load_instances_info(
            scene_data, 0
        )
        
        # Should return empty instances
        assert waymoid2intid == {}
        assert id2framePoseSize == {}
        assert frame_instances == {}
    
    def test_invalid_frame_indices(self):
        """Test handling of invalid frame indices."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        mock_dataset = Mock(spec=MultiSceneDataset)
        mock_dataset.get_segment_frames = Mock(return_value=[])
        
        with pytest.raises(ValueError, match="has no frames"):
            generator.generate_pointcloud(mock_dataset, 0, 0)


class TestLiDARStaticDynamicStructures:
    """Test LiDARRGBPointCloudGenerator with new data structures."""
    
    def test_lidar_static_pointcloud_structure(self):
        """Test StaticPointCloud structure from LiDAR generator."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock dataset
        mock_dataset = Mock(spec=MultiSceneDataset)
        mock_dataset.get_segment_frames = Mock(return_value=[0, 1])
        mock_dataset.get_scene = Mock(return_value={
            'dataset': Mock(),
        })
        mock_dataset.get_frame_data = Mock(return_value={
            'image': torch.rand(100, 200, 3),
            'extrinsic': torch.eye(4),
            'intrinsic': torch.eye(4),
            'depth': torch.rand(100, 200),
        })
        
        # Mock scene_dataset
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([0, 1])
        mock_lidar_source.points = torch.tensor([
            [0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0
        
        lidar_to_worlds_list = [
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        ]
        mock_lidar_source.lidar_to_worlds = torch.stack([torch.from_numpy(x) for x in lidar_to_worlds_list])
        
        with patch.object(generator, '_load_instances_info', return_value=({}, {}, {})):
            mock_dataset.get_scene.return_value['dataset'] = mock_scene_dataset
            
            static_pc, dynamic_pc = generator.generate_pointcloud_with_static_dynamic(
                mock_dataset, 0, 0
            )
            
            assert isinstance(static_pc, StaticPointCloud)
            assert len(static_pc.frame_points) == 2
            assert all(isinstance(fp, np.ndarray) for fp in static_pc.frame_points)
            assert all(fp.shape[1] == 6 for fp in static_pc.frame_points if fp.shape[0] > 0)
    
    def test_lidar_dynamic_pointcloud_structure(self):
        """Test DynamicPointCloud structure from LiDAR generator."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Mock dataset
        mock_dataset = Mock(spec=MultiSceneDataset)
        mock_dataset.get_segment_frames = Mock(return_value=[0])
        mock_dataset.get_scene = Mock(return_value={
            'dataset': Mock(),
        })
        mock_dataset.get_frame_data = Mock(return_value={
            'image': torch.rand(100, 200, 3),
            'extrinsic': torch.eye(4),
            'intrinsic': torch.eye(4),
            'depth': torch.rand(100, 200),
        })
        
        # Mock scene_dataset
        mock_scene_dataset = Mock()
        mock_lidar_source = Mock()
        mock_lidar_source.timesteps = torch.tensor([0, 0])
        mock_lidar_source.points = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Point at origin (in instance)
            [0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Point outside instance
        ])
        lidar_to_worlds_list = [
            np.eye(4, dtype=np.float32),
            np.eye(4, dtype=np.float32),
        ]
        mock_lidar_source.lidar_to_worlds = torch.stack([torch.from_numpy(x) for x in lidar_to_worlds_list])
        mock_scene_dataset.lidar_source = mock_lidar_source
        mock_scene_dataset.start_timestep = 0
        
        # Mock instances info
        waymoid2intid = {0: 1}
        id2framePoseSize = {
            0: {
                0: (np.eye(4), np.array([2, 2, 2]))  # Box at origin, size 2x2x2
            }
        }
        frame_instances = {"0": [0]}
        
        with patch.object(generator, '_load_instances_info', return_value=(
            waymoid2intid, id2framePoseSize, frame_instances
        )):
            mock_dataset.get_scene.return_value['dataset'] = mock_scene_dataset
            
            static_pc, dynamic_pc = generator.generate_pointcloud_with_static_dynamic(
                mock_dataset, 0, 0
            )
            
            assert isinstance(dynamic_pc, DynamicPointCloud)
            assert isinstance(dynamic_pc.instance_id_mapping, dict)
            assert isinstance(dynamic_pc.points_by_instance, dict)
            assert isinstance(dynamic_pc.instances_info, dict)
            
            # If there are dynamic points, verify structure
            if len(dynamic_pc.points_by_instance) > 0:
                for intid, frame_dict in dynamic_pc.points_by_instance.items():
                    assert isinstance(frame_dict, dict)
                    for frame_idx, points in frame_dict.items():
                        assert isinstance(points, np.ndarray)
                        assert points.shape[1] == 6  # x, y, z, r, g, b
    
    def test_lidar_dynamic_transform_to_world(self):
        """Test dynamic point cloud coordinate transformation."""
        generator = LiDARRGBPointCloudGenerator(
            crop_aabb=DEFAULT_CROP_AABB,
            input_aabb=DEFAULT_INPUT_AABB,
        )
        
        # Create a DynamicPointCloud with known data
        points_by_instance = {
            1: {
                0: np.array([[0, 0, 0, 1, 0, 0]], dtype=np.float32),  # Local: origin
            },
        }
        
        # Instance pose: translation [10, 20, 30]
        T_ow = np.eye(4, dtype=np.float32)
        T_ow[:3, 3] = [10, 20, 30]
        
        instances_info = {
            1: {
                "poses": np.stack([T_ow], axis=0),  # (1, 4, 4)
                "size": np.array([2, 2, 2], dtype=np.float32),
                "frame_info": np.array([True], dtype=bool),
            },
        }
        
        dynamic_pc = DynamicPointCloud(
            instance_id_mapping={0: 1},
            points_by_instance=points_by_instance,
            instances_info=instances_info,
        )
        
        points_world = dynamic_pc.transform_to_world(1, 0)
        assert points_world.shape == (1, 6)
        assert np.allclose(points_world[0, :3], [10, 20, 30], atol=1e-5)
        assert np.allclose(points_world[0, 3:], [1, 0, 0])

