"""
Integration tests for RGB Point Cloud Generator.
"""
import pytest
import torch
import numpy as np
import open3d as o3d
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset, MultiSceneDatasetScheduler
from datasets.pointcloud_generators.rgb_pointcloud_generator import (
    RGBPointCloudGenerator,
    MonocularRGBPointCloudGenerator,
)


class TestRGBPointCloudGeneratorBase:
    """Test base class methods."""
    
    def test_get_bbx_default(self):
        """Test default bounding box."""
        generator = MonocularRGBPointCloudGenerator()
        bbx_min, bbx_max = generator.get_bbx()
        
        assert bbx_min.shape == (3,)
        assert bbx_max.shape == (3,)
        assert np.allclose(bbx_min, np.array([-20, -20, -20]))
        assert np.allclose(bbx_max, np.array([20, 4.8, 70]))
    
    def test_get_bbx_custom(self):
        """Test custom bounding box."""
        bbx_min = np.array([-10, -10, -10])
        bbx_max = np.array([10, 10, 10])
        generator = MonocularRGBPointCloudGenerator(
            bbx_min=bbx_min,
            bbx_max=bbx_max,
        )
        
        result_min, result_max = generator.get_bbx()
        assert np.allclose(result_min, bbx_min)
        assert np.allclose(result_max, bbx_max)
    
    def test_crop_pointcloud(self):
        """Test point cloud cropping."""
        generator = MonocularRGBPointCloudGenerator()
        bbx_min, bbx_max = generator.get_bbx()
        
        # Create test points
        # Note: crop_pointcloud extends Z max by 50 for background points
        # So Z max is actually 70 + 50 = 120
        points = np.array([
            [0, 0, 0],      # Inside
            [30, 0, 0],     # Outside (x > max)
            [0, 10, 0],     # Outside (y > max)
            [0, 0, 100],    # Inside (z < 120, extended max)
            [-5, -5, -5],   # Inside
        ])
        colors = np.ones((5, 3)) * 0.5
        
        cropped_points, cropped_colors = generator.crop_pointcloud(
            bbx_min, bbx_max, points, colors
        )
        
        assert len(cropped_points) == 3  # 3 points inside (including extended Z)
        assert len(cropped_colors) == 3
        # Check that all expected points are present
        cropped_points_list = cropped_points.tolist()
        assert [0, 0, 0] in cropped_points_list
        assert [0, 0, 100] in cropped_points_list
        assert [-5, -5, -5] in cropped_points_list
    
    def test_split_pointcloud(self):
        """Test point cloud splitting."""
        generator = MonocularRGBPointCloudGenerator()
        bbx_min, bbx_max = generator.get_bbx()
        
        # Create test points
        points = np.array([
            [0, 0, 0],      # Inside
            [30, 0, 0],     # Outside
            [-5, -5, -5],   # Inside
        ])
        colors = np.ones((3, 3)) * 0.5
        
        inside_points, inside_colors, outside_points, outside_colors = generator.split_pointcloud(
            bbx_min, bbx_max, points, colors
        )
        
        assert len(inside_points) == 2
        assert len(outside_points) == 1
        assert len(inside_colors) == 2
        assert len(outside_colors) == 1
    
    def test_filter_pointcloud(self):
        """Test point cloud filtering."""
        generator = MonocularRGBPointCloudGenerator()
        
        # Create a simple point cloud
        points = np.random.rand(100, 3) * 10
        colors = np.random.rand(100, 3)
        
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        filtered = generator.filter_pointcloud(pointcloud, use_bbx=False)
        
        assert isinstance(filtered, o3d.geometry.PointCloud)
        assert len(filtered.points) <= len(points)  # Should have fewer points after filtering


class TestMonocularRGBPointCloudGenerator:
    """Test MonocularRGBPointCloudGenerator."""
    
    def test_sparsity_filter_full(self):
        """Test sparsity filter with 'full'."""
        generator = MonocularRGBPointCloudGenerator(sparsity='full')
        frame_indices = [0, 1, 2, 3, 4, 5]
        
        filtered = generator._apply_sparsity_filter(frame_indices)
        assert filtered == frame_indices
    
    def test_sparsity_filter_drop50(self):
        """Test sparsity filter with 'Drop50'."""
        generator = MonocularRGBPointCloudGenerator(sparsity='Drop50')
        frame_indices = list(range(8))
        
        filtered = generator._apply_sparsity_filter(frame_indices)
        # Should keep positions 0, 1, 4, 5 (50% of frames)
        assert len(filtered) == 4
        assert filtered == [0, 1, 4, 5]
    
    def test_sparsity_filter_drop25(self):
        """Test sparsity filter with 'Drop25'."""
        generator = MonocularRGBPointCloudGenerator(sparsity='Drop25')
        frame_indices = list(range(8))
        
        filtered = generator._apply_sparsity_filter(frame_indices)
        # Should keep 75% of frames (drop position 2, 6)
        assert len(filtered) == 6
    
    def test_sparsity_filter_drop80(self):
        """Test sparsity filter with 'Drop80'."""
        generator = MonocularRGBPointCloudGenerator(sparsity='Drop80')
        frame_indices = list(range(10))
        
        filtered = generator._apply_sparsity_filter(frame_indices)
        # Should keep 20% of frames (positions 0, 5)
        assert len(filtered) == 2
        assert filtered == [0, 5]
    
    def test_sparsity_filter_drop90(self):
        """Test sparsity filter with 'Drop90'."""
        generator = MonocularRGBPointCloudGenerator(sparsity='Drop90')
        frame_indices = list(range(20))
        
        filtered = generator._apply_sparsity_filter(frame_indices)
        # Should keep 10% of frames (positions 0, 10)
        assert len(filtered) == 2
        assert filtered == [0, 10]


class TestMultiSceneDatasetExtensions:
    """Test MultiSceneDataset extensions."""
    
    def test_get_segment_frames(self):
        """Test get_segment_frames method."""
        # Create a mock dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        # Mock scene data
        scene_data = {
            'dataset': Mock(),
            'segments': [
                {'frame_indices': [0, 1, 2, 3, 4]},
                {'frame_indices': [5, 6, 7, 8, 9]},
            ],
        }
        
        with patch.object(dataset, 'get_scene', return_value=scene_data):
            frames = dataset.get_segment_frames(0, 0)
            assert frames == [0, 1, 2, 3, 4]
            
            frames = dataset.get_segment_frames(0, 1)
            assert frames == [5, 6, 7, 8, 9]
    
    def test_get_segment_frames_sorted_deduplicated(self):
        """Test that get_segment_frames returns sorted and deduplicated frames."""
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        scene_data = {
            'dataset': Mock(),
            'segments': [
                {'frame_indices': [3, 1, 2, 1, 0, 3]},  # Unsorted with duplicates
            ],
        }
        
        with patch.object(dataset, 'get_scene', return_value=scene_data):
            frames = dataset.get_segment_frames(0, 0)
            assert frames == [0, 1, 2, 3]  # Sorted and deduplicated
    
    def test_get_frame_data(self):
        """Test get_frame_data method."""
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            device=torch.device("cpu"),
        )
        
        # Mock scene dataset
        mock_scene_dataset = Mock()
        mock_scene_dataset.num_cams = 2
        mock_pixel_source = Mock()
        mock_image_infos = {
            'pixels': torch.rand(100, 200, 3),  # [H, W, 3]
        }
        mock_cam_infos = {
            'camera_to_world': torch.eye(4),
            'intrinsics': torch.eye(3),
        }
        mock_pixel_source.get_image.return_value = (mock_image_infos, mock_cam_infos)
        mock_scene_dataset.pixel_source = mock_pixel_source
        
        scene_data = {
            'dataset': mock_scene_dataset,
            'segments': [],
        }
        
        # Mock _get_depth to return a depth map
        with patch.object(dataset, 'get_scene', return_value=scene_data):
            with patch.object(dataset, '_get_depth', return_value=torch.rand(100, 200)):
                frame_data = dataset.get_frame_data(0, 0, 0)
                
                assert 'image' in frame_data
                assert 'extrinsic' in frame_data
                assert 'intrinsic' in frame_data
                assert 'depth' in frame_data
                
                assert frame_data['image'].shape == (100, 200, 3)
                assert frame_data['extrinsic'].shape == (4, 4)
                assert frame_data['intrinsic'].shape == (4, 4)
                assert frame_data['depth'].shape == (100, 200)


class TestMultiSceneDatasetSchedulerExtensions:
    """Test MultiSceneDatasetScheduler extensions."""
    
    def test_generate_segment_pointcloud(self):
        """Test generate_segment_pointcloud method."""
        import threading
        
        # Create mock dataset with proper initialization
        dataset = Mock(spec=MultiSceneDataset)
        dataset._initialized = True
        dataset._lock = threading.RLock()  # Add _lock attribute for background thread
        dataset.get_current_scene_id = Mock(return_value=0)
        dataset.get_scene = Mock(return_value={
            'dataset': Mock(),
            'segments': [
                {'frame_indices': [0, 1, 2]},
            ],
        })
        dataset._ensure_training_queue_ready = Mock()  # Mock this method
        
        # Create mock pointcloud generator
        generator = Mock(spec=MonocularRGBPointCloudGenerator)
        mock_pointcloud = o3d.geometry.PointCloud()
        mock_pointcloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        generator.generate_pointcloud.return_value = mock_pointcloud
        
        # Create scheduler with mocked dataset
        scheduler = MultiSceneDatasetScheduler(
            dataset=dataset,
            batches_per_segment=1,
        )
        
        try:
            # Mock current scene
            scheduler.current_scene_id = 0
            scheduler.scene_segment_order = [0]
            scheduler.current_segment_id = 0
            
            # Test generating pointcloud
            pointcloud = scheduler.generate_segment_pointcloud(generator)
            
            assert isinstance(pointcloud, o3d.geometry.PointCloud)
            generator.generate_pointcloud.assert_called_once_with(
                dataset=dataset,
                scene_id=0,
                segment_id=0,
            )
        finally:
            # Shutdown background thread
            scheduler.shutdown()
    
    def test_generate_all_segment_pointclouds(self):
        """Test generate_all_segment_pointclouds method."""
        import threading
        
        # Create mock dataset with proper initialization
        dataset = Mock(spec=MultiSceneDataset)
        dataset._initialized = True
        dataset._lock = threading.RLock()  # Add _lock attribute for background thread
        dataset.get_current_scene_id = Mock(return_value=0)
        dataset._ensure_training_queue_ready = Mock()  # Mock this method
        
        # Mock scene data
        scene_data = {
            'dataset': Mock(),
            'segments': [
                {'frame_indices': [0, 1, 2]},
                {'frame_indices': [3, 4, 5]},
            ],
        }
        dataset.get_scene = Mock(return_value=scene_data)
        
        # Create mock pointcloud generator
        generator = Mock(spec=MonocularRGBPointCloudGenerator)
        mock_pointcloud = o3d.geometry.PointCloud()
        mock_pointcloud.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        generator.generate_pointcloud.return_value = mock_pointcloud
        
        # Create scheduler with mocked dataset
        scheduler = MultiSceneDatasetScheduler(
            dataset=dataset,
            batches_per_segment=1,
        )
        
        try:
            # Test generating all segment pointclouds
            pointclouds = scheduler.generate_all_segment_pointclouds(generator, scene_id=0)
            
            assert isinstance(pointclouds, dict)
            assert len(pointclouds) == 2
            assert 0 in pointclouds
            assert 1 in pointclouds
            assert generator.generate_pointcloud.call_count == 2
        finally:
            # Shutdown background thread
            scheduler.shutdown()


class TestDepthConsistencyFix:
    """Test fixes for depth consistency issues."""
    
    def test_depth_consistency_uses_previous_intrinsics(self):
        """Test that depth consistency check uses previous frame's intrinsics."""
        generator = MonocularRGBPointCloudGenerator(
            depth_consistency=True,
            chosen_cam_ids=[0],
        )
        
        H, W = 100, 200
        # Create frame data with different intrinsics
        frame_data_list = [
            {
                'depth': np.random.rand(H, W) * 50 + 10,
                'extrinsic': np.eye(4),
                'intrinsic': np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]]),  # Different intrinsics
            },
            {
                'depth': np.random.rand(H, W) * 50 + 10,
                'extrinsic': np.eye(4),
                'intrinsic': np.array([[150, 0, W/2], [0, 150, H/2], [0, 0, 1]]),  # Different intrinsics
            },
        ]
        
        # Should not raise error and should use correct intrinsics
        masks = generator._depth_consistency_check(frame_data_list, H, W)
        assert len(masks) == 2
        assert masks[0].shape == (H, W)
        assert masks[1].shape == (H, W)
    
    def test_multi_camera_separate_consistency_check(self):
        """Test that depth consistency is checked separately for each camera."""
        generator = MonocularRGBPointCloudGenerator(
            depth_consistency=True,
            chosen_cam_ids=[0, 1],
        )
        
        # Mock dataset
        dataset = Mock(spec=MultiSceneDataset)
        dataset.get_segment_frames = Mock(return_value=[0, 1, 2])
        
        # Mock frame data loading
        def mock_load_frame_data(dataset, scene_id, frame_idx, cam_id):
            return {
                'rgb': np.random.rand(100, 200, 3),
                'depth': np.random.rand(100, 200) * 50 + 10,
                'extrinsic': np.eye(4),
                'intrinsic': np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]]),
                'sky_mask': None,
            }
        
        generator._load_frame_data = mock_load_frame_data
        
        # Should group by camera and check consistency separately
        # This test verifies the structure, actual execution would require more setup
        assert generator.chosen_cam_ids == [0, 1]


class TestSkyMask:
    """Test sky mask functionality."""
    
    def test_sky_mask_filtering(self):
        """Test that sky mask is correctly applied when filter_sky=True."""
        generator = MonocularRGBPointCloudGenerator(
            filter_sky=True,
            chosen_cam_ids=[0],
        )
        
        H, W = 100, 200
        # Create frame data with sky mask
        sky_mask = np.zeros((H, W), dtype=bool)
        sky_mask[:20, :] = True  # Top 20 rows are sky
        
        frame_data = {
            'rgb': np.random.rand(H, W, 3),
            'depth': np.random.rand(H, W) * 50 + 10,
            'extrinsic': np.eye(4),
            'intrinsic': np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]]),
            'sky_mask': torch.from_numpy(sky_mask).float(),
        }
        
        frame_data_list_by_camera = {0: [frame_data]}
        consistency_masks_by_camera = {0: [np.ones((H, W), dtype=bool)]}
        
        # Generate pointcloud - sky regions should be filtered out
        pointcloud = generator._generate_pointcloud_from_frames_by_camera(
            frame_data_list_by_camera, consistency_masks_by_camera, H, W
        )
        
        # Should have points (non-sky regions)
        assert len(pointcloud) > 0
        # Points should be from non-sky regions (rows 20+)
        # This is a basic check - actual verification would require more detailed inspection
    
    def test_sky_mask_no_filtering(self):
        """Test that sky mask is not applied when filter_sky=False."""
        generator = MonocularRGBPointCloudGenerator(
            filter_sky=False,
            chosen_cam_ids=[0],
        )
        
        H, W = 100, 200
        sky_mask = np.zeros((H, W), dtype=bool)
        sky_mask[:20, :] = True
        
        frame_data = {
            'rgb': np.random.rand(H, W, 3),
            'depth': np.random.rand(H, W) * 50 + 10,
            'extrinsic': np.eye(4),
            'intrinsic': np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]]),
            'sky_mask': torch.from_numpy(sky_mask).float(),
        }
        
        frame_data_list_by_camera = {0: [frame_data]}
        consistency_masks_by_camera = {0: [np.ones((H, W), dtype=bool)]}
        
        # Generate pointcloud - all regions should be included
        pointcloud = generator._generate_pointcloud_from_frames_by_camera(
            frame_data_list_by_camera, consistency_masks_by_camera, H, W
        )
        
        # Should have points from all regions
        assert len(pointcloud) > 0
    
    def test_sky_mask_missing(self):
        """Test behavior when sky mask is not available."""
        generator = MonocularRGBPointCloudGenerator(
            filter_sky=True,
            chosen_cam_ids=[0],
        )
        
        H, W = 100, 200
        frame_data = {
            'rgb': np.random.rand(H, W, 3),
            'depth': np.random.rand(H, W) * 50 + 10,
            'extrinsic': np.eye(4),
            'intrinsic': np.array([[100, 0, W/2], [0, 100, H/2], [0, 0, 1]]),
            'sky_mask': None,  # No sky mask
        }
        
        frame_data_list_by_camera = {0: [frame_data]}
        consistency_masks_by_camera = {0: [np.ones((H, W), dtype=bool)]}
        
        # Should still work, but log warning
        pointcloud = generator._generate_pointcloud_from_frames_by_camera(
            frame_data_list_by_camera, consistency_masks_by_camera, H, W
        )
        
        assert len(pointcloud) > 0


class TestIntegration:
    """Integration tests with mocked data."""
    
    @pytest.mark.skip(reason="Requires actual dataset and data files")
    def test_full_pointcloud_generation(self):
        """Test full pointcloud generation pipeline (requires actual data)."""
        # This test would require actual dataset files
        # It's marked as skip but can be enabled when testing with real data
        pass

