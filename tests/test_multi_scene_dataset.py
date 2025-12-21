"""
Unit and integration tests for MultiSceneDataset.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from datasets.driving_dataset import DrivingDataset


class TestSceneSuitability:
    """Test scene suitability checking."""
    
    def test_sufficient_keyframes(self):
        """Test scene with sufficient keyframes."""
        keyframe_segments = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], 
                            [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_scene=10,
        )
        
        assert dataset._is_scene_suitable(keyframe_segments) == True
    
    def test_insufficient_keyframes(self):
        """Test scene with insufficient keyframes."""
        keyframe_segments = [[0, 1], [2, 3], [4, 5]]
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_scene=10,
        )
        
        assert dataset._is_scene_suitable(keyframe_segments) == False


class TestSegmentSplitting:
    """Test segment splitting logic."""
    
    def test_single_segment_short_distance(self):
        """Test single segment case (short movement distance)."""
        # Mock scene dataset
        scene_dataset = Mock()
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        
        # Create keyframe segments with short total distance
        keyframe_segments = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        keyframe_ranges = torch.tensor([
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ])  # Total distance = 5.0, AABB length = 100.0
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_segment=3,
        )
        
        segments = dataset._split_segments(
            scene_dataset=scene_dataset,
            keyframe_segments=keyframe_segments,
            keyframe_ranges=keyframe_ranges,
            overlap_ratio=0.2,
        )
        
        # Should create single segment
        assert len(segments) == 1
        assert len(segments[0]['keyframe_indices']) == 5
    
    def test_multiple_segments_long_distance(self):
        """Test multiple segments case (long movement distance)."""
        # Mock scene dataset
        scene_dataset = Mock()
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0]
        ])
        
        # Create keyframe segments with long total distance
        keyframe_segments = [[i] for i in range(20)]  # 20 keyframes
        keyframe_ranges = torch.tensor([
            [i * 2.0, (i + 1) * 2.0] for i in range(20)
        ])  # Total distance = 40.0, AABB length = 50.0
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_segment=6,
        )
        
        segments = dataset._split_segments(
            scene_dataset=scene_dataset,
            keyframe_segments=keyframe_segments,
            keyframe_ranges=keyframe_ranges,
            overlap_ratio=0.2,
        )
        
        # Should create multiple segments
        assert len(segments) > 1
        # Each segment should have at least min_keyframes_per_segment keyframes
        for seg in segments:
            assert len(seg['keyframe_indices']) >= 6


class TestKeyframeSelection:
    """Test keyframe selection logic."""
    
    def test_source_keyframe_selection(self):
        """Test source keyframe selection."""
        segment = {
            'keyframe_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            num_source_keyframes=3,
            num_target_keyframes=6,
        )
        
        source_kf_indices, target_kf_indices = dataset._select_source_and_target_keyframes(
            segment=segment,
            num_source_keyframes=3,
            num_target_keyframes=6,
        )
        
        assert len(source_kf_indices) == 3
        assert len(target_kf_indices) == 6
        # Target should include all source keyframes
        assert all(kf in target_kf_indices for kf in source_kf_indices)
    
    def test_insufficient_keyframes(self):
        """Test keyframe selection with insufficient available keyframes."""
        segment = {
            'keyframe_indices': [0, 1]  # Only 2 keyframes available
        }
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            num_source_keyframes=3,
            num_target_keyframes=6,
        )
        
        source_kf_indices, target_kf_indices = dataset._select_source_and_target_keyframes(
            segment=segment,
            num_source_keyframes=3,
            num_target_keyframes=6,
        )
        
        # Should repeat keyframes to meet requirements
        assert len(source_kf_indices) == 3
        assert len(target_kf_indices) == 6


class TestIntrinsicConversion:
    """Test intrinsic matrix conversion."""
    
    def test_3x3_to_4x4(self):
        """Test conversion from 3x3 to 4x4."""
        intrinsic_3x3 = torch.tensor([
            [100.0, 0.0, 320.0],
            [0.0, 100.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        intrinsic_4x4 = dataset._convert_intrinsic_to_4x4(intrinsic_3x3)
        
        assert intrinsic_4x4.shape == (4, 4)
        assert torch.allclose(intrinsic_4x4[:3, :3], intrinsic_3x3)
        assert intrinsic_4x4[3, 3] == 1.0
        assert torch.all(intrinsic_4x4[3, :3] == 0.0)
        assert torch.all(intrinsic_4x4[:3, 3] == 0.0)
    
    def test_already_4x4(self):
        """Test when intrinsic is already 4x4."""
        intrinsic_4x4 = torch.eye(4)
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        result = dataset._convert_intrinsic_to_4x4(intrinsic_4x4)
        
        assert torch.allclose(result, intrinsic_4x4)


class TestFrameSelection:
    """Test frame selection from keyframe."""
    
    def test_select_frame_from_keyframe(self):
        """Test selecting frame from keyframe segment."""
        keyframe_segment = [10, 11, 12, 13, 14]
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        frame_idx = dataset._select_frame_from_keyframe(keyframe_segment)
        
        assert frame_idx in keyframe_segment
    
    def test_empty_keyframe_segment(self):
        """Test error handling for empty keyframe segment."""
        keyframe_segment = []
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        with pytest.raises(ValueError, match="Keyframe segment is empty"):
            dataset._select_frame_from_keyframe(keyframe_segment)


class TestIntegration:
    """Integration tests with mocked DrivingDataset."""
    
    @pytest.fixture
    def mock_driving_dataset(self):
        """Create a mock DrivingDataset."""
        dataset = Mock(spec=DrivingDataset)
        
        # Mock pixel_source
        pixel_source = Mock()
        pixel_source.num_cams = 6
        pixel_source.num_imgs = 100
        pixel_source.camera_list = [0, 1, 2, 3, 4, 5]
        
        # Mock camera_data
        camera_data = {}
        for cam_id in range(6):
            cam_mock = Mock()
            cam_mock.unique_cam_idx = cam_id
            cam_mock.lidar_depth_maps = None
            # Create a dict for depth_maps that can be indexed by frame_idx
            # For testing, we'll create placeholder depth maps for frames 0-15
            cam_mock.depth_maps = {i: torch.ones(100, 200, dtype=torch.float32) * 10.0 for i in range(16)}
            camera_data[cam_id] = cam_mock
        
        pixel_source.camera_data = camera_data
        
        # Mock get_image method
        def mock_get_image(img_idx):
            cam_idx = img_idx % 6
            frame_idx = img_idx // 6
            
            image_infos = {
                'pixels': torch.rand(100, 200, 3),
            }
            cam_infos = {
                'camera_to_world': torch.eye(4),
                'intrinsics': torch.eye(3),
                'height': torch.tensor(100),
                'width': torch.tensor(200),
            }
            return image_infos, cam_infos
        
        pixel_source.get_image = mock_get_image
        pixel_source.parse_img_idx = lambda idx: (idx % 6, idx // 6)
        
        dataset.pixel_source = pixel_source
        dataset.num_img_timesteps = 16  # 100 images / 6 cameras ≈ 16 frames
        dataset.num_cams = 6
        dataset.data_path = "/mock/path"
        
        # Mock get_aabb
        dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        
        # Mock get_novel_render_traj
        def mock_get_traj(traj_types, target_frames):
            # Create a simple linear trajectory
            trajectory = torch.eye(4).unsqueeze(0).repeat(target_frames, 1, 1)
            trajectory[:, 0, 3] = torch.arange(target_frames).float() * 2.0
            return {"front_center_interp": trajectory}
        
        dataset.get_novel_render_traj = mock_get_traj
        
        return dataset
    
    @pytest.fixture
    def mock_data_cfg(self):
        """Create mock data configuration."""
        return OmegaConf.create({
            'dataset': 'nuscenes',
            'data_root': '/mock/data',
            'start_timestep': 0,
            'end_timestep': -1,
            'pixel_source': {
                'type': 'datasets.nuscenes.nuscenes_sourceloader.NuScenesPixelSource',
                'cameras': [0, 1, 2, 3, 4, 5],
                'downscale_when_loading': [1],
                'downscale': 1,
                'undistort': False,
                'test_image_stride': 0,
                'load_sky_mask': False,
                'load_dynamic_mask': False,
                'load_objects': False,
                'load_smpl': False,
            },
            'lidar_source': {
                'type': 'datasets.nuscenes.nuscenes_sourceloader.NuScenesLiDARSource',
                'load_lidar': False,
            },
        })
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_load_scene_success(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test successful scene loading."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        scene_data = dataset._load_scene(0)
        
        assert scene_data is not None
        assert 'dataset' in scene_data
        assert 'keyframe_segments' in scene_data
        assert 'segments' in scene_data
        assert len(scene_data['segments']) > 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_load_scene_insufficient_keyframes(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scene loading with insufficient keyframes."""
        # Modify trajectory to have very few frames
        def mock_get_traj_short(traj_types, target_frames):
            trajectory = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
            return {"front_center_interp": trajectory}
        
        mock_driving_dataset.get_novel_render_traj = mock_get_traj_short
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=10,  # Require 10 keyframes
            min_keyframes_per_segment=3,
        )
        
        scene_data = dataset._load_scene(0)
        
        # Should return None due to insufficient keyframes
        assert scene_data is None
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_get_segment_batch_format(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test batch format from get_segment_batch."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            num_source_keyframes=3,
            num_target_keyframes=6,
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        # Load scene first
        scene_data = dataset._load_scene(0)
        if scene_data is None:
            pytest.skip("Scene loading failed, cannot test batch format")
        
        dataset.train_scenes[0] = scene_data
        
        # Get batch
        batch = dataset.get_segment_batch(0, 0)
        
        # Check batch structure
        assert 'scene_id' in batch
        assert 'segment_id' in batch
        assert 'source' in batch
        assert 'target' in batch
        
        # Check source format
        assert batch['source']['image'].shape[0] == 3 * 6  # 3 frames × 6 cameras
        assert batch['source']['extrinsics'].shape[0] == 3 * 6
        assert batch['source']['intrinsics'].shape[0] == 3 * 6
        assert batch['source']['depth'].shape[0] == 3 * 6
        
        # Check target format
        assert batch['target']['image'].shape[0] == 6 * 6  # 6 frames × 6 cameras
        assert batch['target']['extrinsics'].shape[0] == 6 * 6
        assert batch['target']['intrinsics'].shape[0] == 6 * 6
        assert batch['target']['depth'].shape[0] == 6 * 6
        
        # Check intrinsics are 4x4
        assert batch['source']['intrinsics'].shape[1:] == (4, 4)
        assert batch['target']['intrinsics'].shape[1:] == (4, 4)
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_sample_random_batch(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test random batch sampling."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        # Load scene first
        scene_data = dataset._load_scene(0)
        if scene_data is None:
            pytest.skip("Scene loading failed, cannot test random sampling")
        
        dataset.train_scenes[0] = scene_data
        
        # Sample random batch
        batch = dataset.sample_random_batch()
        
        # Check batch format
        assert 'scene_id' in batch
        assert 'source' in batch
        assert 'target' in batch

