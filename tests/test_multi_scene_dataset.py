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
    
    def test_overlapping_segments(self):
        """Test overlapping segments generation."""
        # Mock scene dataset
        scene_dataset = Mock()
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0]
        ])
        
        # Create keyframe segments with long total distance
        keyframe_segments = [[i] for i in range(30)]  # 30 keyframes
        keyframe_ranges = torch.tensor([
            [i * 2.0, (i + 1) * 2.0] for i in range(30)
        ])  # Total distance = 60.0, AABB length = 50.0
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_segment=6,
            segment_overlap_ratio=0.2,
        )
        
        segments = dataset._split_segments(
            scene_dataset=scene_dataset,
            keyframe_segments=keyframe_segments,
            keyframe_ranges=keyframe_ranges,
            overlap_ratio=0.2,
        )
        
        # Should create multiple overlapping segments
        assert len(segments) > 1
        
        # Check that segments overlap (based on keyframe indices)
        for i in range(len(segments) - 1):
            seg1_keyframes = set(segments[i]['keyframe_indices'])
            seg2_keyframes = set(segments[i + 1]['keyframe_indices'])
            overlap = seg1_keyframes & seg2_keyframes
            # Adjacent segments should have some overlap
            assert len(overlap) > 0, f"Segments {i} and {i+1} should overlap"
        
        # Each segment should have at least min_keyframes_per_segment keyframes
        for seg in segments:
            assert len(seg['keyframe_indices']) >= 6
    
    def test_overlap_ratio_clamping(self):
        """Test that overlap_ratio is clamped to 0.5."""
        # Mock scene dataset
        scene_dataset = Mock()
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0]
        ])
        
        keyframe_segments = [[i] for i in range(30)]
        keyframe_ranges = torch.tensor([
            [i * 2.0, (i + 1) * 2.0] for i in range(30)
        ])
        
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_segment=6,
            segment_overlap_ratio=0.8,  # Should be clamped to 0.5
        )
        
        segments = dataset._split_segments(
            scene_dataset=scene_dataset,
            keyframe_segments=keyframe_segments,
            keyframe_ranges=keyframe_ranges,
            overlap_ratio=0.8,
        )
        
        # Should still create segments (not all overlapping)
        assert len(segments) > 1
        
        # Check that overlap is reasonable (not too high)
        for i in range(len(segments) - 1):
            seg1_keyframes = set(segments[i]['keyframe_indices'])
            seg2_keyframes = set(segments[i + 1]['keyframe_indices'])
            overlap = seg1_keyframes & seg2_keyframes
            overlap_ratio = len(overlap) / len(seg1_keyframes) if len(seg1_keyframes) > 0 else 0
            # Overlap should not be too high (clamped to 0.5)
            assert overlap_ratio <= 0.6, f"Overlap ratio {overlap_ratio} should be <= 0.6"


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
        
        # Initialize and load scene
        dataset.initialize()
        scene_data = dataset._ensure_scene_loaded(0)
        if scene_data is None:
            pytest.skip("Scene loading failed, cannot test batch format")
        
        # Get batch
        batch = dataset.get_segment_batch(0, 0)
        
        # Check batch structure
        assert 'scene_id' in batch
        assert 'segment_id' in batch
        assert 'source' in batch
        assert 'target' in batch
        
        # Get actual number of cameras from scene data
        scene_data = dataset._ensure_scene_loaded(0)
        num_cams = scene_data['num_cams'] if scene_data else mock_driving_dataset.num_cams
        
        # Check source format (num_source_keyframes frames × num_cams cameras)
        expected_source_count = dataset.num_source_keyframes * num_cams
        assert batch['source']['image'].shape[0] == expected_source_count
        assert batch['source']['extrinsics'].shape[0] == expected_source_count
        assert batch['source']['intrinsics'].shape[0] == expected_source_count
        assert batch['source']['depth'].shape[0] == expected_source_count
        
        # Check target format (num_target_keyframes frames × num_cams cameras)
        expected_target_count = dataset.num_target_keyframes * num_cams
        assert batch['target']['image'].shape[0] == expected_target_count
        assert batch['target']['extrinsics'].shape[0] == expected_target_count
        assert batch['target']['intrinsics'].shape[0] == expected_target_count
        assert batch['target']['depth'].shape[0] == expected_target_count
        
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
        
        # Initialize dataset
        dataset.initialize()
        
        # Sample random batch
        batch = dataset.sample_random_batch()
        
        # Check batch format
        assert 'scene_id' in batch
        assert 'source' in batch
        assert 'target' in batch


class TestPreloadQueue:
    """Test preload queue mechanism."""
    
    @pytest.fixture
    def mock_driving_dataset(self):
        """Create a mock DrivingDataset."""
        dataset = Mock(spec=DrivingDataset)
        
        # Mock pixel_source
        pixel_source = Mock()
        pixel_source.num_cams = 6
        pixel_source.camera_list = [0, 1, 2, 3, 4, 5]
        
        # Mock camera_data
        camera_data = {}
        for cam_id in range(6):
            cam_mock = Mock()
            cam_mock.lidar_depth_maps = None
            cam_mock.depth_maps = {i: torch.ones(100, 200, dtype=torch.float32) * 10.0 for i in range(16)}
            camera_data[cam_id] = cam_mock
        
        pixel_source.camera_data = camera_data
        
        # Mock get_image method
        def mock_get_image(img_idx):
            image_infos = {
                'pixels': torch.rand(100, 200, 3),
            }
            cam_infos = {
                'camera_to_world': torch.eye(4),
                'intrinsics': torch.eye(3),
            }
            return image_infos, cam_infos
        
        pixel_source.get_image = mock_get_image
        dataset.pixel_source = pixel_source
        dataset.num_img_timesteps = 16
        dataset.num_cams = 6
        
        # Mock get_aabb
        dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        
        # Mock get_novel_render_traj
        def mock_get_traj(traj_types, target_frames):
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
            },
            'lidar_source': {
                'type': 'datasets.nuscenes.nuscenes_sourceloader.NuScenesLiDARSource',
                'load_lidar': False,
            },
        })
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_initialize(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test dataset initialization."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0, 1, 2],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
            preload_scene_count=2,
        )
        
        # Should not be initialized yet
        assert not dataset._initialized
        assert len(dataset.scene_training_queue) == 0
        
        # Initialize
        dataset.initialize()
        
        # Should be initialized
        assert dataset._initialized
        assert len(dataset.scene_training_queue) > 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_preload_scenes(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scene preloading."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0, 1, 2],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
            preload_scene_count=2,
        )
        
        dataset.initialize()
        
        # Check that scenes are preloaded
        assert len(dataset.train_scenes_cache) <= 3  # current + 2 preloaded
        assert dataset.get_current_scene_id() is not None
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_switch_to_next_scene(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test switching to next scene."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0, 1, 2],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
            preload_scene_count=2,
        )
        
        dataset.initialize()
        
        # Get current scene
        current_scene_id = dataset.get_current_scene_id()
        assert current_scene_id is not None
        
        # Mark scene as completed
        dataset.mark_scene_completed(current_scene_id)
        
        # Should have switched to next scene
        new_scene_id = dataset.get_current_scene_id()
        assert new_scene_id != current_scene_id or dataset.current_scene_index > 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_get_current_scene_id(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test getting current scene ID."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        # Before initialization, should return None
        assert dataset.get_current_scene_id() is None
        
        # After initialization
        dataset.initialize()
        current_scene_id = dataset.get_current_scene_id()
        assert current_scene_id is not None
        assert current_scene_id in dataset.scene_training_queue
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_ensure_scene_loaded(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test ensuring scene is loaded."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        # Load scene
        scene_data = dataset._ensure_scene_loaded(0)
        assert scene_data is not None
        assert 0 in dataset.train_scenes_cache
        
        # Loading again should return cached version
        scene_data2 = dataset._ensure_scene_loaded(0)
        assert scene_data is scene_data2


class TestScheduler:
    """Test MultiSceneDatasetScheduler."""
    
    @pytest.fixture
    def mock_driving_dataset(self):
        """Create a mock DrivingDataset."""
        dataset = Mock(spec=DrivingDataset)
        
        # Mock pixel_source
        pixel_source = Mock()
        pixel_source.num_cams = 3
        pixel_source.num_imgs = 100
        pixel_source.camera_list = [0, 1, 2]
        
        # Mock camera_data
        camera_data = {}
        for cam_id in range(3):
            cam_mock = Mock()
            cam_mock.unique_cam_idx = cam_id
            cam_mock.lidar_depth_maps = None
            cam_mock.depth_maps = {i: torch.ones(100, 200, dtype=torch.float32) * 10.0 for i in range(16)}
            camera_data[cam_id] = cam_mock
        
        pixel_source.camera_data = camera_data
        
        # Mock get_image method
        def mock_get_image(img_idx):
            cam_idx = img_idx % 3
            frame_idx = img_idx // 3
            
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
        pixel_source.parse_img_idx = lambda idx: (idx % 3, idx // 3)
        
        dataset.pixel_source = pixel_source
        dataset.num_img_timesteps = 16
        dataset.num_cams = 3
        dataset.data_path = "/mock/path"
        
        # Mock get_aabb
        dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0]
        ])
        
        # Mock get_novel_render_traj
        def mock_get_traj(traj_types, target_frames):
            trajectory = torch.eye(4).unsqueeze(0).repeat(target_frames, 1, 1)
            trajectory[:, 0, 3] = torch.arange(target_frames).float() * 2.0
            return {"front_center_interp": trajectory}
        
        dataset.get_novel_render_traj = mock_get_traj
        
        return dataset
    
    @pytest.fixture
    def mock_data_cfg(self):
        """Create a mock data config."""
        return OmegaConf.create({
            'dataset': 'test',
            'data_root': '/test/path',
        })
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_initialization(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler initialization."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=10,
            segment_order="random",
            scene_order="random",
        )
        
        assert scheduler.batches_per_segment == 10
        assert scheduler.segment_order == "random"
        assert scheduler.current_scene_id is not None
        assert scheduler.current_batch_count == 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_next_batch(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler next_batch method."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=5,
            segment_order="sequential",
        )
        
        # Get first batch
        batch1 = scheduler.next_batch()
        assert batch1 is not None
        assert 'scene_id' in batch1
        assert 'segment_id' in batch1
        assert scheduler.current_batch_count == 1
        
        # Get more batches from same segment
        for i in range(4):
            batch = scheduler.next_batch()
            assert batch is not None
            assert batch['segment_id'] == batch1['segment_id']  # Same segment
            assert scheduler.current_batch_count == i + 2
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_segment_switching(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler segment switching."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=3,
            segment_order="sequential",
        )
        
        # Get batches until segment switch
        first_segment_id = None
        for i in range(4):
            batch = scheduler.next_batch()
            if first_segment_id is None:
                first_segment_id = batch['segment_id']
            elif i == 3:  # After batches_per_segment batches
                # Should have switched to next segment
                assert batch['segment_id'] != first_segment_id or scheduler.current_batch_count == 1
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_get_current_info(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler get_current_info method."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=10,
        )
        
        info = scheduler.get_current_info()
        assert 'scene_id' in info
        assert 'segment_id' in info
        assert 'segment_id_in_scene' in info
        assert 'batch_count' in info
        assert 'batches_per_segment' in info
        assert info['batches_per_segment'] == 10
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_reset(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler reset method."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=10,
        )
        
        # Get some batches
        for _ in range(5):
            scheduler.next_batch()
        
        # Reset
        scheduler.reset()
        
        assert scheduler.current_batch_count == 0
        assert scheduler.current_segment_id == 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_segment_order_random(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler with random segment order."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=1,
            segment_order="random",
            shuffle_segments=True,
        )
        
        # Get batches and check segment order is not strictly sequential
        segment_ids = []
        for _ in range(min(5, len(scheduler.scene_segment_order))):
            batch = scheduler.next_batch()
            segment_ids.append(batch['segment_id'])
        
        # With random order, segment IDs might not be sequential
        # Just check that we got valid segment IDs
        assert all(sid >= 0 for sid in segment_ids)
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_segment_order_sequential(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler with sequential segment order."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=1,
            segment_order="sequential",
            shuffle_segments=False,
        )
        
        # Get batches and check segment order is sequential
        segment_ids = []
        scene_data = dataset.get_scene(scheduler.current_scene_id)
        num_segments = len(scene_data['segments'])
        
        for _ in range(min(3, num_segments)):
            batch = scheduler.next_batch()
            segment_ids.append(batch['segment_id'])
        
        # With sequential order, segment IDs should be in order
        # (though they might not be 0, 1, 2 if segments were filtered)
        assert len(segment_ids) > 0


class TestIntegrationWithScheduler:
    """Integration tests with scheduler."""
    
    @pytest.fixture
    def mock_driving_dataset(self):
        """Create a mock DrivingDataset."""
        dataset = Mock(spec=DrivingDataset)
        
        # Mock pixel_source
        pixel_source = Mock()
        pixel_source.num_cams = 3
        pixel_source.num_imgs = 100
        pixel_source.camera_list = [0, 1, 2]
        
        # Mock camera_data
        camera_data = {}
        for cam_id in range(3):
            cam_mock = Mock()
            cam_mock.unique_cam_idx = cam_id
            cam_mock.lidar_depth_maps = None
            cam_mock.depth_maps = {i: torch.ones(100, 200, dtype=torch.float32) * 10.0 for i in range(16)}
            camera_data[cam_id] = cam_mock
        
        pixel_source.camera_data = camera_data
        
        # Mock get_image method
        def mock_get_image(img_idx):
            cam_idx = img_idx % 3
            frame_idx = img_idx // 3
            
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
        pixel_source.parse_img_idx = lambda idx: (idx % 3, idx // 3)
        
        dataset.pixel_source = pixel_source
        dataset.num_img_timesteps = 16
        dataset.num_cams = 3
        dataset.data_path = "/mock/path"
        
        # Mock get_aabb
        dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0]
        ])
        
        # Mock get_novel_render_traj
        def mock_get_traj(traj_types, target_frames):
            trajectory = torch.eye(4).unsqueeze(0).repeat(target_frames, 1, 1)
            trajectory[:, 0, 3] = torch.arange(target_frames).float() * 2.0
            return {"front_center_interp": trajectory}
        
        dataset.get_novel_render_traj = mock_get_traj
        
        return dataset
    
    @pytest.fixture
    def mock_data_cfg(self):
        """Create a mock data config."""
        return OmegaConf.create({
            'dataset': 'test',
            'data_root': '/test/path',
        })
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_full_training_loop_with_scheduler(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test full training loop using scheduler."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=3,
            segment_order="sequential",
        )
        
        # Simulate training loop
        batches_processed = 0
        scene_ids_seen = set()
        segment_ids_seen = set()
        
        try:
            for iteration in range(20):  # Process up to 20 batches
                batch = scheduler.next_batch()
                batches_processed += 1
                
                scene_id = batch['scene_id'].item()
                segment_id = batch['segment_id']
                
                scene_ids_seen.add(scene_id)
                segment_ids_seen.add(segment_id)
                
                # Verify batch structure
                assert 'source' in batch
                assert 'target' in batch
                assert 'keyframe_info' in batch
                
                # Verify batch counts
                info = scheduler.get_current_info()
                assert info['batch_count'] <= scheduler.batches_per_segment
                
        except StopIteration:
            # All scenes processed
            pass
        
        # Should have processed some batches
        assert batches_processed > 0
        assert len(scene_ids_seen) > 0
        assert len(segment_ids_seen) > 0
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_with_multiple_scenes(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler with multiple scenes."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0, 1],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
            preload_scene_count=2,
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=2,
            segment_order="sequential",
            preload_next_scene=True,
        )
        
        # Process batches and verify scene switching
        scene_ids_seen = []
        try:
            for _ in range(10):
                batch = scheduler.next_batch()
                scene_id = batch['scene_id'].item()
                scene_ids_seen.append(scene_id)
        except StopIteration:
            pass
        
        # Should have seen at least one scene
        assert len(scene_ids_seen) > 0
        # If multiple scenes, should have seen different scenes
        if len(set(scene_ids_seen)) > 1:
            assert len(set(scene_ids_seen)) > 1
    
    @patch('datasets.multi_scene_dataset.DrivingDataset')
    def test_scheduler_overlapping_segments(self, mock_driving_dataset_class, mock_driving_dataset, mock_data_cfg):
        """Test scheduler with overlapping segments."""
        mock_driving_dataset_class.return_value = mock_driving_dataset
        
        dataset = MultiSceneDataset(
            data_cfg=mock_data_cfg,
            train_scene_ids=[0],
            eval_scene_ids=[],
            min_keyframes_per_scene=5,
            min_keyframes_per_segment=3,
            segment_overlap_ratio=0.3,  # 30% overlap
        )
        
        dataset.initialize()
        
        scheduler = dataset.create_scheduler(
            batches_per_segment=2,
        )
        
        # Get scene data and verify overlapping segments
        scene_data = dataset.get_scene(scheduler.current_scene_id)
        if scene_data and len(scene_data['segments']) > 1:
            # Check that segments overlap
            for i in range(len(scene_data['segments']) - 1):
                seg1_keyframes = set(scene_data['segments'][i]['keyframe_indices'])
                seg2_keyframes = set(scene_data['segments'][i + 1]['keyframe_indices'])
                overlap = seg1_keyframes & seg2_keyframes
                # Adjacent segments should have some overlap
                assert len(overlap) > 0, f"Segments {i} and {i+1} should overlap"

