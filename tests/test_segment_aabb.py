"""
Unit tests for segment AABB computation in MultiSceneDataset.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from omegaconf import OmegaConf

from datasets.multi_scene_dataset import MultiSceneDataset
from datasets.driving_dataset import DrivingDataset


class TestSegmentAABB:
    """Test segment AABB computation."""
    
    def test_compute_segment_aabb_basic(self):
        """Test basic segment AABB computation."""
        # Create mock lidar source
        lidar_source = Mock()
        lidar_source.data_cfg = OmegaConf.create({
            'lidar_downsample_factor': 4,
            'lidar_percentile': 0.02,
        })
        
        # Create mock lidar data for 3 frames
        num_points_per_frame = 1000
        num_frames = 3
        total_points = num_points_per_frame * num_frames
        
        # Create origins, directions, ranges, and timesteps
        # Frame 0: points around [0, 0, 0]
        # Frame 1: points around [10, 0, 0]
        # Frame 2: points around [20, 0, 0]
        origins = torch.zeros(total_points, 3)
        directions = torch.randn(total_points, 3)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        
        # ranges应该是[N, 1]形状（与实际代码一致）
        ranges = torch.ones(total_points, 1) * 10.0
        
        timesteps = torch.cat([
            torch.zeros(num_points_per_frame, dtype=torch.long),
            torch.ones(num_points_per_frame, dtype=torch.long),
            torch.ones(num_points_per_frame, dtype=torch.long) * 2,
        ])
        
        # Compute actual points
        points = origins + directions * ranges
        # Shift points for different frames
        points[timesteps == 1] += torch.tensor([10.0, 0.0, 0.0])
        points[timesteps == 2] += torch.tensor([20.0, 0.0, 0.0])
        
        # Update origins and ranges to match
        origins = points - directions * ranges
        
        lidar_source.origins = origins
        lidar_source.directions = directions
        lidar_source.ranges = ranges
        lidar_source.timesteps = timesteps
        
        # Create mock scene dataset
        scene_dataset = Mock()
        scene_dataset.lidar_source = lidar_source
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        
        # Create dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        # Test computing AABB for frames [0, 1]
        frame_indices = [0, 1]
        aabb = dataset._compute_segment_aabb(scene_dataset, frame_indices)
        
        # Check AABB shape
        assert aabb.shape == (2, 3), f"Expected AABB shape (2, 3), got {aabb.shape}"
        
        # Check that AABB min < max
        assert torch.all(aabb[0] < aabb[1]), "AABB min should be less than max"
        
        # Check that AABB contains the points (with some tolerance for percentile)
        mask = torch.isin(timesteps, torch.tensor([0, 1]))
        segment_points = points[mask]
        assert torch.all(segment_points >= aabb[0] - 1.0), "AABB should contain segment points"
        assert torch.all(segment_points <= aabb[1] + 1.0), "AABB should contain segment points"
    
    def test_compute_segment_aabb_different_segments(self):
        """Test that different segments have different AABBs."""
        # Create mock lidar source
        lidar_source = Mock()
        lidar_source.data_cfg = OmegaConf.create({
            'lidar_downsample_factor': 4,
            'lidar_percentile': 0.02,
        })
        
        # Create mock lidar data for 5 frames
        num_points_per_frame = 1000
        num_frames = 5
        total_points = num_points_per_frame * num_frames
        
        origins = torch.zeros(total_points, 3)
        directions = torch.randn(total_points, 3)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        # ranges应该是[N, 1]形状（与实际代码一致）
        ranges = torch.ones(total_points, 1) * 10.0
        
        timesteps = torch.cat([
            torch.ones(num_points_per_frame, dtype=torch.long) * i
            for i in range(num_frames)
        ])
        
        # Compute actual points with different offsets
        points = origins + directions * ranges
        for i in range(num_frames):
            points[timesteps == i] += torch.tensor([i * 10.0, 0.0, 0.0])
        
        origins = points - directions * ranges
        
        lidar_source.origins = origins
        lidar_source.directions = directions
        lidar_source.ranges = ranges
        lidar_source.timesteps = timesteps
        
        # Create mock scene dataset
        scene_dataset = Mock()
        scene_dataset.lidar_source = lidar_source
        scene_dataset.get_aabb.return_value = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        
        # Create dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        # Compute AABB for segment 1 (frames [0, 1])
        aabb1 = dataset._compute_segment_aabb(scene_dataset, [0, 1])
        
        # Compute AABB for segment 2 (frames [3, 4])
        aabb2 = dataset._compute_segment_aabb(scene_dataset, [3, 4])
        
        # AABBs should be different (segment 2 should be shifted in X)
        assert not torch.allclose(aabb1, aabb2), "Different segments should have different AABBs"
        
        # Segment 2 should have larger X coordinates
        assert aabb2[0, 0] > aabb1[1, 0], "Segment 2 should be to the right of segment 1"
    
    def test_compute_segment_aabb_no_lidar_source(self):
        """Test fallback to scene AABB when lidar source is None."""
        # Create mock scene dataset without lidar source
        scene_dataset = Mock()
        scene_dataset.lidar_source = None
        scene_aabb = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        scene_dataset.get_aabb.return_value = scene_aabb
        
        # Create dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        # Compute AABB should fallback to scene AABB
        aabb = dataset._compute_segment_aabb(scene_dataset, [0, 1, 2])
        
        assert torch.allclose(aabb, scene_aabb), "Should fallback to scene AABB when lidar source is None"
    
    def test_compute_segment_aabb_no_matching_frames(self):
        """Test fallback to scene AABB when no matching frames found."""
        # Create mock lidar source
        lidar_source = Mock()
        lidar_source.data_cfg = OmegaConf.create({
            'lidar_downsample_factor': 4,
            'lidar_percentile': 0.02,
        })
        
        # Create lidar data only for frames 0, 1, 2
        num_points = 1000
        lidar_source.origins = torch.zeros(num_points, 3)
        lidar_source.directions = torch.randn(num_points, 3)
        lidar_source.directions = lidar_source.directions / (lidar_source.directions.norm(dim=-1, keepdim=True) + 1e-8)
        # ranges应该是[N, 1]形状（与实际代码一致）
        lidar_source.ranges = torch.ones(num_points, 1) * 10.0
        lidar_source.timesteps = torch.zeros(num_points, dtype=torch.long)  # All points are from frame 0
        
        # Create mock scene dataset
        scene_dataset = Mock()
        scene_dataset.lidar_source = lidar_source
        scene_aabb = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        scene_dataset.get_aabb.return_value = scene_aabb
        
        # Create dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
        )
        
        # Try to compute AABB for frames [10, 11] which don't exist
        aabb = dataset._compute_segment_aabb(scene_dataset, [10, 11])
        
        # Should fallback to scene AABB
        assert torch.allclose(aabb, scene_aabb), "Should fallback to scene AABB when no matching frames"
    
    def test_segment_aabb_in_split_segments(self):
        """Test that _split_segments uses segment-specific AABBs."""
        # Create mock lidar source
        lidar_source = Mock()
        lidar_source.data_cfg = OmegaConf.create({
            'lidar_downsample_factor': 4,
            'lidar_percentile': 0.02,
        })
        
        # Create mock lidar data
        num_points_per_frame = 1000
        num_frames = 10
        total_points = num_points_per_frame * num_frames
        
        origins = torch.zeros(total_points, 3)
        directions = torch.randn(total_points, 3)
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        # ranges应该是[N, 1]形状（与实际代码一致）
        ranges = torch.ones(total_points, 1) * 10.0
        
        timesteps = torch.cat([
            torch.ones(num_points_per_frame, dtype=torch.long) * i
            for i in range(num_frames)
        ])
        
        # Compute actual points with different offsets
        points = origins + directions * ranges
        for i in range(num_frames):
            points[timesteps == i] += torch.tensor([i * 5.0, 0.0, 0.0])
        
        origins = points - directions * ranges
        
        lidar_source.origins = origins
        lidar_source.directions = directions
        lidar_source.ranges = ranges
        lidar_source.timesteps = timesteps
        
        # Create mock scene dataset
        scene_dataset = Mock()
        scene_dataset.lidar_source = lidar_source
        scene_aabb = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0]
        ])
        scene_dataset.get_aabb.return_value = scene_aabb
        
        # Create keyframe segments
        keyframe_segments = [[i] for i in range(10)]
        keyframe_ranges = torch.tensor([
            [i * 2.0, (i + 1) * 2.0] for i in range(10)
        ])  # Total distance = 20.0
        
        # Create dataset
        dataset = MultiSceneDataset(
            data_cfg=OmegaConf.create({}),
            train_scene_ids=[],
            eval_scene_ids=[],
            min_keyframes_per_segment=3,
        )
        
        # Split segments
        segments = dataset._split_segments(
            scene_dataset=scene_dataset,
            keyframe_segments=keyframe_segments,
            keyframe_ranges=keyframe_ranges,
            overlap_ratio=0.2,
        )
        
        # Check that segments have AABBs
        assert len(segments) > 0, "Should have at least one segment"
        
        # Check that each segment has an AABB
        for segment in segments:
            assert 'aabb' in segment, "Segment should have AABB"
            assert segment['aabb'].shape == (2, 3), "AABB should have shape (2, 3)"
            assert torch.all(segment['aabb'][0] < segment['aabb'][1]), "AABB min should be less than max"
        
        # If there are multiple segments, they should have different AABBs
        if len(segments) > 1:
            aabbs = [seg['aabb'] for seg in segments]
            # Check that at least two segments have different AABBs
            all_same = all(torch.allclose(aabbs[0], aabb) for aabb in aabbs[1:])
            assert not all_same, "Different segments should have different AABBs"

