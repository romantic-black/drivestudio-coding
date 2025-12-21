"""
Unit tests for trajectory_utils module.
"""
import pytest
import torch
from datasets.tools.trajectory_utils import split_trajectory


class TestSplitTrajectory:
    """Test cases for split_trajectory function."""
    
    def test_normal_case(self):
        """Test normal case with multiple frames."""
        # Create a simple linear trajectory
        num_frames = 10
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float()  # Move along x-axis
        
        segments, ranges = split_trajectory(trajectory, num_splits=3)
        
        assert len(segments) == 3
        assert ranges.shape == (3, 2)
        assert all(len(seg) > 0 for seg in segments)
        # Check that all frames are included
        all_indices = [idx for seg in segments for idx in seg]
        assert len(all_indices) == num_frames
        assert set(all_indices) == set(range(num_frames))
    
    def test_single_frame(self):
        """Test boundary case with single frame."""
        trajectory = torch.eye(4).unsqueeze(0)
        
        segments, ranges = split_trajectory(trajectory)
        
        assert len(segments) == 1
        assert segments[0] == [0]
        assert ranges.shape == (1, 2)
    
    def test_empty_trajectory(self):
        """Test boundary case with empty trajectory."""
        trajectory = torch.empty((0, 4, 4))
        
        segments, ranges = split_trajectory(trajectory)
        
        assert len(segments) == 0
        assert ranges.shape == (0, 2)
    
    def test_zero_distance(self):
        """Test case where all frames are at the same position."""
        num_frames = 5
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        # All frames at same position
        
        segments, ranges = split_trajectory(trajectory)
        
        assert len(segments) == 1
        assert len(segments[0]) == num_frames
        assert ranges.shape == (1, 2)
    
    def test_auto_determine_splits(self):
        """Test automatic determination of number of splits."""
        num_frames = 20
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float()
        
        segments, ranges = split_trajectory(trajectory, num_splits=0, min_count=1, min_length=0.0)
        
        assert len(segments) > 0
        assert all(len(seg) >= 1 for seg in segments)
    
    def test_min_count_constraint(self):
        """Test min_count constraint."""
        num_frames = 10
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float()
        
        segments, ranges = split_trajectory(trajectory, num_splits=0, min_count=3, min_length=0.0)
        
        assert all(len(seg) >= 3 for seg in segments)
    
    def test_min_length_constraint(self):
        """Test min_length constraint."""
        num_frames = 20
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float() * 0.1  # Small increments
        
        segments, ranges = split_trajectory(trajectory, num_splits=0, min_count=1, min_length=1.0)
        
        # Check that segment lengths satisfy min_length
        for i, (start, end) in enumerate(ranges):
            segment_length = end - start
            assert segment_length >= 1.0 or len(segments[i]) == 0
    
    def test_specified_num_splits(self):
        """Test with specified number of splits."""
        num_frames = 15
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float()
        
        segments, ranges = split_trajectory(trajectory, num_splits=5)
        
        assert len(segments) == 5
        assert ranges.shape == (5, 2)
    
    def test_3d_trajectory(self):
        """Test with 3D trajectory (not just x-axis)."""
        num_frames = 10
        trajectory = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
        trajectory[:, 0, 3] = torch.arange(num_frames).float()  # x
        trajectory[:, 1, 3] = torch.arange(num_frames).float() * 0.5  # y
        trajectory[:, 2, 3] = torch.arange(num_frames).float() * 0.2  # z
        
        segments, ranges = split_trajectory(trajectory, num_splits=3)
        
        assert len(segments) == 3
        assert all(len(seg) > 0 for seg in segments)

