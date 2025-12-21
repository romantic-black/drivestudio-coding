"""
Utility functions for trajectory processing.
"""
import torch
from torch import Tensor
from typing import Tuple, List


def split_trajectory(
    trajectory: Tensor,
    num_splits: int = 0,
    min_count: int = 1,
    min_length: float = 0.0,
) -> Tuple[List[List[int]], Tensor]:
    """
    Split trajectory into segments based on distance.
    
    Args:
        trajectory (torch.Tensor): Trajectory tensor of shape [frame_num, 4, 4].
        num_splits (int): Number of splits. If 0, the function will automatically determine the number of splits.
        min_count (int): Minimum number of frames in each split.
        min_length (float): Minimum length of each split.
        
    Returns:
        segments (list): List of segments, each segment is a list of frame indices.
        ranges (torch.Tensor): Tensor of shape [num_splits, 2], each row is a range [start, end].
    """
    if trajectory.shape[0] == 0:
        return [], torch.empty((0, 2), dtype=torch.float32)
    
    if trajectory.shape[0] == 1:
        return [[0]], torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    
    positions = trajectory[:, :3, 3].cpu()
    frame_num = positions.shape[0]
    
    # Calculate distances between consecutive frames
    delta_positions = positions[1:] - positions[:-1]  # [frame_num - 1, 3]
    distances = torch.norm(delta_positions, dim=1)    # [frame_num - 1]
    cumulative_distances = torch.cat([
        torch.tensor([0.0], device=distances.device),
        torch.cumsum(distances, dim=0)
    ])  # [frame_num]
    total_distance = cumulative_distances[-1]
    
    # Handle zero distance case
    if total_distance == 0.0:
        # All frames are at the same position, return single segment
        num_splits = 1
    elif num_splits == 0:
        # Automatically determine the number of splits
        max_segments = frame_num
        
        # Find the maximum number of segments that satisfy constraints
        for n in range(max_segments, 0, -1):
            # Calculate segment boundaries
            segment_boundaries = torch.linspace(0, total_distance, steps=n + 1)
            
            # Determine which segment each frame belongs to
            segment_indices = torch.bucketize(cumulative_distances, segment_boundaries, right=False) - 1
            segment_indices = torch.clamp(segment_indices, min=0, max=n - 1)
            
            # Count frames in each segment
            counts = torch.bincount(segment_indices, minlength=n)
            
            # Check if all segments have at least min_count frames and satisfy min_length
            segment_lengths = segment_boundaries[1:] - segment_boundaries[:-1]
            if torch.all(counts >= min_count) and torch.all(segment_lengths >= min_length):
                num_splits = n
                break
        
        # If no valid split found, use 1 segment
        if num_splits == 0:
            num_splits = 1
    
    # Split trajectory into segments
    segment_length = total_distance / num_splits if num_splits > 0 else total_distance
    
    if segment_length == 0.0:
        # All frames at same position
        segment_indices = torch.zeros(frame_num, dtype=torch.long)
    else:
        segment_indices = (cumulative_distances / segment_length).long()
        segment_indices = torch.clamp(segment_indices, max=num_splits - 1)
    
    # Group frames into segments
    segments = [[] for _ in range(num_splits)]
    boundaries = torch.linspace(0, total_distance, steps=num_splits + 1)
    start, end = boundaries[:-1], boundaries[1:]
    ranges = torch.stack([start, end], dim=1)
    
    for i in range(num_splits):
        indices = torch.where(segment_indices == i)[0].tolist()
        segments[i] = indices
    
    return segments, ranges

