"""
Segment-level instance motion helpers for point cloud generation.

Trajectory length matches datasets/driving_dataset.py get_init_objects (only_moving):
sum of L2 norms of consecutive translations on frames where the instance is visible,
restricted to the given frame_indices (e.g. segment frames).
"""

from __future__ import annotations

from typing import List, Set

import torch


def compute_static_instance_intids(
    pixel_source,
    frame_indices: List[int],
    traj_length_thresh_m: float,
) -> Set[int]:
    """
    Instances whose cumulative translation delta along visible segment frames is
    <= traj_length_thresh_m are treated as static (background-only for LiDAR split).

    Uses pixel_source.instances_pose in dataset coordinates (no seg0 transform), matching
    DrivingDataset.get_init_objects trajectory filtering.

    Args:
        pixel_source: Must provide instances_pose [F,N,4,4] and per_frame_instance_mask [F,N].
        frame_indices: Segment frame indices to consider (e.g. after sparsity).
        traj_length_thresh_m: If cumulative motion <= this value, instance is static.

    Returns:
        Set of instance indices (intid / column index in instances_pose) to skip for dynamic.
    """
    if pixel_source is None or getattr(pixel_source, "instances_pose", None) is None:
        return set()

    instances_pose = pixel_source.instances_pose
    per_frame_mask = getattr(pixel_source, "per_frame_instance_mask", None)
    if per_frame_mask is None:
        raise ValueError(
            "Static instance motion filtering requires pixel_source.per_frame_instance_mask; "
            "ensure loader provides per-frame instance visibility or disable static_instance_motion."
        )

    if not isinstance(instances_pose, torch.Tensor):
        instances_pose = torch.as_tensor(instances_pose)
    if not isinstance(per_frame_mask, torch.Tensor):
        per_frame_mask = torch.as_tensor(per_frame_mask, dtype=torch.bool)

    instances_pose = instances_pose.float()
    num_frames, num_instances = instances_pose.shape[0], instances_pose.shape[1]

    frame_set = sorted(int(f) for f in frame_indices if 0 <= int(f) < num_frames)
    if len(frame_set) == 0:
        return set()

    static_ids: Set[int] = set()
    thresh = float(traj_length_thresh_m)

    for ins_id in range(num_instances):
        valid_frames = [f for f in frame_set if bool(per_frame_mask[f, ins_id].item())]
        if len(valid_frames) < 2:
            traj_length = torch.tensor(0.0, device=instances_pose.device, dtype=instances_pose.dtype)
        else:
            trans = instances_pose[valid_frames, ins_id, :3, 3]
            deltas = trans[1:] - trans[:-1]
            traj_length = torch.norm(deltas, dim=-1).sum()

        if float(traj_length.item()) <= thresh:
            static_ids.add(int(ins_id))

    return static_ids
