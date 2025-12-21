"""
MultiSceneDataset for EVolSplat feed-forward 3DGS training.

This module implements a multi-scene dataset class that supports:
- Multiple scene management
- Keyframe-based segmentation
- Segment-based scene splitting
- Source/target image pair generation
"""
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from datasets.driving_dataset import DrivingDataset
from datasets.tools.trajectory_utils import split_trajectory

logger = logging.getLogger(__name__)


class MultiSceneDataset:
    """
    Multi-scene dataset class for EVolSplat feed-forward 3DGS training.
    
    Core functionality:
    1. Manage multiple scenes with train/eval split
    2. Segment scenes based on keyframes
    3. Randomly select source and target keyframes within segments
    4. Package data into EVolSplat format batches
    """
    
    def __init__(
        self,
        data_cfg: OmegaConf,
        train_scene_ids: List[int],
        eval_scene_ids: List[int],
        num_source_keyframes: int = 3,
        num_target_keyframes: int = 6,
        segment_overlap_ratio: float = 0.2,
        keyframe_split_config: Optional[Dict] = None,
        min_keyframes_per_scene: int = 10,
        min_keyframes_per_segment: int = 6,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize MultiSceneDataset.
        
        Args:
            data_cfg: Drivestudio data configuration (OmegaConf)
            train_scene_ids: List of training scene IDs
            eval_scene_ids: List of evaluation scene IDs
            num_source_keyframes: Number of keyframes for source (default 3)
            num_target_keyframes: Number of keyframes for target (default 6, includes source)
            segment_overlap_ratio: Overlap ratio between segments (default 0.2)
            keyframe_split_config: Keyframe splitting configuration
                - num_splits: Number of splits (0 means auto)
                - min_count: Minimum frames per keyframe segment (default 1)
                - min_length: Minimum length per keyframe segment (default 0)
            min_keyframes_per_scene: Minimum keyframes per scene, skip if not met (default 10)
            min_keyframes_per_segment: Minimum keyframes per segment, skip if not met (default 6)
            device: Device (default CPU)
        """
        # Store configuration
        self.data_cfg = data_cfg
        self.train_scene_ids = train_scene_ids
        self.eval_scene_ids = eval_scene_ids
        self.num_source_keyframes = num_source_keyframes
        self.num_target_keyframes = num_target_keyframes
        self.segment_overlap_ratio = segment_overlap_ratio
        self.device = device
        
        # Initialize keyframe split configuration
        self.keyframe_split_config = keyframe_split_config or {
            'num_splits': 0,  # Auto-determine
            'min_count': 1,
            'min_length': 0.0,
        }
        self.min_keyframes_per_scene = min_keyframes_per_scene
        self.min_keyframes_per_segment = min_keyframes_per_segment
        
        # Load all training scenes (skip unsuitable ones)
        self.train_scenes = {}
        for scene_id in train_scene_ids:
            scene_data = self._load_scene(scene_id)
            if scene_data is not None:  # Scene is suitable
                self.train_scenes[scene_id] = scene_data
            else:
                logger.warning(f"Skipping training scene {scene_id} (not suitable)")
        
        # Load all evaluation scenes (skip unsuitable ones)
        self.eval_scenes = {}
        for scene_id in eval_scene_ids:
            scene_data = self._load_scene(scene_id)
            if scene_data is not None:  # Scene is suitable
                self.eval_scenes[scene_id] = scene_data
            else:
                logger.warning(f"Skipping eval scene {scene_id} (not suitable)")
        
        # Build scene to segment mapping
        self._build_segment_mapping()
    
    def _build_segment_mapping(self):
        """Build mapping from scene_id to segment information."""
        self.scene_segment_counts = {}
        for scene_id, scene_data in self.train_scenes.items():
            self.scene_segment_counts[scene_id] = len(scene_data['segments'])
    
    def get_scene(self, scene_id: int) -> Optional[Dict]:
        """
        Get scene data and information.
        
        Args:
            scene_id: Scene ID (global index)
            
        Returns:
            Dict containing:
                - 'dataset': DrivingDataset instance
                - 'segments': List[Dict] - Segment information list
                - 'keyframes': List[List[int]] - Keyframes for each segment
                - 'num_frames': int - Total frames in scene
                - 'num_cams': int - Number of cameras in scene
            Returns None if scene not found
        """
        if scene_id in self.train_scenes:
            return self.train_scenes[scene_id]
        elif scene_id in self.eval_scenes:
            return self.eval_scenes[scene_id]
        else:
            return None
    
    def _load_scene(self, scene_id: int) -> Optional[Dict]:
        """
        Load a single scene's data.
        
        Process:
        1. Create DrivingDataset instance
        2. Get scene trajectory (for keyframe splitting)
        3. Split keyframes
        4. Split segments (based on AABB constraints)
        5. Return scene information
        """
        # 1. Create scene configuration
        scene_cfg = OmegaConf.create(OmegaConf.to_container(self.data_cfg))
        scene_cfg.scene_idx = scene_id
        
        try:
            # 2. Create DrivingDataset instance
            scene_dataset = DrivingDataset(scene_cfg)
            
            # 3. Get scene trajectory (using front camera trajectory)
            trajectory = self._get_scene_trajectory(scene_dataset)
            
            # 4. Split keyframes
            keyframe_segments, keyframe_ranges = self._split_keyframes(trajectory)
            
            # 5. Check if scene is suitable for training (sufficient keyframes)
            if not self._is_scene_suitable(keyframe_segments):
                logger.warning(f"Scene {scene_id} is not suitable for training (insufficient keyframes), skipping...")
                return None  # Return None to indicate scene is not suitable
            
            # 6. Split segments (based on AABB constraints and keyframe distances)
            segments = self._split_segments(
                scene_dataset=scene_dataset,
                keyframe_segments=keyframe_segments,
                keyframe_ranges=keyframe_ranges,
                overlap_ratio=self.segment_overlap_ratio,
            )
            
            if len(segments) == 0:
                logger.warning(f"Scene {scene_id} has no valid segments after filtering, skipping...")
                return None
            
            return {
                'dataset': scene_dataset,
                'trajectory': trajectory,
                'keyframe_segments': keyframe_segments,
                'keyframe_ranges': keyframe_ranges,
                'segments': segments,
                'num_frames': scene_dataset.num_img_timesteps,
                'num_cams': scene_dataset.num_cams,
            }
        except Exception as e:
            logger.error(f"Failed to load scene {scene_id}: {e}")
            return None
    
    def _get_scene_trajectory(self, scene_dataset: DrivingDataset) -> Tensor:
        """
        Get scene trajectory (camera transformation matrices).
        
        Uses DrivingDataset's get_novel_render_traj method to get front camera trajectory.
        
        Returns:
            trajectory: Tensor[num_frames, 4, 4] - Camera transformation matrices
        """
        # Use DrivingDataset's get_novel_render_traj method
        # Get front camera trajectory ("front_center_interp")
        num_frames = scene_dataset.num_img_timesteps
        traj_dict = scene_dataset.get_novel_render_traj(["front_center_interp"], num_frames)
        trajectory = traj_dict["front_center_interp"]  # Tensor[num_frames, 4, 4]
        
        return trajectory
    
    def _is_scene_suitable(
        self,
        keyframe_segments: List[List[int]],
    ) -> bool:
        """
        Check if scene is suitable for training.
        
        Criteria:
        - Sufficient number of keyframes (>= min_keyframes_per_scene)
        
        Args:
            keyframe_segments: List of keyframe segments
            
        Returns:
            bool: True if scene is suitable, False otherwise
        """
        num_keyframes = len(keyframe_segments)
        
        if num_keyframes < self.min_keyframes_per_scene:
            return False
        
        return True
    
    def _split_keyframes(
        self,
        trajectory: Tensor,  # [num_frames, 4, 4]
    ) -> Tuple[List[List[int]], Tensor]:
        """
        Split trajectory into keyframes based on distance.
        
        Uses the split_trajectory function from trajectory_utils.
        
        Returns:
            keyframe_segments: List[List[int]] - Frame indices for each keyframe segment
            keyframe_ranges: Tensor[num_keyframes, 2] - Distance ranges for each keyframe segment
        """
        keyframe_segments, keyframe_ranges = split_trajectory(
            trajectory=trajectory,
            num_splits=self.keyframe_split_config['num_splits'],
            min_count=self.keyframe_split_config['min_count'],
            min_length=self.keyframe_split_config['min_length'],
        )
        
        return keyframe_segments, keyframe_ranges
    
    def _split_segments(
        self,
        scene_dataset: DrivingDataset,
        keyframe_segments: List[List[int]],
        keyframe_ranges: Tensor,  # [num_keyframes, 2] - Distance ranges for each keyframe segment
        overlap_ratio: float,
    ) -> List[Dict]:
        """
        Split scene into segments based on AABB constraints.
        
        Strategy:
        1. Get scene AABB and trajectory
        2. Calculate total keyframe distance
        3. Group keyframes into segments based on distance and AABB length
        4. Filter out segments with insufficient keyframes
        
        Note:
        - Segment splitting doesn't need to be precise, just compare total keyframe distance to AABB length
        - Set minimum keyframe count per segment, skip if not met
        - Segments can overlap (overlap_ratio)
        
        Args:
            scene_dataset: Scene dataset
            keyframe_segments: List of keyframe segments
            keyframe_ranges: Distance ranges for keyframe segments [num_keyframes, 2]
            overlap_ratio: Overlap ratio between segments
        
        Returns:
            segments: List[Dict] - Each segment contains:
                - 'segment_id': int - Segment ID
                - 'keyframe_indices': List[int] - Keyframe indices in this segment (global keyframe indices)
                - 'frame_indices': List[int] - All frame indices in this segment (deduplicated)
                - 'aabb': Tensor[2, 3] - Segment AABB bounds (uses scene AABB)
        """
        # 1. Get scene AABB
        scene_aabb = scene_dataset.get_aabb()  # [2, 3]
        scene_size = scene_aabb[1] - scene_aabb[0]  # [3]
        
        # 2. Calculate scene AABB main direction length (usually x or y, depends on driving direction)
        # Use largest dimension as main length
        aabb_length = scene_size.max().item()  # Scalar
        
        # 3. Calculate total keyframe distance
        # Each row of keyframe_ranges is [start_distance, end_distance]
        # Calculate length of each keyframe segment
        keyframe_lengths = keyframe_ranges[:, 1] - keyframe_ranges[:, 0]  # [num_keyframes]
        total_keyframe_distance = keyframe_lengths.sum().item()  # Total distance of all keyframe segments
        
        # 4. Determine number of segments based on distance and AABB length
        # If total keyframe distance is much smaller than AABB length, vehicle moved short distance, maybe only 1 segment
        # If total keyframe distance is close to AABB length, can split into multiple segments
        if total_keyframe_distance < aabb_length * 0.3:
            # Vehicle moved short distance, create only one segment
            num_segments = 1
        else:
            # Determine number of segments based on keyframe count and distance
            # Each segment needs at least min_keyframes_per_segment keyframes
            max_segments = len(keyframe_segments) // self.min_keyframes_per_segment
            # Use a more aggressive formula: if distance is close to AABB length, create more segments
            # Scale by distance ratio, with minimum of 2 segments if distance >= 0.3 * aabb_length
            distance_ratio = total_keyframe_distance / aabb_length
            num_segments_by_distance = max(2, int(distance_ratio * 3))  # More segments for longer distances
            num_segments = max(1, min(max_segments, num_segments_by_distance))
        
        # 5. Group keyframes into segments based on distance
        segments = []
        segment_id = 0
        
        if num_segments == 1:
            # Only one segment, include all keyframes
            all_frames = []
            for kf_seg in keyframe_segments:
                all_frames.extend(kf_seg)
            
            segments.append({
                'segment_id': segment_id,
                'keyframe_indices': list(range(len(keyframe_segments))),
                'frame_indices': sorted(list(set(all_frames))),
                'aabb': scene_aabb,
            })
        else:
            # Multiple segments, group by cumulative distance
            segment_distance = total_keyframe_distance / num_segments
            overlap_distance = segment_distance * overlap_ratio
            
            current_segment_kf_indices = []
            current_segment_frames = set()
            segment_start_distance = 0.0
            
            for kf_idx in range(len(keyframe_segments)):
                kf_length = keyframe_lengths[kf_idx].item()
                kf_center_distance = (keyframe_ranges[kf_idx, 0] + keyframe_ranges[kf_idx, 1]) / 2.0
                
                # Check if should start new segment
                if (len(current_segment_kf_indices) > 0 and 
                    kf_center_distance - segment_start_distance > segment_distance + overlap_distance):
                    # Check if current segment has enough keyframes
                    if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
                        segments.append({
                            'segment_id': segment_id,
                            'keyframe_indices': current_segment_kf_indices.copy(),
                            'frame_indices': sorted(list(current_segment_frames)),
                            'aabb': scene_aabb,  # Use scene AABB
                        })
                        segment_id += 1
                    
                    # Start new segment (consider overlap)
                    segment_start_distance = kf_center_distance - overlap_distance
                    current_segment_kf_indices = []
                    current_segment_frames = set()
                
                # Add keyframe to current segment
                current_segment_kf_indices.append(kf_idx)
                current_segment_frames.update(keyframe_segments[kf_idx])
            
            # Handle last segment
            if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
                segments.append({
                    'segment_id': segment_id,
                    'keyframe_indices': current_segment_kf_indices,
                    'frame_indices': sorted(list(current_segment_frames)),
                    'aabb': scene_aabb,
                })
        
        # 6. Filter out segments with insufficient keyframes (double check)
        valid_segments = [
            seg for seg in segments
            if len(seg['keyframe_indices']) >= self.min_keyframes_per_segment
        ]
        
        return valid_segments
    
    def _select_source_and_target_keyframes(
        self,
        segment: Dict,
        num_source_keyframes: int,
        num_target_keyframes: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Randomly select source and target keyframes within a segment.
        
        Strategy:
        1. Randomly select num_source_keyframes keyframes as source
        2. Randomly select (num_target_keyframes - num_source_keyframes) additional keyframes from remaining ones
        3. Target includes all source keyframes
        
        Returns:
            source_keyframe_indices: List[int] - Source keyframe index list
            target_keyframe_indices: List[int] - Target keyframe index list (includes source)
        """
        available_keyframes = segment['keyframe_indices']
        
        if len(available_keyframes) < num_source_keyframes:
            # If insufficient available keyframes, repeat
            source_keyframe_indices = available_keyframes * (num_source_keyframes // len(available_keyframes) + 1)
            source_keyframe_indices = source_keyframe_indices[:num_source_keyframes]
        else:
            # Randomly select source keyframes
            source_keyframe_indices = random.sample(available_keyframes, num_source_keyframes)
        
        # Calculate number of additional target keyframes needed
        num_extra_target_keyframes = num_target_keyframes - num_source_keyframes
        
        # Select additional target keyframes from remaining ones
        remaining_keyframes = [kf for kf in available_keyframes if kf not in source_keyframe_indices]
        
        if len(remaining_keyframes) == 0:
            # All keyframes were selected as source, repeat source keyframes for target
            extra_target_keyframes = source_keyframe_indices * (num_extra_target_keyframes // len(source_keyframe_indices) + 1)
            extra_target_keyframes = extra_target_keyframes[:num_extra_target_keyframes]
        elif len(remaining_keyframes) < num_extra_target_keyframes:
            # If insufficient remaining keyframes, repeat
            extra_target_keyframes = remaining_keyframes * (num_extra_target_keyframes // len(remaining_keyframes) + 1)
            extra_target_keyframes = extra_target_keyframes[:num_extra_target_keyframes]
        else:
            # Randomly select additional target keyframes
            extra_target_keyframes = random.sample(remaining_keyframes, num_extra_target_keyframes)
        
        # Target includes all source keyframes
        target_keyframe_indices = source_keyframe_indices + extra_target_keyframes
        
        return source_keyframe_indices, target_keyframe_indices
    
    def _select_frame_from_keyframe(
        self,
        keyframe_segment: List[int],  # Frame indices in keyframe segment
    ) -> int:
        """
        Randomly select one frame from keyframe segment.
        
        Args:
            keyframe_segment: Frame indices in keyframe segment
            
        Returns:
            frame_idx: Selected frame index
        """
        if len(keyframe_segment) == 0:
            raise ValueError("Keyframe segment is empty")
        
        # Randomly select one frame
        frame_idx = random.choice(keyframe_segment)
        
        return frame_idx
    
    def get_segment_batch(
        self,
        scene_id: int,
        segment_id: int,
    ) -> Dict:
        """
        Get training batch for specified scene and segment.
        """
        # 1. Get scene and segment information
        scene_data = self.train_scenes[scene_id]
        segment = scene_data['segments'][segment_id]
        scene_dataset = scene_data['dataset']
        
        # 2. Select source and target keyframes
        source_keyframe_indices, target_keyframe_indices = self._select_source_and_target_keyframes(
            segment=segment,
            num_source_keyframes=self.num_source_keyframes,
            num_target_keyframes=self.num_target_keyframes,
        )
        
        # 3. Select one frame from each keyframe
        source_frame_indices = []
        for kf_idx in source_keyframe_indices:
            keyframe_segment = scene_data['keyframe_segments'][kf_idx]
            frame_idx = self._select_frame_from_keyframe(keyframe_segment)
            source_frame_indices.append(frame_idx)
        
        target_frame_indices = []
        for kf_idx in target_keyframe_indices:
            keyframe_segment = scene_data['keyframe_segments'][kf_idx]
            frame_idx = self._select_frame_from_keyframe(keyframe_segment)
            target_frame_indices.append(frame_idx)
        
        # 4. Load source images (3 frames × 6 cameras = 18 images)
        source_images = []
        source_extrinsics = []
        source_intrinsics = []
        source_depths = []
        source_frame_idxs = []
        source_cam_idxs = []
        
        for frame_idx in source_frame_indices:
            for cam_idx in range(scene_dataset.num_cams):
                img_idx = frame_idx * scene_dataset.num_cams + cam_idx
                image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
                
                source_images.append(image_infos['pixels'])  # [H, W, 3]
                source_extrinsics.append(cam_infos['camera_to_world'])  # [4, 4]
                
                # Convert intrinsics to 4x4
                intrinsic_3x3 = cam_infos['intrinsics']  # [3, 3]
                intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
                source_intrinsics.append(intrinsic_4x4)
                
                # Get depth map
                depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
                if depth is None:
                    # If depth map doesn't exist, create placeholder
                    H, W = image_infos['pixels'].shape[:2]
                    depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
                source_depths.append(depth)
                
                source_frame_idxs.append(frame_idx)
                source_cam_idxs.append(cam_idx)
        
        # 5. Load target images (6 frames × 6 cameras = 36 images)
        target_images = []
        target_extrinsics = []
        target_intrinsics = []
        target_depths = []
        target_frame_idxs = []
        target_cam_idxs = []
        
        for frame_idx in target_frame_indices:
            for cam_idx in range(scene_dataset.num_cams):
                img_idx = frame_idx * scene_dataset.num_cams + cam_idx
                image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
                
                target_images.append(image_infos['pixels'])
                target_extrinsics.append(cam_infos['camera_to_world'])
                
                intrinsic_3x3 = cam_infos['intrinsics']
                intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic_3x3)
                target_intrinsics.append(intrinsic_4x4)
                
                depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
                if depth is None:
                    # If depth map doesn't exist, create placeholder
                    H, W = image_infos['pixels'].shape[:2]
                    depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
                target_depths.append(depth)
                
                target_frame_idxs.append(frame_idx)
                target_cam_idxs.append(cam_idx)
        
        # 6. Assemble batch
        batch = {
            'scene_id': torch.tensor([scene_id], dtype=torch.long),
            'segment_id': segment_id,
            
            'source': {
                'image': torch.stack(source_images, dim=0),  # [18, H, W, 3]
                'extrinsics': torch.stack(source_extrinsics, dim=0),  # [18, 4, 4]
                'intrinsics': torch.stack(source_intrinsics, dim=0),  # [18, 4, 4]
                'depth': torch.stack(source_depths, dim=0),  # [18, H, W]
                'frame_indices': torch.tensor(source_frame_idxs, dtype=torch.long),  # [18]
                'cam_indices': torch.tensor(source_cam_idxs, dtype=torch.long),  # [18]
            },
            
            'target': {
                'image': torch.stack(target_images, dim=0),  # [36, H, W, 3]
                'extrinsics': torch.stack(target_extrinsics, dim=0),  # [36, 4, 4]
                'intrinsics': torch.stack(target_intrinsics, dim=0),  # [36, 4, 4]
                'depth': torch.stack(target_depths, dim=0),  # [36, H, W]
                'frame_indices': torch.tensor(target_frame_idxs, dtype=torch.long),  # [36]
                'cam_indices': torch.tensor(target_cam_idxs, dtype=torch.long),  # [36]
            }
        }
        
        return batch
    
    def sample_random_batch(self) -> Dict:
        """
        Randomly sample a training batch.
        
        Returns:
            Same format as get_segment_batch()
        """
        if len(self.train_scenes) == 0:
            raise ValueError("No training scenes available")
        
        # Randomly select scene
        scene_id = random.choice(list(self.train_scenes.keys()))
        scene_data = self.train_scenes[scene_id]
        
        # Randomly select segment
        if len(scene_data['segments']) == 0:
            raise ValueError(f"Scene {scene_id} has no valid segments")
        
        segment_id = random.choice(range(len(scene_data['segments'])))
        
        return self.get_segment_batch(scene_id, segment_id)
    
    def _get_depth(
        self,
        scene_dataset: DrivingDataset,
        frame_idx: int,
        cam_idx: int,
    ) -> Optional[Tensor]:
        """
        Get depth map for specified frame and camera.
        
        Priority:
        1. Get from camera_data.depth_maps (loaded from files via depth_utils)
        2. Get from camera_data.lidar_depth_maps (from LiDAR projection)
        
        Returns:
            depth: Tensor[H, W] or None
        """
        try:
            pixel_source = scene_dataset.pixel_source
            cam_id = pixel_source.camera_list[cam_idx]
            camera_data = pixel_source.camera_data[cam_id]
            
            # Method 1: Get from depth_maps (loaded from files)
            if hasattr(camera_data, 'depth_maps') and camera_data.depth_maps is not None:
                depth = camera_data.depth_maps[frame_idx]  # Tensor[H, W]
                return depth.to(self.device)
            
            # Method 2: Get from lidar_depth_maps (from LiDAR projection)
            if camera_data.lidar_depth_maps is not None:
                depth = camera_data.lidar_depth_maps[frame_idx]  # Tensor[H, W]
                return depth.to(self.device)
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to get depth map for camera {cam_idx}, frame {frame_idx}: {e}")
        
        return None
    
    def _convert_intrinsic_to_4x4(self, intrinsic: Tensor) -> Tensor:
        """
        Convert 3x3 intrinsic matrix to 4x4 format.
        
        Args:
            intrinsic: Tensor[3, 3] or Tensor[4, 4]
            
        Returns:
            Tensor[4, 4]
        """
        if intrinsic.shape == (4, 4):
            return intrinsic
        
        assert intrinsic.shape == (3, 3), f"Unexpected intrinsic shape: {intrinsic.shape}"
        
        intrinsic_4x4 = torch.eye(4, dtype=intrinsic.dtype, device=intrinsic.device)
        intrinsic_4x4[:3, :3] = intrinsic
        
        return intrinsic_4x4

