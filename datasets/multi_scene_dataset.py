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
        preload_scene_count: int = 3,
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
            preload_scene_count: Number of scenes to preload ahead (default 3)
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
        
        # Initialize preload scene count
        self.preload_scene_count = preload_scene_count
        
        # Initialize scene candidate pool (unvalidated scene IDs)
        self.scene_candidate_pool = train_scene_ids.copy()
        random.shuffle(self.scene_candidate_pool)  # Shuffle for randomness
        
        # Initialize training queue (validated scene IDs in training order)
        self.scene_training_queue = []
        
        # Initialize scene cache (loaded scene data, max preload_scene_count + 1 scenes)
        self.train_scenes_cache = {}
        
        # Initialize evaluation scenes (loaded on demand, can keep all)
        self.eval_scenes = {}
        
        # Initialize current scene index in queue
        self.current_scene_index = 0
        
        # Initialize invalid scene IDs set (validated but not suitable)
        self.invalid_scene_ids = set()
        
        # Track if initialized
        self._initialized = False
    
    def initialize(self):
        """
        Initialize training queue and preload initial scenes.
        
        This method:
        1. Initializes training queue (validates and adds initial scenes)
        2. Preloads initial scenes
        
        This is optional - the dataset will auto-initialize on first use,
        but calling this explicitly allows early error detection.
        """
        if self._initialized:
            logger.debug("Dataset already initialized")
            return
        
        logger.info("Initializing MultiSceneDataset...")
        
        # Ensure training queue has enough scenes
        self._ensure_training_queue_ready()
        
        if len(self.scene_training_queue) == 0:
            logger.warning("No valid training scenes found after validation")
            return
        
        logger.info(f"Training queue initialized with {len(self.scene_training_queue)} scenes")
        
        # Preload initial scenes
        self._preload_scenes()
        
        self._initialized = True
        logger.info("MultiSceneDataset initialization complete")
    
    def _build_segment_mapping(self):
        """Build mapping from scene_id to segment information."""
        self.scene_segment_counts = {}
        for scene_id, scene_data in self.train_scenes_cache.items():
            self.scene_segment_counts[scene_id] = len(scene_data['segments'])
    
    def _ensure_training_queue_ready(self):
        """
        Ensure training queue has enough scenes (at least preload_scene_count + 1).
        
        This method validates scenes from candidate pool and adds them to queue.
        If candidate pool is empty, reshuffle and refill from original scene IDs.
        """
        target_queue_size = self.preload_scene_count + 1
        
        # If queue already has enough scenes, return
        if len(self.scene_training_queue) >= target_queue_size:
            return
        
        # Try to fill queue from candidate pool
        while len(self.scene_training_queue) < target_queue_size and len(self.scene_candidate_pool) > 0:
            scene_id = self.scene_candidate_pool.pop(0)
            if self._validate_and_add_to_queue(scene_id):
                logger.debug(f"Scene {scene_id} validated and added to training queue")
            else:
                logger.debug(f"Scene {scene_id} is not suitable, skipping")
        
        # If candidate pool is empty and queue still not full, try to refill from original IDs
        if len(self.scene_training_queue) < target_queue_size and len(self.scene_candidate_pool) == 0:
            # Get remaining scene IDs that haven't been validated
            remaining_ids = [
                sid for sid in self.train_scene_ids 
                if sid not in self.scene_training_queue and sid not in self.invalid_scene_ids
            ]
            if len(remaining_ids) > 0:
                random.shuffle(remaining_ids)
                self.scene_candidate_pool = remaining_ids
                logger.info(f"Refilled candidate pool with {len(remaining_ids)} remaining scenes")
                # Try to fill queue again
                while len(self.scene_training_queue) < target_queue_size and len(self.scene_candidate_pool) > 0:
                    scene_id = self.scene_candidate_pool.pop(0)
                    if self._validate_and_add_to_queue(scene_id):
                        logger.debug(f"Scene {scene_id} validated and added to training queue")
                    else:
                        logger.debug(f"Scene {scene_id} is not suitable, skipping")
    
    def _validate_and_add_to_queue(self, scene_id: int) -> bool:
        """
        Validate a scene and add it to training queue if suitable.
        
        This method performs a lightweight validation by loading the scene
        and checking if it's suitable. If suitable, adds to queue.
        
        Args:
            scene_id: Scene ID to validate
            
        Returns:
            bool: True if scene is suitable and added to queue, False otherwise
        """
        # Skip if already in queue or invalid
        if scene_id in self.scene_training_queue:
            return True
        if scene_id in self.invalid_scene_ids:
            return False
        
        # Try to load and prepare scene (this does full validation)
        scene_data = self._load_and_prepare_scene(scene_id)
        
        if scene_data is not None:
            # Scene is suitable, add to queue
            self.scene_training_queue.append(scene_id)
            # Don't keep it in cache yet, will be loaded when needed
            # Clean up the loaded data to save memory
            if 'dataset' in scene_data:
                dataset = scene_data['dataset']
                if hasattr(dataset, 'cleanup'):
                    dataset.cleanup()
                if hasattr(dataset, 'pixel_source') and hasattr(dataset.pixel_source, 'cleanup'):
                    dataset.pixel_source.cleanup()
            del scene_data
            return True
        else:
            # Scene is not suitable, mark as invalid
            self.invalid_scene_ids.add(scene_id)
            return False
    
    def _initialize_training_queue(self):
        """
        Initialize training queue by validating all training scene IDs.
        
        This method validates all training scenes and filters out invalid ones,
        then creates a training queue. The queue can be shuffled or kept in order.
        """
        valid_scenes = []
        
        logger.info(f"Validating {len(self.train_scene_ids)} training scenes...")
        for scene_id in self.train_scene_ids:
            # Quick validation: try to load scene metadata (without full loading)
            # We'll do a lightweight check here
            scene_cfg = OmegaConf.create(OmegaConf.to_container(self.data_cfg))
            scene_cfg.scene_idx = scene_id
            
            try:
                # Create a temporary dataset to check if scene exists and is valid
                temp_dataset = DrivingDataset(scene_cfg)
                # Get trajectory to check keyframes
                trajectory = self._get_scene_trajectory(temp_dataset)
                keyframe_segments, _ = self._split_keyframes(trajectory)
                
                # Check if scene is suitable
                if self._is_scene_suitable(keyframe_segments):
                    valid_scenes.append(scene_id)
                    self.valid_train_scene_ids.add(scene_id)
                else:
                    logger.warning(f"Scene {scene_id} is not suitable for training (insufficient keyframes), skipping...")
                
                # Clean up temporary dataset
                del temp_dataset
                del trajectory
                del keyframe_segments
            except Exception as e:
                logger.warning(f"Failed to validate scene {scene_id}: {e}, skipping...")
        
        # Create training queue (can shuffle or keep order)
        self.scene_training_queue = valid_scenes.copy()
        random.shuffle(self.scene_training_queue)  # Shuffle for randomness
        
        logger.info(f"Training queue initialized with {len(self.scene_training_queue)} valid scenes")
    
    def _preload_scenes(self):
        """
        Preload scenes that will be needed next.
        
        This method ensures cache has:
        - Current scene (if exists)
        - Next preload_scene_count scenes
        
        If a scene fails to load, skip it and try the next one.
        """
        # Ensure queue has enough scenes
        self._ensure_training_queue_ready()
        
        if len(self.scene_training_queue) == 0:
            logger.warning("Training queue is empty, cannot preload scenes")
            return
        
        # Load current scene if not already loaded
        if self.current_scene_index < len(self.scene_training_queue):
            current_scene_id = self.scene_training_queue[self.current_scene_index]
            if current_scene_id not in self.train_scenes_cache:
                logger.info(f"Loading current scene {current_scene_id}...")
                scene_data = self._load_and_prepare_scene(current_scene_id)
                if scene_data is not None:
                    self.train_scenes_cache[current_scene_id] = scene_data
                    logger.info(f"Scene {current_scene_id} loaded successfully")
                else:
                    logger.warning(f"Failed to load current scene {current_scene_id}")
                    # Remove from queue if failed
                    if current_scene_id in self.scene_training_queue:
                        self.scene_training_queue.remove(current_scene_id)
        
        # Preload next scenes
        max_cache_size = self.preload_scene_count + 1  # Current + preload
        scenes_to_preload = []
        
        for i in range(1, self.preload_scene_count + 1):
            scene_idx = self.current_scene_index + i
            if scene_idx < len(self.scene_training_queue):
                scene_id = self.scene_training_queue[scene_idx]
                if scene_id not in self.train_scenes_cache:
                    scenes_to_preload.append(scene_id)
        
        # Load scenes one by one, stop if cache is full
        for scene_id in scenes_to_preload:
            if len(self.train_scenes_cache) >= max_cache_size:
                break
            
            logger.info(f"Preloading scene {scene_id}...")
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                self.train_scenes_cache[scene_id] = scene_data
                logger.info(f"Scene {scene_id} preloaded successfully")
            else:
                logger.warning(f"Failed to preload scene {scene_id}, will skip")
                # Remove from queue if failed
                if scene_id in self.scene_training_queue:
                    self.scene_training_queue.remove(scene_id)
    
    def _load_and_prepare_scene(self, scene_id: int) -> Optional[Dict]:
        """
        Load scene and complete all preprocessing.
        
        This method loads a scene and performs all necessary preprocessing:
        - Scene loading (DrivingDataset)
        - Trajectory extraction
        - Keyframe splitting
        - Scene suitability check
        - Segment splitting
        
        Args:
            scene_id: Scene ID to load
            
        Returns:
            Scene data dictionary or None if scene is not suitable
        """
        return self._load_scene(scene_id)
    
    def _unload_scene(self, scene_id: int):
        """
        Unload scene from cache and free memory.
        
        Args:
            scene_id: Scene ID to unload
        """
        if scene_id in self.train_scenes_cache:
            scene_data = self.train_scenes_cache[scene_id]
            
            # Clean up dataset if it has cleanup methods
            if 'dataset' in scene_data:
                dataset = scene_data['dataset']
                # Try to clean up dataset resources
                if hasattr(dataset, 'cleanup'):
                    dataset.cleanup()
                if hasattr(dataset, 'pixel_source') and hasattr(dataset.pixel_source, 'cleanup'):
                    dataset.pixel_source.cleanup()
            
            # Remove from cache
            del self.train_scenes_cache[scene_id]
            logger.info(f"Scene {scene_id} unloaded from cache")
    
    def _switch_to_next_scene(self):
        """
        Switch to next scene: unload current scene and load next from queue.
        
        This method:
        1. Unloads the current scene
        2. Updates current_scene_index
        3. Ensures queue has enough scenes
        4. Preloads the next scenes
        """
        # Get current scene ID
        if self.current_scene_index >= len(self.scene_training_queue):
            logger.warning("No more scenes in training queue")
            return
        
        current_scene_id = self.scene_training_queue[self.current_scene_index]
        
        # Unload current scene
        self._unload_scene(current_scene_id)
        
        # Update index
        self.current_scene_index += 1
        
        # Check if there's a next scene
        if self.current_scene_index >= len(self.scene_training_queue):
            logger.info("All scenes in training queue have been processed")
            # Try to refill queue
            self._ensure_training_queue_ready()
            if self.current_scene_index >= len(self.scene_training_queue):
                return  # Still no scenes available
        
        # Ensure queue has enough scenes
        self._ensure_training_queue_ready()
        
        # Preload next scenes
        self._preload_scenes()
    
    def _ensure_scene_loaded(self, scene_id: int) -> Optional[Dict]:
        """
        Ensure specified scene is loaded in cache.
        
        If scene is already in cache, return it.
        If not, load it using _load_and_prepare_scene.
        If cache is full, unload a non-current scene.
        
        Args:
            scene_id: Scene ID to ensure loaded
            
        Returns:
            Scene data dictionary or None if scene cannot be loaded
        """
        # Check if already in cache
        if scene_id in self.train_scenes_cache:
            return self.train_scenes_cache[scene_id]
        
        # Check if it's an evaluation scene
        if scene_id in self.eval_scene_ids:
            if scene_id not in self.eval_scenes:
                # Load evaluation scene
                scene_data = self._load_and_prepare_scene(scene_id)
                if scene_data is not None:
                    self.eval_scenes[scene_id] = scene_data
                else:
                    return None
            return self.eval_scenes[scene_id]
        
        # It's a training scene, check cache size
        max_cache_size = self.preload_scene_count + 1
        
        # If cache is full, unload a scene that's not current
        if len(self.train_scenes_cache) >= max_cache_size:
            # Find a scene to unload (prefer non-current scenes)
            current_scene_id = self.get_current_scene_id()
            for cached_scene_id in list(self.train_scenes_cache.keys()):
                if cached_scene_id != current_scene_id:
                    self._unload_scene(cached_scene_id)
                    break
            # If still full, unload any scene
            if len(self.train_scenes_cache) >= max_cache_size:
                scene_to_unload = list(self.train_scenes_cache.keys())[0]
                self._unload_scene(scene_to_unload)
        
        # Load the scene
        scene_data = self._load_and_prepare_scene(scene_id)
        if scene_data is not None:
            self.train_scenes_cache[scene_id] = scene_data
            return scene_data
        else:
            return None
    
    def get_current_scene_id(self) -> Optional[int]:
        """
        Get current training scene ID.
        
        Returns:
            Current scene ID or None if no scene is available
        """
        if (self.current_scene_index < len(self.scene_training_queue) and 
            len(self.scene_training_queue) > 0):
            return self.scene_training_queue[self.current_scene_index]
        return None
    
    def mark_scene_completed(self, scene_id: int):
        """
        Mark scene training as completed and switch to next scene.
        
        This method:
        1. Verifies the scene_id matches the current scene
        2. Switches to the next scene in the queue
        3. Unloads the completed scene
        4. Preloads the next scene (if available)
        
        Args:
            scene_id: Scene ID that has been completed
        """
        current_scene_id = self.get_current_scene_id()
        
        if current_scene_id is None:
            logger.warning("No current scene to mark as completed")
            return
        
        if scene_id != current_scene_id:
            logger.warning(f"Scene {scene_id} does not match current scene {current_scene_id}. Ignoring.")
            return
        
        # Switch to next scene
        self._switch_to_next_scene()
    
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
        return self._ensure_scene_loaded(scene_id)
    
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
        
        # 5. Group keyframes into segments based on distance with overlap
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
            # Multiple segments with overlap
            # Calculate segment distance and step distance
            segment_distance = total_keyframe_distance / num_segments
            # Clamp overlap_ratio to [0, 0.5] to avoid excessive overlap
            overlap_ratio_clamped = min(overlap_ratio, 0.5)
            step_distance = segment_distance * (1 - overlap_ratio_clamped)
            
            # Calculate how many overlapping segments we can generate
            max_start_distance = total_keyframe_distance - segment_distance
            if step_distance > 0:
                num_overlap_segments = int(max_start_distance / step_distance) + 1
            else:
                # If step_distance is 0 (overlap_ratio = 1), only generate one segment
                num_overlap_segments = 1
            
            # Generate overlapping segments by iterating multiple times
            for seg_idx in range(num_overlap_segments):
                segment_start_distance = seg_idx * step_distance
                segment_end_distance = segment_start_distance + segment_distance
                
                # Collect keyframes within this segment's distance range
                current_segment_kf_indices = []
                current_segment_frames = set()
                
                for kf_idx in range(len(keyframe_segments)):
                    kf_center_distance = (keyframe_ranges[kf_idx, 0] + keyframe_ranges[kf_idx, 1]) / 2.0
                    
                    # Check if keyframe is within this segment's range
                    if segment_start_distance <= kf_center_distance < segment_end_distance:
                        current_segment_kf_indices.append(kf_idx)
                        current_segment_frames.update(keyframe_segments[kf_idx])
                
                # Only add segment if it has enough keyframes
                if len(current_segment_kf_indices) >= self.min_keyframes_per_segment:
                    segments.append({
                        'segment_id': segment_id,
                        'keyframe_indices': current_segment_kf_indices,
                        'frame_indices': sorted(list(current_segment_frames)),
                        'aabb': scene_aabb,
                    })
                    segment_id += 1
        
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
        # 1. Ensure scene is loaded
        scene_data = self._ensure_scene_loaded(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded or is not suitable")
        
        # 2. Get scene and segment information
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
        
        # 4. Load source images (num_source_keyframes frames × num_cams cameras)
        num_cams = scene_dataset.num_cams
        num_source_images = len(source_frame_indices) * num_cams
        source_images = []
        source_extrinsics = []
        source_intrinsics = []
        source_depths = []
        source_frame_idxs = []
        source_cam_idxs = []
        
        for frame_idx in source_frame_indices:
            for cam_idx in range(num_cams):
                img_idx = frame_idx * num_cams + cam_idx
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
        
        # 5. Load target images (num_target_keyframes frames × num_cams cameras)
        target_images = []
        target_extrinsics = []
        target_intrinsics = []
        target_depths = []
        target_frame_idxs = []
        target_cam_idxs = []
        
        num_target_images = len(target_frame_indices) * num_cams
        for frame_idx in target_frame_indices:
            for cam_idx in range(num_cams):
                img_idx = frame_idx * num_cams + cam_idx
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
            
            # Keyframe information for debugging/display
            'keyframe_info': {
                'segment_keyframes': segment['keyframe_indices'],  # All keyframes in this segment
                'source_keyframes': source_keyframe_indices,  # Selected source keyframe indices
                'target_keyframes': target_keyframe_indices,  # Selected target keyframe indices (includes source)
            },
            
            'source': {
                'image': torch.stack(source_images, dim=0),  # [num_source_keyframes * num_cams, H, W, 3]
                'extrinsics': torch.stack(source_extrinsics, dim=0),  # [num_source_keyframes * num_cams, 4, 4]
                'intrinsics': torch.stack(source_intrinsics, dim=0),  # [num_source_keyframes * num_cams, 4, 4]
                'depth': torch.stack(source_depths, dim=0),  # [num_source_keyframes * num_cams, H, W]
                'frame_indices': torch.tensor(source_frame_idxs, dtype=torch.long),  # [num_source_keyframes * num_cams]
                'cam_indices': torch.tensor(source_cam_idxs, dtype=torch.long),  # [num_source_keyframes * num_cams]
                'keyframe_indices': torch.tensor(source_keyframe_indices, dtype=torch.long),  # [num_source_keyframes]
            },
            
            'target': {
                'image': torch.stack(target_images, dim=0),  # [num_target_keyframes * num_cams, H, W, 3]
                'extrinsics': torch.stack(target_extrinsics, dim=0),  # [num_target_keyframes * num_cams, 4, 4]
                'intrinsics': torch.stack(target_intrinsics, dim=0),  # [num_target_keyframes * num_cams, 4, 4]
                'depth': torch.stack(target_depths, dim=0),  # [num_target_keyframes * num_cams, H, W]
                'frame_indices': torch.tensor(target_frame_idxs, dtype=torch.long),  # [num_target_keyframes * num_cams]
                'cam_indices': torch.tensor(target_cam_idxs, dtype=torch.long),  # [num_target_keyframes * num_cams]
                'keyframe_indices': torch.tensor(target_keyframe_indices, dtype=torch.long),  # [num_target_keyframes]
            }
        }
        
        return batch
    
    def sample_random_batch(self) -> Dict:
        """
        Randomly sample a training batch from current scene.
        
        Returns:
            Same format as get_segment_batch()
        """
        # Ensure initialized
        if not self._initialized:
            self.initialize()
        
        # Get current scene ID
        current_scene_id = self.get_current_scene_id()
        if current_scene_id is None:
            # Try to initialize queue
            self._ensure_training_queue_ready()
            current_scene_id = self.get_current_scene_id()
            if current_scene_id is None:
                raise ValueError("No training scenes available. Please check scene IDs and configuration.")
        
        # Ensure current scene is loaded
        scene_data = self._ensure_scene_loaded(current_scene_id)
        if scene_data is None:
            raise ValueError(f"Current scene {current_scene_id} cannot be loaded")
        
        # Randomly select segment from current scene
        if len(scene_data['segments']) == 0:
            raise ValueError(f"Scene {current_scene_id} has no valid segments")
        
        segment_id = random.choice(range(len(scene_data['segments'])))
        
        return self.get_segment_batch(current_scene_id, segment_id)
    
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
    
    def create_scheduler(
        self,
        batches_per_segment: int = 20,
        segment_order: str = "random",
        scene_order: str = "random",
        shuffle_segments: bool = True,
        preload_next_scene: bool = True,
    ) -> 'MultiSceneDatasetScheduler':
        """
        Create a scheduler instance for managing scene and segment traversal.
        
        Args:
            batches_per_segment: Number of batches to iterate per segment (default 20)
            segment_order: Segment traversal order ("random" or "sequential", default "random")
            scene_order: Scene traversal order ("random" or "sequential", default "random")
            shuffle_segments: Whether to shuffle segments within each scene (default True)
            preload_next_scene: Whether to preload next scene when last segment starts (default True)
            
        Returns:
            MultiSceneDatasetScheduler instance
        """
        return MultiSceneDatasetScheduler(
            dataset=self,
            batches_per_segment=batches_per_segment,
            segment_order=segment_order,
            scene_order=scene_order,
            shuffle_segments=shuffle_segments,
            preload_next_scene=preload_next_scene,
        )


class MultiSceneDatasetScheduler:
    """
    Scheduler for managing scene and segment traversal in MultiSceneDataset.
    
    This class manages the order of scene and segment traversal, automatically
    switching between segments and scenes, and preloading next scenes.
    """
    
    def __init__(
        self,
        dataset: MultiSceneDataset,
        batches_per_segment: int = 20,
        segment_order: str = "random",
        scene_order: str = "random",
        shuffle_segments: bool = True,
        preload_next_scene: bool = True,
    ):
        """
        Initialize scheduler.
        
        Args:
            dataset: MultiSceneDataset instance
            batches_per_segment: Number of batches to iterate per segment (default 20)
            segment_order: Segment traversal order ("random" or "sequential", default "random")
            scene_order: Scene traversal order ("random" or "sequential", default "random")
            shuffle_segments: Whether to shuffle segments within each scene (default True)
            preload_next_scene: Whether to preload next scene when last segment starts (default True)
        """
        self.dataset = dataset
        self.batches_per_segment = batches_per_segment
        self.segment_order = segment_order
        self.scene_order = scene_order
        self.shuffle_segments = shuffle_segments
        self.preload_next_scene = preload_next_scene
        
        # State variables
        self.current_scene_id: Optional[int] = None
        self.current_segment_id: int = 0  # Index in scene_segment_order
        self.current_batch_count: int = 0
        self.scene_segment_order: List[int] = []  # Segment IDs in traversal order
        
        # Initialize scheduler state
        self._initialize_scheduler_state()
    
    def _initialize_scheduler_state(self):
        """Initialize scheduler state."""
        # Ensure dataset is initialized
        if not self.dataset._initialized:
            self.dataset.initialize()
        
        # Get current scene
        self.current_scene_id = self.dataset.get_current_scene_id()
        if self.current_scene_id is None:
            raise ValueError("No training scenes available. Please check scene IDs and configuration.")
        
        # Initialize segment order
        self._initialize_segment_order()
        
        # Reset batch count
        self.current_batch_count = 0
    
    def _initialize_segment_order(self):
        """Initialize segment traversal order for current scene."""
        scene_data = self.dataset.get_scene(self.current_scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {self.current_scene_id} cannot be loaded")
        
        num_segments = len(scene_data['segments'])
        if num_segments == 0:
            raise ValueError(f"Scene {self.current_scene_id} has no valid segments")
        
        # Create segment order
        if self.segment_order == "random":
            self.scene_segment_order = list(range(num_segments))
            if self.shuffle_segments:
                random.shuffle(self.scene_segment_order)
        elif self.segment_order == "sequential":
            self.scene_segment_order = list(range(num_segments))
        else:
            raise ValueError(f"Invalid segment_order: {self.segment_order}. Must be 'random' or 'sequential'")
    
    def _switch_to_next_segment(self):
        """Switch to next segment."""
        self.current_batch_count = 0
        self.current_segment_id += 1
        
        # Check if we've finished all segments in current scene
        scene_data = self.dataset.get_scene(self.current_scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {self.current_scene_id} cannot be loaded")
        
        if self.current_segment_id >= len(self.scene_segment_order):
            # All segments in current scene are done, switch to next scene
            self._switch_to_next_scene()
            # Reinitialize segment order for new scene
            self._initialize_segment_order()
            self.current_segment_id = 0
    
    def _switch_to_next_scene(self):
        """Switch to next scene."""
        # Mark current scene as completed
        if self.current_scene_id is not None:
            self.dataset.mark_scene_completed(self.current_scene_id)
        
        # Get next scene
        self.current_scene_id = self.dataset.get_current_scene_id()
        if self.current_scene_id is None:
            raise StopIteration("All scenes have been processed")
        
        # Ensure next scene is loaded
        scene_data = self.dataset._ensure_scene_loaded(self.current_scene_id)
        if scene_data is None:
            raise ValueError(f"Next scene {self.current_scene_id} cannot be loaded")
    
    def _preload_next_scene_if_needed(self):
        """Preload next scene if we're starting the last segment."""
        scene_data = self.dataset.get_scene(self.current_scene_id)
        if scene_data is None:
            return
        
        # Check if current segment is the last one
        is_last_segment = (self.current_segment_id == len(self.scene_segment_order) - 1)
        
        if is_last_segment and self.preload_next_scene:
            # Get next scene ID from queue
            current_scene_index = self.dataset.scene_training_queue.index(self.current_scene_id)
            next_scene_index = current_scene_index + 1
            
            if next_scene_index < len(self.dataset.scene_training_queue):
                next_scene_id = self.dataset.scene_training_queue[next_scene_index]
                # Preload next scene
                self.dataset._ensure_scene_loaded(next_scene_id)
    
    def next_batch(self) -> Dict:
        """
        Get next training batch.
        
        Automatically manages:
        1. Batch count within current segment
        2. Segment switching (when batches_per_segment is reached)
        3. Scene switching (when all segments are done)
        4. Scene preloading (when last segment starts)
        
        Returns:
            Batch dictionary (same format as get_segment_batch())
            
        Raises:
            StopIteration: When all scenes have been processed
        """
        # Check if we need to switch to next segment
        if self.current_batch_count >= self.batches_per_segment:
            self._switch_to_next_segment()
        
        # Preload next scene if needed (when starting last segment)
        if self.current_batch_count == 0:
            self._preload_next_scene_if_needed()
        
        # Get current segment ID from order
        segment_id = self.scene_segment_order[self.current_segment_id]
        
        # Get batch
        batch = self.dataset.get_segment_batch(self.current_scene_id, segment_id)
        
        # Increment batch count
        self.current_batch_count += 1
        
        return batch
    
    def reset(self):
        """Reset scheduler state."""
        self.current_scene_id = None
        self.current_segment_id = 0
        self.current_batch_count = 0
        self.scene_segment_order = []
        self._initialize_scheduler_state()
    
    def get_current_info(self) -> Dict:
        """
        Get current scheduler state information.
        
        Returns:
            Dict containing:
                - 'scene_id': Current scene ID
                - 'segment_id': Current segment ID (in scene_segment_order)
                - 'segment_id_in_scene': Actual segment ID in scene
                - 'batch_count': Current batch count within segment
                - 'batches_per_segment': Number of batches per segment
        """
        segment_id_in_scene = (
            self.scene_segment_order[self.current_segment_id]
            if self.current_segment_id < len(self.scene_segment_order)
            else None
        )
        
        return {
            'scene_id': self.current_scene_id,
            'segment_id': self.current_segment_id,
            'segment_id_in_scene': segment_id_in_scene,
            'batch_count': self.current_batch_count,
            'batches_per_segment': self.batches_per_segment,
        }

