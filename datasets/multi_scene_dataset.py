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
import queue
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from datasets.driving_dataset import DrivingDataset
from datasets.streetforward_assets import (
    AssetConfig,
    StreetForwardAssetStore,
    normalize_missing_policy,
)
from datasets.sky_mask_semantics import (
    normalize_sky_mask_to_one_is_sky,
    parse_sky_mask_semantics_from_data_cfg,
)
from datasets.tools.trajectory_utils import split_trajectory

if TYPE_CHECKING:
    from datasets.pointcloud_generators import RGBPointCloudGenerator

logger = logging.getLogger(__name__)


def _parse_static_instance_motion_cfg(pointcloud_config: Dict) -> Tuple[bool, Optional[float]]:
    """
    Parse dataset.pointcloud.static_instance_motion for LiDAR / hybrid generators.

    Returns:
        (enable, traj_length_thresh_m). When enable is False, thresh is None.
    """
    block = pointcloud_config.get("static_instance_motion")
    if block is None:
        return False, None
    if "enable" not in block:
        raise ValueError(
            "dataset.pointcloud.static_instance_motion.enable is required when static_instance_motion is set."
        )
    enable = bool(block["enable"])
    if enable:
        if "traj_length_thresh_m" not in block:
            raise ValueError(
                "dataset.pointcloud.static_instance_motion.traj_length_thresh_m is required when enable=true."
            )
        return True, float(block["traj_length_thresh_m"])
    return False, None


def _parse_monocular_dynamic_recovery_cfg(
    pointcloud_config: Dict,
) -> Tuple[bool, Optional[List[float]], Optional[int], str]:
    """
    Parse monocular dynamic recovery config.

    Returns:
        (
            enable,
            bbox_expand_xyz_m,
            max_points_per_instance,
            assignment,
        )
    """
    block = pointcloud_config.get("dynamic_recovery")
    if block is None:
        return False, None, None, "first_hit"
    if "enable" not in block:
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.enable is required when dynamic_recovery is set."
        )
    enable = bool(block["enable"])
    if not enable:
        return False, None, None, str(block.get("assignment", "first_hit"))

    if "bbox_expand_xyz_m" not in block:
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.bbox_expand_xyz_m is required when enable=true."
        )
    if "max_points_per_instance" not in block:
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.max_points_per_instance is required when enable=true."
        )

    bbox_expand = list(block["bbox_expand_xyz_m"])
    if len(bbox_expand) != 3:
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.bbox_expand_xyz_m must have 3 values [dx, dy, dz]."
        )
    bbox_expand = [float(x) for x in bbox_expand]
    if any(x < 0.0 for x in bbox_expand):
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.bbox_expand_xyz_m must be non-negative."
        )

    max_points = int(block["max_points_per_instance"])
    if max_points <= 0:
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.max_points_per_instance must be > 0."
        )

    assignment = str(block.get("assignment", "first_hit"))
    if assignment not in ("first_hit", "nearest_center"):
        raise ValueError(
            "dataset.pointcloud.dynamic_recovery.assignment must be one of ['first_hit', 'nearest_center']."
        )

    return True, bbox_expand, max_points, assignment


def _safe_json_serialize(obj):
    """
    Safely serialize an object to JSON, handling Mock objects and other non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    import json
    from unittest.mock import Mock
    
    if isinstance(obj, Mock):
        return f"<Mock: {type(obj).__name__}>"
    elif isinstance(obj, dict):
        return {k: _safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, torch.Tensor):
        return f"<Tensor: shape={list(obj.shape)}, dtype={obj.dtype}>"
    elif isinstance(obj, np.ndarray):
        return f"<ndarray: shape={obj.shape}, dtype={obj.dtype}>"
    else:
        try:
            # Try to serialize normally
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If serialization fails, return string representation
            return f"<{type(obj).__name__}: {str(obj)[:100]}>"


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
        segment_aabb: Optional[Union[Tensor, List, np.ndarray]] = None,
        pointcloud_config: Optional[Dict] = None,
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
            segment_aabb: Required segment AABB in segment-first-frame (seg0) coordinates.
                Shape: [2, 3] where aabb[0] is [x_min, y_min, z_min] and aabb[1] is [x_max, y_max, z_max].
        """
        # Store configuration
        self.data_cfg = data_cfg
        self._sky_mask_loader_semantics = parse_sky_mask_semantics_from_data_cfg(data_cfg)
        self.train_scene_ids = train_scene_ids
        self.eval_scene_ids = eval_scene_ids
        if num_source_keyframes != 1:
            logger.warning(
                "StreetForward requires a single source keyframe; overriding num_source_keyframes=%s to 1",
                num_source_keyframes,
            )
        self.num_source_keyframes = 1
        if num_target_keyframes < self.num_source_keyframes:
            raise ValueError(
                f"num_target_keyframes ({num_target_keyframes}) must be >= num_source_keyframes (1)"
            )
        self.num_target_keyframes = num_target_keyframes
        self.segment_overlap_ratio = segment_overlap_ratio
        self.device = device
        
        # Initialize keyframe split configuration
        self.keyframe_split_config = keyframe_split_config or {
            'num_splits': 0,  # Auto-determine
            'min_count': 1,
            'min_length': 2.0,
        }
        self.min_keyframes_per_scene = min_keyframes_per_scene
        self.min_keyframes_per_segment = min_keyframes_per_segment
        
        # Initialize preload scene count
        self.preload_scene_count = preload_scene_count

        # Required AABB (single source of truth; seg0 coords)
        if segment_aabb is None:
            raise ValueError("dataset.segment_aabb is required (seg0 coords) and must have shape [2, 3].")

        def _as_aabb_tensor(name: str, aabb) -> Tensor:
            if not isinstance(aabb, Tensor):
                aabb = torch.tensor(aabb, dtype=torch.float32)
            aabb = aabb.to(dtype=torch.float32)
            if aabb.shape != (2, 3):
                raise ValueError(f"{name} must have shape [2, 3], got {tuple(aabb.shape)}")
            if not torch.all(aabb[0] < aabb[1]):
                raise ValueError(f"{name} min must be strictly less than max for all axes")
            return aabb

        self.segment_aabb = _as_aabb_tensor("dataset.segment_aabb", segment_aabb).to(device)
        self.segment_aabb_np = self.segment_aabb.detach().cpu().numpy().astype(np.float32)
        
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
        
        # Thread lock for protecting queue and cache operations
        self._lock = threading.RLock()
        
        # Initialize point cloud generator (if config exists)
        if pointcloud_config is None:
            raise ValueError(
                "pointcloud_config is required for MultiSceneDataset; "
                "StreetForward training depends on point cloud initialization."
            )
        # Store for segment splitting reference length and batch['aabb'] construction
        self.pointcloud_config = pointcloud_config

        self.pointcloud_generator = self._create_pointcloud_generator(
            pointcloud_config, data_cfg, device
        )
        if self.pointcloud_generator is None:
            raise ValueError(
                "Failed to create pointcloud generator from pointcloud_config; "
                "please check the configuration."
            )
        # Segment-level pointcloud cache: {(scene_id, segment_id): pointcloud_dict}
        # Pointclouds are static for a segment and should not be rebuilt per step.
        self._segment_pointcloud_cache: Dict[Tuple[int, int], Dict] = {}

        assets_cfg_raw = data_cfg.get("assets")
        self.asset_config: Optional[AssetConfig] = None
        self.asset_store: Optional[StreetForwardAssetStore] = None
        self.use_prebuilt_assets = False
        self.asset_missing_policy = "error"
        if assets_cfg_raw is not None:
            assets_enable = bool(assets_cfg_raw.get("enable"))
            if assets_enable:
                if assets_cfg_raw.get("root") is None:
                    raise ValueError("data.assets.root is required when data.assets.enable=true.")
                if assets_cfg_raw.get("use_prebuilt_assets") is None:
                    raise ValueError(
                        "data.assets.use_prebuilt_assets is required when data.assets.enable=true."
                    )
                if assets_cfg_raw.get("missing_policy") is None:
                    raise ValueError(
                        "data.assets.missing_policy is required when data.assets.enable=true."
                    )
                missing_policy = normalize_missing_policy(assets_cfg_raw.get("missing_policy"))
                self.asset_config = AssetConfig(
                    enable=True,
                    root=str(assets_cfg_raw.get("root")),
                    use_prebuilt_assets=bool(assets_cfg_raw.get("use_prebuilt_assets")),
                    missing_policy=missing_policy,
                )
                self.asset_store = StreetForwardAssetStore(
                    self.asset_config.root,
                    missing_policy=self.asset_config.missing_policy,
                )
                self.use_prebuilt_assets = bool(self.asset_config.use_prebuilt_assets)
                self.asset_missing_policy = str(self.asset_config.missing_policy)
                if self.use_prebuilt_assets and self.asset_missing_policy == "error":
                    logger.info(
                        "StreetForward assets enabled in strict mode: root=%s, missing_policy=%s",
                        self.asset_config.root,
                        self.asset_missing_policy,
                    )
        
        # Track if initialized
        self._initialized = False

    def _normalize_sky_mask(self, sky_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Map loader ``sky_masks`` to canonical **1=sky, 0=non-sky** (see ``sky_mask_semantics``)."""
        if sky_mask is None:
            return None
        if self._sky_mask_loader_semantics is None:
            raise ValueError(
                "Sky mask tensor is present but data.sky_mask_semantics is not configured. "
                "Set pixel_source.load_sky_mask: true and data.sky_mask_semantics."
            )
        return normalize_sky_mask_to_one_is_sky(sky_mask, self._sky_mask_loader_semantics)
    
    def _create_pointcloud_generator(
        self,
        pointcloud_config: Dict,
        data_cfg: OmegaConf,
        device: torch.device,
    ) -> Optional["RGBPointCloudGenerator"]:
        """
        根据配置创建点云生成器。
        
        Args:
            pointcloud_config: 点云生成器配置字典
            data_cfg: 数据集配置（用于获取相机列表等）
            device: 设备
            
        Returns:
            点云生成器实例或 None
        """
        from datasets.pointcloud_generators import (
            MonocularRGBPointCloudGenerator,
            HybridRGBPointCloudGenerator,
            LiDARRGBPointCloudGenerator,
        )

        # Generator type must be explicit (fail-fast)
        if "type" not in pointcloud_config:
            raise ValueError("dataset.pointcloud.type is required (monocular|lidar|hybrid).")
        generator_type = pointcloud_config["type"]

        if generator_type == "monocular":
            if "chosen_cam_ids" not in pointcloud_config:
                raise ValueError("dataset.pointcloud.chosen_cam_ids is required for monocular generator.")
            chosen_cam_ids = pointcloud_config["chosen_cam_ids"]
            if "dynamic_filter" not in pointcloud_config:
                raise ValueError("dataset.pointcloud.dynamic_filter is required for monocular generator.")
            if not bool(pointcloud_config["dynamic_filter"]):
                raise ValueError(
                    "Monocular pointcloud requires dynamic_filter=true (no bbox fallback)."
                )
            dyn_rec_enable, dyn_rec_bbox_expand, dyn_rec_max_pts, dyn_rec_assignment = (
                _parse_monocular_dynamic_recovery_cfg(pointcloud_config)
            )

            # Require pixel_source to load dynamic masks (fast-fail)
            if (
                getattr(data_cfg, "pixel_source", None) is None
                or not bool(data_cfg.pixel_source.get("load_dynamic_mask", False))
            ):
                raise ValueError(
                    "Monocular pointcloud requires data.pixel_source.load_dynamic_mask: true."
                )

            return MonocularRGBPointCloudGenerator(
                chosen_cam_ids=chosen_cam_ids,
                sparsity=pointcloud_config["sparsity"],
                filter_sky=pointcloud_config["filter_sky"],
                depth_consistency=pointcloud_config["depth_consistency"],
                downscale=pointcloud_config["downscale"],
                dynamic_filter=pointcloud_config["dynamic_filter"],
                dynamic_recovery_enable=dyn_rec_enable,
                dynamic_recovery_bbox_expand_xyz_m=dyn_rec_bbox_expand,
                dynamic_recovery_max_points_per_instance=dyn_rec_max_pts,
                dynamic_recovery_assignment=dyn_rec_assignment,
                device=device,
            )
        elif generator_type == "lidar":
            if "lidar_sparsity" not in pointcloud_config:
                raise ValueError("dataset.pointcloud.lidar_sparsity is required for lidar generator.")
            lidar_sparsity = pointcloud_config["lidar_sparsity"]
            sim_enable, sim_thresh = _parse_static_instance_motion_cfg(pointcloud_config)
            return LiDARRGBPointCloudGenerator(
                sparsity=lidar_sparsity,
                device=device,
                static_instance_motion_enable=sim_enable,
                static_instance_motion_traj_length_thresh_m=sim_thresh,
            )
        elif generator_type == "hybrid":
            # LiDAR生成器参数
            if "lidar_sparsity" not in pointcloud_config:
                raise ValueError("dataset.pointcloud.lidar_sparsity is required for hybrid generator.")
            lidar_sparsity = pointcloud_config["lidar_sparsity"]
            
            # 单目生成器参数（最小必需集合）
            required_mono = [
                "monocular_chosen_cam_ids",
                "monocular_sparsity",
                "monocular_filter_sky",
                "monocular_depth_consistency",
                "monocular_downscale",
                "monocular_dynamic_recovery_bbox_expand_xyz_m",
                "monocular_dynamic_recovery_max_points_per_instance",
            ]
            missing = [k for k in required_mono if k not in pointcloud_config]
            if missing:
                raise ValueError(
                    f"Hybrid pointcloud missing required monocular config keys: {missing}"
                )
            monocular_chosen_cam_ids = pointcloud_config["monocular_chosen_cam_ids"]
            monocular_sparsity = pointcloud_config["monocular_sparsity"]
            monocular_filter_sky = pointcloud_config["monocular_filter_sky"]
            monocular_depth_consistency = pointcloud_config["monocular_depth_consistency"]
            monocular_downscale = pointcloud_config["monocular_downscale"]
            if (
                getattr(data_cfg, "pixel_source", None) is None
                or not bool(data_cfg.pixel_source.get("load_dynamic_mask", False))
            ):
                raise ValueError(
                    "Hybrid pointcloud requires data.pixel_source.load_dynamic_mask: true "
                    "(monocular dynamic filtering)."
                )

            near_max_points = pointcloud_config.get("near_max_points", None)
            distant_max_points = pointcloud_config.get("distant_max_points", None)
            sim_enable, sim_thresh = _parse_static_instance_motion_cfg(pointcloud_config)
            mono_rec_bbox_expand = pointcloud_config["monocular_dynamic_recovery_bbox_expand_xyz_m"]
            mono_rec_max_pts = pointcloud_config["monocular_dynamic_recovery_max_points_per_instance"]

            mono_rec_bbox_expand = list(mono_rec_bbox_expand)
            if len(mono_rec_bbox_expand) != 3:
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_recovery_bbox_expand_xyz_m must have "
                    "3 values [dx, dy, dz]."
                )
            mono_rec_bbox_expand = [float(x) for x in mono_rec_bbox_expand]
            if any(x < 0.0 for x in mono_rec_bbox_expand):
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_recovery_bbox_expand_xyz_m must be non-negative."
                )
            mono_rec_max_pts = int(mono_rec_max_pts)
            if mono_rec_max_pts <= 0:
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_recovery_max_points_per_instance must be > 0."
                )
            if "monocular_dynamic_recovery_enable" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_recovery_enable is removed; "
                    "hybrid now always enables monocular dynamic recovery."
                )
            if "monocular_dynamic_recovery_assignment" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_recovery_assignment is removed; "
                    "hybrid now uses first_hit."
                )
            if "monocular_dynamic_filter" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.monocular_dynamic_filter is removed; "
                    "hybrid always enables monocular dynamic filtering."
                )
            if "fusion_strategy" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.fusion_strategy is removed; hybrid always uses merge."
                )
            if "dynamic_source" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.dynamic_source is removed; hybrid always uses fused dynamic points."
                )
            if "downsample_dynamic" in pointcloud_config:
                raise ValueError(
                    "dataset.pointcloud.downsample_dynamic is removed; hybrid no longer supports this config."
                )

            return HybridRGBPointCloudGenerator(
                lidar_sparsity=lidar_sparsity,
                monocular_chosen_cam_ids=monocular_chosen_cam_ids,
                monocular_sparsity=monocular_sparsity,
                monocular_filter_sky=monocular_filter_sky,
                monocular_depth_consistency=monocular_depth_consistency,
                monocular_downscale=monocular_downscale,
                monocular_dynamic_recovery_bbox_expand_xyz_m=mono_rec_bbox_expand,
                monocular_dynamic_recovery_max_points_per_instance=mono_rec_max_pts,
                near_max_points=near_max_points,
                distant_max_points=distant_max_points,
                static_instance_motion_enable=sim_enable,
                static_instance_motion_traj_length_thresh_m=sim_thresh,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown dataset.pointcloud.type: {generator_type!r} (expected monocular|lidar|hybrid)."
            )
    
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
        
        # Ensure training queue has enough scenes (with lock)
        with self._lock:
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
        
        Note: This method should be called with self._lock held.
        
        Important: Even if queue is full, we should continue adding scenes if
        current_scene_index is approaching the end of the queue, to ensure all
        scenes can be processed.
        """
        target_queue_size = self.preload_scene_count + 1
        
        # Calculate how many scenes ahead of current index we need
        # We want at least target_queue_size scenes ahead of current index
        scenes_ahead_needed = target_queue_size
        scenes_ahead = len(self.scene_training_queue) - self.current_scene_index
        
        # If we have enough scenes ahead, return early
        if scenes_ahead >= scenes_ahead_needed:
            return
        
        # Try to fill queue from candidate pool
        while scenes_ahead < scenes_ahead_needed and len(self.scene_candidate_pool) > 0:
            scene_id = self.scene_candidate_pool.pop(0)
            if self._validate_and_add_to_queue(scene_id):
                logger.debug(f"Scene {scene_id} validated and added to training queue")
                scenes_ahead = len(self.scene_training_queue) - self.current_scene_index
            else:
                logger.debug(f"Scene {scene_id} is not suitable, skipping")
        
        # If candidate pool is empty and we still need more scenes, try to refill from original IDs
        if scenes_ahead < scenes_ahead_needed and len(self.scene_candidate_pool) == 0:
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
                while scenes_ahead < scenes_ahead_needed and len(self.scene_candidate_pool) > 0:
                    scene_id = self.scene_candidate_pool.pop(0)
                    if self._validate_and_add_to_queue(scene_id):
                        logger.debug(f"Scene {scene_id} validated and added to training queue")
                        scenes_ahead = len(self.scene_training_queue) - self.current_scene_index
                    else:
                        logger.debug(f"Scene {scene_id} is not suitable, skipping")
    
    def _validate_and_add_to_queue(self, scene_id: int) -> bool:
        """
        Validate a scene and add it to training queue if suitable.
        
        This method performs a lightweight validation by loading the scene
        and checking if it's suitable. If suitable, adds to queue.
        
        Note: This method should be called with self._lock held.
        However, it releases the lock during I/O operations to avoid blocking.
        
        Args:
            scene_id: Scene ID to validate
            
        Returns:
            bool: True if scene is suitable and added to queue, False otherwise
        """
        # Skip if already in queue or invalid (check with lock held)
        if scene_id in self.scene_training_queue:
            return True
        if scene_id in self.invalid_scene_ids:
            return False
        
        # Release lock before I/O operation to avoid blocking other threads
        # _load_and_prepare_scene is a long I/O operation
        # For RLock, we need to safely release and re-acquire
        # Use try/except to handle the case where lock might not be held
        lock_released = False
        try:
            self._lock.release()
            lock_released = True
        except RuntimeError:
            # Lock was not held, continue without releasing
            pass
        
        try:
            scene_data = self._load_and_prepare_scene(scene_id)
        finally:
            # Re-acquire lock if we released it
            if lock_released:
                self._lock.acquire()
        
        if scene_data is not None:
            # Scene is suitable, add to queue
            # Double-check queue state (might have changed while lock was released)
            if scene_id not in self.scene_training_queue:
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
                # Get trajectory to check keyframes (use training frames only)
                pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
                try:
                    test_image_stride = pixel_source_cfg.get("test_image_stride", 0)
                except Exception:
                    test_image_stride = 0
                train_frame_indices, _ = self._split_train_test_frames(
                    num_frames=temp_dataset.num_img_timesteps,
                    test_image_stride=test_image_stride,
                )
                trajectory_full = self._get_scene_trajectory(temp_dataset)
                trajectory = trajectory_full[train_frame_indices]
                keyframe_segments, _ = self._split_keyframes(trajectory)
                keyframe_segments = [
                    [train_frame_indices[idx] for idx in seg]
                    for seg in keyframe_segments
                ]
                
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
        
        Note: This method should be called with self._lock held.
        
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
            # Drop segment-level pointcloud cache for this scene to free memory.
            stale_keys = [k for k in self._segment_pointcloud_cache.keys() if k[0] == scene_id]
            for k in stale_keys:
                del self._segment_pointcloud_cache[k]
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
        
        Note: This method should be called with self._lock held for cache operations.
        However, _load_and_prepare_scene is a long I/O operation and may be called
        without holding the lock (it will be called in background thread).
        
        Args:
            scene_id: Scene ID to ensure loaded
            
        Returns:
            Scene data dictionary or None if scene cannot be loaded
        """
        with self._lock:
            # Check if already in cache
            if scene_id in self.train_scenes_cache:
                return self.train_scenes_cache[scene_id]
            
            # Check if it's an evaluation scene
            if scene_id in self.eval_scene_ids:
                if scene_id not in self.eval_scenes:
                    # Load evaluation scene (release lock during I/O)
                    pass  # Will load below
                else:
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
        
        # Load the scene (outside lock, as this is a long I/O operation)
        if scene_id in self.eval_scene_ids and scene_id not in self.eval_scenes:
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                with self._lock:
                    self.eval_scenes[scene_id] = scene_data
                return scene_data
            else:
                return None
        else:
            # Training scene
            scene_data = self._load_and_prepare_scene(scene_id)
            if scene_data is not None:
                with self._lock:
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
        with self._lock:
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
        with self._lock:
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
    
    def get_segment_frames(
        self,
        scene_id: int,
        segment_id: int,
    ) -> List[int]:
        """
        获取段内所有帧索引。
        
        Args:
            scene_id: 场景ID
            segment_id: 段ID（场景内索引）
            
        Returns:
            frame_indices: 段内所有帧索引列表（已排序、去重）
        """
        scene_data = self.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        if segment_id >= len(scene_data['segments']):
            raise ValueError(f"Segment {segment_id} not found in scene {scene_id}")
        
        segment = scene_data['segments'][segment_id]
        frame_indices = segment['frame_indices']
        
        # 确保已排序、去重
        frame_indices = sorted(list(set(frame_indices)))
        
        return frame_indices
    
    def get_frame_data(
        self,
        scene_id: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Dict:
        """
        获取指定帧和相机的数据。
        
        Args:
            scene_id: 场景ID
            frame_idx: 帧索引
            cam_idx: 相机索引（在 camera_list 中的索引）
            
        Returns:
            Dict包含：
                - 'image': Tensor [H, W, 3] - RGB图像
                - 'extrinsic': Tensor [4, 4] - 外参（cam_to_world）
                - 'intrinsic': Tensor [4, 4] - 内参（4x4矩阵）
                - 'depth': Tensor [H, W] - 深度图（如果可用）
        """
        scene_data = self.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        scene_dataset = scene_data['dataset']
        num_cams = scene_dataset.num_cams
        
        # 验证相机索引
        if cam_idx >= num_cams:
            raise ValueError(f"Camera index {cam_idx} out of range (num_cams={num_cams})")
        
        # 计算图像索引
        img_idx = frame_idx * num_cams + cam_idx
        
        # 获取图像和相机信息
        try:
            image_infos, cam_infos = scene_dataset.pixel_source.get_image(img_idx)
        except Exception as e:
            raise ValueError(f"Failed to load image {img_idx}: {e}")
        
        # 获取深度图
        depth = self._get_depth(scene_dataset, frame_idx, cam_idx)
        if depth is None:
            # 创建占位符深度图
            H, W = image_infos['pixels'].shape[:2]
            depth = torch.ones(H, W, dtype=torch.float32, device=self.device) * 10.0
        
        # 转换内参为4x4（如果原本是3x3）
        intrinsic = cam_infos['intrinsics']  # [3, 3] or [4, 4]
        intrinsic_4x4 = self._convert_intrinsic_to_4x4(intrinsic)
        
        # 获取天空掩码（如果存在）；batch 内统一为 1=sky, 0=non-sky
        sky_mask = None
        if 'sky_masks' in image_infos and image_infos['sky_masks'] is not None:
            sky_mask = self._normalize_sky_mask(image_infos['sky_masks'])
        
        return {
            'image': image_infos['pixels'],  # [H, W, 3]
            'extrinsic': cam_infos['camera_to_world'],  # [4, 4]
            'intrinsic': intrinsic_4x4,  # [4, 4]
            'depth': depth,  # [H, W]
            'sky_mask': sky_mask,  # Tensor [H, W] or None; canonical 1=sky
        }

    def _split_train_test_frames(
        self,
        num_frames: int,
        test_image_stride: int,
    ) -> Tuple[List[int], List[int]]:
        """
        根据 test_image_stride 抽帧，分离训练帧和测试帧。
        
        Args:
            num_frames: 场景总帧数
            test_image_stride: 测试帧步长（0表示所有帧用于训练和测试）
            
        Returns:
            train_frame_indices: 训练帧索引列表
            test_frame_indices: 测试帧索引列表
        """
        if test_image_stride == 0:
            train_frame_indices = list(range(num_frames))
            test_frame_indices = list(range(num_frames))
        else:
            test_frame_indices = list(range(
                test_image_stride,
                num_frames,
                test_image_stride,
            ))
            train_frame_indices = [
                i for i in range(num_frames)
                if i not in test_frame_indices
            ]
        
        return train_frame_indices, test_frame_indices
    
    def _load_scene(self, scene_id: int) -> Optional[Dict]:
        """
        Load a single scene's data.
        
        Process:
        1. Create DrivingDataset instance
        2. Split train/test frames before keyframe splitting
        3. Get scene trajectory (for keyframe splitting) using training frames
        4. Split keyframes
        5. Split segments (based on AABB constraints)
        6. Return scene information
        """
        # 1. Create scene configuration
        scene_cfg = OmegaConf.create(OmegaConf.to_container(self.data_cfg))
        scene_cfg.scene_idx = scene_id
        
        try:
            # 2. Create DrivingDataset instance
            scene_dataset = DrivingDataset(scene_cfg)
            
            # 3. Split train/test frames before keyframe splitting
            pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
            try:
                test_image_stride = pixel_source_cfg.get("test_image_stride", 0)
            except Exception:
                test_image_stride = 0
            train_frame_indices, test_frame_indices = self._split_train_test_frames(
                num_frames=scene_dataset.num_img_timesteps,
                test_image_stride=test_image_stride,
            )
            
            # 4. Get scene trajectory (using front camera trajectory) and filter training frames
            full_trajectory = self._get_scene_trajectory(scene_dataset)
            trajectory = full_trajectory[train_frame_indices]
            
            # 5. Split keyframes
            keyframe_segments, keyframe_ranges = self._split_keyframes(trajectory)
            # Map keyframe frame indices back to global frame indices (training frames)
            keyframe_segments = [
                [train_frame_indices[idx] for idx in seg]
                for seg in keyframe_segments
            ]
            
            # 6. Check if scene is suitable for training (sufficient keyframes)
            if not self._is_scene_suitable(keyframe_segments):
                logger.warning(
                    "Scene %s is not suitable for training (insufficient keyframe segments: "
                    "got %s segments from split_trajectory, need >= %s; train_frames=%s). skipping...",
                    scene_id,
                    len(keyframe_segments),
                    self.min_keyframes_per_scene,
                    len(train_frame_indices),
                )
                return None  # Return None to indicate scene is not suitable
            
            # 7. Split segments (based on AABB constraints and keyframe distances)
            segments = self._split_segments(
                scene_dataset=scene_dataset,
                keyframe_segments=keyframe_segments,
                keyframe_ranges=keyframe_ranges,
                train_frame_indices=train_frame_indices,
                test_frame_indices=test_frame_indices,
                overlap_ratio=self.segment_overlap_ratio,
            )
            
            if len(segments) == 0:
                logger.warning(f"Scene {scene_id} has no valid segments after filtering, skipping...")
                return None
            
            return {
                'dataset': scene_dataset,
                'trajectory': trajectory,
                'train_frame_indices': train_frame_indices,
                'test_frame_indices': test_frame_indices,
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
    
    def _compute_segment_aabb(
        self,
        scene_dataset: DrivingDataset,
        frame_indices: List[int],
    ) -> Tensor:
        """
        计算段的AABB边界。
        
        使用段内帧的lidar数据，参考 lidar_source.get_aabb 的方式计算。
        
        Args:
            scene_dataset: 场景数据集实例
            frame_indices: 段内帧索引列表
            
        Returns:
            aabb: Tensor[2, 3] - 段的AABB边界 [min, max]
        """
        # 检查lidar_source是否存在
        try:
            lidar_source = scene_dataset.lidar_source
        except AttributeError:
            # lidar_source属性不存在
            logger.warning("Lidar source not available, falling back to scene AABB")
            return scene_dataset.get_aabb()
        
        if lidar_source is None:
            logger.warning("Lidar source not available, falling back to scene AABB")
            return scene_dataset.get_aabb()
        
        # 检查lidar数据是否已加载
        # 注意：需要检查是否是torch.Tensor，因为Mock对象的属性可能不是None但不是Tensor
        try:
            has_origins = isinstance(lidar_source.origins, torch.Tensor)
            has_directions = isinstance(lidar_source.directions, torch.Tensor)
            has_ranges = isinstance(lidar_source.ranges, torch.Tensor)
            has_timesteps = isinstance(lidar_source.timesteps, torch.Tensor)
        except (AttributeError, TypeError):
            # 如果访问属性失败或类型检查失败，fallback到scene AABB
            logger.warning("Lidar source data not properly loaded, falling back to scene AABB")
            return scene_dataset.get_aabb()
        
        if not (has_origins and has_directions and has_ranges and has_timesteps):
            logger.warning("Lidar points not loaded, falling back to scene AABB")
            return scene_dataset.get_aabb()
        
        # 将frame_indices转换为tensor以便比较
        # 处理timesteps可能是Mock对象的情况
        try:
            # 尝试获取真实的dtype和device
            if isinstance(lidar_source.timesteps, torch.Tensor):
                timesteps_dtype = lidar_source.timesteps.dtype
                timesteps_device = lidar_source.timesteps.device
            else:
                # 如果是Mock对象或其他类型，使用默认值
                timesteps_dtype = torch.long
                if isinstance(lidar_source.origins, torch.Tensor):
                    timesteps_device = lidar_source.origins.device
                else:
                    timesteps_device = torch.device('cpu')
        except (AttributeError, TypeError):
            # 如果访问失败，使用默认值
            timesteps_dtype = torch.long
            try:
                if isinstance(lidar_source.origins, torch.Tensor):
                    timesteps_device = lidar_source.origins.device
                else:
                    timesteps_device = torch.device('cpu')
            except (AttributeError, TypeError):
                timesteps_device = torch.device('cpu')
        
        frame_indices_tensor = torch.tensor(frame_indices, dtype=timesteps_dtype, device=timesteps_device)
        
        # 筛选段内帧的lidar点
        # timesteps 中的值应该对应 frame_indices
        mask = torch.isin(lidar_source.timesteps, frame_indices_tensor)
        
        if not mask.any():
            logger.warning(f"No lidar points found for frame indices {frame_indices}, falling back to scene AABB")
            return scene_dataset.get_aabb()
        
        # 获取段内帧的lidar点
        segment_origins = lidar_source.origins[mask]
        segment_directions = lidar_source.directions[mask]
        segment_ranges = lidar_source.ranges[mask]
        
        # 处理ranges的形状：可能是[N]或[N, 1]
        if segment_ranges.dim() == 1:
            segment_ranges = segment_ranges.unsqueeze(-1)  # [N] -> [N, 1]
        
        # 计算lidar点的3D坐标
        lidar_pts = segment_origins + segment_directions * segment_ranges
        
        # 下采样lidar点
        downsample_factor = lidar_source.data_cfg.get('lidar_downsample_factor', 4)
        if downsample_factor > 1 and len(lidar_pts) > downsample_factor:
            lidar_pts = lidar_pts[
                torch.randperm(len(lidar_pts))[
                    : int(len(lidar_pts) / downsample_factor)
                ]
            ]
        
        # 计算实际的min/max（需要在删除lidar_pts之前计算）
        actual_min = lidar_pts.min(dim=0)[0]
        actual_max = lidar_pts.max(dim=0)[0]
        
        # 使用分位数计算AABB（去除异常值）
        percentile = lidar_source.data_cfg.get('lidar_percentile', 0.02)
        aabb_min = torch.quantile(lidar_pts, percentile, dim=0)
        aabb_max = torch.quantile(lidar_pts, 1 - percentile, dim=0)
        
        # 确保AABB包含所有点（扩展边界以确保包含分位数外的点）
        # 这很重要，因为测试和实际使用都期望AABB包含所有点
        # 使用更宽松的边界：取分位数和实际最小/最大值的组合
        aabb_min = torch.minimum(aabb_min, actual_min)
        aabb_max = torch.maximum(aabb_max, actual_max)
        
        # 清理临时变量
        del lidar_pts
        
        # 通常lidar的高度非常小，所以稍微增加AABB的高度
        if aabb_max[-1] < 20:
            aabb_max[-1] = 20.0
        
        # 组合为 [min, max] 格式
        aabb = torch.stack([aabb_min, aabb_max], dim=0)  # [2, 3]
        
        logger.debug(f"[Segment] Computed AABB from {len(frame_indices)} frames: {aabb}")
        
        return aabb
    
    def _split_segments(
        self,
        scene_dataset: DrivingDataset,
        keyframe_segments: List[List[int]],
        keyframe_ranges: Tensor,  # [num_keyframes, 2] - Distance ranges for each keyframe segment
        overlap_ratio: float,
        train_frame_indices: Optional[List[int]] = None,
        test_frame_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Split scene into segments based on trajectory distance and a reference length.
        
        Strategy:
        1. Get reference AABB length (from pointcloud crop_aabb)
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
            train_frame_indices: 训练帧索引（可选，仅用于记录）
            test_frame_indices: 测试帧索引列表（可选，用于段内测试帧记录）
        
        Returns:
            segments: List[Dict] - Each segment contains:
                - 'segment_id': int - Segment ID
                - 'keyframe_indices': List[int] - Keyframe indices in this segment (global keyframe indices)
                - 'frame_indices': List[int] - All frame indices in this segment (deduplicated)
                - 'test_frame_indices': List[int] - Test frames that fall into this segment's frame range
        """
        # 1. Reference AABB length from dataset.segment_aabb (seg0-aligned; same as training)
        aabb_length = float(np.max(self.segment_aabb_np[1].astype(np.float64) - self.segment_aabb_np[0].astype(np.float64)))
        
        # 2. Calculate total keyframe distance
        # Each row of keyframe_ranges is [start_distance, end_distance]
        # Calculate length of each keyframe segment
        keyframe_lengths = keyframe_ranges[:, 1] - keyframe_ranges[:, 0]  # [num_keyframes]
        total_keyframe_distance = keyframe_lengths.sum().item()  # Total distance of all keyframe segments
        
        # 3. Determine number of segments based on distance and AABB length
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
        
        # 4. Group keyframes into segments based on distance with overlap
        segments = []
        segment_id = 0
        
        if num_segments == 1:
            # Only one segment, include all keyframes
            all_frames = []
            for kf_seg in keyframe_segments:
                all_frames.extend(kf_seg)
            
            frame_indices = sorted(list(set(all_frames)))
            segments.append({
                'segment_id': segment_id,
                'keyframe_indices': list(range(len(keyframe_segments))),
                'frame_indices': frame_indices,
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
                    frame_indices = sorted(list(current_segment_frames))
                    segments.append({
                        'segment_id': segment_id,
                        'keyframe_indices': current_segment_kf_indices,
                        'frame_indices': frame_indices,
                    })
                    segment_id += 1
        
        # 5. Filter out segments with insufficient keyframes (double check)
        valid_segments = [
            seg for seg in segments
            if len(seg['keyframe_indices']) >= self.min_keyframes_per_segment
        ]

        # 6. Record test frames that fall into each segment's frame range
        test_frame_indices = test_frame_indices or []
        for seg in valid_segments:
            segment_train_frames = seg.get('frame_indices', [])
            if len(segment_train_frames) == 0:
                seg['test_frame_indices'] = []
                continue
            
            segment_min_frame = min(segment_train_frames)
            segment_max_frame = max(segment_train_frames)
            segment_test_frames = [
                idx for idx in test_frame_indices
                if segment_min_frame <= idx <= segment_max_frame
            ]
            seg['test_frame_indices'] = segment_test_frames
        
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
    
    def _build_dynamic_info(
        self,
        scene_dataset: DrivingDataset,
        frame_indices: List[int],
        instance_mapping: Optional[Dict[int, int]] = None,
        world_to_seg0: Optional[Tensor] = None,
        exclude_instance_intids: Optional[Set[int]] = None,
    ) -> Optional[Dict]:
        """
        从 scene_dataset 的 instances_pose 构建 dynamic_info。
        
        Args:
            scene_dataset: 场景数据集实例
            frame_indices: 需要构建 dynamic_info 的帧索引列表
            instance_mapping: 可选的实例ID映射，格式为 {original_id: intid}，用于将原始的 instance_id 
                            转换为点云中使用的 intid。如果提供，dynamic_info 中的 instance_id 将使用 intid。
            world_to_seg0: 可选的世界坐标 -> segment 第一帧坐标的变换矩阵。
            exclude_instance_intids: 若设置，这些 intid 不写入 dynamic_info（与静止实例留在 background 一致）。
            
        Returns:
            dynamic_info: Dict[int, Dict] 格式，{frame_idx: {"instances": {instance_id: {"quat": ..., "trans": ...}}}}
            如果 instances_pose 不存在，返回 None
            注意：如果提供了 instance_mapping，返回的 instance_id 将是 intid（与点云中的 instance_id 一致）
        """
        pixel_source = scene_dataset.pixel_source
        if pixel_source is None or pixel_source.instances_pose is None:
            return None
        
        instances_pose = pixel_source.instances_pose  # [num_frames, num_instances, 4, 4]
        per_frame_mask = getattr(pixel_source, "per_frame_instance_mask", None)
        instances_true_id = getattr(pixel_source, "instances_true_id", None)
        
        if not isinstance(instances_pose, torch.Tensor):
            instances_pose = torch.as_tensor(instances_pose, device=self.device)
        if per_frame_mask is not None and not isinstance(per_frame_mask, torch.Tensor):
            per_frame_mask = torch.as_tensor(per_frame_mask, device=instances_pose.device)
        if per_frame_mask is not None:
            per_frame_mask = per_frame_mask.to(device=instances_pose.device, dtype=torch.bool)
        if instances_true_id is not None and not isinstance(instances_true_id, torch.Tensor):
            instances_true_id = torch.as_tensor(instances_true_id, device=self.device)
        if instance_mapping is not None and instances_true_id is None:
            raise ValueError(
                "instance_mapping is provided but pixel_source.instances_true_id is missing; "
                "cannot align dynamic instances without original IDs."
            )
        
        # 检查 frame_indices 是否在有效范围内
        num_frames = instances_pose.shape[0]
        valid_frame_indices = [fidx for fidx in frame_indices if 0 <= fidx < num_frames]
        
        if len(valid_frame_indices) == 0:
            return None
        
        dynamic_info = {}
        num_instances = instances_pose.shape[1]
        
        # 如果没有提供 instance_mapping，尝试从 instances_true_id 构建
        if instance_mapping is None and instances_true_id is not None:
            instances_true_id_np = instances_true_id.cpu().numpy() if isinstance(instances_true_id, torch.Tensor) else instances_true_id
            instance_mapping = {int(instances_true_id_np[i]): int(i) for i in range(num_instances)}
        
        for frame_idx in valid_frame_indices:
            frame_instances = {}
            
            if per_frame_mask is not None:
                if frame_idx >= per_frame_mask.shape[0]:
                    visible_instance_ids = []
                else:
                    visible_instance_ids = torch.nonzero(per_frame_mask[frame_idx], as_tuple=False).view(-1).tolist()
            else:
                visible_instance_ids = list(range(num_instances))
            world_to_seg0_local = (
                world_to_seg0.to(device=instances_pose.device, dtype=instances_pose.dtype)
                if world_to_seg0 is not None
                else None
            )

            for instance_id in visible_instance_ids:
                pose_matrix = instances_pose[frame_idx, instance_id]  # [4, 4]
                if world_to_seg0_local is not None:
                    pose_matrix = world_to_seg0_local @ pose_matrix
                
                # 提取旋转矩阵 [3, 3] 和平移向量 [3]
                rot_matrix = pose_matrix[:3, :3]  # [3, 3]
                trans = pose_matrix[:3, 3]  # [3]
                
                # 将旋转矩阵转换为四元数 (wxyz 格式)
                # 使用 Shepperd's method (更稳定的方法)
                trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
                
                if trace > 0:
                    s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
                    w = 0.25 * s
                    x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                    y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                    z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                    s = torch.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
                    w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                    x = 0.25 * s
                    y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                    z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                    s = torch.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
                    w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                    x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                    y = 0.25 * s
                    z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
                    w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                    x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                    y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                    z = 0.25 * s
                
                quat = torch.stack([w, x, y, z])  # [4] wxyz format
                
                # 转换为 numpy 或 Python list (确保可以序列化)
                if isinstance(quat, torch.Tensor):
                    quat = quat.cpu().numpy().tolist()
                if isinstance(trans, torch.Tensor):
                    trans = trans.cpu().numpy().tolist()
                
                # 确定要使用的 instance_id
                # 点云中的 dynamic 字典的 key 是 instances_pose 的索引（intid）
                # 所以 dynamic_info 中的 instance_id 也应该使用 instances_pose 的索引
                # 如果提供了 instance_mapping，我们需要确保使用正确的 intid
                if instance_mapping is not None and instances_true_id is not None:
                    # 获取原始ID
                    original_id = int(instances_true_id[instance_id].item() if isinstance(instances_true_id, torch.Tensor) else instances_true_id[instance_id])
                    # 映射到 intid（点云中使用的 instance_id）
                    # instance_mapping 格式: {original_id: intid}，其中 intid 是 instances_pose 的索引
                    intid = instance_mapping.get(original_id)
                    if intid is None:
                        available_keys = sorted(instance_mapping.keys())
                        raise ValueError(
                            f"Instance mapping missing for original_id={original_id} (instance_index={instance_id}). "
                            f"Available mapping keys: {available_keys}"
                        )
                    final_instance_id = int(intid)
                else:
                    # 如果没有映射，直接使用 instance_id（instances_pose 的索引）
                    # 这与点云中的 dynamic 字典的 key 一致
                    final_instance_id = int(instance_id)
                
                if exclude_instance_intids is not None and final_instance_id in exclude_instance_intids:
                    continue

                frame_instances[final_instance_id] = {
                    "quat": quat,
                    "trans": trans,
                }
            
            dynamic_info[int(frame_idx)] = {
                "instances": frame_instances,
            }
        
        if not dynamic_info:
            return None
        nonempty = any(
            len(finfo.get("instances", {})) > 0 for finfo in dynamic_info.values()
        )
        if not nonempty:
            return None
        return dynamic_info

    def _to_4x4_tensor(self, mat) -> Tensor:
        """
        Convert various matrix formats to a 4x4 float tensor on self.device.
        """
        pose = torch.as_tensor(mat, dtype=torch.float32, device=self.device)
        if pose.shape == (4, 4):
            return pose
        if pose.shape == (3, 4):
            bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            return torch.cat([pose, bottom], dim=0)
        if pose.shape[-2:] == (4, 4) and pose.ndim > 2:
            return pose[..., :4, :4].to(device=self.device, dtype=torch.float32)
        raise ValueError(f"Expected pose shape (4,4) or (3,4), got {pose.shape}")

    def _get_pose_from_lidar(self, scene_dataset: DrivingDataset, frame_idx: int) -> Optional[Tensor]:
        lidar_source = getattr(scene_dataset, "lidar_source", None)
        if lidar_source is None:
            return None
        lidar_to_worlds = getattr(lidar_source, "lidar_to_worlds", None)
        if lidar_to_worlds is None:
            return None
        try:
            if frame_idx >= len(lidar_to_worlds):
                return None
        except TypeError:
            return None
        pose = lidar_to_worlds[frame_idx]
        try:
            return self._to_4x4_tensor(pose)
        except Exception:
            return None

    def _get_pose_from_camera(self, scene_dataset: DrivingDataset, frame_idx: int) -> Optional[Tensor]:
        pixel_source = getattr(scene_dataset, "pixel_source", None)
        if pixel_source is None or not getattr(pixel_source, "camera_list", None):
            return None
        ref_cam_id = pixel_source.camera_list[0]
        # Enforce using front camera with id 0 as the only valid seg0 source.
        if ref_cam_id != 0:
            raise ValueError(
                f"MultiSceneDataset expects pixel_source.camera_list[0] == 0, "
                f"but got {ref_cam_id}. Please ensure front camera has id 0."
            )
        try:
            cam_data = pixel_source.camera_data[ref_cam_id]
        except Exception:
            return None
        cam_to_worlds = getattr(cam_data, "cam_to_worlds", None)
        if cam_to_worlds is None:
            cam_to_worlds = getattr(cam_data, "camera_to_worlds", None)
        if cam_to_worlds is None:
            return None
        try:
            if frame_idx >= len(cam_to_worlds):
                return None
        except TypeError:
            return None
        pose = cam_to_worlds[frame_idx]
        try:
            return self._to_4x4_tensor(pose)
        except Exception:
            return None

    def _get_segment_first_pose(
        self,
        scene_dataset: DrivingDataset,
        segment: Dict,
        segment_id: Optional[int] = None,
    ) -> Tuple[Tensor, int, str]:
        """
        Return (pose, frame_idx, source) where pose is segment-first-frame pose in world coords.
        Seg0 is strictly defined from reference camera-0 cam_to_worlds (front camera). Lidar is not allowed.
        """
        frame_indices = sorted(set(segment.get("frame_indices", [])))
        if len(frame_indices) == 0:
            raise ValueError("Segment has no frame_indices to compute first pose.")
        first_frame_idx = frame_indices[0]
        # Only allow camera-0 as seg0 source; fast-fail if unavailable.
        pose = self._get_pose_from_camera(scene_dataset, first_frame_idx)
        pose_source = "camera"
        if pose is None:
            seg_label = segment_id if segment_id is not None else segment.get("segment_id", "unknown")
            raise ValueError(
                f"Cannot find camera-0 pose for segment {seg_label} first frame {first_frame_idx}; "
                "seg0 must come from camera id 0."
            )
        return pose, first_frame_idx, pose_source

    def get_segment_first_pose(
        self,
        scene_id: int,
        segment_id: int,
    ) -> Tuple[Tensor, int, str]:
        """
        Public helper to fetch the segment-first pose for a given scene/segment.
        Returns (pose, frame_idx, source).
        """
        scene_data = self.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        segment = scene_data["segments"][segment_id]
        pose, frame_idx, source = self._get_segment_first_pose(
            scene_dataset=scene_data["dataset"],
            segment=segment,
            segment_id=segment_id,
        )
        return pose, frame_idx, source
    
    def get_segment_batch(
        self,
        scene_id: int,
        segment_id: int,
        include_test: bool = True,
    ) -> Dict:
        """
        Get training batch for specified scene and segment.
        
        Args:
            scene_id: 场景ID
            segment_id: 段ID
            include_test: 是否包含测试视角（如果可用）
        """
        # 1. Ensure scene is loaded
        scene_data = self._ensure_scene_loaded(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded or is not suitable")
        
        # 2. Get scene and segment information
        segment = scene_data['segments'][segment_id]
        scene_dataset = scene_data['dataset']
        
        # 2.1 Get segment first frame pose (world -> segment-0 transform)
        segment_first_pose, segment_first_frame_idx, segment_pose_source = self._get_segment_first_pose(
            scene_dataset=scene_dataset,
            segment=segment,
            segment_id=segment_id,
        )
        segment_first_pose = segment_first_pose.to(device=self.device, dtype=torch.float32)
        segment_first_pose = segment_first_pose.contiguous().clone()
        try:
            world_to_seg0 = torch.linalg.inv(segment_first_pose)
        except RuntimeError as exc:
            raise ValueError(
                f"Segment {segment_id} first pose is non-invertible; cannot build segment coordinate transform."
            ) from exc

        def _transform_extrinsics_list(extrinsics_list: List[Tensor]) -> List[Tensor]:
            transformed: List[Tensor] = []
            for ext in extrinsics_list:
                ext_tensor = self._to_4x4_tensor(ext).to(device=self.device, dtype=torch.float32)
                transformed.append(world_to_seg0 @ ext_tensor)
            return transformed
        
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
        source_sky_masks: List[Optional[Tensor]] = []
        has_source_sky_mask = False
        source_viewdirs_list: List[Optional[Tensor]] = []
        has_source_viewdirs = False
        source_egocar_masks: List[Optional[Tensor]] = []
        has_source_egocar_mask = False

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
                
                sky_mask = image_infos.get('sky_masks')
                if sky_mask is not None:
                    has_source_sky_mask = True
                source_sky_masks.append(self._normalize_sky_mask(sky_mask))
                
                viewdirs = image_infos.get('viewdirs')
                if viewdirs is not None:
                    has_source_viewdirs = True
                source_viewdirs_list.append(viewdirs)

                egocar_mask = image_infos.get("egocar_masks")
                if egocar_mask is not None:
                    has_source_egocar_mask = True
                source_egocar_masks.append(egocar_mask)
                
                source_frame_idxs.append(frame_idx)
                source_cam_idxs.append(cam_idx)
        
        # 5. Load target images (num_target_keyframes frames × num_cams cameras)
        target_images = []
        target_extrinsics = []
        target_intrinsics = []
        target_depths = []
        target_frame_idxs = []
        target_cam_idxs = []
        target_sky_masks: List[Optional[Tensor]] = []
        has_target_sky_mask = False
        target_viewdirs_list: List[Optional[Tensor]] = []
        has_target_viewdirs = False
        target_egocar_masks: List[Optional[Tensor]] = []
        has_target_egocar_mask = False

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
                
                sky_mask = image_infos.get('sky_masks')
                if sky_mask is not None:
                    has_target_sky_mask = True
                target_sky_masks.append(self._normalize_sky_mask(sky_mask))
                
                viewdirs = image_infos.get('viewdirs')
                if viewdirs is not None:
                    has_target_viewdirs = True
                target_viewdirs_list.append(viewdirs)

                egocar_mask = image_infos.get("egocar_masks")
                if egocar_mask is not None:
                    has_target_egocar_mask = True
                target_egocar_masks.append(egocar_mask)
                
                target_frame_idxs.append(frame_idx)
                target_cam_idxs.append(cam_idx)
        
        # 6. Get segment-level point cloud (cached by scene_id + segment_id)
        pointcloud = None
        if self.pointcloud_generator is not None:
            pc_key = (int(scene_id), int(segment_id))
            pointcloud = self._segment_pointcloud_cache.get(pc_key)
            if pointcloud is None:
                pointcloud = self.pointcloud_generator.generate_pointcloud(
                    dataset=self,
                    scene_id=scene_id,
                    segment_id=segment_id,
                    segment_first_pose=segment_first_pose,
                )
                self._segment_pointcloud_cache[pc_key] = pointcloud
        
        # 6.5. Build dynamic_info (if pointcloud contains dynamic objects)
        dynamic_info = None
        if pointcloud is not None and isinstance(pointcloud, dict) and "dynamic" in pointcloud:
            dynamic_pcd = pointcloud.get("dynamic")
            if isinstance(dynamic_pcd, dict) and len(dynamic_pcd) > 0 and pointcloud.get("instance_mapping") is None:
                raise ValueError(
                    "Dynamic pointcloud provided but instance_mapping is missing; "
                    "cannot build dynamic_info without mapping original IDs to pointcloud intids."
                )
            # 收集所有相关的 frame_idx
            all_frame_indices = set(source_frame_indices + target_frame_indices)
            if include_test:
                all_frame_indices.update(segment.get('test_frame_indices', []))
            
            # 从 pixel_source 获取动态物体信息
            if scene_dataset.pixel_source is not None and scene_dataset.pixel_source.instances_pose is not None:
                # 获取点云的 instance_mapping（如果存在），用于确保 dynamic_info 中的 instance_id 与点云中的一致
                instance_mapping = pointcloud.get("instance_mapping")
                exclude_instance_intids: Optional[Set[int]] = None
                meta = pointcloud.get("metadata") if isinstance(pointcloud, dict) else None
                if meta:
                    raw = meta.get("static_instance_intids")
                    if raw:
                        exclude_instance_intids = {int(x) for x in raw}
                dynamic_info = self._build_dynamic_info(
                    scene_dataset=scene_dataset,
                    frame_indices=list(all_frame_indices),
                    instance_mapping=instance_mapping,
                    world_to_seg0=world_to_seg0,
                    exclude_instance_intids=exclude_instance_intids,
                )

        # 7. Load test views if requested and available
        test_images: List[Tensor] = []
        test_extrinsics: List[Tensor] = []
        test_intrinsics: List[Tensor] = []
        test_depths: List[Tensor] = []
        test_frame_idxs: List[int] = []
        test_cam_idxs: List[int] = []
        test_sky_masks: List[Optional[Tensor]] = []
        has_test_sky_mask = False
        test_egocar_masks: List[Optional[Tensor]] = []
        has_test_egocar_mask = False
        
        if include_test:
            segment_test_frames = segment.get('test_frame_indices', [])
            if len(segment_test_frames) > 0:
                # Get max_test_images from config (if set)
                pixel_source_cfg = getattr(self.data_cfg, "pixel_source", {})
                try:
                    max_test_images = pixel_source_cfg.get("max_test_images", 0)
                except Exception:
                    max_test_images = 0
                
                # Randomly sample test frames if max_test_images is set and > 0
                if max_test_images > 0 and len(segment_test_frames) > max_test_images:
                    selected_test_frames = random.sample(segment_test_frames, max_test_images)
                else:
                    selected_test_frames = segment_test_frames
                
                # Load all cameras for selected test frames
                for frame_idx in selected_test_frames:
                    for cam_idx in range(num_cams):
                        frame_data = self.get_frame_data(scene_id, frame_idx, cam_idx)
                        test_images.append(frame_data['image'])
                        test_extrinsics.append(frame_data['extrinsic'])
                        test_intrinsics.append(frame_data['intrinsic'])
                        test_depths.append(frame_data['depth'])
                        sky_mask = frame_data.get('sky_mask')
                        if sky_mask is not None:
                            has_test_sky_mask = True
                        test_sky_masks.append(sky_mask)
                        egocar_mask = frame_data.get("egocar_mask")
                        if egocar_mask is not None:
                            has_test_egocar_mask = True
                        test_egocar_masks.append(egocar_mask)
                        test_frame_idxs.append(frame_idx)
                        test_cam_idxs.append(cam_idx)
        
        # 7.5 Transform extrinsics to segment-first frame coordinates
        source_extrinsics = _transform_extrinsics_list(source_extrinsics)
        target_extrinsics = _transform_extrinsics_list(target_extrinsics)
        if include_test and len(test_extrinsics) > 0:
            test_extrinsics = _transform_extrinsics_list(test_extrinsics)

        # 8. Assemble batch
        # Get actual scene folder path for debugging
        scene_folder_name = f"{int(scene_id):03d}" if self.data_cfg.get("dataset") not in ["kitti", "nuplan"] else str(scene_id)
        # AABB in segment-first-frame (seg0) coords; single source of truth.
        batch_aabb = self.segment_aabb.to(device=self.device)
        batch = {
            'scene_id': torch.tensor([scene_id], dtype=torch.long),
            'scene_folder_name': scene_folder_name,  # Actual folder name (e.g., "001", "007")
            'segment_id': segment_id,
            'aabb': batch_aabb,  # [2, 3] min/max in segment-first-frame (seg0) coords
            'segment_first_pose': segment_first_pose,  # 4x4 pose of segment first frame in original world
            'segment_first_frame_idx': segment_first_frame_idx,
            'segment_first_pose_source': segment_pose_source,
            
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

        # Attach sky masks if available (canonical 1=sky; missing -> zeros = all non-sky)
        if has_source_sky_mask:
            source_sky_mask_stack = []
            for mask, img in zip(source_sky_masks, source_images):
                if mask is None:
                    H, W = img.shape[:2]
                    source_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    source_sky_mask_stack.append(mask.to(self.device).float())
            batch['source']['sky_mask'] = torch.stack(source_sky_mask_stack, dim=0)

        if has_source_viewdirs:
            source_viewdirs_stack = []
            for vd, img in zip(source_viewdirs_list, source_images):
                if vd is None:
                    H, W = img.shape[:2]
                    source_viewdirs_stack.append(torch.zeros((H, W, 3), dtype=torch.float32, device=self.device))
                else:
                    source_viewdirs_stack.append(vd.to(self.device).float())
            batch['source']['viewdirs'] = torch.stack(source_viewdirs_stack, dim=0)

        if has_source_egocar_mask:
            source_egocar_mask_stack = []
            for mask, img in zip(source_egocar_masks, source_images):
                if mask is None:
                    H, W = img.shape[:2]
                    source_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    source_egocar_mask_stack.append(mask.to(self.device).float())
            batch["source"]["egocar_mask"] = torch.stack(source_egocar_mask_stack, dim=0)

        if has_target_sky_mask:
            target_sky_mask_stack = []
            for mask, img in zip(target_sky_masks, target_images):
                if mask is None:
                    H, W = img.shape[:2]
                    target_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    target_sky_mask_stack.append(mask.to(self.device).float())
            batch['target']['sky_mask'] = torch.stack(target_sky_mask_stack, dim=0)

        # Target viewdirs (for sky / Stage 3.1): from pixel_source.get_image() image_infos['viewdirs']
        if has_target_viewdirs:
            target_viewdirs_stack = []
            for vd, img in zip(target_viewdirs_list, target_images):
                if vd is None:
                    H, W = img.shape[:2]
                    target_viewdirs_stack.append(torch.zeros((H, W, 3), dtype=torch.float32, device=self.device))
                else:
                    target_viewdirs_stack.append(vd.to(self.device).float())
            batch['target']['viewdirs'] = torch.stack(target_viewdirs_stack, dim=0)

        if has_target_egocar_mask:
            target_egocar_mask_stack = []
            for mask, img in zip(target_egocar_masks, target_images):
                if mask is None:
                    H, W = img.shape[:2]
                    target_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                else:
                    target_egocar_mask_stack.append(mask.to(self.device).float())
            batch["target"]["egocar_mask"] = torch.stack(target_egocar_mask_stack, dim=0)
        
        # Add pointcloud to batch if generated
        if pointcloud is not None:
            batch['pointcloud'] = pointcloud
        
        # Add dynamic_info to batch if available
        if dynamic_info is not None:
            batch['dynamic_info'] = dynamic_info

        # Add test views if available
        if include_test and len(test_images) > 0:
            batch['test'] = {
                'image': torch.stack(test_images, dim=0),
                'extrinsics': torch.stack(test_extrinsics, dim=0),
                'intrinsics': torch.stack(test_intrinsics, dim=0),
                'depth': torch.stack(test_depths, dim=0),
                'frame_indices': torch.tensor(test_frame_idxs, dtype=torch.long),
                'cam_indices': torch.tensor(test_cam_idxs, dtype=torch.long),
            }
            if has_test_sky_mask:
                test_sky_mask_stack = []
                for mask, img in zip(test_sky_masks, test_images):
                    if mask is None:
                        H, W = img.shape[:2]
                        test_sky_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                    else:
                        test_sky_mask_stack.append(mask.to(self.device).float())
                batch['test']['sky_mask'] = torch.stack(test_sky_mask_stack, dim=0)
            if has_test_egocar_mask:
                test_egocar_mask_stack = []
                for mask, img in zip(test_egocar_masks, test_images):
                    if mask is None:
                        H, W = img.shape[:2]
                        test_egocar_mask_stack.append(torch.zeros((H, W), dtype=torch.float32, device=self.device))
                    else:
                        test_egocar_mask_stack.append(mask.to(self.device).float())
                batch["test"]["egocar_mask"] = torch.stack(test_egocar_mask_stack, dim=0)
        
        return batch
    
    def sample_random_batch(self, eval: bool = False, include_test: bool = False) -> Dict:
        """
        Randomly sample a training batch from current scene.
        
        Args:
            eval: If True, sample from eval scenes; otherwise from train scenes
            include_test: Whether to include test views if available
        
        Returns:
            Same format as get_segment_batch()
        """
        # Ensure initialized
        if not self._initialized:
            self.initialize()
        
        # Select scene IDs based on eval flag
        if eval:
            scene_ids = self.eval_scene_ids
            if len(scene_ids) == 0:
                raise ValueError("No evaluation scenes available. Please check eval_scene_ids configuration.")
            # Randomly select an eval scene
            current_scene_id = random.choice(scene_ids)
        else:
            # Get current scene ID from training queue
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
        
        return self.get_segment_batch(current_scene_id, segment_id, include_test=include_test)
    
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
        include_test: bool = False,
    ) -> 'MultiSceneDatasetScheduler':
        """
        Create a scheduler instance for managing scene and segment traversal.
        
        Args:
            batches_per_segment: Number of batches to iterate per segment (default 20)
            segment_order: Segment traversal order ("random" or "sequential", default "random")
            scene_order: Scene traversal order ("random" or "sequential", default "random")
            shuffle_segments: Whether to shuffle segments within each scene (default True)
            preload_next_scene: Whether to preload next scene when last segment starts (default True)
            include_test: Whether scheduler batches should include test views
            
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
            include_test=include_test,
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
        include_test: bool = False,
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
            include_test: Whether to include test views in sampled batches
        """
        self.dataset = dataset
        self.batches_per_segment = batches_per_segment
        self.segment_order = segment_order
        self.scene_order = scene_order
        self.shuffle_segments = shuffle_segments
        self.preload_next_scene = preload_next_scene
        self.include_test = include_test
        
        # State variables
        self.current_scene_id: Optional[int] = None
        self.current_segment_id: int = 0  # Index in scene_segment_order
        self.current_batch_count: int = 0
        self.scene_segment_order: List[int] = []  # Segment IDs in traversal order
        
        # Background thread for scene preloading
        self._preload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Preload task queue (Scheduler → background thread)
        self._preload_task_queue = queue.Queue()
        
        # Scene loading status (scene_id -> Event)
        self._scene_loading_events: Dict[int, threading.Event] = {}
        self._loading_lock = threading.RLock()  # Protect loading status dictionary
        
        # Next scene ID (maintained by Scheduler)
        self._next_scene_id: Optional[int] = None
        
        # Initialize scheduler state
        self._initialize_scheduler_state()
        
        # Start background thread
        self._preload_thread = threading.Thread(
            target=self._preload_worker,
            daemon=True,  # Daemon thread, exits when main thread exits
            name="ScenePreloadWorker"
        )
        self._preload_thread.start()
    
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
    
    def _preload_worker(self):
        """Background thread main loop: process preload tasks + ensure queue is full"""
        while not self._stop_event.is_set():
            try:
                # 1. Process preload tasks (high priority)
                try:
                    scene_id = self._preload_task_queue.get(timeout=0.1)
                    self._load_scene_in_background(scene_id)
                except queue.Empty:
                    pass
                
                # 2. Ensure queue is full (continuous monitoring)
                # Check if dataset has _lock attribute (for Mock objects in tests)
                if hasattr(self.dataset, '_lock'):
                    try:
                        with self.dataset._lock:
                            self.dataset._ensure_training_queue_ready()
                    except AttributeError:
                        # Mock object doesn't have proper _lock, skip
                        pass
                else:
                    # No _lock attribute, skip queue management
                    pass
                
                # 3. Preload next scene in queue (if queue has new scenes)
                self._preload_next_scene_in_queue()
                
                # Brief sleep to avoid excessive CPU usage
                time.sleep(0.01)
            except Exception as e:
                # Only log if not a stop event (to avoid noise in tests)
                if not self._stop_event.is_set():
                    logger.error(f"Error in preload worker: {e}", exc_info=True)
    
    def _load_scene_in_background(self, scene_id: int):
        """Load scene in background thread"""
        # Create loading event
        with self._loading_lock:
            if scene_id not in self._scene_loading_events:
                self._scene_loading_events[scene_id] = threading.Event()
            event = self._scene_loading_events[scene_id]
        
        # If already in cache, mark as complete
        # Check if dataset has _lock attribute (for Mock objects in tests)
        if hasattr(self.dataset, '_lock') and hasattr(self.dataset, 'train_scenes_cache'):
            try:
                with self.dataset._lock:
                    if scene_id in self.dataset.train_scenes_cache:
                        event.set()
                        return
            except AttributeError:
                # Mock object doesn't have proper _lock, continue
                pass
        
        # Load scene (without lock, as this is a long I/O operation)
        try:
            if hasattr(self.dataset, '_load_and_prepare_scene'):
                scene_data = self.dataset._load_and_prepare_scene(scene_id)
                if scene_data is not None:
                    # Update cache with lock
                    if hasattr(self.dataset, '_lock') and hasattr(self.dataset, 'train_scenes_cache'):
                        try:
                            with self.dataset._lock:
                                # Check cache size, may need to unload other scenes
                                max_cache_size = self.dataset.preload_scene_count + 1
                                if len(self.dataset.train_scenes_cache) >= max_cache_size:
                                    # Unload non-current scenes
                                    current_scene_id = self.dataset.get_current_scene_id()
                                    for cached_id in list(self.dataset.train_scenes_cache.keys()):
                                        if cached_id != current_scene_id:
                                            self.dataset._unload_scene(cached_id)
                                            break
                                
                                self.dataset.train_scenes_cache[scene_id] = scene_data
                        except AttributeError:
                            # Mock object doesn't have proper _lock, skip cache update
                            pass
                    event.set()  # Mark loading complete
                    if not self._stop_event.is_set():
                        logger.info(f"Scene {scene_id} preloaded in background")
                else:
                    event.set()  # Set even on failure to avoid permanent blocking
                    if not self._stop_event.is_set():
                        logger.warning(f"Failed to preload scene {scene_id}")
            else:
                # Mock object doesn't have _load_and_prepare_scene, just mark complete
                event.set()
        except Exception as e:
            # Only log if not a stop event (to avoid noise in tests)
            if not self._stop_event.is_set():
                logger.error(f"Error loading scene {scene_id}: {e}", exc_info=True)
            event.set()  # Set even on failure to avoid permanent blocking
    
    def _preload_next_scene_in_queue(self):
        """Preload next scene in queue (if exists and not loaded)"""
        # Check if dataset has _lock attribute (for Mock objects in tests)
        if not hasattr(self.dataset, '_lock'):
            return
        
        try:
            with self.dataset._lock:
                if not hasattr(self.dataset, 'get_current_scene_id') or not hasattr(self.dataset, 'scene_training_queue'):
                    return
                
                current_scene_id = self.dataset.get_current_scene_id()
                if current_scene_id is None:
                    return
                
                # Get next scene ID
                try:
                    current_index = self.dataset.scene_training_queue.index(current_scene_id)
                    next_index = current_index + 1
                    if next_index < len(self.dataset.scene_training_queue):
                        next_scene_id = self.dataset.scene_training_queue[next_index]
                        
                        # If not loaded, send preload task
                        if hasattr(self.dataset, 'train_scenes_cache') and next_scene_id not in self.dataset.train_scenes_cache:
                            try:
                                self._preload_task_queue.put_nowait(next_scene_id)
                            except queue.Full:
                                pass  # Queue full, skip
                except (ValueError, IndexError):
                    pass  # Current scene not in queue, or no next scene
        except AttributeError:
            # Mock object doesn't have proper _lock, skip
            pass
    
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
        """Switch to next scene, blocking if scene is not loaded yet."""
        # Mark current scene as completed
        if self.current_scene_id is not None:
            with self.dataset._lock:
                self.dataset.mark_scene_completed(self.current_scene_id)
        
        # Get next scene
        with self.dataset._lock:
            self.current_scene_id = self.dataset.get_current_scene_id()
            if self.current_scene_id is None:
                raise StopIteration("All scenes have been processed")
        
        # Check if scene is loaded, block if not loaded
        with self._loading_lock:
            if self.current_scene_id not in self._scene_loading_events:
                # Create loading event (may be loading)
                self._scene_loading_events[self.current_scene_id] = threading.Event()
            event = self._scene_loading_events[self.current_scene_id]
        
        # Check if already in cache
        scene_data = None
        with self.dataset._lock:
            if self.current_scene_id in self.dataset.train_scenes_cache:
                # Already loaded, use directly
                scene_data = self.dataset.train_scenes_cache[self.current_scene_id]
            else:
                # Not loaded, trigger preload (if not already)
                try:
                    self._preload_task_queue.put_nowait(self.current_scene_id)
                except queue.Full:
                    pass
        
        # If not loaded, wait for loading to complete (outside lock to avoid deadlock)
        if scene_data is None:
            logger.info(f"Waiting for scene {self.current_scene_id} to load...")
            event.wait()  # Block and wait (without holding dataset._lock)
            
            # Check cache again after waiting
            with self.dataset._lock:
                scene_data = self.dataset.train_scenes_cache.get(self.current_scene_id)
                if scene_data is None:
                    raise ValueError(f"Scene {self.current_scene_id} failed to load")
        
        logger.info(f"Switched to scene {self.current_scene_id}")
    
    def _preload_next_scene_if_needed(self):
        """Trigger preload for next scene when starting the last segment."""
        scene_data = self.dataset.get_scene(self.current_scene_id)
        if scene_data is None:
            return
        
        # Check if current segment is the last one
        is_last_segment = (self.current_segment_id == len(self.scene_segment_order) - 1)
        
        if is_last_segment and self.preload_next_scene:
            # Get next scene ID
            with self.dataset._lock:
                current_scene_id = self.dataset.get_current_scene_id()
                if current_scene_id is None:
                    return
                
                try:
                    current_index = self.dataset.scene_training_queue.index(current_scene_id)
                    next_index = current_index + 1
                    if next_index < len(self.dataset.scene_training_queue):
                        next_scene_id = self.dataset.scene_training_queue[next_index]
                        self._next_scene_id = next_scene_id
                        
                        # Send preload task to background thread
                        try:
                            self._preload_task_queue.put_nowait(next_scene_id)
                            logger.debug(f"Triggered preload for next scene {next_scene_id}")
                        except queue.Full:
                            logger.warning("Preload task queue is full")
                except (ValueError, IndexError):
                    pass
    
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
        batch = self.dataset.get_segment_batch(
            self.current_scene_id,
            segment_id,
            include_test=self.include_test,
        )
        
        # Increment batch count
        self.current_batch_count += 1
        
        return batch
    
    def shutdown(self):
        """Stop background thread and cleanup resources."""
        if self._preload_thread is not None:
            self._stop_event.set()
            self._preload_thread.join(timeout=5.0)
            if self._preload_thread.is_alive():
                logger.warning("Preload thread did not stop in time")
            self._preload_thread = None
    
    def reset(self):
        """Reset scheduler state."""
        # Clear loading events
        with self._loading_lock:
            self._scene_loading_events.clear()
        
        # Reset other state
        self.current_scene_id = None
        self.current_segment_id = 0
        self.current_batch_count = 0
        self.scene_segment_order = []
        self._next_scene_id = None
        
        # Reinitialize
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
    
    def generate_segment_pointcloud(
        self,
        pointcloud_generator,
        scene_id: Optional[int] = None,
        segment_id: Optional[int] = None,
    ):
        """
        为当前段（或指定段）生成点云。
        
        Args:
            pointcloud_generator: 点云生成器实例
            scene_id: 场景ID（如果为None，使用当前场景）
            segment_id: 段ID（如果为None，使用当前段）
            
        Returns:
            Dict: 点云结果（背景 + 动态物体）
        """
        if scene_id is None:
            scene_id = self.current_scene_id
            if scene_id is None:
                raise ValueError("No current scene available")
        
        if segment_id is None:
            if self.current_segment_id >= len(self.scene_segment_order):
                raise ValueError("No current segment available")
            segment_id = self.scene_segment_order[self.current_segment_id]
        segment_first_pose, _, _ = self.dataset.get_segment_first_pose(scene_id, segment_id)
        return pointcloud_generator.generate_pointcloud(
            dataset=self.dataset,
            scene_id=scene_id,
            segment_id=segment_id,
            segment_first_pose=segment_first_pose,
        )
    
    def generate_all_segment_pointclouds(
        self,
        pointcloud_generator,
        scene_id: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        为场景的所有段生成点云。
        
        Args:
            pointcloud_generator: 点云生成器实例
            scene_id: 场景ID（如果为None，使用当前场景）
            save_dir: 保存目录（如果为None，不保存）
            
        Returns:
            Dict[segment_id, Dict]: 每个段的点云结果字典
        """
        if scene_id is None:
            scene_id = self.current_scene_id
            if scene_id is None:
                raise ValueError("No current scene available")
        
        scene_data = self.dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} not found")
        
        segments = scene_data['segments']
        pointclouds = {}
        
        for segment_id in range(len(segments)):
            try:
                segment_first_pose, _, _ = self.dataset.get_segment_first_pose(scene_id, segment_id)
                pointcloud = pointcloud_generator.generate_pointcloud(
                    dataset=self.dataset,
                    scene_id=scene_id,
                    segment_id=segment_id,
                    segment_first_pose=segment_first_pose,
                )
                pointclouds[segment_id] = pointcloud
                
                # 保存点云（如果指定了保存目录）
                if save_dir is not None and isinstance(pointcloud, dict):
                    import open3d as o3d
                    
                    os.makedirs(save_dir, exist_ok=True)
                    background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
                    pcd = o3d.geometry.PointCloud()
                    if len(background) > 0:
                        pcd.points = o3d.utility.Vector3dVector(background[:, :3])
                        pcd.colors = o3d.utility.Vector3dVector(background[:, 3:] / 255.0)
                    save_path = os.path.join(save_dir, f"scene_{scene_id}_segment_{segment_id}.ply")
                    o3d.io.write_point_cloud(save_path, pcd)
                    logger.info(f"Saved pointcloud to {save_path}")
                
            except Exception as e:
                logger.warning(f"Failed to generate pointcloud for scene {scene_id}, segment {segment_id}: {e}")
                continue
        
        return pointclouds
