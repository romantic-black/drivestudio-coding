"""
EVolSplat Trainer for drivestudio.

This trainer integrates EVolSplat's feature extraction and Gaussian parameter generation
with drivestudio's rendering system.
"""
from typing import Dict, Optional
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
import imageio.v2 as imageio
import cv2
import open3d as o3d

from models.trainers.scene_graph import MultiTrainer
from models.gaussians.basics import dataclass_gs
from gsplat.cuda._wrapper import spherical_harmonics
from models.evol_splat import (
    SparseCostRegNet,
    construct_sparse_tensor,
    sparse_to_dense_volume,
    Projector,
    create_gaussion_decoder,
    create_mlp_conv,
    create_mlp_opacity,
    create_mlp_offset,
    interpolate_features,
    get_grid_coords,
)
from gsplat.cuda_legacy._wrapper import num_sh_bases
from datasets.nuscenes.nuscenes_mono_pcd import (
    NuScenesMonoPCDGenerator,
    get_image_dimensions
)

logger = logging.getLogger()


class EVolSplatTrainer(MultiTrainer):
    """Trainer that uses EVolSplat components to generate 3DGS parameters dynamically.
    
    Key design:
    - EVolSplat generates 3DGS point cloud parameters (not images)
    - These parameters are converted to drivestudio format
    - drivestudio's rendering system is used for final rendering
    """

    def __init__(
        self,
        num_timesteps: int,
        evol_splat_config: OmegaConf,
        **kwargs
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        
        # EVolSplat configuration
        self.evol_splat_cfg = evol_splat_config
        self.local_radius = evol_splat_config.get("local_radius", 1)
        self.sparse_conv_outdim = evol_splat_config.get("sparse_conv_outdim", 8)
        self.offset_max = evol_splat_config.get("offset_max", 0.1)
        self.num_neighbours = evol_splat_config.get("num_neighbour_select", 4)
        self.voxel_size = evol_splat_config.get("voxel_size", 0.1)
        self.sh_degree = evol_splat_config.get("sh_degree", 1)
        
        # Bounding box (will be set from scene_aabb)
        self.bbx_min = None
        self.bbx_max = None
        
        # Initialize EVolSplat components
        self.projector = Projector()
        self.sparse_conv = SparseCostRegNet(
            d_in=3, d_out=self.sparse_conv_outdim
        ).to(self.device)
        
        # Feature dimensions
        self.feature_dim_out = 3 * num_sh_bases(self.sh_degree)
        self.feature_dim_in = 4 * self.num_neighbours * (2 * self.local_radius + 1) ** 2
        
        # MLP decoders
        self.gaussion_decoder = create_gaussion_decoder(
            feature_dim_in=self.feature_dim_in,
            feature_dim_out=self.feature_dim_out,
            sh_degree=self.sh_degree
        ).to(self.device)
        
        self.mlp_conv = create_mlp_conv(
            sparse_conv_outdim=self.sparse_conv_outdim
        ).to(self.device)
        
        self.mlp_opacity = create_mlp_opacity(
            sparse_conv_outdim=self.sparse_conv_outdim
        ).to(self.device)
        
        self.mlp_offset = create_mlp_offset(
            sparse_conv_outdim=self.sparse_conv_outdim
        ).to(self.device)
        
        # Point cloud data (will be initialized from dataset)
        self.means = None  # [N, 3] initial point cloud coordinates
        self.anchor_feats = None  # [N, 3] initial point cloud colors
        self.scales = None  # [N, 3] initial scales
        self.offset = None  # [N, 3] offset
        
        # Cached gaussians (generated in forward, used in collect_gaussians)
        self._cached_gaussians = None
        self._cached_vol_dim = None
        self._frozen_volume = evol_splat_config.get("freeze_volume", False)
        self.dense_volume = None
        
        logger.info("EVolSplatTrainer initialized")

    def get_param_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for EVolSplat components.
        This is used by the optimizer initialization.
        
        Returns:
            Dictionary mapping parameter group names to parameter lists
        """
        param_groups = {}
        
        # EVolSplat MLP decoders
        param_groups['EVolSplat#gaussion_decoder'] = list(self.gaussion_decoder.parameters())
        param_groups['EVolSplat#mlp_conv'] = list(self.mlp_conv.parameters())
        param_groups['EVolSplat#mlp_opacity'] = list(self.mlp_opacity.parameters())
        param_groups['EVolSplat#mlp_offset'] = list(self.mlp_offset.parameters())
        param_groups['EVolSplat#sparse_conv'] = list(self.sparse_conv.parameters())
        
        # Misc models (Sky, Affine, etc.)
        for class_name, model in self.models.items():
            if hasattr(model, 'get_param_groups'):
                param_groups.update(model.get_param_groups())
        
        return param_groups

    def initialize_optimizer(self) -> None:
        """
        Override parent method to include EVolSplat component parameters.
        """
        # Get param groups from EVolSplat components
        self.param_groups = self.get_param_groups()
        
        # Also get param groups from misc models (Sky, Affine, etc.)
        for class_name, model in self.models.items():
            if hasattr(model, 'get_param_groups'):
                self.param_groups.update(model.get_param_groups())
        
        groups = []
        lr_schedulers = {}
        
        # EVolSplat optimizer config (use default learning rates if not specified)
        evol_splat_optim_cfg = self.evol_splat_cfg.get("optim", {})
        default_lr = evol_splat_optim_cfg.get("default_lr", 0.001)
        
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            
            # Handle EVolSplat parameters
            if class_name == "EVolSplat":
                lr = evol_splat_optim_cfg.get(component_name, {}).get("lr", default_lr)
                groups.append({
                    'params': params,
                    'name': params_name,
                    'lr': lr,
                    'eps': 1e-15,
                    'weight_decay': 0
                })
            else:
                # Handle misc models (Sky, Affine, etc.)
                class_cfg = self.model_config.get(class_name)
                if class_cfg is None:
                    # Skip if not in model_config
                    continue
                class_optim_cfg = class_cfg.get("optim", {})
                
                raw_optim_cfg = class_optim_cfg.get(component_name, None)
                if raw_optim_cfg is None:
                    continue
                    
                lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
                if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                    lr_scale_factor = self.scene_radius
                
                optim_cfg = OmegaConf.create({
                    "lr": raw_optim_cfg.get('lr', 0.0005),
                    "eps": raw_optim_cfg.get('eps', 1.0e-15),
                    "weight_decay": raw_optim_cfg.get('weight_decay', 0),
                })
                optim_cfg.lr = optim_cfg.lr * lr_scale_factor
                lr_init = optim_cfg.lr
                
                groups.append({
                    'params': params,
                    'name': params_name,
                    'lr': optim_cfg.lr,
                    'eps': optim_cfg.eps,
                    'weight_decay': optim_cfg.weight_decay
                })
                
                if raw_optim_cfg.get("lr_final", None) is not None:
                    from models.trainers.base import lr_scheduler_fn
                    sched_cfg = OmegaConf.create({
                        "opt_after": raw_optim_cfg.get('opt_after', 0),
                        "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                        "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                        "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                        "lr_final": raw_optim_cfg.get('lr_final', None),
                        "ramp": raw_optim_cfg.get('ramp', "cosine"),
                    })
                    sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                    sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                    sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                    lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)
        
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.optim_general.get("use_grad_scaler", False)
        )

    def _init_models(self):
        """
        Override parent method: EVolSplatTrainer doesn't use traditional node models.
        Only initialize misc models (Sky, Affine, etc.) if needed.
        """
        # Only initialize misc models (Sky, Affine, CamPose, etc.)
        # Don't initialize gaussian node models since we generate gaussians dynamically
        self.gaussian_classes = {}  # No gaussian classes for EVolSplat
        
        for class_name, model_cfg in self.model_config.items():
            if class_name in self.misc_classes_keys:
                from utils.misc import import_str
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)
                self.models[class_name] = model
        
        logger.info(f"Initialized misc models: {self.models.keys()}")
        
        # Register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)

    def init_mono_points_from_dataset(self, dataset):
        """
        Initialize seed points from dataset using monocular depth maps.
        
        This method generates point cloud from monocular depth maps following
        the notebook implementation in notebooks/nuscenes_pcd_generation.ipynb.
        It reads depth maps directly from the file system without using dataset classes.
        
        Args:
            dataset: DrivingDataset instance
        """
        logger.info("Initializing point cloud from monocular depth maps...")
        
        # 1. Get configuration parameters
        depth_map_cameras = self.evol_splat_cfg.get("depth_map_cameras", [0])
        mono_pcd_cfg = self.evol_splat_cfg.get("mono_pcd", {})
        sparsity = mono_pcd_cfg.get("sparsity", "full")
        filter_sky = mono_pcd_cfg.get("filter_sky", True)
        depth_consistency = mono_pcd_cfg.get("depth_consistency", True)
        use_bbx = mono_pcd_cfg.get("use_bbx", True)
        initial_scale = mono_pcd_cfg.get("initial_scale", 0.01)
        
        logger.info(f"Monocular PCD config: cameras={depth_map_cameras}, sparsity={sparsity}, "
                   f"filter_sky={filter_sky}, depth_consistency={depth_consistency}, use_bbx={use_bbx}")
        
        # 2. Get scene path and frame indices
        scene_dir = dataset.data_path
        if not os.path.exists(scene_dir):
            raise ValueError(f"Scene directory not found: {scene_dir}")
        
        # Convert to absolute frame indices
        # train_timesteps are relative to start_timestep, but depth files use absolute indices
        absolute_frame_indices = dataset.start_timestep + dataset.train_timesteps
        absolute_frame_indices = sorted(absolute_frame_indices.tolist())
        
        if len(absolute_frame_indices) == 0:
            raise ValueError("No training frames found")
        
        logger.info(f"Processing {len(absolute_frame_indices)} training frames "
                   f"(absolute indices: {absolute_frame_indices[0]} to {absolute_frame_indices[-1]})")
        
        # 3. Get bounding box
        if use_bbx and self.aabb is not None:
            bbx_min = self.aabb[0].cpu().numpy()
            bbx_max = self.aabb[1].cpu().numpy()
            logger.info(f"Using bounding box: min={bbx_min}, max={bbx_max}")
        else:
            bbx_min = bbx_max = None
            logger.info("Bounding box filtering disabled")
        
        # 4. Initialize point cloud generator
        pcd_generator = NuScenesMonoPCDGenerator(
            sparsity=sparsity,
            frame_start=absolute_frame_indices[0],
            filter_sky=filter_sky,
            depth_consistency=depth_consistency,
            use_bbx=use_bbx,
            bbx_min=bbx_min,
            bbx_max=bbx_max
        )
        
        # 5. Set directories and image dimensions
        pcd_generator.dir_name = scene_dir
        pcd_generator.depth_dir = os.path.join(scene_dir, 'depth')
        H, W = get_image_dimensions(scene_dir)
        pcd_generator.H, pcd_generator.W = H, W
        
        logger.info(f"Image dimensions: H={H}, W={W}")
        
        # 6. Load and filter depth files (by training frames and camera IDs)
        depth_dir = pcd_generator.depth_dir
        if not os.path.exists(depth_dir):
            raise ValueError(f"Depth directory not found: {depth_dir}")
        
        # Get all depth files
        all_depth_files = sorted([
            f for f in os.listdir(depth_dir) 
            if f.endswith('.npy') and not f.endswith('_meta.npz')
        ])
        
        logger.info(f"Found {len(all_depth_files)} total depth files")
        
        # Filter by camera IDs
        filtered_depth_files = []
        for file_name in all_depth_files:
            try:
                parts = file_name.replace('.npy', '').split('_')
                if len(parts) >= 2:
                    frame_idx = int(parts[0])
                    cam_id = int(parts[1])
                    if cam_id in depth_map_cameras and frame_idx in absolute_frame_indices:
                        filtered_depth_files.append(file_name)
            except (ValueError, IndexError):
                continue
        
        logger.info(f"Filtered to {len(filtered_depth_files)} depth files "
                   f"(cameras={depth_map_cameras}, training frames)")
        
        # 7. Group depth files by frame index and apply sparsity filtering
        frame_groups = {}
        for file_name in filtered_depth_files:
            try:
                frame_idx = int(file_name.split('_')[0])
                if frame_idx not in frame_groups:
                    frame_groups[frame_idx] = []
                frame_groups[frame_idx].append(file_name)
            except (ValueError, IndexError):
                continue
        
        sorted_frame_indices = sorted(frame_groups.keys())
        
        # Apply sparsity filtering based on position in sorted frame list
        selected_frames = []
        for frame_pos, frame_idx in enumerate(sorted_frame_indices):
            selected_frames.append(frame_idx)
        
        # Collect depth files for selected frames
        depth_files = []
        for frame_idx in selected_frames:
            if frame_idx in frame_groups:
                depth_files.extend(sorted(frame_groups[frame_idx]))
        
        logger.info(f"After sparsity filtering: {len(selected_frames)} frames, {len(depth_files)} depth files")
        
        if len(depth_files) == 0:
            raise ValueError("No depth files selected after sparsity filtering")
        
        # 8. Match depth files with poses and intrinsics
        extrinsics_dir = os.path.join(scene_dir, 'extrinsics')
        intrinsics_dir = os.path.join(scene_dir, 'intrinsics')
        
        # Load reference pose for alignment (first frame, first camera)
        camera_front_start = None
        first_frame_cam = None
        for frame_idx in absolute_frame_indices:
            if 0 in depth_map_cameras:  # Use camera 0 as reference
                first_extrinsic_file = os.path.join(extrinsics_dir, f'{frame_idx:03d}_0.txt')
                if os.path.exists(first_extrinsic_file):
                    camera_front_start = np.loadtxt(first_extrinsic_file)
                    first_frame_cam = (frame_idx, 0)
                    break
        
        if camera_front_start is None:
            # Fallback: use first available extrinsic
            if len(absolute_frame_indices) > 0 and len(depth_map_cameras) > 0:
                first_frame_cam = (absolute_frame_indices[0], depth_map_cameras[0])
                first_extrinsic_file = os.path.join(extrinsics_dir, 
                    f'{first_frame_cam[0]:03d}_{first_frame_cam[1]}.txt')
                if os.path.exists(first_extrinsic_file):
                    camera_front_start = np.loadtxt(first_extrinsic_file)
                    logger.warning(f"Using frame {first_frame_cam[0]} camera {first_frame_cam[1]} for alignment")
        
        pcd_generator.camera_front_start = camera_front_start
        
        # Pre-load intrinsics for all cameras
        cam_intrinsics_dict = {}
        for cam_id in depth_map_cameras:
            intrinsic_file = os.path.join(intrinsics_dir, f'{cam_id}.txt')
            if os.path.exists(intrinsic_file):
                intrinsic_data = np.loadtxt(intrinsic_file)
                fx, fy, cx, cy = intrinsic_data[0], intrinsic_data[1], intrinsic_data[2], intrinsic_data[3]
                cam_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                cam_intrinsics_dict[cam_id] = cam_intrinsic
            else:
                logger.warning(f"Intrinsic file not found: {intrinsic_file}")
        
        # Match depth files with poses
        OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        valid_depth_files = []
        pcd_generator.c2w = []
        pcd_generator.intri = []
        
        for file_name in depth_files:
            try:
                parts = file_name.replace('.npy', '').split('_')
                if len(parts) >= 2:
                    frame_idx = int(parts[0])
                    cam_id = int(parts[1])
                    
                    if cam_id not in depth_map_cameras:
                        continue
                    
                    # Load extrinsic
                    extrinsic_file = os.path.join(extrinsics_dir, f'{frame_idx:03d}_{cam_id}.txt')
                    if not os.path.exists(extrinsic_file):
                        logger.warning(f"Extrinsic file not found: {extrinsic_file}, skipping")
                        continue
                    
                    cam2world = np.loadtxt(extrinsic_file)
                    
                    # Align to first frame first camera
                    if camera_front_start is not None:
                        cam2world = np.linalg.inv(camera_front_start) @ cam2world
                    
                    # Convert to OpenCV coordinate system
                    cam2world = cam2world @ OPENCV2DATASET
                    
                    # Get intrinsic
                    if cam_id in cam_intrinsics_dict:
                        cam_intrinsic = cam_intrinsics_dict[cam_id]
                    else:
                        logger.warning(f"Intrinsic not found for cam_id {cam_id}, using identity")
                        cam_intrinsic = np.eye(3)
                    
                    valid_depth_files.append(file_name)
                    pcd_generator.c2w.append(cam2world)
                    pcd_generator.intri.append(cam_intrinsic)
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing {file_name}: {e}")
                continue
        
        depth_files = valid_depth_files
        
        if len(depth_files) == 0:
            raise ValueError("No valid depth files after matching with poses")
        
        if len(pcd_generator.c2w) != len(depth_files) or len(pcd_generator.intri) != len(depth_files):
            raise ValueError(f"Mismatch: {len(pcd_generator.c2w)} poses, {len(pcd_generator.intri)} intrinsics, "
                           f"{len(depth_files)} depth files")
        
        logger.info(f"Matched {len(depth_files)} depth files with poses and intrinsics")
        
        # 9. Depth consistency check (if enabled and frames are continuous enough)
        if depth_consistency:
            # Check if frames are mostly continuous
            frame_indices_in_files = sorted([int(f.split('_')[0]) for f in depth_files])
            frame_gaps = [frame_indices_in_files[i+1] - frame_indices_in_files[i] 
                         for i in range(len(frame_indices_in_files)-1)]
            max_gap = max(frame_gaps) if frame_gaps else 0
            
            if max_gap > 5:  # If gaps are too large, disable depth consistency
                logger.warning(f"Large frame gaps detected (max={max_gap}), disabling depth consistency")
                depth_consistency = False
                pcd_generator.depth_consistency = False
        
        if depth_consistency:
            logger.info("Performing depth consistency check...")
            consistency_masks = pcd_generator.depth_consistency_check(depth_files=depth_files, H=H, W=W)
        else:
            consistency_masks = [np.ones((H, W), dtype=np.bool_) for _ in range(len(depth_files))]
        
        # 10. Point cloud accumulation main loop
        logger.info(f"Starting point cloud accumulation for {len(depth_files)} depth files...")
        
        # Add depth_utils path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        depth_utils_path = os.path.join(
            project_root, 
            'third_party/EVolSplat/preprocess/metric3d/mono/tools'
        )
        import sys
        if depth_utils_path not in sys.path:
            sys.path.insert(0, depth_utils_path)
        
        from depth_utils import process_depth_for_use
        
        color_pointclouds = []
        downscale = 2  # Downsample factor for point cloud generation
        
        # Create downscale mask
        downscale_mask = np.zeros((H, W), dtype=np.bool_)
        downscale_mask[::downscale, ::downscale] = True
        
        for i, file_name in enumerate(depth_files):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processing {i+1}/{len(depth_files)}: {file_name}")
            
            # Load and preprocess depth map
            depth_file = os.path.join(depth_dir, file_name)
            try:
                depth, metadata = process_depth_for_use(depth_file, target_shape=(H, W))
            except Exception as e:
                logger.warning(f"Failed to load depth {file_name}: {e}")
                continue
            
            # Load RGB image
            rgb_file = os.path.join(scene_dir, 'images', file_name.replace('.npy', '.jpg'))
            if not os.path.exists(rgb_file):
                rgb_file = os.path.join(scene_dir, 'images', file_name.replace('.npy', '.png'))
            
            if not os.path.exists(rgb_file):
                logger.warning(f"RGB file not found for {file_name}")
                continue
            
            try:
                rgb = imageio.imread(rgb_file) / 255.0
            except Exception as e:
                logger.warning(f"Failed to load RGB {file_name}: {e}")
                continue
            
            # Apply sky filtering if enabled
            if filter_sky:
                sky_mask_file = os.path.join(scene_dir, 'sky_masks', file_name.replace('.npy', '.png'))
                if os.path.exists(sky_mask_file):
                    sky_mask = cv2.imread(sky_mask_file, cv2.IMREAD_GRAYSCALE)
                    mask = (sky_mask > 0).astype(np.bool_)
                    final_mask = np.logical_and(consistency_masks[i], mask)
                else:
                    logger.warning(f"Sky mask not found: {sky_mask_file}, skipping sky filtering")
                    final_mask = consistency_masks[i]
            else:
                final_mask = consistency_masks[i]
            
            # Apply downscale mask
            final_mask = np.logical_and(downscale_mask, final_mask)
            
            # Extract valid pixels
            kept = np.argwhere(final_mask)
            
            if len(kept) == 0:
                continue
            
            depth_values = depth[kept[:, 0], kept[:, 1]]
            rgb_values = rgb[kept[:, 0], kept[:, 1]]
            
            # Filter invalid depth values
            valid_depth_mask = np.isfinite(depth_values) & (depth_values > 0)
            if not np.any(valid_depth_mask):
                continue
            
            depth_values = depth_values[valid_depth_mask]
            rgb_values = rgb_values[valid_depth_mask]
            kept_valid = kept[valid_depth_mask]
            
            try:
                c2w = pcd_generator.c2w[i]
                K = pcd_generator.intri[i]
            except IndexError as e:
                logger.warning(f"Index error for {file_name}: {e}")
                continue
            
            # Unproject to camera coordinates
            x = np.arange(0, W)
            y = np.arange(0, H)
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(H, W, 2)
            
            # Ensure K is 3x3
            if K.shape == (4, 4):
                K_3x3 = K[:3, :3]
            elif K.shape == (3, 3):
                K_3x3 = K
            else:
                K_3x3 = K[:3, :3] if K.shape[0] >= 3 and K.shape[1] >= 3 else K
                logger.warning(f"Unexpected intrinsics shape {K.shape} for {file_name}")
            
            pixel_coords = pixels[kept_valid[:, 0], kept_valid[:, 1]]
            x_cam = (pixel_coords[:, 0] - K_3x3[0, 2]) * depth_values / K_3x3[0, 0]
            y_cam = (pixel_coords[:, 1] - K_3x3[1, 2]) * depth_values / K_3x3[1, 1]
            z_cam = depth_values
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)
            
            # Filter NaN/inf coordinates
            valid_coords_mask = np.isfinite(coordinates).all(axis=1)
            if not np.any(valid_coords_mask):
                continue
            
            coordinates = coordinates[valid_coords_mask]
            rgb_values = rgb_values[valid_coords_mask]
            coordinates = np.column_stack((coordinates, np.ones(len(coordinates))))
            
            # Transform to world coordinates
            worlds = np.dot(c2w, coordinates.T).T
            worlds = worlds[:, :3]
            
            # Filter NaN/inf world coordinates
            valid_worlds_mask = np.isfinite(worlds).all(axis=1)
            if not np.any(valid_worlds_mask):
                continue
            
            worlds = worlds[valid_worlds_mask]
            rgb_values = rgb_values[valid_worlds_mask]
            
            # Accumulate point cloud chunk
            point_cloud_chunk = np.concatenate([worlds, rgb_values.reshape(-1, 3)], axis=-1)
            color_pointclouds.append(point_cloud_chunk)
        
        # Merge all point cloud chunks
        if len(color_pointclouds) == 0:
            raise ValueError("No valid point cloud generated")
        
        accumulated_pointcloud = np.concatenate(color_pointclouds, axis=0).reshape(-1, 6)
        
        # Final filtering: remove remaining NaN/inf values
        valid_mask = np.isfinite(accumulated_pointcloud[:, :3]).all(axis=1)
        if not np.any(valid_mask):
            raise ValueError("No valid point cloud after NaN filtering")
        
        accumulated_pointcloud = accumulated_pointcloud[valid_mask]
        
        logger.info(f"Accumulated point cloud: {len(accumulated_pointcloud)} points")
        
        # 11. Apply bounding box cropping if enabled
        points = accumulated_pointcloud[:, :3]
        colors = accumulated_pointcloud[:, 3:]
        
        if use_bbx and bbx_min is not None and bbx_max is not None:
            logger.info("Applying bounding box cropping...")
            points, colors = pcd_generator.crop_pointcloud(bbx_min, bbx_max, points, colors)
            logger.info(f"Points after bounding box: {len(points)}")
        
        # 12. Point cloud filtering
        logger.info("Applying point cloud filtering...")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal
        cl, ind = point_cloud.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=1.5
        )
        point_cloud = point_cloud.select_by_index(ind)
        
        # Uniform downsampling
        point_cloud = point_cloud.uniform_down_sample(every_k_points=3)
        
        logger.info(f"Final point cloud: {len(point_cloud.points)} points")
        
        # 13. Convert to torch.Tensor and set
        final_points = np.asarray(point_cloud.points)
        final_colors = np.asarray(point_cloud.colors)
        
        self.means = torch.from_numpy(final_points).float().to(self.device)
        self.anchor_feats = torch.from_numpy(final_colors).float().to(self.device)
        
        # Set initial scales (log scale)
        self.scales = torch.log(torch.ones_like(self.means) * initial_scale)
        self.offset = torch.zeros_like(self.means)
        
        logger.info(f"Initialized point cloud: {len(self.means)} points")
        logger.info(f"Point cloud range: X[{self.means[:, 0].min():.2f}, {self.means[:, 0].max():.2f}], "
                   f"Y[{self.means[:, 1].min():.2f}, {self.means[:, 1].max():.2f}], "
                   f"Z[{self.means[:, 2].min():.2f}, {self.means[:, 2].max():.2f}]")
        
        # 14. Clear depth map data to free memory
        del pcd_generator.c2w, pcd_generator.intri
        if hasattr(pcd_generator, '_last_depth'):
            del pcd_generator._last_depth
        del consistency_masks, color_pointclouds, accumulated_pointcloud
        logger.info("Cleared depth map data from memory")

    def init_gaussians_from_dataset(self, dataset):
        """
        Initialize seed points from dataset.
        
        This method calls init_mono_points_from_dataset to generate point cloud
        from monocular depth maps.
        
        Args:
            dataset: DrivingDataset instance
        """
        self.init_mono_points_from_dataset(dataset)

    def forward(
        self,
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: generate 3DGS parameters and render.
        
        Args:
            image_infos: Image and pixel information
            camera_infos: Camera information
            novel_view: Whether this is a novel view
            
        Returns:
            Dictionary of rendered outputs
        """
        # Check if point cloud is initialized
        if self.means is None:
            raise RuntimeError(
                "Point cloud not initialized. Please call init_gaussians_from_dataset() "
                "or manually set self.means, self.anchor_feats, self.scales, self.offset"
            )
        
        # Set bounding box if not set
        if self.bbx_min is None or self.bbx_max is None:
            self.bbx_min = self.aabb[0].to(self.device)
            self.bbx_max = self.aabb[1].to(self.device)
        
        # Convert drivestudio format to EVolSplat batch format
        batch = self._convert_to_evolsplat_batch(image_infos, camera_infos)
        
        # Generate Gaussian parameters using EVolSplat components
        gs_params = self._generate_gaussians_from_features(batch, camera_infos)
        
        # Convert to drivestudio format and cache
        self._cached_gaussians = self._convert_to_drivestudio_format(
            gs_params, camera_infos
        )
        
        # Call parent forward (which will call collect_gaussians and render_gaussians)
        return super().forward(image_infos, camera_infos, novel_view)

    def collect_gaussians(
        self,
        cam,
        image_ids: torch.Tensor
    ) -> dataclass_gs:
        """
        Override parent method: return EVolSplat-generated gaussians instead of collecting from nodes.
        
        Args:
            cam: Camera dataclass
            image_ids: Image indices (unused here)
            
        Returns:
            dataclass_gs: Gaussian parameters
        """
        if self._cached_gaussians is None:
            raise RuntimeError(
                "Gaussians not generated. Call forward() first."
            )
        
        return self._cached_gaussians

    def _convert_to_evolsplat_batch(
        self,
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor]
    ) -> Dict:
        """
        Convert drivestudio data format to EVolSplat batch format.
        
        This is a placeholder interface. If data format conversion is difficult,
        users can implement this method themselves.
        
        Expected EVolSplat batch format:
        - batch['source']['image']: [N_views, H, W, C] or [N_views, C, H, W]
        - batch['source']['extrinsics']: [N_views, 4, 4] camera-to-world
        - batch['source']['intrinsics']: [N_views, 4, 4] or [N_views, 3, 3]
        - batch['source']['depth']: [N_views, H, W] (optional)
        - batch['target']['image']: [H, W, C] (for loss computation)
        - batch['target']['intrinsics']: [4, 4] or [3, 3]
        
        Args:
            image_infos: drivestudio image information
            camera_infos: drivestudio camera information
            
        Returns:
            Dictionary in EVolSplat batch format
        """
        # TODO: Implement conversion
        # This is a placeholder - users should implement based on their data format
        
        logger.warning(
            "_convert_to_evolsplat_batch() is not implemented. "
            "Please implement data format conversion."
        )
        
        # Placeholder return
        return {
            'source': {
                'image': None,
                'extrinsics': None,
                'intrinsics': None,
                'depth': None,
            },
            'target': {
                'image': None,
                'intrinsics': None,
            }
        }

    def _generate_gaussians_from_features(
        self,
        batch: Dict,
        camera_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate Gaussian parameters using EVolSplat components.
        
        This method implements the core EVolSplat logic:
        1. Extract 3D features using sparse convolution
        2. Extract 2D features using projector
        3. Generate Gaussian parameters using MLP decoders
        
        Args:
            batch: EVolSplat batch format
            camera_infos: Camera information for target view
            
        Returns:
            Dictionary of Gaussian parameters:
            - means: [N, 3]
            - scales: [N, 3] (log scale)
            - quats: [N, 4]
            - opacities: [N, 1] (logit)
            - colors: [N, K, 3] (SH coefficients)
        """
        # Get point cloud data
        means = self.means.to(self.device)
        anchors_feat = self.anchor_feats.to(self.device)
        scales = self.scales.to(self.device)
        offset = self.offset.to(self.device)
        
        # Get source images and cameras
        source_images = batch['source']['image']  # [N_views, C, H, W] or [N_views, H, W, C]
        source_extrinsics = batch['source']['extrinsics']  # [N_views, 4, 4]
        source_intrinsics = batch['source']['intrinsics']  # [N_views, 4, 4] or [N_views, 3, 3]
        source_depth = batch['source'].get('depth', None)  # [N_views, H, W]
        
        # Ensure source_images is [N_views, C, H, W]
        if source_images.dim() == 4 and source_images.shape[-1] == 3:
            # [N_views, H, W, C] -> [N_views, C, H, W]
            source_images = source_images.permute(0, 3, 1, 2)
        
        # Get target camera
        target_camera_to_world = camera_infos['camtoworlds']  # [1, 4, 4] or [4, 4]
        if target_camera_to_world.dim() == 2:
            target_camera_to_world = target_camera_to_world.unsqueeze(0)
        
        # Step 1: Extract 3D features using sparse convolution
        if not self._frozen_volume or self.dense_volume is None:
            sparse_feat, vol_dim, valid_coords = construct_sparse_tensor(
                raw_coords=means.clone(),
                feats=anchors_feat,
                Bbx_max=self.bbx_max,
                Bbx_min=self.bbx_min,
                voxel_size=self.voxel_size,
            )
            feat_3d = self.sparse_conv(sparse_feat)
            dense_volume = sparse_to_dense_volume(
                sparse_tensor=feat_3d,
                coords=valid_coords,
                vol_dim=vol_dim
            ).unsqueeze(dim=0)
            self.dense_volume = rearrange(dense_volume, 'B H W D C -> B C H W D')
            self._cached_vol_dim = vol_dim
        else:
            # Use cached volume dimension
            vol_dim = self._cached_vol_dim
        
        # Step 2: Extract 2D features using projector
        sampled_feat, valid_mask, vis_map = self.projector.sample_within_window(
            xyz=means,
            train_imgs=source_images,  # [N_view, C, H, W]
            train_cameras=source_extrinsics,  # [N_view, 4, 4]
            train_intrinsics=source_intrinsics,  # [N_view, 4, 4] or [N_view, 3, 3]
            source_depth=source_depth,
            local_radius=self.local_radius,
        )  # [N_samples, N_views, local_h*local_w, 3]
        
        sampled_feat = torch.concat([sampled_feat, vis_map], dim=-1).reshape(
            -1, self.feature_dim_in
        )
        valid_mask = valid_mask.reshape(-1, self.feature_dim_in // 4)
        
        # Filter points with sufficient projections
        projection_mask = valid_mask[..., :].sum(dim=1) > self.local_radius ** 2 + 1
        num_points = projection_mask.sum()
        
        if num_points == 0:
            raise RuntimeError("No valid points after projection filtering")
        
        means_crop = means[projection_mask]
        sampled_color = sampled_feat[projection_mask]
        valid_scales = scales[projection_mask]
        last_offset = offset[projection_mask]
        
        # Step 3: Interpolate 3D features
        grid_coords = get_grid_coords(
            position_w=means_crop + last_offset,
            bbx_min=self.bbx_min,
            vol_dim=self._cached_vol_dim,
            voxel_size=self.voxel_size
        )
        feat_3d = interpolate_features(
            grid_coords=grid_coords,
            feature_volume=self.dense_volume
        ).permute(3, 4, 1, 0, 2).squeeze()
        
        # Step 4: Compute view-dependent features
        with torch.no_grad():
            ob_view = means_crop - target_camera_to_world[0, :3, 3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            ob_view = ob_view / ob_dist
        
        # Step 5: Generate Gaussian parameters using MLP decoders
        # Colors (SH coefficients)
        input_feature = torch.cat([sampled_color, ob_dist, ob_view], dim=-1).squeeze(dim=1)
        sh = self.gaussion_decoder(input_feature)  # [N, feature_dim_out]
        features_dc_crop = sh[:, :3]  # [N, 3]
        features_rest_crop = sh[:, 3:].reshape(num_points, -1, 3)  # [N, K-1, 3]
        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )  # [N, K, 3]
        
        # Scales and rotations
        scale_input_feat = torch.cat([feat_3d, ob_dist, ob_view], dim=-1).squeeze(dim=1)
        scales_crop, quats_crop = self.mlp_conv(scale_input_feat).split([3, 4], dim=-1)
        
        # Opacity
        opacities_crop = self.mlp_opacity(scale_input_feat)  # [N, 1]
        
        # Offset
        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        means_crop = means_crop + offset_crop
        
        # Update offset for next iteration (detached)
        if self.training:
            self.offset[projection_mask] = offset_crop.detach().cpu()
        
        return {
            'means': means_crop,
            'scales': scales_crop + valid_scales,  # Add base scales
            'quats': quats_crop,
            'opacities': opacities_crop,
            'colors': colors_crop,  # SH coefficients
        }

    def _convert_to_drivestudio_format(
        self,
        gs_params: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor]
    ) -> dataclass_gs:
        """
        Convert EVolSplat Gaussian parameters to drivestudio format.
        
        Args:
            gs_params: EVolSplat Gaussian parameters
            camera_infos: Camera information for view-dependent SH evaluation
            
        Returns:
            dataclass_gs: drivestudio Gaussian format
        """
        means = gs_params['means']
        scales_log = gs_params['scales']
        quats = gs_params['quats']
        opacities_logit = gs_params['opacities']
        colors_sh = gs_params['colors']  # [N, K, 3] SH coefficients
        
        # Convert scales: log -> exp
        scales = torch.exp(scales_log)
        
        # Normalize quaternions
        quats = quats / quats.norm(dim=-1, keepdim=True)
        
        # Convert opacity: logit -> sigmoid
        opacities = torch.sigmoid(opacities_logit)
        
        # Convert SH coefficients to RGB
        # Get view direction
        target_camera_to_world = camera_infos['camtoworlds']
        if target_camera_to_world.dim() == 2:
            target_camera_to_world = target_camera_to_world.unsqueeze(0)
        
        viewdirs = means.detach() - target_camera_to_world[0, :3, 3]
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        
        # Evaluate SH to get RGB
        # colors_sh: [N, K, 3] where K = num_sh_bases(sh_degree)
        # viewdirs: [N, 3]
        rgbs = spherical_harmonics(self.sh_degree, viewdirs, colors_sh)
        rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        
        return dataclass_gs(
            _means=means,
            _scales=scales,
            _quats=quats,
            _opacities=opacities,
            _rgbs=rgbs,
            detach_keys=[],
            extras=None
        )

