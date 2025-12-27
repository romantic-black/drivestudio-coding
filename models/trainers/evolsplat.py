"""
EVolsplatTrainer for feed-forward 3DGS training.

This module implements a trainer class for EVolsplat feed-forward 3DGS training,
supporting multi-scene, multi-segment training with RGB point cloud initialization.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_msssim import SSIM
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.gaussians.vanilla import VanillaGaussians
from models.gaussians.basics import k_nearest_sklearn, RGB2SH, random_quat_tensor

if TYPE_CHECKING:
    from datasets.multi_scene_dataset import MultiSceneDataset
    from datasets.pointcloud_generators.rgb_pointcloud_generator import RGBPointCloudGenerator

# Import EVolsplat components
try:
    from gsplat.rendering import rasterization
    from gsplat.cuda_legacy._wrapper import num_sh_bases
    from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
except ImportError:
    raise ImportError("Please install gsplat>=1.0.0")

from nerfstudio.model_components.projection import Projector
from nerfstudio.field_components.mlp import MLP
from nerfstudio.model_components.sparse_conv import (
    sparse_to_dense_volume,
    SparseCostRegNet,
    construct_sparse_tensor,
)
from nerfstudio.fields.initial_BgSphere import GaussianBGInitializer

logger = logging.getLogger(__name__)


def get_viewmat(optimized_camera_to_world):
    """
    Convert c2w to gsplat world2camera matrix.
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


class EVolsplatTrainer(nn.Module):
    """
    EVolsplat feed-forward 3DGS trainer.
    
    Core functionality:
    1. Manage multi-scene, multi-segment training flow
    2. Initialize 3DGS nodes from RGB point clouds
    3. Predict Gaussian parameters via MLP (feed-forward)
    4. Support multi-target image training
    5. Segment-level offset management
    """
    
    def __init__(
        self,
        dataset: "MultiSceneDataset",
        pointcloud_generator: "RGBPointCloudGenerator",
        config: OmegaConf,
        device: torch.device = torch.device("cuda"),
        log_dir: str = "./logs",
    ):
        """
        Initialize EVolsplatTrainer.
        
        Args:
            dataset: MultiSceneDataset instance
            pointcloud_generator: RGBPointCloudGenerator instance
            config: Training configuration (OmegaConf)
            device: Training device
            log_dir: Log directory
        """
        super().__init__()
        
        self.dataset = dataset
        self.pointcloud_generator = pointcloud_generator
        self.config = config
        self.device = device
        self.log_dir = log_dir
        
        # Training step counter
        self.step = 0
        
        # Segment-level node and cache management
        # Key: (scene_id, segment_id)
        self.nodes: Dict[Tuple[int, int], VanillaGaussians] = {}
        self.offset_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.frozen_volume_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Current segment info (for tracking)
        self.current_scene_id: Optional[int] = None
        self.current_segment_id: Optional[int] = None
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
        logger.info("EVolsplatTrainer initialized")
    
    def _init_networks(self):
        """Initialize all MLP networks."""
        model_cfg = self.config.model
        
        # Sparse convolution network
        self.sparse_conv = SparseCostRegNet(
            d_in=3,
            d_out=model_cfg.sparseConv_outdim
        ).to(self.device)
        
        # Projector for 2D feature sampling
        self.projector = Projector()
        
        # Feature dimensions
        self.local_radius = model_cfg.get("local_radius", 1)
        
        # Calculate number of source views from dataset configuration
        # source views = num_source_keyframes * num_cams
        num_source_keyframes = self.dataset.num_source_keyframes
        
        # Get number of cameras from dataset
        # Try to get from first scene if available
        num_cams = None
        if hasattr(self.dataset, 'train_scene_ids') and len(self.dataset.train_scene_ids) > 0:
            try:
                # Try to get num_cams from dataset's scene data
                scene_data = self.dataset._ensure_scene_loaded(self.dataset.train_scene_ids[0])
                if scene_data is not None and 'num_cams' in scene_data:
                    num_cams = scene_data['num_cams']
                    logger.info(f"Got num_cams={num_cams} from scene {self.dataset.train_scene_ids[0]}")
            except Exception as e:
                logger.debug(f"Could not get num_cams from scene data: {e}")
        
        # Fallback: get from config if available
        if num_cams is None:
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'pixel_source'):
                cameras = self.config.data.pixel_source.get('cameras', [0, 1, 2])
                num_cams = len(cameras) if isinstance(cameras, list) else 1
                logger.info(f"Using num_cams from config: {num_cams}")
            else:
                # Default fallback
                num_cams = 3
                logger.warning(f"Could not determine num_cams, using default: {num_cams}")
        
        # Number of source views = num_source_keyframes * num_cams
        self.num_source_views = num_source_keyframes * num_cams
        
        # Also calculate num_target_views for reference
        num_target_keyframes = self.dataset.num_target_keyframes
        self.num_target_views = num_target_keyframes * num_cams
        
        # feature_dim_in = 4 * num_source_views * (2*local_radius+1)^2
        # where 4 = 3 (RGB) + 1 (visibility), num_source_views = actual number of source views
        self.feature_dim_in = 4 * self.num_source_views * (2 * self.local_radius + 1) ** 2
        self.feature_dim_out = 3 * num_sh_bases(model_cfg.sh_degree)
        
        logger.info(
            f"Feature dimensions: num_source_views={self.num_source_views} "
            f"(keyframes={num_source_keyframes} × cams={num_cams}), "
            f"local_radius={self.local_radius}, "
            f"feature_dim_in={self.feature_dim_in}"
        )
        
        # Gaussian appearance decoder (predicts SH coefficients)
        self.gaussion_decoder = MLP(
            in_dim=self.feature_dim_in + 4,
            num_layers=3,
            layer_width=128,
            out_dim=self.feature_dim_out,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        ).to(self.device)
        
        # Scale and rotation MLP
        self.mlp_conv = MLP(
            in_dim=model_cfg.sparseConv_outdim + 4,
            num_layers=2,
            layer_width=64,
            out_dim=3 + 4,  # 3 for scale, 4 for quaternion
            activation=nn.Tanh(),
            out_activation=None,
            implementation="torch",
        ).to(self.device)
        
        # Opacity MLP
        self.mlp_opacity = MLP(
            in_dim=model_cfg.sparseConv_outdim + 4,
            num_layers=2,
            layer_width=64,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="torch",
        ).to(self.device)
        
        # Offset MLP
        self.mlp_offset = MLP(
            in_dim=model_cfg.sparseConv_outdim,
            num_layers=2,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Tanh(),
            implementation="torch",
        ).to(self.device)
        
        # Background field (optional)
        self.bg_field = None
        if model_cfg.get("enable_background", True):
            self.bg_field = MLP(
                in_dim=9,
                num_layers=2,
                layer_width=64,
                out_dim=6,  # 3 for RGB, 3 for scale
                activation=nn.ReLU(),
                out_activation=nn.Tanh(),
                implementation="torch",
            ).to(self.device)
        
        # Model parameters
        self.offset_max = model_cfg.offset_max
        self.voxel_size = model_cfg.voxel_size
        self.sh_degree = model_cfg.sh_degree
        self.freeze_volume = model_cfg.get("freeze_volume", False)
        
        # Bounding box
        if hasattr(model_cfg, "bbx_min") and hasattr(model_cfg, "bbx_max"):
            self.bbx_min = torch.tensor(model_cfg.bbx_min, dtype=torch.float32).to(self.device)
            self.bbx_max = torch.tensor(model_cfg.bbx_max, dtype=torch.float32).to(self.device)
        else:
            # Default bounding box
            self.bbx_min = torch.tensor([-20.0, -20.0, -5.0], dtype=torch.float32).to(self.device)
            self.bbx_max = torch.tensor([20.0, 4.8, 20.0], dtype=torch.float32).to(self.device)
        
        logger.info("Networks initialized")
    
    def _init_optimizer(self):
        """Initialize optimizer, learning rate scheduler, and mixed precision scaler."""
        optim_cfg = self.config.optimizer
        
        # Collect all trainable parameters
        params = []
        params.extend(list(self.sparse_conv.parameters()))
        params.extend(list(self.gaussion_decoder.parameters()))
        params.extend(list(self.mlp_conv.parameters()))
        params.extend(list(self.mlp_opacity.parameters()))
        params.extend(list(self.mlp_offset.parameters()))
        if self.bg_field is not None:
            params.extend(list(self.bg_field.parameters()))
        
        # Create optimizer
        optimizer_type = optim_cfg.get("type", "Adam")
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=optim_cfg.get("lr", 0.001),
                eps=optim_cfg.get("eps", 1e-15),
                weight_decay=optim_cfg.get("weight_decay", 0.0),
            )
        elif optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=optim_cfg.get("lr", 0.001),
                eps=optim_cfg.get("eps", 1e-15),
                weight_decay=optim_cfg.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Learning rate scheduler (optional)
        self.scheduler = None
        if "scheduler" in optim_cfg:
            scheduler_cfg = optim_cfg.scheduler
            if scheduler_cfg.get("type") == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_cfg.get("T_max", 30000),
                    eta_min=scheduler_cfg.get("eta_min", 0.0),
                )
        
        # Mixed precision scaler
        self.scaler = None
        if self.config.training.get("use_mixed_precision", False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Optimizer initialized: {optimizer_type}")
    
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        params = []
        params.extend(list(self.sparse_conv.parameters()))
        params.extend(list(self.gaussion_decoder.parameters()))
        params.extend(list(self.mlp_conv.parameters()))
        params.extend(list(self.mlp_opacity.parameters()))
        params.extend(list(self.mlp_offset.parameters()))
        if self.bg_field is not None:
            params.extend(list(self.bg_field.parameters()))
        return params
    
    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        
        Args:
            x: The data tensor of shape [num_samples, num_features]
            k: The number of neighbors to retrieve
            
        Returns:
            distances: [num_samples, k] - distances to k nearest neighbors
            indices: [num_samples, k] - indices of k nearest neighbors
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()
        
        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
        
        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)
        
        # Exclude the point itself from the result
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
    def init_node_from_pointcloud(
        self,
        scene_id: int,
        segment_id: int,
        pointcloud: o3d.geometry.PointCloud,
    ) -> VanillaGaussians:
        """
        Initialize VanillaGaussians node from RGB point cloud.
        
        Args:
            scene_id: Scene ID
            segment_id: Segment ID
            pointcloud: Open3D point cloud with positions and colors
            
        Returns:
            VanillaGaussians node
        """
        # Extract point cloud data
        points = np.asarray(pointcloud.points)  # [N, 3]
        colors = np.asarray(pointcloud.colors)  # [N, 3]
        
        if len(points) == 0:
            raise ValueError(f"Empty point cloud for scene {scene_id}, segment {segment_id}")
        
        # Convert to tensor
        means = torch.from_numpy(points).float().to(self.device)
        colors_rgb = torch.from_numpy(colors).float().to(self.device)
        
        # Create VanillaGaussians node configuration
        from omegaconf import OmegaConf
        
        ctrl_cfg = OmegaConf.create({
            "sh_degree": self.sh_degree,
            "ball_gaussians": False,
            "gaussian_2d": False,
        })
        
        # Create node
        node = VanillaGaussians(
            class_name="Background",
            ctrl=ctrl_cfg,
            scene_scale=30.0,
            scene_origin=torch.zeros(3, device=self.device),
            num_train_images=300,
            device=self.device,
        )
        
        # Initialize from point cloud (uses mixed initialization strategy)
        node.create_from_pcd(means, colors_rgb)
        
        # Calculate initial scales using KNN (EVolsplat style)
        distances, _ = self.k_nearest_sklearn(means.cpu(), k=3)
        distances = torch.from_numpy(distances).to(self.device)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        initial_scales = torch.log(avg_dist.repeat(1, 3))
        
        # Store initial scales (for later use in MLP prediction)
        node._initial_scales = initial_scales
        
        # Initialize offset to zero
        offset = torch.zeros_like(means)
        
        # Store node and offset
        self.nodes[(scene_id, segment_id)] = node
        self.offset_cache[(scene_id, segment_id)] = offset
        
        logger.info(
            f"Initialized node for scene {scene_id}, segment {segment_id} "
            f"with {len(points)} points"
        )
        
        return node
    
    def _check_and_init_segment(self, batch: Dict):
        """
        Check if segment needs initialization and initialize if needed.
        
        Args:
            batch: Training batch with scene_id and segment_id
        """
        scene_id = batch["scene_id"].item() if isinstance(batch["scene_id"], torch.Tensor) else batch["scene_id"]
        segment_id = batch["segment_id"]
        
        segment_key = (scene_id, segment_id)
        
        # Check if node already exists
        if segment_key in self.nodes:
            self.current_scene_id = scene_id
            self.current_segment_id = segment_id
            return
        
        # Generate point cloud
        logger.info(f"Generating point cloud for scene {scene_id}, segment {segment_id}")
        pointcloud = self.pointcloud_generator.generate_pointcloud(
            self.dataset,
            scene_id,
            segment_id,
        )
        
        # Initialize node
        self.init_node_from_pointcloud(scene_id, segment_id, pointcloud)
        
        self.current_scene_id = scene_id
        self.current_segment_id = segment_id
    
    def get_grid_coords(self, position_w: torch.Tensor):
        """
        Convert world coordinates to grid coordinates for trilinear interpolation.
        
        Args:
            position_w: [N, 3] world coordinates
            
        Returns:
            grid_coords: [N, 3] normalized grid coordinates in [-1, 1]
        """
        # Get volume dimensions (will be set during 3D feature extraction)
        if not hasattr(self, "vol_dim"):
            raise RuntimeError("Volume dimensions not set. Call extract_3d_features_for_target first.")
        
        bounding_min = self.bbx_min
        pts = position_w - bounding_min.to(position_w.device)
        
        x_index = pts[..., 0] / self.voxel_size
        y_index = pts[..., 1] / self.voxel_size
        z_index = pts[..., 2] / self.voxel_size
        
        dhw = torch.stack([x_index, y_index, z_index], dim=1)
        
        # Normalize to [-1, 1]
        dhw[..., 0] = dhw[..., 0] / self.vol_dim[0] * 2 - 1
        dhw[..., 1] = dhw[..., 1] / self.vol_dim[1] * 2 - 1
        dhw[..., 2] = dhw[..., 2] / self.vol_dim[2] * 2 - 1
        
        # Reorder to [z, y, x] for grid_sample
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords
    
    def interpolate_features(self, grid_coords: torch.Tensor, feature_volume: torch.Tensor):
        """
        Trilinear interpolation of features from dense volume.
        
        Args:
            grid_coords: [N, 3] normalized grid coordinates
            feature_volume: [1, C, H, W, D] dense feature volume
            
        Returns:
            features: [N, C] interpolated features
        """
        grid_coords = grid_coords[None, None, None, ...]  # [1, 1, 1, N, 3]
        feature = torch.nn.functional.grid_sample(
            feature_volume,
            grid_coords,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        return feature
    
    def extract_reusable_2d_features(
        self,
        batch: Dict,
        node: VanillaGaussians,
        offset: torch.Tensor,
    ) -> Dict:
        """
        Extract reusable 2D features from source views (executed once, shared by all target views).
        
        These features can be safely detached as they don't involve trainable networks.
        They only depend on source views and can be reused across all target views.
        
        Args:
            batch: Training batch
            node: Current segment's node
            offset: Current segment's offset
            
        Returns:
            reusable_2d_features: Dictionary containing sampled_feat, valid_mask, projection_mask, means, etc.
        """
        # Get means
        means = node._means  # [N, 3]
        
        # Prepare source images
        source_images = batch["source"]["image"]  # [num_source_keyframes * num_cams, H, W, 3]
        source_images = rearrange(source_images[None, ...], "b v h w c -> b v c h w")
        source_extrinsics = batch["source"]["extrinsics"]  # [num_source_keyframes * num_cams, 4, 4]
        source_intrinsics = batch["source"]["intrinsics"]  # [num_source_keyframes * num_cams, 4, 4]
        source_depth = batch["source"]["depth"]  # [num_source_keyframes * num_cams, H, W]
        
        # 2D feature sampling (reusable across all target views)
        sampled_feat, valid_mask, vis_map = self.projector.sample_within_window(
            xyz=means,
            train_imgs=source_images.squeeze(0),  # [N_view, c, h, w]
            train_cameras=source_extrinsics,  # [N_view, 4, 4]
            train_intrinsics=source_intrinsics,  # [N_view, 4, 4]
            source_depth=source_depth,
            local_radius=self.local_radius,
        )
        
        # Concatenate sampled features with visibility map
        # According to EVolsplat implementation:
        # sampled_feat: [N, num_views, (2R+1)^2, 3] - RGB features from window
        # vis_map: [N, num_views, (2R+1)^2, 1] - visibility map
        # After concat: [N, num_views, (2R+1)^2, 4]
        # Reshape to: [N, num_views * (2R+1)^2 * 4] = [N, feature_dim_in]
        # where feature_dim_in = 4 * num_source_views * (2*local_radius+1)^2
        # and num_source_views = num_source_keyframes * num_cams
        
        # Get actual number of views from sampled_feat
        num_points = sampled_feat.shape[0]
        num_views = sampled_feat.shape[1]
        window_size = sampled_feat.shape[2]  # (2R+1)^2
        
        # Verify that actual views match expected
        if num_views != self.num_source_views:
            raise ValueError(
                f"Source view count mismatch: expected {self.num_source_views} views "
                f"(keyframes={self.dataset.num_source_keyframes} × cams), "
                f"got {num_views} views from sample_within_window. "
                f"Please check dataset configuration."
            )
        
        # Calculate feature dimension (should match self.feature_dim_in)
        actual_feature_dim_in = 4 * num_views * window_size
        
        # Verify dimension matches
        if actual_feature_dim_in != self.feature_dim_in:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim_in} "
                f"(based on num_source_views={self.num_source_views}, local_radius={self.local_radius}), "
                f"got {actual_feature_dim_in} (based on num_views={num_views}, window_size={window_size}). "
                f"This indicates a configuration error."
            )
        
        # Concatenate along last dimension
        sampled_feat = torch.concat([sampled_feat, vis_map], dim=-1)  # [N, num_views, (2R+1)^2, 4]
        
        # Reshape to [N, feature_dim_in]
        # The reshape flattens: num_views * (2R+1)^2 * 4
        sampled_feat = sampled_feat.reshape(num_points, self.feature_dim_in)  # [N, feature_dim_in]
        
        # Reshape valid_mask: [N, num_views, (2R+1)^2] -> [N, num_views * (2R+1)^2]
        valid_mask = valid_mask.reshape(num_points, -1)  # [N, num_views * (2R+1)^2]
        
        # Create projection mask
        projection_mask = valid_mask[..., :].sum(dim=1) > self.local_radius ** 2 + 1
        
        reusable_2d_features = {
            "sampled_feat": sampled_feat,
            "valid_mask": valid_mask,
            "projection_mask": projection_mask,
            "means": means,
        }
        
        return reusable_2d_features
    
    def extract_3d_features_for_target(
        self,
        batch: Dict,
        node: VanillaGaussians,
        offset: torch.Tensor,
        means_crop: torch.Tensor,
        last_offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract 3D features for a specific target view (needs gradients).
        
        This function recomputes feat_3d through sparse_conv for each target view,
        ensuring gradients can flow back to sparse_conv parameters.
        
        Args:
            batch: Training batch
            node: Current segment's node
            offset: Current segment's offset
            means_crop: Cropped means [num_points, 3] (after projection_mask)
            last_offset: Last offset for cropped points [num_points, 3]
            
        Returns:
            feat_3d: 3D features for cropped points [num_points, C] (with gradients)
        """
        scene_id = batch["scene_id"].item() if isinstance(batch["scene_id"], torch.Tensor) else batch["scene_id"]
        
        # Get means and anchor features
        means = node._means  # [N, 3]
        anchor_feats = node._features_dc  # [N, 3] - RGB converted to SH
        
        # Convert anchor_feats back to RGB for feature extraction
        # (EVolsplat uses RGB as anchor features)
        if node.sh_degree > 0:
            from models.gaussians.basics import SH2RGB
            anchor_feats_rgb = SH2RGB(anchor_feats)  # [N, 3]
        else:
            anchor_feats_rgb = torch.sigmoid(anchor_feats)  # [N, 3]
        
        # Build 3D feature volume (recompute for each target view to maintain gradients)
        # Note: We don't use freeze_volume cache here because we need gradients
        sparse_feat, self.vol_dim, self.valid_coords = construct_sparse_tensor(
            raw_coords=means.clone(),
            feats=anchor_feats_rgb,
            Bbx_max=self.bbx_max,
            Bbx_min=self.bbx_min,
            voxel_size=self.voxel_size,
        )
        feat_3d = self.sparse_conv(sparse_feat)  # This needs gradients!
        dense_volume = sparse_to_dense_volume(
            sparse_tensor=feat_3d,
            coords=self.valid_coords,
            vol_dim=self.vol_dim,
        ).unsqueeze(dim=0)
        dense_volume = rearrange(dense_volume, "B H W D C -> B C H W D")
        
        # Trilinear interpolation of 3D features for cropped points
        grid_coords = self.get_grid_coords(means_crop + last_offset)
        feat_3d_crop = self.interpolate_features(
            grid_coords=grid_coords,
            feature_volume=dense_volume,
        ).permute(3, 4, 1, 0, 2).squeeze()  # [num_points, C]
        
        return feat_3d_crop
    
    def render_for_target_view(
        self,
        target_view: Dict,
        reusable_2d_features: Dict,
        node: VanillaGaussians,
        offset: torch.Tensor,
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Render image for a single target view.
        
        Args:
            target_view: Target view data (image, camera parameters, etc.)
            reusable_2d_features: Reusable 2D features from extract_reusable_2d_features
            node: Current segment's node
            offset: Current segment's offset
            batch: Training batch (needed for extract_3d_features_for_target)
            
        Returns:
            outputs: Dictionary with rgb, depth, accumulation, etc.
        """
        # Extract reusable 2D features
        sampled_feat = reusable_2d_features["sampled_feat"]
        projection_mask = reusable_2d_features["projection_mask"]
        means = reusable_2d_features["means"]
        
        # Get target camera parameters
        target_extrinsic = target_view["extrinsic"]  # [4, 4]
        target_intrinsic = target_view["intrinsic"]  # [4, 4] or [3, 3]
        target_image = target_view["image"]  # [H, W, 3]
        
        # Convert to camera format
        optimized_camera_to_world = target_extrinsic[None, ...]  # [1, 4, 4]
        
        # Crop based on projection mask
        num_points = projection_mask.sum()
        means_crop = means[projection_mask]
        sampled_color = sampled_feat[projection_mask]
        last_offset = offset[projection_mask]
        
        # Get initial scales from node
        if hasattr(node, "_initial_scales"):
            valid_scales = node._initial_scales[projection_mask]
        else:
            # Fallback: compute from KNN
            distances, _ = self.k_nearest_sklearn(means_crop.cpu(), k=3)
            distances = torch.from_numpy(distances).to(self.device)
            avg_dist = distances.mean(dim=-1, keepdim=True)
            valid_scales = torch.log(avg_dist.repeat(1, 3))
        
        # Extract 3D features for this target view (recomputes with gradients)
        feat_3d = self.extract_3d_features_for_target(
            batch=batch,
            node=node,
            offset=offset,
            means_crop=means_crop,
            last_offset=last_offset,
        )
        
        # Compute ob_view and ob_dist
        with torch.no_grad():
            ob_view = means_crop - optimized_camera_to_world[0, :3, 3]
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            ob_view = ob_view / ob_dist
        
        # Predict SH coefficients (color)
        input_feature = torch.cat([sampled_color, ob_dist, ob_view], dim=-1).squeeze(dim=1)
        sh = self.gaussion_decoder(input_feature)
        features_dc_crop = sh[:, :3]
        features_rest_crop = sh[:, 3:].reshape(num_points, -1, 3)
        
        # Predict scale, rotation, and opacity
        scale_input_feat = torch.cat([feat_3d, ob_dist, ob_view], dim=-1).squeeze(dim=1)
        scales_crop, quats_crop = self.mlp_conv(scale_input_feat).split([3, 4], dim=-1)
        opacities_crop = self.mlp_opacity(scale_input_feat)
        
        # Predict offset
        offset_crop = self.offset_max * self.mlp_offset(feat_3d)
        means_crop = means_crop + offset_crop
        
        # Prepare colors for rasterization
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        
        # Rasterization
        BLOCK_WIDTH = 16
        viewmat = get_viewmat(optimized_camera_to_world)
        
        # Extract intrinsic matrix
        if target_intrinsic.shape[-1] == 4:
            K = target_intrinsic[..., :3, :3]
        else:
            K = target_intrinsic
        
        K = K[None, ...]  # [1, 3, 3]
        H, W = target_image.shape[:2]
        
        render_mode = "RGB"
        if not self.training:
            render_mode = "RGB+ED"
        
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop + valid_scales),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )
        
        alpha = alpha[:, ...][0]
        render_rgb = render[:, ..., :3].squeeze(0)
        
        # Background rendering (if enabled)
        background_rgb = None
        if self.bg_field is not None:
            # TODO: Implement background rendering if needed
            # For now, use black background
            background_rgb = torch.zeros_like(render_rgb)
        
        # Composite final image
        if background_rgb is not None:
            rgb = render_rgb + (1 - alpha) * background_rgb
        else:
            rgb = render_rgb
        
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        # Depth (if available)
        depth_im = None
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        
        outputs = {
            "rgb": rgb,
            "depth": depth_im,
            "accumulation": alpha,
            "background": (1 - alpha) * background_rgb if background_rgb is not None else None,
            "offset_crop": offset_crop,  # For offset update
            "projection_mask": projection_mask,  # For offset update
        }
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            outputs: Rendered outputs
            gt_image: Ground truth image [H, W, 3]
            
        Returns:
            loss_dict: Dictionary of losses
        """
        pred_rgb = outputs["rgb"]
        accumulation = outputs["accumulation"]
        
        # L1 loss
        l1_loss = torch.abs(gt_image - pred_rgb).mean()
        
        # SSIM loss
        ssim_lambda = self.config.loss.get("ssim_lambda", 0.2)
        ssim_loss = 1 - self.ssim(
            gt_image.permute(2, 0, 1)[None, ...],
            pred_rgb.permute(2, 0, 1)[None, ...],
        )
        
        # Entropy loss (every 10 steps)
        entropy_loss_weight = self.config.loss.get("entropy_loss", 0.1)
        if self.step % 10 == 0:
            entropy_loss = entropy_loss_weight * (
                -accumulation * torch.log(accumulation + 1e-10)
                - (1 - accumulation) * torch.log(1 - accumulation + 1e-10)
            ).mean()
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss (include entropy loss)
        main_loss = (1 - ssim_lambda) * l1_loss + ssim_lambda * ssim_loss + entropy_loss
        
        loss_dict = {
            "main_loss": main_loss,
            "l1_loss": l1_loss,
            "ssim_loss": ssim_loss,
            "entropy_loss": entropy_loss,
        }
        
        return loss_dict
    
    def update_offset(
        self,
        scene_id: int,
        segment_id: int,
        offset_crop: torch.Tensor,
        projection_mask: torch.Tensor,
    ):
        """
        Update offset for current segment.
        
        Args:
            scene_id: Scene ID
            segment_id: Segment ID
            offset_crop: Predicted offset [num_points, 3] (with gradients)
            projection_mask: Projection mask [N] (boolean)
        """
        # Detach and move to CPU
        offset_crop = offset_crop.detach().cpu()
        
        segment_key = (scene_id, segment_id)
        
        # Initialize offset cache if needed
        if segment_key not in self.offset_cache:
            node = self.nodes[segment_key]
            self.offset_cache[segment_key] = torch.zeros_like(node._means.cpu())
        
        # Update offset at projected points
        self.offset_cache[segment_key][projection_mask] = offset_crop
        
        # Update node's offset attribute if it exists
        node = self.nodes[segment_key]
        if hasattr(node, "offset"):
            node.offset = self.offset_cache[segment_key].clone().to(self.device)
    
    def _get_target_view(self, batch: Dict, target_idx: int) -> Dict:
        """
        Extract single target view from batch.
        
        Args:
            batch: Training batch
            target_idx: Target view index
            
        Returns:
            target_view: Dictionary with image, extrinsic, intrinsic
        """
        target_images = batch["target"]["image"]  # [num_target_keyframes * num_cams, H, W, 3]
        target_extrinsics = batch["target"]["extrinsics"]  # [num_target_keyframes * num_cams, 4, 4]
        target_intrinsics = batch["target"]["intrinsics"]  # [num_target_keyframes * num_cams, 4, 4]
        
        return {
            "image": target_images[target_idx],
            "extrinsic": target_extrinsics[target_idx],
            "intrinsic": target_intrinsics[target_idx],
        }
    
    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Execute one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            loss_dict: Dictionary of losses
        """
        # 1. Check and initialize segment if needed
        self._check_and_init_segment(batch)
        
        scene_id = batch["scene_id"].item() if isinstance(batch["scene_id"], torch.Tensor) else batch["scene_id"]
        segment_id = batch["segment_id"]
        
        # 2. Get node and offset
        segment_key = (scene_id, segment_id)
        node = self.nodes[segment_key]
        offset = self.offset_cache[segment_key].to(self.device)
        
        # 3. Extract reusable 2D features (only once, can be detached)
        reusable_2d_features = self.extract_reusable_2d_features(batch, node, offset)
        
        # Detach reusable 2D features (they don't involve trainable networks)
        reusable_2d_features_detached = {}
        for k, v in reusable_2d_features.items():
            if isinstance(v, torch.Tensor):
                reusable_2d_features_detached[k] = v.detach()
            else:
                reusable_2d_features_detached[k] = v
        
        # 4. Optimizer zero grad
        self.optimizer.zero_grad()
        
        # 5. Loop over each target view
        num_target_views = batch["target"]["image"].shape[0]
        total_loss = 0.0
        
        # Aggregate losses from all target views
        aggregated_loss_dict = {
            "main_loss": 0.0,
            "l1_loss": 0.0,
            "ssim_loss": 0.0,
            "entropy_loss": 0.0,
        }
        
        # Aggregate offsets from all target views
        # Initialize offset accumulator and counter
        means = reusable_2d_features_detached["means"]
        num_all_points = means.shape[0]
        offset_accumulator = torch.zeros(num_all_points, 3, device=self.device)
        offset_counter = torch.zeros(num_all_points, device=self.device, dtype=torch.long)
        
        use_mixed_precision = self.config.training.get("use_mixed_precision", False)
        
        for target_idx in range(num_target_views):
            target_view = self._get_target_view(batch, target_idx)
            
            # Render (feat_3d will be recomputed inside render_for_target_view with gradients)
            autocast_context = torch.cuda.amp.autocast() if use_mixed_precision else nullcontext()
            with autocast_context:
                outputs = self.render_for_target_view(
                    target_view, reusable_2d_features_detached, node, offset, batch
                )
                loss_dict = self.compute_loss(outputs, target_view["image"])
                loss = loss_dict["main_loss"] / num_target_views
            
            # Backward (no retain_graph needed since 3D features are recomputed for each view)
            if use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # Aggregate losses
            for key in aggregated_loss_dict.keys():
                aggregated_loss_dict[key] += loss_dict[key].item() / num_target_views
            
            # Aggregate offsets from this view
            offset_crop = outputs["offset_crop"].detach()  # [num_points_crop, 3]
            projection_mask = outputs["projection_mask"]  # [num_all_points]
            
            # Accumulate offsets for points that are projected in this view
            offset_accumulator[projection_mask] += offset_crop
            offset_counter[projection_mask] += 1
        
        # 6. Gradient clipping (if enabled)
        gradient_clip_val = self.config.training.get("gradient_clip_val", None)
        if gradient_clip_val is not None:
            if use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.get_trainable_parameters(),
                gradient_clip_val,
            )
        
        # 7. Optimizer step
        if use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # 8. Update offset (aggregated from all target views)
        # Compute average offset for points that were projected in at least one view
        has_projection = offset_counter > 0
        if has_projection.any():
            # Average offset for points projected in multiple views
            averaged_offset = torch.zeros_like(offset_accumulator)
            averaged_offset[has_projection] = (
                offset_accumulator[has_projection] / offset_counter[has_projection].float().unsqueeze(-1)
            )
            
            # Update offset cache with aggregated offsets
            self.update_offset(scene_id, segment_id, averaged_offset[has_projection], has_projection)
        
        # Update step counter
        self.step += 1
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Convert aggregated losses to tensors
        aggregated_loss_dict_tensors = {
            key: torch.tensor(value, device=self.device) 
            for key, value in aggregated_loss_dict.items()
        }
        
        return {
            "total_loss": torch.tensor(total_loss, device=self.device),
            **aggregated_loss_dict_tensors,
        }
    
    def save_checkpoint(self, step: int, is_final: bool = False):
        """
        Save checkpoint.
        
        Args:
            step: Current step number
            is_final: Whether this is the final checkpoint
        """
        checkpoint_dir = self.config.training.get("save_checkpoint_dir", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_final:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pth")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step:06d}.pth")
        
        # Collect model state dict
        model_state_dict = {
            "sparse_conv": self.sparse_conv.state_dict(),
            "gaussion_decoder": self.gaussion_decoder.state_dict(),
            "mlp_conv": self.mlp_conv.state_dict(),
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "mlp_offset": self.mlp_offset.state_dict(),
        }
        if self.bg_field is not None:
            model_state_dict["bg_field"] = self.bg_field.state_dict()
        
        # Collect node state dicts
        nodes_state_dict = {}
        for key, node in self.nodes.items():
            scene_id, segment_id = key
            nodes_state_dict[f"scene_{scene_id}_segment_{segment_id}"] = {
                "means": node._means.cpu(),
                "scales": node._scales.cpu() if hasattr(node, "_scales") else None,
                "features_dc": node._features_dc.cpu(),
                "features_rest": node._features_rest.cpu() if hasattr(node, "_features_rest") else None,
                "opacities": node._opacities.cpu() if hasattr(node, "_opacities") else None,
                "quats": node._quats.cpu() if hasattr(node, "_quats") else None,
            }
            if hasattr(node, "_initial_scales"):
                nodes_state_dict[f"scene_{scene_id}_segment_{segment_id}"]["initial_scales"] = node._initial_scales.cpu()
        
        checkpoint = {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "nodes_state_dict": nodes_state_dict,
            "offset_cache": {str(k): v.cpu() for k, v in self.offset_cache.items()},
            "frozen_volume_cache": {
                str(k): v.cpu() for k, v in self.frozen_volume_cache.items()
            },
            "config": self.config,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_only_model: bool = False) -> int:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_only_model: Whether to load only model (not optimizer, etc.)
            
        Returns:
            step: Restored step number
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model state
        model_state_dict = checkpoint["model_state_dict"]
        self.sparse_conv.load_state_dict(model_state_dict["sparse_conv"])
        self.gaussion_decoder.load_state_dict(model_state_dict["gaussion_decoder"])
        self.mlp_conv.load_state_dict(model_state_dict["mlp_conv"])
        self.mlp_opacity.load_state_dict(model_state_dict["mlp_opacity"])
        self.mlp_offset.load_state_dict(model_state_dict["mlp_offset"])
        if self.bg_field is not None and "bg_field" in model_state_dict:
            self.bg_field.load_state_dict(model_state_dict["bg_field"])
        
        if not load_only_model:
            # Restore optimizer state
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Restore scaler state
            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            # Restore scheduler state
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore nodes
        if "nodes_state_dict" in checkpoint:
            nodes_state_dict = checkpoint["nodes_state_dict"]
            for key_str, node_state in nodes_state_dict.items():
                # Parse key: "scene_{scene_id}_segment_{segment_id}"
                parts = key_str.split("_")
                scene_id = int(parts[1])
                segment_id = int(parts[3])
                segment_key = (scene_id, segment_id)
                
                # Create node configuration
                from omegaconf import OmegaConf
                ctrl_cfg = OmegaConf.create({
                    "sh_degree": self.sh_degree,
                    "ball_gaussians": False,
                    "gaussian_2d": False,
                })
                
                # Create node
                node = VanillaGaussians(
                    class_name="Background",
                    ctrl=ctrl_cfg,
                    scene_scale=30.0,
                    scene_origin=torch.zeros(3, device=self.device),
                    num_train_images=300,
                    device=self.device,
                )
                
                # Restore node state
                if "means" in node_state:
                    node._means = node_state["means"].to(self.device)
                if "scales" in node_state and node_state["scales"] is not None:
                    node._scales = node_state["scales"].to(self.device)
                if "features_dc" in node_state and node_state["features_dc"] is not None:
                    node._features_dc = node_state["features_dc"].to(self.device)
                if "features_rest" in node_state and node_state["features_rest"] is not None:
                    node._features_rest = node_state["features_rest"].to(self.device)
                if "opacities" in node_state and node_state["opacities"] is not None:
                    node._opacities = node_state["opacities"].to(self.device)
                if "quats" in node_state and node_state["quats"] is not None:
                    node._quats = node_state["quats"].to(self.device)
                if "initial_scales" in node_state and node_state["initial_scales"] is not None:
                    node._initial_scales = node_state["initial_scales"].to(self.device)
                
                # Store node
                self.nodes[segment_key] = node
                
                logger.info(f"Restored node for scene {scene_id}, segment {segment_id}")
        
        # Restore offset cache
        if "offset_cache" in checkpoint:
            self.offset_cache = {}
            for k, v in checkpoint["offset_cache"].items():
                # Parse key: "(scene_id, segment_id)"
                import ast
                key = ast.literal_eval(k)
                self.offset_cache[key] = v.to(self.device)
        
        # Restore frozen volume cache
        if "frozen_volume_cache" in checkpoint:
            self.frozen_volume_cache = {}
            for k, v in checkpoint["frozen_volume_cache"].items():
                # Parse key: "(scene_id, segment_id)"
                import ast
                key = ast.literal_eval(k)
                self.frozen_volume_cache[key] = v.to(self.device)
        
        step = checkpoint.get("step", 0)
        self.step = step
        
        logger.info(f"Checkpoint loaded, restored to step {step}")
        return step
    
    def evaluate(self, batch: Dict) -> Dict[str, float]:
        """
        Evaluate model on a batch.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            metrics: Dictionary of metrics (PSNR, SSIM, LPIPS)
        """
        self.eval()
        
        with torch.no_grad():
            # Check and initialize segment if needed
            self._check_and_init_segment(batch)
            
            scene_id = batch["scene_id"].item() if isinstance(batch["scene_id"], torch.Tensor) else batch["scene_id"]
            segment_id = batch["segment_id"]
            
            # Get node and offset
            segment_key = (scene_id, segment_id)
            node = self.nodes[segment_key]
            offset = self.offset_cache[segment_key].to(self.device)
            
            # Extract reusable 2D features
            reusable_2d_features = self.extract_reusable_2d_features(batch, node, offset)
            
            # Render first target view (for evaluation)
            target_view = self._get_target_view(batch, 0)
            outputs = self.render_for_target_view(target_view, reusable_2d_features, node, offset, batch)
            
            # Compute metrics
            pred_rgb = outputs["rgb"]
            gt_rgb = target_view["image"]
            
            # Switch to [1, C, H, W] for metrics
            pred_rgb_metric = pred_rgb.permute(2, 0, 1)[None, ...]
            gt_rgb_metric = gt_rgb.permute(2, 0, 1)[None, ...]
            
            psnr = self.psnr(pred_rgb_metric, gt_rgb_metric)
            ssim = self.ssim(pred_rgb_metric, gt_rgb_metric)
            lpips = self.lpips(pred_rgb_metric, gt_rgb_metric)
            
            metrics = {
                "psnr": float(psnr.item()),
                "ssim": float(ssim.item()),
                "lpips": float(lpips.item()),
            }
        
        self.train()
        return metrics
    
    def set_train(self):
        """Set model to training mode."""
        self.train()
        self.sparse_conv.train()
        self.gaussion_decoder.train()
        self.mlp_conv.train()
        self.mlp_opacity.train()
        self.mlp_offset.train()
        if self.bg_field is not None:
            self.bg_field.train()
    
    def set_eval(self):
        """Set model to evaluation mode."""
        self.eval()
        self.sparse_conv.eval()
        self.gaussion_decoder.eval()
        self.mlp_conv.eval()
        self.mlp_opacity.eval()
        self.mlp_offset.eval()
        if self.bg_field is not None:
            self.bg_field.eval()

