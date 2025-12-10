"""
EVolSplat Trainer for drivestudio.

This trainer integrates EVolSplat's feature extraction and Gaussian parameter generation
with drivestudio's rendering system.
"""
from typing import Dict, Optional
import logging
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

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

    def init_gaussians_from_dataset(self, dataset):
        """
        Initialize seed points from dataset.
        
        This is a placeholder interface. If data loading is difficult to map,
        users can implement this method themselves.
        
        Args:
            dataset: DrivingDataset instance
        """
        # TODO: Implement initialization from dataset
        # Expected format:
        # self.means = torch.Tensor([N, 3])  # Point cloud coordinates
        # self.anchor_feats = torch.Tensor([N, 3])  # Point cloud colors (RGB, 0-255 or 0-1)
        # self.scales = torch.Tensor([N, 3])  # Initial scales (log scale)
        # self.offset = torch.Tensor([N, 3])  # Initial offset (zeros)
        
        # Example (placeholder):
        # if hasattr(dataset, 'lidar_source'):
        #     # Get initial points from LiDAR
        #     pass
        
        logger.warning(
            "init_gaussians_from_dataset() is not implemented. "
            "Please initialize self.means, self.anchor_feats, self.scales, self.offset manually."
        )

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

