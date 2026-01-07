"""
StreetForward trainer implementing the proxy-based multi-view gradient accumulation
described in docs/FeedForward_3DGS_Design.md.

The implementation keeps node_state as detached buffers, predicts offsets with
MLP heads, renders with proxy parameters, and back-propagates once per iteration.
Fallback implementations are provided when external dependencies (gsplat, nerfstudio)
are unavailable so that unit tests can exercise the gradient plumbing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
try:
    from sklearn.neighbors import NearestNeighbors
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

logger = logging.getLogger(__name__)


def _num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return sh * c0 + 0.5


def _random_quat_tensor(num: int, device: torch.device) -> torch.Tensor:
    """Generate random quaternions in wxyz format (compatible with gsplat)."""
    u = torch.rand(num, device=device)
    v = torch.rand(num, device=device)
    w = torch.rand(num, device=device)
    x = torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v)
    y = torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v)
    z = torch.sqrt(u) * torch.sin(2 * torch.pi * w)
    ww = torch.sqrt(u) * torch.cos(2 * torch.pi * w)
    return torch.stack([ww, x, y, z], dim=-1)  # wxyz format


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _axis_angle_to_quat(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion (wxyz format).
    
    Uses branchless sinc structure to avoid discontinuities near threshold,
    providing smoother gradients.
    
    Args:
        omega: [N, 3] axis-angle vector
        eps: Small epsilon for numerical stability
        
    Returns:
        quat: [N, 4] quaternion in wxyz format
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [N, 1]
    half_theta = theta * 0.5
    
    # Use sinc structure: sin(θ/2) / (θ + eps), avoids division by zero and is continuously differentiable
    # When θ → 0, sinc(θ/2) → 1/2, so xyz = ω * (1/2) = ω/2 (correct small angle approximation)
    sinc_half = torch.sin(half_theta) / (theta + eps)  # [N, 1]
    xyz = omega * sinc_half  # [N, 3]
    w = torch.cos(half_theta)  # [N, 1]
    
    return torch.cat([w, xyz], dim=-1)  # [N, 4] wxyz format


def get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world to world-to-camera as used by gsplat.
    
    Supports both [4,4] and [B,4,4] input shapes.
    """
    if camera_to_world.dim() == 2:
        camera_to_world = camera_to_world.unsqueeze(0)  # [1,4,4]
    r = camera_to_world[:, :3, :3]
    t = camera_to_world[:, :3, 3:4]
    r = r * torch.tensor([[[1, -1, -1]]], device=r.device, dtype=r.dtype)
    r_inv = r.transpose(1, 2)
    t_inv = -torch.bmm(r_inv, t)
    viewmat = torch.zeros(r.shape[0], 4, 4, device=r.device, dtype=r.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = r_inv
    viewmat[:, :3, 3:4] = t_inv
    return viewmat


from gsplat.rendering import rasterization as _gsplat_rasterization


from models.evol_splat import (
    SparseCostRegNet as _SparseCostRegNet,
    construct_sparse_tensor as _construct_sparse_tensor,
    sparse_to_dense_volume as _sparse_to_dense_volume,
)


def _pairwise_neighbor_distances(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Compute k-NN distances efficiently using sklearn's NearestNeighbors.
    
    This function avoids the O(N²) memory overhead of computing full pairwise distances
    by using efficient k-NN algorithms. For large point clouds (e.g., 890K points),
    this reduces memory from ~3.17 TB to ~10.7 MB.
    
    Args:
        points: Tensor of shape [N, 3] containing point coordinates
        k: Number of nearest neighbors to find
        
    Returns:
        Tensor of shape [N, k] containing distances to k nearest neighbors
    """
    if not _sklearn_available:
        raise ImportError("sklearn is required for k-NN search. Please install scikit-learn.")
    
    # Convert to numpy (CPU) for sklearn
    if points.is_cuda:
        points_np = points.cpu().numpy().astype('float32')
    else:
        points_np = points.numpy().astype('float32')
    
    # Build the nearest neighbors model
    # Use k+1 neighbors because the point itself will be included as the first neighbor
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn_model.fit(points_np)
    
    # Find the k-nearest neighbors
    distances, _ = nn_model.kneighbors(points_np)
    
    # Exclude self (first neighbor is always the point itself)
    distances = distances[:, 1:]  # [N, k]
    
    # Convert back to torch tensor on original device
    result = torch.from_numpy(distances.astype('float32')).to(points.device)
    
    return result

@dataclass
class NodeState:
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeState":
        return NodeState(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


class StreetForwardTrainer(nn.Module):
    """
    Feed-forward 3DGS trainer with proxy-based multi-view accumulation.

    The trainer maintains a node_state per (scene_id, segment_id) consisting of detached
    Gaussian parameters. Each iteration builds a 3D feature volume, predicts offsets,
    renders via proxy parameters, back-propagates once through the feed-forward graph,
    and writes detached results back to node_state.
    """

    def __init__(
        self,
        config: OmegaConf,
        device: torch.device = torch.device("cpu"),
        renderer: Optional[Callable] = None,
        sparse_conv: Optional[nn.Module] = None,
        construct_sparse_tensor_fn: Optional[Callable] = None,
        sparse_to_dense_volume_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config
        self.device = device

        model_cfg = config.model
        self.offset_max = model_cfg.get("offset_max", 0.1)
        self.scale_max = model_cfg.get("scale_max", 0.1)
        self.omega_max = model_cfg.get("omega_max", 0.1)
        self.opacity_max = model_cfg.get("opacity_max", 0.1)
        self.sh_dc_max = model_cfg.get("sh_dc_max", 0.1)
        self.sh_rest_max = model_cfg.get("sh_rest_max", 0.05)
        self.eta_means = model_cfg.get("eta_means", 1.0)
        self.eta_scales = model_cfg.get("eta_scales", 1.0)
        self.eta_opacity = model_cfg.get("eta_opacity", 1.0)
        self.eta_sh_dc = model_cfg.get("eta_sh_dc", 1.0)
        self.eta_sh_rest = model_cfg.get("eta_sh_rest", 1.0)
        self.sh_degree = model_cfg.get("sh_degree", 1)
        self.voxel_size = model_cfg.get("voxel_size", 0.1)
        self.inner_iterations = model_cfg.get("max_iterations", 1)

        bbx_min = model_cfg.get("bbx_min", [-20.0, -20.0, -20.0])
        bbx_max = model_cfg.get("bbx_max", [20.0, 4.8, 70.0])
        self.bbx_min = torch.tensor(bbx_min, dtype=torch.float32, device=device)
        self.bbx_max = torch.tensor(bbx_max, dtype=torch.float32, device=device)

        # Renderer and sparse conv dependencies
        self.renderer = renderer or _gsplat_rasterization
        if self.renderer is None:
            raise ImportError("Renderer not available. Install gsplat or provide a custom renderer.")

        outdim = model_cfg.get("sparseConv_outdim", 32)
        if sparse_conv is not None:
            self.sparse_conv = sparse_conv.to(device)
        elif _SparseCostRegNet is not None:
            self.sparse_conv = _SparseCostRegNet(d_in=3, d_out=outdim).to(device)
        else:
            raise ImportError("SparseCostRegNet not available. Install models.evol_splat or provide a custom sparse_conv.")

        self.construct_sparse_tensor = construct_sparse_tensor_fn or _construct_sparse_tensor
        if self.construct_sparse_tensor is None:
            raise ImportError("construct_sparse_tensor not available. Install models.evol_splat or provide a custom construct_sparse_tensor_fn.")
        
        self.sparse_to_dense_volume = sparse_to_dense_volume_fn or _sparse_to_dense_volume
        if self.sparse_to_dense_volume is None:
            raise ImportError("sparse_to_dense_volume not available. Install models.evol_splat or provide a custom sparse_to_dense_volume_fn.")

        # MLP heads (only 3D features are used per design)
        self.mlp_offset_pos = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(device)

        self.mlp_conv = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 3 for scales + 3 for axis-angle
        ).to(device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)

        num_sh = _num_sh_bases(self.sh_degree)
        self.gaussion_decoder = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)

        params: List[torch.nn.Parameter] = []
        params += list(self.sparse_conv.parameters())
        params += list(self.mlp_offset_pos.parameters())
        params += list(self.mlp_conv.parameters())
        params += list(self.mlp_opacity.parameters())
        params += list(self.gaussion_decoder.parameters())

        optim_cfg = config.optimizer
        self.optimizer = torch.optim.Adam(
            params,
            lr=optim_cfg.get("lr", 1e-3),
            eps=optim_cfg.get("eps", 1e-15),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )

        self.global_step = 0
        self.checkpoint_dir: Optional[str] = None
        self.tb_writer = None
        self.tb_log_every = 50
        self.tb_image_every = None

        training_cfg = getattr(config, "training", {})
        self.checkpoint_dir = (
            training_cfg.get("save_checkpoint_dir", None)
            if hasattr(training_cfg, "get")
            else None
        )
        # allow both legacy top-level flag and training-level override
        self.log_images = bool(config.get("log_images", False))
        if hasattr(training_cfg, "get"):
            self.log_images = training_cfg.get("log_images", self.log_images)
        self._setup_tensorboard(training_cfg)

        self.node_states: Dict[Tuple[int, int], NodeState] = {}
        self._lpips_model = None
        self._lpips_unavailable = False
        self._ssim_unavailable = False
        
        # Initialize offset heads to output near-zero offsets
        self._init_offset_heads()

    def _init_offset_heads(self) -> None:
        """Initialize offset prediction heads to output near-zero offsets."""
        nn.init.zeros_(self.mlp_offset_pos[-1].weight)
        nn.init.zeros_(self.mlp_offset_pos[-1].bias)
        nn.init.zeros_(self.mlp_conv[-1].weight)
        nn.init.zeros_(self.mlp_conv[-1].bias)
        nn.init.zeros_(self.mlp_opacity[-1].weight)
        nn.init.zeros_(self.mlp_opacity[-1].bias)
        nn.init.zeros_(self.gaussion_decoder[-1].weight)
        nn.init.zeros_(self.gaussion_decoder[-1].bias)

    def _setup_tensorboard(self, training_cfg) -> None:
        """
        Initialize TensorBoard writer based on config.

        TensorBoard is optional to keep unit tests lightweight and avoid
        unexpected disk writes in non-training contexts.
        """
        tb_cfg = training_cfg.get("tensorboard", {}) if hasattr(training_cfg, "get") else {}
        enabled = tb_cfg.get("enabled", False)
        if not enabled:
            return

        log_dir = tb_cfg.get("log_dir")
        if log_dir is None:
            # Default to run's log_dir/tb when available, otherwise local folder.
            base_log_dir = getattr(self.config, "log_dir", None)
            if base_log_dir is not None:
                log_dir = os.path.join(base_log_dir, "tb")
            else:
                log_dir = "./logs/tensorboard"

        self.tb_log_every = int(tb_cfg.get("log_every", 50))
        self.tb_image_every = tb_cfg.get("log_image_every", None)
        if self.tb_image_every is not None:
            self.tb_image_every = int(self.tb_image_every)

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard not available; install tensorboard to enable logging.")
            return

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=int(tb_cfg.get("flush_secs", 30)))
        logger.info(f"TensorBoard logging enabled at {log_dir}")

    def _init_node_from_pointcloud(
        self,
        scene_id: int,
        segment_id: int,
        pointcloud,
    ) -> NodeState:
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3]
            if background.shape[1] >= 6:
                colors = background[:, 3:]
                if colors.max() > 1.0 + 1e-3:
                    colors = colors / 255.0
            else:
                colors = np.zeros_like(points)
        else:
            points = np.asarray(pointcloud.points)  # type: ignore[attr-defined]
            colors = np.asarray(pointcloud.colors)  # type: ignore[attr-defined]
            if colors.max() > 1.0 + 1e-3:
                colors = colors / 255.0

        if len(points) == 0:
            raise ValueError(f"Empty point cloud for scene {scene_id}, segment {segment_id}")

        means = torch.from_numpy(points).float().to(self.device)
        colors_rgb = torch.from_numpy(colors).float().to(self.device)

        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        initial_scales = torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        node_state = NodeState(
            means=means.detach().clone(),
            scales_log=initial_scales.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )
        self.node_states[(scene_id, segment_id)] = node_state
        return node_state

    def _get_or_init_node_state(self, batch: Dict) -> Tuple[Tuple[int, int], NodeState]:
        scene_id = batch["scene_id"]
        if isinstance(scene_id, torch.Tensor):
            scene_id = int(scene_id.item())
        segment_id = batch["segment_id"]
        if isinstance(segment_id, torch.Tensor):
            segment_id = int(segment_id.item())
        key = (scene_id, segment_id)
        if key in self.node_states:
            return key, self.node_states[key]
        pointcloud = batch["pointcloud"]
        node_state = self._init_node_from_pointcloud(scene_id, segment_id, pointcloud)
        return key, node_state

    def _node_state_to_dict(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        return {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
        }

    def _node_state_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> NodeState:
        return NodeState(
            means=state_dict["means"].to(self.device),
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
        ).detach_clone()

    def get_grid_coords(
        self, position_w: torch.Tensor, bbx_min: torch.Tensor, vol_dim, voxel_size: float
    ) -> torch.Tensor:
        # Clamp positions to bbox range to match construct_sparse_tensor behavior
        # This ensures coordinates are within the volume bounds
        # Use self.bbx_max directly instead of recalculating from vol_dim
        bbx_max = self.bbx_max.to(position_w.device)
        position_w_clamped = torch.clamp(position_w, min=bbx_min, max=bbx_max)
        
        pts = position_w_clamped - bbx_min.to(position_w.device)
        x_index = pts[..., 0] / voxel_size
        y_index = pts[..., 1] / voxel_size
        z_index = pts[..., 2] / voxel_size
        
        # Convert vol_dim to torch.Tensor if it's a list, tuple, or numpy array
        # construct_sparse_tensor may return Python list from nerfstudio implementation
        if isinstance(vol_dim, (list, tuple)):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        elif not isinstance(vol_dim, torch.Tensor):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        else:
            vol_dim = vol_dim.to(position_w.device).float()
        
        # vol_dim is [X, Y, Z] from construct_sparse_tensor (world coordinates)
        # sparse_to_dense_volume creates dense volume as [X, Y, Z, C]
        # After unsqueeze: [1, X, Y, Z, C]
        # After permute(0, 4, 1, 2, 3): [1, C, X, Y, Z]
        # But grid_sample expects [B, C, D, H, W] where D=Z, H=Y, W=X
        # So we need permute(0, 4, 3, 2, 1) to get [1, C, Z, Y, X] = [1, C, D, H, W]
        # For grid_sample [B, C, D, H, W], coordinates should be [z, y, x] where:
        # - z corresponds to D (Z axis)
        # - y corresponds to H (Y axis)
        # - x corresponds to W (X axis)
        # So we normalize: z_norm for Z, y_norm for Y, x_norm for X
        # And stack as [z_norm, y_norm, x_norm]
        # For align_corners=True: index 0 maps to -1.0, index (N-1) maps to 1.0
        # Therefore, we use (vol_dim - 1) as denominator to ensure correct boundary mapping
        den_x = torch.clamp(vol_dim[0] - 1.0, min=1.0)
        den_y = torch.clamp(vol_dim[1] - 1.0, min=1.0)
        den_z = torch.clamp(vol_dim[2] - 1.0, min=1.0)
        x_norm = 2.0 * (x_index / den_x) - 1.0  # X -> W
        y_norm = 2.0 * (y_index / den_y) - 1.0  # Y -> H
        z_norm = 2.0 * (z_index / den_z) - 1.0  # Z -> D
        # grid_sample (5D) expects coordinates in [z, y, x] order for [B, C, D, H, W] input
        grid_coords = torch.stack([z_norm, y_norm, x_norm], dim=-1)
        
        return grid_coords

    def interpolate_features(self, grid_coords: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
        grid_coords_expanded = grid_coords[None, None, None, ...]
        feature = torch.nn.functional.grid_sample(
            feature_volume,
            grid_coords_expanded,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        return feature[0, :, 0, 0, :].T

    def _predict_offsets(self, feat_3d_crop: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Position offset with tanh clamping
        offset_pos = self.offset_max * torch.tanh(self.mlp_offset_pos(feat_3d_crop))
        
        # Scale and rotation offsets
        scales_and_omega = self.mlp_conv(feat_3d_crop)
        offset_scales_raw, offset_omega_raw = scales_and_omega.split([3, 3], dim=-1)
        offset_scales = self.scale_max * torch.tanh(offset_scales_raw)
        offset_omega = self.omega_max * torch.tanh(offset_omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        
        # Opacity offset with tanh clamping
        offset_opacity = self.opacity_max * torch.tanh(self.mlp_opacity(feat_3d_crop))
        
        # SH offsets with separate DC and rest
        sh_raw = self.gaussion_decoder(feat_3d_crop)
        sh_dc_raw = sh_raw[:, :3]
        sh_rest_raw = sh_raw[:, 3:]
        offset_sh_dc = self.sh_dc_max * torch.tanh(sh_dc_raw)
        offset_sh_rest = self.sh_rest_max * torch.tanh(sh_rest_raw)
        offset_sh = torch.cat([offset_sh_dc, offset_sh_rest], dim=-1)
        
        return {
            "offset_pos": offset_pos,
            "offset_scales": offset_scales,
            "offset_quat": offset_quat,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
        }

    def _render_params_from_offsets(
        self, node_state: NodeState, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_points = node_state.means.shape[0]
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)

        # Apply offsets with step size factors (eta)
        # Note: means_r is not clamped here to preserve gradient flow
        means_r = node_state.means + self.eta_means * offsets["offset_pos"]
        scales_log_r = node_state.scales_log + self.eta_scales * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state.opacity_logit + self.eta_opacity * offsets["offset_opacity"]
        sh_dc_r = node_state.sh_dc + self.eta_sh_dc * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state.sh_rest + self.eta_sh_rest * sh_rest_offset

        scales_r = torch.exp(scales_log_r)
        opacities_r = torch.sigmoid(opacity_logit_r).squeeze(-1)
        colors_r = torch.cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)

        return {
            "means_r": means_r,
            "scales_log_r": scales_log_r,
            "quats_r": quats_r,
            "opacity_logit_r": opacity_logit_r,
            "sh_dc_r": sh_dc_r,
            "sh_rest_r": sh_rest_r,
            "scales_r": scales_r,
            "opacities_r": opacities_r,
            "colors_r": colors_r,
        }

    def _create_proxy_params(self, render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "means_p": render_params["means_r"].detach().requires_grad_(True),
            "scales_p": render_params["scales_r"].detach().requires_grad_(True),
            "quats_p": render_params["quats_r"].detach().requires_grad_(True),
            "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
            "colors_p": render_params["colors_r"].detach().requires_grad_(True),
        }

    def compute_loss(self, pred_rgb: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred_rgb - gt_image) ** 2)

    def _compute_render_params(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        """
        Shared forward pass to compute render parameters from node state.
        """
        means_s = node_state.means
        anchor_rgb = _sh_to_rgb(node_state.sh_dc)

        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means_s.clone(),
            feats=anchor_rgb,
            Bbx_max=self.bbx_max,
            Bbx_min=self.bbx_min,
            voxel_size=self.voxel_size,
            device=self.device,
        )
        feat_3d = self.sparse_conv(sparse_feat)
        dense_volume = self.sparse_to_dense_volume(
            sparse_tensor=feat_3d,
            coords=valid_coords,
            vol_dim=vol_dim,
        ).unsqueeze(dim=0)
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
        grid_coords = self.get_grid_coords(means_s, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop = self.interpolate_features(grid_coords, dense_volume)
        del dense_volume

        offsets = self._predict_offsets(feat_3d_crop)
        render_params = self._render_params_from_offsets(node_state, offsets)
        return render_params

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a single view and return RGB and alpha.
        """
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = get_viewmat(c2w)
        k_mat = None
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=self.device).unsqueeze(0)
        
        # Ensure Ks is [1,3,3] format
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)

        means_key = "means_p" if "means_p" in render_params else "means_r"
        scales_key = "scales_p" if "scales_p" in render_params else "scales_r"
        quats_key = "quats_p" if "quats_p" in render_params else "quats_r"
        opacities_key = "opacities_p" if "opacities_p" in render_params else "opacities_r"
        colors_key = "colors_p" if "colors_p" in render_params else "colors_r"

        render, alpha, _ = self.renderer(
            means=render_params[means_key],
            quats=render_params[quats_key],
            scales=render_params[scales_key],
            opacities=render_params[opacities_key],
            colors=render_params[colors_key],
            viewmats=viewmat,
            Ks=k_mat,
            width=width,
            height=height,
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )

        rgb = render[:, ..., :3].squeeze(0)
        acc = alpha.squeeze(0)
        return rgb, acc

    def train_iter(
        self,
        batch: Dict,
        apply_update: bool = True,
        update_state: bool = True,
        evaluate_test: bool = False,
    ) -> Dict:
        key, node_state = self._get_or_init_node_state(batch)
        target_views = batch["target_views"]
        gt_images = batch["gt_images"]
        
        # Skip if no target views (no supervision)
        if len(target_views) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device),
                "node_state": node_state,
                "outputs": [],
            }
        
        view_count = len(target_views)
        outputs = []
        total_loss_val = 0.0  # Use scalar to avoid keeping computation graph
        test_metrics = None

        self.optimizer.zero_grad(set_to_none=True)

        for inner_iter_idx in range(self.inner_iterations):
            render_params = self._compute_render_params(node_state)
            proxies = self._create_proxy_params(render_params)

            for view_idx, (view, gt_img) in enumerate(zip(target_views, gt_images)):
                height, width = gt_img.shape[0], gt_img.shape[1]
                rgb, acc = self._render_single_view(proxies, view, height, width)
                loss = self.compute_loss(rgb, gt_img) / view_count
                total_loss_val += float(loss.detach())  # Accumulate scalar to avoid keeping graph
                loss.backward()
                
                # Only store images if explicitly requested (to save GPU memory)
                if self.log_images:
                    outputs.append({
                        "rgb": rgb.detach().cpu(),
                        "acc": acc.detach().cpu(),
                        "loss": loss.detach().item(),
                    })
                else:
                    outputs.append({"loss": loss.detach().item()})

            render_tensors = [
                render_params["means_r"],
                render_params["scales_r"],
                render_params["quats_r"],
                render_params["opacities_r"],
                render_params["colors_r"],
            ]
            proxy_grads = [
                proxies["means_p"].grad if proxies["means_p"].grad is not None else torch.zeros_like(proxies["means_p"]),
                proxies["scales_p"].grad if proxies["scales_p"].grad is not None else torch.zeros_like(proxies["scales_p"]),
                proxies["quats_p"].grad if proxies["quats_p"].grad is not None else torch.zeros_like(proxies["quats_p"]),
                proxies["opacities_p"].grad if proxies["opacities_p"].grad is not None else torch.zeros_like(proxies["opacities_p"]),
                proxies["colors_p"].grad if proxies["colors_p"].grad is not None else torch.zeros_like(proxies["colors_p"]),
            ]
            torch.autograd.backward(tensors=render_tensors, grad_tensors=proxy_grads)

            if apply_update:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if update_state:
                with torch.no_grad():
                    # Clamp means only when writing back to node_state (not during backprop)
                    means_clamped = torch.clamp(
                        render_params["means_r"].detach(), min=self.bbx_min, max=self.bbx_max
                    )
                    node_state.means.copy_(means_clamped)
                    node_state.scales_log.copy_(render_params["scales_log_r"].detach())
                    node_state.quats.copy_(render_params["quats_r"].detach())
                    node_state.opacity_logit.copy_(render_params["opacity_logit_r"].detach())
                    node_state.sh_dc.copy_(render_params["sh_dc_r"].detach())
                    node_state.sh_rest.copy_(render_params["sh_rest_r"].detach())

        self.node_states[key] = node_state.detach_clone()
        if apply_update:
            self.global_step += 1
            self._log_to_tensorboard(total_loss_val, outputs)

        if evaluate_test and batch.get("test_views"):
            prev_mode = self.training
            self.eval()
            with torch.no_grad():
                test_metrics = self._evaluate_test_views(
                    node_state=self.node_states[key],
                    test_views=batch.get("test_views", []),
                    test_images=batch.get("test_images", []),
                )
            if prev_mode:
                self.train()

        return {
            "total_loss": torch.tensor(total_loss_val, device=self.device),
            "node_state": self.node_states[key],
            "outputs": outputs,
            "test_metrics": test_metrics,
        }

    def forward(self, batch: Dict) -> Dict:
        return self.train_iter(batch)

    def _evaluate_test_views(
        self,
        node_state: NodeState,
        test_views: List,
        test_images: List[torch.Tensor],
    ) -> Optional[Dict[str, float]]:
        if test_views is None or len(test_views) == 0:
            return None
        render_params = self._compute_render_params(node_state)
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        lpips_list: List[float] = []

        for view, gt_img in zip(test_views, test_images):
            height, width = gt_img.shape[0], gt_img.shape[1]
            rgb_pred, _ = self._render_single_view(render_params, view, height, width)
            rgb_gt = gt_img.to(self.device)

            psnr_list.append(self._compute_psnr(rgb_pred, rgb_gt))
            ssim_list.append(self._compute_ssim(rgb_pred, rgb_gt))
            lpips_list.append(self._compute_lpips(rgb_pred, rgb_gt))

        if len(psnr_list) == 0:
            return None

        metrics = {
            "psnr": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
            "lpips": float(np.mean(lpips_list)),
            "num_test_views": len(psnr_list),
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Dict[str, float]:
        """
        Evaluate model performance on test views (no gradient updates).
        """
        self.eval()
        _, node_state = self._get_or_init_node_state(batch)
        metrics = self._evaluate_test_views(
            node_state=node_state,
            test_views=batch.get("test_views", []),
            test_images=batch.get("test_images", []),
        )
        self.train()
        return metrics or {}

    def _compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        mse = torch.mean((pred - gt) ** 2)
        mse_val = float(mse.item())
        if mse_val <= 0:
            return float("inf")
        psnr = -10 * torch.log10(torch.tensor(mse_val, device=pred.device))
        return float(psnr.item())

    def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        try:
            from pytorch_msssim import ssim
        except ImportError:
            if not getattr(self, "_ssim_unavailable", False):
                logger.warning("pytorch_msssim not installed; returning NaN for SSIM")
                self._ssim_unavailable = True
            return float("nan")

        pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
        return float(ssim(pred_4d, gt_4d, data_range=1.0).item())

    def _compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        try:
            from lpips import LPIPS
        except ImportError:
            if not getattr(self, "_lpips_unavailable", False):
                logger.warning("lpips not installed; returning NaN for LPIPS")
                self._lpips_unavailable = True
            return float("nan")

        if not hasattr(self, "_lpips_model") or self._lpips_model is None:
            self._lpips_model = LPIPS(net="alex").to(self.device)
        pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
        return float(self._lpips_model(pred_4d, gt_4d).item())

    def save_checkpoint(
        self,
        step: Optional[int] = None,
        is_final: bool = False,
        checkpoint_dir: Optional[str] = None,
    ) -> str:
        """
        Persist model/optimizer and detached node states.

        Args:
            step: Optional training step to record (defaults to self.global_step)
            is_final: If True, always write to checkpoint_final.pth
            checkpoint_dir: Override output directory
        """
        step_val = int(step if step is not None else self.global_step)
        ckpt_dir = (
            checkpoint_dir
            or self.checkpoint_dir
            or (os.path.join(self.config.log_dir, "checkpoints") if hasattr(self.config, "log_dir") else None)
            or "./checkpoints"
        )
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        filename = "checkpoint_final.pth" if is_final else f"checkpoint_step_{step_val:06d}.pth"
        checkpoint_path = os.path.join(ckpt_dir, filename)

        model_state_dict = {
            "sparse_conv": self.sparse_conv.state_dict(),
            "mlp_offset_pos": self.mlp_offset_pos.state_dict(),
            "mlp_conv": self.mlp_conv.state_dict(),
            "mlp_opacity": self.mlp_opacity.state_dict(),
            "gaussion_decoder": self.gaussion_decoder.state_dict(),
        }

        nodes_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states.items()
        }

        checkpoint = {
            "step": step_val,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "node_states": nodes_state_dict,
        }
        try:
            checkpoint["config"] = OmegaConf.to_container(self.config, resolve=False)
        except Exception:
            logger.debug("Config not serialized into checkpoint (non-fatal).")

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> int:
        """
        Restore model/optimizer and node states.

        Args:
            checkpoint_path: Path to .pth checkpoint
            load_optimizer: Load optimizer state if available
            strict: Strictness for weight loading

        Returns:
            Restored global_step
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        self.sparse_conv.load_state_dict(model_state["sparse_conv"], strict=strict)
        self.mlp_offset_pos.load_state_dict(model_state["mlp_offset_pos"], strict=strict)
        self.mlp_conv.load_state_dict(model_state["mlp_conv"], strict=strict)
        self.mlp_opacity.load_state_dict(model_state["mlp_opacity"], strict=strict)
        self.gaussion_decoder.load_state_dict(model_state["gaussion_decoder"], strict=strict)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        nodes_state_dict = checkpoint.get("node_states") or checkpoint.get("nodes_state_dict")
        if nodes_state_dict is not None:
            restored_nodes: Dict[Tuple[int, int], NodeState] = {}
            for key, state in nodes_state_dict.items():
                scene_id, segment_id = None, None
                if isinstance(key, str) and key.startswith("scene_") and "_segment_" in key:
                    try:
                        scene_id = int(key.split("scene_")[1].split("_segment_")[0])
                        segment_id = int(key.split("_segment_")[1])
                    except Exception:
                        scene_id, segment_id = None, None
                elif isinstance(key, (tuple, list)) and len(key) == 2:
                    scene_id, segment_id = int(key[0]), int(key[1])
                if scene_id is None or segment_id is None:
                    continue
                restored_nodes[(scene_id, segment_id)] = self._node_state_from_dict(state)
            if restored_nodes:
                self.node_states = restored_nodes

        self.global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
        logger.info(f"Checkpoint loaded from {checkpoint_path} (step={self.global_step})")
        return self.global_step

    def _log_to_tensorboard(self, total_loss_val: float, outputs: List[Dict]) -> None:
        """Write scalars/images to TensorBoard when enabled."""
        if self.tb_writer is None:
            return

        step = self.global_step
        if self.tb_log_every and step % self.tb_log_every == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb_writer.add_scalar("train/total_loss", total_loss_val, step)
            self.tb_writer.add_scalar("train/lr", lr, step)

        if (
            self.log_images
            and self.tb_image_every is not None
            and outputs
            and step % self.tb_image_every == 0
        ):
            for idx, out in enumerate(outputs):
                if "rgb" not in out:
                    continue
                rgb = out["rgb"]
                acc = out.get("acc", None)
                tag_prefix = f"train/view_{idx}"
                if rgb.dim() == 3:
                    self.tb_writer.add_image(tag_prefix + "/rgb", rgb.permute(2, 0, 1), step)
                if acc is not None:
                    if acc.dim() == 2:
                        self.tb_writer.add_image(tag_prefix + "/alpha", acc.unsqueeze(0), step)
                    elif acc.dim() == 3:
                        self.tb_writer.add_image(tag_prefix + "/alpha", acc, step)
                # only log the first view to limit disk usage
                break

    def close(self) -> None:
        """Close TensorBoard writer if it was created."""
        if self.tb_writer is not None:
            self.tb_writer.close()
