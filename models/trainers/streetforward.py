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
import json
import time
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

# Debug logging configuration
_DEBUG_LOG_PATH = "/root/drivestudio-coding/.cursor/debug.log"

def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = None, run_id: str = "initial"):
    """Write debug log entry in NDJSON format."""
    try:
        entry = {
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "sessionId": "debug-session",
            "runId": run_id,
        }
        if hypothesis_id:
            entry["hypothesisId"] = hypothesis_id
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write debug log: {e}")


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


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion in wxyz to rotation matrix."""
    q = _normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


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
class NodeStateBackground:
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateBackground":
        return NodeStateBackground(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


@dataclass
class NodeStateRigid:
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    point_ids: torch.Tensor
    instances_quats: torch.Tensor
    instances_trans: torch.Tensor
    instances_fv: torch.Tensor
    instance_ids: List[int]
    frame_ids: List[int]
    cur_frame: int

    def detach_clone(self) -> "NodeStateRigid":
        return NodeStateRigid(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
            point_ids=self.point_ids.detach().clone(),
            instances_quats=self.instances_quats.detach().clone(),
            instances_trans=self.instances_trans.detach().clone(),
            instances_fv=self.instances_fv.detach().clone(),
            instance_ids=list(self.instance_ids),
            frame_ids=list(self.frame_ids),
            cur_frame=int(self.cur_frame),
        )


NodeState = NodeStateBackground


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
        image_feature_extractor: Optional[nn.Module] = None,
        alpha_t_extractor: Optional[Callable] = None,
        feature_2d_backprojector: Optional[nn.Module] = None,
        feature_fusion: Optional[nn.Module] = None,
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
        self.sparseConv_outdim = outdim
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

        model_2d_flags = model_cfg.get("use_2d_features", False)
        self.use_2d_features = bool(model_2d_flags)
        self.feat_2d_channels = model_cfg.get("feat_2d_channels", 16)
        self.feat_2d_resolution = model_cfg.get("feat_2d_resolution", 0.25)
        self.alpha_t_top_k = model_cfg.get("alpha_t_top_k", 8)

        if self.use_2d_features:
            if image_feature_extractor is not None:
                self.image_feature_extractor = image_feature_extractor.to(device)
            else:
                from models.feature_extractors import ImageFeatureExtractor

                self.image_feature_extractor = ImageFeatureExtractor(
                    in_channels=3,
                    out_channels=self.feat_2d_channels,
                    backbone=model_cfg.get("feat_2d_backbone", "resnet18"),
                    feature_resolution=self.feat_2d_resolution,
                    pretrained=model_cfg.get("feat_2d_pretrained", True),
                    device=device,
                )

            if alpha_t_extractor is not None:
                self.alpha_t_extractor = alpha_t_extractor
            else:
                from models.feature_extractors import AlphaTWeightExtractor

                self.alpha_t_extractor = AlphaTWeightExtractor(
                    renderer=self.renderer,
                    top_k=self.alpha_t_top_k,
                    device=device,
                )

            if feature_2d_backprojector is not None:
                self.feature_2d_backprojector = feature_2d_backprojector.to(device)
            else:
                from models.feature_extractors import Feature2DBackprojector

                self.feature_2d_backprojector = Feature2DBackprojector(
                    feature_channels=self.feat_2d_channels,
                    eps=1e-8,
                    device=device,
                )

            if feature_fusion is not None:
                self.feature_fusion = feature_fusion.to(device)
            else:
                from models.feature_extractors import FeatureFusion

                self.feature_fusion = FeatureFusion(
                    feat_3d_dim=outdim,
                    feat_2d_dim=self.feat_2d_channels,
                    include_visibility=True,
                ).to(device)

            self.feat_fused_dim = outdim + self.feat_2d_channels + 1
        else:
            self.image_feature_extractor = None
            self.alpha_t_extractor = None
            self.feature_2d_backprojector = None
            self.feature_fusion = None
            self.feat_fused_dim = outdim

        mlp_input_dim = self.feat_fused_dim

        self.mlp_offset_pos = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(device)

        self.mlp_conv = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 3 for scales + 3 for axis-angle
        ).to(device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)

        num_sh = _num_sh_bases(self.sh_degree)
        self.gaussion_decoder = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)

        params: List[torch.nn.Parameter] = []
        params += list(self.sparse_conv.parameters())
        if self.use_2d_features:
            params += list(self.image_feature_extractor.parameters())
            params += list(self.feature_2d_backprojector.parameters())
            params += list(self.feature_fusion.parameters())
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
        self.node_states_bg = self.node_states
        self.node_states_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
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

    def _compute_initial_scales(self, means: torch.Tensor) -> torch.Tensor:
        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        return torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

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

        initial_scales = self._compute_initial_scales(means)

        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        node_state = NodeStateBackground(
            means=means.detach().clone(),
            scales_log=initial_scales.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )
        self.node_states[(scene_id, segment_id)] = node_state
        return node_state

    def _init_rigid_node_state_from_pcd(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        point_ids: torch.Tensor,
        dynamic_info: Dict,
        frame_ids: List[int],
        instance_id_map: Dict[int, int],
        instance_ids: List[int],
    ) -> NodeStateRigid:
        means = torch.tensor(points, dtype=torch.float32, device=self.device)
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor
        scales_log = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        num_frames = len(frame_ids)
        num_instances = len(instance_id_map)
        instances_quats = torch.zeros(num_frames, num_instances, 4, device=self.device)
        instances_trans = torch.zeros(num_frames, num_instances, 3, device=self.device)
        instances_fv = torch.zeros(num_frames, num_instances, dtype=torch.bool, device=self.device)
        instances_quats[..., 0] = 1.0

        frame_id_map = {fid: idx for idx, fid in enumerate(frame_ids)}
        for frame_id, frame_info in dynamic_info.items():
            frame_idx = int(frame_id)
            if frame_idx not in frame_id_map:
                continue
            frame_slot = frame_id_map[frame_idx]
            instances = frame_info.get("instances", {})
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        continue
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=self.device)
                    trans = torch.tensor(instance_pose["trans"], device=self.device)
                    instances_quats[frame_slot, ins_slot] = quat
                    instances_trans[frame_slot, ins_slot] = trans
                    instances_fv[frame_slot, ins_slot] = True

        return NodeStateRigid(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
            point_ids=point_ids.detach().clone(),
            instances_quats=instances_quats.detach().clone(),
            instances_trans=instances_trans.detach().clone(),
            instances_fv=instances_fv.detach().clone(),
            instance_ids=list(instance_ids),
            frame_ids=list(frame_ids),
            cur_frame=0,
        )

    def _get_or_init_node_states(
        self, batch: Dict
    ) -> Tuple[Tuple[int, int], NodeState, Optional[NodeStateRigid]]:
        scene_id = batch["scene_id"]
        if isinstance(scene_id, torch.Tensor):
            scene_id = int(scene_id.item())
        segment_id = batch["segment_id"]
        if isinstance(segment_id, torch.Tensor):
            segment_id = int(segment_id.item())
        key = (scene_id, segment_id)
        if key in self.node_states:
            node_state_rigid = self.node_states_rigid.get(key)
            dynamic_info = batch.get("dynamic_info")
            if node_state_rigid is not None and dynamic_info:
                node_state_rigid = self._extend_rigid_frames(node_state_rigid, dynamic_info)
                self.node_states_rigid[key] = node_state_rigid
            return key, self.node_states[key], node_state_rigid
        pointcloud = batch["pointcloud"]
        node_state_bg = self._init_node_from_pointcloud(scene_id, segment_id, pointcloud)

        node_state_rigid: Optional[NodeStateRigid] = None
        if isinstance(pointcloud, dict) and pointcloud.get("dynamic"):
            dynamic_points = []
            dynamic_colors = []
            point_ids = []
            instance_ids = sorted(int(ins_id) for ins_id in pointcloud["dynamic"].keys())
            instance_id_map = {ins_id: idx for idx, ins_id in enumerate(instance_ids)}
            for ins_id in instance_ids:
                instance_pcd = pointcloud["dynamic"][ins_id]
                if instance_pcd is None or len(instance_pcd) == 0:
                    continue
                n_points = instance_pcd.shape[0]
                dynamic_points.append(instance_pcd[:, :3])
                dynamic_colors.append(instance_pcd[:, 3:6])
                point_ids.extend([instance_id_map[ins_id]] * n_points)

            if dynamic_points:
                dynamic_points = np.concatenate(dynamic_points, axis=0)
                dynamic_colors = np.concatenate(dynamic_colors, axis=0)
                point_ids_tensor = torch.tensor(point_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
                dynamic_info = batch.get("dynamic_info")
                if not dynamic_info:
                    raise ValueError("dynamic_info is required when dynamic pointclouds are provided.")
                frame_ids = sorted(int(fid) for fid in dynamic_info.keys())
                node_state_rigid = self._init_rigid_node_state_from_pcd(
                    points=dynamic_points,
                    colors=dynamic_colors,
                    point_ids=point_ids_tensor,
                    dynamic_info=dynamic_info,
                    frame_ids=frame_ids,
                    instance_id_map=instance_id_map,
                    instance_ids=instance_ids,
                )

        self.node_states_rigid[key] = node_state_rigid
        return key, node_state_bg, node_state_rigid

    def _get_or_init_node_state(self, batch: Dict) -> Tuple[Tuple[int, int], NodeState]:
        key, node_state_bg, _ = self._get_or_init_node_states(batch)
        return key, node_state_bg

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

    def _node_state_rigid_to_dict(self, node_state: NodeStateRigid) -> Dict:
        return {
            "means": node_state.means.detach().cpu(),
            "scales_log": node_state.scales_log.detach().cpu(),
            "quats": node_state.quats.detach().cpu(),
            "opacity_logit": node_state.opacity_logit.detach().cpu(),
            "sh_dc": node_state.sh_dc.detach().cpu(),
            "sh_rest": node_state.sh_rest.detach().cpu(),
            "point_ids": node_state.point_ids.detach().cpu(),
            "instances_quats": node_state.instances_quats.detach().cpu(),
            "instances_trans": node_state.instances_trans.detach().cpu(),
            "instances_fv": node_state.instances_fv.detach().cpu(),
            "instance_ids": list(node_state.instance_ids),
            "frame_ids": list(node_state.frame_ids),
            "cur_frame": int(node_state.cur_frame),
        }

    def _node_state_rigid_from_dict(self, state_dict: Dict) -> NodeStateRigid:
        instance_ids = state_dict.get("instance_ids")
        if instance_ids is None:
            num_instances = state_dict["instances_quats"].shape[1]
            instance_ids = list(range(num_instances))
        elif isinstance(instance_ids, torch.Tensor):
            instance_ids = instance_ids.tolist()
        return NodeStateRigid(
            means=state_dict["means"].to(self.device),
            scales_log=state_dict["scales_log"].to(self.device),
            quats=state_dict["quats"].to(self.device),
            opacity_logit=state_dict["opacity_logit"].to(self.device),
            sh_dc=state_dict["sh_dc"].to(self.device),
            sh_rest=state_dict["sh_rest"].to(self.device),
            point_ids=state_dict["point_ids"].to(self.device),
            instances_quats=state_dict["instances_quats"].to(self.device),
            instances_trans=state_dict["instances_trans"].to(self.device),
            instances_fv=state_dict["instances_fv"].to(self.device),
            instance_ids=list(instance_ids),
            frame_ids=list(state_dict.get("frame_ids", [])),
            cur_frame=int(state_dict.get("cur_frame", 0)),
        ).detach_clone()

    def _extend_rigid_frames(self, node_state_rigid: NodeStateRigid, dynamic_info: Dict) -> NodeStateRigid:
        if not dynamic_info:
            return node_state_rigid
        existing_frame_ids = set(node_state_rigid.frame_ids)
        candidate_frame_ids = [int(fid) for fid in dynamic_info.keys()]
        new_frame_ids = [fid for fid in candidate_frame_ids if fid not in existing_frame_ids]
        if not new_frame_ids:
            return node_state_rigid

        new_frame_ids = sorted(new_frame_ids)
        num_new_frames = len(new_frame_ids)
        num_instances = node_state_rigid.instances_quats.shape[1]
        device = node_state_rigid.instances_quats.device

        new_quats = torch.zeros((num_new_frames, num_instances, 4), device=device)
        new_trans = torch.zeros((num_new_frames, num_instances, 3), device=device)
        new_fv = torch.zeros((num_new_frames, num_instances), dtype=torch.bool, device=device)
        new_quats[..., 0] = 1.0

        if node_state_rigid.instance_ids:
            instance_id_map = {int(ins_id): idx for idx, ins_id in enumerate(node_state_rigid.instance_ids)}
        else:
            instance_id_map = {int(idx): idx for idx in range(num_instances)}

        for frame_slot, frame_id in enumerate(new_frame_ids):
            frame_info = dynamic_info.get(frame_id)
            if frame_info is None:
                frame_info = dynamic_info.get(str(frame_id))
            if not frame_info:
                continue
            instances = frame_info.get("instances", {})
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        continue
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=device)
                    trans = torch.tensor(instance_pose["trans"], device=device)
                    new_quats[frame_slot, ins_slot] = quat
                    new_trans[frame_slot, ins_slot] = trans
                    new_fv[frame_slot, ins_slot] = True

        node_state_rigid.instances_quats = torch.cat([node_state_rigid.instances_quats, new_quats], dim=0)
        node_state_rigid.instances_trans = torch.cat([node_state_rigid.instances_trans, new_trans], dim=0)
        node_state_rigid.instances_fv = torch.cat([node_state_rigid.instances_fv, new_fv], dim=0)
        node_state_rigid.frame_ids.extend(new_frame_ids)
        return node_state_rigid

    def _resolve_rigid_frame_idx(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> int:
        """
        将 frame_idx（frame ID）解析为 frame_ids 列表中的索引。
        
        Args:
            frame_idx: 场景全局 frame ID（不是索引）
            
        Returns:
            frame_ids 列表中的索引
            
        Raises:
            ValueError: 如果 frame_idx 不在 frame_ids 列表中
        """
        if not node_state_rigid.frame_ids:
            # 如果没有 frame_ids，假设 frame_idx 就是索引
            return int(frame_idx)
        
        # 首先检查 frame_idx 是否是 frame ID
        if frame_idx in node_state_rigid.frame_ids:
            return node_state_rigid.frame_ids.index(frame_idx)
        
        # 如果找不到，抛出错误
        raise ValueError(
            f"Frame ID {frame_idx} not found in frame_ids {node_state_rigid.frame_ids}. "
            f"Please ensure the frame_idx is a valid frame ID, not an index."
        )

    def _transform_rigid_to_world(
        self, node_state_rigid: NodeStateRigid, means_local: torch.Tensor
    ) -> torch.Tensor:
        # #region agent log
        _debug_log(
            "streetforward.py:_transform_rigid_to_world",
            "Transforming rigid to world",
            {
                "num_points": means_local.shape[0],
                "cur_frame": node_state_rigid.cur_frame,
                "means_local_requires_grad": means_local.requires_grad,
                "means_local_is_leaf": means_local.is_leaf,
            },
            hypothesis_id="H2",
        )
        # #endregion
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        trans_cur_frame = node_state_rigid.instances_trans[frame_idx]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)
        rot_per_pts = rot_cur_frame[node_state_rigid.point_ids[..., 0]]
        trans_per_pts = trans_cur_frame[node_state_rigid.point_ids[..., 0]]
        means_world = torch.bmm(rot_per_pts, means_local.unsqueeze(-1)).squeeze(-1) + trans_per_pts
        # #region agent log
        _debug_log(
            "streetforward.py:_transform_rigid_to_world",
            "Transformation complete",
            {
                "means_world_requires_grad": means_world.requires_grad,
                "means_world_is_leaf": means_world.is_leaf,
                "grad_fn": str(means_world.grad_fn),
            },
            hypothesis_id="H2",
        )
        # #endregion
        return means_world

    def _transform_rigid_quats_to_world(
        self, node_state_rigid: NodeStateRigid, quats_local: torch.Tensor
    ) -> torch.Tensor:
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        quats_per_pts = quats_cur_frame[node_state_rigid.point_ids[..., 0]]
        return _normalize_quat(_quat_multiply(quats_per_pts, quats_local))

    def _transform_offsets_world_to_local(
        self, node_state_rigid: NodeStateRigid, offsets_world: Dict[str, torch.Tensor], frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        将世界坐标的偏移量变换到局部坐标系。
        
        关键：offsets 是向量，变换方式与位置不同（只需要旋转，不需要平移）。
        
        Args:
            node_state_rigid: Rigid node state
            offsets_world: 世界坐标的偏移量字典
            frame_idx: 当前帧的 frame ID
            
        Returns:
            局部坐标的偏移量字典
        """
        resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        
        # 获取当前帧的旋转矩阵
        quats_cur_frame = node_state_rigid.instances_quats[resolved_frame_idx]  # [num_instances, 4]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)  # [num_instances, 3, 3]
        rot_per_pts = rot_cur_frame[node_state_rigid.point_ids[..., 0]]  # [N_rigid, 3, 3]
        
        # 将世界坐标的位置偏移量变换到局部坐标
        # 对于向量（偏移量），只需要旋转，不需要平移
        offset_pos_world = offsets_world["offset_pos"]  # [N_rigid, 3]
        offset_pos_local = torch.bmm(
            rot_per_pts.transpose(-2, -1),  # R^T: [N_rigid, 3, 3]
            offset_pos_world.unsqueeze(-1)  # [N_rigid, 3, 1]
        ).squeeze(-1)  # [N_rigid, 3]
        
        # 将世界坐标的旋转增量转换到局部坐标：q_local = q_inst^{-1} * q_world * q_inst
        offset_quat_world = offsets_world["offset_quat"]
        quats_per_pts = _normalize_quat(node_state_rigid.instances_quats[resolved_frame_idx][node_state_rigid.point_ids[..., 0]])
        quats_inv = _quat_conjugate(quats_per_pts)
        offset_quat = _normalize_quat(_quat_multiply(_quat_multiply(quats_inv, offset_quat_world), quats_per_pts))
        
        # 其他偏移量（scales, opacity, sh）是标量或颜色，不需要坐标变换
        return {
            "offset_pos": offset_pos_local,
            "offset_scales": offsets_world["offset_scales"],  # 尺度不变
            "offset_quat": offset_quat,
            "offset_opacity": offsets_world["offset_opacity"],  # 不变
            "offset_sh": offsets_world["offset_sh"],  # 不变
        }

    def _prepare_source_views_for_2d_features(
        self,
        batch: Dict,
        source_frame_idx: int,
    ) -> Tuple[List, List[torch.Tensor]]:
        source_views: List = []
        source_images: List[torch.Tensor] = []
        # Support both "source_images" and "src_images" keys
        source_images_key = None
        if "source_images" in batch:
            source_images_key = "source_images"
        elif "src_images" in batch:
            source_images_key = "src_images"
        
        if "source_views" in batch and source_images_key is not None:
            source_views = batch["source_views"]
            source_images = batch[source_images_key]
        elif "source_data" in batch:
            for item in batch["source_data"]:
                if item.get("frame_idx") == source_frame_idx:
                    if "view" in item:
                        source_views.append(item["view"])
                    if "image" in item:
                        source_images.append(item["image"])
        else:
            logger.warning("No source views/images found in batch for 2D feature extraction.")

        processed_images: List[torch.Tensor] = []
        for img in source_images:
            if img.dim() == 3:
                processed_images.append(img.to(self.device))
            elif img.dim() == 4:
                processed_images.append(img.squeeze(0).to(self.device))
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

        return source_views, processed_images

    def _build_3d_feature_volume(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Start building 3D feature volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_bg_points": node_state_bg.means.shape[0],
                    "has_rigid": node_state_rigid is not None,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        rigid_visible_mask = None
        if node_state_rigid is not None:
            node_state_rigid.cur_frame = source_frame_idx
            resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, source_frame_idx)
            visibility = node_state_rigid.instances_fv[resolved_frame_idx]
            rigid_visible_mask = visibility[node_state_rigid.point_ids[..., 0]].bool()

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)

        means_rigid_world_all = torch.empty(0, 3, device=self.device)
        anchor_rgb_rigid_all = torch.empty(0, 3, device=self.device)
        if node_state_rigid is not None:
            means_rigid_world_all = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            anchor_rgb_rigid_all = _sh_to_rgb(node_state_rigid.sh_dc)

        if rigid_visible_mask is not None:
            means_rigid_world = means_rigid_world_all[rigid_visible_mask]
            anchor_rgb_rigid = anchor_rgb_rigid_all[rigid_visible_mask]
        else:
            means_rigid_world = means_rigid_world_all
            anchor_rgb_rigid = anchor_rgb_rigid_all

        means_all = torch.cat([means_bg, means_rigid_world], dim=0)
        anchor_rgb_all = torch.cat([anchor_rgb_bg, anchor_rgb_rigid], dim=0)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Before construct_sparse_tensor",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_total_points": means_all.shape[0],
                    "means_all_size_mb": means_all.numel() * 4 / 1024**2,
                    "anchor_rgb_all_size_mb": anchor_rgb_all.numel() * 4 / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion

        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means_all.clone(),
            feats=anchor_rgb_all,
            Bbx_max=self.bbx_max,
            Bbx_min=self.bbx_min,
            voxel_size=self.voxel_size,
            device=self.device,
        )
        
        # #region agent log
        if torch.cuda.is_available():
            num_voxels = sparse_feat.feats.shape[0] if hasattr(sparse_feat, 'feats') else 0
            sparse_feat_size = sparse_feat.feats.numel() * 4 / 1024**2 if hasattr(sparse_feat, 'feats') else 0
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After construct_sparse_tensor, before sparse_conv",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "num_voxels": num_voxels,
                    "vol_dim": vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim,
                    "sparse_feat_size_mb": sparse_feat_size,
                    "valid_coords_size_mb": valid_coords.numel() * 4 / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        feat_3d = self.sparse_conv(sparse_feat)
        
        # #region agent log
        if torch.cuda.is_available():
            # sparse_conv returns torch.Tensor (x.F), not SparseTensor
            if isinstance(feat_3d, torch.Tensor):
                feat_3d_size = feat_3d.numel() * 4 / 1024**2
                feat_3d_shape = list(feat_3d.shape)
            elif hasattr(feat_3d, 'feats'):
                feat_3d_size = feat_3d.feats.numel() * 4 / 1024**2
                feat_3d_shape = list(feat_3d.feats.shape)
            else:
                feat_3d_size = 0
                feat_3d_shape = None
            
            # Calculate expected dense volume size
            vol_dim_list = vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim
            if isinstance(feat_3d, torch.Tensor):
                outdim = feat_3d.shape[-1]
            elif hasattr(feat_3d, 'feats'):
                outdim = feat_3d.feats.shape[-1]
            else:
                outdim = 32  # default
            expected_dense_size_mb = vol_dim_list[0] * vol_dim_list[1] * vol_dim_list[2] * outdim * 4 / 1024**2
            
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After sparse_conv, before sparse_to_dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_size_mb": feat_3d_size,
                    "feat_3d_shape": feat_3d_shape,
                    "feat_3d_type": str(type(feat_3d)),
                    "expected_dense_size_mb": expected_dense_size_mb,
                    "vol_dim": vol_dim_list,
                    "outdim": outdim,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Right before sparse_to_dense_volume (critical memory point)",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved() * 1024**2) / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        try:
            dense_volume = self.sparse_to_dense_volume(
                sparse_tensor=feat_3d,
                coords=valid_coords,
                vol_dim=vol_dim,
            ).unsqueeze(dim=0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:_build_3d_feature_volume",
                        "OOM ERROR in sparse_to_dense_volume",
                        {
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "error": str(e),
                            "vol_dim": vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
            raise
        
        # #region agent log
        if torch.cuda.is_available():
            vol_dim_list = vol_dim.tolist() if isinstance(vol_dim, torch.Tensor) else vol_dim
            dense_volume_size = dense_volume.numel() * 4 / 1024**2
            expected_dense_size = vol_dim_list[0] * vol_dim_list[1] * vol_dim_list[2] * dense_volume.shape[-1] * 4 / 1024**2
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After sparse_to_dense_volume, before permute",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "dense_volume_size_mb": dense_volume_size,
                    "dense_volume_shape": list(dense_volume.shape),
                    "expected_dense_size_mb": expected_dense_size,
                    "vol_dim": vol_dim_list,
                    "memory_increase_mb": torch.cuda.memory_allocated() / 1024**2 - 11048.23,  # Compare with before sparse_conv
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)

        grid_coords_bg = self.get_grid_coords(means_bg, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop_bg = self.interpolate_features(grid_coords_bg, dense_volume)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After interpolate_features for bg",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_bg_size_mb": feat_3d_crop_bg.numel() * 4 / 1024**2,
                    "feat_3d_crop_bg_shape": list(feat_3d_crop_bg.shape),
                },
                hypothesis_id="H5",
            )
        # #endregion

        if node_state_rigid is not None:
            if means_rigid_world_all.shape[0] > 0:
                grid_coords_rigid_all = self.get_grid_coords(
                    means_rigid_world_all, self.bbx_min, vol_dim, self.voxel_size
                )
                feat_3d_crop_rigid_all = self.interpolate_features(grid_coords_rigid_all, dense_volume)
                if rigid_visible_mask is not None:
                    feat_3d_crop_rigid_all = feat_3d_crop_rigid_all * rigid_visible_mask[:, None].float()
                feat_3d_crop_rigid = feat_3d_crop_rigid_all
            else:
                feat_3d_crop_rigid = torch.empty(
                    0, feat_3d_crop_bg.shape[1], device=self.device
                )
        else:
            feat_3d_crop_rigid = torch.empty(0, feat_3d_crop_bg.shape[1], device=self.device)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "Before deleting dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_rigid_size_mb": feat_3d_crop_rigid.numel() * 4 / 1024**2 if feat_3d_crop_rigid.numel() > 0 else 0,
                },
                hypothesis_id="H5",
            )
        # #endregion

        del dense_volume
        
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_build_3d_feature_volume",
                "After deleting dense_volume",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        return feat_3d_crop_bg, feat_3d_crop_rigid, rigid_visible_mask

    def _mask_rigid_offsets(
        self, offsets: Dict[str, torch.Tensor], visible_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if visible_mask is None or visible_mask.numel() == 0:
            return offsets
        mask = visible_mask.to(offsets["offset_pos"].device)
        mask_vec = mask.unsqueeze(-1).float()
        offset_quat = offsets["offset_quat"]
        identity_quat = torch.zeros_like(offset_quat)
        identity_quat[..., 0] = 1.0
        return {
            "offset_pos": offsets["offset_pos"] * mask_vec,
            "offset_scales": offsets["offset_scales"] * mask_vec,
            "offset_quat": torch.where(mask.unsqueeze(-1), offset_quat, identity_quat),
            "offset_opacity": offsets["offset_opacity"] * mask_vec,
            "offset_sh": offsets["offset_sh"] * mask_vec,
        }

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
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_predict_offsets",
                "Start predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_size_mb": feat_3d_crop.numel() * 4 / 1024**2,
                    "feat_3d_crop_shape": list(feat_3d_crop.shape),
                },
                hypothesis_id="H5",
            )
        # #endregion
        
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
        
        # #region agent log
        if torch.cuda.is_available():
            total_offset_size = (
                offset_pos.numel() * 4 + offset_scales.numel() * 4 + offset_quat.numel() * 4 +
                offset_opacity.numel() * 4 + offset_sh.numel() * 4
            ) / 1024**2
            _debug_log(
                "streetforward.py:_predict_offsets",
                "After predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "total_offset_size_mb": total_offset_size,
                    "offset_shapes": {
                        "offset_pos": list(offset_pos.shape),
                        "offset_scales": list(offset_scales.shape),
                        "offset_quat": list(offset_quat.shape),
                        "offset_opacity": list(offset_opacity.shape),
                        "offset_sh": list(offset_sh.shape),
                    },
                },
                hypothesis_id="H5",
            )
        # #endregion
        
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
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Creating proxy params",
            {
                "num_points": render_params["means_r"].shape[0],
                "requires_grad_before": render_params["means_r"].requires_grad,
            },
            hypothesis_id="H1",
        )
        # #endregion
        proxies = {
            "means_p": render_params["means_r"].detach().requires_grad_(True),
            "scales_p": render_params["scales_r"].detach().requires_grad_(True),
            "quats_p": render_params["quats_r"].detach().requires_grad_(True),
            "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
            "colors_p": render_params["colors_r"].detach().requires_grad_(True),
        }
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Proxy params created",
            {
                "requires_grad_after": proxies["means_p"].requires_grad,
                "is_leaf": proxies["means_p"].is_leaf,
                "grad_fn": str(proxies["means_p"].grad_fn),
            },
            hypothesis_id="H1",
        )
        # #endregion
        return proxies

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

        feat_for_offsets = feat_3d_crop
        if self.use_2d_features:
            # In eval/test we may not have source images; fall back to zero 2D features to keep shapes consistent.
            if not getattr(self, "_warned_eval_no_2d", False):
                logger.warning(
                    "use_2d_features is enabled but _compute_render_params has no source views/images; "
                    "using zero 2D features for evaluation."
                )
                self._warned_eval_no_2d = True
            num_bg = feat_3d_crop.shape[0]
            feat_2d_bg = torch.zeros(
                (num_bg, self.feat_2d_channels),
                device=feat_3d_crop.device,
                dtype=feat_3d_crop.dtype,
            )
            vis_bg = torch.zeros(num_bg, device=feat_3d_crop.device, dtype=feat_3d_crop.dtype)
            feat_for_offsets, _ = self.feature_fusion(
                feat_3d_bg=feat_3d_crop,
                feat_3d_rigid=torch.zeros(
                    0, feat_3d_crop.shape[1], device=feat_3d_crop.device, dtype=feat_3d_crop.dtype
                ),
                feat_2d_bg=feat_2d_bg,
                feat_2d_rigid=torch.zeros(
                    0, self.feat_2d_channels, device=feat_3d_crop.device, dtype=feat_3d_crop.dtype
                ),
                vis_bg=vis_bg,
                vis_rigid=torch.zeros(0, device=feat_3d_crop.device, dtype=feat_3d_crop.dtype),
            )

        offsets = self._predict_offsets(feat_for_offsets)
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
        key, node_state_bg, node_state_rigid = self._get_or_init_node_states(batch)
        targets = []
        if "targets" in batch:
            for target in batch["targets"]:
                targets.append(
                    {
                        "frame_idx": target.get("frame_idx", batch.get("source_frame_idx", 0)),
                        "view": target["view"],
                        "gt_image": target["gt_image"],
                    }
                )
        else:
            target_views = batch.get("target_views", [])
            gt_images = batch.get("gt_images", [])
            for view, gt_img in zip(target_views, gt_images):
                targets.append(
                    {
                        "frame_idx": batch.get("source_frame_idx", 0),
                        "view": view,
                        "gt_image": gt_img,
                    }
                )

        # Skip if no target views (no supervision)
        if len(targets) == 0:
            return {
                "total_loss": torch.tensor(0.0, device=self.device),
                "node_state": node_state_bg,
                "node_state_rigid": node_state_rigid,
                "outputs": [],
            }
        
        view_count = len(targets)
        outputs = []
        total_loss_val = 0.0  # Use scalar to avoid keeping computation graph
        test_metrics = None

        self.optimizer.zero_grad(set_to_none=True)

        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:train_iter",
                "Before inner iterations",
                {
                    "num_targets": len(targets),
                    "inner_iterations": self.inner_iterations,
                    "use_2d_features": self.use_2d_features,
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                },
                hypothesis_id="H6",
            )
        # #endregion

        for inner_iter_idx in range(self.inner_iterations):
            source_frame_idx = batch.get("source_frame_idx")
            if source_frame_idx is None:
                raise ValueError(
                    "source_frame_idx is required but not found in batch. "
                    "Please ensure the batch contains source_frame_idx."
                )
            source_frame_idx = int(source_frame_idx)
            feat_2d_bg = None
            feat_2d_rigid = None
            vis_bg = None
            vis_rigid = None

            if self.use_2d_features:
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "Before preparing source views for 2D features",
                        {
                            "inner_iter": inner_iter_idx,
                            "source_frame_idx": source_frame_idx,
                            "batch_keys": list(batch.keys()),
                            "has_source_views": "source_views" in batch,
                            "has_source_images": "source_images" in batch,
                            "has_source_data": "source_data" in batch,
                        },
                        hypothesis_id="H1",
                    )
                # #endregion
                
                source_views, source_images = self._prepare_source_views_for_2d_features(
                    batch=batch, source_frame_idx=source_frame_idx
                )
                
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "After preparing source views for 2D features",
                        {
                            "inner_iter": inner_iter_idx,
                            "num_source_views": len(source_views),
                            "num_source_images": len(source_images),
                            "will_use_zero_features": len(source_views) == 0 or len(source_images) == 0,
                        },
                        hypothesis_id="H1",
                    )
                # #endregion
                
                if len(source_views) == 0 or len(source_images) == 0:
                    logger.warning(
                        "2D features enabled but no source views/images provided. Using zeros for this iteration."
                    )
                    feat_2d_bg = torch.zeros(
                        node_state_bg.means.shape[0],
                        self.feat_2d_channels,
                        device=self.device,
                    )
                    vis_bg = torch.zeros(node_state_bg.means.shape[0], device=self.device)
                    if node_state_rigid is not None:
                        feat_2d_rigid = torch.zeros(
                            node_state_rigid.means.shape[0],
                            self.feat_2d_channels,
                            device=self.device,
                        )
                        vis_rigid = torch.zeros(node_state_rigid.means.shape[0], device=self.device)
                    else:
                        feat_2d_rigid = torch.zeros(
                            0, self.feat_2d_channels, device=self.device
                        )
                        vis_rigid = torch.zeros(0, device=self.device)
                else:
                    # #region agent log
                    if torch.cuda.is_available():
                        _debug_log(
                            "streetforward.py:train_iter",
                            "Before 2D feature extraction",
                            {
                                "inner_iter": inner_iter_idx,
                                "num_source_images": len(source_images),
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            },
                            hypothesis_id="H1",
                        )
                    # #endregion
                    
                    features_2d = self.image_feature_extractor(source_images)
                    
                    # #region agent log
                    if torch.cuda.is_available():
                        feat_2d_size_mb = sum(f.numel() * 4 / 1024**2 for f in features_2d)
                        _debug_log(
                            "streetforward.py:train_iter",
                            "After 2D feature extraction",
                            {
                                "inner_iter": inner_iter_idx,
                                "num_features": len(features_2d),
                                "feat_shapes": [list(f.shape) for f in features_2d],
                                "feat_2d_total_size_mb": feat_2d_size_mb,
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            },
                            hypothesis_id="H1",
                        )
                    # #endregion
                    
                    node_state_rigid_temp = node_state_rigid
                    means_rigid_world = torch.empty(0, 3, device=self.device)
                    quats_rigid_world = torch.empty(0, 4, device=self.device)
                    scales_rigid = torch.empty(0, 3, device=self.device)
                    opacities_rigid = torch.empty(0, device=self.device)
                    num_sh_total = _num_sh_bases(self.sh_degree)
                    colors_rigid = torch.empty(0, num_sh_total, 3, device=self.device)
                    if node_state_rigid_temp is not None:
                        node_state_rigid_temp.cur_frame = source_frame_idx
                        means_rigid_world = self._transform_rigid_to_world(
                            node_state_rigid_temp, node_state_rigid_temp.means
                        )
                        quats_rigid_world = self._transform_rigid_quats_to_world(
                            node_state_rigid_temp, node_state_rigid_temp.quats
                        )
                        scales_rigid = torch.exp(node_state_rigid_temp.scales_log)
                        opacities_rigid = torch.sigmoid(node_state_rigid_temp.opacity_logit).squeeze(-1)
                        colors_rigid = torch.cat(
                            [
                                node_state_rigid_temp.sh_dc[:, None, :],
                                node_state_rigid_temp.sh_rest,
                            ],
                            dim=1,
                        )

                    means_bg = node_state_bg.means
                    quats_bg = node_state_bg.quats
                    scales_bg = torch.exp(node_state_bg.scales_log)
                    opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
                    colors_bg = torch.cat(
                        [node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1
                    )

                    means_merged = torch.cat([means_bg, means_rigid_world], dim=0)
                    quats_merged = torch.cat([quats_bg, quats_rigid_world], dim=0)
                    scales_merged = torch.cat([scales_bg, scales_rigid], dim=0)
                    opacities_merged = torch.cat([opacities_bg, opacities_rigid], dim=0)
                    colors_merged = torch.cat([colors_bg, colors_rigid], dim=0)

                    # Get image dimensions
                    # After image_feature_extractor.forward, source_images is List[torch.Tensor]
                    # where each tensor is [C, H, W] format
                    img0 = source_images[0]
                    # #region agent log
                    try:
                        _debug_log(
                            "streetforward.py:train_iter",
                            "Before calculating Hf and Wf",
                            {
                                "inner_iter": inner_iter_idx,
                                "img0_shape": list(img0.shape),
                                "img0_dim": img0.dim(),
                            },
                            hypothesis_id="H2",
                        )
                    except Exception:
                        pass
                    # #endregion
                    
                    if img0.dim() == 3:
                        # After image_feature_extractor.forward, images are [C, H, W]
                        img_h, img_w = img0.shape[1], img0.shape[2]
                    else:
                        raise ValueError(f"Unexpected image shape: {img0.shape}")
                    
                    # #region agent log
                    try:
                        _debug_log(
                            "streetforward.py:train_iter",
                            "After calculating img_h and img_w",
                            {
                                "inner_iter": inner_iter_idx,
                                "img_h": int(img_h),
                                "img_w": int(img_w),
                            },
                            hypothesis_id="H2",
                        )
                    except Exception:
                        pass
                    # #endregion
                    
                    Hf, Wf = self.image_feature_extractor.get_feature_resolution(img_h, img_w)

                    # #region agent log
                    if torch.cuda.is_available():
                        _debug_log(
                            "streetforward.py:train_iter",
                            "Before alphaT extraction",
                            {
                                "inner_iter": inner_iter_idx,
                                "num_merged_gaussians": means_merged.shape[0],
                                "target_resolution": [Hf, Wf],
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            },
                            hypothesis_id="H2",
                        )
                    # #endregion
                    
                    gaussian_indices, alpha_t_weights = self.alpha_t_extractor.extract_alpha_t_weights(
                        means=means_merged,
                        quats=quats_merged,
                        scales=scales_merged,
                        opacities=opacities_merged,
                        colors=colors_merged,
                        views=source_views,
                        height=Hf,
                        width=Wf,
                        sh_degree=self.sh_degree,
                    )
                    
                    # #region agent log
                    if torch.cuda.is_available():
                        idx_size_mb = sum(idx.numel() * 4 / 1024**2 for idx in gaussian_indices)
                        w_size_mb = sum(w.numel() * 4 / 1024**2 for w in alpha_t_weights)
                        _debug_log(
                            "streetforward.py:train_iter",
                            "After alphaT extraction",
                            {
                                "inner_iter": inner_iter_idx,
                                "num_views": len(gaussian_indices),
                                "idx_total_size_mb": idx_size_mb,
                                "w_total_size_mb": w_size_mb,
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            },
                            hypothesis_id="H2",
                        )
                    # #endregion

                    N_bg = node_state_bg.means.shape[0]
                    N_rigid = means_rigid_world.shape[0]
                    bg_indices = torch.arange(N_bg, device=self.device)
                    rigid_indices = (
                        torch.arange(N_bg, N_bg + N_rigid, device=self.device) if N_rigid > 0 else None
                    )

                    # #region agent log
                    if torch.cuda.is_available():
                        _debug_log(
                            "streetforward.py:train_iter",
                            "Before 2D backprojection",
                            {
                                "inner_iter": inner_iter_idx,
                                "N_bg": N_bg,
                                "N_rigid": N_rigid,
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            },
                            hypothesis_id="H3",
                        )
                    # #endregion
                    
                    (
                        _,
                        _,
                        feat_2d_bg,
                        feat_2d_rigid,
                        vis_bg,
                        vis_rigid,
                    ) = self.feature_2d_backprojector(
                        features_2d=features_2d,
                        gaussian_indices=gaussian_indices,
                        alpha_t_weights=alpha_t_weights,
                        num_gaussians=N_bg + N_rigid,
                        bg_indices=bg_indices,
                        rigid_indices=rigid_indices,
                    )
                    
                    # #region agent log
                    if torch.cuda.is_available():
                        _debug_log(
                            "streetforward.py:train_iter",
                            "After 2D backprojection",
                            {
                                "inner_iter": inner_iter_idx,
                                "feat_2d_bg_shape": list(feat_2d_bg.shape),
                                "feat_2d_rigid_shape": list(feat_2d_rigid.shape),
                                "feat_2d_bg_requires_grad": feat_2d_bg.requires_grad,
                                "feat_2d_rigid_requires_grad": feat_2d_rigid.requires_grad if feat_2d_rigid.numel() > 0 else False,
                                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            },
                            hypothesis_id="H3",
                        )
                    # #endregion
            feat_bg, feat_rigid, rigid_visible_mask = self._build_3d_feature_volume(
                node_state_bg=node_state_bg,
                node_state_rigid=node_state_rigid,
                source_frame_idx=source_frame_idx,
            )

            if self.use_2d_features and feat_2d_bg is not None:
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "Before feature fusion",
                        {
                            "inner_iter": inner_iter_idx,
                            "feat_3d_bg_shape": list(feat_bg.shape),
                            "feat_3d_rigid_shape": list(feat_rigid.shape),
                            "feat_2d_bg_shape": list(feat_2d_bg.shape),
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H4",
                    )
                # #endregion
                
                feat_fused_bg, feat_fused_rigid = self.feature_fusion(
                    feat_3d_bg=feat_bg,
                    feat_3d_rigid=feat_rigid
                    if feat_rigid.shape[0] > 0
                    else torch.empty(0, feat_bg.shape[1], device=self.device),
                    feat_2d_bg=feat_2d_bg,
                    feat_2d_rigid=feat_2d_rigid
                    if feat_2d_rigid is not None and feat_2d_rigid.shape[0] > 0
                    else torch.empty(0, self.feat_2d_channels, device=self.device),
                    vis_bg=vis_bg if vis_bg is not None else torch.zeros(feat_bg.shape[0], device=self.device),
                    vis_rigid=vis_rigid if vis_rigid is not None else torch.zeros(0, device=self.device),
                )
                
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "After feature fusion",
                        {
                            "inner_iter": inner_iter_idx,
                            "feat_fused_bg_shape": list(feat_fused_bg.shape),
                            "feat_fused_rigid_shape": list(feat_fused_rigid.shape),
                            "feat_fused_bg_requires_grad": feat_fused_bg.requires_grad,
                            "expected_dim": self.feat_fused_dim,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H4",
                    )
                # #endregion
            else:
                feat_fused_bg = feat_bg
                feat_fused_rigid = feat_rigid if feat_rigid.shape[0] > 0 else torch.empty(
                    0, feat_bg.shape[1], device=self.device
                )
            
            # #region agent log
            if torch.cuda.is_available():
                _debug_log(
                    "streetforward.py:train_iter",
                    "After _build_3d_feature_volume, before _predict_offsets",
                    {
                        "inner_iter": inner_iter_idx,
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                        "feat_bg_size_mb": feat_bg.numel() * 4 / 1024**2,
                        "feat_rigid_size_mb": feat_rigid.numel() * 4 / 1024**2 if feat_rigid.numel() > 0 else 0,
                    },
                    hypothesis_id="H5",
                )
            # #endregion
            
            offsets_bg = self._predict_offsets(feat_fused_bg)
            offsets_rigid_world = None
            if node_state_rigid is not None and feat_fused_rigid.shape[0] > 0:
                offsets_rigid_world = self._predict_offsets(feat_fused_rigid)
                offsets_rigid_world = self._mask_rigid_offsets(offsets_rigid_world, rigid_visible_mask)
            
            # #region agent log
            if torch.cuda.is_available():
                _debug_log(
                    "streetforward.py:train_iter",
                    "After _predict_offsets, before _render_params_from_offsets",
                    {
                        "inner_iter": inner_iter_idx,
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    },
                    hypothesis_id="H5",
                )
            # #endregion

            # Store offsets for gradient checking (avoid accessing non-leaf tensor grads)
            self._last_offsets_bg = offsets_bg
            self._last_offsets_rigid = offsets_rigid_world
            
            # #region agent log
            # Check offsets requires_grad and is_leaf status
            if offsets_bg:
                offset_status = {}
                for key in ["offset_pos", "offset_scales", "offset_quat", "offset_opacity", "offset_sh"]:
                    if key in offsets_bg:
                        offset_status[f"bg.{key}"] = {
                            "requires_grad": offsets_bg[key].requires_grad,
                            "is_leaf": offsets_bg[key].is_leaf,
                            "grad_fn": str(type(offsets_bg[key].grad_fn)) if offsets_bg[key].grad_fn else "None",
                        }
                _debug_log(
                    "streetforward.py:train_iter",
                    "Offsets status before render_params",
                    {
                        "inner_iter": inner_iter_idx,
                        "offset_status": offset_status,
                    },
                    hypothesis_id="H3",
                )
            # #endregion

            render_params_bg = self._render_params_from_offsets(node_state_bg, offsets_bg)
            render_params_rigid = None
            if node_state_rigid is not None and offsets_rigid_world is not None:
                # 将世界坐标的偏移量变换到局部坐标
                offsets_rigid_local = self._transform_offsets_world_to_local(
                    node_state_rigid, offsets_rigid_world, source_frame_idx
                )
                render_params_rigid = self._render_params_from_offsets(node_state_rigid, offsets_rigid_local)

            proxies_bg = self._create_proxy_params(render_params_bg)
            proxies_rigid = self._create_proxy_params(render_params_rigid) if render_params_rigid is not None else None

            # #region agent log
            _debug_log(
                "streetforward.py:train_iter",
                "Before target views loop",
                {
                    "num_targets": len(targets),
                    "has_rigid": proxies_rigid is not None,
                    "inner_iter": inner_iter_idx,
                },
                hypothesis_id="H1",
            )
            # #endregion

            for view_idx, target in enumerate(targets):
                view = target["view"]
                gt_img = target["gt_image"]
                target_frame_idx = int(target.get("frame_idx", source_frame_idx))
                height, width = gt_img.shape[0], gt_img.shape[1]
                resolved_frame_idx = None
                if node_state_rigid is not None:
                    node_state_rigid.cur_frame = target_frame_idx
                    resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, target_frame_idx)
                if proxies_rigid is not None and node_state_rigid is not None:
                    # #region agent log
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before rigid transform for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "target_frame_idx": target_frame_idx,
                            "proxies_rigid_means_requires_grad": proxies_rigid["means_p"].requires_grad,
                            "proxies_rigid_means_grad": proxies_rigid["means_p"].grad is not None,
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                    means_rigid_world = self._transform_rigid_to_world(node_state_rigid, proxies_rigid["means_p"])
                    quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, proxies_rigid["quats_p"])
                    # #region agent log
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After rigid transform for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "means_rigid_world_requires_grad": means_rigid_world.requires_grad,
                            "means_rigid_world_grad_fn": str(means_rigid_world.grad_fn),
                            "quats_rigid_world_requires_grad": quats_rigid_world.requires_grad,
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                    if resolved_frame_idx is not None:
                        visibility = node_state_rigid.instances_fv[resolved_frame_idx]
                        valid_mask = visibility[node_state_rigid.point_ids[..., 0]].float()
                        opacities_rigid = proxies_rigid["opacities_p"] * valid_mask
                    else:
                        opacities_rigid = proxies_rigid["opacities_p"]
                else:
                    means_rigid_world = torch.empty(0, 3, device=self.device)
                    quats_rigid_world = torch.empty(0, 4, device=self.device)
                    opacities_rigid = None
                merged_means = torch.cat([proxies_bg["means_p"], means_rigid_world], dim=0)
                merged_quats = torch.cat([proxies_bg["quats_p"], quats_rigid_world], dim=0)
                if proxies_rigid is not None:
                    merged_scales = torch.cat([proxies_bg["scales_p"], proxies_rigid["scales_p"]], dim=0)
                    merged_opacities = torch.cat([proxies_bg["opacities_p"], opacities_rigid], dim=0)
                    merged_colors = torch.cat([proxies_bg["colors_p"], proxies_rigid["colors_p"]], dim=0)
                else:
                    merged_scales = proxies_bg["scales_p"]
                    merged_opacities = proxies_bg["opacities_p"]
                    merged_colors = proxies_bg["colors_p"]

                merged_params = {
                    "means_p": merged_means,
                    "scales_p": merged_scales,
                    "quats_p": merged_quats,
                    "opacities_p": merged_opacities,
                    "colors_p": merged_colors,
                }
                # #region agent log
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before _render_single_view for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "merged_means_size_mb": merged_means.numel() * 4 / 1024**2,
                            "merged_colors_size_mb": merged_colors.numel() * 4 / 1024**2,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
                
                rgb, acc = self._render_single_view(merged_params, view, height, width)
                
                # #region agent log
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After _render_single_view for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "rgb_size_mb": rgb.numel() * 4 / 1024**2,
                        },
                        hypothesis_id="H5",
                    )
                # #endregion
                
                loss = self.compute_loss(rgb, gt_img) / view_count
                total_loss_val += float(loss.detach())  # Accumulate scalar to avoid keeping graph
                
                # #region agent log
                # Check proxy gradients before backward (only log every 3 views to reduce overhead)
                if view_idx % 3 == 0 or view_idx == len(targets) - 1:
                    bg_grad_norms_before = {}
                    rigid_grad_norms_before = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        if proxies_bg[key].grad is not None:
                            bg_grad_norms_before[key] = float(proxies_bg[key].grad.norm().item())
                        else:
                            bg_grad_norms_before[key] = 0.0
                        if proxies_rigid is not None and proxies_rigid[key].grad is not None:
                            rigid_grad_norms_before[key] = float(proxies_rigid[key].grad.norm().item())
                        elif proxies_rigid is not None:
                            rigid_grad_norms_before[key] = 0.0
                    
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"Before backward for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "target_frame_idx": target_frame_idx,
                            "loss_value": float(loss.item()),
                            "bg_grad_norms": bg_grad_norms_before,
                            "rigid_grad_norms": rigid_grad_norms_before if proxies_rigid else None,
                        },
                        hypothesis_id="H1",
                    )
                # #endregion
                
                loss.backward()
                
                # #region agent log
                # Check proxy gradients after backward (only log every 3 views to reduce overhead)
                if view_idx % 3 == 0 or view_idx == len(targets) - 1:
                    bg_grad_norms_after = {}
                    rigid_grad_norms_after = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        if proxies_bg[key].grad is not None:
                            bg_grad_norms_after[key] = float(proxies_bg[key].grad.norm().item())
                        else:
                            bg_grad_norms_after[key] = 0.0
                        if proxies_rigid is not None and proxies_rigid[key].grad is not None:
                            rigid_grad_norms_after[key] = float(proxies_rigid[key].grad.norm().item())
                        elif proxies_rigid is not None:
                            rigid_grad_norms_after[key] = 0.0
                    
                    # Check if gradients accumulated
                    bg_grad_accumulated = {}
                    rigid_grad_accumulated = {}
                    for key in ["means_p", "scales_p", "quats_p", "opacities_p", "colors_p"]:
                        bg_grad_accumulated[key] = bg_grad_norms_after[key] > bg_grad_norms_before.get(key, 0.0) if view_idx > 0 else True
                        if proxies_rigid is not None:
                            rigid_grad_accumulated[key] = rigid_grad_norms_after[key] > rigid_grad_norms_before.get(key, 0.0) if view_idx > 0 else True
                    
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"After backward for view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "bg_grad_norms": bg_grad_norms_after,
                            "rigid_grad_norms": rigid_grad_norms_after if proxies_rigid else None,
                            "bg_grad_accumulated": bg_grad_accumulated,
                            "rigid_grad_accumulated": rigid_grad_accumulated if proxies_rigid else None,
                        },
                        hypothesis_id="H1",
                    )
                
                # Check GPU memory (only log every 5 views to reduce overhead)
                if torch.cuda.is_available() and (view_idx % 5 == 0 or view_idx == len(targets) - 1):
                    _debug_log(
                        "streetforward.py:train_iter",
                        f"GPU memory after view {view_idx}",
                        {
                            "view_idx": view_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H4",
                    )
                # #endregion
                
                # Only store images if explicitly requested (to save GPU memory)
                if self.log_images:
                    outputs.append({
                        "rgb": rgb.detach().cpu(),
                        "acc": acc.detach().cpu(),
                        "loss": loss.detach().item(),
                    })
                else:
                    outputs.append({"loss": loss.detach().item()})

            grad_report: Dict[str, float] = {}
            grad_warned = getattr(self, "_proxy_grad_warned", set())

            def _grad_or_zero(proxy_tensor: torch.Tensor, name: str) -> torch.Tensor:
                grad = proxy_tensor.grad
                if grad is None:
                    if name not in grad_warned:
                        logger.warning(f"Proxy gradient for {name} is None; using zeros for backward.")
                        grad_warned.add(name)
                    grad_report[name] = 0.0
                    return torch.zeros_like(proxy_tensor)
                grad_report[name] = float(grad.norm().detach())
                return grad

            render_tensors = [
                render_params_bg["means_r"],
                render_params_bg["scales_r"],
                render_params_bg["quats_r"],
                render_params_bg["opacities_r"],
                render_params_bg["colors_r"],
            ]
            grad_tensors = [
                _grad_or_zero(proxies_bg["means_p"], "bg.means"),
                _grad_or_zero(proxies_bg["scales_p"], "bg.scales"),
                _grad_or_zero(proxies_bg["quats_p"], "bg.quats"),
                _grad_or_zero(proxies_bg["opacities_p"], "bg.opacities"),
                _grad_or_zero(proxies_bg["colors_p"], "bg.colors"),
            ]

            if render_params_rigid is not None and proxies_rigid is not None:
                render_tensors += [
                    render_params_rigid["means_r"],
                    render_params_rigid["scales_r"],
                    render_params_rigid["quats_r"],
                    render_params_rigid["opacities_r"],
                    render_params_rigid["colors_r"],
                ]
                grad_tensors += [
                    _grad_or_zero(proxies_rigid["means_p"], "rigid.means"),
                    _grad_or_zero(proxies_rigid["scales_p"], "rigid.scales"),
                    _grad_or_zero(proxies_rigid["quats_p"], "rigid.quats"),
                    _grad_or_zero(proxies_rigid["opacities_p"], "rigid.opacities"),
                    _grad_or_zero(proxies_rigid["colors_p"], "rigid.colors"),
                ]

            self._proxy_grad_warned = grad_warned
            self._last_proxy_grad_norms = grad_report
            
            # #region agent log
            # Check render params gradients before autograd.backward (only for leaf tensors)
            render_params_grad_before = {}
            for key in ["means_r", "scales_r", "quats_r", "opacities_r", "colors_r"]:
                # Only check grad for leaf tensors to avoid warnings
                if render_params_bg[key].is_leaf and render_params_bg[key].grad is not None:
                    render_params_grad_before[f"bg.{key}"] = float(render_params_bg[key].grad.norm().item())
                else:
                    render_params_grad_before[f"bg.{key}"] = 0.0
                if render_params_rigid is not None:
                    if render_params_rigid[key].is_leaf and render_params_rigid[key].grad is not None:
                        render_params_grad_before[f"rigid.{key}"] = float(render_params_rigid[key].grad.norm().item())
                    else:
                        render_params_grad_before[f"rigid.{key}"] = 0.0
            
            _debug_log(
                "streetforward.py:train_iter",
                "Before autograd.backward",
                {
                    "inner_iter": inner_iter_idx,
                    "proxy_grad_norms": grad_report,
                    "render_params_grad_before": render_params_grad_before,
                },
                hypothesis_id="H3",
            )
            # #endregion
            
            torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)
            
            # #region agent log
            # Check MLP parameters gradients (these are the actual leaf tensors that should have gradients)
            mlp_grads = {}
            mlp_params = {
                "mlp_offset_pos": list(self.mlp_offset_pos.parameters()),
                "mlp_conv": list(self.mlp_conv.parameters()),
                "mlp_opacity": list(self.mlp_opacity.parameters()),
                "gaussion_decoder": list(self.gaussion_decoder.parameters()),
            }
            for mlp_name, params in mlp_params.items():
                for i, param in enumerate(params):
                    param_key = f"{mlp_name}.param_{i}"
                    if param.grad is not None:
                        mlp_grads[param_key] = float(param.grad.norm().item())
                    else:
                        mlp_grads[param_key] = 0.0
            
            # Check CNN parameters gradients (2D feature extractor)
            cnn_grads = {}
            if self.use_2d_features and self.image_feature_extractor is not None:
                for name, param in self.image_feature_extractor.named_parameters():
                    if param.grad is not None:
                        cnn_grads[name] = float(param.grad.norm().item())
                    else:
                        cnn_grads[name] = 0.0
            
            # Check backprojector gradients (should have gradients if feat_2d has gradients)
            backprojector_grads = {}
            if self.use_2d_features and self.feature_2d_backprojector is not None:
                for name, param in self.feature_2d_backprojector.named_parameters():
                    if param.grad is not None:
                        backprojector_grads[name] = float(param.grad.norm().item())
                    else:
                        backprojector_grads[name] = 0.0
            
            # Check feature fusion gradients
            fusion_grads = {}
            if self.use_2d_features and self.feature_fusion is not None:
                for name, param in self.feature_fusion.named_parameters():
                    if param.grad is not None:
                        fusion_grads[name] = float(param.grad.norm().item())
                    else:
                        fusion_grads[name] = 0.0
            
            _debug_log(
                "streetforward.py:train_iter",
                "Gradient check after autograd.backward",
                {
                    "inner_iter": inner_iter_idx,
                    "mlp_grads": mlp_grads,
                    "mlp_has_grad": {k: mlp_grads[k] > 0 for k in mlp_grads},
                    "cnn_grads": cnn_grads,
                    "cnn_has_grad": {k: cnn_grads[k] > 0 for k in cnn_grads},
                    "backprojector_grads": backprojector_grads,
                    "fusion_grads": fusion_grads,
                },
                hypothesis_id="H5",
            )
            # #endregion
            
            # #region agent log
            # Check render params gradients after autograd.backward (only for leaf tensors)
            # Note: render_params are non-leaf tensors, so we check the underlying computation graph
            # by checking if the offsets have gradients instead
            render_params_grad_after = {}
            # Check if offsets have gradients (these are the actual leaf tensors)
            offset_keys = ["offset_pos", "offset_scales", "offset_quat", "offset_opacity", "offset_sh"]
            offset_grads = {}
            offset_status_after = {}
            if hasattr(self, '_last_offsets_bg') and self._last_offsets_bg is not None:
                for key in offset_keys:
                    if key in self._last_offsets_bg:
                        offset_tensor = self._last_offsets_bg[key]
                        has_grad = offset_tensor.grad is not None
                        grad_norm = float(offset_tensor.grad.norm().item()) if has_grad else 0.0
                        offset_grads[f"bg.{key}"] = grad_norm
                        offset_status_after[f"bg.{key}"] = {
                            "has_grad": has_grad,
                            "requires_grad": offset_tensor.requires_grad,
                            "is_leaf": offset_tensor.is_leaf,
                            "grad_fn": str(type(offset_tensor.grad_fn)) if offset_tensor.grad_fn else "None",
                        }
                    else:
                        offset_grads[f"bg.{key}"] = 0.0
            if hasattr(self, '_last_offsets_rigid') and self._last_offsets_rigid is not None:
                for key in offset_keys:
                    if key in self._last_offsets_rigid:
                        offset_tensor = self._last_offsets_rigid[key]
                        has_grad = offset_tensor.grad is not None
                        grad_norm = float(offset_tensor.grad.norm().item()) if has_grad else 0.0
                        offset_grads[f"rigid.{key}"] = grad_norm
                    else:
                        offset_grads[f"rigid.{key}"] = 0.0
            
            # Check if render_params are connected to offsets in computation graph
            render_params_connected = {}
            if hasattr(self, '_last_offsets_bg') and self._last_offsets_bg is not None:
                # Check if render_params_bg are connected to offsets
                for key in ["means_r", "scales_r", "quats_r", "opacities_r", "colors_r"]:
                    render_tensor = render_params_bg[key]
                    render_params_connected[f"bg.{key}"] = {
                        "requires_grad": render_tensor.requires_grad,
                        "is_leaf": render_tensor.is_leaf,
                        "grad_fn": str(type(render_tensor.grad_fn)) if render_tensor.grad_fn else "None",
                    }
            
            _debug_log(
                "streetforward.py:train_iter",
                "After autograd.backward",
                {
                    "inner_iter": inner_iter_idx,
                    "offset_grads": offset_grads,
                    "grad_propagated": {k: offset_grads[k] > 0 for k in offset_grads},
                    "offset_status_after": offset_status_after,
                    "render_params_connected": render_params_connected,
                },
                hypothesis_id="H3",
            )
            # #endregion

            if apply_update:
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "Before optimizer.step",
                        {
                            "inner_iter": inner_iter_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H6",
                    )
                # #endregion
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # #region agent log
                if torch.cuda.is_available():
                    _debug_log(
                        "streetforward.py:train_iter",
                        "After optimizer.step",
                        {
                            "inner_iter": inner_iter_idx,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                        },
                        hypothesis_id="H6",
                    )
                # #endregion

            if update_state:
                with torch.no_grad():
                    # Clamp means only when writing back to node_state (not during backprop)
                    means_clamped = torch.clamp(
                        render_params_bg["means_r"].detach(), min=self.bbx_min, max=self.bbx_max
                    )
                    node_state_bg.means.copy_(means_clamped)
                    node_state_bg.scales_log.copy_(render_params_bg["scales_log_r"].detach())
                    node_state_bg.quats.copy_(render_params_bg["quats_r"].detach())
                    node_state_bg.opacity_logit.copy_(render_params_bg["opacity_logit_r"].detach())
                    node_state_bg.sh_dc.copy_(render_params_bg["sh_dc_r"].detach())
                    node_state_bg.sh_rest.copy_(render_params_bg["sh_rest_r"].detach())
                    if node_state_rigid is not None and render_params_rigid is not None:
                        node_state_rigid.means.copy_(render_params_rigid["means_r"].detach())
                        node_state_rigid.scales_log.copy_(render_params_rigid["scales_log_r"].detach())
                        node_state_rigid.quats.copy_(render_params_rigid["quats_r"].detach())
                        node_state_rigid.opacity_logit.copy_(render_params_rigid["opacity_logit_r"].detach())
                        node_state_rigid.sh_dc.copy_(render_params_rigid["sh_dc_r"].detach())
                        node_state_rigid.sh_rest.copy_(render_params_rigid["sh_rest_r"].detach())

        self.node_states[key] = node_state_bg.detach_clone()
        if node_state_rigid is not None:
            self.node_states_rigid[key] = node_state_rigid.detach_clone()
        else:
            self.node_states_rigid[key] = None
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
            "node_state_rigid": self.node_states_rigid.get(key),
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
        if self.use_2d_features and self.image_feature_extractor is not None:
            model_state_dict["image_feature_extractor"] = self.image_feature_extractor.state_dict()
        if self.use_2d_features and self.feature_2d_backprojector is not None:
            model_state_dict["feature_2d_backprojector"] = self.feature_2d_backprojector.state_dict()
        if self.use_2d_features and self.feature_fusion is not None:
            model_state_dict["feature_fusion"] = self.feature_fusion.state_dict()

        nodes_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_to_dict(state)
            for (scene, segment), state in self.node_states.items()
        }
        rigid_state_dict = {
            f"scene_{scene}_segment_{segment}": self._node_state_rigid_to_dict(state)
            for (scene, segment), state in self.node_states_rigid.items()
            if state is not None
        }

        checkpoint = {
            "step": step_val,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "node_states": nodes_state_dict,
            "node_states_rigid": rigid_state_dict,
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
        if self.use_2d_features and "image_feature_extractor" in model_state and self.image_feature_extractor is not None:
            self.image_feature_extractor.load_state_dict(model_state["image_feature_extractor"], strict=strict)
        if self.use_2d_features and "feature_2d_backprojector" in model_state and self.feature_2d_backprojector is not None:
            self.feature_2d_backprojector.load_state_dict(
                model_state["feature_2d_backprojector"], strict=strict
            )
        if self.use_2d_features and "feature_fusion" in model_state and self.feature_fusion is not None:
            self.feature_fusion.load_state_dict(model_state["feature_fusion"], strict=strict)

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
                self.node_states_bg = self.node_states

        rigid_state_dict = checkpoint.get("node_states_rigid")
        if rigid_state_dict is not None:
            restored_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
            for key, state in rigid_state_dict.items():
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
                restored_rigid[(scene_id, segment_id)] = self._node_state_rigid_from_dict(state)
            if restored_rigid:
                self.node_states_rigid = restored_rigid

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
