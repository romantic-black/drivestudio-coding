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
            nn.Linear(32, 7),
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

        self.node_states: Dict[Tuple[int, int], NodeState] = {}

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
        x_norm = x_index / vol_dim[0] * 2 - 1  # X -> W
        y_norm = y_index / vol_dim[1] * 2 - 1  # Y -> H
        z_norm = z_index / vol_dim[2] * 2 - 1  # Z -> D
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
        offset_pos = self.offset_max * torch.tanh(self.mlp_offset_pos(feat_3d_crop))
        scales_and_quats = self.mlp_conv(feat_3d_crop)
        offset_scales, offset_quat = scales_and_quats.split([3, 4], dim=-1)
        offset_quat = _normalize_quat(offset_quat)
        offset_opacity = self.mlp_opacity(feat_3d_crop)
        offset_sh = self.gaussion_decoder(feat_3d_crop)
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

        means_r = node_state.means + offsets["offset_pos"]
        scales_log_r = node_state.scales_log + offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state.opacity_logit + offsets["offset_opacity"]
        sh_dc_r = node_state.sh_dc + offsets["offset_sh"][:, :3]
        sh_rest_r = node_state.sh_rest + sh_rest_offset

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

    def train_iter(
        self,
        batch: Dict,
        apply_update: bool = True,
        update_state: bool = True,
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

        self.optimizer.zero_grad(set_to_none=True)

        for inner_iter_idx in range(self.inner_iterations):
            means_s = node_state.means
            # Use node_state.sh_dc directly
            # The leaf node will be created in construct_sparse_tensor if needed
            anchor_sh = node_state.sh_dc
            anchor_rgb = _sh_to_rgb(anchor_sh)

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
            # sparse_to_dense_volume returns [X, Y, Z, C] where vol_dim is [X, Y, Z]
            # After unsqueeze: [1, X, Y, Z, C]
            # grid_sample (5D) expects [B, C, D, H, W] where D=Z, H=Y, W=X
            # So we need: permute(0, 4, 3, 2, 1) to get [1, C, Z, Y, X] = [1, C, D, H, W]
            dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
            grid_coords = self.get_grid_coords(means_s, self.bbx_min, vol_dim, self.voxel_size)
            feat_3d_crop = self.interpolate_features(grid_coords, dense_volume)
            del dense_volume

            offsets = self._predict_offsets(feat_3d_crop)
            render_params = self._render_params_from_offsets(node_state, offsets)
            proxies = self._create_proxy_params(render_params)

            for view_idx, (view, gt_img) in enumerate(zip(target_views, gt_images)):
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

                height, width = gt_img.shape[0], gt_img.shape[1]

                render, alpha, _ = self.renderer(
                    means=proxies["means_p"],
                    quats=proxies["quats_p"],
                    scales=proxies["scales_p"],
                    opacities=proxies["opacities_p"],
                    colors=proxies["colors_p"],
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
                loss = self.compute_loss(rgb, gt_img) / view_count
                total_loss_val += float(loss.detach())  # Accumulate scalar to avoid keeping graph
                loss.backward()
                
                # Only store images if explicitly requested (to save GPU memory)
                log_images = self.config.get("log_images", False)
                if log_images:
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
                    node_state.means.copy_(render_params["means_r"].detach())
                    node_state.scales_log.copy_(render_params["scales_log_r"].detach())
                    node_state.quats.copy_(render_params["quats_r"].detach())
                    node_state.opacity_logit.copy_(render_params["opacity_logit_r"].detach())
                    node_state.sh_dc.copy_(render_params["sh_dc_r"].detach())
                    node_state.sh_rest.copy_(render_params["sh_rest_r"].detach())

        self.node_states[key] = node_state.detach_clone()
        return {
            "total_loss": torch.tensor(total_loss_val, device=self.device),
            "node_state": self.node_states[key],
            "outputs": outputs,
        }

    def forward(self, batch: Dict) -> Dict:
        return self.train_iter(batch)
