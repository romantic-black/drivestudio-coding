"""
Minimal StreetForward: point cloud -> 3D features -> 3DGS head -> single-target render.

Implements Stage 0 from docs/trainers/Minimal_StreetForward_Design_Plan.md:
- Input: 3D RGB point cloud only + single target (view + gt_image); no source.
- Middle: 3D feature volume (voxelize -> sparse_conv -> dense -> interpolate) -> per-point feat_3d.
- Head: MLPs predict offset_pos, scales, quats, opacity, SH -> gsplat rasterization.
- Training: single target loss, single backward; no NodeState, proxy, GRU, or 2D features.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.feature_extractors.alpha_t_extractor import _get_viewmat
from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _pairwise_neighbor_distances,
    _rgb_to_sh,
    _sh_to_rgb,
)

logger = logging.getLogger(__name__)

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ImportError:
    _gsplat_rasterization = None

try:
    from models.evol_splat import (
        SparseCostRegNet as _SparseCostRegNet,
        construct_sparse_tensor as _construct_sparse_tensor,
        sparse_to_dense_volume as _sparse_to_dense_volume,
    )
except ImportError:
    _SparseCostRegNet = None
    _construct_sparse_tensor = None
    _sparse_to_dense_volume = None


def _get_grid_coords(
    position_w: torch.Tensor,
    bbx_min: torch.Tensor,
    bbx_max: torch.Tensor,
    vol_dim: Any,
    voxel_size: float,
) -> torch.Tensor:
    """World coordinates -> normalized grid coords [x_norm, y_norm, z_norm] in [-1, 1] for grid_sample."""
    bbx_max = bbx_max.to(position_w.device)
    position_w_clamped = torch.clamp(position_w, min=bbx_min, max=bbx_max)
    pts = position_w_clamped - bbx_min.to(position_w.device)
    x_index = pts[..., 0] / voxel_size
    y_index = pts[..., 1] / voxel_size
    z_index = pts[..., 2] / voxel_size
    if isinstance(vol_dim, (list, tuple)):
        vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
    elif not isinstance(vol_dim, torch.Tensor):
        vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
    else:
        vol_dim = vol_dim.to(position_w.device).float()
    den_x = torch.clamp(vol_dim[0] - 1.0, min=1.0)
    den_y = torch.clamp(vol_dim[1] - 1.0, min=1.0)
    den_z = torch.clamp(vol_dim[2] - 1.0, min=1.0)
    x_norm = 2.0 * (x_index / den_x) - 1.0
    y_norm = 2.0 * (y_index / den_y) - 1.0
    z_norm = 2.0 * (z_index / den_z) - 1.0
    return torch.stack([x_norm, y_norm, z_norm], dim=-1)


def _interpolate_features(grid_coords: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
    """Sample per-point features from [1, C, D, H, W] volume. Returns [N, C]."""
    grid_expanded = grid_coords[None, None, None, ...]
    feat = F.grid_sample(
        feature_volume,
        grid_expanded,
        mode="bilinear",
        align_corners=True,
        padding_mode="zeros",
    )
    return feat[0, :, 0, 0, :].T


class MinimalStreetForward(nn.Module):
    """
    Minimal pipeline: point cloud -> 3D features -> 3DGS params -> single-view render.
    No NodeState, proxy, GRU, or 2D features.
    """

    def __init__(
        self,
        config: OmegaConf,
        device: torch.device,
        renderer: Optional[Callable] = None,
        sparse_conv: Optional[nn.Module] = None,
        construct_sparse_tensor_fn: Optional[Callable] = None,
        sparse_to_dense_volume_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        model_cfg = config.model

        self.offset_max = float(model_cfg.get("offset_max", 0.1))
        self.scale_max = float(model_cfg.get("scale_max", 0.1))
        self.omega_max = float(model_cfg.get("omega_max", 0.1))
        self.opacity_max = float(model_cfg.get("opacity_max", 0.1))
        self.sh_dc_max = float(model_cfg.get("sh_dc_max", 0.1))
        self.sh_rest_max = float(model_cfg.get("sh_rest_max", 0.05))
        self.eta_means = float(model_cfg.get("eta_means", 1.0))
        self.eta_scales = float(model_cfg.get("eta_scales", 1.0))
        self.eta_opacity = float(model_cfg.get("eta_opacity", 1.0))
        self.eta_sh_dc = float(model_cfg.get("eta_sh_dc", 1.0))
        self.eta_sh_rest = float(model_cfg.get("eta_sh_rest", 1.0))

        self.sh_degree = int(model_cfg.get("sh_degree", 1))
        self.voxel_size = float(model_cfg.get("voxel_size", 0.1))
        if not hasattr(config, "dataset") or not hasattr(config.dataset, "segment_aabb"):
            raise ValueError("config.dataset.segment_aabb is required for MinimalStreetForwardTrainer.")
        seg_aabb = config.dataset.segment_aabb
        if seg_aabb is None or len(seg_aabb) != 2:
            raise ValueError("config.dataset.segment_aabb must be [[min],[max]] with shape [2,3].")
        bbx_min = seg_aabb[0]
        bbx_max = seg_aabb[1]
        self.bbx_min = torch.tensor(bbx_min, dtype=torch.float32, device=device)
        self.bbx_max = torch.tensor(bbx_max, dtype=torch.float32, device=device)

        self.renderer = renderer or _gsplat_rasterization
        if self.renderer is None:
            raise ImportError("gsplat not available; install gsplat or pass renderer.")

        outdim = int(model_cfg.get("sparseConv_outdim", 32))
        self.feat_3d_dim = outdim

        if sparse_conv is not None:
            self.sparse_conv = sparse_conv.to(device)
        elif _SparseCostRegNet is not None:
            self.sparse_conv = _SparseCostRegNet(d_in=3, d_out=outdim).to(device)
        else:
            raise ImportError("SparseCostRegNet not available; install models.evol_splat.")

        self.construct_sparse_tensor = construct_sparse_tensor_fn or _construct_sparse_tensor
        self.sparse_to_dense_volume = sparse_to_dense_volume_fn or _sparse_to_dense_volume
        if self.construct_sparse_tensor is None or self.sparse_to_dense_volume is None:
            raise ImportError("evol_splat construct_sparse_tensor / sparse_to_dense_volume required.")

        num_sh = _num_sh_bases(self.sh_degree)

        # Head: 3D feat -> 3DGS params (EVolSplat-style)
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
            nn.Linear(32, 6),  # 3 scale delta + 3 axis-angle
        ).to(device)
        self.mlp_opacity = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        self.gaussion_decoder = nn.Sequential(
            nn.Linear(outdim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)

        # Zero-init last layers so initial offsets are small
        for m in (self.mlp_offset_pos, self.mlp_conv, self.mlp_opacity, self.gaussion_decoder):
            if isinstance(m, nn.Sequential) and len(m) > 0:
                last = m[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

        self.optimizer = torch.optim.Adam(
            list(self.sparse_conv.parameters())
            + list(self.mlp_offset_pos.parameters())
            + list(self.mlp_conv.parameters())
            + list(self.mlp_opacity.parameters())
            + list(self.gaussion_decoder.parameters()),
            lr=float(config.optimizer.get("lr", 1e-3)),
            eps=float(config.optimizer.get("eps", 1e-15)),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        )

    def _pointcloud_to_means_rgb(
        self,
        pointcloud: Dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract means [N,3] and anchor_rgb [N,3] from batch pointcloud (static only)."""
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            if background.shape[1] >= 6:
                colors = background[:, 3:].astype(np.float32)
                if colors.max() > 1.0 + 1e-3:
                    colors = colors / 255.0
            else:
                colors = np.zeros_like(points)
        else:
            points = np.asarray(pointcloud.points)
            colors = np.asarray(pointcloud.colors)
            if colors.max() > 1.0 + 1e-3:
                colors = colors / 255.0
        means = torch.from_numpy(points).float().to(self.device)
        anchor_rgb = torch.from_numpy(colors).float().to(self.device)
        if anchor_rgb.dim() == 1:
            anchor_rgb = anchor_rgb.unsqueeze(-1).expand(-1, 3)
        elif anchor_rgb.shape[1] != 3:
            anchor_rgb = anchor_rgb[:, :3]
        return means, anchor_rgb

    def _build_3d_features(
        self,
        means: torch.Tensor,
        anchor_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Build 3D feature volume and return per-point features [N, outdim]."""
        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means.clone(),
            feats=anchor_rgb,
            Bbx_min=self.bbx_min,
            Bbx_max=self.bbx_max,
            voxel_size=self.voxel_size,
            device=self.device,
        )
        feat_3d = self.sparse_conv(sparse_feat)
        dense_volume = self.sparse_to_dense_volume(
            sparse_tensor=feat_3d,
            coords=valid_coords,
            vol_dim=vol_dim,
        ).unsqueeze(0)
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)  # [1, C, D, H, W]
        grid_coords = _get_grid_coords(
            means, self.bbx_min, self.bbx_max, vol_dim, self.voxel_size
        )
        feat_3d_crop = _interpolate_features(grid_coords, dense_volume)
        return feat_3d_crop

    def _compute_initial_scales(self, means: torch.Tensor) -> torch.Tensor:
        """K-NN based initial scales_log [N, 3]."""
        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        return torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

    def _predict_render_params(
        self,
        means: torch.Tensor,
        anchor_rgb: torch.Tensor,
        feat_3d_crop: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """From means, anchor_rgb, and feat_3d_crop compute 3DGS render params."""
        N = means.shape[0]
        scales_log_init = self._compute_initial_scales(means)

        offset_pos_raw = self.mlp_offset_pos(feat_3d_crop)
        offset_pos = self.offset_max * torch.tanh(offset_pos_raw)
        means_r = means + self.eta_means * offset_pos

        scales_omega = self.mlp_conv(feat_3d_crop)
        scale_delta, omega_raw = scales_omega.split([3, 3], dim=-1)
        offset_scales = self.scale_max * torch.tanh(scale_delta)
        offset_omega = self.omega_max * torch.tanh(omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        scales_log_r = scales_log_init + self.eta_scales * offset_scales
        scales_r = torch.exp(scales_log_r)
        quats_r = F.normalize(offset_quat, dim=-1)

        opacity_raw = self.mlp_opacity(feat_3d_crop)
        offset_opacity = self.opacity_max * torch.tanh(opacity_raw)
        opacity_logit_init = torch.logit(torch.full((N, 1), 0.1, device=self.device))
        opacity_logit_r = opacity_logit_init + self.eta_opacity * offset_opacity
        opacities_r = torch.sigmoid(opacity_logit_r).squeeze(-1)

        sh_raw = self.gaussion_decoder(feat_3d_crop)
        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc_raw = sh_raw[:, :3]
        sh_rest_raw = sh_raw[:, 3:].view(N, num_sh - 1, 3)
        offset_sh_dc = self.sh_dc_max * torch.tanh(sh_dc_raw)
        offset_sh_rest = self.sh_rest_max * torch.tanh(sh_rest_raw)
        sh_dc_init = _rgb_to_sh(anchor_rgb)
        sh_dc_r = sh_dc_init + self.eta_sh_dc * offset_sh_dc
        sh_rest_r = self.eta_sh_rest * offset_sh_rest
        colors_r = torch.cat([sh_dc_r.unsqueeze(1), sh_rest_r], dim=1)  # [N, num_sh, 3]

        return {
            "means_r": means_r,
            "scales_r": scales_r,
            "quats_r": quats_r,
            "opacities_r": opacities_r,
            "colors_r": colors_r,
        }

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view: Any,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one view; returns (rgb [H,W,3], alpha [H,W])."""
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = _get_viewmat(c2w)
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=self.device).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)

        render, alpha, _ = self.renderer(
            means=render_params["means_r"],
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
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

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single forward: point cloud + one target -> pred_rgb, loss, render_params.
        batch must contain "pointcloud", "targets" (list with at least one {view, gt_image}).
        """
        pointcloud = batch["pointcloud"]
        targets = batch["targets"]
        if not targets:
            raise ValueError("MinimalStreetForward requires at least one target.")
        target = targets[0]
        view = target["view"]
        gt_image = target["gt_image"]
        if gt_image.dim() == 4:
            gt_image = gt_image.squeeze(0)
        height, width = gt_image.shape[0], gt_image.shape[1]

        means, anchor_rgb = self._pointcloud_to_means_rgb(pointcloud)
        if means.shape[0] == 0:
            raise ValueError("Empty point cloud in batch.")

        feat_3d_crop = self._build_3d_features(means, anchor_rgb)
        render_params = self._predict_render_params(means, anchor_rgb, feat_3d_crop)
        pred_rgb, _ = self._render_single_view(render_params, view, height, width)
        loss = F.l1_loss(pred_rgb, gt_image)

        return {
            "loss": loss,
            "pred_rgb": pred_rgb,
            "gt_image": gt_image,
            "render_params": render_params,
        }

    def train_step(self, batch: Dict) -> Dict[str, Any]:
        """One training step: forward, backward, step. Batch on device."""
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        loss = out["loss"]
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "pred_rgb": out["pred_rgb"].detach(),
            "gt_image": out["gt_image"].detach(),
        }


__all__ = ["MinimalStreetForward", "_get_grid_coords", "_interpolate_features"]
