"""
Minimal StreetForward Stage 1.1: NodeStateBackground + single target + GRU-style offset prediction.

Extends Stage 1 with GRU-style fusion (feat + param_embed -> GRU -> offsets) and h_cache_bg.
See docs/trainers/Minimal_StreetForward_Next_Steps_Stage1_1_GRU.md.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _pairwise_neighbor_distances,
    _quat_multiply,
    _quat_to_rotmat,
    _normalize_quat,
    _random_quat_tensor,
    _rgb_to_sh,
    _sh_to_rgb,
)
from models.streetforward.node_states import NodeStateBackground

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


def _quat_to_rot6d(quats: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (wxyz) to 6D rotation representation (first two columns of R)."""
    rot = _quat_to_rotmat(_normalize_quat(quats))
    rot6d = rot[..., :3, :2].reshape(quats.shape[:-1] + (6,))
    return rot6d


class MinimalStreetForwardStage1_1(nn.Module):
    """
    Stage 1.1: NodeStateBackground + single target + GRU-style offset prediction.
    Same as Stage 1 but offsets come from feat + param_embed -> GRU -> head; h_cache_bg maintained.
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

        self.update_node_state_interval = int(model_cfg.get("update_node_state_interval", 0))

        self.sh_degree = int(model_cfg.get("sh_degree", 1))
        self.voxel_size = float(model_cfg.get("voxel_size", 0.1))
        if not hasattr(config, "dataset") or not hasattr(config.dataset, "segment_aabb"):
            raise ValueError("config.dataset.segment_aabb is required for MinimalStreetForwardStage1_1.")
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

        # --- GRU-style (Stage 1.1) ---
        self.param_embed_input_dim = 17
        self.param_embed_dim = int(model_cfg.get("param_embed_dim", outdim))
        self.offset_gru_hidden_dim = int(model_cfg.get("offset_gru_hidden_dim", outdim))
        self.offset_gru_use_reset_gate = bool(model_cfg.get("offset_gru_use_reset_gate", True))

        self.mlp_params_embed = nn.Sequential(
            nn.Linear(self.param_embed_input_dim, self.param_embed_dim),
            nn.ReLU(),
            nn.Linear(self.param_embed_dim, self.param_embed_dim),
        ).to(device)
        self.param_embed_norm = nn.LayerNorm(self.param_embed_dim).to(device)

        gru_in_dim = outdim + self.param_embed_dim
        self.gru_update = nn.Linear(
            gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim
        ).to(device)
        self.gru_candidate = nn.Linear(
            gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim
        ).to(device)
        self.gru_reset = (
            nn.Linear(gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim).to(device)
            if self.offset_gru_use_reset_gate
            else None
        )
        if self.offset_gru_hidden_dim != outdim:
            self.gru_to_head = nn.Linear(self.offset_gru_hidden_dim, outdim).to(device)
        else:
            self.gru_to_head = nn.Identity()

        self.h_cache_bg: Dict[Tuple[int, int], torch.Tensor] = {}
        self._h_cache_signatures: Dict[str, Dict[Tuple[int, int], Tuple[int, ...]]] = {}

        # Offset heads (same as Stage 1; input dim = outdim)
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
            nn.Linear(32, 6),
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

        for m in (self.mlp_offset_pos, self.mlp_conv, self.mlp_opacity, self.gaussion_decoder):
            if isinstance(m, nn.Sequential) and len(m) > 0:
                last = m[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

        self.node_states_bg: Dict[Tuple[int, int], NodeStateBackground] = {}

        params = (
            list(self.sparse_conv.parameters())
            + list(self.mlp_params_embed.parameters())
            + list(self.param_embed_norm.parameters())
            + list(self.gru_update.parameters())
            + list(self.gru_candidate.parameters())
        )
        if self.gru_reset is not None:
            params += list(self.gru_reset.parameters())
        if not isinstance(self.gru_to_head, nn.Identity):
            params += list(self.gru_to_head.parameters())
        params += (
            list(self.mlp_offset_pos.parameters())
            + list(self.mlp_conv.parameters())
            + list(self.mlp_opacity.parameters())
            + list(self.gaussion_decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(config.optimizer.get("lr", 1e-3)),
            eps=float(config.optimizer.get("eps", 1e-15)),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        )

    def _cache_signature(self, node_state_bg: NodeStateBackground) -> Tuple[int, ...]:
        """Lightweight signature for h cache invalidation (point count)."""
        return (int(node_state_bg.means.shape[0]),)

    def _get_or_init_hidden(
        self,
        cache: Dict[Tuple[int, int], torch.Tensor],
        key: Tuple[int, int],
        num_points: int,
        node_state: Optional[NodeStateBackground] = None,
        node_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Fetch hidden from cache or init zeros; reset when size/signature changes."""
        h = cache.get(key)
        desired_sig = self._cache_signature(node_state) if node_state is not None else None
        prev_sig = None
        if node_type is not None and hasattr(self, "_h_cache_signatures"):
            prev_sig = self._h_cache_signatures.get(node_type, {}).get(key)
        reset_needed = (
            h is None
            or h.shape[0] != num_points
            or (prev_sig is not None and desired_sig is not None and prev_sig != desired_sig)
        )
        if reset_needed:
            h = torch.zeros(num_points, self.offset_gru_hidden_dim, device=self.device)
            cache[key] = h
        if node_type is not None and desired_sig is not None:
            self._h_cache_signatures.setdefault(node_type, {})[key] = desired_sig
        return h.detach()

    def reset_node_state(self) -> None:
        """Clear cached NodeState and h so next forward re-inits."""
        self.node_states_bg.clear()
        self.h_cache_bg.clear()
        if hasattr(self, "_h_cache_signatures"):
            self._h_cache_signatures.clear()

    def _normalize_params_for_embed(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize NodeState params to [N, 17] for embedding (align offsets_mixin)."""
        means = params["means"]
        scales_log = params["scales_log"]
        quats = params["quats"]
        opacity_logit = params["opacity_logit"]
        sh_dc = params["sh_dc"]
        sh_rest = params["sh_rest"]
        bbx_min = self.bbx_min.to(means.device)
        bbx_max = self.bbx_max.to(means.device)
        denom = (bbx_max - bbx_min).clamp(min=1e-6)
        means_norm = (means - bbx_min) / denom * 2.0 - 1.0
        scales_clamped = scales_log.clamp(-10.0, 10.0)
        scales_norm = F.layer_norm(scales_clamped, scales_clamped.shape[1:])
        rot6d = _quat_to_rot6d(quats)
        opacity_norm = torch.tanh(opacity_logit)
        sh_rest_energy = torch.linalg.norm(sh_rest.reshape(sh_rest.shape[0], -1), dim=-1, keepdim=True)
        param_vec = torch.cat(
            [means_norm, rot6d, scales_norm, opacity_norm, sh_dc, sh_rest_energy], dim=-1
        )
        return param_vec

    def _build_params_for_embed(
        self, node_state_bg: NodeStateBackground, coord_space: str = "world"
    ) -> Dict[str, torch.Tensor]:
        """Gather params for embedding (Stage 1.1 bg-only, no rigid/frame_idx)."""
        return {
            "means": node_state_bg.means,
            "scales_log": node_state_bg.scales_log,
            "quats": node_state_bg.quats,
            "opacity_logit": node_state_bg.opacity_logit,
            "sh_dc": node_state_bg.sh_dc,
            "sh_rest": node_state_bg.sh_rest,
        }

    def _predict_offsets_gru(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        mask_update_rigid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """GRU-style: param_embed + feat -> GRU -> head_input -> _predict_offsets. Stage 1.1 passes mask_update_rigid=None."""
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            identity_quat = torch.zeros(num_points, 4, device=device, dtype=dtype)
            identity_quat[..., 0] = 1.0
            num_sh = _num_sh_bases(self.sh_degree)
            zero_offsets = {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": identity_quat.clone(),
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }
            return zero_offsets, h_old

        param_vec = self._normalize_params_for_embed(params_for_embed)
        param_embed = self.param_embed_norm(self.mlp_params_embed(param_vec))
        x = torch.cat([feat, param_embed], dim=-1)
        hx = torch.cat([h_old, x], dim=-1)
        z = torch.sigmoid(self.gru_update(hx))
        if self.gru_reset is not None:
            r = torch.sigmoid(self.gru_reset(hx))
            h_cand = torch.tanh(self.gru_candidate(torch.cat([r * h_old, x], dim=-1)))
        else:
            h_cand = torch.tanh(self.gru_candidate(hx))
        h_new = (1.0 - z) * h_old + z * h_cand
        head_input = self.gru_to_head(h_new)
        offsets = self._predict_offsets(head_input)
        return offsets, h_new

    def _compute_initial_scales(self, means: torch.Tensor) -> torch.Tensor:
        """K-NN based initial scales_log [N, 3]."""
        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        return torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

    def _init_node_state_bg_from_pointcloud(
        self,
        pointcloud: Dict[str, np.ndarray],
        scene_id: int,
        segment_id: int,
    ) -> NodeStateBackground:
        """Init NodeStateBackground from pointcloud (background only, filtered to bbx)."""
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            if background.shape[1] >= 6:
                colors = background[:, 3:6].astype(np.float32)
                if colors.max() > 1.0 + 1e-3:
                    colors = colors / 255.0
            else:
                colors = np.zeros_like(points, dtype=np.float32)
        else:
            points = np.asarray(getattr(pointcloud, "points", np.zeros((0, 3))), dtype=np.float32)
            raw_colors = getattr(pointcloud, "colors", None)
            if raw_colors is not None:
                colors = np.asarray(raw_colors, dtype=np.float32)
                if colors.ndim == 1:
                    colors = np.expand_dims(colors, axis=0)
                if colors.shape[0] != points.shape[0]:
                    colors = np.zeros_like(points, dtype=np.float32)
            else:
                colors = np.zeros_like(points, dtype=np.float32)

        if len(points) == 0:
            raise ValueError(
                f"Empty point cloud for scene_id={scene_id}, segment_id={segment_id}. "
                "Check batch pointcloud and AABB."
            )
        bbx_min_np = self.bbx_min.cpu().numpy()
        bbx_max_np = self.bbx_max.cpu().numpy()
        in_crop = ((points >= bbx_min_np) & (points <= bbx_max_np)).all(axis=1)
        points = points[in_crop]
        colors = colors[in_crop]
        if len(points) == 0:
            raise ValueError(
                f"No points inside segment_aabb for scene_id={scene_id}, segment_id={segment_id}. "
                "Check segment_aabb and pointcloud."
            )
        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor
        if colors_rgb.dim() == 1:
            colors_rgb = colors_rgb.unsqueeze(-1).expand(-1, 3)
        elif colors_rgb.shape[1] != 3:
            colors_rgb = colors_rgb[:, :3]
        scales_log = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))
        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)
        return NodeStateBackground(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )

    def _get_or_init_node_state_bg(self, batch: Dict) -> NodeStateBackground:
        """Get or init NodeStateBackground for (scene_id, segment_id); cache in node_states_bg."""
        scene_id = batch["scene_id"]
        if torch.is_tensor(scene_id):
            scene_id = int(scene_id.item())
        segment_id = batch["segment_id"]
        if torch.is_tensor(segment_id):
            segment_id = int(segment_id.item()) if segment_id.numel() == 1 else int(segment_id[0].item())
        key = (scene_id, segment_id)
        if key in self.node_states_bg:
            return self.node_states_bg[key]
        pointcloud = batch["pointcloud"]
        node_state_bg = self._init_node_state_bg_from_pointcloud(pointcloud, scene_id, segment_id)
        self.node_states_bg[key] = node_state_bg
        return node_state_bg

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
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
        grid_coords = _get_grid_coords(
            means, self.bbx_min, self.bbx_max, vol_dim, self.voxel_size
        )
        feat_3d_crop = _interpolate_features(grid_coords, dense_volume)
        return feat_3d_crop

    def _predict_offsets(self, feat_3d_crop: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict offset dict from head input (same as Stage 1 / offsets_mixin)."""
        offset_pos = self.offset_max * torch.tanh(self.mlp_offset_pos(feat_3d_crop))
        scales_and_omega = self.mlp_conv(feat_3d_crop)
        offset_scales_raw, offset_omega_raw = scales_and_omega.split([3, 3], dim=-1)
        offset_scales = self.scale_max * torch.tanh(offset_scales_raw)
        offset_omega = self.omega_max * torch.tanh(offset_omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        offset_opacity = self.opacity_max * torch.tanh(self.mlp_opacity(feat_3d_crop))
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
        self,
        node_state_bg: NodeStateBackground,
        offsets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Render params = NodeState + eta * offsets (align Stage 1)."""
        num_points = node_state_bg.means.shape[0]
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)
        means_r = node_state_bg.means + self.eta_means * offsets["offset_pos"]
        scales_log_r = node_state_bg.scales_log + self.eta_scales * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state_bg.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state_bg.opacity_logit + self.eta_opacity * offsets["offset_opacity"]
        sh_dc_r = node_state_bg.sh_dc + self.eta_sh_dc * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state_bg.sh_rest + self.eta_sh_rest * sh_rest_offset
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

    def _update_node_state_bg(
        self,
        node_state_bg: NodeStateBackground,
        render_params: Dict[str, torch.Tensor],
    ) -> None:
        """Write render_params back to NodeState (detach, means clamped to bbx)."""
        with torch.no_grad():
            means_clamped = torch.clamp(
                render_params["means_r"].detach(),
                min=self.bbx_min,
                max=self.bbx_max,
            )
            node_state_bg.means.copy_(means_clamped)
            node_state_bg.scales_log.copy_(render_params["scales_log_r"].detach())
            node_state_bg.quats.copy_(render_params["quats_r"].detach())
            node_state_bg.opacity_logit.copy_(render_params["opacity_logit_r"].detach())
            node_state_bg.sh_dc.copy_(render_params["sh_dc_r"].detach())
            node_state_bg.sh_rest.copy_(render_params["sh_rest_r"].detach())

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view: Any,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one view; returns (rgb [H,W,3], alpha [H,W])."""
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        if c2w.dim() == 2:
            c2w = c2w.unsqueeze(0)
        viewmat = torch.linalg.inv(c2w)
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

    def _batch_key(self, batch: Dict) -> Tuple[int, int]:
        """(scene_id, segment_id) for cache key."""
        scene_id = batch["scene_id"]
        segment_id = batch["segment_id"]
        if torch.is_tensor(scene_id):
            scene_id = int(scene_id.item())
        if torch.is_tensor(segment_id):
            segment_id = int(segment_id.item()) if segment_id.numel() == 1 else int(segment_id[0].item())
        return (scene_id, segment_id)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Single forward: NodeState -> 3D features -> params_bg, h_old -> GRU -> offsets -> render -> loss.
        Returns include h_new for train_step to write h_cache_bg.
        """
        if "pointcloud" not in batch:
            raise ValueError("MinimalStreetForwardStage1_1 batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("MinimalStreetForwardStage1_1 requires at least one target.")
        target = targets[0]
        view = target["view"]
        gt_image = target["gt_image"]
        if gt_image.dim() == 4:
            gt_image = gt_image.squeeze(0)
        height, width = gt_image.shape[0], gt_image.shape[1]

        node_state_bg = self._get_or_init_node_state_bg(batch)
        key = self._batch_key(batch)
        means = node_state_bg.means
        anchor_rgb = _sh_to_rgb(node_state_bg.sh_dc)

        feat_3d_crop = self._build_3d_features(means, anchor_rgb)
        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old = self._get_or_init_hidden(
            self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg"
        )
        offsets, h_new = self._predict_offsets_gru(feat_3d_crop, params_bg, h_old, mask_update_rigid=None)
        render_params = self._render_params_from_offsets(node_state_bg, offsets)
        pred_rgb, _ = self._render_single_view(render_params, view, height, width)
        loss = compute_l1_loss_masked(
            pred_rgb, gt_image, None, sky_mask=target.get("sky_mask")
        )

        return {
            "loss": loss,
            "pred_rgb": pred_rgb,
            "gt_image": gt_image,
            "render_params": render_params,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new,
            "_cache_key": key,
        }

    def train_step(self, batch: Dict, step: Optional[int] = None) -> Dict[str, Any]:
        """One training step: forward, backward, step; write h_cache_bg; optionally update NodeState."""
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        loss = out["loss"]
        loss.backward()
        self.optimizer.step()
        if "_h_new_bg" in out and "_cache_key" in out:
            self.h_cache_bg[out["_cache_key"]] = out["_h_new_bg"].detach()
        if (
            self.update_node_state_interval > 0
            and step is not None
            and step % self.update_node_state_interval == 0
            and "_node_state_bg" in out
        ):
            self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
        return {
            "loss": loss.item(),
            "pred_rgb": out["pred_rgb"].detach(),
            "gt_image": out["gt_image"].detach(),
            "num_gaussians_bg": int(out["_node_state_bg"].means.shape[0]),
        }


__all__ = [
    "MinimalStreetForwardStage1_1",
    "_get_grid_coords",
    "_interpolate_features",
    "_quat_to_rot6d",
]
