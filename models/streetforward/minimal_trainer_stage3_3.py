"""
Minimal StreetForward Stage 3.3: Stage 3.2 + branch-decoupled bg/distant controls.

Key changes:
- strict model.branches.{bg,distant} config parsing (fast-fail)
- branch-specific init/limits/eta + per-branch freeze_means
- unit quaternion init for bg/distant node states
- scale init mode: isotropic | knn
- distant branch uses 2D-only feature path (no zeros_3d concat path)
- distant branch uses independent offset heads
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _normalize_quat,
    _pairwise_neighbor_distances,
    _quat_multiply,
    _sh_to_rgb,
)
from models.streetforward.minimal_trainer_stage3_2 import MinimalStreetForwardStage3_2
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params, _merge_params_bg_distant
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage3_3(MinimalStreetForwardStage3_2):
    """Stage 3.3 trainer built on top of Stage 3.2."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)

        branches = self._require_key(config.model, "branches", "model")
        bg = self._require_key(branches, "bg", "model.branches")
        distant = self._require_key(branches, "distant", "model.branches")

        self.bg_cfg = self._parse_branch_cfg(bg, "bg")
        self.distant_cfg = self._parse_branch_cfg(distant, "distant")

        # Use bg values as default behavior for inherited shared paths.
        self.offset_max = self.bg_cfg["limits"]["offset_max"]
        self.scale_max = self.bg_cfg["limits"]["scale_max"]
        self.omega_max = self.bg_cfg["limits"]["omega_max"]
        self.opacity_max = self.bg_cfg["limits"]["opacity_max"]
        self.sh_dc_max = self.bg_cfg["limits"]["sh_dc_max"]
        self.sh_rest_max = self.bg_cfg["limits"]["sh_rest_max"]
        self.eta_means = self.bg_cfg["eta"]["means"]
        self.eta_scales = self.bg_cfg["eta"]["scales"]
        self.eta_opacity = self.bg_cfg["eta"]["opacity"]
        self.eta_sh_dc = self.bg_cfg["eta"]["sh_dc"]
        self.eta_sh_rest = self.bg_cfg["eta"]["sh_rest"]

        self.bg_freeze_means = bool(self.bg_cfg["freeze_means"])
        self.distant_freeze_means = bool(self.distant_cfg["freeze_means"])
        self.distant_freeze_quat = bool(self.distant_cfg["mlp"]["freeze_quat"])

        feat_2d_dim = int(self.fused_in_dim - self.feat_3d_dim)
        if feat_2d_dim <= 0:
            raise ValueError(
                f"Invalid fused feature dims: fused_in_dim={self.fused_in_dim}, feat_3d_dim={self.feat_3d_dim}"
            )
        self.distant_feat_proj = nn.Linear(feat_2d_dim, self.fused_in_dim).to(device)

        # Distant independent heads.
        num_sh = _num_sh_bases(self.sh_degree)
        self.mlp_offset_pos_distant = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(device)
        self.mlp_conv_distant = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        ).to(device)
        self.mlp_opacity_distant = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        self.gaussion_decoder_distant = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)
        for m in (
            self.mlp_offset_pos_distant,
            self.mlp_conv_distant,
            self.mlp_opacity_distant,
            self.gaussion_decoder_distant,
        ):
            last = m[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        # Rebuild optimizer to include newly added modules.
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(config.optimizer.get("lr")),
            eps=float(config.optimizer.get("eps")),
            weight_decay=float(config.optimizer.get("weight_decay")),
        )

    @staticmethod
    def _require_key(obj, key: str, path: str):
        if not hasattr(obj, "get"):
            raise ValueError(f"{path} must be a mapping-like object.")
        value = obj.get(key)
        if value is None:
            raise ValueError(f"Missing required config: {path}.{key}")
        return value

    def _parse_branch_cfg(self, branch_cfg, name: str) -> Dict[str, Any]:
        init_cfg = self._require_key(branch_cfg, "init", f"model.branches.{name}")
        scale_init = self._require_key(init_cfg, "scale_init", f"model.branches.{name}.init")
        limits = self._require_key(branch_cfg, "limits", f"model.branches.{name}")
        eta = self._require_key(branch_cfg, "eta", f"model.branches.{name}")
        mlp = self._require_key(branch_cfg, "mlp", f"model.branches.{name}")
        freeze_means = self._require_key(branch_cfg, "freeze_means", f"model.branches.{name}")

        mode = self._require_key(scale_init, "mode", f"model.branches.{name}.init.scale_init")
        iso = self._require_key(scale_init, "isotropic_log_value", f"model.branches.{name}.init.scale_init")
        knn_k = self._require_key(scale_init, "knn_k", f"model.branches.{name}.init.scale_init")
        knn_bias = self._require_key(scale_init, "knn_log_scale_bias", f"model.branches.{name}.init.scale_init")
        opacity_init = self._require_key(init_cfg, "opacity_init", f"model.branches.{name}.init")

        parsed = {
            "init": {
                "scale_init_mode": str(mode),
                "isotropic_log_value": float(iso),
                "knn_k": int(knn_k),
                "knn_log_scale_bias": float(knn_bias),
                "opacity_init": float(opacity_init),
            },
            "limits": {
                "offset_max": float(self._require_key(limits, "offset_max", f"model.branches.{name}.limits")),
                "scale_max": float(self._require_key(limits, "scale_max", f"model.branches.{name}.limits")),
                "omega_max": float(self._require_key(limits, "omega_max", f"model.branches.{name}.limits")),
                "opacity_max": float(self._require_key(limits, "opacity_max", f"model.branches.{name}.limits")),
                "sh_dc_max": float(self._require_key(limits, "sh_dc_max", f"model.branches.{name}.limits")),
                "sh_rest_max": float(self._require_key(limits, "sh_rest_max", f"model.branches.{name}.limits")),
            },
            "eta": {
                "means": float(self._require_key(eta, "means", f"model.branches.{name}.eta")),
                "scales": float(self._require_key(eta, "scales", f"model.branches.{name}.eta")),
                "opacity": float(self._require_key(eta, "opacity", f"model.branches.{name}.eta")),
                "sh_dc": float(self._require_key(eta, "sh_dc", f"model.branches.{name}.eta")),
                "sh_rest": float(self._require_key(eta, "sh_rest", f"model.branches.{name}.eta")),
            },
            "mlp": {
                "hidden_dim": int(self._require_key(mlp, "hidden_dim", f"model.branches.{name}.mlp")),
                "use_3d_feat": bool(self._require_key(mlp, "use_3d_feat", f"model.branches.{name}.mlp")),
                "use_2d_feat": bool(self._require_key(mlp, "use_2d_feat", f"model.branches.{name}.mlp")),
                "freeze_quat": bool(self._require_key(mlp, "freeze_quat", f"model.branches.{name}.mlp")),
            },
            "freeze_means": bool(freeze_means),
        }
        if parsed["init"]["scale_init_mode"] not in {"isotropic", "knn"}:
            raise ValueError(
                f"model.branches.{name}.init.scale_init.mode must be one of ['isotropic','knn'], got "
                f"{parsed['init']['scale_init_mode']!r}"
            )
        return parsed

    def _compute_initial_scales_by_cfg(self, means: torch.Tensor, init_cfg: Dict[str, Any]) -> torch.Tensor:
        n = int(means.shape[0])
        if init_cfg["scale_init_mode"] == "isotropic":
            return torch.full((n, 3), float(init_cfg["isotropic_log_value"]), device=means.device, dtype=means.dtype)

        k = int(init_cfg["knn_k"])
        if n <= 1:
            base = torch.full((n, 3), float(init_cfg["isotropic_log_value"]), device=means.device, dtype=means.dtype)
        else:
            distances = _pairwise_neighbor_distances(means, k=min(k, n - 1))
            avg_dist = distances.mean(dim=-1, keepdim=True).clamp(min=1e-3)
            base = torch.log(avg_dist).repeat(1, 3)
        return base + float(init_cfg["knn_log_scale_bias"])

    def _init_node_state_from_arrays_branch(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        state_cls: type,
        branch_name: str,
    ):
        if len(points) == 0:
            raise ValueError("Empty point cloud for node state.")
        branch_cfg = self.bg_cfg if branch_name == "bg" else self.distant_cfg
        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        if colors_tensor.dim() == 1:
            colors_tensor = colors_tensor.unsqueeze(-1).expand(-1, 3)
        elif colors_tensor.shape[1] != 3:
            colors_tensor = colors_tensor[:, :3]

        from models.streetforward.math_utils import _rgb_to_sh

        scales_log = self._compute_initial_scales_by_cfg(means, branch_cfg["init"])
        quats = torch.zeros((means.shape[0], 4), device=self.device, dtype=means.dtype)
        quats[:, 0] = 1.0
        opacity_logit = torch.logit(
            torch.full((means.shape[0], 1), float(branch_cfg["init"]["opacity_init"]), device=self.device)
        )
        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_tensor)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        return state_cls(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )

    def _get_or_init_node_states_bg_distant(
        self, batch: Dict
    ) -> Tuple[NodeStateBackground, Optional[NodeStateDistant]]:
        key = self._batch_key(batch)
        if key in self.node_states_bg and key in self.node_states_distant:
            return self.node_states_bg[key], self.node_states_distant.get(key)

        pointcloud = batch["pointcloud"]
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            colors = background[:, 3:6].astype(np.float32) if background.shape[1] >= 6 else np.zeros_like(points)
        else:
            points = np.asarray(getattr(pointcloud, "points", np.zeros((0, 3))), dtype=np.float32)
            raw_colors = getattr(pointcloud, "colors", None)
            colors = np.asarray(raw_colors, dtype=np.float32) if raw_colors is not None else np.zeros_like(points)

        if len(points) == 0:
            raise ValueError("Empty point cloud for stage3_3 node-state init.")

        crop_min = self.bbx_min.cpu().numpy()
        crop_max = self.bbx_max.cpu().numpy()
        in_crop = ((points >= crop_min) & (points <= crop_max)).all(axis=1)
        fg_points, fg_colors = points[in_crop], colors[in_crop]
        distant_points, distant_colors = points[~in_crop], colors[~in_crop]

        if len(fg_points) == 0:
            raise ValueError("No points inside segment_aabb for stage3_3 bg node-state init.")

        node_state_bg = self._init_node_state_from_arrays_branch(fg_points, fg_colors, NodeStateBackground, "bg")
        node_state_distant: Optional[NodeStateDistant] = None
        if len(distant_points) > 0:
            node_state_distant = self._init_node_state_from_arrays_branch(
                distant_points, distant_colors, NodeStateDistant, "distant"
            )

        self.node_states_bg[key] = node_state_bg
        self.node_states_distant[key] = node_state_distant
        return node_state_bg, node_state_distant

    def _predict_offsets_with_heads(
        self,
        feat_head: torch.Tensor,
        *,
        limits: Dict[str, float],
        mlp_offset_pos: Optional[nn.Module],
        mlp_conv: nn.Module,
        mlp_opacity: nn.Module,
        gaussion_decoder: nn.Module,
        freeze_quat: bool,
        omit_position_offset: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if omit_position_offset:
            offset_pos = torch.zeros(feat_head.shape[0], 3, device=feat_head.device, dtype=feat_head.dtype)
        else:
            if mlp_offset_pos is None:
                raise ValueError("mlp_offset_pos is required when omit_position_offset is False.")
            offset_pos = limits["offset_max"] * torch.tanh(mlp_offset_pos(feat_head))
        scales_and_omega = mlp_conv(feat_head)
        offset_scales_raw, offset_omega_raw = scales_and_omega.split([3, 3], dim=-1)
        offset_scales = limits["scale_max"] * torch.tanh(offset_scales_raw)
        offset_omega = limits["omega_max"] * torch.tanh(offset_omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        if freeze_quat:
            # Keep autograd connection while functionally freezing quaternion update.
            offset_quat = _axis_angle_to_quat(offset_omega * 0.0)
        offset_opacity = limits["opacity_max"] * torch.tanh(mlp_opacity(feat_head))
        sh_raw = gaussion_decoder(feat_head)
        sh_dc_raw = sh_raw[:, :3]
        sh_rest_raw = sh_raw[:, 3:]
        offset_sh_dc = limits["sh_dc_max"] * torch.tanh(sh_dc_raw)
        offset_sh_rest = limits["sh_rest_max"] * torch.tanh(sh_rest_raw)
        offset_sh = torch.cat([offset_sh_dc, offset_sh_rest], dim=-1)
        return {
            "offset_pos": offset_pos,
            "offset_scales": offset_scales,
            "offset_quat": offset_quat,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
        }

    def _predict_offsets_gru_distant(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            identity_quat = torch.zeros(num_points, 4, device=device, dtype=dtype)
            identity_quat[:, 0] = 1.0
            num_sh = _num_sh_bases(self.sh_degree)
            return {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": identity_quat,
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }, h_old

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
        head_input = self._apply_gru_head_rms(head_input, None)
        offsets = self._predict_offsets_with_heads(
            head_input,
            limits=self.distant_cfg["limits"],
            mlp_offset_pos=self.mlp_offset_pos_distant,
            mlp_conv=self.mlp_conv_distant,
            mlp_opacity=self.mlp_opacity_distant,
            gaussion_decoder=self.gaussion_decoder_distant,
            freeze_quat=self.distant_freeze_quat,
        )
        return offsets, h_new

    def _render_params_from_offsets_bg(
        self, node_state_bg: NodeStateBackground, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_points = int(node_state_bg.means.shape[0])
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)

        if self.bg_freeze_means:
            # Keep autograd connection for proxy backward while functionally freezing means.
            means_r = node_state_bg.means + offsets["offset_pos"] * 0.0
        else:
            means_r = node_state_bg.means + self.bg_cfg["eta"]["means"] * offsets["offset_pos"]
        scales_log_r = node_state_bg.scales_log + self.bg_cfg["eta"]["scales"] * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state_bg.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state_bg.opacity_logit + self.bg_cfg["eta"]["opacity"] * offsets["offset_opacity"]
        sh_dc_r = node_state_bg.sh_dc + self.bg_cfg["eta"]["sh_dc"] * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state_bg.sh_rest + self.bg_cfg["eta"]["sh_rest"] * sh_rest_offset
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

    def _render_params_from_offsets_distant(
        self, node_state_distant: NodeStateDistant, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_points = int(node_state_distant.means.shape[0])
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)

        if self.distant_freeze_means:
            # Keep autograd connection for proxy backward while functionally freezing means.
            means_r = node_state_distant.means + offsets["offset_pos"] * 0.0
        else:
            means_r = node_state_distant.means + self.distant_cfg["eta"]["means"] * offsets["offset_pos"]
        scales_log_r = node_state_distant.scales_log + self.distant_cfg["eta"]["scales"] * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state_distant.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state_distant.opacity_logit + self.distant_cfg["eta"]["opacity"] * offsets["offset_opacity"]
        sh_dc_r = node_state_distant.sh_dc + self.distant_cfg["eta"]["sh_dc"] * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state_distant.sh_rest + self.distant_cfg["eta"]["sh_rest"] * sh_rest_offset
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

    def forward(self, batch: Dict) -> Dict[str, Any]:
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 3 requires at least one target.")
        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        if not source_views or not source_images:
            raise ValueError(
                "Stage 3 requires source_views and source_images. "
                "Use convert_batch_to_minimal_format(..., include_source_for_2d=True)."
            )

        node_state_bg, node_state_distant = self._get_or_init_node_states_bg_distant(batch)
        key = self._batch_key(batch)

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)

        sample_img = source_images[0]
        height = int(sample_img.shape[0] if sample_img.dim() == 3 else sample_img.shape[1])
        width = int(sample_img.shape[1] if sample_img.dim() == 3 else sample_img.shape[2])

        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )

        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)

        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru(feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None)
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant"
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru_distant(
                feat_distant_input, params_distant, h_old_distant
            )
            render_params_distant = self._render_params_from_offsets_distant(node_state_distant, offsets_distant)

        if not self.training:
            if render_params_distant is not None:
                merged = {
                    "means_r": torch.cat([render_params_bg["means_r"], render_params_distant["means_r"]]),
                    "scales_r": torch.cat([render_params_bg["scales_r"], render_params_distant["scales_r"]]),
                    "quats_r": torch.cat([render_params_bg["quats_r"], render_params_distant["quats_r"]]),
                    "opacities_r": torch.cat([render_params_bg["opacities_r"], render_params_distant["opacities_r"]]),
                    "colors_r": torch.cat([render_params_bg["colors_r"], render_params_distant["colors_r"]]),
                }
            else:
                merged = {
                    "means_r": render_params_bg["means_r"],
                    "scales_r": render_params_bg["scales_r"],
                    "quats_r": render_params_bg["quats_r"],
                    "opacities_r": render_params_bg["opacities_r"],
                    "colors_r": render_params_bg["colors_r"],
                }

            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []
            multi_result = self._render_multi_view(merged, targets)
            if multi_result is not None:
                pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
                acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
                pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
                for i, target in enumerate(targets):
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    pred_rgbs.append(pred_stack[i])
                    gt_images.append(gt_image)
            else:
                for target in targets:
                    view = target["view"]
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    h, w = gt_image.shape[0], gt_image.shape[1]
                    pred_rgb, acc = self._render_single_view(merged, view, h, w)
                    pred_rgb = self._composite_sky(pred_rgb, acc, target)
                    pred_rgbs.append(pred_rgb)
                    gt_images.append(gt_image)

            loss = torch.tensor(0.0, device=self.device)
            return {
                "loss": loss,
                "render_params": render_params_bg,
                "proxies": None,
                "_node_state_bg": node_state_bg,
                "_h_new_bg": h_new_bg,
                "_cache_key": key,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_h_new_distant": h_new_distant,
                "_node_state_distant": node_state_distant,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        merged_for_render = _merge_params_bg_distant(proxies_bg, proxies_distant)

        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        opacities: List[torch.Tensor] = []

        multi_result = self._render_multi_view(merged_for_render, targets)
        if multi_result is not None:
            pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
            acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
            pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
            for i, target in enumerate(targets):
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                pred_rgbs.append(pred_stack[i])
                gt_images.append(gt_image)
                opacities.append(acc_stack[i])
        else:
            for target in targets:
                view = target["view"]
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                h, w = gt_image.shape[0], gt_image.shape[1]
                pred_rgb, acc = self._render_single_view(merged_for_render, view, h, w)
                pred_rgb = self._composite_sky(pred_rgb, acc, target)
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                opacities.append(acc.squeeze(-1) if acc.dim() == 3 and acc.shape[-1] == 1 else acc)

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_rgb_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        loss_total_list: List[torch.Tensor] = []
        for i, target in enumerate(targets):
            pred_rgb = pred_rgbs[i]
            gt_image = gt_images[i]
            opacity = opacities[i].to(self.device).float()
            if opacity.dim() == 3 and opacity.shape[-1] == 1:
                opacity = opacity.squeeze(-1)
            H, W = int(gt_image.shape[0]), int(gt_image.shape[1])
            valid_loss_mask = self._valid_loss_mask_from_target(target, height=H, width=W)

            from models.streetforward.metrics import compute_l1_loss_masked, compute_ssim_loss_masked

            l1_i = compute_l1_loss_masked(pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None)
            ssim_i = compute_ssim_loss_masked(
                pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None, data_range=1.0
            )
            rgb_i = self.loss_w_l1 * l1_i + self.loss_w_ssim * ssim_i

            sky_mask = target.get("sky_mask")
            if sky_mask is None:
                raise ValueError("Stage 3.3 requires target['sky_mask'] (1=sky, 0=non-sky).")
            sm = sky_mask.to(self.device).float()
            if sm.dim() == 3:
                sm = sm.squeeze(-1)
            gt_occupied = (1.0 - sm) * valid_loss_mask
            pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
            mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)

            p = opacity.clamp(1e-6, 1.0 - 1e-6)
            entropy_map = -p * torch.log(p)
            entropy_i = self.loss_w_opacity_entropy * self._masked_mean(entropy_map, valid_loss_mask)

            total_i = rgb_i + mask_i + entropy_i
            loss_l1_list.append(l1_i)
            loss_ssim_list.append(ssim_i)
            loss_rgb_list.append(rgb_i)
            loss_mask_list.append(mask_i)
            loss_entropy_list.append(entropy_i)
            loss_total_list.append(total_i)

        loss_total = torch.stack(loss_total_list).mean() if loss_total_list else torch.tensor(0.0, device=self.device)
        loss_l1 = torch.stack(loss_l1_list).mean() if loss_l1_list else loss_total * 0.0
        loss_ssim = torch.stack(loss_ssim_list).mean() if loss_ssim_list else loss_total * 0.0
        loss_rgb = torch.stack(loss_rgb_list).mean() if loss_rgb_list else loss_total * 0.0
        loss_mask = torch.stack(loss_mask_list).mean() if loss_mask_list else loss_total * 0.0
        loss_entropy = torch.stack(loss_entropy_list).mean() if loss_entropy_list else loss_total * 0.0

        return {
            "loss": loss_total,
            "loss_l1": loss_l1,
            "loss_ssim": loss_ssim,
            "loss_rgb": loss_rgb,
            "loss_mask": loss_mask,
            "loss_opacity_entropy": loss_entropy,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new_bg,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
            "_render_params_distant": render_params_distant,
            "_proxies_distant": proxies_distant,
            "_h_new_distant": h_new_distant,
            "_node_state_distant": node_state_distant,
        }


__all__ = ["MinimalStreetForwardStage3_3"]

