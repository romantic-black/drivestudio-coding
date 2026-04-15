"""
Minimal StreetForward Stage 4.3: Stage 4.2 + sky GS shell (hemisphere, two-pass render) + one-pass sky 2D.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from datasets.base.pixel_source import get_rays
from models.feature_extractors import FeatureBackprojector
from models.feature_extractors.alpha_t_extractor import AlphaTWeightExtractor, _get_viewmat
from models.streetforward.math_utils import _num_sh_bases, _normalize_quat, _quat_multiply, _rgb_to_sh, _sh_to_rgb
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    _backward_to_render_params_bg_rigid_distant_sky,
    _merge_params_bg_rigid_distant,
    spatial_hw_from_image_tensor,
)
from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid, NodeStateSky
from models.streetforward.sky_shell_init import SKY_UP_MULTISCENE, fibonacci_shell_means

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePolicy:
    do_backward: bool
    do_optimizer_step: bool
    update_hidden_cache: bool
    writeback_node_state: bool
    reset_node_state_after_block: bool


def _composite_sky_gs(pred_rgb: torch.Tensor, opacity: torch.Tensor, rgb_sky: torch.Tensor) -> torch.Tensor:
    if opacity.dim() == 3 and opacity.shape[-1] == 1:
        opacity = opacity.squeeze(-1)
    return pred_rgb + rgb_sky * (1.0 - opacity.clamp(0.0, 1.0)).unsqueeze(-1)


class MinimalStreetForwardStage4_3(MinimalStreetForwardStage4_2):
    """Stage4.2 + sky hemisphere GS (no cubemap); one-pass includes sky; Pass A scene, Pass B sky."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        if "sky_model" in self._modules:
            del self._modules["sky_model"]

        branches = self._require_key(config.model, "branches", "model")
        bg_yaml = self._require_key(branches, "bg", "model.branches")
        distant_yaml = self._require_key(branches, "distant", "model.branches")
        sky_yaml = self._require_key(branches, "sky", "model.branches")

        self.bg_src_backproject_support_min = float(
            self._require_key(bg_yaml, "src_backproject_support_min", "model.branches.bg")
        )
        self.distant_src_backproject_support_min = float(
            self._require_key(distant_yaml, "src_backproject_support_min", "model.branches.distant")
        )
        self.bg_enable_selective_update = bool(
            self._require_key(bg_yaml, "enable_selective_update", "model.branches.bg")
        )
        self.distant_enable_selective_update = bool(
            self._require_key(distant_yaml, "enable_selective_update", "model.branches.distant")
        )
        self.sky_enable_selective_update = bool(
            self._require_key(sky_yaml, "enable_selective_update", "model.branches.sky")
        )

        sky_geom = self._require_key(config.model, "sky", "model")
        self.sky_origin_mode = str(self._require_key(sky_geom, "origin_mode", "model.sky"))
        if self.sky_origin_mode != "camera_centered_rotation_only":
            raise ValueError("model.sky.origin_mode must be 'camera_centered_rotation_only'.")
        if "center" in sky_geom:
            raise ValueError("model.sky.center is removed. Use model.sky.center_local instead.")
        center_raw = self._require_key(sky_geom, "center_local", "model.sky")
        if not hasattr(center_raw, "__len__") or len(center_raw) != 3:
            raise ValueError("model.sky.center_local must be a length-3 list/tuple.")
        self.sky_center_local = torch.tensor(list(center_raw), dtype=torch.float32, device=device)
        self.sky_resolution = int(self._require_key(sky_geom, "resolution", "model.sky"))
        self.sky_radius = float(self._require_key(sky_geom, "radius", "model.sky"))

        self.sky_hemisphere = bool(self._require_key(sky_geom, "hemisphere", "model.sky"))
        up_raw = sky_geom.get("hemisphere_up") if hasattr(sky_geom, "get") else None
        if up_raw is None:
            self.sky_hemisphere_up: Tuple[float, float, float] = tuple(float(x) for x in SKY_UP_MULTISCENE)
        else:
            if not hasattr(up_raw, "__len__") or len(up_raw) != 3:
                raise ValueError("model.sky.hemisphere_up must be a length-3 list/tuple when set.")
            self.sky_hemisphere_up = tuple(float(up_raw[i]) for i in range(3))

        self.sky_cfg = self._parse_branch_cfg(sky_yaml, "sky")
        if bool(self.sky_cfg["mlp"]["use_3d_feat"]):
            raise ValueError("Stage4.3 requires model.branches.sky.mlp.use_3d_feat=false")
        if not bool(self.sky_cfg["mlp"]["use_2d_feat"]):
            raise ValueError("Stage4.3 requires model.branches.sky.mlp.use_2d_feat=true")
        self.sky_src_backproject_support_min = float(
            self._require_key(sky_yaml, "src_backproject_support_min", "model.branches.sky")
        )
        self.sky_freeze_means = bool(self.sky_cfg["freeze_means"])
        self.sky_freeze_quat = bool(self.sky_cfg["mlp"]["freeze_quat"])
        if not self.sky_freeze_means:
            raise ValueError(
                "Stage 4.3 requires model.branches.sky.freeze_means=true (fixed sky shell means; see Stage4.3 docs)."
            )
        if not self.sky_freeze_quat:
            raise ValueError(
                "Stage 4.3 requires model.branches.sky.mlp.freeze_quat=true (fixed sky quaternions; see Stage4.3 docs)."
            )

        self.sky_feat_proj = nn.Linear(self.rigid_feat_in_dim, self.fused_in_dim).to(device)
        num_sh = _num_sh_bases(self.sh_degree)
        fd = self.fused_in_dim
        self.mlp_conv_sky = nn.Sequential(
            nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6)
        ).to(device)
        self.mlp_opacity_sky = nn.Sequential(
            nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(device)
        self.gaussion_decoder_sky = nn.Sequential(
            nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3 * num_sh)
        ).to(device)

        self.node_states_sky: Dict[Tuple[int, int], NodeStateSky] = {}
        self.h_cache_sky: Dict[Tuple[int, int], torch.Tensor] = {}

        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(config.optimizer.get("lr", 1e-3)),
            eps=float(config.optimizer.get("eps", 1e-15)),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        )

    def _get_or_init_node_state_sky(self, batch: Dict) -> NodeStateSky:
        key = self._batch_key(batch)
        if key in self.node_states_sky:
            return self.node_states_sky[key]
        sky_origin = self.sky_center_local.to(self.device)
        means = fibonacci_shell_means(
            self.sky_resolution,
            self.sky_radius,
            sky_origin,
            hemisphere=self.sky_hemisphere,
            device=self.device,
            dtype=torch.float32,
            up=self.sky_hemisphere_up,
        )
        init_cfg = self.sky_cfg["init"]
        scales_log = self._compute_initial_scales_by_cfg(means, init_cfg)
        n = int(means.shape[0])
        quats = torch.zeros(n, 4, device=self.device, dtype=means.dtype)
        quats[:, 0] = 1.0
        opacity_logit = torch.logit(
            torch.full((n, 1), float(init_cfg["opacity_init"]), device=self.device, dtype=means.dtype)
        )
        colors = torch.full((n, 3), 0.5, device=self.device, dtype=means.dtype)
        sh_dc = _rgb_to_sh(colors)
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest = torch.zeros((n, num_sh - 1, 3), device=self.device, dtype=means.dtype)
        ns = NodeStateSky(
            means=means,
            scales_log=scales_log,
            quats=quats,
            opacity_logit=opacity_logit,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )
        self.node_states_sky[key] = ns
        return ns

    def _sky_viewmat_from_view(self, view) -> torch.Tensor:
        cam_ctw = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = _get_viewmat(cam_ctw).clone()
        viewmat[..., :3, 3] = 0.0
        return viewmat

    def _sky_viewmats_from_views(self, views: List[Any]) -> torch.Tensor:
        mats = [self._sky_viewmat_from_view(v) for v in views]
        return torch.cat(mats, dim=0)

    def _prepare_gaussians_sky(self, node_state_sky: NodeStateSky) -> Dict[str, torch.Tensor]:
        num_sh = _num_sh_bases(self.sh_degree)
        means = node_state_sky.means
        quats = node_state_sky.quats
        scales = torch.exp(node_state_sky.scales_log)
        opacities = torch.sigmoid(node_state_sky.opacity_logit).squeeze(-1)
        colors = torch.cat([node_state_sky.sh_dc[:, None, :], node_state_sky.sh_rest], dim=1)
        return {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
        }

    def _render_params_from_offsets_sky(
        self, node_state_sky: NodeStateSky, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_points = int(node_state_sky.means.shape[0])
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)
        # Fixed sky shell geometry: means/quats are frozen and may not require grad.
        # Proxy backward skips render tensors with requires_grad=False (see _append_backward_pair).
        means_r = node_state_sky.means + offsets["offset_pos"] * 0.0
        scales_log_r = node_state_sky.scales_log + self.sky_cfg["eta"]["scales"] * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state_sky.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state_sky.opacity_logit + self.sky_cfg["eta"]["opacity"] * offsets["offset_opacity"]
        sh_dc_r = node_state_sky.sh_dc + self.sky_cfg["eta"]["sh_dc"] * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state_sky.sh_rest + self.sky_cfg["eta"]["sh_rest"] * sh_rest_offset
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

    def _render_sky_single_view(
        self,
        sky_render_params: Dict[str, torch.Tensor],
        view,
        height: int,
        width: int,
    ) -> torch.Tensor:
        sky_viewmat = self._sky_viewmat_from_view(view)
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=self.device).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)
        render, _, _ = self.renderer(
            means=sky_render_params["means_r"],
            quats=sky_render_params["quats_r"],
            scales=sky_render_params["scales_r"],
            opacities=sky_render_params["opacities_r"],
            colors=sky_render_params["colors_r"],
            viewmats=sky_viewmat,
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
        return render[:, ..., :3].squeeze(0)

    def _predict_offsets_gru_sky_masked(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        mask_update_sky: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            num_sh = _num_sh_bases(self.sh_degree)
            offsets = {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": self._identity_quat(num_points, device, dtype),
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }
            h_new = h_old
            if mask_update_sky is not None:
                gate = mask_update_sky.to(dtype=dtype, device=device).unsqueeze(-1).detach()
                identity = self._identity_quat(num_points, device, dtype)
                offsets["offset_pos"] = offsets["offset_pos"] * gate
                offsets["offset_scales"] = offsets["offset_scales"] * gate
                offsets["offset_quat"] = torch.where(
                    gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity
                )
                offsets["offset_opacity"] = offsets["offset_opacity"] * gate
                offsets["offset_sh"] = offsets["offset_sh"] * gate
                h_new = h_old * (1.0 - gate) + h_new * gate
            return offsets, h_new

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
        head_input = self._apply_gru_head_rms(head_input, mask_update_sky)
        offsets = self._predict_offsets_with_heads(
            head_input,
            limits=self.sky_cfg["limits"],
            mlp_offset_pos=None,
            mlp_conv=self.mlp_conv_sky,
            mlp_opacity=self.mlp_opacity_sky,
            gaussion_decoder=self.gaussion_decoder_sky,
            freeze_quat=self.sky_freeze_quat,
            omit_position_offset=True,
        )

        if mask_update_sky is not None:
            gate = mask_update_sky.to(dtype=offsets["offset_pos"].dtype, device=offsets["offset_pos"].device).unsqueeze(-1).detach()
            identity = self._identity_quat(offsets["offset_quat"].shape[0], offsets["offset_quat"].device, offsets["offset_quat"].dtype)
            offsets["offset_pos"] = offsets["offset_pos"] * gate
            offsets["offset_scales"] = offsets["offset_scales"] * gate
            offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
            offsets["offset_opacity"] = offsets["offset_opacity"] * gate
            offsets["offset_sh"] = offsets["offset_sh"] * gate
            h_new = h_old * (1.0 - gate) + h_new * gate
        return offsets, h_new

    def _update_node_state_sky_subset(
        self,
        node_state_sky: NodeStateSky,
        render_params: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            # Fixed shell: do not write means/quats from render_params.
            node_state_sky.scales_log[valid_idx] = render_params["scales_log_r"][valid_idx].detach()
            node_state_sky.opacity_logit[valid_idx] = render_params["opacity_logit_r"][valid_idx].detach()
            node_state_sky.sh_dc[valid_idx] = render_params["sh_dc_r"][valid_idx].detach()
            node_state_sky.sh_rest[valid_idx] = render_params["sh_rest_r"][valid_idx].detach()

    def _update_node_state_sky(
        self,
        node_state_sky: NodeStateSky,
        render_params: Dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            node_state_sky.scales_log.copy_(render_params["scales_log_r"].detach())
            node_state_sky.opacity_logit.copy_(render_params["opacity_logit_r"].detach())
            node_state_sky.sh_dc.copy_(render_params["sh_dc_r"].detach())
            node_state_sky.sh_rest.copy_(render_params["sh_rest_r"].detach())

    def _compute_branch_grad_norms(self) -> Dict[str, float]:
        base = super()._compute_branch_grad_norms()
        sq_sky = 0.0
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            if "sky_feat_proj" in name or "mlp_conv_sky" in name or "mlp_opacity_sky" in name or "gaussion_decoder_sky" in name:
                sq_sky += float(param.grad.detach().float().pow(2).sum().item())
        base["grad_norm_sky"] = float(sq_sky ** 0.5)
        return base

    def _collect_offset_stats(
        self,
        offsets_bg: Optional[Dict[str, torch.Tensor]],
        offsets_rigid: Optional[Dict[str, torch.Tensor]],
        offsets_sky: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        out = super()._collect_offset_stats(offsets_bg, offsets_rigid)
        if offsets_sky is None:
            return out
        for key in ("offset_pos", "offset_scales", "offset_opacity"):
            stats = self._stat_tensor(offsets_sky.get(key))
            out[f"sky_{key}_mean"] = stats["mean"]
            out[f"sky_{key}_std"] = stats["std"]
            out[f"sky_{key}_max"] = stats["max"]
        stats_sh = self._stat_tensor(offsets_sky.get("offset_sh"))
        out["sky_offset_sh_mean"] = stats_sh["mean"]
        out["sky_offset_sh_std"] = stats_sh["std"]
        out["sky_offset_sh_max"] = stats_sh["max"]
        return out

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self.node_states_sky.clear()
        self.h_cache_sky.clear()

    @staticmethod
    def _identity_quat(num_points: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = torch.zeros(num_points, 4, device=device, dtype=dtype)
        q[:, 0] = 1.0
        return q

    def _predict_offsets_gru_distant_masked(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        mask_update_distant: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Distant-specific heads + optional mask gate (hidden/head/offset)."""
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            num_sh = _num_sh_bases(self.sh_degree)
            offsets = {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": self._identity_quat(num_points, device, dtype),
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }
            h_new = h_old
            if mask_update_distant is not None:
                gate = mask_update_distant.to(dtype=dtype, device=device).unsqueeze(-1).detach()
                identity = self._identity_quat(num_points, device, dtype)
                offsets["offset_pos"] = offsets["offset_pos"] * gate
                offsets["offset_scales"] = offsets["offset_scales"] * gate
                offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
                offsets["offset_opacity"] = offsets["offset_opacity"] * gate
                offsets["offset_sh"] = offsets["offset_sh"] * gate
                h_new = h_old * (1.0 - gate) + h_new * gate
            return offsets, h_new

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
        head_input = self._apply_gru_head_rms(head_input, mask_update_distant)
        offsets = self._predict_offsets_with_heads(
            head_input,
            limits=self.distant_cfg["limits"],
            mlp_offset_pos=self.mlp_offset_pos_distant,
            mlp_conv=self.mlp_conv_distant,
            mlp_opacity=self.mlp_opacity_distant,
            gaussion_decoder=self.gaussion_decoder_distant,
            freeze_quat=self.distant_freeze_quat,
        )

        if mask_update_distant is not None:
            gate = mask_update_distant.to(dtype=offsets["offset_pos"].dtype, device=offsets["offset_pos"].device).unsqueeze(-1).detach()
            identity = self._identity_quat(offsets["offset_quat"].shape[0], offsets["offset_quat"].device, offsets["offset_quat"].dtype)
            offsets["offset_pos"] = offsets["offset_pos"] * gate
            offsets["offset_scales"] = offsets["offset_scales"] * gate
            offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
            offsets["offset_opacity"] = offsets["offset_opacity"] * gate
            offsets["offset_sh"] = offsets["offset_sh"] * gate
            h_new = h_old * (1.0 - gate) + h_new * gate
        return offsets, h_new

    def _compute_2d_features_scene_and_sky_gated(
        self,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        node_state_sky: NodeStateSky,
        source_frame_idx: int,
        rigid_idx_S: torch.Tensor,
        source_views: List[Any],
        source_images: List[torch.Tensor],
        height: int,
        width: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Source 2D: scene fused backprojection + sky gated fused backprojection.
        """
        if not hasattr(self, "_render_source_composite_for_cnn"):
            raise ValueError(
                "Stage4.4 sky-gated source 2D requires _render_source_composite_for_cnn implementation."
            )
        if not hasattr(self, "_backproject_scene_features_multi_camera"):
            raise ValueError(
                "Stage4.4 sky-gated source 2D requires _backproject_scene_features_multi_camera implementation."
            )
        if not hasattr(self, "_backproject_sky_features_gated_multi_camera"):
            raise ValueError(
                "Stage4.4 sky-gated source 2D requires _backproject_sky_features_gated_multi_camera implementation."
            )

        gaussians_bg_distant, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        num_rigid_S = int(rigid_idx_S.numel())

        scene_means = [gaussians_bg_distant["means"]]
        scene_scales = [gaussians_bg_distant["scales"]]
        scene_quats = [gaussians_bg_distant["quats"]]
        scene_opacities = [gaussians_bg_distant["opacities"]]
        scene_colors = [gaussians_bg_distant["colors"]]

        if node_state_rigid is not None and num_rigid_S > 0:
            rigid_point_ids_subset = node_state_rigid.point_ids[rigid_idx_S, 0]
            means_local_S = node_state_rigid.means[rigid_idx_S]
            quats_local_S = node_state_rigid.quats[rigid_idx_S]
            rigid_means_world = self._transform_rigid_to_world(
                node_state_rigid, means_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
            )
            scene_means.append(rigid_means_world)
            scene_quats.append(
                self._transform_rigid_quats_to_world(
                    node_state_rigid, quats_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
                )
            )
            scene_scales.append(torch.exp(node_state_rigid.scales_log[rigid_idx_S]))
            scene_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[rigid_idx_S]).squeeze(-1))
            scene_colors.append(torch.cat([node_state_rigid.sh_dc[rigid_idx_S, None, :], node_state_rigid.sh_rest[rigid_idx_S]], dim=1))

        g_sky = self._prepare_gaussians_sky(node_state_sky)
        num_sky = int(g_sky["means"].shape[0])

        gaussians_scene = {
            "means": torch.cat(scene_means, dim=0),
            "scales": torch.cat(scene_scales, dim=0),
            "quats": torch.cat(scene_quats, dim=0),
            "opacities": torch.cat(scene_opacities, dim=0),
            "colors": torch.cat(scene_colors, dim=0),
        }
        gaussians_sky = g_sky

        bp_unfiltered = FeatureBackprojector(
            eps=getattr(self.feature_backprojector, "eps", 1e-8),
            weight_threshold=0.0,
        )

        scene_ctx = self._render_source_composite_for_cnn(
            gaussians_scene=gaussians_scene,
            gaussians_sky=gaussians_sky,
            source_views=source_views,
            source_images=source_images,
            height=height,
            width=width,
        )

        feat_2d_scene, acc_w_scene = self._backproject_scene_features_multi_camera(
            gaussians_scene=gaussians_scene,
            source_views=source_views,
            features_2d=scene_ctx["features_2d"],
            height=height,
            width=width,
            backprojector_override=bp_unfiltered,
        )
        if feat_2d_scene is None or acc_w_scene is None:
            raise ValueError("Stage4.4 scene fused backprojection returned None unexpectedly.")

        feat_2d_sky, acc_w_sky = self._backproject_sky_features_gated_multi_camera(
            gaussians_sky=gaussians_sky,
            source_views=source_views,
            features_2d=scene_ctx["features_2d"],
            gate_image=scene_ctx["gate_image"],
            sky_viewmats=scene_ctx["sky_viewmats"],
            height=height,
            width=width,
            backprojector_override=bp_unfiltered,
        )
        if feat_2d_sky is None or acc_w_sky is None:
            raise ValueError("Stage4.4 sky gated fused backprojection returned None unexpectedly.")
        self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 2.0)

        idx0 = 0
        idx1 = idx0 + num_bg
        idx2 = idx1 + num_distant
        idx3 = idx2 + num_rigid_S
        if idx3 != int(feat_2d_scene.shape[0]):
            raise ValueError("Stage4.4 scene split size mismatch for fused backprojection.")

        feat_2d_bg = feat_2d_scene[idx0:idx1]
        acc_w_bg = acc_w_scene[idx0:idx1]
        feat_2d_distant = feat_2d_scene[idx1:idx2] if num_distant > 0 else None
        acc_w_distant = acc_w_scene[idx1:idx2] if num_distant > 0 else None
        feat_2d_rigid_S = feat_2d_scene[idx2:idx3] if num_rigid_S > 0 else None
        acc_w_rigid_S = acc_w_scene[idx2:idx3] if num_rigid_S > 0 else None

        return {
            "num_bg": num_bg,
            "num_distant": num_distant,
            "num_sky": num_sky,
            "feat_2d_bg": feat_2d_bg,
            "acc_w_bg": acc_w_bg,
            "feat_2d_distant": feat_2d_distant,
            "acc_w_distant": acc_w_distant,
            "feat_2d_rigid_S": feat_2d_rigid_S,
            "acc_w_rigid_S": acc_w_rigid_S,
            "feat_2d_sky": feat_2d_sky,
            "acc_w_sky": acc_w_sky,
            "src_backproject_pass_count": 2,
        }

    def _build_any_target_mask_static(
        self,
        num_points: int,
        enable_selective: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Stage4.3 bg/distant static branches: no per-point target visibility precomputation yet.
        Sky branch uses ``_build_sky_any_target_mask`` instead when selective update matters.
        """
        if num_points <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if not enable_selective:
            return torch.ones(num_points, dtype=torch.bool, device=device)
        return torch.ones(num_points, dtype=torch.bool, device=device)

    def _update_node_state_bg_subset(
        self,
        node_state_bg: NodeStateBackground,
        render_params: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            means_clamped = torch.clamp(
                render_params["means_r"][valid_idx].detach(),
                min=self.bbx_min,
                max=self.bbx_max,
            )
            node_state_bg.means[valid_idx] = means_clamped
            node_state_bg.scales_log[valid_idx] = render_params["scales_log_r"][valid_idx].detach()
            node_state_bg.quats[valid_idx] = render_params["quats_r"][valid_idx].detach()
            node_state_bg.opacity_logit[valid_idx] = render_params["opacity_logit_r"][valid_idx].detach()
            node_state_bg.sh_dc[valid_idx] = render_params["sh_dc_r"][valid_idx].detach()
            node_state_bg.sh_rest[valid_idx] = render_params["sh_rest_r"][valid_idx].detach()

    def _update_node_state_distant_subset(
        self,
        node_state_distant: NodeStateDistant,
        render_params: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            # Distant Gaussians are far-field / segment-exterior by design; do not clamp means to
            # dataset.segment_aabb (input_aabb_*). Clamping collapsed visible distant points onto the
            # AABB shell and destroyed frustum overlap after the first scheduler writeback.
            node_state_distant.means[valid_idx] = render_params["means_r"][valid_idx].detach()
            node_state_distant.scales_log[valid_idx] = render_params["scales_log_r"][valid_idx].detach()
            node_state_distant.quats[valid_idx] = render_params["quats_r"][valid_idx].detach()
            node_state_distant.opacity_logit[valid_idx] = render_params["opacity_logit_r"][valid_idx].detach()
            node_state_distant.sh_dc[valid_idx] = render_params["sh_dc_r"][valid_idx].detach()
            node_state_distant.sh_rest[valid_idx] = render_params["sh_rest_r"][valid_idx].detach()

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.3 requires non-empty batch['targets'].")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        node_state_sky = self._get_or_init_node_state_sky(batch)
        source_frame_idx = self._validate_stage4_1_batch(batch, targets, node_state_rigid)
        key = self._batch_key(batch)

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        sample_img = source_images[0]
        height, width = spatial_hw_from_image_tensor(sample_img)

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)

        N_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        mask_src_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_src_feat_valid_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_tgt_by_frame: Dict[int, torch.Tensor] = {}
        unique_target_frames = sorted({int(t["frame_idx"]) for t in targets})
        if node_state_rigid is not None:
            mask_src_rigid = self._rigid_point_valid_mask(node_state_rigid, source_frame_idx)
            for frame_idx in unique_target_frames:
                mask_tgt_by_frame[frame_idx] = self._rigid_point_valid_mask(node_state_rigid, frame_idx)
        else:
            for frame_idx in unique_target_frames:
                mask_tgt_by_frame[frame_idx] = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_any_tgt_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        for m in mask_tgt_by_frame.values():
            mask_any_tgt_rigid = mask_any_tgt_rigid | m

        S = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
        one_pass = self._compute_2d_features_scene_and_sky_gated(
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            node_state_sky=node_state_sky,
            source_frame_idx=source_frame_idx,
            rigid_idx_S=S,
            source_views=source_views,
            source_images=source_images,
            height=height,
            width=width,
        )
        num_bg = int(one_pass["num_bg"])
        num_distant = int(one_pass["num_distant"])
        num_sky = int(one_pass["num_sky"])
        feat_2d_bg = one_pass["feat_2d_bg"]
        feat_2d_distant = one_pass["feat_2d_distant"]
        feat_2d_rigid_S = one_pass["feat_2d_rigid_S"]
        feat_2d_sky = one_pass["feat_2d_sky"]
        acc_w_bg = one_pass["acc_w_bg"]
        acc_w_distant = one_pass["acc_w_distant"]
        acc_w_rigid_S = one_pass["acc_w_rigid_S"]
        acc_w_sky = one_pass["acc_w_sky"]
        src_backproject_pass_count = int(one_pass.get("src_backproject_pass_count", 0))

        mask_src_feat_valid_bg = acc_w_bg > self.bg_src_backproject_support_min
        mask_any_tgt_bg = self._build_any_target_mask_static(
            num_points=num_bg,
            enable_selective=self.bg_enable_selective_update,
            device=self.device,
        )
        mask_update_bg = mask_src_feat_valid_bg & mask_any_tgt_bg
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)

        mask_src_feat_valid_distant = (
            (acc_w_distant > self.distant_src_backproject_support_min) if acc_w_distant is not None else None
        )
        if num_distant > 0:
            mask_any_tgt_distant = self._build_any_target_mask_static(
                num_points=num_distant,
                enable_selective=self.distant_enable_selective_update,
                device=self.device,
            )
            mask_update_distant = mask_src_feat_valid_distant & mask_any_tgt_distant
        else:
            mask_any_tgt_distant = None
            mask_update_distant = None

        mask_src_feat_valid_sky = (
            (acc_w_sky > self.sky_src_backproject_support_min) if acc_w_sky is not None else None
        )
        if num_sky > 0:
            mask_any_tgt_sky = self._build_any_target_mask_static(
                num_points=num_sky,
                enable_selective=self.sky_enable_selective_update,
                device=self.device,
            )
            mask_update_sky = mask_src_feat_valid_sky & mask_any_tgt_sky
        else:
            mask_any_tgt_sky = None
            mask_update_sky = None

        if node_state_rigid is not None and S.numel() > 0:
            if acc_w_rigid_S is None:
                raise ValueError("Stage4.2 rigid S non-empty but acc_w_rigid_S is None.")
            mask_src_feat_valid_rigid[S] = acc_w_rigid_S > self.src_backproject_support_min
            bad = mask_src_feat_valid_rigid & ~mask_src_rigid
            if bool(bad.any().item()):
                raise ValueError("mask_src_feat_valid_rigid True outside mask_src_rigid.")
        mask_update_rigid = mask_src_feat_valid_rigid & mask_any_tgt_rigid
        U = torch.nonzero(mask_update_rigid, as_tuple=False).squeeze(1)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru(
            feat_bg_input, params_bg, h_old_bg, mask_update_rigid=mask_update_bg
        )
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        render_params_rigid_local: Optional[Dict[str, torch.Tensor]] = None
        h_new_rigid: Optional[torch.Tensor] = None
        offsets_rigid: Optional[Dict[str, torch.Tensor]] = None
        if node_state_rigid is not None and U.numel() > 0 and feat_2d_rigid_S is not None and S.numel() > 0:
            lookup_s = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_s[S] = torch.arange(S.numel(), device=self.device, dtype=torch.long)
            idx_in_S = lookup_s[U]
            feat_U = feat_2d_rigid_S[idx_in_S]
            if int(feat_U.shape[-1]) != int(self.rigid_feat_in_dim):
                raise ValueError(f"Rigid 2D feature dim mismatch: got {feat_U.shape[-1]}, expected {self.rigid_feat_in_dim}")
            feat_U = self.rigid_feat_proj(feat_U)

            class _RigidEmbedState:
                pass

            rigid_embed_state = _RigidEmbedState()
            rigid_embed_state.means = self._transform_rigid_to_world(
                node_state_rigid, node_state_rigid.means[U], source_frame_idx, point_ids_subset=node_state_rigid.point_ids[U, 0]
            )
            rigid_embed_state.quats = self._transform_rigid_quats_to_world(
                node_state_rigid, node_state_rigid.quats[U], source_frame_idx, point_ids_subset=node_state_rigid.point_ids[U, 0]
            )
            rigid_embed_state.scales_log = node_state_rigid.scales_log[U]
            rigid_embed_state.opacity_logit = node_state_rigid.opacity_logit[U]
            rigid_embed_state.sh_dc = node_state_rigid.sh_dc[U]
            rigid_embed_state.sh_rest = node_state_rigid.sh_rest[U]
            params_rigid = self._build_params_for_embed(rigid_embed_state, coord_space="world")
            h_old_rigid = self._get_or_init_hidden(self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid")
            h_old_rigid_U = h_old_rigid[U]
            rigid_head_rms_mask = mask_src_feat_valid_rigid[U].to(dtype=feat_U.dtype, device=feat_U.device)
            offsets_rigid, h_new_rigid_U = self._predict_offsets_gru_rigid(
                feat_U, params_rigid, h_old_rigid_U, head_rms_mask=rigid_head_rms_mask
            )
            render_params_rigid_local = self._render_params_from_offsets_rigid_local(
                NodeStateRigid(
                    means=node_state_rigid.means[U],
                    scales_log=node_state_rigid.scales_log[U],
                    quats=node_state_rigid.quats[U],
                    opacity_logit=node_state_rigid.opacity_logit[U],
                    sh_dc=node_state_rigid.sh_dc[U],
                    sh_rest=node_state_rigid.sh_rest[U],
                    point_ids=node_state_rigid.point_ids[U],
                    instances_quats=node_state_rigid.instances_quats,
                    instances_trans=node_state_rigid.instances_trans,
                    instances_fv=node_state_rigid.instances_fv,
                    instance_ids=node_state_rigid.instance_ids,
                    frame_ids=node_state_rigid.frame_ids,
                    cur_frame=node_state_rigid.cur_frame,
                ),
                offsets_rigid,
            )
            h_new_rigid = h_old_rigid.clone()
            h_new_rigid[U] = h_new_rigid_U
        if node_state_rigid is not None and h_new_rigid is None:
            h_new_rigid = self._get_or_init_hidden(self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid").clone()

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_2d_distant is not None and feat_2d_distant.numel() > 0:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant")
            offsets_distant, h_new_distant = self._predict_offsets_gru_distant_masked(
                feat_distant_input, params_distant, h_old_distant, mask_update_distant
            )
            render_params_distant = self._render_params_from_offsets_distant(node_state_distant, offsets_distant)

        offsets_sky: Optional[Dict[str, torch.Tensor]] = None
        render_params_sky: Optional[Dict[str, torch.Tensor]] = None
        h_new_sky: Optional[torch.Tensor] = None
        if feat_2d_sky is not None and feat_2d_sky.numel() > 0:
            feat_sky_input = self.sky_feat_proj(feat_2d_sky)
            params_sky = self._build_params_for_embed(node_state_sky, coord_space="world")
            h_old_sky = self._get_or_init_hidden(self.h_cache_sky, key, node_state_sky.means.shape[0], node_state_sky, "sky")
            offsets_sky, h_new_sky = self._predict_offsets_gru_sky_masked(
                feat_sky_input, params_sky, h_old_sky, mask_update_sky
            )
            render_params_sky = self._render_params_from_offsets_sky(node_state_sky, offsets_sky)

        by_frame: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
        for i, t in enumerate(targets):
            by_frame[int(t["frame_idx"])].append((i, t))
        sorted_frames = sorted(by_frame.keys())

        def _run_frame_renders(
            training: bool,
            proxies_bg_l: Dict[str, torch.Tensor],
            proxies_dist_l: Optional[Dict[str, torch.Tensor]],
            rigid_local_opt: Optional[Dict[str, torch.Tensor]],
            U_tensor: torch.Tensor,
        ):
            pred_by_idx: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            rigid_pairs_l: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
            for F in sorted_frames:
                group = by_frame[F]
                targets_F = [t for _, t in group]
                idx_tr = torch.nonzero(mask_update_rigid & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                idx_fr = torch.nonzero((~mask_update_rigid) & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                rw: Optional[Dict[str, torch.Tensor]] = None
                if node_state_rigid is not None and (idx_tr.numel() > 0 or idx_fr.numel() > 0):
                    rw = self._build_rigid_world_for_frame(node_state_rigid, F, idx_tr, idx_fr, rigid_local_opt, U_tensor)
                prox_r: Optional[Dict[str, torch.Tensor]] = None
                if rw is not None and training:
                    if idx_tr.numel() > 0:
                        prox_r = _create_proxy_params(rw)
                        rigid_pairs_l.append((rw, prox_r))
                    else:
                        prox_r = {
                            "means_p": rw["means_r"],
                            "scales_p": rw["scales_r"],
                            "quats_p": rw["quats_r"],
                            "opacities_p": rw["opacities_r"],
                            "colors_p": rw["colors_r"],
                        }
                if training:
                    merged_f = _merge_params_bg_rigid_distant(proxies_bg_l, prox_r, proxies_dist_l)
                else:
                    merged_f = self._tensor_merge_bg_rigid_distant_world(render_params_bg, rw, render_params_distant)

                heights = []
                widths = []
                for t in targets_F:
                    g = t["gt_image"]
                    if g.dim() == 4:
                        g = g.squeeze(0)
                    heights.append(int(g.shape[0]))
                    widths.append(int(g.shape[1]))
                h0, w0 = heights[0], widths[0]
                if all(h == h0 and w == w0 for h, w in zip(heights, widths)):
                    multi_result = self._render_multi_view(merged_f, targets_F)
                    if multi_result is not None:
                        for j, (orig_i, _) in enumerate(group):
                            rgb_j, acc_j = multi_result[j]
                            pred_by_idx[orig_i] = (rgb_j, acc_j.squeeze(-1) if acc_j.dim() == 3 else acc_j)
                        continue
                for orig_i, t in group:
                    view = t["view"]
                    g = t["gt_image"]
                    if g.dim() == 4:
                        g = g.squeeze(0)
                    hh, ww = int(g.shape[0]), int(g.shape[1])
                    pred_rgb, acc = self._render_single_view(merged_f, view, hh, ww)
                    pred_by_idx[orig_i] = (pred_rgb, acc.squeeze(-1) if acc.dim() == 3 else acc)
            return pred_by_idx, rigid_pairs_l

        if not self.training:
            pred_by_idx, _ = _run_frame_renders(False, {}, None, render_params_rigid_local, U)
            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []
            for i in range(len(targets)):
                pr, acc = pred_by_idx[i]
                gt = targets[i]["gt_image"]
                if gt.dim() == 4:
                    gt = gt.squeeze(0)
                hh, ww = int(gt.shape[0]), int(gt.shape[1])
                view = targets[i]["view"]
                if render_params_sky is not None:
                    rgb_sky = self._render_sky_single_view(render_params_sky, view, hh, ww)
                    pred_rgbs.append(_composite_sky_gs(pr, acc, rgb_sky))
                else:
                    pred_rgbs.append(pr)
                gt_images.append(gt)
            return {
                "loss": torch.tensor(0.0, device=self.device),
                "render_params": render_params_bg,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_render_params_sky": render_params_sky,
                "_render_params_rigid_world": None,
                "_render_params_rigid_local": render_params_rigid_local,
                "_node_state_bg": node_state_bg,
                "_node_state_distant": node_state_distant,
                "_node_state_rigid": node_state_rigid,
                "_node_state_sky": node_state_sky,
                "_h_new_bg": h_new_bg,
                "_h_new_distant": h_new_distant,
                "_h_new_rigid": h_new_rigid,
                "_h_new_sky": h_new_sky,
                "_bg_writeback_idx": torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1),
                "_distant_writeback_idx": (
                    torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
                ),
                "_sky_writeback_idx": torch.nonzero(mask_update_sky, as_tuple=False).squeeze(1),
                "_rigid_writeback_idx": U,
                "_rigid_valid_idx": S,
                "_num_rigid_valid_src": int(S.numel()),
                "_num_rigid_total": N_rigid,
                "_cache_key": key,
                "_src_backproject_pass_count": src_backproject_pass_count,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        proxies_sky = _create_proxy_params(render_params_sky) if render_params_sky is not None else None
        pred_by_idx, rigid_world_proxy_pairs = _run_frame_renders(True, proxies_bg, proxies_distant, render_params_rigid_local, U)

        pred_rgbs_t: List[torch.Tensor] = []
        gt_images_t: List[torch.Tensor] = []
        opacities_t: List[torch.Tensor] = []
        for i in range(len(targets)):
            pr, acc = pred_by_idx[i]
            gt = targets[i]["gt_image"]
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            hh, ww = int(gt.shape[0]), int(gt.shape[1])
            view = targets[i]["view"]
            if proxies_sky is not None:
                sky_render_params = {
                    "means_r": proxies_sky["means_p"],
                    "scales_r": proxies_sky["scales_p"],
                    "quats_r": proxies_sky["quats_p"],
                    "opacities_r": proxies_sky["opacities_p"],
                    "colors_r": proxies_sky["colors_p"],
                }
                rgb_sky = self._render_sky_single_view(sky_render_params, view, hh, ww)
                pred_rgbs_t.append(_composite_sky_gs(pr, acc, rgb_sky))
            else:
                pred_rgbs_t.append(pr)
            gt_images_t.append(gt)
            opacities_t.append(acc)

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        frame_losses: List[torch.Tensor] = []
        frame_loss_map: Dict[int, float] = {}
        eff_frames = 0
        for F in sorted_frames:
            group = by_frame[F]
            view_losses: List[torch.Tensor] = []
            for orig_i, t in group:
                pred_rgb = pred_rgbs_t[orig_i]
                gt_image = gt_images_t[orig_i]
                opacity = opacities_t[orig_i].to(self.device).float()
                if opacity.dim() == 3 and opacity.shape[-1] == 1:
                    opacity = opacity.squeeze(-1)
                h, w = gt_image.shape[0], gt_image.shape[1]
                valid_loss_mask = self._valid_loss_mask_from_target(t, height=h, width=w)
                if float(valid_loss_mask.sum().item()) <= 0:
                    continue
                l1_i = self.loss_w_l1 * torch.mean(torch.abs((pred_rgb - gt_image) * valid_loss_mask.unsqueeze(-1)))
                ssim_i = self.loss_w_ssim * compute_ssim_loss_masked(
                    pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None, data_range=1.0
                )
                sm = t["sky_mask"].to(self.device).float()
                if sm.dim() == 3:
                    sm = sm.squeeze(-1)
                gt_occupied = (1.0 - sm) * valid_loss_mask
                pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
                mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)
                p = opacity.clamp(1e-6, 1.0 - 1e-6)
                entropy_i = self.loss_w_opacity_entropy * self._masked_mean(-p * torch.log(p), valid_loss_mask)
                total_i = l1_i + ssim_i + mask_i + entropy_i
                loss_l1_list.append(l1_i)
                loss_ssim_list.append(ssim_i)
                loss_mask_list.append(mask_i)
                loss_entropy_list.append(entropy_i)
                view_losses.append(total_i)
            if view_losses:
                frame_loss = torch.stack(view_losses).mean()
                frame_losses.append(frame_loss)
                frame_loss_map[int(F)] = float(frame_loss.detach().item())
                eff_frames += 1
        if frame_losses:
            loss = torch.stack(frame_losses).mean()
        else:
            loss = render_params_bg["means_r"].sum() * 0.0
            logger.warning("Stage4.2: no valid supervision in this step; using zero loss.")

        l1_mean = torch.stack(loss_l1_list).mean() if loss_l1_list else loss * 0.0
        ssim_mean = torch.stack(loss_ssim_list).mean() if loss_ssim_list else loss * 0.0
        mask_mean = torch.stack(loss_mask_list).mean() if loss_mask_list else loss * 0.0
        entropy_mean = torch.stack(loss_entropy_list).mean() if loss_entropy_list else loss * 0.0
        offset_stats = self._collect_offset_stats(offsets_bg, offsets_rigid, offsets_sky)
        hidden_stats = self._collect_hidden_norms(h_new_bg, h_new_distant, h_new_rigid)
        if h_new_sky is not None:
            hidden_stats["hidden_norm_sky_mean"] = float(torch.norm(h_new_sky.detach().float(), dim=-1).mean().item())
        else:
            hidden_stats["hidden_norm_sky_mean"] = 0.0

        bg_writeback_idx = torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1)
        distant_writeback_idx = (
            torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
        )
        sky_writeback_idx = torch.nonzero(mask_update_sky, as_tuple=False).squeeze(1)
        rigid_src_feat_valid = int(mask_src_feat_valid_rigid.sum().item())
        rigid_update_count = int(U.numel())
        rigid_update_ratio = float(rigid_update_count / max(int(N_rigid), 1))
        rigid_update_among_feat_valid = float(rigid_update_count / max(rigid_src_feat_valid, 1))

        return {
            "loss": loss,
            "loss_l1": l1_mean,
            "loss_ssim": ssim_mean,
            "loss_mask": mask_mean,
            "loss_opacity_entropy": entropy_mean,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_proxies_distant": proxies_distant,
            "_proxies_rigid_world": None,
            "_rigid_world_proxy_pairs": rigid_world_proxy_pairs if rigid_world_proxy_pairs else None,
            "_render_params_distant": render_params_distant,
            "_render_params_sky": render_params_sky,
            "_proxies_sky": proxies_sky,
            "_render_params_rigid_world": None,
            "_render_params_rigid_local": render_params_rigid_local,
            "_node_state_bg": node_state_bg,
            "_node_state_distant": node_state_distant,
            "_node_state_rigid": node_state_rigid,
            "_node_state_sky": node_state_sky,
            "_h_new_bg": h_new_bg,
            "_h_new_distant": h_new_distant,
            "_h_new_rigid": h_new_rigid,
            "_h_new_sky": h_new_sky,
            "_bg_writeback_idx": bg_writeback_idx,
            "_distant_writeback_idx": distant_writeback_idx,
            "_sky_writeback_idx": sky_writeback_idx,
            "_rigid_valid_idx": S,
            "_rigid_writeback_idx": U,
            "_num_rigid_valid_src": int(S.numel()),
            "_num_rigid_src_feat_valid": int(mask_src_feat_valid_rigid.sum().item()),
            "_num_rigid_update": int(U.numel()),
            "_num_target_frames": len(sorted_frames),
            "_loss_effective_frames": eff_frames,
            "_num_rigid_total": N_rigid,
            "_frame_loss_map": frame_loss_map,
            "_offset_stats": offset_stats,
            "_hidden_stats": hidden_stats,
            "_rigid_update_ratio": rigid_update_ratio,
            "_rigid_update_among_feat_valid": rigid_update_among_feat_valid,
            "_num_bg_src_feat_valid": int(mask_src_feat_valid_bg.sum().item()),
            "_num_bg_update": int(bg_writeback_idx.numel()),
            "_num_distant_src_feat_valid": int(mask_src_feat_valid_distant.sum().item()) if mask_src_feat_valid_distant is not None else 0,
            "_num_distant_update": int(distant_writeback_idx.numel()) if distant_writeback_idx is not None else 0,
            "_num_sky_src_feat_valid": int(mask_src_feat_valid_sky.sum().item()),
            "_num_sky_update": int(sky_writeback_idx.numel()),
            "_src_backproject_pass_count": src_backproject_pass_count,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs_t,
            "gt_images": gt_images_t,
            "pred_rgb": pred_rgbs_t[0],
            "gt_image": gt_images_t[0],
        }

    def _writeback_node_states_from_out(self, out: Dict[str, Any]) -> None:
        if "_node_state_bg" in out:
            bg_idx = out.get("_bg_writeback_idx")
            if bg_idx is None:
                self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
            else:
                self._update_node_state_bg_subset(out["_node_state_bg"], out["render_params"], bg_idx)
        if out.get("_node_state_distant") is not None and out.get("_render_params_distant") is not None:
            distant_idx = out.get("_distant_writeback_idx")
            if distant_idx is None:
                self._update_node_state_distant(out["_node_state_distant"], out["_render_params_distant"])
            else:
                self._update_node_state_distant_subset(out["_node_state_distant"], out["_render_params_distant"], distant_idx)
        if out.get("_node_state_sky") is not None and out.get("_render_params_sky") is not None:
            sky_idx = out.get("_sky_writeback_idx")
            if sky_idx is None:
                self._update_node_state_sky(out["_node_state_sky"], out["_render_params_sky"])
            else:
                self._update_node_state_sky_subset(out["_node_state_sky"], out["_render_params_sky"], sky_idx)
        if out.get("_node_state_rigid") is not None and out.get("_render_params_rigid_local") is not None:
            valid_idx = out.get("_rigid_writeback_idx", out.get("_rigid_valid_idx"))
            if valid_idx is None:
                raise ValueError("Internal error: missing rigid writeback idx.")
            if valid_idx.numel() > 0:
                self._update_node_state_rigid_local(out["_node_state_rigid"], out["_render_params_rigid_local"], valid_idx)

    def _default_runtime_policy(self) -> RuntimePolicy:
        return RuntimePolicy(
            do_backward=True,
            do_optimizer_step=True,
            update_hidden_cache=True,
            writeback_node_state=True,
            reset_node_state_after_block=True,
        )

    def _resolve_export_key_and_ref_frame(
        self,
        batch_or_key: Dict[str, Any] | Tuple[int, int],
        rigid_export_frame_idx: Optional[int],
    ) -> Tuple[Tuple[int, int], int]:
        if isinstance(batch_or_key, tuple):
            key = (int(batch_or_key[0]), int(batch_or_key[1]))
            if rigid_export_frame_idx is None:
                raise ValueError(
                    "export_3dgs_state(batch_or_key=tuple) requires rigid_export_frame_idx "
                    "to export rigid branch in world/seg0 coordinates."
                )
            return key, int(rigid_export_frame_idx)
        if not isinstance(batch_or_key, dict):
            raise ValueError("batch_or_key must be a batch dict or cache key tuple(scene_id, segment_id)")
        key = self._batch_key(batch_or_key)
        if rigid_export_frame_idx is not None:
            return key, int(rigid_export_frame_idx)
        src_views = batch_or_key.get("source_views") or []
        if src_views:
            return key, int(src_views[0]["frame_idx"])
        targets = batch_or_key.get("targets") or []
        if targets:
            return key, int(targets[0]["frame_idx"])
        raise ValueError(
            "Cannot infer rigid_export_frame_idx from batch; pass rigid_export_frame_idx explicitly."
        )

    def _as_cpu_tensor(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.detach().cpu()

    def export_3dgs_state(
        self,
        batch_or_key: Dict[str, Any] | Tuple[int, int],
        *,
        include_hidden: bool = False,
        rigid_export_frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Export branch states. Rigid branch is exported in world/seg0 semantics under rigid_world.
        """
        key, rigid_ref_frame = self._resolve_export_key_and_ref_frame(batch_or_key, rigid_export_frame_idx)
        node_bg = self.node_states_bg.get(key)
        if node_bg is None:
            raise ValueError(f"No bg node state for cache key {key}")
        node_distant = self.node_states_distant.get(key)
        node_rigid = self.node_states_rigid.get(key)
        node_sky = self.node_states_sky.get(key)

        def _pack_branch(state: Any) -> Optional[Dict[str, torch.Tensor]]:
            if state is None:
                return None
            return {
                "means": self._as_cpu_tensor(state.means),
                "scales_log": self._as_cpu_tensor(state.scales_log),
                "quats": self._as_cpu_tensor(state.quats),
                "opacity_logit": self._as_cpu_tensor(state.opacity_logit),
                "sh_dc": self._as_cpu_tensor(state.sh_dc),
                "sh_rest": self._as_cpu_tensor(state.sh_rest),
            }

        rigid_world: Optional[Dict[str, torch.Tensor]] = None
        rigid_local = _pack_branch(node_rigid)
        if node_rigid is not None:
            point_ids = node_rigid.point_ids[:, 0] if node_rigid.point_ids.dim() > 1 else node_rigid.point_ids
            means_w = self._transform_rigid_to_world(node_rigid, node_rigid.means, rigid_ref_frame, point_ids_subset=point_ids)
            quats_w = self._transform_rigid_quats_to_world(
                node_rigid, node_rigid.quats, rigid_ref_frame, point_ids_subset=point_ids
            )
            rigid_world = {
                "means": self._as_cpu_tensor(means_w),
                "scales_log": self._as_cpu_tensor(node_rigid.scales_log),
                "quats": self._as_cpu_tensor(quats_w),
                "opacity_logit": self._as_cpu_tensor(node_rigid.opacity_logit),
                "sh_dc": self._as_cpu_tensor(node_rigid.sh_dc),
                "sh_rest": self._as_cpu_tensor(node_rigid.sh_rest),
            }

        state: Dict[str, Any] = {
            "cache_key": {"scene_id": int(key[0]), "segment_id": int(key[1])},
            "coordinate_frame": "world/seg0",
            "rigid_export_frame_idx": int(rigid_ref_frame),
            "sky_metadata": {
                "sky_origin_mode": self.sky_origin_mode,
                "sky_center_local": self._as_cpu_tensor(self.sky_center_local),
                "sky_radius": float(self.sky_radius),
                "sky_resolution": int(self.sky_resolution),
                "sky_hemisphere": bool(self.sky_hemisphere),
                "sky_hemisphere_up": list(self.sky_hemisphere_up),
            },
            "branches": {
                "bg": _pack_branch(node_bg),
                "distant": _pack_branch(node_distant),
                "sky": _pack_branch(node_sky),
                "rigid_local": rigid_local,
                "rigid_world": rigid_world,
            },
        }

        if isinstance(batch_or_key, dict):
            req_meta = batch_or_key.get("request_meta") or {}
            src_refs = req_meta.get("source_image_refs")
            test_refs = req_meta.get("test_image_refs")
            if src_refs is not None:
                state["source_image_refs"] = list(src_refs)
            if test_refs is not None:
                state["test_image_refs"] = list(test_refs)
            if batch_or_key.get("aabb") is not None:
                state["segment_aabb"] = self._as_cpu_tensor(batch_or_key["aabb"])
            if batch_or_key.get("segment_first_frame_idx") is not None:
                state["segment_first_frame_idx"] = int(batch_or_key["segment_first_frame_idx"])

        if include_hidden:
            state["hidden"] = {
                "bg": self._as_cpu_tensor(self.h_cache_bg.get(key)),
                "distant": self._as_cpu_tensor(self.h_cache_distant.get(key)),
                "rigid": self._as_cpu_tensor(self.h_cache_rigid.get(key)),
                "sky": self._as_cpu_tensor(self.h_cache_sky.get(key)),
            }
        return state

    def ensure_runtime_state_from_batch(self, batch: Dict[str, Any]) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant], NodeStateSky]:
        targets = batch.get("targets") or []
        if len(targets) == 0:
            raise ValueError("ensure_runtime_state_from_batch requires non-empty batch['targets']")
        node_bg, node_rigid, node_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        node_sky = self._get_or_init_node_state_sky(batch)
        return node_bg, node_rigid, node_distant, node_sky

    def _snapshot_runtime_state(self, key: Tuple[int, int]) -> Dict[str, Any]:
        def _clone_state(s: Any) -> Any:
            if s is None:
                return None
            out: Dict[str, Any] = {}
            for k, v in vars(s).items():
                if torch.is_tensor(v):
                    out[k] = v.detach().clone()
                else:
                    out[k] = v
            return out

        return {
            "bg": _clone_state(self.node_states_bg.get(key)),
            "distant": _clone_state(self.node_states_distant.get(key)),
            "rigid": _clone_state(self.node_states_rigid.get(key)),
            "sky": _clone_state(self.node_states_sky.get(key)),
            "h_bg": self.h_cache_bg.get(key).detach().clone() if key in self.h_cache_bg else None,
            "h_distant": self.h_cache_distant.get(key).detach().clone() if key in self.h_cache_distant else None,
            "h_rigid": self.h_cache_rigid.get(key).detach().clone() if key in self.h_cache_rigid else None,
            "h_sky": self.h_cache_sky.get(key).detach().clone() if key in self.h_cache_sky else None,
        }

    def _restore_runtime_state(self, key: Tuple[int, int], snap: Dict[str, Any]) -> None:
        def _restore(dst: Any, src: Dict[str, Any]) -> None:
            for k, v in src.items():
                if torch.is_tensor(v):
                    setattr(dst, k, v.to(self.device))
                else:
                    setattr(dst, k, v)

        if snap.get("bg") is not None and key in self.node_states_bg:
            _restore(self.node_states_bg[key], snap["bg"])
        if snap.get("distant") is not None and key in self.node_states_distant:
            _restore(self.node_states_distant[key], snap["distant"])
        if snap.get("rigid") is not None and key in self.node_states_rigid:
            _restore(self.node_states_rigid[key], snap["rigid"])
        if snap.get("sky") is not None and key in self.node_states_sky:
            _restore(self.node_states_sky[key], snap["sky"])

        for cache, name in (
            (self.h_cache_bg, "h_bg"),
            (self.h_cache_distant, "h_distant"),
            (self.h_cache_rigid, "h_rigid"),
            (self.h_cache_sky, "h_sky"),
        ):
            v = snap.get(name)
            if v is None:
                cache.pop(key, None)
            else:
                cache[key] = v.to(self.device)

    def import_3dgs_state(self, state: Dict[str, Any], *, batch_context: Optional[Dict[str, Any]] = None) -> None:
        key_block = state.get("cache_key")
        if not isinstance(key_block, dict):
            raise ValueError("state.cache_key is required")
        key = (int(key_block["scene_id"]), int(key_block["segment_id"]))
        branches = state.get("branches")
        if not isinstance(branches, dict):
            raise ValueError("state.branches is required")
        sky_metadata = state.get("sky_metadata")
        if not isinstance(sky_metadata, dict):
            raise ValueError("state.sky_metadata is required")
        sky_origin_mode = sky_metadata.get("sky_origin_mode")
        if sky_origin_mode != "camera_centered_rotation_only":
            raise ValueError(
                "state.sky_metadata.sky_origin_mode must be 'camera_centered_rotation_only'."
            )
        if sky_origin_mode != self.sky_origin_mode:
            raise ValueError("Imported sky_origin_mode does not match model.sky.origin_mode.")
        center_local = sky_metadata.get("sky_center_local")
        if center_local is None:
            raise ValueError("state.sky_metadata.sky_center_local is required")
        center_local_t = torch.as_tensor(center_local, dtype=torch.float32, device=self.device).reshape(-1)
        if center_local_t.numel() != 3:
            raise ValueError("state.sky_metadata.sky_center_local must have 3 elements")
        if not torch.allclose(center_local_t, self.sky_center_local.to(self.device), atol=1e-6, rtol=0.0):
            raise ValueError("Imported sky_center_local does not match current model.sky.center_local.")
        if int(sky_metadata.get("sky_resolution", -1)) != int(self.sky_resolution):
            raise ValueError("Imported sky_resolution does not match current model.sky.resolution.")
        if abs(float(sky_metadata.get("sky_radius", -1.0)) - float(self.sky_radius)) > 1e-6:
            raise ValueError("Imported sky_radius does not match current model.sky.radius.")
        if batch_context is not None:
            self.ensure_runtime_state_from_batch(batch_context)

        def _apply(dst: Any, src: Dict[str, Any]) -> Any:
            dst.means = src["means"].to(self.device)
            dst.scales_log = src["scales_log"].to(self.device)
            dst.quats = src["quats"].to(self.device)
            dst.opacity_logit = src["opacity_logit"].to(self.device)
            dst.sh_dc = src["sh_dc"].to(self.device)
            dst.sh_rest = src["sh_rest"].to(self.device)
            return dst

        if branches.get("bg") is not None:
            if key not in self.node_states_bg:
                raise ValueError(f"Cannot import bg: key {key} does not exist in model caches")
            self.node_states_bg[key] = _apply(self.node_states_bg[key], branches["bg"])
        if branches.get("distant") is not None:
            if key not in self.node_states_distant:
                raise ValueError(f"Cannot import distant: key {key} does not exist in model caches")
            self.node_states_distant[key] = _apply(self.node_states_distant[key], branches["distant"])
        if branches.get("sky") is not None:
            if key not in self.node_states_sky:
                raise ValueError(f"Cannot import sky: key {key} does not exist in model caches")
            self.node_states_sky[key] = _apply(self.node_states_sky[key], branches["sky"])
        if branches.get("rigid_local") is not None:
            if key not in self.node_states_rigid:
                raise ValueError(f"Cannot import rigid_local: key {key} does not exist in model caches")
            self.node_states_rigid[key] = _apply(self.node_states_rigid[key], branches["rigid_local"])

        hidden = state.get("hidden")
        if isinstance(hidden, dict):
            for cache, name in (
                (self.h_cache_bg, "bg"),
                (self.h_cache_distant, "distant"),
                (self.h_cache_rigid, "rigid"),
                (self.h_cache_sky, "sky"),
            ):
                h = hidden.get(name)
                if h is not None:
                    cache[key] = h.to(self.device)

    def build_scene_representation_from_source(
        self,
        batch: Dict,
        *,
        allow_hidden_cache_update: bool,
        allow_node_state_writeback: bool,
    ) -> Dict[str, Any]:
        source_views = batch.get("source_views") or []
        source_images = batch.get("source_images") or []
        if len(source_views) == 0 or len(source_images) == 0:
            raise ValueError("batch must contain non-empty source_views/source_images")
        first_view = source_views[0]
        first_image = source_images[0]
        source_sky_masks = batch.get("source_sky_mask") or []
        source_egocar_masks = batch.get("source_egocar_mask") or []
        source_viewdirs = batch.get("source_viewdirs") or []
        source_frame_idx = int(batch.get("source_frame_idx", 0))
        # convert_batch_to_minimal_format builds lightweight View objects (attribute-based),
        # while some paths may pass dict-like views; support both without branching callers.
        if isinstance(first_view, dict):
            source_cam_idx = int(first_view.get("cam_idx", -1))
        else:
            source_cam_idx = int(getattr(first_view, "cam_idx", -1))
        src_target = {
            "view": first_view,
            "gt_image": first_image,
            "frame_idx": source_frame_idx,
            "cam_idx": source_cam_idx,
            "sky_mask": source_sky_masks[0] if len(source_sky_masks) > 0 else None,
            "egocar_mask": source_egocar_masks[0] if len(source_egocar_masks) > 0 else None,
            "viewdirs": source_viewdirs[0] if len(source_viewdirs) > 0 else None,
        }
        infer_batch = dict(batch)
        infer_batch["targets"] = [src_target]
        # Ensure runtime states exist for this cache key before building representation.
        self.ensure_runtime_state_from_batch(infer_batch)
        key = self._batch_key(infer_batch)
        snap = self._snapshot_runtime_state(key)
        prev_mode = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(infer_batch)
        if prev_mode:
            self.train()

        out_key = out.get("_cache_key")
        if out_key is not None and tuple(out_key) != tuple(key):
            key = (int(out_key[0]), int(out_key[1]))
        if allow_hidden_cache_update:
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()
            if out.get("_h_new_sky") is not None:
                self.h_cache_sky[key] = out["_h_new_sky"].detach()
        # Always materialize one built state for export, but restore runtime when writeback is disallowed.
        self._writeback_node_states_from_out(out)
        gs_state = self.export_3dgs_state(
            infer_batch,
            include_hidden=allow_hidden_cache_update,
            rigid_export_frame_idx=int(src_target["frame_idx"]),
        )
        if not allow_node_state_writeback:
            self._restore_runtime_state(key, snap)
        return {
            "cache_key": key,
            "base_batch": infer_batch,
            "gs_state": gs_state,
        }

    def export_viewer_snapshot(
        self,
        batch: Dict[str, Any],
        *,
        scheduler_meta: Optional[Dict[str, Any]] = None,
        segment_aabb: Optional[torch.Tensor] = None,
        include_hidden: bool = False,
        allow_hidden_cache_update: bool = False,
        allow_node_state_writeback: bool = False,
        rigid_export_frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a viewer-friendly immutable snapshot from one train batch."""
        if not isinstance(batch, dict):
            raise ValueError("export_viewer_snapshot requires batch dict input")

        scene_repr = self.build_scene_representation_from_source(
            batch,
            allow_hidden_cache_update=allow_hidden_cache_update,
            allow_node_state_writeback=allow_node_state_writeback,
        )
        gs_state = scene_repr["gs_state"]
        if include_hidden and not allow_hidden_cache_update:
            base_batch = scene_repr["base_batch"]
            gs_state = self.export_3dgs_state(
                base_batch,
                include_hidden=True,
                rigid_export_frame_idx=rigid_export_frame_idx,
            )

        aligned = batch.get("_scheduler_v4_aligned_info") or {}
        sched = dict(aligned)
        if scheduler_meta is not None:
            sched.update(scheduler_meta)

        source_image_ref = sched.get("source_image_ref")
        if source_image_ref is None:
            req_meta = batch.get("request_meta") or {}
            src_refs = req_meta.get("source_image_refs") or []
            if len(src_refs) > 0:
                source_image_ref = tuple(src_refs[0])
        if source_image_ref is None:
            source_image_ref = (-1, -1)
        source_image_ref = (int(source_image_ref[0]), int(source_image_ref[1]))

        target_image_refs_raw = sched.get("target_image_refs")
        if target_image_refs_raw is None:
            req_meta = batch.get("request_meta") or {}
            target_image_refs_raw = req_meta.get("target_image_refs") or []
        target_image_refs = [(int(r[0]), int(r[1])) for r in target_image_refs_raw]

        stats: Dict[str, Any] = {
            "num_bg_update": int(sched.get("num_bg_update", 0)),
            "num_distant_update": int(sched.get("num_distant_update", 0)),
            "num_sky_update": int(sched.get("num_sky_update", 0)),
            "num_rigid_update": int(sched.get("num_rigid_update", 0)),
            "src_backproject_pass_count": int(sched.get("src_backproject_pass_count", 0)),
        }

        if segment_aabb is None:
            seg_aabb = batch.get("aabb")
            seg_aabb_cpu = self._as_cpu_tensor(seg_aabb) if torch.is_tensor(seg_aabb) else None
        else:
            seg_aabb_cpu = self._as_cpu_tensor(segment_aabb) if torch.is_tensor(segment_aabb) else None

        rigid_ref = rigid_export_frame_idx
        if rigid_ref is None:
            rigid_ref = gs_state.get("rigid_export_frame_idx")

        snapshot: Dict[str, Any] = {
            "cache_key": dict(gs_state.get("cache_key", {})),
            "source_image_ref": source_image_ref,
            "target_image_refs": target_image_refs,
            "block_idx_global": int(sched.get("block_idx_global", -1)),
            "segment_local_step": int(sched.get("segment_local_step", -1)),
            "rigid_export_frame_idx": int(rigid_ref) if rigid_ref is not None else -1,
            "gs_state": gs_state,
            "stats": stats,
            "include_hidden": bool(include_hidden),
            "allow_hidden_cache_update": bool(allow_hidden_cache_update),
            "allow_node_state_writeback": bool(allow_node_state_writeback),
        }
        if seg_aabb_cpu is not None:
            snapshot["segment_aabb"] = seg_aabb_cpu
        return snapshot

    def render_views_from_scene_state(
        self,
        scene_state: Dict[str, Any],
        eval_views: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        if len(eval_views) == 0:
            return []
        if "base_batch" not in scene_state or "gs_state" not in scene_state:
            raise ValueError("scene_state must contain base_batch and gs_state")
        src_batch = dict(scene_state["base_batch"])
        self.ensure_runtime_state_from_batch(src_batch)
        key = self._batch_key(src_batch)
        snap = self._snapshot_runtime_state(key)
        targets: List[Dict[str, Any]] = []
        for v in eval_views:
            if "gt_image" not in v:
                raise ValueError("Each eval_view must provide gt_image for render size inference")
            view = v["view"]
            viewdirs = v.get("viewdirs")
            if viewdirs is None:
                gt = v["gt_image"]
                if gt.dim() == 4:
                    gt = gt.squeeze(0)
                h, w = int(gt.shape[0]), int(gt.shape[1])
                c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
                intr = view.Ks if hasattr(view, "Ks") else view["Ks"]
                if c2w.dim() == 2:
                    c2w = c2w.unsqueeze(0)
                if intr.dim() == 2:
                    intr = intr.unsqueeze(0)
                y_coords = torch.arange(h, device=gt.device, dtype=torch.float32)
                x_coords = torch.arange(w, device=gt.device, dtype=torch.float32)
                x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
                _, viewdirs, _ = get_rays(
                    x_grid.flatten(),
                    y_grid.flatten(),
                    c2w.to(gt.device),
                    intr.to(gt.device),
                )
                viewdirs = viewdirs.reshape(h, w, 3)
            elif torch.is_tensor(viewdirs):
                viewdirs = viewdirs.to(v["gt_image"].device)
            targets.append(
                {
                    "view": view,
                    "gt_image": v["gt_image"],
                    "frame_idx": int(v["frame_idx"]),
                    "cam_idx": int(v.get("cam_idx", -1)),
                    "sky_mask": v.get("sky_mask"),
                    "egocar_mask": v.get("egocar_mask"),
                    "viewdirs": viewdirs,
                }
            )
        src_batch["targets"] = targets
        self.import_3dgs_state(scene_state["gs_state"], batch_context=src_batch)
        prev_mode = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(src_batch)
        if prev_mode:
            self.train()
        self._restore_runtime_state(key, snap)
        return list(out["pred_rgbs"])

    def inference_step_from_train_batch(
        self,
        batch: Dict,
        step: Optional[int] = None,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[RuntimePolicy] = None,
    ) -> Dict[str, Any]:
        policy = runtime_policy or RuntimePolicy(
            do_backward=False,
            do_optimizer_step=False,
            update_hidden_cache=True,
            writeback_node_state=True,
            reset_node_state_after_block=True,
        )
        if policy.do_backward or policy.do_optimizer_step:
            raise ValueError("inference_step_from_train_batch requires do_backward=false and do_optimizer_step=false")

        self.train()
        self._perf_acc = {}
        node_state_sync_update = False
        node_state_sync_reset = False
        with torch.no_grad():
            out = self.forward(batch)

        if policy.update_hidden_cache and "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()
            if out.get("_h_new_sky") is not None:
                self.h_cache_sky[key] = out["_h_new_sky"].detach()

        if scheduler_node_sync is not None and policy.writeback_node_state:
            U = int(scheduler_node_sync["U"])
            seg = int(scheduler_node_sync["segment_local_step"])
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False)) and policy.reset_node_state_after_block
            if U < 1:
                raise ValueError("scheduler_node_sync requires U >= 1 (scheduler time_base.state_write_interval_steps).")
            if seg > 0 and seg % U == 0:
                self._writeback_node_states_from_out(out)
                node_state_sync_update = True
            if reset_after_block:
                self.reset_node_state()
                node_state_sync_reset = True

        loss_val = out.get("loss")
        return {
            "loss": loss_val.item() if torch.is_tensor(loss_val) else float(loss_val) if loss_val is not None else 0.0,
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "node_state_sync_update": node_state_sync_update,
            "node_state_sync_reset": node_state_sync_reset,
        }

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[RuntimePolicy] = None,
    ) -> Dict[str, Any]:
        policy = runtime_policy or self._default_runtime_policy()
        if policy.do_optimizer_step and not policy.do_backward:
            raise ValueError("RuntimePolicy invalid: do_optimizer_step=true requires do_backward=true")
        self.train()
        self._perf_acc = {}
        node_state_sync_update = False
        node_state_sync_reset = False
        timing_ms: Dict[str, float] = {"forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}
        t0 = time.perf_counter()
        if policy.do_backward:
            self.optimizer.zero_grad()
        out = self.forward(batch)
        if logger.isEnabledFor(logging.DEBUG):
            _ns = out.get("_node_state_sky")
            _n_sky = int(_ns.means.shape[0]) if _ns is not None else 0
            _nu = int(out.get("_num_sky_update", 0))
            logger.debug(
                "sky_step: selective=%s num_sky_src_feat_valid=%s num_sky_update=%s sky_update_ratio=%.6f",
                self.sky_enable_selective_update,
                int(out.get("_num_sky_src_feat_valid", 0)),
                _nu,
                float(_nu) / max(_n_sky, 1),
            )
        t1 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["forward_ms"] = float((t1 - t0) * 1000.0)
        if policy.do_backward and torch.is_tensor(out.get("loss")):
            out["loss"].backward()
        if policy.do_backward and out.get("proxies") is not None:
            _backward_to_render_params_bg_rigid_distant_sky(
                out["render_params"],
                out["proxies"],
                out.get("_render_params_rigid_world"),
                out.get("_proxies_rigid_world"),
                out.get("_render_params_distant"),
                out.get("_proxies_distant"),
                out.get("_render_params_sky"),
                out.get("_proxies_sky"),
                rigid_world_proxy_pairs=out.get("_rigid_world_proxy_pairs"),
            )
        grad_norms = self._compute_branch_grad_norms() if policy.do_backward else {}
        t2 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["backward_ms"] = float((t2 - t1) * 1000.0)
        if policy.do_optimizer_step:
            self.optimizer.step()
        t3 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["optimizer_ms"] = float((t3 - t2) * 1000.0)
        if policy.update_hidden_cache and "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()
            if out.get("_h_new_sky") is not None:
                self.h_cache_sky[key] = out["_h_new_sky"].detach()

        if scheduler_node_sync is not None and policy.writeback_node_state:
            U = int(scheduler_node_sync["U"])
            seg = int(scheduler_node_sync["segment_local_step"])
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False)) and policy.reset_node_state_after_block
            if U < 1:
                raise ValueError("scheduler_node_sync requires U >= 1 (scheduler time_base.state_write_interval_steps).")
            if seg > 0 and seg % U == 0:
                self._writeback_node_states_from_out(out)
                node_state_sync_update = True
            if reset_after_block:
                self.reset_node_state()
                node_state_sync_reset = True

        num_gaussians_bg = int(out["_node_state_bg"].means.shape[0])
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")
        node_state_sky = out.get("_node_state_sky")
        num_gaussians_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_gaussians_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        num_gaussians_sky = int(node_state_sky.means.shape[0]) if node_state_sky is not None else 0
        num_rigid_valid_src = int(out.get("_num_rigid_valid_src", 0))
        num_rigid_total = int(out.get("_num_rigid_total", num_gaussians_rigid))
        writeback_idx = out.get("_rigid_writeback_idx")
        writeback_count = int(writeback_idx.numel()) if writeback_idx is not None else 0
        writeback_rigid_ratio = float(writeback_count / max(num_rigid_total, 1))
        bg_w_idx = out.get("_bg_writeback_idx")
        bg_w_count = int(bg_w_idx.numel()) if bg_w_idx is not None else num_gaussians_bg
        writeback_bg_ratio = float(bg_w_count / max(num_gaussians_bg, 1))
        distant_w_idx = out.get("_distant_writeback_idx")
        distant_w_count = int(distant_w_idx.numel()) if distant_w_idx is not None else num_gaussians_distant
        writeback_distant_ratio = float(distant_w_count / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0
        hidden_stats = out.get("_hidden_stats", {})
        offset_stats = out.get("_offset_stats", {})
        frame_loss_map = out.get("_frame_loss_map", {})

        num_bg_src_feat_valid = int(out.get("_num_bg_src_feat_valid", 0))
        num_bg_update = int(out.get("_num_bg_update", 0))
        num_distant_src_feat_valid = int(out.get("_num_distant_src_feat_valid", 0))
        num_distant_update = int(out.get("_num_distant_update", 0))
        perf_metrics: Dict[str, float] = {}
        perf_calls = float(self._perf_acc.get("2d_call_count", 0.0))
        if perf_calls > 0.0:
            for k, v in self._perf_acc.items():
                if k == "2d_call_count":
                    continue
                # Memory fields keep summed values for deltas; timing fields are averaged per call.
                if "cuda_mem_" in k:
                    perf_metrics[f"perf_{k}"] = float(v)
                else:
                    perf_metrics[f"perf_{k}"] = float(v / perf_calls)
        perf_metrics["perf_2d_call_count"] = perf_calls

        return {
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "loss_l1": out["loss_l1"].item() if torch.is_tensor(out.get("loss_l1")) else float(out.get("loss_l1", 0.0)),
            "loss_ssim": out["loss_ssim"].item() if torch.is_tensor(out.get("loss_ssim")) else float(out.get("loss_ssim", 0.0)),
            "loss_mask": out["loss_mask"].item() if torch.is_tensor(out.get("loss_mask")) else float(out.get("loss_mask", 0.0)),
            "loss_opacity_entropy": out["loss_opacity_entropy"].item() if torch.is_tensor(out.get("loss_opacity_entropy")) else float(out.get("loss_opacity_entropy", 0.0)),
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": num_gaussians_bg,
            "num_gaussians_distant": num_gaussians_distant,
            "num_gaussians_rigid": num_gaussians_rigid,
            "num_rigid_valid_src": num_rigid_valid_src,
            "num_rigid_invalid_src": int(max(num_rigid_total - num_rigid_valid_src, 0)),
            "rigid_valid_ratio": float(num_rigid_valid_src / max(num_rigid_total, 1)),
            "num_rigid_src_feat_valid": int(out.get("_num_rigid_src_feat_valid", 0)),
            "num_rigid_update": int(out.get("_num_rigid_update", 0)),
            "rigid_update_ratio": float(out.get("_rigid_update_ratio", 0.0)),
            "rigid_update_among_feat_valid": float(out.get("_rigid_update_among_feat_valid", 0.0)),
            "writeback_rigid_ratio": writeback_rigid_ratio,
            "num_target_frames": int(out.get("_num_target_frames", 0)),
            "loss_effective_frames": int(out.get("_loss_effective_frames", 0)),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "frame_loss_map": frame_loss_map,
            "hidden_norm_bg_mean": float(hidden_stats.get("hidden_norm_bg_mean", 0.0)),
            "hidden_norm_distant_mean": float(hidden_stats.get("hidden_norm_distant_mean", 0.0)),
            "hidden_norm_rigid_mean": float(hidden_stats.get("hidden_norm_rigid_mean", 0.0)),
            "hidden_norm_sky_mean": float(hidden_stats.get("hidden_norm_sky_mean", 0.0)),
            "num_gaussians_sky": num_gaussians_sky,
            "num_sky_src_feat_valid": int(out.get("_num_sky_src_feat_valid", 0)),
            "num_sky_update": int(out.get("_num_sky_update", 0)),
            "sky_update_ratio": float(out.get("_num_sky_update", 0)) / max(num_gaussians_sky, 1),
            "num_bg_src_feat_valid": num_bg_src_feat_valid,
            "num_bg_update": num_bg_update,
            "bg_update_ratio": float(num_bg_update / max(num_gaussians_bg, 1)),
            "num_distant_src_feat_valid": num_distant_src_feat_valid,
            "num_distant_update": num_distant_update,
            "distant_update_ratio": float(num_distant_update / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0,
            "writeback_bg_ratio": writeback_bg_ratio,
            "writeback_distant_ratio": writeback_distant_ratio,
            "src_backproject_pass_count": int(out.get("_src_backproject_pass_count", 0)),
            **{k: float(v) for k, v in offset_stats.items()},
            **grad_norms,
            **timing_ms,
            **perf_metrics,
            "node_state_sync_update": node_state_sync_update,
            "node_state_sync_reset": node_state_sync_reset,
        }


__all__ = ["MinimalStreetForwardStage4_3", "RuntimePolicy"]
