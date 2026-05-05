from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.streetforward.math_utils import get_viewmat
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import _merge_params_bg_rigid_distant
from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4


class Stage5_6ErrorPredictHead(nn.Module):
    """Predict per-pixel scalar nearby render error and a liftable latent feature."""

    def __init__(
        self,
        in_ch: int,
        hidden_dim: int,
        latent_dim: int,
        error_max: float,
        head_type: str = "dilated_conv",
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"Stage5_6ErrorPredictHead hidden_dim must be > 0, got {hidden_dim}.")
        if latent_dim <= 0:
            raise ValueError(f"Stage5_6ErrorPredictHead latent_dim must be > 0, got {latent_dim}.")
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.error_max = float(error_max)
        self.head_type = str(head_type).strip().lower()
        if self.head_type not in {"dilated_conv", "lite_unet"}:
            raise ValueError("Stage5_6 error_pred.head_type must be one of ['dilated_conv', 'lite_unet'].")

        if self.head_type == "lite_unet":
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
            self.down = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.GroupNorm(groups, hidden_dim),
                nn.GELU(),
            )
        self.err = nn.Conv2d(hidden_dim, 1, 1)
        self.latent = nn.Conv2d(hidden_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.head_type == "lite_unet":
            e = self.enc(x)
            h = self.down(e)
            h = F.interpolate(h, size=e.shape[-2:], mode="bilinear", align_corners=False)
            h = self.fuse(torch.cat([e, h], dim=1))
        else:
            h = self.trunk(x)
        return self.error_max * torch.sigmoid(self.err(h)), self.latent(h)


class ErrorSplatProjector(nn.Module):
    """Project post-update hidden state for nearby error splat."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if int(in_dim) <= 0 or int(out_dim) <= 0:
            raise ValueError(f"ErrorSplatProjector expects positive dims, got in={in_dim}, out={out_dim}.")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError(f"ErrorSplatProjector expects [N,C], got {tuple(hidden.shape)}.")
        if int(hidden.shape[1]) != self.in_dim:
            raise ValueError(
                f"ErrorSplatProjector input dim mismatch: got {hidden.shape[1]} expected {self.in_dim}."
            )
        return self.net(hidden)


class Stage5_6FrameFlattenFuser(nn.Module):
    """Zero-init residual adapter over fixed nearby frame slots."""

    def __init__(
        self,
        feat_dim: int,
        feedback_dim: Optional[int] = None,
        num_slots: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        *,
        input_current_source_support: bool = True,
        input_feedback_support: bool = True,
        zero_init_last: bool = True,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.feedback_dim = int(feedback_dim if feedback_dim is not None else feat_dim)
        self.num_slots = int(num_slots)
        self.input_current_source_support = bool(input_current_source_support)
        self.input_feedback_support = bool(input_feedback_support)
        if self.feat_dim <= 0 or self.feedback_dim <= 0 or self.num_slots <= 0:
            raise ValueError("Stage5_6FrameFlattenFuser requires positive feat_dim/feedback_dim/num_slots.")

        extra_dim = self.num_slots * self.feedback_dim  # flattened slot features
        extra_dim += self.num_slots  # slot error
        extra_dim += self.num_slots  # slot valid
        if self.input_current_source_support:
            extra_dim += 1
        if self.input_feedback_support:
            extra_dim += self.num_slots
        in_dim = self.feat_dim + extra_dim
        layers: List[nn.Module] = []
        hdim = int(hidden_dim)
        depth = max(int(num_layers), 1)
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hdim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hdim, self.feat_dim))
        self.net = nn.Sequential(*layers)
        if zero_init_last:
            with torch.no_grad():
                last = self.net[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

    def _fix_slots(
        self,
        tensor: torch.Tensor,
        *,
        width: int,
        default: float = 0.0,
    ) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError(f"Stage5_6 slot tensor must be rank-3 [N,K,D], got {tuple(tensor.shape)}.")
        n, k, d = int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])
        if d < width:
            pad = tensor.new_full((n, k, width - d), float(default))
            tensor = torch.cat([tensor, pad], dim=-1)
        elif d > width:
            tensor = tensor[:, :, :width]
        if k < self.num_slots:
            pad = tensor.new_full((n, self.num_slots - k, width), float(default))
            tensor = torch.cat([tensor, pad], dim=1)
        elif k > self.num_slots:
            tensor = tensor[:, : self.num_slots, :]
        return tensor

    def forward(
        self,
        feat: torch.Tensor,
        feedback: Any,
        *,
        current_support: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if not isinstance(feedback, dict):
            return feat
        pack = {
            k: v.to(device=feat.device, dtype=feat.dtype)
            for k, v in feedback.items()
            if torch.is_tensor(v)
        }
        if "feat" not in pack or "valid" not in pack or int(pack["feat"].shape[0]) != int(feat.shape[0]):
            return feat
        fb_feat = self._fix_slots(pack["feat"], width=self.feedback_dim, default=0.0)
        fb_error = self._fix_slots(pack.get("error", fb_feat.new_zeros((int(feat.shape[0]), self.num_slots, 1))), width=1)
        fb_valid = self._fix_slots(pack.get("valid", fb_feat.new_zeros((int(feat.shape[0]), self.num_slots, 1))), width=1)
        fb_valid = fb_valid.clamp(0.0, 1.0)

        n, k, c = int(fb_feat.shape[0]), int(fb_feat.shape[1]), int(fb_feat.shape[2])
        inputs = [
            feat,
            fb_feat.reshape(n, k * c),
            torch.log1p(fb_error.clamp_min(0.0)).reshape(n, k),
            fb_valid.reshape(n, k),
        ]
        if self.input_current_source_support:
            if current_support is None:
                cur_s = feat.new_zeros((int(feat.shape[0]), 1))
            else:
                cur_s = current_support.to(device=feat.device, dtype=feat.dtype).reshape(int(feat.shape[0]), 1)
            inputs.append(torch.log1p(cur_s.clamp_min(0.0)))
        if self.input_feedback_support:
            fb_sup = self._fix_slots(
                pack.get("support", fb_feat.new_zeros((int(feat.shape[0]), self.num_slots, 1))),
                width=1,
            )
            inputs.append(torch.log1p(fb_sup.clamp_min(0.0)).reshape(n, k))
        valid_any = fb_valid.max(dim=1).values.clamp(0.0, 1.0)
        delta = self.net(torch.cat(inputs, dim=-1))
        delta = delta * valid_any * float(scale)
        return feat + delta


class MinimalStreetForwardStage5_6(MinimalStreetForwardStage5_4):
    def __init__(self, config, device: torch.device, **kwargs):
        self._stage5_6_frame_cache: Dict[Tuple[int, int, int, int], Dict[int, Dict[str, Any]]] = {}
        self._stage5_6_active_cache: Optional[Dict[str, List[Optional[Dict[str, torch.Tensor]]]]] = None
        self._stage5_6_active_fusion_scale = 0.0
        self._stage5_6_fusion_delta_norm_terms: List[torch.Tensor] = []
        self._stage5_6_last_fused_features: Dict[str, torch.Tensor] = {}
        self._stage5_6_last_nearby_debug_images: List[Dict[str, Any]] = []
        self._stage5_6_last_error_debug_images: List[Dict[str, Any]] = []
        super().__init__(config=config, device=device, **kwargs)
        self._debug_check_stage5_6_optimizer_contains_new_modules()

    def _validate_stage5_3_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if str(self._require_key(model_cfg, "stage", "model")).strip().lower() != "5_6":
            raise ValueError("Stage5_6 requires model.stage='5_6'.")
        old_stage = model_cfg.get("stage")
        model_cfg["stage"] = "5_4"
        try:
            super()._validate_stage5_3_config(config)
        finally:
            model_cfg["stage"] = old_stage

        fsu_cfg = config.get("feature_splat_uncertainty", {}) if hasattr(config, "get") else {}
        nef_cfg = config.get("nearby_error_feedback", {}) if hasattr(config, "get") else {}
        bridge_cfg = fsu_cfg.get("bridge", {}) if hasattr(fsu_cfg, "get") else {}
        top_bridge_cfg = config.get("bridge", {}) if hasattr(config, "get") else {}
        if bool(bridge_cfg.get("enable", False)) or bool(top_bridge_cfg.get("enable", False)):
            raise ValueError("Stage5_6 fast-fail: bridge.enable=true is not supported.")
        if bool((fsu_cfg.get("head") or {}).get("predict_rgb_residual", False)):
            raise ValueError(
                "Stage5_6 fast-fail: feature_splat_uncertainty.head.predict_rgb_residual=true is not supported."
            )
        fsu_loss = fsu_cfg.get("loss", {}) if hasattr(fsu_cfg, "get") else {}
        for key in ("rgb_residual_weight", "rgb_residual_supported_weight"):
            if float(fsu_loss.get(key, 0.0)) != 0.0:
                raise ValueError(f"Stage5_6 fast-fail: feature_splat_uncertainty.loss.{key} must be 0.0.")
        cache_cfg = nef_cfg.get("cache", {}) if hasattr(nef_cfg, "get") else {}
        mode = str(cache_cfg.get("mode", "frame_bank")).strip().lower()
        if mode != "frame_bank":
            raise ValueError("Stage5_6 nearby_error_feedback.cache.mode currently only supports 'frame_bank'.")
        if bool(cache_cfg.get("store_age", False)):
            raise ValueError("Stage5_6 nearby_error_feedback.cache.store_age must be false for frame-slot feedback.")
        error_pred_cfg = nef_cfg.get("error_pred", {}) if hasattr(nef_cfg, "get") else {}
        input_feature = str(error_pred_cfg.get("input_feature", "post_gru_hidden")).strip().lower()
        if input_feature not in {"post_gru_hidden", "post_update_node_feature", "post_struct"}:
            raise ValueError(
                "Stage5_6 nearby_error_feedback.error_pred.input_feature must be one of "
                "['post_gru_hidden', 'post_update_node_feature', 'post_struct']."
            )
        if input_feature == "post_struct":
            raise ValueError(
                "Stage5_6 P0 does not implement error_pred.input_feature='post_struct' yet. "
                "Please use 'post_gru_hidden' or 'post_update_node_feature'."
            )
        if int(error_pred_cfg.get("max_frames_per_step", 1)) < 1:
            raise ValueError("Stage5_6 nearby_error_feedback.error_pred.max_frames_per_step must be >= 1.")
        fusion_cfg = nef_cfg.get("fusion", {}) if hasattr(nef_cfg, "get") else {}
        fusion_type = str(fusion_cfg.get("type", "flatten_frame_slots")).strip().lower()
        if fusion_type != "flatten_frame_slots":
            raise ValueError("Stage5_6 nearby_error_feedback.fusion.type must be 'flatten_frame_slots'.")
        if bool(fusion_cfg.get("input_feedback_age", False)):
            raise ValueError("Stage5_6 nearby_error_feedback.fusion.input_feedback_age must be false.")
        target_role = str(nef_cfg.get("target_role", error_pred_cfg.get("target_role", "nearby_direct")))
        scheduler_cfg = config.get("scheduler_v8", {}) if hasattr(config, "get") else {}
        if hasattr(scheduler_cfg, "get") and target_role == "near_random":
            aux_cfg = scheduler_cfg.get("aux_feature_splat_targets", {}) or {}
            near_random_cfg = scheduler_cfg.get("near_random_supervision", {}) or {}
            if bool(aux_cfg.get("enable", False)):
                raise ValueError(
                    "Stage5_6 target_role='near_random' requires scheduler_v8.aux_feature_splat_targets.enable=false."
                )
            if not bool(near_random_cfg.get("enable", False)):
                raise ValueError(
                    "Stage5_6 target_role='near_random' requires scheduler_v8.near_random_supervision.enable=true."
                )
            if not bool(near_random_cfg.get("sample_once_per_block", True)):
                raise ValueError(
                    "Stage5_6 target_role='near_random' requires "
                    "scheduler_v8.near_random_supervision.sample_once_per_block=true."
                )

    def _parse_target_view_weight_cfg(self, config) -> Dict[str, Any]:
        cfg = super()._parse_target_view_weight_cfg(config)
        losses_cfg = config.get("losses", {}) if hasattr(config, "get") else {}
        tvw_cfg = losses_cfg.get("target_view_weights", {}) if hasattr(losses_cfg, "get") else {}
        nearby_tvw = tvw_cfg.get("nearby_direct", {}) if hasattr(tvw_cfg, "get") else {}
        nearby = config.get("nearby_direct", {}) if hasattr(config, "get") else {}
        cfg["nearby_direct_weight"] = float(nearby_tvw.get("weight", nearby.get("weight", 0.7)))
        return cfg

    def _target_role_weight(self, role: str, step: int) -> float:
        if str(role) == "nearby_direct":
            return float(self._target_view_weight_cfg.get("nearby_direct_weight", 0.7))
        return float(super()._target_role_weight(role, step))

    def _init_stage5_3_modules(self, config) -> None:
        super()._init_stage5_3_modules(config)
        nearby = config.get("nearby_direct", {}) if hasattr(config, "get") else {}
        fsu = config.get("feature_splat_uncertainty", {}) if hasattr(config, "get") else {}
        nef = config.get("nearby_error_feedback", {}) if hasattr(config, "get") else {}

        self.stage5_6_nearby_enabled = bool(nearby.get("enable", False))
        self.stage5_6_nearby_weight = float(
            nearby.get("weight", self._target_view_weight_cfg.get("nearby_direct_weight", 0.7))
        )
        self.stage5_6_nearby_warmup_steps = int(nearby.get("warmup_steps", 0))
        self.stage5_6_nearby_max_refs = int(nearby.get("max_refs_per_step", 1))
        self.stage5_6_nearby_mask_sky = bool(nearby.get("mask_sky", True))
        self.stage5_6_nearby_mask_egocar = bool(nearby.get("mask_egocar", True))
        self.stage5_6_nearby_mask_dynamic = bool(nearby.get("mask_dynamic", False))
        self.stage5_6_nearby_min_valid_pixel_ratio = float(nearby.get("min_valid_pixel_ratio", 0.03))
        self.stage5_6_nearby_role = str(nearby.get("role_name", "nearby_direct"))

        warm_cfg = nef.get("warmup", {}) if hasattr(nef, "get") else {}
        error_pred_cfg = nef.get("error_pred", {}) if hasattr(nef, "get") else {}
        lift_cfg = nef.get("feedback_lift", {}) if hasattr(nef, "get") else {}
        cache_cfg = nef.get("cache", {}) if hasattr(nef, "get") else {}
        fusion_cfg = nef.get("fusion", {}) if hasattr(nef, "get") else {}
        loss_cfg = nef.get("loss", {}) if hasattr(nef, "get") else {}
        fsu_loss = fsu.get("loss", {}) if hasattr(fsu, "get") else {}
        fsu_head = fsu.get("head", {}) if hasattr(fsu, "get") else {}
        fsu_splat = fsu.get("splat", {}) if hasattr(fsu, "get") else {}
        fsu_target = fsu.get("target", {}) if hasattr(fsu, "get") else {}
        legacy_cache = fsu.get("short_cycle_feedback", {}) if hasattr(fsu, "get") else {}

        self.stage5_6_feedback_enabled = bool(nef.get("enable", fsu.get("enable", False)))
        self.stage5_6_error_enabled = bool(error_pred_cfg.get("enable", self.stage5_6_feedback_enabled))
        self.stage5_6_error_target_role = str(nef.get("target_role", error_pred_cfg.get("target_role", "nearby_direct")))
        self.stage5_6_error_input_feature = str(error_pred_cfg.get("input_feature", "post_gru_hidden")).strip().lower()
        if self.stage5_6_error_input_feature not in {"post_gru_hidden", "post_update_node_feature", "post_struct"}:
            raise ValueError(
                "Stage5_6 nearby_error_feedback.error_pred.input_feature must be one of "
                "['post_gru_hidden', 'post_update_node_feature', 'post_struct']."
            )
        if self.stage5_6_error_input_feature == "post_struct":
            raise ValueError(
                "Stage5_6 P0 does not implement error_pred.input_feature='post_struct' yet. "
                "Use 'post_gru_hidden' or 'post_update_node_feature'."
            )
        self.stage5_6_error_splat_dim = int(
            error_pred_cfg.get(
                "error_splat_dim",
                self.offset_gru_hidden_dim if self.stage5_6_error_input_feature == "post_gru_hidden" else self.stage5_2_feat_2d_channels,
            )
        )
        if self.stage5_6_error_splat_dim <= 0:
            raise ValueError("Stage5_6 nearby_error_feedback.error_pred.error_splat_dim must be > 0.")
        self.stage5_6_detach_input_hidden = bool(error_pred_cfg.get("detach_input_hidden", True))
        self.stage5_6_detach_projected_feature = bool(error_pred_cfg.get("detach_projected_feature", False))
        self.stage5_6_node_mask_policy = str(error_pred_cfg.get("node_mask_policy", "renderable")).strip().lower()
        if self.stage5_6_node_mask_policy not in {"renderable", "source_support_threshold"}:
            raise ValueError(
                "Stage5_6 nearby_error_feedback.error_pred.node_mask_policy must be "
                "['renderable', 'source_support_threshold']."
            )
        self.stage5_6_renderable_min_opacity = float(error_pred_cfg.get("renderable_min_opacity", 1.0e-4))
        self.stage5_6_renderable_min_scale = float(error_pred_cfg.get("renderable_min_scale", 1.0e-8))
        max_frames_cfg = error_pred_cfg.get("max_frames_per_step", None)
        if max_frames_cfg is None:
            max_targets_fallback = int(error_pred_cfg.get("max_targets_per_step", fsu_target.get("max_targets_per_step", 1)))
            num_cams_fallback = max(int(getattr(self, "num_cams", 1)), 1)
            max_frames_cfg = max(max_targets_fallback // num_cams_fallback, 1)
        self.stage5_6_target_max_frames = int(max_frames_cfg)
        if self.stage5_6_target_max_frames <= 0:
            raise ValueError("Stage5_6 nearby_error_feedback.error_pred.max_frames_per_step must be >= 1.")
        self.stage5_6_target_every_n_steps = int(error_pred_cfg.get("every_n_steps", fsu_target.get("every_n_steps", 1)))
        self.stage5_6_target_skip_if_no_valid_aux = bool(
            error_pred_cfg.get("skip_if_no_valid_aux", fsu_target.get("skip_if_no_valid_aux", True))
        )

        self.stage5_6_detach_geometry = bool(error_pred_cfg.get("detach_geometry", fsu_splat.get("detach_geometry", True)))
        self.stage5_6_detach_alpha_weights = bool(
            error_pred_cfg.get("detach_alpha_weights", fsu_splat.get("detach_alpha_weights", True))
        )
        self.stage5_6_detach_render_context = bool(
            error_pred_cfg.get("detach_render_context", fsu_splat.get("detach_render_context", True))
        )
        memory_cfg = nef.get("memory", {}) if hasattr(nef, "get") else {}
        self.stage5_6_render_checkpoint = bool(memory_cfg.get("render_checkpoint", True))
        self.stage5_6_splat_eps = float(lift_cfg.get("eps", fsu_splat.get("eps", 1.0e-6)))
        self.stage5_6_use_render_rgb = bool(error_pred_cfg.get("use_render_rgb", fsu_head.get("use_render_rgb", True)))
        self.stage5_6_use_render_alpha = bool(error_pred_cfg.get("use_render_alpha", fsu_head.get("use_render_alpha", True)))

        self.stage5_6_lift_support_min = float(lift_cfg.get("support_min", 1.0e-5))
        self.stage5_6_lift_mask_sky = bool(lift_cfg.get("mask_sky", True))
        self.stage5_6_lift_mask_egocar = bool(lift_cfg.get("mask_egocar", True))
        self.stage5_6_lift_mask_dynamic = bool(lift_cfg.get("mask_dynamic", False))
        self.stage5_6_lift_require_render_alpha = bool(lift_cfg.get("require_render_alpha", True))
        self.stage5_6_lift_render_alpha_min = float(lift_cfg.get("render_alpha_min", 0.02))
        self.stage5_6_detach_lifted_feedback = bool(lift_cfg.get("detach_lifted_feedback", True))

        self.stage5_6_error_weight = float(loss_cfg.get("error_weight", fsu_loss.get("all_valid_weight", 0.03)))
        self.stage5_6_error_loss_type = str(loss_cfg.get("error_loss_type", "charbonnier")).strip().lower()
        if self.stage5_6_error_loss_type not in {"charbonnier", "l1"}:
            raise ValueError("nearby_error_feedback.loss.error_loss_type must be one of ['charbonnier', 'l1'].")
        self.stage5_6_error_warmup_steps = int(loss_cfg.get("error_warmup_steps", fsu_loss.get("warmup_steps", 3000)))
        self.stage5_6_error_start_weight_scale = float(
            loss_cfg.get("error_start_weight_scale", fsu_loss.get("start_weight_scale", 0.0))
        )
        self.stage5_6_error_end_weight_scale = float(
            loss_cfg.get("error_end_weight_scale", fsu_loss.get("end_weight_scale", 1.0))
        )
        self.stage5_6_error_min_valid_pixel_ratio = float(
            loss_cfg.get("min_valid_pixel_ratio", fsu_loss.get("min_valid_pixel_ratio", 0.03))
        )

        self.stage5_6_pred_error_only_steps = int(
            warm_cfg.get("pred_error_only_steps", fsu.get("pred_error_only_steps", legacy_cache.get("pred_error_only_steps", 7000)))
        )
        self.stage5_6_fusion_start_step = int(warm_cfg.get("fusion_start_step", self.stage5_6_pred_error_only_steps))
        self.stage5_6_fusion_warmup_steps = int(warm_cfg.get("fusion_warmup_steps", 3000))
        self.stage5_6_fusion_start_scale = float(warm_cfg.get("fusion_start_scale", 0.0))
        self.stage5_6_fusion_end_scale = float(warm_cfg.get("fusion_end_scale", 1.0))
        self.stage5_6_cache_enable = bool(cache_cfg.get("enable", legacy_cache.get("enable", True)))
        self.stage5_6_cache_max_age = int(cache_cfg.get("max_age", legacy_cache.get("max_age", 1)))
        self.stage5_6_cache_keep_only_current_scope = bool(cache_cfg.get("keep_only_current_scope", True))

        self.stage5_6_fusion_enabled = bool(fusion_cfg.get("enable", True))
        self.stage5_6_fusion_apply_to_bg = bool(fusion_cfg.get("apply_to_bg", True))
        self.stage5_6_fusion_apply_to_distant = bool(fusion_cfg.get("apply_to_distant", True))
        self.stage5_6_fusion_apply_to_rigid = bool(fusion_cfg.get("apply_to_rigid", True))
        self.stage5_6_fusion_input_current_source_support = bool(
            fusion_cfg.get("input_current_source_support", True)
        )
        self.stage5_6_fusion_input_feedback_support = bool(fusion_cfg.get("input_feedback_support", True))
        self.stage5_6_fusion_num_slots = int(fusion_cfg.get("num_slots", self.stage5_6_target_max_frames))
        if self.stage5_6_fusion_num_slots <= 0:
            raise ValueError("Stage5_6 nearby_error_feedback.fusion.num_slots must be >= 1.")

        feat_dim = int(self.stage5_2_feat_2d_channels)
        hidden_dim = int(error_pred_cfg.get("hidden_dim", fsu_head.get("hidden_dim", 64)))
        error_feat_dim = int(error_pred_cfg.get("error_feat_dim", 8))
        error_max = float(error_pred_cfg.get("error_max", fsu_head.get("error_max", 0.5)))
        head_type = str(error_pred_cfg.get("head_type", "dilated_conv")).strip().lower()
        in_ch = int(self.stage5_6_error_splat_dim)
        if self.stage5_6_use_render_rgb:
            in_ch += 3
        if self.stage5_6_use_render_alpha:
            in_ch += 1
        self.stage5_6_error_feat_dim = int(error_feat_dim)
        hidden_in_dim = int(self.offset_gru_hidden_dim)
        self.err_splat_proj_bg = ErrorSplatProjector(hidden_in_dim, int(self.stage5_6_error_splat_dim)).to(self.device)
        self.err_splat_proj_distant = ErrorSplatProjector(hidden_in_dim, int(self.stage5_6_error_splat_dim)).to(self.device)
        self.err_splat_proj_rigid = ErrorSplatProjector(hidden_in_dim, int(self.stage5_6_error_splat_dim)).to(self.device)
        self.stage5_6_error_head = Stage5_6ErrorPredictHead(
            in_ch=in_ch,
            hidden_dim=hidden_dim,
            latent_dim=error_feat_dim,
            error_max=error_max,
            head_type=head_type,
        ).to(self.device)

        fuse_hidden = int(fusion_cfg.get("hidden_dim", 64))
        fuse_layers = int(fusion_cfg.get("num_layers", 2))
        fuser_kwargs = {
            "feat_dim": feat_dim,
            "feedback_dim": error_feat_dim,
            "num_slots": self.stage5_6_fusion_num_slots,
            "hidden_dim": fuse_hidden,
            "num_layers": fuse_layers,
            "input_current_source_support": self.stage5_6_fusion_input_current_source_support,
            "input_feedback_support": self.stage5_6_fusion_input_feedback_support,
            "zero_init_last": bool(fusion_cfg.get("zero_init_last", True)),
        }
        self.stage5_6_bg_fuser = Stage5_6FrameFlattenFuser(**fuser_kwargs).to(self.device)
        self.stage5_6_distant_fuser = Stage5_6FrameFlattenFuser(**fuser_kwargs).to(self.device)
        self.stage5_6_rigid_fuser = Stage5_6FrameFlattenFuser(**fuser_kwargs).to(self.device)

    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps)

    def _stage5_6_should_checkpoint_render(self, *tensors: torch.Tensor) -> bool:
        if not bool(getattr(self, "stage5_6_render_checkpoint", True)):
            return False
        if not bool(getattr(self, "training", False)) or not torch.is_grad_enabled():
            return False
        return any(torch.is_tensor(t) and bool(t.requires_grad) for t in tensors)

    def _stage5_6_render_rgb_alpha(
        self,
        *,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        sh_degree: Optional[int],
        absgrad: bool,
        channel_chunk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _render(
            means_t: torch.Tensor,
            quats_t: torch.Tensor,
            scales_t: torch.Tensor,
            opacities_t: torch.Tensor,
            colors_t: torch.Tensor,
            viewmats_t: torch.Tensor,
            Ks_t: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            kwargs = {
                "means": means_t,
                "quats": quats_t,
                "scales": scales_t,
                "opacities": opacities_t,
                "colors": colors_t,
                "viewmats": viewmats_t,
                "Ks": Ks_t,
                "width": int(width),
                "height": int(height),
                "tile_size": 16,
                "packed": False,
                "near_plane": 0.01,
                "far_plane": 1e10,
                "render_mode": "RGB",
                "sh_degree": sh_degree,
                "sparse_grad": False,
                "absgrad": bool(absgrad),
                "rasterize_mode": "classic",
            }
            if channel_chunk is not None:
                kwargs["channel_chunk"] = int(channel_chunk)
            render, alpha, _ = self.renderer(**kwargs)
            return render, alpha

        args = (means, quats, scales, opacities, colors, viewmats, Ks)
        if self._stage5_6_should_checkpoint_render(*args):
            return checkpoint(_render, *args, use_reentrant=False)
        return _render(*args)

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view: Any,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = render_params["means_r"]
        dtype = means.dtype
        dev = means.device
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = get_viewmat(c2w.to(device=dev, dtype=dtype))
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=dev, dtype=dtype).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)
        k_mat = k_mat.to(device=dev, dtype=dtype)
        render, alpha = self._stage5_6_render_rgb_alpha(
            means=means,
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=viewmat,
            Ks=k_mat,
            width=int(width),
            height=int(height),
            sh_degree=self.sh_degree,
            absgrad=True,
        )
        rgb = render[:, ..., :3].squeeze(0)
        acc = alpha.squeeze(0)
        return rgb, acc

    def _render_multi_view(
        self,
        render_params: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if not targets:
            return None
        means = render_params["means_r"]
        dtype = means.dtype
        dev = means.device
        viewmats_list: List[torch.Tensor] = []
        Ks_list: List[torch.Tensor] = []
        heights: List[int] = []
        widths: List[int] = []
        for target in targets:
            view = target["view"]
            gt_image = target["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            h, w = int(gt_image.shape[0]), int(gt_image.shape[1])
            heights.append(h)
            widths.append(w)
            c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
            viewmats_list.append(get_viewmat(c2w.to(device=dev, dtype=dtype)))
            if hasattr(view, "Ks"):
                k_mat = view.Ks[0:1]
            elif hasattr(view, "K"):
                k_mat = view.K
            else:
                k_mat = torch.eye(3, device=dev, dtype=dtype).unsqueeze(0)
            if k_mat.dim() == 2:
                k_mat = k_mat.unsqueeze(0)
            Ks_list.append(k_mat.to(device=dev, dtype=dtype))
        h0, w0 = heights[0], widths[0]
        if any(h != h0 or w != w0 for h, w in zip(heights, widths)):
            return None
        viewmats = torch.cat(viewmats_list, dim=0)
        Ks = torch.cat(Ks_list, dim=0)
        render, alpha = self._stage5_6_render_rgb_alpha(
            means=means,
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=viewmats,
            Ks=Ks,
            width=int(w0),
            height=int(h0),
            sh_degree=self.sh_degree,
            absgrad=True,
        )
        result: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c in range(int(viewmats.shape[0])):
            result.append((render[c, ..., :3], alpha[c, ..., 0]))
        return result

    def _debug_check_stage5_6_optimizer_contains_new_modules(self) -> None:
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        opt_param_ids = {id(p) for group in opt.param_groups for p in group.get("params", [])}
        modules = {
            "stage5_6_error_head": getattr(self, "stage5_6_error_head", None),
            "err_splat_proj_bg": getattr(self, "err_splat_proj_bg", None),
            "err_splat_proj_distant": getattr(self, "err_splat_proj_distant", None),
            "err_splat_proj_rigid": getattr(self, "err_splat_proj_rigid", None),
            "stage5_6_bg_fuser": getattr(self, "stage5_6_bg_fuser", None),
            "stage5_6_distant_fuser": getattr(self, "stage5_6_distant_fuser", None),
            "stage5_6_rigid_fuser": getattr(self, "stage5_6_rigid_fuser", None),
        }
        missing: List[str] = []
        for module_name, module in modules.items():
            if module is None:
                missing.append(f"{module_name}: module is None")
                continue
            for name, p in module.named_parameters():
                if p.requires_grad and id(p) not in opt_param_ids:
                    missing.append(f"{module_name}.{name}")
        if missing:
            raise RuntimeError(
                "Stage5_6 new module parameters are not in optimizer: " + ", ".join(missing[:16])
            )

    def _collect_role_targets(
        self,
        batch: Dict[str, Any],
        *,
        role: str,
        max_targets: int,
        prefer_targets: bool = True,
        fallback_aux: bool = True,
        require_aux_if_requested: bool = False,
    ) -> List[Dict[str, Any]]:
        request_meta = batch.get("request_meta") or {}
        wanted_role = str(role)
        targets = batch.get("targets") if prefer_targets else None
        roles = [str(x) for x in list(request_meta.get("target_image_roles") or [])]
        if isinstance(targets, list) and len(roles) == len(targets) and len(roles) > 0:
            matched: List[Dict[str, Any]] = []
            for idx, (target, target_role) in enumerate(zip(targets, roles)):
                if str(target_role) != wanted_role:
                    continue
                item = dict(target)
                item["role"] = str(target_role)
                item["batch_target_index"] = int(idx)
                item["target_index"] = int(len(matched))
                matched.append(item)
            limit = int(max_targets)
            if limit > 0:
                matched = matched[:limit]
            if len(matched) > 0 or not fallback_aux:
                return matched

        requested_aux = request_meta.get("aux_image_refs") or batch.get("aux_image_refs") or []
        aux = batch.get("aux_targets")
        if require_aux_if_requested and len(requested_aux) > 0:
            if not isinstance(aux, list) or len(aux) == 0:
                raise RuntimeError(
                    "Stage5_6 got aux_image_refs but batch['aux_targets'] is missing or empty. "
                    "Dataset/conversion must materialize nearby aux targets before trainer.forward()."
                )
        if not isinstance(aux, list):
            return []
        aux_roles = [str(x) for x in list(request_meta.get("aux_image_roles") or [])]
        if len(aux_roles) == len(aux) and len(aux_roles) > 0:
            filtered: List[Dict[str, Any]] = []
            for idx, (target, aux_role) in enumerate(zip(aux, aux_roles)):
                if str(aux_role) != wanted_role:
                    continue
                item = dict(target)
                item["role"] = str(aux_role)
                item["batch_target_index"] = int(idx)
                item["target_index"] = int(len(filtered))
                filtered.append(item)
            aux = filtered
        else:
            aux = [dict(t, role=wanted_role, target_index=i) for i, t in enumerate(aux)]
        limit = int(max_targets)
        if limit > 0:
            return aux[:limit]
        return aux

    def _collect_nearby_aux_targets(
        self,
        batch: Dict[str, Any],
        *,
        max_targets: int,
        require_materialized: bool,
    ) -> List[Dict[str, Any]]:
        return self._collect_role_targets(
            batch,
            role=str(getattr(self, "stage5_6_nearby_role", "nearby_direct")),
            max_targets=int(max_targets),
            prefer_targets=False,
            fallback_aux=True,
            require_aux_if_requested=bool(require_materialized),
        )

    def _collect_feedback_targets(
        self,
        batch: Dict[str, Any],
        *,
        max_targets: int,
        require_aux_if_requested: bool,
    ) -> List[Dict[str, Any]]:
        return self._collect_role_targets(
            batch,
            role=str(getattr(self, "stage5_6_error_target_role", "near_random")),
            max_targets=int(max_targets),
            prefer_targets=True,
            fallback_aux=True,
            require_aux_if_requested=bool(require_aux_if_requested),
        )

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        if torch.is_tensor(value):
            if int(value.numel()) == 0:
                return int(default)
            return int(value.reshape(-1)[0].item())
        if value is None:
            return int(default)
        return int(value)

    def _stage5_6_scope_key(self, batch: Dict[str, Any]) -> Tuple[int, int, int, int]:
        scene_id, segment_id = self._batch_key(batch)
        aligned = (
            batch.get("_scheduler_v8_aligned_info")
            or batch.get("_scheduler_v7_aligned_info")
            or batch.get("_scheduler_v4_aligned_info")
            or {}
        )
        request_meta = batch.get("request_meta") or {}
        episode_idx = self._safe_int(
            aligned.get(
                "episode_idx_global",
                aligned.get(
                    "episode_idx",
                    request_meta.get("episode_idx_global", request_meta.get("episode_idx", None)),
                ),
            )
        )
        block_idx = self._safe_int(
            aligned.get(
                "block_idx_global",
                request_meta.get("block_idx_global", None),
            )
        )
        if int(episode_idx) < 0 or int(block_idx) < 0:
            raise RuntimeError(
                "Stage5_6 frame cache requires episode_idx_global and block_idx_global. "
                "Check scheduler aligned info / request_meta propagation."
            )
        return (int(scene_id), int(segment_id), int(episode_idx), int(block_idx))

    def _feedback_frame_indices_for_fusion(self, batch: Dict[str, Any]) -> List[int]:
        request_meta = batch.get("request_meta") or {}
        role = str(getattr(self, "stage5_6_error_target_role", "near_random"))
        direct = request_meta.get(f"{role}_frame_indices")
        if isinstance(direct, list) and len(direct) > 0:
            return [int(x) for x in direct]
        if role == "near_random":
            near_random = request_meta.get("near_random_frame_indices") or []
            if len(near_random) > 0:
                return [int(x) for x in near_random]

        ordered: List[int] = []
        seen: set[int] = set()

        def _collect_from_refs(refs: Any, roles: Any) -> None:
            if not isinstance(refs, list) or not isinstance(roles, list):
                return
            if len(refs) != len(roles):
                return
            for ref, r in zip(refs, roles):
                if str(r) != role:
                    continue
                if not isinstance(ref, (list, tuple)) or len(ref) < 1:
                    continue
                frame_idx = self._safe_int(ref[0], default=-1)
                if frame_idx < 0 or frame_idx in seen:
                    continue
                seen.add(frame_idx)
                ordered.append(frame_idx)

        _collect_from_refs(request_meta.get("target_image_refs"), request_meta.get("target_image_roles"))
        _collect_from_refs(request_meta.get("aux_image_refs"), request_meta.get("aux_image_roles"))
        return ordered

    def _collect_role_frame_targets(
        self,
        batch: Dict[str, Any],
        *,
        role: str,
        max_frames: int,
        require_aux_if_requested: bool,
    ) -> List[Tuple[int, List[Dict[str, Any]]]]:
        image_targets = self._collect_role_targets(
            batch,
            role=str(role),
            max_targets=-1,
            prefer_targets=True,
            fallback_aux=True,
            require_aux_if_requested=bool(require_aux_if_requested),
        )
        if len(image_targets) == 0:
            return []
        all_targets = batch.get("targets") or []
        expected_cam_set = {
            int(t.get("cam_idx"))
            for t in all_targets
            if isinstance(t, dict) and t.get("cam_idx") is not None
        }
        if len(expected_cam_set) == 0:
            expected_cam_set = {
                int(t.get("cam_idx"))
                for t in image_targets
                if isinstance(t, dict) and t.get("cam_idx") is not None
            }
        if len(expected_cam_set) == 0:
            expected_cam_set = {0}
        expected_cam_order = sorted(int(x) for x in expected_cam_set)
        grouped: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for item in image_targets:
            frame_idx = item.get("frame_idx")
            cam_idx = item.get("cam_idx")
            if frame_idx is None or cam_idx is None:
                continue
            frame_i = int(frame_idx)
            cam_i = int(cam_idx)
            if frame_i not in grouped:
                grouped[frame_i] = {}
            grouped[frame_i][cam_i] = item
        if len(grouped) == 0:
            return []
        ordered_frames = sorted(grouped.keys())
        frame_groups: List[Tuple[int, List[Dict[str, Any]]]] = []
        for frame_idx in ordered_frames:
            cam_map = grouped[frame_idx]
            if len(cam_map) != len(expected_cam_order):
                continue
            if set(cam_map.keys()) != set(expected_cam_order):
                continue
            frame_groups.append((int(frame_idx), [cam_map[i] for i in expected_cam_order]))
        if int(max_frames) > 0:
            frame_groups = frame_groups[: int(max_frames)]
        return frame_groups

    def _mask_hw(self, mask: torch.Tensor, h: int, w: int, name: str = "mask") -> torch.Tensor:
        m = mask.to(self.device).float()
        while m.dim() > 2:
            if int(m.shape[0]) == 1:
                m = m.squeeze(0)
            elif int(m.shape[-1]) == 1:
                m = m.squeeze(-1)
            else:
                raise ValueError(f"Stage5_6 {name} cannot be squeezed to [H,W]: got {tuple(m.shape)}.")
        if tuple(m.shape) != (int(h), int(w)):
            raise ValueError(f"Stage5_6 {name} shape mismatch: got {tuple(m.shape)}, expect {(h, w)}.")
        return m

    def _nearby_weight(self, step: int) -> float:
        if self.stage5_6_nearby_warmup_steps <= 0:
            return float(self.stage5_6_nearby_weight)
        ratio = min(1.0, max(0.0, float(step) / float(self.stage5_6_nearby_warmup_steps)))
        return float(self.stage5_6_nearby_weight * ratio)

    def _fusion_scale(self, step: int) -> float:
        if not bool(getattr(self, "stage5_6_cache_enable", True)) or not bool(
            getattr(self, "stage5_6_fusion_enabled", True)
        ):
            return 0.0
        start = int(getattr(self, "stage5_6_fusion_start_step", 7000))
        if int(step) < start:
            return 0.0
        warm = int(getattr(self, "stage5_6_fusion_warmup_steps", 3000))
        s0 = float(getattr(self, "stage5_6_fusion_start_scale", 0.0))
        s1 = float(getattr(self, "stage5_6_fusion_end_scale", 1.0))
        if warm <= 0:
            return float(s1)
        t = min(1.0, max(0.0, float(int(step) - start) / float(warm)))
        return float(s0 + t * (s1 - s0))

    def _cache_ready(self, step: int) -> bool:
        return int(step) >= int(self.stage5_6_pred_error_only_steps)

    def _stage5_6_detach_render_params(self, render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in render_params.items()}

    def _build_rigid_world_for_aux_frame(
        self,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        node_state_rigid = out.get("_node_state_rigid")
        if node_state_rigid is None:
            return None, torch.zeros((0,), dtype=torch.long, device=self.device)
        u_all = out.get("_rigid_writeback_idx")
        if u_all is None:
            u_all = torch.zeros((0,), dtype=torch.long, device=self.device)
        else:
            u_all = u_all.to(device=self.device, dtype=torch.long)
        n_rigid = int(node_state_rigid.means.shape[0])
        if int(u_all.numel()) > 0 and (bool((u_all < 0).any().item()) or bool((u_all >= n_rigid).any().item())):
            raise RuntimeError("Stage5_6 rigid aux U_all contains out-of-range indices.")
        is_updated = torch.zeros((n_rigid,), dtype=torch.bool, device=self.device)
        if int(u_all.numel()) > 0:
            is_updated[u_all] = True
        target_valid = torch.nonzero(
            self._rigid_point_valid_mask(node_state_rigid, int(target_frame_idx)),
            as_tuple=False,
        ).squeeze(1).to(device=self.device, dtype=torch.long)
        if int(target_valid.numel()) == 0:
            return None, target_valid
        idx_train = target_valid[is_updated[target_valid]]
        idx_frozen = target_valid[~is_updated[target_valid]]
        rigid_local = out.get("_render_params_rigid_local")
        if int(idx_train.numel()) > 0 and rigid_local is None:
            raise RuntimeError("Stage5_6 aux needs _render_params_rigid_local for updated rigid nodes.")
        rigid_world = self._build_rigid_world_for_frame(
            node_state_rigid,
            int(target_frame_idx),
            idx_train,
            idx_frozen,
            rigid_local,
            u_all,
        )
        return rigid_world, torch.cat([idx_train, idx_frozen], dim=0)

    def _build_nearby_direct_proxy_render_params(
        self,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        proxies_bg = out.get("_proxies_bg") or out.get("proxies")
        render_bg = out.get("_render_params_bg") or out.get("render_params")
        if proxies_bg is None or render_bg is None:
            return None
        render_distant = out.get("_render_params_distant")
        proxies_distant = out.get("_proxies_distant")
        if render_distant is not None and proxies_distant is None:
            raise RuntimeError("Stage5_6 nearby_direct expected _proxies_distant for distant render params.")
        rigid_world, _rigid_order = self._build_rigid_world_for_aux_frame(out, int(target_frame_idx))
        proxy_rigid = None
        if rigid_world is not None:
            proxy_rigid = _create_proxy_params(rigid_world)
            pairs = out.get("_rigid_world_proxy_pairs")
            if pairs is None:
                pairs = []
                out["_rigid_world_proxy_pairs"] = pairs
            elif not isinstance(pairs, list):
                pairs = list(pairs)
                out["_rigid_world_proxy_pairs"] = pairs
            pairs.append((rigid_world, proxy_rigid))
        return _merge_params_bg_rigid_distant(proxies_bg, proxy_rigid, proxies_distant)

    def _build_nearby_mask(self, target: Dict[str, Any], h: int, w: int) -> torch.Tensor:
        valid_target = target
        egocar_for_valid = target.get("egocar_mask")
        if egocar_for_valid is not None:
            valid_target = dict(target)
            valid_target["egocar_mask"] = self._mask_hw(egocar_for_valid, h, w, "egocar_mask")
        mask = self._valid_loss_mask_from_target(valid_target, height=h, width=w).to(self.device).float()
        if self.stage5_6_nearby_mask_sky and target.get("sky_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["sky_mask"], h, w, "sky_mask")).clamp(0.0, 1.0)
        if self.stage5_6_nearby_mask_egocar and target.get("egocar_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["egocar_mask"], h, w, "egocar_mask")).clamp(0.0, 1.0)
        if self.stage5_6_nearby_mask_dynamic and target.get("dynamic_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["dynamic_mask"], h, w, "dynamic_mask")).clamp(0.0, 1.0)
        return mask

    def _build_error_mask(
        self,
        target: Dict[str, Any],
        h: int,
        w: int,
        render_alpha: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = self._build_nearby_mask(target, h, w)
        if bool(getattr(self, "stage5_6_lift_mask_sky", True)) and target.get("sky_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["sky_mask"], h, w, "sky_mask")).clamp(0.0, 1.0)
        if bool(getattr(self, "stage5_6_lift_mask_egocar", True)) and target.get("egocar_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["egocar_mask"], h, w, "egocar_mask")).clamp(0.0, 1.0)
        if bool(getattr(self, "stage5_6_lift_mask_dynamic", False)) and target.get("dynamic_mask") is not None:
            mask = mask * (1.0 - self._mask_hw(target["dynamic_mask"], h, w, "dynamic_mask")).clamp(0.0, 1.0)
        if bool(getattr(self, "stage5_6_lift_require_render_alpha", True)) and render_alpha is not None:
            alpha = render_alpha
            if alpha.dim() == 3 and int(alpha.shape[-1]) == 1:
                alpha = alpha.squeeze(-1)
            mask = mask * (alpha.detach().to(device=self.device).float() > float(self.stage5_6_lift_render_alpha_min)).float()
        return mask

    def _compute_nearby_direct_loss(self, batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
        zero = out["loss"].new_zeros(())
        if not self.stage5_6_nearby_enabled:
            return {"loss": zero, "processed": 0.0}
        collect_debug = bool(batch.get("_stage5_6_collect_debug_images", False))
        aux = self._collect_nearby_aux_targets(
            batch,
            max_targets=int(self.stage5_6_nearby_max_refs),
            require_materialized=True,
        )
        if len(aux) == 0:
            return {"loss": zero, "processed": 0.0, "skipped_empty": 1.0}
        step = int(self._current_loss_step(batch))
        view_weight = self._nearby_weight(step)
        if view_weight <= 0.0:
            return {"loss": zero, "processed": 0.0, "view_weight": float(view_weight), "skipped_zero_weight": 1.0}

        loss_terms: List[torch.Tensor] = []
        l1_terms: List[torch.Tensor] = []
        psnr_terms: List[torch.Tensor] = []
        valid_terms: List[torch.Tensor] = []
        debug_images: List[Dict[str, Any]] = []
        skipped_low_valid = 0
        for target_idx, t in enumerate(aux):
            gt = t.get("gt_image")
            view = t.get("view")
            frame_idx = t.get("frame_idx")
            if gt is None or view is None or frame_idx is None:
                continue
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            h, w = int(gt.shape[0]), int(gt.shape[1])
            rp = self._build_nearby_direct_proxy_render_params(out, int(frame_idx))
            if rp is None:
                continue
            pred, _alpha = self._render_single_view(rp, view, h, w)
            gt = gt.to(device=pred.device, dtype=pred.dtype)
            mask = self._build_nearby_mask(t, h, w).to(device=pred.device, dtype=pred.dtype)
            valid_ratio = mask.mean()
            if float(valid_ratio.detach().item()) < float(self.stage5_6_nearby_min_valid_pixel_ratio):
                skipped_low_valid += 1
                continue
            denom = mask.sum().clamp_min(1.0)
            l1_raw = (pred - gt).abs().mul(mask.unsqueeze(-1)).sum() / (denom * 3.0)
            ssim = compute_ssim_loss_masked(pred, gt, valid_mask=mask, sky_mask=None, data_range=1.0)
            loss_terms.append(
                pred.new_tensor(float(view_weight)) * (float(self.loss_w_l1) * l1_raw + float(self.loss_w_ssim) * ssim)
            )
            mse = (((pred - gt) ** 2) * mask.unsqueeze(-1)).sum() / (denom * 3.0)
            psnr_terms.append((-10.0 * torch.log10(mse.clamp_min(1.0e-10))).detach())
            l1_terms.append(l1_raw.detach())
            valid_terms.append(valid_ratio.detach())
            if collect_debug:
                debug_images.append(
                    {
                        "target_index": int(t.get("target_index", target_idx)),
                        "frame_idx": int(frame_idx),
                        "cam_idx": int(t.get("cam_idx", -1)),
                        "role": str(t.get("role", self.stage5_6_nearby_role)),
                        "pred": pred.detach().float().cpu(),
                        "gt": gt.detach().float().cpu(),
                    }
                )
        if len(loss_terms) == 0:
            return {
                "loss": zero,
                "processed": 0.0,
                "view_weight": float(view_weight),
                "skipped_low_valid": float(skipped_low_valid),
            }
        return {
            "loss": torch.stack(loss_terms).mean(),
            "processed": float(len(loss_terms)),
            "view_weight": float(view_weight),
            "monitor_l1": torch.stack(l1_terms).mean(),
            "monitor_psnr": torch.stack(psnr_terms).mean(),
            "valid_mask_ratio": torch.stack(valid_terms).mean(),
            "skipped_low_valid": float(skipped_low_valid),
            **({"debug_images": debug_images} if collect_debug else {}),
        }

    def _feature_splat_render(
        self,
        *,
        render_params: Dict[str, torch.Tensor],
        colors: torch.Tensor,
        views: List[Any],
        height: int,
        width: int,
        detach_geometry: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if colors.ndim != 2:
            raise ValueError(f"Stage5_6 feature colors must be [N,C], got {tuple(colors.shape)}.")
        if len(views) == 0:
            raise ValueError("Stage5_6 feature splat requires at least one view.")
        means = render_params["means_r"]
        dtype = means.dtype
        dev = means.device
        viewmats: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []
        for view in views:
            c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
            vm = get_viewmat(c2w.to(device=dev, dtype=dtype))
            if hasattr(view, "Ks"):
                ks = view.Ks[0:1]
            elif hasattr(view, "K"):
                ks = view.K
            else:
                ks = torch.eye(3, device=dev, dtype=dtype).unsqueeze(0)
            if ks.dim() == 2:
                ks = ks.unsqueeze(0)
            viewmats.append(vm.to(device=dev, dtype=dtype))
            intrinsics.append(ks.to(device=dev, dtype=dtype))
        viewmat = torch.cat(viewmats, dim=0)
        Ks = torch.cat(intrinsics, dim=0)
        quats = render_params["quats_r"]
        scales = render_params["scales_r"]
        opacities = render_params["opacities_r"]
        if detach_geometry:
            means = means.detach()
            quats = quats.detach()
            scales = scales.detach()
            opacities = opacities.detach()
        render, alpha = self._stage5_6_render_rgb_alpha(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=int(width),
            height=int(height),
            sh_degree=None,
            absgrad=False,
            channel_chunk=max(32, int(colors.shape[-1])),
        )
        feat = render
        acc = alpha
        if acc.dim() == 4 and int(acc.shape[-1]) == 1:
            acc = acc[..., 0]
        return feat, acc

    def _splat_node_features_to_view(
        self,
        *,
        render_params: Dict[str, torch.Tensor],
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        view: Any,
        height: int,
        width: int,
        detach_geometry: bool,
        detach_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_all, support_all = self._splat_node_features_to_views(
            render_params=render_params,
            node_features=node_features,
            node_mask=node_mask,
            views=[view],
            height=height,
            width=width,
            detach_geometry=detach_geometry,
            detach_weights=detach_weights,
        )
        return feat_all[0], support_all[0]

    def _splat_node_features_to_views(
        self,
        *,
        render_params: Dict[str, torch.Tensor],
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        views: List[Any],
        height: int,
        width: int,
        detach_geometry: bool,
        detach_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(detach_weights) and not bool(detach_geometry):
            raise ValueError("Stage5_6 detach_alpha_weights=true requires detach_geometry=true.")
        if int(node_features.shape[0]) != int(node_mask.shape[0]):
            raise ValueError("Stage5_6 node_features/node_mask length mismatch.")
        n = int(node_features.shape[0])
        c = int(node_features.shape[1])
        v = int(len(views))
        if v <= 0:
            raise ValueError("Stage5_6 splat_to_views expects at least one view.")
        render_n = int(render_params["means_r"].shape[0])
        if render_n != n:
            raise ValueError(f"Stage5_6 render/features length mismatch: {render_n} vs {n}.")
        render_dev = render_params["means_r"].device
        render_dtype = render_params["means_r"].dtype
        if n == 0 or int(node_mask.sum().item()) == 0:
            return (
                torch.zeros((v, height, width, c), dtype=render_dtype, device=render_dev),
                torch.zeros((v, height, width), dtype=render_dtype, device=render_dev),
            )
        mask_f = node_mask.to(device=render_dev, dtype=render_dtype).reshape(n, 1)
        colors = torch.zeros((n, c + 1), dtype=render_dtype, device=render_dev)
        colors[:, :c] = node_features.to(device=render_dev, dtype=render_dtype) * mask_f
        colors[:, c : c + 1] = mask_f
        rendered, _alpha = self._feature_splat_render(
            render_params=render_params,
            colors=colors,
            views=views,
            height=height,
            width=width,
            detach_geometry=detach_geometry,
        )
        feat_sum = rendered[..., :c]  # [V,H,W,C]
        support = rendered[..., c]  # [V,H,W]
        norm_support = support.detach() if detach_weights else support
        return feat_sum / (norm_support.unsqueeze(-1) + float(self.stage5_6_splat_eps)), support

    @staticmethod
    def _merge_lists(items: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        valid = [x for x in items if x is not None and int(x.shape[0]) > 0]
        if len(valid) == 0:
            return None
        return torch.cat(valid, dim=0)

    def _current_or_fused_feature(self, out: Dict[str, Any], key: str, fallback_key: str) -> Optional[torch.Tensor]:
        fused = self._stage5_6_last_fused_features.get(key)
        if fused is not None:
            return fused
        return out.get(fallback_key)

    def _project_hidden_for_error_splat(
        self,
        hidden: Optional[torch.Tensor],
        projector: ErrorSplatProjector,
        *,
        branch_name: str,
    ) -> Optional[torch.Tensor]:
        if hidden is None:
            return None
        if hidden.dim() != 2:
            raise RuntimeError(f"Stage5_6 {branch_name} hidden must be [N,C], got {tuple(hidden.shape)}.")
        h_in = hidden
        if bool(self.stage5_6_detach_input_hidden):
            h_in = h_in.detach()
        projected = projector(h_in.to(device=self.device))
        if bool(self.stage5_6_detach_projected_feature):
            projected = projected.detach()
        return projected

    def _prepare_error_splat_features(self, out: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        if self.stage5_6_error_input_feature != "post_gru_hidden":
            return {
                "bg": out.get("_err_splat_feat_bg"),
                "distant": out.get("_err_splat_feat_distant"),
                "rigid": out.get("_err_splat_feat_rigid"),
            }
        feat_bg = out.get("_err_splat_feat_bg")
        if feat_bg is None:
            feat_bg = self._project_hidden_for_error_splat(
                out.get("_h_new_bg"),
                self.err_splat_proj_bg,
                branch_name="bg",
            )
            if feat_bg is None:
                raise RuntimeError("Stage5_6 post_gru_hidden requires _h_new_bg in forward out.")
            out["_err_splat_feat_bg"] = feat_bg

        feat_distant = out.get("_err_splat_feat_distant")
        hidden_distant = out.get("_h_new_distant")
        if feat_distant is None and hidden_distant is not None:
            feat_distant = self._project_hidden_for_error_splat(
                hidden_distant,
                self.err_splat_proj_distant,
                branch_name="distant",
            )
            out["_err_splat_feat_distant"] = feat_distant

        feat_rigid = out.get("_err_splat_feat_rigid")
        hidden_rigid = out.get("_h_new_rigid")
        if feat_rigid is None and hidden_rigid is not None:
            feat_rigid = self._project_hidden_for_error_splat(
                hidden_rigid,
                self.err_splat_proj_rigid,
                branch_name="rigid",
            )
            out["_err_splat_feat_rigid"] = feat_rigid

        return {"bg": feat_bg, "distant": feat_distant, "rigid": feat_rigid}

    def _renderable_node_mask(
        self,
        *,
        node_features: torch.Tensor,
        render_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        n = int(node_features.shape[0])
        if n == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        if any(int(render_params[k].shape[0]) != n for k in ("means_r", "scales_r", "quats_r", "opacities_r")):
            raise RuntimeError("Stage5_6 renderable mask expects aligned render params and node_features.")
        def _all_finite_per_node(x: torch.Tensor) -> torch.Tensor:
            if int(x.shape[0]) != n:
                raise RuntimeError("Stage5_6 renderable finite-check expects first dim == num nodes.")
            return torch.isfinite(x.reshape(n, -1)).all(dim=1)

        finite_feat = _all_finite_per_node(node_features)
        finite_geo = (
            _all_finite_per_node(render_params["means_r"])
            & _all_finite_per_node(render_params["scales_r"])
            & _all_finite_per_node(render_params["quats_r"])
            & _all_finite_per_node(render_params["opacities_r"])
        )
        opacity_ok = render_params["opacities_r"].reshape(-1) > float(self.stage5_6_renderable_min_opacity)
        scale_ok = render_params["scales_r"].min(dim=1).values > float(self.stage5_6_renderable_min_scale)
        return (finite_feat & finite_geo & opacity_ok & scale_ok).to(device=self.device)

    def _build_feedback_node_pack_legacy(
        self,
        *,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Optional[Dict[str, Any]]:
        feat_bg = self._current_or_fused_feature(out, "bg", "_feat_2d_bg")
        acc_bg = out.get("_acc_w_bg")
        render_bg = out.get("_render_params_bg")
        if feat_bg is None or acc_bg is None or render_bg is None:
            return None
        feats: List[Optional[torch.Tensor]] = [feat_bg]
        masks: List[Optional[torch.Tensor]] = [acc_bg > float(self.bg_src_backproject_support_min)]

        rigid_world, rigid_order = self._build_rigid_world_for_aux_frame(out, int(target_frame_idx))
        node_state_rigid = out.get("_node_state_rigid")
        route = out.get("_route")
        rigid_count = int(rigid_order.numel())
        if rigid_world is not None and rigid_count > 0:
            feat_rigid_ordered = feat_bg.new_zeros((rigid_count, int(feat_bg.shape[1])))
            mask_rigid_ordered = torch.zeros((rigid_count,), dtype=torch.bool, device=self.device)
            if node_state_rigid is not None:
                if route is None:
                    raise RuntimeError("Stage5_6 feedback rigid pack requires _route to align rigid features.")
                route_s = route.S.to(device=self.device, dtype=torch.long)
                if int(route_s.numel()) > 0:
                    feat_rigid_s = self._current_or_fused_feature(out, "rigid_s", "_feat_2d_rigid_S")
                    acc_rigid_s = out.get("_acc_w_rigid_S")
                    if feat_rigid_s is None or acc_rigid_s is None:
                        raise RuntimeError(
                            "Stage5_6 feedback rigid pack requires source rigid features/support when route.S is non-empty."
                        )
                n_rigid = int(node_state_rigid.means.shape[0])
                lookup_s = torch.full((n_rigid,), -1, dtype=torch.long, device=self.device)
                lookup_s[route_s] = torch.arange(int(route_s.numel()), dtype=torch.long, device=self.device)
                rows = lookup_s[rigid_order]
                observed = rows >= 0
                if bool(observed.any().item()):
                    obs_rows = rows[observed]
                    if bool((obs_rows < 0).any().item()) or bool((obs_rows >= int(feat_rigid_s.shape[0])).any().item()) or bool(
                        (obs_rows >= int(acc_rigid_s.shape[0])).any().item()
                    ):
                        raise RuntimeError(
                            "Stage5_6 feedback rigid pack route-to-source index out of range: "
                            f"obs_rows_max={int(obs_rows.max().item())}, "
                            f"feat_rigid_s={int(feat_rigid_s.shape[0])}, acc_rigid_s={int(acc_rigid_s.shape[0])}."
                        )
                    observed_idx = torch.nonzero(observed, as_tuple=False).squeeze(1)
                    feat_rigid_ordered[observed_idx] = feat_rigid_s[obs_rows].to(
                        device=feat_rigid_ordered.device,
                        dtype=feat_rigid_ordered.dtype,
                    )
                    mask_rigid_ordered[observed_idx] = (
                        acc_rigid_s[obs_rows].to(device=self.device) > float(self.rigid_src_backproject_support_min)
                    )
            feats.append(feat_rigid_ordered)
            masks.append(mask_rigid_ordered)

        feat_distant = self._current_or_fused_feature(out, "distant", "_feat_2d_distant")
        acc_distant = out.get("_acc_w_distant")
        render_distant = out.get("_render_params_distant")
        distant_count = 0
        if feat_distant is not None and acc_distant is not None and render_distant is not None and int(feat_distant.shape[0]) > 0:
            distant_count = int(feat_distant.shape[0])
            feats.append(feat_distant)
            masks.append(acc_distant > float(self.distant_src_backproject_support_min))

        merged_features = self._merge_lists(feats)
        merged_mask = self._merge_lists(masks)
        if merged_features is None or merged_mask is None:
            return None
        merged_render = self._tensor_merge_bg_rigid_distant_world(render_bg, rigid_world, render_distant)
        if int(merged_render["means_r"].shape[0]) != int(merged_features.shape[0]):
            raise RuntimeError(
                "Stage5_6 feedback render/features length mismatch: "
                f"{merged_render['means_r'].shape[0]} vs {merged_features.shape[0]}."
            )
        num_bg = int(feat_bg.shape[0])
        return {
            "render": merged_render,
            "features": merged_features,
            "mask": merged_mask.bool(),
            "num_bg": num_bg,
            "num_rigid": rigid_count,
            "num_distant": distant_count,
            "rigid_order": rigid_order,
            "num_rigid_total": int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
            "allow_feature_grad": False,
        }

    def _build_feedback_node_pack(
        self,
        *,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Optional[Dict[str, Any]]:
        if self.stage5_6_error_input_feature == "post_update_node_feature":
            return self._build_feedback_node_pack_legacy(out=out, target_frame_idx=target_frame_idx)
        if self.stage5_6_error_input_feature == "post_struct":
            raise RuntimeError(
                "Stage5_6 P0 does not implement error_pred.input_feature='post_struct'. "
                "Use post_gru_hidden or post_update_node_feature."
            )
        error_feats = self._prepare_error_splat_features(out)
        feat_bg = error_feats.get("bg")
        render_bg = out.get("_render_params_bg")
        if feat_bg is None or render_bg is None:
            return None
        acc_bg = out.get("_acc_w_bg")

        rigid_world, rigid_order = self._build_rigid_world_for_aux_frame(out, int(target_frame_idx))
        node_state_rigid = out.get("_node_state_rigid")
        feats: List[Optional[torch.Tensor]] = [feat_bg]
        bg_renderable = self._renderable_node_mask(node_features=feat_bg, render_params=render_bg)
        if self.stage5_6_node_mask_policy == "source_support_threshold" and acc_bg is not None:
            bg_renderable = bg_renderable & (acc_bg > float(self.bg_src_backproject_support_min)).to(device=self.device)
        masks: List[Optional[torch.Tensor]] = [bg_renderable]

        rigid_count = int(rigid_order.numel())
        if rigid_world is not None and rigid_count > 0:
            feat_rigid_all = error_feats.get("rigid")
            if feat_rigid_all is None:
                raise RuntimeError(
                    "Stage5_6 post_gru_hidden requires _h_new_rigid -> _err_splat_feat_rigid when rigid branch is active."
                )
            rigid_order = rigid_order.to(device=self.device, dtype=torch.long)
            if bool((rigid_order < 0).any().item()) or bool((rigid_order >= int(feat_rigid_all.shape[0])).any().item()):
                raise RuntimeError("Stage5_6 rigid_order out of range for _err_splat_feat_rigid.")
            feat_rigid_ordered = feat_rigid_all[rigid_order]
            mask_rigid_ordered = self._renderable_node_mask(
                node_features=feat_rigid_ordered,
                render_params=rigid_world,
            )
            feats.append(feat_rigid_ordered)
            masks.append(mask_rigid_ordered)

        feat_distant = error_feats.get("distant")
        render_distant = out.get("_render_params_distant")
        acc_distant = out.get("_acc_w_distant")
        distant_count = 0
        if feat_distant is not None and render_distant is not None and int(feat_distant.shape[0]) > 0:
            distant_count = int(feat_distant.shape[0])
            feats.append(feat_distant)
            distant_renderable = self._renderable_node_mask(node_features=feat_distant, render_params=render_distant)
            if self.stage5_6_node_mask_policy == "source_support_threshold" and acc_distant is not None:
                distant_renderable = distant_renderable & (
                    acc_distant > float(self.distant_src_backproject_support_min)
                ).to(device=self.device)
            masks.append(distant_renderable)

        merged_features = self._merge_lists(feats)
        merged_mask = self._merge_lists(masks)
        if merged_features is None or merged_mask is None:
            return None
        merged_render = self._tensor_merge_bg_rigid_distant_world(render_bg, rigid_world, render_distant)
        if int(merged_render["means_r"].shape[0]) != int(merged_features.shape[0]):
            raise RuntimeError(
                "Stage5_6 feedback render/features length mismatch: "
                f"{merged_render['means_r'].shape[0]} vs {merged_features.shape[0]}."
            )
        num_bg = int(feat_bg.shape[0])
        return {
            "render": merged_render,
            "features": merged_features,
            "mask": merged_mask.bool(),
            "num_bg": num_bg,
            "num_rigid": rigid_count,
            "num_distant": distant_count,
            "rigid_order": rigid_order,
            "num_rigid_total": int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
            "allow_feature_grad": not bool(self.stage5_6_detach_projected_feature),
        }

    def _render_params_to_gaussians(self, render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rp = self._stage5_6_detach_render_params(render_params)
        return {
            "means": rp["means_r"],
            "scales": rp["scales_r"],
            "quats": rp["quats_r"],
            "opacities": rp["opacities_r"],
            "colors": rp["colors_r"],
        }

    def _make_feedback_branch_pack(
        self,
        values: torch.Tensor,
        support: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if int(values.shape[1]) != int(self.stage5_6_error_feat_dim) + 1:
            raise RuntimeError(
                "Stage5_6 lifted feedback dim mismatch: "
                f"got {values.shape[1]} expected {int(self.stage5_6_error_feat_dim) + 1}."
            )
        sup = support.reshape(-1, 1)
        pack = {
            "error": values[:, :1],
            "feat": values[:, 1:],
            "support": sup,
            "valid": (sup > float(self.stage5_6_lift_support_min)).to(dtype=values.dtype),
        }
        if bool(self.stage5_6_detach_lifted_feedback):
            pack = {k: v.detach() for k, v in pack.items()}
        return pack

    def _lift_feedback_maps_to_cache(
        self,
        *,
        node_pack: Dict[str, Any],
        views: List[Any],
        height: int,
        width: int,
        error_pred: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        render_params = node_pack["render"]
        if not render_params["means_r"].is_cuda:
            return None
        if latent.dim() != 4:
            raise RuntimeError(f"Stage5_6 latent must be [V,C,H,W], got {tuple(latent.shape)}.")
        if error_pred.dim() != 4 or int(error_pred.shape[1]) != 1:
            raise RuntimeError(f"Stage5_6 error_pred must be [V,1,H,W], got {tuple(error_pred.shape)}.")
        if mask.dim() != 3:
            raise RuntimeError(f"Stage5_6 mask must be [V,H,W], got {tuple(mask.shape)}.")
        if int(error_pred.shape[0]) != int(len(views)) or int(latent.shape[0]) != int(len(views)):
            raise RuntimeError("Stage5_6 lift expected matching view count across tensors and camera list.")
        with torch.no_grad():
            feedback_image = torch.cat([error_pred.detach(), latent.detach()], dim=1)
            feedback_image = feedback_image.permute(0, 2, 3, 1).contiguous()
            valid_mask = (mask.detach() > 0.0).to(device=feedback_image.device)
            lifted, support = self.alpha_t_extractor_v4.render_and_backproject_streaming_fused_multi_camera(
                gaussians=self._render_params_to_gaussians(render_params),
                cameras=views,
                features_2d=feedback_image,
                height=int(height),
                width=int(width),
                num_gaussians=int(render_params["means_r"].shape[0]),
                backprojector=self.feature_backprojector,
                source_pair_valid_mask=valid_mask,
                return_accumulated_weights=True,
            )
            lifted = lifted.to(device=self.device)
            support = support.to(device=self.device)
            num_bg = int(node_pack["num_bg"])
            num_rigid = int(node_pack["num_rigid"])
            num_distant = int(node_pack["num_distant"])
            start = 0
            bg_pack = self._make_feedback_branch_pack(lifted[start : start + num_bg], support[start : start + num_bg])
            start += num_bg
            rigid_pack = None
            if num_rigid > 0:
                rigid_values = lifted[start : start + num_rigid]
                rigid_support = support[start : start + num_rigid]
                start += num_rigid
                n_rigid_total = int(node_pack.get("num_rigid_total", 0))
                rigid_full = lifted.new_zeros((n_rigid_total, int(lifted.shape[1])))
                rigid_support_full = support.new_zeros((n_rigid_total,))
                rigid_order = node_pack["rigid_order"].to(device=self.device, dtype=torch.long)
                if int(rigid_order.numel()) > 0:
                    rigid_full[rigid_order] = rigid_values
                    rigid_support_full[rigid_order] = rigid_support
                rigid_pack = self._make_feedback_branch_pack(rigid_full, rigid_support_full)
            distant_pack = None
            if num_distant > 0:
                distant_pack = self._make_feedback_branch_pack(
                    lifted[start : start + num_distant],
                    support[start : start + num_distant],
                )
            branch_packs = {"bg": bg_pack, "distant": distant_pack, "rigid": rigid_pack}
            stats = self._feedback_cache_stats(branch_packs)
            return {**branch_packs, **stats}

    def _feedback_cache_stats(self, packs: Dict[str, Optional[Dict[str, torch.Tensor]]]) -> Dict[str, float]:
        valid_parts: List[torch.Tensor] = []
        support_parts: List[torch.Tensor] = []
        error_parts: List[torch.Tensor] = []
        for pack in packs.values():
            if not isinstance(pack, dict):
                continue
            valid_parts.append(pack["valid"].detach().float().reshape(-1))
            support_parts.append(pack["support"].detach().float().reshape(-1))
            error_parts.append(pack["error"].detach().float().reshape(-1))
        if len(valid_parts) == 0:
            return {
                "write_node_ratio": 0.0,
                "valid_ratio": 0.0,
                "support_mean": 0.0,
                "error_mean": 0.0,
            }
        valid_all = torch.cat(valid_parts)
        support_all = torch.cat(support_parts)
        error_all = torch.cat(error_parts)
        return {
            "write_node_ratio": float(valid_all.mean().item()),
            "valid_ratio": float(valid_all.mean().item()),
            "support_mean": float(support_all.mean().item()),
            "error_mean": float(error_all.mean().item()),
        }

    def _compute_error_pred_loss(self, batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
        zero = out["loss"].new_zeros(())
        if not self.stage5_6_error_enabled:
            return {"loss": zero, "processed": 0.0}
        collect_debug = bool(batch.get("_stage5_6_collect_debug_images", False))
        step = int(self._current_loss_step(batch))
        if step % max(int(self.stage5_6_target_every_n_steps), 1) != 0:
            return {"loss": zero, "processed": 0.0, "skipped_interval": 1.0}
        frame_groups = self._collect_role_frame_targets(
            batch,
            role=str(getattr(self, "stage5_6_error_target_role", "near_random")),
            max_frames=int(self.stage5_6_target_max_frames),
            require_aux_if_requested=bool(self.stage5_6_target_skip_if_no_valid_aux),
        )
        if len(frame_groups) == 0:
            return {"loss": zero, "processed": 0.0, "skipped_empty": 1.0}

        frame_losses: List[torch.Tensor] = []
        e_abs_terms: List[torch.Tensor] = []
        pred_terms: List[torch.Tensor] = []
        gt_terms: List[torch.Tensor] = []
        valid_terms: List[torch.Tensor] = []
        corr_terms: List[float] = []
        support_terms: List[torch.Tensor] = []
        debug_images: List[Dict[str, Any]] = []
        frame_cache_writes: List[Dict[str, Any]] = []
        processed_views = 0
        processed_frames = 0
        skipped_low_valid = 0

        for frame_idx, frame_targets in frame_groups:
            if len(frame_targets) == 0:
                continue
            node_pack = self._build_feedback_node_pack(out=out, target_frame_idx=int(frame_idx))
            if node_pack is None:
                continue
            first_gt = frame_targets[0].get("gt_image")
            if first_gt is None:
                continue
            if first_gt.dim() == 4:
                first_gt = first_gt.squeeze(0)
            h, w = int(first_gt.shape[0]), int(first_gt.shape[1])
            views = [t["view"] for t in frame_targets if t.get("view") is not None]
            if len(views) != len(frame_targets):
                continue
            feat_tilde_all, support_all = self._splat_node_features_to_views(
                render_params=node_pack["render"],
                # feedback latent is an adapter input for next-step struct features,
                # not a hidden-state residual. In post-GRU mode keep projector grads.
                node_features=(
                    node_pack["features"]
                    if bool(node_pack.get("allow_feature_grad", False))
                    else node_pack["features"].detach()
                ),
                node_mask=node_pack["mask"],
                views=views,
                height=h,
                width=w,
                detach_geometry=bool(self.stage5_6_detach_geometry),
                detach_weights=bool(self.stage5_6_detach_alpha_weights),
            )
            render_ctx_params = (
                self._stage5_6_detach_render_params(node_pack["render"])
                if bool(self.stage5_6_detach_render_context)
                else node_pack["render"]
            )
            pred_rgb_all: List[torch.Tensor] = []
            pred_alpha_all: List[torch.Tensor] = []
            e_gt_all: List[torch.Tensor] = []
            mask_all: List[torch.Tensor] = []
            valid_view_idx: List[int] = []
            local_debug_rows: List[Dict[str, Any]] = []
            for view_idx, t in enumerate(frame_targets):
                gt = t.get("gt_image")
                if gt is None:
                    continue
                if gt.dim() == 4:
                    gt = gt.squeeze(0)
                pred_rgb_ctx, pred_alpha_ctx = self._render_single_view(render_ctx_params, views[view_idx], h, w)
                if pred_alpha_ctx.dim() == 3 and int(pred_alpha_ctx.shape[-1]) == 1:
                    pred_alpha_ctx = pred_alpha_ctx.squeeze(-1)
                gt = gt.to(device=pred_rgb_ctx.device, dtype=pred_rgb_ctx.dtype)
                e_gt = (pred_rgb_ctx.detach() - gt).abs().mean(dim=-1)
                e_gt = e_gt.clamp(min=0.0, max=float(getattr(self.stage5_6_error_head, "error_max", 0.5)))
                mask = self._build_error_mask(t, h, w, pred_alpha_ctx).to(device=pred_rgb_ctx.device, dtype=pred_rgb_ctx.dtype)
                valid_ratio = mask.mean()
                if float(valid_ratio.detach().item()) < float(self.stage5_6_error_min_valid_pixel_ratio):
                    skipped_low_valid += 1
                    continue
                valid_view_idx.append(view_idx)
                pred_rgb_all.append(pred_rgb_ctx)
                pred_alpha_all.append(pred_alpha_ctx)
                e_gt_all.append(e_gt)
                mask_all.append(mask)
                local_debug_rows.append(
                    {
                        "target_index": int(t.get("target_index", view_idx)),
                        "frame_idx": int(frame_idx),
                        "cam_idx": int(t.get("cam_idx", -1)),
                        "role": str(t.get("role", self.stage5_6_error_target_role)),
                        "render": pred_rgb_ctx.detach().float().cpu(),
                        "actual_error": e_gt.detach().float().cpu(),
                    }
                )
            if len(valid_view_idx) == 0:
                continue
            feat_tilde = feat_tilde_all[valid_view_idx].permute(0, 3, 1, 2).contiguous()
            pred_rgb_ctx_v = torch.stack(pred_rgb_all, dim=0).permute(0, 3, 1, 2).contiguous()
            pred_alpha_ctx_v = torch.stack(pred_alpha_all, dim=0).unsqueeze(1).contiguous()
            head_inputs = [feat_tilde]
            if self.stage5_6_use_render_rgb:
                rgb_ctx = pred_rgb_ctx_v.detach() if bool(self.stage5_6_detach_render_context) else pred_rgb_ctx_v
                head_inputs.append(rgb_ctx)
            if self.stage5_6_use_render_alpha:
                alpha_ctx = pred_alpha_ctx_v.detach() if bool(self.stage5_6_detach_render_context) else pred_alpha_ctx_v
                head_inputs.append(alpha_ctx)
            head_in = torch.cat(head_inputs, dim=1)
            e_pred_raw, latent_raw = self.stage5_6_error_head(head_in)
            if e_pred_raw.dim() != 4 or latent_raw.dim() != 4:
                raise RuntimeError("Stage5_6 error head must output [V,1,H,W] and [V,C,H,W].")
            view_losses: List[torch.Tensor] = []
            for local_idx in range(int(e_pred_raw.shape[0])):
                e_pred = e_pred_raw[local_idx, 0]
                e_gt = e_gt_all[local_idx]
                mask = mask_all[local_idx]
                diff = e_pred - e_gt
                per_pixel = self._charbonnier(diff) if self.stage5_6_error_loss_type == "charbonnier" else diff.abs()
                denom = mask.sum().clamp_min(1.0)
                loss_i = (per_pixel * mask).sum() / denom
                view_losses.append(loss_i)
                e_abs_terms.append((diff.abs() * mask).sum().detach() / denom.detach())
                pred_terms.append((e_pred * mask).sum().detach() / denom.detach())
                gt_terms.append((e_gt * mask).sum().detach() / denom.detach())
                valid_terms.append(mask.mean().detach())
                support_map = support_all[valid_view_idx[local_idx]]
                support_terms.append((support_map.detach().clamp_min(0.0) * mask.detach()).sum() / denom.detach())
                with torch.no_grad():
                    x = e_pred[mask > 0.0].reshape(-1).float()
                    y = e_gt[mask > 0.0].reshape(-1).float()
                    if int(x.numel()) > 8:
                        vx = x - x.mean()
                        vy = y - y.mean()
                        den = (vx.norm() * vy.norm()).clamp_min(1.0e-8)
                        corr_terms.append(float((vx * vy).sum().item() / float(den.item())))
                if collect_debug:
                    dbg = dict(local_debug_rows[local_idx])
                    dbg["pred_error"] = e_pred.detach().float().cpu()
                    debug_images.append(dbg)
            if len(view_losses) == 0:
                continue
            frame_losses.append(torch.stack(view_losses).mean())
            processed_views += int(len(view_losses))
            processed_frames += 1

            if bool(self.stage5_6_cache_enable) and self._cache_ready(step):
                frame_mask = torch.stack(mask_all, dim=0)
                cache_write = self._lift_feedback_maps_to_cache(
                    node_pack=node_pack,
                    views=[views[i] for i in valid_view_idx],
                    height=h,
                    width=w,
                    error_pred=e_pred_raw,
                    latent=latent_raw,
                    mask=frame_mask,
                )
                if cache_write is not None:
                    cache_write["nearby_frame_idx"] = int(frame_idx)
                    frame_cache_writes.append(cache_write)

        if len(frame_losses) == 0:
            return {
                "loss": zero,
                "processed": 0.0,
                "processed_frames": 0.0,
                "skipped_low_valid": float(skipped_low_valid),
            }
        scale = self._warmup_linear_value(
            step,
            start_value=float(self.stage5_6_error_start_weight_scale),
            end_value=float(self.stage5_6_error_end_weight_scale),
            warmup_steps=int(self.stage5_6_error_warmup_steps),
        )
        out_pack: Dict[str, Any] = {
            "loss": float(self.stage5_6_error_weight) * float(scale) * torch.stack(frame_losses).mean(),
            "loss_raw": torch.stack(frame_losses).mean(),
            "processed": float(processed_views),
            "processed_frames": float(processed_frames),
            "effective_weight": float(self.stage5_6_error_weight) * float(scale),
            "e_abs": torch.stack(e_abs_terms).mean(),
            "u_pred_mean": torch.stack(pred_terms).mean(),
            "u_gt_mean": torch.stack(gt_terms).mean(),
            "valid_pixel_ratio": torch.stack(valid_terms).mean(),
            "support_mean": torch.stack(support_terms).mean(),
            "e_corr": float(sum(corr_terms) / max(len(corr_terms), 1)),
            "skipped_low_valid": float(skipped_low_valid),
            **({"debug_images": debug_images} if collect_debug else {}),
        }
        if len(frame_cache_writes) > 0:
            out_pack["cache_write"] = frame_cache_writes
        return out_pack

    def _write_cache(self, batch: Dict[str, Any], out: Dict[str, Any], error_pack: Dict[str, Any]) -> None:
        if not self.stage5_6_cache_enable:
            return
        step = int(self._current_loss_step(batch))
        if not self._cache_ready(step):
            return
        cache_write = error_pack.get("cache_write")
        if not isinstance(cache_write, list):
            return
        scope_key = self._stage5_6_scope_key(batch)
        if bool(getattr(self, "stage5_6_cache_keep_only_current_scope", True)):
            current_bank = self._stage5_6_frame_cache.get(scope_key)
            self._stage5_6_frame_cache.clear()
            if current_bank is not None:
                self._stage5_6_frame_cache[scope_key] = current_bank
        bank = self._stage5_6_frame_cache.setdefault(scope_key, {})
        for frame_pack in cache_write:
            if not isinstance(frame_pack, dict):
                continue
            nearby_frame_idx = frame_pack.get("nearby_frame_idx")
            if nearby_frame_idx is None:
                continue
            bank[int(nearby_frame_idx)] = {
                "step": int(step),
                "scene_id": int(scope_key[0]),
                "segment_id": int(scope_key[1]),
                "episode_idx": int(scope_key[2]),
                "block_idx_global": int(scope_key[3]),
                "source_frame_idx": self._safe_int(batch.get("source_frame_idx", None)),
                "nearby_frame_idx": int(nearby_frame_idx),
                "bg": frame_pack.get("bg"),
                "distant": frame_pack.get("distant"),
                "rigid": frame_pack.get("rigid"),
                "write_node_ratio": float(frame_pack.get("write_node_ratio", 0.0)),
                "valid_ratio": float(frame_pack.get("valid_ratio", 0.0)),
                "support_mean": float(frame_pack.get("support_mean", 0.0)),
                "error_mean": float(frame_pack.get("error_mean", 0.0)),
            }

    def _read_feedback_pack(
        self,
        pack: Optional[Dict[str, torch.Tensor]],
        *,
        dtype: torch.dtype,
        indices: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(pack, dict):
            return None
        out: Dict[str, torch.Tensor] = {}
        for k in ("error", "feat", "support", "valid"):
            val = pack.get(k)
            if not torch.is_tensor(val):
                return None
            if indices is not None:
                val = val[indices]
            out[k] = val.to(device=self.device, dtype=dtype)
        return out

    def _read_frame_entry(
        self,
        entry: Optional[Dict[str, Any]],
        *,
        route: RigidRoute,
        dtype: torch.dtype,
        current_step: int,
    ) -> Optional[Dict[str, Optional[Dict[str, torch.Tensor]]]]:
        if entry is None:
            return None
        step = int(entry.get("step", -1))
        if not self._cache_ready(step):
            return None
        elapsed = int(current_step - step)
        if elapsed > int(self.stage5_6_cache_max_age):
            return None
        rigid_s = None
        if entry.get("rigid") is not None and int(route.S.numel()) > 0:
            rigid_s = self._read_feedback_pack(
                entry.get("rigid"),
                dtype=dtype,
                indices=route.S.to(device=self.device, dtype=torch.long),
            )
        return {
            "bg": self._read_feedback_pack(entry.get("bg"), dtype=dtype),
            "distant": self._read_feedback_pack(entry.get("distant"), dtype=dtype),
            "rigid_s": rigid_s,
        }

    def _read_cache(
        self,
        scope_key: Tuple[int, int, int, int],
        route: RigidRoute,
        dtype: torch.dtype,
        current_step: int,
        frame_indices: List[int],
    ) -> Optional[Dict[str, List[Optional[Dict[str, torch.Tensor]]]]]:
        if self._fusion_scale(current_step) <= 0.0:
            return None
        if bool(getattr(self, "stage5_6_cache_keep_only_current_scope", True)):
            current_bank = self._stage5_6_frame_cache.get(scope_key)
            self._stage5_6_frame_cache.clear()
            if current_bank is not None:
                self._stage5_6_frame_cache[scope_key] = current_bank
        bank = self._stage5_6_frame_cache.get(scope_key)
        if bank is None:
            return None
        slots = max(int(self.stage5_6_fusion_num_slots), 1)
        frame_ids = [int(x) for x in frame_indices[:slots]]
        if len(frame_ids) < slots:
            frame_ids.extend([-1] * (slots - len(frame_ids)))
        cache_slots: Dict[str, List[Optional[Dict[str, torch.Tensor]]]] = {"bg": [], "distant": [], "rigid_s": []}
        for frame_idx in frame_ids:
            frame_entry = bank.get(int(frame_idx)) if int(frame_idx) >= 0 else None
            frame_pack = self._read_frame_entry(
                frame_entry,
                route=route,
                dtype=dtype,
                current_step=current_step,
            )
            cache_slots["bg"].append(frame_pack.get("bg") if isinstance(frame_pack, dict) else None)
            cache_slots["distant"].append(frame_pack.get("distant") if isinstance(frame_pack, dict) else None)
            cache_slots["rigid_s"].append(frame_pack.get("rigid_s") if isinstance(frame_pack, dict) else None)
        return cache_slots

    def _stack_feedback_slots(
        self,
        feedback_slots: Optional[List[Optional[Dict[str, torch.Tensor]]]],
        *,
        n_points: int,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(feedback_slots, list):
            return None
        k = max(int(self.stage5_6_fusion_num_slots), 1)
        slots = feedback_slots[:k]
        if len(slots) < k:
            slots.extend([None] * (k - len(slots)))
        feat_slots: List[torch.Tensor] = []
        error_slots: List[torch.Tensor] = []
        support_slots: List[torch.Tensor] = []
        valid_slots: List[torch.Tensor] = []
        for slot in slots:
            if not isinstance(slot, dict):
                feat_slots.append(torch.zeros((n_points, int(self.stage5_6_error_feat_dim)), device=self.device, dtype=dtype))
                error_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                support_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                valid_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                continue
            feat = slot.get("feat")
            error = slot.get("error")
            support = slot.get("support")
            valid = slot.get("valid")
            if not all(torch.is_tensor(x) for x in (feat, error, support, valid)):
                feat_slots.append(torch.zeros((n_points, int(self.stage5_6_error_feat_dim)), device=self.device, dtype=dtype))
                error_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                support_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                valid_slots.append(torch.zeros((n_points, 1), device=self.device, dtype=dtype))
                continue
            if int(feat.shape[0]) != int(n_points):
                return None
            feat_slots.append(feat.to(device=self.device, dtype=dtype))
            error_slots.append(error.to(device=self.device, dtype=dtype))
            support_slots.append(support.to(device=self.device, dtype=dtype))
            valid_slots.append(valid.to(device=self.device, dtype=dtype))
        return {
            "feat": torch.stack(feat_slots, dim=1),
            "error": torch.stack(error_slots, dim=1),
            "support": torch.stack(support_slots, dim=1),
            "valid": torch.stack(valid_slots, dim=1),
        }

    def _fuse(
        self,
        feat: Optional[torch.Tensor],
        feedback_slots: Optional[List[Optional[Dict[str, torch.Tensor]]]],
        fuser: Stage5_6FrameFlattenFuser,
        *,
        current_support: Optional[torch.Tensor],
        branch_name: str,
    ) -> Optional[torch.Tensor]:
        if feat is None:
            return feat
        feedback = self._stack_feedback_slots(
            feedback_slots,
            n_points=int(feat.shape[0]),
            dtype=feat.dtype,
        )
        if feedback is None:
            return feat
        scale = float(self._stage5_6_active_fusion_scale)
        fused = fuser(feat, feedback, current_support=current_support, scale=scale)
        with torch.no_grad():
            delta_norm = torch.norm((fused - feat).detach(), dim=-1).mean() if int(feat.shape[0]) > 0 else feat.new_zeros(())
            self._stage5_6_fusion_delta_norm_terms.append(delta_norm)
        self._stage5_6_last_fused_features[branch_name] = fused
        return fused

    def _compute_full_routed_gru_inputs(self, **kwargs):
        self._stage5_6_active_cache = None
        self._stage5_6_active_fusion_scale = 0.0
        self._stage5_6_fusion_delta_norm_terms = []
        self._stage5_6_last_fused_features = {}
        batch = kwargs.get("batch")
        route = kwargs.get("route")
        feat_2d_bg = kwargs.get("feat_2d_bg")
        if isinstance(batch, dict) and route is not None and feat_2d_bg is not None:
            step = int(self._current_loss_step(batch))
            self._stage5_6_active_fusion_scale = float(self._fusion_scale(step))
            if bool(self.stage5_6_cache_enable) and self._stage5_6_active_fusion_scale > 0.0:
                frame_indices = self._feedback_frame_indices_for_fusion(batch)
                self._stage5_6_active_cache = self._read_cache(
                    self._stage5_6_scope_key(batch),
                    route,
                    feat_2d_bg.dtype,
                    current_step=step,
                    frame_indices=frame_indices,
                )
        try:
            return super()._compute_full_routed_gru_inputs(**kwargs)
        finally:
            self._stage5_6_active_cache = None
            self._stage5_6_active_fusion_scale = 0.0

    def _build_struct_decoder_input_near(self, **kwargs):
        cache = self._stage5_6_active_cache
        if cache is not None:
            if bool(self.stage5_6_fusion_apply_to_bg):
                kwargs["feat_2d_bg"] = self._fuse(
                    kwargs.get("feat_2d_bg"),
                    cache.get("bg"),
                    self.stage5_6_bg_fuser,
                    current_support=kwargs.get("acc_w_bg"),
                    branch_name="bg",
                )
            if bool(self.stage5_6_fusion_apply_to_rigid):
                kwargs["feat_2d_rigid_S"] = self._fuse(
                    kwargs.get("feat_2d_rigid_S"),
                    cache.get("rigid_s"),
                    self.stage5_6_rigid_fuser,
                    current_support=kwargs.get("acc_w_rigid_S"),
                    branch_name="rigid_s",
                )
        return super()._build_struct_decoder_input_near(**kwargs)

    def _build_struct_decoder_input_far(self, **kwargs):
        cache = self._stage5_6_active_cache
        if cache is not None:
            if bool(self.stage5_6_fusion_apply_to_distant):
                kwargs["feat_2d_distant"] = self._fuse(
                    kwargs.get("feat_2d_distant"),
                    cache.get("distant"),
                    self.stage5_6_distant_fuser,
                    current_support=kwargs.get("acc_w_distant"),
                    branch_name="distant",
                )
            if bool(self.stage5_6_fusion_apply_to_rigid):
                kwargs["feat_2d_rigid_S"] = self._fuse(
                    kwargs.get("feat_2d_rigid_S"),
                    cache.get("rigid_s"),
                    self.stage5_6_rigid_fuser,
                    current_support=kwargs.get("acc_w_rigid_S"),
                    branch_name="rigid_s",
                )
        return super()._build_struct_decoder_input_far(**kwargs)

    def _log_nearby_pack(self, out: Dict[str, Any], nearby: Dict[str, Any]) -> None:
        loss_val = nearby.get("loss")
        out["loss_stage5_6_nearby_direct"] = float(loss_val.detach().item()) if torch.is_tensor(loss_val) else 0.0
        out["loss/nearby_direct"] = float(out["loss_stage5_6_nearby_direct"])
        out["loss/target_weight/nearby_direct"] = float(nearby.get("view_weight", 0.0))
        out["monitor/nearby_direct/processed_targets"] = float(nearby.get("processed", 0.0))
        for src_key, out_key in (
            ("monitor_psnr", "monitor/psnr/nearby_direct"),
            ("monitor_l1", "monitor/l1/nearby_direct"),
            ("valid_mask_ratio", "monitor/nearby_direct/valid_mask_ratio"),
        ):
            val = nearby.get(src_key)
            if torch.is_tensor(val):
                out[out_key] = float(val.detach().item())
            elif isinstance(val, (int, float)):
                out[out_key] = float(val)
        if nearby.get("skipped_empty"):
            out["monitor/nearby_direct/skipped_empty_aux_list"] = 1.0
        if nearby.get("skipped_zero_weight"):
            out["monitor/nearby_direct/skipped_zero_weight"] = 1.0
        if nearby.get("skipped_low_valid"):
            out["monitor/nearby_direct/skipped_low_valid"] = float(nearby.get("skipped_low_valid", 0.0))

    def _log_error_pack(self, out: Dict[str, Any], err: Dict[str, Any]) -> None:
        loss_val = err.get("loss")
        out["loss_stage5_6_error_pred"] = float(loss_val.detach().item()) if torch.is_tensor(loss_val) else 0.0
        out["loss/error_pred"] = float(out["loss_stage5_6_error_pred"])
        out["error_pred/processed_targets"] = float(err.get("processed", 0.0))
        out["error_pred/processed_frames"] = float(err.get("processed_frames", 0.0))
        out["error_pred/effective_weight"] = float(err.get("effective_weight", 0.0))
        out["monitor/stage5_6/error_pred_processed_targets"] = float(err.get("processed", 0.0))
        out["monitor/stage5_6/error_pred_processed_frames"] = float(err.get("processed_frames", 0.0))
        for key in ("e_abs", "u_pred_mean", "u_gt_mean", "valid_pixel_ratio", "support_mean"):
            val = err.get(key)
            if torch.is_tensor(val):
                out[f"error_pred/{key}"] = float(val.detach().item())
            elif isinstance(val, (int, float)):
                out[f"error_pred/{key}"] = float(val)
        out["error_pred/e_corr"] = float(err.get("e_corr", 0.0))
        if err.get("skipped_interval"):
            out["error_pred/skipped_interval"] = 1.0
        if err.get("skipped_empty"):
            out["error_pred/skipped_empty_aux_list"] = 1.0
        if err.get("skipped_low_valid"):
            out["error_pred/skipped_low_valid"] = float(err.get("skipped_low_valid", 0.0))

    def _log_feedback_state(self, out: Dict[str, Any], step: int, err: Dict[str, Any]) -> None:
        cache_write = err.get("cache_write") if isinstance(err, dict) else None
        scope_key = out.get("_stage5_6_scope_key")
        frame_bank = self._stage5_6_frame_cache.get(scope_key) if scope_key is not None else None
        entry = None
        if isinstance(cache_write, list) and len(cache_write) > 0:
            latest_frame = cache_write[-1].get("nearby_frame_idx")
            if isinstance(frame_bank, dict) and latest_frame is not None:
                entry = frame_bank.get(int(latest_frame))
        if entry is None and isinstance(frame_bank, dict) and len(frame_bank) > 0:
            newest = sorted(frame_bank.values(), key=lambda x: int(x.get("step", -1)))
            entry = newest[-1]
        fusion_delta = 0.0
        if len(self._stage5_6_fusion_delta_norm_terms) > 0:
            fusion_delta = float(torch.stack(self._stage5_6_fusion_delta_norm_terms).mean().detach().item())
        age_mean = 0.0
        if isinstance(entry, dict):
            age_mean = float(max(int(step) - int(entry.get("step", step)), 0))
        out["feedback/fusion_enabled"] = float(
            1.0 if bool(self.stage5_6_cache_enable) and bool(self.stage5_6_fusion_enabled) else 0.0
        )
        out["feedback/fusion_scale"] = float(self._fusion_scale(step))
        out["feedback/fusion_delta_norm"] = float(fusion_delta)
        out["feedback/cache_size"] = float(len(self._stage5_6_frame_cache))
        out["feedback/frame_cache_size"] = float(len(frame_bank) if isinstance(frame_bank, dict) else 0)
        out["feedback/age_mean"] = float(age_mean)
        for key in ("write_node_ratio", "valid_ratio", "support_mean", "error_mean"):
            val = 0.0
            if isinstance(cache_write, list) and len(cache_write) > 0:
                val = float(cache_write[-1].get(key, 0.0))
            elif isinstance(entry, dict):
                val = float(entry.get(key, 0.0))
            out[f"feedback/{key}"] = float(val)
        for branch in ("bg", "distant", "rigid"):
            ratio = 0.0
            pack = entry.get(branch) if isinstance(entry, dict) else None
            if isinstance(pack, dict) and torch.is_tensor(pack.get("valid")) and int(pack["valid"].numel()) > 0:
                ratio = float(pack["valid"].detach().float().mean().item())
            out[f"branch/{branch}_feedback_valid"] = float(ratio)
        out["monitor/stage5_6/cache_size"] = float(len(self._stage5_6_frame_cache))

    def forward(self, batch: Dict) -> Dict[str, Any]:
        self._stage5_6_last_nearby_debug_images = []
        self._stage5_6_last_error_debug_images = []
        out = super().forward(batch)
        out["_stage5_6_scope_key"] = self._stage5_6_scope_key(batch)
        if self.training and self.stage5_6_nearby_enabled:
            nearby = self._compute_nearby_direct_loss(batch, out)
            if torch.is_tensor(nearby.get("loss")):
                out["loss"] = out["loss"] + nearby["loss"]
                if bool(nearby["loss"].requires_grad) and out.get("proxies") is not None:
                    out["_retain_graph_for_proxy_backward"] = True
            self._log_nearby_pack(out, nearby)
            if isinstance(nearby.get("debug_images"), list):
                self._stage5_6_last_nearby_debug_images = nearby["debug_images"]
                out["_stage5_6_nearby_debug_images"] = nearby["debug_images"]

        err = {"loss": out["loss"].new_zeros(()), "processed": 0.0}
        if self.training and self.stage5_6_error_enabled:
            err = self._compute_error_pred_loss(batch, out)
            if torch.is_tensor(err.get("loss")):
                out["loss"] = out["loss"] + err["loss"]
            self._log_error_pack(out, err)
            if isinstance(err.get("debug_images"), list):
                self._stage5_6_last_error_debug_images = err["debug_images"]
                out["_stage5_6_error_debug_images"] = err["debug_images"]
        self._write_cache(batch, out, err)
        self._log_feedback_state(out, int(self._current_loss_step(batch)), err)
        return out

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._stage5_6_last_nearby_debug_images = []
        self._stage5_6_last_error_debug_images = []
        out = super().train_step(
            batch=batch,
            step=step,
            profile_phase_timing=profile_phase_timing,
            sync_cuda_timing=sync_cuda_timing,
            scheduler_node_sync=scheduler_node_sync,
        )
        nearby_debug = getattr(self, "_stage5_6_last_nearby_debug_images", [])
        if isinstance(nearby_debug, list) and len(nearby_debug) > 0:
            out["_stage5_6_nearby_debug_images"] = nearby_debug
        error_debug = getattr(self, "_stage5_6_last_error_debug_images", [])
        if isinstance(error_debug, list) and len(error_debug) > 0:
            out["_stage5_6_error_debug_images"] = error_debug
        return out

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self._stage5_6_frame_cache.clear()
        self._stage5_6_active_cache = None
        self._stage5_6_last_fused_features = {}

    @torch.no_grad()
    def record_block_history(self, batch: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        stats = super().record_block_history(batch=batch, event=event)
        self._stage5_6_frame_cache.clear()
        self._stage5_6_active_cache = None
        self._stage5_6_last_fused_features = {}
        return stats


__all__ = ["MinimalStreetForwardStage5_6", "Stage5_6ErrorPredictHead", "Stage5_6FrameFlattenFuser"]
