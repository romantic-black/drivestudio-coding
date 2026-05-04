from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.streetforward.math_utils import get_viewmat
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import _merge_params_bg_rigid_distant
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4


class FeatureSplatUncertaintyHeadV3(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_dim: int = 64,
        error_max: float = 0.5,
        residual_max: float = 0.5,
        predict_rgb_residual: bool = False,
    ):
        super().__init__()
        if hidden_dim % 8 != 0:
            raise ValueError(
                f"FeatureSplatUncertaintyHeadV3 requires hidden_dim % 8 == 0, got {hidden_dim}."
            )
        self.error_max = float(error_max)
        self.residual_max = float(residual_max)
        self.predict_rgb_residual = bool(predict_rgb_residual)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.err_head = nn.Conv2d(hidden_dim, 1, 1)
        self.rgb_residual_head = nn.Conv2d(hidden_dim, 3, 1) if self.predict_rgb_residual else None

    def forward(self, x: torch.Tensor) -> Any:
        h = self.in_proj(x)
        h = h + self.blocks(h)
        e_pred = self.error_max * torch.sigmoid(self.err_head(h))
        if self.rgb_residual_head is None:
            return e_pred
        r_pred = self.residual_max * torch.tanh(self.rgb_residual_head(h))
        return e_pred, r_pred


FeatureSplatUncertaintyHeadV2 = FeatureSplatUncertaintyHeadV3


class MinimalStreetForwardStage5_5(MinimalStreetForwardStage5_4):
    def __init__(self, config, device: torch.device, **kwargs):
        self._stage5_5_last_scalar_logs: Dict[str, float] = {}
        super().__init__(config=config, device=device, **kwargs)
        self._debug_check_stage5_5_optimizer_contains_aux_head()

    def _init_stage5_3_modules(self, config) -> None:
        super()._init_stage5_3_modules(config)
        self._init_stage5_5_nearby_direct(config)
        self._init_stage5_5_feature_splat_uncertainty(config)

    def _parse_target_view_weight_cfg(self, config) -> Dict[str, Any]:
        cfg = super()._parse_target_view_weight_cfg(config)
        losses_cfg = config.get("losses", {}) if hasattr(config, "get") else {}
        tvw_cfg = losses_cfg.get("target_view_weights", {}) if hasattr(losses_cfg, "get") else {}
        nearby_tvw_cfg = tvw_cfg.get("nearby_direct", {}) if hasattr(tvw_cfg, "get") else {}
        nearby_cfg = config.get("nearby_direct", {}) if hasattr(config, "get") else {}
        cfg["nearby_direct_weight"] = float(
            nearby_tvw_cfg.get("weight", nearby_cfg.get("weight", 0.7))
            if hasattr(nearby_tvw_cfg, "get")
            else nearby_cfg.get("weight", 0.7)
        )
        return cfg

    def _target_role_weight(self, role: str, step: int) -> float:
        if str(role) == "nearby_direct":
            return float(self._target_view_weight_cfg.get("nearby_direct_weight", 0.7))
        return float(super()._target_role_weight(role, step))

    def _validate_stage5_3_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model")).strip().lower()
        if stage != "5_5":
            raise ValueError("Stage5_5 requires model.stage='5_5'.")
        old_stage = model_cfg.get("stage")
        model_cfg["stage"] = "5_4"
        try:
            super()._validate_stage5_3_config(config)
        finally:
            model_cfg["stage"] = old_stage

    def _init_stage5_5_feature_splat_uncertainty(self, config) -> None:
        cfg = config.get("feature_splat_uncertainty", {})
        self.stage5_5_aux_enabled = bool(cfg.get("enable", False))
        self.stage5_5_aux_mode = str(cfg.get("mode", "predict_l1_residual")).strip().lower()
        if self.stage5_5_aux_enabled and self.stage5_5_aux_mode != "predict_l1_residual":
            raise ValueError("Stage5_5 only supports feature_splat_uncertainty.mode='predict_l1_residual'.")

        splat_cfg = cfg.get("splat", {})
        target_cfg = cfg.get("target", {})
        head_cfg = cfg.get("head", {})
        loss_cfg = cfg.get("loss", {})
        debug_cfg = cfg.get("debug", {})
        bridge_cfg = cfg.get("bridge", {})

        self.stage5_5_detach_geometry = bool(splat_cfg.get("detach_geometry", True))
        self.stage5_5_detach_alpha_weights = bool(splat_cfg.get("detach_alpha_weights", True))
        self.stage5_5_detach_render_context = bool(splat_cfg.get("detach_render_context", True))
        self.stage5_5_splat_eps = float(splat_cfg.get("eps", 1.0e-6))
        self.stage5_5_support_min_for_extra_loss = float(splat_cfg.get("support_min_for_extra_loss", 1.0e-4))
        self.stage5_5_zero_invalid_input = bool(head_cfg.get("zero_invalid_input", True))
        self.stage5_5_concat_log_support = bool(head_cfg.get("concat_log_support", True))
        self.stage5_5_concat_valid_mask = bool(head_cfg.get("concat_valid_mask", True))
        self.stage5_5_use_render_rgb = bool(head_cfg.get("use_render_rgb", True))
        self.stage5_5_use_render_alpha = bool(head_cfg.get("use_render_alpha", True))
        self.stage5_5_render_context_dropout = float(head_cfg.get("render_context_dropout", 0.0))
        self.stage5_5_render_context_dropout = float(max(0.0, min(1.0, self.stage5_5_render_context_dropout)))
        self.stage5_5_error_max = float(head_cfg.get("error_max", 0.5))
        self.stage5_5_residual_max = float(head_cfg.get("residual_max", 0.5))
        self.stage5_5_predict_rgb_residual = bool(head_cfg.get("predict_rgb_residual", False))
        self.stage5_5_loss_all_weight = float(loss_cfg.get("all_valid_weight", 0.03))
        self.stage5_5_loss_supported_weight = float(loss_cfg.get("supported_region_weight", 0.05))
        self.stage5_5_loss_rgb_residual_weight = float(loss_cfg.get("rgb_residual_weight", 0.0))
        self.stage5_5_loss_rgb_residual_supported_weight = float(
            loss_cfg.get("rgb_residual_supported_weight", 0.0)
        )
        self.stage5_5_loss_mask_sky = bool(loss_cfg.get("mask_sky", True))
        self.stage5_5_loss_mask_egocar = bool(loss_cfg.get("mask_egocar", True))
        self.stage5_5_min_valid_pixel_ratio = float(loss_cfg.get("min_valid_pixel_ratio", 0.03))
        self.stage5_5_skip_empty_supported_loss = bool(loss_cfg.get("skip_empty_supported_loss", True))
        self.stage5_5_warmup_steps = int(loss_cfg.get("warmup_steps", 3000))
        self.stage5_5_start_weight_scale = float(loss_cfg.get("start_weight_scale", 0.0))
        self.stage5_5_end_weight_scale = float(loss_cfg.get("end_weight_scale", 1.0))
        self.stage5_5_target_every_n_steps = int(target_cfg.get("every_n_steps", 1))
        self.stage5_5_target_skip_if_no_valid_aux = bool(target_cfg.get("skip_if_no_valid_aux", True))
        self.stage5_5_target_max_targets = int(target_cfg.get("max_targets_per_step", 1))
        self.stage5_5_no_render_probe = bool(debug_cfg.get("no_render_probe", True))
        self.stage5_5_no_render_probe_interval = int(debug_cfg.get("no_render_probe_interval", 500))
        self.stage5_5_perf_sync_cuda = bool(debug_cfg.get("perf_sync_cuda", False))

        bridge_error_cfg = bridge_cfg.get("error_confidence", {})
        bridge_support_cfg = bridge_cfg.get("support_confidence", {})
        bridge_mask_cfg = bridge_cfg.get("mask", {})
        bridge_loss_cfg = bridge_cfg.get("loss", {})
        bridge_grad_cfg = bridge_cfg.get("grad", {})
        bridge_debug_cfg = bridge_cfg.get("debug", {})
        self.stage5_5_bridge_enabled = bool(bridge_cfg.get("enable", False))
        self.stage5_5_bridge_start_after_steps = int(bridge_cfg.get("start_after_steps", 10000))
        self.stage5_5_bridge_warmup_steps = int(bridge_cfg.get("warmup_steps", 5000))
        self.stage5_5_bridge_weight = float(bridge_cfg.get("weight", 0.005))
        self.stage5_5_bridge_max_weight = float(bridge_cfg.get("max_weight", self.stage5_5_bridge_weight))
        self.stage5_5_bridge_error_mode = str(bridge_error_cfg.get("mode", "exp")).strip().lower()
        self.stage5_5_bridge_error_tau = float(bridge_error_cfg.get("tau", 0.15))
        self.stage5_5_bridge_error_min_conf = float(bridge_error_cfg.get("min_conf", 0.0))
        self.stage5_5_bridge_error_max_conf = float(bridge_error_cfg.get("max_conf", 1.0))
        self.stage5_5_bridge_support_mode = str(bridge_support_cfg.get("mode", "soft")).strip().lower()
        self.stage5_5_bridge_support_tau = float(bridge_support_cfg.get("tau", 1.0e-4))
        self.stage5_5_bridge_support_gamma = float(bridge_support_cfg.get("gamma", 0.5))
        self.stage5_5_bridge_support_hard_min = float(bridge_support_cfg.get("hard_min", 1.0e-5))
        self.stage5_5_bridge_mask_use_valid_loss_mask = bool(bridge_mask_cfg.get("use_valid_loss_mask", True))
        self.stage5_5_bridge_mask_sky = bool(bridge_mask_cfg.get("mask_sky", True))
        self.stage5_5_bridge_mask_egocar = bool(bridge_mask_cfg.get("mask_egocar", True))
        self.stage5_5_bridge_mask_dynamic = bool(bridge_mask_cfg.get("mask_dynamic", False))
        self.stage5_5_bridge_require_render_alpha = bool(bridge_mask_cfg.get("require_render_alpha", True))
        self.stage5_5_bridge_render_alpha_min = float(bridge_mask_cfg.get("render_alpha_min", 0.02))
        self.stage5_5_bridge_min_effective_pixel_ratio = float(bridge_mask_cfg.get("min_effective_pixel_ratio", 0.005))
        self.stage5_5_bridge_loss_type = str(bridge_loss_cfg.get("type", "l1")).strip().lower()
        self.stage5_5_bridge_normalize_by_weight_sum = bool(bridge_loss_cfg.get("normalize_by_weight_sum", True))
        self.stage5_5_bridge_detach_confidence = bool(bridge_loss_cfg.get("detach_confidence", True))
        self.stage5_5_bridge_detach_mask = bool(bridge_loss_cfg.get("detach_mask", True))
        self.stage5_5_bridge_rgb_reduce = str(bridge_loss_cfg.get("rgb_reduce", "mean")).strip().lower()
        self.stage5_5_bridge_stopgrad_uncertainty_head = bool(
            bridge_grad_cfg.get("stopgrad_uncertainty_head_from_bridge", True)
        )
        self.stage5_5_bridge_debug_log_interval = int(bridge_debug_cfg.get("log_interval", 100))
        self.stage5_5_bridge_debug_save_maps_interval = int(bridge_debug_cfg.get("save_maps_interval", 0))
        if self.stage5_5_bridge_error_mode not in {"exp", "linear", "sigmoid"}:
            raise ValueError(
                "feature_splat_uncertainty.bridge.error_confidence.mode must be one of "
                "['exp', 'linear', 'sigmoid']."
            )
        if self.stage5_5_bridge_support_mode not in {"soft", "hard", "none"}:
            raise ValueError(
                "feature_splat_uncertainty.bridge.support_confidence.mode must be one of "
                "['soft', 'hard', 'none']."
            )
        if self.stage5_5_bridge_loss_type not in {"l1", "charbonnier"}:
            raise ValueError("feature_splat_uncertainty.bridge.loss.type must be one of ['l1', 'charbonnier'].")
        if self.stage5_5_bridge_rgb_reduce != "mean":
            raise ValueError("feature_splat_uncertainty.bridge.loss.rgb_reduce currently only supports 'mean'.")
        for name in ("allow_geometry_grad", "allow_color_grad", "allow_feature_extractor_grad"):
            if not bool(bridge_grad_cfg.get(name, True)):
                raise ValueError(
                    f"feature_splat_uncertainty.bridge.grad.{name}=false is not supported in Stage5_5 bridge v1."
                )
        if not self.stage5_5_bridge_stopgrad_uncertainty_head:
            raise ValueError(
                "feature_splat_uncertainty.bridge.grad.stopgrad_uncertainty_head_from_bridge=false "
                "is not supported in Stage5_5 bridge v1."
            )

        src_support_cfg = splat_cfg.get("src_support_min", {})
        self.stage5_5_src_support_min_bg = float(src_support_cfg.get("bg", self.bg_src_backproject_support_min))
        self.stage5_5_src_support_min_distant = float(src_support_cfg.get("distant", self.distant_src_backproject_support_min))
        self.stage5_5_src_support_min_rigid = float(src_support_cfg.get("rigid", self.rigid_src_backproject_support_min))

        in_ch = int(self.stage5_2_feat_2d_channels) + 1
        if self.stage5_5_concat_log_support:
            in_ch += 1
        if self.stage5_5_concat_valid_mask:
            in_ch += 1
        if self.stage5_5_use_render_rgb:
            in_ch += 3
        if self.stage5_5_use_render_alpha:
            in_ch += 1
        hidden_dim = int(head_cfg.get("hidden_dim", 64))
        if not self.stage5_5_predict_rgb_residual and (
            self.stage5_5_loss_rgb_residual_weight > 0.0
            or self.stage5_5_loss_rgb_residual_supported_weight > 0.0
        ):
            raise ValueError(
                "Stage5_5 rgb residual loss weights require "
                "feature_splat_uncertainty.head.predict_rgb_residual=true."
            )
        self.stage5_5_uncertainty_head = FeatureSplatUncertaintyHeadV3(
            in_ch=in_ch,
            hidden_dim=hidden_dim,
            error_max=self.stage5_5_error_max,
            residual_max=self.stage5_5_residual_max,
            predict_rgb_residual=self.stage5_5_predict_rgb_residual,
        ).to(self.device)

    def _init_stage5_5_nearby_direct(self, config) -> None:
        cfg = config.get("nearby_direct", {}) if hasattr(config, "get") else {}
        losses_cfg = config.get("losses", {}) if hasattr(config, "get") else {}
        tvw_cfg = losses_cfg.get("target_view_weights", {}) if hasattr(losses_cfg, "get") else {}
        nearby_tvw_cfg = tvw_cfg.get("nearby_direct", {}) if hasattr(tvw_cfg, "get") else {}
        self.stage5_5_nearby_direct_enabled = bool(cfg.get("enable", False))
        self.stage5_5_nearby_direct_role = str(cfg.get("role_name", "nearby_direct"))
        self.stage5_5_nearby_direct_policy = str(cfg.get("policy", "adjacent_frame_same_camera")).strip().lower()
        if self.stage5_5_nearby_direct_policy not in {"adjacent_frame_same_camera", "near_random"}:
            raise ValueError(
                "nearby_direct.policy must be one of ['adjacent_frame_same_camera', 'near_random']."
            )
        self.stage5_5_nearby_direct_weight = float(
            cfg.get(
                "weight",
                nearby_tvw_cfg.get(
                    "weight",
                    self._target_view_weight_cfg.get("nearby_direct_weight", 0.7),
                ),
            )
        )
        self.stage5_5_nearby_direct_warmup_steps = int(cfg.get("warmup_steps", 0))
        self.stage5_5_nearby_direct_max_refs = int(cfg.get("max_refs_per_step", 1))
        self.stage5_5_nearby_direct_mask_sky = bool(cfg.get("mask_sky", True))
        self.stage5_5_nearby_direct_mask_egocar = bool(cfg.get("mask_egocar", True))
        self.stage5_5_nearby_direct_mask_dynamic = bool(cfg.get("mask_dynamic", False))
        self.stage5_5_nearby_direct_min_valid_pixel_ratio = float(cfg.get("min_valid_pixel_ratio", 0.03))

    def _debug_check_stage5_5_optimizer_contains_aux_head(self) -> None:
        opt = getattr(self, "optimizer", None)
        head = getattr(self, "stage5_5_uncertainty_head", None)
        if opt is None or head is None:
            return
        opt_param_ids = {
            id(p)
            for group in opt.param_groups
            for p in group.get("params", [])
        }
        missing = [
            name
            for name, p in head.named_parameters()
            if p.requires_grad and id(p) not in opt_param_ids
        ]
        if missing:
            raise RuntimeError(
                "Stage5_5 aux head parameters are not in optimizer: "
                + ", ".join(missing[:8])
            )

    def _stage5_5_maybe_sync_cuda_for_perf(self) -> None:
        if not bool(getattr(self, "stage5_5_perf_sync_cuda", False)):
            return
        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def _feature_splat_render(
        self,
        *,
        render_params: Dict[str, torch.Tensor],
        colors: torch.Tensor,
        view: Any,
        height: int,
        width: int,
        detach_geometry: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if colors.ndim != 2:
            raise ValueError(f"Stage5_5 feature colors must be [N,C], got {tuple(colors.shape)}.")
        means = render_params["means_r"]
        dtype = means.dtype
        dev = means.device
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        c2w = c2w.to(device=dev, dtype=dtype)
        viewmat = get_viewmat(c2w)
        if hasattr(view, "Ks"):
            Ks = view.Ks[0:1]
        elif hasattr(view, "K"):
            Ks = view.K
        else:
            Ks = torch.eye(3, device=dev, dtype=dtype).unsqueeze(0)
        if Ks.dim() == 2:
            Ks = Ks.unsqueeze(0)
        Ks = Ks.to(device=dev, dtype=dtype)
        quats = render_params["quats_r"]
        scales = render_params["scales_r"]
        opacities = render_params["opacities_r"]
        if detach_geometry:
            means = means.detach()
            quats = quats.detach()
            scales = scales.detach()
            opacities = opacities.detach()
        render, alpha, _ = self.renderer(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=int(width),
            height=int(height),
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            channel_chunk=max(32, int(colors.shape[-1])),
        )
        if render.dim() != 4:
            raise RuntimeError(f"Stage5_5 feature render must be 4D [1,H,W,C], got {tuple(render.shape)}.")
        if int(render.shape[0]) != 1:
            raise RuntimeError(f"Stage5_5 feature render batch must be 1, got {int(render.shape[0])}.")
        if tuple(render.shape[1:3]) != (int(height), int(width)):
            raise RuntimeError(
                f"Stage5_5 feature render HW mismatch: got {tuple(render.shape[1:3])} expected {(height, width)}."
            )
        if int(render.shape[-1]) != int(colors.shape[-1]):
            raise RuntimeError(
                f"Stage5_5 feature render expected last channel {colors.shape[-1]}, got {render.shape[-1]}."
            )
        if alpha.dim() != 4 or int(alpha.shape[0]) != 1 or tuple(alpha.shape[1:3]) != (int(height), int(width)):
            raise RuntimeError(f"Stage5_5 feature alpha shape mismatch: got {tuple(alpha.shape)}.")
        feat = render.squeeze(0)
        acc = alpha.squeeze(0)
        if acc.dim() == 3:
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
        detach_geometry: bool = True,
        detach_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(detach_weights) and not bool(detach_geometry):
            raise ValueError(
                "Stage5_5 detach_alpha_weights=true currently requires detach_geometry=true; "
                "true stop-grad alpha weights need a custom weighted feature splat op."
            )
        if node_features.ndim != 2:
            raise ValueError(f"node_features must be [N,C], got {tuple(node_features.shape)}.")
        if node_mask.ndim != 1:
            raise ValueError(f"node_mask must be [N], got {tuple(node_mask.shape)}.")
        if int(node_features.shape[0]) != int(node_mask.shape[0]):
            raise ValueError(
                f"node_features/node_mask length mismatch: {node_features.shape[0]} vs {node_mask.shape[0]}."
            )
        n = int(node_features.shape[0])
        c = int(node_features.shape[1])
        render_n = int(render_params["means_r"].shape[0])
        if render_n != n:
            raise ValueError(f"render_params/node_features length mismatch: {render_n} vs {n}.")
        render_dev = render_params["means_r"].device
        render_dtype = render_params["means_r"].dtype
        if n == 0 or int(node_mask.sum().item()) == 0:
            zero_feat = torch.zeros((height, width, c), dtype=render_dtype, device=render_dev)
            zero_support = torch.zeros((height, width), dtype=render_dtype, device=render_dev)
            return zero_feat, zero_support
        mask_f = node_mask.to(device=render_dev, dtype=render_dtype).reshape(n, 1)
        feat_for_render = node_features.to(device=render_dev, dtype=render_dtype)
        colors_full = torch.zeros((n, c + 1), dtype=render_dtype, device=render_dev)
        colors_full[:, :c] = feat_for_render * mask_f
        colors_full[:, c : c + 1] = mask_f
        rendered, _scene_alpha = self._feature_splat_render(
            render_params=render_params,
            colors=colors_full,
            view=view,
            height=height,
            width=width,
            detach_geometry=detach_geometry,
        )
        if int(rendered.shape[-1]) != c + 1:
            raise RuntimeError(
                f"Stage5_5 feature splat channel mismatch: got {rendered.shape[-1]} expected {c + 1}."
            )
        feat_sum = rendered[..., :c]
        support_map = rendered[..., c]
        support_for_norm = support_map.detach() if detach_weights else support_map
        feat = feat_sum / (support_for_norm.unsqueeze(-1) + float(self.stage5_5_splat_eps))
        return feat, support_map

    @staticmethod
    def _charbonnier(diff: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
        return torch.sqrt(diff * diff + eps * eps)

    @staticmethod
    def _stage5_5_unpack_head_output(head_out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(head_out, (tuple, list)):
            if len(head_out) < 1:
                raise RuntimeError("Stage5_5 uncertainty head returned an empty output tuple.")
            e_pred = head_out[0]
            r_pred = head_out[1] if len(head_out) > 1 else None
            return e_pred, r_pred
        return head_out, None

    def _stage5_5_bridge_weight(self, step: int) -> float:
        if not bool(getattr(self, "stage5_5_bridge_enabled", False)):
            return 0.0
        start = int(getattr(self, "stage5_5_bridge_start_after_steps", 10000))
        if int(step) < start:
            return 0.0
        base = float(getattr(self, "stage5_5_bridge_weight", 0.005))
        max_weight = float(getattr(self, "stage5_5_bridge_max_weight", base))
        warmup = int(getattr(self, "stage5_5_bridge_warmup_steps", 5000))
        if warmup > 0:
            ratio = min(1.0, max(0.0, float(int(step) - start) / float(warmup)))
            base = base * ratio
        return float(min(base, max_weight))

    def _stage5_5_bridge_error_confidence(self, e_pred: torch.Tensor) -> torch.Tensor:
        e = e_pred.detach().float().clamp_min(0.0)
        tau = max(float(getattr(self, "stage5_5_bridge_error_tau", 0.15)), 1.0e-6)
        mode = str(getattr(self, "stage5_5_bridge_error_mode", "exp")).strip().lower()
        if mode == "exp":
            conf = torch.exp(-e / tau)
        elif mode == "linear":
            conf = 1.0 - e / tau
        elif mode == "sigmoid":
            conf = torch.sigmoid((tau - e) / tau)
        else:
            raise ValueError(f"unsupported Stage5_5 bridge error confidence mode: {mode!r}")
        min_conf = float(getattr(self, "stage5_5_bridge_error_min_conf", 0.0))
        max_conf = float(getattr(self, "stage5_5_bridge_error_max_conf", 1.0))
        return conf.clamp(min=min_conf, max=max_conf)

    def _stage5_5_bridge_support_confidence(self, support: torch.Tensor) -> torch.Tensor:
        s = support.detach().float().clamp_min(0.0)
        mode = str(getattr(self, "stage5_5_bridge_support_mode", "soft")).strip().lower()
        hard_min = float(getattr(self, "stage5_5_bridge_support_hard_min", 1.0e-5))
        if mode == "none":
            return torch.ones_like(s)
        if mode == "hard":
            return (s > hard_min).float()
        if mode != "soft":
            raise ValueError(f"unsupported Stage5_5 bridge support confidence mode: {mode!r}")
        tau = max(float(getattr(self, "stage5_5_bridge_support_tau", 1.0e-4)), 1.0e-12)
        gamma = float(getattr(self, "stage5_5_bridge_support_gamma", 0.5))
        conf = (s / (s + tau)).clamp(0.0, 1.0).pow(gamma)
        if hard_min > 0.0:
            conf = conf * (s > hard_min).float()
        return conf

    def _build_stage5_5_bridge_mask(
        self,
        target: Dict[str, Any],
        h: int,
        w: int,
        render_alpha: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dtype = render_alpha.dtype if torch.is_tensor(render_alpha) else torch.float32
        if bool(getattr(self, "stage5_5_bridge_mask_use_valid_loss_mask", True)):
            valid_target = target
            egocar_for_valid = target.get("egocar_mask")
            if egocar_for_valid is not None:
                valid_target = dict(target)
                valid_target["egocar_mask"] = self._stage5_5_hw_mask(
                    egocar_for_valid,
                    h=h,
                    w=w,
                    name="egocar_mask",
                )
            mask = self._valid_loss_mask_from_target(valid_target, height=h, width=w).to(self.device).float()
        else:
            mask = torch.ones((h, w), dtype=torch.float32, device=self.device)

        if bool(getattr(self, "stage5_5_bridge_mask_sky", True)):
            sky = target.get("sky_mask")
            if sky is None:
                if self.require_sky_mask_for_loss:
                    raise ValueError("Stage5_5 bridge requires target['sky_mask'] when mask_sky=true.")
            else:
                sm = self._stage5_5_hw_mask(sky, h=h, w=w, name="sky_mask")
                mask = mask * (1.0 - sm).clamp(0.0, 1.0)

        if bool(getattr(self, "stage5_5_bridge_mask_egocar", True)):
            egocar = target.get("egocar_mask")
            if egocar is not None:
                ego = self._stage5_5_hw_mask(egocar, h=h, w=w, name="egocar_mask")
                mask = mask * (1.0 - ego).clamp(0.0, 1.0)

        if bool(getattr(self, "stage5_5_bridge_mask_dynamic", False)):
            dynamic = target.get("dynamic_mask")
            if dynamic is not None:
                dyn = self._stage5_5_hw_mask(dynamic, h=h, w=w, name="dynamic_mask")
                mask = mask * (1.0 - dyn).clamp(0.0, 1.0)

        if bool(getattr(self, "stage5_5_bridge_require_render_alpha", True)) and render_alpha is not None:
            alpha = self._stage5_5_hw_mask(render_alpha, h=h, w=w, name="render_alpha")
            alpha_min = float(getattr(self, "stage5_5_bridge_render_alpha_min", 0.02))
            mask = mask * (alpha.detach() > alpha_min).float()

        mask = mask.to(dtype=dtype)
        if bool(getattr(self, "stage5_5_bridge_detach_mask", True)):
            mask = mask.detach()
        return mask

    def _stage5_5_weighted_bridge_loss(
        self,
        pred_rgb_live: torch.Tensor,
        gt_rgb: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diff = pred_rgb_live - gt_rgb
        abs_l1 = diff.abs().mean(dim=-1)
        if str(getattr(self, "stage5_5_bridge_loss_type", "l1")).strip().lower() == "charbonnier":
            per_pixel = self._charbonnier(diff).mean(dim=-1)
        else:
            per_pixel = abs_l1

        conf = confidence.detach() if bool(getattr(self, "stage5_5_bridge_detach_confidence", True)) else confidence
        weight_mask = mask.detach() if bool(getattr(self, "stage5_5_bridge_detach_mask", True)) else mask
        weight = (conf * weight_mask).to(dtype=per_pixel.dtype)
        weight_sum = weight.sum()
        if bool(getattr(self, "stage5_5_bridge_normalize_by_weight_sum", True)):
            loss = (weight * per_pixel).sum() / weight_sum.clamp_min(1.0e-6)
            weighted_l1 = (weight * abs_l1).sum() / weight_sum.clamp_min(1.0e-6)
        else:
            loss = (weight * per_pixel).mean()
            weighted_l1 = (weight * abs_l1).mean()
        stats = {
            "raw_l1": abs_l1.detach().mean(),
            "weighted_l1": weighted_l1.detach(),
            "weight_sum": weight_sum.detach(),
            "active_ratio": (weight.detach() > 1.0e-6).float().mean(),
        }
        return loss, stats

    def _stage5_5_save_bridge_debug_maps(
        self,
        *,
        step: int,
        target_index: int,
        gt: torch.Tensor,
        pred_rgb: torch.Tensor,
        e_pred: torch.Tensor,
        support: torch.Tensor,
        confidence: torch.Tensor,
        bridge_mask: torch.Tensor,
        bridge_weight_map: torch.Tensor,
    ) -> None:
        interval = int(self._stage5_5_aux_image_interval_steps())
        if interval <= 0 or int(step) % interval != 0 or int(target_index) != 0:
            return
        log_dir = getattr(getattr(self, "config", None), "log_dir", None)
        if log_dir is None:
            return
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            return

        out_dir = os.path.join(str(log_dir), "images", "aux_bridge")
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"step_{int(step):07d}_target_{int(target_index):02d}"

        def _rgb_u8(x: torch.Tensor) -> "np.ndarray":
            arr = x.detach().float().cpu().clamp(0.0, 1.0).numpy()
            return (arr * 255.0 + 0.5).astype(np.uint8)

        def _map_u8(x: torch.Tensor, *, normalize: bool = True) -> "np.ndarray":
            arr = x.detach().float().cpu()
            arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if normalize:
                mn = float(arr.min().item())
                mx = float(arr.max().item())
                if mx > mn:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = arr * 0.0
            else:
                arr = arr.clamp(0.0, 1.0)
            return (arr.numpy() * 255.0 + 0.5).astype(np.uint8)

        abs_err = (pred_rgb.detach() - gt.detach()).abs().mean(dim=-1)
        Image.fromarray(_rgb_u8(gt)).save(os.path.join(out_dir, f"{prefix}_gt.png"))
        Image.fromarray(_rgb_u8(pred_rgb)).save(os.path.join(out_dir, f"{prefix}_render.png"))
        Image.fromarray(_map_u8(abs_err)).save(os.path.join(out_dir, f"{prefix}_abs_error.png"))
        Image.fromarray(_map_u8(e_pred)).save(os.path.join(out_dir, f"{prefix}_pred_error.png"))
        Image.fromarray(_map_u8(support)).save(os.path.join(out_dir, f"{prefix}_support.png"))
        Image.fromarray(_map_u8(confidence, normalize=False)).save(os.path.join(out_dir, f"{prefix}_confidence.png"))
        Image.fromarray(_map_u8(bridge_mask, normalize=False)).save(os.path.join(out_dir, f"{prefix}_mask.png"))
        Image.fromarray(_map_u8(bridge_weight_map, normalize=False)).save(
            os.path.join(out_dir, f"{prefix}_weight.png")
        )

    def _stage5_5_hw_mask(self, mask: torch.Tensor, *, h: int, w: int, name: str) -> torch.Tensor:
        m = mask.to(self.device).float()
        while m.dim() > 2:
            if int(m.shape[0]) == 1:
                m = m.squeeze(0)
            elif int(m.shape[-1]) == 1:
                m = m.squeeze(-1)
            else:
                raise ValueError(f"Stage5_5 {name} cannot be squeezed to [H,W]: got {tuple(m.shape)}.")
        if tuple(m.shape) != (h, w):
            raise ValueError(f"Stage5_5 {name} shape mismatch: got {tuple(m.shape)} expected {(h, w)}.")
        return m

    def _build_aux_loss_mask(self, target: Dict[str, Any], h: int, w: int) -> torch.Tensor:
        valid_target = target
        egocar_for_valid = target.get("egocar_mask")
        if egocar_for_valid is not None:
            valid_target = dict(target)
            valid_target["egocar_mask"] = self._stage5_5_hw_mask(
                egocar_for_valid,
                h=h,
                w=w,
                name="egocar_mask",
            )
        valid_loss_mask = self._valid_loss_mask_from_target(valid_target, height=h, width=w).to(self.device).float()
        if self.stage5_5_loss_mask_egocar:
            egocar = target.get("egocar_mask")
            if egocar is not None:
                ego = self._stage5_5_hw_mask(egocar, h=h, w=w, name="egocar_mask")
                valid_loss_mask = valid_loss_mask * (1.0 - ego).clamp(0.0, 1.0)
        if not self.stage5_5_loss_mask_sky:
            return valid_loss_mask
        sky = target.get("sky_mask")
        if sky is None:
            if self.require_sky_mask_for_loss:
                raise ValueError("Stage5_5 requires target['sky_mask'] when mask_sky=true.")
            return valid_loss_mask
        sm = self._stage5_5_hw_mask(sky, h=h, w=w, name="sky_mask")
        return valid_loss_mask * (1.0 - sm).clamp(0.0, 1.0)

    def _collect_aux_targets(
        self,
        batch: Dict[str, Any],
        *,
        max_targets: Optional[int] = None,
        require_materialized: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        request_meta = batch.get("request_meta") or {}
        requested_aux = request_meta.get("aux_image_refs") or batch.get("aux_image_refs") or []
        aux = batch.get("aux_targets")
        require_aux = bool(self.stage5_5_aux_enabled) if require_materialized is None else bool(require_materialized)
        if require_aux and len(requested_aux) > 0:
            if not isinstance(aux, list) or len(aux) == 0:
                raise RuntimeError(
                    "Stage5_5 got aux_image_refs but batch['aux_targets'] is missing or empty. "
                    "Dataset/conversion must materialize aux targets before trainer.forward()."
                )
        if not isinstance(aux, list):
            return []
        limit = int(self.stage5_5_target_max_targets) if max_targets is None else int(max_targets)
        if limit > 0:
            return aux[:limit]
        return aux

    def _stage5_5_nearby_direct_weight(self, step: int) -> float:
        base = float(getattr(self, "stage5_5_nearby_direct_weight", 0.7))
        warmup = int(getattr(self, "stage5_5_nearby_direct_warmup_steps", 0))
        if warmup <= 0:
            return base
        ratio = min(1.0, max(0.0, float(int(step)) / float(warmup)))
        return float(base * ratio)

    def _stage5_5_aux_image_interval_steps(self) -> int:
        cfg = getattr(self, "config", None)
        logging_cfg = cfg.get("logging", {}) if hasattr(cfg, "get") else {}
        sched_cfg = cfg.get("scheduler_v8", {}) if hasattr(cfg, "get") else {}
        block_cfg = sched_cfg.get("block", {}) if hasattr(sched_cfg, "get") else {}
        image_blocks = int(logging_cfg.get("image_interval_blocks", 0)) if hasattr(logging_cfg, "get") else 0
        steps_per_block = int(block_cfg.get("steps_per_block", 1)) if hasattr(block_cfg, "get") else 1
        if image_blocks > 0:
            return int(image_blocks * max(steps_per_block, 1))
        return int(getattr(self, "stage5_5_bridge_debug_save_maps_interval", 0))

    def _stage5_5_main_loss_denominator(self, batch: Dict[str, Any], step: int, like: torch.Tensor) -> torch.Tensor:
        targets = batch.get("targets") or []
        num_targets = int(len(targets))
        if num_targets <= 0:
            return like.new_zeros(())
        if bool(self._target_view_weight_cfg.get("normalize_by_weight_sum", True)):
            weights, _roles = self._build_target_view_weights(batch, step=int(step), num_targets=num_targets)
            return weights.to(device=like.device, dtype=like.dtype).sum()
        return like.new_tensor(float(num_targets))

    def _build_nearby_direct_loss_mask(self, target: Dict[str, Any], h: int, w: int) -> torch.Tensor:
        valid_target = target
        egocar_for_valid = target.get("egocar_mask")
        if egocar_for_valid is not None:
            valid_target = dict(target)
            valid_target["egocar_mask"] = self._stage5_5_hw_mask(
                egocar_for_valid,
                h=h,
                w=w,
                name="egocar_mask",
            )
        mask = self._valid_loss_mask_from_target(valid_target, height=h, width=w).to(self.device).float()

        if bool(getattr(self, "stage5_5_nearby_direct_mask_sky", True)):
            sky = target.get("sky_mask")
            if sky is None:
                if self.require_sky_mask_for_loss:
                    raise ValueError("Stage5_5 nearby_direct requires target['sky_mask'] when mask_sky=true.")
            else:
                sm = self._stage5_5_hw_mask(sky, h=h, w=w, name="sky_mask")
                mask = mask * (1.0 - sm).clamp(0.0, 1.0)

        if bool(getattr(self, "stage5_5_nearby_direct_mask_egocar", True)):
            egocar = target.get("egocar_mask")
            if egocar is not None:
                ego = self._stage5_5_hw_mask(egocar, h=h, w=w, name="egocar_mask")
                mask = mask * (1.0 - ego).clamp(0.0, 1.0)

        if bool(getattr(self, "stage5_5_nearby_direct_mask_dynamic", False)):
            dynamic = target.get("dynamic_mask")
            if dynamic is not None:
                dyn = self._stage5_5_hw_mask(dynamic, h=h, w=w, name="dynamic_mask")
                mask = mask * (1.0 - dyn).clamp(0.0, 1.0)

        return mask

    def _build_stage5_5_rigid_world_for_aux_frame(
        self,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        node_state_rigid = out.get("_node_state_rigid")
        if node_state_rigid is None:
            return None
        u_all = out.get("_rigid_writeback_idx")
        if u_all is None:
            u_all = torch.zeros((0,), dtype=torch.long, device=self.device)
        else:
            u_all = u_all.to(device=self.device, dtype=torch.long)
        n_rigid = int(node_state_rigid.means.shape[0])
        if int(u_all.numel()) > 0:
            if bool((u_all < 0).any().item()) or bool((u_all >= n_rigid).any().item()):
                raise RuntimeError("Stage5_5 aux U_all contains out-of-range indices.")
        is_updated = torch.zeros((n_rigid,), dtype=torch.bool, device=self.device)
        if int(u_all.numel()) > 0:
            is_updated[u_all] = True
        target_valid = torch.nonzero(
            self._rigid_point_valid_mask(node_state_rigid, int(target_frame_idx)),
            as_tuple=False,
        ).squeeze(1).to(device=self.device, dtype=torch.long)
        if int(target_valid.numel()) == 0:
            return None
        idx_train = target_valid[is_updated[target_valid]]
        idx_frozen = target_valid[~is_updated[target_valid]]
        rigid_local = out.get("_render_params_rigid_local")
        if int(idx_train.numel()) > 0 and rigid_local is None:
            raise RuntimeError("Stage5_5 aux needs _render_params_rigid_local for updated rigid nodes.")
        return self._build_rigid_world_for_frame(
            node_state_rigid,
            int(target_frame_idx),
            idx_train,
            idx_frozen,
            rigid_local,
            u_all,
        )

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
            raise RuntimeError("Stage5_5 nearby_direct expected _proxies_distant for distant render params.")

        rigid_world = self._build_stage5_5_rigid_world_for_aux_frame(out, int(target_frame_idx))
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

    @staticmethod
    def _merge_lists(items: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        valid = [x for x in items if x is not None and int(x.shape[0]) > 0]
        if len(valid) == 0:
            return None
        return torch.cat(valid, dim=0)

    def _build_aux_node_pack(
        self,
        *,
        out: Dict[str, Any],
        target_frame_idx: int,
    ) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        feat_bg = out.get("_feat_2d_bg")
        acc_bg = out.get("_acc_w_bg")
        render_bg = out.get("_render_params_bg")
        if feat_bg is None or acc_bg is None or render_bg is None:
            return None

        feats: List[Optional[torch.Tensor]] = []
        masks: List[Optional[torch.Tensor]] = []

        feats.append(feat_bg)
        masks.append(acc_bg > float(self.stage5_5_src_support_min_bg))
        rigid_world = None

        node_state_rigid = out.get("_node_state_rigid")
        route = out.get("_route")
        if node_state_rigid is not None:
            if route is None:
                raise RuntimeError("Stage5_5 rigid aux requires _route to align source-observed rigid features.")
            route_s = route.S.to(device=self.device, dtype=torch.long)
            if int(route_s.numel()) > 0 and (out.get("_feat_2d_rigid_S") is None or out.get("_acc_w_rigid_S") is None):
                raise RuntimeError("Stage5_5 rigid aux requires _feat_2d_rigid_S and _acc_w_rigid_S for route.S.")
            u_all = out.get("_rigid_writeback_idx")
            if u_all is None:
                u_all = torch.zeros((0,), dtype=torch.long, device=self.device)
            else:
                u_all = u_all.to(device=self.device, dtype=torch.long)
            n_rigid = int(node_state_rigid.means.shape[0])
            if int(u_all.numel()) > 0:
                if bool((u_all < 0).any().item()) or bool((u_all >= n_rigid).any().item()):
                    raise RuntimeError("Stage5_5 rigid aux U_all contains out-of-range indices.")
            is_updated = torch.zeros((n_rigid,), dtype=torch.bool, device=self.device)
            if int(u_all.numel()) > 0:
                is_updated[u_all] = True
            target_valid = torch.nonzero(
                self._rigid_point_valid_mask(node_state_rigid, int(target_frame_idx)),
                as_tuple=False,
            ).squeeze(1).to(device=self.device, dtype=torch.long)
            idx_train = target_valid[is_updated[target_valid]]
            idx_frozen = target_valid[~is_updated[target_valid]]
            rigid_local = out.get("_render_params_rigid_local")
            if int(idx_train.numel()) > 0 and rigid_local is None:
                raise RuntimeError("Stage5_5 rigid aux needs _render_params_rigid_local for updated rigid nodes.")
            rigid_world = self._build_rigid_world_for_frame(
                node_state_rigid,
                int(target_frame_idx),
                idx_train,
                idx_frozen,
                rigid_local,
                u_all,
            )
            rigid_order = torch.cat([idx_train, idx_frozen], dim=0)
            if rigid_world is not None and int(rigid_order.numel()) > 0:
                feat_rigid_s = out.get("_feat_2d_rigid_S")
                acc_rigid_s = out.get("_acc_w_rigid_S")
                feat_rigid_ordered = feat_bg.new_zeros((int(rigid_order.numel()), int(feat_bg.shape[1])))
                mask_rigid_ordered = torch.zeros((int(rigid_order.numel()),), dtype=torch.bool, device=self.device)
                if int(route_s.numel()) > 0:
                    lookup_s = torch.full((n_rigid,), -1, dtype=torch.long, device=self.device)
                    lookup_s[route_s] = torch.arange(int(route_s.numel()), dtype=torch.long, device=self.device)
                    rows = lookup_s[rigid_order]
                    observed = rows >= 0
                    if bool(observed.any().item()):
                        obs_rows = rows[observed]
                        if int(feat_rigid_s.shape[1]) != int(feat_bg.shape[1]):
                            raise RuntimeError(
                                "Stage5_5 rigid aux feature dim mismatch: "
                                f"{feat_rigid_s.shape[1]} vs bg {feat_bg.shape[1]}."
                            )
                        feat_rigid_ordered[observed] = feat_rigid_s[obs_rows].to(
                            device=feat_rigid_ordered.device,
                            dtype=feat_rigid_ordered.dtype,
                        )
                        mask_rigid_ordered[observed] = (
                            acc_rigid_s[obs_rows].to(device=self.device) > float(self.stage5_5_src_support_min_rigid)
                        )
                feats.append(feat_rigid_ordered)
                masks.append(mask_rigid_ordered)

        feat_distant = out.get("_feat_2d_distant")
        acc_distant = out.get("_acc_w_distant")
        render_distant = out.get("_render_params_distant")
        if feat_distant is not None and acc_distant is not None and render_distant is not None and int(feat_distant.shape[0]) > 0:
            if int(feat_distant.shape[1]) != int(feat_bg.shape[1]):
                raise RuntimeError(
                    "Stage5_5 distant aux feature dim mismatch: "
                    f"{feat_distant.shape[1]} vs bg {feat_bg.shape[1]}."
                )
            feats.append(feat_distant)
            masks.append(acc_distant > float(self.stage5_5_src_support_min_distant))

        merged_features = self._merge_lists(feats)
        merged_mask = self._merge_lists(masks)
        if merged_features is None or merged_mask is None:
            return None
        merged_render = self._tensor_merge_bg_rigid_distant_world(
            out["_render_params_bg"],
            rigid_world,
            render_distant,
        )
        if int(merged_render["means_r"].shape[0]) != int(merged_features.shape[0]):
            raise RuntimeError(
                "Stage5_5 merged render/features length mismatch: "
                f"{merged_render['means_r'].shape[0]} vs {merged_features.shape[0]}."
            )
        return merged_render, merged_features, merged_mask.bool()

    def _compute_nearby_direct_loss_from_aux_targets(
        self,
        *,
        batch: Dict[str, Any],
        out: Dict[str, Any],
    ) -> Dict[str, Any]:
        zero = out["loss"].new_zeros(())
        if not bool(getattr(self, "stage5_5_nearby_direct_enabled", False)):
            return {"loss": zero, "skipped_disabled": 1.0}

        step = int(self._current_loss_step(batch))
        view_weight_value = float(self._stage5_5_nearby_direct_weight(step))
        aux_targets = self._collect_aux_targets(
            batch,
            max_targets=int(getattr(self, "stage5_5_nearby_direct_max_refs", 1)),
            require_materialized=True,
        )
        if len(aux_targets) == 0:
            return {
                "loss": zero,
                "loss_rgb": zero,
                "loss_ssim": zero,
                "monitor_l1": zero,
                "monitor_psnr": zero,
                "weight_sum": zero,
                "denominator": zero,
                "view_weight": float(view_weight_value),
                "processed_targets": 0.0,
                "skipped_empty": 1.0,
            }
        if view_weight_value <= 0.0:
            return {
                "loss": zero,
                "loss_rgb": zero,
                "loss_ssim": zero,
                "monitor_l1": zero,
                "monitor_psnr": zero,
                "weight_sum": zero,
                "denominator": zero,
                "view_weight": float(view_weight_value),
                "processed_targets": 0.0,
                "skipped_zero_weight": 1.0,
            }

        weighted_rgb_sum = zero.clone()
        weighted_ssim_sum = zero.clone()
        weighted_total_sum = zero.clone()
        weight_sum = zero.clone()
        l1_terms: List[torch.Tensor] = []
        psnr_terms: List[torch.Tensor] = []
        valid_ratio_terms: List[torch.Tensor] = []
        processed = 0
        skipped_low_valid = 0

        for target in aux_targets:
            gt = target.get("gt_image")
            view = target.get("view")
            if "frame_idx" not in target:
                raise RuntimeError("Stage5_5 nearby_direct target must provide frame_idx.")
            if gt is None or view is None:
                raise RuntimeError("Stage5_5 nearby_direct target must provide gt_image and view.")
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            h, w = int(gt.shape[0]), int(gt.shape[1])
            render_params = self._build_nearby_direct_proxy_render_params(out, int(target["frame_idx"]))
            if render_params is None:
                continue
            pred_rgb, _pred_alpha = self._render_single_view(render_params, view, h, w)
            gt = gt.to(device=self.device, dtype=pred_rgb.dtype)
            mask = self._build_nearby_direct_loss_mask(target, h, w).to(device=self.device, dtype=pred_rgb.dtype)
            valid_ratio = mask.mean()
            if float(valid_ratio.detach().item()) < float(
                getattr(self, "stage5_5_nearby_direct_min_valid_pixel_ratio", 0.03)
            ):
                skipped_low_valid += 1
                continue
            denom_pixels = mask.sum().clamp_min(1.0)
            abs_rgb = (pred_rgb - gt).abs()
            l1_raw = (abs_rgb * mask.unsqueeze(-1)).sum() / (denom_pixels * 3.0)
            l1_loss = float(self.loss_w_l1) * l1_raw
            ssim_loss = float(self.loss_w_ssim) * compute_ssim_loss_masked(
                pred_rgb,
                gt,
                valid_mask=mask,
                sky_mask=None,
                data_range=1.0,
            )
            total_i = l1_loss + ssim_loss
            view_weight = pred_rgb.new_tensor(float(view_weight_value))
            weighted_rgb_sum = weighted_rgb_sum + l1_loss * view_weight
            weighted_ssim_sum = weighted_ssim_sum + ssim_loss * view_weight
            weighted_total_sum = weighted_total_sum + total_i * view_weight
            weight_sum = weight_sum + view_weight
            mse = (((pred_rgb - gt) ** 2) * mask.unsqueeze(-1)).sum() / (denom_pixels * 3.0)
            psnr_terms.append((-10.0 * torch.log10(mse.clamp_min(1.0e-10))).detach())
            l1_terms.append(l1_raw.detach())
            valid_ratio_terms.append(valid_ratio.detach())
            processed += 1

        if processed == 0:
            return {
                "loss": zero,
                "loss_rgb": zero,
                "loss_ssim": zero,
                "monitor_l1": zero,
                "monitor_psnr": zero,
                "weight_sum": zero,
                "denominator": zero,
                "view_weight": float(view_weight_value),
                "processed_targets": 0.0,
                "skipped_low_valid": float(skipped_low_valid),
            }

        if bool(self._target_view_weight_cfg.get("normalize_by_weight_sum", True)):
            denom = weight_sum.clamp_min(1.0e-8)
        else:
            denom = zero.new_tensor(float(processed)).clamp_min(1.0)
        return {
            "loss": weighted_total_sum / denom,
            "loss_rgb": weighted_rgb_sum / denom,
            "loss_ssim": weighted_ssim_sum / denom,
            "monitor_l1": torch.stack(l1_terms).mean(),
            "monitor_psnr": torch.stack(psnr_terms).mean(),
            "valid_mask_ratio": torch.stack(valid_ratio_terms).mean(),
            "weight_sum": weight_sum.detach(),
            "denominator": denom.detach(),
            "view_weight": float(view_weight_value),
            "processed_targets": float(processed),
            "skipped_low_valid": float(skipped_low_valid),
        }

    def _compute_feature_splat_uncertainty_loss(
        self,
        *,
        batch: Dict[str, Any],
        out: Dict[str, Any],
    ) -> Dict[str, Any]:
        aux_targets = self._collect_aux_targets(
            batch,
            max_targets=int(self.stage5_5_target_max_targets),
            require_materialized=bool(self.stage5_5_aux_enabled),
        )
        if len(aux_targets) == 0:
            return {"loss": out["loss"].new_zeros(()), "skipped_empty": 1.0}
        step = int(self._current_loss_step(batch))
        every_n = max(int(self.stage5_5_target_every_n_steps), 1)
        if step % every_n != 0:
            return {"loss": out["loss"].new_zeros(()), "skipped_interval": 1.0}
        bridge_weight_value = self._stage5_5_bridge_weight(step)

        loss_all_terms: List[torch.Tensor] = []
        loss_supported_terms: List[torch.Tensor] = []
        loss_rgb_residual_terms: List[torch.Tensor] = []
        loss_rgb_residual_supported_terms: List[torch.Tensor] = []
        bridge_loss_terms: List[torch.Tensor] = []
        pred_means: List[torch.Tensor] = []
        gt_means: List[torch.Tensor] = []
        abs_err_means: List[torch.Tensor] = []
        residual_abs_err_means: List[torch.Tensor] = []
        residual_pred_abs_means: List[torch.Tensor] = []
        residual_gt_abs_means: List[torch.Tensor] = []
        corr_terms: List[float] = []
        support_means: List[torch.Tensor] = []
        support_maxs: List[torch.Tensor] = []
        support_valid_ratios: List[torch.Tensor] = []
        loss_mask_ratios: List[torch.Tensor] = []
        supported_mask_ratios: List[torch.Tensor] = []
        bridge_raw_l1_terms: List[torch.Tensor] = []
        bridge_weighted_l1_terms: List[torch.Tensor] = []
        bridge_conf_means: List[torch.Tensor] = []
        bridge_conf_p10s: List[torch.Tensor] = []
        bridge_conf_p50s: List[torch.Tensor] = []
        bridge_conf_p90s: List[torch.Tensor] = []
        bridge_error_conf_means: List[torch.Tensor] = []
        bridge_support_conf_means: List[torch.Tensor] = []
        bridge_active_ratios: List[torch.Tensor] = []
        bridge_weight_sums: List[torch.Tensor] = []
        skipped_empty_supported = 0
        skipped_low_active_bridge = 0
        processed = 0
        aux_feature_render_calls = 0
        aux_feature_splat_time_ms = 0.0
        aux_render_context_time_ms = 0.0
        no_render_probe_loss = None
        with_render_probe_loss = None
        aux_debug_maps: Optional[Dict[str, Any]] = None

        for target_idx, target in enumerate(aux_targets):
            gt = target.get("gt_image")
            view = target.get("view")
            if "frame_idx" not in target:
                raise RuntimeError("Stage5_5 aux target must provide frame_idx.")
            frame_idx = int(target["frame_idx"])
            if gt is None or view is None:
                raise RuntimeError("Stage5_5 aux target must provide gt_image and view.")
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            h, w = int(gt.shape[0]), int(gt.shape[1])

            node_pack = self._build_aux_node_pack(out=out, target_frame_idx=frame_idx)
            if node_pack is None:
                continue
            merged_render, merged_features, merged_mask = node_pack
            if int(merged_mask.sum().item()) == 0:
                continue

            self._stage5_5_maybe_sync_cuda_for_perf()
            t_splat0 = time.perf_counter()
            feat_tilde, support = self._splat_node_features_to_view(
                render_params=merged_render,
                node_features=merged_features,
                node_mask=merged_mask,
                view=view,
                height=h,
                width=w,
                detach_geometry=bool(self.stage5_5_detach_geometry),
                detach_weights=bool(self.stage5_5_detach_alpha_weights),
            )
            self._stage5_5_maybe_sync_cuda_for_perf()
            aux_feature_splat_time_ms += float((time.perf_counter() - t_splat0) * 1000.0)
            aux_feature_render_calls += 1
            self._stage5_5_maybe_sync_cuda_for_perf()
            t_rgb0 = time.perf_counter()
            pred_rgb_live, pred_alpha_live = self._render_single_view(merged_render, view, h, w)
            self._stage5_5_maybe_sync_cuda_for_perf()
            aux_render_context_time_ms += float((time.perf_counter() - t_rgb0) * 1000.0)
            if pred_alpha_live.dim() == 3 and int(pred_alpha_live.shape[-1]) == 1:
                pred_alpha_live = pred_alpha_live.squeeze(-1)
            if tuple(pred_alpha_live.shape) != (h, w):
                raise RuntimeError(
                    f"Stage5_5 pred_alpha_t shape mismatch: got {tuple(pred_alpha_live.shape)} expected {(h, w)}"
                )
            gt = gt.to(device=self.device, dtype=pred_rgb_live.dtype)
            pred_rgb_ctx = pred_rgb_live
            pred_alpha_ctx = pred_alpha_live
            if self.stage5_5_detach_render_context:
                pred_rgb_ctx = pred_rgb_ctx.detach()
                pred_alpha_ctx = pred_alpha_ctx.detach()
            e_gt = (gt - pred_rgb_ctx).abs().mean(dim=-1)
            e_gt = torch.clamp(e_gt, min=0.0, max=float(self.stage5_5_error_max))
            r_gt = (gt - pred_rgb_ctx.detach()).clamp(
                min=-float(getattr(self, "stage5_5_residual_max", 0.5)),
                max=float(getattr(self, "stage5_5_residual_max", 0.5)),
            )

            mask_loss = self._build_aux_loss_mask(target, h, w)
            valid_ratio = float(mask_loss.mean().item())
            if valid_ratio < float(self.stage5_5_min_valid_pixel_ratio):
                continue
            support_pos = support.clamp_min(0.0)
            support_log = torch.log1p(support_pos)
            if self.stage5_5_detach_alpha_weights:
                support_pos = support_pos.detach()
                support_log = support_log.detach()
            render_ctx_rgb = pred_rgb_ctx
            render_ctx_alpha = pred_alpha_ctx
            use_render_ctx = True
            if self.training and self.stage5_5_render_context_dropout > 0.0:
                if float(torch.rand(1, device=self.device).item()) < float(self.stage5_5_render_context_dropout):
                    use_render_ctx = False
            if not use_render_ctx:
                render_ctx_rgb = torch.zeros_like(render_ctx_rgb)
                render_ctx_alpha = torch.zeros_like(render_ctx_alpha)

            if self.stage5_5_zero_invalid_input:
                feat_tilde = feat_tilde * mask_loss.unsqueeze(-1)
                render_ctx_rgb = render_ctx_rgb * mask_loss.unsqueeze(-1)
                render_ctx_alpha = render_ctx_alpha * mask_loss
                support_pos = support_pos * mask_loss
                support_log = support_log * mask_loss

            head_inputs = [feat_tilde.permute(2, 0, 1), support_pos.unsqueeze(0)]
            if self.stage5_5_concat_log_support:
                head_inputs.append(support_log.unsqueeze(0))
            if self.stage5_5_concat_valid_mask:
                head_inputs.append(mask_loss.unsqueeze(0))
            if self.stage5_5_use_render_rgb:
                head_inputs.append(render_ctx_rgb.permute(2, 0, 1))
            if self.stage5_5_use_render_alpha:
                head_inputs.append(render_ctx_alpha.unsqueeze(0))
            head_in = torch.cat(head_inputs, dim=0).unsqueeze(0)
            head_out = self.stage5_5_uncertainty_head(head_in)
            e_pred_raw, r_pred_raw = self._stage5_5_unpack_head_output(head_out)
            e_pred = e_pred_raw.squeeze(0).squeeze(0)
            r_pred = None
            if r_pred_raw is not None:
                r_pred_chw = r_pred_raw.squeeze(0)
                if r_pred_chw.dim() != 3 or int(r_pred_chw.shape[0]) != 3:
                    raise RuntimeError(
                        f"Stage5_5 rgb residual head must return [1,3,H,W], got {tuple(r_pred_raw.shape)}."
                    )
                r_pred = r_pred_chw.permute(1, 2, 0)
                if tuple(r_pred.shape) != (h, w, 3):
                    raise RuntimeError(
                        f"Stage5_5 rgb residual pred shape mismatch: got {tuple(r_pred.shape)} expected {(h, w, 3)}."
                    )
            if aux_debug_maps is None:
                with torch.no_grad():
                    aux_debug_maps = {
                        "target_index": int(target_idx),
                        "frame_idx": int(frame_idx),
                        "gt": gt.detach().float().cpu(),
                        "render": pred_rgb_live.detach().float().cpu(),
                        "abs_error": (pred_rgb_live.detach() - gt.detach()).abs().mean(dim=-1).float().cpu(),
                        "pred_error": e_pred.detach().float().cpu(),
                        "support": support.detach().float().cpu(),
                        "loss_mask": mask_loss.detach().float().cpu(),
                        "rgb_residual_max": float(getattr(self, "stage5_5_residual_max", 0.5)),
                    }
                    if r_pred is not None:
                        aux_debug_maps["pred_rgb_residual"] = r_pred.detach().float().cpu()
                        aux_debug_maps["gt_rgb_residual"] = r_gt.detach().float().cpu()

            diff = self._charbonnier(e_pred - e_gt)
            denom_all = mask_loss.sum().clamp_min(1.0)
            loss_all = (diff * mask_loss).sum() / denom_all

            mask_supported = mask_loss * (support_pos > float(self.stage5_5_support_min_for_extra_loss)).float()
            den_supported = mask_supported.sum()
            if float(den_supported.detach().item()) > 0.0:
                loss_sup = (diff * mask_supported).sum() / den_supported
                loss_supported_terms.append(loss_sup)
            else:
                skipped_empty_supported += 1

            if r_pred is not None:
                residual_diff = self._charbonnier(r_pred - r_gt).mean(dim=-1)
                loss_residual = (residual_diff * mask_loss).sum() / denom_all
                loss_rgb_residual_terms.append(loss_residual)
                if float(den_supported.detach().item()) > 0.0:
                    loss_residual_sup = (residual_diff * mask_supported).sum() / den_supported
                    loss_rgb_residual_supported_terms.append(loss_residual_sup)
                residual_abs_err_means.append(((r_pred - r_gt).abs().mean(dim=-1) * mask_loss).sum() / denom_all)
                residual_pred_abs_means.append((r_pred.abs().mean(dim=-1) * mask_loss).sum() / denom_all)
                residual_gt_abs_means.append((r_gt.abs().mean(dim=-1) * mask_loss).sum() / denom_all)

            loss_all_terms.append(loss_all)
            pred_means.append((e_pred * mask_loss).sum() / denom_all)
            gt_means.append((e_gt * mask_loss).sum() / denom_all)
            abs_err_means.append(((e_pred - e_gt).abs() * mask_loss).sum() / denom_all)
            support_means.append((support_pos * mask_loss).sum() / denom_all)
            support_maxs.append(support_pos.max())
            support_valid_ratios.append((support_pos > 0).float().mean())
            loss_mask_ratios.append(mask_loss.mean())
            supported_mask_ratios.append(mask_supported.mean())
            processed += 1

            with torch.no_grad():
                x = e_pred[mask_loss > 0.0].reshape(-1)
                y = e_gt[mask_loss > 0.0].reshape(-1)
                if int(x.numel()) > 8:
                    x = x.float()
                    y = y.float()
                    vx = x - x.mean()
                    vy = y - y.mean()
                    den = (vx.norm() * vy.norm()).clamp_min(1.0e-8)
                    corr_terms.append(float((vx * vy).sum().item() / float(den.item())))

            if bridge_weight_value > 0.0:
                with torch.no_grad():
                    c_e = self._stage5_5_bridge_error_confidence(e_pred)
                    c_s = self._stage5_5_bridge_support_confidence(support)
                    confidence = (c_e * c_s).clamp(0.0, 1.0)
                    bridge_mask = self._build_stage5_5_bridge_mask(
                        target,
                        h,
                        w,
                        pred_alpha_live.detach(),
                    )
                    bridge_weight_map = (confidence * bridge_mask).detach()
                    bridge_l1_map_det = (pred_rgb_live.detach() - gt.detach()).abs().mean(dim=-1)
                    bridge_weight_sum = bridge_weight_map.sum()
                    bridge_active_ratio = (bridge_weight_map > 1.0e-6).float().mean()
                    bridge_raw_l1 = bridge_l1_map_det.mean()
                    bridge_weighted_l1 = (
                        (bridge_weight_map * bridge_l1_map_det).sum() / bridge_weight_sum.clamp_min(1.0e-6)
                    )
                    conf_flat = confidence.reshape(-1).float()
                    bridge_raw_l1_terms.append(bridge_raw_l1)
                    bridge_weighted_l1_terms.append(bridge_weighted_l1)
                    bridge_conf_means.append(conf_flat.mean())
                    bridge_conf_p10s.append(torch.quantile(conf_flat, 0.10))
                    bridge_conf_p50s.append(torch.quantile(conf_flat, 0.50))
                    bridge_conf_p90s.append(torch.quantile(conf_flat, 0.90))
                    bridge_error_conf_means.append(c_e.float().mean())
                    bridge_support_conf_means.append(c_s.float().mean())
                    bridge_active_ratios.append(bridge_active_ratio)
                    bridge_weight_sums.append(bridge_weight_sum)
                    if aux_debug_maps is not None and int(aux_debug_maps.get("target_index", -1)) == int(target_idx):
                        aux_debug_maps["error_confidence"] = c_e.detach().float().cpu()
                        aux_debug_maps["support_confidence"] = c_s.detach().float().cpu()
                        aux_debug_maps["confidence"] = confidence.detach().float().cpu()
                        aux_debug_maps["bridge_mask"] = bridge_mask.detach().float().cpu()
                        aux_debug_maps["bridge_weight"] = bridge_weight_map.detach().float().cpu()

                if float(bridge_active_ratio.item()) >= float(
                    getattr(self, "stage5_5_bridge_min_effective_pixel_ratio", 0.005)
                ):
                    bridge_raw_loss, bridge_stats = self._stage5_5_weighted_bridge_loss(
                        pred_rgb_live=pred_rgb_live,
                        gt_rgb=gt,
                        confidence=confidence,
                        mask=bridge_mask,
                    )
                    bridge_loss_terms.append(float(bridge_weight_value) * bridge_raw_loss)
                    if torch.is_tensor(bridge_stats.get("raw_l1")):
                        bridge_raw_l1_terms[-1] = bridge_stats["raw_l1"]
                    if torch.is_tensor(bridge_stats.get("weighted_l1")):
                        bridge_weighted_l1_terms[-1] = bridge_stats["weighted_l1"]
                    self._stage5_5_save_bridge_debug_maps(
                        step=step,
                        target_index=target_idx,
                        gt=gt,
                        pred_rgb=pred_rgb_live,
                        e_pred=e_pred,
                        support=support,
                        confidence=confidence,
                        bridge_mask=bridge_mask,
                        bridge_weight_map=bridge_weight_map,
                    )
                else:
                    skipped_low_active_bridge += 1

            if self.stage5_5_no_render_probe and self.stage5_5_no_render_probe_interval > 0 and step % self.stage5_5_no_render_probe_interval == 0:
                probe_inputs = [feat_tilde.permute(2, 0, 1), support_pos.unsqueeze(0)]
                if self.stage5_5_concat_log_support:
                    probe_inputs.append(support_log.unsqueeze(0))
                if self.stage5_5_concat_valid_mask:
                    probe_inputs.append(mask_loss.unsqueeze(0))
                if self.stage5_5_use_render_rgb:
                    probe_inputs.append(torch.zeros_like(render_ctx_rgb).permute(2, 0, 1))
                if self.stage5_5_use_render_alpha:
                    probe_inputs.append(torch.zeros_like(render_ctx_alpha).unsqueeze(0))
                with torch.no_grad():
                    probe_head_out = self.stage5_5_uncertainty_head(torch.cat(probe_inputs, dim=0).unsqueeze(0))
                    e_pred_probe_raw, _r_pred_probe_raw = self._stage5_5_unpack_head_output(probe_head_out)
                    e_pred_probe = e_pred_probe_raw.squeeze(0).squeeze(0)
                    probe_loss = (self._charbonnier(e_pred_probe - e_gt) * mask_loss).sum() / denom_all
                    with_render_probe_loss = float(loss_all.detach().item())
                    no_render_probe_loss = float(probe_loss.detach().item())

        if processed == 0:
            return {"loss": out["loss"].new_zeros(()), "skipped_no_valid_aux": 1.0}

        loss_all_mean = torch.stack(loss_all_terms).mean()
        if len(loss_supported_terms) > 0:
            loss_sup_mean = torch.stack(loss_supported_terms).mean()
        else:
            loss_sup_mean = loss_all_mean * 0.0
        if len(loss_rgb_residual_terms) > 0:
            loss_rgb_residual_mean = torch.stack(loss_rgb_residual_terms).mean()
        else:
            loss_rgb_residual_mean = loss_all_mean * 0.0
        if len(loss_rgb_residual_supported_terms) > 0:
            loss_rgb_residual_sup_mean = torch.stack(loss_rgb_residual_supported_terms).mean()
        else:
            loss_rgb_residual_sup_mean = loss_all_mean * 0.0
        warm_scale = self._warmup_linear_value(
            step,
            start_value=float(self.stage5_5_start_weight_scale),
            end_value=float(self.stage5_5_end_weight_scale),
            warmup_steps=int(self.stage5_5_warmup_steps),
        )
        loss_raw = (
            float(self.stage5_5_loss_all_weight) * loss_all_mean
            + float(self.stage5_5_loss_supported_weight) * loss_sup_mean
            + float(getattr(self, "stage5_5_loss_rgb_residual_weight", 0.0)) * loss_rgb_residual_mean
            + float(getattr(self, "stage5_5_loss_rgb_residual_supported_weight", 0.0)) * loss_rgb_residual_sup_mean
        )
        loss_uncertainty = float(warm_scale) * loss_raw
        if len(bridge_loss_terms) > 0:
            loss_bridge = torch.stack(bridge_loss_terms).mean()
        else:
            loss_bridge = loss_all_mean * 0.0
        loss_total = loss_uncertainty + loss_bridge
        pred_mean = torch.stack(pred_means).mean()
        gt_mean = torch.stack(gt_means).mean()
        abs_err_mean = torch.stack(abs_err_means).mean()
        ece_like = (pred_mean - gt_mean).abs()
        corr_val = float(sum(corr_terms) / max(len(corr_terms), 1))
        support_mean = torch.stack(support_means).mean()
        support_max = torch.stack(support_maxs).max()
        support_valid_ratio = torch.stack(support_valid_ratios).mean()
        loss_mask_ratio = torch.stack(loss_mask_ratios).mean()
        supported_loss_mask_ratio = torch.stack(supported_mask_ratios).mean()
        zero_stat = loss_all_mean.detach() * 0.0
        residual_abs_err_mean = (
            torch.stack(residual_abs_err_means).mean() if len(residual_abs_err_means) > 0 else zero_stat
        )
        residual_pred_abs_mean = (
            torch.stack(residual_pred_abs_means).mean() if len(residual_pred_abs_means) > 0 else zero_stat
        )
        residual_gt_abs_mean = (
            torch.stack(residual_gt_abs_means).mean() if len(residual_gt_abs_means) > 0 else zero_stat
        )
        bridge_raw_l1 = torch.stack(bridge_raw_l1_terms).mean() if len(bridge_raw_l1_terms) > 0 else zero_stat
        bridge_weighted_l1 = (
            torch.stack(bridge_weighted_l1_terms).mean() if len(bridge_weighted_l1_terms) > 0 else zero_stat
        )
        bridge_conf_mean = torch.stack(bridge_conf_means).mean() if len(bridge_conf_means) > 0 else zero_stat
        bridge_conf_p10 = torch.stack(bridge_conf_p10s).mean() if len(bridge_conf_p10s) > 0 else zero_stat
        bridge_conf_p50 = torch.stack(bridge_conf_p50s).mean() if len(bridge_conf_p50s) > 0 else zero_stat
        bridge_conf_p90 = torch.stack(bridge_conf_p90s).mean() if len(bridge_conf_p90s) > 0 else zero_stat
        bridge_error_conf_mean = (
            torch.stack(bridge_error_conf_means).mean() if len(bridge_error_conf_means) > 0 else zero_stat
        )
        bridge_support_conf_mean = (
            torch.stack(bridge_support_conf_means).mean() if len(bridge_support_conf_means) > 0 else zero_stat
        )
        bridge_active_ratio = torch.stack(bridge_active_ratios).mean() if len(bridge_active_ratios) > 0 else zero_stat
        bridge_weight_sum = torch.stack(bridge_weight_sums).mean() if len(bridge_weight_sums) > 0 else zero_stat

        out_pack: Dict[str, Any] = {
            "loss": loss_total,
            "loss_uncertainty": loss_uncertainty,
            "loss_bridge": loss_bridge,
            "loss_all": loss_all_mean,
            "loss_support": loss_sup_mean,
            "loss_rgb_residual": loss_rgb_residual_mean,
            "loss_rgb_residual_support": loss_rgb_residual_sup_mean,
            "effective_weight": float(warm_scale),
            "bridge_effective_weight": float(bridge_weight_value),
            "bridge_raw_l1": bridge_raw_l1,
            "bridge_weighted_l1": bridge_weighted_l1,
            "bridge_conf_mean": bridge_conf_mean,
            "bridge_conf_p10": bridge_conf_p10,
            "bridge_conf_p50": bridge_conf_p50,
            "bridge_conf_p90": bridge_conf_p90,
            "bridge_error_conf_mean": bridge_error_conf_mean,
            "bridge_support_conf_mean": bridge_support_conf_mean,
            "bridge_active_ratio": bridge_active_ratio,
            "bridge_weight_sum": bridge_weight_sum,
            "bridge_render_l1_before_weight": bridge_raw_l1,
            "bridge_skipped_low_active_ratio": float(skipped_low_active_bridge / max(processed, 1)),
            "processed_targets": float(processed),
            "skipped_empty_supported": float(skipped_empty_supported),
            "e_gt_mean": gt_mean,
            "e_pred_mean": pred_mean,
            "e_abs_error": abs_err_mean,
            "rgb_residual_abs_error": residual_abs_err_mean,
            "rgb_residual_pred_abs_mean": residual_pred_abs_mean,
            "rgb_residual_gt_abs_mean": residual_gt_abs_mean,
            "e_corr": float(corr_val),
            "ece_like": ece_like,
            "support_mean": support_mean,
            "support_max": support_max,
            "support_valid_ratio": support_valid_ratio,
            "loss_mask_ratio": loss_mask_ratio,
            "supported_loss_mask_ratio": supported_loss_mask_ratio,
            "feature_render_calls": float(aux_feature_render_calls),
            "feature_splat_time_ms": float(aux_feature_splat_time_ms),
            "render_context_time_ms": float(aux_render_context_time_ms),
            "total_render_time_ms": float(aux_feature_splat_time_ms + aux_render_context_time_ms),
        }
        if aux_debug_maps is not None:
            out_pack["_stage5_5_aux_debug_maps"] = aux_debug_maps
        if with_render_probe_loss is not None and no_render_probe_loss is not None:
            out_pack["loss_with_render_ctx"] = float(with_render_probe_loss)
            out_pack["loss_no_render_ctx_probe"] = float(no_render_probe_loss)
            out_pack["render_ctx_gain"] = float(no_render_probe_loss - with_render_probe_loss)
        return out_pack

    @staticmethod
    def _param_grad_norm(params: Sequence[nn.Parameter]) -> float:
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())
        return float(total ** 0.5)

    def _capture_stage5_5_scalar_logs(self, out: Dict[str, Any]) -> None:
        logs: Dict[str, float] = {}
        prefixes = (
            "loss/",
            "monitor/",
            "perf/aux_",
            "loss_stage5_5",
        )
        for key, val in out.items():
            ks = str(key)
            if not any(ks.startswith(prefix) for prefix in prefixes):
                continue
            if torch.is_tensor(val):
                if int(val.numel()) == 1:
                    logs[ks] = float(val.detach().item())
            elif isinstance(val, bool):
                logs[ks] = float(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                logs[ks] = float(val)
        self._stage5_5_last_scalar_logs = logs

    def _apply_nearby_direct_loss(
        self,
        *,
        batch: Dict[str, Any],
        out: Dict[str, Any],
        nearby_pack: Dict[str, Any],
    ) -> None:
        nearby_loss = nearby_pack.get("loss")
        if not torch.is_tensor(nearby_loss):
            return
        step = int(self._current_loss_step(batch))
        near_den = nearby_pack.get("denominator")
        if not torch.is_tensor(near_den):
            near_den = nearby_loss.new_tensor(float(nearby_pack.get("processed_targets", 0.0)))
        main_den = self._stage5_5_main_loss_denominator(batch, step, nearby_loss)
        combined_den = (main_den + near_den.to(device=nearby_loss.device, dtype=nearby_loss.dtype)).clamp_min(1.0e-8)
        if float(near_den.detach().item()) > 0.0:
            main_scale = main_den / combined_den
            near_scale = near_den.to(device=nearby_loss.device, dtype=nearby_loss.dtype) / combined_den
            out["loss"] = out["loss"] * main_scale + nearby_loss * near_scale
            if torch.is_tensor(out.get("loss_l1")) and torch.is_tensor(nearby_pack.get("loss_rgb")):
                out["loss_l1"] = out["loss_l1"] * main_scale + nearby_pack["loss_rgb"].to(out["loss_l1"].device) * near_scale
            if torch.is_tensor(out.get("loss_ssim")) and torch.is_tensor(nearby_pack.get("loss_ssim")):
                out["loss_ssim"] = out["loss_ssim"] * main_scale + nearby_pack["loss_ssim"].to(out["loss_ssim"].device) * near_scale
            if torch.is_tensor(out.get("loss_mask")):
                out["loss_mask"] = out["loss_mask"] * main_scale
            if torch.is_tensor(out.get("loss_opacity_entropy")):
                out["loss_opacity_entropy"] = out["loss_opacity_entropy"] * main_scale

        out["loss_stage5_5_nearby_direct"] = float(nearby_loss.detach().item())
        if torch.is_tensor(nearby_pack.get("loss_rgb")):
            out["loss/rgb/nearby_direct"] = float(nearby_pack["loss_rgb"].detach().item())
        else:
            out["loss/rgb/nearby_direct"] = 0.0
        if torch.is_tensor(nearby_pack.get("loss_ssim")):
            out["loss/ssim/nearby_direct"] = float(nearby_pack["loss_ssim"].detach().item())
        else:
            out["loss/ssim/nearby_direct"] = 0.0
        out["loss/target_weight/nearby_direct"] = float(nearby_pack.get("view_weight", 0.0))
        if torch.is_tensor(nearby_pack.get("monitor_psnr")):
            out["monitor/psnr/nearby_direct"] = float(nearby_pack["monitor_psnr"].detach().item())
        else:
            out["monitor/psnr/nearby_direct"] = 0.0
        if torch.is_tensor(nearby_pack.get("monitor_l1")):
            out["monitor/l1/nearby_direct"] = float(nearby_pack["monitor_l1"].detach().item())
        else:
            out["monitor/l1/nearby_direct"] = 0.0
        if torch.is_tensor(nearby_pack.get("valid_mask_ratio")):
            out["monitor/nearby_direct/valid_mask_ratio"] = float(nearby_pack["valid_mask_ratio"].detach().item())
        out["monitor/nearby_direct/processed_targets"] = float(nearby_pack.get("processed_targets", 0.0))
        if nearby_pack.get("skipped_empty"):
            out["monitor/nearby_direct/skipped_empty_aux_list"] = 1.0
        if nearby_pack.get("skipped_zero_weight"):
            out["monitor/nearby_direct/skipped_zero_weight"] = 1.0
        if nearby_pack.get("skipped_low_valid"):
            out["monitor/nearby_direct/skipped_low_valid"] = float(nearby_pack.get("skipped_low_valid", 0.0))

    def forward(self, batch: Dict) -> Dict[str, Any]:
        self._stage5_5_last_scalar_logs = {}
        out = super().forward(batch)
        if self.training and bool(getattr(self, "stage5_5_nearby_direct_enabled", False)):
            nearby_pack = self._compute_nearby_direct_loss_from_aux_targets(batch=batch, out=out)
            self._apply_nearby_direct_loss(batch=batch, out=out, nearby_pack=nearby_pack)
        if not self.training or not bool(self.stage5_5_aux_enabled):
            self._capture_stage5_5_scalar_logs(out)
            return out
        aux_pack = self._compute_feature_splat_uncertainty_loss(batch=batch, out=out)
        aux_loss = aux_pack.get("loss")
        if isinstance(aux_pack.get("_stage5_5_aux_debug_maps"), dict):
            out["_stage5_5_aux_debug_maps"] = aux_pack["_stage5_5_aux_debug_maps"]
        if torch.is_tensor(aux_loss):
            out["loss"] = out["loss"] + aux_loss
            if bool(aux_loss.requires_grad) and out.get("proxies") is not None:
                out["_retain_graph_for_proxy_backward"] = True
        out["loss_main"] = float((out["loss"] - aux_loss).detach().item()) if torch.is_tensor(aux_loss) else float(out["loss"].detach().item())
        out["loss_stage5_5_aux"] = float(aux_loss.detach().item()) if torch.is_tensor(aux_loss) else float(aux_loss or 0.0)
        aux_uncertainty_loss = aux_pack.get("loss_uncertainty", aux_loss)
        aux_bridge_loss = aux_pack.get("loss_bridge")
        out["loss_stage5_5_aux_uncertainty"] = (
            float(aux_uncertainty_loss.detach().item())
            if torch.is_tensor(aux_uncertainty_loss)
            else float(aux_uncertainty_loss or 0.0)
        )
        out["loss_stage5_5_aux_bridge"] = (
            float(aux_bridge_loss.detach().item()) if torch.is_tensor(aux_bridge_loss) else float(aux_bridge_loss or 0.0)
        )
        if torch.is_tensor(aux_pack.get("loss_all")):
            out["loss/aux_uncertainty/all_valid"] = float(aux_pack["loss_all"].detach().item())
        if torch.is_tensor(aux_pack.get("loss_support")):
            out["loss/aux_uncertainty/supported"] = float(aux_pack["loss_support"].detach().item())
        if torch.is_tensor(aux_pack.get("loss_rgb_residual")):
            out["loss/aux_uncertainty/rgb_residual"] = float(aux_pack["loss_rgb_residual"].detach().item())
        if torch.is_tensor(aux_pack.get("loss_rgb_residual_support")):
            out["loss/aux_uncertainty/rgb_residual_supported"] = float(
                aux_pack["loss_rgb_residual_support"].detach().item()
            )
        out["loss/aux_uncertainty/total"] = float(out["loss_stage5_5_aux_uncertainty"])
        out["loss/aux_uncertainty/effective_weight"] = float(aux_pack.get("effective_weight", 0.0))
        out["loss/aux_bridge/total"] = float(out["loss_stage5_5_aux_bridge"])
        out["loss/aux_bridge/effective_weight"] = float(aux_pack.get("bridge_effective_weight", 0.0))
        for key, out_key in (
            ("bridge_raw_l1", "loss/aux_bridge/raw_l1"),
            ("bridge_weighted_l1", "loss/aux_bridge/weighted_l1"),
        ):
            val = aux_pack.get(key)
            if torch.is_tensor(val):
                out[out_key] = float(val.detach().item())
            elif isinstance(val, (int, float)):
                out[out_key] = float(val)
        out["monitor/aux_uncertainty/processed_targets"] = float(aux_pack.get("processed_targets", 0.0))
        out["monitor/aux_uncertainty/skipped_empty"] = float(aux_pack.get("skipped_empty_supported", 0.0))
        out["monitor/aux_uncertainty/e_corr"] = float(aux_pack.get("e_corr", 0.0))
        for key in (
            "e_gt_mean",
            "e_pred_mean",
            "e_abs_error",
            "rgb_residual_abs_error",
            "rgb_residual_pred_abs_mean",
            "rgb_residual_gt_abs_mean",
            "ece_like",
            "support_mean",
            "support_max",
            "support_valid_ratio",
            "loss_mask_ratio",
            "supported_loss_mask_ratio",
        ):
            val = aux_pack.get(key)
            if torch.is_tensor(val):
                out[f"monitor/aux_uncertainty/{key}"] = float(val.detach().item())
            elif isinstance(val, (int, float)):
                out[f"monitor/aux_uncertainty/{key}"] = float(val)
        for key in (
            "bridge_conf_mean",
            "bridge_conf_p10",
            "bridge_conf_p50",
            "bridge_conf_p90",
            "bridge_error_conf_mean",
            "bridge_support_conf_mean",
            "bridge_active_ratio",
            "bridge_weight_sum",
            "bridge_render_l1_before_weight",
            "bridge_skipped_low_active_ratio",
        ):
            val = aux_pack.get(key)
            name = key[len("bridge_") :] if key.startswith("bridge_") else key
            if torch.is_tensor(val):
                out[f"monitor/aux_bridge/{name}"] = float(val.detach().item())
            elif isinstance(val, (int, float)):
                out[f"monitor/aux_bridge/{name}"] = float(val)
        if "loss_with_render_ctx" in aux_pack:
            out["monitor/aux_uncertainty/loss_with_render_ctx"] = float(aux_pack["loss_with_render_ctx"])
            out["monitor/aux_uncertainty/loss_no_render_ctx_probe"] = float(aux_pack["loss_no_render_ctx_probe"])
            out["monitor/aux_uncertainty/render_ctx_gain"] = float(aux_pack["render_ctx_gain"])
        out["perf/aux_feature_splat/render_calls"] = float(aux_pack.get("feature_render_calls", 0.0))
        out["perf/aux_feature_splat/time_ms"] = float(aux_pack.get("total_render_time_ms", 0.0))
        out["perf/aux_feature_splat/splat_time_ms"] = float(aux_pack.get("feature_splat_time_ms", 0.0))
        out["perf/aux_feature_splat/render_context_time_ms"] = float(aux_pack.get("render_context_time_ms", 0.0))
        if aux_pack.get("skipped_interval"):
            out["monitor/aux_uncertainty/skipped_interval"] = 1.0
        if aux_pack.get("skipped_no_valid_aux"):
            out["monitor/aux_uncertainty/skipped_no_valid_aux"] = 1.0
        if aux_pack.get("skipped_empty"):
            out["monitor/aux_uncertainty/skipped_empty_aux_list"] = 1.0
        self._capture_stage5_5_scalar_logs(out)
        return out

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._stage5_5_last_scalar_logs = {}
        out = super().train_step(
            batch=batch,
            step=step,
            profile_phase_timing=profile_phase_timing,
            sync_cuda_timing=sync_cuda_timing,
            scheduler_node_sync=scheduler_node_sync,
        )
        out.update(getattr(self, "_stage5_5_last_scalar_logs", {}))
        out["grad/aux_head_norm"] = self._param_grad_norm(list(self.stage5_5_uncertainty_head.parameters()))
        feat_params = [p for p in self.image_feature_extractor.parameters() if p.requires_grad]
        out["grad/feature_extractor_total_norm"] = self._param_grad_norm(feat_params)
        fusion_params = [
            p for n, p in self.image_feature_extractor.named_parameters() if p.requires_grad and ("fusion" in n or "fusion_neck" in n)
        ]
        out["grad/fusion_total_norm"] = self._param_grad_norm(fusion_params)
        return out


__all__ = ["MinimalStreetForwardStage5_5", "FeatureSplatUncertaintyHeadV2", "FeatureSplatUncertaintyHeadV3"]
