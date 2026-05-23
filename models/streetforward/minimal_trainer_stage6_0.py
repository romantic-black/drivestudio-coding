from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.streetforward.minimal_trainer_stage4_0 import spatial_hw_from_image_tensor
from models.streetforward.math_utils import _num_sh_bases
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0 import (
    CurrentContextAdapter,
    LocalGSState,
    Stage6EventEncoder,
    Stage6ParamEncoder,
    Stage6PosteriorUpdater,
    resolve_v9_phase_a_batch,
)
from models.streetforward.stage6_0.phase_a_losses import (
    delta_regularization,
    masked_rgb_loss,
    target_valid_mask,
)
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


class MinimalStreetForwardStage6_0(MinimalStreetForwardStage5_4):
    """
    Stage6_0 Phase A trainer.

    The class reuses Stage5_4's V4 source measurement helpers and renderer, but
    does not execute Stage5_3/5_4 recurrent update, history, gate, support EMA,
    or train_step paths.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        self._stage6_orig_config = config
        self._stage6_bootstrapping_parent = True
        parent_cfg = self._compat_stage5_4_config(config)
        try:
            super().__init__(config=parent_cfg, device=device, **kwargs)
        finally:
            self._stage6_bootstrapping_parent = False
        self.config = config
        self._validate_stage6_0_phase_a_config(config)
        self._configure_measurement_frontend_trainability(config)
        self._init_stage6_modules(config)
        self._rebuild_stage6_optimizer(config)

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get(key, default)
        if hasattr(node, "get"):
            value = node.get(key, default)
            return default if value is None else value
        if hasattr(node, key):
            value = getattr(node, key)
            return default if value is None else value
        return default

    def _compat_stage5_4_config(self, config: Any) -> Any:
        cfg = copy.deepcopy(config)
        model_cfg = self._require_key(cfg, "model", "config")
        model_cfg["stage"] = "5_4"
        hist = self._cfg_get(model_cfg, "history_memory", None)
        if hist is not None:
            hist["enable"] = True
        gate = self._cfg_get(model_cfg, "update_gate", None)
        if gate is not None:
            gate["enable"] = True
        view_transient = self._cfg_get(model_cfg, "view_transient", None)
        if view_transient is not None:
            view_transient["enable"] = True
        if self._cfg_get(cfg, "scheduler_v8", None) is None:
            sv9 = self._require_key(cfg, "scheduler_v9", "config")
            ep = self._require_key(sv9, "episode", "scheduler_v9")
            execution = self._cfg_get(sv9, "execution", {}) or {}
            block = self._cfg_get(sv9, "block", {}) or {}
            traversal = self._cfg_get(sv9, "traversal", {}) or {}
            preload = self._cfg_get(sv9, "preload", {}) or {}
            cfg["scheduler_v8"] = {
                "enable": True,
                "block": {
                    "steps_per_block": int(self._cfg_get(block, "steps_per_block", 1)),
                },
                "episode": {
                    "blocks_per_episode": int(self._require_key(ep, "blocks_per_episode", "scheduler_v9.episode")),
                    "total_target_frames": 1,
                    "include_source_frame": bool(self._cfg_get(ep, "include_source_frame", True)),
                    "target_policy": str(self._cfg_get(ep, "target_policy", "visited_episode_frames")),
                    "block_source_frame_policy": str(
                        self._cfg_get(ep, "block_source_frame_policy", "random_within_keyframe_per_visit")
                    ),
                    "frame_within_keyframe_policy": str(
                        self._cfg_get(ep, "frame_within_keyframe_policy", "random_once_per_episode")
                    ),
                    "min_keyframes_required_policy": str(
                        self._cfg_get(ep, "min_keyframes_required_policy", "skip_if_less_than_window")
                    ),
                },
                "traversal": {
                    "mode": str(self._cfg_get(traversal, "mode", "round_robin_episode_interleave")),
                    "switch_after_episode": bool(self._cfg_get(traversal, "switch_after_episode", True)),
                    "fixed_scene_id": self._cfg_get(traversal, "fixed_scene_id", None),
                    "fixed_segment_id": self._cfg_get(traversal, "fixed_segment_id", None),
                    "segment_order": str(self._cfg_get(traversal, "segment_order", "ascending")),
                    "scene_order": str(self._cfg_get(traversal, "scene_order", "shuffle_per_epoch")),
                },
                "execution": {
                    "block_order": str(self._cfg_get(execution, "block_order", "step_major")),
                    "step_major_switch_interval_steps": int(
                        self._cfg_get(execution, "step_major_switch_interval_steps", 1)
                    ),
                    "reset_policy": str(self._cfg_get(execution, "reset_policy", "episode_end")),
                },
                "preload": {
                    "emit_hints": bool(self._cfg_get(preload, "emit_hints", True)),
                    "warm_next_block_exact": bool(self._cfg_get(preload, "warm_next_block_exact", True)),
                    "warm_next_episode_chain": bool(self._cfg_get(preload, "warm_next_episode_chain", True)),
                },
            }
        return cfg

    def _validate_stage5_3_config(self, config) -> None:
        if bool(getattr(self, "_stage6_bootstrapping_parent", False)):
            return MinimalStreetForwardStage5_4._validate_stage5_3_config(self, config)

        self._validate_stage6_0_phase_a_config(config)

    def _validate_stage6_0_phase_a_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if str(self._require_key(model_cfg, "stage", "model")) != "6_0":
            raise ValueError("Stage6_0 requires model.stage='6_0'.")
        if str(self._cfg_get(model_cfg, "phase", "phase_A_block_local_unroll")) != "phase_A_block_local_unroll":
            raise ValueError("Stage6_0 Phase A requires model.phase=phase_A_block_local_unroll.")

        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        phase_a_mode = str(self._cfg_get(stage6, "phase_a_mode", "updater_only")).strip()
        if phase_a_mode not in {"updater_only", "from_scratch"}:
            raise ValueError("Stage6_0 Phase A requires phase_a_mode to be 'updater_only' or 'from_scratch'.")
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        if str(self._cfg_get(base_measurement, "type", "")) != "stage5_4_v4":
            raise ValueError("Stage6_0 Phase A requires base_measurement.type=stage5_4_v4.")
        if bool(self._cfg_get(base_measurement, "require_fused_v4", True)) is not True:
            raise ValueError("Stage6_0 Phase A requires fused V4; fallback is forbidden.")
        if bool(self._cfg_get(base_measurement, "require_obs_code", True)) is not True:
            raise ValueError("Stage6_0 Phase A requires V4 obs_code.")
        if int(self._cfg_get(base_measurement, "obs_code_dim", 2)) != 2:
            raise ValueError("Stage6_0 Phase A requires base_measurement.obs_code_dim=2.")
        source_grad_mode = str(self._cfg_get(base_measurement, "source_evidence_grad_mode", "no_grad_v4")).strip()
        detach_v4_outputs = bool(self._cfg_get(base_measurement, "detach_v4_outputs", True))
        train_2d_frontend = bool(self._cfg_get(base_measurement, "train_2d_frontend", False))
        train_residual_unet = bool(self._cfg_get(base_measurement, "train_residual_unet", train_2d_frontend))
        train_fusion_neck = bool(self._cfg_get(base_measurement, "train_fusion_neck", train_2d_frontend))
        train_v4_lift = bool(self._cfg_get(base_measurement, "train_v4_lift", False))
        train_dinov2 = bool(self._cfg_get(base_measurement, "train_dinov2", False))
        if train_v4_lift:
            raise ValueError("Stage6_0 Phase A P0 requires base_measurement.train_v4_lift=false.")
        if train_dinov2:
            raise ValueError("Stage6_0 Phase A P0 requires base_measurement.train_dinov2=false.")
        if phase_a_mode == "updater_only":
            if source_grad_mode != "no_grad_v4":
                raise ValueError(
                    "Stage6_0 Phase A updater_only requires source_evidence_grad_mode=no_grad_v4."
                )
            if not detach_v4_outputs:
                raise ValueError("Stage6_0 Phase A updater_only requires base_measurement.detach_v4_outputs=true.")
            if train_2d_frontend or train_residual_unet or train_fusion_neck:
                raise ValueError("Stage6_0 Phase A updater_only must keep the 2D frontend frozen.")
        else:
            if source_grad_mode != "train_2d_detach_alpha":
                raise ValueError(
                    "Stage6_0 Phase A from_scratch requires "
                    "base_measurement.source_evidence_grad_mode=train_2d_detach_alpha."
                )
            if detach_v4_outputs:
                raise ValueError("Stage6_0 Phase A from_scratch requires base_measurement.detach_v4_outputs=false.")
            if not train_2d_frontend:
                raise ValueError("Stage6_0 Phase A from_scratch requires base_measurement.train_2d_frontend=true.")
            if not train_residual_unet or not train_fusion_neck:
                raise ValueError(
                    "Stage6_0 Phase A from_scratch requires train_residual_unet=true and train_fusion_neck=true."
                )
        if bool(self._cfg_get(self._cfg_get(stage6, "vsm", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 Phase A must not enable VSM.")
        if bool(self._cfg_get(self._cfg_get(stage6, "query_decoder", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 Phase A must not enable QueryDecoder.")
        if bool(self._cfg_get(self._cfg_get(model_cfg, "history_memory", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 Phase A forbids model.history_memory.enable=true")
        if bool(self._cfg_get(self._cfg_get(model_cfg, "update_gate", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 Phase A forbids model.update_gate.enable=true")
        if bool(self._cfg_get(self._cfg_get(model_cfg, "view_transient", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 Phase A forbids model.view_transient.enable=true")

        sv9 = self._require_key(config, "scheduler_v9", "config")
        if bool(self._cfg_get(sv9, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase A requires scheduler_v9.enable=true.")
        if str(self._cfg_get(sv9, "phase", "")) != "phase_A_block_local_unroll":
            raise ValueError("Stage6_0 Phase A requires scheduler_v9.phase=phase_A_block_local_unroll.")
        if bool(self._cfg_get(self._cfg_get(config, "scheduler_v8", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 runtime must not enable scheduler_v8.")
        losses = self._require_key(config, "losses", "config")
        phase_a_losses = self._require_key(losses, "phase_a", "losses")
        disabled = self._cfg_get(phase_a_losses, "disabled", {}) or {}
        if bool(self._cfg_get(disabled, "query_observation", True)) is not True:
            raise ValueError("Phase A must disable query_observation.")
        if bool(self._cfg_get(disabled, "prefix_render", True)) is not True:
            raise ValueError("Phase A must disable prefix_render.")

        updater_cfg = self._cfg_get(stage6, "posterior_updater", {}) or {}
        branch_scope = self._cfg_get(updater_cfg, "branch_scope", {}) or {}
        distant_scope = self._cfg_get(branch_scope, "distant", {}) or {}
        if bool(self._cfg_get(distant_scope, "update_means", False)):
            raise ValueError("Stage6_0 Phase A P0 requires distant update_means=false.")
        if bool(self._cfg_get(distant_scope, "update_scales", False)):
            raise ValueError("Stage6_0 Phase A P0 requires distant update_scales=false.")
        if bool(self._cfg_get(distant_scope, "update_quat", False)):
            raise ValueError("Stage6_0 Phase A P0 requires distant update_quat=false.")

    def _configure_measurement_frontend_trainability(self, config: Any) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        self.stage6_phase_a_mode = str(self._cfg_get(stage6, "phase_a_mode", "updater_only")).strip()
        self.stage6_source_evidence_grad_mode = str(
            self._cfg_get(base_measurement, "source_evidence_grad_mode", "no_grad_v4")
        ).strip()
        self.stage6_detach_v4_outputs = bool(self._cfg_get(base_measurement, "detach_v4_outputs", True))
        self.stage6_measurement_trainable_param_names: set[str] = set()

        for param in self.parameters():
            param.requires_grad_(False)

        if self.stage6_phase_a_mode != "from_scratch":
            return

        image_feature_extractor = getattr(self, "image_feature_extractor", None)
        if image_feature_extractor is None:
            raise ValueError("Stage6_0 Phase A from_scratch requires image_feature_extractor.")

        train_residual_unet = bool(self._cfg_get(base_measurement, "train_residual_unet", True))
        train_fusion_neck = bool(self._cfg_get(base_measurement, "train_fusion_neck", True))

        def mark_trainable(module: nn.Module, prefix: str) -> None:
            for name, param in module.named_parameters(recurse=True):
                param.requires_grad_(True)
                self.stage6_measurement_trainable_param_names.add(f"{prefix}.{name}")

        if hasattr(image_feature_extractor, "residual_unet"):
            if train_residual_unet:
                mark_trainable(image_feature_extractor.residual_unet, "image_feature_extractor.residual_unet")
        elif train_residual_unet:
            mark_trainable(image_feature_extractor, "image_feature_extractor")

        if hasattr(image_feature_extractor, "fusion_neck"):
            if train_fusion_neck:
                mark_trainable(image_feature_extractor.fusion_neck, "image_feature_extractor.fusion_neck")
        elif train_fusion_neck and not hasattr(image_feature_extractor, "residual_unet"):
            mark_trainable(image_feature_extractor, "image_feature_extractor")

        if len(self.stage6_measurement_trainable_param_names) == 0:
            raise ValueError("Stage6_0 Phase A from_scratch did not enable any 2D frontend parameters.")

    def _init_stage6_modules(self, config: Any) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        event_cfg = self._cfg_get(stage6, "event_encoder", {}) or {}
        param_encoder_cfg = self._cfg_get(stage6, "param_encoder", {}) or {}
        ctx_cfg = self._cfg_get(stage6, "current_context_adapter", {}) or {}
        updater_cfg = self._cfg_get(stage6, "posterior_updater", {}) or {}
        clamp_cfg = self._cfg_get(updater_cfg, "clamps", {}) or {}
        self.stage6_view_code_policy = str(self._cfg_get(event_cfg, "view_code_policy", "zero_phase_a_debug"))
        if self.stage6_view_code_policy != "zero_phase_a_debug":
            raise ValueError("Stage6_0 Phase A P0 only supports event_encoder.view_code_policy=zero_phase_a_debug")
        self.stage6_event_dim = int(self._cfg_get(event_cfg, "output_dim", 96))
        self.stage6_ctx_dim = int(self._cfg_get(ctx_cfg, "ctx_dim", self.stage6_event_dim))
        self.stage6_hidden_dim = int(self._cfg_get(updater_cfg, "stage_hidden_dim", 32))
        self.stage6_param_encoder: Optional[Stage6ParamEncoder]
        param_encoder_enable = bool(self._cfg_get(param_encoder_cfg, "enable", True))
        if param_encoder_enable:
            sh_rest_input_dim = max(int(_num_sh_bases(int(self.sh_degree)) - 1), 0) * 3
            self.stage6_param_encoder = Stage6ParamEncoder(
                sh_rest_input_dim=sh_rest_input_dim,
                quat_scales_summary_dim=int(self._cfg_get(param_encoder_cfg, "quat_scales_summary_dim", 4)),
                sh_rest_summary_dim=int(self._cfg_get(param_encoder_cfg, "sh_rest_summary_dim", 8)),
                detach_inputs=bool(self._cfg_get(param_encoder_cfg, "detach_inputs", True)),
            ).to(self.device)
            param_embed_dim = int(self.stage6_param_encoder.output_dim)
        else:
            self.stage6_param_encoder = None
            param_embed_dim = int(self._cfg_get(event_cfg, "param_embed_dim", 10))
        inputs_cfg = self._cfg_get(event_cfg, "inputs", {}) or {}
        if not param_encoder_enable and self._cfg_get(inputs_cfg, "param_embed", None) is not None:
            param_embed_dim = int(self._cfg_get(event_cfg, "param_embed_dim", param_embed_dim))
        self.stage6_param_embed_dim = int(param_embed_dim)
        self.stage6_event_encoder = Stage6EventEncoder(
            z_dim=int(getattr(self, "stage5_2_feat_2d_channels", getattr(self, "feat_2d_channels", 32))),
            output_dim=self.stage6_event_dim,
            hidden_dim=int(self._cfg_get(event_cfg, "hidden_dim", 128)),
            num_layers=int(self._cfg_get(event_cfg, "num_layers", 2)),
            obs_code_dim=2,
            view_code_dim=2,
            param_embed_dim=self.stage6_param_embed_dim,
            branch_embed_dim=int(self._cfg_get(event_cfg, "branch_embed_dim", 8)),
            allow_missing_view_code=bool(self._cfg_get(event_cfg, "allow_zero_view_code_phase_a", False)),
        ).to(self.device)
        self.stage6_current_context_adapter = CurrentContextAdapter(
            event_dim=self.stage6_event_dim,
            ctx_dim=self.stage6_ctx_dim,
            hidden_dim=int(self._cfg_get(ctx_cfg, "hidden_dim", 128)),
        ).to(self.device)
        phase_b_hooks = self._cfg_get(updater_cfg, "phase_b_hooks", {}) or {}
        self.stage6_posterior_updater = Stage6PosteriorUpdater(
            event_dim=self.stage6_event_dim,
            ctx_dim=self.stage6_ctx_dim,
            hidden_dim=int(self._cfg_get(updater_cfg, "hidden_dim", 128)),
            stage_hidden_dim=self.stage6_hidden_dim,
            sh_degree=int(self.sh_degree),
            means_max_step_m=float(self._cfg_get(clamp_cfg, "means_max_step_m", 0.25)),
            scales_log_max_step=float(self._cfg_get(clamp_cfg, "scales_log_max_step", 0.08)),
            quat_axis_angle_max_step_rad=float(self._cfg_get(clamp_cfg, "quat_axis_angle_max_step_rad", 0.08)),
            opacity_logit_max_step=float(self._cfg_get(clamp_cfg, "opacity_logit_max_step", 0.25)),
            sh_max_step=float(self._cfg_get(clamp_cfg, "sh_max_step", 0.10)),
            hidden_max_step=float(self._cfg_get(clamp_cfg, "hidden_max_step", 1.0)),
            accept_vsm_ctx=bool(self._cfg_get(phase_b_hooks, "accept_vsm_ctx", True)),
            vsm_ctx_dim=int(self._cfg_get(phase_b_hooks, "vsm_ctx_dim", self.stage6_ctx_dim)),
        ).to(self.device)

        losses_cfg = self._cfg_get(config, "losses", {}) or {}
        phase_a = self._cfg_get(losses_cfg, "phase_a", {}) or {}
        block_render = self._cfg_get(phase_a, "block_render", {}) or {}
        nearby_render = self._cfg_get(phase_a, "nearby_render", {}) or {}
        regularization = self._cfg_get(phase_a, "regularization", {}) or {}
        self.stage6_block_weight = float(self._cfg_get(block_render, "weight", 1.0))
        self.stage6_step_gamma = float(self._cfg_get(block_render, "step_gamma", 0.8))
        self.stage6_block_mask_policy = str(self._cfg_get(block_render, "mask_policy", "non_sky_non_egocar"))
        self.stage6_nearby_enable = bool(self._cfg_get(nearby_render, "enable", True))
        self.stage6_nearby_weight = float(self._cfg_get(nearby_render, "weight", 0.25))
        self.stage6_nearby_warmup_steps = int(self._cfg_get(nearby_render, "warmup_steps", 2000))
        self.stage6_nearby_final_step_only = bool(self._cfg_get(nearby_render, "final_step_only", True))
        self.stage6_nearby_mask_policy = str(self._cfg_get(nearby_render, "mask_policy", "non_sky_non_egocar"))
        self.stage6_delta_l2_weight = float(self._cfg_get(regularization, "delta_l2_weight", 1.0e-3))
        self.stage6_writeback_policy = str(
            self._cfg_get(self._cfg_get(stage6, "local_rollout", {}) or {}, "writeback_policy", "block_end_detached")
        )
        self.stage6_branch_scope = self._parse_stage6_branch_scope(updater_cfg)

    def _parse_stage6_branch_scope(self, updater_cfg: Any) -> Dict[str, Dict[str, bool]]:
        raw = self._cfg_get(updater_cfg, "branch_scope", {}) or {}
        defaults = {
            "bg": {
                "update_means": True,
                "update_scales": True,
                "update_quat": True,
                "update_opacity": True,
                "update_sh": True,
            },
            "distant": {
                "update_means": False,
                "update_scales": False,
                "update_quat": False,
                "update_opacity": True,
                "update_sh": True,
            },
            "rigid": {
                "update_means": True,
                "update_scales": True,
                "update_quat": True,
                "update_opacity": True,
                "update_sh": True,
            },
        }
        out: Dict[str, Dict[str, bool]] = {}
        for branch, branch_defaults in defaults.items():
            cfg = self._cfg_get(raw, branch, {}) or {}
            out[branch] = {
                key: bool(self._cfg_get(cfg, key, default))
                for key, default in branch_defaults.items()
            }
        return out

    def _rebuild_stage6_optimizer(self, config: Any) -> None:
        opt_cfg = self._cfg_get(config, "optimizer", {}) or {}
        lr_cfg = self._cfg_get(opt_cfg, "lr", 1.0e-4)
        weight_decay = float(self._cfg_get(opt_cfg, "weight_decay", 0.0))
        betas = tuple(float(x) for x in list(self._cfg_get(opt_cfg, "betas", [0.9, 0.95])))
        eps = float(self._cfg_get(opt_cfg, "eps", 1.0e-8))

        def lr_for(name: str) -> float:
            if hasattr(lr_cfg, "get"):
                return float(lr_cfg.get(name, lr_cfg.get("default", 1.0e-4)))
            return float(lr_cfg)

        groups = [
            {
                "params": list(self.stage6_event_encoder.parameters()),
                "lr": lr_for("event_encoder"),
                "weight_decay": weight_decay,
                "logical_name": "stage6_event_encoder",
            },
            {
                "params": list(self.stage6_current_context_adapter.parameters()),
                "lr": lr_for("current_context_adapter"),
                "weight_decay": weight_decay,
                "logical_name": "stage6_current_context_adapter",
            },
            {
                "params": list(self.stage6_posterior_updater.parameters()),
                "lr": lr_for("posterior_updater"),
                "weight_decay": weight_decay,
                "logical_name": "stage6_posterior_updater",
            },
        ]
        if self.stage6_param_encoder is not None:
            groups.append(
                {
                    "params": list(self.stage6_param_encoder.parameters()),
                    "lr": lr_for("param_encoder"),
                    "weight_decay": weight_decay,
                    "logical_name": "stage6_param_encoder",
                }
            )
        measurement_params = [
            p
            for name, p in self.named_parameters()
            if name in getattr(self, "stage6_measurement_trainable_param_names", set()) and p.requires_grad
        ]
        measurement_lr = lr_for("measurement_frontend")
        if len(measurement_params) > 0:
            if float(measurement_lr) <= 0.0:
                raise ValueError(
                    "Stage6_0 Phase A from_scratch enabled trainable 2D frontend params but "
                    "optimizer.lr.measurement_frontend <= 0."
                )
            groups.append(
                {
                    "params": measurement_params,
                    "lr": measurement_lr,
                    "weight_decay": weight_decay,
                    "logical_name": "stage6_measurement_frontend",
                }
            )
        groups = [g for g in groups if float(g["lr"]) > 0.0 and len(g["params"]) > 0]
        if not groups:
            raise ValueError("Stage6_0 optimizer has no trainable parameter groups.")
        opt_type = str(self._cfg_get(opt_cfg, "type", "adamw")).lower()
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(groups, betas=betas, eps=eps)
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(groups, betas=betas, eps=eps)
        else:
            raise ValueError(f"Stage6_0 unsupported optimizer.type={opt_type!r}")

    def _nearby_weight(self, *, global_step: int, k: int, K: int) -> float:
        if not self.stage6_nearby_enable:
            return 0.0
        if self.stage6_nearby_final_step_only and int(k) != int(K) - 1:
            return 0.0
        warm = min(float(global_step) / max(int(self.stage6_nearby_warmup_steps), 1), 1.0)
        return float(self.stage6_nearby_weight) * warm

    def _source_subset(self, batch: Dict[str, Any], indices: List[int]) -> tuple[List[Any], List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        views = list(batch.get("source_views") or [])
        images = list(batch.get("source_images") or [])
        sky = batch.get("source_sky_masks") or batch.get("source_sky_mask")
        ego = batch.get("source_egocar_masks") or batch.get("source_egocar_mask")
        if len(views) == 0 or len(images) == 0:
            raise ValueError("Stage6_0 Phase A requires non-empty source_views/source_images.")
        sub_views = [views[int(i)] for i in indices]
        sub_images = [images[int(i)] for i in indices]
        sub_sky = [sky[int(i)] for i in indices] if sky is not None else None
        sub_ego = [ego[int(i)] for i in indices] if ego is not None else None
        return sub_views, sub_images, sub_sky, sub_ego

    def _local_to_node_states_detached(self, local_state: LocalGSState) -> tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        return local_state.to_node_states_detached()

    def _observe_v4_measurement(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
    ) -> Dict[str, Any]:
        grad_enabled = str(getattr(self, "stage6_source_evidence_grad_mode", "no_grad_v4")) != "no_grad_v4"
        ctx_mgr = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx_mgr:
            bg_m, distant_m, rigid_m = self._local_to_node_states_detached(local_state)
            route = None
            if rigid_m is not None:
                mask_src_rigid = self._rigid_point_valid_mask(rigid_m, int(source_frame_idx))
                S = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
                route = self._route_rigid_source_points(rigid_m, int(source_frame_idx), S)
            else:
                route = self._route_rigid_source_points(
                    NodeStateRigid(
                        means=torch.zeros((0, 3), device=self.device),
                        scales_log=torch.zeros((0, 3), device=self.device),
                        quats=torch.zeros((0, 4), device=self.device),
                        opacity_logit=torch.zeros((0, 1), device=self.device),
                        sh_dc=torch.zeros((0, 3), device=self.device),
                        sh_rest=torch.zeros((0, max(int(self.sh_degree + 1) ** 2 - 1, 0), 3), device=self.device),
                        point_ids=torch.zeros((0, 1), dtype=torch.long, device=self.device),
                        instances_quats=torch.zeros((0, 0, 4), device=self.device),
                        instances_trans=torch.zeros((0, 0, 3), device=self.device),
                        instances_fv=torch.zeros((0, 0), dtype=torch.bool, device=self.device),
                        instance_ids=[],
                        frame_ids=[],
                        cur_frame=0,
                    ),
                    int(source_frame_idx),
                    torch.zeros((0,), dtype=torch.long, device=self.device),
                )
            source_views, source_images, source_sky_masks, source_egocar_masks = self._source_subset(batch, source_indices)
            height, width = spatial_hw_from_image_tensor(source_images[0])
            one_pass = self._compute_2d_features_all_branches_once_routed(
                node_state_bg=bg_m,
                node_state_distant=distant_m,
                node_state_rigid=rigid_m,
                route=route,
                source_views=source_views,
                source_images=source_images,
                source_sky_masks=source_sky_masks,
                source_egocar_masks=source_egocar_masks,
                height=height,
                width=width,
            )
            obs_bg, obs_distant, obs_rigid_s = self._split_obs_code(
                num_bg=int(one_pass["num_bg"]),
                num_distant=int(one_pass["num_distant"]),
                num_rigid_s=int(route.S.numel()),
                device=self.device,
                dtype=one_pass["feat_2d_bg"].dtype,
            )
            return {
                **one_pass,
                "route": route,
                "obs_bg": obs_bg,
                "obs_distant": obs_distant,
                "obs_rigid_S": obs_rigid_s,
            }

    def _param_embed(
        self,
        branch: Optional[Any],
        n: int,
        dtype: torch.dtype,
        indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if branch is None:
            return None
        if int(n) == 0:
            return torch.zeros((0, self.stage6_param_embed_dim), device=self.device, dtype=dtype)
        if getattr(self, "stage6_param_encoder", None) is not None:
            out = self.stage6_param_encoder(
                branch=branch,
                indices=indices,
                aabb_min=getattr(self, "bbx_min", None),
                aabb_max=getattr(self, "bbx_max", None),
                dtype=dtype,
            )
            if int(out.shape[0]) != int(n):
                raise ValueError(f"param_embed row mismatch: got {int(out.shape[0])}, expected {int(n)}")
            if int(out.shape[1]) != int(self.stage6_param_embed_dim):
                raise ValueError(
                    "Stage6 param_embed dim mismatch: "
                    f"got {int(out.shape[1])}, expected {int(self.stage6_param_embed_dim)}"
                )
            return out
        raw = torch.cat(
            [
                branch.means,
                branch.scales_log,
                branch.opacity_logit,
                branch.sh_dc,
            ],
            dim=-1,
        )
        if indices is not None:
            raw = raw[indices]
        if int(raw.shape[0]) != int(n):
            raise ValueError(f"param_embed row mismatch: got {int(raw.shape[0])}, expected {int(n)}")
        if int(raw.shape[1]) != int(self.stage6_param_embed_dim):
            raise ValueError(
                "Stage6 param_embed dim mismatch: "
                f"got {int(raw.shape[1])}, expected {int(self.stage6_param_embed_dim)}"
            )
        return raw.detach()

    def _view_code(self, n: int, ref: torch.Tensor) -> torch.Tensor:
        if self.stage6_view_code_policy != "zero_phase_a_debug":
            raise ValueError(f"unsupported Stage6 view_code_policy={self.stage6_view_code_policy!r}")
        return ref.new_zeros((int(n), 2))

    @staticmethod
    def _mask_branch_delta(delta: BranchDelta, scope: Dict[str, bool]) -> BranchDelta:
        return BranchDelta(
            means=delta.means if bool(scope["update_means"]) else torch.zeros_like(delta.means),
            scales_log=delta.scales_log if bool(scope["update_scales"]) else torch.zeros_like(delta.scales_log),
            quat_axis_angle=(
                delta.quat_axis_angle
                if bool(scope["update_quat"])
                else torch.zeros_like(delta.quat_axis_angle)
            ),
            opacity_logit=(
                delta.opacity_logit
                if bool(scope["update_opacity"])
                else torch.zeros_like(delta.opacity_logit)
            ),
            sh=delta.sh if bool(scope["update_sh"]) else torch.zeros_like(delta.sh),
            hidden=delta.hidden,
            confidence=delta.confidence,
            noop=delta.noop,
        )

    def _apply_branch_scope(self, delta: DeltaPack) -> DeltaPack:
        return DeltaPack(
            bg=self._mask_branch_delta(delta.bg, self.stage6_branch_scope["bg"]),
            distant=(
                self._mask_branch_delta(delta.distant, self.stage6_branch_scope["distant"])
                if delta.distant is not None
                else None
            ),
            rigid=(
                self._mask_branch_delta(delta.rigid, self.stage6_branch_scope["rigid"])
                if delta.rigid is not None
                else None
            ),
            aux=delta.aux,
        )

    def _expand_branch_delta(self, delta: BranchDelta, *, indices: torch.Tensor, total: int) -> BranchDelta:
        if int(delta.means.shape[0]) == int(total) and int(indices.numel()) == int(total):
            return delta

        def fill(value: torch.Tensor) -> torch.Tensor:
            out = value.new_zeros((int(total),) + tuple(value.shape[1:]))
            if value.numel() > 0:
                out[indices] = value
            return out

        return BranchDelta(
            means=fill(delta.means),
            scales_log=fill(delta.scales_log),
            quat_axis_angle=fill(delta.quat_axis_angle),
            opacity_logit=fill(delta.opacity_logit),
            sh=fill(delta.sh),
            hidden=fill(delta.hidden),
            confidence=fill(delta.confidence),
            noop=fill(delta.noop),
        )

    def _encode_and_update(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        route = measurement["route"]
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))

        def maybe_detach_feature(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return x.detach() if detach_features else x

        def detach_alpha(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach()

        z_bg = maybe_detach_feature(measurement["feat_2d_bg"])
        acc_bg = detach_alpha(measurement["acc_w_bg"])
        obs_bg = detach_alpha(measurement["obs_bg"])
        z_distant = measurement.get("feat_2d_distant")
        acc_distant = measurement.get("acc_w_distant")
        obs_distant = measurement.get("obs_distant")
        z_rigid = measurement.get("feat_2d_rigid_S")
        acc_rigid = measurement.get("acc_w_rigid_S")
        obs_rigid = measurement.get("obs_rigid_S")
        z_distant = maybe_detach_feature(z_distant)
        acc_distant = detach_alpha(acc_distant)
        obs_distant = detach_alpha(obs_distant)
        z_rigid = maybe_detach_feature(z_rigid)
        acc_rigid = detach_alpha(acc_rigid)
        obs_rigid = detach_alpha(obs_rigid)

        event = self.stage6_event_encoder(
            z_bg=z_bg,
            acc_w_bg=acc_bg,
            obs_code_bg=obs_bg,
            view_code_bg=self._view_code(int(z_bg.shape[0]), z_bg),
            param_embed_bg=self._param_embed(local_state.bg, int(z_bg.shape[0]), z_bg.dtype),
            z_distant=z_distant,
            acc_w_distant=acc_distant,
            obs_code_distant=obs_distant,
            view_code_distant=self._view_code(int(z_distant.shape[0]), z_distant) if z_distant is not None else None,
            param_embed_distant=(
                self._param_embed(local_state.distant, int(z_distant.shape[0]), z_distant.dtype)
                if z_distant is not None
                else None
            ),
            z_rigid=z_rigid,
            acc_w_rigid=acc_rigid,
            obs_code_rigid=obs_rigid,
            view_code_rigid=self._view_code(int(z_rigid.shape[0]), z_rigid) if z_rigid is not None else None,
            param_embed_rigid=(
                self._param_embed(local_state.rigid, int(z_rigid.shape[0]), z_rigid.dtype, indices=route.S)
                if z_rigid is not None and local_state.rigid is not None
                else None
            ),
        )
        ctx = self.stage6_current_context_adapter(event)
        delta, aux = self.stage6_posterior_updater(event=event, ctx_current=ctx, ctx_vsm=None)
        if delta.rigid is not None and local_state.rigid is not None:
            delta = DeltaPack(
                bg=delta.bg,
                distant=delta.distant,
                rigid=self._expand_branch_delta(
                    delta.rigid,
                    indices=route.S,
                    total=int(local_state.rigid.means.shape[0]),
                ),
                aux=delta.aux,
            )
        delta = self._apply_branch_scope(delta)
        return local_state.apply_delta(delta), delta, {**event.aux, **(ctx.aux or {}), **aux}

    @staticmethod
    def _branch_render_params(branch: Any) -> Dict[str, torch.Tensor]:
        return {
            "means_r": branch.means,
            "scales_r": torch.exp(branch.scales_log),
            "quats_r": branch.quats,
            "opacities_r": torch.sigmoid(branch.opacity_logit).squeeze(-1),
            "colors_r": torch.cat([branch.sh_dc[:, None, :], branch.sh_rest], dim=1),
        }

    @staticmethod
    def _cat_render_params(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "means_r": torch.cat([p["means_r"] for p in parts], dim=0),
            "scales_r": torch.cat([p["scales_r"] for p in parts], dim=0),
            "quats_r": torch.cat([p["quats_r"] for p in parts], dim=0),
            "opacities_r": torch.cat([p["opacities_r"] for p in parts], dim=0),
            "colors_r": torch.cat([p["colors_r"] for p in parts], dim=0),
        }

    def _local_rigid_node_state(self, local_state: LocalGSState) -> Optional[NodeStateRigid]:
        if local_state.rigid is None or local_state.rigid_template is None:
            return None
        _, _, rigid = local_state.to_node_states_detached()
        if rigid is None:
            return None
        rigid.means = local_state.rigid.means
        rigid.scales_log = local_state.rigid.scales_log
        rigid.quats = local_state.rigid.quats
        rigid.opacity_logit = local_state.rigid.opacity_logit
        rigid.sh_dc = local_state.rigid.sh_dc
        rigid.sh_rest = local_state.rigid.sh_rest
        return rigid

    def _render_target(self, *, local_state: LocalGSState, target: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self._branch_render_params(local_state.bg)]
        frame_idx = int(target.get("frame_idx", 0))
        rigid_node = self._local_rigid_node_state(local_state)
        if local_state.rigid is not None and rigid_node is not None:
            valid = self._rigid_point_valid_mask(rigid_node, frame_idx)
            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                rigid_local_all = self._branch_render_params(local_state.rigid)
                rigid_local = {k: v[idx] for k, v in rigid_local_all.items()}
                point_ids = rigid_node.point_ids[idx, 0]
                parts.append(
                    self._rigid_local_to_world_render_params(
                        rigid_node,
                        rigid_local,
                        frame_idx,
                        point_ids_subset=point_ids,
                    )
                )
        if local_state.distant is not None:
            parts.append(self._branch_render_params(local_state.distant))
        gt = target["gt_image"]
        height, width = spatial_hw_from_image_tensor(gt)
        return self._render_single_view(self._cat_render_params(parts), target["view"], height, width)

    def _render_loss_for_indices(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        target_indices: List[int],
        mask_policy: str,
        pred_rgbs_out: Optional[List[torch.Tensor]] = None,
        gt_images_out: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if len(target_indices) == 0:
            return local_state.bg.means.new_tensor(0.0), {"num_refs": 0.0}
        losses: List[torch.Tensor] = []
        stats: Dict[str, float] = {"num_refs": float(len(target_indices))}
        psnr_vals: List[float] = []
        l1_vals: List[float] = []
        for idx in target_indices:
            target = batch["targets"][int(idx)]
            pred, _alpha = self._render_target(local_state=local_state, target=target)
            gt = target["gt_image"].to(device=pred.device, dtype=pred.dtype)
            mask = target_valid_mask(target, mask_policy=mask_policy, device=pred.device)
            loss_i, stat_i = masked_rgb_loss(
                pred,
                gt,
                mask=mask,
                l1_weight=1.0,
                ssim_weight=float(getattr(self, "loss_w_ssim", 0.0)),
            )
            if pred_rgbs_out is not None:
                pred_rgbs_out.append(pred.detach())
            if gt_images_out is not None:
                gt_images_out.append(gt.detach())
            losses.append(loss_i)
            psnr_vals.append(float(stat_i["psnr"]))
            l1_vals.append(float(stat_i["l1"]))
        stats["psnr"] = float(sum(psnr_vals) / max(len(psnr_vals), 1))
        stats["l1"] = float(sum(l1_vals) / max(len(l1_vals), 1))
        return torch.stack(losses).mean(), stats

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        roles = resolve_v9_phase_a_batch(batch)
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("Stage6_0 Phase A requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("Stage6_0 Phase A requires non-empty targets.")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=self.stage6_hidden_dim,
        )
        total_loss = local_state.bg.means.new_tensor(0.0)
        per_step: List[Dict[str, float]] = []
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        step = int(batch.get("global_step", 0) or 0)
        for k in range(roles.inner_K):
            evidence_refs = roles.evidence_refs_by_step[int(k)]
            source_frame_idx = int(evidence_refs[0][0])
            measurement = self._observe_v4_measurement(
                local_state=local_state,
                batch=batch,
                source_indices=roles.evidence_source_indices_by_step[int(k)],
                source_frame_idx=source_frame_idx,
            )
            local_state, delta, update_aux = self._encode_and_update(local_state=local_state, measurement=measurement)
            block_loss, block_stats = self._render_loss_for_indices(
                local_state=local_state,
                batch=batch,
                target_indices=roles.block_target_indices_by_step[int(k)],
                mask_policy=self.stage6_block_mask_policy,
                pred_rgbs_out=pred_rgbs if int(k) == roles.inner_K - 1 else None,
                gt_images_out=gt_images if int(k) == roles.inner_K - 1 else None,
            )
            nearby_loss, nearby_stats = self._render_loss_for_indices(
                local_state=local_state,
                batch=batch,
                target_indices=roles.nearby_target_indices_by_step[int(k)],
                mask_policy=self.stage6_nearby_mask_policy,
                pred_rgbs_out=pred_rgbs if int(k) == roles.inner_K - 1 else None,
                gt_images_out=gt_images if int(k) == roles.inner_K - 1 else None,
            )
            reg_loss, reg_stats = delta_regularization(delta, weight=self.stage6_delta_l2_weight)
            near_weight = self._nearby_weight(global_step=step, k=int(k), K=roles.inner_K)
            step_weight = float(self.stage6_step_gamma) ** float(roles.inner_K - 1 - int(k))
            loss_k = step_weight * (self.stage6_block_weight * block_loss + near_weight * nearby_loss + reg_loss)
            if not torch.isfinite(loss_k).all():
                raise RuntimeError("Stage6_0 Phase A loss became NaN/Inf.")
            total_loss = total_loss + loss_k
            per_step.append(
                {
                    "k": float(k),
                    "loss_block": float(block_loss.detach().item()),
                    "loss_nearby": float(nearby_loss.detach().item()),
                    "nearby_weight": float(near_weight),
                    "block_psnr": float(block_stats.get("psnr", 0.0)),
                    "nearby_psnr": float(nearby_stats.get("psnr", 0.0)),
                    **{k2: float(v) for k2, v in reg_stats.items()},
                    **{k2: float(v) for k2, v in update_aux.items() if isinstance(v, (int, float))},
                }
            )
        return {
            "loss": total_loss,
            "local_G": local_state,
            "node_state_bg": node_state_bg,
            "node_state_distant": node_state_distant,
            "node_state_rigid": node_state_rigid,
            "roles": roles,
            "per_step": per_step,
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
        }

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        _ = (profile_phase_timing, sync_cuda_timing, runtime_policy)
        batch = dict(batch)
        batch["global_step"] = int(step or 0)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self.forward(batch)
        loss = out["loss"]
        loss.backward()
        grad_norm = self._stage6_compute_and_check_grad_norm()
        self.optimizer.step()
        if self.stage6_writeback_policy == "block_end_detached":
            out["local_G"].writeback_detached(
                bg=out["node_state_bg"],
                distant=out["node_state_distant"],
                rigid=out["node_state_rigid"],
            )
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.reset_node_state()
        per_step = list(out.get("per_step") or [])
        final = per_step[-1] if per_step else {}
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phaseA/loss_total": float(loss.detach().item()),
            "stage6/phase": "A",
            "stage6/inner_K": float(out["roles"].inner_K),
            "num_targets": int(out.get("num_targets", 0)),
            "num_source_views": int(out.get("num_source_views", 0)),
            "pred_rgbs": list(out.get("pred_rgbs") or []),
            "gt_images": list(out.get("gt_images") or []),
            "num_gaussians_bg": int(out["node_state_bg"].means.shape[0]),
            "num_gaussians_distant": int(out["node_state_distant"].means.shape[0]) if out["node_state_distant"] is not None else 0,
            "num_gaussians_rigid": int(out["node_state_rigid"].means.shape[0]) if out["node_state_rigid"] is not None else 0,
            "phaseA/loss_block_final": float(final.get("loss_block", 0.0)),
            "phaseA/loss_nearby_final": float(final.get("loss_nearby", 0.0)),
            "phaseA/block_psnr_final": float(final.get("block_psnr", 0.0)),
            "phaseA/nearby_psnr_final": float(final.get("nearby_psnr", 0.0)),
            "phaseA/grad_norm_total": float(grad_norm.detach().item()),
        }
        for item in per_step:
            k = int(item["k"])
            logs[f"phaseA/loss_block_k{k}"] = float(item.get("loss_block", 0.0))
            logs[f"phaseA/loss_nearby_k{k}"] = float(item.get("loss_nearby", 0.0))
            logs[f"phaseA/block_psnr_k{k}"] = float(item.get("block_psnr", 0.0))
        return logs

    def _stage6_params_with_grads(self) -> List[torch.nn.Parameter]:
        return [
            p
            for p in self.parameters()
            if p.requires_grad and p.grad is not None
        ]

    def _stage6_compute_and_check_grad_norm(self) -> torch.Tensor:
        params = self._stage6_params_with_grads()
        training_cfg = self._cfg_get(self.config, "training", {}) or {}
        grad_clip_cfg = self._cfg_get(training_cfg, "grad_clip", {}) or {}
        bad_step_cfg = self._cfg_get(training_cfg, "bad_step", {}) or {}
        max_norm = float(self._cfg_get(grad_clip_cfg, "max_norm", 1.0))
        if len(params) == 0:
            ref_param = next(self.stage6_event_encoder.parameters())
            total_norm = ref_param.new_tensor(0.0)
        elif bool(self._cfg_get(grad_clip_cfg, "enable", True)):
            total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
            if not torch.is_tensor(total_norm):
                total_norm = params[0].grad.new_tensor(float(total_norm))  # type: ignore[union-attr]
        else:
            norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2) if norms else params[0].new_tensor(0.0)

        if bool(self._cfg_get(bad_step_cfg, "fail_on_nonfinite_grad", True)) and not torch.isfinite(total_norm):
            raise RuntimeError(f"non-finite Stage6_0 gradient norm: {float(total_norm.detach().cpu().item())}")
        fail_gt = float(self._cfg_get(bad_step_cfg, "fail_on_grad_norm_gt", 0.0))
        if fail_gt > 0.0 and torch.isfinite(total_norm) and float(total_norm.detach().cpu().item()) > fail_gt:
            raise RuntimeError(
                "Stage6_0 gradient norm exceeded configured fail_on_grad_norm_gt: "
                f"{float(total_norm.detach().cpu().item()):.6g} > {fail_gt:.6g}"
            )
        return total_norm.detach()

    def build_phase_b_export_checkpoint(self) -> Dict[str, Any]:
        normalizer_stats = {}
        if hasattr(self.stage6_event_encoder, "state_dict"):
            normalizer_stats["event_encoder_buffers"] = {
                k: v.detach().cpu()
                for k, v in self.stage6_event_encoder.state_dict().items()
                if "running" in k or "normalizer" in k
            }
        measurement_prefixes = (
            "image_feature_extractor.",
            "feature_backprojector.",
            "alpha_t_extractor_v4.",
            "current_obs_",
        )
        return {
            "export_type": "stage6_0_phase_a_for_phase_b",
            "measurement_frontend": {
                k: v.detach().cpu()
                for k, v in self.state_dict().items()
                if k.startswith(measurement_prefixes)
            },
            "event_encoder": self.stage6_event_encoder.state_dict(),
            "posterior_updater_base": self.stage6_posterior_updater.base_state_dict(),
            "current_context_adapter": self.stage6_current_context_adapter.state_dict(),
            "normalizer_stats": normalizer_stats,
            "phase_b_init_policy": {
                "freeze_measurement_frontend": True,
                "freeze_event_encoder": True,
                "freeze_posterior_updater_base": True,
                "init_vsm_tokens": "zeros_or_small_random",
                "init_query_decoder": "new",
                "init_vsm_context_adapter": "zero_last",
                "residual_scale_init": 0.0,
            },
        }

    def build_light_checkpoint_extra(self, *, step: int) -> Dict[str, Any]:
        return {
            "model_stage": "6_0",
            "phase": "phase_A_block_local_unroll",
            "global_step": int(step),
        }


__all__ = ["MinimalStreetForwardStage6_0"]
