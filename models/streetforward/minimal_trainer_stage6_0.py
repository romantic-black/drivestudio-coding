from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.minimal_trainer_stage4_0 import spatial_hw_from_image_tensor
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders.common import cat_param_dict
from models.streetforward.stage6_0 import (
    LocalGSState,
    Stage6PosteriorUpdater,
    Stage6RoutedStructEventDecoder,
    Stage6StructInput,
    empty_stage6_struct_input,
    resolve_v9_phase_a_batch,
    stage6_to_struct_decoder_input,
)
from models.streetforward.stage6_0.phase_a_losses import (
    delta_regularization,
    masked_rgb_loss,
    target_valid_mask,
)
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


logger = logging.getLogger(__name__)


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

    @staticmethod
    def _mem_debug_enabled() -> bool:
        return str(os.environ.get("STAGE6_MEM_DEBUG", "")).lower() in {"1", "true", "yes", "on"}

    def _mem_debug(self, label: str, **extra: Any) -> None:
        if not self._mem_debug_enabled() or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        alloc = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
        reserved = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
        peak = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
        extras = " ".join(f"{k}={v}" for k, v in extra.items())
        logger.info("STAGE6_MEM %s alloc_gb=%.3f reserved_gb=%.3f peak_gb=%.3f %s", label, alloc, reserved, peak, extras)

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
        optimizer_cfg = self._cfg_get(cfg, "optimizer", None)
        if optimizer_cfg is not None:
            raw_lr = self._cfg_get(optimizer_cfg, "lr", 1.0e-3)
            if hasattr(raw_lr, "get") and not isinstance(raw_lr, (str, bytes)):
                raw_lr = self._cfg_get(raw_lr, "default", self._cfg_get(raw_lr, "measurement_frontend", 1.0e-3))
            optimizer_cfg["lr"] = float(raw_lr)
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
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        phase_a_mode = str(
            self._cfg_get(stage6, "phase_a_mode", self._cfg_get(base_measurement, "mode", "updater_only"))
        ).strip()
        if phase_a_mode not in {"updater_only", "from_scratch"}:
            raise ValueError("Stage6_0 Phase A requires phase_a_mode to be 'updater_only' or 'from_scratch'.")
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
        struct_cfg = self._require_key(stage6, "struct_event_decoder", "model.stage6_0")
        if bool(self._cfg_get(struct_cfg, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase A requires struct_event_decoder.enable=true.")
        event_cfg = self._cfg_get(stage6, "event_encoder", {}) or {}
        if bool(self._cfg_get(event_cfg, "enable", False)):
            raise ValueError("Stage6_0 Phase A forbids direct concat EventEncoder as the main event path.")
        if str(self._cfg_get(event_cfg, "mode", "disabled_direct_concat_mlp")) == "direct_concat_mlp":
            raise ValueError("Stage6_0 Phase A forbids direct concat EventEncoder as the main event path.")
        near_cfg = self._cfg_get(struct_cfg, "near", {}) or {}
        far_cfg = self._cfg_get(struct_cfg, "far", {}) or {}
        token_cfg = self._cfg_get(struct_cfg, "token", {}) or {}
        for token_key in ("zero_invalid_2d_feat", "use_2d_feat", "use_support", "use_branch_embed", "use_param_obs_embed"):
            if bool(self._cfg_get(token_cfg, token_key, True)) is not True:
                raise ValueError(f"Stage6_0 Phase A P0 requires struct_event_decoder.token.{token_key}=true.")
        codec_cfg = self._cfg_get(struct_cfg, "param_obs_codec", {}) or {}
        if bool(self._cfg_get(codec_cfg, "enable", True)) is not True:
            raise ValueError("Stage6_0 Phase A requires param_obs_codec.enable=true.")
        if str(self._cfg_get(codec_cfg, "raw_param_mode", "stage5_normalize_params_17")) != "stage5_normalize_params_17":
            raise ValueError("Stage6_0 Phase A P0 requires param_obs_codec.raw_param_mode=stage5_normalize_params_17.")
        if str(self._cfg_get(near_cfg, "type", "xcpe")) != "xcpe":
            raise ValueError("Stage6_0 Phase A requires struct_event_decoder.near.type=xcpe.")
        if str(self._cfg_get(far_cfg, "type", "point_mlp")) not in {"point_mlp", "mlp"}:
            raise ValueError("Stage6_0 Phase A requires struct_event_decoder.far.type=point_mlp.")
        token_dim = int(self._cfg_get(token_cfg, "token_dim", 48))
        event_dim = int(self._cfg_get(struct_cfg, "event_dim", self._cfg_get(self._cfg_get(stage6, "posterior_updater", {}) or {}, "event_dim", 48)))
        if int(event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6_0 Phase A requires struct_event_decoder.event_dim==token.token_dim, "
                f"got event_dim={int(event_dim)} token_dim={int(token_dim)}."
            )
        ctx_cfg = self._cfg_get(stage6, "current_context_adapter", {}) or {}
        if bool(self._cfg_get(ctx_cfg, "enable", False)):
            raise ValueError("Phase A should not enable current_context_adapter; use event_only updater.")
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
        if bool(self._cfg_get(updater_cfg, "input_current_ctx", False)):
            raise ValueError("Stage6_0 Phase A requires posterior_updater.input_current_ctx=false.")
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
        self.stage6_phase_a_mode = str(
            self._cfg_get(stage6, "phase_a_mode", self._cfg_get(base_measurement, "mode", "updater_only"))
        ).strip()
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
        struct_cfg = self._require_key(stage6, "struct_event_decoder", "model.stage6_0")
        token_cfg = self._cfg_get(struct_cfg, "token", {}) or {}
        codec_cfg = self._cfg_get(struct_cfg, "param_obs_codec", {}) or {}
        near_cfg = self._cfg_get(struct_cfg, "near", {}) or {}
        far_cfg = self._cfg_get(struct_cfg, "far", {}) or {}
        updater_cfg = self._cfg_get(stage6, "posterior_updater", {}) or {}
        clamp_cfg = self._cfg_get(updater_cfg, "clamps", {}) or {}
        token_dim = int(self._cfg_get(token_cfg, "token_dim", 48))
        self.stage6_event_dim = int(self._cfg_get(struct_cfg, "event_dim", self._cfg_get(updater_cfg, "event_dim", token_dim)))
        if int(self.stage6_event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6_0 struct_event_decoder.event_dim must equal token.token_dim ({int(token_dim)}), "
                f"got {int(self.stage6_event_dim)}."
            )
        self.stage6_ctx_dim = self.stage6_event_dim
        self.stage6_hidden_dim = int(self._cfg_get(updater_cfg, "stage_hidden_dim", self.stage6_event_dim))
        self.stage6_feat_2d_dim = int(
            self._cfg_get(
                struct_cfg,
                "feat_2d_dim",
                getattr(self, "stage5_2_feat_2d_channels", getattr(self, "feat_2d_channels", 32)),
            )
        )
        if int(self.stage6_feat_2d_dim) != int(getattr(self, "stage5_2_feat_2d_channels", self.stage6_feat_2d_dim)):
            raise ValueError(
                "Stage6_0 Phase A P0 expects struct_event_decoder.feat_2d_dim to match the V4 lifted feature dim."
            )
        param_obs_cfg = {
            "obs_code_dim": int(self._cfg_get(codec_cfg, "obs_code_dim", 2)),
            "support_dim": int(self._cfg_get(codec_cfg, "support_dim", 2)),
            "branch_embed_dim": int(self._cfg_get(codec_cfg, "branch_embed_dim", 4)),
            "output_dim": int(self._cfg_get(codec_cfg, "output_dim", 24)),
            "detach_params": bool(self._cfg_get(codec_cfg, "detach_params", True)),
            "detach_obs_code": bool(self._cfg_get(codec_cfg, "detach_obs_code", True)),
            "detach_acc_w": bool(self._cfg_get(codec_cfg, "detach_acc_w", True)),
            "norm": str(self._cfg_get(codec_cfg, "norm", "layernorm")),
            "activation": str(self._cfg_get(codec_cfg, "activation", "gelu")),
        }
        self.stage6_struct_event_decoder = Stage6RoutedStructEventDecoder(
            feat_2d_dim=int(self.stage6_feat_2d_dim),
            event_dim=int(self.stage6_event_dim),
            token_dim=token_dim,
            param_obs_dim=int(param_obs_cfg["output_dim"]),
            support_embed_dim=int(self._cfg_get(struct_cfg, "support_embed_dim", self._cfg_get(token_cfg, "support_embed_dim", 4))),
            branch_embed_dim=int(self._cfg_get(struct_cfg, "branch_embed_dim", self._cfg_get(token_cfg, "branch_embed_dim", 4))),
            near_num_blocks=int(self._cfg_get(near_cfg, "num_blocks", 2)),
            near_kernel_size=int(self._cfg_get(near_cfg, "kernel_size", 3)),
            near_voxel_size=float(self._cfg_get(near_cfg, "voxel_size", 0.25)),
            near_residual_scale_init=float(self._cfg_get(near_cfg, "residual_scale_init", 5.0e-3)),
            near_sparse_backend=str(self._cfg_get(near_cfg, "sparse_backend", "spconv")),
            far_hidden_dim=int(self._cfg_get(far_cfg, "hidden_dim", self.stage6_event_dim)),
            far_num_layers=int(self._cfg_get(far_cfg, "num_layers", 2)),
            param_obs_codec_cfg=param_obs_cfg,
        ).to(self.device)
        phase_b_hooks = self._cfg_get(updater_cfg, "phase_b_hooks", {}) or {}
        self.stage6_posterior_updater = Stage6PosteriorUpdater(
            event_dim=self.stage6_event_dim,
            ctx_dim=self.stage6_event_dim,
            hidden_dim=int(self._cfg_get(updater_cfg, "hidden_dim", 96)),
            stage_hidden_dim=self.stage6_hidden_dim,
            sh_degree=int(self.sh_degree),
            means_max_step_m=float(self._cfg_get(clamp_cfg, "means_max_step_m", 0.25)),
            scales_log_max_step=float(self._cfg_get(clamp_cfg, "scales_log_max_step", 0.08)),
            quat_axis_angle_max_step_rad=float(self._cfg_get(clamp_cfg, "quat_axis_angle_max_step_rad", 0.08)),
            opacity_logit_max_step=float(self._cfg_get(clamp_cfg, "opacity_logit_max_step", 0.25)),
            sh_max_step=float(self._cfg_get(clamp_cfg, "sh_max_step", 0.10)),
            hidden_max_step=float(self._cfg_get(clamp_cfg, "hidden_max_step", 1.0)),
            accept_vsm_ctx=bool(self._cfg_get(phase_b_hooks, "accept_vsm_ctx", True)),
            vsm_ctx_dim=int(self._cfg_get(phase_b_hooks, "vsm_ctx_dim", self.stage6_event_dim)),
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
        self.stage6_opacity_delta_l2_weight = float(
            self._cfg_get(regularization, "opacity_delta_l2_weight", 0.0)
        )
        self.stage6_sh_delta_l2_weight = float(self._cfg_get(regularization, "sh_delta_l2_weight", 0.0))
        self.stage6_scale_barrier_weight = float(self._cfg_get(regularization, "scale_barrier_weight", 0.0))
        self.stage6_scale_log_min = float(self._cfg_get(regularization, "scale_log_min", -10.0))
        self.stage6_scale_log_max = float(self._cfg_get(regularization, "scale_log_max", 4.0))
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
        groups_cfg = self._cfg_get(opt_cfg, "groups", {}) or {}
        no_wd_cfg = self._cfg_get(opt_cfg, "no_weight_decay", {}) or {}
        no_wd_enable = bool(self._cfg_get(no_wd_cfg, "enable", False))
        no_wd_keywords = [str(x) for x in list(self._cfg_get(no_wd_cfg, "name_keywords", []) or [])]
        no_wd_ndim_leq = int(self._cfg_get(no_wd_cfg, "ndim_leq", 1))

        def lr_for(name: str) -> float:
            if hasattr(lr_cfg, "get"):
                return float(lr_cfg.get(name, lr_cfg.get("default", 1.0e-4)))
            return float(lr_cfg)

        def group_cfg(name: str) -> Any:
            return self._cfg_get(groups_cfg, name, {}) or {}

        def group_lr(name: str, fallback_lr_name: str) -> float:
            cfg = group_cfg(name)
            raw = self._cfg_get(cfg, "lr", None)
            return float(raw) if raw is not None else lr_for(fallback_lr_name)

        def group_wd(name: str) -> float:
            cfg = group_cfg(name)
            raw = self._cfg_get(cfg, "weight_decay", None)
            return float(raw) if raw is not None else weight_decay

        def group_prefixes(name: str, defaults: List[str]) -> List[str]:
            cfg = group_cfg(name)
            match = self._cfg_get(cfg, "match", {}) or {}
            raw = self._cfg_get(match, "prefixes", None)
            return [str(x) for x in list(raw)] if raw is not None else list(defaults)

        def split_decay(named_params: List[tuple[str, torch.nn.Parameter]]) -> tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
            decay: List[torch.nn.Parameter] = []
            no_decay: List[torch.nn.Parameter] = []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                lower = str(name).lower()
                keyword_match = any(str(kw).lower() in lower for kw in no_wd_keywords)
                if no_wd_enable and (int(param.ndim) <= no_wd_ndim_leq or keyword_match):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        groups: List[Dict[str, Any]] = []
        seen_param_ids: set[int] = set()

        def add_group(
            *,
            logical_name: str,
            named_params: List[tuple[str, torch.nn.Parameter]],
            lr: float,
            wd: float,
        ) -> None:
            if float(lr) <= 0.0:
                return
            unique_named: List[tuple[str, torch.nn.Parameter]] = []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in seen_param_ids:
                    continue
                seen_param_ids.add(pid)
                unique_named.append((name, param))
            if len(unique_named) == 0:
                return
            decay, no_decay = split_decay(unique_named)
            if decay:
                groups.append(
                    {
                        "params": decay,
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "logical_name": logical_name,
                    }
                )
            if no_decay:
                groups.append(
                    {
                        "params": no_decay,
                        "lr": float(lr),
                        "weight_decay": 0.0,
                        "logical_name": f"{logical_name}_no_weight_decay",
                    }
                )

        add_group(
            logical_name="stage6_struct_event_decoder_near",
            named_params=[
                (f"stage6_struct_event_decoder.near.{name}", param)
                for name, param in self.stage6_struct_event_decoder.near.named_parameters()
                if not name.startswith("param_obs_codec.")
            ],
            lr=lr_for("struct_event_decoder_near"),
            wd=weight_decay,
        )
        add_group(
            logical_name="stage6_struct_event_decoder_far",
            named_params=[
                (f"stage6_struct_event_decoder.far.{name}", param)
                for name, param in self.stage6_struct_event_decoder.far.named_parameters()
                if not name.startswith("param_obs_codec.")
            ],
            lr=lr_for("struct_event_decoder_far"),
            wd=weight_decay,
        )
        add_group(
            logical_name="stage6_param_obs_codec",
            named_params=[
                (f"stage6_struct_event_decoder.param_obs_codec.{name}", param)
                for name, param in self.stage6_struct_event_decoder.param_obs_codec.named_parameters()
            ],
            lr=lr_for("param_obs_codec"),
            wd=weight_decay,
        )
        add_group(
            logical_name="stage6_posterior_updater",
            named_params=[
                (f"stage6_posterior_updater.{name}", param)
                for name, param in self.stage6_posterior_updater.named_parameters()
            ],
            lr=lr_for("posterior_updater"),
            wd=weight_decay,
        )

        measurement_named = [
            (name, param)
            for name, param in self.named_parameters()
            if name in getattr(self, "stage6_measurement_trainable_param_names", set()) and param.requires_grad
        ]
        residual_prefixes = group_prefixes(
            "residual_unet",
            ["image_feature_extractor.residual", "image_feature_extractor.residual_unet"],
        )
        fusion_prefixes = group_prefixes(
            "fusion_neck",
            ["image_feature_extractor.fusion", "image_feature_extractor.fusion_neck"],
        )
        residual_named = [(n, p) for n, p in measurement_named if any(n.startswith(prefix) for prefix in residual_prefixes)]
        fusion_named = [
            (n, p)
            for n, p in measurement_named
            if any(n.startswith(prefix) for prefix in fusion_prefixes)
            and id(p) not in {id(param) for _, param in residual_named}
        ]
        assigned_measurement_ids = {id(param) for _, param in residual_named + fusion_named}
        remaining_measurement_named = [(n, p) for n, p in measurement_named if id(p) not in assigned_measurement_ids]

        if measurement_named and float(lr_for("measurement_frontend")) <= 0.0:
            raise ValueError(
                "Stage6_0 Phase A from_scratch enabled trainable 2D frontend params but "
                "optimizer.lr.measurement_frontend <= 0."
            )
        add_group(
            logical_name="stage6_measurement_frontend_residual_unet",
            named_params=residual_named,
            lr=group_lr("residual_unet", "measurement_frontend"),
            wd=group_wd("residual_unet"),
        )
        add_group(
            logical_name="stage6_measurement_frontend_fusion_neck",
            named_params=fusion_named,
            lr=group_lr("fusion_neck", "measurement_frontend"),
            wd=group_wd("fusion_neck"),
        )
        add_group(
            logical_name="stage6_measurement_frontend",
            named_params=remaining_measurement_named,
            lr=lr_for("measurement_frontend"),
            wd=weight_decay,
        )
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
        if sub_sky is None:
            raise ValueError("Stage6 Phase A requires source_sky_masks for V4 evidence.")
        if sub_ego is None:
            raise ValueError("Stage6 Phase A requires source_egocar_masks for V4 evidence.")
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
            self._mem_debug("observe/begin", grad_enabled=int(grad_enabled), source_frame_idx=int(source_frame_idx))
            bg_m, distant_m, rigid_m = self._local_to_node_states_detached(local_state)
            self._mem_debug(
                "observe/after_detached_clone",
                num_bg=int(bg_m.means.shape[0]),
                num_distant=int(distant_m.means.shape[0]) if distant_m is not None else 0,
                num_rigid=int(rigid_m.means.shape[0]) if rigid_m is not None else 0,
            )
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
            self._mem_debug(
                "observe/after_route",
                num_rigid_s=int(route.S.numel()),
                num_rigid_in=int(route.S_in.numel()),
                num_rigid_out=int(route.S_out.numel()),
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
            self._mem_debug(
                "observe/after_v4_measurement",
                feat_bg=tuple(one_pass["feat_2d_bg"].shape),
                feat_distant=tuple(one_pass["feat_2d_distant"].shape) if one_pass.get("feat_2d_distant") is not None else None,
                feat_rigid=tuple(one_pass["feat_2d_rigid_S"].shape) if one_pass.get("feat_2d_rigid_S") is not None else None,
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
                "source_frame_idx": int(source_frame_idx),
            }

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

    def _constrain_local_state_after_delta(self, local_state: LocalGSState) -> LocalGSState:
        aabb_min, aabb_max = self._stage6_aabb(local_state.bg.means)
        bg_means = torch.clamp(local_state.bg.means, min=aabb_min, max=aabb_max)
        if not torch.isfinite(bg_means).all():
            raise RuntimeError("Stage6 local bg means contain NaN/Inf after AABB constraint.")
        return replace(local_state, bg=replace(local_state.bg, means=bg_means))

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

    def _stage6_aabb(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = getattr(self, "bbx_min", None)
        hi = getattr(self, "bbx_max", None)
        if lo is None or hi is None:
            raise RuntimeError(
                "Stage6_0 Phase A requires segment AABB: self.bbx_min/self.bbx_max missing."
            )
        return lo.to(device=ref.device, dtype=ref.dtype), hi.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _maybe_detach_feature(x: Optional[torch.Tensor], *, detach: bool) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.detach() if bool(detach) else x

    @staticmethod
    def _detach_optional(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if x is None else x.detach()

    def _build_stage6_struct_input_near(
        self,
        *,
        local_state: LocalGSState,
        rigid_node: Optional[NodeStateRigid],
        route: Any,
        measurement: Dict[str, Any],
        source_frame_idx: int,
    ) -> Stage6StructInput:
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        feat_2d_bg = self._maybe_detach_feature(measurement["feat_2d_bg"], detach=detach_features)
        if feat_2d_bg is None:
            raise RuntimeError("Stage6 near input requires feat_2d_bg")
        acc_w_bg = self._detach_optional(measurement["acc_w_bg"])
        obs_bg = self._detach_optional(measurement["obs_bg"])
        if acc_w_bg is None or obs_bg is None:
            raise RuntimeError("Stage6 near input requires acc_w_bg and obs_bg")
        num_bg = int(local_state.bg.means.shape[0])
        if int(feat_2d_bg.shape[0]) != num_bg:
            raise ValueError(f"Stage6 near bg feature row mismatch: {int(feat_2d_bg.shape[0])} vs {num_bg}")
        if obs_bg.dim() != 2 or int(obs_bg.shape[0]) != num_bg or int(obs_bg.shape[1]) != 2:
            raise ValueError(f"Stage6 near obs_bg must be [N_bg,2], got {tuple(obs_bg.shape)}")

        feat_parts = [feat_2d_bg]
        acc_parts = [acc_w_bg.reshape(-1)]
        obs_parts = [obs_bg]
        coords_parts = [local_state.bg.means]
        branch_ids = [torch.zeros((num_bg,), dtype=torch.long, device=self.device)]
        params_bg = self._build_params_for_embed(local_state.bg, coord_space="world")

        num_rigid_in = int(route.S_in.numel()) if route is not None and hasattr(route, "S_in") else 0
        params_rigid_in = None
        if num_rigid_in > 0:
            feat_2d_rigid_s = self._maybe_detach_feature(measurement.get("feat_2d_rigid_S"), detach=detach_features)
            acc_w_rigid_s = self._detach_optional(measurement.get("acc_w_rigid_S"))
            obs_rigid_s = self._detach_optional(measurement.get("obs_rigid_S"))
            if rigid_node is None or feat_2d_rigid_s is None or acc_w_rigid_s is None or obs_rigid_s is None:
                raise RuntimeError("Stage6 near input requires rigid source tensors when S_in > 0")
            rows_in_s = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_parts.append(feat_2d_rigid_s[rows_in_s])
            acc_parts.append(acc_w_rigid_s.reshape(-1)[rows_in_s])
            obs_parts.append(obs_rigid_s[rows_in_s])
            coords_parts.append(route.means_world_S[route.inside_mask_S])
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=self.device))
            params_rigid_in = self._build_rigid_params_for_embed_source_world(
                rigid_node,
                int(source_frame_idx),
                route.S_in,
            )

        return Stage6StructInput(
            feat_2d=torch.cat(feat_parts, dim=0),
            acc_w=torch.cat(acc_parts, dim=0),
            obs_code=torch.cat(obs_parts, dim=0),
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=cat_param_dict(params_bg, params_rigid_in),
            split_0=num_bg,
            split_1=num_rigid_in,
            meta={
                "path": "near",
                "support_threshold_bg": float(getattr(self, "bg_src_backproject_support_min", 0.0)),
                "support_threshold_rigid": float(getattr(self, "rigid_src_backproject_support_min", 0.0)),
            },
        )

    def _build_stage6_struct_input_far(
        self,
        *,
        local_state: LocalGSState,
        rigid_node: Optional[NodeStateRigid],
        route: Any,
        measurement: Dict[str, Any],
        source_frame_idx: int,
    ) -> Stage6StructInput:
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        ref = measurement["feat_2d_bg"]
        num_distant = int(local_state.distant.means.shape[0]) if local_state.distant is not None else 0
        num_rigid_out = int(route.S_out.numel()) if route is not None and hasattr(route, "S_out") else 0
        if num_distant + num_rigid_out == 0:
            return empty_stage6_struct_input(
                ref=ref,
                feat_2d_dim=int(getattr(self, "stage6_feat_2d_dim", int(ref.shape[-1]))),
                sh_rest_bases=int(local_state.bg.sh_rest.shape[1]),
                path="far",
            )

        feat_parts: List[torch.Tensor] = []
        acc_parts: List[torch.Tensor] = []
        obs_parts: List[torch.Tensor] = []
        coords_parts: List[torch.Tensor] = []
        branch_ids: List[torch.Tensor] = []
        params_for_embed = None
        if num_distant > 0:
            feat_2d_distant = self._maybe_detach_feature(measurement.get("feat_2d_distant"), detach=detach_features)
            acc_w_distant = self._detach_optional(measurement.get("acc_w_distant"))
            obs_distant = self._detach_optional(measurement.get("obs_distant"))
            if local_state.distant is None or feat_2d_distant is None or acc_w_distant is None or obs_distant is None:
                raise RuntimeError("Stage6 far input expected distant tensors")
            feat_parts.append(feat_2d_distant)
            acc_parts.append(acc_w_distant.reshape(-1))
            obs_parts.append(obs_distant)
            coords_parts.append(local_state.distant.means)
            branch_ids.append(torch.zeros((num_distant,), dtype=torch.long, device=self.device))
            params_for_embed = self._build_params_for_embed(local_state.distant, coord_space="world")

        params_rigid_out = None
        if num_rigid_out > 0:
            feat_2d_rigid_s = self._maybe_detach_feature(measurement.get("feat_2d_rigid_S"), detach=detach_features)
            acc_w_rigid_s = self._detach_optional(measurement.get("acc_w_rigid_S"))
            obs_rigid_s = self._detach_optional(measurement.get("obs_rigid_S"))
            if rigid_node is None or feat_2d_rigid_s is None or acc_w_rigid_s is None or obs_rigid_s is None:
                raise RuntimeError("Stage6 far input expected rigid tensors for S_out")
            rows_out_s = torch.nonzero(~route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_parts.append(feat_2d_rigid_s[rows_out_s])
            acc_parts.append(acc_w_rigid_s.reshape(-1)[rows_out_s])
            obs_parts.append(obs_rigid_s[rows_out_s])
            coords_parts.append(route.means_world_S[~route.inside_mask_S])
            branch_ids.append(torch.ones((num_rigid_out,), dtype=torch.long, device=self.device))
            params_rigid_out = self._build_rigid_params_for_embed_source_world(
                rigid_node,
                int(source_frame_idx),
                route.S_out,
            )

        if params_for_embed is None:
            params_for_embed = params_rigid_out
        elif params_rigid_out is not None:
            params_for_embed = cat_param_dict(params_for_embed, params_rigid_out)
        if params_for_embed is None:
            raise RuntimeError("Stage6 far input internal empty params_for_embed")

        return Stage6StructInput(
            feat_2d=torch.cat(feat_parts, dim=0),
            acc_w=torch.cat(acc_parts, dim=0),
            obs_code=torch.cat(obs_parts, dim=0),
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=params_for_embed,
            split_0=num_distant,
            split_1=num_rigid_out,
            meta={
                "path": "far",
                "support_threshold_distant": float(getattr(self, "distant_src_backproject_support_min", 0.0)),
                "support_threshold_rigid_out": float(getattr(self, "rigid_src_backproject_support_min", 0.0)),
            },
        )

    def _encode_and_update(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        route = measurement["route"]
        rigid_node = self._local_rigid_node_state(local_state)
        source_frame_idx = int(measurement.get("source_frame_idx", 0))
        self._mem_debug("encode/begin", source_frame_idx=source_frame_idx)
        near_in = self._build_stage6_struct_input_near(
            local_state=local_state,
            rigid_node=rigid_node,
            route=route,
            measurement=measurement,
            source_frame_idx=source_frame_idx,
        )
        self._mem_debug("encode/after_near_input", near_n=int(near_in.coords.shape[0]))
        far_in = self._build_stage6_struct_input_far(
            local_state=local_state,
            rigid_node=rigid_node,
            route=route,
            measurement=measurement,
            source_frame_idx=source_frame_idx,
        )
        self._mem_debug("encode/after_far_input", far_n=int(far_in.coords.shape[0]))
        aabb_min, aabb_max = self._stage6_aabb(measurement["feat_2d_bg"])
        event = self.stage6_struct_event_decoder(
            near_in=near_in,
            far_in=far_in,
            route=route,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            near_batch_offsets=self._build_struct_batch_offsets(stage6_to_struct_decoder_input(near_in), device=self.device),
            far_batch_offsets=self._build_struct_batch_offsets(stage6_to_struct_decoder_input(far_in), device=self.device),
        )
        self._mem_debug("encode/after_struct_event")
        delta, aux = self.stage6_posterior_updater(event=event, ctx_current=None, ctx_vsm=None)
        self._mem_debug("encode/after_posterior")
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
        next_state = local_state.apply_delta(delta)
        next_state = self._constrain_local_state_after_delta(next_state)
        return next_state, delta, {**event.aux, **aux}

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
        render_params = self._render_params_for_frame(local_state=local_state, frame_idx=int(target.get("frame_idx", 0)))
        gt = target["gt_image"]
        height, width = spatial_hw_from_image_tensor(gt)
        return self._render_single_view(render_params, target["view"], height, width)

    def _render_params_for_frame(self, *, local_state: LocalGSState, frame_idx: int) -> Dict[str, torch.Tensor]:
        parts = [self._branch_render_params(local_state.bg)]
        rigid_node = self._local_rigid_node_state(local_state)
        if local_state.rigid is not None and rigid_node is not None:
            valid = self._rigid_point_valid_mask(rigid_node, int(frame_idx))
            idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                rigid_local_all = self._branch_render_params(local_state.rigid)
                rigid_local = {k: v[idx] for k, v in rigid_local_all.items()}
                point_ids = rigid_node.point_ids[idx, 0]
                parts.append(
                    self._rigid_local_to_world_render_params(
                        rigid_node,
                        rigid_local,
                        int(frame_idx),
                        point_ids_subset=point_ids,
                    )
                )
        if local_state.distant is not None:
            parts.append(self._branch_render_params(local_state.distant))
        return self._cat_render_params(parts)

    def _render_targets_grouped_by_frame(
        self,
        *,
        local_state: LocalGSState,
        targets_with_indices: List[Tuple[int, Dict[str, Any]]],
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        by_frame: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for idx, target in targets_with_indices:
            by_frame[int(target.get("frame_idx", 0))].append((int(idx), target))

        pred_by_idx: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for frame_idx in sorted(by_frame.keys()):
            group = by_frame[int(frame_idx)]
            targets_f = [target for _, target in group]
            render_params = self._render_params_for_frame(local_state=local_state, frame_idx=int(frame_idx))

            heights: List[int] = []
            widths: List[int] = []
            for target in targets_f:
                height, width = spatial_hw_from_image_tensor(target["gt_image"])
                heights.append(int(height))
                widths.append(int(width))
            h0, w0 = int(heights[0]), int(widths[0])
            if all(int(h) == h0 and int(w) == w0 for h, w in zip(heights, widths)):
                multi_result = self._render_multi_view(render_params, targets_f)
                if multi_result is not None:
                    for (orig_idx, _target), (rgb, acc) in zip(group, multi_result):
                        pred_by_idx[int(orig_idx)] = (rgb, acc.squeeze(-1) if acc.dim() == 3 else acc)
                    continue

            for orig_idx, target in group:
                height, width = spatial_hw_from_image_tensor(target["gt_image"])
                pred_rgb, acc = self._render_single_view(render_params, target["view"], int(height), int(width))
                pred_by_idx[int(orig_idx)] = (pred_rgb, acc.squeeze(-1) if acc.dim() == 3 else acc)
        return pred_by_idx

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
            return local_state.bg.means.new_tensor(0.0), {
                "num_refs": 0.0,
                "psnr": 0.0,
                "l1": 0.0,
                "ssim": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
        losses: List[torch.Tensor] = []
        stats: Dict[str, float] = {"num_refs": float(len(target_indices))}
        psnr_vals: List[float] = []
        l1_vals: List[float] = []
        ssim_vals: List[float] = []
        valid_ratios: List[float] = []
        skip_count = 0.0
        targets_with_indices = [(int(idx), batch["targets"][int(idx)]) for idx in target_indices]
        pred_by_idx = self._render_targets_grouped_by_frame(
            local_state=local_state,
            targets_with_indices=targets_with_indices,
        )
        for idx in target_indices:
            target = batch["targets"][int(idx)]
            pred, _alpha = pred_by_idx[int(idx)]
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
            ssim_vals.append(float(stat_i.get("ssim", 0.0)))
            valid_ratios.append(float(stat_i.get("valid_ratio", 0.0)))
            skip_count += float(stat_i.get("skipped_no_valid_pixels", 0.0))
        stats["psnr"] = float(sum(psnr_vals) / max(len(psnr_vals), 1))
        stats["l1"] = float(sum(l1_vals) / max(len(l1_vals), 1))
        stats["ssim"] = float(sum(ssim_vals) / max(len(ssim_vals), 1))
        stats["valid_ratio"] = float(sum(valid_ratios) / max(len(valid_ratios), 1))
        stats["skipped_no_valid_pixels"] = float(skip_count)
        return torch.stack(losses).mean(), stats

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        roles = resolve_v9_phase_a_batch(batch)
        self._mem_debug("forward/begin", inner_K=int(roles.inner_K))
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("Stage6_0 Phase A requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("Stage6_0 Phase A requires non-empty targets.")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        self._mem_debug(
            "forward/after_node_state",
            num_bg=int(node_state_bg.means.shape[0]),
            num_distant=int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
            num_rigid=int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        )
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=self.stage6_hidden_dim,
        )
        self._mem_debug("forward/after_local_state_clone")
        total_loss = local_state.bg.means.new_tensor(0.0)
        per_step: List[Dict[str, float]] = []
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        step = int(batch.get("global_step", 0) or 0)
        for k in range(roles.inner_K):
            self._mem_debug("forward/k_begin", k=int(k))
            evidence_refs = roles.evidence_refs_by_step[int(k)]
            source_frame_idx = int(evidence_refs[0][0])
            measurement = self._observe_v4_measurement(
                local_state=local_state,
                batch=batch,
                source_indices=roles.evidence_source_indices_by_step[int(k)],
                source_frame_idx=source_frame_idx,
            )
            local_state, delta, update_aux = self._encode_and_update(local_state=local_state, measurement=measurement)
            self._mem_debug("forward/after_encode_update", k=int(k))
            block_loss, block_stats = self._render_loss_for_indices(
                local_state=local_state,
                batch=batch,
                target_indices=roles.block_target_indices_by_step[int(k)],
                mask_policy=self.stage6_block_mask_policy,
                pred_rgbs_out=pred_rgbs if int(k) == roles.inner_K - 1 else None,
                gt_images_out=gt_images if int(k) == roles.inner_K - 1 else None,
            )
            self._mem_debug("forward/after_block_loss", k=int(k))
            nearby_loss, nearby_stats = self._render_loss_for_indices(
                local_state=local_state,
                batch=batch,
                target_indices=roles.nearby_target_indices_by_step[int(k)],
                mask_policy=self.stage6_nearby_mask_policy,
                pred_rgbs_out=pred_rgbs if int(k) == roles.inner_K - 1 else None,
                gt_images_out=gt_images if int(k) == roles.inner_K - 1 else None,
            )
            self._mem_debug("forward/after_nearby_loss", k=int(k))
            reg_loss, reg_stats = delta_regularization(
                delta,
                weight=self.stage6_delta_l2_weight,
                local_state=local_state,
                opacity_delta_l2_weight=self.stage6_opacity_delta_l2_weight,
                sh_delta_l2_weight=self.stage6_sh_delta_l2_weight,
                scale_barrier_weight=self.stage6_scale_barrier_weight,
                scale_log_min=self.stage6_scale_log_min,
                scale_log_max=self.stage6_scale_log_max,
            )
            near_weight = self._nearby_weight(global_step=step, k=int(k), K=roles.inner_K)
            step_weight = float(self.stage6_step_gamma) ** float(roles.inner_K - 1 - int(k))
            loss_k = step_weight * (self.stage6_block_weight * block_loss + near_weight * nearby_loss + reg_loss)
            if not torch.isfinite(loss_k).all():
                raise RuntimeError("Stage6_0 Phase A loss became NaN/Inf.")
            total_loss = total_loss + loss_k
            self._mem_debug("forward/k_end", k=int(k))
            per_step.append(
                {
                    "k": float(k),
                    "loss_block": float(block_loss.detach().item()),
                    "loss_nearby": float(nearby_loss.detach().item()),
                    "nearby_weight": float(near_weight),
                    "block_psnr": float(block_stats.get("psnr", 0.0)),
                    "nearby_psnr": float(nearby_stats.get("psnr", 0.0)),
                    "block_valid_ratio": float(block_stats.get("valid_ratio", 0.0)),
                    "nearby_valid_ratio": float(nearby_stats.get("valid_ratio", 0.0)),
                    "block_skipped": float(block_stats.get("skipped_no_valid_pixels", 0.0)),
                    "nearby_skipped": float(nearby_stats.get("skipped_no_valid_pixels", 0.0)),
                    "block_ssim": float(block_stats.get("ssim", 0.0)),
                    "nearby_ssim": float(nearby_stats.get("ssim", 0.0)),
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

    @torch.no_grad()
    def validate_v9_phase_a(
        self,
        batch: Dict[str, Any],
        *,
        k_values: List[int],
        max_K: int,
        mask_cfg: Optional[Dict[str, Any]] = None,
        save_images: bool = False,
        save_dir: Optional[str] = None,
        save_image_k_values: Optional[List[int]] = None,
        max_saved_cams: int = 1,
    ) -> Dict[str, Any]:
        from models.streetforward.validation_v9_runner import validate_v9_phase_a

        return validate_v9_phase_a(
            self,
            batch,
            k_values=[int(x) for x in k_values],
            max_K=int(max_K),
            mask_cfg=mask_cfg,
            save_images=bool(save_images),
            save_dir=save_dir,
            save_image_k_values=save_image_k_values,
            max_saved_cams=int(max_saved_cams),
        )

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
        grad_group_sums = self._stage6_assert_required_group_grads(out)
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
            "mask/block_valid_ratio_final": float(final.get("block_valid_ratio", 0.0)),
            "mask/nearby_valid_ratio_final": float(final.get("nearby_valid_ratio", 0.0)),
            "mask/block_skipped_no_valid_pixels_final": float(final.get("block_skipped", 0.0)),
            "mask/nearby_skipped_no_valid_pixels_final": float(final.get("nearby_skipped", 0.0)),
            "phaseA/grad_norm_total": float(grad_norm.detach().item()),
            **grad_group_sums,
        }
        for item in per_step:
            k = int(item["k"])
            logs[f"phaseA/loss_block_k{k}"] = float(item.get("loss_block", 0.0))
            logs[f"phaseA/loss_nearby_k{k}"] = float(item.get("loss_nearby", 0.0))
            logs[f"phaseA/block_psnr_k{k}"] = float(item.get("block_psnr", 0.0))
            logs[f"mask/block_valid_ratio_k{k}"] = float(item.get("block_valid_ratio", 0.0))
            logs[f"mask/nearby_valid_ratio_k{k}"] = float(item.get("nearby_valid_ratio", 0.0))
            logs[f"mask/block_skipped_no_valid_pixels_k{k}"] = float(item.get("block_skipped", 0.0))
            logs[f"mask/nearby_skipped_no_valid_pixels_k{k}"] = float(item.get("nearby_skipped", 0.0))
        return logs

    def _assert_group_nonzero_grad(
        self,
        *,
        group_name: str,
        params: List[torch.nn.Parameter],
        required: bool = True,
    ) -> float:
        total = 0.0
        seen = 0
        for param in params:
            if not param.requires_grad:
                continue
            seen += 1
            if param.grad is not None:
                total += float(param.grad.detach().abs().sum().item())
        if required and (seen == 0 or total == 0.0):
            raise RuntimeError(f"{group_name} has zero gradient in Stage6_0 Phase A.")
        return float(total)

    def _stage6_assert_required_group_grads(self, out: Dict[str, Any]) -> Dict[str, float]:
        per_step = list(out.get("per_step") or [])
        required_far = any(
            float(item.get("stage6/struct/far_num_distant", 0.0)) > 0.0
            or float(item.get("stage6/struct/far_num_rigid_out", 0.0)) > 0.0
            for item in per_step
        )
        near_params = [
            param
            for name, param in self.stage6_struct_event_decoder.near.named_parameters()
            if not name.startswith("param_obs_codec.")
        ]
        far_params = [
            param
            for name, param in self.stage6_struct_event_decoder.far.named_parameters()
            if not name.startswith("param_obs_codec.")
        ]
        param_obs_params = list(self.stage6_struct_event_decoder.param_obs_codec.parameters())
        posterior_params = list(self.stage6_posterior_updater.parameters())
        sums: Dict[str, float] = {
            "grad/stage6_struct_event_decoder_near_sum": self._assert_group_nonzero_grad(
                group_name="stage6_struct_event_decoder.near",
                params=near_params,
                required=True,
            ),
            "grad/stage6_struct_event_decoder_far_sum": self._assert_group_nonzero_grad(
                group_name="stage6_struct_event_decoder.far",
                params=far_params,
                required=bool(required_far),
            ),
            "grad/stage6_param_obs_codec_sum": self._assert_group_nonzero_grad(
                group_name="stage6_param_obs_codec",
                params=param_obs_params,
                required=True,
            ),
            "grad/stage6_posterior_updater_sum": self._assert_group_nonzero_grad(
                group_name="stage6_posterior_updater",
                params=posterior_params,
                required=True,
            ),
        }
        if str(getattr(self, "stage6_phase_a_mode", "updater_only")) == "from_scratch":
            named = [
                (name, param)
                for name, param in self.named_parameters()
                if name in getattr(self, "stage6_measurement_trainable_param_names", set())
            ]
            sums["grad/stage6_measurement_frontend_sum"] = self._assert_group_nonzero_grad(
                group_name="stage6_measurement_frontend",
                params=[param for _, param in named],
                required=True,
            )
            residual_params = [
                param
                for name, param in named
                if name.startswith("image_feature_extractor.residual")
                or name.startswith("image_feature_extractor.residual_unet")
            ]
            fusion_params = [
                param
                for name, param in named
                if name.startswith("image_feature_extractor.fusion")
                or name.startswith("image_feature_extractor.fusion_neck")
            ]
            if residual_params:
                sums["grad/measurement_frontend_residual_unet_sum"] = self._assert_group_nonzero_grad(
                    group_name="image_feature_extractor.residual_unet",
                    params=residual_params,
                    required=True,
                )
            if fusion_params:
                sums["grad/measurement_frontend_fusion_neck_sum"] = self._assert_group_nonzero_grad(
                    group_name="image_feature_extractor.fusion_neck",
                    params=fusion_params,
                    required=True,
                )
        return sums

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
            ref_param = next(self.stage6_struct_event_decoder.parameters())
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
            "struct_event_decoder": {
                k: v.detach().cpu()
                for k, v in self.stage6_struct_event_decoder.state_dict().items()
            },
            "param_obs_codec": {
                k: v.detach().cpu()
                for k, v in self.stage6_struct_event_decoder.param_obs_codec.state_dict().items()
            },
            "posterior_updater_base": self.stage6_posterior_updater.base_state_dict(),
            "legacy_event_encoder": None,
            "current_context_adapter": None,
            "normalizer_stats": normalizer_stats,
            "event_schema": {
                "event_dim": int(self.stage6_event_dim),
                "feat_2d_dim": int(self.stage6_feat_2d_dim),
                "param_obs_dim": int(self.stage6_struct_event_decoder.param_obs_codec.output_dim),
                "obs_code_dim": 2,
                "near_path": "bg+rigid_in:xCPE",
                "far_path": "distant+rigid_out:MLP",
            },
            "phase_b_init_policy": {
                "freeze_measurement_frontend": True,
                "freeze_struct_event_decoder": True,
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
