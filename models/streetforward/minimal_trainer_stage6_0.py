from __future__ import annotations

import copy
import gc
import logging
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.minimal_trainer_stage4_0 import spatial_hw_from_image_tensor
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders.common import cat_param_dict
from models.streetforward.stage6_0 import (
    ContextPack,
    LocalGSState,
    PHASE_B_NAME,
    PHASE_B_LONG_NAME,
    LongCellStreamingVSM,
    LongStreamingVSM,
    PhaseBOffsetState,
    Stage6QueryDecoder,
    Stage6PosteriorUpdater,
    Stage6RoutedStructEventDecoder,
    Stage6StructInput,
    Stage6VSMState,
    Stage6ViewSetMemory,
    VSMOffsetDecoder,
    empty_stage6_struct_input,
    materialize_phase_b_state,
    offset_regularization as phase_b_long_offset_regularization,
    phase_b_long_final_render_loss,
    resolve_v9_phase_a_batch,
    resolve_v9_phase_b_batch,
    stage6_to_struct_decoder_input,
)
from models.streetforward.stage6_0.phase_a_losses import (
    delta_regularization,
    masked_rgb_loss,
    target_valid_mask,
)
from models.streetforward.stage6_0.phase_b_long.streaming_vsm import DISTANT_MODE_APPEARANCE_SCALE, DISTANT_MODE_FROZEN
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack
from models.streetforward.stage6_0.vsm import Stage6QueryPred, masked_smooth_l1


logger = logging.getLogger(__name__)


def _to_plain_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return {
            str(k): _to_plain_dict(v)
            if isinstance(v, dict) or hasattr(v, "keys")
            else [x for x in v]
            if isinstance(v, (list, tuple))
            else v
            for k, v in node.items()
        }
    if hasattr(node, "keys"):
        out: Dict[str, Any] = {}
        for k in node.keys():
            v = node[k]
            if isinstance(v, dict) or hasattr(v, "keys"):
                out[str(k)] = _to_plain_dict(v)
            elif isinstance(v, (list, tuple)):
                out[str(k)] = [x for x in v]
            else:
                out[str(k)] = v
        return out
    return {}


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
        self._validate_stage6_0_config(config)
        self._configure_measurement_frontend_trainability(config)
        self._init_stage6_modules(config)
        self._configure_stage6_trainability_after_module_init(config)
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
        if self._cfg_get(cfg, "current_observation", None) is None:
            cfg["current_observation"] = {
                "enable": True,
                "dim": 2,
                "rho_source": "feature",
                "eps": 1.0e-6,
                "input_to_struct_decoder": False,
                "input_to_far_mlp": False,
                "input_to_gru": False,
                "input_to_history_gate": False,
                "record_to_history_memory": False,
            }
        if self._cfg_get(model_cfg, "history_memory", None) is None:
            model_cfg["history_memory"] = {
                "enable": True,
                "record_on": "block_exit",
                "record_views": "source_image_refs",
                "support": {
                    "fast_ema_beta_visible": 0.35,
                    "fast_ema_beta_invisible": 0.60,
                    "slow_ema_beta_visible": 0.90,
                    "slow_ema_beta_invisible": 0.95,
                },
                "residual": {
                    "fast_error_beta": 0.35,
                    "slow_error_beta": 0.9,
                    "error_eps": 1.0e-6,
                },
                "update": {
                    "fast_ema_beta": 0.45,
                    "slow_ema_beta": 0.92,
                    "apply_in_eval": True,
                },
            }
        if self._cfg_get(model_cfg, "view_transient", None) is None:
            model_cfg["view_transient"] = {
                "enable": True,
                "source": "ego_to_point",
                "input_to_gate": True,
                "input_to_struct_decoder": False,
                "use_delta_xyz": True,
                "use_delta_norm": True,
                "use_angle_delta": False,
                "use_initialized_flag": False,
                "detach": True,
                "update_in_train": True,
                "update_in_eval": True,
            }
        if self._cfg_get(model_cfg, "update_gate", None) is None:
            model_cfg["update_gate"] = {
                "enable": True,
                "type": "attribute_5",
                "hidden_dim": 48,
                "require_initialized_in_input": True,
                "include_visible_now": True,
                "bind_with_mask_update": True,
                "min_gate": {
                    "means": 0.03,
                    "scales": 0.03,
                    "quat": 0.01,
                    "opacity": 0.05,
                    "sh": 0.05,
                },
                "init_bias": {
                    "means": -1.40,
                    "scales": -1.70,
                    "quat": -2.00,
                    "opacity": -0.40,
                    "sh": 0.40,
                },
                "branch_bias": {
                    "bg": {"means": 0.0, "scales": 0.0, "quat": -0.2, "opacity": 0.0, "sh": 0.1},
                    "distant": {"means": -1.0, "scales": -0.3, "quat": -1.0, "opacity": 0.0, "sh": 0.0},
                    "rigid_in": {"means": -0.2, "scales": -0.2, "quat": -0.3, "opacity": 0.1, "sh": 0.2},
                    "rigid_out": {"means": -0.8, "scales": -0.3, "quat": -0.8, "opacity": 0.0, "sh": 0.0},
                },
                "hidden_gate": {
                    "mode": "weighted_sum",
                    "weights": {"means": 0.2, "scales": 0.0, "quat": 0.0, "opacity": 0.3, "sh": 0.5},
                },
            }
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
            sv9 = self._cfg_get(cfg, "scheduler_v9", None)
            slong = self._cfg_get(cfg, "scheduler_long_phase_b", None)
            use_long = slong is not None and bool(self._cfg_get(slong, "enable", False))
            sched_src = slong if use_long else sv9
            sched_label = "scheduler_long_phase_b" if use_long else "scheduler_v9"
            if sched_src is None:
                raise ValueError("Stage6_0 requires scheduler_v9 or scheduler_long_phase_b for Stage5_4 compatibility.")
            if use_long:
                ep_window = self._cfg_get(sched_src, "episode_window", {}) or {}
                ep = {
                    "blocks_per_episode": int(self._cfg_get(ep_window, "frames_per_window", 2)),
                    "include_source_frame": True,
                    "target_policy": "visited_episode_frames",
                    "block_source_frame_policy": "random_within_keyframe_per_visit",
                    "frame_within_keyframe_policy": "random_once_per_episode",
                    "min_keyframes_required_policy": "use_available_if_less_than_window",
                }
                execution = {"block_order": "step_major", "step_major_switch_interval_steps": 1, "reset_policy": "episode_end"}
                block = {"steps_per_block": 1}
            else:
                ep = self._require_key(sched_src, "episode", sched_label)
                if self._cfg_get(ep, "blocks_per_episode", None) is None and str(
                    self._cfg_get(sched_src, "phase", "")
                ) == PHASE_B_NAME:
                    phase_b = self._cfg_get(sched_src, "phase_B", {}) or {}
                    rollout = self._cfg_get(phase_b, "rollout", {}) or {}
                    shapes = [dict(x) for x in list(self._cfg_get(rollout, "shapes", []) or [])]
                    if shapes:
                        max_blocks = max(int(self._cfg_get(shape, "blocks_per_rollout", 0) or 0) for shape in shapes)
                        ep["blocks_per_episode"] = int(self._cfg_get(ep, "rollouts_per_episode", 1)) * int(max_blocks)
                execution = self._cfg_get(sched_src, "execution", {}) or {}
                block = self._cfg_get(sched_src, "block", {}) or {}
            traversal = self._cfg_get(sched_src, "traversal", {}) or {}
            preload = self._cfg_get(sched_src, "preload", {}) or {}
            cfg["scheduler_v8"] = {
                "enable": True,
                "block": {
                    "steps_per_block": int(self._cfg_get(block, "steps_per_block", 1)),
                },
                "episode": {
                    "blocks_per_episode": int(self._require_key(ep, "blocks_per_episode", f"{sched_label}.episode")),
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

        self._validate_stage6_0_config(config)

    def _stage6_config_phase(self, config: Any) -> str:
        model_cfg = self._require_key(config, "model", "config")
        return str(self._cfg_get(model_cfg, "phase", "phase_A_block_local_unroll"))

    def _validate_stage6_0_config(self, config) -> None:
        phase = self._stage6_config_phase(config)
        if phase == "phase_A_block_local_unroll":
            self._validate_stage6_0_phase_a_config(config)
            return
        if phase == PHASE_B_NAME:
            self._validate_stage6_0_phase_b_config(config)
            return
        if phase == PHASE_B_LONG_NAME:
            self._validate_stage6_0_phase_b_long_config(config)
            return
        raise ValueError(f"unsupported Stage6_0 model.phase={phase!r}")

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
        if train_dinov2:
            raise ValueError("Stage6_0 Phase A P0 requires base_measurement.train_dinov2=false.")
        if phase_a_mode == "updater_only":
            if source_grad_mode != "no_grad_v4":
                raise ValueError(
                    "Stage6_0 Phase A updater_only requires source_evidence_grad_mode=no_grad_v4."
                )
            if not detach_v4_outputs:
                raise ValueError("Stage6_0 Phase A updater_only requires base_measurement.detach_v4_outputs=true.")
            if train_2d_frontend or train_residual_unet or train_fusion_neck or train_v4_lift:
                raise ValueError("Stage6_0 Phase A updater_only must keep the 2D/V4 frontend frozen.")
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

    def _validate_stage6_0_phase_b_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if str(self._require_key(model_cfg, "stage", "model")) != "6_0":
            raise ValueError("Stage6_0 requires model.stage='6_0'.")
        if str(self._cfg_get(model_cfg, "phase", "")) != PHASE_B_NAME:
            raise ValueError("Stage6_0 Phase B requires model.phase=phase_B_viewset_rollout.")

        for key in ("history_memory", "update_gate", "view_transient"):
            if bool(self._cfg_get(self._cfg_get(model_cfg, key, {}) or {}, "enable", False)):
                raise ValueError(f"Stage6_0 Phase B forbids model.{key}.enable=true")

        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        if str(self._cfg_get(base_measurement, "type", "")) != "stage5_4_v4":
            raise ValueError("Stage6_0 Phase B requires base_measurement.type=stage5_4_v4.")
        if bool(self._cfg_get(base_measurement, "require_fused_v4", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires fused V4; fallback is forbidden.")
        if bool(self._cfg_get(base_measurement, "require_obs_code", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires V4 obs_code.")
        if int(self._cfg_get(base_measurement, "obs_code_dim", 2)) != 2:
            raise ValueError("Stage6_0 Phase B requires base_measurement.obs_code_dim=2.")
        if str(self._cfg_get(base_measurement, "source_evidence_grad_mode", "no_grad_v4")) != "no_grad_v4":
            raise ValueError("Stage6_0 Phase B requires base_measurement.source_evidence_grad_mode=no_grad_v4.")
        if bool(self._cfg_get(base_measurement, "detach_v4_outputs", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires base_measurement.detach_v4_outputs=true.")
        for key in ("train_2d_frontend", "train_residual_unet", "train_fusion_neck", "train_dinov2", "train_v4_lift"):
            if bool(self._cfg_get(base_measurement, key, False)):
                raise ValueError(f"Stage6_0 Phase B requires base_measurement.{key}=false.")

        struct_cfg = self._require_key(stage6, "struct_event_decoder", "model.stage6_0")
        if bool(self._cfg_get(struct_cfg, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires struct_event_decoder.enable=true.")
        if bool(self._cfg_get(struct_cfg, "freeze", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires struct_event_decoder.freeze=true.")
        event_cfg = self._cfg_get(stage6, "event_encoder", {}) or {}
        if bool(self._cfg_get(event_cfg, "enable", False)):
            raise ValueError("Stage6_0 Phase B forbids direct concat EventEncoder.")
        ctx_cfg = self._cfg_get(stage6, "current_context_adapter", {}) or {}
        if bool(self._cfg_get(ctx_cfg, "enable", False)):
            raise ValueError("Stage6_0 Phase B forbids current_context_adapter.")

        sv9_probe = self._require_key(config, "scheduler_v9", "config")
        phase_b_probe = self._require_key(sv9_probe, "phase_B", "scheduler_v9")
        rollout_probe = self._cfg_get(phase_b_probe, "rollout", {}) or {}
        rollout_mode_probe = str(self._cfg_get(rollout_probe, "mode", "random_viewset_local"))
        if rollout_mode_probe == "episode_rollout_grouped_repeat_tbptt":
            if bool(self._cfg_get(sv9_probe, "enable", False)) is not True:
                raise ValueError("Stage6_0 Phase B final rollout requires scheduler_v9.enable=true.")
            if str(self._cfg_get(sv9_probe, "phase", "")) != PHASE_B_NAME:
                raise ValueError("Stage6_0 Phase B final rollout requires scheduler_v9.phase=phase_B_viewset_rollout.")
            block_cfg = self._cfg_get(sv9_probe, "block", {}) or {}
            if int(self._cfg_get(block_cfg, "steps_per_block", 1)) != 1:
                raise ValueError("Stage6_0 Phase B final rollout requires scheduler_v9.block.steps_per_block=1.")
            episode_cfg = self._require_key(sv9_probe, "episode", "scheduler_v9")
            if int(self._cfg_get(episode_cfg, "rollouts_per_episode", 0) or 0) < 1:
                raise ValueError("Stage6_0 Phase B final rollout requires episode.rollouts_per_episode >= 1.")
            if not list(self._cfg_get(rollout_probe, "shapes", []) or []):
                raise ValueError("Stage6_0 Phase B final rollout requires phase_B.rollout.shapes.")
            final_cfg = self._require_key(phase_b_probe, "final_supervision", "scheduler_v9.phase_B")
            if str(self._cfg_get(final_cfg, "apply", "")) != "rollout_final_only":
                raise ValueError("Stage6_0 Phase B final rollout requires final_supervision.apply=rollout_final_only.")
            masks = self._require_key(phase_b_probe, "masks", "scheduler_v9.phase_B")
            for key in ("vsm_scope", "evidence_mask", "prefix_loss_mask", "query_label_mask"):
                actual = str(self._cfg_get(masks, key, ""))
                if "dynamic" in actual:
                    raise ValueError("Stage6_0 Phase B final rollout must not use dynamic mask policies.")
            local_rollout = self._cfg_get(stage6, "local_rollout", {}) or {}
            if str(self._cfg_get(local_rollout, "source", "")) != "scheduler_v9":
                raise ValueError("Stage6_0 Phase B final rollout requires local_rollout.source=scheduler_v9.")
            long_cfg = self._require_key(stage6, "phase_b_long", "model.stage6_0")
            if bool(self._cfg_get(long_cfg, "enable", False)) is not True:
                raise ValueError("Stage6_0 Phase B final rollout requires phase_b_long.enable=true.")
            q_cfg = self._cfg_get(long_cfg, "query_decoder", {}) or {}
            if bool(self._cfg_get(q_cfg, "enable", False)):
                raise ValueError("Stage6_0 Phase B final rollout requires phase_b_long.query_decoder.enable=false.")
            vsm_long_cfg = self._require_key(long_cfg, "vsm", "model.stage6_0.phase_b_long")
            vsm_type = str(self._cfg_get(vsm_long_cfg, "type", "streaming_selective_ssm"))
            if vsm_type not in {"streaming_selective_ssm", "cell_streaming_selective_ssm"}:
                raise ValueError("Stage6_0 Phase B final rollout requires a Long VSM implementation.")
            dec_cfg = self._require_key(long_cfg, "offset_decoder", "model.stage6_0.phase_b_long")
            if str(self._cfg_get(dec_cfg, "input_source", "vsm_readout_only")) != "vsm_readout_only":
                raise ValueError("Stage6_0 Phase B final rollout requires offset_decoder.input_source=vsm_readout_only.")
            losses_cfg = self._cfg_get(config, "losses", {}) or {}
            phase_b_long = self._require_key(losses_cfg, "phase_b_long", "losses")
            for key in ("query_observation", "nearby_render", "per_step_prefix_render"):
                node = self._cfg_get(phase_b_long, key, {}) or {}
                if bool(self._cfg_get(node, "enable", False)):
                    raise ValueError(f"Stage6_0 Phase B final rollout requires losses.phase_b_long.{key}.enable=false.")
            return

        vsm_cfg = self._cfg_get(stage6, "vsm", {}) or {}
        if bool(self._cfg_get(vsm_cfg, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires stage6_0.vsm.enable=true.")
        if str(self._cfg_get(vsm_cfg, "scope", "bg_rigid")) != "bg_rigid":
            raise ValueError("Stage6_0 Phase B-R requires stage6_0.vsm.scope=bg_rigid.")
        vsm_branches = list(self._cfg_get(vsm_cfg, "branches", ["bg", "rigid"]) or [])
        if [str(x) for x in vsm_branches] != ["bg", "rigid"]:
            raise ValueError("Stage6_0 Phase B-R requires stage6_0.vsm.branches=[bg, rigid].")
        query_cfg = self._cfg_get(stage6, "query_decoder", {}) or {}
        if bool(self._cfg_get(query_cfg, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires stage6_0.query_decoder.enable=true.")
        query_branches = list(self._cfg_get(query_cfg, "branches", ["bg", "rigid"]) or [])
        if [str(x) for x in query_branches] != ["bg", "rigid"]:
            raise ValueError("Stage6_0 Phase B-R requires stage6_0.query_decoder.branches=[bg, rigid].")

        updater_cfg = self._cfg_get(stage6, "posterior_updater", {}) or {}
        if bool(self._cfg_get(updater_cfg, "input_event", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.input_event=true.")
        if bool(self._cfg_get(updater_cfg, "input_current_ctx", False)):
            raise ValueError("Stage6_0 Phase B requires posterior_updater.input_current_ctx=false.")
        if bool(self._cfg_get(updater_cfg, "input_vsm_ctx", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.input_vsm_ctx=true.")
        if bool(self._cfg_get(updater_cfg, "freeze_base", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.freeze_base=true.")
        if bool(self._cfg_get(updater_cfg, "train_vsm_ctx_adapter", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.train_vsm_ctx_adapter=true.")
        phase_b_hooks = self._cfg_get(updater_cfg, "phase_b_hooks", {}) or {}
        if bool(self._cfg_get(phase_b_hooks, "accept_vsm_ctx", True)) is not True:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.phase_b_hooks.accept_vsm_ctx=true.")
        branch_scope = self._cfg_get(updater_cfg, "branch_scope", {}) or {}
        distant_cfg = self._cfg_get(branch_scope, "distant", {}) or {}
        if bool(self._cfg_get(distant_cfg, "enable", False)):
            raise ValueError("Stage6_0 Phase B-R requires posterior_updater.branch_scope.distant.enable=false.")
        rigid_cfg = self._cfg_get(branch_scope, "rigid", {}) or {}
        if bool(self._cfg_get(rigid_cfg, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase B-R requires posterior_updater.branch_scope.rigid.enable=true.")

        sv9 = self._require_key(config, "scheduler_v9", "config")
        if bool(self._cfg_get(sv9, "enable", False)) is not True:
            raise ValueError("Stage6_0 Phase B requires scheduler_v9.enable=true.")
        if str(self._cfg_get(sv9, "phase", "")) != PHASE_B_NAME:
            raise ValueError("Stage6_0 Phase B requires scheduler_v9.phase=phase_B_viewset_rollout.")
        if bool(self._cfg_get(self._cfg_get(config, "scheduler_v8", {}) or {}, "enable", False)):
            raise ValueError("Stage6_0 runtime must not enable scheduler_v8.")
        phase_b = self._require_key(sv9, "phase_B", "scheduler_v9")
        masks = self._require_key(phase_b, "masks", "scheduler_v9.phase_B")
        required_masks = {
            "vsm_scope": "bg_rigid",
            "evidence_mask": "non_sky_non_egocar",
            "prefix_loss_mask": "non_sky_non_egocar",
            "query_label_mask": "non_sky_non_egocar",
        }
        for key, expected in required_masks.items():
            actual = str(self._cfg_get(masks, key, ""))
            if "dynamic" in actual:
                raise ValueError("Stage6_0 Phase B-R must not use dynamic mask policies.")
            if actual != expected:
                raise ValueError(f"Stage6_0 Phase B requires scheduler_v9.phase_B.masks.{key}={expected}.")
        rollout = self._cfg_get(phase_b, "rollout", {}) or {}
        rollout_mode = str(self._cfg_get(rollout, "mode", "random_viewset_local"))
        if rollout_mode == "episode_stream_tbptt":
            if str(self._cfg_get(rollout, "sample_event_frames", "")) != "sequential_blocks_in_episode":
                raise ValueError("Stage6_0 Phase B strict TBPTT requires sequential_blocks_in_episode.")
            if str(self._cfg_get(rollout, "event_order", "chronological")) != "chronological":
                raise ValueError("Stage6_0 Phase B strict TBPTT requires event_order=chronological.")
            if bool(self._cfg_get(rollout, "distinct_event_frames", True)) is not True:
                raise ValueError("Stage6_0 Phase B strict TBPTT requires distinct_event_frames=true.")
            tbptt_cfg = self._cfg_get(vsm_cfg, "tbptt", {}) or {}
            if bool(self._cfg_get(tbptt_cfg, "enable", True)) is not True:
                raise ValueError("Stage6_0 Phase B strict TBPTT requires stage6_0.vsm.tbptt.enable=true.")
            if bool(self._cfg_get(tbptt_cfg, "strict", False)) is not True:
                raise ValueError("Stage6_0 Phase B strict TBPTT requires stage6_0.vsm.tbptt.strict=true.")
            if bool(self._cfg_get(tbptt_cfg, "forbid_cache_eviction", False)) is not True:
                raise ValueError(
                    "Stage6_0 Phase B strict TBPTT requires stage6_0.vsm.tbptt.forbid_cache_eviction=true."
                )
            local_rollout = self._cfg_get(stage6, "local_rollout", {}) or {}
            if str(self._cfg_get(local_rollout, "writeback_policy", "")) != "tbptt_cache_only":
                raise ValueError("Stage6_0 Phase B strict TBPTT requires local_rollout.writeback_policy=tbptt_cache_only.")
        elif rollout_mode == "episode_block_repeat_tbptt":
            raise ValueError("episode_block_repeat_tbptt is deprecated; use episode_grouped_repeat_tbptt.")
        elif rollout_mode == "episode_grouped_repeat_tbptt":
            if str(self._cfg_get(rollout, "sample_event_frames", "")) != "sequential_blocks_in_episode":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires sequential_blocks_in_episode.")
            if str(self._cfg_get(rollout, "event_order", "chronological")) != "chronological":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires event_order=chronological.")
            if bool(self._cfg_get(rollout, "distinct_event_frames", True)) is not True:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires distinct_event_frames=true.")
            block_cfg = self._cfg_get(sv9, "block", {}) or {}
            if int(self._cfg_get(block_cfg, "steps_per_block", 1)) != 1:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires scheduler_v9.block.steps_per_block=1.")
            if str(self._cfg_get(rollout, "repeat_source_frame_policy", "fixed_within_block")) != "fixed_within_block":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires repeat_source_frame_policy=fixed_within_block.")
            if str(self._cfg_get(rollout, "repeat_memory_write_policy", "first_repeat_only")) != "first_repeat_only":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires repeat_memory_write_policy=first_repeat_only.")
            if str(self._cfg_get(rollout, "evidence_recompute_policy", "every_repeat")) != "every_repeat":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires evidence_recompute_policy=every_repeat.")
            if bool(self._cfg_get(self._cfg_get(phase_b, "query_observation", {}) or {}, "allow_empty_on_last_chunk", False)):
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires allow_empty_on_last_chunk=false.")
            patterns = list(self._cfg_get(rollout, "repeat_patterns", []) or [])
            if not patterns:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires repeat_patterns.")
            max_inner_k = int(self._cfg_get(rollout, "max_inner_K", 8))
            if max_inner_k < 1:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires max_inner_K >= 1.")
            blocks_per_episode = int(self._cfg_get(self._require_key(sv9, "episode", "scheduler_v9"), "blocks_per_episode", 0))
            all_patterns = [dict(x) for x in patterns]
            for stage in list(self._cfg_get(rollout, "curriculum", []) or []):
                all_patterns.extend(dict(x) for x in list(self._cfg_get(stage, "repeat_patterns", []) or []))
            for pattern in all_patterns:
                r = int(self._cfg_get(pattern, "repeats_per_block", 0) or 0)
                b = int(self._cfg_get(pattern, "blocks_per_chunk", 0) or 0)
                if r < 1 or b < 1:
                    raise ValueError("Stage6_0 Phase B grouped repeat patterns require repeats_per_block and blocks_per_chunk >= 1.")
                if int(r * b) > max_inner_k:
                    raise ValueError("Stage6_0 Phase B grouped repeat inner_K exceeds max_inner_K.")
                if b > blocks_per_episode:
                    raise ValueError("Stage6_0 Phase B grouped repeat blocks_per_chunk exceeds blocks_per_episode.")
            tbptt_cfg = self._cfg_get(vsm_cfg, "tbptt", {}) or {}
            if bool(self._cfg_get(tbptt_cfg, "enable", True)) is not True:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires stage6_0.vsm.tbptt.enable=true.")
            if bool(self._cfg_get(tbptt_cfg, "strict", False)) is not True:
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires stage6_0.vsm.tbptt.strict=true.")
            if bool(self._cfg_get(tbptt_cfg, "forbid_cache_eviction", False)) is not True:
                raise ValueError(
                    "Stage6_0 Phase B grouped repeat TBPTT requires stage6_0.vsm.tbptt.forbid_cache_eviction=true."
                )
            local_rollout = self._cfg_get(stage6, "local_rollout", {}) or {}
            if str(self._cfg_get(local_rollout, "writeback_policy", "")) != "tbptt_cache_only":
                raise ValueError("Stage6_0 Phase B grouped repeat TBPTT requires local_rollout.writeback_policy=tbptt_cache_only.")
        episode = self._require_key(sv9, "episode", "scheduler_v9")
        if bool(self._cfg_get(rollout, "distinct_event_frames", True)) and rollout_mode != "episode_grouped_repeat_tbptt":
            k_choices = [int(x) for x in list(self._cfg_get(rollout, "K_choices", [2, 4]) or [2, 4])]
            for stage in list(self._cfg_get(rollout, "curriculum", []) or []):
                k_choices.extend(int(x) for x in list(self._cfg_get(stage, "K_choices", []) or []))
            blocks_per_episode = int(self._cfg_get(episode, "blocks_per_episode", 0))
            if k_choices and max(k_choices) > blocks_per_episode:
                raise ValueError("Phase B long rollout K exceeds blocks_per_episode; silent K cap is forbidden.")

        validation_v9 = self._cfg_get(config, "validation_v9", {}) or {}
        if bool(self._cfg_get(validation_v9, "enable", False)):
            raise ValueError("Stage6_0 Phase B validation_v9 runner is not implemented; set validation_v9.enable=false.")

        losses = self._require_key(config, "losses", "config")
        phase_b_losses = self._require_key(losses, "phase_b", "losses")
        prefix_render = self._cfg_get(phase_b_losses, "prefix_render", {}) or {}
        prefix_mask = str(self._cfg_get(prefix_render, "mask_policy", "non_sky_non_egocar"))
        if "dynamic" in prefix_mask:
            raise ValueError("Stage6_0 Phase B-R must not use dynamic mask policies.")

    def _validate_stage6_0_phase_b_long_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if str(self._require_key(model_cfg, "stage", "model")) != "6_0":
            raise ValueError("6_0_phase_b requires model.stage='6_0'.")
        if str(self._cfg_get(model_cfg, "phase", "")) != PHASE_B_LONG_NAME:
            raise ValueError("6_0_phase_b requires model.phase='6_0_phase_b'.")
        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        long_cfg = self._require_key(stage6, "phase_b_long", "model.stage6_0")
        if bool(self._cfg_get(long_cfg, "enable", False)) is not True:
            raise ValueError("6_0_phase_b requires model.stage6_0.phase_b_long.enable=true.")
        sensor_cfg = self._cfg_get(long_cfg, "sensor", {}) or self._cfg_get(stage6, "base_measurement", {}) or {}
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        if str(self._cfg_get(base_measurement, "type", self._cfg_get(sensor_cfg, "base_measurement_type", ""))) != "stage5_4_v4":
            raise ValueError("6_0_phase_b requires base_measurement.type=stage5_4_v4.")
        if bool(self._cfg_get(base_measurement, "require_fused_v4", self._cfg_get(sensor_cfg, "require_fused_v4", True))) is not True:
            raise ValueError("6_0_phase_b requires fused V4 measurement.")
        if str(self._cfg_get(base_measurement, "source_evidence_grad_mode", self._cfg_get(sensor_cfg, "source_evidence_grad_mode", "no_grad_v4"))) != "no_grad_v4":
            raise ValueError("6_0_phase_b V1 requires source_evidence_grad_mode=no_grad_v4.")
        if bool(self._cfg_get(base_measurement, "detach_v4_outputs", self._cfg_get(sensor_cfg, "detach_v4_outputs", True))) is not True:
            raise ValueError("6_0_phase_b V1 requires detach_v4_outputs=true.")
        if bool(self._cfg_get(base_measurement, "train_2d_frontend", self._cfg_get(sensor_cfg, "train_2d_frontend", False))):
            raise ValueError("6_0_phase_b V1 freezes the measurement frontend.")

        struct_cfg = self._require_key(stage6, "struct_event_decoder", "model.stage6_0")
        if bool(self._cfg_get(struct_cfg, "enable", False)) is not True:
            raise ValueError("6_0_phase_b requires struct_event_decoder.enable=true.")
        q_cfg = self._cfg_get(long_cfg, "query_decoder", self._cfg_get(stage6, "query_decoder", {}) or {}) or {}
        if bool(self._cfg_get(q_cfg, "enable", False)):
            raise ValueError("6_0_phase_b V1 requires query_decoder.enable=false.")
        vsm_cfg = self._require_key(long_cfg, "vsm", "model.stage6_0.phase_b_long")
        vsm_type = str(self._cfg_get(vsm_cfg, "type", "streaming_selective_ssm"))
        if vsm_type not in {"streaming_selective_ssm", "cell_streaming_selective_ssm"}:
            raise ValueError("6_0_phase_b requires streaming_selective_ssm or cell_streaming_selective_ssm VSM.")
        if vsm_type == "streaming_selective_ssm":
            for forbidden in ("use_spatial_mamba", "use_cell_memory", "use_global_memory"):
                if bool(self._cfg_get(vsm_cfg, forbidden, False)):
                    raise ValueError(f"6_0_phase_b V1 forbids vsm.{forbidden}=true.")
        else:
            if bool(self._cfg_get(vsm_cfg, "use_cell_memory", False)) is not True:
                raise ValueError("6_0_phase_b cell_streaming_selective_ssm requires vsm.use_cell_memory=true.")
            if bool(self._cfg_get(vsm_cfg, "use_spatial_mamba", False)):
                raise ValueError("6_0_phase_b cell_streaming_selective_ssm does not support vsm.use_spatial_mamba=true yet.")
            bg_cfg = self._cfg_get(vsm_cfg, "bg", {}) or {}
            for key in ("point_context_source", "final_read_context_source"):
                value = str(self._cfg_get(bg_cfg, key, "previous_cell_global" if key == "point_context_source" else "updated_cell_global"))
                if value not in {"previous_cell_global", "updated_cell_global"}:
                    raise ValueError(
                        f"6_0_phase_b cell_streaming_selective_ssm requires vsm.bg.{key} "
                        "to be previous_cell_global or updated_cell_global."
                    )
        distant_cfg = self._cfg_get(vsm_cfg, "distant", {}) or {}
        distant_mode = str(self._cfg_get(distant_cfg, "mode", DISTANT_MODE_FROZEN))
        if distant_mode not in {DISTANT_MODE_FROZEN, DISTANT_MODE_APPEARANCE_SCALE}:
            raise ValueError("6_0_phase_b requires distant.mode=frozen_render_only or appearance_scale_only.")
        if bool(self._cfg_get(distant_cfg, "update_means", False)):
            raise ValueError("6_0_phase_b distant VSM must keep distant.update_means=false.")
        if bool(self._cfg_get(distant_cfg, "update_quat", False)):
            raise ValueError("6_0_phase_b distant VSM must keep distant.update_quat=false.")
        if distant_mode == DISTANT_MODE_APPEARANCE_SCALE:
            for key in ("update_scales", "update_opacity", "update_sh_dc"):
                if bool(self._cfg_get(distant_cfg, key, False)) is not True:
                    raise ValueError(f"6_0_phase_b distant appearance_scale_only requires distant.{key}=true.")
        dec_cfg = self._require_key(long_cfg, "offset_decoder", "model.stage6_0.phase_b_long")
        if str(self._cfg_get(dec_cfg, "input_source", "vsm_readout_only")) != "vsm_readout_only":
            raise ValueError("6_0_phase_b requires offset_decoder.input_source=vsm_readout_only.")
        if bool(self._cfg_get(dec_cfg, "allow_event_bypass", False)):
            raise ValueError("6_0_phase_b forbids offset_decoder.allow_event_bypass.")

        sched = self._require_key(config, "scheduler_long_phase_b", "config")
        if bool(self._cfg_get(sched, "enable", False)) is not True:
            raise ValueError("6_0_phase_b requires scheduler_long_phase_b.enable=true.")
        if str(self._cfg_get(sched, "version", "long_v1")) != "long_v1":
            raise ValueError("6_0_phase_b requires scheduler_long_phase_b.version=long_v1.")
        if str(self._cfg_get(sched, "phase", PHASE_B_LONG_NAME)) != PHASE_B_LONG_NAME:
            raise ValueError("6_0_phase_b requires scheduler_long_phase_b.phase=6_0_phase_b.")
        for key in ("episode_window", "anchor_sampling", "final_supervision"):
            self._require_key(sched, key, "scheduler_long_phase_b")
        if not (list(self._cfg_get(sched, "rollout_shapes", []) or []) or list(self._cfg_get(sched, "rollout_shapes_schedule", []) or [])):
            raise ValueError("6_0_phase_b requires scheduler_long_phase_b.rollout_shapes or rollout_shapes_schedule.")
        sv9 = self._cfg_get(config, "scheduler_v9", {}) or {}
        if bool(self._cfg_get(sv9, "enable", False)):
            raise ValueError("6_0_phase_b uses scheduler_long_phase_b; scheduler_v9 must stay disabled for this phase.")
        validation_v9 = self._cfg_get(config, "validation_v9", {}) or {}
        if bool(self._cfg_get(validation_v9, "enable", False)):
            raise ValueError("6_0_phase_b uses validation_long_phase_b; validation_v9 must stay disabled for this phase.")
        validation_long = self._require_key(config, "validation_long_phase_b", "config")
        if bool(self._cfg_get(validation_long, "enable", False)) is not True:
            raise ValueError("6_0_phase_b requires validation_long_phase_b.enable=true.")
        init_cfg = self._cfg_get(config, "initialization", {}) or {}
        phase_b_init = self._cfg_get(init_cfg, "phase_b_from_phase_a", {}) or {}
        if bool(self._cfg_get(phase_b_init, "enable", False)):
            if str(self._cfg_get(phase_b_init, "export_type", "stage6_0_phase_a_for_phase_b")) != "stage6_0_phase_a_for_phase_b":
                raise ValueError(
                    "6_0_phase_b initialization.phase_b_from_phase_a.export_type must be "
                    "stage6_0_phase_a_for_phase_b."
                )
            if bool(self._cfg_get(phase_b_init, "reject_plain_model_state_dict", True)) is not True:
                raise ValueError(
                    "6_0_phase_b requires initialization.phase_b_from_phase_a.reject_plain_model_state_dict=true; "
                    "plain Phase A resume checkpoints are not a supported Phase B Long bootstrap path."
                )
            for key in ("load_modules", "freeze_after_load", "train_new_modules"):
                if self._cfg_get(phase_b_init, key, None) is not None:
                    raise ValueError(
                        "6_0_phase_b initialization.phase_b_from_phase_a no longer accepts "
                        f"{key}; module loading, freezing, and trainability are fixed by the "
                        "Phase B export contract."
                    )
        losses_cfg = self._cfg_get(config, "losses", {}) or {}
        phase_b_long = self._require_key(losses_cfg, "phase_b_long", "losses")
        for key in ("query_observation", "nearby_render", "per_step_prefix_render"):
            node = self._cfg_get(phase_b_long, key, {}) or {}
            if bool(self._cfg_get(node, "enable", False)):
                raise ValueError(f"6_0_phase_b V1 requires losses.phase_b_long.{key}.enable=false.")

    def _configure_measurement_frontend_trainability(self, config: Any) -> None:
        model_cfg = self._require_key(config, "model", "config")
        self.stage6_phase = str(self._cfg_get(model_cfg, "phase", "phase_A_block_local_unroll"))
        stage6 = self._require_key(model_cfg, "stage6_0", "model")
        base_measurement = self._require_key(stage6, "base_measurement", "model.stage6_0")
        self.stage6_phase_a_mode = str(
            self._cfg_get(stage6, "phase_a_mode", self._cfg_get(base_measurement, "mode", "updater_only"))
        ).strip()
        self.stage6_source_evidence_grad_mode = str(
            self._cfg_get(base_measurement, "source_evidence_grad_mode", "no_grad_v4")
        ).strip()
        self.stage6_detach_v4_outputs = bool(self._cfg_get(base_measurement, "detach_v4_outputs", True))
        self.stage6_train_v4_lift = bool(self._cfg_get(base_measurement, "train_v4_lift", False))
        self.stage6_measurement_trainable_param_names: set[str] = set()

        for param in self.parameters():
            param.requires_grad_(False)

        if self.stage6_phase in {PHASE_B_NAME, PHASE_B_LONG_NAME}:
            self.stage6_phase_a_mode = "phase_b_frozen"
            self.stage6_source_evidence_grad_mode = "no_grad_v4"
            self.stage6_detach_v4_outputs = True
            return

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

        if self.stage6_train_v4_lift:
            for attr_name in ("feature_backprojector", "alpha_t_extractor_v4"):
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Module):
                    mark_trainable(module, attr_name)

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
        render_cfg = self._cfg_get(stage6, "render", {}) or {}
        self.stage6_render_grouped_multiview_train = bool(
            self._cfg_get(render_cfg, "grouped_multiview_train", True)
        )
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
        if str(getattr(self, "stage6_phase", "")) == PHASE_B_LONG_NAME:
            phase_b_hooks = dict(phase_b_hooks)
            phase_b_hooks["accept_vsm_ctx"] = False
        clamp_keys = (
            "means_max_step_m",
            "scales_log_max_step",
            "quat_axis_angle_max_step_rad",
            "opacity_logit_max_step",
            "sh_max_step",
            "hidden_max_step",
        )
        branch_clamps_cfg = self._cfg_get(updater_cfg, "branch_clamps", {}) or {}
        branch_clamps: Dict[str, Dict[str, float]] = {}
        for branch in ("bg", "distant", "rigid"):
            cfg = self._cfg_get(branch_clamps_cfg, branch, {}) or {}
            branch_clamps[branch] = {
                key: float(value)
                for key in clamp_keys
                if (value := self._cfg_get(cfg, key, None)) is not None
            }
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
            branch_clamps=branch_clamps,
        ).to(self.device)
        vsm_cfg = self._cfg_get(stage6, "vsm", {}) or {}
        query_cfg = self._cfg_get(stage6, "query_decoder", {}) or {}
        self.stage6_vsm: Optional[Stage6ViewSetMemory] = None
        self.stage6_query_decoder: Optional[Stage6QueryDecoder] = None
        self.stage6_long_vsm: Optional[LongStreamingVSM] = None
        self.stage6_long_offset_decoder: Optional[VSMOffsetDecoder] = None
        self.stage6_phase_b_tbptt_cache: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
        if bool(self._cfg_get(vsm_cfg, "enable", False)):
            vsm_bg_cfg = self._cfg_get(vsm_cfg, "bg", {}) or {}
            vsm_rigid_cfg = self._cfg_get(vsm_cfg, "rigid", {}) or {}
            self.stage6_vsm = Stage6ViewSetMemory(
                event_dim=int(self.stage6_event_dim),
                view_code_dim=2,
                num_tokens=int(self._cfg_get(vsm_cfg, "num_tokens", self._cfg_get(vsm_bg_cfg, "num_tokens", 4))),
                token_dim=int(self._cfg_get(vsm_cfg, "token_dim", self._cfg_get(vsm_bg_cfg, "token_dim", self.stage6_event_dim))),
                proto_dim=int(self._cfg_get(vsm_cfg, "proto_dim", self._cfg_get(vsm_bg_cfg, "proto_dim", 8))),
                global_dim=int(self._cfg_get(vsm_cfg, "global_dim", self._cfg_get(vsm_bg_cfg, "global_dim", self.stage6_event_dim))),
                ctx_dim=int(
                    self._cfg_get(
                        vsm_cfg,
                        "ctx_dim",
                        self._cfg_get(vsm_bg_cfg, "ctx_dim", self._cfg_get(phase_b_hooks, "vsm_ctx_dim", self.stage6_event_dim)),
                    )
                ),
                hidden_dim=int(self._cfg_get(vsm_cfg, "hidden_dim", self._cfg_get(vsm_bg_cfg, "hidden_dim", max(96, self.stage6_event_dim)))),
                bg_zero_unseen_ctx=bool(self._cfg_get(vsm_bg_cfg, "zero_unseen_ctx", False)),
                rigid_zero_unseen_ctx=bool(self._cfg_get(vsm_rigid_cfg, "zero_unseen_ctx", True)),
            ).to(self.device)
        if bool(self._cfg_get(query_cfg, "enable", False)):
            self.stage6_query_decoder = Stage6QueryDecoder(
                input_dim=int(self._cfg_get(query_cfg, "input_dim", self._cfg_get(phase_b_hooks, "vsm_ctx_dim", self.stage6_event_dim))),
                event_dim=int(self.stage6_event_dim),
                obs_code_dim=int(self._cfg_get(query_cfg, "obs_code_dim", 2)),
                hidden_dim=int(self._cfg_get(query_cfg, "hidden_dim", max(96, self.stage6_event_dim))),
            ).to(self.device)
        long_cfg = self._cfg_get(stage6, "phase_b_long", {}) or {}
        if bool(self._cfg_get(long_cfg, "enable", False)):
            long_vsm_cfg = self._cfg_get(long_cfg, "vsm", {}) or {}
            long_bg_cfg = self._cfg_get(long_vsm_cfg, "bg", {}) or {}
            long_rigid_cfg = self._cfg_get(long_vsm_cfg, "rigid", {}) or {}
            long_distant_cfg = self._cfg_get(long_vsm_cfg, "distant", {}) or {}
            mem_long_cfg = self._cfg_get(self._cfg_get(config, "memory", {}) or {}, "phase_b_long", {}) or {}
            self.stage6_phase_b_long_vsm_type = str(
                self._cfg_get(long_vsm_cfg, "type", "streaming_selective_ssm")
            )
            self.stage6_phase_b_long_distant_mode = str(
                self._cfg_get(long_distant_cfg, "mode", "frozen_render_only")
            )
            self.stage6_phase_b_long_vsm_dtype = str(self._cfg_get(long_vsm_cfg, "dtype", "bf16"))
            self.stage6_phase_b_long_amp_dtype = str(
                self._cfg_get(mem_long_cfg, "amp_dtype", self.stage6_phase_b_long_vsm_dtype)
            )
            self._phase_b_long_autocast_torch_dtype(self.stage6_phase_b_long_amp_dtype)
            if self.stage6_phase_b_long_vsm_type == "cell_streaming_selective_ssm":
                bg_cell_mem_dim = int(self._cfg_get(long_bg_cfg, "cell_mem_dim", self._cfg_get(long_bg_cfg, "mem_dim", 64)))
                bg_read_dim = int(self._cfg_get(long_bg_cfg, "read_dim", bg_cell_mem_dim))
                rigid_object_mem_dim = int(self._cfg_get(long_rigid_cfg, "object_mem_dim", self._cfg_get(long_rigid_cfg, "mem_dim", 64)))
                rigid_read_dim = int(self._cfg_get(long_rigid_cfg, "read_dim", self._cfg_get(long_rigid_cfg, "mem_dim", rigid_object_mem_dim)))
                local_grid_raw = list(self._cfg_get(long_rigid_cfg, "local_grid", [8, 8, 4]) or [8, 8, 4])
                if len(local_grid_raw) != 3:
                    raise ValueError("phase_b_long.vsm.rigid.local_grid must have length 3.")
                self.stage6_phase_b_long_bg_mem_dim = int(bg_read_dim)
                self.stage6_phase_b_long_rigid_mem_dim = int(rigid_read_dim)
                self.stage6_long_vsm = LongCellStreamingVSM(
                    event_dim=int(self.stage6_event_dim),
                    view_dim=2,
                    bg_point_mem_dim=int(self._cfg_get(long_bg_cfg, "point_mem_dim", 32)),
                    bg_cell_mem_dim=int(bg_cell_mem_dim),
                    bg_global_mem_dim=int(self._cfg_get(long_bg_cfg, "global_mem_dim", 64)),
                    bg_read_dim=int(bg_read_dim),
                    bg_cell_voxel_size=float(self._cfg_get(long_bg_cfg, "voxel_size", self._cfg_get(long_bg_cfg, "cell_voxel_size", 0.5))),
                    use_global_memory=bool(self._cfg_get(long_vsm_cfg, "use_global_memory", self._cfg_get(long_bg_cfg, "use_global_memory", True))),
                    rigid_point_mem_dim=int(self._cfg_get(long_rigid_cfg, "point_mem_dim", 32)),
                    rigid_object_mem_dim=int(rigid_object_mem_dim),
                    rigid_cell_mem_dim=int(self._cfg_get(long_rigid_cfg, "cell_mem_dim", 64)),
                    rigid_read_dim=int(rigid_read_dim),
                    rigid_local_grid=(int(local_grid_raw[0]), int(local_grid_raw[1]), int(local_grid_raw[2])),
                    distant_mem_dim=int(self._cfg_get(long_distant_cfg, "mem_dim", 32)),
                    input_dim=int(self._cfg_get(long_vsm_cfg, "input_dim", max(96, self.stage6_event_dim))),
                    dtype=str(self.stage6_phase_b_long_vsm_dtype),
                    distant_mode=str(self.stage6_phase_b_long_distant_mode),
                    support_fallback_when_no_valid=bool(
                        self._cfg_get(long_vsm_cfg, "support_fallback_when_no_valid", False)
                    ),
                    support_fallback_min=float(self._cfg_get(long_vsm_cfg, "support_fallback_min", 0.0)),
                    support_fallback_scale=float(self._cfg_get(long_vsm_cfg, "support_fallback_scale", 1.0)),
                    bg_active_sparse=bool(self._cfg_get(long_bg_cfg, "active_sparse", True)),
                    bg_outside_policy=str(self._cfg_get(long_bg_cfg, "outside_policy", "mark_invalid")),
                    bg_point_context_source=str(
                        self._cfg_get(long_bg_cfg, "point_context_source", "previous_cell_global")
                    ),
                    bg_final_read_context_source=str(
                        self._cfg_get(long_bg_cfg, "final_read_context_source", "updated_cell_global")
                    ),
                ).to(self.device)
            else:
                self.stage6_phase_b_long_bg_mem_dim = int(self._cfg_get(long_bg_cfg, "mem_dim", 64))
                self.stage6_phase_b_long_rigid_mem_dim = int(self._cfg_get(long_rigid_cfg, "mem_dim", 64))
                self.stage6_long_vsm = LongStreamingVSM(
                    event_dim=int(self.stage6_event_dim),
                    view_dim=2,
                    bg_mem_dim=int(self.stage6_phase_b_long_bg_mem_dim),
                    rigid_mem_dim=int(self.stage6_phase_b_long_rigid_mem_dim),
                    distant_mem_dim=int(self._cfg_get(long_distant_cfg, "mem_dim", 32)),
                    input_dim=int(self._cfg_get(long_vsm_cfg, "input_dim", max(96, self.stage6_event_dim))),
                    dtype=str(self.stage6_phase_b_long_vsm_dtype),
                    distant_mode=str(self.stage6_phase_b_long_distant_mode),
                    support_fallback_when_no_valid=bool(
                        self._cfg_get(long_vsm_cfg, "support_fallback_when_no_valid", False)
                    ),
                    support_fallback_min=float(self._cfg_get(long_vsm_cfg, "support_fallback_min", 0.0)),
                    support_fallback_scale=float(self._cfg_get(long_vsm_cfg, "support_fallback_scale", 1.0)),
                    bg_active_sparse=bool(self._cfg_get(long_bg_cfg, "active_sparse", True)),
                ).to(self.device)
            dec_cfg = self._cfg_get(long_cfg, "offset_decoder", {}) or {}
            clamps = self._cfg_get(dec_cfg, "clamps", {}) or {}
            self.stage6_phase_b_long_offset_dtype = str(
                self._cfg_get(mem_long_cfg, "offset_state_dtype", self._cfg_get(long_vsm_cfg, "dtype", "bf16"))
            )
            dec_update_scope = self._cfg_get(dec_cfg, "update_scope", {}) or {}
            dec_distant_scope = self._cfg_get(dec_update_scope, "distant", {}) or {}
            distant_sh_rest_bases = max(int(self.sh_degree + 1) ** 2 - 1, 0)
            self.stage6_long_offset_decoder = VSMOffsetDecoder(
                bg_mem_dim=int(getattr(self, "stage6_phase_b_long_bg_mem_dim", self._cfg_get(long_bg_cfg, "mem_dim", 64))),
                rigid_mem_dim=int(getattr(self, "stage6_phase_b_long_rigid_mem_dim", self._cfg_get(long_rigid_cfg, "mem_dim", 64))),
                distant_mem_dim=int(self._cfg_get(long_distant_cfg, "mem_dim", 32)),
                distant_sh_rest_bases=int(distant_sh_rest_bases),
                distant_sh_rest_update_bases=int(
                    self._cfg_get(
                        dec_distant_scope,
                        "sh_rest_lowfreq_bases",
                        self._cfg_get(
                            long_distant_cfg,
                            "sh_rest_lowfreq_bases",
                            min(int(distant_sh_rest_bases), 3),
                        ),
                    )
                ),
                hidden_dim=int(self._cfg_get(dec_cfg, "hidden_dim", 128)),
                clamps=dict(clamps),
                distant_mode=str(self.stage6_phase_b_long_distant_mode),
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
        phase_b = self._cfg_get(losses_cfg, "phase_b", {}) or {}
        prefix_render = self._cfg_get(phase_b, "prefix_render", {}) or {}
        query_observation = self._cfg_get(phase_b, "query_observation", {}) or {}
        phase_b_reg = self._cfg_get(phase_b, "regularization", {}) or {}
        tbptt_cfg = self._cfg_get(self._cfg_get(stage6, "vsm", {}) or {}, "tbptt", {}) or {}
        self.stage6_phase_b_prefix_enable = bool(self._cfg_get(prefix_render, "enable", True))
        self.stage6_phase_b_prefix_weight = float(self._cfg_get(prefix_render, "weight", 1.0))
        self.stage6_phase_b_prefix_l1_weight = float(self._cfg_get(prefix_render, "l1_weight", 0.8))
        self.stage6_phase_b_prefix_ssim_weight = float(self._cfg_get(prefix_render, "ssim_weight", 0.2))
        self.stage6_phase_b_prefix_mask_policy = str(
            self._cfg_get(prefix_render, "mask_policy", "non_sky_non_egocar")
        )
        if str(getattr(self, "stage6_phase", "")) == PHASE_B_NAME and "dynamic" in self.stage6_phase_b_prefix_mask_policy:
            raise ValueError("Stage6_0 Phase B-R must not use dynamic mask policies.")
        self.stage6_phase_b_prefix_step_weight = str(self._cfg_get(prefix_render, "step_weight", "late_heavy_linear"))
        self.stage6_phase_b_query_enable = bool(self._cfg_get(query_observation, "enable", True))
        self.stage6_phase_b_query_weight = float(self._cfg_get(query_observation, "weight", 0.05))
        self.stage6_phase_b_query_warmup_steps = int(self._cfg_get(query_observation, "warmup_steps", 5000))
        self.stage6_phase_b_query_event_weight = float(
            self._cfg_get(query_observation, "event_weight", self._cfg_get(query_observation, "event_bg_weight", 1.0))
        )
        self.stage6_phase_b_query_visible_weight = float(self._cfg_get(query_observation, "visible_weight", 0.2))
        self.stage6_phase_b_query_support_weight = float(self._cfg_get(query_observation, "support_weight", 0.2))
        self.stage6_phase_b_query_obs_code_weight = float(self._cfg_get(query_observation, "obs_code_weight", 0.1))
        self.stage6_phase_b_delta_norm_weight = float(self._cfg_get(phase_b_reg, "delta_norm_weight", 1.0e-3))
        self.stage6_phase_b_tbptt_enable = bool(self._cfg_get(tbptt_cfg, "enable", True))
        self.stage6_phase_b_tbptt_max_items = int(self._cfg_get(tbptt_cfg, "max_items", 8))
        self.stage6_phase_b_tbptt_strict = bool(self._cfg_get(tbptt_cfg, "strict", False))
        self.stage6_phase_b_tbptt_forbid_cache_eviction = bool(
            self._cfg_get(tbptt_cfg, "forbid_cache_eviction", False)
        )
        phase_b_long = self._cfg_get(losses_cfg, "phase_b_long", {}) or {}
        final_history = self._cfg_get(phase_b_long, "final_history_render", {}) or {}
        final_current = self._cfg_get(phase_b_long, "final_current_render", {}) or {}
        final_history_recon = self._cfg_get(phase_b_long, "final_history_recon_render", final_history) or final_history
        final_history_nvs = self._cfg_get(phase_b_long, "final_history_nvs_render", final_history) or final_history
        final_current_recon = self._cfg_get(phase_b_long, "final_current_recon_render", final_current) or final_current
        final_current_nvs = self._cfg_get(phase_b_long, "final_current_nvs_render", final_current) or final_current
        offset_reg = self._cfg_get(phase_b_long, "offset_regularization", {}) or {}
        self.stage6_phase_b_long_history_weight = float(self._cfg_get(final_history, "weight", 1.0))
        self.stage6_phase_b_long_history_l1_weight = float(self._cfg_get(final_history, "l1_weight", 0.8))
        self.stage6_phase_b_long_history_ssim_weight = float(self._cfg_get(final_history, "ssim_weight", 0.2))
        self.stage6_phase_b_long_history_mask_policy = str(
            self._cfg_get(final_history, "mask_policy", "non_sky_non_egocar")
        )
        self.stage6_phase_b_long_current_weight = float(self._cfg_get(final_current, "weight", 1.0))
        self.stage6_phase_b_long_current_l1_weight = float(self._cfg_get(final_current, "l1_weight", 0.8))
        self.stage6_phase_b_long_current_ssim_weight = float(self._cfg_get(final_current, "ssim_weight", 0.2))
        self.stage6_phase_b_long_current_mask_policy = str(
            self._cfg_get(final_current, "mask_policy", "non_sky_non_egocar")
        )
        self.stage6_phase_b_long_role_render_cfg = {
            "history_recon": dict(final_history_recon),
            "history_nvs": dict(final_history_nvs),
            "current_recon": dict(final_current_recon),
            "current_nvs": dict(final_current_nvs),
        }
        self.stage6_phase_b_long_offset_reg_cfg = dict(offset_reg)
        self.stage6_phase_b_long_offset_reg_weight = float(self._cfg_get(offset_reg, "weight", 1.0e-4))

    def _configure_stage6_trainability_after_module_init(self, config: Any) -> None:
        model_cfg = self._require_key(config, "model", "config")
        phase = str(self._cfg_get(model_cfg, "phase", "phase_A_block_local_unroll"))
        if phase == PHASE_B_LONG_NAME:
            for param in self.parameters():
                param.requires_grad_(False)
            if self.stage6_long_vsm is None:
                raise ValueError("6_0_phase_b internal error: stage6_long_vsm was not initialized.")
            if self.stage6_long_offset_decoder is None:
                raise ValueError("6_0_phase_b internal error: stage6_long_offset_decoder was not initialized.")
            for param in self.stage6_long_vsm.parameters():
                param.requires_grad_(True)
            for param in self.stage6_long_offset_decoder.parameters():
                param.requires_grad_(True)
            return
        if phase != PHASE_B_NAME:
            return
        sv9 = self._cfg_get(config, "scheduler_v9", {}) or {}
        phase_b = self._cfg_get(sv9, "phase_B", {}) or {}
        rollout = self._cfg_get(phase_b, "rollout", {}) or {}
        if str(self._cfg_get(rollout, "mode", "")) == "episode_rollout_grouped_repeat_tbptt":
            for param in self.parameters():
                param.requires_grad_(False)
            if self.stage6_long_vsm is None:
                raise ValueError("Stage6_0 Phase B final rollout internal error: stage6_long_vsm was not initialized.")
            if self.stage6_long_offset_decoder is None:
                raise ValueError(
                    "Stage6_0 Phase B final rollout internal error: stage6_long_offset_decoder was not initialized."
                )
            for param in self.stage6_long_vsm.parameters():
                param.requires_grad_(True)
            for param in self.stage6_long_offset_decoder.parameters():
                param.requires_grad_(True)
            return
        for param in self.parameters():
            param.requires_grad_(False)
        if self.stage6_vsm is None:
            raise ValueError("Stage6_0 Phase B internal error: VSM module was not initialized.")
        if self.stage6_query_decoder is None:
            raise ValueError("Stage6_0 Phase B internal error: QueryDecoder module was not initialized.")
        for param in self.stage6_vsm.parameters():
            param.requires_grad_(True)
        for param in self.stage6_query_decoder.parameters():
            param.requires_grad_(True)
        adapter = getattr(self.stage6_posterior_updater, "vsm_ctx_adapter", None)
        if adapter is None:
            raise ValueError("Stage6_0 Phase B requires posterior_updater.vsm_ctx_adapter.")
        for param in adapter.parameters():
            param.requires_grad_(True)

    def _parse_stage6_branch_scope(self, updater_cfg: Any) -> Dict[str, Dict[str, bool]]:
        raw = self._cfg_get(updater_cfg, "branch_scope", {}) or {}
        defaults = {
            "bg": {
                "enable": True,
                "update_means": True,
                "update_scales": True,
                "update_quat": True,
                "update_opacity": True,
                "update_sh": True,
            },
            "distant": {
                "enable": True,
                "update_means": False,
                "update_scales": False,
                "update_quat": False,
                "update_opacity": True,
                "update_sh": True,
            },
            "rigid": {
                "enable": True,
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
            enabled = bool(self._cfg_get(cfg, "enable", branch_defaults.get("enable", True)))
            out[branch] = {
                key: bool(self._cfg_get(cfg, key, default))
                for key, default in branch_defaults.items()
            }
            out[branch]["enable"] = enabled
            if not enabled:
                for key in list(out[branch].keys()):
                    if key.startswith("update_"):
                        out[branch][key] = False
                out[branch]["update_hidden"] = False
            elif "update_hidden" not in out[branch]:
                out[branch]["update_hidden"] = True
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
                decay_ids = {id(param) for param in decay}
                groups.append(
                    {
                        "params": decay,
                        "lr": float(lr),
                        "weight_decay": float(wd),
                        "logical_name": logical_name,
                        "name": logical_name,
                        "param_names": [name for name, param in unique_named if id(param) in decay_ids],
                    }
                )
            if no_decay:
                no_decay_ids = {id(param) for param in no_decay}
                no_decay_name = f"{logical_name}_no_weight_decay"
                groups.append(
                    {
                        "params": no_decay,
                        "lr": float(lr),
                        "weight_decay": 0.0,
                        "logical_name": no_decay_name,
                        "name": no_decay_name,
                        "param_names": [name for name, param in unique_named if id(param) in no_decay_ids],
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
                if not name.startswith("vsm_ctx_adapter.")
            ],
            lr=lr_for("posterior_updater"),
            wd=weight_decay,
        )
        adapter = getattr(self.stage6_posterior_updater, "vsm_ctx_adapter", None)
        if adapter is not None:
            add_group(
                logical_name="stage6_vsm_ctx_adapter",
                named_params=[
                    (f"stage6_posterior_updater.vsm_ctx_adapter.{name}", param)
                    for name, param in adapter.named_parameters()
                ],
                lr=lr_for("vsm_ctx_adapter"),
                wd=weight_decay,
            )
        if getattr(self, "stage6_vsm", None) is not None:
            add_group(
                logical_name="stage6_vsm",
                named_params=[
                    (f"stage6_vsm.{name}", param)
                    for name, param in self.stage6_vsm.named_parameters()  # type: ignore[union-attr]
                ],
                lr=lr_for("vsm"),
                wd=weight_decay,
            )
        if getattr(self, "stage6_query_decoder", None) is not None:
            add_group(
                logical_name="stage6_query_decoder",
                named_params=[
                    (f"stage6_query_decoder.{name}", param)
                    for name, param in self.stage6_query_decoder.named_parameters()  # type: ignore[union-attr]
                ],
                lr=lr_for("query_decoder"),
                wd=weight_decay,
            )
        if getattr(self, "stage6_long_vsm", None) is not None:
            add_group(
                logical_name="stage6_long_vsm",
                named_params=[
                    (f"stage6_long_vsm.{name}", param)
                    for name, param in self.stage6_long_vsm.named_parameters()  # type: ignore[union-attr]
                ],
                lr=lr_for("long_vsm"),
                wd=weight_decay,
            )
        if getattr(self, "stage6_long_offset_decoder", None) is not None:
            add_group(
                logical_name="stage6_long_offset_decoder",
                named_params=[
                    (f"stage6_long_offset_decoder.{name}", param)
                    for name, param in self.stage6_long_offset_decoder.named_parameters()  # type: ignore[union-attr]
                ],
                lr=lr_for("offset_decoder"),
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

    def _stage6_optimizer_signature(self) -> Dict[str, Any]:
        name_by_id = {id(param): str(name) for name, param in self.named_parameters()}
        groups: List[Dict[str, Any]] = []
        for group in self.optimizer.param_groups:
            param_names = [
                name_by_id.get(id(param), "")
                for param in list(group.get("params", []))
            ]
            groups.append(
                {
                    "name": str(group.get("name", group.get("logical_name", ""))),
                    "num_params": int(len(param_names)),
                    "param_names": [str(x) for x in param_names],
                }
            )
        return {"num_groups": int(len(groups)), "groups": groups}

    @staticmethod
    def _stage6_ckpt_cache_key(key: Any) -> Tuple[int, int]:
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            return (int(key[0]), int(key[1]))
        raise ValueError(f"Invalid Stage6 runtime cache key: {key!r}")

    @staticmethod
    def _stage6_ckpt_clone_state(state: Any) -> Any:
        if state is None:
            return None
        if hasattr(state, "detach_clone"):
            out = state.detach_clone()
        else:
            out = copy.deepcopy(state)
        for name, value in vars(out).items():
            if torch.is_tensor(value):
                setattr(out, name, value.detach().cpu().clone())
        return out

    def _stage6_ckpt_state_to_device(self, state: Any) -> Any:
        if state is None:
            return None
        out = copy.deepcopy(state)
        for name, value in vars(out).items():
            if torch.is_tensor(value):
                setattr(out, name, value.to(self.device))
        return out

    @staticmethod
    def _stage6_ckpt_clone_hidden_cache(cache: Dict[Any, torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        return {
            MinimalStreetForwardStage6_0._stage6_ckpt_cache_key(key): value.detach().cpu().clone()
            for key, value in dict(cache).items()
            if torch.is_tensor(value)
        }

    def build_runtime_checkpoint_extra(self) -> Dict[str, Any]:
        def clone_state_cache(cache: Dict[Any, Any]) -> Dict[Tuple[int, int], Any]:
            return {
                self._stage6_ckpt_cache_key(key): self._stage6_ckpt_clone_state(value)
                for key, value in dict(cache).items()
            }

        return {
            "model_runtime_state": {
                "runtime_format": "stage6_0_node_state_runtime_v1",
                "node_states_bg": clone_state_cache(getattr(self, "node_states_bg", {})),
                "node_states_distant": clone_state_cache(getattr(self, "node_states_distant", {})),
                "node_states_rigid": clone_state_cache(getattr(self, "node_states_rigid", {})),
                "node_states_sky": clone_state_cache(getattr(self, "node_states_sky", {})),
                "h_cache_bg": self._stage6_ckpt_clone_hidden_cache(getattr(self, "h_cache_bg", {})),
                "h_cache_distant": self._stage6_ckpt_clone_hidden_cache(getattr(self, "h_cache_distant", {})),
                "h_cache_rigid": self._stage6_ckpt_clone_hidden_cache(getattr(self, "h_cache_rigid", {})),
                "h_cache_sky": self._stage6_ckpt_clone_hidden_cache(getattr(self, "h_cache_sky", {})),
            }
        }

    def load_runtime_state_from_checkpoint(self, payload: Dict[str, Any]) -> bool:
        runtime = payload.get("model_runtime_state")
        if not isinstance(runtime, dict):
            logger.warning("Resume checkpoint has no model_runtime_state; Stage6 node-state runtime is fresh.")
            return False

        def restore_state_cache(cache_name: str) -> None:
            raw = runtime.get(cache_name, {})
            if not isinstance(raw, dict):
                raise ValueError(f"model_runtime_state.{cache_name} must be a dict")
            restored = {
                self._stage6_ckpt_cache_key(key): self._stage6_ckpt_state_to_device(value)
                for key, value in raw.items()
            }
            setattr(self, cache_name, restored)

        def restore_hidden_cache(cache_name: str) -> None:
            raw = runtime.get(cache_name, {})
            if not isinstance(raw, dict):
                raise ValueError(f"model_runtime_state.{cache_name} must be a dict")
            restored = {
                self._stage6_ckpt_cache_key(key): value.to(self.device)
                for key, value in raw.items()
                if torch.is_tensor(value)
            }
            setattr(self, cache_name, restored)

        for cache_name in ("node_states_bg", "node_states_distant", "node_states_rigid", "node_states_sky"):
            restore_state_cache(cache_name)
        for cache_name in ("h_cache_bg", "h_cache_distant", "h_cache_rigid", "h_cache_sky"):
            restore_hidden_cache(cache_name)
        logger.info(
            "Restored Stage6 runtime node-state caches: bg=%s distant=%s rigid=%s sky=%s.",
            len(getattr(self, "node_states_bg", {})),
            len(getattr(self, "node_states_distant", {})),
            len(getattr(self, "node_states_rigid", {})),
            len(getattr(self, "node_states_sky", {})),
        )
        return True

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

    def _stage6_rigid_point_valid_mask(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, int(frame_idx))
        if resolved is not None:
            return self._rigid_point_valid_mask(node_state_rigid, int(frame_idx))

        warned = getattr(self, "_stage6_missing_rigid_frame_warned", set())
        key = int(frame_idx)
        if key not in warned:
            if len(warned) < 8:
                logger.warning(
                    "Stage6_0 rigid frame_idx=%s missing in dynamic_info frame_ids=%s; "
                    "treat rigid branch as invisible for this frame.",
                    int(frame_idx),
                    list(node_state_rigid.frame_ids),
                )
            elif len(warned) == 8:
                logger.warning(
                    "Stage6_0 encountered more rigid frames missing from dynamic_info; "
                    "further missing-rigid-frame warnings are suppressed."
                )
            warned = set(warned)
            warned.add(key)
            self._stage6_missing_rigid_frame_warned = warned
        return torch.zeros(
            (int(node_state_rigid.means.shape[0]),),
            dtype=torch.bool,
            device=node_state_rigid.means.device,
        )

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
                mask_src_rigid = self._stage6_rigid_point_valid_mask(rigid_m, int(source_frame_idx))
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
            hidden=delta.hidden if bool(scope.get("update_hidden", True)) else torch.zeros_like(delta.hidden),
            confidence=delta.confidence,
            noop=delta.noop,
        )

    def _apply_branch_scope(self, delta: DeltaPack) -> DeltaPack:
        return DeltaPack(
            bg=self._mask_branch_delta(delta.bg, self.stage6_branch_scope["bg"]),
            distant=(
                self._mask_branch_delta(delta.distant, self.stage6_branch_scope["distant"])
                if delta.distant is not None and bool(self.stage6_branch_scope["distant"].get("enable", True))
                else None
            ),
            rigid=(
                self._mask_branch_delta(delta.rigid, self.stage6_branch_scope["rigid"])
                if delta.rigid is not None and bool(self.stage6_branch_scope["rigid"].get("enable", True))
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
        num_distant_total = int(local_state.distant.means.shape[0]) if local_state.distant is not None else 0
        include_distant_event = num_distant_total > 0 and not self._phase_b_skip_distant_event()
        num_distant = int(num_distant_total) if bool(include_distant_event) else 0
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
        if bool(include_distant_event):
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

    @staticmethod
    def _event_with_default_view_code(event: Any) -> Any:
        if getattr(event, "view_code_bg", None) is None and getattr(event, "obs_code_bg", None) is not None:
            event.view_code_bg = event.obs_code_bg
        if getattr(event, "view_code_distant", None) is None and getattr(event, "obs_code_distant", None) is not None:
            event.view_code_distant = event.obs_code_distant
        if getattr(event, "view_code_rigid", None) is None and getattr(event, "obs_code_rigid", None) is not None:
            event.view_code_rigid = event.obs_code_rigid
        return event

    @staticmethod
    def _detach_event_pack(event: Any) -> Any:
        for name, value in list(event.__dict__.items()):
            if torch.is_tensor(value):
                setattr(event, name, value.detach())
        return event

    def _build_stage6_event_from_measurement(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Any:
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
        return self._event_with_default_view_code(event)

    def _apply_event_update(
        self,
        *,
        local_state: LocalGSState,
        event: Any,
        ctx_vsm: Optional[ContextPack] = None,
    ) -> tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        delta, aux = self.stage6_posterior_updater(event=event, ctx_current=None, ctx_vsm=ctx_vsm)
        self._mem_debug("encode/after_posterior")
        route = event.route
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

    def _encode_and_update(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        event = self._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)
        return self._apply_event_update(local_state=local_state, event=event, ctx_vsm=None)

    @staticmethod
    def _branch_render_params(branch: Any, *, detach: bool = False) -> Dict[str, torch.Tensor]:
        means = branch.means.detach() if bool(detach) else branch.means
        scales_log = branch.scales_log.detach() if bool(detach) else branch.scales_log
        quats = branch.quats.detach() if bool(detach) else branch.quats
        opacity_logit = branch.opacity_logit.detach() if bool(detach) else branch.opacity_logit
        sh_dc = branch.sh_dc.detach() if bool(detach) else branch.sh_dc
        sh_rest = branch.sh_rest.detach() if bool(detach) else branch.sh_rest
        return {
            "means_r": means,
            "scales_r": torch.exp(scales_log),
            "quats_r": quats,
            "opacities_r": torch.sigmoid(opacity_logit).squeeze(-1),
            "colors_r": torch.cat([sh_dc[:, None, :], sh_rest], dim=1),
        }

    def _phase_b_freeze_distant_branch(self, local_state: LocalGSState) -> LocalGSState:
        if str(getattr(self, "stage6_phase", "")) != PHASE_B_NAME:
            return local_state
        if bool(self.stage6_branch_scope.get("distant", {}).get("enable", True)):
            return local_state
        if local_state.distant is None:
            return local_state
        return replace(
            local_state,
            distant=replace(
                local_state.distant,
                means=local_state.distant.means.detach(),
                scales_log=local_state.distant.scales_log.detach(),
                quats=local_state.distant.quats.detach(),
                opacity_logit=local_state.distant.opacity_logit.detach(),
                sh_dc=local_state.distant.sh_dc.detach(),
                sh_rest=local_state.distant.sh_rest.detach(),
                hidden=local_state.distant.hidden.detach(),
            ),
        )

    def _phase_b_skip_distant_event(self) -> bool:
        return (
            str(getattr(self, "stage6_phase", "")) == PHASE_B_NAME
            and not bool(self.stage6_branch_scope.get("distant", {}).get("enable", True))
        )

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
            valid = self._stage6_rigid_point_valid_mask(rigid_node, int(frame_idx))
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
            detach_distant = (
                str(getattr(self, "stage6_phase", "")) == PHASE_B_NAME
                and not bool(self.stage6_branch_scope.get("distant", {}).get("enable", True))
            )
            parts.append(self._branch_render_params(local_state.distant, detach=detach_distant))
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
            use_grouped_multiview = bool(getattr(self, "stage6_render_grouped_multiview_train", True))
            if use_grouped_multiview and all(int(h) == h0 and int(w) == w0 for h, w in zip(heights, widths)):
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
        l1_weight: Optional[float] = None,
        ssim_weight: Optional[float] = None,
        pred_rgbs_out: Optional[List[torch.Tensor]] = None,
        gt_images_out: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if len(target_indices) == 0:
            return local_state.bg.means.new_tensor(0.0), {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
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
                l1_weight=1.0 if l1_weight is None else float(l1_weight),
                ssim_weight=float(getattr(self, "loss_w_ssim", 0.0)) if ssim_weight is None else float(ssim_weight),
            )
            if pred_rgbs_out is not None:
                pred_rgbs_out.append(pred.detach().float().clamp(0.0, 1.0).cpu())
            if gt_images_out is not None:
                gt_images_out.append(gt.detach().float().clamp(0.0, 1.0).cpu())
            losses.append(loss_i)
            if float(stat_i.get("skipped_no_valid_pixels", 0.0)) < 0.5:
                psnr_vals.append(float(stat_i["psnr"]))
                l1_vals.append(float(stat_i["l1"]))
                ssim_vals.append(float(stat_i.get("ssim", 0.0)))
            valid_ratios.append(float(stat_i.get("valid_ratio", 0.0)))
            skip_count += float(stat_i.get("skipped_no_valid_pixels", 0.0))
        if psnr_vals:
            stats["psnr"] = float(sum(psnr_vals) / len(psnr_vals))
            stats["l1"] = float(sum(l1_vals) / len(l1_vals))
            stats["ssim"] = float(sum(ssim_vals) / len(ssim_vals))
        stats["num_metric_refs"] = float(len(psnr_vals))
        stats["metric_valid"] = float(1.0 if psnr_vals else 0.0)
        stats["valid_ratio"] = float(sum(valid_ratios) / max(len(valid_ratios), 1))
        stats["skipped_no_valid_pixels"] = float(skip_count)
        return torch.stack(losses).mean(), stats

    @staticmethod
    def _as_int(x: Any, default: int = -1) -> int:
        if x is None:
            return int(default)
        if torch.is_tensor(x):
            return int(x.reshape(-1)[0].item()) if x.numel() else int(default)
        return int(x)

    def _phase_b_cache_key_from_batch(self, batch: Dict[str, Any]) -> Tuple[int, int, int, str]:
        meta = dict(batch.get("request_meta") or {})
        scene_id = self._as_int(meta.get("scene_id", batch.get("scene_id", -1)))
        segment_id = self._as_int(meta.get("segment_id", batch.get("segment_id", -1)))
        episode_id = self._as_int(meta.get("episode_id", meta.get("episode_idx_global", -1)))
        if scene_id < 0 or segment_id < 0 or episode_id < 0:
            raise ValueError("Stage6_0 Phase B requires scene_id, segment_id, and episode_id in batch/request_meta.")
        tbptt = dict(meta.get("tbptt") or {})
        stream_id = str(tbptt.get("stream_id", "default"))
        return int(scene_id), int(segment_id), int(episode_id), stream_id

    @staticmethod
    def _detach_local_branch(branch: Any) -> Any:
        if branch is None:
            return None
        return replace(
            branch,
            means=branch.means.detach().clone(),
            scales_log=branch.scales_log.detach().clone(),
            quats=branch.quats.detach().clone(),
            opacity_logit=branch.opacity_logit.detach().clone(),
            sh_dc=branch.sh_dc.detach().clone(),
            sh_rest=branch.sh_rest.detach().clone(),
            hidden=branch.hidden.detach().clone(),
        )

    def _detach_local_state(self, local_state: LocalGSState) -> LocalGSState:
        return LocalGSState(
            bg=self._detach_local_branch(local_state.bg),
            distant=self._detach_local_branch(local_state.distant),
            rigid=self._detach_local_branch(local_state.rigid),
            rigid_template=local_state.rigid_template.detach_clone() if local_state.rigid_template is not None else None,
        )

    def _phase_b_prior_written_refs(self, key: Tuple[int, int, int, str]) -> set[Tuple[int, int]]:
        if not bool(getattr(self, "stage6_phase_b_tbptt_enable", True)):
            return set()
        item = self.stage6_phase_b_tbptt_cache.get(tuple(key))
        if not item:
            return set()
        return set(item.get("written_refs") or set())

    def _phase_b_init_or_load_state(
        self,
        *,
        key: Tuple[int, int, int, str],
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
    ) -> tuple[LocalGSState, Stage6VSMState, set[Tuple[int, int]], bool]:
        if self.stage6_vsm is None:
            raise RuntimeError("Stage6_0 Phase B requires stage6_vsm.")
        use_cache = bool(getattr(self, "stage6_phase_b_tbptt_enable", True))
        cached = self.stage6_phase_b_tbptt_cache.get(tuple(key)) if use_cache else None
        if cached is not None:
            local_cached = self._detach_local_state(cached["local_G"])
            local_cached = self._phase_b_freeze_distant_branch(local_cached)
            vsm_cached = cached["vsm"].detach()
            cached_rigid_n = int(local_cached.rigid.means.shape[0]) if local_cached.rigid is not None else 0
            node_rigid_n = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
            if int(local_cached.bg.means.shape[0]) == int(node_state_bg.means.shape[0]) and cached_rigid_n == node_rigid_n:
                if local_cached.rigid is not None and node_state_rigid is not None:
                    # The cached local branch carries learned per-row state across TBPTT chunks,
                    # but the rigid template's frame slots must track the current batch dynamic_info.
                    local_cached.rigid_template = node_state_rigid.detach_clone()
                return local_cached, vsm_cached, set(cached.get("written_refs") or set()), True
            if bool(getattr(self, "stage6_phase_b_tbptt_strict", False)):
                raise ValueError("Stage6_0 Phase B strict TBPTT cache shape mismatch.")
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=self.stage6_hidden_dim,
        )
        local_state = self._phase_b_freeze_distant_branch(local_state)
        vsm_state = self.stage6_vsm.init_state(
            num_bg=int(local_state.bg.means.shape[0]),
            num_rigid=int(local_state.rigid.means.shape[0]) if local_state.rigid is not None else 0,
            device=local_state.bg.means.device,
            dtype=local_state.bg.means.dtype,
            episode_id=int(key[2]),
            written_refs=set(),
        )
        return local_state, vsm_state, set(), False

    def _phase_b_assert_vsm_state_matches_local(
        self,
        *,
        local_state: LocalGSState,
        vsm_state: Stage6VSMState,
        label: str,
    ) -> None:
        expected_bg = int(local_state.bg.means.shape[0])
        expected_rigid = int(local_state.rigid.means.shape[0]) if local_state.rigid is not None else 0
        checks = (
            ("tokens_bg", vsm_state.tokens_bg, expected_bg),
            ("proto_bg", vsm_state.proto_bg, expected_bg),
            ("global_bg", vsm_state.global_bg, expected_bg),
            ("valid_count_bg", vsm_state.valid_count_bg, expected_bg),
            ("tokens_rigid", vsm_state.tokens_rigid, expected_rigid),
            ("proto_rigid", vsm_state.proto_rigid, expected_rigid),
            ("global_rigid", vsm_state.global_rigid, expected_rigid),
            ("valid_count_rigid", vsm_state.valid_count_rigid, expected_rigid),
        )
        for name, tensor, expected in checks:
            actual = int(tensor.shape[0])
            if actual != int(expected):
                raise ValueError(
                    "Stage6_0 Phase B VSM/local row mismatch "
                    f"at {label}: {name} rows={actual} expected={int(expected)}"
                )

    def _phase_b_store_state(
        self,
        *,
        key: Tuple[int, int, int, str],
        local_state: LocalGSState,
        vsm_state: Stage6VSMState,
        written_refs: set[Tuple[int, int]],
        tbptt_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(getattr(self, "stage6_phase_b_tbptt_enable", True)):
            return
        self._phase_b_assert_vsm_state_matches_local(local_state=local_state, vsm_state=vsm_state, label="cache_store")
        max_items = int(getattr(self, "stage6_phase_b_tbptt_max_items", 8))
        if (
            bool(getattr(self, "stage6_phase_b_tbptt_strict", False))
            and bool(getattr(self, "stage6_phase_b_tbptt_forbid_cache_eviction", False))
            and tuple(key) not in self.stage6_phase_b_tbptt_cache
            and len(self.stage6_phase_b_tbptt_cache) >= max(max_items, 1)
        ):
            raise RuntimeError(
                "TBPTT cache full; eviction would break long sequence continuity: "
                f"max_items={max_items}, active={len(self.stage6_phase_b_tbptt_cache)}"
            )
        meta = dict(tbptt_meta or {})
        event_frames = [int(x) for x in list(meta.get("event_frame_indices", []) or [])]
        chunk_idx = int(meta.get("chunk_idx", -1)) if meta else -1
        self.stage6_phase_b_tbptt_cache[tuple(key)] = {
            "local_G": self._detach_local_state(local_state),
            "vsm": vsm_state.detach(),
            "written_refs": set(written_refs),
            "last_event_frame_idx": max(event_frames) if event_frames else -1,
            "next_chunk_idx": int(chunk_idx) + 1 if chunk_idx >= 0 else 0,
        }
        while len(self.stage6_phase_b_tbptt_cache) > max(max_items, 1):
            oldest = next(iter(self.stage6_phase_b_tbptt_cache.keys()))
            self.stage6_phase_b_tbptt_cache.pop(oldest, None)

    def _phase_b_clear_tbptt_cache(self) -> None:
        self.stage6_phase_b_tbptt_cache.clear()

    def _phase_b_clear_tbptt_cache_key(self, key: Tuple[int, int, int, str]) -> None:
        self.stage6_phase_b_tbptt_cache.pop(tuple(key), None)

    def _phase_b_long_store_v9_state(
        self,
        *,
        key: Tuple[int, int, int, str],
        base_state: LocalGSState,
        vsm_state: Any,
        offset_state: PhaseBOffsetState,
        written_refs: set[Tuple[int, int]],
        tbptt_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(getattr(self, "stage6_phase_b_tbptt_enable", True)):
            return
        max_items = int(getattr(self, "stage6_phase_b_tbptt_max_items", 8))
        if (
            bool(getattr(self, "stage6_phase_b_tbptt_strict", False))
            and bool(getattr(self, "stage6_phase_b_tbptt_forbid_cache_eviction", False))
            and tuple(key) not in self.stage6_phase_b_tbptt_cache
            and len(self.stage6_phase_b_tbptt_cache) >= max(max_items, 1)
        ):
            raise RuntimeError(
                "Phase B final rollout cache full; eviction would break episode continuity: "
                f"max_items={max_items}, active={len(self.stage6_phase_b_tbptt_cache)}"
            )
        meta = dict(tbptt_meta or {})
        event_frames = [int(x) for x in list(meta.get("event_frame_indices", []) or [])]
        chunk_idx = int(meta.get("chunk_idx", -1)) if meta else -1
        detach_vsm = getattr(vsm_state, "detach_to_cache_optional", None)
        self.stage6_phase_b_tbptt_cache[tuple(key)] = {
            "base_G": self._detach_local_state(base_state),
            "long_vsm_state": detach_vsm() if callable(detach_vsm) else vsm_state.detach(),
            "offset_state": offset_state.detach_for_sensor(),
            "written_refs": set(written_refs),
            "last_event_frame_idx": max(event_frames) if event_frames else -1,
            "next_chunk_idx": int(chunk_idx) + 1 if chunk_idx >= 0 else 0,
            "phase_b_v9_final_rollout_long": True,
        }
        while len(self.stage6_phase_b_tbptt_cache) > max(max_items, 1):
            oldest = next(iter(self.stage6_phase_b_tbptt_cache.keys()))
            self.stage6_phase_b_tbptt_cache.pop(oldest, None)

    @staticmethod
    def _phase_b_ref_set(raw_refs: Any) -> set[Tuple[int, int]]:
        out: set[Tuple[int, int]] = set()
        for ref in list(raw_refs or []):
            if isinstance(ref, (list, tuple)) and len(ref) == 2:
                out.add((int(ref[0]), int(ref[1])))
        return out

    def _phase_b_validate_strict_tbptt_start(
        self,
        *,
        key: Tuple[int, int, int, str],
        tbptt_meta: Dict[str, Any],
        query_label_refs: List[Tuple[int, int]],
        cache_hit: bool,
        cached_item: Optional[Dict[str, Any]],
    ) -> None:
        if not bool(getattr(self, "stage6_phase_b_tbptt_strict", False)):
            return
        if not bool(tbptt_meta.get("enable", False)):
            raise ValueError("Phase B strict TBPTT requires request_meta.tbptt.enable=true.")
        if not bool(tbptt_meta.get("strict", False)):
            raise ValueError("Phase B strict TBPTT requires request_meta.tbptt.strict=true.")
        chunk_idx = int(tbptt_meta.get("chunk_idx", -1))
        if chunk_idx < 0:
            raise ValueError("Phase B strict TBPTT requires non-negative tbptt.chunk_idx.")
        is_first = bool(tbptt_meta.get("is_first_chunk", False))
        if is_first and cache_hit:
            raise ValueError("first TBPTT chunk unexpectedly hit cache.")
        if not is_first and not cache_hit:
            raise ValueError("non-first TBPTT chunk requires cache hit.")
        event_frames = [int(x) for x in list(tbptt_meta.get("event_frame_indices", []) or [])]
        if not event_frames:
            raise ValueError("Phase B strict TBPTT requires tbptt.event_frame_indices.")
        if event_frames != sorted(event_frames) or len(set(event_frames)) != len(event_frames):
            raise ValueError("TBPTT event frames are not strictly chronological.")
        if cached_item is not None and cache_hit:
            expected_next = int(cached_item.get("next_chunk_idx", -1))
            if int(chunk_idx) != int(expected_next):
                raise ValueError(
                    f"TBPTT chunk_idx discontinuity: got {int(chunk_idx)} expected {int(expected_next)}"
                )
            last_event = int(cached_item.get("last_event_frame_idx", -1))
            if min(event_frames) <= int(last_event):
                raise ValueError(
                    "TBPTT event frames are not chronological across chunks: "
                    f"first={min(event_frames)} previous_last={int(last_event)}"
                )
        query_set = set(tuple(x) for x in query_label_refs)
        written_refs = self._phase_b_ref_set(tbptt_meta.get("prior_written_refs", []))
        if query_set & written_refs:
            raise ValueError("query_label_refs overlap scheduler TBPTT prior_written_refs.")
        prior_frames = {int(x) for x in list(tbptt_meta.get("prior_written_frames", []) or [])}
        query_frames = {int(ref[0]) for ref in query_set}
        if query_frames & prior_frames:
            raise ValueError("query_label_refs overlap scheduler TBPTT prior_written_frames.")
        cached_written_refs = self._phase_b_ref_set((cached_item or {}).get("written_refs", []))
        cached_written_frames = {int(ref[0]) for ref in cached_written_refs}
        if query_frames & cached_written_frames:
            raise ValueError("query_label_refs overlap cached TBPTT written frame indices.")

    def _phase_b_prefix_step_weight(self, *, k: int, K: int) -> float:
        if self.stage6_phase_b_prefix_step_weight == "late_heavy_linear":
            return 0.5 + 0.5 * float(int(k) + 1) / max(float(K), 1.0)
        if self.stage6_phase_b_prefix_step_weight in {"uniform", "none"}:
            return 1.0
        raise ValueError(f"unsupported Phase B prefix step_weight={self.stage6_phase_b_prefix_step_weight!r}")

    def _phase_b_query_weight(self, *, global_step: int) -> float:
        if not self.stage6_phase_b_query_enable:
            return 0.0
        warmup = min(float(int(global_step) + 1) / max(int(self.stage6_phase_b_query_warmup_steps), 1), 1.0)
        return float(self.stage6_phase_b_query_weight) * warmup

    def _observe_targets_as_stage6_event(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        targets: List[Dict[str, Any]],
    ) -> Any:
        if len(targets) == 0:
            raise ValueError("Phase B query observation requires non-empty query targets.")
        query_frames = {int(t.get("frame_idx", -1)) for t in targets}
        if len(query_frames) != 1:
            raise ValueError("Phase B P0 query observation supports exactly one query frame per rollout.")
        source_batch = dict(batch)
        source_batch["source_views"] = [t["view"] for t in targets]
        source_batch["source_images"] = [t["gt_image"] for t in targets]
        if any("sky_mask" not in t for t in targets):
            raise ValueError("Phase B query observation requires query target sky_mask.")
        if any("egocar_mask" not in t for t in targets):
            raise ValueError("Phase B query observation requires query target egocar_mask.")
        source_batch["source_sky_masks"] = [t["sky_mask"] for t in targets]
        source_batch["source_sky_mask"] = source_batch["source_sky_masks"]
        source_batch["source_egocar_masks"] = [t["egocar_mask"] for t in targets]
        source_batch["source_egocar_mask"] = source_batch["source_egocar_masks"]
        source_frame_idx = int(targets[0].get("frame_idx", 0))
        measurement = self._observe_v4_measurement(
            local_state=self._detach_local_state(local_state),
            batch=source_batch,
            source_indices=list(range(len(targets))),
            source_frame_idx=source_frame_idx,
        )
        event = self._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)
        return self._detach_event_pack(self._event_with_default_view_code(event))

    def _phase_b_rigid_route_indices(
        self,
        *,
        event: Any,
        local_state: LocalGSState,
        label: str,
    ) -> torch.Tensor:
        n_rigid = int(local_state.rigid.means.shape[0]) if local_state.rigid is not None else 0
        route = getattr(event, "route", None)
        S = getattr(route, "S", None) if route is not None else None
        device = event.event_bg.device
        if S is None:
            event_rigid = getattr(event, "event_rigid", None)
            if n_rigid > 0 and event_rigid is not None and int(event_rigid.shape[0]) > 0:
                raise ValueError(f"Stage6_0 Phase B-R requires event.route.S for {label} rigid events.")
            return torch.zeros((0,), dtype=torch.long, device=device)
        S = S.reshape(-1).to(device=device, dtype=torch.long)
        if int(S.numel()) == 0:
            event_rigid = getattr(event, "event_rigid", None)
            if event_rigid is not None and int(event_rigid.shape[0]) != 0:
                raise ValueError(f"Stage6_0 Phase B-R {label} rigid event/route.S shape mismatch.")
            return S
        if n_rigid <= 0:
            raise ValueError(f"Stage6_0 Phase B-R {label} has rigid route.S but local state has no rigid rows.")
        if int(S.unique().numel()) != int(S.numel()):
            raise ValueError(f"Stage6_0 Phase B-R {label} route.S contains duplicate rigid row indices.")
        if int(S.min().item()) < 0 or int(S.max().item()) >= n_rigid:
            raise ValueError(f"Stage6_0 Phase B-R {label} route.S contains out-of-range rigid row indices.")
        required = {
            "event_rigid": getattr(event, "event_rigid", None),
            "valid_rigid": getattr(event, "valid_rigid", None),
            "support_rigid": getattr(event, "support_rigid", None),
            "obs_code_rigid": getattr(event, "obs_code_rigid", None),
        }
        for name, value in required.items():
            if value is None:
                raise ValueError(f"Stage6_0 Phase B-R {label} requires {name} when route.S is non-empty.")
            if int(value.shape[0]) != int(S.numel()):
                raise ValueError(
                    f"Stage6_0 Phase B-R {label} {name}/route.S shape mismatch: "
                    f"{tuple(value.shape)} vs route={int(S.numel())}"
                )
        return S

    def _phase_b_branch_query_observation_loss(
        self,
        *,
        pred: Any,
        event_label: torch.Tensor,
        visible_label: torch.Tensor,
        support_label_raw: torch.Tensor,
        obs_label: torch.Tensor,
        branch_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        visible = visible_label.reshape(-1, 1).to(device=pred.event_hat.device, dtype=pred.event_hat.dtype)
        event_target = event_label.to(device=pred.event_hat.device, dtype=pred.event_hat.dtype)
        support_target = torch.log1p(
            support_label_raw.reshape(-1, 1).to(device=pred.event_hat.device, dtype=pred.event_hat.dtype).clamp_min(0.0)
        )
        obs_target = obs_label.to(device=pred.event_hat.device, dtype=pred.event_hat.dtype)
        event_loss = masked_smooth_l1(pred.event_hat, event_target, visible)
        visible_loss = F.binary_cross_entropy_with_logits(pred.visible_logit, visible)
        support_loss = masked_smooth_l1(pred.support_log_hat, support_target, visible)
        obs_loss = masked_smooth_l1(pred.obs_code_hat, obs_target, visible)
        total = (
            float(self.stage6_phase_b_query_event_weight) * event_loss
            + float(self.stage6_phase_b_query_visible_weight) * visible_loss
            + float(self.stage6_phase_b_query_support_weight) * support_loss
            + float(self.stage6_phase_b_query_obs_code_weight) * obs_loss
        )
        visible_pred = (torch.sigmoid(pred.visible_logit.detach()) > 0.5).to(dtype=torch.float32)
        visible_acc = (
            float((visible_pred == (visible.detach() > 0.5).to(dtype=torch.float32)).float().mean().item())
            if visible.numel()
            else 0.0
        )
        row_weight = visible.detach().sum().to(device=total.device, dtype=total.dtype)
        if int(visible.numel()) > 0:
            row_weight = row_weight.clamp_min(1.0)
        stats = {
            f"query_event_l1_{branch_name}": float(event_loss.detach().item()),
            f"query_visible_bce_{branch_name}": float(visible_loss.detach().item()),
            f"query_visible_acc_{branch_name}": visible_acc,
            f"query_support_l1_{branch_name}": float(support_loss.detach().item()),
            f"query_obs_code_l1_{branch_name}": float(obs_loss.detach().item()),
            f"query_rows_{branch_name}": float(row_weight.detach().item()),
        }
        return total, row_weight, stats

    def _phase_b_query_observation_loss(
        self,
        *,
        pred: Stage6QueryPred,
        label_event: Any,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        bg_loss, bg_weight, stats = self._phase_b_branch_query_observation_loss(
            pred=pred.bg,
            event_label=label_event.event_bg,
            visible_label=label_event.valid_bg,
            support_label_raw=label_event.support_bg,
            obs_label=label_event.obs_code_bg,
            branch_name="bg",
        )
        weighted_total = bg_loss * bg_weight
        total_weight = bg_weight

        if pred.rigid is not None and getattr(label_event, "event_rigid", None) is not None:
            event_rigid = label_event.event_rigid
            if int(event_rigid.shape[0]) > 0:
                rigid_loss, rigid_weight, rigid_stats = self._phase_b_branch_query_observation_loss(
                    pred=pred.rigid,
                    event_label=event_rigid,
                    visible_label=label_event.valid_rigid,
                    support_label_raw=label_event.support_rigid,
                    obs_label=label_event.obs_code_rigid,
                    branch_name="rigid",
                )
                weighted_total = weighted_total + rigid_loss * rigid_weight
                total_weight = total_weight + rigid_weight
                stats.update(rigid_stats)

        total = weighted_total / total_weight.clamp_min(1.0)
        event_all = (
            float(stats.get("query_event_l1_bg", 0.0)) * float(stats.get("query_rows_bg", 0.0))
            + float(stats.get("query_event_l1_rigid", 0.0)) * float(stats.get("query_rows_rigid", 0.0))
        ) / max(float(total_weight.detach().item()), 1.0)
        stats.update(
            {
                "query_event_l1": stats.get("query_event_l1_bg", 0.0),
                "query_visible_bce": stats.get("query_visible_bce_bg", 0.0),
                "query_visible_acc": stats.get("query_visible_acc_bg", 0.0),
                "query_support_l1": stats.get("query_support_l1_bg", 0.0),
                "query_obs_code_l1": stats.get("query_obs_code_l1_bg", 0.0),
                "query_event_l1_all": float(event_all),
                "query_rows_all": float(total_weight.detach().item()),
            }
        )
        return total, stats

    def _forward_phase_a(self, batch: Dict[str, Any]) -> Dict[str, Any]:
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
            item = {
                "k": float(k),
                "loss_block": float(block_loss.detach().item()),
                "loss_nearby": float(nearby_loss.detach().item()),
                "nearby_weight": float(near_weight),
                "block_valid_ratio": float(block_stats.get("valid_ratio", 0.0)),
                "nearby_valid_ratio": float(nearby_stats.get("valid_ratio", 0.0)),
                "block_skipped": float(block_stats.get("skipped_no_valid_pixels", 0.0)),
                "nearby_skipped": float(nearby_stats.get("skipped_no_valid_pixels", 0.0)),
                "block_metric_valid": float(block_stats.get("metric_valid", 0.0)),
                "nearby_metric_valid": float(nearby_stats.get("metric_valid", 0.0)),
                "block_num_metric_refs": float(block_stats.get("num_metric_refs", 0.0)),
                "nearby_num_metric_refs": float(nearby_stats.get("num_metric_refs", 0.0)),
                **{k2: float(v) for k2, v in reg_stats.items()},
                **{k2: float(v) for k2, v in update_aux.items() if isinstance(v, (int, float))},
            }
            for prefix, stats in (("block", block_stats), ("nearby", nearby_stats)):
                for metric_name in ("psnr", "ssim", "l1"):
                    value = stats.get(metric_name)
                    if value is None:
                        continue
                    value_f = float(value)
                    if math.isfinite(value_f):
                        item[f"{prefix}_{metric_name}"] = value_f
            per_step.append(item)
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

    def _forward_phase_b_v9_final_rollout_long(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        key = self._phase_b_cache_key_from_batch(batch)
        request_meta = dict(batch.get("request_meta") or {})
        tbptt_meta = dict(request_meta.get("tbptt") or {})
        prior_written_refs = self._phase_b_prior_written_refs(key) | self._phase_b_ref_set(
            tbptt_meta.get("prior_written_refs", [])
        )
        roles = resolve_v9_phase_b_batch(batch, written_refs=prior_written_refs)
        if int(roles.final_supervision_step_idx) != int(roles.inner_K) - 1:
            raise ValueError("V9 Phase B final rollout requires final supervision at inner_K - 1.")
        self._mem_debug("forward_phase_b_v9_final_long/begin", inner_K=int(roles.inner_K))
        if self.stage6_long_vsm is None:
            raise RuntimeError("V9 Phase B final rollout requires stage6_long_vsm.")
        if self.stage6_long_offset_decoder is None:
            raise RuntimeError("V9 Phase B final rollout requires stage6_long_offset_decoder.")
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("V9 Phase B final rollout requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("V9 Phase B final rollout requires non-empty final targets.")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        local_base = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=self.stage6_hidden_dim,
        )
        base_state = self._detach_local_state(local_base)
        cached = self.stage6_phase_b_tbptt_cache.get(tuple(key)) if self.stage6_phase_b_tbptt_enable else None
        written_refs = set(prior_written_refs)
        offset_dtype = self._phase_b_long_state_dtype(
            base_state.bg.means,
            str(getattr(self, "stage6_phase_b_long_offset_dtype", "bf16")),
        )
        vsm_dtype = self._phase_b_long_state_dtype(
            base_state.bg.means,
            str(getattr(self, "stage6_phase_b_long_vsm_dtype", "bf16")),
        )
        if cached is not None and bool(cached.get("phase_b_v9_final_rollout_long", False)):
            cached_base = self._detach_local_state(cached["base_G"])
            cached_rigid_n = int(cached_base.rigid.means.shape[0]) if cached_base.rigid is not None else 0
            node_rigid_n = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
            if int(cached_base.bg.means.shape[0]) == int(base_state.bg.means.shape[0]) and cached_rigid_n == node_rigid_n:
                if cached_base.rigid is not None and node_state_rigid is not None:
                    cached_base.rigid_template = node_state_rigid.detach_clone()
                base_state = cached_base
                detach_vsm = getattr(cached["long_vsm_state"], "detach_to_cache_optional", None)
                vsm_state = detach_vsm() if callable(detach_vsm) else cached["long_vsm_state"].detach()
                offset = cached["offset_state"].detach_for_sensor()
                written_refs = set(cached.get("written_refs") or set())
            elif bool(getattr(self, "stage6_phase_b_tbptt_strict", False)):
                raise ValueError("V9 Phase B final rollout cache shape mismatch.")
            else:
                cached = None
        if cached is None or not bool(cached.get("phase_b_v9_final_rollout_long", False)):
            offset = PhaseBOffsetState.zeros_like(base_state=base_state, dtype=offset_dtype)
            episode_id = int(request_meta.get("episode_id", request_meta.get("episode_idx_global", -1)) or -1)
            init_kwargs: Dict[str, Any] = {
                "base_state": base_state,
                "dtype": vsm_dtype,
                "rigid_meta": dict(request_meta.get("rigid_meta") or {}),
                "distant_mode": str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                "episode_id": episode_id,
            }
            if str(getattr(self, "stage6_phase_b_long_vsm_type", "streaming_selective_ssm")) == "cell_streaming_selective_ssm":
                init_kwargs["batch"] = batch
            vsm_state = self.stage6_long_vsm.init_state(**init_kwargs)

        rigid_meta = dict(request_meta.get("rigid_meta") or {})
        per_step: List[Dict[str, float]] = []
        for k in range(int(roles.inner_K)):
            evidence_refs = roles.evidence_refs_by_step[int(k)]
            frame_idx = int(roles.step_source_frame_indices[int(k)])
            repeat_idx = int(roles.step_repeat_indices[int(k)])
            memory_write = bool(roles.memory_write_flags_by_step[int(k)])
            with torch.no_grad():
                sensor_state = materialize_phase_b_state(
                    base_state=base_state,
                    offset=offset.detach_for_sensor(),
                    target_frame_idx=int(frame_idx),
                    rigid_meta=rigid_meta,
                )
                sensor_state = self._phase_b_long_clamp_sensor_state_to_aabb(sensor_state)
                measurement = self._observe_v4_measurement(
                    local_state=sensor_state,
                    batch=batch,
                    source_indices=roles.evidence_source_indices_by_step[int(k)],
                    source_frame_idx=int(frame_idx),
                )
                event = self._build_stage6_event_from_measurement(local_state=sensor_state, measurement=measurement)
                event = self._detach_event_pack(self._event_with_default_view_code(event))
            if not torch.isfinite(event.event_bg).all():
                raise RuntimeError("V9 Phase B final rollout event_bg contains NaN/Inf.")
            self._phase_b_rigid_route_indices(event=event, local_state=base_state, label=f"v9 final step {int(k)}")
            vsm_compute_dtype = self._phase_b_long_vsm_compute_dtype(event.event_bg)
            time_code = (roles.visit_time_codes or [(0.0, 0.0, 0.0, 0.0) for _ in range(int(roles.inner_K))])[int(k)]
            with self._phase_b_long_autocast_context(event.event_bg):
                vsm_state, read_pack, vsm_aux = self.stage6_long_vsm.write_read(
                    state=vsm_state,
                    event=event,
                    step_idx=int(k),
                    frame_idx=int(frame_idx),
                    repeat_idx=int(repeat_idx),
                    rigid_meta=rigid_meta,
                    distant_mode=str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                    visit_time_code=torch.tensor(
                        time_code,
                        device=event.event_bg.device,
                        dtype=vsm_compute_dtype or event.event_bg.dtype,
                    ),
                    compute_dtype=vsm_compute_dtype,
                    commit_memory=bool(memory_write),
                )
                delta = self.stage6_long_offset_decoder(
                    read=read_pack,
                    distant_mode=str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                )
            offset = offset.apply(delta, frame_idx=int(frame_idx), rigid_meta=rigid_meta)
            if memory_write:
                written_refs.update(set(evidence_refs))
            per_step.append(
                {
                    "k": float(k),
                    "frame_idx": float(frame_idx),
                    "repeat_idx": float(repeat_idx),
                    "block_idx": float(roles.step_block_indices[int(k)]),
                    "memory_write": float(1.0 if memory_write else 0.0),
                    "evidence_ref_count": float(len(evidence_refs)),
                    "final_ref_count": float(len(roles.prefix_loss_refs_by_step[int(k)])),
                    **{key2: float(value) for key2, value in vsm_aux.items()},
                    **{key2: float(value) for key2, value in delta.stats(prefix=f"k{int(k)}").items()},
                }
            )
            self._mem_debug("forward_phase_b_v9_final_long/after_step", k=int(k), memory_write=bool(memory_write))

        by_role = roles.final_target_indices_by_role or {}
        role_specs = [
            ("history_recon", by_role.get("final_history_recon", [])),
            ("history_nvs", by_role.get("final_history_nvs", [])),
            ("current_recon", by_role.get("final_current_recon", [])),
            ("current_nvs", by_role.get("final_current_nvs", [])),
        ]
        role_losses: Dict[str, torch.Tensor] = {}
        role_stats: Dict[str, float] = {}
        role_total = base_state.bg.means.new_tensor(0.0)
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        for role_name, target_indices in role_specs:
            cfg_role = dict(getattr(self, "stage6_phase_b_long_role_render_cfg", {}).get(role_name, {}) or {})
            default_is_history = str(role_name).startswith("history")
            default_weight = self.stage6_phase_b_long_history_weight if default_is_history else self.stage6_phase_b_long_current_weight
            default_l1 = self.stage6_phase_b_long_history_l1_weight if default_is_history else self.stage6_phase_b_long_current_l1_weight
            default_ssim = self.stage6_phase_b_long_history_ssim_weight if default_is_history else self.stage6_phase_b_long_current_ssim_weight
            default_mask = self.stage6_phase_b_long_history_mask_policy if default_is_history else self.stage6_phase_b_long_current_mask_policy
            loss_i, stats_i = phase_b_long_final_render_loss(
                self,
                base_state=base_state,
                offset=offset,
                batch=batch,
                target_indices=target_indices,
                role=role_name,
                rigid_meta=rigid_meta,
                mask_policy=str(self._cfg_get(cfg_role, "mask_policy", default_mask)),
                l1_weight=float(self._cfg_get(cfg_role, "l1_weight", default_l1)),
                ssim_weight=float(self._cfg_get(cfg_role, "ssim_weight", default_ssim)),
                pred_rgbs_out=pred_rgbs,
                gt_images_out=gt_images,
            )
            weight_i = float(self._cfg_get(cfg_role, "weight", default_weight))
            role_losses[str(role_name)] = loss_i
            role_total = role_total + weight_i * loss_i
            role_stats.update({key2: float(value) for key2, value in stats_i.items()})
            role_stats[f"phase_b_long/final_{role_name}_weight"] = float(weight_i)
        zero = base_state.bg.means.new_tensor(0.0)
        history_loss = 0.5 * (role_losses.get("history_recon", zero) + role_losses.get("history_nvs", zero))
        current_loss = 0.5 * (role_losses.get("current_recon", zero) + role_losses.get("current_nvs", zero))
        reg_loss, reg_stats = phase_b_long_offset_regularization(
            offset,
            weights=dict(getattr(self, "stage6_phase_b_long_offset_reg_cfg", {}) or {}),
        )
        total_loss = role_total + float(self.stage6_phase_b_long_offset_reg_weight) * reg_loss
        if not torch.isfinite(total_loss).all():
            raise RuntimeError("V9 Phase B final rollout loss became NaN/Inf.")
        rollout_meta = dict(roles.phase_b_rollout or {})
        final_meta = dict(roles.final_supervision or {})
        final_roles = [str(x) for x in list(final_meta.get("roles", []) or [])]
        current_recon_matches = bool(final_meta.get("current_recon_matches_trained_frames", False))
        if not current_recon_matches:
            raise RuntimeError("Phase B current supervision does not match trained frames.")
        stats = {
            **role_stats,
            **reg_stats,
            "phase_b_v9/final_supervision_step_idx": float(roles.final_supervision_step_idx),
            "phase_b_v9/final_supervision_ref_count": float(len(roles.final_target_indices or [])),
            "phase_b_v9/intermediate_loss_ref_count": float(
                sum(len(x) for x in roles.prefix_loss_refs_by_step[:-1])
            ),
            "phase_b_v9/nvs_evidence_overlap_count": float(
                (roles.final_supervision or {}).get("nvs_evidence_overlap_count", 0.0)
            ),
            "phase_b_long/final_history_weight": float(self.stage6_phase_b_long_history_weight),
            "phase_b_long/final_current_weight": float(self.stage6_phase_b_long_current_weight),
            "phase_b_long/offset_reg_weight": float(self.stage6_phase_b_long_offset_reg_weight),
            "phase_b_long/shape/" + str(request_meta.get("shape_name", "unknown")): 1.0,
            "phase_b_long/effective_shape/" + str(rollout_meta.get("effective_shape_name", request_meta.get("shape_name", "unknown"))): 1.0,
            "phase_b_v9/requested_blocks_per_rollout": float(rollout_meta.get("requested_blocks_per_rollout", request_meta.get("blocks_per_rollout", 0))),
            "phase_b_v9/actual_blocks_per_rollout": float(rollout_meta.get("actual_blocks_per_rollout", 0)),
            "phase_b_v9/repeats_per_block": float(rollout_meta.get("repeats_per_block", 0)),
            "phase_b_v9/requested_inner_K": float(rollout_meta.get("requested_inner_K", 0)),
            "phase_b_v9/actual_inner_K": float(rollout_meta.get("actual_inner_K", roles.inner_K)),
            "phase_b_v9/short_rollout": float(bool(rollout_meta.get("short_rollout", False))),
            "phase_b_v9/trained_current_frame_count": float(len(final_meta.get("trained_current_frames", []) or [])),
            "phase_b_v9/supervised_current_frame_count": float(len(final_meta.get("current_recon_frames", []) or [])),
            "phase_b_v9/current_recon_matches_trained_frames": float(current_recon_matches),
            "phase_b_v9/final_current_recon_frame_count": float(len(final_meta.get("current_recon_frames", []) or [])),
            "phase_b_v9/final_current_recon_ref_count": float(
                sum(1 for role in final_roles if str(role) == "final_current_recon")
            ),
            "phase_b_v9/expected_final_current_recon_ref_count": float(
                final_meta.get("expected_current_recon_ref_count", 0)
            ),
            "phase_b_v9/final_history_recon_frame_count": float(len(final_meta.get("history_recon_frames", []) or [])),
            "phase_b_v9/final_current_nvs_frame_count": float(len(final_meta.get("current_nvs_frames", []) or [])),
        }
        return {
            "loss": total_loss,
            "base_G": base_state,
            "local_G": base_state,
            "offset_state": offset,
            "vsm_state": vsm_state.detach_to_cache_optional(),
            "node_state_bg": node_state_bg,
            "node_state_distant": node_state_distant,
            "node_state_rigid": node_state_rigid,
            "roles": roles,
            "per_step": per_step,
            "stats": stats,
            "history_loss": history_loss.detach(),
            "current_loss": current_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "num_query_targets": len(batch.get("query_targets", [])),
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "written_refs": set(written_refs),
            "tbptt_key": key,
            "tbptt_meta": tbptt_meta,
            "tbptt_cache_hit": bool(cached is not None),
            "phase_b_v9_final_rollout_long": True,
        }

    def _forward_phase_b(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        request_meta = dict(batch.get("request_meta") or {})
        if str(request_meta.get("phase_b_loss_timing", "")) == "rollout_final_only":
            return self._forward_phase_b_v9_final_rollout_long(batch)
        key = self._phase_b_cache_key_from_batch(batch)
        request_meta = dict(batch.get("request_meta") or {})
        tbptt_meta = dict(request_meta.get("tbptt") or {})
        prior_written_refs = self._phase_b_prior_written_refs(key) | self._phase_b_ref_set(
            tbptt_meta.get("prior_written_refs", [])
        )
        roles = resolve_v9_phase_b_batch(batch, written_refs=prior_written_refs)
        self._mem_debug("forward_phase_b/begin", inner_K=int(roles.inner_K), cache_written=len(prior_written_refs))
        if self.stage6_vsm is None:
            raise RuntimeError("Stage6_0 Phase B requires stage6_vsm.")
        if self.stage6_query_decoder is None:
            raise RuntimeError("Stage6_0 Phase B requires stage6_query_decoder.")
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("Stage6_0 Phase B requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("Stage6_0 Phase B requires non-empty prefix targets.")
        if len(batch.get("query_targets", [])) == 0:
            raise ValueError("Stage6_0 Phase B requires non-empty query_targets.")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        cached_item = self.stage6_phase_b_tbptt_cache.get(tuple(key)) if self.stage6_phase_b_tbptt_enable else None
        local_state, vsm_state, written_refs, cache_hit = self._phase_b_init_or_load_state(
            key=key,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
        )
        self._phase_b_assert_vsm_state_matches_local(local_state=local_state, vsm_state=vsm_state, label="init_or_load")
        self._phase_b_validate_strict_tbptt_start(
            key=key,
            tbptt_meta=tbptt_meta,
            query_label_refs=roles.query_label_refs,
            cache_hit=bool(cache_hit),
            cached_item=cached_item,
        )
        if set(roles.query_label_refs) & set(written_refs):
            raise ValueError("query_label_refs already written into persistent VSM in this episode.")
        total_loss = local_state.bg.means.new_tensor(0.0)
        rollout_loss = local_state.bg.means.new_tensor(0.0)
        step_weight_sum = 0.0
        per_step: List[Dict[str, float]] = []
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        latest_vsm_update_aux: Dict[str, float] = {}
        step = int(batch.get("global_step", 0) or 0)
        step_repeat_indices = [int(x) for x in list(getattr(roles, "step_repeat_indices", []) or [])]
        step_block_indices = [int(x) for x in list(getattr(roles, "step_block_indices", []) or [])]
        if len(step_repeat_indices) != int(roles.inner_K):
            step_repeat_indices = [0 for _ in range(int(roles.inner_K))]
        if len(step_block_indices) != int(roles.inner_K):
            step_block_indices = [-1 for _ in range(int(roles.inner_K))]

        for k in range(int(roles.inner_K)):
            evidence_refs = roles.evidence_refs_by_step[int(k)]
            memory_write = bool(roles.memory_write_flags_by_step[int(k)])
            source_frame_idx = int(evidence_refs[0][0])
            with torch.no_grad():
                measurement = self._observe_v4_measurement(
                    local_state=local_state,
                    batch=batch,
                    source_indices=roles.evidence_source_indices_by_step[int(k)],
                    source_frame_idx=source_frame_idx,
                )
                event = self._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)
                event = self._detach_event_pack(self._event_with_default_view_code(event))

            if not torch.isfinite(event.event_bg).all():
                raise RuntimeError("Stage6_0 Phase B event_bg contains NaN/Inf.")
            rigid_indices = self._phase_b_rigid_route_indices(event=event, local_state=local_state, label=f"step {int(k)}")
            vsm_update_aux_bg: Dict[str, float] = {}
            if memory_write:
                vsm_state, vsm_update_aux_bg = self.stage6_vsm.update_bg(
                    state=vsm_state,
                    event_bg=event.event_bg,
                    view_code_bg=getattr(event, "view_code_bg", None),
                    valid_bg=getattr(event, "valid_bg", None),
                    support_bg=getattr(event, "support_bg", None),
                    return_aux=True,
                )
            ctx_bg, vsm_aux_bg = self.stage6_vsm.query_bg(
                state=vsm_state,
                view_code_bg=getattr(event, "view_code_bg", None),
            )
            ctx_rigid = None
            vsm_aux_rigid: Dict[str, float] = {}
            vsm_update_aux_rigid: Dict[str, float] = {}
            if int(rigid_indices.numel()) > 0:
                if memory_write:
                    vsm_state, vsm_update_aux_rigid = self.stage6_vsm.update_rigid(
                        state=vsm_state,
                        indices=rigid_indices,
                        event_rigid=event.event_rigid,
                        view_code_rigid=getattr(event, "view_code_rigid", getattr(event, "obs_code_rigid", None)),
                        valid_rigid=getattr(event, "valid_rigid", None),
                        support_rigid=getattr(event, "support_rigid", None),
                        return_aux=True,
                    )
                ctx_rigid, vsm_aux_rigid = self.stage6_vsm.query_rigid(
                    state=vsm_state,
                    indices=rigid_indices,
                    view_code_rigid=getattr(event, "view_code_rigid", getattr(event, "obs_code_rigid", None)),
                )
            if int(ctx_bg.shape[0]) != int(event.event_bg.shape[0]) or int(ctx_bg.shape[1]) != int(self.stage6_event_dim):
                raise ValueError(
                    "Stage6_0 Phase B ctx_bg shape mismatch: "
                    f"ctx={tuple(ctx_bg.shape)} event={tuple(event.event_bg.shape)}"
                )
            if ctx_rigid is not None and (
                int(ctx_rigid.shape[0]) != int(rigid_indices.numel()) or int(ctx_rigid.shape[1]) != int(self.stage6_event_dim)
            ):
                raise ValueError(
                    "Stage6_0 Phase B-R ctx_rigid shape mismatch: "
                    f"ctx={tuple(ctx_rigid.shape)} route={int(rigid_indices.numel())}"
                )
            if memory_write:
                latest_vsm_update_aux = {**vsm_update_aux_bg, **vsm_update_aux_rigid}
            vsm_aux = {**latest_vsm_update_aux, **vsm_aux_bg, **vsm_aux_rigid}
            self._mem_debug(
                "forward_phase_b/after_vsm",
                k=int(k),
                memory_write=bool(memory_write),
                rigid_rows=int(rigid_indices.numel()),
                bg_rows=int(event.event_bg.shape[0]),
            )
            ctx_vsm = ContextPack(ctx_bg=ctx_bg, ctx_distant=None, ctx_rigid=ctx_rigid, aux=vsm_aux)
            local_state, delta, update_aux = self._apply_event_update(
                local_state=local_state,
                event=event,
                ctx_vsm=ctx_vsm,
            )
            self._mem_debug("forward_phase_b/after_update", k=int(k))

            prefix_loss = local_state.bg.means.new_tensor(0.0)
            prefix_stats: Dict[str, float] = {}
            if self.stage6_phase_b_prefix_enable:
                prefix_loss, prefix_stats = self._render_loss_for_indices(
                    local_state=local_state,
                    batch=batch,
                    target_indices=roles.prefix_target_indices_by_step[int(k)],
                    mask_policy=self.stage6_phase_b_prefix_mask_policy,
                    pred_rgbs_out=pred_rgbs if int(k) == int(roles.inner_K) - 1 else None,
                    gt_images_out=gt_images if int(k) == int(roles.inner_K) - 1 else None,
                    l1_weight=self.stage6_phase_b_prefix_l1_weight,
                    ssim_weight=self.stage6_phase_b_prefix_ssim_weight,
                )
            self._mem_debug(
                "forward_phase_b/after_prefix_loss",
                k=int(k),
                prefix_refs=int(len(roles.prefix_loss_refs_by_step[int(k)])),
            )
            reg_loss, reg_stats = delta_regularization(
                delta,
                weight=float(self.stage6_phase_b_delta_norm_weight),
                local_state=local_state,
                opacity_delta_l2_weight=0.0,
                sh_delta_l2_weight=0.0,
                scale_barrier_weight=0.0,
                scale_log_min=self.stage6_scale_log_min,
                scale_log_max=self.stage6_scale_log_max,
            )
            step_weight = self._phase_b_prefix_step_weight(k=int(k), K=int(roles.inner_K))
            loss_k = step_weight * (float(self.stage6_phase_b_prefix_weight) * prefix_loss + reg_loss)
            if not torch.isfinite(loss_k).all():
                raise RuntimeError("Stage6_0 Phase B prefix loss became NaN/Inf.")
            total_loss = total_loss + loss_k
            rollout_loss = rollout_loss + loss_k
            step_weight_sum += float(step_weight)
            if memory_write:
                written_refs.update(set(evidence_refs))
            vsm_state = replace(vsm_state, written_refs=set(written_refs))
            per_step.append(
                {
                    "k": float(k),
                    "memory_write": float(1.0 if memory_write else 0.0),
                    "repeat_idx": float(step_repeat_indices[int(k)]),
                    "block_idx": float(step_block_indices[int(k)]),
                    "loss_prefix": float(prefix_loss.detach().item()),
                    "loss_reg": float(reg_loss.detach().item()),
                    "step_weight": float(step_weight),
                    "prefix_psnr": float(prefix_stats.get("psnr", 0.0)),
                    "prefix_l1": float(prefix_stats.get("l1", 0.0)),
                    "prefix_ssim": float(prefix_stats.get("ssim", 0.0)),
                    "prefix_valid_ratio": float(prefix_stats.get("valid_ratio", 0.0)),
                    "prefix_skipped": float(prefix_stats.get("skipped_no_valid_pixels", 0.0)),
                    "evidence_ref_count": float(len(evidence_refs)),
                    "prefix_ref_count": float(len(roles.prefix_loss_refs_by_step[int(k)])),
                    "delta_bg_means_norm": float(delta.bg.means.detach().norm(dim=-1).mean().item()) if delta.bg.means.numel() else 0.0,
                    "delta_bg_opacity_norm": float(delta.bg.opacity_logit.detach().abs().mean().item()) if delta.bg.opacity_logit.numel() else 0.0,
                    "rigid_seen_ratio": float(int(rigid_indices.numel()) / max(int(vsm_state.tokens_rigid.shape[0]), 1)),
                    "rigid_delta_means_norm": (
                        float(delta.rigid.means.detach().norm(dim=-1).mean().item())
                        if delta.rigid is not None and delta.rigid.means.numel()
                        else 0.0
                    ),
                    "rigid_delta_opacity_norm": (
                        float(delta.rigid.opacity_logit.detach().abs().mean().item())
                        if delta.rigid is not None and delta.rigid.opacity_logit.numel()
                        else 0.0
                    ),
                    "rigid_noop_mean": (
                        float(delta.rigid.noop.detach().mean().item())
                        if delta.rigid is not None and delta.rigid.noop.numel()
                        else 0.0
                    ),
                    "confidence_mean": float(delta.bg.confidence.detach().mean().item()) if delta.bg.confidence.numel() else 0.0,
                    "noop_mean": float(delta.bg.noop.detach().mean().item()) if delta.bg.noop.numel() else 0.0,
                    **{k2: float(v) for k2, v in reg_stats.items()},
                    **{k2: float(v) for k2, v in vsm_aux.items() if isinstance(v, (int, float))},
                    **{k2: float(v) for k2, v in update_aux.items() if isinstance(v, (int, float))},
                }
            )

        query_weight = self._phase_b_query_weight(global_step=step)
        query_stats: Dict[str, float] = {}
        query_loss = total_loss.new_tensor(0.0)
        if self.stage6_phase_b_query_enable and query_weight > 0.0:
            query_targets_all = list(batch.get("query_targets") or [])
            query_targets = [query_targets_all[int(i)] for i in roles.query_target_indices]
            with torch.no_grad():
                label_event = self._observe_targets_as_stage6_event(
                    local_state=local_state,
                    batch=batch,
                    targets=query_targets,
                )
            query_rigid_indices = self._phase_b_rigid_route_indices(
                event=label_event,
                local_state=local_state,
                label="query",
            )
            query_pred = self.stage6_query_decoder(
                state=vsm_state,
                query_view_code_bg=getattr(label_event, "view_code_bg", None),
                query_view_code_rigid=getattr(label_event, "view_code_rigid", getattr(label_event, "obs_code_rigid", None)),
                rigid_indices=query_rigid_indices,
                memory=self.stage6_vsm,
            )
            query_loss, query_stats = self._phase_b_query_observation_loss(pred=query_pred, label_event=label_event)
            total_loss = total_loss + float(query_weight) * query_loss
            self._mem_debug("forward_phase_b/after_query_loss", query_refs=int(len(roles.query_label_refs)))

        if not torch.isfinite(total_loss).all():
            raise RuntimeError("Stage6_0 Phase B total loss became NaN/Inf.")
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
            "num_query_targets": len(batch.get("query_targets", [])),
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "vsm_state": vsm_state,
            "written_refs": set(written_refs),
            "tbptt_key": key,
            "tbptt_meta": tbptt_meta,
            "tbptt_cache_hit": bool(cache_hit),
            "query_weight": float(query_weight),
            "query_loss": float(query_loss.detach().item()) if torch.is_tensor(query_loss) else float(query_loss),
            "query_stats": query_stats,
            "rollout_loss": float(rollout_loss.detach().item()),
            "loss_total_norm_by_weight": (
                float((rollout_loss / max(float(step_weight_sum), 1.0e-8)).detach().item())
                if torch.is_tensor(rollout_loss)
                else 0.0
            ),
            "loss_total_norm_by_K": (
                float((rollout_loss / max(int(roles.inner_K), 1)).detach().item())
                if torch.is_tensor(rollout_loss)
                else 0.0
            ),
            "step_weight_sum": float(step_weight_sum),
            "leak/query_evidence_overlap": float(len(set(roles.query_label_refs) & set(_ref for group in roles.evidence_refs_by_step for _ref in group))),
            "leak/query_written_overlap": float(len(set(roles.query_label_refs) & set(written_refs))),
        }

    def _phase_b_long_state_dtype(self, ref: torch.Tensor, dtype_name: str) -> torch.dtype:
        name = str(dtype_name).lower()
        if name in {"bf16", "bfloat16"} and ref.is_cuda:
            return torch.bfloat16
        if name in {"fp16", "float16"}:
            return torch.float16
        return ref.dtype

    @staticmethod
    def _phase_b_long_autocast_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
        name = str(dtype_name).strip().lower()
        if name in {"", "none", "off", "false", "fp32", "float32"}:
            return None
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if name in {"fp16", "float16"}:
            return torch.float16
        raise ValueError(f"unsupported phase_b_long amp_dtype={dtype_name!r}")

    def _phase_b_long_vsm_compute_dtype(self, ref: torch.Tensor) -> Optional[torch.dtype]:
        dtype = self._phase_b_long_autocast_torch_dtype(
            str(getattr(self, "stage6_phase_b_long_amp_dtype", "bf16"))
        )
        if dtype is None or not ref.is_cuda:
            return None
        return dtype

    def _phase_b_long_autocast_context(self, ref: torch.Tensor) -> Any:
        dtype = self._phase_b_long_vsm_compute_dtype(ref)
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)

    def _phase_b_long_clamp_sensor_state_to_aabb(self, state: LocalGSState) -> LocalGSState:
        """Keep offset-materialized sensor rows compatible with the frozen xCPE grid."""

        def clamp_branch(branch: Optional[Any]) -> Optional[Any]:
            if branch is None:
                return None
            lo, hi = self._stage6_aabb(branch.means)
            return replace(branch, means=branch.means.clamp(min=lo, max=hi))

        return LocalGSState(
            bg=clamp_branch(state.bg),
            distant=state.distant,
            rigid=clamp_branch(state.rigid),
            rigid_template=state.rigid_template,
        )

    def _forward_6_0_phase_b_long(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from models.streetforward.stage6_0.phase_b_long.resolver import resolve_long_phase_b_batch

        roles = resolve_long_phase_b_batch(batch)
        self._mem_debug("forward_phase_b_long/begin", inner_K=int(roles.inner_K))
        if self.stage6_long_vsm is None:
            raise RuntimeError("6_0_phase_b requires stage6_long_vsm.")
        if self.stage6_long_offset_decoder is None:
            raise RuntimeError("6_0_phase_b requires stage6_long_offset_decoder.")
        if len(batch.get("source_views", [])) == 0:
            raise ValueError("6_0_phase_b requires non-empty source_views.")
        if len(batch.get("targets", [])) == 0:
            raise ValueError("6_0_phase_b requires non-empty final targets.")
        collect_images = bool(
            batch.get("_stage5_6_collect_debug_images", False)
            or batch.get("_collect_train_images", False)
        )
        max_collect_images = int(batch.get("_max_collect_train_images", 8) or 8)
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        local_base = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=self.stage6_hidden_dim,
        )
        base_state = self._detach_local_state(local_base)
        offset_dtype = self._phase_b_long_state_dtype(
            base_state.bg.means,
            str(getattr(self, "stage6_phase_b_long_offset_dtype", "bf16")),
        )
        vsm_dtype = self._phase_b_long_state_dtype(
            base_state.bg.means,
            str(getattr(self, "stage6_phase_b_long_vsm_dtype", "bf16")),
        )
        offset = PhaseBOffsetState.zeros_like(base_state=base_state, dtype=offset_dtype)
        episode_id = int(roles.request_meta.get("episode_id", roles.request_meta.get("episode_idx_global", -1)) or -1)
        init_kwargs: Dict[str, Any] = {
            "base_state": base_state,
            "dtype": vsm_dtype,
            "rigid_meta": roles.rigid_meta,
            "distant_mode": str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
            "episode_id": episode_id,
        }
        if str(getattr(self, "stage6_phase_b_long_vsm_type", "streaming_selective_ssm")) == "cell_streaming_selective_ssm":
            init_kwargs["batch"] = batch
        vsm_state = self.stage6_long_vsm.init_state(**init_kwargs)
        per_step: List[Dict[str, float]] = []
        for k in range(int(roles.inner_K)):
            visit = roles.visits[int(k)]
            frame_idx = int(visit.frame_idx)
            repeat_idx = int(visit.repeat_idx)
            with torch.no_grad():
                sensor_state = materialize_phase_b_state(
                    base_state=base_state,
                    offset=offset.detach_for_sensor(),
                    target_frame_idx=int(frame_idx),
                    rigid_meta=roles.rigid_meta,
                )
                sensor_state = self._phase_b_long_clamp_sensor_state_to_aabb(sensor_state)
                measurement = self._observe_v4_measurement(
                    local_state=sensor_state,
                    batch=batch,
                    source_indices=roles.evidence_source_indices_by_step[int(k)],
                    source_frame_idx=int(frame_idx),
                )
                event = self._build_stage6_event_from_measurement(local_state=sensor_state, measurement=measurement)
                event = self._detach_event_pack(self._event_with_default_view_code(event))
            if not torch.isfinite(event.event_bg).all():
                raise RuntimeError("6_0_phase_b event_bg contains NaN/Inf.")
            self._phase_b_rigid_route_indices(event=event, local_state=base_state, label=f"long step {int(k)}")
            vsm_compute_dtype = self._phase_b_long_vsm_compute_dtype(event.event_bg)
            with self._phase_b_long_autocast_context(event.event_bg):
                vsm_state, read_pack, vsm_aux = self.stage6_long_vsm.write_read(
                    state=vsm_state,
                    event=event,
                    step_idx=int(k),
                    frame_idx=int(frame_idx),
                    repeat_idx=int(repeat_idx),
                    rigid_meta=roles.rigid_meta,
                    distant_mode=str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                    visit_time_code=torch.tensor(
                        roles.visit_time_codes[int(k)],
                        device=event.event_bg.device,
                        dtype=vsm_compute_dtype or event.event_bg.dtype,
                    ),
                    compute_dtype=vsm_compute_dtype,
                )
                delta = self.stage6_long_offset_decoder(
                    read=read_pack,
                    distant_mode=str(getattr(self, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                )
            offset = offset.apply(delta, frame_idx=int(frame_idx), rigid_meta=roles.rigid_meta)
            step_stats = {
                "k": float(k),
                "frame_idx": float(frame_idx),
                "repeat_idx": float(repeat_idx),
                "anchor_id": float(visit.anchor_id),
                "rollout_order_rank": float(visit.rollout_order_rank),
                "chronological_rank": float(visit.chronological_rank),
                "visit_pos_code": float(visit.visit_pos_code),
                "frame_time_code": float(visit.frame_time_code),
                "evidence_ref_count": float(len(roles.evidence_refs_by_step[int(k)])),
                **{key: float(value) for key, value in vsm_aux.items()},
                **{key: float(value) for key, value in delta.stats(prefix=f"k{int(k)}").items()},
            }
            per_step.append(step_stats)
            self._mem_debug("forward_phase_b_long/after_step", k=int(k))

        role_specs = [
            ("history_recon", roles.final_history_recon_target_indices),
            ("history_nvs", roles.final_history_nvs_target_indices),
            ("current_recon", roles.final_current_recon_target_indices),
            ("current_nvs", roles.final_current_nvs_target_indices),
        ]
        role_losses: Dict[str, torch.Tensor] = {}
        role_stats: Dict[str, float] = {}
        role_total = base_state.bg.means.new_tensor(0.0)
        for role_name, target_indices in role_specs:
            cfg_role = dict(getattr(self, "stage6_phase_b_long_role_render_cfg", {}).get(role_name, {}) or {})
            default_is_history = str(role_name).startswith("history")
            default_weight = self.stage6_phase_b_long_history_weight if default_is_history else self.stage6_phase_b_long_current_weight
            default_l1 = self.stage6_phase_b_long_history_l1_weight if default_is_history else self.stage6_phase_b_long_current_l1_weight
            default_ssim = self.stage6_phase_b_long_history_ssim_weight if default_is_history else self.stage6_phase_b_long_current_ssim_weight
            default_mask = self.stage6_phase_b_long_history_mask_policy if default_is_history else self.stage6_phase_b_long_current_mask_policy
            loss_i, stats_i = phase_b_long_final_render_loss(
                self,
                base_state=base_state,
                offset=offset,
                batch=batch,
                target_indices=target_indices,
                role=role_name,
                rigid_meta=roles.rigid_meta,
                mask_policy=str(self._cfg_get(cfg_role, "mask_policy", default_mask)),
                l1_weight=float(self._cfg_get(cfg_role, "l1_weight", default_l1)),
                ssim_weight=float(self._cfg_get(cfg_role, "ssim_weight", default_ssim)),
                pred_rgbs_out=pred_rgbs if collect_images else None,
                gt_images_out=gt_images if collect_images else None,
            )
            weight_i = float(self._cfg_get(cfg_role, "weight", default_weight))
            role_losses[str(role_name)] = loss_i
            role_total = role_total + weight_i * loss_i
            role_stats.update({key: float(value) for key, value in stats_i.items()})
            role_stats[f"phase_b_long/final_{role_name}_weight"] = float(weight_i)
        history_loss = 0.5 * (role_losses["history_recon"] + role_losses["history_nvs"])
        current_loss = 0.5 * (role_losses["current_recon"] + role_losses["current_nvs"])
        reg_loss, reg_stats = phase_b_long_offset_regularization(
            offset,
            weights=dict(getattr(self, "stage6_phase_b_long_offset_reg_cfg", {}) or {}),
        )
        total_loss = (
            role_total
            + float(self.stage6_phase_b_long_offset_reg_weight) * reg_loss
        )
        if not torch.isfinite(total_loss).all():
            raise RuntimeError("6_0_phase_b loss became NaN/Inf.")
        stats = {
            **role_stats,
            **reg_stats,
            "phase_b_long/final_history_weight": float(self.stage6_phase_b_long_history_weight),
            "phase_b_long/final_current_weight": float(self.stage6_phase_b_long_current_weight),
            "phase_b_long/offset_reg_weight": float(self.stage6_phase_b_long_offset_reg_weight),
            "phase_b_long/shape/" + str(roles.shape_name): 1.0,
            "phase_b_long/nvs_fallback_to_evidence_cam_ratio": float(
                roles.request_meta.get("nvs_fallback_to_evidence_cam_ratio", 0.0) or 0.0
            ),
        }
        return {
            "loss": total_loss,
            "base_G": base_state,
            "offset_state": offset,
            "vsm_state": vsm_state.detach_to_cache_optional(),
            "node_state_bg": node_state_bg,
            "node_state_distant": node_state_distant,
            "node_state_rigid": node_state_rigid,
            "roles": roles,
            "per_step": per_step,
            "stats": stats,
            "history_loss": history_loss.detach(),
            "current_loss": current_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "pred_rgbs": pred_rgbs[:max_collect_images] if collect_images else [],
            "gt_images": gt_images[:max_collect_images] if collect_images else [],
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        phase = str(getattr(self, "stage6_phase", "phase_A_block_local_unroll"))
        if phase == PHASE_B_LONG_NAME:
            return self._forward_6_0_phase_b_long(batch)
        if phase == PHASE_B_NAME:
            return self._forward_phase_b(batch)
        return self._forward_phase_a(batch)

    @torch.no_grad()
    def validate_v9_phase_a(
        self,
        batch: Dict[str, Any],
        *,
        k_values: List[int],
        max_K: int,
        mask_cfg: Optional[Dict[str, Any]] = None,
        compute_delta_stats: bool = True,
        compute_runtime_stats: bool = True,
        compute_memory_stats: bool = True,
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
            compute_delta_stats=bool(compute_delta_stats),
            compute_runtime_stats=bool(compute_runtime_stats),
            compute_memory_stats=bool(compute_memory_stats),
            save_images=bool(save_images),
            save_dir=save_dir,
            save_image_k_values=save_image_k_values,
            max_saved_cams=int(max_saved_cams),
        )

    def validate_long_phase_b(
        self,
        batch: Dict[str, Any],
        *,
        mask_policy: str = "non_sky_non_egocar",
        min_valid_pixels: int = 1,
        ablations: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        from models.streetforward.validation_long_phase_b_runner import DEFAULT_LONG_VSM_ABLATIONS, validate_long_phase_b

        self.eval()
        with torch.no_grad():
            return validate_long_phase_b(
                self,
                batch,
                mask_policy=str(mask_policy),
                min_valid_pixels=int(min_valid_pixels),
                ablations=tuple(ablations or DEFAULT_LONG_VSM_ABLATIONS),
            )

    def _train_step_6_0_phase_b_long(
        self,
        *,
        batch: Dict[str, Any],
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self._forward_6_0_phase_b_long(batch)
        loss = out["loss"]
        loss.backward()
        grad_group_sums = self._stage6_assert_required_group_grads_phase_b_long(out)
        skip_optimizer = bool(float(grad_group_sums.get("phase_b_long/skipped_no_support_rollout", 0.0)) > 0.0)
        if skip_optimizer:
            grad_norm = loss.detach().new_tensor(0.0)
        else:
            grad_norm = self._stage6_compute_and_check_grad_norm()
            self.optimizer.step()
        did_reset_node_state = False
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.reset_node_state()
            did_reset_node_state = True
        self.optimizer.zero_grad(set_to_none=True)
        roles = out["roles"]
        stats = dict(out.get("stats") or {})
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phase_b_long/loss_total": float(loss.detach().item()),
            "stage6/phase": "6_0_phase_b",
            "stage6/inner_K": float(roles.inner_K),
            "num_targets": int(out.get("num_targets", 0)),
            "num_source_views": int(out.get("num_source_views", 0)),
            "pred_rgbs": list(out.get("pred_rgbs") or []),
            "gt_images": list(out.get("gt_images") or []),
            "num_gaussians_bg": int(out["node_state_bg"].means.shape[0]),
            "num_gaussians_distant": int(out["node_state_distant"].means.shape[0]) if out["node_state_distant"] is not None else 0,
            "num_gaussians_rigid": int(out["node_state_rigid"].means.shape[0]) if out["node_state_rigid"] is not None else 0,
            "phase_b_long/grad_norm_total": float(grad_norm.detach().item()),
            "node_state_sync_reset": bool(did_reset_node_state),
            "node_state_cache_segments_bg": int(len(getattr(self, "node_states_bg", {}))),
            "node_state_cache_segments_distant": int(len(getattr(self, "node_states_distant", {}))),
            "node_state_cache_segments_rigid": int(len(getattr(self, "node_states_rigid", {}))),
            **{key: float(value) for key, value in stats.items() if isinstance(value, (int, float))},
            **grad_group_sums,
        }
        for item in list(out.get("per_step") or []):
            k = int(item.get("k", 0))
            for key, value in item.items():
                if key == "k" or not isinstance(value, (int, float)):
                    continue
                logs[f"phase_b_long/k{k}/{key}"] = float(value)
        if torch.cuda.is_available():
            logs["memory/allocated_gb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
            logs["memory/reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
            logs["memory/peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
        return logs

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
        phase = str(getattr(self, "stage6_phase", "phase_A_block_local_unroll"))
        if phase == PHASE_B_LONG_NAME:
            return self._train_step_6_0_phase_b_long(batch=batch, scheduler_node_sync=scheduler_node_sync)
        if phase == PHASE_B_NAME:
            return self._train_step_phase_b(batch=batch, scheduler_node_sync=scheduler_node_sync)
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
        did_reset_node_state = False
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.reset_node_state()
            did_reset_node_state = True
        self.optimizer.zero_grad(set_to_none=True)
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
            "mask/block_valid_ratio_final": float(final.get("block_valid_ratio", 0.0)),
            "mask/nearby_valid_ratio_final": float(final.get("nearby_valid_ratio", 0.0)),
            "mask/block_skipped_no_valid_pixels_final": float(final.get("block_skipped", 0.0)),
            "mask/nearby_skipped_no_valid_pixels_final": float(final.get("nearby_skipped", 0.0)),
            "mask/block_metric_valid_final": float(final.get("block_metric_valid", 0.0)),
            "mask/nearby_metric_valid_final": float(final.get("nearby_metric_valid", 0.0)),
            "mask/block_num_metric_refs_final": float(final.get("block_num_metric_refs", 0.0)),
            "mask/nearby_num_metric_refs_final": float(final.get("nearby_num_metric_refs", 0.0)),
            "phaseA/grad_norm_total": float(grad_norm.detach().item()),
            "node_state_sync_reset": bool(did_reset_node_state),
            "node_state_cache_segments_bg": int(len(getattr(self, "node_states_bg", {}))),
            "node_state_cache_segments_distant": int(len(getattr(self, "node_states_distant", {}))),
            "node_state_cache_segments_rigid": int(len(getattr(self, "node_states_rigid", {}))),
            **grad_group_sums,
        }
        for prefix in ("block", "nearby"):
            for metric_name in ("psnr", "ssim", "l1"):
                value = final.get(f"{prefix}_{metric_name}")
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    logs[f"phaseA/{prefix}_{metric_name}_final"] = value_f
        for item in per_step:
            k = int(item["k"])
            logs[f"phaseA/loss_block_k{k}"] = float(item.get("loss_block", 0.0))
            logs[f"phaseA/loss_nearby_k{k}"] = float(item.get("loss_nearby", 0.0))
            logs[f"mask/block_valid_ratio_k{k}"] = float(item.get("block_valid_ratio", 0.0))
            logs[f"mask/nearby_valid_ratio_k{k}"] = float(item.get("nearby_valid_ratio", 0.0))
            logs[f"mask/block_skipped_no_valid_pixels_k{k}"] = float(item.get("block_skipped", 0.0))
            logs[f"mask/nearby_skipped_no_valid_pixels_k{k}"] = float(item.get("nearby_skipped", 0.0))
            logs[f"mask/block_metric_valid_k{k}"] = float(item.get("block_metric_valid", 0.0))
            logs[f"mask/nearby_metric_valid_k{k}"] = float(item.get("nearby_metric_valid", 0.0))
            logs[f"mask/block_num_metric_refs_k{k}"] = float(item.get("block_num_metric_refs", 0.0))
            logs[f"mask/nearby_num_metric_refs_k{k}"] = float(item.get("nearby_num_metric_refs", 0.0))
            for prefix in ("block", "nearby"):
                for metric_name in ("psnr", "ssim", "l1"):
                    value = item.get(f"{prefix}_{metric_name}")
                    if value is None:
                        continue
                    value_f = float(value)
                    if math.isfinite(value_f):
                        logs[f"phaseA/{prefix}_{metric_name}_k{k}"] = value_f
        if did_reset_node_state:
            del out, loss
            empty_cache = str(os.environ.get("STAGE6_EMPTY_CACHE_ON_RESET", "")).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if empty_cache:
                gc.collect()
            if empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return logs

    def _train_step_phase_b(
        self,
        *,
        batch: Dict[str, Any],
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self._forward_phase_b(batch)
        loss = out["loss"]
        loss.backward()
        if bool(out.get("phase_b_v9_final_rollout_long", False)):
            grad_group_sums = self._stage6_assert_required_group_grads_phase_b_long(out)
            skip_optimizer = bool(float(grad_group_sums.get("phase_b_long/skipped_no_support_rollout", 0.0)) > 0.0)
            if skip_optimizer:
                grad_norm = loss.detach().new_tensor(0.0)
            else:
                grad_norm = self._stage6_compute_and_check_grad_norm()
                self.optimizer.step()
            tbptt_meta = dict(out.get("tbptt_meta") or {})
            tbptt_is_last_chunk = bool(tbptt_meta.get("is_last_chunk", False)) if tbptt_meta else False
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False)) if scheduler_node_sync is not None else False
            did_reset_node_state = False
            if tbptt_is_last_chunk:
                self._phase_b_clear_tbptt_cache_key(out["tbptt_key"])
                if reset_after_block:
                    self.reset_node_state()
                    did_reset_node_state = True
            elif reset_after_block:
                self.reset_node_state()
                self._phase_b_clear_tbptt_cache_key(out["tbptt_key"])
                did_reset_node_state = True
            else:
                self._phase_b_long_store_v9_state(
                    key=out["tbptt_key"],
                    base_state=out["base_G"],
                    vsm_state=out["vsm_state"],
                    offset_state=out["offset_state"],
                    written_refs=set(out.get("written_refs") or set()),
                    tbptt_meta=tbptt_meta,
                )
            self.optimizer.zero_grad(set_to_none=True)
            roles = out["roles"]
            stats = dict(out.get("stats") or {})
            logs: Dict[str, Any] = {
                "loss": float(loss.detach().item()),
                "phase_b/loss_total": float(loss.detach().item()),
                "phase_b_long/loss_total": float(loss.detach().item()),
                "stage6/phase": PHASE_B_NAME,
                "stage6/inner_K": float(roles.inner_K),
                "num_targets": int(out.get("num_targets", 0)),
                "num_source_views": int(out.get("num_source_views", 0)),
                "num_query_targets": int(out.get("num_query_targets", 0)),
                "pred_rgbs": list(out.get("pred_rgbs") or []),
                "gt_images": list(out.get("gt_images") or []),
                "num_gaussians_bg": int(out["node_state_bg"].means.shape[0]),
                "num_gaussians_distant": int(out["node_state_distant"].means.shape[0]) if out["node_state_distant"] is not None else 0,
                "num_gaussians_rigid": int(out["node_state_rigid"].means.shape[0]) if out["node_state_rigid"] is not None else 0,
                "phase_b_v9/final_supervision_step_idx": float(getattr(roles, "final_supervision_step_idx", -1)),
                "phase_b_v9/final_supervision_ref_count": float(len(getattr(roles, "final_target_indices", []) or [])),
                "phase_b_v9/intermediate_loss_ref_count": float(
                    sum(len(x) for x in roles.prefix_loss_refs_by_step[:-1])
                ),
                "phase_b_v9/evidence_ref_count": float(sum(len(x) for x in roles.evidence_refs_by_step)),
                "phase_b_v9/memory_write_steps": float(sum(1 for x in roles.memory_write_flags_by_step if bool(x))),
                "phase_b_v9/memory_write_ratio": float(
                    sum(1 for x in roles.memory_write_flags_by_step if bool(x)) / max(int(roles.inner_K), 1)
                ),
                "phase_b/tbptt_cache_hit": bool(out.get("tbptt_cache_hit", False)),
                "phase_b/tbptt_cache_size": int(len(getattr(self, "stage6_phase_b_tbptt_cache", {}))),
                "phase_b/tbptt_chunk_idx": int(tbptt_meta.get("chunk_idx", -1)) if tbptt_meta else -1,
                "phase_b/tbptt_is_last_chunk": bool(tbptt_is_last_chunk),
                "phase_b/grad_norm_total": float(grad_norm.detach().item()),
                "node_state_sync_reset": bool(did_reset_node_state),
                "node_state_cache_segments_bg": int(len(getattr(self, "node_states_bg", {}))),
                "node_state_cache_segments_distant": int(len(getattr(self, "node_states_distant", {}))),
                "node_state_cache_segments_rigid": int(len(getattr(self, "node_states_rigid", {}))),
                **{key: float(value) for key, value in stats.items() if isinstance(value, (int, float))},
                **grad_group_sums,
            }
            for item in list(out.get("per_step") or []):
                k = int(item.get("k", 0))
                for key2, value in item.items():
                    if key2 == "k" or not isinstance(value, (int, float)):
                        continue
                    logs[f"phase_b_v9/k{k}/{key2}"] = float(value)
            if torch.cuda.is_available():
                logs["memory/allocated_gb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
                logs["memory/reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
                logs["memory/peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
            return logs
        grad_group_sums = self._stage6_assert_required_group_grads_phase_b(out)
        grad_norm = self._stage6_compute_and_check_grad_norm()
        self.optimizer.step()
        tbptt_meta = dict(out.get("tbptt_meta") or {})
        strict_tbptt = bool(getattr(self, "stage6_phase_b_tbptt_strict", False))
        tbptt_is_last_chunk = bool(tbptt_meta.get("is_last_chunk", False)) if strict_tbptt else False
        if strict_tbptt and self.stage6_writeback_policy != "tbptt_cache_only":
            raise ValueError("Phase B strict TBPTT requires writeback_policy=tbptt_cache_only.")
        if self.stage6_writeback_policy == "block_end_detached":
            out["local_G"].writeback_detached(
                bg=out["node_state_bg"],
                distant=out["node_state_distant"],
                rigid=out["node_state_rigid"],
            )
        elif self.stage6_writeback_policy == "tbptt_cache_only":
            pass
        else:
            raise ValueError(f"unsupported Stage6_0 writeback_policy={self.stage6_writeback_policy!r}")
        did_reset_node_state = False
        reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False)) if scheduler_node_sync is not None else False
        if strict_tbptt:
            if reset_after_block and not tbptt_is_last_chunk:
                raise ValueError("Phase B strict TBPTT requires reset only at episode end / last chunk.")
            if tbptt_is_last_chunk:
                self._phase_b_clear_tbptt_cache_key(out["tbptt_key"])
                if reset_after_block:
                    self.reset_node_state()
                    did_reset_node_state = True
            else:
                self._phase_b_store_state(
                    key=out["tbptt_key"],
                    local_state=out["local_G"],
                    vsm_state=out["vsm_state"],
                    written_refs=set(out.get("written_refs") or set()),
                    tbptt_meta=tbptt_meta,
                )
        elif reset_after_block:
            self.reset_node_state()
            self._phase_b_clear_tbptt_cache()
            did_reset_node_state = True
        else:
            self._phase_b_store_state(
                key=out["tbptt_key"],
                local_state=out["local_G"],
                vsm_state=out["vsm_state"],
                written_refs=set(out.get("written_refs") or set()),
                tbptt_meta=tbptt_meta,
            )
        self.optimizer.zero_grad(set_to_none=True)

        per_step = list(out.get("per_step") or [])
        final = per_step[-1] if per_step else {}
        query_stats = dict(out.get("query_stats") or {})
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phase_b/loss_total": float(loss.detach().item()),
            "phase_b/loss_total_norm_by_weight": float(out.get("loss_total_norm_by_weight", 0.0)),
            "phase_b/loss_total_norm_by_K": float(out.get("loss_total_norm_by_K", 0.0)),
            f"phase_b/K{int(out['roles'].inner_K)}/loss_total_norm_by_weight": float(
                out.get("loss_total_norm_by_weight", 0.0)
            ),
            f"phase_b/K{int(out['roles'].inner_K)}/loss_total_norm_by_K": float(out.get("loss_total_norm_by_K", 0.0)),
            "stage6/phase": "B",
            "stage6/inner_K": float(out["roles"].inner_K),
            "phase_b/rollout_K": float(out["roles"].inner_K),
            "num_targets": int(out.get("num_targets", 0)),
            "num_source_views": int(out.get("num_source_views", 0)),
            "num_query_targets": int(out.get("num_query_targets", 0)),
            "pred_rgbs": list(out.get("pred_rgbs") or []),
            "gt_images": list(out.get("gt_images") or []),
            "num_gaussians_bg": int(out["node_state_bg"].means.shape[0]),
            "num_gaussians_distant": int(out["node_state_distant"].means.shape[0]) if out["node_state_distant"] is not None else 0,
            "num_gaussians_rigid": int(out["node_state_rigid"].means.shape[0]) if out["node_state_rigid"] is not None else 0,
            "phase_b/prefix_loss_final": float(final.get("loss_prefix", 0.0)),
            "phase_b/prefix_rgb_l1_final": float(final.get("prefix_l1", 0.0)),
            "phase_b/prefix_ssim_loss_final": float(final.get("prefix_ssim", 0.0)),
            "phase_b/prefix_static_psnr_final": float(final.get("prefix_psnr", 0.0)),
            "phase_b/prefix_final_static_psnr": float(final.get("prefix_psnr", 0.0)),
            "phase_b/prefix_valid_ratio_final": float(final.get("prefix_valid_ratio", 0.0)),
            "phase_b/query_loss": float(out.get("query_loss", 0.0)),
            f"phase_b/K{int(out['roles'].inner_K)}/query_loss": float(out.get("query_loss", 0.0)),
            "phase_b/query_weight": float(out.get("query_weight", 0.0)),
            "phase_b/query_event_l1": float(query_stats.get("query_event_l1", 0.0)),
            "phase_b/query_visible_acc": float(query_stats.get("query_visible_acc", 0.0)),
            "phase_b/query_visible_bce": float(query_stats.get("query_visible_bce", 0.0)),
            "phase_b/query_support_l1": float(query_stats.get("query_support_l1", 0.0)),
            "phase_b/query_obs_code_l1": float(query_stats.get("query_obs_code_l1", 0.0)),
            "phase_b/query_event_l1_all": float(query_stats.get("query_event_l1_all", 0.0)),
            "phase_b/query_event_l1_rigid": float(query_stats.get("query_event_l1_rigid", 0.0)),
            "phase_b/query_visible_acc_rigid": float(query_stats.get("query_visible_acc_rigid", 0.0)),
            "phase_b/query_rows_all": float(query_stats.get("query_rows_all", 0.0)),
            "phase_b/vsm_token_usage_mean": float(final.get("vsm_token_usage_mean", 0.0)),
            "phase_b/vsm_router_entropy": float(final.get("vsm_router_entropy", 0.0)),
            f"phase_b/K{int(out['roles'].inner_K)}/vsm_router_entropy": float(final.get("vsm_router_entropy", 0.0)),
            f"phase_b/K{int(out['roles'].inner_K)}/vsm_token_usage_mean": float(final.get("vsm_token_usage_mean", 0.0)),
            "phase_b/vsm_update_norm": float(final.get("vsm_update_token_delta_norm", 0.0)),
            "phase_b/vsm_update_count_mean": float(final.get("vsm_update_count_mean", 0.0)),
            "phase_b/vsm_update_token_delta_norm": float(final.get("vsm_update_token_delta_norm", 0.0)),
            "phase_b/vsm_update_proto_delta_norm": float(final.get("vsm_update_proto_delta_norm", 0.0)),
            "phase_b/vsm_update_global_delta_norm": float(final.get("vsm_update_global_delta_norm", 0.0)),
            "phase_b/vsm_update_assign_entropy": float(final.get("vsm_update_assign_entropy", 0.0)),
            "phase_b/vsm_update_assign_max": float(final.get("vsm_update_assign_max", 0.0)),
            "phase_b/vsm_update_assign_usage_max": float(final.get("vsm_update_assign_usage_max", 0.0)),
            "phase_b/vsm_token_pair_cosine_mean": float(final.get("vsm_token_pair_cosine_mean", 0.0)),
            "phase_b/vsm_token_pair_cosine_max": float(final.get("vsm_token_pair_cosine_max", 0.0)),
            "phase_b/vsm_token_variance_mean": float(final.get("vsm_token_variance_mean", 0.0)),
            "phase_b/vsm_proto_pair_distance_mean": float(final.get("vsm_proto_pair_distance_mean", 0.0)),
            "phase_b/vsm_ctx_norm": float(final.get("vsm_ctx_norm", 0.0)),
            "phase_b/vsm_ctx_norm_bg": float(final.get("vsm_bg_vsm_ctx_norm", final.get("vsm_ctx_norm", 0.0))),
            "phase_b/vsm_ctx_norm_rigid": float(final.get("vsm_rigid_vsm_ctx_norm", 0.0)),
            "phase_b/vsm_update_count_bg": float(final.get("vsm_bg_vsm_update_count_mean", final.get("vsm_update_count_mean", 0.0))),
            "phase_b/vsm_update_count_rigid": float(final.get("vsm_rigid_vsm_update_count_mean", 0.0)),
            "phase_b/delta_bg_means_norm": float(final.get("delta_bg_means_norm", 0.0)),
            "phase_b/delta_bg_opacity_norm": float(final.get("delta_bg_opacity_norm", 0.0)),
            "phase_b/rigid_seen_ratio": float(final.get("rigid_seen_ratio", 0.0)),
            "phase_b/rigid_delta_means_norm": float(final.get("rigid_delta_means_norm", 0.0)),
            "phase_b/rigid_delta_opacity_norm": float(final.get("rigid_delta_opacity_norm", 0.0)),
            "phase_b/rigid_noop_mean": float(final.get("rigid_noop_mean", 0.0)),
            "phase_b/noop_mean": float(final.get("noop_mean", 0.0)),
            "phase_b/confidence_mean": float(final.get("confidence_mean", 0.0)),
            "phase_b/evidence_ref_count": float(sum(len(x) for x in out["roles"].evidence_refs_by_step)),
            "phase_b/prefix_ref_count": float(sum(len(x) for x in out["roles"].prefix_loss_refs_by_step)),
            "phase_b/query_ref_count": float(len(out["roles"].query_label_refs)),
            "phase_b/leak/query_evidence_overlap": float(out.get("leak/query_evidence_overlap", 0.0)),
            "phase_b/leak/query_written_overlap": float(out.get("leak/query_written_overlap", 0.0)),
            "phase_b/tbptt_cache_hit": bool(out.get("tbptt_cache_hit", False)),
            "phase_b/tbptt_cache_size": int(len(getattr(self, "stage6_phase_b_tbptt_cache", {}))),
            "phase_b/tbptt_chunk_idx": int(tbptt_meta.get("chunk_idx", -1)) if tbptt_meta else -1,
            "phase_b/tbptt_is_last_chunk": bool(tbptt_meta.get("is_last_chunk", False)) if tbptt_meta else False,
            "phase_b/grad_norm_total": float(grad_norm.detach().item()),
            "node_state_sync_reset": bool(did_reset_node_state),
            "node_state_cache_segments_bg": int(len(getattr(self, "node_states_bg", {}))),
            "node_state_cache_segments_distant": int(len(getattr(self, "node_states_distant", {}))),
            "node_state_cache_segments_rigid": int(len(getattr(self, "node_states_rigid", {}))),
            **grad_group_sums,
        }
        adapter = getattr(self.stage6_posterior_updater, "vsm_ctx_adapter", None)
        if adapter is not None:
            weight = getattr(adapter, "weight", None)
            if weight is not None:
                logs["phase_b/vsm_ctx_adapter_weight_norm"] = float(weight.detach().norm().item())
        for item in per_step:
            k = int(item["k"])
            logs[f"phase_b/memory_write_k{k}"] = float(item.get("memory_write", 0.0))
            logs[f"phase_b/repeat_idx_k{k}"] = float(item.get("repeat_idx", 0.0))
            logs[f"phase_b/block_idx_k{k}"] = float(item.get("block_idx", -1.0))
            logs[f"phase_b/prefix_loss_k{k}"] = float(item.get("loss_prefix", 0.0))
            logs[f"phase_b/prefix_rgb_l1_k{k}"] = float(item.get("prefix_l1", 0.0))
            logs[f"phase_b/prefix_static_psnr_k{k}"] = float(item.get("prefix_psnr", 0.0))
            logs[f"phase_b/prefix_valid_ratio_k{k}"] = float(item.get("prefix_valid_ratio", 0.0))
            logs[f"phase_b/vsm_ctx_norm_k{k}"] = float(item.get("vsm_ctx_norm", 0.0))
            logs[f"phase_b/vsm_update_assign_entropy_k{k}"] = float(item.get("vsm_update_assign_entropy", 0.0))
            logs[f"phase_b/vsm_update_token_delta_norm_k{k}"] = float(item.get("vsm_update_token_delta_norm", 0.0))
            logs[f"phase_b/vsm_token_pair_cosine_mean_k{k}"] = float(item.get("vsm_token_pair_cosine_mean", 0.0))
            logs[f"phase_b/vsm_ctx_norm_rigid_k{k}"] = float(item.get("vsm_rigid_vsm_ctx_norm", 0.0))
            logs[f"phase_b/delta_bg_means_norm_k{k}"] = float(item.get("delta_bg_means_norm", 0.0))
            logs[f"phase_b/rigid_delta_means_norm_k{k}"] = float(item.get("rigid_delta_means_norm", 0.0))
        return logs

    def _assert_group_nonzero_grad(
        self,
        *,
        group_name: str,
        params: List[torch.nn.Parameter],
        required: bool = True,
        detail: Optional[str] = None,
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
            msg = f"{group_name} has zero gradient in Stage6_0."
            if detail:
                msg = f"{msg} {detail}"
            raise RuntimeError(msg)
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

    def _stage6_assert_required_group_grads_phase_b(self, out: Dict[str, Any]) -> Dict[str, float]:
        query_weight = float(out.get("query_weight", 0.0))
        adapter = getattr(self.stage6_posterior_updater, "vsm_ctx_adapter", None)
        adapter_params = list(adapter.parameters()) if adapter is not None else []
        sums: Dict[str, float] = {
            "grad/stage6_vsm_ctx_adapter_sum": self._assert_group_nonzero_grad(
                group_name="stage6_posterior_updater.vsm_ctx_adapter",
                params=adapter_params,
                required=bool(self.stage6_phase_b_prefix_enable and float(self.stage6_phase_b_prefix_weight) > 0.0),
            ),
            "grad/stage6_vsm_sum": self._assert_group_nonzero_grad(
                group_name="stage6_vsm",
                params=list(self.stage6_vsm.parameters()) if self.stage6_vsm is not None else [],
                required=bool(query_weight > 0.0),
            ),
            "grad/stage6_query_decoder_sum": self._assert_group_nonzero_grad(
                group_name="stage6_query_decoder",
                params=list(self.stage6_query_decoder.parameters()) if self.stage6_query_decoder is not None else [],
                required=bool(query_weight > 0.0),
            ),
        }
        return sums

    @staticmethod
    def _stage6_phase_b_long_grad_detail(out: Dict[str, Any], grad_sums: Dict[str, float]) -> str:
        roles = out.get("roles")
        stats = dict(out.get("stats") or {})
        per_step = list(out.get("per_step") or [])

        def _f(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _max_step(key: str) -> float:
            vals = [_f(item.get(key, 0.0)) for item in per_step if isinstance(item, dict)]
            return float(max(vals)) if vals else 0.0

        def _last_step(key: str) -> float:
            if not per_step or not isinstance(per_step[-1], dict):
                return 0.0
            return _f(per_step[-1].get(key, 0.0))

        role_parts: List[str] = []
        for role_name in ("history_recon", "history_nvs", "current_recon", "current_nvs"):
            prefix = f"phase_b_long/final_{role_name}"
            role_parts.append(
                (
                    f"{role_name}:refs={_f(stats.get(prefix + '_num_refs', 0.0)):.0f},"
                    f"loss={_f(stats.get(prefix + '_loss', 0.0)):.6g},"
                    f"valid={_f(stats.get(prefix + '_valid_ratio', 0.0)):.4f},"
                    f"skipped={_f(stats.get(prefix + '_skipped_no_valid_pixels', 0.0)):.0f}"
                )
            )

        request_meta = dict(getattr(roles, "request_meta", {}) or {}) if roles is not None else {}
        target_roles = list(request_meta.get("target_image_roles") or [])
        target_role_counts = {
            str(role): int(target_roles.count(role))
            for role in sorted(set(str(x) for x in target_roles))
        }
        source_refs = list(request_meta.get("source_image_refs") or [])[:8]
        target_refs = list(request_meta.get("target_image_refs") or [])[:8]
        scene_id = request_meta.get("scene_id", "unknown")
        segment_id = request_meta.get("segment_id", "unknown")
        shape_name = getattr(roles, "shape_name", request_meta.get("shape_name", "unknown")) if roles is not None else "unknown"
        inner_k = getattr(roles, "inner_K", request_meta.get("inner_K", 0)) if roles is not None else request_meta.get("inner_K", 0)
        return (
            "6_0_phase_b debug: "
            f"scene={scene_id} segment={segment_id} shape={shape_name} inner_K={inner_k}; "
            f"grad_sums={{{', '.join(f'{k}={v:.6g}' for k, v in grad_sums.items())}}}; "
            f"vsm_bg_seen_rows_max={_max_step('vsm_bg_seen_rows'):.0f}, "
            f"vsm_bg_seen_ratio_max={_max_step('vsm_bg_seen_ratio'):.6f}, "
            f"vsm_bg_write_gate_max={_max_step('vsm_bg_write_gate_mean'):.6f}, "
            f"vsm_bg_hard_valid_ratio_max={_max_step('vsm_bg_hard_valid_ratio'):.6f}, "
            f"vsm_bg_support_max={_max_step('vsm_bg_support_max'):.6g}, "
            f"vsm_bg_support_positive_ratio_max={_max_step('vsm_bg_support_positive_ratio'):.6f}, "
            f"vsm_bg_support_fallback_used_max={_max_step('vsm_bg_support_fallback_used'):.0f}, "
            f"vsm_bg_h_norm_last={_last_step('vsm_bg_h_norm'):.6f}, "
            f"vsm_distant_active_rows_max={_max_step('vsm_distant_active_rows'):.0f}, "
            f"vsm_distant_seen_rows_max={_max_step('vsm_distant_seen_rows'):.0f}, "
            f"vsm_distant_support_max={_max_step('vsm_distant_support_max'):.6g}, "
            f"offset_bg_delta_means_norm_max={_max_step('offset_bg_delta_means_norm'):.6f}, "
            f"offset_bg_delta_opacity_norm_max={_max_step('offset_bg_delta_opacity_norm'):.6f}; "
            f"roles=[{'; '.join(role_parts)}]; "
            f"target_role_counts={target_role_counts}; "
            f"source_refs_preview={source_refs}; target_refs_preview={target_refs}"
        )

    def _stage6_phase_b_long_allow_no_support_skip(self) -> bool:
        model_cfg = self._cfg_get(self.config, "model", {}) or {}
        stage_cfg = self._cfg_get(model_cfg, "stage6_0", {}) or {}
        long_cfg = self._cfg_get(stage_cfg, "phase_b_long", {}) or {}
        return bool(self._cfg_get(long_cfg, "skip_no_support_rollout", True))

    @staticmethod
    def _stage6_phase_b_long_is_no_support_rollout(out: Dict[str, Any]) -> bool:
        per_step = list(out.get("per_step") or [])
        if not per_step:
            return False

        def _f(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        bg_support_max = max((_f(item.get("vsm_bg_support_max", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        bg_seen_rows = max((_f(item.get("vsm_bg_seen_rows", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        rigid_support_max = max((_f(item.get("vsm_rigid_support_max", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        rigid_seen_rows = max((_f(item.get("vsm_rigid_seen_rows", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        distant_support_max = max((_f(item.get("vsm_distant_support_max", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        distant_seen_rows = max((_f(item.get("vsm_distant_seen_rows", 0.0)) for item in per_step if isinstance(item, dict)), default=0.0)
        return (
            float(bg_support_max) <= 0.0
            and float(bg_seen_rows) <= 0.0
            and float(rigid_support_max) <= 0.0
            and float(rigid_seen_rows) <= 0.0
            and float(distant_support_max) <= 0.0
            and float(distant_seen_rows) <= 0.0
        )

    def _stage6_assert_required_group_grads_phase_b_long(self, out: Dict[str, Any]) -> Dict[str, float]:
        grad_sums = {
            "grad/stage6_long_vsm_sum": self._assert_group_nonzero_grad(
                group_name="stage6_long_vsm",
                params=list(self.stage6_long_vsm.parameters()) if self.stage6_long_vsm is not None else [],
                required=False,
            ),
            "grad/stage6_long_offset_decoder_sum": self._assert_group_nonzero_grad(
                group_name="stage6_long_offset_decoder",
                params=list(self.stage6_long_offset_decoder.parameters()) if self.stage6_long_offset_decoder is not None else [],
                required=False,
            ),
        }
        zero_groups = [key for key, value in grad_sums.items() if float(value) == 0.0]
        if zero_groups:
            detail = self._stage6_phase_b_long_grad_detail(out, grad_sums)
            if self._stage6_phase_b_long_allow_no_support_skip() and self._stage6_phase_b_long_is_no_support_rollout(out):
                skip_count = int(getattr(self, "_phase_b_long_no_support_skip_count", 0)) + 1
                self._phase_b_long_no_support_skip_count = int(skip_count)
                if skip_count <= 8 or skip_count in {16, 32, 64, 128}:
                    logger.warning(
                        "Skipping 6_0_phase_b optimizer update for no-support rollout (%d): %s",
                        int(skip_count),
                        detail,
                    )
                grad_sums["phase_b_long/skipped_no_support_rollout"] = 1.0
                grad_sums["phase_b_long/no_support_skip_count"] = float(skip_count)
                return grad_sums
            raise RuntimeError(
                "6_0_phase_b has zero gradient in required trainable groups: "
                f"{zero_groups}. {detail}"
            )
        grad_sums["phase_b_long/skipped_no_support_rollout"] = 0.0
        return grad_sums

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

    @staticmethod
    def _stage6_to_device_state_dict(sd: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {
            str(k): (v.to(device) if torch.is_tensor(v) else v)
            for k, v in dict(sd or {}).items()
        }

    def _load_phase_b_export_payload(self, ckpt: Dict[str, Any], *, device: torch.device) -> None:
        measurement_sd = self._stage6_to_device_state_dict(dict(ckpt.get("measurement_frontend") or {}), device)
        if measurement_sd:
            self.load_state_dict(measurement_sd, strict=False)

        struct_sd = self._stage6_to_device_state_dict(dict(ckpt.get("struct_event_decoder") or {}), device)
        if not struct_sd:
            raise ValueError("Stage6_0 Phase B export payload missing struct_event_decoder.")
        self.stage6_struct_event_decoder.load_state_dict(struct_sd, strict=True)

        updater_sd = self._stage6_to_device_state_dict(dict(ckpt.get("posterior_updater_base") or {}), device)
        if not updater_sd:
            raise ValueError("Stage6_0 Phase B export payload missing posterior_updater_base.")
        missing, unexpected = self.stage6_posterior_updater.load_state_dict(updater_sd, strict=False)
        bad_missing = [k for k in missing if not str(k).startswith("vsm_ctx_adapter.")]
        bad_unexpected = [k for k in unexpected if not str(k).startswith("vsm_ctx_adapter.")]
        if bad_missing or bad_unexpected:
            raise ValueError(
                "Stage6_0 Phase B failed to load posterior updater base payload: "
                f"missing={bad_missing} unexpected={bad_unexpected}"
            )

    def load_init_checkpoint_payload(
        self,
        ckpt: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        weights_only: bool = True,
        path: Optional[str] = None,
    ) -> bool:
        _ = (weights_only, path)
        if str(getattr(self, "stage6_phase", "phase_A_block_local_unroll")) not in {PHASE_B_NAME, PHASE_B_LONG_NAME}:
            return False
        target_device = device if device is not None else self.device
        if str(ckpt.get("export_type", "")) == "stage6_0_phase_a_for_phase_b":
            self._load_phase_b_export_payload(ckpt, device=target_device)
            return True

        init_cfg = self._cfg_get(self.config, "initialization", {}) or {}
        phase_b_init = self._cfg_get(init_cfg, "phase_b_from_phase_a", {}) or {}
        if (
            str(getattr(self, "stage6_phase", "phase_A_block_local_unroll")) == PHASE_B_LONG_NAME
            and bool(self._cfg_get(phase_b_init, "enable", False))
            and bool(self._cfg_get(phase_b_init, "reject_plain_model_state_dict", True))
        ):
            raise ValueError(
                "6_0_phase_b requires an init checkpoint with export_type="
                "stage6_0_phase_a_for_phase_b. Plain model_state_dict checkpoints are rejected "
                "for Phase B Long initialization."
            )

        sd = ckpt.get("model_state_dict")
        if sd is None:
            return False
        missing, unexpected = self.load_state_dict(sd, strict=False)
        allowed_missing_prefixes = (
            "stage6_vsm.",
            "stage6_query_decoder.",
            "stage6_long_vsm.",
            "stage6_long_offset_decoder.",
            "stage6_posterior_updater.vsm_ctx_adapter.",
        )
        allowed_unexpected_prefixes = (
            "stage6_posterior_updater.vsm_ctx_adapter.",
        )
        if bool(self._cfg_get(self._cfg_get(self.config, "model", {}) or {}, "allow_missing_legacy_sparse_conv", False)):
            allowed_unexpected_prefixes = allowed_unexpected_prefixes + ("sparse_conv.",)
        bad_missing = [k for k in missing if not str(k).startswith(allowed_missing_prefixes)]
        bad_unexpected = [k for k in unexpected if not str(k).startswith(allowed_unexpected_prefixes)]
        if bad_missing or bad_unexpected:
            raise ValueError(
                "Stage6_0 Phase B ordinary checkpoint load was not compatible: "
                f"missing={bad_missing} unexpected={bad_unexpected}"
            )
        return True

    def build_light_checkpoint_extra(self, *, step: int) -> Dict[str, Any]:
        optimizer_cfg = self._cfg_get(self.config, "optimizer", {}) or {}
        lr_scheduler_cfg = self._cfg_get(self.config, "lr_scheduler", {}) or {}
        return {
            "format": "streetforward_stage6_0_ckpt_v2",
            "resume_semantics": "resume_model_optimizer_runtime_when_available",
            "restore_train_scheduler_runtime": True,
            "restore_rng_state": True,
            "restore_node_state_runtime": True,
            "model_stage": "6_0",
            "phase": str(getattr(self, "stage6_phase", "phase_A_block_local_unroll")),
            "global_step": int(step),
            "optimizer_signature": self._stage6_optimizer_signature(),
            "optimizer_cfg": _to_plain_dict(optimizer_cfg),
            "lr_scheduler_cfg": _to_plain_dict(lr_scheduler_cfg),
            "lr_scheduler": {
                "type": str(self._cfg_get(lr_scheduler_cfg, "type", "")),
                "global_step": int(step),
                "active": False,
            },
        }

    def load_optimizer_state_from_checkpoint(self, payload: Dict[str, Any]) -> bool:
        old_sig = payload.get("optimizer_signature")
        if old_sig is not None:
            cur_sig = self._stage6_optimizer_signature()
            if old_sig != cur_sig:
                logger.warning("Skip Stage6_0 optimizer load: signature mismatch.")
                return False
        opt_state = payload.get("optimizer_state_dict")
        if opt_state is None:
            logger.warning("Skip Stage6_0 optimizer load: checkpoint has no optimizer_state_dict.")
            return False
        self.optimizer.load_state_dict(opt_state)
        if old_sig is None:
            logger.warning(
                "Loaded Stage6_0 optimizer from checkpoint without optimizer_signature; "
                "group compatibility was validated only by torch.optim."
            )
        return True


__all__ = ["MinimalStreetForwardStage6_0"]
