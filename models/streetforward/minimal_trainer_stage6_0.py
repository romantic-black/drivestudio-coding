from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import math
import os
import time
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.iforward.biggs_assignment import build_biggs_assignments, build_rigid_active_assignment
from models.iforward.biggs_event_decoder import BigGSToFineEventDecoder
from models.iforward.biggs_parent_projector import (
    BigGSParentProjection,
    project_biggs_active_rigid_parents,
    project_biggs_parents,
)
from models.iforward.biggs_parent_stats import (
    init_parent_branch_runtime,
    projection_from_runtime,
    update_parent_branch_runtime,
)
from models.iforward.biggs_state import BigGSBlockRuntime, IForwardBigGSState
from models.iforward.dino_feature_cache import DINOFeatureCache
from models.iforward.fwhr_lift import aggregate_fwhr_child_lift
from models.iforward.amp_policy import amp_dtype_id, build_amp_policy, storage_dtype_from_name
from models.iforward.parent_spatial_backbone import ParentStructInput, empty_parent_struct_input
from models.iforward.stage3_0 import (
    GatherConfig,
    ParentContextFusion,
    ParentQueryBuilder,
    SparseGatherLift,
    build_cuda_scalar_anchor_stats,
    build_projected_meta_anchor_stats,
    center_child_detail_by_parent,
    support_center_sparse_gather,
)
from models.iforward.stage3_0.losses import merge_stage3_reg_terms
from models.iforward.stage3_0.sparse_grid_sample import prepare_value_nchw
from models.iforward.versions import (
    STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION,
    is_stage3_optimizer_memory_iforward_version,
)
from models.streetforward.minimal_trainer_stage4_0 import spatial_hw_from_image_tensor
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.struct_decoders.common import cat_param_dict
from models.streetforward.stage6_0 import (
    AppearanceDetailPack,
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
IFORWARD_STAGE3_0_VERSION = STAGE3_0_SCALAR_ANCHOR_CHILD_SUPPORT_PARENT_LEGACY_VERSION


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

    def _stage3_0_memory_aux_enabled(self) -> bool:
        if not bool(getattr(self, "stage3_0_enabled", False)) or not torch.cuda.is_available():
            return False
        lifting_cfg = getattr(self, "stage3_0_lifting_cfg", {}) or {}
        default_interval = int(self._cfg_get(lifting_cfg, "gather_aux_interval", 100))
        interval = int(self._cfg_get(lifting_cfg, "memory_aux_interval", default_interval))
        if interval <= 0:
            return False
        global_step = int(getattr(self, "stage3_0_global_step", 0))
        return bool(global_step % int(interval) == 0)

    def _stage3_0_memory_aux(self, label: str, *, include_step_max: bool = False) -> Dict[str, float]:
        if not self._stage3_0_memory_aux_enabled():
            return {}
        prefix = f"iforward/stage3/mem_{label}"
        out = {
            f"{prefix}_allocated_mb": float(torch.cuda.memory_allocated() / (1024.0 ** 2)),
            f"{prefix}_reserved_mb": float(torch.cuda.memory_reserved() / (1024.0 ** 2)),
            f"{prefix}_max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)),
        }
        if bool(include_step_max):
            out["iforward/stage3/mem_step_max_allocated_mb"] = out[f"{prefix}_max_allocated_mb"]
        return out

    def _repair_training_cfg(self) -> Any:
        return getattr(self, "iforward_repair_training_cfg", {}) or {}

    def _repair_training_visit_kind(self, visit_meta: Optional[Dict[str, Any]]) -> str:
        if isinstance(visit_meta, dict):
            return str(visit_meta.get("visit_kind", "") or "")
        return ""

    def _repair_training_train_2d_mode(self, visit_meta: Optional[Dict[str, Any]]) -> str:
        if not isinstance(visit_meta, dict):
            return ""
        mode = str(visit_meta.get("train_2d_mode", "") or "")
        if mode:
            return mode
        stage32 = visit_meta.get("iforward_stage3_2")
        if isinstance(stage32, dict):
            return str(stage32.get("train_2d_mode", "") or "")
        return ""

    def _repair_training_stage3_2_frozen_no_grad(self, visit_meta: Optional[Dict[str, Any]]) -> bool:
        cfg = self._repair_training_cfg()
        return bool(
            bool(self._cfg_get(cfg, "enable", False))
            and self._repair_training_train_2d_mode(visit_meta) == "frozen_no_grad"
        )

    def _repair_training_enabled_for_visit(self, visit_meta: Optional[Dict[str, Any]]) -> bool:
        cfg = self._repair_training_cfg()
        if not bool(self._cfg_get(cfg, "enable", False)):
            return False
        if self._repair_training_stage3_2_frozen_no_grad(visit_meta):
            return True
        start_step = int(self._cfg_get(cfg, "start_step", 0) or 0)
        global_step = int(visit_meta.get("global_step", 0) or 0) if isinstance(visit_meta, dict) else 0
        if int(global_step) < int(start_step):
            return False
        kinds = self._cfg_get(cfg, "kinds", ["repair"]) or ["repair"]
        kind_set = {str(kind) for kind in list(kinds)}
        return self._repair_training_visit_kind(visit_meta) in kind_set

    def _repair_training_freeze_2d_for_visit(self, visit_meta: Optional[Dict[str, Any]]) -> bool:
        cfg = self._repair_training_cfg()
        if self._repair_training_stage3_2_frozen_no_grad(visit_meta):
            return True
        return bool(
            self._repair_training_enabled_for_visit(visit_meta)
            and bool(self._cfg_get(cfg, "freeze_2d_frontend", True))
        )

    def _repair_training_no_grad_2d_for_visit(self, visit_meta: Optional[Dict[str, Any]]) -> bool:
        cfg = self._repair_training_cfg()
        if self._repair_training_stage3_2_frozen_no_grad(visit_meta):
            return True
        return bool(
            self._repair_training_freeze_2d_for_visit(visit_meta)
            and bool(self._cfg_get(cfg, "no_grad_2d_forward", True))
        )

    def _detach_cnn_inputs_for_repair_training(self, cnn_inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(cnn_inputs)
        for key in ("features_2d", "fwhr_detail_2d", "stage3_dino_native_2d"):
            value = out.get(key)
            if torch.is_tensor(value):
                out[key] = value.detach()
        return out

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

        iforward_cfg = self._cfg_get(model_cfg, "iforward", {}) or {}
        ifwd_version = str(self._cfg_get(iforward_cfg, "version", ""))
        biggs_cfg = self._cfg_get(iforward_cfg, "biggs", {}) or {}
        biggs_enabled = ifwd_version in {
            "stage2_0_biggs_parent_lifting",
            "stage2_0_biggs_cuda_exact_diagonal_projector",
            "stage2_0_biggs_incremental_whdd",
            "stage2_0_biggs_compact16_residualonly",
            "stage2_0_biggs_grld_dinov2base_concat48",
            "stage2_0_fwhr_lift_grld_dinov2base",
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or is_stage3_optimizer_memory_iforward_version(ifwd_version)
        if bool(biggs_enabled):
            if bool(self._cfg_get(biggs_cfg, "enable", True)) is not True:
                raise ValueError(f"{ifwd_version} requires model.iforward.biggs.enable=true")
            if bool(self._cfg_get(self._cfg_get(iforward_cfg, "history_gate", {}) or {}, "enable", False)):
                raise ValueError("stage2_0_biggs_parent_lifting requires history_gate.enable=false")
            if bool(self._cfg_get(self._cfg_get(iforward_cfg, "history_gate_v2", {}) or {}, "enable", False)):
                raise ValueError("stage2_0_biggs_parent_lifting requires history_gate_v2.enable=false")
            if bool(self._cfg_get(self._cfg_get(iforward_cfg, "adc_lite", {}) or {}, "enable", False)):
                raise ValueError("stage2_0_biggs_parent_lifting requires adc_lite.enable=false")
            observe_cfg = self._cfg_get(biggs_cfg, "observe", {}) or {}
            lifting_cfg = self._cfg_get(biggs_cfg, "lifting", {}) or {}
            stage3_enabled = is_stage3_optimizer_memory_iforward_version(ifwd_version)
            if bool(stage3_enabled):
                stage3_lifting = self._cfg_get(iforward_cfg, "lifting", None)
                if stage3_lifting is None:
                    raise ValueError("Stage3_0 requires model.iforward.lifting.")
                if str(self._cfg_get(stage3_lifting, "type", "")).lower() != "full_sparse_gather":
                    raise ValueError("Stage3_0 requires model.iforward.lifting.type=full_sparse_gather.")
                if str(self._cfg_get(lifting_cfg, "type", "")).lower() == "fwhr":
                    raise ValueError("Stage3_0 forbids legacy model.iforward.biggs.lifting.type=fwhr.")
                backend = str(self._cfg_get(stage3_lifting, "scalar_anchor_backend", "cuda_scalar_anchor")).lower()
                if backend not in {"projected_meta", "cuda_scalar_anchor"}:
                    raise ValueError(f"unsupported Stage3_0 scalar_anchor_backend={backend!r}")
                parent_lift_cfg = self._cfg_get(stage3_lifting, "parent", {}) or {}
                parent_lift_type = str(self._cfg_get(parent_lift_cfg, "type", "legacy_direct_lift")).lower()
                if parent_lift_type not in {"legacy_direct_lift", "sparse_gather"}:
                    raise ValueError(f"unsupported Stage3_0 parent.type={parent_lift_type!r}")
                dino_native_cfg = self._cfg_get(stage3_lifting, "dino_native", {}) or {}
                if parent_lift_type == "legacy_direct_lift" and bool(self._cfg_get(dino_native_cfg, "enable", False)):
                    raise ValueError("Stage3_0 parent.type=legacy_direct_lift forbids dino_native.enable=true.")
                if bool(self._cfg_get(dino_native_cfg, "enable", False)):
                    dino_cache_cfg = (
                        self._cfg_get(
                            self._cfg_get(self._cfg_get(model_cfg, "feature_extractor", {}) or {}, "dino", {}) or {},
                            "cache",
                            {},
                        )
                        or {}
                    )
                    if not bool(self._cfg_get(dino_cache_cfg, "enable", False)):
                        raise ValueError("Stage3_0 dino_native.enable=true requires feature_extractor.dino.cache.enable=true.")
                    if str(self._cfg_get(dino_cache_cfg, "level", "")).lower() != "backbone_intermediate":
                        raise ValueError(
                            "Stage3_0 dino_native.enable=true requires "
                            "feature_extractor.dino.cache.level=backbone_intermediate."
                        )
            is_fwhr = str(self._cfg_get(lifting_cfg, "type", "")).lower() == "fwhr"
            if not bool(stage3_enabled) and not bool(is_fwhr) and bool(self._cfg_get(observe_cfg, "parent_scene_for_lifting", True)) is not True:
                raise ValueError("stage2_0_biggs_parent_lifting requires parent_scene_for_lifting=true")
            skip_cfg = self._cfg_get(biggs_cfg, "child_observation_skip", {}) or {}
            if bool(self._cfg_get(skip_cfg, "enable", False)) and (
                bool(self._cfg_get(skip_cfg, "trainable", False))
                or not bool(self._cfg_get(skip_cfg, "no_grad", True))
            ):
                raise ValueError("stage2_0_biggs_parent_lifting forbids trainable child_observation_skip")

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
        is_stage2_1_parent_temporal = ifwd_version in {
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or is_stage3_optimizer_memory_iforward_version(ifwd_version)
        if bool(self._cfg_get(base_measurement, "require_obs_code", True)) is not True and not is_stage2_1_parent_temporal:
            raise ValueError("Stage6_0 Phase A requires V4 obs_code.")
        expected_obs_dim = 0 if is_stage2_1_parent_temporal else 2
        if int(self._cfg_get(base_measurement, "obs_code_dim", expected_obs_dim)) != expected_obs_dim:
            raise ValueError(f"Stage6_0 Phase A requires base_measurement.obs_code_dim={expected_obs_dim}.")
        source_grad_mode = str(self._cfg_get(base_measurement, "source_evidence_grad_mode", "no_grad_v4")).strip()
        detach_v4_outputs = bool(self._cfg_get(base_measurement, "detach_v4_outputs", True))
        train_2d_frontend = bool(self._cfg_get(base_measurement, "train_2d_frontend", False))
        train_residual_unet = bool(self._cfg_get(base_measurement, "train_residual_unet", train_2d_frontend))
        train_fusion_neck = bool(self._cfg_get(base_measurement, "train_fusion_neck", train_2d_frontend))
        train_v4_lift = bool(self._cfg_get(base_measurement, "train_v4_lift", False))
        train_dinov2 = bool(self._cfg_get(base_measurement, "train_dinov2", False))
        feature_extractor_cfg = self._cfg_get(model_cfg, "feature_extractor", {}) or {}
        feature_extractor_type = str(self._cfg_get(feature_extractor_cfg, "type", "")).strip().lower()
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
            if bool(biggs_enabled) and source_grad_mode == "no_grad_v4":
                if not detach_v4_outputs:
                    raise ValueError(
                        "stage2_0_biggs_parent_lifting with no_grad_v4 requires "
                        "base_measurement.detach_v4_outputs=true."
                    )
                if train_2d_frontend or train_residual_unet or train_fusion_neck or train_v4_lift:
                    raise ValueError(
                        "stage2_0_biggs_parent_lifting with no_grad_v4 must keep the 2D/V4 frontend frozen."
                    )
            elif source_grad_mode != "train_2d_detach_alpha":
                raise ValueError(
                    "Stage6_0 Phase A from_scratch requires "
                    "base_measurement.source_evidence_grad_mode=train_2d_detach_alpha."
                )
            elif detach_v4_outputs:
                raise ValueError("Stage6_0 Phase A from_scratch requires base_measurement.detach_v4_outputs=false.")
            elif not train_2d_frontend:
                raise ValueError("Stage6_0 Phase A from_scratch requires base_measurement.train_2d_frontend=true.")
            elif not train_residual_unet:
                raise ValueError("Stage6_0 Phase A from_scratch requires train_residual_unet=true.")
            elif feature_extractor_type not in {"residual_only", "dinov2_residual_concat", "fwhr_dinov2_residual"} and not train_fusion_neck:
                raise ValueError(
                    "Stage6_0 Phase A from_scratch requires train_fusion_neck=true unless "
                    "model.feature_extractor.type is residual_only, dinov2_residual_concat, or fwhr_dinov2_residual."
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
        event_dim = int(self._cfg_get(struct_cfg, "event_dim", token_dim))
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
        if bool(self._cfg_get(distant_scope, "update_quat", False)):
            raise ValueError("Stage6_0 Phase A requires distant update_quat=false.")

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
        iforward_cfg = self._cfg_get(model_cfg, "iforward", {}) or {}
        biggs_cfg = self._cfg_get(iforward_cfg, "biggs", {}) or {}
        ifwd_version = str(self._cfg_get(iforward_cfg, "version", ""))
        is_stage2_biggs_version = ifwd_version in {
            "stage2_0_biggs_parent_lifting",
            "stage2_0_biggs_cuda_exact_diagonal_projector",
            "stage2_0_biggs_incremental_whdd",
            "stage2_0_biggs_compact16_residualonly",
                "stage2_0_biggs_grld_dinov2base_concat48",
                "stage2_0_fwhr_lift_grld_dinov2base",
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or is_stage3_optimizer_memory_iforward_version(ifwd_version)
        is_stage2_biggs = bool(is_stage2_biggs_version) and bool(self._cfg_get(biggs_cfg, "enable", True))
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
        self.stage6_detach_source_render_for_cnn = bool(
            self._cfg_get(base_measurement, "detach_source_render_for_cnn", True)
        )
        self.stage6_cnn_view_chunk_size = int(self._cfg_get(base_measurement, "cnn_view_chunk_size", 0) or 0)
        if int(self.stage6_cnn_view_chunk_size) < 0:
            raise ValueError(
                "model.stage6_0.base_measurement.cnn_view_chunk_size must be >= 0, "
                f"got {int(self.stage6_cnn_view_chunk_size)}"
            )
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
        train_dino_adapter = bool(self._cfg_get(base_measurement, "train_dino_adapter", False))

        def mark_trainable(module: nn.Module, prefix: str) -> None:
            for name, param in module.named_parameters(recurse=True):
                param.requires_grad_(True)
                self.stage6_measurement_trainable_param_names.add(f"{prefix}.{name}")

        if hasattr(image_feature_extractor, "residual_unet"):
            if train_residual_unet:
                mark_trainable(image_feature_extractor.residual_unet, "image_feature_extractor.residual_unet")
                detail_head = getattr(image_feature_extractor, "detail_head", None)
                if isinstance(detail_head, nn.Module):
                    mark_trainable(detail_head, "image_feature_extractor.detail_head")
        elif train_residual_unet:
            mark_trainable(image_feature_extractor, "image_feature_extractor")

        if hasattr(image_feature_extractor, "fusion_neck"):
            if train_fusion_neck:
                mark_trainable(image_feature_extractor.fusion_neck, "image_feature_extractor.fusion_neck")
        elif train_fusion_neck and not hasattr(image_feature_extractor, "residual_unet"):
            mark_trainable(image_feature_extractor, "image_feature_extractor")

        dino_adapter = getattr(image_feature_extractor, "dino_adapter", None)
        if train_dino_adapter and isinstance(dino_adapter, nn.Module):
            for attr_name in ("proj", "fuse"):
                module = getattr(dino_adapter, attr_name, None)
                if isinstance(module, nn.Module):
                    mark_trainable(module, f"image_feature_extractor.dino_adapter.{attr_name}")

        if self.stage6_train_v4_lift:
            for attr_name in ("feature_backprojector", "alpha_t_extractor_v4"):
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Module):
                    mark_trainable(module, attr_name)

        if len(self.stage6_measurement_trainable_param_names) == 0 and not (
            bool(is_stage2_biggs) and self.stage6_source_evidence_grad_mode == "no_grad_v4"
        ):
            raise ValueError("Stage6_0 Phase A from_scratch did not enable any 2D frontend parameters.")

    def _init_stage6_modules(self, config: Any) -> None:
        model_cfg = self._require_key(config, "model", "config")
        iforward_cfg = self._cfg_get(model_cfg, "iforward", {}) or {}
        biggs_cfg = self._cfg_get(iforward_cfg, "biggs", {}) or {}
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
        self.stage6_event_dim = int(self._cfg_get(struct_cfg, "event_dim", token_dim))
        if int(self.stage6_event_dim) != int(token_dim):
            raise ValueError(
                f"Stage6_0 struct_event_decoder.event_dim must equal token.token_dim ({int(token_dim)}), "
                f"got {int(self.stage6_event_dim)}."
            )
        self.stage6_posterior_event_dim = int(self._cfg_get(updater_cfg, "event_dim", self.stage6_event_dim))
        self.stage6_ctx_dim = self.stage6_posterior_event_dim
        self.stage6_hidden_dim = max(int(self._cfg_get(updater_cfg, "stage_hidden_dim", self.stage6_event_dim)), 0)
        self.stage6_feat_2d_dim = int(
            self._cfg_get(
                struct_cfg,
                "feat_2d_dim",
                getattr(self, "stage5_2_feat_2d_channels", getattr(self, "feat_2d_channels", 32)),
            )
        )
        self.stage6_near_debug_check_spconv_order = bool(
            self._cfg_get(near_cfg, "debug_check_spconv_order", False)
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
        appearance_detail_cfg = self._cfg_get(updater_cfg, "appearance_detail", {}) or {}
        appearance_detail_gate_init = self._cfg_get(appearance_detail_cfg, "gate_init", {}) or {}
        appearance_detail_attribute_gates = self._cfg_get(appearance_detail_cfg, "attribute_gates", {}) or {}
        appearance_detail_attribute_gate_max = self._cfg_get(appearance_detail_cfg, "attribute_gate_max", {}) or {}
        self.stage6_posterior_updater = Stage6PosteriorUpdater(
            event_dim=self.stage6_posterior_event_dim,
            ctx_dim=self.stage6_posterior_event_dim,
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
            vsm_ctx_dim=int(self._cfg_get(phase_b_hooks, "vsm_ctx_dim", self.stage6_posterior_event_dim)),
            branch_clamps=branch_clamps,
            output_hidden=bool(self._cfg_get(updater_cfg, "output_hidden", True)),
            output_confidence=bool(self._cfg_get(updater_cfg, "output_confidence", True)),
            output_noop=bool(self._cfg_get(updater_cfg, "output_noop", True)),
            appearance_detail_enable=bool(self._cfg_get(appearance_detail_cfg, "enable", False)),
            appearance_detail_dim=int(self._cfg_get(appearance_detail_cfg, "detail_dim", 8)),
            appearance_detail_gate_init=dict(appearance_detail_gate_init),
            appearance_detail_gate_max=float(self._cfg_get(appearance_detail_cfg, "gate_max", 1.0)),
            appearance_detail_attribute_gates=dict(appearance_detail_attribute_gates),
            appearance_detail_attribute_gate_max=dict(appearance_detail_attribute_gate_max),
            invalid_update_policy=str(self._cfg_get(updater_cfg, "invalid_update_policy", "none")),
        ).to(self.device)
        ifwd_version = str(self._cfg_get(iforward_cfg, "version", ""))
        self.stage2_0_biggs_enabled = ifwd_version in {
            "stage2_0_biggs_parent_lifting",
            "stage2_0_biggs_cuda_exact_diagonal_projector",
            "stage2_0_biggs_incremental_whdd",
            "stage2_0_biggs_compact16_residualonly",
            "stage2_0_biggs_grld_dinov2base_concat48",
            "stage2_0_fwhr_lift_grld_dinov2base",
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or is_stage3_optimizer_memory_iforward_version(ifwd_version)
        self.stage2_1_parent_temporal_enabled = ifwd_version in {
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or is_stage3_optimizer_memory_iforward_version(ifwd_version)
        self.stage2_0_biggs_cfg = dict(biggs_cfg or {})
        self.stage2_0_biggs_assignment_cfg = self._cfg_get(biggs_cfg, "assignment", {}) or {}
        self.stage2_0_biggs_projector_cfg = self._cfg_get(biggs_cfg, "parent_projector", {}) or {}
        self.stage2_0_biggs_parent_state_cfg = self._cfg_get(biggs_cfg, "parent_state", {}) or {}
        self.stage2_0_biggs_observe_cfg = self._cfg_get(biggs_cfg, "observe", {}) or {}
        self.stage2_0_biggs_lifting_cfg = self._cfg_get(biggs_cfg, "lifting", {}) or {}
        self.stage3_0_enabled = is_stage3_optimizer_memory_iforward_version(ifwd_version)
        self.stage3_0_lifting_cfg = self._cfg_get(iforward_cfg, "lifting", {}) or {}
        self.iforward_amp_policy = build_amp_policy(config, inference_only=True)
        self.iforward_repair_training_cfg = self._cfg_get(iforward_cfg, "repair_training", {}) or {}
        self.stage3_0_gather_reg_terms: Dict[str, torch.Tensor] = {}
        self.stage3_0_last_aux: Dict[str, float] = {}
        self.stage3_0_global_step = 0
        detail_support_cfg = self._cfg_get(self.stage2_0_biggs_lifting_cfg, "detail_support_min", {}) or {}
        self.stage2_0_fwhr_detail_support_min = {
            "bg": float(self._cfg_get(detail_support_cfg, "bg", getattr(self, "bg_src_backproject_support_min", 0.0))),
            "distant": float(
                self._cfg_get(detail_support_cfg, "distant", getattr(self, "distant_src_backproject_support_min", 0.0))
            ),
            "rigid": float(
                self._cfg_get(detail_support_cfg, "rigid", getattr(self, "rigid_src_backproject_support_min", 0.0))
            ),
        }
        parent_state_policy = str(self._cfg_get(self.stage2_0_biggs_parent_state_cfg, "exact_refresh_policy", "block_enter")).lower()
        if parent_state_policy not in {"block_enter", "none"}:
            raise ValueError(
                "stage2_0 BigGS parent_state.exact_refresh_policy currently supports "
                f"'block_enter' or 'none', got {parent_state_policy!r}."
            )
        self._stage2_0_biggs_parent_runtime_block_counter = 0
        self._stage2_0_biggs_parent_last_drift_block = -1
        projector_backend = str(self._cfg_get(self.stage2_0_biggs_projector_cfg, "backend", "")).lower()
        if projector_backend in {"cuda_exact_diag_forward_only", "cuda_exact_diagonal_forward_only"}:
            if bool(self._cfg_get(self.stage2_0_biggs_projector_cfg, "grad_to_local_state", False)):
                raise ValueError("cuda_exact_diag_forward_only requires parent_projector.grad_to_local_state=false")
            grad_mode = str(self._cfg_get(self.stage2_0_biggs_projector_cfg, "grad_mode", "stop_geometry")).lower()
            if grad_mode != "stop_geometry":
                raise ValueError("cuda_exact_diag_forward_only requires parent_projector.grad_mode=stop_geometry")
        self.stage2_0_biggs_assignment_cache_scope = str(
            self._cfg_get(self.stage2_0_biggs_assignment_cfg, "cache_scope", "episode")
        )
        self.stage2_0_biggs_assignment_ignore_episode_id = bool(
            self._cfg_get(
                self.stage2_0_biggs_assignment_cfg,
                "ignore_episode_id",
                self.stage2_0_biggs_assignment_cache_scope != "episode",
            )
        )
        self.stage2_0_biggs_assignment_cache_max_items = int(
            self._cfg_get(self.stage2_0_biggs_assignment_cfg, "cache_max_items", 0) or 0
        )
        self.stage2_0_biggs_assignment_cache_device_copy = bool(
            self._cfg_get(self.stage2_0_biggs_assignment_cfg, "cache_device_copy", False)
        )
        self._stage2_0_biggs_assignment_cache: "OrderedDict[Tuple[Any, ...], IForwardBigGSState]" = OrderedDict()
        self._stage2_0_biggs_assignment_device_cache: "OrderedDict[Tuple[Any, ...], IForwardBigGSState]" = OrderedDict()
        self.stage2_0_biggs_return_debug_stats = bool(
            self._cfg_get(self.stage2_0_biggs_observe_cfg, "return_debug_stats", True)
        )
        dino_cache_cfg = (
            self._cfg_get(self._cfg_get(self._cfg_get(model_cfg, "feature_extractor", {}) or {}, "dino", {}) or {}, "cache", {})
            or {}
        )
        self.dino_feature_cache = None
        self.dino_feature_cache_level = "adapter_output"
        if bool(self.stage2_0_biggs_enabled) and bool(self._cfg_get(dino_cache_cfg, "enable", False)):
            level = str(self._cfg_get(dino_cache_cfg, "level", "adapter_output")).lower()
            if level not in {"adapter_output", "backbone_intermediate"}:
                raise ValueError(
                    "Stage2_0 BigGS DINO cache supports level=adapter_output or "
                    f"backbone_intermediate, got {level!r}"
                )
            self.dino_feature_cache_level = level
            self.dino_feature_cache = DINOFeatureCache(
                dtype=str(self._cfg_get(dino_cache_cfg, "dtype", "float16")),
                cpu_pinned=bool(self._cfg_get(dino_cache_cfg, "cpu_pinned", True)),
                cpu_max_items=int(self._cfg_get(dino_cache_cfg, "cpu_max_items", 64)),
                gpu_max_items=int(self._cfg_get(dino_cache_cfg, "gpu_max_items", 2)),
                async_copy=bool(self._cfg_get(dino_cache_cfg, "async_copy", True)),
                fail_if_trainable=bool(self._cfg_get(dino_cache_cfg, "fail_if_trainable", True)),
            )
        self.biggs_child_decoder: Optional[BigGSToFineEventDecoder] = None
        if bool(self.stage2_0_biggs_enabled):
            child_cfg = self._cfg_get(biggs_cfg, "child_decoder", {}) or {}
            self.biggs_child_decoder = BigGSToFineEventDecoder.from_config(
                child_cfg,
                event_dim=int(self.stage6_event_dim),
            ).to(self.device)
        self.stage3_parent_query: Optional[ParentQueryBuilder] = None
        self.stage3_parent_query_obs2d: Optional[nn.Module] = None
        self.stage3_parent_context_fusion: Optional[ParentContextFusion] = None
        self.stage3_child_query: Optional[nn.Module] = None
        self.stage3_parent_gather: Optional[SparseGatherLift] = None
        self.stage3_child_gather: Optional[SparseGatherLift] = None
        self.stage3_parent_query_use_obs2d_lift = False
        self.stage3_parent_query_use_dino_native_lift = False
        self.stage3_parent_context_use_dino_native_fusion = False
        self.stage3_dino_native_enabled = False
        self.stage3_dino_native_dim = 0
        self.stage3_parent_lifting_type = "legacy_direct_lift"
        if bool(self.stage3_0_enabled):
            if str(self._cfg_get(self.stage3_0_lifting_cfg, "type", "")).lower() != "full_sparse_gather":
                raise ValueError("Stage3_0 requires model.iforward.lifting.type=full_sparse_gather")
            scalar_anchor_backend = str(
                self._cfg_get(self.stage3_0_lifting_cfg, "scalar_anchor_backend", "cuda_scalar_anchor")
            ).lower()
            if scalar_anchor_backend not in {"projected_meta", "cuda_scalar_anchor"}:
                raise ValueError(f"unsupported Stage3_0 scalar_anchor_backend={scalar_anchor_backend!r}")
            context_dim = int(self._cfg_get(self.stage3_0_lifting_cfg, "context_dim", self.stage6_feat_2d_dim))
            detail_dim = int(self._cfg_get(self.stage3_0_lifting_cfg, "detail_dim", 8))
            if int(context_dim) != int(self.stage6_feat_2d_dim):
                raise ValueError(
                    f"Stage3_0 context_dim must match Stage6 feat_2d_dim={int(self.stage6_feat_2d_dim)}, got {context_dim}"
                )
            schedule_cfg = _to_plain_dict(self._cfg_get(self.stage3_0_lifting_cfg, "training_schedule", {}) or {})

            def _gather_cfg(name: str) -> Dict[str, Any]:
                base = _to_plain_dict(self._cfg_get(self.stage3_0_lifting_cfg, name, {}) or {})
                for key, value in schedule_cfg.items():
                    base.setdefault(str(key), value)
                return base

            parent_cfg = _gather_cfg("parent_gather")
            child_cfg = _gather_cfg("child_gather")
            parent_lift_cfg = self._cfg_get(self.stage3_0_lifting_cfg, "parent", {}) or {}
            self.stage3_parent_lifting_type = str(
                self._cfg_get(parent_lift_cfg, "type", "legacy_direct_lift")
            ).lower()
            if self.stage3_parent_lifting_type not in {"legacy_direct_lift", "sparse_gather"}:
                raise ValueError(f"unsupported Stage3_0 parent.type={self.stage3_parent_lifting_type!r}")
            parent_query_dim = int(self._cfg_get(parent_cfg, "query_dim", 96))
            parent_query_cfg = self._cfg_get(self.stage3_0_lifting_cfg, "parent_query", {}) or {}
            parent_context_cfg = self._cfg_get(self.stage3_0_lifting_cfg, "parent_context", {}) or {}
            dino_native_cfg = self._cfg_get(self.stage3_0_lifting_cfg, "dino_native", {}) or {}
            if self.stage3_parent_lifting_type == "legacy_direct_lift" and bool(
                self._cfg_get(dino_native_cfg, "enable", False)
            ):
                raise ValueError("Stage3_0 parent.type=legacy_direct_lift forbids dino_native.enable=true")
            obs2d_dim = int(self._cfg_get(parent_query_cfg, "obs2d_lift_dim", 0) or 0)
            self.stage3_parent_query_use_obs2d_lift = bool(
                self._cfg_get(parent_query_cfg, "use_obs2d_lift", False)
            ) and obs2d_dim > 0 and self.stage3_parent_lifting_type == "sparse_gather"
            self.stage3_parent_query_use_dino_native_lift = bool(
                self._cfg_get(parent_query_cfg, "use_dino_native_lift", False)
            ) and self.stage3_parent_lifting_type == "sparse_gather"
            self.stage3_parent_context_use_dino_native_fusion = bool(
                self._cfg_get(parent_context_cfg, "use_dino_native_fusion", False)
            ) and self.stage3_parent_lifting_type == "sparse_gather"
            self.stage3_dino_native_enabled = bool(self._cfg_get(dino_native_cfg, "enable", False)) and (
                self.stage3_parent_lifting_type == "sparse_gather"
            )
            self.stage3_dino_native_dim = int(self._cfg_get(dino_native_cfg, "out_channels", 16))
            if bool(self.stage3_dino_native_enabled):
                expected_level = str(self._cfg_get(dino_native_cfg, "cache_level", "backbone_intermediate")).lower()
                if expected_level != "backbone_intermediate":
                    raise ValueError("Stage3 dino_native.cache_level must be backbone_intermediate")
                if getattr(self, "dino_feature_cache", None) is None:
                    raise ValueError("Stage3 dino_native.enable=true requires model.feature_extractor.dino.cache.enable=true")
                if str(getattr(self, "dino_feature_cache_level", "adapter_output")).lower() != "backbone_intermediate":
                    raise ValueError("Stage3 dino_native requires model.feature_extractor.dino.cache.level=backbone_intermediate")
            if bool(self.stage3_parent_query_use_dino_native_lift) and not bool(self.stage3_dino_native_enabled):
                raise ValueError("parent_query.use_dino_native_lift=true requires dino_native.enable=true")
            if bool(self.stage3_parent_context_use_dino_native_fusion) and not bool(self.stage3_dino_native_enabled):
                raise ValueError("parent_context.use_dino_native_fusion=true requires dino_native.enable=true")
            extra_query_dim = 0
            if bool(self.stage3_parent_query_use_obs2d_lift):
                extra_query_dim += int(obs2d_dim)
                self.stage3_parent_query_obs2d = nn.Sequential(
                    nn.LayerNorm(int(context_dim)),
                    nn.Linear(int(context_dim), int(obs2d_dim)),
                ).to(self.device)
            if bool(self.stage3_parent_query_use_dino_native_lift):
                extra_query_dim += int(self.stage3_dino_native_dim)
            if self.stage3_parent_lifting_type == "sparse_gather":
                self.stage3_parent_query = ParentQueryBuilder(
                    query_dim=parent_query_dim,
                    extra_input_dim=int(extra_query_dim),
                ).to(self.device)
                if bool(self.stage3_parent_context_use_dino_native_fusion):
                    self.stage3_parent_context_fusion = ParentContextFusion(
                        context_dim=int(context_dim),
                        dino_dim=int(self.stage3_dino_native_dim),
                    ).to(self.device)
                self.stage3_parent_gather = SparseGatherLift(
                    value_dim=int(context_dim),
                    config=GatherConfig.from_config(parent_cfg, defaults=GatherConfig(query_dim=parent_query_dim)),
                ).to(self.device)
            child_type = str(self._cfg_get(child_cfg, "type", "support_center")).lower()
            if child_type != "support_center":
                raise ValueError("Stage3_0 child_gather.type must be support_center")
            self.stage3_child_query = None
            self.stage3_child_gather = None
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
        if getattr(self, "biggs_child_decoder", None) is not None:
            add_group(
                logical_name="biggs_child_decoder",
                named_params=[
                    (f"biggs_child_decoder.{name}", param)
                    for name, param in self.biggs_child_decoder.named_parameters()  # type: ignore[union-attr]
                ],
                lr=lr_for("biggs_child_decoder"),
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
        return local_state.to(device=self.device).to_node_states_detached()

    def _local_to_node_states(
        self,
        local_state: LocalGSState,
        *,
        detach: bool,
    ) -> tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        state = local_state.to(device=self.device)
        if bool(detach):
            return state.to_node_states_detached()
        return state.to_node_states_grad()

    def _stage2_0_biggs_ids_from_batch(self, batch: Dict[str, Any]) -> tuple[int, int, int]:
        ifwd = self._cfg_get(batch, "_iforward", {}) or self._cfg_get(self._cfg_get(batch, "request_meta", {}) or {}, "iforward", {}) or {}
        scene_id = int(self._cfg_get(ifwd, "scene_id", self._cfg_get(batch, "scene_id", -1)) or -1)
        segment_id = int(self._cfg_get(ifwd, "segment_id", self._cfg_get(batch, "segment_id", -1)) or -1)
        episode_id = int(self._cfg_get(ifwd, "episode_id", -1) or -1)
        return scene_id, segment_id, episode_id

    @staticmethod
    def _stage2_0_biggs_cfg_hash(cfg: Any) -> str:
        try:
            payload = json.dumps(_to_plain_dict(cfg), sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            payload = repr(cfg)
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    @staticmethod
    def _stage2_0_biggs_rigid_id_hash(rigid: Optional[NodeStateRigid]) -> tuple[int, int, int, int, int]:
        if rigid is None or not hasattr(rigid, "point_ids"):
            return (0, 0, 0, 0, 0)
        ids = rigid.point_ids
        if ids is None or int(ids.numel()) == 0:
            return (0, 0, 0, 0, 0)
        flat = ids.detach().reshape(-1).long()
        n = int(flat.numel())
        first = int(flat[0].detach().cpu().item())
        last = int(flat[-1].detach().cpu().item())
        sum_mod = int((flat.sum(dtype=torch.long) % 2147483647).detach().cpu().item())
        sq_sum_mod = int(((flat * flat).sum(dtype=torch.long) % 2147483647).detach().cpu().item())
        return (n, first, last, sum_mod, sq_sum_mod)

    def _stage2_0_biggs_assignment_builder_id(self) -> int:
        builder = str(self._cfg_get(getattr(self, "stage2_0_biggs_assignment_cfg", {}) or {}, "builder", "python_bucket"))
        return 1 if builder.lower() in ("vectorized_sort_segment", "vectorized") else 0

    def _stage2_0_biggs_assignment_scope_id(self) -> int:
        scope = str(getattr(self, "stage2_0_biggs_assignment_cache_scope", "episode"))
        return 1 if scope == "scene_segment_topology" else 0

    def _stage2_0_biggs_cache_key(
        self,
        *,
        ids: tuple[int, int, int],
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
    ) -> tuple[Any, ...]:
        scene_id, segment_id, episode_id = ids
        scope = str(getattr(self, "stage2_0_biggs_assignment_cache_scope", "episode"))
        if scope == "episode" and not bool(getattr(self, "stage2_0_biggs_assignment_ignore_episode_id", False)):
            episode_part: Any = int(episode_id)
        else:
            episode_part = "any_episode"
        return (
            str(scope),
            int(scene_id),
            int(segment_id),
            episode_part,
            int(bg.means.shape[0]),
            int(distant.means.shape[0]) if distant is not None else 0,
            int(rigid.means.shape[0]) if rigid is not None else 0,
            self._stage2_0_biggs_rigid_id_hash(rigid),
            int(getattr(self, "sh_degree", 0)),
            self._stage2_0_biggs_cfg_hash(getattr(self, "stage2_0_biggs_assignment_cfg", {}) or {}),
        )

    def _stage2_0_biggs_cache_get(self, key: tuple[Any, ...]) -> Optional[IForwardBigGSState]:
        cache = getattr(self, "_stage2_0_biggs_assignment_cache", None)
        if not isinstance(cache, OrderedDict):
            self._stage2_0_biggs_assignment_cache = OrderedDict()
            return None
        state = cache.get(key)
        if state is None:
            return None
        cache.move_to_end(key)
        return state

    def _stage2_0_biggs_device_cache_get(self, key: tuple[Any, ...], device: torch.device) -> Optional[IForwardBigGSState]:
        if not bool(getattr(self, "stage2_0_biggs_assignment_cache_device_copy", False)):
            return None
        cache = getattr(self, "_stage2_0_biggs_assignment_device_cache", None)
        if not isinstance(cache, OrderedDict):
            self._stage2_0_biggs_assignment_device_cache = OrderedDict()
            return None
        device_key = tuple(key) + (str(device),)
        state = cache.get(device_key)
        if state is None:
            return None
        cache.move_to_end(device_key)
        return state

    def _stage2_0_biggs_cache_put(
        self,
        *,
        key: tuple[Any, ...],
        state_cpu: IForwardBigGSState,
        state_device: Optional[IForwardBigGSState] = None,
    ) -> None:
        max_items = int(getattr(self, "stage2_0_biggs_assignment_cache_max_items", 0) or 0)
        if max_items <= 0:
            return
        cache = getattr(self, "_stage2_0_biggs_assignment_cache", None)
        if not isinstance(cache, OrderedDict):
            self._stage2_0_biggs_assignment_cache = OrderedDict()
            cache = self._stage2_0_biggs_assignment_cache
        cache[key] = state_cpu.detach()
        cache.move_to_end(key)
        device_cache = getattr(self, "_stage2_0_biggs_assignment_device_cache", None)
        if not isinstance(device_cache, OrderedDict):
            self._stage2_0_biggs_assignment_device_cache = OrderedDict()
            device_cache = self._stage2_0_biggs_assignment_device_cache
        if state_device is not None and bool(getattr(self, "stage2_0_biggs_assignment_cache_device_copy", False)):
            device_key = tuple(key) + (str(self.device),)
            device_cache[device_key] = state_device
            device_cache.move_to_end(device_key)
        while len(cache) > max_items:
            old_key, _ = cache.popitem(last=False)
            prefix = tuple(old_key)
            for dkey in [k for k in list(device_cache.keys()) if tuple(k[: len(prefix)]) == prefix]:
                device_cache.pop(dkey, None)
        while len(device_cache) > max_items:
            device_cache.popitem(last=False)

    @staticmethod
    def _stage2_0_branch_match(assign: Any, n: int) -> bool:
        return assign is not None and int(getattr(assign, "num_children", -1)) == int(n)

    def _stage2_0_biggs_state_matches(
        self,
        *,
        state: Optional[IForwardBigGSState],
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
        ids: tuple[int, int, int],
    ) -> bool:
        if state is None:
            return False
        scene_id, segment_id, episode_id = ids
        if int(scene_id) >= 0 and int(state.scene_id) != int(scene_id):
            return False
        if int(segment_id) >= 0 and int(state.segment_id) != int(segment_id):
            return False
        scope = str(getattr(self, "stage2_0_biggs_assignment_cache_scope", "episode"))
        ignore_episode = bool(getattr(self, "stage2_0_biggs_assignment_ignore_episode_id", False))
        if scope == "episode" and not ignore_episode and int(episode_id) >= 0 and int(state.episode_id) != int(episode_id):
            return False
        if not self._stage2_0_branch_match(state.bg, int(bg.means.shape[0])):
            return False
        if distant is not None and not self._stage2_0_branch_match(state.distant, int(distant.means.shape[0])):
            return False
        if distant is None and state.distant is not None and int(state.distant.num_children) > 0:
            return False
        if rigid is not None and not self._stage2_0_branch_match(state.rigid, int(rigid.means.shape[0])):
            return False
        if rigid is None and state.rigid is not None and int(state.rigid.num_children) > 0:
            return False
        return True

    def _stage2_0_get_or_build_biggs_state(
        self,
        *,
        existing: Optional[IForwardBigGSState],
        batch: Dict[str, Any],
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
        ids_override: Optional[tuple[int, int, int]] = None,
    ) -> IForwardBigGSState:
        state_cpu, _, _ = self._stage2_0_get_or_build_biggs_state_for_observe(
            existing=existing,
            batch=batch,
            bg=bg,
            distant=distant,
            rigid=rigid,
            ids_override=ids_override,
        )
        return state_cpu

    def _stage2_0_get_or_build_biggs_state_for_observe(
        self,
        *,
        existing: Optional[IForwardBigGSState],
        batch: Dict[str, Any],
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
        ids_override: Optional[tuple[int, int, int]] = None,
    ) -> tuple[IForwardBigGSState, IForwardBigGSState, Dict[str, float]]:
        ids = ids_override if ids_override is not None else self._stage2_0_biggs_ids_from_batch(batch)
        if int(ids[0]) < 0 or int(ids[1]) < 0:
            raise ValueError(
                "BigGS requires valid scene_id and segment_id for stable assignment cache: "
                f"ids={tuple(int(x) for x in ids)} batch_keys={sorted(str(k) for k in batch.keys())}"
            )
        key = self._stage2_0_biggs_cache_key(ids=ids, bg=bg, distant=distant, rigid=rigid)
        stats = {
            "iforward/biggs/assignment_cache_hit": 0.0,
            "iforward/biggs/assignment_build_ms": 0.0,
            "iforward/biggs/assignment_to_device_ms": 0.0,
            "iforward/biggs/assignment_cache_size": float(len(getattr(self, "_stage2_0_biggs_assignment_cache", {}) or {})),
            "iforward/biggs/assignment_cache_scope_id": float(self._stage2_0_biggs_assignment_scope_id()),
            "iforward/biggs/assignment_builder_id": float(self._stage2_0_biggs_assignment_builder_id()),
        }

        state_cpu: Optional[IForwardBigGSState] = None
        if self._stage2_0_biggs_state_matches(state=existing, bg=bg, distant=distant, rigid=rigid, ids=ids):
            state_cpu = existing.detach()  # type: ignore[union-attr]
            stats["iforward/biggs/assignment_cache_hit"] = 1.0
        else:
            cached = self._stage2_0_biggs_cache_get(key)
            if cached is not None and self._stage2_0_biggs_state_matches(
                state=cached,
                bg=bg,
                distant=distant,
                rigid=rigid,
                ids=ids,
            ):
                state_cpu = cached.detach()
                stats["iforward/biggs/assignment_cache_hit"] = 1.0

        device_obj = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        state_device = self._stage2_0_biggs_device_cache_get(key, device_obj)
        if state_cpu is None:
            t_build = time.perf_counter()
            bg_assign, distant_assign, rigid_assign = build_biggs_assignments(
                bg=bg,
                distant=distant,
                rigid=rigid,
                assignment_cfg=getattr(self, "stage2_0_biggs_assignment_cfg", {}) or {},
            )
            stats["iforward/biggs/assignment_build_ms"] = float((time.perf_counter() - t_build) * 1000.0)
            episode_id = int(ids[2])
            if str(getattr(self, "stage2_0_biggs_assignment_cache_scope", "episode")) != "episode":
                episode_id = -1
            state_cpu = IForwardBigGSState(
                bg=bg_assign,
                distant=distant_assign,
                rigid=rigid_assign,
                scene_id=int(ids[0]),
                segment_id=int(ids[1]),
                episode_id=int(episode_id),
            ).detach()

        if state_device is None:
            t_dev = time.perf_counter()
            state_device = state_cpu.to(device=self.device)
            stats["iforward/biggs/assignment_to_device_ms"] = float((time.perf_counter() - t_dev) * 1000.0)

        self._stage2_0_biggs_cache_put(key=key, state_cpu=state_cpu, state_device=state_device)
        stats["iforward/biggs/assignment_cache_size"] = float(len(getattr(self, "_stage2_0_biggs_assignment_cache", {}) or {}))
        return state_cpu.detach(), state_device, stats

    def _stage2_0_project_branch(
        self,
        *,
        branch: Any,
        assignment: Any,
        branch_name: str,
    ) -> BigGSParentProjection:
        projector_cfg = dict(getattr(self, "stage2_0_biggs_projector_cfg", {}) or {})
        max_scale = self._stage2_0_biggs_max_scale(str(branch_name))
        projector_cfg["tau_parent_scale"] = self._stage2_0_biggs_tau_parent_scale(str(branch_name))
        with self._iforward_amp_fp32():
            return project_biggs_parents(
                branch=branch,
                assignment=assignment,
                cfg=projector_cfg,
                max_scale=max_scale,
            )

    def _stage2_0_project_active_rigid_parents(self, **kwargs: Any) -> BigGSParentProjection:
        with self._iforward_amp_fp32():
            return project_biggs_active_rigid_parents(**kwargs)

    def _stage2_0_init_parent_branch_runtime(self, **kwargs: Any) -> Any:
        with self._iforward_amp_fp32():
            return init_parent_branch_runtime(**kwargs)

    def _stage2_0_update_parent_branch_runtime(self, **kwargs: Any) -> Any:
        with self._iforward_amp_fp32():
            return update_parent_branch_runtime(**kwargs)

    def _stage2_0_biggs_max_scale(self, branch_name: str) -> float:
        projector_cfg = getattr(self, "stage2_0_biggs_projector_cfg", {}) or {}
        default_max = {"bg": 1.5, "distant": 10.0, "rigid": 1.0}.get(str(branch_name), 10.0)
        return float(self._cfg_get(projector_cfg, f"max_scale_{branch_name}", default_max))

    def _stage2_0_biggs_tau_parent_scale(self, branch_name: str) -> float:
        projector_cfg = getattr(self, "stage2_0_biggs_projector_cfg", {}) or {}
        default_tau = {"bg": 1.0, "distant": 1.0, "rigid": 1.0}.get(str(branch_name), 1.0)
        return float(self._cfg_get(projector_cfg, f"tau_parent_scale_{branch_name}", self._cfg_get(projector_cfg, "tau_parent_scale", default_tau)))

    @staticmethod
    def _stage2_0_scene_parts_from_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "means": params["means"],
            "scales": torch.exp(params["scales_log"]),
            "quats": params["quats"],
            "opacities": torch.sigmoid(params["opacity_logit"]).reshape(-1),
            "colors": torch.cat([params["sh_dc"][:, None, :], params["sh_rest"]], dim=1),
        }

    @staticmethod
    def _stage2_0_cat_scene(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not parts:
            raise ValueError("BigGS parent scene requires at least one part")
        return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0].keys()}

    def _stage2_0_dino_cache_key(
        self,
        *,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_views: List[Any],
        source_images: List[torch.Tensor],
        source_frame_idx: int,
        height: int,
        width: int,
    ) -> Optional[tuple[Any, ...]]:
        if getattr(self, "dino_feature_cache", None) is None:
            return None
        extractor = getattr(self, "image_feature_extractor", None)
        if extractor is None or not hasattr(extractor, "get_feature_resolution"):
            return None
        scene_id, segment_id, _episode_id = self._stage2_0_biggs_ids_from_batch(batch)
        feature_hw = extractor.get_feature_resolution(int(height), int(width))
        fingerprint = (
            extractor.dino_adapter.fingerprint()  # type: ignore[attr-defined]
            if hasattr(extractor, "dino_adapter") and hasattr(extractor.dino_adapter, "fingerprint")
            else str(type(extractor))
        )

        def view_key(view: Any) -> Any:
            for name in ("camera_id", "cam_id", "image_id", "frame_id", "uid", "id"):
                value = getattr(view, name, None)
                if isinstance(value, (str, int)):
                    return (name, value)
            return str(type(view))

        image_hw = tuple(tuple(int(x) for x in spatial_hw_from_image_tensor(img)) for img in source_images)
        return (
            f"stage2_0_biggs_dino_{str(getattr(self, 'dino_feature_cache_level', 'adapter_output'))}",
            int(scene_id),
            int(segment_id),
            int(source_frame_idx),
            tuple(int(x) for x in source_indices),
            tuple(view_key(view) for view in source_views),
            image_hw,
            (int(height), int(width)),
            (int(feature_hw[0]), int(feature_hw[1])),
            fingerprint,
        )

    def _stage2_0_fine_scene_from_state(
        self,
        *,
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
        route: Any,
    ) -> Dict[str, torch.Tensor]:
        gaussians_bg_distant, _, _ = self._prepare_gaussians_bg_distant(bg, distant)
        means = [gaussians_bg_distant["means"]]
        scales = [gaussians_bg_distant["scales"]]
        quats = [gaussians_bg_distant["quats"]]
        opacities = [gaussians_bg_distant["opacities"]]
        colors = [gaussians_bg_distant["colors"]]
        if rigid is not None and int(route.S.numel()) > 0:
            s = route.S.long()
            means.append(route.means_world_S)
            quats.append(route.quats_world_S)
            scales.append(torch.exp(rigid.scales_log.index_select(0, s)))
            opacities.append(torch.sigmoid(rigid.opacity_logit.index_select(0, s)).squeeze(-1))
            colors.append(torch.cat([rigid.sh_dc.index_select(0, s)[:, None, :], rigid.sh_rest.index_select(0, s)], dim=1))
        return {
            "means": torch.cat(means, dim=0),
            "scales": torch.cat(scales, dim=0),
            "quats": torch.cat(quats, dim=0),
            "opacities": torch.cat(opacities, dim=0),
            "colors": torch.cat(colors, dim=0),
        }

    @staticmethod
    def _stage2_0_empty_projection_like(ref: torch.Tensor, *, sh_rest_bases: int) -> BigGSParentProjection:
        params = {
            "means": ref.new_zeros((0, 3)),
            "scales_log": ref.new_zeros((0, 3)),
            "quats": ref.new_zeros((0, 4)),
            "opacity_logit": ref.new_zeros((0, 1)),
            "sh_dc": ref.new_zeros((0, 3)),
            "sh_rest": ref.new_zeros((0, int(sh_rest_bases), 3)),
        }
        return BigGSParentProjection(params=params, child_mass_sum=ref.new_zeros((0,)), child_mass_mean=ref.new_zeros((0,)))

    def _stage2_0_fwhr_child_to_parent_global(
        self,
        *,
        state: IForwardBigGSState,
        active_rigid: Any,
        num_bg: int,
        num_distant: int,
        num_rigid_s: int,
        num_parent_bg: int,
        num_parent_distant: int,
    ) -> torch.Tensor:
        if state.bg is None:
            raise RuntimeError("FW-HR requires bg assignment")
        parts = [state.bg.child_to_parent.long()]
        if int(num_distant) > 0:
            if state.distant is None:
                raise RuntimeError("FW-HR distant points require distant assignment")
            parts.append(state.distant.child_to_parent.long() + int(num_parent_bg))
        if int(num_rigid_s) > 0:
            if active_rigid is None:
                raise RuntimeError("FW-HR rigid points require active rigid assignment")
            rigid_parent = active_rigid.child_to_active_parent_S.long() + int(num_parent_bg + num_parent_distant)
            if int(rigid_parent.numel()) != int(num_rigid_s):
                raise RuntimeError(
                    f"FW-HR rigid active assignment row mismatch: {int(rigid_parent.numel())} vs {int(num_rigid_s)}"
                )
            parts.append(rigid_parent)
        out = torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.long, device=self.device)
        expected = int(num_bg + num_distant + num_rigid_s)
        if int(out.numel()) != expected:
            raise RuntimeError(f"FW-HR child_to_parent row mismatch: {int(out.numel())} vs {expected}")
        return out

    @staticmethod
    def _stage3_0_branch_params(branch: Any) -> Dict[str, torch.Tensor]:
        return {
            "means": branch.means,
            "scales_log": branch.scales_log,
            "quats": branch.quats,
            "opacity_logit": branch.opacity_logit,
            "sh_dc": branch.sh_dc,
            "sh_rest": branch.sh_rest,
        }

    def _stage3_0_optimizer_prior(
        self,
        *,
        parent_optimizer_state: Optional[Any],
        branch: str,
        rows: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        out = ref.new_zeros((int(rows), 4))
        if parent_optimizer_state is None or int(rows) == 0:
            return out
        branch_state = getattr(parent_optimizer_state, str(branch), None)
        dense = getattr(branch_state, "dense", None) if branch_state is not None else None
        if dense is None:
            return out
        seen = getattr(dense, "seen", None)
        update_count = getattr(dense, "update_count", None)
        last_visit_step = getattr(dense, "last_visit_step", None)
        last_visit_kind = getattr(dense, "last_visit_kind", None)
        if not torch.is_tensor(seen) or int(seen.numel()) == 0:
            return out
        n = min(int(rows), int(seen.numel()))
        out[:n, 0] = seen[:n].to(device=ref.device, dtype=ref.dtype).reshape(-1)
        if torch.is_tensor(update_count) and int(update_count.numel()) >= n:
            out[:n, 1] = torch.log1p(update_count[:n].to(device=ref.device, dtype=ref.dtype).clamp_min(0.0))
        if torch.is_tensor(last_visit_step) and int(last_visit_step.numel()) >= n:
            out[:n, 2] = last_visit_step[:n].to(device=ref.device, dtype=ref.dtype).clamp_min(0.0) / 1000.0
        if torch.is_tensor(last_visit_kind) and int(last_visit_kind.numel()) >= n:
            out[:n, 3] = last_visit_kind[:n].to(device=ref.device, dtype=ref.dtype).clamp_min(0.0) / 10.0
        return out

    def _stage3_0_build_anchor_stats(
        self,
        *,
        fine_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        source_pair_valid_mask: Optional[torch.Tensor],
        child_to_parent_global: torch.Tensor,
        num_children: int,
        num_parents: int,
        height: int,
        width: int,
    ) -> Tuple[Any, Dict[str, float]]:
        extractor = getattr(self, "alpha_t_extractor_v4", None) or getattr(self, "alpha_t_extractor_v3", None)
        builder = getattr(extractor, "_build_multi_camera_meta_from_views", None)
        if not callable(builder):
            raise RuntimeError("Stage3_0 requires alpha_t_extractor_v3/v4 meta builder.")
        t0 = time.perf_counter()
        meta, meta_stats = builder(
            gaussians=fine_scene,
            cameras=source_views,
            height=int(height),
            width=int(width),
        )
        anchor_cfg = self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "scalar_anchor", {}) or {}
        threshold_cfg = self._cfg_get(anchor_cfg, "support_threshold", {}) or {}
        threshold_values = [
            float(value)
            for key, value in dict(threshold_cfg or {}).items()
            if str(key) not in {"child", "parent"} and isinstance(value, (int, float))
        ]
        default_threshold = min(threshold_values) if threshold_values else 1.0e-4
        child_threshold = float(self._cfg_get(threshold_cfg, "child", default_threshold))
        parent_threshold = float(self._cfg_get(threshold_cfg, "parent", child_threshold))
        detach_geometry = bool(
            self._cfg_get(
                anchor_cfg,
                "detach_geometry",
                self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "detach_geometry", True),
            )
        )
        backend = str(
            self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "scalar_anchor_backend", "cuda_scalar_anchor")
        ).lower()
        anchor_aux: Dict[str, float] = {}
        if backend == "cuda_scalar_anchor":
            anchor_weight_threshold = float(self._cfg_get(anchor_cfg, "weight_threshold", 0.0))
            global_step = int(getattr(self, "stage3_0_global_step", 0))
            heavy_aux_interval = int(self._cfg_get(anchor_cfg, "heavy_aux_interval", 100))
            emit_heavy_aux = bool(heavy_aux_interval > 0 and global_step % int(heavy_aux_interval) == 0)
            cuda_event_timing = bool(self._cfg_get(anchor_cfg, "cuda_event_timing", False))
            count_pairs = bool(self._cfg_get(anchor_cfg, "count_pairs", False))
            requested_anchor_mode = str(self._cfg_get(anchor_cfg, "anchor_mode", "auto")).lower()
            if requested_anchor_mode not in {"auto", "full", "fast_uv_support"}:
                raise ValueError(f"unsupported Stage3_0 scalar_anchor.anchor_mode={requested_anchor_mode!r}")
            parent_lifting_type = str(getattr(self, "stage3_parent_lifting_type", "legacy_direct_lift")).lower()
            parent_legacy_direct = parent_lifting_type == "legacy_direct_lift"
            parent_gather = getattr(self, "stage3_parent_gather", None)
            fast_anchor_allowed = (
                bool(parent_legacy_direct)
                or (
                    parent_gather is not None
                    and bool(parent_gather.use_fixed_center_fast_path(global_step))
                    and not bool(parent_gather.config.fixed_center_use_geometry_pe)
                )
            )
            if requested_anchor_mode == "auto":
                anchor_mode = "fast_uv_support" if bool(fast_anchor_allowed) else "full"
            elif requested_anchor_mode == "fast_uv_support":
                if not bool(fast_anchor_allowed):
                    raise RuntimeError(
                        "Stage3 fast_uv_support scalar anchor is only valid while parent gather is fixed-center "
                        "with fixed_center_use_geometry_pe=false. Child gather is always support_center."
                    )
                anchor_mode = "fast_uv_support"
            else:
                if bool(parent_legacy_direct):
                    raise RuntimeError("Stage3 legacy_direct_lift parent requires scalar_anchor.anchor_mode=auto or fast_uv_support.")
                anchor_mode = "full"
            anchor, anchor_aux = build_cuda_scalar_anchor_stats(
                meta=meta,
                child_to_parent=child_to_parent_global,
                num_children=int(num_children),
                num_parents=int(num_parents),
                num_views=int(len(source_views)),
                image_height=int(height),
                image_width=int(width),
                source_pair_valid_mask=source_pair_valid_mask,
                child_support_threshold=child_threshold,
                parent_support_threshold=parent_threshold,
                weight_threshold=anchor_weight_threshold,
                anchor_mode=anchor_mode,
                count_pairs=count_pairs,
                child_only=bool(parent_legacy_direct),
                detach_geometry=detach_geometry,
                emit_heavy_aux=emit_heavy_aux,
                use_cuda_event_timing=cuda_event_timing,
                return_aux=True,
            )
        elif backend == "projected_meta":
            anchor = build_projected_meta_anchor_stats(
                meta=meta,
                child_to_parent=child_to_parent_global,
                num_children=int(num_children),
                num_parents=int(num_parents),
                num_views=int(len(source_views)),
                image_height=int(height),
                image_width=int(width),
                source_pair_valid_mask=source_pair_valid_mask,
                child_support_threshold=child_threshold,
                parent_support_threshold=parent_threshold,
                detach_geometry=detach_geometry,
            )
            anchor_aux = {
                "iforward/stage3/anchor_backend_id": 0.0,
                "iforward/stage3/anchor_mode_id": 0.0,
                "iforward/stage3/anchor_fast_uv_support_enabled": 0.0,
                "iforward/stage3/anchor_parent_aggregate_backend_id": 0.0,
                "iforward/stage3/anchor_parent_aggregate_cuda_enabled": 0.0,
                "iforward/stage3/anchor_heavy_aux_enabled": 0.0,
                "iforward/stage3/anchor_pair_count_enabled": 0.0,
                "iforward/stage3/anchor_cuda_ms": 0.0,
                "iforward/stage3/anchor_cuda_event_ms": 0.0,
                "iforward/stage3/anchor_normalize_ms": 0.0,
                "iforward/stage3/anchor_pair_count_total": 0.0,
                "iforward/stage3/anchor_pair_count_threshold": 0.0,
            }
        else:
            raise ValueError(f"unsupported Stage3_0 scalar_anchor_backend={backend!r}")
        stats = {
            "iforward/stage3/scalar_anchor_ms": float((time.perf_counter() - t0) * 1000.0),
            "iforward/stage3/anchor_nnz": float(int(meta.get("means2d", torch.zeros((0, 2))).shape[0])),
            **anchor_aux,
            "iforward/stage3/child_support_valid_ratio": (
                float(anchor.child_valid.detach().float().any(dim=1).float().mean().item())
                if int(anchor.child_valid.numel()) > 0
                else 0.0
            ),
            "iforward/stage3/parent_support_valid_ratio": (
                float(anchor.parent_valid.detach().float().any(dim=1).float().mean().item())
                if int(anchor.parent_valid.numel()) > 0
                else 0.0
            ),
        }
        for key, value in dict(meta_stats or {}).items():
            if isinstance(value, (int, float)):
                stats[f"iforward/stage3/meta_{key}"] = float(value)
        return anchor, stats

    def _iforward_amp_autocast(self) -> Any:
        policy = getattr(self, "iforward_amp_policy", None)
        return policy.autocast() if policy is not None else nullcontext()

    def _iforward_amp_fp32(self) -> Any:
        policy = getattr(self, "iforward_amp_policy", None)
        return policy.fp32() if policy is not None else nullcontext()

    def _stage3_amp_cfg(self) -> Any:
        training_cfg = self._cfg_get(getattr(self, "config", {}) or {}, "training", {}) or {}
        amp_cfg = self._cfg_get(training_cfg, "amp", {}) or {}
        return self._cfg_get(amp_cfg, "stage3", {}) or {}

    def _stage3_storage_cfg(self) -> Any:
        training_cfg = self._cfg_get(getattr(self, "config", {}) or {}, "training", {}) or {}
        amp_cfg = self._cfg_get(training_cfg, "amp", {}) or {}
        return self._cfg_get(amp_cfg, "storage", {}) or {}

    @staticmethod
    def _dtype_from_name(name: Any, *, default: torch.dtype) -> torch.dtype:
        label = str(name).strip().lower()
        if label in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if label in {"fp16", "float16", "half", "16"}:
            return torch.float16
        if label in {"fp32", "float32", "32", "none", "off", "false"}:
            return torch.float32
        if label == "amp":
            return default
        return default

    def _storage_dtype_from_name(self, name: Any, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        return storage_dtype_from_name(
            name,
            amp_dtype=getattr(policy, "dtype", None),
            default=ref.dtype,
        )

    def _stage3_features_2d_cache_dtype(self, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        name = self._cfg_get(
            self._stage3_storage_cfg(),
            "features_2d_cache_dtype",
            getattr(policy, "features_2d_cache_dtype", "fp32"),
        )
        return self._storage_dtype_from_name(name, ref)

    def _stage3_parent_context_cache_dtype(self, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        name = self._cfg_get(
            self._stage3_storage_cfg(),
            "parent_context_cache_dtype",
            getattr(policy, "parent_context_cache_dtype", "fp32"),
        )
        return self._storage_dtype_from_name(name, ref)

    def _stage3_cast_feature_cache(self, cnn_inputs: Dict[str, Any]) -> Dict[str, Any]:
        features = cnn_inputs.get("features_2d")
        if not torch.is_tensor(features):
            return cnn_inputs
        dtype = self._stage3_features_2d_cache_dtype(features)
        if features.dtype == dtype:
            return cnn_inputs
        out = dict(cnn_inputs)
        out["features_2d"] = features.to(dtype=dtype)
        return out

    def _stage3_parent_lift_amp_enabled(self) -> bool:
        policy = getattr(self, "iforward_amp_policy", None)
        return bool(
            policy is not None
            and bool(getattr(policy, "enabled", False))
            and getattr(policy, "dtype", None) is not None
            and bool(getattr(policy, "parent_lift_amp", False))
        )

    def _stage3_child_gather_amp_enabled(self) -> bool:
        policy = getattr(self, "iforward_amp_policy", None)
        return bool(
            policy is not None
            and bool(getattr(policy, "enabled", False))
            and getattr(policy, "dtype", None) is not None
            and bool(getattr(policy, "child_gather_amp", False))
        )

    def _stage3_parent_lift_dtype(self, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        if not self._stage3_parent_lift_amp_enabled() or policy is None:
            return torch.float32
        default = getattr(policy, "dtype", None) or ref.dtype
        name = self._cfg_get(self._stage3_amp_cfg(), "parent_lift_dtype", "amp")
        return self._dtype_from_name(name, default=default)

    def _stage3_child_detail_dtype(self, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        cfg = self._stage3_amp_cfg()
        name = self._cfg_get(cfg, "child_detail_output_dtype", getattr(policy, "child_detail_output_dtype", "fp32"))
        if str(name).strip().lower() in {"fp32", "float32", "32", "none", "off", "false"}:
            return torch.float32
        if not self._stage3_child_gather_amp_enabled() or policy is None:
            return torch.float32
        default = getattr(policy, "dtype", None) or ref.dtype
        return self._dtype_from_name(name, default=default)

    def _stage3_child_detail_output_dtype(self, ref: torch.Tensor) -> torch.dtype:
        policy = getattr(self, "iforward_amp_policy", None)
        cfg = self._stage3_amp_cfg()
        name = self._cfg_get(cfg, "child_detail_output_dtype", getattr(policy, "child_detail_output_dtype", "fp32"))
        return self._storage_dtype_from_name(name, ref)

    def _stage3_0_parent_sparse_gather(
        self,
        *,
        cnn_inputs: Dict[str, Any],
        anchor_stats: Any,
        bg_proj: BigGSParentProjection,
        distant_proj: Optional[BigGSParentProjection],
        rigid_proj_active: Optional[BigGSParentProjection],
        parent_optimizer_state: Optional[Any],
        height: int,
        width: int,
        num_parent_bg: int,
        num_parent_distant: int,
        num_parent_rigid: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        parent_query = getattr(self, "stage3_parent_query", None)
        parent_gather = getattr(self, "stage3_parent_gather", None)
        if parent_query is None or parent_gather is None:
            raise RuntimeError("Stage3_0 parent gather modules are not initialized.")
        context_2d = cnn_inputs["features_2d"]
        if context_2d.dim() != 4:
            raise ValueError(f"Stage3_0 context_2d must be [V,H,W,C], got {tuple(context_2d.shape)}")
        stats: Dict[str, float] = {
            "iforward/stage3/parent_lift_amp_enabled": 1.0 if self._stage3_parent_lift_amp_enabled() else 0.0,
            "iforward/stage3/parent_lift_dtype_id": float(amp_dtype_id(context_2d.dtype)),
        }
        obs2d_map = None
        obs2d_module = getattr(self, "stage3_parent_query_obs2d", None)
        if bool(getattr(self, "stage3_parent_query_use_obs2d_lift", False)):
            if obs2d_module is None:
                raise RuntimeError("Stage3_0 parent query obs2d lift module is not initialized.")
            obs2d_map = obs2d_module(context_2d)
        dino_native_2d = None
        needs_dino_native = bool(getattr(self, "stage3_parent_query_use_dino_native_lift", False)) or bool(
            getattr(self, "stage3_parent_context_use_dino_native_fusion", False)
        )
        if bool(needs_dino_native):
            dino_native_2d = cnn_inputs.get("stage3_dino_native_2d")
            if not torch.is_tensor(dino_native_2d):
                raise RuntimeError("Stage3_0 parent DINO native lift requires cnn_inputs['stage3_dino_native_2d'].")
            if dino_native_2d.dim() != 4 or int(dino_native_2d.shape[0]) != int(context_2d.shape[0]):
                raise ValueError(f"stage3_dino_native_2d must be [V,Hd,Wd,C], got {tuple(dino_native_2d.shape)}")
            dino_native_2d = dino_native_2d.to(device=context_2d.device, dtype=context_2d.dtype).detach()
            if int(dino_native_2d.shape[-1]) != int(getattr(self, "stage3_dino_native_dim", int(dino_native_2d.shape[-1]))):
                raise ValueError("Stage3_0 DINO native feature channel mismatch.")
        gather_cfg = self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "parent_gather", {}) or {}
        prepared_context = prepare_value_nchw(context_2d) if parent_gather._should_prepare_value(context_2d) else None
        gather_aux_interval = int(
            self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "gather_aux_interval", 100)
        )
        scalar_cfg = self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "scalar_anchor", {}) or {}
        threshold_cfg = self._cfg_get(scalar_cfg, "support_threshold", {}) or {}
        valid_row_filter = bool(self._cfg_get(gather_cfg, "valid_row_filter", True))
        query_chunked = bool(self._cfg_get(gather_cfg, "query_chunked", True))
        parent_threshold = float(
            self._cfg_get(
                threshold_cfg,
                "parent",
                min(
                    [
                        float(v)
                        for k, v in dict(threshold_cfg or {}).items()
                        if str(k) not in {"child"} and isinstance(v, (int, float))
                    ]
                    or [1.0e-4]
                ),
            )
        )
        global_step = int(getattr(self, "stage3_0_global_step", 0))
        emit_gather_heavy_aux = bool(gather_aux_interval > 0 and global_step % int(gather_aux_interval) == 0)
        queries = []
        start = 0
        regs: Dict[str, torch.Tensor] = {}

        def _run_branch(name: str, branch_id: int, params: Dict[str, torch.Tensor], rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal start, stats, regs
            end = start + int(rows)
            feat = context_2d.new_zeros((int(rows), int(context_2d.shape[-1])))
            conf = context_2d.new_zeros((int(rows),))
            if int(rows) == 0:
                start = end
                return feat, conf
            prior = self._stage3_0_optimizer_prior(
                parent_optimizer_state=parent_optimizer_state,
                branch=name,
                rows=int(rows),
                ref=context_2d,
            )
            support_total = anchor_stats.parent_support_total[start:end]
            row_valid = anchor_stats.parent_valid[start:end].any(dim=1)
            if bool(valid_row_filter):
                row_valid = row_valid & (support_total.to(device=row_valid.device) >= float(parent_threshold))
            row_idx = torch.nonzero(row_valid, as_tuple=False).reshape(-1).to(device=context_2d.device, dtype=torch.long)
            stats[f"iforward/stage3/parent_{name}_rows_total"] = float(int(rows))
            stats[f"iforward/stage3/parent_{name}_rows_valid"] = float(int(row_idx.numel()))
            stats[f"iforward/stage3/parent_{name}_rows_valid_ratio"] = float(int(row_idx.numel())) / float(max(int(rows), 1))
            chunk_limit = parent_gather.effective_chunk_size(global_step, rows=int(row_idx.numel()))
            if not bool(query_chunked):
                chunk_limit = max(int(row_idx.numel()), 1)
            stats[f"iforward/stage3/parent_{name}_chunk_size"] = float(chunk_limit)
            stats[f"iforward/stage3/parent_{name}_num_chunks"] = float(
                math.ceil(float(int(row_idx.numel())) / float(chunk_limit)) if int(row_idx.numel()) > 0 else 0
            )
            aux_sum: Dict[str, float] = {}
            aux_rows = 0
            reg_chunks = []

            def _slice_params(row_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
                out_params: Dict[str, torch.Tensor] = {}
                for key, value in params.items():
                    if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == int(rows):
                        out_params[str(key)] = value.index_select(0, row_ids.to(device=value.device, dtype=torch.long))
                    else:
                        out_params[str(key)] = value
                return out_params

            def _merge_aux(weight: int, aux_items: Dict[str, float]) -> None:
                for key, value in aux_items.items():
                    aux_sum[str(key)] = aux_sum.get(str(key), 0.0) + float(value) * float(weight)

            for cstart in range(0, int(row_idx.numel()), chunk_limit):
                cidx = row_idx[cstart : cstart + chunk_limit]
                if int(cidx.numel()) == 0:
                    continue
                anchor_uv_c = anchor_stats.parent_uv[start:end].index_select(
                    0, cidx.to(device=anchor_stats.parent_uv.device, dtype=torch.long)
                )
                support_c = anchor_stats.parent_support[start:end].index_select(
                    0, cidx.to(device=anchor_stats.parent_support.device, dtype=torch.long)
                )
                valid_c = anchor_stats.parent_valid[start:end].index_select(
                    0, cidx.to(device=anchor_stats.parent_valid.device, dtype=torch.long)
                )
                rows_c = int(cidx.numel())
                obs2d_lift_c = None
                dino_lift_c = None
                if obs2d_map is not None and not parent_gather.use_fixed_center_fast_path(global_step):
                    obs2d_lift_c, _obs_conf, obs_aux = support_center_sparse_gather(
                        value_map=obs2d_map,
                        anchor_uv=anchor_uv_c,
                        support=support_c,
                        valid=valid_c,
                        image_height=int(height),
                        image_width=int(width),
                        backend=str(parent_gather.config.backend).lower(),
                        chunk_size=int(chunk_limit),
                        emit_heavy_aux=emit_gather_heavy_aux,
                        prefix=f"stage3/parent_{name}_obs2d_query_lift",
                    )
                    _merge_aux(rows_c, obs_aux)
                if dino_native_2d is not None:
                    dino_lift_c, _dino_conf, dino_aux = support_center_sparse_gather(
                        value_map=dino_native_2d,
                        anchor_uv=anchor_uv_c,
                        support=support_c,
                        valid=valid_c,
                        image_height=int(height),
                        image_width=int(width),
                        backend=str(parent_gather.config.backend).lower(),
                        chunk_size=int(chunk_limit),
                        emit_heavy_aux=emit_gather_heavy_aux,
                        prefix=f"stage3/parent_{name}_dino_native_lift",
                    )
                    _merge_aux(rows_c, dino_aux)
                if parent_gather.use_fixed_center_fast_path(global_step):
                    q = None
                else:
                    q = parent_query(
                        params=_slice_params(cidx),
                        support_total=support_total.index_select(0, cidx.to(device=support_total.device, dtype=torch.long)),
                        branch_id=int(branch_id),
                        optimizer_prior=prior.index_select(0, cidx.to(device=prior.device, dtype=torch.long)),
                        obs2d_lift=obs2d_lift_c,
                        dino_lift=dino_lift_c if bool(getattr(self, "stage3_parent_query_use_dino_native_lift", False)) else None,
                    )
                out_c, conf_c, aux_c, reg_c = parent_gather(
                    value_map=context_2d,
                    anchor_uv=anchor_uv_c,
                    support=support_c,
                    valid=valid_c,
                    depth=anchor_stats.parent_depth[start:end].index_select(
                        0, cidx.to(device=anchor_stats.parent_depth.device, dtype=torch.long)
                    ),
                    radius=anchor_stats.parent_radius[start:end].index_select(
                        0, cidx.to(device=anchor_stats.parent_radius.device, dtype=torch.long)
                    ),
                    image_height=int(height),
                    image_width=int(width),
                    query=q,
                    prepared_value_nchw=prepared_context,
                    global_step=global_step,
                    emit_heavy_aux=emit_gather_heavy_aux,
                    prefix=f"stage3/parent_{name}",
                )
                if bool(getattr(self, "stage3_parent_context_use_dino_native_fusion", False)):
                    fusion = getattr(self, "stage3_parent_context_fusion", None)
                    if fusion is None or dino_lift_c is None:
                        raise RuntimeError("Stage3_0 parent context DINO fusion is not initialized.")
                    out_c = fusion(out_c, dino_lift_c)
                out_c = out_c.to(device=feat.device, dtype=feat.dtype)
                conf_c = conf_c.to(device=conf.device, dtype=conf.dtype)
                feat.index_copy_(0, cidx, out_c)
                conf.index_copy_(0, cidx, conf_c)
                aux_rows += rows_c
                _merge_aux(rows_c, aux_c)
                reg_chunks.append(reg_c)
            if aux_rows > 0:
                stats.update({key: float(value) / float(aux_rows) for key, value in aux_sum.items()})
            regs = merge_stage3_reg_terms(regs, merge_stage3_reg_terms(*reg_chunks))
            start = end
            return feat, conf

        feat_bg, conf_bg = _run_branch("bg", 0, bg_proj.params, int(num_parent_bg))
        queries.append((feat_bg, conf_bg))
        if int(num_parent_distant) > 0:
            if distant_proj is None:
                raise RuntimeError("Stage3_0 distant parent gather expected distant projection")
            queries.append(_run_branch("distant", 1, distant_proj.params, int(num_parent_distant)))
        if int(num_parent_rigid) > 0:
            if rigid_proj_active is None:
                raise RuntimeError("Stage3_0 rigid parent gather expected active rigid projection")
            queries.append(_run_branch("rigid", 2, rigid_proj_active.params, int(num_parent_rigid)))
        feat_all = torch.cat([x[0] for x in queries], dim=0) if queries else context_2d.new_zeros((0, int(context_2d.shape[-1])))
        conf_all = torch.cat([x[1] for x in queries], dim=0) if queries else context_2d.new_zeros((0,))
        if bool(emit_gather_heavy_aux):
            stats["iforward/stage3/parent_context_rms"] = (
                float(feat_all.detach().float().square().mean().sqrt().item()) if int(feat_all.numel()) else 0.0
            )
            stats["iforward/stage3/parent_confidence_mean"] = (
                float(conf_all.detach().float().mean().item()) if int(conf_all.numel()) else 0.0
            )
        return feat_all, anchor_stats.parent_support_total.to(device=feat_all.device, dtype=feat_all.dtype), stats, regs

    def _stage3_0_gather_child_detail(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> None:
        if not bool(getattr(self, "stage3_0_enabled", False)):
            return
        anchor = measurement.get("stage3_anchor_stats")
        detail_2d = measurement.get("stage3_detail_2d")
        if anchor is None or not torch.is_tensor(detail_2d):
            raise RuntimeError("Stage3_0 child gather requires anchor stats and detail_2d in measurement.")
        child_detail_dtype = self._stage3_child_detail_dtype(detail_2d)
        detail_2d = detail_2d.to(dtype=child_detail_dtype)
        measurement["stage3_detail_2d"] = detail_2d
        height = int(measurement["stage3_image_height"])
        width = int(measurement["stage3_image_width"])
        gather_aux_interval = int(
            self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "gather_aux_interval", 100)
        )
        child_to_parent_global = measurement["stage3_child_to_parent_global"].to(device=detail_2d.device, dtype=torch.long)
        num_bg = int(measurement.get("num_bg", 0))
        num_distant = int(measurement.get("num_distant", 0))
        num_rigid = int(measurement.get("num_rigid_S", 0))
        aux: Dict[str, float] = {}
        reg_items = [measurement.get("stage3_gather_reg_terms", {})]
        child_gather_cfg = self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "child_gather", {}) or {}
        child_type = str(self._cfg_get(child_gather_cfg, "type", "support_center")).lower()
        if child_type != "support_center":
            raise ValueError("Stage3_0 child gather only supports type=support_center")
        center_by_parent = bool(self._cfg_get(child_gather_cfg, "center_by_parent", True))
        valid_row_filter = bool(self._cfg_get(child_gather_cfg, "valid_row_filter", True))
        backend = str(self._cfg_get(child_gather_cfg, "backend", "auto")).lower()
        chunk_limit_cfg = int(self._cfg_get(child_gather_cfg, "fixed_center_chunk_size", self._cfg_get(child_gather_cfg, "chunk_size", 65536)))
        if chunk_limit_cfg <= 0:
            chunk_limit_cfg = 2**30
        child_threshold = float(
            self._cfg_get(
                self._cfg_get(self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "scalar_anchor", {}) or {}, "support_threshold", {}) or {},
                "child",
                1.0e-4,
            )
        )
        global_step = int(getattr(self, "stage3_0_global_step", 0))
        emit_gather_heavy_aux = bool(gather_aux_interval > 0 and global_step % int(gather_aux_interval) == 0)
        detach_child_detail_cfg = bool(self._cfg_get(child_gather_cfg, "detach_child_detail", False))
        train_child_detail_every_n = max(int(self._cfg_get(child_gather_cfg, "train_child_detail_every_n", 1)), 1)
        child_detail_train_enabled = (not bool(detach_child_detail_cfg)) and (
            int(train_child_detail_every_n) <= 1 or int(global_step) % int(train_child_detail_every_n) == 0
        )
        child_amp_enabled = self._stage3_child_gather_amp_enabled()
        child_detail_output_dtype = self._stage3_child_detail_output_dtype(detail_2d)
        prepared_detail = (
            prepare_value_nchw(detail_2d)
            if backend == "pytorch" or (backend == "auto" and not torch.is_tensor(detail_2d))
            else None
        )
        aux["iforward/stage3/child_support_center_enabled"] = 1.0
        aux["iforward/stage3/child_event_dependency_removed"] = 1.0
        aux["iforward/stage3/child_learned_path_enabled"] = 0.0
        aux["iforward/stage3/child_gather_amp_enabled"] = 1.0 if bool(child_amp_enabled) else 0.0
        aux["iforward/stage3/child_detail_gather_dtype_id"] = float(amp_dtype_id(detail_2d.dtype))
        aux["iforward/stage3/child_detail_output_dtype_id"] = float(amp_dtype_id(child_detail_output_dtype))
        aux["amp/dtype/child_detail"] = float(amp_dtype_id(child_detail_output_dtype))
        aux["iforward/stage3/child_detail_detached"] = 0.0 if bool(child_detail_train_enabled) else 1.0
        aux["iforward/stage3/child_detail_train_every_n"] = float(train_child_detail_every_n)
        ignored_child_keys = {
            "query_dim",
            "offset_scale",
            "max_offset_px",
            "train_weights_steps",
            "offset_warmup_steps",
            "use_geometry_pe",
        }
        aux["iforward/stage3/child_ignored_learned_config"] = (
            1.0 if any(str(key) in child_gather_cfg for key in ignored_child_keys) else 0.0
        )

        def _run(
            *,
            name: str,
            start: int,
            rows: int,
            parent_id_local: torch.Tensor,
            num_parents: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal aux, reg_items
            ref = detail_2d
            if int(rows) == 0:
                return (
                    ref.new_zeros((0, int(detail_2d.shape[-1]))),
                    torch.zeros((0,), device=ref.device, dtype=torch.bool),
                    ref.new_zeros((0,)),
                )
            end = int(start) + int(rows)
            support_total = anchor.child_support_total[start:end]
            detail = ref.new_zeros((int(rows), int(detail_2d.shape[-1])))
            confidence = ref.new_zeros((int(rows),))
            row_valid = anchor.child_valid[start:end].any(dim=1)
            if bool(valid_row_filter):
                row_valid = row_valid & (support_total.to(device=row_valid.device) >= float(child_threshold))
            row_idx = torch.nonzero(row_valid, as_tuple=False).reshape(-1).to(device=detail_2d.device, dtype=torch.long)
            aux[f"iforward/stage3/child_{name}_rows_total"] = float(int(rows))
            aux[f"iforward/stage3/child_{name}_rows_valid"] = float(int(row_idx.numel()))
            aux[f"iforward/stage3/child_{name}_rows_valid_ratio"] = float(int(row_idx.numel())) / float(max(int(rows), 1))
            chunk_limit = min(int(chunk_limit_cfg), max(int(row_idx.numel()), 1))
            aux[f"iforward/stage3/child_{name}_chunk_size"] = float(chunk_limit)
            aux[f"iforward/stage3/child_{name}_num_chunks"] = float(
                math.ceil(float(int(row_idx.numel())) / float(chunk_limit)) if int(row_idx.numel()) > 0 else 0
            )
            aux_sum: Dict[str, float] = {}
            aux_rows = 0

            for cstart in range(0, int(row_idx.numel()), chunk_limit):
                cidx = row_idx[cstart : cstart + chunk_limit]
                if int(cidx.numel()) == 0:
                    continue
                amp_ctx = self._iforward_amp_autocast() if bool(child_amp_enabled) else self._iforward_amp_fp32()
                with amp_ctx:
                    detail_c, confidence_c, gather_aux = support_center_sparse_gather(
                        value_map=detail_2d,
                        anchor_uv=anchor.child_uv[start:end].index_select(0, cidx.to(device=anchor.child_uv.device, dtype=torch.long)),
                        support=anchor.child_support[start:end].index_select(0, cidx.to(device=anchor.child_support.device, dtype=torch.long)),
                        valid=anchor.child_valid[start:end].index_select(0, cidx.to(device=anchor.child_valid.device, dtype=torch.long)),
                        image_height=int(height),
                        image_width=int(width),
                        backend=backend,
                        prepared_value_nchw=prepared_detail,
                        chunk_size=int(chunk_limit),
                        emit_heavy_aux=emit_gather_heavy_aux,
                        prefix=f"stage3/child_{name}",
                    )
                detail_c = detail_c.to(device=detail.device, dtype=detail.dtype)
                confidence_c = confidence_c.to(device=confidence.device, dtype=confidence.dtype)
                detail.index_copy_(0, cidx, detail_c)
                confidence.index_copy_(0, cidx, confidence_c)
                rows_c = int(cidx.numel())
                aux_rows += rows_c
                for key, value in gather_aux.items():
                    aux_sum[str(key)] = aux_sum.get(str(key), 0.0) + float(value) * float(rows_c)
            if aux_rows > 0:
                aux.update({key: float(value) / float(aux_rows) for key, value in aux_sum.items()})
            valid = (support_total.to(device=detail.device, dtype=detail.dtype) >= float(child_threshold)) & (confidence > 0.0)
            valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1) if bool(center_by_parent) else None
            if valid_idx is not None and int(valid_idx.numel()) > 0:
                centered_valid, center_err = center_child_detail_by_parent(
                    detail.index_select(0, valid_idx),
                    child_to_parent=parent_id_local.to(device=detail.device, dtype=torch.long).index_select(
                        0, valid_idx.to(device=parent_id_local.device)
                    ),
                    weights=confidence.index_select(0, valid_idx),
                    num_parents=int(num_parents),
                )
                detail.index_copy_(0, valid_idx, centered_valid)
                aux[f"iforward/stage3/child_{name}_centering_error"] = (
                    float(center_err.item()) if bool(emit_gather_heavy_aux) else 0.0
                )
            elif bool(center_by_parent):
                aux[f"iforward/stage3/child_{name}_centering_error"] = 0.0
            detail = torch.where(valid.unsqueeze(-1), detail, torch.zeros_like(detail))
            if bool(emit_gather_heavy_aux):
                aux[f"iforward/stage3/child_{name}_detail_rms"] = (
                    float(detail.detach().float().square().mean().sqrt().item()) if detail.numel() else 0.0
                )
                aux[f"iforward/stage3/child_{name}_valid_ratio"] = (
                    float(valid.detach().float().mean().item()) if valid.numel() else 0.0
                )
            return detail, valid, confidence

        cursor = 0
        detail_bg, valid_bg, support_bg = _run(
            name="bg",
            start=cursor,
            rows=num_bg,
            parent_id_local=measurement["assign_bg"].child_to_parent,
            num_parents=int(measurement["assign_bg"].num_parents),
        )
        cursor += num_bg
        detail_d = valid_d = support_d = None
        if num_distant > 0 and local_state.distant is not None:
            detail_d, valid_d, support_d = _run(
                name="distant",
                start=cursor,
                rows=num_distant,
                parent_id_local=measurement["assign_distant"].child_to_parent,
                num_parents=int(measurement["assign_distant"].num_parents),
            )
        cursor += num_distant
        detail_r = valid_r = support_r = None
        active = measurement.get("assign_rigid_active")
        if num_rigid > 0 and local_state.rigid is not None and active is not None:
            detail_r, valid_r, support_r = _run(
                name="rigid",
                start=cursor,
                rows=num_rigid,
                parent_id_local=active.child_to_active_parent_S,
                num_parents=int(active.active_parent_count.numel()),
            )
        detail_bg = detail_bg.to(dtype=child_detail_output_dtype)
        if torch.is_tensor(detail_d):
            detail_d = detail_d.to(dtype=child_detail_output_dtype)
        if torch.is_tensor(detail_r):
            detail_r = detail_r.to(dtype=child_detail_output_dtype)
        if not bool(child_detail_train_enabled):
            detail_bg = detail_bg.detach()
            if torch.is_tensor(detail_d):
                detail_d = detail_d.detach()
            if torch.is_tensor(detail_r):
                detail_r = detail_r.detach()
        measurement.update(
            {
                "child_detail_bg": detail_bg,
                "child_detail_distant": detail_d,
                "child_detail_rigid_S": detail_r,
                "child_detail_valid_bg": valid_bg,
                "child_detail_valid_distant": valid_d,
                "child_detail_valid_rigid_S": valid_r,
                "child_detail_support_bg": support_bg,
                "child_detail_support_distant": support_d,
                "child_detail_support_rigid_S": support_r,
                "stage3_gather_reg_terms": merge_stage3_reg_terms(*reg_items),
            }
        )
        if bool(emit_gather_heavy_aux):
            aux["iforward/stage3/child_detail_rms"] = float(
                torch.cat(
                    [
                        x.reshape(-1)
                        for x in (detail_bg, detail_d, detail_r)
                        if torch.is_tensor(x) and int(x.numel()) > 0
                    ],
                    dim=0,
                )
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
                .item()
            ) if any(torch.is_tensor(x) and int(x.numel()) > 0 for x in (detail_bg, detail_d, detail_r)) else 0.0
        measurement.update(aux)

    def _stage2_0_fwhr_lift_from_fine_scene(
        self,
        *,
        fine_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        cnn_inputs: Dict[str, Any],
        height: int,
        width: int,
        child_to_parent_global: torch.Tensor,
        num_parents: int,
        context_dim: int,
        detail_dim: int,
        num_bg: int,
        num_distant: int,
        num_rigid_s: int,
    ) -> Tuple[Any, Dict[str, float]]:
        detail_2d = cnn_inputs.get("fwhr_detail_2d")
        if detail_2d is None or not torch.is_tensor(detail_2d):
            raise RuntimeError("FW-HR lifting requires image_feature_extractor.forward_fwhr/detail output")
        context_2d = cnn_inputs["features_2d"]
        if context_2d.dim() != 4 or detail_2d.dim() != 4:
            raise ValueError("FW-HR context/detail feature maps must be [V,H,W,C]")
        if tuple(context_2d.shape[:3]) != tuple(detail_2d.shape[:3]):
            raise ValueError(
                "FW-HR context/detail spatial mismatch: "
                f"context={tuple(context_2d.shape)} detail={tuple(detail_2d.shape)}"
            )
        if int(context_2d.shape[-1]) != int(context_dim):
            raise ValueError(f"FW-HR context dim mismatch: got {int(context_2d.shape[-1])}, expected {context_dim}")
        if int(detail_2d.shape[-1]) != int(detail_dim):
            raise ValueError(f"FW-HR detail dim mismatch: got {int(detail_2d.shape[-1])}, expected {detail_dim}")
        feat_2d = torch.cat([context_2d, detail_2d.to(device=context_2d.device, dtype=context_2d.dtype)], dim=-1)
        child_feature_sum, child_weight_sum_feature, child_support = self._backproject_scene_features_multi_camera(
            gaussians_scene=fine_scene,
            source_views=source_views,
            features_2d=feat_2d,
            source_pair_valid_mask=cnn_inputs["source_pair_valid_mask"],
            height=int(height),
            width=int(width),
            return_debug_stats=bool(getattr(self, "stage2_0_biggs_return_debug_stats", True)),
            return_raw_lift=True,
        )
        if child_feature_sum is None or child_weight_sum_feature is None or child_support is None:
            raise RuntimeError("FW-HR fine lifting returned empty features")
        thresholds = []
        if int(num_bg) > 0:
            thresholds.append(
                torch.full(
                    (int(num_bg),),
                    float(getattr(self, "stage2_0_fwhr_detail_support_min", {}).get("bg", getattr(self, "bg_src_backproject_support_min", 0.0))),
                    device=child_feature_sum.device,
                    dtype=child_feature_sum.dtype,
                )
            )
        if int(num_distant) > 0:
            thresholds.append(
                torch.full(
                    (int(num_distant),),
                    float(
                        getattr(self, "stage2_0_fwhr_detail_support_min", {}).get(
                            "distant",
                            getattr(self, "distant_src_backproject_support_min", 0.0),
                        )
                    ),
                    device=child_feature_sum.device,
                    dtype=child_feature_sum.dtype,
                )
            )
        if int(num_rigid_s) > 0:
            thresholds.append(
                torch.full(
                    (int(num_rigid_s),),
                    float(
                        getattr(self, "stage2_0_fwhr_detail_support_min", {}).get(
                            "rigid",
                            getattr(self, "rigid_src_backproject_support_min", 0.0),
                        )
                    ),
                    device=child_feature_sum.device,
                    dtype=child_feature_sum.dtype,
                )
            )
        detail_valid_threshold = (
            torch.cat(thresholds, dim=0)
            if thresholds
            else torch.zeros((0,), device=child_feature_sum.device, dtype=child_feature_sum.dtype)
        )
        lift = aggregate_fwhr_child_lift(
            child_feature_sum=child_feature_sum,
            child_weight_sum_feature=child_weight_sum_feature,
            child_support=child_support,
            child_to_parent=child_to_parent_global.to(device=child_feature_sum.device),
            num_parents=int(num_parents),
            context_dim=int(context_dim),
            detail_dim=int(detail_dim),
            eps=float(self._cfg_get(getattr(self, "stage2_0_biggs_lifting_cfg", {}) or {}, "eps", 1.0e-6)),
            detail_valid_threshold=detail_valid_threshold,
            parent_obs_mode=str(self._cfg_get(getattr(self, "stage2_0_biggs_lifting_cfg", {}) or {}, "parent_obs_mode", "zero")),
        )
        stats: Dict[str, float] = {
            "iforward/fwhr/context_dim": float(context_dim),
            "iforward/fwhr/detail_dim": float(detail_dim),
            "iforward/fwhr/detail_support_min_bg": float(
                getattr(self, "stage2_0_fwhr_detail_support_min", {}).get("bg", 0.0)
            ),
            "iforward/fwhr/detail_support_min_distant": float(
                getattr(self, "stage2_0_fwhr_detail_support_min", {}).get("distant", 0.0)
            ),
            "iforward/fwhr/detail_support_min_rigid": float(
                getattr(self, "stage2_0_fwhr_detail_support_min", {}).get("rigid", 0.0)
            ),
        }
        for key, value in dict(lift.aux or {}).items():
            if torch.is_tensor(value):
                stats[f"iforward/fwhr/{key}"] = float(value.item())
            elif isinstance(value, (int, float)):
                stats[f"iforward/fwhr/{key}"] = float(value)
        valid_all = lift.child_detail_valid.detach().to(dtype=torch.float32)
        cursor = 0
        if int(num_bg) > 0:
            stats["iforward/fwhr/detail_valid_ratio_bg"] = float(valid_all[cursor : cursor + int(num_bg)].mean().item())
            cursor += int(num_bg)
        if int(num_distant) > 0:
            stats["iforward/fwhr/detail_valid_ratio_distant"] = float(
                valid_all[cursor : cursor + int(num_distant)].mean().item()
            )
            cursor += int(num_distant)
        if int(num_rigid_s) > 0:
            stats["iforward/fwhr/detail_valid_ratio_rigid"] = float(
                valid_all[cursor : cursor + int(num_rigid_s)].mean().item()
            )
        return lift, stats

    def _stage2_0_biggs_projection_stats(
        self,
        *,
        prefix: str,
        projection: Optional[BigGSParentProjection],
        parent_count: Optional[torch.Tensor],
        max_scale: Optional[float] = None,
    ) -> Dict[str, float]:
        if projection is None or int(projection.num_parents) == 0:
            return {
                f"{prefix}/child_count_mean": 0.0,
                f"{prefix}/child_count_p95": 0.0,
                f"{prefix}/child_count_max": 0.0,
                f"{prefix}/parent_opacity_saturation_ratio": 0.0,
                f"{prefix}/parent_scale_p95": 0.0,
                f"{prefix}/parent_scale_max": 0.0,
                f"{prefix}/parent_scale_clip_ratio": 0.0,
                f"{prefix}/projector_backend_id": 0.0,
                f"{prefix}/parent_runtime_init_backend_id": 0.0,
                f"{prefix}/parent_runtime_update_backend_id": 0.0,
            }
        interval = int(self._cfg_get(getattr(self, "stage2_0_biggs_projector_cfg", {}) or {}, "stats_interval", 1) or 1)
        if interval > 1:
            counter = int(getattr(self, "_stage2_0_biggs_projection_stats_counter", 0)) + 1
            self._stage2_0_biggs_projection_stats_counter = counter
            if counter % interval != 0:
                return {
                    f"{prefix}/child_count_mean": 0.0,
                    f"{prefix}/child_count_p95": 0.0,
                    f"{prefix}/child_count_max": 0.0,
                    f"{prefix}/parent_opacity_saturation_ratio": 0.0,
                    f"{prefix}/parent_scale_p95": 0.0,
                    f"{prefix}/parent_scale_max": 0.0,
                    f"{prefix}/parent_scale_clip_ratio": 0.0,
                    f"{prefix}/projector_backend_id": float((projection.aux_stats or {}).get("projector_backend_id", 0.0)),
                    f"{prefix}/parent_runtime_init_backend_id": float((projection.aux_stats or {}).get("parent_runtime_init_backend_id", 0.0)),
                    f"{prefix}/parent_runtime_update_backend_id": float((projection.aux_stats or {}).get("parent_runtime_update_backend_id", 0.0)),
                }
        params = projection.params
        ref = params["means"]
        if parent_count is None:
            counts = ref.new_zeros((int(projection.num_parents),))
        else:
            counts = parent_count.to(device=ref.device, dtype=ref.dtype).reshape(-1)
        scales_3 = torch.exp(params["scales_log"])
        scales = scales_3.reshape(-1)
        opacity = torch.sigmoid(params["opacity_logit"]).reshape(-1)
        cap = float(self._cfg_get(getattr(self, "stage2_0_biggs_projector_cfg", {}) or {}, "opacity_cap", 0.98))
        opacity_sat = (opacity >= float(cap) * 0.99).to(dtype=ref.dtype)
        max_scale_f = float(max_scale) if max_scale is not None else float("inf")
        if math.isfinite(max_scale_f) and max_scale_f > 0.0 and scales_3.numel():
            scale_clip = (scales_3 >= max_scale_f * 0.999).any(dim=-1).to(dtype=ref.dtype)
            scale_clip_ratio = float(scale_clip.detach().mean().item())
        else:
            scale_clip_ratio = 0.0
        return {
            f"{prefix}/child_count_mean": float(counts.detach().mean().item()) if counts.numel() else 0.0,
            f"{prefix}/child_count_p95": float(torch.quantile(counts.detach(), 0.95).item()) if counts.numel() else 0.0,
            f"{prefix}/child_count_max": float(counts.detach().max().item()) if counts.numel() else 0.0,
            f"{prefix}/parent_opacity_saturation_ratio": float(opacity_sat.detach().mean().item()) if opacity_sat.numel() else 0.0,
            f"{prefix}/parent_scale_p95": float(torch.quantile(scales.detach(), 0.95).item()) if scales.numel() else 0.0,
            f"{prefix}/parent_scale_max": float(scales.detach().max().item()) if scales.numel() else 0.0,
            f"{prefix}/parent_scale_clip_ratio": scale_clip_ratio,
            f"{prefix}/projector_backend_id": float((projection.aux_stats or {}).get("projector_backend_id", 0.0)),
            f"{prefix}/parent_runtime_init_backend_id": float((projection.aux_stats or {}).get("parent_runtime_init_backend_id", 0.0)),
            f"{prefix}/parent_runtime_update_backend_id": float((projection.aux_stats or {}).get("parent_runtime_update_backend_id", 0.0)),
        }

    def _observe_stage2_0_biggs_measurement(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
        biggs_state: Optional[IForwardBigGSState] = None,
        biggs_parent_runtime: Optional[BigGSBlockRuntime] = None,
        biggs_scene_id: Optional[int] = None,
        biggs_segment_id: Optional[int] = None,
        biggs_episode_id: Optional[int] = None,
        parent_optimizer_state: Optional[Any] = None,
        visit_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        observe_total_t0 = time.perf_counter()
        biggs_observe_perf: Dict[str, float] = {}
        if isinstance(visit_meta, dict) and "global_step" in visit_meta:
            self.stage3_0_global_step = int(visit_meta.get("global_step", 0) or 0)

        def _record_observe_time(name: str, start: float) -> None:
            biggs_observe_perf[f"iforward/biggs/time_observe_{name}_ms"] = float((time.perf_counter() - start) * 1000.0)

        projector_cfg = getattr(self, "stage2_0_biggs_projector_cfg", {}) or {}
        detach_local = not bool(self._cfg_get(projector_cfg, "grad_to_local_state", False))
        t0 = time.perf_counter()
        if bool(detach_local):
            bg_m, distant_m, rigid_m = local_state.to(device=self.device).to_node_states_detached_view()
        else:
            bg_m, distant_m, rigid_m = self._local_to_node_states(local_state, detach=False)
        _record_observe_time("local_state_view", t0)
        t0 = time.perf_counter()
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
        _record_observe_time("rigid_route", t0)
        t0 = time.perf_counter()
        state_cpu, state, assignment_cache_stats = self._stage2_0_get_or_build_biggs_state_for_observe(
            existing=biggs_state,
            batch=batch,
            bg=bg_m,
            distant=distant_m,
            rigid=rigid_m,
            ids_override=(
                int(biggs_scene_id) if biggs_scene_id is not None else -1,
                int(biggs_segment_id) if biggs_segment_id is not None else -1,
                int(biggs_episode_id) if biggs_episode_id is not None else -1,
            )
            if biggs_scene_id is not None or biggs_segment_id is not None or biggs_episode_id is not None
            else None,
        )
        time_assignment_ms = (time.perf_counter() - t0) * 1000.0
        biggs_observe_perf["iforward/biggs/time_observe_assignment_ms"] = float(time_assignment_ms)
        if state.bg is None:
            raise RuntimeError("BigGS stage2_0 requires bg assignment")
        self._mem_debug(
            "biggs/after_state",
            num_bg=int(bg_m.means.shape[0]),
            num_distant=int(distant_m.means.shape[0]) if distant_m is not None else 0,
            num_rigid=int(rigid_m.means.shape[0]) if rigid_m is not None else 0,
            parent_bg=int(state.bg.num_parents),
            parent_distant=int(state.distant.num_parents) if state.distant is not None else 0,
            parent_rigid=int(state.rigid.num_parents) if state.rigid is not None else 0,
        )
        t0 = time.perf_counter()
        parent_state_mode = str(self._cfg_get(getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}, "mode", "none")).lower()
        parent_runtime_enabled = parent_state_mode == "incremental_sufficient_stats"
        runtime_reuse = (
            bool(parent_runtime_enabled)
            and isinstance(biggs_parent_runtime, BigGSBlockRuntime)
            and int(biggs_parent_runtime.source_frame_idx) == int(source_frame_idx)
            and biggs_parent_runtime.bg is not None
        )
        active_rigid = None
        rigid_proj_active = None
        next_parent_runtime = biggs_parent_runtime if bool(runtime_reuse) else None

        if bool(runtime_reuse) and biggs_parent_runtime is not None:
            t_reuse = time.perf_counter()
            bg_proj = projection_from_runtime(biggs_parent_runtime.bg)  # type: ignore[arg-type]
            distant_proj = (
                projection_from_runtime(biggs_parent_runtime.distant)
                if biggs_parent_runtime.distant is not None
                else None
            )
            active_rigid = biggs_parent_runtime.rigid_active_assignment
            rigid_proj_active = (
                projection_from_runtime(biggs_parent_runtime.rigid_active)
                if biggs_parent_runtime.rigid_active is not None
                else None
            )
            _record_observe_time("parent_runtime_reuse", t_reuse)
        else:
            t_bg_distant = time.perf_counter()
            if bool(parent_runtime_enabled):
                projector_cfg_runtime = self._stage2_0_parent_runtime_cfg_for_branch("bg")
                bg_runtime = self._stage2_0_init_parent_branch_runtime(
                    params={
                        "means": bg_m.means,
                        "scales_log": bg_m.scales_log,
                        "quats": bg_m.quats,
                        "opacity_logit": bg_m.opacity_logit,
                        "sh_dc": bg_m.sh_dc,
                        "sh_rest": bg_m.sh_rest,
                    },
                    child_to_parent=state.bg.child_to_parent,
                    parent_count=state.bg.parent_count,
                    child_mass=state.bg.child_mass,
                    cfg=projector_cfg_runtime,
                    child_order=state.bg.child_order,
                    parent_start=state.bg.parent_start,
                    max_scale=self._stage2_0_biggs_max_scale("bg"),
                    assignment_signature="bg",
                )
                bg_proj = projection_from_runtime(bg_runtime)
                distant_runtime = None
                distant_proj = None
                if distant_m is not None and state.distant is not None:
                    projector_cfg_runtime = self._stage2_0_parent_runtime_cfg_for_branch("distant")
                    distant_runtime = self._stage2_0_init_parent_branch_runtime(
                        params={
                            "means": distant_m.means,
                            "scales_log": distant_m.scales_log,
                            "quats": distant_m.quats,
                            "opacity_logit": distant_m.opacity_logit,
                            "sh_dc": distant_m.sh_dc,
                            "sh_rest": distant_m.sh_rest,
                        },
                        child_to_parent=state.distant.child_to_parent,
                        parent_count=state.distant.parent_count,
                        child_mass=state.distant.child_mass,
                        cfg=projector_cfg_runtime,
                        child_order=state.distant.child_order,
                        parent_start=state.distant.parent_start,
                        max_scale=self._stage2_0_biggs_max_scale("distant"),
                        assignment_signature="distant",
                    )
                    distant_proj = projection_from_runtime(distant_runtime)
                next_parent_runtime = BigGSBlockRuntime(
                    bg=bg_runtime,
                    distant=distant_runtime,
                    bg_assignment=state.bg,
                    distant_assignment=state.distant,
                    source_frame_idx=int(source_frame_idx),
                    exact_refresh_count=1,
                    incremental_update_count=0,
                )
            else:
                bg_proj = self._stage2_0_project_branch(branch=bg_m, assignment=state.bg, branch_name="bg")
                distant_proj = None
                if distant_m is not None and state.distant is not None:
                    distant_proj = self._stage2_0_project_branch(branch=distant_m, assignment=state.distant, branch_name="distant")
            _record_observe_time("parent_project_bg_distant", t_bg_distant)

        self._mem_debug("biggs/after_project_bg", m_bg=int(bg_proj.num_parents))
        if distant_proj is not None:
            self._mem_debug("biggs/after_project_distant", m_distant=int(distant_proj.num_parents))

        t_rigid_project = time.perf_counter()
        if not bool(runtime_reuse) and rigid_m is not None and state.rigid is not None:
            active_rigid = build_rigid_active_assignment(
                rigid_assignment=state.rigid,
                fine_S=route.S,
                inside_mask_S=route.inside_mask_S,
            )
            if active_rigid is not None and int(active_rigid.active_parent_global.numel()) > 0:
                s = active_rigid.fine_S.long().to(device=rigid_m.means.device)
                projector_cfg = (
                    self._stage2_0_parent_runtime_cfg_for_branch("rigid")
                    if bool(parent_runtime_enabled)
                    else dict(getattr(self, "stage2_0_biggs_projector_cfg", {}) or {})
                )
                max_scale = float(self._cfg_get(projector_cfg, "max_scale_rigid", 1.0))
                projector_cfg["tau_parent_scale"] = self._stage2_0_biggs_tau_parent_scale("rigid")
                rigid_params = {
                    "means": route.means_world_S,
                    "quats": route.quats_world_S,
                    "scales_log": rigid_m.scales_log.index_select(0, s),
                    "opacity_logit": rigid_m.opacity_logit.index_select(0, s),
                    "sh_dc": rigid_m.sh_dc.index_select(0, s),
                    "sh_rest": rigid_m.sh_rest.index_select(0, s),
                }
                if bool(parent_runtime_enabled):
                    rigid_runtime = self._stage2_0_init_parent_branch_runtime(
                        params=rigid_params,
                        child_to_parent=active_rigid.child_to_active_parent_S,
                        parent_count=active_rigid.active_parent_count,
                        child_mass=active_rigid.child_mass_S,
                        cfg=projector_cfg,
                        child_order=active_rigid.active_child_order_S,
                        parent_start=active_rigid.active_parent_start,
                        max_scale=max_scale,
                        assignment_signature="rigid_active",
                    )
                    rigid_proj_active = projection_from_runtime(rigid_runtime)
                    if next_parent_runtime is not None:
                        next_parent_runtime.rigid_active = rigid_runtime
                        next_parent_runtime.rigid_active_assignment = active_rigid
                else:
                    rigid_proj_active = self._stage2_0_project_active_rigid_parents(
                        means_world_S=route.means_world_S,
                        quats_world_S=route.quats_world_S,
                        scales_log_S=rigid_m.scales_log.index_select(0, s),
                        opacity_logit_S=rigid_m.opacity_logit.index_select(0, s),
                        sh_dc_S=rigid_m.sh_dc.index_select(0, s),
                        sh_rest_S=rigid_m.sh_rest.index_select(0, s),
                        child_to_active_parent_S=active_rigid.child_to_active_parent_S,
                        child_mass_S=active_rigid.child_mass_S,
                        active_parent_count=active_rigid.active_parent_count,
                        cfg=projector_cfg,
                        max_scale=max_scale,
                        active_child_order_S=active_rigid.active_child_order_S,
                        active_parent_start=active_rigid.active_parent_start,
                    )
            else:
                rigid_proj_active = self._stage2_0_empty_projection_like(
                    bg_m.means,
                    sh_rest_bases=int(bg_m.sh_rest.shape[1]),
                )
                if next_parent_runtime is not None:
                    next_parent_runtime.rigid_active_assignment = active_rigid
            self._mem_debug(
                "biggs/after_project_rigid_active",
                m_rigid=int(rigid_proj_active.num_parents) if rigid_proj_active is not None else 0,
            )
        elif not bool(runtime_reuse) and rigid_m is None:
            active_rigid = build_rigid_active_assignment(
                rigid_assignment=None,
                fine_S=route.S,
                inside_mask_S=route.inside_mask_S,
            )
            if next_parent_runtime is not None:
                next_parent_runtime.rigid_active_assignment = active_rigid
        time_parent_project_ms = (time.perf_counter() - t0) * 1000.0
        _record_observe_time("parent_project_rigid_active", t_rigid_project)
        if bool(parent_runtime_enabled) and not bool(runtime_reuse) and next_parent_runtime is not None:
            self._stage2_0_biggs_parent_runtime_block_counter = (
                int(getattr(self, "_stage2_0_biggs_parent_runtime_block_counter", 0)) + 1
            )
            next_parent_runtime.block_id = int(self._stage2_0_biggs_parent_runtime_block_counter)
        drift_stats: Dict[str, float] = {}
        if bool(parent_runtime_enabled) and bool(runtime_reuse):
            drift_stats = self._stage2_0_maybe_check_parent_runtime_drift(
                runtime=next_parent_runtime,
                bg=bg_m,
                distant=distant_m,
                rigid=rigid_m,
                source_frame_idx=int(source_frame_idx),
            )

        t_scene_source = time.perf_counter()
        parts = [self._stage2_0_scene_parts_from_params(bg_proj.params)]
        if distant_proj is not None and int(distant_proj.num_parents) > 0:
            parts.append(self._stage2_0_scene_parts_from_params(distant_proj.params))
        if rigid_proj_active is not None and int(rigid_proj_active.num_parents) > 0:
            parts.append(self._stage2_0_scene_parts_from_params(rigid_proj_active.params))
        parent_scene = self._stage2_0_cat_scene(parts)
        self._mem_debug("biggs/after_parent_scene", num_parent=int(parent_scene["means"].shape[0]))
        source_views, source_images, source_sky_masks, source_egocar_masks = self._source_subset(batch, source_indices)
        height, width = spatial_hw_from_image_tensor(source_images[0])
        lifting_cfg = getattr(self, "stage2_0_biggs_lifting_cfg", {}) or {}
        stage3_enabled = bool(getattr(self, "stage3_0_enabled", False))
        fwhr_enabled = str(self._cfg_get(lifting_cfg, "type", "")).lower() == "fwhr"
        parent_scene_for_cnn = bool(self._cfg_get(getattr(self, "stage2_0_biggs_observe_cfg", {}) or {}, "parent_scene_for_cnn", True))
        cnn_scene = parent_scene
        fine_scene = None
        if bool(stage3_enabled):
            fwhr_enabled = False
            parent_scene_for_cnn = False
        if bool(fwhr_enabled) or bool(stage3_enabled) or not bool(parent_scene_for_cnn):
            fine_scene = self._stage2_0_fine_scene_from_state(bg=bg_m, distant=distant_m, rigid=rigid_m, route=route)
        if bool(fwhr_enabled) or bool(stage3_enabled):
            cnn_scene = fine_scene
        elif not bool(parent_scene_for_cnn):
            cnn_scene = fine_scene
        _record_observe_time("parent_scene_source", t_scene_source)
        repair_training_active = bool(self._repair_training_enabled_for_visit(visit_meta))
        repair_freeze_2d = bool(self._repair_training_freeze_2d_for_visit(visit_meta))
        repair_no_grad_2d = bool(self._repair_training_no_grad_2d_for_visit(visit_meta))
        repair_train_2d_mode = self._repair_training_train_2d_mode(visit_meta)
        repair_train_2d_mode_id = {"trainable": 1, "frozen_no_grad": 2, "auto": 3}.get(str(repair_train_2d_mode), 0)
        repair_training_aux: Dict[str, float] = {
            "iforward/repair_training/enabled": 1.0 if bool(repair_training_active) else 0.0,
            "iforward/repair_training/freeze_2d_frontend": 1.0 if bool(repair_freeze_2d) else 0.0,
            "iforward/repair_training/no_grad_2d_forward": 1.0 if bool(repair_no_grad_2d) else 0.0,
            "iforward/repair_training/train_2d_mode_id": float(repair_train_2d_mode_id),
            "iforward/repair_training/stage3_2_policy_override": (
                1.0 if bool(self._repair_training_stage3_2_frozen_no_grad(visit_meta)) else 0.0
            ),
        }
        t0 = time.perf_counter()
        dino_cache_key = self._stage2_0_dino_cache_key(
            batch=batch,
            source_indices=source_indices,
            source_views=source_views,
            source_images=source_images,
            source_frame_idx=int(source_frame_idx),
            height=height,
            width=width,
        )
        cnn_ctx = torch.no_grad() if bool(repair_no_grad_2d) else nullcontext()
        with cnn_ctx:
            cnn_inputs = self._render_source_scene_only_for_cnn(
                gaussians_scene=cnn_scene,
                source_views=source_views,
                source_images=source_images,
                source_sky_masks=source_sky_masks,
                source_egocar_masks=source_egocar_masks,
                height=height,
                width=width,
                dino_cache_key=dino_cache_key,
            )
        if bool(repair_freeze_2d):
            cnn_inputs = self._detach_cnn_inputs_for_repair_training(cnn_inputs)
        if bool(stage3_enabled):
            cnn_inputs = self._stage3_cast_feature_cache(cnn_inputs)
        repair_training_aux["iforward/repair_training/features_2d_requires_grad"] = float(
            1.0 if torch.is_tensor(cnn_inputs.get("features_2d")) and bool(cnn_inputs["features_2d"].requires_grad) else 0.0
        )
        repair_training_aux["iforward/repair_training/detail_2d_requires_grad"] = float(
            1.0 if torch.is_tensor(cnn_inputs.get("fwhr_detail_2d")) and bool(cnn_inputs["fwhr_detail_2d"].requires_grad) else 0.0
        )
        time_parent_render_cnn_ms = (time.perf_counter() - t0) * 1000.0
        self._mem_debug("biggs/after_parent_render_cnn")
        perf_before = {
            str(k): float(v)
            for k, v in dict(getattr(self, "_perf_acc", {}) or {}).items()
            if str(k).startswith("2d_bp_scene_")
        }
        t0 = time.perf_counter()
        fwhr_stats: Dict[str, float] = {}
        m_bg = int(bg_proj.num_parents)
        m_d = int(distant_proj.num_parents) if distant_proj is not None else 0
        m_r = int(rigid_proj_active.num_parents) if rigid_proj_active is not None else 0
        num_parent = int(m_bg + m_d + m_r)
        child_detail_bg = None
        child_detail_d = None
        child_detail_r = None
        child_detail_valid_bg = None
        child_detail_valid_d = None
        child_detail_valid_r = None
        child_detail_support_bg = None
        child_detail_support_d = None
        child_detail_support_r = None
        stage3_anchor_stats = None
        stage3_context_2d = None
        stage3_detail_2d = None
        stage3_dino_native_2d = None
        stage3_child_to_parent_global = None
        stage3_parent_reg_terms: Dict[str, torch.Tensor] = {}
        if bool(fwhr_enabled):
            if fine_scene is None:
                raise RuntimeError("FW-HR expected fine_scene to be built")
            context_dim = int(self._cfg_get(lifting_cfg, "context_dim", int(cnn_inputs["features_2d"].shape[-1])))
            detail_dim = int(self._cfg_get(lifting_cfg, "detail_dim", 8))
            lifting_backend = str(self._cfg_get(lifting_cfg, "backend", "")).strip().lower()
            legacy_fused_flag = bool(self._cfg_get(lifting_cfg, "fused_cuda", True))
            if not bool(legacy_fused_flag) and lifting_backend not in {"child_v4_raw_torch_aggregate"}:
                raise RuntimeError(
                    "FW-HR training path requires biggs.lifting.fused_cuda=true or "
                    "biggs.lifting.backend=child_v4_raw_torch_aggregate"
                )
            child_to_parent_global = self._stage2_0_fwhr_child_to_parent_global(
                state=state,
                active_rigid=active_rigid,
                num_bg=int(bg_m.means.shape[0]),
                num_distant=int(distant_m.means.shape[0]) if distant_m is not None else 0,
                num_rigid_s=int(route.S.numel()),
                num_parent_bg=int(m_bg),
                num_parent_distant=int(m_d),
            )
            fwhr_lift, fwhr_stats = self._stage2_0_fwhr_lift_from_fine_scene(
                fine_scene=fine_scene,
                source_views=source_views,
                cnn_inputs=cnn_inputs,
                height=height,
                width=width,
                child_to_parent_global=child_to_parent_global,
                num_parents=int(num_parent),
                context_dim=int(context_dim),
                detail_dim=int(detail_dim),
                num_bg=int(bg_m.means.shape[0]),
                num_distant=int(distant_m.means.shape[0]) if distant_m is not None else 0,
                num_rigid_s=int(route.S.numel()),
            )
            feat_all = fwhr_lift.parent_context
            acc_all = fwhr_lift.parent_support
            obs_all = fwhr_lift.parent_obs_code
            num_bg_f = int(bg_m.means.shape[0])
            num_d_f = int(distant_m.means.shape[0]) if distant_m is not None else 0
            num_r_f = int(route.S.numel())
            cstart = 0
            child_detail_bg = fwhr_lift.child_detail[cstart : cstart + num_bg_f]
            child_detail_valid_bg = fwhr_lift.child_detail_valid[cstart : cstart + num_bg_f]
            child_detail_support_bg = fwhr_lift.child_detail_support[cstart : cstart + num_bg_f]
            cstart += num_bg_f
            if num_d_f > 0:
                child_detail_d = fwhr_lift.child_detail[cstart : cstart + num_d_f]
                child_detail_valid_d = fwhr_lift.child_detail_valid[cstart : cstart + num_d_f]
                child_detail_support_d = fwhr_lift.child_detail_support[cstart : cstart + num_d_f]
            cstart += num_d_f
            if num_r_f > 0:
                child_detail_r = fwhr_lift.child_detail[cstart : cstart + num_r_f]
                child_detail_valid_r = fwhr_lift.child_detail_valid[cstart : cstart + num_r_f]
                child_detail_support_r = fwhr_lift.child_detail_support[cstart : cstart + num_r_f]
        elif bool(stage3_enabled):
            if fine_scene is None:
                raise RuntimeError("Stage3_0 expected fine_scene to be built")
            context_dim = int(self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "context_dim", int(cnn_inputs["features_2d"].shape[-1])))
            detail_dim = int(self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "detail_dim", 8))
            detail_2d = cnn_inputs.get("fwhr_detail_2d")
            context_2d = cnn_inputs["features_2d"]
            if detail_2d is None or not torch.is_tensor(detail_2d):
                raise RuntimeError("Stage3_0 requires image_feature_extractor.forward_fwhr/detail output")
            if int(context_2d.shape[-1]) != int(context_dim):
                raise ValueError(f"Stage3_0 context dim mismatch: got {int(context_2d.shape[-1])}, expected {context_dim}")
            if int(detail_2d.shape[-1]) != int(detail_dim):
                raise ValueError(f"Stage3_0 detail dim mismatch: got {int(detail_2d.shape[-1])}, expected {detail_dim}")
            parent_lift_dtype = self._stage3_parent_lift_dtype(context_2d)
            child_detail_dtype = self._stage3_child_detail_dtype(detail_2d)
            parent_context_cache_dtype = self._stage3_parent_context_cache_dtype(context_2d)
            child_detail_output_dtype = self._stage3_child_detail_output_dtype(detail_2d)
            parent_context_2d = context_2d.to(dtype=parent_lift_dtype)
            child_detail_2d = detail_2d.to(dtype=child_detail_dtype)
            if self._stage3_0_memory_aux_enabled() and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            child_to_parent_global = self._stage2_0_fwhr_child_to_parent_global(
                state=state,
                active_rigid=active_rigid,
                num_bg=int(bg_m.means.shape[0]),
                num_distant=int(distant_m.means.shape[0]) if distant_m is not None else 0,
                num_rigid_s=int(route.S.numel()),
                num_parent_bg=int(m_bg),
                num_parent_distant=int(m_d),
            )
            anchor_stats, anchor_aux = self._stage3_0_build_anchor_stats(
                fine_scene=fine_scene,
                source_views=source_views,
                source_pair_valid_mask=cnn_inputs["source_pair_valid_mask"],
                child_to_parent_global=child_to_parent_global,
                num_children=int(child_to_parent_global.numel()),
                num_parents=int(num_parent),
                height=int(height),
                width=int(width),
            )
            anchor_aux.update(self._stage3_0_memory_aux("after_anchor"))
            stage3_anchor_stats = anchor_stats
            stage3_context_2d = parent_context_2d
            stage3_detail_2d = child_detail_2d
            stage3_dino_native_2d = None
            stage3_child_to_parent_global = child_to_parent_global
            parent_lifting_type = str(getattr(self, "stage3_parent_lifting_type", "legacy_direct_lift")).lower()
            parent_reg: Dict[str, torch.Tensor] = {}
            if parent_lifting_type == "legacy_direct_lift":
                amp_ctx = self._iforward_amp_autocast() if self._stage3_parent_lift_amp_enabled() else self._iforward_amp_fp32()
                with amp_ctx:
                    feat_all, acc_all = self._backproject_scene_features_multi_camera(
                        gaussians_scene=parent_scene,
                        source_views=source_views,
                        features_2d=parent_context_2d,
                        source_pair_valid_mask=cnn_inputs["source_pair_valid_mask"],
                        height=height,
                        width=width,
                        return_debug_stats=bool(getattr(self, "stage2_0_biggs_return_debug_stats", True)),
                    )
                feat_all = feat_all.to(dtype=parent_context_cache_dtype)
                acc_all = acc_all.to(device=feat_all.device, dtype=torch.float32)
                parent_aux = {
                    "iforward/stage3/parent_legacy_direct_lift_enabled": 1.0,
                    "iforward/stage3/parent_sparse_gather_enabled": 0.0,
                    "iforward/stage3/parent_dino_native_stage3_enabled": 0.0,
                }
            elif parent_lifting_type == "sparse_gather":
                stage3_dino_native_2d = cnn_inputs.get("stage3_dino_native_2d")
                cnn_inputs_parent = dict(cnn_inputs)
                cnn_inputs_parent["features_2d"] = parent_context_2d
                amp_ctx = self._iforward_amp_autocast() if self._stage3_parent_lift_amp_enabled() else self._iforward_amp_fp32()
                with amp_ctx:
                    feat_all, acc_all, parent_aux, parent_reg = self._stage3_0_parent_sparse_gather(
                        cnn_inputs=cnn_inputs_parent,
                        anchor_stats=anchor_stats,
                        bg_proj=bg_proj,
                        distant_proj=distant_proj,
                        rigid_proj_active=rigid_proj_active,
                        parent_optimizer_state=parent_optimizer_state,
                        height=int(height),
                        width=int(width),
                        num_parent_bg=int(m_bg),
                        num_parent_distant=int(m_d),
                        num_parent_rigid=int(m_r),
                    )
                parent_aux = {
                    **parent_aux,
                    "iforward/stage3/parent_legacy_direct_lift_enabled": 0.0,
                    "iforward/stage3/parent_sparse_gather_enabled": 1.0,
                }
                feat_all = feat_all.to(dtype=parent_context_cache_dtype)
                acc_all = acc_all.to(device=feat_all.device, dtype=torch.float32)
            else:
                raise ValueError(f"unsupported Stage3_0 parent lifting type={parent_lifting_type!r}")
            parent_aux.update(self._stage3_0_memory_aux("after_parent_lift"))
            stage3_parent_reg_terms = parent_reg
            obs_all = None
            fwhr_stats = {
                **anchor_aux,
                **parent_aux,
                "iforward/stage3/enabled": 1.0,
                "iforward/stage3/context_dim": float(context_dim),
                "iforward/stage3/detail_dim": float(detail_dim),
                "iforward/stage3/optimizer_prior_present": 1.0 if parent_optimizer_state is not None else 0.0,
                "amp/dtype/features_2d_cache": float(amp_dtype_id(context_2d.dtype)),
                "amp/dtype/parent_context_cache": float(amp_dtype_id(feat_all.dtype)),
                "amp/dtype/child_detail": float(amp_dtype_id(child_detail_output_dtype)),
                "iforward/stage3/parent_lift_amp_enabled": 1.0 if self._stage3_parent_lift_amp_enabled() else 0.0,
                "iforward/stage3/parent_lift_dtype_id": float(amp_dtype_id(parent_context_2d.dtype)),
                "iforward/stage3/parent_context_cache_dtype_id": float(amp_dtype_id(feat_all.dtype)),
                "iforward/stage3/child_gather_amp_enabled": 1.0 if self._stage3_child_gather_amp_enabled() else 0.0,
                "iforward/stage3/child_detail_gather_dtype_id": float(amp_dtype_id(child_detail_2d.dtype)),
                "iforward/stage3/child_detail_output_dtype_id": float(amp_dtype_id(child_detail_output_dtype)),
            }
            child_measurement = {
                "stage3_anchor_stats": stage3_anchor_stats,
                "stage3_detail_2d": stage3_detail_2d,
                "stage3_child_to_parent_global": stage3_child_to_parent_global,
                "stage3_gather_reg_terms": stage3_parent_reg_terms,
                "stage3_image_height": int(height),
                "stage3_image_width": int(width),
                "num_bg": int(bg_m.means.shape[0]),
                "num_distant": int(distant_m.means.shape[0]) if distant_m is not None else 0,
                "num_rigid_S": int(route.S.numel()),
                "assign_bg": state.bg,
                "assign_distant": state.distant,
                "assign_rigid_active": active_rigid,
                "route": route,
            }
            self._stage3_0_gather_child_detail(local_state=local_state, measurement=child_measurement)
            child_measurement.update(self._stage3_0_memory_aux("after_child_gather"))
            stage3_parent_reg_terms = child_measurement.get("stage3_gather_reg_terms", stage3_parent_reg_terms)
            child_detail_bg = child_measurement.get("child_detail_bg")
            child_detail_d = child_measurement.get("child_detail_distant")
            child_detail_r = child_measurement.get("child_detail_rigid_S")
            child_detail_valid_bg = child_measurement.get("child_detail_valid_bg")
            child_detail_valid_d = child_measurement.get("child_detail_valid_distant")
            child_detail_valid_r = child_measurement.get("child_detail_valid_rigid_S")
            child_detail_support_bg = child_measurement.get("child_detail_support_bg")
            child_detail_support_d = child_measurement.get("child_detail_support_distant")
            child_detail_support_r = child_measurement.get("child_detail_support_rigid_S")
            for key, value in child_measurement.items():
                if str(key).startswith("iforward/stage3/") and isinstance(value, (int, float)):
                    fwhr_stats[str(key)] = float(value)
        else:
            feat_all, acc_all = self._backproject_scene_features_multi_camera(
                gaussians_scene=parent_scene,
                source_views=source_views,
                features_2d=cnn_inputs["features_2d"],
                source_pair_valid_mask=cnn_inputs["source_pair_valid_mask"],
                height=height,
                width=width,
                return_debug_stats=bool(getattr(self, "stage2_0_biggs_return_debug_stats", True)),
            )
            obs_all = self._stage5_4_obs_code_all
        time_parent_lifting_ms = (time.perf_counter() - t0) * 1000.0
        perf_after = {
            str(k): float(v)
            for k, v in dict(getattr(self, "_perf_acc", {}) or {}).items()
            if str(k).startswith("2d_bp_scene_")
        }
        bp_delta_stats = {
            f"iforward/biggs/{key}": float(value) - float(perf_before.get(key, 0.0))
            for key, value in perf_after.items()
        }
        self._mem_debug("biggs/after_parent_lifting")
        if feat_all is None or acc_all is None:
            raise RuntimeError("BigGS parent lifting returned empty features")
        t_slice_stats = time.perf_counter()
        parent_obs_mode_default = "none" if bool(stage3_enabled) else "zero"
        parent_obs_mode = str(
            self._cfg_get(
                getattr(self, "stage2_0_biggs_lifting_cfg", {}) or {},
                "parent_obs_mode",
                parent_obs_mode_default,
            )
        ).lower()
        if obs_all is None and parent_obs_mode != "none":
            raise RuntimeError("BigGS parent lifting expected V4 obs_code")
        if obs_all is not None:
            obs_all = obs_all.to(device=feat_all.device, dtype=feat_all.dtype)
        if int(feat_all.shape[0]) != int(m_bg + m_d + m_r):
            raise RuntimeError("BigGS parent lifting row mismatch")
        start = 0
        feat_bg = feat_all[start : start + m_bg]
        acc_bg = acc_all[start : start + m_bg]
        obs_bg = obs_all[start : start + m_bg] if obs_all is not None else None
        start += m_bg
        feat_d = feat_all[start : start + m_d] if m_d > 0 else None
        acc_d = acc_all[start : start + m_d] if m_d > 0 else None
        obs_d = obs_all[start : start + m_d] if obs_all is not None and m_d > 0 else None
        start += m_d
        feat_r = feat_all[start : start + m_r] if m_r > 0 else None
        acc_r = acc_all[start : start + m_r] if m_r > 0 else None
        obs_r = obs_all[start : start + m_r] if obs_all is not None and m_r > 0 else None
        num_fine = int(bg_m.means.shape[0]) + (int(distant_m.means.shape[0]) if distant_m is not None else 0) + int(route.S.numel())
        stats = {
            **self._stage2_0_biggs_projection_stats(
                prefix="iforward/biggs/bg",
                projection=bg_proj,
                parent_count=state.bg.parent_count if state.bg is not None else None,
                max_scale=self._stage2_0_biggs_max_scale("bg"),
            ),
            **self._stage2_0_biggs_projection_stats(
                prefix="iforward/biggs/distant",
                projection=distant_proj,
                parent_count=state.distant.parent_count if state.distant is not None else None,
                max_scale=self._stage2_0_biggs_max_scale("distant"),
            ),
            **self._stage2_0_biggs_projection_stats(
                prefix="iforward/biggs/rigid_active",
                projection=rigid_proj_active,
                parent_count=active_rigid.active_parent_count if active_rigid is not None else None,
                max_scale=self._stage2_0_biggs_max_scale("rigid"),
            ),
            **fwhr_stats,
        }
        _record_observe_time("slice_stats", t_slice_stats)
        _record_observe_time("total", observe_total_t0)
        result = {
            "biggs_enabled": True,
            "biggs_mode": (
                "stage3_sparse_gather_event_decode"
                if bool(stage3_enabled)
                else "fwhr_lift_event_decode"
                if bool(fwhr_enabled)
                else "parent_lifting_event_decode"
            ),
            "biggs_state": state_cpu.detach(),
            "biggs_parent_runtime": next_parent_runtime,
            "route": route,
            "source_frame_idx": int(source_frame_idx),
            "assign_bg": state.bg,
            "assign_distant": state.distant,
            "assign_rigid": state.rigid,
            "assign_rigid_active": active_rigid,
            "parent_feat_2d_bg": feat_bg,
            "parent_acc_w_bg": acc_bg,
            "parent_obs_bg": obs_bg,
            "parent_params_bg": bg_proj.params,
            "parent_coords_bg": bg_proj.params["means"],
            "parent_mass_mean_bg": bg_proj.child_mass_mean,
            "parent_feat_2d_distant": feat_d,
            "parent_acc_w_distant": acc_d,
            "parent_obs_distant": obs_d,
            "parent_params_distant": None if distant_proj is None else distant_proj.params,
            "parent_coords_distant": None if distant_proj is None else distant_proj.params["means"],
            "parent_mass_mean_distant": None if distant_proj is None else distant_proj.child_mass_mean,
            "parent_feat_2d_rigid_S": feat_r,
            "parent_acc_w_rigid_S": acc_r,
            "parent_obs_rigid_S": obs_r,
            "parent_params_rigid_active": None if rigid_proj_active is None else rigid_proj_active.params,
            "parent_coords_rigid_S": None if rigid_proj_active is None else rigid_proj_active.params["means"],
            "parent_mass_mean_rigid_active": None if rigid_proj_active is None else rigid_proj_active.child_mass_mean,
            "child_detail_bg": child_detail_bg,
            "child_detail_distant": child_detail_d,
            "child_detail_rigid_S": child_detail_r,
            "child_detail_valid_bg": child_detail_valid_bg,
            "child_detail_valid_distant": child_detail_valid_d,
            "child_detail_valid_rigid_S": child_detail_valid_r,
            "child_detail_support_bg": child_detail_support_bg,
            "child_detail_support_distant": child_detail_support_d,
            "child_detail_support_rigid_S": child_detail_support_r,
            "stage3_gather_reg_terms": stage3_parent_reg_terms,
            "stage3_visit_meta": dict(visit_meta or {}),
            "num_bg": int(bg_m.means.shape[0]),
            "num_distant": int(distant_m.means.shape[0]) if distant_m is not None else 0,
            "num_rigid_S": int(route.S.numel()),
            "num_parent_bg": float(m_bg),
            "num_parent_distant": float(m_d),
            "num_parent_rigid_S": float(m_r),
            "iforward/biggs/num_fine_active": float(num_fine),
            "iforward/biggs/num_parent_total": float(num_parent),
            "iforward/biggs/compression_total_active": float(num_fine) / float(max(num_parent, 1)),
            "iforward/biggs/parent_scene_for_cnn": 1.0 if bool(parent_scene_for_cnn) else 0.0,
            "iforward/fwhr/enabled": 1.0 if bool(fwhr_enabled) else 0.0,
            "iforward/stage3/enabled": 1.0 if bool(stage3_enabled) else 0.0,
            "iforward/biggs/time_assignment_ms": float(time_assignment_ms),
            "iforward/biggs/time_parent_project_ms": float(time_parent_project_ms),
            "iforward/biggs/time_parent_render_cnn_ms": float(time_parent_render_cnn_ms),
            "iforward/biggs/time_parent_lifting_ms": float(time_parent_lifting_ms),
            "iforward/biggs/parent_runtime_reuse": 1.0 if bool(runtime_reuse) else 0.0,
            "iforward/biggs/exact_refresh_count": float(getattr(next_parent_runtime, "exact_refresh_count", 0) if next_parent_runtime is not None else 0),
            "iforward/biggs/incremental_update_count": float(getattr(next_parent_runtime, "incremental_update_count", 0) if next_parent_runtime is not None else 0),
            **assignment_cache_stats,
            **stats,
            **bp_delta_stats,
            **biggs_observe_perf,
            **drift_stats,
            **dict(cnn_inputs.get("dino_cache_stats", {}) or {}),
            **dict(cnn_inputs.get("cnn_perf_stats", {}) or {}),
            **repair_training_aux,
            "src_backproject_pass_count": 1,
        }
        if bool(stage3_enabled) and bool(
            self._cfg_get(getattr(self, "stage3_0_lifting_cfg", {}) or {}, "return_stage3_debug_tensors", False)
        ):
            result.update(
                {
                    "stage3_anchor_stats": stage3_anchor_stats,
                    "stage3_context_2d": stage3_context_2d,
                    "stage3_detail_2d": stage3_detail_2d,
                    "stage3_dino_native_2d": stage3_dino_native_2d,
                    "stage3_child_to_parent_global": stage3_child_to_parent_global,
                    "stage3_image_height": int(height),
                    "stage3_image_width": int(width),
                }
            )
        return result

    def _observe_v4_measurement(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
        biggs_state: Optional[IForwardBigGSState] = None,
        biggs_parent_runtime: Optional[BigGSBlockRuntime] = None,
        biggs_scene_id: Optional[int] = None,
        biggs_segment_id: Optional[int] = None,
        biggs_episode_id: Optional[int] = None,
        parent_optimizer_state: Optional[Any] = None,
        visit_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        grad_enabled = str(getattr(self, "stage6_source_evidence_grad_mode", "no_grad_v4")) != "no_grad_v4"
        ctx_mgr = torch.enable_grad() if grad_enabled else torch.no_grad()
        if bool(getattr(self, "stage2_0_biggs_enabled", False)):
            with ctx_mgr:
                return self._observe_stage2_0_biggs_measurement(
                    local_state=local_state,
                    batch=batch,
                    source_indices=source_indices,
                    source_frame_idx=int(source_frame_idx),
                    biggs_state=biggs_state,
                    biggs_parent_runtime=biggs_parent_runtime,
                    biggs_scene_id=biggs_scene_id,
                    biggs_segment_id=biggs_segment_id,
                    biggs_episode_id=biggs_episode_id,
                    parent_optimizer_state=parent_optimizer_state,
                    visit_meta=visit_meta,
                )
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
    def _stage2_0_params_from_node_state(branch: Any) -> Dict[str, torch.Tensor]:
        return {
            "means": branch.means,
            "scales_log": branch.scales_log,
            "quats": branch.quats,
            "opacity_logit": branch.opacity_logit,
            "sh_dc": branch.sh_dc,
            "sh_rest": branch.sh_rest,
        }

    def _stage2_0_projector_cfg_for_branch(self, branch_name: str) -> Dict[str, Any]:
        projector_cfg = dict(getattr(self, "stage2_0_biggs_projector_cfg", {}) or {})
        projector_cfg["tau_parent_scale"] = self._stage2_0_biggs_tau_parent_scale(str(branch_name))
        return projector_cfg

    def _stage2_0_parent_runtime_cfg_for_branch(self, branch_name: str) -> Dict[str, Any]:
        runtime_cfg = self._stage2_0_projector_cfg_for_branch(str(branch_name))
        parent_state_cfg = getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}
        runtime_cfg["child_cache_dtype"] = str(self._cfg_get(parent_state_cfg, "child_cache_dtype", "float32"))
        return runtime_cfg

    @staticmethod
    def _stage2_0_max_abs_param_error(a: torch.Tensor, b: torch.Tensor) -> float:
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"BigGS parent drift check shape mismatch: runtime={tuple(a.shape)} exact={tuple(b.shape)}")
        if int(a.numel()) == 0:
            return 0.0
        return float((a.detach().float() - b.detach().to(device=a.device).float()).abs().max().item())

    @staticmethod
    def _stage2_0_reanchored_weighted_second(stats_obj: Any, target_anchor: Optional[torch.Tensor]) -> torch.Tensor:
        second = stats_obj.weighted_second_sum.detach().float()
        if target_anchor is None:
            return second
        target = target_anchor.detach().to(device=second.device).float()
        source_anchor = stats_obj.second_anchor
        if source_anchor is None:
            source = torch.zeros_like(target)
        else:
            source = source_anchor.detach().to(device=second.device).float()
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"BigGS parent drift check second-anchor shape mismatch: source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        mass = stats_obj.weight_sum.detach().to(device=second.device).float().reshape(-1, 1)
        mass_safe = mass.clamp_min(1.0e-8)
        mean = stats_obj.weighted_mean_sum.detach().to(device=second.device).float() / mass_safe
        return second + mass * ((mean - target).square() - (mean - source).square())

    def _stage2_0_check_parent_runtime_branch_drift(
        self,
        *,
        branch_name: str,
        runtime: Any,
        params: Dict[str, torch.Tensor],
        child_to_parent: torch.Tensor,
        parent_count: torch.Tensor,
        child_mass: torch.Tensor,
        child_order: Optional[torch.Tensor] = None,
        parent_start: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        cfg = self._stage2_0_parent_runtime_cfg_for_branch(str(branch_name))
        exact = self._stage2_0_init_parent_branch_runtime(
            params=params,
            child_to_parent=child_to_parent,
            parent_count=parent_count,
            child_mass=child_mass,
            cfg=cfg,
            child_order=child_order,
            parent_start=parent_start,
            max_scale=self._stage2_0_biggs_max_scale(str(branch_name)),
            assignment_signature=f"{branch_name}_drift_exact",
        )
        stats: Dict[str, float] = {}
        max_err = 0.0
        for key in ("means", "scales_log", "opacity_logit", "sh_dc", "sh_rest"):
            err = self._stage2_0_max_abs_param_error(runtime.params[key], exact.params[key])
            stats[f"iforward/biggs/drift_{branch_name}_{key}_max"] = float(err)
            max_err = max(max_err, float(err))
        for key in (
            "weight_sum",
            "weighted_mean_sum",
            "tau_area_sum",
            "weighted_sh_dc_sum",
            "weighted_sh_rest_sum",
        ):
            err = self._stage2_0_max_abs_param_error(getattr(runtime.stats, key), getattr(exact.stats, key))
            stats[f"iforward/biggs/drift_{branch_name}_stats_{key}_max"] = float(err)
        runtime_second = self._stage2_0_reanchored_weighted_second(runtime.stats, exact.stats.second_anchor)
        exact_second = exact.stats.weighted_second_sum.detach().to(device=runtime_second.device).float()
        second_err = self._stage2_0_max_abs_param_error(runtime_second, exact_second)
        stats[f"iforward/biggs/drift_{branch_name}_stats_weighted_second_sum_max"] = float(second_err)
        if runtime.stats.second_anchor is not None and exact.stats.second_anchor is not None:
            anchor_err = self._stage2_0_max_abs_param_error(runtime.stats.second_anchor, exact.stats.second_anchor)
            stats[f"iforward/biggs/drift_{branch_name}_second_anchor_max"] = float(anchor_err)
        stats[f"iforward/biggs/drift_{branch_name}_max"] = float(max_err)
        return stats

    def _stage2_0_parent_runtime_drift_stats_for_nodes(
        self,
        *,
        runtime: BigGSBlockRuntime,
        bg: Any,
        distant: Optional[Any],
        rigid: Optional[Any],
        source_frame_idx: int,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if runtime.bg is not None and runtime.bg_assignment is not None:
            stats.update(
                self._stage2_0_check_parent_runtime_branch_drift(
                    branch_name="bg",
                    runtime=runtime.bg,
                    params=self._stage2_0_params_from_node_state(bg),
                    child_to_parent=runtime.bg_assignment.child_to_parent,
                    parent_count=runtime.bg_assignment.parent_count,
                    child_mass=runtime.bg_assignment.child_mass,
                    child_order=runtime.bg_assignment.child_order,
                    parent_start=runtime.bg_assignment.parent_start,
                )
            )
        if runtime.distant is not None and runtime.distant_assignment is not None and distant is not None:
            stats.update(
                self._stage2_0_check_parent_runtime_branch_drift(
                    branch_name="distant",
                    runtime=runtime.distant,
                    params=self._stage2_0_params_from_node_state(distant),
                    child_to_parent=runtime.distant_assignment.child_to_parent,
                    parent_count=runtime.distant_assignment.parent_count,
                    child_mass=runtime.distant_assignment.child_mass,
                    child_order=runtime.distant_assignment.child_order,
                    parent_start=runtime.distant_assignment.parent_start,
                )
            )
        active = runtime.rigid_active_assignment
        if runtime.rigid_active is not None and active is not None and rigid is not None and int(active.fine_S.numel()) > 0:
            s = active.fine_S.long().to(device=self.device)
            route = self._route_rigid_source_points(rigid, int(source_frame_idx), s)
            stats.update(
                self._stage2_0_check_parent_runtime_branch_drift(
                    branch_name="rigid",
                    runtime=runtime.rigid_active,
                    params={
                        "means": route.means_world_S,
                        "quats": route.quats_world_S,
                        "scales_log": rigid.scales_log.index_select(0, s),
                        "opacity_logit": rigid.opacity_logit.index_select(0, s),
                        "sh_dc": rigid.sh_dc.index_select(0, s),
                        "sh_rest": rigid.sh_rest.index_select(0, s),
                    },
                    child_to_parent=active.child_to_active_parent_S,
                    parent_count=active.active_parent_count,
                    child_mass=active.child_mass_S,
                    child_order=active.active_child_order_S,
                    parent_start=active.active_parent_start,
                )
            )
        return stats

    @staticmethod
    def _stage2_0_parent_runtime_drift_max(stats: Dict[str, float]) -> float:
        return max(
            (
                float(v)
                for k, v in stats.items()
                if k.endswith("_max") and "_stats_" not in k and "_second_anchor_" not in k
            ),
            default=0.0,
        )

    @staticmethod
    def _stage2_0_parent_runtime_drift_detail(stats: Dict[str, float], *, limit: int = 16) -> str:
        detail_items = [
            (float(value), key)
            for key, value in stats.items()
            if key.endswith("_max")
            and "_stats_" not in key
            and "_second_anchor_" not in key
            and float(value) > 0.0
        ]
        detail_items.sort(reverse=True)
        return ", ".join(f"{key}={value:.6g}" for value, key in detail_items[: int(limit)])

    def _stage2_0_maybe_check_parent_runtime_drift(
        self,
        *,
        runtime: Optional[BigGSBlockRuntime],
        bg: Any,
        distant: Optional[Any],
        rigid: Optional[Any],
        source_frame_idx: int,
    ) -> Dict[str, float]:
        if runtime is None:
            return {}
        parent_state_cfg = getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}
        interval = int(self._cfg_get(parent_state_cfg, "drift_check_interval_blocks", 0) or 0)
        if interval <= 0:
            return {}
        block_idx = int(getattr(self, "_stage2_0_biggs_parent_runtime_block_counter", 0))
        if block_idx <= 0 or block_idx % int(interval) != 0:
            return {}
        if int(getattr(self, "_stage2_0_biggs_parent_last_drift_block", -1)) == block_idx:
            return {}
        self._stage2_0_biggs_parent_last_drift_block = block_idx
        stats: Dict[str, float] = {"iforward/biggs/drift_checked": 1.0, "iforward/biggs/drift_block_idx": float(block_idx)}
        stats.update(
            self._stage2_0_parent_runtime_drift_stats_for_nodes(
                runtime=runtime,
                bg=bg,
                distant=distant,
                rigid=rigid,
                source_frame_idx=int(source_frame_idx),
            )
        )
        threshold = float(self._cfg_get(parent_state_cfg, "drift_fail_threshold", 0.0) or 0.0)
        total_max = self._stage2_0_parent_runtime_drift_max(stats)
        stats["iforward/biggs/drift_max"] = float(total_max)
        if threshold > 0.0 and total_max > threshold:
            stats["iforward/biggs/drift_exceeded_threshold"] = 1.0
            detail = self._stage2_0_parent_runtime_drift_detail(stats)
            action = str(self._cfg_get(parent_state_cfg, "drift_fail_action", "raise")).lower()
            if action in {"exact_refresh", "refresh", "warn_and_refresh"}:
                setattr(runtime, "_force_exact_refresh_due_to_drift", True)
                stats["iforward/biggs/drift_exact_refresh_scheduled"] = 1.0
                return stats
            raise RuntimeError(
                f"BigGS parent runtime drift check failed: max_error={total_max:.6g} "
                f"> threshold={threshold:.6g} at block={block_idx}. {detail}"
            )
        stats["iforward/biggs/drift_exceeded_threshold"] = 0.0
        return stats

    @torch.no_grad()
    def _stage2_0_update_parent_runtime(
        self,
        *,
        runtime: Optional[BigGSBlockRuntime],
        old_local_state: LocalGSState,
        new_local_state: LocalGSState,
    ) -> Optional[BigGSBlockRuntime]:
        if runtime is None:
            return None
        parent_state_mode = str(self._cfg_get(getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}, "mode", "none")).lower()
        if parent_state_mode != "incremental_sufficient_stats":
            return runtime
        old_bg, old_distant, old_rigid = old_local_state.to(device=self.device).to_node_states_detached_view()
        new_bg, new_distant, new_rigid = new_local_state.to(device=self.device).to_node_states_detached_view()
        update_backend = str(
            self._cfg_get(getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}, "update_backend", "incremental")
        ).lower()
        force_exact_refresh = bool(getattr(runtime, "_force_exact_refresh_due_to_drift", False))
        if force_exact_refresh or update_backend in {"exact", "exact_refresh", "reference_exact", "reference_refresh"}:
            bg_runtime = None
            if runtime.bg_assignment is not None:
                bg_cfg = self._stage2_0_parent_runtime_cfg_for_branch("bg")
                bg_runtime = self._stage2_0_init_parent_branch_runtime(
                    params=self._stage2_0_params_from_node_state(new_bg),
                    child_to_parent=runtime.bg_assignment.child_to_parent,
                    parent_count=runtime.bg_assignment.parent_count,
                    child_mass=runtime.bg_assignment.child_mass,
                    cfg=bg_cfg,
                    child_order=runtime.bg_assignment.child_order,
                    parent_start=runtime.bg_assignment.parent_start,
                    max_scale=self._stage2_0_biggs_max_scale("bg"),
                    assignment_signature="bg",
                )
            distant_runtime = None
            if runtime.distant_assignment is not None and new_distant is not None:
                distant_cfg = self._stage2_0_parent_runtime_cfg_for_branch("distant")
                distant_runtime = self._stage2_0_init_parent_branch_runtime(
                    params=self._stage2_0_params_from_node_state(new_distant),
                    child_to_parent=runtime.distant_assignment.child_to_parent,
                    parent_count=runtime.distant_assignment.parent_count,
                    child_mass=runtime.distant_assignment.child_mass,
                    cfg=distant_cfg,
                    child_order=runtime.distant_assignment.child_order,
                    parent_start=runtime.distant_assignment.parent_start,
                    max_scale=self._stage2_0_biggs_max_scale("distant"),
                    assignment_signature="distant",
                )
            rigid_runtime = None
            active = runtime.rigid_active_assignment
            if active is not None and new_rigid is not None and int(active.fine_S.numel()) > 0:
                s = active.fine_S.long().to(device=self.device)
                route_new = self._route_rigid_source_points(new_rigid, int(runtime.source_frame_idx), s)
                rigid_cfg = self._stage2_0_parent_runtime_cfg_for_branch("rigid")
                rigid_runtime = self._stage2_0_init_parent_branch_runtime(
                    params={
                        "means": route_new.means_world_S,
                        "quats": route_new.quats_world_S,
                        "scales_log": new_rigid.scales_log.index_select(0, s),
                        "opacity_logit": new_rigid.opacity_logit.index_select(0, s),
                        "sh_dc": new_rigid.sh_dc.index_select(0, s),
                        "sh_rest": new_rigid.sh_rest.index_select(0, s),
                    },
                    child_to_parent=active.child_to_active_parent_S,
                    parent_count=active.active_parent_count,
                    child_mass=active.child_mass_S,
                    cfg=rigid_cfg,
                    child_order=active.active_child_order_S,
                    parent_start=active.active_parent_start,
                    max_scale=self._stage2_0_biggs_max_scale("rigid"),
                    assignment_signature="rigid_active",
                )
            refreshed_runtime = BigGSBlockRuntime(
                bg=bg_runtime,
                distant=distant_runtime,
                rigid_active=rigid_runtime,
                bg_assignment=runtime.bg_assignment,
                distant_assignment=runtime.distant_assignment,
                rigid_active_assignment=runtime.rigid_active_assignment,
                source_frame_idx=int(runtime.source_frame_idx),
                block_id=int(runtime.block_id),
                exact_refresh_count=int(runtime.exact_refresh_count) + 1,
                incremental_update_count=int(runtime.incremental_update_count),
            )
            if force_exact_refresh:
                setattr(refreshed_runtime, "_exact_refresh_due_to_drift", True)
            parent_state_cfg = getattr(self, "stage2_0_biggs_parent_state_cfg", {}) or {}
            interval = int(self._cfg_get(parent_state_cfg, "drift_check_interval_blocks", 0) or 0)
            threshold = float(self._cfg_get(parent_state_cfg, "drift_fail_threshold", 0.0) or 0.0)
            block_id = int(getattr(refreshed_runtime, "block_id", -1))
            if interval > 0 and block_id > 0 and block_id % int(interval) == 0:
                update_drift = self._stage2_0_parent_runtime_drift_stats_for_nodes(
                    runtime=refreshed_runtime,
                    bg=new_bg,
                    distant=new_distant,
                    rigid=new_rigid,
                    source_frame_idx=int(refreshed_runtime.source_frame_idx),
                )
                update_max = self._stage2_0_parent_runtime_drift_max(update_drift)
                if threshold > 0.0 and update_max > threshold:
                    detail = self._stage2_0_parent_runtime_drift_detail(update_drift)
                    raise RuntimeError(
                        f"BigGS parent runtime drift check failed immediately after exact refresh: "
                        f"max_error={update_max:.6g} > threshold={threshold:.6g} "
                        f"at block={block_id}. {detail}"
                    )
            return refreshed_runtime
        if update_backend not in {"incremental", "incremental_sufficient_stats"}:
            raise ValueError(f"unsupported BigGS parent_state.update_backend={update_backend!r}")

        bg_runtime = runtime.bg
        if bg_runtime is not None and runtime.bg_assignment is not None:
            bg_cfg = self._stage2_0_parent_runtime_cfg_for_branch("bg")
            bg_runtime = self._stage2_0_update_parent_branch_runtime(
                runtime=bg_runtime,
                old_params=self._stage2_0_params_from_node_state(old_bg),
                new_params=self._stage2_0_params_from_node_state(new_bg),
                child_to_parent=runtime.bg_assignment.child_to_parent,
                parent_count=runtime.bg_assignment.parent_count,
                child_mass=runtime.bg_assignment.child_mass,
                cfg=bg_cfg,
                child_order=runtime.bg_assignment.child_order,
                parent_start=runtime.bg_assignment.parent_start,
                max_scale=self._stage2_0_biggs_max_scale("bg"),
            )

        distant_runtime = runtime.distant
        if (
            distant_runtime is not None
            and runtime.distant_assignment is not None
            and old_distant is not None
            and new_distant is not None
        ):
            distant_cfg = self._stage2_0_parent_runtime_cfg_for_branch("distant")
            distant_runtime = self._stage2_0_update_parent_branch_runtime(
                runtime=distant_runtime,
                old_params=self._stage2_0_params_from_node_state(old_distant),
                new_params=self._stage2_0_params_from_node_state(new_distant),
                child_to_parent=runtime.distant_assignment.child_to_parent,
                parent_count=runtime.distant_assignment.parent_count,
                child_mass=runtime.distant_assignment.child_mass,
                cfg=distant_cfg,
                child_order=runtime.distant_assignment.child_order,
                parent_start=runtime.distant_assignment.parent_start,
                max_scale=self._stage2_0_biggs_max_scale("distant"),
            )

        rigid_runtime = runtime.rigid_active
        active = runtime.rigid_active_assignment
        if (
            rigid_runtime is not None
            and active is not None
            and old_rigid is not None
            and new_rigid is not None
            and int(active.fine_S.numel()) > 0
        ):
            s = active.fine_S.long().to(device=self.device)
            route_old = self._route_rigid_source_points(old_rigid, int(runtime.source_frame_idx), s)
            route_new = self._route_rigid_source_points(new_rigid, int(runtime.source_frame_idx), s)
            rigid_cfg = self._stage2_0_parent_runtime_cfg_for_branch("rigid")
            rigid_runtime = self._stage2_0_update_parent_branch_runtime(
                runtime=rigid_runtime,
                old_params={
                    "means": route_old.means_world_S,
                    "quats": route_old.quats_world_S,
                    "scales_log": old_rigid.scales_log.index_select(0, s),
                    "opacity_logit": old_rigid.opacity_logit.index_select(0, s),
                    "sh_dc": old_rigid.sh_dc.index_select(0, s),
                    "sh_rest": old_rigid.sh_rest.index_select(0, s),
                },
                new_params={
                    "means": route_new.means_world_S,
                    "quats": route_new.quats_world_S,
                    "scales_log": new_rigid.scales_log.index_select(0, s),
                    "opacity_logit": new_rigid.opacity_logit.index_select(0, s),
                    "sh_dc": new_rigid.sh_dc.index_select(0, s),
                    "sh_rest": new_rigid.sh_rest.index_select(0, s),
                },
                child_to_parent=active.child_to_active_parent_S,
                parent_count=active.active_parent_count,
                child_mass=active.child_mass_S,
                cfg=rigid_cfg,
                child_order=active.active_child_order_S,
                parent_start=active.active_parent_start,
                max_scale=self._stage2_0_biggs_max_scale("rigid"),
            )

        return BigGSBlockRuntime(
            bg=bg_runtime,
            distant=distant_runtime,
            rigid_active=rigid_runtime,
            bg_assignment=runtime.bg_assignment,
            distant_assignment=runtime.distant_assignment,
            rigid_active_assignment=runtime.rigid_active_assignment,
            source_frame_idx=int(runtime.source_frame_idx),
            block_id=int(runtime.block_id),
            exact_refresh_count=int(runtime.exact_refresh_count),
            incremental_update_count=int(runtime.incremental_update_count) + 1,
        )

    @staticmethod
    def _mask_branch_delta(delta: BranchDelta, scope: Dict[str, bool]) -> BranchDelta:
        active_attrs = {
            "means": bool(scope["update_means"]) and delta.is_active("means"),
            "scales_log": bool(scope["update_scales"]) and delta.is_active("scales_log"),
            "quat_axis_angle": bool(scope["update_quat"]) and delta.is_active("quat_axis_angle"),
            "opacity_logit": bool(scope["update_opacity"]) and delta.is_active("opacity_logit"),
            "sh": bool(scope["update_sh"]) and delta.is_active("sh"),
            "hidden": bool(scope.get("update_hidden", True)) and delta.is_active("hidden"),
        }
        return BranchDelta(
            means=delta.means if bool(active_attrs["means"]) else torch.zeros_like(delta.means),
            scales_log=delta.scales_log if bool(active_attrs["scales_log"]) else torch.zeros_like(delta.scales_log),
            quat_axis_angle=(
                delta.quat_axis_angle
                if bool(active_attrs["quat_axis_angle"])
                else torch.zeros_like(delta.quat_axis_angle)
            ),
            opacity_logit=(
                delta.opacity_logit
                if bool(active_attrs["opacity_logit"])
                else torch.zeros_like(delta.opacity_logit)
            ),
            sh=delta.sh if bool(active_attrs["sh"]) else torch.zeros_like(delta.sh),
            hidden=delta.hidden if bool(active_attrs["hidden"]) else torch.zeros_like(delta.hidden),
            confidence=delta.confidence,
            noop=delta.noop,
            active_attrs=active_attrs,
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

    @staticmethod
    def _branch_delta_to_float32(delta: Optional[BranchDelta]) -> Optional[BranchDelta]:
        if delta is None:
            return None

        def cast(value: torch.Tensor) -> torch.Tensor:
            return value.float() if torch.is_tensor(value) and torch.is_floating_point(value) else value

        return BranchDelta(
            means=cast(delta.means),
            scales_log=cast(delta.scales_log),
            quat_axis_angle=cast(delta.quat_axis_angle),
            opacity_logit=cast(delta.opacity_logit),
            sh=cast(delta.sh),
            hidden=cast(delta.hidden),
            confidence=cast(delta.confidence),
            noop=cast(delta.noop),
            active_attrs=delta.active_attrs,
        )

    @classmethod
    def _delta_pack_to_float32(cls, delta: DeltaPack) -> DeltaPack:
        return DeltaPack(
            bg=cls._branch_delta_to_float32(delta.bg),
            distant=cls._branch_delta_to_float32(delta.distant),
            rigid=cls._branch_delta_to_float32(delta.rigid),
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
            active_attrs=delta.active_attrs,
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
        near_aabb_min, near_aabb_max = self._stage6_aabb(feat_2d_bg)

        def clamp_near_coords(coords: torch.Tensor) -> torch.Tensor:
            # The near xCPE voxel layout is strict about the fixed segment AABB.
            # Clamp only the coordinates used for grid indexing; state/render params
            # stay unchanged so routing and supervision semantics are preserved.
            lo = near_aabb_min.to(device=coords.device, dtype=coords.dtype)
            hi = near_aabb_max.to(device=coords.device, dtype=coords.dtype)
            return coords.clamp(min=lo, max=hi)

        feat_parts = [feat_2d_bg]
        acc_parts = [acc_w_bg.reshape(-1)]
        obs_parts = [obs_bg]
        coords_parts = [clamp_near_coords(local_state.bg.means)]
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
            coords_parts.append(clamp_near_coords(route.means_world_S[route.inside_mask_S]))
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
                "debug_check_spconv_order": bool(getattr(self, "stage6_near_debug_check_spconv_order", False)),
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
    def _stage2_0_select_param_rows(
        params: Optional[Dict[str, torch.Tensor]],
        rows: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if params is None:
            return None
        return {
            key: value.index_select(0, rows.to(device=value.device, dtype=torch.long))
            for key, value in params.items()
        }

    def _stage2_0_parent_route_from_measurement(self, measurement: Dict[str, Any]) -> Any:
        ref = measurement["parent_feat_2d_bg"]
        active = measurement.get("assign_rigid_active")
        if active is None or int(active.active_parent_global.numel()) == 0:
            empty_l = torch.zeros((0,), dtype=torch.long, device=ref.device)
            empty_b = torch.zeros((0,), dtype=torch.bool, device=ref.device)
            return SimpleNamespace(
                S=empty_l,
                S_in=empty_l,
                S_out=empty_l,
                inside_mask_S=empty_b,
                means_world_S=ref.new_zeros((0, 3)),
                quats_world_S=ref.new_zeros((0, 4)),
            )
        inside = active.parent_inside_mask.to(device=ref.device, dtype=torch.bool).reshape(-1)
        parent_S = torch.arange(int(inside.numel()), dtype=torch.long, device=ref.device)
        coords = measurement.get("parent_coords_rigid_S")
        params = measurement.get("parent_params_rigid_active")
        if coords is None or params is None:
            raise RuntimeError("BigGS parent route requires active rigid parent coords/params")
        return SimpleNamespace(
            S=parent_S,
            S_in=parent_S[inside],
            S_out=parent_S[~inside],
            inside_mask_S=inside,
            means_world_S=coords.to(device=ref.device, dtype=ref.dtype),
            quats_world_S=params["quats"].to(device=ref.device, dtype=ref.dtype),
        )

    def _build_stage2_0_parent_struct_input_near(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
        parent_route: Any,
    ) -> Stage6StructInput:
        _ = local_state
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        feat_2d_bg = self._maybe_detach_feature(measurement["parent_feat_2d_bg"], detach=detach_features)
        if feat_2d_bg is None:
            raise RuntimeError("BigGS parent near input requires parent_feat_2d_bg")
        acc_w_bg = self._detach_optional(measurement.get("parent_acc_w_bg"))
        obs_bg = self._detach_optional(measurement.get("parent_obs_bg"))
        if acc_w_bg is None or obs_bg is None:
            raise RuntimeError("BigGS parent near input requires parent acc/obs tensors")
        num_bg = int(feat_2d_bg.shape[0])
        if obs_bg.dim() != 2 or int(obs_bg.shape[0]) != num_bg or int(obs_bg.shape[1]) != 2:
            raise ValueError(f"BigGS parent obs_bg must be [M_bg,2], got {tuple(obs_bg.shape)}")
        near_aabb_min, near_aabb_max = self._stage6_aabb(feat_2d_bg)

        def clamp_near_coords(coords: torch.Tensor) -> torch.Tensor:
            lo = near_aabb_min.to(device=coords.device, dtype=coords.dtype)
            hi = near_aabb_max.to(device=coords.device, dtype=coords.dtype)
            return coords.clamp(min=lo, max=hi)

        feat_parts = [feat_2d_bg]
        acc_parts = [acc_w_bg.reshape(-1)]
        obs_parts = [obs_bg]
        coords_parts = [clamp_near_coords(measurement["parent_coords_bg"])]
        branch_ids = [torch.zeros((num_bg,), dtype=torch.long, device=feat_2d_bg.device)]
        params_bg = measurement["parent_params_bg"]

        rows_in = parent_route.S_in.long()
        params_rigid_in = None
        num_rigid_in = int(rows_in.numel())
        if num_rigid_in > 0:
            feat_2d_rigid = self._maybe_detach_feature(measurement.get("parent_feat_2d_rigid_S"), detach=detach_features)
            acc_w_rigid = self._detach_optional(measurement.get("parent_acc_w_rigid_S"))
            obs_rigid = self._detach_optional(measurement.get("parent_obs_rigid_S"))
            if feat_2d_rigid is None or acc_w_rigid is None or obs_rigid is None:
                raise RuntimeError("BigGS parent near input requires active rigid tensors when S_in > 0")
            feat_parts.append(feat_2d_rigid.index_select(0, rows_in.to(device=feat_2d_rigid.device)))
            acc_parts.append(acc_w_rigid.reshape(-1).index_select(0, rows_in.to(device=acc_w_rigid.device)))
            obs_parts.append(obs_rigid.index_select(0, rows_in.to(device=obs_rigid.device)))
            coords_parts.append(clamp_near_coords(parent_route.means_world_S.index_select(0, rows_in)))
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=feat_2d_bg.device))
            params_rigid_in = self._stage2_0_select_param_rows(
                measurement.get("parent_params_rigid_active"),
                rows_in,
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
                "debug_check_spconv_order": bool(getattr(self, "stage6_near_debug_check_spconv_order", False)),
            },
        )

    def _build_stage2_0_parent_struct_input_far(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
        parent_route: Any,
    ) -> Stage6StructInput:
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        ref = measurement["parent_feat_2d_bg"]
        feat_distant_ref = measurement.get("parent_feat_2d_distant")
        include_distant_event = (
            feat_distant_ref is not None
            and measurement.get("parent_params_distant") is not None
            and local_state.distant is not None
            and not self._phase_b_skip_distant_event()
        )
        num_distant = int(feat_distant_ref.shape[0]) if bool(include_distant_event) else 0
        rows_out = parent_route.S_out.long()
        num_rigid_out = int(rows_out.numel())
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
            feat_2d_distant = self._maybe_detach_feature(feat_distant_ref, detach=detach_features)
            acc_w_distant = self._detach_optional(measurement.get("parent_acc_w_distant"))
            obs_distant = self._detach_optional(measurement.get("parent_obs_distant"))
            if feat_2d_distant is None or acc_w_distant is None or obs_distant is None:
                raise RuntimeError("BigGS parent far input expected distant tensors")
            feat_parts.append(feat_2d_distant)
            acc_parts.append(acc_w_distant.reshape(-1))
            obs_parts.append(obs_distant)
            coords_parts.append(measurement["parent_coords_distant"])
            branch_ids.append(torch.zeros((num_distant,), dtype=torch.long, device=ref.device))
            params_for_embed = measurement["parent_params_distant"]

        params_rigid_out = None
        if num_rigid_out > 0:
            feat_2d_rigid = self._maybe_detach_feature(measurement.get("parent_feat_2d_rigid_S"), detach=detach_features)
            acc_w_rigid = self._detach_optional(measurement.get("parent_acc_w_rigid_S"))
            obs_rigid = self._detach_optional(measurement.get("parent_obs_rigid_S"))
            if feat_2d_rigid is None or acc_w_rigid is None or obs_rigid is None:
                raise RuntimeError("BigGS parent far input expected rigid tensors for S_out")
            feat_parts.append(feat_2d_rigid.index_select(0, rows_out.to(device=feat_2d_rigid.device)))
            acc_parts.append(acc_w_rigid.reshape(-1).index_select(0, rows_out.to(device=acc_w_rigid.device)))
            obs_parts.append(obs_rigid.index_select(0, rows_out.to(device=obs_rigid.device)))
            coords_parts.append(parent_route.means_world_S.index_select(0, rows_out))
            branch_ids.append(torch.ones((num_rigid_out,), dtype=torch.long, device=ref.device))
            params_rigid_out = self._stage2_0_select_param_rows(
                measurement.get("parent_params_rigid_active"),
                rows_out,
            )

        if params_for_embed is None:
            params_for_embed = params_rigid_out
        elif params_rigid_out is not None:
            params_for_embed = cat_param_dict(params_for_embed, params_rigid_out)
        if params_for_embed is None:
            raise RuntimeError("BigGS parent far input internal empty params_for_embed")

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

    def _build_stage2_1_parent_struct_input_near(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
        parent_route: Any,
    ) -> ParentStructInput:
        _ = local_state
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        feat_bg = self._maybe_detach_feature(measurement["parent_feat_2d_bg"], detach=detach_features)
        if feat_bg is None:
            raise RuntimeError("Stage2_1 parent near input requires parent_feat_2d_bg")
        support_bg = self._detach_optional(measurement.get("parent_acc_w_bg"))
        if support_bg is None:
            raise RuntimeError("Stage2_1 parent near input requires parent_acc_w_bg")
        num_bg = int(feat_bg.shape[0])
        aabb_min, aabb_max = self._stage6_aabb(feat_bg)

        def clamp_near(coords: torch.Tensor) -> torch.Tensor:
            lo = aabb_min.to(device=coords.device, dtype=coords.dtype)
            hi = aabb_max.to(device=coords.device, dtype=coords.dtype)
            return coords.clamp(min=lo, max=hi)

        feat_parts = [feat_bg]
        support_parts = [support_bg.reshape(-1)]
        coords_parts = [clamp_near(measurement["parent_coords_bg"])]
        branch_ids = [torch.zeros((num_bg,), dtype=torch.long, device=feat_bg.device)]
        params_bg = measurement["parent_params_bg"]
        rows_in = parent_route.S_in.long()
        params_rigid_in = None
        num_rigid_in = int(rows_in.numel())
        if num_rigid_in > 0:
            feat_rigid = self._maybe_detach_feature(measurement.get("parent_feat_2d_rigid_S"), detach=detach_features)
            support_rigid = self._detach_optional(measurement.get("parent_acc_w_rigid_S"))
            if feat_rigid is None or support_rigid is None:
                raise RuntimeError("Stage2_1 parent near input requires active rigid tensors when S_in > 0")
            rows = rows_in.to(device=feat_rigid.device)
            feat_parts.append(feat_rigid.index_select(0, rows))
            support_parts.append(support_rigid.reshape(-1).index_select(0, rows.to(device=support_rigid.device)))
            coords_parts.append(clamp_near(parent_route.means_world_S.index_select(0, rows_in)))
            branch_ids.append(torch.ones((num_rigid_in,), dtype=torch.long, device=feat_bg.device))
            params_rigid_in = self._stage2_0_select_param_rows(measurement.get("parent_params_rigid_active"), rows_in)

        return ParentStructInput(
            parent_context=torch.cat(feat_parts, dim=0),
            support=torch.cat(support_parts, dim=0),
            valid=None,
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=cat_param_dict(params_bg, params_rigid_in),
            split_0=num_bg,
            split_1=num_rigid_in,
            meta={"path": "near"},
        )

    def _build_stage2_1_parent_struct_input_far(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
        parent_route: Any,
    ) -> ParentStructInput:
        detach_features = bool(getattr(self, "stage6_detach_v4_outputs", True))
        ref = measurement["parent_feat_2d_bg"]
        feat_distant_ref = measurement.get("parent_feat_2d_distant")
        include_distant = (
            feat_distant_ref is not None
            and measurement.get("parent_params_distant") is not None
            and local_state.distant is not None
            and not self._phase_b_skip_distant_event()
        )
        num_distant = int(feat_distant_ref.shape[0]) if bool(include_distant) else 0
        rows_out = parent_route.S_out.long()
        num_rigid_out = int(rows_out.numel())
        if num_distant + num_rigid_out == 0:
            return empty_parent_struct_input(
                ref=ref,
                context_dim=int(getattr(self, "stage6_feat_2d_dim", int(ref.shape[-1]))),
                sh_rest_bases=int(local_state.bg.sh_rest.shape[1]),
                path="far",
            )

        feat_parts: List[torch.Tensor] = []
        support_parts: List[torch.Tensor] = []
        coords_parts: List[torch.Tensor] = []
        branch_ids: List[torch.Tensor] = []
        params_for_embed = None
        if bool(include_distant):
            feat_distant = self._maybe_detach_feature(feat_distant_ref, detach=detach_features)
            support_distant = self._detach_optional(measurement.get("parent_acc_w_distant"))
            if feat_distant is None or support_distant is None:
                raise RuntimeError("Stage2_1 parent far input expected distant tensors")
            feat_parts.append(feat_distant)
            support_parts.append(support_distant.reshape(-1))
            coords_parts.append(measurement["parent_coords_distant"])
            branch_ids.append(torch.zeros((num_distant,), dtype=torch.long, device=ref.device))
            params_for_embed = measurement["parent_params_distant"]

        params_rigid_out = None
        if num_rigid_out > 0:
            feat_rigid = self._maybe_detach_feature(measurement.get("parent_feat_2d_rigid_S"), detach=detach_features)
            support_rigid = self._detach_optional(measurement.get("parent_acc_w_rigid_S"))
            if feat_rigid is None or support_rigid is None:
                raise RuntimeError("Stage2_1 parent far input expected rigid tensors for S_out")
            rows = rows_out.to(device=feat_rigid.device)
            feat_parts.append(feat_rigid.index_select(0, rows))
            support_parts.append(support_rigid.reshape(-1).index_select(0, rows.to(device=support_rigid.device)))
            coords_parts.append(parent_route.means_world_S.index_select(0, rows_out))
            branch_ids.append(torch.ones((num_rigid_out,), dtype=torch.long, device=ref.device))
            params_rigid_out = self._stage2_0_select_param_rows(measurement.get("parent_params_rigid_active"), rows_out)

        if params_for_embed is None:
            params_for_embed = params_rigid_out
        elif params_rigid_out is not None:
            params_for_embed = cat_param_dict(params_for_embed, params_rigid_out)
        if params_for_embed is None:
            raise RuntimeError("Stage2_1 parent far input internal empty params_for_embed")

        return ParentStructInput(
            parent_context=torch.cat(feat_parts, dim=0),
            support=torch.cat(support_parts, dim=0),
            valid=None,
            coords=torch.cat(coords_parts, dim=0),
            branch_id=torch.cat(branch_ids, dim=0),
            params_for_embed=params_for_embed,
            split_0=num_distant,
            split_1=num_rigid_out,
            meta={"path": "far"},
        )

    def _build_stage2_1_parent_inputs_from_measurement(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Dict[str, Any]:
        parent_route = self._stage2_0_parent_route_from_measurement(measurement)
        near_in = self._build_stage2_1_parent_struct_input_near(
            local_state=local_state,
            measurement=measurement,
            parent_route=parent_route,
        )
        far_in = self._build_stage2_1_parent_struct_input_far(
            local_state=local_state,
            measurement=measurement,
            parent_route=parent_route,
        )
        aabb_min, aabb_max = self._stage6_aabb(measurement["parent_feat_2d_bg"])
        return {
            "near_in": near_in,
            "far_in": far_in,
            "route": parent_route,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "near_batch_offsets": self._build_struct_batch_offsets(near_in, device=self.device),
            "far_batch_offsets": self._build_struct_batch_offsets(far_in, device=self.device),
        }

    def _decode_stage2_1_biggs_child_event(
        self,
        *,
        parent_event: Any,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Any:
        decoder = getattr(self, "biggs_child_decoder", None)
        if decoder is None:
            raise RuntimeError("Stage2_1 requires runtime.biggs_child_decoder")
        # GRLD fused decode currently supports FP32 only.
        with self._iforward_amp_fp32():
            fine_event = decoder(
                parent_event_pack=parent_event,
                local_state=local_state,
                measurement=measurement,
            )
        fine_event.obs_code_bg = None
        fine_event.obs_code_distant = None
        fine_event.obs_code_rigid = None
        if torch.is_tensor(measurement.get("child_detail_bg")):
            fine_event.appearance_detail = AppearanceDetailPack(
                detail_bg=measurement["child_detail_bg"],
                detail_distant=measurement.get("child_detail_distant"),
                detail_rigid=measurement.get("child_detail_rigid_S"),
                valid_bg=measurement.get("child_detail_valid_bg"),
                valid_distant=measurement.get("child_detail_valid_distant"),
                valid_rigid=measurement.get("child_detail_valid_rigid_S"),
            )
        aux = dict(getattr(fine_event, "aux", {}) or {})
        aux.update({str(k): float(v) for k, v in dict(getattr(parent_event, "aux", {}) or {}).items() if isinstance(v, (int, float))})
        for key, value in measurement.items():
            if not (
                str(key).startswith("iforward/biggs/")
                or str(key).startswith("iforward/fwhr/")
                or str(key).startswith("iforward/stage3/")
                or str(key).startswith("num_parent_")
            ):
                continue
            if isinstance(value, (int, float)):
                aux[str(key)] = float(value)
        if bool(float(measurement.get("iforward/stage3/enabled", 0.0) or 0.0) > 0.0):
            aux.update(self._stage3_0_memory_aux("after_event_decode", include_step_max=True))
        fine_event.aux = aux
        return self._event_with_default_view_code(fine_event)

    def _build_stage2_0_biggs_event_from_measurement(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Any:
        decoder = getattr(self, "biggs_child_decoder", None)
        if decoder is None:
            raise RuntimeError("BigGS Stage 2.0 requires runtime.biggs_child_decoder")
        source_frame_idx = int(measurement.get("source_frame_idx", 0))
        parent_route = self._stage2_0_parent_route_from_measurement(measurement)
        self._mem_debug("encode/biggs_parent_begin", source_frame_idx=source_frame_idx)
        near_in = self._build_stage2_0_parent_struct_input_near(
            local_state=local_state,
            measurement=measurement,
            parent_route=parent_route,
        )
        self._mem_debug("encode/biggs_parent_after_near_input", near_n=int(near_in.coords.shape[0]))
        far_in = self._build_stage2_0_parent_struct_input_far(
            local_state=local_state,
            measurement=measurement,
            parent_route=parent_route,
        )
        self._mem_debug("encode/biggs_parent_after_far_input", far_n=int(far_in.coords.shape[0]))
        aabb_min, aabb_max = self._stage6_aabb(measurement["parent_feat_2d_bg"])
        parent_event = self.stage6_struct_event_decoder(
            near_in=near_in,
            far_in=far_in,
            route=parent_route,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            near_batch_offsets=self._build_struct_batch_offsets(stage6_to_struct_decoder_input(near_in), device=self.device),
            far_batch_offsets=self._build_struct_batch_offsets(stage6_to_struct_decoder_input(far_in), device=self.device),
        )
        # GRLD fused decode currently supports FP32 only.
        with self._iforward_amp_fp32():
            fine_event = decoder(
                parent_event_pack=parent_event,
                local_state=local_state,
                measurement=measurement,
            )
        if torch.is_tensor(measurement.get("child_detail_bg")):
            fine_event.appearance_detail = AppearanceDetailPack(
                detail_bg=measurement["child_detail_bg"],
                detail_distant=measurement.get("child_detail_distant"),
                detail_rigid=measurement.get("child_detail_rigid_S"),
                valid_bg=measurement.get("child_detail_valid_bg"),
                valid_distant=measurement.get("child_detail_valid_distant"),
                valid_rigid=measurement.get("child_detail_valid_rigid_S"),
            )
        aux = dict(getattr(fine_event, "aux", {}) or {})
        for key, value in measurement.items():
            if not (
                str(key).startswith("iforward/biggs/")
                or str(key).startswith("iforward/fwhr/")
                or str(key).startswith("iforward/stage3/")
                or str(key).startswith("num_parent_")
            ):
                continue
            if isinstance(value, (int, float)):
                aux[str(key)] = float(value)
        aux["iforward/biggs/parent_event_rows_bg"] = float(int(parent_event.event_bg.shape[0]))
        aux["iforward/biggs/parent_event_rows_distant"] = (
            float(int(parent_event.event_distant.shape[0])) if parent_event.event_distant is not None else 0.0
        )
        aux["iforward/biggs/parent_event_rows_rigid"] = (
            float(int(parent_event.event_rigid.shape[0])) if parent_event.event_rigid is not None else 0.0
        )
        if bool(float(measurement.get("iforward/stage3/enabled", 0.0) or 0.0) > 0.0):
            aux.update(self._stage3_0_memory_aux("after_event_decode", include_step_max=True))
        fine_event.aux = aux
        self._mem_debug("encode/biggs_after_child_decode")
        return self._event_with_default_view_code(fine_event)

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
        if bool(measurement.get("biggs_enabled", False)):
            return self._build_stage2_0_biggs_event_from_measurement(
                local_state=local_state,
                measurement=measurement,
            )
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
        with self._iforward_amp_autocast():
            delta, aux = self.stage6_posterior_updater(
                event=event,
                ctx_current=None,
                ctx_vsm=ctx_vsm,
                appearance_detail=getattr(event, "appearance_detail", None),
                branch_scope=getattr(self, "stage6_branch_scope", None),
            )
        delta = self._delta_pack_to_float32(delta)
        local_state = local_state.to(device=local_state.bg.means.device, dtype=torch.float32)
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
        return_per_ref_loss: bool = False,
    ) -> tuple:
        if len(target_indices) == 0:
            zero = local_state.bg.means.new_tensor(0.0)
            stats0 = {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
            if bool(return_per_ref_loss):
                return zero, stats0, local_state.bg.means.new_zeros((0,))
            return zero, stats0
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
        per_ref = torch.stack(losses)
        if bool(return_per_ref_loss):
            return per_ref.mean(), stats, per_ref
        return per_ref.mean(), stats

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
