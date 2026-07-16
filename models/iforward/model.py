from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import DeltaPack, LocalGSState

from .amp_policy import build_amp_policy, storage_dtype_from_name
from .adc_lite import (
    ADC_STAT_PREFIX,
    GateSuppressedADCAccumulator,
    adc_bank_stats,
    apply_bg_clone_episode_local,
    build_adc_lite_bank_from_losses,
    build_gate_suppressed_adc_bank,
    ensure_adc_meta_for_state,
)
from .bridge import IForwardStage6Bridge
from .context_adapter import IForwardContextAdapter
from .delta_ops import gate_delta_pack
from .gru_memory import IForwardGRUMemoryState, IForwardTimeAwarePointGRU
from .history_ema import IForwardHistoryEMAState
from .history_gate import IForwardHistoryGate
from .history_gate_v2_features import (
    HGV2_GRAD_FEATURE_DIM,
    compute_history_gate_v2_features,
    history_gate_v2_auxiliary_loss,
)
from .history_gradient_bank import HGV2_ATTRS, build_history_gradient_bank_from_loss
from .history_damage_loss import HistoryDamageProbe, history_damage_hinge
from .history_safe_projection import IForwardHistorySafeProjection
from .iforward_v6_state import IForwardV6MemoryState
from .local_conflict_xcpe import IForwardLocalConflictXcpe
from .memory import IForwardMemoryStepContext, IForwardSceneMemory
from .parent_spatial_backbone import ParentSpatialBackbone
from .observation_feedback import ObservationFeedbackPolicy
from .parent_temporal_keys import ParentTemporalKeys, build_parent_temporal_keys
from .parent_temporal_mamba import ParentTemporalMemory
from .parent_temporal_state import ParentTemporalState
from .stage2_2 import (
    EpisodeHistoryBankV2,
    ParentTemporalMemoryV2,
    ParentTemporalStateV2,
    build_parent_temporal_keys_v2,
    history_damage_hinge_v2,
)
from .stage2_3 import (
    DeltaKVOptimizerBranchState,
    DenseOptimizerState,
    DenseDeltaKVOptimizerState,
    EpisodeHistoryBankV3,
    KeyedOptimizerState,
    KeyedDeltaKVOptimizerState,
    OptimizerBranchState,
    ParentOptimizerDeltaKVState,
    ParentOptimizerGatedDeltaKV,
    ParentOptimizerMamba,
    ParentOptimizerMambaState,
    VisitMeta,
    build_parent_delta_summary,
    history_damage_hinge_v3,
)
from .stage3_0.losses import stage3_gather_regularization
from .point_mamba_memory import IForwardPointMambaMemory
from .resolver import (
    IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS,
    IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
    IFORWARD_STAGE2_1_SCHEDULER_VERSION,
    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
    IFORWARD_STAGE2_3_SCHEDULER_VERSION,
    IFORWARD_STAGE3_0_SCHEDULER_VERSION,
    IFORWARD_STAGE3_2_SCHEDULER_VERSION,
    IFORWARD_V3_SCHEDULER_VERSION,
    IFORWARD_V4_SCHEDULER_VERSION,
    IForwardBatchResolver,
    IForwardResolvedBatch,
)
from .sequence10_history_bank import Sequence10HistoryBank, sequence10_damage_hinge_from_bank
from .sequence10_resolver import IForwardSequence10Resolver
from .state import IForwardMemoryState, IForwardShortWindowHistory, IForwardState
from .utils import cfg_ensure_child, cfg_get, cfg_set, clone_config
from .versions import is_stage3_1_iforward_version, is_stage3_optimizer_memory_iforward_version


def _cfg_set_missing(node: Any, key: str, value: Any) -> None:
    if cfg_get(node, key, None) is None:
        cfg_set(node, key, value)


def _cfg_merge_missing(node: Any, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict):
            child = cfg_ensure_child(node, key)
            _cfg_merge_missing(child, value)
        else:
            _cfg_set_missing(node, key, value)


@dataclass
class IForwardRolloutOutput:
    loss: torch.Tensor
    next_state: IForwardState
    resolved: IForwardResolvedBatch
    per_step: List[Dict[str, float]]
    losses: Dict[str, torch.Tensor]
    stats: Dict[str, Any]
    pred_rgbs: List[torch.Tensor]
    gt_images: List[torch.Tensor]
    image_refs: List[Tuple[int, int]]
    image_roles: List[str]

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "local_G": self.next_state.local_gs,
            "node_state_bg": self.next_state.node_state_bg,
            "node_state_distant": self.next_state.node_state_distant,
            "node_state_rigid": self.next_state.node_state_rigid,
            "roles": self.resolved,
            "per_step": self.per_step,
            "pred_rgbs": list(self.pred_rgbs),
            "gt_images": list(self.gt_images),
            "image_refs": [tuple(x) for x in self.image_refs],
            "image_roles": list(self.image_roles),
            "num_targets": len(self.resolved.target_refs),
            "num_source_views": len(self.resolved.source_refs),
        }


@dataclass
class IForwardFinalRenderRole:
    target_indices: Tuple[int, ...]
    mean_loss: torch.Tensor
    per_ref_loss: torch.Tensor
    stats: Dict[str, float]


@dataclass
class IForwardFinalRenderPack:
    current: IForwardFinalRenderRole
    history: IForwardFinalRenderRole
    nearby: IForwardFinalRenderRole


def _stage6_runtime_config(config: Any) -> Any:
    cfg = clone_config(config)
    root_model = cfg_ensure_child(cfg, "model")
    cfg_set(root_model, "stage", "6_0")
    cfg_set(root_model, "phase", "phase_A_block_local_unroll")
    _cfg_merge_missing(
        root_model,
        {
            "param_embed_dim": 48,
            "offset_gru_hidden_dim": 48,
            "offset_gru_use_reset_gate": True,
            "rigid_routed": {
                "route_space": "source_frame_world",
                "route_aabb": "segment_aabb",
                "inside_decoder": "bg",
                "outside_decoder": "distant",
                "update_means": True,
                "update_quat": True,
            },
            "struct_decoder": {
                "enable": True,
                "type": "routed_near_far",
                "scope": "full_routed",
                "output_role": "gru_input",
                "point_preserving": True,
                "include_bg": True,
                "include_distant": True,
                "include_rigid_in": True,
                "include_rigid_out": True,
                "feat_2d_channels": 24,
                "param_embed_dim": 32,
                "branch_embed_dim": 8,
                "support_embed_dim": 8,
                "history_embed_dim": 16,
                "token": {
                    "use_2d_feat": True,
                    "use_support": True,
                    "use_branch_embed": True,
                    "use_param_embed": True,
                    "use_anchor_rgb": False,
                    "use_hidden_state": False,
                    "zero_invalid_2d_feat": True,
                },
                "near": {
                    "type": "xcpe",
                    "branches": ["bg", "rigid_in"],
                    "channels": 48,
                    "voxel_size": 0.20,
                    "sparse_backend": "spconv",
                    "clamp_grid_coord": False,
                    "xcpe": {
                        "num_layers": 2,
                        "kernel_size": 3,
                        "residual_scale_init": 5.0e-3,
                        "norm": "layernorm",
                        "act": "gelu",
                    },
                },
                "far": {
                    "type": "mlp",
                    "branches": ["distant", "rigid_out"],
                    "channels": 48,
                    "hidden_dim": 48,
                    "num_layers": 2,
                    "norm": "layernorm",
                    "act": "gelu",
                    "history_embed": {"enable": False},
                },
            },
            "history_memory": {
                "enable": False,
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
            },
            "view_transient": {
                "enable": False,
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
            },
            "update_gate": {
                "enable": False,
                "type": "attribute_5",
                "hidden_dim": 48,
                "require_initialized_in_input": True,
                "include_visible_now": True,
                "bind_with_mask_update": True,
                "min_gate": {"means": 0.03, "scales": 0.03, "quat": 0.01, "opacity": 0.05, "sh": 0.05},
                "init_bias": {"means": -1.40, "scales": -1.70, "quat": -2.00, "opacity": -0.40, "sh": 0.40},
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
            },
        },
    )
    branches = cfg_ensure_child(root_model, "branches")
    _cfg_merge_missing(
        cfg_ensure_child(branches, "bg"),
        {
            "src_backproject_support_min": 1.0e-2,
            "enable_selective_update": False,
            "freeze_means": False,
            "init": {
                "scale_init": {
                    "mode": "knn",
                    "isotropic_log_value": -2.30,
                    "knn_k": 8,
                    "knn_log_scale_bias": -1.5,
                },
                "opacity_init": 0.1,
            },
            "limits": {
                "offset_max": 0.1,
                "scale_max": 0.1,
                "omega_max": 0.1,
                "opacity_max": 0.15,
                "sh_dc_max": 0.15,
                "sh_rest_max": 0.08,
            },
            "eta": {"means": 1.0, "scales": 1.0, "opacity": 1.0, "sh_dc": 1.0, "sh_rest": 1.0},
            "mlp": {"hidden_dim": 48, "use_3d_feat": True, "use_2d_feat": True},
        },
    )
    _cfg_merge_missing(
        cfg_ensure_child(branches, "distant"),
        {
            "src_backproject_support_min": 1.0e-2,
            "enable_selective_update": False,
            "freeze_means": True,
            "init": {
                "scale_init": {
                    "mode": "isotropic",
                    "isotropic_log_value": -1.90,
                    "knn_k": 8,
                    "knn_log_scale_bias": -1.2,
                },
                "opacity_init": 0.05,
            },
            "limits": {
                "offset_max": 0.02,
                "scale_max": 0.08,
                "omega_max": 0.01,
                "opacity_max": 0.08,
                "sh_dc_max": 0.12,
                "sh_rest_max": 0.03,
            },
            "eta": {"means": 0.2, "scales": 0.7, "opacity": 0.8, "sh_dc": 1.0, "sh_rest": 0.6},
            "mlp": {"hidden_dim": 48, "use_3d_feat": False, "use_2d_feat": True, "freeze_quat": True},
        },
    )
    _cfg_merge_missing(
        cfg_ensure_child(branches, "rigid"),
        {
            "src_backproject_support_min": 1.0e-2,
            "init": {
                "scale_init": {
                    "mode": "knn",
                    "isotropic_log_value": -2.90,
                    "knn_k": 8,
                    "knn_log_scale_bias": -1.5,
                },
                "opacity_init": 0.2,
            },
            "eta": {"means": 0.5, "scales": 0.8, "opacity": 1.0, "sh_dc": 1.0, "sh_rest": 0.6},
        },
    )
    current_observation = cfg_ensure_child(cfg, "current_observation")
    _cfg_merge_missing(
        current_observation,
        {
            "enable": True,
            "dim": 2,
            "rho_source": "feature",
            "eps": 1.0e-6,
            "input_to_struct_decoder": False,
            "input_to_far_mlp": False,
            "input_to_gru": False,
            "input_to_history_gate": False,
            "record_to_history_memory": False,
        },
    )
    scheduler_v9 = cfg_ensure_child(cfg, "scheduler_v9")
    cfg_set(scheduler_v9, "enable", True)
    cfg_set(scheduler_v9, "phase", "phase_A_block_local_unroll")
    cfg_ensure_child(scheduler_v9, "phase_A")
    block = cfg_ensure_child(scheduler_v9, "block")
    if cfg_get(block, "steps_per_block", None) is None:
        cfg_set(block, "steps_per_block", 1)
    episode = cfg_ensure_child(scheduler_v9, "episode")
    scheduler_iforward = cfg_get(cfg, "scheduler_iforward", {}) or {}
    ifwd_episode = cfg_get(scheduler_iforward, "episode", {}) or {}
    if cfg_get(episode, "blocks_per_episode", None) is None:
        cfg_set(episode, "blocks_per_episode", int(cfg_get(ifwd_episode, "blocks_per_episode", 8)))
    defaults_episode = {
        "include_source_frame": True,
        "target_policy": "visited_episode_frames",
        "reset_policy": "episode_end",
        "block_source_frame_policy": "random_within_keyframe_per_visit",
        "frame_within_keyframe_policy": "random_once_per_episode",
        "min_keyframes_required_policy": "use_available_if_less_than_window",
        "source_mode": "keyframes",
    }
    for key, value in defaults_episode.items():
        if cfg_get(episode, key, None) is None:
            cfg_set(episode, key, value)
    traversal = cfg_ensure_child(scheduler_v9, "traversal")
    ifwd_traversal = cfg_get(scheduler_iforward, "traversal", {}) or {}
    defaults_traversal = {
        "mode": "round_robin_episode_interleave",
        "switch_after_episode": True,
        "fixed_scene_id": cfg_get(ifwd_traversal, "fixed_scene_id", None),
        "fixed_segment_id": cfg_get(ifwd_traversal, "fixed_segment_id", None),
        "segment_order": "ascending",
        "scene_order": "shuffle_per_epoch",
    }
    for key, value in defaults_traversal.items():
        if cfg_get(traversal, key, None) is None:
            cfg_set(traversal, key, value)
    execution = cfg_ensure_child(scheduler_v9, "execution")
    defaults_execution = {
        "block_order": "step_major",
        "step_major_switch_interval_steps": 4,
        "reset_policy": "episode_end",
    }
    for key, value in defaults_execution.items():
        if cfg_get(execution, key, None) is None:
            cfg_set(execution, key, value)
    preload = cfg_ensure_child(scheduler_v9, "preload")
    defaults_preload = {
        "emit_hints": True,
        "warm_next_block_exact": True,
        "warm_next_episode_chain": True,
    }
    for key, value in defaults_preload.items():
        if cfg_get(preload, key, None) is None:
            cfg_set(preload, key, value)
    return cfg


def _build_stage6_runtime(config: Any, device: torch.device) -> Any:
    from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0

    return MinimalStreetForwardStage6_0(config=_stage6_runtime_config(config), device=device)


def _loss_float(config: Any, path: List[str], default: float) -> float:
    node = config
    for key in path[:-1]:
        node = cfg_get(node, key, {}) or {}
    return float(cfg_get(node, path[-1], default))


class IForwardModel(nn.Module):
    """Independent IForward rollout model.

    Public API is IForward-specific. A Stage6 runtime may be held privately only
    to supply V4 observation/render/updater primitives.
    """

    @staticmethod
    def _validate_v3_scheduler_contract(config: Any) -> None:
        scheduler_cfg = cfg_get(config, "scheduler_iforward", {}) or {}
        if not bool(cfg_get(scheduler_cfg, "enable", False)):
            return
        version = str(cfg_get(scheduler_cfg, "version", ""))
        if version in {
            IFORWARD_V3_SCHEDULER_VERSION,
            IFORWARD_V4_SCHEDULER_VERSION,
            IFORWARD_STAGE2_1_SCHEDULER_VERSION,
            IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            IFORWARD_STAGE2_3_SCHEDULER_VERSION,
            IFORWARD_STAGE3_0_SCHEDULER_VERSION,
            IFORWARD_STAGE3_2_SCHEDULER_VERSION,
        }:
            return
        raise ValueError(
            "IForward v3/v4/stage2_1 requires scheduler_iforward.version=iforward_v3_random_window, "
            "iforward_v4_coverage_ordered, iforward_stage2_1_parent_temporal, "
            "iforward_sequence10_v1, iforward_stage2_2_stream10_rawframe, "
            "stage3_0_optimizer_sequence_v1, or stage3_2_distributional_episode_v1 "
            f"when scheduler_iforward is enabled, got {version!r}."
        )

    def __init__(
        self,
        config: Any = None,
        device: Optional[torch.device] = None,
        *,
        bridge: Optional[Any] = None,
        phase_a_runtime: Optional[Any] = None,
        resolver: Optional[IForwardBatchResolver] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_policy = build_amp_policy(config, inference_only=True)
        if resolver is not None:
            self.resolver = resolver
        else:
            random_window_cfg = cfg_get(config, "scheduler_iforward_random_window", {}) or {}
            if bool(cfg_get(random_window_cfg, "enable", False)):
                from .random_window_resolver import IForwardRandomWindowBatchResolver

                self.resolver = IForwardRandomWindowBatchResolver()
            else:
                scheduler32_cfg = cfg_get(config, "scheduler_stage3_2", None)
                scheduler30_cfg = cfg_get(config, "scheduler_stage3_0", None)
                scheduler23_cfg = (
                    scheduler32_cfg
                    if scheduler32_cfg is not None and bool(cfg_get(scheduler32_cfg, "enable", False))
                    else (
                        scheduler30_cfg
                        if scheduler30_cfg is not None and bool(cfg_get(scheduler30_cfg, "enable", False))
                        else (cfg_get(config, "scheduler_v3", {}) or {})
                    )
                )
                scheduler22_cfg = cfg_get(config, "scheduler_stage2_2", {}) or {}
                scheduler_cfg = cfg_get(config, "scheduler_iforward", {}) or {}
                if bool(cfg_get(scheduler23_cfg, "enable", False)) and str(
                    cfg_get(scheduler23_cfg, "version", "")
                ) in {"optimizer_sequence_v1", IFORWARD_STAGE3_0_SCHEDULER_VERSION, IFORWARD_STAGE3_2_SCHEDULER_VERSION}:
                    from datasets.iforward_stage2_3.resolver import Stage23BatchResolver

                    self.resolver = Stage23BatchResolver()
                    if scheduler23_cfg is scheduler32_cfg:
                        scheduler_version = IFORWARD_STAGE3_2_SCHEDULER_VERSION
                    elif scheduler23_cfg is scheduler30_cfg:
                        scheduler_version = IFORWARD_STAGE3_0_SCHEDULER_VERSION
                    else:
                        scheduler_version = IFORWARD_STAGE2_3_SCHEDULER_VERSION
                elif bool(cfg_get(scheduler22_cfg, "enable", False)):
                    from datasets.iforward_stage2_2.resolver import Stage22BatchResolver

                    self.resolver = Stage22BatchResolver()
                    scheduler_version = IFORWARD_STAGE2_2_SCHEDULER_VERSION
                else:
                    scheduler_version = str(cfg_get(scheduler_cfg, "version", "iforward_v1"))
                if scheduler_version == IFORWARD_SEQUENCE10_SCHEDULER_VERSION:
                    self.resolver = IForwardSequence10Resolver()
                elif scheduler_version in {
                    IFORWARD_V3_SCHEDULER_VERSION,
                    IFORWARD_V4_SCHEDULER_VERSION,
                    IFORWARD_STAGE2_1_SCHEDULER_VERSION,
                    IFORWARD_STAGE2_2_SCHEDULER_VERSION,
                    IFORWARD_STAGE2_3_SCHEDULER_VERSION,
                    IFORWARD_STAGE3_0_SCHEDULER_VERSION,
                    IFORWARD_STAGE3_2_SCHEDULER_VERSION,
                }:
                    if scheduler_version not in {IFORWARD_STAGE2_2_SCHEDULER_VERSION, *IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS}:
                        self.resolver = IForwardBatchResolver(expected_scheduler_version=scheduler_version)
                elif scheduler_version not in {IFORWARD_STAGE2_2_SCHEDULER_VERSION, *IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS}:
                    self.resolver = IForwardBatchResolver()

        if bridge is None:
            runtime = phase_a_runtime
            if runtime is None:
                if config is None:
                    raise ValueError("IForwardModel requires config when bridge/runtime is not provided.")
                runtime = _build_stage6_runtime(config, self.device)
            if isinstance(runtime, nn.Module):
                self.phase_a_runtime = runtime
            bridge = IForwardStage6Bridge(runtime)
        if isinstance(bridge, nn.Module):
            self.bridge_module = bridge
        self.bridge = bridge

        event_dim = int(getattr(self.bridge, "event_dim", 48))
        iforward_cfg = cfg_get(cfg_get(config, "model", {}) or {}, "iforward", {}) or {}
        self.observation_feedback_policy = ObservationFeedbackPolicy.from_config(config or {})
        self.iforward_version = str(cfg_get(iforward_cfg, "version", "v1"))
        self.is_stage3_1_lowrank_gdkv = is_stage3_1_iforward_version(self.iforward_version)
        self.is_stage3_0_full_sparse_gather_lift = is_stage3_optimizer_memory_iforward_version(self.iforward_version)
        self.is_v3_gru_history_gate = self.iforward_version == "v3_gru_history_gate"
        self.is_v6_point_mamba_xcpe = self.iforward_version == "v6_point_mamba_xcpe"
        self.is_stage2_1_parent_temporal = self.iforward_version == "stage2_1_fwhr_parent_ptv3_temporal_mamba"
        self.is_stage2_2_parent_temporal = self.iforward_version == "stage2_2_stream10_rawframe_temporal_mamba_v2"
        self.is_stage2_3_optimizer_mamba = self.iforward_version in {
            "iforward_2_3_optimizer_mamba",
        } or bool(is_stage3_optimizer_memory_iforward_version(self.iforward_version))
        self.is_stage2_0_biggs_parent_lifting = self.iforward_version in {
            "stage2_0_biggs_parent_lifting",
            "stage2_0_biggs_cuda_exact_diagonal_projector",
            "stage2_0_biggs_incremental_whdd",
            "stage2_0_biggs_compact16_residualonly",
            "stage2_0_biggs_grld_dinov2base_concat48",
            "stage2_0_fwhr_lift_grld_dinov2base",
            "stage2_1_fwhr_parent_ptv3_temporal_mamba",
            "stage2_2_stream10_rawframe_temporal_mamba_v2",
            "iforward_2_3_optimizer_mamba",
        } or bool(is_stage3_optimizer_memory_iforward_version(self.iforward_version))
        self.history_safe_projection = None
        self.adc_lite_cfg = cfg_get(iforward_cfg, "adc_lite", {}) or {}
        self.adc_lite_enabled = bool(cfg_get(self.adc_lite_cfg, "enable", False))
        self.adc_lite_version = str(cfg_get(self.adc_lite_cfg, "version", "fixed_score_v1"))
        self.adc_lite_gate_suppressed = self.adc_lite_version.lower() in {
            "gate_suppressed_update",
            "gate_suppressed_update_v1",
        }
        if bool(self.adc_lite_enabled) and not self.is_v3_gru_history_gate:
            raise ValueError("model.iforward.adc_lite.enable currently requires version=v3_gru_history_gate")
        if self.is_stage2_0_biggs_parent_lifting:
            biggs_cfg = cfg_get(iforward_cfg, "biggs", {}) or {}
            if bool(cfg_get(biggs_cfg, "enable", True)) is not True:
                raise ValueError(f"{self.iforward_version} requires model.iforward.biggs.enable=true")
            if bool(cfg_get(cfg_get(iforward_cfg, "history_gate", {}) or {}, "enable", False)):
                raise ValueError("stage2_0_biggs_parent_lifting requires history_gate.enable=false")
            if bool(cfg_get(cfg_get(iforward_cfg, "history_gate_v2", {}) or {}, "enable", False)):
                raise ValueError("stage2_0_biggs_parent_lifting requires history_gate_v2.enable=false")
            if bool(self.adc_lite_enabled):
                raise ValueError("stage2_0_biggs_parent_lifting requires adc_lite.enable=false")
            observe_cfg = cfg_get(biggs_cfg, "observe", {}) or {}
            lifting_cfg = cfg_get(biggs_cfg, "lifting", {}) or {}
            if self.is_stage3_0_full_sparse_gather_lift:
                stage3_lifting_cfg = cfg_get(iforward_cfg, "lifting", None)
                if stage3_lifting_cfg is None:
                    raise ValueError("Stage3_0 requires model.iforward.lifting.")
                if str(cfg_get(stage3_lifting_cfg, "type", "")).lower() != "full_sparse_gather":
                    raise ValueError("Stage3_0 requires model.iforward.lifting.type=full_sparse_gather.")
                if str(cfg_get(lifting_cfg, "type", "")).lower() == "fwhr":
                    raise ValueError("Stage3_0 forbids legacy model.iforward.biggs.lifting.type=fwhr.")
                scalar_anchor_backend = str(cfg_get(stage3_lifting_cfg, "scalar_anchor_backend", "cuda_scalar_anchor")).lower()
                if scalar_anchor_backend not in {"projected_meta", "cuda_scalar_anchor"}:
                    raise ValueError(f"Stage3_0 unsupported scalar_anchor_backend={scalar_anchor_backend!r}.")
                parent_lift_cfg = cfg_get(stage3_lifting_cfg, "parent", {}) or {}
                parent_lift_type = str(cfg_get(parent_lift_cfg, "type", "legacy_direct_lift")).lower()
                if parent_lift_type not in {"legacy_direct_lift", "sparse_gather"}:
                    raise ValueError(f"Stage3_0 unsupported parent.type={parent_lift_type!r}.")
                if parent_lift_type == "legacy_direct_lift" and bool(
                    cfg_get(cfg_get(stage3_lifting_cfg, "dino_native", {}) or {}, "enable", False)
                ):
                    raise ValueError("Stage3_0 parent.type=legacy_direct_lift forbids dino_native.enable=true.")
            is_fwhr = str(cfg_get(lifting_cfg, "type", "")).lower() == "fwhr"
            if (
                not bool(self.is_stage3_0_full_sparse_gather_lift)
                and not bool(is_fwhr)
                and bool(cfg_get(observe_cfg, "parent_scene_for_lifting", True)) is not True
            ):
                raise ValueError("stage2_0_biggs_parent_lifting requires parent_scene_for_lifting=true")
            skip_cfg = cfg_get(biggs_cfg, "child_observation_skip", {}) or {}
            if bool(cfg_get(skip_cfg, "enable", False)) and (
                bool(cfg_get(skip_cfg, "trainable", False)) or not bool(cfg_get(skip_cfg, "no_grad", True))
            ):
                raise ValueError("stage2_0_biggs_parent_lifting forbids trainable child_observation_skip")
            parent_state_cfg = cfg_get(biggs_cfg, "parent_state", {}) or {}
            exact_refresh_policy = str(cfg_get(parent_state_cfg, "exact_refresh_policy", "block_enter")).lower()
            if exact_refresh_policy not in {"block_enter", "none"}:
                raise ValueError(
                    "stage2_0 BigGS parent_state.exact_refresh_policy currently supports "
                    f"'block_enter' or 'none', got {exact_refresh_policy!r}."
                )
            self.stage2_0_biggs_parent_exact_refresh_policy = exact_refresh_policy
            self.stage2_0_biggs_update_after_each_nonfinal_repeat = bool(
                cfg_get(parent_state_cfg, "update_after_each_nonfinal_repeat", True)
            )
            self.stage2_0_biggs_skip_update_on_block_exit = bool(
                cfg_get(parent_state_cfg, "skip_update_on_block_exit", True)
            )
        if self.is_v3_gru_history_gate:
            self._validate_v3_scheduler_contract(config)
        debug_cfg = cfg_get(iforward_cfg, "debug", {}) or {}
        self.enable_nvtx_ranges = bool(cfg_get(debug_cfg, "nvtx_ranges", False))
        memory_cfg = cfg_get(iforward_cfg, "memory", {}) or {}
        if self.is_v3_gru_history_gate:
            point_gru_cfg = cfg_get(iforward_cfg, "point_gru", {}) or {}
            ctx_dim = int(cfg_get(point_gru_cfg, "ctx_dim", 48))
            self.point_gru = IForwardTimeAwarePointGRU(
                event_dim=event_dim,
                hidden_dim=int(cfg_get(point_gru_cfg, "hidden_dim", 48)),
                ctx_dim=ctx_dim,
                dt_clip=float(cfg_get(point_gru_cfg, "dt_clip", 32.0)),
                hard_valid_required=bool(cfg_get(point_gru_cfg, "hard_valid_required", True)),
                hard_support_min_optimizer=float(cfg_get(point_gru_cfg, "hard_support_min_optimizer", 0.0)),
            )
            history_cfg_v3 = cfg_get(iforward_cfg, "history_memory", {}) or {}
            support_cfg = cfg_get(history_cfg_v3, "support", {}) or {}
            residual_cfg = cfg_get(history_cfg_v3, "residual", {}) or {}
            update_cfg = cfg_get(history_cfg_v3, "update", {}) or {}
            self.v3_history_support_betas = {
                "fast_beta_visible": float(cfg_get(support_cfg, "fast_ema_beta_visible", 0.35)),
                "fast_beta_invisible": float(cfg_get(support_cfg, "fast_ema_beta_invisible", 0.60)),
                "slow_beta_visible": float(cfg_get(support_cfg, "slow_ema_beta_visible", 0.90)),
                "slow_beta_invisible": float(cfg_get(support_cfg, "slow_ema_beta_invisible", 0.95)),
            }
            self.v3_history_residual_betas = {
                "fast_beta": float(cfg_get(residual_cfg, "fast_error_beta", 0.35)),
                "slow_beta": float(cfg_get(residual_cfg, "slow_error_beta", 0.90)),
            }
            self.v3_history_update_betas = {
                "fast_beta": float(cfg_get(update_cfg, "fast_ema_beta", 0.45)),
                "slow_beta": float(cfg_get(update_cfg, "slow_ema_beta", 0.92)),
            }
            support_min_cfg = cfg_get(history_cfg_v3, "support_min", {}) or {}
            self.v3_history_support_min = {
                "bg": float(cfg_get(support_min_cfg, "bg", 0.0)),
                "distant": float(cfg_get(support_min_cfg, "distant", 0.0)),
                "rigid": float(cfg_get(support_min_cfg, "rigid", 0.0)),
            }
            hgv2_cfg = cfg_get(iforward_cfg, "history_gate_v2", {}) or {}
            self.history_gate_v2_cfg = hgv2_cfg
            self.history_gate_v2_enabled = bool(cfg_get(hgv2_cfg, "enable", False))
            hgv2_features_cfg = cfg_get(hgv2_cfg, "features", {}) or {}
            gate_cfg = cfg_get(iforward_cfg, "history_gate", {}) or {}
            hidden_gate_cfg = cfg_get(cfg_get(gate_cfg, "hidden_gate", {}) or {}, "weights", {}) or {}
            self.history_gate = IForwardHistoryGate(
                event_dim=event_dim,
                ctx_dim=ctx_dim,
                history_embed_dim=int(cfg_get(gate_cfg, "history_embed_dim", 16)),
                hidden_dim=int(cfg_get(gate_cfg, "hidden_dim", 64)),
                branch_embed_dim=int(cfg_get(gate_cfg, "branch_embed_dim", 8)),
                min_gate=dict(cfg_get(gate_cfg, "min_gate", {}) or {}),
                init_bias=dict(cfg_get(gate_cfg, "init_bias", {}) or {}),
                branch_bias=dict(cfg_get(gate_cfg, "branch_bias", {}) or {}),
                hidden_gate_weights=dict(hidden_gate_cfg),
                cold_open_uninitialized=bool(cfg_get(gate_cfg, "cold_open_uninitialized", True)),
                bind_with_mask_update=bool(cfg_get(gate_cfg, "bind_with_mask_update", True)),
                support_min=self.v3_history_support_min,
                grad_feature_dim=HGV2_GRAD_FEATURE_DIM if bool(self.history_gate_v2_enabled) else 0,
                grad_embed_dim=int(cfg_get(hgv2_features_cfg, "grad_embed_dim", 16)),
                grad_prior_scale_init=float(cfg_get(hgv2_features_cfg, "grad_prior_scale_init", 0.0)),
            )
            hsp_cfg = cfg_get(iforward_cfg, "history_safe_projection", {}) or {}
            self.history_safe_projection = (
                IForwardHistorySafeProjection(hsp_cfg)
                if bool(cfg_get(hsp_cfg, "enable", False)) and not bool(self.history_gate_v2_enabled)
                else None
            )
            self.memory = None
        elif self.is_v6_point_mamba_xcpe:
            self.history_gate_v2_cfg = {}
            self.history_gate_v2_enabled = False
            point_cfg = cfg_get(memory_cfg, "point_mamba", {}) or {}
            write_policy_cfg = cfg_get(point_cfg, "write_policy", {}) or {}
            point_ctx_dim = int(cfg_get(point_cfg, "output_dim", cfg_get(point_cfg, "model_dim", 16)))
            long_write_policy = str(
                cfg_get(
                    write_policy_cfg,
                    "update_optimizer_memory",
                    cfg_get(point_cfg, "long_write_policy", "every_repeat"),
                )
            )
            self.point_mamba = IForwardPointMambaMemory(
                event_dim=event_dim,
                point_ctx_dim=point_ctx_dim,
                model_dim=int(cfg_get(point_cfg, "model_dim", 16)),
                state_dim=int(cfg_get(point_cfg, "state_dim", 4)),
                conv_kernel=int(cfg_get(point_cfg, "conv_kernel", 2)),
                dense_bg=bool(cfg_get(point_cfg, "dense_point_memory", cfg_get(memory_cfg, "dense_point_memory", True))),
                dense_distant=bool(cfg_get(point_cfg, "dense_point_memory", cfg_get(memory_cfg, "dense_point_memory", True))),
                hard_valid_required=bool(cfg_get(write_policy_cfg, "hard_valid_required", True)),
                hard_support_min_commit=float(cfg_get(write_policy_cfg, "hard_support_min_commit", 0.0)),
                hard_support_min_optimizer=float(cfg_get(write_policy_cfg, "hard_support_min_optimizer", 0.0)),
                long_write_policy=long_write_policy,
                learnable_soft_gate=bool(cfg_get(write_policy_cfg, "learnable_soft_gate", False)),
            )
            local_cfg = cfg_get(iforward_cfg, "local_conflict", {}) or {}
            self.local_conflict = IForwardLocalConflictXcpe(
                event_dim=event_dim,
                point_ctx_dim=point_ctx_dim,
                hidden_dim=int(cfg_get(local_cfg, "hidden_dim", event_dim)),
                output_dim=int(cfg_get(local_cfg, "output_dim", event_dim)),
                num_blocks=int(cfg_get(local_cfg, "num_blocks", 1)),
                kernel_size=int(cfg_get(local_cfg, "kernel_size", 3)),
                voxel_size=float(cfg_get(local_cfg, "voxel_size", 0.25)),
                sparse_backend=str(cfg_get(local_cfg, "sparse_backend", "spconv")),
            )
            adapter_cfg = cfg_get(iforward_cfg, "context_adapter", {}) or {}
            self.context_adapter = IForwardContextAdapter(
                event_dim=event_dim,
                point_ctx_dim=point_ctx_dim,
                local_ctx_dim=int(cfg_get(local_cfg, "output_dim", event_dim)),
                output_dim=int(cfg_get(adapter_cfg, "output_dim", event_dim)),
                output_scale_init=float(cfg_get(adapter_cfg, "output_scale_init", 1.0)),
                output_scale_learnable=bool(cfg_get(adapter_cfg, "output_scale_learnable", False)),
            )
            self.memory = None
        elif self.is_stage2_0_biggs_parent_lifting:
            self.history_gate_v2_cfg = {}
            self.history_gate_v2_enabled = False
            self.memory = None
            self.history_safe_projection = None
            if self.is_stage2_1_parent_temporal or self.is_stage2_2_parent_temporal or self.is_stage2_3_optimizer_mamba:
                parent_spatial_cfg = cfg_get(iforward_cfg, "parent_spatial", {}) or {}
                parent_ptv3_cfg = cfg_get(parent_spatial_cfg, "ptv3", {}) or {}
                parent_support_cfg = cfg_get(parent_spatial_cfg, "support_threshold", {}) or {}
                self.parent_spatial_backbone = ParentSpatialBackbone(
                    context_dim=int(cfg_get(parent_spatial_cfg, "context_dim", 48)),
                    event_dim=int(cfg_get(parent_spatial_cfg, "event_dim", 64)),
                    token_dim=int(cfg_get(parent_spatial_cfg, "token_dim", cfg_get(parent_spatial_cfg, "event_dim", 64))),
                    param_support_dim=int(cfg_get(parent_spatial_cfg, "param_support_dim", 24)),
                    param_codec_detach_params=not bool(
                        self.observation_feedback_policy.enable
                        and self.observation_feedback_policy.parent_projection.enable
                    ),
                    param_codec_detach_support=True,
                    support_embed_dim=int(cfg_get(parent_spatial_cfg, "support_embed_dim", 4)),
                    branch_embed_dim=int(cfg_get(parent_spatial_cfg, "branch_embed_dim", 4)),
                    near_depth=int(cfg_get(parent_ptv3_cfg, "depth", 4)),
                    near_heads=int(cfg_get(parent_ptv3_cfg, "num_heads", 4)),
                    near_patch_size=int(cfg_get(parent_ptv3_cfg, "patch_size", 64)),
                    near_orders=tuple(cfg_get(parent_ptv3_cfg, "orders", ("z", "z_trans"))),
                    support_threshold_bg=float(cfg_get(parent_support_cfg, "bg", 0.0)),
                    support_threshold_distant=float(cfg_get(parent_support_cfg, "distant", 0.0)),
                    support_threshold_rigid=float(cfg_get(parent_support_cfg, "rigid", 0.0)),
                    support_threshold_rigid_out=float(cfg_get(parent_support_cfg, "rigid_out", cfg_get(parent_support_cfg, "rigid", 0.0))),
                    xcpe_backend=str(cfg_get(parent_ptv3_cfg, "xcpe_backend", "fallback_neighbor_mean")),
                    xcpe_voxel_size=float(cfg_get(parent_ptv3_cfg, "xcpe_voxel_size", 0.5)),
                    use_xcpe=bool(cfg_get(parent_ptv3_cfg, "use_xcpe", True)),
                    zero_invalid_context=bool(cfg_get(parent_spatial_cfg, "zero_invalid_context", True)),
                )
                if self.is_stage2_3_optimizer_mamba:
                    parent_optimizer_cfg = (
                        cfg_get(iforward_cfg, "parent_optimizer_memory", None)
                        or cfg_get(iforward_cfg, "parent_optimizer_mamba", None)
                        or cfg_get(iforward_cfg, "parent_temporal_mamba_v2", None)
                        or cfg_get(iforward_cfg, "parent_temporal_mamba", {})
                        or {}
                    )
                    write_mask_cfg = cfg_get(parent_optimizer_cfg, "write_mask", {}) or {}
                    write_token_cfg = cfg_get(parent_optimizer_cfg, "write_token", {}) or {}
                    fusion_cfg = cfg_get(parent_optimizer_cfg, "fusion", {}) or {}
                    memory_type = str(
                        cfg_get(
                            parent_optimizer_cfg,
                            "type",
                            "lowrank_gated_delta_kv" if bool(self.is_stage3_1_lowrank_gdkv) else "mamba",
                        )
                    ).lower()
                    self.stage2_3_include_delta_summary = bool(cfg_get(write_token_cfg, "include_delta_summary", True))
                    include_spatial_event = bool(cfg_get(write_token_cfg, "include_spatial_event", True))
                    include_parent_event = bool(cfg_get(write_token_cfg, "include_parent_event", True))
                    include_delta_summary = bool(cfg_get(write_token_cfg, "include_delta_summary", True))
                    include_visit_embedding = bool(cfg_get(write_token_cfg, "include_visit_embedding", True))
                    self.stage2_3_delta_summary_fail_fast = bool(
                        cfg_get(
                            write_token_cfg,
                            "fail_fast",
                            cfg_get(parent_optimizer_cfg, "fail_fast", cfg_get(iforward_cfg, "fail_fast", True)),
                        )
                    )
                    if memory_type in {"lowrank_gated_delta_kv", "gated_delta_kv", "gdkv"}:
                        gdkv_cfg = cfg_get(parent_optimizer_cfg, "gated_delta_kv", {}) or {}
                        self.parent_temporal_mamba = ParentOptimizerGatedDeltaKV(
                            event_dim=int(cfg_get(parent_optimizer_cfg, "event_dim", cfg_get(parent_spatial_cfg, "event_dim", 64))),
                            ctx_dim=int(cfg_get(parent_optimizer_cfg, "ctx_dim", cfg_get(gdkv_cfg, "V", 32))),
                            token_dim=int(cfg_get(parent_optimizer_cfg, "token_dim", cfg_get(parent_optimizer_cfg, "event_dim", 64))),
                            key_dim=int(cfg_get(parent_optimizer_cfg, "key_dim", cfg_get(gdkv_cfg, "K", 16))),
                            value_dim=int(cfg_get(parent_optimizer_cfg, "value_dim", cfg_get(gdkv_cfg, "V", 32))),
                            adapter_hidden_dim=int(cfg_get(parent_optimizer_cfg, "adapter_hidden_dim", 64)),
                            visit_dim=int(cfg_get(cfg_get(parent_optimizer_cfg, "visit_embedding", {}) or {}, "output_dim", 32)),
                            support_min=float(cfg_get(write_mask_cfg, "support_min", 0.001)),
                            dense_bg=bool(cfg_get(parent_optimizer_cfg, "dense_bg", True)),
                            dense_distant=bool(cfg_get(parent_optimizer_cfg, "dense_distant", True)),
                            gate_init=dict(cfg_get(fusion_cfg, "gate_init", {}) or {}),
                            value_rms_max=float(cfg_get(gdkv_cfg, "value_rms_max", 2.0)),
                            ctx_rms_max=float(cfg_get(gdkv_cfg, "ctx_rms_max", 4.0)),
                            state_rms_max=float(cfg_get(gdkv_cfg, "state_rms_max", 4.0)),
                            erase_gate_max=float(cfg_get(gdkv_cfg, "erase_gate_max", 1.0)),
                            write_gate_max=float(cfg_get(gdkv_cfg, "write_gate_max", 1.0)),
                            erase_bias=float(cfg_get(gdkv_cfg, "erase_bias", 0.0)),
                            write_bias=float(cfg_get(gdkv_cfg, "write_bias", 0.0)),
                            decay_bias=float(cfg_get(gdkv_cfg, "decay_bias", 0.0)),
                            decay_min=cfg_get(gdkv_cfg, "decay_min", None),
                            query_rms_unit=bool(cfg_get(gdkv_cfg, "query_rms_unit", True)),
                            key_rms_unit=bool(cfg_get(gdkv_cfg, "key_rms_unit", True)),
                            include_spatial_event=include_spatial_event,
                            include_parent_event=include_parent_event,
                            include_delta_summary=include_delta_summary,
                            include_visit_embedding=include_visit_embedding,
                            update_rule=str(cfg_get(gdkv_cfg, "update_rule", "gdn2_legacy")),
                            alpha_mode=str(cfg_get(gdkv_cfg, "alpha_mode", "value_channel")),
                            alpha_max=float(cfg_get(gdkv_cfg, "alpha_max", 1.0)),
                            alpha_init=float(cfg_get(gdkv_cfg, "alpha_init", 0.1)),
                            surprise_gating=bool(cfg_get(gdkv_cfg, "surprise_gating", True)),
                            surprise_target_rms=float(cfg_get(gdkv_cfg, "surprise_target_rms", 1.0)),
                            min_alpha_on_unseen=float(cfg_get(gdkv_cfg, "min_alpha_on_unseen", 0.5)),
                            cleanup_enable=bool(cfg_get(gdkv_cfg, "cleanup_enable", False)),
                            cleanup_key=str(cfg_get(gdkv_cfg, "cleanup_key", "learned")),
                            cleanup_max=float(cfg_get(gdkv_cfg, "cleanup_max", 0.2)),
                            cleanup_init=float(cfg_get(gdkv_cfg, "cleanup_init", 0.02)),
                            cleanup_by_kind=cfg_get(gdkv_cfg, "cleanup_by_kind", None),
                            state_dtype=storage_dtype_from_name(
                                cfg_get(
                                    cfg_get(cfg_get(cfg_get(self.config, "training", {}) or {}, "amp", {}) or {}, "memory", {}) or {},
                                    "gdkv_state_dtype",
                                    "fp32",
                                ),
                                amp_dtype=self.amp_policy.dtype,
                            ),
                        )
                    else:
                        self.parent_temporal_mamba = ParentOptimizerMamba(
                            event_dim=int(cfg_get(parent_optimizer_cfg, "event_dim", cfg_get(parent_spatial_cfg, "event_dim", 64))),
                            ctx_dim=int(cfg_get(parent_optimizer_cfg, "ctx_dim", 32)),
                            model_dim=int(cfg_get(parent_optimizer_cfg, "model_dim", 32)),
                            state_dim=int(cfg_get(parent_optimizer_cfg, "state_dim", 8)),
                            conv_kernel=int(cfg_get(parent_optimizer_cfg, "conv_kernel", 2)),
                            adapter_hidden_dim=int(cfg_get(parent_optimizer_cfg, "adapter_hidden_dim", 64)),
                            visit_dim=int(cfg_get(cfg_get(parent_optimizer_cfg, "visit_embedding", {}) or {}, "output_dim", 32)),
                            support_min=float(cfg_get(write_mask_cfg, "support_min", 0.001)),
                            dense_bg=bool(cfg_get(parent_optimizer_cfg, "dense_bg", True)),
                            dense_distant=bool(cfg_get(parent_optimizer_cfg, "dense_distant", True)),
                            gate_init=dict(cfg_get(fusion_cfg, "gate_init", {}) or {}),
                            include_spatial_event=include_spatial_event,
                            include_parent_event=include_parent_event,
                            include_delta_summary=include_delta_summary,
                            include_visit_embedding=include_visit_embedding,
                        )
                elif self.is_stage2_2_parent_temporal:
                    parent_temporal_cfg = (
                        cfg_get(iforward_cfg, "parent_temporal_mamba_v2", None)
                        or cfg_get(iforward_cfg, "parent_temporal_mamba", {})
                        or {}
                    )
                    self.parent_temporal_mamba = ParentTemporalMemoryV2(
                        event_dim=int(cfg_get(parent_temporal_cfg, "event_dim", cfg_get(parent_spatial_cfg, "event_dim", 64))),
                        ctx_dim=int(cfg_get(parent_temporal_cfg, "ctx_dim", 32)),
                        model_dim=int(cfg_get(parent_temporal_cfg, "model_dim", 32)),
                        state_dim=int(cfg_get(parent_temporal_cfg, "state_dim", 8)),
                        conv_kernel=int(cfg_get(parent_temporal_cfg, "conv_kernel", 2)),
                        adapter_hidden_dim=int(cfg_get(parent_temporal_cfg, "adapter_hidden_dim", 64)),
                        motion_embed_dim=int(cfg_get(parent_temporal_cfg, "motion_embed_dim", 16)),
                        dense_bg=bool(cfg_get(parent_temporal_cfg, "dense_bg", True)),
                        dense_distant=bool(cfg_get(parent_temporal_cfg, "dense_distant", True)),
                    )
                else:
                    parent_temporal_cfg = cfg_get(iforward_cfg, "parent_temporal_mamba", {}) or {}
                    self.parent_temporal_mamba = ParentTemporalMemory(
                        event_dim=int(cfg_get(parent_temporal_cfg, "event_dim", cfg_get(parent_spatial_cfg, "event_dim", 64))),
                        ctx_dim=int(cfg_get(parent_temporal_cfg, "ctx_dim", 32)),
                        model_dim=int(cfg_get(parent_temporal_cfg, "model_dim", 32)),
                        state_dim=int(cfg_get(parent_temporal_cfg, "state_dim", 8)),
                        conv_kernel=int(cfg_get(parent_temporal_cfg, "conv_kernel", 2)),
                        adapter_hidden_dim=int(cfg_get(parent_temporal_cfg, "adapter_hidden_dim", 64)),
                        dense_bg=bool(cfg_get(parent_temporal_cfg, "dense_bg", True)),
                        dense_distant=bool(cfg_get(parent_temporal_cfg, "dense_distant", True)),
                    )
            else:
                self.parent_spatial_backbone = None
                self.parent_temporal_mamba = None
        else:
            self.history_gate_v2_cfg = {}
            self.history_gate_v2_enabled = False
            self.memory = IForwardSceneMemory(
                event_dim=event_dim,
                model_dim=int(cfg_get(memory_cfg, "model_dim", event_dim)),
                state_dim=int(cfg_get(memory_cfg, "state_dim", 16)),
                conv_kernel=int(cfg_get(memory_cfg, "conv_kernel", 4)),
                bg_cell_size=float(cfg_get(memory_cfg, "bg_cell_size", 0.5)),
                distant_cell_size=float(cfg_get(memory_cfg, "distant_cell_size", 2.0)),
                rigid_cell_size=float(cfg_get(memory_cfg, "rigid_cell_size", 0.5)),
                enable_aux_stats=bool(cfg_get(debug_cfg, "enable_memory_aux_stats", False)),
                log_per_k_aux_interval=int(cfg_get(debug_cfg, "log_per_k_aux_interval", 50)),
                dense_point_memory=bool(cfg_get(memory_cfg, "dense_point_memory", True)),
                long_write_policy=str(cfg_get(memory_cfg, "long_write_policy", "every_repeat")),
                short_entry_policy=str(cfg_get(memory_cfg, "short_entry_policy", "frame_exit_only")),
                short_entry_detach=bool(cfg_get(memory_cfg, "short_entry_detach", True)),
                hard_valid_required=bool(
                    cfg_get(cfg_get(memory_cfg, "write_gate", {}) or {}, "hard_valid_required", True)
                ),
                hard_support_min_commit=float(
                    cfg_get(cfg_get(memory_cfg, "write_gate", {}) or {}, "hard_support_min_commit", 0.0)
                ),
                hard_support_min_optimizer=float(
                    cfg_get(cfg_get(memory_cfg, "write_gate", {}) or {}, "hard_support_min_optimizer", 0.0)
                ),
            )
            self.history_safe_projection = None
        history_cfg = cfg_get(iforward_cfg, "short_window_history", {}) or {}
        self.history_max_entries = int(cfg_get(history_cfg, "max_entries", 24))
        self.history_max_memory_entries = (
            0
            if self.is_v6_point_mamba_xcpe or self.is_v3_gru_history_gate or self.is_stage2_0_biggs_parent_lifting
            else int(cfg_get(history_cfg, "max_memory_entries", 8))
        )
        loss_cfg = cfg_get(iforward_cfg, "loss", {}) or {}
        self.loss_current_weight = float(cfg_get(cfg_get(loss_cfg, "current", {}) or {}, "weight", 1.0))
        self.loss_nearby_weight = float(
            cfg_get(
                cfg_get(loss_cfg, "nearby", {}) or {},
                "weight",
                _loss_float(config, ["losses", "phase_a", "nearby_render", "weight"], 0.25) if config is not None else 0.25,
            )
        )
        in_rollout_history_loss_cfg = cfg_get(loss_cfg, "in_rollout_history", {}) or {}
        self.loss_in_rollout_history_weight = float(cfg_get(in_rollout_history_loss_cfg, "weight", 0.1))
        history_warmup_cfg = cfg_get(in_rollout_history_loss_cfg, "warmup", {}) or {}
        self.loss_in_rollout_history_warmup_enable = bool(cfg_get(history_warmup_cfg, "enable", False))
        self.loss_in_rollout_history_warmup_steps = int(cfg_get(history_warmup_cfg, "steps", 0))
        self.loss_in_rollout_history_warmup_start_step = int(cfg_get(history_warmup_cfg, "start_step", 0))
        self.loss_in_rollout_history_warmup_start_factor = float(cfg_get(history_warmup_cfg, "start_factor", 0.0))
        self.loss_short_window_history_weight = float(
            cfg_get(cfg_get(loss_cfg, "short_window_history", {}) or {}, "weight", 0.1)
        )
        history_damage_cfg = cfg_get(loss_cfg, "history_damage", {}) or {}
        self.loss_history_damage_enable = bool(cfg_get(history_damage_cfg, "enable", False))
        self.loss_history_damage_target_weight = float(cfg_get(history_damage_cfg, "weight", 0.0))
        self.loss_history_damage_margin = float(cfg_get(history_damage_cfg, "margin", 0.0))
        history_damage_warmup_cfg = cfg_get(history_damage_cfg, "warmup", {}) or {}
        self.loss_history_damage_warmup_enable = bool(cfg_get(history_damage_warmup_cfg, "enable", True))
        self.loss_history_damage_warmup_start_step = int(cfg_get(history_damage_warmup_cfg, "start_step", 10000))
        self.loss_history_damage_warmup_steps = int(cfg_get(history_damage_warmup_cfg, "steps", 15000))
        self.loss_delta_reg_weight = float(cfg_get(cfg_get(loss_cfg, "delta_regularization", {}) or {}, "weight", 1.0))
        stage3_lifting_cfg = cfg_get(iforward_cfg, "lifting", {}) or {}
        stage3_reg_cfg = cfg_get(stage3_lifting_cfg, "regularization", {}) or {}
        self.stage3_offset_l2_weight = float(cfg_get(stage3_reg_cfg, "offset_l2", 0.0))
        self.stage3_out_of_bounds_weight = float(cfg_get(stage3_reg_cfg, "out_of_bounds", 0.0))
        hsp_cfg = cfg_get(iforward_cfg, "history_safe_projection", {}) or {}
        hsp_damage_cfg = cfg_get(hsp_cfg, "damage_loss", {}) or {}
        self.hsp_damage_loss_weight = (
            float(cfg_get(hsp_damage_cfg, "weight", 0.0))
            if (
                self.is_v3_gru_history_gate
                and bool(cfg_get(hsp_cfg, "enable", False))
                and not bool(getattr(self, "history_gate_v2_enabled", False))
                and bool(cfg_get(hsp_damage_cfg, "enable", True))
            )
            else 0.0
        )
        hgv2_aux_cfg = cfg_get(getattr(self, "history_gate_v2_cfg", {}) or {}, "auxiliary_loss", {}) or {}
        self.loss_hgv2_gate_weight = (
            float(cfg_get(hgv2_aux_cfg, "weight", 1.0))
            if self.is_v3_gru_history_gate
            and bool(getattr(self, "history_gate_v2_enabled", False))
            and bool(cfg_get(hgv2_aux_cfg, "enable", True))
            else 0.0
        )
        train_ifwd_cfg = cfg_get(cfg_get(config, "training", {}) or {}, "iforward", {}) or {}
        self.allow_missing_carried_state_reset = bool(
            cfg_get(train_ifwd_cfg, "allow_missing_carried_state_reset", False)
        )
        if self.is_v3_gru_history_gate:
            self.allowed_ablations = {
                "full",
                "full_adc",
                "no_gru",
                "no_history_gate",
                "no_adc",
                "freeze_write",
            }
        elif self.is_v6_point_mamba_xcpe:
            self.allowed_ablations = {
                "full",
                "point_only",
                "xcpe_only",
                "no_memory",
                "disable_rigid_xcpe",
                "freeze_write",
                "shuffle_context",
            }
        elif self.is_stage2_3_optimizer_mamba:
            self.allowed_ablations = {
                "full",
                "mamba_off",
                "mamba_read_only",
                "mamba_read_write",
                "mamba_shuffle_state",
                "mamba_freeze_write",
                "mamba_shuffle_read_write_state",
                "mamba_wrong_parent_key_fixed",
            }
        else:
            self.allowed_ablations = {
                "full",
                "zero_all",
                "zero_point",
                "zero_cell",
                "zero_global",
                "drop_short_window",
                "freeze_write",
                "shuffle_memory",
                "bypass_memory",
            }
        self.to(self.device)

    def _nvtx_range(self, name: str) -> Any:
        if bool(self.enable_nvtx_ranges) and torch.cuda.is_available():
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _amp_fp32_context(self, *, enabled: bool = True) -> Any:
        if bool(enabled):
            return self.amp_policy.fp32()
        return nullcontext()

    def load_init_checkpoint_payload(
        self,
        ckpt: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        weights_only: bool = True,
        path: Optional[str] = None,
    ) -> bool:
        _ = (weights_only, path)
        if str(ckpt.get("export_type", "")) != "stage6_0_phase_a_for_phase_b":
            init_cfg = cfg_get(self.config, "initialization", {}) or {}
            skip_keys = [str(x) for x in list(cfg_get(init_cfg, "skip_keys", []) or []) if str(x)]
            sd = ckpt.get("model_state_dict")
            if not isinstance(sd, dict):
                return False
            raw_sd = {str(k): v for k, v in dict(sd).items() if str(k) != "_extra_state"}
            current_keys = set(self.state_dict().keys())
            direct_matches = sum(1 for key in raw_sd if key in current_keys)
            stripped_sd = {
                (key[len("model.") :] if key.startswith("model.") else key): value
                for key, value in raw_sd.items()
            }
            stripped_matches = sum(1 for key in stripped_sd if key in current_keys)
            normalized_sd = stripped_sd if stripped_matches > direct_matches else raw_sd
            filtered = {
                key: value
                for key, value in normalized_sd.items()
                if not any(str(key).startswith(prefix) or f".{prefix}" in str(key) for prefix in skip_keys)
            }
            missing, unexpected = self.load_state_dict(filtered, strict=False)

            def _skip_allowed(name: str) -> bool:
                return any(str(name).startswith(prefix) or f".{prefix}" in str(name) for prefix in skip_keys)

            def _new_gdkv_param(name: str) -> bool:
                text = str(name)
                return ".alpha_proj." in text or ".cleanup_key_proj." in text or ".cleanup_proj." in text

            def _legacy_sparse_conv(name: str) -> bool:
                return str(name).startswith("phase_a_runtime.sparse_conv.") or str(name).startswith(
                    "model.phase_a_runtime.sparse_conv."
                )

            bad_missing = [str(k) for k in missing if not (_skip_allowed(str(k)) or _new_gdkv_param(str(k)))]
            bad_unexpected = [str(k) for k in unexpected if not (_skip_allowed(str(k)) or _legacy_sparse_conv(str(k)))]
            if bad_missing or bad_unexpected:
                raise ValueError(
                    "IForward init_checkpoint skip_keys load failed: "
                    f"missing={bad_missing[:20]} unexpected={bad_unexpected[:20]}"
                )
            return True
        runtime = getattr(self, "phase_a_runtime", None)
        loader = getattr(runtime, "_load_phase_b_export_payload", None)
        if not callable(loader):
            return False
        target_device = device if device is not None else self.device
        loader(ckpt, device=target_device)
        return True

    def init_iforward_state_from_batch_assets(self, batch: Dict[str, Any], resolved: IForwardResolvedBatch) -> IForwardState:
        local_state, node_bg, node_distant, node_rigid = self.bridge.make_local_state(batch=batch)
        local_state = local_state.to(device=self.device)
        history_ema = None
        if self.is_v3_gru_history_gate:
            point_gru = getattr(self, "point_gru", None)
            if point_gru is None:
                raise RuntimeError("IForward-v3 point_gru is not initialized.")
            memory_state = point_gru.init_state(local_state)
            history_ema = IForwardHistoryEMAState.from_local_state(local_state)
        elif self.is_v6_point_mamba_xcpe:
            memory_state = IForwardV6MemoryState.empty()
        else:
            memory_state = IForwardMemoryState.empty()
        if self.is_stage2_3_optimizer_mamba:
            parent_memory = getattr(self, "parent_temporal_mamba", None)
            empty_state = getattr(parent_memory, "empty_state", None)
            parent_temporal = empty_state() if callable(empty_state) else ParentOptimizerMambaState.empty()
        elif self.is_stage2_2_parent_temporal:
            parent_temporal = ParentTemporalStateV2.empty()
        elif self.is_stage2_1_parent_temporal:
            parent_temporal = ParentTemporalState.empty()
        else:
            parent_temporal = None
        sequence10_bank = (
            Sequence10HistoryBank.empty(device=self.device)
            if str(getattr(resolved, "scheduler_version", "")) == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
            else None
        )
        stage2_2_bank = (
            EpisodeHistoryBankV2.empty(device=self.device)
            if str(getattr(resolved, "scheduler_version", "")) == IFORWARD_STAGE2_2_SCHEDULER_VERSION
            else None
        )
        stage2_3_bank = (
            EpisodeHistoryBankV3.empty(device=self.device)
            if str(getattr(resolved, "scheduler_version", "")) in IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS
            else None
        )
        return IForwardState(
            local_gs=local_state,
            memory=memory_state,
            history=IForwardShortWindowHistory.empty(
                max_entries=int(self.history_max_entries),
                max_memory_entries=int(self.history_max_memory_entries),
            ),
            scene_id=int(resolved.scene_id),
            segment_id=int(resolved.segment_id),
            episode_id=int(resolved.episode_id),
            history_ema=history_ema,
            history_gradient_bank=None,
            adc_bank=None,
            adc_meta=None,
            node_state_bg=node_bg,
            node_state_distant=node_distant,
            node_state_rigid=node_rigid,
            biggs_state=None,
            parent_temporal=parent_temporal,
            sequence10_bank=sequence10_bank,
            stage2_2_bank=stage2_2_bank,
            stage2_3_bank=stage2_3_bank,
        )

    def _adc_lite_aabb(
        self,
        local_state: LocalGSState,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not hasattr(self.bridge, "stage6_aabb"):
            return None, None
        aabb_min, aabb_max = self.bridge.stage6_aabb(local_state.bg.means)
        return aabb_min.to(device=self.device), aabb_max.to(device=self.device)

    def _adc_lite_near_voxel_size(self) -> Optional[float]:
        model_cfg = cfg_get(self.config, "model", {}) or {}
        stage6_cfg = cfg_get(model_cfg, "stage6_0", {}) or {}
        struct_cfg = cfg_get(stage6_cfg, "struct_event_decoder", {}) or {}
        near_cfg = cfg_get(struct_cfg, "near", {}) or {}
        value = cfg_get(near_cfg, "voxel_size", None)
        if value is None:
            legacy_struct_cfg = cfg_get(model_cfg, "struct_decoder", {}) or {}
            legacy_near_cfg = cfg_get(legacy_struct_cfg, "near", {}) or {}
            value = cfg_get(legacy_near_cfg, "voxel_size", None)
        return None if value is None else float(value)

    def _adc_lite_policy_stats(
        self,
        *,
        global_step: int,
        resolved: IForwardResolvedBatch,
    ) -> Tuple[bool, Dict[str, float]]:
        cfg = getattr(self, "adc_lite_cfg", {}) or {}
        policy_cfg = cfg_get(cfg, "enable_policy", {}) or {}
        start_step = int(cfg_get(cfg, "start_step", 0))
        min_blocks = int(cfg_get(policy_cfg, "min_blocks_per_rollout", 1))
        blocks = int(len(getattr(resolved, "window_block_ids", []) or []))
        if blocks <= 0:
            blocks = len({int(getattr(step, "block_id", idx)) for idx, step in enumerate(getattr(resolved, "steps", []) or [])})
        step_ok = int(global_step) >= int(start_step)
        blocks_ok = int(blocks) >= int(min_blocks)
        allowed = bool(step_ok and blocks_ok)
        return allowed, {
            f"{ADC_STAT_PREFIX}/policy/start_step": float(start_step),
            f"{ADC_STAT_PREFIX}/policy/min_blocks_per_rollout": float(min_blocks),
            f"{ADC_STAT_PREFIX}/policy/blocks_per_rollout": float(blocks),
            f"{ADC_STAT_PREFIX}/policy/step_ok": 1.0 if bool(step_ok) else 0.0,
            f"{ADC_STAT_PREFIX}/policy/blocks_ok": 1.0 if bool(blocks_ok) else 0.0,
            f"{ADC_STAT_PREFIX}/policy/apply_build_allowed": 1.0 if bool(allowed) else 0.0,
            f"{ADC_STAT_PREFIX}/policy/log_only_before_start": (
                1.0 if bool(cfg_get(policy_cfg, "log_only_before_start", False)) else 0.0
            ),
        }

    def _adc_lite_rollout_start_planning(
        self,
        *,
        state: IForwardState,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        resolved: IForwardResolvedBatch,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        cfg = getattr(self, "adc_lite_cfg", {}) or {}
        planning_cfg = cfg_get(cfg, "planning", {}) or {}
        enabled = bool(cfg_get(planning_cfg, "enable", False))
        stats: Dict[str, float] = {
            f"{ADC_STAT_PREFIX}/planning/pass_enabled": 1.0 if enabled else 0.0,
            f"{ADC_STAT_PREFIX}/planning/pass_ran": 0.0,
        }
        if not enabled:
            return None, None, stats
        bank = getattr(state, "adc_bank", None)
        if bank is None or not bool(getattr(bank, "valid", False)):
            stats[f"{ADC_STAT_PREFIX}/planning/skipped_no_bank"] = 1.0
            return None, None, stats
        steps = list(getattr(resolved, "steps", []) or [])
        if not steps:
            stats[f"{ADC_STAT_PREFIX}/planning/skipped_no_steps"] = 1.0
            return None, None, stats
        scope = str(cfg_get(planning_cfg, "scope", "first_step")).lower()
        if scope in {"block_enter", "block_enters"}:
            selected = [step for step in steps if bool(getattr(step, "is_block_enter", False))]
        elif scope in {"all", "all_steps"}:
            selected = steps
        else:
            selected = steps[:1]
        max_steps = max(1, int(cfg_get(planning_cfg, "max_steps_per_rollout", 1)))
        selected = selected[:max_steps]
        if not selected:
            stats[f"{ADC_STAT_PREFIX}/planning/skipped_no_selected_steps"] = 1.0
            return None, None, stats

        n_bg = int(local_state.bg.means.shape[0])
        support_sum: Optional[torch.Tensor] = None
        valid_any: Optional[torch.Tensor] = None
        used = 0
        observe_planning = getattr(self.bridge, "observe_planning", None)
        with torch.no_grad():
            for step in selected:
                if callable(observe_planning):
                    measurement = observe_planning(
                        local_state=local_state,
                        batch=batch,
                        source_indices=list(step.source_indices),
                        source_frame_idx=int(step.source_frame_idx),
                    )
                else:
                    measurement = self.bridge.observe(
                        local_state=local_state,
                        batch=batch,
                        source_indices=list(step.source_indices),
                        source_frame_idx=int(step.source_frame_idx),
                    )
                event = self.bridge.build_event(local_state=local_state, measurement=measurement)
                support = getattr(event, "support_bg", None)
                if not torch.is_tensor(support) or int(support.shape[0]) != n_bg:
                    continue
                support_flat = support.detach().to(device=self.device, dtype=torch.float32).reshape(n_bg, -1).mean(dim=-1)
                valid = getattr(event, "valid_bg", None)
                if torch.is_tensor(valid) and int(valid.shape[0]) == n_bg:
                    valid_flat = valid.detach().to(device=self.device, dtype=torch.bool).reshape(n_bg, -1).any(dim=-1)
                else:
                    valid_flat = torch.isfinite(support_flat)
                support_sum = support_flat if support_sum is None else support_sum + support_flat
                valid_any = valid_flat if valid_any is None else (valid_any | valid_flat)
                used += 1
        if support_sum is None or valid_any is None or used <= 0:
            stats[f"{ADC_STAT_PREFIX}/planning/skipped_no_support"] = 1.0
            return None, None, stats
        support_avg = support_sum / float(max(used, 1))
        stats.update(
            {
                f"{ADC_STAT_PREFIX}/planning/pass_ran": 1.0,
                f"{ADC_STAT_PREFIX}/planning/pass_steps": float(used),
                f"{ADC_STAT_PREFIX}/planning/pass_visible_ratio": float(valid_any.float().mean().item()),
                f"{ADC_STAT_PREFIX}/planning/pass_support_mean": float(support_avg[valid_any].mean().item())
                if bool(valid_any.any().item())
                else 0.0,
            }
        )
        return support_avg, valid_any, stats

    @staticmethod
    def _zero_loss(ref: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        return ref.new_tensor(0.0), {
            "num_refs": 0.0,
            "num_metric_refs": 0.0,
            "metric_valid": 0.0,
            "valid_ratio": 0.0,
            "skipped_no_valid_pixels": 0.0,
            }

    def _history_rollout_loss_weight_for_step(self, global_step: int) -> float:
        base = float(self.loss_in_rollout_history_weight)
        if not bool(self.loss_in_rollout_history_warmup_enable):
            return base
        warmup_steps = int(self.loss_in_rollout_history_warmup_steps)
        if warmup_steps <= 0:
            return base
        start_step = int(self.loss_in_rollout_history_warmup_start_step)
        step = max(0, int(global_step) - start_step)
        progress = min(max(float(step) / float(warmup_steps), 0.0), 1.0)
        start_factor = min(max(float(self.loss_in_rollout_history_warmup_start_factor), 0.0), 1.0)
        factor = start_factor + (1.0 - start_factor) * progress
        return float(base * factor)

    @staticmethod
    def _should_emit_gdkv_aux_stats(debug_cfg: Any, *, global_step: int) -> bool:
        interval_raw = cfg_get(debug_cfg, "gdkv_aux_interval", None)
        if interval_raw is None:
            return True
        interval = int(interval_raw)
        return bool(interval > 0 and int(global_step) % interval == 0)

    def _history_damage_loss_weight_for_step(self, global_step: int) -> float:
        if not bool(getattr(self, "loss_history_damage_enable", False)):
            return 0.0
        base = float(getattr(self, "loss_history_damage_target_weight", 0.0))
        if base <= 0.0:
            return 0.0
        if not bool(getattr(self, "loss_history_damage_warmup_enable", True)):
            return base
        start = int(getattr(self, "loss_history_damage_warmup_start_step", 10000))
        steps = int(getattr(self, "loss_history_damage_warmup_steps", 15000))
        if steps <= 0:
            return base if int(global_step) >= start else 0.0
        progress = min(max(float(int(global_step) - start) / float(steps), 0.0), 1.0)
        return float(base * progress)

    def _per_pos_loss_from_final_roles(
        self,
        *,
        resolved: IForwardResolvedBatch,
        source_frames: List[int],
        roles: Tuple[IForwardFinalRenderRole, ...],
        ref: torch.Tensor,
        expected_positions: int = 10,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not source_frames:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        frame_to_pos = {int(frame): int(pos) for pos, frame in enumerate(source_frames[: int(expected_positions)])}
        per_pos_values: Dict[int, List[torch.Tensor]] = {}
        for role in roles:
            per_ref = role.per_ref_loss.reshape(-1)
            target_indices = tuple(int(x) for x in role.target_indices)
            for offset, target_idx in enumerate(target_indices[: int(per_ref.numel())]):
                frame_idx = int(resolved.target_refs[int(target_idx)][0])
                pos = frame_to_pos.get(int(frame_idx))
                if pos is None:
                    continue
                per_pos_values.setdefault(int(pos), []).append(per_ref[int(offset)])
        per_pos_loss = {
            int(pos): torch.stack(values).mean()
            for pos, values in per_pos_values.items()
            if values
        }
        after_by_pos = []
        valid_by_pos = []
        zero = per_ref.new_tensor(0.0)
        for pos in range(int(expected_positions)):
            value = per_pos_loss.get(int(pos))
            if value is None:
                after_by_pos.append(zero)
                valid_by_pos.append(False)
            else:
                after_by_pos.append(value)
                valid_by_pos.append(True)
        after = torch.stack(after_by_pos) if after_by_pos else per_ref.new_zeros((0,))
        valid = torch.tensor(valid_by_pos, device=after.device, dtype=torch.bool)
        return per_pos_loss, after, valid

    def _sequence10_current_per_pos_loss_from_final_pack(
        self,
        *,
        resolved: IForwardResolvedBatch,
        final_pack: IForwardFinalRenderPack,
        ref: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        if str(getattr(resolved, "scheduler_version", "")) != IFORWARD_SEQUENCE10_SCHEDULER_VERSION:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        meta = dict(getattr(resolved, "meta", {}) or {})
        source_frames = [int(x) for x in list(meta.get("sequence_source_frame_indices", []) or [])]
        if len(source_frames) != 10:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        return self._per_pos_loss_from_final_roles(
            resolved=resolved,
            source_frames=source_frames,
            roles=(final_pack.current,),
            ref=ref,
            expected_positions=10,
        )

    def _stage2_2_per_pos_loss(
        self,
        *,
        resolved: IForwardResolvedBatch,
        final_pack: IForwardFinalRenderPack,
        ref: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        if str(getattr(resolved, "scheduler_version", "")) != IFORWARD_STAGE2_2_SCHEDULER_VERSION:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        meta = dict(getattr(resolved, "meta", {}) or {})
        source_frames = [int(x) for x in list(meta.get("sequence_source_frame_indices", []) or [])]
        if not source_frames:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        return self._per_pos_loss_from_final_roles(
            resolved=resolved,
            source_frames=source_frames,
            roles=(final_pack.current, final_pack.history),
            ref=ref,
            expected_positions=10,
        )

    def _stage2_3_per_pos_loss(
        self,
        *,
        resolved: IForwardResolvedBatch,
        final_pack: IForwardFinalRenderPack,
        ref: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        if str(getattr(resolved, "scheduler_version", "")) not in IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        meta = dict(getattr(resolved, "meta", {}) or {})
        source_frames = [int(x) for x in list(meta.get("sequence_source_frame_indices", []) or [])]
        if not source_frames:
            request_meta = dict(meta.get("request_meta", {}) or {})
            source_frames = [
                int(x)
                for x in list(dict(request_meta.get("iforward_stage2_3", {}) or {}).get("raw_frame_ids", []) or [])
            ]
        if not source_frames:
            return {}, ref.new_zeros((0,)), torch.zeros((0,), device=ref.device, dtype=torch.bool)
        return self._per_pos_loss_from_final_roles(
            resolved=resolved,
            source_frames=source_frames,
            roles=(final_pack.current, final_pack.history),
            ref=ref,
            expected_positions=len(source_frames),
        )

    def _render_final_losses(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        resolved: IForwardResolvedBatch,
        carried_state: Optional[IForwardState],
        ablation: str,
        in_rollout_history_loss_weight: float,
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, float],
        List[torch.Tensor],
        List[torch.Tensor],
        List[Tuple[int, int]],
        List[str],
        IForwardFinalRenderPack,
    ]:
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        image_refs: List[Tuple[int, int]] = []
        image_roles: List[str] = []
        zero_ref = local_state.bg.means
        current_indices = list(resolved.current_latest_target_indices)
        before = len(pred_rgbs)
        current_loss, current_stats, current_per_ref = self.bridge.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=current_indices,
            mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
            pred_rgbs_out=pred_rgbs,
            gt_images_out=gt_images,
            return_per_ref_loss=True,
        )
        current_role = IForwardFinalRenderRole(
            target_indices=tuple(int(x) for x in current_indices),
            mean_loss=current_loss,
            per_ref_loss=current_per_ref,
            stats=dict(current_stats),
        )
        appended = len(pred_rgbs) - before
        image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in current_indices[:appended]])
        image_roles.extend(["current_latest"] * int(appended))
        force_history_render = bool((getattr(resolved, "meta", {}) or {}).get("validation_force_history_render", False))
        if (float(in_rollout_history_loss_weight) > 0.0 or bool(force_history_render)) and len(resolved.history_rollout_target_indices) > 0:
            history_indices = list(resolved.history_rollout_target_indices)
            before = len(pred_rgbs)
            in_rollout_history_loss, in_rollout_stats, history_per_ref = self.bridge.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=history_indices,
                mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
                pred_rgbs_out=pred_rgbs,
                gt_images_out=gt_images,
                return_per_ref_loss=True,
            )
            appended = len(pred_rgbs) - before
            image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in history_indices[:appended]])
            image_roles.extend(["history_rollout"] * int(appended))
        else:
            in_rollout_history_loss, in_rollout_stats = self._zero_loss(local_state.bg.means)
            history_indices = []
            history_per_ref = zero_ref.new_zeros((0,))
        history_role = IForwardFinalRenderRole(
            target_indices=tuple(int(x) for x in history_indices),
            mean_loss=in_rollout_history_loss,
            per_ref_loss=history_per_ref,
            stats=dict(in_rollout_stats),
        )

        short_history_loss, short_history_stats = self._zero_loss(local_state.bg.means)
        if (
            carried_state is not None
            and self.loss_short_window_history_weight > 0.0
            and carried_state.history.entries
        ):
            short_targets = [dict(x) for x in carried_state.history.entries]
            before = len(pred_rgbs)
            short_history_loss, short_history_stats = self.bridge.render_loss_for_targets(
                local_state=local_state,
                ref_batch=batch,
                targets=short_targets,
                mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
                pred_rgbs_out=pred_rgbs,
                gt_images_out=gt_images,
            )
            appended = len(pred_rgbs) - before
            for target in short_targets[:appended]:
                image_refs.append((int(target.get("frame_idx", -1)), int(target.get("cam_idx", -1))))
                image_roles.append("short_window_history")
        nearby_indices = list(resolved.nearby_target_indices)
        before = len(pred_rgbs)
        nearby_loss, nearby_stats, nearby_per_ref = self.bridge.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=nearby_indices,
            mask_policy=str(getattr(self.bridge, "nearby_mask_policy", "non_sky_non_egocar")),
            pred_rgbs_out=pred_rgbs,
            gt_images_out=gt_images,
            return_per_ref_loss=True,
        )
        nearby_role = IForwardFinalRenderRole(
            target_indices=tuple(int(x) for x in nearby_indices),
            mean_loss=nearby_loss,
            per_ref_loss=nearby_per_ref,
            stats=dict(nearby_stats),
        )
        appended = len(pred_rgbs) - before
        image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in nearby_indices[:appended]])
        image_roles.extend(["nearby"] * int(appended))
        eval_role_stats: Dict[str, float] = {}
        for eval_role in ("eval_recon_all_blocks", "eval_nearby_nvs_all_blocks"):
            eval_indices = list(resolved.target_indices_by_role.get(eval_role, ()))
            if not eval_indices:
                continue
            before = len(pred_rgbs)
            _eval_loss, eval_stats = self.bridge.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=eval_indices,
                mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
                pred_rgbs_out=pred_rgbs,
                gt_images_out=gt_images,
            )
            appended = len(pred_rgbs) - before
            image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in eval_indices[:appended]])
            image_roles.extend([eval_role] * int(appended))
            safe_prefix = str(eval_role).replace("/", "_")
            for metric in ("psnr", "ssim", "l1", "valid_ratio", "num_refs", "num_metric_refs", "metric_valid"):
                value = eval_stats.get(metric)
                if value is not None and math.isfinite(float(value)):
                    eval_role_stats[f"{safe_prefix}_{metric}"] = float(value)
        losses = {
            "current": current_loss,
            "current_latest": current_loss,
            "history": in_rollout_history_loss,
            "nearby": nearby_loss,
            "in_rollout_history": in_rollout_history_loss,
            "short_window_history": short_history_loss,
        }
        stats = {
            "current_valid_ratio": float(current_stats.get("valid_ratio", 0.0)),
            "current_latest_valid_ratio": float(current_stats.get("valid_ratio", 0.0)),
            "nearby_valid_ratio": float(nearby_stats.get("valid_ratio", 0.0)),
            "in_rollout_history_valid_ratio": float(in_rollout_stats.get("valid_ratio", 0.0)),
            "short_window_history_valid_ratio": float(short_history_stats.get("valid_ratio", 0.0)),
            "current_num_refs": float(current_stats.get("num_refs", len(resolved.current_latest_target_indices))),
            "current_latest_num_refs": float(current_stats.get("num_refs", len(resolved.current_latest_target_indices))),
            "history_rollout_num_refs": float(
                in_rollout_stats.get("num_refs", len(resolved.history_rollout_target_indices))
            ),
            "history_num_refs": float(
                in_rollout_stats.get("num_refs", len(resolved.history_rollout_target_indices))
            ),
            "in_rollout_history_num_refs": float(
                in_rollout_stats.get("num_refs", len(resolved.history_rollout_target_indices))
            ),
            "nearby_num_refs": float(nearby_stats.get("num_refs", len(resolved.nearby_target_indices))),
            "short_window_history_num_refs": float(short_history_stats.get("num_refs", 0.0)),
        }
        stats.update(eval_role_stats)
        for prefix, item in (
            ("current", current_stats),
            ("nearby", nearby_stats),
            ("history_rollout", in_rollout_stats),
            ("short_window_history", short_history_stats),
        ):
            for metric in ("psnr", "ssim", "l1"):
                value = item.get(metric)
                if value is not None and math.isfinite(float(value)):
                    stats[f"{prefix}_{metric}"] = float(value)
                    if prefix == "current":
                        stats[f"current_latest_{metric}"] = float(value)
                    if prefix == "history_rollout":
                        stats[f"in_rollout_history_{metric}"] = float(value)
                        stats[f"history_{metric}"] = float(value)
        render_pack = IForwardFinalRenderPack(
            current=current_role,
            history=history_role,
            nearby=nearby_role,
        )
        return losses, stats, pred_rgbs, gt_images, image_refs, image_roles, render_pack

    def _build_v6_context(
        self,
        *,
        event: Any,
        local_state: LocalGSState,
        memory_state: Any,
        step_context: IForwardMemoryStepContext,
        ablation: str,
    ) -> tuple[IForwardV6MemoryState, Any, Dict[str, float]]:
        point_mamba = getattr(self, "point_mamba", None)
        local_conflict = getattr(self, "local_conflict", None)
        context_adapter = getattr(self, "context_adapter", None)
        if point_mamba is None or local_conflict is None or context_adapter is None:
            raise RuntimeError("IForward-v6 modules are not initialized.")
        v6_state = memory_state if isinstance(memory_state, IForwardV6MemoryState) else IForwardV6MemoryState.empty()
        next_memory, point_pack, point_aux = point_mamba(
            event=event,
            local_state=local_state,
            state=v6_state,
            step_context=step_context,
            ablation=str(ablation),
        )
        if hasattr(self.bridge, "stage6_aabb"):
            aabb_min, aabb_max = self.bridge.stage6_aabb(local_state.bg.means)
        else:
            ref = local_state.bg.means
            aabb_min = ref.detach().amin(dim=0) - 1.0 if ref.numel() else ref.new_full((3,), -1.0)
            aabb_max = ref.detach().amax(dim=0) + 1.0 if ref.numel() else ref.new_full((3,), 1.0)
        local_pack = local_conflict(
            event=event,
            point_ctx=point_pack,
            local_state=local_state,
            step_context=step_context,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            ablation=str(ablation),
        )
        ctx_pack = context_adapter(
            event=event,
            point_ctx=point_pack,
            local_ctx=local_pack,
            step_context=step_context,
            ablation=str(ablation),
        )
        aux: Dict[str, float] = {}
        for source in (point_aux, getattr(local_pack, "aux", None), getattr(ctx_pack, "aux", None)):
            if isinstance(source, dict):
                aux.update({str(k): float(v) for k, v in source.items() if isinstance(v, (int, float))})
        return next_memory, ctx_pack, aux

    def _ensure_v3_state(
        self,
        *,
        local_state: LocalGSState,
        memory_state: Any,
        history_ema: Optional[IForwardHistoryEMAState],
    ) -> tuple[IForwardGRUMemoryState, IForwardHistoryEMAState]:
        point_gru = getattr(self, "point_gru", None)
        if point_gru is None:
            raise RuntimeError("IForward-v3 point_gru is not initialized.")
        def branch_ok(state_branch: Any, local_branch: Any) -> bool:
            expected_rows = int(local_branch.means.shape[0]) if local_branch is not None else 0
            h = getattr(state_branch, "h", None)
            return (
                torch.is_tensor(h)
                and int(h.shape[0]) == int(expected_rows)
                and int(h.shape[1]) == int(point_gru.hidden_dim)
            )

        memory_ok = (
            isinstance(memory_state, IForwardGRUMemoryState)
            and branch_ok(memory_state.bg, local_state.bg)
            and branch_ok(memory_state.distant, local_state.distant)
            and branch_ok(memory_state.rigid, local_state.rigid)
        )
        if not memory_ok:
            memory_state = point_gru.init_state(local_state)

        def history_ok(hist_branch: Any, local_branch: Any) -> bool:
            expected_rows = int(local_branch.means.shape[0]) if local_branch is not None else 0
            if hist_branch is None:
                return expected_rows == 0
            support_fast = getattr(hist_branch, "support_fast", None)
            return torch.is_tensor(support_fast) and int(support_fast.shape[0]) == int(expected_rows)

        history_state_ok = (
            history_ema is not None
            and history_ok(history_ema.bg, local_state.bg)
            and history_ok(history_ema.distant, local_state.distant)
            and history_ok(history_ema.rigid, local_state.rigid)
        )
        if not history_state_ok:
            history_ema = IForwardHistoryEMAState.from_local_state(local_state)
        return memory_state, history_ema

    @staticmethod
    def _legacy_step_block_exit(step: Any, next_step: Optional[Any]) -> bool:
        if next_step is None:
            return True
        step_block = getattr(step, "block_id", getattr(step, "episode_block_idx", getattr(step, "rollout_block_rank", None)))
        next_block = getattr(
            next_step,
            "block_id",
            getattr(next_step, "episode_block_idx", getattr(next_step, "rollout_block_rank", None)),
        )
        if step_block is not None and next_block is not None:
            return int(step_block) != int(next_block)
        return int(next_step.source_frame_idx) != int(step.source_frame_idx)

    @classmethod
    def _resolved_step_block_flags(cls, step: Any, next_step: Optional[Any]) -> tuple[bool, bool]:
        is_block_enter = (
            bool(getattr(step, "is_block_enter"))
            if hasattr(step, "is_block_enter")
            else bool(int(getattr(step, "repeat_idx", 0)) == 0)
        )
        is_block_exit = (
            bool(getattr(step, "is_block_exit"))
            if hasattr(step, "is_block_exit")
            else cls._legacy_step_block_exit(step, next_step)
        )
        return bool(is_block_enter), bool(is_block_exit)

    def _normalize_ablation_name(self, ablation: Optional[str]) -> str:
        name = str(ablation or "full")
        if not bool(getattr(self, "is_stage2_3_optimizer_mamba", False)):
            return name
        aliases = {
            "off": "mamba_off",
            "read_only": "mamba_read_only",
            "read_write": "mamba_read_write",
            "shuffled": "mamba_shuffle_state",
            "shuffle_memory": "mamba_shuffle_state",
            "bypass_memory": "mamba_off",
            "freeze_write": "mamba_freeze_write",
            "shuffle_read_write_state": "mamba_shuffle_read_write_state",
            "shuffle_rw_state": "mamba_shuffle_read_write_state",
            "wrong_parent_key_fixed": "mamba_wrong_parent_key_fixed",
        }
        return str(aliases.get(name, name))

    @staticmethod
    def _shuffle_stage2_3_optimizer_state(
        state: Optional[ParentOptimizerMambaState | ParentOptimizerDeltaKVState],
    ) -> Optional[ParentOptimizerMambaState | ParentOptimizerDeltaKVState]:
        if state is None:
            return None

        def _shuffle_mamba_dense(dense: Optional[DenseOptimizerState]) -> Optional[DenseOptimizerState]:
            if dense is None or int(dense.seen.numel()) <= 1:
                return dense
            order = torch.randperm(int(dense.seen.numel()), device=dense.seen.device)
            return DenseOptimizerState(
                conv_state=dense.conv_state.index_select(0, order.to(device=dense.conv_state.device)),
                ssm_state=dense.ssm_state.index_select(0, order.to(device=dense.ssm_state.device)),
                seen=dense.seen.index_select(0, order),
                update_count=dense.update_count.index_select(0, order.to(device=dense.update_count.device)),
                last_visit_step=dense.last_visit_step.index_select(0, order.to(device=dense.last_visit_step.device)),
                last_frame_id=dense.last_frame_id.index_select(0, order.to(device=dense.last_frame_id.device)),
                last_visit_kind=dense.last_visit_kind.index_select(0, order.to(device=dense.last_visit_kind.device)),
            )

        def _shuffle_mamba_keyed(keyed: Optional[KeyedOptimizerState]) -> Optional[KeyedOptimizerState]:
            if keyed is None or int(keyed.keys.numel()) <= 1:
                return keyed
            order = torch.randperm(int(keyed.keys.numel()), device=keyed.keys.device)
            return KeyedOptimizerState(
                keys=keyed.keys,
                conv_state=keyed.conv_state.index_select(0, order.to(device=keyed.conv_state.device)),
                ssm_state=keyed.ssm_state.index_select(0, order.to(device=keyed.ssm_state.device)),
                seen=keyed.seen.index_select(0, order.to(device=keyed.seen.device)),
                update_count=keyed.update_count.index_select(0, order.to(device=keyed.update_count.device)),
                last_visit_step=keyed.last_visit_step.index_select(0, order.to(device=keyed.last_visit_step.device)),
                last_frame_id=keyed.last_frame_id.index_select(0, order.to(device=keyed.last_frame_id.device)),
                last_visit_kind=keyed.last_visit_kind.index_select(0, order.to(device=keyed.last_visit_kind.device)),
            )

        def _shuffle_delta_dense(dense: Optional[DenseDeltaKVOptimizerState]) -> Optional[DenseDeltaKVOptimizerState]:
            if dense is None or int(dense.seen.numel()) <= 1:
                return dense
            order = torch.randperm(int(dense.seen.numel()), device=dense.seen.device)
            return DenseDeltaKVOptimizerState(
                kv_state=dense.kv_state.index_select(0, order.to(device=dense.kv_state.device)),
                seen=dense.seen.index_select(0, order),
                update_count=dense.update_count.index_select(0, order.to(device=dense.update_count.device)),
                last_visit_step=dense.last_visit_step.index_select(0, order.to(device=dense.last_visit_step.device)),
                last_frame_id=dense.last_frame_id.index_select(0, order.to(device=dense.last_frame_id.device)),
                last_visit_kind=dense.last_visit_kind.index_select(0, order.to(device=dense.last_visit_kind.device)),
            )

        def _shuffle_delta_keyed(keyed: Optional[KeyedDeltaKVOptimizerState]) -> Optional[KeyedDeltaKVOptimizerState]:
            if keyed is None or int(keyed.keys.numel()) <= 1:
                return keyed
            order = torch.randperm(int(keyed.keys.numel()), device=keyed.keys.device)
            return KeyedDeltaKVOptimizerState(
                keys=keyed.keys,
                kv_state=keyed.kv_state.index_select(0, order.to(device=keyed.kv_state.device)),
                seen=keyed.seen.index_select(0, order.to(device=keyed.seen.device)),
                update_count=keyed.update_count.index_select(0, order.to(device=keyed.update_count.device)),
                last_visit_step=keyed.last_visit_step.index_select(0, order.to(device=keyed.last_visit_step.device)),
                last_frame_id=keyed.last_frame_id.index_select(0, order.to(device=keyed.last_frame_id.device)),
                last_visit_kind=keyed.last_visit_kind.index_select(0, order.to(device=keyed.last_visit_kind.device)),
            )

        if isinstance(state, ParentOptimizerDeltaKVState):
            def _shuffle_delta_branch(branch: DeltaKVOptimizerBranchState) -> DeltaKVOptimizerBranchState:
                return DeltaKVOptimizerBranchState(dense=_shuffle_delta_dense(branch.dense), keyed=_shuffle_delta_keyed(branch.keyed))

            return ParentOptimizerDeltaKVState(
                bg=_shuffle_delta_branch(state.bg),
                distant=_shuffle_delta_branch(state.distant),
                rigid=_shuffle_delta_branch(state.rigid),
                global_update_step=int(state.global_update_step),
            )

        def _shuffle_mamba_branch(branch: OptimizerBranchState) -> OptimizerBranchState:
            return OptimizerBranchState(dense=_shuffle_mamba_dense(branch.dense), keyed=_shuffle_mamba_keyed(branch.keyed))

        return ParentOptimizerMambaState(
            bg=_shuffle_mamba_branch(state.bg),
            distant=_shuffle_mamba_branch(state.distant),
            rigid=_shuffle_mamba_branch(state.rigid),
            global_update_step=int(state.global_update_step),
        )

    def forward_rollout(
        self,
        batch: Dict[str, Any],
        *,
        carried_state: Optional[IForwardState] = None,
        ablation: Optional[str] = None,
    ) -> IForwardRolloutOutput:
        ablation_name = self._normalize_ablation_name(ablation)
        if ablation_name not in self.allowed_ablations:
            raise ValueError(f"unsupported IForward ablation={ablation_name!r}")
        module_ablation_name = "full" if ablation_name in {"full_adc", "no_adc"} else ablation_name
        if self.is_stage2_3_optimizer_mamba:
            module_ablation_name = "full"
        stage2_3_mamba_off = bool(self.is_stage2_3_optimizer_mamba and ablation_name == "mamba_off")
        stage2_3_mamba_freeze_write = bool(
            self.is_stage2_3_optimizer_mamba
            and ablation_name in {"mamba_read_only", "mamba_freeze_write", "mamba_wrong_parent_key_fixed"}
        )
        stage2_3_mamba_shuffle_state = bool(
            self.is_stage2_3_optimizer_mamba and ablation_name == "mamba_shuffle_state"
        )
        stage2_3_mamba_shuffle_read_write_state = bool(
            self.is_stage2_3_optimizer_mamba
            and ablation_name in {"mamba_shuffle_read_write_state", "mamba_wrong_parent_key_fixed"}
        )
        adc_disabled_by_ablation = ablation_name == "no_adc"
        resolved = self.resolver.resolve(batch)
        global_step = int(batch.get("global_step", 0) or 0)
        in_rollout_history_loss_weight = self._history_rollout_loss_weight_for_step(global_step)
        if bool(resolved.reset_scene_state_before_rollout):
            state = self.init_iforward_state_from_batch_assets(batch, resolved)
            prior_state_for_history = None
        elif carried_state is None:
            if not bool(self.allow_missing_carried_state_reset):
                raise RuntimeError(
                    "IForward missing carried_state for non-reset rollout; this would break episode-local "
                    "GS/memory carry. Resume from an episode boundary or restore IForward state cache."
                )
            state = self.init_iforward_state_from_batch_assets(batch, resolved)
            prior_state_for_history = None
        else:
            state = carried_state
            prior_state_for_history = carried_state
        if bool(stage2_3_mamba_shuffle_read_write_state) and getattr(state, "parent_temporal", None) is not None:
            state.parent_temporal = self._shuffle_stage2_3_optimizer_state(state.parent_temporal)
            if prior_state_for_history is state:
                prior_state_for_history = state
            if tuple(state.cache_key) != tuple(resolved.cache_key):
                raise ValueError(f"IForward carried state key {state.cache_key} does not match batch {resolved.cache_key}.")

        local_state = state.local_gs.to(device=self.device)
        state.local_gs = local_state
        if hasattr(self.bridge, "sync_local_state_template_from_batch"):
            node_bg, node_distant, node_rigid = self.bridge.sync_local_state_template_from_batch(
                local_state=local_state,
                batch=batch,
            )
            local_state = local_state.to(device=self.device)
            state.local_gs = local_state
            state.node_state_bg = node_bg
            state.node_state_distant = node_distant
            state.node_state_rigid = node_rigid
        adc_apply_stats: Dict[str, float] = {
            f"{ADC_STAT_PREFIX}/enabled": 1.0 if bool(getattr(self, "adc_lite_enabled", False)) else 0.0,
            f"{ADC_STAT_PREFIX}/disabled_by_ablation": 1.0 if bool(adc_disabled_by_ablation) else 0.0,
            f"{ADC_STAT_PREFIX}/bank_valid": 0.0,
            f"{ADC_STAT_PREFIX}/bank_dropped_without_apply": 0.0,
            f"{ADC_STAT_PREFIX}/bank_shape_mismatch": 0.0,
            f"{ADC_STAT_PREFIX}/applied": 0.0,
            f"{ADC_STAT_PREFIX}/num_cloned_this_rollout": 0.0,
            f"{ADC_STAT_PREFIX}/num_cloned_episode": float(
                getattr(getattr(state, "adc_meta", None), "num_bg_clones_created_episode", 0)
            ),
            f"{ADC_STAT_PREFIX}/bg_count_before": float(local_state.bg.means.shape[0]),
            f"{ADC_STAT_PREFIX}/bg_count_after": float(local_state.bg.means.shape[0]),
            "adc_suppressed/parent_gate_mean": 0.0,
            "adc_suppressed/parent_delta_demand_mean": 0.0,
            "adc_suppressed/parent_support_mean": 0.0,
            "adc_suppressed/selected_parent_suppression_rank_percentile": 0.0,
        }
        if bool(getattr(self, "adc_lite_enabled", False)):
            adc_policy_allowed, adc_policy_stats = self._adc_lite_policy_stats(global_step=global_step, resolved=resolved)
            state.adc_meta = ensure_adc_meta_for_state(
                local_state=local_state,
                adc_meta=getattr(state, "adc_meta", None),
                device=self.device,
            )
            bank = getattr(state, "adc_bank", None)
            bank_valid = bank is not None and bool(getattr(bank, "valid", False))
            if bool(adc_disabled_by_ablation) or not bool(adc_policy_allowed):
                state.adc_bank = None
                adc_apply_stats.update(adc_policy_stats)
                adc_apply_stats.update(
                    {
                        f"{ADC_STAT_PREFIX}/bank_valid": 1.0 if bool(bank_valid) else 0.0,
                        f"{ADC_STAT_PREFIX}/bank_dropped_without_apply": 1.0 if bool(bank_valid) else 0.0,
                        f"{ADC_STAT_PREFIX}/num_cloned_episode": float(state.adc_meta.num_bg_clones_created_episode),
                    }
                )
            else:
                adc_aabb_min, adc_aabb_max = self._adc_lite_aabb(local_state)
                planning_support_bg, planning_valid_bg, adc_planning_stats = self._adc_lite_rollout_start_planning(
                    state=state,
                    local_state=local_state,
                    batch=batch,
                    resolved=resolved,
                )
                state, adc_apply_stats = apply_bg_clone_episode_local(
                    state=state,
                    cfg=getattr(self, "adc_lite_cfg", {}) or {},
                    rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                    device=self.device,
                    planning_support_bg=planning_support_bg,
                    planning_valid_bg=planning_valid_bg,
                    aabb_min=adc_aabb_min,
                    aabb_max=adc_aabb_max,
                    voxel_size=self._adc_lite_near_voxel_size(),
                )
                adc_apply_stats.update(adc_planning_stats)
                adc_apply_stats.update(adc_policy_stats)
                local_state = state.local_gs
        memory_state = state.memory
        parent_temporal_state = (
            getattr(state, "parent_temporal", None)
            if (self.is_stage2_1_parent_temporal or self.is_stage2_2_parent_temporal or self.is_stage2_3_optimizer_mamba)
            else None
        )
        if self.is_stage2_1_parent_temporal and not isinstance(parent_temporal_state, ParentTemporalState):
            parent_temporal_state = ParentTemporalState.empty()
            state.parent_temporal = parent_temporal_state
        if self.is_stage2_2_parent_temporal and not isinstance(parent_temporal_state, ParentTemporalStateV2):
            parent_temporal_state = ParentTemporalStateV2.empty()
            state.parent_temporal = parent_temporal_state
        if self.is_stage2_3_optimizer_mamba:
            parent_memory = getattr(self, "parent_temporal_mamba", None)
            expected_cls = getattr(parent_memory, "state_cls", ParentOptimizerMambaState)
            if not isinstance(parent_temporal_state, expected_cls):
                empty_state = getattr(parent_memory, "empty_state", None)
                parent_temporal_state = empty_state() if callable(empty_state) else ParentOptimizerMambaState.empty()
                state.parent_temporal = parent_temporal_state
        history_ema = state.history_ema
        if self.is_v3_gru_history_gate:
            memory_state, history_ema = self._ensure_v3_state(
                local_state=local_state,
                memory_state=memory_state,
                history_ema=history_ema,
            )
        history_gradient_bank = None
        if self.is_v3_gru_history_gate and bool(getattr(self, "history_gate_v2_enabled", False)):
            history_gradient_bank = getattr(state, "history_gradient_bank", None)
            if history_gradient_bank is not None:
                history_gradient_bank = history_gradient_bank.to(device=self.device)
        working_history = state.history
        history_entries_before = int(len(working_history.entries))
        memory_entries_before = int(len(working_history.memory_entries))
        per_step: List[Dict[str, float]] = []
        reg_terms: List[torch.Tensor] = []
        reg_stats_sum: Dict[str, float] = {}
        stage3_reg_terms: List[torch.Tensor] = []
        stage3_reg_stats_sum: Dict[str, float] = {}
        hsp_losses: List[torch.Tensor] = []
        hsp_stats_sum: Dict[str, float] = {}
        hsp_stats_count = 0
        hgv2_losses: List[torch.Tensor] = []
        hgv2_stats_sum: Dict[str, float] = {}
        hgv2_stats_count = 0
        block_hsp_cache: Dict[str, Any] = {}
        timings: Dict[str, float] = {
            "observe_ms": 0.0,
            "event_ms": 0.0,
            "memory_ms": 0.0,
            "update_ms": 0.0,
            "parent_runtime_update_ms": 0.0,
            "delta_reg_ms": 0.0,
            "final_render_ms": 0.0,
        }
        model_cfg = cfg_get(self.config, "model", {}) or {}
        iforward_cfg_for_mem = cfg_get(model_cfg, "iforward", {}) or {}
        debug_cfg_for_mem = cfg_get(iforward_cfg_for_mem, "debug", {}) or {}
        lifting_cfg_for_mem = cfg_get(iforward_cfg_for_mem, "lifting", {}) or {}
        default_mem_interval = int(cfg_get(lifting_cfg_for_mem, "memory_aux_interval", 0))
        forward_mem_interval = int(
            cfg_get(debug_cfg_for_mem, "forward_memory_aux_interval", default_mem_interval)
        )
        emit_gdkv_aux_stats = self._should_emit_gdkv_aux_stats(debug_cfg_for_mem, global_step=global_step)
        emit_forward_mem_aux = bool(
            torch.cuda.is_available()
            and int(forward_mem_interval) > 0
            and int(global_step) % int(forward_mem_interval) == 0
        )
        rollout_mem_aux: Dict[str, float] = {
            "iforward/forward_mem/interval": float(max(int(forward_mem_interval), 0)),
            "iforward/forward_mem/sampled": 1.0 if bool(emit_forward_mem_aux) else 0.0,
        }

        def _cuda_mem_snapshot() -> Optional[Dict[str, float]]:
            if not bool(emit_forward_mem_aux):
                return None
            try:
                device = local_state.bg.means.device
                if torch.device(device).type != "cuda":
                    device = torch.cuda.current_device()
                allocated = float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0))
                reserved = float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0))
                max_allocated = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            except Exception:
                return None
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated,
            }

        def _record_stage_mem(
            stage: str,
            *,
            out: Dict[str, float],
            prev: Optional[Dict[str, float]] = None,
            prefix: str = "iforward/forward_mem",
        ) -> Optional[Dict[str, float]]:
            snap = _cuda_mem_snapshot()
            if snap is None:
                return prev
            for name, value in snap.items():
                out[f"{prefix}/{stage}_{name}"] = float(value)
            if prev is not None:
                out[f"{prefix}/{stage}_allocated_delta_mb"] = float(
                    snap["allocated_mb"] - prev.get("allocated_mb", snap["allocated_mb"])
                )
                out[f"{prefix}/{stage}_reserved_delta_mb"] = float(
                    snap["reserved_mb"] - prev.get("reserved_mb", snap["reserved_mb"])
                )
            return snap

        rollout_mem_prev = _record_stage_mem("rollout_start", out=rollout_mem_aux)
        adc_suppression_accumulator = None
        if (
            bool(getattr(self, "adc_lite_enabled", False))
            and bool(getattr(self, "adc_lite_gate_suppressed", False))
            and not bool(adc_disabled_by_ablation)
        ):
            adc_suppression_accumulator = GateSuppressedADCAccumulator.from_local_state(local_state)

        observe_batch = batch
        if self.is_stage2_0_biggs_parent_lifting:
            if int(resolved.scene_id) < 0 or int(resolved.segment_id) < 0:
                raise ValueError(
                    "IForward Stage2 BigGS resolved batch missing valid scene_id/segment_id: "
                    f"scene_id={int(resolved.scene_id)} segment_id={int(resolved.segment_id)} "
                    f"batch_keys={sorted(str(k) for k in batch.keys())}"
                )
            observe_batch = dict(batch)
            ifwd_meta = dict(resolved.meta or {})
            ifwd_meta["scene_id"] = int(resolved.scene_id)
            ifwd_meta["segment_id"] = int(resolved.segment_id)
            ifwd_meta["episode_id"] = int(resolved.episode_id)
            ifwd_meta["rollout_id_global"] = int(resolved.rollout_id_global)
            request_meta = dict(observe_batch.get("request_meta") or {})
            request_meta["scene_id"] = int(resolved.scene_id)
            request_meta["segment_id"] = int(resolved.segment_id)
            request_meta["episode_id"] = int(resolved.episode_id)
            request_meta["rollout_id_global"] = int(resolved.rollout_id_global)
            request_meta["iforward"] = ifwd_meta
            observe_batch["scene_id"] = int(resolved.scene_id)
            observe_batch["segment_id"] = int(resolved.segment_id)
            observe_batch["request_meta"] = request_meta
            observe_batch["_iforward"] = ifwd_meta

        global_step = int(batch.get("global_step", 0))
        feedback_schedule_step = int(batch.get("feedback_schedule_step", global_step) or 0)
        feedback_activation_global_step = int(
            batch.get("feedback_activation_global_step", 0) or 0
        )
        history_damage_weight = self._history_damage_loss_weight_for_step(global_step)
        history_damage_probe: Optional[HistoryDamageProbe] = None
        if self.is_stage2_1_parent_temporal and float(history_damage_weight) > 0.0:
            history_damage_indices = list(resolved.history_rollout_target_indices)
            if history_damage_indices:
                with torch.no_grad():
                    with self._amp_fp32_context(enabled=bool(self.amp_policy.render_force_fp32)):
                        _before_loss, _before_stats, before_per_ref = self.bridge.render_loss(
                            local_state=local_state,
                            batch=batch,
                            target_indices=history_damage_indices,
                            mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
                            return_per_ref_loss=True,
                        )
                history_damage_probe = HistoryDamageProbe(
                    target_indices=[int(x) for x in history_damage_indices],
                    before_per_ref=before_per_ref.detach(),
                )
            else:
                history_damage_probe = HistoryDamageProbe.empty(ref=local_state.bg.means)
        biggs_parent_runtime = None
        stage2_1_parent_block_cache: Dict[str, Any] = {}
        for step_pos, step in enumerate(resolved.steps):
            step_mem_aux: Dict[str, float] = {}
            step_mem_prev = _record_stage_mem("before_observe", out=step_mem_aux)
            next_step = resolved.steps[step_pos + 1] if step_pos + 1 < len(resolved.steps) else None
            is_block_enter, is_block_exit = self._resolved_step_block_flags(step, next_step)
            if bool(is_block_enter):
                block_hsp_cache = {}
                stage2_1_parent_block_cache = {}
                if self.is_stage2_0_biggs_parent_lifting:
                    if str(getattr(self, "stage2_0_biggs_parent_exact_refresh_policy", "block_enter")) == "block_enter":
                        biggs_parent_runtime = None
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/observe"):
                observe_kwargs = {
                    "local_state": local_state,
                    "batch": observe_batch,
                    "source_indices": list(step.source_indices),
                    "source_frame_idx": int(step.source_frame_idx),
                }
                if self.is_stage2_0_biggs_parent_lifting:
                    observe_kwargs["biggs_state"] = getattr(state, "biggs_state", None)
                    observe_kwargs["biggs_parent_runtime"] = biggs_parent_runtime
                    observe_kwargs["biggs_scene_id"] = int(resolved.scene_id)
                    observe_kwargs["biggs_segment_id"] = int(resolved.segment_id)
                    observe_kwargs["biggs_episode_id"] = int(resolved.episode_id)
                    stage3_2_visit_meta = dict(
                        dict(observe_batch.get("request_meta") or {}).get("iforward_stage3_2", {}) or {}
                    )
                    visit_meta = {
                        "global_step": int(global_step),
                        "feedback_schedule_step": int(feedback_schedule_step),
                        "feedback_activation_global_step": int(feedback_activation_global_step),
                        "step_idx": int(getattr(step, "step_idx", 0)),
                        "repeat_idx": int(getattr(step, "repeat_idx", 0)),
                        "repeat_budget": int(getattr(step, "repeat_budget", 0)),
                        "visit_kind": str(getattr(step, "visit_kind", "")),
                    }
                    feedback_eval_mode = dict(observe_batch.get("request_meta") or {}).get(
                        "observation_feedback_eval_mode", None
                    )
                    if feedback_eval_mode is not None:
                        visit_meta["observation_feedback_eval_mode"] = str(feedback_eval_mode)
                    if stage3_2_visit_meta:
                        for meta_key in (
                            "distribution_type",
                            "distribution_type_id",
                            "episode_stage",
                            "episode_stage_id",
                            "train_2d_mode",
                            "train_2d_mode_id",
                        ):
                            if meta_key in stage3_2_visit_meta:
                                visit_meta[str(meta_key)] = stage3_2_visit_meta[meta_key]
                        visit_meta["iforward_stage3_2"] = stage3_2_visit_meta
                    observe_kwargs["visit_meta"] = visit_meta
                    if bool(self.is_stage3_0_full_sparse_gather_lift):
                        observe_kwargs["parent_optimizer_state"] = parent_temporal_state
                measurement = self.bridge.observe(**observe_kwargs)
                if (
                    self.is_stage2_0_biggs_parent_lifting
                    and isinstance(measurement, dict)
                    and measurement.get("biggs_state") is not None
                ):
                    next_biggs_state = measurement["biggs_state"]
                    detach_biggs = getattr(next_biggs_state, "detach", None)
                    state.biggs_state = detach_biggs() if callable(detach_biggs) else next_biggs_state
                    biggs_parent_runtime = measurement.get("biggs_parent_runtime", biggs_parent_runtime)
            step_mem_prev = _record_stage_mem("after_observe", out=step_mem_aux, prev=step_mem_prev)
            observe_step_ms = (time.perf_counter() - t0) * 1000.0
            timings["observe_ms"] += observe_step_ms
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/event"):
                if self.is_stage2_1_parent_temporal or self.is_stage2_2_parent_temporal or self.is_stage2_3_optimizer_mamba:
                    if not isinstance(measurement, dict):
                        raise RuntimeError("Stage2 parent temporal requires dict measurement from BigGS observation.")
                    parent_spatial = getattr(self, "parent_spatial_backbone", None)
                    parent_temporal = getattr(self, "parent_temporal_mamba", None)
                    if parent_spatial is None or parent_temporal is None:
                        raise RuntimeError("Stage2 parent modules are not initialized.")
                    parent_inputs = self.bridge.build_stage2_1_parent_inputs(
                        local_state=local_state,
                        measurement=measurement,
                    )
                    parent_event_spatial, near_layout_cache = parent_spatial(
                        near_in=parent_inputs["near_in"],
                        far_in=parent_inputs["far_in"],
                        route=parent_inputs["route"],
                        aabb_min=parent_inputs["aabb_min"],
                        aabb_max=parent_inputs["aabb_max"],
                        near_batch_offsets=parent_inputs.get("near_batch_offsets"),
                        far_batch_offsets=parent_inputs.get("far_batch_offsets"),
                        near_layout_cache=stage2_1_parent_block_cache.get("near_layout_cache"),
                        frame_gap=int(getattr(step, "frame_gap", 0)),
                        visit_kind=str(getattr(step, "visit_kind", "causal_first") or "causal_first"),
                    )
                    stage2_1_parent_block_cache["near_layout_cache"] = near_layout_cache
                    if self.is_stage2_2_parent_temporal or self.is_stage2_3_optimizer_mamba:
                        parent_keys = build_parent_temporal_keys_v2(
                            parent_event=parent_event_spatial,
                            measurement=measurement,
                        )
                    else:
                        parent_keys = build_parent_temporal_keys(
                            parent_event=parent_event_spatial,
                            measurement=measurement,
                        )
                    is_sequence10_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
                    is_stage2_2_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_STAGE2_2_SCHEDULER_VERSION
                    is_stage2_3_scheduler = str(getattr(resolved, "scheduler_version", "")) in IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS
                    temporal_read_enabled = (
                        bool(getattr(step, "optimizer_memory_read", True))
                        if bool(is_stage2_3_scheduler)
                        else
                        bool(getattr(step, "temporal_read", True))
                        if (is_sequence10_scheduler or is_stage2_2_scheduler)
                        else True
                    )
                    if bool(self.is_stage2_3_optimizer_mamba and stage2_3_mamba_off):
                        temporal_read_enabled = False
                    if bool(temporal_read_enabled):
                        if self.is_stage2_3_optimizer_mamba:
                            preview_state = (
                                self._shuffle_stage2_3_optimizer_state(parent_temporal_state)
                                if bool(stage2_3_mamba_shuffle_state)
                                else parent_temporal_state
                            )
                            parent_preview = parent_temporal.preview(
                                event=parent_event_spatial,
                                state=preview_state,
                                keys=parent_keys,
                                visit_meta=VisitMeta.from_step(step),
                                **(
                                    {"emit_aux_stats": bool(emit_gdkv_aux_stats)}
                                    if bool(getattr(self, "is_stage3_1_lowrank_gdkv", False))
                                    else {}
                                ),
                            )
                        elif self.is_stage2_2_parent_temporal:
                            parent_preview = parent_temporal.preview(
                                event=parent_event_spatial,
                                state=parent_temporal_state,
                                keys=parent_keys,
                                timestamp_sec=float(getattr(step, "timestamp_sec", 0.0)),
                                motion_meta={
                                    "delta_t_sec": float(getattr(step, "delta_t_sec", 0.0)),
                                    "frame_gap": float(getattr(step, "frame_gap", 0)),
                                    "visit_kind": str(getattr(step, "visit_kind", "")),
                                    "ego_delta_translation": torch.tensor(
                                        list(getattr(step, "ego_delta_translation", (0.0, 0.0, 0.0))),
                                        device=parent_event_spatial.event_bg.device,
                                        dtype=parent_event_spatial.event_bg.dtype,
                                    ).reshape(1, 3),
                                    "ego_delta_yaw": float(getattr(step, "ego_delta_yaw", 0.0)),
                                },
                            )
                        else:
                            parent_preview = parent_temporal.preview(
                                event=parent_event_spatial,
                                state=parent_temporal_state,
                                keys=parent_keys,
                            )
                        parent_event_for_decode = parent_preview.event
                    else:
                        parent_event_for_decode = parent_event_spatial
                        aux = dict(parent_event_for_decode.aux or {})
                        aux[
                            "iforward/parent_optimizer_mamba/read_skipped"
                            if self.is_stage2_3_optimizer_mamba
                            else "iforward/parent_temporal/read_skipped"
                        ] = 1.0
                        if bool(getattr(self, "is_stage3_1_lowrank_gdkv", False)):
                            aux["iforward/parent_optimizer_gdkv/read_skipped"] = 1.0
                            aux["iforward/parent_optimizer_memory/type_id"] = 1.0
                            aux["iforward/parent_optimizer_memory/is_gdkv"] = 1.0
                            aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 1.0
                        elif self.is_stage2_3_optimizer_mamba:
                            aux["iforward/parent_optimizer_memory/type_id"] = 0.0
                            aux["iforward/parent_optimizer_memory/is_gdkv"] = 0.0
                            aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 0.0
                        parent_event_for_decode.aux = aux
                    if self.is_stage2_3_optimizer_mamba:
                        stage2_1_parent_block_cache["optimizer_spatial_event"] = parent_event_spatial
                        stage2_1_parent_block_cache["optimizer_fused_event"] = parent_event_for_decode
                        stage2_1_parent_block_cache["optimizer_keys"] = parent_keys
                    block_can_temporal_commit = (
                        str(getattr(step, "visit_kind", "")) not in {"repair", "stress"}
                        if bool(is_sequence10_scheduler or is_stage2_2_scheduler)
                        else True
                    )
                    if (not self.is_stage2_3_optimizer_mamba) and bool(block_can_temporal_commit) and (
                        bool(is_block_enter) or "commit_event" not in stage2_1_parent_block_cache
                    ):
                        step_block_id = int(
                            getattr(
                                step,
                                "block_id",
                                getattr(step, "episode_block_idx", getattr(step, "rollout_block_rank", step.step_idx)),
                            )
                        )
                        stage2_1_parent_block_cache["commit_event"] = parent_event_spatial
                        stage2_1_parent_block_cache["commit_keys"] = parent_keys
                        stage2_1_parent_block_cache["commit_block_id"] = step_block_id
                        stage2_1_parent_block_cache["commit_timestamp_sec"] = float(getattr(step, "timestamp_sec", 0.0))
                        stage2_1_parent_block_cache["commit_physical_time_advance"] = bool(
                            getattr(step, "physical_time_advance", True)
                        )
                        stage2_1_parent_block_cache["commit_motion_meta"] = {
                            "delta_t_sec": float(getattr(step, "delta_t_sec", 0.0)),
                            "frame_gap": float(getattr(step, "frame_gap", 0)),
                            "visit_kind": str(getattr(step, "visit_kind", "")),
                            "ego_delta_translation": torch.tensor(
                                list(getattr(step, "ego_delta_translation", (0.0, 0.0, 0.0))),
                                device=parent_event_spatial.event_bg.device,
                                dtype=parent_event_spatial.event_bg.dtype,
                            ).reshape(1, 3),
                            "ego_delta_yaw": float(getattr(step, "ego_delta_yaw", 0.0)),
                        }
                    event = self.bridge.decode_stage2_1_biggs_child_event(
                        parent_event=parent_event_for_decode,
                        local_state=local_state,
                        measurement=measurement,
                    )
                else:
                    event = self.bridge.build_event(local_state=local_state, measurement=measurement)
                if self.is_stage3_0_full_sparse_gather_lift and isinstance(measurement, dict):
                    raw_terms = measurement.get("stage3_gather_reg_terms", {})
                    if isinstance(raw_terms, dict) and any(torch.is_tensor(v) for v in raw_terms.values()):
                        stage3_reg_loss, stage3_reg_stats = stage3_gather_regularization(
                            raw_terms,
                            offset_l2_weight=float(getattr(self, "stage3_offset_l2_weight", 0.0)),
                            out_of_bounds_weight=float(getattr(self, "stage3_out_of_bounds_weight", 0.0)),
                        )
                        stage3_reg_terms.append(stage3_reg_loss)
                        for key, value in stage3_reg_stats.items():
                            if isinstance(value, (int, float)):
                                stage3_reg_stats_sum[key] = stage3_reg_stats_sum.get(key, 0.0) + float(value)
            step_mem_prev = _record_stage_mem("after_event", out=step_mem_aux, prev=step_mem_prev)
            event_step_ms = (time.perf_counter() - t0) * 1000.0
            timings["event_ms"] += event_step_ms
            is_frame_exit = bool(is_block_exit)
            step_context = IForwardMemoryStepContext(
                step_idx=int(step.step_idx),
                source_frame_idx=int(step.source_frame_idx),
                commit_observation_memory=bool(step.commit_observation_memory),
                update_optimizer_memory=bool(step.update_optimizer_memory),
                repeat_pos_code=float(step.repeat_pos_code),
                frame_pos_code=float(step.frame_pos_code),
                rollout_pos_code=float(step.rollout_pos_code),
                global_step=int(global_step),
                is_frame_exit=bool(is_frame_exit),
                episode_visit_idx=int(getattr(step, "episode_visit_idx", -1)),
                rollout_visit_idx=int(getattr(step, "rollout_visit_idx", getattr(step, "rollout_block_rank", -1))),
                optimizer_step_idx_in_episode=int(getattr(step, "optimizer_step_idx_in_episode", -1)),
            )
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/memory"):
                if self.is_v3_gru_history_gate:
                    point_gru = getattr(self, "point_gru", None)
                    if point_gru is None or history_ema is None:
                        raise RuntimeError("IForward-v3 modules are not initialized.")
                    memory_aux = {}
                    if bool(is_block_enter):
                        memory_aux.update(history_ema.record_block_support_snapshot(event=event, local_state=local_state))
                    ctx_memory, gru_prepared, gru_read_aux = point_gru.read(
                        event=event,
                        local_state=local_state,
                        state=memory_state,
                        step_context=step_context,
                        ablation=module_ablation_name,
                    )
                    memory_aux.update(gru_read_aux)
                    short_entries = []
                elif self.is_v6_point_mamba_xcpe:
                    memory_state, ctx_memory, memory_aux = self._build_v6_context(
                        event=event,
                        local_state=local_state,
                        memory_state=memory_state,
                        step_context=step_context,
                        ablation=module_ablation_name,
                    )
                    short_entries = []
                elif self.is_stage2_0_biggs_parent_lifting:
                    ctx_memory = None
                    if self.is_stage2_3_optimizer_mamba:
                        memory_aux = {
                            "iforward/stage2_3_parent_optimizer_mamba": 1.0,
                            "iforward/biggs/memory_noop": 1.0,
                            "iforward/parent_optimizer_memory/type_id": 0.0,
                            "iforward/parent_optimizer_memory/is_gdkv": 0.0,
                            "iforward/parent_optimizer_memory/legacy_mamba_alias": 0.0,
                        }
                        if bool(getattr(self, "is_stage3_1_lowrank_gdkv", False)):
                            memory_aux["iforward/stage3_1_parent_optimizer_gdkv"] = 1.0
                            memory_aux["iforward/parent_optimizer_memory/type_id"] = 1.0
                            memory_aux["iforward/parent_optimizer_memory/is_gdkv"] = 1.0
                            memory_aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 1.0
                    elif self.is_stage2_2_parent_temporal:
                        memory_aux = {
                            "iforward/stage2_2_parent_temporal_memory_v2": 1.0,
                            "iforward/biggs/memory_noop": 1.0,
                        }
                    elif self.is_stage2_1_parent_temporal:
                        memory_aux = {
                            "iforward/stage2_1_parent_temporal_memory": 1.0,
                            "iforward/biggs/memory_noop": 1.0,
                        }
                    else:
                        memory_aux = {
                            "iforward/stage2_0_memory_bypass": 1.0,
                            "iforward/biggs/memory_noop": 1.0,
                        }
                    short_entries = []
                else:
                    memory_state, ctx_memory, memory_aux, short_entries = self.memory(
                        event=event,
                        local_state=local_state,
                        state=memory_state,
                        short_history=working_history,
                        step_context=step_context,
                        commit_observation_memory=bool(step.commit_observation_memory),
                        update_optimizer_memory=bool(step.update_optimizer_memory),
                        ablation=module_ablation_name,
                    )
            step_mem_prev = _record_stage_mem("after_memory", out=step_mem_aux, prev=step_mem_prev)
            memory_step_ms = (time.perf_counter() - t0) * 1000.0
            timings["memory_ms"] += memory_step_ms
            working_history = working_history.commit_memory_entries(
                short_entries,
                detach=bool(getattr(self.memory, "short_entry_detach", True)) if self.memory is not None else True,
            )
            old_local_state_before_update = local_state
            validation_render_only = bool(getattr(step, "validation_render_only", False))
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/update"):
                delta = None
                if bool(validation_render_only):
                    update_aux = {
                        "iforward/stage2_3/validation_render_only": 1.0,
                        "iforward/stage2_3/update_skipped_for_final_all": 1.0,
                    }
                elif self.is_v3_gru_history_gate:
                    if history_ema is None:
                        raise RuntimeError("IForward-v3 history EMA is not initialized.")
                    history_gate = getattr(self, "history_gate", None)
                    point_gru = getattr(self, "point_gru", None)
                    if history_gate is None or point_gru is None:
                        raise RuntimeError("IForward-v3 modules are not initialized.")
                    delta_raw, pred_aux = self.bridge.predict_delta(
                        local_state=local_state,
                        event=event,
                        ctx_memory=ctx_memory,
                    )
                    delta_scoped = self.bridge.apply_branch_scope_event_rows(delta_raw)
                    update_aux = {}
                    hgv2_features = None
                    if bool(getattr(self, "history_gate_v2_enabled", False)):
                        bank_valid = history_gradient_bank is not None and bool(getattr(history_gradient_bank, "valid", False))
                        source_rollout_id = (
                            int(getattr(history_gradient_bank, "source_rollout_id", -1)) if bank_valid else -1
                        )
                        rollout_id = int(getattr(resolved, "rollout_id_global", -1))
                        update_aux.update(
                            {
                                "hgv2/bank_valid": 1.0 if bank_valid else 0.0,
                                "hgv2/bank_source_history_loss": (
                                    float(getattr(history_gradient_bank, "source_history_loss", 0.0)) if bank_valid else 0.0
                                ),
                                "hgv2/bank_source_history_num_refs": (
                                    float(getattr(history_gradient_bank, "source_history_num_refs", 0)) if bank_valid else 0.0
                                ),
                                "hgv2/bank_rollout_gap": (
                                    float(rollout_id - source_rollout_id)
                                    if bank_valid and rollout_id >= 0 and source_rollout_id >= 0
                                    else 0.0
                                ),
                            }
                        )
                        if bank_valid:
                            hgv2_features = compute_history_gate_v2_features(
                                bank=history_gradient_bank,
                                event=event,
                                delta_event=delta_scoped,
                                local_state=local_state,
                                cfg=getattr(self, "history_gate_v2_cfg", {}) or {},
                            )
                            if hgv2_features is not None and isinstance(hgv2_features.aux, dict):
                                update_aux.update(
                                    {str(k): float(v) for k, v in hgv2_features.aux.items() if isinstance(v, (int, float))}
                                )
                        else:
                            for attr in HGV2_ATTRS:
                                update_aux[f"hgv2/damage_pos_ratio/{attr}"] = 0.0
                    gate_pack = history_gate(
                        event=event,
                        ctx_memory=ctx_memory,
                        history_ema=history_ema,
                        local_state=local_state,
                        grad_features=hgv2_features,
                        ablation=module_ablation_name,
                    )
                    if adc_suppression_accumulator is not None:
                        update_aux.update(
                            adc_suppression_accumulator.accumulate_from_bg_delta_gate(
                                delta_bg=delta_scoped.bg,
                                gate_bg=gate_pack.bg,
                                cfg=getattr(self, "adc_lite_cfg", {}) or {},
                            )
                        )
                    hgv2_aux: Dict[str, float] = {}
                    if hgv2_features is not None and float(getattr(self, "loss_hgv2_gate_weight", 0.0)) > 0.0:
                        hgv2_loss, hgv2_aux = history_gate_v2_auxiliary_loss(
                            gate=gate_pack,
                            features=hgv2_features,
                            cfg=getattr(self, "history_gate_v2_cfg", {}) or {},
                        )
                        hgv2_losses.append(hgv2_loss)
                        update_aux.update({str(k): float(v) for k, v in hgv2_aux.items() if isinstance(v, (int, float))})
                    if hgv2_features is not None:
                        hgv2_stats_count += 1
                        for key, value in {**(hgv2_features.aux or {}), **hgv2_aux}.items():
                            if isinstance(value, (int, float)):
                                hgv2_stats_sum[str(key)] = hgv2_stats_sum.get(str(key), 0.0) + float(value)
                    delta_gated_event = gate_delta_pack(delta_scoped, gate_pack)
                    hsp = getattr(self, "history_safe_projection", None)
                    if hsp is not None:
                        delta_safe_event, hsp_aux, hsp_loss = hsp(
                            local_state=local_state,
                            event=event,
                            delta_event=delta_gated_event,
                            resolved=resolved,
                            batch=batch,
                            step=step,
                            step_context=step_context,
                            history_ema=history_ema,
                            bridge=self.bridge,
                            probe_cache=block_hsp_cache,
                        )
                        delta_gated_event = delta_safe_event
                        hsp_losses.append(hsp_loss)
                        update_aux.update(
                            {str(k): float(v) for k, v in hsp_aux.items() if isinstance(v, (int, float))}
                        )
                        hsp_stats_count += 1
                        for key, value in hsp_aux.items():
                            if isinstance(value, (int, float)):
                                hsp_stats_sum[str(key)] = hsp_stats_sum.get(str(key), 0.0) + float(value)
                    delta = self.bridge.expand_rigid_delta(
                        delta=delta_gated_event,
                        event=event,
                        local_state=local_state,
                    )
                    local_state = self.bridge.apply_delta_only(local_state=local_state, delta=delta)
                    if isinstance(pred_aux, dict):
                        update_aux.update({str(k): float(v) for k, v in pred_aux.items() if isinstance(v, (int, float))})
                    if isinstance(gate_pack.aux, dict):
                        update_aux.update({str(k): float(v) for k, v in gate_pack.aux.items() if isinstance(v, (int, float))})
                    if bool(getattr(step, "record_update_norm", True)):
                        update_aux.update(
                            history_ema.record_update_norm(
                                delta=delta,
                                update_betas=self.v3_history_update_betas,
                            )
                        )
                    memory_state, gru_write_aux = point_gru.write_after_update(
                        prepared=gru_prepared,
                        state=memory_state,
                        delta_raw=delta_gated_event,
                        gate=gate_pack,
                        step_context=step_context,
                        ablation=module_ablation_name,
                    )
                    update_aux.update(gru_write_aux)
                    if bool(getattr(step, "commit_residual_on_exit", is_block_exit)):
                        residual_pack = self.bridge.compute_block_residual_history(
                            local_state=local_state,
                            batch=batch,
                            source_indices=list(step.source_indices),
                            source_frame_idx=int(step.source_frame_idx),
                        )
                        update_aux.update(
                            history_ema.commit_residual(
                                residual_pack,
                                residual_betas=self.v3_history_residual_betas,
                                support_min=self.v3_history_support_min,
                            )
                        )
                    if bool(getattr(step, "commit_support_on_exit", is_block_exit)):
                        update_aux.update(
                            history_ema.commit_block_support(
                                support_betas=self.v3_history_support_betas,
                                support_min=self.v3_history_support_min,
                            )
                        )
                else:
                    local_state, delta, update_aux = self.bridge.apply_update(
                        local_state=local_state,
                        event=event,
                        ctx_memory=ctx_memory,
                    )
                if self.is_stage2_0_biggs_parent_lifting and isinstance(measurement, dict):
                    parent_runtime_update_step_ms = 0.0
                    current_runtime = measurement.get("biggs_parent_runtime", biggs_parent_runtime)
                    skip_exit = bool(getattr(self, "stage2_0_biggs_skip_update_on_block_exit", True))
                    update_nonfinal = bool(getattr(self, "stage2_0_biggs_update_after_each_nonfinal_repeat", True))
                    should_update_runtime = (not bool(validation_render_only)) and current_runtime is not None and (
                        (not bool(is_block_exit) and bool(update_nonfinal))
                        or (bool(is_block_exit) and not bool(skip_exit))
                    )
                    if bool(is_block_exit) and bool(skip_exit):
                        biggs_parent_runtime = None
                    elif bool(should_update_runtime):
                        parent_runtime_t0 = time.perf_counter()
                        biggs_parent_runtime = self.bridge.update_biggs_parent_runtime(
                            runtime=current_runtime,
                            old_local_state=old_local_state_before_update,
                            new_local_state=local_state,
                        )
                        parent_runtime_update_step_ms = (time.perf_counter() - parent_runtime_t0) * 1000.0
                    else:
                        biggs_parent_runtime = current_runtime
                    update_aux["iforward/biggs/time_parent_runtime_update_ms"] = float(parent_runtime_update_step_ms)
                    update_aux["iforward/biggs/parent_runtime_update_performed"] = float(
                        1.0 if bool(should_update_runtime) else 0.0
                    )
                    timings["parent_runtime_update_ms"] += float(parent_runtime_update_step_ms)
                if self.is_stage2_3_optimizer_mamba:
                    parent_temporal = getattr(self, "parent_temporal_mamba", None)
                    spatial_event = stage2_1_parent_block_cache.get("optimizer_spatial_event")
                    fused_event = stage2_1_parent_block_cache.get("optimizer_fused_event")
                    optimizer_keys = stage2_1_parent_block_cache.get("optimizer_keys")
                    optimizer_write_requested = bool(
                        getattr(step, "optimizer_memory_write", getattr(step, "update_optimizer_memory", False))
                    )
                    if bool(validation_render_only or stage2_3_mamba_off or stage2_3_mamba_freeze_write):
                        optimizer_write_requested = False
                    if (
                        bool(optimizer_write_requested)
                        and parent_temporal is not None
                        and spatial_event is not None
                        and optimizer_keys is not None
                    ):
                        delta_for_optimizer_write = None
                        if bool(getattr(self, "stage2_3_include_delta_summary", True)):
                            runtime_for_delta = biggs_parent_runtime
                            if isinstance(measurement, dict):
                                runtime_for_delta = runtime_for_delta or measurement.get("biggs_parent_runtime", None)
                            delta_for_optimizer_write, delta_summary_aux = build_parent_delta_summary(
                                delta=delta,
                                runtime=runtime_for_delta,
                                spatial_event=spatial_event,
                                fail_fast=bool(getattr(self, "stage2_3_delta_summary_fail_fast", True)),
                            )
                            update_aux.update(
                                {str(k): float(v) for k, v in delta_summary_aux.items() if isinstance(v, (int, float))}
                            )
                        parent_temporal_state, optimizer_write_aux = parent_temporal.write(
                            spatial_event=spatial_event,
                            fused_event=fused_event,
                            state=parent_temporal_state,
                            keys=optimizer_keys,
                            visit_meta=VisitMeta.from_step(step),
                            delta=delta_for_optimizer_write,
                            **(
                                {"emit_aux_stats": bool(emit_gdkv_aux_stats)}
                                if bool(getattr(self, "is_stage3_1_lowrank_gdkv", False))
                                else {}
                            ),
                        )
                        state.parent_temporal = parent_temporal_state
                        update_aux.update(
                            {str(k): float(v) for k, v in optimizer_write_aux.items() if isinstance(v, (int, float))}
                        )
                    else:
                        update_aux["iforward/parent_optimizer_mamba/write_skipped"] = 1.0
                        if bool(getattr(self, "is_stage3_1_lowrank_gdkv", False)):
                            update_aux["iforward/parent_optimizer_gdkv/write_skipped"] = 1.0
                            update_aux["iforward/parent_optimizer_memory/type_id"] = 1.0
                            update_aux["iforward/parent_optimizer_memory/is_gdkv"] = 1.0
                            update_aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 1.0
                        else:
                            update_aux["iforward/parent_optimizer_memory/type_id"] = 0.0
                            update_aux["iforward/parent_optimizer_memory/is_gdkv"] = 0.0
                            update_aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 0.0
                if (self.is_stage2_1_parent_temporal or self.is_stage2_2_parent_temporal) and bool(is_block_exit):
                    parent_temporal = getattr(self, "parent_temporal_mamba", None)
                    commit_event = stage2_1_parent_block_cache.get("commit_event")
                    commit_keys = stage2_1_parent_block_cache.get("commit_keys")
                    is_sequence10_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
                    is_stage2_2_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_STAGE2_2_SCHEDULER_VERSION
                    should_temporal_commit = (
                        bool(getattr(step, "temporal_commit", False))
                        if bool(is_sequence10_scheduler or is_stage2_2_scheduler)
                        else True
                    )
                    if (
                        bool(should_temporal_commit)
                        and parent_temporal is not None
                        and commit_event is not None
                        and commit_keys is not None
                    ):
                        if self.is_stage2_2_parent_temporal:
                            parent_temporal_state, temporal_commit_aux = parent_temporal.commit(
                                event=commit_event,
                                state=parent_temporal_state,
                                keys=commit_keys,
                                block_id=int(stage2_1_parent_block_cache.get("commit_block_id", step.step_idx)),
                                timestamp_sec=float(
                                    stage2_1_parent_block_cache.get(
                                        "commit_timestamp_sec", float(getattr(step, "timestamp_sec", 0.0))
                                    )
                                ),
                                physical_time_advance=bool(
                                    getattr(step, "physical_time_advance", True)
                                ),
                                motion_meta=stage2_1_parent_block_cache.get("commit_motion_meta", {}),
                            )
                        else:
                            parent_temporal_state, temporal_commit_aux = parent_temporal.commit(
                                event=commit_event,
                                state=parent_temporal_state,
                                keys=commit_keys,
                                block_id=int(stage2_1_parent_block_cache.get("commit_block_id", step.step_idx)),
                            )
                        state.parent_temporal = parent_temporal_state
                        update_aux.update(
                            {str(k): float(v) for k, v in temporal_commit_aux.items() if isinstance(v, (int, float))}
                        )
                    elif bool(is_sequence10_scheduler or is_stage2_2_scheduler):
                        update_aux["iforward/parent_temporal/block_commit_skipped"] = 1.0
                    stage2_1_parent_block_cache = {}
            step_mem_prev = _record_stage_mem("after_update", out=step_mem_aux, prev=step_mem_prev)
            update_step_ms = (time.perf_counter() - t0) * 1000.0
            timings["update_ms"] += update_step_ms
            t0 = time.perf_counter()
            if delta is None:
                reg_loss = local_state.bg.means.new_tensor(0.0)
                reg_stats = {}
            else:
                reg_loss, reg_stats = self.bridge.delta_regularization(delta, local_state=local_state)
            delta_reg_step_ms = (time.perf_counter() - t0) * 1000.0
            timings["delta_reg_ms"] += delta_reg_step_ms
            reg_terms.append(reg_loss)
            for key, value in reg_stats.items():
                if isinstance(value, (int, float)):
                    reg_stats_sum[key] = reg_stats_sum.get(key, 0.0) + float(value)
            _record_stage_mem("after_delta_reg", out=step_mem_aux, prev=step_mem_prev)
            item: Dict[str, float] = {
                "k": float(step.step_idx),
                "source_frame_idx": float(step.source_frame_idx),
                "repeat_idx": float(step.repeat_idx),
                "num_source_indices": float(len(step.source_indices)),
                "commit_observation_memory": float(1.0 if step.commit_observation_memory else 0.0),
                "update_optimizer_memory": float(1.0 if step.update_optimizer_memory else 0.0),
                "is_block_enter": float(1.0 if is_block_enter else 0.0),
                "is_block_exit": float(1.0 if is_block_exit else 0.0),
                "is_frame_exit": float(1.0 if is_frame_exit else 0.0),
                "sequence_pos": float(int(getattr(step, "sequence_pos", -1))),
                "frame_gap": float(int(getattr(step, "frame_gap", 0))),
                "temporal_read": float(1.0 if bool(getattr(step, "temporal_read", True)) else 0.0),
                "temporal_commit": float(1.0 if bool(getattr(step, "temporal_commit", False)) else 0.0),
                "validation_render_only": float(1.0 if bool(getattr(step, "validation_render_only", False)) else 0.0),
                "physical_time_advance": float(
                    1.0 if bool(getattr(step, "physical_time_advance", False)) else 0.0
                ),
                "is_sequence10_repair": float(1.0 if str(getattr(step, "visit_kind", "")) == "repair" else 0.0),
                "short_entries_added": float(len(short_entries)),
                "memory_entries_after_step": float(len(working_history.memory_entries)),
                "observe_ms": float(observe_step_ms),
                "event_ms": float(event_step_ms),
                "memory_ms": float(memory_step_ms),
                "update_ms": float(update_step_ms),
                "delta_reg_ms": float(delta_reg_step_ms),
            }
            if isinstance(measurement, dict):
                item.update({str(k): float(v) for k, v in measurement.items() if isinstance(v, (int, float))})
            item.update({str(k): float(v) for k, v in memory_aux.items() if isinstance(v, (int, float))})
            item.update({str(k): float(v) for k, v in update_aux.items() if isinstance(v, (int, float))})
            item.update(step_mem_aux)
            per_step.append(item)

        rollout_mem_prev = _record_stage_mem("after_rollout_loop", out=rollout_mem_aux, prev=rollout_mem_prev)
        t0 = time.perf_counter()
        with self._nvtx_range("iforward/final_render"):
            with self._amp_fp32_context(enabled=bool(self.amp_policy.render_force_fp32)):
                (
                    final_losses,
                    final_stats,
                    pred_rgbs,
                    gt_images,
                    image_refs,
                    image_roles,
                    final_render_pack,
                ) = self._render_final_losses(
                    local_state=local_state,
                    batch=batch,
                    resolved=resolved,
                    carried_state=prior_state_for_history,
                    ablation=module_ablation_name,
                    in_rollout_history_loss_weight=float(in_rollout_history_loss_weight),
                )
        timings["final_render_ms"] += (time.perf_counter() - t0) * 1000.0
        rollout_mem_prev = _record_stage_mem("after_final_render", out=rollout_mem_aux, prev=rollout_mem_prev)
        is_sequence10_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
        is_stage2_2_scheduler = str(getattr(resolved, "scheduler_version", "")) == IFORWARD_STAGE2_2_SCHEDULER_VERSION
        is_stage2_3_scheduler = str(getattr(resolved, "scheduler_version", "")) in IFORWARD_OPTIMIZER_SEQUENCE_SCHEDULER_VERSIONS
        sequence10_bank = getattr(state, "sequence10_bank", None)
        if bool(is_sequence10_scheduler):
            if not isinstance(sequence10_bank, Sequence10HistoryBank):
                sequence10_bank = Sequence10HistoryBank.empty(device=self.device)
            else:
                sequence10_bank = sequence10_bank.to(device=self.device, dtype=local_state.bg.means.dtype)
        sequence10_per_pos_loss: Dict[int, torch.Tensor] = {}
        sequence10_after_by_pos = local_state.bg.means.new_zeros((0,))
        sequence10_after_valid = torch.zeros((0,), device=local_state.bg.means.device, dtype=torch.bool)
        sequence10_bank_damage_loss = local_state.bg.means.new_tensor(0.0)
        sequence10_bank_damage_num_pos = 0
        stage2_2_bank = getattr(state, "stage2_2_bank", None)
        if bool(is_stage2_2_scheduler):
            if not isinstance(stage2_2_bank, EpisodeHistoryBankV2):
                stage2_2_bank = EpisodeHistoryBankV2.empty(device=self.device)
        stage2_2_per_pos_loss: Dict[int, torch.Tensor] = {}
        stage2_2_after_by_pos = local_state.bg.means.new_zeros((0,))
        stage2_2_after_valid = torch.zeros((0,), device=local_state.bg.means.device, dtype=torch.bool)
        stage2_2_bank_damage_loss = local_state.bg.means.new_tensor(0.0)
        stage2_2_bank_damage_num_pos = 0.0
        stage2_2_damage_stats: Dict[str, float] = {}
        stage2_3_bank = getattr(state, "stage2_3_bank", None)
        if bool(is_stage2_3_scheduler):
            if not isinstance(stage2_3_bank, EpisodeHistoryBankV3):
                stage2_3_bank = EpisodeHistoryBankV3.empty(device=self.device)
        stage2_3_per_pos_loss: Dict[int, torch.Tensor] = {}
        stage2_3_after_by_pos = local_state.bg.means.new_zeros((0,))
        stage2_3_after_valid = torch.zeros((0,), device=local_state.bg.means.device, dtype=torch.bool)
        stage2_3_bank_damage_loss = local_state.bg.means.new_tensor(0.0)
        stage2_3_bank_damage_num_pos = 0.0
        stage2_3_damage_stats: Dict[str, float] = {}
        if bool(is_sequence10_scheduler):
            sequence10_per_pos_loss, sequence10_after_by_pos, sequence10_after_valid = (
                self._sequence10_current_per_pos_loss_from_final_pack(
                    resolved=resolved,
                    final_pack=final_render_pack,
                    ref=local_state.bg.means,
                )
            )
        if bool(is_stage2_2_scheduler):
            stage2_2_per_pos_loss, stage2_2_after_by_pos, stage2_2_after_valid = self._stage2_2_per_pos_loss(
                resolved=resolved,
                final_pack=final_render_pack,
                ref=local_state.bg.means,
            )
        if bool(is_stage2_3_scheduler):
            stage2_3_per_pos_loss, stage2_3_after_by_pos, stage2_3_after_valid = self._stage2_3_per_pos_loss(
                resolved=resolved,
                final_pack=final_render_pack,
                ref=local_state.bg.means,
            )
        if reg_terms:
            delta_reg = torch.stack(reg_terms).mean()
        else:
            delta_reg = local_state.bg.means.new_tensor(0.0)
        if stage3_reg_terms:
            stage3_gather_reg = torch.stack([x.to(device=local_state.bg.means.device) for x in stage3_reg_terms]).mean()
        else:
            stage3_gather_reg = local_state.bg.means.new_tensor(0.0)
        if hsp_losses:
            hsp_damage_loss = torch.stack(hsp_losses).mean()
        else:
            hsp_damage_loss = local_state.bg.means.new_tensor(0.0)
        if hgv2_losses:
            hgv2_gate_loss = torch.stack(hgv2_losses).mean()
        else:
            hgv2_gate_loss = local_state.bg.means.new_tensor(0.0)
        history_damage_loss = local_state.bg.means.new_tensor(0.0)
        history_damage_num_refs = 0
        if history_damage_probe is not None and history_damage_probe.valid:
            with self._amp_fp32_context(enabled=bool(self.amp_policy.render_force_fp32)):
                _after_loss, _after_stats, after_per_ref = self.bridge.render_loss(
                    local_state=local_state,
                    batch=batch,
                    target_indices=list(history_damage_probe.target_indices),
                    mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
                    return_per_ref_loss=True,
                )
            history_damage_loss = history_damage_hinge(
                after_per_ref=after_per_ref,
                before_per_ref=history_damage_probe.before_per_ref.to(device=after_per_ref.device, dtype=after_per_ref.dtype),
                margin=float(getattr(self, "loss_history_damage_margin", 0.0)),
            )
            history_damage_num_refs = int(after_per_ref.numel())
        if (
            bool(is_sequence10_scheduler)
            and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "repair"
            and isinstance(sequence10_bank, Sequence10HistoryBank)
            and float(history_damage_weight) > 0.0
        ):
            before_by_pos, bank_valid = sequence10_bank.before_for_positions(
                range(10),
                device=sequence10_after_by_pos.device,
                dtype=sequence10_after_by_pos.dtype,
            )
            valid = bank_valid.to(device=sequence10_after_valid.device) & sequence10_after_valid
            sequence10_bank_damage_loss = sequence10_damage_hinge_from_bank(
                after_loss=sequence10_after_by_pos,
                before_loss=before_by_pos,
                valid=valid,
                margin=float(getattr(self, "loss_history_damage_margin", 0.0)),
            )
            sequence10_bank_damage_num_pos = int(valid.sum().detach().item())
            history_damage_loss = history_damage_loss + sequence10_bank_damage_loss
        stage2_2_phase = str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", ""))
        if (
            bool(is_stage2_2_scheduler)
            and stage2_2_phase == "causal"
            and isinstance(stage2_2_bank, EpisodeHistoryBankV2)
            and float(history_damage_weight) > 0.0
        ):
            terms: List[torch.Tensor] = []
            valid_mask = stage2_2_after_valid.to(device=stage2_2_after_by_pos.device, dtype=torch.bool).reshape(-1)
            losses_by_pos = stage2_2_after_by_pos.reshape(-1)
            for pos in range(min(10, int(losses_by_pos.numel()))):
                if pos >= int(valid_mask.numel()) or not bool(valid_mask[pos].detach().item()):
                    continue
                entry = stage2_2_bank.entries.get(int(pos))
                if entry is None or not bool(entry.seen):
                    continue
                before = entry.last_loss.to(device=losses_by_pos.device, dtype=losses_by_pos.dtype)
                terms.append(torch.relu(losses_by_pos[pos] - before - float(getattr(self, "loss_history_damage_margin", 0.0))))
            if terms:
                stacked = torch.stack(terms)
                stage2_2_bank_damage_loss = stacked.mean()
                stage2_2_bank_damage_num_pos = float(len(terms))
                detached = stacked.detach().float()
                stage2_2_damage_stats = {
                    "stage2_2/last_damage_num_pos": float(len(terms)),
                    "stage2_2/last_damage_mean": float(detached.mean().item()),
                    "stage2_2/last_damage_p90": float(torch.quantile(detached, 0.9).item()),
                    "stage2_2/last_damage_max": float(detached.max().item()),
                }
                history_damage_loss = history_damage_loss + stage2_2_bank_damage_loss
        if (
            bool(is_stage2_2_scheduler)
            and stage2_2_phase == "repair"
            and isinstance(stage2_2_bank, EpisodeHistoryBankV2)
            and float(history_damage_weight) > 0.0
        ):
            stage2_2_bank_damage_loss, stage2_2_damage_stats = history_damage_hinge_v2(
                repair_losses=stage2_2_after_by_pos,
                bank=stage2_2_bank,
                positions=range(10),
                valid=stage2_2_after_valid,
                margin=float(getattr(self, "loss_history_damage_margin", 0.0)),
            )
            stage2_2_bank_damage_num_pos = float(stage2_2_damage_stats.get("stage2_2/best_damage_num_pos", 0.0))
            history_damage_loss = history_damage_loss + stage2_2_bank_damage_loss
        stage2_3_phase = str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", ""))
        if (
            bool(is_stage2_3_scheduler)
            and stage2_3_phase == "repair"
            and isinstance(stage2_3_bank, EpisodeHistoryBankV3)
            and float(history_damage_weight) > 0.0
        ):
            stage2_3_bank_damage_loss, stage2_3_damage_stats = history_damage_hinge_v3(
                repair_losses=stage2_3_after_by_pos,
                bank=stage2_3_bank,
                positions=range(int(stage2_3_after_by_pos.numel())),
                valid=stage2_3_after_valid,
                margin=float(getattr(self, "loss_history_damage_margin", 0.0)),
            )
            stage2_3_bank_damage_num_pos = float(stage2_3_damage_stats.get("stage2_3/best_damage_num_pos", 0.0))
            history_damage_loss = history_damage_loss + stage2_3_bank_damage_loss
        final_losses["delta_regularization"] = delta_reg
        final_losses["stage3_gather_regularization"] = stage3_gather_reg
        final_losses["hsp_damage_loss"] = hsp_damage_loss
        final_losses["hgv2_gate"] = hgv2_gate_loss
        final_losses["history_damage"] = history_damage_loss
        rollout_mem_prev = _record_stage_mem("after_loss_terms", out=rollout_mem_aux, prev=rollout_mem_prev)
        next_sequence10_bank = sequence10_bank
        if bool(is_sequence10_scheduler) and isinstance(next_sequence10_bank, Sequence10HistoryBank):
            next_sequence10_bank.update(sequence10_per_pos_loss)
            next_sequence10_bank = next_sequence10_bank.detach()
        next_stage2_2_bank = stage2_2_bank
        if (
            bool(is_stage2_2_scheduler)
            and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "causal"
            and isinstance(next_stage2_2_bank, EpisodeHistoryBankV2)
        ):
            for pos, loss_value in stage2_2_per_pos_loss.items():
                psnr = -10.0 * torch.log10(loss_value.detach().clamp_min(1.0e-8))
                next_stage2_2_bank = next_stage2_2_bank.update(
                    sequence_pos=int(pos),
                    loss=loss_value,
                    psnr=psnr,
                    rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                )
            next_stage2_2_bank = next_stage2_2_bank.detach()
        next_stage2_3_bank = stage2_3_bank
        if bool(is_stage2_3_scheduler) and isinstance(next_stage2_3_bank, EpisodeHistoryBankV3):
            for pos, loss_value in stage2_3_per_pos_loss.items():
                psnr = -10.0 * torch.log10(loss_value.detach().clamp_min(1.0e-8))
                next_stage2_3_bank = next_stage2_3_bank.update(
                    sequence_pos=int(pos),
                    loss=loss_value,
                    psnr=psnr,
                    rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                )
            next_stage2_3_bank = next_stage2_3_bank.detach()

        next_history_gradient_bank = None
        if self.is_v3_gru_history_gate and bool(getattr(self, "history_gate_v2_enabled", False)):
            history_num_refs = int(float(final_stats.get("in_rollout_history_num_refs", 0.0)))
            next_history_gradient_bank = build_history_gradient_bank_from_loss(
                loss_history=final_losses["in_rollout_history"],
                final_local_state=local_state,
                rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                history_num_refs=history_num_refs,
                cfg=getattr(self, "history_gate_v2_cfg", {}) or {},
            )
        next_adc_bank = None
        adc_build_stats: Dict[str, float] = {}
        if bool(getattr(self, "adc_lite_enabled", False)):
            adc_policy_allowed, _ = self._adc_lite_policy_stats(global_step=global_step, resolved=resolved)
            if not bool(adc_disabled_by_ablation) and bool(adc_policy_allowed):
                adc_aabb_min, adc_aabb_max = self._adc_lite_aabb(local_state)
                num_current_refs = int(float(final_stats.get("current_num_refs", len(resolved.current_latest_target_indices))))
                num_history_refs = int(
                    float(final_stats.get("in_rollout_history_num_refs", len(resolved.history_rollout_target_indices)))
                )
                if bool(getattr(self, "adc_lite_gate_suppressed", False)):
                    next_adc_bank = build_gate_suppressed_adc_bank(
                        accumulator=adc_suppression_accumulator,
                        final_local_state=local_state,
                        history_ema=history_ema,
                        cfg=getattr(self, "adc_lite_cfg", {}) or {},
                        rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                        episode_id=int(getattr(resolved, "episode_id", -1)),
                        num_current_refs=num_current_refs,
                        num_history_refs=num_history_refs,
                        adc_meta=getattr(state, "adc_meta", None),
                        aabb_min=adc_aabb_min,
                        aabb_max=adc_aabb_max,
                        diagnostics=adc_build_stats,
                    )
                else:
                    next_adc_bank = build_adc_lite_bank_from_losses(
                        loss_current=final_losses["current"],
                        loss_history=final_losses.get("in_rollout_history"),
                        final_local_state=local_state,
                        cfg=getattr(self, "adc_lite_cfg", {}) or {},
                        rollout_id=int(getattr(resolved, "rollout_id_global", -1)),
                        episode_id=int(getattr(resolved, "episode_id", -1)),
                        num_current_refs=num_current_refs,
                        num_history_refs=num_history_refs,
                        adc_meta=getattr(state, "adc_meta", None),
                        aabb_min=adc_aabb_min,
                        aabb_max=adc_aabb_max,
                    )

        with self._amp_fp32_context(enabled=bool(self.amp_policy.loss_force_fp32)):
            final_losses = {name: value.float() if torch.is_tensor(value) else value for name, value in final_losses.items()}
            total = (
                self.loss_current_weight * final_losses["current"]
                + self.loss_nearby_weight * final_losses["nearby"]
                + float(in_rollout_history_loss_weight) * final_losses["in_rollout_history"]
                + self.loss_short_window_history_weight * final_losses["short_window_history"]
                + self.loss_delta_reg_weight * final_losses["delta_regularization"]
                + final_losses["stage3_gather_regularization"]
                + self.hsp_damage_loss_weight * final_losses["hsp_damage_loss"]
                + float(getattr(self, "loss_hgv2_gate_weight", 0.0)) * final_losses["hgv2_gate"]
                + float(history_damage_weight) * final_losses["history_damage"]
            )
        if not torch.isfinite(total).all():
            raise RuntimeError("IForward rollout loss became NaN/Inf.")
        rollout_mem_prev = _record_stage_mem("after_total_loss", out=rollout_mem_aux, prev=rollout_mem_prev)

        history_indices = tuple(()) if self.is_v3_gru_history_gate else tuple(
            resolved.history_commit_target_indices or resolved.current_target_indices
        )
        history = working_history.commit_targets(batch, history_indices)
        history_entries_after = int(len(history.entries))
        memory_entries_after = int(len(history.memory_entries))
        next_state = IForwardState(
            local_gs=local_state,
            memory=memory_state,
            history=history,
            scene_id=int(resolved.scene_id),
            segment_id=int(resolved.segment_id),
            episode_id=int(resolved.episode_id),
            history_ema=history_ema,
            history_gradient_bank=next_history_gradient_bank,
            adc_bank=next_adc_bank,
            adc_meta=getattr(state, "adc_meta", None),
            biggs_state=getattr(state, "biggs_state", None),
            parent_temporal=parent_temporal_state,
            sequence10_bank=next_sequence10_bank,
            stage2_2_bank=next_stage2_2_bank,
            stage2_3_bank=next_stage2_3_bank,
            node_state_bg=state.node_state_bg,
            node_state_distant=state.node_state_distant,
            node_state_rigid=state.node_state_rigid,
        )
        memory_tokens = memory_state.count_tokens() if hasattr(memory_state, "count_tokens") else {}
        if (
            self.is_stage2_1_parent_temporal
            or self.is_stage2_2_parent_temporal
            or self.is_stage2_3_optimizer_mamba
        ) and parent_temporal_state is not None:
            memory_tokens = {**memory_tokens, **parent_temporal_state.count_tokens()}
        if self.is_v3_gru_history_gate and history_ema is not None:
            memory_tokens = {**memory_tokens, **history_ema.count_tokens()}
        if isinstance(next_stage2_2_bank, EpisodeHistoryBankV2):
            memory_tokens = {**memory_tokens, **next_stage2_2_bank.count_tokens()}
        if isinstance(next_stage2_3_bank, EpisodeHistoryBankV3):
            memory_tokens = {**memory_tokens, **next_stage2_3_bank.count_tokens()}
        resolved_meta = dict(getattr(resolved, "meta", {}) or {})
        repeat0_steps = [step for step in list(resolved.steps or []) if int(getattr(step, "repeat_idx", 0)) == 0]
        actual_blocks_per_rollout = int(
            resolved_meta.get(
                "actual_blocks_per_rollout",
                len(repeat0_steps) if repeat0_steps else (len(resolved.window_block_ids) if resolved.window_block_ids else 0),
            )
        )
        stage2_3_rollout_positions = [
            int(x)
            for x in list(
                resolved_meta.get(
                    "rollout_positions",
                    resolved_meta.get("sequence_positions", [int(getattr(step, "sequence_pos", -1)) for step in repeat0_steps]),
                )
                or []
            )
        ]
        stage2_3_repeat_budgets = [
            int(x)
            for x in list(
                resolved_meta.get("repeat_budgets", [int(getattr(step, "repeat_budget", 1)) for step in repeat0_steps])
                or []
            )
        ]
        stage2_3_frame_gaps = [
            int(x)
            for x in list(
                resolved_meta.get("frame_gaps", [int(getattr(step, "frame_gap", 0)) for step in list(resolved.steps or [])])
                or []
            )
        ]
        stage2_3_visit_kind_to_id = {"bootstrap": 0, "assimilate": 1, "assimilation": 1, "repair": 2, "repeat_stability": 3}
        stage2_3_visit_kind_ids = [
            int(stage2_3_visit_kind_to_id.get(str(getattr(step, "visit_kind", "")), -1))
            for step in list(resolved.steps or [])
        ]
        stage2_3_visit_kind_id_mean = (
            float(sum(stage2_3_visit_kind_ids)) / float(len(stage2_3_visit_kind_ids))
            if stage2_3_visit_kind_ids
            else 0.0
        )
        stats: Dict[str, Any] = {
            **final_stats,
            "inner_K": int(resolved.inner_K),
            "ablation": ablation_name,
            "scheduler_version": str(resolved.scheduler_version),
            "rollouts_per_episode": int(resolved.rollouts_per_episode),
            "window_start": int(resolved.window_start),
            "window_end": int(resolved.window_end),
            "window_hash": int(resolved.window_hash),
            "window_revisit_count": int(resolved.window_revisit_count),
            "unique_windows_seen": int(resolved.unique_windows_seen),
            "is_repeated_window": bool(resolved.is_repeated_window),
            "blocks_per_rollout": int(actual_blocks_per_rollout),
            "repeats_per_block": int(resolved.steps[0].repeats_per_block) if resolved.steps else 0,
            "num_source_views": int(len(resolved.source_refs)),
            "num_targets": int(len(resolved.target_refs)),
            "num_gaussians_bg": int(local_state.bg.means.shape[0]),
            "num_gaussians_distant": int(local_state.distant.means.shape[0]) if local_state.distant is not None else 0,
            "num_gaussians_rigid": int(local_state.rigid.means.shape[0]) if local_state.rigid is not None else 0,
            "num_gaussians_sky": 0,
            "history_entries": int(len(history.entries)),
            "history_entries_before": int(history_entries_before),
            "history_entries_after": int(history_entries_after),
            "memory_entries_before": int(memory_entries_before),
            "memory_entries_after": int(memory_entries_after),
            **adc_apply_stats,
            "loss_weight/current": float(self.loss_current_weight),
            "loss_weight/nearby": float(self.loss_nearby_weight),
            "loss_weight/in_rollout_history": float(in_rollout_history_loss_weight),
            "loss_weight/in_rollout_history_base": float(self.loss_in_rollout_history_weight),
            "loss_weight/in_rollout_history_warmup_factor": (
                float(in_rollout_history_loss_weight) / float(self.loss_in_rollout_history_weight)
                if float(self.loss_in_rollout_history_weight) > 0.0
                else 0.0
            ),
            "loss_weight/short_window_history": float(self.loss_short_window_history_weight),
            "loss_weight/delta_regularization": float(self.loss_delta_reg_weight),
            "loss_weight/stage3_offset_l2": float(getattr(self, "stage3_offset_l2_weight", 0.0)),
            "loss_weight/stage3_out_of_bounds": float(getattr(self, "stage3_out_of_bounds_weight", 0.0)),
            "loss_weight/hsp_damage": float(self.hsp_damage_loss_weight),
            "loss_weight/hgv2_gate": float(getattr(self, "loss_hgv2_gate_weight", 0.0)),
            "loss_weight/history_damage": float(history_damage_weight),
            "iforward/stage3/loss_gather_regularization": float(stage3_gather_reg.detach().item())
            if bool(getattr(self, "is_stage3_0_full_sparse_gather_lift", False))
            else 0.0,
            "hsp/damage_loss": float(hsp_damage_loss.detach().item()) if hsp_damage_loss.numel() else 0.0,
            "hgv2/loss_gate_aux": float(hgv2_gate_loss.detach().item()) if hgv2_gate_loss.numel() else 0.0,
            "history_damage/loss": float(history_damage_loss.detach().item()) if history_damage_loss.numel() else 0.0,
            "history_damage/num_refs": float(history_damage_num_refs),
            "sequence10/best_damage_loss": (
                float(sequence10_bank_damage_loss.detach().item())
                if bool(is_sequence10_scheduler) and sequence10_bank_damage_loss.numel()
                else 0.0
            ),
            "sequence10/best_damage_num_pos": float(sequence10_bank_damage_num_pos),
            "sequence10/bank_valid_count": (
                float(next_sequence10_bank.valid.detach().to(dtype=torch.float32).sum().item())
                if isinstance(next_sequence10_bank, Sequence10HistoryBank)
                else 0.0
            ),
            "sequence10/bank_update_count": float(len(sequence10_per_pos_loss)),
            "stage2_2/best_damage_loss": (
                float(stage2_2_bank_damage_loss.detach().item())
                if bool(is_stage2_2_scheduler) and stage2_2_phase == "repair" and stage2_2_bank_damage_loss.numel()
                else 0.0
            ),
            "stage2_2/best_damage_num_pos": (
                float(stage2_2_bank_damage_num_pos) if str(stage2_2_phase) == "repair" else 0.0
            ),
            "stage2_2/best_damage_p90": float(stage2_2_damage_stats.get("stage2_2/best_damage_p90", 0.0)),
            "stage2_2/best_damage_max": float(stage2_2_damage_stats.get("stage2_2/best_damage_max", 0.0)),
            "stage2_2/last_damage_loss": (
                float(stage2_2_bank_damage_loss.detach().item())
                if bool(is_stage2_2_scheduler) and stage2_2_phase == "causal" and stage2_2_bank_damage_loss.numel()
                else 0.0
            ),
            "stage2_2/last_damage_num_pos": (
                float(stage2_2_bank_damage_num_pos) if str(stage2_2_phase) == "causal" else 0.0
            ),
            "stage2_2/last_damage_p90": float(stage2_2_damage_stats.get("stage2_2/last_damage_p90", 0.0)),
            "stage2_2/last_damage_max": float(stage2_2_damage_stats.get("stage2_2/last_damage_max", 0.0)),
            "stage2_2/bank_valid_count": (
                float(sum(1 for item in getattr(next_stage2_2_bank, "entries", {}).values() if bool(getattr(item, "seen", False))))
                if isinstance(next_stage2_2_bank, EpisodeHistoryBankV2)
                else 0.0
            ),
            "stage2_2/bank_update_count": float(len(stage2_2_per_pos_loss)),
            "stage2_2/bootstrap/current_raw": (
                float(final_losses["current"].detach().item())
                if bool(is_stage2_2_scheduler)
                and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "bootstrap"
                else 0.0
            ),
            "stage2_2/causal/current_raw": (
                float(final_losses["current"].detach().item())
                if bool(is_stage2_2_scheduler)
                and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "causal"
                else 0.0
            ),
            "stage2_2/causal/history_raw": (
                float(final_losses["in_rollout_history"].detach().item())
                if bool(is_stage2_2_scheduler)
                and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "causal"
                else 0.0
            ),
            "stage2_2/repair/current_all10": (
                float(final_losses["current"].detach().item())
                if bool(is_stage2_2_scheduler)
                and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "repair"
                else 0.0
            ),
            "stage2_2/repair/current_raw": (
                float(final_losses["current"].detach().item())
                if bool(is_stage2_2_scheduler)
                and str((getattr(resolved, "meta", {}) or {}).get("scheduler_phase", "")) == "repair"
                else 0.0
            ),
            "stage2_2/monitor_fixed_objective": (
                float((final_losses["current"] + float(self.loss_in_rollout_history_weight) * final_losses["in_rollout_history"]).detach().item())
                if bool(is_stage2_2_scheduler)
                else 0.0
            ),
            "stage2_3/best_damage_loss": (
                float(stage2_3_bank_damage_loss.detach().item())
                if bool(is_stage2_3_scheduler) and stage2_3_phase == "repair" and stage2_3_bank_damage_loss.numel()
                else 0.0
            ),
            "stage2_3/best_damage_num_pos": (
                float(stage2_3_bank_damage_num_pos) if str(stage2_3_phase) == "repair" else 0.0
            ),
            "stage2_3/best_damage_p90": float(stage2_3_damage_stats.get("stage2_3/best_damage_p90", 0.0)),
            "stage2_3/best_damage_max": float(stage2_3_damage_stats.get("stage2_3/best_damage_max", 0.0)),
            "stage2_3/bank_valid_count": (
                float(sum(1 for item in getattr(next_stage2_3_bank, "entries", {}).values() if bool(getattr(item, "seen", False))))
                if isinstance(next_stage2_3_bank, EpisodeHistoryBankV3)
                else 0.0
            ),
            "stage2_3/bank_update_count": float(len(stage2_3_per_pos_loss)),
            "stage2_3/current_raw": (
                float(final_losses["current"].detach().item()) if bool(is_stage2_3_scheduler) else 0.0
            ),
            "stage2_3/history_raw": (
                float(final_losses["in_rollout_history"].detach().item()) if bool(is_stage2_3_scheduler) else 0.0
            ),
            "stage2_3/monitor_fixed_objective": (
                float((final_losses["current"] + float(self.loss_in_rollout_history_weight) * final_losses["in_rollout_history"]).detach().item())
                if bool(is_stage2_3_scheduler)
                else 0.0
            ),
            "iforward/stage2_3/phase": stage2_3_phase if bool(is_stage2_3_scheduler) else "",
            "iforward/stage2_3/rollout_phase": str(resolved_meta.get("rollout_phase", "")) if bool(is_stage2_3_scheduler) else "",
            "iforward/stage2_3/rollout_positions": stage2_3_rollout_positions if bool(is_stage2_3_scheduler) else [],
            "iforward/stage2_3/episode_positions": (
                [int(x) for x in list(resolved_meta.get("episode_positions", resolved.window_block_ids or []) or [])]
                if bool(is_stage2_3_scheduler)
                else []
            ),
            "iforward/stage2_3/history_positions": (
                [int(x) for x in list(resolved_meta.get("history_positions", []) or [])] if bool(is_stage2_3_scheduler) else []
            ),
            "iforward/stage2_3/repair_positions": (
                [int(x) for x in list(resolved_meta.get("repair_positions", []) or [])] if bool(is_stage2_3_scheduler) else []
            ),
            "iforward/stage2_3/repeat_budgets": stage2_3_repeat_budgets if bool(is_stage2_3_scheduler) else [],
            "iforward/stage2_3/frame_gaps": stage2_3_frame_gaps if bool(is_stage2_3_scheduler) else [],
            "iforward/stage2_3/visit_kinds": (
                [str(x) for x in list(resolved_meta.get("visit_kinds", [str(getattr(step, "visit_kind", "")) for step in list(resolved.steps or [])]) or [])]
                if bool(is_stage2_3_scheduler)
                else []
            ),
            "iforward/stage2_3/visit_kind_id_mean": (
                float(stage2_3_visit_kind_id_mean) if bool(is_stage2_3_scheduler) else 0.0
            ),
            "iforward/stage2_3/repair_round_idx": (
                int(resolved_meta.get("repair_round_idx", -1)) if bool(is_stage2_3_scheduler) else -1
            ),
            "iforward/stage2_3/repair_pattern_name": (
                str(resolved_meta.get("repair_pattern_name", "")) if bool(is_stage2_3_scheduler) else ""
            ),
            "iforward/stage2_3/actual_blocks_per_rollout": (
                int(actual_blocks_per_rollout) if bool(is_stage2_3_scheduler) else 0
            ),
            "iforward/stage2_3/sequence_length": (
                int(resolved_meta.get("sequence_length", getattr(resolved, "episode_num_blocks", 0)))
                if bool(is_stage2_3_scheduler)
                else 0
            ),
            "hgv2/enabled": bool(getattr(self, "history_gate_v2_enabled", False)),
            "hgv2/bank_valid": (
                1.0
                if history_gradient_bank is not None and bool(getattr(history_gradient_bank, "valid", False))
                else 0.0
            ),
            "hgv2/bank_source_history_loss": (
                float(getattr(history_gradient_bank, "source_history_loss", 0.0))
                if history_gradient_bank is not None and bool(getattr(history_gradient_bank, "valid", False))
                else 0.0
            ),
            "hgv2/bank_source_history_num_refs": (
                float(getattr(history_gradient_bank, "source_history_num_refs", 0))
                if history_gradient_bank is not None and bool(getattr(history_gradient_bank, "valid", False))
                else 0.0
            ),
            "hgv2/bank_rollout_gap": (
                float(int(getattr(resolved, "rollout_id_global", -1)) - int(getattr(history_gradient_bank, "source_rollout_id", -1)))
                if history_gradient_bank is not None
                and bool(getattr(history_gradient_bank, "valid", False))
                and int(getattr(resolved, "rollout_id_global", -1)) >= 0
                and int(getattr(history_gradient_bank, "source_rollout_id", -1)) >= 0
                else 0.0
            ),
            "hgv2/next_bank_valid": (
                1.0
                if next_history_gradient_bank is not None and bool(getattr(next_history_gradient_bank, "valid", False))
                else 0.0
            ),
            "hgv2/grad_prior_scale": (
                float(getattr(self.history_gate, "grad_prior_scale").detach().item())
                if self.is_v3_gru_history_gate
                and getattr(self.history_gate, "grad_prior_scale", None) is not None
                else 0.0
            ),
            "memory_tokens": memory_tokens,
        }
        if self.is_v3_gru_history_gate and history_ema is not None:
            stats.update(history_ema.stats())
        current_psnr = stats.get("current_psnr")
        history_psnr = stats.get("history_rollout_psnr")
        short_psnr = stats.get("short_window_history_psnr")
        if current_psnr is not None and history_psnr is not None:
            stats["psnr_gap/current_minus_rollout_history"] = float(current_psnr) - float(history_psnr)
        if current_psnr is not None and short_psnr is not None:
            stats["psnr_gap/current_minus_short_history"] = float(current_psnr) - float(short_psnr)
        stats.update({key: float(value) for key, value in timings.items()})
        stats.update(rollout_mem_aux)
        per_step_metric_keys = sorted(
            {
                str(key)
                for item in per_step
                for key, value in item.items()
                if isinstance(value, (int, float))
                and (
                    str(key).startswith("iforward/")
                    or str(key).startswith("num_parent_")
                    or str(key) == "src_backproject_pass_count"
                )
            }
        )
        for key in per_step_metric_keys:
            values = [
                float(item[key])
                for item in per_step
                if key in item and isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))
            ]
            if not values:
                continue
            stats[str(key)] = float(values[-1])
            stats[f"{key}_last"] = float(values[-1])
            stats[f"{key}_mean"] = float(sum(values) / float(len(values)))
            if str(key).startswith("iforward/forward_mem/"):
                stats[f"{key}_max"] = float(max(values))
        if bool(emit_forward_mem_aux):
            for item in per_step:
                raw_k = item.get("k", None)
                if raw_k is None:
                    continue
                try:
                    k = int(raw_k)
                except Exception:
                    continue
                for key, value in item.items():
                    if not (
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and str(key).startswith("iforward/forward_mem/")
                        and (
                            str(key).endswith("_allocated_mb")
                            or str(key).endswith("_allocated_delta_mb")
                            or str(key).endswith("_max_allocated_mb")
                        )
                    ):
                        continue
                    short_key = str(key)[len("iforward/forward_mem/") :]
                    stats[f"iforward/forward_mem/k{k}/{short_key}"] = float(value)
        for key, value in reg_stats_sum.items():
            stats[f"delta_reg/{key}_mean"] = float(value) / float(max(len(reg_terms), 1))
        for key, value in stage3_reg_stats_sum.items():
            stats[str(key)] = float(value) / float(max(len(stage3_reg_terms), 1))
        for key, value in hsp_stats_sum.items():
            stats[str(key)] = float(value) / float(max(int(hsp_stats_count), 1))
        for key, value in hgv2_stats_sum.items():
            stats[str(key)] = float(value) / float(max(int(hgv2_stats_count), 1))
        if bool(getattr(self, "history_gate_v2_enabled", False)):
            for attr in HGV2_ATTRS:
                stats.setdefault(f"hgv2/damage_pos_ratio/{attr}", 0.0)
                stats.setdefault(f"hgv2/gate_harmful_mean/{attr}", 0.0)
                stats.setdefault(f"hgv2/gate_safe_mean/{attr}", 0.0)
        adc_applied_compare_stats: Dict[str, Any] = {}
        if float(adc_apply_stats.get(f"{ADC_STAT_PREFIX}/applied", 0.0)) > 0.0:
            for key in (
                "adc_suppressed/parent_gate_mean",
                "adc_suppressed/all_gate_mean",
                "adc_suppressed/parent_delta_demand_mean",
                "adc_suppressed/all_delta_demand_mean",
                "adc_suppressed/parent_support_mean",
                "adc_suppressed/all_support_mean",
                "adc_suppressed/selected_parent_suppression_rank_percentile",
                "adc_suppressed/gate_distribution_p20",
                "adc_suppressed/gate_distribution_p50",
                "adc_suppressed/gate_distribution_p80",
                "adc/raw_score/selected_rank_percentile",
                "adc/planning_score/selected_rank_percentile",
                "adc/final_score/selected_rank_percentile",
                "adc/raw_score/parent_mean",
                "adc/planning_score/parent_mean",
                "adc/final_score/parent_mean",
                "adc/parent_gate_mean",
                "adc/all_gate_mean",
                "adc/parent_delta_demand_mean",
                "adc/all_delta_demand_mean",
                "adc/parent_gate_contrast",
                "adc/gate_distribution_p20",
                "adc/gate_distribution_p50",
                "adc/gate_distribution_p80",
            ):
                if key in adc_apply_stats:
                    adc_applied_compare_stats[key] = adc_apply_stats[key]
        if adc_suppression_accumulator is not None:
            adc_budget_cfg = cfg_get(getattr(self, "adc_lite_cfg", {}) or {}, "budget", {}) or {}
            adc_topk = int(cfg_get(adc_budget_cfg, "max_new_points_per_rollout", 1000))
            stats.update(adc_suppression_accumulator.stats(topk=adc_topk))
        stats.update(adc_build_stats)
        stats.update(adc_bank_stats(next_adc_bank))
        stats.update(adc_applied_compare_stats)
        return IForwardRolloutOutput(
            loss=total,
            next_state=next_state,
            resolved=resolved,
            per_step=per_step,
            losses=final_losses,
            stats=stats,
            pred_rgbs=pred_rgbs,
            gt_images=gt_images,
            image_refs=image_refs,
            image_roles=image_roles,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward_rollout(batch).to_legacy_dict()
