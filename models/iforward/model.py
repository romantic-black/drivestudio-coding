from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import DeltaPack, LocalGSState

from .bridge import IForwardStage6Bridge
from .memory import IForwardMemoryStepContext, IForwardSceneMemory
from .resolver import IForwardBatchResolver, IForwardResolvedBatch
from .state import IForwardMemoryState, IForwardShortWindowHistory, IForwardState
from .utils import cfg_ensure_child, cfg_get, cfg_set, clone_config


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
        self.resolver = resolver if resolver is not None else IForwardBatchResolver()

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
        debug_cfg = cfg_get(iforward_cfg, "debug", {}) or {}
        self.enable_nvtx_ranges = bool(cfg_get(debug_cfg, "nvtx_ranges", False))
        memory_cfg = cfg_get(iforward_cfg, "memory", {}) or {}
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
        history_cfg = cfg_get(iforward_cfg, "short_window_history", {}) or {}
        self.history_max_entries = int(cfg_get(history_cfg, "max_entries", 24))
        self.history_max_memory_entries = int(cfg_get(history_cfg, "max_memory_entries", 8))
        loss_cfg = cfg_get(iforward_cfg, "loss", {}) or {}
        self.loss_current_weight = float(cfg_get(cfg_get(loss_cfg, "current", {}) or {}, "weight", 1.0))
        self.loss_nearby_weight = float(
            cfg_get(
                cfg_get(loss_cfg, "nearby", {}) or {},
                "weight",
                _loss_float(config, ["losses", "phase_a", "nearby_render", "weight"], 0.25) if config is not None else 0.25,
            )
        )
        self.loss_in_rollout_history_weight = float(
            cfg_get(cfg_get(loss_cfg, "in_rollout_history", {}) or {}, "weight", 0.1)
        )
        self.loss_short_window_history_weight = float(
            cfg_get(cfg_get(loss_cfg, "short_window_history", {}) or {}, "weight", 0.1)
        )
        self.loss_delta_reg_weight = float(cfg_get(cfg_get(loss_cfg, "delta_regularization", {}) or {}, "weight", 1.0))
        train_ifwd_cfg = cfg_get(cfg_get(config, "training", {}) or {}, "iforward", {}) or {}
        self.allow_missing_carried_state_reset = bool(
            cfg_get(train_ifwd_cfg, "allow_missing_carried_state_reset", False)
        )
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
            return False
        runtime = getattr(self, "phase_a_runtime", None)
        loader = getattr(runtime, "_load_phase_b_export_payload", None)
        if not callable(loader):
            return False
        target_device = device if device is not None else self.device
        loader(ckpt, device=target_device)
        return True

    def init_iforward_state_from_batch_assets(self, batch: Dict[str, Any], resolved: IForwardResolvedBatch) -> IForwardState:
        local_state, node_bg, node_distant, node_rigid = self.bridge.make_local_state(batch=batch)
        return IForwardState(
            local_gs=local_state,
            memory=IForwardMemoryState.empty(),
            history=IForwardShortWindowHistory.empty(
                max_entries=int(self.history_max_entries),
                max_memory_entries=int(self.history_max_memory_entries),
            ),
            scene_id=int(resolved.scene_id),
            segment_id=int(resolved.segment_id),
            episode_id=int(resolved.episode_id),
            node_state_bg=node_bg,
            node_state_distant=node_distant,
            node_state_rigid=node_rigid,
        )

    @staticmethod
    def _zero_loss(ref: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        return ref.new_tensor(0.0), {
            "num_refs": 0.0,
            "num_metric_refs": 0.0,
            "metric_valid": 0.0,
            "valid_ratio": 0.0,
            "skipped_no_valid_pixels": 0.0,
        }

    def _render_final_losses(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        resolved: IForwardResolvedBatch,
        carried_state: Optional[IForwardState],
        ablation: str,
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, float],
        List[torch.Tensor],
        List[torch.Tensor],
        List[Tuple[int, int]],
        List[str],
    ]:
        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        image_refs: List[Tuple[int, int]] = []
        image_roles: List[str] = []
        current_indices = list(resolved.current_latest_target_indices)
        before = len(pred_rgbs)
        current_loss, current_stats = self.bridge.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=current_indices,
            mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
            pred_rgbs_out=pred_rgbs,
            gt_images_out=gt_images,
        )
        appended = len(pred_rgbs) - before
        image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in current_indices[:appended]])
        image_roles.extend(["current_latest"] * int(appended))
        nearby_indices = list(resolved.nearby_target_indices)
        before = len(pred_rgbs)
        nearby_loss, nearby_stats = self.bridge.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=nearby_indices,
            mask_policy=str(getattr(self.bridge, "nearby_mask_policy", "non_sky_non_egocar")),
            pred_rgbs_out=pred_rgbs,
            gt_images_out=gt_images,
        )
        appended = len(pred_rgbs) - before
        image_refs.extend([tuple(int(x) for x in resolved.target_refs[int(i)]) for i in nearby_indices[:appended]])
        image_roles.extend(["nearby"] * int(appended))
        if self.loss_in_rollout_history_weight > 0.0 and len(resolved.history_rollout_target_indices) > 0:
            in_rollout_history_loss, in_rollout_stats = self.bridge.render_loss(
                local_state=local_state,
                batch=batch,
                target_indices=list(resolved.history_rollout_target_indices),
                mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
            )
        else:
            in_rollout_history_loss, in_rollout_stats = self._zero_loss(local_state.bg.means)

        short_history_loss, short_history_stats = self._zero_loss(local_state.bg.means)
        if (
            carried_state is not None
            and self.loss_short_window_history_weight > 0.0
            and carried_state.history.entries
        ):
            short_history_loss, short_history_stats = self.bridge.render_loss_for_targets(
                local_state=local_state,
                ref_batch=batch,
                targets=[dict(x) for x in carried_state.history.entries],
                mask_policy=str(getattr(self.bridge, "current_mask_policy", "non_sky_non_egocar")),
            )
        losses = {
            "current": current_loss,
            "nearby": nearby_loss,
            "in_rollout_history": in_rollout_history_loss,
            "short_window_history": short_history_loss,
        }
        stats = {
            "current_valid_ratio": float(current_stats.get("valid_ratio", 0.0)),
            "nearby_valid_ratio": float(nearby_stats.get("valid_ratio", 0.0)),
            "in_rollout_history_valid_ratio": float(in_rollout_stats.get("valid_ratio", 0.0)),
            "short_window_history_valid_ratio": float(short_history_stats.get("valid_ratio", 0.0)),
            "current_num_refs": float(current_stats.get("num_refs", len(resolved.current_latest_target_indices))),
            "history_rollout_num_refs": float(
                in_rollout_stats.get("num_refs", len(resolved.history_rollout_target_indices))
            ),
            "nearby_num_refs": float(nearby_stats.get("num_refs", len(resolved.nearby_target_indices))),
            "short_window_history_num_refs": float(short_history_stats.get("num_refs", 0.0)),
        }
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
        return losses, stats, pred_rgbs, gt_images, image_refs, image_roles

    def forward_rollout(
        self,
        batch: Dict[str, Any],
        *,
        carried_state: Optional[IForwardState] = None,
        ablation: Optional[str] = None,
    ) -> IForwardRolloutOutput:
        ablation_name = str(ablation or "full")
        if ablation_name not in self.allowed_ablations:
            raise ValueError(f"unsupported IForward ablation={ablation_name!r}")
        resolved = self.resolver.resolve(batch)
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
            if tuple(state.cache_key) != tuple(resolved.cache_key):
                raise ValueError(f"IForward carried state key {state.cache_key} does not match batch {resolved.cache_key}.")

        local_state = state.local_gs
        if hasattr(self.bridge, "sync_local_state_template_from_batch"):
            node_bg, node_distant, node_rigid = self.bridge.sync_local_state_template_from_batch(
                local_state=local_state,
                batch=batch,
            )
            state.node_state_bg = node_bg
            state.node_state_distant = node_distant
            state.node_state_rigid = node_rigid
        memory_state = state.memory
        working_history = state.history
        history_entries_before = int(len(working_history.entries))
        memory_entries_before = int(len(working_history.memory_entries))
        per_step: List[Dict[str, float]] = []
        reg_terms: List[torch.Tensor] = []
        reg_stats_sum: Dict[str, float] = {}
        timings: Dict[str, float] = {
            "observe_ms": 0.0,
            "event_ms": 0.0,
            "memory_ms": 0.0,
            "update_ms": 0.0,
            "delta_reg_ms": 0.0,
            "final_render_ms": 0.0,
        }

        global_step = int(batch.get("global_step", 0))
        for step_pos, step in enumerate(resolved.steps):
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/observe"):
                measurement = self.bridge.observe(
                    local_state=local_state,
                    batch=batch,
                    source_indices=list(step.source_indices),
                    source_frame_idx=int(step.source_frame_idx),
                )
            timings["observe_ms"] += (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/event"):
                event = self.bridge.build_event(local_state=local_state, measurement=measurement)
            timings["event_ms"] += (time.perf_counter() - t0) * 1000.0
            next_step = resolved.steps[step_pos + 1] if step_pos + 1 < len(resolved.steps) else None
            is_frame_exit = next_step is None or int(next_step.source_frame_idx) != int(step.source_frame_idx)
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
            )
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/memory"):
                memory_state, ctx_memory, memory_aux, short_entries = self.memory(
                    event=event,
                    local_state=local_state,
                    state=memory_state,
                    short_history=working_history,
                    step_context=step_context,
                    commit_observation_memory=bool(step.commit_observation_memory),
                    update_optimizer_memory=bool(step.update_optimizer_memory),
                    ablation=ablation_name,
                )
            timings["memory_ms"] += (time.perf_counter() - t0) * 1000.0
            working_history = working_history.commit_memory_entries(
                short_entries,
                detach=bool(getattr(self.memory, "short_entry_detach", True)),
            )
            t0 = time.perf_counter()
            with self._nvtx_range("iforward/update"):
                local_state, delta, update_aux = self.bridge.apply_update(
                    local_state=local_state,
                    event=event,
                    ctx_memory=ctx_memory,
                )
            timings["update_ms"] += (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            reg_loss, reg_stats = self.bridge.delta_regularization(delta, local_state=local_state)
            timings["delta_reg_ms"] += (time.perf_counter() - t0) * 1000.0
            reg_terms.append(reg_loss)
            for key, value in reg_stats.items():
                if isinstance(value, (int, float)):
                    reg_stats_sum[key] = reg_stats_sum.get(key, 0.0) + float(value)
            item: Dict[str, float] = {
                "k": float(step.step_idx),
                "source_frame_idx": float(step.source_frame_idx),
                "repeat_idx": float(step.repeat_idx),
                "num_source_indices": float(len(step.source_indices)),
                "commit_observation_memory": float(1.0 if step.commit_observation_memory else 0.0),
                "update_optimizer_memory": float(1.0 if step.update_optimizer_memory else 0.0),
                "is_frame_exit": float(1.0 if is_frame_exit else 0.0),
                "short_entries_added": float(len(short_entries)),
                "memory_entries_after_step": float(len(working_history.memory_entries)),
            }
            item.update({str(k): float(v) for k, v in memory_aux.items() if isinstance(v, (int, float))})
            item.update({str(k): float(v) for k, v in update_aux.items() if isinstance(v, (int, float))})
            per_step.append(item)

        t0 = time.perf_counter()
        with self._nvtx_range("iforward/final_render"):
            final_losses, final_stats, pred_rgbs, gt_images, image_refs, image_roles = self._render_final_losses(
                local_state=local_state,
                batch=batch,
                resolved=resolved,
                carried_state=prior_state_for_history,
                ablation=ablation_name,
            )
        timings["final_render_ms"] += (time.perf_counter() - t0) * 1000.0
        if reg_terms:
            delta_reg = torch.stack(reg_terms).mean()
        else:
            delta_reg = local_state.bg.means.new_tensor(0.0)
        final_losses["delta_regularization"] = delta_reg

        total = (
            self.loss_current_weight * final_losses["current"]
            + self.loss_nearby_weight * final_losses["nearby"]
            + self.loss_in_rollout_history_weight * final_losses["in_rollout_history"]
            + self.loss_short_window_history_weight * final_losses["short_window_history"]
            + self.loss_delta_reg_weight * final_losses["delta_regularization"]
        )
        if not torch.isfinite(total).all():
            raise RuntimeError("IForward rollout loss became NaN/Inf.")

        history = working_history.commit_targets(batch, tuple(resolved.current_target_indices))
        history_entries_after = int(len(history.entries))
        memory_entries_after = int(len(history.memory_entries))
        next_state = IForwardState(
            local_gs=local_state,
            memory=memory_state,
            history=history,
            scene_id=int(resolved.scene_id),
            segment_id=int(resolved.segment_id),
            episode_id=int(resolved.episode_id),
            node_state_bg=state.node_state_bg,
            node_state_distant=state.node_state_distant,
            node_state_rigid=state.node_state_rigid,
        )
        stats: Dict[str, Any] = {
            **final_stats,
            "inner_K": int(resolved.inner_K),
            "ablation": ablation_name,
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
            "loss_weight/current": float(self.loss_current_weight),
            "loss_weight/nearby": float(self.loss_nearby_weight),
            "loss_weight/in_rollout_history": float(self.loss_in_rollout_history_weight),
            "loss_weight/short_window_history": float(self.loss_short_window_history_weight),
            "loss_weight/delta_regularization": float(self.loss_delta_reg_weight),
            "memory_tokens": memory_state.count_tokens(),
        }
        current_psnr = stats.get("current_psnr")
        history_psnr = stats.get("history_rollout_psnr")
        short_psnr = stats.get("short_window_history_psnr")
        if current_psnr is not None and history_psnr is not None:
            stats["psnr_gap/current_minus_rollout_history"] = float(current_psnr) - float(history_psnr)
        if current_psnr is not None and short_psnr is not None:
            stats["psnr_gap/current_minus_short_history"] = float(current_psnr) - float(short_psnr)
        stats.update({key: float(value) for key, value in timings.items()})
        for key, value in reg_stats_sum.items():
            stats[f"delta_reg/{key}_mean"] = float(value) / float(max(len(reg_terms), 1))
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
