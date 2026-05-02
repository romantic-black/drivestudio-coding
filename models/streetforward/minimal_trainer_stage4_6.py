"""
Minimal StreetForward Stage 4.6:
- remove rigid-specific decoder/head branch
- keep rigid as dynamic node in local coordinates
- route source rigid points in source-frame world space:
    inside segment_aabb -> bg heads
    outside segment_aabb -> distant heads
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.math_utils import _normalize_quat, _num_sh_bases, _quat_multiply, _quat_to_rotmat, _sh_to_rgb
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    _backward_to_render_params_bg_rigid_distant,
    _merge_params_bg_rigid_distant,
    merge_debug_stats_as_perf_floats,
    spatial_hw_from_image_tensor,
)
from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid

logger = logging.getLogger(__name__)


@dataclass
class RigidRoute:
    S: torch.Tensor
    S_in: torch.Tensor
    S_out: torch.Tensor
    inside_mask_S: torch.Tensor
    route_inside_global: torch.Tensor
    means_world_S: torch.Tensor
    quats_world_S: torch.Tensor


@dataclass
class BgRigidInGRUInputs:
    feat_bg_input: torch.Tensor
    feat_rigid_in_input_all: Optional[torch.Tensor]
    aux: Dict[str, Any]


class MinimalStreetForwardStage4_5BaseNoRigidHead(MinimalStreetForwardStage4_5):
    """
    Compatibility base:
    - keeps Stage4.5 no-sky + fused source 2D + loss semantics
    - allows Stage4.6 rigid config without rigid.mlp/limits/freeze fields
    - removes rigid-specific trainable heads after parent init
    """

    def __init__(self, config, device: torch.device, **kwargs):
        self._stage4_6_orig_config = config
        compat_config = self._make_stage4_6_compat_config(config)
        super().__init__(compat_config, device, **kwargs)
        self._drop_rigid_specific_modules()
        self._init_stage4_6_rigid_cfg_from_original(self._stage4_6_orig_config)
        # Keep runtime config semantics aligned with Stage4_6 yaml after compat init.
        self.config = self._stage4_6_orig_config
        self.bg_freeze_quat = bool(self.bg_cfg["mlp"]["freeze_quat"])
        if hasattr(self, "mlp_offset_pos_rigid"):
            raise RuntimeError("Stage4_6 must not create rigid-specific decoder heads.")

    def _make_stage4_6_compat_config(self, config):
        cfg = copy.deepcopy(config)
        model_cfg = self._require_key(cfg, "model", "config")
        branches = self._require_key(model_cfg, "branches", "model")
        rigid_yaml = self._require_key(branches, "rigid", "model.branches")
        bg_yaml = self._require_key(branches, "bg", "model.branches")
        distant_yaml = self._require_key(branches, "distant", "model.branches")

        def _ensure_branch_mlp_defaults(branch_yaml, *, use_3d_feat: bool, freeze_quat: bool) -> None:
            mlp = branch_yaml.get("mlp")
            if mlp is None:
                branch_yaml["mlp"] = {
                    "hidden_dim": 64,
                    "use_3d_feat": bool(use_3d_feat),
                    "use_2d_feat": True,
                    "freeze_quat": bool(freeze_quat),
                }
                return
            if mlp.get("hidden_dim") is None:
                mlp["hidden_dim"] = 64
            if mlp.get("use_3d_feat") is None:
                mlp["use_3d_feat"] = bool(use_3d_feat)
            if mlp.get("use_2d_feat") is None:
                mlp["use_2d_feat"] = True
            if mlp.get("freeze_quat") is None:
                mlp["freeze_quat"] = bool(freeze_quat)

        _ensure_branch_mlp_defaults(bg_yaml, use_3d_feat=True, freeze_quat=False)
        _ensure_branch_mlp_defaults(distant_yaml, use_3d_feat=False, freeze_quat=True)

        if rigid_yaml.get("mlp") is None:
            rigid_yaml["mlp"] = {
                "hidden_dim": 64,
                "use_3d_feat": False,
                "use_2d_feat": True,
                "freeze_quat": False,
            }
        if rigid_yaml.get("limits") is None:
            rigid_yaml["limits"] = copy.deepcopy(self._require_key(bg_yaml, "limits", "model.branches.bg"))
        if rigid_yaml.get("freeze_means") is None:
            rigid_yaml["freeze_means"] = False
        return cfg

    def _init_stage4_6_rigid_cfg_from_original(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        branches = self._require_key(model_cfg, "branches", "model")
        rigid_yaml = self._require_key(branches, "rigid", "model.branches")
        init_cfg = self._require_key(rigid_yaml, "init", "model.branches.rigid")
        scale_init = self._require_key(init_cfg, "scale_init", "model.branches.rigid.init")
        eta = self._require_key(rigid_yaml, "eta", "model.branches.rigid")
        knn_k_primary = int(self._require_key(scale_init, "knn_k", "model.branches.rigid.init.scale_init"))
        self.rigid_cfg = {
            "init": {
                "scale_init_mode": str(self._require_key(scale_init, "mode", "model.branches.rigid.init.scale_init")),
                "isotropic_log_value": float(
                    self._require_key(scale_init, "isotropic_log_value", "model.branches.rigid.init.scale_init")
                ),
                "knn_k": int(knn_k_primary),
                "knn_log_scale_bias": float(
                    self._require_key(scale_init, "knn_log_scale_bias", "model.branches.rigid.init.scale_init")
                ),
                "opacity_init": float(self._require_key(init_cfg, "opacity_init", "model.branches.rigid.init")),
            },
            "eta": {
                "means": float(self._require_key(eta, "means", "model.branches.rigid.eta")),
                "scales": float(self._require_key(eta, "scales", "model.branches.rigid.eta")),
                "opacity": float(self._require_key(eta, "opacity", "model.branches.rigid.eta")),
                "sh_dc": float(self._require_key(eta, "sh_dc", "model.branches.rigid.eta")),
                "sh_rest": float(self._require_key(eta, "sh_rest", "model.branches.rigid.eta")),
            },
        }
        self.rigid_src_backproject_support_min = float(
            self._require_key(rigid_yaml, "src_backproject_support_min", "model.branches.rigid")
        )

    def _drop_rigid_specific_modules(self) -> None:
        for attr in (
            "rigid_feat_proj",
            "mlp_offset_pos_rigid",
            "mlp_conv_rigid",
            "mlp_opacity_rigid",
            "gaussion_decoder_rigid",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(self.config.optimizer.get("lr")),
            eps=float(self.config.optimizer.get("eps")),
            weight_decay=float(self.config.optimizer.get("weight_decay")),
        )


class MinimalStreetForwardStage4_6(MinimalStreetForwardStage4_5BaseNoRigidHead):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_stage4_6_config(config)
        super().__init__(config, device, **kwargs)
        self.rigid_routed_cfg = self._parse_rigid_routed_cfg(config)
        self._debug_check_rigid_roundtrip = bool(config.get("debug", {}).get("rigid_roundtrip_check", False))
        self._warned_source_mask_legacy_keys = False
        self._target_view_weight_cfg = self._parse_target_view_weight_cfg(config)

    def _validate_stage4_6_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if model_cfg.get("sky") is not None:
            raise ValueError("Stage4_6 removes model.sky.")
        branches = self._require_key(model_cfg, "branches", "model")
        if branches.get("sky") is not None:
            raise ValueError("Stage4_6 removes branches.sky.")
        rigid_yaml = self._require_key(branches, "rigid", "model.branches")
        for forbidden in ("mlp", "limits", "freeze_means", "freeze_quat"):
            if rigid_yaml.get(forbidden) is not None:
                raise ValueError(f"Stage4_6 removes rigid.{forbidden}.")

    def _parse_rigid_routed_cfg(self, config) -> Dict[str, Any]:
        model_cfg = self._require_key(config, "model", "config")
        routed_cfg = self._require_key(model_cfg, "rigid_routed", "model")
        route_space = str(self._require_key(routed_cfg, "route_space", "model.rigid_routed"))
        route_aabb = str(self._require_key(routed_cfg, "route_aabb", "model.rigid_routed"))
        inside_decoder = str(self._require_key(routed_cfg, "inside_decoder", "model.rigid_routed"))
        outside_decoder = str(self._require_key(routed_cfg, "outside_decoder", "model.rigid_routed"))
        update_means = bool(self._require_key(routed_cfg, "update_means", "model.rigid_routed"))
        update_quat = bool(self._require_key(routed_cfg, "update_quat", "model.rigid_routed"))
        if inside_decoder != "bg":
            raise ValueError("Stage4_6 rigid_routed.inside_decoder must be 'bg'.")
        if outside_decoder != "distant":
            raise ValueError("Stage4_6 rigid_routed.outside_decoder must be 'distant'.")
        if route_space != "source_frame_world":
            raise ValueError("Stage4_6 rigid_routed.route_space must be 'source_frame_world'.")
        if route_aabb != "segment_aabb":
            raise ValueError("Stage4_6 rigid_routed.route_aabb must be 'segment_aabb'.")
        return {
            "route_space": route_space,
            "route_aabb": route_aabb,
            "inside_decoder": inside_decoder,
            "outside_decoder": outside_decoder,
            "update_means": update_means,
            "update_quat": update_quat,
        }

    def _parse_target_view_weight_cfg(self, config) -> Dict[str, Any]:
        losses_cfg = config.get("losses", {}) if hasattr(config, "get") else {}
        tvw_cfg = losses_cfg.get("target_view_weights", {}) if hasattr(losses_cfg, "get") else {}
        source_cfg = tvw_cfg.get("source", {}) if hasattr(tvw_cfg, "get") else {}
        visited_cfg = tvw_cfg.get("visited", {}) if hasattr(tvw_cfg, "get") else {}
        near_cfg = tvw_cfg.get("near_random", {}) if hasattr(tvw_cfg, "get") else {}
        sched_cfg = near_cfg.get("schedule", {}) if hasattr(near_cfg, "get") else {}
        return {
            "enable": bool(tvw_cfg.get("enable", False)),
            "normalize_by_weight_sum": bool(tvw_cfg.get("normalize_by_weight_sum", True)),
            "source_weight": float(source_cfg.get("weight", 1.0)),
            "visited_weight": float(visited_cfg.get("weight", 1.0)),
            "near_random_weight": float(near_cfg.get("weight", 1.0)),
            "near_random_schedule_enable": bool(sched_cfg.get("enable", False)),
            "near_random_schedule_type": str(sched_cfg.get("type", "warmup_linear")),
            "near_random_start_weight": float(sched_cfg.get("start_weight", near_cfg.get("weight", 1.0))),
            "near_random_end_weight": float(sched_cfg.get("end_weight", near_cfg.get("weight", 1.0))),
            "near_random_warmup_steps": int(sched_cfg.get("warmup_steps", 0)),
        }

    def _current_loss_step(self, batch: Dict[str, Any]) -> int:
        opt = getattr(self, "optimizer", None)
        if opt is not None and hasattr(opt, "global_step"):
            return int(getattr(opt, "global_step"))
        if hasattr(self, "global_step"):
            return int(getattr(self, "global_step"))
        aligned = batch.get("_scheduler_v9_aligned_info") or batch.get("_scheduler_v8_aligned_info") or {}
        return int(aligned.get("global_step", 0))

    @staticmethod
    def _warmup_linear_value(step: int, *, start_value: float, end_value: float, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return float(end_value)
        t = min(1.0, max(0.0, float(step) / float(warmup_steps)))
        return float(start_value) + t * (float(end_value) - float(start_value))

    def _near_random_loss_weight(self, step: int) -> float:
        cfg = self._target_view_weight_cfg
        base = float(cfg["near_random_weight"])
        if not bool(cfg["near_random_schedule_enable"]):
            return base
        sched_type = str(cfg["near_random_schedule_type"])
        if sched_type != "warmup_linear":
            raise ValueError(f"unsupported near_random weight schedule type: {sched_type!r}")
        return self._warmup_linear_value(
            int(step),
            start_value=float(cfg["near_random_start_weight"]),
            end_value=float(cfg["near_random_end_weight"]),
            warmup_steps=int(cfg["near_random_warmup_steps"]),
        )

    def _target_role_weight(self, role: str, step: int) -> float:
        cfg = self._target_view_weight_cfg
        if role == "source":
            return float(cfg["source_weight"])
        if role == "visited":
            return float(cfg["visited_weight"])
        if role == "near_random":
            return float(self._near_random_loss_weight(int(step)))
        raise ValueError(f"unknown target role: {role}")

    def _build_target_view_weights(
        self,
        batch: Dict[str, Any],
        *,
        step: int,
        num_targets: int,
    ) -> Tuple[torch.Tensor, List[str]]:
        cfg = self._target_view_weight_cfg
        if not bool(cfg["enable"]):
            return torch.ones((int(num_targets),), dtype=torch.float32, device=self.device), ["source"] * int(num_targets)
        meta = batch.get("request_meta") or {}
        roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
        if len(roles) == 0:
            return torch.ones((int(num_targets),), dtype=torch.float32, device=self.device), ["source"] * int(num_targets)
        if len(roles) != int(num_targets):
            raise ValueError(f"target_image_roles length mismatch with targets: {len(roles)} vs {int(num_targets)}")
        vals = [float(self._target_role_weight(role, int(step))) for role in roles]
        return torch.tensor(vals, dtype=torch.float32, device=self.device), roles

    def _get_source_masks_from_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        source_sky_masks = batch.get("source_sky_masks")
        source_egocar_masks = batch.get("source_egocar_masks")
        used_legacy = False
        if source_sky_masks is None and "source_sky_mask" in batch:
            source_sky_masks = batch.get("source_sky_mask")
            used_legacy = True
        if source_egocar_masks is None and "source_egocar_mask" in batch:
            source_egocar_masks = batch.get("source_egocar_mask")
            used_legacy = True
        if used_legacy and not self._warned_source_mask_legacy_keys:
            logger.warning(
                "Stage4_6 received legacy source mask keys source_sky_mask/source_egocar_mask; "
                "prefer source_sky_masks/source_egocar_masks."
            )
            self._warned_source_mask_legacy_keys = True
        return source_sky_masks, source_egocar_masks

    def _route_rigid_source_points(
        self,
        node_state_rigid: NodeStateRigid,
        source_frame_idx: int,
        S: torch.Tensor,
    ) -> RigidRoute:
        N_rigid = int(node_state_rigid.means.shape[0])
        if S.numel() == 0:
            return RigidRoute(
                S=S,
                S_in=S,
                S_out=S,
                inside_mask_S=torch.zeros((0,), dtype=torch.bool, device=self.device),
                route_inside_global=torch.zeros((N_rigid,), dtype=torch.bool, device=self.device),
                means_world_S=torch.zeros((0, 3), dtype=node_state_rigid.means.dtype, device=self.device),
                quats_world_S=torch.zeros((0, 4), dtype=node_state_rigid.quats.dtype, device=self.device),
            )
        point_ids_S = node_state_rigid.point_ids[S, 0]
        means_world_S = self._transform_rigid_to_world(
            node_state_rigid,
            node_state_rigid.means[S],
            source_frame_idx,
            point_ids_subset=point_ids_S,
        )
        quats_world_S = self._transform_rigid_quats_to_world(
            node_state_rigid,
            node_state_rigid.quats[S],
            source_frame_idx,
            point_ids_subset=point_ids_S,
        )
        inside_mask_S = ((means_world_S >= self.bbx_min) & (means_world_S <= self.bbx_max)).all(dim=-1)
        S_in = S[inside_mask_S]
        S_out = S[~inside_mask_S]
        route_inside_global = torch.zeros((N_rigid,), dtype=torch.bool, device=self.device)
        route_inside_global[S_in] = True
        if S_in.numel() + S_out.numel() != S.numel():
            raise RuntimeError("Rigid route split mismatch.")
        return RigidRoute(
            S=S,
            S_in=S_in,
            S_out=S_out,
            inside_mask_S=inside_mask_S,
            route_inside_global=route_inside_global,
            means_world_S=means_world_S,
            quats_world_S=quats_world_S,
        )

    def _compute_2d_features_all_branches_once_routed(
        self,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        source_views: List[Any],
        source_images: List[torch.Tensor],
        source_sky_masks: Optional[List[torch.Tensor]],
        source_egocar_masks: Optional[List[torch.Tensor]],
        height: int,
        width: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        gaussians_bg_distant, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        num_rigid_S = int(route.S.numel())
        parts_means = [gaussians_bg_distant["means"]]
        parts_scales = [gaussians_bg_distant["scales"]]
        parts_quats = [gaussians_bg_distant["quats"]]
        parts_opacities = [gaussians_bg_distant["opacities"]]
        parts_colors = [gaussians_bg_distant["colors"]]
        if node_state_rigid is not None and num_rigid_S > 0:
            parts_means.append(route.means_world_S)
            parts_quats.append(route.quats_world_S)
            parts_scales.append(torch.exp(node_state_rigid.scales_log[route.S]))
            parts_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[route.S]).squeeze(-1))
            parts_colors.append(
                torch.cat(
                    [
                        node_state_rigid.sh_dc[route.S, None, :],
                        node_state_rigid.sh_rest[route.S],
                    ],
                    dim=1,
                )
            )
        gaussians_scene = {
            "means": torch.cat(parts_means, dim=0),
            "scales": torch.cat(parts_scales, dim=0),
            "quats": torch.cat(parts_quats, dim=0),
            "opacities": torch.cat(parts_opacities, dim=0),
            "colors": torch.cat(parts_colors, dim=0),
        }
        cnn_inputs = self._render_source_scene_only_for_cnn(
            gaussians_scene=gaussians_scene,
            source_views=source_views,
            source_images=source_images,
            source_sky_masks=source_sky_masks,
            source_egocar_masks=source_egocar_masks,
            height=height,
            width=width,
        )
        feat_2d_all, acc_w_all = self._backproject_scene_features_multi_camera(
            gaussians_scene=gaussians_scene,
            source_views=source_views,
            features_2d=cnn_inputs["features_2d"],
            source_pair_valid_mask=cnn_inputs["source_pair_valid_mask"],
            height=height,
            width=width,
        )
        if feat_2d_all is None or acc_w_all is None:
            raise ValueError("Stage4_6 one-pass routed source backprojection returned None.")
        start = 0
        feat_2d_bg = feat_2d_all[start : start + num_bg]
        acc_w_bg = acc_w_all[start : start + num_bg]
        start += num_bg
        feat_2d_distant = feat_2d_all[start : start + num_distant] if num_distant > 0 else None
        acc_w_distant = acc_w_all[start : start + num_distant] if num_distant > 0 else None
        start += num_distant
        feat_2d_rigid_S = feat_2d_all[start : start + num_rigid_S] if num_rigid_S > 0 else None
        acc_w_rigid_S = acc_w_all[start : start + num_rigid_S] if num_rigid_S > 0 else None
        return {
            "num_bg": num_bg,
            "num_distant": num_distant,
            "feat_2d_bg": feat_2d_bg,
            "feat_2d_distant": feat_2d_distant,
            "feat_2d_rigid_S": feat_2d_rigid_S,
            "acc_w_bg": acc_w_bg,
            "acc_w_distant": acc_w_distant,
            "acc_w_rigid_S": acc_w_rigid_S,
            "src_backproject_pass_count": 1,
        }

    def _build_3d_features_bg_plus_rigid_in(
        self,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        means_bg = node_state_bg.means
        rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        means_parts = [means_bg]
        rgb_parts = [rgb_bg]
        if node_state_rigid is not None and route.S_in.numel() > 0:
            means_parts.append(route.means_world_S[route.inside_mask_S])
            rgb_parts.append(_sh_to_rgb(node_state_rigid.sh_dc[route.S_in]))
        means_all = torch.cat(means_parts, dim=0)
        rgb_all = torch.cat(rgb_parts, dim=0)
        feat_all = self._build_3d_features(means_all, rgb_all)
        N_bg = int(means_bg.shape[0])
        feat_3d_bg = feat_all[:N_bg]
        feat_3d_rigid_in = feat_all[N_bg:] if route.S_in.numel() > 0 else None
        return feat_3d_bg, feat_3d_rigid_in

    def _compute_bg_rigid_in_gru_inputs(
        self,
        *,
        batch: Optional[Dict[str, Any]] = None,
        source_frame_idx: int,
        node_state_bg: NodeStateBackground,
        node_state_rigid: Optional[NodeStateRigid],
        route: RigidRoute,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid_S: Optional[torch.Tensor],
        acc_w_bg: torch.Tensor,
        acc_w_rigid_S: Optional[torch.Tensor],
        node_state_distant: Optional[NodeStateDistant] = None,
        feat_2d_distant: Optional[torch.Tensor] = None,
        acc_w_distant: Optional[torch.Tensor] = None,
    ) -> BgRigidInGRUInputs:
        _ = batch
        _ = source_frame_idx
        _ = node_state_distant
        _ = feat_2d_distant
        _ = acc_w_distant
        _ = acc_w_rigid_S
        feat_3d_bg, feat_3d_rigid_in = self._build_3d_features_bg_plus_rigid_in(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            route=route,
        )
        feat_bg_input = self._fuse_features(
            feat_3d_bg,
            feat_2d_bg,
            visibility=(acc_w_bg > self.bg_src_backproject_support_min),
        )
        feat_rigid_in_input_all = None
        if route.S_in.numel() > 0:
            if feat_2d_rigid_S is None or feat_3d_rigid_in is None:
                raise RuntimeError("Stage4_6 expected rigid source features for S_in path.")
            rows_rigid_in_in_S = torch.nonzero(route.inside_mask_S, as_tuple=False).squeeze(1)
            feat_2d_rigid_in = feat_2d_rigid_S[rows_rigid_in_in_S]
            feat_rigid_in_input_all = self._fuse_features(
                feat_3d_rigid_in,
                feat_2d_rigid_in,
                visibility=torch.ones(route.S_in.numel(), dtype=torch.bool, device=self.device),
            )
        return BgRigidInGRUInputs(
            feat_bg_input=feat_bg_input,
            feat_rigid_in_input_all=feat_rigid_in_input_all,
            aux={},
        )

    def _predict_offsets_gru_with_heads(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        *,
        mask_update: Optional[torch.Tensor],
        limits: Dict[str, float],
        mlp_offset_pos,
        mlp_conv,
        mlp_opacity,
        gaussion_decoder,
        freeze_quat: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        num_points = int(params_for_embed["means"].shape[0])
        device = params_for_embed["means"].device
        dtype = params_for_embed["means"].dtype
        if feat is None or feat.numel() == 0 or num_points == 0:
            num_sh = _num_sh_bases(self.sh_degree)
            offsets = {
                "offset_pos": torch.zeros((num_points, 3), device=device, dtype=dtype),
                "offset_scales": torch.zeros((num_points, 3), device=device, dtype=dtype),
                "offset_quat": self._identity_quat(num_points, device, dtype),
                "offset_opacity": torch.zeros((num_points, 1), device=device, dtype=dtype),
                "offset_sh": torch.zeros((num_points, 3 * num_sh), device=device, dtype=dtype),
            }
            h_new = h_old
        else:
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
            head_input = self._apply_gru_head_rms(head_input, mask_update)
            offsets = self._predict_offsets_with_heads(
                head_input,
                limits=limits,
                mlp_offset_pos=mlp_offset_pos,
                mlp_conv=mlp_conv,
                mlp_opacity=mlp_opacity,
                gaussion_decoder=gaussion_decoder,
                freeze_quat=freeze_quat,
            )
        if mask_update is not None:
            gate = mask_update.to(dtype=dtype, device=device).unsqueeze(-1).detach()
            identity = self._identity_quat(num_points, device, dtype)
            offsets["offset_pos"] = offsets["offset_pos"] * gate
            offsets["offset_scales"] = offsets["offset_scales"] * gate
            offsets["offset_opacity"] = offsets["offset_opacity"] * gate
            offsets["offset_sh"] = offsets["offset_sh"] * gate
            offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
            h_new = h_old * (1.0 - gate) + h_new * gate
        return offsets, h_new

    def _build_rigid_params_for_embed_source_world(
        self,
        node_state_rigid: NodeStateRigid,
        source_frame_idx: int,
        U: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        class _RigidEmbedState:
            pass

        rigid_embed_state = _RigidEmbedState()
        point_ids = node_state_rigid.point_ids[U, 0]
        rigid_embed_state.means = self._transform_rigid_to_world(
            node_state_rigid,
            node_state_rigid.means[U],
            source_frame_idx,
            point_ids_subset=point_ids,
        )
        rigid_embed_state.quats = self._transform_rigid_quats_to_world(
            node_state_rigid,
            node_state_rigid.quats[U],
            source_frame_idx,
            point_ids_subset=point_ids,
        )
        rigid_embed_state.scales_log = node_state_rigid.scales_log[U]
        rigid_embed_state.opacity_logit = node_state_rigid.opacity_logit[U]
        rigid_embed_state.sh_dc = node_state_rigid.sh_dc[U]
        rigid_embed_state.sh_rest = node_state_rigid.sh_rest[U]
        return self._build_params_for_embed(rigid_embed_state, coord_space="world")

    def _transform_rigid_points_world_to_local(
        self,
        node_state_rigid: NodeStateRigid,
        means_world: torch.Tensor,
        frame_idx: int,
        point_ids_subset: torch.Tensor,
    ) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved is None:
            raise ValueError(f"Rigid frame_idx={frame_idx} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}")
        quats_cur = node_state_rigid.instances_quats[resolved]
        trans_cur = node_state_rigid.instances_trans[resolved]
        rot_cur = _quat_to_rotmat(quats_cur)
        rot_pts = rot_cur[point_ids_subset.long()]
        trans_pts = trans_cur[point_ids_subset.long()]
        world_centered = means_world - trans_pts
        return torch.bmm(rot_pts.transpose(1, 2), world_centered.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
        q = quat.clone()
        q[..., 1:] = -q[..., 1:]
        return q

    def _transform_rigid_quats_world_to_local(
        self,
        node_state_rigid: NodeStateRigid,
        quats_world: torch.Tensor,
        frame_idx: int,
        point_ids_subset: torch.Tensor,
    ) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved is None:
            raise ValueError(f"Rigid frame_idx={frame_idx} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}")
        quats_cur = node_state_rigid.instances_quats[resolved]
        quats_pts = quats_cur[point_ids_subset.long()]
        quats_pts_inv = self._quat_conjugate(_normalize_quat(quats_pts))
        return _normalize_quat(_quat_multiply(quats_pts_inv, quats_world))

    def _render_params_from_routed_offsets_rigid_local(
        self,
        node_state_rigid: NodeStateRigid,
        source_frame_idx: int,
        U: torch.Tensor,
        offsets_world: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if U.numel() == 0:
            raise ValueError("Stage4_6 expects non-empty U in _render_params_from_routed_offsets_rigid_local.")
        point_ids = node_state_rigid.point_ids[U, 0]
        means_local_old = node_state_rigid.means[U]
        means_world_old = self._transform_rigid_to_world(
            node_state_rigid,
            means_local_old,
            source_frame_idx,
            point_ids_subset=point_ids,
        )
        means_world_new = means_world_old
        if self.rigid_routed_cfg["update_means"]:
            means_world_new = means_world_new + self.rigid_cfg["eta"]["means"] * offsets_world["offset_pos"]
        means_local_new = self._transform_rigid_points_world_to_local(
            node_state_rigid,
            means_world_new,
            source_frame_idx,
            point_ids_subset=point_ids,
        )

        quats_local_old = node_state_rigid.quats[U]
        quats_world_old = self._transform_rigid_quats_to_world(
            node_state_rigid,
            quats_local_old,
            source_frame_idx,
            point_ids_subset=point_ids,
        )
        if self.rigid_routed_cfg["update_quat"]:
            quats_world_new = _normalize_quat(_quat_multiply(quats_world_old, offsets_world["offset_quat"]))
        else:
            quats_world_new = quats_world_old
        quats_local_new = self._transform_rigid_quats_world_to_local(
            node_state_rigid,
            quats_world_new,
            source_frame_idx,
            point_ids_subset=point_ids,
        )

        num_sh = _num_sh_bases(self.sh_degree)
        scales_log_r = node_state_rigid.scales_log[U] + self.rigid_cfg["eta"]["scales"] * offsets_world["offset_scales"]
        opacity_logit_r = node_state_rigid.opacity_logit[U] + self.rigid_cfg["eta"]["opacity"] * offsets_world["offset_opacity"]
        sh_dc_r = node_state_rigid.sh_dc[U] + self.rigid_cfg["eta"]["sh_dc"] * offsets_world["offset_sh"][:, :3]
        sh_rest_offset = offsets_world["offset_sh"][:, 3:].view(U.numel(), num_sh - 1, 3)
        sh_rest_r = node_state_rigid.sh_rest[U] + self.rigid_cfg["eta"]["sh_rest"] * sh_rest_offset
        out = {
            "means_r": means_local_new,
            "scales_log_r": scales_log_r,
            "quats_r": quats_local_new,
            "opacity_logit_r": opacity_logit_r,
            "sh_dc_r": sh_dc_r,
            "sh_rest_r": sh_rest_r,
            "scales_r": torch.exp(scales_log_r),
            "opacities_r": torch.sigmoid(opacity_logit_r).squeeze(-1),
            "colors_r": torch.cat([sh_dc_r[:, None, :], sh_rest_r], dim=1),
        }
        if self._debug_check_rigid_roundtrip:
            means_world_recovered = self._transform_rigid_to_world(
                node_state_rigid,
                out["means_r"],
                source_frame_idx,
                point_ids_subset=point_ids,
            )
            max_err = (means_world_recovered - means_world_new).abs().max()
            if float(max_err.item()) > 1e-4:
                raise RuntimeError("Rigid world->local->world position roundtrip failed.")
        return out

    def _update_node_state_rigid_local_subset(
        self,
        node_state_rigid: NodeStateRigid,
        render_params_rigid_local_U: Dict[str, torch.Tensor],
        U: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if U.numel() == 0:
                return
            node_state_rigid.means[U] = render_params_rigid_local_U["means_r"].detach().to(
                device=node_state_rigid.means.device, dtype=node_state_rigid.means.dtype
            )
            node_state_rigid.scales_log[U] = render_params_rigid_local_U["scales_log_r"].detach().to(
                device=node_state_rigid.scales_log.device, dtype=node_state_rigid.scales_log.dtype
            )
            node_state_rigid.quats[U] = render_params_rigid_local_U["quats_r"].detach().to(
                device=node_state_rigid.quats.device, dtype=node_state_rigid.quats.dtype
            )
            node_state_rigid.opacity_logit[U] = render_params_rigid_local_U["opacity_logit_r"].detach().to(
                device=node_state_rigid.opacity_logit.device, dtype=node_state_rigid.opacity_logit.dtype
            )
            node_state_rigid.sh_dc[U] = render_params_rigid_local_U["sh_dc_r"].detach().to(
                device=node_state_rigid.sh_dc.device, dtype=node_state_rigid.sh_dc.dtype
            )
            node_state_rigid.sh_rest[U] = render_params_rigid_local_U["sh_rest_r"].detach().to(
                device=node_state_rigid.sh_rest.device, dtype=node_state_rigid.sh_rest.dtype
            )

    @staticmethod
    def _pack_rigid_local_subsets(
        render_in: Optional[Dict[str, torch.Tensor]],
        render_out: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        keys = (
            "means_r",
            "scales_log_r",
            "quats_r",
            "opacity_logit_r",
            "sh_dc_r",
            "sh_rest_r",
            "scales_r",
            "opacities_r",
            "colors_r",
        )
        ref_chunk = render_in if render_in is not None else render_out
        if ref_chunk is None:
            raise RuntimeError("Stage4_6 internal error: both render_in and render_out are None in rigid subset pack.")

        def _empty_like_chunk(ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            return {
                k: ref[k].new_empty((0, *ref[k].shape[1:]))
                for k in keys
            }

        in_chunk = render_in if render_in is not None else _empty_like_chunk(ref_chunk)
        out_chunk = render_out if render_out is not None else _empty_like_chunk(ref_chunk)
        return {
            k: torch.cat([in_chunk[k], out_chunk[k]], dim=0)
            for k in keys
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

        if out.get("_node_state_rigid") is not None and out.get("_render_params_rigid_local") is not None:
            U = out.get("_rigid_writeback_idx")
            if U is None:
                raise ValueError("Stage4_6 internal error: missing _rigid_writeback_idx.")
            self._update_node_state_rigid_local_subset(
                out["_node_state_rigid"],
                out["_render_params_rigid_local"],
                U,
            )

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.6 requires non-empty batch['targets'].")
        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        source_frame_idx = self._validate_stage4_1_batch(batch, targets, node_state_rigid)
        key = self._batch_key(batch)

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        source_sky_masks, source_egocar_masks = self._get_source_masks_from_batch(batch)
        sample_img = source_images[0]
        height, width = spatial_hw_from_image_tensor(sample_img)

        N_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        unique_target_frames = sorted({int(t["frame_idx"]) for t in targets})
        mask_tgt_by_frame: Dict[int, torch.Tensor] = {}
        mask_src_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
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
        if node_state_rigid is None:
            route = RigidRoute(
                S=S,
                S_in=S,
                S_out=S,
                inside_mask_S=torch.zeros((0,), dtype=torch.bool, device=self.device),
                route_inside_global=torch.zeros((N_rigid,), dtype=torch.bool, device=self.device),
                means_world_S=torch.zeros((0, 3), device=self.device),
                quats_world_S=torch.zeros((0, 4), device=self.device),
            )
        else:
            route = self._route_rigid_source_points(node_state_rigid, source_frame_idx, S)

        one_pass = self._compute_2d_features_all_branches_once_routed(
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            source_views=source_views,
            source_images=source_images,
            source_sky_masks=source_sky_masks,
            source_egocar_masks=source_egocar_masks,
            height=height,
            width=width,
        )
        num_bg = int(one_pass["num_bg"])
        num_distant = int(one_pass["num_distant"])
        feat_2d_bg = one_pass["feat_2d_bg"]
        feat_2d_distant = one_pass["feat_2d_distant"]
        feat_2d_rigid_S = one_pass["feat_2d_rigid_S"]
        acc_w_bg = one_pass["acc_w_bg"]
        acc_w_distant = one_pass["acc_w_distant"]
        acc_w_rigid_S = one_pass["acc_w_rigid_S"]
        src_backproject_pass_count = int(one_pass.get("src_backproject_pass_count", 0))

        bg_rigid_in_inputs = self._compute_bg_rigid_in_gru_inputs(
            batch=batch,
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            feat_2d_bg=feat_2d_bg,
            feat_2d_distant=feat_2d_distant,
            feat_2d_rigid_S=feat_2d_rigid_S,
            acc_w_bg=acc_w_bg,
            acc_w_distant=acc_w_distant,
            acc_w_rigid_S=acc_w_rigid_S,
        )
        feat_bg_input = bg_rigid_in_inputs.feat_bg_input
        feat_rigid_in_input_all = bg_rigid_in_inputs.feat_rigid_in_input_all
        bg_rigid_in_aux = dict(bg_rigid_in_inputs.aux)
        stage5_2_full = getattr(self, "_stage5_2_last_full_inputs", None)
        apply_update_gate_fn = getattr(self, "_apply_update_gate", None)
        gate_bg = getattr(stage5_2_full, "gate_bg", None) if stage5_2_full is not None else None
        gate_distant = getattr(stage5_2_full, "gate_distant", None) if stage5_2_full is not None else None
        gate_rigid_in = getattr(stage5_2_full, "gate_rigid_in", None) if stage5_2_full is not None else None
        gate_rigid_out = getattr(stage5_2_full, "gate_rigid_out", None) if stage5_2_full is not None else None
        feat_distant_input_from_struct = (
            getattr(stage5_2_full, "feat_distant_input", None) if stage5_2_full is not None else None
        )
        feat_rigid_out_input_all_from_struct = (
            getattr(stage5_2_full, "feat_rigid_out_input_all", None) if stage5_2_full is not None else None
        )
        eff_gate_bg = None
        eff_gate_distant = None
        eff_gate_rigid_in = None
        eff_gate_rigid_out = None

        def _select_gate_rows(gate_obj, rows):
            if gate_obj is None:
                return None
            if hasattr(gate_obj, "select_rows"):
                return gate_obj.select_rows(rows)
            return gate_obj[rows]
        mask_src_feat_valid_bg = acc_w_bg > self.bg_src_backproject_support_min
        mask_any_tgt_bg = self._build_any_target_mask_static(
            num_points=num_bg,
            enable_selective=self.bg_enable_selective_update,
            device=self.device,
        )
        mask_update_bg = mask_src_feat_valid_bg & mask_any_tgt_bg
        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru_with_heads(
            feat_bg_input,
            params_bg,
            h_old_bg,
            mask_update=mask_update_bg,
            limits=self.bg_cfg["limits"],
            mlp_offset_pos=self.mlp_offset_pos,
            mlp_conv=self.mlp_conv,
            mlp_opacity=self.mlp_opacity,
            gaussion_decoder=self.gaussion_decoder,
            freeze_quat=self.bg_freeze_quat,
        )
        if callable(apply_update_gate_fn) and gate_bg is not None:
            offsets_bg, h_new_bg, eff_gate_bg = apply_update_gate_fn(
                offsets_bg,
                h_old=h_old_bg,
                h_candidate=h_new_bg,
                gate=gate_bg,
                mask_update=mask_update_bg,
            )
        bg_rigid_in_aux["stage5_3_applied_delta_bg_means_abs"] = float(offsets_bg["offset_pos"].abs().mean().detach().item())
        bg_rigid_in_aux["stage5_3_applied_delta_bg_opacity_abs"] = float(
            offsets_bg["offset_opacity"].abs().mean().detach().item()
        )
        bg_rigid_in_aux["stage5_3_applied_delta_bg_sh_abs"] = float(offsets_bg["offset_sh"].abs().mean().detach().item())
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        if num_distant > 0:
            mask_src_feat_valid_distant = acc_w_distant > self.distant_src_backproject_support_min
            mask_any_tgt_distant = self._build_any_target_mask_static(
                num_points=num_distant,
                enable_selective=self.distant_enable_selective_update,
                device=self.device,
            )
            mask_update_distant = mask_src_feat_valid_distant & mask_any_tgt_distant
        else:
            mask_src_feat_valid_distant = None
            mask_update_distant = None
        render_params_distant = None
        offsets_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_2d_distant is not None and feat_2d_distant.numel() > 0:
            feat_distant_input = (
                feat_distant_input_from_struct
                if feat_distant_input_from_struct is not None
                else self.distant_feat_proj(feat_2d_distant)
            )
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant"
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru_with_heads(
                feat_distant_input,
                params_distant,
                h_old_distant,
                mask_update=mask_update_distant,
                limits=self.distant_cfg["limits"],
                mlp_offset_pos=self.mlp_offset_pos_distant,
                mlp_conv=self.mlp_conv_distant,
                mlp_opacity=self.mlp_opacity_distant,
                gaussion_decoder=self.gaussion_decoder_distant,
                freeze_quat=self.distant_freeze_quat,
            )
            if callable(apply_update_gate_fn) and gate_distant is not None:
                offsets_distant, h_new_distant, eff_gate_distant = apply_update_gate_fn(
                    offsets_distant,
                    h_old=h_old_distant,
                    h_candidate=h_new_distant,
                    gate=gate_distant,
                    mask_update=mask_update_distant,
                )
            bg_rigid_in_aux["stage5_3_applied_delta_distant_means_abs"] = float(
                offsets_distant["offset_pos"].abs().mean().detach().item()
            )
            bg_rigid_in_aux["stage5_3_applied_delta_distant_opacity_abs"] = float(
                offsets_distant["offset_opacity"].abs().mean().detach().item()
            )
            bg_rigid_in_aux["stage5_3_applied_delta_distant_sh_abs"] = float(
                offsets_distant["offset_sh"].abs().mean().detach().item()
            )
            render_params_distant = self._render_params_from_offsets_distant(node_state_distant, offsets_distant)

        mask_src_feat_valid_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        if node_state_rigid is not None and route.S.numel() > 0:
            if acc_w_rigid_S is None:
                raise ValueError("Stage4_6 rigid S non-empty but acc_w_rigid_S is None.")
            mask_src_feat_valid_rigid[route.S] = acc_w_rigid_S > self.rigid_src_backproject_support_min
        mask_update_rigid = mask_src_feat_valid_rigid & mask_any_tgt_rigid
        U = torch.nonzero(mask_update_rigid, as_tuple=False).squeeze(1)
        U_in = U[route.route_inside_global[U]] if U.numel() > 0 else U
        U_out = U[~route.route_inside_global[U]] if U.numel() > 0 else U

        h_new_rigid_full = None
        render_params_rigid_local_U = None
        offsets_rigid_in_world = None
        offsets_rigid_out_world = None
        rigid_means_world_for_stats = None
        U_all = torch.zeros((0,), dtype=torch.long, device=self.device)
        rigid_in_acc_w_mean = 0.0
        rigid_out_acc_w_mean = 0.0
        if node_state_rigid is not None:
            h_old_rigid = self._get_or_init_hidden(
                self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid"
            )
            h_new_rigid_full = h_old_rigid.clone()
            lookup_S = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_S[route.S] = torch.arange(route.S.numel(), device=self.device, dtype=torch.long)
            lookup_S_in = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_S_in[route.S_in] = torch.arange(route.S_in.numel(), device=self.device, dtype=torch.long)
            lookup_S_out = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_S_out[route.S_out] = torch.arange(route.S_out.numel(), device=self.device, dtype=torch.long)

            render_in = None
            render_out = None
            h_new_rigid_in = None
            h_new_rigid_out = None
            if U_in.numel() > 0:
                rows_S = lookup_S[U_in]
                rows_S_in = lookup_S_in[U_in]
                if bool((rows_S < 0).any().item()):
                    raise RuntimeError("Routed rigid update point not present in source visible S.")
                if bool((rows_S_in < 0).any().item()):
                    raise RuntimeError("U_in contains rigid point not present in S_in.")
                if feat_rigid_in_input_all is None:
                    raise RuntimeError("Stage4_6 expected rigid source features for U_in path.")
                rigid_in_acc_w_mean = float(acc_w_rigid_S[rows_S].mean().item()) if acc_w_rigid_S is not None else 0.0
                feat_rigid_in_input = feat_rigid_in_input_all[rows_S_in]
                params_rigid_in_world = self._build_rigid_params_for_embed_source_world(node_state_rigid, source_frame_idx, U_in)
                offsets_rigid_in_world, h_new_rigid_in = self._predict_offsets_gru_with_heads(
                    feat_rigid_in_input,
                    params_rigid_in_world,
                    h_old_rigid[U_in],
                    mask_update=torch.ones(U_in.numel(), dtype=torch.bool, device=self.device),
                    limits=self.bg_cfg["limits"],
                    mlp_offset_pos=self.mlp_offset_pos,
                    mlp_conv=self.mlp_conv,
                    mlp_opacity=self.mlp_opacity,
                    gaussion_decoder=self.gaussion_decoder,
                    freeze_quat=self.bg_freeze_quat,
                )
                if callable(apply_update_gate_fn) and gate_rigid_in is not None:
                    gate_rigid_in_sel = _select_gate_rows(gate_rigid_in, rows_S_in)
                    offsets_rigid_in_world, h_new_rigid_in, eff_gate_rigid_in = apply_update_gate_fn(
                        offsets_rigid_in_world,
                        h_old=h_old_rigid[U_in],
                        h_candidate=h_new_rigid_in,
                        gate=gate_rigid_in_sel,
                        mask_update=torch.ones(U_in.numel(), dtype=torch.bool, device=self.device),
                    )
                render_in = self._render_params_from_routed_offsets_rigid_local(
                    node_state_rigid=node_state_rigid,
                    source_frame_idx=source_frame_idx,
                    U=U_in,
                    offsets_world=offsets_rigid_in_world,
                )
            if U_out.numel() > 0:
                rows_S = lookup_S[U_out]
                rows_S_out = lookup_S_out[U_out]
                if bool((rows_S < 0).any().item()):
                    raise RuntimeError("Routed rigid update point not present in source visible S.")
                if bool((rows_S_out < 0).any().item()):
                    raise RuntimeError("U_out contains rigid point not present in S_out.")
                if feat_2d_rigid_S is None:
                    raise RuntimeError("Stage4_6 expected rigid source features for U_out path.")
                rigid_out_acc_w_mean = float(acc_w_rigid_S[rows_S].mean().item()) if acc_w_rigid_S is not None else 0.0
                if feat_rigid_out_input_all_from_struct is not None:
                    feat_rigid_out_input = feat_rigid_out_input_all_from_struct[rows_S_out]
                else:
                    feat_2d_U_out = feat_2d_rigid_S[rows_S]
                    feat_rigid_out_input = self.distant_feat_proj(feat_2d_U_out)
                params_rigid_out_world = self._build_rigid_params_for_embed_source_world(
                    node_state_rigid, source_frame_idx, U_out
                )
                offsets_rigid_out_world, h_new_rigid_out = self._predict_offsets_gru_with_heads(
                    feat_rigid_out_input,
                    params_rigid_out_world,
                    h_old_rigid[U_out],
                    mask_update=torch.ones(U_out.numel(), dtype=torch.bool, device=self.device),
                    limits=self.distant_cfg["limits"],
                    mlp_offset_pos=self.mlp_offset_pos_distant,
                    mlp_conv=self.mlp_conv_distant,
                    mlp_opacity=self.mlp_opacity_distant,
                    gaussion_decoder=self.gaussion_decoder_distant,
                    freeze_quat=self.distant_freeze_quat,
                )
                if callable(apply_update_gate_fn) and gate_rigid_out is not None:
                    gate_rigid_out_sel = _select_gate_rows(gate_rigid_out, rows_S_out)
                    offsets_rigid_out_world, h_new_rigid_out, eff_gate_rigid_out = apply_update_gate_fn(
                        offsets_rigid_out_world,
                        h_old=h_old_rigid[U_out],
                        h_candidate=h_new_rigid_out,
                        gate=gate_rigid_out_sel,
                        mask_update=torch.ones(U_out.numel(), dtype=torch.bool, device=self.device),
                    )
                render_out = self._render_params_from_routed_offsets_rigid_local(
                    node_state_rigid=node_state_rigid,
                    source_frame_idx=source_frame_idx,
                    U=U_out,
                    offsets_world=offsets_rigid_out_world,
                )
            if h_new_rigid_in is not None:
                h_new_rigid_full[U_in] = h_new_rigid_in
            if h_new_rigid_out is not None:
                h_new_rigid_full[U_out] = h_new_rigid_out
            U_all = torch.cat([U_in, U_out], dim=0) if (U_in.numel() + U_out.numel()) > 0 else U_all
            if U_all.numel() > 0:
                render_params_rigid_local_U = self._pack_rigid_local_subsets(render_in, render_out)
                rigid_means_world_for_stats = self._transform_rigid_to_world(
                    node_state_rigid,
                    render_params_rigid_local_U["means_r"].detach(),
                    source_frame_idx,
                    point_ids_subset=node_state_rigid.point_ids[U_all, 0],
                )
        if node_state_rigid is None:
            h_new_rigid_full = None
        if node_state_rigid is not None:
            if U_all.dtype != torch.long:
                raise RuntimeError("Stage4_6 expects U_all to be torch.long.")
            if U_all.device.type != self.device.type:
                raise RuntimeError(
                    f"Stage4_6 expects U_all device type {self.device.type}, got {U_all.device.type}."
                )
            # torch.device("cuda") != torch.device("cuda:0"), so only enforce index
            # consistency when self.device explicitly pins one.
            if self.device.index is not None and U_all.device.index != self.device.index:
                raise RuntimeError(
                    f"Stage4_6 expects U_all on {self.device}, got {U_all.device}."
                )
            if bool((U_all < 0).any().item()) or bool((U_all >= N_rigid).any().item()):
                raise RuntimeError("Stage4_6 U_all out of range.")
            if render_params_rigid_local_U is not None and U_all.numel() != int(render_params_rigid_local_U["means_r"].shape[0]):
                raise RuntimeError("Stage4_6 U_all and render_params_rigid_local_U row mismatch.")

        by_frame: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
        for i, t in enumerate(targets):
            by_frame[int(t["frame_idx"])].append((i, t))
        sorted_frames = sorted(by_frame.keys())
        mask_train_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        if U_all.numel() > 0:
            mask_train_rigid[U_all] = True

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
                idx_tr = torch.nonzero(mask_train_rigid & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                idx_fr = torch.nonzero((~mask_train_rigid) & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
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
                merged_f = (
                    _merge_params_bg_rigid_distant(proxies_bg_l, prox_r, proxies_dist_l)
                    if training
                    else self._tensor_merge_bg_rigid_distant_world(render_params_bg, rw, render_params_distant)
                )

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
            pred_by_idx, _ = _run_frame_renders(False, {}, None, render_params_rigid_local_U, U_all)
            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []
            for i in range(len(targets)):
                pr, _acc = pred_by_idx[i]
                pred_rgbs.append(pr)
                gt = targets[i]["gt_image"]
                if gt.dim() == 4:
                    gt = gt.squeeze(0)
                gt_images.append(gt)
            return {
                "loss": torch.tensor(0.0, device=self.device),
                "render_params": render_params_bg,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_render_params_rigid_world": None,
                "_render_params_rigid_local": render_params_rigid_local_U,
                "_node_state_bg": node_state_bg,
                "_node_state_distant": node_state_distant,
                "_node_state_rigid": node_state_rigid,
                "_h_new_bg": h_new_bg,
                "_h_new_distant": h_new_distant,
                "_h_new_rigid": h_new_rigid_full,
                "_bg_writeback_idx": torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1),
                "_distant_writeback_idx": (
                    torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
                ),
                "_rigid_writeback_idx": U_all,
                "_rigid_valid_idx": route.S,
                "_num_rigid_valid_src": int(route.S.numel()),
                "_num_rigid_total": N_rigid,
                "_cache_key": key,
                "_src_backproject_pass_count": src_backproject_pass_count,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        pred_by_idx, rigid_world_proxy_pairs = _run_frame_renders(
            True, proxies_bg, proxies_distant, render_params_rigid_local_U, U_all
        )
        pred_rgbs_t: List[torch.Tensor] = []
        gt_images_t: List[torch.Tensor] = []
        opacities_t: List[torch.Tensor] = []
        for i in range(len(targets)):
            pr, acc = pred_by_idx[i]
            pred_rgbs_t.append(pr)
            gt = targets[i]["gt_image"]
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            gt_images_t.append(gt)
            opacities_t.append(acc)

        current_loss_step = self._current_loss_step(batch)
        target_view_weights, target_view_roles = self._build_target_view_weights(
            batch,
            step=current_loss_step,
            num_targets=len(targets),
        )
        normalize_by_weight_sum = bool(self._target_view_weight_cfg["normalize_by_weight_sum"])
        weight_eps = 1e-8
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        weighted_l1_sum = zero.clone()
        weighted_ssim_sum = zero.clone()
        weighted_mask_sum = zero.clone()
        weighted_entropy_sum = zero.clone()
        total_weight_sum = zero.clone()
        frame_loss_num: Dict[int, torch.Tensor] = {}
        frame_loss_den: Dict[int, torch.Tensor] = {}
        frame_loss_map: Dict[int, float] = {}
        eff_frames = 0
        views_no_non_sky = 0
        role_rgb_num: Dict[str, torch.Tensor] = {}
        role_rgb_den: Dict[str, torch.Tensor] = {}
        role_counts: Dict[str, int] = {}
        monitor_role_l1_sum: Dict[str, float] = {}
        monitor_role_ssim_sum: Dict[str, float] = {}
        monitor_role_rgb_sum: Dict[str, float] = {}
        monitor_role_psnr_sum: Dict[str, float] = {}
        monitor_role_count: Dict[str, int] = {}
        monitor_role_weight_sum: Dict[str, float] = {}
        monitor_all_l1_sum = 0.0
        monitor_all_ssim_sum = 0.0
        monitor_all_rgb_sum = 0.0
        monitor_all_psnr_sum = 0.0
        monitor_all_count = 0
        for F in sorted_frames:
            group = by_frame[F]
            for orig_i, t in group:
                pred_rgb = pred_rgbs_t[orig_i]
                gt_image = gt_images_t[orig_i]
                opacity = opacities_t[orig_i].to(self.device).float()
                view_weight = target_view_weights[orig_i]
                role = str(target_view_roles[orig_i]) if orig_i < len(target_view_roles) else "source"
                if opacity.dim() == 3 and opacity.shape[-1] == 1:
                    opacity = opacity.squeeze(-1)
                h, w = gt_image.shape[0], gt_image.shape[1]
                valid_loss_mask = self._valid_loss_mask_from_target(t, height=h, width=w)
                if float(valid_loss_mask.sum().item()) <= 0:
                    continue
                sky_mask = t.get("sky_mask")
                if self.require_sky_mask_for_loss and sky_mask is None:
                    raise ValueError("Stage4_6 requires target['sky_mask'] for loss computation.")
                if sky_mask is None:
                    sm = torch.zeros_like(valid_loss_mask)
                else:
                    sm = sky_mask.to(self.device).float()
                    if sm.dim() == 3:
                        sm = sm.squeeze(-1)
                    if sm.shape != valid_loss_mask.shape:
                        raise ValueError(
                            "target['sky_mask'] shape mismatch with gt image. "
                            f"got {tuple(sm.shape)} expected {tuple(valid_loss_mask.shape)}"
                        )
                valid_non_sky_mask = valid_loss_mask * (1.0 - sm).clamp(0.0, 1.0)
                non_sky_pixels = float(valid_non_sky_mask.sum().item())
                if non_sky_pixels > 0.0:
                    l1_numer = (torch.abs(pred_rgb - gt_image) * valid_non_sky_mask.unsqueeze(-1)).sum()
                    l1_i = self.loss_w_l1 * (l1_numer / (valid_non_sky_mask.sum() * 3.0))
                    mse_i = (
                        ((pred_rgb - gt_image) ** 2 * valid_non_sky_mask.unsqueeze(-1)).sum()
                        / (valid_non_sky_mask.sum() * 3.0)
                    )
                    ssim_i = self.loss_w_ssim * compute_ssim_loss_masked(
                        pred_rgb, gt_image, valid_mask=valid_non_sky_mask, sky_mask=None, data_range=1.0
                    )
                else:
                    views_no_non_sky += 1
                    l1_i = pred_rgb.sum() * 0.0
                    mse_i = pred_rgb.sum() * 0.0
                    ssim_i = pred_rgb.sum() * 0.0
                gt_occupied = (1.0 - sm) * valid_loss_mask
                pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
                mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)
                p = opacity.clamp(1e-6, 1.0 - 1e-6)
                entropy_i = self.loss_w_opacity_entropy * self._masked_mean(-p * torch.log(p), valid_loss_mask)
                total_i = l1_i + ssim_i + mask_i + entropy_i
                weighted_l1_sum = weighted_l1_sum + l1_i * view_weight
                weighted_ssim_sum = weighted_ssim_sum + ssim_i * view_weight
                weighted_mask_sum = weighted_mask_sum + mask_i * view_weight
                weighted_entropy_sum = weighted_entropy_sum + entropy_i * view_weight
                total_weight_sum = total_weight_sum + view_weight
                if int(F) not in frame_loss_num:
                    frame_loss_num[int(F)] = total_i * view_weight
                    frame_loss_den[int(F)] = view_weight
                else:
                    frame_loss_num[int(F)] = frame_loss_num[int(F)] + (total_i * view_weight)
                    frame_loss_den[int(F)] = frame_loss_den[int(F)] + view_weight
                role_rgb_num[role] = role_rgb_num.get(role, zero.clone()) + (l1_i * view_weight)
                role_rgb_den[role] = role_rgb_den.get(role, zero.clone()) + view_weight
                role_counts[role] = int(role_counts.get(role, 0)) + 1
                l1_det = float(l1_i.detach().item())
                ssim_det = float(ssim_i.detach().item())
                mse_det = float(mse_i.detach().item())
                psnr_det = float(-10.0 * math.log10(max(mse_det, 1.0e-12))) if non_sky_pixels > 0.0 else 0.0
                rgb_det = float(l1_det + ssim_det)
                vw_det = float(view_weight.detach().item())
                monitor_all_l1_sum += l1_det
                monitor_all_ssim_sum += ssim_det
                monitor_all_rgb_sum += rgb_det
                monitor_all_psnr_sum += psnr_det
                monitor_all_count += 1
                monitor_role_l1_sum[role] = float(monitor_role_l1_sum.get(role, 0.0)) + l1_det
                monitor_role_ssim_sum[role] = float(monitor_role_ssim_sum.get(role, 0.0)) + ssim_det
                monitor_role_rgb_sum[role] = float(monitor_role_rgb_sum.get(role, 0.0)) + rgb_det
                monitor_role_psnr_sum[role] = float(monitor_role_psnr_sum.get(role, 0.0)) + psnr_det
                monitor_role_count[role] = int(monitor_role_count.get(role, 0)) + 1
                monitor_role_weight_sum[role] = float(monitor_role_weight_sum.get(role, 0.0)) + vw_det
        if normalize_by_weight_sum:
            denom = torch.clamp(total_weight_sum, min=weight_eps)
        else:
            denom = torch.clamp(
                torch.tensor(float(sum(int(v) for v in role_counts.values())), dtype=torch.float32, device=self.device),
                min=1.0,
            )
        loss = (weighted_l1_sum + weighted_ssim_sum + weighted_mask_sum + weighted_entropy_sum) / denom
        l1_mean = weighted_l1_sum / denom
        ssim_mean = weighted_ssim_sum / denom
        mask_mean = weighted_mask_sum / denom
        entropy_mean = weighted_entropy_sum / denom
        monitor_den = max(int(monitor_all_count), 1)
        monitor_l1_all = float(monitor_all_l1_sum / float(monitor_den))
        monitor_ssim_all = float(monitor_all_ssim_sum / float(monitor_den))
        monitor_rgb_all = float(monitor_all_rgb_sum / float(monitor_den))
        monitor_psnr_all = float(monitor_all_psnr_sum / float(monitor_den))
        for fidx, num in frame_loss_num.items():
            den = torch.clamp(frame_loss_den[fidx], min=weight_eps)
            frame_loss_map[int(fidx)] = float((num / den).detach().item())
        eff_frames = int(len(frame_loss_num))
        offsets_rigid_for_stats = None
        if offsets_rigid_in_world is not None and offsets_rigid_out_world is not None:
            offsets_rigid_for_stats = {
                k: torch.cat([offsets_rigid_in_world[k], offsets_rigid_out_world[k]], dim=0)
                for k in offsets_rigid_in_world.keys()
            }
        elif offsets_rigid_in_world is not None:
            offsets_rigid_for_stats = offsets_rigid_in_world
        elif offsets_rigid_out_world is not None:
            offsets_rigid_for_stats = offsets_rigid_out_world
        offset_stats = self._collect_offset_stats(offsets_bg, offsets_rigid_for_stats)
        hidden_stats = self._collect_hidden_norms(h_new_bg, h_new_distant, h_new_rigid_full)

        bg_writeback_idx = torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1)
        distant_writeback_idx = (
            torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
        )
        rigid_src_feat_valid = int(mask_src_feat_valid_rigid.sum().item())
        rigid_update_count = int(U_all.numel())
        gate_attr_keys = ("means", "scales", "quat", "opacity", "sh", "hidden")

        def _gate_mean(gate_eff, key: str) -> Optional[float]:
            if gate_eff is None:
                return None
            if isinstance(gate_eff, dict):
                v = gate_eff.get(key)
                if v is None or not torch.is_tensor(v) or int(v.numel()) == 0:
                    return None
                return float(v.mean().detach().item())
            if torch.is_tensor(gate_eff) and int(gate_eff.numel()) > 0:
                return float(gate_eff.mean().detach().item())
            return None

        if eff_gate_bg is not None:
            for k in gate_attr_keys:
                mv = _gate_mean(eff_gate_bg, k)
                if mv is not None:
                    bg_rigid_in_aux[f"stage5_3_gate_bg_{k}_mean"] = mv
            hidden_mv = _gate_mean(eff_gate_bg, "hidden")
            if hidden_mv is not None:
                bg_rigid_in_aux["stage5_2_gate_bg_mean"] = hidden_mv
        if eff_gate_distant is not None:
            for k in gate_attr_keys:
                mv = _gate_mean(eff_gate_distant, k)
                if mv is not None:
                    bg_rigid_in_aux[f"stage5_3_gate_distant_{k}_mean"] = mv
            hidden_mv = _gate_mean(eff_gate_distant, "hidden")
            if hidden_mv is not None:
                bg_rigid_in_aux["stage5_2_gate_distant_mean"] = hidden_mv
        if eff_gate_rigid_in is not None:
            for k in gate_attr_keys:
                mv = _gate_mean(eff_gate_rigid_in, k)
                if mv is not None:
                    bg_rigid_in_aux[f"stage5_3_gate_rigid_in_{k}_mean"] = mv
            hidden_mv = _gate_mean(eff_gate_rigid_in, "hidden")
            if hidden_mv is not None:
                bg_rigid_in_aux["stage5_2_gate_rigid_in_mean"] = hidden_mv
        if eff_gate_rigid_out is not None:
            for k in gate_attr_keys:
                mv = _gate_mean(eff_gate_rigid_out, k)
                if mv is not None:
                    bg_rigid_in_aux[f"stage5_3_gate_rigid_out_{k}_mean"] = mv
            hidden_mv = _gate_mean(eff_gate_rigid_out, "hidden")
            if hidden_mv is not None:
                bg_rigid_in_aux["stage5_2_gate_rigid_out_mean"] = hidden_mv
        if stage5_2_full is not None and hasattr(self, "_stage5_2_last_full_inputs"):
            self._stage5_2_last_full_inputs = None

        return {
            "loss": loss,
            "loss_l1": l1_mean,
            "loss_ssim": ssim_mean,
            "loss_mask": mask_mean,
            "loss_opacity_entropy": entropy_mean,
            "loss_optim": float(loss.detach().item()),
            "loss_optim_l1": float(l1_mean.detach().item()),
            "loss_optim_ssim": float(ssim_mean.detach().item()),
            "loss_optim_mask": float(mask_mean.detach().item()),
            "loss_optim_opacity_entropy": float(entropy_mean.detach().item()),
            "loss_optim_weight_sum": float(total_weight_sum.detach().item()),
            "loss_optim_num_images": int(sum(int(v) for v in role_counts.values())),
            "loss_optim_normalize_by_weight_sum": float(1.0 if normalize_by_weight_sum else 0.0),
            "monitor/l1/all": float(monitor_l1_all),
            "monitor/ssim/all": float(monitor_ssim_all),
            "monitor/rgb/all": float(monitor_rgb_all),
            "monitor/psnr/all": float(monitor_psnr_all),
            "monitor/l1_all_unweighted": float(monitor_l1_all),
            "monitor/ssim_all_unweighted": float(monitor_ssim_all),
            "monitor/rgb_all_unweighted": float(monitor_rgb_all),
            "monitor/psnr_all_unweighted": float(monitor_psnr_all),
            "render_params": render_params_bg,
            "_render_params_bg": render_params_bg,
            "proxies": proxies_bg,
            "_proxies_bg": proxies_bg,
            "_proxies_distant": proxies_distant,
            "_proxies_rigid_world": None,
            "_rigid_world_proxy_pairs": rigid_world_proxy_pairs if rigid_world_proxy_pairs else None,
            "_render_params_distant": render_params_distant,
            "_render_params_rigid_world": None,
            "_render_params_rigid_local": render_params_rigid_local_U,
            "_node_state_bg": node_state_bg,
            "_node_state_distant": node_state_distant,
            "_node_state_rigid": node_state_rigid,
            "_h_new_bg": h_new_bg,
            "_h_new_distant": h_new_distant,
            "_h_new_rigid": h_new_rigid_full,
            "_bg_writeback_idx": bg_writeback_idx,
            "_distant_writeback_idx": distant_writeback_idx,
            "_rigid_valid_idx": route.S,
            "_rigid_writeback_idx": U_all,
            "_num_rigid_valid_src": int(route.S.numel()),
            "_num_rigid_src_feat_valid": rigid_src_feat_valid,
            "_num_rigid_update": rigid_update_count,
            "_num_target_frames": len(sorted_frames),
            "_loss_effective_frames": eff_frames,
            "_num_rigid_total": N_rigid,
            "_frame_loss_map": frame_loss_map,
            "_offset_stats": offset_stats,
            "_hidden_stats": hidden_stats,
            "_rigid_update_ratio": float(rigid_update_count / max(int(N_rigid), 1)),
            "_rigid_update_among_feat_valid": float(rigid_update_count / max(rigid_src_feat_valid, 1)),
            "_num_bg_src_feat_valid": int(mask_src_feat_valid_bg.sum().item()),
            "_num_bg_update": int(bg_writeback_idx.numel()),
            "_num_distant_src_feat_valid": int(mask_src_feat_valid_distant.sum().item()) if mask_src_feat_valid_distant is not None else 0,
            "_num_distant_update": int(distant_writeback_idx.numel()) if distant_writeback_idx is not None else 0,
            "_src_backproject_pass_count": src_backproject_pass_count,
            "_cache_key": key,
            "_num_views_no_non_sky_supervision": int(views_no_non_sky),
            **{
                f"loss/target_weight/{str(role)}": float(self._target_role_weight(str(role), current_loss_step))
                for role in sorted(role_rgb_num.keys(), key=str)
            },
            **{
                f"loss/rgb/{str(role)}": float(
                    (role_rgb_num[str(role)] / torch.clamp(role_rgb_den[str(role)], min=weight_eps)).detach().item()
                )
                for role in sorted(role_rgb_num.keys(), key=str)
            },
            **{
                f"monitor/l1/{str(role)}": float(
                    monitor_role_l1_sum[str(role)] / max(int(monitor_role_count[str(role)]), 1)
                )
                for role in sorted(monitor_role_count.keys(), key=str)
            },
            **{
                f"monitor/ssim/{str(role)}": float(
                    monitor_role_ssim_sum[str(role)] / max(int(monitor_role_count[str(role)]), 1)
                )
                for role in sorted(monitor_role_count.keys(), key=str)
            },
            **{
                f"monitor/rgb/{str(role)}": float(
                    monitor_role_rgb_sum[str(role)] / max(int(monitor_role_count[str(role)]), 1)
                )
                for role in sorted(monitor_role_count.keys(), key=str)
            },
            **{
                f"monitor/psnr/{str(role)}": float(
                    monitor_role_psnr_sum[str(role)] / max(int(monitor_role_count[str(role)]), 1)
                )
                for role in sorted(monitor_role_count.keys(), key=str)
            },
            **{
                f"monitor/count/{str(role)}": float(monitor_role_count[str(role)])
                for role in sorted(monitor_role_count.keys(), key=str)
            },
            **{
                f"monitor/weight_sum/{str(role)}": float(monitor_role_weight_sum[str(role)])
                for role in sorted(monitor_role_weight_sum.keys(), key=str)
            },
            "pred_rgbs": pred_rgbs_t,
            "gt_images": gt_images_t,
            "pred_rgb": pred_rgbs_t[0],
            "gt_image": gt_images_t[0],
            "rigid_route_num_S": int(route.S.numel()),
            "rigid_route_num_in": int(route.S_in.numel()),
            "rigid_route_num_out": int(route.S_out.numel()),
            "rigid_route_ratio_in": float(route.S_in.numel() / max(int(route.S.numel()), 1)),
            "rigid_route_ratio_out": float(route.S_out.numel() / max(int(route.S.numel()), 1)),
            "rigid_in_update_count": int(U_in.numel()),
            "rigid_out_update_count": int(U_out.numel()),
            "rigid_in_acc_w_mean": float(rigid_in_acc_w_mean),
            "rigid_out_acc_w_mean": float(rigid_out_acc_w_mean),
            "rigid_writeback_count": int(U_all.numel()),
            **bg_rigid_in_aux,
        }

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = super().train_step(
            batch=batch,
            step=step,
            profile_phase_timing=profile_phase_timing,
            sync_cuda_timing=sync_cuda_timing,
            scheduler_node_sync=scheduler_node_sync,
        )
        # Stage4_6 no longer has rigid-specific trainable heads.
        # Keep legacy metric for compatibility and expose shared-head diagnostics.
        out["grad_norm_rigid_legacy"] = float(out.get("grad_norm_rigid", 0.0))
        out["rigid_grad_norm_routed_to_bg_shared"] = float(out.get("grad_norm_bg", 0.0))
        out["rigid_grad_norm_routed_to_distant_shared"] = float(out.get("grad_norm_distant", 0.0))
        out["rigid_grad_norm_legacy_flag"] = 1.0
        return out


__all__ = [
    "BgRigidInGRUInputs",
    "MinimalStreetForwardStage4_5BaseNoRigidHead",
    "MinimalStreetForwardStage4_6",
    "RigidRoute",
]
