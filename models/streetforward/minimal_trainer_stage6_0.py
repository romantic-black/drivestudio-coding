from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors import StudentPriorFusionUNet
from models.streetforward.math_utils import get_viewmat
from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_bridge import TeacherPriorAdapter, update_teacher_prior_cache_detached
from models.streetforward.teacher_student_prior import TeacherPriorCache, create_teacher_prior_cache


class MinimalStreetForwardStage6_0(MinimalStreetForwardStage5_4):
    """Stage6_0 trainer with strict domain naming and fast-fail guards."""

    def __init__(self, config, device: torch.device, **kwargs):
        self._stage6_active_batch: Optional[Dict[str, Any]] = None
        self._stage6_teacher_prior_caches: Dict[Tuple[int, int], TeacherPriorCache] = {}
        self._stage6_last_role: str = "teacher"
        self._stage6_last_prior_conf_map: Optional[torch.Tensor] = None
        self._stage6_role_fallback: bool = False
        super().__init__(config=config, device=device, **kwargs)

        sv10 = self._cfg_get(config, "scheduler_v10", {})
        targets = self._cfg_get(sv10, "targets", {})
        weights = self._cfg_get(targets, "weights", {})
        self._scheduler_v10_target_weights = {
            "teacher_source": float(self._cfg_get(weights, "teacher_source", 1.0)),
            "student_source": float(self._cfg_get(weights, "student_source", 1.0)),
            "teacher_anchor": float(self._cfg_get(weights, "teacher_anchor", 0.1)),
            "history_visited": float(self._cfg_get(weights, "history_visited", 0.1)),
            "probe_near": float(self._cfg_get(weights, "probe_near", 0.0)),
        }

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return default
        if hasattr(node, "get"):
            out = node.get(key)
            return default if out is None else out
        if isinstance(node, dict):
            out = node.get(key)
            return default if out is None else out
        if hasattr(node, key):
            out = getattr(node, key)
            return default if out is None else out
        return default

    def _validate_stage5_3_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model"))
        if stage != "6_0":
            raise ValueError("Stage6_0 requires model.stage='6_0'.")

        old_stage = model_cfg.get("stage")
        model_cfg["stage"] = "5_4"
        try:
            super()._validate_stage5_3_config(config)
        finally:
            model_cfg["stage"] = old_stage
        self._validate_stage6_0_config(config)

    @staticmethod
    def _contains_forbidden_key(node: Any, forbidden: List[str]) -> Optional[str]:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k)
                if key in forbidden:
                    return key
                out = MinimalStreetForwardStage6_0._contains_forbidden_key(v, forbidden)
                if out is not None:
                    return out
        elif isinstance(node, list):
            for v in node:
                out = MinimalStreetForwardStage6_0._contains_forbidden_key(v, forbidden)
                if out is not None:
                    return out
        return None

    def _validate_stage6_0_config(self, config) -> None:
        scheduler_v10 = self._require_key(config, "scheduler_v10", "config")
        if bool(self._require_key(scheduler_v10, "enable", "scheduler_v10")) is not True:
            raise ValueError("Stage6_0 requires scheduler_v10.enable=true.")
        stage6_cfg = self._require_key(config, "stage6_0", "config")
        if bool(self._require_key(stage6_cfg, "enable", "stage6_0")) is not True:
            raise ValueError("Stage6_0 requires stage6_0.enable=true.")

        forbidden = ["student_preserve", "teacher_preserve", "preserve"]
        forbidden_hit = self._contains_forbidden_key(stage6_cfg, forbidden)
        if forbidden_hit is not None:
            raise ValueError(
                f"Stage6_0 forbids key '{forbidden_hit}'. Use 'teacher_anchor' and structured domains."
            )

        losses_cfg = self._cfg_get(config, "losses", {}) or {}
        stage6_losses = self._cfg_get(losses_cfg, "stage6_0", {}) or {}
        forbidden_loss = self._contains_forbidden_key(stage6_losses, forbidden)
        if forbidden_loss is not None:
            raise ValueError(f"Stage6_0 losses forbid '{forbidden_loss}'.")

        bridge_cfg = self._require_key(stage6_cfg, "bridge", "stage6_0")
        live_cfg = self._require_key(bridge_cfg, "live", "stage6_0.bridge")
        cache_cfg = self._require_key(bridge_cfg, "cache", "stage6_0.bridge")
        student_cfg = self._require_key(stage6_cfg, "student", "stage6_0")
        valid_mask_cfg = self._require_key(student_cfg, "valid_mask", "stage6_0.student")

        if bool(self._cfg_get(live_cfg, "enable", False)) and not bool(
            self._cfg_get(live_cfg, "rerun_teacher_2d_current_step", False)
        ):
            raise ValueError("Stage6_0 requires bridge.live.rerun_teacher_2d_current_step=true when live is enabled.")
        if bool(self._cfg_get(cache_cfg, "detach_write", True)) is not True:
            raise ValueError("Stage6_0 requires bridge.cache.detach_write=true.")
        if bool(self._cfg_get(valid_mask_cfg, "apply_before_unet", False)) is not True:
            raise ValueError("Stage6_0 requires student.valid_mask.apply_before_unet=true.")
        if bool(self._cfg_get(valid_mask_cfg, "append_as_channel", False)) is not True:
            raise ValueError("Stage6_0 requires student.valid_mask.append_as_channel=true.")
        if bool(self._cfg_get(student_cfg, "input_history_context", False)):
            raise ValueError("Stage6_0 student.input_history_context is not implemented yet.")
        if bool(self._cfg_get(valid_mask_cfg, "mask_history_context", False)):
            raise ValueError(
                "Stage6_0 student.valid_mask.mask_history_context must be false until history context is wired."
            )

        probe_near_cfg = self._cfg_get(self._cfg_get(stage6_losses, "probe", {}), "near", {}) or {}
        near_loss_weight = float(self._cfg_get(probe_near_cfg, "loss_weight", 0.0))
        phase = str(self._cfg_get(stage6_cfg, "phase", "default"))
        if near_loss_weight != 0.0 and phase != "explicit_near_training":
            raise ValueError("Stage6_0 requires probe.near.loss_weight=0 unless phase=explicit_near_training.")
        target_weights = self._cfg_get(self._cfg_get(scheduler_v10, "targets", {}) or {}, "weights", {}) or {}
        scheduler_probe_weight = float(self._cfg_get(target_weights, "probe_near", 0.0))
        if scheduler_probe_weight != 0.0 and phase != "explicit_near_training":
            raise ValueError(
                "Stage6_0 requires scheduler_v10.targets.weights.probe_near=0 "
                "unless phase=explicit_near_training."
            )
        if phase == "warmup" and bool(self._cfg_get(live_cfg, "student_loss_to_teacher_backbone", False)):
            raise ValueError("Stage6_0 warmup forbids student_loss_to_teacher_backbone=true.")

    def _init_stage5_3_modules(self, config) -> None:
        super()._init_stage5_3_modules(config)
        stage6_cfg = self._require_key(config, "stage6_0", "config")
        prior_cfg = self._cfg_get(stage6_cfg, "teacher_prior", {}) or {}
        student_cfg = self._cfg_get(stage6_cfg, "student", {}) or {}
        student_extractor_cfg = self._cfg_get(student_cfg, "student_extractor", {}) or {}
        bridge_cfg = self._cfg_get(stage6_cfg, "bridge", {}) or {}
        live_cfg = self._cfg_get(bridge_cfg, "live", {}) or {}
        cache_cfg = self._cfg_get(bridge_cfg, "cache", {}) or {}
        valid_mask_cfg = self._cfg_get(student_cfg, "valid_mask", {}) or {}
        prior_dim = int(self._cfg_get(prior_cfg, "dim", int(self.stage5_2_feat_2d_channels)))
        if prior_dim != int(self.stage5_2_feat_2d_channels):
            raise ValueError(
                f"stage6_0.teacher_prior.dim must equal stage5_2_feat_2d_channels={int(self.stage5_2_feat_2d_channels)}."
            )

        use_conf = bool(self._cfg_get(student_extractor_cfg, "use_confidence", True))
        base_channels = int(self._cfg_get(student_extractor_cfg, "base_channels", 64))
        self.stage6_prior_dim = int(prior_dim)
        self.stage6_prior_conf_norm = float(self._cfg_get(prior_cfg, "confidence_norm", 1.0))
        self.stage6_student_use_conf = bool(use_conf)
        self.stage6_live_enable = bool(self._cfg_get(live_cfg, "enable", True))
        self.stage6_live_rerun_teacher_2d = bool(self._cfg_get(live_cfg, "rerun_teacher_2d_current_step", True))
        self.stage6_live_to_teacher_backbone = bool(self._cfg_get(live_cfg, "student_loss_to_teacher_backbone", False))
        self.stage6_live_detach_geometry = bool(self._cfg_get(live_cfg, "detach_geometry", True))
        self.stage6_live_detach_opacity = bool(self._cfg_get(live_cfg, "detach_opacity", True))
        self.stage6_live_require_on_student = bool(self._cfg_get(live_cfg, "require_on_student", False))
        self.stage6_student_input_history_context = bool(self._cfg_get(student_cfg, "input_history_context", False))
        if self.stage6_student_input_history_context:
            raise ValueError("Stage6_0 student.input_history_context is not implemented yet.")
        self.stage6_cache_detach_write = bool(self._cfg_get(cache_cfg, "detach_write", True))
        self.stage6_cache_detach_read = bool(self._cfg_get(cache_cfg, "detach_read", True))
        self.stage6_student_append_valid_mask = bool(self._cfg_get(valid_mask_cfg, "append_as_channel", True))
        self.stage6_student_apply_mask_before_unet = bool(self._cfg_get(valid_mask_cfg, "apply_before_unet", True))
        self.stage6_student_zero_invalid_output = bool(self._cfg_get(valid_mask_cfg, "zero_invalid_output", True))
        self.stage6_prior_eps = float(self._cfg_get(prior_cfg, "eps", 1.0e-6))
        self.stage6_prior_support_min_bg = float(self._cfg_get(prior_cfg, "support_min_bg", self.bg_src_backproject_support_min))
        self.stage6_prior_support_min_distant = float(
            self._cfg_get(prior_cfg, "support_min_distant", self.distant_src_backproject_support_min)
        )
        self.stage6_prior_support_min_rigid = float(
            self._cfg_get(prior_cfg, "support_min_rigid", self.rigid_src_backproject_support_min)
        )
        self.stage6_teacher_prior_adapter = TeacherPriorAdapter(dim=int(prior_dim)).to(self.device)
        self.student_prior_fusion_unet = StudentPriorFusionUNet(
            prior_dim=int(prior_dim),
            out_dim=int(self.stage5_2_feat_2d_channels),
            base_dim=int(base_channels),
            use_confidence=bool(use_conf),
            extra_input_channels=1 if self.stage6_student_append_valid_mask else 0,
        ).to(self.device)

    def _current_global_step(self, batch: Dict[str, Any]) -> int:
        opt = getattr(self, "optimizer", None)
        if opt is not None and hasattr(opt, "global_step"):
            return int(getattr(opt, "global_step"))
        aligned = (
            batch.get("_scheduler_v10_aligned_info")
            or batch.get("_scheduler_v9_aligned_info")
            or batch.get("_scheduler_v8_aligned_info")
            or {}
        )
        return int(aligned.get("global_step", 0))

    @staticmethod
    def _teacher_prior_available(*, cache: Optional[TeacherPriorCache], route: RigidRoute) -> bool:
        if cache is None:
            return False
        if bool(cache.bg.valid.any().item()):
            return True
        if bool(cache.distant.valid.any().item()):
            return True
        if int(route.S.numel()) > 0 and bool(cache.rigid.valid[route.S].any().item()):
            return True
        return False

    def _get_stage6_role(self, *, batch: Dict[str, Any], cache: TeacherPriorCache, route: RigidRoute) -> str:
        meta = batch.get("request_meta") or {}
        requested_role = str(meta.get("stage6_role", meta.get("stage5_5_role", "teacher"))).strip().lower()
        if requested_role not in {"teacher", "student"}:
            raise ValueError(f"Unknown Stage6_0 role: {requested_role}")
        role = requested_role
        if role == "student" and not self._teacher_prior_available(cache=cache, route=route):
            req_v10 = self._cfg_get(self._cfg_get(batch, "request_meta", {}), "scheduler_request_v10", {}) or {}
            if bool(self._cfg_get(req_v10, "fallback_to_teacher", True)):
                role = "teacher"
            else:
                raise RuntimeError("Stage6_0 student step requested but teacher prior is unavailable.")
        self._stage6_role_fallback = bool(requested_role == "student" and role == "teacher")
        return role

    def _batch_key(self, batch: Dict[str, Any]) -> Tuple[int, int]:
        scene_id = int(batch.get("scene_id", -1))
        segment_id = int(batch.get("segment_id", -1))
        if scene_id < 0 or segment_id < 0:
            raise ValueError("batch must include non-negative scene_id/segment_id for Stage6_0 cache.")
        return int(scene_id), int(segment_id)

    def _get_or_init_teacher_prior_cache(
        self,
        *,
        batch: Dict[str, Any],
        num_bg: int,
        num_distant: int,
        num_rigid: int,
        feat_dim: int,
        dtype: torch.dtype,
    ) -> TeacherPriorCache:
        key = self._batch_key(batch)
        cache = self._stage6_teacher_prior_caches.get(key)
        if cache is None:
            cache = create_teacher_prior_cache(
                num_bg=int(num_bg),
                num_distant=int(num_distant),
                num_rigid=int(num_rigid),
                feat_dim=int(feat_dim),
                device=self.device,
                dtype=dtype,
            )
            self._stage6_teacher_prior_caches[key] = cache
            return cache
        if (
            int(cache.bg.feat.shape[0]) != int(num_bg)
            or int(cache.distant.feat.shape[0]) != int(num_distant)
            or int(cache.rigid.feat.shape[0]) != int(num_rigid)
        ):
            cache = create_teacher_prior_cache(
                num_bg=int(num_bg),
                num_distant=int(num_distant),
                num_rigid=int(num_rigid),
                feat_dim=int(feat_dim),
                device=self.device,
                dtype=dtype,
            )
            self._stage6_teacher_prior_caches[key] = cache
        return cache

    def _build_prior_confidence(self, *, support: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        conf = torch.log1p(torch.clamp(support, min=0.0))
        conf = conf / float(max(self.stage6_prior_conf_norm, 1.0e-6))
        conf = conf.clamp(0.0, 1.0)
        conf = conf * valid.float()
        return conf.unsqueeze(-1)

    def _render_prior_from_components(
        self,
        *,
        feat_bg: torch.Tensor,
        support_bg: torch.Tensor,
        valid_bg: torch.Tensor,
        feat_distant: Optional[torch.Tensor],
        support_distant: Optional[torch.Tensor],
        valid_distant: Optional[torch.Tensor],
        feat_rigid_s: Optional[torch.Tensor],
        support_rigid_s: Optional[torch.Tensor],
        valid_rigid_s: Optional[torch.Tensor],
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_parts = [feat_bg]
        support_parts = [support_bg]
        valid_parts = [valid_bg]
        if feat_distant is not None:
            feat_parts.append(feat_distant)
            support_parts.append(support_distant if support_distant is not None else feat_distant.new_zeros((feat_distant.shape[0],)))
            valid_parts.append(valid_distant if valid_distant is not None else feat_distant.new_zeros((feat_distant.shape[0],), dtype=torch.bool))
        if feat_rigid_s is not None:
            feat_parts.append(feat_rigid_s)
            support_parts.append(support_rigid_s if support_rigid_s is not None else feat_rigid_s.new_zeros((feat_rigid_s.shape[0],)))
            valid_parts.append(valid_rigid_s if valid_rigid_s is not None else feat_rigid_s.new_zeros((feat_rigid_s.shape[0],), dtype=torch.bool))
        prior_feat_all = torch.cat(feat_parts, dim=0).to(device=self.device)
        prior_support = torch.cat([x.to(device=self.device) for x in support_parts], dim=0)
        prior_valid = torch.cat([x.to(device=self.device) for x in valid_parts], dim=0)

        conf = self._build_prior_confidence(support=prior_support, valid=prior_valid)
        packed = torch.cat([prior_feat_all * conf, conf], dim=-1)
        viewmats_list: List[torch.Tensor] = []
        ks_list: List[torch.Tensor] = []
        for i, v in enumerate(source_views):
            cam_ctw = v.camtoworlds if hasattr(v, "camtoworlds") else v["camtoworlds"]
            vm = get_viewmat(cam_ctw)
            if vm.dim() == 2:
                vm = vm.unsqueeze(0)
            if vm.dim() != 3 or tuple(vm.shape[-2:]) != (4, 4) or int(vm.shape[0]) != 1:
                raise ValueError(f"Stage6_0 invalid viewmat shape at source_view[{i}]: {tuple(vm.shape)}")
            viewmats_list.append(vm)
            if hasattr(v, "Ks"):
                km = v.Ks[0]
            elif hasattr(v, "K"):
                km = v.K
                if km.dim() == 3:
                    km = km[0]
            else:
                raise ValueError("Stage6_0 source view is missing intrinsic matrix K/Ks.")
            ks_list.append(km.to(device=self.device))
        viewmats = torch.cat(viewmats_list, dim=0)
        Ks = torch.stack(ks_list, dim=0)
        means = gaussians_scene["means"]
        quats = gaussians_scene["quats"]
        scales = gaussians_scene["scales"]
        opacities = gaussians_scene["opacities"]
        if bool(getattr(self, "stage6_live_detach_geometry", True)):
            means = means.detach()
            quats = quats.detach()
            scales = scales.detach()
        if bool(getattr(self, "stage6_live_detach_opacity", True)):
            opacities = opacities.detach()
        rendered, _, _ = self.renderer(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=packed,
            viewmats=viewmats,
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
        )
        feat_sum = rendered[..., : self.stage6_prior_dim]
        conf_map = rendered[..., self.stage6_prior_dim : self.stage6_prior_dim + 1]
        prior_map = feat_sum / (conf_map + float(self.stage6_prior_eps))
        return prior_map, conf_map

    def _render_teacher_prior_to_source_view(
        self,
        *,
        cache: TeacherPriorCache,
        route: RigidRoute,
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rigid_idx = route.S
        feat_rigid = cache.rigid.feat[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.feat[:0]
        support_rigid = cache.rigid.support[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.support[:0]
        valid_rigid = cache.rigid.valid[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.valid[:0]
        feat_bg = cache.bg.feat.detach() if self.stage6_cache_detach_read else cache.bg.feat
        feat_distant = cache.distant.feat.detach() if self.stage6_cache_detach_read else cache.distant.feat
        feat_rigid = feat_rigid.detach() if self.stage6_cache_detach_read else feat_rigid
        return self._render_prior_from_components(
            feat_bg=feat_bg,
            support_bg=cache.bg.support,
            valid_bg=cache.bg.valid,
            feat_distant=feat_distant,
            support_distant=cache.distant.support,
            valid_distant=cache.distant.valid,
            feat_rigid_s=feat_rigid,
            support_rigid_s=support_rigid,
            valid_rigid_s=valid_rigid,
            gaussians_scene=gaussians_scene,
            source_views=source_views,
            height=height,
            width=width,
        )

    def _apply_teacher_prior_adapter(self, prior_map: torch.Tensor, *, layout: Optional[str] = None) -> torch.Tensor:
        if prior_map.dim() != 4:
            raise ValueError(f"Stage6_0 prior_map must be 4D, got {tuple(prior_map.shape)}")
        if layout is None:
            is_nhwc = int(prior_map.shape[-1]) == int(self.stage6_prior_dim)
            is_nchw = int(prior_map.shape[1]) == int(self.stage6_prior_dim)
            if is_nhwc and not is_nchw:
                layout = "nhwc"
            elif is_nchw and not is_nhwc:
                layout = "nchw"
            elif is_nhwc and is_nchw:
                raise ValueError(f"ambiguous teacher prior adapter layout for shape={tuple(prior_map.shape)}")
            else:
                raise ValueError(f"cannot infer teacher prior adapter layout for shape={tuple(prior_map.shape)}")
        if layout not in {"nhwc", "nchw"}:
            raise ValueError(f"unsupported teacher prior adapter layout={layout!r}")
        was_nhwc = layout == "nhwc"
        x = prior_map.permute(0, 3, 1, 2).contiguous() if was_nhwc else prior_map
        y = self.stage6_teacher_prior_adapter(x)
        return y.permute(0, 2, 3, 1).contiguous() if was_nhwc else y

    def _enforce_live_bridge_requirement(self, *, role: str, live_used: bool) -> None:
        if (
            str(role) == "student"
            and bool(getattr(self, "stage6_live_enable", True))
            and bool(getattr(self, "stage6_live_require_on_student", False))
            and not bool(live_used)
        ):
            raise RuntimeError(
                "Stage6_0 live bridge required on student step, "
                "but no teacher live inputs were available."
            )

    def _update_teacher_prior_cache(
        self,
        *,
        cache: TeacherPriorCache,
        feat_bg: torch.Tensor,
        acc_bg: torch.Tensor,
        feat_distant: Optional[torch.Tensor],
        acc_distant: Optional[torch.Tensor],
        feat_rigid_s: Optional[torch.Tensor],
        acc_rigid_s: Optional[torch.Tensor],
        route: RigidRoute,
        global_step: int,
    ) -> None:
        if not self.stage6_cache_detach_write:
            raise ValueError("Stage6_0 cache writes must be detached.")
        update_teacher_prior_cache_detached(
            cache=cache,
            feat_bg=feat_bg,
            acc_bg=acc_bg,
            feat_distant=feat_distant,
            acc_distant=acc_distant,
            feat_rigid=feat_rigid_s,
            acc_rigid=acc_rigid_s,
            rigid_idx=route.S,
            global_step=int(global_step),
            support_min_bg=float(self.stage6_prior_support_min_bg),
            support_min_distant=float(self.stage6_prior_support_min_distant),
            support_min_rigid=float(self.stage6_prior_support_min_rigid),
        )

    def _extract_teacher_live_inputs_from_targets(
        self,
        *,
        batch: Dict[str, Any],
    ) -> Tuple[List[Any], List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        targets = list(batch.get("targets") or [])
        if len(targets) == 0:
            return [], [], None, None
        meta = batch.get("request_meta") or {}
        req_v10 = meta.get("scheduler_request_v10") or {}
        live_req = req_v10.get("live_teacher_bridge") or {}
        refs = [tuple(x) for x in list(live_req.get("image_refs") or [])]
        frame_idx = int(live_req.get("frame_idx", -1))
        selected: List[Dict[str, Any]] = []
        if len(refs) > 0:
            refs_set = {(int(f), int(c)) for f, c in refs}
            for t in targets:
                key = (int(t.get("frame_idx", -1)), int(t.get("cam_idx", -1)))
                if key in refs_set:
                    selected.append(t)
        if len(selected) == 0 and frame_idx >= 0:
            selected = [t for t in targets if int(t.get("frame_idx", -1)) == frame_idx]
        if len(selected) == 0:
            return [], [], None, None
        views = [t["view"] for t in selected]
        images = [t["gt_image"] if t["gt_image"].dim() == 3 else t["gt_image"].squeeze(0) for t in selected]
        sky_masks = [t.get("sky_mask") for t in selected]
        ego_masks = [t.get("egocar_mask") for t in selected]
        if all(x is None for x in sky_masks):
            sky_masks = None
        if all(x is None for x in ego_masks):
            ego_masks = None
        return views, images, sky_masks, ego_masks

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
        batch = self._stage6_active_batch
        if batch is None:
            raise RuntimeError("Stage6_0 internal error: active batch is None during routed feature pass.")
        gaussians_bg_distant, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        num_rigid_s = int(route.S.numel())
        parts_means = [gaussians_bg_distant["means"]]
        parts_scales = [gaussians_bg_distant["scales"]]
        parts_quats = [gaussians_bg_distant["quats"]]
        parts_opacities = [gaussians_bg_distant["opacities"]]
        parts_colors = [gaussians_bg_distant["colors"]]
        if node_state_rigid is not None and num_rigid_s > 0:
            parts_means.append(route.means_world_S)
            parts_quats.append(route.quats_world_S)
            parts_scales.append(torch.exp(node_state_rigid.scales_log[route.S]))
            parts_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[route.S]).squeeze(-1))
            parts_colors.append(torch.cat([node_state_rigid.sh_dc[route.S, None, :], node_state_rigid.sh_rest[route.S]], dim=1))
        gaussians_scene = {
            "means": torch.cat(parts_means, dim=0),
            "scales": torch.cat(parts_scales, dim=0),
            "quats": torch.cat(parts_quats, dim=0),
            "opacities": torch.cat(parts_opacities, dim=0),
            "colors": torch.cat(parts_colors, dim=0),
        }

        scene_rgbs, _ = self.alpha_t_extractor.render_rgb_only(
            gaussians_scene,
            source_views,
            int(height),
            int(width),
            return_acc=True,
            return_debug_stats=False,
        )
        scene_rgb_batch = torch.stack(scene_rgbs, dim=0)
        image_batch = torch.stack([img.to(self.device) for img in source_images], dim=0)
        if image_batch.dim() == 4 and int(image_batch.shape[1]) == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)

        cache = self._get_or_init_teacher_prior_cache(
            batch=batch,
            num_bg=int(node_state_bg.means.shape[0]),
            num_distant=int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
            num_rigid=int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
            feat_dim=int(self.stage6_prior_dim),
            dtype=torch.float32,
        )
        role = self._get_stage6_role(batch=batch, cache=cache, route=route)
        meta = batch.get("request_meta") or {}
        req_v10 = meta.get("scheduler_request_v10") or {}
        prior_conf_map = None
        live_used = False
        cache_fallback = False

        if role == "teacher":
            feat_input = torch.cat([image_batch, scene_rgb_batch.detach()], dim=-1)
            features_2d = self.image_feature_extractor(feat_input)
        else:
            prior_map = None
            if self.stage6_live_enable and self.stage6_live_rerun_teacher_2d:
                live_req = req_v10.get("live_teacher_bridge") or {}
                if bool(live_req.get("enable", False)):
                    tv, ti, tsky, tego = self._extract_teacher_live_inputs_from_targets(batch=batch)
                    if len(tv) > 0:
                        teacher_scene_rgbs, _ = self.alpha_t_extractor.render_rgb_only(
                            gaussians_scene,
                            tv,
                            int(height),
                            int(width),
                            return_acc=True,
                            return_debug_stats=False,
                        )
                        teacher_scene_rgb_batch = torch.stack(teacher_scene_rgbs, dim=0)
                        teacher_img_batch = torch.stack([img.to(self.device) for img in ti], dim=0)
                        if teacher_img_batch.dim() == 4 and int(teacher_img_batch.shape[1]) == 3:
                            teacher_img_batch = teacher_img_batch.permute(0, 2, 3, 1)
                        feat_input_teacher = torch.cat([teacher_img_batch, teacher_scene_rgb_batch.detach()], dim=-1)
                        teacher_feat_2d = self.image_feature_extractor(feat_input_teacher)
                        if not self.stage6_live_to_teacher_backbone:
                            teacher_feat_2d = teacher_feat_2d.detach()
                        teacher_feat_2d = self._apply_teacher_prior_adapter(teacher_feat_2d, layout="nhwc")
                        teacher_pair_valid_mask = self._build_source_pair_valid_mask(
                            source_images=ti,
                            source_sky_masks=tsky,
                            source_egocar_masks=tego,
                        )
                        feat_teacher_all, acc_teacher_all = self._backproject_scene_features_multi_camera(
                            gaussians_scene=gaussians_scene,
                            source_views=tv,
                            features_2d=teacher_feat_2d,
                            source_pair_valid_mask=teacher_pair_valid_mask,
                            height=int(height),
                            width=int(width),
                        )
                        if feat_teacher_all is not None and acc_teacher_all is not None:
                            start = 0
                            feat_teacher_bg = feat_teacher_all[start : start + num_bg]
                            acc_teacher_bg = acc_teacher_all[start : start + num_bg]
                            start += num_bg
                            feat_teacher_dist = feat_teacher_all[start : start + num_distant] if num_distant > 0 else None
                            acc_teacher_dist = acc_teacher_all[start : start + num_distant] if num_distant > 0 else None
                            start += num_distant
                            feat_teacher_rigid_s = (
                                feat_teacher_all[start : start + num_rigid_s] if num_rigid_s > 0 else None
                            )
                            acc_teacher_rigid_s = (
                                acc_teacher_all[start : start + num_rigid_s] if num_rigid_s > 0 else None
                            )
                            valid_bg = acc_teacher_bg > float(self.stage6_prior_support_min_bg)
                            valid_dist = (
                                acc_teacher_dist > float(self.stage6_prior_support_min_distant)
                                if acc_teacher_dist is not None
                                else None
                            )
                            valid_rigid = (
                                acc_teacher_rigid_s > float(self.stage6_prior_support_min_rigid)
                                if acc_teacher_rigid_s is not None
                                else None
                            )
                            prior_map, prior_conf_map = self._render_prior_from_components(
                                feat_bg=feat_teacher_bg,
                                support_bg=acc_teacher_bg,
                                valid_bg=valid_bg,
                                feat_distant=feat_teacher_dist,
                                support_distant=acc_teacher_dist,
                                valid_distant=valid_dist,
                                feat_rigid_s=feat_teacher_rigid_s,
                                support_rigid_s=acc_teacher_rigid_s,
                                valid_rigid_s=valid_rigid,
                                gaussians_scene=gaussians_scene,
                                source_views=source_views,
                                height=int(height),
                                width=int(width),
                            )
                            live_used = True
            if prior_map is None:
                self._enforce_live_bridge_requirement(role=role, live_used=live_used)
                prior_map, prior_conf_map = self._render_teacher_prior_to_source_view(
                    cache=cache,
                    route=route,
                    gaussians_scene=gaussians_scene,
                    source_views=source_views,
                    height=int(height),
                    width=int(width),
                )
                cache_fallback = True

            source_pair_valid_mask = self._build_source_pair_valid_mask(
                source_images=source_images,
                source_sky_masks=source_sky_masks,
                source_egocar_masks=source_egocar_masks,
            )
            valid_mask = source_pair_valid_mask.unsqueeze(-1)
            if self.stage6_student_apply_mask_before_unet:
                scene_rgb_masked = scene_rgb_batch.detach() * valid_mask
                prior_map_masked = prior_map * valid_mask
                prior_conf_masked = prior_conf_map * valid_mask if prior_conf_map is not None else None
                extra_inputs = valid_mask if self.stage6_student_append_valid_mask else None
            else:
                scene_rgb_masked = scene_rgb_batch.detach()
                prior_map_masked = prior_map
                prior_conf_masked = prior_conf_map
                extra_inputs = valid_mask if self.stage6_student_append_valid_mask else None

            features_2d = self.student_prior_fusion_unet(
                render_rgb=scene_rgb_masked,
                prior_map=prior_map_masked,
                prior_conf=prior_conf_masked,
                extra_inputs=extra_inputs,
            )
            if self.stage6_student_zero_invalid_output:
                mask_feat = valid_mask
                if features_2d.shape[-2] != mask_feat.shape[-2] or features_2d.shape[-1] != mask_feat.shape[-1]:
                    mask_feat = F.interpolate(mask_feat.permute(0, 3, 1, 2), size=features_2d.shape[1:3], mode="nearest")
                    mask_feat = mask_feat.permute(0, 2, 3, 1).contiguous()
                features_2d = features_2d * mask_feat

        source_pair_valid_mask = self._build_source_pair_valid_mask(
            source_images=source_images,
            source_sky_masks=source_sky_masks,
            source_egocar_masks=source_egocar_masks,
        )
        feat_2d_all, acc_w_all = self._backproject_scene_features_multi_camera(
            gaussians_scene=gaussians_scene,
            source_views=source_views,
            features_2d=features_2d,
            source_pair_valid_mask=source_pair_valid_mask,
            height=int(height),
            width=int(width),
        )
        if feat_2d_all is None or acc_w_all is None:
            raise ValueError("Stage6_0 source backprojection returned None.")
        start = 0
        feat_2d_bg = feat_2d_all[start : start + num_bg]
        acc_w_bg = acc_w_all[start : start + num_bg]
        start += num_bg
        feat_2d_distant = feat_2d_all[start : start + num_distant] if num_distant > 0 else None
        acc_w_distant = acc_w_all[start : start + num_distant] if num_distant > 0 else None
        start += num_distant
        feat_2d_rigid_s = feat_2d_all[start : start + num_rigid_s] if num_rigid_s > 0 else None
        acc_w_rigid_s = acc_w_all[start : start + num_rigid_s] if num_rigid_s > 0 else None

        if role == "teacher":
            self._update_teacher_prior_cache(
                cache=cache,
                feat_bg=feat_2d_bg,
                acc_bg=acc_w_bg,
                feat_distant=feat_2d_distant,
                acc_distant=acc_w_distant,
                feat_rigid_s=feat_2d_rigid_s,
                acc_rigid_s=acc_w_rigid_s,
                route=route,
                global_step=self._current_global_step(batch),
            )
        self._stage6_last_role = role
        self._stage6_last_prior_conf_map = prior_conf_map
        if prior_conf_map is not None and prior_conf_map.numel() > 0:
            self._perf_acc["bridge/live/prior_conf_mean"] = float(prior_conf_map.float().mean().item()) if live_used else 0.0
            self._perf_acc["bridge/live/prior_conf_nonzero_ratio"] = (
                float((prior_conf_map > 1e-6).float().mean().item()) if live_used else 0.0
            )
            self._perf_acc["bridge/cache/prior_conf_mean"] = float(prior_conf_map.float().mean().item()) if cache_fallback else 0.0
            self._perf_acc["bridge/cache/prior_conf_nonzero_ratio"] = (
                float((prior_conf_map > 1e-6).float().mean().item()) if cache_fallback else 0.0
            )
        self._perf_acc["bridge/cache/fallback_ratio"] = 1.0 if cache_fallback else 0.0
        self._perf_acc["bridge/live/enabled"] = 1.0 if live_used else 0.0
        self._perf_acc["stage6_0/role_teacher"] = 1.0 if role == "teacher" else 0.0
        self._perf_acc["stage6_0/role_student"] = 1.0 if role == "student" else 0.0
        self._perf_acc["stage6_0/fallback_to_teacher_runtime"] = 1.0 if self._stage6_role_fallback else 0.0
        return {
            "num_bg": num_bg,
            "num_distant": num_distant,
            "feat_2d_bg": feat_2d_bg,
            "feat_2d_distant": feat_2d_distant,
            "feat_2d_rigid_S": feat_2d_rigid_s,
            "acc_w_bg": acc_w_bg,
            "acc_w_distant": acc_w_distant,
            "acc_w_rigid_S": acc_w_rigid_s,
            "src_backproject_pass_count": 1,
        }

    def _target_role_weight(self, role: str, step: int) -> float:
        if role in {"teacher_source", "teacher_self"}:
            return float(self._scheduler_v10_target_weights["teacher_source"])
        if role in {"student_source", "student_self"}:
            return float(self._scheduler_v10_target_weights["student_source"])
        if role == "teacher_anchor":
            return float(self._scheduler_v10_target_weights["teacher_anchor"])
        if role == "history_visited":
            return float(self._scheduler_v10_target_weights["history_visited"])
        if role == "probe_near":
            return float(self._scheduler_v10_target_weights["probe_near"])
        return super()._target_role_weight(role=role, step=step)

    def _build_target_view_weights(
        self,
        batch: Dict[str, Any],
        *,
        step: int,
        num_targets: int,
    ) -> Tuple[torch.Tensor, List[str]]:
        meta = batch.get("request_meta") or {}
        scheduler_version = str(meta.get("scheduler_version", "")).strip().lower()
        if scheduler_version in {"v10", "v9"}:
            weights = [float(x) for x in list(meta.get("train_target_image_loss_base_weights") or meta.get("target_image_loss_base_weights") or [])]
            roles = [str(x) for x in list(meta.get("train_target_image_roles") or meta.get("target_image_roles") or [])]
            refs = list(meta.get("train_target_image_refs") or meta.get("target_image_refs") or [])
            if len(weights) == int(num_targets) and len(roles) == int(num_targets) and len(refs) == int(num_targets):
                return torch.tensor(weights, dtype=torch.float32, device=self.device), roles
        return super()._build_target_view_weights(batch=batch, step=int(step), num_targets=int(num_targets))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._stage6_active_batch = batch
        try:
            out = super().forward(batch=batch)
        finally:
            self._stage6_active_batch = None
        meta = batch.get("request_meta") or {}
        requested_role = str(meta.get("stage6_role", meta.get("stage5_5_role", "teacher")))
        actual_role = str(getattr(self, "_stage6_last_role", requested_role))
        out["stage6_0/role_requested_teacher"] = float(requested_role == "teacher")
        out["stage6_0/role_requested_student"] = float(requested_role == "student")
        out["stage6_0/role_actual_teacher"] = float(actual_role == "teacher")
        out["stage6_0/role_actual_student"] = float(actual_role == "student")
        out["stage6_0/role_fallback"] = float(bool(self._stage6_role_fallback))
        out["scheduler/v10_is_compat_v9"] = float(1.0 if str(meta.get("scheduler_version", "")) == "v10" else 0.0)
        probe_refs = list(meta.get("probe_target_image_refs") or [])
        target_roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
        target_weights = [float(x) for x in list(meta.get("target_image_loss_base_weights") or [])]
        probe_weight_sum = sum(
            float(w) for role, w in zip(target_roles, target_weights) if str(role) == "probe_near"
        )
        out["probe/near/num_targets"] = float(len(probe_refs))
        out["loss/probe_near_weight_sum"] = float(probe_weight_sum)
        return out

    @staticmethod
    def _pop_first(d: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            if key in d:
                return d.pop(key)
        return None

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
        anchor_l1 = self._pop_first(out, ["monitor/l1/teacher_anchor", "monitor/l1/teacher_preserve"])
        if anchor_l1 is not None:
            out["monitor/l1/teacher_anchor"] = float(anchor_l1)
            out["loss/teacher_anchor"] = float(anchor_l1)
        anchor_psnr = self._pop_first(out, ["monitor/psnr/teacher_anchor", "monitor/psnr/teacher_preserve"])
        if anchor_psnr is not None:
            out["monitor/psnr/teacher_anchor"] = float(anchor_psnr)
        near_l1 = self._pop_first(out, ["monitor/l1/probe_near", "monitor/l1/near_random"])
        if near_l1 is not None:
            out["probe/near/l1"] = float(near_l1)
        near_psnr = self._pop_first(out, ["monitor/psnr/probe_near", "monitor/psnr/near_random"])
        if near_psnr is not None:
            out["probe/near/psnr"] = float(near_psnr)
        if "loss" in out and torch.is_tensor(out["loss"]):
            out["loss/total_train"] = float(out["loss"].detach().item())
        return out

    def reset_node_state(self, clear_caches: bool = True):
        out = super().reset_node_state(clear_caches=clear_caches)
        self._stage6_teacher_prior_caches.clear()
        return out
