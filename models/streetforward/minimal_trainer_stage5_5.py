from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors import StudentPriorFusionUNet
from models.streetforward.minimal_trainer_stage4_3 import RuntimePolicy
from models.streetforward.math_utils import get_viewmat
from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.teacher_student_prior import TeacherPriorCache, create_teacher_prior_cache

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage5_5(MinimalStreetForwardStage5_4):
    def __init__(self, config, device: torch.device, **kwargs):
        self._stage5_5_active_batch: Optional[Dict[str, Any]] = None
        self._stage5_5_teacher_prior_caches: Dict[Tuple[int, int], TeacherPriorCache] = {}
        self._stage5_5_last_role: str = "teacher"
        self._stage5_5_role_fallback: bool = False
        self._stage5_5_last_prior_conf_map: Optional[torch.Tensor] = None
        self._stage5_5_observed_teacher_exit_count: int = 0
        super().__init__(config=config, device=device, **kwargs)
        self._maybe_load_stage5_4_init_checkpoint(config=config)
        sv9 = self._cfg_get(config, "scheduler_v9", {})
        targets = self._cfg_get(sv9, "targets", {})
        weights = self._cfg_get(targets, "weights", {})
        required_w = ("teacher_source", "student_source", "teacher_preserve", "visited", "near_random")
        missing_w = [k for k in required_w if weights.get(k) is None]
        if missing_w:
            raise ValueError(
                "Stage5_5 requires scheduler_v9.targets.weights entries "
                f"{list(required_w)}; missing {missing_w}."
            )
        self._scheduler_v9_target_weights = {k: float(weights[k]) for k in required_w}

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

    @staticmethod
    def _as_str_list(value: Any, *, default: List[str]) -> List[str]:
        if value is None:
            return [str(x) for x in default]
        if isinstance(value, str):
            return [str(value)]
        if isinstance(value, (list, tuple)):
            return [str(x) for x in list(value)]
        try:
            return [str(x) for x in list(value)]
        except Exception:
            return [str(x) for x in default]

    @staticmethod
    def _is_key_allowed_by_prefixes(key: str, prefixes: List[str]) -> bool:
        k = str(key)
        for p in prefixes:
            if k.startswith(str(p)):
                return True
        return False

    def _maybe_load_stage5_4_init_checkpoint(self, *, config: Any) -> None:
        stage5_5_cfg = self._cfg_get(config, "stage5_5", None)
        init_cfg = self._cfg_get(stage5_5_cfg, "init", None)
        if init_cfg is None:
            return
        use_init = bool(self._cfg_get(init_cfg, "from_stage5_4_checkpoint", False))
        if not use_init:
            return
        ckpt_path = str(self._cfg_get(init_cfg, "checkpoint_path", "")).strip()
        if not ckpt_path:
            raise ValueError(
                "stage5_5.init.from_stage5_4_checkpoint=true requires a non-empty stage5_5.init.checkpoint_path."
            )
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"stage5_5 init checkpoint not found: {ckpt_path}")

        strict = bool(self._cfg_get(init_cfg, "strict", False))
        missing_allow_prefixes = self._as_str_list(
            self._cfg_get(init_cfg, "missing_key_prefix_allowlist", ["student_prior_fusion_unet."]),
            default=["student_prior_fusion_unet."],
        )
        unexpected_allow_prefixes = self._as_str_list(
            self._cfg_get(init_cfg, "unexpected_key_prefix_allowlist", []),
            default=[],
        )
        reset_teacher_prior_cache = bool(self._cfg_get(init_cfg, "reset_teacher_prior_cache", True))

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_state = None
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get("model_state_dict")
            if model_state is None:
                model_state = checkpoint.get("state_dict")
            if model_state is None:
                model_state = checkpoint.get("model")
            if model_state is None and all(torch.is_tensor(v) for v in checkpoint.values()):
                model_state = checkpoint
        if model_state is None:
            raise ValueError(
                "Stage5_5 init checkpoint must contain model_state_dict/state_dict/model or be a raw state_dict."
            )

        if strict:
            self.load_state_dict(model_state, strict=True)
        else:
            incompatible = self.load_state_dict(model_state, strict=False)
            missing_keys = [str(k) for k in list(getattr(incompatible, "missing_keys", []))]
            unexpected_keys = [str(k) for k in list(getattr(incompatible, "unexpected_keys", []))]
            bad_missing = [
                k
                for k in missing_keys
                if not self._is_key_allowed_by_prefixes(k, missing_allow_prefixes)
            ]
            bad_unexpected = [
                k
                for k in unexpected_keys
                if not self._is_key_allowed_by_prefixes(k, unexpected_allow_prefixes)
            ]
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    "Stage5_5 init checkpoint key mismatch after non-strict load. "
                    f"bad_missing={bad_missing[:20]} bad_unexpected={bad_unexpected[:20]} "
                    f"(allow missing prefixes={missing_allow_prefixes}, "
                    f"allow unexpected prefixes={unexpected_allow_prefixes})."
                )
        if reset_teacher_prior_cache:
            self._stage5_5_teacher_prior_caches.clear()
        logger.info(
            "Loaded Stage5_4 init checkpoint for Stage5_5 from %s (strict=%s, reset_teacher_prior_cache=%s).",
            ckpt_path,
            bool(strict),
            bool(reset_teacher_prior_cache),
        )

    def _validate_stage5_3_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model"))
        if stage != "5_5":
            raise ValueError("Stage5_5 requires model.stage='5_5'.")

        old_stage = model_cfg.get("stage")
        model_cfg["stage"] = "5_4"
        try:
            super()._validate_stage5_3_config(config)
        finally:
            model_cfg["stage"] = old_stage
        self._validate_stage5_5_config(config)

    def _validate_stage5_5_config(self, config) -> None:
        scheduler_v9 = self._require_key(config, "scheduler_v9", "config")
        if bool(self._require_key(scheduler_v9, "enable", "scheduler_v9")) is not True:
            raise ValueError("Stage5_5 requires scheduler_v9.enable=true.")
        stage5_5_cfg = self._require_key(config, "stage5_5", "config")
        if bool(self._require_key(stage5_5_cfg, "enable", "stage5_5")) is not True:
            raise ValueError("Stage5_5 requires stage5_5.enable=true.")
        hist5 = stage5_5_cfg.get("history_record")
        if hist5 is not None and str(hist5.get("policy", "")).strip().lower() == "teacher_only":
            raise ValueError(
                "Stage5_5 no longer supports history_record.policy=teacher_only. "
                "Use scheduler_v9.history_record.observed/runtime split."
            )
        prior_cfg = self._require_key(stage5_5_cfg, "teacher_prior", "stage5_5")
        prior_dim = int(self._require_key(prior_cfg, "dim", "stage5_5.teacher_prior"))
        if prior_dim <= 0:
            raise ValueError("stage5_5.teacher_prior.dim must be > 0.")
        splat_cfg = self._require_key(stage5_5_cfg, "feature_splatting", "stage5_5")
        feature_dim = int(self._require_key(splat_cfg, "feature_dim", "stage5_5.feature_splatting"))
        if feature_dim != prior_dim:
            raise ValueError("stage5_5.feature_splatting.feature_dim must match stage5_5.teacher_prior.dim.")
        packed_channels = int(self._require_key(splat_cfg, "packed_channels", "stage5_5.feature_splatting"))
        if feature_dim + 1 != packed_channels:
            raise ValueError("stage5_5.feature_splatting.packed_channels must equal feature_dim + 1.")
        if packed_channels not in {33, 65, 129, 257, 513}:
            raise ValueError("Packed feature channels must be supported by gsplat.")

    def _init_stage5_3_modules(self, config) -> None:
        super()._init_stage5_3_modules(config)
        stage5_5_cfg = self._require_key(config, "stage5_5", "config")
        prior_cfg = self._require_key(stage5_5_cfg, "teacher_prior", "stage5_5")
        splat_cfg = self._require_key(stage5_5_cfg, "feature_splatting", "stage5_5")
        student_cfg = self._require_key(stage5_5_cfg, "student_extractor", "stage5_5")

        self.stage5_5_cfg = stage5_5_cfg
        self.stage5_5_prior_dim = int(self._require_key(prior_cfg, "dim", "stage5_5.teacher_prior"))
        self.stage5_5_prior_eps = float(self._require_key(prior_cfg, "eps", "stage5_5.teacher_prior"))
        self.stage5_5_prior_conf_norm = float(
            self._require_key(
                self._require_key(prior_cfg, "confidence", "stage5_5.teacher_prior"),
                "norm",
                "stage5_5.teacher_prior.confidence",
            )
        )
        feat_dim = int(self.stage5_2_feat_2d_channels)
        if int(self.stage5_5_prior_dim) != feat_dim:
            raise ValueError(
                f"stage5_5.teacher_prior.dim must equal stage5_2_feat_2d_channels={feat_dim}, "
                f"got {self.stage5_5_prior_dim}."
            )
        self.stage5_5_prior_support_min_bg = float(prior_cfg.get("support_min_bg", self.bg_src_backproject_support_min))
        self.stage5_5_prior_support_min_distant = float(
            prior_cfg.get("support_min_distant", self.distant_src_backproject_support_min)
        )
        self.stage5_5_prior_support_min_rigid = float(
            prior_cfg.get("support_min_rigid", self.rigid_src_backproject_support_min)
        )
        self.stage5_5_fallback_to_teacher = bool(
            self._require_key(stage5_5_cfg, "fallback_to_teacher_if_no_prior", "stage5_5")
        )
        self.stage5_5_feature_splat_enabled = bool(self._require_key(splat_cfg, "enable", "stage5_5.feature_splatting"))
        self.stage5_5_student_use_conf = bool(student_cfg.get("use_confidence", True))
        self.stage5_5_teacher_use_gt = bool(self._require_key(self._require_key(stage5_5_cfg, "teacher", "stage5_5"), "input_gt", "stage5_5.teacher"))
        if not self.stage5_5_teacher_use_gt:
            raise ValueError("Stage5_5 requires stage5_5.teacher.input_gt=true.")

        self.student_prior_fusion_unet = StudentPriorFusionUNet(
            prior_dim=self.stage5_5_prior_dim,
            out_dim=int(self.stage5_2_feat_2d_channels),
            base_dim=int(self._require_key(student_cfg, "base_channels", "stage5_5.student_extractor")),
            use_confidence=self.stage5_5_student_use_conf,
        ).to(self.device)

    def _current_global_step(self, batch: Dict[str, Any]) -> int:
        opt = getattr(self, "optimizer", None)
        if opt is not None and hasattr(opt, "global_step"):
            return int(getattr(opt, "global_step"))
        aligned = batch.get("_scheduler_v9_aligned_info") or batch.get("_scheduler_v8_aligned_info") or {}
        return int(aligned.get("global_step", 0))

    def _get_stage5_5_role(
        self,
        *,
        batch: Dict[str, Any],
        cache: Optional[TeacherPriorCache],
        route: RigidRoute,
    ) -> str:
        meta = batch.get("request_meta") or {}
        requested_role = str(meta.get("stage5_5_role", "teacher")).strip().lower()
        role = str(requested_role)
        if role not in ("teacher", "student"):
            raise ValueError(f"Unknown Stage5_5 role: {role}")
        if role == "student" and not self._teacher_prior_available(cache=cache, route=route):
            if self.stage5_5_fallback_to_teacher:
                role = "teacher"
            else:
                raise RuntimeError("Student step requested but teacher prior is unavailable.")
        self._stage5_5_role_fallback = bool(requested_role == "student" and role == "teacher")
        return role

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

    def _get_or_init_teacher_prior_cache(
        self,
        *,
        batch: Dict[str, Any],
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
    ) -> TeacherPriorCache:
        key = self._batch_key(batch)
        cache = self._stage5_5_teacher_prior_caches.get(key)
        num_bg = int(node_state_bg.means.shape[0])
        num_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        if cache is None:
            cache = create_teacher_prior_cache(
                num_bg=num_bg,
                num_distant=num_distant,
                num_rigid=num_rigid,
                feat_dim=self.stage5_5_prior_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self._stage5_5_teacher_prior_caches[key] = cache
            return cache
        if (
            int(cache.bg.feat.shape[0]) != num_bg
            or int(cache.distant.feat.shape[0]) != num_distant
            or int(cache.rigid.feat.shape[0]) != num_rigid
        ):
            cache = create_teacher_prior_cache(
                num_bg=num_bg,
                num_distant=num_distant,
                num_rigid=num_rigid,
                feat_dim=self.stage5_5_prior_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self._stage5_5_teacher_prior_caches[key] = cache
        return cache

    def _build_prior_confidence(self, *, support: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        conf = torch.log1p(support.float()) / float(self.stage5_5_prior_conf_norm)
        conf = conf.clamp(0.0, 1.0)
        conf = conf * valid.float()
        return conf.unsqueeze(-1)

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
        prior_feat_all = torch.cat(
            [cache.bg.feat, cache.distant.feat, cache.rigid.feat[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.feat[:0]],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        prior_support = torch.cat(
            [
                cache.bg.support,
                cache.distant.support,
                cache.rigid.support[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.support[:0],
            ],
            dim=0,
        ).to(device=self.device)
        prior_valid = torch.cat(
            [cache.bg.valid, cache.distant.valid, cache.rigid.valid[rigid_idx] if int(rigid_idx.numel()) > 0 else cache.rigid.valid[:0]],
            dim=0,
        ).to(device=self.device)

        conf = self._build_prior_confidence(support=prior_support, valid=prior_valid)
        packed = torch.cat([prior_feat_all * conf, conf], dim=-1)
        render_params = {
            "means": gaussians_scene["means"],
            "quats": gaussians_scene["quats"],
            "scales": gaussians_scene["scales"],
            "opacities": gaussians_scene["opacities"],
            "colors": packed,
        }
        viewmats_list: List[torch.Tensor] = []
        for i, v in enumerate(source_views):
            cam_ctw = v.camtoworlds if hasattr(v, "camtoworlds") else v["camtoworlds"]
            vm = get_viewmat(cam_ctw)
            if vm.dim() == 2:
                vm = vm.unsqueeze(0)
            if vm.dim() != 3 or tuple(vm.shape[-2:]) != (4, 4):
                raise ValueError(
                    "Stage5_5 expected get_viewmat(...) to return [B,4,4] or [4,4], "
                    f"got shape={tuple(vm.shape)} at source_view[{i}]"
                )
            if int(vm.shape[0]) != 1:
                raise ValueError(
                    "Stage5_5 expects one camera pose per source view, "
                    f"got batch={int(vm.shape[0])} at source_view[{i}]"
                )
            viewmats_list.append(vm)
        viewmats = torch.cat(viewmats_list, dim=0)
        ks_list = []
        for v in source_views:
            if hasattr(v, "Ks"):
                km = v.Ks[0]
            elif hasattr(v, "K"):
                km = v.K
                if km.dim() == 3:
                    km = km[0]
            else:
                raise ValueError("Stage5_5 source view is missing intrinsic matrix K/Ks.")
            ks_list.append(km.to(device=self.device))
        Ks = torch.stack(ks_list, dim=0)
        rendered, _, _ = self.renderer(
            means=render_params["means"],
            quats=render_params["quats"],
            scales=render_params["scales"],
            opacities=render_params["opacities"],
            colors=render_params["colors"],
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
        feat_sum = rendered[..., : self.stage5_5_prior_dim]
        conf_map = rendered[..., self.stage5_5_prior_dim : self.stage5_5_prior_dim + 1]
        prior_map = feat_sum / (conf_map + float(self.stage5_5_prior_eps))
        return prior_map, conf_map

    def _stage5_5_update_teacher_prior_cache(
        self,
        *,
        cache: TeacherPriorCache,
        feat_bg: torch.Tensor,
        acc_bg: torch.Tensor,
        feat_distant: Optional[torch.Tensor],
        acc_distant: Optional[torch.Tensor],
        feat_rigid: Optional[torch.Tensor],
        acc_rigid: Optional[torch.Tensor],
        route: RigidRoute,
        global_step: int,
    ) -> None:
        bg_valid = acc_bg > float(self.stage5_5_prior_support_min_bg)
        cache.bg.feat[bg_valid] = feat_bg[bg_valid].detach().float()
        cache.bg.support[bg_valid] = acc_bg[bg_valid].detach().float()
        cache.bg.valid[bg_valid] = True
        cache.bg.last_update_step[bg_valid] = int(global_step)
        if feat_distant is not None and acc_distant is not None and int(feat_distant.shape[0]) > 0:
            distant_valid = acc_distant > float(self.stage5_5_prior_support_min_distant)
            cache.distant.feat[distant_valid] = feat_distant[distant_valid].detach().float()
            cache.distant.support[distant_valid] = acc_distant[distant_valid].detach().float()
            cache.distant.valid[distant_valid] = True
            cache.distant.last_update_step[distant_valid] = int(global_step)
        if feat_rigid is not None and acc_rigid is not None and int(feat_rigid.shape[0]) > 0:
            rigid_valid_s = acc_rigid > float(self.stage5_5_prior_support_min_rigid)
            rigid_local_idx = route.S[rigid_valid_s]
            cache.rigid.feat[rigid_local_idx] = feat_rigid[rigid_valid_s].detach().float()
            cache.rigid.support[rigid_local_idx] = acc_rigid[rigid_valid_s].detach().float()
            cache.rigid.valid[rigid_local_idx] = True
            cache.rigid.last_update_step[rigid_local_idx] = int(global_step)

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
        batch = self._stage5_5_active_batch
        if batch is None:
            raise RuntimeError("Stage5_5 internal error: active batch is None during routed feature pass.")
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
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
        )
        role = self._get_stage5_5_role(batch=batch, cache=cache, route=route)
        prior_conf_map = None
        if role == "teacher":
            feat_input = torch.cat([image_batch, scene_rgb_batch.detach()], dim=-1)
            features_2d = self.image_feature_extractor(feat_input)
        else:
            if not self.stage5_5_feature_splat_enabled:
                raise ValueError("Stage5_5 student role requires stage5_5.feature_splatting.enable=true.")
            prior_map, prior_conf_map = self._render_teacher_prior_to_source_view(
                cache=cache,
                route=route,
                gaussians_scene=gaussians_scene,
                source_views=source_views,
                height=int(height),
                width=int(width),
            )
            features_2d = self.student_prior_fusion_unet(
                render_rgb=scene_rgb_batch,
                prior_map=prior_map,
                prior_conf=prior_conf_map if self.stage5_5_student_use_conf else None,
            )

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
            raise ValueError("Stage5_5 source backprojection returned None.")
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
            self._stage5_5_update_teacher_prior_cache(
                cache=cache,
                feat_bg=feat_2d_bg,
                acc_bg=acc_w_bg,
                feat_distant=feat_2d_distant,
                acc_distant=acc_w_distant,
                feat_rigid=feat_2d_rigid_s,
                acc_rigid=acc_w_rigid_s,
                route=route,
                global_step=self._current_global_step(batch),
            )
        self._stage5_5_last_role = role
        self._stage5_5_last_prior_conf_map = prior_conf_map
        if prior_conf_map is not None and prior_conf_map.numel() > 0:
            self._perf_acc["stage5_5_prior_conf_mean"] = float(prior_conf_map.float().mean().item())
            self._perf_acc["stage5_5_prior_conf_nonzero_ratio"] = float((prior_conf_map > 1e-6).float().mean().item())
        else:
            self._perf_acc["stage5_5_prior_conf_mean"] = 0.0
            self._perf_acc["stage5_5_prior_conf_nonzero_ratio"] = 0.0
        self._perf_acc["stage5_5_fallback_to_teacher_runtime"] = 1.0 if self._stage5_5_role_fallback else 0.0
        self._perf_acc["stage5_5_role_teacher"] = 1.0 if role == "teacher" else 0.0
        self._perf_acc["stage5_5_role_student"] = 1.0 if role == "student" else 0.0
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
        r = str(role)
        if r in ("teacher_source", "student_source"):
            return float(self._target_view_weight_cfg["source_weight"])
        if r == "teacher_preserve":
            return float(self._scheduler_v9_target_weights["teacher_preserve"])
        if r == "visited":
            return float(self._scheduler_v9_target_weights["visited"])
        if r == "near_random":
            return float(self._scheduler_v9_target_weights["near_random"])
        return super()._target_role_weight(role, step)

    def _build_target_view_weights(
        self,
        batch: Dict[str, Any],
        *,
        step: int,
        num_targets: int,
    ) -> Tuple[torch.Tensor, List[str]]:
        meta = batch.get("request_meta") or {}
        if str(meta.get("scheduler_version", "")) == "v9":
            w = meta.get("target_image_loss_base_weights")
            roles = [str(x) for x in list(meta.get("target_image_roles") or [])]
            if isinstance(w, (list, tuple)) and len(w) == int(num_targets) and len(roles) == int(num_targets):
                return (
                    torch.tensor([float(x) for x in w], dtype=torch.float32, device=self.device),
                    roles,
                )
        return super()._build_target_view_weights(batch, step=step, num_targets=num_targets)

    def _record_observed_history_teacher_exit(self, *, batch: Dict[str, Any], out: Dict[str, Any]) -> None:
        key_any = out.get("_cache_key")
        if key_any is None:
            return
        key = (int(key_any[0]), int(key_any[1]))
        node_state_bg = out.get("_node_state_bg")
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")
        if node_state_bg is None:
            return
        hist_bg = self._get_or_init_history(self.stage5_2_history_bg, key, int(node_state_bg.means.shape[0]))
        hist_dist = self._get_or_init_history(
            self.stage5_2_history_distant,
            key,
            int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
        )
        hist_rigid = self._get_or_init_history(
            self.stage5_2_history_rigid,
            key,
            int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        )
        meta = batch.get("request_meta") or {}
        record_frame_idx = int(meta.get("history_record/observed_record_frame_idx", -1))
        record_image_refs = [tuple(x) for x in list(meta.get("history_record/observed_record_image_refs", []))]
        if record_frame_idx < 0 or len(record_image_refs) == 0:
            raise RuntimeError(
                "Teacher-exit observed record requires observed_record_frame_idx/image_refs."
            )
        record_batch = dict(batch)
        record_batch["source_frame_idx"] = int(record_frame_idx)
        record_meta = dict(meta)
        record_meta["source_image_refs"] = [tuple(x) for x in record_image_refs]
        record_batch["request_meta"] = record_meta
        record_targets = self._build_record_targets(record_batch)
        source_frame_idx = int(record_frame_idx)
        n_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        mask_src_rigid = torch.zeros(n_rigid, dtype=torch.bool, device=self.device)
        if node_state_rigid is not None:
            mask_src_rigid = self._rigid_point_valid_mask(node_state_rigid, source_frame_idx)
        s = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
        route = (
            self._route_rigid_source_points(node_state_rigid, source_frame_idx, s)
            if node_state_rigid is not None
            else RigidRoute(
                S=s,
                S_in=s,
                S_out=s,
                inside_mask_S=torch.zeros((0,), dtype=torch.bool, device=self.device),
                route_inside_global=torch.zeros((n_rigid,), dtype=torch.bool, device=self.device),
                means_world_S=torch.zeros((0, 3), device=self.device),
                quats_world_S=torch.zeros((0, 4), device=self.device),
            )
        )
        rec = self._compute_record_support_error_all_branches_once_routed(
            batch=record_batch,
            source_frame_idx=source_frame_idx,
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            route=route,
            record_targets=record_targets,
        )
        self._apply_residual_history_update(
            history=hist_bg,
            error_cur=rec["error_bg"],
            visible_mask=(rec["support_bg"].squeeze(-1) > self.stage5_2_bg_visible_min),
        )
        self._apply_residual_history_update(
            history=hist_dist,
            error_cur=rec["error_distant"],
            visible_mask=(rec["support_distant"].squeeze(-1) > self.stage5_2_distant_visible_min),
        )
        self._apply_residual_history_update(
            history=hist_rigid,
            error_cur=rec["error_rigid"],
            visible_mask=(rec["support_rigid"].squeeze(-1) > self.stage5_2_rigid_visible_min),
        )
        self._stage5_5_observed_teacher_exit_count = int(self._stage5_5_observed_teacher_exit_count) + 1

    def forward(self, batch: Dict) -> Dict[str, Any]:
        self._stage5_5_active_batch = batch
        try:
            out = super().forward(batch)
        finally:
            self._stage5_5_active_batch = None
        meta = batch.get("request_meta") or {}
        if bool(meta.get("history_record/record_observed_on_step_exit", False)) and self._stage5_5_last_role == "teacher":
            self._record_observed_history_teacher_exit(batch=batch, out=out)
        if self._stage5_5_last_prior_conf_map is not None:
            out["stage5_5_prior_conf_mean"] = float(self._stage5_5_last_prior_conf_map.float().mean().item())
            out["stage5_5_prior_conf_nonzero_ratio"] = float(
                (self._stage5_5_last_prior_conf_map > 1e-6).float().mean().item()
            )
        else:
            out["stage5_5_prior_conf_mean"] = 0.0
            out["stage5_5_prior_conf_nonzero_ratio"] = 0.0
        out["stage5_5_role"] = self._stage5_5_last_role
        out["stage5_5_role_fallback"] = bool(self._stage5_5_role_fallback)
        out["history/observed_teacher_exit_count"] = float(self._stage5_5_observed_teacher_exit_count)
        return out

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[RuntimePolicy] = None,
    ) -> Dict[str, Any]:
        _ = runtime_policy
        out = super().train_step(
            batch=batch,
            step=step,
            profile_phase_timing=profile_phase_timing,
            sync_cuda_timing=sync_cuda_timing,
            scheduler_node_sync=scheduler_node_sync,
        )
        meta = batch.get("request_meta") or {}
        if str(meta.get("scheduler_version", "")) == "v9":
            for k, v in meta.items():
                sk = str(k)
                if not sk.startswith("scheduler_v9/"):
                    continue
                if isinstance(v, bool):
                    out[sk] = float(v)
                elif isinstance(v, (int, float)):
                    out[sk] = float(v)
            roles = [str(x) for x in list(meta.get("target_frame_roles") or [])]
            out["scheduler_v9/target_count_teacher_source"] = float(sum(1 for r in roles if r == "teacher_source"))
            out["scheduler_v9/target_count_student_source"] = float(sum(1 for r in roles if r == "student_source"))
            out["scheduler_v9/target_count_teacher_preserve"] = float(sum(1 for r in roles if r == "teacher_preserve"))
            out["scheduler_v9/target_count_visited"] = float(sum(1 for r in roles if r == "visited"))
            out["scheduler_v9/target_count_near_random"] = float(sum(1 for r in roles if r == "near_random"))
            ent_step = meta.get("stage5_5_block_entry_step", meta.get("scheduler_v9/block_entry_step"))
            if ent_step is not None:
                out["scheduler_v9/block_entry_step"] = float(ent_step)
            ft = meta.get("stage5_5_force_teacher_on_block_entry")
            if ft is not None:
                out["scheduler_v9/force_teacher_on_block_entry_cfg"] = float(1.0 if bool(ft) else 0.0)
        out["history/record_observed_on_step_exit"] = float(
            bool(meta.get("history_record/record_observed_on_step_exit", False))
        )
        out["history/observed_teacher_exit_count"] = float(self._stage5_5_observed_teacher_exit_count)
        out["stage5_5_role"] = str(self._stage5_5_last_role)
        out["stage5_5_role_fallback"] = bool(self._stage5_5_role_fallback)
        if "stage5_5_prior_conf_mean" not in out:
            out["stage5_5_prior_conf_mean"] = 0.0
        if "stage5_5_prior_conf_nonzero_ratio" not in out:
            out["stage5_5_prior_conf_nonzero_ratio"] = 0.0
        return out

    def _commit_block_runtime_support_to_history(
        self,
        *,
        key: Tuple[int, int],
        history_bg: Dict[str, torch.Tensor],
        history_distant: Dict[str, torch.Tensor],
        history_rigid: Dict[str, torch.Tensor],
    ) -> None:
        self._commit_block_support_to_history(
            key=key,
            history_bg=history_bg,
            history_distant=history_distant,
            history_rigid=history_rigid,
        )

    def record_block_history(self, batch: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        _ = event
        # Stage5_5 observed history is recorded on teacher-exit. At block-exit only flush runtime support accumulators.
        key = self._batch_key(batch)
        node_state_bg = self.node_states_bg.get(key)
        node_state_distant = self.node_states_distant.get(key)
        node_state_rigid = self.node_states_rigid.get(key)
        if node_state_bg is None:
            return {}
        hist_bg = self._get_or_init_history(self.stage5_2_history_bg, key, int(node_state_bg.means.shape[0]))
        hist_dist = self._get_or_init_history(
            self.stage5_2_history_distant,
            key,
            int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0,
        )
        hist_rigid = self._get_or_init_history(
            self.stage5_2_history_rigid,
            key,
            int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0,
        )
        self._commit_block_runtime_support_to_history(
            key=key,
            history_bg=hist_bg,
            history_distant=hist_dist,
            history_rigid=hist_rigid,
        )
        self._clear_block_support_acc(key)
        return {"stage5_5_block_exit_runtime_flush": 1.0}

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self._stage5_5_teacher_prior_caches.clear()
        self._stage5_5_role_fallback = False


__all__ = ["MinimalStreetForwardStage5_5"]
