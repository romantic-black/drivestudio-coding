"""
Minimal StreetForward Stage 4.5:
- no sky node
- no sky rendering/composition
- scene-only mask-aware fused multi-camera source 2D backprojection
- photometric loss only on non-sky region
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors import AlphaTWeightExtractorV3
from models.streetforward.math_utils import _num_sh_bases, _sh_to_rgb
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    _backward_to_render_params_bg_rigid_distant,
    _merge_params_bg_rigid_distant,
    merge_debug_stats_as_perf_floats,
    spatial_hw_from_image_tensor,
)
from models.streetforward.minimal_trainer_stage4_3 import RuntimePolicy
from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage4_5(MinimalStreetForwardStage4_2):
    """Stage4.2 + scene-only mask-aware fused-v3 source 2D + no-sky-render + non-sky photometric loss."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        model_cfg = config.model
        self.use_fused_cuda_backproject_v4 = bool(model_cfg.get("use_fused_cuda_backproject_v4", False))
        self.fused_cuda_backproject_v4_force_fallback = bool(
            model_cfg.get("fused_cuda_backproject_v4_force_fallback", False)
        )
        if self.use_fused_cuda_backproject_v4 and not self.fused_cuda_backproject_v4_force_fallback:
            raise ValueError(
                "Stage4.5 does not implement fused_cuda_backproject_v4 yet. "
                "Set model.use_fused_cuda_backproject_v4=false or model.fused_cuda_backproject_v4_force_fallback=true."
            )
        self.use_fused_cuda_backproject_v3 = bool(model_cfg.get("use_fused_cuda_backproject_v3", True))
        if not self.use_fused_cuda_backproject_v3:
            raise ValueError("Stage4.5 requires model.use_fused_cuda_backproject_v3=true.")
        self.alpha_t_extractor_v3 = AlphaTWeightExtractorV3(
            renderer=self.renderer,
            sh_degree=self.sh_degree,
            tile_size=16,
        )

        losses_cfg = config.get("losses") if hasattr(config, "get") else None
        losses_cfg = losses_cfg or {}
        photometric_cfg = losses_cfg.get("photometric", {}) or {}
        self.exclude_sky_region_photometric = bool(photometric_cfg.get("exclude_sky_region", True))
        if not self.exclude_sky_region_photometric:
            raise ValueError("Stage4.5 requires losses.photometric.exclude_sky_region=true.")
        mask_cfg = losses_cfg.get("mask", {}) or {}
        self.require_sky_mask_for_loss = bool(mask_cfg.get("require_sky_mask", True))
        if not self.require_sky_mask_for_loss:
            raise ValueError("Stage4.5 requires losses.mask.require_sky_mask=true.")

    def _build_source_pair_valid_mask(
        self,
        source_images: List[torch.Tensor],
        source_sky_masks: Optional[List[torch.Tensor]],
        source_egocar_masks: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        if len(source_images) < 1:
            raise ValueError("Stage4.5 requires non-empty source_images.")
        valid_masks: List[torch.Tensor] = []
        num_views = len(source_images)
        if source_egocar_masks is not None and len(source_egocar_masks) != num_views:
            raise ValueError(
                "Stage4.5 requires source_egocar_masks length to exactly match source_images length. "
                f"Got len(source_egocar_masks)={len(source_egocar_masks)} vs len(source_images)={num_views}."
            )
        if source_sky_masks is not None and len(source_sky_masks) != num_views:
            raise ValueError(
                "Stage4.5 requires source_sky_masks length to exactly match source_images length. "
                f"Got len(source_sky_masks)={len(source_sky_masks)} vs len(source_images)={num_views}."
            )
        for i in range(num_views):
            h, w = spatial_hw_from_image_tensor(source_images[i])
            valid = torch.ones((h, w), device=self.device, dtype=torch.float32)
            if source_egocar_masks is not None and source_egocar_masks[i] is not None:
                ego = source_egocar_masks[i].to(device=self.device, dtype=torch.float32)
                if ego.dim() == 3:
                    ego = ego.squeeze(-1)
                if tuple(ego.shape) != (h, w):
                    raise ValueError(
                        f"source_egocar_mask[{i}] shape mismatch: got {tuple(ego.shape)} expected {(h, w)}."
                    )
                valid = valid * (1.0 - ego).clamp(0.0, 1.0)
            if source_sky_masks is not None and source_sky_masks[i] is not None:
                sky = source_sky_masks[i].to(device=self.device, dtype=torch.float32)
                if sky.dim() == 3:
                    sky = sky.squeeze(-1)
                if tuple(sky.shape) != (h, w):
                    raise ValueError(
                        f"source_sky_mask[{i}] shape mismatch: got {tuple(sky.shape)} expected {(h, w)}."
                    )
                valid = valid * (1.0 - sky).clamp(0.0, 1.0)
            valid_masks.append(valid)
        return torch.stack(valid_masks, dim=0)

    def _render_source_scene_only_for_cnn(
        self,
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        source_images: List[torch.Tensor],
        source_sky_masks: Optional[List[torch.Tensor]],
        source_egocar_masks: Optional[List[torch.Tensor]],
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        if len(source_views) != len(source_images):
            raise ValueError(
                f"Stage4.5 len(source_views)={len(source_views)} != len(source_images)={len(source_images)}."
            )
        if len(source_views) < 1:
            raise ValueError("Stage4.5 requires at least one source view.")
        ref_hw = spatial_hw_from_image_tensor(source_images[0])
        for i, img in enumerate(source_images):
            hw = spatial_hw_from_image_tensor(img)
            if hw != ref_hw:
                raise ValueError(
                    "Stage4.5 multi-src currently requires identical H/W across all source_images. "
                    f"Mismatch at idx={i}: {hw} vs ref={ref_hw}."
                )

        scene_render_out = self.alpha_t_extractor.render_rgb_only(
            gaussians_scene,
            source_views,
            height,
            width,
            return_acc=True,
            return_debug_stats=False,
        )
        scene_rgbs, scene_accs = scene_render_out
        scene_rgb_batch = torch.stack(scene_rgbs, dim=0).detach()
        image_batch = torch.stack([img.to(self.device) for img in source_images], dim=0)
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)
        if scene_rgb_batch.shape[1:3] != image_batch.shape[1:3]:
            scene_rgb_batch = F.interpolate(
                scene_rgb_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)
        multi = torch.cat([image_batch, scene_rgb_batch], dim=-1)
        features_2d = self.image_feature_extractor(multi)
        source_pair_valid_mask = self._build_source_pair_valid_mask(
            source_images=source_images,
            source_sky_masks=source_sky_masks,
            source_egocar_masks=source_egocar_masks,
        )
        return {
            "features_2d": features_2d,
            "source_pair_valid_mask": source_pair_valid_mask,
        }

    def _backproject_scene_features_multi_camera(
        self,
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List[Any],
        features_2d: torch.Tensor,
        source_pair_valid_mask: torch.Tensor,
        height: int,
        width: int,
        backprojector_override=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_gaussians = int(gaussians_scene["means"].shape[0])
        if num_gaussians == 0:
            return None, None
        if not hasattr(self, "alpha_t_extractor_v3"):
            raise AttributeError(
                "Stage4.5 expects `self.alpha_t_extractor_v3` to be initialized in __init__. "
                "Do not bypass __init__ without injecting a compatible extractor."
            )
        backprojector_impl = backprojector_override if backprojector_override is not None else self.feature_backprojector
        feat_2d_all, acc_w, bp_stats = self.alpha_t_extractor_v3.render_and_backproject_streaming_fused_multi_camera(
            gaussians=gaussians_scene,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=num_gaussians,
            backprojector=backprojector_impl,
            source_pair_valid_mask=source_pair_valid_mask,
            return_accumulated_weights=True,
            return_debug_stats=True,
        )
        merge_debug_stats_as_perf_floats(self._perf_acc, "2d_bp_scene_", bp_stats)
        self._perf_acc["2d_bp_scene_call_count"] = float(self._perf_acc.get("2d_bp_scene_call_count", 0.0) + 1.0)
        return feat_2d_all, acc_w

    def _compute_2d_features_all_branches_once(
        self,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
        rigid_idx_S: torch.Tensor,
        source_views: List[Any],
        source_images: List[torch.Tensor],
        source_sky_masks: Optional[List[torch.Tensor]],
        source_egocar_masks: Optional[List[torch.Tensor]],
        height: int,
        width: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        gaussians_bg_distant, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        num_rigid_S = int(rigid_idx_S.numel())

        parts_means = [gaussians_bg_distant["means"]]
        parts_scales = [gaussians_bg_distant["scales"]]
        parts_quats = [gaussians_bg_distant["quats"]]
        parts_opacities = [gaussians_bg_distant["opacities"]]
        parts_colors = [gaussians_bg_distant["colors"]]

        if node_state_rigid is not None and num_rigid_S > 0:
            rigid_point_ids_subset = node_state_rigid.point_ids[rigid_idx_S, 0]
            means_local_S = node_state_rigid.means[rigid_idx_S]
            quats_local_S = node_state_rigid.quats[rigid_idx_S]
            rigid_means_world = self._transform_rigid_to_world(
                node_state_rigid, means_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
            )
            parts_means.append(rigid_means_world)
            parts_quats.append(
                self._transform_rigid_quats_to_world(
                    node_state_rigid, quats_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
                )
            )
            parts_scales.append(torch.exp(node_state_rigid.scales_log[rigid_idx_S]))
            parts_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[rigid_idx_S]).squeeze(-1))
            parts_colors.append(
                torch.cat(
                    [
                        node_state_rigid.sh_dc[rigid_idx_S, None, :],
                        node_state_rigid.sh_rest[rigid_idx_S],
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

        scene_ctx = self._render_source_scene_only_for_cnn(
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
            features_2d=scene_ctx["features_2d"],
            source_pair_valid_mask=scene_ctx["source_pair_valid_mask"],
            height=height,
            width=width,
            backprojector_override=None,
        )
        if feat_2d_all is None or acc_w_all is None:
            raise ValueError("Stage4.5 scene fused backprojection returned None unexpectedly.")
        self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 1.0)

        idx0 = 0
        idx1 = idx0 + num_bg
        idx2 = idx1 + num_distant
        idx3 = idx2 + num_rigid_S
        if idx3 != int(feat_2d_all.shape[0]):
            raise ValueError("Stage4.5 split size mismatch for scene-only fused backprojection.")

        feat_2d_bg = feat_2d_all[idx0:idx1]
        acc_w_bg = acc_w_all[idx0:idx1]
        feat_2d_distant = feat_2d_all[idx1:idx2] if num_distant > 0 else None
        acc_w_distant = acc_w_all[idx1:idx2] if num_distant > 0 else None
        feat_2d_rigid_S = feat_2d_all[idx2:idx3] if num_rigid_S > 0 else None
        acc_w_rigid_S = acc_w_all[idx2:idx3] if num_rigid_S > 0 else None

        return {
            "num_bg": num_bg,
            "num_distant": num_distant,
            "feat_2d_bg": feat_2d_bg,
            "acc_w_bg": acc_w_bg,
            "feat_2d_distant": feat_2d_distant,
            "acc_w_distant": acc_w_distant,
            "feat_2d_rigid_S": feat_2d_rigid_S,
            "acc_w_rigid_S": acc_w_rigid_S,
            "src_backproject_pass_count": 1,
        }

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.5 requires non-empty batch['targets'].")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        source_frame_idx = self._validate_stage4_1_batch(batch, targets, node_state_rigid)
        key = self._batch_key(batch)

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        source_sky_masks = batch.get("source_sky_mask")
        source_egocar_masks = batch.get("source_egocar_mask")
        sample_img = source_images[0]
        height, width = spatial_hw_from_image_tensor(sample_img)

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)

        N_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        mask_src_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_src_feat_valid_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_tgt_by_frame: Dict[int, torch.Tensor] = {}
        unique_target_frames = sorted({int(t["frame_idx"]) for t in targets})
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
        one_pass = self._compute_2d_features_all_branches_once(
            node_state_bg=node_state_bg,
            node_state_distant=node_state_distant,
            node_state_rigid=node_state_rigid,
            source_frame_idx=source_frame_idx,
            rigid_idx_S=S,
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

        mask_src_feat_valid_bg = acc_w_bg > self.bg_src_backproject_support_min
        mask_any_tgt_bg = self._build_any_target_mask_static(
            num_points=num_bg,
            enable_selective=self.bg_enable_selective_update,
            device=self.device,
        )
        mask_update_bg = mask_src_feat_valid_bg & mask_any_tgt_bg
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)

        mask_src_feat_valid_distant = (
            (acc_w_distant > self.distant_src_backproject_support_min) if acc_w_distant is not None else None
        )
        if num_distant > 0:
            mask_any_tgt_distant = self._build_any_target_mask_static(
                num_points=num_distant,
                enable_selective=self.distant_enable_selective_update,
                device=self.device,
            )
            mask_update_distant = mask_src_feat_valid_distant & mask_any_tgt_distant
        else:
            mask_any_tgt_distant = None
            mask_update_distant = None

        if node_state_rigid is not None and S.numel() > 0:
            if acc_w_rigid_S is None:
                raise ValueError("Stage4.5 rigid S non-empty but acc_w_rigid_S is None.")
            mask_src_feat_valid_rigid[S] = acc_w_rigid_S > self.src_backproject_support_min
            bad = mask_src_feat_valid_rigid & ~mask_src_rigid
            if bool(bad.any().item()):
                raise ValueError("mask_src_feat_valid_rigid True outside mask_src_rigid.")
        mask_update_rigid = mask_src_feat_valid_rigid & mask_any_tgt_rigid
        U = torch.nonzero(mask_update_rigid, as_tuple=False).squeeze(1)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru(
            feat_bg_input, params_bg, h_old_bg, mask_update_rigid=mask_update_bg
        )
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        render_params_rigid_local: Optional[Dict[str, torch.Tensor]] = None
        h_new_rigid: Optional[torch.Tensor] = None
        offsets_rigid: Optional[Dict[str, torch.Tensor]] = None
        if node_state_rigid is not None and U.numel() > 0 and feat_2d_rigid_S is not None and S.numel() > 0:
            lookup_s = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_s[S] = torch.arange(S.numel(), device=self.device, dtype=torch.long)
            idx_in_S = lookup_s[U]
            feat_U = feat_2d_rigid_S[idx_in_S]
            if int(feat_U.shape[-1]) != int(self.rigid_feat_in_dim):
                raise ValueError(
                    f"Rigid 2D feature dim mismatch: got {feat_U.shape[-1]}, expected {self.rigid_feat_in_dim}"
                )
            feat_U = self.rigid_feat_proj(feat_U)

            class _RigidEmbedState:
                pass

            rigid_embed_state = _RigidEmbedState()
            rigid_embed_state.means = self._transform_rigid_to_world(
                node_state_rigid,
                node_state_rigid.means[U],
                source_frame_idx,
                point_ids_subset=node_state_rigid.point_ids[U, 0],
            )
            rigid_embed_state.quats = self._transform_rigid_quats_to_world(
                node_state_rigid,
                node_state_rigid.quats[U],
                source_frame_idx,
                point_ids_subset=node_state_rigid.point_ids[U, 0],
            )
            rigid_embed_state.scales_log = node_state_rigid.scales_log[U]
            rigid_embed_state.opacity_logit = node_state_rigid.opacity_logit[U]
            rigid_embed_state.sh_dc = node_state_rigid.sh_dc[U]
            rigid_embed_state.sh_rest = node_state_rigid.sh_rest[U]
            params_rigid = self._build_params_for_embed(rigid_embed_state, coord_space="world")
            h_old_rigid = self._get_or_init_hidden(
                self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid"
            )
            h_old_rigid_U = h_old_rigid[U]
            rigid_head_rms_mask = mask_src_feat_valid_rigid[U].to(dtype=feat_U.dtype, device=feat_U.device)
            offsets_rigid, h_new_rigid_U = self._predict_offsets_gru_rigid(
                feat_U, params_rigid, h_old_rigid_U, head_rms_mask=rigid_head_rms_mask
            )
            render_params_rigid_local = self._render_params_from_offsets_rigid_local(
                NodeStateRigid(
                    means=node_state_rigid.means[U],
                    scales_log=node_state_rigid.scales_log[U],
                    quats=node_state_rigid.quats[U],
                    opacity_logit=node_state_rigid.opacity_logit[U],
                    sh_dc=node_state_rigid.sh_dc[U],
                    sh_rest=node_state_rigid.sh_rest[U],
                    point_ids=node_state_rigid.point_ids[U],
                    instances_quats=node_state_rigid.instances_quats,
                    instances_trans=node_state_rigid.instances_trans,
                    instances_fv=node_state_rigid.instances_fv,
                    instance_ids=node_state_rigid.instance_ids,
                    frame_ids=node_state_rigid.frame_ids,
                    cur_frame=node_state_rigid.cur_frame,
                ),
                offsets_rigid,
            )
            h_new_rigid = h_old_rigid.clone()
            h_new_rigid[U] = h_new_rigid_U
        if node_state_rigid is not None and h_new_rigid is None:
            h_new_rigid = self._get_or_init_hidden(
                self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid"
            ).clone()

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_2d_distant is not None and feat_2d_distant.numel() > 0:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant"
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru_distant_masked(
                feat_distant_input, params_distant, h_old_distant, mask_update_distant
            )
            render_params_distant = self._render_params_from_offsets_distant(node_state_distant, offsets_distant)

        by_frame: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
        for i, t in enumerate(targets):
            by_frame[int(t["frame_idx"])].append((i, t))
        sorted_frames = sorted(by_frame.keys())

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
                idx_tr = torch.nonzero(mask_update_rigid & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                idx_fr = torch.nonzero((~mask_update_rigid) & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
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
                if training:
                    merged_f = _merge_params_bg_rigid_distant(proxies_bg_l, prox_r, proxies_dist_l)
                else:
                    merged_f = self._tensor_merge_bg_rigid_distant_world(render_params_bg, rw, render_params_distant)

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
            pred_by_idx, _ = _run_frame_renders(False, {}, None, render_params_rigid_local, U)
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
                "_render_params_rigid_local": render_params_rigid_local,
                "_node_state_bg": node_state_bg,
                "_node_state_distant": node_state_distant,
                "_node_state_rigid": node_state_rigid,
                "_h_new_bg": h_new_bg,
                "_h_new_distant": h_new_distant,
                "_h_new_rigid": h_new_rigid,
                "_bg_writeback_idx": torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1),
                "_distant_writeback_idx": (
                    torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
                ),
                "_rigid_writeback_idx": U,
                "_rigid_valid_idx": S,
                "_num_rigid_valid_src": int(S.numel()),
                "_num_rigid_total": N_rigid,
                "_cache_key": key,
                "_src_backproject_pass_count": src_backproject_pass_count,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        pred_by_idx, rigid_world_proxy_pairs = _run_frame_renders(True, proxies_bg, proxies_distant, render_params_rigid_local, U)

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

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        frame_losses: List[torch.Tensor] = []
        frame_loss_map: Dict[int, float] = {}
        eff_frames = 0
        views_no_non_sky = 0
        for F in sorted_frames:
            group = by_frame[F]
            view_losses: List[torch.Tensor] = []
            for orig_i, t in group:
                pred_rgb = pred_rgbs_t[orig_i]
                gt_image = gt_images_t[orig_i]
                opacity = opacities_t[orig_i].to(self.device).float()
                if opacity.dim() == 3 and opacity.shape[-1] == 1:
                    opacity = opacity.squeeze(-1)
                h, w = gt_image.shape[0], gt_image.shape[1]
                valid_loss_mask = self._valid_loss_mask_from_target(t, height=h, width=w)
                if float(valid_loss_mask.sum().item()) <= 0:
                    continue
                sky_mask = t.get("sky_mask")
                if self.require_sky_mask_for_loss and sky_mask is None:
                    raise ValueError("Stage4.5 requires target['sky_mask'] for loss computation.")
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
                    ssim_i = self.loss_w_ssim * compute_ssim_loss_masked(
                        pred_rgb, gt_image, valid_mask=valid_non_sky_mask, sky_mask=None, data_range=1.0
                    )
                else:
                    views_no_non_sky += 1
                    l1_i = pred_rgb.sum() * 0.0
                    ssim_i = pred_rgb.sum() * 0.0

                gt_occupied = (1.0 - sm) * valid_loss_mask
                pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
                mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)
                p = opacity.clamp(1e-6, 1.0 - 1e-6)
                entropy_i = self.loss_w_opacity_entropy * self._masked_mean(-p * torch.log(p), valid_loss_mask)
                total_i = l1_i + ssim_i + mask_i + entropy_i
                loss_l1_list.append(l1_i)
                loss_ssim_list.append(ssim_i)
                loss_mask_list.append(mask_i)
                loss_entropy_list.append(entropy_i)
                view_losses.append(total_i)
            if view_losses:
                frame_loss = torch.stack(view_losses).mean()
                frame_losses.append(frame_loss)
                frame_loss_map[int(F)] = float(frame_loss.detach().item())
                eff_frames += 1
        if frame_losses:
            loss = torch.stack(frame_losses).mean()
        else:
            loss = render_params_bg["means_r"].sum() * 0.0
            logger.warning("Stage4.5: no valid supervision in this step; using zero loss.")

        l1_mean = torch.stack(loss_l1_list).mean() if loss_l1_list else loss * 0.0
        ssim_mean = torch.stack(loss_ssim_list).mean() if loss_ssim_list else loss * 0.0
        mask_mean = torch.stack(loss_mask_list).mean() if loss_mask_list else loss * 0.0
        entropy_mean = torch.stack(loss_entropy_list).mean() if loss_entropy_list else loss * 0.0
        offset_stats = self._collect_offset_stats(offsets_bg, offsets_rigid)
        hidden_stats = self._collect_hidden_norms(h_new_bg, h_new_distant, h_new_rigid)

        bg_writeback_idx = torch.nonzero(mask_update_bg, as_tuple=False).squeeze(1)
        distant_writeback_idx = (
            torch.nonzero(mask_update_distant, as_tuple=False).squeeze(1) if mask_update_distant is not None else None
        )
        rigid_src_feat_valid = int(mask_src_feat_valid_rigid.sum().item())
        rigid_update_count = int(U.numel())
        rigid_update_ratio = float(rigid_update_count / max(int(N_rigid), 1))
        rigid_update_among_feat_valid = float(rigid_update_count / max(rigid_src_feat_valid, 1))

        return {
            "loss": loss,
            "loss_l1": l1_mean,
            "loss_ssim": ssim_mean,
            "loss_mask": mask_mean,
            "loss_opacity_entropy": entropy_mean,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_proxies_distant": proxies_distant,
            "_proxies_rigid_world": None,
            "_rigid_world_proxy_pairs": rigid_world_proxy_pairs if rigid_world_proxy_pairs else None,
            "_render_params_distant": render_params_distant,
            "_render_params_rigid_world": None,
            "_render_params_rigid_local": render_params_rigid_local,
            "_node_state_bg": node_state_bg,
            "_node_state_distant": node_state_distant,
            "_node_state_rigid": node_state_rigid,
            "_h_new_bg": h_new_bg,
            "_h_new_distant": h_new_distant,
            "_h_new_rigid": h_new_rigid,
            "_bg_writeback_idx": bg_writeback_idx,
            "_distant_writeback_idx": distant_writeback_idx,
            "_rigid_valid_idx": S,
            "_rigid_writeback_idx": U,
            "_num_rigid_valid_src": int(S.numel()),
            "_num_rigid_src_feat_valid": int(mask_src_feat_valid_rigid.sum().item()),
            "_num_rigid_update": int(U.numel()),
            "_num_target_frames": len(sorted_frames),
            "_loss_effective_frames": eff_frames,
            "_num_rigid_total": N_rigid,
            "_frame_loss_map": frame_loss_map,
            "_offset_stats": offset_stats,
            "_hidden_stats": hidden_stats,
            "_rigid_update_ratio": rigid_update_ratio,
            "_rigid_update_among_feat_valid": rigid_update_among_feat_valid,
            "_num_bg_src_feat_valid": int(mask_src_feat_valid_bg.sum().item()),
            "_num_bg_update": int(bg_writeback_idx.numel()),
            "_num_distant_src_feat_valid": int(mask_src_feat_valid_distant.sum().item())
            if mask_src_feat_valid_distant is not None
            else 0,
            "_num_distant_update": int(distant_writeback_idx.numel()) if distant_writeback_idx is not None else 0,
            "_src_backproject_pass_count": src_backproject_pass_count,
            "_cache_key": key,
            "_num_views_no_non_sky_supervision": int(views_no_non_sky),
            "pred_rgbs": pred_rgbs_t,
            "gt_images": gt_images_t,
            "pred_rgb": pred_rgbs_t[0],
            "gt_image": gt_images_t[0],
        }

    def _resolve_export_key_and_ref_frame(
        self,
        batch_or_key: Dict[str, Any] | Tuple[int, int],
        rigid_export_frame_idx: Optional[int],
    ) -> Tuple[Tuple[int, int], int]:
        if isinstance(batch_or_key, tuple):
            key = (int(batch_or_key[0]), int(batch_or_key[1]))
            if rigid_export_frame_idx is None:
                raise ValueError(
                    "export_3dgs_state(batch_or_key=tuple) requires rigid_export_frame_idx "
                    "to export rigid branch in world/seg0 coordinates."
                )
            return key, int(rigid_export_frame_idx)
        if not isinstance(batch_or_key, dict):
            raise ValueError("batch_or_key must be a batch dict or cache key tuple(scene_id, segment_id)")
        key = self._batch_key(batch_or_key)
        if rigid_export_frame_idx is not None:
            return key, int(rigid_export_frame_idx)
        src_views = batch_or_key.get("source_views") or []
        if src_views:
            src_view = src_views[0]
            if isinstance(src_view, dict) and "frame_idx" in src_view:
                return key, int(src_view["frame_idx"])
            if hasattr(src_view, "frame_idx"):
                return key, int(getattr(src_view, "frame_idx"))
        targets = batch_or_key.get("targets") or []
        if targets:
            return key, int(targets[0]["frame_idx"])
        raise ValueError(
            "Cannot infer rigid_export_frame_idx from batch; pass rigid_export_frame_idx explicitly."
        )

    @staticmethod
    def _as_cpu_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.detach().cpu()

    def export_3dgs_state(
        self,
        batch_or_key: Dict[str, Any] | Tuple[int, int],
        *,
        include_hidden: bool = False,
        rigid_export_frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Export stage4.5 no-sky branch states.

        Rigid branch is exported in world/seg0 semantics under `rigid_world`.
        """
        key, rigid_ref_frame = self._resolve_export_key_and_ref_frame(batch_or_key, rigid_export_frame_idx)
        node_bg = self.node_states_bg.get(key)
        if node_bg is None:
            raise ValueError(f"No bg node state for cache key {key}")
        node_distant = self.node_states_distant.get(key)
        node_rigid = self.node_states_rigid.get(key)

        def _pack_branch(state: Any) -> Optional[Dict[str, torch.Tensor]]:
            if state is None:
                return None
            return {
                "means": self._as_cpu_tensor(state.means),
                "scales_log": self._as_cpu_tensor(state.scales_log),
                "quats": self._as_cpu_tensor(state.quats),
                "opacity_logit": self._as_cpu_tensor(state.opacity_logit),
                "sh_dc": self._as_cpu_tensor(state.sh_dc),
                "sh_rest": self._as_cpu_tensor(state.sh_rest),
            }

        rigid_world: Optional[Dict[str, torch.Tensor]] = None
        rigid_local = _pack_branch(node_rigid)
        if node_rigid is not None:
            point_ids = node_rigid.point_ids[:, 0] if node_rigid.point_ids.dim() > 1 else node_rigid.point_ids
            means_w = self._transform_rigid_to_world(node_rigid, node_rigid.means, rigid_ref_frame, point_ids_subset=point_ids)
            quats_w = self._transform_rigid_quats_to_world(
                node_rigid, node_rigid.quats, rigid_ref_frame, point_ids_subset=point_ids
            )
            rigid_world = {
                "means": self._as_cpu_tensor(means_w),
                "scales_log": self._as_cpu_tensor(node_rigid.scales_log),
                "quats": self._as_cpu_tensor(quats_w),
                "opacity_logit": self._as_cpu_tensor(node_rigid.opacity_logit),
                "sh_dc": self._as_cpu_tensor(node_rigid.sh_dc),
                "sh_rest": self._as_cpu_tensor(node_rigid.sh_rest),
            }

        state: Dict[str, Any] = {
            "cache_key": {"scene_id": int(key[0]), "segment_id": int(key[1])},
            "coordinate_frame": "world/seg0",
            "rigid_export_frame_idx": int(rigid_ref_frame),
            "stage": "stage4_5_no_sky",
            "branches": {
                "bg": _pack_branch(node_bg),
                "distant": _pack_branch(node_distant),
                "rigid_local": rigid_local,
                "rigid_world": rigid_world,
            },
        }

        if isinstance(batch_or_key, dict):
            req_meta = batch_or_key.get("request_meta") or {}
            src_refs = req_meta.get("source_image_refs")
            test_refs = req_meta.get("test_image_refs")
            if src_refs is not None:
                state["source_image_refs"] = list(src_refs)
            if test_refs is not None:
                state["test_image_refs"] = list(test_refs)
            if batch_or_key.get("aabb") is not None:
                state["segment_aabb"] = self._as_cpu_tensor(batch_or_key["aabb"])
            if batch_or_key.get("segment_first_frame_idx") is not None:
                state["segment_first_frame_idx"] = int(batch_or_key["segment_first_frame_idx"])

        if include_hidden:
            state["hidden"] = {
                "bg": self._as_cpu_tensor(self.h_cache_bg.get(key)),
                "distant": self._as_cpu_tensor(self.h_cache_distant.get(key)),
                "rigid": self._as_cpu_tensor(self.h_cache_rigid.get(key)),
            }
        return state

    def ensure_runtime_state_from_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        targets = batch.get("targets") or []
        if len(targets) == 0:
            raise ValueError("ensure_runtime_state_from_batch requires non-empty batch['targets']")
        return self._get_or_init_node_states_bg_rigid_distant(batch)

    def _snapshot_runtime_state(self, key: Tuple[int, int]) -> Dict[str, Any]:
        def _clone_state(s: Any) -> Any:
            if s is None:
                return None
            out: Dict[str, Any] = {}
            for k, v in vars(s).items():
                if torch.is_tensor(v):
                    out[k] = v.detach().clone()
                else:
                    out[k] = v
            return out

        return {
            "bg": _clone_state(self.node_states_bg.get(key)),
            "distant": _clone_state(self.node_states_distant.get(key)),
            "rigid": _clone_state(self.node_states_rigid.get(key)),
            "h_bg": self.h_cache_bg.get(key).detach().clone() if key in self.h_cache_bg else None,
            "h_distant": self.h_cache_distant.get(key).detach().clone() if key in self.h_cache_distant else None,
            "h_rigid": self.h_cache_rigid.get(key).detach().clone() if key in self.h_cache_rigid else None,
        }

    def _restore_runtime_state(self, key: Tuple[int, int], snap: Dict[str, Any]) -> None:
        def _restore(dst: Any, src: Dict[str, Any]) -> None:
            for k, v in src.items():
                if torch.is_tensor(v):
                    setattr(dst, k, v.to(self.device))
                else:
                    setattr(dst, k, v)

        if snap.get("bg") is not None and key in self.node_states_bg:
            _restore(self.node_states_bg[key], snap["bg"])
        if snap.get("distant") is not None and key in self.node_states_distant:
            _restore(self.node_states_distant[key], snap["distant"])
        if snap.get("rigid") is not None and key in self.node_states_rigid:
            _restore(self.node_states_rigid[key], snap["rigid"])

        for cache, name in (
            (self.h_cache_bg, "h_bg"),
            (self.h_cache_distant, "h_distant"),
            (self.h_cache_rigid, "h_rigid"),
        ):
            v = snap.get(name)
            if v is None:
                cache.pop(key, None)
            else:
                cache[key] = v.to(self.device)

    def import_3dgs_state(self, state: Dict[str, Any], *, batch_context: Optional[Dict[str, Any]] = None) -> None:
        key_block = state.get("cache_key")
        if not isinstance(key_block, dict):
            raise ValueError("state.cache_key is required")
        key = (int(key_block["scene_id"]), int(key_block["segment_id"]))
        branches = state.get("branches")
        if not isinstance(branches, dict):
            raise ValueError("state.branches is required")
        if batch_context is not None:
            self.ensure_runtime_state_from_batch(batch_context)

        def _apply(dst: Any, src: Dict[str, Any]) -> Any:
            dst.means = src["means"].to(self.device)
            dst.scales_log = src["scales_log"].to(self.device)
            dst.quats = src["quats"].to(self.device)
            dst.opacity_logit = src["opacity_logit"].to(self.device)
            dst.sh_dc = src["sh_dc"].to(self.device)
            dst.sh_rest = src["sh_rest"].to(self.device)
            return dst

        if branches.get("bg") is not None:
            if key not in self.node_states_bg:
                raise ValueError(f"Cannot import bg: key {key} does not exist in model caches")
            self.node_states_bg[key] = _apply(self.node_states_bg[key], branches["bg"])
        if branches.get("distant") is not None:
            if key not in self.node_states_distant:
                raise ValueError(f"Cannot import distant: key {key} does not exist in model caches")
            self.node_states_distant[key] = _apply(self.node_states_distant[key], branches["distant"])
        if branches.get("rigid_local") is not None:
            if key not in self.node_states_rigid:
                raise ValueError(f"Cannot import rigid_local: key {key} does not exist in model caches")
            self.node_states_rigid[key] = _apply(self.node_states_rigid[key], branches["rigid_local"])

        hidden = state.get("hidden")
        if isinstance(hidden, dict):
            for cache, name in (
                (self.h_cache_bg, "bg"),
                (self.h_cache_distant, "distant"),
                (self.h_cache_rigid, "rigid"),
            ):
                h = hidden.get(name)
                if h is not None:
                    cache[key] = h.to(self.device)

    def render_views_from_scene_state(
        self,
        scene_state: Dict[str, Any],
        eval_views: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        if len(eval_views) == 0:
            return []
        if "base_batch" not in scene_state or "gs_state" not in scene_state:
            raise ValueError("scene_state must contain base_batch and gs_state")

        src_batch = dict(scene_state["base_batch"])
        self.ensure_runtime_state_from_batch(src_batch)
        key = self._batch_key(src_batch)
        snap = self._snapshot_runtime_state(key)

        targets: List[Dict[str, Any]] = []
        for v in eval_views:
            if "gt_image" not in v:
                raise ValueError("Each eval_view must provide gt_image for render size inference")
            targets.append(
                {
                    "view": v["view"],
                    "gt_image": v["gt_image"],
                    "frame_idx": int(v["frame_idx"]),
                    "cam_idx": int(v.get("cam_idx", -1)),
                    "sky_mask": v.get("sky_mask"),
                    "egocar_mask": v.get("egocar_mask"),
                    "viewdirs": v.get("viewdirs"),
                }
            )
        src_batch["targets"] = targets

        prev_mode = self.training
        self.eval()
        try:
            self.import_3dgs_state(scene_state["gs_state"], batch_context=src_batch)
            with torch.no_grad():
                out = self.forward(src_batch)
        finally:
            if prev_mode:
                self.train()
            self._restore_runtime_state(key, snap)
        return list(out["pred_rgbs"])

    def inference_step_from_train_batch(
        self,
        batch: Dict,
        step: Optional[int] = None,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[RuntimePolicy] = None,
    ) -> Dict[str, Any]:
        policy = runtime_policy or RuntimePolicy(
            do_backward=False,
            do_optimizer_step=False,
            update_hidden_cache=True,
            writeback_node_state=True,
            reset_node_state_after_block=True,
        )
        if policy.do_backward or policy.do_optimizer_step:
            raise ValueError("inference_step_from_train_batch requires do_backward=false and do_optimizer_step=false")

        self.train()
        self._perf_acc = {}
        node_state_sync_update = False
        node_state_sync_reset = False
        with torch.no_grad():
            out = self.forward(batch)

        if policy.update_hidden_cache and "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()

        if scheduler_node_sync is not None and policy.writeback_node_state:
            u_steps = int(scheduler_node_sync["U"])
            seg = int(scheduler_node_sync["segment_local_step"])
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False)) and policy.reset_node_state_after_block
            if u_steps < 1:
                raise ValueError("scheduler_node_sync requires U >= 1 (scheduler time_base.state_write_interval_steps).")
            if seg > 0 and seg % u_steps == 0:
                self._writeback_node_states_from_out(out)
                node_state_sync_update = True
            if reset_after_block:
                self.reset_node_state()
                node_state_sync_reset = True

        loss_val = out.get("loss")
        _ = step  # keep signature parity with stage4.3 API
        return {
            "loss": loss_val.item() if torch.is_tensor(loss_val) else float(loss_val) if loss_val is not None else 0.0,
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "node_state_sync_update": node_state_sync_update,
            "node_state_sync_reset": node_state_sync_reset,
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
        num_bg = int(out.get("num_gaussians_bg", 0))
        num_distant = int(out.get("num_gaussians_distant", 0))
        num_rigid = int(out.get("num_gaussians_rigid", 0))
        out["num_gaussians_sky"] = 0
        out["num_sky_src_feat_valid"] = 0
        out["num_sky_update"] = 0
        out["sky_update_ratio"] = 0.0
        out["hidden_norm_sky_mean"] = 0.0
        out["grad_norm_sky"] = 0.0
        out["branch_presence"] = {
            "bg": bool(num_bg > 0),
            "distant": bool(num_distant > 0),
            "rigid": bool(num_rigid > 0),
            "sky": False,
        }
        return out


__all__ = ["MinimalStreetForwardStage4_5"]
