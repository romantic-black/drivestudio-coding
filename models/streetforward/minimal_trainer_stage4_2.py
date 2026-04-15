"""
Minimal StreetForward Stage 4.2: unify source 2D backprojection across bg/distant/rigid
and add support-based update masks for bg/distant (plus rigid).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.feature_extractors import FeatureBackprojector
from models.streetforward.math_utils import _num_sh_bases, _sh_to_rgb
from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    _backward_to_render_params_bg_rigid_distant,
    _merge_params_bg_rigid_distant,
    spatial_hw_from_image_tensor,
)
from models.streetforward.minimal_trainer_stage4_1 import MinimalStreetForwardStage4_1
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage4_2(MinimalStreetForwardStage4_1):
    """Stage4.1 + unified source backprojection + bg/distant update mask gating."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        branches = self._require_key(config.model, "branches", "model")
        bg_yaml = self._require_key(branches, "bg", "model.branches")
        distant_yaml = self._require_key(branches, "distant", "model.branches")

        self.bg_src_backproject_support_min = float(
            self._require_key(bg_yaml, "src_backproject_support_min", "model.branches.bg")
        )
        self.distant_src_backproject_support_min = float(
            self._require_key(distant_yaml, "src_backproject_support_min", "model.branches.distant")
        )
        self.bg_enable_selective_update = bool(
            self._require_key(bg_yaml, "enable_selective_update", "model.branches.bg")
        )
        self.distant_enable_selective_update = bool(
            self._require_key(distant_yaml, "enable_selective_update", "model.branches.distant")
        )

    @staticmethod
    def _identity_quat(num_points: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = torch.zeros(num_points, 4, device=device, dtype=dtype)
        q[:, 0] = 1.0
        return q

    def _predict_offsets_gru_distant_masked(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        mask_update_distant: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Distant-specific heads + optional mask gate (hidden/head/offset)."""
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            num_sh = _num_sh_bases(self.sh_degree)
            offsets = {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": self._identity_quat(num_points, device, dtype),
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }
            h_new = h_old
            if mask_update_distant is not None:
                gate = mask_update_distant.to(dtype=dtype, device=device).unsqueeze(-1).detach()
                identity = self._identity_quat(num_points, device, dtype)
                offsets["offset_pos"] = offsets["offset_pos"] * gate
                offsets["offset_scales"] = offsets["offset_scales"] * gate
                offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
                offsets["offset_opacity"] = offsets["offset_opacity"] * gate
                offsets["offset_sh"] = offsets["offset_sh"] * gate
                h_new = h_old * (1.0 - gate) + h_new * gate
            return offsets, h_new

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
        head_input = self._apply_gru_head_rms(head_input, mask_update_distant)
        offsets = self._predict_offsets_with_heads(
            head_input,
            limits=self.distant_cfg["limits"],
            mlp_offset_pos=self.mlp_offset_pos_distant,
            mlp_conv=self.mlp_conv_distant,
            mlp_opacity=self.mlp_opacity_distant,
            gaussion_decoder=self.gaussion_decoder_distant,
            freeze_quat=self.distant_freeze_quat,
        )

        if mask_update_distant is not None:
            gate = mask_update_distant.to(dtype=offsets["offset_pos"].dtype, device=offsets["offset_pos"].device).unsqueeze(-1).detach()
            identity = self._identity_quat(offsets["offset_quat"].shape[0], offsets["offset_quat"].device, offsets["offset_quat"].dtype)
            offsets["offset_pos"] = offsets["offset_pos"] * gate
            offsets["offset_scales"] = offsets["offset_scales"] * gate
            offsets["offset_quat"] = torch.where(gate.expand_as(offsets["offset_quat"]).bool(), offsets["offset_quat"], identity)
            offsets["offset_opacity"] = offsets["offset_opacity"] * gate
            offsets["offset_sh"] = offsets["offset_sh"] * gate
            h_new = h_old * (1.0 - gate) + h_new * gate
        return offsets, h_new

    def _compute_2d_features_all_branches_once(
        self,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
        rigid_idx_S: torch.Tensor,
        source_views: List[Any],
        source_images: List[torch.Tensor],
        height: int,
        width: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        One-pass source backprojection for [bg, distant, rigid_S], then split by ranges.
        """
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
            parts_means.append(
                rigid_means_world
            )
            parts_quats.append(
                self._transform_rigid_quats_to_world(
                    node_state_rigid, quats_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
                )
            )
            parts_scales.append(torch.exp(node_state_rigid.scales_log[rigid_idx_S]))
            parts_opacities.append(torch.sigmoid(node_state_rigid.opacity_logit[rigid_idx_S]).squeeze(-1))
            parts_colors.append(torch.cat([node_state_rigid.sh_dc[rigid_idx_S, None, :], node_state_rigid.sh_rest[rigid_idx_S]], dim=1))
        else:
            rigid_means_world = None

        gaussians_all = {
            "means": torch.cat(parts_means, dim=0),
            "scales": torch.cat(parts_scales, dim=0),
            "quats": torch.cat(parts_quats, dim=0),
            "opacities": torch.cat(parts_opacities, dim=0),
            "colors": torch.cat(parts_colors, dim=0),
        }

        bp_unfiltered = FeatureBackprojector(
            eps=getattr(self.feature_backprojector, "eps", 1e-8),
            weight_threshold=0.0,
        )
        pass_count = 0
        pass_count += 1
        feat_2d_all, acc_w_all = self._compute_2d_features_for_gaussians(
            gaussians=gaussians_all,
            source_views=source_views,
            source_images=source_images,
            height=height,
            width=width,
            return_accumulated_weights=True,
            backprojector_override=bp_unfiltered,
        )
        if feat_2d_all is None or acc_w_all is None:
            raise ValueError("Stage4.2 one-pass backprojection returned None unexpectedly.")

        idx0 = 0
        idx1 = idx0 + num_bg
        idx2 = idx1 + num_distant
        idx3 = idx2 + num_rigid_S
        if idx3 != int(feat_2d_all.shape[0]):
            raise ValueError("Stage4.2 split size mismatch for one-pass backprojection.")

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
            "src_backproject_pass_count": pass_count,
        }

    def _build_any_target_mask_static(
        self,
        num_points: int,
        enable_selective: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Stage4_2 static branches currently do not have per-point target visibility precomputation.
        Selective toggle is kept for compatibility; mask remains all-ones when enabled.
        """
        if num_points <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if not enable_selective:
            return torch.ones(num_points, dtype=torch.bool, device=device)
        return torch.ones(num_points, dtype=torch.bool, device=device)

    def _update_node_state_bg_subset(
        self,
        node_state_bg: NodeStateBackground,
        render_params: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            means_clamped = torch.clamp(
                render_params["means_r"][valid_idx].detach(),
                min=self.bbx_min,
                max=self.bbx_max,
            )
            node_state_bg.means[valid_idx] = means_clamped
            node_state_bg.scales_log[valid_idx] = render_params["scales_log_r"][valid_idx].detach()
            node_state_bg.quats[valid_idx] = render_params["quats_r"][valid_idx].detach()
            node_state_bg.opacity_logit[valid_idx] = render_params["opacity_logit_r"][valid_idx].detach()
            node_state_bg.sh_dc[valid_idx] = render_params["sh_dc_r"][valid_idx].detach()
            node_state_bg.sh_rest[valid_idx] = render_params["sh_rest_r"][valid_idx].detach()

    def _update_node_state_distant_subset(
        self,
        node_state_distant: NodeStateDistant,
        render_params: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            # Distant Gaussians are far-field / segment-exterior by design; do not clamp means to
            # dataset.segment_aabb (input_aabb_*). Clamping collapsed visible distant points onto the
            # AABB shell and destroyed frustum overlap after the first scheduler writeback.
            node_state_distant.means[valid_idx] = render_params["means_r"][valid_idx].detach()
            node_state_distant.scales_log[valid_idx] = render_params["scales_log_r"][valid_idx].detach()
            node_state_distant.quats[valid_idx] = render_params["quats_r"][valid_idx].detach()
            node_state_distant.opacity_logit[valid_idx] = render_params["opacity_logit_r"][valid_idx].detach()
            node_state_distant.sh_dc[valid_idx] = render_params["sh_dc_r"][valid_idx].detach()
            node_state_distant.sh_rest[valid_idx] = render_params["sh_rest_r"][valid_idx].detach()

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.2 requires non-empty batch['targets'].")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        source_frame_idx = self._validate_stage4_1_batch(batch, targets, node_state_rigid)
        key = self._batch_key(batch)

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        source_images = self._apply_source_egocar_mask(source_images, batch.get("source_egocar_mask"))
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
                raise ValueError("Stage4.2 rigid S non-empty but acc_w_rigid_S is None.")
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
                raise ValueError(f"Rigid 2D feature dim mismatch: got {feat_U.shape[-1]}, expected {self.rigid_feat_in_dim}")
            feat_U = self.rigid_feat_proj(feat_U)

            class _RigidEmbedState:
                pass

            rigid_embed_state = _RigidEmbedState()
            rigid_embed_state.means = self._transform_rigid_to_world(
                node_state_rigid, node_state_rigid.means[U], source_frame_idx, point_ids_subset=node_state_rigid.point_ids[U, 0]
            )
            rigid_embed_state.quats = self._transform_rigid_quats_to_world(
                node_state_rigid, node_state_rigid.quats[U], source_frame_idx, point_ids_subset=node_state_rigid.point_ids[U, 0]
            )
            rigid_embed_state.scales_log = node_state_rigid.scales_log[U]
            rigid_embed_state.opacity_logit = node_state_rigid.opacity_logit[U]
            rigid_embed_state.sh_dc = node_state_rigid.sh_dc[U]
            rigid_embed_state.sh_rest = node_state_rigid.sh_rest[U]
            params_rigid = self._build_params_for_embed(rigid_embed_state, coord_space="world")
            h_old_rigid = self._get_or_init_hidden(self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid")
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
            h_new_rigid = self._get_or_init_hidden(self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid").clone()

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_2d_distant is not None and feat_2d_distant.numel() > 0:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant")
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
                pr, acc = pred_by_idx[i]
                pred_rgbs.append(self._composite_sky(pr, acc, targets[i]))
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
            pred_rgbs_t.append(self._composite_sky(pr, acc, targets[i]))
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
                l1_i = self.loss_w_l1 * torch.mean(torch.abs((pred_rgb - gt_image) * valid_loss_mask.unsqueeze(-1)))
                ssim_i = self.loss_w_ssim * compute_ssim_loss_masked(
                    pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None, data_range=1.0
                )
                sm = t["sky_mask"].to(self.device).float()
                if sm.dim() == 3:
                    sm = sm.squeeze(-1)
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
            logger.warning("Stage4.2: no valid supervision in this step; using zero loss.")

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
            "_num_distant_src_feat_valid": int(mask_src_feat_valid_distant.sum().item()) if mask_src_feat_valid_distant is not None else 0,
            "_num_distant_update": int(distant_writeback_idx.numel()) if distant_writeback_idx is not None else 0,
            "_src_backproject_pass_count": src_backproject_pass_count,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs_t,
            "gt_images": gt_images_t,
            "pred_rgb": pred_rgbs_t[0],
            "gt_image": gt_images_t[0],
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
            valid_idx = out.get("_rigid_writeback_idx", out.get("_rigid_valid_idx"))
            if valid_idx is None:
                raise ValueError("Internal error: missing rigid writeback idx.")
            if valid_idx.numel() > 0:
                self._update_node_state_rigid_local(out["_node_state_rigid"], out["_render_params_rigid_local"], valid_idx)

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.train()
        self._perf_acc = {}
        node_state_sync_update = False
        node_state_sync_reset = False
        timing_ms: Dict[str, float] = {"forward_ms": 0.0, "backward_ms": 0.0, "optimizer_ms": 0.0}
        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        t1 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["forward_ms"] = float((t1 - t0) * 1000.0)
        if torch.is_tensor(out.get("loss")):
            out["loss"].backward()
        if out.get("proxies") is not None:
            _backward_to_render_params_bg_rigid_distant(
                out["render_params"],
                out["proxies"],
                out.get("_render_params_rigid_world"),
                out.get("_proxies_rigid_world"),
                out.get("_render_params_distant"),
                out.get("_proxies_distant"),
                rigid_world_proxy_pairs=out.get("_rigid_world_proxy_pairs"),
            )
        grad_norms = self._compute_branch_grad_norms()
        t2 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["backward_ms"] = float((t2 - t1) * 1000.0)
        self.optimizer.step()
        t3 = time.perf_counter()
        if profile_phase_timing:
            if sync_cuda_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_ms["optimizer_ms"] = float((t3 - t2) * 1000.0)
        if "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()

        if scheduler_node_sync is not None:
            U = int(scheduler_node_sync["U"])
            seg = int(scheduler_node_sync["segment_local_step"])
            reset_after_block = bool(scheduler_node_sync.get("reset_after_block", False))
            if U < 1:
                raise ValueError("scheduler_node_sync requires U >= 1 (scheduler time_base.state_write_interval_steps).")
            if seg > 0 and seg % U == 0:
                self._writeback_node_states_from_out(out)
                node_state_sync_update = True
            if reset_after_block:
                self.reset_node_state()
                node_state_sync_reset = True
        elif self.update_node_state_interval > 0 and step is not None and step % self.update_node_state_interval == 0:
            self._writeback_node_states_from_out(out)
            if self.reset_node_state_interval > 0 and step % self.reset_node_state_interval == 0:
                self.reset_node_state()

        num_gaussians_bg = int(out["_node_state_bg"].means.shape[0])
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")
        num_gaussians_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_gaussians_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        num_rigid_valid_src = int(out.get("_num_rigid_valid_src", 0))
        num_rigid_total = int(out.get("_num_rigid_total", num_gaussians_rigid))
        writeback_idx = out.get("_rigid_writeback_idx")
        writeback_count = int(writeback_idx.numel()) if writeback_idx is not None else 0
        writeback_rigid_ratio = float(writeback_count / max(num_rigid_total, 1))
        bg_w_idx = out.get("_bg_writeback_idx")
        bg_w_count = int(bg_w_idx.numel()) if bg_w_idx is not None else num_gaussians_bg
        writeback_bg_ratio = float(bg_w_count / max(num_gaussians_bg, 1))
        distant_w_idx = out.get("_distant_writeback_idx")
        distant_w_count = int(distant_w_idx.numel()) if distant_w_idx is not None else num_gaussians_distant
        writeback_distant_ratio = float(distant_w_count / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0
        hidden_stats = out.get("_hidden_stats", {})
        offset_stats = out.get("_offset_stats", {})
        frame_loss_map = out.get("_frame_loss_map", {})

        num_bg_src_feat_valid = int(out.get("_num_bg_src_feat_valid", 0))
        num_bg_update = int(out.get("_num_bg_update", 0))
        num_distant_src_feat_valid = int(out.get("_num_distant_src_feat_valid", 0))
        num_distant_update = int(out.get("_num_distant_update", 0))
        perf_metrics: Dict[str, float] = {}
        perf_calls = float(self._perf_acc.get("2d_call_count", 0.0))
        if perf_calls > 0.0:
            for k, v in self._perf_acc.items():
                if k == "2d_call_count":
                    continue
                # Memory fields keep summed values for deltas; timing fields are averaged per call.
                if "cuda_mem_" in k:
                    perf_metrics[f"perf_{k}"] = float(v)
                else:
                    perf_metrics[f"perf_{k}"] = float(v / perf_calls)
        perf_metrics["perf_2d_call_count"] = perf_calls

        return {
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "loss_l1": out["loss_l1"].item() if torch.is_tensor(out.get("loss_l1")) else float(out.get("loss_l1", 0.0)),
            "loss_ssim": out["loss_ssim"].item() if torch.is_tensor(out.get("loss_ssim")) else float(out.get("loss_ssim", 0.0)),
            "loss_mask": out["loss_mask"].item() if torch.is_tensor(out.get("loss_mask")) else float(out.get("loss_mask", 0.0)),
            "loss_opacity_entropy": out["loss_opacity_entropy"].item() if torch.is_tensor(out.get("loss_opacity_entropy")) else float(out.get("loss_opacity_entropy", 0.0)),
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": num_gaussians_bg,
            "num_gaussians_distant": num_gaussians_distant,
            "num_gaussians_rigid": num_gaussians_rigid,
            "num_rigid_valid_src": num_rigid_valid_src,
            "num_rigid_invalid_src": int(max(num_rigid_total - num_rigid_valid_src, 0)),
            "rigid_valid_ratio": float(num_rigid_valid_src / max(num_rigid_total, 1)),
            "num_rigid_src_feat_valid": int(out.get("_num_rigid_src_feat_valid", 0)),
            "num_rigid_update": int(out.get("_num_rigid_update", 0)),
            "rigid_update_ratio": float(out.get("_rigid_update_ratio", 0.0)),
            "rigid_update_among_feat_valid": float(out.get("_rigid_update_among_feat_valid", 0.0)),
            "writeback_rigid_ratio": writeback_rigid_ratio,
            "num_target_frames": int(out.get("_num_target_frames", 0)),
            "loss_effective_frames": int(out.get("_loss_effective_frames", 0)),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
            "frame_loss_map": frame_loss_map,
            "hidden_norm_bg_mean": float(hidden_stats.get("hidden_norm_bg_mean", 0.0)),
            "hidden_norm_distant_mean": float(hidden_stats.get("hidden_norm_distant_mean", 0.0)),
            "hidden_norm_rigid_mean": float(hidden_stats.get("hidden_norm_rigid_mean", 0.0)),
            "num_bg_src_feat_valid": num_bg_src_feat_valid,
            "num_bg_update": num_bg_update,
            "bg_update_ratio": float(num_bg_update / max(num_gaussians_bg, 1)),
            "num_distant_src_feat_valid": num_distant_src_feat_valid,
            "num_distant_update": num_distant_update,
            "distant_update_ratio": float(num_distant_update / max(num_gaussians_distant, 1)) if num_gaussians_distant > 0 else 0.0,
            "writeback_bg_ratio": writeback_bg_ratio,
            "writeback_distant_ratio": writeback_distant_ratio,
            "src_backproject_pass_count": int(out.get("_src_backproject_pass_count", 0)),
            **{k: float(v) for k, v in offset_stats.items()},
            **grad_norms,
            **timing_ms,
            **perf_metrics,
            "node_state_sync_update": node_state_sync_update,
            "node_state_sync_reset": node_state_sync_reset,
        }


__all__ = ["MinimalStreetForwardStage4_2"]

