"""
Minimal StreetForward Stage 4.1: multi target-frame rigid with train/frozen split and
mask_src_feat_valid from alpha-T backprojection accumulated weights (not feature norm).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.math_utils import _sh_to_rgb
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    MinimalStreetForwardStage4_0,
    _backward_to_render_params_bg_rigid_distant,
    _merge_params_bg_rigid_distant,
)
from models.streetforward.node_states import NodeStateRigid
from models.streetforward.metrics import compute_ssim_loss_masked
from models.feature_extractors import FeatureBackprojector

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage4_1(MinimalStreetForwardStage4_0):
    """Stage 4.0 + multi target frames, feat-valid mask from backproject support, train/frozen rigid."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        branches = self._require_key(config.model, "branches", "model")
        rigid_yaml = self._require_key(branches, "rigid", "model.branches")
        self.src_backproject_support_min = float(
            self._require_key(rigid_yaml, "src_backproject_support_min", "model.branches.rigid")
        )
        if self.src_backproject_support_min < 0:
            raise ValueError("model.branches.rigid.src_backproject_support_min must be non-negative.")

    def _validate_stage4_1_batch(self, batch: Dict, targets: List[Dict], node_state_rigid: Optional[NodeStateRigid]) -> int:
        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        if not source_views or not source_images:
            raise ValueError("Stage4.1 requires source_views/source_images.")
        if len(source_views) != len(source_images):
            raise ValueError(
                f"Stage4.1 len(source_views)={len(source_views)} != len(source_images)={len(source_images)}."
            )
        source_frame_idx = int(batch.get("source_frame_idx", targets[0].get("frame_idx", 0)))
        target_frames = {int(t["frame_idx"]) for t in targets}
        if node_state_rigid is not None:
            for fid in {source_frame_idx, *target_frames}:
                if self._resolve_rigid_frame_idx(node_state_rigid, fid) is None:
                    raise ValueError(
                        f"Rigid frame_idx={fid} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}"
                    )
        for i, target in enumerate(targets):
            if "sky_mask" not in target or target["sky_mask"] is None:
                raise ValueError(f"Stage4.1 requires targets[{i}].sky_mask.")
            if "viewdirs" not in target or target["viewdirs"] is None:
                raise ValueError(f"Stage4.1 requires targets[{i}].viewdirs.")
            gt_image = target["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            h, w = int(gt_image.shape[0]), int(gt_image.shape[1])
            sm = target["sky_mask"]
            if sm.dim() == 3:
                sm = sm.squeeze(-1)
            if int(sm.shape[0]) != h or int(sm.shape[1]) != w:
                raise ValueError(f"targets[{i}] sky_mask HW mismatch vs gt_image.")
            vd = target["viewdirs"]
            if vd.dim() == 4:
                vd = vd.squeeze(0)
            if int(vd.shape[0]) != h or int(vd.shape[1]) != w or int(vd.shape[2]) != 3:
                raise ValueError(f"targets[{i}] viewdirs shape mismatch vs gt_image.")
        return source_frame_idx

    @staticmethod
    def _global_to_subset_rows(global_idx: torch.Tensor, subset_U: torch.Tensor, n_rigid: int, device: torch.device) -> torch.Tensor:
        """Map global point indices (subset of U) to rows in arrays aligned with U."""
        lookup = torch.full((n_rigid,), -1, dtype=torch.long, device=device)
        lookup[subset_U] = torch.arange(subset_U.numel(), device=device, dtype=torch.long)
        rows = lookup[global_idx]
        if bool((rows < 0).any().item()):
            raise ValueError("Internal error: global_idx not contained in subset_U.")
        return rows

    def _build_rigid_world_for_frame(
        self,
        node_state_rigid: NodeStateRigid,
        frame_idx: int,
        idx_train: torch.Tensor,
        idx_frozen: torch.Tensor,
        render_params_rigid_local: Dict[str, torch.Tensor],
        U: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        chunks_means: List[torch.Tensor] = []
        chunks_quats: List[torch.Tensor] = []
        chunks_scales: List[torch.Tensor] = []
        chunks_opacities: List[torch.Tensor] = []
        chunks_colors: List[torch.Tensor] = []
        if idx_train.numel() > 0:
            if render_params_rigid_local is None or U.numel() == 0:
                raise ValueError("Internal error: trainable rigid indices need render_params_rigid_local and U.")
            rows = self._global_to_subset_rows(idx_train, U, int(node_state_rigid.means.shape[0]), self.device)
            point_ids_t = node_state_rigid.point_ids[idx_train, 0]
            means_l = render_params_rigid_local["means_r"][rows]
            quats_l = render_params_rigid_local["quats_r"][rows]
            means_w = self._transform_rigid_to_world(node_state_rigid, means_l, frame_idx, point_ids_subset=point_ids_t)
            quats_w = self._transform_rigid_quats_to_world(node_state_rigid, quats_l, frame_idx, point_ids_subset=point_ids_t)
            chunks_means.append(means_w)
            chunks_quats.append(quats_w)
            chunks_scales.append(render_params_rigid_local["scales_r"][rows])
            chunks_opacities.append(render_params_rigid_local["opacities_r"][rows])
            chunks_colors.append(render_params_rigid_local["colors_r"][rows])
        if idx_frozen.numel() > 0:
            point_ids_f = node_state_rigid.point_ids[idx_frozen, 0]
            means_l = node_state_rigid.means[idx_frozen].detach()
            quats_l = node_state_rigid.quats[idx_frozen].detach()
            means_w = self._transform_rigid_to_world(node_state_rigid, means_l, frame_idx, point_ids_subset=point_ids_f)
            quats_w = self._transform_rigid_quats_to_world(node_state_rigid, quats_l, frame_idx, point_ids_subset=point_ids_f)
            scales = torch.exp(node_state_rigid.scales_log[idx_frozen].detach())
            opacities = torch.sigmoid(node_state_rigid.opacity_logit[idx_frozen].detach()).squeeze(-1)
            sh_dc = node_state_rigid.sh_dc[idx_frozen].detach()
            sh_rest = node_state_rigid.sh_rest[idx_frozen].detach()
            colors = torch.cat([sh_dc[:, None, :], sh_rest], dim=1)
            chunks_means.append(means_w)
            chunks_quats.append(quats_w)
            chunks_scales.append(scales)
            chunks_opacities.append(opacities)
            chunks_colors.append(colors)
        if not chunks_means:
            return None
        return {
            "means_r": torch.cat(chunks_means, dim=0),
            "scales_r": torch.cat(chunks_scales, dim=0),
            "quats_r": torch.cat(chunks_quats, dim=0),
            "opacities_r": torch.cat(chunks_opacities, dim=0),
            "colors_r": torch.cat(chunks_colors, dim=0),
        }

    def _tensor_merge_bg_rigid_distant_world(
        self,
        render_params_bg: Dict[str, torch.Tensor],
        rigid_world: Optional[Dict[str, torch.Tensor]],
        render_params_distant: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        keys = ("means_r", "scales_r", "quats_r", "opacities_r", "colors_r")
        out = {k: render_params_bg[k] for k in keys}
        if rigid_world is not None:
            out = {k: torch.cat([out[k], rigid_world[k]], dim=0) for k in keys}
        if render_params_distant is not None:
            out = {k: torch.cat([out[k], render_params_distant[k]], dim=0) for k in keys}
        return out

    @staticmethod
    def _stat_tensor(t: Optional[torch.Tensor]) -> Dict[str, float]:
        if t is None or t.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0}
        x = t.detach().float()
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "max": float(x.abs().max().item()),
        }

    def _collect_offset_stats(
        self,
        offsets_bg: Optional[Dict[str, torch.Tensor]],
        offsets_rigid: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for branch, offsets in (("bg", offsets_bg), ("rigid", offsets_rigid)):
            for key in ("offset_pos", "offset_scales", "offset_opacity"):
                stats = self._stat_tensor(None if offsets is None else offsets.get(key))
                out[f"{branch}_{key}_mean"] = stats["mean"]
                out[f"{branch}_{key}_std"] = stats["std"]
                out[f"{branch}_{key}_max"] = stats["max"]
        return out

    def _collect_hidden_norms(
        self,
        h_new_bg: Optional[torch.Tensor],
        h_new_distant: Optional[torch.Tensor],
        h_new_rigid: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        def _mean_norm(t: Optional[torch.Tensor]) -> float:
            if t is None or t.numel() == 0:
                return 0.0
            return float(torch.norm(t.detach().float(), dim=-1).mean().item())

        return {
            "hidden_norm_bg_mean": _mean_norm(h_new_bg),
            "hidden_norm_distant_mean": _mean_norm(h_new_distant),
            "hidden_norm_rigid_mean": _mean_norm(h_new_rigid),
        }

    def _compute_branch_grad_norms(self) -> Dict[str, float]:
        sq_sum = {"bg": 0.0, "distant": 0.0, "rigid": 0.0}
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            g2 = float(param.grad.detach().float().pow(2).sum().item())
            if "distant" in name:
                sq_sum["distant"] += g2
            elif "rigid" in name:
                sq_sum["rigid"] += g2
            else:
                sq_sum["bg"] += g2
        return {
            "grad_norm_bg": float(sq_sum["bg"] ** 0.5),
            "grad_norm_distant": float(sq_sum["distant"] ** 0.5),
            "grad_norm_rigid": float(sq_sum["rigid"] ** 0.5),
        }

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.1 requires non-empty batch['targets'].")

        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        source_frame_idx = self._validate_stage4_1_batch(batch, targets, node_state_rigid)
        key = self._batch_key(batch)

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        sample_img = source_images[0]
        height = int(sample_img.shape[0] if sample_img.dim() == 3 else sample_img.shape[1])
        width = int(sample_img.shape[1] if sample_img.dim() == 3 else sample_img.shape[2])

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)

        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)

        N_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        mask_src_rigid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_src_feat_valid = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        mask_tgt_by_frame: Dict[int, torch.Tensor] = {}
        unique_target_frames = sorted({int(t["frame_idx"]) for t in targets})

        if node_state_rigid is not None:
            mask_src_rigid = self._rigid_point_valid_mask(node_state_rigid, source_frame_idx)
            for F in unique_target_frames:
                mask_tgt_by_frame[F] = self._rigid_point_valid_mask(node_state_rigid, F)
        else:
            for F in unique_target_frames:
                mask_tgt_by_frame[F] = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)

        mask_any_tgt = torch.zeros(N_rigid, dtype=torch.bool, device=self.device)
        for m in mask_tgt_by_frame.values():
            mask_any_tgt = mask_any_tgt | m

        S = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
        feat_S: Optional[torch.Tensor] = None
        acc_w: Optional[torch.Tensor] = None
        if node_state_rigid is not None and S.numel() > 0:
            # For rigid support/mask, disable FeatureBackprojector weight filtering.
            # This keeps the mask definition strictly based on alpha-T accumulated weights
            # (support strength), not feature norm.
            bp_unfiltered = FeatureBackprojector(
                eps=getattr(self.feature_backprojector, "eps", 1e-8),
                weight_threshold=0.0,
            )
            rigid_point_ids_subset = node_state_rigid.point_ids[S, 0]
            means_local_S = node_state_rigid.means[S]
            quats_local_S = node_state_rigid.quats[S]
            gaussians_rigid = {
                "means": self._transform_rigid_to_world(
                    node_state_rigid, means_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
                ),
                "quats": self._transform_rigid_quats_to_world(
                    node_state_rigid, quats_local_S, source_frame_idx, point_ids_subset=rigid_point_ids_subset
                ),
                "scales": torch.exp(node_state_rigid.scales_log[S]),
                "opacities": torch.sigmoid(node_state_rigid.opacity_logit[S]).squeeze(-1),
                "colors": torch.cat([node_state_rigid.sh_dc[S, None, :], node_state_rigid.sh_rest[S]], dim=1),
            }
            feat_S, acc_w = self._compute_2d_features_for_gaussians(
                gaussians=gaussians_rigid,
                source_views=source_views,
                source_images=source_images,
                height=height,
                width=width,
                return_accumulated_weights=True,
                backprojector_override=bp_unfiltered,
            )
            if feat_S is None or acc_w is None:
                raise ValueError("Stage4.1 rigid backprojection returned None despite non-empty S.")
            mask_src_feat_valid[S] = acc_w > self.src_backproject_support_min
            bad = mask_src_feat_valid & ~mask_src_rigid
            if bool(bad.any().item()):
                raise ValueError("mask_src_feat_valid True outside mask_src_rigid; check backprojection indexing.")

        mask_update = mask_src_feat_valid & mask_any_tgt
        U = torch.nonzero(mask_update, as_tuple=False).squeeze(1)

        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru(feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None)
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        render_params_rigid_local: Optional[Dict[str, torch.Tensor]] = None
        render_params_rigid_world_example: Optional[Dict[str, torch.Tensor]] = None
        h_new_rigid: Optional[torch.Tensor] = None
        offsets_rigid: Optional[Dict[str, torch.Tensor]] = None
        if (
            node_state_rigid is not None
            and U.numel() > 0
            and feat_S is not None
            and S.numel() > 0
        ):
            lookup_s = torch.full((N_rigid,), -1, dtype=torch.long, device=self.device)
            lookup_s[S] = torch.arange(S.numel(), device=self.device, dtype=torch.long)
            idx_in_S = lookup_s[U]
            feat_U = feat_S[idx_in_S]
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
            h_old_rigid = self._get_or_init_hidden(
                self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid"
            )
            h_old_rigid_U = h_old_rigid[U]
            offsets_rigid, h_new_rigid_U = self._predict_offsets_gru_rigid(feat_U, params_rigid, h_old_rigid_U)
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
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant"
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru_distant(
                feat_distant_input, params_distant, h_old_distant
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
        ) -> Tuple[
            Dict[int, Tuple[torch.Tensor, torch.Tensor]],
            List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        ]:
            pred_by_idx: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            rigid_pairs_l: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
            for F in sorted_frames:
                group = by_frame[F]
                targets_F = [t for _, t in group]
                idx_tr = torch.nonzero(mask_update & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                idx_fr = torch.nonzero((~mask_update) & mask_tgt_by_frame[F], as_tuple=False).squeeze(1)
                rw: Optional[Dict[str, torch.Tensor]] = None
                if node_state_rigid is not None and (idx_tr.numel() > 0 or idx_fr.numel() > 0):
                    if idx_tr.numel() > 0 and (rigid_local_opt is None or U_tensor.numel() == 0):
                        raise ValueError("Trainable rigid points require render_params_rigid_local and non-empty U.")
                    rw = self._build_rigid_world_for_frame(
                        node_state_rigid, F, idx_tr, idx_fr, rigid_local_opt, U_tensor
                    )
                prox_r: Optional[Dict[str, torch.Tensor]] = None
                if rw is not None and training:
                    # If this frame has no trainable rigid points, rw is frozen-only and detached.
                    # Keep rigid in render merge, but skip proxy/backward pair to avoid autograd
                    # calling backward on tensors that do not require grad.
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
                for j, (orig_i, t) in enumerate(group):
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
                pred_rgb = self._composite_sky(pr, acc, targets[i])
                pred_rgbs.append(pred_rgb)
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
                "_rigid_valid_idx": S,
                "_rigid_writeback_idx": U,
                "_num_rigid_valid_src": int(S.numel()),
                "_num_rigid_total": N_rigid,
                "_cache_key": key,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None

        pred_by_idx, rigid_world_proxy_pairs = _run_frame_renders(
            True, proxies_bg, proxies_distant, render_params_rigid_local, U
        )

        pred_rgbs_t: List[torch.Tensor] = []
        gt_images_t: List[torch.Tensor] = []
        opacities_t: List[torch.Tensor] = []
        for i in range(len(targets)):
            pr, acc = pred_by_idx[i]
            pred_rgb_sky = self._composite_sky(pr, acc, targets[i])
            pred_rgbs_t.append(pred_rgb_sky)
            gt = targets[i]["gt_image"]
            if gt.dim() == 4:
                gt = gt.squeeze(0)
            gt_images_t.append(gt)
            opacities_t.append(acc)

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        loss_total_list: List[torch.Tensor] = []

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
                loss_total_list.append(total_i)
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
            logger.warning("Stage4.1: no valid supervision in this step; using zero loss.")

        l1_mean = torch.stack(loss_l1_list).mean() if loss_l1_list else loss * 0.0
        ssim_mean = torch.stack(loss_ssim_list).mean() if loss_ssim_list else loss * 0.0
        mask_mean = torch.stack(loss_mask_list).mean() if loss_mask_list else loss * 0.0
        entropy_mean = torch.stack(loss_entropy_list).mean() if loss_entropy_list else loss * 0.0
        offset_stats = self._collect_offset_stats(offsets_bg, offsets_rigid)
        hidden_stats = self._collect_hidden_norms(h_new_bg, h_new_distant, h_new_rigid)
        rigid_src_feat_valid = int(mask_src_feat_valid.sum().item())
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
            "_rigid_valid_idx": S,
            "_rigid_writeback_idx": U,
            "_num_rigid_valid_src": int(S.numel()),
            "_num_rigid_src_feat_valid": int(mask_src_feat_valid.sum().item()),
            "_num_rigid_update": int(U.numel()),
            "_num_target_frames": len(sorted_frames),
            "_loss_effective_frames": eff_frames,
            "_num_rigid_total": N_rigid,
            "_frame_loss_map": frame_loss_map,
            "_offset_stats": offset_stats,
            "_hidden_stats": hidden_stats,
            "_rigid_update_ratio": rigid_update_ratio,
            "_rigid_update_among_feat_valid": rigid_update_among_feat_valid,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs_t,
            "gt_images": gt_images_t,
            "pred_rgb": pred_rgbs_t[0],
            "gt_image": gt_images_t[0],
        }

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
    ) -> Dict[str, Any]:
        self.train()
        timing_ms: Dict[str, float] = {
            "forward_ms": 0.0,
            "backward_ms": 0.0,
            "optimizer_ms": 0.0,
        }
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

        if self.update_node_state_interval > 0 and step is not None and step % self.update_node_state_interval == 0:
            if "_node_state_bg" in out:
                self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
            if out.get("_node_state_distant") is not None and out.get("_render_params_distant") is not None:
                self._update_node_state_distant(out["_node_state_distant"], out["_render_params_distant"])
            if out.get("_node_state_rigid") is not None and out.get("_render_params_rigid_local") is not None:
                valid_idx = out.get("_rigid_writeback_idx", out.get("_rigid_valid_idx"))
                if valid_idx is None:
                    raise ValueError("Internal error: missing rigid writeback idx.")
                if valid_idx.numel() > 0:
                    self._update_node_state_rigid_local(
                        out["_node_state_rigid"], out["_render_params_rigid_local"], valid_idx
                    )
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
        hidden_stats = out.get("_hidden_stats", {})
        offset_stats = out.get("_offset_stats", {})
        frame_loss_map = out.get("_frame_loss_map", {})
        return {
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "loss_l1": out["loss_l1"].item() if torch.is_tensor(out.get("loss_l1")) else float(out.get("loss_l1", 0.0)),
            "loss_ssim": out["loss_ssim"].item() if torch.is_tensor(out.get("loss_ssim")) else float(out.get("loss_ssim", 0.0)),
            "loss_mask": out["loss_mask"].item() if torch.is_tensor(out.get("loss_mask")) else float(out.get("loss_mask", 0.0)),
            "loss_opacity_entropy": out["loss_opacity_entropy"].item()
            if torch.is_tensor(out.get("loss_opacity_entropy"))
            else float(out.get("loss_opacity_entropy", 0.0)),
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
            **{k: float(v) for k, v in offset_stats.items()},
            **grad_norms,
            **timing_ms,
        }


__all__ = ["MinimalStreetForwardStage4_1"]
