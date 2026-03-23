"""
Minimal StreetForward Stage 2.2: Multi-target, no proxy, batched multi-view render.

Same as Stage 2.0 but when all targets share the same (H, W), uses one gsplat
rasterization call for all views instead of C single-view calls.
See docs/trainers/Minimal_StreetForward_Next_Steps_Stage2_2_MultiView_Render.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.math_utils import _sh_to_rgb
from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.minimal_trainer_stage2_0 import MinimalStreetForwardStage2_0


class MinimalStreetForwardStage2_2(MinimalStreetForwardStage2_0):
    """
    Stage 2.2: Multi-target, no proxy. One set of render_params; when all targets
    have the same resolution, render all views in one gsplat call; otherwise fallback
    to per-view _render_single_view (same as 2.0). loss = mean(loss_i), single backward.
    """

    def _render_multi_view(
        self,
        render_params: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Render all targets in one gsplat call. Returns list of (rgb [H,W,3], alpha [H,W])
        or None if targets have inconsistent (height, width).
        """
        if not targets:
            return None
        viewmats_list: List[torch.Tensor] = []
        Ks_list: List[torch.Tensor] = []
        heights: List[int] = []
        widths: List[int] = []
        for target in targets:
            view = target["view"]
            gt_image = target["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            h, w = int(gt_image.shape[0]), int(gt_image.shape[1])
            heights.append(h)
            widths.append(w)
            c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            viewmat = torch.linalg.inv(c2w)
            if viewmat.dim() == 2:
                viewmat = viewmat.unsqueeze(0)
            viewmats_list.append(viewmat)
            if hasattr(view, "Ks"):
                k_mat = view.Ks[0:1]
            elif hasattr(view, "K"):
                k_mat = view.K
            else:
                k_mat = torch.eye(3, device=self.device).unsqueeze(0)
            if k_mat.dim() == 2:
                k_mat = k_mat.unsqueeze(0)
            Ks_list.append(k_mat)
        h0, w0 = heights[0], widths[0]
        if any(h != h0 or w != w0 for h, w in zip(heights, widths)):
            return None
        viewmats = torch.cat(viewmats_list, dim=0)
        Ks = torch.cat(Ks_list, dim=0)
        # gsplat expects viewmats [..., C, 4, 4]; means are [N, 3] so batch_dims=(), use (C, 4, 4) not (1, C, 4, 4)
        render, alpha, _ = self.renderer(
            means=render_params["means_r"],
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=viewmats,
            Ks=Ks,
            width=w0,
            height=h0,
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )
        C = viewmats.shape[0]
        result: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c in range(C):
            rgb = render[c, ..., :3]
            acc = alpha[c, ..., 0]
            result.append((rgb, acc))
        return result

    def forward(self, batch: Dict) -> Dict[str, Any]:
        """
        Compute one render_params, then either batch-render all targets (if same res)
        or render per view (fallback). loss = mean of per-view L1.
        Returns pred_rgbs, gt_images for multi-view logging; pred_rgb/gt_image = first view.
        """
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 2.2 requires at least one target.")

        node_state_bg = self._get_or_init_node_state_bg(batch)
        key = self._batch_key(batch)
        means = node_state_bg.means
        anchor_rgb = _sh_to_rgb(node_state_bg.sh_dc)

        feat_3d_crop = self._build_3d_features(means, anchor_rgb)
        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old = self._get_or_init_hidden(
            self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg"
        )
        offsets, h_new = self._predict_offsets_gru(feat_3d_crop, params_bg, h_old, mask_update_rigid=None)
        render_params = self._render_params_from_offsets(node_state_bg, offsets)

        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []

        multi_result = self._render_multi_view(render_params, targets)

        if multi_result is not None:
            for i, target in enumerate(targets):
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                pred_rgb = multi_result[i][0]
                loss_i = compute_l1_loss_masked(
                    pred_rgb,
                    gt_image,
                    None,
                    sky_mask=target.get("sky_mask"),
                    mask_region="non_sky",
                )
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(loss_i)
        else:
            for target in targets:
                view = target["view"]
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                height, width = gt_image.shape[0], gt_image.shape[1]
                pred_rgb, _ = self._render_single_view(render_params, view, height, width)
                loss_i = compute_l1_loss_masked(
                    pred_rgb,
                    gt_image,
                    None,
                    sky_mask=target.get("sky_mask"),
                    mask_region="non_sky",
                )
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(loss_i)

        loss = torch.stack(losses).mean()

        return {
            "loss": loss,
            "render_params": render_params,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
        }


__all__ = ["MinimalStreetForwardStage2_2"]
