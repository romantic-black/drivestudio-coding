"""
Minimal StreetForward Stage 2.0: Multi-target, no proxy.

Extends Stage 1.1: same render_params for all targets, loss = mean(loss_i), single backward.
See docs/trainers/Minimal_StreetForward_Next_Steps_Stage2_MultiTarget.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from models.streetforward.math_utils import _sh_to_rgb
from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.minimal_trainer_stage1_1 import MinimalStreetForwardStage1_1


class MinimalStreetForwardStage2_0(MinimalStreetForwardStage1_1):
    """
    Stage 2.0: Multi-target, no proxy. One set of render_params, render each target,
    loss = mean(loss_i), single backward.
    """

    def forward(self, batch: Dict) -> Dict[str, Any]:
        """
        Compute one render_params, then render all targets with it; loss = mean of per-view L1.
        Returns pred_rgbs, gt_images for multi-view logging; pred_rgb/gt_image = first view.
        """
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 2.0 requires at least one target.")

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

        for target in targets:
            point_coverage_mask = target.get("point_coverage_mask")
            if point_coverage_mask is None:
                raise ValueError(
                    "target['point_coverage_mask'] is required. Ensure batch provides point_coverage_mask."
                )
            view = target["view"]
            gt_image = target["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            height, width = gt_image.shape[0], gt_image.shape[1]
            pred_rgb, _ = self._render_single_view(render_params, view, height, width)
            loss_i = compute_l1_loss_masked(
                pred_rgb, gt_image, point_coverage_mask, sky_mask=target.get("sky_mask")
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

    def train_step(self, batch: Dict, step: Optional[int] = None) -> Dict[str, Any]:
        """One step: forward, single backward, step; write h_cache_bg; optionally update NodeState."""
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        loss = out["loss"]
        loss.backward()
        self.optimizer.step()
        if "_h_new_bg" in out and "_cache_key" in out:
            self.h_cache_bg[out["_cache_key"]] = out["_h_new_bg"].detach()
        if (
            self.update_node_state_interval > 0
            and step is not None
            and step % self.update_node_state_interval == 0
            and "_node_state_bg" in out
        ):
            self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
        return {
            "loss": loss.item(),
            "pred_rgbs": [p.detach() for p in out["pred_rgbs"]],
            "gt_images": [g.detach() for g in out["gt_images"]],
            "pred_rgb": out["pred_rgb"].detach(),
            "gt_image": out["gt_image"].detach(),
            "num_gaussians_bg": int(out["_node_state_bg"].means.shape[0]),
            "num_targets": len(batch.get("targets", [])),
        }


__all__ = ["MinimalStreetForwardStage2_0"]
