"""
Minimal StreetForward Stage 3.1: Stage 3.2d + learnable sky cubemap.

Composites rgb_composite = rgb_gaussians + rgb_sky * (1 - opacity); loss on rgb_composite.
Target viewdirs must be provided by MultiSceneDataset (pixel_source.get_image() returns viewdirs)
or by convert_batch_to_minimal_format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.minimal_trainer_stage3_2d import (
    MinimalStreetForwardStage3_2d,
    _backward_to_render_params_bg_distant,
    _create_proxy_params,
    _merge_params_bg_distant,
)
from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.sky_cubemap import SkyCubemap

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage3_1(MinimalStreetForwardStage3_2d):
    """
    Stage 3.1: Stage 3.2d with sky (cubemap + dr.texture), composite with gaussian render, joint backward.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        sky_cfg = config.model.get("sky", {})
        resolution = int(sky_cfg.get("resolution", 1024))
        init_value = float(sky_cfg.get("init_value", 0.5))
        self.sky_model = SkyCubemap(
            resolution=resolution,
            init_value=init_value,
            device=device,
        ).to(device)
        # Rebuild optimizer so it includes sky parameters
        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(config.optimizer.get("lr", 1e-3)),
            eps=float(config.optimizer.get("eps", 1e-15)),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        )

    def _get_viewdirs_for_target(
        self, target: Dict, height: int, width: int
    ) -> torch.Tensor:
        """Return viewdirs [H, W, 3] for the given target. Must be provided by the dataset (MultiSceneDataset) or by convert_batch_to_minimal_format."""
        viewdirs = target.get("viewdirs")
        if viewdirs is None:
            raise ValueError(
                "Stage 3.1 (sky) requires target viewdirs. "
                "Provide them via MultiSceneDataset (pixel_source.get_image() must return image_infos['viewdirs']) "
                "or ensure convert_batch_to_minimal_format supplies viewdirs for each target."
            )
        v = viewdirs.to(self.device)
        if v.dim() != 3 or v.shape[-1] != 3:
            raise ValueError(
                f"target['viewdirs'] must have shape [H, W, 3], got {tuple(v.shape)}"
            )
        if v.shape[0] != height or v.shape[1] != width:
            raise ValueError(
                "Stage 3.1 requires viewdirs at the exact render resolution. "
                f"Got viewdirs {tuple(v.shape)} but render expects (H,W,3)=({height},{width},3). "
                "Fix the dataset to emit matching-resolution viewdirs (recommended), "
                "or recompute viewdirs with get_rays at the desired resolution during batch conversion."
            )
        v = torch.nn.functional.normalize(v, dim=-1)
        return v

    def _composite_sky(
        self,
        pred_rgb: torch.Tensor,
        opacity: torch.Tensor,
        target: Dict,
    ) -> torch.Tensor:
        """Return rgb_composite = pred_rgb + rgb_sky * (1 - opacity)."""
        h, w = int(pred_rgb.shape[0]), int(pred_rgb.shape[1])
        viewdirs = self._get_viewdirs_for_target(target, h, w)
        rgb_sky = self.sky_model({"viewdirs": viewdirs})
        # Ensure (H, W, 3): if sky model returned (H, 3, W), permute to (H, W, 3) so broadcast matches pred_rgb.
        if rgb_sky.dim() == 3 and rgb_sky.shape[2] != 3:
            rgb_sky = rgb_sky.permute(0, 2, 1).contiguous()
        # opacity may be [H, W] or [H, W, 1] depending on renderer path; normalize to [H, W].
        if opacity.dim() == 3 and opacity.shape[-1] == 1:
            opacity = opacity.squeeze(-1)
        one_minus_opacity = (1.0 - opacity.clamp(0.0, 1.0)).unsqueeze(-1)
        return pred_rgb + rgb_sky * one_minus_opacity

    def _composite_sky_batched(
        self,
        pred_rgbs: torch.Tensor,
        opacities: torch.Tensor,
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        Batched sky composite for multi-view rendering.

        Inputs:
          - pred_rgbs: (T, H, W, 3)
          - opacities: (T, H, W) or (T, H, W, 1)
          - targets: list of len T, each contains viewdirs (H, W, 3)
        Output:
          - rgb_composite: (T, H, W, 3)
        """
        if pred_rgbs.dim() != 4 or pred_rgbs.shape[-1] != 3:
            raise ValueError(f"pred_rgbs must have shape (T,H,W,3), got {tuple(pred_rgbs.shape)}")
        T, H, W, _ = pred_rgbs.shape
        if len(targets) != T:
            raise ValueError(f"targets length {len(targets)} must match T={T}")

        viewdirs_batched = torch.stack(
            [self._get_viewdirs_for_target(t, H, W) for t in targets], dim=0
        )
        rgb_sky = self.sky_model({"viewdirs": viewdirs_batched})

        if rgb_sky.shape != pred_rgbs.shape:
            raise ValueError(
                f"rgb_sky shape {tuple(rgb_sky.shape)} must match pred_rgbs {tuple(pred_rgbs.shape)}"
            )

        if opacities.dim() == 4 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        if opacities.dim() != 3:
            raise ValueError(f"opacities must have shape (T,H,W) or (T,H,W,1), got {tuple(opacities.shape)}")
        one_minus_opacity = (1.0 - opacities.clamp(0.0, 1.0)).unsqueeze(-1)
        return pred_rgbs + rgb_sky * one_minus_opacity

    def forward(self, batch: Dict) -> Dict[str, Any]:
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 3 requires at least one target.")
        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        if not source_views or not source_images:
            raise ValueError(
                "Stage 3 requires source_views and source_images. "
                "Use convert_batch_to_minimal_format(..., include_source_for_2d=True)."
            )

        node_state_bg, node_state_distant = self._get_or_init_node_states_bg_distant(batch)
        key = self._batch_key(batch)
        means_bg = node_state_bg.means
        from models.streetforward.math_utils import _sh_to_rgb

        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)

        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)
        sample_img = source_images[0]
        height = int(sample_img.shape[0] if sample_img.dim() == 3 else sample_img.shape[1])
        width = int(sample_img.shape[1] if sample_img.dim() == 3 else sample_img.shape[2])
        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(
            node_state_bg, node_state_distant
        )
        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)
        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            zeros_3d = torch.zeros(
                num_distant, self.feat_3d_dim, device=self.device, dtype=feat_2d_distant.dtype
            )
            vis_d = torch.ones(num_distant, device=self.device)
            feat_distant_input = self._fuse_features(zeros_3d, feat_2d_distant, vis_d)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(
            self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg"
        )
        offsets_bg, h_new_bg = self._predict_offsets_gru(
            feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None
        )
        render_params_bg = self._render_params_from_offsets(node_state_bg, offsets_bg)

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant,
                key,
                node_state_distant.means.shape[0],
                node_state_distant,
                "distant",
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru(
                feat_distant_input, params_distant, h_old_distant, mask_update_rigid=None
            )
            render_params_distant = self._render_params_from_offsets_distant(
                node_state_distant, offsets_distant
            )

        if not self.training:
            if render_params_distant is not None:
                merged = {
                    "means_r": torch.cat([render_params_bg["means_r"], render_params_distant["means_r"]]),
                    "scales_r": torch.cat([render_params_bg["scales_r"], render_params_distant["scales_r"]]),
                    "quats_r": torch.cat([render_params_bg["quats_r"], render_params_distant["quats_r"]]),
                    "opacities_r": torch.cat([render_params_bg["opacities_r"], render_params_distant["opacities_r"]]),
                    "colors_r": torch.cat([render_params_bg["colors_r"], render_params_distant["colors_r"]]),
                }
            else:
                merged = {
                    "means_r": render_params_bg["means_r"],
                    "scales_r": render_params_bg["scales_r"],
                    "quats_r": render_params_bg["quats_r"],
                    "opacities_r": render_params_bg["opacities_r"],
                    "colors_r": render_params_bg["colors_r"],
                }

            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []

            multi_result = self._render_multi_view(merged, targets)
            if multi_result is not None:
                pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
                acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
                pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
                for i, target in enumerate(targets):
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    pred_rgbs.append(pred_stack[i])
                    gt_images.append(gt_image)
            else:
                for target in targets:
                    view = target["view"]
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    h, w = gt_image.shape[0], gt_image.shape[1]
                    pred_rgb, acc = self._render_single_view(merged, view, h, w)
                    pred_rgb = self._composite_sky(pred_rgb, acc, target)
                    pred_rgbs.append(pred_rgb)
                    gt_images.append(gt_image)

            loss = torch.tensor(0.0, device=self.device)
            return {
                "loss": loss,
                "render_params": render_params_bg,
                "proxies": None,
                "_node_state_bg": node_state_bg,
                "_h_new_bg": h_new_bg,
                "_cache_key": key,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_h_new_distant": h_new_distant,
                "_node_state_distant": node_state_distant,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        merged_for_render = _merge_params_bg_distant(proxies_bg, proxies_distant)

        pred_rgbs = []
        gt_images = []
        losses = []

        multi_result = self._render_multi_view(merged_for_render, targets)

        if multi_result is not None:
            pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
            acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
            pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
            for i, target in enumerate(targets):
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                pred_rgb = pred_stack[i]
                loss_i = compute_l1_loss_masked(
                    pred_rgb, gt_image,
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
                pred_rgb, acc = self._render_single_view(merged_for_render, view, height, width)
                pred_rgb = self._composite_sky(pred_rgb, acc, target)
                loss_i = compute_l1_loss_masked(
                    pred_rgb, gt_image
                )
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(loss_i)

        loss = torch.stack(losses).mean() if losses else torch.tensor(
            0.0, device=render_params_bg["means_r"].device, dtype=render_params_bg["means_r"].dtype
        )
        return {
            "loss": loss,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new_bg,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
            "_render_params_distant": render_params_distant,
            "_proxies_distant": proxies_distant,
            "_h_new_distant": h_new_distant,
            "_node_state_distant": node_state_distant,
        }


__all__ = ["MinimalStreetForwardStage3_1"]
