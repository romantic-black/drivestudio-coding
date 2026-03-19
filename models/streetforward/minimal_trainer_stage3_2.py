"""
Minimal StreetForward Stage 3.2: Stage 3.1 + SSIM loss.

- Inherits Stage 3.1 (2D + bg + distant + learnable sky cubemap).
- Uses rgb_composite = rgb_gaussians + rgb_sky * (1 - opacity).
- Loss: 0.8 * L1 + 0.2 * (1 - SSIM), aligned with models/trainers/base.py compute_losses().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.metrics import compute_l1_loss_masked, compute_ssim_loss_masked
from models.streetforward.minimal_trainer_stage3_1 import MinimalStreetForwardStage3_1

logger = logging.getLogger(__name__)


class MinimalStreetForwardStage3_2(MinimalStreetForwardStage3_1):
    """
    Stage 3.2: Stage 3.1 + mixed L1/SSIM loss on rgb_composite.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is None:
            raise ValueError(
                "Stage 3.2 requires top-level config.losses with rgb/ssim/mask/opacity_entropy."
            )
        rgb_cfg = losses_cfg.get("rgb") if hasattr(losses_cfg, "get") else getattr(losses_cfg, "rgb", None)
        ssim_cfg = losses_cfg.get("ssim") if hasattr(losses_cfg, "get") else getattr(losses_cfg, "ssim", None)
        mask_cfg = losses_cfg.get("mask") if hasattr(losses_cfg, "get") else getattr(losses_cfg, "mask", None)
        entropy_cfg = (
            losses_cfg.get("opacity_entropy")
            if hasattr(losses_cfg, "get")
            else getattr(losses_cfg, "opacity_entropy", None)
        )
        if rgb_cfg is None or ssim_cfg is None or mask_cfg is None or entropy_cfg is None:
            raise ValueError(
                "Stage 3.2 requires config.losses.rgb, config.losses.ssim, "
                "config.losses.mask, and config.losses.opacity_entropy."
            )
        self.loss_w_l1 = float(rgb_cfg.get("w") if hasattr(rgb_cfg, "get") else getattr(rgb_cfg, "w"))
        self.loss_w_ssim = float(ssim_cfg.get("w") if hasattr(ssim_cfg, "get") else getattr(ssim_cfg, "w"))
        self.loss_w_mask = float(mask_cfg.get("w") if hasattr(mask_cfg, "get") else getattr(mask_cfg, "w"))
        self.opacity_loss_type = str(
            mask_cfg.get("opacity_loss_type") if hasattr(mask_cfg, "get") else getattr(mask_cfg, "opacity_loss_type")
        )
        self.loss_w_opacity_entropy = float(
            entropy_cfg.get("w") if hasattr(entropy_cfg, "get") else getattr(entropy_cfg, "w")
        )

        if self.opacity_loss_type not in {"bce", "safe_bce"}:
            raise ValueError(
                f"losses.mask.opacity_loss_type must be one of ['bce','safe_bce'], got {self.opacity_loss_type!r}"
            )

    def _valid_loss_mask_from_target(self, target: Dict, *, height: int, width: int) -> torch.Tensor:
        egocar_mask = target.get("egocar_mask")
        if egocar_mask is None:
            return torch.ones((height, width), dtype=torch.float32, device=self.device)
        m = egocar_mask.to(self.device).float()
        if m.dim() == 3:
            m = m.squeeze(-1)
        if m.shape[0] != height or m.shape[1] != width:
            raise ValueError(
                f"target['egocar_mask'] must match image shape [H,W], got {tuple(m.shape)} vs H,W=({height},{width})"
            )
        # egocar_mask: 1 means ignore; valid_loss_mask: 1 means keep
        return (1.0 - m).clamp(0.0, 1.0)

    def _masked_mean(self, value_2d: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        if value_2d.shape != mask_2d.shape:
            raise ValueError(f"masked_mean expects same shapes, got value={tuple(value_2d.shape)} mask={tuple(mask_2d.shape)}")
        denom = mask_2d.sum()
        if denom > 0:
            return (value_2d * mask_2d).sum() / denom
        return value_2d.sum() * 0.0

    def _mask_bce(self, pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        if self.opacity_loss_type == "safe_bce":
            pred = pred.clamp(1e-6, 1.0 - 1e-6)
        gt = gt.clamp(0.0, 1.0)
        bce_map = F.binary_cross_entropy(pred, gt, reduction="none")
        return self._masked_mean(bce_map, valid_mask)

    def forward(self, batch: Dict) -> Dict[str, Any]:
        # Re-implement Stage 3.1 forward training branch so we can access per-view opacity (acc).
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

        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)

        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )

        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)
        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            zeros_3d = torch.zeros(num_distant, self.feat_3d_dim, device=self.device, dtype=feat_2d_distant.dtype)
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
            # Keep eval behavior identical to Stage 3.1.
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

        # Training: render with proxies (same as Stage 3.1) but keep opacity per view.
        from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params, _merge_params_bg_distant

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        merged_for_render = _merge_params_bg_distant(proxies_bg, proxies_distant)

        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        opacities: List[torch.Tensor] = []

        multi_result = self._render_multi_view(merged_for_render, targets)
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
                opacities.append(acc_stack[i])
        else:
            for target in targets:
                view = target["view"]
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                h, w = gt_image.shape[0], gt_image.shape[1]
                pred_rgb, acc = self._render_single_view(merged_for_render, view, h, w)
                pred_rgb = self._composite_sky(pred_rgb, acc, target)
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                opacities.append(acc.squeeze(-1) if acc.dim() == 3 and acc.shape[-1] == 1 else acc)

        if len(pred_rgbs) != len(targets) or len(opacities) != len(targets):
            raise ValueError(
                f"Internal error: pred/opacities/targets length mismatch: pred={len(pred_rgbs)} "
                f"opacity={len(opacities)} targets={len(targets)}"
            )

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_rgb_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        loss_total_list: List[torch.Tensor] = []

        for i, target in enumerate(targets):
            pred_rgb = pred_rgbs[i]
            gt_image = gt_images[i]
            opacity = opacities[i].to(self.device).float()
            if opacity.dim() == 3 and opacity.shape[-1] == 1:
                opacity = opacity.squeeze(-1)
            if opacity.dim() != 2:
                raise ValueError(f"opacity must have shape [H,W], got {tuple(opacity.shape)}")

            H, W = int(gt_image.shape[0]), int(gt_image.shape[1])
            valid_loss_mask = self._valid_loss_mask_from_target(target, height=H, width=W)

            # RGB reconstruction (L1 + SSIM) with egocar valid_loss_mask only.
            l1_i = compute_l1_loss_masked(pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None)
            ssim_i = compute_ssim_loss_masked(
                pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None, data_range=1.0
            )
            rgb_i = self.loss_w_l1 * l1_i + self.loss_w_ssim * ssim_i

            # P0: opacity mask supervision using sky_mask (1=non-sky, 0=sky) => gt_occupied = sky_mask
            sky_mask = target.get("sky_mask")
            if sky_mask is None:
                raise ValueError("Stage 3.2 P0 requires target['sky_mask'] (1=non-sky, 0=sky).")
            sm = sky_mask.to(self.device).float()
            if sm.dim() == 3:
                sm = sm.squeeze(-1)
            if sm.shape[0] != H or sm.shape[1] != W:
                raise ValueError(
                    f"target['sky_mask'] must match image shape [H,W], got {tuple(sm.shape)} vs H,W=({H},{W})"
                )
            gt_occupied = sm * valid_loss_mask
            pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
            mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)

            # P1: opacity entropy (masked)
            p = opacity.clamp(1e-6, 1.0 - 1e-6)
            entropy_map = (-p * torch.log(p))
            entropy_i = self.loss_w_opacity_entropy * self._masked_mean(entropy_map, valid_loss_mask)

            total_i = rgb_i + mask_i + entropy_i

            loss_l1_list.append(l1_i)
            loss_ssim_list.append(ssim_i)
            loss_rgb_list.append(rgb_i)
            loss_mask_list.append(mask_i)
            loss_entropy_list.append(entropy_i)
            loss_total_list.append(total_i)

        loss_total = torch.stack(loss_total_list).mean() if loss_total_list else torch.tensor(0.0, device=self.device)
        loss_l1 = torch.stack(loss_l1_list).mean() if loss_l1_list else loss_total * 0.0
        loss_ssim = torch.stack(loss_ssim_list).mean() if loss_ssim_list else loss_total * 0.0
        loss_rgb = torch.stack(loss_rgb_list).mean() if loss_rgb_list else loss_total * 0.0
        loss_mask = torch.stack(loss_mask_list).mean() if loss_mask_list else loss_total * 0.0
        loss_entropy = torch.stack(loss_entropy_list).mean() if loss_entropy_list else loss_total * 0.0

        return {
            "loss": loss_total,
            "loss_l1": loss_l1,
            "loss_ssim": loss_ssim,
            "loss_rgb": loss_rgb,
            "loss_mask": loss_mask,
            "loss_opacity_entropy": loss_entropy,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new_bg,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
            "opacities": opacities,
            "_render_params_distant": render_params_distant,
            "_proxies_distant": proxies_distant,
            "_h_new_distant": h_new_distant,
            "_node_state_distant": node_state_distant,
        }


__all__ = ["MinimalStreetForwardStage3_2"]

