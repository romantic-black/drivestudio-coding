"""
Minimal StreetForward Stage 2.1: Multi-target with proxy params and gradient accumulation.

Extends Stage 2.0: same render_params, create proxies, per-view loss_i.backward(), then
_backward_to_render_params to push proxy grads to render_params. Compare with 2.0 for consistency.
See docs/trainers/Minimal_StreetForward_Next_Steps_Stage2_MultiTarget.md.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from models.streetforward.math_utils import _sh_to_rgb
from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.minimal_trainer_stage2_0 import MinimalStreetForwardStage2_0

logger = logging.getLogger(__name__)


def _create_proxy_params(render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Create proxy tensors (detach + requires_grad) for multi-view gradient accumulation."""
    return {
        "means_p": render_params["means_r"].detach().requires_grad_(True),
        "scales_p": render_params["scales_r"].detach().requires_grad_(True),
        "quats_p": render_params["quats_r"].detach().requires_grad_(True),
        "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
        "colors_p": render_params["colors_r"].detach().requires_grad_(True),
    }


def _proxy_params_to_render_dict(proxies: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Build dict acceptable by _render_single_view (means_r, scales_r, ...) from proxy dict."""
    return {
        "means_r": proxies["means_p"],
        "scales_r": proxies["scales_p"],
        "quats_r": proxies["quats_p"],
        "opacities_r": proxies["opacities_p"],
        "colors_r": proxies["colors_p"],
    }


def _backward_to_render_params_bg(
    render_params: Dict[str, torch.Tensor],
    proxies: Dict[str, torch.Tensor],
) -> None:
    """Push proxy gradients to render_params (bg only); then gradients flow to offsets/network."""
    def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
        g = t.grad
        return g if g is not None else torch.zeros_like(t)

    render_tensors = [
        render_params["means_r"],
        render_params["scales_r"],
        render_params["quats_r"],
        render_params["opacities_r"],
        render_params["colors_r"],
    ]
    grad_tensors = [
        _grad_or_zero(proxies["means_p"]),
        _grad_or_zero(proxies["scales_p"]),
        _grad_or_zero(proxies["quats_p"]),
        _grad_or_zero(proxies["opacities_p"]),
        _grad_or_zero(proxies["colors_p"]),
    ]
    torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)


class MinimalStreetForwardStage2_1(MinimalStreetForwardStage2_0):
    """
    Stage 2.1: Multi-target with proxy. One render_params -> create proxies -> per-view
    render with proxy, loss_i/num_targets, loss_i.backward() -> _backward_to_render_params -> step.
    """

    def forward(self, batch: Dict) -> Dict[str, Any]:
        """
        Compute one render_params, create proxies, render each target with proxy,
        loss_i = L1/num_targets, loss_i.backward() (gradients accumulate on proxies).
        Returns pred_rgbs, gt_images for logging; loss is sum of detached per-view losses.
        """
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 2.1 requires at least one target.")

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

        if not self.training:
            # Eval: no proxy, no backward; just render with render_params for test/logging.
            pred_rgbs = []
            gt_images = []
            losses = []
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
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(
                    compute_l1_loss_masked(
                        pred_rgb, gt_image, point_coverage_mask, sky_mask=target.get("sky_mask")
                    )
                )
            loss = torch.stack(losses).mean()
            return {
                "loss": loss,
                "render_params": render_params,
                "proxies": None,
                "_node_state_bg": node_state_bg,
                "_h_new_bg": h_new,
                "_cache_key": key,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
            }

        proxies = _create_proxy_params(render_params)
        params_for_render = _proxy_params_to_render_dict(proxies)

        pred_rgbs = []
        gt_images = []
        n = len(targets)
        total_loss_val = 0.0

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
            pred_rgb, _ = self._render_single_view(params_for_render, view, height, width)
            loss_i = (
                compute_l1_loss_masked(
                    pred_rgb, gt_image, point_coverage_mask, sky_mask=target.get("sky_mask")
                )
                / n
            )
            total_loss_val += loss_i.detach().item()
            loss_i.backward()
            pred_rgbs.append(pred_rgb.detach())
            gt_images.append(gt_image)

        loss = torch.tensor(
            total_loss_val, device=render_params["means_r"].device, dtype=render_params["means_r"].dtype
        )

        return {
            "loss": loss,
            "render_params": render_params,
            "proxies": proxies,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
        }

    def train_step(self, batch: Dict, step: Optional[int] = None) -> Dict[str, Any]:
        """Forward (per-view backward in forward), _backward_to_render_params_bg, step, h_cache, NodeState."""
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        if out.get("proxies") is not None:
            _backward_to_render_params_bg(out["render_params"], out["proxies"])
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
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": int(out["_node_state_bg"].means.shape[0]),
            "num_targets": len(batch.get("targets", [])),
        }


__all__ = ["MinimalStreetForwardStage2_1", "_create_proxy_params", "_backward_to_render_params_bg"]
