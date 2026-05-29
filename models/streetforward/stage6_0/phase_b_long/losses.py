from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from models.streetforward.stage6_0.local_gs_state import LocalGSState

from .offset_state import PhaseBOffsetState, materialize_phase_b_state


def _cfg_float(cfg: Optional[Dict[str, Any]], key: str, default: float) -> float:
    if not isinstance(cfg, dict):
        return float(default)
    return float(cfg.get(key, default))


def offset_regularization(
    offset: PhaseBOffsetState,
    *,
    weights: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    cfg = dict(weights or {})
    ref = offset.bg_means
    loss = ref.new_tensor(0.0)
    if bool(cfg.get("enable", True)) is False:
        return loss, {"phase_b_long/offset_reg_loss": 0.0}
    bg_means_w = _cfg_float(cfg, "bg_means_l2", 1.0)
    rigid_means_w = _cfg_float(cfg, "rigid_means_l2", 1.0)
    opacity_w = _cfg_float(cfg, "opacity_l2", 0.1)
    scales_w = _cfg_float(cfg, "scales_l2", 0.1)

    loss = loss + float(bg_means_w) * offset.bg_means.float().pow(2).mean()
    loss = loss + float(opacity_w) * offset.bg_opacity_logit.float().pow(2).mean()
    loss = loss + float(scales_w) * offset.bg_scales_log.float().pow(2).mean()
    if offset.distant_scales_log is not None:
        loss = loss + float(scales_w) * offset.distant_scales_log.float().pow(2).mean()
        loss = loss + float(opacity_w) * offset.distant_opacity_logit.float().pow(2).mean()
    if offset.rigid_means_local is not None:
        loss = loss + float(rigid_means_w) * offset.rigid_means_local.float().pow(2).mean()
        loss = loss + float(opacity_w) * offset.rigid_opacity_logit.float().pow(2).mean()
        loss = loss + float(scales_w) * offset.rigid_scales_log.float().pow(2).mean()
    return loss.to(dtype=ref.dtype), {
        "phase_b_long/offset_reg_loss": float(loss.detach().item()),
        **offset.stats(),
    }


def phase_b_long_final_render_loss(
    trainer: Any,
    *,
    base_state: LocalGSState,
    offset: PhaseBOffsetState,
    batch: Dict[str, Any],
    target_indices: List[int],
    role: str,
    rigid_meta: Optional[Dict[str, Any]] = None,
    mask_policy: str = "valid",
    l1_weight: float = 0.8,
    ssim_weight: float = 0.2,
    pred_rgbs_out: Optional[List[torch.Tensor]] = None,
    gt_images_out: Optional[List[torch.Tensor]] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if len(target_indices) == 0:
        return base_state.bg.means.new_tensor(0.0), {
            f"phase_b_long/final_{role}_loss": 0.0,
            f"phase_b_long/final_{role}_psnr": 0.0,
            f"phase_b_long/final_{role}_l1": 0.0,
            f"phase_b_long/final_{role}_ssim": 0.0,
            f"phase_b_long/final_{role}_valid_ratio": 0.0,
            f"phase_b_long/final_{role}_skipped_no_valid_pixels": 0.0,
            f"phase_b_long/final_{role}_num_refs": 0.0,
            f"phase_b_long/offset_rigid_{role}_fallback_rows": 0.0,
        }
    by_frame: Dict[int, List[int]] = defaultdict(list)
    for idx in target_indices:
        target = batch["targets"][int(idx)]
        by_frame[int(target.get("frame_idx", 0))].append(int(idx))

    losses: List[torch.Tensor] = []
    weighted_refs = 0
    psnr_vals: List[float] = []
    l1_vals: List[float] = []
    ssim_vals: List[float] = []
    valid_vals: List[float] = []
    skipped_no_valid_pixels = 0.0
    fallback_rows = 0.0
    for frame_idx, indices in sorted(by_frame.items()):
        frame_state = materialize_phase_b_state(
            base_state=base_state,
            offset=offset,
            target_frame_idx=int(frame_idx),
            rigid_meta=rigid_meta,
        )
        loss_f, stats_f = trainer._render_loss_for_indices(
            local_state=frame_state,
            batch=batch,
            target_indices=indices,
            mask_policy=str(mask_policy),
            l1_weight=float(l1_weight),
            ssim_weight=float(ssim_weight),
            pred_rgbs_out=pred_rgbs_out,
            gt_images_out=gt_images_out,
        )
        losses.append(loss_f * float(len(indices)))
        weighted_refs += int(len(indices))
        psnr_vals.append(float(stats_f.get("psnr", 0.0)))
        l1_vals.append(float(stats_f.get("l1", 0.0)))
        ssim_vals.append(float(stats_f.get("ssim", 0.0)))
        valid_vals.append(float(stats_f.get("valid_ratio", 0.0)))
        skipped_no_valid_pixels += float(stats_f.get("skipped_no_valid_pixels", 0.0))
        fallback_rows += float(getattr(frame_state, "_phase_b_long_rigid_fallback_rows", 0))
    denom = max(int(weighted_refs), 1)
    total = torch.stack(losses).sum() / float(denom)
    stats = {
        f"phase_b_long/final_{role}_loss": float(total.detach().item()),
        f"phase_b_long/final_{role}_psnr": float(sum(psnr_vals) / max(len(psnr_vals), 1)),
        f"phase_b_long/final_{role}_l1": float(sum(l1_vals) / max(len(l1_vals), 1)),
        f"phase_b_long/final_{role}_ssim": float(sum(ssim_vals) / max(len(ssim_vals), 1)),
        f"phase_b_long/final_{role}_valid_ratio": float(sum(valid_vals) / max(len(valid_vals), 1)),
        f"phase_b_long/final_{role}_skipped_no_valid_pixels": float(skipped_no_valid_pixels),
        f"phase_b_long/final_{role}_num_refs": float(len(target_indices)),
        f"phase_b_long/offset_rigid_{role}_fallback_rows": float(fallback_rows),
    }
    return total, stats
