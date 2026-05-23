from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from models.streetforward.metrics import compute_ssim_loss_masked
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _mask_tensor(x: Any, *, device: torch.device) -> torch.Tensor:
    t = x.to(device=device, dtype=torch.float32)
    if t.dim() == 3 and int(t.shape[-1]) == 1:
        t = t.squeeze(-1)
    return t.clamp(0.0, 1.0)


def _combine_mask(mask: Optional[torch.Tensor], component: torch.Tensor) -> torch.Tensor:
    return component if mask is None else mask * component


def target_valid_mask(target: Dict[str, Any], *, mask_policy: str, device: torch.device) -> Optional[torch.Tensor]:
    policy = str(mask_policy)
    if policy not in ("none", "non_sky_non_egocar", "valid_non_sky_non_egocar_non_dynamic"):
        raise ValueError(f"unsupported Stage6 Phase A mask_policy={mask_policy!r}")
    if policy == "none":
        return None
    mask: Optional[torch.Tensor] = None
    valid = target.get("valid_mask")
    if valid is None:
        valid = target.get("mask")
    if valid is None:
        valid = target.get("loss_mask")
    if valid is not None:
        mask = _combine_mask(mask, _mask_tensor(valid, device=device))
    sky = target.get("sky_mask")
    if sky is None:
        raise ValueError(f"Stage6 Phase A mask_policy={policy!r} requires target sky_mask.")
    mask = _combine_mask(mask, 1.0 - _mask_tensor(sky, device=device))
    ego = target.get("egocar_mask")
    if ego is None:
        raise ValueError(f"Stage6 Phase A mask_policy={policy!r} requires target egocar_mask.")
    mask = _combine_mask(mask, 1.0 - _mask_tensor(ego, device=device))
    if policy == "valid_non_sky_non_egocar_non_dynamic":
        dyn = target.get("dynamic_mask")
        if dyn is None:
            raise ValueError("Stage6 Phase A bg-static mask requires target dynamic_mask.")
        mask = _combine_mask(mask, 1.0 - _mask_tensor(dyn, device=device))
    if mask is None:
        raise ValueError(f"Stage6 Phase A mask_policy={policy!r} produced no mask.")
    return mask


def masked_rgb_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    l1_weight: float,
    ssim_weight: float,
    min_valid_pixels: int = 1,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if mask is not None:
        m = mask.to(device=pred.device, dtype=pred.dtype)
        if m.dim() == 2:
            m3 = m[..., None]
        elif m.dim() == 3 and int(m.shape[-1]) == 1:
            m3 = m
        else:
            raise ValueError(f"mask must have shape [H,W] or [H,W,1], got {tuple(m.shape)}")
        valid_pixels = int((m > 0.5).detach().sum().item())
        if valid_pixels < int(min_valid_pixels):
            zero = pred.sum() * 0.0
            return zero, {
                "l1": 0.0,
                "ssim": 0.0,
                "psnr": 0.0,
                "valid_ratio": float((m > 0.5).float().mean().detach().item()),
                "skipped_no_valid_pixels": 1.0,
            }
        denom = m3.sum().clamp_min(1.0) * float(pred.shape[-1])
        l1 = ((pred - gt).abs() * m3).sum() / denom
        valid_ratio = float((m > 0.5).float().mean().detach().item())
    else:
        l1 = F.l1_loss(pred, gt)
        valid_ratio = 1.0
    ssim = pred.new_tensor(0.0)
    if float(ssim_weight) != 0.0:
        ssim = compute_ssim_loss_masked(pred, gt, valid_mask=mask, sky_mask=None, data_range=1.0)
    loss = float(l1_weight) * l1 + float(ssim_weight) * ssim
    if mask is not None:
        err2 = (pred.detach() - gt.detach()) ** 2
        m = mask.to(device=pred.device, dtype=pred.dtype)
        m3 = m[..., None] if m.dim() == 2 else m
        mse = (err2 * m3).sum() / (m3.sum().clamp_min(1.0) * float(pred.shape[-1]))
    else:
        mse = torch.mean((pred.detach() - gt.detach()) ** 2)
    mse = mse.clamp_min(1.0e-12)
    psnr = float((-10.0 * torch.log10(mse)).item())
    return loss, {
        "l1": float(l1.detach().item()),
        "ssim": float(ssim.detach().item()) if torch.is_tensor(ssim) else float(ssim),
        "psnr": psnr,
        "valid_ratio": valid_ratio,
        "skipped_no_valid_pixels": 0.0,
    }


def _branch_delta_l2(delta: Optional[BranchDelta]) -> torch.Tensor:
    if delta is None:
        raise ValueError("delta branch is None")
    vals = [
        delta.means,
        delta.scales_log,
        delta.quat_axis_angle,
        delta.opacity_logit,
        delta.sh,
        delta.hidden,
    ]
    out = vals[0].new_tensor(0.0)
    count = 0
    for val in vals:
        if val.numel() == 0:
            continue
        out = out + val.pow(2).mean()
        count += 1
    return out / max(count, 1)


def _branch_attr_l2(delta: Optional[BranchDelta], attr: str) -> Optional[torch.Tensor]:
    if delta is None:
        return None
    val = getattr(delta, attr)
    if val.numel() == 0:
        return val.new_tensor(0.0)
    return val.pow(2).mean()


def _delta_attr_l2(delta: DeltaPack, attr: str) -> torch.Tensor:
    vals = [_branch_attr_l2(delta.bg, attr)]
    vals.extend([_branch_attr_l2(delta.distant, attr), _branch_attr_l2(delta.rigid, attr)])
    present = [v for v in vals if v is not None]
    if not present:
        return delta.bg.means.new_tensor(0.0)
    return torch.stack(present).mean()


def _branch_scale_barrier(branch: Any, *, scale_log_min: float, scale_log_max: float) -> Optional[torch.Tensor]:
    if branch is None:
        return None
    scales = getattr(branch, "scales_log", None)
    if scales is None:
        return None
    if scales.numel() == 0:
        return scales.new_tensor(0.0)
    hi = F.relu(scales - float(scale_log_max)).pow(2).mean()
    lo = F.relu(float(scale_log_min) - scales).pow(2).mean()
    return hi + lo


def _state_scale_barrier(
    local_state: Any,
    *,
    ref: torch.Tensor,
    scale_log_min: float,
    scale_log_max: float,
) -> torch.Tensor:
    if local_state is None:
        return ref.new_tensor(0.0)
    vals = [
        _branch_scale_barrier(getattr(local_state, "bg", None), scale_log_min=scale_log_min, scale_log_max=scale_log_max),
        _branch_scale_barrier(getattr(local_state, "distant", None), scale_log_min=scale_log_min, scale_log_max=scale_log_max),
        _branch_scale_barrier(getattr(local_state, "rigid", None), scale_log_min=scale_log_min, scale_log_max=scale_log_max),
    ]
    present = [v for v in vals if v is not None]
    if not present:
        return ref.new_tensor(0.0)
    return torch.stack(present).mean()


def delta_regularization(
    delta: DeltaPack,
    *,
    weight: float,
    local_state: Any = None,
    opacity_delta_l2_weight: float = 0.0,
    sh_delta_l2_weight: float = 0.0,
    scale_barrier_weight: float = 0.0,
    scale_log_min: float = -10.0,
    scale_log_max: float = 4.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    reg = _branch_delta_l2(delta.bg)
    branches = 1
    if delta.distant is not None:
        reg = reg + _branch_delta_l2(delta.distant)
        branches += 1
    if delta.rigid is not None:
        reg = reg + _branch_delta_l2(delta.rigid)
        branches += 1
    reg = reg / float(branches)
    opacity_l2 = _delta_attr_l2(delta, "opacity_logit")
    sh_l2 = _delta_attr_l2(delta, "sh")
    scale_barrier = _state_scale_barrier(
        local_state,
        ref=delta.bg.means,
        scale_log_min=float(scale_log_min),
        scale_log_max=float(scale_log_max),
    )
    out = (
        float(weight) * reg
        + float(opacity_delta_l2_weight) * opacity_l2
        + float(sh_delta_l2_weight) * sh_l2
        + float(scale_barrier_weight) * scale_barrier
    )
    return out, {
        "loss_delta_reg": float(out.detach().item()),
        "delta_l2": float(reg.detach().item()),
        "delta_opacity_l2": float(opacity_l2.detach().item()),
        "delta_sh_l2": float(sh_l2.detach().item()),
        "scale_barrier": float(scale_barrier.detach().item()),
    }


def assert_finite_tensors(items: Sequence[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in items:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
