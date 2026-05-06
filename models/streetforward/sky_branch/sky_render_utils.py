from __future__ import annotations

from typing import Any, List, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors.alpha_t_extractor import _get_viewmat


def get_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def ensure_hwc3(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dim() == 3 and int(x.shape[-1]) == 3:
        return x
    if x.dim() == 3 and int(x.shape[0]) == 3:
        return x.permute(1, 2, 0).contiguous()
    raise ValueError(f"{name} must be [H,W,3] or [3,H,W], got {tuple(x.shape)}")


def ensure_hw1(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3 and int(x.shape[-1]) == 1:
        return x
    if x.dim() == 3 and int(x.shape[0]) == 1:
        return x.permute(1, 2, 0).contiguous()
    raise ValueError(f"{name} must be [H,W], [H,W,1], or [1,H,W], got {tuple(x.shape)}")


def stack_hwc3(items: List[torch.Tensor], *, name: str) -> torch.Tensor:
    if len(items) == 0:
        raise ValueError(f"{name} must be non-empty.")
    out = [ensure_hwc3(x, name=f"{name}[{i}]") for i, x in enumerate(items)]
    hw = tuple(out[0].shape[:2])
    for i, x in enumerate(out):
        if tuple(x.shape[:2]) != hw:
            raise ValueError(f"{name} must share H/W across views. idx={i} has {tuple(x.shape[:2])}, expected {hw}.")
    return torch.stack(out, dim=0)


def stack_hw1(items: List[torch.Tensor], *, name: str) -> torch.Tensor:
    if len(items) == 0:
        raise ValueError(f"{name} must be non-empty.")
    out = [ensure_hw1(x, name=f"{name}[{i}]") for i, x in enumerate(items)]
    hw = tuple(out[0].shape[:2])
    for i, x in enumerate(out):
        if tuple(x.shape[:2]) != hw:
            raise ValueError(f"{name} must share H/W across views. idx={i} has {tuple(x.shape[:2])}, expected {hw}.")
    return torch.stack(out, dim=0)


def squeeze_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.dim() == 3 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.dim() == 3 and int(mask.shape[0]) == 1:
        mask = mask[0]
    if mask.dim() != 2:
        raise ValueError(f"{name} must be [H,W] or singleton-channel image mask, got {tuple(mask.shape)}")
    return mask


def erode_binary_mask(mask_vhw: torch.Tensor, kernel: int) -> torch.Tensor:
    if mask_vhw.dim() != 3:
        raise ValueError(f"mask_vhw must be [V,H,W], got {tuple(mask_vhw.shape)}")
    if int(kernel) <= 1:
        return (mask_vhw > 0.5).to(dtype=mask_vhw.dtype)
    if int(kernel) % 2 != 1:
        raise ValueError("erosion kernel must be odd.")
    x = (mask_vhw > 0.5).to(dtype=torch.float32).unsqueeze(1)
    inv = 1.0 - x
    pad = int(kernel) // 2
    eroded = 1.0 - F.max_pool2d(inv, kernel_size=int(kernel), stride=1, padding=pad)
    return eroded[:, 0].to(dtype=mask_vhw.dtype).clamp(0.0, 1.0)


def build_sky_regions(sky_mask_vhw: torch.Tensor, *, erode_kernel: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sky = sky_mask_vhw.float().clamp(0.0, 1.0)
    sky_core = erode_binary_mask(sky, int(erode_kernel))
    sky_boundary = (sky - sky_core).clamp(0.0, 1.0)
    non_sky = (1.0 - sky).clamp(0.0, 1.0)
    return sky_core, sky_boundary, non_sky


def rotation_only_viewmat_from_view(view: Any) -> torch.Tensor:
    c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    c2w_rot = c2w.clone()
    c2w_rot[..., :3, 3] = 0.0
    return _get_viewmat(c2w_rot)


def resolve_view_intrinsics(view: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if hasattr(view, "Ks"):
        k_mat = view.Ks
    elif hasattr(view, "K"):
        k_mat = view.K
    else:
        k_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    if k_mat.dim() == 2:
        k_mat = k_mat.unsqueeze(0)
    return k_mat.to(device=device, dtype=dtype)
