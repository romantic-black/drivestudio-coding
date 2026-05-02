from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Stage6StudentInput:
    render_rgb: torch.Tensor
    prior_map: torch.Tensor
    prior_conf: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    history_context: Optional[torch.Tensor] = None


@dataclass
class MaskedStudentInputs:
    render_rgb: torch.Tensor
    prior_map: torch.Tensor
    prior_conf: Optional[torch.Tensor]
    extra_inputs: Optional[torch.Tensor]
    history_context: Optional[torch.Tensor] = None


def _infer_layout(t: torch.Tensor, *, expected_channels: Optional[int] = None, name: str = "tensor") -> str:
    if t.dim() != 4:
        raise ValueError(f"{name} expects a 4D tensor, got shape={tuple(t.shape)}")
    if expected_channels is not None:
        if int(t.shape[1]) == int(expected_channels):
            return "nchw"
        if int(t.shape[-1]) == int(expected_channels):
            return "nhwc"
        raise ValueError(
            f"{name} expected {int(expected_channels)} channels in NCHW or NHWC layout, got {tuple(t.shape)}"
        )
    raise ValueError(f"{name} layout is ambiguous without expected_channels.")


def _mask_to_layout(mask: torch.Tensor, *, layout: str, height: int, width: int) -> torch.Tensor:
    mask_layout = _infer_layout(mask, expected_channels=1, name="valid_mask")
    mask_nchw = mask if mask_layout == "nchw" else mask.permute(0, 3, 1, 2).contiguous()
    mask_nchw = mask_nchw.float()
    if int(mask_nchw.shape[-2]) != int(height) or int(mask_nchw.shape[-1]) != int(width):
        mask_nchw = F.interpolate(mask_nchw, size=(int(height), int(width)), mode="nearest")
    if layout == "nchw":
        return mask_nchw
    if layout == "nhwc":
        return mask_nchw.permute(0, 2, 3, 1).contiguous()
    raise ValueError(f"unsupported layout={layout!r}")


def _height_width(t: torch.Tensor, *, layout: str) -> tuple[int, int]:
    if layout == "nchw":
        return int(t.shape[-2]), int(t.shape[-1])
    if layout == "nhwc":
        return int(t.shape[1]), int(t.shape[2])
    raise ValueError(f"unsupported layout={layout!r}")


def apply_student_valid_mask(
    *,
    render_rgb: torch.Tensor,
    prior_map: torch.Tensor,
    prior_conf: torch.Tensor,
    valid_mask: torch.Tensor,
    history_context: Optional[torch.Tensor] = None,
    append_as_channel: bool = True,
) -> MaskedStudentInputs:
    """
    Build masked student UNet inputs while preserving NCHW/NHWC layout.
    """

    layout = _infer_layout(render_rgb, expected_channels=3, name="render_rgb")
    render_h, render_w = _height_width(render_rgb, layout=layout)
    vm = _mask_to_layout(valid_mask, layout=layout, height=render_h, width=render_w)
    render_m = render_rgb * vm

    prior_h, prior_w = _height_width(prior_map, layout=layout)
    prior_vm = vm if (prior_h, prior_w) == (render_h, render_w) else _mask_to_layout(
        valid_mask,
        layout=layout,
        height=prior_h,
        width=prior_w,
    )
    prior_m = prior_map * prior_vm

    conf_m = None
    if prior_conf is not None:
        conf_h, conf_w = _height_width(prior_conf, layout=layout)
        conf_vm = vm if (conf_h, conf_w) == (render_h, render_w) else _mask_to_layout(
            valid_mask,
            layout=layout,
            height=conf_h,
            width=conf_w,
        )
        conf_m = prior_conf * conf_vm

    if history_context is not None:
        hist_h, hist_w = _height_width(history_context, layout=layout)
        hm = vm if (hist_h, hist_w) == (render_h, render_w) else _mask_to_layout(
            valid_mask,
            layout=layout,
            height=hist_h,
            width=hist_w,
        )
        history_context = history_context * hm
    return MaskedStudentInputs(
        render_rgb=render_m,
        prior_map=prior_m,
        prior_conf=conf_m,
        extra_inputs=vm if append_as_channel else None,
        history_context=history_context,
    )
