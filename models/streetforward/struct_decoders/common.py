from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat


@dataclass
class StructDecoderInput:
    feat_2d: torch.Tensor
    acc_w: torch.Tensor
    coords: torch.Tensor
    branch_id: torch.Tensor
    params_for_embed: Dict[str, torch.Tensor]
    split_bg: int
    split_rigid_in: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructDecoderOutput:
    feat: torch.Tensor
    aux: Dict[str, Any] = field(default_factory=dict)


class StreetForwardStructDecoderBase(nn.Module):
    def forward(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        raise NotImplementedError


@dataclass
class VoxelLayout:
    grid_coord_xyz: torch.Tensor
    batch_ids: torch.Tensor
    unique_key: torch.Tensor
    inverse: torch.Tensor
    indices_bzyx: torch.Tensor
    spatial_shape_zyx: torch.Tensor
    spatial_shape_xyz: torch.Tensor


def _quat_to_rot6d(quats: torch.Tensor) -> torch.Tensor:
    # Keep Stage5_0 param embedding semantics aligned with Stage4 (OffsetsMixin):
    # flatten first two rotation columns in row-major order.
    rot = _quat_to_rotmat(_normalize_quat(quats))
    return rot[..., :3, :2].reshape(quats.shape[:-1] + (6,))


def normalize_params_for_embed(
    params: Dict[str, torch.Tensor],
    *,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> torch.Tensor:
    means = params["means"]
    scales_log = params["scales_log"]
    quats = params["quats"]
    opacity_logit = params["opacity_logit"]
    sh_dc = params["sh_dc"]
    sh_rest = params["sh_rest"]

    bbx_min = aabb_min.to(device=means.device, dtype=means.dtype)
    bbx_max = aabb_max.to(device=means.device, dtype=means.dtype)
    denom = (bbx_max - bbx_min).clamp(min=1e-6)
    means_norm = (means - bbx_min) / denom * 2.0 - 1.0

    scales_clamped = scales_log.clamp(-10.0, 10.0)
    scales_norm = F.layer_norm(scales_clamped, scales_clamped.shape[1:])
    rot6d = _quat_to_rot6d(quats)
    opacity_norm = torch.tanh(opacity_logit)
    sh_rest_energy = torch.linalg.norm(sh_rest.reshape(sh_rest.shape[0], -1), dim=-1, keepdim=True)

    return torch.cat(
        [
            means_norm,
            rot6d,
            scales_norm,
            opacity_norm,
            sh_dc,
            sh_rest_energy,
        ],
        dim=-1,
    )


def cat_param_dict(
    left: Dict[str, torch.Tensor],
    right: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if right is None:
        return {k: v for k, v in left.items()}
    out: Dict[str, torch.Tensor] = {}
    for key, left_val in left.items():
        if key not in right:
            raise KeyError(f"Missing key in param dict merge: {key}")
        out[key] = torch.cat([left_val, right[key]], dim=0)
    return out


def offsets_to_batch_ids(
    batch_offsets: Optional[torch.Tensor],
    *,
    num_points: int,
    device: torch.device,
) -> torch.Tensor:
    if num_points <= 0:
        return torch.zeros((0,), device=device, dtype=torch.long)
    if batch_offsets is None:
        return torch.zeros((num_points,), device=device, dtype=torch.long)

    offs = batch_offsets.to(device=device, dtype=torch.long).flatten()
    if offs.numel() == 0:
        return torch.zeros((num_points,), device=device, dtype=torch.long)
    if bool((offs[1:] <= offs[:-1]).any().item()):
        raise ValueError("batch_offsets must be strictly increasing cumulative offsets.")
    if int(offs[-1].item()) != int(num_points):
        raise ValueError(f"batch_offsets last value must equal num_points ({num_points}).")

    point_idx = torch.arange(num_points, device=device, dtype=torch.long)
    return torch.bucketize(point_idx, offs, right=True)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim_size: int,
) -> torch.Tensor:
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.shape[-1]))
    if src.dim() != 2:
        raise ValueError("scatter_mean expects src as [N, C].")
    if index.dim() != 1 or index.shape[0] != src.shape[0]:
        raise ValueError("scatter_mean expects index as [N].")
    out = src.new_zeros((dim_size, src.shape[1]))
    out.index_add_(0, index, src)
    ones = src.new_ones((src.shape[0], 1))
    cnt = src.new_zeros((dim_size, 1))
    cnt.index_add_(0, index, ones)
    return out / cnt.clamp(min=1.0)
