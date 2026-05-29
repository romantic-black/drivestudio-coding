from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .common import VoxelLayout, offsets_to_batch_ids


@dataclass
class SegmentCellIndex:
    point_cell_id: torch.Tensor
    unique_key: torch.Tensor
    indices_bzyx: torch.Tensor
    spatial_shape_xyz: torch.Tensor
    spatial_shape_zyx: torch.Tensor
    cell_center_xyz: torch.Tensor
    voxel_size: float
    aabb_min: torch.Tensor
    aabb_max: torch.Tensor

    @property
    def num_cells(self) -> int:
        return int(self.unique_key.shape[0])


def _spatial_shapes(
    *,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_shape_xyz = torch.floor((aabb_max - aabb_min) / float(voxel_size)).long() + 1
    if bool((spatial_shape_xyz <= 0).any().item()):
        raise RuntimeError("Invalid spatial shape from segment_aabb / voxel_size.")
    spatial_shape_zyx = torch.stack(
        [spatial_shape_xyz[2], spatial_shape_xyz[1], spatial_shape_xyz[0]],
        dim=0,
    ).long()
    return spatial_shape_xyz, spatial_shape_zyx


def _indices_bzyx_from_unique_key(unique_key_bxyz: torch.Tensor) -> torch.Tensor:
    if unique_key_bxyz.numel() == 0:
        return unique_key_bxyz.new_zeros((0, 4), dtype=torch.int32)
    b = unique_key_bxyz[:, 0]
    x = unique_key_bxyz[:, 1]
    y = unique_key_bxyz[:, 2]
    z = unique_key_bxyz[:, 3]
    return torch.stack([b, z, y, x], dim=1).int()


def build_voxel_layout(
    coords: torch.Tensor,
    *,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    voxel_size: float,
    batch_offsets: Optional[torch.Tensor] = None,
    strict_inside: bool = True,
) -> VoxelLayout:
    if coords.dim() != 2 or int(coords.shape[1]) != 3:
        raise ValueError(f"coords must be [N,3], got {tuple(coords.shape)}")
    if not torch.isfinite(coords).all():
        raise RuntimeError("voxel layout coords contain NaN/Inf")
    if float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be > 0.")

    bbx_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
    bbx_max = aabb_max.to(device=coords.device, dtype=coords.dtype)
    if strict_inside:
        outside = ((coords < bbx_min) | (coords > bbx_max)).any(dim=-1)
        if bool(outside.any().item()):
            raise RuntimeError("voxel layout received points outside segment_aabb")

    n = int(coords.shape[0])
    batch_ids = offsets_to_batch_ids(batch_offsets, num_points=n, device=coords.device)
    spatial_shape_xyz, spatial_shape_zyx = _spatial_shapes(
        aabb_min=bbx_min,
        aabb_max=bbx_max,
        voxel_size=float(voxel_size),
    )
    grid_coord_xyz = torch.floor((coords - bbx_min) / float(voxel_size)).long()
    grid_key = torch.cat([batch_ids[:, None], grid_coord_xyz], dim=1)
    unique_key, inverse = torch.unique(grid_key, dim=0, sorted=True, return_inverse=True)
    return VoxelLayout(
        grid_coord_xyz=grid_coord_xyz,
        batch_ids=batch_ids,
        unique_key=unique_key,
        inverse=inverse,
        indices_bzyx=_indices_bzyx_from_unique_key(unique_key),
        spatial_shape_zyx=spatial_shape_zyx,
        spatial_shape_xyz=spatial_shape_xyz,
    )


def build_segment_cell_index(
    coords: torch.Tensor,
    *,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    voxel_size: float,
    batch_offsets: Optional[torch.Tensor] = None,
    strict_inside: bool = True,
    outside_policy: str = "mark_invalid",
) -> SegmentCellIndex:
    if coords.dim() != 2 or int(coords.shape[1]) != 3:
        raise ValueError(f"coords must be [N,3], got {tuple(coords.shape)}")
    if not torch.isfinite(coords).all():
        raise RuntimeError("segment cell index coords contain NaN/Inf")
    if float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be > 0.")

    policy = str(outside_policy)
    if policy not in {"mark_invalid", "raise", "clamp"}:
        raise ValueError("outside_policy must be one of mark_invalid, raise, clamp")

    bbx_min = aabb_min.to(device=coords.device, dtype=coords.dtype)
    bbx_max = aabb_max.to(device=coords.device, dtype=coords.dtype)
    spatial_shape_xyz, spatial_shape_zyx = _spatial_shapes(
        aabb_min=bbx_min,
        aabb_max=bbx_max,
        voxel_size=float(voxel_size),
    )

    outside = ((coords < bbx_min) | (coords > bbx_max)).any(dim=-1)
    if bool(outside.any().item()) and (bool(strict_inside) or policy == "raise"):
        raise RuntimeError("segment cell index received points outside segment_aabb")

    n = int(coords.shape[0])
    batch_ids_all = offsets_to_batch_ids(batch_offsets, num_points=n, device=coords.device)
    inside = ~outside
    coords_for_grid = coords
    if policy == "clamp":
        coords_for_grid = coords.clamp(min=bbx_min, max=bbx_max)
        inside = torch.ones((n,), device=coords.device, dtype=torch.bool)

    grid_all = torch.floor((coords_for_grid - bbx_min) / float(voxel_size)).long()
    point_cell_id = torch.full((n,), -1, device=coords.device, dtype=torch.long)
    if bool(inside.any().item()):
        grid_key = torch.cat([batch_ids_all[inside, None], grid_all[inside]], dim=1)
        unique_key, inverse_inside = torch.unique(grid_key, dim=0, sorted=True, return_inverse=True)
        point_cell_id[inside] = inverse_inside
    else:
        unique_key = torch.zeros((0, 4), device=coords.device, dtype=torch.long)

    cell_grid = unique_key[:, 1:].to(device=coords.device, dtype=coords.dtype)
    cell_center = bbx_min.view(1, 3) + (cell_grid + 0.5) * float(voxel_size)
    return SegmentCellIndex(
        point_cell_id=point_cell_id,
        unique_key=unique_key,
        indices_bzyx=_indices_bzyx_from_unique_key(unique_key),
        spatial_shape_xyz=spatial_shape_xyz,
        spatial_shape_zyx=spatial_shape_zyx,
        cell_center_xyz=cell_center,
        voxel_size=float(voxel_size),
        aabb_min=bbx_min.detach(),
        aabb_max=bbx_max.detach(),
    )


__all__ = ["SegmentCellIndex", "build_segment_cell_index", "build_voxel_layout"]
