from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ParentSerializedLayout:
    order: torch.Tensor
    inverse: torch.Tensor
    pad_mask: torch.Tensor
    patch_size: int
    num_patches: int
    order_name: str
    patch_batch_ids: torch.Tensor

    @property
    def flat_valid(self) -> torch.Tensor:
        return self.pad_mask.reshape(-1)


def _batch_ids_from_offsets(num_rows: int, offsets: Optional[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    if offsets is None:
        return torch.zeros((int(num_rows),), device=device, dtype=torch.long)
    off = offsets.to(device=device, dtype=torch.long).reshape(-1)
    if int(off.numel()) == 0:
        return torch.zeros((int(num_rows),), device=device, dtype=torch.long)
    if int(off[0].item()) != 0:
        off = torch.cat([off.new_zeros((1,)), off], dim=0)
    if int(off[-1].item()) != int(num_rows):
        off = torch.cat([off, off.new_tensor([int(num_rows)])], dim=0)
    ids = torch.empty((int(num_rows),), device=device, dtype=torch.long)
    for b in range(int(off.numel()) - 1):
        ids[int(off[b].item()) : int(off[b + 1].item())] = int(b)
    return ids


def _part1by2(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.long) & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def _morton_key(q: torch.Tensor, *, order_name: str) -> torch.Tensor:
    if str(order_name).lower() == "z_trans":
        q = q[:, [2, 1, 0]]
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def build_parent_serialized_layout(
    coords: torch.Tensor,
    *,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    patch_size: int = 64,
    order_name: str = "z",
    batch_offsets: Optional[torch.Tensor] = None,
    grid_size: int = 1 << 12,
) -> ParentSerializedLayout:
    if coords.dim() != 2 or int(coords.shape[-1]) != 3:
        raise ValueError(f"parent coords must be [N,3], got {tuple(coords.shape)}")
    if int(patch_size) <= 0:
        raise ValueError("parent serialized patch_size must be positive")
    n = int(coords.shape[0])
    device = coords.device
    if n == 0:
        empty_l = torch.zeros((0,), device=device, dtype=torch.long)
        return ParentSerializedLayout(
            order=empty_l,
            inverse=empty_l,
            pad_mask=torch.zeros((0, int(patch_size)), device=device, dtype=torch.bool),
            patch_size=int(patch_size),
            num_patches=0,
            order_name=str(order_name),
            patch_batch_ids=empty_l,
        )
    lo = aabb_min.to(device=device, dtype=coords.dtype).reshape(1, 3)
    hi = aabb_max.to(device=device, dtype=coords.dtype).reshape(1, 3)
    q = ((coords - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    q = torch.floor(q * float(int(grid_size) - 1)).to(dtype=torch.long).clamp(0, int(grid_size) - 1)
    batch_ids = _batch_ids_from_offsets(n, batch_offsets, device=device)
    parts = []
    patch_batch_ids = []
    inverse = torch.empty((n,), device=device, dtype=torch.long)
    flat_pos = 0
    for batch_id in torch.unique(batch_ids, sorted=True).detach().cpu().tolist():
        rows = torch.nonzero(batch_ids == int(batch_id), as_tuple=False).squeeze(1)
        if int(rows.numel()) == 0:
            continue
        key = _morton_key(q.index_select(0, rows), order_name=str(order_name))
        sort = torch.argsort(key, stable=True)
        sorted_rows = rows.index_select(0, sort)
        rem = int(sorted_rows.numel()) % int(patch_size)
        if rem:
            pad = int(patch_size) - rem
            sorted_rows = torch.cat([sorted_rows, sorted_rows.new_full((pad,), int(sorted_rows[-1].item()))], dim=0)
        parts.append(sorted_rows)
        num_p = int(sorted_rows.numel()) // int(patch_size)
        patch_batch_ids.extend([int(batch_id)] * num_p)
        valid_count = int(rows.numel())
        valid_flat = torch.arange(flat_pos, flat_pos + valid_count, device=device, dtype=torch.long)
        inverse.index_copy_(0, sorted_rows[:valid_count], valid_flat)
        flat_pos += int(sorted_rows.numel())
    order = torch.cat(parts, dim=0) if parts else torch.zeros((0,), device=device, dtype=torch.long)
    num_patches = int(order.numel()) // int(patch_size)
    valid_flat = torch.zeros((int(order.numel()),), device=device, dtype=torch.bool)
    if n > 0:
        valid_flat.index_fill_(0, inverse, True)
    patch_batch = torch.tensor(patch_batch_ids, device=device, dtype=torch.long)
    return ParentSerializedLayout(
        order=order,
        inverse=inverse,
        pad_mask=valid_flat.reshape(num_patches, int(patch_size)),
        patch_size=int(patch_size),
        num_patches=num_patches,
        order_name=str(order_name),
        patch_batch_ids=patch_batch,
    )


__all__ = ["ParentSerializedLayout", "build_parent_serialized_layout"]
