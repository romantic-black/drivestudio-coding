from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def normalize_uv_for_grid_sample(
    uv_px: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image_width and image_height must be positive")
    x = (uv_px[..., 0] + 0.5) * (2.0 / float(image_width)) - 1.0
    y = (uv_px[..., 1] + 0.5) * (2.0 / float(image_height)) - 1.0
    return torch.stack([x, y], dim=-1)


def prepare_value_nchw(value_map: torch.Tensor) -> torch.Tensor:
    """Prepare [V,H,W,C] feature maps once for repeated sparse sampling."""

    if value_map.dim() != 4:
        raise ValueError(f"value_map must be [V,H,W,C], got {tuple(value_map.shape)}")
    return value_map.permute(0, 3, 1, 2).contiguous()


def sparse_grid_sample_prepared(
    value_nchw: torch.Tensor,
    uv_px: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample prepared [V,C,H,W] maps at [R,V,K,2] source-image pixel coordinates."""

    if value_nchw.dim() != 4:
        raise ValueError(f"value_nchw must be [V,C,H,W], got {tuple(value_nchw.shape)}")
    if uv_px.dim() != 4 or int(uv_px.shape[-1]) != 2:
        raise ValueError(f"uv_px must be [R,V,K,2], got {tuple(uv_px.shape)}")
    v, c, _h, _w = value_nchw.shape
    r, vu, k, _ = uv_px.shape
    if int(vu) != int(v):
        raise ValueError(f"uv view count mismatch: value_nchw={int(v)} uv={int(vu)}")
    out_chunks = []
    inbound_chunks = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, int(r), chunk):
        end = min(start + chunk, int(r))
        uv_c = uv_px[start:end]
        rows = int(end - start)
        grid = normalize_uv_for_grid_sample(
            uv_c,
            image_height=int(image_height),
            image_width=int(image_width),
        )
        inbound = (
            (grid[..., 0] >= -1.0)
            & (grid[..., 0] <= 1.0)
            & (grid[..., 1] >= -1.0)
            & (grid[..., 1] <= 1.0)
        )
        grid = grid.permute(1, 0, 2, 3).reshape(int(v), rows * int(k), 1, 2)
        sampled = F.grid_sample(
            value_nchw,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(int(v), int(c), rows, int(k)).permute(2, 0, 3, 1).contiguous()
        out_chunks.append(sampled)
        inbound_chunks.append(inbound)
    if out_chunks:
        return torch.cat(out_chunks, dim=0), torch.cat(inbound_chunks, dim=0)
    return (
        value_nchw.new_zeros((0, int(v), int(k), int(c))),
        torch.zeros((0, int(v), int(k)), device=value_nchw.device, dtype=torch.bool),
    )


def chunked_sparse_grid_sample(
    value_map: torch.Tensor,
    uv_px: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample [V,H,W,C] maps at [R,V,K,2] source-image pixel coordinates."""

    return sparse_grid_sample_prepared(
        prepare_value_nchw(value_map),
        uv_px,
        image_height=int(image_height),
        image_width=int(image_width),
        chunk_size=int(chunk_size),
    )


__all__ = [
    "chunked_sparse_grid_sample",
    "normalize_uv_for_grid_sample",
    "prepare_value_nchw",
    "sparse_grid_sample_prepared",
]
