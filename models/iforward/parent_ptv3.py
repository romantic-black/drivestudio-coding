from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.struct_decoders.common import scatter_mean
from models.streetforward.struct_decoders.voxel_layout_utils import build_voxel_layout
from models.streetforward.struct_decoders.xcpe_decoder import _SPCONV_AVAILABLE, _XCPEResidualLayer

from .parent_serialization import ParentSerializedLayout, build_parent_serialized_layout


class ParentPTv3Block(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("ParentPTv3Block dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // int(num_heads)
        self.norm1 = nn.LayerNorm(int(dim))
        self.qkv = nn.Linear(int(dim), 3 * int(dim))
        self.proj = nn.Linear(int(dim), int(dim))
        self.norm2 = nn.LayerNorm(int(dim))
        hidden = max(int(round(float(mlp_ratio) * int(dim))), int(dim))
        self.mlp = nn.Sequential(nn.Linear(int(dim), hidden), nn.GELU(), nn.Linear(hidden, int(dim)))
        self.gamma_attn = nn.Parameter(torch.full((int(dim),), float(layer_scale_init)))
        self.gamma_mlp = nn.Parameter(torch.full((int(dim),), float(layer_scale_init)))

    def _patch_attention(self, x: torch.Tensor, layout: ParentSerializedLayout) -> torch.Tensor:
        n = int(x.shape[0])
        if n == 0:
            return x
        p = int(layout.num_patches)
        s = int(layout.patch_size)
        x_pad = x.index_select(0, layout.order.to(device=x.device)).reshape(p, s, self.dim)
        valid = layout.pad_mask.to(device=x.device, dtype=torch.bool)
        x_pad = torch.where(valid.unsqueeze(-1), x_pad, torch.zeros_like(x_pad))
        h = self.norm1(x_pad)
        qkv = self.qkv(h).reshape(p, s, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = torch.zeros((p, 1, 1, s), device=x.device, dtype=x.dtype)
        mask = mask.masked_fill(~valid[:, None, None, :], torch.finfo(x.dtype).min)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(p, s, self.dim)
        attn = self.proj(attn)
        flat = attn.reshape(p * s, self.dim)
        out = flat.index_select(0, layout.inverse.to(device=x.device))
        return out

    def forward(self, x: torch.Tensor, layout: ParentSerializedLayout) -> torch.Tensor:
        x = x + self.gamma_attn.to(dtype=x.dtype) * self._patch_attention(x, layout)
        x = x + self.gamma_mlp.to(dtype=x.dtype) * self.mlp(self.norm2(x))
        if not torch.isfinite(x).all():
            raise RuntimeError("ParentPTv3Block output contains NaN/Inf")
        return x


class ParentPTv3Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: int = 64,
        orders: Iterable[str] = ("z", "z_trans"),
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1.0e-3,
        use_xcpe: bool = True,
        xcpe_kernel_size: int = 3,
        xcpe_voxel_size: float = 0.5,
        xcpe_backend: str = "fallback_neighbor_mean",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.orders = tuple(str(o) for o in orders) or ("z",)
        self.blocks = nn.ModuleList(
            [
                ParentPTv3Block(
                    dim=int(dim),
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    layer_scale_init=float(layer_scale_init),
                )
                for _ in range(int(depth))
            ]
        )
        backend = str(xcpe_backend).lower()
        if backend not in {"spconv", "fallback_neighbor_mean"}:
            raise ValueError("ParentPTv3 xcpe_backend must be 'spconv' or 'fallback_neighbor_mean'")
        if backend == "spconv" and not _SPCONV_AVAILABLE:
            raise ImportError("ParentPTv3 requires spconv when xcpe_backend='spconv'")
        self.use_xcpe = bool(use_xcpe)
        self.xcpe_backend = backend
        self.xcpe_voxel_size = float(xcpe_voxel_size)
        self.xcpe_layers = nn.ModuleList(
            [
                _XCPEResidualLayer(
                    int(dim),
                    kernel_size=int(xcpe_kernel_size),
                    use_spconv=backend == "spconv",
                    norm="layernorm",
                    act="gelu",
                    residual_scale_init=float(layer_scale_init),
                    indice_key=f"ifwd_parent_ptv3_xcpe_{i}",
                )
                for i in range(int(depth))
            ]
        )
        self.out_norm = nn.LayerNorm(int(dim))

    def build_layouts(
        self,
        coords: torch.Tensor,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> Dict[str, ParentSerializedLayout]:
        return {
            order: build_parent_serialized_layout(
                coords,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                patch_size=int(self.patch_size),
                order_name=order,
                batch_offsets=batch_offsets,
            )
            for order in self.orders
        }

    def _apply_xcpe(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor],
        layer: _XCPEResidualLayer,
    ) -> torch.Tensor:
        if int(x.shape[0]) == 0:
            return x
        layout = build_voxel_layout(
            coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            voxel_size=float(self.xcpe_voxel_size),
            batch_offsets=batch_offsets,
            strict_inside=True,
        )
        voxel_feat = scatter_mean(x, layout.inverse, dim_size=int(layout.unique_key.shape[0]))
        voxel_delta = layer(
            voxel_feat=voxel_feat,
            unique_key_bxyz=layout.unique_key,
            indices_bzyx=layout.indices_bzyx,
            spatial_shape_zyx=layout.spatial_shape_zyx,
            batch_size=int(batch_offsets.numel()) if batch_offsets is not None else 1,
            debug_check_output_order=False,
        )
        return x + layer.residual_scale.to(dtype=x.dtype) * voxel_delta[layout.inverse]

    def forward(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
        layout_cache: Optional[Dict[str, ParentSerializedLayout]] = None,
    ) -> tuple[torch.Tensor, Dict[str, ParentSerializedLayout], Dict[str, Any]]:
        if x.dim() != 2 or int(x.shape[-1]) != int(self.dim):
            raise ValueError(f"ParentPTv3 expected x [N,{self.dim}], got {tuple(x.shape)}")
        if coords.dim() != 2 or tuple(coords.shape) != (int(x.shape[0]), 3):
            raise ValueError(
                "ParentPTv3 token/coordinate row mismatch: "
                f"x={tuple(x.shape)}, coords={tuple(coords.shape)}"
            )
        layouts = dict(layout_cache or {})
        n = int(x.shape[0])
        for order_name in tuple(self.orders):
            layout = layouts.get(order_name)
            if layout is None:
                continue
            order = layout.order
            inverse = layout.inverse
            expected_flat = int(layout.num_patches) * int(layout.patch_size)
            compatible = (
                int(layout.patch_size) == int(self.patch_size)
                and int(order.numel()) == expected_flat
                and int(inverse.numel()) == n
                and tuple(layout.pad_mask.shape)
                == (int(layout.num_patches), int(layout.patch_size))
                and (n == 0 or bool(((order >= 0) & (order < n)).all().item()))
                and (
                    n == 0
                    or bool(((inverse >= 0) & (inverse < max(expected_flat, 1))).all().item())
                )
            )
            if not bool(compatible):
                layouts.pop(order_name, None)
        missing = [order for order in self.orders if order not in layouts]
        if missing:
            layouts.update(
                self.build_layouts(
                    coords,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    batch_offsets=batch_offsets,
                )
            )
        h = x
        for idx, block in enumerate(self.blocks):
            order = self.orders[int(idx) % len(self.orders)]
            h = block(h, layouts[order])
            if self.use_xcpe:
                h = self._apply_xcpe(
                    h,
                    coords=coords,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    batch_offsets=batch_offsets,
                    layer=self.xcpe_layers[int(idx)],
                )
        h = self.out_norm(h)
        return h, layouts, {
            "iforward/parent_ptv3/patch_size": float(self.patch_size),
            "iforward/parent_ptv3/num_patches": float(next(iter(layouts.values())).num_patches if layouts else 0),
        }


__all__ = ["ParentPTv3Block", "ParentPTv3Encoder"]
