"""
Extractor v2: fused single-view rasterize + backproject path.

This module keeps the same streaming contract as v1 but removes explicit
materialization of (gaussian_id, pixel_id, weight) tuples when fused CUDA op
is available.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

from models.feature_extractors.alpha_t_extractor import AlphaTWeightExtractor, _get_viewmat

try:
    from gsplat.cuda._wrapper import (
        rasterize_and_backproject_in_range,
        backproject_feature_grad_in_range,
    )
except Exception:  # pragma: no cover
    rasterize_and_backproject_in_range = None
    backproject_feature_grad_in_range = None

if TYPE_CHECKING:
    from models.feature_extractors.feature_2d_backprojector import FeatureBackprojector


class _RasterizeAndBackprojectFeatOnlyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        isect_offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        packed_to_global_gaussian_ids: torch.Tensor,
        feat2d: torch.Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        num_gaussians: int,
        weight_threshold: float,
        return_support: bool,
    ):
        feat2d_c = feat2d.contiguous()
        transmittances = torch.ones(
            (image_height, image_width),
            device=means2d.device,
            dtype=means2d.dtype,
        )
        out = rasterize_and_backproject_in_range(
            range_start=0,
            range_end=int(1e9),
            transmittances=transmittances,
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            packed_to_global_gaussian_ids=packed_to_global_gaussian_ids,
            feat2d=feat2d_c,
            num_gaussians=num_gaussians,
            weight_threshold=weight_threshold,
            return_support=return_support,
        )
        feat_sum, w_feat, w_sup, pairs_total, pairs_kept = out
        ctx.save_for_backward(
            means2d,
            conics,
            opacities,
            isect_offsets,
            flatten_ids,
            packed_to_global_gaussian_ids,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tile_size = int(tile_size)
        ctx.weight_threshold = float(weight_threshold)
        ctx.feat_h = int(feat2d_c.shape[0])
        ctx.feat_w = int(feat2d_c.shape[1])
        ctx.channels = int(feat2d_c.shape[2])
        ctx.mark_non_differentiable(w_feat, w_sup, pairs_total, pairs_kept)
        return feat_sum, w_feat, w_sup, pairs_total, pairs_kept

    @staticmethod
    def backward(ctx, grad_feat_sum, grad_w_feat, grad_w_sup, grad_pairs_total, grad_pairs_kept):
        del grad_w_feat, grad_w_sup, grad_pairs_total, grad_pairs_kept
        if grad_feat_sum is None:
            grad_feat2d = None
        else:
            means2d, conics, opacities, isect_offsets, flatten_ids, packed_to_global = ctx.saved_tensors
            transmittances = torch.ones(
                (ctx.image_height, ctx.image_width),
                device=means2d.device,
                dtype=means2d.dtype,
            )
            grad_feat2d = backproject_feature_grad_in_range(
                range_start=0,
                range_end=int(1e9),
                transmittances=transmittances,
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                image_width=ctx.image_width,
                image_height=ctx.image_height,
                tile_size=ctx.tile_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                packed_to_global_gaussian_ids=packed_to_global,
                grad_feat_sum=grad_feat_sum.contiguous(),
                feat_h=ctx.feat_h,
                feat_w=ctx.feat_w,
                channels=ctx.channels,
                weight_threshold=ctx.weight_threshold,
            )
        return (
            None, None, None, None, None, None,
            grad_feat2d,
            None, None, None, None, None, None,
        )


class AlphaTWeightExtractorV2(AlphaTWeightExtractor):
    """Extractor v2 with strict fused backprojection (fast-fail)."""

    @property
    def fused_available(self) -> bool:
        return rasterize_and_backproject_in_range is not None

    def extract_single_weight_fused(
        self,
        meta: Dict[str, torch.Tensor],
        feat_2d: torch.Tensor,
        height: int,
        width: int,
        num_gaussians: int,
        weight_threshold: float,
        return_support_weight: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if meta is None:
            c = int(feat_2d.shape[-1])
            zf = torch.zeros(num_gaussians, c, device=feat_2d.device, dtype=feat_2d.dtype)
            zw = torch.zeros(num_gaussians, device=feat_2d.device, dtype=feat_2d.dtype)
            return zf, zw, zw, 0, 0

        if "gaussian_ids" not in meta or meta["gaussian_ids"] is None:
            raise ValueError("Packed render meta missing gaussian_ids for local->global remap.")
        packed_to_global = meta["gaussian_ids"]
        if not torch.is_tensor(packed_to_global):
            raise TypeError("Packed render meta gaussian_ids must be a tensor.")
        if packed_to_global.dtype != torch.int64:
            raise TypeError(
                f"Packed render meta gaussian_ids must be int64 for fused path, got {packed_to_global.dtype}."
            )
        if packed_to_global.device != meta["means2d"].device:
            raise ValueError("Packed gaussian_ids device mismatch with means2d device.")
        if packed_to_global.numel() != meta["means2d"].shape[0]:
            raise ValueError(
                "Packed gaussian_ids size mismatch: "
                f"{packed_to_global.numel()} vs means2d N={meta['means2d'].shape[0]}."
            )
        if packed_to_global.numel() > 0:
            g_min = int(packed_to_global.min().item())
            g_max = int(packed_to_global.max().item())
            if g_min < 0:
                raise ValueError(f"Packed global gaussian id is negative: min={g_min}.")
            if g_max >= int(num_gaussians):
                raise ValueError(
                    f"Packed global gaussian id out of range: max={g_max}, num_gaussians={num_gaussians}."
                )
        if not torch.is_tensor(meta["flatten_ids"]) or meta["flatten_ids"].dtype != torch.int32:
            raise TypeError("meta['flatten_ids'] must be int32 tensor for fused path.")
        if rasterize_and_backproject_in_range is None:
            raise RuntimeError("fused op is unavailable")

        isect_offsets_raw = meta["isect_offsets"]
        image_dims = meta["means2d"].shape[:-2]
        if len(isect_offsets_raw.shape) > len(image_dims) + 2:
            n_dims_to_remove = len(isect_offsets_raw.shape) - len(image_dims) - 2
            for _ in range(n_dims_to_remove):
                if isect_offsets_raw.shape[0] == 1:
                    isect_offsets_raw = isect_offsets_raw.squeeze(0)
                else:
                    break

        if feat_2d.requires_grad:
            if backproject_feature_grad_in_range is None:
                raise RuntimeError("fused backward op is unavailable")
            (
                feat_sum,
                weight_sum_feature,
                weight_sum_support,
                pair_count_total,
                pair_count_after_threshold,
            ) = _RasterizeAndBackprojectFeatOnlyFn.apply(
                meta["means2d"],
                meta["conics"],
                meta["opacities"],
                isect_offsets_raw,
                meta["flatten_ids"],
                packed_to_global,
                feat_2d,
                int(width),
                int(height),
                int(meta.get("tile_size", 16)),
                int(num_gaussians),
                float(weight_threshold),
                bool(return_support_weight),
            )
        else:
            transmittances = torch.ones(
                (height, width), device=meta["means2d"].device, dtype=meta["means2d"].dtype
            )
            (
                feat_sum,
                weight_sum_feature,
                weight_sum_support,
                pair_count_total,
                pair_count_after_threshold,
            ) = rasterize_and_backproject_in_range(
                range_start=0,
                range_end=int(1e9),
                transmittances=transmittances,
                means2d=meta["means2d"],
                conics=meta["conics"],
                opacities=meta["opacities"],
                image_width=width,
                image_height=height,
                tile_size=int(meta.get("tile_size", 16)),
                isect_offsets=isect_offsets_raw,
                flatten_ids=meta["flatten_ids"],
                packed_to_global_gaussian_ids=packed_to_global,
                feat2d=feat_2d,
                num_gaussians=int(num_gaussians),
                weight_threshold=float(weight_threshold),
                return_support=bool(return_support_weight),
            )
        total_pairs = int(pair_count_total.item())
        kept_pairs = int(pair_count_after_threshold.item())
        return feat_sum, weight_sum_feature, weight_sum_support, total_pairs, kept_pairs

    def render_and_backproject_streaming_fused(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        features_2d: torch.Tensor,
        height: int,
        width: int,
        num_gaussians: int,
        backprojector: "FeatureBackprojector",
        return_accumulated_weights: bool = False,
        return_debug_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        if not self.fused_available:
            raise RuntimeError("ExtractorV2 fast-fail: fused op is unavailable in current gsplat build.")
        if not features_2d.is_cuda:
            raise RuntimeError("ExtractorV2 fast-fail: features_2d must be CUDA tensor.")
        if features_2d.requires_grad and backproject_feature_grad_in_range is None:
            raise RuntimeError("ExtractorV2 fast-fail: fused backward op is unavailable.")
        orig_dtype = features_2d.dtype
        if orig_dtype != torch.float32:
            features_2d = features_2d.float()

        t_total_start = time.perf_counter()
        device = features_2d.device
        channels = features_2d.shape[-1]
        eps = getattr(backprojector, "eps", 1e-8)
        weight_threshold = float(getattr(backprojector, "weight_threshold", 0.0))

        accumulated_feat = torch.zeros(num_gaussians, channels, device=device, dtype=features_2d.dtype)
        accumulated_weight_feature = torch.zeros(num_gaussians, device=device, dtype=features_2d.dtype)
        accumulated_weight_support = (
            torch.zeros(num_gaussians, device=device, dtype=features_2d.dtype) if return_accumulated_weights else None
        )

        stats = {
            "render_packed_total_ms": 0.0,
            "fused_backproject_total_ms": 0.0,
            "pairs_total": 0,
            "pairs_after_threshold": 0,
            "num_views": int(len(cameras)),
            "num_gaussians": int(num_gaussians),
        }

        for i, cam in enumerate(cameras):
            cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
            viewmat = _get_viewmat(cam_ctw)
            k_mat = self._resolve_intrinsics(cam)

            t_render = time.perf_counter()
            with torch.no_grad():
                _, _, meta = self.renderer(
                    means=gaussians["means"],
                    quats=gaussians["quats"],
                    scales=gaussians["scales"],
                    opacities=gaussians["opacities"],
                    colors=gaussians["colors"],
                    viewmats=viewmat,
                    Ks=k_mat,
                    width=width,
                    height=height,
                    tile_size=self.tile_size,
                    packed=True,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=self.sh_degree,
                    sparse_grad=False,
                    absgrad=True,
                    rasterize_mode="classic",
                )
            stats["render_packed_total_ms"] += float((time.perf_counter() - t_render) * 1000.0)

            t_fused = time.perf_counter()
            feat_sum, weight_sum_feature, weight_sum_support, pairs_total, pairs_kept = self.extract_single_weight_fused(
                meta=meta,
                feat_2d=features_2d[i],
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                weight_threshold=weight_threshold,
                return_support_weight=return_accumulated_weights,
            )
            stats["fused_backproject_total_ms"] += float((time.perf_counter() - t_fused) * 1000.0)
            stats["pairs_total"] += int(pairs_total)
            stats["pairs_after_threshold"] += int(pairs_kept)

            accumulated_feat = accumulated_feat + feat_sum
            accumulated_weight_feature = accumulated_weight_feature + weight_sum_feature
            if return_accumulated_weights:
                if accumulated_weight_support is None:
                    raise RuntimeError("Internal error: accumulated_weight_support is None.")
                accumulated_weight_support = accumulated_weight_support + weight_sum_support

        feat_out = accumulated_feat / (accumulated_weight_feature.unsqueeze(-1) + eps)
        stats["streaming_total_ms"] = float((time.perf_counter() - t_total_start) * 1000.0)
        if orig_dtype != torch.float32:
            feat_out = feat_out.to(orig_dtype)
        if return_accumulated_weights:
            if accumulated_weight_support is None:
                raise RuntimeError("Internal error: accumulated_weight_support is None.")
            if orig_dtype != torch.float32:
                accumulated_weight_support = accumulated_weight_support.to(orig_dtype)
            if return_debug_stats:
                return feat_out, accumulated_weight_support, stats
            return feat_out, accumulated_weight_support
        if return_debug_stats:
            return feat_out, stats
        return feat_out
