"""
Extractor v3: fused multi-source streaming backprojection.

v3 keeps v2 fused numerical semantics:
- single packed multi-camera render + fused backproject
- single global normalization at the end
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from models.feature_extractors.alpha_t_extractor import _get_viewmat
from models.feature_extractors.alpha_t_extractor_v2 import AlphaTWeightExtractorV2, backproject_feature_grad_in_range
try:
    from gsplat.cuda._wrapper import (
        backproject_feature_grad_multi_camera_sharded_in_range,
        rasterize_and_backproject_multi_camera_in_range,
    )
except Exception:  # pragma: no cover
    backproject_feature_grad_multi_camera_sharded_in_range = None
    rasterize_and_backproject_multi_camera_in_range = None


class _RasterizeAndBackprojectFeatOnlyMultiCamFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        isect_offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        packed_global_gaussian_ids: torch.Tensor,
        feat2d: torch.Tensor,
        image_width: int,
        image_height: int,
        tile_size: int,
        num_gaussians: int,
        pair_valid_mask: Optional[torch.Tensor],
        weight_threshold: float,
        return_support: bool,
    ):
        feat2d_c = feat2d.contiguous()
        pair_valid_mask_c = None
        if pair_valid_mask is not None:
            pair_valid_mask_c = pair_valid_mask.contiguous()
        out = rasterize_and_backproject_multi_camera_in_range(
            range_start=0,
            range_end=int(1e9),
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            packed_global_gaussian_ids=packed_global_gaussian_ids,
            feat2d=feat2d_c,
            num_gaussians=num_gaussians,
            pair_valid_mask=pair_valid_mask_c,
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
            packed_global_gaussian_ids,
        )
        ctx.image_width = int(image_width)
        ctx.image_height = int(image_height)
        ctx.tile_size = int(tile_size)
        ctx.weight_threshold = float(weight_threshold)
        ctx.feat_h = int(feat2d_c.shape[1])
        ctx.feat_w = int(feat2d_c.shape[2])
        ctx.channels = int(feat2d_c.shape[3])
        ctx.pair_valid_mask = pair_valid_mask_c
        ctx.mark_non_differentiable(w_feat, w_sup, pairs_total, pairs_kept)
        return feat_sum, w_feat, w_sup, pairs_total, pairs_kept

    @staticmethod
    def backward(ctx, grad_feat_sum, grad_w_feat, grad_w_sup, grad_pairs_total, grad_pairs_kept):
        del grad_w_feat, grad_w_sup, grad_pairs_total, grad_pairs_kept
        if grad_feat_sum is None:
            grad_feat2d = None
        else:
            means2d, conics, opacities, isect_offsets, flatten_ids, packed_global = ctx.saved_tensors
            grad_feat2d = backproject_feature_grad_multi_camera_sharded_in_range(
                range_start=0,
                range_end=int(1e9),
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                image_width=ctx.image_width,
                image_height=ctx.image_height,
                tile_size=ctx.tile_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                packed_global_gaussian_ids=packed_global,
                grad_feat_sum=grad_feat_sum.contiguous(),
                feat_h=ctx.feat_h,
                feat_w=ctx.feat_w,
                channels=ctx.channels,
                pair_valid_mask=ctx.pair_valid_mask,
                weight_threshold=ctx.weight_threshold,
            )
        return (
            None, None, None, None, None, None,
            grad_feat2d,
            None, None, None, None, None, None, None,
        )


class AlphaTWeightExtractorV3(AlphaTWeightExtractorV2):
    """Extractor v3 with explicit multi-src contract on top of v2 fused path."""

    @property
    def fused_multi_camera_available(self) -> bool:
        return (
            rasterize_and_backproject_multi_camera_in_range is not None
            and backproject_feature_grad_multi_camera_sharded_in_range is not None
        )

    def _render_single_view_packed_meta(
        self,
        gaussians: Dict[str, torch.Tensor],
        cam,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
        viewmat = _get_viewmat(cam_ctw)
        k_mat = self._resolve_intrinsics(cam)
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
        if meta is None:
            raise RuntimeError("Packed render returned None meta in multi-camera v4 path.")
        return meta

    @staticmethod
    def _normalize_isect_offsets(isect_offsets_raw: torch.Tensor, image_dims: torch.Size) -> torch.Tensor:
        isect_offsets = isect_offsets_raw
        if len(isect_offsets.shape) > len(image_dims) + 2:
            n_dims_to_remove = len(isect_offsets.shape) - len(image_dims) - 2
            for _ in range(n_dims_to_remove):
                if isect_offsets.shape[0] == 1:
                    isect_offsets = isect_offsets.squeeze(0)
                else:
                    break
        return isect_offsets

    def _build_multi_camera_meta_from_viewmats(
        self,
        gaussians: Dict[str, torch.Tensor],
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        t_start = time.perf_counter()
        if viewmats.dim() == 2:
            viewmats = viewmats.unsqueeze(0)
        if Ks.dim() == 2:
            Ks = Ks.unsqueeze(0)
        if int(viewmats.shape[0]) < 1:
            raise ValueError("Multi-camera meta builder requires at least one camera.")
        if int(Ks.shape[0]) != int(viewmats.shape[0]):
            raise ValueError(
                f"Ks/viewmats first dim mismatch: {Ks.shape[0]} vs {viewmats.shape[0]}."
            )

        t_render = time.perf_counter()
        with torch.no_grad():
            _, _, meta = self.renderer(
                means=gaussians["means"],
                quats=gaussians["quats"],
                scales=gaussians["scales"],
                opacities=gaussians["opacities"],
                colors=gaussians["colors"],
                viewmats=viewmats,
                Ks=Ks,
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
        total_render_ms = float((time.perf_counter() - t_render) * 1000.0)
        if meta is None:
            raise RuntimeError("Packed render returned None meta in multi-camera v4 path.")

        means2d = meta["means2d"]
        conics = meta["conics"]
        opacities = meta["opacities"]
        packed_global_ids = meta.get("gaussian_ids", None)
        if packed_global_ids is None:
            raise ValueError("Packed render meta missing gaussian_ids for multi-camera path.")
        if packed_global_ids.dtype != torch.int64:
            packed_global_ids = packed_global_ids.to(torch.int64)

        flatten_ids = meta["flatten_ids"]
        tile_offsets = meta["isect_offsets"]
        while tile_offsets.dim() > 3 and tile_offsets.shape[0] == 1:
            tile_offsets = tile_offsets.squeeze(0)
        if tile_offsets.dim() != 3:
            raise ValueError(f"isect_offsets must be rank-3 [V, tile_h, tile_w], got {tile_offsets.shape}")
        if int(tile_offsets.shape[0]) != int(viewmats.shape[0]):
            raise ValueError(
                f"isect_offsets first dim ({tile_offsets.shape[0]}) must equal num cameras ({viewmats.shape[0]})."
            )
        if tile_offsets.dtype != torch.int32:
            tile_offsets = tile_offsets.to(torch.int32)
        if flatten_ids.dtype != torch.int32:
            flatten_ids = flatten_ids.to(torch.int32)

        meta_out = {
            "means2d": means2d,
            "conics": conics,
            "opacities": opacities,
            "packed_global_gaussian_ids": packed_global_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": tile_offsets,
            "tile_size": int(self.tile_size),
        }
        stats = {
            "render_packed_total_ms": total_render_ms,
            "build_multi_meta_ms": float((time.perf_counter() - t_start) * 1000.0),
            "nnz_total": int(means2d.shape[0]),
            "isects_total": int(flatten_ids.numel()),
        }
        return meta_out, stats

    def _build_multi_camera_meta_from_views(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        height: int,
        width: int,
        viewmats_override: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        if len(cameras) < 1:
            raise ValueError("Multi-camera meta builder requires at least one camera.")
        ks_list = [self._resolve_intrinsics(cam) for cam in cameras]
        Ks = torch.cat(ks_list, dim=0)
        if viewmats_override is None:
            viewmats_list = []
            for cam in cameras:
                cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
                viewmats_list.append(_get_viewmat(cam_ctw))
            viewmats = torch.cat(viewmats_list, dim=0)
        else:
            viewmats = viewmats_override
            if viewmats.dim() == 2:
                viewmats = viewmats.unsqueeze(0)
            if int(viewmats.shape[0]) != int(len(cameras)):
                raise ValueError(
                    f"viewmats_override.shape[0] ({viewmats.shape[0]}) must equal len(cameras) ({len(cameras)})."
                )
        return self._build_multi_camera_meta_from_viewmats(
            gaussians=gaussians,
            viewmats=viewmats,
            Ks=Ks,
            height=height,
            width=width,
        )

    def render_and_backproject_streaming_fused_multi_camera(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        features_2d: torch.Tensor,
        height: int,
        width: int,
        num_gaussians: int,
        backprojector,
        viewmats_override: Optional[torch.Tensor] = None,
        source_pair_valid_mask: Optional[torch.Tensor] = None,
        return_accumulated_weights: bool = False,
        return_debug_stats: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
    ]:
        if not self.fused_multi_camera_available:
            raise RuntimeError("ExtractorV3 fast-fail: multi-camera fused op is unavailable in current gsplat build.")
        if not features_2d.is_cuda:
            raise RuntimeError("ExtractorV3 fast-fail: features_2d must be CUDA tensor.")
        if features_2d.ndim != 4:
            raise ValueError(
                "ExtractorV3 multi-camera expects features_2d with shape [V, Hf, Wf, C], "
                f"got ndim={features_2d.ndim}."
            )
        if len(cameras) < 1:
            raise ValueError("ExtractorV3 multi-camera requires at least one camera.")
        if int(features_2d.shape[0]) != int(len(cameras)):
            raise ValueError(
                f"features_2d first dim ({features_2d.shape[0]}) must equal num cameras ({len(cameras)})."
            )

        t_total_start = time.perf_counter()
        orig_dtype = features_2d.dtype
        if orig_dtype != torch.float32:
            features_2d = features_2d.float()
        eps = float(getattr(backprojector, "eps", 1e-8))
        weight_threshold = float(getattr(backprojector, "weight_threshold", 0.0))
        pair_valid_mask: Optional[torch.Tensor] = None
        masked_pixel_count = 0
        valid_pixel_count = 0
        source_pair_valid_ratio = 1.0
        if source_pair_valid_mask is not None:
            if source_pair_valid_mask.dim() != 3:
                raise ValueError(
                    "source_pair_valid_mask must have shape [V, H, W], "
                    f"got {tuple(source_pair_valid_mask.shape)}."
                )
            if int(source_pair_valid_mask.shape[0]) != int(len(cameras)):
                raise ValueError(
                    f"source_pair_valid_mask.shape[0] ({source_pair_valid_mask.shape[0]}) "
                    f"must equal len(cameras) ({len(cameras)})."
                )
            if int(source_pair_valid_mask.shape[1]) != int(height) or int(source_pair_valid_mask.shape[2]) != int(width):
                raise ValueError(
                    "source_pair_valid_mask spatial shape mismatch with source render size: "
                    f"expected ({height}, {width}), got ({source_pair_valid_mask.shape[1]}, {source_pair_valid_mask.shape[2]})."
                )
            pair_valid_mask = source_pair_valid_mask.to(device=features_2d.device)
            if pair_valid_mask.dtype != torch.bool:
                pair_valid_mask = pair_valid_mask > 0.5
            pair_valid_mask = pair_valid_mask.contiguous()
            valid_pixel_count = int(pair_valid_mask.sum().item())
            total_pixel_count = int(pair_valid_mask.numel())
            masked_pixel_count = int(total_pixel_count - valid_pixel_count)
            source_pair_valid_ratio = float(valid_pixel_count / max(total_pixel_count, 1))

        meta, meta_stats = self._build_multi_camera_meta_from_views(
            gaussians=gaussians,
            cameras=cameras,
            height=height,
            width=width,
            viewmats_override=viewmats_override,
        )

        t_fused = time.perf_counter()
        if features_2d.requires_grad:
            (
                feat_sum,
                weight_sum_feature,
                weight_sum_support,
                pair_count_total,
                pair_count_after_threshold,
            ) = _RasterizeAndBackprojectFeatOnlyMultiCamFn.apply(
                meta["means2d"],
                meta["conics"],
                meta["opacities"],
                meta["isect_offsets"],
                meta["flatten_ids"],
                meta["packed_global_gaussian_ids"],
                features_2d,
                int(width),
                int(height),
                int(meta.get("tile_size", 16)),
                int(num_gaussians),
                pair_valid_mask,
                float(weight_threshold),
                bool(return_accumulated_weights),
            )
        else:
            (
                feat_sum,
                weight_sum_feature,
                weight_sum_support,
                pair_count_total,
                pair_count_after_threshold,
            ) = rasterize_and_backproject_multi_camera_in_range(
                range_start=0,
                range_end=int(1e9),
                means2d=meta["means2d"],
                conics=meta["conics"],
                opacities=meta["opacities"],
                image_width=int(width),
                image_height=int(height),
                tile_size=int(meta.get("tile_size", 16)),
                isect_offsets=meta["isect_offsets"],
                flatten_ids=meta["flatten_ids"],
                packed_global_gaussian_ids=meta["packed_global_gaussian_ids"],
                feat2d=features_2d,
                num_gaussians=int(num_gaussians),
                pair_valid_mask=pair_valid_mask,
                weight_threshold=float(weight_threshold),
                return_support=bool(return_accumulated_weights),
            )
        fused_ms = float((time.perf_counter() - t_fused) * 1000.0)
        feat_out = feat_sum / (weight_sum_feature.unsqueeze(-1) + eps)
        if orig_dtype != torch.float32:
            feat_out = feat_out.to(orig_dtype)
            if return_accumulated_weights:
                weight_sum_support = weight_sum_support.to(orig_dtype)

        pairs_after_mask = int(pair_count_total.item())
        pairs_total = pairs_after_mask
        pairs_total_recount_ms = 0.0
        # When mask is enabled, fused op only reports post-mask pair count.
        # Recount once without mask (debug only) to preserve mask-before/after stats semantics.
        if pair_valid_mask is not None and return_debug_stats:
            t_recount = time.perf_counter()
            with torch.no_grad():
                _, _, _, pair_count_total_nomask, _ = rasterize_and_backproject_multi_camera_in_range(
                    range_start=0,
                    range_end=int(1e9),
                    means2d=meta["means2d"],
                    conics=meta["conics"],
                    opacities=meta["opacities"],
                    image_width=int(width),
                    image_height=int(height),
                    tile_size=int(meta.get("tile_size", 16)),
                    isect_offsets=meta["isect_offsets"],
                    flatten_ids=meta["flatten_ids"],
                    packed_global_gaussian_ids=meta["packed_global_gaussian_ids"],
                    feat2d=features_2d.detach(),
                    num_gaussians=int(num_gaussians),
                    pair_valid_mask=None,
                    weight_threshold=float(weight_threshold),
                    return_support=False,
                )
            pairs_total = int(pair_count_total_nomask.item())
            pairs_total_recount_ms = float((time.perf_counter() - t_recount) * 1000.0)

        stats = {
            "num_views": int(len(cameras)),
            "num_gaussians": int(num_gaussians),
            "render_packed_total_ms": float(meta_stats["render_packed_total_ms"]),
            "build_multi_meta_ms": float(meta_stats["build_multi_meta_ms"]),
            "fused_backproject_total_ms": float(fused_ms),
            "pairs_total_recount_ms": float(pairs_total_recount_ms),
            "streaming_total_ms": float((time.perf_counter() - t_total_start) * 1000.0),
            "pairs_total": int(pairs_total),
            "pairs_after_mask": int(pairs_after_mask),
            "pairs_after_threshold": int(pair_count_after_threshold.item()),
            "nnz_total": int(meta_stats["nnz_total"]),
            "isects_total": int(meta_stats["isects_total"]),
            "masked_pixel_count": int(masked_pixel_count),
            "valid_pixel_count": int(valid_pixel_count),
            "source_pair_valid_ratio": float(source_pair_valid_ratio),
        }
        if return_accumulated_weights:
            if return_debug_stats:
                return feat_out, weight_sum_support, stats
            return feat_out, weight_sum_support
        if return_debug_stats:
            return feat_out, stats
        return feat_out

    def render_and_backproject_streaming_fused_per_view_fallback(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        features_2d: torch.Tensor,
        height: int,
        width: int,
        num_gaussians: int,
        backprojector,
        return_accumulated_weights: bool = False,
        return_debug_stats: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
    ]:
        if not self.fused_available:
            raise RuntimeError("ExtractorV3 fast-fail: fused op is unavailable in current gsplat build.")
        if not features_2d.is_cuda:
            raise RuntimeError("ExtractorV3 fast-fail: features_2d must be CUDA tensor.")
        if features_2d.requires_grad and backproject_feature_grad_in_range is None:
            raise RuntimeError("ExtractorV3 fast-fail: fused backward op is unavailable.")
        if features_2d.ndim != 4:
            raise ValueError(
                "ExtractorV3 multi-src expects features_2d with shape [V, Hf, Wf, C], "
                f"got ndim={features_2d.ndim}."
            )
        if len(cameras) < 1:
            raise ValueError("ExtractorV3 multi-src requires at least one camera.")
        if int(features_2d.shape[0]) != int(len(cameras)):
            raise ValueError(
                f"features_2d first dim ({features_2d.shape[0]}) must equal num cameras ({len(cameras)})."
            )
        orig_dtype = features_2d.dtype
        if orig_dtype != torch.float32:
            features_2d = features_2d.float()

        t_total_start = time.perf_counter()
        device = features_2d.device
        channels = int(features_2d.shape[-1])
        eps = float(getattr(backprojector, "eps", 1e-8))
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
            "pairs_total_per_view": [],
            "pairs_after_threshold_per_view": [],
            "render_packed_ms_per_view": [],
            "fused_backproject_ms_per_view": [],
        }

        for i, cam in enumerate(cameras):
            t_render = time.perf_counter()
            meta = self._render_single_view_packed_meta(gaussians, cam, height=height, width=width)
            render_ms = float((time.perf_counter() - t_render) * 1000.0)
            stats["render_packed_total_ms"] += render_ms
            stats["render_packed_ms_per_view"].append(render_ms)

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
            fused_ms = float((time.perf_counter() - t_fused) * 1000.0)
            stats["fused_backproject_total_ms"] += fused_ms
            stats["fused_backproject_ms_per_view"].append(fused_ms)
            stats["pairs_total"] += int(pairs_total)
            stats["pairs_after_threshold"] += int(pairs_kept)
            stats["pairs_total_per_view"].append(int(pairs_total))
            stats["pairs_after_threshold_per_view"].append(int(pairs_kept))

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

    def render_and_backproject_streaming_fused(self, *args, **kwargs):
        raise RuntimeError(
            "AlphaTWeightExtractorV3.render_and_backproject_streaming_fused is deprecated. "
            "Use render_and_backproject_streaming_fused_multi_camera (preferred) or "
            "render_and_backproject_streaming_fused_per_view_fallback instead."
        )


__all__ = ["AlphaTWeightExtractorV3"]
