"""
Extractor v4: fused multi-source streaming backprojection with observation code.

v4 keeps v3 fused numerical semantics:
- single packed multi-camera render + fused backproject
- single global normalization at the end
- adds current observation code per gaussian: [log1p(rho), overlap]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from models.feature_extractors.alpha_t_extractor_v3 import AlphaTWeightExtractorV3
try:
    from gsplat.cuda._wrapper import (
        backproject_feature_grad_multi_camera_sharded_in_range,
        rasterize_and_backproject_multi_camera_in_range,
        rasterize_and_backproject_multi_camera_obs_in_range,
    )
except Exception:  # pragma: no cover
    backproject_feature_grad_multi_camera_sharded_in_range = None
    rasterize_and_backproject_multi_camera_in_range = None
    rasterize_and_backproject_multi_camera_obs_in_range = None


class _RasterizeAndBackprojectFeatObsMultiCamFn(torch.autograd.Function):
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
        obs_eps: float,
    ):
        feat2d_c = feat2d.contiguous()
        pair_valid_mask_c = None
        if pair_valid_mask is not None:
            pair_valid_mask_c = pair_valid_mask.contiguous()
        out = rasterize_and_backproject_multi_camera_obs_in_range(
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
            obs_eps=obs_eps,
        )
        feat_sum, w_feat, w_sup, obs_code, pairs_total, pairs_kept = out
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
        ctx.mark_non_differentiable(w_feat, w_sup, obs_code, pairs_total, pairs_kept)
        return feat_sum, w_feat, w_sup, obs_code, pairs_total, pairs_kept

    @staticmethod
    def backward(ctx, grad_feat_sum, grad_w_feat, grad_w_sup, grad_obs_code, grad_pairs_total, grad_pairs_kept):
        del grad_w_feat, grad_w_sup, grad_obs_code, grad_pairs_total, grad_pairs_kept
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
            None, None, None, None, None, None, None, None,
        )


class AlphaTWeightExtractorV4(AlphaTWeightExtractorV3):
    @property
    def fused_multi_camera_obs_available(self) -> bool:
        return (
            rasterize_and_backproject_multi_camera_obs_in_range is not None
            and backproject_feature_grad_multi_camera_sharded_in_range is not None
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
        return_obs_code: bool = False,
        return_debug_stats: bool = False,
        return_raw_sums: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]],
    ]:
        if not self.fused_multi_camera_obs_available:
            raise RuntimeError("ExtractorV4 fast-fail: multi-camera obs fused op is unavailable in current gsplat build.")
        if not features_2d.is_cuda:
            raise RuntimeError("ExtractorV4 fast-fail: features_2d must be CUDA tensor.")
        if features_2d.ndim != 4:
            raise ValueError(
                "ExtractorV4 multi-camera expects features_2d with shape [V, Hf, Wf, C], "
                f"got ndim={features_2d.ndim}."
            )
        if len(cameras) < 1:
            raise ValueError("ExtractorV4 multi-camera requires at least one camera.")
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
        obs_eps = float(getattr(backprojector, "obs_eps", 1.0e-6))
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
                obs_code,
                pair_count_total,
                pair_count_after_threshold,
            ) = _RasterizeAndBackprojectFeatObsMultiCamFn.apply(
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
                float(obs_eps),
            )
        else:
            (
                feat_sum,
                weight_sum_feature,
                weight_sum_support,
                obs_code,
                pair_count_total,
                pair_count_after_threshold,
            ) = rasterize_and_backproject_multi_camera_obs_in_range(
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
                obs_eps=float(obs_eps),
            )
        fused_ms = float((time.perf_counter() - t_fused) * 1000.0)
        feat_out = feat_sum / (weight_sum_feature.unsqueeze(-1) + eps)
        if orig_dtype != torch.float32 and not bool(return_raw_sums):
            feat_out = feat_out.to(orig_dtype)
            obs_code = obs_code.to(orig_dtype)
            if return_accumulated_weights:
                weight_sum_support = weight_sum_support.to(orig_dtype)

        pairs_after_mask = int(pair_count_total.item())
        pairs_total = pairs_after_mask
        pairs_total_recount_ms = 0.0
        if pair_valid_mask is not None and return_debug_stats and rasterize_and_backproject_multi_camera_in_range is not None:
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
            "obs_rho_log_mean": float(obs_code[:, 0].float().mean().item()) if obs_code.numel() > 0 else 0.0,
            "obs_overlap_mean": float(obs_code[:, 1].float().mean().item()) if obs_code.numel() > 0 else 0.0,
        }

        if bool(return_raw_sums):
            if not bool(return_accumulated_weights):
                raise ValueError("return_raw_sums=True requires return_accumulated_weights=True")
            if not bool(return_obs_code):
                raise ValueError("return_raw_sums=True requires return_obs_code=True")
            if return_debug_stats:
                return feat_sum, weight_sum_feature, weight_sum_support, obs_code, stats
            return feat_sum, weight_sum_feature, weight_sum_support, obs_code
        if return_accumulated_weights and return_obs_code and return_debug_stats:
            return feat_out, weight_sum_support, obs_code, stats
        if return_accumulated_weights and return_obs_code:
            return feat_out, weight_sum_support, obs_code
        if return_obs_code and return_debug_stats:
            return feat_out, obs_code, stats
        if return_obs_code:
            return feat_out, obs_code
        if return_accumulated_weights:
            if return_debug_stats:
                return feat_out, weight_sum_support, stats
            return feat_out, weight_sum_support
        if return_debug_stats:
            return feat_out, stats
        return feat_out


__all__ = ["AlphaTWeightExtractorV4"]
