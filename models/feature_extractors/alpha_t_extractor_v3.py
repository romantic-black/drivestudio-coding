"""
Extractor v3: fused multi-source streaming backprojection.

v3 keeps v2 fused numerical semantics:
- per-view packed render + fused backproject
- online accumulation over views
- single global normalization at the end
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Union

import torch

from models.feature_extractors.alpha_t_extractor import _get_viewmat
from models.feature_extractors.alpha_t_extractor_v2 import AlphaTWeightExtractorV2, backproject_feature_grad_in_range


class AlphaTWeightExtractorV3(AlphaTWeightExtractorV2):
    """Extractor v3 with explicit multi-src contract on top of v2 fused path."""

    def render_and_backproject_streaming_fused(
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


__all__ = ["AlphaTWeightExtractorV3"]
