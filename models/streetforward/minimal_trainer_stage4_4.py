"""
Minimal StreetForward Stage 4.4:
- inherits Stage 4.3 training/rendering/logging pipeline
- uses AlphaTWeightExtractorV3 for fused multi-src streaming backprojection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors import AlphaTWeightExtractorV3
from models.streetforward.minimal_trainer_stage4_0 import merge_debug_stats_as_perf_floats, spatial_hw_from_image_tensor
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3


class MinimalStreetForwardStage4_4(MinimalStreetForwardStage4_3):
    """Stage 4.4 = Stage 4.3 + explicit multi-src fused-v3 2D path."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        model_cfg = config.model
        self.use_fused_cuda_backproject_v3 = bool(model_cfg.get("use_fused_cuda_backproject_v3", True))
        self.alpha_t_extractor_v3 = AlphaTWeightExtractorV3(
            renderer=self.renderer,
            sh_degree=self.sh_degree,
            tile_size=16,
        )

    def _compute_2d_features_for_gaussians(
        self,
        gaussians: Dict[str, torch.Tensor],
        source_views: List,
        source_images: List[torch.Tensor],
        height: int,
        width: int,
        return_accumulated_weights: bool = False,
        backprojector_override=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_gaussians = int(gaussians["means"].shape[0])
        if num_gaussians == 0:
            return None, None
        if source_views is None or source_images is None:
            raise ValueError("Stage4.4 requires source_views and source_images.")
        if len(source_views) == 0:
            raise ValueError("Stage4.4 requires at least one source view.")
        if len(source_views) != len(source_images):
            raise ValueError(
                f"Stage4.4 len(source_views)={len(source_views)} != len(source_images)={len(source_images)}."
            )

        ref_hw = spatial_hw_from_image_tensor(source_images[0])
        for i, img in enumerate(source_images):
            hw = spatial_hw_from_image_tensor(img)
            if hw != ref_hw:
                raise ValueError(
                    "Stage4.4 multi-src currently requires identical H/W across all source_images. "
                    f"Mismatch at idx={i}: {hw} vs ref={ref_hw}."
                )

        stats: Dict[str, float] = {}  # all values must be float for _perf_acc accumulation
        if torch.cuda.is_available():
            stats["cuda_mem_alloc_before"] = float(torch.cuda.memory_allocated())
            stats["cuda_mem_reserved_before"] = float(torch.cuda.memory_reserved())
        with torch.no_grad():
            render_rgb_out = self.alpha_t_extractor.render_rgb_only(
                gaussians, source_views, height, width, return_debug_stats=True
            )
        if isinstance(render_rgb_out, tuple):
            rendered_rgbs, rgb_stats = render_rgb_out
            merge_debug_stats_as_perf_floats(stats, "2d_rgb_", rgb_stats)
        else:
            rendered_rgbs = render_rgb_out
        image_batch = torch.stack([img.to(self.device) for img in source_images], dim=0)
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)
        rendered_batch = torch.stack(rendered_rgbs, dim=0).detach()
        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        multi = torch.cat([image_batch, rendered_batch], dim=-1)
        features_2d = self.image_feature_extractor(multi)
        backprojector_impl = backprojector_override if backprojector_override is not None else self.feature_backprojector
        use_fused_v3 = bool(getattr(self, "use_fused_cuda_backproject_v3", True))
        use_fused_v2 = bool(getattr(self, "use_fused_cuda_backproject_v2", False))
        if use_fused_v3:
            back_out = self.alpha_t_extractor_v3.render_and_backproject_streaming_fused(
                gaussians=gaussians,
                cameras=source_views,
                features_2d=features_2d,
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                backprojector=backprojector_impl,
                return_accumulated_weights=return_accumulated_weights,
                return_debug_stats=True,
            )
        elif use_fused_v2:
            back_out = self.alpha_t_extractor_v2.render_and_backproject_streaming_fused(
                gaussians=gaussians,
                cameras=source_views,
                features_2d=features_2d,
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                backprojector=backprojector_impl,
                return_accumulated_weights=return_accumulated_weights,
                return_debug_stats=True,
            )
        else:
            back_out = self.alpha_t_extractor.render_and_backproject_streaming(
                gaussians=gaussians,
                cameras=source_views,
                features_2d=features_2d,
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                backprojector=backprojector_impl,
                return_accumulated_weights=return_accumulated_weights,
                return_debug_stats=True,
            )
        if return_accumulated_weights:
            feat_2d_all, acc_w, bp_stats = back_out
            merge_debug_stats_as_perf_floats(stats, "2d_bp_", bp_stats)
            if torch.cuda.is_available():
                stats["cuda_mem_alloc_after"] = float(torch.cuda.memory_allocated())
                stats["cuda_mem_reserved_after"] = float(torch.cuda.memory_reserved())
                stats["cuda_mem_alloc_delta"] = float(stats["cuda_mem_alloc_after"] - stats["cuda_mem_alloc_before"])
                stats["cuda_mem_reserved_delta"] = float(
                    stats["cuda_mem_reserved_after"] - stats["cuda_mem_reserved_before"]
                )
            for k, v in stats.items():
                self._perf_acc[k] = float(self._perf_acc.get(k, 0.0) + float(v))
            self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 1.0)
            return feat_2d_all, acc_w
        feat_2d_all, bp_stats = back_out
        merge_debug_stats_as_perf_floats(stats, "2d_bp_", bp_stats)
        if torch.cuda.is_available():
            stats["cuda_mem_alloc_after"] = float(torch.cuda.memory_allocated())
            stats["cuda_mem_reserved_after"] = float(torch.cuda.memory_reserved())
            stats["cuda_mem_alloc_delta"] = float(stats["cuda_mem_alloc_after"] - stats["cuda_mem_alloc_before"])
            stats["cuda_mem_reserved_delta"] = float(
                stats["cuda_mem_reserved_after"] - stats["cuda_mem_reserved_before"]
            )
        for k, v in stats.items():
            self._perf_acc[k] = float(self._perf_acc.get(k, 0.0) + float(v))
        self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 1.0)
        return feat_2d_all, None


__all__ = ["MinimalStreetForwardStage4_4"]
