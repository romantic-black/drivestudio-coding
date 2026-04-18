"""
Minimal StreetForward Stage 4.4:
- inherits Stage 4.3 training/rendering/logging pipeline
- uses AlphaTWeightExtractorV3 for fused multi-src streaming backprojection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.feature_extractors import AlphaTWeightExtractorV3
from models.feature_extractors.alpha_t_extractor import _get_viewmat
from models.streetforward.minimal_trainer_stage4_0 import merge_debug_stats_as_perf_floats, spatial_hw_from_image_tensor
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3


class MinimalStreetForwardStage4_4(MinimalStreetForwardStage4_3):
    """Stage 4.4 = Stage 4.3 + explicit multi-src fused-v3 2D path."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        model_cfg = config.model
        self.use_fused_cuda_backproject_v4 = bool(model_cfg.get("use_fused_cuda_backproject_v4", False))
        self.fused_cuda_backproject_v4_force_fallback = bool(
            model_cfg.get("fused_cuda_backproject_v4_force_fallback", False)
        )
        self.use_fused_cuda_backproject_v3 = bool(model_cfg.get("use_fused_cuda_backproject_v3", True))
        self.alpha_t_extractor_v3 = AlphaTWeightExtractorV3(
            renderer=self.renderer,
            sh_degree=self.sh_degree,
            tile_size=16,
        )

    def _build_viewmats_from_views(self, views: List[Any]) -> torch.Tensor:
        mats = []
        for view in views:
            cam_ctw = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
            mats.append(_get_viewmat(cam_ctw))
        return torch.cat(mats, dim=0)

    def _render_source_composite_for_cnn(
        self,
        gaussians_scene: Dict[str, torch.Tensor],
        gaussians_sky: Dict[str, torch.Tensor],
        source_views: List,
        source_images: List[torch.Tensor],
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        if len(source_views) != len(source_images):
            raise ValueError(
                f"Stage4.4 len(source_views)={len(source_views)} != len(source_images)={len(source_images)}."
            )
        if len(source_views) < 1:
            raise ValueError("Stage4.4 requires at least one source view.")
        ref_hw = spatial_hw_from_image_tensor(source_images[0])
        for i, img in enumerate(source_images):
            hw = spatial_hw_from_image_tensor(img)
            if hw != ref_hw:
                raise ValueError(
                    "Stage4.4 multi-src currently requires identical H/W across all source_images. "
                    f"Mismatch at idx={i}: {hw} vs ref={ref_hw}."
                )

        scene_render_out = self.alpha_t_extractor.render_rgb_only(
            gaussians_scene,
            source_views,
            height,
            width,
            return_acc=True,
            return_debug_stats=False,
        )
        scene_rgbs, scene_accs = scene_render_out
        scene_rgb_batch = torch.stack(scene_rgbs, dim=0).detach()
        acc_scene_src = torch.stack(scene_accs, dim=0).detach()
        if acc_scene_src.dim() == 4 and acc_scene_src.shape[-1] == 1:
            acc_scene_src = acc_scene_src.squeeze(-1)

        sky_viewmats = self._sky_viewmats_from_views(source_views)
        sky_render_out = self.alpha_t_extractor.render_rgb_only(
            gaussians_sky,
            source_views,
            height,
            width,
            return_acc=False,
            viewmats_override=sky_viewmats,
            return_debug_stats=False,
        )
        sky_rgb_batch = torch.stack(sky_render_out, dim=0).detach()

        rendered_batch = scene_rgb_batch + sky_rgb_batch * (1.0 - acc_scene_src.clamp(0.0, 1.0)).unsqueeze(-1)
        image_batch = torch.stack([img.to(self.device) for img in source_images], dim=0)
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)
        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            acc_scene_src = F.interpolate(
                acc_scene_src.unsqueeze(1),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        multi = torch.cat([image_batch, rendered_batch], dim=-1)
        features_2d = self.image_feature_extractor(multi)
        gate_image = (1.0 - acc_scene_src.clamp(0.0, 1.0)).detach().float()
        return {
            "features_2d": features_2d,
            "gate_image": gate_image,
            "sky_viewmats": sky_viewmats,
        }

    def _backproject_scene_features_multi_camera(
        self,
        gaussians_scene: Dict[str, torch.Tensor],
        source_views: List,
        features_2d: torch.Tensor,
        height: int,
        width: int,
        backprojector_override=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_gaussians = int(gaussians_scene["means"].shape[0])
        if num_gaussians == 0:
            return None, None
        backprojector_impl = backprojector_override if backprojector_override is not None else self.feature_backprojector
        feat_2d_all, acc_w, bp_stats = self.alpha_t_extractor_v3.render_and_backproject_streaming_fused_multi_camera(
            gaussians=gaussians_scene,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=num_gaussians,
            backprojector=backprojector_impl,
            return_accumulated_weights=True,
            return_debug_stats=True,
        )
        merge_debug_stats_as_perf_floats(self._perf_acc, "2d_bp_scene_", bp_stats)
        self._perf_acc["2d_bp_scene_call_count"] = float(self._perf_acc.get("2d_bp_scene_call_count", 0.0) + 1.0)
        return feat_2d_all, acc_w

    def _backproject_sky_features_gated_multi_camera(
        self,
        gaussians_sky: Dict[str, torch.Tensor],
        source_views: List,
        features_2d: torch.Tensor,
        gate_image: torch.Tensor,
        sky_viewmats: torch.Tensor,
        height: int,
        width: int,
        backprojector_override=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        del gaussians_sky, source_views, features_2d, gate_image, sky_viewmats, height, width, backprojector_override
        raise RuntimeError(
            "Stage4.4 sky gated fused backprojection has been removed in no-gated mode. "
            "Use Stage4.5 no-sky pipeline instead."
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
        raise RuntimeError(
            "Stage4.4 legacy _compute_2d_features_for_gaussians() is deprecated. "
            "Use _compute_2d_features_scene_and_sky_gated() pipeline instead."
        )


__all__ = ["MinimalStreetForwardStage4_4"]
