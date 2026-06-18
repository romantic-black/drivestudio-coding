from __future__ import annotations

import torch
import torch.nn as nn

from .image_feature_extractor import ImageFeatureExtractor


class ResidualOnlyFeatureExtractor(nn.Module):
    """Residual-image frontend without DINO or fusion branches."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        feat_channels: int = 16,
        base_channels: int = 24,
        feature_downscale: int = 1,
        depth: int = 3,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.residual_unet = ImageFeatureExtractor(
            in_channels=int(in_channels),
            feat_channels=int(feat_channels),
            base_channels=int(base_channels),
            feature_downscale=int(feature_downscale),
            depth=int(depth),
            bilinear=bool(bilinear),
        )

    def get_feature_resolution(self, height: int, width: int) -> tuple[int, int]:
        return self.residual_unet.get_feature_resolution(int(height), int(width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.residual_unet(images)


__all__ = ["ResidualOnlyFeatureExtractor"]
