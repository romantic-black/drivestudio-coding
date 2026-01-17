"""
Backproject 2D features to Gaussians using alpha-T weights.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


class FeatureBackprojector:
    """
    Aggregate per-pixel CNN features into per-Gaussian descriptors.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @staticmethod
    def sample_features_at_pixels(
        features_2d: torch.Tensor,
        pixel_ids: torch.Tensor,
        view_ids: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Bilinearly sample 2D features at the provided pixel indices.
        """
        if features_2d.dim() != 4:
            raise ValueError(f"features_2d must be [V, H, W, C], got {features_2d.shape}")
        device = features_2d.device
        V, _, _, C2 = features_2d.shape
        coords = torch.zeros(len(pixel_ids), 2, device=device, dtype=features_2d.dtype)
        coords[:, 0] = (pixel_ids % width).float() / float(width)
        coords[:, 1] = (pixel_ids // width).float() / float(height)
        coords = coords * 2.0 - 1.0  # [-1, 1]

        sampled = torch.zeros(len(pixel_ids), C2, device=device, dtype=features_2d.dtype)
        for v in range(V):
            mask = view_ids == v
            if not torch.any(mask):
                continue
            feat_v = features_2d[v].permute(2, 0, 1).unsqueeze(0)  # [1, C2, H, W]
            coords_v = coords[mask].view(1, 1, -1, 2)
            sampled_v = F.grid_sample(
                feat_v,
                coords_v,
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            )
            sampled[mask] = sampled_v.squeeze(0).squeeze(2).t()
        return sampled

    def aggregate_features_per_gaussian(
        self,
        sampled_features: torch.Tensor,
        weights: torch.Tensor,
        gaussian_ids: torch.Tensor,
        num_gaussians: int,
    ) -> torch.Tensor:
        """
        Scatter-add aggregation:
            f_k = sum(w * f) / sum(w)
        """
        device = sampled_features.device
        C2 = sampled_features.shape[1]
        weighted = sampled_features * weights.unsqueeze(-1)
        num = torch.zeros(num_gaussians, C2, device=device, dtype=sampled_features.dtype)
        num.scatter_add_(0, gaussian_ids.unsqueeze(-1).expand(-1, C2), weighted)

        den = torch.zeros(num_gaussians, device=device, dtype=sampled_features.dtype)
        den.scatter_add_(0, gaussian_ids, weights)
        return num / (den.unsqueeze(-1) + self.eps)

    def backproject(
        self,
        features_2d_list: List[torch.Tensor],
        weights_info: List[Dict[str, torch.Tensor]],
        height: int,
        width: int,
        num_gaussians: int,
    ) -> torch.Tensor:
        """
        Full backprojection pipeline on GPU.
        """
        if len(features_2d_list) == 0:
            return torch.zeros(num_gaussians, 0)
        device = features_2d_list[0].device
        channels = features_2d_list[0].shape[-1]
        if num_gaussians == 0:
            return torch.zeros(0, channels, device=device)
        if len(weights_info) == 0:
            return torch.zeros(num_gaussians, channels, device=device)

        gaussian_ids = torch.cat([w["gaussian_ids"] for w in weights_info], dim=0).long().to(device)
        pixel_ids = torch.cat([w["pixel_ids"] for w in weights_info], dim=0).long().to(device)
        weights = torch.cat([w["weights"] for w in weights_info], dim=0).to(device).detach()

        if gaussian_ids.numel() == 0:
            return torch.zeros(num_gaussians, channels, device=device)

        view_ids = torch.cat(
            [
                torch.full((len(w["gaussian_ids"]),), idx, device=device, dtype=torch.long)
                for idx, w in enumerate(weights_info)
            ],
            dim=0,
        )

        features_2d = torch.stack(features_2d_list, dim=0).to(device)
        sampled = self.sample_features_at_pixels(features_2d, pixel_ids, view_ids, height, width)
        return self.aggregate_features_per_gaussian(sampled, weights, gaussian_ids, num_gaussians)
