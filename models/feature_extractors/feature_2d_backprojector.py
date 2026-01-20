"""
Backproject 2D features to Gaussians using alpha-T weights.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class FeatureBackprojector:
    """
    Aggregate per-pixel CNN features into per-Gaussian descriptors.
    """

    def __init__(self, eps: float = 1e-8, weight_threshold: float = 1e-2) -> None:
        """
        Args:
            eps: Small epsilon for numerical stability in division.
            weight_threshold: Minimum αT weight threshold. Gaussian-pixel pairs with weights 
                            below this threshold will be filtered out to reduce memory usage.
                            Default: 0.0 (no filtering). Suggested: 1e-4 to 1e-3.
        """
        self.eps = eps
        self.weight_threshold = weight_threshold

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
                feat_v,  # [1, C2, H, W]
                coords_v,  # [1, 1, n_mask, 2]
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            )  # Output: [1, C2, 1, n_mask]
            
            # Remove batch and height dimensions, then transpose
            # [1, C2, 1, n_mask] -> [C2, n_mask] -> [n_mask, C2]
            sampled_v_processed = sampled_v.squeeze(0).squeeze(1).t()  # [n_mask, C2]
            sampled[mask] = sampled_v_processed
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

    def _sample_features_single_view(
        self,
        feat_2d: torch.Tensor,
        pixel_ids: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Bilinearly sample a single-view feature map at the specified pixel ids.
        """
        device = feat_2d.device
        coords = torch.zeros(len(pixel_ids), 2, device=device, dtype=feat_2d.dtype)
        coords[:, 0] = (pixel_ids % width).float() / float(width)
        coords[:, 1] = (pixel_ids // width).float() / float(height)
        coords = coords * 2.0 - 1.0  # [-1, 1]

        feat_2d_chw = feat_2d.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        coords_grid = coords.view(1, 1, -1, 2)  # [1, 1, M, 2]

        sampled = F.grid_sample(
            feat_2d_chw,
            coords_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )  # [1, C, 1, M]
        return sampled.squeeze(0).squeeze(1).t()  # [M, C]

    def backproject_single_view(
        self,
        feat_2d: torch.Tensor,
        weight_info: Dict[str, torch.Tensor],
        height: int,
        width: int,
        num_gaussians: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backproject a single view: return weighted feature sums and weight sums.
        """
        gaussian_ids = weight_info["gaussian_ids"].long()
        pixel_ids = weight_info["pixel_ids"].long()
        weights = weight_info["weights"].detach()

        device = feat_2d.device
        channels = feat_2d.shape[-1]

        if gaussian_ids.numel() == 0:
            return (
                torch.zeros(num_gaussians, channels, device=device, dtype=feat_2d.dtype),
                torch.zeros(num_gaussians, device=device, dtype=feat_2d.dtype),
            )

        if self.weight_threshold > 0.0:
            mask = weights >= self.weight_threshold
            gaussian_ids = gaussian_ids[mask]
            pixel_ids = pixel_ids[mask]
            weights = weights[mask]

        if gaussian_ids.numel() == 0:
            return (
                torch.zeros(num_gaussians, channels, device=device, dtype=feat_2d.dtype),
                torch.zeros(num_gaussians, device=device, dtype=feat_2d.dtype),
            )

        sampled = self._sample_features_single_view(feat_2d, pixel_ids, height, width)
        weighted_feat = sampled * weights.unsqueeze(-1)

        feat_sum = torch.zeros(num_gaussians, channels, device=device, dtype=feat_2d.dtype)
        feat_sum.scatter_add_(0, gaussian_ids.unsqueeze(-1).expand(-1, channels), weighted_feat)

        weight_sum = torch.zeros(num_gaussians, device=device, dtype=feat_2d.dtype)
        weight_sum.scatter_add_(0, gaussian_ids, weights)
        return feat_sum, weight_sum

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
        view_ids_raw = torch.cat(
            [
                torch.full((len(w["gaussian_ids"]),), idx, device=device, dtype=torch.long)
                for idx, w in enumerate(weights_info)
            ],
            dim=0,
        )
        
        # Filter out low-weight gaussian-pixel pairs to reduce memory usage
        M_before_filter = len(gaussian_ids)
        if self.weight_threshold > 0.0:
            mask = weights >= self.weight_threshold
            gaussian_ids = gaussian_ids[mask]
            pixel_ids = pixel_ids[mask]
            weights = weights[mask]
            view_ids = view_ids_raw[mask]  # Apply same filter to view_ids
            M_after_filter = len(gaussian_ids)
        else:
            view_ids = view_ids_raw
            M_after_filter = M_before_filter

        if gaussian_ids.numel() == 0:
            return torch.zeros(num_gaussians, channels, device=device)

        features_2d = torch.stack(features_2d_list, dim=0).to(device)
        sampled = self.sample_features_at_pixels(features_2d, pixel_ids, view_ids, height, width)
        aggregated = self.aggregate_features_per_gaussian(sampled, weights, gaussian_ids, num_gaussians)
        return aggregated
