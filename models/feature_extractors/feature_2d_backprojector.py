from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Feature2DBackprojector(nn.Module):
    """
    Scatter-add aggregation of 2D features to per-Gaussian descriptors.
    """

    def __init__(self, feature_channels: int, eps: float = 1e-8, device: Optional[torch.device] = None):
        super().__init__()
        self.feature_channels = feature_channels
        self.eps = eps
        if device is not None:
            self.to(device)

    def forward(
        self,
        features_2d: List[torch.Tensor],
        gaussian_indices: List[torch.Tensor],
        alpha_t_weights: List[torch.Tensor],
        num_gaussians: int,
        bg_indices: Optional[torch.Tensor] = None,
        rigid_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features_2d) != len(gaussian_indices):
            raise ValueError("features_2d and gaussian_indices must have the same length.")
        device = features_2d[0].device
        dtype = features_2d[0].dtype
        accum_dtype = torch.float32 if dtype.is_floating_point else dtype
        num = torch.zeros((num_gaussians, self.feature_channels), device=device, dtype=accum_dtype)
        den = torch.zeros((num_gaussians,), device=device, dtype=accum_dtype)

        for feat_map, idx_map, w_map in zip(features_2d, gaussian_indices, alpha_t_weights):
            feat_flat = feat_map.permute(1, 2, 0).reshape(-1, self.feature_channels).to(accum_dtype)
            idx_flat = idx_map.reshape(-1, idx_map.shape[-1]).long()
            w_flat = w_map.reshape(-1, w_map.shape[-1]).to(accum_dtype)
            for k in range(idx_flat.shape[1]):
                idx_k = idx_flat[:, k]
                valid = idx_k >= 0
                if not valid.any():
                    continue
                idx_valid = idx_k[valid]
                w_valid = w_flat[valid, k]
                feat_valid = feat_flat[valid]
                weighted = feat_valid * w_valid.unsqueeze(-1)
                num = torch.index_add(num, 0, idx_valid, weighted)
                den = torch.index_add(den, 0, idx_valid, w_valid)

        feat_all = (num / (den.unsqueeze(-1) + self.eps)).to(dtype)
        vis_all = torch.clamp(den, 0.0, 1.0).to(dtype)

        feat_2d_bg = feat_all[bg_indices] if bg_indices is not None else feat_all
        vis_bg = vis_all[bg_indices] if bg_indices is not None else vis_all
        if rigid_indices is not None and rigid_indices.numel() > 0:
            feat_2d_rigid = feat_all[rigid_indices]
            vis_rigid = vis_all[rigid_indices]
        else:
            feat_2d_rigid = torch.zeros(0, self.feature_channels, device=device, dtype=dtype)
            vis_rigid = torch.zeros(0, device=device, dtype=dtype)

        return feat_all, vis_all, feat_2d_bg, feat_2d_rigid, vis_bg, vis_rigid
