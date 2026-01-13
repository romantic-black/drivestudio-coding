from typing import Tuple

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    Concatenate 3D and 2D features (plus visibility) for offset prediction.
    """

    def __init__(self, feat_3d_dim: int, feat_2d_dim: int, include_visibility: bool = True):
        super().__init__()
        self.feat_3d_dim = feat_3d_dim
        self.feat_2d_dim = feat_2d_dim
        self.include_visibility = include_visibility

    def forward(
        self,
        feat_3d_bg: torch.Tensor,
        feat_3d_rigid: torch.Tensor,
        feat_2d_bg: torch.Tensor,
        feat_2d_rigid: torch.Tensor,
        vis_bg: torch.Tensor,
        vis_rigid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts_bg = [feat_3d_bg, feat_2d_bg]
        parts_rigid = [feat_3d_rigid, feat_2d_rigid]
        if self.include_visibility:
            parts_bg.append(vis_bg.unsqueeze(-1))
            parts_rigid.append(vis_rigid.unsqueeze(-1))
        feat_fused_bg = torch.cat(parts_bg, dim=-1) if parts_bg else feat_3d_bg
        feat_fused_rigid = torch.cat(parts_rigid, dim=-1) if parts_rigid else feat_3d_rigid
        return feat_fused_bg, feat_fused_rigid
