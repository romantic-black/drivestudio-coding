"""
Utilities to fuse 2D and 3D features for StreetForward.
"""

from __future__ import annotations

import torch


class FeatureFusion:
    """
    Concatenate 3D features, 2D backprojected features, and optional visibility.
    """

    def __init__(self, use_visibility: bool = True) -> None:
        self.use_visibility = use_visibility

    def fuse(
        self,
        feat_3d: torch.Tensor,
        feat_2d: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [feat_3d, feat_2d]
        if self.use_visibility and visibility is not None:
            pieces.append(visibility.unsqueeze(-1))
        return torch.cat(pieces, dim=-1)
