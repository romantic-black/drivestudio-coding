from typing import Tuple
import json
import time

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
        # #region agent log
        _debug_log_path = "/root/drivestudio-coding/.cursor/debug.log"
        try:
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "feature_fusion.py:forward",
                "message": "Feature fusion start",
                "data": {
                    "feat_3d_bg_shape": list(feat_3d_bg.shape),
                    "feat_3d_rigid_shape": list(feat_3d_rigid.shape),
                    "feat_2d_bg_shape": list(feat_2d_bg.shape),
                    "feat_2d_rigid_shape": list(feat_2d_rigid.shape),
                    "include_visibility": self.include_visibility,
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H4",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        parts_bg = [feat_3d_bg, feat_2d_bg]
        parts_rigid = [feat_3d_rigid, feat_2d_rigid]
        if self.include_visibility:
            parts_bg.append(vis_bg.unsqueeze(-1))
            parts_rigid.append(vis_rigid.unsqueeze(-1))
        feat_fused_bg = torch.cat(parts_bg, dim=-1) if parts_bg else feat_3d_bg
        feat_fused_rigid = torch.cat(parts_rigid, dim=-1) if parts_rigid else feat_3d_rigid
        
        # #region agent log
        try:
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "feature_fusion.py:forward",
                "message": "After feature fusion",
                "data": {
                    "feat_fused_bg_shape": list(feat_fused_bg.shape),
                    "feat_fused_rigid_shape": list(feat_fused_rigid.shape),
                    "expected_dim": self.feat_3d_dim + self.feat_2d_dim + (1 if self.include_visibility else 0),
                    "bg_dim_match": feat_fused_bg.shape[-1] == (self.feat_3d_dim + self.feat_2d_dim + (1 if self.include_visibility else 0)),
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H4",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        return feat_fused_bg, feat_fused_rigid
