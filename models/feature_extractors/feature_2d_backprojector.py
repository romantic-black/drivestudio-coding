from typing import List, Optional, Tuple
import json
import time

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
        # #region agent log
        _debug_log_path = "/root/drivestudio-coding/.cursor/debug.log"
        try:
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "feature_2d_backprojector.py:forward",
                "message": "2D backprojection start",
                "data": {
                    "num_views": len(features_2d),
                    "num_gaussians": num_gaussians,
                    "feature_channels": self.feature_channels,
                    "feat_shapes": [list(f.shape) for f in features_2d],
                    "idx_shapes": [list(idx.shape) for idx in gaussian_indices],
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H3",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if len(features_2d) != len(gaussian_indices):
            raise ValueError("features_2d and gaussian_indices must have the same length.")
        device = features_2d[0].device
        dtype = features_2d[0].dtype
        accum_dtype = torch.float32 if dtype.is_floating_point else dtype
        num = torch.zeros((num_gaussians, self.feature_channels), device=device, dtype=accum_dtype)
        den = torch.zeros((num_gaussians,), device=device, dtype=accum_dtype)
        
        # #region agent log
        try:
            if torch.cuda.is_available():
                entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "feature_2d_backprojector.py:forward",
                    "message": "Before accumulation loop",
                    "data": {
                        "accumulator_shape": [num_gaussians, self.feature_channels],
                        "accum_dtype": str(accum_dtype),
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    },
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H3",
                }
                with open(_debug_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion

        total_valid_contribs = 0
        for view_idx, (feat_map, idx_map, w_map) in enumerate(zip(features_2d, gaussian_indices, alpha_t_weights)):
            # #region agent log
            try:
                if view_idx < 2:  # Log first 2 views
                    entry = {
                        "timestamp": int(time.time() * 1000),
                        "location": "feature_2d_backprojector.py:forward",
                        "message": f"Before processing view {view_idx}",
                        "data": {
                            "view_idx": view_idx,
                            "feat_map_shape": list(feat_map.shape),
                            "idx_map_shape": list(idx_map.shape),
                            "w_map_shape": list(w_map.shape),
                        },
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H3",
                    }
                    with open(_debug_log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            # feat_map is [C, H, W], convert to [H, W, C] then flatten to [H*W, C]
            if feat_map.dim() == 3:
                # [C, H, W] -> [H, W, C]
                feat_flat = feat_map.permute(1, 2, 0).reshape(-1, self.feature_channels).to(accum_dtype)
                feat_h, feat_w = feat_map.shape[1], feat_map.shape[2]
            else:
                raise ValueError(f"Expected 3D feature map, got shape {feat_map.shape}")
            
            # idx_map and w_map should be [H, W, K]
            # Verify spatial dimensions match
            if idx_map.dim() == 3:
                idx_h, idx_w = idx_map.shape[0], idx_map.shape[1]
                if idx_h != feat_h or idx_w != feat_w:
                    # Resize idx_map and w_map to match feat_map spatial dimensions
                    import torch.nn.functional as F
                    idx_map_resized = F.interpolate(
                        idx_map.permute(2, 0, 1).float().unsqueeze(0),  # [1, K, H, W]
                        size=(feat_h, feat_w),
                        mode='nearest'
                    ).squeeze(0).permute(1, 2, 0).long()  # [H, W, K]
                    w_map_resized = F.interpolate(
                        w_map.permute(2, 0, 1).unsqueeze(0),  # [1, K, H, W]
                        size=(feat_h, feat_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # [H, W, K]
                    idx_map = idx_map_resized
                    w_map = w_map_resized
                    idx_h, idx_w = feat_h, feat_w
            else:
                raise ValueError(f"Expected 3D index map, got shape {idx_map.shape}")
            
            idx_flat = idx_map.reshape(-1, idx_map.shape[-1]).long()
            w_flat = w_map.reshape(-1, w_map.shape[-1]).to(accum_dtype)
            
            # #region agent log
            try:
                if view_idx < 2:  # Log first 2 views
                    entry = {
                        "timestamp": int(time.time() * 1000),
                        "location": "feature_2d_backprojector.py:forward",
                        "message": f"After flattening view {view_idx}",
                        "data": {
                            "view_idx": view_idx,
                            "feat_flat_shape": list(feat_flat.shape),
                            "idx_flat_shape": list(idx_flat.shape),
                            "w_flat_shape": list(w_flat.shape),
                        },
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H3",
                    }
                    with open(_debug_log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
            # #endregion
            
            view_valid_count = 0
            for k in range(idx_flat.shape[1]):
                idx_k = idx_flat[:, k]
                valid = idx_k >= 0
                if not valid.any():
                    continue
                
                # #region agent log
                try:
                    if view_idx < 2 and k == 0:  # Log first view, first k
                        entry = {
                            "timestamp": int(time.time() * 1000),
                            "location": "feature_2d_backprojector.py:forward",
                            "message": f"Before indexing view {view_idx} k {k}",
                            "data": {
                                "view_idx": view_idx,
                                "k": k,
                                "idx_k_shape": list(idx_k.shape),
                                "valid_shape": list(valid.shape),
                                "valid_count": int(valid.sum().item()),
                                "feat_flat_shape": list(feat_flat.shape),
                            },
                            "sessionId": "debug-session",
                            "runId": "initial",
                            "hypothesisId": "H3",
                        }
                        with open(_debug_log_path, "a") as f:
                            f.write(json.dumps(entry) + "\n")
                except Exception:
                    pass
                # #endregion
                
                idx_valid = idx_k[valid]
                w_valid = w_flat[valid, k]
                feat_valid = feat_flat[valid]
                weighted = feat_valid * w_valid.unsqueeze(-1)
                num = torch.index_add(num, 0, idx_valid, weighted)
                den = torch.index_add(den, 0, idx_valid, w_valid)
                view_valid_count += valid.sum().item()
            total_valid_contribs += view_valid_count
            
            # #region agent log
            try:
                if view_idx < 2 or view_idx == len(features_2d) - 1:  # Log first 2 and last view
                    den_nonzero = (den > 1e-8).sum().item()
                    entry = {
                        "timestamp": int(time.time() * 1000),
                        "location": "feature_2d_backprojector.py:forward",
                        "message": f"After view {view_idx} accumulation",
                        "data": {
                            "view_idx": view_idx,
                            "view_valid_contribs": view_valid_count,
                            "den_nonzero_count": den_nonzero,
                            "den_mean": float(den[den > 1e-8].mean().item()) if den_nonzero > 0 else 0.0,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                        },
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "H3",
                    }
                    with open(_debug_log_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
            # #endregion

        feat_all = (num / (den.unsqueeze(-1) + self.eps)).to(dtype)
        vis_all = torch.clamp(den, 0.0, 1.0).to(dtype)
        
        # #region agent log
        try:
            den_nonzero = (den > 1e-8).sum().item()
            feat_stats = {
                "mean": float(feat_all[den > 1e-8].mean().item()) if den_nonzero > 0 else 0.0,
                "std": float(feat_all[den > 1e-8].std().item()) if den_nonzero > 0 else 0.0,
                "min": float(feat_all[den > 1e-8].min().item()) if den_nonzero > 0 else 0.0,
                "max": float(feat_all[den > 1e-8].max().item()) if den_nonzero > 0 else 0.0,
            }
            vis_stats = {
                "mean": float(vis_all.mean().item()),
                "std": float(vis_all.std().item()),
                "min": float(vis_all.min().item()),
                "max": float(vis_all.max().item()),
                "nonzero_count": den_nonzero,
                "nonzero_ratio": den_nonzero / num_gaussians if num_gaussians > 0 else 0.0,
            }
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "feature_2d_backprojector.py:forward",
                "message": "After normalization",
                "data": {
                    "feat_all_shape": list(feat_all.shape),
                    "feat_stats": feat_stats,
                    "vis_stats": vis_stats,
                    "total_valid_contribs": total_valid_contribs,
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H3",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion

        feat_2d_bg = feat_all[bg_indices] if bg_indices is not None else feat_all
        vis_bg = vis_all[bg_indices] if bg_indices is not None else vis_all
        if rigid_indices is not None and rigid_indices.numel() > 0:
            feat_2d_rigid = feat_all[rigid_indices]
            vis_rigid = vis_all[rigid_indices]
        else:
            feat_2d_rigid = torch.zeros(0, self.feature_channels, device=device, dtype=dtype)
            vis_rigid = torch.zeros(0, device=device, dtype=dtype)
        
        # #region agent log
        try:
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "feature_2d_backprojector.py:forward",
                "message": "After splitting bg/rigid",
                "data": {
                    "feat_2d_bg_shape": list(feat_2d_bg.shape),
                    "feat_2d_rigid_shape": list(feat_2d_rigid.shape),
                    "vis_bg_mean": float(vis_bg.mean().item()) if vis_bg.numel() > 0 else 0.0,
                    "vis_rigid_mean": float(vis_rigid.mean().item()) if vis_rigid.numel() > 0 else 0.0,
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H3",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion

        return feat_all, vis_all, feat_2d_bg, feat_2d_rigid, vis_bg, vis_rigid
