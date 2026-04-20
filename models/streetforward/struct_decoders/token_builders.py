from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StructTokenBuilder(nn.Module):
    def __init__(
        self,
        *,
        feat_2d_channels: int,
        param_dim: int,
        channels: int,
        param_embed_dim: int,
        branch_embed_dim: int,
        support_embed_dim: int,
        use_2d_feat: bool,
        use_support: bool,
        use_branch_embed: bool,
        use_param_embed: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_2d_feat = bool(use_2d_feat)
        self.use_support = bool(use_support)
        self.use_branch_embed = bool(use_branch_embed)
        self.use_param_embed = bool(use_param_embed)

        self.feat2d_proj = (
            nn.Linear(int(feat_2d_channels), self.channels)
            if self.use_2d_feat
            else None
        )
        self.support_proj = (
            nn.Sequential(
                nn.Linear(2, int(support_embed_dim)),
                nn.GELU(),
                nn.Linear(int(support_embed_dim), self.channels),
            )
            if self.use_support
            else None
        )
        if self.use_branch_embed:
            self.branch_embed = nn.Embedding(2, int(branch_embed_dim))
            self.branch_proj = nn.Linear(int(branch_embed_dim), self.channels)
        else:
            self.branch_embed = None
            self.branch_proj = None
        self.param_proj = (
            nn.Sequential(
                nn.Linear(int(param_dim), int(param_embed_dim)),
                nn.GELU(),
                nn.Linear(int(param_embed_dim), self.channels),
            )
            if self.use_param_embed
            else None
        )
        self.token_norm = nn.LayerNorm(self.channels)

    @staticmethod
    def _as_1d(name: str, t: torch.Tensor, n: int) -> torch.Tensor:
        flat = t.reshape(-1)
        if flat.shape[0] != n:
            raise ValueError(f"{name} must have {n} elements, got {flat.shape[0]}.")
        return flat

    def forward(
        self,
        *,
        feat_2d: torch.Tensor,
        acc_w: torch.Tensor,
        branch_id: torch.Tensor,
        param_vec: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feat_2d.dim() != 2:
            raise ValueError("feat_2d must be [N, C].")
        num_points = int(feat_2d.shape[0])
        acc_w = self._as_1d("acc_w", acc_w, num_points)
        branch_id = self._as_1d("branch_id", branch_id, num_points)
        if param_vec.dim() != 2 or int(param_vec.shape[0]) != num_points:
            raise ValueError("param_vec must be [N, param_dim].")
        if valid_mask is not None:
            valid_mask = self._as_1d("valid_mask", valid_mask, num_points)

        x = feat_2d.new_zeros((num_points, self.channels))

        if self.feat2d_proj is not None:
            x = x + self.feat2d_proj(feat_2d)

        if self.support_proj is not None:
            valid = (
                valid_mask.to(device=acc_w.device, dtype=acc_w.dtype)
                if valid_mask is not None
                else torch.ones_like(acc_w)
            )
            support_vec = torch.stack([torch.log1p(acc_w.clamp_min(0.0)), valid], dim=-1)
            x = x + self.support_proj(support_vec)

        if self.branch_embed is not None and self.branch_proj is not None:
            branch_id_long = branch_id.long()
            if bool(
                ((branch_id_long < 0) | (branch_id_long >= self.branch_embed.num_embeddings))
                .any()
                .item()
            ):
                raise ValueError("branch_id must be 0=bg or 1=rigid_in.")
            branch_vec = self.branch_embed(branch_id_long)
            x = x + self.branch_proj(branch_vec)

        if self.param_proj is not None:
            x = x + self.param_proj(param_vec)

        return self.token_norm(x)
