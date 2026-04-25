from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.streetforward.struct_decoders.common import (
    StructDecoderInput,
    StructDecoderOutput,
    StreetForwardStructDecoderBase,
    normalize_params_for_embed,
)
from models.streetforward.struct_decoders.token_builders import StructTokenBuilder


class FarBranchMLPStructDecoder(StreetForwardStructDecoderBase):
    def __init__(
        self,
        *,
        feat_2d_channels: int,
        out_channels: int,
        param_dim: int = 17,
        branch_embed_dim: int = 8,
        support_embed_dim: int = 8,
        param_embed_dim: int = 32,
        channels: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        norm: str = "layernorm",
        act: str = "gelu",
        use_2d_feat: bool = True,
        use_support: bool = True,
        use_branch_embed: bool = True,
        use_param_embed: bool = True,
        zero_invalid_2d_feat: bool = True,
        history_dim: int = 0,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("FarBranchMLPStructDecoder num_layers must be >= 1.")
        self.out_channels = int(out_channels)
        self.param_dim = int(param_dim)
        self.zero_invalid_2d_feat = bool(zero_invalid_2d_feat)
        self.history_dim = int(history_dim)
        self.token_builder = StructTokenBuilder(
            feat_2d_channels=int(feat_2d_channels),
            param_dim=int(param_dim),
            channels=int(channels),
            param_embed_dim=int(param_embed_dim),
            branch_embed_dim=int(branch_embed_dim),
            support_embed_dim=int(support_embed_dim),
            use_2d_feat=bool(use_2d_feat),
            use_support=bool(use_support),
            use_branch_embed=bool(use_branch_embed),
            use_param_embed=bool(use_param_embed),
        )

        act_layer: nn.Module
        if str(act).lower() == "gelu":
            act_layer = nn.GELU()
        elif str(act).lower() == "relu":
            act_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported far decoder act={act!r}")

        blocks = []
        in_dim = int(channels) + self.history_dim
        for _ in range(int(num_layers) - 1):
            blocks.append(nn.Linear(in_dim, int(hidden_dim)))
            if str(norm).lower() == "layernorm":
                blocks.append(nn.LayerNorm(int(hidden_dim)))
            blocks.append(act_layer)
            in_dim = int(hidden_dim)
        blocks.append(nn.Linear(in_dim, self.out_channels))
        self.output = nn.Sequential(*blocks)

    def forward(
        self,
        x: StructDecoderInput,
        *,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> StructDecoderOutput:
        _ = batch_offsets
        num_points = int(x.coords.shape[0])
        if num_points <= 0:
            return StructDecoderOutput(feat=x.coords.new_zeros((0, self.out_channels)))
        num_far0 = int(x.split_bg)  # far-local branch 0: distant
        num_far1 = int(x.split_rigid_in)  # far-local branch 1: rigid_out
        if int(num_far0 + num_far1) != num_points:
            raise ValueError("FarBranchMLPStructDecoder split mismatch with total points.")
        if bool((x.branch_id < 0).any().item()) or bool((x.branch_id > 1).any().item()):
            raise ValueError("FarBranchMLPStructDecoder branch_id must be in far-local {0,1}.")

        param_vec = normalize_params_for_embed(
            x.params_for_embed,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        if int(param_vec.shape[-1]) != int(self.param_dim):
            raise RuntimeError(
                f"FarBranchMLPStructDecoder param dim mismatch: got {param_vec.shape[-1]}, expected {self.param_dim}."
            )

        support_thr_0 = float(x.meta.get("support_threshold_distant", x.meta.get("support_threshold_0", 0.0)))
        support_thr_1 = float(
            x.meta.get(
                "support_threshold_rigid_out",
                x.meta.get("support_threshold_1", support_thr_0),
            )
        )
        support_thr = torch.where(
            x.branch_id.long() == 0,
            x.acc_w.new_full((num_points,), support_thr_0),
            x.acc_w.new_full((num_points,), support_thr_1),
        )
        valid = x.acc_w > support_thr
        feat_2d = x.feat_2d
        if self.zero_invalid_2d_feat:
            feat_2d = feat_2d * valid.to(dtype=feat_2d.dtype).unsqueeze(-1)
        token = self.token_builder(
            feat_2d=feat_2d,
            acc_w=x.acc_w,
            branch_id=x.branch_id,
            param_vec=param_vec,
            valid_mask=valid.to(dtype=feat_2d.dtype),
        )
        history_embed = x.meta.get("history_embed")
        if history_embed is not None:
            if not torch.is_tensor(history_embed):
                raise TypeError("FarBranchMLPStructDecoder meta['history_embed'] must be a tensor when provided.")
            if int(history_embed.shape[0]) != num_points:
                raise ValueError("FarBranchMLPStructDecoder history_embed row count mismatch.")
            if int(history_embed.shape[-1]) != self.history_dim:
                raise ValueError(
                    f"FarBranchMLPStructDecoder history_embed dim mismatch: got {history_embed.shape[-1]}, expected {self.history_dim}."
                )
            token = torch.cat([token, history_embed.to(dtype=token.dtype, device=token.device)], dim=-1)
        elif self.history_dim > 0:
            token = torch.cat([token, token.new_zeros((num_points, self.history_dim))], dim=-1)
        feat = self.output(token)
        return StructDecoderOutput(
            feat=feat,
            aux={
                "num_struct_points": int(num_points),
                "num_struct_voxels": 0,
                "far_decoder": 1.0,
            },
        )


__all__ = ["FarBranchMLPStructDecoder"]
