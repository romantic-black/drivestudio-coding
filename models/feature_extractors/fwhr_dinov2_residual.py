from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .dinov2_residual_concat import DINOv2ResidualConcatExtractor


@dataclass
class FWHRImageFeatures:
    context: torch.Tensor
    detail: torch.Tensor
    aux: Dict[str, Any]


def _group_norm_channels(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class FWHRDetailHead2D(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = 8, init_std: float = 1.0e-3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _group_norm_channels(int(in_channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True),
        )
        conv = self.net[-1]
        if isinstance(conv, nn.Conv2d):
            nn.init.normal_(conv.weight, mean=0.0, std=float(init_std))
            nn.init.zeros_(conv.bias)

    def forward(self, residual_nhwc: torch.Tensor) -> torch.Tensor:
        if residual_nhwc.dim() != 4:
            raise ValueError(f"FWHRDetailHead2D expects [B,H,W,C], got {tuple(residual_nhwc.shape)}")
        x = residual_nhwc.permute(0, 3, 1, 2).contiguous()
        y = self.net(x)
        return y.permute(0, 2, 3, 1).contiguous()


class FWHRDINOv2ResidualExtractor(DINOv2ResidualConcatExtractor):
    """FW-HR 2D frontend with 48D context and 8D fine residual detail."""

    def __init__(
        self,
        *,
        dino_model_name: str = "vit_base_patch14_reg4_dinov2",
        dino_pretrained: bool = True,
        dino_weights_path: str | None = None,
        dino_freeze: bool = True,
        dino_freeze_adapter: bool = True,
        dino_out_channels: int = 16,
        dino_intermediate_layers: Sequence[int] = (4, 8, 11),
        dino_pad_to_patch_multiple: int = 14,
        residual_in_channels: int = 6,
        residual_feat_channels: int = 32,
        residual_base_channels: int = 48,
        residual_feature_downscale: int = 1,
        residual_depth: int = 4,
        residual_bilinear: bool = True,
        residual_norm: str = "groupnorm",
        detail_channels: int = 8,
        detail_init_std: float = 1.0e-3,
        concat_order: Sequence[str] = ("residual", "dino"),
        normalize_dino: str = "fixed_layernorm",
    ) -> None:
        super().__init__(
            dino_model_name=dino_model_name,
            dino_pretrained=bool(dino_pretrained),
            dino_weights_path=dino_weights_path,
            dino_freeze=bool(dino_freeze),
            dino_freeze_adapter=bool(dino_freeze_adapter),
            dino_out_channels=int(dino_out_channels),
            dino_intermediate_layers=dino_intermediate_layers,
            dino_pad_to_patch_multiple=int(dino_pad_to_patch_multiple),
            residual_in_channels=int(residual_in_channels),
            residual_feat_channels=int(residual_feat_channels),
            residual_base_channels=int(residual_base_channels),
            residual_feature_downscale=int(residual_feature_downscale),
            residual_depth=int(residual_depth),
            residual_bilinear=bool(residual_bilinear),
            residual_norm=str(residual_norm),
            concat_order=concat_order,
            normalize_dino=str(normalize_dino),
        )
        self.detail_channels = int(detail_channels)
        self.detail_head = FWHRDetailHead2D(
            in_channels=int(residual_feat_channels),
            out_channels=int(detail_channels),
            init_std=float(detail_init_std),
        )

    def forward_fwhr(self, images: torch.Tensor, *, cached_dino: torch.Tensor | None = None) -> FWHRImageFeatures:
        x6 = self._to_nchw_6(images)
        rgb = x6[:, :3, :, :]
        residual = self.residual_unet(x6)
        target_hw = (int(residual.shape[1]), int(residual.shape[2]))
        if cached_dino is None:
            dino_feat = self.extract_dino_feature(
                rgb,
                target_hw=target_hw,
                detach=not self.dino_adapter_has_trainable_params(),
            )
        else:
            dino_feat = cached_dino.to(device=residual.device, dtype=residual.dtype)
            if tuple(dino_feat.shape[:3]) != tuple(residual.shape[:3]):
                raise ValueError(
                    "cached DINO feature shape mismatch: "
                    f"cached={tuple(dino_feat.shape)} residual={tuple(residual.shape)}"
                )
        context = self.fuse_features(dino_feat, residual)
        detail = self.detail_head(residual)
        aux = {
            "residual_feature_rms": residual.detach().float().square().mean().sqrt(),
            "dino_feature_rms": dino_feat.detach().float().square().mean().sqrt(),
            "detail_feature_rms": detail.detach().float().square().mean().sqrt(),
        }
        return FWHRImageFeatures(context=context, detail=detail, aux=aux)


__all__ = ["FWHRDINOv2ResidualExtractor", "FWHRImageFeatures"]
