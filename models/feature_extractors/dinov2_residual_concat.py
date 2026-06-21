from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .dinov2_unet_fusion import DINOv2BackboneAdapter
from .image_feature_extractor import ImageFeatureExtractor


class DINOv2ResidualConcatExtractor(nn.Module):
    """Frozen DINOv2 semantic features concatenated with trainable residual UNet features."""

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
        residual_base_channels: int = 32,
        residual_feature_downscale: int = 1,
        residual_depth: int = 3,
        residual_bilinear: bool = True,
        residual_norm: str = "batchnorm",
        concat_order: Sequence[str] = ("residual", "dino"),
        normalize_dino: str = "fixed_layernorm",
    ) -> None:
        super().__init__()
        if int(residual_in_channels) != 6:
            raise ValueError(
                "DINOv2ResidualConcatExtractor requires residual branch in_channels=6 "
                f"(RGB + rendered RGB), got {residual_in_channels}."
            )
        order = tuple(str(x).lower() for x in concat_order)
        if order != ("residual", "dino"):
            raise ValueError("DINOv2ResidualConcatExtractor currently requires concat.order=[residual, dino].")
        norm_l = str(normalize_dino).lower()
        if norm_l not in {"fixed_layernorm", "none", "identity"}:
            raise ValueError(f"unsupported DINO concat normalize_dino={normalize_dino!r}")

        self.dino_out_channels = int(dino_out_channels)
        self.residual_feat_channels = int(residual_feat_channels)
        self.out_channels = int(self.residual_feat_channels + self.dino_out_channels)
        self.concat_order = order
        self.normalize_dino_mode = norm_l
        self.dino_freeze_adapter = bool(dino_freeze_adapter)

        self.residual_unet = ImageFeatureExtractor(
            in_channels=int(residual_in_channels),
            feat_channels=int(residual_feat_channels),
            base_channels=int(residual_base_channels),
            feature_downscale=int(residual_feature_downscale),
            depth=int(residual_depth),
            bilinear=bool(residual_bilinear),
            norm=str(residual_norm),
        )
        self.dino_adapter = DINOv2BackboneAdapter(
            model_name=str(dino_model_name),
            pretrained=bool(dino_pretrained),
            weights_path=dino_weights_path,
            freeze_backbone=bool(dino_freeze),
            out_channels=int(dino_out_channels),
            intermediate_layers=dino_intermediate_layers,
            pad_to_patch_multiple=int(dino_pad_to_patch_multiple),
        )
        self.dino_norm = nn.LayerNorm(int(dino_out_channels), elementwise_affine=False) if norm_l == "fixed_layernorm" else nn.Identity()
        self._set_dino_adapter_trainable(trainable=not self.dino_freeze_adapter)

    def _set_dino_adapter_trainable(self, *, trainable: bool) -> None:
        for param in self.dino_adapter.parameters():
            param.requires_grad_(bool(trainable))
        if not bool(trainable):
            self.dino_adapter.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dino_freeze_adapter:
            self.dino_adapter.eval()
        return self

    def get_feature_resolution(self, height: int, width: int) -> tuple[int, int]:
        return self.residual_unet.get_feature_resolution(int(height), int(width))

    def _to_nchw_6(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"DINOv2ResidualConcatExtractor expects 4D input, got {tuple(images.shape)}")
        if int(images.shape[1]) == 6:
            return images
        if int(images.shape[-1]) == 6:
            return images.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "DINOv2ResidualConcatExtractor expects 6-channel input tensor shaped [B,6,H,W] or [B,H,W,6], "
            f"got {tuple(images.shape)}"
        )

    def dino_fingerprint(self) -> tuple:
        return (
            self.dino_adapter.fingerprint(),
            int(self.dino_out_channels),
            str(self.normalize_dino_mode),
        )

    def dino_adapter_has_trainable_params(self) -> bool:
        return any(bool(p.requires_grad) for p in self.dino_adapter.parameters())

    def extract_residual_feature(self, images: torch.Tensor) -> torch.Tensor:
        x6 = self._to_nchw_6(images)
        return self.residual_unet(x6)

    def extract_dino_feature(
        self,
        rgb: torch.Tensor,
        *,
        target_hw: tuple[int, int],
        detach: bool = True,
    ) -> torch.Tensor:
        if rgb.dim() != 4:
            raise ValueError(f"DINO cache expects 4D RGB tensor, got {tuple(rgb.shape)}")
        if int(rgb.shape[1]) != 3:
            if int(rgb.shape[-1]) == 3:
                rgb = rgb.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"DINO cache expects RGB [B,3,H,W] or [B,H,W,3], got {tuple(rgb.shape)}")
        if bool(detach):
            with torch.no_grad():
                return self._normalize_dino(self.dino_adapter(rgb, target_hw=target_hw)).detach()
        return self._normalize_dino(self.dino_adapter(rgb, target_hw=target_hw))

    def extract_dino_backbone_intermediates(self, rgb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if rgb.dim() != 4:
            raise ValueError(f"DINO cache expects 4D RGB tensor, got {tuple(rgb.shape)}")
        if int(rgb.shape[1]) != 3:
            if int(rgb.shape[-1]) == 3:
                rgb = rgb.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"DINO cache expects RGB [B,3,H,W] or [B,H,W,3], got {tuple(rgb.shape)}")
        return self.dino_adapter.extract_backbone_intermediates(rgb)

    def adapt_dino_backbone_intermediates(
        self,
        feats: Sequence[torch.Tensor],
        *,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        return self._normalize_dino(self.dino_adapter.adapt_backbone_intermediates(feats, target_hw=target_hw))

    def _normalize_dino(self, dino_feat: torch.Tensor) -> torch.Tensor:
        return self.dino_norm(dino_feat)

    def fuse_features(self, dino_feat: torch.Tensor, residual_feat: torch.Tensor) -> torch.Tensor:
        dino_feat = dino_feat.to(device=residual_feat.device, dtype=residual_feat.dtype)
        if tuple(dino_feat.shape[:3]) != tuple(residual_feat.shape[:3]):
            raise ValueError(
                "DINO feature shape mismatch: "
                f"dino={tuple(dino_feat.shape)} residual={tuple(residual_feat.shape)}"
            )
        return torch.cat([residual_feat, dino_feat], dim=-1).contiguous()

    def forward(self, images: torch.Tensor, *, cached_dino: torch.Tensor | None = None) -> torch.Tensor:
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
        return self.fuse_features(dino_feat, residual)


__all__ = ["DINOv2ResidualConcatExtractor"]
