from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_feature_extractor import ImageFeatureExtractor


class DINOv2BackboneAdapter(nn.Module):
    """DINOv2 backbone adapter that outputs channels-last dense features."""

    def __init__(
        self,
        *,
        model_name: str = "vit_base_patch14_reg4_dinov2",
        pretrained: bool = True,
        weights_path: str | None = None,
        out_channels: int = 32,
        intermediate_layers: Sequence[int] = (4, 8, 11),
        pad_to_patch_multiple: int = 14,
        proj_channels: int = 128,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.weights_path = str(weights_path) if weights_path else None
        self.pad_to_patch_multiple = max(int(pad_to_patch_multiple), 1)
        self.intermediate_layers = [int(x) for x in intermediate_layers]
        self.freeze_backbone = bool(freeze_backbone)
        if len(self.intermediate_layers) < 1:
            raise ValueError("DINOv2BackboneAdapter requires at least one intermediate layer.")

        try:
            import timm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Stage5_2 dinov2_unet_fusion requires timm. Install timm>=1.0.26 to use register DINOv2 models."
            ) from e

        available_reg_models = set(timm.list_models("*reg4_dinov2*"))
        if self.model_name not in available_reg_models:
            version = str(getattr(timm, "__version__", "unknown"))
            hint = sorted(list(available_reg_models))[:8]
            raise RuntimeError(
                "Stage5_2 dinov2_unet_fusion preflight failed: requested "
                f"DINO model '{self.model_name}' is unavailable in timm=={version}. "
                "Expected a register model (e.g., 'vit_base_patch14_reg4_dinov2'). "
                f"Detected register-capable models: {hint}. Upgrade timm to >=1.0.26 if needed."
            )

        load_pretrained = bool(pretrained) and self.weights_path is None
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=load_pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )

        if self.weights_path is not None:
            ckpt = torch.load(self.weights_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    ckpt = ckpt["state_dict"]
                elif "model" in ckpt and isinstance(ckpt["model"], dict):
                    ckpt = ckpt["model"]
            if not isinstance(ckpt, dict):
                raise ValueError(f"Invalid DINO weights checkpoint format at {self.weights_path!r}.")
            missing, unexpected = self.backbone.load_state_dict(ckpt, strict=False)
            if len(unexpected) > 0:
                raise RuntimeError(
                    "Unexpected keys when loading DINO weights: "
                    f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                )

        if not hasattr(self.backbone, "forward_intermediates"):
            raise RuntimeError(
                "Selected timm DINO model does not implement forward_intermediates; "
                "upgrade timm to a version that supports ViT intermediates."
            )

        self._set_backbone_trainable(trainable=not self.freeze_backbone)

        embed_dim = int(getattr(self.backbone, "num_features", 0) or 0)
        if embed_dim <= 0:
            raise RuntimeError("Unable to infer DINO backbone embedding dimension (num_features).")

        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, int(proj_channels), kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=int(proj_channels)),
                    nn.GELU(),
                )
                for _ in self.intermediate_layers
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(int(proj_channels) * len(self.intermediate_layers), int(proj_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=int(proj_channels)),
            nn.GELU(),
            nn.Conv2d(int(proj_channels), int(out_channels), kernel_size=1, bias=True),
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _set_backbone_trainable(self, *, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(bool(trainable))
        if trainable:
            self.backbone.train(bool(self.training))
        else:
            self.backbone.eval()

    def set_freeze_backbone(self, freeze: bool) -> None:
        self.freeze_backbone = bool(freeze)
        self._set_backbone_trainable(trainable=not self.freeze_backbone)

    def fingerprint(self) -> tuple:
        return (
            str(self.model_name),
            str(self.weights_path),
            tuple(int(x) for x in self.intermediate_layers),
            int(self.pad_to_patch_multiple),
            int(self.fuse[-1].out_channels) if isinstance(self.fuse[-1], nn.Conv2d) else -1,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if self.pad_to_patch_multiple <= 1:
            return x, 0, 0
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ph = (self.pad_to_patch_multiple - h % self.pad_to_patch_multiple) % self.pad_to_patch_multiple
        pw = (self.pad_to_patch_multiple - w % self.pad_to_patch_multiple) % self.pad_to_patch_multiple
        if ph == 0 and pw == 0:
            return x, 0, 0
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, ph, pw

    def extract_backbone_intermediates(self, rgb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if rgb.dim() != 4:
            raise ValueError(f"DINOv2BackboneAdapter expects 4D tensor, got {tuple(rgb.shape)}")
        if int(rgb.shape[1]) != 3:
            raise ValueError(
                "DINOv2BackboneAdapter expects RGB input shaped [B,3,H,W]. "
                f"Got {tuple(rgb.shape)}"
            )

        x = (rgb - self.pixel_mean.to(dtype=rgb.dtype, device=rgb.device)) / self.pixel_std.to(dtype=rgb.dtype, device=rgb.device)
        x, _pad_h, _pad_w = self._pad_to_multiple(x)

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone.forward_intermediates(
                    x,
                    indices=self.intermediate_layers,
                    output_fmt="NCHW",
                    intermediates_only=True,
                )
        else:
            feats = self.backbone.forward_intermediates(
                x,
                indices=self.intermediate_layers,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        if not isinstance(feats, list) or len(feats) != len(self.intermediate_layers):
            raise RuntimeError("Unexpected DINO intermediates output format.")
        return tuple(feats)

    def adapt_backbone_intermediates(
        self,
        feats: Sequence[torch.Tensor],
        *,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        h_t, w_t = int(target_hw[0]), int(target_hw[1])
        proj_feats = []
        for i, feat in enumerate(feats):
            if not torch.is_tensor(feat) or feat.dim() != 4:
                raise RuntimeError(f"Invalid DINO intermediate at idx={i}: expected BCHW tensor.")
            proj_weight = next(self.proj[i].parameters())
            feat = feat.to(device=proj_weight.device, dtype=proj_weight.dtype)
            p = self.proj[i](feat)
            if int(p.shape[-2]) != h_t or int(p.shape[-1]) != w_t:
                p = F.interpolate(p, size=(h_t, w_t), mode="bilinear", align_corners=False)
            proj_feats.append(p)

        fused = self.fuse(torch.cat(proj_feats, dim=1))
        return fused.permute(0, 2, 3, 1).contiguous()

    def forward(self, rgb: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
        feats = self.extract_backbone_intermediates(rgb)
        return self.adapt_backbone_intermediates(feats, target_hw=target_hw)


class FusionNeck2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 48,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(out_channels), kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv2UNetFusionExtractor(nn.Module):
    """
    Stage5_2 2D extractor:
    - DINO branch consumes RGB only (first 3 channels)
    - Residual UNet branch consumes full 6-channel input
    - Fusion neck outputs channels-last [B, H_feat, W_feat, C]
    """

    def __init__(
        self,
        *,
        dino_model_name: str = "vit_base_patch14_reg4_dinov2",
        dino_pretrained: bool = True,
        dino_weights_path: str | None = None,
        dino_freeze: bool = True,
        dino_out_channels: int = 32,
        dino_intermediate_layers: Sequence[int] = (4, 8, 11),
        dino_pad_to_patch_multiple: int = 14,
        residual_in_channels: int = 6,
        residual_feat_channels: int = 32,
        residual_base_channels: int = 32,
        residual_feature_downscale: int = 1,
        residual_depth: int = 4,
        residual_bilinear: bool = True,
        fusion_hidden_channels: int = 64,
        fusion_out_channels: int = 48,
    ) -> None:
        super().__init__()
        if int(residual_in_channels) != 6:
            raise ValueError(
                "DINOv2UNetFusionExtractor requires residual branch in_channels=6 "
                f"(RGB + rendered RGB), got {residual_in_channels}."
            )

        self.residual_unet = ImageFeatureExtractor(
            in_channels=int(residual_in_channels),
            feat_channels=int(residual_feat_channels),
            base_channels=int(residual_base_channels),
            feature_downscale=int(residual_feature_downscale),
            depth=int(residual_depth),
            bilinear=bool(residual_bilinear),
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

        self.fusion_neck = FusionNeck2D(
            in_channels=int(dino_out_channels) + int(residual_feat_channels),
            hidden_channels=int(fusion_hidden_channels),
            out_channels=int(fusion_out_channels),
        )

    def get_feature_resolution(self, height: int, width: int) -> tuple[int, int]:
        return self.residual_unet.get_feature_resolution(height, width)

    def _to_nchw_6(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"DINOv2UNetFusionExtractor expects 4D input, got {tuple(images.shape)}")
        if int(images.shape[1]) == 6:
            return images
        if int(images.shape[-1]) == 6:
            return images.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "DINOv2UNetFusionExtractor expects 6-channel input tensor shaped [B,6,H,W] or [B,H,W,6], "
            f"got {tuple(images.shape)}"
        )

    def dino_fingerprint(self) -> tuple:
        return (
            self.dino_adapter.fingerprint(),
            tuple(int(x) for x in self.get_feature_resolution(128, 128)),
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
                return self.dino_adapter(rgb, target_hw=target_hw).detach()
        return self.dino_adapter(rgb, target_hw=target_hw)

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
        return self.dino_adapter.adapt_backbone_intermediates(feats, target_hw=target_hw)

    def fuse_features(self, dino_feat: torch.Tensor, residual_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([dino_feat, residual_feat], dim=-1)
        fused_nchw = fused.permute(0, 3, 1, 2).contiguous()
        out = self.fusion_neck(fused_nchw)
        return out.permute(0, 2, 3, 1).contiguous()

    def forward(self, images: torch.Tensor, *, cached_dino: torch.Tensor | None = None) -> torch.Tensor:
        x6 = self._to_nchw_6(images)
        rgb = x6[:, :3, :, :]
        unet_feat = self.residual_unet(x6)  # [B, Hf, Wf, C_u]
        target_hw = (int(unet_feat.shape[1]), int(unet_feat.shape[2]))
        dino_feat = cached_dino
        if dino_feat is None:
            dino_feat = self.extract_dino_feature(
                rgb,
                target_hw=target_hw,
                detach=not self.dino_adapter_has_trainable_params(),
            )
        else:
            dino_feat = dino_feat.to(device=unet_feat.device, dtype=unet_feat.dtype)
            if tuple(dino_feat.shape[:3]) != tuple(unet_feat.shape[:3]):
                raise ValueError(
                    "cached DINO feature shape mismatch: "
                    f"cached={tuple(dino_feat.shape)} residual={tuple(unet_feat.shape)}"
                )
        return self.fuse_features(dino_feat, unet_feat)
