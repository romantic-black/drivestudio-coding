import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

try:
    from torchvision import models
except ImportError:  # pragma: no cover - torchvision is expected in training envs
    models = None


def _resolve_resnet_weights(backbone: str, pretrained: bool):
    if not pretrained or models is None:
        return None
    weights_attr = f"{backbone.upper()}_Weights"
    weights_enum = getattr(models, weights_attr, None)
    if weights_enum is None:
        return None
    return getattr(weights_enum, "DEFAULT", None)


class ImageFeatureExtractor(nn.Module):
    """Lightweight CNN to extract 2D features from source images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        backbone: str = "resnet18",
        feature_resolution: float = 0.25,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.feature_resolution = feature_resolution
        self.out_channels = out_channels
        self.backbone, backbone_out_dim = self._build_backbone(
            backbone=backbone,
            in_channels=in_channels,
            pretrained=pretrained,
        )
        self.proj = nn.Conv2d(backbone_out_dim, out_channels, kernel_size=1)
        if device is not None:
            self.to(device)

    def _build_backbone(
        self, backbone: str, in_channels: int, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        if models is not None and backbone.lower().startswith("resnet"):
            resnet_fn = getattr(models, backbone, None)
            if resnet_fn is None:
                raise ValueError(f"Unsupported backbone: {backbone}")
            weights = _resolve_resnet_weights(backbone, pretrained)
            try:
                resnet = resnet_fn(weights=weights)
            except TypeError:
                resnet = resnet_fn(pretrained=pretrained)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(
                    in_channels,
                    resnet.conv1.out_channels,
                    kernel_size=resnet.conv1.kernel_size,
                    stride=resnet.conv1.stride,
                    padding=resnet.conv1.padding,
                    bias=resnet.conv1.bias is not None,
                )
            layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1]
            out_dim = resnet.layer1[-1].conv2.out_channels
            if self.feature_resolution <= 0.125:
                layers.append(resnet.layer2)
                out_dim = resnet.layer2[-1].conv2.out_channels
            if self.feature_resolution <= 0.0625:
                layers.append(resnet.layer3)
                out_dim = resnet.layer3[-1].conv2.out_channels
            backbone_net = nn.Sequential(*layers)
            return backbone_net, out_dim

        # Fallback lightweight CNN when torchvision is unavailable or backbone not supported
        stride = 2 if self.feature_resolution >= 0.25 else 4
        channels = [max(out_channels, 32), max(out_channels, 48)]
        backbone_net = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        )
        return backbone_net, channels[1]

    def get_feature_resolution(self, image_height: int, image_width: int) -> Tuple[int, int]:
        h = max(1, int(round(image_height * self.feature_resolution)))
        w = max(1, int(round(image_width * self.feature_resolution)))
        return h, w

    def forward(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        # #region agent log
        import json
        import time
        import os
        _debug_log_path = "/root/drivestudio-coding/.cursor/debug.log"
        try:
            entry = {
                "timestamp": int(time.time() * 1000),
                "location": "image_feature_extractor.py:forward",
                "message": "2D feature extraction start",
                "data": {
                    "num_images": len(images),
                    "image_shapes": [list(img.shape) if isinstance(img, torch.Tensor) else None for img in images],
                },
                "sessionId": "debug-session",
                "runId": "initial",
                "hypothesisId": "H1",
            }
            with open(_debug_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        if len(images) == 0:
            return []
        processed = []
        for img in images:
            # Handle different input formats
            if img.dim() == 3:
                # [H, W, C] -> [C, H, W] for CNN
                if img.shape[-1] == 3 or img.shape[-1] == 1:  # HWC format
                    img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                # Now img is [C, H, W], add batch dimension
                processed.append(img.unsqueeze(0))  # [C, H, W] -> [1, C, H, W]
            elif img.dim() == 4:
                if img.shape[1] == 3 or img.shape[1] == 1:  # Already [B, C, H, W]
                    processed.append(img)
                elif img.shape[-1] == 3 or img.shape[-1] == 1:  # [B, H, W, C] format
                    img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    processed.append(img)
                else:
                    raise ValueError(f"Unexpected 4D image shape: {img.shape}")
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

        x = torch.cat(processed, dim=0).to(self.proj.weight.device)
        
        # #region agent log
        try:
            if torch.cuda.is_available():
                entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "image_feature_extractor.py:forward",
                    "message": "Before backbone forward",
                    "data": {
                        "batch_shape": list(x.shape),
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    },
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H1",
                }
                with open(_debug_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion
        
        feats = self.backbone(x)
        feats = self.proj(feats)

        target_h, target_w = self.get_feature_resolution(x.shape[-2], x.shape[-1])
        if feats.shape[-2] != target_h or feats.shape[-1] != target_w:
            feats = F.interpolate(feats, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # #region agent log
        try:
            if torch.cuda.is_available():
                feat_stats = {
                    "mean": float(feats.mean().item()),
                    "std": float(feats.std().item()),
                    "min": float(feats.min().item()),
                    "max": float(feats.max().item()),
                }
                entry = {
                    "timestamp": int(time.time() * 1000),
                    "location": "image_feature_extractor.py:forward",
                    "message": "After feature extraction",
                    "data": {
                        "feat_shape": list(feats.shape),
                        "target_resolution": [target_h, target_w],
                        "feat_stats": feat_stats,
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    },
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "H1",
                }
                with open(_debug_log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        # #endregion

        return [feats[i] for i in range(feats.shape[0])]
