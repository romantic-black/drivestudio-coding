"""
Lightweight 2D CNN feature extractor for StreetForward using UNet architecture.

This module produces per-pixel features for a batch of images using a standard UNet
architecture with encoder-decoder structure and skip connections. It is designed to
be memory-efficient while remaining fully differentiable so gradients can flow back
to the input images when needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, channels: int) -> nn.Module:
    norm_l = str(norm).lower()
    if norm_l in {"batchnorm", "batch_norm", "bn"}:
        return nn.BatchNorm2d(int(channels))
    if norm_l in {"groupnorm", "group_norm", "gn"}:
        groups = min(8, int(channels))
        while groups > 1 and int(channels) % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, int(channels))
    if norm_l in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"unsupported ImageFeatureExtractor norm={norm!r}")


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU
    Standard building block for UNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batchnorm") -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concatenate -> DoubleConv
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After upsampling: in_channels, after concat with skip: in_channels + skip_channels
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After transpose conv: in_channels // 2, after concat: in_channels // 2 + skip_channels
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from decoder (to be upsampled)
            x2: Feature map from encoder (skip connection)
        """
        x1 = self.up(x1)
        # Handle size mismatch due to odd input dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ImageFeatureExtractor(nn.Module):
    """
    UNet architecture for 2D feature extraction.

    Standard UNet with encoder-decoder structure and skip connections.
    Outputs feature maps with the same spatial resolution as input (or downscaled
    if feature_downscale > 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 16,
        base_channels: int = 32,
        feature_downscale: int = 1,
        depth: int = 4,
        bilinear: bool = True,
        norm: str = "batchnorm",
    ) -> None:
        """
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            feat_channels: Number of output feature channels
            base_channels: Base number of channels in the first layer
            feature_downscale: Downscale factor for feature resolution (1 = no downscale)
            depth: Depth of UNet (number of down/up sampling levels, default: 4)
            bilinear: Use bilinear upsampling instead of transposed convolution
        """
        super().__init__()
        self.feature_downscale = max(int(feature_downscale), 1)
        self.depth = depth
        self.bilinear = bilinear
        self.norm = str(norm)

        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, base_channels, norm=self.norm)

        # Build encoder layers and track channel sizes for skip connections
        self.downs = nn.ModuleList()
        encoder_channels = [base_channels]  # Track channels at each level
        ch = base_channels
        for i in range(depth):
            ch_next = min(ch * 2, 256)  # Cap at 256 channels to keep it lightweight
            self.downs.append(Down(ch, ch_next, norm=self.norm))
            encoder_channels.append(ch_next)
            ch = ch_next

        # Decoder (upsampling path)
        # encoder_channels: [base, ch1, ch2, ch3, ch4] (e.g., [32, 64, 128, 256, 256])
        # For decoder, we go backwards: ch4 -> ch3 -> ch2 -> ch1 -> base -> feat_channels
        self.ups = nn.ModuleList()
        for i in range(depth):
            # Current decoder input channels (from previous decoder layer or bottleneck)
            ch_in = encoder_channels[-(i + 1)]  # Start from bottleneck
            # Skip connection channels (from corresponding encoder layer)
            ch_skip = encoder_channels[-(i + 2)]  # Corresponding encoder level
            # Output channels (same as skip for intermediate, feat_channels for final)
            ch_out = ch_skip if i < depth - 1 else feat_channels
            self.ups.append(Up(ch_in, ch_skip, ch_out, bilinear, norm=self.norm))

    def get_feature_resolution(self, height: int, width: int) -> tuple[int, int]:
        """
        Compute output feature resolution after optional downscaling.
        """
        if self.feature_downscale <= 1:
            return height, width
        return max(height // self.feature_downscale, 1), max(width // self.feature_downscale, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor shaped [B, C, H, W] or [B, H, W, C]
                   where C can be 3 (RGB) or 6 (RGB + rendered RGB)

        Returns:
            Feature maps shaped [B, H_feat, W_feat, C]
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {images.shape}")
        # Handle channels_last format: [B, H, W, C] -> [B, C, H, W]
        if images.shape[1] != 3 and images.shape[1] != 6 and images.shape[-1] in [3, 6]:
            images = images.permute(0, 3, 1, 2)

        # Optional downscaling before UNet
        if self.feature_downscale > 1:
            images = F.interpolate(
                images,
                scale_factor=1.0 / float(self.feature_downscale),
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )

        # Encoder path (with skip connections)
        x = self.inc(images)
        skip_connections = [x]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        # Decoder path (with skip connections)
        skip_connections = skip_connections[:-1]  # Remove the last one (bottleneck)
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i])

        # Convert from [B, C, H, W] to [B, H, W, C]
        return x.permute(0, 2, 3, 1)
