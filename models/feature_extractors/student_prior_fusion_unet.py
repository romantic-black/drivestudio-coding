from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentPriorFusionUNet(nn.Module):
    def __init__(
        self,
        *,
        prior_dim: int,
        out_dim: int,
        base_dim: int,
        use_confidence: bool,
    ) -> None:
        super().__init__()
        if int(base_dim) % 8 != 0:
            raise ValueError("student_extractor.base_channels must be divisible by 8 for GroupNorm.")
        in_ch = 3 + int(prior_dim) + (1 if bool(use_confidence) else 0)
        c1 = int(base_dim)
        c2 = int(base_dim) * 2

        self.prior_dim = int(prior_dim)
        self.use_confidence = bool(use_confidence)
        self.enc1 = _DoubleConv(in_ch, c1)
        self.down = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.enc2 = _DoubleConv(c2, c2)
        self.up = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec = _DoubleConv(c1 + c1, c1)
        self.out = nn.Conv2d(c1, int(out_dim), kernel_size=1)

    @staticmethod
    def _to_nchw(t: torch.Tensor, *, expected_channels: Optional[int] = None) -> torch.Tensor:
        if t.dim() != 4:
            raise ValueError(f"StudentPriorFusionUNet expects 4D tensor, got shape={tuple(t.shape)}")
        if expected_channels is not None and int(t.shape[1]) == int(expected_channels):
            return t
        if expected_channels is not None and int(t.shape[-1]) == int(expected_channels):
            return t.permute(0, 3, 1, 2).contiguous()
        if int(t.shape[1]) <= 8:
            return t
        return t.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        *,
        render_rgb: torch.Tensor,
        prior_map: torch.Tensor,
        prior_conf: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_rgb = self._to_nchw(render_rgb, expected_channels=3)
        x_prior = self._to_nchw(prior_map, expected_channels=self.prior_dim)
        if x_prior.shape[-2:] != x_rgb.shape[-2:]:
            x_prior = F.interpolate(x_prior, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
        parts = [x_rgb, x_prior]
        if self.use_confidence:
            if prior_conf is None:
                raise ValueError("StudentPriorFusionUNet requires prior_conf when use_confidence=true.")
            x_conf = self._to_nchw(prior_conf, expected_channels=1)
            if x_conf.shape[-2:] != x_rgb.shape[-2:]:
                x_conf = F.interpolate(x_conf, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
            parts.append(x_conf)
        x = torch.cat(parts, dim=1)

        e1 = self.enc1(x)
        x2 = self.enc2(self.down(e1))
        x3 = self.up(x2)
        if x3.shape[-2:] != e1.shape[-2:]:
            x3 = F.interpolate(x3, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec(torch.cat([e1, x3], dim=1))
        out = self.out(y)
        return out.permute(0, 2, 3, 1).contiguous()


__all__ = ["StudentPriorFusionUNet"]
