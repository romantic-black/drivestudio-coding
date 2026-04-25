from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from models.feature_extractors.dinov2_unet_fusion import DINOv2BackboneAdapter, DINOv2UNetFusionExtractor


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 768
        self.dummy = nn.Parameter(torch.ones((1,), dtype=torch.float32))

    def forward_intermediates(
        self,
        x: torch.Tensor,
        *,
        indices,
        output_fmt: str,
        intermediates_only: bool,
    ):
        assert output_fmt == "NCHW"
        assert intermediates_only is True
        b, _c, h, w = x.shape
        hh = max(h // 14, 1)
        ww = max(w // 14, 1)
        return [torch.ones((b, 768, hh, ww), dtype=x.dtype, device=x.device) for _ in indices]


class _SpyDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_rgb: torch.Tensor | None = None

    def forward(self, rgb: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
        self.last_rgb = rgb
        b = int(rgb.shape[0])
        h, w = target_hw
        return torch.zeros((b, h, w, 32), dtype=rgb.dtype, device=rgb.device)


class _SpyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_multi: torch.Tensor | None = None

    def forward(self, multi: torch.Tensor) -> torch.Tensor:
        self.last_multi = multi
        b, _c, h, w = multi.shape
        return torch.zeros((b, h, w, 32), dtype=multi.dtype, device=multi.device)

    @staticmethod
    def get_feature_resolution(height: int, width: int) -> tuple[int, int]:
        return max(height // 2, 1), max(width // 2, 1)


@pytest.fixture
def fake_timm(monkeypatch):
    fake = types.SimpleNamespace()
    fake.__version__ = "1.0.26"
    fake.list_models = lambda pattern="": ["vit_base_patch14_reg4_dinov2"]
    fake.create_model = (
        lambda model_name, pretrained=False, num_classes=0, dynamic_img_size=False: _FakeBackbone()
    )
    monkeypatch.setitem(sys.modules, "timm", fake)
    return fake


def test_dino_adapter_preflight_fails_when_register_model_is_missing(monkeypatch):
    fake = types.SimpleNamespace()
    fake.__version__ = "0.9.5"
    fake.list_models = lambda pattern="": []
    fake.create_model = lambda *args, **kwargs: _FakeBackbone()
    monkeypatch.setitem(sys.modules, "timm", fake)

    with pytest.raises(RuntimeError, match="preflight failed"):
        DINOv2BackboneAdapter(
            model_name="vit_base_patch14_reg4_dinov2",
            pretrained=False,
        )


def test_dino_adapter_freezes_backbone_by_default(fake_timm):
    _ = fake_timm
    adapter = DINOv2BackboneAdapter(
        model_name="vit_base_patch14_reg4_dinov2",
        pretrained=False,
    )
    assert adapter.freeze_backbone is True
    assert all(not p.requires_grad for p in adapter.backbone.parameters())
    adapter.train()
    assert adapter.backbone.training is False


def test_dino_adapter_can_unfreeze_backbone(fake_timm):
    _ = fake_timm
    adapter = DINOv2BackboneAdapter(
        model_name="vit_base_patch14_reg4_dinov2",
        pretrained=False,
        freeze_backbone=False,
    )
    assert adapter.freeze_backbone is False
    assert all(p.requires_grad for p in adapter.backbone.parameters())
    adapter.train()
    assert adapter.backbone.training is True
    adapter.set_freeze_backbone(True)
    assert adapter.freeze_backbone is True
    assert all(not p.requires_grad for p in adapter.backbone.parameters())
    assert adapter.backbone.training is False


def test_fusion_extractor_contract_rgb_only_for_dino_and_full_6ch_for_unet(fake_timm):
    _ = fake_timm
    extractor = DINOv2UNetFusionExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        fusion_out_channels=48,
    )
    assert extractor.dino_adapter.freeze_backbone is True
    spy_dino = _SpyDINO()
    spy_unet = _SpyUNet()
    extractor.dino_adapter = spy_dino
    extractor.residual_unet = spy_unet

    x_nchw = torch.randn(2, 6, 8, 10)
    out = extractor(x_nchw)
    assert tuple(out.shape) == (2, 8, 10, 48)
    assert spy_dino.last_rgb is not None
    assert spy_unet.last_multi is not None
    assert tuple(spy_dino.last_rgb.shape) == (2, 3, 8, 10)
    assert tuple(spy_unet.last_multi.shape) == (2, 6, 8, 10)
    assert torch.allclose(spy_dino.last_rgb, x_nchw[:, :3])
    assert torch.allclose(spy_unet.last_multi, x_nchw)


def test_fusion_extractor_accepts_channels_last_and_exposes_feature_resolution(fake_timm):
    _ = fake_timm
    extractor = DINOv2UNetFusionExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        dino_freeze=False,
        fusion_out_channels=48,
    )
    assert extractor.dino_adapter.freeze_backbone is False
    spy_dino = _SpyDINO()
    spy_unet = _SpyUNet()
    extractor.dino_adapter = spy_dino
    extractor.residual_unet = spy_unet

    x_nhwc = torch.randn(2, 7, 9, 6)
    out = extractor(x_nhwc)
    assert tuple(out.shape) == (2, 7, 9, 48)
    assert spy_dino.last_rgb is not None
    assert spy_unet.last_multi is not None
    assert tuple(spy_dino.last_rgb.shape) == (2, 3, 7, 9)
    assert tuple(spy_unet.last_multi.shape) == (2, 6, 7, 9)

    assert extractor.get_feature_resolution(11, 13) == (5, 6)
