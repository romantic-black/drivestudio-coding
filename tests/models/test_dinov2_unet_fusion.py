from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from models.feature_extractors.dinov2_residual_concat import DINOv2ResidualConcatExtractor
from models.feature_extractors.dinov2_unet_fusion import DINOv2BackboneAdapter, DINOv2UNetFusionExtractor
from models.feature_extractors.residual_only import ResidualOnlyFeatureExtractor
from models.iforward.dino_feature_cache import DINOFeatureCache


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
    def __init__(self, out_channels: int = 32, fill: float = 0.0) -> None:
        super().__init__()
        self.last_rgb: torch.Tensor | None = None
        self.out_channels = int(out_channels)
        self.fill = float(fill)

    def forward(self, rgb: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
        self.last_rgb = rgb
        b = int(rgb.shape[0])
        h, w = target_hw
        return torch.full((b, h, w, self.out_channels), self.fill, dtype=rgb.dtype, device=rgb.device)


class _SpyUNet(nn.Module):
    def __init__(self, out_channels: int = 32, fill: float = 0.0) -> None:
        super().__init__()
        self.last_multi: torch.Tensor | None = None
        self.out_channels = int(out_channels)
        self.fill = float(fill)

    def forward(self, multi: torch.Tensor) -> torch.Tensor:
        self.last_multi = multi
        b, _c, h, w = multi.shape
        return torch.full((b, h, w, self.out_channels), self.fill, dtype=multi.dtype, device=multi.device)

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


def test_residual_only_extractor_uses_6ch_unet_without_dino_or_fusion() -> None:
    extractor = ResidualOnlyFeatureExtractor(
        in_channels=6,
        feat_channels=16,
        base_channels=8,
        feature_downscale=1,
        depth=1,
        bilinear=True,
    )
    x = torch.randn(2, 6, 8, 10)
    out = extractor(x)
    assert tuple(out.shape) == (2, 8, 10, 16)
    assert not hasattr(extractor, "dino_adapter")
    assert not hasattr(extractor, "fusion_neck")


def test_dinov2_residual_concat_contract_order_and_shapes(fake_timm):
    _ = fake_timm
    extractor = DINOv2ResidualConcatExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        dino_out_channels=8,
        residual_feat_channels=10,
        residual_base_channels=8,
        normalize_dino="none",
    )
    spy_dino = _SpyDINO(out_channels=8, fill=2.0)
    spy_unet = _SpyUNet(out_channels=10, fill=1.0)
    extractor.dino_adapter = spy_dino
    extractor.residual_unet = spy_unet
    x = torch.randn(2, 6, 8, 10)
    out = extractor(x)
    assert tuple(out.shape) == (2, 8, 10, 18)
    assert torch.allclose(out[..., :10], torch.ones_like(out[..., :10]))
    assert torch.allclose(out[..., 10:], torch.full_like(out[..., 10:], 2.0))
    assert spy_dino.last_rgb is not None
    assert spy_unet.last_multi is not None
    assert torch.allclose(spy_dino.last_rgb, x[:, :3])
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out_nhwc = extractor(x_nhwc)
    assert tuple(out_nhwc.shape) == (2, 8, 10, 18)


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


def test_fusion_extractor_cached_dino_matches_uncached_and_keeps_grad(fake_timm):
    _ = fake_timm
    extractor = DINOv2UNetFusionExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        dino_out_channels=8,
        residual_feat_channels=8,
        residual_base_channels=8,
        fusion_hidden_channels=8,
        fusion_out_channels=4,
    )
    extractor.eval()
    x = torch.randn(1, 6, 32, 32)
    residual = extractor.extract_residual_feature(x)
    cached = extractor.extract_dino_feature(x[:, :3], target_hw=(int(residual.shape[1]), int(residual.shape[2])))
    assert cached.requires_grad is False
    out_cached = extractor(x, cached_dino=cached)
    out_uncached = extractor(x)
    assert torch.allclose(out_cached, out_uncached, atol=1.0e-6)
    loss = out_cached.square().mean()
    loss.backward()
    residual_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in extractor.residual_unet.parameters()
        if p.grad is not None
    )
    fusion_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in extractor.fusion_neck.parameters()
        if p.grad is not None
    )
    assert residual_grad > 0.0
    assert fusion_grad > 0.0


def test_dinov2_residual_concat_cached_dino_matches_uncached_and_freezes_dino(fake_timm):
    _ = fake_timm
    extractor = DINOv2ResidualConcatExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        dino_out_channels=8,
        residual_feat_channels=8,
        residual_base_channels=8,
        normalize_dino="fixed_layernorm",
    )
    assert extractor.dino_adapter_has_trainable_params() is False
    extractor.eval()
    x = torch.randn(1, 6, 32, 32)
    residual = extractor.extract_residual_feature(x)
    cached = extractor.extract_dino_feature(x[:, :3], target_hw=(int(residual.shape[1]), int(residual.shape[2])))
    assert cached.requires_grad is False
    out_cached = extractor(x, cached_dino=cached)
    out_uncached = extractor(x)
    assert torch.allclose(out_cached, out_uncached, atol=1.0e-6)
    loss = out_cached.square().mean()
    loss.backward()
    residual_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in extractor.residual_unet.parameters()
        if p.grad is not None
    )
    dino_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in extractor.dino_adapter.parameters()
        if p.grad is not None
    )
    assert residual_grad > 0.0
    assert dino_grad == 0.0
    assert all(not p.requires_grad for p in extractor.dino_adapter.parameters())


def test_fusion_extractor_cached_backbone_intermediates_keep_adapter_grad(fake_timm):
    _ = fake_timm
    extractor = DINOv2UNetFusionExtractor(
        dino_model_name="vit_base_patch14_reg4_dinov2",
        dino_pretrained=False,
        dino_out_channels=8,
        residual_feat_channels=8,
        residual_base_channels=8,
        fusion_hidden_channels=8,
        fusion_out_channels=4,
    )
    extractor.eval()
    x = torch.randn(1, 6, 32, 32)
    residual = extractor.extract_residual_feature(x)
    target_hw = (int(residual.shape[1]), int(residual.shape[2]))
    feats = extractor.extract_dino_backbone_intermediates(x[:, :3])
    assert isinstance(feats, tuple)
    cache = DINOFeatureCache(dtype="float16", cpu_pinned=False, cpu_max_items=1, gpu_max_items=1)
    cached, stats = cache.get_or_compute(
        key=("scene", 0),
        device=torch.device("cpu"),
        compute=lambda: feats,
        trainable=False,
    )
    assert stats.miss == 1.0
    assert isinstance(cached, tuple)
    assert all(item.dtype == torch.float16 for item in cached)
    dino_feat = extractor.adapt_dino_backbone_intermediates(cached, target_hw=target_hw)
    out = extractor.fuse_features(dino_feat, residual)
    loss = out.square().mean()
    loss.backward()
    adapter_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in list(extractor.dino_adapter.proj.parameters()) + list(extractor.dino_adapter.fuse.parameters())
        if p.grad is not None
    )
    backbone_grad = sum(
        float(p.grad.detach().abs().sum().item())
        for p in extractor.dino_adapter.backbone.parameters()
        if p.grad is not None
    )
    assert adapter_grad > 0.0
    assert backbone_grad == 0.0


def test_dino_feature_cache_lru_and_trainable_guard() -> None:
    cache = DINOFeatureCache(dtype="float16", cpu_pinned=False, cpu_max_items=1, gpu_max_items=1)
    calls = {"n": 0}

    def compute() -> torch.Tensor:
        calls["n"] += 1
        return torch.full((1, 2, 2, 3), float(calls["n"]))

    out1, stats1 = cache.get_or_compute(key=("a",), device=torch.device("cpu"), compute=compute, trainable=False)
    out2, stats2 = cache.get_or_compute(key=("a",), device=torch.device("cpu"), compute=compute, trainable=False)
    out3, stats3 = cache.get_or_compute(key=("b",), device=torch.device("cpu"), compute=compute, trainable=False)
    out4, stats4 = cache.get_or_compute(key=("a",), device=torch.device("cpu"), compute=compute, trainable=False)
    assert stats1.miss == 1.0
    assert stats2.hit_l1 == 1.0
    assert stats3.miss == 1.0
    assert stats4.miss == 1.0
    assert out1.dtype == torch.float16
    assert out2.dtype == torch.float16
    assert out3.dtype == torch.float16
    assert out4.dtype == torch.float16
    assert calls["n"] == 3
    assert float(out1.mean().item()) == 1.0
    assert float(out2.mean().item()) == 1.0
    assert float(out3.mean().item()) == 2.0
    assert float(out4.mean().item()) == 3.0
    with pytest.raises(RuntimeError, match="cannot be used"):
        cache.get_or_compute(key=("c",), device=torch.device("cpu"), compute=compute, trainable=True)
