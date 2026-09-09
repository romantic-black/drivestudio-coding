from __future__ import annotations

from types import SimpleNamespace

import torch

from models.iforward.parent_ptv3 import ParentPTv3Encoder
from models.iforward.parent_serialization import build_parent_serialized_layout
from models.iforward.parent_spatial_backbone import (
    ParentSpatialBackbone,
    ParentStructInput,
    Stage34ParentGeometryResidualAdapter,
    Stage6ParentParamSupportCodec,
    empty_parent_struct_input,
)
from models.iforward.trainer import IForwardTrainer


def _params(n: int) -> dict[str, torch.Tensor]:
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    return {
        "means": torch.linspace(0.0, 0.9, max(n, 1)).reshape(-1, 1).repeat(1, 3)[:n],
        "quats": quats,
        "scales_log": torch.zeros(n, 3),
        "opacity_logit": torch.zeros(n, 1),
        "sh_dc": torch.zeros(n, 3),
        "sh_rest": torch.zeros(n, 1, 3),
    }


def test_parent_serialization_roundtrip_and_no_cross_batch_patches():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ]
    )
    offsets = torch.tensor([3, 5])
    layout = build_parent_serialized_layout(
        coords,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        patch_size=2,
        order_name="z",
        batch_offsets=offsets,
    )
    x = torch.randn(5, 3)
    restored = x.index_select(0, layout.order).index_select(0, layout.inverse)
    assert torch.allclose(restored, x)
    batch_ids = torch.tensor([0, 0, 0, 1, 1])
    for patch_idx in range(layout.num_patches):
        valid = layout.pad_mask[patch_idx]
        rows = layout.order.reshape(layout.num_patches, layout.patch_size)[patch_idx][valid]
        assert int(batch_ids[rows].unique().numel()) <= 1


def test_parent_ptv3_layout_reuse_rebuild_and_backward():
    torch.manual_seed(4)
    encoder = ParentPTv3Encoder(dim=8, depth=2, num_heads=2, patch_size=4, use_xcpe=False)
    coords = torch.rand(7, 3)
    x = torch.randn(7, 8, requires_grad=True)
    out1, cache1, _ = encoder(
        x,
        coords=coords,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        batch_offsets=torch.tensor([7]),
    )
    out2, cache2, _ = encoder(
        x,
        coords=coords,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        batch_offsets=torch.tensor([7]),
        layout_cache=cache1,
    )
    out3, cache3, _ = encoder(
        x,
        coords=coords,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        batch_offsets=torch.tensor([7]),
    )
    x_short = torch.randn(5, 8, requires_grad=True)
    coords_short = coords[:5]
    out4, cache4, _ = encoder(
        x_short,
        coords=coords_short,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        batch_offsets=torch.tensor([5]),
        layout_cache=cache1,
    )
    assert cache2["z"] is cache1["z"]
    assert cache3["z"] is not cache1["z"]
    assert cache4["z"] is not cache1["z"]
    assert tuple(out4.shape) == (5, 8)
    loss = out1.pow(2).mean() + out2.abs().mean() + out3.square().mean() + out4.abs().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x_short.grad is not None
    assert torch.isfinite(x_short.grad).all()


def test_parent_spatial_backbone_empty_far_and_rigid_forward_backward():
    torch.manual_seed(5)
    backbone = ParentSpatialBackbone(
        context_dim=4,
        event_dim=8,
        token_dim=8,
        near_depth=1,
        near_heads=2,
        near_patch_size=4,
        use_xcpe=False,
    )
    parent_context = torch.randn(3, 4, requires_grad=True)
    near = ParentStructInput(
        parent_context=parent_context,
        support=torch.ones(3),
        valid=torch.ones(3, dtype=torch.bool),
        coords=torch.rand(3, 3),
        branch_id=torch.zeros(3, dtype=torch.long),
        params_for_embed=_params(3),
        split_0=3,
        split_1=0,
    )
    far = empty_parent_struct_input(ref=parent_context, context_dim=4, sh_rest_bases=1, path="far")
    event, _cache = backbone(
        near_in=near,
        far_in=far,
        route=SimpleNamespace(
            S=torch.zeros(0, dtype=torch.long),
            S_in=torch.zeros(0, dtype=torch.long),
            S_out=torch.zeros(0, dtype=torch.long),
        ),
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        near_batch_offsets=torch.tensor([3]),
        far_batch_offsets=torch.tensor([0]),
    )
    assert tuple(event.event_bg.shape) == (3, 8)
    assert event.event_distant is None
    assert event.event_rigid is None
    loss = event.event_bg.pow(2).mean()
    loss.backward()
    assert parent_context.grad is not None
    assert torch.isfinite(parent_context.grad).all()


def _stage34_params(n: int, *, requires_grad: bool) -> dict[str, torch.Tensor]:
    quats = torch.randn(n, 4, requires_grad=requires_grad)
    return {
        "means": torch.rand(n, 3, requires_grad=requires_grad),
        "quats": quats,
        "scales_log": torch.full((n, 3), -2.0, requires_grad=requires_grad),
        "opacity_logit": torch.randn(n, 1, requires_grad=requires_grad),
        "sh_dc": torch.randn(n, 3, requires_grad=requires_grad),
        "sh_rest": torch.randn(n, 2, 3, requires_grad=requires_grad),
    }


def test_stage34_geometry_residual_is_8d_observes_uniform_scale_and_ignores_quat_sh() -> None:
    torch.manual_seed(34)
    n = 5
    params = _stage34_params(n, requires_grad=True)
    branch = torch.tensor([0, 0, 1, 1, 2])
    adapter = Stage34ParentGeometryResidualAdapter()
    geometry = adapter.geometry_vector(
        params,
        geometry_branch_id=branch,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
    )
    shifted_params = dict(params)
    shifted_params["scales_log"] = params["scales_log"] + 0.2
    shifted = adapter.geometry_vector(
        shifted_params,
        geometry_branch_id=branch,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
    )
    assert tuple(geometry.shape) == (n, 8)
    assert adapter.raw_geometry_dim == 8
    assert torch.all(shifted[:, 3] > geometry[:, 3])
    torch.testing.assert_close(shifted[:, 4:7], geometry[:, 4:7])

    grads = torch.autograd.grad(
        geometry.sum(),
        (
            params["means"],
            params["scales_log"],
            params["opacity_logit"],
            params["quats"],
            params["sh_dc"],
            params["sh_rest"],
        ),
        allow_unused=True,
    )
    for grad in grads[:3]:
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum().item()) > 0.0
    assert grads[3:] == (None, None, None)


def test_stage34_geometry_residual_zero_init_and_legacy_codec_parity() -> None:
    torch.manual_seed(35)
    n = 4
    params = _stage34_params(n, requires_grad=False)
    support = torch.rand(n)
    valid = torch.ones(n, dtype=torch.bool)
    branch = torch.tensor([0, 0, 1, 1])
    geometry_branch = torch.tensor([0, 0, 2, 2])
    legacy = Stage6ParentParamSupportCodec(output_dim=24, detach_params=True, detach_support=True)
    backbone = ParentSpatialBackbone(
        context_dim=4,
        event_dim=8,
        token_dim=8,
        param_codec_mode="legacy17d_plus_geometry8d_residual",
        param_codec_detach_params=False,
        near_heads=2,
        use_xcpe=False,
    )
    backbone.param_support_codec.load_state_dict(legacy.state_dict())
    legacy_out = legacy(
        params_for_embed=params,
        support=support,
        valid_mask=valid,
        branch_id=branch,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
    )
    residual = backbone.geometry_residual_adapter(
        params_for_embed=params,
        geometry_branch_id=geometry_branch,
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
        alpha=0.0,
    )
    assert backbone.geometry_residual_adapter.is_zero_initialized()
    assert torch.count_nonzero(backbone.geometry_residual_adapter.output_proj.weight) == 0
    assert torch.count_nonzero(backbone.geometry_residual_adapter.output_proj.bias) == 0
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy_out + residual, legacy_out, rtol=0.0, atol=0.0)

    # Alpha zero detaches the input Jacobian but must still train the adapter.
    adapter_grads = torch.autograd.grad(residual.sum(), backbone.geometry_residual_adapter.parameters())
    assert float(adapter_grads[-2].abs().sum().item()) > 0.0
    assert float(adapter_grads[-1].abs().sum().item()) > 0.0


def test_stage34_geometry_residual_alpha_preserves_forward_and_scales_jacobian() -> None:
    torch.manual_seed(36)
    adapter = Stage34ParentGeometryResidualAdapter()
    with torch.no_grad():
        adapter.output_proj.weight.normal_(mean=0.0, std=0.1)
        adapter.output_proj.bias.normal_(mean=0.0, std=0.1)
    base = _stage34_params(3, requires_grad=False)
    branch = torch.tensor([0, 1, 2])
    upstream = torch.randn(3, 24)

    outputs: list[torch.Tensor] = []
    geometry_grads: list[tuple[torch.Tensor, ...]] = []
    for alpha in (0.0, 0.25, 1.0):
        params = {
            name: value.detach().clone().requires_grad_(
                name in {"means", "scales_log", "opacity_logit"}
            )
            for name, value in base.items()
        }
        out = adapter(
            params_for_embed=params,
            geometry_branch_id=branch,
            aabb_min=torch.zeros(3),
            aabb_max=torch.ones(3),
            alpha=alpha,
        )
        outputs.append(out.detach())
        geometry_grads.append(
            torch.autograd.grad(
                (out * upstream).sum(),
                (params["means"], params["scales_log"], params["opacity_logit"]),
            )
        )

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0.0, atol=0.0)
    for grad_zero, grad_quarter, grad_full in zip(*geometry_grads):
        torch.testing.assert_close(grad_zero, torch.zeros_like(grad_zero), rtol=0.0, atol=0.0)
        torch.testing.assert_close(grad_quarter, grad_full * 0.25, rtol=2.0e-5, atol=2.0e-6)
        assert float(grad_full.abs().sum().item()) > 0.0


def test_stage34_parent_spatial_detaches_coords_and_all_support_at_call_boundary() -> None:
    torch.manual_seed(35)
    backbone = ParentSpatialBackbone(
        context_dim=4,
        event_dim=8,
        token_dim=8,
        param_codec_mode="legacy17d_plus_geometry8d_residual",
        param_codec_detach_params=False,
        param_codec_detach_support=True,
        ptv3_detach_coords=True,
        near_depth=1,
        near_heads=2,
        near_patch_size=4,
        use_xcpe=False,
    )
    assert backbone.geometry_residual_adapter is not None
    with torch.no_grad():
        backbone.geometry_residual_adapter.output_proj.weight.normal_(mean=0.0, std=0.1)

    class _CapturePTv3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coords_requires_grad = None

        def forward(self, x, *, coords, **_kwargs):
            self.coords_requires_grad = bool(coords.requires_grad)
            return x, {}, {}

    capture = _CapturePTv3()
    backbone.near_ptv3 = capture
    context = torch.randn(3, 4, requires_grad=True)
    coords = torch.rand(3, 3, requires_grad=True)
    support = torch.ones(3, requires_grad=True)
    params = {
        "means": torch.rand(3, 3, requires_grad=True),
        "quats": torch.randn(3, 4, requires_grad=True),
        "scales_log": torch.randn(3, 3, requires_grad=True),
        "opacity_logit": torch.randn(3, 1, requires_grad=True),
        "sh_dc": torch.randn(3, 3, requires_grad=True),
        "sh_rest": torch.randn(3, 1, 3, requires_grad=True),
    }
    near = ParentStructInput(
        parent_context=context,
        support=support,
        valid=torch.ones(3, dtype=torch.bool),
        coords=coords,
        branch_id=torch.zeros(3, dtype=torch.long),
        params_for_embed=params,
        split_0=3,
        split_1=0,
        geometry_branch_id=torch.zeros(3, dtype=torch.long),
        geometry_alpha=1.0,
    )
    far = empty_parent_struct_input(ref=context, context_dim=4, sh_rest_bases=1, path="far")
    event, _ = backbone(
        near_in=near,
        far_in=far,
        route=SimpleNamespace(
            S=torch.zeros(0, dtype=torch.long),
            S_in=torch.zeros(0, dtype=torch.long),
            S_out=torch.zeros(0, dtype=torch.long),
        ),
        aabb_min=torch.zeros(3),
        aabb_max=torch.ones(3),
    )
    assert capture.coords_requires_grad is False
    assert event.aux["feedback/ptv3_coords/boundary_assertion_passed"] == 1.0
    assert event.support_bg.requires_grad is False
    grads = torch.autograd.grad(
        event.event_bg.square().sum(),
        (
            context,
            params["means"],
            params["quats"],
            params["scales_log"],
            params["opacity_logit"],
            params["sh_dc"],
            params["sh_rest"],
            coords,
            support,
        ),
        allow_unused=True,
    )
    for grad in (grads[0], grads[1], grads[3], grads[4]):
        assert grad is not None
        assert torch.isfinite(grad).all()
    assert grads[2] is None
    assert grads[5] is None
    assert grads[6] is None
    assert grads[7] is None
    assert grads[8] is None


def test_stage34_geometry_residual_adapter_is_in_parent_token_optimizer_group() -> None:
    class _Updater(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.vsm_ctx_adapter = torch.nn.Linear(2, 2)

    class _Runtime(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage6_posterior_updater = _Updater()
            self.stage6_measurement_trainable_param_names = set()

    class _Temporal(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Linear(2, 2)
            self.adapters = torch.nn.ModuleDict({"bg": torch.nn.Linear(2, 2)})

    class _Model(torch.nn.Module):
        is_stage2_1_parent_temporal = True

        def __init__(self) -> None:
            super().__init__()
            self.parent_spatial_backbone = ParentSpatialBackbone(
                context_dim=4,
                event_dim=8,
                token_dim=8,
                param_codec_mode="legacy17d_plus_geometry8d_residual",
                param_codec_detach_params=False,
                near_depth=1,
                near_heads=2,
                near_patch_size=4,
                use_xcpe=False,
            )
            self.parent_temporal_mamba = _Temporal()
            self.phase_a_runtime = _Runtime()

    trainer = IForwardTrainer.__new__(IForwardTrainer)
    torch.nn.Module.__init__(trainer)
    trainer.model = _Model()
    trainer.config = {}
    groups = trainer._group_param_lists()
    names = {name for name, _ in groups["parent_token_builder"]}
    adapter_names = {
        f"parent_spatial_backbone.geometry_residual_adapter.{name}"
        for name, _ in trainer.model.parent_spatial_backbone.geometry_residual_adapter.named_parameters()
    }
    assert adapter_names
    assert adapter_names <= names
