from __future__ import annotations

from types import SimpleNamespace

import torch

from models.iforward.parent_ptv3 import ParentPTv3Encoder
from models.iforward.parent_serialization import build_parent_serialized_layout
from models.iforward.parent_spatial_backbone import ParentSpatialBackbone, ParentStructInput, empty_parent_struct_input


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
    assert cache2["z"] is cache1["z"]
    assert cache3["z"] is not cache1["z"]
    loss = out1.pow(2).mean() + out2.abs().mean() + out3.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


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
