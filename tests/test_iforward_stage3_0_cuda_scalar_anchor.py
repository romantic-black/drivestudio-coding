from __future__ import annotations

import math
from typing import Dict, Tuple

import pytest
import torch

from models.iforward.stage3_0.scalar_anchor import (
    build_cuda_scalar_anchor_stats,
    build_projected_meta_anchor_stats,
    cuda_scalar_anchor_available,
)


ALPHA_THRESHOLD = 1.0 / 255.0


def _cuda_anchor_available() -> bool:
    return torch.cuda.is_available() and cuda_scalar_anchor_available()


def _small_meta(device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    means2d = torch.tensor(
        [[0.45, 0.55], [1.35, 0.65], [0.75, 1.35]],
        device=device,
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [[0.20, 0.00, 0.25], [0.35, 0.02, 0.30], [0.25, -0.03, 0.45]],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.40, 0.35, 0.55], device=device, dtype=torch.float32)
    depths = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=torch.float32)
    radii = torch.tensor([[1, 2], [3, 1], [2, 2]], device=device, dtype=torch.int32)
    flatten_ids = torch.tensor([0, 1, 2, 0, 1, 2], device=device, dtype=torch.int32)
    isect_offsets = torch.tensor([[[0]], [[3]]], device=device, dtype=torch.int32)
    packed_ids = torch.tensor([0, 1, 2], device=device, dtype=torch.int64)
    meta = {
        "means2d": means2d,
        "conics": conics,
        "opacities": opacities,
        "depths": depths,
        "radii": radii,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "packed_global_gaussian_ids": packed_ids,
        "tile_size": 2,
    }
    child_to_parent = torch.tensor([0, 0, 1], device=device, dtype=torch.int64)
    pair_mask = torch.ones((2, 2, 2), device=device, dtype=torch.bool)
    pair_mask[1, 1, 1] = False
    return meta, child_to_parent, pair_mask


def _reference_raw(
    *,
    meta: Dict[str, torch.Tensor],
    child_to_parent: torch.Tensor,
    pair_mask: torch.Tensor | None,
    num_children: int,
    num_parents: int,
    image_height: int,
    image_width: int,
    weight_threshold: float,
) -> Tuple[torch.Tensor, ...]:
    means2d = meta["means2d"].detach().cpu()
    conics = meta["conics"].detach().cpu()
    opacities = meta["opacities"].detach().cpu()
    depths = meta["depths"].detach().cpu()
    radii = meta["radii"].detach().cpu().reshape(means2d.shape[0], -1).float().max(dim=1).values
    flatten_ids = meta["flatten_ids"].detach().cpu()
    isect_offsets = meta["isect_offsets"].detach().cpu()
    packed_ids = meta["packed_global_gaussian_ids"].detach().cpu()
    ctp = child_to_parent.detach().cpu()
    mask = pair_mask.detach().cpu() if pair_mask is not None else None
    num_views = int(isect_offsets.shape[0])
    child_support = torch.zeros((num_children, num_views), dtype=torch.float64)
    child_uv_sum = torch.zeros((num_children, num_views, 2), dtype=torch.float64)
    child_depth_sum = torch.zeros((num_children, num_views), dtype=torch.float64)
    child_radius_sum = torch.zeros((num_children, num_views), dtype=torch.float64)
    child_conic_sum = torch.zeros((num_children, num_views, 3), dtype=torch.float64)
    parent_support = torch.zeros((num_parents, num_views), dtype=torch.float64)
    parent_uv_sum = torch.zeros((num_parents, num_views, 2), dtype=torch.float64)
    parent_depth_sum = torch.zeros((num_parents, num_views), dtype=torch.float64)
    parent_radius_sum = torch.zeros((num_parents, num_views), dtype=torch.float64)
    parent_conic_sum = torch.zeros((num_parents, num_views, 3), dtype=torch.float64)
    pair_total = 0
    pair_kept = 0
    for view in range(num_views):
        start = int(isect_offsets[view, 0, 0].item())
        end = int(isect_offsets[view + 1, 0, 0].item()) if view + 1 < num_views else int(flatten_ids.numel())
        for y in range(image_height):
            for x in range(image_width):
                if mask is not None and not bool(mask[view, y, x].item()):
                    continue
                trans = 1.0
                for idx in range(start, end):
                    g_local = int(flatten_ids[idx].item())
                    xy = means2d[g_local]
                    conic = conics[g_local]
                    dx = float(x) + 0.5 - float(xy[0].item())
                    dy = float(y) + 0.5 - float(xy[1].item())
                    sigma = 0.5 * (float(conic[0]) * dx * dx + float(conic[2]) * dy * dy) + float(conic[1]) * dx * dy
                    alpha = min(0.999, float(opacities[g_local]) * math.exp(-sigma))
                    if sigma < 0.0 or alpha < ALPHA_THRESHOLD:
                        continue
                    vis = alpha * trans
                    g_global = int(packed_ids[g_local].item())
                    if 0 <= g_global < num_children:
                        pair_total += 1
                    if 0 <= g_global < num_children and vis >= weight_threshold:
                        child_support[g_global, view] += vis
                        child_uv_sum[g_global, view, 0] += vis * float(x)
                        child_uv_sum[g_global, view, 1] += vis * float(y)
                        child_depth_sum[g_global, view] += vis * float(depths[g_local])
                        child_radius_sum[g_global, view] += vis * float(radii[g_local])
                        child_conic_sum[g_global, view] += vis * conic.double()
                        p_global = int(ctp[g_global].item())
                        if 0 <= p_global < num_parents:
                            parent_support[p_global, view] += vis
                            parent_uv_sum[p_global, view, 0] += vis * float(x)
                            parent_uv_sum[p_global, view, 1] += vis * float(y)
                            parent_depth_sum[p_global, view] += vis * float(depths[g_local])
                            parent_radius_sum[p_global, view] += vis * float(radii[g_local])
                            parent_conic_sum[p_global, view] += vis * conic.double()
                        pair_kept += 1
                    trans *= 1.0 - alpha
                    if trans <= 1.0e-4:
                        break
    return (
        child_support.float(),
        child_uv_sum.float(),
        child_depth_sum.float(),
        child_radius_sum.float(),
        child_conic_sum.float(),
        parent_support.float(),
        parent_uv_sum.float(),
        parent_depth_sum.float(),
        parent_radius_sum.float(),
        parent_conic_sum.float(),
        torch.tensor([pair_total], dtype=torch.int64),
        torch.tensor([pair_kept], dtype=torch.int64),
    )


def _parent_from_child_raw(
    *,
    child_to_parent: torch.Tensor,
    num_parents: int,
    child_support: torch.Tensor,
    child_uv_sum: torch.Tensor,
    child_depth_sum: torch.Tensor,
    child_radius_sum: torch.Tensor,
    child_conic_sum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctp = child_to_parent.detach().cpu().to(dtype=torch.long)
    parent_support = child_support.detach().cpu().new_zeros((int(num_parents), int(child_support.shape[1])))
    parent_uv_sum = child_uv_sum.detach().cpu().new_zeros((int(num_parents), int(child_uv_sum.shape[1]), 2))
    parent_depth_sum = child_depth_sum.detach().cpu().new_zeros((int(num_parents), int(child_depth_sum.shape[1])))
    parent_radius_sum = child_radius_sum.detach().cpu().new_zeros((int(num_parents), int(child_radius_sum.shape[1])))
    parent_conic_sum = child_conic_sum.detach().cpu().new_zeros((int(num_parents), int(child_conic_sum.shape[1]), 3))
    parent_support.index_add_(0, ctp, child_support.detach().cpu())
    parent_uv_sum.index_add_(0, ctp, child_uv_sum.detach().cpu())
    parent_depth_sum.index_add_(0, ctp, child_depth_sum.detach().cpu())
    parent_radius_sum.index_add_(0, ctp, child_radius_sum.detach().cpu())
    parent_conic_sum.index_add_(0, ctp, child_conic_sum.detach().cpu())
    return parent_support, parent_uv_sum, parent_depth_sum, parent_radius_sum, parent_conic_sum


def test_projected_meta_fallback_still_works_on_cpu() -> None:
    means2d = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    meta = {
        "means2d": means2d,
        "gaussian_ids": torch.tensor([0], dtype=torch.long),
        "camera_ids": torch.tensor([0], dtype=torch.long),
        "opacities": torch.tensor([0.25], dtype=torch.float32),
        "depths": torch.tensor([2.0], dtype=torch.float32),
        "radii": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "conics": torch.ones((1, 3), dtype=torch.float32),
    }
    anchor = build_projected_meta_anchor_stats(
        meta=meta,
        child_to_parent=torch.tensor([0], dtype=torch.long),
        num_children=1,
        num_parents=1,
        num_views=1,
        image_height=4,
        image_width=4,
    )
    assert torch.allclose(anchor.child_support, torch.tensor([[0.25]]))


def test_cuda_scalar_anchor_failfast_on_cpu() -> None:
    meta, ctp, mask = _small_meta(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="cuda_scalar_anchor"):
        build_cuda_scalar_anchor_stats(
            meta=meta,
            child_to_parent=ctp,
            num_children=3,
            num_parents=2,
            num_views=2,
            image_height=2,
            image_width=2,
            source_pair_valid_mask=mask,
        )


def test_low_level_wrapper_rejects_non_float32_before_cuda_call() -> None:
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    meta, ctp, _mask = _small_meta(torch.device("cpu"))
    with pytest.raises(ValueError, match="means2d must be float32"):
        rasterize_scalar_anchor_multi_camera_in_range(
            0,
            int(1e9),
            meta["means2d"].double(),
            meta["conics"],
            meta["opacities"],
            meta["depths"],
            torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32),
            2,
            2,
            2,
            meta["isect_offsets"],
            meta["flatten_ids"],
            meta["packed_global_gaussian_ids"],
            ctp,
            3,
            2,
        )


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_raw_matches_reference_and_backproject_support() -> None:
    from gsplat.cuda._wrapper import rasterize_and_backproject_multi_camera_in_range
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    radii = meta["radii"].reshape(3, -1).float().max(dim=1).values
    raw = rasterize_scalar_anchor_multi_camera_in_range(
        0,
        int(1e9),
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["depths"],
        radii,
        2,
        2,
        2,
        meta["isect_offsets"],
        meta["flatten_ids"],
        meta["packed_global_gaussian_ids"],
        ctp,
        3,
        2,
        mask,
        0.0,
    )
    ref = _reference_raw(
        meta=meta,
        child_to_parent=ctp,
        pair_mask=mask,
        num_children=3,
        num_parents=2,
        image_height=2,
        image_width=2,
        weight_threshold=0.0,
    )
    for got, expected in zip(raw, ref):
        assert torch.allclose(got.detach().cpu().float(), expected.float(), atol=2.0e-5, rtol=2.0e-5)

    feat2d = torch.ones((2, 2, 2, 1), device=device, dtype=torch.float32)
    _feat_sum, weight_sum_feature, _weight_sum_support, _pairs_total, _pairs_kept = (
        rasterize_and_backproject_multi_camera_in_range(
            range_start=0,
            range_end=int(1e9),
            means2d=meta["means2d"],
            conics=meta["conics"],
            opacities=meta["opacities"],
            image_width=2,
            image_height=2,
            tile_size=2,
            isect_offsets=meta["isect_offsets"],
            flatten_ids=meta["flatten_ids"],
            packed_global_gaussian_ids=meta["packed_global_gaussian_ids"],
            feat2d=feat2d,
            num_gaussians=3,
            pair_valid_mask=mask,
            weight_threshold=0.0,
            return_support=True,
        )
    )
    assert torch.allclose(raw[0].sum(dim=1), weight_sum_feature, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_parent_raw_sums_match_child_python_aggregate() -> None:
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    radii = meta["radii"].reshape(3, -1).float().max(dim=1).values
    raw = rasterize_scalar_anchor_multi_camera_in_range(
        0,
        int(1e9),
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["depths"],
        radii,
        2,
        2,
        2,
        meta["isect_offsets"],
        meta["flatten_ids"],
        meta["packed_global_gaussian_ids"],
        ctp,
        3,
        2,
        mask,
        0.0,
    )
    expected_parent = _parent_from_child_raw(
        child_to_parent=ctp,
        num_parents=2,
        child_support=raw[0],
        child_uv_sum=raw[1],
        child_depth_sum=raw[2],
        child_radius_sum=raw[3],
        child_conic_sum=raw[4],
    )
    for got, expected in zip(raw[5:10], expected_parent):
        assert torch.allclose(got.detach().cpu(), expected, atol=2.0e-5, rtol=2.0e-5)


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_fast_mode_and_parent_aggregate_match_full_child_raw() -> None:
    from gsplat.cuda._wrapper import aggregate_scalar_anchor_parent_from_child
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    radii = meta["radii"].reshape(3, -1).float().max(dim=1).values
    raw_full = rasterize_scalar_anchor_multi_camera_in_range(
        0,
        int(1e9),
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["depths"],
        radii,
        2,
        2,
        2,
        meta["isect_offsets"],
        meta["flatten_ids"],
        meta["packed_global_gaussian_ids"],
        ctp,
        3,
        2,
        mask,
        0.0,
        anchor_mode="full",
        count_pairs=False,
    )
    raw_fast = rasterize_scalar_anchor_multi_camera_in_range(
        0,
        int(1e9),
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["depths"],
        radii,
        2,
        2,
        2,
        meta["isect_offsets"],
        meta["flatten_ids"],
        meta["packed_global_gaussian_ids"],
        ctp,
        3,
        2,
        mask,
        0.0,
        anchor_mode="fast_uv_support",
        count_pairs=False,
    )
    assert torch.allclose(raw_fast[0], raw_full[0], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(raw_fast[1], raw_full[1], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(raw_fast[5], raw_full[5], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(raw_fast[6], raw_full[6], atol=2.0e-5, rtol=2.0e-5)
    assert all(int(x.numel()) == 0 for x in raw_fast[2:5])
    assert all(int(x.numel()) == 0 for x in raw_fast[7:10])
    assert int(raw_fast[10].item()) == 0
    assert int(raw_fast[11].item()) == 0

    parent_full = aggregate_scalar_anchor_parent_from_child(
        raw_full[0],
        raw_full[1],
        raw_full[2],
        raw_full[3],
        raw_full[4],
        ctp,
        2,
        fast_uv_support=False,
    )
    for got, expected in zip(parent_full, raw_full[5:10]):
        assert torch.allclose(got, expected, atol=2.0e-5, rtol=2.0e-5)

    parent_fast = aggregate_scalar_anchor_parent_from_child(
        raw_fast[0],
        raw_fast[1],
        raw_fast[2],
        raw_fast[3],
        raw_fast[4],
        ctp,
        2,
        fast_uv_support=True,
    )
    assert torch.allclose(parent_fast[0], raw_fast[5], atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(parent_fast[1], raw_fast[6], atol=2.0e-5, rtol=2.0e-5)
    assert all(int(x.numel()) == 0 for x in parent_fast[2:])


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_normalizes_parent_aggregation_and_detaches() -> None:
    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    anchor, aux = build_cuda_scalar_anchor_stats(
        meta=meta,
        child_to_parent=ctp,
        num_children=3,
        num_parents=2,
        num_views=2,
        image_height=2,
        image_width=2,
        source_pair_valid_mask=mask,
        child_support_threshold=1.0e-6,
        parent_support_threshold=1.0e-6,
        return_aux=True,
    )
    ref = _reference_raw(
        meta=meta,
        child_to_parent=ctp,
        pair_mask=mask,
        num_children=3,
        num_parents=2,
        image_height=2,
        image_width=2,
        weight_threshold=0.0,
    )
    child_support, child_uv_sum, child_depth_sum, child_radius_sum, child_conic_sum = ref[:5]
    parent_support, parent_uv_sum, parent_depth_sum, parent_radius_sum, parent_conic_sum = ref[5:10]
    assert torch.allclose(anchor.child_support.cpu(), child_support, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(anchor.child_uv.cpu(), child_uv_sum / child_support.clamp_min(1.0e-8)[..., None], atol=2.0e-5)
    assert torch.allclose(anchor.child_depth.cpu(), child_depth_sum / child_support.clamp_min(1.0e-8), atol=2.0e-5)
    assert torch.allclose(anchor.child_radius.cpu(), child_radius_sum / child_support.clamp_min(1.0e-8), atol=2.0e-5)
    assert torch.allclose(anchor.child_conic.cpu(), child_conic_sum / child_support.clamp_min(1.0e-8)[..., None], atol=2.0e-5)
    assert torch.allclose(anchor.parent_support.cpu(), parent_support, atol=2.0e-5, rtol=2.0e-5)
    assert torch.allclose(anchor.parent_uv.cpu(), parent_uv_sum / parent_support.clamp_min(1.0e-8)[..., None], atol=2.0e-5)
    assert torch.allclose(anchor.parent_depth.cpu(), parent_depth_sum / parent_support.clamp_min(1.0e-8), atol=2.0e-5)
    assert torch.allclose(anchor.parent_radius.cpu(), parent_radius_sum / parent_support.clamp_min(1.0e-8), atol=2.0e-5)
    assert torch.allclose(anchor.parent_conic_approx.cpu(), parent_conic_sum / parent_support.clamp_min(1.0e-8)[..., None], atol=2.0e-5)
    assert aux["iforward/stage3/anchor_backend_id"] == 1.0
    assert aux["iforward/stage3/anchor_parent_aggregate_backend_id"] == 1.0
    assert aux["iforward/stage3/anchor_parent_aggregate_cuda_enabled"] == 1.0
    assert aux["iforward/stage3/anchor_heavy_aux_enabled"] == 0.0
    assert aux["iforward/stage3/anchor_pair_count_enabled"] == 0.0
    assert aux["iforward/stage3/anchor_pair_count_threshold"] == 0.0
    assert "iforward/stage3/anchor_child_support_mean" not in aux
    assert not anchor.child_support.requires_grad
    assert not anchor.child_uv.requires_grad


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_fast_mode_normalization_fills_geometry_defaults() -> None:
    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    anchor, aux = build_cuda_scalar_anchor_stats(
        meta=meta,
        child_to_parent=ctp,
        num_children=3,
        num_parents=2,
        num_views=2,
        image_height=2,
        image_width=2,
        source_pair_valid_mask=mask,
        child_support_threshold=1.0e-6,
        parent_support_threshold=1.0e-6,
        anchor_mode="fast_uv_support",
        count_pairs=False,
        return_aux=True,
    )
    assert aux["iforward/stage3/anchor_mode_id"] == 1.0
    assert aux["iforward/stage3/anchor_fast_uv_support_enabled"] == 1.0
    assert torch.allclose(anchor.child_depth, torch.zeros_like(anchor.child_depth))
    assert torch.allclose(anchor.parent_depth, torch.zeros_like(anchor.parent_depth))
    assert torch.allclose(anchor.child_radius, torch.ones_like(anchor.child_radius))
    assert torch.allclose(anchor.parent_radius, torch.ones_like(anchor.parent_radius))
    assert torch.allclose(anchor.child_conic, torch.zeros_like(anchor.child_conic))
    assert torch.allclose(anchor.parent_conic_approx, torch.zeros_like(anchor.parent_conic_approx))


@pytest.mark.skipif(not _cuda_anchor_available(), reason="Stage3 CUDA scalar anchor op unavailable")
def test_cuda_scalar_anchor_child_only_skips_parent_aggregate() -> None:
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    device = torch.device("cuda")
    meta, ctp, mask = _small_meta(device)
    radii = meta["radii"].reshape(3, -1).float().max(dim=1).values
    raw = rasterize_scalar_anchor_multi_camera_in_range(
        0,
        int(1e9),
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["depths"],
        radii,
        2,
        2,
        2,
        meta["isect_offsets"],
        meta["flatten_ids"],
        meta["packed_global_gaussian_ids"],
        ctp,
        3,
        2,
        mask,
        0.0,
        anchor_mode="fast_uv_support",
        count_pairs=False,
        child_only=True,
    )
    assert int(raw[0].numel()) > 0
    assert int(raw[1].numel()) > 0
    assert all(int(x.numel()) == 0 for x in raw[2:10])

    anchor, aux = build_cuda_scalar_anchor_stats(
        meta=meta,
        child_to_parent=ctp,
        num_children=3,
        num_parents=2,
        num_views=2,
        image_height=2,
        image_width=2,
        source_pair_valid_mask=mask,
        child_support_threshold=1.0e-6,
        parent_support_threshold=1.0e-6,
        anchor_mode="fast_uv_support",
        count_pairs=False,
        child_only=True,
        return_aux=True,
    )
    assert aux["iforward/stage3/anchor_child_only_enabled"] == 1.0
    assert aux["iforward/stage3/anchor_parent_aggregate_backend_id"] == 0.0
    assert aux["iforward/stage3/anchor_parent_aggregate_cuda_enabled"] == 0.0
    assert tuple(anchor.parent_support.shape) == (2, 2)
    assert torch.allclose(anchor.parent_support, torch.zeros_like(anchor.parent_support))
    assert torch.allclose(anchor.parent_radius, torch.ones_like(anchor.parent_radius))
