from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch


@dataclass
class SparseAnchorStats:
    child_uv: torch.Tensor
    child_support: torch.Tensor
    child_valid: torch.Tensor
    child_depth: torch.Tensor
    child_radius: torch.Tensor
    child_conic: torch.Tensor
    child_ray: torch.Tensor
    parent_uv: torch.Tensor
    parent_support: torch.Tensor
    parent_valid: torch.Tensor
    parent_depth: torch.Tensor
    parent_radius: torch.Tensor
    parent_conic_approx: torch.Tensor
    child_support_total: torch.Tensor
    parent_support_total: torch.Tensor

    def detach(self) -> "SparseAnchorStats":
        return SparseAnchorStats(
            child_uv=self.child_uv.detach(),
            child_support=self.child_support.detach(),
            child_valid=self.child_valid.detach(),
            child_depth=self.child_depth.detach(),
            child_radius=self.child_radius.detach(),
            child_conic=self.child_conic.detach(),
            child_ray=self.child_ray.detach(),
            parent_uv=self.parent_uv.detach(),
            parent_support=self.parent_support.detach(),
            parent_valid=self.parent_valid.detach(),
            parent_depth=self.parent_depth.detach(),
            parent_radius=self.parent_radius.detach(),
            parent_conic_approx=self.parent_conic_approx.detach(),
            child_support_total=self.child_support_total.detach(),
            parent_support_total=self.parent_support_total.detach(),
        )


def _meta_tensor(meta: Dict[str, Any], *names: str) -> Optional[torch.Tensor]:
    for name in names:
        value = meta.get(name)
        if torch.is_tensor(value):
            return value
    return None


def _cuda_fp32_context() -> Any:
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def cuda_scalar_anchor_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from gsplat.cuda._backend import _C  # type: ignore
    except Exception:
        return False
    return hasattr(_C, "rasterize_scalar_anchor_3dgs_multi_camera")


def _row_scalar(value: torch.Tensor, *, rows: int, name: str, reduce: str = "first") -> torch.Tensor:
    if int(rows) == 0:
        return value.reshape(0)
    x = value.reshape(int(rows), -1)
    if int(x.shape[1]) == 1:
        return x[:, 0]
    if reduce == "max":
        return x.max(dim=1).values
    if reduce == "mean":
        return x.mean(dim=1)
    if reduce == "first":
        return x[:, 0]
    raise ValueError(f"unsupported scalar reduction for {name}: {reduce!r}")


def _scatter_weighted_view_stats(
    *,
    row_id: torch.Tensor,
    view_id: torch.Tensor,
    support: torch.Tensor,
    uv: torch.Tensor,
    depth: torch.Tensor,
    radius: torch.Tensor,
    conic: torch.Tensor,
    num_rows: int,
    num_views: int,
) -> Dict[str, torch.Tensor]:
    device = support.device
    dtype = support.dtype
    flat = row_id.long() * int(num_views) + view_id.long()
    total = int(num_rows) * int(num_views)
    support_view = support.new_zeros((total,))
    uv_sum = support.new_zeros((total, 2))
    depth_sum = support.new_zeros((total,))
    radius_sum = support.new_zeros((total,))
    conic_sum = support.new_zeros((total, 3))
    if int(flat.numel()) > 0:
        support_view.index_add_(0, flat, support)
        uv_sum.index_add_(0, flat, uv.to(device=device, dtype=dtype) * support[:, None])
        depth_sum.index_add_(0, flat, depth.to(device=device, dtype=dtype).reshape(-1) * support)
        radius_sum.index_add_(0, flat, radius.to(device=device, dtype=dtype).reshape(-1) * support)
        conic_sum.index_add_(0, flat, conic.to(device=device, dtype=dtype) * support[:, None])
    denom = support_view.clamp_min(1.0e-8)
    return {
        "support": support_view.reshape(int(num_rows), int(num_views)),
        "uv": (uv_sum / denom[:, None]).reshape(int(num_rows), int(num_views), 2),
        "depth": (depth_sum / denom).reshape(int(num_rows), int(num_views)),
        "radius": (radius_sum / denom).reshape(int(num_rows), int(num_views)),
        "conic": (conic_sum / denom[:, None]).reshape(int(num_rows), int(num_views), 3),
    }


def _apply_pair_valid_mask(
    *,
    valid: torch.Tensor,
    uv: torch.Tensor,
    view_id: torch.Tensor,
    source_pair_valid_mask: Optional[torch.Tensor],
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    if source_pair_valid_mask is None or int(valid.numel()) == 0:
        return valid
    mask = source_pair_valid_mask.to(device=valid.device)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.dim() != 3:
        raise ValueError(f"source_pair_valid_mask must be [V,H,W], got {tuple(mask.shape)}")
    if int(mask.shape[1]) != int(image_height) or int(mask.shape[2]) != int(image_width):
        raise ValueError(
            "source_pair_valid_mask spatial mismatch: "
            f"expected {(int(image_height), int(image_width))}, got {tuple(mask.shape[1:])}"
        )
    px = uv[:, 0].round().long().clamp(0, int(image_width) - 1)
    py = uv[:, 1].round().long().clamp(0, int(image_height) - 1)
    return valid & mask[view_id.long(), py, px]


def _require_meta_tensor(meta: Dict[str, Any], name: str) -> torch.Tensor:
    value = _meta_tensor(meta, name)
    if value is None:
        raise ValueError(f"cuda_scalar_anchor requires meta[{name!r}]")
    return value


def _packed_radius_scalar(radii: torch.Tensor, *, rows: int, ref: torch.Tensor) -> torch.Tensor:
    if int(rows) == 0:
        return ref.new_zeros((0,))
    x = radii.to(device=ref.device, dtype=ref.dtype).reshape(int(rows), -1)
    if int(x.shape[1]) == 1:
        return x[:, 0].clamp_min(0.0)
    return x.max(dim=1).values.clamp_min(0.0)


def _view_entropy_mean(support: torch.Tensor) -> float:
    if int(support.numel()) == 0:
        return 0.0
    total = support.sum(dim=1, keepdim=True)
    valid = total[:, 0] > 0.0
    if not bool(valid.any().item()):
        return 0.0
    probs = support[valid] / total[valid].clamp_min(1.0e-8)
    entropy = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1)
    return float(entropy.detach().float().mean().item())


def _uv_oob_ratio(uv: torch.Tensor, support: torch.Tensor, *, image_height: int, image_width: int) -> float:
    if int(uv.numel()) == 0 or int(support.numel()) == 0:
        return 0.0
    valid = support > 0.0
    if not bool(valid.any().item()):
        return 0.0
    oob = (
        (uv[..., 0] < 0.0)
        | (uv[..., 0] >= float(image_width))
        | (uv[..., 1] < 0.0)
        | (uv[..., 1] >= float(image_height))
    )
    return float((oob & valid).detach().float().sum().item() / max(float(valid.detach().float().sum().item()), 1.0))


def _normalize_cuda_anchor_outputs(
    *,
    child_support: torch.Tensor,
    child_uv_sum: torch.Tensor,
    child_depth_sum: torch.Tensor,
    child_radius_sum: torch.Tensor,
    child_conic_sum: torch.Tensor,
    parent_support: torch.Tensor,
    parent_uv_sum: torch.Tensor,
    parent_depth_sum: torch.Tensor,
    parent_radius_sum: torch.Tensor,
    parent_conic_sum: torch.Tensor,
    child_support_threshold: float,
    parent_support_threshold: float,
    num_parents: Optional[int] = None,
    num_views: Optional[int] = None,
) -> SparseAnchorStats:
    child_denom = child_support.clamp_min(1.0e-8)
    parent_denom = parent_support.clamp_min(1.0e-8)
    num_children = int(child_support.shape[0])
    num_views_i = int(num_views) if num_views is not None else (int(child_support.shape[1]) if child_support.dim() == 2 else 0)
    num_parents_i = int(num_parents) if num_parents is not None else int(parent_support.shape[0])
    if int(parent_support.numel()) == 0:
        parent_support = child_support.new_zeros((num_parents_i, num_views_i))
        parent_uv_sum = child_support.new_zeros((num_parents_i, num_views_i, 2))
        parent_denom = parent_support.clamp_min(1.0e-8)
    child_ray = child_support.new_zeros((num_children, num_views_i, 3))
    if int(child_depth_sum.numel()) == 0:
        child_depth = child_support.new_zeros((num_children, num_views_i))
        child_radius = child_support.new_ones((num_children, num_views_i))
        child_conic = child_support.new_zeros((num_children, num_views_i, 3))
    else:
        child_depth = child_depth_sum / child_denom
        child_radius = child_radius_sum / child_denom
        child_conic = child_conic_sum / child_denom[..., None]
    if int(parent_depth_sum.numel()) == 0:
        parent_depth = parent_support.new_zeros((num_parents_i, num_views_i))
        parent_radius = parent_support.new_ones((num_parents_i, num_views_i))
        parent_conic = parent_support.new_zeros((num_parents_i, num_views_i, 3))
    else:
        parent_depth = parent_depth_sum / parent_denom
        parent_radius = parent_radius_sum / parent_denom
        parent_conic = parent_conic_sum / parent_denom[..., None]
    return SparseAnchorStats(
        child_uv=child_uv_sum / child_denom[..., None],
        child_support=child_support,
        child_valid=child_support >= float(child_support_threshold),
        child_depth=child_depth,
        child_radius=child_radius,
        child_conic=child_conic,
        child_ray=child_ray,
        parent_uv=parent_uv_sum / parent_denom[..., None],
        parent_support=parent_support,
        parent_valid=parent_support >= float(parent_support_threshold),
        parent_depth=parent_depth,
        parent_radius=parent_radius,
        parent_conic_approx=parent_conic,
        child_support_total=child_support.sum(dim=1),
        parent_support_total=parent_support.sum(dim=1),
    )


def build_cuda_scalar_anchor_stats(
    *,
    meta: Dict[str, Any],
    child_to_parent: torch.Tensor,
    num_children: int,
    num_parents: int,
    num_views: int,
    image_height: int,
    image_width: int,
    source_pair_valid_mask: Optional[torch.Tensor] = None,
    child_support_threshold: float = 1.0e-4,
    parent_support_threshold: float = 1.0e-4,
    weight_threshold: float = 0.0,
    anchor_mode: str = "full",
    count_pairs: bool = False,
    child_only: bool = False,
    detach_geometry: bool = True,
    emit_heavy_aux: bool = False,
    use_cuda_event_timing: bool = False,
    return_aux: bool = False,
) -> Union[SparseAnchorStats, Tuple[SparseAnchorStats, Dict[str, float]]]:
    """Build Stage3 anchor stats from occlusion-aware gsplat scalar visibility.

    The CUDA op is forward-only by design: anchor geometry and support are
    observation metadata, not trainable feature values.
    """

    if not cuda_scalar_anchor_available():
        raise RuntimeError("Stage3 cuda_scalar_anchor requires CUDA and gsplat rasterize_scalar_anchor_3dgs_multi_camera.")
    anchor_mode = str(anchor_mode).lower()
    if anchor_mode == "auto":
        anchor_mode = "full"
    if anchor_mode not in {"full", "fast_uv_support"}:
        raise ValueError(f"unsupported cuda scalar anchor mode={anchor_mode!r}")
    means2d = _require_meta_tensor(meta, "means2d")
    conics = _require_meta_tensor(meta, "conics")
    opacities = _require_meta_tensor(meta, "opacities")
    depths = _require_meta_tensor(meta, "depths")
    radii_raw = _require_meta_tensor(meta, "radii")
    flatten_ids = _require_meta_tensor(meta, "flatten_ids")
    isect_offsets = _require_meta_tensor(meta, "isect_offsets")
    packed_ids = _require_meta_tensor(meta, "packed_global_gaussian_ids")
    if not means2d.is_cuda:
        raise RuntimeError("Stage3 cuda_scalar_anchor requires CUDA tensors.")
    means2d = means2d.to(dtype=torch.float32)
    conics = conics.to(device=means2d.device, dtype=torch.float32)
    opacities = opacities.to(device=means2d.device, dtype=torch.float32)
    depths = depths.to(device=means2d.device, dtype=torch.float32)
    for name, tensor in {
        "means2d": means2d,
        "conics": conics,
        "opacities": opacities,
        "depths": depths,
    }.items():
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Stage3 cuda_scalar_anchor requires float32 {name}, got {tensor.dtype}.")
    rows = int(means2d.reshape(-1, 2).shape[0])
    means2d = means2d.reshape(rows, 2).contiguous()
    conics = conics.to(device=means2d.device).reshape(rows, 3).contiguous()
    opacities = opacities.to(device=means2d.device).reshape(rows).contiguous()
    depths = depths.to(device=means2d.device).reshape(rows).contiguous()
    radii = _packed_radius_scalar(radii_raw, rows=rows, ref=means2d).contiguous()
    ctp = child_to_parent.to(device=means2d.device, dtype=torch.long).reshape(-1).contiguous()
    if int(ctp.numel()) != int(num_children):
        raise ValueError(f"child_to_parent row mismatch: {int(ctp.numel())} vs {int(num_children)}")
    if int(num_views) != int(isect_offsets.shape[0]):
        raise ValueError(f"num_views/isect_offsets mismatch: {num_views} vs {int(isect_offsets.shape[0])}")
    mask = None
    if source_pair_valid_mask is not None:
        mask = source_pair_valid_mask.to(device=means2d.device)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
    from gsplat.cuda._wrapper import rasterize_scalar_anchor_multi_camera_in_range

    event_start = None
    event_end = None
    if bool(use_cuda_event_timing):
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    t_cuda = time.perf_counter()
    with _cuda_fp32_context():
        raw = rasterize_scalar_anchor_multi_camera_in_range(
            range_start=0,
            range_end=int(1e9),
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            depths=depths,
            radii=radii,
            image_width=int(image_width),
            image_height=int(image_height),
            tile_size=int(meta.get("tile_size", 16)),
            isect_offsets=isect_offsets.to(device=means2d.device, dtype=torch.int32),
            flatten_ids=flatten_ids.to(device=means2d.device, dtype=torch.int32),
            packed_global_gaussian_ids=packed_ids.to(device=means2d.device, dtype=torch.long),
            child_to_parent=ctp,
            num_children=int(num_children),
            num_parents=int(num_parents),
            pair_valid_mask=mask,
            weight_threshold=float(weight_threshold),
            anchor_mode=anchor_mode,
            count_pairs=bool(count_pairs),
            child_only=bool(child_only),
        )
    if event_end is not None:
        event_end.record()
        event_end.synchronize()
    cuda_ms = float((time.perf_counter() - t_cuda) * 1000.0)
    cuda_event_ms = float(event_start.elapsed_time(event_end)) if event_start is not None and event_end is not None else 0.0
    (
        child_support,
        child_uv_sum,
        child_depth_sum,
        child_radius_sum,
        child_conic_sum,
        parent_support,
        parent_uv_sum,
        parent_depth_sum,
        parent_radius_sum,
        parent_conic_sum,
        pair_count_total,
        pair_count_threshold,
    ) = raw
    t_norm = time.perf_counter()
    anchor = _normalize_cuda_anchor_outputs(
        child_support=child_support,
        child_uv_sum=child_uv_sum,
        child_depth_sum=child_depth_sum,
        child_radius_sum=child_radius_sum,
        child_conic_sum=child_conic_sum,
        parent_support=parent_support,
        parent_uv_sum=parent_uv_sum,
        parent_depth_sum=parent_depth_sum,
        parent_radius_sum=parent_radius_sum,
        parent_conic_sum=parent_conic_sum,
        child_support_threshold=float(child_support_threshold),
        parent_support_threshold=float(parent_support_threshold),
        num_parents=int(num_parents),
        num_views=int(num_views),
    )
    norm_ms = float((time.perf_counter() - t_norm) * 1000.0)
    if bool(count_pairs):
        pair_count_total_value = float(pair_count_total.detach().cpu().item())
        pair_count_threshold_value = float(pair_count_threshold.detach().cpu().item())
    else:
        pair_count_total_value = 0.0
        pair_count_threshold_value = 0.0
    aux = {
        "iforward/stage3/anchor_backend_id": 1.0,
        "iforward/stage3/anchor_mode_id": 1.0 if anchor_mode == "fast_uv_support" else 0.0,
        "iforward/stage3/anchor_fast_uv_support_enabled": 1.0 if anchor_mode == "fast_uv_support" else 0.0,
        "iforward/stage3/anchor_child_only_enabled": 1.0 if bool(child_only) else 0.0,
        "iforward/stage3/anchor_parent_aggregate_backend_id": 0.0 if bool(child_only) else 1.0,
        "iforward/stage3/anchor_parent_aggregate_cuda_enabled": 0.0 if bool(child_only) else 1.0,
        "iforward/stage3/anchor_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
        "iforward/stage3/anchor_pair_count_enabled": 1.0 if bool(count_pairs) else 0.0,
        "iforward/stage3/anchor_cuda_ms": float(cuda_ms),
        "iforward/stage3/anchor_cuda_event_ms": float(cuda_event_ms),
        "iforward/stage3/anchor_normalize_ms": float(norm_ms),
        "iforward/stage3/anchor_pair_count_total": pair_count_total_value,
        "iforward/stage3/anchor_pair_count_threshold": pair_count_threshold_value,
    }
    if bool(emit_heavy_aux):
        aux.update(
            {
                "iforward/stage3/anchor_child_support_mean": float(child_support.detach().float().mean().item())
                if int(child_support.numel())
                else 0.0,
                "iforward/stage3/anchor_parent_support_mean": float(parent_support.detach().float().mean().item())
                if int(parent_support.numel())
                else 0.0,
                "iforward/stage3/anchor_child_view_entropy": _view_entropy_mean(child_support),
                "iforward/stage3/anchor_parent_view_entropy": _view_entropy_mean(parent_support),
                "iforward/stage3/anchor_child_uv_oob_ratio": _uv_oob_ratio(
                    anchor.child_uv,
                    child_support,
                    image_height=int(image_height),
                    image_width=int(image_width),
                ),
                "iforward/stage3/anchor_parent_uv_oob_ratio": _uv_oob_ratio(
                    anchor.parent_uv,
                    parent_support,
                    image_height=int(image_height),
                    image_width=int(image_width),
                ),
            }
        )
    anchor = anchor.detach() if bool(detach_geometry) else anchor
    if bool(return_aux):
        return anchor, aux
    return anchor


def build_projected_meta_anchor_stats(
    *,
    meta: Dict[str, Any],
    child_to_parent: torch.Tensor,
    num_children: int,
    num_parents: int,
    num_views: int,
    image_height: int,
    image_width: int,
    source_pair_valid_mask: Optional[torch.Tensor] = None,
    child_support_threshold: float = 1.0e-4,
    parent_support_threshold: float = 1.0e-4,
    detach_geometry: bool = True,
) -> SparseAnchorStats:
    """Build P0 Stage3 anchor stats from gsplat packed projection metadata.

    This intentionally uses projected opacity as a visibility approximation. It
    does not call a feature backproject kernel and therefore does not provide
    true alpha/T occlusion support.
    """

    means2d = _meta_tensor(meta, "means2d")
    if means2d is None:
        raise ValueError("projected_meta anchor requires meta['means2d']")
    device = means2d.device
    dtype = means2d.dtype
    gaussian_ids = _meta_tensor(meta, "packed_global_gaussian_ids", "gaussian_ids")
    camera_ids = _meta_tensor(meta, "camera_ids")
    opacities = _meta_tensor(meta, "opacities")
    if gaussian_ids is None or camera_ids is None or opacities is None:
        raise ValueError("projected_meta anchor requires gaussian_ids, camera_ids, and opacities in meta")
    depths = _meta_tensor(meta, "depths")
    radii = _meta_tensor(meta, "radii")
    conics = _meta_tensor(meta, "conics")
    if depths is None:
        depths = means2d.new_zeros((int(means2d.shape[0]),))
    if radii is None:
        radii = means2d.new_ones((int(means2d.shape[0]),))
    if conics is None:
        conics = means2d.new_zeros((int(means2d.shape[0]), 3))

    gid = gaussian_ids.to(device=device, dtype=torch.long).reshape(-1)
    vid = camera_ids.to(device=device, dtype=torch.long).reshape(-1)
    uv = means2d.to(device=device, dtype=dtype).reshape(-1, 2)
    rows = int(uv.shape[0])
    if int(gid.numel()) != rows or int(vid.numel()) != rows:
        raise ValueError(
            "projected_meta row mismatch: "
            f"means2d={rows} gaussian_ids={int(gid.numel())} camera_ids={int(vid.numel())}"
        )
    support = _row_scalar(opacities.to(device=device, dtype=dtype), rows=rows, name="opacities").clamp_min(0.0)
    depth = _row_scalar(depths.to(device=device, dtype=dtype), rows=rows, name="depths")
    radius = _row_scalar(radii.to(device=device, dtype=dtype), rows=rows, name="radii", reduce="max").clamp_min(0.0)
    conic = conics.to(device=device, dtype=dtype).reshape(-1, 3)

    valid = (
        (gid >= 0)
        & (gid < int(num_children))
        & (vid >= 0)
        & (vid < int(num_views))
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
        & (radius > 0.0)
        & (support > 0.0)
    )
    valid = _apply_pair_valid_mask(
        valid=valid,
        uv=uv,
        view_id=vid,
        source_pair_valid_mask=source_pair_valid_mask,
        image_height=int(image_height),
        image_width=int(image_width),
    )
    keep = torch.nonzero(valid, as_tuple=False).reshape(-1)
    gid = gid.index_select(0, keep)
    vid = vid.index_select(0, keep)
    uv = uv.index_select(0, keep)
    support = support.index_select(0, keep)
    depth = depth.index_select(0, keep)
    radius = radius.index_select(0, keep)
    conic = conic.index_select(0, keep)

    child_stats = _scatter_weighted_view_stats(
        row_id=gid,
        view_id=vid,
        support=support,
        uv=uv,
        depth=depth,
        radius=radius,
        conic=conic,
        num_rows=int(num_children),
        num_views=int(num_views),
    )
    child_support = child_stats["support"]
    child_valid = child_support >= float(child_support_threshold)
    child_support_total = child_support.sum(dim=1)

    ctp = child_to_parent.to(device=device, dtype=torch.long).reshape(-1)
    if int(ctp.numel()) != int(num_children):
        raise ValueError(f"child_to_parent row mismatch: {int(ctp.numel())} vs {int(num_children)}")
    parent_support = means2d.new_zeros((int(num_parents), int(num_views)))
    parent_uv_sum = means2d.new_zeros((int(num_parents), int(num_views), 2))
    parent_depth_sum = means2d.new_zeros((int(num_parents), int(num_views)))
    parent_radius_sum = means2d.new_zeros((int(num_parents), int(num_views)))
    parent_conic_sum = means2d.new_zeros((int(num_parents), int(num_views), 3))
    if int(num_children) > 0 and int(num_parents) > 0:
        for view_idx in range(int(num_views)):
            w = child_support[:, view_idx].reshape(-1)
            parent_support[:, view_idx].index_add_(0, ctp, w)
            parent_uv_sum[:, view_idx].index_add_(0, ctp, child_stats["uv"][:, view_idx] * w[:, None])
            parent_depth_sum[:, view_idx].index_add_(0, ctp, child_stats["depth"][:, view_idx] * w)
            parent_radius_sum[:, view_idx].index_add_(0, ctp, child_stats["radius"][:, view_idx] * w)
            parent_conic_sum[:, view_idx].index_add_(0, ctp, child_stats["conic"][:, view_idx] * w[:, None])
    parent_denom = parent_support.clamp_min(1.0e-8)
    parent_uv = parent_uv_sum / parent_denom[..., None]
    parent_depth = parent_depth_sum / parent_denom
    parent_radius = parent_radius_sum / parent_denom
    parent_conic = parent_conic_sum / parent_denom[..., None]
    parent_valid = parent_support >= float(parent_support_threshold)
    parent_support_total = parent_support.sum(dim=1)

    child_ray = means2d.new_zeros((int(num_children), int(num_views), 3))
    out = SparseAnchorStats(
        child_uv=child_stats["uv"],
        child_support=child_support,
        child_valid=child_valid,
        child_depth=child_stats["depth"],
        child_radius=child_stats["radius"],
        child_conic=child_stats["conic"],
        child_ray=child_ray,
        parent_uv=parent_uv,
        parent_support=parent_support,
        parent_valid=parent_valid,
        parent_depth=parent_depth,
        parent_radius=parent_radius,
        parent_conic_approx=parent_conic,
        child_support_total=child_support_total,
        parent_support_total=parent_support_total,
    )
    return out.detach() if bool(detach_geometry) else out


__all__ = [
    "SparseAnchorStats",
    "build_cuda_scalar_anchor_stats",
    "build_projected_meta_anchor_stats",
    "cuda_scalar_anchor_available",
]
