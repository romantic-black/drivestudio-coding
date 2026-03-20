"""
Explicit (frame_idx, cam_id) selection for MultiScene-style batches.

Used by convert_batch_to_minimal_format when training.view_selection.mode == explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ExplicitViewSelection:
    """Parsed training.view_selection for mode=explicit."""

    source_refs: List[Tuple[int, int]]
    target_refs: List[Tuple[int, int]]
    primary_source_frame_idx: Optional[int]


def parse_view_selection(view_selection: Any) -> Optional[ExplicitViewSelection]:
    """
    Return ExplicitViewSelection if mode is explicit and source/target lists are non-empty.
    Otherwise return None (caller uses legacy num_targets slicing).
    """
    if view_selection is None:
        return None
    if hasattr(view_selection, "get"):
        mode = view_selection.get("mode")
        source_list = view_selection.get("source")
        target_list = view_selection.get("target")
        primary = view_selection.get("primary_source_frame_idx")
    else:
        mode = getattr(view_selection, "mode", None)
        source_list = getattr(view_selection, "source", None)
        target_list = getattr(view_selection, "target", None)
        primary = getattr(view_selection, "primary_source_frame_idx", None)

    if str(mode).lower() != "explicit":
        return None
    if not target_list:
        raise ValueError(
            "training.view_selection.mode=explicit requires non-empty 'target' observation list."
        )
    if source_list is None:
        source_list = []

    source_refs = _refs_from_list(source_list, field_name="source") if source_list else []
    target_refs = _refs_from_list(target_list, field_name="target")
    if source_refs:
        _assert_no_duplicate_refs(source_refs, which="source")
    _assert_no_duplicate_refs(target_refs, which="target")

    primary_i: Optional[int] = None
    if primary is not None:
        primary_i = int(primary)

    return ExplicitViewSelection(
        source_refs=source_refs,
        target_refs=target_refs,
        primary_source_frame_idx=primary_i,
    )


def _refs_from_list(obs_list: Any, *, field_name: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, item in enumerate(obs_list):
        if hasattr(item, "get"):
            fj = item.get("frame_idx")
            cid = item.get("cam_id")
        else:
            fj = getattr(item, "frame_idx", None)
            cid = getattr(item, "cam_id", None)
        if fj is None or cid is None:
            raise ValueError(
                f"training.view_selection.{field_name}[{i}] must have frame_idx and cam_id, got {item!r}"
            )
        out.append((int(fj), int(cid)))
    return out


def _assert_no_duplicate_refs(refs: List[Tuple[int, int]], *, which: str) -> None:
    seen: set[Tuple[int, int]] = set()
    for f, c in refs:
        key = (f, c)
        if key in seen:
            raise ValueError(f"Duplicate (frame_idx, cam_id)={key} in view_selection.{which}")
        seen.add(key)


def _available_pairs_hint(
    fi: torch.Tensor,
    ci: torch.Tensor,
    *,
    max_pairs: int = 40,
) -> str:
    """Compact sorted list of unique (frame_idx, cam_id) for error messages."""
    pairs = sorted(
        set(zip(fi.detach().cpu().long().tolist(), ci.detach().cpu().long().tolist()))
    )
    if not pairs:
        return " (no rows in tensor)"
    shown = pairs[:max_pairs]
    more = len(pairs) - len(shown)
    suffix = f" ... (+{more} more)" if more > 0 else ""
    return f" Available unique (frame_idx, cam_id) in this tensor ({len(pairs)} total): {shown}{suffix}"


def find_row(
    frame_indices: torch.Tensor,
    cam_indices: torch.Tensor,
    frame_idx: int,
    cam_id: int,
    *,
    role: str,
) -> int:
    """Return the unique row index where frame_indices[i]==frame_idx and cam_indices[i]==cam_id."""
    if frame_indices.shape[0] != cam_indices.shape[0]:
        raise ValueError(
            f"{role}: frame_indices and cam_indices length mismatch: "
            f"{frame_indices.shape[0]} vs {cam_indices.shape[0]}"
        )
    fi = frame_indices.long()
    ci = cam_indices.long()
    mask = (fi == int(frame_idx)) & (ci == int(cam_id))
    count = int(mask.sum().item())
    if count == 0:
        hint = _available_pairs_hint(fi, ci)
        raise ValueError(
            f"{role}: no row with frame_idx={frame_idx}, cam_id={cam_id}.{hint} "
            "Use pairs from your captured batch (inspect target['frame_indices']/['cam_indices']), "
            "or increase num_source_keyframes/num_target_keyframes and re-capture."
        )
    if count > 1:
        raise ValueError(
            f"{role}: multiple ({count}) rows with frame_idx={frame_idx}, cam_id={cam_id}; batch is ambiguous."
        )
    return int(torch.nonzero(mask, as_tuple=False)[0].item())


def build_minimal_target_entry_from_row(
    target_data: Dict[str, Any],
    row: int,
    device: torch.device,
) -> Dict[str, Any]:
    """One target dict matching convert_batch_to_minimal_format target loop."""
    from datasets.base.pixel_source import get_rays

    view = type(
        "View",
        (),
        {
            "camtoworlds": target_data["extrinsics"][row].to(device),
            "Ks": target_data["intrinsics"][row][:3, :3].unsqueeze(0).to(device),
        },
    )()
    gt_image = target_data["image"][row].to(device)
    viewdirs_list: Optional[torch.Tensor] = None
    target_viewdirs = target_data.get("viewdirs")
    if target_viewdirs is not None:
        viewdirs_list = target_viewdirs[row].to(device)
    else:
        gt = target_data["image"][row]
        h, w = int(gt.shape[0]), int(gt.shape[1])
        c2w = target_data["extrinsics"][row]
        intrinsic = target_data["intrinsics"][row][:3, :3]
        if c2w.dim() == 2:
            c2w = c2w.unsqueeze(0)
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)
        y_coords = torch.arange(h, device=device, dtype=torch.float32)
        x_coords = torch.arange(w, device=device, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
        _, viewdirs, _ = get_rays(
            x_grid.flatten(),
            y_grid.flatten(),
            c2w.to(device),
            intrinsic.to(device),
        )
        viewdirs_list = viewdirs.reshape(h, w, 3)

    frame_indices = target_data.get("frame_indices")
    frame_idx_i = int(frame_indices[row]) if frame_indices is not None else 0

    sky_mask = target_data.get("sky_mask")
    egocar_mask = target_data.get("egocar_mask")
    out: Dict[str, Any] = {
        "frame_idx": frame_idx_i,
        "view": view,
        "gt_image": gt_image,
    }
    if sky_mask is not None:
        out["sky_mask"] = sky_mask[row].to(device)
    if egocar_mask is not None:
        out["egocar_mask"] = egocar_mask[row].to(device)
    if viewdirs_list is not None:
        out["viewdirs"] = viewdirs_list
    return out


def build_source_view_image_from_row(
    source_data: Dict[str, Any],
    row: int,
    device: torch.device,
) -> Tuple[Any, torch.Tensor, int]:
    """One source view + image + frame_idx for Stage3 2D path."""
    view = type(
        "View",
        (),
        {
            "camtoworlds": source_data["extrinsics"][row].to(device),
            "Ks": source_data["intrinsics"][row][:3, :3].unsqueeze(0).to(device),
        },
    )()
    image = source_data["image"][row].to(device)
    frame_indices = source_data.get("frame_indices")
    frame_idx_i = int(frame_indices[row]) if frame_indices is not None else 0
    return view, image, frame_idx_i


def build_explicit_targets_only(
    batch: Dict[str, Any],
    device: torch.device,
    selection: ExplicitViewSelection,
) -> List[Dict[str, Any]]:
    """Explicit target list from batch['target'] only (Stage 1.x without 2D source)."""
    target_data = batch.get("target")
    if not isinstance(target_data, dict):
        raise ValueError(
            "explicit view_selection requires batch['target'] dict when include_source_for_2d=False."
        )
    tfi = target_data.get("frame_indices")
    tci = target_data.get("cam_indices")
    if tfi is None or tci is None:
        raise ValueError(
            "explicit view_selection requires frame_indices and cam_indices on batch['target']."
        )
    targets_minimal: List[Dict[str, Any]] = []
    for frame_idx, cam_id in selection.target_refs:
        row = find_row(tfi, tci, frame_idx, cam_id, role="batch['target']")
        targets_minimal.append(build_minimal_target_entry_from_row(target_data, row, device))
    return targets_minimal


def build_explicit_minimal_batch_parts(
    batch: Dict[str, Any],
    device: torch.device,
    selection: ExplicitViewSelection,
) -> Tuple[List[Dict[str, Any]], List[Any], List[torch.Tensor], int]:
    """
    Returns:
      targets_minimal, source_views, source_images, source_frame_idx
    """
    if not selection.source_refs:
        raise ValueError(
            "explicit view_selection requires non-empty 'source' when using Stage3 (include_source_for_2d=True)."
        )
    source_data = batch.get("source")
    target_data = batch.get("target")
    if not isinstance(source_data, dict) or not isinstance(target_data, dict):
        raise ValueError(
            "explicit view_selection requires batch['source'] and batch['target'] dicts "
            "(MultiScene get_segment_batch / overfit .pt format)."
        )
    sfi = source_data.get("frame_indices")
    sci = source_data.get("cam_indices")
    tfi = target_data.get("frame_indices")
    tci = target_data.get("cam_indices")
    if sfi is None or sci is None or tfi is None or tci is None:
        raise ValueError(
            "explicit view_selection requires frame_indices and cam_indices on both batch['source'] and batch['target']."
        )

    targets_minimal: List[Dict[str, Any]] = []
    for frame_idx, cam_id in selection.target_refs:
        row = find_row(tfi, tci, frame_idx, cam_id, role="batch['target']")
        targets_minimal.append(build_minimal_target_entry_from_row(target_data, row, device))

    source_views: List[Any] = []
    source_images: List[torch.Tensor] = []
    for frame_idx, cam_id in selection.source_refs:
        row = find_row(sfi, sci, frame_idx, cam_id, role="batch['source']")
        view, img, _ = build_source_view_image_from_row(source_data, row, device)
        source_views.append(view)
        source_images.append(img)

    if selection.primary_source_frame_idx is not None:
        source_frame_idx = int(selection.primary_source_frame_idx)
    else:
        r0 = find_row(
            sfi,
            sci,
            selection.source_refs[0][0],
            selection.source_refs[0][1],
            role="batch['source']",
        )
        _, _, source_frame_idx = build_source_view_image_from_row(source_data, r0, device)

    return targets_minimal, source_views, source_images, source_frame_idx


__all__ = [
    "ExplicitViewSelection",
    "parse_view_selection",
    "find_row",
    "build_explicit_targets_only",
    "build_explicit_minimal_batch_parts",
    "build_minimal_target_entry_from_row",
    "build_source_view_image_from_row",
]
