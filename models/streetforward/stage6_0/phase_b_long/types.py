from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

ImageRef = Tuple[int, int]

PHASE_B_LONG_NAME = "6_0_phase_b"
LONG_TARGET_ROLES = (
    "final_history_recon",
    "final_history_nvs",
    "final_current_recon",
    "final_current_nvs",
)


@dataclass(frozen=True)
class LongEpisodeWindow:
    scene_id: int
    segment_id: int
    frame_pool: List[int]
    cam_pool: List[int]
    segment_start_frame: int
    segment_end_frame: int
    rigid_meta: Dict[str, Any]
    distant_meta: Dict[str, Any]
    episode_seed: int
    rollout_budget: int


@dataclass(frozen=True)
class LongRolloutShape:
    name: str
    repeats_per_anchor: int
    anchors_per_rollout: int

    @property
    def inner_K(self) -> int:
        return int(self.repeats_per_anchor) * int(self.anchors_per_rollout)


@dataclass(frozen=True)
class LongAnchor:
    anchor_id: int
    frame_idx: int
    chronological_rank: int
    rollout_order_rank: int


@dataclass(frozen=True)
class LongVisit:
    step_idx: int
    anchor_id: int
    frame_idx: int
    cam_idx: int
    repeat_idx: int
    rollout_order_rank: int
    chronological_rank: int
    visit_pos_code: float
    frame_time_code: float
    chronological_rank_code: float
    repeat_idx_code: float


@dataclass(frozen=True)
class LongRolloutPlan:
    scheduler_version: str
    phase: str
    scene_id: int
    segment_id: int
    episode_window_id: int
    rollout_id_in_episode: int
    shape_name: str
    repeats_per_anchor: int
    anchors_per_rollout: int
    inner_K: int
    anchor_frames_chronological: List[int]
    anchor_frames_rollout_order: List[int]
    visits: List[LongVisit]
    evidence_refs_by_step: List[List[ImageRef]]
    final_history_recon_refs: List[ImageRef]
    final_history_nvs_refs: List[ImageRef]
    final_current_recon_refs: List[ImageRef]
    final_current_nvs_refs: List[ImageRef]
    source_image_refs: List[ImageRef]
    target_image_refs: List[ImageRef]
    target_image_roles: List[str]
    rigid_meta: Dict[str, Any]
    distant_meta: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedLongPhaseBBatch:
    inner_K: int
    evidence_refs_by_step: List[List[ImageRef]]
    evidence_source_indices_by_step: List[List[int]]
    visits: List[LongVisit]
    shape_name: str
    repeats_per_anchor: int
    anchors_per_rollout: int
    anchor_frames_chronological: List[int]
    anchor_frames_rollout_order: List[int]
    target_role_indices: Dict[str, List[int]]
    final_history_recon_refs: List[ImageRef]
    final_history_nvs_refs: List[ImageRef]
    final_current_recon_refs: List[ImageRef]
    final_current_nvs_refs: List[ImageRef]
    final_history_recon_target_indices: List[int]
    final_history_nvs_target_indices: List[int]
    final_current_recon_target_indices: List[int]
    final_current_nvs_target_indices: List[int]
    final_history_refs: List[ImageRef]
    final_current_refs: List[ImageRef]
    final_history_target_indices: List[int]
    final_current_target_indices: List[int]
    step_frame_indices: List[int]
    step_repeat_indices: List[int]
    step_anchor_ids: List[int]
    step_rollout_order_ranks: List[int]
    step_chronological_ranks: List[int]
    visit_time_codes: List[Tuple[float, float, float, float]]
    source_index_by_ref: Dict[ImageRef, int]
    target_index_by_ref: Dict[ImageRef, int]
    rigid_meta: Dict[str, Any] = field(default_factory=dict)
    distant_meta: Dict[str, Any] = field(default_factory=dict)
    tbptt_meta: Dict[str, Any] = field(default_factory=dict)
    request_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LongVSMReadPack:
    bg: torch.Tensor
    seen_bg: torch.Tensor
    bg_indices: Optional[torch.Tensor] = None
    rigid: Optional[torch.Tensor] = None
    rigid_indices: Optional[torch.Tensor] = None
    rigid_seen: Optional[torch.Tensor] = None
    rigid_stable_mask: Optional[torch.Tensor] = None
    distant: Optional[torch.Tensor] = None
    distant_indices: Optional[torch.Tensor] = None
    distant_seen: Optional[torch.Tensor] = None


@dataclass
class BgOffsetDelta:
    means: torch.Tensor
    scales_log: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    mask: torch.Tensor
    indices: Optional[torch.Tensor] = None


@dataclass
class RigidOffsetDelta:
    indices: torch.Tensor
    stable_mask: torch.Tensor
    means_local: torch.Tensor
    scales_log: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor


@dataclass
class DistantOffsetDelta:
    indices: torch.Tensor
    scales_log: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    mask: torch.Tensor


@dataclass
class LongOffsetDelta:
    bg: BgOffsetDelta
    rigid: Optional[RigidOffsetDelta] = None
    distant: Optional[DistantOffsetDelta] = None
    aux: Dict[str, float] = field(default_factory=dict)

    def stats(self, prefix: str = "") -> Dict[str, float]:
        p = f"{prefix}/" if prefix else ""
        out = {
            f"{p}offset_bg_delta_means_norm": _mean_norm(self.bg.means),
            f"{p}offset_bg_delta_scales_norm": _mean_norm(self.bg.scales_log),
            f"{p}offset_bg_delta_opacity_norm": _mean_norm(self.bg.opacity_logit),
            f"{p}offset_bg_delta_sh_dc_norm": _mean_norm(self.bg.sh_dc),
            f"{p}offset_bg_active_rows": float(int(self.bg.means.shape[0])),
        }
        if self.rigid is not None:
            out.update(
                {
                    f"{p}offset_rigid_delta_means_local_norm": _mean_norm(self.rigid.means_local),
                    f"{p}offset_rigid_delta_scales_norm": _mean_norm(self.rigid.scales_log),
                    f"{p}offset_rigid_delta_opacity_norm": _mean_norm(self.rigid.opacity_logit),
                    f"{p}offset_rigid_delta_sh_dc_norm": _mean_norm(self.rigid.sh_dc),
                    f"{p}offset_rigid_active_rows": float(int(self.rigid.indices.numel())),
                    f"{p}offset_rigid_stable_rows": float(int(self.rigid.stable_mask.sum().item()))
                    if self.rigid.stable_mask.numel()
                    else 0.0,
                }
            )
        if self.distant is not None:
            out.update(
                {
                    f"{p}offset_distant_active_rows": float(int(self.distant.indices.numel())),
                    f"{p}offset_distant_delta_scales_norm": _mean_norm(self.distant.scales_log),
                    f"{p}offset_distant_opacity_norm": _mean_norm(self.distant.opacity_logit),
                    f"{p}offset_distant_sh_dc_norm": _mean_norm(self.distant.sh_dc),
                    f"{p}offset_distant_sh_rest_norm": _mean_norm(self.distant.sh_rest),
                }
            )
        out.update({f"{p}{k}": float(v) for k, v in self.aux.items()})
        return out


def _mean_norm(x: Optional[torch.Tensor]) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(x.detach().float().norm(dim=-1).mean().item())


def coerce_branch_valid(
    value: Optional[torch.Tensor],
    *,
    n: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return ref.new_ones((int(n), 1))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() == 1:
        out = out[:, None]
    if int(out.shape[0]) != int(n):
        raise ValueError(f"valid row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    return out[:, :1].clamp(0.0, 1.0)


def coerce_branch_support(
    value: Optional[torch.Tensor],
    *,
    n: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return ref.new_ones((int(n), 1))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() == 1:
        out = out[:, None]
    if int(out.shape[0]) != int(n):
        raise ValueError(f"support row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    return out[:, :1].clamp_min(0.0)


def coerce_view_code(
    value: Optional[torch.Tensor],
    *,
    n: int,
    ref: torch.Tensor,
    view_dim: int,
) -> torch.Tensor:
    if value is None:
        return ref.new_zeros((int(n), int(view_dim)))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() == 1:
        out = out[:, None]
    if int(out.shape[0]) != int(n):
        raise ValueError(f"view_code row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) == int(view_dim):
        return out
    if int(out.shape[1]) > int(view_dim):
        return out[:, : int(view_dim)]
    pad = out.new_zeros((int(n), int(view_dim) - int(out.shape[1])))
    return torch.cat([out, pad], dim=-1)


def rigid_stable_mask_from_meta(
    rigid_meta: Optional[Dict[str, Any]],
    *,
    num_rows: int,
    device: torch.device,
) -> torch.Tensor:
    meta = dict(rigid_meta or {})
    for key in ("stable_mask", "is_stable"):
        raw = meta.get(key)
        if raw is None:
            continue
        mask = raw.to(device=device, dtype=torch.bool).reshape(-1) if torch.is_tensor(raw) else torch.as_tensor(raw, device=device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != int(num_rows):
            raise ValueError(f"rigid_meta.{key} length {int(mask.numel())} != rigid rows {int(num_rows)}")
        return mask
    all_stable = bool(meta.get("has_stable_ids", False)) and bool(meta.get("canonical_available", False)) and bool(
        meta.get("object_transforms_available", False)
    )
    return torch.full((int(num_rows),), bool(all_stable), dtype=torch.bool, device=device)
