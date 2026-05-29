from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.local_gs_state import LocalGSState
from models.streetforward.struct_decoders.voxel_layout_utils import SegmentCellIndex, build_segment_cell_index

from .streaming_vsm import (
    DISTANT_MODE_APPEARANCE_SCALE,
    DISTANT_MODE_FROZEN,
    StreamingSelectiveSSMBranch,
    _check_distant_mode,
    _prefix,
)
from .types import (
    LongVSMReadPack,
    coerce_branch_support,
    coerce_branch_valid,
    coerce_view_code,
    rigid_stable_mask_from_meta,
)


@dataclass
class RigidObjectCellIndex:
    stable_ids: torch.Tensor
    row_to_stable_pos: torch.Tensor
    object_ids: torch.Tensor
    row_to_object_pos: Optional[torch.Tensor]
    cell_keys: torch.Tensor
    row_to_cell_pos: Optional[torch.Tensor]
    local_grid: Tuple[int, int, int]


@dataclass
class LongCellIndexPack:
    bg: SegmentCellIndex
    rigid: Optional[RigidObjectCellIndex] = None


@dataclass
class LongCellVSMState:
    bg_point_h: torch.Tensor
    bg_point_seen: torch.Tensor
    bg_cell_h: torch.Tensor
    bg_cell_seen: torch.Tensor
    bg_global_h: torch.Tensor
    bg_global_seen: torch.Tensor
    index: LongCellIndexPack
    rigid_point_ids: Optional[torch.Tensor] = None
    rigid_point_h: Optional[torch.Tensor] = None
    rigid_point_seen: Optional[torch.Tensor] = None
    rigid_object_h: Optional[torch.Tensor] = None
    rigid_object_seen: Optional[torch.Tensor] = None
    rigid_cell_h: Optional[torch.Tensor] = None
    rigid_cell_seen: Optional[torch.Tensor] = None
    distant_h: Optional[torch.Tensor] = None
    distant_seen: Optional[torch.Tensor] = None
    episode_id: int = -1

    @property
    def bg_h(self) -> torch.Tensor:
        return self.bg_point_h

    @property
    def bg_seen(self) -> torch.Tensor:
        return self.bg_point_seen

    def detach(self) -> "LongCellVSMState":
        def d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach()

        return replace(
            self,
            bg_point_h=self.bg_point_h.detach(),
            bg_point_seen=self.bg_point_seen.detach(),
            bg_cell_h=self.bg_cell_h.detach(),
            bg_cell_seen=self.bg_cell_seen.detach(),
            bg_global_h=self.bg_global_h.detach(),
            bg_global_seen=self.bg_global_seen.detach(),
            rigid_point_ids=d(self.rigid_point_ids),
            rigid_point_h=d(self.rigid_point_h),
            rigid_point_seen=d(self.rigid_point_seen),
            rigid_object_h=d(self.rigid_object_h),
            rigid_object_seen=d(self.rigid_object_seen),
            rigid_cell_h=d(self.rigid_cell_h),
            rigid_cell_seen=d(self.rigid_cell_seen),
            distant_h=d(self.distant_h),
            distant_seen=d(self.distant_seen),
        )

    def detach_to_cache_optional(self) -> "LongCellVSMState":
        return self.detach()

    def seen_pack(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "bg": self.bg_point_seen,
            "rigid": self.rigid_point_seen,
            "distant": self.distant_seen,
        }


def _index_optional_rows(value: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.index_select(0, idx.to(device=value.device, dtype=torch.long))


def _scatter_weighted_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    *,
    dim_size: int,
) -> torch.Tensor:
    if src.dim() != 2:
        raise ValueError("scatter weighted mean expects src [N,C].")
    if index.dim() != 1 or int(index.shape[0]) != int(src.shape[0]):
        raise ValueError("scatter weighted mean index must be [N].")
    if weight.dim() == 1:
        weight = weight[:, None]
    if int(weight.shape[0]) != int(src.shape[0]):
        raise ValueError("scatter weighted mean weight rows must match src.")
    out = src.new_zeros((int(dim_size), int(src.shape[1])))
    denom = src.new_zeros((int(dim_size), 1))
    if int(src.shape[0]) == 0:
        return out
    w = weight.to(device=src.device, dtype=src.dtype).clamp_min(0.0)
    out.index_add_(0, index.to(device=src.device, dtype=torch.long), src * w)
    denom.index_add_(0, index.to(device=src.device, dtype=torch.long), w)
    return out / denom.clamp(min=1.0e-6)


def _scatter_sum_1d(value: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    if value.dim() == 2:
        value = value[:, 0]
    out = value.new_zeros((int(dim_size),))
    if int(value.numel()) > 0:
        out.index_add_(0, index.to(device=value.device, dtype=torch.long), value.reshape(-1))
    return out


class LongCellStreamingVSM(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        view_dim: int = 2,
        bg_point_mem_dim: int = 32,
        bg_cell_mem_dim: int = 64,
        bg_global_mem_dim: int = 64,
        bg_read_dim: Optional[int] = None,
        bg_cell_voxel_size: float = 0.5,
        use_global_memory: bool = True,
        rigid_point_mem_dim: int = 32,
        rigid_object_mem_dim: int = 64,
        rigid_cell_mem_dim: int = 64,
        rigid_read_dim: Optional[int] = None,
        rigid_local_grid: Tuple[int, int, int] = (8, 8, 4),
        distant_mem_dim: int = 32,
        input_dim: int = 96,
        dtype: str = "bf16",
        distant_mode: str = "frozen_render_only",
        support_fallback_when_no_valid: bool = False,
        support_fallback_min: float = 0.0,
        support_fallback_scale: float = 1.0,
        bg_active_sparse: bool = True,
        bg_outside_policy: str = "mark_invalid",
        bg_point_context_source: str = "previous_cell_global",
        bg_final_read_context_source: str = "updated_cell_global",
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.view_dim = int(view_dim)
        self.bg_point_mem_dim = int(bg_point_mem_dim)
        self.bg_cell_mem_dim = int(bg_cell_mem_dim)
        self.bg_global_mem_dim = int(bg_global_mem_dim)
        self.bg_mem_dim = int(bg_read_dim or bg_cell_mem_dim)
        self.bg_cell_voxel_size = float(bg_cell_voxel_size)
        self.use_global_memory = bool(use_global_memory)
        self.rigid_point_mem_dim = int(rigid_point_mem_dim)
        self.rigid_object_mem_dim = int(rigid_object_mem_dim)
        self.rigid_cell_mem_dim = int(rigid_cell_mem_dim)
        self.rigid_mem_dim = int(rigid_read_dim or rigid_object_mem_dim)
        self.rigid_local_grid = tuple(int(x) for x in rigid_local_grid)
        if len(self.rigid_local_grid) != 3 or any(int(x) <= 0 for x in self.rigid_local_grid):
            raise ValueError("rigid_local_grid must be three positive integers.")
        self.distant_mem_dim = int(distant_mem_dim)
        self.state_dtype_name = str(dtype)
        self.distant_mode = _check_distant_mode(str(distant_mode))
        self.support_fallback_when_no_valid = bool(support_fallback_when_no_valid)
        self.support_fallback_min = float(support_fallback_min)
        self.support_fallback_scale = float(support_fallback_scale)
        self.bg_active_sparse = bool(bg_active_sparse)
        self.bg_outside_policy = str(bg_outside_policy)
        self.bg_point_context_source = str(bg_point_context_source)
        self.bg_final_read_context_source = str(bg_final_read_context_source)
        allowed_context_sources = {"previous_cell_global", "updated_cell_global"}
        if self.bg_point_context_source not in allowed_context_sources:
            raise ValueError(
                "bg_point_context_source must be one of "
                f"{sorted(allowed_context_sources)}, got {self.bg_point_context_source!r}."
            )
        if self.bg_final_read_context_source not in allowed_context_sources:
            raise ValueError(
                "bg_final_read_context_source must be one of "
                f"{sorted(allowed_context_sources)}, got {self.bg_final_read_context_source!r}."
            )

        branch_kwargs = dict(
            event_dim=int(event_dim),
            view_dim=int(view_dim),
            input_dim=int(input_dim),
            support_fallback_when_no_valid=bool(self.support_fallback_when_no_valid),
            support_fallback_min=float(self.support_fallback_min),
            support_fallback_scale=float(self.support_fallback_scale),
        )
        self.bg_point_ssm = StreamingSelectiveSSMBranch(mem_dim=int(bg_point_mem_dim), **branch_kwargs)
        self.bg_cell_ssm = StreamingSelectiveSSMBranch(mem_dim=int(bg_cell_mem_dim), **branch_kwargs)
        self.bg_global_ssm = StreamingSelectiveSSMBranch(mem_dim=int(bg_global_mem_dim), **branch_kwargs)
        self.rigid_point_ssm = StreamingSelectiveSSMBranch(mem_dim=int(rigid_point_mem_dim), **branch_kwargs)
        self.rigid_object_ssm = StreamingSelectiveSSMBranch(mem_dim=int(rigid_object_mem_dim), **branch_kwargs)
        self.rigid_cell_ssm = StreamingSelectiveSSMBranch(mem_dim=int(rigid_cell_mem_dim), **branch_kwargs)
        self.distant_ssm = StreamingSelectiveSSMBranch(mem_dim=int(distant_mem_dim), **branch_kwargs)

        self.bg_point_event_proj = nn.Linear(int(event_dim) + int(bg_cell_mem_dim) + int(bg_global_mem_dim), int(event_dim))
        self.bg_point_read_proj = nn.Linear(int(bg_point_mem_dim), int(self.bg_mem_dim))
        self.bg_cell_read_proj = nn.Linear(int(bg_cell_mem_dim), int(self.bg_mem_dim))
        self.bg_global_read_proj = nn.Linear(int(bg_global_mem_dim), int(self.bg_mem_dim))
        self.bg_read_gate = nn.Linear(int(self.bg_mem_dim) * 3 + int(view_dim) + 1, int(self.bg_mem_dim))
        self.bg_read_norm = nn.LayerNorm(int(self.bg_mem_dim))

        self.rigid_point_event_proj = nn.Linear(
            int(event_dim) + int(rigid_object_mem_dim) + int(rigid_cell_mem_dim),
            int(event_dim),
        )
        self.rigid_point_read_proj = nn.Linear(int(rigid_point_mem_dim), int(self.rigid_mem_dim))
        self.rigid_object_read_proj = nn.Linear(int(rigid_object_mem_dim), int(self.rigid_mem_dim))
        self.rigid_cell_read_proj = nn.Linear(int(rigid_cell_mem_dim), int(self.rigid_mem_dim))
        self.rigid_read_gate = nn.Linear(int(self.rigid_mem_dim) * 3 + int(view_dim) + 1, int(self.rigid_mem_dim))
        self.rigid_read_norm = nn.LayerNorm(int(self.rigid_mem_dim))

    def _state_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.state_dtype_name.lower() in {"bf16", "bfloat16"} and ref.is_cuda:
            return torch.bfloat16
        if self.state_dtype_name.lower() in {"fp16", "float16"}:
            return torch.float16
        return ref.dtype

    def _active_indices_and_signal(
        self,
        *,
        event: torch.Tensor,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        n = int(event.shape[0])
        valid_f = coerce_branch_valid(valid, n=n, ref=event)
        support_f = coerce_branch_support(support, n=n, ref=event)
        signal = valid_f
        fallback_used = 0.0
        if bool(self.support_fallback_when_no_valid) and support is not None and float(valid_f.detach().sum().item()) == 0.0:
            signal = (support_f.clamp_min(0.0) * float(self.support_fallback_scale)).clamp(max=1.0)
            if float(self.support_fallback_min) > 0.0:
                signal = signal * (support_f > float(self.support_fallback_min)).to(dtype=signal.dtype)
            fallback_used = 1.0 if float(signal.detach().sum().item()) > 0.0 else 0.0
        active = signal.reshape(-1) > 0.0
        idx = torch.nonzero(active, as_tuple=False).squeeze(1).to(dtype=torch.long)
        aux = {
            "active_rows": float(int(idx.numel())),
            "total_rows": float(n),
            "active_ratio": float(int(idx.numel()) / max(n, 1)),
            "hard_valid_ratio": float((valid_f.detach() > 0.0).float().mean().item()) if n else 0.0,
            "support_mean": float(support_f.detach().float().mean().item()) if n else 0.0,
            "support_max": float(support_f.detach().float().max().item()) if n else 0.0,
            "support_positive_ratio": float((support_f.detach() > 0.0).float().mean().item()) if n else 0.0,
            "support_fallback_used": float(fallback_used),
        }
        return idx, signal, support_f, aux

    @staticmethod
    def _select_visit_time_code(
        visit_time_code: Optional[torch.Tensor],
        *,
        idx: torch.Tensor,
        n_total: int,
    ) -> Optional[torch.Tensor]:
        if visit_time_code is None:
            return None
        if visit_time_code.dim() >= 2 and int(visit_time_code.shape[0]) == int(n_total):
            return visit_time_code.index_select(0, idx.to(device=visit_time_code.device, dtype=torch.long))
        return visit_time_code

    def _write_subset(
        self,
        branch: StreamingSelectiveSSMBranch,
        *,
        h_full: torch.Tensor,
        seen_full: torch.Tensor,
        rows: torch.Tensor,
        event: torch.Tensor,
        view_code: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        step_idx: int,
        repeat_idx: int,
        branch_id: int,
        visit_time_code: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        rows_state = rows.to(device=h_full.device, dtype=torch.long).reshape(-1)
        if int(rows_state.numel()) == 0:
            empty_read = event.new_zeros((0, int(branch.mem_dim)), dtype=compute_dtype)
            empty_seen = event.new_zeros((0, 1), dtype=compute_dtype)
            return h_full, seen_full, empty_read, empty_seen, branch._empty_aux(h_full)
        h_old = h_full.index_select(0, rows_state).to(device=event.device, dtype=compute_dtype)
        seen_old = seen_full.index_select(0, rows_state).to(device=event.device, dtype=compute_dtype)
        h_new, seen_new, read, aux = branch.write(
            h=h_old,
            seen=seen_old,
            event=event.to(dtype=compute_dtype),
            view_code=view_code,
            valid=valid,
            support=support,
            step_idx=int(step_idx),
            repeat_idx=int(repeat_idx),
            branch_id=int(branch_id),
            visit_time_code=visit_time_code,
        )
        h_out = h_full.index_copy(0, rows_state, h_new.to(device=h_full.device, dtype=h_full.dtype))
        seen_out = seen_full.index_copy(0, rows_state, seen_new.to(device=seen_full.device, dtype=seen_full.dtype))
        return h_out, seen_out, read, seen_new, aux

    @staticmethod
    def _batch_aabb(batch: Optional[Dict[str, Any]], ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, dict) or "aabb" not in batch:
            lo = ref.detach().amin(dim=0) if int(ref.shape[0]) > 0 else ref.new_zeros((3,))
            hi = ref.detach().amax(dim=0) if int(ref.shape[0]) > 0 else ref.new_ones((3,))
            pad = ref.new_full((3,), 1.0e-3)
            return lo - pad, hi + pad
        aabb = batch["aabb"]
        aabb_t = aabb.to(device=ref.device, dtype=ref.dtype) if torch.is_tensor(aabb) else torch.as_tensor(aabb, device=ref.device, dtype=ref.dtype)
        if tuple(aabb_t.shape) != (2, 3):
            raise ValueError(f"batch['aabb'] must have shape [2,3], got {tuple(aabb_t.shape)}")
        return aabb_t[0], aabb_t[1]

    def _build_rigid_index(self, *, base_state: LocalGSState, stable_mask: torch.Tensor) -> Optional[RigidObjectCellIndex]:
        if base_state.rigid is None:
            return None
        n = int(base_state.rigid.means.shape[0])
        device = base_state.rigid.means.device
        stable_ids = torch.nonzero(stable_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(1).long()
        row_to_stable = torch.full((n,), -1, device=device, dtype=torch.long)
        if int(stable_ids.numel()) > 0:
            row_to_stable[stable_ids] = torch.arange(int(stable_ids.numel()), device=device, dtype=torch.long)

        template = base_state.rigid_template
        if template is None or template.point_ids is None or int(template.point_ids.shape[0]) != n:
            return RigidObjectCellIndex(
                stable_ids=stable_ids,
                row_to_stable_pos=row_to_stable,
                object_ids=torch.zeros((0,), device=device, dtype=torch.long),
                row_to_object_pos=None,
                cell_keys=torch.zeros((0, 2), device=device, dtype=torch.long),
                row_to_cell_pos=None,
                local_grid=self.rigid_local_grid,
            )

        point_ids = template.point_ids.to(device=device, dtype=torch.long).reshape(n, -1)[:, 0]
        if int(point_ids.numel()) == 0:
            object_ids = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            object_ids = torch.unique(point_ids, sorted=True)
        row_to_object = torch.full((n,), -1, device=device, dtype=torch.long)
        for pos, obj_id in enumerate(object_ids.tolist()):
            row_to_object[point_ids == int(obj_id)] = int(pos)

        gx, gy, gz = self.rigid_local_grid
        row_to_cell = torch.full((n,), -1, device=device, dtype=torch.long)
        cell_keys_all = []
        cell_rows = []
        means = base_state.rigid.means.detach()
        for pos, obj_id in enumerate(object_ids.tolist()):
            rows = torch.nonzero(point_ids == int(obj_id), as_tuple=False).squeeze(1)
            if int(rows.numel()) == 0:
                continue
            xyz = means.index_select(0, rows)
            lo = xyz.amin(dim=0)
            hi = xyz.amax(dim=0)
            denom = (hi - lo).clamp(min=1.0e-6)
            norm = ((xyz - lo) / denom).clamp(0.0, 1.0 - 1.0e-6)
            grid = torch.floor(norm * xyz.new_tensor([gx, gy, gz])).long()
            linear = grid[:, 0] + int(gx) * (grid[:, 1] + int(gy) * grid[:, 2])
            cell_keys_all.append(torch.stack([torch.full_like(linear, int(pos)), linear], dim=1))
            cell_rows.append(rows)
        if cell_keys_all:
            cell_keys_cat = torch.cat(cell_keys_all, dim=0)
            rows_cat = torch.cat(cell_rows, dim=0)
            unique_cell_keys, inverse = torch.unique(cell_keys_cat, dim=0, sorted=True, return_inverse=True)
            row_to_cell[rows_cat] = inverse
        else:
            unique_cell_keys = torch.zeros((0, 2), device=device, dtype=torch.long)

        return RigidObjectCellIndex(
            stable_ids=stable_ids,
            row_to_stable_pos=row_to_stable,
            object_ids=object_ids,
            row_to_object_pos=row_to_object,
            cell_keys=unique_cell_keys,
            row_to_cell_pos=row_to_cell,
            local_grid=self.rigid_local_grid,
        )

    def init_state(
        self,
        *,
        base_state: LocalGSState,
        batch: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        rigid_meta: Optional[Dict[str, Any]] = None,
        distant_mode: Optional[str] = None,
        episode_id: int = -1,
    ) -> LongCellVSMState:
        mode = _check_distant_mode(str(distant_mode or self.distant_mode))
        ref = base_state.bg.means
        state_dtype = dtype or self._state_dtype(ref)
        aabb_min, aabb_max = self._batch_aabb(batch, ref)
        bg_index = build_segment_cell_index(
            ref.detach(),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            voxel_size=float(self.bg_cell_voxel_size),
            strict_inside=False,
            outside_policy=str(self.bg_outside_policy),
        )
        n_bg = int(ref.shape[0])
        n_distant = int(base_state.distant.means.shape[0]) if base_state.distant is not None else 0
        n_rigid = int(base_state.rigid.means.shape[0]) if base_state.rigid is not None else 0
        stable_mask = rigid_stable_mask_from_meta(rigid_meta, num_rows=n_rigid, device=ref.device) if n_rigid > 0 else ref.new_zeros((0,), dtype=torch.bool)
        rigid_index = self._build_rigid_index(base_state=base_state, stable_mask=stable_mask) if n_rigid > 0 else None
        num_stable = int(rigid_index.stable_ids.numel()) if rigid_index is not None else 0
        num_objects = int(rigid_index.object_ids.numel()) if rigid_index is not None else 0
        num_cells = int(rigid_index.cell_keys.shape[0]) if rigid_index is not None else 0

        return LongCellVSMState(
            bg_point_h=torch.zeros((n_bg, int(self.bg_point_mem_dim)), device=ref.device, dtype=state_dtype),
            bg_point_seen=torch.zeros((n_bg, 1), device=ref.device, dtype=state_dtype),
            bg_cell_h=torch.zeros((bg_index.num_cells, int(self.bg_cell_mem_dim)), device=ref.device, dtype=state_dtype),
            bg_cell_seen=torch.zeros((bg_index.num_cells, 1), device=ref.device, dtype=state_dtype),
            bg_global_h=torch.zeros((1, int(self.bg_global_mem_dim)), device=ref.device, dtype=state_dtype),
            bg_global_seen=torch.zeros((1, 1), device=ref.device, dtype=state_dtype),
            index=LongCellIndexPack(bg=bg_index, rigid=rigid_index),
            rigid_point_ids=rigid_index.stable_ids.detach() if rigid_index is not None else None,
            rigid_point_h=(
                torch.zeros((num_stable, int(self.rigid_point_mem_dim)), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_point_seen=(
                torch.zeros((num_stable, 1), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_object_h=(
                torch.zeros((num_objects, int(self.rigid_object_mem_dim)), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_object_seen=(
                torch.zeros((num_objects, 1), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_cell_h=(
                torch.zeros((num_cells, int(self.rigid_cell_mem_dim)), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_cell_seen=(
                torch.zeros((num_cells, 1), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            distant_h=(
                torch.zeros((n_distant, int(self.distant_mem_dim)), device=ref.device, dtype=state_dtype)
                if mode == DISTANT_MODE_APPEARANCE_SCALE and n_distant > 0
                else None
            ),
            distant_seen=(
                torch.zeros((n_distant, 1), device=ref.device, dtype=state_dtype)
                if mode == DISTANT_MODE_APPEARANCE_SCALE and n_distant > 0
                else None
            ),
            episode_id=int(episode_id),
        )

    def _write_read_bg(
        self,
        *,
        state: LongCellVSMState,
        event: Any,
        step_idx: int,
        frame_idx: int,
        repeat_idx: int,
        visit_time_code: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> tuple[LongCellVSMState, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        event_bg = event.event_bg
        if int(event_bg.shape[0]) != int(state.bg_point_h.shape[0]):
            raise ValueError(
                "event_bg rows must match bg memory rows: "
                f"{int(event_bg.shape[0])} vs {int(state.bg_point_h.shape[0])}."
            )
        bg_indices, signal_bg, support_bg, signal_aux = self._active_indices_and_signal(
            event=event_bg,
            valid=getattr(event, "valid_bg", None),
            support=getattr(event, "support_bg", None),
        )
        if not bool(self.bg_active_sparse):
            bg_indices = torch.arange(int(event_bg.shape[0]), device=event_bg.device, dtype=torch.long)
        idx_dev = bg_indices.to(device=event_bg.device, dtype=torch.long)
        n_active = int(idx_dev.numel())
        if n_active == 0:
            read_bg = event_bg.new_zeros((0, int(self.bg_mem_dim)), dtype=compute_dtype)
            seen_bg = event_bg.new_zeros((0, 1), dtype=compute_dtype)
            aux = {
                **_prefix(self.bg_point_ssm._empty_aux(state.bg_point_h), "vsm_bg_point"),
                **_prefix(self.bg_cell_ssm._empty_aux(state.bg_cell_h), "vsm_bg_cell"),
                "vsm_bg_active_rows": 0.0,
                "vsm_bg_write_gate_mean": 0.0,
                "vsm_bg_dt_mean": 0.0,
                "vsm_bg_h_norm": 0.0,
                "vsm_bg_seen_rows": float((state.bg_point_seen.detach() > 0).sum().item()),
                "vsm_bg_seen_ratio": float((state.bg_point_seen.detach() > 0).float().mean().item()) if state.bg_point_seen.numel() else 0.0,
                "vsm_bg_cell_seen_ratio": float((state.bg_cell_seen.detach() > 0).float().mean().item()) if state.bg_cell_seen.numel() else 0.0,
                "vsm_step_frame_idx": float(int(frame_idx)),
            }
            aux.update(_prefix(signal_aux, "vsm_bg"))
            return state, read_bg, seen_bg, bg_indices, aux

        event_active = event_bg.index_select(0, idx_dev).to(dtype=compute_dtype)
        view_active = _index_optional_rows(getattr(event, "view_code_bg", None), bg_indices)
        view_for_fusion = coerce_view_code(view_active, n=n_active, ref=event_active, view_dim=self.view_dim)
        valid_active = _index_optional_rows(getattr(event, "valid_bg", None), bg_indices)
        support_active_raw = _index_optional_rows(getattr(event, "support_bg", None), bg_indices)
        signal_active = signal_bg.index_select(0, idx_dev).to(dtype=compute_dtype)
        support_active = support_bg.index_select(0, idx_dev).to(dtype=compute_dtype)
        cell_ids = state.index.bg.point_cell_id.index_select(
            0,
            bg_indices.to(device=state.index.bg.point_cell_id.device, dtype=torch.long),
        ).to(device=event_bg.device, dtype=torch.long)
        valid_cell = cell_ids >= 0
        active_cell_mask = valid_cell & (signal_active.reshape(-1) > 0.0)

        out_state = state
        pre_cell_h = state.bg_cell_h
        pre_global_h = state.bg_global_h
        cell_aux: Dict[str, float] = self.bg_cell_ssm._empty_aux(state.bg_cell_h)
        if bool(active_cell_mask.any().item()):
            cell_rows_for_points = torch.nonzero(active_cell_mask, as_tuple=False).squeeze(1)
            cell_index_for_points = cell_ids.index_select(0, cell_rows_for_points)
            weights = signal_active.index_select(0, cell_rows_for_points) * torch.log1p(
                support_active.index_select(0, cell_rows_for_points).clamp_min(0.0)
            ).clamp_min(1.0e-6)
            cell_event_all = _scatter_weighted_mean(
                event_active.index_select(0, cell_rows_for_points),
                cell_index_for_points,
                weights,
                dim_size=state.index.bg.num_cells,
            )
            cell_support_all = _scatter_sum_1d(weights, cell_index_for_points, dim_size=state.index.bg.num_cells)[:, None]
            active_cell_ids = torch.unique(cell_index_for_points, sorted=True)
            out_cell_h, out_cell_seen, _cell_read, _cell_seen_rows, cell_aux = self._write_subset(
                self.bg_cell_ssm,
                h_full=out_state.bg_cell_h,
                seen_full=out_state.bg_cell_seen,
                rows=active_cell_ids,
                event=cell_event_all.index_select(0, active_cell_ids),
                view_code=None,
                valid=torch.ones((int(active_cell_ids.numel()), 1), device=event_bg.device, dtype=event_active.dtype),
                support=cell_support_all.index_select(0, active_cell_ids),
                step_idx=int(step_idx),
                repeat_idx=int(repeat_idx),
                branch_id=0,
                visit_time_code=visit_time_code,
                compute_dtype=compute_dtype,
            )
            out_state = replace(out_state, bg_cell_h=out_cell_h, bg_cell_seen=out_cell_seen)

        global_aux: Dict[str, float] = self.bg_global_ssm._empty_aux(state.bg_global_h)
        if bool(self.use_global_memory) and bool((signal_active.reshape(-1) > 0.0).any().item()):
            rows = torch.nonzero(signal_active.reshape(-1) > 0.0, as_tuple=False).squeeze(1)
            weights = signal_active.index_select(0, rows) * torch.log1p(support_active.index_select(0, rows).clamp_min(0.0)).clamp_min(1.0e-6)
            global_event = (event_active.index_select(0, rows) * weights).sum(dim=0, keepdim=True) / weights.sum().clamp(min=1.0e-6)
            global_support = weights.sum().reshape(1, 1)
            global_h, global_seen, _global_read, _global_seen_rows, global_aux = self._write_subset(
                self.bg_global_ssm,
                h_full=out_state.bg_global_h,
                seen_full=out_state.bg_global_seen,
                rows=torch.zeros((1,), device=event_bg.device, dtype=torch.long),
                event=global_event,
                view_code=None,
                valid=torch.ones((1, 1), device=event_bg.device, dtype=event_active.dtype),
                support=global_support,
                step_idx=int(step_idx),
                repeat_idx=int(repeat_idx),
                branch_id=0,
                visit_time_code=visit_time_code,
                compute_dtype=compute_dtype,
            )
            out_state = replace(out_state, bg_global_h=global_h, bg_global_seen=global_seen)

        def gather_cell_ctx(cell_h: torch.Tensor) -> torch.Tensor:
            ctx = event_active.new_zeros((n_active, int(self.bg_cell_mem_dim)))
            if not bool(valid_cell.any().item()) or int(cell_h.shape[0]) <= 0:
                return ctx
            rows_valid = torch.nonzero(valid_cell, as_tuple=False).squeeze(1)
            ctx[rows_valid] = cell_h.index_select(
                0,
                cell_ids.index_select(0, rows_valid).to(device=cell_h.device),
            ).to(device=event_bg.device, dtype=compute_dtype)
            return ctx

        def gather_global_ctx(global_h: torch.Tensor) -> torch.Tensor:
            if bool(self.use_global_memory):
                return global_h.to(device=event_bg.device, dtype=compute_dtype).expand(n_active, -1)
            return event_active.new_zeros((n_active, int(self.bg_global_mem_dim)))

        point_cell_h = pre_cell_h if self.bg_point_context_source == "previous_cell_global" else out_state.bg_cell_h
        point_global_h = pre_global_h if self.bg_point_context_source == "previous_cell_global" else out_state.bg_global_h
        cell_ctx_for_point = gather_cell_ctx(point_cell_h)
        global_ctx_for_point = gather_global_ctx(point_global_h)

        point_event = self.bg_point_event_proj(torch.cat([event_active, cell_ctx_for_point, global_ctx_for_point], dim=-1))
        rows_state = bg_indices.to(device=out_state.bg_point_h.device, dtype=torch.long)
        point_h, point_seen, point_read, point_seen_rows, point_aux = self._write_subset(
            self.bg_point_ssm,
            h_full=out_state.bg_point_h,
            seen_full=out_state.bg_point_seen,
            rows=rows_state,
            event=point_event,
            view_code=view_active,
            valid=valid_active,
            support=support_active_raw,
            step_idx=int(step_idx),
            repeat_idx=int(repeat_idx),
            branch_id=0,
            visit_time_code=self._select_visit_time_code(visit_time_code, idx=bg_indices, n_total=int(event_bg.shape[0])),
            compute_dtype=compute_dtype,
        )
        out_state = replace(out_state, bg_point_h=point_h, bg_point_seen=point_seen)

        read_cell_h = pre_cell_h if self.bg_final_read_context_source == "previous_cell_global" else out_state.bg_cell_h
        read_global_h = pre_global_h if self.bg_final_read_context_source == "previous_cell_global" else out_state.bg_global_h
        cell_ctx_for_read = gather_cell_ctx(read_cell_h)
        global_ctx_for_read = gather_global_ctx(read_global_h)
        point_part = self.bg_point_read_proj(point_read)
        cell_part = self.bg_cell_read_proj(cell_ctx_for_read)
        global_part = self.bg_global_read_proj(global_ctx_for_read)
        seen_feat = torch.log1p(point_seen_rows.to(device=event_bg.device, dtype=torch.float32).clamp_min(0.0)).to(dtype=compute_dtype)
        gate = torch.sigmoid(self.bg_read_gate(torch.cat([point_part, cell_part, global_part, view_for_fusion, seen_feat], dim=-1)))
        memory_part = 0.5 * (cell_part + global_part) if bool(self.use_global_memory) else cell_part
        read_bg = self.bg_read_norm(gate * point_part + (1.0 - gate) * memory_part).to(dtype=compute_dtype)
        aux = {
            **_prefix(point_aux, "vsm_bg_point"),
            **_prefix(cell_aux, "vsm_bg_cell"),
            **_prefix(global_aux, "vsm_bg_global"),
            **_prefix(signal_aux, "vsm_bg"),
            "vsm_bg_write_gate_mean": float(point_aux.get("write_gate_mean", 0.0)),
            "vsm_bg_dt_mean": float(point_aux.get("dt_mean", 0.0)),
            "vsm_bg_h_norm": float(point_aux.get("h_norm", 0.0)),
            "vsm_bg_seen_rows": float((out_state.bg_point_seen.detach() > 0).sum().item()),
            "vsm_bg_seen_ratio": float((out_state.bg_point_seen.detach() > 0).float().mean().item()) if out_state.bg_point_seen.numel() else 0.0,
            "vsm_bg_cell_seen_ratio": float((out_state.bg_cell_seen.detach() > 0).float().mean().item()) if out_state.bg_cell_seen.numel() else 0.0,
            "vsm_bg_num_cells": float(int(out_state.bg_cell_h.shape[0])),
            "vsm_bg_point_context_previous_cell_global": 1.0 if self.bg_point_context_source == "previous_cell_global" else 0.0,
            "vsm_bg_final_read_context_updated_cell_global": 1.0 if self.bg_final_read_context_source == "updated_cell_global" else 0.0,
            "vsm_step_frame_idx": float(int(frame_idx)),
        }
        return out_state, read_bg, point_seen_rows, bg_indices, aux

    def _rigid_context(
        self,
        *,
        state: LongCellVSMState,
        rigid_rows: torch.Tensor,
        ref: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(rigid_rows.numel())
        obj_ctx = ref.new_zeros((n, int(self.rigid_object_mem_dim)), dtype=dtype)
        cell_ctx = ref.new_zeros((n, int(self.rigid_cell_mem_dim)), dtype=dtype)
        idx = state.index.rigid
        if idx is None:
            return obj_ctx, cell_ctx
        if idx.row_to_object_pos is not None and state.rigid_object_h is not None and int(state.rigid_object_h.shape[0]) > 0:
            obj_pos = idx.row_to_object_pos.index_select(0, rigid_rows.to(device=idx.row_to_object_pos.device, dtype=torch.long)).to(device=ref.device)
            mask = obj_pos >= 0
            if bool(mask.any().item()):
                rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
                obj_ctx[rows] = state.rigid_object_h.index_select(
                    0,
                    obj_pos.index_select(0, rows).to(device=state.rigid_object_h.device, dtype=torch.long),
                ).to(device=ref.device, dtype=dtype)
        if idx.row_to_cell_pos is not None and state.rigid_cell_h is not None and int(state.rigid_cell_h.shape[0]) > 0:
            cell_pos = idx.row_to_cell_pos.index_select(0, rigid_rows.to(device=idx.row_to_cell_pos.device, dtype=torch.long)).to(device=ref.device)
            mask = cell_pos >= 0
            if bool(mask.any().item()):
                rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
                cell_ctx[rows] = state.rigid_cell_h.index_select(
                    0,
                    cell_pos.index_select(0, rows).to(device=state.rigid_cell_h.device, dtype=torch.long),
                ).to(device=ref.device, dtype=dtype)
        return obj_ctx, cell_ctx

    def _fuse_rigid_read(
        self,
        *,
        point_read: torch.Tensor,
        obj_ctx: torch.Tensor,
        cell_ctx: torch.Tensor,
        seen: torch.Tensor,
        view: torch.Tensor,
    ) -> torch.Tensor:
        point_part = self.rigid_point_read_proj(point_read)
        obj_part = self.rigid_object_read_proj(obj_ctx)
        cell_part = self.rigid_cell_read_proj(cell_ctx)
        seen_feat = torch.log1p(seen.to(device=point_read.device, dtype=torch.float32).clamp_min(0.0)).to(dtype=point_read.dtype)
        gate = torch.sigmoid(self.rigid_read_gate(torch.cat([point_part, obj_part, cell_part, view, seen_feat], dim=-1)))
        return self.rigid_read_norm(gate * point_part + (1.0 - gate) * 0.5 * (obj_part + cell_part)).to(dtype=point_read.dtype)

    def _write_read_rigid(
        self,
        *,
        state: LongCellVSMState,
        event: Any,
        step_idx: int,
        repeat_idx: int,
        rigid_meta: Optional[Dict[str, Any]],
        visit_time_code: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> tuple[LongCellVSMState, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        event_rigid = getattr(event, "event_rigid", None)
        route = getattr(event, "route", None)
        S = getattr(route, "S", None) if route is not None else None
        if event_rigid is None or S is None or int(event_rigid.shape[0]) == 0:
            return state, None, None, None, None, {
                "vsm_rigid_active_rows": 0.0,
                "vsm_rigid_stable_rows": 0.0,
                "vsm_rigid_unstable_rows": 0.0,
                "vsm_rigid_seen_ratio": 0.0,
            }
        if state.rigid_point_h is None or state.rigid_point_seen is None:
            raise ValueError("rigid events are present but LongCellVSMState has no rigid memory.")
        rigid_indices = S.to(device=event_rigid.device, dtype=torch.long).reshape(-1)
        if int(rigid_indices.numel()) != int(event_rigid.shape[0]):
            raise ValueError("event_rigid rows must match route.S.")
        n_rigid_total = int(state.index.rigid.row_to_stable_pos.shape[0]) if state.index.rigid is not None else 0
        stable_full = rigid_stable_mask_from_meta(rigid_meta, num_rows=n_rigid_total, device=event_rigid.device)
        stable_active = stable_full.index_select(0, rigid_indices)
        n_active = int(event_rigid.shape[0])
        event_in = event_rigid.to(dtype=compute_dtype)
        view_rigid_raw = getattr(event, "view_code_rigid", None)
        view_rigid = coerce_view_code(view_rigid_raw, n=n_active, ref=event_in, view_dim=self.view_dim)
        active_signal_idx, signal, support, _signal_aux = self._active_indices_and_signal(
            event=event_rigid,
            valid=getattr(event, "valid_rigid", None),
            support=getattr(event, "support_rigid", None),
        )
        _ = active_signal_idx
        out_state = state
        aux: Dict[str, float] = {
            "vsm_rigid_active_rows": float(n_active),
            "vsm_rigid_stable_rows": float(int(stable_active.sum().item())),
            "vsm_rigid_unstable_rows": float(int((~stable_active).sum().item())),
            "vsm_rigid_hard_valid_ratio": float(_signal_aux.get("hard_valid_ratio", 0.0)),
            "vsm_rigid_support_mean": float(_signal_aux.get("support_mean", 0.0)),
            "vsm_rigid_support_max": float(_signal_aux.get("support_max", 0.0)),
            "vsm_rigid_support_positive_ratio": float(_signal_aux.get("support_positive_ratio", 0.0)),
            "vsm_rigid_support_fallback_used": float(_signal_aux.get("support_fallback_used", 0.0)),
        }

        stable_event_rows = torch.nonzero(stable_active, as_tuple=False).squeeze(1)
        if int(stable_event_rows.numel()) > 0 and state.index.rigid is not None:
            stable_rigid_rows = rigid_indices.index_select(0, stable_event_rows)
            obj_pos = None
            if state.index.rigid.row_to_object_pos is not None:
                obj_pos = state.index.rigid.row_to_object_pos.index_select(
                    0,
                    stable_rigid_rows.to(device=state.index.rigid.row_to_object_pos.device, dtype=torch.long),
                ).to(device=event_rigid.device)
            cell_pos = None
            if state.index.rigid.row_to_cell_pos is not None:
                cell_pos = state.index.rigid.row_to_cell_pos.index_select(
                    0,
                    stable_rigid_rows.to(device=state.index.rigid.row_to_cell_pos.device, dtype=torch.long),
                ).to(device=event_rigid.device)
            stable_signal = signal.index_select(0, stable_event_rows).to(dtype=compute_dtype)
            stable_support = support.index_select(0, stable_event_rows).to(dtype=compute_dtype)
            write_mask = stable_signal.reshape(-1) > 0.0
            if obj_pos is not None and state.rigid_object_h is not None and state.rigid_object_seen is not None and bool(write_mask.any().item()):
                rows = torch.nonzero((obj_pos >= 0) & write_mask, as_tuple=False).squeeze(1)
                if int(rows.numel()) > 0:
                    scatter_index = obj_pos.index_select(0, rows)
                    weights = stable_signal.index_select(0, rows) * torch.log1p(stable_support.index_select(0, rows).clamp_min(0.0)).clamp_min(1.0e-6)
                    obj_event_all = _scatter_weighted_mean(
                        event_in.index_select(0, stable_event_rows.index_select(0, rows)),
                        scatter_index,
                        weights,
                        dim_size=int(state.rigid_object_h.shape[0]),
                    )
                    obj_support_all = _scatter_sum_1d(weights, scatter_index, dim_size=int(state.rigid_object_h.shape[0]))[:, None]
                    active_obj = torch.unique(scatter_index, sorted=True)
                    obj_h, obj_seen, _obj_read, _obj_seen_rows, obj_aux = self._write_subset(
                        self.rigid_object_ssm,
                        h_full=out_state.rigid_object_h,
                        seen_full=out_state.rigid_object_seen,
                        rows=active_obj,
                        event=obj_event_all.index_select(0, active_obj),
                        view_code=None,
                        valid=torch.ones((int(active_obj.numel()), 1), device=event_rigid.device, dtype=event_in.dtype),
                        support=obj_support_all.index_select(0, active_obj),
                        step_idx=int(step_idx),
                        repeat_idx=int(repeat_idx),
                        branch_id=2,
                        visit_time_code=visit_time_code,
                        compute_dtype=compute_dtype,
                    )
                    out_state = replace(out_state, rigid_object_h=obj_h, rigid_object_seen=obj_seen)
                    aux.update(_prefix(obj_aux, "vsm_rigid_object"))
            if cell_pos is not None and state.rigid_cell_h is not None and state.rigid_cell_seen is not None and bool(write_mask.any().item()):
                rows = torch.nonzero((cell_pos >= 0) & write_mask, as_tuple=False).squeeze(1)
                if int(rows.numel()) > 0:
                    scatter_index = cell_pos.index_select(0, rows)
                    weights = stable_signal.index_select(0, rows) * torch.log1p(stable_support.index_select(0, rows).clamp_min(0.0)).clamp_min(1.0e-6)
                    cell_event_all = _scatter_weighted_mean(
                        event_in.index_select(0, stable_event_rows.index_select(0, rows)),
                        scatter_index,
                        weights,
                        dim_size=int(state.rigid_cell_h.shape[0]),
                    )
                    cell_support_all = _scatter_sum_1d(weights, scatter_index, dim_size=int(state.rigid_cell_h.shape[0]))[:, None]
                    active_cell = torch.unique(scatter_index, sorted=True)
                    cell_h, cell_seen, _cell_read, _cell_seen_rows, cell_aux = self._write_subset(
                        self.rigid_cell_ssm,
                        h_full=out_state.rigid_cell_h,
                        seen_full=out_state.rigid_cell_seen,
                        rows=active_cell,
                        event=cell_event_all.index_select(0, active_cell),
                        view_code=None,
                        valid=torch.ones((int(active_cell.numel()), 1), device=event_rigid.device, dtype=event_in.dtype),
                        support=cell_support_all.index_select(0, active_cell),
                        step_idx=int(step_idx),
                        repeat_idx=int(repeat_idx),
                        branch_id=2,
                        visit_time_code=visit_time_code,
                        compute_dtype=compute_dtype,
                    )
                    out_state = replace(out_state, rigid_cell_h=cell_h, rigid_cell_seen=cell_seen)
                    aux.update(_prefix(cell_aux, "vsm_rigid_cell"))

        point_read_all = event_in.new_zeros((n_active, int(self.rigid_point_mem_dim)))
        seen_active = event_in.new_zeros((n_active, 1))
        if int(stable_event_rows.numel()) > 0 and state.index.rigid is not None:
            stable_rigid_rows = rigid_indices.index_select(0, stable_event_rows)
            stable_pos = state.index.rigid.row_to_stable_pos.index_select(
                0,
                stable_rigid_rows.to(device=state.index.rigid.row_to_stable_pos.device, dtype=torch.long),
            ).to(device=event_rigid.device)
            valid_stable_pos = stable_pos >= 0
            if bool(valid_stable_pos.any().item()):
                rows = torch.nonzero(valid_stable_pos, as_tuple=False).squeeze(1)
                event_rows = stable_event_rows.index_select(0, rows)
                rigid_rows = stable_rigid_rows.index_select(0, rows)
                obj_ctx, cell_ctx = self._rigid_context(state=out_state, rigid_rows=rigid_rows, ref=event_in, dtype=compute_dtype)
                point_event = self.rigid_point_event_proj(torch.cat([event_in.index_select(0, event_rows), obj_ctx, cell_ctx], dim=-1))
                point_h, point_seen, point_read, point_seen_rows, point_aux = self._write_subset(
                    self.rigid_point_ssm,
                    h_full=out_state.rigid_point_h,
                    seen_full=out_state.rigid_point_seen,
                    rows=stable_pos.index_select(0, rows),
                    event=point_event,
                    view_code=_index_optional_rows(view_rigid_raw, event_rows),
                    valid=_index_optional_rows(getattr(event, "valid_rigid", None), event_rows),
                    support=_index_optional_rows(getattr(event, "support_rigid", None), event_rows),
                    step_idx=int(step_idx),
                    repeat_idx=int(repeat_idx),
                    branch_id=2,
                    visit_time_code=self._select_visit_time_code(visit_time_code, idx=event_rows, n_total=n_active),
                    compute_dtype=compute_dtype,
                )
                out_state = replace(out_state, rigid_point_h=point_h, rigid_point_seen=point_seen)
                point_read_all[event_rows] = point_read
                seen_active[event_rows] = point_seen_rows
                aux.update(_prefix(point_aux, "vsm_rigid"))

        rows_unstable = torch.nonzero(~stable_active, as_tuple=False).squeeze(1)
        if int(rows_unstable.numel()) > 0:
            unstable_rigid_rows = rigid_indices.index_select(0, rows_unstable)
            obj_ctx, cell_ctx = self._rigid_context(state=out_state, rigid_rows=unstable_rigid_rows, ref=event_in, dtype=compute_dtype)
            point_event = self.rigid_point_event_proj(torch.cat([event_in.index_select(0, rows_unstable), obj_ctx, cell_ctx], dim=-1))
            h0 = event_in.new_zeros((int(rows_unstable.numel()), int(self.rigid_point_mem_dim)))
            seen0 = event_in.new_zeros((int(rows_unstable.numel()), 1))
            _h_tmp, seen_tmp, read_tmp, tmp_aux = self.rigid_point_ssm.write(
                h=h0,
                seen=seen0,
                event=point_event,
                view_code=_index_optional_rows(view_rigid_raw, rows_unstable),
                valid=_index_optional_rows(getattr(event, "valid_rigid", None), rows_unstable),
                support=_index_optional_rows(getattr(event, "support_rigid", None), rows_unstable),
                step_idx=int(step_idx),
                repeat_idx=int(repeat_idx),
                branch_id=2,
                visit_time_code=self._select_visit_time_code(visit_time_code, idx=rows_unstable, n_total=n_active),
            )
            point_read_all[rows_unstable] = read_tmp
            seen_active[rows_unstable] = seen_tmp
            aux.update(_prefix(tmp_aux, "vsm_rigid_unstable"))

        obj_ctx_all, cell_ctx_all = self._rigid_context(state=out_state, rigid_rows=rigid_indices, ref=event_in, dtype=compute_dtype)
        read_rigid = self._fuse_rigid_read(
            point_read=point_read_all,
            obj_ctx=obj_ctx_all,
            cell_ctx=cell_ctx_all,
            seen=seen_active,
            view=view_rigid,
        )
        aux["vsm_rigid_seen_ratio"] = (
            float((out_state.rigid_point_seen.detach() > 0).float().mean().item())
            if out_state.rigid_point_seen is not None and out_state.rigid_point_seen.numel()
            else 0.0
        )
        aux["vsm_rigid_seen_rows"] = (
            float((out_state.rigid_point_seen.detach() > 0).sum().item())
            if out_state.rigid_point_seen is not None and out_state.rigid_point_seen.numel()
            else float((seen_active.detach() > 0).sum().item())
        )
        aux.setdefault("vsm_rigid_write_gate_mean", 0.0)
        aux.setdefault("vsm_rigid_dt_mean", 0.0)
        aux.setdefault("vsm_rigid_h_norm", 0.0)
        return out_state, read_rigid, rigid_indices, seen_active, stable_active, aux

    def _write_read_distant(
        self,
        *,
        state: LongCellVSMState,
        event: Any,
        step_idx: int,
        repeat_idx: int,
        distant_mode: str,
        visit_time_code: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> tuple[LongCellVSMState, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        mode = _check_distant_mode(str(distant_mode))
        out_state = state
        read_distant = None
        distant_indices = None
        distant_seen_active = None
        event_distant = getattr(event, "event_distant", None)
        valid_distant = getattr(event, "valid_distant", None)
        support_distant = getattr(event, "support_distant", None)
        aux: Dict[str, float] = {
            "vsm_distant_mode_frozen_render_only": 1.0 if mode == DISTANT_MODE_FROZEN else 0.0,
            "vsm_distant_mode_appearance_scale_only": 1.0 if mode == DISTANT_MODE_APPEARANCE_SCALE else 0.0,
            "vsm_distant_event_rows": float(int(event_distant.shape[0])) if event_distant is not None else 0.0,
            "vsm_distant_event_norm": (
                float(event_distant.detach().float().norm(dim=-1).mean().item())
                if event_distant is not None and event_distant.numel()
                else 0.0
            ),
            "vsm_distant_valid_ratio": (
                float(valid_distant.detach().float().mean().item()) if valid_distant is not None and valid_distant.numel() else 0.0
            ),
            "vsm_distant_support_mean_all": (
                float(support_distant.detach().float().mean().item()) if support_distant is not None and support_distant.numel() else 0.0
            ),
            "vsm_distant_support_max": (
                float(support_distant.detach().float().max().item()) if support_distant is not None and support_distant.numel() else 0.0
            ),
            "vsm_distant_active_rows": 0.0,
            "vsm_distant_seen_ratio": (
                float((state.distant_seen.detach() > 0).float().mean().item())
                if state.distant_seen is not None and state.distant_seen.numel()
                else 0.0
            ),
        }
        if mode == DISTANT_MODE_FROZEN:
            aux.update(
                {
                    "vsm_distant_write_gate_mean": 0.0,
                    "vsm_distant_dt_mean": 0.0,
                    "vsm_distant_seen_rows": 0.0,
                    "vsm_distant_h_norm": 0.0,
                    "vsm_distant_support_max": (
                        float(support_distant.detach().float().max().item())
                        if support_distant is not None and support_distant.numel()
                        else 0.0
                    ),
                }
            )
            return out_state, read_distant, distant_indices, distant_seen_active, aux
        if event_distant is None or int(event_distant.shape[0]) == 0:
            return out_state, read_distant, distant_indices, distant_seen_active, aux
        if out_state.distant_h is None or out_state.distant_seen is None:
            raise ValueError("distant events are present but LongCellVSMState has no distant memory.")
        if int(event_distant.shape[0]) != int(out_state.distant_h.shape[0]):
            raise ValueError(
                "event_distant rows must match distant memory rows: "
                f"{int(event_distant.shape[0])} vs {int(out_state.distant_h.shape[0])}."
            )
        distant_indices, _signal, _support, _signal_aux = self._active_indices_and_signal(
            event=event_distant,
            valid=valid_distant,
            support=support_distant,
        )
        aux["vsm_distant_active_rows"] = float(int(distant_indices.numel()))
        if int(distant_indices.numel()) == 0:
            aux.update(_prefix(self.distant_ssm._empty_aux(out_state.distant_h), "vsm_distant"))
            return out_state, read_distant, distant_indices, distant_seen_active, aux
        view_distant = getattr(event, "view_code_distant", None)
        if view_distant is None:
            view_distant = getattr(event, "obs_code_distant", None)
        distant_h, distant_seen, read_distant, distant_seen_active, distant_aux = self._write_subset(
            self.distant_ssm,
            h_full=out_state.distant_h,
            seen_full=out_state.distant_seen,
            rows=distant_indices,
            event=event_distant.index_select(0, distant_indices.to(device=event_distant.device, dtype=torch.long)),
            view_code=_index_optional_rows(view_distant, distant_indices),
            valid=_index_optional_rows(valid_distant, distant_indices),
            support=_index_optional_rows(support_distant, distant_indices),
            step_idx=int(step_idx),
            repeat_idx=int(repeat_idx),
            branch_id=1,
            visit_time_code=self._select_visit_time_code(visit_time_code, idx=distant_indices, n_total=int(event_distant.shape[0])),
            compute_dtype=compute_dtype,
        )
        out_state = replace(out_state, distant_h=distant_h, distant_seen=distant_seen)
        aux.update(_prefix(distant_aux, "vsm_distant"))
        aux["vsm_distant_seen_ratio"] = (
            float((out_state.distant_seen.detach() > 0).float().mean().item())
            if out_state.distant_seen is not None and out_state.distant_seen.numel()
            else 0.0
        )
        return out_state, read_distant, distant_indices, distant_seen_active, aux

    def write_read(
        self,
        *,
        state: LongCellVSMState,
        event: Any,
        step_idx: int,
        frame_idx: int,
        repeat_idx: int,
        rigid_meta: Optional[Dict[str, Any]] = None,
        distant_mode: Optional[str] = None,
        visit_time_code: Optional[torch.Tensor] = None,
        compute_dtype: Optional[torch.dtype] = None,
        commit_memory: bool = True,
    ) -> tuple[LongCellVSMState, LongVSMReadPack, Dict[str, float]]:
        mode = _check_distant_mode(str(distant_mode or self.distant_mode))
        dtype = compute_dtype or event.event_bg.dtype
        out_state, read_bg, seen_bg, bg_indices, aux_bg = self._write_read_bg(
            state=state,
            event=event,
            step_idx=int(step_idx),
            frame_idx=int(frame_idx),
            repeat_idx=int(repeat_idx),
            visit_time_code=visit_time_code,
            compute_dtype=dtype,
        )
        out_state, read_rigid, rigid_indices, rigid_seen, stable_active, aux_rigid = self._write_read_rigid(
            state=out_state,
            event=event,
            step_idx=int(step_idx),
            repeat_idx=int(repeat_idx),
            rigid_meta=rigid_meta,
            visit_time_code=visit_time_code,
            compute_dtype=dtype,
        )
        out_state, read_distant, distant_indices, distant_seen, aux_distant = self._write_read_distant(
            state=out_state,
            event=event,
            step_idx=int(step_idx),
            repeat_idx=int(repeat_idx),
            distant_mode=mode,
            visit_time_code=visit_time_code,
            compute_dtype=dtype,
        )
        aux = {**aux_bg, **aux_rigid, **aux_distant}
        read = LongVSMReadPack(
            bg=read_bg,
            seen_bg=seen_bg,
            bg_indices=bg_indices,
            rigid=read_rigid,
            rigid_indices=rigid_indices,
            rigid_seen=rigid_seen,
            rigid_stable_mask=stable_active,
            distant=read_distant,
            distant_indices=distant_indices,
            distant_seen=distant_seen,
        )
        return (out_state if bool(commit_memory) else state), read, aux


__all__ = ["LongCellIndexPack", "LongCellStreamingVSM", "LongCellVSMState", "RigidObjectCellIndex"]
