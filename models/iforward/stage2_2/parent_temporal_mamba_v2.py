from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack

from models.iforward.mamba import StreamingMambaCell, StreamingMambaCellState
from .parent_temporal_keys_v2 import ParentTemporalKeysV2
from .parent_temporal_state_v2 import (
    ParentTemporalBranchStateV2,
    ParentTemporalDenseStateV2,
    ParentTemporalKeyedStateV2,
    ParentTemporalStateV2,
)
from .temporal_motion_embedding import TemporalMotionEmbedding


@dataclass
class ParentTemporalPreviewV2:
    event: EventPack
    aux: Dict[str, Any]


class _TemporalAdapterV2(nn.Module):
    def __init__(self, *, ctx_dim: int, event_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(ctx_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(event_dim)),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _empty_dense(cell: StreamingMambaCell, *, rows: int, device: torch.device, dtype: torch.dtype) -> ParentTemporalDenseStateV2:
    init = cell.init_state(int(rows), device=device, dtype=dtype)
    return ParentTemporalDenseStateV2(
        conv_state=init.conv_state,
        ssm_state=init.ssm_state,
        seen=init.seen,
        last_timestamp_sec=torch.full((int(rows),), -1.0, device=device, dtype=dtype),
    )


def _ensure_dense(
    cell: StreamingMambaCell,
    state: Optional[ParentTemporalDenseStateV2],
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ParentTemporalDenseStateV2:
    if state is None:
        return _empty_dense(cell, rows=int(rows), device=device, dtype=dtype)
    base = state.to(device=device, dtype=dtype)
    if int(base.seen.numel()) >= int(rows):
        return base
    extra = _empty_dense(cell, rows=int(rows) - int(base.seen.numel()), device=device, dtype=dtype)
    return ParentTemporalDenseStateV2(
        conv_state=torch.cat([base.conv_state, extra.conv_state], dim=0),
        ssm_state=torch.cat([base.ssm_state, extra.ssm_state], dim=0),
        seen=torch.cat([base.seen, extra.seen], dim=0),
        last_timestamp_sec=torch.cat([base.last_timestamp_sec, extra.last_timestamp_sec], dim=0),
    )


def _empty_keyed(cell: StreamingMambaCell, *, device: torch.device, dtype: torch.dtype) -> ParentTemporalKeyedStateV2:
    init = cell.init_state(0, device=device, dtype=dtype)
    return ParentTemporalKeyedStateV2(
        keys=torch.zeros((0,), device=device, dtype=torch.long),
        conv_state=init.conv_state,
        ssm_state=init.ssm_state,
        seen=init.seen,
        last_timestamp_sec=torch.zeros((0,), device=device, dtype=dtype),
    )


def _sort_keyed(state: ParentTemporalKeyedStateV2) -> ParentTemporalKeyedStateV2:
    if int(state.keys.numel()) <= 1:
        return state
    keys, order = torch.sort(state.keys.to(dtype=torch.long))
    return ParentTemporalKeyedStateV2(
        keys=keys,
        conv_state=state.conv_state[order],
        ssm_state=state.ssm_state[order],
        seen=state.seen[order],
        last_timestamp_sec=state.last_timestamp_sec[order],
    )


def _gather_keyed(
    cell: StreamingMambaCell,
    state: Optional[ParentTemporalKeyedStateV2],
    keys: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[StreamingMambaCellState, torch.Tensor]:
    gathered = cell.init_state(int(keys.numel()), device=device, dtype=dtype)
    timestamps = torch.full((int(keys.numel()),), -1.0, device=device, dtype=dtype)
    if state is None or int(keys.numel()) == 0 or int(state.keys.numel()) == 0:
        return gathered, timestamps
    base = state.to(device=device, dtype=dtype)
    query = keys.to(device=device, dtype=torch.long)
    pos = torch.searchsorted(base.keys, query)
    n_state = int(base.keys.numel())
    safe = pos.clamp(max=max(n_state - 1, 0))
    hit = (pos < n_state) & (base.keys[safe] == query)
    dst = torch.nonzero(hit, as_tuple=False).squeeze(1)
    if int(dst.numel()) == 0:
        return gathered, timestamps
    src = safe[dst]
    gathered.conv_state[dst] = base.conv_state[src]
    gathered.ssm_state[dst] = base.ssm_state[src]
    gathered.seen[dst] = base.seen[src]
    timestamps[dst] = base.last_timestamp_sec[src]
    return gathered, timestamps


def _scatter_keyed(
    cell: StreamingMambaCell,
    state: Optional[ParentTemporalKeyedStateV2],
    keys: torch.Tensor,
    updated: StreamingMambaCellState,
    timestamp_sec: torch.Tensor,
    *,
    write_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[ParentTemporalKeyedStateV2]:
    mask_rows = torch.nonzero(write_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(1)
    if int(mask_rows.numel()) == 0:
        return state
    base = state.to(device=device, dtype=dtype) if state is not None else _empty_keyed(cell, device=device, dtype=dtype)
    write_keys = keys.to(device=device, dtype=torch.long)[mask_rows]
    missing = write_keys
    if int(base.keys.numel()) > 0:
        pos0 = torch.searchsorted(base.keys, write_keys)
        n0 = int(base.keys.numel())
        safe0 = pos0.clamp(max=max(n0 - 1, 0))
        hit0 = (pos0 < n0) & (base.keys[safe0] == write_keys)
        missing = write_keys[~hit0]
    if int(missing.numel()) > 0:
        init = cell.init_state(int(missing.numel()), device=device, dtype=dtype)
        base = _sort_keyed(
            ParentTemporalKeyedStateV2(
                keys=torch.cat([base.keys, missing], dim=0),
                conv_state=torch.cat([base.conv_state, init.conv_state], dim=0),
                ssm_state=torch.cat([base.ssm_state, init.ssm_state], dim=0),
                seen=torch.cat([base.seen, init.seen], dim=0),
                last_timestamp_sec=torch.cat(
                    [
                        base.last_timestamp_sec,
                        torch.full((int(missing.numel()),), -1.0, device=device, dtype=dtype),
                    ],
                    dim=0,
                ),
            )
        )
    rows = torch.searchsorted(base.keys, write_keys)
    conv = base.conv_state.clone()
    ssm = base.ssm_state.clone()
    seen = base.seen.clone()
    last_ts = base.last_timestamp_sec.clone()
    conv[rows] = updated.conv_state[mask_rows]
    ssm[rows] = updated.ssm_state[mask_rows]
    seen[rows] = updated.seen[mask_rows]
    last_ts[rows] = timestamp_sec.to(device=device, dtype=dtype)[mask_rows]
    return ParentTemporalKeyedStateV2(keys=base.keys, conv_state=conv, ssm_state=ssm, seen=seen, last_timestamp_sec=last_ts)


def _weighted_aggregate(x: torch.Tensor, keys: torch.Tensor, support: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(keys.to(dtype=torch.long), sorted=True, return_inverse=True)
    if support is None:
        weights = torch.ones((int(x.shape[0]),), device=x.device, dtype=x.dtype)
    else:
        weights = support.to(device=x.device, dtype=x.dtype).reshape(int(x.shape[0]), -1).mean(dim=-1).clamp_min(0.0)
    denom = x.new_zeros((int(unique.numel()),))
    denom.index_add_(0, inverse, weights)
    out = x.new_zeros((int(unique.numel()), int(x.shape[-1])))
    out.index_add_(0, inverse, x * weights[:, None])
    counts = torch.bincount(inverse, minlength=int(unique.numel())).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    mean = x.new_zeros((int(unique.numel()), int(x.shape[-1])))
    mean.index_add_(0, inverse, x)
    mean = mean / counts[:, None]
    out = torch.where(denom[:, None] > 0, out / denom.clamp_min(1.0e-8)[:, None], mean)
    return unique, inverse, out


class ParentTemporalMemoryV2(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 64,
        ctx_dim: int = 32,
        model_dim: int = 32,
        state_dim: int = 8,
        conv_kernel: int = 2,
        adapter_hidden_dim: int = 64,
        motion_embed_dim: int = 16,
        dense_bg: bool = True,
        dense_distant: bool = True,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.motion_embed_dim = int(motion_embed_dim)
        self.dense_bg = bool(dense_bg)
        self.dense_distant = bool(dense_distant)
        input_dim = int(event_dim) + int(motion_embed_dim)
        self.motion_embedding = TemporalMotionEmbedding(output_dim=int(motion_embed_dim))
        self.cells = nn.ModuleDict(
            {
                branch: StreamingMambaCell(
                    input_dim=input_dim,
                    model_dim=int(model_dim),
                    state_dim=int(state_dim),
                    conv_kernel=int(conv_kernel),
                    output_dim=int(ctx_dim),
                )
                for branch in ("bg", "distant", "rigid")
            }
        )
        self.adapters = nn.ModuleDict(
            {
                branch: _TemporalAdapterV2(ctx_dim=int(ctx_dim), event_dim=int(event_dim), hidden_dim=int(adapter_hidden_dim))
                for branch in ("bg", "distant", "rigid")
            }
        )
        self.branch_gate_raw = nn.Parameter(torch.zeros(3))

    def _motion(
        self,
        x: torch.Tensor,
        *,
        motion_meta: Optional[Dict[str, Any]],
        seen: Optional[torch.Tensor],
    ) -> torch.Tensor:
        meta = dict(motion_meta or {})
        return self.motion_embedding(
            num_rows=int(x.shape[0]),
            ref=x,
            delta_t_sec=meta.get("delta_t_sec", 0.0),
            gap=meta.get("gap", meta.get("frame_gap", 0.0)),
            ego_delta_translation=meta.get("ego_delta_translation", None),
            ego_delta_yaw=meta.get("ego_delta_yaw", 0.0),
            seen_flag=seen,
            visit_kind=meta.get("visit_kind", None),
        )

    def _with_parent_delta(
        self,
        *,
        x: torch.Tensor,
        motion_meta: Optional[Dict[str, Any]],
        timestamp_sec: Optional[float],
        last_timestamp_sec: torch.Tensor,
        seen: torch.Tensor,
    ) -> Dict[str, Any]:
        meta = dict(motion_meta or {})
        if str(meta.get("visit_kind", "")) in {"repair", "refinement"}:
            meta["delta_t_sec"] = x.new_zeros((int(x.shape[0]),))
            meta["frame_gap"] = x.new_zeros((int(x.shape[0]),))
            meta["ego_delta_translation"] = x.new_zeros((int(x.shape[0]), 3))
            meta["ego_delta_yaw"] = x.new_zeros((int(x.shape[0]),))
            return meta
        if timestamp_sec is not None:
            ts = torch.full((int(x.shape[0]),), float(timestamp_sec), device=x.device, dtype=x.dtype)
            last = last_timestamp_sec.to(device=x.device, dtype=x.dtype).reshape(-1)
            if int(last.numel()) == 1:
                last = last.expand(int(x.shape[0]))
            dt = (ts - last).clamp_min(0.0)
            dt = torch.where(seen.to(device=x.device, dtype=torch.bool), dt, torch.zeros_like(dt))
            meta["delta_t_sec"] = dt
        return meta

    def _preview_dense(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        state: Optional[ParentTemporalBranchStateV2],
        motion_meta: Optional[Dict[str, Any]],
        timestamp_sec: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell = self.cells[str(branch)]
        base = _ensure_dense(cell, None if state is None else state.dense, rows=int(x.shape[0]), device=x.device, dtype=x.dtype)
        seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        last_ts = base.last_timestamp_sec[: int(x.shape[0])].to(device=x.device, dtype=x.dtype)
        row_meta = self._with_parent_delta(
            x=x,
            motion_meta=motion_meta,
            timestamp_sec=timestamp_sec,
            last_timestamp_sec=last_ts,
            seen=seen,
        )
        token = torch.cat([x, self._motion(x, motion_meta=row_meta, seen=seen)], dim=-1)
        cell_state = StreamingMambaCellState(
            conv_state=base.conv_state[: int(x.shape[0])],
            ssm_state=base.ssm_state[: int(x.shape[0])],
            seen=seen,
        )
        out, _ = cell(token, cell_state, write_mask=torch.zeros((int(x.shape[0]),), device=x.device, dtype=torch.bool))
        out = torch.where(seen[:, None], out, torch.zeros_like(out))
        return out, seen

    def _preview_keyed(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        keys: torch.Tensor,
        state: Optional[ParentTemporalBranchStateV2],
        motion_meta: Optional[Dict[str, Any]],
        support: Optional[torch.Tensor],
        timestamp_sec: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(x.shape[0]) == 0:
            return x.new_zeros((0, self.ctx_dim)), x.new_zeros((0,), dtype=torch.bool)
        cell = self.cells[str(branch)]
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        cell_state, timestamps = _gather_keyed(cell, None if state is None else state.keyed, keys_u, device=x.device, dtype=x.dtype)
        seen_u = cell_state.seen.to(device=x.device, dtype=torch.bool)
        row_meta = self._with_parent_delta(
            x=x_u,
            motion_meta=motion_meta,
            timestamp_sec=timestamp_sec,
            last_timestamp_sec=timestamps,
            seen=seen_u,
        )
        token = torch.cat([x_u, self._motion(x_u, motion_meta=row_meta, seen=seen_u)], dim=-1)
        out_u, _ = cell(token, cell_state, write_mask=torch.zeros((int(keys_u.numel()),), device=x.device, dtype=torch.bool))
        out_u = torch.where(seen_u[:, None], out_u, torch.zeros_like(out_u))
        return out_u.index_select(0, inverse), seen_u.index_select(0, inverse)

    def _branch_preview(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: Optional[ParentTemporalBranchStateV2],
        dense: bool,
        motion_meta: Optional[Dict[str, Any]],
        support: Optional[torch.Tensor],
        timestamp_sec: Optional[float],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x is None:
            return None, None
        if dense:
            return self._preview_dense(branch=branch, x=x, state=branch_state, motion_meta=motion_meta, timestamp_sec=timestamp_sec)
        if keys is None:
            raise ValueError(f"ParentTemporalMemoryV2 {branch} preview requires keys")
        return self._preview_keyed(
            branch=branch,
            x=x,
            keys=keys,
            state=branch_state,
            motion_meta=motion_meta,
            support=support,
            timestamp_sec=timestamp_sec,
        )

    def _fuse(self, *, branch: str, event: Optional[torch.Tensor], ctx: Optional[torch.Tensor], seen: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if event is None:
            return None
        if ctx is None or seen is None:
            return event
        idx = {"bg": 0, "distant": 1, "rigid": 2}[str(branch)]
        gate = torch.sigmoid(self.branch_gate_raw[idx]).to(device=event.device, dtype=event.dtype)
        contribution = gate * self.adapters[str(branch)](ctx)
        contribution = torch.where(seen.to(device=event.device, dtype=torch.bool)[:, None], contribution, torch.zeros_like(contribution))
        out = event + contribution
        if not torch.isfinite(out).all():
            raise RuntimeError(f"ParentTemporalMemoryV2 fused {branch} event contains NaN/Inf")
        return out

    def preview(
        self,
        *,
        event: EventPack,
        state: Optional[ParentTemporalStateV2],
        keys: ParentTemporalKeysV2,
        timestamp_sec: Optional[float] = None,
        motion_meta: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> ParentTemporalPreviewV2:
        state = state if state is not None else ParentTemporalStateV2.empty()
        meta = dict(motion_meta or {})
        ctx_bg, seen_bg = self._branch_preview(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            dense=self.dense_bg,
            motion_meta=meta,
            support=event.support_bg,
            timestamp_sec=timestamp_sec,
        )
        ctx_distant, seen_distant = self._branch_preview(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            dense=self.dense_distant,
            motion_meta=meta,
            support=event.support_distant,
            timestamp_sec=timestamp_sec,
        )
        ctx_rigid, seen_rigid = self._branch_preview(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            dense=False,
            motion_meta=meta,
            support=event.support_rigid,
            timestamp_sec=timestamp_sec,
        )
        fused = EventPack(
            event_bg=self._fuse(branch="bg", event=event.event_bg, ctx=ctx_bg, seen=seen_bg),
            event_distant=self._fuse(branch="distant", event=event.event_distant, ctx=ctx_distant, seen=seen_distant),
            event_rigid=self._fuse(branch="rigid", event=event.event_rigid, ctx=ctx_rigid, seen=seen_rigid),
            support_bg=event.support_bg,
            support_distant=event.support_distant,
            support_rigid=event.support_rigid,
            valid_bg=event.valid_bg,
            valid_distant=event.valid_distant,
            valid_rigid=event.valid_rigid,
            view_code_bg=event.view_code_bg,
            obs_code_bg=None,
            obs_code_distant=None,
            obs_code_rigid=None,
            acc_w_bg=event.acc_w_bg,
            route=event.route,
            branch_slices=dict(event.branch_slices or {}),
            aux=dict(event.aux or {}),
        )
        aux = dict(event.aux or {})
        for name, seen in (("bg", seen_bg), ("distant", seen_distant), ("rigid", seen_rigid)):
            if seen is not None:
                aux[f"iforward/parent_temporal_v2/{name}_preview_seen_ratio"] = (
                    float(seen.detach().float().mean().item()) if int(seen.numel()) else 0.0
                )
        fused.aux = aux
        return ParentTemporalPreviewV2(event=fused, aux=aux)

    def _valid_write_mask(self, x: Optional[torch.Tensor], valid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if valid is None:
            return torch.ones((int(x.shape[0]),), device=x.device, dtype=torch.bool)
        mask = valid.reshape(int(x.shape[0]), -1).to(device=x.device, dtype=torch.bool).any(dim=-1)
        return mask

    def _commit_dense(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        branch_state: ParentTemporalBranchStateV2,
        valid: Optional[torch.Tensor],
        timestamp_sec: float,
        motion_meta: Optional[Dict[str, Any]],
    ) -> Tuple[ParentTemporalBranchStateV2, Dict[str, float]]:
        if x is None:
            return branch_state, {f"iforward/parent_temporal_v2/{branch}_committed": 0.0}
        write = self._valid_write_mask(x, valid)
        assert write is not None
        cell = self.cells[str(branch)]
        base = _ensure_dense(cell, branch_state.dense, rows=int(x.shape[0]), device=x.device, dtype=x.dtype)
        old_seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        row_meta = self._with_parent_delta(
            x=x,
            motion_meta=motion_meta,
            timestamp_sec=float(timestamp_sec),
            last_timestamp_sec=base.last_timestamp_sec[: int(x.shape[0])],
            seen=old_seen,
        )
        token = torch.cat([x, self._motion(x, motion_meta=row_meta, seen=old_seen)], dim=-1)
        cell_state = StreamingMambaCellState(
            conv_state=base.conv_state[: int(x.shape[0])],
            ssm_state=base.ssm_state[: int(x.shape[0])],
            seen=old_seen,
        )
        _, updated = cell(token, cell_state, write_mask=write)
        conv = base.conv_state.clone()
        ssm = base.ssm_state.clone()
        seen = base.seen.clone()
        last_ts = base.last_timestamp_sec.clone()
        n = int(x.shape[0])
        conv[:n] = updated.conv_state
        ssm[:n] = updated.ssm_state
        seen[:n] = updated.seen
        last_ts[:n] = torch.where(write, torch.full((n,), float(timestamp_sec), device=x.device, dtype=x.dtype), last_ts[:n])
        dense = ParentTemporalDenseStateV2(conv_state=conv, ssm_state=ssm, seen=seen, last_timestamp_sec=last_ts)
        count = float(write.detach().float().sum().item())
        return ParentTemporalBranchStateV2(dense=dense, keyed=branch_state.keyed), {f"iforward/parent_temporal_v2/{branch}_committed": count}

    def _commit_keyed(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: ParentTemporalBranchStateV2,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        timestamp_sec: float,
        motion_meta: Optional[Dict[str, Any]],
    ) -> Tuple[ParentTemporalBranchStateV2, Dict[str, float]]:
        if x is None:
            return branch_state, {f"iforward/parent_temporal_v2/{branch}_committed": 0.0}
        if keys is None:
            raise ValueError(f"ParentTemporalMemoryV2 {branch} commit requires keys")
        write = self._valid_write_mask(x, valid)
        assert write is not None
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        write_counts = x.new_zeros((int(keys_u.numel()),))
        write_counts.index_add_(0, inverse, write.to(dtype=x.dtype))
        write_u = write_counts > 0
        cell = self.cells[str(branch)]
        cell_state, timestamps = _gather_keyed(cell, branch_state.keyed, keys_u, device=x.device, dtype=x.dtype)
        old_seen = cell_state.seen.to(device=x.device, dtype=torch.bool)
        row_meta = self._with_parent_delta(
            x=x_u,
            motion_meta=motion_meta,
            timestamp_sec=float(timestamp_sec),
            last_timestamp_sec=timestamps,
            seen=old_seen,
        )
        token = torch.cat([x_u, self._motion(x_u, motion_meta=row_meta, seen=old_seen)], dim=-1)
        _, updated = cell(token, cell_state, write_mask=write_u)
        ts = torch.full((int(keys_u.numel()),), float(timestamp_sec), device=x.device, dtype=x.dtype)
        keyed = _scatter_keyed(
            cell,
            branch_state.keyed,
            keys_u,
            updated,
            ts,
            write_mask=write_u,
            device=x.device,
            dtype=x.dtype,
        )
        count = float(write.detach().float().sum().item())
        return ParentTemporalBranchStateV2(dense=branch_state.dense, keyed=keyed), {f"iforward/parent_temporal_v2/{branch}_committed": count}

    def commit(
        self,
        *,
        event: EventPack,
        state: Optional[ParentTemporalStateV2],
        keys: ParentTemporalKeysV2,
        block_id: int = -1,
        timestamp_sec: Optional[float] = None,
        physical_time_advance: bool = True,
        motion_meta: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Tuple[ParentTemporalStateV2, Dict[str, float]]:
        state = state if state is not None else ParentTemporalStateV2.empty()
        if not bool(physical_time_advance):
            return state, {"iforward/parent_temporal_v2/commit_skipped_no_time_advance": 1.0}
        ts = float(timestamp_sec if timestamp_sec is not None else max(float(state.last_timestamp_sec), 0.0))
        meta = dict(motion_meta or {})
        bg, aux_bg = self._commit_dense(
            branch="bg",
            x=event.event_bg,
            branch_state=state.bg,
            valid=event.valid_bg,
            timestamp_sec=ts,
            motion_meta=meta,
        ) if self.dense_bg else self._commit_keyed(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            valid=event.valid_bg,
            support=event.support_bg,
            timestamp_sec=ts,
            motion_meta=meta,
        )
        distant, aux_distant = self._commit_dense(
            branch="distant",
            x=event.event_distant,
            branch_state=state.distant,
            valid=event.valid_distant,
            timestamp_sec=ts,
            motion_meta=meta,
        ) if self.dense_distant else self._commit_keyed(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            valid=event.valid_distant,
            support=event.support_distant,
            timestamp_sec=ts,
            motion_meta=meta,
        )
        rigid, aux_rigid = self._commit_keyed(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            valid=event.valid_rigid,
            support=event.support_rigid,
            timestamp_sec=ts,
            motion_meta=meta,
        )
        out = ParentTemporalStateV2(
            bg=bg,
            distant=distant,
            rigid=rigid,
            last_committed_block_id=int(block_id),
            last_timestamp_sec=float(ts),
        )
        aux = {
            **aux_bg,
            **aux_distant,
            **aux_rigid,
            "iforward/parent_temporal_v2/raw_frame_commit": 1.0,
            "iforward/parent_temporal_v2/last_timestamp_sec": float(ts),
        }
        return out, aux


__all__ = ["ParentTemporalMemoryV2", "ParentTemporalPreviewV2"]
