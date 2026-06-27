from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack

from models.iforward.mamba import StreamingMambaCell, StreamingMambaCellState
from models.iforward.stage2_2.parent_temporal_keys_v2 import ParentTemporalKeysV2

from .optimizer_memory_schema import (
    DenseOptimizerState,
    KeyedOptimizerState,
    OptimizerBranchState,
    ParentOptimizerMambaState,
)
from .optimizer_visit_embedding import OptimizerVisitEmbedding, VISIT_KIND_TO_ID, VisitMeta
from .optimizer_write_token import OptimizerWriteTokenBuilder


@dataclass
class ParentOptimizerPreview:
    event: EventPack
    aux: Dict[str, Any]


def _empty_dense(cell: StreamingMambaCell, *, rows: int, device: torch.device, dtype: torch.dtype) -> DenseOptimizerState:
    init = cell.init_state(int(rows), device=device, dtype=dtype)
    return DenseOptimizerState(
        conv_state=init.conv_state,
        ssm_state=init.ssm_state,
        seen=init.seen,
        update_count=torch.zeros((int(rows),), device=device, dtype=torch.long),
        last_visit_step=torch.full((int(rows),), -1, device=device, dtype=torch.long),
        last_frame_id=torch.full((int(rows),), -1, device=device, dtype=torch.long),
        last_visit_kind=torch.full((int(rows),), -1, device=device, dtype=torch.long),
    )


def _ensure_dense(
    cell: StreamingMambaCell,
    state: Optional[DenseOptimizerState],
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DenseOptimizerState:
    if state is None:
        return _empty_dense(cell, rows=int(rows), device=device, dtype=dtype)
    base = state.to(device=device, dtype=dtype)
    if int(base.seen.numel()) >= int(rows):
        return base
    extra = _empty_dense(cell, rows=int(rows) - int(base.seen.numel()), device=device, dtype=dtype)
    return DenseOptimizerState(
        conv_state=torch.cat([base.conv_state, extra.conv_state], dim=0),
        ssm_state=torch.cat([base.ssm_state, extra.ssm_state], dim=0),
        seen=torch.cat([base.seen, extra.seen], dim=0),
        update_count=torch.cat([base.update_count, extra.update_count], dim=0),
        last_visit_step=torch.cat([base.last_visit_step, extra.last_visit_step], dim=0),
        last_frame_id=torch.cat([base.last_frame_id, extra.last_frame_id], dim=0),
        last_visit_kind=torch.cat([base.last_visit_kind, extra.last_visit_kind], dim=0),
    )


def _empty_keyed(cell: StreamingMambaCell, *, device: torch.device, dtype: torch.dtype) -> KeyedOptimizerState:
    init = cell.init_state(0, device=device, dtype=dtype)
    return KeyedOptimizerState(
        keys=torch.zeros((0,), device=device, dtype=torch.long),
        conv_state=init.conv_state,
        ssm_state=init.ssm_state,
        seen=init.seen,
        update_count=torch.zeros((0,), device=device, dtype=torch.long),
        last_visit_step=torch.zeros((0,), device=device, dtype=torch.long),
        last_frame_id=torch.zeros((0,), device=device, dtype=torch.long),
        last_visit_kind=torch.zeros((0,), device=device, dtype=torch.long),
    )


def _sort_keyed(state: KeyedOptimizerState) -> KeyedOptimizerState:
    if int(state.keys.numel()) <= 1:
        return state
    keys, order = torch.sort(state.keys.to(dtype=torch.long))
    return KeyedOptimizerState(
        keys=keys,
        conv_state=state.conv_state[order],
        ssm_state=state.ssm_state[order],
        seen=state.seen[order],
        update_count=state.update_count[order],
        last_visit_step=state.last_visit_step[order],
        last_frame_id=state.last_frame_id[order],
        last_visit_kind=state.last_visit_kind[order],
    )


def _gather_keyed(
    cell: StreamingMambaCell,
    state: Optional[KeyedOptimizerState],
    keys: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[StreamingMambaCellState, Dict[str, torch.Tensor]]:
    gathered = cell.init_state(int(keys.numel()), device=device, dtype=dtype)
    meta = {
        "update_count": torch.zeros((int(keys.numel()),), device=device, dtype=torch.long),
        "last_visit_step": torch.full((int(keys.numel()),), -1, device=device, dtype=torch.long),
        "last_frame_id": torch.full((int(keys.numel()),), -1, device=device, dtype=torch.long),
        "last_visit_kind": torch.full((int(keys.numel()),), -1, device=device, dtype=torch.long),
    }
    if state is None or int(keys.numel()) == 0 or int(state.keys.numel()) == 0:
        return gathered, meta
    base = state.to(device=device, dtype=dtype)
    query = keys.to(device=device, dtype=torch.long)
    pos = torch.searchsorted(base.keys, query)
    n_state = int(base.keys.numel())
    safe = pos.clamp(max=max(n_state - 1, 0))
    hit = (pos < n_state) & (base.keys[safe] == query)
    dst = torch.nonzero(hit, as_tuple=False).squeeze(1)
    if int(dst.numel()) == 0:
        return gathered, meta
    src = safe[dst]
    gathered.conv_state[dst] = base.conv_state[src]
    gathered.ssm_state[dst] = base.ssm_state[src]
    gathered.seen[dst] = base.seen[src]
    meta["update_count"][dst] = base.update_count[src]
    meta["last_visit_step"][dst] = base.last_visit_step[src]
    meta["last_frame_id"][dst] = base.last_frame_id[src]
    meta["last_visit_kind"][dst] = base.last_visit_kind[src]
    return gathered, meta


def _scatter_keyed(
    cell: StreamingMambaCell,
    state: Optional[KeyedOptimizerState],
    keys: torch.Tensor,
    updated: StreamingMambaCellState,
    meta: Dict[str, torch.Tensor],
    *,
    write_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[KeyedOptimizerState]:
    rows_mask = torch.nonzero(write_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(1)
    if int(rows_mask.numel()) == 0:
        return state
    base = state.to(device=device, dtype=dtype) if state is not None else _empty_keyed(cell, device=device, dtype=dtype)
    write_keys = keys.to(device=device, dtype=torch.long)[rows_mask]
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
            KeyedOptimizerState(
                keys=torch.cat([base.keys, missing], dim=0),
                conv_state=torch.cat([base.conv_state, init.conv_state], dim=0),
                ssm_state=torch.cat([base.ssm_state, init.ssm_state], dim=0),
                seen=torch.cat([base.seen, init.seen], dim=0),
                update_count=torch.cat([base.update_count, torch.zeros((int(missing.numel()),), device=device, dtype=torch.long)], dim=0),
                last_visit_step=torch.cat([base.last_visit_step, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
                last_frame_id=torch.cat([base.last_frame_id, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
                last_visit_kind=torch.cat([base.last_visit_kind, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
            )
        )
    rows = torch.searchsorted(base.keys, write_keys)
    conv = base.conv_state.clone()
    ssm = base.ssm_state.clone()
    seen = base.seen.clone()
    update_count = base.update_count.clone()
    last_visit_step = base.last_visit_step.clone()
    last_frame_id = base.last_frame_id.clone()
    last_visit_kind = base.last_visit_kind.clone()
    conv[rows] = updated.conv_state[rows_mask]
    ssm[rows] = updated.ssm_state[rows_mask]
    seen[rows] = updated.seen[rows_mask]
    update_count[rows] = meta["update_count"][rows_mask]
    last_visit_step[rows] = meta["last_visit_step"][rows_mask]
    last_frame_id[rows] = meta["last_frame_id"][rows_mask]
    last_visit_kind[rows] = meta["last_visit_kind"][rows_mask]
    return KeyedOptimizerState(
        keys=base.keys,
        conv_state=conv,
        ssm_state=ssm,
        seen=seen,
        update_count=update_count,
        last_visit_step=last_visit_step,
        last_frame_id=last_frame_id,
        last_visit_kind=last_visit_kind,
    )


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


class _OptimizerAdapter(nn.Module):
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


class ParentOptimizerMamba(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 64,
        ctx_dim: int = 32,
        model_dim: int = 32,
        state_dim: int = 8,
        conv_kernel: int = 2,
        adapter_hidden_dim: int = 64,
        visit_dim: int = 32,
        support_min: float = 0.001,
        dense_bg: bool = True,
        dense_distant: bool = True,
        gate_init: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.visit_dim = int(visit_dim)
        self.support_min = float(support_min)
        self.dense_bg = bool(dense_bg)
        self.dense_distant = bool(dense_distant)
        input_dim = int(event_dim) + int(visit_dim)
        self.visit_embedding = OptimizerVisitEmbedding(output_dim=int(visit_dim))
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
                branch: _OptimizerAdapter(ctx_dim=int(ctx_dim), event_dim=int(event_dim), hidden_dim=int(adapter_hidden_dim))
                for branch in ("bg", "distant", "rigid")
            }
        )
        self.write_builder = OptimizerWriteTokenBuilder(
            event_dim=int(event_dim),
            visit_dim=int(visit_dim),
            token_dim=int(event_dim),
            hidden_dim=int(adapter_hidden_dim),
        )
        self.branch_gate_raw = nn.Parameter(torch.zeros(3))
        self.visit_gate_raw = nn.Parameter(torch.zeros(4))
        init = dict(gate_init or {})
        for kind, value in init.items():
            if str(kind) in VISIT_KIND_TO_ID:
                p = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
                with torch.no_grad():
                    self.visit_gate_raw[int(VISIT_KIND_TO_ID[str(kind)])] = torch.logit(torch.tensor(p))

    def _visit(self, meta: Optional[VisitMeta | Dict[str, object] | object], *, ref: torch.Tensor, rows: int) -> torch.Tensor:
        return self.visit_embedding(meta, ref=ref, rows=int(rows))

    def _token(self, x: torch.Tensor, visit: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, visit.to(device=x.device, dtype=x.dtype)], dim=-1)

    def _preview_dense(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        state: Optional[OptimizerBranchState],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cell = self.cells[str(branch)]
        base = _ensure_dense(cell, None if state is None else state.dense, rows=int(x.shape[0]), device=x.device, dtype=x.dtype)
        seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        visit = self._visit(visit_meta, ref=x, rows=int(x.shape[0]))
        cell_state = StreamingMambaCellState(
            conv_state=base.conv_state[: int(x.shape[0])],
            ssm_state=base.ssm_state[: int(x.shape[0])],
            seen=seen,
        )
        out, _ = cell(self._token(x, visit), cell_state, write_mask=torch.zeros((int(x.shape[0]),), device=x.device, dtype=torch.bool))
        out = torch.where(seen[:, None], out, torch.zeros_like(out))
        return out, seen, visit

    def _preview_keyed(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        keys: torch.Tensor,
        state: Optional[OptimizerBranchState],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(x.shape[0]) == 0:
            return x.new_zeros((0, self.ctx_dim)), x.new_zeros((0,), dtype=torch.bool), x.new_zeros((0, self.visit_dim))
        cell = self.cells[str(branch)]
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        cell_state, _ = _gather_keyed(cell, None if state is None else state.keyed, keys_u, device=x.device, dtype=x.dtype)
        seen_u = cell_state.seen.to(device=x.device, dtype=torch.bool)
        visit_u = self._visit(visit_meta, ref=x_u, rows=int(keys_u.numel()))
        out_u, _ = cell(self._token(x_u, visit_u), cell_state, write_mask=torch.zeros((int(keys_u.numel()),), device=x.device, dtype=torch.bool))
        out_u = torch.where(seen_u[:, None], out_u, torch.zeros_like(out_u))
        return out_u.index_select(0, inverse), seen_u.index_select(0, inverse), visit_u.index_select(0, inverse)

    def _branch_preview(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: Optional[OptimizerBranchState],
        dense: bool,
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x is None:
            return None, None, None
        if dense:
            return self._preview_dense(branch=branch, x=x, state=branch_state, visit_meta=visit_meta)
        if keys is None:
            raise ValueError(f"ParentOptimizerMamba {branch} preview requires keys")
        return self._preview_keyed(
            branch=branch,
            x=x,
            keys=keys,
            state=branch_state,
            support=support,
            visit_meta=visit_meta,
        )

    def _fuse(
        self,
        *,
        branch: str,
        event: Optional[torch.Tensor],
        ctx: Optional[torch.Tensor],
        seen: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Optional[torch.Tensor]:
        if event is None or ctx is None or seen is None:
            return event
        idx = {"bg": 0, "distant": 1, "rigid": 2}[str(branch)]
        kind = "bootstrap" if visit_meta is None else str(getattr(visit_meta, "visit_kind", "bootstrap"))
        if not isinstance(visit_meta, VisitMeta):
            try:
                meta_obj = VisitMeta.from_mapping(visit_meta or {}) if isinstance(visit_meta, dict) else VisitMeta.from_step(visit_meta)
                kind = str(meta_obj.visit_kind)
            except Exception:
                kind = "bootstrap"
        kind_id = VISIT_KIND_TO_ID.get(kind, 0)
        branch_gate = torch.sigmoid(self.branch_gate_raw[idx]).to(device=event.device, dtype=event.dtype)
        visit_gate = torch.sigmoid(self.visit_gate_raw[kind_id]).to(device=event.device, dtype=event.dtype)
        if support is None:
            support_gate = event.new_ones((int(event.shape[0]), 1))
        else:
            support_mean = support.to(device=event.device, dtype=event.dtype).reshape(int(event.shape[0]), -1).mean(dim=-1, keepdim=True)
            support_gate = (support_mean / (support_mean + 1.0)).clamp(0.0, 1.0)
        contribution = branch_gate * visit_gate * support_gate * self.adapters[str(branch)](ctx)
        contribution = torch.where(seen.to(device=event.device, dtype=torch.bool)[:, None], contribution, torch.zeros_like(contribution))
        out = event + contribution
        if not torch.isfinite(out).all():
            raise RuntimeError(f"ParentOptimizerMamba fused {branch} event contains NaN/Inf")
        return out

    def preview(
        self,
        *,
        event: EventPack,
        state: Optional[ParentOptimizerMambaState],
        keys: ParentTemporalKeysV2,
        visit_meta: Optional[VisitMeta | Dict[str, object] | object] = None,
        **_: Any,
    ) -> ParentOptimizerPreview:
        state = state if state is not None else ParentOptimizerMambaState.empty()
        ctx_bg, seen_bg, visit_bg = self._branch_preview(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            dense=self.dense_bg,
            support=event.support_bg,
            visit_meta=visit_meta,
        )
        ctx_distant, seen_distant, visit_distant = self._branch_preview(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            dense=self.dense_distant,
            support=event.support_distant,
            visit_meta=visit_meta,
        )
        ctx_rigid, seen_rigid, visit_rigid = self._branch_preview(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            dense=False,
            support=event.support_rigid,
            visit_meta=visit_meta,
        )
        fused = EventPack(
            event_bg=self._fuse(branch="bg", event=event.event_bg, ctx=ctx_bg, seen=seen_bg, support=event.support_bg, visit_meta=visit_meta),
            event_distant=self._fuse(
                branch="distant",
                event=event.event_distant,
                ctx=ctx_distant,
                seen=seen_distant,
                support=event.support_distant,
                visit_meta=visit_meta,
            ),
            event_rigid=self._fuse(
                branch="rigid",
                event=event.event_rigid,
                ctx=ctx_rigid,
                seen=seen_rigid,
                support=event.support_rigid,
                visit_meta=visit_meta,
            ),
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
                aux[f"iforward/parent_optimizer_mamba/{name}_preview_seen_ratio"] = (
                    float(seen.detach().float().mean().item()) if int(seen.numel()) else 0.0
                )
        aux["iforward/parent_optimizer_mamba/read"] = 1.0
        fused.aux = aux
        return ParentOptimizerPreview(event=fused, aux=aux)

    def _valid_write_mask(self, x: Optional[torch.Tensor], valid: Optional[torch.Tensor], support: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        rows = int(x.shape[0])
        if valid is None:
            mask = torch.ones((rows,), device=x.device, dtype=torch.bool)
        else:
            mask = valid.reshape(rows, -1).to(device=x.device, dtype=torch.bool).any(dim=-1)
        if support is not None and float(self.support_min) > 0.0:
            support_mean = support.to(device=x.device, dtype=x.dtype).reshape(rows, -1).mean(dim=-1)
            mask = mask & (support_mean >= float(self.support_min))
        return mask

    @staticmethod
    def _meta_values(visit_meta: Optional[VisitMeta | Dict[str, object] | object], *, device: torch.device) -> Dict[str, int]:
        if isinstance(visit_meta, VisitMeta):
            meta = visit_meta
        elif isinstance(visit_meta, dict):
            meta = VisitMeta.from_mapping(visit_meta)
        elif visit_meta is not None:
            meta = VisitMeta.from_step(visit_meta)
        else:
            meta = VisitMeta.from_mapping({})
        return {
            "visit_step": int(meta.global_update_idx_in_episode),
            "frame_id": int(meta.frame_id),
            "kind": int(VISIT_KIND_TO_ID.get(str(meta.visit_kind), 0)),
        }

    def _write_dense(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        branch_state: OptimizerBranchState,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[OptimizerBranchState, Dict[str, float]]:
        if x is None:
            return branch_state, {f"iforward/parent_optimizer_mamba/{branch}_written": 0.0}
        write = self._valid_write_mask(x, valid, support)
        assert write is not None
        cell = self.cells[str(branch)]
        base = _ensure_dense(cell, branch_state.dense, rows=int(x.shape[0]), device=x.device, dtype=x.dtype)
        old_seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        visit = self._visit(visit_meta, ref=x, rows=int(x.shape[0]))
        cell_state = StreamingMambaCellState(
            conv_state=base.conv_state[: int(x.shape[0])],
            ssm_state=base.ssm_state[: int(x.shape[0])],
            seen=old_seen,
        )
        _, updated = cell(self._token(x, visit), cell_state, write_mask=write)
        n = int(x.shape[0])
        conv = base.conv_state.clone()
        ssm = base.ssm_state.clone()
        seen = base.seen.clone()
        update_count = base.update_count.clone()
        last_visit_step = base.last_visit_step.clone()
        last_frame_id = base.last_frame_id.clone()
        last_visit_kind = base.last_visit_kind.clone()
        conv[:n] = updated.conv_state
        ssm[:n] = updated.ssm_state
        seen[:n] = updated.seen
        meta = self._meta_values(visit_meta, device=x.device)
        update_count[:n] = torch.where(write, update_count[:n] + 1, update_count[:n])
        last_visit_step[:n] = torch.where(write, torch.full((n,), int(meta["visit_step"]), device=x.device, dtype=torch.long), last_visit_step[:n])
        last_frame_id[:n] = torch.where(write, torch.full((n,), int(meta["frame_id"]), device=x.device, dtype=torch.long), last_frame_id[:n])
        last_visit_kind[:n] = torch.where(write, torch.full((n,), int(meta["kind"]), device=x.device, dtype=torch.long), last_visit_kind[:n])
        dense = DenseOptimizerState(
            conv_state=conv,
            ssm_state=ssm,
            seen=seen,
            update_count=update_count,
            last_visit_step=last_visit_step,
            last_frame_id=last_frame_id,
            last_visit_kind=last_visit_kind,
        )
        return OptimizerBranchState(dense=dense, keyed=branch_state.keyed), {
            f"iforward/parent_optimizer_mamba/{branch}_written": float(write.detach().float().sum().item())
        }

    def _write_keyed(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: OptimizerBranchState,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[OptimizerBranchState, Dict[str, float]]:
        if x is None:
            return branch_state, {f"iforward/parent_optimizer_mamba/{branch}_written": 0.0}
        if keys is None:
            raise ValueError(f"ParentOptimizerMamba {branch} write requires keys")
        write = self._valid_write_mask(x, valid, support)
        assert write is not None
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        write_counts = x.new_zeros((int(keys_u.numel()),))
        write_counts.index_add_(0, inverse, write.to(dtype=x.dtype))
        write_u = write_counts > 0
        cell = self.cells[str(branch)]
        cell_state, meta_state = _gather_keyed(cell, branch_state.keyed, keys_u, device=x.device, dtype=x.dtype)
        visit_u = self._visit(visit_meta, ref=x_u, rows=int(keys_u.numel()))
        _, updated = cell(self._token(x_u, visit_u), cell_state, write_mask=write_u)
        meta_vals = self._meta_values(visit_meta, device=x.device)
        meta_state = {str(k): v.clone() for k, v in meta_state.items()}
        meta_state["update_count"] = torch.where(write_u, meta_state["update_count"] + 1, meta_state["update_count"])
        meta_state["last_visit_step"] = torch.where(
            write_u,
            torch.full((int(keys_u.numel()),), int(meta_vals["visit_step"]), device=x.device, dtype=torch.long),
            meta_state["last_visit_step"],
        )
        meta_state["last_frame_id"] = torch.where(
            write_u,
            torch.full((int(keys_u.numel()),), int(meta_vals["frame_id"]), device=x.device, dtype=torch.long),
            meta_state["last_frame_id"],
        )
        meta_state["last_visit_kind"] = torch.where(
            write_u,
            torch.full((int(keys_u.numel()),), int(meta_vals["kind"]), device=x.device, dtype=torch.long),
            meta_state["last_visit_kind"],
        )
        keyed = _scatter_keyed(
            cell,
            branch_state.keyed,
            keys_u,
            updated,
            meta_state,
            write_mask=write_u,
            device=x.device,
            dtype=x.dtype,
        )
        return OptimizerBranchState(dense=branch_state.dense, keyed=keyed), {
            f"iforward/parent_optimizer_mamba/{branch}_written": float(write.detach().float().sum().item())
        }

    def write(
        self,
        *,
        spatial_event: EventPack,
        fused_event: Optional[EventPack] = None,
        write_event: Optional[EventPack] = None,
        state: Optional[ParentOptimizerMambaState],
        keys: ParentTemporalKeysV2,
        visit_meta: Optional[VisitMeta | Dict[str, object] | object] = None,
        delta: Optional[object] = None,
        **_: Any,
    ) -> Tuple[ParentOptimizerMambaState, Dict[str, float]]:
        state = state if state is not None else ParentOptimizerMambaState.empty()
        fused = fused_event if fused_event is not None else spatial_event
        if write_event is None:
            write_event = self.write_builder(
                spatial_event=spatial_event,
                fused_event=fused,
                visit_bg=self._visit(visit_meta, ref=spatial_event.event_bg, rows=int(spatial_event.event_bg.shape[0])),
                visit_distant=(
                    None
                    if spatial_event.event_distant is None
                    else self._visit(visit_meta, ref=spatial_event.event_distant, rows=int(spatial_event.event_distant.shape[0]))
                ),
                visit_rigid=(
                    None
                    if spatial_event.event_rigid is None
                    else self._visit(visit_meta, ref=spatial_event.event_rigid, rows=int(spatial_event.event_rigid.shape[0]))
                ),
                delta=delta,
            )
        bg, aux_bg = (
            self._write_dense(
                branch="bg",
                x=write_event.event_bg,
                branch_state=state.bg,
                valid=write_event.valid_bg,
                support=write_event.support_bg,
                visit_meta=visit_meta,
            )
            if self.dense_bg
            else self._write_keyed(
                branch="bg",
                x=write_event.event_bg,
                keys=keys.bg,
                branch_state=state.bg,
                valid=write_event.valid_bg,
                support=write_event.support_bg,
                visit_meta=visit_meta,
            )
        )
        distant, aux_distant = (
            self._write_dense(
                branch="distant",
                x=write_event.event_distant,
                branch_state=state.distant,
                valid=write_event.valid_distant,
                support=write_event.support_distant,
                visit_meta=visit_meta,
            )
            if self.dense_distant
            else self._write_keyed(
                branch="distant",
                x=write_event.event_distant,
                keys=keys.distant,
                branch_state=state.distant,
                valid=write_event.valid_distant,
                support=write_event.support_distant,
                visit_meta=visit_meta,
            )
        )
        rigid, aux_rigid = self._write_keyed(
            branch="rigid",
            x=write_event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            valid=write_event.valid_rigid,
            support=write_event.support_rigid,
            visit_meta=visit_meta,
        )
        next_state = ParentOptimizerMambaState(
            bg=bg,
            distant=distant,
            rigid=rigid,
            global_update_step=int(state.global_update_step) + 1,
        )
        aux = {
            **aux_bg,
            **aux_distant,
            **aux_rigid,
            "iforward/parent_optimizer_mamba/write": 1.0,
            "iforward/parent_optimizer_mamba/global_update_step": float(next_state.global_update_step),
        }
        return next_state, aux


__all__ = ["ParentOptimizerMamba", "ParentOptimizerPreview"]
