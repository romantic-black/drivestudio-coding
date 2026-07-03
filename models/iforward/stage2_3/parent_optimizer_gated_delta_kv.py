from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack

from models.iforward.stage2_2.parent_temporal_keys_v2 import ParentTemporalKeysV2

from .optimizer_memory_schema import (
    DeltaKVOptimizerBranchState,
    DenseDeltaKVOptimizerState,
    KeyedDeltaKVOptimizerState,
    ParentOptimizerDeltaKVState,
)
from .optimizer_visit_embedding import OptimizerVisitEmbedding, VISIT_KIND_TO_ID, VisitMeta
from .optimizer_write_token import OptimizerWriteTokenBuilder
from .parent_optimizer_mamba import ParentOptimizerPreview


def rms(x: torch.Tensor, dim=-1, keepdim: bool = True, eps: float = 1.0e-6) -> torch.Tensor:
    if int(x.numel()) == 0:
        return torch.sqrt(torch.mean(x.float() * x.float(), dim=dim, keepdim=keepdim) + eps).to(dtype=x.dtype)
    return torch.sqrt(torch.mean(x.float() * x.float(), dim=dim, keepdim=keepdim) + eps).to(dtype=x.dtype)


def rms_unit(x: torch.Tensor, dim=-1, eps: float = 1.0e-6) -> torch.Tensor:
    return x / rms(x, dim=dim, keepdim=True, eps=eps)


def rms_clamp(x: torch.Tensor, max_rms: float, dims=(-1,), eps: float = 1.0e-6) -> torch.Tensor:
    if float(max_rms) <= 0.0 or int(x.numel()) == 0:
        return x
    r = torch.sqrt(torch.mean(x.float() * x.float(), dim=dims, keepdim=True) + eps).to(dtype=x.dtype)
    scale = torch.clamp(float(max_rms) / r.clamp_min(eps), max=1.0)
    return x * scale


def _rms_rows(x: torch.Tensor, dims=(-1,)) -> torch.Tensor:
    if int(x.numel()) == 0:
        return x.new_zeros((0,))
    return torch.sqrt(torch.mean(x.float() * x.float(), dim=dims) + 1.0e-6).to(device=x.device)


def _stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
    if int(values.numel()) == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    detached = values.detach().float()
    return {
        f"{prefix}_mean": float(detached.mean().item()),
        f"{prefix}_max": float(detached.max().item()),
    }


GDKV_COMPUTE_DTYPE = torch.float32


def _resolve_state_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value in {torch.float32, torch.float16, torch.bfloat16}:
            return value
        raise ValueError(f"unsupported GDKV state dtype={value!r}")
    name = str(value).strip().lower()
    if name in {"fp32", "float32", "32", "none", "off", "false"}:
        return torch.float32
    if name in {"fp16", "float16", "half", "16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported GDKV state dtype={name!r}")


def _dtype_id(dtype: torch.dtype) -> float:
    if dtype is torch.float16:
        return 1.0
    if dtype is torch.bfloat16:
        return 2.0
    return 0.0


@dataclass
class DeltaKVCellState:
    kv_state: torch.Tensor
    seen: torch.Tensor


class LowRankGatedDeltaKVCell(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        token_dim: int,
        key_dim: int = 16,
        value_dim: int = 32,
        hidden_dim: int = 64,
        value_rms_max: float = 2.0,
        ctx_rms_max: float = 4.0,
        state_rms_max: float = 4.0,
        erase_gate_max: float = 1.0,
        write_gate_max: float = 1.0,
        erase_bias: float = 0.0,
        write_bias: float = 0.0,
        decay_bias: float = 0.0,
        decay_min: Optional[Dict[str, float] | float] = None,
        query_rms_unit: bool = True,
        key_rms_unit: bool = True,
        state_dtype: torch.dtype | str = torch.float32,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.token_dim = int(token_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.value_rms_max = float(value_rms_max)
        self.ctx_rms_max = float(ctx_rms_max)
        self.state_rms_max = float(state_rms_max)
        self.erase_gate_max = float(erase_gate_max)
        self.write_gate_max = float(write_gate_max)
        self.erase_bias = float(erase_bias)
        self.write_bias = float(write_bias)
        self.decay_bias = float(decay_bias)
        self.query_rms_unit = bool(query_rms_unit)
        self.key_rms_unit = bool(key_rms_unit)
        self.state_dtype = _resolve_state_dtype(state_dtype)
        self.state_dtype_id = _dtype_id(self.state_dtype)
        if hasattr(decay_min, "items"):
            self.decay_min_by_kind = {str(k): float(v) for k, v in dict(decay_min).items()}
        elif decay_min is None:
            self.decay_min_by_kind = {
                "bootstrap": 1.0,
                "assimilate": 0.98,
                "assimilation": 0.98,
                "repair": 1.0,
                "repeat_stability": 1.0,
                "stress": 1.0,
            }
        else:
            self.decay_min_by_kind = {"default": float(decay_min)}

        self.q_proj = nn.Linear(int(event_dim), self.key_dim)
        self.key_proj = nn.Linear(int(token_dim), self.key_dim)
        self.value_proj = nn.Linear(int(token_dim), self.value_dim)
        self.erase_proj = nn.Linear(int(token_dim), self.key_dim)
        self.write_proj = nn.Linear(int(token_dim), self.value_dim)
        self.decay_proj = nn.Linear(int(token_dim), self.key_dim)

        for module in (self.q_proj, self.key_proj, self.value_proj, self.erase_proj, self.write_proj, self.decay_proj):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def init_state(self, rows: int, *, device: torch.device, dtype: torch.dtype) -> DeltaKVCellState:
        return DeltaKVCellState(
            kv_state=torch.zeros((int(rows), self.key_dim, self.value_dim), device=device, dtype=dtype),
            seen=torch.zeros((int(rows),), device=device, dtype=torch.bool),
        )

    def read(self, event: torch.Tensor, state: DeltaKVCellState) -> Tuple[torch.Tensor, Dict[str, float]]:
        q_raw = self.q_proj(event).float()
        q = (rms_unit(q_raw, dim=-1) if self.query_rms_unit else q_raw) / math.sqrt(float(self.key_dim))
        kv_state = state.kv_state.to(device=event.device, dtype=GDKV_COMPUTE_DTYPE)
        ctx = torch.einsum("nkv,nk->nv", kv_state, q)
        ctx = rms_clamp(ctx, self.ctx_rms_max, dims=(-1,)).float()
        seen = state.seen.to(device=event.device, dtype=torch.bool)
        ctx = torch.where(seen[:, None], ctx, torch.zeros_like(ctx))
        if not torch.isfinite(ctx).all():
            raise RuntimeError("LowRankGatedDeltaKVCell read produced NaN/Inf")
        aux = {"state_dtype_id": float(self.state_dtype_id)}
        aux.update(_stats("ctx_rms", _rms_rows(ctx, dims=(-1,))))
        aux.update(_stats("query_rms", _rms_rows(q, dims=(-1,))))
        return ctx, aux

    @staticmethod
    def _kind(visit_meta: Optional[VisitMeta | Dict[str, object] | object]) -> str:
        if isinstance(visit_meta, VisitMeta):
            return str(visit_meta.visit_kind)
        if isinstance(visit_meta, dict):
            return str(VisitMeta.from_mapping(visit_meta).visit_kind)
        if visit_meta is not None:
            return str(VisitMeta.from_step(visit_meta).visit_kind)
        return "bootstrap"

    def _decay_min(self, visit_meta: Optional[VisitMeta | Dict[str, object] | object]) -> float:
        kind = self._kind(visit_meta)
        return float(self.decay_min_by_kind.get(kind, self.decay_min_by_kind.get("default", 0.98)))

    def write(
        self,
        token: torch.Tensor,
        state: DeltaKVCellState,
        write_mask: torch.Tensor,
        *,
        visit_meta: Optional[VisitMeta | Dict[str, object] | object] = None,
    ) -> Tuple[DeltaKVCellState, Dict[str, float]]:
        if int(token.shape[0]) == 0:
            return DeltaKVCellState(
                kv_state=state.kv_state.to(device=token.device, dtype=self.state_dtype),
                seen=state.seen.to(device=token.device, dtype=torch.bool),
            ), {
                "state_rms_mean": 0.0,
                "state_rms_max": 0.0,
                "state_dtype_id": float(self.state_dtype_id),
                "key_rms_mean": 0.0,
                "key_rms_max": 0.0,
                "value_rms_mean": 0.0,
                "value_rms_max": 0.0,
                "erase_gate_mean": 0.0,
                "erase_gate_max": 0.0,
                "write_gate_mean": 0.0,
                "write_gate_max": 0.0,
                "decay_mean": 0.0,
                "decay_max": 0.0,
            }
        s_old = state.kv_state.to(device=token.device, dtype=GDKV_COMPUTE_DTYPE)
        seen_old = state.seen.to(device=token.device, dtype=torch.bool)
        k_raw = self.key_proj(token).float()
        k = (rms_unit(k_raw, dim=-1) if self.key_rms_unit else k_raw) / math.sqrt(float(self.key_dim))
        v = rms_clamp(self.value_proj(token).float(), self.value_rms_max, dims=(-1,)).float()
        erase = torch.sigmoid(self.erase_proj(token).float() + float(self.erase_bias)) * float(self.erase_gate_max)
        write = torch.sigmoid(self.write_proj(token).float() + float(self.write_bias)) * float(self.write_gate_max)
        decay_min = self._decay_min(visit_meta)
        decay = float(decay_min) + (1.0 - float(decay_min)) * torch.sigmoid(
            self.decay_proj(token).float() + float(self.decay_bias)
        )
        s_decay = s_old * decay[:, :, None]
        old = torch.einsum("nkv,nk->nv", s_decay, erase * k)
        s_erased = s_decay - torch.einsum("nk,nv->nkv", k, old)
        s_write = s_erased + torch.einsum("nk,nv->nkv", k, write * v)
        s_new = rms_clamp(s_write, self.state_rms_max, dims=(-2, -1)).float()
        mask = write_mask.to(device=token.device, dtype=torch.bool)
        kv_state = torch.where(mask[:, None, None], s_new, s_old)
        seen = seen_old | mask
        if not torch.isfinite(kv_state).all():
            raise RuntimeError("LowRankGatedDeltaKVCell write produced NaN/Inf")
        aux = {"state_dtype_id": float(self.state_dtype_id)}
        aux.update(_stats("state_rms", _rms_rows(kv_state, dims=(-2, -1))))
        aux.update(_stats("key_rms", _rms_rows(k, dims=(-1,))))
        aux.update(_stats("value_rms", _rms_rows(v, dims=(-1,))))
        aux.update(_stats("erase_gate", erase.detach().float().mean(dim=-1)))
        aux.update(_stats("write_gate", write.detach().float().mean(dim=-1)))
        aux.update(_stats("decay", decay.detach().float().mean(dim=-1)))
        return DeltaKVCellState(kv_state=kv_state.to(dtype=self.state_dtype), seen=seen), aux


def _empty_delta_dense(
    cell: LowRankGatedDeltaKVCell,
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DenseDeltaKVOptimizerState:
    init = cell.init_state(int(rows), device=device, dtype=dtype)
    return DenseDeltaKVOptimizerState(
        kv_state=init.kv_state,
        seen=init.seen,
        update_count=torch.zeros((int(rows),), device=device, dtype=torch.long),
        last_visit_step=torch.full((int(rows),), -1, device=device, dtype=torch.long),
        last_frame_id=torch.full((int(rows),), -1, device=device, dtype=torch.long),
        last_visit_kind=torch.full((int(rows),), -1, device=device, dtype=torch.long),
    )


def _ensure_delta_dense(
    cell: LowRankGatedDeltaKVCell,
    state: Optional[DenseDeltaKVOptimizerState],
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DenseDeltaKVOptimizerState:
    if state is None:
        return _empty_delta_dense(cell, rows=int(rows), device=device, dtype=dtype)
    base = state.to(device=device, dtype=dtype)
    if int(base.seen.numel()) >= int(rows):
        return base
    extra = _empty_delta_dense(cell, rows=int(rows) - int(base.seen.numel()), device=device, dtype=dtype)
    return DenseDeltaKVOptimizerState(
        kv_state=torch.cat([base.kv_state, extra.kv_state], dim=0),
        seen=torch.cat([base.seen, extra.seen], dim=0),
        update_count=torch.cat([base.update_count, extra.update_count], dim=0),
        last_visit_step=torch.cat([base.last_visit_step, extra.last_visit_step], dim=0),
        last_frame_id=torch.cat([base.last_frame_id, extra.last_frame_id], dim=0),
        last_visit_kind=torch.cat([base.last_visit_kind, extra.last_visit_kind], dim=0),
    )


def _empty_delta_keyed(
    cell: LowRankGatedDeltaKVCell,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> KeyedDeltaKVOptimizerState:
    init = cell.init_state(0, device=device, dtype=dtype)
    return KeyedDeltaKVOptimizerState(
        keys=torch.zeros((0,), device=device, dtype=torch.long),
        kv_state=init.kv_state,
        seen=init.seen,
        update_count=torch.zeros((0,), device=device, dtype=torch.long),
        last_visit_step=torch.zeros((0,), device=device, dtype=torch.long),
        last_frame_id=torch.zeros((0,), device=device, dtype=torch.long),
        last_visit_kind=torch.zeros((0,), device=device, dtype=torch.long),
    )


def _sort_delta_keyed(state: KeyedDeltaKVOptimizerState) -> KeyedDeltaKVOptimizerState:
    if int(state.keys.numel()) <= 1:
        return state
    keys, order = torch.sort(state.keys.to(dtype=torch.long))
    return KeyedDeltaKVOptimizerState(
        keys=keys,
        kv_state=state.kv_state[order],
        seen=state.seen[order],
        update_count=state.update_count[order],
        last_visit_step=state.last_visit_step[order],
        last_frame_id=state.last_frame_id[order],
        last_visit_kind=state.last_visit_kind[order],
    )


def _gather_delta_keyed(
    cell: LowRankGatedDeltaKVCell,
    state: Optional[KeyedDeltaKVOptimizerState],
    keys: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[DeltaKVCellState, Dict[str, torch.Tensor]]:
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
    gathered.kv_state[dst] = base.kv_state[src]
    gathered.seen[dst] = base.seen[src]
    meta["update_count"][dst] = base.update_count[src]
    meta["last_visit_step"][dst] = base.last_visit_step[src]
    meta["last_frame_id"][dst] = base.last_frame_id[src]
    meta["last_visit_kind"][dst] = base.last_visit_kind[src]
    return gathered, meta


def _scatter_delta_keyed(
    cell: LowRankGatedDeltaKVCell,
    state: Optional[KeyedDeltaKVOptimizerState],
    keys: torch.Tensor,
    updated: DeltaKVCellState,
    meta: Dict[str, torch.Tensor],
    *,
    write_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[KeyedDeltaKVOptimizerState]:
    rows_mask = torch.nonzero(write_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(1)
    if int(rows_mask.numel()) == 0:
        return state
    base = state.to(device=device, dtype=dtype) if state is not None else _empty_delta_keyed(cell, device=device, dtype=dtype)
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
        base = _sort_delta_keyed(
            KeyedDeltaKVOptimizerState(
                keys=torch.cat([base.keys, missing], dim=0),
                kv_state=torch.cat([base.kv_state, init.kv_state], dim=0),
                seen=torch.cat([base.seen, init.seen], dim=0),
                update_count=torch.cat([base.update_count, torch.zeros((int(missing.numel()),), device=device, dtype=torch.long)], dim=0),
                last_visit_step=torch.cat([base.last_visit_step, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
                last_frame_id=torch.cat([base.last_frame_id, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
                last_visit_kind=torch.cat([base.last_visit_kind, torch.full((int(missing.numel()),), -1, device=device, dtype=torch.long)], dim=0),
            )
        )
    rows = torch.searchsorted(base.keys, write_keys)
    kv_state = base.kv_state.clone()
    seen = base.seen.clone()
    update_count = base.update_count.clone()
    last_visit_step = base.last_visit_step.clone()
    last_frame_id = base.last_frame_id.clone()
    last_visit_kind = base.last_visit_kind.clone()
    kv_state[rows] = updated.kv_state[rows_mask]
    seen[rows] = updated.seen[rows_mask]
    update_count[rows] = meta["update_count"][rows_mask]
    last_visit_step[rows] = meta["last_visit_step"][rows_mask]
    last_frame_id[rows] = meta["last_frame_id"][rows_mask]
    last_visit_kind[rows] = meta["last_visit_kind"][rows_mask]
    return KeyedDeltaKVOptimizerState(
        keys=base.keys,
        kv_state=kv_state,
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


class ParentOptimizerGatedDeltaKV(nn.Module):
    state_cls = ParentOptimizerDeltaKVState

    def __init__(
        self,
        *,
        event_dim: int = 64,
        ctx_dim: int = 32,
        token_dim: int = 64,
        key_dim: int = 16,
        value_dim: int = 32,
        adapter_hidden_dim: int = 64,
        visit_dim: int = 32,
        support_min: float = 0.001,
        dense_bg: bool = True,
        dense_distant: bool = True,
        gate_init: Optional[Dict[str, float]] = None,
        value_rms_max: float = 2.0,
        ctx_rms_max: float = 4.0,
        state_rms_max: float = 4.0,
        erase_gate_max: float = 1.0,
        write_gate_max: float = 1.0,
        erase_bias: float = 0.0,
        write_bias: float = 0.0,
        decay_bias: float = 0.0,
        decay_min: Optional[Dict[str, float] | float] = None,
        query_rms_unit: bool = True,
        key_rms_unit: bool = True,
        include_spatial_event: bool = True,
        include_parent_event: bool = True,
        include_delta_summary: bool = True,
        include_visit_embedding: bool = True,
        state_dtype: torch.dtype | str = torch.float32,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.visit_dim = int(visit_dim)
        self.support_min = float(support_min)
        self.dense_bg = bool(dense_bg)
        self.dense_distant = bool(dense_distant)
        self.state_dtype = _resolve_state_dtype(state_dtype)
        self.state_dtype_id = _dtype_id(self.state_dtype)
        self.visit_embedding = OptimizerVisitEmbedding(output_dim=int(visit_dim))
        self.cells = nn.ModuleDict(
            {
                branch: LowRankGatedDeltaKVCell(
                    event_dim=int(event_dim),
                    token_dim=int(token_dim),
                    key_dim=int(key_dim),
                    value_dim=int(value_dim),
                    hidden_dim=int(adapter_hidden_dim),
                    value_rms_max=float(value_rms_max),
                    ctx_rms_max=float(ctx_rms_max),
                    state_rms_max=float(state_rms_max),
                    erase_gate_max=float(erase_gate_max),
                    write_gate_max=float(write_gate_max),
                    erase_bias=float(erase_bias),
                    write_bias=float(write_bias),
                    decay_bias=float(decay_bias),
                    decay_min=decay_min,
                    query_rms_unit=bool(query_rms_unit),
                    key_rms_unit=bool(key_rms_unit),
                    state_dtype=self.state_dtype,
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
            token_dim=int(token_dim),
            hidden_dim=int(adapter_hidden_dim),
            include_spatial_event=bool(include_spatial_event),
            include_parent_event=bool(include_parent_event),
            include_delta_summary=bool(include_delta_summary),
            include_visit_embedding=bool(include_visit_embedding),
        )
        self.branch_gate_raw = nn.Parameter(torch.zeros(3))
        self.visit_gate_raw = nn.Parameter(torch.zeros(4))
        init = dict(gate_init or {})
        for kind, value in init.items():
            if str(kind) in VISIT_KIND_TO_ID:
                p = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
                with torch.no_grad():
                    self.visit_gate_raw[int(VISIT_KIND_TO_ID[str(kind)])] = torch.logit(torch.tensor(p))

    @staticmethod
    def empty_state() -> ParentOptimizerDeltaKVState:
        return ParentOptimizerDeltaKVState.empty()

    def _visit(self, meta: Optional[VisitMeta | Dict[str, object] | object], *, ref: torch.Tensor, rows: int) -> torch.Tensor:
        return self.visit_embedding(meta, ref=ref, rows=int(rows))

    def _preview_dense(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        state: Optional[DeltaKVOptimizerBranchState],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        cell = self.cells[str(branch)]
        base = _ensure_delta_dense(
            cell,
            None if state is None else state.dense,
            rows=int(x.shape[0]),
            device=x.device,
            dtype=self.state_dtype,
        )
        seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        visit = self._visit(visit_meta, ref=x, rows=int(x.shape[0]))
        cell_state = DeltaKVCellState(
            kv_state=base.kv_state[: int(x.shape[0])],
            seen=seen,
        )
        out, aux = cell.read(x, cell_state)
        return out, seen, visit, aux

    def _preview_keyed(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        keys: torch.Tensor,
        state: Optional[DeltaKVOptimizerBranchState],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        if int(x.shape[0]) == 0:
            return (
                x.new_zeros((0, self.ctx_dim)),
                x.new_zeros((0,), dtype=torch.bool),
                x.new_zeros((0, self.visit_dim)),
                {
                    "ctx_rms_mean": 0.0,
                    "ctx_rms_max": 0.0,
                    "query_rms_mean": 0.0,
                    "query_rms_max": 0.0,
                    "state_dtype_id": float(self.state_dtype_id),
                },
            )
        cell = self.cells[str(branch)]
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        cell_state, _ = _gather_delta_keyed(
            cell,
            None if state is None else state.keyed,
            keys_u,
            device=x.device,
            dtype=self.state_dtype,
        )
        seen_u = cell_state.seen.to(device=x.device, dtype=torch.bool)
        visit_u = self._visit(visit_meta, ref=x_u, rows=int(keys_u.numel()))
        out_u, aux = cell.read(x_u, cell_state)
        return out_u.index_select(0, inverse), seen_u.index_select(0, inverse), visit_u.index_select(0, inverse), aux

    def _branch_preview(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: Optional[DeltaKVOptimizerBranchState],
        dense: bool,
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        if x is None:
            return None, None, None, {}
        if dense:
            return self._preview_dense(branch=branch, x=x, state=branch_state, visit_meta=visit_meta)
        if keys is None:
            raise ValueError(f"ParentOptimizerGatedDeltaKV {branch} preview requires keys")
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
            raise RuntimeError(f"ParentOptimizerGatedDeltaKV fused {branch} event contains NaN/Inf")
        return out

    def preview(
        self,
        *,
        event: EventPack,
        state: Optional[ParentOptimizerDeltaKVState],
        keys: ParentTemporalKeysV2,
        visit_meta: Optional[VisitMeta | Dict[str, object] | object] = None,
        **_: Any,
    ) -> ParentOptimizerPreview:
        state = state if state is not None else ParentOptimizerDeltaKVState.empty()
        ctx_bg, seen_bg, visit_bg, aux_bg = self._branch_preview(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            dense=self.dense_bg,
            support=event.support_bg,
            visit_meta=visit_meta,
        )
        ctx_distant, seen_distant, visit_distant, aux_distant = self._branch_preview(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            dense=self.dense_distant,
            support=event.support_distant,
            visit_meta=visit_meta,
        )
        ctx_rigid, seen_rigid, visit_rigid, aux_rigid = self._branch_preview(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            dense=False,
            support=event.support_rigid,
            visit_meta=visit_meta,
        )
        _ = (visit_bg, visit_distant, visit_rigid)
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
                aux[f"iforward/parent_optimizer_gdkv/{name}_preview_seen_ratio"] = (
                    float(seen.detach().float().mean().item()) if int(seen.numel()) else 0.0
                )
                aux[f"iforward/parent_optimizer_mamba/{name}_preview_seen_ratio"] = (
                    float(seen.detach().float().mean().item()) if int(seen.numel()) else 0.0
                )
        for branch, branch_aux in (("bg", aux_bg), ("distant", aux_distant), ("rigid", aux_rigid)):
            for key, value in branch_aux.items():
                aux[f"iforward/parent_optimizer_gdkv/{branch}_{key}"] = float(value)
        aux["iforward/parent_optimizer_gdkv/read"] = 1.0
        aux["iforward/parent_optimizer_gdkv/state_dtype_id"] = float(self.state_dtype_id)
        aux["iforward/parent_optimizer_mamba/read"] = 1.0
        aux["iforward/parent_optimizer_memory/type_id"] = 1.0
        aux["iforward/parent_optimizer_memory/is_gdkv"] = 1.0
        aux["iforward/parent_optimizer_memory/legacy_mamba_alias"] = 1.0
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
        _ = device
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

    @staticmethod
    def _written_aux(prefix: str, branch: str, count: float, cell_aux: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        out = {
            f"iforward/parent_optimizer_gdkv/{branch}_written": float(count),
            f"iforward/parent_optimizer_mamba/{branch}_written": float(count),
        }
        for key, value in dict(cell_aux or {}).items():
            out[f"iforward/parent_optimizer_gdkv/{branch}_{key}"] = float(value)
        return out

    def _write_dense(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        branch_state: DeltaKVOptimizerBranchState,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[DeltaKVOptimizerBranchState, Dict[str, float]]:
        if x is None:
            return branch_state, self._written_aux("gdkv", branch, 0.0)
        write = self._valid_write_mask(x, valid, support)
        assert write is not None
        cell = self.cells[str(branch)]
        base = _ensure_delta_dense(
            cell,
            branch_state.dense,
            rows=int(x.shape[0]),
            device=x.device,
            dtype=self.state_dtype,
        )
        old_seen = base.seen[: int(x.shape[0])].to(device=x.device, dtype=torch.bool)
        cell_state = DeltaKVCellState(
            kv_state=base.kv_state[: int(x.shape[0])],
            seen=old_seen,
        )
        updated, cell_aux = cell.write(x, cell_state, write_mask=write, visit_meta=visit_meta)
        n = int(x.shape[0])
        kv_state = base.kv_state.clone()
        seen = base.seen.clone()
        update_count = base.update_count.clone()
        last_visit_step = base.last_visit_step.clone()
        last_frame_id = base.last_frame_id.clone()
        last_visit_kind = base.last_visit_kind.clone()
        kv_state[:n] = updated.kv_state
        seen[:n] = updated.seen
        meta = self._meta_values(visit_meta, device=x.device)
        update_count[:n] = torch.where(write, update_count[:n] + 1, update_count[:n])
        last_visit_step[:n] = torch.where(write, torch.full((n,), int(meta["visit_step"]), device=x.device, dtype=torch.long), last_visit_step[:n])
        last_frame_id[:n] = torch.where(write, torch.full((n,), int(meta["frame_id"]), device=x.device, dtype=torch.long), last_frame_id[:n])
        last_visit_kind[:n] = torch.where(write, torch.full((n,), int(meta["kind"]), device=x.device, dtype=torch.long), last_visit_kind[:n])
        dense = DenseDeltaKVOptimizerState(
            kv_state=kv_state,
            seen=seen,
            update_count=update_count,
            last_visit_step=last_visit_step,
            last_frame_id=last_frame_id,
            last_visit_kind=last_visit_kind,
        )
        return DeltaKVOptimizerBranchState(dense=dense, keyed=branch_state.keyed), self._written_aux(
            "gdkv",
            branch,
            float(write.detach().float().sum().item()),
            cell_aux,
        )

    def _write_keyed(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: DeltaKVOptimizerBranchState,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        visit_meta: Optional[VisitMeta | Dict[str, object] | object],
    ) -> Tuple[DeltaKVOptimizerBranchState, Dict[str, float]]:
        if x is None:
            return branch_state, self._written_aux("gdkv", branch, 0.0)
        if keys is None:
            raise ValueError(f"ParentOptimizerGatedDeltaKV {branch} write requires keys")
        write = self._valid_write_mask(x, valid, support)
        assert write is not None
        keys_u, inverse, x_u = _weighted_aggregate(x, keys.to(device=x.device, dtype=torch.long), support=support)
        write_counts = x.new_zeros((int(keys_u.numel()),))
        write_counts.index_add_(0, inverse, write.to(dtype=x.dtype))
        write_u = write_counts > 0
        cell = self.cells[str(branch)]
        cell_state, meta_state = _gather_delta_keyed(
            cell,
            branch_state.keyed,
            keys_u,
            device=x.device,
            dtype=self.state_dtype,
        )
        updated, cell_aux = cell.write(x_u, cell_state, write_mask=write_u, visit_meta=visit_meta)
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
        keyed = _scatter_delta_keyed(
            cell,
            branch_state.keyed,
            keys_u,
            updated,
            meta_state,
            write_mask=write_u,
            device=x.device,
            dtype=self.state_dtype,
        )
        return DeltaKVOptimizerBranchState(dense=branch_state.dense, keyed=keyed), self._written_aux(
            "gdkv",
            branch,
            float(write.detach().float().sum().item()),
            cell_aux,
        )

    def write(
        self,
        *,
        spatial_event: EventPack,
        fused_event: Optional[EventPack] = None,
        write_event: Optional[EventPack] = None,
        state: Optional[ParentOptimizerDeltaKVState],
        keys: ParentTemporalKeysV2,
        visit_meta: Optional[VisitMeta | Dict[str, object] | object] = None,
        delta: Optional[object] = None,
        **_: Any,
    ) -> Tuple[ParentOptimizerDeltaKVState, Dict[str, float]]:
        state = state if state is not None else ParentOptimizerDeltaKVState.empty()
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
        next_state = ParentOptimizerDeltaKVState(
            bg=bg,
            distant=distant,
            rigid=rigid,
            global_update_step=int(state.global_update_step) + 1,
        )
        aux = {
            **aux_bg,
            **aux_distant,
            **aux_rigid,
            "iforward/parent_optimizer_gdkv/write": 1.0,
            "iforward/parent_optimizer_gdkv/global_update_step": float(next_state.global_update_step),
            "iforward/parent_optimizer_gdkv/state_dtype_id": float(self.state_dtype_id),
            "iforward/parent_optimizer_mamba/write": 1.0,
            "iforward/parent_optimizer_mamba/global_update_step": float(next_state.global_update_step),
            "iforward/parent_optimizer_memory/type_id": 1.0,
            "iforward/parent_optimizer_memory/is_gdkv": 1.0,
            "iforward/parent_optimizer_memory/legacy_mamba_alias": 1.0,
        }
        return next_state, aux


__all__ = [
    "DeltaKVCellState",
    "LowRankGatedDeltaKVCell",
    "ParentOptimizerGatedDeltaKV",
    "rms",
    "rms_clamp",
    "rms_unit",
]
