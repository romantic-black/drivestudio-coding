from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState

from .mamba import StreamingMambaCell, StreamingMambaCellState
from .state import (
    BranchMemoryState,
    DenseMambaState,
    IForwardMemoryState,
    IForwardShortMemoryEntry,
    IForwardShortWindowHistory,
    KeyedMambaState,
)


_HASH_PRIMES = (73856093, 19349663, 83492791)


@dataclass(frozen=True)
class IForwardMemoryStepContext:
    step_idx: int
    source_frame_idx: int
    commit_observation_memory: bool
    update_optimizer_memory: bool
    repeat_pos_code: float
    frame_pos_code: float
    rollout_pos_code: float
    global_step: int = 0
    is_frame_exit: bool = False


def _empty_keyed_from_cell(cell: StreamingMambaCell, *, device: torch.device, dtype: torch.dtype) -> KeyedMambaState:
    init = cell.init_state(0, device=device, dtype=dtype)
    return KeyedMambaState(
        keys=torch.zeros((0,), device=device, dtype=torch.long),
        conv_state=init.conv_state,
        ssm_state=init.ssm_state,
        seen=init.seen,
    )


def _empty_dense_from_cell(cell: StreamingMambaCell, *, num_rows: int, device: torch.device, dtype: torch.dtype) -> DenseMambaState:
    init = cell.init_state(int(num_rows), device=device, dtype=dtype)
    return DenseMambaState(conv_state=init.conv_state, ssm_state=init.ssm_state, seen=init.seen)


def _ensure_dense_capacity(
    cell: StreamingMambaCell,
    state: Optional[DenseMambaState],
    *,
    num_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DenseMambaState:
    if state is None:
        return _empty_dense_from_cell(cell, num_rows=int(num_rows), device=device, dtype=dtype)
    base = state.to(device=device, dtype=dtype)
    rows = int(base.seen.numel())
    if rows >= int(num_rows):
        return base
    extra = cell.init_state(int(num_rows) - rows, device=device, dtype=dtype)
    return DenseMambaState(
        conv_state=torch.cat([base.conv_state, extra.conv_state], dim=0),
        ssm_state=torch.cat([base.ssm_state, extra.ssm_state], dim=0),
        seen=torch.cat([base.seen, extra.seen], dim=0),
    )


def _aggregate_by_key(x: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique, inverse = torch.unique(keys.to(dtype=torch.long), sorted=True, return_inverse=True)
    out = x.new_zeros((int(unique.numel()), int(x.shape[-1])))
    out.index_add_(0, inverse, x)
    counts = torch.bincount(inverse, minlength=int(unique.numel())).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    out = out / counts[:, None]
    return unique, inverse, out


def _gather_state_for_keys(
    cell: StreamingMambaCell,
    state: Optional[KeyedMambaState],
    keys: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> StreamingMambaCellState:
    gathered = cell.init_state(int(keys.numel()), device=device, dtype=dtype)
    if state is None or int(keys.numel()) == 0 or int(state.keys.numel()) == 0:
        return gathered
    state = state.to(device=device, dtype=dtype)
    index = {int(k): int(i) for i, k in enumerate(state.keys.detach().cpu().tolist())}
    src_rows = []
    dst_rows = []
    for dst, key in enumerate(keys.detach().cpu().tolist()):
        src = index.get(int(key))
        if src is not None:
            src_rows.append(int(src))
            dst_rows.append(int(dst))
    if not src_rows:
        return gathered
    src_t = torch.tensor(src_rows, device=device, dtype=torch.long)
    dst_t = torch.tensor(dst_rows, device=device, dtype=torch.long)
    gathered.conv_state[dst_t] = state.conv_state[src_t]
    gathered.ssm_state[dst_t] = state.ssm_state[src_t]
    gathered.seen[dst_t] = state.seen[src_t]
    return gathered


def _scatter_state_for_keys(
    cell: StreamingMambaCell,
    state: Optional[KeyedMambaState],
    keys: torch.Tensor,
    updated: StreamingMambaCellState,
    *,
    write_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[KeyedMambaState]:
    if int(keys.numel()) == 0 or int(write_mask.numel()) == 0:
        return state
    base = state.to(device=device, dtype=dtype) if state is not None else _empty_keyed_from_cell(cell, device=device, dtype=dtype)
    existing = {int(k): int(i) for i, k in enumerate(base.keys.detach().cpu().tolist())}
    key_list = [int(k) for k in keys.detach().cpu().tolist()]
    mask_list = [bool(x) for x in write_mask.to(device=device, dtype=torch.bool).detach().cpu().tolist()]
    missing_keys = [int(k) for k, should_write in zip(key_list, mask_list) if should_write and int(k) not in existing]
    if missing_keys:
        new_keys = torch.tensor(missing_keys, device=device, dtype=torch.long)
        init = cell.init_state(len(missing_keys), device=device, dtype=dtype)
        all_keys = torch.cat([base.keys, new_keys], dim=0)
        all_conv = torch.cat([base.conv_state, init.conv_state], dim=0)
        all_ssm = torch.cat([base.ssm_state, init.ssm_state], dim=0)
        all_seen = torch.cat([base.seen, init.seen], dim=0)
        base = KeyedMambaState(keys=all_keys, conv_state=all_conv, ssm_state=all_ssm, seen=all_seen)
        existing = {int(k): int(i) for i, k in enumerate(base.keys.detach().cpu().tolist())}

    mask_rows = torch.nonzero(write_mask.to(device=device, dtype=torch.bool), as_tuple=False).squeeze(1)
    if int(mask_rows.numel()) == 0:
        return base
    rows = torch.tensor([existing[int(key_list[int(i)])] for i in mask_rows.detach().cpu().tolist()], device=device, dtype=torch.long)
    conv = base.conv_state.clone()
    ssm = base.ssm_state.clone()
    seen = base.seen.clone()
    conv[rows] = updated.conv_state[mask_rows]
    ssm[rows] = updated.ssm_state[mask_rows]
    seen[rows] = updated.seen[mask_rows]
    return KeyedMambaState(keys=base.keys, conv_state=conv, ssm_state=ssm, seen=seen)


def _update_keyed(
    cell: StreamingMambaCell,
    state: Optional[KeyedMambaState],
    *,
    keys: torch.Tensor,
    x: torch.Tensor,
    write_mask: torch.Tensor,
) -> Tuple[Optional[KeyedMambaState], torch.Tensor]:
    if int(x.shape[0]) == 0:
        return state, x.new_zeros((0, cell.output_dim))
    if int(keys.numel()) != int(x.shape[0]):
        raise ValueError(f"IForward memory key/event row mismatch: {int(keys.numel())} vs {int(x.shape[0])}")
    row_write_mask = write_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
    if int(row_write_mask.numel()) != int(x.shape[0]):
        raise ValueError("IForward memory write_mask row count mismatch.")
    keys_u, inverse, x_u = _aggregate_by_key(x, keys.to(device=x.device, dtype=torch.long))
    write_counts = x.new_zeros((int(keys_u.numel()),))
    write_counts.index_add_(0, inverse, row_write_mask.to(dtype=x.dtype))
    write_mask_u = write_counts > 0
    masked_sum = x.new_zeros((int(keys_u.numel()), int(x.shape[-1])))
    masked_sum.index_add_(0, inverse, x * row_write_mask.to(dtype=x.dtype)[:, None])
    masked_mean = masked_sum / write_counts.clamp_min(1.0)[:, None]
    x_u = torch.where(write_mask_u[:, None], masked_mean, x_u)
    cell_state = _gather_state_for_keys(cell, state, keys_u, device=x.device, dtype=x.dtype)
    out_u, updated = cell(x_u, cell_state, write_mask=write_mask_u)
    new_state = _scatter_state_for_keys(
        cell,
        state,
        keys_u,
        updated,
        write_mask=write_mask_u,
        device=x.device,
        dtype=x.dtype,
    )
    return new_state, out_u[inverse]


def _update_dense_point(
    cell: StreamingMambaCell,
    state: Optional[DenseMambaState],
    *,
    x: torch.Tensor,
    write_mask: torch.Tensor,
) -> Tuple[Optional[DenseMambaState], torch.Tensor]:
    n = int(x.shape[0])
    if n == 0:
        return state, x.new_zeros((0, cell.output_dim))
    base = _ensure_dense_capacity(cell, state, num_rows=n, device=x.device, dtype=x.dtype)
    cell_state = StreamingMambaCellState(
        conv_state=base.conv_state[:n],
        ssm_state=base.ssm_state[:n],
        seen=base.seen[:n],
    )
    mask = write_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != n:
        raise ValueError("IForward dense point write_mask row count mismatch.")
    out, updated = cell(x, cell_state, write_mask=mask)
    conv = base.conv_state.clone()
    ssm = base.ssm_state.clone()
    seen = base.seen.clone()
    conv[:n] = updated.conv_state
    ssm[:n] = updated.ssm_state
    seen[:n] = updated.seen
    return DenseMambaState(conv_state=conv, ssm_state=ssm, seen=seen), out


def _hash_cells(coords: torch.Tensor, *, cell_size: float, branch_offset: int) -> torch.Tensor:
    if int(coords.shape[0]) == 0:
        return torch.zeros((0,), device=coords.device, dtype=torch.long)
    cell = torch.floor(coords / float(cell_size)).to(dtype=torch.long)
    key = (
        cell[:, 0] * int(_HASH_PRIMES[0])
        ^ cell[:, 1] * int(_HASH_PRIMES[1])
        ^ cell[:, 2] * int(_HASH_PRIMES[2])
    )
    return key + int(branch_offset)


def _shuffle_rows(x: torch.Tensor) -> torch.Tensor:
    if int(x.shape[0]) <= 1:
        return x
    perm = torch.randperm(int(x.shape[0]), device=x.device)
    return x[perm]


class IForwardBranchMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        token_extra_dim: int,
        model_dim: int,
        state_dim: int,
        conv_kernel: int,
        enable_aux_stats: bool = False,
        dense_point_memory: bool = False,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.token_extra_dim = int(token_extra_dim)
        self.enable_aux_stats = bool(enable_aux_stats)
        self.dense_point_memory = bool(dense_point_memory)
        kwargs = {
            "input_dim": int(event_dim) + int(token_extra_dim),
            "model_dim": int(model_dim),
            "state_dim": int(state_dim),
            "conv_kernel": int(conv_kernel),
            "output_dim": int(event_dim),
        }
        self.point = StreamingMambaCell(**kwargs)
        self.cell = StreamingMambaCell(**kwargs)
        self.global_token = StreamingMambaCell(**kwargs)
        self.fuse = nn.Sequential(
            nn.Linear(4 * int(event_dim), int(event_dim)),
            nn.LayerNorm(int(event_dim)),
            nn.GELU(),
            nn.Linear(int(event_dim), int(event_dim)),
        )

    def forward(
        self,
        *,
        event: torch.Tensor,
        point_keys: torch.Tensor,
        cell_keys: torch.Tensor,
        global_keys: torch.Tensor,
        token_extra: torch.Tensor,
        short_ctx: torch.Tensor,
        state: BranchMemoryState,
        write_optimizer_memory: bool,
        write_short_entry: bool,
        hard_valid_required: bool,
        hard_support_min: float,
        ablation: str,
        frame_idx: int,
        step_idx: int,
        branch_name: str,
        support: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        enable_aux_stats: Optional[bool] = None,
    ) -> Tuple[BranchMemoryState, torch.Tensor, Dict[str, float], Optional[IForwardShortMemoryEntry]]:
        if int(event.shape[0]) == 0:
            return state, event.new_zeros((0, int(event.shape[-1]))), {"rows": 0.0}, None
        if int(token_extra.shape[0]) != int(event.shape[0]) or int(token_extra.shape[1]) != int(self.token_extra_dim):
            raise ValueError(
                f"IForward token_extra must be [N,{self.token_extra_dim}], got {tuple(token_extra.shape)}"
            )
        if int(short_ctx.shape[0]) != int(event.shape[0]) or int(short_ctx.shape[1]) != int(self.event_dim):
            raise ValueError(f"IForward short_ctx must be [N,{self.event_dim}], got {tuple(short_ctx.shape)}")
        aux_stats = self.enable_aux_stats if enable_aux_stats is None else bool(enable_aux_stats)
        if str(ablation) == "bypass_memory":
            ctx = torch.zeros_like(event)
            aux = {"rows": float(event.shape[0]), "bypass_memory": 1.0}
            if aux_stats:
                aux["ctx_norm"] = 0.0
                aux["short_ctx_norm"] = 0.0
            return state, ctx, aux, None
        zero_all = str(ablation) == "zero_all"
        freeze_write = str(ablation) in {"freeze_write", "zero_all"}
        x = torch.cat([event, token_extra.to(device=event.device, dtype=event.dtype)], dim=-1)
        write = bool(write_optimizer_memory) and not freeze_write
        if valid is None:
            valid_bool = torch.ones((int(event.shape[0]),), device=event.device, dtype=torch.bool)
        else:
            valid_bool = valid.to(device=event.device).reshape(int(event.shape[0]), -1)
            valid_bool = valid_bool.any(dim=-1).to(dtype=torch.bool)
        if support is None:
            support_f = event.new_zeros((int(event.shape[0]),))
        else:
            support_f = support.to(device=event.device, dtype=event.dtype).reshape(int(event.shape[0]), -1).mean(dim=-1)
        hard_write = torch.full((int(event.shape[0]),), bool(write), device=event.device, dtype=torch.bool)
        if bool(hard_valid_required):
            hard_write = hard_write & valid_bool
        hard_write = hard_write & (support_f >= float(hard_support_min))
        dense_point_state = state.dense_point
        point_state = state.point
        if bool(self.dense_point_memory):
            dense_point_state, point_ctx = _update_dense_point(
                self.point,
                state.dense_point,
                x=x,
                write_mask=hard_write,
            )
        else:
            point_state, point_ctx = _update_keyed(
                self.point,
                state.point,
                keys=point_keys,
                x=x,
                write_mask=hard_write,
            )
        cell_state, cell_ctx = _update_keyed(
            self.cell,
            state.cell,
            keys=cell_keys,
            x=x,
            write_mask=hard_write,
        )
        global_state, global_ctx = _update_keyed(
            self.global_token,
            state.global_token,
            keys=global_keys,
            x=x,
            write_mask=hard_write,
        )
        if str(ablation) == "zero_point":
            point_ctx = torch.zeros_like(point_ctx)
        if str(ablation) == "zero_cell":
            cell_ctx = torch.zeros_like(cell_ctx)
        if str(ablation) == "zero_global":
            global_ctx = torch.zeros_like(global_ctx)
        ctx = self.fuse(torch.cat([point_ctx, cell_ctx, global_ctx, short_ctx], dim=-1))
        if str(ablation) == "shuffle_memory":
            ctx = _shuffle_rows(ctx)
        if zero_all:
            ctx = torch.zeros_like(ctx)
        if not torch.isfinite(ctx).all():
            raise RuntimeError("IForward memory context contains NaN/Inf")
        entry = None
        entry_rows = torch.nonzero(hard_write, as_tuple=False).squeeze(1)
        if bool(write_short_entry) and int(entry_rows.numel()) > 0:
            entry = IForwardShortMemoryEntry(
                frame_idx=int(frame_idx),
                step_idx=int(step_idx),
                branch=str(branch_name),
                point_keys=point_keys[entry_rows].detach().clone(),
                cell_keys=cell_keys[entry_rows].detach().clone(),
                global_keys=global_keys[entry_rows].detach().clone(),
                event=event[entry_rows].detach().clone(),
                ctx=ctx[entry_rows].detach().clone(),
                support=None if support is None else support.reshape(int(event.shape[0]), -1)[entry_rows].detach().clone(),
                valid=None if valid is None else valid.reshape(int(event.shape[0]), -1)[entry_rows].detach().clone(),
            )
        aux = {"rows": float(event.shape[0])}
        if aux_stats:
            aux["hard_write_ratio"] = float(hard_write.float().mean().item()) if hard_write.numel() else 0.0
            aux["valid_true_ratio"] = float(valid_bool.float().mean().item()) if valid_bool.numel() else 0.0
            aux["support_mean"] = float(support_f.detach().mean().item()) if support_f.numel() else 0.0
            aux["support_max"] = float(support_f.detach().max().item()) if support_f.numel() else 0.0
            aux["support_positive_ratio"] = (
                float((support_f.detach() > 0).float().mean().item()) if support_f.numel() else 0.0
            )
            aux["short_entry_rows"] = float(entry_rows.numel()) if entry is not None else 0.0
            aux["ctx_norm"] = float(ctx.detach().norm(dim=-1).mean().item()) if ctx.numel() else 0.0
            aux["short_ctx_norm"] = (
                float(short_ctx.detach().norm(dim=-1).mean().item()) if short_ctx.numel() else 0.0
            )
            for prefix, mem_state in (
                ("point", dense_point_state if bool(self.dense_point_memory) else point_state),
                ("cell", cell_state),
                ("global", global_state),
            ):
                seen = getattr(mem_state, "seen", None)
                if seen is None:
                    aux[f"{prefix}_seen_count"] = 0.0
                    aux[f"{prefix}_capacity"] = 0.0
                    aux[f"{prefix}_seen_ratio"] = 0.0
                else:
                    seen_bool = seen.detach().to(dtype=torch.bool)
                    capacity = int(seen_bool.numel())
                    seen_count = int(seen_bool.sum().item()) if capacity else 0
                    aux[f"{prefix}_seen_count"] = float(seen_count)
                    aux[f"{prefix}_capacity"] = float(capacity)
                    aux[f"{prefix}_seen_ratio"] = float(seen_count) / float(max(capacity, 1))
        return (
            BranchMemoryState(
                point=point_state,
                cell=cell_state,
                global_token=global_state,
                dense_point=dense_point_state,
            ),
            ctx,
            aux,
            entry,
        )


class IForwardSceneMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        model_dim: Optional[int] = None,
        state_dim: int = 16,
        conv_kernel: int = 4,
        bg_cell_size: float = 0.5,
        distant_cell_size: float = 2.0,
        rigid_cell_size: float = 0.5,
        enable_aux_stats: bool = False,
        log_per_k_aux_interval: int = 50,
        dense_point_memory: bool = True,
        long_write_policy: str = "every_repeat",
        short_entry_policy: str = "frame_exit_only",
        short_entry_detach: bool = True,
        hard_valid_required: bool = True,
        hard_support_min_commit: float = 0.0,
        hard_support_min_optimizer: float = 0.0,
    ) -> None:
        super().__init__()
        model_dim = int(model_dim or event_dim)
        self.event_dim = int(event_dim)
        self.token_extra_dim = 8
        self.bg_cell_size = float(bg_cell_size)
        self.distant_cell_size = float(distant_cell_size)
        self.rigid_cell_size = float(rigid_cell_size)
        self.enable_aux_stats = bool(enable_aux_stats)
        self.log_per_k_aux_interval = max(1, int(log_per_k_aux_interval))
        self.dense_point_memory = bool(dense_point_memory)
        self.long_write_policy = str(long_write_policy or "every_repeat")
        self.short_entry_policy = str(short_entry_policy or "frame_exit_only")
        self.short_entry_detach = bool(short_entry_detach)
        self.hard_valid_required = bool(hard_valid_required)
        self.hard_support_min_commit = float(hard_support_min_commit)
        self.hard_support_min_optimizer = float(hard_support_min_optimizer)
        if self.long_write_policy not in {"every_repeat", "commit_only", "none"}:
            raise ValueError(f"unsupported IForward long_write_policy={self.long_write_policy!r}")
        if self.short_entry_policy not in {"frame_exit_only", "observation_commit_only", "every_repeat", "none"}:
            raise ValueError(f"unsupported IForward short_entry_policy={self.short_entry_policy!r}")
        branch_kwargs = {
            "event_dim": int(event_dim),
            "token_extra_dim": int(self.token_extra_dim),
            "model_dim": int(model_dim),
            "state_dim": int(state_dim),
            "conv_kernel": int(conv_kernel),
        }
        self.bg = IForwardBranchMemory(**branch_kwargs, dense_point_memory=bool(self.dense_point_memory))
        self.distant = IForwardBranchMemory(**branch_kwargs, dense_point_memory=bool(self.dense_point_memory))
        self.rigid = IForwardBranchMemory(**branch_kwargs, dense_point_memory=False)

    @staticmethod
    def empty_state() -> IForwardMemoryState:
        return IForwardMemoryState.empty()

    @staticmethod
    def _rigid_route_indices(event: EventPack, n: int, device: torch.device) -> torch.Tensor:
        route = getattr(event, "route", None)
        raw = getattr(route, "S", None)
        if raw is None:
            return torch.arange(n, device=device, dtype=torch.long)
        out = raw.to(device=device, dtype=torch.long).reshape(-1)
        if int(out.numel()) != int(n):
            return torch.arange(n, device=device, dtype=torch.long)
        return out

    @staticmethod
    def _rigid_object_ids(local_state: LocalGSState, indices: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        template = getattr(local_state, "rigid_template", None)
        if template is None or not hasattr(template, "point_ids") or int(indices.numel()) == 0:
            return indices.to(device=device, dtype=torch.long)
        point_ids = template.point_ids.to(device=device)
        if point_ids.dim() < 2 or int(point_ids.shape[0]) <= int(indices.max().item()):
            return indices.to(device=device, dtype=torch.long)
        return point_ids[indices, 0].to(dtype=torch.long)

    @classmethod
    def _rigid_object_keys(cls, local_state: LocalGSState, indices: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        return cls._rigid_object_ids(local_state, indices, device=device) + 5_000_000_000

    @staticmethod
    def _event_tensor(event: EventPack, name: str) -> Optional[torch.Tensor]:
        value = getattr(event, name, None)
        if value is None:
            return None
        if value.dim() != 2:
            raise ValueError(f"IForward memory expected {name} [N,C], got {tuple(value.shape)}")
        return value

    @staticmethod
    def _branch_signal(
        value: Optional[torch.Tensor],
        *,
        n: int,
        ref: torch.Tensor,
        default: float,
    ) -> torch.Tensor:
        if value is None:
            return ref.new_full((int(n), 1), float(default))
        x = value.to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 1:
            x = x[:, None]
        if int(x.shape[0]) != int(n):
            return ref.new_full((int(n), 1), float(default))
        if int(x.shape[1]) != 1:
            x = x.reshape(int(n), -1).mean(dim=-1, keepdim=True)
        return x

    def _token_extra(
        self,
        *,
        event: EventPack,
        branch: str,
        ref: torch.Tensor,
        n: int,
        step: IForwardMemoryStepContext,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        support = self._branch_signal(
            getattr(event, f"support_{branch}", None),
            n=n,
            ref=ref,
            default=0.0,
        )
        valid = self._branch_signal(
            getattr(event, f"valid_{branch}", None),
            n=n,
            ref=ref,
            default=1.0,
        )
        extra = torch.cat(
            [
                ref.new_full((int(n), 1), 1.0 if bool(step.commit_observation_memory) else 0.0),
                ref.new_full((int(n), 1), 1.0 if bool(step.update_optimizer_memory) else 0.0),
                ref.new_full((int(n), 1), float(step.repeat_pos_code)),
                ref.new_full((int(n), 1), float(step.frame_pos_code)),
                ref.new_full((int(n), 1), float(step.rollout_pos_code)),
                support.clamp_min(0.0),
                valid.clamp(0.0, 1.0),
                ref.new_full((int(n), 1), float(step.step_idx)),
            ],
            dim=-1,
        )
        return extra, support, valid

    @staticmethod
    def _short_read(
        *,
        short_history: Optional[IForwardShortWindowHistory],
        branch: str,
        point_keys: torch.Tensor,
        cell_keys: torch.Tensor,
        global_keys: torch.Tensor,
        ref: torch.Tensor,
        ablation: str,
    ) -> Tuple[torch.Tensor, float]:
        if short_history is None or str(ablation) == "bypass_memory":
            return ref.new_zeros(ref.shape), 0.0
        return short_history.read_context(
            branch=branch,
            point_keys=point_keys,
            cell_keys=cell_keys,
            global_keys=global_keys,
            ref=ref,
            drop=str(ablation) == "drop_short_window",
        )

    def _enable_aux_for_step(self, step_context: IForwardMemoryStepContext) -> bool:
        if not bool(self.enable_aux_stats):
            return False
        return int(step_context.global_step) % int(self.log_per_k_aux_interval) == 0

    def _write_long_memory(self, *, commit_observation_memory: bool, update_optimizer_memory: bool) -> bool:
        if self.long_write_policy == "none":
            return False
        if self.long_write_policy == "commit_only":
            return bool(commit_observation_memory) and bool(update_optimizer_memory)
        return bool(update_optimizer_memory)

    def _write_short_entry(self, *, step_context: IForwardMemoryStepContext, write_long_memory: bool) -> bool:
        if not bool(write_long_memory):
            return False
        if self.short_entry_policy == "none":
            return False
        if self.short_entry_policy == "every_repeat":
            return True
        if self.short_entry_policy == "observation_commit_only":
            return bool(step_context.commit_observation_memory)
        return bool(step_context.is_frame_exit)

    def _hard_support_min(self, *, commit_observation_memory: bool) -> float:
        if bool(commit_observation_memory):
            return float(self.hard_support_min_commit)
        return float(self.hard_support_min_optimizer)

    def forward(
        self,
        *,
        event: EventPack,
        local_state: LocalGSState,
        state: Optional[IForwardMemoryState],
        short_history: Optional[IForwardShortWindowHistory],
        step_context: IForwardMemoryStepContext,
        commit_observation_memory: bool,
        update_optimizer_memory: bool,
        ablation: str = "full",
    ) -> Tuple[IForwardMemoryState, ContextPack, Dict[str, float], List[IForwardShortMemoryEntry]]:
        memory_state = state if state is not None else IForwardMemoryState.empty()
        aux: Dict[str, float] = {}
        short_entries: List[IForwardShortMemoryEntry] = []
        enable_aux_stats = self._enable_aux_for_step(step_context)
        write_long_memory = self._write_long_memory(
            commit_observation_memory=bool(commit_observation_memory),
            update_optimizer_memory=bool(update_optimizer_memory),
        )
        write_short_entry = self._write_short_entry(step_context=step_context, write_long_memory=bool(write_long_memory))
        hard_support_min = self._hard_support_min(commit_observation_memory=bool(commit_observation_memory))
        if enable_aux_stats:
            aux["memory/write_long_memory"] = float(1.0 if write_long_memory else 0.0)
            aux["memory/write_short_entry"] = float(1.0 if write_short_entry else 0.0)
            aux["memory/short_entry_policy_code"] = float(
                {"none": 0, "observation_commit_only": 1, "frame_exit_only": 2, "every_repeat": 3}[self.short_entry_policy]
            )

        event_bg = self._event_tensor(event, "event_bg")
        if event_bg is None:
            raise RuntimeError("IForward memory requires event.event_bg.")
        n_bg = int(event_bg.shape[0])
        bg_point_keys = torch.arange(n_bg, device=event_bg.device, dtype=torch.long)
        bg_cell_keys = _hash_cells(local_state.bg.means.detach(), cell_size=self.bg_cell_size, branch_offset=1_000_000_000)
        bg_global_keys = torch.zeros((n_bg,), device=event_bg.device, dtype=torch.long) + 1
        bg_extra, bg_support, bg_valid = self._token_extra(event=event, branch="bg", ref=event_bg, n=n_bg, step=step_context)
        bg_short, bg_short_hit = self._short_read(
            short_history=short_history,
            branch="bg",
            point_keys=bg_point_keys,
            cell_keys=bg_cell_keys,
            global_keys=bg_global_keys,
            ref=event_bg,
            ablation=ablation,
        )
        bg_state, ctx_bg, bg_aux, bg_short_entry = self.bg(
            event=event_bg,
            point_keys=bg_point_keys,
            cell_keys=bg_cell_keys,
            global_keys=bg_global_keys,
            token_extra=bg_extra,
            short_ctx=bg_short,
            state=memory_state.bg,
            write_optimizer_memory=bool(write_long_memory),
            write_short_entry=bool(write_short_entry),
            hard_valid_required=bool(self.hard_valid_required),
            hard_support_min=float(hard_support_min),
            ablation=ablation,
            frame_idx=int(step_context.source_frame_idx),
            step_idx=int(step_context.step_idx),
            branch_name="bg",
            support=bg_support,
            valid=bg_valid,
            enable_aux_stats=enable_aux_stats,
        )
        if bg_short_entry is not None:
            short_entries.append(bg_short_entry)
        aux.update({f"memory/bg/{k}": float(v) for k, v in bg_aux.items()})
        if enable_aux_stats:
            aux["memory/bg/short_hit_ratio"] = float(bg_short_hit)

        ctx_distant = None
        distant_state = memory_state.distant
        event_distant = self._event_tensor(event, "event_distant")
        if event_distant is not None:
            n = int(event_distant.shape[0])
            point_keys = torch.arange(n, device=event_distant.device, dtype=torch.long) + 2_000_000_000
            coords = local_state.distant.means.detach() if local_state.distant is not None else event_distant.new_zeros((n, 3))
            cell_keys = _hash_cells(coords, cell_size=self.distant_cell_size, branch_offset=2_500_000_000)
            global_keys = torch.zeros((n,), device=event_distant.device, dtype=torch.long) + 2
            dist_extra, dist_support, dist_valid = self._token_extra(event=event, branch="distant", ref=event_distant, n=n, step=step_context)
            dist_short, dist_short_hit = self._short_read(
                short_history=short_history,
                branch="distant",
                point_keys=point_keys,
                cell_keys=cell_keys,
                global_keys=global_keys,
                ref=event_distant,
                ablation=ablation,
            )
            distant_state, ctx_distant, dist_aux, dist_short_entry = self.distant(
                event=event_distant,
                point_keys=point_keys,
                cell_keys=cell_keys,
                global_keys=global_keys,
                token_extra=dist_extra,
                short_ctx=dist_short,
                state=memory_state.distant,
                write_optimizer_memory=bool(write_long_memory),
                write_short_entry=bool(write_short_entry),
                hard_valid_required=bool(self.hard_valid_required),
                hard_support_min=float(hard_support_min),
                ablation=ablation,
                frame_idx=int(step_context.source_frame_idx),
                step_idx=int(step_context.step_idx),
                branch_name="distant",
                support=dist_support,
                valid=dist_valid,
                enable_aux_stats=enable_aux_stats,
            )
            if dist_short_entry is not None:
                short_entries.append(dist_short_entry)
            aux.update({f"memory/distant/{k}": float(v) for k, v in dist_aux.items()})
            if enable_aux_stats:
                aux["memory/distant/short_hit_ratio"] = float(dist_short_hit)

        ctx_rigid = None
        rigid_state = memory_state.rigid
        event_rigid = self._event_tensor(event, "event_rigid")
        if event_rigid is not None:
            n = int(event_rigid.shape[0])
            row_indices = self._rigid_route_indices(event, n, event_rigid.device)
            point_keys = row_indices + 3_000_000_000
            if local_state.rigid is not None and int(row_indices.numel()) > 0:
                coords = local_state.rigid.means[row_indices].detach()
            else:
                coords = event_rigid.new_zeros((n, 3))
            object_ids = self._rigid_object_ids(local_state, row_indices, device=event_rigid.device)
            local_cell_hash = _hash_cells(coords, cell_size=self.rigid_cell_size, branch_offset=0)
            cell_keys = object_ids * 1_000_003 + local_cell_hash + 3_500_000_000
            global_keys = self._rigid_object_keys(local_state, row_indices, device=event_rigid.device)
            rigid_extra, rigid_support, rigid_valid = self._token_extra(event=event, branch="rigid", ref=event_rigid, n=n, step=step_context)
            rigid_short, rigid_short_hit = self._short_read(
                short_history=short_history,
                branch="rigid",
                point_keys=point_keys,
                cell_keys=cell_keys,
                global_keys=global_keys,
                ref=event_rigid,
                ablation=ablation,
            )
            rigid_state, ctx_rigid, rigid_aux, rigid_short_entry = self.rigid(
                event=event_rigid,
                point_keys=point_keys,
                cell_keys=cell_keys,
                global_keys=global_keys,
                token_extra=rigid_extra,
                short_ctx=rigid_short,
                state=memory_state.rigid,
                write_optimizer_memory=bool(write_long_memory),
                write_short_entry=bool(write_short_entry),
                hard_valid_required=bool(self.hard_valid_required),
                hard_support_min=float(hard_support_min),
                ablation=ablation,
                frame_idx=int(step_context.source_frame_idx),
                step_idx=int(step_context.step_idx),
                branch_name="rigid",
                support=rigid_support,
                valid=rigid_valid,
                enable_aux_stats=enable_aux_stats,
            )
            if rigid_short_entry is not None:
                short_entries.append(rigid_short_entry)
            aux.update({f"memory/rigid/{k}": float(v) for k, v in rigid_aux.items()})
            if enable_aux_stats:
                aux["memory/rigid/short_hit_ratio"] = float(rigid_short_hit)

        return (
            IForwardMemoryState(bg=bg_state, distant=distant_state, rigid=rigid_state),
            ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux=aux),
            aux,
            short_entries,
        )
