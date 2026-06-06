from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack

from .history_gate import IForwardAttributeGate, IForwardGatePack


def _state_ref(local_state: LocalGSState) -> torch.Tensor:
    return local_state.bg.means


def _branch_rows(branch: Any) -> int:
    return int(branch.means.shape[0]) if branch is not None else 0


def _col(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if value is None:
        return ref.new_full((int(n), 1), float(default))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 GRU column row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) != 1:
        out = out.reshape(int(n), -1).mean(dim=-1, keepdim=True)
    return out


def _bool_col(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: bool = True) -> torch.Tensor:
    if value is None:
        return torch.full((int(n), 1), bool(default), device=ref.device, dtype=torch.bool)
    out = value.to(device=ref.device)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 GRU bool row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) != 1:
        out = out.reshape(int(n), -1).any(dim=-1, keepdim=True)
    return out.to(dtype=torch.bool)


def _obs2(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor) -> torch.Tensor:
    if value is None:
        return ref.new_zeros((int(n), 2))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 GRU obs_code row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) >= 2:
        return out[:, :2]
    return torch.cat([out, out.new_zeros((int(n), 2 - int(out.shape[1])))], dim=-1)


@dataclass
class IForwardGRUBranchState:
    h: torch.Tensor
    seen: torch.Tensor
    last_visit_idx: torch.Tensor
    last_source_frame_idx: torch.Tensor

    @classmethod
    def empty(cls, *, num_rows: int, hidden_dim: int, ref: torch.Tensor) -> "IForwardGRUBranchState":
        h = ref.detach().new_zeros((int(num_rows), int(hidden_dim)))
        seen = torch.zeros((int(num_rows),), device=ref.device, dtype=torch.bool)
        last = torch.full((int(num_rows),), -1, device=ref.device, dtype=torch.long)
        return cls(h=h, seen=seen, last_visit_idx=last.clone(), last_source_frame_idx=last)

    @property
    def last_frame_idx(self) -> torch.Tensor:
        return self.last_source_frame_idx

    def detach(self) -> "IForwardGRUBranchState":
        return IForwardGRUBranchState(
            h=self.h.detach().clone(),
            seen=self.seen.detach().clone(),
            last_visit_idx=self.last_visit_idx.detach().clone(),
            last_source_frame_idx=self.last_source_frame_idx.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "IForwardGRUBranchState":
        return IForwardGRUBranchState(
            h=self.h.to(device=device, dtype=dtype or self.h.dtype),
            seen=self.seen.to(device=device),
            last_visit_idx=self.last_visit_idx.to(device=device),
            last_source_frame_idx=self.last_source_frame_idx.to(device=device),
        )


@dataclass
class IForwardGRUMemoryState:
    bg: IForwardGRUBranchState
    distant: IForwardGRUBranchState
    rigid: IForwardGRUBranchState

    @classmethod
    def empty(cls) -> "IForwardGRUMemoryState":
        ref = torch.zeros((0, 1), dtype=torch.float32)
        return cls(
            bg=IForwardGRUBranchState.empty(num_rows=0, hidden_dim=1, ref=ref),
            distant=IForwardGRUBranchState.empty(num_rows=0, hidden_dim=1, ref=ref),
            rigid=IForwardGRUBranchState.empty(num_rows=0, hidden_dim=1, ref=ref),
        )

    @classmethod
    def from_local_state(cls, local_state: LocalGSState, *, hidden_dim: int) -> "IForwardGRUMemoryState":
        ref = _state_ref(local_state)
        return cls(
            bg=IForwardGRUBranchState.empty(num_rows=_branch_rows(local_state.bg), hidden_dim=int(hidden_dim), ref=ref),
            distant=IForwardGRUBranchState.empty(
                num_rows=_branch_rows(local_state.distant),
                hidden_dim=int(hidden_dim),
                ref=ref,
            ),
            rigid=IForwardGRUBranchState.empty(
                num_rows=_branch_rows(local_state.rigid),
                hidden_dim=int(hidden_dim),
                ref=ref,
            ),
        )

    def detach(self) -> "IForwardGRUMemoryState":
        return IForwardGRUMemoryState(bg=self.bg.detach(), distant=self.distant.detach(), rigid=self.rigid.detach())

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in ("bg", "distant", "rigid"):
            branch = getattr(self, name)
            seen = branch.seen.detach().to(dtype=torch.bool)
            capacity = int(seen.numel())
            seen_count = int(seen.sum().item()) if capacity > 0 else 0
            out[f"{name}_point"] = float(seen_count)
            out[f"{name}_point_seen"] = float(seen_count)
            out[f"{name}_point_capacity"] = float(capacity)
            out[f"{name}_point_seen_ratio"] = float(seen_count) / float(max(capacity, 1))
        return out


@dataclass
class IForwardGRUBranchPrepared:
    branch: str
    rows: Optional[torch.Tensor]
    h_prior: torch.Tensor
    dt: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor
    obs_code: torch.Tensor
    event_x: torch.Tensor


@dataclass
class IForwardGRUPrepared:
    bg: IForwardGRUBranchPrepared
    distant: Optional[IForwardGRUBranchPrepared]
    rigid: Optional[IForwardGRUBranchPrepared]


class _BranchTimeAwareGRU(nn.Module):
    def __init__(self, *, event_dim: int, hidden_dim: int, ctx_dim: int, dt_clip: float) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.hidden_dim = int(hidden_dim)
        self.ctx_dim = int(ctx_dim)
        self.dt_clip = float(dt_clip)
        self.decay_log_rate = nn.Parameter(torch.full((int(hidden_dim),), -3.0))
        self.read_norm = nn.LayerNorm(int(hidden_dim))
        self.read_proj = nn.Linear(int(hidden_dim), int(ctx_dim))
        self.write_norm = nn.LayerNorm(int(event_dim) + 13)
        self.cell = nn.GRUCell(int(event_dim) + 13, int(hidden_dim))

    def read(
        self,
        *,
        event_x: torch.Tensor,
        state: IForwardGRUBranchState,
        rows: Optional[torch.Tensor],
        source_frame_idx: int,
        visit_idx: int,
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        ablation: str,
    ) -> tuple[torch.Tensor, IForwardGRUBranchPrepared, Dict[str, float]]:
        n = int(event_x.shape[0])
        if n == 0:
            z_ctx = event_x.new_zeros((0, int(self.ctx_dim)))
            prepared = IForwardGRUBranchPrepared(
                branch="",
                rows=rows,
                h_prior=event_x.new_zeros((0, int(self.hidden_dim))),
                dt=event_x.new_zeros((0, 1)),
                support=event_x.new_zeros((0, 1)),
                valid=torch.zeros((0, 1), device=event_x.device, dtype=torch.bool),
                obs_code=event_x.new_zeros((0, 2)),
                event_x=event_x,
            )
            return z_ctx, prepared, {"ctx_norm": 0.0, "dt_mean": 0.0}
        state = state.to(device=event_x.device, dtype=event_x.dtype)
        if rows is None:
            if int(state.h.shape[0]) != n:
                raise ValueError(f"IForward v3 GRU state row mismatch: state={int(state.h.shape[0])} event={n}")
            h_old = state.h
            seen = state.seen
            last = state.last_visit_idx
        else:
            idx = rows.to(device=event_x.device, dtype=torch.long).reshape(-1)
            if int(idx.numel()) != n:
                raise ValueError("IForward v3 GRU route rows do not match event rows.")
            h_old = state.h.to(device=event_x.device, dtype=event_x.dtype)[idx]
            seen = state.seen.to(device=event_x.device)[idx]
            last = state.last_visit_idx.to(device=event_x.device)[idx]
        zeros = torch.zeros_like(h_old)
        h_seen = torch.where(seen[:, None], h_old, zeros)
        del source_frame_idx
        current_visit = int(visit_idx)
        if current_visit < 0:
            current_visit = 0
        dt = torch.full_like(last, int(current_visit), dtype=torch.long) - last
        dt = torch.where(seen, dt.clamp_min(0), torch.zeros_like(dt))
        dt_f = dt.to(dtype=event_x.dtype).clamp(min=0.0, max=float(self.dt_clip)).unsqueeze(-1)
        rate = F.softplus(self.decay_log_rate).to(device=event_x.device, dtype=event_x.dtype).view(1, -1)
        h_prior = h_seen * torch.exp(-rate * dt_f)
        ctx = self.read_proj(self.read_norm(h_prior))
        if str(ablation) == "no_gru":
            ctx = torch.zeros_like(ctx)
        prepared = IForwardGRUBranchPrepared(
            branch="",
            rows=rows,
            h_prior=h_prior,
            dt=dt_f,
            support=_col(support, n=n, ref=event_x, default=1.0),
            valid=_bool_col(valid, n=n, ref=event_x, default=True),
            obs_code=_obs2(obs_code, n=n, ref=event_x),
            event_x=event_x,
        )
        aux = {
            "ctx_norm": float(ctx.detach().norm(dim=-1).mean().item()) if ctx.numel() else 0.0,
            "dt_mean": float(dt_f.detach().mean().item()) if dt_f.numel() else 0.0,
        }
        return ctx, prepared, aux

    def write(
        self,
        *,
        prepared: IForwardGRUBranchPrepared,
        state: IForwardGRUBranchState,
        delta: Optional[BranchDelta],
        gate: Optional[IForwardAttributeGate],
        source_frame_idx: int,
        visit_idx: int,
        repeat_pos_code: float,
        frame_pos_code: float,
        rollout_pos_code: float,
        update_optimizer_memory: bool,
        hard_valid_required: bool,
        hard_support_min_optimizer: float,
        ablation: str,
    ) -> tuple[IForwardGRUBranchState, Dict[str, float]]:
        event_x = prepared.event_x
        n = int(event_x.shape[0])
        if n == 0:
            return state, {"write_ratio": 0.0}
        state = state.to(device=event_x.device, dtype=event_x.dtype)
        if gate is None:
            one = event_x.new_ones((n, 1))
            gate = IForwardAttributeGate(means=one, scales=one, quat=one, opacity=one, sh=one, hidden=one)
        delta_norm = (
            delta.means.detach().norm(dim=-1, keepdim=True).to(device=event_x.device, dtype=event_x.dtype)
            if delta is not None
            else event_x.new_zeros((n, 1))
        )
        pos = event_x.new_full((n, 3), 0.0)
        pos[:, 0] = float(repeat_pos_code)
        pos[:, 1] = float(frame_pos_code)
        pos[:, 2] = float(rollout_pos_code)
        write_token = torch.cat(
            [
                event_x,
                prepared.obs_code.to(device=event_x.device, dtype=event_x.dtype),
                torch.log1p(prepared.support.to(device=event_x.device, dtype=event_x.dtype).clamp_min(0.0)),
                prepared.valid.to(device=event_x.device, dtype=event_x.dtype),
                pos,
                gate.means.to(device=event_x.device, dtype=event_x.dtype),
                gate.scales.to(device=event_x.device, dtype=event_x.dtype),
                gate.quat.to(device=event_x.device, dtype=event_x.dtype),
                gate.opacity.to(device=event_x.device, dtype=event_x.dtype),
                gate.sh.to(device=event_x.device, dtype=event_x.dtype),
                delta_norm,
            ],
            dim=-1,
        )
        if int(write_token.shape[-1]) != int(self.event_dim) + 13:
            raise RuntimeError("IForward v3 GRU write token dim mismatch.")
        write_mask = torch.full((n,), bool(update_optimizer_memory), device=event_x.device, dtype=torch.bool)
        if bool(hard_valid_required):
            write_mask = write_mask & prepared.valid.reshape(-1).to(device=event_x.device)
        write_mask = write_mask & (prepared.support.reshape(-1).to(device=event_x.device, dtype=event_x.dtype) >= float(hard_support_min_optimizer))
        if str(ablation) in {"freeze_write", "no_gru"}:
            write_mask = torch.zeros_like(write_mask)
        h_candidate = self.cell(self.write_norm(write_token), prepared.h_prior)
        h_new_rows = torch.where(write_mask[:, None], h_candidate, prepared.h_prior)
        h = state.h.clone()
        seen = state.seen.clone()
        last_visit = state.last_visit_idx.clone()
        last_source = state.last_source_frame_idx.clone()
        current_visit = int(visit_idx)
        if current_visit < 0:
            current_visit = 0
        if prepared.rows is None:
            if int(h.shape[0]) != n:
                raise ValueError("IForward v3 GRU write state/event row mismatch.")
            persist_mask = write_mask | seen
            h = torch.where(persist_mask[:, None], h_new_rows, h)
            seen = seen | write_mask
            last_visit = torch.where(persist_mask, torch.full_like(last_visit, int(current_visit)), last_visit)
            last_source = torch.where(write_mask, torch.full_like(last_source, int(source_frame_idx)), last_source)
        else:
            idx = prepared.rows.to(device=event_x.device, dtype=torch.long).reshape(-1)
            seen_rows = seen[idx]
            persist_mask = write_mask | seen_rows
            h[idx] = torch.where(persist_mask[:, None], h_new_rows, h[idx])
            seen[idx] = seen[idx] | write_mask
            last_visit[idx] = torch.where(persist_mask, torch.full_like(last_visit[idx], int(current_visit)), last_visit[idx])
            last_source[idx] = torch.where(write_mask, torch.full_like(last_source[idx], int(source_frame_idx)), last_source[idx])
        return IForwardGRUBranchState(h=h, seen=seen, last_visit_idx=last_visit, last_source_frame_idx=last_source), {
            "write_ratio": float(write_mask.detach().to(dtype=torch.float32).mean().item()) if write_mask.numel() else 0.0,
            "delta_means_norm_mean": float(delta_norm.detach().mean().item()) if delta_norm.numel() else 0.0,
        }


class IForwardTimeAwarePointGRU(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        hidden_dim: int = 48,
        ctx_dim: int = 48,
        dt_clip: float = 32.0,
        hard_valid_required: bool = True,
        hard_support_min_optimizer: float = 0.0,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.hidden_dim = int(hidden_dim)
        self.ctx_dim = int(ctx_dim)
        self.hard_valid_required = bool(hard_valid_required)
        self.hard_support_min_optimizer = float(hard_support_min_optimizer)
        self.bg = _BranchTimeAwareGRU(event_dim=event_dim, hidden_dim=hidden_dim, ctx_dim=ctx_dim, dt_clip=dt_clip)
        self.distant = _BranchTimeAwareGRU(event_dim=event_dim, hidden_dim=hidden_dim, ctx_dim=ctx_dim, dt_clip=dt_clip)
        self.rigid = _BranchTimeAwareGRU(event_dim=event_dim, hidden_dim=hidden_dim, ctx_dim=ctx_dim, dt_clip=dt_clip)

    def init_state(self, local_state: LocalGSState) -> IForwardGRUMemoryState:
        return IForwardGRUMemoryState.from_local_state(local_state, hidden_dim=int(self.hidden_dim))

    def _read_branch(
        self,
        *,
        branch_name: str,
        event_x: Optional[torch.Tensor],
        state: IForwardGRUBranchState,
        rows: Optional[torch.Tensor],
        source_frame_idx: int,
        visit_idx: int,
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        ablation: str,
    ) -> tuple[Optional[torch.Tensor], Optional[IForwardGRUBranchPrepared], Dict[str, float]]:
        if event_x is None:
            return None, None, {}
        ctx, prepared, aux = getattr(self, branch_name).read(
            event_x=event_x,
            state=state,
            rows=rows,
            source_frame_idx=int(source_frame_idx),
            visit_idx=int(visit_idx),
            support=support,
            valid=valid,
            obs_code=obs_code,
            ablation=str(ablation),
        )
        prepared.branch = str(branch_name)
        return ctx, prepared, {f"v3/gru/{branch_name}_{k}": v for k, v in aux.items()}

    def read(
        self,
        *,
        event: EventPack,
        local_state: LocalGSState,
        state: IForwardGRUMemoryState,
        step_context: Any,
        ablation: str = "full",
    ) -> tuple[ContextPack, IForwardGRUPrepared, Dict[str, float]]:
        del local_state
        aux: Dict[str, float] = {}
        ctx_bg, prep_bg, bg_aux = self._read_branch(
            branch_name="bg",
            event_x=event.event_bg,
            state=state.bg,
            rows=None,
            source_frame_idx=int(step_context.source_frame_idx),
            visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
            support=event.support_bg,
            valid=event.valid_bg,
            obs_code=event.obs_code_bg,
            ablation=str(ablation),
        )
        if ctx_bg is None or prep_bg is None:
            raise RuntimeError("IForward v3 GRU requires bg event.")
        aux.update(bg_aux)
        ctx_distant = None
        prep_distant = None
        if event.event_distant is not None:
            ctx_distant, prep_distant, d_aux = self._read_branch(
                branch_name="distant",
                event_x=event.event_distant,
                state=state.distant,
                rows=None,
                source_frame_idx=int(step_context.source_frame_idx),
                visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
                support=event.support_distant,
                valid=event.valid_distant,
                obs_code=event.obs_code_distant,
                ablation=str(ablation),
            )
            aux.update(d_aux)
        ctx_rigid = None
        prep_rigid = None
        route = getattr(event, "route", None)
        rows = getattr(route, "S", None) if route is not None else None
        if event.event_rigid is not None and rows is not None:
            ctx_rigid, prep_rigid, r_aux = self._read_branch(
                branch_name="rigid",
                event_x=event.event_rigid,
                state=state.rigid,
                rows=rows,
                source_frame_idx=int(step_context.source_frame_idx),
                visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
                support=event.support_rigid,
                valid=event.valid_rigid,
                obs_code=event.obs_code_rigid,
                ablation=str(ablation),
            )
            aux.update(r_aux)
        return (
            ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux=aux),
            IForwardGRUPrepared(bg=prep_bg, distant=prep_distant, rigid=prep_rigid),
            aux,
        )

    def write_after_update(
        self,
        *,
        prepared: IForwardGRUPrepared,
        state: IForwardGRUMemoryState,
        delta_raw: DeltaPack,
        gate: IForwardGatePack,
        step_context: Any,
        ablation: str = "full",
    ) -> tuple[IForwardGRUMemoryState, Dict[str, float]]:
        aux: Dict[str, float] = {}
        bg_state, bg_aux = self.bg.write(
            prepared=prepared.bg,
            state=state.bg,
            delta=delta_raw.bg,
            gate=gate.bg,
            source_frame_idx=int(step_context.source_frame_idx),
            visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
            repeat_pos_code=float(step_context.repeat_pos_code),
            frame_pos_code=float(step_context.frame_pos_code),
            rollout_pos_code=float(step_context.rollout_pos_code),
            update_optimizer_memory=bool(step_context.update_optimizer_memory),
            hard_valid_required=bool(self.hard_valid_required),
            hard_support_min_optimizer=float(self.hard_support_min_optimizer),
            ablation=str(ablation),
        )
        aux.update({f"v3/gru/bg_{k}": v for k, v in bg_aux.items()})
        distant_state = state.distant
        if prepared.distant is not None:
            distant_state, d_aux = self.distant.write(
                prepared=prepared.distant,
                state=state.distant,
                delta=delta_raw.distant,
                gate=gate.distant,
                source_frame_idx=int(step_context.source_frame_idx),
                visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
                repeat_pos_code=float(step_context.repeat_pos_code),
                frame_pos_code=float(step_context.frame_pos_code),
                rollout_pos_code=float(step_context.rollout_pos_code),
                update_optimizer_memory=bool(step_context.update_optimizer_memory),
                hard_valid_required=bool(self.hard_valid_required),
                hard_support_min_optimizer=float(self.hard_support_min_optimizer),
                ablation=str(ablation),
            )
            aux.update({f"v3/gru/distant_{k}": v for k, v in d_aux.items()})
        rigid_state = state.rigid
        if prepared.rigid is not None:
            rigid_state, r_aux = self.rigid.write(
                prepared=prepared.rigid,
                state=state.rigid,
                delta=delta_raw.rigid,
                gate=gate.rigid,
                source_frame_idx=int(step_context.source_frame_idx),
                visit_idx=int(getattr(step_context, "episode_visit_idx", -1)),
                repeat_pos_code=float(step_context.repeat_pos_code),
                frame_pos_code=float(step_context.frame_pos_code),
                rollout_pos_code=float(step_context.rollout_pos_code),
                update_optimizer_memory=bool(step_context.update_optimizer_memory),
                hard_valid_required=bool(self.hard_valid_required),
                hard_support_min_optimizer=float(self.hard_support_min_optimizer),
                ablation=str(ablation),
            )
            aux.update({f"v3/gru/rigid_{k}": v for k, v in r_aux.items()})
        return IForwardGRUMemoryState(bg=bg_state, distant=distant_state, rigid=rigid_state), aux


__all__ = [
    "IForwardGRUBranchPrepared",
    "IForwardGRUBranchState",
    "IForwardGRUMemoryState",
    "IForwardGRUPrepared",
    "IForwardTimeAwarePointGRU",
]
