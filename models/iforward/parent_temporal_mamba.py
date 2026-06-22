from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0.event_encoder import EventPack

from .mamba import StreamingMambaCellState, StreamingMambaCell
from .memory import _aggregate_by_key, _ensure_dense_capacity, _gather_state_for_keys, _update_dense_point, _update_keyed
from .parent_temporal_keys import ParentTemporalKeys
from .parent_temporal_state import ParentTemporalBranchState, ParentTemporalState


@dataclass
class ParentTemporalPreview:
    event: EventPack
    aux: Dict[str, Any]


class _TemporalAdapter(nn.Module):
    def __init__(self, *, ctx_dim: int, event_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(ctx_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(event_dim)),
        )
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParentTemporalMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 64,
        ctx_dim: int = 32,
        model_dim: int = 32,
        state_dim: int = 8,
        conv_kernel: int = 2,
        adapter_hidden_dim: int = 64,
        dense_bg: bool = True,
        dense_distant: bool = True,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.dense_bg = bool(dense_bg)
        self.dense_distant = bool(dense_distant)
        self.cells = nn.ModuleDict(
            {
                branch: StreamingMambaCell(
                    input_dim=int(event_dim),
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
                branch: _TemporalAdapter(ctx_dim=int(ctx_dim), event_dim=int(event_dim), hidden_dim=int(adapter_hidden_dim))
                for branch in ("bg", "distant", "rigid")
            }
        )
        self.branch_gate_raw = nn.Parameter(torch.zeros(3))
        self.fusion_norm = nn.ModuleDict({branch: nn.LayerNorm(int(event_dim)) for branch in ("bg", "distant", "rigid")})

    def _preview_dense(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        state: Optional[ParentTemporalBranchState],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell = self.cells[str(branch)]
        dense = None if state is None else state.dense
        base = _ensure_dense_capacity(cell, dense, num_rows=int(x.shape[0]), device=x.device, dtype=x.dtype)
        cell_state = StreamingMambaCellState(
            conv_state=base.conv_state[: int(x.shape[0])],
            ssm_state=base.ssm_state[: int(x.shape[0])],
            seen=base.seen[: int(x.shape[0])],
        )
        out, _ = cell(x, cell_state, write_mask=torch.zeros((int(x.shape[0]),), device=x.device, dtype=torch.bool))
        seen = cell_state.seen.to(device=x.device, dtype=torch.bool)
        out = torch.where(seen[:, None], out, torch.zeros_like(out))
        return out, seen

    def _preview_keyed(
        self,
        *,
        branch: str,
        x: torch.Tensor,
        keys: torch.Tensor,
        state: Optional[ParentTemporalBranchState],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(x.shape[0]) == 0:
            return x.new_zeros((0, self.ctx_dim)), x.new_zeros((0,), dtype=torch.bool)
        cell = self.cells[str(branch)]
        keys_u, inverse, x_u = _aggregate_by_key(x, keys.to(device=x.device, dtype=torch.long))
        cell_state = _gather_state_for_keys(cell, None if state is None else state.keyed, keys_u, device=x.device, dtype=x.dtype)
        out_u, _ = cell(x_u, cell_state, write_mask=torch.zeros((int(keys_u.numel()),), device=x.device, dtype=torch.bool))
        seen_u = cell_state.seen.to(device=x.device, dtype=torch.bool)
        out_u = torch.where(seen_u[:, None], out_u, torch.zeros_like(out_u))
        return out_u.index_select(0, inverse), seen_u.index_select(0, inverse)

    def _branch_preview(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: Optional[ParentTemporalBranchState],
        dense: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x is None:
            return None, None
        if dense:
            return self._preview_dense(branch=branch, x=x, state=branch_state)
        if keys is None:
            raise ValueError(f"ParentTemporalMemory {branch} preview requires keys")
        return self._preview_keyed(branch=branch, x=x, keys=keys, state=branch_state)

    def _fuse(self, *, branch: str, event: Optional[torch.Tensor], ctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if event is None:
            return None
        if ctx is None:
            return event
        idx = {"bg": 0, "distant": 1, "rigid": 2}[str(branch)]
        gate = torch.sigmoid(self.branch_gate_raw[idx]).to(device=event.device, dtype=event.dtype)
        out = self.fusion_norm[str(branch)](event + gate * self.adapters[str(branch)](ctx))
        if not torch.isfinite(out).all():
            raise RuntimeError(f"ParentTemporalMemory fused {branch} event contains NaN/Inf")
        return out

    def preview(
        self,
        *,
        event: EventPack,
        state: Optional[ParentTemporalState],
        keys: ParentTemporalKeys,
    ) -> ParentTemporalPreview:
        state = state if state is not None else ParentTemporalState.empty()
        ctx_bg, seen_bg = self._branch_preview(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            dense=self.dense_bg,
        )
        ctx_distant, seen_distant = self._branch_preview(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            dense=self.dense_distant,
        )
        ctx_rigid, seen_rigid = self._branch_preview(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            dense=False,
        )
        fused = EventPack(
            event_bg=self._fuse(branch="bg", event=event.event_bg, ctx=ctx_bg),
            event_distant=self._fuse(branch="distant", event=event.event_distant, ctx=ctx_distant),
            event_rigid=self._fuse(branch="rigid", event=event.event_rigid, ctx=ctx_rigid),
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
                aux[f"iforward/parent_temporal/{name}_preview_seen_ratio"] = (
                    float(seen.detach().float().mean().item()) if int(seen.numel()) else 0.0
                )
        fused.aux = aux
        return ParentTemporalPreview(event=fused, aux=aux)

    def _valid_write_mask(self, x: Optional[torch.Tensor], valid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if valid is None:
            return torch.ones((int(x.shape[0]),), device=x.device, dtype=torch.bool)
        mask = valid.reshape(-1).to(device=x.device, dtype=torch.bool)
        if int(mask.numel()) != int(x.shape[0]):
            raise ValueError("ParentTemporalMemory valid mask row mismatch")
        return mask

    def _commit_branch(
        self,
        *,
        branch: str,
        x: Optional[torch.Tensor],
        keys: Optional[torch.Tensor],
        branch_state: ParentTemporalBranchState,
        dense: bool,
        valid: Optional[torch.Tensor],
    ) -> Tuple[ParentTemporalBranchState, Dict[str, float]]:
        if x is None:
            return branch_state, {f"iforward/parent_temporal/{branch}_committed": 0.0}
        write_mask = self._valid_write_mask(x, valid)
        cell = self.cells[str(branch)]
        if dense:
            new_dense, _ = _update_dense_point(cell, branch_state.dense, x=x, write_mask=write_mask)
            count = float(write_mask.detach().float().sum().item()) if write_mask is not None else 0.0
            return ParentTemporalBranchState(dense=new_dense, keyed=branch_state.keyed), {
                f"iforward/parent_temporal/{branch}_committed": count
            }
        if keys is None:
            raise ValueError(f"ParentTemporalMemory {branch} commit requires keys")
        new_keyed, _ = _update_keyed(cell, branch_state.keyed, keys=keys, x=x, write_mask=write_mask)
        count = float(write_mask.detach().float().sum().item()) if write_mask is not None else 0.0
        return ParentTemporalBranchState(dense=branch_state.dense, keyed=new_keyed), {
            f"iforward/parent_temporal/{branch}_committed": count
        }

    def commit(
        self,
        *,
        event: EventPack,
        state: Optional[ParentTemporalState],
        keys: ParentTemporalKeys,
        block_id: int,
    ) -> Tuple[ParentTemporalState, Dict[str, float]]:
        state = state if state is not None else ParentTemporalState.empty()
        bg, aux_bg = self._commit_branch(
            branch="bg",
            x=event.event_bg,
            keys=keys.bg,
            branch_state=state.bg,
            dense=self.dense_bg,
            valid=event.valid_bg,
        )
        distant, aux_distant = self._commit_branch(
            branch="distant",
            x=event.event_distant,
            keys=keys.distant,
            branch_state=state.distant,
            dense=self.dense_distant,
            valid=event.valid_distant,
        )
        rigid, aux_rigid = self._commit_branch(
            branch="rigid",
            x=event.event_rigid,
            keys=keys.rigid,
            branch_state=state.rigid,
            dense=False,
            valid=event.valid_rigid,
        )
        out = ParentTemporalState(bg=bg, distant=distant, rigid=rigid, last_committed_block_id=int(block_id))
        aux = {**aux_bg, **aux_distant, **aux_rigid, "iforward/parent_temporal/block_commit": 1.0}
        return out, aux


__all__ = ["ParentTemporalMemory", "ParentTemporalPreview"]
