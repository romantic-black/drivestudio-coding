from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import EventPack, LocalGSState

from .iforward_v6_state import IForwardV6BranchPointState, IForwardV6MemoryState
from .mamba import StreamingMambaCell
from .memory import IForwardMemoryStepContext, _update_dense_point, _update_keyed


@dataclass
class IForwardPointMemoryPack:
    ctx_bg: torch.Tensor
    ctx_distant: Optional[torch.Tensor] = None
    ctx_rigid: Optional[torch.Tensor] = None
    aux: Dict[str, Any] | None = None


class IForwardPointMambaBranch(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        state_dim: int,
        conv_kernel: int,
        output_dim: int,
        dense_point_memory: bool,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.dense_point_memory = bool(dense_point_memory)
        self.point = StreamingMambaCell(
            input_dim=int(input_dim),
            model_dim=int(model_dim),
            state_dim=int(state_dim),
            conv_kernel=int(conv_kernel),
            output_dim=int(output_dim),
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        keys: torch.Tensor,
        state: IForwardV6BranchPointState,
        write_mask: torch.Tensor,
    ) -> Tuple[IForwardV6BranchPointState, torch.Tensor]:
        if int(x.shape[0]) == 0:
            return state, x.new_zeros((0, self.output_dim))
        if bool(self.dense_point_memory):
            dense_state, ctx = _update_dense_point(
                self.point,
                state.dense_point,
                x=x,
                write_mask=write_mask,
            )
            return IForwardV6BranchPointState(point=state.point, dense_point=dense_state), ctx
        point_state, ctx = _update_keyed(
            self.point,
            state.point,
            keys=keys,
            x=x,
            write_mask=write_mask,
        )
        return IForwardV6BranchPointState(point=point_state, dense_point=state.dense_point), ctx


class IForwardPointMambaMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        point_ctx_dim: int = 16,
        model_dim: int = 16,
        state_dim: int = 4,
        conv_kernel: int = 2,
        obs_code_dim: int = 2,
        branch_embed_dim: int = 4,
        repeat_embed_dim: int = 4,
        dense_bg: bool = True,
        dense_distant: bool = True,
        hard_valid_required: bool = True,
        hard_support_min_commit: float = 0.0,
        hard_support_min_optimizer: float = 0.0,
        long_write_policy: str = "every_repeat",
        learnable_soft_gate: bool = False,
    ) -> None:
        super().__init__()
        if bool(learnable_soft_gate):
            raise ValueError("IForward-v6 PointMamba first version does not support learnable_soft_gate=true.")
        self.event_dim = int(event_dim)
        self.point_ctx_dim = int(point_ctx_dim)
        self.obs_code_dim = int(obs_code_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.repeat_embed_dim = int(repeat_embed_dim)
        self.hard_valid_required = bool(hard_valid_required)
        self.hard_support_min_commit = float(hard_support_min_commit)
        self.hard_support_min_optimizer = float(hard_support_min_optimizer)
        self.long_write_policy = str(long_write_policy or "every_repeat")
        if self.long_write_policy not in {"every_repeat", "commit_only", "none"}:
            raise ValueError(f"unsupported v6 point_mamba long_write_policy={self.long_write_policy!r}")
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        self.repeat_embed = nn.Sequential(
            nn.Linear(3, int(repeat_embed_dim)),
            nn.GELU(),
        )
        raw_dim = int(event_dim) + int(obs_code_dim) + 1 + 1 + int(branch_embed_dim) + int(repeat_embed_dim) + 2
        branch_kwargs = {
            "input_dim": raw_dim,
            "model_dim": int(model_dim),
            "state_dim": int(state_dim),
            "conv_kernel": int(conv_kernel),
            "output_dim": int(point_ctx_dim),
        }
        self.bg = IForwardPointMambaBranch(**branch_kwargs, dense_point_memory=bool(dense_bg))
        self.distant = IForwardPointMambaBranch(**branch_kwargs, dense_point_memory=bool(dense_distant))
        self.rigid = IForwardPointMambaBranch(**branch_kwargs, dense_point_memory=False)

    @staticmethod
    def empty_state() -> IForwardV6MemoryState:
        return IForwardV6MemoryState.empty()

    @staticmethod
    def _event_tensor(event: EventPack, name: str) -> Optional[torch.Tensor]:
        value = getattr(event, name, None)
        if value is None:
            return None
        if value.dim() != 2:
            raise ValueError(f"IForward-v6 PointMamba expected {name} [N,C], got {tuple(value.shape)}")
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

    @staticmethod
    def _rigid_route_indices(event: EventPack, n: int, device: torch.device) -> torch.Tensor:
        route = getattr(event, "route", None)
        raw = getattr(route, "S", None) if route is not None else None
        if raw is None:
            return torch.arange(n, device=device, dtype=torch.long)
        out = raw.to(device=device, dtype=torch.long).reshape(-1)
        if int(out.numel()) != int(n):
            return torch.arange(n, device=device, dtype=torch.long)
        return out

    def _write_long_memory(self, step: IForwardMemoryStepContext, ablation: str) -> bool:
        if str(ablation) in {"no_memory", "xcpe_only"}:
            return False
        if str(ablation) == "freeze_write":
            return False
        if self.long_write_policy == "none":
            return False
        if self.long_write_policy == "commit_only":
            return bool(step.commit_observation_memory) and bool(step.update_optimizer_memory)
        return bool(step.update_optimizer_memory)

    def _hard_support_min(self, step: IForwardMemoryStepContext) -> float:
        if bool(step.commit_observation_memory):
            return float(self.hard_support_min_commit)
        return float(self.hard_support_min_optimizer)

    def _token(
        self,
        *,
        event_x: torch.Tensor,
        branch_id: int,
        branch_name: str,
        step: IForwardMemoryStepContext,
        event: EventPack,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(event_x.shape[0])
        obs = getattr(event, f"obs_code_{branch_name}", None)
        if obs is None:
            obs_x = event_x.new_zeros((n, self.obs_code_dim))
        else:
            obs_x = obs.to(device=event_x.device, dtype=event_x.dtype)
            if obs_x.dim() == 1:
                obs_x = obs_x[:, None]
            if int(obs_x.shape[0]) != n:
                obs_x = event_x.new_zeros((n, self.obs_code_dim))
            elif int(obs_x.shape[1]) != int(self.obs_code_dim):
                obs_x = obs_x.reshape(n, -1)[:, : self.obs_code_dim]
                if int(obs_x.shape[1]) < int(self.obs_code_dim):
                    obs_x = torch.cat(
                        [obs_x, event_x.new_zeros((n, int(self.obs_code_dim) - int(obs_x.shape[1])))],
                        dim=-1,
                    )
        support = self._branch_signal(getattr(event, f"support_{branch_name}", None), n=n, ref=event_x, default=0.0)
        valid = self._branch_signal(getattr(event, f"valid_{branch_name}", None), n=n, ref=event_x, default=1.0)
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=event_x.device, dtype=torch.long)
        ).to(dtype=event_x.dtype)
        pos = event_x.new_tensor(
            [float(step.repeat_pos_code), float(step.frame_pos_code), float(step.rollout_pos_code)]
        ).reshape(1, 3).expand(n, -1)
        repeat = self.repeat_embed(pos).to(dtype=event_x.dtype)
        flags = torch.cat(
            [
                event_x.new_full((n, 1), 1.0 if bool(step.commit_observation_memory) else 0.0),
                event_x.new_full((n, 1), 1.0 if bool(step.update_optimizer_memory) else 0.0),
            ],
            dim=-1,
        )
        token = torch.cat([event_x, obs_x, support.clamp_min(0.0), valid.clamp(0.0, 1.0), branch, repeat, flags], dim=-1)
        return token, support, valid

    def _write_mask(
        self,
        *,
        support: torch.Tensor,
        valid: torch.Tensor,
        write_long_memory: bool,
        hard_support_min: float,
    ) -> torch.Tensor:
        n = int(support.shape[0])
        hard_write = torch.full((n,), bool(write_long_memory), device=support.device, dtype=torch.bool)
        if bool(self.hard_valid_required):
            hard_write = hard_write & valid.reshape(n, -1).any(dim=-1).to(dtype=torch.bool)
        support_f = support.reshape(n, -1).mean(dim=-1)
        return hard_write & (support_f >= float(hard_support_min))

    @staticmethod
    def _aux_for_branch(
        *,
        prefix: str,
        ctx: torch.Tensor,
        state: IForwardV6BranchPointState,
        write_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, float]:
        mem_state = state.dense_point if state.dense_point is not None else state.point
        seen = mem_state.seen.detach().to(dtype=torch.bool) if mem_state is not None else None
        capacity = int(seen.numel()) if seen is not None else 0
        seen_count = int(seen.sum().item()) if seen is not None and capacity > 0 else 0
        valid_bool = valid.reshape(int(valid.shape[0]), -1).any(dim=-1).to(dtype=torch.bool) if valid.numel() else valid.new_zeros((0,), dtype=torch.bool)
        return {
            f"point_mamba/{prefix}_ctx_norm": float(ctx.detach().norm(dim=-1).mean().item()) if ctx.numel() else 0.0,
            f"point_mamba/{prefix}_seen_ratio": float(seen_count) / float(max(capacity, 1)),
            f"point_mamba/{prefix}_seen_count": float(seen_count),
            f"point_mamba/{prefix}_capacity": float(capacity),
            f"point_mamba/{prefix}_update_ratio": float(write_mask.detach().float().mean().item()) if write_mask.numel() else 0.0,
            f"point_mamba/{prefix}_valid_ratio": float(valid_bool.detach().float().mean().item()) if valid_bool.numel() else 0.0,
        }

    def _forward_branch(
        self,
        *,
        module: IForwardPointMambaBranch,
        branch_state: IForwardV6BranchPointState,
        event_x: torch.Tensor,
        keys: torch.Tensor,
        branch_id: int,
        branch_name: str,
        event: EventPack,
        step: IForwardMemoryStepContext,
        write_long_memory: bool,
        hard_support_min: float,
        ablation: str,
    ) -> Tuple[IForwardV6BranchPointState, torch.Tensor, Dict[str, float]]:
        token, support, valid = self._token(
            event_x=event_x,
            branch_id=int(branch_id),
            branch_name=str(branch_name),
            step=step,
            event=event,
        )
        write_mask = self._write_mask(
            support=support,
            valid=valid,
            write_long_memory=bool(write_long_memory),
            hard_support_min=float(hard_support_min),
        )
        out_state, ctx = module(x=token, keys=keys, state=branch_state, write_mask=write_mask)
        if str(ablation) in {"xcpe_only", "no_memory"}:
            ctx = torch.zeros_like(ctx)
        aux = self._aux_for_branch(prefix=str(branch_name), ctx=ctx, state=out_state, write_mask=write_mask, valid=valid)
        return out_state, ctx, aux

    def forward(
        self,
        *,
        event: EventPack,
        local_state: LocalGSState,
        state: Optional[IForwardV6MemoryState],
        step_context: IForwardMemoryStepContext,
        ablation: str = "full",
    ) -> Tuple[IForwardV6MemoryState, IForwardPointMemoryPack, Dict[str, float]]:
        memory_state = state if isinstance(state, IForwardV6MemoryState) else IForwardV6MemoryState.empty()
        aux: Dict[str, float] = {}
        write_long_memory = self._write_long_memory(step_context, str(ablation))
        hard_support_min = self._hard_support_min(step_context)

        event_bg = self._event_tensor(event, "event_bg")
        if event_bg is None:
            raise RuntimeError("IForward-v6 PointMamba requires event.event_bg.")
        n_bg = int(event_bg.shape[0])
        bg_keys = torch.arange(n_bg, device=event_bg.device, dtype=torch.long)
        bg_state, ctx_bg, bg_aux = self._forward_branch(
            module=self.bg,
            branch_state=memory_state.bg,
            event_x=event_bg,
            keys=bg_keys,
            branch_id=0,
            branch_name="bg",
            event=event,
            step=step_context,
            write_long_memory=write_long_memory,
            hard_support_min=hard_support_min,
            ablation=ablation,
        )
        aux.update(bg_aux)

        distant_state = memory_state.distant
        ctx_distant = None
        event_distant = self._event_tensor(event, "event_distant")
        if event_distant is not None:
            n = int(event_distant.shape[0])
            keys = torch.arange(n, device=event_distant.device, dtype=torch.long) + 2_000_000_000
            distant_state, ctx_distant, dist_aux = self._forward_branch(
                module=self.distant,
                branch_state=memory_state.distant,
                event_x=event_distant,
                keys=keys,
                branch_id=1,
                branch_name="distant",
                event=event,
                step=step_context,
                write_long_memory=write_long_memory,
                hard_support_min=hard_support_min,
                ablation=ablation,
            )
            aux.update(dist_aux)

        rigid_state = memory_state.rigid
        ctx_rigid = None
        event_rigid = self._event_tensor(event, "event_rigid")
        if event_rigid is not None:
            n = int(event_rigid.shape[0])
            row_indices = self._rigid_route_indices(event, n, event_rigid.device)
            keys = row_indices + 3_000_000_000
            rigid_state, ctx_rigid, rigid_aux = self._forward_branch(
                module=self.rigid,
                branch_state=memory_state.rigid,
                event_x=event_rigid,
                keys=keys,
                branch_id=2,
                branch_name="rigid",
                event=event,
                step=step_context,
                write_long_memory=write_long_memory,
                hard_support_min=hard_support_min,
                ablation=ablation,
            )
            aux.update(rigid_aux)

        next_state = IForwardV6MemoryState(bg=bg_state, distant=distant_state, rigid=rigid_state)
        pack = IForwardPointMemoryPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux=aux)
        return next_state, pack, aux

