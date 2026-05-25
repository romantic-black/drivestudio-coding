from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ImageRef = Tuple[int, int]


@dataclass
class Stage6VSMState:
    tokens_bg: torch.Tensor
    proto_bg: torch.Tensor
    global_bg: torch.Tensor
    valid_count_bg: torch.Tensor
    tokens_rigid: torch.Tensor
    proto_rigid: torch.Tensor
    global_rigid: torch.Tensor
    valid_count_rigid: torch.Tensor
    written_refs: Set[ImageRef] = field(default_factory=set)
    episode_id: int = -1

    def detach(self) -> "Stage6VSMState":
        return Stage6VSMState(
            tokens_bg=self.tokens_bg.detach().clone(),
            proto_bg=self.proto_bg.detach().clone(),
            global_bg=self.global_bg.detach().clone(),
            valid_count_bg=self.valid_count_bg.detach().clone(),
            tokens_rigid=self.tokens_rigid.detach().clone(),
            proto_rigid=self.proto_rigid.detach().clone(),
            global_rigid=self.global_rigid.detach().clone(),
            valid_count_rigid=self.valid_count_rigid.detach().clone(),
            written_refs=set(self.written_refs),
            episode_id=int(self.episode_id),
        )

    def assert_finite(self, label: str = "vsm") -> None:
        for name, value in (
            ("tokens_bg", self.tokens_bg),
            ("proto_bg", self.proto_bg),
            ("global_bg", self.global_bg),
            ("valid_count_bg", self.valid_count_bg),
            ("tokens_rigid", self.tokens_rigid),
            ("proto_rigid", self.proto_rigid),
            ("global_rigid", self.global_rigid),
            ("valid_count_rigid", self.valid_count_rigid),
        ):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{label}.{name} contains NaN/Inf")


@dataclass
class Stage6BranchQueryPred:
    event_hat: torch.Tensor
    visible_logit: torch.Tensor
    support_log_hat: torch.Tensor
    obs_code_hat: torch.Tensor


@dataclass
class Stage6QueryPred:
    bg: Stage6BranchQueryPred
    rigid: Optional[Stage6BranchQueryPred] = None
    aux: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_bg_hat(self) -> torch.Tensor:
        return self.bg.event_hat

    @property
    def visible_logit_bg(self) -> torch.Tensor:
        return self.bg.visible_logit

    @property
    def support_log_bg_hat(self) -> torch.Tensor:
        return self.bg.support_log_hat

    @property
    def obs_code_bg_hat(self) -> torch.Tensor:
        return self.bg.obs_code_hat


@dataclass
class _BranchRows:
    tokens: torch.Tensor
    proto: torch.Tensor
    global_token: torch.Tensor
    valid_count: torch.Tensor


class BranchViewSetMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        view_code_dim: int,
        num_tokens: int,
        token_dim: int,
        proto_dim: int,
        global_dim: int,
        ctx_dim: int,
        hidden_dim: int,
        branch_name: str,
        zero_unseen_ctx: bool = False,
    ) -> None:
        super().__init__()
        if int(token_dim) != int(ctx_dim):
            raise ValueError("BranchViewSetMemory requires token_dim == ctx_dim")
        self.branch_name = str(branch_name)
        self.event_dim = int(event_dim)
        self.view_code_dim = int(view_code_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.proto_dim = int(proto_dim)
        self.global_dim = int(global_dim)
        self.ctx_dim = int(ctx_dim)
        self.zero_unseen_ctx = bool(zero_unseen_ctx)

        in_dim = int(event_dim) + int(view_code_dim) + 2
        self.router = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_tokens)),
        )
        self.token_update = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(token_dim)),
        )
        self.token_gate = nn.Linear(in_dim, int(num_tokens))
        self.proto_update = nn.Sequential(
            nn.Linear(int(view_code_dim) + 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(proto_dim)),
        )
        self.global_update = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(global_dim)),
        )
        self.global_gate = nn.Linear(in_dim, 1)
        self.query_router = nn.Sequential(
            nn.Linear(int(view_code_dim) + int(global_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_tokens)),
        )
        self.ctx_norm = nn.LayerNorm(int(token_dim))
        self.global_to_ctx = nn.Linear(int(global_dim), int(ctx_dim))
        if int(global_dim) == int(ctx_dim):
            nn.init.zeros_(self.global_to_ctx.weight)
            nn.init.zeros_(self.global_to_ctx.bias)

    def init_rows(self, *, num_rows: int, device: torch.device, dtype: torch.dtype) -> _BranchRows:
        return _BranchRows(
            tokens=torch.zeros((int(num_rows), self.num_tokens, self.token_dim), device=device, dtype=dtype),
            proto=torch.zeros((int(num_rows), self.num_tokens, self.proto_dim), device=device, dtype=dtype),
            global_token=torch.zeros((int(num_rows), self.global_dim), device=device, dtype=dtype),
            valid_count=torch.zeros((int(num_rows), 1), device=device, dtype=dtype),
        )

    def _coerce_view_code(self, view_code: Optional[torch.Tensor], event: torch.Tensor) -> torch.Tensor:
        n = int(event.shape[0])
        if view_code is None:
            return event.new_zeros((n, self.view_code_dim))
        if view_code.dim() != 2 or int(view_code.shape[0]) != n or int(view_code.shape[1]) != self.view_code_dim:
            raise ValueError(
                f"{self.branch_name} view_code must be [N,{self.view_code_dim}], got {tuple(view_code.shape)}"
            )
        return view_code.to(device=event.device, dtype=event.dtype)

    def _coerce_valid_support(
        self,
        *,
        event: torch.Tensor,
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(event.shape[0])
        if valid is None:
            valid_out = event.new_ones((n, 1))
        else:
            valid_out = valid.reshape(n, 1).to(device=event.device, dtype=event.dtype)
        if support is None:
            support_log = valid_out
        else:
            support_log = torch.log1p(
                support.reshape(n, 1).to(device=event.device, dtype=event.dtype).clamp_min(0.0)
            )
        return valid_out.clamp(0.0, 1.0), support_log

    def update_rows(
        self,
        rows: _BranchRows,
        *,
        event: torch.Tensor,
        view_code: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
    ) -> _BranchRows:
        if event.dim() != 2 or int(event.shape[1]) != self.event_dim:
            raise ValueError(f"{self.branch_name} event must be [N,{self.event_dim}], got {tuple(event.shape)}")
        n = int(event.shape[0])
        if int(rows.tokens.shape[0]) != n:
            raise ValueError(
                f"{self.branch_name} VSM row mismatch: state={int(rows.tokens.shape[0])} event={n}"
            )
        if n == 0:
            return rows
        view = self._coerce_view_code(view_code, event)
        valid_f, support_log = self._coerce_valid_support(event=event, valid=valid, support=support)
        x = torch.cat([event, view, support_log, valid_f], dim=-1)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"Stage6ViewSetMemory {self.branch_name} update input contains NaN/Inf")
        assign = torch.softmax(self.router(x), dim=-1)
        gate = torch.sigmoid(self.token_gate(x)).unsqueeze(-1)
        token_prop = self.token_update(x).unsqueeze(1)
        valid3 = valid_f.unsqueeze(-1)
        tokens = rows.tokens + valid3 * assign.unsqueeze(-1) * gate * (token_prop - rows.tokens)

        proto_x = torch.cat([view, support_log, valid_f], dim=-1)
        proto_prop = self.proto_update(proto_x).unsqueeze(1)
        proto = rows.proto + valid3 * assign.unsqueeze(-1) * (proto_prop - rows.proto)

        global_prop = self.global_update(x)
        global_gate = torch.sigmoid(self.global_gate(x))
        global_token = rows.global_token + valid_f * global_gate * (global_prop - rows.global_token)
        valid_count = rows.valid_count + valid_f
        return _BranchRows(tokens=tokens, proto=proto, global_token=global_token, valid_count=valid_count)

    def query_rows(
        self,
        rows: _BranchRows,
        *,
        view_code: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        n = int(rows.tokens.shape[0])
        if n == 0:
            ctx = rows.tokens.new_zeros((0, self.ctx_dim))
            return ctx, {
                "vsm_ctx_norm": 0.0,
                "vsm_router_entropy": 0.0,
                "vsm_token_usage_mean": 0.0,
                "vsm_update_count_mean": 0.0,
                "vsm_seen_ratio": 0.0,
            }
        if view_code is None:
            view = rows.tokens.new_zeros((n, self.view_code_dim))
        else:
            view = view_code.to(device=rows.tokens.device, dtype=rows.tokens.dtype)
            if view.dim() != 2 or int(view.shape[0]) != n or int(view.shape[1]) != self.view_code_dim:
                raise ValueError(
                    f"{self.branch_name} query view_code must be [N,{self.view_code_dim}], got {tuple(view.shape)}"
                )
        q = torch.cat([view, rows.global_token], dim=-1)
        attn = torch.softmax(self.query_router(q), dim=-1)
        token_ctx = (attn.unsqueeze(-1) * rows.tokens).sum(dim=1)
        ctx = self.ctx_norm(token_ctx) + self.global_to_ctx(rows.global_token)
        seen = (rows.valid_count > 0).to(device=ctx.device, dtype=ctx.dtype)
        if self.zero_unseen_ctx:
            ctx = ctx * seen
        if not torch.isfinite(ctx).all():
            raise RuntimeError(f"Stage6ViewSetMemory {self.branch_name} query ctx contains NaN/Inf")
        entropy = -(attn * attn.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        usage = attn.mean(dim=0).max()
        return ctx, {
            "vsm_ctx_norm": float(ctx.detach().norm(dim=-1).mean().item()) if ctx.numel() else 0.0,
            "vsm_router_entropy": float(entropy.detach().item()),
            "vsm_token_usage_mean": float(usage.detach().item()),
            "vsm_update_count_mean": float(rows.valid_count.detach().mean().item()) if rows.valid_count.numel() else 0.0,
            "vsm_seen_ratio": float(seen.detach().mean().item()) if seen.numel() else 0.0,
        }


def _prefix_aux(aux: Dict[str, float], prefix: str, *, include_legacy: bool = False) -> Dict[str, float]:
    out = {f"{prefix}_{key}": float(value) for key, value in aux.items()}
    if include_legacy:
        out.update({key: float(value) for key, value in aux.items()})
    return out


class Stage6ViewSetMemory(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 48,
        view_code_dim: int = 2,
        num_tokens: int = 4,
        token_dim: int = 48,
        proto_dim: int = 8,
        global_dim: int = 48,
        ctx_dim: int = 48,
        hidden_dim: int = 96,
        bg_zero_unseen_ctx: bool = False,
        rigid_zero_unseen_ctx: bool = True,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.view_code_dim = int(view_code_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.proto_dim = int(proto_dim)
        self.global_dim = int(global_dim)
        self.ctx_dim = int(ctx_dim)
        self.bg_memory = BranchViewSetMemory(
            event_dim=event_dim,
            view_code_dim=view_code_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            proto_dim=proto_dim,
            global_dim=global_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            branch_name="bg",
            zero_unseen_ctx=bool(bg_zero_unseen_ctx),
        )
        self.rigid_memory = BranchViewSetMemory(
            event_dim=event_dim,
            view_code_dim=view_code_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            proto_dim=proto_dim,
            global_dim=global_dim,
            ctx_dim=ctx_dim,
            hidden_dim=hidden_dim,
            branch_name="rigid",
            zero_unseen_ctx=bool(rigid_zero_unseen_ctx),
        )

    def init_state(
        self,
        *,
        num_bg: int,
        num_rigid: int = 0,
        device: torch.device,
        dtype: torch.dtype,
        episode_id: int = -1,
        written_refs: Optional[Set[ImageRef]] = None,
    ) -> Stage6VSMState:
        bg = self.bg_memory.init_rows(num_rows=int(num_bg), device=device, dtype=dtype)
        rigid = self.rigid_memory.init_rows(num_rows=int(num_rigid), device=device, dtype=dtype)
        return Stage6VSMState(
            tokens_bg=bg.tokens,
            proto_bg=bg.proto,
            global_bg=bg.global_token,
            valid_count_bg=bg.valid_count,
            tokens_rigid=rigid.tokens,
            proto_rigid=rigid.proto,
            global_rigid=rigid.global_token,
            valid_count_rigid=rigid.valid_count,
            written_refs=set(written_refs or set()),
            episode_id=int(episode_id),
        )

    @staticmethod
    def _bg_rows(state: Stage6VSMState) -> _BranchRows:
        return _BranchRows(
            tokens=state.tokens_bg,
            proto=state.proto_bg,
            global_token=state.global_bg,
            valid_count=state.valid_count_bg,
        )

    @staticmethod
    def _rigid_rows(state: Stage6VSMState) -> _BranchRows:
        return _BranchRows(
            tokens=state.tokens_rigid,
            proto=state.proto_rigid,
            global_token=state.global_rigid,
            valid_count=state.valid_count_rigid,
        )

    @staticmethod
    def _normalize_indices(indices: torch.Tensor, *, total: int, label: str) -> torch.Tensor:
        if indices.dim() != 1:
            indices = indices.reshape(-1)
        indices = indices.to(dtype=torch.long)
        if int(indices.numel()) == 0:
            return indices
        if int(indices.unique().numel()) != int(indices.numel()):
            raise ValueError(f"{label} contains duplicate rigid row indices")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(total):
            raise ValueError(f"{label} contains out-of-range rigid row indices for total={int(total)}")
        return indices

    def _validate_optional_rigid_rows(
        self,
        *,
        name: str,
        value: Optional[torch.Tensor],
        n: int,
        cols: Optional[int] = None,
    ) -> None:
        if value is None:
            return
        if cols is None and value.dim() == 1 and int(value.shape[0]) == int(n):
            return
        if value.dim() != 2 or int(value.shape[0]) != int(n):
            raise ValueError(f"{name} must have {int(n)} rows to match route.S, got {tuple(value.shape)}")
        if cols is not None and int(value.shape[1]) != int(cols):
            raise ValueError(f"{name} must be [len(route.S),{int(cols)}], got {tuple(value.shape)}")

    def update_bg(
        self,
        *,
        state: Stage6VSMState,
        event_bg: torch.Tensor,
        view_code_bg: Optional[torch.Tensor],
        valid_bg: Optional[torch.Tensor],
        support_bg: Optional[torch.Tensor],
    ) -> Stage6VSMState:
        rows = self.bg_memory.update_rows(
            self._bg_rows(state),
            event=event_bg,
            view_code=view_code_bg,
            valid=valid_bg,
            support=support_bg,
        )
        out = replace(
            state,
            tokens_bg=rows.tokens,
            proto_bg=rows.proto,
            global_bg=rows.global_token,
            valid_count_bg=rows.valid_count,
        )
        out.assert_finite("vsm_after_bg_update")
        return out

    def query_bg(
        self,
        *,
        state: Stage6VSMState,
        view_code_bg: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        ctx, aux = self.bg_memory.query_rows(self._bg_rows(state), view_code=view_code_bg)
        return ctx, _prefix_aux(aux, "vsm_bg", include_legacy=True)

    def update_rigid(
        self,
        *,
        state: Stage6VSMState,
        indices: torch.Tensor,
        event_rigid: torch.Tensor,
        view_code_rigid: Optional[torch.Tensor],
        valid_rigid: Optional[torch.Tensor],
        support_rigid: Optional[torch.Tensor],
    ) -> Stage6VSMState:
        idx = self._normalize_indices(
            indices.to(device=state.tokens_rigid.device),
            total=int(state.tokens_rigid.shape[0]),
            label="route.S",
        )
        if event_rigid.dim() != 2 or int(event_rigid.shape[0]) != int(idx.numel()):
            raise ValueError(
                f"event_rigid must be [len(route.S),C], got event={tuple(event_rigid.shape)} route={int(idx.numel())}"
            )
        self._validate_optional_rigid_rows(name="valid_rigid", value=valid_rigid, n=int(idx.numel()))
        self._validate_optional_rigid_rows(name="support_rigid", value=support_rigid, n=int(idx.numel()))
        self._validate_optional_rigid_rows(
            name="view_code_rigid",
            value=view_code_rigid,
            n=int(idx.numel()),
            cols=self.view_code_dim,
        )
        if int(idx.numel()) == 0:
            return state
        idx_event = idx.to(device=event_rigid.device)
        rows = _BranchRows(
            tokens=state.tokens_rigid.index_select(0, idx).to(device=event_rigid.device, dtype=event_rigid.dtype),
            proto=state.proto_rigid.index_select(0, idx).to(device=event_rigid.device, dtype=event_rigid.dtype),
            global_token=state.global_rigid.index_select(0, idx).to(device=event_rigid.device, dtype=event_rigid.dtype),
            valid_count=state.valid_count_rigid.index_select(0, idx).to(device=event_rigid.device, dtype=event_rigid.dtype),
        )
        updated = self.rigid_memory.update_rows(
            rows,
            event=event_rigid,
            view_code=view_code_rigid,
            valid=valid_rigid,
            support=support_rigid,
        )
        scatter_idx = idx.to(device=state.tokens_rigid.device)
        tokens = state.tokens_rigid.clone()
        proto = state.proto_rigid.clone()
        global_token = state.global_rigid.clone()
        valid_count = state.valid_count_rigid.clone()
        tokens[scatter_idx] = updated.tokens.to(device=tokens.device, dtype=tokens.dtype)
        proto[scatter_idx] = updated.proto.to(device=proto.device, dtype=proto.dtype)
        global_token[scatter_idx] = updated.global_token.to(device=global_token.device, dtype=global_token.dtype)
        valid_count[scatter_idx] = updated.valid_count.to(device=valid_count.device, dtype=valid_count.dtype)
        out = replace(
            state,
            tokens_rigid=tokens,
            proto_rigid=proto,
            global_rigid=global_token,
            valid_count_rigid=valid_count,
        )
        out.assert_finite("vsm_after_rigid_update")
        _ = idx_event
        return out

    def query_rigid(
        self,
        *,
        state: Stage6VSMState,
        indices: torch.Tensor,
        view_code_rigid: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        idx = self._normalize_indices(
            indices.to(device=state.tokens_rigid.device),
            total=int(state.tokens_rigid.shape[0]),
            label="route.S",
        )
        if int(idx.numel()) == 0:
            ctx = state.tokens_rigid.new_zeros((0, self.ctx_dim))
            return ctx, {
                "vsm_rigid_vsm_ctx_norm": 0.0,
                "vsm_rigid_vsm_router_entropy": 0.0,
                "vsm_rigid_vsm_token_usage_mean": 0.0,
                "vsm_rigid_vsm_update_count_mean": 0.0,
                "vsm_rigid_vsm_seen_ratio": 0.0,
            }
        self._validate_optional_rigid_rows(
            name="view_code_rigid",
            value=view_code_rigid,
            n=int(idx.numel()),
            cols=self.view_code_dim,
        )
        rows = _BranchRows(
            tokens=state.tokens_rigid.index_select(0, idx),
            proto=state.proto_rigid.index_select(0, idx),
            global_token=state.global_rigid.index_select(0, idx),
            valid_count=state.valid_count_rigid.index_select(0, idx),
        )
        ctx, aux = self.rigid_memory.query_rows(rows, view_code=view_code_rigid)
        return ctx, _prefix_aux(aux, "vsm_rigid", include_legacy=False)

    def update(
        self,
        *,
        state: Stage6VSMState,
        event_bg: torch.Tensor,
        view_code_bg: Optional[torch.Tensor],
        valid_bg: Optional[torch.Tensor],
        support_bg: Optional[torch.Tensor],
    ) -> Stage6VSMState:
        return self.update_bg(
            state=state,
            event_bg=event_bg,
            view_code_bg=view_code_bg,
            valid_bg=valid_bg,
            support_bg=support_bg,
        )

    def query(
        self,
        *,
        state: Stage6VSMState,
        view_code_bg: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        return self.query_bg(state=state, view_code_bg=view_code_bg)


class _BranchQueryDecoder(nn.Module):
    def __init__(self, *, input_dim: int, event_dim: int, obs_code_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.event_head = nn.Linear(int(hidden_dim), int(event_dim))
        self.visible_head = nn.Linear(int(hidden_dim), 1)
        self.support_head = nn.Linear(int(hidden_dim), 1)
        self.obs_code_head = nn.Linear(int(hidden_dim), int(obs_code_dim))

    def forward(self, ctx: torch.Tensor) -> Stage6BranchQueryPred:
        h = self.net(ctx)
        return Stage6BranchQueryPred(
            event_hat=self.event_head(h),
            visible_logit=self.visible_head(h),
            support_log_hat=self.support_head(h),
            obs_code_hat=self.obs_code_head(h),
        )


class Stage6QueryDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 48,
        event_dim: int = 48,
        obs_code_dim: int = 2,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.event_dim = int(event_dim)
        self.obs_code_dim = int(obs_code_dim)
        self.bg_decoder = _BranchQueryDecoder(
            input_dim=input_dim,
            event_dim=event_dim,
            obs_code_dim=obs_code_dim,
            hidden_dim=hidden_dim,
        )
        self.rigid_decoder = _BranchQueryDecoder(
            input_dim=input_dim,
            event_dim=event_dim,
            obs_code_dim=obs_code_dim,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def _assert_pred_finite(pred: Stage6BranchQueryPred, label: str) -> None:
        for name, value in pred.__dict__.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"Stage6QueryDecoder {label}.{name} contains NaN/Inf")

    def forward(
        self,
        *,
        state: Stage6VSMState,
        query_view_code_bg: Optional[torch.Tensor],
        memory: Stage6ViewSetMemory,
        query_view_code_rigid: Optional[torch.Tensor] = None,
        rigid_indices: Optional[torch.Tensor] = None,
    ) -> Stage6QueryPred:
        ctx_bg, aux_bg = memory.query_bg(state=state, view_code_bg=query_view_code_bg)
        if ctx_bg.dim() != 2 or int(ctx_bg.shape[1]) != self.input_dim:
            raise ValueError(f"query bg ctx must be [N,{self.input_dim}], got {tuple(ctx_bg.shape)}")
        pred_bg = self.bg_decoder(ctx_bg)
        self._assert_pred_finite(pred_bg, "bg")

        pred_rigid: Optional[Stage6BranchQueryPred] = None
        aux: Dict[str, Any] = dict(aux_bg)
        if rigid_indices is not None:
            ctx_rigid, aux_rigid = memory.query_rigid(
                state=state,
                indices=rigid_indices,
                view_code_rigid=query_view_code_rigid,
            )
            if ctx_rigid.dim() != 2 or int(ctx_rigid.shape[1]) != self.input_dim:
                raise ValueError(f"query rigid ctx must be [N,{self.input_dim}], got {tuple(ctx_rigid.shape)}")
            pred_rigid = self.rigid_decoder(ctx_rigid)
            self._assert_pred_finite(pred_rigid, "rigid")
            aux.update(aux_rigid)
        return Stage6QueryPred(bg=pred_bg, rigid=pred_rigid, aux=aux)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"smooth_l1 shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()
    m = mask.to(device=pred.device, dtype=pred.dtype)
    while m.dim() < loss.dim():
        m = m.unsqueeze(-1)
    denom = m.sum().clamp_min(1.0) * float(loss.shape[-1])
    return (loss * m).sum() / denom


__all__ = [
    "BranchViewSetMemory",
    "Stage6BranchQueryPred",
    "Stage6QueryDecoder",
    "Stage6QueryPred",
    "Stage6VSMState",
    "Stage6ViewSetMemory",
    "masked_smooth_l1",
]
