from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState

from .history_ema import IForwardHistoryBranchEMA, IForwardHistoryEMAState


GATE_ATTRS = ("means", "scales", "quat", "opacity", "sh")
BRANCH_IDS = {"bg": 0, "distant": 1, "rigid": 2}


@dataclass
class IForwardAttributeGate:
    means: torch.Tensor
    scales: torch.Tensor
    quat: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    hidden: torch.Tensor


@dataclass
class IForwardGatePack:
    bg: IForwardAttributeGate
    distant: Optional[IForwardAttributeGate] = None
    rigid: Optional[IForwardAttributeGate] = None
    aux: Optional[Dict[str, float]] = None


def _col(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if value is None:
        return ref.new_full((int(n), 1), float(default))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 gate column row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
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
        raise ValueError(f"IForward v3 gate bool row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) != 1:
        out = out.reshape(int(n), -1).any(dim=-1, keepdim=True)
    return out.to(dtype=torch.bool)


def _obs2(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor) -> torch.Tensor:
    if value is None:
        return ref.new_zeros((int(n), 2))
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 obs_code row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) >= 2:
        return out[:, :2]
    return torch.cat([out, out.new_zeros((int(n), 2 - int(out.shape[1])))], dim=-1)


class IForwardHistoryGate(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        ctx_dim: int,
        history_embed_dim: int = 16,
        hidden_dim: int = 64,
        branch_embed_dim: int = 8,
        min_gate: Optional[Dict[str, float]] = None,
        init_bias: Optional[Dict[str, float]] = None,
        branch_bias: Optional[Dict[str, Dict[str, float]]] = None,
        hidden_gate_weights: Optional[Dict[str, float]] = None,
        cold_open_uninitialized: bool = True,
        bind_with_mask_update: bool = True,
        support_min: Optional[Dict[str, float]] = None,
        grad_feature_dim: int = 0,
        grad_embed_dim: int = 16,
        grad_prior_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.ctx_dim = int(ctx_dim)
        self.grad_feature_dim = int(grad_feature_dim)
        self.cold_open_uninitialized = bool(cold_open_uninitialized)
        self.bind_with_mask_update = bool(bind_with_mask_update)
        self.support_min = {
            "bg": float((support_min or {}).get("bg", 0.0)),
            "distant": float((support_min or {}).get("distant", 0.0)),
            "rigid": float((support_min or {}).get("rigid", 0.0)),
        }
        self.history_proj = nn.Sequential(
            nn.Linear(12, int(history_embed_dim)),
            nn.LayerNorm(int(history_embed_dim)),
            nn.GELU(),
        )
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        input_dim = int(event_dim) + int(ctx_dim) + int(history_embed_dim) + int(branch_embed_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), len(GATE_ATTRS)),
        )
        if int(self.grad_feature_dim) > 0:
            self.grad_prior_embed = nn.Sequential(
                nn.Linear(int(self.grad_feature_dim), int(grad_embed_dim)),
                nn.LayerNorm(int(grad_embed_dim)),
                nn.GELU(),
                nn.Linear(int(grad_embed_dim), int(grad_embed_dim)),
                nn.GELU(),
            )
            self.grad_prior_logits = nn.Linear(int(grad_embed_dim), len(GATE_ATTRS))
            self.grad_prior_scale = nn.Parameter(torch.tensor(float(grad_prior_scale_init), dtype=torch.float32))
        else:
            self.grad_prior_embed = None
            self.grad_prior_logits = None
            self.grad_prior_scale = None
        min_gate = dict(min_gate or {})
        self.register_buffer(
            "min_gate_values",
            torch.tensor([float(min_gate.get(name, 0.0)) for name in GATE_ATTRS], dtype=torch.float32),
        )
        init_bias = dict(init_bias or {})
        final = self.gate_mlp[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias.copy_(torch.tensor([float(init_bias.get(name, 0.0)) for name in GATE_ATTRS], dtype=final.bias.dtype))

        branch_bias = dict(branch_bias or {})
        bias_rows = []
        for branch in ("bg", "distant", "rigid"):
            cfg = dict(branch_bias.get(branch, {}) or {})
            bias_rows.append([float(cfg.get(name, 0.0)) for name in GATE_ATTRS])
        self.register_buffer("branch_bias_table", torch.tensor(bias_rows, dtype=torch.float32))
        hidden_gate_weights = dict(hidden_gate_weights or {})
        self.register_buffer(
            "hidden_gate_weights",
            torch.tensor(
                [
                    float(hidden_gate_weights.get("means", 0.2)),
                    float(hidden_gate_weights.get("scales", 0.0)),
                    float(hidden_gate_weights.get("quat", 0.0)),
                    float(hidden_gate_weights.get("opacity", 0.3)),
                    float(hidden_gate_weights.get("sh", 0.5)),
                ],
                dtype=torch.float32,
            ),
        )

    def _grad_prior_logit_delta(self, grad_features: Optional[torch.Tensor], *, ref: torch.Tensor, n: int) -> torch.Tensor:
        if (
            grad_features is None
            or self.grad_prior_embed is None
            or self.grad_prior_logits is None
            or self.grad_prior_scale is None
        ):
            return ref.new_zeros((int(n), len(GATE_ATTRS)))
        features = grad_features.to(device=ref.device, dtype=ref.dtype)
        if features.dim() != 2 or int(features.shape[0]) != int(n) or int(features.shape[1]) != int(self.grad_feature_dim):
            raise ValueError(
                "IForward HGV2 grad feature mismatch: "
                f"got {tuple(features.shape)}, expected ({int(n)}, {int(self.grad_feature_dim)})"
            )
        scale = self.grad_prior_scale.to(device=ref.device, dtype=ref.dtype)
        return scale * self.grad_prior_logits(self.grad_prior_embed(features))

    @staticmethod
    def empty_gate(ref: torch.Tensor) -> IForwardAttributeGate:
        z = ref.new_zeros((0, 1))
        return IForwardAttributeGate(means=z, scales=z, quat=z, opacity=z, sh=z, hidden=z)

    @staticmethod
    def ones_gate(event_x: torch.Tensor) -> IForwardAttributeGate:
        one = event_x.new_ones((int(event_x.shape[0]), 1))
        return IForwardAttributeGate(means=one, scales=one, quat=one, opacity=one, sh=one, hidden=one)

    def _history_raw(
        self,
        *,
        branch_state: IForwardHistoryBranchEMA,
        rows: Optional[torch.Tensor],
        event_x: torch.Tensor,
        support_now: Optional[torch.Tensor],
        valid_now: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(event_x.shape[0])
        hist = branch_state.select(rows)
        support = _col(support_now, n=n, ref=event_x, default=1.0)
        valid_bool = _bool_col(valid_now, n=n, ref=event_x, default=True)
        valid = valid_bool.to(dtype=event_x.dtype)
        visible_now = ((support >= float(self.support_min[str(branch)])) & valid_bool).to(dtype=event_x.dtype)
        obs = _obs2(obs_code, n=n, ref=event_x)
        raw = torch.cat(
            [
                hist["support_fast"].to(device=event_x.device, dtype=event_x.dtype),
                hist["error_fast"].to(device=event_x.device, dtype=event_x.dtype),
                hist["update_norm_fast"].to(device=event_x.device, dtype=event_x.dtype),
                hist["support_slow"].to(device=event_x.device, dtype=event_x.dtype),
                hist["error_slow"].to(device=event_x.device, dtype=event_x.dtype),
                hist["update_norm_slow"].to(device=event_x.device, dtype=event_x.dtype),
                hist["initialized"].to(device=event_x.device, dtype=event_x.dtype),
                visible_now,
                torch.log1p(support.clamp_min(0.0)),
                valid,
                obs,
            ],
            dim=-1,
        )
        if int(raw.shape[-1]) != 12:
            raise RuntimeError(f"IForward v3 history_raw dim mismatch: expected 12, got {int(raw.shape[-1])}")
        initialized = hist["initialized"].to(device=event_x.device, dtype=event_x.dtype)
        mask_update = visible_now.to(dtype=event_x.dtype)
        return raw, initialized, mask_update, support

    def branch_gate(
        self,
        *,
        branch: str,
        event_x: Optional[torch.Tensor],
        ctx: Optional[torch.Tensor],
        branch_state: Optional[IForwardHistoryBranchEMA],
        rows: Optional[torch.Tensor],
        support_now: Optional[torch.Tensor],
        valid_now: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        grad_features: Optional[torch.Tensor] = None,
        ablation: str = "full",
    ) -> tuple[Optional[IForwardAttributeGate], Dict[str, float]]:
        if event_x is None:
            return None, {}
        n = int(event_x.shape[0])
        if n == 0:
            return self.empty_gate(event_x), {f"v3/gate/{branch}_rows": 0.0}
        if branch_state is None:
            raise ValueError(f"IForward v3 history gate missing history branch {branch!r}.")
        raw, initialized, mask_update, support = self._history_raw(
            branch_state=branch_state,
            rows=rows,
            event_x=event_x,
            support_now=support_now,
            valid_now=valid_now,
            obs_code=obs_code,
            branch=str(branch),
        )
        if str(ablation) == "no_history_gate":
            gate = event_x.new_ones((n, len(GATE_ATTRS)))
            if bool(self.bind_with_mask_update):
                gate = gate * mask_update.expand_as(gate)
            hidden_weights = self.hidden_gate_weights.to(device=event_x.device, dtype=event_x.dtype).view(1, -1)
            hidden = (gate * hidden_weights).sum(dim=-1, keepdim=True)
            out = IForwardAttributeGate(
                means=gate[:, 0:1],
                scales=gate[:, 1:2],
                quat=gate[:, 2:3],
                opacity=gate[:, 3:4],
                sh=gate[:, 4:5],
                hidden=hidden,
            )
            return out, {
                f"v3/gate/{branch}_rows": float(n),
                f"v3/gate/{branch}_means_mean": float(out.means.detach().mean().item()),
                f"v3/gate/{branch}_mask_update_ratio": float(mask_update.detach().mean().item()),
                f"v3/gate/{branch}_bypass": 1.0,
            }
        if ctx is None:
            ctx = event_x.new_zeros((n, int(self.ctx_dim)))
        if int(ctx.shape[0]) != n or int(ctx.shape[1]) != int(self.ctx_dim):
            raise ValueError(f"IForward v3 ctx for {branch} must be [N,{self.ctx_dim}], got {tuple(ctx.shape)}")
        hist_embed = self.history_proj(raw)
        branch_id = int(BRANCH_IDS[str(branch)])
        branch_ids = torch.full((n,), branch_id, device=event_x.device, dtype=torch.long)
        x = torch.cat([event_x, ctx.to(device=event_x.device, dtype=event_x.dtype), hist_embed, self.branch_embed(branch_ids)], dim=-1)
        logits = self.gate_mlp(x)
        logits = logits + self._grad_prior_logit_delta(grad_features, ref=event_x, n=n)
        logits = logits + self.branch_bias_table[branch_id].to(device=event_x.device, dtype=event_x.dtype).view(1, -1)
        raw_gate = torch.sigmoid(logits)
        min_gate = self.min_gate_values.to(device=event_x.device, dtype=event_x.dtype).view(1, -1)
        gate = min_gate + (1.0 - min_gate) * raw_gate
        cold = initialized <= 0
        if bool(self.cold_open_uninitialized):
            gate = torch.where(cold.expand_as(gate), torch.ones_like(gate), gate)
        if bool(self.bind_with_mask_update):
            gate = gate * mask_update.expand_as(gate)
        hidden_weights = self.hidden_gate_weights.to(device=event_x.device, dtype=event_x.dtype).view(1, -1)
        hidden = (gate * hidden_weights).sum(dim=-1, keepdim=True)
        out = IForwardAttributeGate(
            means=gate[:, 0:1],
            scales=gate[:, 1:2],
            quat=gate[:, 2:3],
            opacity=gate[:, 3:4],
            sh=gate[:, 4:5],
            hidden=hidden,
        )
        aux = {
            f"v3/gate/{branch}_rows": float(n),
            f"v3/gate/{branch}_means_mean": float(out.means.detach().mean().item()),
            f"v3/gate/{branch}_scales_mean": float(out.scales.detach().mean().item()),
            f"v3/gate/{branch}_quat_mean": float(out.quat.detach().mean().item()),
            f"v3/gate/{branch}_opacity_mean": float(out.opacity.detach().mean().item()),
            f"v3/gate/{branch}_sh_mean": float(out.sh.detach().mean().item()),
            f"v3/gate/{branch}_hidden_mean": float(out.hidden.detach().mean().item()),
            f"v3/gate/{branch}_mask_update_ratio": float(mask_update.detach().mean().item()),
            f"v3/gate/{branch}_cold_open_ratio": float(cold.detach().to(dtype=torch.float32).mean().item()),
            f"v3/gate/{branch}_support_now_mean": float(support.detach().mean().item()),
        }
        return out, aux

    def forward(
        self,
        *,
        event: EventPack,
        ctx_memory: ContextPack,
        history_ema: IForwardHistoryEMAState,
        local_state: LocalGSState,
        grad_features: Optional[Any] = None,
        ablation: str = "full",
    ) -> IForwardGatePack:
        aux: Dict[str, float] = {}
        bg, bg_aux = self.branch_gate(
            branch="bg",
            event_x=event.event_bg,
            ctx=getattr(ctx_memory, "ctx_bg", None),
            branch_state=history_ema.bg,
            rows=None,
            support_now=event.support_bg,
            valid_now=event.valid_bg,
            obs_code=event.obs_code_bg,
            grad_features=None if grad_features is None else getattr(getattr(grad_features, "bg", None), "features", None),
            ablation=str(ablation),
        )
        if bg is None:
            raise RuntimeError("IForward v3 requires bg gate.")
        aux.update(bg_aux)
        distant = None
        if local_state.distant is not None and event.event_distant is not None and history_ema.distant is not None:
            distant, d_aux = self.branch_gate(
                branch="distant",
                event_x=event.event_distant,
                ctx=getattr(ctx_memory, "ctx_distant", None),
                branch_state=history_ema.distant,
                rows=None,
                support_now=event.support_distant,
                valid_now=event.valid_distant,
                obs_code=event.obs_code_distant,
                grad_features=(
                    None if grad_features is None else getattr(getattr(grad_features, "distant", None), "features", None)
                ),
                ablation=str(ablation),
            )
            aux.update(d_aux)
        rigid = None
        route = getattr(event, "route", None)
        rows = getattr(route, "S", None) if route is not None else None
        if local_state.rigid is not None and event.event_rigid is not None and history_ema.rigid is not None and rows is not None:
            rigid, r_aux = self.branch_gate(
                branch="rigid",
                event_x=event.event_rigid,
                ctx=getattr(ctx_memory, "ctx_rigid", None),
                branch_state=history_ema.rigid,
                rows=rows,
                support_now=event.support_rigid,
                valid_now=event.valid_rigid,
                obs_code=event.obs_code_rigid,
                grad_features=None if grad_features is None else getattr(getattr(grad_features, "rigid", None), "features", None),
                ablation=str(ablation),
            )
            aux.update(r_aux)
        return IForwardGatePack(bg=bg, distant=distant, rigid=rigid, aux=aux)


__all__ = [
    "IForwardAttributeGate",
    "IForwardGatePack",
    "IForwardHistoryGate",
]
