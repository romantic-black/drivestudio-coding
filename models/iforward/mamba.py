from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StreamingMambaCellState:
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor

    def detach(self) -> "StreamingMambaCellState":
        return StreamingMambaCellState(
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
        )


class StreamingMambaCell(nn.Module):
    """Small self-contained streaming selective-SSM cell.

    This is intentionally a single-step cell: callers own token routing and
    carry the returned `conv_state`/`ssm_state` across rollout steps.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        dt_rank: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        if self.conv_kernel < 1:
            raise ValueError("StreamingMambaCell conv_kernel must be >= 1")
        self.dt_rank = int(dt_rank or max(4, int(model_dim) // 16))
        self.output_dim = int(output_dim or input_dim)

        self.in_proj = nn.Linear(self.input_dim, 2 * self.model_dim)
        self.conv_weight = nn.Parameter(torch.empty(self.model_dim, self.conv_kernel))
        self.conv_bias = nn.Parameter(torch.zeros(self.model_dim))
        self.x_proj = nn.Linear(self.model_dim, self.dt_rank + 2 * self.state_dim)
        self.dt_proj = nn.Linear(self.dt_rank, self.model_dim)
        self.A_log = nn.Parameter(torch.empty(self.model_dim, self.state_dim))
        self.D = nn.Parameter(torch.ones(self.model_dim))
        self.out_proj = nn.Linear(self.model_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.conv_weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.constant_(self.dt_proj.bias, -3.0)
        nn.init.uniform_(self.A_log, a=-3.0, b=-1.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def init_state(
        self,
        num_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StreamingMambaCellState:
        return StreamingMambaCellState(
            conv_state=torch.zeros(
                int(num_tokens),
                self.model_dim,
                self.conv_kernel,
                device=device,
                dtype=dtype,
            ),
            ssm_state=torch.zeros(
                int(num_tokens),
                self.model_dim,
                self.state_dim,
                device=device,
                dtype=dtype,
            ),
            seen=torch.zeros(int(num_tokens), device=device, dtype=torch.bool),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: StreamingMambaCellState,
        *,
        write_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, StreamingMambaCellState]:
        if x.dim() != 2 or int(x.shape[-1]) != self.input_dim:
            raise ValueError(f"StreamingMambaCell expected x [N,{self.input_dim}], got {tuple(x.shape)}")
        n = int(x.shape[0])
        if int(state.conv_state.shape[0]) != n or int(state.ssm_state.shape[0]) != n:
            raise ValueError("StreamingMambaCell state row count must match x.")
        if write_mask is None:
            mask = torch.ones(n, device=x.device, dtype=torch.bool)
        else:
            mask = write_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
            if int(mask.numel()) != n:
                raise ValueError("StreamingMambaCell write_mask row count mismatch.")

        xz = self.in_proj(x)
        u_raw, z = xz.chunk(2, dim=-1)
        next_conv = torch.cat([state.conv_state[:, :, 1:], u_raw.unsqueeze(-1)], dim=-1)
        conv_u = (next_conv * self.conv_weight.to(dtype=x.dtype).unsqueeze(0)).sum(dim=-1)
        conv_u = F.silu(conv_u + self.conv_bias.to(dtype=x.dtype))

        projected = self.x_proj(conv_u)
        dt_raw = projected[:, : self.dt_rank]
        b_raw = projected[:, self.dt_rank : self.dt_rank + self.state_dim]
        c_raw = projected[:, self.dt_rank + self.state_dim :]
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(max=10.0)
        a = -torch.exp(self.A_log.to(dtype=x.dtype)).unsqueeze(0)
        d_a = torch.exp(dt.unsqueeze(-1) * a).clamp(min=0.0, max=1.0)
        d_b = dt.unsqueeze(-1) * b_raw.unsqueeze(1)
        next_ssm = d_a * state.ssm_state + d_b * conv_u.unsqueeze(-1)
        y = (next_ssm * c_raw.unsqueeze(1)).sum(dim=-1) + self.D.to(dtype=x.dtype).unsqueeze(0) * conv_u
        y = y * torch.sigmoid(z)
        out = self.out_proj(y)

        mask3 = mask[:, None, None]
        final_conv = torch.where(mask3, next_conv, state.conv_state)
        final_ssm = torch.where(mask3, next_ssm, state.ssm_state)
        final_seen = torch.where(mask, torch.ones_like(state.seen, dtype=torch.bool), state.seen)
        new_state = StreamingMambaCellState(conv_state=final_conv, ssm_state=final_ssm, seen=final_seen)
        if not torch.isfinite(out).all():
            raise RuntimeError("StreamingMambaCell output contains NaN/Inf")
        if not torch.isfinite(new_state.conv_state).all() or not torch.isfinite(new_state.ssm_state).all():
            raise RuntimeError("StreamingMambaCell state contains NaN/Inf")
        return out, new_state
