from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TemporalMotionEmbedding(nn.Module):
    def __init__(self, *, output_dim: int = 16, hidden_dim: int = 32, num_frequencies: int = 4) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_frequencies = int(num_frequencies)
        self.visit_kind_embed = nn.Embedding(4, 4)
        in_dim = 8 * (1 + 2 * int(num_frequencies)) + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )
        nn.init.zeros_(self.net[-1].bias)

    def _visit_kind_ids(self, visit_kind: str | torch.Tensor | int | None, *, n: int, device: torch.device) -> torch.Tensor:
        if visit_kind is None:
            return torch.zeros((n,), device=device, dtype=torch.long)
        if torch.is_tensor(visit_kind):
            ids = visit_kind.to(device=device, dtype=torch.long).reshape(-1)
            return ids.expand(n) if int(ids.numel()) == 1 else ids
        if isinstance(visit_kind, int):
            return torch.full((n,), int(visit_kind), device=device, dtype=torch.long)
        table = {"bootstrap": 0, "causal": 1, "causal_first": 1, "repair": 2, "refinement": 2, "stress": 3}
        return torch.full((n,), int(table.get(str(visit_kind), 0)), device=device, dtype=torch.long)

    def forward(
        self,
        *,
        num_rows: int,
        ref: torch.Tensor,
        delta_t_sec: float | torch.Tensor = 0.0,
        gap: float | torch.Tensor = 0.0,
        ego_delta_translation: Optional[torch.Tensor] = None,
        ego_delta_yaw: float | torch.Tensor = 0.0,
        seen_flag: Optional[torch.Tensor] = None,
        visit_kind: str | torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        n = int(num_rows)
        if n == 0:
            return ref.new_zeros((0, int(self.output_dim)))
        device = ref.device
        dtype = ref.dtype
        dt = torch.as_tensor(delta_t_sec, device=device, dtype=dtype).reshape(-1)
        if int(dt.numel()) == 1:
            dt = dt.expand(n)
        gap_t = torch.as_tensor(gap, device=device, dtype=dtype).reshape(-1)
        if int(gap_t.numel()) == 1:
            gap_t = gap_t.expand(n)
        if ego_delta_translation is None:
            trans = ref.new_zeros((n, 3))
        else:
            trans = ego_delta_translation.to(device=device, dtype=dtype).reshape(-1, 3)
            if int(trans.shape[0]) == 1:
                trans = trans.expand(n, 3)
        yaw = torch.as_tensor(ego_delta_yaw, device=device, dtype=dtype).reshape(-1)
        if int(yaw.numel()) == 1:
            yaw = yaw.expand(n)
        seen = torch.zeros((n,), device=device, dtype=dtype) if seen_flag is None else seen_flag.to(device=device, dtype=dtype).reshape(-1)
        if int(seen.numel()) == 1:
            seen = seen.expand(n)
        visit_ids = self._visit_kind_ids(visit_kind, n=n, device=device).clamp(min=0, max=3)
        if torch.any(visit_ids == 2):
            repair = (visit_ids == 2).to(dtype=dtype)
            dt = torch.where(repair > 0, torch.zeros_like(dt), dt)
            gap_t = torch.where(repair > 0, torch.zeros_like(gap_t), gap_t)
            trans = torch.where(repair[:, None] > 0, torch.zeros_like(trans), trans)
            yaw = torch.where(repair > 0, torch.zeros_like(yaw), yaw)
        base = torch.stack([dt, gap_t, trans[:, 0], trans[:, 1], trans[:, 2], torch.sin(yaw), torch.cos(yaw), seen], dim=-1)
        feats = [base]
        freqs = torch.pow(torch.tensor(2.0, device=device, dtype=dtype), torch.arange(self.num_frequencies, device=device, dtype=dtype))
        for f in freqs:
            feats.append(torch.sin(base * f))
            feats.append(torch.cos(base * f))
        visit = self.visit_kind_embed(visit_ids).to(device=device, dtype=dtype)
        return self.net(torch.cat([torch.cat(feats, dim=-1), visit], dim=-1))


__all__ = ["TemporalMotionEmbedding"]
