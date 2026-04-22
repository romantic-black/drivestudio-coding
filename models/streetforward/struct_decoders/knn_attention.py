from __future__ import annotations

import torch
import torch.nn as nn


class EdgeGatedKNNAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        attn_dim: int,
        pos_dim: int,
        residual_scale_init: float,
        chunk_size: int,
        pos_scale: float,
        use_same_branch_flag: bool,
        use_support: bool,
        use_pos_value: bool,
        debug_validate: bool,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if int(attn_dim) <= 0:
            raise ValueError(f"attn_dim must be > 0, got {attn_dim}")
        if int(pos_dim) <= 0:
            raise ValueError(f"pos_dim must be > 0, got {pos_dim}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if float(pos_scale) <= 0.0:
            raise ValueError(f"pos_scale must be > 0, got {pos_scale}")

        self.channels = int(channels)
        self.attn_dim = int(attn_dim)
        self.pos_dim = int(pos_dim)
        self.chunk_size = int(chunk_size)
        self.pos_scale = float(pos_scale)
        self.use_same_branch_flag = bool(use_same_branch_flag)
        self.use_support = bool(use_support)
        self.use_pos_value = bool(use_pos_value)
        self.debug_validate = bool(debug_validate)

        self.norm = nn.LayerNorm(self.channels)
        self.xi_proj = nn.Linear(self.channels, self.attn_dim)
        self.delta_proj = nn.Linear(self.channels, self.attn_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(4, self.pos_dim),
            nn.GELU(),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos_score_proj = nn.Linear(self.pos_dim, self.attn_dim)

        if self.use_support:
            self.support_score_proj = nn.Linear(2, self.attn_dim)
        else:
            self.support_score_proj = None
        if self.use_same_branch_flag:
            self.branch_score_proj = nn.Linear(1, self.attn_dim)
        else:
            self.branch_score_proj = None

        self.score_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.attn_dim, 1),
        )
        self.value_proj = nn.Linear(self.channels, self.channels)
        if self.use_pos_value:
            self.pos_value_proj = nn.Linear(self.pos_dim, self.channels)
        else:
            self.pos_value_proj = None
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

    def _validate_inputs(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        branch_id: torch.Tensor,
        acc_w: torch.Tensor,
        *,
        run_heavy_checks: bool,
    ) -> None:
        if x.dim() != 2:
            raise ValueError(f"x must have shape [N,C], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.channels:
            raise ValueError(f"x channels mismatch: got {x.shape[1]}, expected {self.channels}")
        n = int(x.shape[0])
        if coords.shape != (n, 3):
            raise ValueError(f"coords must have shape [{n},3], got {tuple(coords.shape)}")
        if branch_id.shape != (n,):
            raise ValueError(f"branch_id must have shape [{n}], got {tuple(branch_id.shape)}")
        if acc_w.shape != (n,):
            raise ValueError(f"acc_w must have shape [{n}], got {tuple(acc_w.shape)}")
        if neighbor_idx.dim() != 2:
            raise ValueError(f"neighbor_idx must be [N,K], got {tuple(neighbor_idx.shape)}")
        if neighbor_mask.shape != neighbor_idx.shape:
            raise ValueError(
                f"neighbor_mask shape must match neighbor_idx, got {tuple(neighbor_mask.shape)} vs {tuple(neighbor_idx.shape)}"
            )
        if int(neighbor_idx.shape[0]) != n:
            raise ValueError(f"neighbor_idx first dim must equal N={n}, got {neighbor_idx.shape[0]}")
        if int(neighbor_idx.shape[1]) <= 0:
            raise ValueError("neighbor_idx K must be > 0")
        if neighbor_mask.dtype != torch.bool:
            raise ValueError(f"neighbor_mask must be bool, got {neighbor_mask.dtype}")
        if run_heavy_checks:
            if not bool(neighbor_mask[:, 0].all().item()):
                raise RuntimeError("slot-0 neighbor_mask must be true for all queries.")
            if bool((neighbor_mask.sum(dim=1) <= 0).any().item()):
                raise RuntimeError("each query must have at least one valid neighbor (slot-0 self).")
            if bool((neighbor_idx < 0).any().item()) or bool((neighbor_idx >= n).any().item()):
                raise RuntimeError(f"neighbor_idx out of range [0, {n}).")

    def forward(
        self,
        x: torch.Tensor,
        *,
        coords: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        branch_id: torch.Tensor,
        acc_w: torch.Tensor,
    ) -> torch.Tensor:
        run_heavy_checks = (not self.training) or self.debug_validate
        self._validate_inputs(
            x,
            coords,
            neighbor_idx,
            neighbor_mask,
            branch_id,
            acc_w,
            run_heavy_checks=run_heavy_checks,
        )

        neighbor_idx = neighbor_idx.long().contiguous()
        neighbor_mask = neighbor_mask.contiguous()

        n, c = x.shape
        k = int(neighbor_idx.shape[1])
        x_norm = self.norm(x)
        out = torch.empty_like(x_norm)

        for st in range(0, n, self.chunk_size):
            ed = min(st + self.chunk_size, n)
            idx = neighbor_idx[st:ed]
            mask = neighbor_mask[st:ed]

            xi = x_norm[st:ed]
            xj = x_norm[idx]
            ci = coords[st:ed]
            cj = coords[idx]

            rel = (cj - ci[:, None, :]) / self.pos_scale
            rel = rel.clamp(min=-16.0, max=16.0)
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            pos = self.pos_mlp(torch.cat([rel, torch.log1p(dist)], dim=-1))

            score_h = self.xi_proj(xi)[:, None, :]
            score_h = score_h + self.delta_proj(xj - xi[:, None, :])
            score_h = score_h + self.pos_score_proj(pos)

            if self.support_score_proj is not None:
                support_j = torch.stack(
                    [
                        torch.log1p(acc_w[idx].clamp_min(0.0)),
                        mask.to(dtype=x.dtype),
                    ],
                    dim=-1,
                )
                score_h = score_h + self.support_score_proj(support_j)

            if self.branch_score_proj is not None:
                same_branch = (branch_id[idx].long() == branch_id[st:ed, None].long()).to(dtype=x.dtype)
                score_h = score_h + self.branch_score_proj(same_branch.unsqueeze(-1))

            logits = self.score_head(score_h).squeeze(-1)
            if logits.shape != (ed - st, k):
                raise RuntimeError(f"logits shape mismatch: got {tuple(logits.shape)}, expected {(ed - st, k)}")
            logits = logits.masked_fill(~mask, -1.0e4)
            alpha = torch.softmax(logits, dim=-1)

            value = self.value_proj(xj)
            if self.pos_value_proj is not None:
                value = value + self.pos_value_proj(pos)
            yi = torch.sum(alpha.unsqueeze(-1) * value, dim=1)
            out[st:ed] = self.out_proj(yi)

        return x + self.residual_scale.to(dtype=x.dtype) * out

