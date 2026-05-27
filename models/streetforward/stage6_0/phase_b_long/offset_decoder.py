from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .streaming_vsm import DISTANT_MODE_APPEARANCE_SCALE, DISTANT_MODE_FROZEN, _check_distant_mode
from .types import BgOffsetDelta, DistantOffsetDelta, LongOffsetDelta, LongVSMReadPack, RigidOffsetDelta


def _cfg_float(cfg: Optional[Dict[str, Any]], *path: str, default: float) -> float:
    node: Any = cfg or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float(default)
        node = node[key]
    return float(node)


class _OffsetHead(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VSMOffsetDecoder(nn.Module):
    def __init__(
        self,
        *,
        bg_mem_dim: int = 64,
        rigid_mem_dim: int = 64,
        distant_mem_dim: int = 32,
        distant_sh_rest_bases: int = 0,
        distant_sh_rest_update_bases: int = 0,
        hidden_dim: int = 128,
        clamps: Optional[Dict[str, Any]] = None,
        distant_mode: str = "frozen_render_only",
    ) -> None:
        super().__init__()
        self.bg_mem_dim = int(bg_mem_dim)
        self.rigid_mem_dim = int(rigid_mem_dim)
        self.distant_mem_dim = int(distant_mem_dim)
        self.distant_sh_rest_bases = max(int(distant_sh_rest_bases), 0)
        self.distant_sh_rest_update_bases = min(
            max(int(distant_sh_rest_update_bases), 0),
            int(self.distant_sh_rest_bases),
        )
        self.distant_mode = _check_distant_mode(str(distant_mode))
        self.clamps = dict(clamps or {})
        self.bg_head = _OffsetHead(in_dim=int(bg_mem_dim) + 1, hidden_dim=int(hidden_dim), out_dim=10)
        self.rigid_head = _OffsetHead(in_dim=int(rigid_mem_dim) + 1, hidden_dim=int(hidden_dim), out_dim=10)
        distant_out_dim = 3 + 1 + 3 + int(self.distant_sh_rest_update_bases) * 3
        self.distant_head = _OffsetHead(
            in_dim=int(distant_mem_dim) + 1,
            hidden_dim=int(hidden_dim),
            out_dim=int(distant_out_dim),
        )

    def _clamp(self, raw: torch.Tensor, value: float) -> torch.Tensor:
        limit = float(value)
        return (torch.tanh(raw.float()) * limit).clamp(min=-limit, max=limit)

    def forward_bg(
        self,
        read_bg: torch.Tensor,
        seen_bg: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> BgOffsetDelta:
        if read_bg.dim() != 2 or int(read_bg.shape[1]) != int(self.bg_mem_dim):
            raise ValueError(f"read_bg must be [N,{self.bg_mem_dim}], got {tuple(read_bg.shape)}")
        if seen_bg.dim() == 1:
            seen_bg = seen_bg[:, None]
        if int(seen_bg.shape[0]) != int(read_bg.shape[0]):
            raise ValueError("seen_bg rows must match read_bg.")
        seen_feat = torch.log1p(seen_bg.to(device=read_bg.device, dtype=torch.float32).clamp_min(0.0)).to(
            dtype=read_bg.dtype
        )
        raw = self.bg_head(torch.cat([read_bg, seen_feat], dim=-1))
        means = self._clamp(raw[:, 0:3], _cfg_float(self.clamps, "bg", "means_step_m", default=0.02))
        scales = self._clamp(raw[:, 3:6], _cfg_float(self.clamps, "bg", "scales_log_step", default=0.015))
        opacity = self._clamp(raw[:, 6:7], _cfg_float(self.clamps, "bg", "opacity_logit_step", default=0.04))
        sh_dc = self._clamp(raw[:, 7:10], _cfg_float(self.clamps, "bg", "sh_dc_step", default=0.03))
        mask = (seen_bg.to(device=read_bg.device) > 0).to(dtype=read_bg.dtype)
        return BgOffsetDelta(
            means=means * mask,
            scales_log=scales * mask,
            opacity_logit=opacity * mask,
            sh_dc=sh_dc * mask,
            mask=mask,
            indices=None if indices is None else indices.to(device=read_bg.device, dtype=torch.long).reshape(-1),
        )

    def forward_rigid(
        self,
        read_rigid: Optional[torch.Tensor],
        rigid_indices: Optional[torch.Tensor],
        seen_rigid: Optional[torch.Tensor],
        stable_mask: Optional[torch.Tensor],
    ) -> Optional[RigidOffsetDelta]:
        if read_rigid is None or rigid_indices is None or stable_mask is None:
            return None
        if read_rigid.dim() != 2 or int(read_rigid.shape[1]) != int(self.rigid_mem_dim):
            raise ValueError(f"read_rigid must be [N,{self.rigid_mem_dim}], got {tuple(read_rigid.shape)}")
        n = int(read_rigid.shape[0])
        if int(rigid_indices.numel()) != n or int(stable_mask.numel()) != n:
            raise ValueError("rigid read/index/stable rows must match.")
        if seen_rigid is None:
            seen_rigid = read_rigid.new_ones((n, 1))
        if seen_rigid.dim() == 1:
            seen_rigid = seen_rigid[:, None]
        if int(seen_rigid.shape[0]) != n:
            raise ValueError("seen_rigid rows must match read_rigid.")
        seen_feat = torch.log1p(seen_rigid.to(device=read_rigid.device, dtype=torch.float32).clamp_min(0.0)).to(
            dtype=read_rigid.dtype
        )
        raw = self.rigid_head(torch.cat([read_rigid, seen_feat], dim=-1))
        means = self._clamp(raw[:, 0:3], _cfg_float(self.clamps, "rigid", "means_local_step_m", default=0.03))
        scales = self._clamp(raw[:, 3:6], _cfg_float(self.clamps, "rigid", "scales_log_step", default=0.015))
        opacity = self._clamp(raw[:, 6:7], _cfg_float(self.clamps, "rigid", "opacity_logit_step", default=0.04))
        sh_dc = self._clamp(raw[:, 7:10], _cfg_float(self.clamps, "rigid", "sh_dc_step", default=0.03))
        mask = (seen_rigid.to(device=read_rigid.device) > 0).to(dtype=read_rigid.dtype)
        return RigidOffsetDelta(
            indices=rigid_indices.to(device=read_rigid.device, dtype=torch.long).reshape(-1),
            stable_mask=stable_mask.to(device=read_rigid.device, dtype=torch.bool).reshape(-1),
            means_local=means * mask,
            scales_log=scales * mask,
            opacity_logit=opacity * mask,
            sh_dc=sh_dc * mask,
        )

    def forward_distant(
        self,
        read_distant: Optional[torch.Tensor],
        distant_indices: Optional[torch.Tensor],
        seen_distant: Optional[torch.Tensor],
        *,
        mode: str,
    ) -> Optional[DistantOffsetDelta]:
        if mode == DISTANT_MODE_FROZEN:
            if read_distant is not None:
                raise ValueError("6_0_phase_b frozen distant mode forbids distant offset decoding.")
            return None
        if mode != DISTANT_MODE_APPEARANCE_SCALE:
            raise ValueError(f"unsupported distant mode {mode!r}")
        if read_distant is None:
            return None
        if distant_indices is None:
            raise ValueError("distant readout requires distant_indices.")
        if read_distant.dim() != 2 or int(read_distant.shape[1]) != int(self.distant_mem_dim):
            raise ValueError(f"read_distant must be [N,{self.distant_mem_dim}], got {tuple(read_distant.shape)}")
        n = int(read_distant.shape[0])
        if int(distant_indices.numel()) != n:
            raise ValueError("distant read/index rows must match.")
        if seen_distant is None:
            seen_distant = read_distant.new_ones((n, 1))
        if seen_distant.dim() == 1:
            seen_distant = seen_distant[:, None]
        if int(seen_distant.shape[0]) != n:
            raise ValueError("seen_distant rows must match read_distant.")
        seen_feat = torch.log1p(seen_distant.to(device=read_distant.device, dtype=torch.float32).clamp_min(0.0)).to(
            dtype=read_distant.dtype
        )
        raw = self.distant_head(torch.cat([read_distant, seen_feat], dim=-1))
        scales = self._clamp(raw[:, 0:3], _cfg_float(self.clamps, "distant", "scales_log_step", default=0.01))
        opacity = self._clamp(raw[:, 3:4], _cfg_float(self.clamps, "distant", "opacity_logit_step", default=0.02))
        sh_dc = self._clamp(raw[:, 4:7], _cfg_float(self.clamps, "distant", "sh_dc_step", default=0.02))
        rest_dim = int(self.distant_sh_rest_update_bases) * 3
        if rest_dim > 0:
            sh_rest = self._clamp(
                raw[:, 7 : 7 + rest_dim],
                _cfg_float(self.clamps, "distant", "sh_rest_step", default=0.01),
            ).view(n, int(self.distant_sh_rest_update_bases), 3)
        else:
            sh_rest = read_distant.new_zeros((n, 0, 3))
        mask = (seen_distant.to(device=read_distant.device) > 0).to(dtype=read_distant.dtype)
        return DistantOffsetDelta(
            indices=distant_indices.to(device=read_distant.device, dtype=torch.long).reshape(-1),
            scales_log=scales * mask,
            opacity_logit=opacity * mask,
            sh_dc=sh_dc * mask,
            sh_rest=sh_rest * mask[:, None, :],
            mask=mask,
        )

    def forward(self, *, read: LongVSMReadPack, distant_mode: Optional[str] = None) -> LongOffsetDelta:
        mode = _check_distant_mode(str(distant_mode or self.distant_mode))
        bg = self.forward_bg(read.bg, read.seen_bg, indices=read.bg_indices)
        rigid = self.forward_rigid(
            read.rigid,
            read.rigid_indices,
            read.rigid_seen,
            read.rigid_stable_mask,
        )
        distant = self.forward_distant(
            read.distant,
            read.distant_indices,
            read.distant_seen,
            mode=mode,
        )
        return LongOffsetDelta(bg=bg, rigid=rigid, distant=distant)
