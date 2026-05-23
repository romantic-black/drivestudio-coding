from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EventPack:
    event_bg: torch.Tensor
    event_distant: Optional[torch.Tensor] = None
    event_rigid: Optional[torch.Tensor] = None
    support_bg: Optional[torch.Tensor] = None
    support_distant: Optional[torch.Tensor] = None
    support_rigid: Optional[torch.Tensor] = None
    valid_bg: Optional[torch.Tensor] = None
    valid_distant: Optional[torch.Tensor] = None
    valid_rigid: Optional[torch.Tensor] = None
    view_code_bg: Optional[torch.Tensor] = None
    obs_code_bg: Optional[torch.Tensor] = None
    obs_code_distant: Optional[torch.Tensor] = None
    obs_code_rigid: Optional[torch.Tensor] = None
    acc_w_bg: Optional[torch.Tensor] = None
    route: Optional[Any] = None
    branch_slices: Dict[str, slice] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_rigid_S(self) -> Optional[torch.Tensor]:
        return self.event_rigid


def _empty_like_rows(reference: torch.Tensor, cols: int) -> torch.Tensor:
    return reference.new_zeros((0, int(cols)))


class Stage6ParamEncoder(nn.Module):
    """
    Compact trainable embedding for per-GS parameters.

    The raw state tensors are detached by default so the EventEncoder sees a
    stable description of the current local state without creating a direct
    gradient path into LocalGSState that bypasses the updater.
    """

    def __init__(
        self,
        *,
        sh_rest_input_dim: int,
        quat_scales_summary_dim: int = 4,
        sh_rest_summary_dim: int = 8,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.sh_rest_input_dim = int(sh_rest_input_dim)
        self.quat_scales_summary_dim = int(quat_scales_summary_dim)
        self.sh_rest_summary_dim = int(sh_rest_summary_dim)
        self.detach_inputs = bool(detach_inputs)
        self.quat_scales_proj = nn.Linear(7, int(quat_scales_summary_dim))
        self.sh_rest_proj: Optional[nn.Linear]
        if int(sh_rest_input_dim) > 0 and int(sh_rest_summary_dim) > 0:
            self.sh_rest_proj = nn.Linear(int(sh_rest_input_dim), int(sh_rest_summary_dim))
        else:
            self.sh_rest_proj = None
        self.output_dim = 3 + int(quat_scales_summary_dim) + 1 + 3 + int(sh_rest_summary_dim)

    def _select(self, x: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
        y = x[indices] if indices is not None else x
        return y.detach() if self.detach_inputs else y

    def forward(
        self,
        *,
        branch: Any,
        indices: Optional[torch.Tensor] = None,
        aabb_min: Optional[torch.Tensor] = None,
        aabb_max: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        means = self._select(branch.means, indices)
        scales_log = self._select(branch.scales_log, indices)
        quats = self._select(branch.quats, indices)
        opacity_logit = self._select(branch.opacity_logit, indices)
        sh_dc = self._select(branch.sh_dc, indices)
        sh_rest = self._select(branch.sh_rest, indices)

        if dtype is not None:
            means = means.to(dtype=dtype)
            scales_log = scales_log.to(dtype=dtype)
            quats = quats.to(dtype=dtype)
            opacity_logit = opacity_logit.to(dtype=dtype)
            sh_dc = sh_dc.to(dtype=dtype)
            sh_rest = sh_rest.to(dtype=dtype)

        n = int(means.shape[0])
        if n == 0:
            return means.new_zeros((0, int(self.output_dim)))
        if aabb_min is not None and aabb_max is not None:
            lo = aabb_min.to(device=means.device, dtype=means.dtype).reshape(1, 3)
            hi = aabb_max.to(device=means.device, dtype=means.dtype).reshape(1, 3)
            xyz = (means - lo) / (hi - lo).clamp(min=1.0e-6) * 2.0 - 1.0
            xyz = xyz.clamp(-4.0, 4.0)
        else:
            xyz = torch.tanh(means / 10.0)

        quat_norm = F.normalize(quats, dim=-1, eps=1.0e-8)
        quat_scales = self.quat_scales_proj(torch.cat([quat_norm, scales_log], dim=-1))

        sh_rest_flat = sh_rest.reshape(n, -1)
        if int(sh_rest_flat.shape[1]) != int(self.sh_rest_input_dim):
            raise ValueError(
                "Stage6ParamEncoder sh_rest dim mismatch: "
                f"got {int(sh_rest_flat.shape[1])}, expected {int(self.sh_rest_input_dim)}"
            )
        if self.sh_rest_proj is not None:
            sh_rest_summary = self.sh_rest_proj(sh_rest_flat)
        else:
            sh_rest_summary = means.new_zeros((n, int(self.sh_rest_summary_dim)))

        out = torch.cat([xyz, quat_scales, opacity_logit, sh_dc, sh_rest_summary], dim=-1)
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage6ParamEncoder output contains NaN/Inf")
        return out


class Stage6EventEncoder(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        output_dim: int = 96,
        hidden_dim: int = 128,
        num_layers: int = 2,
        obs_code_dim: int = 2,
        view_code_dim: int = 2,
        param_embed_dim: int = 16,
        branch_embed_dim: int = 8,
        allow_missing_view_code: bool = False,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.output_dim = int(output_dim)
        self.obs_code_dim = int(obs_code_dim)
        self.view_code_dim = int(view_code_dim)
        self.param_embed_dim = int(param_embed_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.allow_missing_view_code = bool(allow_missing_view_code)
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        input_dim = (
            int(z_dim)
            + 1
            + int(obs_code_dim)
            + int(view_code_dim)
            + int(param_embed_dim)
            + int(branch_embed_dim)
        )
        layers = []
        last = input_dim
        for _ in range(max(int(num_layers) - 1, 1)):
            layers.extend([nn.Linear(last, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()])
            last = int(hidden_dim)
        layers.append(nn.Linear(last, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def _coerce_feature(
        self,
        x: Optional[torch.Tensor],
        n: int,
        dim: int,
        ref: torch.Tensor,
        *,
        name: str,
        allow_missing: bool = False,
    ) -> torch.Tensor:
        if x is None:
            if allow_missing:
                return ref.new_zeros((int(n), int(dim)))
            raise ValueError(f"{name} is required")
        if x.numel() == 0:
            if int(n) == 0 or allow_missing:
                return ref.new_zeros((int(n), int(dim)))
            raise ValueError(f"{name} is empty for non-empty branch")
        if x.dim() == 1:
            x = x[:, None]
        if int(x.shape[0]) != int(n):
            raise ValueError(f"{name} row mismatch: got {tuple(x.shape)}, expected rows={int(n)}")
        if int(x.shape[1]) != int(dim):
            raise ValueError(f"{name} dim mismatch: got {int(x.shape[1])}, expected {int(dim)}")
        return x

    def encode_branch(
        self,
        *,
        z: torch.Tensor,
        acc_w: Optional[torch.Tensor],
        obs_code: Optional[torch.Tensor],
        view_code: Optional[torch.Tensor],
        param_embed: Optional[torch.Tensor],
        branch_id: int,
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be rank-2 [N,C], got {tuple(z.shape)}")
        n = int(z.shape[0])
        if n == 0:
            return _empty_like_rows(z, self.output_dim)
        z_in = self._coerce_feature(z, n, self.z_dim, z, name="z")
        acc = self._coerce_feature(acc_w, n, 1, z, name="acc_w").clamp_min(0.0)
        obs = self._coerce_feature(obs_code, n, self.obs_code_dim, z, name="obs_code")
        view = self._coerce_feature(
            view_code,
            n,
            self.view_code_dim,
            z,
            name="view_code",
            allow_missing=self.allow_missing_view_code,
        )
        param = self._coerce_feature(param_embed, n, self.param_embed_dim, z, name="param_embed")
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=z.device, dtype=torch.long)
        ).to(dtype=z.dtype)
        x = torch.cat([z_in, torch.log1p(acc), obs, view, param, branch], dim=-1)
        if not torch.isfinite(x).all():
            raise RuntimeError("Stage6EventEncoder input contains NaN/Inf")
        out = self.net(x)
        if not torch.isfinite(out).all():
            raise RuntimeError("Stage6EventEncoder output contains NaN/Inf")
        return out

    def forward(
        self,
        *,
        z_bg: torch.Tensor,
        acc_w_bg: Optional[torch.Tensor],
        obs_code_bg: Optional[torch.Tensor],
        view_code_bg: Optional[torch.Tensor],
        param_embed_bg: Optional[torch.Tensor],
        z_distant: Optional[torch.Tensor] = None,
        acc_w_distant: Optional[torch.Tensor] = None,
        obs_code_distant: Optional[torch.Tensor] = None,
        view_code_distant: Optional[torch.Tensor] = None,
        param_embed_distant: Optional[torch.Tensor] = None,
        z_rigid: Optional[torch.Tensor] = None,
        acc_w_rigid: Optional[torch.Tensor] = None,
        obs_code_rigid: Optional[torch.Tensor] = None,
        view_code_rigid: Optional[torch.Tensor] = None,
        param_embed_rigid: Optional[torch.Tensor] = None,
    ) -> EventPack:
        event_bg = self.encode_branch(
            z=z_bg,
            acc_w=acc_w_bg,
            obs_code=obs_code_bg,
            view_code=view_code_bg,
            param_embed=param_embed_bg,
            branch_id=0,
        )
        event_distant = None
        if z_distant is not None:
            event_distant = self.encode_branch(
                z=z_distant,
                acc_w=acc_w_distant,
                obs_code=obs_code_distant,
                view_code=view_code_distant,
                param_embed=param_embed_distant,
                branch_id=1,
            )
        event_rigid = None
        if z_rigid is not None:
            event_rigid = self.encode_branch(
                z=z_rigid,
                acc_w=acc_w_rigid,
                obs_code=obs_code_rigid,
                view_code=view_code_rigid,
                param_embed=param_embed_rigid,
                branch_id=2,
            )
        aux: Dict[str, Any] = {
            "event_bg_norm": float(event_bg.detach().norm(dim=-1).mean().item()) if event_bg.numel() else 0.0,
        }
        if event_distant is not None:
            aux["event_distant_norm"] = (
                float(event_distant.detach().norm(dim=-1).mean().item()) if event_distant.numel() else 0.0
            )
        if event_rigid is not None:
            aux["event_rigid_norm"] = (
                float(event_rigid.detach().norm(dim=-1).mean().item()) if event_rigid.numel() else 0.0
            )
        return EventPack(
            event_bg=event_bg,
            event_distant=event_distant,
            event_rigid=event_rigid,
            view_code_bg=view_code_bg,
            obs_code_bg=obs_code_bg,
            acc_w_bg=acc_w_bg,
            aux=aux,
        )
