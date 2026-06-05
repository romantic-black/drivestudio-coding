from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.stage6_0 import ContextPack, EventPack, LocalGSState
from models.streetforward.struct_decoders.common import scatter_mean
from models.streetforward.struct_decoders.voxel_layout_utils import build_voxel_layout
from models.streetforward.struct_decoders.xcpe_decoder import _SPCONV_AVAILABLE, _XCPEResidualLayer

from .memory import IForwardMemoryStepContext
from .point_mamba_memory import IForwardPointMemoryPack


class _ConflictPointMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[0]) == 0:
            return x.new_zeros((0, self.output_dim))
        y = self.net(x)
        if not torch.isfinite(y).all():
            raise RuntimeError("IForward local conflict MLP output contains NaN/Inf")
        return y


class IForwardLocalConflictXcpe(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 48,
        point_ctx_dim: int = 16,
        hidden_dim: int = 48,
        output_dim: int = 48,
        num_blocks: int = 1,
        kernel_size: int = 3,
        voxel_size: float = 0.25,
        sparse_backend: str = "spconv",
        obs_code_dim: int = 2,
        branch_embed_dim: int = 4,
        repeat_embed_dim: int = 4,
        residual_scale_init: float = 5.0e-3,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.point_ctx_dim = int(point_ctx_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.obs_code_dim = int(obs_code_dim)
        self.branch_embed_dim = int(branch_embed_dim)
        self.repeat_embed_dim = int(repeat_embed_dim)
        self.voxel_size = float(voxel_size)
        backend = str(sparse_backend).lower()
        if backend not in {"spconv", "fallback_neighbor_mean"}:
            raise ValueError("IForward local_conflict sparse_backend must be 'spconv' or 'fallback_neighbor_mean'.")
        if backend == "spconv" and not _SPCONV_AVAILABLE:
            raise ImportError("IForward local_conflict requires spconv when sparse_backend='spconv'.")
        self.backend = backend
        self.use_spconv = backend == "spconv"
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        self.repeat_embed = nn.Sequential(nn.Linear(3, int(repeat_embed_dim)), nn.GELU())
        raw_dim = int(event_dim) + int(point_ctx_dim) + int(obs_code_dim) + 1 + 1 + int(branch_embed_dim) + int(repeat_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(raw_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                _XCPEResidualLayer(
                    int(hidden_dim),
                    kernel_size=int(kernel_size),
                    use_spconv=self.use_spconv,
                    norm="layernorm",
                    act="gelu",
                    residual_scale_init=float(residual_scale_init),
                    indice_key=f"iforward_v6_local_xcpe_{i}",
                )
                for i in range(int(num_blocks))
            ]
        )
        self.output_proj = nn.Linear(int(hidden_dim), int(output_dim))
        self.point_mlp = _ConflictPointMLP(input_dim=raw_dim, hidden_dim=int(hidden_dim), output_dim=int(output_dim))
        self.fallback_max_points = 20000

    @staticmethod
    def _coerce_feature(
        value: Optional[torch.Tensor],
        *,
        n: int,
        dim: int,
        ref: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        if value is None:
            return ref.new_full((int(n), int(dim)), float(default))
        x = value.to(device=ref.device, dtype=ref.dtype)
        if x.dim() == 1:
            x = x[:, None]
        if int(x.shape[0]) != int(n):
            return ref.new_full((int(n), int(dim)), float(default))
        if int(x.shape[1]) == int(dim):
            return x
        if int(x.shape[1]) > int(dim):
            return x[:, : int(dim)]
        return torch.cat([x, ref.new_full((int(n), int(dim) - int(x.shape[1])), float(default))], dim=-1)

    @staticmethod
    def _select_rows(value: Optional[torch.Tensor], rows: torch.Tensor) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if int(rows.numel()) == 0:
            return value.new_zeros((0,) + tuple(value.shape[1:]))
        return value.index_select(0, rows.to(device=value.device, dtype=torch.long))

    @staticmethod
    def _event_tensor(event: EventPack, name: str) -> Optional[torch.Tensor]:
        value = getattr(event, name, None)
        if value is None:
            return None
        if value.dim() != 2:
            raise ValueError(f"IForward local_conflict expected {name} [N,C], got {tuple(value.shape)}")
        return value

    def _raw(
        self,
        *,
        event_x: torch.Tensor,
        point_ctx: torch.Tensor,
        obs_code: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        branch_id: int,
        step: IForwardMemoryStepContext,
    ) -> torch.Tensor:
        n = int(event_x.shape[0])
        if int(point_ctx.shape[0]) != n:
            raise ValueError("IForward local_conflict event/point_ctx row mismatch")
        obs = self._coerce_feature(obs_code, n=n, dim=self.obs_code_dim, ref=event_x, default=0.0)
        support_x = self._coerce_feature(support, n=n, dim=1, ref=event_x, default=0.0).clamp_min(0.0)
        valid_x = self._coerce_feature(valid, n=n, dim=1, ref=event_x, default=1.0).clamp(0.0, 1.0)
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=event_x.device, dtype=torch.long)
        ).to(dtype=event_x.dtype)
        pos = event_x.new_tensor(
            [float(step.repeat_pos_code), float(step.frame_pos_code), float(step.rollout_pos_code)]
        ).reshape(1, 3).expand(n, -1)
        repeat = self.repeat_embed(pos).to(dtype=event_x.dtype)
        return torch.cat([event_x, point_ctx.to(dtype=event_x.dtype), obs, support_x, valid_x, branch, repeat], dim=-1)

    def _run_xcpe(
        self,
        *,
        raw: torch.Tensor,
        coords: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        batch_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        n = int(raw.shape[0])
        if n == 0:
            return raw.new_zeros((0, self.output_dim)), {"num_points": 0.0, "num_voxels": 0.0, "active_voxel_ratio": 0.0}
        if (not self.use_spconv) and n > self.fallback_max_points:
            raise RuntimeError("IForward fallback_neighbor_mean local XCPE is for tiny tests only; use spconv for training")
        feat = self.input_proj(raw)
        layout = build_voxel_layout(
            coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            voxel_size=float(self.voxel_size),
            batch_offsets=batch_offsets,
            strict_inside=True,
        )
        batch_size = int(batch_offsets.numel() if batch_offsets is not None else 1)
        for layer in self.layers:
            voxel_feat = scatter_mean(feat, layout.inverse, dim_size=int(layout.unique_key.shape[0]))
            voxel_delta = layer(
                voxel_feat=voxel_feat,
                unique_key_bxyz=layout.unique_key,
                indices_bzyx=layout.indices_bzyx,
                spatial_shape_zyx=layout.spatial_shape_zyx,
                batch_size=batch_size,
            )
            feat = feat + layer.residual_scale.to(dtype=feat.dtype) * voxel_delta[layout.inverse]
        out = self.output_proj(feat)
        if not torch.isfinite(out).all():
            raise RuntimeError("IForward local XCPE output contains NaN/Inf")
        return out, {
            "num_points": float(n),
            "num_voxels": float(int(layout.unique_key.shape[0])),
            "active_voxel_ratio": float(int(layout.unique_key.shape[0]) / max(n, 1)),
        }

    @staticmethod
    def _rigid_inside_mask(event: EventPack, n: int, device: torch.device) -> torch.Tensor:
        route = getattr(event, "route", None)
        raw = getattr(route, "inside_mask_S", None) if route is not None else None
        if raw is None:
            return torch.ones((int(n),), device=device, dtype=torch.bool)
        mask = raw.to(device=device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != int(n):
            return torch.ones((int(n),), device=device, dtype=torch.bool)
        return mask

    @staticmethod
    def _rigid_coords(event: EventPack, local_state: LocalGSState, n: int, ref: torch.Tensor) -> torch.Tensor:
        route = getattr(event, "route", None)
        coords = getattr(route, "means_world_S", None) if route is not None else None
        if coords is not None and int(coords.shape[0]) == int(n):
            return coords.to(device=ref.device, dtype=ref.dtype)
        if local_state.rigid is None:
            return ref.new_zeros((int(n), 3))
        rows = getattr(route, "S", None) if route is not None else None
        if rows is None or int(rows.numel()) != int(n):
            rows = torch.arange(int(n), device=ref.device, dtype=torch.long)
        return local_state.rigid.means.index_select(0, rows.to(device=local_state.rigid.means.device, dtype=torch.long)).to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _zero_pack(event: EventPack, output_dim: int) -> ContextPack:
        ctx_bg = event.event_bg.new_zeros((int(event.event_bg.shape[0]), int(output_dim)))
        ctx_distant = None
        if event.event_distant is not None:
            ctx_distant = event.event_distant.new_zeros((int(event.event_distant.shape[0]), int(output_dim)))
        ctx_rigid = None
        if event.event_rigid is not None:
            ctx_rigid = event.event_rigid.new_zeros((int(event.event_rigid.shape[0]), int(output_dim)))
        return ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid, aux={})

    def forward(
        self,
        *,
        event: EventPack,
        point_ctx: IForwardPointMemoryPack,
        local_state: LocalGSState,
        step_context: IForwardMemoryStepContext,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        ablation: str = "full",
    ) -> ContextPack:
        if str(ablation) in {"point_only", "no_memory"}:
            pack = self._zero_pack(event, self.output_dim)
            pack.aux = {"local_xcpe/disabled": 1.0}
            return pack

        t0 = time.perf_counter()
        aux: Dict[str, float] = {"local_xcpe/backend_is_spconv": 1.0 if self.use_spconv else 0.0}
        event_bg = event.event_bg
        ctx_bg_point = point_ctx.ctx_bg
        bg_raw = self._raw(
            event_x=event_bg,
            point_ctx=ctx_bg_point,
            obs_code=getattr(event, "obs_code_bg", None),
            support=getattr(event, "support_bg", None),
            valid=getattr(event, "valid_bg", None),
            branch_id=0,
            step=step_context,
        )

        event_rigid = self._event_tensor(event, "event_rigid")
        ctx_rigid_out: Optional[torch.Tensor] = None
        near_raw = bg_raw
        near_coords = local_state.bg.means.to(device=event_bg.device, dtype=event_bg.dtype)
        near_event_for_ratio = event_bg
        if event_rigid is not None:
            n_rigid = int(event_rigid.shape[0])
            rigid_point = point_ctx.ctx_rigid
            if rigid_point is None:
                rigid_point = event_rigid.new_zeros((n_rigid, self.point_ctx_dim))
            rigid_raw_all = self._raw(
                event_x=event_rigid,
                point_ctx=rigid_point,
                obs_code=getattr(event, "obs_code_rigid", None),
                support=getattr(event, "support_rigid", None),
                valid=getattr(event, "valid_rigid", None),
                branch_id=2,
                step=step_context,
            )
            rigid_coords_all = self._rigid_coords(event, local_state, n_rigid, event_rigid)
            inside = self._rigid_inside_mask(event, n_rigid, event_rigid.device)
            ctx_rigid_out = event_rigid.new_zeros((n_rigid, self.output_dim))
            if str(ablation) != "disable_rigid_xcpe":
                inside_rows = torch.nonzero(inside, as_tuple=False).squeeze(1)
                if int(inside_rows.numel()) > 0:
                    near_raw = torch.cat([near_raw, rigid_raw_all.index_select(0, inside_rows)], dim=0)
                    near_coords = torch.cat([near_coords, rigid_coords_all.index_select(0, inside_rows)], dim=0)
                    near_event_for_ratio = torch.cat([near_event_for_ratio, event_rigid.index_select(0, inside_rows)], dim=0)

        near_out, xcpe_aux = self._run_xcpe(
            raw=near_raw,
            coords=near_coords,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        n_bg = int(event_bg.shape[0])
        ctx_bg = near_out[:n_bg]
        if event_rigid is not None and ctx_rigid_out is not None and str(ablation) != "disable_rigid_xcpe":
            inside = self._rigid_inside_mask(event, int(event_rigid.shape[0]), event_rigid.device)
            inside_rows = torch.nonzero(inside, as_tuple=False).squeeze(1)
            if int(inside_rows.numel()) > 0:
                ctx_rigid_out[inside_rows] = near_out[n_bg:]

        ctx_distant = None
        event_distant = self._event_tensor(event, "event_distant")
        if event_distant is not None:
            n = int(event_distant.shape[0])
            distant_point = point_ctx.ctx_distant
            if distant_point is None:
                distant_point = event_distant.new_zeros((n, self.point_ctx_dim))
            far_raw = self._raw(
                event_x=event_distant,
                point_ctx=distant_point,
                obs_code=getattr(event, "obs_code_distant", None),
                support=getattr(event, "support_distant", None),
                valid=getattr(event, "valid_distant", None),
                branch_id=1,
                step=step_context,
            )
            ctx_distant = self.point_mlp(far_raw)

        output_norm = near_out.detach().norm(dim=-1).mean() if near_out.numel() else near_out.new_tensor(0.0)
        event_norm = near_event_for_ratio.detach().norm(dim=-1).mean() if near_event_for_ratio.numel() else near_out.new_tensor(0.0)
        aux.update(
            {
                "local_xcpe/near_input_norm": float(near_raw.detach().norm(dim=-1).mean().item()) if near_raw.numel() else 0.0,
                "local_xcpe/near_output_norm": float(output_norm.item()),
                "local_xcpe/near_output_event_ratio": float((output_norm / event_norm.clamp_min(1.0e-6)).item()),
                "local_xcpe/num_points": float(xcpe_aux.get("num_points", 0.0)),
                "local_xcpe/num_voxels": float(xcpe_aux.get("num_voxels", 0.0)),
                "local_xcpe/active_voxel_ratio": float(xcpe_aux.get("active_voxel_ratio", 0.0)),
                "local_xcpe/ms": float((time.perf_counter() - t0) * 1000.0),
            }
        )
        return ContextPack(ctx_bg=ctx_bg, ctx_distant=ctx_distant, ctx_rigid=ctx_rigid_out, aux=aux)
