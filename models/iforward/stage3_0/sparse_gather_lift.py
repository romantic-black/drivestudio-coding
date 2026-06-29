from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .cuda_sparse_gather import can_use_cuda_sparse_gather, sparse_gather_2d
from .sparse_grid_sample import prepare_value_nchw


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def _zero_module(module: nn.Module) -> None:
    for param in module.parameters():
        nn.init.zeros_(param)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    seq = []
    last = int(in_dim)
    for _ in range(max(int(layers) - 1, 0)):
        seq.extend([nn.Linear(last, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()])
        last = int(hidden_dim)
    seq.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*seq)


@dataclass(frozen=True)
class GatherConfig:
    num_taps: int = 5
    offset_scale: float = 0.5
    max_offset_px: float = 8.0
    query_dim: int = 96
    chunk_size: int = 32768
    fixed_center_chunk_size: int = 65536
    center_tap_bias: float = 2.0
    use_geometry_pe: bool = True
    fixed_center_steps: int = 1000
    train_weights_steps: int = 3000
    offset_warmup_steps: int = 5000
    offset_scale_start: float = 0.1
    fixed_center_fast_path: bool = True
    fixed_center_use_geometry_pe: bool = False
    backend: str = "auto"

    @classmethod
    def from_config(cls, cfg: Any, *, defaults: Optional["GatherConfig"] = None) -> "GatherConfig":
        base = defaults or cls()
        return cls(
            num_taps=int(_cfg_get(cfg, "num_taps", base.num_taps)),
            offset_scale=float(_cfg_get(cfg, "offset_scale", base.offset_scale)),
            max_offset_px=float(_cfg_get(cfg, "max_offset_px", base.max_offset_px)),
            query_dim=int(_cfg_get(cfg, "query_dim", base.query_dim)),
            chunk_size=int(_cfg_get(cfg, "chunk_size", base.chunk_size)),
            fixed_center_chunk_size=int(_cfg_get(cfg, "fixed_center_chunk_size", base.fixed_center_chunk_size)),
            center_tap_bias=float(_cfg_get(cfg, "center_tap_bias", base.center_tap_bias)),
            use_geometry_pe=bool(_cfg_get(cfg, "use_geometry_pe", base.use_geometry_pe)),
            fixed_center_steps=int(_cfg_get(cfg, "fixed_center_steps", base.fixed_center_steps)),
            train_weights_steps=int(_cfg_get(cfg, "train_weights_steps", base.train_weights_steps)),
            offset_warmup_steps=int(_cfg_get(cfg, "offset_warmup_steps", base.offset_warmup_steps)),
            offset_scale_start=float(_cfg_get(cfg, "offset_scale_start", base.offset_scale_start)),
            fixed_center_fast_path=bool(_cfg_get(cfg, "fixed_center_fast_path", base.fixed_center_fast_path)),
            fixed_center_use_geometry_pe=bool(
                _cfg_get(cfg, "fixed_center_use_geometry_pe", base.fixed_center_use_geometry_pe)
            ),
            backend=str(_cfg_get(cfg, "backend", base.backend)).lower(),
        )


@dataclass
class Stage3ChildDetailPack:
    child_detail_bg: torch.Tensor
    child_detail_distant: Optional[torch.Tensor]
    child_detail_rigid: Optional[torch.Tensor]
    child_detail_valid_bg: torch.Tensor
    child_detail_valid_distant: Optional[torch.Tensor]
    child_detail_valid_rigid: Optional[torch.Tensor]
    child_detail_support_bg: torch.Tensor
    child_detail_support_distant: Optional[torch.Tensor]
    child_detail_support_rigid: Optional[torch.Tensor]
    aux: Dict[str, float]


class ParentQueryBuilder(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        hidden_dim: int = 96,
        branch_embed_dim: int = 4,
        extra_input_dim: int = 0,
    ) -> None:
        super().__init__()
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        self.extra_input_dim = int(extra_input_dim)
        in_dim = 3 + 3 + 4 + 1 + 1 + 4 + int(branch_embed_dim) + int(extra_input_dim)
        self.net = _mlp(in_dim, int(hidden_dim), int(query_dim), layers=2)

    def forward(
        self,
        *,
        params: Dict[str, torch.Tensor],
        support_total: torch.Tensor,
        branch_id: int,
        optimizer_prior: Optional[torch.Tensor] = None,
        obs2d_lift: Optional[torch.Tensor] = None,
        dino_lift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        means = params["means"]
        n = int(means.shape[0])
        if n == 0:
            return means.new_zeros((0, int(self.net[-1].out_features)))  # type: ignore[index]
        quat = params["quats"].to(device=means.device, dtype=means.dtype)
        quat = torch.nn.functional.normalize(quat, dim=-1, eps=1.0e-8)
        support = torch.log1p(support_total.to(device=means.device, dtype=means.dtype).reshape(-1, 1).clamp_min(0.0))
        if optimizer_prior is None:
            prior = means.new_zeros((n, 4))
        else:
            prior = optimizer_prior.to(device=means.device, dtype=means.dtype).reshape(n, -1)
            if int(prior.shape[1]) < 4:
                prior = torch.cat([prior, means.new_zeros((n, 4 - int(prior.shape[1])))], dim=-1)
            prior = prior[:, :4]
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=means.device, dtype=torch.long)
        ).to(dtype=means.dtype)
        parts = [
            torch.tanh(means / 10.0),
            params["scales_log"].to(device=means.device, dtype=means.dtype).clamp(-12.0, 8.0),
            quat,
            params["opacity_logit"].to(device=means.device, dtype=means.dtype).reshape(n, 1).clamp(-20.0, 20.0),
            support,
            prior,
            branch,
        ]
        extra_parts = []
        for item in (obs2d_lift, dino_lift):
            if item is None:
                continue
            item_t = item.to(device=means.device, dtype=means.dtype)
            if item_t.dim() != 2 or int(item_t.shape[0]) != n:
                raise ValueError(f"parent query extra feature must be [N,C], got {tuple(item_t.shape)}")
            extra_parts.append(item_t)
        extra_dim = int(sum(int(x.shape[-1]) for x in extra_parts))
        if extra_dim != int(self.extra_input_dim):
            raise ValueError(
                f"parent query extra dim mismatch: got {extra_dim}, expected {int(self.extra_input_dim)}"
            )
        parts.extend(extra_parts)
        x = torch.cat(parts, dim=-1)
        return self.net(x)


class ChildQueryBuilder(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        parent_event_dim: int,
        hidden_dim: int = 128,
        branch_embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.parent_event_dim = int(parent_event_dim)
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        in_dim = 3 + 3 + 1 + 1 + int(parent_event_dim) + int(branch_embed_dim)
        self.net = _mlp(in_dim, int(hidden_dim), int(query_dim), layers=2)

    def forward(
        self,
        *,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        parent_id: torch.Tensor,
        parent_event: torch.Tensor,
        support_total: torch.Tensor,
        branch_id: int,
    ) -> torch.Tensor:
        means = child_params["means"]
        n = int(means.shape[0])
        if n == 0:
            return means.new_zeros((0, int(self.net[-1].out_features)))  # type: ignore[index]
        pid = parent_id.to(device=means.device, dtype=torch.long).reshape(-1)
        if int(pid.numel()) != n:
            raise ValueError(f"child query parent_id row mismatch: {int(pid.numel())} vs {n}")
        parent_means = parent_params["means"].to(device=means.device, dtype=means.dtype).index_select(0, pid)
        parent_scales_log = parent_params["scales_log"].to(device=means.device, dtype=means.dtype).index_select(0, pid)
        parent_scales = parent_scales_log.exp().clamp_min(1.0e-3)
        rel_xyz = ((means - parent_means) / parent_scales).clamp(-8.0, 8.0)
        rel_scale = (
            child_params["scales_log"].to(device=means.device, dtype=means.dtype) - parent_scales_log
        ).clamp(-8.0, 8.0)
        parent_opacity = parent_params["opacity_logit"].to(device=means.device, dtype=means.dtype).index_select(0, pid)
        rel_opacity = (
            child_params["opacity_logit"].to(device=means.device, dtype=means.dtype).reshape(n, 1) - parent_opacity.reshape(n, 1)
        ).clamp(-20.0, 20.0)
        support = torch.log1p(support_total.to(device=means.device, dtype=means.dtype).reshape(n, 1).clamp_min(0.0))
        pe = parent_event.to(device=means.device, dtype=means.dtype).index_select(0, pid)
        if int(pe.shape[-1]) != int(self.parent_event_dim):
            raise ValueError(f"parent event dim mismatch: got {int(pe.shape[-1])}, expected {self.parent_event_dim}")
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=means.device, dtype=torch.long)
        ).to(dtype=means.dtype)
        return self.net(torch.cat([rel_xyz, rel_scale, rel_opacity, support, pe, branch], dim=-1))


class ParentContextFusion(nn.Module):
    def __init__(self, *, context_dim: int, dino_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.dino_dim = int(dino_dim)
        in_dim = int(context_dim) + int(dino_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(context_dim)),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, context: torch.Tensor, dino_lift: torch.Tensor) -> torch.Tensor:
        if context.dim() != 2 or dino_lift.dim() != 2:
            raise ValueError("parent context fusion expects [N,C] tensors")
        if int(context.shape[0]) != int(dino_lift.shape[0]):
            raise ValueError("parent context fusion row mismatch")
        if int(context.shape[-1]) != int(self.context_dim) or int(dino_lift.shape[-1]) != int(self.dino_dim):
            raise ValueError(
                "parent context fusion channel mismatch: "
                f"context={tuple(context.shape)}, dino={tuple(dino_lift.shape)}"
            )
        x = torch.cat([context, dino_lift.to(device=context.device, dtype=context.dtype)], dim=-1)
        return context + self.net(x)


class SparseGatherLift(nn.Module):
    def __init__(
        self,
        *,
        value_dim: int,
        config: GatherConfig,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.config = config
        self.num_taps = int(config.num_taps)
        self.center_tap = int(self.num_taps // 2)
        geom_dim = 7
        head_in = int(config.query_dim) + geom_dim
        self.head = _mlp(head_in, int(hidden_dim), int(hidden_dim), layers=2)
        self.view_logit_head = nn.Linear(int(hidden_dim), 1)
        self.tap_logit_head = nn.Linear(int(hidden_dim), int(self.num_taps))
        self.offset_head = nn.Linear(int(hidden_dim), int(self.num_taps) * 2)
        self.gate_head = nn.Linear(int(config.query_dim), int(value_dim))
        self.geometry_pe = nn.Linear(geom_dim, int(value_dim)) if bool(config.use_geometry_pe) else None
        _zero_module(self.view_logit_head)
        _zero_module(self.tap_logit_head)
        _zero_module(self.offset_head)
        _zero_module(self.gate_head)
        with torch.no_grad():
            self.tap_logit_head.bias[self.center_tap] = float(config.center_tap_bias)

    @classmethod
    def from_config(cls, cfg: Any, *, value_dim: int, default_query_dim: int) -> "SparseGatherLift":
        defaults = GatherConfig(query_dim=int(default_query_dim), offset_scale=0.5 if int(value_dim) != 8 else 0.75)
        return cls(value_dim=int(value_dim), config=GatherConfig.from_config(cfg, defaults=defaults))

    def _offset_scale_factor(self, global_step: Optional[int]) -> Tuple[bool, float]:
        step = 10**12 if global_step is None else int(global_step)
        cfg = self.config
        if step < int(cfg.fixed_center_steps):
            return True, 0.0
        if step < int(cfg.train_weights_steps):
            return False, 0.0
        if step >= int(cfg.offset_warmup_steps):
            return False, 1.0
        span = max(int(cfg.offset_warmup_steps) - int(cfg.train_weights_steps), 1)
        progress = float(step - int(cfg.train_weights_steps)) / float(span)
        return False, float(cfg.offset_scale_start) + (1.0 - float(cfg.offset_scale_start)) * max(0.0, min(progress, 1.0))

    def use_fixed_center_fast_path(self, global_step: Optional[int]) -> bool:
        fixed_center, _scale = self._offset_scale_factor(global_step)
        return bool(self.config.fixed_center_fast_path) and bool(fixed_center)

    def effective_chunk_size(self, global_step: Optional[int], *, rows: Optional[int] = None) -> int:
        if self.use_fixed_center_fast_path(global_step):
            fixed_chunk = int(self.config.fixed_center_chunk_size)
            if fixed_chunk <= 0:
                return max(int(rows or 0), 1)
            return max(fixed_chunk, 1)
        return max(int(self.config.chunk_size), 1)

    @staticmethod
    def _entropy(weights: torch.Tensor, dim: int) -> torch.Tensor:
        w = weights.clamp_min(1.0e-8)
        return -(w * w.log()).sum(dim=dim)

    @staticmethod
    def _source_inbound_mask(uv_px: torch.Tensor, *, image_height: int, image_width: int) -> torch.Tensor:
        x = (uv_px[..., 0] + 0.5) * (2.0 / float(max(int(image_width), 1))) - 1.0
        y = (uv_px[..., 1] + 0.5) * (2.0 / float(max(int(image_height), 1))) - 1.0
        return (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)

    def _should_prepare_value(self, value_map: torch.Tensor) -> bool:
        backend = str(self.config.backend).lower()
        if backend == "pytorch":
            return True
        if backend == "cuda":
            return False
        return not can_use_cuda_sparse_gather(value_map, backend="auto")

    def _sample_weighted_sum(
        self,
        *,
        value_map: torch.Tensor,
        sample_uv: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        image_height: int,
        image_width: int,
        prepared_value_nchw: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return sparse_gather_2d(
            value_map,
            sample_uv,
            weights,
            valid,
            image_height=int(image_height),
            image_width=int(image_width),
            backend=str(self.config.backend).lower(),  # type: ignore[arg-type]
            prepared_value_nchw=prepared_value_nchw,
            chunk_size=int(self.config.chunk_size),
        )

    def forward(
        self,
        *,
        value_map: torch.Tensor,
        anchor_uv: torch.Tensor,
        support: torch.Tensor,
        valid: torch.Tensor,
        depth: torch.Tensor,
        radius: torch.Tensor,
        image_height: int,
        image_width: int,
        query: Optional[torch.Tensor] = None,
        prepared_value_nchw: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        emit_heavy_aux: bool = True,
        prefix: str = "stage3/gather",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        if query is not None and query.dim() != 2:
            raise ValueError(f"query must be [R,Q], got {tuple(query.shape)}")
        r = int(anchor_uv.shape[0])
        v = int(anchor_uv.shape[1]) if anchor_uv.dim() >= 2 else 0
        if anchor_uv.shape[:2] != support.shape[:2] or anchor_uv.shape[:2] != valid.shape[:2]:
            raise ValueError("anchor/support/valid shape mismatch")
        if query is not None and int(query.shape[0]) != r:
            raise ValueError(f"query/anchor row mismatch: {int(query.shape[0])} vs {r}")
        fast_center = self.use_fixed_center_fast_path(global_step)
        if not bool(fast_center) and query is None:
            raise ValueError("query is required outside fixed-center fast path")
        prepared = prepared_value_nchw
        if prepared is None and self._should_prepare_value(value_map):
            prepared = prepare_value_nchw(value_map)
        row_chunk = self.effective_chunk_size(global_step, rows=r)
        if r > row_chunk:
            outs = []
            confs = []
            aux_weighted: Dict[str, float] = {}
            reg_weighted: Dict[str, torch.Tensor] = {}
            total_rows = 0
            for start in range(0, r, row_chunk):
                end = min(start + row_chunk, r)
                rows = int(end - start)
                out_c, conf_c, aux_c, reg_c = self.forward(
                    value_map=value_map,
                    anchor_uv=anchor_uv[start:end],
                    support=support[start:end],
                    valid=valid[start:end],
                    depth=depth[start:end],
                    radius=radius[start:end],
                    query=None if query is None else query[start:end],
                    prepared_value_nchw=prepared,
                    image_height=int(image_height),
                    image_width=int(image_width),
                    global_step=global_step,
                    emit_heavy_aux=emit_heavy_aux,
                    prefix=prefix,
                )
                outs.append(out_c)
                confs.append(conf_c)
                total_rows += rows
                for key, value in aux_c.items():
                    aux_weighted[str(key)] = aux_weighted.get(str(key), 0.0) + float(value) * float(rows)
                for key, value in reg_c.items():
                    if not torch.is_tensor(value):
                        continue
                    reg_weighted[str(key)] = value * float(rows) if key not in reg_weighted else reg_weighted[str(key)] + value * float(rows)
            denom = float(max(int(total_rows), 1))
            out = torch.cat(outs, dim=0) if outs else value_map.new_zeros((0, int(self.value_dim)))
            conf = torch.cat(confs, dim=0) if confs else value_map.new_zeros((0,))
            aux = {key: float(value) / denom for key, value in aux_weighted.items()}
            reg = {key: value / denom for key, value in reg_weighted.items()}
            return out, conf, aux, reg
        if r == 0:
            out = value_map.new_zeros((0, int(self.value_dim)))
            aux = {
                f"iforward/{prefix}_inbound_ratio": 0.0,
                f"iforward/{prefix}_offset_norm_mean": 0.0,
                f"iforward/{prefix}_offset_norm_p95": 0.0,
                f"iforward/{prefix}_view_entropy": 0.0,
                f"iforward/{prefix}_tap_entropy": 0.0,
                f"iforward/{prefix}_output_rms": 0.0,
                f"iforward/{prefix}_valid_ratio": 0.0,
                f"iforward/{prefix}_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
            }
            reg = {"offset_l2": out.sum() * 0.0, "out_of_bounds": out.sum() * 0.0}
            return out, value_map.new_zeros((0,)), aux, reg

        device = value_map.device
        dtype = value_map.dtype
        uv = anchor_uv.to(device=device, dtype=dtype)
        supp = support.to(device=device, dtype=dtype).clamp_min(0.0)
        val = valid.to(device=device, dtype=torch.bool)
        dep = depth.to(device=device, dtype=dtype)
        rad = radius.to(device=device, dtype=dtype).clamp_min(1.0)

        if bool(fast_center):
            sample_uv = uv[:, :, None, :]
            cheap_inbound = self._source_inbound_mask(
                sample_uv,
                image_height=int(image_height),
                image_width=int(image_width),
            )
            inbound_center = cheap_inbound[:, :, 0]
            valid_view = val & inbound_center
            weight_raw = torch.where(valid_view, supp, torch.zeros_like(supp))
            denom = weight_raw.sum(dim=1, keepdim=True)
            row_valid = denom.squeeze(-1) > 0.0
            view_weights = weight_raw / denom.clamp_min(1.0e-8)
            out, inbound, backend_used = self._sample_weighted_sum(
                value_map=value_map,
                sample_uv=sample_uv,
                weights=view_weights.unsqueeze(-1),
                valid=val[:, :, None],
                image_height=int(image_height),
                image_width=int(image_width),
                prepared_value_nchw=prepared,
            )
            inbound_center = inbound[:, :, 0]
            if self.geometry_pe is not None and bool(self.config.fixed_center_use_geometry_pe):
                uv_norm = torch.stack(
                    [
                        (uv[..., 0] + 0.5) / float(max(int(image_width), 1)) * 2.0 - 1.0,
                        (uv[..., 1] + 0.5) / float(max(int(image_height), 1)) * 2.0 - 1.0,
                    ],
                    dim=-1,
                )
                view_pos = torch.linspace(0.0, 1.0, max(v, 1), device=device, dtype=dtype).reshape(1, v, 1).expand(r, v, 1)
                geom = torch.cat(
                    [
                        uv_norm,
                        torch.log1p(dep.clamp_min(0.0)).unsqueeze(-1),
                        torch.log1p(rad).unsqueeze(-1),
                        torch.log1p(supp).unsqueeze(-1),
                        val.to(dtype).unsqueeze(-1),
                        view_pos,
                    ],
                    dim=-1,
                )
                out = out + (self.geometry_pe(geom) * view_weights.unsqueeze(-1)).sum(dim=1)
            out = torch.where(row_valid.unsqueeze(-1), out, torch.zeros_like(out))
            confidence = (view_weights * supp).sum(dim=1) * row_valid.to(dtype)
            aux = {
                f"iforward/{prefix}_offset_norm_mean": 0.0,
                f"iforward/{prefix}_offset_norm_p95": 0.0,
                f"iforward/{prefix}_tap_entropy": 0.0,
                f"iforward/{prefix}_fixed_fast_path_enabled": 1.0,
                f"iforward/{prefix}_cuda_backend_enabled": 1.0 if backend_used == "cuda" else 0.0,
                f"iforward/{prefix}_pytorch_fallback_enabled": 1.0 if backend_used == "pytorch" else 0.0,
                f"iforward/{prefix}_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
            }
            if bool(emit_heavy_aux):
                aux.update(
                    {
                        f"iforward/{prefix}_inbound_ratio": float((inbound_center & val).detach().float().mean().item()),
                        f"iforward/{prefix}_view_entropy": float(
                            self._entropy(view_weights, dim=1).detach().float().mean().item()
                        )
                        if int(view_weights.numel())
                        else 0.0,
                        f"iforward/{prefix}_output_rms": float(out.detach().float().square().mean().sqrt().item())
                        if out.numel()
                        else 0.0,
                        f"iforward/{prefix}_valid_ratio": float(row_valid.detach().float().mean().item())
                        if row_valid.numel()
                        else 0.0,
                    }
                )
            zero = out.sum() * 0.0
            reg = {"offset_l2": zero, "out_of_bounds": zero}
            return out, confidence, aux, reg

        q = query.to(device=device, dtype=dtype)  # type: ignore[union-attr]

        uv_norm = torch.stack(
            [
                (uv[..., 0] + 0.5) / float(max(int(image_width), 1)) * 2.0 - 1.0,
                (uv[..., 1] + 0.5) / float(max(int(image_height), 1)) * 2.0 - 1.0,
            ],
            dim=-1,
        )
        view_pos = torch.linspace(0.0, 1.0, max(v, 1), device=device, dtype=dtype).reshape(1, v, 1).expand(r, v, 1)
        geom = torch.cat(
            [
                uv_norm,
                torch.log1p(dep.clamp_min(0.0)).unsqueeze(-1),
                torch.log1p(rad).unsqueeze(-1),
                torch.log1p(supp).unsqueeze(-1),
                val.to(dtype).unsqueeze(-1),
                view_pos,
            ],
            dim=-1,
        )
        x = torch.cat([q[:, None, :].expand(r, v, int(q.shape[-1])), geom], dim=-1)
        h = self.head(x.reshape(r * v, -1)).reshape(r, v, -1)
        view_logits = self.view_logit_head(h).squeeze(-1) + torch.log(supp.clamp_min(1.0e-8))
        tap_logits = self.tap_logit_head(h)
        raw_offsets = self.offset_head(h).reshape(r, v, int(self.num_taps), 2)
        fixed_center, scale_factor = self._offset_scale_factor(global_step)
        radius_bound = torch.minimum(
            rad.unsqueeze(-1).unsqueeze(-1) * float(self.config.offset_scale) * float(scale_factor),
            rad.new_tensor(float(self.config.max_offset_px)),
        )
        offsets = torch.tanh(raw_offsets) * radius_bound
        if bool(fixed_center) or float(scale_factor) <= 0.0:
            offsets = torch.zeros_like(offsets)
        if bool(fixed_center):
            tap_logits = torch.full_like(tap_logits, -1.0e4)
            tap_logits[..., self.center_tap] = 0.0
        center_bias = torch.zeros((int(self.num_taps), 2), device=device, dtype=dtype)
        if int(self.num_taps) >= 5:
            center_bias[0] = torch.tensor([-1.0, 0.0], device=device, dtype=dtype)
            center_bias[1] = torch.tensor([0.0, -1.0], device=device, dtype=dtype)
            center_bias[2] = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
            center_bias[3] = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
            center_bias[4] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        else:
            center_bias[self.center_tap] = 0.0
        sample_uv = uv[:, :, None, :] + center_bias.reshape(1, 1, int(self.num_taps), 2) + offsets
        inbound = self._source_inbound_mask(
            sample_uv,
            image_height=int(image_height),
            image_width=int(image_width),
        )
        logits = view_logits[:, :, None] + tap_logits
        valid_tap = val[:, :, None] & inbound
        logits = torch.where(valid_tap, logits, torch.full_like(logits, -1.0e4))
        flat_weights = torch.softmax(logits.reshape(r, -1), dim=-1).reshape(r, v, int(self.num_taps))
        any_valid = valid_tap.reshape(r, -1).any(dim=-1)
        flat_weights = flat_weights * any_valid.to(dtype).reshape(r, 1, 1)
        out, inbound, backend_used = self._sample_weighted_sum(
            value_map=value_map,
            sample_uv=sample_uv,
            weights=flat_weights,
            valid=valid_tap,
            image_height=int(image_height),
            image_width=int(image_width),
            prepared_value_nchw=prepared,
        )
        valid_tap = val[:, :, None] & inbound
        any_valid = valid_tap.reshape(r, -1).any(dim=-1)
        if self.geometry_pe is not None:
            geom_pe = self.geometry_pe(geom)
            view_weights_for_pe = flat_weights.sum(dim=2)
            out = out + (geom_pe * view_weights_for_pe.unsqueeze(-1)).sum(dim=1)
        gate = 1.0 + 0.1 * torch.tanh(self.gate_head(q))
        out = out * gate
        confidence = (flat_weights * supp[:, :, None]).sum(dim=(1, 2))

        offset_norm = offsets.detach().norm(dim=-1).reshape(-1)
        valid_offsets = offset_norm[valid_tap.detach().reshape(-1)]
        if int(valid_offsets.numel()) == 0:
            valid_offsets = offset_norm.new_zeros((1,))
        view_weights = flat_weights.sum(dim=2)
        tap_weights = flat_weights.sum(dim=1)
        aux = {
            f"iforward/{prefix}_fixed_fast_path_enabled": 0.0,
            f"iforward/{prefix}_cuda_backend_enabled": 1.0 if backend_used == "cuda" else 0.0,
            f"iforward/{prefix}_pytorch_fallback_enabled": 1.0 if backend_used == "pytorch" else 0.0,
            f"iforward/{prefix}_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
        }
        if bool(emit_heavy_aux):
            aux.update(
                {
                    f"iforward/{prefix}_inbound_ratio": float((inbound & val[:, :, None]).detach().float().mean().item()),
                    f"iforward/{prefix}_offset_norm_mean": float(valid_offsets.float().mean().item()),
                    f"iforward/{prefix}_offset_norm_p95": float(torch.quantile(valid_offsets.float(), 0.95).item()),
                    f"iforward/{prefix}_view_entropy": float(
                        self._entropy(view_weights, dim=1).detach().float().mean().item()
                    ),
                    f"iforward/{prefix}_tap_entropy": float(
                        self._entropy(tap_weights, dim=1).detach().float().mean().item()
                    ),
                    f"iforward/{prefix}_output_rms": float(out.detach().float().square().mean().sqrt().item())
                    if out.numel()
                    else 0.0,
                    f"iforward/{prefix}_valid_ratio": float(any_valid.detach().float().mean().item())
                    if any_valid.numel()
                    else 0.0,
                }
            )
        oob = (~inbound & val[:, :, None]).to(dtype)
        reg = {
            "offset_l2": (offsets.square().sum(dim=-1) * flat_weights.detach()).sum(dim=(1, 2)).mean(),
            "out_of_bounds": (oob * flat_weights.detach()).sum(dim=(1, 2)).mean(),
        }
        return out, confidence, aux, reg


def support_center_sparse_gather(
    *,
    value_map: torch.Tensor,
    anchor_uv: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    image_height: int,
    image_width: int,
    backend: str = "auto",
    prepared_value_nchw: Optional[torch.Tensor] = None,
    chunk_size: int = 65536,
    emit_heavy_aux: bool = True,
    prefix: str = "stage3/support_center",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if value_map.dim() != 4:
        raise ValueError(f"value_map must be [V,H,W,C], got {tuple(value_map.shape)}")
    if anchor_uv.shape[:2] != support.shape[:2] or anchor_uv.shape[:2] != valid.shape[:2]:
        raise ValueError("anchor/support/valid shape mismatch")
    r = int(anchor_uv.shape[0])
    c = int(value_map.shape[-1])
    if r == 0:
        aux = {
            f"iforward/{prefix}_fixed_support_center_enabled": 1.0,
            f"iforward/{prefix}_cuda_backend_enabled": 0.0,
            f"iforward/{prefix}_pytorch_fallback_enabled": 0.0,
            f"iforward/{prefix}_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
        }
        return value_map.new_zeros((0, c)), value_map.new_zeros((0,)), aux

    row_chunk = int(chunk_size)
    if row_chunk <= 0:
        row_chunk = max(r, 1)
    if r > row_chunk:
        outs = []
        confs = []
        aux_weighted: Dict[str, float] = {}
        total_rows = 0
        prepared = prepared_value_nchw
        if prepared is None and (
            str(backend).lower() == "pytorch"
            or (str(backend).lower() == "auto" and not can_use_cuda_sparse_gather(value_map, backend="auto"))
        ):
            prepared = prepare_value_nchw(value_map)
        for start in range(0, r, row_chunk):
            end = min(start + row_chunk, r)
            rows = int(end - start)
            out_c, conf_c, aux_c = support_center_sparse_gather(
                value_map=value_map,
                anchor_uv=anchor_uv[start:end],
                support=support[start:end],
                valid=valid[start:end],
                image_height=int(image_height),
                image_width=int(image_width),
                backend=str(backend),
                prepared_value_nchw=prepared,
                chunk_size=row_chunk,
                emit_heavy_aux=emit_heavy_aux,
                prefix=prefix,
            )
            outs.append(out_c)
            confs.append(conf_c)
            total_rows += rows
            for key, value in aux_c.items():
                aux_weighted[str(key)] = aux_weighted.get(str(key), 0.0) + float(value) * float(rows)
        denom = float(max(total_rows, 1))
        return (
            torch.cat(outs, dim=0) if outs else value_map.new_zeros((0, c)),
            torch.cat(confs, dim=0) if confs else value_map.new_zeros((0,)),
            {key: float(value) / denom for key, value in aux_weighted.items()},
        )

    device = value_map.device
    dtype = value_map.dtype
    uv = anchor_uv.to(device=device, dtype=dtype)
    supp = support.to(device=device, dtype=dtype).clamp_min(0.0)
    val = valid.to(device=device, dtype=torch.bool)
    sample_uv = uv[:, :, None, :]
    inbound_pre = SparseGatherLift._source_inbound_mask(
        sample_uv,
        image_height=int(image_height),
        image_width=int(image_width),
    )[:, :, 0]
    valid_view = val & inbound_pre
    weight_raw = torch.where(valid_view, supp, torch.zeros_like(supp))
    denom = weight_raw.sum(dim=1, keepdim=True)
    row_valid = denom.squeeze(-1) > 0.0
    view_weights = weight_raw / denom.clamp_min(1.0e-8)
    prepared = prepared_value_nchw
    if prepared is None and (
        str(backend).lower() == "pytorch"
        or (str(backend).lower() == "auto" and not can_use_cuda_sparse_gather(value_map, backend="auto"))
    ):
        prepared = prepare_value_nchw(value_map)
    out, inbound, backend_used = sparse_gather_2d(
        value_map,
        sample_uv,
        view_weights.unsqueeze(-1),
        val[:, :, None],
        image_height=int(image_height),
        image_width=int(image_width),
        backend=str(backend).lower(),  # type: ignore[arg-type]
        prepared_value_nchw=prepared,
        chunk_size=int(row_chunk),
    )
    out = torch.where(row_valid.unsqueeze(-1), out, torch.zeros_like(out))
    confidence = (view_weights * supp).sum(dim=1) * row_valid.to(dtype)
    aux = {
        f"iforward/{prefix}_fixed_support_center_enabled": 1.0,
        f"iforward/{prefix}_cuda_backend_enabled": 1.0 if backend_used == "cuda" else 0.0,
        f"iforward/{prefix}_pytorch_fallback_enabled": 1.0 if backend_used == "pytorch" else 0.0,
        f"iforward/{prefix}_heavy_aux_enabled": 1.0 if bool(emit_heavy_aux) else 0.0,
    }
    if bool(emit_heavy_aux):
        inbound_center = inbound[:, :, 0]
        aux.update(
            {
                f"iforward/{prefix}_inbound_ratio": float((inbound_center & val).detach().float().mean().item()),
                f"iforward/{prefix}_view_entropy": float(
                    SparseGatherLift._entropy(view_weights, dim=1).detach().float().mean().item()
                )
                if int(view_weights.numel())
                else 0.0,
                f"iforward/{prefix}_output_rms": float(out.detach().float().square().mean().sqrt().item())
                if int(out.numel())
                else 0.0,
                f"iforward/{prefix}_valid_ratio": float(row_valid.detach().float().mean().item())
                if int(row_valid.numel())
                else 0.0,
            }
        )
    return out, confidence, aux


def center_child_detail_by_parent(
    detail: torch.Tensor,
    *,
    child_to_parent: torch.Tensor,
    weights: torch.Tensor,
    num_parents: int,
    eps: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if detail.dim() != 2:
        raise ValueError(f"detail must be [N,C], got {tuple(detail.shape)}")
    n = int(detail.shape[0])
    if int(child_to_parent.numel()) != n or int(weights.numel()) != n:
        raise ValueError("child_to_parent/weights row mismatch")
    if n == 0:
        return detail, detail.new_tensor(0.0)
    pid = child_to_parent.to(device=detail.device, dtype=torch.long).reshape(-1)
    acc_dtype = torch.float32 if detail.dtype in (torch.float16, torch.bfloat16, torch.float32) else detail.dtype
    detail_acc = detail.to(dtype=acc_dtype)
    w = weights.to(device=detail.device, dtype=acc_dtype).reshape(-1).clamp_min(0.0)
    weighted = torch.zeros((int(num_parents), int(detail.shape[-1])), device=detail.device, dtype=acc_dtype)
    denom = torch.zeros((int(num_parents), 1), device=detail.device, dtype=acc_dtype)
    weighted.index_add_(0, pid, detail_acc * w[:, None])
    denom.index_add_(0, pid, w[:, None])
    mean = weighted / denom.clamp_min(float(eps))
    centered_acc = detail_acc - mean.index_select(0, pid)
    for _ in range(2):
        check = torch.zeros((int(num_parents), int(detail.shape[-1])), device=detail.device, dtype=acc_dtype)
        check.index_add_(0, pid, centered_acc * w[:, None])
        residual = check / denom.clamp_min(float(eps))
        centered_acc = centered_acc - residual.index_select(0, pid)
    centered = centered_acc.to(dtype=detail.dtype)
    check_final = torch.zeros((int(num_parents), int(detail.shape[-1])), device=detail.device, dtype=acc_dtype)
    check_final.index_add_(0, pid, centered.to(dtype=acc_dtype) * w[:, None])
    err_acc = (
        (check_final / denom.clamp_min(float(eps))).detach().abs().max()
        if int(check_final.numel())
        else detail_acc.new_tensor(0.0)
    )
    err = err_acc.to(dtype=detail.dtype)
    return centered, err


__all__ = [
    "ChildQueryBuilder",
    "GatherConfig",
    "ParentContextFusion",
    "ParentQueryBuilder",
    "SparseGatherLift",
    "Stage3ChildDetailPack",
    "center_child_detail_by_parent",
    "support_center_sparse_gather",
]
