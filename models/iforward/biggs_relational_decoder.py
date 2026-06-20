from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .biggs_state import BigGSChildContributionCache, BigGSParentStats
from .cuda_grld_decode import grld_decode


def _zero_linear(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _last_linear(module: nn.Module) -> Optional[nn.Linear]:
    last = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last = child
    return last


def _normal_last_linear(module: nn.Module, *, std: float) -> None:
    last = _last_linear(module)
    if last is None:
        return
    nn.init.normal_(last.weight, mean=0.0, std=float(std))
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def _scatter_sum(values: torch.Tensor, parent_id: torch.Tensor, *, num_parents: int) -> torch.Tensor:
    out = values.new_zeros((int(num_parents), int(values.shape[-1])))
    if int(values.numel()) > 0:
        out.index_add_(0, parent_id.long(), values)
    return out


def _scatter_weighted_mean(
    values: torch.Tensor,
    parent_id: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_parents: int,
    eps: float,
) -> torch.Tensor:
    w = weights.reshape(-1, 1).to(device=values.device, dtype=values.dtype).clamp_min(float(eps))
    num = _scatter_sum(values * w, parent_id, num_parents=int(num_parents))
    den = _scatter_sum(w, parent_id, num_parents=int(num_parents))
    return num / den.clamp_min(float(eps))


def grld_decode_reference(
    *,
    base: torch.Tensor,
    detail: torch.Tensor,
    gate: torch.Tensor,
    coeff: torch.Tensor,
    child_to_parent: torch.Tensor,
    branch_scale: torch.Tensor,
    chunk_size: int = 65536,
) -> torch.Tensor:
    n = int(coeff.shape[0])
    if n == 0:
        return base.new_zeros((0, int(base.shape[-1])))
    pid = child_to_parent.long().to(device=base.device)
    scale = branch_scale.to(device=base.device, dtype=base.dtype).reshape(())
    chunks = []
    chunk = max(int(chunk_size), 1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        pid_c = pid[start:end]
        base_c = base.index_select(0, pid_c)
        detail_c = detail.index_select(0, pid_c)
        gate_c = gate.index_select(0, pid_c)
        coeff_c = coeff[start:end]
        residual_c = torch.einsum("nr,nr,nre->ne", coeff_c, gate_c, detail_c)
        chunks.append(base_c + scale * residual_c)
    return torch.cat(chunks, dim=0)


class GaussianRelationCodec(nn.Module):
    def __init__(
        self,
        *,
        relation_dim: int = 12,
        eps: float = 1.0e-6,
        detach_inputs: bool = True,
        rigid_relation_space: str = "world",
    ) -> None:
        super().__init__()
        if int(relation_dim) != 12:
            raise ValueError(f"GaussianRelationCodec v1 requires relation_dim=12, got {relation_dim}.")
        space_l = str(rigid_relation_space).lower()
        if space_l not in {"world", "canonical"}:
            raise ValueError(f"unsupported GRLD rigid_relation_space={rigid_relation_space!r}")
        self.relation_dim = int(relation_dim)
        self.eps = float(eps)
        self.detach_inputs = bool(detach_inputs)
        self.rigid_relation_space = space_l

    def _param(self, params: Dict[str, torch.Tensor], name: str, *, ref: torch.Tensor) -> torch.Tensor:
        value = params[name]
        if bool(self.detach_inputs):
            value = value.detach()
        return value.to(device=ref.device, dtype=ref.dtype)

    def _cache_tensor(self, value: torch.Tensor, *, ref: torch.Tensor) -> torch.Tensor:
        if bool(self.detach_inputs):
            value = value.detach()
        return value.to(device=ref.device, dtype=ref.dtype)

    def build_relation(
        self,
        *,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        child_cache: BigGSChildContributionCache,
        parent_stats: BigGSParentStats,
        child_to_parent: torch.Tensor,
        parent_count: torch.Tensor,
        branch_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        ref = child_params["means"]
        n = int(ref.shape[0])
        parent_id = child_to_parent.long().to(device=ref.device)
        if n == 0:
            return ref.new_zeros((0, self.relation_dim)), ref.new_zeros((0,)), {}
        if child_cache is None or parent_stats is None:
            raise RuntimeError("GRLD requires BigGS parent runtime child_cache and parent_stats")

        child_means = self._param(child_params, "means", ref=ref)
        child_opacity = self._param(child_params, "opacity_logit", ref=ref)
        child_sh_dc = self._param(child_params, "sh_dc", ref=ref)
        child_sh_rest = self._param(child_params, "sh_rest", ref=ref)

        parent_means_all = self._param(parent_params, "means", ref=ref)
        parent_scales_log_all = self._param(parent_params, "scales_log", ref=ref)
        parent_opacity_all = self._param(parent_params, "opacity_logit", ref=ref)
        parent_sh_dc_all = self._param(parent_params, "sh_dc", ref=ref)
        parent_sh_rest_all = self._param(parent_params, "sh_rest", ref=ref)

        parent_means = parent_means_all.index_select(0, parent_id)
        parent_scales_log = parent_scales_log_all.index_select(0, parent_id)
        parent_scales = torch.exp(parent_scales_log).clamp_min(float(self.eps))
        parent_opacity = parent_opacity_all.index_select(0, parent_id)
        parent_sh_dc = parent_sh_dc_all.index_select(0, parent_id)
        parent_sh_rest = parent_sh_rest_all.index_select(0, parent_id)

        mass = self._cache_tensor(child_cache.mass, ref=ref).reshape(-1).clamp_min(float(self.eps))
        if "diag_cov" in child_params:
            child_diag_cov = self._param(child_params, "diag_cov", ref=ref).reshape(n, 3).clamp_min(float(self.eps))
        else:
            child_diag_cov = self._cache_tensor(child_cache.diag_cov, ref=ref).reshape(n, 3).clamp_min(float(self.eps))
        parent_diag_cov = torch.exp(2.0 * parent_scales_log).clamp_min(float(self.eps))

        parent_counts_all = parent_count.to(device=ref.device, dtype=ref.dtype).reshape(-1).clamp_min(1.0)
        parent_weight_sum = self._cache_tensor(parent_stats.weight_sum, ref=ref).reshape(-1).clamp_min(float(self.eps))
        mean_mass_all = parent_weight_sum / parent_counts_all
        mean_mass = mean_mass_all.index_select(0, parent_id).clamp_min(float(self.eps))

        r_xyz = (child_means - parent_means) / parent_scales
        r_cov = torch.log(child_diag_cov + float(self.eps)) - torch.log(parent_diag_cov + float(self.eps))
        r_mass = (torch.log(mass + float(self.eps)) - torch.log(mean_mass + float(self.eps))).reshape(-1, 1)
        r_opacity = child_opacity.reshape(-1, 1) - parent_opacity.reshape(-1, 1)
        r_sh_dc = child_sh_dc - parent_sh_dc
        child_sh_energy = child_sh_rest.float().square().sum(dim=tuple(range(1, child_sh_rest.dim()))).sqrt().to(dtype=ref.dtype)
        parent_sh_energy = parent_sh_rest.float().square().sum(dim=tuple(range(1, parent_sh_rest.dim()))).sqrt().to(dtype=ref.dtype)
        r_sh_rest_energy = (
            torch.log(child_sh_energy + float(self.eps)) - torch.log(parent_sh_energy + float(self.eps))
        ).reshape(-1, 1)
        relation = torch.cat([r_xyz, r_cov, r_mass, r_opacity, r_sh_dc, r_sh_rest_energy], dim=-1)
        if not torch.isfinite(relation).all():
            raise RuntimeError("GRLD relation contains NaN/Inf")
        aux = {
            "relation_xyz_norm": float(r_xyz.detach().norm(dim=-1).mean().item()) if r_xyz.numel() else 0.0,
            "relation_cov_norm": float(r_cov.detach().norm(dim=-1).mean().item()) if r_cov.numel() else 0.0,
            "relation_cov_norm_before_norm": float(r_cov.detach().norm(dim=-1).mean().item()) if r_cov.numel() else 0.0,
            "relation_mass_norm": float(r_mass.detach().abs().mean().item()) if r_mass.numel() else 0.0,
            "relation_opacity_norm": float(r_opacity.detach().abs().mean().item()) if r_opacity.numel() else 0.0,
            "relation_sh_norm": float(torch.cat([r_sh_dc, r_sh_rest_energy], dim=-1).detach().norm(dim=-1).mean().item())
            if r_sh_dc.numel()
            else 0.0,
            "dynamic_mass_nan_ratio": float((~torch.isfinite(mass)).to(dtype=ref.dtype).mean().item()) if mass.numel() else 0.0,
            "rigid_relation_world_mode": 1.0 if int(branch_id) == 2 and self.rigid_relation_space == "world" else 0.0,
            "rigid_relation_canonical_mode": 1.0 if int(branch_id) == 2 and self.rigid_relation_space == "canonical" else 0.0,
        }
        return relation, mass, aux


class GaussianRelationalLiftingDecoder(nn.Module):
    def __init__(
        self,
        *,
        parent_event_dim: int = 64,
        fine_event_dim: int = 16,
        relation_dim: int = 12,
        rank: int = 4,
        fused_cuda: bool = True,
        detach_relation_inputs: bool = True,
        decode_chunk_size: int = 65536,
        relation_normalization: str = "none",
        relation_rms_floor: float = 0.05,
        relation_clip: float = 0.0,
        rigid_relation_space: str = "world",
        detail_head_init_std: float = 0.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.parent_event_dim = int(parent_event_dim)
        self.fine_event_dim = int(fine_event_dim)
        self.relation_dim = int(relation_dim)
        self.rank = int(rank)
        self.fused_cuda = bool(fused_cuda)
        self.decode_chunk_size = max(int(decode_chunk_size), 1)
        norm_l = str(relation_normalization).lower()
        if norm_l not in {"none", "sibling_rms"}:
            raise ValueError(f"unsupported GRLD relation_normalization={relation_normalization!r}")
        self.relation_normalization = norm_l
        self.relation_rms_floor = float(relation_rms_floor)
        self.relation_clip = float(relation_clip)
        self.eps = float(eps)
        self.codec = GaussianRelationCodec(
            relation_dim=int(relation_dim),
            eps=float(eps),
            detach_inputs=bool(detach_relation_inputs),
            rigid_relation_space=str(rigid_relation_space),
        )
        self.summary_dim = int(relation_dim) + 2
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(self.summary_dim),
            nn.Linear(self.summary_dim, int(parent_event_dim)),
        )
        self.base_head = nn.Linear(int(parent_event_dim), int(fine_event_dim))
        self.detail_head = nn.Sequential(
            nn.LayerNorm(int(parent_event_dim)),
            nn.Linear(int(parent_event_dim), int(rank) * int(fine_event_dim)),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(int(parent_event_dim)),
            nn.Linear(int(parent_event_dim), int(rank)),
        )
        self.relation_proj = nn.Linear(int(relation_dim), int(rank), bias=False)
        _zero_linear(self.summary_proj)
        _zero_linear(self.detail_head)
        _zero_linear(self.gate_head)
        if float(detail_head_init_std) > 0.0:
            _normal_last_linear(self.detail_head, std=float(detail_head_init_std))

    def _center_and_summarize(
        self,
        *,
        relation: torch.Tensor,
        parent_id: torch.Tensor,
        weights: torch.Tensor,
        parent_count: torch.Tensor,
        num_parents: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if int(relation.numel()) == 0:
            return relation, relation.new_zeros((int(num_parents), self.summary_dim)), {}
        pid = parent_id.long().to(device=relation.device)
        mass = weights.reshape(-1).to(device=relation.device, dtype=relation.dtype).clamp_min(float(self.eps))
        mean = _scatter_weighted_mean(relation, pid, mass, num_parents=int(num_parents), eps=float(self.eps))
        centered = relation - mean.index_select(0, pid)
        rms = _scatter_weighted_mean(centered.square(), pid, mass, num_parents=int(num_parents), eps=float(self.eps))
        rms = torch.sqrt(rms + float(self.eps))
        coeff_input = centered
        if self.relation_normalization == "sibling_rms":
            scale = rms.clamp_min(float(self.relation_rms_floor))
            coeff_input = centered / scale.index_select(0, pid)
            if float(self.relation_clip) > 0.0:
                coeff_input = coeff_input.clamp(-float(self.relation_clip), float(self.relation_clip))
            coeff_mean = _scatter_weighted_mean(
                coeff_input,
                pid,
                mass,
                num_parents=int(num_parents),
                eps=float(self.eps),
            )
            coeff_input = coeff_input - coeff_mean.index_select(0, pid)
        denom = _scatter_sum(mass.reshape(-1, 1), pid, num_parents=int(num_parents)).clamp_min(float(self.eps))
        pi = mass.reshape(-1, 1) / denom.index_select(0, pid)
        entropy = _scatter_sum(-pi * torch.log(pi.clamp_min(float(self.eps))), pid, num_parents=int(num_parents))
        log_child_count = torch.log(
            parent_count.to(device=relation.device, dtype=relation.dtype).reshape(-1, 1).clamp_min(1.0)
        )
        summary = torch.cat([rms, entropy, log_child_count], dim=-1)
        active = parent_count.to(device=relation.device).reshape(-1) > 0
        centered_mean = _scatter_weighted_mean(centered, pid, mass, num_parents=int(num_parents), eps=float(self.eps))
        coeff_mean = _scatter_weighted_mean(coeff_input, pid, mass, num_parents=int(num_parents), eps=float(self.eps))
        raw_err = float(centered_mean[active].detach().norm(dim=-1).max().item()) if bool(active.any()) else 0.0
        err = float(coeff_mean[active].detach().norm(dim=-1).max().item()) if bool(active.any()) else 0.0
        aux = {
            "relation_centering_error": err,
            "relation_raw_centering_error": raw_err,
            "relation_channel_rms_min": float(rms[active].detach().min().item()) if bool(active.any()) else 0.0,
            "relation_channel_rms_max": float(rms[active].detach().max().item()) if bool(active.any()) else 0.0,
            "relation_cov_norm_after_norm": (
                float(coeff_input[:, 3:6].detach().norm(dim=-1).mean().item()) if coeff_input.numel() else 0.0
            ),
        }
        return coeff_input, summary, aux

    def _decode(
        self,
        *,
        base: torch.Tensor,
        detail: torch.Tensor,
        gate: torch.Tensor,
        coeff: torch.Tensor,
        child_to_parent: torch.Tensor,
        child_order: torch.Tensor,
        parent_start: torch.Tensor,
        parent_count: torch.Tensor,
        branch_scale: torch.Tensor,
    ) -> torch.Tensor:
        if bool(self.fused_cuda):
            return grld_decode(
                base,
                detail,
                gate,
                coeff,
                child_to_parent,
                child_order,
                parent_start,
                parent_count,
                branch_scale,
            )
        return grld_decode_reference(
            base=base,
            detail=detail,
            gate=gate,
            coeff=coeff,
            child_to_parent=child_to_parent,
            branch_scale=branch_scale,
            chunk_size=int(self.decode_chunk_size),
        )

    def decode_branch(
        self,
        *,
        parent_event: torch.Tensor,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        child_cache: BigGSChildContributionCache,
        parent_stats: BigGSParentStats,
        child_to_parent: torch.Tensor,
        parent_start: torch.Tensor,
        parent_count: torch.Tensor,
        child_order: torch.Tensor,
        branch_id: int,
        branch_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        t0 = time.perf_counter()
        pid = child_to_parent.long().to(device=parent_event.device)
        relation, mass, rel_aux = self.codec.build_relation(
            child_params=child_params,
            parent_params=parent_params,
            child_cache=child_cache,
            parent_stats=parent_stats,
            child_to_parent=pid,
            parent_count=parent_count,
            branch_id=int(branch_id),
        )
        relation = relation.to(device=parent_event.device, dtype=parent_event.dtype)
        mass = mass.to(device=parent_event.device, dtype=parent_event.dtype)
        centered, summary, center_aux = self._center_and_summarize(
            relation=relation,
            parent_id=pid,
            weights=mass,
            parent_count=parent_count.to(device=parent_event.device),
            num_parents=int(parent_event.shape[0]),
        )
        coeff = self.relation_proj(centered)
        h_parent = parent_event + self.summary_proj(summary.to(device=parent_event.device, dtype=parent_event.dtype))
        base = self.base_head(h_parent)
        detail = self.detail_head(h_parent).reshape(int(parent_event.shape[0]), int(self.rank), int(self.fine_event_dim))
        gate = 1.0 + torch.tanh(self.gate_head(h_parent))
        t_relation_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        fine = self._decode(
            base=base,
            detail=detail,
            gate=gate,
            coeff=coeff,
            child_to_parent=pid,
            child_order=child_order.to(device=parent_event.device, dtype=torch.long),
            parent_start=parent_start.to(device=parent_event.device, dtype=torch.long),
            parent_count=parent_count.to(device=parent_event.device, dtype=torch.long),
            branch_scale=branch_scale.to(device=parent_event.device, dtype=parent_event.dtype),
        )
        t_decode_ms = (time.perf_counter() - t0) * 1000.0
        base_child = base.index_select(0, pid)
        residual = fine - base_child
        mean_residual = _scatter_weighted_mean(
            residual,
            pid,
            mass,
            num_parents=int(parent_event.shape[0]),
            eps=float(self.eps),
        )
        active = parent_count.to(device=parent_event.device).reshape(-1) > 0
        mean_error = float(mean_residual[active].detach().norm(dim=-1).max().item()) if bool(active.any()) else 0.0
        aux = {
            **rel_aux,
            **center_aux,
            "weighted_mean_error": mean_error,
            "relation_ms": float(t_relation_ms),
            "decode_ms": float(t_decode_ms),
            "detail_to_base_ratio": (
                float(residual.detach().norm(dim=-1).mean().item() / max(float(base_child.detach().norm(dim=-1).mean().item()), 1.0e-8))
                if residual.numel()
                else 0.0
            ),
        }
        return fine, base_child, residual, aux


__all__ = [
    "GaussianRelationCodec",
    "GaussianRelationalLiftingDecoder",
    "grld_decode_reference",
]
