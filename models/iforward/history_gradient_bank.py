from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from models.streetforward.stage6_0 import LocalGSState
from models.streetforward.stage6_0.local_gs_state import LocalBranchState

from .utils import cfg_get


HGV2_ATTRS = ("means", "scales", "quat", "opacity", "sh")


@dataclass
class GradientBankAttr:
    direction: torch.Tensor
    log_norm: torch.Tensor
    valid: torch.Tensor

    def detach(self) -> "GradientBankAttr":
        return GradientBankAttr(
            direction=self.direction.detach().clone(),
            log_norm=self.log_norm.detach().clone(),
            valid=self.valid.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "GradientBankAttr":
        out_dtype = dtype or self.direction.dtype
        return GradientBankAttr(
            direction=self.direction.to(device=device, dtype=out_dtype),
            log_norm=self.log_norm.to(device=device, dtype=out_dtype),
            valid=self.valid.to(device=device),
        )


@dataclass
class HistoryGradientBranchBank:
    means: GradientBankAttr
    scales: GradientBankAttr
    quat: GradientBankAttr
    opacity: GradientBankAttr
    sh: GradientBankAttr

    def detach(self) -> "HistoryGradientBranchBank":
        return HistoryGradientBranchBank(
            means=self.means.detach(),
            scales=self.scales.detach(),
            quat=self.quat.detach(),
            opacity=self.opacity.detach(),
            sh=self.sh.detach(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "HistoryGradientBranchBank":
        return HistoryGradientBranchBank(
            means=self.means.to(device=device, dtype=dtype),
            scales=self.scales.to(device=device, dtype=dtype),
            quat=self.quat.to(device=device, dtype=dtype),
            opacity=self.opacity.to(device=device, dtype=dtype),
            sh=self.sh.to(device=device, dtype=dtype),
        )


@dataclass
class HistoryGradientBank:
    bg: HistoryGradientBranchBank
    distant: Optional[HistoryGradientBranchBank] = None
    rigid: Optional[HistoryGradientBranchBank] = None
    valid: bool = False
    source_rollout_id: int = -1
    source_history_loss: float = 0.0
    source_history_num_refs: int = 0

    def detach(self) -> "HistoryGradientBank":
        return HistoryGradientBank(
            bg=self.bg.detach(),
            distant=None if self.distant is None else self.distant.detach(),
            rigid=None if self.rigid is None else self.rigid.detach(),
            valid=bool(self.valid),
            source_rollout_id=int(self.source_rollout_id),
            source_history_loss=float(self.source_history_loss),
            source_history_num_refs=int(self.source_history_num_refs),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "HistoryGradientBank":
        return HistoryGradientBank(
            bg=self.bg.to(device=device, dtype=dtype),
            distant=None if self.distant is None else self.distant.to(device=device, dtype=dtype),
            rigid=None if self.rigid is None else self.rigid.to(device=device, dtype=dtype),
            valid=bool(self.valid),
            source_rollout_id=int(self.source_rollout_id),
            source_history_loss=float(self.source_history_loss),
            source_history_num_refs=int(self.source_history_num_refs),
        )


def _bool_cfg(node: Any, key: str, default: bool) -> bool:
    return bool(cfg_get(node, key, default))


def _storage_dtype(cfg: Any, ref_dtype: torch.dtype) -> torch.dtype:
    raw = str(cfg_get(cfg, "dtype", "fp16")).lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"fp32", "float32", "single"}:
        return torch.float32
    if raw in {"same", "input"}:
        return ref_dtype
    raise ValueError(f"unsupported history_gate_v2.bank.dtype={raw!r}")


def _attr_enabled(attrs_cfg: Any) -> Dict[str, bool]:
    return {name: _bool_cfg(attrs_cfg, name, True) for name in HGV2_ATTRS}


def _flatten_rows(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(int(value.shape[0]), -1)


def _empty_attr_like(value: torch.Tensor, *, dtype: torch.dtype) -> GradientBankAttr:
    n = int(value.shape[0])
    direction = torch.zeros_like(value.detach(), dtype=dtype)
    log_norm = value.detach().new_zeros((n, 1), dtype=dtype)
    valid = torch.zeros((n,), device=value.device, dtype=torch.bool)
    return GradientBankAttr(direction=direction, log_norm=log_norm, valid=valid)


def _pack_attr(
    grad: Optional[torch.Tensor],
    ref: torch.Tensor,
    *,
    enabled: bool,
    dtype: torch.dtype,
    min_grad_norm: float,
    eps: float,
) -> GradientBankAttr:
    if (not bool(enabled)) or grad is None:
        return _empty_attr_like(ref, dtype=dtype)
    g = grad.detach().to(device=ref.device, dtype=ref.dtype)
    if tuple(g.shape) != tuple(ref.shape):
        raise ValueError(f"HGV2 grad shape {tuple(g.shape)} does not match ref {tuple(ref.shape)}")
    flat = _flatten_rows(g)
    finite = torch.isfinite(flat).all(dim=-1)
    norm = flat.norm(dim=-1, keepdim=True)
    valid = finite & (norm[:, 0] > float(min_grad_norm))
    direction = g / norm.reshape((int(g.shape[0]),) + (1,) * (g.dim() - 1)).clamp_min(float(eps))
    direction = torch.where(valid.reshape((int(g.shape[0]),) + (1,) * (g.dim() - 1)), direction, torch.zeros_like(direction))
    log_norm = torch.log(norm.clamp_min(float(eps)))
    log_norm = torch.where(valid[:, None], log_norm, torch.zeros_like(log_norm))
    return GradientBankAttr(
        direction=direction.to(dtype=dtype),
        log_norm=log_norm.to(dtype=dtype),
        valid=valid.to(dtype=torch.bool),
    )


def _zero_sh_ref(branch: LocalBranchState) -> torch.Tensor:
    n = int(branch.sh_dc.shape[0])
    return torch.cat([branch.sh_dc[:, None, :], branch.sh_rest], dim=1).reshape(n, -1)


def _combine_sh_grad(
    branch: LocalBranchState,
    grad_map: Mapping[Tuple[str, str], Optional[torch.Tensor]],
    branch_name: str,
) -> torch.Tensor:
    sh_dc = grad_map.get((str(branch_name), "sh_dc"))
    sh_rest = grad_map.get((str(branch_name), "sh_rest"))
    sh_dc = torch.zeros_like(branch.sh_dc) if sh_dc is None else sh_dc
    sh_rest = torch.zeros_like(branch.sh_rest) if sh_rest is None else sh_rest
    n = int(branch.sh_dc.shape[0])
    return torch.cat([sh_dc[:, None, :], sh_rest], dim=1).reshape(n, -1)


def _collect_branch_params(
    branch: Optional[LocalBranchState],
    *,
    branch_name: str,
    params: List[torch.Tensor],
    names: List[Tuple[str, str]],
) -> None:
    if branch is None:
        return
    for attr_name, tensor in (
        ("means", branch.means),
        ("scales", branch.scales_log),
        ("quat", branch.quats),
        ("opacity", branch.opacity_logit),
        ("sh_dc", branch.sh_dc),
        ("sh_rest", branch.sh_rest),
    ):
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor) and bool(tensor.requires_grad):
            params.append(tensor)
            names.append((str(branch_name), str(attr_name)))


def _branch_from_grad_map(
    branch: Optional[LocalBranchState],
    *,
    branch_name: str,
    grad_map: Mapping[Tuple[str, str], Optional[torch.Tensor]],
    attrs: Mapping[str, bool],
    dtype: torch.dtype,
    min_grad_norm: float,
    eps: float,
) -> Optional[HistoryGradientBranchBank]:
    if branch is None:
        return None
    sh_ref = _zero_sh_ref(branch)
    return HistoryGradientBranchBank(
        means=_pack_attr(
            grad_map.get((branch_name, "means")),
            branch.means,
            enabled=bool(attrs.get("means", True)),
            dtype=dtype,
            min_grad_norm=float(min_grad_norm),
            eps=float(eps),
        ),
        scales=_pack_attr(
            grad_map.get((branch_name, "scales")),
            branch.scales_log,
            enabled=bool(attrs.get("scales", True)),
            dtype=dtype,
            min_grad_norm=float(min_grad_norm),
            eps=float(eps),
        ),
        quat=_pack_attr(
            grad_map.get((branch_name, "quat")),
            branch.quats,
            enabled=bool(attrs.get("quat", True)),
            dtype=dtype,
            min_grad_norm=float(min_grad_norm),
            eps=float(eps),
        ),
        opacity=_pack_attr(
            grad_map.get((branch_name, "opacity")),
            branch.opacity_logit,
            enabled=bool(attrs.get("opacity", True)),
            dtype=dtype,
            min_grad_norm=float(min_grad_norm),
            eps=float(eps),
        ),
        sh=_pack_attr(
            _combine_sh_grad(branch, grad_map, branch_name),
            sh_ref,
            enabled=bool(attrs.get("sh", True)),
            dtype=dtype,
            min_grad_norm=float(min_grad_norm),
            eps=float(eps),
        ),
    )


def _bank_has_valid_rows(bank: HistoryGradientBank) -> bool:
    for branch in (bank.bg, bank.distant, bank.rigid):
        if branch is None:
            continue
        for attr in HGV2_ATTRS:
            if bool(getattr(branch, attr).valid.any().item()):
                return True
    return False


def build_history_gradient_bank_from_loss(
    *,
    loss_history: torch.Tensor,
    final_local_state: LocalGSState,
    rollout_id: int,
    history_num_refs: int,
    cfg: Any,
) -> Optional[HistoryGradientBank]:
    if int(history_num_refs) <= 0:
        return None
    if not torch.is_tensor(loss_history) or not bool(getattr(loss_history, "requires_grad", False)):
        return None
    if int(loss_history.numel()) != 1:
        loss_history = loss_history.reshape(-1).mean()
    if not bool(torch.isfinite(loss_history.detach()).all().item()):
        return None

    bank_cfg = cfg_get(cfg, "bank", {}) or {}
    attrs_cfg = cfg_get(cfg, "attrs", {}) or {}
    attrs = _attr_enabled(attrs_cfg)
    dtype = _storage_dtype(bank_cfg, final_local_state.bg.means.dtype)
    min_grad_norm = float(cfg_get(bank_cfg, "min_grad_norm", 1.0e-8))
    eps = float(cfg_get(bank_cfg, "eps", 1.0e-8))

    params: List[torch.Tensor] = []
    names: List[Tuple[str, str]] = []
    _collect_branch_params(final_local_state.bg, branch_name="bg", params=params, names=names)
    _collect_branch_params(final_local_state.distant, branch_name="distant", params=params, names=names)
    _collect_branch_params(final_local_state.rigid, branch_name="rigid", params=params, names=names)
    if not params:
        return None

    grads = torch.autograd.grad(
        loss_history,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_map: Dict[Tuple[str, str], Optional[torch.Tensor]] = {
        name: grad for name, grad in zip(names, grads)
    }
    bank = HistoryGradientBank(
        bg=_branch_from_grad_map(
            final_local_state.bg,
            branch_name="bg",
            grad_map=grad_map,
            attrs=attrs,
            dtype=dtype,
            min_grad_norm=min_grad_norm,
            eps=eps,
        ),
        distant=_branch_from_grad_map(
            final_local_state.distant,
            branch_name="distant",
            grad_map=grad_map,
            attrs=attrs,
            dtype=dtype,
            min_grad_norm=min_grad_norm,
            eps=eps,
        ),
        rigid=_branch_from_grad_map(
            final_local_state.rigid,
            branch_name="rigid",
            grad_map=grad_map,
            attrs=attrs,
            dtype=dtype,
            min_grad_norm=min_grad_norm,
            eps=eps,
        ),
        valid=True,
        source_rollout_id=int(rollout_id),
        source_history_loss=float(loss_history.detach().item()),
        source_history_num_refs=int(history_num_refs),
    )
    bank.valid = bool(_bank_has_valid_rows(bank))
    return bank if bool(bank.valid) else None


__all__ = [
    "GradientBankAttr",
    "HGV2_ATTRS",
    "HistoryGradientBank",
    "HistoryGradientBranchBank",
    "build_history_gradient_bank_from_loss",
]
