from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat
from models.streetforward.stage6_0.event_encoder import EventPack

from .biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from .utils import cfg_get


def _zero_last_linear(module: nn.Module) -> None:
    last = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last = child
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    last = int(in_dim)
    for _ in range(max(int(num_layers) - 1, 0)):
        layers.extend([nn.Linear(last, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU()])
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


def _scatter_weighted_mean(
    values: torch.Tensor,
    parent_id: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_parents: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((int(num_parents), int(values.shape[-1])))
    w = weights.reshape(-1, 1).to(device=values.device, dtype=values.dtype).clamp_min(1.0e-8)
    out = values.new_zeros((int(num_parents), int(values.shape[-1])))
    denom = values.new_zeros((int(num_parents), 1))
    out.index_add_(0, parent_id.long(), values * w)
    denom.index_add_(0, parent_id.long(), w)
    return out / denom.clamp_min(1.0e-8)


def _broadcast_optional(x: Optional[torch.Tensor], parent_id: torch.Tensor) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if int(parent_id.numel()) == 0:
        shape = (0,) + tuple(x.shape[1:])
        return x.new_zeros(shape)
    return x.index_select(0, parent_id.long())


class BigGSToFineEventDecoder(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int = 48,
        parent_event_dim: Optional[int] = None,
        fine_event_dim: Optional[int] = None,
        mode: str = "low_rank_basis",
        rank: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        residual_scale_init: float = 1.0e-3,
        residual_scale_learnable: bool = True,
        mean_preserve: bool = True,
        final_norm: str = "identity",
        branch_embed_dim: int = 4,
        zero_init_last: bool = True,
        detach_child_code_inputs: bool = True,
        detach_child_params: Optional[bool] = None,
        detach_parent_params: Optional[bool] = None,
        child_code_parent_local_frame: bool = True,
        residual_scale_per_branch: bool = False,
        whdd_chunk_size: int = 65536,
        fused_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.parent_event_dim = int(parent_event_dim) if parent_event_dim is not None else int(event_dim)
        self.fine_event_dim = int(fine_event_dim) if fine_event_dim is not None else int(event_dim)
        self.event_dim = int(self.fine_event_dim)
        self.mode = str(mode)
        self.rank = int(rank)
        self.fused_cuda = bool(fused_cuda)
        self.mean_preserve = bool(mean_preserve)
        self.detach_child_code_inputs = bool(detach_child_code_inputs)
        self.detach_child_params = (
            bool(detach_child_params) if detach_child_params is not None else bool(detach_child_code_inputs)
        )
        self.detach_parent_params = (
            bool(detach_parent_params) if detach_parent_params is not None else bool(detach_child_code_inputs)
        )
        self.child_code_parent_local_frame = bool(child_code_parent_local_frame)
        self.residual_scale_per_branch = bool(residual_scale_per_branch)
        self.whdd_chunk_size = max(int(whdd_chunk_size), 1)
        mode_l = str(self.mode).lower()
        if mode_l != "whdd_compact_fixed_basis" and int(self.parent_event_dim) != int(self.fine_event_dim):
            raise ValueError(
                f"BigGS child decoder mode={self.mode!r} requires parent_event_dim == fine_event_dim; "
                f"got parent={int(self.parent_event_dim)} fine={int(self.fine_event_dim)}"
            )
        self.branch_embed = nn.Embedding(3, int(branch_embed_dim))
        self.child_code_dim = 13 + int(branch_embed_dim)
        self.residual_mlp = _mlp(
            self.child_code_dim + int(self.fine_event_dim),
            int(hidden_dim),
            int(self.fine_event_dim),
            int(num_layers),
        )
        self.basis_mlp = _mlp(
            int(self.parent_event_dim),
            int(hidden_dim),
            int(rank) * int(self.fine_event_dim),
            int(num_layers),
        )
        self.coeff_mlp = _mlp(self.child_code_dim, int(hidden_dim), int(rank), int(num_layers))
        self.base_proj: nn.Module
        if mode_l == "whdd_compact_fixed_basis":
            self.base_proj = nn.Linear(int(self.parent_event_dim), int(self.fine_event_dim))
        else:
            self.base_proj = nn.Identity()
        self.detail_head = nn.Sequential(
            nn.LayerNorm(int(self.parent_event_dim)),
            nn.Linear(int(self.parent_event_dim), int(rank) * int(self.fine_event_dim)),
        )
        if bool(zero_init_last):
            if mode_l == "low_rank_basis":
                _zero_last_linear(self.coeff_mlp)
            elif mode_l == "residual_mlp":
                _zero_last_linear(self.residual_mlp)
            elif mode_l in {"whdd_fixed_basis", "whdd_compact_fixed_basis"}:
                _zero_last_linear(self.detail_head)
        scale = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        if bool(residual_scale_learnable):
            init = scale.repeat(3) if bool(self.residual_scale_per_branch) else scale
            self.residual_scale = nn.Parameter(init)
        else:
            init = scale.repeat(3) if bool(self.residual_scale_per_branch) else scale
            self.register_buffer("residual_scale", init)
        norm_l = str(final_norm).lower()
        if norm_l == "layernorm":
            self.final_norm = nn.LayerNorm(int(self.fine_event_dim))
        elif norm_l in {"identity", "none"}:
            self.final_norm = nn.Identity()
        else:
            raise ValueError(f"unsupported BigGS child decoder final_norm={final_norm!r}")

    @classmethod
    def from_config(cls, cfg: Any, *, event_dim: int) -> "BigGSToFineEventDecoder":
        parent_event_dim = cfg_get(cfg, "parent_event_dim", event_dim)
        fine_event_dim = cfg_get(cfg, "fine_event_dim", event_dim)
        return cls(
            event_dim=int(event_dim),
            parent_event_dim=int(parent_event_dim),
            fine_event_dim=int(fine_event_dim),
            mode=str(cfg_get(cfg, "mode", "low_rank_basis")),
            rank=int(cfg_get(cfg, "rank", 4)),
            hidden_dim=int(cfg_get(cfg, "hidden_dim", 64)),
            num_layers=int(cfg_get(cfg, "num_layers", 2)),
            residual_scale_init=float(cfg_get(cfg, "residual_scale_init", 1.0e-3)),
            residual_scale_learnable=bool(cfg_get(cfg, "residual_scale_learnable", True)),
            mean_preserve=bool(cfg_get(cfg, "mean_preserve", True)),
            final_norm=str(cfg_get(cfg, "final_norm", "identity")),
            branch_embed_dim=int(cfg_get(cfg_get(cfg, "child_code", {}) or {}, "branch_embed_dim", 4)),
            zero_init_last=bool(cfg_get(cfg, "zero_init_last", True)),
            detach_child_code_inputs=bool(cfg_get(cfg, "detach_child_code_inputs", True)),
            detach_child_params=cfg_get(cfg, "detach_child_params", None),
            detach_parent_params=cfg_get(cfg, "detach_parent_params", None),
            child_code_parent_local_frame=bool(cfg_get(cfg, "child_code_parent_local_frame", True)),
            residual_scale_per_branch=bool(cfg_get(cfg, "residual_scale_per_branch", False)),
            whdd_chunk_size=int(cfg_get(cfg, "whdd_chunk_size", 65536)),
            fused_cuda=bool(cfg_get(cfg, "fused_cuda", False)),
        )

    def _residual_scale_for_branch(self, *, branch_id: int, ref: torch.Tensor) -> torch.Tensor:
        scale = self.residual_scale.to(device=ref.device, dtype=ref.dtype)
        if scale.dim() == 0:
            return scale
        idx = max(0, min(int(branch_id), int(scale.numel()) - 1))
        return scale[idx]

    def _decode_whdd_residual(
        self,
        *,
        parent_event: torch.Tensor,
        parent_id: torch.Tensor,
        child_basis: torch.Tensor,
    ) -> torch.Tensor:
        n = int(parent_id.numel())
        if n == 0:
            return parent_event.new_zeros((0, int(self.fine_event_dim)))
        detail = self.detail_head(parent_event).reshape(
            int(parent_event.shape[0]),
            int(self.rank),
            int(self.fine_event_dim),
        )
        basis = child_basis.to(device=parent_event.device, dtype=parent_event.dtype)
        if int(basis.shape[0]) != n:
            raise ValueError("BigGS WHDD child_basis row mismatch")
        if int(basis.shape[1]) < int(self.rank):
            raise ValueError(f"BigGS WHDD child_basis rank {int(basis.shape[1])} < decoder rank {self.rank}")
        basis = basis[:, : int(self.rank)].detach()
        out_chunks = []
        chunk = int(self.whdd_chunk_size)
        pid = parent_id.long().to(device=parent_event.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pid_c = pid[start:end]
            detail_c = detail.index_select(0, pid_c)
            basis_c = basis[start:end]
            out_chunks.append(torch.einsum("nr,nre->ne", basis_c, detail_c))
        return torch.cat(out_chunks, dim=0) if out_chunks else parent_event.new_zeros((0, int(self.fine_event_dim)))

    def _child_code(
        self,
        *,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        parent_id: torch.Tensor,
        child_mass: torch.Tensor,
        parent_count: torch.Tensor,
        parent_mass_mean: torch.Tensor,
        branch_id: int,
        route_flag: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n = int(parent_id.numel())
        ref = child_params["means"]
        if n == 0:
            return ref.new_zeros((0, self.child_code_dim))
        pid = parent_id.long()
        parent_means = parent_params["means"].index_select(0, pid)
        parent_scales_log = parent_params["scales_log"].index_select(0, pid)
        parent_scales = torch.exp(parent_scales_log).clamp_min(1.0e-3)
        rel_delta = child_params["means"] - parent_means
        if bool(self.child_code_parent_local_frame):
            parent_quats = parent_params["quats"].index_select(0, pid)
            parent_rot = _quat_to_rotmat(_normalize_quat(parent_quats))
            rel_delta = torch.bmm(parent_rot.transpose(1, 2), rel_delta.unsqueeze(-1)).squeeze(-1)
        rel_xyz = rel_delta / parent_scales
        rel_scale = child_params["scales_log"] - parent_scales_log
        parent_opacity = parent_params["opacity_logit"].index_select(0, pid)
        rel_opacity = child_params["opacity_logit"] - parent_opacity
        mass = child_mass.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype).clamp_min(1.0e-8)
        parent_mass = parent_mass_mean.index_select(0, pid).reshape(-1, 1).to(device=ref.device, dtype=ref.dtype).clamp_min(1.0e-8)
        log_mass_ratio = torch.log(mass / parent_mass)
        child_count = parent_count.index_select(0, pid).reshape(-1, 1).to(device=ref.device, dtype=ref.dtype).clamp_min(1.0)
        log_child_count = torch.log(child_count)
        if route_flag is None:
            route = ref.new_zeros((n, 1))
        else:
            route = route_flag.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)
            if int(route.shape[0]) != n:
                raise ValueError("BigGS child decoder route_flag row mismatch")
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), dtype=torch.long, device=ref.device)
        ).to(dtype=ref.dtype)
        code = torch.cat(
            [
                rel_xyz,
                rel_xyz.square(),
                rel_scale,
                rel_opacity,
                log_mass_ratio,
                log_child_count,
                route,
                branch,
            ],
            dim=-1,
        )
        if not torch.isfinite(code).all():
            raise RuntimeError("BigGS child code contains NaN/Inf")
        return code

    def _decode_branch(
        self,
        *,
        parent_event: Optional[torch.Tensor],
        parent_support: Optional[torch.Tensor],
        parent_valid: Optional[torch.Tensor],
        parent_obs: Optional[torch.Tensor],
        parent_id: torch.Tensor,
        child_mass: torch.Tensor,
        parent_count: torch.Tensor,
        parent_mass_mean: torch.Tensor,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        branch_id: int,
        route_flag: Optional[torch.Tensor] = None,
        child_basis: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        n = int(parent_id.numel())
        if n == 0:
            ref = child_params["means"]
            return (
                ref.new_zeros((0, self.fine_event_dim)),
                _broadcast_optional(parent_support, parent_id),
                _broadcast_optional(parent_valid, parent_id),
                _broadcast_optional(parent_obs, parent_id),
                {},
            )
        if parent_event is None:
            raise RuntimeError("BigGS child decoder requires parent_event for non-empty branch")
        if int(parent_event.shape[-1]) != int(self.parent_event_dim):
            raise ValueError(
                f"BigGS parent_event dim mismatch: got {int(parent_event.shape[-1])}, "
                f"expected {int(self.parent_event_dim)}"
            )
        if int(parent_event.shape[0]) <= int(parent_id.max().item()):
            raise ValueError("BigGS parent_id exceeds parent_event rows")
        pid = parent_id.long().to(device=parent_event.device)
        mode = str(self.mode).lower()
        residual_scale = self._residual_scale_for_branch(branch_id=int(branch_id), ref=parent_event)
        if mode == "broadcast":
            parent_e = parent_event.index_select(0, pid)
            fine = parent_e
            residual = parent_e.new_zeros(parent_e.shape)
            scaled_residual = residual
        elif mode in {"whdd_fixed_basis", "whdd_compact_fixed_basis"}:
            if child_basis is None:
                raise RuntimeError("BigGS WHDD decoder requires assignment child_basis")
            residual = self._decode_whdd_residual(
                parent_event=parent_event,
                parent_id=pid,
                child_basis=child_basis,
            )
            scaled_residual = residual_scale * residual
            if mode == "whdd_compact_fixed_basis":
                base = self.base_proj(parent_event).index_select(0, pid)
            else:
                base = parent_event.index_select(0, pid)
            fine = base + scaled_residual
        else:
            parent_e = parent_event.index_select(0, pid)
            def prepare_param_dict(params: Dict[str, torch.Tensor], *, detach: bool) -> Dict[str, torch.Tensor]:
                out = {}
                for key, value in params.items():
                    x = value.detach() if bool(detach) else value
                    out[key] = x.to(device=parent_event.device, dtype=parent_event.dtype)
                return out

            code = self._child_code(
                child_params=prepare_param_dict(child_params, detach=bool(self.detach_child_params)),
                parent_params=prepare_param_dict(parent_params, detach=bool(self.detach_parent_params)),
                parent_id=pid,
                child_mass=child_mass.to(device=parent_event.device, dtype=parent_event.dtype),
                parent_count=parent_count.to(device=parent_event.device),
                parent_mass_mean=parent_mass_mean.to(device=parent_event.device, dtype=parent_event.dtype),
                branch_id=int(branch_id),
                route_flag=None if route_flag is None else route_flag.to(device=parent_event.device),
            )
            if mode == "residual_mlp":
                residual = self.residual_mlp(torch.cat([parent_e, code], dim=-1))
                if self.mean_preserve:
                    mean = _scatter_weighted_mean(
                        residual,
                        pid,
                        child_mass.to(device=parent_event.device, dtype=parent_event.dtype),
                        num_parents=int(parent_event.shape[0]),
                    )
                    residual = residual - mean.index_select(0, pid)
            elif mode == "low_rank_basis":
                basis = self.basis_mlp(parent_event).reshape(
                    int(parent_event.shape[0]),
                    int(self.rank),
                    int(self.fine_event_dim),
                )
                coeff = self.coeff_mlp(code)
                if self.mean_preserve:
                    mean = _scatter_weighted_mean(
                        coeff,
                        pid,
                        child_mass.to(device=parent_event.device, dtype=parent_event.dtype),
                        num_parents=int(parent_event.shape[0]),
                    )
                    coeff = coeff - mean.index_select(0, pid)
                residual = torch.einsum("nr,nre->ne", coeff, basis.index_select(0, pid))
            else:
                raise ValueError(f"unsupported BigGS child decoder mode={self.mode!r}")
            scaled_residual = residual_scale * residual
            fine = parent_e + scaled_residual
        fine = self.final_norm(fine)
        if not torch.isfinite(fine).all():
            raise RuntimeError("BigGS fine event contains NaN/Inf")
        mean_preserve_error = 0.0
        if n > 0 and bool(self.mean_preserve):
            mean_residual = _scatter_weighted_mean(
                residual,
                pid,
                child_mass.to(device=parent_event.device, dtype=parent_event.dtype),
                num_parents=int(parent_event.shape[0]),
            )
            active = parent_count.to(device=parent_event.device).reshape(-1) > 0
            if int(active.numel()) > 0:
                mean_preserve_error = float(mean_residual[active].detach().norm(dim=-1).max().item())
        aux = {
            "fine_event_norm": float(fine.detach().norm(dim=-1).mean().item()) if fine.numel() else 0.0,
            "child_residual_norm": float(residual.detach().norm(dim=-1).mean().item()) if residual.numel() else 0.0,
            "scaled_child_residual_norm": (
                float(scaled_residual.detach().norm(dim=-1).mean().item()) if scaled_residual.numel() else 0.0
            ),
            "residual_scale": float(residual_scale.detach().abs().reshape(-1).mean().item()),
            "mean_preserve_error": mean_preserve_error,
        }
        return (
            fine,
            _broadcast_optional(parent_support, pid),
            _broadcast_optional(parent_valid, pid),
            _broadcast_optional(parent_obs, pid),
            aux,
        )

    @staticmethod
    def _branch_params(branch: Any) -> Dict[str, torch.Tensor]:
        return {
            "means": branch.means,
            "scales_log": branch.scales_log,
            "quats": branch.quats,
            "opacity_logit": branch.opacity_logit,
            "sh_dc": branch.sh_dc,
            "sh_rest": branch.sh_rest,
        }

    def forward(
        self,
        *,
        parent_event_pack: EventPack,
        local_state: Any,
        measurement: Dict[str, Any],
    ) -> EventPack:
        route = measurement["route"]
        assign_bg: BigGSBranchAssignment = measurement["assign_bg"]
        event_bg, support_bg, valid_bg, obs_bg, aux_bg = self._decode_branch(
            parent_event=parent_event_pack.event_bg,
            parent_support=parent_event_pack.support_bg,
            parent_valid=parent_event_pack.valid_bg,
            parent_obs=parent_event_pack.obs_code_bg,
            parent_id=assign_bg.child_to_parent,
            child_mass=assign_bg.child_mass,
            parent_count=assign_bg.parent_count,
            parent_mass_mean=measurement["parent_mass_mean_bg"],
            child_params=self._branch_params(local_state.bg),
            parent_params=measurement["parent_params_bg"],
            branch_id=0,
            child_basis=assign_bg.child_basis,
        )
        event_distant = support_distant = valid_distant = obs_distant = None
        aux_distant: Dict[str, float] = {}
        assign_distant = measurement.get("assign_distant")
        if (
            local_state.distant is not None
            and assign_distant is not None
            and parent_event_pack.event_distant is not None
        ):
            event_distant, support_distant, valid_distant, obs_distant, aux_distant = self._decode_branch(
                parent_event=parent_event_pack.event_distant,
                parent_support=parent_event_pack.support_distant,
                parent_valid=parent_event_pack.valid_distant,
                parent_obs=parent_event_pack.obs_code_distant,
                parent_id=assign_distant.child_to_parent,
                child_mass=assign_distant.child_mass,
                parent_count=assign_distant.parent_count,
                parent_mass_mean=measurement["parent_mass_mean_distant"],
                child_params=self._branch_params(local_state.distant),
                parent_params=measurement["parent_params_distant"],
                branch_id=1,
                child_basis=assign_distant.child_basis,
            )
        event_rigid = support_rigid = valid_rigid = obs_rigid = None
        aux_rigid: Dict[str, float] = {}
        active: Optional[BigGSRigidActiveAssignment] = measurement.get("assign_rigid_active")
        if local_state.rigid is not None and active is not None and int(active.fine_S.numel()) > 0:
            s = active.fine_S.long().to(device=local_state.rigid.means.device)
            child_params = {
                "means": measurement["route"].means_world_S,
                "scales_log": local_state.rigid.scales_log.index_select(0, s),
                "quats": measurement["route"].quats_world_S,
                "opacity_logit": local_state.rigid.opacity_logit.index_select(0, s),
                "sh_dc": local_state.rigid.sh_dc.index_select(0, s),
                "sh_rest": local_state.rigid.sh_rest.index_select(0, s),
            }
            route_flag = active.parent_inside_mask.to(device=active.child_to_active_parent_S.device)[active.child_to_active_parent_S.long()].float()
            event_rigid, support_rigid, valid_rigid, obs_rigid, aux_rigid = self._decode_branch(
                parent_event=parent_event_pack.event_rigid,
                parent_support=parent_event_pack.support_rigid,
                parent_valid=parent_event_pack.valid_rigid,
                parent_obs=parent_event_pack.obs_code_rigid,
                parent_id=active.child_to_active_parent_S,
                child_mass=active.child_mass_S,
                parent_count=active.active_parent_count,
                parent_mass_mean=measurement["parent_mass_mean_rigid_active"],
                child_params=child_params,
                parent_params=measurement["parent_params_rigid_active"],
                branch_id=2,
                route_flag=route_flag,
                child_basis=active.child_basis_S,
            )
        aux = {
            **dict(getattr(parent_event_pack, "aux", {}) or {}),
            "iforward/biggs/decoder_mode_id": float({"broadcast": 0, "residual_mlp": 1, "low_rank_basis": 2, "whdd_fixed_basis": 3, "whdd_compact_fixed_basis": 4}.get(str(self.mode).lower(), -1)),
            "iforward/biggs/decoder_rank": float(self.rank),
            "iforward/biggs/parent_event_norm_bg": (
                float(parent_event_pack.event_bg.detach().norm(dim=-1).mean().item())
                if parent_event_pack.event_bg is not None and parent_event_pack.event_bg.numel()
                else 0.0
            ),
            "iforward/biggs/child_decoder_residual_scale": aux_bg.get(
                "residual_scale",
                aux_distant.get("residual_scale", aux_rigid.get("residual_scale", 0.0)),
            ),
            "iforward/biggs/fine_event_norm_bg": aux_bg.get("fine_event_norm", 0.0),
            "iforward/biggs/child_residual_norm_bg": aux_bg.get("child_residual_norm", 0.0),
            "iforward/biggs/scaled_child_residual_norm_bg": aux_bg.get("scaled_child_residual_norm", 0.0),
            "iforward/biggs/mean_preserve_error_bg": aux_bg.get("mean_preserve_error", 0.0),
            "iforward/biggs/fine_event_norm_distant": aux_distant.get("fine_event_norm", 0.0),
            "iforward/biggs/child_residual_norm_distant": aux_distant.get("child_residual_norm", 0.0),
            "iforward/biggs/scaled_child_residual_norm_distant": aux_distant.get("scaled_child_residual_norm", 0.0),
            "iforward/biggs/mean_preserve_error_distant": aux_distant.get("mean_preserve_error", 0.0),
            "iforward/biggs/fine_event_norm_rigid": aux_rigid.get("fine_event_norm", 0.0),
            "iforward/biggs/child_residual_norm_rigid": aux_rigid.get("child_residual_norm", 0.0),
            "iforward/biggs/scaled_child_residual_norm_rigid": aux_rigid.get("scaled_child_residual_norm", 0.0),
            "iforward/biggs/mean_preserve_error_rigid": aux_rigid.get("mean_preserve_error", 0.0),
            "iforward/whdd/gamma_bg": float(self._residual_scale_for_branch(branch_id=0, ref=parent_event_pack.event_bg).detach().item())
            if parent_event_pack.event_bg is not None
            else 0.0,
            "iforward/whdd/gamma_distant": float(self._residual_scale_for_branch(branch_id=1, ref=parent_event_pack.event_bg).detach().item())
            if parent_event_pack.event_bg is not None
            else 0.0,
            "iforward/whdd/gamma_rigid": float(self._residual_scale_for_branch(branch_id=2, ref=parent_event_pack.event_bg).detach().item())
            if parent_event_pack.event_bg is not None
            else 0.0,
        }
        return EventPack(
            event_bg=event_bg,
            event_distant=event_distant,
            event_rigid=event_rigid,
            support_bg=support_bg,
            support_distant=support_distant,
            support_rigid=support_rigid,
            valid_bg=valid_bg,
            valid_distant=valid_distant,
            valid_rigid=valid_rigid,
            obs_code_bg=obs_bg,
            obs_code_distant=obs_distant,
            obs_code_rigid=obs_rigid,
            route=route,
            aux=aux,
        )


__all__ = ["BigGSToFineEventDecoder"]
