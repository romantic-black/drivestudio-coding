from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.streetforward.math_utils import _normalize_quat, _quat_to_rotmat
from models.streetforward.stage6_0.event_encoder import EventPack

from .biggs_relational_decoder import GaussianRelationalLiftingDecoder
from .biggs_state import BigGSBranchAssignment, BigGSRigidActiveAssignment
from .observation_feedback import scale_feedback
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


def _scatter_sum(values: torch.Tensor, parent_id: torch.Tensor, *, num_parents: int) -> torch.Tensor:
    out = values.new_zeros((int(num_parents), int(values.shape[-1])))
    if int(values.numel()) > 0:
        out.index_add_(0, parent_id.long(), values)
    return out


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
        decode_chunk_size: Optional[int] = None,
        fused_cuda: bool = False,
        relation_dim: int = 12,
        detach_relation_inputs: bool = True,
        relation_normalization: str = "none",
        relation_rms_floor: float = 0.05,
        relation_clip: float = 0.0,
        rigid_relation_space: str = "world",
        detail_head_init_std: float = 0.0,
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
        self.decode_chunk_size = max(int(decode_chunk_size) if decode_chunk_size is not None else int(whdd_chunk_size), 1)
        self.rigid_relation_space = str(rigid_relation_space).lower()
        self._relation_feedback_enabled = False
        self._relation_feedback_alpha = 0.0
        self._relation_feedback_branches = frozenset({"bg", "distant"})
        self._relation_grad_to_child_geometry = True
        self._relation_grad_to_parent_geometry = True
        self._relation_grad_to_child_code = False
        self._relation_grad_to_parent_event = True
        self._relation_checkpoint = False
        mode_l = str(self.mode).lower()
        if mode_l not in {"whdd_compact_fixed_basis", "gaussian_relational"} and int(self.parent_event_dim) != int(self.fine_event_dim):
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
        self.grld_decoder: Optional[GaussianRelationalLiftingDecoder] = None
        if mode_l == "gaussian_relational":
            self.grld_decoder = GaussianRelationalLiftingDecoder(
                parent_event_dim=int(self.parent_event_dim),
                fine_event_dim=int(self.fine_event_dim),
                relation_dim=int(relation_dim),
                rank=int(rank),
                fused_cuda=bool(fused_cuda),
                detach_relation_inputs=bool(detach_relation_inputs),
                decode_chunk_size=int(self.decode_chunk_size),
                relation_normalization=str(relation_normalization),
                relation_rms_floor=float(relation_rms_floor),
                relation_clip=float(relation_clip),
                rigid_relation_space=str(rigid_relation_space),
                detail_head_init_std=float(detail_head_init_std),
            )

    def set_relation_feedback(
        self,
        *,
        enabled: bool,
        alpha: float,
        branches: Sequence[str],
        grad_to_child_geometry: bool,
        grad_to_parent_geometry: bool,
        grad_to_child_code: bool,
        grad_to_parent_event: bool,
        grad_to_support: bool,
        checkpoint: bool,
    ) -> None:
        """Set the per-visit GRLD continuous feedback policy.

        The values are copied into immutable Python scalars/sets.  Each
        ``decode_branch`` call snapshots them into its checkpoint closure, so a
        later rollout visit cannot change the policy used during recomputation.
        """

        alpha_f = float(alpha)
        if not math.isfinite(alpha_f) or not 0.0 <= alpha_f <= 1.0:
            raise ValueError("relation feedback alpha must be finite and in [0, 1]")
        normalized = tuple(str(branch).lower() for branch in branches)
        if len(set(normalized)) != len(normalized):
            raise ValueError("relation feedback branches must not contain duplicates")
        unsupported = sorted(set(normalized) - {"bg", "distant"})
        if unsupported:
            raise ValueError(
                "relation feedback supports only bg/distant; unsupported branches="
                + ",".join(unsupported)
            )
        if bool(enabled) and not normalized:
            raise ValueError("enabled relation feedback requires at least one bg/distant branch")
        if bool(enabled) and str(self.mode).lower() != "gaussian_relational":
            raise ValueError("relation feedback requires child_decoder.mode=gaussian_relational")
        if bool(grad_to_support):
            raise ValueError("relation feedback grad_to_support=true is unsupported; support must remain detached")

        self._relation_feedback_enabled = bool(enabled)
        self._relation_feedback_alpha = alpha_f
        self._relation_feedback_branches = frozenset(normalized)
        self._relation_grad_to_child_geometry = bool(grad_to_child_geometry)
        self._relation_grad_to_parent_geometry = bool(grad_to_parent_geometry)
        self._relation_grad_to_child_code = bool(grad_to_child_code)
        self._relation_grad_to_parent_event = bool(grad_to_parent_event)
        self._relation_checkpoint = bool(checkpoint)

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
            decode_chunk_size=int(cfg_get(cfg, "decode_chunk_size", cfg_get(cfg, "whdd_chunk_size", 65536))),
            fused_cuda=bool(cfg_get(cfg, "fused_cuda", False)),
            relation_dim=int(cfg_get(cfg, "relation_dim", 12)),
            detach_relation_inputs=bool(cfg_get(cfg, "detach_relation_inputs", cfg_get(cfg, "detach_child_code_inputs", True))),
            relation_normalization=str(cfg_get(cfg, "relation_normalization", "none")),
            relation_rms_floor=float(cfg_get(cfg, "relation_rms_floor", 0.05)),
            relation_clip=float(cfg_get(cfg, "relation_clip", 0.0)),
            rigid_relation_space=str(cfg_get(cfg, "rigid_relation_space", "world")),
            detail_head_init_std=float(cfg_get(cfg, "detail_head_init_std", 0.0)),
        )

    def _residual_scale_for_branch(self, *, branch_id: int, ref: torch.Tensor) -> torch.Tensor:
        scale = self.residual_scale.to(device=ref.device, dtype=ref.dtype)
        if scale.dim() == 0:
            return scale
        idx = max(0, min(int(branch_id), int(scale.numel()) - 1))
        return scale[idx]

    @staticmethod
    def _relation_branch_name(branch_id: int) -> str:
        names = {0: "bg", 1: "distant", 2: "rigid"}
        if int(branch_id) not in names:
            raise ValueError(f"unsupported BigGS relation branch_id={branch_id}")
        return names[int(branch_id)]

    def _relation_feedback_active(self, *, branch_id: int) -> bool:
        return bool(self._relation_feedback_enabled) and self._relation_branch_name(
            int(branch_id)
        ) in self._relation_feedback_branches

    def _prepare_relation_feedback_inputs(
        self,
        *,
        parent_event: torch.Tensor,
        child_params: Dict[str, torch.Tensor],
        parent_params: Dict[str, torch.Tensor],
        child_cache: Any,
        branch_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool, float]:
        active = self._relation_feedback_active(branch_id=int(branch_id))
        if not active:
            return parent_event, child_params, parent_params, False, 0.0
        if int(branch_id) == 2:
            raise RuntimeError("rigid relation feedback is unsupported and must remain detached")
        if child_cache is None:
            raise RuntimeError("relation feedback requires a graph-free BigGS child runtime cache")

        # Snapshot the scalar policy before building the checkpoint closure.
        alpha = float(self._relation_feedback_alpha)

        def routed(value: torch.Tensor, *, enabled: bool) -> torch.Tensor:
            return scale_feedback(value, alpha) if bool(enabled) else value.detach()

        child = {key: value.detach() for key, value in child_params.items()}
        parent = {key: value.detach() for key, value in parent_params.items()}

        child["means"] = routed(
            child_params["means"],
            enabled=bool(self._relation_grad_to_child_geometry),
        )
        live_diag_cov = self._diag_cov_from_scales_quats(
            child_params["scales_log"],
            child_params["quats"],
        )
        cached_diag_cov = child_cache.diag_cov.detach().to(
            device=live_diag_cov.device,
            dtype=live_diag_cov.dtype,
        )
        if tuple(cached_diag_cov.shape) != tuple(live_diag_cov.shape):
            raise ValueError(
                "relation feedback child_cache.diag_cov shape mismatch: "
                f"cache={tuple(cached_diag_cov.shape)} live={tuple(live_diag_cov.shape)}"
            )
        if bool(self._relation_grad_to_child_geometry):
            live_scaled = scale_feedback(live_diag_cov, alpha)
            # Preserve the incremental runtime cache value exactly in forward,
            # while sourcing the backward Jacobian from live scales/quats.
            child["diag_cov"] = cached_diag_cov + (live_scaled - live_scaled.detach())
        else:
            child["diag_cov"] = cached_diag_cov

        for key in ("opacity_logit", "sh_dc", "sh_rest"):
            child[key] = routed(
                child_params[key],
                enabled=bool(self._relation_grad_to_child_code),
            )
            parent[key] = routed(
                parent_params[key],
                enabled=bool(self._relation_grad_to_child_code),
            )
        for key in ("means", "scales_log"):
            parent[key] = routed(
                parent_params[key],
                enabled=bool(self._relation_grad_to_parent_geometry),
            )

        event = parent_event if bool(self._relation_grad_to_parent_event) else parent_event.detach()
        return event, child, parent, bool(self._relation_checkpoint), alpha

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
        child_cache: Optional[Any] = None,
        parent_stats: Optional[Any] = None,
        parent_start: Optional[torch.Tensor] = None,
        child_order: Optional[torch.Tensor] = None,
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
        extra_aux: Dict[str, float] = {}
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
        elif mode == "gaussian_relational":
            if self.grld_decoder is None:
                raise RuntimeError("BigGS gaussian_relational decoder was not initialized")
            if child_cache is None or parent_stats is None or parent_start is None or child_order is None:
                raise RuntimeError("BigGS gaussian_relational decoder requires runtime child_cache/stats and parent assignment order")
            (
                relation_parent_event,
                relation_child_params,
                relation_parent_params,
                relation_checkpoint,
                relation_alpha,
            ) = self._prepare_relation_feedback_inputs(
                parent_event=parent_event,
                child_params=child_params,
                parent_params=parent_params,
                child_cache=child_cache,
                branch_id=int(branch_id),
            )
            relation_feedback_active = self._relation_feedback_active(branch_id=int(branch_id))
            fine, parent_e, residual, extra_aux = self.grld_decoder.decode_branch(
                parent_event=relation_parent_event,
                child_params=relation_child_params,
                parent_params=relation_parent_params,
                child_cache=child_cache,
                parent_stats=parent_stats,
                child_to_parent=pid,
                parent_start=parent_start,
                parent_count=parent_count,
                child_order=child_order,
                branch_id=int(branch_id),
                branch_scale=residual_scale,
                checkpoint_branch=bool(relation_checkpoint),
                detach_relation_params=False if bool(relation_feedback_active) else None,
                detach_support=True if bool(relation_feedback_active) else None,
            )
            extra_aux["feedback_enabled"] = 1.0 if bool(relation_feedback_active) else 0.0
            extra_aux["feedback_alpha"] = float(relation_alpha)
            scaled_residual = residual
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
        compact_checkpoint_aux = bool(
            mode == "gaussian_relational"
            and float(extra_aux.get("checkpoint_compact_aux", 0.0)) > 0.0
        )
        if n > 0 and bool(self.mean_preserve) and not compact_checkpoint_aux:
            mean_residual = _scatter_weighted_mean(
                residual,
                pid,
                child_mass.to(device=parent_event.device, dtype=parent_event.dtype),
                num_parents=int(parent_event.shape[0]),
            )
            active = parent_count.to(device=parent_event.device).reshape(-1) > 0
            if int(active.numel()) > 0:
                mean_preserve_error = float(mean_residual[active].detach().norm(dim=-1).max().item())
        if mode == "gaussian_relational" and "weighted_mean_error" in extra_aux:
            mean_preserve_error = float(extra_aux.get("weighted_mean_error", mean_preserve_error))
        aux = {
            "fine_event_norm": float(fine.detach().norm(dim=-1).mean().item()) if fine.numel() else 0.0,
            "child_residual_norm": float(residual.detach().norm(dim=-1).mean().item()) if residual.numel() else 0.0,
            "scaled_child_residual_norm": (
                float(scaled_residual.detach().norm(dim=-1).mean().item()) if scaled_residual.numel() else 0.0
            ),
            "residual_scale": float(residual_scale.detach().abs().reshape(-1).mean().item()),
            "mean_preserve_error": mean_preserve_error,
        }
        for key, value in extra_aux.items():
            aux[f"grld/{key}"] = float(value)
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

    @staticmethod
    def _diag_cov_from_scales_quats(scales_log: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
        scales = torch.exp(scales_log).clamp_min(1.0e-6)
        rot = _quat_to_rotmat(_normalize_quat(quats))
        return (rot.square() * scales.square().unsqueeze(-2)).sum(dim=-1).clamp_min(1.0e-8)

    def _canonical_rigid_relation_parent_params(
        self,
        *,
        child_params: Dict[str, torch.Tensor],
        base_parent_params: Dict[str, torch.Tensor],
        child_to_parent: torch.Tensor,
        child_mass: torch.Tensor,
        num_parents: int,
    ) -> Dict[str, torch.Tensor]:
        means = child_params["means"]
        device = means.device
        dtype = means.dtype
        pid = child_to_parent.long().to(device=device)
        mass = child_mass.reshape(-1, 1).to(device=device, dtype=dtype).clamp_min(1.0e-8)
        diag_cov = self._diag_cov_from_scales_quats(
            child_params["scales_log"].to(device=device, dtype=dtype),
            child_params["quats"].to(device=device, dtype=dtype),
        )
        denom = _scatter_sum(mass, pid, num_parents=int(num_parents)).clamp_min(1.0e-8)
        parent_means = _scatter_sum(means * mass, pid, num_parents=int(num_parents)) / denom
        parent_second = _scatter_sum((diag_cov + means.square()) * mass, pid, num_parents=int(num_parents)) / denom
        parent_diag_cov = (parent_second - parent_means.square()).clamp_min(1.0e-8)
        parent_quats = means.new_zeros((int(num_parents), 4))
        parent_quats[:, 0] = 1.0
        parent_params = {
            key: value.to(device=device, dtype=dtype)
            for key, value in base_parent_params.items()
        }
        parent_params["means"] = parent_means
        parent_params["scales_log"] = 0.5 * torch.log(parent_diag_cov)
        parent_params["quats"] = parent_quats
        child_params["diag_cov"] = diag_cov
        return parent_params

    def forward(
        self,
        *,
        parent_event_pack: EventPack,
        local_state: Any,
        measurement: Dict[str, Any],
    ) -> EventPack:
        route = measurement["route"]
        runtime = measurement.get("biggs_parent_runtime")
        bg_runtime = getattr(runtime, "bg", None) if runtime is not None else None
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
            child_cache=None if bg_runtime is None else bg_runtime.child_cache,
            parent_stats=None if bg_runtime is None else bg_runtime.stats,
            parent_start=assign_bg.parent_start,
            child_order=assign_bg.child_order,
        )
        event_distant = support_distant = valid_distant = obs_distant = None
        aux_distant: Dict[str, float] = {}
        assign_distant = measurement.get("assign_distant")
        if (
            local_state.distant is not None
            and assign_distant is not None
            and parent_event_pack.event_distant is not None
        ):
            distant_runtime = getattr(runtime, "distant", None) if runtime is not None else None
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
                child_cache=None if distant_runtime is None else distant_runtime.child_cache,
                parent_stats=None if distant_runtime is None else distant_runtime.stats,
                parent_start=assign_distant.parent_start,
                child_order=assign_distant.child_order,
            )
        event_rigid = support_rigid = valid_rigid = obs_rigid = None
        aux_rigid: Dict[str, float] = {}
        active: Optional[BigGSRigidActiveAssignment] = measurement.get("assign_rigid_active")
        if local_state.rigid is not None and active is not None and int(active.fine_S.numel()) > 0:
            rigid_runtime = getattr(runtime, "rigid_active", None) if runtime is not None else None
            s = active.fine_S.long().to(device=local_state.rigid.means.device)
            world_child_params = {
                "means": measurement["route"].means_world_S,
                "scales_log": local_state.rigid.scales_log.index_select(0, s),
                "quats": measurement["route"].quats_world_S,
                "opacity_logit": local_state.rigid.opacity_logit.index_select(0, s),
                "sh_dc": local_state.rigid.sh_dc.index_select(0, s),
                "sh_rest": local_state.rigid.sh_rest.index_select(0, s),
            }
            parent_params_rigid = measurement["parent_params_rigid_active"]
            child_params = world_child_params
            if str(self.mode).lower() == "gaussian_relational" and self.rigid_relation_space == "canonical":
                child_params = {
                    "means": local_state.rigid.means.index_select(0, s),
                    "scales_log": local_state.rigid.scales_log.index_select(0, s),
                    "quats": local_state.rigid.quats.index_select(0, s),
                    "opacity_logit": local_state.rigid.opacity_logit.index_select(0, s),
                    "sh_dc": local_state.rigid.sh_dc.index_select(0, s),
                    "sh_rest": local_state.rigid.sh_rest.index_select(0, s),
                }
                child_mass_for_relation = (
                    active.child_mass_S if rigid_runtime is None else rigid_runtime.child_cache.mass
                )
                parent_params_rigid = self._canonical_rigid_relation_parent_params(
                    child_params=child_params,
                    base_parent_params=parent_params_rigid,
                    child_to_parent=active.child_to_active_parent_S,
                    child_mass=child_mass_for_relation,
                    num_parents=int(active.active_parent_count.numel()),
                )
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
                parent_params=parent_params_rigid,
                branch_id=2,
                route_flag=route_flag,
                child_basis=active.child_basis_S,
                child_cache=None if rigid_runtime is None else rigid_runtime.child_cache,
                parent_stats=None if rigid_runtime is None else rigid_runtime.stats,
                parent_start=active.active_parent_start,
                child_order=active.active_child_order_S,
            )
        grld_branch_aux = {"bg": aux_bg, "distant": aux_distant, "rigid": aux_rigid}

        def grld_max(name: str) -> float:
            values = [float(aux.get(f"grld/{name}", 0.0)) for aux in grld_branch_aux.values()]
            return max(values) if values else 0.0

        def grld_sum(name: str) -> float:
            return float(sum(float(aux.get(f"grld/{name}", 0.0)) for aux in grld_branch_aux.values()))

        aux = {
            **dict(getattr(parent_event_pack, "aux", {}) or {}),
            "iforward/biggs/decoder_mode_id": float({"broadcast": 0, "residual_mlp": 1, "low_rank_basis": 2, "whdd_fixed_basis": 3, "whdd_compact_fixed_basis": 4, "gaussian_relational": 5}.get(str(self.mode).lower(), -1)),
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
            "iforward/grld/weighted_mean_error": grld_max("weighted_mean_error"),
            "iforward/grld/relation_centering_error": grld_max("relation_centering_error"),
            "iforward/grld/dynamic_mass_nan_ratio": grld_max("dynamic_mass_nan_ratio"),
            "iforward/grld/relation_xyz_norm": grld_max("relation_xyz_norm"),
            "iforward/grld/relation_cov_norm": grld_max("relation_cov_norm"),
            "iforward/grld/relation_cov_norm_before_norm": grld_max("relation_cov_norm_before_norm"),
            "iforward/grld/relation_cov_norm_after_norm": grld_max("relation_cov_norm_after_norm"),
            "iforward/grld/relation_mass_norm": grld_max("relation_mass_norm"),
            "iforward/grld/relation_opacity_norm": grld_max("relation_opacity_norm"),
            "iforward/grld/relation_sh_norm": grld_max("relation_sh_norm"),
            "iforward/grld/relation_channel_rms_min": grld_max("relation_channel_rms_min"),
            "iforward/grld/relation_channel_rms_max": grld_max("relation_channel_rms_max"),
            "iforward/grld/relation_ms": grld_sum("relation_ms"),
            "iforward/grld/decode_ms": grld_sum("decode_ms"),
            "iforward/grld/rigid_relation_world_mode": grld_max("rigid_relation_world_mode"),
            "iforward/grld/rigid_relation_canonical_mode": grld_max("rigid_relation_canonical_mode"),
            "iforward/grld/feedback_enabled": grld_max("feedback_enabled"),
            "iforward/grld/feedback_alpha": grld_max("feedback_alpha"),
            "iforward/grld/checkpoint_enabled": grld_max("checkpoint_enabled"),
            "iforward/grld/lambda_bg": float(self._residual_scale_for_branch(branch_id=0, ref=parent_event_pack.event_bg).detach().item())
            if parent_event_pack.event_bg is not None
            else 0.0,
            "iforward/grld/lambda_distant": float(self._residual_scale_for_branch(branch_id=1, ref=parent_event_pack.event_bg).detach().item())
            if parent_event_pack.event_bg is not None
            else 0.0,
            "iforward/grld/lambda_rigid": float(self._residual_scale_for_branch(branch_id=2, ref=parent_event_pack.event_bg).detach().item())
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
