from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.stage6_0.local_gs_state import LocalGSState

from .types import (
    LongVSMReadPack,
    coerce_branch_support,
    coerce_branch_valid,
    coerce_view_code,
    rigid_stable_mask_from_meta,
)


DISTANT_MODE_FROZEN = "frozen_render_only"
DISTANT_MODE_APPEARANCE_SCALE = "appearance_scale_only"
_DISTANT_MODES = {DISTANT_MODE_FROZEN, DISTANT_MODE_APPEARANCE_SCALE}


def _check_distant_mode(mode: str) -> str:
    mode_s = str(mode)
    if mode_s not in _DISTANT_MODES:
        raise ValueError(
            "6_0_phase_b supports distant.mode in "
            f"{sorted(_DISTANT_MODES)}, got {mode_s!r}."
        )
    return mode_s


@dataclass
class LongVSMState:
    bg_h: torch.Tensor
    bg_seen: torch.Tensor
    distant_h: Optional[torch.Tensor] = None
    distant_seen: Optional[torch.Tensor] = None
    rigid_h: Optional[torch.Tensor] = None
    rigid_seen: Optional[torch.Tensor] = None
    rigid_sparse_ids: Optional[torch.Tensor] = None
    rigid_sparse_h: Optional[torch.Tensor] = None
    rigid_sparse_seen: Optional[torch.Tensor] = None
    episode_id: int = -1

    def detach(self) -> "LongVSMState":
        def d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach()

        return replace(
            self,
            bg_h=self.bg_h.detach(),
            bg_seen=self.bg_seen.detach(),
            distant_h=d(self.distant_h),
            distant_seen=d(self.distant_seen),
            rigid_h=d(self.rigid_h),
            rigid_seen=d(self.rigid_seen),
            rigid_sparse_ids=d(self.rigid_sparse_ids),
            rigid_sparse_h=d(self.rigid_sparse_h),
            rigid_sparse_seen=d(self.rigid_sparse_seen),
        )

    def detach_to_cache_optional(self) -> "LongVSMState":
        return self.detach()

    def seen_pack(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "bg": self.bg_seen,
            "rigid": self.rigid_seen,
            "distant": self.distant_seen,
        }


class StreamingSelectiveSSMBranch(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        view_dim: int,
        mem_dim: int = 64,
        input_dim: int = 96,
        support_fallback_when_no_valid: bool = False,
        support_fallback_min: float = 0.0,
        support_fallback_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.view_dim = int(view_dim)
        self.mem_dim = int(mem_dim)
        self.support_fallback_when_no_valid = bool(support_fallback_when_no_valid)
        self.support_fallback_min = float(support_fallback_min)
        self.support_fallback_scale = float(support_fallback_scale)
        self.branch_embed = nn.Embedding(3, 8)
        self.time_embed = nn.Embedding(64, 8)
        self.repeat_embed = nn.Embedding(16, 8)
        packed_dim = int(event_dim) + int(view_dim) + 2 + 4 + 8 + 8 + 8
        self.input_proj = nn.Linear(packed_dim, int(input_dim))
        self.dt_proj = nn.Linear(int(input_dim), int(mem_dim))
        self.B_proj = nn.Linear(int(input_dim), int(mem_dim))
        self.gate_proj = nn.Linear(int(input_dim), int(mem_dim))
        self.A_log = nn.Parameter(torch.zeros(int(mem_dim)))
        self.norm = nn.LayerNorm(int(mem_dim))

    def _empty_aux(self, ref: torch.Tensor) -> Dict[str, float]:
        _ = ref
        return {
            "write_gate_mean": 0.0,
            "dt_mean": 0.0,
            "seen_rows": 0.0,
            "h_norm": 0.0,
        }

    def write(
        self,
        *,
        h: torch.Tensor,
        seen: torch.Tensor,
        event: torch.Tensor,
        view_code: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
        support: Optional[torch.Tensor],
        step_idx: int,
        repeat_idx: int,
        branch_id: int,
        visit_time_code: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        if event.dim() != 2:
            raise ValueError(f"event must be [N,C], got {tuple(event.shape)}")
        n = int(event.shape[0])
        if int(event.shape[1]) != int(self.event_dim):
            raise ValueError(f"event dim {int(event.shape[1])} != expected {int(self.event_dim)}")
        if int(h.shape[0]) != n or int(h.shape[1]) != int(self.mem_dim):
            raise ValueError(f"h shape {tuple(h.shape)} does not match event rows={n} mem_dim={self.mem_dim}")
        if int(seen.shape[0]) != n:
            raise ValueError(f"seen rows {int(seen.shape[0])} != event rows {n}")
        if n == 0:
            return h, seen, h.new_zeros((0, int(self.mem_dim))), self._empty_aux(h)

        event_in = event.to(dtype=h.dtype)
        view = coerce_view_code(view_code, n=n, ref=event_in, view_dim=self.view_dim)
        valid_f = coerce_branch_valid(valid, n=n, ref=event_in)
        support_f = coerce_branch_support(support, n=n, ref=event_in)
        hard_valid_ratio = float((valid_f.detach() > 0.0).float().mean().item())
        support_positive_ratio = float((support_f.detach() > 0.0).float().mean().item())
        support_fallback_used = 0.0
        if bool(self.support_fallback_when_no_valid) and support is not None and float(valid_f.detach().sum().item()) == 0.0:
            support_gate = support_f.clamp_min(0.0) * float(self.support_fallback_scale)
            support_gate = support_gate.clamp(max=1.0)
            if float(self.support_fallback_min) > 0.0:
                support_gate = support_gate * (support_f > float(self.support_fallback_min)).to(dtype=support_gate.dtype)
            if float(support_gate.detach().sum().item()) > 0.0:
                valid_f = support_gate
                support_fallback_used = 1.0
        if visit_time_code is None:
            time_code = event_in.new_zeros((n, 4))
        else:
            time_code = visit_time_code.to(device=event_in.device, dtype=event_in.dtype)
            if time_code.dim() == 1:
                time_code = time_code.view(1, -1)
            if int(time_code.shape[-1]) != 4:
                raise ValueError(f"visit_time_code must have 4 channels, got {tuple(time_code.shape)}")
            if int(time_code.shape[0]) == 1:
                time_code = time_code.expand(n, 4)
            elif int(time_code.shape[0]) != n:
                raise ValueError(f"visit_time_code rows {int(time_code.shape[0])} != event rows {n}")
        branch = self.branch_embed(
            torch.full((n,), int(branch_id), device=event_in.device, dtype=torch.long)
        ).to(dtype=event_in.dtype)
        time = self.time_embed(
            torch.full((n,), int(step_idx) % 64, device=event_in.device, dtype=torch.long)
        ).to(dtype=event_in.dtype)
        repeat = self.repeat_embed(
            torch.full((n,), int(repeat_idx) % 16, device=event_in.device, dtype=torch.long)
        ).to(dtype=event_in.dtype)
        x = torch.cat([event_in, view, torch.log1p(support_f), valid_f, time_code, branch, time, repeat], dim=-1)
        u = self.input_proj(x)
        u_dtype = u.dtype
        dt_raw = self.dt_proj(u)
        dt = F.softplus(dt_raw.float()).to(dtype=u_dtype) + u.new_tensor(1.0e-4)
        A = -F.softplus(self.A_log.float()).to(device=u.device, dtype=u_dtype).view(1, -1)
        decay_arg = (dt.float() * A.float()).clamp(min=-20.0, max=0.0)
        decay = torch.exp(decay_arg).to(dtype=u_dtype)
        candidate = torch.tanh(self.B_proj(u))
        gate_raw = self.gate_proj(u)
        write_gate = torch.sigmoid(gate_raw.float()).to(dtype=u_dtype) * valid_f
        h_new = h * (1.0 - write_gate) + write_gate * (decay * h + (1.0 - decay) * candidate)
        seen_new = seen + valid_f
        read = self.norm(h_new).to(dtype=h_new.dtype)
        aux = {
            "write_gate_mean": float(write_gate.detach().float().mean().item()),
            "dt_mean": float(dt.detach().float().mean().item()),
            "seen_rows": float((seen_new.detach() > 0).sum().item()),
            "h_norm": float(h_new.detach().float().norm(dim=-1).mean().item()),
            "hard_valid_ratio": hard_valid_ratio,
            "support_mean": float(support_f.detach().float().mean().item()),
            "support_max": float(support_f.detach().float().max().item()),
            "support_positive_ratio": support_positive_ratio,
            "support_fallback_used": float(support_fallback_used),
        }
        return h_new, seen_new, read, aux


def _prefix(aux: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in aux.items()}


def _index_optional_rows(value: Optional[torch.Tensor], idx: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.index_select(0, idx.to(device=value.device, dtype=torch.long))


class LongStreamingVSM(nn.Module):
    def __init__(
        self,
        *,
        event_dim: int,
        view_dim: int = 2,
        bg_mem_dim: int = 64,
        rigid_mem_dim: int = 64,
        distant_mem_dim: int = 32,
        input_dim: int = 96,
        dtype: str = "bf16",
        distant_mode: str = "frozen_render_only",
        support_fallback_when_no_valid: bool = False,
        support_fallback_min: float = 0.0,
        support_fallback_scale: float = 1.0,
        bg_active_sparse: bool = True,
    ) -> None:
        super().__init__()
        self.event_dim = int(event_dim)
        self.view_dim = int(view_dim)
        self.bg_mem_dim = int(bg_mem_dim)
        self.rigid_mem_dim = int(rigid_mem_dim)
        self.distant_mem_dim = int(distant_mem_dim)
        self.state_dtype_name = str(dtype)
        self.distant_mode = _check_distant_mode(str(distant_mode))
        self.support_fallback_when_no_valid = bool(support_fallback_when_no_valid)
        self.support_fallback_min = float(support_fallback_min)
        self.support_fallback_scale = float(support_fallback_scale)
        self.bg_active_sparse = bool(bg_active_sparse)
        self.bg_ssm = StreamingSelectiveSSMBranch(
            event_dim=int(event_dim),
            view_dim=int(view_dim),
            mem_dim=int(bg_mem_dim),
            input_dim=int(input_dim),
            support_fallback_when_no_valid=bool(self.support_fallback_when_no_valid),
            support_fallback_min=float(self.support_fallback_min),
            support_fallback_scale=float(self.support_fallback_scale),
        )
        self.distant_ssm = StreamingSelectiveSSMBranch(
            event_dim=int(event_dim),
            view_dim=int(view_dim),
            mem_dim=int(distant_mem_dim),
            input_dim=int(input_dim),
            support_fallback_when_no_valid=bool(self.support_fallback_when_no_valid),
            support_fallback_min=float(self.support_fallback_min),
            support_fallback_scale=float(self.support_fallback_scale),
        )
        self.rigid_ssm = StreamingSelectiveSSMBranch(
            event_dim=int(event_dim),
            view_dim=int(view_dim),
            mem_dim=int(rigid_mem_dim),
            input_dim=int(input_dim),
            support_fallback_when_no_valid=bool(self.support_fallback_when_no_valid),
            support_fallback_min=float(self.support_fallback_min),
            support_fallback_scale=float(self.support_fallback_scale),
        )

    def _select_active_bg(
        self,
        *,
        event_bg: torch.Tensor,
        valid_bg: Optional[torch.Tensor],
        support_bg: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        n = int(event_bg.shape[0])
        valid_f = coerce_branch_valid(valid_bg, n=n, ref=event_bg)
        support_f = coerce_branch_support(support_bg, n=n, ref=event_bg)
        active = valid_f.reshape(-1) > 0.0
        fallback_used = 0.0
        if (
            bool(self.support_fallback_when_no_valid)
            and support_bg is not None
            and float(valid_f.detach().sum().item()) == 0.0
        ):
            support_gate = support_f.clamp_min(0.0) * float(self.support_fallback_scale)
            support_gate = support_gate.clamp(max=1.0)
            if float(self.support_fallback_min) > 0.0:
                support_gate = support_gate * (support_f > float(self.support_fallback_min)).to(dtype=support_gate.dtype)
            active = support_gate.reshape(-1) > 0.0
            fallback_used = 1.0 if float(active.detach().sum().item()) > 0.0 else 0.0
        idx = torch.nonzero(active, as_tuple=False).squeeze(1).to(dtype=torch.long)
        aux = {
            "active_rows": float(int(idx.numel())),
            "total_rows": float(n),
            "active_ratio": float(int(idx.numel()) / max(n, 1)),
            "hard_valid_ratio": float((valid_f.detach() > 0.0).float().mean().item()) if n else 0.0,
            "support_mean": float(support_f.detach().float().mean().item()) if n else 0.0,
            "support_max": float(support_f.detach().float().max().item()) if n else 0.0,
            "support_positive_ratio": float((support_f.detach() > 0.0).float().mean().item()) if n else 0.0,
            "support_fallback_used": float(fallback_used),
        }
        return idx, aux

    @staticmethod
    def _select_visit_time_code(
        visit_time_code: Optional[torch.Tensor],
        *,
        idx: torch.Tensor,
        n_total: int,
    ) -> Optional[torch.Tensor]:
        if visit_time_code is None:
            return None
        if visit_time_code.dim() >= 2 and int(visit_time_code.shape[0]) == int(n_total):
            return visit_time_code.index_select(0, idx.to(device=visit_time_code.device, dtype=torch.long))
        return visit_time_code

    def _state_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if self.state_dtype_name.lower() in {"bf16", "bfloat16"} and ref.is_cuda:
            return torch.bfloat16
        if self.state_dtype_name.lower() in {"fp16", "float16"}:
            return torch.float16
        return ref.dtype

    def init_state(
        self,
        *,
        base_state: LocalGSState,
        dtype: Optional[torch.dtype] = None,
        rigid_meta: Optional[Dict[str, Any]] = None,
        distant_mode: Optional[str] = None,
        episode_id: int = -1,
    ) -> LongVSMState:
        mode = _check_distant_mode(str(distant_mode or self.distant_mode))
        ref = base_state.bg.means
        state_dtype = dtype or self._state_dtype(ref)
        n_bg = int(base_state.bg.means.shape[0])
        n_distant = int(base_state.distant.means.shape[0]) if base_state.distant is not None else 0
        n_rigid = int(base_state.rigid.means.shape[0]) if base_state.rigid is not None else 0
        _ = rigid_meta
        return LongVSMState(
            bg_h=torch.zeros((n_bg, int(self.bg_mem_dim)), device=ref.device, dtype=state_dtype),
            bg_seen=torch.zeros((n_bg, 1), device=ref.device, dtype=state_dtype),
            distant_h=(
                torch.zeros((n_distant, int(self.distant_mem_dim)), device=ref.device, dtype=state_dtype)
                if mode == DISTANT_MODE_APPEARANCE_SCALE and n_distant > 0
                else None
            ),
            distant_seen=(
                torch.zeros((n_distant, 1), device=ref.device, dtype=state_dtype)
                if mode == DISTANT_MODE_APPEARANCE_SCALE and n_distant > 0
                else None
            ),
            rigid_h=(
                torch.zeros((n_rigid, int(self.rigid_mem_dim)), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            rigid_seen=(
                torch.zeros((n_rigid, 1), device=ref.device, dtype=state_dtype)
                if n_rigid > 0
                else None
            ),
            episode_id=int(episode_id),
        )

    def write_read(
        self,
        *,
        state: LongVSMState,
        event: Any,
        step_idx: int,
        frame_idx: int,
        repeat_idx: int,
        rigid_meta: Optional[Dict[str, Any]] = None,
        distant_mode: Optional[str] = None,
        visit_time_code: Optional[torch.Tensor] = None,
        compute_dtype: Optional[torch.dtype] = None,
        commit_memory: bool = True,
    ) -> tuple[LongVSMState, LongVSMReadPack, Dict[str, float]]:
        mode = _check_distant_mode(str(distant_mode or self.distant_mode))
        event_bg = event.event_bg
        if int(event_bg.shape[0]) != int(state.bg_h.shape[0]):
            raise ValueError(
                "event_bg rows must match bg memory rows: "
                f"{int(event_bg.shape[0])} vs {int(state.bg_h.shape[0])}."
            )
        bg_compute_dtype = compute_dtype or event_bg.dtype
        bg_indices: Optional[torch.Tensor] = None
        bg_seen_for_read: torch.Tensor
        if bool(self.bg_active_sparse):
            bg_indices, bg_signal_aux = self._select_active_bg(
                event_bg=event_bg,
                valid_bg=getattr(event, "valid_bg", None),
                support_bg=getattr(event, "support_bg", None),
            )
            bg_indices_state = bg_indices.to(device=state.bg_h.device, dtype=torch.long)
            bg_h_prev = state.bg_h.index_select(0, bg_indices_state).to(device=event_bg.device, dtype=bg_compute_dtype)
            bg_seen_prev = state.bg_seen.index_select(0, bg_indices_state).to(
                device=event_bg.device,
                dtype=bg_compute_dtype,
            )
            bg_h_active, bg_seen_active, read_bg, bg_aux = self.bg_ssm.write(
                h=bg_h_prev,
                seen=bg_seen_prev,
                event=event_bg.index_select(0, bg_indices.to(device=event_bg.device, dtype=torch.long)),
                view_code=_index_optional_rows(getattr(event, "view_code_bg", None), bg_indices),
                valid=_index_optional_rows(getattr(event, "valid_bg", None), bg_indices),
                support=_index_optional_rows(getattr(event, "support_bg", None), bg_indices),
                step_idx=int(step_idx),
                repeat_idx=int(repeat_idx),
                branch_id=0,
                visit_time_code=self._select_visit_time_code(
                    visit_time_code,
                    idx=bg_indices,
                    n_total=int(event_bg.shape[0]),
                ),
            )
            bg_h = state.bg_h.index_copy(
                0,
                bg_indices_state,
                bg_h_active.to(device=state.bg_h.device, dtype=state.bg_h.dtype),
            )
            bg_seen = state.bg_seen.index_copy(
                0,
                bg_indices_state,
                bg_seen_active.to(device=state.bg_seen.device, dtype=state.bg_seen.dtype),
            )
            bg_seen_for_read = bg_seen_active
            bg_aux.update(bg_signal_aux)
            bg_aux["seen_rows"] = float((bg_seen.detach() > 0).sum().item())
        else:
            bg_h, bg_seen, read_bg, bg_aux = self.bg_ssm.write(
                h=state.bg_h.to(device=event_bg.device, dtype=bg_compute_dtype),
                seen=state.bg_seen.to(device=event_bg.device, dtype=bg_compute_dtype),
                event=event_bg,
                view_code=getattr(event, "view_code_bg", None),
                valid=getattr(event, "valid_bg", None),
                support=getattr(event, "support_bg", None),
                step_idx=int(step_idx),
                repeat_idx=int(repeat_idx),
                branch_id=0,
                visit_time_code=visit_time_code,
            )
            bg_seen_for_read = bg_seen
        out_state = replace(
            state,
            bg_h=bg_h.to(device=state.bg_h.device, dtype=state.bg_h.dtype),
            bg_seen=bg_seen.to(device=state.bg_seen.device, dtype=state.bg_seen.dtype),
        )
        aux: Dict[str, float] = {
            **_prefix(bg_aux, "vsm_bg"),
            "vsm_bg_seen_ratio": float((bg_seen.detach() > 0).float().mean().item()) if bg_seen.numel() else 0.0,
            "vsm_step_frame_idx": float(int(frame_idx)),
        }
        read_rigid = None
        rigid_indices = None
        rigid_seen_active = None
        stable_active = None
        event_rigid = getattr(event, "event_rigid", None)
        route = getattr(event, "route", None)
        S = getattr(route, "S", None) if route is not None else None
        if event_rigid is not None and S is not None and int(event_rigid.shape[0]) > 0:
            if out_state.rigid_h is None or out_state.rigid_seen is None:
                raise ValueError("rigid events are present but LongVSMState has no rigid memory.")
            rigid_compute_dtype = compute_dtype or event_rigid.dtype
            rigid_indices = S.to(device=event_rigid.device, dtype=torch.long).reshape(-1)
            if int(rigid_indices.numel()) != int(event_rigid.shape[0]):
                raise ValueError("event_rigid rows must match route.S.")
            stable_full = rigid_stable_mask_from_meta(
                rigid_meta,
                num_rows=int(out_state.rigid_h.shape[0]),
                device=event_rigid.device,
            )
            stable_active = stable_full.index_select(0, rigid_indices)
            read_rigid = torch.zeros(
                (int(event_rigid.shape[0]), int(self.rigid_mem_dim)),
                device=event_rigid.device,
                dtype=rigid_compute_dtype,
            )
            rigid_seen_active = torch.zeros(
                (int(event_rigid.shape[0]), 1),
                device=event_rigid.device,
                dtype=rigid_compute_dtype,
            )
            if bool(stable_active.any().item()):
                rows = torch.nonzero(stable_active, as_tuple=False).squeeze(1)
                idx_stable = rigid_indices.index_select(0, rows).to(device=out_state.rigid_h.device)
                h_old = out_state.rigid_h.index_select(0, idx_stable).to(
                    device=event_rigid.device, dtype=rigid_compute_dtype
                )
                seen_old = out_state.rigid_seen.index_select(0, idx_stable).to(
                    device=event_rigid.device, dtype=rigid_compute_dtype
                )
                h_new, seen_new, read_new, rigid_aux = self.rigid_ssm.write(
                    h=h_old,
                    seen=seen_old,
                    event=event_rigid.index_select(0, rows),
                    view_code=(
                        getattr(event, "view_code_rigid", None).index_select(0, rows)
                        if getattr(event, "view_code_rigid", None) is not None
                        else None
                    ),
                    valid=(
                        getattr(event, "valid_rigid", None).index_select(0, rows)
                        if getattr(event, "valid_rigid", None) is not None
                        else None
                    ),
                    support=(
                        getattr(event, "support_rigid", None).index_select(0, rows)
                        if getattr(event, "support_rigid", None) is not None
                        else None
                    ),
                    step_idx=int(step_idx),
                    repeat_idx=int(repeat_idx),
                    branch_id=2,
                    visit_time_code=visit_time_code,
                )
                rigid_h = out_state.rigid_h.clone()
                rigid_seen = out_state.rigid_seen.clone()
                rigid_h[idx_stable] = h_new.to(device=rigid_h.device, dtype=rigid_h.dtype)
                rigid_seen[idx_stable] = seen_new.to(device=rigid_seen.device, dtype=rigid_seen.dtype)
                out_state = replace(out_state, rigid_h=rigid_h, rigid_seen=rigid_seen)
                read_rigid[rows] = read_new.to(dtype=read_rigid.dtype)
                rigid_seen_active[rows] = seen_new.to(dtype=rigid_seen_active.dtype)
                aux.update(_prefix(rigid_aux, "vsm_rigid"))
            else:
                aux.update(
                    {
                        "vsm_rigid_write_gate_mean": 0.0,
                        "vsm_rigid_dt_mean": 0.0,
                        "vsm_rigid_seen_rows": 0.0,
                        "vsm_rigid_h_norm": 0.0,
                    }
                )
            if bool((~stable_active).any().item()):
                rows_unstable = torch.nonzero(~stable_active, as_tuple=False).squeeze(1)
                h0 = torch.zeros(
                    (int(rows_unstable.numel()), int(self.rigid_mem_dim)),
                    device=event_rigid.device,
                    dtype=rigid_compute_dtype,
                )
                seen0 = torch.zeros(
                    (int(rows_unstable.numel()), 1),
                    device=event_rigid.device,
                    dtype=rigid_compute_dtype,
                )
                _, seen_tmp, read_tmp, _ = self.rigid_ssm.write(
                    h=h0,
                    seen=seen0,
                    event=event_rigid.index_select(0, rows_unstable),
                    view_code=(
                        getattr(event, "view_code_rigid", None).index_select(0, rows_unstable)
                        if getattr(event, "view_code_rigid", None) is not None
                        else None
                    ),
                    valid=(
                        getattr(event, "valid_rigid", None).index_select(0, rows_unstable)
                        if getattr(event, "valid_rigid", None) is not None
                        else None
                    ),
                    support=(
                        getattr(event, "support_rigid", None).index_select(0, rows_unstable)
                        if getattr(event, "support_rigid", None) is not None
                        else None
                    ),
                    step_idx=int(step_idx),
                    repeat_idx=int(repeat_idx),
                    branch_id=2,
                    visit_time_code=visit_time_code,
                )
                read_rigid[rows_unstable] = read_tmp.to(dtype=read_rigid.dtype)
                rigid_seen_active[rows_unstable] = seen_tmp.to(dtype=rigid_seen_active.dtype)
            aux.update(
                {
                    "vsm_rigid_active_rows": float(int(rigid_indices.numel())),
                    "vsm_rigid_stable_rows": float(int(stable_active.sum().item())),
                    "vsm_rigid_unstable_rows": float(int((~stable_active).sum().item())),
                    "vsm_rigid_seen_ratio": (
                        float((out_state.rigid_seen.detach() > 0).float().mean().item())
                        if out_state.rigid_seen is not None and out_state.rigid_seen.numel()
                        else 0.0
                    ),
                }
            )
        else:
            aux.update(
                {
                    "vsm_rigid_active_rows": 0.0,
                    "vsm_rigid_stable_rows": 0.0,
                    "vsm_rigid_unstable_rows": 0.0,
                    "vsm_rigid_seen_ratio": 0.0,
                }
            )

        read_distant = None
        distant_indices = None
        distant_seen_active = None
        event_distant = getattr(event, "event_distant", None)
        valid_distant = getattr(event, "valid_distant", None)
        support_distant = getattr(event, "support_distant", None)
        aux["vsm_distant_mode_frozen_render_only"] = 1.0 if mode == DISTANT_MODE_FROZEN else 0.0
        aux["vsm_distant_mode_appearance_scale_only"] = 1.0 if mode == DISTANT_MODE_APPEARANCE_SCALE else 0.0
        aux["vsm_distant_event_rows"] = float(int(event_distant.shape[0])) if event_distant is not None else 0.0
        aux["vsm_distant_event_norm"] = (
            float(event_distant.detach().float().norm(dim=-1).mean().item())
            if event_distant is not None and event_distant.numel()
            else 0.0
        )
        aux["vsm_distant_valid_ratio"] = (
            float(valid_distant.detach().float().mean().item()) if valid_distant is not None and valid_distant.numel() else 0.0
        )
        aux["vsm_distant_support_mean_all"] = (
            float(support_distant.detach().float().mean().item())
            if support_distant is not None and support_distant.numel()
            else 0.0
        )
        aux["vsm_distant_active_rows"] = 0.0
        aux["vsm_distant_seen_ratio"] = (
            float((out_state.distant_seen.detach() > 0).float().mean().item())
            if out_state.distant_seen is not None and out_state.distant_seen.numel()
            else 0.0
        )
        if mode == DISTANT_MODE_APPEARANCE_SCALE and event_distant is not None and int(event_distant.shape[0]) > 0:
            if out_state.distant_h is None or out_state.distant_seen is None:
                raise ValueError("distant events are present but LongVSMState has no distant memory.")
            if int(event_distant.shape[0]) != int(out_state.distant_h.shape[0]):
                raise ValueError(
                    "event_distant rows must match distant memory rows: "
                    f"{int(event_distant.shape[0])} vs {int(out_state.distant_h.shape[0])}."
                )
            valid_f = coerce_branch_valid(valid_distant, n=int(event_distant.shape[0]), ref=event_distant)
            support_f = coerce_branch_support(support_distant, n=int(event_distant.shape[0]), ref=event_distant)
            active_mask = valid_f.reshape(-1) > 0.0
            if not bool(active_mask.any().item()) and bool(self.support_fallback_when_no_valid) and support_distant is not None:
                min_support = float(self.support_fallback_min)
                active_mask = support_f.reshape(-1) > min_support
            distant_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            aux["vsm_distant_active_rows"] = float(int(distant_indices.numel()))
            if int(distant_indices.numel()) > 0:
                distant_compute_dtype = compute_dtype or event_distant.dtype
                idx_state = distant_indices.to(device=out_state.distant_h.device, dtype=torch.long)
                h_old = out_state.distant_h.index_select(0, idx_state).to(
                    device=event_distant.device,
                    dtype=distant_compute_dtype,
                )
                seen_old = out_state.distant_seen.index_select(0, idx_state).to(
                    device=event_distant.device,
                    dtype=distant_compute_dtype,
                )
                view_distant = getattr(event, "view_code_distant", None)
                if view_distant is None:
                    view_distant = getattr(event, "obs_code_distant", None)
                h_new, seen_new, read_new, distant_aux = self.distant_ssm.write(
                    h=h_old,
                    seen=seen_old,
                    event=event_distant.index_select(0, distant_indices),
                    view_code=(
                        view_distant.index_select(0, distant_indices)
                        if view_distant is not None
                        else None
                    ),
                    valid=valid_f.index_select(0, distant_indices),
                    support=support_f.index_select(0, distant_indices),
                    step_idx=int(step_idx),
                    repeat_idx=int(repeat_idx),
                    branch_id=1,
                    visit_time_code=visit_time_code,
                )
                distant_h = out_state.distant_h.clone()
                distant_seen = out_state.distant_seen.clone()
                distant_h[idx_state] = h_new.to(device=distant_h.device, dtype=distant_h.dtype)
                distant_seen[idx_state] = seen_new.to(device=distant_seen.device, dtype=distant_seen.dtype)
                out_state = replace(out_state, distant_h=distant_h, distant_seen=distant_seen)
                read_distant = read_new
                distant_seen_active = seen_new
                aux.update(_prefix(distant_aux, "vsm_distant"))
                aux["vsm_distant_seen_ratio"] = (
                    float((out_state.distant_seen.detach() > 0).float().mean().item())
                    if out_state.distant_seen is not None and out_state.distant_seen.numel()
                    else 0.0
                )
            else:
                aux.update(
                    {
                        "vsm_distant_write_gate_mean": 0.0,
                        "vsm_distant_dt_mean": 0.0,
                        "vsm_distant_seen_rows": 0.0,
                        "vsm_distant_h_norm": 0.0,
                        "vsm_distant_hard_valid_ratio": float((valid_f.detach() > 0).float().mean().item())
                        if valid_f.numel()
                        else 0.0,
                        "vsm_distant_support_mean": float(support_f.detach().float().mean().item())
                        if support_f.numel()
                        else 0.0,
                        "vsm_distant_support_max": float(support_f.detach().float().max().item())
                        if support_f.numel()
                        else 0.0,
                        "vsm_distant_support_positive_ratio": float((support_f.detach() > 0).float().mean().item())
                        if support_f.numel()
                        else 0.0,
                        "vsm_distant_support_fallback_used": 0.0,
                    }
                )
        elif mode == DISTANT_MODE_FROZEN:
            aux.update(
                {
                    "vsm_distant_write_gate_mean": 0.0,
                    "vsm_distant_dt_mean": 0.0,
                    "vsm_distant_seen_rows": 0.0,
                    "vsm_distant_h_norm": 0.0,
                    "vsm_distant_support_max": (
                        float(support_distant.detach().float().max().item())
                        if support_distant is not None and support_distant.numel()
                        else 0.0
                    ),
                }
            )

        read = LongVSMReadPack(
            bg=read_bg,
            seen_bg=bg_seen_for_read,
            bg_indices=bg_indices,
            rigid=read_rigid,
            rigid_indices=rigid_indices,
            rigid_seen=rigid_seen_active,
            rigid_stable_mask=stable_active,
            distant=read_distant,
            distant_indices=distant_indices,
            distant_seen=distant_seen_active,
        )
        return (out_state if bool(commit_memory) else state), read, aux
