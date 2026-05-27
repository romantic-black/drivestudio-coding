from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import torch

from models.streetforward.stage6_0.local_gs_state import LocalBranchState, LocalGSState

from .types import LongOffsetDelta, rigid_stable_mask_from_meta


@dataclass
class RigidFrameOffsetSnapshot:
    means_local: torch.Tensor
    scales_log: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    mask: torch.Tensor

    def detach(self) -> "RigidFrameOffsetSnapshot":
        return RigidFrameOffsetSnapshot(
            means_local=self.means_local.detach(),
            scales_log=self.scales_log.detach(),
            opacity_logit=self.opacity_logit.detach(),
            sh_dc=self.sh_dc.detach(),
            mask=self.mask.detach(),
        )


@dataclass
class PhaseBOffsetState:
    bg_means: torch.Tensor
    bg_scales_log: torch.Tensor
    bg_opacity_logit: torch.Tensor
    bg_sh_dc: torch.Tensor
    bg_sh_rest: Optional[torch.Tensor] = None
    distant_scales_log: Optional[torch.Tensor] = None
    distant_opacity_logit: Optional[torch.Tensor] = None
    distant_sh_dc: Optional[torch.Tensor] = None
    distant_sh_rest: Optional[torch.Tensor] = None
    rigid_means_local: Optional[torch.Tensor] = None
    rigid_scales_log: Optional[torch.Tensor] = None
    rigid_opacity_logit: Optional[torch.Tensor] = None
    rigid_sh_dc: Optional[torch.Tensor] = None
    rigid_frame_snapshots: Dict[int, RigidFrameOffsetSnapshot] = field(default_factory=dict)

    @classmethod
    def zeros_like(
        cls,
        *,
        base_state: LocalGSState,
        dtype: Optional[torch.dtype] = None,
    ) -> "PhaseBOffsetState":
        ref = base_state.bg.means
        dt = dtype or ref.dtype

        def z_like(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x, dtype=dt, device=x.device)

        distant_scales = distant_opacity = distant_sh_dc = distant_sh_rest = None
        if base_state.distant is not None:
            distant_scales = z_like(base_state.distant.scales_log)
            distant_opacity = z_like(base_state.distant.opacity_logit)
            distant_sh_dc = z_like(base_state.distant.sh_dc)
            distant_sh_rest = z_like(base_state.distant.sh_rest)
        rigid_means = rigid_scales = rigid_opacity = rigid_sh_dc = None
        if base_state.rigid is not None:
            rigid_means = z_like(base_state.rigid.means)
            rigid_scales = z_like(base_state.rigid.scales_log)
            rigid_opacity = z_like(base_state.rigid.opacity_logit)
            rigid_sh_dc = z_like(base_state.rigid.sh_dc)
        return cls(
            bg_means=z_like(base_state.bg.means),
            bg_scales_log=z_like(base_state.bg.scales_log),
            bg_opacity_logit=z_like(base_state.bg.opacity_logit),
            bg_sh_dc=z_like(base_state.bg.sh_dc),
            distant_scales_log=distant_scales,
            distant_opacity_logit=distant_opacity,
            distant_sh_dc=distant_sh_dc,
            distant_sh_rest=distant_sh_rest,
            rigid_means_local=rigid_means,
            rigid_scales_log=rigid_scales,
            rigid_opacity_logit=rigid_opacity,
            rigid_sh_dc=rigid_sh_dc,
        )

    def detach_for_sensor(self) -> "PhaseBOffsetState":
        def d(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.detach()

        return PhaseBOffsetState(
            bg_means=self.bg_means.detach(),
            bg_scales_log=self.bg_scales_log.detach(),
            bg_opacity_logit=self.bg_opacity_logit.detach(),
            bg_sh_dc=self.bg_sh_dc.detach(),
            bg_sh_rest=d(self.bg_sh_rest),
            distant_scales_log=d(self.distant_scales_log),
            distant_opacity_logit=d(self.distant_opacity_logit),
            distant_sh_dc=d(self.distant_sh_dc),
            distant_sh_rest=d(self.distant_sh_rest),
            rigid_means_local=d(self.rigid_means_local),
            rigid_scales_log=d(self.rigid_scales_log),
            rigid_opacity_logit=d(self.rigid_opacity_logit),
            rigid_sh_dc=d(self.rigid_sh_dc),
            rigid_frame_snapshots={int(k): v.detach() for k, v in self.rigid_frame_snapshots.items()},
        )

    def _empty_snapshot(self, frame_idx: int) -> RigidFrameOffsetSnapshot:
        if self.rigid_means_local is None or self.rigid_scales_log is None or self.rigid_opacity_logit is None or self.rigid_sh_dc is None:
            raise ValueError("cannot create rigid snapshot without rigid offset tensors")
        _ = frame_idx
        return RigidFrameOffsetSnapshot(
            means_local=torch.zeros_like(self.rigid_means_local),
            scales_log=torch.zeros_like(self.rigid_scales_log),
            opacity_logit=torch.zeros_like(self.rigid_opacity_logit),
            sh_dc=torch.zeros_like(self.rigid_sh_dc),
            mask=torch.zeros(
                (int(self.rigid_means_local.shape[0]), 1),
                device=self.rigid_means_local.device,
                dtype=self.rigid_means_local.dtype,
            ),
        )

    def apply(
        self,
        delta: LongOffsetDelta,
        *,
        frame_idx: int,
        rigid_meta: Optional[Dict[str, Any]] = None,
    ) -> "PhaseBOffsetState":
        bg_mask = delta.bg.mask.to(device=self.bg_means.device, dtype=self.bg_means.dtype)
        if delta.bg.indices is None:
            out = replace(
                self,
                bg_means=self.bg_means
                + delta.bg.means.to(device=self.bg_means.device, dtype=self.bg_means.dtype) * bg_mask,
                bg_scales_log=self.bg_scales_log
                + delta.bg.scales_log.to(device=self.bg_scales_log.device, dtype=self.bg_scales_log.dtype) * bg_mask,
                bg_opacity_logit=self.bg_opacity_logit
                + delta.bg.opacity_logit.to(device=self.bg_opacity_logit.device, dtype=self.bg_opacity_logit.dtype) * bg_mask,
                bg_sh_dc=self.bg_sh_dc
                + delta.bg.sh_dc.to(device=self.bg_sh_dc.device, dtype=self.bg_sh_dc.dtype) * bg_mask,
                rigid_frame_snapshots=dict(self.rigid_frame_snapshots),
            )
        else:
            bg_idx = delta.bg.indices.to(device=self.bg_means.device, dtype=torch.long).reshape(-1)
            if int(bg_idx.numel()) != int(delta.bg.means.shape[0]):
                raise ValueError("sparse bg delta indices must match delta rows.")
            out = replace(
                self,
                bg_means=self.bg_means.index_add(
                    0,
                    bg_idx,
                    delta.bg.means.to(device=self.bg_means.device, dtype=self.bg_means.dtype) * bg_mask,
                ),
                bg_scales_log=self.bg_scales_log.index_add(
                    0,
                    bg_idx,
                    delta.bg.scales_log.to(device=self.bg_scales_log.device, dtype=self.bg_scales_log.dtype)
                    * bg_mask.to(device=self.bg_scales_log.device, dtype=self.bg_scales_log.dtype),
                ),
                bg_opacity_logit=self.bg_opacity_logit.index_add(
                    0,
                    bg_idx,
                    delta.bg.opacity_logit.to(
                        device=self.bg_opacity_logit.device,
                        dtype=self.bg_opacity_logit.dtype,
                    )
                    * bg_mask.to(device=self.bg_opacity_logit.device, dtype=self.bg_opacity_logit.dtype),
                ),
                bg_sh_dc=self.bg_sh_dc.index_add(
                    0,
                    bg_idx,
                    delta.bg.sh_dc.to(device=self.bg_sh_dc.device, dtype=self.bg_sh_dc.dtype)
                    * bg_mask.to(device=self.bg_sh_dc.device, dtype=self.bg_sh_dc.dtype),
                ),
                rigid_frame_snapshots=dict(self.rigid_frame_snapshots),
            )
        if delta.distant is not None:
            if (
                out.distant_scales_log is None
                or out.distant_opacity_logit is None
                or out.distant_sh_dc is None
                or out.distant_sh_rest is None
            ):
                raise ValueError("distant delta present but offset state has no distant branch")
            d_idx = delta.distant.indices.to(device=out.distant_scales_log.device, dtype=torch.long).reshape(-1)
            if int(d_idx.numel()) != int(delta.distant.scales_log.shape[0]):
                raise ValueError("distant delta indices must match delta rows.")
            if int(d_idx.numel()) > 0:
                d_mask = delta.distant.mask.to(device=out.distant_scales_log.device, dtype=out.distant_scales_log.dtype)
                d_scales = out.distant_scales_log.clone()
                d_opacity = out.distant_opacity_logit.clone()
                d_sh_dc = out.distant_sh_dc.clone()
                d_sh_rest = out.distant_sh_rest.clone()
                d_scales[d_idx] = d_scales[d_idx] + delta.distant.scales_log.to(
                    device=d_scales.device,
                    dtype=d_scales.dtype,
                ) * d_mask
                d_opacity[d_idx] = d_opacity[d_idx] + delta.distant.opacity_logit.to(
                    device=d_opacity.device,
                    dtype=d_opacity.dtype,
                ) * d_mask.to(device=d_opacity.device, dtype=d_opacity.dtype)
                d_sh_dc[d_idx] = d_sh_dc[d_idx] + delta.distant.sh_dc.to(
                    device=d_sh_dc.device,
                    dtype=d_sh_dc.dtype,
                ) * d_mask.to(device=d_sh_dc.device, dtype=d_sh_dc.dtype)
                sh_rest_delta = delta.distant.sh_rest.to(device=d_sh_rest.device, dtype=d_sh_rest.dtype)
                if int(sh_rest_delta.shape[1]) > int(d_sh_rest.shape[1]):
                    raise ValueError("distant sh_rest delta has more bases than offset state.")
                if int(sh_rest_delta.shape[1]) > 0:
                    bases = int(sh_rest_delta.shape[1])
                    d_sh_rest[d_idx, :bases, :] = (
                        d_sh_rest[d_idx, :bases, :]
                        + sh_rest_delta * d_mask.to(device=d_sh_rest.device, dtype=d_sh_rest.dtype)[:, None, :]
                    )
                out = replace(
                    out,
                    distant_scales_log=d_scales,
                    distant_opacity_logit=d_opacity,
                    distant_sh_dc=d_sh_dc,
                    distant_sh_rest=d_sh_rest,
                )

        if delta.rigid is None:
            return out
        if (
            out.rigid_means_local is None
            or out.rigid_scales_log is None
            or out.rigid_opacity_logit is None
            or out.rigid_sh_dc is None
        ):
            raise ValueError("rigid delta present but offset state has no rigid branch")

        idx = delta.rigid.indices.to(device=out.rigid_means_local.device, dtype=torch.long).reshape(-1)
        stable_mask = delta.rigid.stable_mask.to(device=out.rigid_means_local.device, dtype=torch.bool).reshape(-1)
        if int(idx.numel()) == 0:
            return out
        if int(stable_mask.numel()) != int(idx.numel()):
            raise ValueError("rigid delta stable_mask rows must match indices.")

        if bool(stable_mask.any().item()):
            rows = torch.nonzero(stable_mask, as_tuple=False).squeeze(1)
            stable_idx = idx.index_select(0, rows)
            rigid_means = out.rigid_means_local.clone()
            rigid_scales = out.rigid_scales_log.clone()
            rigid_opacity = out.rigid_opacity_logit.clone()
            rigid_sh_dc = out.rigid_sh_dc.clone()
            rigid_means[stable_idx] = rigid_means[stable_idx] + delta.rigid.means_local.index_select(0, rows).to(
                device=rigid_means.device, dtype=rigid_means.dtype
            )
            rigid_scales[stable_idx] = rigid_scales[stable_idx] + delta.rigid.scales_log.index_select(0, rows).to(
                device=rigid_scales.device, dtype=rigid_scales.dtype
            )
            rigid_opacity[stable_idx] = rigid_opacity[stable_idx] + delta.rigid.opacity_logit.index_select(0, rows).to(
                device=rigid_opacity.device, dtype=rigid_opacity.dtype
            )
            rigid_sh_dc[stable_idx] = rigid_sh_dc[stable_idx] + delta.rigid.sh_dc.index_select(0, rows).to(
                device=rigid_sh_dc.device, dtype=rigid_sh_dc.dtype
            )
            out = replace(
                out,
                rigid_means_local=rigid_means,
                rigid_scales_log=rigid_scales,
                rigid_opacity_logit=rigid_opacity,
                rigid_sh_dc=rigid_sh_dc,
            )

        unstable_mask = ~stable_mask
        if bool(unstable_mask.any().item()):
            rows = torch.nonzero(unstable_mask, as_tuple=False).squeeze(1)
            unstable_idx = idx.index_select(0, rows)
            snap = out.rigid_frame_snapshots.get(int(frame_idx))
            if snap is None:
                snap = out._empty_snapshot(int(frame_idx))
            means = snap.means_local.clone()
            scales = snap.scales_log.clone()
            opacity = snap.opacity_logit.clone()
            sh_dc = snap.sh_dc.clone()
            mask = snap.mask.clone()
            means[unstable_idx] = means[unstable_idx] + delta.rigid.means_local.index_select(0, rows).to(
                device=means.device, dtype=means.dtype
            )
            scales[unstable_idx] = scales[unstable_idx] + delta.rigid.scales_log.index_select(0, rows).to(
                device=scales.device, dtype=scales.dtype
            )
            opacity[unstable_idx] = opacity[unstable_idx] + delta.rigid.opacity_logit.index_select(0, rows).to(
                device=opacity.device, dtype=opacity.dtype
            )
            sh_dc[unstable_idx] = sh_dc[unstable_idx] + delta.rigid.sh_dc.index_select(0, rows).to(
                device=sh_dc.device, dtype=sh_dc.dtype
            )
            mask[unstable_idx] = 1.0
            out.rigid_frame_snapshots[int(frame_idx)] = RigidFrameOffsetSnapshot(
                means_local=means,
                scales_log=scales,
                opacity_logit=opacity,
                sh_dc=sh_dc,
                mask=mask,
            )
        _ = rigid_meta
        return out

    def stats(self) -> Dict[str, float]:
        out = {
            "phase_b_long/offset_bg_means_norm": _mean_norm(self.bg_means),
            "phase_b_long/offset_bg_scales_norm": _mean_norm(self.bg_scales_log),
            "phase_b_long/offset_bg_opacity_norm": _mean_norm(self.bg_opacity_logit),
            "phase_b_long/offset_bg_sh_dc_norm": _mean_norm(self.bg_sh_dc),
            "phase_b_long/offset_distant_enabled": 1.0 if self.distant_scales_log is not None else 0.0,
        }
        if self.distant_scales_log is not None:
            out.update(
                {
                    "phase_b_long/offset_distant_scales_norm": _mean_norm(self.distant_scales_log),
                    "phase_b_long/offset_distant_opacity_norm": _mean_norm(self.distant_opacity_logit),
                    "phase_b_long/offset_distant_sh_dc_norm": _mean_norm(self.distant_sh_dc),
                    "phase_b_long/offset_distant_sh_rest_norm": _mean_norm(self.distant_sh_rest),
                }
            )
        if self.rigid_means_local is not None:
            out["phase_b_long/offset_rigid_means_local_norm"] = _mean_norm(self.rigid_means_local)
            out["phase_b_long/vsm_rigid_snapshot_frames"] = float(len(self.rigid_frame_snapshots))
        return out


def _mean_norm(x: Optional[torch.Tensor]) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    return float(x.detach().float().norm(dim=-1).mean().item())


def _apply_offsets(branch: LocalBranchState, *, means: torch.Tensor, scales: torch.Tensor, opacity: torch.Tensor, sh_dc: torch.Tensor) -> LocalBranchState:
    return replace(
        branch,
        means=branch.means + means.to(device=branch.means.device, dtype=branch.means.dtype),
        scales_log=branch.scales_log + scales.to(device=branch.scales_log.device, dtype=branch.scales_log.dtype),
        opacity_logit=branch.opacity_logit + opacity.to(device=branch.opacity_logit.device, dtype=branch.opacity_logit.dtype),
        sh_dc=branch.sh_dc + sh_dc.to(device=branch.sh_dc.device, dtype=branch.sh_dc.dtype),
    )


def materialize_phase_b_state(
    *,
    base_state: LocalGSState,
    offset: PhaseBOffsetState,
    target_frame_idx: Optional[int] = None,
    rigid_meta: Optional[Dict[str, Any]] = None,
) -> LocalGSState:
    bg = _apply_offsets(
        base_state.bg,
        means=offset.bg_means,
        scales=offset.bg_scales_log,
        opacity=offset.bg_opacity_logit,
        sh_dc=offset.bg_sh_dc,
    )
    distant = base_state.distant
    if distant is not None:
        distant_updates: Dict[str, torch.Tensor] = {}
        if offset.distant_scales_log is not None:
            distant_updates["scales_log"] = distant.scales_log + offset.distant_scales_log.to(
                device=distant.scales_log.device,
                dtype=distant.scales_log.dtype,
            )
        if offset.distant_opacity_logit is not None:
            distant_updates["opacity_logit"] = distant.opacity_logit + offset.distant_opacity_logit.to(
                device=distant.opacity_logit.device,
                dtype=distant.opacity_logit.dtype,
            )
        if offset.distant_sh_dc is not None:
            distant_updates["sh_dc"] = distant.sh_dc + offset.distant_sh_dc.to(
                device=distant.sh_dc.device,
                dtype=distant.sh_dc.dtype,
            )
        if offset.distant_sh_rest is not None:
            distant_updates["sh_rest"] = distant.sh_rest + offset.distant_sh_rest.to(
                device=distant.sh_rest.device,
                dtype=distant.sh_rest.dtype,
            )
        if distant_updates:
            distant = replace(distant, **distant_updates)

    rigid = base_state.rigid
    fallback_rows = 0
    if rigid is not None and offset.rigid_means_local is not None:
        n_rigid = int(rigid.means.shape[0])
        stable_mask = rigid_stable_mask_from_meta(rigid_meta, num_rows=n_rigid, device=rigid.means.device)
        stable_f = stable_mask[:, None].to(dtype=rigid.means.dtype)
        means = offset.rigid_means_local.to(device=rigid.means.device, dtype=rigid.means.dtype) * stable_f
        scales = offset.rigid_scales_log.to(device=rigid.scales_log.device, dtype=rigid.scales_log.dtype) * stable_f
        opacity = offset.rigid_opacity_logit.to(device=rigid.opacity_logit.device, dtype=rigid.opacity_logit.dtype) * stable_f
        sh_dc = offset.rigid_sh_dc.to(device=rigid.sh_dc.device, dtype=rigid.sh_dc.dtype) * stable_f
        if target_frame_idx is not None:
            snap = offset.rigid_frame_snapshots.get(int(target_frame_idx))
            unstable_f = (~stable_mask)[:, None].to(dtype=rigid.means.dtype)
            if snap is not None:
                snap_mask = snap.mask.to(device=rigid.means.device, dtype=rigid.means.dtype)
                means = means + snap.means_local.to(device=rigid.means.device, dtype=rigid.means.dtype) * snap_mask
                scales = scales + snap.scales_log.to(device=rigid.scales_log.device, dtype=rigid.scales_log.dtype) * snap_mask
                opacity = opacity + snap.opacity_logit.to(device=rigid.opacity_logit.device, dtype=rigid.opacity_logit.dtype) * snap_mask
                sh_dc = sh_dc + snap.sh_dc.to(device=rigid.sh_dc.device, dtype=rigid.sh_dc.dtype) * snap_mask
                fallback_rows = int(((unstable_f > 0) & (snap_mask <= 0)).sum().item())
            else:
                fallback_rows = int(unstable_f.sum().item())
        rigid = _apply_offsets(rigid, means=means, scales=scales, opacity=opacity, sh_dc=sh_dc)
    out = LocalGSState(bg=bg, distant=distant, rigid=rigid, rigid_template=base_state.rigid_template)
    setattr(out, "_phase_b_long_rigid_fallback_rows", int(fallback_rows))
    return out
