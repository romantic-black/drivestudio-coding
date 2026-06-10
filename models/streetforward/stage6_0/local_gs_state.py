from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch

from models.streetforward.math_utils import _axis_angle_to_quat, _normalize_quat, _quat_multiply
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


@dataclass
class LocalBranchState:
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    hidden: torch.Tensor

    @classmethod
    def from_tensors(cls, *, state: NodeStateBackground | NodeStateDistant | NodeStateRigid, hidden_dim: int) -> "LocalBranchState":
        def _leaf(x: torch.Tensor) -> torch.Tensor:
            y = x.detach().clone()
            if torch.is_floating_point(y):
                y.requires_grad_(True)
            return y

        n = int(state.means.shape[0])
        hidden = state.means.new_zeros((n, int(hidden_dim)))
        hidden.requires_grad_(True)
        return cls(
            means=_leaf(state.means),
            scales_log=_leaf(state.scales_log),
            quats=_leaf(state.quats),
            opacity_logit=_leaf(state.opacity_logit),
            sh_dc=_leaf(state.sh_dc),
            sh_rest=_leaf(state.sh_rest),
            hidden=hidden,
        )

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        yield self.means
        yield self.scales_log
        yield self.quats
        yield self.opacity_logit
        yield self.sh_dc
        yield self.sh_rest
        yield self.hidden

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "LocalBranchState":
        out_dtype = dtype or self.means.dtype
        return LocalBranchState(
            means=self.means.to(device=device, dtype=out_dtype),
            scales_log=self.scales_log.to(device=device, dtype=out_dtype),
            quats=self.quats.to(device=device, dtype=out_dtype),
            opacity_logit=self.opacity_logit.to(device=device, dtype=out_dtype),
            sh_dc=self.sh_dc.to(device=device, dtype=out_dtype),
            sh_rest=self.sh_rest.to(device=device, dtype=out_dtype),
            hidden=self.hidden.to(device=device, dtype=out_dtype),
        )


@dataclass
class LocalGSState:
    bg: LocalBranchState
    distant: Optional[LocalBranchState] = None
    rigid: Optional[LocalBranchState] = None
    rigid_template: Optional[NodeStateRigid] = None

    @classmethod
    def from_node_states(
        cls,
        *,
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
        hidden_dim: int,
    ) -> "LocalGSState":
        return cls(
            bg=LocalBranchState.from_tensors(state=bg, hidden_dim=int(hidden_dim)),
            distant=(
                LocalBranchState.from_tensors(state=distant, hidden_dim=int(hidden_dim))
                if distant is not None
                else None
            ),
            rigid=(
                LocalBranchState.from_tensors(state=rigid, hidden_dim=int(hidden_dim))
                if rigid is not None
                else None
            ),
            rigid_template=rigid.detach_clone() if rigid is not None else None,
        )

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        yield from self.bg.iter_tensors()
        if self.distant is not None:
            yield from self.distant.iter_tensors()
        if self.rigid is not None:
            yield from self.rigid.iter_tensors()

    @staticmethod
    def _rigid_template_to(
        rigid: Optional[NodeStateRigid],
        *,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> Optional[NodeStateRigid]:
        if rigid is None:
            return None
        out_dtype = dtype or rigid.means.dtype
        return NodeStateRigid(
            means=rigid.means.to(device=device, dtype=out_dtype),
            scales_log=rigid.scales_log.to(device=device, dtype=out_dtype),
            quats=rigid.quats.to(device=device, dtype=out_dtype),
            opacity_logit=rigid.opacity_logit.to(device=device, dtype=out_dtype),
            sh_dc=rigid.sh_dc.to(device=device, dtype=out_dtype),
            sh_rest=rigid.sh_rest.to(device=device, dtype=out_dtype),
            point_ids=rigid.point_ids.to(device=device),
            instances_quats=rigid.instances_quats.to(device=device, dtype=out_dtype),
            instances_trans=rigid.instances_trans.to(device=device, dtype=out_dtype),
            instances_fv=rigid.instances_fv.to(device=device),
            instance_ids=list(rigid.instance_ids),
            frame_ids=list(rigid.frame_ids),
            cur_frame=int(rigid.cur_frame),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "LocalGSState":
        out_dtype = dtype or self.bg.means.dtype
        return LocalGSState(
            bg=self.bg.to(device=device, dtype=out_dtype),
            distant=self.distant.to(device=device, dtype=out_dtype) if self.distant is not None else None,
            rigid=self.rigid.to(device=device, dtype=out_dtype) if self.rigid is not None else None,
            rigid_template=self._rigid_template_to(self.rigid_template, device=device, dtype=out_dtype),
        )

    def assert_finite(self, label: str = "local_G") -> None:
        for idx, tensor in enumerate(self.iter_tensors()):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{label} tensor[{idx}] contains NaN/Inf")

    @staticmethod
    def _apply_branch(state: LocalBranchState, delta: BranchDelta) -> LocalBranchState:
        n = int(state.means.shape[0])
        if int(delta.means.shape[0]) != n:
            raise ValueError(f"delta/state row mismatch: {int(delta.means.shape[0])} vs {n}")
        sh_delta = delta.sh.view(n, -1, 3)
        sh_dc_delta = sh_delta[:, 0, :]
        sh_rest_delta = sh_delta[:, 1:, :] if int(sh_delta.shape[1]) > 1 else state.sh_rest.new_zeros(state.sh_rest.shape)
        if tuple(sh_rest_delta.shape) != tuple(state.sh_rest.shape):
            raise ValueError(f"delta sh_rest shape {tuple(sh_rest_delta.shape)} != state {tuple(state.sh_rest.shape)}")
        quat_delta = _axis_angle_to_quat(delta.quat_axis_angle)
        return LocalBranchState(
            means=state.means + delta.means,
            scales_log=state.scales_log + delta.scales_log,
            quats=_normalize_quat(_quat_multiply(state.quats, quat_delta)),
            opacity_logit=state.opacity_logit + delta.opacity_logit,
            sh_dc=state.sh_dc + sh_dc_delta,
            sh_rest=state.sh_rest + sh_rest_delta,
            hidden=state.hidden + delta.hidden,
        )

    def apply_delta(self, delta: DeltaPack) -> "LocalGSState":
        out = LocalGSState(
            bg=self._apply_branch(self.bg, delta.bg),
            distant=self.distant,
            rigid=self.rigid,
            rigid_template=self.rigid_template,
        )
        if self.distant is not None and delta.distant is not None:
            out.distant = self._apply_branch(self.distant, delta.distant)
        if self.rigid is not None and delta.rigid is not None:
            out.rigid = self._apply_branch(self.rigid, delta.rigid)
        out.assert_finite("local_G_after_delta")
        return out

    def to_node_states_detached(self) -> Tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        bg = NodeStateBackground(
            means=self.bg.means.detach().clone(),
            scales_log=self.bg.scales_log.detach().clone(),
            quats=self.bg.quats.detach().clone(),
            opacity_logit=self.bg.opacity_logit.detach().clone(),
            sh_dc=self.bg.sh_dc.detach().clone(),
            sh_rest=self.bg.sh_rest.detach().clone(),
        )
        distant = None
        if self.distant is not None:
            distant = NodeStateDistant(
                means=self.distant.means.detach().clone(),
                scales_log=self.distant.scales_log.detach().clone(),
                quats=self.distant.quats.detach().clone(),
                opacity_logit=self.distant.opacity_logit.detach().clone(),
                sh_dc=self.distant.sh_dc.detach().clone(),
                sh_rest=self.distant.sh_rest.detach().clone(),
            )
        rigid = None
        if self.rigid is not None:
            if self.rigid_template is None:
                raise ValueError("rigid local state is present without rigid_template")
            rigid = NodeStateRigid(
                means=self.rigid.means.detach().clone(),
                scales_log=self.rigid.scales_log.detach().clone(),
                quats=self.rigid.quats.detach().clone(),
                opacity_logit=self.rigid.opacity_logit.detach().clone(),
                sh_dc=self.rigid.sh_dc.detach().clone(),
                sh_rest=self.rigid.sh_rest.detach().clone(),
                point_ids=self.rigid_template.point_ids.detach().clone(),
                instances_quats=self.rigid_template.instances_quats.detach().clone(),
                instances_trans=self.rigid_template.instances_trans.detach().clone(),
                instances_fv=self.rigid_template.instances_fv.detach().clone(),
                instance_ids=list(self.rigid_template.instance_ids),
                frame_ids=list(self.rigid_template.frame_ids),
                cur_frame=int(self.rigid_template.cur_frame),
            )
        for node in (bg, distant, rigid):
            if node is None:
                continue
            for value in node.__dict__.values():
                if torch.is_tensor(value) and value.requires_grad:
                    raise RuntimeError("Persistent writeback tensor must be detached.")
        return bg, distant, rigid

    def writeback_detached(
        self,
        *,
        bg: NodeStateBackground,
        distant: Optional[NodeStateDistant],
        rigid: Optional[NodeStateRigid],
    ) -> None:
        bg_new, distant_new, rigid_new = self.to_node_states_detached()
        bg.__dict__.update(bg_new.__dict__)
        if distant is not None and distant_new is not None:
            distant.__dict__.update(distant_new.__dict__)
        if rigid is not None and rigid_new is not None:
            rigid.__dict__.update(rigid_new.__dict__)
