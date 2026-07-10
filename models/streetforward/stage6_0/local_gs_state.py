from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Optional, Tuple

import torch

from models.streetforward.math_utils import _axis_angle_to_quat, _normalize_quat, _quat_multiply
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


DEFAULT_APPEARANCE_SIGMA = {
    "bg": 0.08,
    "distant": 0.12,
    "rigid": 0.10,
}


def _cfg_get(node: object, key: str, default: object) -> object:
    if node is None:
        return default
    if isinstance(node, dict):
        value = node.get(key, default)
    elif hasattr(node, "get"):
        value = node.get(key, default)
    else:
        value = getattr(node, key, default)
    return default if value is None else value


def _uncertainty_state_values(cfg: object, branch_name: str) -> tuple[float, float, float, float]:
    init_sigma_cfg = _cfg_get(cfg, "init_sigma", {})
    sigma0 = float(_cfg_get(init_sigma_cfg, str(branch_name), DEFAULT_APPEARANCE_SIGMA[str(branch_name)]))
    sigma_min = float(_cfg_get(cfg, "sigma_min", 0.01))
    sigma_max = float(_cfg_get(cfg, "sigma_max", 0.50))
    prior_pull = float(_cfg_get(cfg, "prior_pull", 0.0))
    if not (0.0 < sigma_min <= sigma0 <= sigma_max):
        raise ValueError(
            f"Invalid appearance uncertainty sigma range for {branch_name}: "
            f"sigma_min={sigma_min}, sigma0={sigma0}, sigma_max={sigma_max}"
        )
    return 2.0 * math.log(sigma0), 2.0 * math.log(sigma_min), 2.0 * math.log(sigma_max), prior_pull


@dataclass
class LocalBranchState:
    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    hidden: torch.Tensor
    appearance_logvar: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.appearance_logvar is None:
            self.appearance_logvar = torch.full(
                (int(self.means.shape[0]), 1),
                2.0 * math.log(0.10),
                device=self.means.device,
                dtype=torch.float32,
            )
        else:
            self.appearance_logvar = self.appearance_logvar.to(
                device=self.means.device,
                dtype=torch.float32,
            )
        expected = (int(self.means.shape[0]), 1)
        if tuple(self.appearance_logvar.shape) != expected:
            raise ValueError(
                f"appearance_logvar must be [N,1], got {tuple(self.appearance_logvar.shape)} "
                f"for N={expected[0]}"
            )

    @classmethod
    def from_tensors(
        cls,
        *,
        state: NodeStateBackground | NodeStateDistant | NodeStateRigid,
        hidden_dim: int,
        branch_name: str = "bg",
        uncertainty_state_cfg: object = None,
    ) -> "LocalBranchState":
        def _leaf(x: torch.Tensor) -> torch.Tensor:
            y = x.detach().clone()
            if torch.is_floating_point(y):
                y.requires_grad_(True)
            return y

        n = int(state.means.shape[0])
        hidden = state.means.new_zeros((n, max(int(hidden_dim), 0)))
        if hidden.numel() > 0:
            hidden.requires_grad_(True)
        appearance = getattr(state, "appearance_logvar", None)
        if appearance is None:
            prior_logvar, _, _, _ = _uncertainty_state_values(uncertainty_state_cfg, str(branch_name))
            appearance = torch.full(
                (n, 1),
                float(prior_logvar),
                device=state.means.device,
                dtype=torch.float32,
            )
        else:
            appearance = appearance.detach().clone().to(device=state.means.device, dtype=torch.float32)
        if tuple(appearance.shape) != (n, 1):
            raise ValueError(
                f"{branch_name}.appearance_logvar must be [N,1], got {tuple(appearance.shape)} for N={n}"
            )
        appearance.requires_grad_(True)
        return cls(
            means=_leaf(state.means),
            scales_log=_leaf(state.scales_log),
            quats=_leaf(state.quats),
            opacity_logit=_leaf(state.opacity_logit),
            sh_dc=_leaf(state.sh_dc),
            sh_rest=_leaf(state.sh_rest),
            hidden=hidden,
            appearance_logvar=appearance,
        )

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        yield self.means
        yield self.scales_log
        yield self.quats
        yield self.opacity_logit
        yield self.sh_dc
        yield self.sh_rest
        yield self.hidden
        yield self.appearance_logvar

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
            appearance_logvar=self.appearance_logvar.to(device=device, dtype=torch.float32),
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
        uncertainty_state_cfg: object = None,
    ) -> "LocalGSState":
        return cls(
            bg=LocalBranchState.from_tensors(
                state=bg,
                hidden_dim=int(hidden_dim),
                branch_name="bg",
                uncertainty_state_cfg=uncertainty_state_cfg,
            ),
            distant=(
                LocalBranchState.from_tensors(
                    state=distant,
                    hidden_dim=int(hidden_dim),
                    branch_name="distant",
                    uncertainty_state_cfg=uncertainty_state_cfg,
                )
                if distant is not None
                else None
            ),
            rigid=(
                LocalBranchState.from_tensors(
                    state=rigid,
                    hidden_dim=int(hidden_dim),
                    branch_name="rigid",
                    uncertainty_state_cfg=uncertainty_state_cfg,
                )
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
            appearance_logvar=(
                None
                if getattr(rigid, "appearance_logvar", None) is None
                else rigid.appearance_logvar.to(device=device, dtype=torch.float32)
            ),
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
    def _apply_branch(
        state: LocalBranchState,
        delta: BranchDelta,
        *,
        branch_name: str,
        uncertainty_state_cfg: object = None,
    ) -> LocalBranchState:
        n = int(state.means.shape[0])
        if int(delta.means.shape[0]) != n:
            raise ValueError(f"delta/state row mismatch: {int(delta.means.shape[0])} vs {n}")
        sh_delta = delta.sh.view(n, -1, 3)
        sh_dc_delta = sh_delta[:, 0, :]
        sh_rest_delta = sh_delta[:, 1:, :] if int(sh_delta.shape[1]) > 1 else state.sh_rest.new_zeros(state.sh_rest.shape)
        if tuple(sh_rest_delta.shape) != tuple(state.sh_rest.shape):
            raise ValueError(f"delta sh_rest shape {tuple(sh_rest_delta.shape)} != state {tuple(state.sh_rest.shape)}")
        quat_delta = _axis_angle_to_quat(delta.quat_axis_angle) if delta.is_active("quat_axis_angle") else None
        prior_logvar, logvar_min, logvar_max, prior_pull = _uncertainty_state_values(
            uncertainty_state_cfg,
            str(branch_name),
        )
        appearance_delta = delta.appearance_logvar_delta.to(
            device=state.appearance_logvar.device,
            dtype=torch.float32,
        )
        appearance_next = state.appearance_logvar.float()
        if delta.is_active("appearance_logvar_delta"):
            appearance_next = appearance_next + appearance_delta
        if float(prior_pull) != 0.0:
            appearance_next = appearance_next + float(prior_pull) * (
                appearance_next.new_tensor(float(prior_logvar)) - state.appearance_logvar.float()
            )
        appearance_next = appearance_next.clamp(min=float(logvar_min), max=float(logvar_max))
        return LocalBranchState(
            means=state.means + delta.means if delta.is_active("means") else state.means,
            scales_log=state.scales_log + delta.scales_log if delta.is_active("scales_log") else state.scales_log,
            quats=(
                _normalize_quat(_quat_multiply(state.quats, quat_delta))
                if quat_delta is not None
                else state.quats
            ),
            opacity_logit=(
                state.opacity_logit + delta.opacity_logit
                if delta.is_active("opacity_logit")
                else state.opacity_logit
            ),
            sh_dc=state.sh_dc + sh_dc_delta if delta.is_active("sh") else state.sh_dc,
            sh_rest=state.sh_rest + sh_rest_delta if delta.is_active("sh") else state.sh_rest,
            hidden=state.hidden + delta.hidden if delta.is_active("hidden") else state.hidden,
            appearance_logvar=appearance_next,
        )

    def apply_delta(self, delta: DeltaPack, *, uncertainty_state_cfg: object = None) -> "LocalGSState":
        out = LocalGSState(
            bg=self._apply_branch(
                self.bg,
                delta.bg,
                branch_name="bg",
                uncertainty_state_cfg=uncertainty_state_cfg,
            ),
            distant=self.distant,
            rigid=self.rigid,
            rigid_template=self.rigid_template,
        )
        if self.distant is not None and delta.distant is not None:
            out.distant = self._apply_branch(
                self.distant,
                delta.distant,
                branch_name="distant",
                uncertainty_state_cfg=uncertainty_state_cfg,
            )
        if self.rigid is not None and delta.rigid is not None:
            out.rigid = self._apply_branch(
                self.rigid,
                delta.rigid,
                branch_name="rigid",
                uncertainty_state_cfg=uncertainty_state_cfg,
            )
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
            appearance_logvar=self.bg.appearance_logvar.detach().clone(),
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
                appearance_logvar=self.distant.appearance_logvar.detach().clone(),
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
                appearance_logvar=self.rigid.appearance_logvar.detach().clone(),
            )
        for node in (bg, distant, rigid):
            if node is None:
                continue
            for value in node.__dict__.values():
                if torch.is_tensor(value) and value.requires_grad:
                    raise RuntimeError("Persistent writeback tensor must be detached.")
        return bg, distant, rigid

    def to_node_states_detached_view(self) -> Tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        bg = NodeStateBackground(
            means=self.bg.means.detach(),
            scales_log=self.bg.scales_log.detach(),
            quats=self.bg.quats.detach(),
            opacity_logit=self.bg.opacity_logit.detach(),
            sh_dc=self.bg.sh_dc.detach(),
            sh_rest=self.bg.sh_rest.detach(),
            appearance_logvar=self.bg.appearance_logvar.detach(),
        )
        distant = None
        if self.distant is not None:
            distant = NodeStateDistant(
                means=self.distant.means.detach(),
                scales_log=self.distant.scales_log.detach(),
                quats=self.distant.quats.detach(),
                opacity_logit=self.distant.opacity_logit.detach(),
                sh_dc=self.distant.sh_dc.detach(),
                sh_rest=self.distant.sh_rest.detach(),
                appearance_logvar=self.distant.appearance_logvar.detach(),
            )
        rigid = None
        if self.rigid is not None:
            if self.rigid_template is None:
                raise ValueError("rigid local state is present without rigid_template")
            rigid = NodeStateRigid(
                means=self.rigid.means.detach(),
                scales_log=self.rigid.scales_log.detach(),
                quats=self.rigid.quats.detach(),
                opacity_logit=self.rigid.opacity_logit.detach(),
                sh_dc=self.rigid.sh_dc.detach(),
                sh_rest=self.rigid.sh_rest.detach(),
                point_ids=self.rigid_template.point_ids.detach(),
                instances_quats=self.rigid_template.instances_quats.detach(),
                instances_trans=self.rigid_template.instances_trans.detach(),
                instances_fv=self.rigid_template.instances_fv.detach(),
                instance_ids=list(self.rigid_template.instance_ids),
                frame_ids=list(self.rigid_template.frame_ids),
                cur_frame=int(self.rigid_template.cur_frame),
                appearance_logvar=self.rigid.appearance_logvar.detach(),
            )
        return bg, distant, rigid

    def to_node_states_grad(self) -> Tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        bg = NodeStateBackground(
            means=self.bg.means,
            scales_log=self.bg.scales_log,
            quats=self.bg.quats,
            opacity_logit=self.bg.opacity_logit,
            sh_dc=self.bg.sh_dc,
            sh_rest=self.bg.sh_rest,
            appearance_logvar=self.bg.appearance_logvar,
        )
        distant = None
        if self.distant is not None:
            distant = NodeStateDistant(
                means=self.distant.means,
                scales_log=self.distant.scales_log,
                quats=self.distant.quats,
                opacity_logit=self.distant.opacity_logit,
                sh_dc=self.distant.sh_dc,
                sh_rest=self.distant.sh_rest,
                appearance_logvar=self.distant.appearance_logvar,
            )
        rigid = None
        if self.rigid is not None:
            if self.rigid_template is None:
                raise ValueError("rigid local state is present without rigid_template")
            rigid = NodeStateRigid(
                means=self.rigid.means,
                scales_log=self.rigid.scales_log,
                quats=self.rigid.quats,
                opacity_logit=self.rigid.opacity_logit,
                sh_dc=self.rigid.sh_dc,
                sh_rest=self.rigid.sh_rest,
                point_ids=self.rigid_template.point_ids,
                instances_quats=self.rigid_template.instances_quats,
                instances_trans=self.rigid_template.instances_trans,
                instances_fv=self.rigid_template.instances_fv,
                instance_ids=list(self.rigid_template.instance_ids),
                frame_ids=list(self.rigid_template.frame_ids),
                cur_frame=int(self.rigid_template.cur_frame),
                appearance_logvar=self.rigid.appearance_logvar,
            )
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
