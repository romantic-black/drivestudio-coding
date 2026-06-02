from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0 import ContextPack, DeltaPack, EventPack, LocalGSState
from models.streetforward.stage6_0.phase_a_losses import delta_regularization


class IForwardStage6Bridge:
    """Private low-level bridge into the Stage6 V4 measurement/render stack.

    IForward does not inherit the Stage6 trainer and never calls Stage6 phase
    forward/train paths. This adapter exposes only the reusable primitives:
    node-state initialization, V4 observation, event construction, updater and
    render supervision.
    """

    def __init__(self, runtime: Any):
        self.runtime = runtime

    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.runtime, "stage6_hidden_dim", 48))

    @property
    def event_dim(self) -> int:
        return int(getattr(self.runtime, "stage6_event_dim", 48))

    @property
    def current_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_block_mask_policy", "non_sky_non_egocar"))

    @property
    def nearby_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_nearby_mask_policy", "non_sky_non_egocar"))

    def get_or_init_node_states(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        return self.runtime._get_or_init_node_states_bg_rigid_distant(batch)

    def make_local_state(
        self,
        *,
        batch: Dict[str, Any],
    ) -> Tuple[LocalGSState, NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        node_state_bg, node_state_rigid, node_state_distant = self.get_or_init_node_states(batch)
        local_state = LocalGSState.from_node_states(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
            hidden_dim=int(self.hidden_dim),
        )
        return local_state, node_state_bg, node_state_distant, node_state_rigid

    def sync_local_state_template_from_batch(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
    ) -> Tuple[NodeStateBackground, Optional[NodeStateDistant], Optional[NodeStateRigid]]:
        node_state_bg, node_state_rigid, node_state_distant = self.get_or_init_node_states(batch)
        if local_state.rigid is None:
            if node_state_rigid is not None:
                raise ValueError("IForward carried local state has no rigid branch but batch node state has rigid branch.")
            return node_state_bg, node_state_distant, node_state_rigid
        if node_state_rigid is None:
            raise ValueError("IForward carried local state has rigid branch but batch node state has no rigid branch.")
        if int(local_state.rigid.means.shape[0]) != int(node_state_rigid.means.shape[0]):
            raise ValueError(
                "IForward carried rigid row count does not match current batch node state: "
                f"local={int(local_state.rigid.means.shape[0])} node={int(node_state_rigid.means.shape[0])}"
            )
        template = getattr(local_state, "rigid_template", None)
        if template is not None and tuple(template.point_ids.shape) == tuple(node_state_rigid.point_ids.shape):
            if not torch.equal(template.point_ids.to(device=node_state_rigid.point_ids.device), node_state_rigid.point_ids):
                raise ValueError("IForward carried rigid point ids do not match current batch node state.")
        local_state.rigid_template = node_state_rigid.detach_clone()
        return node_state_bg, node_state_distant, node_state_rigid

    def observe(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
    ) -> Dict[str, Any]:
        return self.runtime._observe_v4_measurement(
            local_state=local_state,
            batch=batch,
            source_indices=[int(x) for x in source_indices],
            source_frame_idx=int(source_frame_idx),
        )

    def build_event(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> EventPack:
        return self.runtime._build_stage6_event_from_measurement(local_state=local_state, measurement=measurement)

    def apply_update(
        self,
        *,
        local_state: LocalGSState,
        event: EventPack,
        ctx_memory: Optional[ContextPack],
    ) -> Tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        return self.runtime._apply_event_update(local_state=local_state, event=event, ctx_vsm=ctx_memory)

    def render_loss(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        target_indices: List[int],
        mask_policy: str,
        pred_rgbs_out: Optional[List[torch.Tensor]] = None,
        gt_images_out: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.runtime._render_loss_for_indices(
            local_state=local_state,
            batch=batch,
            target_indices=[int(x) for x in target_indices],
            mask_policy=str(mask_policy),
            pred_rgbs_out=pred_rgbs_out,
            gt_images_out=gt_images_out,
        )

    def render_loss_for_targets(
        self,
        *,
        local_state: LocalGSState,
        ref_batch: Dict[str, Any],
        targets: List[Dict[str, Any]],
        mask_policy: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not targets:
            ref = local_state.bg.means
            return ref.new_tensor(0.0), {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
        batch = dict(ref_batch)
        batch["targets"] = list(targets)
        return self.render_loss(
            local_state=local_state,
            batch=batch,
            target_indices=list(range(len(targets))),
            mask_policy=mask_policy,
        )

    def delta_regularization(
        self,
        delta: DeltaPack,
        *,
        local_state: LocalGSState,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return delta_regularization(
            delta,
            weight=float(getattr(self.runtime, "stage6_delta_l2_weight", 1.0e-3)),
            local_state=local_state,
            opacity_delta_l2_weight=float(getattr(self.runtime, "stage6_opacity_delta_l2_weight", 0.0)),
            sh_delta_l2_weight=float(getattr(self.runtime, "stage6_sh_delta_l2_weight", 0.0)),
            scale_barrier_weight=float(getattr(self.runtime, "stage6_scale_barrier_weight", 0.0)),
            scale_log_min=float(getattr(self.runtime, "stage6_scale_log_min", -10.0)),
            scale_log_max=float(getattr(self.runtime, "stage6_scale_log_max", 4.0)),
        )

    def writeback_detached(
        self,
        *,
        local_state: LocalGSState,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
        node_state_rigid: Optional[NodeStateRigid],
    ) -> None:
        local_state.writeback_detached(
            bg=node_state_bg,
            distant=node_state_distant,
            rigid=node_state_rigid,
        )

    def reset_runtime_node_state(self) -> Dict[str, int]:
        before = {
            "bg": int(len(getattr(self.runtime, "node_states_bg", {}) or {})),
            "distant": int(len(getattr(self.runtime, "node_states_distant", {}) or {})),
            "rigid": int(len(getattr(self.runtime, "node_states_rigid", {}) or {})),
            "sky": int(len(getattr(self.runtime, "node_states_sky", {}) or {})),
        }
        reset = getattr(self.runtime, "reset_node_state", None)
        if callable(reset):
            reset()
        else:
            for name in ("node_states_bg", "node_states_distant", "node_states_rigid", "node_states_sky"):
                cache = getattr(self.runtime, name, None)
                if hasattr(cache, "clear"):
                    cache.clear()
            for name in ("h_cache_bg", "h_cache_distant", "h_cache_rigid", "h_cache_sky"):
                cache = getattr(self.runtime, name, None)
                if hasattr(cache, "clear"):
                    cache.clear()
        after = {
            "bg": int(len(getattr(self.runtime, "node_states_bg", {}) or {})),
            "distant": int(len(getattr(self.runtime, "node_states_distant", {}) or {})),
            "rigid": int(len(getattr(self.runtime, "node_states_rigid", {}) or {})),
            "sky": int(len(getattr(self.runtime, "node_states_sky", {}) or {})),
        }
        return {f"before_{k}": int(v) for k, v in before.items()} | {f"after_{k}": int(v) for k, v in after.items()}
