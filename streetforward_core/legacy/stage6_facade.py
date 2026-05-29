from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0.local_gs_state import LocalGSState
from models.streetforward.stage6_0.posterior_updater import DeltaPack


class Stage6LegacyFacade:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.runtime, "stage6_hidden_dim"))

    @property
    def block_weight(self) -> float:
        return float(getattr(self.runtime, "stage6_block_weight", 1.0))

    @property
    def step_gamma(self) -> float:
        return float(getattr(self.runtime, "stage6_step_gamma", 1.0))

    @property
    def block_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_block_mask_policy", "non_sky_non_egocar"))

    @property
    def nearby_mask_policy(self) -> str:
        return str(getattr(self.runtime, "stage6_nearby_mask_policy", "non_sky_non_egocar"))

    def nearby_weight(self, *, global_step: int, k: int, K: int) -> float:
        if hasattr(self.runtime, "_nearby_weight"):
            return float(self.runtime._nearby_weight(global_step=int(global_step), k=int(k), K=int(K)))
        if not bool(getattr(self.runtime, "stage6_nearby_enable", True)):
            return 0.0
        if bool(getattr(self.runtime, "stage6_nearby_final_step_only", True)) and int(k) != int(K) - 1:
            return 0.0
        warmup_steps = max(int(getattr(self.runtime, "stage6_nearby_warmup_steps", 1)), 1)
        warm = min(float(global_step) / float(warmup_steps), 1.0)
        return float(getattr(self.runtime, "stage6_nearby_weight", 0.0)) * warm

    def delta_regularization_kwargs(self) -> Dict[str, Any]:
        return {
            "weight": float(getattr(self.runtime, "stage6_delta_l2_weight", 1.0e-3)),
            "opacity_delta_l2_weight": float(getattr(self.runtime, "stage6_opacity_delta_l2_weight", 0.0)),
            "sh_delta_l2_weight": float(getattr(self.runtime, "stage6_sh_delta_l2_weight", 0.0)),
            "scale_barrier_weight": float(getattr(self.runtime, "stage6_scale_barrier_weight", 0.0)),
            "scale_log_min": float(getattr(self.runtime, "stage6_scale_log_min", -10.0)),
            "scale_log_max": float(getattr(self.runtime, "stage6_scale_log_max", 4.0)),
        }


class Stage5NodeStateProviderAdapter:
    def __init__(self, facade: Stage6LegacyFacade):
        self.facade = facade

    def get_or_init(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        return self.facade.runtime._get_or_init_node_states_bg_rigid_distant(batch)


class Stage5V4MeasurementAdapter:
    def __init__(self, facade: Stage6LegacyFacade):
        self.facade = facade

    def observe(
        self,
        *,
        local_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
    ) -> Dict[str, Any]:
        return self.facade.runtime._observe_v4_measurement(
            local_state=local_state,
            batch=batch,
            source_indices=[int(x) for x in source_indices],
            source_frame_idx=int(source_frame_idx),
        )


class PhaseAEventBuilderAdapter:
    def __init__(self, facade: Stage6LegacyFacade):
        self.facade = facade

    def build_event(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Any:
        return self.facade.runtime._build_stage6_event_from_measurement(
            local_state=local_state,
            measurement=measurement,
        )


class PosteriorUpdaterAdapter:
    def __init__(self, facade: Stage6LegacyFacade):
        self.facade = facade

    def apply_update(
        self,
        *,
        local_state: LocalGSState,
        event: Any,
    ) -> Tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        return self.facade.runtime._apply_event_update(local_state=local_state, event=event, ctx_vsm=None)


class Stage6EventUpdateAdapter:
    """Compatibility shim for old tests/callers that still expect encode_and_update."""

    def __init__(self, facade: Stage6LegacyFacade):
        self.event_builder = PhaseAEventBuilderAdapter(facade)
        self.posterior_updater = PosteriorUpdaterAdapter(facade)

    def encode_and_update(
        self,
        *,
        local_state: LocalGSState,
        measurement: Dict[str, Any],
    ) -> Tuple[LocalGSState, DeltaPack, Dict[str, Any]]:
        event = self.event_builder.build_event(local_state=local_state, measurement=measurement)
        return self.posterior_updater.apply_update(local_state=local_state, event=event)


class Stage5RendererAdapter:
    def __init__(self, facade: Stage6LegacyFacade):
        self.facade = facade

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
        return self.facade.runtime._render_loss_for_indices(
            local_state=local_state,
            batch=batch,
            target_indices=[int(x) for x in target_indices],
            mask_policy=str(mask_policy),
            pred_rgbs_out=pred_rgbs_out,
            gt_images_out=gt_images_out,
        )
