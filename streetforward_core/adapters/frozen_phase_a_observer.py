from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from models.streetforward.stage6_0.local_gs_state import LocalGSState
from streetforward_core.legacy.stage6_facade import (
    PhaseAEventBuilderAdapter,
    Stage5V4MeasurementAdapter,
    Stage6LegacyFacade,
)


class FrozenPhaseAObserver:
    """Frozen Phase A observation/event adapter for Phase B Long."""

    def __init__(
        self,
        facade: Stage6LegacyFacade,
        *,
        measurement: Optional[Stage5V4MeasurementAdapter] = None,
        event_builder: Optional[PhaseAEventBuilderAdapter] = None,
    ):
        self.facade = facade
        self.measurement = measurement if measurement is not None else Stage5V4MeasurementAdapter(facade)
        self.event_builder = event_builder if event_builder is not None else PhaseAEventBuilderAdapter(facade)

    def observe_event(
        self,
        *,
        sensor_state: LocalGSState,
        batch: Dict[str, Any],
        source_indices: List[int],
        source_frame_idx: int,
    ) -> Any:
        with torch.no_grad():
            measurement = self.measurement.observe(
                local_state=sensor_state,
                batch=batch,
                source_indices=[int(x) for x in source_indices],
                source_frame_idx=int(source_frame_idx),
            )
            event = self.event_builder.build_event(local_state=sensor_state, measurement=measurement)
            runtime = self.facade.runtime
            if hasattr(runtime, "_event_with_default_view_code"):
                event = runtime._event_with_default_view_code(event)
            if hasattr(runtime, "_detach_event_pack"):
                event = runtime._detach_event_pack(event)
            else:
                for name, value in list(getattr(event, "__dict__", {}).items()):
                    if torch.is_tensor(value):
                        setattr(event, name, value.detach())
        for name, value in list(getattr(event, "__dict__", {}).items()):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"FrozenPhaseAObserver produced non-finite tensor {name}")
        return event
