from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from streetforward_core.legacy.stage6_facade import Stage6LegacyFacade
from streetforward_core.protocols.phase_b_long import PHASE_B_LONG_NAME
from streetforward_core.recipes.phase_b_long_recipe import PhaseBLongRecipe
from streetforward_core.train.phase_b_long_runner import PhaseBLongTrainRunner


class Stage6PhaseBLongFacadeTrainer:
    """Legacy facade trainer for the 6_0 Phase B Long mainline."""

    def __init__(self, config: Any, device: torch.device, **kwargs: Any):
        self.runtime = MinimalStreetForwardStage6_0(config=config, device=device, **kwargs)
        self.config = config
        self.device = device
        self.stage6_phase = PHASE_B_LONG_NAME
        self.facade = Stage6LegacyFacade(self.runtime)
        self.recipe = PhaseBLongRecipe(facade=self.facade)
        self.runner = PhaseBLongTrainRunner(runtime=self.runtime, recipe=self.recipe)

    @property
    def optimizer(self) -> Any:
        return getattr(self.runtime, "optimizer", None)

    @optimizer.setter
    def optimizer(self, value: Any) -> None:
        self.runtime.optimizer = value

    @property
    def training(self) -> bool:
        return bool(getattr(self.runtime, "training", False))

    def __getattr__(self, name: str) -> Any:
        runtime = self.__dict__.get("runtime")
        if runtime is not None:
            return getattr(runtime, name)
        raise AttributeError(name)

    def train(self, mode: bool = True) -> "Stage6PhaseBLongFacadeTrainer":
        self.runtime.train(mode)
        self.recipe.train(mode)
        return self

    def eval(self) -> "Stage6PhaseBLongFacadeTrainer":
        return self.train(False)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.runtime.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.load_state_dict(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.named_parameters(*args, **kwargs)

    def forward_recipe(self, batch: Dict[str, Any]) -> Any:
        return self.recipe(batch)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward_recipe(batch).to_legacy_dict()

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.forward(batch)

    def validate_long_phase_b(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.validate_long_phase_b(*args, **kwargs)

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        _ = (step, profile_phase_timing, sync_cuda_timing, runtime_policy)
        return self.runner.train_step(batch=batch, scheduler_node_sync=scheduler_node_sync)


__all__ = ["Stage6PhaseBLongFacadeTrainer"]

