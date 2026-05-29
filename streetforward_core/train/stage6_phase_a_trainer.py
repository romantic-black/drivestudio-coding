from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.stage6_0 import PHASE_A_NAME, PHASE_B_LONG_NAME, PHASE_B_NAME
from streetforward_core.legacy.stage6_facade import Stage6LegacyFacade
from streetforward_core.recipes.phase_a_recipe import PhaseARecipe
from streetforward_core.train.runner import PhaseATrainRunner


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


class Stage6PhaseAFacadeTrainer:
    """Legacy facade parity trainer for Phase A.

    This wrapper does not inherit Stage5_4, but its internal runtime is still
    MinimalStreetForwardStage6_0 for checkpoint, optimizer, renderer, and
    measurement parity.
    """

    def __init__(self, config: Any, device: torch.device, **kwargs: Any):
        self.runtime = MinimalStreetForwardStage6_0(config=config, device=device, **kwargs)
        self.config = config
        self.device = device
        model_cfg = _cfg_get(config, "model", {}) or {}
        self.stage6_phase = str(_cfg_get(model_cfg, "phase", PHASE_A_NAME))
        self.facade: Optional[Stage6LegacyFacade] = None
        self.recipe: Optional[PhaseARecipe] = None
        self.runner: Optional[PhaseATrainRunner] = None
        if self.stage6_phase == PHASE_A_NAME:
            self.facade = Stage6LegacyFacade(self.runtime)
            self.recipe = PhaseARecipe(facade=self.facade)
            self.runner = PhaseATrainRunner(runtime=self.runtime, recipe=self.recipe)

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

    def train(self, mode: bool = True) -> "Stage6PhaseAFacadeTrainer":
        self.runtime.train(mode)
        if self.recipe is not None:
            self.recipe.train(mode)
        return self

    def eval(self) -> "Stage6PhaseAFacadeTrainer":
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
        if self.stage6_phase in {PHASE_B_NAME, PHASE_B_LONG_NAME}:
            return self.runtime.forward(batch)
        if self.recipe is None:
            raise RuntimeError("Stage6PhaseAFacadeTrainer Phase A recipe was not initialized.")
        return self.recipe(batch)

    def forward(self, batch: Dict[str, Any]) -> Any:
        out = self.forward_recipe(batch)
        if self.stage6_phase in {PHASE_B_NAME, PHASE_B_LONG_NAME}:
            return out
        return out.to_legacy_dict()

    def __call__(self, batch: Dict[str, Any]) -> Any:
        return self.forward(batch)

    def validate_v9_phase_a(self, *args: Any, **kwargs: Any) -> Any:
        """Legacy validation path; recipe-backed validation is not implemented yet."""

        return self.runtime.validate_v9_phase_a(*args, **kwargs)

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if self.stage6_phase in {PHASE_B_NAME, PHASE_B_LONG_NAME}:
            return self.runtime.train_step(
                batch=batch,
                step=step,
                profile_phase_timing=profile_phase_timing,
                sync_cuda_timing=sync_cuda_timing,
                scheduler_node_sync=scheduler_node_sync,
                runtime_policy=runtime_policy,
            )
        _ = (profile_phase_timing, sync_cuda_timing, runtime_policy)
        if self.runner is None:
            raise RuntimeError("Stage6PhaseAFacadeTrainer Phase A runner was not initialized.")
        return self.runner.train_step(batch=batch, step=step, scheduler_node_sync=scheduler_node_sync)


Stage6PhaseATrainer = Stage6PhaseAFacadeTrainer


__all__ = ["Stage6PhaseAFacadeTrainer", "Stage6PhaseATrainer"]
