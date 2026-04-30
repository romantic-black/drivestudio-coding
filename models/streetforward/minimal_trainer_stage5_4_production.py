from __future__ import annotations

import torch

from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4


class MinimalStreetForwardStage5_4_Production(
    MinimalStreetForwardStage5_3_Production,
    MinimalStreetForwardStage5_4,
):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_production_config(config)
        MinimalStreetForwardStage5_4.__init__(self, config=config, device=device, **kwargs)
        self._bound_dataset = None
        self._log_optimizer_groups_once()

    def _validate_production_config(self, config) -> None:
        super()._validate_production_config(config)
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model")).strip().lower()
        if stage != "5_4":
            raise ValueError("Stage5_4_Production requires model.stage='5_4'.")
        backprojector_version = str(self._require_key(model_cfg, "backprojector_version", "model")).strip().lower()
        if backprojector_version != "v4":
            raise ValueError("Stage5_4_Production requires model.backprojector_version='v4'.")
        if bool(model_cfg.get("use_fused_cuda_backproject_v4", False)) is not True:
            raise ValueError("Stage5_4_Production requires model.use_fused_cuda_backproject_v4=true.")
        obs_cfg = self._require_key(config, "current_observation", "config")
        if bool(self._require_key(obs_cfg, "enable", "current_observation")) is not True:
            raise ValueError("Stage5_4_Production requires current_observation.enable=true.")


__all__ = ["MinimalStreetForwardStage5_4_Production"]
