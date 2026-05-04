from __future__ import annotations

import torch

from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from models.streetforward.minimal_trainer_stage5_6 import MinimalStreetForwardStage5_6


class MinimalStreetForwardStage5_6_Production(
    MinimalStreetForwardStage5_3_Production,
    MinimalStreetForwardStage5_6,
):
    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config=config, device=device, **kwargs)
        if not hasattr(self, "_bound_dataset"):
            self._bound_dataset = None
        self._debug_check_stage5_6_optimizer_contains_new_modules()
        self._log_optimizer_groups_once()

    def _validate_production_config(self, config) -> None:
        super()._validate_production_config(config)
        # Keep production path in sync with Stage5_6 schema fast-fails.
        MinimalStreetForwardStage5_6._validate_stage5_3_config(self, config)
        model_cfg = self._require_key(config, "model", "config")
        stage = str(self._require_key(model_cfg, "stage", "model")).strip().lower()
        if stage != "5_6":
            raise ValueError("Stage5_6_Production requires model.stage='5_6'.")
        backprojector_version = str(self._require_key(model_cfg, "backprojector_version", "model")).strip().lower()
        if backprojector_version != "v4":
            raise ValueError("Stage5_6_Production requires model.backprojector_version='v4'.")
        if bool(model_cfg.get("use_fused_cuda_backproject_v4", False)) is not True:
            raise ValueError("Stage5_6_Production requires model.use_fused_cuda_backproject_v4=true.")
        obs_cfg = self._require_key(config, "current_observation", "config")
        if bool(self._require_key(obs_cfg, "enable", "current_observation")) is not True:
            raise ValueError("Stage5_6_Production requires current_observation.enable=true.")
        if int(obs_cfg.get("dim", 2)) != 2:
            raise ValueError("Stage5_6_Production requires current_observation.dim=2.")
        if str(obs_cfg.get("rho_source", "feature")).strip().lower() != "feature":
            raise ValueError("Stage5_6_Production requires current_observation.rho_source='feature'.")
        if bool(obs_cfg.get("record_to_history_memory", False)):
            raise ValueError("Stage5_6_Production requires current_observation.record_to_history_memory=false.")

        fsu_cfg = config.get("feature_splat_uncertainty", {}) if hasattr(config, "get") else {}
        bridge_cfg = fsu_cfg.get("bridge", {}) if hasattr(fsu_cfg, "get") else {}
        head_cfg = fsu_cfg.get("head", {}) if hasattr(fsu_cfg, "get") else {}
        loss_cfg = fsu_cfg.get("loss", {}) if hasattr(fsu_cfg, "get") else {}
        if bool(bridge_cfg.get("enable", False)):
            raise ValueError("Stage5_6_Production does not support bridge.enable=true.")
        if bool(head_cfg.get("predict_rgb_residual", False)):
            raise ValueError("Stage5_6_Production does not support predict_rgb_residual=true.")
        if float(loss_cfg.get("rgb_residual_weight", 0.0)) != 0.0:
            raise ValueError("Stage5_6_Production requires rgb_residual_weight=0.0.")
        if float(loss_cfg.get("rgb_residual_supported_weight", 0.0)) != 0.0:
            raise ValueError("Stage5_6_Production requires rgb_residual_supported_weight=0.0.")


__all__ = ["MinimalStreetForwardStage5_6_Production"]
