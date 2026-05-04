"""
Stage 5.6 multi-scene training entry for V4 dataset + V8 scheduler.
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage5_6 import MinimalStreetForwardStage5_6
from models.streetforward.minimal_trainer_stage5_6_production import MinimalStreetForwardStage5_6_Production
from omegaconf import OmegaConf
import tools.train_minimal_streetforward_stage4_3_multi_scene_v8 as base
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
    resolve_fixed_scene_segment_v8,
)


def _resolve_config_file_from_argv(default_path: str) -> str:
    if "--config_file" not in sys.argv:
        return default_path
    idx = sys.argv.index("--config_file")
    if idx + 1 >= len(sys.argv):
        return default_path
    return str(sys.argv[idx + 1])


def _select_stage5_6_trainer(config_path: str):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("config.model is required.")
    use_production = bool(model_cfg.get("production_training", False))
    if use_production:
        return MinimalStreetForwardStage5_6_Production
    return MinimalStreetForwardStage5_6


def main() -> None:
    default_config = "configs/minimal_streetforward_stage5_6_production_multi_scene_v8.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                default_config,
            ]
        )
    config_path = _resolve_config_file_from_argv(default_config)
    trainer_cls = _select_stage5_6_trainer(config_path)
    base.base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.base.build_train_scheduler_from_cfg = build_train_scheduler_v8_from_cfg
    base.base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v8
    base.base.TRAINER_CLASS = trainer_cls
    base.base.MinimalStreetForwardStage4_3 = trainer_cls
    base.base.CKPT_PREFIX = "minimal_sf_stage5_6_multi_scene_v8"
    base.base.DEFAULT_CONFIG_FILE = default_config
    base.main()


if __name__ == "__main__":
    main()
