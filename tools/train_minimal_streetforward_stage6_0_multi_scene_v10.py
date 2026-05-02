"""
Stage 6.0 multi-scene training entry for V4 dataset + V10 scheduler.
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from omegaconf import OmegaConf
import tools.train_minimal_streetforward_stage4_3_multi_scene_v8 as base
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v10_from_cfg,
    resolve_fixed_scene_segment_v10,
)


def _resolve_config_file_from_argv(default_path: str) -> str:
    if "--config_file" not in sys.argv:
        return default_path
    idx = sys.argv.index("--config_file")
    if idx + 1 >= len(sys.argv):
        return default_path
    return str(sys.argv[idx + 1])


def _select_stage6_0_trainer(config_path: str):
    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise ValueError("config.model is required.")
    if str(model_cfg.get("stage")) != "6_0":
        raise ValueError("Stage6_0 entry requires model.stage='6_0'.")
    if bool((cfg.get("scheduler_v10") or {}).get("enable", False)) is not True:
        raise ValueError("Stage6_0 entry requires scheduler_v10.enable=true.")
    return MinimalStreetForwardStage6_0


def main() -> None:
    default_config = "configs/minimal_streetforward_stage6_0_multi_scene_v10.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(["--config_file", default_config])
    config_path = _resolve_config_file_from_argv(default_config)
    trainer_cls = _select_stage6_0_trainer(config_path)
    base.base.setup = base._setup_v8
    base.base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.base.build_train_scheduler_from_cfg = build_train_scheduler_v10_from_cfg
    base.base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v10
    base.base.parse_validation_v7_config = base._parse_validation_v8_config_with_context
    base.base.build_validation_episode_specs_v7 = base._build_validation_specs_v8_proxy
    base.base._run_validation_v7_round = base._run_validation_v8_round
    base.base.TRAINER_CLASS = trainer_cls
    base.base.MinimalStreetForwardStage4_3 = trainer_cls
    base.base.CKPT_PREFIX = "minimal_sf_stage6_0_multi_scene_v10"
    base.base.DEFAULT_CONFIG_FILE = default_config
    base.base.main()


if __name__ == "__main__":
    main()

