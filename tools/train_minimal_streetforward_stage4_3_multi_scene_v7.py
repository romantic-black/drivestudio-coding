"""
Stage 4.3 multi-scene training entry for V4 dataset + V7 scheduler.

Thin wrapper over the stable multi-scene v4 training loop:
- swap dataset builder to MultiSceneDatasetV4
- swap scheduler builder to TrainSchedulerV7
"""

from __future__ import annotations

import sys

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.train_minimal_streetforward_stage4_3_v7_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v7_from_cfg,
    resolve_fixed_scene_segment_v7,
)


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_multi_scene_v7.yaml",
            ]
        )
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v7_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v7
    if getattr(base, "TRAINER_CLASS", None) is None or getattr(base.TRAINER_CLASS, "__name__", "") == "MinimalStreetForwardStage4_3":
        base.TRAINER_CLASS = MinimalStreetForwardStage4_3
    if str(getattr(base, "CKPT_PREFIX", "")) == "minimal_sf_stage4_3_multi_scene_v4":
        base.CKPT_PREFIX = "minimal_sf_stage4_3_multi_scene_v7"
    base.main()


if __name__ == "__main__":
    main()
