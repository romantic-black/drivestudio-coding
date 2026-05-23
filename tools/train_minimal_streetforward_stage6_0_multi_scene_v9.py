"""
Stage6_0 Phase A multi-scene training entry for V4 dataset + V9 scheduler.
"""

from __future__ import annotations

import sys

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from tools.train_minimal_streetforward_stage4_3_v9_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v9_from_cfg,
    resolve_fixed_scene_segment_v9,
)


def main() -> None:
    default_config = "configs/stage6_0_phase_a.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(["--config_file", default_config])
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v9_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v9
    base.TRAINER_CLASS = MinimalStreetForwardStage6_0
    base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage6_0
    base.CKPT_PREFIX = "minimal_sf_stage6_0_phase_a_v9"
    base.DEFAULT_CONFIG_FILE = default_config
    base.main()


if __name__ == "__main__":
    main()

