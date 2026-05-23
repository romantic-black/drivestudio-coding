"""
Stage 4.3 multi-scene training entry for V4 dataset + V9 scheduler.

This wrapper keeps the existing trainer loop and only swaps the scheduler/data
builder to the V9 role-isolated batch schema.
"""

from __future__ import annotations

import sys

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.train_minimal_streetforward_stage4_3_v9_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v9_from_cfg,
    resolve_fixed_scene_segment_v9,
)


def main() -> None:
    default_config = "configs/scheduler_v9_phase_a.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(["--config_file", default_config])
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v9_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v9
    base.TRAINER_CLASS = MinimalStreetForwardStage4_3
    base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_3
    base.CKPT_PREFIX = "minimal_sf_stage4_3_multi_scene_v9"
    base.DEFAULT_CONFIG_FILE = default_config
    base.main()


if __name__ == "__main__":
    main()
