"""
Stage 5.3 multi-scene training entry for V4 dataset + V8 scheduler.
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage5_3 import MinimalStreetForwardStage5_3
import tools.train_minimal_streetforward_stage4_3_multi_scene_v8 as base
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
    resolve_fixed_scene_segment_v8,
)


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml",
            ]
        )
    base.base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.base.build_train_scheduler_from_cfg = build_train_scheduler_v8_from_cfg
    base.base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v8
    base.base.TRAINER_CLASS = MinimalStreetForwardStage5_3
    base.base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage5_3
    base.base.CKPT_PREFIX = "minimal_sf_stage5_3_multi_scene_v8"
    base.base.DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml"
    base.main()


if __name__ == "__main__":
    main()

