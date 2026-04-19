"""
Stage 4.6 multi-scene training entry for V4 dataset + V7 scheduler.

Thin wrapper over Stage 4.3 multi-scene v7 entry:
- swap trainer class to MinimalStreetForwardStage4_6
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage4_6 import MinimalStreetForwardStage4_6
import tools.train_minimal_streetforward_stage4_3_multi_scene_v7 as base
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
                "configs/minimal_streetforward_stage4_6_multi_scene_v7.yaml",
            ]
        )
    # Explicitly pin V4 dataset + V7 scheduler binding in the shared v4 loop.
    base.base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.base.build_train_scheduler_from_cfg = build_train_scheduler_v7_from_cfg
    base.base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v7
    base.base.TRAINER_CLASS = MinimalStreetForwardStage4_6
    base.base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_6
    base.base.CKPT_PREFIX = "minimal_sf_stage4_6_multi_scene_v7"
    base.base.DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_6_multi_scene_v7.yaml"
    base.main()


if __name__ == "__main__":
    main()
