"""
Stage 4.5 multi-scene training entry for V4 dataset + V7 scheduler.

Thin wrapper over Stage 4.3 multi-scene v7 entry:
- swap trainer class to MinimalStreetForwardStage4_5
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5
import tools.train_minimal_streetforward_stage4_3_multi_scene_v7 as base


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_5_multi_scene_v7.yaml",
            ]
        )
    base.base.TRAINER_CLASS = MinimalStreetForwardStage4_5
    base.base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_5
    base.base.CKPT_PREFIX = "minimal_sf_stage4_5_multi_scene_v7"
    base.base.DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_5_multi_scene_v7.yaml"
    base.main()


if __name__ == "__main__":
    main()
