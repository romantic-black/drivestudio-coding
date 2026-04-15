"""
Stage 4.4 multi-scene training entry for V4 dataset + V7 scheduler.

Log location can be configured in YAML via:
- logging.log_dir (absolute path, highest priority), or
- logging.output_root + logging.project + output_name
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage4_4 import MinimalStreetForwardStage4_4
import tools.train_minimal_streetforward_stage4_3_multi_scene_v7 as base


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_multi_scene_v7.yaml",
            ]
        )
    base.base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_4
    base.base.CKPT_PREFIX = "minimal_sf_stage4_4_multi_scene_v7"
    base.main()


if __name__ == "__main__":
    main()

