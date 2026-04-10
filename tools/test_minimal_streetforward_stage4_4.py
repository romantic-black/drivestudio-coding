"""
Stage 4.4 test/eval entry.

Wrapper over stage4_3 test script:
- swaps trainer class to MinimalStreetForwardStage4_4
- routes scheduler builder to common v4/v5 factory
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage4_4 import MinimalStreetForwardStage4_4
import tools.test_minimal_streetforward_stage4_3 as base
from tools.train_minimal_streetforward_stage4_3_v4_common import build_train_scheduler_from_cfg


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_multi_scene_v5.yaml",
            ]
        )
    base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_4
    base.build_train_scheduler_v4_from_cfg = build_train_scheduler_from_cfg
    base.main()


if __name__ == "__main__":
    main()
