"""
Stage 4.4 one-segment training entry for V4 dataset + V6 scheduler.
"""

from __future__ import annotations

import sys

from models.streetforward.minimal_trainer_stage4_4 import MinimalStreetForwardStage4_4
import tools.train_minimal_streetforward_stage4_3_one_segment_v6 as base


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_one_segment_v6.yaml",
            ]
        )
    base.base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage4_4
    base.base.CKPT_PREFIX = "minimal_sf_stage4_4_one_segment_v6"
    base.main()


if __name__ == "__main__":
    main()

