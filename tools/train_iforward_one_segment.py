"""IForward production trainer entry restricted to one fixed scene segment.

This keeps the regular IForward dataset, scheduler, trainer, hooks, checkpointing,
and metrics path.  The only behavioral difference from ``train_iforward.py`` is
that a fixed traversal is allowed (and required), which is useful for bounded
smoke/overfit runs.
"""

from __future__ import annotations

import sys

import tools.train_iforward as iforward


def main() -> None:
    default_config = "configs/iforward/iforward_stage3_3_observation_feedback.yaml"
    if not any(arg == "--config_file" or arg.startswith("--config_file=") for arg in sys.argv):
        sys.argv.extend(["--config_file", default_config])
    if iforward._route_random_window_entrypoint_if_needed(default_config):
        return

    base = iforward.base
    base.build_multi_scene_dataset_v3 = iforward.build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = iforward.build_train_scheduler_iforward_from_cfg
    base.resolve_fixed_scene_segment = iforward.resolve_fixed_scene_segment_iforward
    base.TRAINER_CLASS = iforward.build_iforward_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = iforward.build_iforward_trainer_from_cfg
    base.RUN_START_HOOK = iforward._iforward_run_start_hook
    base.TRAIN_START_HOOK = iforward._iforward_train_start_hook
    base.STEP_END_HOOK = iforward._iforward_step_end_hook
    base.CKPT_PREFIX = "iforward_v1"
    base.CHECKPOINT_PREFIX_RESOLVER = iforward.checkpoint_prefix_iforward_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.ALLOW_ONE_SEGMENT = True
    base.ALLOW_OPTIONAL_ONE_SEGMENT = False
    base.main()


if __name__ == "__main__":
    main()
