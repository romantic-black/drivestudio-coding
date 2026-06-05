"""
IForward-v6 random-window persistent training entry.

This entrypoint wires only the random-window persistent scheduler path.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.iforward_random_window_validation import (
    random_window_validation_cfg,
    write_random_window_validation_rows,
)
from models.iforward import IForwardTrainer
from tools.train_minimal_streetforward_stage4_3_iforward_random_window_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_random_window_from_cfg,
    resolve_fixed_scene_segment_iforward_random_window,
)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def build_iforward_random_window_trainer_from_cfg(config: Any, device: torch.device) -> IForwardTrainer:
    return IForwardTrainer(config=config, device=device)


def checkpoint_prefix_iforward_random_window_from_cfg(cfg: Any) -> str:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    version = str(_cfg_get(iforward_cfg, "version", "v6_point_mamba_xcpe"))
    return f"iforward_random_window_{version}"


def _train_start_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    val_cfg = random_window_validation_cfg(cfg)
    if bool(val_cfg["enable"]) and bool(val_cfg["run_at_train_start"]):
        write_random_window_validation_rows(
            **kwargs,
            convert_batch_to_minimal_format=base.convert_batch_to_minimal_format,
            write_metrics_history=base._write_metrics_history,
        )


def _step_end_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    val_cfg = random_window_validation_cfg(cfg)
    interval = int(val_cfg["interval_steps"])
    step = int(kwargs.get("trigger_step", 0))
    if not bool(val_cfg["enable"]) or interval <= 0:
        return
    if step < 0 or (step + 1) % int(interval) != 0:
        return
    write_random_window_validation_rows(
        **kwargs,
        convert_batch_to_minimal_format=base.convert_batch_to_minimal_format,
        write_metrics_history=base._write_metrics_history,
    )


def main() -> None:
    default_config = "configs/iforward/iforward_v6_random_window.yaml"
    if not any(arg == "--config_file" or arg.startswith("--config_file=") for arg in sys.argv):
        sys.argv.extend(["--config_file", default_config])
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_iforward_random_window_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_iforward_random_window
    base.TRAINER_CLASS = build_iforward_random_window_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = build_iforward_random_window_trainer_from_cfg
    base.TRAIN_START_HOOK = _train_start_hook
    base.STEP_END_HOOK = _step_end_hook
    base.CKPT_PREFIX = "iforward_random_window_v1"
    base.CHECKPOINT_PREFIX_RESOLVER = checkpoint_prefix_iforward_random_window_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.ALLOW_ONE_SEGMENT = False
    base.ALLOW_OPTIONAL_ONE_SEGMENT = True
    base.main()


if __name__ == "__main__":
    main()
