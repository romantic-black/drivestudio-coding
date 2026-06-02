"""
IForward multi-scene training entry.

Uses the existing V4 dataset materializer and scheduler_iforward batch contract,
but builds an independent IForward trainer.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Tuple

import torch

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from models.iforward import IForwardTrainer
from tools.train_minimal_streetforward_stage4_3_iforward_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_from_cfg,
    resolve_fixed_scene_segment_iforward,
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


def build_iforward_trainer_from_cfg(config: Any, device: torch.device) -> IForwardTrainer:
    return IForwardTrainer(config=config, device=device)


def checkpoint_prefix_iforward_from_cfg(cfg: Any) -> str:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    version = str(_cfg_get(iforward_cfg, "version", "v1"))
    return f"iforward_{version}"


def main() -> None:
    default_config = "configs/iforward/iforward_base.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(["--config_file", default_config])
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_iforward_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_iforward
    base.TRAINER_CLASS = build_iforward_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = build_iforward_trainer_from_cfg
    base.CKPT_PREFIX = "iforward_v1"
    base.CHECKPOINT_PREFIX_RESOLVER = checkpoint_prefix_iforward_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.ALLOW_ONE_SEGMENT = False
    base.main()


if __name__ == "__main__":
    main()
