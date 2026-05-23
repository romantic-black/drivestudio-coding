from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "MultiSceneDataset": ("datasets.multi_scene_dataset", "MultiSceneDataset"),
    "MultiSceneDatasetV2": ("datasets.multi_scene_dataset_v2", "MultiSceneDatasetV2"),
    "MultiSceneDatasetV3": ("datasets.multi_scene_dataset_v3", "MultiSceneDatasetV3"),
    "BatchRequestV3": ("datasets.multi_scene_dataset_v3", "BatchRequestV3"),
    "BatchRequestV4": ("datasets.multi_scene_dataset_v4", "BatchRequestV4"),
    "SegmentIndex": ("datasets.multi_scene_dataset_v3", "SegmentIndex"),
    "SegmentIndexV4": ("datasets.multi_scene_dataset_v4", "SegmentIndexV4"),
    "TrainSchedulerV4": ("datasets.multi_scene_dataset_v3", "TrainSchedulerV4"),
    "TrainSchedulerV6": ("datasets.train_scheduler_v6", "TrainSchedulerV6"),
    "TrainSchedulerV7": ("datasets.train_scheduler_v7", "TrainSchedulerV7"),
    "TrainSchedulerV8": ("datasets.train_scheduler_v8", "TrainSchedulerV8"),
    "TrainSchedulerV9": ("datasets.train_scheduler_v9", "TrainSchedulerV9"),
    "BatchRequestV9": ("datasets.train_scheduler_v9", "BatchRequestV9"),
    "MultiSceneDatasetV4": ("datasets.multi_scene_dataset_v4", "MultiSceneDatasetV4"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    entry = _LAZY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module 'datasets' has no attribute {name!r}")
    module_name, attr_name = entry
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
