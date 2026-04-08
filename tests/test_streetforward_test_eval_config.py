from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from datasets.multi_scene_dataset_v3 import TrainSchedulerV4
from tools.streetforward_test_config import validate_dataset_test_split_or_raise, validate_test_config
from tools.streetforward_test_export import save_3dgs_ply


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "eval_scene_ids": [1, 2],
                "pixel_source": {
                    "test_image_stride": 0,
                    "max_test_images": 0,
                },
            },
            "test": {
                "enable": True,
                "mode": "adapt_supervised",
                "runner": {
                    "deterministic": True,
                    "seed": 123,
                    "source_protocol": "middle_keyframe_middle_frame_cam0",
                    "max_segments_per_scene": 0,
                    "min_test_views_per_segment": 4,
                },
                "split": {
                    "require_eval_scene_ids": True,
                    "require_exhaustive_test_refs": True,
                    "require_nonempty_test_views": True,
                    "allow_train_test_overlap_when_stride_zero": True,
                },
                "adapt_supervised": {
                    "enable": True,
                    "max_steps_per_segment": 10,
                    "validate_every_blocks": 1,
                    "early_stop_patience": 2,
                    "keep_best_by": "psnr",
                    "reset_runtime_state_each_segment": True,
                },
                "inference_only": {
                    "enable": True,
                    "eval_trigger": "episode_end",
                    "aggregate_across_episodes": "mean",
                    "max_episodes_per_segment": 0,
                    "save_per_episode_metrics_json": True,
                    "save_per_episode_per_view_metrics_json": False,
                    "allow_hidden_cache_update": False,
                    "allow_node_state_writeback": False,
                },
                "export": {
                    "save_3dgs_init": True,
                    "save_3dgs_best": True,
                    "save_3dgs_final": True,
                    "save_ply": True,
                    "save_rendered_images": True,
                    "save_per_view_metrics_json": True,
                },
            },
        }
    )


def test_validate_test_config_passes_with_required_fields() -> None:
    cfg = _make_cfg()
    out = validate_test_config(cfg)
    assert out["mode"] == "adapt_supervised"
    assert out["eval_scene_ids"] == [1, 2]
    assert out["pixel_test_image_stride"] == 0
    assert out["pixel_max_test_images"] == 0


def test_validate_test_config_allows_zero_stride_when_overlap_enabled() -> None:
    cfg = _make_cfg()
    out = validate_test_config(cfg)
    assert out["pixel_test_image_stride"] == 0


def test_validate_test_config_fails_for_negative_stride() -> None:
    cfg = _make_cfg()
    cfg.data.pixel_source.test_image_stride = -1
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "test_image_stride" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when test_image_stride < 0")


def test_validate_test_config_fails_for_non_exhaustive_test_cap() -> None:
    cfg = _make_cfg()
    cfg.data.pixel_source.max_test_images = 2
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "max_test_images" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when max_test_images != 0")


def test_validate_test_config_fails_for_stride_zero_overlap_disallowed() -> None:
    cfg = _make_cfg()
    cfg.test.split.allow_train_test_overlap_when_stride_zero = False
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "allow_train_test_overlap_when_stride_zero" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when stride=0 overlap is disallowed")


def test_validate_test_config_fails_when_split_new_field_missing() -> None:
    cfg = _make_cfg()
    del cfg.test.split["require_exhaustive_test_refs"]
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "require_exhaustive_test_refs" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when new split fields are missing")


def test_validate_test_config_fails_for_invalid_inference_eval_trigger() -> None:
    cfg = _make_cfg()
    cfg.test.inference_only.eval_trigger = "block_end"
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "eval_trigger" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when inference_only.eval_trigger is invalid")


def test_validate_test_config_fails_for_invalid_inference_aggregate_mode() -> None:
    cfg = _make_cfg()
    cfg.test.inference_only.aggregate_across_episodes = "median"
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "aggregate_across_episodes" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when aggregate_across_episodes is invalid")


class _FakeDataset:
    def __init__(self, scene_map):
        self._scene_map = scene_map

    def get_scene(self, scene_id: int):
        return self._scene_map.get(int(scene_id))


def test_validate_dataset_test_split_or_raise_fails_for_empty_test_segment_stride_positive() -> None:
    cfg = _make_cfg()
    cfg.data.pixel_source.test_image_stride = 2
    test_cfg = validate_test_config(cfg)
    fake = _FakeDataset(
        {
            1: {"segments": [{"frame_indices": [0, 1], "test_frame_indices": []}]},
            2: {"segments": [{"frame_indices": [0, 1], "test_frame_indices": [2]}]},
        }
    )
    try:
        validate_dataset_test_split_or_raise(fake, test_cfg)
    except ValueError as exc:
        assert "test_frame_indices is empty" in str(exc)
        return
    raise AssertionError("dataset split validation must fail for stride>0 with empty test frames")


def test_validate_dataset_test_split_or_raise_fails_for_empty_train_segment_stride_positive() -> None:
    cfg = _make_cfg()
    cfg.data.pixel_source.test_image_stride = 2
    test_cfg = validate_test_config(cfg)
    fake = _FakeDataset(
        {
            1: {"segments": [{"frame_indices": [], "test_frame_indices": [2]}]},
            2: {"segments": [{"frame_indices": [0, 1], "test_frame_indices": [2]}]},
        }
    )
    try:
        validate_dataset_test_split_or_raise(fake, test_cfg)
    except ValueError as exc:
        assert "train_frame_indices is empty" in str(exc)
        return
    raise AssertionError("dataset split validation must fail for stride>0 with empty train frames")


def test_save_3dgs_ply_requires_rigid_world(tmp_path) -> None:
    n = 2
    branch = {
        "means": torch.zeros((n, 3), dtype=torch.float32),
        "opacity_logit": torch.zeros((n, 1), dtype=torch.float32),
        "sh_dc": torch.zeros((n, 3), dtype=torch.float32),
    }
    state = {
        "branches": {
            "bg": branch,
            "rigid_local": branch,
            "rigid_world": None,
        }
    }
    out = os.path.join(tmp_path, "x.ply")
    try:
        save_3dgs_ply(out, state)
    except ValueError as exc:
        assert "rigid_world" in str(exc)
        return
    raise AssertionError("save_3dgs_ply must fail when rigid_local exists but rigid_world is missing")


def test_train_scheduler_v4_end_segment_emits_episode_end_then_segment_end() -> None:
    sched = object.__new__(TrainSchedulerV4)
    sched.epoch_idx = 3
    sched.global_step = 12
    sched._reset_episode_idx = 2
    sched._pending_events = []
    sched.plan_cursor = 0
    sched.dataset = SimpleNamespace()
    sched.current_segment_state = {
        "scene_id": 1,
        "segment_id": 4,
        "segment_local_u": 8,
        "source_image_ref": (10, 0),
        "episodes_started": 1,
        "pair_list": [(1, 0), (2, 0)],
        "episode_window_keyframes": [5, 6],
    }

    TrainSchedulerV4._end_segment(sched)
    events = list(sched._pending_events)
    assert [e["type"] for e in events[:2]] == ["episode_end", "segment_end"]
    assert int(events[0]["reset_episode_idx"]) == 2
    assert sched.current_segment_state is None
