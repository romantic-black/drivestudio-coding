from __future__ import annotations

import pytest

from tools.streetforward_validation_v8_config import parse_validation_v8_config


def _base_cfg() -> dict:
    return {
        "data": {
            "eval_scene_ids": [10, 11],
            "pixel_source": {
                "test_image_stride": 0,
                "max_test_images": 0,
            },
        },
        "multi_scene": {"include_test": False},
        "eval": {"run_test_at_end": False},
        "validation_v8": {
            "eval_enable": True,
            "trigger": {
                "by": "train_episode_interval",
                "validate_every_n_episodes": 20,
                "run_at_train_start": True,
            },
            "mode": "inference_only",
            "block": {"steps_per_block": 1},
            "episode": {
                "blocks_per_episode": 3,
                "total_target_frames": 3,
                "target_policy": "visited_episode_frames",
            },
            "execution": {"block_order": "block_major", "reset_policy": "block_end"},
            "episode_selection": {"policy": "middle"},
            "render": {"save_images": True, "save_dir": "validation/episodes"},
            "cache": {"persist_across_training": True},
        },
    }


def test_parse_validation_v8_config_success():
    cfg = parse_validation_v8_config(_base_cfg())
    assert cfg.eval_enable is True
    assert cfg.validate_every_n_episodes == 20
    assert cfg.eval_scene_ids == [10, 11]
    assert cfg.mode == "inference_only"
    assert cfg.steps_per_block == 1
    assert cfg.blocks_per_episode == 3
    assert cfg.total_target_frames == 3
    assert cfg.block_order == "block_major"
    assert cfg.target_policy == "visited_episode_frames"


def test_parse_validation_v8_config_fast_fail_on_target_policy():
    raw = _base_cfg()
    raw["validation_v8"]["episode"]["target_policy"] = "rolling_future"
    with pytest.raises(ValueError, match="target_policy"):
        parse_validation_v8_config(raw)


def test_parse_validation_v8_config_fast_fail_on_target_frames_vs_blocks():
    raw = _base_cfg()
    raw["validation_v8"]["episode"]["total_target_frames"] = 4
    with pytest.raises(ValueError, match="must be <= blocks_per_episode"):
        parse_validation_v8_config(raw)

