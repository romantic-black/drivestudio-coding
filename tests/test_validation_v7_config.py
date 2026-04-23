from __future__ import annotations

import pytest

from tools.streetforward_validation_v7_config import parse_validation_v7_config


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
        "validation_v7": {
            "eval_enable": True,
            "trigger": {
                "by": "train_episode_interval",
                "validate_every_n_episodes": 20,
                "run_at_train_start": True,
            },
            "mode": "inference_only",
            "block": {"steps_per_block": 1},
            "episode": {"blocks_per_episode": 3},
            "execution": {"block_order": "block_major"},
            "episode_selection": {"policy": "middle"},
            "render": {"save_images": True, "save_dir": "validation/episodes"},
            "cache": {"persist_across_training": True},
        },
    }


def test_parse_validation_v7_config_success():
    cfg = parse_validation_v7_config(_base_cfg())
    assert cfg.eval_enable is True
    assert cfg.validate_every_n_episodes == 20
    assert cfg.eval_scene_ids == [10, 11]
    assert cfg.mode == "inference_only"
    assert cfg.steps_per_block == 1
    assert cfg.blocks_per_episode == 3
    assert cfg.block_order == "block_major"
    assert cfg.reset_policy == "block_end"
    assert cfg.use_sky_mask_regions is False
    assert cfg.min_valid_pixels_per_region == 32
    assert cfg.require_sky_mask is False


def test_parse_validation_v7_config_fast_fail_on_legacy_fields():
    raw = _base_cfg()
    raw["data"]["pixel_source"]["test_image_stride"] = 5
    with pytest.raises(ValueError, match="legacy test/eval split"):
        parse_validation_v7_config(raw)


def test_parse_validation_v7_config_disabled_defaults():
    raw = _base_cfg()
    raw["validation_v7"]["eval_enable"] = False
    cfg = parse_validation_v7_config(raw)
    assert cfg.eval_enable is False
    assert cfg.validate_every_n_episodes == 0
    assert cfg.mode == "inference_only"
    assert cfg.steps_per_block == 1
    assert cfg.blocks_per_episode is None
    assert cfg.block_order == "block_major"
    assert cfg.reset_policy == "block_end"
    assert cfg.use_sky_mask_regions is False
    assert cfg.min_valid_pixels_per_region == 32
    assert cfg.require_sky_mask is False


def test_parse_validation_v7_config_metrics_block_success():
    raw = _base_cfg()
    raw["validation_v7"]["metrics"] = {
        "use_sky_mask_regions": True,
        "min_valid_pixels_per_region": 64,
        "require_sky_mask": True,
    }
    cfg = parse_validation_v7_config(raw)
    assert cfg.use_sky_mask_regions is True
    assert cfg.min_valid_pixels_per_region == 64
    assert cfg.require_sky_mask is True


def test_parse_validation_v7_config_metrics_fast_fail_invalid():
    raw = _base_cfg()
    raw["validation_v7"]["metrics"] = {
        "use_sky_mask_regions": False,
        "min_valid_pixels_per_region": 0,
        "require_sky_mask": False,
    }
    with pytest.raises(ValueError, match="min_valid_pixels_per_region"):
        parse_validation_v7_config(raw)

    raw2 = _base_cfg()
    raw2["validation_v7"]["metrics"] = {
        "use_sky_mask_regions": False,
        "min_valid_pixels_per_region": 16,
        "require_sky_mask": True,
    }
    with pytest.raises(ValueError, match="requires validation_v7.metrics.use_sky_mask_regions=true"):
        parse_validation_v7_config(raw2)


def test_parse_validation_v7_config_train_mode_success():
    raw = _base_cfg()
    raw["validation_v7"]["mode"] = "segment_finetune_train"
    raw["validation_v7"]["block"]["steps_per_block"] = 4
    raw["validation_v7"]["episode"]["blocks_per_episode"] = 5
    raw["validation_v7"]["execution"]["block_order"] = "step_major"
    cfg = parse_validation_v7_config(raw)
    assert cfg.mode == "segment_finetune_train"
    assert cfg.steps_per_block == 4
    assert cfg.blocks_per_episode == 5
    assert cfg.block_order == "step_major"
    assert cfg.reset_policy == "episode_end"


def test_parse_validation_v7_config_mode_fast_fail_invalid():
    raw = _base_cfg()
    raw["validation_v7"]["mode"] = "train_only"
    with pytest.raises(ValueError, match="validation_v7.mode"):
        parse_validation_v7_config(raw)


def test_parse_validation_v7_config_block_fast_fail_invalid():
    raw = _base_cfg()
    raw["validation_v7"]["block"]["steps_per_block"] = 0
    with pytest.raises(ValueError, match="validation_v7.block.steps_per_block"):
        parse_validation_v7_config(raw)


def test_parse_validation_v7_config_episode_fast_fail_invalid():
    raw = _base_cfg()
    raw["validation_v7"]["episode"]["blocks_per_episode"] = 0
    with pytest.raises(ValueError, match="validation_v7.episode.blocks_per_episode"):
        parse_validation_v7_config(raw)


def test_parse_validation_v7_config_execution_fast_fail_invalid():
    raw = _base_cfg()
    raw["validation_v7"]["execution"]["block_order"] = "random"
    with pytest.raises(ValueError, match="validation_v7.execution.block_order"):
        parse_validation_v7_config(raw)

    raw2 = _base_cfg()
    raw2["validation_v7"]["execution"]["reset_policy"] = "unknown"
    with pytest.raises(ValueError, match="validation_v7.execution.reset_policy"):
        parse_validation_v7_config(raw2)

    raw3 = _base_cfg()
    raw3["validation_v7"]["execution"]["block_order"] = "step_major"
    raw3["validation_v7"]["execution"]["reset_policy"] = "block_end"
    with pytest.raises(ValueError, match="incompatible with execution.block_order=step_major"):
        parse_validation_v7_config(raw3)
