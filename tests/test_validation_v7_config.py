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

