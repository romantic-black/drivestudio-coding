from __future__ import annotations

import pytest

from tools.streetforward_validation_v9_config import parse_validation_v9_config


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
        "validation_v9": {
            "enable": True,
            "phase": "phase_A_block_local_unroll",
            "fail_fast": True,
            "trigger": {
                "by": "train_episode_interval",
                "validate_every_n_episodes": 20,
                "run_at_train_start": True,
            },
            "selection": {
                "seed": 20260524,
                "eval_scene_ids_from_data": True,
                "segments_per_scene": 1,
                "segment_policy": "random_seeded",
                "episode_policy": "random_seeded",
                "blocks_per_segment": 2,
                "block_policy": "random_without_replacement",
                "source_frame_policy": "middle_in_keyframe",
            },
            "phase_A": {
                "k_values": [0, 2, 4],
                "max_K": 4,
                "nearby": {
                    "enable": True,
                    "frames_per_block": 1,
                    "policy": "adjacent_then_random_same_keyframe",
                    "same_keyframe_only": True,
                    "camera_policy": "all_cams",
                },
                "render": {
                    "save_images": True,
                    "save_image_k_values": [0, 4],
                    "max_saved_blocks": 1,
                    "max_saved_cams": 1,
                    "save_dir": "validation_v9/phase_a",
                },
            },
            "masks": {
                "block_loss_mask": "non_sky_non_egocar",
                "nearby_loss_mask": "non_sky_non_egocar",
                "min_valid_pixels": 32,
                "require_sky_mask": True,
                "require_egocar_mask": True,
            },
            "phase_B": {
                "reserved": True,
                "prefix_render": {"enable": False},
                "query_observation": {"enable": False},
            },
        },
    }


def test_parse_validation_v9_config_success():
    cfg = parse_validation_v9_config(_base_cfg())
    assert cfg.eval_enable is True
    assert cfg.eval_scene_ids == [10, 11]
    assert cfg.validate_every_n_episodes == 20
    assert cfg.run_at_train_start is True
    assert cfg.k_values == [0, 2, 4]
    assert cfg.max_K == 4
    assert cfg.blocks_per_segment == 2
    assert cfg.save_images is True


def test_parse_validation_v9_disabled_defaults():
    raw = _base_cfg()
    raw["validation_v9"]["enable"] = False
    cfg = parse_validation_v9_config(raw)
    assert cfg.eval_enable is False
    assert cfg.eval_scene_ids == [10, 11]
    assert cfg.max_K == 16


def test_parse_validation_v9_requires_eval_scene_ids():
    raw = _base_cfg()
    raw["data"]["eval_scene_ids"] = []
    with pytest.raises(ValueError, match="eval_scene_ids"):
        parse_validation_v9_config(raw)


def test_parse_validation_v9_k_values_include_zero():
    raw = _base_cfg()
    raw["validation_v9"]["phase_A"]["k_values"] = [2, 4]
    with pytest.raises(ValueError, match="include 0"):
        parse_validation_v9_config(raw)


def test_parse_validation_v9_max_k_must_match():
    raw = _base_cfg()
    raw["validation_v9"]["phase_A"]["max_K"] = 8
    with pytest.raises(ValueError, match="max_K"):
        parse_validation_v9_config(raw)


def test_parse_validation_v9_segments_per_scene_p0():
    raw = _base_cfg()
    raw["validation_v9"]["selection"]["segments_per_scene"] = 2
    with pytest.raises(ValueError, match="segments_per_scene"):
        parse_validation_v9_config(raw)
