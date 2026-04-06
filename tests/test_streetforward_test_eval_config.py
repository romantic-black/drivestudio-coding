from __future__ import annotations

import os

import torch
from omegaconf import OmegaConf

from tools.streetforward_test_config import validate_test_config
from tools.streetforward_test_export import save_3dgs_ply


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "eval_scene_ids": [1, 2],
                "pixel_source": {
                    "test_image_stride": 8,
                    "max_test_images": 4,
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
                    "require_nonzero_test_stride": True,
                    "require_nonempty_test_views": True,
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
    assert out["pixel_test_image_stride"] == 8
    assert out["pixel_max_test_images"] == 4


def test_validate_test_config_fails_for_zero_stride() -> None:
    cfg = _make_cfg()
    cfg.data.pixel_source.test_image_stride = 0
    try:
        validate_test_config(cfg)
    except ValueError as exc:
        assert "test_image_stride" in str(exc)
        return
    raise AssertionError("validate_test_config must fail when test_image_stride == 0")


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
