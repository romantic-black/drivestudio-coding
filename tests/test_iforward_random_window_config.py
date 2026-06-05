from __future__ import annotations

from omegaconf import OmegaConf


def test_iforward_v6_random_window_config_uses_only_new_scheduler_keys():
    cfg = OmegaConf.load("configs/iforward/iforward_v6_random_window.yaml")
    assert cfg.output_name == "iforward_v6_random_window"
    assert "scheduler_iforward_random_window" in cfg
    assert "iforward_random_window_validation" in cfg
    assert "scheduler_iforward" not in cfg
    assert "iforward_validation" not in cfg
    assert cfg.scheduler_iforward_random_window.rollout.blocks_per_rollout == 4
    assert cfg.scheduler_iforward_random_window.rollout.repeats_per_block == 2
    assert cfg.scheduler_iforward_random_window.episode.rollouts_per_episode == 8
    assert cfg.iforward_random_window_validation.rollouts_per_segment == 8
    assert cfg.logging.train_step_metrics_interval == 1
    assert cfg.logging.random_window_diagnostics_interval == 100


def test_random_window_entrypoint_imports_without_legacy_scheduler_reference():
    import tools.train_iforward_random_window as entry

    assert callable(entry.main)


def test_random_window_metrics_history_uses_compact_train_step_schema():
    from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import (
        _build_iforward_random_window_diagnostics_row,
        _build_iforward_random_window_train_step_row,
    )

    result = {
        "iforward/scheduler_version": "random_window_v1",
        "iforward/loss_total": 0.2,
        "iforward/scene_id": 1,
        "iforward/segment_id": 2,
        "iforward/episode_id": 3,
        "iforward/rollout_idx_in_episode": 4.0,
        "iforward/rollouts_per_episode": 8,
        "iforward/window_start": 5,
        "iforward/window_end": 9,
        "iforward/window_hash": 123456,
        "iforward/window_block_ids": [5, 6, 7, 8],
        "iforward/is_repeated_window": True,
        "iforward/loss_current_latest": 0.03,
        "iforward/current_latest_psnr": 24.0,
        "iforward/current_latest_num_refs": 3.0,
        "iforward/in_rollout_history_num_refs": 9.0,
        "iforward/short_window_history_num_refs": 24.0,
        "iforward/nearby_num_refs": 3.0,
        "iforward/optimizer/point_mamba/lr": 1e-4,
        "iforward/grad/point_mamba": 0.5,
    }
    row = _build_iforward_random_window_train_step_row(
        step=7,
        minimal_batch={},
        scheduler_info={"epoch_idx": 0, "global_step": 7},
        step_events=[{"type": "rollout_batch_emitted"}],
        result=result,
        loss_val=0.2,
        num_views=15,
        step_time_ms=10.0,
        batch_fetch_ms=1.0,
        batch_convert_ms=2.0,
    )
    diag = _build_iforward_random_window_diagnostics_row(
        step=7,
        result=result,
        scheduler_info={"global_step": 7},
        diag_row={},
    )

    assert row["split"] == "train_step"
    assert row["scheduler_version"] == "random_window_v1"
    assert row["window_block_ids"] == [5, 6, 7, 8]
    assert row["current_latest_num_refs"] == 3.0
    assert "source_image_ref" not in row
    assert "target_image_refs" not in row
    assert "iforward/optimizer/point_mamba/lr" not in row
    assert diag["split"] == "train_step_diagnostics"
    assert diag["iforward/optimizer/point_mamba/lr"] == 1e-4
