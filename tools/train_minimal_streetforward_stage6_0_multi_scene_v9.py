"""
Stage6_0 Phase A multi-scene training entry for V4 dataset + V9 scheduler.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.validation_scheduler_v9 import (
    build_validation_plan_v9,
    make_phase_a_eval_rollout_batch,
    materialize_validation_v9_batch,
)
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.validation_v9_runner import aggregate_validation_v9_phase_a_rows
from tools.streetforward_validation_v9_config import ValidationV9Config, parse_validation_v9_config
from tools.train_minimal_streetforward_stage4_3_v9_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v9_from_cfg,
    resolve_fixed_scene_segment_v9,
)


def _scheduler_v9_blocks_per_episode(cfg: Any) -> int:
    sv9 = cfg.get("scheduler_v9") if hasattr(cfg, "get") else None
    if sv9 is None:
        raise ValueError("validation_v9 requires scheduler_v9")
    ep = sv9.get("episode") if hasattr(sv9, "get") else None
    if ep is None:
        raise ValueError("validation_v9 requires scheduler_v9.episode")
    return int(ep["blocks_per_episode"])


def _run_validation_v9_round(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_train_episode_counter: int,
    trigger_step: int,
    metrics_fh: Optional[Any],
    writer: Optional[Any] = None,
) -> None:
    validation_cfg: ValidationV9Config = parse_validation_v9_config(cfg)
    if not bool(validation_cfg.eval_enable):
        return
    if not hasattr(model, "validate_v9_phase_a"):
        raise ValueError("validation_v9 requires model.validate_v9_phase_a")
    blocks_per_episode = _scheduler_v9_blocks_per_episode(cfg)
    plan = build_validation_plan_v9(
        dataset=dataset,
        eval_scene_ids=[int(x) for x in validation_cfg.eval_scene_ids],
        cfg=validation_cfg,
        blocks_per_episode=int(blocks_per_episode),
    )
    if len(plan.block_specs) == 0:
        msg = "validation_v9 enabled but no valid block specs can be built"
        if bool(validation_cfg.fail_fast):
            raise ValueError(msg)
        base.logger.warning(msg)
        return

    base.logger.info(
        "VALIDATION_V9_BEGIN trigger_episode_counter=%s trigger_step=%s num_blocks=%s k_values=%s max_K=%s",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(plan.block_specs)),
        [int(x) for x in plan.k_values],
        int(plan.max_K),
    )
    root_dir = os.path.join(cfg.log_dir, str(validation_cfg.save_dir))
    rows = []
    for block_i, spec in enumerate(plan.block_specs):
        rollout = make_phase_a_eval_rollout_batch(
            spec,
            max_K=int(plan.max_K),
            k_values=[int(x) for x in plan.k_values],
            block_loss_mask=str(validation_cfg.block_loss_mask),
            nearby_loss_mask=str(validation_cfg.nearby_loss_mask),
        )
        raw_batch = materialize_validation_v9_batch(dataset, rollout, include_test=False)
        minimal_batch = base.convert_batch_to_minimal_format(
            raw_batch,
            device,
            num_targets=int(raw_batch["target"]["image"].shape[0]),
            include_source_for_2d=True,
            view_selection=None,
        )
        save_dir = None
        if bool(validation_cfg.save_images) and int(block_i) < int(validation_cfg.max_saved_blocks):
            save_dir = os.path.join(
                root_dir,
                f"step_{int(trigger_step):06d}",
                f"scene_{int(spec.scene_id):03d}_segment_{int(spec.segment_id):03d}_block_{int(spec.block_idx):02d}",
            )
        row = model.validate_v9_phase_a(
            minimal_batch,
            k_values=[int(x) for x in plan.k_values],
            max_K=int(plan.max_K),
            mask_cfg={
                "block_loss_mask": str(validation_cfg.block_loss_mask),
                "nearby_loss_mask": str(validation_cfg.nearby_loss_mask),
                "min_valid_pixels": int(validation_cfg.min_valid_pixels),
            },
            compute_delta_stats=bool(validation_cfg.compute_delta_stats),
            compute_runtime_stats=bool(validation_cfg.compute_runtime_stats),
            compute_memory_stats=bool(validation_cfg.compute_memory_stats),
            save_images=bool(save_dir),
            save_dir=save_dir,
            save_image_k_values=[int(x) for x in validation_cfg.save_image_k_values],
            max_saved_cams=int(validation_cfg.max_saved_cams),
        )
        row.update(
            {
                "split": "validation_v9_block",
                "trigger_train_episode_counter": int(trigger_train_episode_counter),
                "trigger_step": int(trigger_step),
                "scene_id": int(spec.scene_id),
                "segment_id": int(spec.segment_id),
                "block_idx": int(spec.block_idx),
                "source_frame_idx": int(spec.source_frame_idx),
            }
        )
        rows.append(row)
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, row)
        base.logger.info(
            "VALIDATION_V9_BLOCK scene_id=%s segment_id=%s block=%s source_frame=%s block_psnr@%s=%.4f nearby_psnr@%s=%.4f",
            int(spec.scene_id),
            int(spec.segment_id),
            int(spec.block_idx),
            int(spec.source_frame_idx),
            int(plan.max_K),
            float(row.get(f"block_psnr@{int(plan.max_K)}", 0.0)),
            int(plan.max_K),
            float(row.get(f"nearby_psnr@{int(plan.max_K)}", 0.0)),
        )
        del minimal_batch, raw_batch
    summary: Dict[str, Any] = aggregate_validation_v9_phase_a_rows(rows, k_values=[int(x) for x in plan.k_values])
    summary.update(
        {
            "split": "validation_v9_global",
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "trigger_step": int(trigger_step),
        }
    )
    if metrics_fh is not None:
        base._write_metrics_history(metrics_fh, summary)
    if writer is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(str(key), float(value), int(trigger_step))
    base.logger.info(
        "VALIDATION_V9_END trigger_episode_counter=%s trigger_step=%s num_blocks=%s mean_block_psnr@%s=%.4f mean_nearby_psnr@%s=%.4f",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(rows)),
        int(plan.max_K),
        float(summary.get(f"val_v9/phaseA/mean_block_psnr@{int(plan.max_K)}", 0.0)),
        int(plan.max_K),
        float(summary.get(f"val_v9/phaseA/mean_nearby_psnr@{int(plan.max_K)}", 0.0)),
    )


def _validation_v9_episode_end_hook(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_train_episode_counter: int,
    trigger_step: int,
    metrics_fh: Optional[Any],
    writer: Optional[Any] = None,
    **_: Any,
) -> None:
    validation_cfg = parse_validation_v9_config(cfg)
    if not bool(validation_cfg.eval_enable):
        return
    if int(trigger_train_episode_counter) % int(validation_cfg.validate_every_n_episodes) != 0:
        return
    _run_validation_v9_round(
        cfg=cfg,
        dataset=dataset,
        model=model,
        device=device,
        trigger_train_episode_counter=int(trigger_train_episode_counter),
        trigger_step=int(trigger_step),
        metrics_fh=metrics_fh,
        writer=writer,
    )


def _validation_v9_train_start_hook(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_train_episode_counter: int,
    trigger_step: int,
    metrics_fh: Optional[Any],
    writer: Optional[Any] = None,
    **_: Any,
) -> None:
    validation_cfg = parse_validation_v9_config(cfg)
    if not bool(validation_cfg.eval_enable) or not bool(validation_cfg.run_at_train_start):
        return
    _run_validation_v9_round(
        cfg=cfg,
        dataset=dataset,
        model=model,
        device=device,
        trigger_train_episode_counter=int(trigger_train_episode_counter),
        trigger_step=int(trigger_step),
        metrics_fh=metrics_fh,
        writer=writer,
    )


def main() -> None:
    default_config = "configs/stage6_0_phase_a.yaml"
    if "--config_file" not in sys.argv:
        sys.argv.extend(["--config_file", default_config])
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v9_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v9
    base.EPISODE_END_HOOK = _validation_v9_episode_end_hook
    base.TRAIN_START_HOOK = _validation_v9_train_start_hook
    base.TRAINER_CLASS = MinimalStreetForwardStage6_0
    base.MinimalStreetForwardStage4_3 = MinimalStreetForwardStage6_0
    base.CKPT_PREFIX = "minimal_sf_stage6_0_phase_a_v9"
    base.DEFAULT_CONFIG_FILE = default_config
    base.main()


if __name__ == "__main__":
    main()
