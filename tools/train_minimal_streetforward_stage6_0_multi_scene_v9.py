"""
Stage6_0 Phase A multi-scene training entry for V4 dataset + V9 scheduler.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _normalize_omp_num_threads_silent(*, fallback: int = 8) -> None:
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        os.environ["OMP_NUM_THREADS"] = str(fallback)
        return
    if value <= 0:
        os.environ["OMP_NUM_THREADS"] = str(fallback)


_normalize_omp_num_threads_silent()

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.validation_scheduler_v9 import (
    build_validation_plan_v9,
    make_phase_a_eval_rollout_batch,
    materialize_validation_v9_batch,
)
from datasets.validation_long_phase_b import (
    build_validation_plan_long_phase_b,
    materialize_validation_long_phase_b_batch,
)
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.validation_v9_runner import aggregate_validation_v9_phase_a_rows
from streetforward_core.data.schedulers.legacy_v9_phase_a_adapter import LegacyV9PhaseASchedulerAdapter
from streetforward_core.protocols.phase_b_long import PHASE_B_LONG_NAME
from streetforward_core.protocols.rollout import PHASE_A_NAME
from streetforward_core.train.stage6_phase_b_long_trainer import Stage6PhaseBLongFacadeTrainer
from streetforward_core.train.stage6_phase_a_trainer import Stage6PhaseAFacadeTrainer
from tools.streetforward_validation_v9_config import ValidationV9Config, parse_validation_v9_config
from tools.train_minimal_streetforward_stage4_3_v7_common import parse_include_test, validate_train_scene_for_fixed
from tools.train_minimal_streetforward_stage4_3_v9_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v9_from_cfg,
    resolve_fixed_scene_segment_v9,
)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        out = node.get(key, default)
        return default if out is None else out
    if hasattr(node, key):
        out = getattr(node, key)
        return default if out is None else out
    return default


def _metric_or_nan(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _scheduler_long_phase_b_enabled(cfg: Any) -> bool:
    raw = cfg.get("scheduler_long_phase_b") if hasattr(cfg, "get") else None
    return raw is not None and bool(_cfg_get(raw, "enable", False))


def _validation_long_phase_b_enabled(cfg: Any) -> bool:
    raw = cfg.get("validation_long_phase_b") if hasattr(cfg, "get") else None
    return raw is not None and bool(_cfg_get(raw, "enable", False))


def resolve_fixed_scene_segment_stage6(cfg: Any) -> tuple[Optional[int], Optional[int]]:
    if _scheduler_long_phase_b_enabled(cfg):
        slong = cfg.get("scheduler_long_phase_b") if hasattr(cfg, "get") else None
        traversal = _cfg_get(slong, "traversal", {}) or {}
        fixed_scene_id = _cfg_get(traversal, "fixed_scene_id", None)
        fixed_segment_id = _cfg_get(traversal, "fixed_segment_id", None)
        return (
            None if fixed_scene_id is None else int(fixed_scene_id),
            None if fixed_segment_id is None else int(fixed_segment_id),
        )
    return resolve_fixed_scene_segment_v9(cfg)


def build_train_scheduler_long_phase_b_from_cfg(cfg: Any, dataset: Any) -> Any:
    slong = cfg.get("scheduler_long_phase_b") if hasattr(cfg, "get") else None
    if slong is None or not bool(_cfg_get(slong, "enable", False)):
        raise ValueError("scheduler_long_phase_b.enable must be true")
    if str(_cfg_get(slong, "version", "long_v1")) != "long_v1":
        raise ValueError("scheduler_long_phase_b.version must be long_v1")
    if str(_cfg_get(slong, "phase", PHASE_B_LONG_NAME)) != PHASE_B_LONG_NAME:
        raise ValueError("scheduler_long_phase_b.phase must be 6_0_phase_b")
    traversal = _cfg_get(slong, "traversal", {}) or {}
    fixed_scene_id, fixed_segment_id = resolve_fixed_scene_segment_stage6(cfg)
    validate_train_scene_for_fixed(cfg, fixed_scene_id)
    return dataset.create_train_scheduler_long_phase_b(
        episode_window_cfg=_cfg_get(slong, "episode_window", {}) or {},
        rollout_shapes=_cfg_get(slong, "rollout_shapes", []) or [],
        rollout_shapes_schedule=_cfg_get(slong, "rollout_shapes_schedule", []) or [],
        anchor_sampling_cfg=_cfg_get(slong, "anchor_sampling", {}) or {},
        traversal_cfg=traversal,
        preload_cfg=_cfg_get(slong, "preload", {}) or {},
        include_test=parse_include_test(cfg),
        fixed_scene_id=fixed_scene_id,
        fixed_segment_id=fixed_segment_id,
        evidence_cfg=_cfg_get(slong, "evidence", {}) or {},
        final_supervision_cfg=_cfg_get(slong, "final_supervision", {}) or {},
        rigid_meta_cfg=_cfg_get(slong, "rigid_meta", {}) or {},
        distant_meta_cfg=_cfg_get(slong, "distant_meta", {}) or {},
        fail_fast=bool(_cfg_get(slong, "fail_fast", True)),
    )


def build_train_scheduler_stage6_from_cfg(cfg: Any, dataset: Any) -> Any:
    if _scheduler_long_phase_b_enabled(cfg):
        return build_train_scheduler_long_phase_b_from_cfg(cfg, dataset)
    scheduler = build_train_scheduler_v9_from_cfg(cfg, dataset)
    sv9 = cfg.get("scheduler_v9") if hasattr(cfg, "get") else None
    phase = str(_cfg_get(sv9, "phase", PHASE_A_NAME))
    if phase == PHASE_A_NAME:
        return LegacyV9PhaseASchedulerAdapter(scheduler)
    return scheduler


def build_stage6_trainer_from_cfg(config: Any, device: torch.device) -> Any:
    model_cfg = _cfg_get(config, "model", {}) or {}
    phase = str(_cfg_get(model_cfg, "phase", PHASE_A_NAME))
    if phase == PHASE_A_NAME:
        return Stage6PhaseAFacadeTrainer(config=config, device=device)
    if phase == PHASE_B_LONG_NAME:
        return Stage6PhaseBLongFacadeTrainer(config=config, device=device)
    return MinimalStreetForwardStage6_0(config=config, device=device)


def checkpoint_prefix_stage6_from_cfg(cfg: Any) -> str:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    phase = str(_cfg_get(model_cfg, "phase", PHASE_A_NAME))
    if phase == PHASE_B_LONG_NAME:
        return "minimal_sf_stage6_0_phase_b_long_v1"
    return "minimal_sf_stage6_0_phase_a_v9"


def _scheduler_v9_blocks_per_episode(cfg: Any) -> int:
    sv9 = cfg.get("scheduler_v9") if hasattr(cfg, "get") else None
    if sv9 is None:
        raise ValueError("validation_v9 requires scheduler_v9")
    ep = sv9.get("episode") if hasattr(sv9, "get") else None
    if ep is None:
        raise ValueError("validation_v9 requires scheduler_v9.episode")
    blocks = _cfg_get(ep, "blocks_per_episode", None)
    if blocks is not None:
        return int(blocks)
    phase_b = _cfg_get(sv9, "phase_B", {}) or {}
    rollout = _cfg_get(phase_b, "rollout", {}) or {}
    shapes = [dict(x) for x in list(_cfg_get(rollout, "shapes", []) or [])]
    if not shapes:
        raise ValueError("validation_v9 requires scheduler_v9.episode.blocks_per_episode or phase_B.rollout.shapes")
    max_blocks = max(int(_cfg_get(shape, "blocks_per_rollout", 0) or 0) for shape in shapes)
    return int(_cfg_get(ep, "rollouts_per_episode", 1)) * int(max_blocks)


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
            _metric_or_nan(row, f"block_psnr@{int(plan.max_K)}"),
            int(plan.max_K),
            _metric_or_nan(row, f"nearby_psnr@{int(plan.max_K)}"),
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
        _metric_or_nan(summary, f"val_v9/phaseA/mean_block_psnr@{int(plan.max_K)}"),
        int(plan.max_K),
        _metric_or_nan(summary, f"val_v9/phaseA/mean_nearby_psnr@{int(plan.max_K)}"),
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
    if _validation_long_phase_b_enabled(cfg):
        _validation_long_phase_b_episode_end_hook(
            cfg=cfg,
            dataset=dataset,
            model=model,
            device=device,
            trigger_train_episode_counter=int(trigger_train_episode_counter),
            trigger_step=int(trigger_step),
            metrics_fh=metrics_fh,
            writer=writer,
        )
        return
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


def _run_validation_long_phase_b_round(
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
    raw = cfg.get("validation_long_phase_b") if hasattr(cfg, "get") else None
    if raw is None or not bool(_cfg_get(raw, "enable", False)):
        return
    if not hasattr(model, "validate_long_phase_b"):
        raise ValueError("validation_long_phase_b requires model.validate_long_phase_b")
    data_cfg = cfg.get("data") if hasattr(cfg, "get") else {}
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    if not eval_scene_ids:
        eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "train_scene_ids", []) or [])[:1]]
    if not eval_scene_ids:
        raise ValueError("validation_long_phase_b requires data.eval_scene_ids or data.train_scene_ids.")
    plan = build_validation_plan_long_phase_b(dataset=dataset, cfg=cfg, eval_scene_ids=eval_scene_ids)
    if not plan.specs:
        raise ValueError("validation_long_phase_b enabled but no validation specs were built.")
    mask_cfg = _cfg_get(raw, "masks", {}) or {}
    mask_policy = str(_cfg_get(mask_cfg, "mask_policy", "non_sky_non_egocar"))
    min_valid_pixels = int(_cfg_get(mask_cfg, "min_valid_pixels", 1))
    ablations_cfg = _cfg_get(raw, "ablations", {}) or {}
    ablations = ["normal"]
    if bool(_cfg_get(ablations_cfg, "zero_vsm", True)):
        ablations.append("zero_vsm")
    if bool(_cfg_get(ablations_cfg, "zero_read_keep_seen", True)):
        ablations.append("zero_read_keep_seen")
    if bool(_cfg_get(ablations_cfg, "zero_read_zero_seen", False)):
        ablations.append("zero_read_zero_seen")
    if bool(_cfg_get(ablations_cfg, "shuffle_vsm", True)):
        ablations.append("shuffle_vsm")
    if bool(_cfg_get(ablations_cfg, "shuffle_read", False)):
        ablations.append("shuffle_read")
    if bool(_cfg_get(ablations_cfg, "zero_delta", True)):
        ablations.append("zero_delta")
    if bool(_cfg_get(ablations_cfg, "seen_only", False)):
        ablations.append("seen_only")
    base.logger.info(
        "VALIDATION_LONG_PHASE_B_BEGIN trigger_episode_counter=%s trigger_step=%s specs=%s T=%s orders=%s",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(plan.specs)),
        [int(x) for x in plan.interval_T_values],
        [str(x) for x in plan.orders],
    )
    rows = []
    for spec in plan.specs:
        raw_batch = materialize_validation_long_phase_b_batch(dataset, spec, include_test=False)
        minimal_batch = base.convert_batch_to_minimal_format(
            raw_batch,
            device,
            num_targets=int(raw_batch["target"]["image"].shape[0]),
            include_source_for_2d=True,
            view_selection=None,
        )
        row = model.validate_long_phase_b(
            minimal_batch,
            mask_policy=str(mask_policy),
            min_valid_pixels=int(min_valid_pixels),
            ablations=ablations,
        )
        row.update(
            {
                "split": "validation_long_phase_b",
                "trigger_train_episode_counter": int(trigger_train_episode_counter),
                "trigger_step": int(trigger_step),
                "scene_id": int(spec.scene_id),
                "segment_id": int(spec.segment_id),
                "interval_T": int(spec.interval_T),
                "order": str(spec.order),
            }
        )
        rows.append(row)
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, row)
        if writer is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(str(key), float(value), int(trigger_step))
    summary: Dict[str, Any] = {
        "split": "validation_long_phase_b_global",
        "trigger_train_episode_counter": int(trigger_train_episode_counter),
        "trigger_step": int(trigger_step),
        "num_specs": int(len(rows)),
    }
    for metric in ("segment_all_psnr", "segment_all_ssim", "segment_all_l1", "segment_all_lpips"):
        vals = [float(r.get(f"val_long/{metric}", 0.0)) for r in rows if isinstance(r.get(f"val_long/{metric}", None), (int, float))]
        if vals:
            summary[f"val_long/mean_{metric}"] = float(sum(vals) / max(len(vals), 1))
    if metrics_fh is not None:
        base._write_metrics_history(metrics_fh, summary)
    if writer is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(str(key), float(value), int(trigger_step))
    base.logger.info(
        "VALIDATION_LONG_PHASE_B_END trigger_episode_counter=%s trigger_step=%s specs=%s mean_segment_psnr=%.4f",
        int(trigger_train_episode_counter),
        int(trigger_step),
        int(len(rows)),
        float(summary.get("val_long/mean_segment_all_psnr", 0.0)),
    )


def _validation_long_phase_b_episode_end_hook(
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
    raw = cfg.get("validation_long_phase_b") if hasattr(cfg, "get") else None
    if raw is None or not bool(_cfg_get(raw, "enable", False)):
        return
    trigger = _cfg_get(raw, "trigger", {}) or {}
    every = max(int(_cfg_get(trigger, "validate_every_n_episodes", 100)), 1)
    if int(trigger_train_episode_counter) % int(every) != 0:
        return
    _run_validation_long_phase_b_round(
        cfg=cfg,
        dataset=dataset,
        model=model,
        device=device,
        trigger_train_episode_counter=int(trigger_train_episode_counter),
        trigger_step=int(trigger_step),
        metrics_fh=metrics_fh,
        writer=writer,
    )


def _validation_long_phase_b_train_start_hook(
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
    raw = cfg.get("validation_long_phase_b") if hasattr(cfg, "get") else None
    if raw is None or not bool(_cfg_get(raw, "enable", False)):
        return
    trigger = _cfg_get(raw, "trigger", {}) or {}
    if not bool(_cfg_get(trigger, "run_at_train_start", False)):
        return
    _run_validation_long_phase_b_round(
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
    if _validation_long_phase_b_enabled(cfg):
        _validation_long_phase_b_train_start_hook(
            cfg=cfg,
            dataset=dataset,
            model=model,
            device=device,
            trigger_train_episode_counter=int(trigger_train_episode_counter),
            trigger_step=int(trigger_step),
            metrics_fh=metrics_fh,
            writer=writer,
        )
        return
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
    base.build_train_scheduler_from_cfg = build_train_scheduler_stage6_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_stage6
    base.EPISODE_END_HOOK = _validation_v9_episode_end_hook
    base.TRAIN_START_HOOK = _validation_v9_train_start_hook
    base.TRAINER_CLASS = build_stage6_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = build_stage6_trainer_from_cfg
    base.CKPT_PREFIX = "minimal_sf_stage6_0_phase_a_v9"
    base.CHECKPOINT_PREFIX_RESOLVER = checkpoint_prefix_stage6_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.main()


if __name__ == "__main__":
    main()
