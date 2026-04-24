"""
Stage 4.3 multi-scene training entry for V4 dataset + V8 scheduler.

Thin wrapper over the stable multi-scene v4 training loop:
- swap dataset builder to MultiSceneDatasetV4
- swap scheduler builder to TrainSchedulerV8
- swap validation config/spec parser to validation_v8
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, List, Optional, TextIO

import torch
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.validation_scheduler_v8 import (
    ValidationEpisodeSpecV8,
    build_validation_episode_specs_v8,
)
from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.streetforward_validation_v8_config import ValidationV8Config, parse_validation_v8_config
from tools.train_minimal_streetforward_stage4_3_v8_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_v8_from_cfg,
    resolve_fixed_scene_segment_v8,
)


def _parse_validation_v8_config_with_context(cfg: Any) -> ValidationV8Config:
    out = parse_validation_v8_config(cfg)
    base._validation_v8_cfg_runtime = out
    return out


def _build_validation_specs_v8_proxy(
    *,
    dataset: Any,
    eval_scene_ids: List[int],
    blocks_per_episode: int,
    total_target_frames: int,
) -> List[ValidationEpisodeSpecV8]:
    vcfg: Optional[ValidationV8Config] = getattr(base, "_validation_v8_cfg_runtime", None)
    steps_per_block = int(getattr(vcfg, "steps_per_block", 1))
    block_order = str(getattr(vcfg, "block_order", "block_major"))
    switch_steps = int(getattr(vcfg, "step_major_switch_interval_steps", 1))
    return build_validation_episode_specs_v8(
        dataset=dataset,
        eval_scene_ids=[int(x) for x in eval_scene_ids],
        blocks_per_episode=int(blocks_per_episode),
        total_target_frames=int(total_target_frames),
        steps_per_block=int(steps_per_block),
        block_order=str(block_order),
        step_major_switch_interval_steps=int(switch_steps),
    )


def _run_validation_v8_round(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    specs: List[ValidationEpisodeSpecV8],
    validation_cfg: ValidationV8Config,
    device: torch.device,
    trigger_train_episode_counter: int,
    trigger_step: int,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: SSIM,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    metrics_fh: Optional[TextIO],
    writer: Optional[Any] = None,
) -> None:
    _ = (trigger_train_episode_counter, trigger_step, psnr_metric, ssim_metric, lpips_metric, metrics_fh, writer)
    if len(specs) == 0:
        base.logger.warning("validation_v8 enabled but no valid episode specs from eval_scene_ids")
        return
    validation_mode = str(validation_cfg.mode)
    use_train_finetune = validation_mode == "segment_finetune_train"
    infer_policy = base.RuntimePolicy(
        do_backward=False,
        do_optimizer_step=False,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    train_policy = base.RuntimePolicy(
        do_backward=True,
        do_optimizer_step=True,
        update_hidden_cache=True,
        writeback_node_state=True,
        reset_node_state_after_block=False,
    )
    base_ckpt_bytes: Optional[bytes] = None
    if use_train_finetune:
        base_ckpt_bytes = base._snapshot_train_checkpoint_bytes(model)
    train_step_supports_runtime_policy = "runtime_policy" in inspect.signature(model.train_step).parameters
    infer_step_supports_runtime_policy = (
        "runtime_policy" in inspect.signature(model.inference_step_from_train_batch).parameters
    )

    for spec in specs:
        if use_train_finetune:
            base._restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
        model.reset_node_state()
        validation_local_step = 0
        for visit_idx, (block_idx_in_episode, block_frames) in enumerate(
            zip(spec.block_visit_order, spec.visit_target_windows)
        ):
            src_frame = int(block_frames[0])
            source_ref = (int(src_frame), 0)
            source_refs = [(int(src_frame), int(cam_id)) for cam_id in range(int(spec.num_cams))]
            target_refs = [
                (int(frame_idx), int(cam_id))
                for frame_idx in block_frames
                for cam_id in range(int(spec.num_cams))
            ]
            req = base._BatchRequestValidationV7(
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                source_image_ref=source_ref,
                source_image_refs=source_refs,
                target_image_refs=target_refs,
                include_test=False,
                test_image_refs=None,
            )
            raw_batch = dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
            minimal_batch = base.convert_batch_to_minimal_format(
                raw_batch,
                device,
                num_targets=int(raw_batch["target"]["image"].shape[0]),
                include_source_for_2d=True,
                view_selection=None,
            )
            scheduler_node_sync = {
                "U": 1,
                "segment_local_step": int(validation_local_step + 1),
                "reset_after_block": False,
            }
            if use_train_finetune:
                kwargs: Dict[str, Any] = {
                    "batch": minimal_batch,
                    "step": None,
                    "profile_phase_timing": False,
                    "sync_cuda_timing": False,
                    "scheduler_node_sync": scheduler_node_sync,
                }
                if train_step_supports_runtime_policy:
                    kwargs["runtime_policy"] = train_policy
                step_result = model.train_step(**kwargs)
            else:
                kwargs = {
                    "batch": minimal_batch,
                    "step": None,
                    "scheduler_node_sync": scheduler_node_sync,
                }
                if infer_step_supports_runtime_policy:
                    kwargs["runtime_policy"] = infer_policy
                step_result = model.inference_step_from_train_batch(**kwargs)
            validation_local_step += 1
            base.logger.info(
                "VALIDATION_V8_BLOCK_VISIT mode=%s block_order=%s scene_id=%s segment_id=%s "
                "block=%s visit=%s/%s source_frame=%s target_frames=%s loss=%.6f",
                validation_mode,
                str(validation_cfg.block_order),
                int(spec.scene_id),
                int(spec.segment_id),
                int(block_idx_in_episode),
                int(visit_idx + 1),
                int(len(spec.block_visit_order)),
                int(src_frame),
                [int(x) for x in block_frames],
                float(step_result.get("loss", 0.0)),
            )


def _setup_v8(args: Any) -> Any:
    cfg = _ORIG_SETUP(args)
    if cfg.get("scheduler_v8") is not None and cfg.get("scheduler_v7") is None:
        cfg["scheduler_v7"] = cfg.get("scheduler_v8")
    if cfg.get("validation_v8") is not None and cfg.get("validation_v7") is None:
        cfg["validation_v7"] = cfg.get("validation_v8")
    return cfg


_ORIG_SETUP = base.setup


def main() -> None:
    if "--config_file" not in sys.argv:
        sys.argv.extend(
            [
                "--config_file",
                "configs/minimal_streetforward_stage4_4_multi_scene_v8.yaml",
            ]
        )
    base.setup = _setup_v8
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_v8_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_v8
    base.parse_validation_v7_config = _parse_validation_v8_config_with_context
    base.build_validation_episode_specs_v7 = _build_validation_specs_v8_proxy
    base._run_validation_v7_round = _run_validation_v8_round
    if (
        getattr(base, "TRAINER_CLASS", None) is None
        or getattr(base.TRAINER_CLASS, "__name__", "") == "MinimalStreetForwardStage4_3"
    ):
        base.TRAINER_CLASS = MinimalStreetForwardStage4_3
    if str(getattr(base, "CKPT_PREFIX", "")) == "minimal_sf_stage4_3_multi_scene_v4":
        base.CKPT_PREFIX = "minimal_sf_stage4_3_multi_scene_v8"
    if str(getattr(base, "DEFAULT_CONFIG_FILE", "")) == "configs/minimal_streetforward_stage4_4_multi_scene_v4.yaml":
        base.DEFAULT_CONFIG_FILE = "configs/minimal_streetforward_stage4_4_multi_scene_v8.yaml"
    base.main()


if __name__ == "__main__":
    main()

