"""Reproducible CUDA K=15 profile for IForward observation feedback.

The profiler uses the production Stage3.3 dataset, scheduler, model, optimizer,
and ``train_step`` transaction.  It intentionally does not create checkpoints
or long-running experiment artifacts.  A one-rollout prelude initializes the
episode carry; the following fixed B5R3 repair rollout is the measured K=15
transaction for every variant.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import OmegaConf

from models.iforward import IForwardTrainer
from tools.train_minimal_streetforward_stage4_3_iforward_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_from_cfg,
)
from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import (
    convert_batch_to_minimal_format,
)


VARIANTS = (
    "baseline_frozen_no_grad",
    "render_eager",
    "render_checkpoint",
    "render_parent_vjp",
    "full_relation",
)


def _clone_cfg(cfg: Any) -> Any:
    return OmegaConf.create(copy.deepcopy(OmegaConf.to_container(cfg, resolve=False)))


def _fixed_k15_cfg(base_cfg: Any, *, variant: str, scene_id: int, segment_id: int, seed: int) -> Any:
    if variant not in VARIANTS:
        raise ValueError(f"unknown profile variant {variant!r}; expected one of {VARIANTS}")
    cfg = _clone_cfg(base_cfg)
    cfg.scheduler_stage3_0.traversal.fixed_scene_id = int(scene_id)
    cfg.scheduler_stage3_0.traversal.fixed_segment_id = int(segment_id)
    cfg.scheduler_stage3_0.traversal.seed = int(seed)
    cfg.scheduler_stage3_0.producer.enable = False
    cfg.scheduler_stage3_2.episode_recipe.prelude.min_rollouts = 1
    cfg.scheduler_stage3_2.episode_recipe.prelude.max_rollouts = 1
    cfg.scheduler_stage3_2.episode_recipe.repair_tail.min_rollouts = 1
    cfg.scheduler_stage3_2.episode_recipe.repair_tail.max_rollouts = 1
    # B5R3 is exactly K=15 and exercises both cross-block and repeat paths.
    cfg.scheduler_stage3_2.distributions.high_block_repair.b_choices = {5: 1.0}
    cfg.scheduler_stage3_2.distributions.high_block_repair.r_choices = {3: 1.0}
    # The prelude is forward/state-identical across variants; only the measured
    # step enables the requested feedback Jacobians.
    cfg.model.iforward.observation_feedback.source_render.alpha_schedule = [[0, 0.0], [1, 1.0]]
    cfg.model.iforward.observation_feedback.parent_projection.alpha_schedule = [[0, 0.0], [1, 0.3]]
    cfg.model.iforward.observation_feedback.relation.alpha_schedule = [[0, 0.0], [1, 0.3]]
    cfg.model.iforward.observation_feedback.debug.grad_probe_interval = 1
    cfg.model.iforward.observation_feedback.debug.forward_parity_interval = 0
    cfg.data.preload_scene_count = 1
    cfg.data.train_scene_ids = [int(scene_id)]
    cfg.data.eval_scene_ids = []

    feedback = cfg.model.iforward.observation_feedback
    feedback.schedule.activation_step = 0
    # The profiler's fixed scene is a kernel/memory diagnostic; production
    # training remains fail-fast until nuScenes ego-mask assets are installed.
    cfg.data.pixel_source.require_egocar_mask_template = False
    parent_enabled = variant in {"render_parent_vjp", "full_relation"}
    relation_enabled = variant == "full_relation"
    source_enabled = variant != "baseline_frozen_no_grad"
    feedback.source_render.enable = bool(source_enabled)
    feedback.parent_projection.enable = bool(parent_enabled)
    feedback.relation.enable = bool(relation_enabled)

    if variant == "baseline_frozen_no_grad":
        modes = {
            "repeat_refine": "trainable_checkpointed",
            "shuffled_coverage": "trainable_checkpointed",
            "high_block_repair": "frozen_no_grad",
        }
    else:
        modes = {
            "repeat_refine": "trainable_checkpointed",
            "shuffled_coverage": "trainable_checkpointed",
            "high_block_repair": "frozen_input_grad_checkpointed",
        }
    feedback.modes = dict(modes)
    cfg.scheduler_stage3_2.episode_recipe.train_2d_policy = dict(modes)
    return cfg


def _minimal_batch(raw_batch: Dict[str, Any], *, device: torch.device, step: int) -> Dict[str, Any]:
    target = raw_batch.get("target") or {}
    image = target.get("image") if isinstance(target, dict) else None
    if not torch.is_tensor(image):
        raise ValueError("profile scheduler batch must contain target.image")
    batch = convert_batch_to_minimal_format(
        raw_batch,
        device,
        num_targets=int(image.shape[0]),
        include_source_for_2d=True,
        view_selection=None,
    )
    batch["global_step"] = int(step)
    return batch


def _stage32_meta(raw_batch: Dict[str, Any]) -> Dict[str, Any]:
    request = dict((raw_batch.get("_iforward") or {}).get("request_meta") or {})
    if not request:
        request = dict(raw_batch.get("request_meta") or {})
    return dict(request.get("iforward_stage3_2") or {})


def _install_eager_source_override(trainer: IForwardTrainer) -> None:
    runtime = trainer._phase_a_runtime()
    if runtime is None:
        raise RuntimeError("render_eager profile requires the Stage6 runtime")
    original = runtime._render_source_scene_only_for_cnn

    def eager_source(_self: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if bool(kwargs.get("feedback_enabled", False)):
            kwargs["checkpoint_dynamic"] = False
        return original(*args, **kwargs)

    runtime._render_source_scene_only_for_cnn = types.MethodType(eager_source, runtime)


def _numeric_result_is_finite(result: Dict[str, Any]) -> bool:
    for value in result.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
        if torch.is_tensor(value) and torch.is_floating_point(value):
            if not bool(torch.isfinite(value.detach()).all().item()):
                return False
    return True


def _profile_variant(
    *,
    base_cfg: Any,
    dataset: Any,
    variant: str,
    device: torch.device,
    scene_id: int,
    segment_id: int,
    seed: int,
    trace_dir: str = "",
) -> Dict[str, Any]:
    cfg = _fixed_k15_cfg(
        base_cfg,
        variant=variant,
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        seed=int(seed),
    )
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    scheduler = build_train_scheduler_iforward_from_cfg(cfg, dataset)
    trainer = IForwardTrainer(config=cfg, device=device).train()
    if variant == "render_eager":
        _install_eager_source_override(trainer)

    prelude_raw = scheduler.next_batch()
    prelude_meta = _stage32_meta(prelude_raw)
    prelude = _minimal_batch(prelude_raw, device=device, step=0)
    trainer.train_step(prelude, step=0)

    measured_raw = scheduler.next_batch()
    measured_meta = _stage32_meta(measured_raw)
    actual_k = int(measured_meta.get("K", 0))
    if str(measured_meta.get("distribution_type", "")) != "high_block_repair" or actual_k != 15:
        raise RuntimeError(f"expected fixed high_block_repair K=15, got {measured_meta}")
    measured = _minimal_batch(measured_raw, device=device, step=1)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    started = time.perf_counter()
    profiler = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        if str(trace_dir)
        else nullcontext()
    )
    with profiler as active_profiler:
        result = trainer.train_step(
            measured,
            step=1,
            profile_phase_timing=True,
            sync_cuda_timing=True,
            profile_cuda_memory=True,
        )
    torch.cuda.synchronize(device)
    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
    trace_path = ""
    if str(trace_dir):
        trace_path = str(Path(str(trace_dir)) / f"{variant}.json")
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        active_profiler.export_chrome_trace(trace_path)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    mb = 1024.0 * 1024.0
    row = {
        "variant": str(variant),
        "status": "ok",
        "scene_id": int(scene_id),
        "segment_id": int(segment_id),
        "seed": int(seed),
        "prelude_distribution": str(prelude_meta.get("distribution_type", "")),
        "prelude_k": int(prelude_meta.get("K", 0)),
        "distribution": str(measured_meta.get("distribution_type", "")),
        "B": int(measured_meta.get("B", 0)),
        "R": int(measured_meta.get("R", 0)),
        "K": int(actual_k),
        "mode": str(measured_meta.get("train_2d_mode", "")),
        "step_time_ms": elapsed_ms,
        "allocated_before_mb": allocated_before / mb,
        "reserved_before_mb": reserved_before / mb,
        "peak_allocated_mb": peak_allocated / mb,
        "peak_reserved_mb": peak_reserved / mb,
        "peak_incremental_allocated_mb": (peak_allocated - allocated_before) / mb,
        "finite": bool(_numeric_result_is_finite(result)),
        "optimizer_step_skipped": float(result.get("amp/optimizer_step_skipped", 0.0)),
        "loss": float(result.get("loss", result.get("iforward/loss_total", float("nan")))),
        "source_grad_norm": float(result.get("feedback/source_render_input_grad_norm", 0.0)),
        "frontend_grad_count": float(result.get("feedback/2d_param_grad_count", 0.0)),
        "parent_vjp_bg_reports": float(result.get("feedback/parent_vjp/bg/backward_reports", 0.0)),
        "relation_enabled": float(result.get("iforward/feedback/relation_enabled", 0.0)),
        "forward_ms": float(result.get("forward_ms", result.get("timing/forward_ms", 0.0))),
        "backward_ms": float(result.get("backward_ms", result.get("timing/backward_ms", 0.0))),
        "profiler_trace": trace_path,
    }
    shutdown_scheduler = getattr(scheduler, "shutdown", None)
    if callable(shutdown_scheduler):
        shutdown_scheduler()
    del measured, measured_raw, prelude, prelude_raw, trainer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _parse_variants(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    unknown = sorted(set(values) - set(VARIANTS))
    if not values or unknown:
        raise ValueError(f"invalid --variants={raw!r}; unknown={unknown}, supported={VARIANTS}")
    return values


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/iforward/iforward_stage3_3_observation_feedback.yaml",
    )
    parser.add_argument("--scene-id", type=int, default=131)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--trace-dir",
        default="",
        help="Optional Chrome-trace directory; profiling changes timing/memory, so leave empty for acceptance runs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the observation feedback K=15 profile")

    device = torch.device("cuda")
    base_cfg = OmegaConf.load(str(args.config))
    dataset_cfg = _fixed_k15_cfg(
        base_cfg,
        variant="render_checkpoint",
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(args.seed),
    )
    dataset = build_multi_scene_dataset_v4(dataset_cfg, device)
    dataset.initialize()
    rows = []
    for variant in _parse_variants(args.variants):
        try:
            rows.append(
                _profile_variant(
                    base_cfg=base_cfg,
                    dataset=dataset,
                    variant=variant,
                    device=device,
                    scene_id=int(args.scene_id),
                    segment_id=int(args.segment_id),
                    seed=int(args.seed),
                    trace_dir=str(args.trace_dir),
                )
            )
        except torch.cuda.OutOfMemoryError as exc:
            rows.append({"variant": variant, "status": "oom", "error": str(exc)})
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    shutdown_dataset = getattr(dataset, "shutdown_preload", None)
    if callable(shutdown_dataset):
        shutdown_dataset()

    baseline = next((row for row in rows if row.get("variant") == "baseline_frozen_no_grad" and row.get("status") == "ok"), None)
    baseline_peak = float(baseline["peak_allocated_mb"]) if baseline is not None else 0.0
    for row in rows:
        if row.get("status") == "ok" and baseline_peak > 0.0:
            row["peak_vs_baseline"] = float(row["peak_allocated_mb"]) / baseline_peak
            row["within_1_15x_baseline"] = bool(row["peak_vs_baseline"] <= 1.15)
    report = {
        "schema": "iforward_observation_feedback_k15_profile_v1",
        "config": str(args.config),
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "seed": int(args.seed),
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.output_json):
        path = Path(str(args.output_json))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
