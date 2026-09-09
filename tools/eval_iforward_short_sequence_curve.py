from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import random
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.iforward_stage2_3.scheduler import Stage23Scheduler
from datasets.iforward_stage2_3.schema import EpisodePlanV3
from datasets.iforward_stage2_3.validation_runner import (
    _clone_state_for_validation,
    _detach_next_state,
    _manual_stage2_3_plan,
    _run_plan,
    _row_from_output,
    _should_carry,
)
from tools.iforward_validate_v4 import build_iforward_runtime_from_cfg
from tools.train_iforward import _sequence10_minimal_from_scheduler_batch


LOGGER = logging.getLogger("eval_iforward_short_sequence_curve")
DEFAULT_LENGTHS = (5, 10, 15, 20, 26, 30, 35, 40)


def _checkpoint_step(payload: Dict[str, Any]) -> int:
    for key in ("step", "global_step", "iteration", "iter"):
        if payload.get(key) is not None:
            return int(payload[key])
    return 0


def _eval_cfg(
    cfg: Any,
    *,
    scene_id: int,
    segment_id: int,
    max_length: int,
    updates_per_frame: int,
) -> Any:
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if out.get("scheduler_stage3_2") is not None:
        out.scheduler_stage3_2.enable = False
    sched = out.scheduler_stage3_0
    sched.enable = True
    sched.traversal.fixed_scene_id = int(scene_id)
    sched.traversal.fixed_segment_id = int(segment_id)
    sched.traversal.seed = 41
    sched.producer.enable = False
    sched.repair.enable = False
    sched.sequence.min_frames = 1
    sched.sequence.max_frames = int(max_length)
    sched.sequence.min_unique_keyframes = 1
    sched.sequence.min_frame_span = 0
    sched.sequence.max_frame_span = max(1000, int(max_length) * 4)
    sched.sequence.frame_count_schedule = [
        {
            "start_step": 0,
            "target_frames": int(max_length),
            "min_frames": int(max_length),
            "allow_short": False,
        }
    ]
    sched.sequence.assimilation_order = {"chronological": 1.0, "local_shuffle": 0.0}
    sched.assimilation.max_inner_k = int(2 * updates_per_frame)
    sched.assimilation.rollout_options = {}
    sched.assimilation.repeat_pairs = {
        f"{int(updates_per_frame)},{int(updates_per_frame)}": 1.0,
    }
    sched.assimilation.single_repeat_distribution = {int(updates_per_frame): 1.0}
    return out


def _segment_row(scheduler: Stage23Scheduler, scene_id: int, segment_id: int) -> int:
    matches = [
        idx
        for idx, row in enumerate(scheduler.index.segments)
        if int(row["scene_id"]) == int(scene_id) and int(row["segment_id"]) == int(segment_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one index row for scene={scene_id} segment={segment_id}, got {matches}"
        )
    return int(matches[0])


def _force_rows(
    scheduler: Stage23Scheduler,
    *,
    segment_row: int,
    selected_rows: np.ndarray,
) -> None:
    rule = {
        "start_step": 0,
        "target_frames": int(selected_rows.shape[0]),
        "min_frames": int(selected_rows.shape[0]),
        "allow_short": False,
        "scheduled": True,
    }

    def _sample_sequence_rows(_self: Stage23Scheduler) -> Any:
        return (
            int(segment_row),
            selected_rows.copy(),
            tuple(range(int(selected_rows.shape[0]))),
            dict(rule),
        )

    scheduler._sample_sequence_rows = types.MethodType(_sample_sequence_rows, scheduler)


def _reset_runtime(model: Any) -> None:
    reset = getattr(model, "reset_iforward_state_cache", None)
    if callable(reset):
        reset()
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    if callable(reset_bridge):
        reset_bridge()


def _run_causal_episode_streaming(
    *,
    scheduler: Stage23Scheduler,
    episode: EpisodePlanV3,
    model: Any,
    device: torch.device,
    trigger_step: int,
) -> tuple[Any, List[Dict[str, Any]], Any]:
    carried_state = None
    rows: List[Dict[str, Any]] = []
    last_out = None
    for rollout_idx, plan in enumerate(tuple(episode.rollouts)):
        out = _run_plan(
            scheduler=scheduler,
            plan=plan,
            model=model,
            carried_state=carried_state,
            mode="full",
            device=device,
            trigger_step=int(trigger_step),
            convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
        )
        rows.append(
            _row_from_output(
                out=out,
                protocol="Assimilation-Causal-FinalAll",
                rollout_idx=int(rollout_idx),
                mode="full",
                trigger_step=int(trigger_step),
            )
        )
        carried_state = _detach_next_state(out) if _should_carry(out) else None
        last_out = out
    if last_out is not None:
        carried_state = _detach_next_state(last_out)
    return carried_state, rows, last_out


def _build_sequence_sweep_episode(
    *,
    scheduler: Stage23Scheduler,
    base_episode: EpisodePlanV3,
    rows: np.ndarray,
    sequence_passes: int,
    frames_per_rollout: int = 2,
) -> EpisodePlanV3:
    if int(sequence_passes) == 1:
        return base_episode
    if int(sequence_passes) < 1:
        raise ValueError("sequence_passes must be positive")
    if int(frames_per_rollout) < 1:
        raise ValueError("frames_per_rollout must be positive")

    length = int(rows.shape[0])
    chunks = [
        list(range(start, min(start + int(frames_per_rollout), length)))
        for start in range(0, length, int(frames_per_rollout))
    ]
    total_rollouts = int(sequence_passes) * len(chunks)
    rollouts = []
    visit_counts: Dict[int, int] = {}
    last_visit_step_by_pos: Dict[int, int] = {}
    step_offset = 0
    rollout_idx = 0
    sampled_order: List[int] = []
    for pass_idx in range(int(sequence_passes)):
        # A new optimizer sweep revisits the same frozen timeline. Keep GS and
        # optimizer memory, but do not create a synthetic negative time jump
        # from the final frame of pass k back to frame zero of pass k+1.
        last_visit_context: Dict[str, Any] = {}
        if pass_idx > 0:
            first = rows[0]
            last_visit_context = {
                "sequence_pos": 0,
                "frame_idx": int(first["frame_idx"]),
                "timestamp_us": int(first["timestamp_us"]),
                "ego_translation": np.asarray(first["ego_translation"], dtype=np.float32),
                "ego_yaw": float(first["ego_yaw"]),
                "global_update_idx": int(max(0, step_offset - 1)),
            }
        for chunk in chunks:
            history = [
                pos
                for pos in range(length)
                if int(visit_counts.get(pos, 0)) > 0 and pos not in set(chunk)
            ]
            plan = scheduler._rollout_from_positions(
                rows=rows,
                scene_id=int(base_episode.scene_id),
                segment_id=int(base_episode.segment_id),
                sequence_id=int(base_episode.sequence_id),
                positions=chunk,
                repeat_budgets=[1 for _ in chunk],
                rollout_idx=int(rollout_idx),
                rollouts_per_episode=int(total_rollouts),
                phase="assimilation",
                visit_kind="assimilate",
                history_positions=history,
                repair_positions=[],
                repair_enabled=False,
                repair_hash=-1,
                episode_step_offset=int(step_offset),
                visit_counts=visit_counts,
                last_visit_step_by_pos=last_visit_step_by_pos,
                is_last_rollout=bool(rollout_idx == total_rollouts - 1),
                last_visit_context=last_visit_context,
                phase_max_inner_k=max(2, int(frames_per_rollout)),
                requested_inner_k=len(chunk),
                requested_blocks_per_rollout=len(chunk),
                sequence_target_frames=length,
                sequence_min_frames=length,
                sequence_allow_short=False,
            )
            request_meta = dict(getattr(plan, "request_meta", {}) or {})
            request_meta["sequence_sweep_pass_idx"] = int(pass_idx)
            request_meta["sequence_sweep_passes"] = int(sequence_passes)
            plan = dataclasses.replace(plan, request_meta=request_meta)
            rollouts.append(plan)
            sampled_order.extend(chunk)
            step_offset += len(tuple(plan.steps))
            rollout_idx += 1

    return dataclasses.replace(
        base_episode,
        sampled_order=tuple(int(pos) for pos in sampled_order),
        rollouts=tuple(rollouts),
        repair_enabled=False,
        metadata={
            **dict(base_episode.metadata or {}),
            "sequence_sweep_passes": int(sequence_passes),
            "sequence_sweep_frames_per_rollout": int(frames_per_rollout),
        },
    )


def _full_image_psnr(out: Any) -> float:
    values: List[float] = []
    for pred, gt, role in zip(
        list(getattr(out, "pred_rgbs", []) or []),
        list(getattr(out, "gt_images", []) or []),
        list(getattr(out, "image_roles", []) or []),
    ):
        if str(role) != "current_latest":
            continue
        pred_t = torch.as_tensor(pred).detach().float().clamp(0.0, 1.0)
        gt_t = torch.as_tensor(gt).detach().float().clamp(0.0, 1.0)
        mse = (pred_t - gt_t).pow(2).mean()
        values.append(float((-10.0 * torch.log10(mse.clamp_min(1.0e-12))).item()))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _tensor_hwc(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().clamp(0.0, 1.0).cpu()
    if tensor.ndim != 3:
        raise ValueError(f"expected a 3D RGB tensor, got {tuple(tensor.shape)}")
    if int(tensor.shape[-1]) == 3:
        return tensor
    if int(tensor.shape[0]) == 3:
        return tensor.permute(1, 2, 0)
    raise ValueError(f"expected HWC or CHW RGB tensor, got {tuple(tensor.shape)}")


def _output_diagnostics(out: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pred_raw, gt_raw, ref, role in zip(
        list(getattr(out, "pred_rgbs", []) or []),
        list(getattr(out, "gt_images", []) or []),
        list(getattr(out, "image_refs", []) or []),
        list(getattr(out, "image_roles", []) or []),
    ):
        if str(role) != "current_latest":
            continue
        pred = _tensor_hwc(pred_raw)
        gt = _tensor_hwc(gt_raw)
        mse = (pred - gt).pow(2).mean()
        rows.append(
            {
                "frame_id": int(ref[0]),
                "cam_id": int(ref[1]),
                "psnr_full": float((-10.0 * torch.log10(mse.clamp_min(1.0e-12))).item()),
                "pred_min": float(pred.min().item()),
                "pred_max": float(pred.max().item()),
                "pred_mean": float(pred.mean().item()),
                "pred_std": float(pred.std().item()),
                "gt_mean": float(gt.mean().item()),
                "gt_std": float(gt.std().item()),
                "finite": bool(torch.isfinite(pred).all().item()),
            }
        )
    return rows


def _save_diagnostic_images(
    *,
    output_dir: Path,
    label: str,
    out: Any,
    selected_frames: Sequence[int],
) -> None:
    wanted = {int(value) for value in selected_frames}
    output_dir.mkdir(parents=True, exist_ok=True)
    for pred_raw, gt_raw, ref, role in zip(
        list(getattr(out, "pred_rgbs", []) or []),
        list(getattr(out, "gt_images", []) or []),
        list(getattr(out, "image_refs", []) or []),
        list(getattr(out, "image_roles", []) or []),
    ):
        if str(role) != "current_latest" or int(ref[0]) not in wanted:
            continue
        pred = _tensor_hwc(pred_raw)
        gt = _tensor_hwc(gt_raw)
        for kind, tensor in (("pred", pred), ("gt", gt)):
            image = tensor.mul(255.0).round().byte().numpy()
            Image.fromarray(image).save(
                output_dir
                / f"{label}_frame{int(ref[0]):03d}_cam{int(ref[1])}_{kind}.png"
            )


def _state_diagnostics(state: Any) -> Dict[str, Any]:
    local = getattr(state, "local_gs", None)
    result: Dict[str, Any] = {}
    for branch_name in ("bg", "distant", "rigid"):
        branch = getattr(local, branch_name, None)
        if branch is None:
            continue
        branch_result: Dict[str, Any] = {}
        for field in ("means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest"):
            value = getattr(branch, field, None)
            if not torch.is_tensor(value) or int(value.numel()) == 0:
                continue
            tensor = value.detach().float()
            finite = torch.isfinite(tensor)
            finite_values = tensor[finite]
            branch_result[field] = {
                "shape": list(tensor.shape),
                "finite_ratio": float(finite.float().mean().item()),
                "min": float(finite_values.min().item()) if int(finite_values.numel()) else float("nan"),
                "max": float(finite_values.max().item()) if int(finite_values.numel()) else float("nan"),
                "mean": float(finite_values.mean().item()) if int(finite_values.numel()) else float("nan"),
                "rms": (
                    float(finite_values.square().mean().sqrt().item())
                    if int(finite_values.numel())
                    else float("nan")
                ),
            }
        result[branch_name] = branch_result
    return result


def _evaluate_length(
    *,
    base_cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int,
    scene_id: int,
    segment_id: int,
    selected_rows: np.ndarray,
    seed: int,
    updates_per_frame: int,
    sequence_passes: int,
    diagnostics_dir: Path | None = None,
) -> Dict[str, Any]:
    length = int(selected_rows.shape[0])
    cfg = _eval_cfg(
        base_cfg,
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        max_length=int(length),
        updates_per_frame=int(updates_per_frame),
    )
    producer_cfg = dict(cfg.scheduler_stage3_0.producer or {})
    producer_cfg["enable"] = False
    scheduler = Stage23Scheduler(
        dataset=dataset,
        cfg=cfg,
        producer_cfg=producer_cfg,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=int(seed),
        fail_fast=False,
    )
    segment_row = _segment_row(scheduler, int(scene_id), int(segment_id))
    _force_rows(scheduler, segment_row=segment_row, selected_rows=selected_rows)
    scheduler.global_step = max(
        int(trigger_step),
        int(scheduler.bootstrap_cfg.get("end_step", 0)),
    )
    _reset_runtime(model)
    episode = _build_sequence_sweep_episode(
        scheduler=scheduler,
        base_episode=scheduler._build_episode(),
        rows=selected_rows,
        sequence_passes=int(sequence_passes),
    )
    causal_state, causal_rows, last_causal_out = _run_causal_episode_streaming(
        scheduler=scheduler,
        episode=episode,
        model=model,
        device=device,
        trigger_step=int(trigger_step),
    )
    final_positions = list(range(int(length)))
    final_plan = _manual_stage2_3_plan(
        scheduler=scheduler,
        episode=episode,
        rows=selected_rows,
        positions=[int(final_positions[-1])],
        repeat_budgets=[1],
        phase="final_all",
        visit_kind="final_all",
        rollout_idx=len(tuple(episode.rollouts)),
        rollouts_per_episode=len(tuple(episode.rollouts)) + 1,
        target_positions=final_positions,
        validation_render_only=True,
    )
    out = _run_plan(
        scheduler=scheduler,
        plan=final_plan,
        model=model,
        carried_state=_clone_state_for_validation(causal_state),
        mode="full",
        device=device,
        trigger_step=int(trigger_step),
        convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
    )
    stats = dict(getattr(out, "stats", {}) or {})
    frame_ids = [int(row["frame_idx"]) for row in selected_rows]
    repeat_budgets = [
        int(step.repeats_per_block)
        for rollout in tuple(episode.rollouts)
        for step in tuple(rollout.steps)
        if int(step.repeat_idx) == 0
    ]
    result = {
        "sequence_length": int(length),
        "psnr": float(stats.get("current_psnr", float("nan"))),
        "psnr_full": _full_image_psnr(out),
        "num_input_views": int(stats.get("current_num_refs", length * 3)),
        "num_rollouts": int(len(tuple(episode.rollouts))),
        "num_optimizer_updates": int(
            sum(len(tuple(rollout.steps)) for rollout in tuple(episode.rollouts))
        ),
        "mean_updates_per_frame": float(np.mean(repeat_budgets)) if repeat_budgets else 0.0,
        "configured_updates_per_frame": int(updates_per_frame),
        "sequence_passes": int(sequence_passes),
        "first_frame_id": int(frame_ids[0]),
        "last_frame_id": int(frame_ids[-1]),
        "frame_ids": frame_ids,
    }
    if diagnostics_dir is not None:
        baseline_plan = _manual_stage2_3_plan(
            scheduler=scheduler,
            episode=episode,
            rows=selected_rows,
            positions=[int(final_positions[-1])],
            repeat_budgets=[1],
            phase="final_all",
            visit_kind="final_all",
            rollout_idx=0,
            rollouts_per_episode=1,
            target_positions=final_positions,
            validation_render_only=True,
        )
        _reset_runtime(model)
        baseline_out = _run_plan(
            scheduler=scheduler,
            plan=baseline_plan,
            model=model,
            carried_state=None,
            mode="full",
            device=device,
            trigger_step=int(trigger_step),
            convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
        )
        baseline_stats = dict(getattr(baseline_out, "stats", {}) or {})
        diagnostic_frames = sorted({frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]})
        _save_diagnostic_images(
            output_dir=diagnostics_dir,
            label="optimized",
            out=out,
            selected_frames=diagnostic_frames,
        )
        _save_diagnostic_images(
            output_dir=diagnostics_dir,
            label="zero_update",
            out=baseline_out,
            selected_frames=diagnostic_frames,
        )
        result.update(
            {
                "zero_update_psnr": float(
                    baseline_stats.get("current_psnr", float("nan"))
                ),
                "zero_update_psnr_full": _full_image_psnr(baseline_out),
                "psnr_delta_vs_zero_update": float(
                    stats.get("current_psnr", float("nan"))
                    - baseline_stats.get("current_psnr", float("nan"))
                ),
                "causal_trace": causal_rows,
                "optimized_view_diagnostics": _output_diagnostics(out),
                "zero_update_view_diagnostics": _output_diagnostics(baseline_out),
                "optimized_state_diagnostics": _state_diagnostics(causal_state),
                "zero_update_state_diagnostics": _state_diagnostics(
                    getattr(baseline_out, "next_state", None)
                ),
                "last_rollout_stats": (
                    dict(getattr(last_causal_out, "stats", {}) or {})
                    if last_causal_out is not None
                    else {}
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sequence_length",
        "psnr",
        "psnr_full",
        "num_input_views",
        "num_rollouts",
        "num_optimizer_updates",
        "mean_updates_per_frame",
        "first_frame_id",
        "last_frame_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IForward input reconstruction PSNR versus short-sequence length."
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--updates_per_frame",
        type=int,
        default=1,
        help="Learned optimizer updates applied to each input frame; direct inference uses 1.",
    )
    parser.add_argument(
        "--sequence_passes",
        type=int,
        default=1,
        help="Number of chronological sweeps over the complete input sequence.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also run a zero-update baseline and save representative prediction images.",
    )
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths]
    if not lengths or any(value < 1 for value in lengths):
        raise ValueError(f"lengths must be positive, got {lengths}")
    if len(set(lengths)) != len(lengths):
        raise ValueError(f"lengths must be unique, got {lengths}")
    if int(args.updates_per_frame) < 1:
        raise ValueError("updates_per_frame must be positive")
    if int(args.sequence_passes) < 1:
        raise ValueError("sequence_passes must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = OmegaConf.load(args.config_file)
    bundle = build_iforward_runtime_from_cfg(
        cfg,
        checkpoint=str(args.checkpoint),
        device=str(args.device),
    )
    checkpoint_step = _checkpoint_step(bundle.checkpoint_payload)
    index_cfg = _eval_cfg(
        cfg,
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        max_length=max(lengths),
        updates_per_frame=int(args.updates_per_frame),
    )
    index_scheduler = Stage23Scheduler(
        dataset=bundle.dataset,
        cfg=index_cfg,
        fixed_scene_id=int(args.scene_id),
        fixed_segment_id=int(args.segment_id),
        seed=int(args.seed),
        fail_fast=False,
        producer_cfg={"enable": False},
    )
    segment_row = _segment_row(index_scheduler, int(args.scene_id), int(args.segment_id))
    all_rows = index_scheduler.index.frames_for_segment_row(segment_row)
    if int(all_rows.shape[0]) < max(lengths):
        raise ValueError(
            f"scene={args.scene_id} segment={args.segment_id} only has {all_rows.shape[0]} frames; "
            f"need {max(lengths)}"
        )
    common_rows = all_rows[: max(lengths)].copy()
    LOGGER.info(
        "testing nested chronological prefixes on scene=%d segment=%d frames=%d..%d",
        int(args.scene_id),
        int(args.segment_id),
        int(common_rows[0]["frame_idx"]),
        int(common_rows[-1]["frame_idx"]),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for length in lengths:
            row = _evaluate_length(
                base_cfg=cfg,
                dataset=bundle.dataset,
                model=bundle.model,
                device=bundle.device,
                trigger_step=int(checkpoint_step),
                scene_id=int(args.scene_id),
                segment_id=int(args.segment_id),
                selected_rows=common_rows[: int(length)].copy(),
                seed=int(args.seed),
                updates_per_frame=int(args.updates_per_frame),
                sequence_passes=int(args.sequence_passes),
                diagnostics_dir=(
                    output_dir / "diagnostics" / f"length_{int(length)}"
                    if bool(args.diagnostics)
                    else None
                ),
            )
            results.append(row)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            LOGGER.info(
                "length=%d psnr=%.4f dB full_psnr=%.4f dB updates=%d",
                int(length),
                float(row["psnr"]),
                float(row["psnr_full"]),
                int(row["num_optimizer_updates"]),
            )

    _write_csv(output_dir / "psnr_by_sequence_length.csv", results)
    manifest = {
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(checkpoint_step),
        "iforward_version": str(
            bundle.checkpoint_payload.get("iforward_version", cfg.model.iforward.version)
        ),
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "lengths": lengths,
        "frame_selection": "nested chronological prefixes of the segment's indexed frames",
        "evaluation_targets": "the input frames themselves, all configured cameras",
        "updates_per_frame": int(args.updates_per_frame),
        "sequence_passes": int(args.sequence_passes),
        "primary_metric": (
            "arithmetic mean of per-view RGB PSNR over non-sky/non-egocar pixels"
        ),
        "seed": int(args.seed),
        "device": str(bundle.device),
        "results": results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote evaluation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
