from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.eval_sparse_scheduler import (
    EvalSparseEpisodeSpec,
    EvalSparseStepSpec,
    build_eval_sparse_episode_specs,
    build_eval_sparse_steps,
)
from datasets.multi_scene_dataset_v4 import BatchRequestV4
from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import (
    _compute_masked_lpips,
    _compute_masked_psnr,
    _compute_masked_ssim,
)
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4

logger = logging.getLogger("eval_sparse_scheduler")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_total_episodes", type=int, default=None)
    return p.parse_args()


def load_cfg(config_file: str) -> Any:
    cfg = OmegaConf.load(config_file)
    base_config_file = cfg.get("base_config_file")
    if base_config_file is None:
        return cfg
    base_cfg = OmegaConf.load(str(base_config_file))
    merged = OmegaConf.merge(base_cfg, cfg)
    return merged


def to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
    y = torch.clamp(x.detach().float().cpu(), 0.0, 1.0).numpy()
    y = (y * 255.0).round().astype(np.uint8)
    return y


def _save_tensor_image(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8_image(x)
    Image.fromarray(arr).save(str(path))


def _save_error_image(path: Path, pred: torch.Tensor, gt: torch.Tensor) -> None:
    err = torch.abs(torch.clamp(pred, 0.0, 1.0) - torch.clamp(gt, 0.0, 1.0)).mean(dim=-1)
    err = torch.clamp(err * 4.0, 0.0, 1.0)
    err_rgb = torch.stack([err, err, err], dim=-1)
    _save_tensor_image(path, err_rgb)


def _safe_mean(vals: List[float]) -> float:
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def _category_from_offset(offset: int, input_offsets: List[int]) -> str:
    return "input_reconstruction" if int(offset) in set(int(x) for x in input_offsets) else "nvs_intermediate"


def load_model(cfg: Any, ckpt_path: str, device: torch.device) -> MinimalStreetForwardStage5_3_Production:
    model = MinimalStreetForwardStage5_3_Production(cfg, device=device).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("loaded ckpt=%s missing=%d unexpected=%d", ckpt_path, len(missing), len(unexpected))
    if len(missing) > 0:
        logger.warning("missing keys sample: %s", missing[:20])
    if len(unexpected) > 0:
        logger.warning("unexpected keys sample: %s", unexpected[:20])
    model.eval()
    return model


def make_update_batch(dataset: Any, episode: EvalSparseEpisodeSpec, step: EvalSparseStepSpec, device: torch.device) -> Dict[str, Any]:
    if len(step.target_image_refs) != len(step.target_image_roles):
        raise ValueError(
            f"target_image_refs/target_image_roles mismatch: {len(step.target_image_refs)} vs {len(step.target_image_roles)}"
        )

    raw = dataset.get_segment_batch_from_image_refs(
        BatchRequestV4(
            scene_id=int(episode.scene_id),
            segment_id=int(episode.segment_id),
            source_image_ref=tuple(step.source_image_refs[0]),
            source_image_refs=[tuple(x) for x in step.source_image_refs],
            target_image_refs=[tuple(x) for x in step.target_image_refs],
            include_test=False,
        ),
        enforce_target0_equals_source=False,
    )

    rm = dict(raw.get("request_meta") or {})
    rm.update(
        {
            "scheduler_version": "eval_sparse",
            "eval_sparse/episode_idx": int(episode.episode_idx),
            "eval_sparse/step_idx": int(step.step_idx),
            "eval_sparse/source_offset": int(step.source_offset),
            "eval_sparse/source_frame": int(step.source_frame),
            "eval_sparse/input_frames": [int(x) for x in episode.input_frames],
            "eval_sparse/frames20": [int(x) for x in episode.frames20],
            "target_frame_indices": [int(x) for x in step.target_frames],
            "target_frame_roles": [str(x) for x in step.target_frame_roles],
            "target_image_roles": [str(x) for x in step.target_image_roles],
            "eval_sparse/data_mode": "segment_finetune_train",
        }
    )
    raw["request_meta"] = rm
    aligned = {
        "scheduler_version": "eval_sparse",
        "scene_id": int(episode.scene_id),
        "segment_id": int(episode.segment_id),
        "episode_idx_global": int(episode.episode_idx),
        "block_idx_in_episode": int(step.step_idx),
        "block_idx_in_segment": int(step.step_idx),
        "block_idx_global": int(step.step_idx),
        "source_frame_idx": int(step.source_frame),
        "target_frame_indices": [int(x) for x in step.target_frames],
        "target_frame_roles": [str(x) for x in step.target_frame_roles],
    }
    raw["_scheduler_v4_aligned_info"] = dict(aligned)
    raw["_scheduler_v7_aligned_info"] = dict(aligned)
    raw["_scheduler_v8_aligned_info"] = dict(aligned)
    return convert_batch_to_minimal_format(
        raw,
        device,
        num_targets=None,
        include_source_for_2d=True,
    )


def compute_eval_sparse_metrics(
    *,
    render_out: Dict[str, Any],
    episode: EvalSparseEpisodeSpec,
    escfg: Any,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    save_images: bool,
    save_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    per_image_rows: List[Dict[str, Any]] = []

    input_offsets = [int(x) for x in episode.input_offsets]
    frame_to_offset = {int(f): i for i, f in enumerate(episode.frames20)}

    use_sky = bool(escfg.metrics.use_sky_mask_regions)
    require_sky = bool(escfg.metrics.require_sky_mask)
    min_valid = int(escfg.metrics.min_valid_pixels_per_region)

    for row in render_out.get("rows", []):
        pred = row.get("pred_rgb")
        gt = row.get("gt_image")
        if pred is None or gt is None:
            continue
        frame_idx = int(row["frame_idx"])
        cam_idx = int(row["cam_idx"])
        offset = int(frame_to_offset[frame_idx])
        role = _category_from_offset(offset, input_offsets)

        sky_mask = row.get("sky_mask")
        valid_mask = None
        if use_sky:
            if sky_mask is None and require_sky:
                raise ValueError("eval_sparse metrics require sky_mask but target is missing sky_mask")
            if sky_mask is not None:
                valid_mask = (sky_mask <= 0.5).float()
                if float(valid_mask.sum().item()) < float(min_valid):
                    continue

        if valid_mask is None:
            valid_mask = torch.ones((int(gt.shape[0]), int(gt.shape[1])), dtype=torch.float32, device=gt.device)

        psnr = _compute_masked_psnr(pred, gt, valid_mask)
        ssim = _compute_masked_ssim(pred, gt, valid_mask)
        lpips = _compute_masked_lpips(pred, gt, valid_mask, lpips_metric)
        if psnr is None or ssim is None or lpips is None:
            continue

        item = {
            "frame_idx": int(frame_idx),
            "offset": int(offset),
            "cam_id": int(cam_idx),
            "role": str(role),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips),
            "valid_pixels": int(valid_mask.sum().item()),
            "is_input_frame": bool(role == "input_reconstruction"),
        }
        per_image_rows.append(item)

        if save_images:
            stem = f"frame_{frame_idx:03d}_offset_{offset:02d}_cam_{cam_idx}"
            _save_tensor_image(save_dir / "renders" / f"{stem}_pred.png", pred)
            _save_tensor_image(save_dir / "renders" / f"{stem}_gt.png", gt)
            _save_error_image(save_dir / "renders" / f"{stem}_err.png", pred, gt)

    def _agg(rows: List[Dict[str, Any]], key: str) -> float:
        return _safe_mean([float(r[key]) for r in rows])

    all_rows = per_image_rows
    input_rows = [r for r in per_image_rows if str(r["role"]) == "input_reconstruction"]
    nvs_rows = [r for r in per_image_rows if str(r["role"]) == "nvs_intermediate"]
    metrics = {
        "psnr/all": _agg(all_rows, "psnr"),
        "psnr/input_reconstruction": _agg(input_rows, "psnr"),
        "psnr/nvs_intermediate": _agg(nvs_rows, "psnr"),
        "ssim/all": _agg(all_rows, "ssim"),
        "ssim/input_reconstruction": _agg(input_rows, "ssim"),
        "ssim/nvs_intermediate": _agg(nvs_rows, "ssim"),
        "lpips/all": _agg(all_rows, "lpips"),
        "lpips/input_reconstruction": _agg(input_rows, "lpips"),
        "lpips/nvs_intermediate": _agg(nvs_rows, "lpips"),
    }
    return per_image_rows, metrics


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


@torch.no_grad()
def run_one_episode(
    *,
    model: MinimalStreetForwardStage5_3_Production,
    dataset: Any,
    cfg: Any,
    episode: EvalSparseEpisodeSpec,
    out_dir: Path,
    device: torch.device,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    escfg = cfg.eval_sparse_scheduler
    steps = build_eval_sparse_steps(
        episode=episode,
        total_target_frames=int(escfg.update.total_target_frames),
        include_source_frame=bool(escfg.update.include_source_frame),
        history_order=str(escfg.update.history_order),
    )
    if len(steps) == 0:
        raise ValueError("No sparse update steps generated")

    init_batch = make_update_batch(dataset, episode, steps[0], device)
    model.reset_for_segment_eval(init_batch)

    per_step_logs: List[Dict[str, Any]] = []
    for step in steps:
        update_batch = make_update_batch(dataset, episode, step, device)
        step_loss_values: List[float] = []
        for local_iter in range(int(escfg.update.steps_per_input)):
            out = model.eval_sparse_update_step(
                update_batch,
                local_iter=int(local_iter),
                num_local_iters=int(escfg.update.steps_per_input),
            )
            step_loss_values.append(float(out.get("loss", 0.0)))

        if bool(escfg.update.record_history):
            model.eval_sparse_record_history(update_batch)

        step_log = {
            "episode_idx": int(episode.episode_idx),
            "step_idx": int(step.step_idx),
            "source_offset": int(step.source_offset),
            "source_frame": int(step.source_frame),
            "target_frames": [int(x) for x in step.target_frames],
            "num_target_images": int(len(step.target_image_refs)),
            "num_source_images": int(len(step.source_image_refs)),
            "steps_per_input": int(escfg.update.steps_per_input),
            "loss_first": float(step_loss_values[0]),
            "loss_last": float(step_loss_values[-1]),
            "loss_mean": float(np.mean(step_loss_values)),
        }
        per_step_logs.append(step_log)
        logger.info(
            "[eval_sparse][step] ep=%04d step=%d src=%s targets=%s source_imgs=%d target_imgs=%d loss %.4f -> %.4f",
            int(episode.episode_idx),
            int(step.step_idx),
            int(step.source_frame),
            [int(x) for x in step.target_frames],
            int(len(step.source_image_refs)),
            int(len(step.target_image_refs)),
            float(step_log["loss_first"]),
            float(step_log["loss_last"]),
        )

    episode_dir = out_dir / "episodes" / f"episode_{int(episode.episode_idx):04d}"
    render_out = model.eval_sparse_render_frames(
        scene_id=int(episode.scene_id),
        segment_id=int(episode.segment_id),
        image_refs=[tuple(x) for x in episode.eval_image_refs],
        camera_ids=[int(x) for x in episode.camera_ids],
        save_dir=episode_dir if bool(escfg.render.save_images) else None,
    )
    logger.info(
        "[eval_sparse][render] ep=%04d render_images=%d frames=%d cams=%d save=%s",
        int(episode.episode_idx),
        int(render_out.get("num_images", 0)),
        int(len(episode.eval_frames)),
        int(len(episode.camera_ids)),
        bool(escfg.render.save_images),
    )

    per_image_rows, metrics = compute_eval_sparse_metrics(
        render_out=render_out,
        episode=episode,
        escfg=escfg,
        lpips_metric=lpips_metric,
        save_images=bool(escfg.render.save_images),
        save_dir=episode_dir,
    )

    with open(episode_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "episode_idx": int(episode.episode_idx),
                "scene_id": int(episode.scene_id),
                "segment_id": int(episode.segment_id),
                "frames20": [int(x) for x in episode.frames20],
                "input_offsets": [int(x) for x in episode.input_offsets],
                "input_frames": [int(x) for x in episode.input_frames],
                "camera_ids": [int(x) for x in episode.camera_ids],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(episode_dir / "update_steps.json", "w", encoding="utf-8") as f:
        json.dump(per_step_logs, f, ensure_ascii=False, indent=2)
    with open(episode_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    record = {
        "episode_idx": int(episode.episode_idx),
        "scene_id": int(episode.scene_id),
        "segment_id": int(episode.segment_id),
        "window_start_pos": int(episode.window_start_pos),
        "frames20": [int(x) for x in episode.frames20],
        "input_offsets": [int(x) for x in episode.input_offsets],
        "input_frames": [int(x) for x in episode.input_frames],
        "camera_ids": [int(x) for x in episode.camera_ids],
        "steps": per_step_logs,
        "metrics": metrics,
    }
    return record, per_image_rows


def summarize_eval_sparse_records(
    *,
    all_records: List[Dict[str, Any]],
    per_image_rows: List[Dict[str, Any]],
    escfg: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_cam: Dict[int, List[Dict[str, Any]]] = {}
    by_offset: Dict[int, List[Dict[str, Any]]] = {}
    for row in per_image_rows:
        by_cam.setdefault(int(row["cam_id"]), []).append(row)
        by_offset.setdefault(int(row["offset"]), []).append(row)

    per_camera_rows: List[Dict[str, Any]] = []
    for cam_id in sorted(by_cam.keys()):
        rows = by_cam[cam_id]
        per_camera_rows.append(
            {
                "cam_id": int(cam_id),
                "num_images": int(len(rows)),
                "psnr": _safe_mean([float(r["psnr"]) for r in rows]),
                "ssim": _safe_mean([float(r["ssim"]) for r in rows]),
                "lpips": _safe_mean([float(r["lpips"]) for r in rows]),
            }
        )

    per_offset_rows: List[Dict[str, Any]] = []
    for offset in sorted(by_offset.keys()):
        rows = by_offset[offset]
        per_offset_rows.append(
            {
                "offset": int(offset),
                "num_images": int(len(rows)),
                "role": _category_from_offset(int(offset), [int(x) for x in escfg.protocol.input_offsets]),
                "psnr": _safe_mean([float(r["psnr"]) for r in rows]),
                "ssim": _safe_mean([float(r["ssim"]) for r in rows]),
                "lpips": _safe_mean([float(r["lpips"]) for r in rows]),
            }
        )

    input_rows = [r for r in per_image_rows if bool(r["is_input_frame"])]
    nvs_rows = [r for r in per_image_rows if not bool(r["is_input_frame"])]
    summary = {
        "protocol": "eval_sparse_scheduler",
        "data_mode": str(escfg.data_mode),
        "sequence_length": int(escfg.protocol.sequence_length),
        "input_offsets": [int(x) for x in escfg.protocol.input_offsets],
        "camera_ids": [int(x) for x in escfg.protocol.camera_ids],
        "num_episodes": int(len(all_records)),
        "num_images_all": int(len(per_image_rows)),
        "num_images_input": int(len(input_rows)),
        "num_images_nvs": int(len(nvs_rows)),
        "psnr/all": _safe_mean([float(r["psnr"]) for r in per_image_rows]),
        "psnr/input_reconstruction": _safe_mean([float(r["psnr"]) for r in input_rows]),
        "psnr/nvs_intermediate": _safe_mean([float(r["psnr"]) for r in nvs_rows]),
        "ssim/all": _safe_mean([float(r["ssim"]) for r in per_image_rows]),
        "ssim/input_reconstruction": _safe_mean([float(r["ssim"]) for r in input_rows]),
        "ssim/nvs_intermediate": _safe_mean([float(r["ssim"]) for r in nvs_rows]),
        "lpips/all": _safe_mean([float(r["lpips"]) for r in per_image_rows]),
        "lpips/input_reconstruction": _safe_mean([float(r["lpips"]) for r in input_rows]),
        "lpips/nvs_intermediate": _safe_mean([float(r["lpips"]) for r in nvs_rows]),
    }
    return summary, per_offset_rows, per_camera_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = load_cfg(args.config_file)
    escfg = cfg.eval_sparse_scheduler
    if not bool(escfg.enable):
        raise ValueError("eval_sparse_scheduler.enable=false")
    if bool(escfg.participate_in_training):
        raise ValueError("eval_sparse_scheduler should not participate in formal training.")
    if str(escfg.data_mode) != "segment_finetune_train":
        raise ValueError("eval_sparse_scheduler.data_mode must be segment_finetune_train")
    if str(escfg.protocol.eval_offsets) != "all":
        raise ValueError("eval_sparse_scheduler.protocol.eval_offsets must be 'all'")
    if not bool(escfg.render.eval_all_20_frames):
        raise ValueError("eval_sparse_scheduler.render.eval_all_20_frames must be true")
    if str(escfg.update.target_policy) != "source_plus_recent_history":
        raise ValueError("eval_sparse_scheduler.update.target_policy must be source_plus_recent_history")
    if bool(escfg.update.allow_future_input_targets):
        raise ValueError("eval_sparse_scheduler.update.allow_future_input_targets must be false")
    if int(escfg.update.steps_per_input) < 1:
        raise ValueError("eval_sparse_scheduler.update.steps_per_input must be >= 1")
    if str(escfg.update.record_history_on) != "block_exit":
        raise ValueError("eval_sparse_scheduler.update.record_history_on must be block_exit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_multi_scene_dataset_v4(cfg, device)
    dataset.initialize()

    model = load_model(cfg, args.ckpt, device)
    model.bind_eval_dataset(dataset)

    out_dir = Path(args.output_dir or str(escfg.render.save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    logger.info("[eval_sparse][protocol]")
    logger.info("data_mode=%s", str(escfg.data_mode))
    logger.info("update_target_policy=%s", str(escfg.update.target_policy))
    logger.info("allow_future_input_targets=%s", bool(escfg.update.allow_future_input_targets))
    logger.info("eval_frames=all_20")
    logger.info("history_record_views=source_image_refs")
    logger.info("record_history_on=%s", str(escfg.update.record_history_on))
    logger.info("steps_per_input=%d", int(escfg.update.steps_per_input))

    scene_ids = [int(x) for x in list(escfg.episode_selection.scene_ids)]
    max_total_episodes = (
        int(args.max_total_episodes)
        if args.max_total_episodes is not None
        else (
            int(escfg.episode_selection.max_total_episodes)
            if escfg.episode_selection.max_total_episodes is not None
            else None
        )
    )
    episodes = build_eval_sparse_episode_specs(
        dataset=dataset,
        scene_ids=scene_ids,
        sequence_length=int(escfg.protocol.sequence_length),
        input_offsets=[int(x) for x in escfg.protocol.input_offsets],
        camera_ids=[int(x) for x in escfg.protocol.camera_ids],
        window_policy=str(escfg.episode_selection.window_policy),
        stride=int(escfg.episode_selection.stride),
        max_episodes_per_scene=(
            int(escfg.episode_selection.max_episodes_per_scene)
            if escfg.episode_selection.max_episodes_per_scene is not None
            else None
        ),
        max_total_episodes=max_total_episodes,
    )
    logger.info(
        "[eval_sparse][begin] episodes=%d scene_ids=%s sequence_length=%d input_offsets=%s cams=%s",
        len(episodes),
        scene_ids,
        int(escfg.protocol.sequence_length),
        [int(x) for x in escfg.protocol.input_offsets],
        [int(x) for x in escfg.protocol.camera_ids],
    )

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    lpips_metric.eval()

    all_records: List[Dict[str, Any]] = []
    all_per_image_rows: List[Dict[str, Any]] = []
    jsonl_path = out_dir / "episodes.jsonl"
    with open(jsonl_path, "w", encoding="utf-8"):
        pass

    for episode in episodes:
        logger.info(
            "[eval_sparse][episode_begin] ep=%04d scene=%d seg=%d frames=%s..%s inputs=%s",
            int(episode.episode_idx),
            int(episode.scene_id),
            int(episode.segment_id),
            int(episode.frames20[0]),
            int(episode.frames20[-1]),
            [int(x) for x in episode.input_frames],
        )
        rec, per_image_rows = run_one_episode(
            model=model,
            dataset=dataset,
            cfg=cfg,
            episode=episode,
            out_dir=out_dir,
            device=device,
            lpips_metric=lpips_metric,
        )
        all_records.append(rec)
        for row in per_image_rows:
            row_out = dict(row)
            row_out["episode_idx"] = int(rec["episode_idx"])
            row_out["scene_id"] = int(rec["scene_id"])
            row_out["segment_id"] = int(rec["segment_id"])
            row_out["window_start_pos"] = int(rec["window_start_pos"])
            all_per_image_rows.append(row_out)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            "[eval_sparse][episode_end] ep=%04d psnr_all=%.3f psnr_input=%.3f psnr_nvs=%.3f ssim_all=%.3f",
            int(rec["episode_idx"]),
            float(rec["metrics"]["psnr/all"]),
            float(rec["metrics"]["psnr/input_reconstruction"]),
            float(rec["metrics"]["psnr/nvs_intermediate"]),
            float(rec["metrics"]["ssim/all"]),
        )

    summary, per_offset_rows, per_camera_rows = summarize_eval_sparse_records(
        all_records=all_records,
        per_image_rows=all_per_image_rows,
        escfg=escfg,
    )
    if len(all_per_image_rows) == 0:
        raise RuntimeError("eval_sparse_scheduler produced zero valid metric rows.")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(
        out_dir / "per_image_metrics.csv",
        all_per_image_rows,
        fieldnames=[
            "episode_idx",
            "scene_id",
            "segment_id",
            "window_start_pos",
            "frame_idx",
            "offset",
            "cam_id",
            "role",
            "psnr",
            "ssim",
            "lpips",
            "valid_pixels",
            "is_input_frame",
        ],
    )
    write_csv(
        out_dir / "per_offset_metrics.csv",
        per_offset_rows,
        fieldnames=["offset", "num_images", "role", "psnr", "ssim", "lpips"],
    )
    write_csv(
        out_dir / "per_camera_metrics.csv",
        per_camera_rows,
        fieldnames=["cam_id", "num_images", "psnr", "ssim", "lpips"],
    )

    logger.info("[eval_sparse][done] %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

