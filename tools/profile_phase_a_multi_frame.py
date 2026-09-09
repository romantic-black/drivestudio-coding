from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def _install_headless_dash_comm_stub() -> None:
    try:
        import comm  # type: ignore
    except Exception:
        return

    def _raise_import_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("dash comm disabled for headless evaluation")

    comm.create_comm = _raise_import_error  # type: ignore[attr-defined]


_install_headless_dash_comm_stub()

from tools.eval_phase_a_single_frame_curve import _make_batch
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo
from tools.train_minimal_streetforward_stage6_0_multi_scene_v9 import build_stage6_trainer_from_cfg


LOGGER = logging.getLogger("profile_phase_a_multi_frame")
FrameSpec = Tuple[int, int, int]


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _chronological_unique_frames(dataset: Any, scene_id: int, count: int) -> List[FrameSpec]:
    seen: set[int] = set()
    specs: List[FrameSpec] = []
    for segment_id in sorted(int(x) for x in dataset.list_segment_ids(int(scene_id))):
        sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
        for frame_id in sorted(int(x) for x in sidx.frame_indices):
            if frame_id in seen or frame_id not in sidx.train_frame_set:
                continue
            seen.add(frame_id)
            specs.append((int(scene_id), int(segment_id), int(frame_id)))
            if len(specs) >= int(count):
                return specs
    raise ValueError(f"scene={scene_id} only exposes {len(specs)} unique train frames; requested {count}")


def _summary(rows: Sequence[Dict[str, Any]], count: int, wall_ms: float) -> Dict[str, Any]:
    selected = list(rows[: int(count)])
    psnr = [float(row["psnr"]) for row in selected]
    model_ms = [float(row["model_ms"]) for row in selected]
    e2e_ms = [float(row["e2e_ms"]) for row in selected]
    return {
        "num_frames": int(count),
        "num_camera_images": int(count) * 3,
        "psnr_median": float(np.median(psnr)),
        "psnr_p10": _percentile(psnr, 10),
        "psnr_p90": _percentile(psnr, 90),
        "model_latency_ms_median": float(np.median(model_ms)),
        "model_latency_ms_p90": _percentile(model_ms, 90),
        "e2e_latency_ms_median": float(np.median(e2e_ms)),
        "e2e_latency_ms_p90": _percentile(e2e_ms, 90),
        "measured_wall_ms": float(wall_ms),
        "throughput_fps": float(int(count) * 1000.0 / max(float(wall_ms), 1.0e-9)),
        "peak_allocated_gib": max(float(row["peak_allocated_gib"]) for row in selected),
        "peak_reserved_gib": max(float(row["peak_reserved_gib"]) for row in selected),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile independent Phase-A K-step refinement over 25/50 frames")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--counts", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = sorted(set(int(x) for x in args.counts))
    if not counts or counts[0] < 1:
        raise ValueError("counts must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config_file)
    cfg.data.train_scene_ids = [int(args.scene_id)]
    cfg.data.eval_scene_ids = []
    device = torch.device(str(args.device))
    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    dataset.initialize()
    frame_specs = _chronological_unique_frames(dataset, int(args.scene_id), max(counts))

    model = build_stage6_trainer_from_cfg(cfg, device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("model_state_dict") is None:
        raise ValueError("checkpoint is missing model_state_dict")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    warm_scene, warm_segment, warm_frame = frame_specs[0]
    warm_batch = _make_batch(
        dataset=dataset,
        device=device,
        scene_id=warm_scene,
        segment_id=warm_segment,
        frame_id=warm_frame,
        max_k=int(args.iterations),
        k_values=[0, int(args.iterations)],
    )
    model.validate_v9_phase_a(
        warm_batch,
        k_values=[0, int(args.iterations)],
        max_K=int(args.iterations),
        mask_cfg={
            "block_loss_mask": "non_sky_non_egocar",
            "nearby_loss_mask": "non_sky_non_egocar",
            "min_valid_pixels": 1,
        },
        compute_delta_stats=True,
        compute_runtime_stats=False,
        compute_memory_stats=False,
        save_images=False,
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    rows: List[Dict[str, Any]] = []
    prefix_wall_ms: Dict[int, float] = {}
    all_t0 = time.perf_counter()
    for index, (scene_id, segment_id, frame_id) in enumerate(frame_specs, start=1):
        frame_t0 = time.perf_counter()
        batch = _make_batch(
            dataset=dataset,
            device=device,
            scene_id=scene_id,
            segment_id=segment_id,
            frame_id=frame_id,
            max_k=int(args.iterations),
            k_values=[0, int(args.iterations)],
        )
        torch.cuda.synchronize(device)
        model_t0 = time.perf_counter()
        result = model.validate_v9_phase_a(
            batch,
            k_values=[0, int(args.iterations)],
            max_K=int(args.iterations),
            mask_cfg={
                "block_loss_mask": "non_sky_non_egocar",
                "nearby_loss_mask": "non_sky_non_egocar",
                "min_valid_pixels": 1,
            },
            compute_delta_stats=True,
            compute_runtime_stats=False,
            compute_memory_stats=False,
            save_images=False,
        )
        torch.cuda.synchronize(device)
        model_ms = float((time.perf_counter() - model_t0) * 1000.0)
        e2e_ms = float((time.perf_counter() - frame_t0) * 1000.0)
        row = {
            "index": int(index),
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "frame_id": int(frame_id),
            "iterations": int(args.iterations),
            "psnr": float(result[f"block_psnr@{int(args.iterations)}"]),
            "model_ms": model_ms,
            "e2e_ms": e2e_ms,
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / (1024.0**3)),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / (1024.0**3)),
        }
        rows.append(row)
        if index in counts:
            prefix_wall_ms[index] = float((time.perf_counter() - all_t0) * 1000.0)
        LOGGER.info(
            "frame=%d/%d scene=%d segment=%d frame_id=%d psnr=%.4f model_ms=%.1f peak_alloc=%.3fGiB",
            index,
            max(counts),
            scene_id,
            segment_id,
            frame_id,
            row["psnr"],
            model_ms,
            row["peak_allocated_gib"],
        )

    summaries = [_summary(rows, count, prefix_wall_ms[count]) for count in counts]
    _write_csv(output_dir / "per_frame.csv", rows)
    _write_csv(output_dir / "summary.csv", summaries)
    manifest = {
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(payload.get("step", payload.get("global_step", 0))),
        "scene_id": int(args.scene_id),
        "counts": counts,
        "iterations_per_frame": int(args.iterations),
        "num_cameras_per_frame": 3,
        "semantics": "independent Phase-A local rollout per frame; no cross-frame persistent state",
        "frame_selection": "first N chronological unique train frames; first segment owning each overlap frame",
        "warmup": "one unreported K-step frame before CUDA peak and latency measurement",
        "device": str(device),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("summaries=%s", summaries)


if __name__ == "__main__":
    main()
