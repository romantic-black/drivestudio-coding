from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


def _install_headless_dash_comm_stub() -> None:
    """Make open3d's optional dash import safe in non-notebook CLI runs."""
    try:
        import comm  # type: ignore
    except Exception:
        return

    def _raise_import_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("dash comm disabled for headless evaluation")

    comm.create_comm = _raise_import_error  # type: ignore[attr-defined]


_install_headless_dash_comm_stub()

from datasets.validation_scheduler_v9 import (
    ValidationBlockSpecV9,
    make_phase_a_eval_rollout_batch,
    materialize_validation_v9_batch,
)
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo
from tools.train_minimal_streetforward_stage6_0_multi_scene_v9 import build_stage6_trainer_from_cfg


LOGGER = logging.getLogger("eval_phase_a_single_frame_curve")
ImageRef = Tuple[int, int]
DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _parse_budgets(values: Sequence[int]) -> List[int]:
    budgets = sorted(set(int(value) for value in values))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must contain positive integers")
    return budgets


def _checkpoint_step(payload: Dict[str, Any]) -> int:
    for key in ("step", "global_step", "iteration", "iter"):
        if payload.get(key) is not None:
            return int(payload[key])
    return 0


def _make_batch(
    *,
    dataset: Any,
    device: torch.device,
    scene_id: int,
    segment_id: int,
    frame_id: int,
    max_k: int,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
    if int(frame_id) not in set(int(value) for value in sidx.train_frame_set):
        raise ValueError(
            f"frame {frame_id} is not a train frame for scene={scene_id} segment={segment_id}; "
            f"available={list(sidx.frame_indices)}"
        )
    source_keyframe = int(sidx.frame_to_keyframe[int(frame_id)])
    refs: List[ImageRef] = [(int(frame_id), cam_id) for cam_id in range(int(sidx.num_cams))]
    spec = ValidationBlockSpecV9(
        phase="phase_A_block_local_unroll",
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        segment_choice_rank=0,
        episode_start_keyframe_pos=0,
        keyframe_window=[source_keyframe],
        frame_chain=[int(frame_id)],
        block_idx=0,
        source_keyframe_idx=source_keyframe,
        source_frame_idx=int(frame_id),
        evidence_refs=list(refs),
        block_loss_refs=list(refs),
        nearby_loss_refs=[],
        num_cams=int(sidx.num_cams),
        meta={"fixed_single_frame_eval": True},
    )
    rollout = make_phase_a_eval_rollout_batch(
        spec,
        max_K=int(max_k),
        k_values=[int(value) for value in k_values],
    )
    raw = materialize_validation_v9_batch(dataset, rollout, include_test=False)
    return convert_batch_to_minimal_format(
        raw,
        device,
        num_targets=int(raw["target"]["image"].shape[0]),
        include_source_for_2d=True,
        view_selection=None,
    )


def _curve_rows(result: Dict[str, Any], budgets: Sequence[int]) -> List[Dict[str, Any]]:
    rows = []
    for k in [0, *budgets]:
        rows.append(
            {
                "optimization_steps": int(k),
                "psnr": float(result[f"block_psnr@{int(k)}"]),
                "l1": float(result[f"block_l1@{int(k)}"]),
                "ssim_loss": float(result[f"block_ssim@{int(k)}"]),
                "valid_ratio": float(result[f"block_valid_ratio@{int(k)}"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping plot")
        return
    steps = [int(row["optimization_steps"]) for row in rows]
    psnr = [float(row["psnr"]) for row in rows]
    best_i = int(np.argmax(np.asarray(psnr)))
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(steps, psnr, marker="o", linewidth=2.0)
    axis.scatter([steps[best_i]], [psnr[best_i]], color="tab:red", zorder=4)
    axis.annotate(
        f"peak: K={steps[best_i]}, {psnr[best_i]:.2f} dB",
        (steps[best_i], psnr[best_i]),
        xytext=(8, -20),
        textcoords="offset points",
    )
    axis.set_xscale("symlog", base=2, linthresh=1)
    axis.set_xticks(steps, labels=[str(value) for value in steps])
    axis.set_xlabel("RAFT-like refinement iterations (K)")
    axis.set_ylabel("Mean-view non-sky PSNR (dB)")
    axis.set_title("Stage6 Phase-A fixed single-frame rollout")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pre-IForward Phase-A model on one fixed frame")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--frame_id", type=int, default=34)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    budgets = _parse_budgets(args.budgets)
    k_values = [0, *budgets]
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = OmegaConf.load(args.config_file)
    cfg.data.train_scene_ids = [int(args.scene_id)]
    cfg.data.eval_scene_ids = []
    device = torch.device(str(args.device))
    LOGGER.info("initializing dataset assets from %s", cfg.data.assets.root)
    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    dataset.initialize()

    LOGGER.info("building Phase-A model and loading %s", args.checkpoint)
    model = build_stage6_trainer_from_cfg(cfg, device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("model_state_dict") is None:
        raise ValueError(f"checkpoint is missing model_state_dict: {args.checkpoint}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    batch = _make_batch(
        dataset=dataset,
        device=device,
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        frame_id=int(args.frame_id),
        max_k=max(budgets),
        k_values=k_values,
    )
    result = model.validate_v9_phase_a(
        batch,
        k_values=k_values,
        max_K=max(budgets),
        mask_cfg={
            "block_loss_mask": "non_sky_non_egocar",
            "nearby_loss_mask": "non_sky_non_egocar",
            "min_valid_pixels": 1,
        },
        compute_delta_stats=True,
        compute_runtime_stats=True,
        compute_memory_stats=True,
        save_images=True,
        save_dir=str(image_dir),
        save_image_k_values=k_values,
        max_saved_cams=3,
    )
    rows = _curve_rows(result, budgets)
    _write_csv(output_dir / "psnr_by_optimization_steps.csv", rows)
    _write_plot(output_dir / "psnr_curve.png", rows)
    manifest = {
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": _checkpoint_step(payload),
        "model_stage": payload.get("model_stage"),
        "phase": payload.get("phase"),
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "frame_id": int(args.frame_id),
        "budgets": budgets,
        "budget_semantics": "one continuous fixed-frame rollout; metrics sampled after K recurrent refinements",
        "primary_metric": "arithmetic mean of per-camera PSNR over non-sky, non-egocar pixels",
        "seed": int(args.seed),
        "device": str(device),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "raw_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        LOGGER.info("K=%d psnr=%.4f", int(row["optimization_steps"]), float(row["psnr"]))
    LOGGER.info("wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()
