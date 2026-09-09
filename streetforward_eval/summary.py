from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals) / len(vals))


def _row_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _is_metric_group(row: Dict[str, Any], group: str) -> bool:
    row_group = row.get("metric_group")
    if row_group is not None:
        return str(row_group) == str(group)
    # Backward-compatible fallback for rows produced before metric_group existed.
    is_input_any = row.get("is_input_frame")
    if isinstance(is_input_any, str):
        is_input = str(is_input_any).strip().lower() in ("1", "true", "yes")
    elif is_input_any is None:
        is_input = str(row.get("frame_group")) == "input"
    else:
        is_input = bool(is_input_any)
    return bool(is_input) if str(group) == "reconstruction" else not bool(is_input)


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    return _mean([_row_float(r, key) for r in rows])


def _mean_key_for_group(rows: List[Dict[str, Any]], key: str, group: str) -> float:
    return _mean([_row_float(r, key) for r in rows if _is_metric_group(r, group)])


def build_summary_rows(final_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(final_rows) == 0:
        return []
    by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for row in final_rows:
        key_parts = [
            str(row.get("exp_name", "")),
            str(row.get("checkpoint", row.get("checkpoint_tag", ""))),
            str(row.get("variant", "")),
            str(row.get("input_count_label", row.get("input_count", ""))),
            str(row.get("train_block_size_label", "")),
        ]
        by_exp.setdefault("|".join(key_parts), []).append(row)

    out: List[Dict[str, Any]] = []
    for _, rows in by_exp.items():
        first = rows[0]
        reconstruction_rows = [r for r in rows if _is_metric_group(r, "reconstruction")]
        nvs_rows = [r for r in rows if _is_metric_group(r, "nvs")]
        out.append(
            {
                "exp_name": str(first.get("exp_name", "")),
                "checkpoint": str(first.get("checkpoint", first.get("checkpoint_tag", ""))),
                "variant": str(first.get("variant", "")),
                "input_count_label": str(first.get("input_count_label", first.get("input_count", ""))),
                "train_block_size_label": str(first.get("train_block_size_label", "")),
                "num_episodes": int(len(set(str(r["episode_uid"]) for r in rows))),
                "num_views": int(len(rows)),
                "num_views_reconstruction": int(len(reconstruction_rows)),
                "num_views_nvs": int(len(nvs_rows)),
                "mean_psnr": _mean_key(rows, "psnr"),
                "mean_l1": _mean_key(rows, "l1"),
                "mean_ssim": _mean_key(rows, "ssim"),
                "mean_ssim_non_sky": _mean_key(rows, "ssim_non_sky"),
                "mean_lpips": _mean_key(rows, "lpips"),
                "mean_lpips_non_sky": _mean_key(rows, "lpips_non_sky"),
                "mean_psnr_reconstruction": _mean_key_for_group(rows, "psnr", "reconstruction"),
                "mean_psnr_nvs": _mean_key_for_group(rows, "psnr", "nvs"),
                "mean_l1_reconstruction": _mean_key_for_group(rows, "l1", "reconstruction"),
                "mean_l1_nvs": _mean_key_for_group(rows, "l1", "nvs"),
                "mean_ssim_reconstruction": _mean_key_for_group(rows, "ssim", "reconstruction"),
                "mean_ssim_nvs": _mean_key_for_group(rows, "ssim", "nvs"),
                "mean_ssim_non_sky_reconstruction": _mean_key_for_group(
                    rows, "ssim_non_sky", "reconstruction"
                ),
                "mean_ssim_non_sky_nvs": _mean_key_for_group(rows, "ssim_non_sky", "nvs"),
                "mean_lpips_reconstruction": _mean_key_for_group(rows, "lpips", "reconstruction"),
                "mean_lpips_nvs": _mean_key_for_group(rows, "lpips", "nvs"),
                "mean_lpips_non_sky_reconstruction": _mean_key_for_group(
                    rows, "lpips_non_sky", "reconstruction"
                ),
                "mean_lpips_non_sky_nvs": _mean_key_for_group(rows, "lpips_non_sky", "nvs"),
                "mean_psnr_input_frames": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("frame_group")) == "input"]
                ),
                "mean_psnr_interp_frames": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("frame_group")) == "interp"]
                ),
                "mean_psnr_extrap_frames": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("frame_group")) == "extrap"]
                ),
                "mean_psnr_front": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("cam_name")) == "front"]
                ),
                "mean_psnr_front_left": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("cam_name")) == "front_left"]
                ),
                "mean_psnr_front_right": _mean(
                    [float(r["psnr"]) for r in rows if str(r.get("cam_name")) == "front_right"]
                ),
            }
        )
    return out


def build_optimization_curve_rows(iter_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate PSNR by completed optimization count.

    Each output row represents the same cumulative optimizer trajectory at one
    reporting point. Camera columns are included when those camera names are
    present; ``mean_psnr`` is the mean over all evaluated views/episodes.
    """
    post_update_rows = [r for r in iter_rows if not bool(r.get("is_pre_update", False))]
    by_iteration: Dict[int, List[Dict[str, Any]]] = {}
    for row in post_update_rows:
        by_iteration.setdefault(int(row["global_iter"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for iteration in sorted(by_iteration):
        rows = by_iteration[iteration]
        out.append(
            {
                "optimization_steps": int(iteration),
                "num_episodes": int(len(set(str(r["episode_uid"]) for r in rows))),
                "num_views": int(len(rows)),
                "mean_psnr": _mean_key(rows, "psnr"),
                "mean_psnr_non_sky": _mean_key(rows, "psnr_non_sky"),
                "mean_psnr_full": _mean_key(rows, "psnr_full"),
                "psnr_front": _mean(
                    [_row_float(r, "psnr") for r in rows if str(r.get("cam_name")) == "front"]
                ),
                "psnr_front_left": _mean(
                    [_row_float(r, "psnr") for r in rows if str(r.get("cam_name")) == "front_left"]
                ),
                "psnr_front_right": _mean(
                    [_row_float(r, "psnr") for r in rows if str(r.get("cam_name")) == "front_right"]
                ),
            }
        )
    return out


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
