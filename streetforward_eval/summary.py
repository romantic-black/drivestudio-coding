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
        out.append(
            {
                "exp_name": str(first.get("exp_name", "")),
                "checkpoint": str(first.get("checkpoint", first.get("checkpoint_tag", ""))),
                "variant": str(first.get("variant", "")),
                "input_count_label": str(first.get("input_count_label", first.get("input_count", ""))),
                "train_block_size_label": str(first.get("train_block_size_label", "")),
                "num_episodes": int(len(set(str(r["episode_uid"]) for r in rows))),
                "num_views": int(len(rows)),
                "mean_psnr": _mean([float(r["psnr"]) for r in rows]),
                "mean_l1": _mean([float(r["l1"]) for r in rows]),
                "mean_ssim": _mean([float(r.get("ssim", float("nan"))) for r in rows]),
                "mean_ssim_non_sky": _mean([float(r.get("ssim_non_sky", float("nan"))) for r in rows]),
                "mean_lpips": _mean([float(r["lpips"]) for r in rows]),
                "mean_lpips_non_sky": _mean([float(r.get("lpips_non_sky", float("nan"))) for r in rows]),
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
