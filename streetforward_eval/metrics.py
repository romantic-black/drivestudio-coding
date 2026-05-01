from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .episode_builder import TestEpisodeSpec, classify_eval_frame
from .protocols import TestProtocolSpec

logger = logging.getLogger(__name__)


def _safe_float(v: Optional[float]) -> float:
    if v is None:
        return float("nan")
    return float(v)


def _to_hwc_rgb(x: torch.Tensor) -> torch.Tensor:
    t = x.detach().float()
    if t.dim() == 4:
        if int(t.shape[0]) != 1:
            raise ValueError(f"expected batch size 1 for RGB image tensor, got shape={tuple(t.shape)}")
        t = t.squeeze(0)
    if t.dim() != 3:
        raise ValueError(f"expected 3D RGB tensor, got shape={tuple(t.shape)}")
    if int(t.shape[-1]) == 3:
        return t
    if int(t.shape[0]) == 3:
        return t.permute(1, 2, 0)
    raise ValueError(f"expected HWC or CHW RGB tensor, got shape={tuple(t.shape)}")


def _to_hw_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    t = mask.detach().float()
    if t.dim() == 4:
        if int(t.shape[0]) != 1:
            raise ValueError(f"expected batch size 1 for mask tensor, got shape={tuple(t.shape)}")
        t = t.squeeze(0)
    if t.dim() == 3 and int(t.shape[-1]) == 1:
        t = t.squeeze(-1)
    if t.dim() == 3 and int(t.shape[0]) == 1:
        t = t.squeeze(0)
    if t.dim() != 2:
        raise ValueError(f"expected 2D mask tensor, got shape={tuple(t.shape)}")
    return t


def _masked_metrics(
    *,
    pred: torch.Tensor,
    gt: torch.Tensor,
    sky_mask: Optional[torch.Tensor],
    egocar_mask: Optional[torch.Tensor],
    primary_mask: str,
    min_valid_pixels: int,
) -> Dict[str, Optional[float]]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")
    if pred.dim() != 3 or int(pred.shape[-1]) != 3:
        raise ValueError(f"expected HWC RGB tensors, got pred shape={tuple(pred.shape)}")

    h = int(pred.shape[0])
    w = int(pred.shape[1])
    valid = torch.ones((h, w), dtype=torch.float32, device=pred.device)
    use_non_sky = "non_sky" in str(primary_mask)
    use_non_ego = "non_ego" in str(primary_mask)

    if use_non_sky:
        if sky_mask is None:
            raise ValueError("primary_mask requires non_sky but sky_mask is missing")
        valid = valid * (1.0 - (sky_mask > 0.5).float())
    if use_non_ego:
        if egocar_mask is not None:
            valid = valid * (1.0 - (egocar_mask > 0.5).float())

    valid_count = int((valid > 0.5).sum().item())
    if valid_count < int(min_valid_pixels):
        return {
            "psnr": None,
            "l1": None,
            "psnr_full": None,
            "l1_full": None,
            "valid_pixels": float(valid_count),
        }

    diff = (pred - gt).abs()
    mse_masked = ((pred - gt).pow(2) * valid.unsqueeze(-1)).sum() / (valid.sum() * 3.0 + 1e-8)
    l1_masked = (diff * valid.unsqueeze(-1)).sum() / (valid.sum() * 3.0 + 1e-8)
    mse_full = (pred - gt).pow(2).mean()
    l1_full = diff.mean()

    psnr = float(-10.0 * torch.log10(mse_masked + 1e-12).item())
    psnr_full = float(-10.0 * torch.log10(mse_full + 1e-12).item())
    return {
        "psnr": float(psnr),
        "l1": float(l1_masked.item()),
        "psnr_full": float(psnr_full),
        "l1_full": float(l1_full.item()),
        "valid_pixels": float(valid_count),
    }


class MetricAccumulator:
    def __init__(
        self,
        *,
        output_dir: Path,
        protocol: TestProtocolSpec,
        min_valid_pixels: int,
        compute_psnr: bool = True,
        compute_l1: bool = True,
        compute_ssim: bool,
        compute_lpips: bool,
    ):
        self.output_dir = Path(output_dir)
        self.protocol = protocol
        self.min_valid_pixels = int(min_valid_pixels)
        self.compute_psnr = bool(compute_psnr)
        self.compute_l1 = bool(compute_l1)
        self.compute_ssim = bool(compute_ssim)
        self.compute_lpips = bool(compute_lpips)
        if self.compute_ssim or self.compute_lpips:
            raise NotImplementedError(
                "compute_ssim/compute_lpips are configured but not implemented in MetricAccumulator."
            )
        self.iter_rows: List[Dict[str, Any]] = []
        self.episode_rows: Dict[str, List[Dict[str, Any]]] = {}
        self._warned_missing_egocar_for_cam: set[int] = set()

    def _build_row(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        view_row: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        pred = view_row.get("pred_rgb")
        gt = view_row.get("gt_image")
        if pred is None or gt is None:
            return None
        if not torch.is_tensor(pred) or not torch.is_tensor(gt):
            return None
        pred_t = _to_hwc_rgb(pred)
        gt_t = _to_hwc_rgb(gt)

        frame_idx = int(view_row["frame_idx"])
        cam_id = int(view_row["cam_idx"])
        frame_to_offset = {int(fid): int(off) for off, fid in enumerate(spec.frame_ids)}
        if frame_idx not in frame_to_offset:
            return None
        eval_offset = int(frame_to_offset[frame_idx])
        cam_name_map = {int(cid): str(name) for cid, name in zip(spec.camera_ids, spec.camera_names)}
        cam_name = cam_name_map.get(int(cam_id), f"cam_{cam_id}")

        sky_mask = view_row.get("sky_mask")
        ego_mask = view_row.get("egocar_mask")
        sky_t = _to_hw_mask(sky_mask)
        ego_t = _to_hw_mask(ego_mask)
        if "non_ego" in str(self.protocol.metric_primary_mask) and ego_t is None:
            if int(cam_id) not in self._warned_missing_egocar_for_cam:
                self._warned_missing_egocar_for_cam.add(int(cam_id))
                logger.warning(
                    "egocar_mask is missing for cam_id=%d (cam_name=%s); "
                    "non_ego suppression is disabled for this camera in metrics.",
                    int(cam_id),
                    str(cam_name),
                )
        vals = _masked_metrics(
            pred=pred_t,
            gt=gt_t,
            sky_mask=sky_t,
            egocar_mask=ego_t,
            primary_mask=str(self.protocol.metric_primary_mask),
            min_valid_pixels=int(self.min_valid_pixels),
        )
        row = {
            "exp_name": str(spec.exp_name),
            "episode_uid": str(spec.episode_uid),
            "scene_id": int(spec.scene_id),
            "segment_id": int(spec.segment_id),
            "global_iter": int(global_iter),
            "input_index": None if input_index is None else int(input_index),
            "input_frame_id": None if input_frame_id is None else int(input_frame_id),
            "local_step": int(local_step),
            "is_pre_update": bool(is_pre_update),
            "is_final": bool(global_iter == int(len(spec.input_frame_ids) * self.protocol.steps_per_input)),
            "eval_frame_id": int(frame_idx),
            "eval_offset": int(eval_offset),
            "cam_id": int(cam_id),
            "cam_name": str(cam_name),
            "frame_group": str(classify_eval_frame(int(eval_offset), [int(x) for x in spec.input_offsets])),
            "is_input_frame": bool(int(eval_offset) in set(int(x) for x in spec.input_offsets)),
            "input_count": int(len(spec.input_frame_ids)),
            "psnr": _safe_float(vals["psnr"]) if self.compute_psnr else float("nan"),
            "l1": _safe_float(vals["l1"]) if self.compute_l1 else float("nan"),
            "psnr_full": (
                _safe_float(vals["psnr_full"])
                if (self.compute_psnr and bool(self.protocol.report_full_image))
                else float("nan")
            ),
            "l1_full": (
                _safe_float(vals["l1_full"])
                if (self.compute_l1 and bool(self.protocol.report_full_image))
                else float("nan")
            ),
            "ssim": float("nan"),
            "lpips": float("nan"),
            "ssim_full": float("nan"),
            "lpips_full": float("nan"),
            "valid_pixels": int(vals["valid_pixels"]),
        }
        return row

    def add_iteration_rows(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        render_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for view_row in render_rows:
            out = self._build_row(
                spec=spec,
                global_iter=int(global_iter),
                is_pre_update=bool(is_pre_update),
                input_index=input_index,
                input_frame_id=input_frame_id,
                local_step=int(local_step),
                view_row=view_row,
            )
            if out is None:
                continue
            rows.append(out)
            self.iter_rows.append(out)
            self.episode_rows.setdefault(str(spec.episode_uid), []).append(out)
        return rows

    def finalize_episode(self, spec: TestEpisodeSpec) -> Dict[str, Any]:
        rows = self.episode_rows.get(str(spec.episode_uid), [])
        if len(rows) == 0:
            return {
                "episode_uid": str(spec.episode_uid),
                "num_rows": 0,
                "mean_psnr": float("nan"),
                "mean_l1": float("nan"),
            }
        final_iter = max(int(r["global_iter"]) for r in rows)
        final_rows = [r for r in rows if int(r["global_iter"]) == int(final_iter)]
        psnr_vals = [float(r["psnr"]) for r in final_rows if not math.isnan(float(r["psnr"]))]
        l1_vals = [float(r["l1"]) for r in final_rows if not math.isnan(float(r["l1"]))]
        return {
            "episode_uid": str(spec.episode_uid),
            "num_rows": int(len(rows)),
            "final_iter": int(final_iter),
            "mean_psnr": float(sum(psnr_vals) / len(psnr_vals)) if len(psnr_vals) > 0 else float("nan"),
            "mean_l1": float(sum(l1_vals) / len(l1_vals)) if len(l1_vals) > 0 else float("nan"),
        }

    def write_csvs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        iter_path = self.output_dir / "metrics_iter.csv"
        if len(self.iter_rows) == 0:
            with open(iter_path, "w", encoding="utf-8") as f:
                f.write("")
            with open(self.output_dir / "metrics_final.csv", "w", encoding="utf-8") as f:
                f.write("")
            return

        fields = list(self.iter_rows[0].keys())
        with open(iter_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.iter_rows:
                writer.writerow({k: row.get(k) for k in fields})

        final_by_episode: List[Dict[str, Any]] = []
        for uid, rows in self.episode_rows.items():
            _ = uid
            if len(rows) == 0:
                continue
            final_iter = max(int(r["global_iter"]) for r in rows)
            final_by_episode.extend([r for r in rows if int(r["global_iter"]) == int(final_iter)])
        final_path = self.output_dir / "metrics_final.csv"
        with open(final_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in final_by_episode:
                writer.writerow({k: row.get(k) for k in fields})
