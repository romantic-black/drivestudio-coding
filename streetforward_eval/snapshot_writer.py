from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .episode_builder import TestEpisodeSpec


@dataclass(frozen=True)
class RenderSaveConfig:
    save_png: bool = True
    save_numpy: bool = False
    save_video: bool = False
    save_depth_or_acc: bool = False


def _to_float_hwc(x: torch.Tensor) -> torch.Tensor:
    t = x.detach().float()
    if t.dim() == 4:
        if int(t.shape[0]) != 1:
            raise ValueError(f"expected batch size 1 image tensor, got shape={tuple(t.shape)}")
        t = t.squeeze(0)
    if t.dim() != 3:
        raise ValueError(f"expected 3D image tensor, got shape={tuple(t.shape)}")
    if int(t.shape[-1]) == 3:
        hwc = t
    elif int(t.shape[0]) == 3:
        hwc = t.permute(1, 2, 0)
    else:
        raise ValueError(f"expected HWC or CHW RGB tensor, got shape={tuple(t.shape)}")
    return torch.clamp(hwc, 0.0, 1.0)


def _hwc_to_uint8(hwc: torch.Tensor) -> np.ndarray:
    return (hwc.detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


class SnapshotWriter:
    def __init__(self, *, output_dir: Path, save_cfg: Optional[RenderSaveConfig] = None):
        self.output_dir = Path(output_dir)
        self.save_cfg = save_cfg or RenderSaveConfig()
        if bool(self.save_cfg.save_video):
            raise NotImplementedError("render.save_video=true is not implemented in SnapshotWriter yet.")
        if bool(self.save_cfg.save_depth_or_acc):
            raise NotImplementedError("render.save_depth_or_acc=true is not implemented in SnapshotWriter yet.")

    @staticmethod
    def _iter_name(
        *,
        global_iter: int,
        is_pre_update: bool,
        input_frame_id: Optional[int],
        local_step: int,
    ) -> str:
        if bool(is_pre_update):
            return "iter_000_pre"
        if input_frame_id is None:
            raise ValueError("input_frame_id is required for non-pre iteration naming")
        return f"iter_{int(global_iter):03d}_input{int(input_frame_id):06d}_step{int(local_step):02d}"

    def write_iteration(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        render_rows: List[Dict[str, Any]],
    ) -> str:
        _ = input_index
        iter_name = self._iter_name(
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_frame_id=input_frame_id,
            local_step=int(local_step),
        )
        episode_dir = (
            self.output_dir
            / f"scene_{int(spec.scene_id):03d}"
            / f"segment_{int(spec.segment_id):03d}"
            / str(spec.episode_uid)
            / iter_name
        )
        cam_name_map = {int(cid): str(name) for cid, name in zip(spec.camera_ids, spec.camera_names)}
        for row in render_rows:
            pred = row.get("pred_rgb")
            gt = row.get("gt_image")
            if pred is None or gt is None:
                continue
            if not torch.is_tensor(pred) or not torch.is_tensor(gt):
                continue
            frame_idx = int(row["frame_idx"])
            cam_id = int(row["cam_idx"])
            cam_name = cam_name_map.get(int(cam_id), f"cam_{cam_id}")
            frame_dir = episode_dir / f"frame_{int(frame_idx):06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            pred_hwc = _to_float_hwc(pred)
            gt_hwc = _to_float_hwc(gt)
            err_hwc = torch.clamp((pred_hwc - gt_hwc).abs(), 0.0, 1.0)
            if bool(self.save_cfg.save_png):
                Image.fromarray(_hwc_to_uint8(pred_hwc)).save(frame_dir / f"{cam_name}_pred.png")
                Image.fromarray(_hwc_to_uint8(gt_hwc)).save(frame_dir / f"{cam_name}_gt.png")
                Image.fromarray(_hwc_to_uint8(err_hwc)).save(frame_dir / f"{cam_name}_error.png")
            if bool(self.save_cfg.save_numpy):
                np.save(frame_dir / f"{cam_name}_pred.npy", pred_hwc.detach().cpu().numpy())
                np.save(frame_dir / f"{cam_name}_gt.npy", gt_hwc.detach().cpu().numpy())
                np.save(frame_dir / f"{cam_name}_error.npy", err_hwc.detach().cpu().numpy())
        return str(episode_dir)
