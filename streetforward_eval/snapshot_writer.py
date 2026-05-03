from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def _fs_token(s: str, *, max_len: int = 96) -> str:
    out: List[str] = []
    for ch in str(s):
        if ch.isalnum() or ch in "-._":
            out.append(ch)
        else:
            out.append("_")
    t = "".join(out)
    if len(t) > int(max_len):
        t = t[: int(max_len)]
    return t


def _format_psnr(psnr: float) -> str:
    if math.isnan(float(psnr)):
        return "NA"
    return f"{float(psnr):.4f}"


def _title_lines(
    *,
    spec: TestEpisodeSpec,
    global_iter: int,
    is_pre_update: bool,
    input_frame_id: Optional[int],
    local_step: int,
    frame_idx: int,
    cam_id: int,
    cam_name: str,
    psnr: float,
    kind: str,
) -> List[str]:
    pre_s = "1" if bool(is_pre_update) else "0"
    in_f = "NA" if input_frame_id is None else str(int(input_frame_id))
    psnr_s = _format_psnr(psnr)
    return [
        f"scene_id={int(spec.scene_id)} segment_id={int(spec.segment_id)} "
        f"eval_frame_id={int(frame_idx)} episode_uid={spec.episode_uid}",
        f"cam_id={int(cam_id)} cam_name={cam_name} PSNR_dB_non_sky={psnr_s} ({kind})",
        f"global_iter={int(global_iter)} is_pre_update={pre_s} "
        f"input_frame_id={in_f} local_step={int(local_step)}",
    ]


def _rgb_with_title_footer(rgb: Image.Image, lines: List[str]) -> Image.Image:
    font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    margins = 4
    line_spacing = 2
    heights: List[int] = []
    max_w = 0
    for line in lines:
        bbox = probe.textbbox((0, 0), line, font=font)
        lh = int(bbox[3] - bbox[1])
        lw = int(bbox[2] - bbox[0])
        heights.append(max(lh, 1))
        max_w = max(max_w, lw)
    footer_h = sum(heights) + line_spacing * max(0, len(lines) - 1) + 2 * margins
    w, h = rgb.size
    footer_w = max(w, max_w + 2 * margins)
    out = Image.new("RGB", (footer_w, h + footer_h), (24, 24, 24))
    out.paste(rgb, (0, 0))
    d = ImageDraw.Draw(out)
    y = h + margins
    for line, lh in zip(lines, heights):
        d.text((margins, y), line, fill=(255, 255, 255), font=font)
        y += lh + line_spacing
    return out


class SnapshotWriter:
    def __init__(self, *, output_dir: Path, save_cfg: Optional[RenderSaveConfig] = None):
        self.output_dir = Path(output_dir)
        self.save_cfg = save_cfg or RenderSaveConfig()
        if bool(self.save_cfg.save_video):
            raise NotImplementedError("render.save_video=true is not implemented in SnapshotWriter yet.")
        if bool(self.save_cfg.save_depth_or_acc):
            raise NotImplementedError("render.save_depth_or_acc=true is not implemented in SnapshotWriter yet.")

    @staticmethod
    def _iter_tag(
        *,
        global_iter: int,
        is_pre_update: bool,
        input_frame_id: Optional[int],
        local_step: int,
    ) -> str:
        if bool(is_pre_update):
            return f"g{int(global_iter):03d}_pre"
        if input_frame_id is None:
            raise ValueError("input_frame_id is required for non-pre iteration naming")
        return f"g{int(global_iter):03d}_in{int(input_frame_id):06d}_st{int(local_step):02d}"

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
        psnr_by_view: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> str:
        _ = input_index
        lookup = psnr_by_view or {}
        iter_tag = self._iter_tag(
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_frame_id=input_frame_id,
            local_step=int(local_step),
        )
        image_dir = self.output_dir / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
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
            psnr = float(lookup.get((int(frame_idx), int(cam_id)), float("nan")))
            psnr_token = _format_psnr(psnr)
            base = (
                f"{_fs_token(spec.episode_uid)}_sc{int(spec.scene_id):03d}_"
                f"sg{int(spec.segment_id):03d}_{iter_tag}_fr{int(frame_idx):06d}_"
                f"c{int(cam_id):02d}_{_fs_token(cam_name)}_psnr{psnr_token}"
            )
            pred_hwc = _to_float_hwc(pred)
            gt_hwc = _to_float_hwc(gt)
            err_hwc = torch.clamp((pred_hwc - gt_hwc).abs(), 0.0, 1.0)
            if bool(self.save_cfg.save_png):
                for kind, hwc in (("pred", pred_hwc), ("gt", gt_hwc), ("error", err_hwc)):
                    pil = Image.fromarray(_hwc_to_uint8(hwc)).convert("RGB")
                    lines = _title_lines(
                        spec=spec,
                        global_iter=int(global_iter),
                        is_pre_update=bool(is_pre_update),
                        input_frame_id=input_frame_id,
                        local_step=int(local_step),
                        frame_idx=int(frame_idx),
                        cam_id=int(cam_id),
                        cam_name=str(cam_name),
                        psnr=psnr,
                        kind=str(kind),
                    )
                    titled = _rgb_with_title_footer(pil, lines)
                    titled.save(image_dir / f"{base}_{kind}.png")
            if bool(self.save_cfg.save_numpy):
                for kind, arr in (
                    ("pred", pred_hwc.detach().cpu().numpy()),
                    ("gt", gt_hwc.detach().cpu().numpy()),
                    ("error", err_hwc.detach().cpu().numpy()),
                ):
                    np.save(image_dir / f"{base}_{kind}.npy", arr)
        return str(image_dir)
