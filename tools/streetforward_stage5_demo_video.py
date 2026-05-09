from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

ImageRef = Tuple[int, int]


@dataclass(frozen=True)
class RenderViewRecord:
    frame_idx: int
    cam_id: int
    c2w: torch.Tensor
    K: torch.Tensor
    height: int
    width: int


@dataclass(frozen=True)
class RenderSample:
    frame0: int
    frame1: int
    alpha: float
    rigid_frame_idx: int
    window_index: int
    sequence_start_pos: int


class _VideoWriterSet:
    def __init__(
        self,
        *,
        output_dir: Path,
        base_name: str,
        fps: int,
        camera_ids: Sequence[int],
        camera_names: Sequence[str],
        write_combined: bool,
        write_separate: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.paths: Dict[str, Path] = {}
        self._writers: Dict[str, Any] = {}
        if bool(write_combined):
            path = self.output_dir / f"{base_name}.mp4"
            self.paths["combined"] = path
            self._writers["combined"] = imageio.get_writer(
                str(path),
                format="FFMPEG",
                mode="I",
                fps=int(fps),
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
        if bool(write_separate):
            for cam_id, name in zip(camera_ids, camera_names):
                stem = _safe_stem(f"{base_name}_{name or ('cam' + str(cam_id))}")
                path = self.output_dir / f"{stem}.mp4"
                key = f"cam_{int(cam_id)}"
                self.paths[key] = path
                self._writers[key] = imageio.get_writer(
                    str(path),
                    format="FFMPEG",
                    mode="I",
                    fps=int(fps),
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
        if not self._writers:
            raise ValueError("video.output must enable write_combined or write_separate_per_camera")

    @staticmethod
    def _pad_even_hw(frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or int(frame.shape[-1]) != 3:
            raise ValueError(f"video frame must be HWC RGB, got shape={tuple(frame.shape)}")
        h, w = int(frame.shape[0]), int(frame.shape[1])
        pad_h = int(h % 2)
        pad_w = int(w % 2)
        if pad_h == 0 and pad_w == 0:
            return frame
        return np.pad(
            frame,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )

    def append(self, *, combined: Optional[np.ndarray], per_camera: Dict[int, np.ndarray]) -> None:
        if "combined" in self._writers and combined is not None:
            self._writers["combined"].append_data(self._pad_even_hw(combined))
        for key, writer in self._writers.items():
            if not key.startswith("cam_"):
                continue
            cam_id = int(key.split("_", 1)[1])
            frame = per_camera.get(cam_id)
            if frame is not None:
                writer.append_data(self._pad_even_hw(frame))

    def close(self) -> None:
        first_exc: Optional[BaseException] = None
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as exc:  # pragma: no cover - close best effort
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc


def _safe_stem(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "video"


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        val = node.get(key, default)
        return default if val is None else val
    return getattr(node, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        from omegaconf import ListConfig

        if isinstance(value, ListConfig):
            return list(value)
    except Exception:
        pass
    return [value]


def derive_input_offsets(*, window_size: int, input_gap_frames: int, explicit: Optional[Any] = None) -> List[int]:
    explicit_list = _as_list(explicit)
    if explicit is not None and len(explicit_list) > 0:
        out = [int(x) for x in explicit_list]
    else:
        step = int(input_gap_frames) + 1
        if step < 1:
            raise ValueError("video.reconstruction.input_gap_frames must be >= 0")
        out = list(range(0, int(window_size), int(step)))
    if len(out) == 0:
        raise ValueError("derived video input offsets are empty")
    bad = [x for x in out if int(x) < 0 or int(x) >= int(window_size)]
    if bad:
        raise ValueError(f"video input offsets out of range for window_size={window_size}: {bad}")
    return [int(x) for x in out]


def _frames_for_segment(dataset: Any, scene_id: int, segment_id: int) -> List[int]:
    sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
    frames = [int(x) for x in sorted(sidx.frame_indices)]
    train_frame_set = getattr(sidx, "train_frame_set", None)
    if train_frame_set is not None:
        train_set = set(int(x) for x in train_frame_set)
        frames = [int(f) for f in frames if int(f) in train_set]
    return frames


def _select_sequence_starts(
    *,
    controller: Any,
    explicit_starts: Sequence[int],
    max_windows: Optional[int],
) -> List[int]:
    if explicit_starts:
        starts = [int(x) for x in explicit_starts]
    else:
        starts = [int(x) for x in controller.list_sequence_start_positions()]
    if len(starts) == 0:
        raise ValueError("no valid sequence starts for video export")
    if max_windows is not None and int(max_windows) > 0:
        starts = starts[: int(max_windows)]
    return [int(x) for x in starts]


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(R)))
        if axis == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1.0e-12)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q.tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1.0e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1.0e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    a = float(alpha)
    if dot > 0.9995:
        out = q0 + a * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1.0e-12)
    theta_0 = math.acos(dot)
    theta = theta_0 * a
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_c2w(c2w0: torch.Tensor, c2w1: torch.Tensor, alpha: float) -> torch.Tensor:
    dev = c2w0.device
    dtype = c2w0.dtype
    a = float(alpha)
    c0 = c2w0.detach().float().cpu().numpy()
    c1 = c2w1.detach().float().cpu().numpy()
    q0 = _rotmat_to_quat_wxyz(c0[:3, :3])
    q1 = _rotmat_to_quat_wxyz(c1[:3, :3])
    q = _slerp_quat_wxyz(q0, q1, a)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = _quat_wxyz_to_rotmat(q).astype(np.float32)
    out[:3, 3] = ((1.0 - a) * c0[:3, 3] + a * c1[:3, 3]).astype(np.float32)
    return torch.as_tensor(out, dtype=dtype, device=dev)


def _scale_intrinsics(K: torch.Tensor, *, src_h: int, src_w: int, dst_h: int, dst_w: int) -> torch.Tensor:
    out = K.clone()
    sx = float(dst_w) / float(max(int(src_w), 1))
    sy = float(dst_h) / float(max(int(src_h), 1))
    out[0, :] *= sx
    out[1, :] *= sy
    return out


def _to_uint8(rgb: torch.Tensor) -> np.ndarray:
    arr = torch.clamp(rgb.detach().float().cpu(), 0.0, 1.0).numpy()
    return (arr * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    if not label:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = str(label)
    try:
        bbox = draw.textbbox((0, 0), text)
        tw, th = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        tw, th = max(8 * len(text), 1), 12
    pad = 8
    draw.rectangle((0, 0, tw + pad * 2, th + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return np.asarray(img)


def _tile_frames(frames: Sequence[np.ndarray], *, layout: str, labels: Sequence[str], show_labels: bool) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError("cannot tile empty frame list")
    proc = []
    for i, frame in enumerate(frames):
        label = labels[i] if bool(show_labels) and i < len(labels) else ""
        proc.append(_draw_label(frame, label))
    mode = str(layout).strip().lower()
    if mode == "auto":
        mode = "single" if len(proc) == 1 else "horizontal"
    if mode == "single":
        return proc[0]
    if mode == "horizontal":
        return np.concatenate(proc, axis=1)
    if mode == "vertical":
        return np.concatenate(proc, axis=0)
    if mode == "grid":
        if len(proc) == 1:
            return proc[0]
        if len(proc) <= 3:
            return np.concatenate(proc, axis=1)
        cols = int(math.ceil(math.sqrt(len(proc))))
        rows = int(math.ceil(len(proc) / cols))
        h, w = int(proc[0].shape[0]), int(proc[0].shape[1])
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        out_rows = []
        for r in range(rows):
            row = []
            for c in range(cols):
                idx = r * cols + c
                row.append(proc[idx] if idx < len(proc) else blank)
            out_rows.append(np.concatenate(row, axis=1))
        return np.concatenate(out_rows, axis=0)
    raise ValueError("video.output.layout must be one of: auto, single, horizontal, vertical, grid")


class Stage5DemoVideoExporter:
    def __init__(
        self,
        *,
        cfg: Any,
        dataset: Any,
        controller: Any,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.controller = controller
        self.device = device
        self.output_dir = Path(output_dir)
        video_cfg = _cfg_get(cfg, "video", {}) or {}
        recon_cfg = _cfg_get(video_cfg, "reconstruction", {}) or {}
        interp_cfg = _cfg_get(video_cfg, "interpolation", {}) or {}
        output_cfg = _cfg_get(video_cfg, "output", {}) or {}
        render_cfg = _cfg_get(video_cfg, "render", {}) or {}
        cameras_cfg = _cfg_get(video_cfg, "cameras", {}) or {}

        self.window_size = int(_cfg_get(recon_cfg, "window_size", _cfg_get(cfg.demo.scheduler, "sequence_length", 8)))
        self.window_stride = int(_cfg_get(recon_cfg, "window_stride", self.window_size))
        self.input_offsets = derive_input_offsets(
            window_size=int(self.window_size),
            input_gap_frames=int(_cfg_get(recon_cfg, "input_gap_frames", 1)),
            explicit=_cfg_get(recon_cfg, "input_offsets", None),
        )
        self.max_windows = _cfg_get(recon_cfg, "max_windows", None)
        self.max_windows = None if self.max_windows is None else int(self.max_windows)
        self.explicit_sequence_starts = [int(x) for x in _as_list(_cfg_get(recon_cfg, "sequence_start_positions", []))]

        self.source_fps = float(_cfg_get(interp_cfg, "source_fps", 10.0))
        self.subframes_per_interval = int(_cfg_get(interp_cfg, "subframes_per_source_interval", 2))
        self.include_tail_interval = bool(_cfg_get(interp_cfg, "include_window_tail_interval", True))
        self.rigid_frame_policy = str(_cfg_get(interp_cfg, "rigid_frame_policy", "nearest")).strip().lower()
        if self.rigid_frame_policy not in {"floor", "nearest", "ceil"}:
            raise ValueError("video.interpolation.rigid_frame_policy must be one of: floor, nearest, ceil")
        if int(self.subframes_per_interval) < 1:
            raise ValueError("video.interpolation.subframes_per_source_interval must be >= 1")

        fps_default = int(round(float(self.source_fps) * int(self.subframes_per_interval)))
        self.fps = int(_cfg_get(output_cfg, "fps", fps_default))
        if int(self.fps) < 20:
            raise ValueError("video.output.fps must be >= 20 for the requested demo output")
        self.layout = str(_cfg_get(output_cfg, "layout", "auto")).strip().lower()
        self.write_combined = bool(_cfg_get(output_cfg, "write_combined", True))
        self.write_separate = bool(_cfg_get(output_cfg, "write_separate_per_camera", False))
        self.save_png_frames = bool(_cfg_get(output_cfg, "save_png_frames", False))
        self.show_labels = bool(_cfg_get(output_cfg, "show_camera_labels", False))
        self.base_name = _safe_stem(str(_cfg_get(output_cfg, "name", "stage5_6_demo_video")))

        self.camera_ids = [int(x) for x in _as_list(_cfg_get(cameras_cfg, "ids", [0]))]
        camera_names = [str(x) for x in _as_list(_cfg_get(cameras_cfg, "names", []))]
        if not camera_names:
            camera_names = [f"cam{int(x)}" for x in self.camera_ids]
        if len(camera_names) != len(self.camera_ids):
            raise ValueError("video.cameras.ids and video.cameras.names must have the same length")
        self.camera_names = camera_names

        self.render_height = _cfg_get(render_cfg, "height", None)
        self.render_width = _cfg_get(render_cfg, "width", None)
        self.render_height = None if self.render_height is None else int(self.render_height)
        self.render_width = None if self.render_width is None else int(self.render_width)
        if (self.render_height is None) != (self.render_width is None):
            raise ValueError("video.render.height and video.render.width must be set together")

        trainer = getattr(self.controller, "trainer", None)
        if trainer is None or not hasattr(trainer, "_render_scene_views_from_current_state"):
            raise ValueError("Stage5 demo video export requires trainer._render_scene_views_from_current_state")

    def _load_render_views(
        self,
        *,
        scene_id: int,
        segment_id: int,
        frame_ids: Sequence[int],
    ) -> Dict[ImageRef, RenderViewRecord]:
        refs = [(int(f), int(c)) for f in frame_ids for c in self.camera_ids]
        if len(refs) == 0:
            raise ValueError("cannot load render views for empty frame/camera refs")
        if not hasattr(self.dataset, "_load_image_meta") or not hasattr(self.dataset, "_resolve_segment_bundle"):
            raise ValueError("Stage5 demo video requires MultiSceneDatasetV4 metadata loaders")
        bundle = self.dataset._resolve_segment_bundle(int(scene_id), int(segment_id))
        world_to_seg0 = bundle.segment_pose["world_to_seg0"].to(device=self.device, dtype=torch.float32)
        out: Dict[ImageRef, RenderViewRecord] = {}
        for ref in refs:
            if hasattr(self.dataset, "validate_image_ref"):
                self.dataset.validate_image_ref(int(scene_id), int(segment_id), tuple(ref), purpose="train")
            meta = self.dataset._load_image_meta(int(scene_id), int(segment_id), tuple(ref))
            h, w = int(meta["height"]), int(meta["width"])
            K = torch.as_tensor(meta["intrinsic_4x4"], dtype=torch.float32, device=self.device)[:3, :3]
            c2w = world_to_seg0 @ torch.as_tensor(
                meta["camera_to_world"],
                dtype=torch.float32,
                device=self.device,
            )
            if self.render_height is not None and self.render_width is not None:
                K = _scale_intrinsics(
                    K,
                    src_h=h,
                    src_w=w,
                    dst_h=int(self.render_height),
                    dst_w=int(self.render_width),
                )
                h = int(self.render_height)
                w = int(self.render_width)
            out[(int(ref[0]), int(ref[1]))] = RenderViewRecord(
                frame_idx=int(ref[0]),
                cam_id=int(ref[1]),
                c2w=c2w,
                K=K,
                height=h,
                width=w,
            )
        return out

    def _rigid_frame_for_sample(self, *, f0: int, f1: int, alpha: float, valid_frames: Iterable[int]) -> int:
        valid = set(int(x) for x in valid_frames)
        if self.rigid_frame_policy == "floor":
            choice = int(f0)
        elif self.rigid_frame_policy == "ceil":
            choice = int(f1)
        else:
            choice = int(f1) if float(alpha) >= 0.5 else int(f0)
        if int(choice) not in valid:
            choice = int(f0)
        if int(choice) not in valid:
            choice = sorted(valid)[0]
        return int(choice)

    def _build_render_samples(
        self,
        *,
        window_frame_ids: Sequence[int],
        next_frame_id: Optional[int],
        window_index: int,
        sequence_start_pos: int,
    ) -> List[RenderSample]:
        frames = [int(x) for x in window_frame_ids]
        if len(frames) < 1:
            return []
        samples: List[RenderSample] = []
        valid_rigid_frames = set(frames)
        interval_pairs: List[Tuple[int, int]] = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        if bool(self.include_tail_interval) and next_frame_id is not None:
            interval_pairs.append((int(frames[-1]), int(next_frame_id)))
        for f0, f1 in interval_pairs:
            for sub in range(int(self.subframes_per_interval)):
                alpha = float(sub) / float(self.subframes_per_interval)
                samples.append(
                    RenderSample(
                        frame0=int(f0),
                        frame1=int(f1),
                        alpha=float(alpha),
                        rigid_frame_idx=self._rigid_frame_for_sample(
                            f0=int(f0),
                            f1=int(f1),
                            alpha=float(alpha),
                            valid_frames=valid_rigid_frames,
                        ),
                        window_index=int(window_index),
                        sequence_start_pos=int(sequence_start_pos),
                    )
                )
        if not bool(self.include_tail_interval) or next_frame_id is None:
            samples.append(
                RenderSample(
                    frame0=int(frames[-1]),
                    frame1=int(frames[-1]),
                    alpha=0.0,
                    rigid_frame_idx=int(frames[-1]),
                    window_index=int(window_index),
                    sequence_start_pos=int(sequence_start_pos),
                )
            )
        return samples

    def _make_render_items(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for cam_id in self.camera_ids:
            rec0 = records[(int(sample.frame0), int(cam_id))]
            rec1 = records[(int(sample.frame1), int(cam_id))]
            c2w = interpolate_c2w(rec0.c2w, rec1.c2w, float(sample.alpha))
            K = (1.0 - float(sample.alpha)) * rec0.K + float(sample.alpha) * rec1.K
            view = type(
                "View",
                (),
                {
                    "camtoworlds": c2w,
                    "Ks": K.unsqueeze(0),
                },
            )()
            items.append(
                {
                    "view": view,
                    "height": int(rec0.height),
                    "width": int(rec0.width),
                    "frame_idx": int(sample.rigid_frame_idx),
                }
            )
        return items

    @torch.no_grad()
    def _render_sample(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
    ) -> Dict[int, np.ndarray]:
        minimal = self.controller.display.last_minimal_batch
        if not isinstance(minimal, dict):
            raise ValueError("controller has no current minimal batch for rendering")
        items = self._make_render_items(records=records, sample=sample)
        rgb, _alpha = self.controller.trainer._render_scene_views_from_current_state(minimal, items)
        out: Dict[int, np.ndarray] = {}
        for idx, cam_id in enumerate(self.camera_ids):
            out[int(cam_id)] = _to_uint8(rgb[idx])
        return out

    def export(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        starts = _select_sequence_starts(
            controller=self.controller,
            explicit_starts=self.explicit_sequence_starts,
            max_windows=self.max_windows,
        )
        writers = _VideoWriterSet(
            output_dir=self.output_dir,
            base_name=self.base_name,
            fps=int(self.fps),
            camera_ids=self.camera_ids,
            camera_names=self.camera_names,
            write_combined=bool(self.write_combined),
            write_separate=bool(self.write_separate),
        )
        png_dir = self.output_dir / "frames"
        if self.save_png_frames:
            png_dir.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {
            "fps": int(self.fps),
            "source_fps": float(self.source_fps),
            "subframes_per_source_interval": int(self.subframes_per_interval),
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "input_offsets": [int(x) for x in self.input_offsets],
            "camera_ids": [int(x) for x in self.camera_ids],
            "camera_names": [str(x) for x in self.camera_names],
            "sequence_starts": [int(x) for x in starts],
            "videos": {k: str(v) for k, v in writers.paths.items()},
            "samples": [],
        }
        total_samples = 0
        try:
            for window_index, start in enumerate(starts):
                if window_index == 0:
                    cur_info = self.controller.scheduler.get_current_info()
                    if int(cur_info.get("sequence_start_pos", start)) != int(start):
                        self.controller.set_sequence_start_pos(int(start))
                else:
                    self.controller.set_sequence_start_pos(int(start))
                stats = self.controller.run_episode()
                scene_id = int(stats.get("scene_id", self.controller.scheduler.get_current_info().get("scene_id", -1)))
                segment_id = int(
                    stats.get("segment_id", self.controller.scheduler.get_current_info().get("segment_id", -1))
                )
                segment_frames = _frames_for_segment(self.dataset, scene_id, segment_id)
                window_frames = segment_frames[int(start) : int(start) + int(self.window_size)]
                if len(window_frames) < int(self.window_size):
                    raise ValueError(
                        f"sequence_start_pos={start} only has {len(window_frames)} frames, "
                        f"expected window_size={self.window_size}"
                    )
                next_frame_id = None
                tail_pos = int(start) + int(self.window_size)
                if bool(self.include_tail_interval) and tail_pos < len(segment_frames):
                    next_frame_id = int(segment_frames[tail_pos])
                pose_frames = list(window_frames)
                if next_frame_id is not None:
                    pose_frames.append(int(next_frame_id))
                records = self._load_render_views(
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    frame_ids=pose_frames,
                )
                samples = self._build_render_samples(
                    window_frame_ids=window_frames,
                    next_frame_id=next_frame_id,
                    window_index=int(window_index),
                    sequence_start_pos=int(start),
                )
                logger.info(
                    "video window %d/%d scene=%d segment=%d start=%d frames=%s samples=%d",
                    int(window_index) + 1,
                    len(starts),
                    scene_id,
                    segment_id,
                    int(start),
                    [int(x) for x in window_frames],
                    len(samples),
                )
                for sample in samples:
                    per_camera = self._render_sample(records=records, sample=sample)
                    frames = [per_camera[int(c)] for c in self.camera_ids]
                    combined = (
                        _tile_frames(
                            frames,
                            layout=str(self.layout),
                            labels=self.camera_names,
                            show_labels=bool(self.show_labels),
                        )
                        if self.write_combined
                        else None
                    )
                    writers.append(combined=combined, per_camera=per_camera)
                    if self.save_png_frames:
                        if combined is not None:
                            Image.fromarray(combined).save(png_dir / f"frame_{total_samples:06d}.png")
                        for cam_id, frame in per_camera.items():
                            Image.fromarray(frame).save(png_dir / f"frame_{total_samples:06d}_cam{int(cam_id)}.png")
                    metadata["samples"].append(
                        {
                            "index": int(total_samples),
                            "window_index": int(sample.window_index),
                            "sequence_start_pos": int(sample.sequence_start_pos),
                            "frame0": int(sample.frame0),
                            "frame1": int(sample.frame1),
                            "alpha": float(sample.alpha),
                            "rigid_frame_idx": int(sample.rigid_frame_idx),
                        }
                    )
                    total_samples += 1
        finally:
            writers.close()

        metadata["num_video_frames"] = int(total_samples)
        metadata_path = self.output_dir / f"{self.base_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        metadata["metadata_path"] = str(metadata_path)
        logger.info("video export wrote %d frames to %s", int(total_samples), self.output_dir)
        return metadata
