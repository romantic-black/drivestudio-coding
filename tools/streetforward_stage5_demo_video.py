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
    local_source_pos: float
    global_source_pos: float
    global_time_seconds: float
    global_output_time_index: int
    anchor_input_offset: int
    is_transition: bool


@dataclass(frozen=True)
class WindowPlan:
    scene_id: int
    segment_id: int
    sequence_start_pos: int
    timeline_start_pos: float


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
        val = node.get(key, default)
        return default if val is None else val
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


def _is_auto(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}:
        return True
    return False


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


def _window_starts_for_frames(
    *,
    num_frames: int,
    window_size: int,
    window_stride: int,
    require_full_window: bool,
    window_policy: str,
) -> List[int]:
    if int(num_frames) <= 0:
        return []
    if int(window_size) < 1:
        raise ValueError("video.reconstruction.window_size must be >= 1")
    if int(window_stride) < 1:
        raise ValueError("video.reconstruction.window_stride must be >= 1")
    policy = str(window_policy).strip().lower()
    if policy == "middle":
        if bool(require_full_window) and int(num_frames) < int(window_size):
            return []
        return [max(0, (int(num_frames) - int(window_size)) // 2)]
    if policy == "sliding":
        if bool(require_full_window):
            if int(num_frames) < int(window_size):
                return []
            return list(range(0, int(num_frames) - int(window_size) + 1, int(window_stride)))
        return list(range(0, int(num_frames), int(window_stride)))
    raise ValueError("video.reconstruction.window_policy must be one of: sliding, middle")


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
        sky_branch: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.controller = controller
        self.device = device
        self.output_dir = Path(output_dir)
        self.sky_branch = sky_branch
        self._single_sky_state: Optional[Any] = None
        self._sky_pre_render_done = False
        video_cfg = _cfg_get(cfg, "video", {}) or {}
        recon_cfg = _cfg_get(video_cfg, "reconstruction", {}) or {}
        interp_cfg = _cfg_get(video_cfg, "interpolation", {}) or {}
        output_cfg = _cfg_get(video_cfg, "output", {}) or {}
        render_cfg = _cfg_get(video_cfg, "render", {}) or {}
        cameras_cfg = _cfg_get(video_cfg, "cameras", {}) or {}
        sky_cfg = _cfg_get(video_cfg, "sky", {}) or {}
        camera_path_cfg = _cfg_get(video_cfg, "camera_path", None)
        if camera_path_cfg is None:
            camera_path_cfg = _cfg_get(video_cfg, "route", {}) or {}

        self.window_size = int(_cfg_get(recon_cfg, "window_size", _cfg_get(cfg.demo.scheduler, "sequence_length", 8)))
        self.window_stride = int(_cfg_get(recon_cfg, "window_stride", self.window_size))
        self.require_full_window = bool(_cfg_get(recon_cfg, "require_full_window", False))
        self.window_policy = str(_cfg_get(recon_cfg, "window_policy", "sliding")).strip().lower()
        self.multi_segment = bool(_cfg_get(recon_cfg, "multi_segment", False))
        self.segment_policy = str(
            _cfg_get(recon_cfg, "segment_policy", "from_initial" if self.multi_segment else "current")
        ).strip().lower()
        if self.segment_policy not in {"current", "all", "from_initial"}:
            raise ValueError("video.reconstruction.segment_policy must be one of: current, all, from_initial")
        self.input_offsets = derive_input_offsets(
            window_size=int(self.window_size),
            input_gap_frames=int(_cfg_get(recon_cfg, "input_gap_frames", 1)),
            explicit=_cfg_get(recon_cfg, "input_offsets", None),
        )
        self.max_windows = _cfg_get(recon_cfg, "max_windows", None)
        self.max_windows = None if self.max_windows is None else int(self.max_windows)
        self.explicit_sequence_starts = [int(x) for x in _as_list(_cfg_get(recon_cfg, "sequence_start_positions", []))]
        transition_frames = int(_cfg_get(recon_cfg, "transition_frames", 0))
        self.transition_frames_before = int(_cfg_get(recon_cfg, "transition_frames_before", transition_frames))
        self.transition_frames_after = int(_cfg_get(recon_cfg, "transition_frames_after", transition_frames))
        if self.transition_frames_before < 0 or self.transition_frames_after < 0:
            raise ValueError("video.reconstruction transition frame counts must be >= 0")
        if self.transition_frames_before + self.transition_frames_after >= int(self.window_size):
            raise ValueError(
                "video.reconstruction transition frames leave no stitchable core: "
                f"before={self.transition_frames_before} after={self.transition_frames_after} "
                f"window_size={self.window_size}"
            )
        self.state_carryover = str(_cfg_get(recon_cfg, "state_carryover", "reset")).strip().lower()
        if self.state_carryover not in {"reset", "node_state"}:
            raise ValueError("video.reconstruction.state_carryover must be one of: reset, node_state")
        self.discard_history_between_windows = bool(_cfg_get(recon_cfg, "discard_history_between_windows", True))

        self.source_fps = float(_cfg_get(interp_cfg, "source_fps", 10.0))
        if float(self.source_fps) <= 0.0:
            raise ValueError("video.interpolation.source_fps must be > 0")
        target_fps_raw = _cfg_get(interp_cfg, "target_fps", None)
        subframes_raw = _cfg_get(interp_cfg, "subframes_per_source_interval", None)
        output_fps_raw = _cfg_get(output_cfg, "fps", None)
        if _is_auto(target_fps_raw) and (not _is_auto(output_fps_raw)) and _is_auto(subframes_raw):
            target_fps_raw = output_fps_raw
        if not _is_auto(target_fps_raw):
            target_fps = float(target_fps_raw)
            if target_fps < 20.0:
                raise ValueError("video.interpolation.target_fps must be >= 20")
            ratio = target_fps / float(self.source_fps)
            subframes = int(round(ratio))
            if subframes < 1 or abs(float(subframes) - float(ratio)) > 1.0e-6:
                raise ValueError(
                    "video.interpolation.target_fps must be an integer multiple of source_fps "
                    f"for deterministic interpolation; got target_fps={target_fps} source_fps={self.source_fps}."
                )
            if not _is_auto(subframes_raw) and int(subframes_raw) != int(subframes):
                raise ValueError(
                    "video.interpolation.subframes_per_source_interval conflicts with target_fps: "
                    f"subframes={subframes_raw} but target/source requires {subframes}."
                )
            self.subframes_per_interval = int(subframes)
            fps_default = int(round(target_fps))
        else:
            self.subframes_per_interval = int(2 if _is_auto(subframes_raw) else subframes_raw)
            fps_default = int(round(float(self.source_fps) * int(self.subframes_per_interval)))
        self.include_tail_interval = bool(_cfg_get(interp_cfg, "include_window_tail_interval", True))
        self.rigid_frame_policy = str(_cfg_get(interp_cfg, "rigid_frame_policy", "nearest")).strip().lower()
        if self.rigid_frame_policy not in {"floor", "nearest", "ceil"}:
            raise ValueError("video.interpolation.rigid_frame_policy must be one of: floor, nearest, ceil")
        if int(self.subframes_per_interval) < 1:
            raise ValueError("video.interpolation.subframes_per_source_interval must be >= 1")

        self.fps = int(_cfg_get(output_cfg, "fps", fps_default))
        if int(self.fps) < 20:
            raise ValueError("video.output.fps must be >= 20 for the requested demo output")
        if int(self.fps) != int(fps_default):
            raise ValueError(
                "video.output.fps must match the interpolation sample rate so all videos share one time base: "
                f"output.fps={self.fps}, derived={fps_default}. Set video.output.fps=null or make it match."
            )
        self.target_fps = int(fps_default)
        self.layout = str(_cfg_get(output_cfg, "layout", "auto")).strip().lower()
        self.write_combined = bool(_cfg_get(output_cfg, "write_combined", True))
        self.write_separate = bool(_cfg_get(output_cfg, "write_separate_per_camera", False))
        self.save_png_frames = bool(_cfg_get(output_cfg, "save_png_frames", False))
        self.save_all_images = bool(_cfg_get(output_cfg, "save_all_images", self.save_png_frames))
        self.show_labels = bool(_cfg_get(output_cfg, "show_camera_labels", False))
        self.base_name = _safe_stem(str(_cfg_get(output_cfg, "name", "stage5_6_demo_video")))

        self.sky_enabled = bool(self.sky_branch is not None and _cfg_get(sky_cfg, "enable", True))
        self.sky_reuse_single_state = bool(_cfg_get(sky_cfg, "reuse_single_state", True))
        self.sky_update_during_video = bool(_cfg_get(sky_cfg, "update_during_video", False))
        self.sky_compose_mode = str(_cfg_get(sky_cfg, "compose_mode", "alpha_gap")).strip().lower()
        self.sky_alpha_scale = float(_cfg_get(sky_cfg, "alpha_scale", 1.0))
        self.sky_pre_render_update_steps = int(
            _cfg_get(sky_cfg, "pre_render_update_steps", _cfg_get(sky_cfg, "warmup_steps", 0))
        )
        self.sky_pre_render_update_each_window = bool(_cfg_get(sky_cfg, "pre_render_update_each_window", True))
        self.sky_reset_runtime_before_export = bool(_cfg_get(sky_cfg, "reset_runtime_before_export", False))
        self.sky_reset_runtime_per_window = bool(_cfg_get(sky_cfg, "reset_runtime_per_window", False))
        self.sky_pre_render_fail_on_error = bool(_cfg_get(sky_cfg, "pre_render_fail_on_error", False))
        if self.sky_compose_mode not in {"alpha_gap", "replace"}:
            raise ValueError("video.sky.compose_mode must be one of: alpha_gap, replace")
        if self.sky_update_during_video:
            raise ValueError("video.sky.update_during_video=true is not supported; demo video sky is render-only.")
        if self.sky_pre_render_update_steps < 0:
            raise ValueError("video.sky.pre_render_update_steps must be >= 0")
        if self.sky_enabled:
            if bool(self.sky_reset_runtime_before_export) and hasattr(self.sky_branch, "reset_runtime_state"):
                self.sky_branch.reset_runtime_state()
            self._single_sky_state = self._select_single_sky_state()

        self.camera_ids = [int(x) for x in _as_list(_cfg_get(cameras_cfg, "ids", [0]))]
        camera_names = [str(x) for x in _as_list(_cfg_get(cameras_cfg, "names", []))]
        if not camera_names:
            camera_names = [f"cam{int(x)}" for x in self.camera_ids]
        if len(camera_names) != len(self.camera_ids):
            raise ValueError("video.cameras.ids and video.cameras.names must have the same length")
        self.camera_names = camera_names
        if not self.camera_ids:
            raise ValueError("video.cameras.ids must not be empty")

        self.camera_path_mode = str(_cfg_get(camera_path_cfg, "mode", "original")).strip().lower()
        if self.camera_path_mode in {"", "none", "off", "dataset"}:
            self.camera_path_mode = "original"
        if self.camera_path_mode in {"yaw_sine", "sine_rotate", "sine_rotation"}:
            self.camera_path_mode = "sine_yaw"
        if self.camera_path_mode in {"lateral_sine", "sine_translate", "sine_translation"}:
            self.camera_path_mode = "sine_lateral"
        if self.camera_path_mode in {"yaw_lateral_sine", "sine_yaw_lateral"}:
            self.camera_path_mode = "sine_yaw_lateral"
        if self.camera_path_mode not in {"original", "sine_yaw", "sine_lateral", "sine_yaw_lateral"}:
            raise ValueError("video.camera_path.mode must be one of: original, sine_yaw, sine_lateral, sine_yaw_lateral")
        self.camera_path_source_camera_id = int(_cfg_get(camera_path_cfg, "source_camera_id", self.camera_ids[0]))
        self.camera_path_left_camera_id = int(_cfg_get(camera_path_cfg, "left_camera_id", 1))
        self.camera_path_right_camera_id = int(_cfg_get(camera_path_cfg, "right_camera_id", 2))
        self.camera_path_amplitude = float(_cfg_get(camera_path_cfg, "amplitude", 1.0))
        self.camera_path_cycles = float(_cfg_get(camera_path_cfg, "cycles", 1.0))
        self.camera_path_phase = float(_cfg_get(camera_path_cfg, "phase", 0.0))
        self.camera_path_period_frames = _cfg_get(camera_path_cfg, "period_frames", None)
        self.camera_path_period_frames = (
            None if self.camera_path_period_frames is None else float(self.camera_path_period_frames)
        )
        self.camera_path_clamp_to_side_cameras = bool(_cfg_get(camera_path_cfg, "clamp_to_side_cameras", True))
        self.camera_path_max_yaw_degrees = _cfg_get(camera_path_cfg, "max_yaw_degrees", None)
        self.camera_path_max_yaw_degrees = (
            None if self.camera_path_max_yaw_degrees is None else float(self.camera_path_max_yaw_degrees)
        )
        self.camera_path_fallback_yaw_degrees = float(_cfg_get(camera_path_cfg, "fallback_yaw_degrees", 20.0))
        self.camera_path_lateral_meters = _cfg_get(camera_path_cfg, "lateral_meters", None)
        self.camera_path_lateral_meters = (
            None if self.camera_path_lateral_meters is None else float(self.camera_path_lateral_meters)
        )
        self.camera_path_fallback_lateral_meters = float(_cfg_get(camera_path_cfg, "fallback_lateral_meters", 0.35))
        route_ref_ids = [int(x) for x in _as_list(_cfg_get(camera_path_cfg, "reference_camera_ids", []))]
        if self.camera_path_mode != "original" and self.camera_path_clamp_to_side_cameras:
            route_ref_ids.extend([self.camera_path_source_camera_id, self.camera_path_left_camera_id, self.camera_path_right_camera_id])
        self.render_camera_ids = list(dict.fromkeys([int(x) for x in self.camera_ids] + [int(x) for x in route_ref_ids]))

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
        refs = [(int(f), int(c)) for f in frame_ids for c in self.render_camera_ids]
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

    def _interpolated_camera_record(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
        cam_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        rec0 = records[(int(sample.frame0), int(cam_id))]
        rec1 = records[(int(sample.frame1), int(cam_id))]
        c2w = interpolate_c2w(rec0.c2w, rec1.c2w, float(sample.alpha))
        K = (1.0 - float(sample.alpha)) * rec0.K + float(sample.alpha) * rec1.K
        return c2w, K, int(rec0.height), int(rec0.width)

    def _camera_path_signal(self, sample: RenderSample) -> float:
        period = self.camera_path_period_frames
        if period is None or float(period) <= 0.0:
            period = float(max(int(self.window_size), 1))
        phase = 2.0 * math.pi * float(self.camera_path_cycles) * float(sample.global_source_pos) / float(period)
        phase += float(self.camera_path_phase)
        return float(self.camera_path_amplitude) * math.sin(float(phase))

    @staticmethod
    def _local_yaw_matrix(angle_rad: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        return torch.tensor(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _relative_local_yaw(base_c2w: torch.Tensor, side_c2w: torch.Tensor) -> float:
        rel = base_c2w[:3, :3].transpose(0, 1) @ side_c2w[:3, :3]
        return float(torch.atan2(rel[0, 2], rel[2, 2]).detach().cpu().item())

    def _side_camera_yaw_bound(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
        base_c2w: torch.Tensor,
    ) -> float:
        bounds: List[float] = []
        for side_id in (int(self.camera_path_left_camera_id), int(self.camera_path_right_camera_id)):
            try:
                side_c2w, _side_K, _side_h, _side_w = self._interpolated_camera_record(
                    records=records,
                    sample=sample,
                    cam_id=int(side_id),
                )
            except KeyError:
                continue
            yaw = abs(self._relative_local_yaw(base_c2w, side_c2w))
            if yaw > 1.0e-6:
                bounds.append(float(yaw))
        if bounds:
            bound = min(bounds)
        else:
            bound = math.radians(float(self.camera_path_fallback_yaw_degrees))
        if self.camera_path_max_yaw_degrees is not None:
            bound = min(float(bound), math.radians(float(self.camera_path_max_yaw_degrees)))
        return max(float(bound), 0.0)

    def _side_camera_lateral_bound(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
        base_c2w: torch.Tensor,
    ) -> float:
        if self.camera_path_lateral_meters is not None:
            requested = abs(float(self.camera_path_lateral_meters))
        else:
            requested = float("inf")
        bounds: List[float] = []
        right_axis = base_c2w[:3, 0]
        base_t = base_c2w[:3, 3]
        for side_id in (int(self.camera_path_left_camera_id), int(self.camera_path_right_camera_id)):
            try:
                side_c2w, _side_K, _side_h, _side_w = self._interpolated_camera_record(
                    records=records,
                    sample=sample,
                    cam_id=int(side_id),
                )
            except KeyError:
                continue
            offset = float(torch.dot(side_c2w[:3, 3] - base_t, right_axis).detach().cpu().item())
            if abs(offset) > 1.0e-6:
                bounds.append(abs(offset))
        if bool(self.camera_path_clamp_to_side_cameras) and bounds:
            requested = min(float(requested), min(bounds))
        if not math.isfinite(requested):
            requested = float(self.camera_path_fallback_lateral_meters)
        return max(float(requested), 0.0)

    def _apply_camera_path(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
        c2w: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.camera_path_mode == "original":
            return c2w, {"signal": 0.0, "yaw_degrees": 0.0, "lateral_meters": 0.0}
        signal = self._camera_path_signal(sample)
        out = c2w.clone()
        yaw_rad = 0.0
        lateral_m = 0.0
        if self.camera_path_mode in {"sine_yaw", "sine_yaw_lateral"}:
            yaw_bound = self._side_camera_yaw_bound(records=records, sample=sample, base_c2w=c2w)
            yaw_rad = float(np.clip(float(signal), -1.0, 1.0)) * float(yaw_bound)
            out[:3, :3] = out[:3, :3] @ self._local_yaw_matrix(yaw_rad, device=out.device, dtype=out.dtype)
        if self.camera_path_mode in {"sine_lateral", "sine_yaw_lateral"}:
            lateral_bound = self._side_camera_lateral_bound(records=records, sample=sample, base_c2w=c2w)
            lateral_m = float(np.clip(float(signal), -1.0, 1.0)) * float(lateral_bound)
            out[:3, 3] = out[:3, 3] + out[:3, 0] * float(lateral_m)
        return out, {
            "signal": float(signal),
            "yaw_degrees": float(math.degrees(float(yaw_rad))),
            "lateral_meters": float(lateral_m),
        }

    def _sample_camera_path_metadata(
        self,
        *,
        records: Dict[ImageRef, RenderViewRecord],
        sample: RenderSample,
    ) -> Dict[str, float]:
        if self.camera_path_mode == "original":
            return {"signal": 0.0, "yaw_degrees": 0.0, "lateral_meters": 0.0}
        cam_id = int(self.camera_path_source_camera_id)
        if (int(sample.frame0), cam_id) not in records:
            cam_id = int(self.camera_ids[0])
        c2w, _K, _h, _w = self._interpolated_camera_record(records=records, sample=sample, cam_id=cam_id)
        _out, meta = self._apply_camera_path(records=records, sample=sample, c2w=c2w)
        return dict(meta)

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

    def _effective_transition_bounds(self, actual_window_len: int) -> Tuple[float, float]:
        n = int(actual_window_len)
        if n <= 1:
            return 0.0, float(max(n, 1))
        before = min(int(self.transition_frames_before), max(0, n - 1))
        after = min(int(self.transition_frames_after), max(0, n - before - 1))
        return float(before), float(n - after)

    def _build_render_samples(
        self,
        *,
        window_frame_ids: Sequence[int],
        next_frame_id: Optional[int],
        window_index: int,
        sequence_start_pos: int,
        timeline_start_pos: Optional[float] = None,
    ) -> List[RenderSample]:
        frames = [int(x) for x in window_frame_ids]
        if len(frames) < 1:
            return []
        samples: List[RenderSample] = []
        valid_rigid_frames = set(frames)
        interval_pairs: List[Tuple[int, int, int]] = [(i, frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        if bool(self.include_tail_interval) and next_frame_id is not None:
            interval_pairs.append((len(frames) - 1, int(frames[-1]), int(next_frame_id)))
        input_offsets = sorted({int(x) for x in self.input_offsets})
        stitch_start, stitch_end = self._effective_transition_bounds(actual_window_len=len(frames))
        timeline_base = float(sequence_start_pos if timeline_start_pos is None else timeline_start_pos)
        for interval_offset, f0, f1 in interval_pairs:
            for sub in range(int(self.subframes_per_interval)):
                alpha = float(sub) / float(self.subframes_per_interval)
                local_source_pos = float(interval_offset) + float(alpha)
                anchors = [int(x) for x in input_offsets if float(x) <= local_source_pos]
                if not anchors:
                    continue
                anchor_input_offset = int(anchors[-1])
                global_source_pos = float(timeline_base) + float(local_source_pos)
                global_time_seconds = float(global_source_pos) / float(self.source_fps)
                global_output_time_index = int(round(global_time_seconds * float(self.fps)))
                is_transition = bool(local_source_pos < stitch_start or local_source_pos >= stitch_end)
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
                        local_source_pos=float(local_source_pos),
                        global_source_pos=float(global_source_pos),
                        global_time_seconds=float(global_time_seconds),
                        global_output_time_index=int(global_output_time_index),
                        anchor_input_offset=int(anchor_input_offset),
                        is_transition=bool(is_transition),
                    )
                )
        if not bool(self.include_tail_interval) or next_frame_id is None:
            local_source_pos = float(len(frames) - 1)
            anchors = [int(x) for x in input_offsets if float(x) <= local_source_pos]
            if not anchors:
                return samples
            global_source_pos = float(timeline_base) + float(local_source_pos)
            global_time_seconds = float(global_source_pos) / float(self.source_fps)
            samples.append(
                RenderSample(
                    frame0=int(frames[-1]),
                    frame1=int(frames[-1]),
                    alpha=0.0,
                    rigid_frame_idx=int(frames[-1]),
                    window_index=int(window_index),
                    sequence_start_pos=int(sequence_start_pos),
                    local_source_pos=float(local_source_pos),
                    global_source_pos=float(global_source_pos),
                    global_time_seconds=float(global_time_seconds),
                    global_output_time_index=int(round(global_time_seconds * float(self.fps))),
                    anchor_input_offset=int(anchors[-1]),
                    is_transition=bool(local_source_pos < stitch_start or local_source_pos >= stitch_end),
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
            c2w, K, height, width = self._interpolated_camera_record(
                records=records,
                sample=sample,
                cam_id=int(cam_id),
            )
            c2w, path_meta = self._apply_camera_path(records=records, sample=sample, c2w=c2w)
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
                    "height": int(height),
                    "width": int(width),
                    "frame_idx": int(sample.rigid_frame_idx),
                    "camera_path": dict(path_meta),
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
        rgb, alpha = self.controller.trainer._render_scene_views_from_current_state(minimal, items)
        if bool(self.sky_enabled):
            rgb = self._compose_sky_rgb(minimal=minimal, items=items, scene_rgb=rgb, scene_alpha=alpha)
        out: Dict[int, np.ndarray] = {}
        for idx, cam_id in enumerate(self.camera_ids):
            out[int(cam_id)] = _to_uint8(rgb[idx])
        return out

    def _select_single_sky_state(self) -> Optional[Any]:
        if self.sky_branch is None:
            return None
        states = getattr(self.sky_branch, "node_states_sky", None)
        if isinstance(states, dict) and states:
            key = sorted(states.keys(), key=lambda x: str(x))[0]
            state = states[key]
            if bool(self.sky_reuse_single_state):
                states.clear()
                states[key] = state
                h_cache = getattr(self.sky_branch, "h_cache_sky", None)
                if isinstance(h_cache, dict):
                    h = h_cache.get(key)
                    h_cache.clear()
                    if h is not None:
                        h_cache[key] = h
            return state
        return None

    def _get_single_sky_state(self, minimal: Dict[str, Any]) -> Any:
        if self.sky_branch is None:
            raise ValueError("sky_branch is not loaded")
        if bool(self.sky_reuse_single_state) and self._single_sky_state is not None:
            return self._single_sky_state
        if self._single_sky_state is None:
            if hasattr(self.sky_branch, "get_or_init_node_state"):
                self._single_sky_state = self.sky_branch.get_or_init_node_state(minimal)
            else:
                raise ValueError("sky_branch lacks get_or_init_node_state()")
        return self._single_sky_state

    def _build_scene_pack_for_sky_update(self, minimal: Dict[str, Any]) -> Any:
        from models.streetforward.sky_branch.scene_render_provider import SceneRenderPack

        source_views = list(minimal.get("source_views") or [])
        source_images = list(minimal.get("source_images") or [])
        targets = list(minimal.get("targets") or [])
        if len(source_views) == 0 or len(source_images) == 0 or len(targets) == 0:
            raise ValueError("sky pre-render update requires source_views/source_images and targets in the minimal batch")
        if len(source_views) != len(source_images):
            raise ValueError(f"source_views/source_images length mismatch: {len(source_views)} vs {len(source_images)}")
        source_frame_idx = int(minimal.get("source_frame_idx", 0))
        source_items = [
            {
                "view": view,
                "gt_image": image,
                "frame_idx": int(source_frame_idx),
            }
            for view, image in zip(source_views, source_images)
        ]
        target_items = []
        for target in targets:
            item = {
                "view": target["view"],
                "gt_image": target["gt_image"],
                "frame_idx": int(target.get("frame_idx", target.get("target_frame_idx", source_frame_idx))),
            }
            if target.get("cam_idx") is not None:
                item["cam_idx"] = int(target.get("cam_idx"))
            target_items.append(item)
        trainer = self.controller.trainer
        src_rgb, src_alpha = trainer._render_scene_views_from_current_state(minimal, source_items)
        tgt_rgb, tgt_alpha = trainer._render_scene_views_from_current_state(minimal, target_items)
        return SceneRenderPack(
            source_rgb=src_rgb.detach(),
            source_alpha=src_alpha.detach(),
            target_rgb=tgt_rgb.detach(),
            target_alpha=tgt_alpha.detach(),
        )

    @torch.no_grad()
    def _pre_render_update_sky(self, *, minimal: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        if not bool(self.sky_enabled) or self.sky_branch is None:
            return {}
        if int(self.sky_pre_render_update_steps) <= 0:
            return {}
        if self._sky_pre_render_done and not bool(self.sky_pre_render_update_each_window):
            return {}
        if bool(self.sky_reset_runtime_per_window) and hasattr(self.sky_branch, "reset_runtime_state"):
            self.sky_branch.reset_runtime_state()
            self._single_sky_state = None
        logs: Dict[str, Any] = {
            "window_index": int(window_index),
            "steps": int(self.sky_pre_render_update_steps),
        }
        try:
            for step_idx in range(int(self.sky_pre_render_update_steps)):
                scene_pack = self._build_scene_pack_for_sky_update(minimal)
                out = self.sky_branch.forward_scene_batch(minimal, scene_pack, writeback=True)
                if bool(self.sky_reuse_single_state):
                    self._single_sky_state = out.node_state_sky
                logs.update(
                    {
                        "last_step_index": int(step_idx),
                        "last_loss": float(out.loss.detach().item()),
                        "last_skip_step": float(
                            out.logs.get("skip_step", 0.0).detach().item()
                            if torch.is_tensor(out.logs.get("skip_step", 0.0))
                            else out.logs.get("skip_step", 0.0)
                        ),
                    }
                )
                for key in ("sky_support_ratio", "sky_updated_node_ratio", "target_sky_valid_ratio"):
                    value = out.logs.get(key)
                    if value is not None:
                        logs[f"last_{key}"] = float(value.detach().item() if torch.is_tensor(value) else value)
                del out
                del scene_pack
            self._sky_pre_render_done = True
            logger.info(
                "sky pre-render update window=%d steps=%d loss=%.6f skip=%.3f support=%.4f",
                int(window_index),
                int(self.sky_pre_render_update_steps),
                float(logs.get("last_loss", 0.0)),
                float(logs.get("last_skip_step", 0.0)),
                float(logs.get("last_sky_support_ratio", 0.0)),
            )
            return logs
        except Exception as exc:
            if bool(self.sky_pre_render_fail_on_error):
                raise
            logger.warning("sky pre-render update failed for window=%d: %s", int(window_index), exc)
            return {
                "window_index": int(window_index),
                "steps": int(self.sky_pre_render_update_steps),
                "error": str(exc),
            }

    @torch.no_grad()
    def _compose_sky_rgb(
        self,
        *,
        minimal: Dict[str, Any],
        items: List[Dict[str, Any]],
        scene_rgb: torch.Tensor,
        scene_alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self.sky_branch is None:
            return scene_rgb
        state = self._get_single_sky_state(minimal)
        render_params = self.sky_branch.state_to_render_params(state)
        sky_rgbs: List[torch.Tensor] = []
        for item in items:
            sky_rgb, _sky_alpha = self.sky_branch.render_sky_single_view(
                render_params,
                item["view"],
                int(item["height"]),
                int(item["width"]),
            )
            sky_rgbs.append(sky_rgb.to(device=scene_rgb.device, dtype=scene_rgb.dtype))
        sky_rgb_stack = torch.stack(sky_rgbs, dim=0)
        if self.sky_compose_mode == "replace":
            return sky_rgb_stack.clamp(0.0, 1.0)
        alpha = scene_alpha.to(device=scene_rgb.device, dtype=scene_rgb.dtype)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(-1)
        if alpha.dim() == 4 and int(alpha.shape[-1]) != 1 and int(alpha.shape[1]) == 1:
            alpha = alpha.permute(0, 2, 3, 1).contiguous()
        if alpha.dim() != 4 or int(alpha.shape[-1]) != 1:
            raise ValueError(f"scene_alpha must be [V,H,W,1], got {tuple(scene_alpha.shape)}")
        sky_weight = (1.0 - alpha.clamp(0.0, 1.0)).mul(float(self.sky_alpha_scale)).clamp(0.0, 1.0)
        return (scene_rgb + sky_rgb_stack * sky_weight).clamp(0.0, 1.0)

    def _discard_history_keep_node_state(self) -> None:
        if not bool(self.discard_history_between_windows):
            return
        info = self.controller.scheduler.get_current_info()
        key = (int(info.get("scene_id", -1)), int(info.get("segment_id", -1)))
        keyed_cache_names = (
            "h_cache_bg",
            "h_cache_distant",
            "h_cache_rigid",
            "stage5_2_history_bg",
            "stage5_2_history_distant",
            "stage5_2_history_rigid",
            "stage5_2_last_step_update_norm",
            "stage5_2_block_support_bg",
            "stage5_2_block_support_distant",
            "stage5_2_block_support_rigid",
        )
        for name in keyed_cache_names:
            cache = getattr(self.controller.trainer, name, None)
            if isinstance(cache, dict):
                cache.pop(key, None)
        if hasattr(self.controller.trainer, "_stage5_2_last_full_inputs"):
            self.controller.trainer._stage5_2_last_full_inputs = None
        if hasattr(self.controller.trainer, "_stage5_6_active_cache"):
            self.controller.trainer._stage5_6_active_cache = None
        if hasattr(self.controller.trainer, "_stage5_6_last_fused_features"):
            self.controller.trainer._stage5_6_last_fused_features = {}
        if hasattr(self.controller.trainer, "_stage5_6_last_nearby_debug_images"):
            self.controller.trainer._stage5_6_last_nearby_debug_images = []
        if hasattr(self.controller.trainer, "_stage5_6_last_error_debug_images"):
            self.controller.trainer._stage5_6_last_error_debug_images = []
        if hasattr(self.controller.trainer, "_stage5_6_fusion_delta_norm_terms"):
            self.controller.trainer._stage5_6_fusion_delta_norm_terms = []
        for name in ("_stage5_6_frame_cache", "_stage5_6_fusion_delta_norm_terms"):
            value = getattr(self.controller.trainer, name, None)
            if isinstance(value, dict):
                value.clear()
        if hasattr(self.controller, "_recorded_block_update_counts"):
            self.controller._recorded_block_update_counts.clear()
        if hasattr(self.controller, "_clear_display_render_cache"):
            self.controller._clear_display_render_cache()

    def _list_scene_ids_for_export(self) -> List[int]:
        scheduler = self.controller.scheduler
        if hasattr(scheduler, "list_scene_ids"):
            return [int(x) for x in scheduler.list_scene_ids()]
        info = scheduler.get_current_info()
        return [int(info.get("scene_id", -1))]

    def _list_segment_ids_for_export(self, scene_id: int) -> List[int]:
        scheduler = self.controller.scheduler
        if hasattr(scheduler, "list_segment_ids"):
            return [int(x) for x in scheduler.list_segment_ids(int(scene_id))]
        info = scheduler.get_current_info()
        if int(info.get("scene_id", -1)) != int(scene_id):
            return []
        return [int(info.get("segment_id", -1))]

    def _select_window_plans(self) -> List[WindowPlan]:
        cur_info = self.controller.scheduler.get_current_info()
        cur_scene = int(cur_info.get("scene_id", -1))
        cur_segment = int(cur_info.get("segment_id", -1))
        cur_start = int(cur_info.get("sequence_start_pos", 0))

        if self.explicit_sequence_starts:
            plans = [
                WindowPlan(
                    scene_id=int(cur_scene),
                    segment_id=int(cur_segment),
                    sequence_start_pos=int(start),
                    timeline_start_pos=float(i * int(self.window_stride)),
                )
                for i, start in enumerate(self.explicit_sequence_starts)
            ]
            return plans[: int(self.max_windows)] if self.max_windows is not None and int(self.max_windows) > 0 else plans

        if not bool(self.multi_segment) or self.segment_policy == "current":
            frames = _frames_for_segment(self.dataset, cur_scene, cur_segment)
            starts = _window_starts_for_frames(
                num_frames=len(frames),
                window_size=int(self.window_size),
                window_stride=int(self.window_stride),
                require_full_window=bool(self.require_full_window),
                window_policy=str(self.window_policy),
            )
            if (
                int(cur_start) not in set(int(x) for x in starts)
                and int(cur_start) >= 0
                and len(frames[int(cur_start) : int(cur_start) + int(self.window_size)]) > 0
                and (
                    not bool(self.require_full_window)
                    or len(frames[int(cur_start) : int(cur_start) + int(self.window_size)]) >= int(self.window_size)
                )
            ):
                stop = (
                    len(frames) - int(self.window_size) + 1
                    if bool(self.require_full_window)
                    else len(frames)
                )
                starts = list(range(int(cur_start), int(stop), int(self.window_stride)))
            else:
                starts = [int(x) for x in starts if int(x) >= int(cur_start)]
            plans = [
                WindowPlan(
                    scene_id=int(cur_scene),
                    segment_id=int(cur_segment),
                    sequence_start_pos=int(start),
                    timeline_start_pos=float(int(start) - int(cur_start)),
                )
                for start in starts
            ]
            return plans[: int(self.max_windows)] if self.max_windows is not None and int(self.max_windows) > 0 else plans

        plans: List[WindowPlan] = []
        timeline_cursor = 0.0
        reached_initial = self.segment_policy != "from_initial"
        scene_ids = self._list_scene_ids_for_export()
        for scene_id in scene_ids:
            segment_ids = self._list_segment_ids_for_export(int(scene_id))
            for segment_id in segment_ids:
                if self.segment_policy == "from_initial" and not reached_initial:
                    if int(scene_id) != int(cur_scene) or int(segment_id) != int(cur_segment):
                        continue
                    reached_initial = True
                frames = _frames_for_segment(self.dataset, int(scene_id), int(segment_id))
                starts = _window_starts_for_frames(
                    num_frames=len(frames),
                    window_size=int(self.window_size),
                    window_stride=int(self.window_stride),
                    require_full_window=bool(self.require_full_window),
                    window_policy=str(self.window_policy),
                )
                if self.segment_policy == "from_initial" and int(scene_id) == int(cur_scene) and int(segment_id) == int(cur_segment):
                    if (
                        int(cur_start) not in set(int(x) for x in starts)
                        and int(cur_start) >= 0
                        and len(frames[int(cur_start) : int(cur_start) + int(self.window_size)]) > 0
                        and (
                            not bool(self.require_full_window)
                            or len(frames[int(cur_start) : int(cur_start) + int(self.window_size)]) >= int(self.window_size)
                        )
                    ):
                        stop = (
                            len(frames) - int(self.window_size) + 1
                            if bool(self.require_full_window)
                            else len(frames)
                        )
                        starts = list(range(int(cur_start), int(stop), int(self.window_stride)))
                    else:
                        starts = [int(x) for x in starts if int(x) >= int(cur_start)]
                    segment_origin = int(cur_start)
                else:
                    starts = [int(x) for x in starts]
                    segment_origin = int(starts[0]) if starts else 0
                for start in starts:
                    plans.append(
                        WindowPlan(
                            scene_id=int(scene_id),
                            segment_id=int(segment_id),
                            sequence_start_pos=int(start),
                            timeline_start_pos=float(timeline_cursor + (int(start) - int(segment_origin))),
                        )
                    )
                    if self.max_windows is not None and int(self.max_windows) > 0 and len(plans) >= int(self.max_windows):
                        return plans
                if starts:
                    timeline_cursor += float(max(0, len(frames) - int(segment_origin)))
        return plans

    def _set_window_scope(self, *, window_index: int, plan: WindowPlan) -> None:
        cur_info = self.controller.scheduler.get_current_info()
        cur_scene = int(cur_info.get("scene_id", -1))
        cur_segment = int(cur_info.get("segment_id", -1))
        cur_start = int(cur_info.get("sequence_start_pos", -1))
        same_scope = int(cur_scene) == int(plan.scene_id) and int(cur_segment) == int(plan.segment_id)
        if int(window_index) == 0 and bool(same_scope) and int(cur_start) == int(plan.sequence_start_pos):
            return
        if str(self.state_carryover) != "node_state" or not bool(same_scope):
            if hasattr(self.controller, "set_scope_and_sequence_start_pos"):
                self.controller.set_scope_and_sequence_start_pos(
                    int(plan.scene_id),
                    int(plan.segment_id),
                    int(plan.sequence_start_pos),
                )
            else:
                if not bool(same_scope):
                    self.controller.set_scope(int(plan.scene_id), int(plan.segment_id))
                self.controller.set_sequence_start_pos(int(plan.sequence_start_pos))
            return
        if getattr(self.controller, "busy", False):
            raise ValueError("controller is busy")
        self.controller.busy = True
        try:
            raw_batch = self.controller.scheduler.set_sequence_start_pos(int(plan.sequence_start_pos))
            self._discard_history_keep_node_state()
            self.controller._refresh_display_from_raw_batch(
                raw_batch,
                stats={
                    "manual_set_sequence_start_pos": 1.0,
                    "preserve_node_state_between_video_windows": 1.0,
                    "discard_history_between_video_windows": float(
                        1.0 if self.discard_history_between_windows else 0.0
                    ),
                },
            )
        finally:
            self.controller.busy = False

    def _save_all_images(
        self,
        *,
        frame_dir: Path,
        sample: RenderSample,
        combined: Optional[np.ndarray],
        per_camera: Dict[int, np.ndarray],
    ) -> None:
        role = "transition" if bool(sample.is_transition) else "stitch"
        stem = (
            f"t{int(sample.global_output_time_index):010d}_"
            f"w{int(sample.window_index):04d}_"
            f"src{float(sample.global_source_pos):010.3f}_{role}"
        )
        if combined is not None:
            Image.fromarray(combined).save(frame_dir / f"{stem}_combined.png")
        for cam_id, frame in per_camera.items():
            Image.fromarray(frame).save(frame_dir / f"{stem}_cam{int(cam_id)}.png")

    def export(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plans = self._select_window_plans()
        if len(plans) == 0:
            raise ValueError("no valid video windows for export")
        writers = _VideoWriterSet(
            output_dir=self.output_dir,
            base_name=self.base_name,
            fps=int(self.fps),
            camera_ids=self.camera_ids,
            camera_names=self.camera_names,
            write_combined=bool(self.write_combined),
            write_separate=bool(self.write_separate),
        )
        all_frames_dir = self.output_dir / "frames_all"
        if self.save_all_images:
            all_frames_dir.mkdir(parents=True, exist_ok=True)

        metadata: Dict[str, Any] = {
            "fps": int(self.fps),
            "target_fps": int(self.target_fps),
            "source_fps": float(self.source_fps),
            "subframes_per_source_interval": int(self.subframes_per_interval),
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "require_full_window": bool(self.require_full_window),
            "window_policy": str(self.window_policy),
            "multi_segment": bool(self.multi_segment),
            "segment_policy": str(self.segment_policy),
            "transition_frames_before": int(self.transition_frames_before),
            "transition_frames_after": int(self.transition_frames_after),
            "state_carryover": str(self.state_carryover),
            "discard_history_between_windows": bool(self.discard_history_between_windows),
            "sky_enabled": bool(self.sky_enabled),
            "sky_reuse_single_state": bool(self.sky_reuse_single_state),
            "sky_update_during_video": bool(self.sky_update_during_video),
            "sky_compose_mode": str(self.sky_compose_mode),
            "sky_pre_render_update_steps": int(self.sky_pre_render_update_steps),
            "sky_pre_render_update_each_window": bool(self.sky_pre_render_update_each_window),
            "input_offsets": [int(x) for x in self.input_offsets],
            "camera_ids": [int(x) for x in self.camera_ids],
            "camera_names": [str(x) for x in self.camera_names],
            "camera_path": {
                "mode": str(self.camera_path_mode),
                "source_camera_id": int(self.camera_path_source_camera_id),
                "left_camera_id": int(self.camera_path_left_camera_id),
                "right_camera_id": int(self.camera_path_right_camera_id),
                "amplitude": float(self.camera_path_amplitude),
                "cycles": float(self.camera_path_cycles),
                "period_frames": self.camera_path_period_frames,
                "clamp_to_side_cameras": bool(self.camera_path_clamp_to_side_cameras),
                "max_yaw_degrees": self.camera_path_max_yaw_degrees,
                "lateral_meters": self.camera_path_lateral_meters,
            },
            "windows": [
                {
                    "scene_id": int(plan.scene_id),
                    "segment_id": int(plan.segment_id),
                    "sequence_start_pos": int(plan.sequence_start_pos),
                    "timeline_start_pos": float(plan.timeline_start_pos),
                }
                for plan in plans
            ],
            "videos": {k: str(v) for k, v in writers.paths.items()},
            "all_frames_dir": str(all_frames_dir) if self.save_all_images else None,
            "sky_pre_render_updates": [],
            "samples": [],
        }
        total_rendered_samples = 0
        total_stitched_samples = 0
        try:
            for window_index, plan in enumerate(plans):
                self._set_window_scope(window_index=int(window_index), plan=plan)
                stats = self.controller.run_episode()
                scene_id = int(stats.get("scene_id", self.controller.scheduler.get_current_info().get("scene_id", -1)))
                segment_id = int(
                    stats.get("segment_id", self.controller.scheduler.get_current_info().get("segment_id", -1))
                )
                if int(scene_id) != int(plan.scene_id) or int(segment_id) != int(plan.segment_id):
                    raise ValueError(
                        "scheduler scope mismatch after selecting video window: "
                        f"expected scene={plan.scene_id} segment={plan.segment_id}, got scene={scene_id} segment={segment_id}"
                    )
                minimal_for_sky = self.controller.display.last_minimal_batch
                if isinstance(minimal_for_sky, dict):
                    sky_update = self._pre_render_update_sky(minimal=minimal_for_sky, window_index=int(window_index))
                    if sky_update:
                        metadata["sky_pre_render_updates"].append(dict(sky_update))
                segment_frames = _frames_for_segment(self.dataset, scene_id, segment_id)
                start = int(plan.sequence_start_pos)
                window_frames = segment_frames[int(start) : int(start) + int(self.window_size)]
                if len(window_frames) == 0:
                    raise ValueError(f"sequence_start_pos={start} has no frames")
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
                    timeline_start_pos=float(plan.timeline_start_pos),
                )
                logger.info(
                    "video window %d/%d scene=%d segment=%d start=%d timeline=%.3f frames=%s samples=%d",
                    int(window_index) + 1,
                    len(plans),
                    scene_id,
                    segment_id,
                    int(start),
                    float(plan.timeline_start_pos),
                    [int(x) for x in window_frames],
                    len(samples),
                )
                for sample in samples:
                    camera_path_meta = self._sample_camera_path_metadata(records=records, sample=sample)
                    per_camera = self._render_sample(records=records, sample=sample)
                    frames = [per_camera[int(c)] for c in self.camera_ids]
                    combined = _tile_frames(
                        frames,
                        layout=str(self.layout),
                        labels=self.camera_names,
                        show_labels=bool(self.show_labels),
                    )
                    if self.save_all_images:
                        self._save_all_images(
                            frame_dir=all_frames_dir,
                            sample=sample,
                            combined=combined,
                            per_camera=per_camera,
                        )
                    stitched_frame_index: Optional[int] = None
                    if not bool(sample.is_transition):
                        writers.append(
                            combined=combined if self.write_combined else None,
                            per_camera=per_camera,
                        )
                        stitched_frame_index = int(total_stitched_samples)
                        total_stitched_samples += 1
                    metadata["samples"].append(
                        {
                            "rendered_index": int(total_rendered_samples),
                            "stitched_frame_index": stitched_frame_index,
                            "window_index": int(sample.window_index),
                            "scene_id": int(scene_id),
                            "segment_id": int(segment_id),
                            "sequence_start_pos": int(sample.sequence_start_pos),
                            "timeline_start_pos": float(plan.timeline_start_pos),
                            "frame0": int(sample.frame0),
                            "frame1": int(sample.frame1),
                            "alpha": float(sample.alpha),
                            "rigid_frame_idx": int(sample.rigid_frame_idx),
                            "local_source_pos": float(sample.local_source_pos),
                            "global_source_pos": float(sample.global_source_pos),
                            "global_time_seconds": float(sample.global_time_seconds),
                            "global_output_time_index": int(sample.global_output_time_index),
                            "anchor_input_offset": int(sample.anchor_input_offset),
                            "is_transition": bool(sample.is_transition),
                            "camera_path": dict(camera_path_meta),
                        }
                    )
                    total_rendered_samples += 1
        finally:
            writers.close()

        metadata["num_rendered_frames_including_transition"] = int(total_rendered_samples)
        metadata["num_video_frames"] = int(total_stitched_samples)
        metadata_path = self.output_dir / f"{self.base_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        metadata["metadata_path"] = str(metadata_path)
        logger.info(
            "video export rendered %d frames (%d stitched video frames) to %s",
            int(total_rendered_samples),
            int(total_stitched_samples),
            self.output_dir,
        )
        return metadata
