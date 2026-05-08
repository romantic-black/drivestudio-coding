from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

STREETFORWARD_ASSET_COORDINATE_FRAME = "seg0_camera_opencv"

WAYMO_DEFAULT_CAMERAS: Tuple[int, ...] = (0, 1, 2)
WAYMO_PIXEL_SOURCE_TYPE = "datasets.waymo.waymo_sourceloader.WaymoPixelSource"
WAYMO_LIDAR_SOURCE_TYPE = "datasets.waymo.waymo_sourceloader.WaymoLiDARSource"
WAYMO_REQUIRED_LIDAR_KEYS: Tuple[str, ...] = (
    "only_use_top_lidar",
    "truncated_max_range",
    "truncated_min_range",
    "lidar_downsample_factor",
    "lidar_percentile",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _cfg_has(cfg: Any, key: str) -> bool:
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        return key in cfg
    try:
        return key in cfg
    except Exception:
        return hasattr(cfg, key)


def _is_missing_or_empty(value: Any) -> bool:
    if value is None:
        return True
    if str(value) == "???":
        return True
    if not isinstance(value, (str, bytes, dict)):
        try:
            if len(value) == 0:
                return True
        except TypeError:
            pass
    return False


def _to_int_list(value: Any, *, field_name: str) -> List[int]:
    if value is None:
        return []
    try:
        return [int(x) for x in list(value)]
    except Exception as exc:
        raise ValueError(f"{field_name} must be a list of integers, got {value!r}") from exc


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _scene_dir(data_root: Path, scene_id: Any) -> Path:
    try:
        padded = data_root / f"{int(scene_id):03d}"
        if padded.exists():
            return padded
    except Exception:
        pass
    raw = data_root / str(scene_id)
    if raw.exists():
        return raw
    try:
        return data_root / f"{int(scene_id):03d}"
    except Exception:
        return raw


def _frame_range_for_scene(scene_dir: Path, data_cfg: Any) -> range:
    pose_dir = scene_dir / "ego_pose"
    if not pose_dir.exists():
        pose_dir = scene_dir / "lidar_pose"
    if not pose_dir.exists():
        raise ValueError(
            f"Waymo scene {scene_dir} is missing ego_pose/ or lidar_pose/; cannot infer frame range."
        )
    total_frames = len([p for p in pose_dir.iterdir() if p.suffix == ".txt"])
    if total_frames <= 0:
        raise ValueError(f"Waymo scene {scene_dir} has no pose txt files under {pose_dir}")

    start = int(_cfg_get(data_cfg, "start_timestep", 0))
    raw_end = int(_cfg_get(data_cfg, "end_timestep", -1))
    end_exclusive = total_frames if raw_end == -1 else raw_end + 1
    if start < 0 or start >= end_exclusive or end_exclusive > total_frames:
        raise ValueError(
            "Invalid Waymo frame range: "
            f"scene={scene_dir} start_timestep={start} end_timestep={raw_end} total_frames={total_frames}"
        )
    return range(start, end_exclusive)


def _validate_zero_based_contiguous_prefix(cameras: Sequence[int]) -> None:
    expected = list(range(len(cameras)))
    actual = [int(x) for x in cameras]
    if actual != expected:
        raise ValueError(
            "Waymo StreetForward export requires data.pixel_source.cameras to be a zero-based "
            f"contiguous prefix so V3 image refs stay aligned with camera ids. Got {actual}; "
            f"use {expected} for this camera count, typically [0, 1, 2]."
        )
    if len(actual) == 0 or len(actual) > 5:
        raise ValueError(f"Waymo cameras must contain 1..5 cameras, got {actual}")


def _validate_waymo_source_types(data_cfg: Any) -> None:
    pixel_source = _cfg_get(data_cfg, "pixel_source")
    lidar_source = _cfg_get(data_cfg, "lidar_source")
    pixel_type = str(_cfg_get(pixel_source, "type", ""))
    lidar_type = str(_cfg_get(lidar_source, "type", ""))
    if pixel_type != WAYMO_PIXEL_SOURCE_TYPE:
        raise ValueError(
            f"data.dataset=waymo requires data.pixel_source.type={WAYMO_PIXEL_SOURCE_TYPE!r}, "
            f"got {pixel_type!r}"
        )
    if lidar_type != WAYMO_LIDAR_SOURCE_TYPE:
        raise ValueError(
            f"data.dataset=waymo requires data.lidar_source.type={WAYMO_LIDAR_SOURCE_TYPE!r}, "
            f"got {lidar_type!r}"
        )


def _validate_waymo_lidar_config(data_cfg: Any) -> None:
    lidar_source = _cfg_get(data_cfg, "lidar_source")
    missing = [key for key in WAYMO_REQUIRED_LIDAR_KEYS if not _cfg_has(lidar_source, key)]
    if missing:
        raise ValueError(
            "Waymo StreetForward export requires explicit lidar_source fields "
            f"{list(WAYMO_REQUIRED_LIDAR_KEYS)}; missing {missing}"
        )


def _validate_waymo_scene_layout(
    *,
    data_root: Path,
    scene_ids: Sequence[Any],
    cameras: Sequence[int],
    data_cfg: Any,
) -> None:
    load_depth_maps = bool(_cfg_get(_cfg_get(data_cfg, "pixel_source"), "load_depth_maps", False))
    load_lidar = bool(_cfg_get(_cfg_get(data_cfg, "lidar_source"), "load_lidar", False))
    missing_depth: List[Tuple[Any, int, int, Path]] = []
    missing_basic: List[str] = []

    for scene_id in scene_ids:
        scene_dir = _scene_dir(data_root, scene_id)
        if not scene_dir.exists():
            raise ValueError(f"Waymo scene directory does not exist: {scene_dir}")
        for rel in ("images", "ego_pose", "extrinsics", "intrinsics"):
            if not (scene_dir / rel).exists():
                missing_basic.append(f"{scene_dir}/{rel}")
        if load_lidar and not (scene_dir / "lidar").exists():
            missing_basic.append(f"{scene_dir}/lidar")
        if load_depth_maps and not (scene_dir / "depth").exists():
            missing_basic.append(f"{scene_dir}/depth")
        if missing_basic:
            break

        frame_range = _frame_range_for_scene(scene_dir, data_cfg)
        start_frame = int(frame_range.start)
        for cam_id in cameras:
            for rel in (
                f"extrinsics/{int(cam_id)}.txt",
                f"intrinsics/{int(cam_id)}.txt",
                f"images/{start_frame:03d}_{int(cam_id)}.jpg",
            ):
                if not (scene_dir / rel).exists():
                    missing_basic.append(str(scene_dir / rel))
            if load_lidar and not (scene_dir / "lidar" / f"{start_frame:03d}.bin").exists():
                missing_basic.append(str(scene_dir / "lidar" / f"{start_frame:03d}.bin"))
        if missing_basic:
            break

        if not load_depth_maps:
            continue
        depth_dir = scene_dir / "depth"
        for frame_idx in frame_range:
            for cam_id in cameras:
                depth_path = depth_dir / f"{int(frame_idx):03d}_{int(cam_id)}.npy"
                if not depth_path.exists():
                    missing_depth.append((scene_id, int(frame_idx), int(cam_id), depth_path))
                    if len(missing_depth) >= 12:
                        break
            if len(missing_depth) >= 12:
                break
        if missing_depth:
            break

    if missing_basic:
        preview = missing_basic[:12]
        raise ValueError(
            "Waymo scene layout is missing required files/directories before export: "
            f"{preview}"
        )

    if missing_depth:
        side_cams = sorted({cam for _, _, cam, _ in missing_depth if cam in (3, 4)})
        preview = [
            f"scene={scene} frame={frame} cam={cam} path={path}"
            for scene, frame, cam, path in missing_depth[:8]
        ]
        hint = ""
        if side_cams:
            hint = (
                " Waymo side cameras 3/4 often do not have depth .npy in this processed data; "
                "use data.pixel_source.cameras=[0, 1, 2] or set load_depth_maps=false."
            )
        raise ValueError(f"Missing Waymo depth maps for selected cameras: {preview}.{hint}")


def _apply_waymo_defaults(cfg: Any) -> None:
    data_cfg = cfg.data
    pixel_source = data_cfg.pixel_source
    dataset_cfg = cfg.dataset
    pointcloud_cfg = dataset_cfg.pointcloud

    cameras_were_defaulted = False
    if _is_missing_or_empty(_cfg_get(pixel_source, "cameras")):
        _set_cfg_value(pixel_source, "cameras", list(WAYMO_DEFAULT_CAMERAS))
        cameras_were_defaulted = True
    if _cfg_get(pixel_source, "load_depth_maps") is None:
        _set_cfg_value(pixel_source, "load_depth_maps", True)
    if _is_missing_or_empty(_cfg_get(pointcloud_cfg, "monocular_chosen_cam_ids")):
        _set_cfg_value(pointcloud_cfg, "monocular_chosen_cam_ids", list(WAYMO_DEFAULT_CAMERAS))
    if cameras_were_defaulted:
        downscale = _cfg_get(pixel_source, "downscale_when_loading")
        if downscale is None:
            _set_cfg_value(pixel_source, "downscale_when_loading", [2 for _ in WAYMO_DEFAULT_CAMERAS])
        else:
            vals = list(downscale)
            if len(vals) != len(WAYMO_DEFAULT_CAMERAS):
                if len(vals) >= len(WAYMO_DEFAULT_CAMERAS):
                    vals = vals[: len(WAYMO_DEFAULT_CAMERAS)]
                else:
                    fill = vals[-1] if vals else 2
                    vals = vals + [fill for _ in range(len(WAYMO_DEFAULT_CAMERAS) - len(vals))]
                _set_cfg_value(pixel_source, "downscale_when_loading", vals)


def _validate_waymo_config(cfg: Any) -> None:
    _apply_waymo_defaults(cfg)

    data_cfg = cfg.data
    pixel_source = data_cfg.pixel_source
    dataset_cfg = cfg.dataset
    pointcloud_cfg = dataset_cfg.pointcloud

    cameras = _to_int_list(_cfg_get(pixel_source, "cameras"), field_name="data.pixel_source.cameras")
    _validate_zero_based_contiguous_prefix(cameras)
    _validate_waymo_source_types(data_cfg)
    _validate_waymo_lidar_config(data_cfg)

    downscale = _cfg_get(pixel_source, "downscale_when_loading")
    if downscale is not None and len(list(downscale)) != len(cameras):
        raise ValueError(
            "Waymo data.pixel_source.downscale_when_loading length must match selected cameras: "
            f"len={len(list(downscale))} cameras={cameras}"
        )

    chosen = _to_int_list(
        _cfg_get(pointcloud_cfg, "monocular_chosen_cam_ids"),
        field_name="dataset.pointcloud.monocular_chosen_cam_ids",
    )
    chosen_not_loaded = sorted(set(chosen) - set(cameras))
    if chosen_not_loaded:
        raise ValueError(
            "dataset.pointcloud.monocular_chosen_cam_ids must be a subset of data.pixel_source.cameras "
            f"for Waymo export. chosen={chosen} cameras={cameras} not_loaded={chosen_not_loaded}"
        )

    data_root_raw = _cfg_get(data_cfg, "data_root")
    if data_root_raw is None:
        raise ValueError("data.data_root is required for Waymo StreetForward export")
    scene_ids = list(_cfg_get(data_cfg, "train_scene_ids", []) or [])
    if not scene_ids:
        scene_idx = _cfg_get(data_cfg, "scene_idx")
        if scene_idx is not None:
            scene_ids = [scene_idx]
    if not scene_ids:
        raise ValueError("data.train_scene_ids is required for Waymo StreetForward asset export")

    _validate_waymo_scene_layout(
        data_root=Path(str(data_root_raw)),
        scene_ids=scene_ids,
        cameras=cameras,
        data_cfg=data_cfg,
    )


def prepare_streetforward_asset_export_config(cfg: Any) -> None:
    """Apply export-time dataset defaults and fail fast on known dataset-specific mismatches."""
    dataset = str(_cfg_get(cfg.data, "dataset", "")).lower()
    if dataset != "waymo":
        return
    _validate_waymo_config(cfg)


def build_streetforward_coordinate_metadata(dataset_name: str) -> Dict[str, Any]:
    dataset = str(dataset_name).lower()
    metadata: Dict[str, Any] = {
        "asset_coordinate_frame": STREETFORWARD_ASSET_COORDINATE_FRAME,
        "seg0_pose_contract": "camera0_first_frame",
    }
    if dataset == "waymo":
        metadata["source_dataset_world_coordinate_frame"] = "waymo_ego_aligned_x_front_y_left_z_up"
    elif dataset == "nuscenes":
        metadata["source_dataset_world_coordinate_frame"] = "nuscenes_front_camera_aligned_opencv"
    else:
        metadata["source_dataset_world_coordinate_frame"] = f"{dataset}_loader_world"
    return metadata
