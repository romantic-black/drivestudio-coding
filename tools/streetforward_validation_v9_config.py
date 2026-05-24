from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ValidationV9Config:
    eval_enable: bool
    phase: str
    fail_fast: bool

    validate_every_n_episodes: int
    run_at_train_start: bool
    eval_scene_ids: List[int]

    selection_seed: int
    segments_per_scene: int
    segment_policy: str
    resample_segments_every_validation: bool
    episode_policy: str
    resample_episode_every_validation: bool
    blocks_per_segment: int
    block_policy: str
    resample_blocks_every_validation: bool
    source_frame_policy: str

    k_values: List[int]
    max_K: int

    nearby_enable: bool
    nearby_frames_per_block: int
    nearby_policy: str
    nearby_same_keyframe_only: bool
    nearby_camera_policy: str
    nearby_adjacent_radius: int
    nearby_random_fill: bool

    compute_block_metrics: bool
    compute_nearby_metrics: bool
    compute_delta_stats: bool
    compute_runtime_stats: bool
    compute_memory_stats: bool
    compute_struct_stats: bool

    save_images: bool
    save_image_k_values: List[int]
    max_saved_blocks: int
    max_saved_cams: int
    save_dir: str

    block_loss_mask: str
    nearby_loss_mask: str
    min_valid_pixels: int
    require_sky_mask: bool
    require_egocar_mask: bool

    phase_b_reserved: bool
    phase_b_vsm_scope: str


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            value = cfg.get(key, default)
            return default if value is None else value
        except TypeError:
            pass
    if hasattr(cfg, key):
        value = getattr(cfg, key)
        return default if value is None else value
    return default


def _legacy_nondefault_forbidden(cfg: Any) -> Dict[str, Any]:
    data = _cfg_get(cfg, "data")
    pixel_source = _cfg_get(data, "pixel_source", {}) or {}
    multi_scene = _cfg_get(cfg, "multi_scene", {}) or {}
    eval_cfg = _cfg_get(cfg, "eval", {}) or {}
    offenders: Dict[str, Any] = {}
    test_stride = _cfg_get(pixel_source, "test_image_stride", 0)
    if test_stride not in (0, None):
        offenders["data.pixel_source.test_image_stride"] = test_stride
    max_test = _cfg_get(pixel_source, "max_test_images", 0)
    if max_test not in (0, None):
        offenders["data.pixel_source.max_test_images"] = max_test
    include_test = _cfg_get(multi_scene, "include_test", False)
    if bool(include_test):
        offenders["multi_scene.include_test"] = include_test
    run_test_at_end = _cfg_get(eval_cfg, "run_test_at_end", False)
    if bool(run_test_at_end):
        offenders["eval.run_test_at_end"] = run_test_at_end
    return offenders


def _disabled(eval_scene_ids: List[int]) -> ValidationV9Config:
    return ValidationV9Config(
        eval_enable=False,
        phase="phase_A_block_local_unroll",
        fail_fast=True,
        validate_every_n_episodes=0,
        run_at_train_start=False,
        eval_scene_ids=[int(x) for x in eval_scene_ids],
        selection_seed=20260524,
        segments_per_scene=1,
        segment_policy="random_seeded",
        resample_segments_every_validation=False,
        episode_policy="random_seeded",
        resample_episode_every_validation=False,
        blocks_per_segment=2,
        block_policy="random_without_replacement",
        resample_blocks_every_validation=False,
        source_frame_policy="middle_in_keyframe",
        k_values=[0, 4, 8, 16],
        max_K=16,
        nearby_enable=True,
        nearby_frames_per_block=1,
        nearby_policy="adjacent_then_random_same_keyframe",
        nearby_same_keyframe_only=True,
        nearby_camera_policy="all_cams",
        nearby_adjacent_radius=1,
        nearby_random_fill=True,
        compute_block_metrics=True,
        compute_nearby_metrics=True,
        compute_delta_stats=True,
        compute_runtime_stats=True,
        compute_memory_stats=True,
        compute_struct_stats=False,
        save_images=False,
        save_image_k_values=[0, 8, 16],
        max_saved_blocks=2,
        max_saved_cams=1,
        save_dir="validation_v9/phase_a",
        block_loss_mask="non_sky_non_egocar",
        nearby_loss_mask="non_sky_non_egocar",
        min_valid_pixels=32,
        require_sky_mask=True,
        require_egocar_mask=True,
        phase_b_reserved=True,
        phase_b_vsm_scope="bg_static",
    )


def parse_validation_v9_config(cfg: Any) -> ValidationV9Config:
    raw = _cfg_get(cfg, "validation_v9", {}) or {}
    data_cfg = _cfg_get(cfg, "data")
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    eval_enable = bool(_cfg_get(raw, "enable", _cfg_get(raw, "eval_enable", False)))
    if not eval_enable:
        return _disabled(eval_scene_ids)

    if len(eval_scene_ids) == 0:
        raise ValueError("validation_v9.enable=true requires non-empty data.eval_scene_ids")
    offenders = _legacy_nondefault_forbidden(cfg)
    if offenders:
        raise ValueError(
            "validation_v9 forbids legacy test/eval split fields when enabled. "
            f"Found non-default fields: {offenders}"
        )

    phase = str(_cfg_get(raw, "phase", "phase_A_block_local_unroll"))
    if phase != "phase_A_block_local_unroll":
        raise ValueError("P0 validation_v9 only supports phase_A_block_local_unroll")
    fail_fast = bool(_cfg_get(raw, "fail_fast", True))

    trigger = _cfg_get(raw, "trigger", {}) or {}
    by = str(_cfg_get(trigger, "by", "train_episode_interval"))
    if by != "train_episode_interval":
        raise ValueError("validation_v9.trigger.by must be train_episode_interval")
    validate_every = int(_cfg_get(trigger, "validate_every_n_episodes", 100))
    if validate_every < 1:
        raise ValueError("validation_v9.trigger.validate_every_n_episodes must be >= 1")
    run_at_train_start = bool(_cfg_get(trigger, "run_at_train_start", False))

    selection = _cfg_get(raw, "selection", {}) or {}
    selection_seed = int(_cfg_get(selection, "seed", 20260524))
    eval_scene_ids_from_data = bool(_cfg_get(selection, "eval_scene_ids_from_data", True))
    if not eval_scene_ids_from_data:
        raise ValueError("validation_v9 P0 requires selection.eval_scene_ids_from_data=true")
    segments_per_scene = int(_cfg_get(selection, "segments_per_scene", 1))
    if segments_per_scene != 1:
        raise ValueError("P0 validation_v9 supports segments_per_scene=1 only")
    segment_policy = str(_cfg_get(selection, "segment_policy", "random_seeded"))
    if segment_policy not in ("first", "middle", "random_seeded"):
        raise ValueError("validation_v9.selection.segment_policy must be first, middle, or random_seeded")
    episode_policy = str(_cfg_get(selection, "episode_policy", "random_seeded"))
    if episode_policy not in ("first", "middle", "random_seeded"):
        raise ValueError("validation_v9.selection.episode_policy must be first, middle, or random_seeded")
    block_policy = str(_cfg_get(selection, "block_policy", "random_without_replacement"))
    if block_policy not in ("random_without_replacement", "first_n", "middle_n"):
        raise ValueError("validation_v9.selection.block_policy must be random_without_replacement, first_n, or middle_n")
    if bool(_cfg_get(selection, "resample_segments_every_validation", False)):
        raise ValueError("validation_v9 P0 requires resample_segments_every_validation=false")
    if bool(_cfg_get(selection, "resample_episode_every_validation", False)):
        raise ValueError("validation_v9 P0 requires resample_episode_every_validation=false")
    if bool(_cfg_get(selection, "resample_blocks_every_validation", False)):
        raise ValueError("validation_v9 P0 requires resample_blocks_every_validation=false")
    blocks_per_segment = int(_cfg_get(selection, "blocks_per_segment", 2))
    if blocks_per_segment < 1:
        raise ValueError("validation_v9.selection.blocks_per_segment must be >= 1")
    source_frame_policy = str(_cfg_get(selection, "source_frame_policy", "middle_in_keyframe"))
    if source_frame_policy not in ("middle_in_keyframe", "first_in_keyframe", "random_seeded"):
        raise ValueError(
            "validation_v9.selection.source_frame_policy must be middle_in_keyframe, first_in_keyframe, or random_seeded"
        )

    phase_a = _cfg_get(raw, "phase_A", {}) or {}
    k_values = [int(x) for x in list(_cfg_get(phase_a, "k_values", [0, 4, 8, 16]) or [])]
    if not k_values:
        raise ValueError("validation_v9.phase_A.k_values must not be empty")
    if 0 not in k_values:
        raise ValueError("validation_v9.phase_A.k_values must include 0")
    max_K = int(_cfg_get(phase_a, "max_K", max(k_values)))
    if int(max(k_values)) != int(max_K):
        raise ValueError("validation_v9.phase_A.max_K must equal max(k_values)")
    if max_K < 1:
        raise ValueError("validation_v9.phase_A.max_K must be >= 1")
    k_values = sorted(set(k_values))

    nearby = _cfg_get(phase_a, "nearby", {}) or {}
    nearby_enable = bool(_cfg_get(nearby, "enable", True))
    nearby_frames_per_block = int(_cfg_get(nearby, "frames_per_block", 1))
    if nearby_frames_per_block < 0:
        raise ValueError("validation_v9.phase_A.nearby.frames_per_block must be >= 0")
    nearby_policy = str(_cfg_get(nearby, "policy", "adjacent_then_random_same_keyframe"))
    if nearby_policy != "adjacent_then_random_same_keyframe":
        raise ValueError("validation_v9.phase_A.nearby.policy must be adjacent_then_random_same_keyframe")
    nearby_same_keyframe_only = bool(_cfg_get(nearby, "same_keyframe_only", True))
    if not nearby_same_keyframe_only:
        raise ValueError("validation_v9.phase_A.nearby.same_keyframe_only must be true")
    nearby_camera_policy = str(_cfg_get(nearby, "camera_policy", "all_cams"))
    if nearby_camera_policy != "all_cams":
        raise ValueError("validation_v9.phase_A.nearby.camera_policy must be all_cams")
    nearby_adjacent_radius = int(_cfg_get(nearby, "adjacent_radius", 1))
    if nearby_adjacent_radius < 1:
        raise ValueError("validation_v9.phase_A.nearby.adjacent_radius must be >= 1")
    nearby_random_fill = bool(_cfg_get(nearby, "random_fill", True))

    metrics = _cfg_get(phase_a, "metrics", {}) or {}
    render = _cfg_get(phase_a, "render", {}) or {}
    save_images = bool(_cfg_get(render, "save_images", False))
    save_dir = str(_cfg_get(render, "save_dir", "validation_v9/phase_a"))
    if save_images and not save_dir.strip():
        raise ValueError("validation_v9.phase_A.render.save_dir must be non-empty")
    save_image_k_values = [int(x) for x in list(_cfg_get(render, "save_image_k_values", [0, max_K]) or [])]
    for k in save_image_k_values:
        if int(k) not in set(k_values):
            raise ValueError("validation_v9.phase_A.render.save_image_k_values must be subset of k_values")
    max_saved_blocks = int(_cfg_get(render, "max_saved_blocks", 2))
    max_saved_cams = int(_cfg_get(render, "max_saved_cams", 1))
    if max_saved_blocks < 0 or max_saved_cams < 0:
        raise ValueError("validation_v9 phase_A render max_saved values must be >= 0")

    masks = _cfg_get(raw, "masks", {}) or {}
    block_loss_mask = str(_cfg_get(masks, "block_loss_mask", "non_sky_non_egocar"))
    nearby_loss_mask = str(_cfg_get(masks, "nearby_loss_mask", "non_sky_non_egocar"))
    min_valid_pixels = int(_cfg_get(masks, "min_valid_pixels", 32))
    if min_valid_pixels < 1:
        raise ValueError("validation_v9.masks.min_valid_pixels must be >= 1")

    phase_b = _cfg_get(raw, "phase_B", {}) or {}
    phase_b_prefix = _cfg_get(phase_b, "prefix_render", {}) or {}
    phase_b_query = _cfg_get(phase_b, "query_observation", {}) or {}
    if bool(_cfg_get(phase_b_prefix, "enable", False)):
        raise ValueError("validation_v9 Phase A P0 requires phase_B.prefix_render.enable=false")
    if bool(_cfg_get(phase_b_query, "enable", False)):
        raise ValueError("validation_v9 Phase A P0 requires phase_B.query_observation.enable=false")
    phase_b_masks = _cfg_get(phase_b, "masks", {}) or {}

    return ValidationV9Config(
        eval_enable=True,
        phase=phase,
        fail_fast=fail_fast,
        validate_every_n_episodes=validate_every,
        run_at_train_start=run_at_train_start,
        eval_scene_ids=eval_scene_ids,
        selection_seed=selection_seed,
        segments_per_scene=segments_per_scene,
        segment_policy=segment_policy,
        resample_segments_every_validation=bool(_cfg_get(selection, "resample_segments_every_validation", False)),
        episode_policy=episode_policy,
        resample_episode_every_validation=bool(_cfg_get(selection, "resample_episode_every_validation", False)),
        blocks_per_segment=blocks_per_segment,
        block_policy=block_policy,
        resample_blocks_every_validation=bool(_cfg_get(selection, "resample_blocks_every_validation", False)),
        source_frame_policy=source_frame_policy,
        k_values=k_values,
        max_K=max_K,
        nearby_enable=nearby_enable,
        nearby_frames_per_block=nearby_frames_per_block,
        nearby_policy=nearby_policy,
        nearby_same_keyframe_only=nearby_same_keyframe_only,
        nearby_camera_policy=nearby_camera_policy,
        nearby_adjacent_radius=nearby_adjacent_radius,
        nearby_random_fill=nearby_random_fill,
        compute_block_metrics=bool(_cfg_get(metrics, "compute_block_metrics", True)),
        compute_nearby_metrics=bool(_cfg_get(metrics, "compute_nearby_metrics", True)),
        compute_delta_stats=bool(_cfg_get(metrics, "compute_delta_stats", True)),
        compute_runtime_stats=bool(_cfg_get(metrics, "compute_runtime_stats", True)),
        compute_memory_stats=bool(_cfg_get(metrics, "compute_memory_stats", True)),
        compute_struct_stats=bool(_cfg_get(metrics, "compute_struct_stats", False)),
        save_images=save_images,
        save_image_k_values=save_image_k_values,
        max_saved_blocks=max_saved_blocks,
        max_saved_cams=max_saved_cams,
        save_dir=save_dir,
        block_loss_mask=block_loss_mask,
        nearby_loss_mask=nearby_loss_mask,
        min_valid_pixels=min_valid_pixels,
        require_sky_mask=bool(_cfg_get(masks, "require_sky_mask", True)),
        require_egocar_mask=bool(_cfg_get(masks, "require_egocar_mask", True)),
        phase_b_reserved=bool(_cfg_get(phase_b, "reserved", True)),
        phase_b_vsm_scope=str(_cfg_get(phase_b_masks, "vsm_scope", "bg_static")),
    )


__all__ = ["ValidationV9Config", "parse_validation_v9_config"]
