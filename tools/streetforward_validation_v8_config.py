from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ValidationV8Config:
    eval_enable: bool
    validate_every_n_episodes: int
    run_at_train_start: bool
    mode: str
    steps_per_block: int
    blocks_per_episode: Optional[int]
    total_target_frames: Optional[int]
    block_order: str
    step_major_switch_interval_steps: int
    reset_policy: str
    target_policy: str
    episode_selection_policy: str
    save_images: bool
    save_dir: str
    persist_across_training: bool
    eval_scene_ids: List[int]
    use_sky_mask_regions: bool
    min_valid_pixels_per_region: int
    require_sky_mask: bool


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


def parse_validation_v8_config(cfg: Any) -> ValidationV8Config:
    raw = _cfg_get(cfg, "validation_v8", {}) or {}
    eval_enable = bool(_cfg_get(raw, "eval_enable", False))
    data_cfg = _cfg_get(cfg, "data")
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    if not eval_enable:
        return ValidationV8Config(
            eval_enable=False,
            validate_every_n_episodes=0,
            run_at_train_start=False,
            mode="inference_only",
            steps_per_block=1,
            blocks_per_episode=None,
            total_target_frames=None,
            block_order="block_major",
            step_major_switch_interval_steps=1,
            reset_policy="block_end",
            target_policy="visited_episode_frames",
            episode_selection_policy="middle",
            save_images=False,
            save_dir="validation/episodes",
            persist_across_training=False,
            eval_scene_ids=eval_scene_ids,
            use_sky_mask_regions=False,
            min_valid_pixels_per_region=32,
            require_sky_mask=False,
        )

    if len(eval_scene_ids) == 0:
        raise ValueError("validation_v8.eval_enable=true requires non-empty data.eval_scene_ids")
    offenders = _legacy_nondefault_forbidden(cfg)
    if offenders:
        raise ValueError(
            "validation_v8 forbids legacy test/eval split fields when enabled. "
            f"Found non-default fields: {offenders}"
        )

    trigger = _cfg_get(raw, "trigger", {}) or {}
    by = str(_cfg_get(trigger, "by", "train_episode_interval"))
    if by != "train_episode_interval":
        raise ValueError("validation_v8.trigger.by must be train_episode_interval")
    validate_every = int(_cfg_get(trigger, "validate_every_n_episodes"))
    if validate_every < 1:
        raise ValueError("validation_v8.trigger.validate_every_n_episodes must be >= 1")
    run_at_train_start = bool(_cfg_get(trigger, "run_at_train_start", False))

    mode = str(_cfg_get(raw, "mode", "inference_only")).strip()
    if mode not in ("inference_only", "segment_finetune_train"):
        raise ValueError(
            "validation_v8.mode must be one of ['inference_only', 'segment_finetune_train']"
        )

    block = _cfg_get(raw, "block", {}) or {}
    steps_per_block = int(_cfg_get(block, "steps_per_block", 1))
    if steps_per_block < 1:
        raise ValueError("validation_v8.block.steps_per_block must be >= 1")

    episode_cfg = _cfg_get(raw, "episode", {}) or {}
    blocks_per_episode_raw = _cfg_get(episode_cfg, "blocks_per_episode", None)
    total_target_frames_raw = _cfg_get(episode_cfg, "total_target_frames", None)
    target_policy = str(_cfg_get(episode_cfg, "target_policy", "visited_episode_frames")).strip()
    if target_policy != "visited_episode_frames":
        raise ValueError("validation_v8.episode.target_policy must be visited_episode_frames")
    blocks_per_episode: Optional[int]
    total_target_frames: Optional[int]
    if blocks_per_episode_raw is None:
        blocks_per_episode = None
    else:
        blocks_per_episode = int(blocks_per_episode_raw)
        if blocks_per_episode < 1:
            raise ValueError("validation_v8.episode.blocks_per_episode must be >= 1")
    if total_target_frames_raw is None:
        total_target_frames = None
    else:
        total_target_frames = int(total_target_frames_raw)
        if total_target_frames < 1:
            raise ValueError("validation_v8.episode.total_target_frames must be >= 1")
    if blocks_per_episode is not None and total_target_frames is not None:
        if total_target_frames > blocks_per_episode:
            raise ValueError(
                "validation_v8 does not use future frames; episode.total_target_frames must be <= blocks_per_episode"
            )

    execution = _cfg_get(raw, "execution", {}) or {}
    block_order = str(_cfg_get(execution, "block_order", "block_major")).strip()
    if block_order not in ("block_major", "step_major"):
        raise ValueError("validation_v8.execution.block_order must be one of ['block_major', 'step_major']")
    step_major_switch_interval_steps = int(_cfg_get(execution, "step_major_switch_interval_steps", 1))
    if step_major_switch_interval_steps < 1:
        raise ValueError("validation_v8.execution.step_major_switch_interval_steps must be >= 1")
    default_reset_policy = "episode_end" if block_order == "step_major" else "block_end"
    reset_policy = str(_cfg_get(execution, "reset_policy", default_reset_policy)).strip()
    if reset_policy not in ("block_end", "episode_end", "never"):
        raise ValueError("validation_v8.execution.reset_policy must be one of ['block_end', 'episode_end', 'never']")
    if block_order == "step_major" and reset_policy != "episode_end":
        raise ValueError("validation_v8.execution.reset_policy must be episode_end when block_order=step_major")

    episode_selection = _cfg_get(raw, "episode_selection", {}) or {}
    policy = str(_cfg_get(episode_selection, "policy", "middle"))
    if policy != "middle":
        raise ValueError("validation_v8.episode_selection.policy must be middle")

    render = _cfg_get(raw, "render", {}) or {}
    save_images = bool(_cfg_get(render, "save_images", True))
    save_dir = str(_cfg_get(render, "save_dir", "validation/episodes"))
    if not save_dir.strip():
        raise ValueError("validation_v8.render.save_dir must be non-empty")

    cache = _cfg_get(raw, "cache", {}) or {}
    persist = bool(_cfg_get(cache, "persist_across_training", True))
    metrics = _cfg_get(raw, "metrics", {}) or {}
    use_sky_mask_regions = bool(_cfg_get(metrics, "use_sky_mask_regions", False))
    min_valid_pixels_per_region = int(_cfg_get(metrics, "min_valid_pixels_per_region", 32))
    require_sky_mask = bool(_cfg_get(metrics, "require_sky_mask", use_sky_mask_regions))
    if min_valid_pixels_per_region < 1:
        raise ValueError("validation_v8.metrics.min_valid_pixels_per_region must be >= 1")
    if require_sky_mask and not use_sky_mask_regions:
        raise ValueError(
            "validation_v8.metrics.require_sky_mask=true requires validation_v8.metrics.use_sky_mask_regions=true"
        )

    return ValidationV8Config(
        eval_enable=True,
        validate_every_n_episodes=validate_every,
        run_at_train_start=run_at_train_start,
        mode=mode,
        steps_per_block=steps_per_block,
        blocks_per_episode=blocks_per_episode,
        total_target_frames=total_target_frames,
        block_order=block_order,
        step_major_switch_interval_steps=step_major_switch_interval_steps,
        reset_policy=reset_policy,
        target_policy=target_policy,
        episode_selection_policy=policy,
        save_images=save_images,
        save_dir=save_dir,
        persist_across_training=persist,
        eval_scene_ids=eval_scene_ids,
        use_sky_mask_regions=use_sky_mask_regions,
        min_valid_pixels_per_region=min_valid_pixels_per_region,
        require_sky_mask=require_sky_mask,
    )

