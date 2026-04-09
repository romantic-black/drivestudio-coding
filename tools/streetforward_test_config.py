from __future__ import annotations

from typing import Any, Dict, List


def _as_bool(cfg: Any, key: str, parent: str) -> bool:
    if key not in cfg:
        raise ValueError(f"{parent}.{key} is required")
    return bool(cfg[key])


def _as_int(cfg: Any, key: str, parent: str) -> int:
    if key not in cfg:
        raise ValueError(f"{parent}.{key} is required")
    return int(cfg[key])


def _as_str(cfg: Any, key: str, parent: str) -> str:
    if key not in cfg:
        raise ValueError(f"{parent}.{key} is required")
    v = str(cfg[key]).strip()
    if not v:
        raise ValueError(f"{parent}.{key} must be non-empty")
    return v


def resolve_eval_scene_ids(cfg: Any) -> List[int]:
    if cfg.get("data") is None:
        raise ValueError("config.data is required")
    raw = cfg.data.get("eval_scene_ids")
    if raw is None:
        return []
    return [int(x) for x in list(raw)]


def validate_test_config(cfg: Any) -> Dict[str, Any]:
    """
    Validate formal test runner config with fast-fail semantics.
    """
    t = cfg.get("test")
    if t is None:
        raise ValueError("config.test is required for formal test runner")
    if not _as_bool(t, "enable", "test"):
        raise ValueError("test.enable must be true when running the test runner")

    mode = _as_str(t, "mode", "test")
    if mode not in ("adapt_supervised", "inference_only", "both"):
        raise ValueError("test.mode must be one of ['adapt_supervised', 'inference_only', 'both']")

    runner = t.get("runner")
    if runner is None:
        raise ValueError("test.runner is required")
    _ = _as_bool(runner, "deterministic", "test.runner")
    _ = _as_int(runner, "seed", "test.runner")
    source_protocol = _as_str(runner, "source_protocol", "test.runner")
    if source_protocol not in ("first_train_frame_cam0", "middle_keyframe_middle_frame_cam0"):
        raise ValueError(
            "test.runner.source_protocol must be one of "
            "['first_train_frame_cam0', 'middle_keyframe_middle_frame_cam0']"
        )
    max_segments_per_scene = _as_int(runner, "max_segments_per_scene", "test.runner")
    if max_segments_per_scene < 0:
        raise ValueError("test.runner.max_segments_per_scene must be >= 0")
    min_test_views = _as_int(runner, "min_test_views_per_segment", "test.runner")
    if min_test_views < 1:
        raise ValueError("test.runner.min_test_views_per_segment must be >= 1")

    split = t.get("split")
    if split is None:
        raise ValueError("test.split is required")
    req_eval_scene_ids = _as_bool(split, "require_eval_scene_ids", "test.split")
    req_exhaustive_test_refs = _as_bool(split, "require_exhaustive_test_refs", "test.split")
    req_nonempty_test = _as_bool(split, "require_nonempty_test_views", "test.split")
    allow_overlap_stride_zero = _as_bool(split, "allow_train_test_overlap_when_stride_zero", "test.split")

    eval_scene_ids = resolve_eval_scene_ids(cfg)
    if req_eval_scene_ids and len(eval_scene_ids) == 0:
        raise ValueError("data.eval_scene_ids must be non-empty when test.split.require_eval_scene_ids=true")

    if cfg.get("data") is None or cfg.data.get("pixel_source") is None:
        raise ValueError("config.data.pixel_source is required")
    pixel = cfg.data.pixel_source
    test_stride = _as_int(pixel, "test_image_stride", "data.pixel_source")
    max_test_images = _as_int(pixel, "max_test_images", "data.pixel_source")
    if test_stride < 0:
        raise ValueError("data.pixel_source.test_image_stride must be >= 0")
    if req_exhaustive_test_refs and max_test_images != 0:
        raise ValueError(
            "data.pixel_source.max_test_images must be 0 when "
            "test.split.require_exhaustive_test_refs=true"
        )
    if test_stride == 0 and not allow_overlap_stride_zero:
        raise ValueError(
            "test_image_stride=0 requires "
            "test.split.allow_train_test_overlap_when_stride_zero=true"
        )

    adapt = t.get("adapt_supervised")
    if adapt is None:
        raise ValueError("test.adapt_supervised is required")
    adapt_enable = _as_bool(adapt, "enable", "test.adapt_supervised")
    adapt_steps = _as_int(adapt, "max_steps_per_segment", "test.adapt_supervised")
    if adapt_steps < 1:
        raise ValueError("test.adapt_supervised.max_steps_per_segment must be >= 1")
    validate_every = _as_int(adapt, "validate_every_blocks", "test.adapt_supervised")
    if validate_every < 1:
        raise ValueError("test.adapt_supervised.validate_every_blocks must be >= 1")
    early_stop_patience = _as_int(adapt, "early_stop_patience", "test.adapt_supervised")
    if early_stop_patience < 0:
        raise ValueError("test.adapt_supervised.early_stop_patience must be >= 0")
    keep_best_by = _as_str(adapt, "keep_best_by", "test.adapt_supervised")
    if keep_best_by not in ("psnr", "ssim", "lpips"):
        raise ValueError("test.adapt_supervised.keep_best_by must be one of ['psnr', 'ssim', 'lpips']")
    _ = _as_bool(adapt, "reset_runtime_state_each_segment", "test.adapt_supervised")

    inf = t.get("inference_only")
    if inf is None:
        raise ValueError("test.inference_only is required")
    inf_enable = _as_bool(inf, "enable", "test.inference_only")
    _ = _as_bool(inf, "allow_hidden_cache_update", "test.inference_only")
    _ = _as_bool(inf, "allow_node_state_writeback", "test.inference_only")
    eval_trigger = _as_str(inf, "eval_trigger", "test.inference_only")
    if eval_trigger != "episode_end":
        raise ValueError("test.inference_only.eval_trigger must be 'episode_end'")
    aggregate_mode = _as_str(inf, "aggregate_across_episodes", "test.inference_only")
    if aggregate_mode != "mean":
        raise ValueError("test.inference_only.aggregate_across_episodes must be 'mean'")
    max_episodes_per_segment = _as_int(inf, "max_episodes_per_segment", "test.inference_only")
    if max_episodes_per_segment < 0:
        raise ValueError("test.inference_only.max_episodes_per_segment must be >= 0")
    _ = _as_bool(inf, "save_per_episode_metrics_json", "test.inference_only")
    _ = _as_bool(inf, "save_per_episode_per_view_metrics_json", "test.inference_only")

    exp = t.get("export")
    if exp is None:
        raise ValueError("test.export is required")
    for key in (
        "save_3dgs_init",
        "save_3dgs_best",
        "save_3dgs_final",
        "save_ply",
        "save_rendered_images",
        "save_per_view_metrics_json",
    ):
        _ = _as_bool(exp, key, "test.export")
    if bool(exp.get("save_ply")):
        raise ValueError(
            "test.export.save_ply must be false. "
            "PLY export is a lossy visualization format and cannot preserve complete 3DGS state; "
            "use save_3dgs_init/best/final (.pt) for full-fidelity export."
        )

    if mode == "adapt_supervised" and not adapt_enable:
        raise ValueError("test.mode=adapt_supervised requires test.adapt_supervised.enable=true")
    if mode == "inference_only" and not inf_enable:
        raise ValueError("test.mode=inference_only requires test.inference_only.enable=true")
    if mode == "both" and (not adapt_enable or not inf_enable):
        raise ValueError("test.mode=both requires both adapt_supervised.enable and inference_only.enable to be true")

    return {
        "mode": mode,
        "eval_scene_ids": eval_scene_ids,
        "max_segments_per_scene": max_segments_per_scene,
        "source_protocol": source_protocol,
        "min_test_views_per_segment": min_test_views,
        "adapt_max_steps_per_segment": adapt_steps,
        "adapt_validate_every_blocks": validate_every,
        "adapt_early_stop_patience": early_stop_patience,
        "adapt_keep_best_by": keep_best_by,
        "pixel_test_image_stride": test_stride,
        "pixel_max_test_images": max_test_images,
        "require_nonempty_test_views": req_nonempty_test,
        "allow_train_test_overlap_when_stride_zero": allow_overlap_stride_zero,
        "inference_eval_trigger": eval_trigger,
        "inference_aggregate_across_episodes": aggregate_mode,
        "inference_max_episodes_per_segment": max_episodes_per_segment,
    }


def validate_dataset_test_split_or_raise(dataset: Any, test_cfg: Dict[str, Any]) -> None:
    """
    Validate dataset-level split semantics after dataset initialization.
    Fast-fail on first invalid eval segment.
    """
    eval_scene_ids = [int(x) for x in list(test_cfg["eval_scene_ids"])]
    stride = int(test_cfg["pixel_test_image_stride"])
    require_nonempty = bool(test_cfg["require_nonempty_test_views"])

    for scene_id in eval_scene_ids:
        scene_data = dataset.get_scene(int(scene_id))
        if scene_data is None:
            raise ValueError(f"scene_id={scene_id} cannot be loaded for test split validation")
        segments = list(scene_data.get("segments") or [])
        for segment_id, segment in enumerate(segments):
            train_frames = list(segment.get("frame_indices") or [])
            test_frames = list(segment.get("test_frame_indices") or [])
            if stride > 0 and len(train_frames) == 0:
                raise ValueError(
                    f"test split invalid: stride>0 but train_frame_indices is empty "
                    f"(scene={scene_id} segment={segment_id})"
                )
            if stride > 0 and len(test_frames) == 0:
                raise ValueError(
                    f"test split invalid: stride>0 but test_frame_indices is empty "
                    f"(scene={scene_id} segment={segment_id})"
                )
            if require_nonempty and len(test_frames) == 0:
                raise ValueError(
                    f"test split invalid: require_nonempty_test_views=true but "
                    f"test_frame_indices is empty (scene={scene_id} segment={segment_id})"
                )


def ensure_dataset_initialized_for_test(dataset: Any, cfg: Any) -> None:
    """
    Ensure dataset can be used by test runner under eval-only configs.
    """
    dataset.initialize()
    eval_ids = resolve_eval_scene_ids(cfg)
    train_ids = [int(x) for x in list(cfg.data.get("train_scene_ids", []))]
    if getattr(dataset, "_initialized", False):
        return
    if len(train_ids) == 0 and len(eval_ids) > 0:
        # Base MultiSceneDataset.initialize() may return early when training queue is empty.
        # Formal test runner can still operate on eval scenes; mark initialized explicitly.
        dataset._initialized = True
        return
    raise ValueError(
        "Dataset initialize() did not complete; check train/eval scene settings and dataset validity."
    )

