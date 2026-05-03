from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from models.streetforward.minimal_trainer_stage5_4_production import (
    MinimalStreetForwardStage5_4_Production,
)
from streetforward_eval.episode_builder import TestEpisodeSpec, build_test_episode_specs
from streetforward_eval.metrics import MetricAccumulator
from streetforward_eval.protocols import protocol_from_dict, validate_protocol
from streetforward_eval.runner import RunnerRuntimeConfig, StreetForwardBatchEvalRunner
from streetforward_eval.snapshot_writer import RenderSaveConfig, SnapshotWriter
from streetforward_eval.summary import build_summary_rows, write_summary_csv
from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo

logger = logging.getLogger("streetforward_batcheval")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def load_cfg(config_file: str) -> Any:
    cfg = OmegaConf.load(config_file)
    base_cfg_file = cfg.get("base_config_file")
    if base_cfg_file:
        base = OmegaConf.load(str(base_cfg_file))
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: Any, name: str) -> List[Any]:
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    raise ValueError(f"{name} must be a non-empty list, got {type(value).__name__}")


def _as_mapping(value: Any, name: str) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")


def _build_model(cfg: Any, device: torch.device) -> Any:
    stage = str(cfg.model.stage).strip().lower()
    production_training = bool(cfg.model.get("production_training", False))
    if not production_training:
        raise ValueError(
            "BatchEval requires production trainer with eval_sparse interfaces; "
            "set model.production_training=true."
        )
    if stage == "5_4":
        model = MinimalStreetForwardStage5_4_Production(cfg, device=device).to(device)
    elif stage == "5_3":
        model = MinimalStreetForwardStage5_3_Production(cfg, device=device).to(device)
    else:
        raise ValueError(f"unsupported model.stage={cfg.model.stage!r}; expected '5_3' or '5_4'")
    return model


def _load_checkpoint(model: Any, ckpt_path: str, strict: bool) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=model.device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be dict-like, got {type(ckpt).__name__}")
    state = (
        ckpt.get("model")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt.get("net")
        or ckpt.get("module")
        or ckpt
    )
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state must be dict-like, got {type(state).__name__}")
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    try:
        incompatible = model.load_state_dict(state, strict=bool(strict))
    except RuntimeError as e:
        raise RuntimeError(f"failed to load checkpoint={ckpt_path} strict={bool(strict)}: {e}") from e
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    logger.info(
        "loaded checkpoint=%s strict=%s missing=%d unexpected=%d",
        ckpt_path,
        bool(strict),
        len(missing),
        len(unexpected),
    )
    if (not bool(strict)) and (len(missing) > 0 or len(unexpected) > 0):
        logger.warning(
            "non-strict checkpoint load mismatched keys (showing up to 20): missing=%s unexpected=%s",
            missing[:20],
            unexpected[:20],
        )
    model.eval()


def _snapshot_train_checkpoint_bytes(model: Any) -> bytes:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if hasattr(model, "optimizer") and getattr(model, "optimizer", None) is not None:
        payload["optimizer_state_dict"] = model.optimizer.state_dict()
    if hasattr(model, "build_light_checkpoint_extra"):
        try:
            step = int(getattr(getattr(model, "optimizer", None), "global_step", 0))
            payload.update(model.build_light_checkpoint_extra(step=step))
        except Exception as e:
            logger.warning("skip build_light_checkpoint_extra while snapshotting eval checkpoint: %s", e)
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _restore_train_checkpoint_bytes(model: Any, ckpt_bytes: bytes, device: torch.device) -> None:
    payload = torch.load(io.BytesIO(ckpt_bytes), map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if "optimizer_state_dict" in payload and hasattr(model, "optimizer") and getattr(model, "optimizer", None) is not None:
        if hasattr(model, "load_optimizer_state_from_checkpoint"):
            loaded = bool(model.load_optimizer_state_from_checkpoint(payload))
            if not loaded:
                logger.warning("optimizer restore via load_optimizer_state_from_checkpoint returned false; fallback to raw optimizer_state_dict.")
                model.optimizer.load_state_dict(payload["optimizer_state_dict"])
        else:
            model.optimizer.load_state_dict(payload["optimizer_state_dict"])


def _resolve_experiments(batch_eval_cfg: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    exps_any = batch_eval_cfg.get("experiments")
    exps_list = _as_list(exps_any, "batch_eval.experiments")
    if len(exps_list) == 0:
        raise ValueError("batch_eval.experiments must be a non-empty list")
    exps = [_as_mapping(_to_plain(x), "batch_eval.experiments[]") for x in exps_list]
    if args.experiment:
        names = {str(args.experiment)}
        exps = [x for x in exps if str(x.get("name")) in names]
    if args.experiments:
        names = set(str(x) for x in args.experiments)
        exps = [x for x in exps if str(x.get("name")) in names]
    if len(exps) == 0:
        raise ValueError("no experiments selected")
    return exps


def _collect_episode_scene_ids(cfg: Any) -> List[int]:
    scene_ids_any = cfg.batch_eval.dataset.get("scene_ids")
    scene_ids = _as_list(scene_ids_any, "batch_eval.dataset.scene_ids")
    if len(scene_ids) == 0:
        raise ValueError("batch_eval.dataset.scene_ids must be non-empty list[int]")
    return [int(x) for x in scene_ids]


def _scope_dataset_to_batch_eval_scene_ids(cfg: Any) -> None:
    scene_ids = _collect_episode_scene_ids(cfg)
    data_mode = str(cfg.batch_eval.get("data_mode", "segment_finetune_train")).strip()
    if data_mode != "segment_finetune_train":
        raise ValueError(
            f"unsupported batch_eval.data_mode={data_mode!r}; "
            "MultiSceneDatasetV4 batch eval currently uses segment_finetune_train assets."
        )
    if cfg.get("data") is None:
        raise ValueError("config.data is required")
    cfg.data.train_scene_ids = list(scene_ids)
    logger.info(
        "scoped dataset data.train_scene_ids to batch_eval.dataset.scene_ids: %s",
        scene_ids,
    )


def _resolve_output_root(cfg: Any, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(str(args.output_dir))
    output_dir_cfg = cfg.batch_eval.get("output_dir")
    if output_dir_cfg is None:
        raise ValueError("batch_eval.output_dir is required when --output_dir is not provided")
    return Path(str(output_dir_cfg))


def _run_one_experiment(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    exp_cfg: Dict[str, Any],
    output_root: Path,
    device: torch.device,
    max_total_episodes_override: int | None,
    base_ckpt_bytes: bytes | None,
    restore_checkpoint_on_segment: bool,
) -> None:
    global_cfg = _as_mapping(_to_plain(cfg.batch_eval), "batch_eval")
    protocol = protocol_from_dict(exp_cfg=exp_cfg, global_cfg=global_cfg)
    require_sparse4 = str(protocol.name) == "exp2_storm20_sparse4"
    validate_protocol(protocol, require_20frame_sparse4=require_sparse4)

    ds_cfg = cfg.batch_eval.dataset
    max_total_episodes = (
        int(max_total_episodes_override)
        if max_total_episodes_override is not None
        else (
            None
            if ds_cfg.get("max_total_episodes") is None
            else int(ds_cfg.get("max_total_episodes"))
        )
    )
    episode_specs = build_test_episode_specs(
        dataset=dataset,
        scene_ids=_collect_episode_scene_ids(cfg),
        protocol=protocol,
        segment_policy=str(ds_cfg.get("segment_policy", "all")),
        window_policy=str(ds_cfg.get("window_policy", "sliding")),
        stride=int(ds_cfg.get("stride", protocol.sequence_length)),
        require_full_window=bool(ds_cfg.get("require_full_window", True)),
        max_episodes_per_scene=(
            None if ds_cfg.get("max_episodes_per_scene") is None else int(ds_cfg.get("max_episodes_per_scene"))
        ),
        max_total_episodes=max_total_episodes,
    )
    if len(episode_specs) == 0:
        raise ValueError(f"experiment={protocol.name} produced no episode specs")

    exp_dir = output_root / str(protocol.name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config_resolved.yaml")
    OmegaConf.save(OmegaConf.create({"protocol": exp_cfg}), exp_dir / "protocol.yaml")

    metric_cfg = cfg.batch_eval.metrics
    metric_acc = MetricAccumulator(
        output_dir=exp_dir,
        protocol=protocol,
        min_valid_pixels=int(metric_cfg.get("min_valid_pixels", 32)),
        compute_psnr=bool(metric_cfg.get("compute_psnr", True)),
        compute_l1=bool(metric_cfg.get("compute_l1", True)),
        compute_ssim=bool(metric_cfg.get("compute_ssim", False)),
        compute_lpips=bool(metric_cfg.get("compute_lpips", False)),
    )
    render_cfg = _as_mapping(_to_plain(cfg.batch_eval.get("render", {})) or {}, "batch_eval.render")
    writer = SnapshotWriter(
        output_dir=exp_dir,
        save_cfg=RenderSaveConfig(
            save_png=bool(render_cfg.get("save_png", True)),
            save_numpy=bool(render_cfg.get("save_numpy", False)),
            save_video=bool(render_cfg.get("save_video", False)),
            save_depth_or_acc=bool(render_cfg.get("save_depth_or_acc", False)),
        ),
    )
    runtime = _as_mapping(_to_plain(cfg.batch_eval.get("runtime", {})) or {}, "batch_eval.runtime")
    history = _as_mapping(_to_plain(cfg.batch_eval.get("history", {})) or {}, "batch_eval.history")
    align_with_scheduler_v8 = bool(runtime.get("align_with_scheduler_v8", False))
    default_block_order = str(runtime.get("block_order", "block_major"))
    default_step_major_switch = int(runtime.get("step_major_switch_interval_steps", 1))
    default_reset_policy = str(runtime.get("reset_policy", "episode_end"))
    default_target_policy = str(runtime.get("target_frame_policy", "all_observed"))
    default_max_targets = runtime.get("max_target_frames_including_source")
    if align_with_scheduler_v8:
        sv8 = cfg.get("scheduler_v8")
        if sv8 is None:
            raise ValueError("batch_eval.runtime.align_with_scheduler_v8=true requires config.scheduler_v8.")
        sv8_exec = sv8.get("execution") if hasattr(sv8, "get") else None
        sv8_episode = sv8.get("episode") if hasattr(sv8, "get") else None
        if sv8_exec is not None:
            default_block_order = str(sv8_exec.get("block_order", default_block_order))
            default_step_major_switch = int(
                sv8_exec.get("step_major_switch_interval_steps", default_step_major_switch)
            )
            default_reset_policy = str(sv8_exec.get("reset_policy", default_reset_policy))
        default_target_policy = "visited_episode_frames"
        if sv8_episode is not None and sv8_episode.get("total_target_frames") is not None:
            default_max_targets = int(sv8_episode.get("total_target_frames"))
    runtime_cfg = RunnerRuntimeConfig(
        no_grad=bool(runtime.get("no_grad", True)),
        amp=bool(runtime.get("amp", True)),
        reset_state_per_episode=bool(runtime.get("reset_state_per_episode", True)),
        update_node_state=bool(runtime.get("update_node_state", True)),
        update_hidden_state=bool(runtime.get("update_hidden_state", True)),
        update_view_transient=bool(runtime.get("update_view_transient", True)),
        update_step_norm_ema=bool(history.get("update_step_norm_ema", True)),
        history_record_on_input_exit=bool(history.get("record_support_residual_on_input_exit", True)),
        history_record_each_step=bool(history.get("record_each_step", False)),
        block_order=str(default_block_order),
        step_major_switch_interval_steps=int(default_step_major_switch),
        reset_policy=str(default_reset_policy),
        target_frame_policy=str(default_target_policy),
        max_target_frames_including_source=(
            None if default_max_targets is None else int(default_max_targets)
        ),
    )
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=dataset,
        protocol=protocol,
        writer=writer,
        metric_acc=metric_acc,
        device=device,
        runtime_cfg=runtime_cfg,
    )

    if bool(restore_checkpoint_on_segment):
        if base_ckpt_bytes is None:
            raise ValueError("restore_checkpoint_on_segment=true but no base checkpoint snapshot is available.")
        _restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
        logger.info("[batcheval] restored base checkpoint before experiment=%s", protocol.name)

    logger.info("experiment=%s episodes=%d", protocol.name, len(episode_specs))
    prev_seg_key: tuple[int, int] | None = None
    for i, spec in enumerate(episode_specs):
        seg_key = (int(spec.scene_id), int(spec.segment_id))
        if bool(restore_checkpoint_on_segment) and prev_seg_key is not None and seg_key != prev_seg_key:
            if base_ckpt_bytes is None:
                raise ValueError("restore_checkpoint_on_segment=true but no base checkpoint snapshot is available.")
            _restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
            logger.info(
                "[batcheval] restored base checkpoint at segment boundary %s->%s",
                prev_seg_key,
                seg_key,
            )
        prev_seg_key = seg_key
        _ = runner.run_episode(spec)
        logger.info(
            "[batcheval] exp=%s episode=%d/%d uid=%s",
            protocol.name,
            int(i + 1),
            int(len(episode_specs)),
            spec.episode_uid,
        )

    metric_acc.write_csvs()
    final_rows: List[Dict[str, Any]] = []
    for rows in metric_acc.episode_rows.values():
        if len(rows) == 0:
            continue
        final_iter = max(int(r["global_iter"]) for r in rows)
        final_rows.extend([r for r in rows if int(r["global_iter"]) == int(final_iter)])
    write_summary_csv(exp_dir / "summary.csv", build_summary_rows(final_rows))
    logger.info(
        "experiment=%s wrote outputs to %s (png=%s metrics_iter=%s summary=%s final_views=%d)",
        protocol.name,
        exp_dir,
        exp_dir / "image",
        exp_dir / "metrics_iter.csv",
        exp_dir / "summary.csv",
        int(len(final_rows)),
    )
    if bool(restore_checkpoint_on_segment) and base_ckpt_bytes is not None:
        _restore_train_checkpoint_bytes(model, base_ckpt_bytes, device)
        logger.info("[batcheval] restored base checkpoint after experiment=%s", protocol.name)


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser("StreetForward BatchEval benchmark runner")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--experiments", type=str, nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_total_episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(str(args.config_file))
    if cfg.get("batch_eval") is None:
        raise ValueError("config.batch_eval is required")
    if bool(cfg.batch_eval.get("enable", False)) is not True:
        raise ValueError("batch_eval.enable must be true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _scope_dataset_to_batch_eval_scene_ids(cfg)
    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    dataset.initialize()

    model = _build_model(cfg, device)
    model.bind_eval_dataset(dataset)

    ckpt_cfg_path = cfg.batch_eval.checkpoint.get("path")
    ckpt_path = str(args.checkpoint or ckpt_cfg_path)
    if not ckpt_path:
        raise ValueError("checkpoint path must be provided by --checkpoint or batch_eval.checkpoint.path")
    _load_checkpoint(
        model,
        ckpt_path=ckpt_path,
        strict=bool(cfg.batch_eval.checkpoint.get("strict", True)),
    )
    runtime_cfg_any = cfg.batch_eval.get("runtime", {})
    runtime_cfg = _as_mapping(_to_plain(runtime_cfg_any), "batch_eval.runtime")
    restore_checkpoint_on_segment = bool(
        runtime_cfg.get(
            "restore_checkpoint_on_segment",
            runtime_cfg.get("reset_ckpt_per_segment", not bool(runtime_cfg.get("no_grad", True))),
        )
    )
    base_ckpt_bytes: bytes | None = None
    if restore_checkpoint_on_segment:
        base_ckpt_bytes = _snapshot_train_checkpoint_bytes(model)
        logger.info(
            "[batcheval] base checkpoint snapshot captured for segment-wise restore (bytes=%d)",
            len(base_ckpt_bytes),
        )

    output_root = _resolve_output_root(cfg, args)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_exps = _resolve_experiments(cfg.batch_eval, args)
    for exp_cfg in selected_exps:
        _run_one_experiment(
            cfg=cfg,
            dataset=dataset,
            model=model,
            exp_cfg=exp_cfg,
            output_root=output_root,
            device=device,
            max_total_episodes_override=args.max_total_episodes,
            base_ckpt_bytes=base_ckpt_bytes,
            restore_checkpoint_on_segment=bool(restore_checkpoint_on_segment),
        )


if __name__ == "__main__":
    main()
