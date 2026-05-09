from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from models.streetforward.minimal_trainer_stage4_6 import MinimalStreetForwardStage4_6
from models.streetforward.minimal_trainer_stage5_0 import MinimalStreetForwardStage5_0
from models.streetforward.minimal_trainer_stage5_2 import MinimalStreetForwardStage5_2
from models.streetforward.minimal_trainer_stage5_3 import MinimalStreetForwardStage5_3
from models.streetforward.minimal_trainer_stage5_3_production import (
    MinimalStreetForwardStage5_3_Production,
)
from models.streetforward.minimal_trainer_stage5_4_production import (
    MinimalStreetForwardStage5_4_Production,
)
from models.streetforward.minimal_trainer_stage5_5_production import (
    MinimalStreetForwardStage5_5_Production,
)
from models.streetforward.minimal_trainer_stage5_6_production import (
    MinimalStreetForwardStage5_6_Production,
)
from datasets.validation_scheduler_v7 import build_validation_episode_specs_v7
from streetforward_eval.episode_builder import TestEpisodeSpec, build_test_episode_specs
from streetforward_eval.metrics import MetricAccumulator
from streetforward_eval.protocols import protocol_from_dict, resolve_eval_offsets, validate_protocol
from streetforward_eval.runner import RunnerRuntimeConfig, StreetForwardBatchEvalRunner
from streetforward_eval.stage5_6_runtime import configure_segment_finetune_optimizer
from streetforward_eval.snapshot_writer import RenderSaveConfig, SnapshotWriter
from streetforward_eval.summary import build_summary_rows, write_summary_csv

logger = logging.getLogger("streetforward_batcheval")


class _BatchEvalOptimizerAdapter:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip_norm: float = 0.0,
        global_step: int = 0,
    ) -> None:
        self._optimizer = optimizer
        self.grad_clip_norm = float(grad_clip_norm)
        self.global_step = int(global_step)
        self.last_grad_norm = 0.0

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state = self._optimizer.state_dict()
        state["_sf_global_step"] = int(self.global_step)
        state["_sf_last_grad_norm"] = float(self.last_grad_norm)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        state = dict(state_dict)
        self.global_step = int(state.pop("_sf_global_step", 0))
        self.last_grad_norm = float(state.pop("_sf_last_grad_norm", 0.0))
        return self._optimizer.load_state_dict(state)

    def step(self, *args, **kwargs):
        params = [p for group in self._optimizer.param_groups for p in group["params"]]
        if float(self.grad_clip_norm) > 0.0 and len(params) > 0:
            total = torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.grad_clip_norm), error_if_nonfinite=True)
            self.last_grad_norm = float(total.item() if torch.is_tensor(total) else total)
        else:
            sq = 0.0
            for p in params:
                if p.grad is None:
                    continue
                sq += float(p.grad.detach().float().pow(2).sum().item())
            self.last_grad_norm = float(sq ** 0.5)
        out = self._optimizer.step(*args, **kwargs)
        self.global_step += 1
        return out


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
    if stage == "4_6":
        model = MinimalStreetForwardStage4_6(cfg, device=device).to(device)
    elif stage == "5_0":
        model = MinimalStreetForwardStage5_0(cfg, device=device).to(device)
    elif stage == "5_2":
        model = MinimalStreetForwardStage5_2(cfg, device=device).to(device)
    elif stage == "5_3" and not production_training:
        model = MinimalStreetForwardStage5_3(cfg, device=device).to(device)
    elif stage == "5_3":
        model = MinimalStreetForwardStage5_3_Production(cfg, device=device).to(device)
    elif stage == "5_4":
        if not production_training:
            raise ValueError("Stage5_4 BatchEval requires model.production_training=true.")
        model = MinimalStreetForwardStage5_4_Production(cfg, device=device).to(device)
    elif stage == "5_5":
        if not production_training:
            raise ValueError("Stage5_5 BatchEval requires model.production_training=true.")
        model = MinimalStreetForwardStage5_5_Production(cfg, device=device).to(device)
    elif stage == "5_6":
        if not production_training:
            raise ValueError("Stage5_6 BatchEval requires model.production_training=true.")
        model = MinimalStreetForwardStage5_6_Production(cfg, device=device).to(device)
    else:
        raise ValueError(
            f"unsupported model.stage={cfg.model.stage!r}; "
            "expected '4_6', '5_0', '5_2', '5_3', '5_4', '5_5', or '5_6'"
        )
    return model


def _checkpoint_step(payload: Dict[str, Any]) -> int:
    for key in ("global_step", "step", "iteration", "iter"):
        if payload.get(key) is not None:
            try:
                return int(payload.get(key))
            except Exception:
                pass
    lr_info = payload.get("lr_scheduler")
    if isinstance(lr_info, dict) and lr_info.get("global_step") is not None:
        try:
            return int(lr_info.get("global_step"))
        except Exception:
            pass
    opt_state = payload.get("optimizer_state_dict")
    if isinstance(opt_state, dict) and opt_state.get("_sf_global_step") is not None:
        try:
            return int(opt_state.get("_sf_global_step"))
        except Exception:
            pass
    return 0


def _load_checkpoint(model: Any, ckpt_path: str, strict: bool) -> int:
    ckpt = torch.load(str(ckpt_path), map_location=model.device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be dict-like, got {type(ckpt).__name__}")
    ckpt_step = _checkpoint_step(ckpt)
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
    setattr(model, "global_step", int(ckpt_step))
    opt = getattr(model, "optimizer", None)
    if opt is not None and hasattr(opt, "global_step"):
        try:
            opt.global_step = int(ckpt_step)
        except Exception:
            logger.warning("failed to set optimizer.global_step from checkpoint step=%s", ckpt_step)
    model.eval()
    return int(ckpt_step)


def _load_checkpoint_optimizer_state(model: Any, ckpt_path: str, device: torch.device) -> bool:
    opt = getattr(model, "optimizer", None)
    if opt is None:
        raise ValueError("segment_finetune_train requires model.optimizer to restore checkpoint optimizer state")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be dict-like, got {type(ckpt).__name__}")
    opt_state = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
    if opt_state is None:
        logger.warning("[batcheval] checkpoint has no optimizer_state_dict; using freshly initialized optimizer")
        return False
    if hasattr(model, "load_optimizer_state_from_checkpoint"):
        loaded = bool(model.load_optimizer_state_from_checkpoint(ckpt))
        if loaded:
            logger.info("[batcheval] restored optimizer state through model.load_optimizer_state_from_checkpoint")
            return True
    opt.load_state_dict(opt_state)
    logger.info("[batcheval] restored optimizer_state_dict from checkpoint")
    return True


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
    if (
        "optimizer_state_dict" in payload
        and hasattr(model, "optimizer")
        and getattr(model, "optimizer", None) is not None
    ):
        if hasattr(model, "load_optimizer_state_from_checkpoint"):
            loaded = bool(model.load_optimizer_state_from_checkpoint(payload))
            if not loaded:
                logger.warning(
                    "optimizer restore via load_optimizer_state_from_checkpoint returned false; "
                    "fallback to raw optimizer_state_dict."
                )
                model.optimizer.load_state_dict(payload["optimizer_state_dict"])
        else:
            model.optimizer.load_state_dict(payload["optimizer_state_dict"])
    restored_step = _checkpoint_step(payload)
    opt = getattr(model, "optimizer", None)
    if opt is not None and hasattr(opt, "global_step"):
        restored_step = int(getattr(opt, "global_step"))
    setattr(model, "global_step", int(restored_step))


_SEGMENT_FINETUNE_MAIN_PREFIXES = (
    "image_feature_extractor.residual",
    "image_feature_extractor.residual_unet",
    "image_feature_extractor.fusion",
    "image_feature_extractor.fusion_neck",
    "struct_decoder",
    "stage5_2_history_proj",
    "stage5_2_gate_branch_embed",
    "stage5_2_gate_mlp",
    "current_obs_",
)

_SEGMENT_FINETUNE_MAIN_TOKENS = (
    "offset_gru",
    "gru_update",
    "gru_candidate",
    "gru_reset",
    "gru_to_head",
    "mlp_offset",
    "mlp_conv",
    "mlp_opacity",
    "gaussion_decoder",
    "gaussian_decoder",
)


def _is_stage5_6_error_predictor_param(name: str) -> bool:
    return (
        name.startswith("stage5_6_error_head")
        or name.startswith("err_splat_proj_bg")
        or name.startswith("err_splat_proj_distant")
        or name.startswith("err_splat_proj_rigid")
    )


def _is_stage5_6_feedback_fuser_param(name: str) -> bool:
    return (
        name.startswith("stage5_6_bg_fuser")
        or name.startswith("stage5_6_distant_fuser")
        or name.startswith("stage5_6_rigid_fuser")
    )


def _is_sky_param(name: str) -> bool:
    return (
        name.startswith("sky_branch")
        or name.startswith("sky_model")
        or "_sky" in name
        or name.startswith("sky_")
    )


def _is_segment_finetune_main_param(name: str) -> bool:
    if any(name.startswith(prefix) for prefix in _SEGMENT_FINETUNE_MAIN_PREFIXES):
        return True
    return any(token in name for token in _SEGMENT_FINETUNE_MAIN_TOKENS)


def _configure_segment_finetune_optimizer(
    model: Any,
    *,
    finetune_cfg: Dict[str, Any],
    start_step: int,
) -> None:
    configure_segment_finetune_optimizer(
        model,
        finetune_cfg=dict(finetune_cfg),
        start_step=int(start_step),
        log_prefix="batcheval",
    )


def _load_sky_branch_from_cfg(cfg: Any, device: torch.device) -> Optional[Any]:
    sky_eval = cfg.batch_eval.get("sky_eval")
    if sky_eval is None or not bool(sky_eval.get("enable", False)):
        return None
    config_file = str(sky_eval.get("config_file") or "")
    checkpoint = str(sky_eval.get("checkpoint") or sky_eval.get("path") or "")
    if not config_file:
        raise ValueError("batch_eval.sky_eval.config_file is required when sky_eval.enable=true")
    if not checkpoint:
        raise ValueError("batch_eval.sky_eval.checkpoint is required when sky_eval.enable=true")

    from models.streetforward.sky_branch import SkyBranchV0

    sky_cfg = OmegaConf.load(config_file)
    sky_branch = SkyBranchV0(sky_cfg, device=device).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"SkyBranch checkpoint must be dict-like, got {type(payload).__name__}")
    state = payload.get("sky_branch_state_dict") or payload.get("model_state_dict") or payload.get("state_dict") or payload
    if not isinstance(state, dict):
        raise TypeError(f"SkyBranch checkpoint state must be dict-like, got {type(state).__name__}")
    incompatible = sky_branch.load_state_dict(state, strict=bool(sky_eval.get("strict", True)))
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if bool(sky_eval.get("freeze_params", True)):
        for p in sky_branch.parameters():
            p.requires_grad_(False)
    sky_branch.eval()
    sky_branch.reset_runtime_state()
    logger.info(
        "[batcheval] loaded SkyBranch checkpoint=%s strict=%s missing=%d unexpected=%d",
        checkpoint,
        bool(sky_eval.get("strict", True)),
        len(missing),
        len(unexpected),
    )
    return sky_branch


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


def _build_scheduler_v7_episode_specs(
    *,
    cfg: Any,
    dataset: Any,
    protocol: Any,
    max_total_episodes: int | None,
) -> List[TestEpisodeSpec]:
    sv7 = cfg.get("scheduler_v7")
    if sv7 is None or not bool(sv7.get("enable", False)):
        raise ValueError("batch_eval.runtime.align_with_scheduler_v7=true requires scheduler_v7.enable=true")
    ep = sv7.get("episode")
    if ep is None:
        raise ValueError("scheduler_v7.episode is required")
    blocks_per_episode = int(ep.get("blocks_per_episode"))
    total_target_frames = int(ep.get("total_target_frames"))
    expected_sequence_length = int(blocks_per_episode + total_target_frames - 1)
    protocol_input_offsets = [int(x) for x in protocol.input_offsets]
    eval_offsets = resolve_eval_offsets(protocol.eval_offsets, sequence_length=int(protocol.sequence_length))
    required_sequence_length = max(
        int(expected_sequence_length),
        int(protocol.sequence_length),
        int(max(eval_offsets) + 1) if len(eval_offsets) > 0 else 0,
        int(max(protocol_input_offsets) + total_target_frames) if len(protocol_input_offsets) > 0 else 0,
    )
    if int(protocol.sequence_length) != int(expected_sequence_length):
        logger.warning(
            "scheduler_v7-aligned batch_eval using non-strict sequence_length: "
            "scheduler=%d protocol=%d required_window=%d",
            int(expected_sequence_length),
            int(protocol.sequence_length),
            int(required_sequence_length),
        )
    expected_input_offsets = list(range(int(blocks_per_episode)))
    if protocol_input_offsets != expected_input_offsets:
        logger.warning(
            "scheduler_v7-aligned batch_eval using non-strict input_offsets: "
            "scheduler=%s protocol=%s required_window=%d",
            expected_input_offsets,
            protocol_input_offsets,
            int(required_sequence_length),
        )

    val_specs = build_validation_episode_specs_v7(
        dataset=dataset,
        eval_scene_ids=_collect_episode_scene_ids(cfg),
        blocks_per_episode=int(blocks_per_episode),
        total_target_frames=int(total_target_frames),
        min_window_keyframes=int(required_sequence_length),
    )
    out: List[TestEpisodeSpec] = []
    for idx, vs in enumerate(val_specs):
        frame_ids = [int(x) for x in vs.frame_chain]
        for cam_id in protocol.camera_ids:
            if int(cam_id) < 0 or int(cam_id) >= int(vs.num_cams):
                raise ValueError(
                    f"camera id out of range for scheduler_v7 episode scene={int(vs.scene_id)} "
                    f"segment={int(vs.segment_id)}: cam_id={int(cam_id)} num_cams={int(vs.num_cams)}"
                )
        input_frame_ids = [int(frame_ids[int(o)]) for o in protocol.input_offsets]
        eval_frame_ids = [int(frame_ids[int(o)]) for o in eval_offsets]
        input_image_refs = [
            (int(f), int(c)) for f in input_frame_ids for c in [int(x) for x in protocol.camera_ids]
        ]
        eval_image_refs = [
            (int(f), int(c)) for f in eval_frame_ids for c in [int(x) for x in protocol.camera_ids]
        ]
        out.append(
            TestEpisodeSpec(
                exp_name=str(protocol.name),
                scene_id=int(vs.scene_id),
                segment_id=int(vs.segment_id),
                episode_idx=int(idx),
                sequence_start_pos=int(vs.episode_start_keyframe_pos),
                frame_offsets=list(range(len(frame_ids))),
                frame_ids=[int(x) for x in frame_ids],
                input_offsets=[int(x) for x in protocol.input_offsets],
                eval_offsets=[int(x) for x in eval_offsets],
                input_frame_ids=[int(x) for x in input_frame_ids],
                eval_frame_ids=[int(x) for x in eval_frame_ids],
                camera_ids=[int(x) for x in protocol.camera_ids],
                camera_names=[str(x) for x in protocol.camera_names],
                input_image_refs=[(int(f), int(c)) for f, c in input_image_refs],
                eval_image_refs=[(int(f), int(c)) for f, c in eval_image_refs],
                episode_uid=(
                    f"scene{int(vs.scene_id):03d}_seg{int(vs.segment_id):03d}_"
                    f"kfstart{int(vs.episode_start_keyframe_pos):06d}"
                ),
            )
        )
        if max_total_episodes is not None and len(out) >= int(max_total_episodes):
            return out[: int(max_total_episodes)]
    return out


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
    sky_branch: Optional[Any],
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
    runtime_early = _as_mapping(_to_plain(cfg.batch_eval.get("runtime", {})) or {}, "batch_eval.runtime")
    if bool(runtime_early.get("align_with_scheduler_v7", False)):
        episode_specs = _build_scheduler_v7_episode_specs(
            cfg=cfg,
            dataset=dataset,
            protocol=protocol,
            max_total_episodes=max_total_episodes,
        )
    else:
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
            start_at=(_to_plain(ds_cfg.get("start_at")) if ds_cfg.get("start_at") is not None else None),
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
    stage5_6_eval = _as_mapping(_to_plain(cfg.batch_eval.get("stage5_6_eval", {})) or {}, "batch_eval.stage5_6_eval")
    sky_eval = _as_mapping(_to_plain(cfg.batch_eval.get("sky_eval", {})) or {}, "batch_eval.sky_eval")
    align_with_scheduler_v7 = bool(runtime.get("align_with_scheduler_v7", False))
    align_with_scheduler_v8 = bool(runtime.get("align_with_scheduler_v8", False))
    if align_with_scheduler_v7 and align_with_scheduler_v8:
        raise ValueError("batch_eval.runtime cannot enable both align_with_scheduler_v7 and align_with_scheduler_v8")
    runtime_mode = runtime.get("mode")
    if runtime_mode is None:
        runtime_mode = "inference_only" if bool(runtime.get("no_grad", True)) else "segment_finetune_train"
    runtime_mode = str(runtime_mode)
    default_block_order = str(runtime.get("block_order", "block_major"))
    default_step_major_switch = int(runtime.get("step_major_switch_interval_steps", 1))
    default_reset_policy = str(runtime.get("reset_policy", "episode_end"))
    default_target_policy = str(runtime.get("target_frame_policy", "all_observed"))
    default_max_targets = runtime.get("max_target_frames_including_source")
    if align_with_scheduler_v7:
        sv7 = cfg.get("scheduler_v7")
        if sv7 is None:
            raise ValueError("batch_eval.runtime.align_with_scheduler_v7=true requires config.scheduler_v7.")
        sv7_block = sv7.get("block") if hasattr(sv7, "get") else None
        sv7_episode = sv7.get("episode") if hasattr(sv7, "get") else None
        sv7_exec = sv7.get("execution") if hasattr(sv7, "get") else None
        if sv7_block is None or sv7_episode is None:
            raise ValueError("scheduler_v7-aligned batch_eval requires scheduler_v7.block and scheduler_v7.episode")
        scheduler_steps_per_block = int(sv7_block.get("steps_per_block"))
        if int(protocol.steps_per_input) != int(scheduler_steps_per_block):
            logger.warning(
                "scheduler_v7-aligned batch_eval using non-strict steps_per_input: "
                "scheduler=%d protocol=%d",
                int(scheduler_steps_per_block),
                int(protocol.steps_per_input),
            )
        if sv7_exec is not None:
            default_block_order = str(sv7_exec.get("block_order", default_block_order))
            default_step_major_switch = int(
                sv7_exec.get("step_major_switch_interval_steps", default_step_major_switch)
            )
            default_reset_policy = str(sv7_exec.get("reset_policy", default_reset_policy))
        else:
            default_block_order = "block_major"
            default_step_major_switch = 1
            sv3_mns = None
            sv3 = cfg.get("scheduler_v3") if hasattr(cfg, "get") else None
            if sv3 is not None and hasattr(sv3, "get"):
                sv3_mns = sv3.get("model_node_state")
            if sv3_mns is not None and bool(sv3_mns.get("sync_with_scheduler", False)):
                default_reset_policy = str(
                    sv3_mns.get(
                        "reset_policy",
                        "episode_end" if default_block_order == "step_major" else "block_end",
                    )
                )
            else:
                default_reset_policy = "episode_end" if default_block_order == "step_major" else "block_end"
        default_target_policy = "scheduler_v7_block_window"
        default_max_targets = int(sv7_episode.get("total_target_frames"))
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
    if bool(runtime.get("override_scheduler_execution", False)):
        if "block_order" in runtime:
            default_block_order = str(runtime.get("block_order"))
        if "step_major_switch_interval_steps" in runtime:
            default_step_major_switch = int(runtime.get("step_major_switch_interval_steps"))
        if "reset_policy" in runtime:
            default_reset_policy = str(runtime.get("reset_policy"))
    update_cameras = cfg.batch_eval.get("update_cameras")
    if update_cameras is None:
        update_camera_ids = None
    else:
        update_cameras = _as_mapping(_to_plain(update_cameras), "batch_eval.update_cameras")
        update_camera_ids = [int(x) for x in _as_list(update_cameras.get("ids"), "batch_eval.update_cameras.ids")]
    runtime_cfg = RunnerRuntimeConfig(
        mode=str(runtime_mode),
        no_grad=(str(runtime_mode) == "inference_only"),
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
        update_camera_ids=update_camera_ids,
        stage5_6_enable_nearby_feedback=bool(stage5_6_eval.get("enable_nearby_feedback", False)),
        stage5_6_nearby_policy=str(stage5_6_eval.get("nearby_policy", "adjacent_non_input")),
        stage5_6_nearby_role_name=str(stage5_6_eval.get("nearby_role_name", "near_random")),
        stage5_6_allow_partial_nearby=bool(stage5_6_eval.get("allow_partial_nearby", True)),
        sky_compose_for_metrics=bool(sky_eval.get("compose_for_metrics", sky_branch is not None)),
        sky_reset_state_per_episode=bool(sky_eval.get("reset_runtime_state_per_episode", True)),
    )
    runner = StreetForwardBatchEvalRunner(
        model=model,
        dataset=dataset,
        protocol=protocol,
        writer=writer,
        metric_acc=metric_acc,
        device=device,
        runtime_cfg=runtime_cfg,
        sky_branch=sky_branch,
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
    from tools.train_minimal_streetforward_stage4_3_v8_common import build_multi_scene_dataset_v4_for_demo

    dataset = build_multi_scene_dataset_v4_for_demo(cfg, device)
    dataset.initialize()

    model = _build_model(cfg, device)
    if hasattr(model, "bind_eval_dataset"):
        model.bind_eval_dataset(dataset)
    else:
        setattr(model, "_bound_dataset", dataset)

    ckpt_cfg_path = cfg.batch_eval.checkpoint.get("path")
    ckpt_path = str(args.checkpoint or ckpt_cfg_path or "")
    if not ckpt_path:
        raise ValueError("checkpoint path must be provided by --checkpoint or batch_eval.checkpoint.path")
    ckpt_step = _load_checkpoint(
        model,
        ckpt_path=ckpt_path,
        strict=bool(cfg.batch_eval.checkpoint.get("strict", True)),
    )
    runtime_cfg_any = cfg.batch_eval.get("runtime", {})
    runtime_cfg = _as_mapping(_to_plain(runtime_cfg_any), "batch_eval.runtime")
    runtime_mode = runtime_cfg.get("mode")
    if runtime_mode is None:
        runtime_mode = "inference_only" if bool(runtime_cfg.get("no_grad", True)) else "segment_finetune_train"
    runtime_mode = str(runtime_mode)
    if runtime_mode == "segment_finetune_train":
        finetune_cfg = _as_mapping(
            _to_plain(cfg.batch_eval.get("finetune", {})) or {},
            "batch_eval.finetune",
        )
        stage = str(cfg.model.stage).strip().lower()
        use_restricted_finetune = bool(finetune_cfg.get("enable", stage == "5_6"))
        if use_restricted_finetune:
            _configure_segment_finetune_optimizer(
                model,
                finetune_cfg=finetune_cfg,
                start_step=int(ckpt_step),
            )
        else:
            if bool(finetune_cfg.get("load_optimizer_state", True)):
                _load_checkpoint_optimizer_state(model, ckpt_path=ckpt_path, device=device)
            setattr(model, "global_step", int(ckpt_step))
            opt = getattr(model, "optimizer", None)
            if opt is not None and hasattr(opt, "global_step"):
                opt.global_step = int(ckpt_step)
            logger.info(
                "[batcheval] segment_finetune_train uses model's training optimizer unchanged "
                "(stage=%s checkpoint_step=%d).",
                stage,
                int(ckpt_step),
            )
    sky_branch = _load_sky_branch_from_cfg(cfg, device)
    restore_checkpoint_on_segment = bool(
        runtime_cfg.get(
            "restore_checkpoint_on_segment",
            runtime_cfg.get("reset_ckpt_per_segment", str(runtime_mode) == "segment_finetune_train"),
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
            sky_branch=sky_branch,
            exp_cfg=exp_cfg,
            output_root=output_root,
            device=device,
            max_total_episodes_override=args.max_total_episodes,
            base_ckpt_bytes=base_ckpt_bytes,
            restore_checkpoint_on_segment=bool(restore_checkpoint_on_segment),
        )


if __name__ == "__main__":
    main()
