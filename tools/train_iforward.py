"""
IForward multi-scene training entry.

Uses the existing V4 dataset materializer and scheduler_iforward batch contract,
but builds an independent IForward trainer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _install_headless_dash_comm_stub() -> None:
    """Make open3d's optional dash import safe in non-notebook CLI runs."""
    try:
        import comm  # type: ignore
    except Exception:
        return

    def _raise_import_error(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("dash comm disabled for headless training")

    try:
        comm.create_comm = _raise_import_error  # type: ignore[attr-defined]
    except Exception:
        return


_install_headless_dash_comm_stub()

import torch
from omegaconf import OmegaConf

import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as base
from datasets.iforward_coverage_validation import (
    iforward_coverage_validation_cfg,
    write_iforward_coverage_validation_rows,
)
from datasets.iforward_sequence10_validation import (
    SEQUENCE10_VALIDATION_PROTOCOLS,
    build_sequence10_manifest,
    write_sequence10_manifest,
)
from datasets.iforward_stage2_2.validation_runner import (
    run_stage2_2_validation,
    run_stage2_2_validation_manifest_only,
    stage2_2_validation_cfg,
)
from datasets.iforward_stage2_3.validation_runner import (
    run_stage2_3_validation,
    run_stage2_3_validation_manifest_only,
    stage2_3_validation_cfg,
)
from datasets.iforward_stage2_3.scheduler import Stage23Scheduler
from datasets.train_scheduler_iforward import TrainSchedulerIForward
from datasets.train_scheduler_iforward_sequence10 import TrainSchedulerIForwardSequence10
from models.iforward import IForwardTrainer
from models.iforward.protocols.validation_recipes import build_validation_v4_plans, iforward_validation_v4_cfg
from models.iforward.resolver import IFORWARD_STAGE3_2_SCHEDULER_VERSION
from models.iforward.runtime.adapter_stage3 import Stage3SchedulerAdapter
from models.iforward.runtime.runner import IForwardRunner, RunnerOptions
from models.iforward.runtime.trace import TraceRecorder
from models.iforward.validation_v4.html_exporter import export_html_report
from models.iforward.versions import (
    is_stage3_optimizer_memory_iforward_version,
    uncertainty_schema_versions,
)
from tools.train_minimal_streetforward_stage4_3_iforward_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_from_cfg,
    resolve_fixed_scene_segment_iforward,
)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def _git_text(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *list(args)],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    if int(proc.returncode) != 0:
        return ""
    return str(proc.stdout)


def _git_manifest_fields() -> Dict[str, Any]:
    status = _git_text(["status", "--short"])
    diff = _git_text(["diff", "--no-ext-diff", "HEAD", "--", "."])
    return {
        "git_sha": _git_text(["rev-parse", "HEAD"]).strip(),
        "git_dirty": bool(status.strip()),
        "source_diff_sha256": _sha256_text(status + "\n" + diff),
    }


def _active_scheduler_manifest_fields(cfg: Any) -> Dict[str, Any]:
    inherited_scheduler_cfg: Any = {}
    sched32 = _cfg_get(cfg, "scheduler_stage3_2", None)
    if sched32 is not None and bool(_cfg_get(sched32, "enable", False)):
        scheduler_key = "scheduler_stage3_2"
        scheduler_cfg = sched32
        if str(_cfg_get(sched32, "inherit_from", "scheduler_stage3_0") or "") == "scheduler_stage3_0":
            inherited_scheduler_cfg = _cfg_get(cfg, "scheduler_stage3_0", {}) or {}
    else:
        sched30 = _cfg_get(cfg, "scheduler_stage3_0", None)
        if sched30 is not None and bool(_cfg_get(sched30, "enable", False)):
            scheduler_key = "scheduler_stage3_0"
            scheduler_cfg = sched30
        else:
            scheduler_key = "scheduler_v3"
            scheduler_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    return {
        "scheduler_key": str(scheduler_key),
        "scheduler_version": str(_cfg_get(scheduler_cfg, "version", "")),
        "index_dir": str(_cfg_get(scheduler_cfg, "index_dir", _cfg_get(inherited_scheduler_cfg, "index_dir", "")) or ""),
        "index_fingerprint": str(
            _cfg_get(scheduler_cfg, "index_fingerprint", _cfg_get(inherited_scheduler_cfg, "index_fingerprint", "")) or ""
        ),
    }


def _active_validation_manifest_fields(cfg: Any) -> Dict[str, Any]:
    candidates = [
        ("scheduler_stage3_0_validation", _cfg_get(cfg, "scheduler_stage3_0_validation", None)),
        ("iforward_validation_v4", _cfg_get(cfg, "iforward_validation_v4", None)),
        ("validation_v3", _cfg_get(cfg, "validation_v3", None)),
        ("iforward_stage2_2_validation", _cfg_get(cfg, "iforward_stage2_2_validation", None)),
        ("iforward_sequence10_validation", _cfg_get(cfg, "iforward_sequence10_validation", None)),
        ("iforward_coverage_validation", _cfg_get(cfg, "iforward_coverage_validation", None)),
        ("iforward_validation", _cfg_get(cfg, "iforward_validation", None)),
    ]
    validation_key = ""
    validation_cfg: Any = {}
    fallback_key = ""
    fallback_cfg: Any = {}
    for key, raw in candidates:
        if raw is not None:
            if not fallback_key:
                fallback_key = str(key)
                fallback_cfg = raw
            if bool(_cfg_get(raw, "enable", False)):
                validation_key = str(key)
                validation_cfg = raw
                break
    if not validation_key and fallback_key:
        validation_key = fallback_key
        validation_cfg = fallback_cfg
    return {
        "validation_key": validation_key,
        "validation_enable": bool(_cfg_get(validation_cfg, "enable", False)),
        "validation_run_at_train_start": bool(_cfg_get(validation_cfg, "run_at_train_start", True))
        if validation_key
        else False,
    }


def _iforward_lifting_manifest_fields(cfg: Any) -> Dict[str, Any]:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    lifting_cfg = _cfg_get(iforward_cfg, "lifting", {}) or {}
    parent_cfg = _cfg_get(lifting_cfg, "parent", {}) or {}
    child_cfg = _cfg_get(lifting_cfg, "child_gather", {}) or {}
    return {
        "iforward_version": str(_cfg_get(iforward_cfg, "version", "")),
        "lifting_type": str(_cfg_get(lifting_cfg, "type", "") or ""),
        "parent_lifting_type": str(_cfg_get(parent_cfg, "type", "") or ""),
        "child_gather_type": str(_cfg_get(child_cfg, "type", "") or ""),
    }


def _config_path_from_hook_kwargs(kwargs: Dict[str, Any]) -> str:
    args = kwargs.get("args", None)
    config_file = str(getattr(args, "config_file", "") or "")
    if config_file:
        return config_file
    return _config_file_from_argv(list(sys.argv), "configs/iforward/iforward_base.yaml")


def _build_iforward_run_manifest(**kwargs: Any) -> Tuple[Dict[str, Any], str]:
    cfg = kwargs["cfg"]
    config_path = _config_path_from_hook_kwargs(kwargs)
    snapshot_yaml = OmegaConf.to_yaml(cfg, resolve=False)
    manifest: Dict[str, Any] = {
        "schema_version": "iforward_run_manifest_v1",
        "config_path": str(config_path),
        "config_sha256": _sha256_file(str(config_path)),
        "config_snapshot_sha256": _sha256_text(snapshot_yaml),
        "output_name": str(_cfg_get(cfg, "output_name", "") or ""),
        "log_dir": str(_cfg_get(cfg, "log_dir", "") or ""),
        "resume_checkpoint": str(kwargs.get("resume_checkpoint", "") or ""),
        "init_checkpoint": str(kwargs.get("init_checkpoint", "") or ""),
        "checkpoint_prefix": str(kwargs.get("checkpoint_prefix", checkpoint_prefix_iforward_from_cfg(cfg)) or ""),
    }
    manifest.update(_git_manifest_fields())
    manifest.update(_active_scheduler_manifest_fields(cfg))
    manifest.update(_active_validation_manifest_fields(cfg))
    manifest.update(_iforward_lifting_manifest_fields(cfg))
    iforward_cfg = _cfg_get(_cfg_get(cfg, "model", {}) or {}, "iforward", {}) or {}
    manifest.update({"local_gs_state_schema_version": 2})
    manifest.update(uncertainty_schema_versions(_cfg_get(iforward_cfg, "version", "")))
    return manifest, snapshot_yaml


def _iforward_run_start_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    log_dir = str(_cfg_get(cfg, "log_dir", "") or "")
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    manifest, snapshot_yaml = _build_iforward_run_manifest(**kwargs)
    snapshot_path = os.path.join(log_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        fh.write(snapshot_yaml)
    manifest["config_snapshot_path"] = snapshot_path
    manifest_path = os.path.join(log_dir, "run_manifest.json")
    manifest["run_manifest_path"] = manifest_path
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    row = {"split": "run_manifest", **manifest}
    base._write_metrics_history(kwargs.get("metrics_fh", None), row)


def build_iforward_trainer_from_cfg(config: Any, device: torch.device) -> IForwardTrainer:
    return IForwardTrainer(config=config, device=device)


def checkpoint_prefix_iforward_from_cfg(cfg: Any) -> str:
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    version = str(_cfg_get(iforward_cfg, "version", "v1"))
    return f"iforward_{version}"


def _config_file_from_argv(argv: List[str], default_config: str) -> str:
    for idx, arg in enumerate(argv):
        if arg == "--config_file" and idx + 1 < len(argv):
            return str(argv[idx + 1])
        if arg.startswith("--config_file="):
            return str(arg.split("=", 1)[1])
    return str(default_config)


def _route_random_window_entrypoint_if_needed(default_config: str) -> bool:
    config_file = _config_file_from_argv(list(sys.argv), default_config)
    cfg = OmegaConf.load(config_file)
    random_cfg = _cfg_get(cfg, "scheduler_iforward_random_window", None)
    legacy_cfg = _cfg_get(cfg, "scheduler_iforward", None)
    stage22_cfg = _cfg_get(cfg, "scheduler_stage2_2", None)
    stage32_cfg = _cfg_get(cfg, "scheduler_stage3_2", None)
    stage30_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    stage23_cfg = (
        stage32_cfg
        if stage32_cfg is not None and bool(_cfg_get(stage32_cfg, "enable", False))
        else (
            stage30_cfg
            if stage30_cfg is not None and bool(_cfg_get(stage30_cfg, "enable", False))
            else _cfg_get(cfg, "scheduler_v3", None)
        )
    )
    random_enabled = random_cfg is not None and bool(_cfg_get(random_cfg, "enable", True))
    stage22_enabled = stage22_cfg is not None and bool(_cfg_get(stage22_cfg, "enable", False))
    stage23_enabled = (
        stage23_cfg is not None
        and bool(_cfg_get(stage23_cfg, "enable", False))
        and str(_cfg_get(stage23_cfg, "version", ""))
        in {"optimizer_sequence_v1", "stage3_0_optimizer_sequence_v1", IFORWARD_STAGE3_2_SCHEDULER_VERSION, "distributional_episode_v1"}
    )
    if random_enabled and legacy_cfg is None:
        from tools import train_iforward_random_window

        train_iforward_random_window.main()
        return True
    if legacy_cfg is None and not stage22_enabled and not stage23_enabled:
        raise ValueError(
            "tools/train_iforward.py requires scheduler_iforward, scheduler_stage2_2, "
            "scheduler_v3 optimizer_sequence_v1, or scheduler_stage3_0 stage3_0_optimizer_sequence_v1. "
            "For scheduler_iforward_random_window configs, use tools/train_iforward_random_window.py."
        )
    return False


def _iforward_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_validation", {}) or {}
    tb_images_raw = _cfg_get(raw, "tensorboard_images", {}) or {}
    modes_raw = _cfg_get(raw, "modes", None)
    modes = None
    if modes_raw is not None:
        if isinstance(modes_raw, str):
            modes = [modes_raw]
        else:
            modes = [str(x) for x in modes_raw]
    rollout_shapes = [dict(x) for x in list(_cfg_get(raw, "rollout_shapes", []) or [])]
    fixed_shape_names = [str(x) for x in list(_cfg_get(raw, "fixed_shape_names", []) or [])]
    shape_eval_mode = str(_cfg_get(raw, "shape_eval_mode", "sample")).lower()
    if shape_eval_mode == "independent_all" and rollout_shapes:
        rollouts_per_segment = max(int(_cfg_get(raw, "rollouts_per_segment", 1)), len(rollout_shapes))
        if not fixed_shape_names:
            fixed_shape_names = [str(shape.get("name", "")) for shape in rollout_shapes if str(shape.get("name", ""))]
    else:
        rollouts_per_segment = int(_cfg_get(raw, "rollouts_per_segment", 1))
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 1000)),
        "segments_per_scene": int(_cfg_get(raw, "segments_per_scene", 1)),
        "rollouts_per_segment": int(rollouts_per_segment),
        "blocks_per_episode": _cfg_get(raw, "blocks_per_episode", None),
        "modes": modes,
        "use_train_rollout_shapes": bool(_cfg_get(raw, "use_train_rollout_shapes", False)),
        "rollout_shapes": rollout_shapes,
        "fixed_shape_names": fixed_shape_names,
        "shape_eval_mode": shape_eval_mode,
        "tensorboard_images_enable": bool(_cfg_get(tb_images_raw, "enable", True)),
        "tensorboard_images_max_per_role": int(_cfg_get(tb_images_raw, "max_images_per_role", 2)),
    }


def _iforward_sequence10_validation_cfg(cfg: Any) -> Dict[str, Any]:
    raw = _cfg_get(cfg, "iforward_sequence10_validation", {}) or {}
    tb_images_raw = _cfg_get(raw, "tensorboard_images", {}) or {}
    protocols_raw = _cfg_get(raw, "protocols", SEQUENCE10_VALIDATION_PROTOCOLS)
    if protocols_raw is None:
        protocols = list(SEQUENCE10_VALIDATION_PROTOCOLS)
    elif isinstance(protocols_raw, str):
        protocols = [str(protocols_raw)]
    else:
        protocols = [str(x) for x in list(protocols_raw)]
    unknown = [p for p in protocols if p not in set(SEQUENCE10_VALIDATION_PROTOCOLS)]
    if unknown:
        raise ValueError(f"Unknown iforward_sequence10_validation.protocols entries: {unknown}")
    modes_raw = _cfg_get(raw, "modes", ["full"])
    if modes_raw is None:
        modes = ["full"]
    elif isinstance(modes_raw, str):
        modes = [str(modes_raw)]
    else:
        modes = [str(x) for x in list(modes_raw)]
    strides_raw = _cfg_get(raw, "strides", [1, 2])
    if isinstance(strides_raw, (int, float)):
        strides = [int(strides_raw)]
    else:
        strides = [int(x) for x in list(strides_raw or [1, 2])]
    return {
        "enable": bool(_cfg_get(raw, "enable", False)),
        "run_at_train_start": bool(_cfg_get(raw, "run_at_train_start", True)),
        "interval_steps": int(_cfg_get(raw, "interval_steps", 5000)),
        "segments_per_scene": int(_cfg_get(raw, "segments_per_scene", 1)),
        "max_segments_total": int(_cfg_get(raw, "max_segments_total", 2)),
        "max_entries": int(_cfg_get(raw, "max_entries", 8)),
        "seed": int(_cfg_get(raw, "seed", 20260623)),
        "protocols": protocols,
        "modes": modes,
        "strides": strides,
        "manifest_path": str(_cfg_get(raw, "manifest_path", "")),
        "tensorboard_images_enable": bool(_cfg_get(tb_images_raw, "enable", True)),
        "tensorboard_images_max_per_role": int(_cfg_get(tb_images_raw, "max_images_per_role", 2)),
    }


def _is_sequence10_scheduler_cfg(cfg: Any) -> bool:
    sched = _cfg_get(cfg, "scheduler_iforward", {}) or {}
    return str(_cfg_get(sched, "version", "")) == "iforward_sequence10_v1"


def _is_stage2_2_scheduler_cfg(cfg: Any) -> bool:
    sched = _cfg_get(cfg, "scheduler_stage2_2", {}) or {}
    return bool(_cfg_get(sched, "enable", False))


def _is_stage2_3_scheduler_cfg(cfg: Any) -> bool:
    sched32 = _cfg_get(cfg, "scheduler_stage3_2", None)
    sched30 = _cfg_get(cfg, "scheduler_stage3_0", None)
    sched = (
        sched32
        if sched32 is not None and bool(_cfg_get(sched32, "enable", False))
        else (
            sched30
            if sched30 is not None and bool(_cfg_get(sched30, "enable", False))
            else (_cfg_get(cfg, "scheduler_v3", {}) or {})
        )
    )
    model_cfg = _cfg_get(cfg, "model", {}) or {}
    iforward_cfg = _cfg_get(model_cfg, "iforward", {}) or {}
    return (
        bool(_cfg_get(sched, "enable", False))
        and str(_cfg_get(sched, "version", ""))
        in {"optimizer_sequence_v1", "stage3_0_optimizer_sequence_v1", IFORWARD_STAGE3_2_SCHEDULER_VERSION, "distributional_episode_v1"}
        and (
            str(_cfg_get(iforward_cfg, "version", "")) == "iforward_2_3_optimizer_mamba"
            or is_stage3_optimizer_memory_iforward_version(_cfg_get(iforward_cfg, "version", ""))
        )
    )


def _stage2_3_scheduler_key(cfg: Any) -> str:
    sched32 = _cfg_get(cfg, "scheduler_stage3_2", None)
    if sched32 is not None and bool(_cfg_get(sched32, "enable", False)):
        return "scheduler_stage3_2"
    sched30 = _cfg_get(cfg, "scheduler_stage3_0", None)
    if sched30 is not None and bool(_cfg_get(sched30, "enable", False)):
        return "scheduler_stage3_0"
    return "scheduler_v3"


def _stage2_3_validation_key(cfg: Any) -> str:
    stage3_raw = _cfg_get(cfg, "scheduler_stage3_0_validation", None)
    if stage3_raw is not None:
        return "scheduler_stage3_0_validation"
    legacy_raw = _cfg_get(cfg, "validation_v3", None)
    if legacy_raw is not None:
        return "validation_v3"
    return ""


def _stage2_3_validation_status_split(cfg: Any) -> str:
    return (
        "iforward_stage3_0_validation_status"
        if _stage2_3_scheduler_key(cfg) == "scheduler_stage3_0"
        else "iforward_stage2_3_validation_status"
    )


def _exception_tail(exc: BaseException, *, max_chars: int = 4096, max_lines: int = 24) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = "".join(lines).strip().splitlines()[-int(max_lines):]
    text = "\n".join(tail)
    if len(text) > int(max_chars):
        text = text[-int(max_chars):]
    return text


def _make_stage2_3_validation_status_writer(
    *,
    cfg: Any,
    metrics_fh: Any,
    trigger: str,
    trigger_step: int,
) -> Callable[[Dict[str, Any]], None]:
    split = _stage2_3_validation_status_split(cfg)
    validation_key = _stage2_3_validation_key(cfg)
    scheduler_fields = _active_scheduler_manifest_fields(cfg)

    def _write(row: Dict[str, Any]) -> None:
        status_row = {
            "step": int(trigger_step),
            "trigger_step": int(trigger_step),
            "split": split,
            "trigger": str(trigger),
            "validation_key": str(validation_key),
            **scheduler_fields,
        }
        status_row.update(dict(row))
        base._write_metrics_history(metrics_fh, status_row)

    return _write


def _write_stage2_3_validation_skip_status(
    *,
    cfg: Any,
    metrics_fh: Any,
    trigger: str,
    trigger_step: int,
    reason: str,
    val_cfg: Dict[str, Any],
) -> None:
    writer = _make_stage2_3_validation_status_writer(
        cfg=cfg,
        metrics_fh=metrics_fh,
        trigger=trigger,
        trigger_step=int(trigger_step),
    )
    writer(
        {
            "status": "skip",
            "reason": str(reason),
            "validation_enable": bool(val_cfg.get("enable", False)),
            "validation_run_at_train_start": bool(val_cfg.get("run_at_train_start", False)),
            "protocols": [str(x) for x in list(val_cfg.get("protocols", []) or [])],
            "modes": [str(x) for x in list(val_cfg.get("modes", []) or [])],
        }
    )


def _run_stage2_3_validation_with_status(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_step: int,
    trigger: str,
    val_cfg: Dict[str, Any],
    metrics_fh: Any,
    writer: Any,
    convert_batch_to_minimal_format: Any,
) -> List[Dict[str, Any]]:
    modes = [str(x) for x in list(val_cfg.get("modes", ["full"]) or ["full"])]
    protocols = [str(x) for x in list(val_cfg.get("protocols", []) or [])]
    status_writer = _make_stage2_3_validation_status_writer(
        cfg=cfg,
        metrics_fh=metrics_fh,
        trigger=trigger,
        trigger_step=int(trigger_step),
    )
    status_writer(
        {
            "status": "start",
            "validation_enable": bool(val_cfg.get("enable", False)),
            "validation_run_at_train_start": bool(val_cfg.get("run_at_train_start", False)),
            "protocols": protocols,
            "modes": modes,
        }
    )
    try:
        if model is None or dataset is None:
            entries = run_stage2_3_validation_manifest_only(cfg=cfg, dataset=dataset)
            status_writer(
                {
                    "status": "manifest_built",
                    "manifest_only": True,
                    "max_entries": int(val_cfg.get("max_entries", len(entries))),
                    "planned_protocol_count": int(len(protocols)),
                    "num_entries": int(len(entries)),
                }
            )
            status_writer(
                {
                    "status": "empty" if not entries else "completed",
                    "num_rows": 0,
                    "num_entries": int(len(entries)),
                    "protocols_completed": [],
                }
            )
            return list(entries)
        rows = run_stage2_3_validation(
            cfg=cfg,
            dataset=dataset,
            model=model,
            device=device,
            trigger_step=int(trigger_step),
            modes=modes,
            convert_batch_to_minimal_format=convert_batch_to_minimal_format,
            writer=writer,
            status_writer=status_writer,
        )
        completed = sorted({str(row.get("protocol", "")) for row in rows if str(row.get("protocol", ""))})
        if rows:
            status_writer(
                {
                    "status": "completed",
                    "num_rows": int(len(rows)),
                    "protocols_completed": completed,
                }
            )
        else:
            status_writer(
                {
                    "status": "empty",
                    "num_rows": 0,
                    "protocols_completed": completed,
                }
            )
        for row in rows:
            if metrics_fh is not None:
                base._write_metrics_history(metrics_fh, row)
        return rows
    except Exception as exc:
        if "produced no rows" in str(exc):
            status_writer(
                {
                    "status": "empty",
                    "num_rows": 0,
                    "protocols_completed": [],
                }
            )
        status_writer(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception_tail": _exception_tail(exc),
            }
        )
        raise


def _validation_v4_enabled(cfg: Any) -> bool:
    return bool(iforward_validation_v4_cfg(cfg).get("enable", False))


def _validation_v4_due(cfg: Any, *, trigger: str, step: int) -> bool:
    val = iforward_validation_v4_cfg(cfg)
    if not bool(val.get("enable", False)):
        return False
    if str(trigger) == "train_start":
        return bool(val.get("run_at_train_start", False))
    interval = int(val.get("interval_steps", 0) or 0)
    return bool(interval > 0 and int(step) >= 0 and (int(step) + 1) % int(interval) == 0)


def _validation_v4_status_row(cfg: Any, *, trigger: str, trigger_step: int, row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "step": int(trigger_step),
        "trigger_step": int(trigger_step),
        "split": "iforward_validation_v4_status",
        "trigger": str(trigger),
        "validation_key": "iforward_validation_v4",
        **_active_scheduler_manifest_fields(cfg),
    }
    out.update(dict(row))
    return out


def _write_validation_v4_status(cfg: Any, metrics_fh: Any, *, trigger: str, trigger_step: int, row: Dict[str, Any]) -> None:
    if metrics_fh is None:
        return
    base._write_metrics_history(
        metrics_fh,
        _validation_v4_status_row(cfg, trigger=trigger, trigger_step=int(trigger_step), row=row),
    )


def _safe_path_component(value: Any) -> str:
    text = str(value or "item").strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(out).strip("._") or "item"


def _validation_v4_log_dir(cfg: Any, *, trigger_step: int) -> str:
    log_dir = str(_cfg_get(cfg, "log_dir", "") or "")
    if not log_dir:
        logging_cfg = _cfg_get(cfg, "logging", {}) or {}
        log_dir = str(_cfg_get(logging_cfg, "log_dir", "") or "")
    return os.path.join(log_dir or ".", "iforward_validation_v4", f"step{int(trigger_step):06d}")


def _validation_v4_record_images(cfg: Any, *, plan_idx: int) -> bool:
    val = iforward_validation_v4_cfg(cfg)
    report = dict(val.get("report", {}) or {})
    if not bool(report.get("images", True)):
        return False
    policy = str(report.get("image_policy", "first_plan_only") or "first_plan_only")
    if policy == "none":
        return False
    if policy == "all":
        return True
    return int(plan_idx) == 0


def _make_validation_v4_scheduler(cfg: Any, dataset: Any) -> Stage23Scheduler:
    sched_cfg = _cfg_get(cfg, "scheduler_stage3_2", None)
    if sched_cfg is None or not bool(_cfg_get(sched_cfg, "enable", False)):
        sched_cfg = _cfg_get(cfg, "scheduler_stage3_0", None)
    if sched_cfg is None or not bool(_cfg_get(sched_cfg, "enable", False)):
        sched_cfg = _cfg_get(cfg, "scheduler_v3", {}) or {}
    producer_cfg = dict(_cfg_get(sched_cfg, "producer", {}) or {})
    producer_cfg["enable"] = False
    return Stage23Scheduler(dataset=dataset, cfg=cfg, producer_cfg=producer_cfg, fail_fast=False)


def _run_validation_v4_with_status(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: Any,
    trigger_step: int,
    trigger: str,
    metrics_fh: Any,
) -> List[Dict[str, Any]]:
    val = iforward_validation_v4_cfg(cfg)
    output_dir = _validation_v4_log_dir(cfg, trigger_step=int(trigger_step))
    _write_validation_v4_status(
        cfg,
        metrics_fh,
        trigger=trigger,
        trigger_step=int(trigger_step),
        row={
            "status": "start",
            "validation_enable": bool(val.get("enable", False)),
            "validation_run_at_train_start": bool(val.get("run_at_train_start", False)),
            "interval_steps": int(val.get("interval_steps", 0) or 0),
            "output_dir": str(output_dir),
        },
    )
    try:
        plans = build_validation_v4_plans(
            cfg=cfg,
            dataset=dataset,
            max_entries=int(val.get("max_entries_debug", 1) or 1),
            repair_permutations=int(val.get("repair_permutations", 3) or 3),
            memory_ablation=list(val.get("memory_ablation", ["full"]) or ["full"]),
        )
        _write_validation_v4_status(
            cfg,
            metrics_fh,
            trigger=trigger,
            trigger_step=int(trigger_step),
            row={
                "status": "plans_built",
                "num_plans": int(len(plans)),
                "output_dir": str(output_dir),
                "image_policy": str((dict(val.get("report", {}) or {})).get("image_policy", "first_plan_only")),
            },
        )
        prev_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        summaries: List[Dict[str, Any]] = []
        html_paths: List[str] = []
        try:
            for idx, plan in enumerate(plans):
                plan_id = str(getattr(plan, "plan_id", f"plan{idx}") or f"plan{idx}")
                plan_dir = os.path.join(str(output_dir), f"{idx:04d}_{_safe_path_component(plan_id)}")
                scheduler = _make_validation_v4_scheduler(cfg, dataset)
                adapter = Stage3SchedulerAdapter(scheduler)
                runner = IForwardRunner(model, adapter, _sequence10_minimal_from_scheduler_batch)
                record_images = bool(_validation_v4_record_images(cfg, plan_idx=int(idx)))
                recorder = TraceRecorder(plan_dir, record_images=record_images)
                trace = runner.run(
                    plan,
                    recorder,
                    RunnerOptions.for_mode("validate", device=str(device), trigger_step=int(trigger_step)),
                )
                html_path = export_html_report(trace, plan_dir, title=f"IForward Validation v4 {plan.episode.protocol_name}")
                html_paths.append(str(html_path))
                summary = dict(getattr(trace, "summary", {}) or {})
                protocol = str(getattr(getattr(plan, "episode", None), "protocol_name", plan_id))
                compact = {
                    "plan_idx": int(idx),
                    "plan_id": str(plan_id),
                    "protocol": protocol,
                    "num_events": int(len(list(getattr(trace, "events", []) or []))),
                    "record_images": bool(record_images),
                    "html_path": str(html_path),
                    "current_psnr_mean": _safe_float(summary.get("current_psnr/mean")),
                    "history_rollout_psnr_mean": _safe_float(summary.get("history_rollout_psnr/mean")),
                    "loss_mean": _safe_float(summary.get("loss/mean")),
                }
                summaries.append(compact)
                _write_validation_v4_status(
                    cfg,
                    metrics_fh,
                    trigger=trigger,
                    trigger_step=int(trigger_step),
                    row={"status": "plan_completed", **compact},
                )
        finally:
            if prev_training and hasattr(model, "train"):
                model.train()

        index_path = os.path.join(str(output_dir), "index.html")
        os.makedirs(str(output_dir), exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write("<!doctype html><meta charset='utf-8'><h1>IForward Validation v4</h1><ul>")
            for path in html_paths:
                rel = os.path.relpath(path, str(output_dir))
                fh.write(f"<li><a href='{rel}'>{rel}</a></li>")
            fh.write("</ul>\n")

        global_row = {
            "step": int(trigger_step),
            "trigger_step": int(trigger_step),
            "split": "iforward_validation_v4_global",
            "status": "completed" if summaries else "empty",
            "num_plans": int(len(plans)),
            "num_completed_plans": int(len(summaries)),
            "num_events": int(sum(int(x.get("num_events", 0) or 0) for x in summaries)),
            "output_dir": str(output_dir),
            "html_index_path": str(index_path),
            "current_psnr_mean": _mean([_safe_float(x.get("current_psnr_mean")) for x in summaries]),
            "history_rollout_psnr_mean": _mean([_safe_float(x.get("history_rollout_psnr_mean")) for x in summaries]),
            "loss_mean": _mean([_safe_float(x.get("loss_mean")) for x in summaries]),
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, global_row)
        _write_validation_v4_status(
            cfg,
            metrics_fh,
            trigger=trigger,
            trigger_step=int(trigger_step),
            row={
                "status": "completed" if summaries else "empty",
                "num_plans": int(len(plans)),
                "num_completed_plans": int(len(summaries)),
                "output_dir": str(output_dir),
                "html_index_path": str(index_path),
            },
        )
        return summaries
    except Exception as exc:
        _write_validation_v4_status(
            cfg,
            metrics_fh,
            trigger=trigger,
            trigger_step=int(trigger_step),
            row={
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception_tail": _exception_tail(exc),
                "output_dir": str(output_dir),
            },
        )
        raise


def _max_inner_k_from_shapes(shapes: Sequence[Dict[str, Any]]) -> int:
    max_k = 0
    for shape in list(shapes or []):
        blocks = int(_cfg_get(shape, "blocks_per_rollout", 1) or 1)
        repeats = int(_cfg_get(shape, "repeats_per_block", 1) or 1)
        max_k = max(int(max_k), int(blocks) * int(repeats))
    return int(max_k)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: List[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_image_role(role: Any) -> str:
    text = str(role or "view").strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out).strip("_") or "view"


def _tb_hwc01(img: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(torch.nan_to_num(img.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if x.dim() != 3:
        raise ValueError(f"expected image tensor [H,W,C], got {tuple(x.shape)}")
    if int(x.shape[-1]) == 1:
        x = x.expand(*x.shape[:-1], 3)
    if int(x.shape[-1]) != 3:
        raise ValueError(f"expected image tensor channel dim=3, got {tuple(x.shape)}")
    return x


def _write_iforward_validation_tb_images(
    *,
    writer: Any,
    out: Any,
    step: int,
    scene_id: int,
    segment_id: int,
    rollout_idx: int,
    max_images_per_role: int,
    tag_root: str = "iforward_validation/images",
) -> None:
    if writer is None or int(max_images_per_role) <= 0:
        return
    pred_rgbs = list(getattr(out, "pred_rgbs", []) or [])
    gt_images = list(getattr(out, "gt_images", []) or [])
    image_refs = list(getattr(out, "image_refs", []) or [])
    image_roles = list(getattr(out, "image_roles", []) or [])
    n = min(len(pred_rgbs), len(gt_images))
    role_counts: Dict[str, int] = {}
    for idx in range(int(n)):
        role = _safe_image_role(image_roles[idx] if idx < len(image_roles) else "view")
        count = int(role_counts.get(role, 0))
        if count >= int(max_images_per_role):
            continue
        role_counts[role] = count + 1
        ref = image_refs[idx] if idx < len(image_refs) else (-1, -1)
        try:
            frame_idx = int(ref[0])  # type: ignore[index]
            cam_idx = int(ref[1])  # type: ignore[index]
        except Exception:
            frame_idx = -1
            cam_idx = -1
        try:
            pred = _tb_hwc01(pred_rgbs[idx])
            gt = _tb_hwc01(gt_images[idx])
            err = (pred - gt).abs()
            max_err = float(err.max().item()) if err.numel() else 0.0
            if max_err > 0.0:
                err = err / max_err
            tag = (
                f"{str(tag_root).strip('/')}/"
                f"scene_{int(scene_id):03d}_segment_{int(segment_id):03d}/"
                f"{role}/rollout_{int(rollout_idx)}/view_{count}_f{frame_idx:05d}_c{cam_idx}"
            )
            writer.add_image(f"{tag}/pred", pred.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag}/gt", gt.permute(2, 0, 1), int(step))
            writer.add_image(f"{tag}/error", err.permute(2, 0, 1), int(step))
        except Exception as exc:
            base.logger.warning(
                "Failed to write IForward validation TensorBoard image scene=%s segment=%s role=%s idx=%s: %s",
                int(scene_id),
                int(segment_id),
                role,
                int(idx),
                exc,
            )


def _first_valid_iforward_eval_segments(cfg: Any, dataset: Any) -> List[Tuple[int, int]]:
    val_cfg = _iforward_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return []
    sched = _cfg_get(cfg, "scheduler_iforward", {}) or {}
    scheduler_version = str(_cfg_get(sched, "version", "iforward_v1"))
    min_keyframes = 8 if scheduler_version == "iforward_v3_random_window" else 4
    eval_scene_ids = [int(x) for x in list(_cfg_get(_cfg_get(cfg, "data", {}) or {}, "eval_scene_ids", []) or [])]
    out: List[Tuple[int, int]] = []
    for scene_id in eval_scene_ids:
        found = 0
        for segment_id in sorted(int(x) for x in list(dataset.list_segment_ids(int(scene_id)) or [])):
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
            if len(keyframes) < int(min_keyframes):
                continue
            out.append((int(scene_id), int(segment_id)))
            found += 1
            if found >= int(val_cfg["segments_per_scene"]):
                break
    return out


def _make_validation_scheduler(cfg: Any, dataset: Any, scene_id: int, segment_id: int) -> TrainSchedulerIForward:
    sched = _cfg_get(cfg, "scheduler_iforward", {}) or {}
    val_cfg = _iforward_validation_cfg(cfg)
    scheduler_version = str(_cfg_get(sched, "version", "iforward_v1"))
    episode_cfg = copy.deepcopy(dict(_cfg_get(sched, "episode", {}) or {}))
    rollout_cfg = copy.deepcopy(dict(_cfg_get(sched, "rollout", {}) or {}))
    if scheduler_version == "iforward_v3_random_window":
        episode_cfg.update(
            {
                "blocks_per_episode": 8,
                "episode_stride": 8,
                "allow_short_last_episode": False,
                "min_blocks_per_episode": 8,
                "rollouts_per_episode": max(3, int(_cfg_get(episode_cfg, "rollouts_per_episode", 3))),
            }
        )
        rollout_cfg.update(
            {
                "allow_short_final_rollout": False,
                "min_blocks_per_rollout": 1,
                "avoid_single_block_tail": False,
                "fixed_shape_names": ["r8b1", "r4b2", "r2b4"],
                "fixed_window_starts": [0, 2, 4],
                "shapes": [
                    {"name": "r8b1", "blocks_per_rollout": 1, "repeats_per_block": 8, "prob": 1.0},
                    {"name": "r4b2", "blocks_per_rollout": 2, "repeats_per_block": 4, "prob": 1.0},
                    {"name": "r2b4", "blocks_per_rollout": 4, "repeats_per_block": 2, "prob": 1.0},
                ],
                "shapes_schedule": [],
            }
        )
    else:
        configured_shapes = [dict(x) for x in list(val_cfg.get("rollout_shapes", []) or [])]
        if not configured_shapes and bool(val_cfg.get("use_train_rollout_shapes", False)):
            configured_shapes = [dict(x) for x in list(_cfg_get(rollout_cfg, "shapes", []) or [])]
        if configured_shapes:
            blocks = [int(_cfg_get(shape, "blocks_per_rollout", 1)) for shape in configured_shapes]
            max_blocks = max(blocks) if blocks else 1
            min_blocks = min(blocks) if blocks else 1
            val_rollouts_per_segment = max(1, int(val_cfg.get("rollouts_per_segment", 1)))
            configured_blocks_per_episode = val_cfg.get("blocks_per_episode", None)
            if configured_blocks_per_episode is None:
                blocks_per_episode = max(int(max_blocks), int(val_rollouts_per_segment))
            else:
                blocks_per_episode = max(int(configured_blocks_per_episode), int(max_blocks))
            episode_cfg.update(
                {
                    "blocks_per_episode": int(blocks_per_episode),
                    "episode_stride": int(blocks_per_episode),
                    "allow_short_last_episode": True,
                    "min_blocks_per_episode": int(min_blocks),
                    "rollouts_per_episode": int(val_rollouts_per_segment),
                }
            )
            rollout_update = {
                "allow_short_final_rollout": False,
                "min_blocks_per_rollout": int(min_blocks),
                "avoid_single_block_tail": False,
                "max_inner_K": max(
                    int(_cfg_get(rollout_cfg, "max_inner_K", 0) or 0),
                    _max_inner_k_from_shapes(configured_shapes),
                ),
                "shapes": configured_shapes,
                "shapes_schedule": [],
                "fixed_shape_names": [],
            }
            if val_cfg.get("fixed_shape_names"):
                rollout_update["fixed_shape_names"] = [str(x) for x in list(val_cfg.get("fixed_shape_names", []) or [])]
            rollout_cfg.update(rollout_update)
        else:
            episode_cfg.update(
                {
                    "blocks_per_episode": 4,
                    "episode_stride": 4,
                    "allow_short_last_episode": False,
                    "min_blocks_per_episode": 4,
                    "rollouts_per_episode": int(_cfg_get(episode_cfg, "rollouts_per_episode", 1)),
                }
            )
            rollout_cfg.update(
                {
                    "allow_short_final_rollout": False,
                    "min_blocks_per_rollout": 4,
                    "avoid_single_block_tail": True,
                    "shapes": [
                        {
                            "name": "b4_r2",
                            "blocks_per_rollout": 4,
                            "repeats_per_block": 2,
                            "prob": 1.0,
                        }
                    ],
                    "shapes_schedule": [],
                }
            )
    traversal_cfg = copy.deepcopy(dict(_cfg_get(sched, "traversal", {}) or {}))
    traversal_cfg.update(
        {
            "fixed_scene_id": int(scene_id),
            "fixed_segment_id": int(segment_id),
            "scene_order": "ascending",
            "segment_order": "ascending",
            "traversal_mode": "episode_serial",
            "seed": 0,
        }
    )
    preload_cfg = copy.deepcopy(dict(_cfg_get(sched, "preload", {}) or {}))
    preload_cfg["emit_hints"] = False
    return TrainSchedulerIForward(
        dataset=dataset,
        episode_cfg=episode_cfg,
        rollout_cfg=rollout_cfg,
        traversal_cfg=traversal_cfg,
        evidence_cfg=copy.deepcopy(dict(_cfg_get(sched, "evidence", {}) or {})),
        supervision_cfg=copy.deepcopy(dict(_cfg_get(sched, "supervision", {}) or {})),
        memory_cfg=copy.deepcopy(dict(_cfg_get(sched, "memory", {}) or {})),
        loss_timing_cfg=copy.deepcopy(dict(_cfg_get(sched, "loss_timing", {}) or {})),
        leakage_check_cfg=copy.deepcopy(dict(_cfg_get(sched, "leakage_check", {}) or {})),
        preload_cfg=preload_cfg,
        include_test=False,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=0,
        version=str(scheduler_version),
        fail_fast=True,
    )


def _sequence10_protocol_stride(protocol: str) -> Optional[int]:
    if "D2" in str(protocol):
        return 2
    if "D1" in str(protocol):
        return 1
    return None


def _sequence10_protocol_rollout_count(protocol: str) -> int:
    if str(protocol) in {"S10-D1-Repair", "S10-D2-Repair"}:
        return 6
    if str(protocol) in {"S10-D1-Causal", "S10-D2-Causal"}:
        return 5
    return 1


def _sequence10_eval_pairs(cfg: Any, dataset: Any, val_cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    data_cfg = _cfg_get(cfg, "data", {}) or {}
    eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "eval_scene_ids", []) or [])]
    if not eval_scene_ids:
        eval_scene_ids = [int(x) for x in list(_cfg_get(data_cfg, "train_scene_ids", []) or [])]
    strides = [int(x) for x in list(val_cfg.get("strides", [1, 2]) or [1, 2])]
    max_segments_total = max(1, int(val_cfg.get("max_segments_total", 2)))
    segments_per_scene = max(1, int(val_cfg.get("segments_per_scene", 1)))
    out: List[Tuple[int, int]] = []
    for scene_id in eval_scene_ids:
        scene_count = 0
        try:
            segment_ids = sorted(int(x) for x in list(dataset.list_segment_ids(int(scene_id)) or []))
        except Exception as exc:
            base.logger.warning("Sequence10 validation could not list segments for scene=%s: %s", int(scene_id), exc)
            continue
        for segment_id in segment_ids:
            try:
                manifest = build_sequence10_manifest(
                    dataset=dataset,
                    scene_segment_pairs=[(int(scene_id), int(segment_id))],
                    strides=strides,
                    max_entries=1,
                )
            except Exception as exc:
                base.logger.warning(
                    "Sequence10 validation skipped scene=%s segment=%s while checking eligibility: %s",
                    int(scene_id),
                    int(segment_id),
                    exc,
                )
                continue
            if not list(manifest.get("entries", []) or []):
                continue
            out.append((int(scene_id), int(segment_id)))
            scene_count += 1
            if len(out) >= int(max_segments_total) or scene_count >= int(segments_per_scene):
                break
        if len(out) >= int(max_segments_total):
            break
    return out


def _sequence10_validation_entries_for_protocol(
    *,
    dataset: Any,
    pairs: Sequence[Tuple[int, int]],
    protocol: str,
    val_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    stride = _sequence10_protocol_stride(protocol)
    strides = [int(stride)] if stride is not None else [int(x) for x in list(val_cfg.get("strides", [1, 2]) or [1, 2])]
    manifest = build_sequence10_manifest(
        dataset=dataset,
        scene_segment_pairs=[(int(s), int(g)) for s, g in pairs],
        strides=strides,
        max_entries=max(1, int(val_cfg.get("max_entries", 8))),
    )
    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()
    for raw in list(manifest.get("entries", []) or []):
        entry = dict(raw)
        key = (int(entry.get("scene_id", -1)), int(entry.get("segment_id", -1)), int(entry.get("stride", 1)))
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
        if len(entries) >= max(1, int(val_cfg.get("max_entries", 8))):
            break
    return entries


def _make_sequence10_validation_scheduler(
    *,
    cfg: Any,
    dataset: Any,
    scene_id: int,
    segment_id: int,
    stride: int,
    protocol: str,
    seed: int,
) -> TrainSchedulerIForwardSequence10:
    sched = _cfg_get(cfg, "scheduler_iforward", {}) or {}
    traversal_cfg = copy.deepcopy(dict(_cfg_get(sched, "traversal", {}) or {}))
    traversal_cfg.update(
        {
            "fixed_scene_id": int(scene_id),
            "fixed_segment_id": int(segment_id),
            "scene_order": "ordered",
            "segment_order": "ordered",
            "traversal_mode": "scene_round_robin_episode",
            "forbid_consecutive_same_scene": True,
            "seed": int(seed),
        }
    )
    sequence_cfg = copy.deepcopy(dict(_cfg_get(sched, "sequence", {}) or {}))
    sequence_cfg.update(
        {
            "length": 10,
            "block_source": "keyframes",
            "strides": [int(stride)],
            "max_inner_K": 10,
        }
    )
    bootstrap_cfg = copy.deepcopy(dict(_cfg_get(sched, "bootstrap", {}) or {}))
    causal_cfg = copy.deepcopy(dict(_cfg_get(sched, "causal", {}) or {}))
    repair_cfg = copy.deepcopy(dict(_cfg_get(sched, "repair", {}) or {}))
    if str(protocol) in {"SingleFrame-K8", "Repeat Stability"}:
        bootstrap_cfg.update(
            {
                "end_step": 1_000_000,
                "repeat_choices": [{"repeats": 8, "prob": 1.0}],
                "current_only": True,
            }
        )
        causal_cfg.update(
            {
                "start_step": 1_000_000,
                "rollouts_per_episode": 5,
                "blocks_per_rollout": 2,
                "repeats_per_block": 4,
                "temporal_read": True,
                "temporal_commit": True,
                "physical_time_advance": True,
            }
        )
        repair_cfg.update(
            {
                "start_step": 1_000_000,
                "prob": 0.0,
                "blocks_per_rollout": 10,
                "repeats_per_block": 1,
                "non_identity_permutation": True,
                "temporal_read": True,
                "temporal_commit": False,
                "observation_commit": False,
                "update_optimizer_memory": False,
                "physical_time_advance": False,
            }
        )
    else:
        bootstrap_cfg.update({"end_step": 0})
        causal_cfg.update(
            {
                "start_step": 0,
                "rollouts_per_episode": 5,
                "blocks_per_rollout": 2,
                "repeats_per_block": 4,
                "temporal_read": True,
                "temporal_commit": True,
                "physical_time_advance": True,
            }
        )
        repair_cfg.update(
            {
                "start_step": 0 if "Repair" in str(protocol) else 1_000_000,
                "prob": 1.0 if "Repair" in str(protocol) else 0.0,
                "blocks_per_rollout": 10,
                "repeats_per_block": 1,
                "non_identity_permutation": True,
                "temporal_read": True,
                "temporal_commit": False,
                "observation_commit": False,
                "update_optimizer_memory": False,
                "physical_time_advance": False,
            }
        )
    supervision_cfg = copy.deepcopy(dict(_cfg_get(sched, "supervision", {}) or {}))
    history_cfg = copy.deepcopy(dict(_cfg_get(supervision_cfg, "history_replay", {}) or {}))
    history_cfg.update({"enable": True, "start_step": 0, "max_frames_per_rollout": 10})
    supervision_cfg["history_replay"] = history_cfg
    preload_cfg = copy.deepcopy(dict(_cfg_get(sched, "preload", {}) or {}))
    preload_cfg["emit_hints"] = False
    return TrainSchedulerIForwardSequence10(
        dataset=dataset,
        traversal_cfg=traversal_cfg,
        bootstrap_cfg=bootstrap_cfg,
        sequence_cfg=sequence_cfg,
        causal_cfg=causal_cfg,
        repair_cfg=repair_cfg,
        supervision_cfg=supervision_cfg,
        history_loss_cfg={},
        damage_loss_cfg={},
        preload_cfg=preload_cfg,
        include_test=False,
        fixed_scene_id=int(scene_id),
        fixed_segment_id=int(segment_id),
        seed=int(seed),
        fail_fast=True,
    )


def _tensor_to_float(value: Any, default: float = float("nan")) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        value = value.detach().float().mean().item()
    return _safe_float(value, default)


def _sequence10_row_from_output(
    *,
    out: Any,
    ifwd_meta: Dict[str, Any],
    protocol: str,
    mode: str,
    trigger_step: int,
    trigger_train_episode_counter: int,
    rollout_idx: int,
    repeat_idx: int = 0,
) -> Dict[str, Any]:
    stats = dict(getattr(out, "stats", {}) or {})
    losses = dict(getattr(out, "losses", {}) or {})
    resolved = getattr(out, "resolved", None)
    return {
        "step": int(trigger_step),
        "split": "iforward_sequence10_validation",
        "protocol": str(protocol),
        "mode": str(mode),
        "trigger_step": int(trigger_step),
        "trigger_train_episode_counter": int(trigger_train_episode_counter),
        "scene_id": int(ifwd_meta.get("scene_id", -1)),
        "segment_id": int(ifwd_meta.get("segment_id", -1)),
        "rollout_idx": int(rollout_idx),
        "repeat_idx": int(repeat_idx),
        "scheduler_phase": str(ifwd_meta.get("scheduler_phase", "")),
        "rollout_phase": str(ifwd_meta.get("rollout_phase", "")),
        "rollout_shape": str(ifwd_meta.get("shape_name", "unknown")),
        "sequence_id": int(ifwd_meta.get("sequence_id", -1) or -1),
        "sequence_stride": int(ifwd_meta.get("sequence_stride", 0) or 0),
        "sequence_positions": [int(x) for x in list(ifwd_meta.get("sequence_positions", []) or [])],
        "sequence_keyframe_indices": [
            int(x) for x in list(ifwd_meta.get("sequence_keyframe_indices", []) or [])
        ],
        "sequence_source_frame_indices": [
            int(x) for x in list(ifwd_meta.get("sequence_source_frame_indices", []) or [])
        ],
        "history_positions": [int(x) for x in list(ifwd_meta.get("history_positions", []) or [])],
        "repair_positions": [int(x) for x in list(ifwd_meta.get("repair_positions", []) or [])],
        "repair_flag": bool(str(ifwd_meta.get("scheduler_phase", "")) == "repair"),
        "repair_permutation_hash": int(ifwd_meta.get("repair_permutation_hash", -1) or -1),
        "temporal_read_count": int(ifwd_meta.get("temporal_read_count", 0) or 0),
        "temporal_commit_count": int(ifwd_meta.get("temporal_commit_count", 0) or 0),
        "observation_commit_count": int(ifwd_meta.get("observation_commit_count", 0) or 0),
        "optimizer_memory_update_count": int(ifwd_meta.get("optimizer_memory_update_count", 0) or 0),
        "history_frame_count": int(ifwd_meta.get("history_frame_count", 0) or 0),
        "history_ref_count": int(ifwd_meta.get("history_ref_count", 0) or 0),
        "inference_only": True,
        "loss": _tensor_to_float(getattr(out, "loss", None)),
        "current_loss": _tensor_to_float(losses.get("current")),
        "history_rollout_loss": _tensor_to_float(losses.get("in_rollout_history")),
        "history_damage_loss": _tensor_to_float(losses.get("history_damage")),
        "current_psnr": _safe_float(stats.get("current_psnr", stats.get("current_latest_psnr"))),
        "history_rollout_psnr": _safe_float(stats.get("history_psnr", stats.get("history_rollout_psnr"))),
        "current_valid_ratio": _safe_float(stats.get("current_valid_ratio", stats.get("current_latest_valid_ratio"))),
        "history_rollout_valid_ratio": _safe_float(
            stats.get("history_valid_ratio", stats.get("in_rollout_history_valid_ratio"))
        ),
        "current_num_refs": _safe_float(stats.get("current_num_refs", stats.get("current_latest_num_refs"))),
        "history_rollout_num_refs": _safe_float(stats.get("history_num_refs", stats.get("history_rollout_num_refs"))),
        "sequence10_best_damage_loss": _safe_float(stats.get("sequence10/best_damage_loss", 0.0), 0.0),
        "sequence10_best_damage_num_pos": _safe_float(stats.get("sequence10/best_damage_num_pos", 0.0), 0.0),
        "sequence10_bank_valid_count": _safe_float(stats.get("sequence10/bank_valid_count", 0.0), 0.0),
        "sequence10_bank_update_count": _safe_float(stats.get("sequence10/bank_update_count", 0.0), 0.0),
        "loss_weight_history": _safe_float(stats.get("loss_weight/in_rollout_history", 0.0), 0.0),
        "loss_weight_history_damage": _safe_float(stats.get("loss_weight/history_damage", 0.0), 0.0),
        "carry_scene_state_after_rollout": bool(getattr(resolved, "carry_scene_state_after_rollout", False)),
        "episode_end_after_rollout": bool(getattr(resolved, "episode_end_after_rollout", False)),
    }


def _reset_iforward_eval_runtime(model: Any) -> None:
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    if callable(reset_bridge):
        reset_bridge()
    if hasattr(model, "reset_iforward_state_cache"):
        model.reset_iforward_state_cache()


def _sequence10_minimal_from_scheduler_batch(raw_batch: Dict[str, Any], device: torch.device, trigger_step: int) -> Dict[str, Any]:
    target = raw_batch.get("target") or {}
    image = target.get("image") if isinstance(target, dict) else None
    num_targets = int(image.shape[0]) if torch.is_tensor(image) else 0
    minimal_batch = base.convert_batch_to_minimal_format(
        raw_batch,
        device,
        num_targets=num_targets,
        include_source_for_2d=True,
        view_selection=None,
    )
    minimal_batch["global_step"] = int(trigger_step)
    return minimal_batch


def _write_iforward_sequence10_validation_rows(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int,
    trigger_train_episode_counter: int,
    metrics_fh: Any,
    writer: Any,
    **_: Any,
) -> None:
    val_cfg = _iforward_sequence10_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return
    pairs = _sequence10_eval_pairs(cfg, dataset, val_cfg)
    if not pairs:
        row = {
            "step": int(trigger_step),
            "split": "iforward_sequence10_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_rollouts": 0,
            "status": "no_valid_sequence10_segments",
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, row)
        return
    manifest = build_sequence10_manifest(
        dataset=dataset,
        scene_segment_pairs=[(int(s), int(g)) for s, g in pairs],
        strides=[int(x) for x in list(val_cfg.get("strides", [1, 2]) or [1, 2])],
        max_entries=max(1, int(val_cfg.get("max_entries", 8))),
    )
    manifest_path = str(val_cfg.get("manifest_path", "") or "")
    if manifest_path:
        write_sequence10_manifest(manifest_path, manifest)

    was_training = bool(model.training)
    saved_cache = dict(getattr(model, "_state_cache", {}) or {})
    rows: List[Dict[str, Any]] = []
    model.eval()
    _reset_iforward_eval_runtime(model)
    try:
        with torch.no_grad():
            for protocol in list(val_cfg["protocols"]):
                entries = _sequence10_validation_entries_for_protocol(
                    dataset=dataset,
                    pairs=pairs,
                    protocol=str(protocol),
                    val_cfg=val_cfg,
                )
                if not entries:
                    row = {
                        "step": int(trigger_step),
                        "split": "iforward_sequence10_validation_global",
                        "protocol": str(protocol),
                        "trigger_step": int(trigger_step),
                        "trigger_train_episode_counter": int(trigger_train_episode_counter),
                        "num_rollouts": 0,
                        "status": "no_valid_protocol_entries",
                    }
                    if metrics_fh is not None:
                        base._write_metrics_history(metrics_fh, row)
                    continue
                for entry_idx, entry in enumerate(entries):
                    scene_id = int(entry["scene_id"])
                    segment_id = int(entry["segment_id"])
                    stride = int(entry["stride"])
                    seed = int(val_cfg["seed"]) + 1009 * int(entry_idx) + 97 * int(stride)
                    for mode in list(val_cfg["modes"]):
                        scheduler = _make_sequence10_validation_scheduler(
                            cfg=cfg,
                            dataset=dataset,
                            scene_id=scene_id,
                            segment_id=segment_id,
                            stride=stride,
                            protocol=str(protocol),
                            seed=int(seed),
                        )
                        if str(protocol) == "Repeat Stability":
                            raw_batch = scheduler.next_batch()
                            ifwd_meta = dict(raw_batch.get("_iforward", {}) or {})
                            minimal_batch = _sequence10_minimal_from_scheduler_batch(raw_batch, device, int(trigger_step))
                            repeat_rows: List[Dict[str, Any]] = []
                            for repeat_idx in range(2):
                                _reset_iforward_eval_runtime(model)
                                out = model.forward_rollout(minimal_batch, carried_state=None, ablation=str(mode))
                                row = _sequence10_row_from_output(
                                    out=out,
                                    ifwd_meta=ifwd_meta,
                                    protocol=str(protocol),
                                    mode=str(mode),
                                    trigger_step=int(trigger_step),
                                    trigger_train_episode_counter=int(trigger_train_episode_counter),
                                    rollout_idx=0,
                                    repeat_idx=int(repeat_idx),
                                )
                                rows.append(row)
                                repeat_rows.append(row)
                                if metrics_fh is not None:
                                    base._write_metrics_history(metrics_fh, row)
                            if len(repeat_rows) == 2:
                                diff_row = {
                                    "step": int(trigger_step),
                                    "split": "iforward_sequence10_validation_global",
                                    "protocol": str(protocol),
                                    "mode": str(mode),
                                    "scene_id": int(scene_id),
                                    "segment_id": int(segment_id),
                                    "sequence_stride": int(stride),
                                    "trigger_step": int(trigger_step),
                                    "trigger_train_episode_counter": int(trigger_train_episode_counter),
                                    "num_rollouts": 2,
                                    "repeat_stability_loss_abs_diff": abs(
                                        float(repeat_rows[0]["loss"]) - float(repeat_rows[1]["loss"])
                                    ),
                                    "repeat_stability_current_psnr_abs_diff": abs(
                                        float(repeat_rows[0]["current_psnr"]) - float(repeat_rows[1]["current_psnr"])
                                    ),
                                }
                                if metrics_fh is not None:
                                    base._write_metrics_history(metrics_fh, diff_row)
                            continue

                        carried_state = None
                        _reset_iforward_eval_runtime(model)
                        rollout_count = _sequence10_protocol_rollout_count(str(protocol))
                        for rollout_idx in range(int(rollout_count)):
                            raw_batch = scheduler.next_batch()
                            ifwd_meta = dict(raw_batch.get("_iforward", {}) or {})
                            minimal_batch = _sequence10_minimal_from_scheduler_batch(raw_batch, device, int(trigger_step))
                            out = model.forward_rollout(minimal_batch, carried_state=carried_state, ablation=str(mode))
                            row = _sequence10_row_from_output(
                                out=out,
                                ifwd_meta=ifwd_meta,
                                protocol=str(protocol),
                                mode=str(mode),
                                trigger_step=int(trigger_step),
                                trigger_train_episode_counter=int(trigger_train_episode_counter),
                                rollout_idx=int(rollout_idx),
                            )
                            rows.append(row)
                            if metrics_fh is not None:
                                base._write_metrics_history(metrics_fh, row)
                            if writer is not None:
                                tag = (
                                    f"iforward_sequence10_validation/{str(protocol)}/{str(mode)}/"
                                    f"scene_{scene_id:03d}_segment_{segment_id:03d}"
                                )
                                writer.add_scalar(f"{tag}/current_psnr", float(row["current_psnr"]), int(trigger_step))
                                writer.add_scalar(f"{tag}/history_rollout_psnr", float(row["history_rollout_psnr"]), int(trigger_step))
                                writer.add_scalar(f"{tag}/sequence10_best_damage_loss", float(row["sequence10_best_damage_loss"]), int(trigger_step))
                                if bool(val_cfg["tensorboard_images_enable"]) and (
                                    int(rollout_idx) == int(rollout_count) - 1 or bool(row["repair_flag"])
                                ):
                                    _write_iforward_validation_tb_images(
                                        writer=writer,
                                        out=out,
                                        step=int(trigger_step),
                                        scene_id=int(scene_id),
                                        segment_id=int(segment_id),
                                        rollout_idx=int(rollout_idx),
                                        max_images_per_role=int(val_cfg["tensorboard_images_max_per_role"]),
                                        tag_root=f"iforward_sequence10_validation/images/{str(protocol)}/{str(mode)}",
                                    )
                            resolved = getattr(out, "resolved", None)
                            carry_after = bool(getattr(resolved, "carry_scene_state_after_rollout", False))
                            episode_end = bool(getattr(resolved, "episode_end_after_rollout", False))
                            if bool(carry_after) and not bool(episode_end):
                                next_state = getattr(out, "next_state", None)
                                detach = getattr(next_state, "detach_for_next_rollout", None)
                                carried_state = detach() if callable(detach) else next_state
                            else:
                                carried_state = None
                                _reset_iforward_eval_runtime(model)
    finally:
        if hasattr(model, "_state_cache"):
            model._state_cache = saved_cache
        _reset_iforward_eval_runtime(model)
        model.train(was_training)

    if rows:
        global_rows: List[Dict[str, Any]] = []
        for protocol in list(val_cfg["protocols"]):
            for mode in list(val_cfg["modes"]):
                mode_rows = [r for r in rows if str(r.get("protocol")) == str(protocol) and str(r.get("mode")) == str(mode)]
                if not mode_rows:
                    continue
                global_rows.append(
                    {
                        "step": int(trigger_step),
                        "split": "iforward_sequence10_validation_global",
                        "protocol": str(protocol),
                        "mode": str(mode),
                        "trigger_step": int(trigger_step),
                        "trigger_train_episode_counter": int(trigger_train_episode_counter),
                        "num_rollouts": int(len(mode_rows)),
                        "loss": _mean([float(r["loss"]) for r in mode_rows]),
                        "current_psnr": _mean([float(r["current_psnr"]) for r in mode_rows]),
                        "history_rollout_psnr": _mean([float(r["history_rollout_psnr"]) for r in mode_rows]),
                        "current_valid_ratio": _mean([float(r["current_valid_ratio"]) for r in mode_rows]),
                        "history_rollout_valid_ratio": _mean([float(r["history_rollout_valid_ratio"]) for r in mode_rows]),
                        "sequence10_best_damage_loss": _mean(
                            [float(r["sequence10_best_damage_loss"]) for r in mode_rows]
                        ),
                        "sequence10_bank_valid_count": _mean(
                            [float(r["sequence10_bank_valid_count"]) for r in mode_rows]
                        ),
                    }
                )
        for global_row in global_rows:
            if metrics_fh is not None:
                base._write_metrics_history(metrics_fh, global_row)
            if writer is not None:
                protocol = str(global_row.get("protocol", "all"))
                mode = str(global_row.get("mode", "full"))
                tag = f"iforward_sequence10_validation/global/{protocol}/{mode}"
                writer.add_scalar(f"{tag}/current_psnr", float(global_row["current_psnr"]), int(trigger_step))
                writer.add_scalar(f"{tag}/history_rollout_psnr", float(global_row["history_rollout_psnr"]), int(trigger_step))
                writer.add_scalar(f"{tag}/sequence10_best_damage_loss", float(global_row["sequence10_best_damage_loss"]), int(trigger_step))
        if writer is not None:
            flush = getattr(writer, "flush", None)
            if callable(flush):
                flush()


def _write_iforward_validation_rows(
    *,
    cfg: Any,
    dataset: Any,
    model: Any,
    device: torch.device,
    trigger_step: int,
    trigger_train_episode_counter: int,
    metrics_fh: Any,
    writer: Any,
    **_: Any,
) -> None:
    val_cfg = _iforward_validation_cfg(cfg)
    if not bool(val_cfg["enable"]):
        return
    segments = _first_valid_iforward_eval_segments(cfg, dataset)
    if not segments:
        row = {
            "step": int(trigger_step),
            "split": "iforward_validation_global",
            "trigger_step": int(trigger_step),
            "trigger_train_episode_counter": int(trigger_train_episode_counter),
            "num_rollouts": 0,
            "status": "no_valid_eval_segments",
        }
        if metrics_fh is not None:
            base._write_metrics_history(metrics_fh, row)
        return

    was_training = bool(model.training)
    saved_cache = dict(getattr(model, "_state_cache", {}) or {})
    rows: List[Dict[str, Any]] = []
    if hasattr(model, "reset_iforward_state_cache"):
        model.reset_iforward_state_cache()
    reset_bridge = getattr(model, "_reset_bridge_runtime_node_state", None)
    if callable(reset_bridge):
        reset_bridge()
    model.eval()
    configured_modes = val_cfg.get("modes")
    if configured_modes:
        validation_modes = tuple(str(x) for x in configured_modes)
    else:
        inner_model = getattr(model, "model", model)
        allowed_ablations = set(str(x) for x in (getattr(inner_model, "allowed_ablations", None) or ()))
        validation_modes = (
            ("full_adc", "no_adc")
            if not allowed_ablations or {"full_adc", "no_adc"}.issubset(allowed_ablations)
            else ("full",)
        )
    try:
        with torch.no_grad():
            for scene_id, segment_id in segments:
                scheduler = _make_validation_scheduler(cfg, dataset, int(scene_id), int(segment_id))
                fixed_rollouts: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
                for rollout_idx in range(int(val_cfg["rollouts_per_segment"])):
                    raw_batch = scheduler.next_batch()
                    ifwd_meta = dict(raw_batch.get("_iforward", {}) or {})
                    target = raw_batch.get("target") or {}
                    num_targets = int(target["image"].shape[0])
                    minimal_batch = base.convert_batch_to_minimal_format(
                        raw_batch,
                        device,
                        num_targets=num_targets,
                        include_source_for_2d=True,
                        view_selection=None,
                    )
                    minimal_batch["global_step"] = int(trigger_step)
                    fixed_rollouts.append((int(rollout_idx), ifwd_meta, minimal_batch))

                segment_rows: List[Dict[str, Any]] = []
                segment_outputs: List[Tuple[Dict[str, Any], Any]] = []
                for mode in validation_modes:
                    carried_state = None
                    if callable(reset_bridge):
                        reset_bridge()
                    if hasattr(model, "reset_iforward_state_cache"):
                        model.reset_iforward_state_cache()
                    for rollout_idx, ifwd_meta, minimal_batch in fixed_rollouts:
                        out = model.forward_rollout(minimal_batch, carried_state=carried_state, ablation=mode)
                        stats = dict(out.stats or {})
                        losses = {name: _safe_float(value.detach().item()) for name, value in out.losses.items()}
                        resolved = getattr(out, "resolved", None)
                        carry_after = bool(getattr(resolved, "carry_scene_state_after_rollout", False))
                        episode_end = bool(getattr(resolved, "episode_end_after_rollout", False))
                        row = {
                            "step": int(trigger_step),
                            "split": "iforward_validation",
                            "mode": str(mode),
                            "trigger_step": int(trigger_step),
                            "trigger_train_episode_counter": int(trigger_train_episode_counter),
                            "scene_id": int(scene_id),
                            "segment_id": int(segment_id),
                            "rollout_idx": int(rollout_idx),
                            "rollout_shape": str(
                                ifwd_meta.get("shape_name", ifwd_meta.get("requested_shape_name", "unknown"))
                            ),
                            "inference_only": True,
                            "loss": _safe_float(out.loss.detach().item()),
                            "current_loss": losses.get("current", losses.get("current_latest", float("nan"))),
                            "history_rollout_loss": losses.get("in_rollout_history", float("nan")),
                            "nearby_loss": losses.get("nearby", float("nan")),
                            "current_psnr": _safe_float(stats.get("current_psnr", stats.get("current_latest_psnr"))),
                            "history_rollout_psnr": _safe_float(stats.get("history_rollout_psnr")),
                            "nearby_psnr": _safe_float(stats.get("nearby_psnr")),
                            "current_valid_ratio": _safe_float(
                                stats.get("current_valid_ratio", stats.get("current_latest_valid_ratio"))
                            ),
                            "history_rollout_valid_ratio": _safe_float(stats.get("in_rollout_history_valid_ratio")),
                            "nearby_valid_ratio": _safe_float(stats.get("nearby_valid_ratio")),
                            "current_num_refs": _safe_float(
                                stats.get("current_num_refs", stats.get("current_latest_num_refs"))
                            ),
                            "history_rollout_num_refs": _safe_float(stats.get("history_rollout_num_refs")),
                            "nearby_num_refs": _safe_float(stats.get("nearby_num_refs")),
                            "adc_applied": _safe_float(stats.get("adc_lite/applied", 0.0), 0.0),
                            "adc_num_cloned_this_rollout": _safe_float(
                                stats.get("adc_lite/num_cloned_this_rollout", 0.0), 0.0
                            ),
                            "adc_num_cloned_episode": _safe_float(stats.get("adc_lite/num_cloned_episode", 0.0), 0.0),
                            "adc_bg_count_before": _safe_float(stats.get("adc_lite/bg_count_before", 0.0), 0.0),
                            "adc_bg_count_after": _safe_float(stats.get("adc_lite/bg_count_after", 0.0), 0.0),
                            "adc_parent_score_mean": _safe_float(stats.get("adc_lite/parent_score_mean", 0.0), 0.0),
                            "adc_parent_gate_mean": _safe_float(stats.get("adc_suppressed/parent_gate_mean", 0.0), 0.0),
                            "adc_parent_delta_demand_mean": _safe_float(
                                stats.get("adc_suppressed/parent_delta_demand_mean", 0.0), 0.0
                            ),
                            "adc_suppression_score_topk_mean": _safe_float(
                                stats.get("adc_suppressed/score_topk_mean", 0.0), 0.0
                            ),
                            "carry_scene_state_after_rollout": bool(carry_after),
                            "episode_end_after_rollout": bool(episode_end),
                        }
                        segment_rows.append(row)
                        segment_outputs.append((row, out))
                        if bool(carry_after) and not bool(episode_end):
                            next_state = getattr(out, "next_state", None)
                            detach = getattr(next_state, "detach_for_next_rollout", None)
                            carried_state = detach() if callable(detach) else next_state
                        else:
                            carried_state = None
                            if callable(reset_bridge):
                                reset_bridge()
                            if hasattr(model, "reset_iforward_state_cache"):
                                model.reset_iforward_state_cache()

                by_mode_rollout = {(str(row["mode"]), int(row["rollout_idx"])): row for row in segment_rows}
                for rollout_idx, _, _ in fixed_rollouts:
                    full_row = by_mode_rollout.get(("full_adc", int(rollout_idx)))
                    no_row = by_mode_rollout.get(("no_adc", int(rollout_idx)))
                    if full_row is None or no_row is None:
                        continue
                    deltas = {
                        "delta_full_minus_noadc_current_psnr": full_row["current_psnr"] - no_row["current_psnr"],
                        "delta_full_minus_noadc_history_rollout_psnr": (
                            full_row["history_rollout_psnr"] - no_row["history_rollout_psnr"]
                        ),
                        "delta_full_minus_noadc_nearby_psnr": full_row["nearby_psnr"] - no_row["nearby_psnr"],
                    }
                    full_row.update(deltas)
                    no_row.update(deltas)

                for row, out in segment_outputs:
                    row.setdefault("delta_full_minus_noadc_current_psnr", float("nan"))
                    row.setdefault("delta_full_minus_noadc_history_rollout_psnr", float("nan"))
                    row.setdefault("delta_full_minus_noadc_nearby_psnr", float("nan"))
                    rows.append(row)
                    if metrics_fh is not None:
                        base._write_metrics_history(metrics_fh, row)
                    if writer is not None:
                        tag = (
                            f"iforward_validation/{str(row['mode'])}/"
                            f"scene_{int(scene_id):03d}_segment_{int(segment_id):03d}"
                        )
                        writer.add_scalar(f"{tag}/current_psnr", float(row["current_psnr"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/history_rollout_psnr", float(row["history_rollout_psnr"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/nearby_psnr", float(row["nearby_psnr"]), int(trigger_step))
                        writer.add_scalar(f"{tag}/adc_applied", float(row["adc_applied"]), int(trigger_step))
                        if bool(val_cfg["tensorboard_images_enable"]):
                            _write_iforward_validation_tb_images(
                                writer=writer,
                                out=out,
                                step=int(trigger_step),
                                scene_id=int(scene_id),
                                segment_id=int(segment_id),
                                rollout_idx=int(row["rollout_idx"]),
                                max_images_per_role=int(val_cfg["tensorboard_images_max_per_role"]),
                            )
    finally:
        if hasattr(model, "_state_cache"):
            model._state_cache = saved_cache
        if callable(reset_bridge):
            reset_bridge()
        model.train(was_training)

    if rows:
        global_rows: List[Dict[str, Any]] = []
        for mode in validation_modes:
            mode_rows = [r for r in rows if str(r.get("mode")) == mode]
            if not mode_rows:
                continue
            global_rows.append(
                {
                    "step": int(trigger_step),
                    "split": "iforward_validation_global",
                    "mode": mode,
                    "trigger_step": int(trigger_step),
                    "trigger_train_episode_counter": int(trigger_train_episode_counter),
                    "num_rollouts": int(len(mode_rows)),
                    "current_psnr": _mean([float(r["current_psnr"]) for r in mode_rows]),
                    "history_rollout_psnr": _mean([float(r["history_rollout_psnr"]) for r in mode_rows]),
                    "nearby_psnr": _mean([float(r["nearby_psnr"]) for r in mode_rows]),
                    "current_valid_ratio": _mean([float(r["current_valid_ratio"]) for r in mode_rows]),
                    "history_rollout_valid_ratio": _mean([float(r["history_rollout_valid_ratio"]) for r in mode_rows]),
                    "nearby_valid_ratio": _mean([float(r["nearby_valid_ratio"]) for r in mode_rows]),
                    "adc_applied_ratio": _mean([float(r["adc_applied"]) for r in mode_rows]),
                    "adc_num_cloned_mean": _mean([float(r["adc_num_cloned_this_rollout"]) for r in mode_rows]),
                    "adc_bg_count_after_mean": _mean([float(r["adc_bg_count_after"]) for r in mode_rows]),
                }
            )
        by_mode = {str(r["mode"]): r for r in global_rows}
        if "full_adc" in by_mode and "no_adc" in by_mode:
            full = by_mode["full_adc"]
            no_adc = by_mode["no_adc"]
            global_rows.append(
                {
                    "step": int(trigger_step),
                    "split": "iforward_validation_global",
                    "mode": "full_minus_no_adc",
                    "trigger_step": int(trigger_step),
                    "trigger_train_episode_counter": int(trigger_train_episode_counter),
                    "num_rollouts": int(min(full["num_rollouts"], no_adc["num_rollouts"])),
                    "current_psnr": float(full["current_psnr"]) - float(no_adc["current_psnr"]),
                    "history_rollout_psnr": float(full["history_rollout_psnr"]) - float(no_adc["history_rollout_psnr"]),
                    "nearby_psnr": float(full["nearby_psnr"]) - float(no_adc["nearby_psnr"]),
                    "current_valid_ratio": float("nan"),
                    "history_rollout_valid_ratio": float("nan"),
                    "nearby_valid_ratio": float("nan"),
                    "adc_applied_ratio": float(full.get("adc_applied_ratio", 0.0)),
                    "adc_num_cloned_mean": float(full.get("adc_num_cloned_mean", 0.0)),
                    "adc_bg_count_after_mean": float(full.get("adc_bg_count_after_mean", 0.0)),
                }
            )
        for global_row in global_rows:
            if metrics_fh is not None:
                base._write_metrics_history(metrics_fh, global_row)
            if writer is not None:
                mode = str(global_row.get("mode", "all"))
                writer.add_scalar(
                    f"iforward_validation/global/{mode}/current_psnr",
                    float(global_row["current_psnr"]),
                    int(trigger_step),
                )
                writer.add_scalar(
                    f"iforward_validation/global/{mode}/history_rollout_psnr",
                    float(global_row["history_rollout_psnr"]),
                    int(trigger_step),
                )
                writer.add_scalar(
                    f"iforward_validation/global/{mode}/nearby_psnr",
                    float(global_row["nearby_psnr"]),
                    int(trigger_step),
                )
        if writer is not None:
            flush = getattr(writer, "flush", None)
            if callable(flush):
                flush()


def _iforward_train_start_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    if _is_stage2_2_scheduler_cfg(cfg):
        val_cfg = stage2_2_validation_cfg(cfg)
        if bool(val_cfg["enable"]) and bool(val_cfg["run_at_train_start"]):
            metrics_fh = kwargs.get("metrics_fh", None)
            if kwargs.get("model", None) is not None and kwargs.get("dataset", None) is not None:
                rows = run_stage2_2_validation(
                    cfg=cfg,
                    dataset=kwargs["dataset"],
                    model=kwargs["model"],
                    device=kwargs.get("device", torch.device("cpu")),
                    trigger_step=int(kwargs.get("trigger_step", 0)),
                    modes=list(val_cfg.get("modes", ["full"])),
                    convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
                )
                for row in rows:
                    if metrics_fh is not None:
                        base._write_metrics_history(metrics_fh, row)
                entries = rows
            else:
                entries = run_stage2_2_validation_manifest_only(cfg=cfg)
            if metrics_fh is not None:
                base._write_metrics_history(
                    metrics_fh,
                    {
                        "step": int(kwargs.get("trigger_step", 0)),
                        "split": "iforward_stage2_2_validation_global",
                        "num_entries": int(len(entries)),
                        "protocols": sorted({str(e.get("protocol", "")) for e in entries}),
                        "status": "completed" if entries else "empty",
                    },
                )
        return
    if _is_stage2_3_scheduler_cfg(cfg):
        val_cfg = stage2_3_validation_cfg(cfg)
        metrics_fh = kwargs.get("metrics_fh", None)
        trigger_step = int(kwargs.get("trigger_step", 0))
        if not bool(val_cfg["enable"]):
            _write_stage2_3_validation_skip_status(
                cfg=cfg,
                metrics_fh=metrics_fh,
                trigger="train_start",
                trigger_step=trigger_step,
                reason="disabled",
                val_cfg=val_cfg,
            )
        elif not bool(val_cfg["run_at_train_start"]):
            _write_stage2_3_validation_skip_status(
                cfg=cfg,
                metrics_fh=metrics_fh,
                trigger="train_start",
                trigger_step=trigger_step,
                reason="run_at_train_start_false",
                val_cfg=val_cfg,
            )
        else:
            entries = _run_stage2_3_validation_with_status(
                cfg=cfg,
                dataset=kwargs.get("dataset", None),
                model=kwargs.get("model", None),
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=trigger_step,
                trigger="train_start",
                val_cfg=val_cfg,
                metrics_fh=metrics_fh,
                writer=kwargs.get("writer", None),
                convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
            )
            if metrics_fh is not None:
                base._write_metrics_history(
                    metrics_fh,
                    {
                        "step": trigger_step,
                        "split": "iforward_stage2_3_validation_global",
                        "num_entries": int(len(entries)),
                        "protocols": sorted({str(e.get("protocol", "")) for e in entries}),
                        "status": "completed" if entries else "empty",
                    },
                )
        if _validation_v4_due(cfg, trigger="train_start", step=trigger_step):
            _run_validation_v4_with_status(
                cfg=cfg,
                dataset=kwargs.get("dataset", None),
                model=kwargs.get("model", None),
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=trigger_step,
                trigger="train_start",
                metrics_fh=metrics_fh,
            )
        return
    if _is_sequence10_scheduler_cfg(cfg):
        val_cfg = _iforward_sequence10_validation_cfg(cfg)
        if bool(val_cfg["enable"]) and bool(val_cfg["run_at_train_start"]):
            _write_iforward_sequence10_validation_rows(**kwargs)
        return
    coverage_cfg = iforward_coverage_validation_cfg(cfg)
    if bool(coverage_cfg["enable"]) and bool(coverage_cfg["run_at_train_start"]):
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("convert_batch_to_minimal_format", base.convert_batch_to_minimal_format)
        call_kwargs.setdefault("write_metrics_history", base._write_metrics_history)
        write_iforward_coverage_validation_rows(**call_kwargs)
    val_cfg = _iforward_validation_cfg(cfg)
    if bool(val_cfg["enable"]) and bool(val_cfg["run_at_train_start"]):
        _write_iforward_validation_rows(**kwargs)


def _iforward_step_end_hook(**kwargs: Any) -> None:
    cfg = kwargs["cfg"]
    if _is_stage2_2_scheduler_cfg(cfg):
        val_cfg = stage2_2_validation_cfg(cfg)
        interval = int(val_cfg["interval_steps"])
        step = int(kwargs.get("trigger_step", 0))
        if bool(val_cfg["enable"]) and interval > 0 and step >= 0 and (step + 1) % int(interval) == 0:
            metrics_fh = kwargs.get("metrics_fh", None)
            rows = run_stage2_2_validation(
                cfg=cfg,
                dataset=kwargs["dataset"],
                model=kwargs["model"],
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=int(step),
                modes=list(val_cfg.get("modes", ["full"])),
                convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
            )
            for row in rows:
                if metrics_fh is not None:
                    base._write_metrics_history(metrics_fh, row)
            if metrics_fh is not None:
                base._write_metrics_history(
                    metrics_fh,
                    {
                        "step": int(step),
                        "split": "iforward_stage2_2_validation_global",
                        "num_entries": int(len(rows)),
                        "protocols": sorted({str(e.get("protocol", "")) for e in rows}),
                        "status": "completed" if rows else "empty",
                    },
                )
        if _validation_v4_due(cfg, trigger="interval", step=step):
            _run_validation_v4_with_status(
                cfg=cfg,
                dataset=kwargs.get("dataset", None),
                model=kwargs.get("model", None),
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=int(step),
                trigger="interval",
                metrics_fh=kwargs.get("metrics_fh", None),
            )
        return
    if _is_stage2_3_scheduler_cfg(cfg):
        val_cfg = stage2_3_validation_cfg(cfg)
        interval = int(val_cfg["interval_steps"])
        step = int(kwargs.get("trigger_step", 0))
        if bool(val_cfg["enable"]) and interval > 0 and step >= 0 and (step + 1) % int(interval) == 0:
            metrics_fh = kwargs.get("metrics_fh", None)
            rows = _run_stage2_3_validation_with_status(
                cfg=cfg,
                dataset=kwargs.get("dataset", None),
                model=kwargs.get("model", None),
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=int(step),
                trigger="interval",
                val_cfg=val_cfg,
                metrics_fh=metrics_fh,
                writer=kwargs.get("writer", None),
                convert_batch_to_minimal_format=_sequence10_minimal_from_scheduler_batch,
            )
            if metrics_fh is not None:
                base._write_metrics_history(
                    metrics_fh,
                    {
                        "step": int(step),
                        "split": "iforward_stage2_3_validation_global",
                        "num_entries": int(len(rows)),
                        "protocols": sorted({str(e.get("protocol", "")) for e in rows}),
                        "status": "completed" if rows else "empty",
                    },
                )
        if _validation_v4_due(cfg, trigger="interval", step=step):
            _run_validation_v4_with_status(
                cfg=cfg,
                dataset=kwargs.get("dataset", None),
                model=kwargs.get("model", None),
                device=kwargs.get("device", torch.device("cpu")),
                trigger_step=int(step),
                trigger="interval",
                metrics_fh=kwargs.get("metrics_fh", None),
            )
        return
    if _is_sequence10_scheduler_cfg(cfg):
        val_cfg = _iforward_sequence10_validation_cfg(cfg)
        interval = int(val_cfg["interval_steps"])
        step = int(kwargs.get("trigger_step", 0))
        if bool(val_cfg["enable"]) and interval > 0 and step >= 0 and (step + 1) % int(interval) == 0:
            _write_iforward_sequence10_validation_rows(**kwargs)
        return
    coverage_cfg = iforward_coverage_validation_cfg(cfg)
    coverage_interval = int(coverage_cfg["interval_steps"])
    val_cfg = _iforward_validation_cfg(cfg)
    interval = int(val_cfg["interval_steps"])
    step = int(kwargs.get("trigger_step", 0))
    if bool(coverage_cfg["enable"]) and coverage_interval > 0 and step >= 0 and (step + 1) % int(coverage_interval) == 0:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("convert_batch_to_minimal_format", base.convert_batch_to_minimal_format)
        call_kwargs.setdefault("write_metrics_history", base._write_metrics_history)
        write_iforward_coverage_validation_rows(**call_kwargs)
    if not bool(val_cfg["enable"]) or interval <= 0:
        return
    if step < 0 or (step + 1) % int(interval) != 0:
        return
    _write_iforward_validation_rows(**kwargs)


def main() -> None:
    default_config = "configs/iforward/iforward_base.yaml"
    if not any(arg == "--config_file" or arg.startswith("--config_file=") for arg in sys.argv):
        sys.argv.extend(["--config_file", default_config])
    if _route_random_window_entrypoint_if_needed(default_config):
        return
    base.build_multi_scene_dataset_v3 = build_multi_scene_dataset_v4
    base.build_train_scheduler_from_cfg = build_train_scheduler_iforward_from_cfg
    base.resolve_fixed_scene_segment = resolve_fixed_scene_segment_iforward
    base.TRAINER_CLASS = build_iforward_trainer_from_cfg
    base.MinimalStreetForwardStage4_3 = build_iforward_trainer_from_cfg
    base.RUN_START_HOOK = _iforward_run_start_hook
    base.TRAIN_START_HOOK = _iforward_train_start_hook
    base.STEP_END_HOOK = _iforward_step_end_hook
    base.CKPT_PREFIX = "iforward_v1"
    base.CHECKPOINT_PREFIX_RESOLVER = checkpoint_prefix_iforward_from_cfg
    base.DEFAULT_CONFIG_FILE = default_config
    base.ALLOW_ONE_SEGMENT = False
    base.main()


if __name__ == "__main__":
    main()
