"""Bounded CUDA acceptance profile for IForward Stage 3.4 Functional ParentGS.

The profile compares the Stage 3.3 source-only configuration with the Stage
3.4 functional ParentGS configuration.  It uses the production dataset,
scheduler, trainer, and ``train_step`` transaction, but fixes one
scene/segment and compiles every measured rollout as B5R3 (K=15).  One prelude
rollout warms kernels and establishes carried state; the following repair
rollouts are timed and checked.

This is intentionally a short acceptance tool.  It never writes checkpoints
and is not a replacement for the documented 1000-step A/B follow-up.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import shlex
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from models.iforward import IForwardTrainer
from tools.train_minimal_streetforward_stage4_3_iforward_common import (
    build_multi_scene_dataset_v4,
    build_train_scheduler_iforward_from_cfg,
)
from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import (
    convert_batch_to_minimal_format,
)


DEFAULT_BASELINE_CONFIG = "configs/iforward/iforward_stage3_3_observation_feedback.yaml"
DEFAULT_CANDIDATE_CONFIG = "configs/iforward/iforward_stage3_4_functional_parentgs_lift.yaml"
PROFILE_GLOBAL_STEP = 15_000

_STAGE34_SENTINELS: Mapping[str, float] = {
    "iforward/stage3_4/enabled": 1.0,
    "iforward/stage3_4/functional_parent_enabled": 1.0,
    "iforward/stage3_4/functional_parent_direct_lift_enabled": 1.0,
    "iforward/stage3_4/parent_runtime_enabled": 0.0,
    "iforward/stage3_4/surrogate_vjp_enabled": 0.0,
    "iforward/stage3_4/relation_feedback_enabled": 0.0,
    "iforward/stage3_4/parent_lift_geometry_grad": 0.0,
    "iforward/stage3_4/lift_geometry_grad_enabled": 0.0,
    "iforward/feedback/relation_enabled": 0.0,
}

_ISOLATION_SENTINELS: Mapping[str, float] = {
    "feedback/parent_lift/geometry_grad_configured_off": 1.0,
    "feedback/ptv3_coords/geometry_grad_configured_off": 1.0,
    "feedback/relation/geometry_grad_configured_off": 1.0,
}

_ISOLATION_ASSERTIONS: Mapping[str, float] = {
    "feedback/parent_lift/boundary_assertion_passed": 1.0,
    "feedback/ptv3_coords/boundary_assertion_passed": 1.0,
    "feedback/relation/boundary_assertion_passed": 1.0,
}

_BRANCH_DIAGNOSTICS = (
    "num_children",
    "num_parents",
    "project_ms",
    "lift_ms",
    "parent_scale_clamp_ratio",
    "parent_opacity_cap_ratio",
    "parent_support_mean",
)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value
    value = getattr(node, key, default)
    return default if value is None else value


def _clone_cfg(cfg: Any) -> Any:
    return OmegaConf.create(copy.deepcopy(OmegaConf.to_container(cfg, resolve=False)))


def _feedback_cfg(cfg: Any) -> Any:
    return cfg.model.iforward.observation_feedback


def _validate_profile_identity(cfg: Any, *, role: str) -> None:
    iforward = cfg.model.iforward
    version = str(_cfg_get(iforward, "version", ""))
    variant = str(_cfg_get(iforward, "training_variant", ""))
    feedback = _feedback_cfg(cfg)
    parent_enabled = bool(_cfg_get(_cfg_get(feedback, "parent_projection", {}), "enable", False))
    relation_enabled = bool(_cfg_get(_cfg_get(feedback, "relation", {}), "enable", False))
    if parent_enabled or relation_enabled:
        raise ValueError(
            f"{role} profile must be source-only at the feedback policy boundary; "
            f"parent_projection.enable={parent_enabled}, relation.enable={relation_enabled}"
        )
    if role == "candidate":
        expected = "stage3_4_functional_parentgs_lift"
        if version != expected or variant != expected:
            raise ValueError(
                "candidate config must retain the independent Stage 3.4 identity: "
                f"version={version!r}, training_variant={variant!r}"
            )
    elif version == "stage3_4_functional_parentgs_lift":
        raise ValueError("baseline config unexpectedly selects Stage 3.4")


def _fixed_k15_cfg(
    base_cfg: Any,
    *,
    role: str,
    scene_id: int,
    segment_id: int,
    seed: int,
    measured_rollouts: int,
) -> Any:
    """Return a production config narrowed to one deterministic K=15 episode."""

    _validate_profile_identity(base_cfg, role=role)
    cfg = _clone_cfg(base_cfg)
    cfg.scheduler_stage3_0.traversal.fixed_scene_id = int(scene_id)
    cfg.scheduler_stage3_0.traversal.fixed_segment_id = int(segment_id)
    cfg.scheduler_stage3_0.traversal.seed = int(seed)
    cfg.scheduler_stage3_0.producer.enable = False
    cfg.scheduler_stage3_2.max_inner_k_hard_cap = 15
    cfg.scheduler_stage3_2.episode_recipe.prelude.min_rollouts = 1
    cfg.scheduler_stage3_2.episode_recipe.prelude.max_rollouts = 1
    cfg.scheduler_stage3_2.episode_recipe.repair_tail.min_rollouts = int(measured_rollouts)
    cfg.scheduler_stage3_2.episode_recipe.repair_tail.max_rollouts = int(measured_rollouts)
    # B5R3 is exactly K=15 and covers repeated updates plus cross-block routing.
    cfg.scheduler_stage3_2.distributions.high_block_repair.b_choices = {5: 1.0}
    cfg.scheduler_stage3_2.distributions.high_block_repair.r_choices = {3: 1.0}
    for phase in cfg.scheduler_stage3_2.curriculum:
        phase.weights.high_block_repair = 1.0
        phase.max_k.frozen_2d.high_block_repair = 15

    feedback = _feedback_cfg(cfg)
    feedback.enable = True
    feedback.schedule.activation_step = 0
    feedback.source_render.enable = True
    # Keep the prelude forward-identical, then exercise source feedback on all
    # measured rollouts.  Functional Parent gradients do not use the legacy
    # parent_projection policy switch.
    feedback.source_render.alpha_schedule = [[0, 0.0], [1, 1.0]]
    if str(role) == "candidate":
        functional_parent = getattr(feedback, "functional_parent", None)
        if functional_parent is None or not bool(functional_parent.enable):
            raise ValueError("Stage 3.4 profiler candidate requires observation_feedback.functional_parent")
        if list(functional_parent.branches) != ["bg", "distant", "rigid_active"]:
            raise ValueError("Stage 3.4 profiler candidate requires all three Functional Parent branches")
    feedback.modes.high_block_repair = "frozen_input_grad_checkpointed"
    feedback.debug.grad_probe_interval = 1
    feedback.debug.forward_parity_interval = 0
    cfg.scheduler_stage3_2.episode_recipe.train_2d_policy.high_block_repair = (
        "frozen_input_grad_checkpointed"
    )

    cfg.data.train_scene_ids = [int(scene_id)]
    cfg.data.eval_scene_ids = []
    cfg.data.pixel_source.require_egocar_mask_template = False
    if _cfg_get(cfg, "dataset", None) is not None:
        cfg.dataset.preload_scene_count = 1
    for key in (
        "scheduler_stage3_0_validation",
        "iforward_validation_v4",
        "iforward_sequence10_validation",
        "iforward_coverage_validation",
    ):
        section = _cfg_get(cfg, key, None)
        if section is not None and _cfg_get(section, "enable", None) is not None:
            section.enable = False
    return cfg


def _minimal_batch(raw_batch: Dict[str, Any], *, device: torch.device, step: int) -> Dict[str, Any]:
    target = raw_batch.get("target") or {}
    image = target.get("image") if isinstance(target, dict) else None
    if not torch.is_tensor(image):
        raise ValueError("profile scheduler batch must contain target.image")
    batch = convert_batch_to_minimal_format(
        raw_batch,
        device,
        num_targets=int(image.shape[0]),
        include_source_for_2d=True,
        view_selection=None,
    )
    batch["global_step"] = int(step)
    return batch


def _stage32_meta(raw_batch: Dict[str, Any]) -> Dict[str, Any]:
    request = dict((raw_batch.get("_iforward") or {}).get("request_meta") or {})
    if not request:
        request = dict(raw_batch.get("request_meta") or {})
    return dict(request.get("iforward_stage3_2") or {})


def _seed_profile_rng(seed: int) -> None:
    """Reset every process-global RNG used by dataset assembly/train_step."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _metadata_signature(meta: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(meta), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonfinite_paths(value: Any, *, path: str = "result", limit: int = 24) -> list[str]:
    failures: list[str] = []

    def visit(item: Any, item_path: str) -> None:
        if len(failures) >= int(limit):
            return
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)):
                failures.append(item_path)
            return
        if torch.is_tensor(item):
            if torch.is_floating_point(item) and not bool(torch.isfinite(item.detach()).all().item()):
                failures.append(item_path)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{item_path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for idx, child in enumerate(item):
                visit(child, f"{item_path}[{idx}]")

    visit(value, str(path))
    return failures


def _numeric_metric(result: Mapping[str, Any], key: str) -> Optional[float]:
    value = result.get(str(key))
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if torch.is_tensor(value) and int(value.numel()) == 1:
        scalar = float(value.detach().item())
        return scalar if math.isfinite(scalar) else None
    return None


def _legacy_parent_runtime_keys(result: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for raw_key in result:
        key = str(raw_key)
        if key.startswith("iforward/stage3_4/parent_runtime_enabled"):
            continue
        if key == "iforward/feedback/parent_vjp_enabled":
            continue
        if (
            "parent_runtime" in key
            or "runtime_update" in key
            or "incremental_update" in key
            or "exact_refresh" in key
            or "/drift" in key
            or "drift_" in key
            or key.startswith("feedback/parent_vjp/")
        ):
            keys.append(key)
    return sorted(keys)


def _interesting_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    prefixes = (
        "iforward/stage3_4/",
        "iforward/feedback/",
        "feedback/functional_parent/",
        "feedback/parent_lift/",
        "feedback/ptv3_coords/",
        "feedback/relation/",
    )
    metrics: Dict[str, float] = {}
    for key in sorted(str(item) for item in result):
        if not key.startswith(prefixes):
            continue
        value = _numeric_metric(result, key)
        if value is not None:
            metrics[key] = float(value)
    return metrics


def _check_equal_metrics(
    result: Mapping[str, Any],
    expected: Mapping[str, float],
    *,
    tolerance: float = 1.0e-8,
) -> Dict[str, bool]:
    return {
        key: (
            _numeric_metric(result, key) is not None
            and abs(float(_numeric_metric(result, key)) - float(want)) <= float(tolerance)
        )
        for key, want in expected.items()
    }


def _stage34_sample_checks(result: Mapping[str, Any], *, actual_k: int) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    checks.update({f"sentinel:{key}": ok for key, ok in _check_equal_metrics(result, _STAGE34_SENTINELS).items()})
    checks.update({f"isolation:{key}": ok for key, ok in _check_equal_metrics(result, _ISOLATION_SENTINELS).items()})
    checks.update(
        {
            f"boundary_assertion:{key}": ok
            for key, ok in _check_equal_metrics(result, _ISOLATION_ASSERTIONS).items()
        }
    )

    gradient_keys = [
        f"feedback/functional_parent/{branch}/{role}_{attr}_grad_norm"
        for branch in ("bg", "distant", "rigid_active")
        for role in ("parent",)
        for attr in ("means", "scales", "opacity")
    ]
    gradient_keys.extend(
        [
            "feedback/functional_parent/earlier_delta_grad_norm/distance_1",
            "feedback/functional_parent/earlier_delta_grad_norm/distance_2",
            "feedback/parent_lift/features_2d_grad_norm",
        ]
    )
    for key in gradient_keys:
        checks[f"gradient_metric_finite:{key}"] = _numeric_metric(result, key) is not None

    # Every present branch must publish its projector/lifting/support/clamp row.
    present_branches = [
        branch
        for branch in ("bg", "distant", "rigid_active")
        if _numeric_metric(result, f"iforward/stage3_4/{branch}/num_parents") is not None
    ]
    checks["branch_diagnostics:bg_present"] = "bg" in present_branches
    for branch in present_branches:
        for suffix in _BRANCH_DIAGNOSTICS:
            key = f"iforward/stage3_4/{branch}/{suffix}"
            checks[f"branch_diagnostic_finite:{key}"] = _numeric_metric(result, key) is not None

        parent_grad = sum(
            float(_numeric_metric(result, f"feedback/functional_parent/{branch}/parent_{attr}_grad_norm") or 0.0)
            for attr in ("means", "scales", "opacity")
        )
        checks[f"functional_parent_grad_positive:{branch}"] = parent_grad > 0.0

    checks["earlier_delta_distance_1_positive"] = (
        float(_numeric_metric(result, "feedback/functional_parent/earlier_delta_grad_norm/distance_1") or 0.0)
        > 0.0
    )
    checks["earlier_delta_distance_2_positive"] = (
        float(_numeric_metric(result, "feedback/functional_parent/earlier_delta_grad_norm/distance_2") or 0.0)
        > 0.0
    )
    checks["parent_lift_feature_grad_positive"] = (
        float(_numeric_metric(result, "feedback/parent_lift/features_2d_grad_norm") or 0.0) > 0.0
    )
    # These aggregate identities prove that the functional gate reset at the
    # rollout boundary, used forward-only on exactly the first visit, and then
    # attached every visit after a real updater delta.
    first_visit_mean = _numeric_metric(
        result,
        "iforward/stage3_4/first_visit_forward_only_mean",
    )
    ancestor_mean = _numeric_metric(
        result,
        "iforward/stage3_4/has_update_ancestor_mean",
    )
    update_count = _numeric_metric(result, "iforward/stage3_4/model_update_count")
    functional_grad_mean = _numeric_metric(
        result,
        "iforward/feedback/functional_parent/grad_active_mean",
    )
    functional_forward_only_mean = _numeric_metric(
        result,
        "iforward/feedback/functional_parent/forward_only_mean",
    )
    expected_first_mean = 1.0 / float(max(int(actual_k), 1))
    expected_ancestor_mean = float(max(int(actual_k) - 1, 0)) / float(max(int(actual_k), 1))
    checks["first_visit_forward_only_exactly_once"] = (
        first_visit_mean is not None
        and abs(float(first_visit_mean) - expected_first_mean) <= 1.0e-6
    )
    checks["update_ancestor_gate_after_first_visit"] = (
        ancestor_mean is not None
        and abs(float(ancestor_mean) - expected_ancestor_mean) <= 1.0e-6
    )
    checks["functional_parent_gate_after_first_visit"] = (
        functional_grad_mean is not None
        and abs(float(functional_grad_mean) - expected_ancestor_mean) <= 1.0e-6
    )
    checks["functional_parent_forward_only_exactly_first_visit"] = (
        functional_forward_only_mean is not None
        and abs(float(functional_forward_only_mean) - expected_first_mean) <= 1.0e-6
    )
    checks["model_update_count_matches_k"] = (
        update_count is not None and int(round(float(update_count))) == int(actual_k)
    )
    checks["legacy_parent_runtime_keys_absent"] = not _legacy_parent_runtime_keys(result)
    optimizer_skipped = _numeric_metric(result, "amp/optimizer_step_skipped")
    checks["optimizer_step_metric_present"] = optimizer_skipped is not None
    checks["optimizer_step_not_skipped"] = (
        optimizer_skipped is not None and float(optimizer_skipped) == 0.0
    )
    return checks


def _baseline_sample_checks(result: Mapping[str, Any]) -> Dict[str, bool]:
    source_norm_metric = _numeric_metric(result, "feedback/source_render_input_grad_norm")
    render_enabled = _numeric_metric(result, "iforward/feedback/render_enabled")
    parent_vjp_enabled = _numeric_metric(result, "iforward/feedback/parent_vjp_enabled")
    relation_enabled = _numeric_metric(result, "iforward/feedback/relation_enabled")
    optimizer_skipped = _numeric_metric(result, "amp/optimizer_step_skipped")
    return {
        "source_feedback_enabled": render_enabled is not None and float(render_enabled) == 1.0,
        "source_feedback_grad_positive": (
            source_norm_metric is not None and float(source_norm_metric) > 0.0
        ),
        "parent_vjp_disabled": (
            parent_vjp_enabled is not None and float(parent_vjp_enabled) == 0.0
        ),
        "relation_feedback_disabled": (
            relation_enabled is not None and float(relation_enabled) == 0.0
        ),
        "optimizer_step_metric_present": optimizer_skipped is not None,
        "optimizer_step_not_skipped": (
            optimizer_skipped is not None and float(optimizer_skipped) == 0.0
        ),
    }


def _profile_variant(
    *,
    base_cfg: Any,
    dataset: Any,
    role: str,
    device: torch.device,
    scene_id: int,
    segment_id: int,
    seed: int,
    samples: int,
) -> Dict[str, Any]:
    cfg = _fixed_k15_cfg(
        base_cfg,
        role=str(role),
        scene_id=int(scene_id),
        segment_id=int(segment_id),
        seed=int(seed),
        measured_rollouts=int(samples),
    )
    _seed_profile_rng(int(seed))
    scheduler = None
    trainer = None
    rows: list[Dict[str, Any]] = []
    prelude_meta: Dict[str, Any] = {}
    try:
        scheduler = build_train_scheduler_iforward_from_cfg(cfg, dataset)
        # Model construction consumes a version-dependent number of random
        # values.  Restore the generator state afterwards so both variants
        # hand the scheduler the same RNG stream for batch assembly.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all()
        trainer = IForwardTrainer(config=cfg, device=device).train()
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state_all(cuda_rng_states)

        _seed_profile_rng(int(seed) + 10_000)
        prelude_raw = scheduler.next_batch()
        prelude_meta = _stage32_meta(prelude_raw)
        prelude_stage = str(prelude_meta.get("episode_stage", ""))
        prelude_distribution = str(prelude_meta.get("distribution_type", ""))
        if prelude_stage != "prelude" or prelude_distribution not in {
            "repeat_refine",
            "shuffled_coverage",
        }:
            raise RuntimeError(
                "expected one Stage 3.2 prelude rollout before repair measurement, "
                f"metadata={prelude_meta}"
            )
        prelude_step = int(PROFILE_GLOBAL_STEP)
        prelude = _minimal_batch(prelude_raw, device=device, step=prelude_step)
        _seed_profile_rng(int(seed) + 20_000)
        prelude_result = trainer.train_step(prelude, step=prelude_step)
        prelude_nonfinite = _nonfinite_paths(prelude_result, path="prelude")
        if prelude_nonfinite:
            raise RuntimeError(f"non-finite prelude outputs: {prelude_nonfinite}")
        del prelude, prelude_raw, prelude_result

        for sample_idx in range(int(samples)):
            batch_rng_seed = int(seed) + 10_001 + int(sample_idx)
            step_rng_seed = int(seed) + 20_001 + int(sample_idx)
            _seed_profile_rng(batch_rng_seed)
            raw = scheduler.next_batch()
            meta = _stage32_meta(raw)
            actual_k = int(meta.get("K", 0) or 0)
            actual_b = int(meta.get("B", 0) or 0)
            actual_r = int(meta.get("R", 0) or 0)
            episode_stage = str(meta.get("episode_stage", ""))
            if (
                str(meta.get("distribution_type", "")) != "high_block_repair"
                or episode_stage != "repair_tail"
                or actual_b != 5
                or actual_r != 3
                or actual_k != 15
            ):
                raise RuntimeError(
                    "expected fixed repair_tail high_block_repair B5R3/K=15, "
                    f"sample={sample_idx} metadata={meta}"
                )
            profile_step = int(PROFILE_GLOBAL_STEP + sample_idx + 1)
            batch = _minimal_batch(raw, device=device, step=profile_step)
            _seed_profile_rng(step_rng_seed)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            started = time.perf_counter()
            result = trainer.train_step(
                batch,
                step=profile_step,
                profile_phase_timing=True,
                sync_cuda_timing=True,
                profile_cuda_memory=True,
            )
            torch.cuda.synchronize(device)
            elapsed_ms = float((time.perf_counter() - started) * 1000.0)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            nonfinite = _nonfinite_paths(result)
            checks = (
                _stage34_sample_checks(result, actual_k=actual_k)
                if role == "candidate"
                else _baseline_sample_checks(result)
            )
            checks["all_outputs_finite"] = not nonfinite
            loss_metric = _numeric_metric(result, "loss")
            episode_end_metric = _numeric_metric(
                result,
                "iforward/episode_end_after_rollout",
            )
            mb = 1024.0 * 1024.0
            row: Dict[str, Any] = {
                "sample": int(sample_idx),
                "status": "ok" if all(checks.values()) else "gate_failed",
                "distribution": str(meta.get("distribution_type", "")),
                "episode_stage": episode_stage,
                "B": int(actual_b),
                "R": int(actual_r),
                "K": int(actual_k),
                "mode": str(meta.get("train_2d_mode", "")),
                "metadata_signature": _metadata_signature(meta),
                "batch_rng_seed": batch_rng_seed,
                "step_rng_seed": step_rng_seed,
                "episode_end_after_rollout": bool(
                    episode_end_metric is not None and float(episode_end_metric) != 0.0
                ),
                "step_time_ms": float(elapsed_ms),
                "allocated_before_mb": float(allocated_before / mb),
                "reserved_before_mb": float(reserved_before / mb),
                "peak_allocated_mb": float(peak_allocated / mb),
                "peak_reserved_mb": float(peak_reserved / mb),
                "peak_incremental_allocated_mb": float((peak_allocated - allocated_before) / mb),
                "loss": float(loss_metric) if loss_metric is not None else float("nan"),
                "nonfinite_paths": nonfinite,
                "legacy_parent_runtime_keys": _legacy_parent_runtime_keys(result),
                "checks": checks,
                "failed_checks": sorted(key for key, passed in checks.items() if not passed),
                "metrics": _interesting_metrics(result),
            }
            # Retained-allocation sampling happens after temporary batch/result
            # graphs are released and outside the timed interval.  Terminal
            # episode rows are excluded from the spread below because their
            # carried state has a different lifecycle.
            del batch, raw, result
            gc.collect()
            torch.cuda.synchronize(device)
            row["allocated_after_mb"] = float(
                torch.cuda.memory_allocated(device) / mb
            )
            rows.append(row)
            print(json.dumps({"role": role, **row}, sort_keys=True), flush=True)

        step_times = [float(row["step_time_ms"]) for row in rows]
        peak_values = [float(row["peak_allocated_mb"]) for row in rows]
        retained_rows = [
            row for row in rows if not bool(row["episode_end_after_rollout"])
        ]
        allocated_after_values = [
            float(row["allocated_after_mb"]) for row in retained_rows
        ]
        if len(allocated_after_values) < 2:
            raise RuntimeError(
                "retained CUDA spread requires at least two non-terminal repair rollouts"
            )
        retained_cuda_spread_mb = float(
            max(allocated_after_values) - min(allocated_after_values)
        )
        all_legacy_keys = sorted(
            {key for row in rows for key in list(row.get("legacy_parent_runtime_keys", []))}
        )
        return {
            "role": str(role),
            "status": "ok" if rows and all(row["status"] == "ok" for row in rows) else "gate_failed",
            "prelude_distribution": prelude_distribution,
            "prelude_episode_stage": prelude_stage,
            "prelude_k": int(prelude_meta.get("K", 0) or 0),
            "sample_count": int(len(rows)),
            "median_step_time_ms": float(statistics.median(step_times)),
            "max_peak_allocated_mb": float(max(peak_values)),
            "median_peak_allocated_mb": float(statistics.median(peak_values)),
            "retained_cuda_growth_mb": float(allocated_after_values[-1] - allocated_after_values[0]),
            "retained_cuda_spread_mb": retained_cuda_spread_mb,
            "retained_cuda_sample_count": int(len(allocated_after_values)),
            "legacy_parent_runtime_keys": all_legacy_keys,
            "legacy_parent_runtime_keys_detected": bool(all_legacy_keys),
            "rows": rows,
        }
    finally:
        shutdown_scheduler = getattr(scheduler, "shutdown", None)
        if callable(shutdown_scheduler):
            shutdown_scheduler()
        del trainer, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def _common_one_segment_overrides(*, scene_id: int, segment_id: int, seed: int, log_dir: str) -> list[str]:
    return [
        f"output_name={Path(log_dir).name}",
        f"logging.project={Path(log_dir).name}",
        f"logging.log_dir={log_dir}",
        "logging.metrics_history_append=false",
        "logging.train_step_metrics_interval=1",
        "logging.scheduler_metrics_interval=1",
        "logging.performance.enable=true",
        "logging.performance.phase_timing=true",
        "logging.performance.cuda_memory=true",
        "training.save_checkpoint_freq=1000",
        f"training.seed={int(seed)}",
        f"scheduler_stage3_0.traversal.fixed_scene_id={int(scene_id)}",
        f"scheduler_stage3_0.traversal.fixed_segment_id={int(segment_id)}",
        f"scheduler_stage3_0.traversal.seed={int(seed)}",
        "scheduler_stage3_0.producer.enable=false",
        f"data.train_scene_ids=[{int(scene_id)}]",
        "data.eval_scene_ids=[]",
        "data.pixel_source.require_egocar_mask_template=false",
        "dataset.preload_scene_count=1",
        "scheduler_stage3_0_validation.enable=false",
        "iforward_validation_v4.enable=false",
        "eval.run_test_at_end=false",
        "model.iforward.observation_feedback.schedule.activation_step=0",
        "model.iforward.observation_feedback.source_render.alpha_schedule=[[0,1.0]]",
        "model.iforward.observation_feedback.debug.grad_probe_interval=1",
    ]


def _command_plan(args: argparse.Namespace) -> Dict[str, Any]:
    env_prefix = [
        "conda",
        "run",
        "-n",
        "drivestudio-new",
        "env",
        "PYTHONPATH=/root/drivestudio-coding",
    ]
    profile_command = [
        *env_prefix,
        "python",
        "tools/profile_iforward_stage3_4_k15.py",
        "--baseline-config",
        str(args.baseline_config),
        "--candidate-config",
        str(args.candidate_config),
        "--scene-id",
        str(args.scene_id),
        "--segment-id",
        str(args.segment_id),
        "--seed",
        str(args.seed),
        "--samples",
        str(args.samples),
        "--max-peak-ratio",
        str(args.max_peak_ratio),
        "--max-time-ratio",
        str(args.max_time_ratio),
        "--max-retained-growth-mb",
        str(args.max_retained_growth_mb),
        "--output-json",
        str(args.output_json or "/root/autodl-tmp/outputs/iforward_stage3_4_k15/profile.json"),
    ]
    output_root = "/root/autodl-tmp/outputs/iforward_stage3_4_fixed_segment_ab"
    baseline_overrides = _common_one_segment_overrides(
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(args.seed),
        log_dir=f"{output_root}/stage3_3_source_only",
    )
    baseline_overrides.extend(
        [
            "initialization.skip_keys=[]",
            "model.iforward.observation_feedback.parent_projection.enable=false",
            "model.iforward.observation_feedback.relation.enable=false",
        ]
    )
    candidate_overrides = _common_one_segment_overrides(
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(args.seed),
        log_dir=f"{output_root}/stage3_4_functional_parentgs",
    )
    candidate_overrides.extend(
        [
            "model.iforward.observation_feedback.parent_projection.enable=false",
            "model.iforward.observation_feedback.relation.enable=false",
        ]
    )

    def train_command(config: str, overrides: Sequence[str]) -> list[str]:
        return [
            *env_prefix,
            "python",
            "tools/train_iforward_one_segment.py",
            "--config_file",
            str(config),
            "--max_steps",
            "1000",
            "--seed",
            str(args.seed),
            "--init_checkpoint",
            "${INIT_CKPT}",
            "--init_weights_only",
            *list(overrides),
        ]

    def render(command: Sequence[str]) -> str:
        # Preserve intentional environment expansion for the documented
        # checkpoint variable while shell-quoting every other argument.
        return " ".join(
            '"${INIT_CKPT}"' if str(token) == "${INIT_CKPT}" else shlex.quote(str(token))
            for token in command
        )

    commands = {
        "short_k15_profile": render(profile_command),
        "stage3_3_source_only_1000": render(
            train_command(str(args.baseline_config), baseline_overrides)
        ),
        "stage3_4_candidate_1000": render(
            train_command(str(args.candidate_config), candidate_overrides)
        ),
    }
    return {
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "seed": int(args.seed),
        "init_checkpoint_prerequisite": "export INIT_CKPT=/absolute/path/to/native_stage3_3_checkpoint.pt",
        "commands": commands,
        "gates": {
            "candidate_peak_memory_ratio_max": float(args.max_peak_ratio),
            "candidate_median_step_time_ratio_max": float(args.max_time_ratio),
            "candidate_retained_cuda_spread_mb_max": float(args.max_retained_growth_mb),
            "nonfinite_or_oom_allowed": False,
            "legacy_parent_runtime_keys_allowed_in_candidate": False,
            "baseline_candidate_schedule_metadata_must_match": True,
            "stage3_4_gradient_and_isolation_metrics_required": True,
        },
    }


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage 3.3 source-only and Stage 3.4 Functional ParentGS on fixed K=15 rollouts."
    )
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--candidate-config", default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--scene-id", type=int, default=131)
    parser.add_argument("--segment-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--samples", type=int, default=3, help="Measured K=15 repair rollouts per variant.")
    parser.add_argument("--max-peak-ratio", type=float, default=1.15)
    parser.add_argument("--max-time-ratio", type=float, default=1.20)
    parser.add_argument(
        "--max-retained-growth-mb",
        type=float,
        default=64.0,
        help="Maximum candidate spread in allocated CUDA memory after measured rollouts.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the bounded profile and fixed-segment 1000-step follow-up commands before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and gates without loading configs, data, or CUDA.",
    )
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="Write/print a failed report but exit zero (useful for exploratory profiling only).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if int(args.samples) < 3:
        parser.error("--samples must be >= 3 so retained CUDA spread has two carry-state samples")
    if not float(args.max_peak_ratio) > 0.0 or not float(args.max_time_ratio) > 0.0:
        parser.error("ratio gates must be > 0")
    if float(args.max_retained_growth_mb) < 0.0:
        parser.error("--max-retained-growth-mb must be >= 0")
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    command_plan = _command_plan(args)
    if bool(args.print_commands or args.dry_run):
        print(json.dumps(command_plan, indent=2, sort_keys=True))
    if bool(args.dry_run):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage 3.4 K=15 acceptance profile")

    device = torch.device("cuda")
    baseline_cfg = OmegaConf.load(str(args.baseline_config))
    candidate_cfg = OmegaConf.load(str(args.candidate_config))
    _validate_profile_identity(baseline_cfg, role="baseline")
    _validate_profile_identity(candidate_cfg, role="candidate")

    dataset_cfg = _fixed_k15_cfg(
        baseline_cfg,
        role="baseline",
        scene_id=int(args.scene_id),
        segment_id=int(args.segment_id),
        seed=int(args.seed),
        measured_rollouts=int(args.samples),
    )
    dataset = build_multi_scene_dataset_v4(dataset_cfg, device)
    dataset.initialize()
    variants: list[Dict[str, Any]] = []
    try:
        for role, cfg in (("baseline", baseline_cfg), ("candidate", candidate_cfg)):
            try:
                variants.append(
                    _profile_variant(
                        base_cfg=cfg,
                        dataset=dataset,
                        role=role,
                        device=device,
                        scene_id=int(args.scene_id),
                        segment_id=int(args.segment_id),
                        seed=int(args.seed),
                        samples=int(args.samples),
                    )
                )
            except torch.cuda.OutOfMemoryError as exc:
                variants.append(
                    {"role": role, "status": "oom", "error_type": type(exc).__name__, "error": str(exc)}
                )
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:  # Keep the other variant/report available for diagnosis.
                variants.append(
                    {"role": role, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
                )
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        shutdown_dataset = getattr(dataset, "shutdown_preload", None)
        if callable(shutdown_dataset):
            shutdown_dataset()

    baseline = next((row for row in variants if row.get("role") == "baseline"), None)
    candidate = next((row for row in variants if row.get("role") == "candidate"), None)
    comparison: Dict[str, Any] = {
        "peak_memory_ratio": None,
        "median_step_time_ratio": None,
        "peak_memory_gate": False,
        "median_step_time_gate": False,
        "candidate_retained_cuda_spread_gate": False,
        "input_schedule_gate": False,
        "baseline_gate": bool(baseline and baseline.get("status") == "ok"),
        "candidate_gate": bool(candidate and candidate.get("status") == "ok"),
    }
    if baseline and candidate and baseline.get("status") == "ok" and candidate.get("status") == "ok":
        baseline_peak = float(baseline["max_peak_allocated_mb"])
        baseline_time = float(baseline["median_step_time_ms"])
        peak_ratio = float(candidate["max_peak_allocated_mb"]) / max(baseline_peak, 1.0e-12)
        time_ratio = float(candidate["median_step_time_ms"]) / max(baseline_time, 1.0e-12)
        comparison.update(
            {
                "peak_memory_ratio": float(peak_ratio),
                "median_step_time_ratio": float(time_ratio),
                "peak_memory_gate": bool(peak_ratio <= float(args.max_peak_ratio)),
                "median_step_time_gate": bool(time_ratio <= float(args.max_time_ratio)),
                "candidate_retained_cuda_spread_gate": bool(
                    float(candidate["retained_cuda_spread_mb"])
                    <= float(args.max_retained_growth_mb)
                ),
                "input_schedule_gate": bool(
                    [row["metadata_signature"] for row in baseline["rows"]]
                    == [row["metadata_signature"] for row in candidate["rows"]]
                ),
            }
        )
    comparison["accepted"] = bool(
        comparison["baseline_gate"]
        and comparison["candidate_gate"]
        and comparison["peak_memory_gate"]
        and comparison["median_step_time_gate"]
        and comparison["candidate_retained_cuda_spread_gate"]
        and comparison["input_schedule_gate"]
    )

    report = {
        "schema": "iforward_stage3_4_functional_parentgs_k15_profile_v1",
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "baseline_config": str(args.baseline_config),
        "candidate_config": str(args.candidate_config),
        "scene_id": int(args.scene_id),
        "segment_id": int(args.segment_id),
        "seed": int(args.seed),
        "samples": int(args.samples),
        "thresholds": {
            "max_peak_memory_ratio": float(args.max_peak_ratio),
            "max_median_step_time_ratio": float(args.max_time_ratio),
            "max_candidate_retained_cuda_spread_mb": float(args.max_retained_growth_mb),
        },
        "comparison": comparison,
        "variants": variants,
        "follow_up": command_plan,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.output_json):
        output_path = Path(str(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not bool(comparison["accepted"]) and not bool(args.allow_gate_failure):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
