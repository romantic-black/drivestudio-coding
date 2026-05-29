from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from models.streetforward.stage6_0 import LocalGSState, resolve_v9_phase_a_batch
from models.streetforward.stage6_0.phase_a_losses import masked_rgb_loss, target_valid_mask
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / max(len(vals), 1)) if vals else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _branch_delta_stats(branch: Optional[BranchDelta]) -> Dict[str, float]:
    if branch is None:
        return {}

    def _norm(name: str) -> float:
        tensor = getattr(branch, name)
        if tensor.numel() == 0:
            return 0.0
        flat = tensor.detach().float()
        if flat.dim() >= 2:
            return float(flat.reshape(int(flat.shape[0]), -1).norm(dim=-1).mean().item())
        return float(flat.abs().mean().item())

    return {
        "delta_means_norm": _norm("means"),
        "delta_scales_norm": _norm("scales_log"),
        "delta_opacity_norm": _norm("opacity_logit"),
        "delta_sh_norm": _norm("sh"),
        "noop_mean": float(branch.noop.detach().float().mean().item()) if branch.noop.numel() else 0.0,
        "confidence_mean": (
            float(branch.confidence.detach().float().mean().item()) if branch.confidence.numel() else 0.0
        ),
    }


def _collect_delta_stats(delta: DeltaPack, aux: Dict[str, Any], *, k: int) -> Dict[str, float]:
    branches = [_branch_delta_stats(delta.bg)]
    branches.extend(_branch_delta_stats(x) for x in (delta.distant, delta.rigid) if x is not None)
    keys = {
        "delta_means_norm",
        "delta_scales_norm",
        "delta_opacity_norm",
        "delta_sh_norm",
        "noop_mean",
        "confidence_mean",
    }
    out = {"k": float(k)}
    for key in keys:
        out[key] = _mean([float(item[key]) for item in branches if key in item])
    for key, value in dict(aux or {}).items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[f"aux_{key}"] = float(value)
    return out


def _iter_local_tensors(local_state: Any) -> Iterable[torch.Tensor]:
    if local_state is None:
        return
    for branch_name in ("bg", "distant", "rigid"):
        branch = getattr(local_state, branch_name, None)
        if branch is None:
            continue
        for attr in ("means", "scales_log", "quats", "opacity_logit", "sh_dc", "sh_rest", "hidden"):
            value = getattr(branch, attr, None)
            if torch.is_tensor(value):
                yield value


def _collect_state_stats(local_state: Any, *, k: int) -> Dict[str, float]:
    scale_vals: List[torch.Tensor] = []
    opacity_vals: List[torch.Tensor] = []
    sh_vals: List[torch.Tensor] = []
    nan_inf = 0
    for branch_name in ("bg", "distant", "rigid"):
        branch = getattr(local_state, branch_name, None)
        if branch is None:
            continue
        scales = torch.exp(branch.scales_log.detach().float()).reshape(-1)
        opacity = torch.sigmoid(branch.opacity_logit.detach().float()).reshape(-1)
        scale_vals.append(scales)
        opacity_vals.append(opacity)
        sh_vals.append(branch.sh_dc.detach().float().reshape(-1))
        sh_vals.append(branch.sh_rest.detach().float().reshape(-1))
    for tensor in _iter_local_tensors(local_state):
        finite = torch.isfinite(tensor.detach())
        nan_inf += int((~finite).sum().item())

    def _q99(parts: List[torch.Tensor]) -> float:
        if not parts:
            return 0.0
        vals = torch.cat([x for x in parts if x.numel() > 0], dim=0)
        if vals.numel() == 0:
            return 0.0
        return float(torch.quantile(vals, 0.99).item())

    sh_energy = 0.0
    if sh_vals:
        sh_all = torch.cat([x for x in sh_vals if x.numel() > 0], dim=0)
        sh_energy = float(torch.mean(sh_all.pow(2)).item()) if sh_all.numel() else 0.0
    return {
        "k": float(k),
        "bg_scale_p99": _q99(scale_vals),
        "bg_opacity_p99": _q99(opacity_vals),
        "sh_energy": float(sh_energy),
        "num_nan_inf": float(nan_inf),
    }


def _save_png(path: str, image: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (torch.clamp(image.detach().cpu(), 0.0, 1.0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError:
        np.save(path.replace(".png", ".npy"), arr)
        return
    Image.fromarray(arr).save(path)


def _save_render_pair(
    *,
    save_dir: Optional[str],
    role: str,
    k: int,
    local_view_idx: int,
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> None:
    if not save_dir:
        return
    prefix = f"{role}_view{int(local_view_idx):02d}"
    _save_png(os.path.join(save_dir, f"{prefix}_pred_k{int(k)}.png"), pred)
    _save_png(os.path.join(save_dir, f"{prefix}_gt.png"), gt)


def _render_metrics_for_indices(
    model: Any,
    *,
    local_state: Any,
    batch: Dict[str, Any],
    target_indices: Sequence[int],
    mask_policy: str,
    min_valid_pixels: int,
    save_dir: Optional[str],
    save_role: str,
    save_k: Optional[int],
    max_saved_cams: int,
) -> Dict[str, float]:
    if len(target_indices) == 0:
        return {
            "num_refs": 0.0,
            "num_metric_refs": 0.0,
            "metric_valid": 0.0,
            "valid_ratio": 0.0,
            "skipped_no_valid_pixels": 0.0,
        }
    psnr_vals: List[float] = []
    l1_vals: List[float] = []
    ssim_vals: List[float] = []
    valid_ratios: List[float] = []
    skipped = 0.0
    saved = 0
    for local_i, idx in enumerate(target_indices):
        target = batch["targets"][int(idx)]
        pred, _alpha = model._render_target(local_state=local_state, target=target)
        gt = target["gt_image"].to(device=pred.device, dtype=pred.dtype)
        mask = target_valid_mask(target, mask_policy=str(mask_policy), device=pred.device)
        _loss, stats = masked_rgb_loss(
            pred,
            gt,
            mask=mask,
            l1_weight=1.0,
            ssim_weight=float(getattr(model, "loss_w_ssim", 0.0)),
            min_valid_pixels=int(min_valid_pixels),
        )
        if _safe_float(stats.get("skipped_no_valid_pixels", 0.0)) < 0.5:
            psnr_vals.append(_safe_float(stats.get("psnr", 0.0)))
            l1_vals.append(_safe_float(stats.get("l1", 0.0)))
            ssim_vals.append(_safe_float(stats.get("ssim", 0.0)))
        valid_ratios.append(_safe_float(stats.get("valid_ratio", 0.0)))
        skipped += _safe_float(stats.get("skipped_no_valid_pixels", 0.0))
        if save_dir and save_k is not None and saved < int(max_saved_cams):
            _save_render_pair(
                save_dir=save_dir,
                role=str(save_role),
                k=int(save_k),
                local_view_idx=int(local_i),
                pred=pred,
                gt=gt,
            )
            saved += 1
    out = {
        "num_refs": float(len(target_indices)),
        "num_metric_refs": float(len(psnr_vals)),
        "metric_valid": float(1.0 if psnr_vals else 0.0),
        "valid_ratio": _mean(valid_ratios),
        "skipped_no_valid_pixels": float(skipped),
    }
    if psnr_vals:
        out["psnr"] = _mean(psnr_vals)
        out["l1"] = _mean(l1_vals)
        out["ssim"] = _mean(ssim_vals)
    return out


def _indices_for_k_or_first_nonempty(groups: Sequence[Sequence[int]], k: int) -> Sequence[int]:
    if not groups:
        return []
    idx = int(k) - 1 if int(k) > 0 else 0
    idx = max(0, min(idx, len(groups) - 1))
    current = groups[idx]
    if len(current) > 0:
        return current
    for group in groups:
        if len(group) > 0:
            return group
    return current


def _batch_key(batch: Dict[str, Any]) -> tuple[int, int]:
    return (int(batch.get("scene_id", 0)), int(batch.get("segment_id", 0)))


def _drop_runtime_key(model: Any, key: tuple[int, int], had_key: Dict[str, bool]) -> None:
    for attr in (
        "node_states_bg",
        "node_states_distant",
        "node_states_rigid",
        "node_states_sky",
        "h_cache_bg",
        "h_cache_distant",
        "h_cache_rigid",
        "h_cache_sky",
    ):
        cache = getattr(model, attr, None)
        if isinstance(cache, dict) and not bool(had_key.get(attr, False)):
            cache.pop(key, None)


def _runtime_key_presence(model: Any, key: tuple[int, int]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for attr in (
        "node_states_bg",
        "node_states_distant",
        "node_states_rigid",
        "node_states_sky",
        "h_cache_bg",
        "h_cache_distant",
        "h_cache_rigid",
        "h_cache_sky",
    ):
        cache = getattr(model, attr, None)
        out[attr] = bool(isinstance(cache, dict) and key in cache)
    return out


def validate_v9_phase_a(
    model: Any,
    batch: Dict[str, Any],
    *,
    k_values: Sequence[int],
    max_K: int,
    mask_cfg: Optional[Dict[str, Any]] = None,
    compute_delta_stats: bool = True,
    compute_runtime_stats: bool = True,
    compute_memory_stats: bool = True,
    save_images: bool = False,
    save_dir: Optional[str] = None,
    save_image_k_values: Optional[Sequence[int]] = None,
    max_saved_cams: int = 1,
) -> Dict[str, Any]:
    roles = resolve_v9_phase_a_batch(batch)
    k_values_i = sorted(set(int(x) for x in list(k_values)))
    if 0 not in k_values_i:
        raise ValueError("validate_v9_phase_a requires k_values to include 0")
    if int(max_K) != int(max(k_values_i)):
        raise ValueError("validate_v9_phase_a requires max_K == max(k_values)")
    if int(roles.inner_K) != int(max_K):
        raise ValueError("validation batch inner_K must equal max_K.")
    mask_cfg = dict(mask_cfg or {})
    block_mask = str(mask_cfg.get("block_loss_mask", getattr(model, "stage6_block_mask_policy", "non_sky_non_egocar")))
    nearby_mask = str(mask_cfg.get("nearby_loss_mask", getattr(model, "stage6_nearby_mask_policy", "non_sky_non_egocar")))
    min_valid_pixels = int(mask_cfg.get("min_valid_pixels", 1))
    save_ks = set(int(x) for x in list(save_image_k_values or []))

    prev_training = bool(getattr(model, "training", False))
    key = _batch_key(batch)
    had_key = _runtime_key_presence(model, key)
    snap = model._snapshot_runtime_state(key) if hasattr(model, "_snapshot_runtime_state") else None
    row: Dict[str, Any] = {
        "scene_id": int(batch.get("scene_id", -1)),
        "segment_id": int(batch.get("segment_id", -1)),
        "block_idx": int((batch.get("request_meta") or {}).get("validation_block_idx", -1)),
        "source_frame_idx": int(roles.evidence_refs_by_step[0][0][0]),
    }
    timings = {
        "observe_ms": 0.0,
        "update_ms": 0.0,
        "render_metric_ms": 0.0,
    }
    if bool(compute_memory_stats) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_t0 = time.perf_counter() if bool(compute_runtime_stats) else 0.0
    local_state = None
    measurement = None
    delta = None
    update_aux = None
    node_state_bg = None
    node_state_rigid = None
    node_state_distant = None
    try:
        model.eval()
        with torch.no_grad():
            row["no_grad_enabled"] = float(not torch.is_grad_enabled())
            node_state_bg, node_state_rigid, node_state_distant = model._get_or_init_node_states_bg_rigid_distant(batch)
            local_state = LocalGSState.from_node_states(
                bg=node_state_bg,
                distant=node_state_distant,
                rigid=node_state_rigid,
                hidden_dim=int(getattr(model, "stage6_hidden_dim", 0)),
            )

            block_save_dir = save_dir if bool(save_images) else None
            if block_save_dir:
                os.makedirs(block_save_dir, exist_ok=True)
                with open(os.path.join(block_save_dir, "meta.json"), "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "scene_id": int(row["scene_id"]),
                            "segment_id": int(row["segment_id"]),
                            "block_idx": int(row["block_idx"]),
                            "source_frame_idx": int(row["source_frame_idx"]),
                            "k_values": [int(x) for x in k_values_i],
                        },
                        fh,
                        indent=2,
                    )

            def record_at(k: int) -> None:
                render_t0 = time.perf_counter() if bool(compute_runtime_stats) else 0.0
                save_this = block_save_dir if int(k) in save_ks else None
                block_stats = _render_metrics_for_indices(
                    model,
                    local_state=local_state,
                    batch=batch,
                    target_indices=roles.block_target_indices_by_step[int(k - 1)] if int(k) > 0 else roles.block_target_indices_by_step[0],
                    mask_policy=block_mask,
                    min_valid_pixels=int(min_valid_pixels),
                    save_dir=save_this,
                    save_role="block",
                    save_k=int(k),
                    max_saved_cams=int(max_saved_cams),
                )
                nearby_indices = _indices_for_k_or_first_nonempty(
                    roles.nearby_target_indices_by_step,
                    int(k),
                )
                nearby_stats = _render_metrics_for_indices(
                    model,
                    local_state=local_state,
                    batch=batch,
                    target_indices=nearby_indices,
                    mask_policy=nearby_mask,
                    min_valid_pixels=int(min_valid_pixels),
                    save_dir=save_this,
                    save_role="nearby",
                    save_k=int(k),
                    max_saved_cams=int(max_saved_cams),
                )
                if bool(compute_runtime_stats):
                    timings["render_metric_ms"] += float((time.perf_counter() - render_t0) * 1000.0)
                for prefix, stats in (("block", block_stats), ("nearby", nearby_stats)):
                    row[f"{prefix}_valid_ratio@{int(k)}"] = float(stats.get("valid_ratio", 0.0))
                    row[f"{prefix}_skipped_no_valid_pixels@{int(k)}"] = float(
                        stats.get("skipped_no_valid_pixels", 0.0)
                    )
                    row[f"{prefix}_metric_valid@{int(k)}"] = float(stats.get("metric_valid", 0.0))
                    row[f"{prefix}_num_metric_refs@{int(k)}"] = float(stats.get("num_metric_refs", 0.0))
                    row[f"val_v9/phaseA/{prefix}_valid_ratio@{int(k)}"] = row[f"{prefix}_valid_ratio@{int(k)}"]
                    row[f"val_v9/phaseA/{prefix}_skipped_no_valid_pixels@{int(k)}"] = row[
                        f"{prefix}_skipped_no_valid_pixels@{int(k)}"
                    ]
                    row[f"val_v9/phaseA/{prefix}_metric_valid@{int(k)}"] = row[
                        f"{prefix}_metric_valid@{int(k)}"
                    ]
                    row[f"val_v9/phaseA/{prefix}_num_metric_refs@{int(k)}"] = row[
                        f"{prefix}_num_metric_refs@{int(k)}"
                    ]
                    for metric_name in ("psnr", "l1", "ssim"):
                        value = stats.get(metric_name)
                        if value is None:
                            continue
                        value_f = float(value)
                        if not math.isfinite(value_f):
                            continue
                        row[f"{prefix}_{metric_name}@{int(k)}"] = value_f
                        row[f"val_v9/phaseA/{prefix}_{metric_name}@{int(k)}"] = value_f
                block_psnr = row.get(f"block_psnr@{int(k)}")
                nearby_psnr = row.get(f"nearby_psnr@{int(k)}")
                if block_psnr is not None and nearby_psnr is not None:
                    gap = float(block_psnr) - float(nearby_psnr)
                    row[f"generalization_gap@{int(k)}"] = gap
                    row[f"val_v9/phaseA/generalization_gap@{int(k)}"] = gap
                if bool(compute_delta_stats):
                    for name, value in _collect_state_stats(local_state, k=int(k)).items():
                        if name == "k":
                            continue
                        row[f"{name}@{int(k)}"] = float(value)
                        row[f"val_v9/phaseA/{name}@{int(k)}"] = float(value)

            record_at(0)
            delta_by_k: Dict[int, Dict[str, float]] = {}
            for k in range(1, int(max_K) + 1):
                evidence_refs = roles.evidence_refs_by_step[int(k) - 1]
                source_frame_idx = int(evidence_refs[0][0])
                observe_t0 = time.perf_counter() if bool(compute_runtime_stats) else 0.0
                measurement = model._observe_v4_measurement(
                    local_state=local_state,
                    batch=batch,
                    source_indices=roles.evidence_source_indices_by_step[int(k) - 1],
                    source_frame_idx=int(source_frame_idx),
                )
                if bool(compute_runtime_stats):
                    timings["observe_ms"] += float((time.perf_counter() - observe_t0) * 1000.0)
                update_t0 = time.perf_counter() if bool(compute_runtime_stats) else 0.0
                local_state, delta, update_aux = model._encode_and_update(
                    local_state=local_state,
                    measurement=measurement,
                )
                if bool(compute_runtime_stats):
                    timings["update_ms"] += float((time.perf_counter() - update_t0) * 1000.0)
                if bool(compute_delta_stats):
                    delta_by_k[int(k)] = _collect_delta_stats(delta, update_aux, k=int(k))
                if bool(compute_delta_stats) and int(k) in k_values_i:
                    for name, value in delta_by_k[int(k)].items():
                        if name == "k":
                            continue
                        row[f"{name}@{int(k)}"] = float(value)
                        row[f"val_v9/phaseA/{name}@{int(k)}"] = float(value)
                    record_at(int(k))

            for k in k_values_i:
                if int(k) == 0:
                    continue
                for prefix in ("block", "nearby"):
                    current = row.get(f"{prefix}_psnr@{int(k)}")
                    baseline = row.get(f"{prefix}_psnr@0")
                    if current is None or baseline is None:
                        continue
                    gain = float(current) - float(baseline)
                    row[f"{prefix}_psnr_gain@{int(k)}"] = float(gain)
                    row[f"val_v9/phaseA/{prefix}_psnr_gain@{int(k)}"] = float(gain)

            for prefix in ("block", "nearby"):
                vals = [
                    (int(k), float(row[f"{prefix}_psnr@{int(k)}"]))
                    for k in k_values_i
                    if f"{prefix}_psnr@{int(k)}" in row
                ]
                if not vals:
                    continue
                best_k, best_val = max(vals, key=lambda item: item[1])
                final_raw = row.get(f"{prefix}_psnr@{int(max_K)}")
                row[f"best_{prefix}_psnr"] = float(best_val)
                row[f"best_{prefix}_k"] = float(best_k)
                row[f"val_v9/phaseA/best_{prefix}_psnr"] = float(best_val)
                row[f"val_v9/phaseA/best_{prefix}_k"] = float(best_k)
                if final_raw is not None:
                    final_val = float(final_raw)
                    row[f"final_{prefix}_psnr_drop"] = float(best_val - final_val)
                    row[f"val_v9/phaseA/final_{prefix}_psnr_drop"] = float(best_val - final_val)
    finally:
        if snap is not None and hasattr(model, "_restore_runtime_state"):
            model._restore_runtime_state(key, snap)
        snap = None
        _drop_runtime_key(model, key, had_key)
        if prev_training:
            model.train()
        local_state = None
        measurement = None
        delta = None
        update_aux = None
        node_state_bg = None
        node_state_rigid = None
        node_state_distant = None

    if bool(compute_runtime_stats):
        total_ms = float((time.perf_counter() - total_t0) * 1000.0)
        row["time_total_ms"] = total_ms
        row["time_per_block_ms"] = total_ms
        row["time_observe_ms_per_iter"] = float(timings["observe_ms"] / max(int(max_K), 1))
        row["time_struct_event_ms_per_iter"] = float(timings["update_ms"] / max(int(max_K), 1))
        row["time_updater_ms_per_iter"] = float(timings["update_ms"] / max(int(max_K), 1))
        row["time_render_metric_ms"] = float(timings["render_metric_ms"])
        row["time_ms_per_iter"] = float((timings["observe_ms"] + timings["update_ms"]) / max(int(max_K), 1))
        row["val_v9/phaseA/time_total_ms"] = float(row["time_total_ms"])
        row["val_v9/phaseA/time_per_block_ms"] = float(row["time_per_block_ms"])
        row["val_v9/phaseA/time_observe_ms_per_iter"] = float(row["time_observe_ms_per_iter"])
        row["val_v9/phaseA/time_struct_event_ms_per_iter"] = float(row["time_struct_event_ms_per_iter"])
        row["val_v9/phaseA/time_updater_ms_per_iter"] = float(row["time_updater_ms_per_iter"])
        row["val_v9/phaseA/time_per_iter_ms"] = float(row["time_ms_per_iter"])
        row["val_v9/phaseA/time_render_metric_ms"] = float(row["time_render_metric_ms"])
    if bool(compute_memory_stats):
        if torch.cuda.is_available():
            row["cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            row["cuda_max_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
        else:
            row["cuda_max_allocated_mb"] = 0.0
            row["cuda_max_reserved_mb"] = 0.0
        row["val_v9/phaseA/cuda_max_allocated_mb"] = float(row["cuda_max_allocated_mb"])
        row["val_v9/phaseA/cuda_max_reserved_mb"] = float(row["cuda_max_reserved_mb"])
    return row


def aggregate_validation_v9_phase_a_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    k_values: Sequence[int],
) -> Dict[str, float]:
    if not rows:
        return {
            "val_v9/phaseA/num_blocks": 0.0,
            "val_v9/phaseA/num_scenes": 0.0,
        }
    out: Dict[str, float] = {
        "val_v9/phaseA/num_blocks": float(len(rows)),
        "val_v9/phaseA/num_scenes": float(len({int(r.get("scene_id", -1)) for r in rows})),
    }
    skip_prefixes = ("val_v9/",)
    skip_keys = {"scene_id", "segment_id", "block_idx", "source_frame_idx", "no_grad_enabled"}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in skip_keys and not key.startswith(skip_prefixes) and isinstance(value, (int, float))
        }
    )
    time_keys = {
        "time_total_ms",
        "time_per_block_ms",
        "time_observe_ms_per_iter",
        "time_struct_event_ms_per_iter",
        "time_updater_ms_per_iter",
        "time_render_metric_ms",
        "time_ms_per_iter",
    }
    time_key_map = {
        "time_per_block_ms": "time_per_block_ms",
        "time_observe_ms_per_iter": "time_observe_ms_per_iter",
        "time_struct_event_ms_per_iter": "time_struct_event_ms_per_iter",
        "time_updater_ms_per_iter": "time_updater_ms_per_iter",
        "time_render_metric_ms": "time_render_metric_ms",
        "time_ms_per_iter": "time_per_iter_ms",
    }
    memory_keys = {"cuda_max_allocated_mb", "cuda_max_reserved_mb"}
    for key in numeric_keys:
        values = [_safe_float(row.get(key, 0.0)) for row in rows if key in row]
        if not values:
            continue
        if key == "time_total_ms":
            out["val_v9/phaseA/time_total_ms"] = float(sum(values))
            continue
        if key in time_keys:
            out[f"val_v9/phaseA/{time_key_map[key]}"] = _mean(values)
            continue
        if key in memory_keys:
            out[f"val_v9/phaseA/{key}"] = float(max(values))
            continue
        out[f"val_v9/phaseA/mean_{key}"] = _mean(values)
        out[f"val_v9/phaseA/median_{key}"] = _percentile(values, 50.0)
        out[f"val_v9/phaseA/p10_{key}"] = _percentile(values, 10.0)
        out[f"val_v9/phaseA/p90_{key}"] = _percentile(values, 90.0)
    for k in [int(x) for x in k_values]:
        b = out.get(f"val_v9/phaseA/mean_block_psnr@{int(k)}")
        n = out.get(f"val_v9/phaseA/mean_nearby_psnr@{int(k)}")
        if b is not None and n is not None:
            out[f"val_v9/phaseA/mean_generalization_gap@{int(k)}"] = float(b - n)
    return out


__all__ = ["aggregate_validation_v9_phase_a_rows", "validate_v9_phase_a"]
