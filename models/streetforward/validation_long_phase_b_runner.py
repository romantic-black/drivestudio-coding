from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import torch

from models.streetforward.metrics import compute_lpips
from models.streetforward.stage6_0.local_gs_state import LocalGSState
from models.streetforward.stage6_0.phase_a_losses import masked_rgb_loss, target_valid_mask
from models.streetforward.stage6_0.phase_b_long import (
    PhaseBOffsetState,
    materialize_phase_b_state,
)
from models.streetforward.stage6_0.phase_b_long.resolver import resolve_long_phase_b_batch
from models.streetforward.stage6_0.phase_b_long.types import LongVSMReadPack

DEFAULT_LONG_VSM_ABLATIONS = (
    "normal",
    "zero_vsm",
    "zero_read_keep_seen",
    "shuffle_vsm",
    "zero_delta",
)


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / max(len(vals), 1)) if vals else 0.0


def _zero_like_optional(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.detach() * 0.0


def _zero_read(read: LongVSMReadPack, *, zero_seen: bool = True) -> LongVSMReadPack:
    seen_bg = read.seen_bg.detach() * 0.0 if bool(zero_seen) else read.seen_bg
    return LongVSMReadPack(
        bg=read.bg.detach() * 0.0,
        seen_bg=seen_bg,
        bg_indices=read.bg_indices,
        rigid=_zero_like_optional(read.rigid),
        rigid_indices=read.rigid_indices,
        rigid_seen=_zero_like_optional(read.rigid_seen) if bool(zero_seen) else read.rigid_seen,
        rigid_stable_mask=read.rigid_stable_mask,
        distant=_zero_like_optional(read.distant),
        distant_indices=read.distant_indices,
        distant_seen=_zero_like_optional(read.distant_seen) if bool(zero_seen) else read.distant_seen,
    )


def _shuffle_read(read: LongVSMReadPack) -> LongVSMReadPack:
    if read.bg.shape[0] > 1:
        perm = torch.randperm(int(read.bg.shape[0]), device=read.bg.device)
        bg = read.bg.index_select(0, perm)
        seen_bg = read.seen_bg.index_select(0, perm.to(device=read.seen_bg.device))
    else:
        bg = read.bg
        seen_bg = read.seen_bg
    rigid = read.rigid
    rigid_seen = read.rigid_seen
    if rigid is not None and int(rigid.shape[0]) > 1:
        rperm = torch.randperm(int(rigid.shape[0]), device=rigid.device)
        rigid = rigid.index_select(0, rperm)
        if rigid_seen is not None:
            rigid_seen = rigid_seen.index_select(0, rperm.to(device=rigid_seen.device))
    distant = read.distant
    distant_seen = read.distant_seen
    if distant is not None and int(distant.shape[0]) > 1:
        dperm = torch.randperm(int(distant.shape[0]), device=distant.device)
        distant = distant.index_select(0, dperm)
        if distant_seen is not None:
            distant_seen = distant_seen.index_select(0, dperm.to(device=distant_seen.device))
    return replace(read, bg=bg, seen_bg=seen_bg, rigid=rigid, rigid_seen=rigid_seen, distant=distant, distant_seen=distant_seen)


def _render_metrics_for_indices(
    model: Any,
    *,
    base_state: LocalGSState,
    offset: PhaseBOffsetState,
    batch: Dict[str, Any],
    target_indices: Sequence[int],
    rigid_meta: Optional[Dict[str, Any]],
    mask_policy: str,
    min_valid_pixels: int,
    lpips_model: Optional[Any],
) -> tuple[Dict[str, float], Optional[Any]]:
    if len(target_indices) == 0:
        return {"num_refs": 0.0, "psnr": 0.0, "l1": 0.0, "ssim": 0.0, "lpips": 0.0}, lpips_model
    psnr_vals: List[float] = []
    l1_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    fallback_rows = 0.0
    for idx in target_indices:
        target = batch["targets"][int(idx)]
        frame_state = materialize_phase_b_state(
            base_state=base_state,
            offset=offset,
            target_frame_idx=int(target.get("frame_idx", 0)),
            rigid_meta=rigid_meta,
        )
        pred, _alpha = model._render_target(local_state=frame_state, target=target)
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
        psnr_vals.append(float(stats.get("psnr", 0.0)))
        l1_vals.append(float(stats.get("l1", 0.0)))
        ssim_vals.append(float(stats.get("ssim", 0.0)))
        lpips_val, lpips_model = compute_lpips(
            pred.detach().float().clamp(0.0, 1.0),
            gt.detach().float().clamp(0.0, 1.0),
            lpips_model=lpips_model,
            device=pred.device,
        )
        lpips_vals.append(float(lpips_val))
        fallback_rows += float(getattr(frame_state, "_phase_b_long_rigid_fallback_rows", 0.0))
    return {
        "num_refs": float(len(target_indices)),
        "psnr": _safe_mean(psnr_vals),
        "l1": _safe_mean(l1_vals),
        "ssim": _safe_mean(ssim_vals),
        "lpips": _safe_mean(lpips_vals),
        "rigid_fallback_rows": float(fallback_rows),
    }, lpips_model


def run_long_phase_b_inference(
    model: Any,
    batch: Dict[str, Any],
    *,
    ablation: str = "normal",
) -> Dict[str, Any]:
    roles = resolve_long_phase_b_batch(batch)
    node_state_bg, node_state_rigid, node_state_distant = model._get_or_init_node_states_bg_rigid_distant(batch)
    local_base = LocalGSState.from_node_states(
        bg=node_state_bg,
        distant=node_state_distant,
        rigid=node_state_rigid,
        hidden_dim=model.stage6_hidden_dim,
    )
    base_state = model._detach_local_state(local_base)
    offset_dtype = model._phase_b_long_state_dtype(base_state.bg.means, str(getattr(model, "stage6_phase_b_long_offset_dtype", "bf16")))
    vsm_dtype = model._phase_b_long_state_dtype(base_state.bg.means, str(getattr(model, "stage6_phase_b_long_vsm_dtype", "bf16")))
    offset = PhaseBOffsetState.zeros_like(base_state=base_state, dtype=offset_dtype)
    init_kwargs: Dict[str, Any] = {
        "base_state": base_state,
        "dtype": vsm_dtype,
        "rigid_meta": roles.rigid_meta,
        "distant_mode": str(getattr(model, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
        "episode_id": int(roles.request_meta.get("episode_id", -1) or -1),
    }
    if str(getattr(model, "stage6_phase_b_long_vsm_type", "streaming_selective_ssm")) == "cell_streaming_selective_ssm":
        init_kwargs["batch"] = batch
    vsm_state = model.stage6_long_vsm.init_state(**init_kwargs)
    with torch.no_grad():
        for k, visit in enumerate(roles.visits):
            frame_idx = int(visit.frame_idx)
            sensor_state = materialize_phase_b_state(
                base_state=base_state,
                offset=offset.detach_for_sensor(),
                target_frame_idx=int(frame_idx),
                rigid_meta=roles.rigid_meta,
            )
            if hasattr(model, "_phase_b_long_clamp_sensor_state_to_aabb"):
                sensor_state = model._phase_b_long_clamp_sensor_state_to_aabb(sensor_state)
            measurement = model._observe_v4_measurement(
                local_state=sensor_state,
                batch=batch,
                source_indices=roles.evidence_source_indices_by_step[int(k)],
                source_frame_idx=int(frame_idx),
            )
            event = model._build_stage6_event_from_measurement(local_state=sensor_state, measurement=measurement)
            event = model._detach_event_pack(model._event_with_default_view_code(event))
            vsm_compute_dtype = (
                model._phase_b_long_vsm_compute_dtype(event.event_bg)
                if hasattr(model, "_phase_b_long_vsm_compute_dtype")
                else None
            )
            autocast_ctx = (
                model._phase_b_long_autocast_context(event.event_bg)
                if hasattr(model, "_phase_b_long_autocast_context")
                else nullcontext()
            )
            with autocast_ctx:
                vsm_state, read_pack, _aux = model.stage6_long_vsm.write_read(
                    state=vsm_state,
                    event=event,
                    step_idx=int(k),
                    frame_idx=int(frame_idx),
                    repeat_idx=int(visit.repeat_idx),
                    rigid_meta=roles.rigid_meta,
                    distant_mode=str(getattr(model, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                    visit_time_code=torch.tensor(
                        roles.visit_time_codes[int(k)],
                        device=event.event_bg.device,
                        dtype=vsm_compute_dtype or event.event_bg.dtype,
                    ),
                    compute_dtype=vsm_compute_dtype,
                )
                if str(ablation) in {"zero_vsm", "zero_read_zero_seen"}:
                    read_pack = _zero_read(read_pack, zero_seen=True)
                elif str(ablation) in {"zero_read_keep_seen", "seen_only"}:
                    read_pack = _zero_read(read_pack, zero_seen=False)
                elif str(ablation) in {"shuffle_vsm", "shuffle_read"}:
                    read_pack = _shuffle_read(read_pack)
                elif str(ablation) == "zero_delta":
                    continue
                elif str(ablation) != "normal":
                    raise ValueError(f"unsupported Long validation ablation={ablation!r}")
                delta = model.stage6_long_offset_decoder(
                    read=read_pack,
                    distant_mode=str(getattr(model, "stage6_phase_b_long_distant_mode", "frozen_render_only")),
                )
            offset = offset.apply(delta, frame_idx=int(frame_idx), rigid_meta=roles.rigid_meta)
    return {"roles": roles, "base_state": base_state, "offset": offset}


def validate_long_phase_b(
    model: Any,
    batch: Dict[str, Any],
    *,
    mask_policy: str = "non_sky_non_egocar",
    min_valid_pixels: int = 1,
    ablations: Sequence[str] = DEFAULT_LONG_VSM_ABLATIONS,
) -> Dict[str, float]:
    meta = dict(batch.get("request_meta") or {})
    buckets = dict(meta.get("validation_buckets") or {})
    target_refs = [tuple(x) for x in list(meta.get("target_image_refs") or [])]
    index_by_ref = {tuple(ref): idx for idx, ref in enumerate(target_refs)}
    out: Dict[str, float] = {
        "val_long/interval_T": float(meta.get("validation_interval_T", 0) or 0),
        "val_long/distant_mode_frozen_render_only": (
            1.0 if str(getattr(model, "stage6_phase_b_long_distant_mode", "frozen_render_only")) == "frozen_render_only" else 0.0
        ),
        "val_long/distant_mode_appearance_scale_only": (
            1.0 if str(getattr(model, "stage6_phase_b_long_distant_mode", "frozen_render_only")) == "appearance_scale_only" else 0.0
        ),
    }
    for group_name, metric_prefix in (
        ("validation_bucket_requested_counts", "requested"),
        ("validation_bucket_materialized_counts", "materialized"),
        ("validation_bucket_dropped_counts", "dropped"),
    ):
        counts = dict(meta.get(group_name) or {})
        for bucket_name, value in counts.items():
            out[f"val_long/{metric_prefix}_{bucket_name}_refs"] = float(value)
    normal_psnr: Dict[str, float] = {}
    lpips_model = getattr(model, "_phase_b_long_lpips_model", None)
    for ablation in ablations:
        inference = run_long_phase_b_inference(model, batch, ablation=str(ablation))
        prefix = "val_long" if str(ablation) == "normal" else f"val_long/{ablation}"
        for bucket_name in ("reconstruction", "nvs_same_frame", "temporal_nvs", "segment_all"):
            refs = [tuple(x) for x in list(buckets.get(bucket_name, []) or [])]
            indices = [int(index_by_ref[ref]) for ref in refs if ref in index_by_ref]
            metrics, lpips_model = _render_metrics_for_indices(
                model,
                base_state=inference["base_state"],
                offset=inference["offset"],
                batch=batch,
                target_indices=indices,
                rigid_meta=inference["roles"].rigid_meta,
                mask_policy=str(mask_policy),
                min_valid_pixels=int(min_valid_pixels),
                lpips_model=lpips_model,
            )
            for key, value in metrics.items():
                out[f"{prefix}/{bucket_name}_{key}"] = float(value)
            if str(ablation) == "normal":
                normal_psnr[bucket_name] = float(metrics.get("psnr", 0.0))
            else:
                out[f"val_long/{ablation}_gain/{bucket_name}_psnr"] = float(normal_psnr.get(bucket_name, 0.0)) - float(
                    metrics.get("psnr", 0.0)
                )
        roles = inference["roles"]
        out[f"{prefix}/rigid_snapshot_frames"] = float(len(inference["offset"].rigid_frame_snapshots))
        out[f"{prefix}/rigid_stable_rows"] = float(sum(1 for _ in roles.step_chronological_ranks)) * 0.0
    try:
        setattr(model, "_phase_b_long_lpips_model", lpips_model)
    except Exception:
        pass
    return out


__all__ = ["run_long_phase_b_inference", "validate_long_phase_b"]
