from __future__ import annotations

import gc
import math
import os
from typing import Any, Dict, List, Optional

import torch

from streetforward_core.recipes.phase_a_recipe import PhaseAForwardOutput, PhaseARecipe


def _numeric_scalar(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def _add_phase_a_per_step_scalars(logs: Dict[str, Any], *, per_step: Any) -> None:
    stage6_values: Dict[str, List[float]] = {}
    rigid_source_counts: List[float] = []
    for item in list(per_step or []):
        if not isinstance(item, dict):
            continue
        k = int(item.get("k", 0))
        for key, value in item.items():
            if key == "k":
                continue
            value_f = _numeric_scalar(value)
            if value_f is None:
                continue
            logs.setdefault(f"phaseA/k{k}/{key}", value_f)
            logs.setdefault(f"phase_a/k{k}/{key}", value_f)
            if key.startswith("stage6/"):
                logs[f"{key}_k{k}"] = value_f
                stage6_values.setdefault(key, []).append(value_f)
        near_rigid = _numeric_scalar(item.get("stage6/struct/near_num_rigid_in")) or 0.0
        far_rigid = _numeric_scalar(item.get("stage6/struct/far_num_rigid_out")) or 0.0
        if "stage6/struct/near_num_rigid_in" in item or "stage6/struct/far_num_rigid_out" in item:
            rigid_source_counts.append(float(near_rigid + far_rigid))
    for key, values in stage6_values.items():
        if not values:
            continue
        logs[f"{key}_final"] = float(values[-1])
        logs[f"{key}_mean"] = float(sum(values) / len(values))
        logs[f"{key}_max"] = float(max(values))
    if rigid_source_counts:
        num_rigid = float(max(int(logs.get("num_gaussians_rigid", 0)), 1))
        logs["state/num_rigid_source_visible_final"] = int(round(rigid_source_counts[-1]))
        logs["state/num_rigid_source_visible_max"] = int(round(max(rigid_source_counts)))
        logs["state/rigid_source_visible_ratio_final"] = float(rigid_source_counts[-1] / num_rigid)
        logs["state/rigid_source_visible_ratio_max"] = float(max(rigid_source_counts) / num_rigid)


def _add_phase_a_aliases(logs: Dict[str, Any], *, final: Dict[str, float], per_step: Any) -> None:
    logs["phase_a/loss_total"] = float(logs.get("phaseA/loss_total", logs.get("loss", 0.0)))
    logs["phase_a/inner_K"] = float(logs.get("stage6/inner_K", 0.0))
    logs["phase_a/loss_block_final"] = float(final.get("loss_block", 0.0))
    logs["phase_a/loss_nearby_final"] = float(final.get("loss_nearby", 0.0))
    if "phaseA/grad_norm_total" in logs:
        logs["phase_a/grad_norm_total"] = float(logs["phaseA/grad_norm_total"])
    for prefix in ("block", "nearby"):
        for metric_name in ("psnr", "ssim", "l1"):
            old_key = f"phaseA/{prefix}_{metric_name}_final"
            if old_key in logs:
                logs[f"phase_a/{prefix}_{metric_name}_final"] = float(logs[old_key])
    for item in list(per_step or []):
        k = int(item["k"])
        logs[f"phase_a/k{k}/loss_block"] = float(item.get("loss_block", 0.0))
        logs[f"phase_a/k{k}/loss_nearby"] = float(item.get("loss_nearby", 0.0))
        logs[f"phase_a/k{k}/block_valid_ratio"] = float(item.get("block_valid_ratio", 0.0))
        logs[f"phase_a/k{k}/nearby_valid_ratio"] = float(item.get("nearby_valid_ratio", 0.0))
    grad_aliases = {
        "grad/stage6_struct_event_decoder_near_sum": "grad/struct_event_decoder_near",
        "grad/stage6_struct_event_decoder_far_sum": "grad/struct_event_decoder_far",
        "grad/stage6_param_obs_codec_sum": "grad/param_obs_codec",
        "grad/stage6_posterior_updater_sum": "grad/posterior_updater",
        "grad/stage6_measurement_frontend_sum": "grad/measurement_frontend",
    }
    for old_key, new_key in grad_aliases.items():
        if old_key in logs:
            logs[new_key] = float(logs[old_key])
    logs["state/num_bg"] = int(logs.get("num_gaussians_bg", 0))
    logs["state/num_distant"] = int(logs.get("num_gaussians_distant", 0))
    logs["state/num_rigid"] = int(logs.get("num_gaussians_rigid", 0))


class PhaseATrainRunner:
    def __init__(self, *, runtime: Any, recipe: PhaseARecipe):
        self.runtime = runtime
        self.recipe = recipe

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        batch = dict(batch)
        batch["global_step"] = int(step or 0)
        self.runtime.train()
        self.recipe.train()
        if not hasattr(self.runtime, "optimizer") or self.runtime.optimizer is None:
            raise ValueError("Phase A runner requires runtime.optimizer.")
        self.runtime.optimizer.zero_grad(set_to_none=True)
        out: PhaseAForwardOutput = self.recipe(batch)
        legacy_out = out.to_legacy_dict()
        loss = out.loss
        loss.backward()
        grad_group_sums = self.runtime._stage6_assert_required_group_grads(legacy_out)
        grad_norm = self.runtime._stage6_compute_and_check_grad_norm()
        self.runtime.optimizer.step()
        if str(getattr(self.runtime, "stage6_writeback_policy", "block_end_detached")) == "block_end_detached":
            out.local_state.writeback_detached(
                bg=out.node_state_bg,
                distant=out.node_state_distant,
                rigid=out.node_state_rigid,
            )
        elif str(getattr(self.runtime, "stage6_writeback_policy", "block_end_detached")) != "none":
            raise ValueError(f"unsupported Stage6 Phase A writeback_policy={self.runtime.stage6_writeback_policy!r}")

        did_reset_node_state = False
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.runtime.reset_node_state()
            did_reset_node_state = True
        self.runtime.optimizer.zero_grad(set_to_none=True)

        per_step = list(out.per_step or [])
        final = per_step[-1] if per_step else {}
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phaseA/loss_total": float(loss.detach().item()),
            "stage6/phase": "A",
            "stage6/inner_K": float(out.resolved.inner_K),
            "num_targets": int(out.num_targets),
            "num_source_views": int(out.num_source_views),
            "pred_rgbs": list(out.pred_rgbs),
            "gt_images": list(out.gt_images),
            "num_gaussians_bg": int(out.node_state_bg.means.shape[0]),
            "num_gaussians_distant": int(out.node_state_distant.means.shape[0]) if out.node_state_distant is not None else 0,
            "num_gaussians_rigid": int(out.node_state_rigid.means.shape[0]) if out.node_state_rigid is not None else 0,
            "phaseA/loss_block_final": float(final.get("loss_block", 0.0)),
            "phaseA/loss_nearby_final": float(final.get("loss_nearby", 0.0)),
            "mask/block_valid_ratio_final": float(final.get("block_valid_ratio", 0.0)),
            "mask/nearby_valid_ratio_final": float(final.get("nearby_valid_ratio", 0.0)),
            "mask/block_skipped_no_valid_pixels_final": float(final.get("block_skipped", 0.0)),
            "mask/nearby_skipped_no_valid_pixels_final": float(final.get("nearby_skipped", 0.0)),
            "mask/block_metric_valid_final": float(final.get("block_metric_valid", 0.0)),
            "mask/nearby_metric_valid_final": float(final.get("nearby_metric_valid", 0.0)),
            "mask/block_num_metric_refs_final": float(final.get("block_num_metric_refs", 0.0)),
            "mask/nearby_num_metric_refs_final": float(final.get("nearby_num_metric_refs", 0.0)),
            "phaseA/grad_norm_total": float(grad_norm.detach().item()),
            "node_state_sync_reset": bool(did_reset_node_state),
            "node_state_cache_segments_bg": int(len(getattr(self.runtime, "node_states_bg", {}))),
            "node_state_cache_segments_distant": int(len(getattr(self.runtime, "node_states_distant", {}))),
            "node_state_cache_segments_rigid": int(len(getattr(self.runtime, "node_states_rigid", {}))),
            **grad_group_sums,
        }
        for prefix in ("block", "nearby"):
            for metric_name in ("psnr", "ssim", "l1"):
                value = final.get(f"{prefix}_{metric_name}")
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    logs[f"phaseA/{prefix}_{metric_name}_final"] = value_f
        for item in per_step:
            k = int(item["k"])
            logs[f"phaseA/loss_block_k{k}"] = float(item.get("loss_block", 0.0))
            logs[f"phaseA/loss_nearby_k{k}"] = float(item.get("loss_nearby", 0.0))
            logs[f"mask/block_valid_ratio_k{k}"] = float(item.get("block_valid_ratio", 0.0))
            logs[f"mask/nearby_valid_ratio_k{k}"] = float(item.get("nearby_valid_ratio", 0.0))
            logs[f"mask/block_skipped_no_valid_pixels_k{k}"] = float(item.get("block_skipped", 0.0))
            logs[f"mask/nearby_skipped_no_valid_pixels_k{k}"] = float(item.get("nearby_skipped", 0.0))
            logs[f"mask/block_metric_valid_k{k}"] = float(item.get("block_metric_valid", 0.0))
            logs[f"mask/nearby_metric_valid_k{k}"] = float(item.get("nearby_metric_valid", 0.0))
            logs[f"mask/block_num_metric_refs_k{k}"] = float(item.get("block_num_metric_refs", 0.0))
            logs[f"mask/nearby_num_metric_refs_k{k}"] = float(item.get("nearby_num_metric_refs", 0.0))
            for prefix in ("block", "nearby"):
                for metric_name in ("psnr", "ssim", "l1"):
                    value = item.get(f"{prefix}_{metric_name}")
                    if value is None:
                        continue
                    value_f = float(value)
                    if math.isfinite(value_f):
                        logs[f"phaseA/{prefix}_{metric_name}_k{k}"] = value_f
        _add_phase_a_per_step_scalars(logs, per_step=per_step)
        _add_phase_a_aliases(logs, final=final, per_step=per_step)

        if did_reset_node_state:
            del out, legacy_out, loss
            empty_cache = str(os.environ.get("STAGE6_EMPTY_CACHE_ON_RESET", "")).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if empty_cache:
                gc.collect()
            if empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return logs
