from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from streetforward_core.recipes.phase_b_long_recipe import PhaseBLongForwardOutput, PhaseBLongRecipe


class PhaseBLongTrainRunner:
    def __init__(self, *, runtime: Any, recipe: PhaseBLongRecipe):
        self.runtime = runtime
        self.recipe = recipe

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _sync_cuda() -> None:
            if bool(sync_cuda_timing) and torch.cuda.is_available():
                torch.cuda.synchronize()

        self.runtime.train()
        self.recipe.train()
        if not hasattr(self.runtime, "optimizer") or self.runtime.optimizer is None:
            raise ValueError("Phase B Long runner requires runtime.optimizer.")
        self.runtime.optimizer.zero_grad(set_to_none=True)
        if bool(profile_phase_timing) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        _sync_cuda()
        t0 = time.perf_counter()
        out: PhaseBLongForwardOutput = self.recipe(batch)
        _sync_cuda()
        t1 = time.perf_counter()
        legacy = out.to_legacy_dict()
        loss = out.loss
        loss.backward()
        _sync_cuda()
        t2 = time.perf_counter()
        grad_group_sums = self.runtime._stage6_assert_required_group_grads_phase_b_long(legacy)
        skip_optimizer = bool(float(grad_group_sums.get("phase_b_long/skipped_no_support_rollout", 0.0)) > 0.0)
        if skip_optimizer:
            grad_norm = loss.detach().new_tensor(0.0)
        else:
            grad_norm = self.runtime._stage6_compute_and_check_grad_norm()
        _sync_cuda()
        t3 = time.perf_counter()
        if not skip_optimizer:
            self.runtime.optimizer.step()
        _sync_cuda()
        t4 = time.perf_counter()
        did_reset_node_state = False
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.runtime.reset_node_state()
            did_reset_node_state = True
        self.runtime.optimizer.zero_grad(set_to_none=True)

        roles = legacy["roles"]
        request_meta = dict(getattr(roles, "request_meta", {}) or {})
        stats = dict(legacy.get("stats") or {})
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phase_b_long/loss_total": float(loss.detach().item()),
            "stage6/phase": "6_0_phase_b",
            "stage6/inner_K": float(roles.inner_K),
            "train_iter": int(step if step is not None else batch.get("global_step", -1)),
            "rollout_step": int(request_meta.get("global_step", batch.get("global_step", -1))),
            "rollout_id": int(request_meta.get("rollout_id", -1)),
            "rollout_id_in_episode": int(request_meta.get("rollout_id_in_episode", -1)),
            "episode_window_id": int(request_meta.get("episode_window_id", request_meta.get("episode_id", -1))),
            "rollout_budget_per_episode": int(request_meta.get("rollout_budget_per_episode", -1)),
            "shape_name": str(request_meta.get("shape_name", getattr(roles, "shape_name", "")) or ""),
            "visit_count": int(getattr(roles, "inner_K", -1)),
            "num_targets": int(legacy.get("num_targets", 0)),
            "num_source_views": int(legacy.get("num_source_views", 0)),
            "pred_rgbs": list(legacy.get("pred_rgbs") or []),
            "gt_images": list(legacy.get("gt_images") or []),
            "num_gaussians_bg": int(legacy["node_state_bg"].means.shape[0]),
            "num_gaussians_distant": int(legacy["node_state_distant"].means.shape[0])
            if legacy["node_state_distant"] is not None
            else 0,
            "num_gaussians_rigid": int(legacy["node_state_rigid"].means.shape[0])
            if legacy["node_state_rigid"] is not None
            else 0,
            "phase_b_long/grad_norm_total": float(grad_norm.detach().item()),
            "node_state_sync_reset": bool(did_reset_node_state),
            "node_state_cache_segments_bg": int(len(getattr(self.runtime, "node_states_bg", {}))),
            "node_state_cache_segments_distant": int(len(getattr(self.runtime, "node_states_distant", {}))),
            "node_state_cache_segments_rigid": int(len(getattr(self.runtime, "node_states_rigid", {}))),
            "forward_ms": float((t1 - t0) * 1000.0),
            "backward_ms": float((t2 - t1) * 1000.0),
            "grad_check_ms": float((t3 - t2) * 1000.0),
            "optimizer_ms": float((t4 - t3) * 1000.0),
            **{key: float(value) for key, value in stats.items() if isinstance(value, (int, float))},
            **grad_group_sums,
        }
        per_step_items = list(legacy.get("per_step") or [])
        selected_k = {0}
        if per_step_items:
            selected_k.add(len(per_step_items) - 1)
        for item in per_step_items:
            k = int(item.get("k", 0))
            if k not in selected_k:
                continue
            for key, value in item.items():
                if key == "k" or not isinstance(value, (int, float)):
                    continue
                logs[f"phase_b_long/k{k}/{key}"] = float(value)
        if torch.cuda.is_available():
            logs["memory/allocated_gb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
            logs["memory/reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
            logs["memory/peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
            logs["memory/peak_reserved_gb"] = float(torch.cuda.max_memory_reserved() / (1024.0 ** 3))
        return logs
