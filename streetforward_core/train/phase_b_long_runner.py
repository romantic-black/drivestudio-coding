from __future__ import annotations

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
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.runtime.train()
        self.recipe.train()
        if not hasattr(self.runtime, "optimizer") or self.runtime.optimizer is None:
            raise ValueError("Phase B Long runner requires runtime.optimizer.")
        self.runtime.optimizer.zero_grad(set_to_none=True)
        out: PhaseBLongForwardOutput = self.recipe(batch)
        legacy = out.to_legacy_dict()
        loss = out.loss
        loss.backward()
        grad_group_sums = self.runtime._stage6_assert_required_group_grads_phase_b_long(legacy)
        skip_optimizer = bool(float(grad_group_sums.get("phase_b_long/skipped_no_support_rollout", 0.0)) > 0.0)
        if skip_optimizer:
            grad_norm = loss.detach().new_tensor(0.0)
        else:
            grad_norm = self.runtime._stage6_compute_and_check_grad_norm()
            self.runtime.optimizer.step()
        did_reset_node_state = False
        if scheduler_node_sync is not None and bool(scheduler_node_sync.get("reset_after_block", False)):
            self.runtime.reset_node_state()
            did_reset_node_state = True
        self.runtime.optimizer.zero_grad(set_to_none=True)

        roles = legacy["roles"]
        stats = dict(legacy.get("stats") or {})
        logs: Dict[str, Any] = {
            "loss": float(loss.detach().item()),
            "phase_b_long/loss_total": float(loss.detach().item()),
            "stage6/phase": "6_0_phase_b",
            "stage6/inner_K": float(roles.inner_K),
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
            **{key: float(value) for key, value in stats.items() if isinstance(value, (int, float))},
            **grad_group_sums,
        }
        for item in list(legacy.get("per_step") or []):
            k = int(item.get("k", 0))
            for key, value in item.items():
                if key == "k" or not isinstance(value, (int, float)):
                    continue
                logs[f"phase_b_long/k{k}/{key}"] = float(value)
        if torch.cuda.is_available():
            logs["memory/allocated_gb"] = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
            logs["memory/reserved_gb"] = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
            logs["memory/peak_gb"] = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
        return logs

