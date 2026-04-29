from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional

import torch

from models.streetforward.minimal_trainer_stage5_3 import MinimalStreetForwardStage5_3
from models.streetforward.training_optim import (
    build_streetforward_lr_scheduler,
    build_streetforward_optimizer,
    optimizer_group_signature,
)

logger = logging.getLogger(__name__)


def _to_plain_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return {k: _to_plain_dict(v) if isinstance(v, dict) else v for k, v in node.items()}
    if hasattr(node, "keys"):
        out: Dict[str, Any] = {}
        for k in node.keys():
            v = node[k]
            if isinstance(v, dict) or hasattr(v, "keys"):
                out[str(k)] = _to_plain_dict(v)
            elif isinstance(v, (list, tuple)):
                out[str(k)] = [x for x in v]
            else:
                out[str(k)] = v
        return out
    return {}


class _ProductionOptimizerAdapter:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        grad_clip_norm: float,
        grad_clip_norm_type: float,
        fail_on_grad_norm_gt: float,
        scheduler: Optional[Any],
    ):
        self._optimizer = optimizer
        self._grad_clip_norm = float(grad_clip_norm)
        self._grad_clip_norm_type = float(grad_clip_norm_type)
        self._fail_on_grad_norm_gt = float(fail_on_grad_norm_gt)
        self._scheduler = scheduler
        self.global_step = 0
        self.last_grad_norm = 0.0

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state = self._optimizer.state_dict()
        state["_sf_global_step"] = int(self.global_step)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        state = dict(state_dict)
        self.global_step = int(state.pop("_sf_global_step", 0))
        return self._optimizer.load_state_dict(state)

    def step(self, *args, **kwargs):
        params = []
        for group in self._optimizer.param_groups:
            params.extend(group["params"])
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=float(self._grad_clip_norm),
            norm_type=float(self._grad_clip_norm_type),
            error_if_nonfinite=True,
        )
        self.last_grad_norm = float(total_grad_norm.item() if torch.is_tensor(total_grad_norm) else total_grad_norm)
        if self.last_grad_norm > self._fail_on_grad_norm_gt:
            raise FloatingPointError(
                f"grad_norm overflow: {self.last_grad_norm:.6f} > fail_on_grad_norm_gt={self._fail_on_grad_norm_gt:.6f}"
            )
        out = self._optimizer.step(*args, **kwargs)
        self.global_step += 1
        if self._scheduler is not None:
            self._scheduler.step()
        return out


class MinimalStreetForwardStage5_3_Production(MinimalStreetForwardStage5_3):
    def __init__(self, config, device: torch.device, **kwargs):
        self._validate_production_config(config)
        super().__init__(config=config, device=device, **kwargs)
        self._log_optimizer_groups_once()

    def _validate_production_config(self, config) -> None:
        model_cfg = self._require_key(config, "model", "config")
        if bool(self._require_key(model_cfg, "production_training", "model")) is not True:
            raise ValueError("Stage5_3_Production requires model.production_training=true.")
        recipe_cfg = self._require_key(config, "training_recipe", "config")
        recipe_name = str(self._require_key(recipe_cfg, "name", "training_recipe")).strip()
        if not recipe_name:
            raise ValueError("Stage5_3_Production requires non-empty training_recipe.name.")
        optimizer_cfg = self._require_key(config, "optimizer", "config")
        optimizer_type = str(self._require_key(optimizer_cfg, "type", "optimizer")).strip().lower()
        if optimizer_type != "adamw":
            raise ValueError("Stage5_3_Production requires optimizer.type=adamw.")
        groups_cfg = self._require_key(optimizer_cfg, "groups", "optimizer")
        if getattr(groups_cfg, "get", None) is None or groups_cfg.get("default") is None:
            raise ValueError("Stage5_3_Production requires optimizer.groups.default.")
        lr_cfg = self._require_key(config, "lr_scheduler", "config")
        if bool(self._require_key(lr_cfg, "enable", "lr_scheduler")) is not True:
            raise ValueError("Stage5_3_Production requires lr_scheduler.enable=true.")
        lr_type = str(self._require_key(lr_cfg, "type", "lr_scheduler")).strip().lower()
        if lr_type != "cosine":
            raise ValueError("Stage5_3_Production requires lr_scheduler.type=cosine.")
        _ = int(self._require_key(lr_cfg, "total_steps", "lr_scheduler"))
        _ = float(self._require_key(lr_cfg, "min_lr_ratio", "lr_scheduler"))
        train_cfg = self._require_key(config, "training", "config")
        amp_cfg = train_cfg.get("amp") if hasattr(train_cfg, "get") else None
        if amp_cfg is not None and bool(amp_cfg.get("enable", False)):
            raise ValueError("Stage5_3_Production requires training.amp.enable=false for this recipe.")
        bad_step_cfg = train_cfg.get("bad_step") if hasattr(train_cfg, "get") else None
        if bad_step_cfg is not None and bool(bad_step_cfg.get("fail_on_amp_overflow", False)):
            raise ValueError(
                "Stage5_3_Production requires training.bad_step.fail_on_amp_overflow=false when AMP is disabled."
            )
        _ = self._parse_production_runtime_cfg(train_cfg)

    def _parse_production_runtime_cfg(self, train_cfg: Any) -> Dict[str, float]:
        grad_clip_cfg = train_cfg.get("grad_clip") if hasattr(train_cfg, "get") else None
        if grad_clip_cfg is not None:
            if bool(grad_clip_cfg.get("enable", True)) is not True:
                raise ValueError("Stage5_3_Production requires training.grad_clip.enable=true.")
            grad_clip_norm = float(self._require_key(grad_clip_cfg, "max_norm", "training.grad_clip"))
            grad_clip_norm_type = float(grad_clip_cfg.get("norm_type", 2.0))
        else:
            # Backward-compatible fallback for earlier recipe drafts.
            grad_clip_norm = float(self._require_key(train_cfg, "grad_clip_norm", "training"))
            grad_clip_norm_type = float(train_cfg.get("grad_clip_norm_type", 2.0))

        bad_step_cfg = train_cfg.get("bad_step") if hasattr(train_cfg, "get") else None
        if bad_step_cfg is not None:
            fail_on_grad_norm_gt = float(self._require_key(bad_step_cfg, "fail_on_grad_norm_gt", "training.bad_step"))
        else:
            # Backward-compatible fallback for earlier recipe drafts.
            fail_on_grad_norm_gt = float(self._require_key(train_cfg, "fail_on_grad_norm_gt", "training"))

        if (not math.isfinite(grad_clip_norm)) or grad_clip_norm <= 0.0:
            raise ValueError(f"training.grad_clip.max_norm must be finite and > 0, got {grad_clip_norm}.")
        if (not math.isfinite(grad_clip_norm_type)) or grad_clip_norm_type <= 0.0:
            raise ValueError(f"training.grad_clip.norm_type must be finite and > 0, got {grad_clip_norm_type}.")
        if (not math.isfinite(fail_on_grad_norm_gt)) or fail_on_grad_norm_gt <= 0.0:
            raise ValueError(
                f"training.bad_step.fail_on_grad_norm_gt must be finite and > 0, got {fail_on_grad_norm_gt}."
            )
        return {
            "grad_clip_norm": float(grad_clip_norm),
            "grad_clip_norm_type": float(grad_clip_norm_type),
            "fail_on_grad_norm_gt": float(fail_on_grad_norm_gt),
        }

    def _log_resume_restore_flags(self, *, optimizer_restored: bool, lr_step_restored: bool) -> None:
        logger.info("resume/semantics=warm_runtime_no_scheduler_state")
        logger.info("resume/optimizer_restored=%d", int(optimizer_restored))
        logger.info("resume/lr_step_restored=%d", int(lr_step_restored))
        logger.info("resume/scheduler_runtime_restored=0")
        logger.info("resume/history_runtime_restored=0")
        logger.info("resume/node_state_runtime_restored=0")

    def _rebuild_optimizer_after_stage5_modules(self) -> None:
        train_cfg = self._require_key(self.config, "training", "config")
        runtime_cfg = self._parse_production_runtime_cfg(train_cfg)
        base_optimizer = build_streetforward_optimizer(self, self.config, strict=True)
        self.lr_scheduler = build_streetforward_lr_scheduler(
            optimizer=base_optimizer,
            config=self.config,
            start_step=0,
            strict=True,
        )
        self.optimizer = _ProductionOptimizerAdapter(
            optimizer=base_optimizer,
            grad_clip_norm=float(runtime_cfg["grad_clip_norm"]),
            grad_clip_norm_type=float(runtime_cfg["grad_clip_norm_type"]),
            fail_on_grad_norm_gt=float(runtime_cfg["fail_on_grad_norm_gt"]),
            scheduler=self.lr_scheduler,
        )

    def _log_optimizer_groups_once(self) -> None:
        base_optimizer = self.optimizer._optimizer
        meta = getattr(base_optimizer, "_streetforward_meta", {})
        logical_counts = dict(meta.get("logical_group_counts", {}))
        for key in (
            "default",
            "residual_unet",
            "fusion_neck",
            "struct_near_xcpe",
            "struct_far_mlp",
            "gate_history",
            "recurrent_update",
        ):
            logger.info("optimizer/group/%s/num_params=%s", key, int(logical_counts.get(key, 0)))
        logger.info("optimizer/frozen/dino/num_params=%s", int(meta.get("frozen_dino_params", 0)))
        logger.info("optimizer/unassigned_trainable_params=%s", int(meta.get("unassigned_trainable_params", 0)))
        logger.info("optimizer/group/signature=%s", optimizer_group_signature(base_optimizer))

    def forward(self, batch: Dict) -> Dict[str, Any]:
        out = super().forward(batch)
        loss = out.get("loss")
        if torch.is_tensor(loss):
            loss_finite = torch.isfinite(loss.detach())
            if not bool(loss_finite.all()):
                raise FloatingPointError("nonfinite loss detected before backward.")
        return out

    def train_step(
        self,
        batch: Dict,
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = super().train_step(
            batch=batch,
            step=step,
            profile_phase_timing=profile_phase_timing,
            sync_cuda_timing=sync_cuda_timing,
            scheduler_node_sync=scheduler_node_sync,
        )
        out["optimizer/global_step"] = float(self.optimizer.global_step)
        out["optimizer/grad_norm_total"] = float(self.optimizer.last_grad_norm)
        for group in self.optimizer.param_groups:
            logical_name = str(group.get("logical_name", ""))
            if logical_name == "default":
                out["lr/default"] = float(group["lr"])
            elif logical_name == "struct_near_xcpe":
                out["lr/struct_near_xcpe"] = float(group["lr"])
            elif logical_name == "fusion_neck":
                out["lr/fusion_neck"] = float(group["lr"])
            elif logical_name == "gate_history":
                out["lr/gate_history"] = float(group["lr"])
        return out

    def build_light_checkpoint_extra(self, *, step: int) -> Dict[str, Any]:
        scheduler_cfg = self.config.get("lr_scheduler", {})
        optimizer_cfg = self.config.get("optimizer", {})
        return {
            "format": "streetforward_light_ckpt_v2",
            "resume_semantics": "warm_runtime_no_scheduler_state",
            "restore_train_scheduler_runtime": False,
            "restore_history_runtime": False,
            "restore_node_state_runtime": False,
            "model_stage": str(self.config.model.stage),
            "global_step": int(step),
            "optimizer_signature": optimizer_group_signature(self.optimizer._optimizer),
            "optimizer_cfg": _to_plain_dict(optimizer_cfg),
            "lr_scheduler_cfg": _to_plain_dict(scheduler_cfg),
            "lr_scheduler": {
                "type": str(scheduler_cfg.get("type", "")),
                "global_step": int(self.optimizer.global_step),
                "cfg": _to_plain_dict(scheduler_cfg),
            },
        }

    def load_optimizer_state_from_checkpoint(self, payload: Dict[str, Any]) -> bool:
        optimizer_restored = False
        lr_step_restored = False
        old_sig = payload.get("optimizer_signature")
        cur_sig = optimizer_group_signature(self.optimizer._optimizer)
        if old_sig != cur_sig:
            logger.warning("Skip optimizer load: signature mismatch.")
            self._log_resume_restore_flags(optimizer_restored=optimizer_restored, lr_step_restored=lr_step_restored)
            return False
        if "optimizer_state_dict" not in payload:
            logger.warning("Skip optimizer load: checkpoint has no optimizer_state_dict.")
            self._log_resume_restore_flags(optimizer_restored=optimizer_restored, lr_step_restored=lr_step_restored)
            return False
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_restored = True
        if getattr(self, "lr_scheduler", None) is not None:
            lr_info = payload.get("lr_scheduler", {})
            start_step = int(lr_info.get("global_step", payload.get("global_step", self.optimizer.global_step)))
            self.lr_scheduler.set_step(start_step)
            lr_step_restored = True
        self._log_resume_restore_flags(optimizer_restored=optimizer_restored, lr_step_restored=lr_step_restored)
        return True


__all__ = ["MinimalStreetForwardStage5_3_Production"]
