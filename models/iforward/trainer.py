from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .model import IForwardModel, IForwardRolloutOutput
from .state import IForwardState
from .utils import cfg_get


class IForwardTrainer(nn.Module):
    def __init__(
        self,
        config: Any,
        device: torch.device,
        *,
        model: Optional[IForwardModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.model = model if model is not None else IForwardModel(config=config, device=device)
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer(config)
        self._state_cache: Dict[Tuple[int, int, int], IForwardState] = {}

    def _build_optimizer(self, config: Any) -> torch.optim.Optimizer:
        opt_cfg = cfg_get(config, "optimizer", {}) or {}
        lr_cfg = cfg_get(opt_cfg, "lr", 1.0e-4)
        lr = float(cfg_get(lr_cfg, "default", 1.0e-4) if hasattr(lr_cfg, "get") or isinstance(lr_cfg, dict) else lr_cfg)
        weight_decay = float(cfg_get(opt_cfg, "weight_decay", 0.0))
        betas = tuple(float(x) for x in list(cfg_get(opt_cfg, "betas", [0.9, 0.95]) or [0.9, 0.95]))
        eps = float(cfg_get(opt_cfg, "eps", 1.0e-8))
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("IForwardTrainer has no trainable parameters.")
        opt_type = str(cfg_get(opt_cfg, "type", "adamw")).lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if opt_type == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        raise ValueError(f"IForward unsupported optimizer.type={opt_type!r}")

    def forward_rollout(self, *args: Any, **kwargs: Any) -> IForwardRolloutOutput:
        return self.model.forward_rollout(*args, **kwargs)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.forward(batch)

    @staticmethod
    def _grad_norm(parameters: Any) -> torch.Tensor:
        total = None
        ref = None
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            ref = grad
            value = grad.pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            if ref is not None:
                return ref.new_tensor(0.0)
            return torch.tensor(0.0)
        return torch.sqrt(total.clamp_min(0.0))

    def _cache_key_from_output(self, out: IForwardRolloutOutput) -> Tuple[int, int, int]:
        return tuple(out.resolved.cache_key)

    @staticmethod
    def _sync_cuda(enabled: bool) -> None:
        if bool(enabled) and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _reset_bridge_runtime_node_state(self) -> Dict[str, int]:
        bridge = getattr(self.model, "bridge", None)
        reset = getattr(bridge, "reset_runtime_node_state", None)
        if not callable(reset):
            return {}
        return dict(reset())

    def train_step(
        self,
        batch: Dict[str, Any],
        step: Optional[int] = None,
        profile_phase_timing: bool = False,
        sync_cuda_timing: bool = False,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        runtime_policy: Optional[Any] = None,
        ablation: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = (scheduler_node_sync, runtime_policy)
        profile_cuda = bool(profile_phase_timing or sync_cuda_timing)
        timings: Dict[str, float] = {}
        batch = dict(batch)
        batch["global_step"] = int(step or 0)
        t0 = time.perf_counter()
        resolved = self.model.resolver.resolve(batch)
        timings["resolve_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        key = tuple(resolved.cache_key)
        runtime_reset_before: Dict[str, int] = {}
        if bool(resolved.reset_scene_state_before_rollout):
            self._state_cache.pop(key, None)
            runtime_reset_before = self._reset_bridge_runtime_node_state()
        carried = self._state_cache.get(key)
        timings["state_cache_ms"] = (time.perf_counter() - t0) * 1000.0

        self.train(True)
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        timings["optimizer_ms"] = (time.perf_counter() - t0) * 1000.0
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        out = self.model.forward_rollout(batch, carried_state=carried, ablation=ablation)
        self._sync_cuda(profile_cuda)
        timings["forward_ms"] = (time.perf_counter() - t0) * 1000.0
        loss = out.loss
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        loss.backward()
        self._sync_cuda(profile_cuda)
        timings["backward_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        params_with_grad = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm_unclipped = self._grad_norm(params_with_grad).to(device=loss.device)
        if not torch.isfinite(grad_norm_unclipped).all():
            raise RuntimeError("IForward gradient norm became NaN/Inf.")
        grad_clip_cfg = cfg_get(cfg_get(self.config, "training", {}) or {}, "grad_clip", {}) or {}
        grad_clip_enable = bool(cfg_get(grad_clip_cfg, "enable", False))
        grad_clip_max_norm = float(cfg_get(grad_clip_cfg, "max_norm", 1.0))
        grad_clip_applied = False
        if grad_clip_enable and params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=float(grad_clip_max_norm))
            grad_clip_applied = True
        grad_norm_after_clip = self._grad_norm(params_with_grad).to(device=loss.device)
        if not torch.isfinite(grad_norm_after_clip).all():
            raise RuntimeError("IForward clipped gradient norm became NaN/Inf.")
        timings["grad_norm_ms"] = (time.perf_counter() - t0) * 1000.0
        self._sync_cuda(profile_cuda)
        t0 = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._sync_cuda(profile_cuda)
        timings["optimizer_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        runtime_reset_after: Dict[str, int] = {}
        if bool(out.resolved.carry_scene_state_after_rollout) and not bool(out.resolved.episode_end_after_rollout):
            self._state_cache[key] = out.next_state.detach_for_next_rollout()
        else:
            self._state_cache.pop(key, None)
            runtime_reset_after = self._reset_bridge_runtime_node_state()
        timings["state_cache_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        losses = {name: float(value.detach().item()) for name, value in out.losses.items()}
        final = {
            "loss": float(loss.detach().item()),
            "iforward/loss_total": float(loss.detach().item()),
            "iforward/inner_K": float(out.resolved.inner_K),
            "iforward/rollout_id_global": float(out.resolved.rollout_id_global),
            "iforward/rollout_idx_in_episode": float(out.resolved.rollout_idx_in_episode),
            "iforward/episode_end_after_rollout": bool(out.resolved.episode_end_after_rollout),
            "iforward/carry_scene_state_after_rollout": bool(out.resolved.carry_scene_state_after_rollout),
            "iforward/state_cache_size": int(len(self._state_cache)),
            "iforward/grad_norm_total": float(grad_norm_after_clip.detach().item()),
            "iforward/grad_norm_unclipped": float(grad_norm_unclipped.detach().item()),
            "iforward/grad_norm_after_clip": float(grad_norm_after_clip.detach().item()),
            "iforward/grad_clip_max_norm": float(grad_clip_max_norm),
            "iforward/grad_clip_applied": bool(grad_clip_applied),
            "iforward/runtime_node_state_reset_before": bool(runtime_reset_before),
            "iforward/runtime_node_state_reset_after": bool(runtime_reset_after),
            "num_targets": int(out.stats.get("num_targets", 0)),
            "num_source_views": int(out.stats.get("num_source_views", 0)),
            "num_gaussians_bg": int(out.stats.get("num_gaussians_bg", 0)),
            "num_gaussians_distant": int(out.stats.get("num_gaussians_distant", 0)),
            "num_gaussians_rigid": int(out.stats.get("num_gaussians_rigid", 0)),
            "num_gaussians_sky": int(out.stats.get("num_gaussians_sky", 0)),
            "pred_rgbs": [x.detach().float().cpu() for x in out.pred_rgbs],
            "gt_images": [x.detach().float().cpu() for x in out.gt_images],
            "image_refs": [tuple(int(v) for v in ref) for ref in out.image_refs],
            "image_roles": [str(role) for role in out.image_roles],
        }
        for name, value in timings.items():
            final[name] = float(value)
        for prefix, values in (
            ("iforward/runtime_node_state_reset_before", runtime_reset_before),
            ("iforward/runtime_node_state_reset_after", runtime_reset_after),
        ):
            for name, value in values.items():
                final[f"{prefix}/{name}"] = int(value)
        for name, value in losses.items():
            final[f"iforward/loss_{name}"] = float(value)
        for name, value in out.stats.items():
            if isinstance(value, bool):
                final[f"iforward/{name}"] = bool(value)
            elif isinstance(value, int):
                final[f"iforward/{name}"] = int(value)
            elif isinstance(value, float) and math.isfinite(float(value)):
                final[f"iforward/{name}"] = float(value)
            elif isinstance(value, str):
                final[f"iforward/{name}"] = value
        memory_tokens = out.stats.get("memory_tokens")
        if isinstance(memory_tokens, dict):
            for name, value in memory_tokens.items():
                if isinstance(value, bool):
                    final[f"iforward/memory_tokens/{name}"] = bool(value)
                elif isinstance(value, int):
                    final[f"iforward/memory_tokens/{name}"] = int(value)
                elif isinstance(value, float) and math.isfinite(float(value)):
                    final[f"iforward/memory_tokens/{name}"] = float(value)
        for item in out.per_step:
            k = int(item.get("k", 0))
            for name, value in item.items():
                if name == "k" or not isinstance(value, (int, float)):
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    final[f"iforward/k{k}/{name}"] = value_f
        final["logging_pack_ms"] = float((time.perf_counter() - t0) * 1000.0)
        return final

    def reset_iforward_state_cache(self) -> None:
        self._state_cache.clear()

    def load_init_checkpoint_payload(
        self,
        ckpt: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        weights_only: bool = True,
        path: Optional[str] = None,
    ) -> bool:
        loader = getattr(self.model, "load_init_checkpoint_payload", None)
        if callable(loader):
            return bool(loader(ckpt, device=device or self.device, weights_only=weights_only, path=path))
        return False

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "format": "iforward_trainer_extra_state_v1",
            "state_cache": {
                tuple(key): value.detach_for_next_rollout()
                for key, value in self._state_cache.items()
            },
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            self._state_cache = {}
            return
        raw_cache = state.get("state_cache", {})
        if not isinstance(raw_cache, dict):
            self._state_cache = {}
            return
        self._state_cache = {
            tuple(int(x) for x in key): value.detach_for_next_rollout()
            for key, value in raw_cache.items()
        }

    def load_optimizer_state_from_checkpoint(self, payload: Dict[str, Any]) -> bool:
        opt_state = payload.get("optimizer_state_dict")
        if opt_state is None:
            return False
        self.optimizer.load_state_dict(opt_state)
        return True
