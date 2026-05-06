from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from .scene_render_provider import FrozenStreetForwardSceneProvider
from .sky_branch_v0 import SkyBranchV0
from .sky_render_utils import get_cfg


class MinimalSkyBranchTrainer:
    def __init__(
        self,
        config: Any,
        device: torch.device,
        *,
        scene_provider: Optional[FrozenStreetForwardSceneProvider] = None,
        sky_branch: Optional[SkyBranchV0] = None,
    ) -> None:
        self.config = config
        self.device = device
        if scene_provider is None:
            sf_cfg = get_cfg(config, "streetforward", {}) or {}
            sf_config_path = get_cfg(sf_cfg, "config", None)
            sf_checkpoint = get_cfg(sf_cfg, "checkpoint", None)
            if not sf_config_path or not sf_checkpoint:
                raise ValueError("streetforward.config and streetforward.checkpoint are required.")
            scene_provider = FrozenStreetForwardSceneProvider.from_paths(
                config_path=str(sf_config_path),
                checkpoint_path=str(sf_checkpoint),
                device=device,
            )
        self.scene_provider = scene_provider
        self.sky_branch = sky_branch or SkyBranchV0(config, device=device)
        opt_cfg = get_cfg(config, "optimizer", {}) or {}
        opt_type = str(get_cfg(opt_cfg, "type", "adamw")).lower()
        opt_cls = torch.optim.AdamW if opt_type == "adamw" else torch.optim.Adam
        self.optimizer = opt_cls(
            self.sky_branch.parameters(),
            lr=float(get_cfg(opt_cfg, "lr", 2.0e-4)),
            eps=float(get_cfg(opt_cfg, "eps", 1.0e-8)),
            weight_decay=float(get_cfg(opt_cfg, "weight_decay", 1.0e-4)),
        )
        training_cfg = get_cfg(config, "training", {}) or {}
        self.use_amp = bool(get_cfg(training_cfg, "amp", True)) and torch.cuda.is_available()
        self.grad_clip_norm = float(get_cfg(training_cfg, "grad_clip_norm", 1.0))
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.global_step = 0

    def train_step(
        self,
        minimal_batch: Dict[str, Any],
        *,
        step: Optional[int] = None,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.sky_branch.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scene_pack = self.scene_provider.render_batch(
                minimal_batch,
                scheduler_node_sync=scheduler_node_sync,
                update_scene_state=True,
            )
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.sky_branch.forward_scene_batch(minimal_batch, scene_pack, writeback=False)
        self.grad_scaler.scale(out.loss).backward()
        if self.grad_clip_norm > 0.0:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.sky_branch.parameters(), self.grad_clip_norm)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.sky_branch.commit_forward_output(out)
        self.scene_provider.apply_pending_reset()
        self.global_step = int(step if step is not None else self.global_step + 1)
        logs: Dict[str, Any] = {
            "loss": float(out.loss.detach().item()),
            "global_step": int(self.global_step),
        }
        logs.update({k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in out.logs.items()})
        return logs

    def save_checkpoint(self, path: str, *, kind: str = "resume") -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if kind not in {"resume", "model"}:
            raise ValueError("kind must be 'resume' or 'model'.")
        payload: Dict[str, Any] = {
            "kind": kind,
            "global_step": int(self.global_step),
            "sky_branch_state_dict": self.sky_branch.state_dict(),
        }
        try:
            payload["config"] = OmegaConf.to_container(self.config, resolve=False)
        except Exception:
            payload["config"] = self.config
        if kind == "resume":
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
            payload["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
            payload.update(self.sky_branch.runtime_state_dict())
        torch.save(payload, path)
        return path

    def load_model_checkpoint(self, path: str, *, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self.sky_branch.load_state_dict(payload["sky_branch_state_dict"], strict=strict)
        self.global_step = int(payload.get("global_step", 0))
        return payload

    def load_resume_checkpoint(self, path: str, *, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        if payload.get("kind") != "resume":
            raise ValueError(f"Expected a resume checkpoint, got kind={payload.get('kind')!r}.")
        self.sky_branch.load_state_dict(payload["sky_branch_state_dict"], strict=strict)
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "grad_scaler_state_dict" in payload:
            self.grad_scaler.load_state_dict(payload["grad_scaler_state_dict"])
        self.sky_branch.load_runtime_state_dict(payload)
        self.global_step = int(payload.get("global_step", 0))
        return payload
