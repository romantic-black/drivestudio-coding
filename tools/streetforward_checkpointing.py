from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

_RUNTIME_SKIP_PREFIXES = (
    "node_states_",
    "h_cache_",
    "stage5_2_history_",
    "stage5_2_block_support_",
    "_last_full_inputs",
)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _prune_keep_last_k(*, checkpoint_dir: str, prefix: str, keep_last_k: int) -> None:
    k = int(keep_last_k)
    if k <= 0:
        return
    patt = os.path.join(str(checkpoint_dir), f"{prefix}_ep*.pt")
    files = sorted(glob.glob(patt))
    if len(files) <= k:
        return
    to_delete = files[: len(files) - k]
    for p in to_delete:
        try:
            os.remove(p)
        except OSError:
            continue


def production_model_state_dict(model: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in model.state_dict().items():
        if str(k).startswith(_RUNTIME_SKIP_PREFIXES):
            continue
        out[k] = v
    return out


def save_stage5_3_production_lightweight_checkpoint(
    path: str,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    grad_scaler: torch.cuda.amp.GradScaler,
    train_scheduler: Any,
    global_step: int,
    epoch_idx: int,
    cfg: Any,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not hasattr(train_scheduler, "is_at_episode_boundary"):
        raise ValueError("Train scheduler must expose is_at_episode_boundary() for lightweight checkpoints.")
    if not bool(train_scheduler.is_at_episode_boundary()):
        raise ValueError("Lightweight checkpoint save requires episode boundary state.")
    if not hasattr(train_scheduler, "production_state_dict"):
        raise ValueError("Train scheduler must expose production_state_dict() for lightweight checkpoints.")

    payload = {
        "format_version": 1,
        "checkpoint_type": "stage5_3_production_lightweight",
        "global_step": int(global_step),
        "epoch_idx": int(epoch_idx),
        "model_state_dict": production_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "amp_scaler_state_dict": grad_scaler.state_dict(),
        "train_scheduler_state": train_scheduler.production_state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "meta": {
            "stage": "5_3",
            "production_training": True,
            "resume_semantics": "lightweight_no_runtime_history",
            "torch_version": str(torch.__version__),
            **(dict(extra_meta) if extra_meta is not None else {}),
        },
    }
    _ensure_parent_dir(path)
    torch.save(payload, path)
    return payload


def load_stage5_3_production_lightweight_checkpoint(
    path: str,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    grad_scaler: torch.cuda.amp.GradScaler,
    train_scheduler: Any,
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if str(ckpt.get("checkpoint_type", "")) != "stage5_3_production_lightweight":
        raise ValueError(
            f"Unsupported checkpoint_type={ckpt.get('checkpoint_type')!r}, expected stage5_3_production_lightweight."
        )
    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if bool(strict):
        missing = [k for k in list(incompatible.missing_keys) if not str(k).startswith(_RUNTIME_SKIP_PREFIXES)]
        unexpected = [k for k in list(incompatible.unexpected_keys) if not str(k).startswith(_RUNTIME_SKIP_PREFIXES)]
        if missing or unexpected:
            raise RuntimeError(
                "Model state_dict strict load failed for non-runtime keys: "
                f"missing={missing}, unexpected={unexpected}"
            )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    grad_scaler.load_state_dict(ckpt["amp_scaler_state_dict"])
    if not hasattr(train_scheduler, "load_production_state_dict"):
        raise ValueError("Train scheduler must expose load_production_state_dict() for lightweight resume.")
    train_scheduler.load_production_state_dict(ckpt["train_scheduler_state"])
    if not hasattr(model, "clear_runtime_state_for_lightweight_resume"):
        raise ValueError("Model must implement clear_runtime_state_for_lightweight_resume().")
    model.clear_runtime_state_for_lightweight_resume()
    ckpt_global_step = int(ckpt.get("global_step", 0))
    scheduler_global_step = int(getattr(train_scheduler, "global_step", ckpt_global_step))
    if scheduler_global_step != ckpt_global_step:
        raise ValueError(
            f"Scheduler global_step mismatch after resume: scheduler={scheduler_global_step}, ckpt={ckpt_global_step}."
        )
    lr_last_epoch = getattr(lr_scheduler, "last_epoch", None)
    if lr_last_epoch is not None:
        last_epoch = int(lr_last_epoch)
        if last_epoch not in {ckpt_global_step, ckpt_global_step - 1}:
            raise ValueError(
                f"LR scheduler last_epoch mismatch after resume: last_epoch={last_epoch}, ckpt_global_step={ckpt_global_step}."
            )
    return {
        "global_step": ckpt_global_step,
        "epoch_idx": int(ckpt.get("epoch_idx", 0)),
        "meta": dict(ckpt.get("meta", {})),
    }


__all__ = [
    "_prune_keep_last_k",
    "load_stage5_3_production_lightweight_checkpoint",
    "production_model_state_dict",
    "save_stage5_3_production_lightweight_checkpoint",
]
