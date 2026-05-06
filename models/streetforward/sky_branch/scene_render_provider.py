from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf

from models.streetforward.minimal_trainer_stage4_3 import RuntimePolicy
from models.streetforward.minimal_trainer_stage5_6 import MinimalStreetForwardStage5_6
from models.streetforward.minimal_trainer_stage5_6_production import MinimalStreetForwardStage5_6_Production


@dataclass
class SceneRenderPack:
    source_rgb: torch.Tensor
    source_alpha: torch.Tensor
    target_rgb: torch.Tensor
    target_alpha: torch.Tensor


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    raise ValueError(f"Expected list/tuple, got {type(x)!r}")


class FrozenStreetForwardSceneProvider:
    """
    Two-stage frozen StreetForward provider.

    `update_scene_state` is the only method allowed to advance the StreetForward
    runtime state. `render_scene_views` only renders the current state and returns
    rendered alpha maps, not Gaussian opacity parameters.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        config: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.device = device
        if model is None:
            if config is None:
                raise ValueError("config is required when model is not provided.")
            model_cfg = config.get("model") if hasattr(config, "get") else getattr(config, "model")
            use_production = bool(model_cfg.get("production_training", False))
            cls = MinimalStreetForwardStage5_6_Production if use_production else MinimalStreetForwardStage5_6
            model = cls(config=config, device=device).to(device)
            if checkpoint_path:
                model.load_checkpoint(checkpoint_path, load_optimizer=False, strict=False)
        self.model = model.to(device) if hasattr(model, "to") else model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._pending_reset = False

    @classmethod
    def from_paths(cls, *, config_path: str, checkpoint_path: str, device: torch.device) -> "FrozenStreetForwardSceneProvider":
        cfg = OmegaConf.load(config_path)
        return cls(device=device, config=cfg, checkpoint_path=checkpoint_path)

    def update_scene_state(
        self,
        minimal_batch: Dict[str, Any],
        *,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sync = dict(scheduler_node_sync) if scheduler_node_sync is not None else None
        self._pending_reset = bool(sync.get("reset_after_block", False)) if sync is not None else False
        if sync is not None:
            sync["reset_after_block"] = False
        policy = RuntimePolicy(
            do_backward=False,
            do_optimizer_step=False,
            update_hidden_cache=True,
            writeback_node_state=True,
            reset_node_state_after_block=False,
        )
        with torch.no_grad():
            if hasattr(self.model, "inference_step_from_train_batch"):
                return self.model.inference_step_from_train_batch(
                    minimal_batch,
                    scheduler_node_sync=sync,
                    runtime_policy=policy,
                )
            out = self.model.forward(minimal_batch)
            if hasattr(self.model, "_writeback_node_states_from_out"):
                self.model._writeback_node_states_from_out(out)
            return out

    def apply_pending_reset(self) -> None:
        if self._pending_reset:
            self.model.reset_node_state()
            self._pending_reset = False

    @staticmethod
    def _items_from_targets(targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for t in targets:
            items.append(
                {
                    "view": t["view"],
                    "gt_image": t["gt_image"],
                    "frame_idx": int(t.get("frame_idx", 0)),
                    "cam_idx": int(t.get("cam_idx", -1)),
                }
            )
        return items

    @staticmethod
    def _items_from_views(
        *,
        views: Sequence[Any],
        images: Sequence[torch.Tensor],
        frame_indices: Optional[Sequence[int]] = None,
        default_frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        if len(views) != len(images):
            raise ValueError(f"views/images length mismatch: {len(views)} vs {len(images)}")
        items: List[Dict[str, Any]] = []
        for i, (view, img) in enumerate(zip(views, images)):
            frame_idx = int(frame_indices[i]) if frame_indices is not None else int(default_frame_idx)
            items.append({"view": view, "gt_image": img, "frame_idx": frame_idx})
        return items

    def render_scene_views(
        self,
        minimal_batch: Dict[str, Any],
        views_or_targets: Sequence[Any],
        *,
        images: Optional[Sequence[torch.Tensor]] = None,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = _as_list(views_or_targets)
        if len(seq) == 0:
            return (
                torch.empty(0, 0, 0, 3, device=self.device),
                torch.empty(0, 0, 0, 1, device=self.device),
            )
        if isinstance(seq[0], dict) and "view" in seq[0]:
            items = self._items_from_targets(seq)  # type: ignore[arg-type]
        else:
            if images is None:
                raise ValueError("images are required when rendering bare views.")
            default_frame_idx = int(minimal_batch.get("source_frame_idx", 0))
            items = self._items_from_views(
                views=seq,
                images=images,
                frame_indices=frame_indices,
                default_frame_idx=default_frame_idx,
            )
        with torch.no_grad():
            rgb, alpha = self.model._render_scene_views_from_current_state(minimal_batch, items)
        return rgb.detach(), alpha.detach()

    def render_batch(
        self,
        minimal_batch: Dict[str, Any],
        *,
        scheduler_node_sync: Optional[Dict[str, Any]] = None,
        update_scene_state: bool = True,
    ) -> SceneRenderPack:
        if update_scene_state:
            self.update_scene_state(minimal_batch, scheduler_node_sync=scheduler_node_sync)
        source_views = _as_list(minimal_batch.get("source_views") or [])
        source_images = _as_list(minimal_batch.get("source_images") or [])
        targets = _as_list(minimal_batch.get("targets") or [])
        if len(source_views) == 0 or len(source_images) == 0 or len(targets) == 0:
            raise ValueError("minimal_batch must contain source_views/source_images and targets.")
        source_frame_idx = int(minimal_batch.get("source_frame_idx", 0))
        src_rgb, src_alpha = self.render_scene_views(
            minimal_batch,
            source_views,
            images=source_images,
            frame_indices=[source_frame_idx] * len(source_views),
        )
        tgt_rgb, tgt_alpha = self.render_scene_views(minimal_batch, targets)
        return SceneRenderPack(
            source_rgb=src_rgb,
            source_alpha=src_alpha,
            target_rgb=tgt_rgb,
            target_alpha=tgt_alpha,
        )
