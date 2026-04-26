from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ImportError:
    _gsplat_rasterization = None

from nerfview import CameraState

from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format


@dataclass
class Stage5DemoDisplayState:
    current_scheduler_info: Dict[str, Any] = field(default_factory=dict)
    last_events: List[Dict[str, Any]] = field(default_factory=list)
    last_stats: Dict[str, Any] = field(default_factory=dict)
    last_raw_batch: Optional[Dict[str, Any]] = None
    last_minimal_batch: Optional[Dict[str, Any]] = None
    global_step: int = 0


class Stage5DemoController:
    def __init__(
        self,
        *,
        cfg: Any,
        dataset: Any,
        scheduler: Any,
        trainer: Any,
        device: torch.device,
        stage: str,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.scheduler = scheduler
        self.trainer = trainer
        self.device = device
        self.stage = str(stage)
        self.busy = False
        self.display = Stage5DemoDisplayState()
        demo_cfg = cfg.get("demo") or {}
        infer_cfg = demo_cfg.get("inference") or {}
        self.update_node_state = bool(infer_cfg.get("update_node_state", True))
        self.update_hidden_state = bool(infer_cfg.get("update_hidden_state", True))
        self.update_history_memory = bool(infer_cfg.get("update_history_memory", True))
        self.record_block_history_on_block_exit = bool(infer_cfg.get("record_block_history_on_block_exit", True))

    def _batch_to_minimal(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        tgt = raw_batch.get("target")
        if not isinstance(tgt, dict) or tgt.get("image") is None:
            raise ValueError("scheduler batch must contain target.image")
        num_target_views = int(tgt["image"].shape[0])
        return convert_batch_to_minimal_format(
            raw_batch,
            self.device,
            num_targets=num_target_views,
            include_source_for_2d=True,
            view_selection=None,
        )

    def prime(self) -> Dict[str, Any]:
        if not hasattr(self.scheduler, "materialize_current_batch_without_advance"):
            raise ValueError("scheduler must implement materialize_current_batch_without_advance for demo prime")
        raw_batch = self.scheduler.materialize_current_batch_without_advance()
        info = (
            raw_batch.get("_scheduler_v8_aligned_info")
            or raw_batch.get("_scheduler_v7_aligned_info")
            or raw_batch.get("_scheduler_v4_aligned_info")
            or self.scheduler.get_current_info()
        )
        events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
        minimal = self._batch_to_minimal(raw_batch)
        self.display.current_scheduler_info = dict(info)
        self.display.last_events = list(events)
        self.display.last_raw_batch = raw_batch
        self.display.last_minimal_batch = minimal
        self.display.last_stats = self._build_status(
            stats={},
            minimal_batch=minimal,
            events=events,
            scheduler_info=info,
        )
        return dict(self.display.last_stats)

    def _build_status(
        self,
        *,
        stats: Dict[str, Any],
        minimal_batch: Dict[str, Any],
        events: List[Dict[str, Any]],
        scheduler_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_meta = minimal_batch.get("request_meta") or {}
        source_refs = request_meta.get("source_image_refs") or []
        target_refs = request_meta.get("target_image_refs") or []
        latest_event = events[-1] if len(events) > 0 else {}
        out: Dict[str, Any] = {
            "stage": self.stage,
            "global_step": int(self.display.global_step),
            "scene_id": int(scheduler_info.get("scene_id", -1)),
            "segment_id": int(scheduler_info.get("segment_id", -1)),
            "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
            "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
            "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
            "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
            "source_image_refs": [tuple(x) for x in source_refs],
            "target_image_refs": [tuple(x) for x in target_refs],
            "last_event_type": str(latest_event.get("type", "")),
            "last_event_block_idx_global": int(latest_event.get("block_idx_global", -1))
            if isinstance(latest_event, dict)
            else -1,
        }
        out.update(stats)
        return out

    def step_once(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            raw_batch = self.scheduler.next_batch()
            events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
            info = (
                raw_batch.get("_scheduler_v8_aligned_info")
                or raw_batch.get("_scheduler_v7_aligned_info")
                or raw_batch.get("_scheduler_v4_aligned_info")
                or self.scheduler.get_current_info()
            )
            minimal = self._batch_to_minimal(raw_batch)
            stats = self.trainer.demo_infer_step(
                minimal,
                scheduler_events=events,
                update_node_state=self.update_node_state,
                update_hidden_state=self.update_hidden_state,
                update_history_memory=self.update_history_memory,
            )
            if (
                self.record_block_history_on_block_exit
                and hasattr(self.trainer, "record_block_history")
                and str(self.stage) in ("5_2", "5_3")
            ):
                for ev in events:
                    if str(ev.get("type", "")) == "block_exit":
                        rec_stats = self.trainer.record_block_history(minimal, ev)
                        stats.update(rec_stats)
            self.display.global_step += 1
            self.display.current_scheduler_info = dict(info)
            self.display.last_events = list(events)
            self.display.last_raw_batch = raw_batch
            self.display.last_minimal_batch = minimal
            self.display.last_stats = self._build_status(
                stats=stats,
                minimal_batch=minimal,
                events=events,
                scheduler_info=info,
            )
            return dict(self.display.last_stats)
        finally:
            self.busy = False

    def step_block(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        out: Dict[str, Any] = {}
        while True:
            out = self.step_once()
            if str(out.get("last_event_type", "")) == "block_exit":
                return out

    def reset_current_scene_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.trainer.reset_node_state()
        info = self.display.current_scheduler_info or self.scheduler.get_current_info()
        minimal = self.display.last_minimal_batch
        if minimal is None and hasattr(self.scheduler, "materialize_current_batch_without_advance"):
            raw_batch = self.scheduler.materialize_current_batch_without_advance()
            minimal = self._batch_to_minimal(raw_batch)
            self.display.last_raw_batch = raw_batch
        events = self.display.last_events
        self.display.last_stats = self._build_status(
            stats={"reset_scene_state": 1.0},
            minimal_batch=minimal or {},
            events=events,
            scheduler_info=info,
        )
        return dict(self.display.last_stats)

    @staticmethod
    def _to_numpy_uint8(rgb: torch.Tensor) -> np.ndarray:
        rgb01 = torch.clamp(rgb, 0.0, 1.0).detach().cpu().numpy()
        return (rgb01 * 255.0).astype(np.uint8)

    def _get_active_key(self) -> Tuple[int, int]:
        info = self.display.current_scheduler_info or self.scheduler.get_current_info()
        return int(info.get("scene_id", -1)), int(info.get("segment_id", -1))

    @torch.no_grad()
    def render(self, camera_state: CameraState, img_wh: Tuple[int, int], *, show_distant: bool = True) -> np.ndarray:
        if _gsplat_rasterization is None:
            raise ImportError("gsplat is not available; demo viewer requires gsplat.rendering.rasterization.")
        scene_id, segment_id = self._get_active_key()
        key = (int(scene_id), int(segment_id))
        node_state_bg = self.trainer.node_states_bg.get(key)
        if node_state_bg is None:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)
        node_state_distant = self.trainer.node_states_distant.get(key)
        means_list: List[torch.Tensor] = [node_state_bg.means]
        scales_list: List[torch.Tensor] = [torch.exp(node_state_bg.scales_log)]
        quats_list: List[torch.Tensor] = [node_state_bg.quats]
        opacities_list: List[torch.Tensor] = [torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)]
        colors_list: List[torch.Tensor] = [torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)]
        if show_distant and node_state_distant is not None and int(node_state_distant.means.shape[0]) > 0:
            means_list.append(node_state_distant.means)
            scales_list.append(torch.exp(node_state_distant.scales_log))
            quats_list.append(node_state_distant.quats)
            opacities_list.append(torch.sigmoid(node_state_distant.opacity_logit).squeeze(-1))
            colors_list.append(torch.cat([node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1))
        means = torch.cat(means_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        quats = torch.cat(quats_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        colors = torch.cat(colors_list, dim=0)
        w, h = img_wh
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        k = torch.from_numpy(camera_state.get_K(img_wh)).float().to(self.device)
        render_colors, _, _ = _gsplat_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w)[None, ...],
            Ks=k[None, ...],
            width=int(w),
            height=int(h),
            packed=False,
            rasterize_mode="antialiased",
        )
        return self._to_numpy_uint8(render_colors[0])

