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
        viewer_cfg = demo_cfg.get("viewer") or {}
        self.align_gaussians_to_nerfview = bool(viewer_cfg.get("align_gaussians_to_nerfview", True))
        self.coord_align_mode = str(viewer_cfg.get("coord_align_mode", "gaussian")).strip().lower()
        if self.coord_align_mode not in ("gaussian", "camera", "off"):
            raise ValueError("demo.viewer.coord_align_mode must be one of: gaussian, camera, off")
        if self.coord_align_mode == "off":
            self.align_gaussians_to_nerfview = False
        mode = str(demo_cfg.get("mode", "frozen_recurrent_inference")).strip()
        self.mode = mode
        mode_norm = mode.lower()
        self._mode_frozen_infer = mode_norm == "frozen_recurrent_inference"
        self._mode_train_and_infer = mode_norm in ("segment_finetune_train", "validation_v8_segment_finetune_train")
        if not (self._mode_frozen_infer or self._mode_train_and_infer):
            raise ValueError(
                "Unsupported demo.mode="
                f"{mode!r}; expected one of: frozen_recurrent_inference, segment_finetune_train"
            )

        infer_cfg = demo_cfg.get("inference") or {}
        self.update_node_state = bool(infer_cfg.get("update_node_state", True))
        self.update_hidden_state = bool(infer_cfg.get("update_hidden_state", True))
        self.update_history_memory = bool(infer_cfg.get("update_history_memory", True))
        self.record_block_history_on_block_exit = bool(infer_cfg.get("record_block_history_on_block_exit", True))
        self.no_optimizer_step = bool(infer_cfg.get("no_optimizer_step", False))
        self.no_backward = bool(infer_cfg.get("no_backward", False))
        if self._mode_train_and_infer and (self.no_optimizer_step or self.no_backward):
            raise ValueError(
                "demo.inference.no_optimizer_step/no_backward are unsupported in segment_finetune_train mode; "
                "remove them or switch demo.mode to frozen_recurrent_inference."
            )
        train_infer_cfg = demo_cfg.get("train_infer") or {}
        self.train_infer_state_write_interval_steps = int(train_infer_cfg.get("state_write_interval_steps", 1))
        self.train_infer_reset_node_state_after_block = bool(train_infer_cfg.get("reset_node_state_after_block", False))
        if self.train_infer_state_write_interval_steps < 1:
            raise ValueError("demo.train_infer.state_write_interval_steps must be >= 1")
        self._recorded_block_update_counts: Dict[Tuple[int, int, int, int], int] = {}
        self._initial_model_state_dict = self._snapshot_model_state_dict()
        self.train_steps_total = 0
        self.train_steps_since_param_reset = 0
        self.train_param_reset_count = 0

    def _snapshot_model_state_dict(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        for name, value in self.trainer.state_dict().items():
            if isinstance(value, torch.Tensor):
                snap[name] = value.detach().cpu().clone()
            else:
                snap[name] = value
        return snap

    @staticmethod
    def _clone_state_dict_for_load(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                out[name] = value.detach().clone()
            else:
                out[name] = value
        return out

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

    def _extract_scheduler_info(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        return (
            raw_batch.get("_scheduler_demo_v1_info")
            or raw_batch.get("_scheduler_v8_aligned_info")
            or raw_batch.get("_scheduler_v7_aligned_info")
            or raw_batch.get("_scheduler_v4_aligned_info")
            or self.scheduler.get_current_info()
        )

    def _current_scheduler_info(self) -> Dict[str, Any]:
        return self.display.current_scheduler_info or self.scheduler.get_current_info()

    def _refresh_display_from_raw_batch(
        self,
        raw_batch: Dict[str, Any],
        *,
        stats: Optional[Dict[str, Any]] = None,
        scheduler_info: Optional[Dict[str, Any]] = None,
        extra_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        info = dict(scheduler_info) if scheduler_info is not None else self._extract_scheduler_info(raw_batch)
        events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
        if extra_events is not None and len(extra_events) > 0:
            events.extend(list(extra_events))
        minimal = self._batch_to_minimal(raw_batch)
        self.display.current_scheduler_info = dict(info)
        self.display.last_events = list(events)
        self.display.last_raw_batch = raw_batch
        self.display.last_minimal_batch = minimal
        self.display.last_stats = self._build_status(
            stats=dict(stats or {}),
            minimal_batch=minimal,
            events=events,
            scheduler_info=info,
        )
        return dict(self.display.last_stats)

    def _make_block_exit_event_for_recording(self, scheduler_info: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "type": "block_exit",
            "scheduler_version": str(scheduler_info.get("scheduler_version", "demo_v1")),
            "scene_id": int(scheduler_info.get("scene_id", -1)),
            "segment_id": int(scheduler_info.get("segment_id", -1)),
            "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
            "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
            "demo_block_uid": int(scheduler_info.get("demo_block_uid", -1)),
            "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
            "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
            "target_frame_indices": [int(x) for x in scheduler_info.get("target_frame_indices", [])],
            "manual": True,
            "model_update": False,
            "consumed_step": False,
            "reason": str(reason),
        }

    def _maybe_record_current_block_history(self, *, reason: str) -> Dict[str, Any]:
        if not self.record_block_history_on_block_exit:
            return {}
        if not hasattr(self.trainer, "record_block_history"):
            return {}
        if str(self.stage) not in ("5_2", "5_3"):
            return {}
        minimal = self.display.last_minimal_batch
        if minimal is None:
            return {}
        info = self._current_scheduler_info()
        scene_id = int(info.get("scene_id", -1))
        segment_id = int(info.get("segment_id", -1))
        episode_idx = int(info.get("episode_idx_global", -1))
        block_idx = int(info.get("block_idx_in_episode", -1))
        if min(scene_id, segment_id, episode_idx, block_idx) < 0:
            return {}
        updated_counts = info.get("updated_block_counts") or {}
        current_update_count = int(updated_counts.get(block_idx, 0))
        if current_update_count <= 0:
            return {}
        key = (scene_id, segment_id, episode_idx, block_idx)
        recorded_count = int(self._recorded_block_update_counts.get(key, 0))
        if current_update_count <= recorded_count:
            return {}
        event = self._make_block_exit_event_for_recording(info, reason=reason)
        rec_stats = self.trainer.record_block_history(minimal, event)
        self._recorded_block_update_counts[key] = int(current_update_count)
        if isinstance(rec_stats, dict):
            return dict(rec_stats)
        return {}

    def prime(self) -> Dict[str, Any]:
        if not hasattr(self.scheduler, "materialize_current_batch_without_advance"):
            raise ValueError("scheduler must implement materialize_current_batch_without_advance for demo prime")
        raw_batch = self.scheduler.materialize_current_batch_without_advance()
        return self._refresh_display_from_raw_batch(raw_batch, stats={"primed": 1.0})

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
            "mode": self.mode,
            "global_step": int(self.display.global_step),
            "trained_steps_total": int(self.train_steps_total),
            "trained_steps_since_param_reset": int(self.train_steps_since_param_reset),
            "train_param_reset_count": int(self.train_param_reset_count),
            "scene_id": int(scheduler_info.get("scene_id", -1)),
            "segment_id": int(scheduler_info.get("segment_id", -1)),
            "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
            "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
            "demo_block_uid": int(scheduler_info.get("demo_block_uid", -1)),
            "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
            "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
            "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
            "target_frame_indices": [int(x) for x in scheduler_info.get("target_frame_indices", [])],
            "source_image_refs": [tuple(x) for x in source_refs],
            "target_image_refs": [tuple(x) for x in target_refs],
            "last_event_type": str(latest_event.get("type", "")),
            "last_event_block_idx_global": int(latest_event.get("block_idx_global", -1))
            if isinstance(latest_event, dict)
            else -1,
        }
        out.update(stats)
        return out

    def step_current_block_once(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            raw_batch = self.scheduler.materialize_current_batch_without_advance()
            events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
            info = self._extract_scheduler_info(raw_batch)
            minimal = self._batch_to_minimal(raw_batch)
            did_train_step = False
            if self._mode_train_and_infer:
                scheduler_node_sync = {
                    "U": int(self.train_infer_state_write_interval_steps),
                    "segment_local_step": int(info.get("segment_local_step", self.display.global_step)) + 1,
                    "reset_after_block": bool(self.train_infer_reset_node_state_after_block),
                }
                stats = self.trainer.train_step(
                    minimal,
                    step=int(self.display.global_step + 1),
                    profile_phase_timing=False,
                    sync_cuda_timing=False,
                    scheduler_node_sync=scheduler_node_sync,
                )
                did_train_step = True
            else:
                stats = self.trainer.demo_infer_step(
                    minimal,
                    scheduler_events=events,
                    update_node_state=self.update_node_state,
                    update_hidden_state=self.update_hidden_state,
                    update_history_memory=self.update_history_memory,
                )
            if hasattr(self.scheduler, "mark_current_block_updated"):
                self.scheduler.mark_current_block_updated()
            post_events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
            all_events = list(events) + list(post_events)
            self.display.global_step += 1
            if did_train_step:
                self.train_steps_total += 1
                self.train_steps_since_param_reset += 1
            info_after = self.scheduler.get_current_info()
            self.display.current_scheduler_info = dict(info_after)
            self.display.last_events = list(all_events)
            self.display.last_raw_batch = raw_batch
            self.display.last_minimal_batch = minimal
            self.display.last_stats = self._build_status(
                stats=stats,
                minimal_batch=minimal,
                events=all_events,
                scheduler_info=info_after,
            )
            return dict(self.display.last_stats)
        finally:
            self.busy = False

    def step_once(self) -> Dict[str, Any]:
        return self.step_current_block_once()

    def _navigate_without_update(self, op_name: str, *, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason=op_name)
            op = getattr(self.scheduler, op_name, None)
            if op is None:
                raise ValueError(f"scheduler does not support {op_name}")
            raw_batch = op()
            merge_stats = dict(stats or {})
            merge_stats.update(rec_stats)
            return self._refresh_display_from_raw_batch(raw_batch, stats=merge_stats)
        finally:
            self.busy = False

    def step_block(self) -> Dict[str, Any]:
        return self.next_block()

    def next_block(self) -> Dict[str, Any]:
        return self._navigate_without_update("next_block", stats={"manual_next_block": 1.0})

    def prev_block(self) -> Dict[str, Any]:
        return self._navigate_without_update("prev_block", stats={"manual_prev_block": 1.0})

    def next_scene(self) -> Dict[str, Any]:
        return self._navigate_without_update("next_scene", stats={"manual_next_scene": 1.0})

    def prev_scene(self) -> Dict[str, Any]:
        return self._navigate_without_update("prev_scene", stats={"manual_prev_scene": 1.0})

    def next_segment(self) -> Dict[str, Any]:
        return self._navigate_without_update("next_segment", stats={"manual_next_segment": 1.0})

    def prev_segment(self) -> Dict[str, Any]:
        return self._navigate_without_update("prev_segment", stats={"manual_prev_segment": 1.0})

    def set_scope(self, scene_id: int, segment_id: int) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason="set_scope")
            if not hasattr(self.scheduler, "set_scope"):
                raise ValueError("scheduler does not support set_scope")
            raw_batch = self.scheduler.set_scope(int(scene_id), int(segment_id))
            rec_stats.update({"manual_set_scope": 1.0})
            return self._refresh_display_from_raw_batch(raw_batch, stats=rec_stats)
        finally:
            self.busy = False

    def new_episode_and_reset_segment_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason="resample_episode")
            if not hasattr(self.scheduler, "resample_episode"):
                raise ValueError("scheduler does not support resample_episode")
            raw_batch = self.scheduler.resample_episode()
            self._clear_state_for_active_segment()
            rec_stats.update({"manual_resample_episode": 1.0, "reset_segment_state": 1.0})
            return self._refresh_display_from_raw_batch(raw_batch, stats=rec_stats)
        finally:
            self.busy = False

    def list_scene_ids(self) -> List[int]:
        if hasattr(self.scheduler, "list_scene_ids"):
            return [int(x) for x in self.scheduler.list_scene_ids()]
        info = self._current_scheduler_info()
        sid = int(info.get("scene_id", -1))
        return [] if sid < 0 else [sid]

    def list_segment_ids(self, scene_id: int) -> List[int]:
        if hasattr(self.scheduler, "list_segment_ids"):
            return [int(x) for x in self.scheduler.list_segment_ids(int(scene_id))]
        info = self._current_scheduler_info()
        if int(info.get("scene_id", -1)) != int(scene_id):
            return []
        seg = int(info.get("segment_id", -1))
        return [] if seg < 0 else [seg]

    def reset_current_scene_state(self) -> Dict[str, Any]:
        return self.reset_current_segment_state()

    def _clear_state_for_active_segment(self) -> Tuple[int, int]:
        key = self._get_active_key()
        for name in (
            "node_states_bg",
            "node_states_distant",
            "node_states_rigid",
            "h_cache_bg",
            "h_cache_distant",
            "h_cache_rigid",
            "stage5_2_history_bg",
            "stage5_2_history_distant",
            "stage5_2_history_rigid",
            "stage5_2_last_step_update_norm",
            "stage5_2_block_support_bg",
            "stage5_2_block_support_distant",
            "stage5_2_block_support_rigid",
        ):
            cache = getattr(self.trainer, name, None)
            if isinstance(cache, dict):
                cache.pop(key, None)
        if hasattr(self.trainer, "_stage5_2_last_full_inputs"):
            self.trainer._stage5_2_last_full_inputs = None
        scene_id, segment_id = int(key[0]), int(key[1])
        stale_keys = [k for k in self._recorded_block_update_counts if int(k[0]) == scene_id and int(k[1]) == segment_id]
        for k in stale_keys:
            self._recorded_block_update_counts.pop(k, None)
        return key

    def reset_current_segment_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self._clear_state_for_active_segment()
        info = self._current_scheduler_info()
        minimal = self.display.last_minimal_batch
        if minimal is None and hasattr(self.scheduler, "materialize_current_batch_without_advance"):
            raw_batch = self.scheduler.materialize_current_batch_without_advance()
            minimal = self._batch_to_minimal(raw_batch)
            self.display.last_raw_batch = raw_batch
        events = self.display.last_events
        self.display.last_stats = self._build_status(
            stats={"reset_segment_state": 1.0},
            minimal_batch=minimal or {},
            events=events,
            scheduler_info=info,
        )
        return dict(self.display.last_stats)

    def reset_all_demo_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        if hasattr(self.trainer, "reset_node_state"):
            self.trainer.reset_node_state()
        if hasattr(self.trainer, "h_cache_bg"):
            self.trainer.h_cache_bg.clear()
        if hasattr(self.trainer, "h_cache_distant"):
            self.trainer.h_cache_distant.clear()
        if hasattr(self.trainer, "h_cache_rigid"):
            self.trainer.h_cache_rigid.clear()
        if hasattr(self.trainer, "_stage5_2_last_full_inputs"):
            self.trainer._stage5_2_last_full_inputs = None
        self._recorded_block_update_counts.clear()
        info = self._current_scheduler_info()
        minimal = self.display.last_minimal_batch or {}
        self.display.last_stats = self._build_status(
            stats={"reset_all_demo_state": 1.0},
            minimal_batch=minimal,
            events=self.display.last_events,
            scheduler_info=info,
        )
        return dict(self.display.last_stats)

    def reset_training_parameters(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            state_dict = self._clone_state_dict_for_load(self._initial_model_state_dict)
            missing_keys, unexpected_keys = self.trainer.load_state_dict(state_dict, strict=False)
            if hasattr(self.trainer, "zero_grad"):
                try:
                    self.trainer.zero_grad(set_to_none=True)
                except TypeError:
                    self.trainer.zero_grad()
            optimizer = getattr(self.trainer, "optimizer", None)
            if optimizer is not None:
                if hasattr(optimizer, "zero_grad"):
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except TypeError:
                        optimizer.zero_grad()
                if hasattr(optimizer, "state") and isinstance(optimizer.state, dict):
                    optimizer.state.clear()
            self.train_steps_since_param_reset = 0
            self.train_param_reset_count += 1
            info = self._current_scheduler_info()
            minimal = self.display.last_minimal_batch
            if minimal is None and hasattr(self.scheduler, "materialize_current_batch_without_advance"):
                raw_batch = self.scheduler.materialize_current_batch_without_advance()
                minimal = self._batch_to_minimal(raw_batch)
                self.display.last_raw_batch = raw_batch
            self.display.last_stats = self._build_status(
                stats={
                    "reset_training_parameters": 1.0,
                    "reset_training_parameters_missing_keys": float(len(missing_keys)),
                    "reset_training_parameters_unexpected_keys": float(len(unexpected_keys)),
                },
                minimal_batch=minimal or {},
                events=self.display.last_events,
                scheduler_info=info,
            )
            return dict(self.display.last_stats)
        finally:
            self.busy = False

    @staticmethod
    def _to_numpy_uint8(rgb: torch.Tensor) -> np.ndarray:
        rgb01 = torch.clamp(rgb, 0.0, 1.0).detach().cpu().numpy()
        return (rgb01 * 255.0).astype(np.uint8)

    @staticmethod
    def _to_nerfview_world_means(means: torch.Tensor) -> torch.Tensor:
        # MultiSceneDataset world: x-right, y-down, z-forward.
        # nerfview default:         x-right, y-up,   z-backward.
        axis = torch.tensor([1.0, -1.0, -1.0], dtype=means.dtype, device=means.device)
        return means * axis[None, :]

    @staticmethod
    def _to_nerfview_world_quats(quats: torch.Tensor) -> torch.Tensor:
        # Equivalent to conjugation by 180deg rotation around +X axis.
        # For wxyz quaternions this maps (w, x, y, z) -> (w, x, -y, -z).
        if int(quats.shape[-1]) != 4:
            return quats
        axis = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=quats.dtype, device=quats.device)
        return quats * axis[None, :]

    @staticmethod
    def _nerfview_camera_c2w_to_dataset_c2w(c2w_nerfview: torch.Tensor) -> torch.Tensor:
        # nerfview camera local axes are OpenGL-like: x-right, y-up, z-backward.
        # Dataset/gsplat path uses OpenCV-like camera local axes: x-right, y-down, z-forward.
        # Convert camera basis by right-multiplying diag(1, -1, -1, 1).
        d = torch.eye(4, dtype=c2w_nerfview.dtype, device=c2w_nerfview.device)
        d[1, 1] = -1.0
        d[2, 2] = -1.0
        return c2w_nerfview @ d

    @staticmethod
    def _infer_sh_degree(colors: torch.Tensor) -> Optional[int]:
        if int(colors.dim()) < 3 or int(colors.shape[-1]) != 3:
            return None
        num_coeff = int(colors.shape[-2])
        if num_coeff <= 0:
            return None
        sh_degree = 0
        while (sh_degree + 2) * (sh_degree + 2) <= num_coeff:
            sh_degree += 1
        return int(sh_degree)

    def _get_active_key(self) -> Tuple[int, int]:
        info = self.display.current_scheduler_info or self.scheduler.get_current_info()
        return int(info.get("scene_id", -1)), int(info.get("segment_id", -1))

    @torch.no_grad()
    def render(
        self,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
        *,
        show_bg: bool = True,
        show_distant: bool = True,
        show_rigid: bool = False,
        rigid_frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        if _gsplat_rasterization is None:
            raise ImportError("gsplat is not available; demo viewer requires gsplat.rendering.rasterization.")
        scene_id, segment_id = self._get_active_key()
        key = (int(scene_id), int(segment_id))
        node_state_bg = self.trainer.node_states_bg.get(key)
        node_state_distant = self.trainer.node_states_distant.get(key)
        node_state_rigid = getattr(self.trainer, "node_states_rigid", {}).get(key)
        means_list: List[torch.Tensor] = []
        scales_list: List[torch.Tensor] = []
        quats_list: List[torch.Tensor] = []
        opacities_list: List[torch.Tensor] = []
        colors_list: List[torch.Tensor] = []

        if show_bg and node_state_bg is not None and int(node_state_bg.means.shape[0]) > 0:
            means_list.append(node_state_bg.means)
            scales_list.append(torch.exp(node_state_bg.scales_log))
            quats_list.append(node_state_bg.quats)
            opacities_list.append(torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1))
            colors_list.append(torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1))

        if show_distant and node_state_distant is not None and int(node_state_distant.means.shape[0]) > 0:
            means_list.append(node_state_distant.means)
            scales_list.append(torch.exp(node_state_distant.scales_log))
            quats_list.append(node_state_distant.quats)
            opacities_list.append(torch.sigmoid(node_state_distant.opacity_logit).squeeze(-1))
            colors_list.append(torch.cat([node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1))

        if show_rigid and node_state_rigid is not None and int(node_state_rigid.means.shape[0]) > 0:
            render_rigid_frame = rigid_frame_idx
            if render_rigid_frame is None:
                render_rigid_frame = int(self._current_scheduler_info().get("source_frame_idx", -1))
            if (
                int(render_rigid_frame) >= 0
                and hasattr(self.trainer, "_rigid_point_valid_mask")
                and hasattr(self.trainer, "_route_rigid_source_points")
            ):
                try:
                    mask_src_rigid = self.trainer._rigid_point_valid_mask(node_state_rigid, int(render_rigid_frame))
                except Exception:
                    mask_src_rigid = None
                if isinstance(mask_src_rigid, torch.Tensor):
                    S = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)
                    if int(S.numel()) > 0:
                        route = self.trainer._route_rigid_source_points(node_state_rigid, int(render_rigid_frame), S)
                        means_list.append(route.means_world_S)
                        quats_list.append(route.quats_world_S)
                        scales_list.append(torch.exp(node_state_rigid.scales_log[S]))
                        opacities_list.append(torch.sigmoid(node_state_rigid.opacity_logit[S]).squeeze(-1))
                        colors_list.append(torch.cat([node_state_rigid.sh_dc[S, None, :], node_state_rigid.sh_rest[S]], dim=1))

        if len(means_list) == 0:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)

        means = torch.cat(means_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        quats = torch.cat(quats_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        colors = torch.cat(colors_list, dim=0)
        w, h = img_wh
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        if self.align_gaussians_to_nerfview:
            if self.coord_align_mode == "gaussian":
                means = self._to_nerfview_world_means(means)
                quats = self._to_nerfview_world_quats(quats)
            elif self.coord_align_mode == "camera":
                c2w = self._nerfview_camera_c2w_to_dataset_c2w(c2w)
        k = torch.from_numpy(camera_state.get_K(img_wh)).float().to(self.device)
        sh_degree = self._infer_sh_degree(colors)
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
            sh_degree=sh_degree,
            packed=False,
            rasterize_mode="antialiased",
        )
        return self._to_numpy_uint8(render_colors[0])
