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

from streetforward_eval.stage5_6_runtime import run_stage5_6_update_step, stage5_6_runtime_policy
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
        self.viewer_rasterize_mode = str(viewer_cfg.get("rasterize_mode", "auto")).strip().lower()
        if self.viewer_rasterize_mode not in ("auto", "classic", "antialiased"):
            raise ValueError("demo.viewer.rasterize_mode must be one of: auto, classic, antialiased")
        self.use_forward_render_cache = bool(viewer_cfg.get("use_forward_render_cache", False))
        self._display_render_cache: Optional[Dict[str, Any]] = None
        self._display_render_cache_warned = False
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
        self.update_view_transient = bool(infer_cfg.get("update_view_transient", True))
        self.record_block_history_on_block_exit = bool(infer_cfg.get("record_block_history_on_block_exit", True))
        history_cfg = cfg.get("batch_eval", {}).get("history", {}) if cfg.get("batch_eval") is not None else {}
        demo_history_cfg = demo_cfg.get("history") or {}
        self.record_each_step = bool(demo_history_cfg.get("record_each_step", history_cfg.get("record_each_step", False)))
        self.record_block_history_on_block_exit = bool(
            demo_history_cfg.get(
                "record_support_residual_on_input_exit",
                history_cfg.get("record_support_residual_on_input_exit", self.record_block_history_on_block_exit),
            )
        )
        self.reset_train_params_on_scope_change = bool(demo_cfg.get("reset_train_params_on_scope_change", True))
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
        self._initial_optimizer_global_step = int(
            getattr(getattr(self.trainer, "optimizer", None), "global_step", getattr(self.trainer, "global_step", 0))
        )
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
        if isinstance(raw_batch, dict) and isinstance(raw_batch.get("targets"), list):
            return raw_batch
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
            or raw_batch.get("_scheduler_eval_v8_demo_info")
            or raw_batch.get("_scheduler_train_v8_demo_info")
            or raw_batch.get("_scheduler_v8_aligned_info")
            or raw_batch.get("_scheduler_v7_aligned_info")
            or raw_batch.get("_scheduler_v4_aligned_info")
            or self.scheduler.get_current_info()
        )

    def _current_scheduler_info(self) -> Dict[str, Any]:
        return self.display.current_scheduler_info or self.scheduler.get_current_info()

    def _is_train_v8_demo(self) -> bool:
        return bool(
            getattr(self.scheduler, "is_train_v8_demo", False)
            or getattr(self.scheduler, "is_stage5_6_train_v8_demo", False)
            or getattr(self.scheduler, "is_stage5_4_train_v8_demo", False)
        )

    def _is_eval_v8_demo(self) -> bool:
        return bool(getattr(self.scheduler, "is_stage5_6_eval_demo", False))

    def _is_v8_scope_managed_demo(self) -> bool:
        return bool(self._is_eval_v8_demo() or self._is_train_v8_demo())

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
        self._capture_display_render_cache(minimal, scheduler_info=info)
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
        if str(self.stage) not in ("5_2", "5_3", "5_4", "5_6"):
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

    def _record_history_for_batch(
        self,
        minimal: Dict[str, Any],
        scheduler_info: Dict[str, Any],
        *,
        reason: str,
    ) -> Dict[str, Any]:
        if not hasattr(self.trainer, "record_block_history"):
            return {}
        event = self._make_block_exit_event_for_recording(scheduler_info, reason=reason)
        out = self.trainer.record_block_history(minimal, event)
        return dict(out) if isinstance(out, dict) else {}

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
            "demo_scheduler_type": str(scheduler_info.get("demo_scheduler_type", "")),
            "global_step": int(self.display.global_step),
            "optimizer_global_step": int(
                getattr(getattr(self.trainer, "optimizer", None), "global_step", getattr(self.trainer, "global_step", 0))
            ),
            "trained_steps_total": int(self.train_steps_total),
            "trained_steps_since_param_reset": int(self.train_steps_since_param_reset),
            "train_param_reset_count": int(self.train_param_reset_count),
            "scene_id": int(scheduler_info.get("scene_id", -1)),
            "segment_id": int(scheduler_info.get("segment_id", -1)),
            "episode_idx_global": int(scheduler_info.get("episode_idx_global", -1)),
            "block_idx_global": int(scheduler_info.get("block_idx_global", -1)),
            "demo_block_uid": int(scheduler_info.get("demo_block_uid", -1)),
            "block_idx_in_episode": int(scheduler_info.get("block_idx_in_episode", -1)),
            "block_repeat_step": int(scheduler_info.get("block_repeat_step", -1)),
            "segment_local_step": int(scheduler_info.get("segment_local_step", -1)),
            "visit_cursor": int(scheduler_info.get("visit_cursor", -1)),
            "visit_total": int(scheduler_info.get("visit_total", -1)),
            "episode_done": bool(scheduler_info.get("episode_done", False)),
            "sequence_start_pos": int(scheduler_info.get("sequence_start_pos", -1)),
            "source_frame_idx": int(scheduler_info.get("source_frame_idx", -1)),
            "target_frame_indices": [int(x) for x in scheduler_info.get("target_frame_indices", [])],
            "target_frame_roles": [str(x) for x in scheduler_info.get("target_frame_roles", [])],
            "target_image_roles": [str(x) for x in request_meta.get("target_image_roles", [])],
            "near_random_frame_indices": [int(x) for x in request_meta.get("near_random_frame_indices", [])],
            "source_image_refs": [tuple(x) for x in source_refs],
            "target_image_refs": [tuple(x) for x in target_refs],
            "last_event_type": str(latest_event.get("type", "")),
            "last_event_block_idx_global": int(latest_event.get("block_idx_global", -1))
            if isinstance(latest_event, dict)
            else -1,
        }
        out.update(stats)
        return out

    @staticmethod
    def _detach_render_pack(pack: Any) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(pack, dict):
            return None
        keys = ("means_r", "scales_r", "quats_r", "opacities_r", "colors_r")
        if not all(torch.is_tensor(pack.get(k)) for k in keys):
            return None
        return {k: pack[k].detach() for k in keys}

    def _clear_display_render_cache(self) -> None:
        self._display_render_cache = None

    @staticmethod
    def _snapshot_transient_value(value: Any) -> Any:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            return list(value)
        return value

    def _snapshot_display_forward_transients(self) -> Dict[str, Any]:
        sentinel = object()
        names = (
            "_stage5_2_last_full_inputs",
            "_stage5_6_active_cache",
            "_stage5_6_active_fusion_scale",
            "_stage5_6_fusion_delta_norm_terms",
            "_stage5_6_last_fused_features",
            "_stage5_6_last_nearby_debug_images",
            "_stage5_6_last_error_debug_images",
        )
        snap: Dict[str, Any] = {"__sentinel__": sentinel}
        for name in names:
            snap[name] = self._snapshot_transient_value(getattr(self.trainer, name, sentinel))
        return snap

    def _restore_display_forward_transients(self, snap: Dict[str, Any]) -> None:
        sentinel = snap.get("__sentinel__")
        for name, value in snap.items():
            if name == "__sentinel__":
                continue
            if value is sentinel:
                if hasattr(self.trainer, name):
                    delattr(self.trainer, name)
            else:
                setattr(self.trainer, name, value)

    def _capture_display_render_cache(
        self,
        minimal_batch: Optional[Dict[str, Any]],
        *,
        scheduler_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(getattr(self, "use_forward_render_cache", False)):
            self._display_render_cache = None
            return
        if not isinstance(minimal_batch, dict):
            self._display_render_cache = None
            return
        forward_fn = getattr(self.trainer, "forward", None)
        if forward_fn is None:
            self._display_render_cache = None
            return
        was_training = bool(getattr(self.trainer, "training", False))
        transients = self._snapshot_display_forward_transients()
        try:
            self.trainer.eval()
            with torch.no_grad():
                out = self.trainer.forward(minimal_batch)
        except Exception as exc:
            self._display_render_cache = None
            if not bool(self._display_render_cache_warned):
                print(f"[stage5-demo] forward render cache unavailable for current batch: {exc}")
                self._display_render_cache_warned = True
            return
        finally:
            if was_training:
                self.trainer.train()
            self._restore_display_forward_transients(transients)
        if not isinstance(out, dict):
            self._display_render_cache = None
            return

        info = dict(scheduler_info or self._extract_scheduler_info(minimal_batch))
        render_bg = self._detach_render_pack(out.get("_render_params_bg") or out.get("render_params"))
        render_distant = self._detach_render_pack(out.get("_render_params_distant"))
        render_rigid_local = self._detach_render_pack(out.get("_render_params_rigid_local"))
        rigid_u = out.get("_rigid_writeback_idx")
        if torch.is_tensor(rigid_u):
            rigid_u = rigid_u.detach().to(device=self.device, dtype=torch.long)
        else:
            rigid_u = None
        self._display_render_cache = {
            "key": (
                int(info.get("scene_id", minimal_batch.get("scene_id", -1))),
                int(info.get("segment_id", minimal_batch.get("segment_id", -1))),
            ),
            "source_frame_idx": int(info.get("source_frame_idx", minimal_batch.get("source_frame_idx", -1))),
            "render_bg": render_bg,
            "render_distant": render_distant,
            "render_rigid_local": render_rigid_local,
            "rigid_u": rigid_u,
            "node_state_rigid": out.get("_node_state_rigid"),
        }

    def _append_render_pack(
        self,
        pack: Optional[Dict[str, torch.Tensor]],
        means_list: List[torch.Tensor],
        scales_list: List[torch.Tensor],
        quats_list: List[torch.Tensor],
        opacities_list: List[torch.Tensor],
        colors_list: List[torch.Tensor],
    ) -> bool:
        if not isinstance(pack, dict):
            return False
        keys = ("means_r", "scales_r", "quats_r", "opacities_r", "colors_r")
        if not all(torch.is_tensor(pack.get(k)) for k in keys):
            return False
        if int(pack["means_r"].shape[0]) <= 0:
            return False
        means_list.append(pack["means_r"].to(self.device))
        scales_list.append(pack["scales_r"].to(self.device))
        quats_list.append(pack["quats_r"].to(self.device))
        opacities_list.append(pack["opacities_r"].reshape(-1).to(self.device))
        colors_list.append(pack["colors_r"].to(self.device))
        return True

    def _cached_rigid_world_render_pack(
        self,
        cache: Dict[str, Any],
        *,
        frame_idx: Optional[int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        node_state_rigid = cache.get("node_state_rigid")
        render_rigid_local = cache.get("render_rigid_local")
        rigid_u = cache.get("rigid_u")
        if node_state_rigid is None or not isinstance(render_rigid_local, dict) or not torch.is_tensor(rigid_u):
            return None
        render_frame = frame_idx
        if render_frame is None:
            render_frame = int(cache.get("source_frame_idx", -1))
        if int(render_frame) < 0:
            return None
        if hasattr(self.trainer, "_build_rigid_world_for_frame") and hasattr(self.trainer, "_rigid_point_valid_mask"):
            target_valid = torch.nonzero(
                self.trainer._rigid_point_valid_mask(node_state_rigid, int(render_frame)),
                as_tuple=False,
            ).squeeze(1).to(device=self.device, dtype=torch.long)
            if int(target_valid.numel()) == 0:
                return None
            n_rigid = int(node_state_rigid.means.shape[0])
            u = rigid_u.to(device=self.device, dtype=torch.long)
            if int(u.numel()) > 0 and (
                bool((u < 0).any().item()) or bool((u >= n_rigid).any().item())
            ):
                return None
            is_updated = torch.zeros((n_rigid,), dtype=torch.bool, device=self.device)
            if int(u.numel()) > 0:
                is_updated[u] = True
            idx_train = target_valid[is_updated[target_valid]]
            idx_frozen = target_valid[~is_updated[target_valid]]
            if int(idx_train.numel()) > 0 and int(render_rigid_local["means_r"].shape[0]) <= 0:
                return None
            return self.trainer._build_rigid_world_for_frame(
                node_state_rigid,
                int(render_frame),
                idx_train,
                idx_frozen,
                render_rigid_local,
                u,
            )
        if hasattr(self.trainer, "_rigid_local_to_world_render_params") and int(rigid_u.numel()) > 0:
            point_ids = node_state_rigid.point_ids[rigid_u.to(device=self.device, dtype=torch.long), 0]
            return self.trainer._rigid_local_to_world_render_params(
                node_state_rigid,
                render_rigid_local,
                int(render_frame),
                point_ids_subset=point_ids,
            )
        return None

    def _stage5_6_scheduler_node_sync_from_events(
        self,
        scheduler_info: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sv8 = self.cfg.get("scheduler_v8") if self.cfg is not None and hasattr(self.cfg, "get") else None
        execution = sv8.get("execution") if sv8 is not None and hasattr(sv8, "get") else None
        block_order = str(scheduler_info.get("block_order", "block_major")).strip()
        if execution is not None and hasattr(execution, "get"):
            reset_policy = str(
                execution.get(
                    "reset_policy",
                    "episode_end" if block_order == "step_major" else "block_end",
                )
            ).strip()
        else:
            reset_policy = "episode_end" if block_order == "step_major" else "block_end"
        if reset_policy not in ("block_end", "episode_end", "never"):
            raise ValueError("scheduler_v8.execution.reset_policy must be one of ['block_end', 'episode_end', 'never']")
        if reset_policy == "block_end":
            should_reset = any(isinstance(ev, dict) and ev.get("type") == "block_end" for ev in events)
        elif reset_policy == "episode_end":
            should_reset = any(isinstance(ev, dict) and ev.get("type") == "episode_end" for ev in events)
        else:
            should_reset = False
        u = int(scheduler_info.get("U", 1))
        if u < 1:
            raise ValueError("scheduler_v8 scheduler_info.U must be >= 1 for model node-state sync")
        return {
            "U": int(u),
            "segment_local_step": int(scheduler_info.get("segment_local_step", 0)),
            "reset_after_block": bool(should_reset),
            "reset_policy": str(reset_policy),
        }

    def _run_stage5_6_train_scheduler_step(
        self,
        *,
        minimal: Dict[str, Any],
        scheduler_info: Dict[str, Any],
        events: List[Dict[str, Any]],
        defer_node_state_reset: bool,
    ) -> Dict[str, Any]:
        sync = self._stage5_6_scheduler_node_sync_from_events(scheduler_info, events)
        if bool(defer_node_state_reset) and bool(sync.get("reset_after_block", False)):
            sync = dict(sync)
            sync["reset_after_block"] = False
        policy = stage5_6_runtime_policy(
            do_train=bool(self._mode_train_and_infer),
            update_hidden_state=bool(getattr(self.scheduler, "update_hidden_state", self.update_hidden_state)),
            update_node_state=bool(getattr(self.scheduler, "update_node_state", self.update_node_state)),
            reset_node_state_after_block=bool(sync.get("reset_after_block", False)),
            force_eval_mode=False,
        )
        if self._mode_train_and_infer:
            return self.trainer.train_step(
                minimal,
                step=None,
                profile_phase_timing=False,
                sync_cuda_timing=False,
                scheduler_node_sync=sync,
                runtime_policy=policy,
            )
        with torch.no_grad():
            return self.trainer.inference_step_from_train_batch(
                minimal,
                step=None,
                scheduler_node_sync=sync,
                runtime_policy=policy,
            )

    def step_current_block_once(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            train_v8_demo = self._is_train_v8_demo()
            if train_v8_demo:
                raw_batch = self.scheduler.next_batch_for_update()
            else:
                raw_batch = self.scheduler.materialize_current_batch_without_advance()
            events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
            info = self._extract_scheduler_info(raw_batch)
            minimal = self._batch_to_minimal(raw_batch)
            did_train_step = False
            event_block_exit = any(isinstance(e, dict) and e.get("type") == "block_exit" for e in events)
            should_record = bool(self.record_each_step) or (
                bool(self.record_block_history_on_block_exit) and bool(event_block_exit)
            )
            defer_node_state_reset = False
            if train_v8_demo and should_record:
                sync_probe = self._stage5_6_scheduler_node_sync_from_events(info, events)
                defer_node_state_reset = bool(sync_probe.get("reset_after_block", False))
            if self._mode_train_and_infer:
                if train_v8_demo:
                    stats = self._run_stage5_6_train_scheduler_step(
                        minimal=minimal,
                        scheduler_info=info,
                        events=events,
                        defer_node_state_reset=bool(defer_node_state_reset),
                    )
                elif self._is_eval_v8_demo():
                    aligned = minimal.get("_scheduler_v8_aligned_info") or {}
                    stats = run_stage5_6_update_step(
                        model=self.trainer,
                        update_batch=minimal,
                        mode="segment_finetune_train",
                        segment_local_step=int(aligned.get("segment_local_step", self.display.global_step + 1)),
                        update_hidden_state=bool(getattr(self.scheduler, "update_hidden_state", self.update_hidden_state)),
                        update_node_state=bool(getattr(self.scheduler, "update_node_state", self.update_node_state)),
                    )
                else:
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
                if train_v8_demo:
                    stats = self._run_stage5_6_train_scheduler_step(
                        minimal=minimal,
                        scheduler_info=info,
                        events=events,
                        defer_node_state_reset=bool(defer_node_state_reset),
                    )
                elif self._is_eval_v8_demo():
                    aligned = minimal.get("_scheduler_v8_aligned_info") or {}
                    stats = run_stage5_6_update_step(
                        model=self.trainer,
                        update_batch=minimal,
                        mode="inference_only",
                        segment_local_step=int(aligned.get("segment_local_step", self.display.global_step + 1)),
                        update_hidden_state=bool(getattr(self.scheduler, "update_hidden_state", self.update_hidden_state)),
                        update_node_state=bool(getattr(self.scheduler, "update_node_state", self.update_node_state)),
                    )
                else:
                    stats = self.trainer.demo_infer_step(
                        minimal,
                        scheduler_events=events,
                        update_node_state=self.update_node_state,
                        update_hidden_state=self.update_hidden_state,
                        update_history_memory=self.update_history_memory,
                        update_view_transient=self.update_view_transient,
                    )
            if (not train_v8_demo) and hasattr(self.scheduler, "mark_current_block_updated"):
                self.scheduler.mark_current_block_updated()
            post_events = self.scheduler.pop_events() if hasattr(self.scheduler, "pop_events") else []
            all_events = list(events) + list(post_events)
            should_record = bool(self.record_each_step) or (
                bool(self.record_block_history_on_block_exit)
                and any(
                    isinstance(e, dict)
                    and (
                        (e.get("type") == "demo_block_exit" and bool(e.get("model_update", False)))
                        or e.get("type") == "block_exit"
                    )
                    for e in all_events
                )
            )
            if should_record and (
                self._is_eval_v8_demo() or bool(train_v8_demo)
            ):
                rec = self._record_history_for_batch(
                    minimal,
                    minimal.get("_scheduler_v8_aligned_info") or info,
                    reason="train_v8_step" if train_v8_demo else "eval_cursor_step",
                )
                if rec:
                    stats = dict(stats or {})
                    stats.update(rec)
            if bool(defer_node_state_reset) and hasattr(self.trainer, "reset_node_state"):
                self.trainer.reset_node_state()
                stats = dict(stats or {})
                stats["deferred_node_state_reset"] = 1.0
            self.display.global_step += 1
            if did_train_step:
                self.train_steps_total += 1
                self.train_steps_since_param_reset += 1
            live_info = self.scheduler.get_current_info()
            status_info = dict(info)
            if bool(live_info.get("episode_done", False)):
                status_info["episode_done"] = True
            stats = dict(stats or {})
            stats["next_source_frame_idx"] = float(live_info.get("source_frame_idx", -1))
            stats["next_block_idx_in_episode"] = float(live_info.get("block_idx_in_episode", -1))
            if bool(stats.get("deferred_node_state_reset", False)):
                self._clear_display_render_cache()
            else:
                self._capture_display_render_cache(minimal, scheduler_info=status_info)
            self.display.current_scheduler_info = dict(status_info)
            self.display.last_events = list(all_events)
            self.display.last_raw_batch = raw_batch
            self.display.last_minimal_batch = minimal
            self.display.last_stats = self._build_status(
                stats=stats,
                minimal_batch=minimal,
                events=all_events,
                scheduler_info=status_info,
            )
            return dict(self.display.last_stats)
        finally:
            self.busy = False

    def step_once(self) -> Dict[str, Any]:
        return self.step_current_block_once()

    def _steps_per_current_block(self) -> int:
        for obj in (self.scheduler, getattr(self.scheduler, "scheduler", None)):
            if obj is None:
                continue
            for name in ("steps_per_block", "steps_per_input"):
                if hasattr(obj, name):
                    value = int(getattr(obj, name))
                    if value > 0:
                        return int(value)
        return 1

    def _num_blocks_current_episode(self) -> int:
        info = self.scheduler.get_current_info()
        for key in ("sequence_length",):
            value = int(info.get(key, 0))
            if value > 0:
                return int(value)
        for key in ("input_offsets", "input_frame_ids", "frame_chain", "keyframe_window"):
            value = info.get(key)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return int(len(value))
        for obj in (self.scheduler, getattr(self.scheduler, "scheduler", None)):
            if obj is not None and hasattr(obj, "blocks_per_episode"):
                value = int(getattr(obj, "blocks_per_episode"))
                if value > 0:
                    return int(value)
        return 1

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
            v8_reset_scope = self._is_v8_scope_managed_demo()
            if v8_reset_scope and op_name in {
                "next_scene",
                "prev_scene",
                "next_segment",
                "prev_segment",
                "resample_episode",
            }:
                merge_stats.update(self._reset_for_eval_scope_change())
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
            if self._is_v8_scope_managed_demo():
                rec_stats.update(self._reset_for_eval_scope_change())
            return self._refresh_display_from_raw_batch(raw_batch, stats=rec_stats)
        finally:
            self.busy = False

    def set_sequence_start_pos(self, sequence_start_pos: int) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason="set_sequence_start_pos")
            if not hasattr(self.scheduler, "set_sequence_start_pos"):
                raise ValueError("scheduler does not support set_sequence_start_pos")
            raw_batch = self.scheduler.set_sequence_start_pos(int(sequence_start_pos))
            rec_stats.update({"manual_set_sequence_start_pos": 1.0})
            if self._is_v8_scope_managed_demo():
                rec_stats.update(self._reset_for_eval_scope_change())
            return self._refresh_display_from_raw_batch(raw_batch, stats=rec_stats)
        finally:
            self.busy = False

    def set_scope_and_sequence_start_pos(
        self,
        scene_id: int,
        segment_id: int,
        sequence_start_pos: int,
    ) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason="set_scope_and_sequence_start_pos")
            if hasattr(self.scheduler, "set_scope_and_sequence_start_pos"):
                raw_batch = self.scheduler.set_scope_and_sequence_start_pos(
                    int(scene_id),
                    int(segment_id),
                    int(sequence_start_pos),
                )
            elif hasattr(self.scheduler, "_set_scope_and_start"):
                raw_batch = self.scheduler._set_scope_and_start(
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    sequence_start_pos=int(sequence_start_pos),
                    reason="set_scope_and_sequence_start_pos",
                )
            else:
                if not hasattr(self.scheduler, "set_scope") or not hasattr(self.scheduler, "set_sequence_start_pos"):
                    raise ValueError("scheduler does not support set_scope_and_sequence_start_pos")
                raw_batch = self.scheduler.set_scope(int(scene_id), int(segment_id))
                info = self.scheduler.get_current_info()
                if int(info.get("sequence_start_pos", -1)) != int(sequence_start_pos):
                    raw_batch = self.scheduler.set_sequence_start_pos(int(sequence_start_pos))
            rec_stats.update(
                {
                    "manual_set_scope": 1.0,
                    "manual_set_sequence_start_pos": 1.0,
                }
            )
            if self._is_v8_scope_managed_demo():
                rec_stats.update(self._reset_for_eval_scope_change())
            return self._refresh_display_from_raw_batch(raw_batch, stats=rec_stats)
        finally:
            self.busy = False

    def run_current_chunk(self) -> Dict[str, Any]:
        info = self.scheduler.get_current_info()
        start_block = int(info.get("block_idx_in_episode", -1))
        if bool(info.get("episode_done", False)):
            last = dict(self.display.last_stats or {})
            last["run_current_chunk_steps"] = 0
            self.display.last_stats.update(last)
            return last
        max_steps = max(1, int(self._steps_per_current_block()))
        last: Dict[str, Any] = dict(self.display.last_stats or {})
        steps = 0
        while start_block >= 0 and steps < max_steps:
            last = self.step_current_block_once()
            steps += 1
            cur = self.scheduler.get_current_info()
            if int(cur.get("block_idx_in_episode", -1)) != start_block:
                break
        last = dict(last)
        last["run_current_chunk_steps"] = int(steps)
        self.display.last_stats.update(last)
        return last

    def run_episode(self) -> Dict[str, Any]:
        last: Dict[str, Any] = dict(self.display.last_stats or {})
        num_blocks = max(1, int(self._num_blocks_current_episode()))
        steps = 0
        for block_i in range(num_blocks):
            last = self.run_current_chunk()
            steps += int(last.get("run_current_chunk_steps", 0))
            if block_i + 1 >= num_blocks:
                break
            if not hasattr(self.scheduler, "next_block"):
                break
            last = self.next_block()
        last = dict(last)
        last["run_episode_steps"] = int(steps)
        self.display.last_stats.update(last)
        return last

    def new_episode_and_reset_segment_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            rec_stats = self._maybe_record_current_block_history(reason="resample_episode")
            if not hasattr(self.scheduler, "resample_episode"):
                raise ValueError("scheduler does not support resample_episode")
            raw_batch = self.scheduler.resample_episode()
            if self._is_v8_scope_managed_demo():
                rec_stats.update(self._reset_for_eval_scope_change())
                rec_stats.update({"manual_resample_episode": 1.0})
            else:
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

    def list_sequence_start_positions(self) -> List[int]:
        if hasattr(self.scheduler, "list_sequence_start_positions"):
            return [int(x) for x in self.scheduler.list_sequence_start_positions()]
        info = self._current_scheduler_info()
        start = int(info.get("sequence_start_pos", -1))
        return [] if start < 0 else [start]

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
        self._clear_display_render_cache()
        return key

    def _clear_all_runtime_state(self) -> None:
        self._clear_display_render_cache()
        if hasattr(self.trainer, "reset_node_state"):
            self.trainer.reset_node_state()
        for name in (
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
                cache.clear()
        for name in (
            "_stage5_6_frame_cache",
            "_stage5_6_active_cache",
            "_stage5_6_last_fused_features",
        ):
            if hasattr(self.trainer, name):
                value = getattr(self.trainer, name)
                if isinstance(value, dict):
                    value.clear()
                else:
                    setattr(self.trainer, name, None if name == "_stage5_6_active_cache" else {})
        if hasattr(self.trainer, "_stage5_2_last_full_inputs"):
            self.trainer._stage5_2_last_full_inputs = None
        self._recorded_block_update_counts.clear()

    def _restore_training_parameters_in_place(self) -> Tuple[int, int]:
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
            if hasattr(optimizer, "global_step"):
                optimizer.global_step = int(self._initial_optimizer_global_step)
        setattr(self.trainer, "global_step", int(self._initial_optimizer_global_step))
        self.train_steps_since_param_reset = 0
        self.train_param_reset_count += 1
        self._clear_display_render_cache()
        return len(missing_keys), len(unexpected_keys)

    def _reset_for_eval_scope_change(self) -> Dict[str, float]:
        self._clear_all_runtime_state()
        stats: Dict[str, float] = {"reset_all_demo_state": 1.0}
        if self.reset_train_params_on_scope_change and self._mode_train_and_infer:
            missing, unexpected = self._restore_training_parameters_in_place()
            stats.update(
                {
                    "reset_training_parameters": 1.0,
                    "reset_training_parameters_missing_keys": float(missing),
                    "reset_training_parameters_unexpected_keys": float(unexpected),
                }
            )
        return stats

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
        self._capture_display_render_cache(minimal, scheduler_info=info)
        return dict(self.display.last_stats)

    def reset_all_demo_state(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self._clear_all_runtime_state()
        info = self._current_scheduler_info()
        minimal = self.display.last_minimal_batch or {}
        self.display.last_stats = self._build_status(
            stats={"reset_all_demo_state": 1.0},
            minimal_batch=minimal,
            events=self.display.last_events,
            scheduler_info=info,
        )
        self._capture_display_render_cache(minimal, scheduler_info=info)
        return dict(self.display.last_stats)

    def reset_training_parameters(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.busy = True
        try:
            missing_count, unexpected_count = self._restore_training_parameters_in_place()
            self.train_steps_since_param_reset = 0
            info = self._current_scheduler_info()
            minimal = self.display.last_minimal_batch
            if minimal is None and hasattr(self.scheduler, "materialize_current_batch_without_advance"):
                raw_batch = self.scheduler.materialize_current_batch_without_advance()
                minimal = self._batch_to_minimal(raw_batch)
                self.display.last_raw_batch = raw_batch
            self.display.last_stats = self._build_status(
                stats={
                    "reset_training_parameters": 1.0,
                    "reset_training_parameters_missing_keys": float(missing_count),
                    "reset_training_parameters_unexpected_keys": float(unexpected_count),
                },
                minimal_batch=minimal or {},
                events=self.display.last_events,
                scheduler_info=info,
            )
            self._capture_display_render_cache(minimal, scheduler_info=info)
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

    @staticmethod
    def _sanitize_render_tensors(
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quat_norm = torch.linalg.norm(quats, dim=-1, keepdim=True)
        quat_identity = torch.zeros_like(quats)
        quat_identity[..., 0] = 1.0
        quats = torch.where(
            torch.isfinite(quat_norm) & (quat_norm > 1.0e-8),
            quats / quat_norm.clamp_min(1.0e-8),
            quat_identity,
        )
        valid = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(quats).all(dim=-1)
            & torch.isfinite(opacities.reshape(-1))
            & torch.isfinite(colors.reshape(int(colors.shape[0]), -1)).all(dim=-1)
            & (scales > 0.0).all(dim=-1)
            & (opacities.reshape(-1) > 1.0e-6)
        )
        if bool(valid.all().item()):
            return means, scales, quats, opacities.reshape(-1), colors
        return means[valid], scales[valid], quats[valid], opacities.reshape(-1)[valid], colors[valid]

    def _get_active_key(self) -> Tuple[int, int]:
        info = self.display.current_scheduler_info or self.scheduler.get_current_info()
        return int(info.get("scene_id", -1)), int(info.get("segment_id", -1))

    def _ensure_render_node_state_initialized(self, key: Tuple[int, int]) -> None:
        has_bg = isinstance(getattr(self.trainer, "node_states_bg", None), dict) and key in self.trainer.node_states_bg
        has_distant = (
            isinstance(getattr(self.trainer, "node_states_distant", None), dict)
            and key in self.trainer.node_states_distant
        )
        has_rigid = (
            isinstance(getattr(self.trainer, "node_states_rigid", None), dict)
            and key in self.trainer.node_states_rigid
        )
        if has_bg or has_distant or has_rigid:
            return
        minimal = self.display.last_minimal_batch
        if not isinstance(minimal, dict):
            return
        batch_key = (
            int(minimal.get("scene_id", -1)),
            int(minimal.get("segment_id", -1)),
        )
        if batch_key != key:
            return
        if hasattr(self.trainer, "_get_or_init_node_states_bg_rigid_distant"):
            self.trainer._get_or_init_node_states_bg_rigid_distant(minimal)
        elif hasattr(self.trainer, "_get_or_init_node_states_bg_distant"):
            self.trainer._get_or_init_node_states_bg_distant(minimal)
        elif hasattr(self.trainer, "_get_or_init_node_state_bg"):
            self.trainer._get_or_init_node_state_bg(minimal)

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
        self._ensure_render_node_state_initialized(key)
        node_state_bg = self.trainer.node_states_bg.get(key)
        node_state_distant = self.trainer.node_states_distant.get(key)
        node_state_rigid = getattr(self.trainer, "node_states_rigid", {}).get(key)
        means_list: List[torch.Tensor] = []
        scales_list: List[torch.Tensor] = []
        quats_list: List[torch.Tensor] = []
        opacities_list: List[torch.Tensor] = []
        colors_list: List[torch.Tensor] = []
        cache = self._display_render_cache if bool(getattr(self, "use_forward_render_cache", False)) else None
        cache_key = cache.get("key") if isinstance(cache, dict) else None
        if isinstance(cache, dict) and (
            not isinstance(cache_key, tuple) or len(cache_key) != 2 or (int(cache_key[0]), int(cache_key[1])) != key
        ):
            cache = None
        used_cache_bg = False
        used_cache_distant = False
        used_cache_rigid = False
        if isinstance(cache, dict):
            if show_bg:
                used_cache_bg = self._append_render_pack(
                    cache.get("render_bg"),
                    means_list,
                    scales_list,
                    quats_list,
                    opacities_list,
                    colors_list,
                )
            if show_distant:
                used_cache_distant = self._append_render_pack(
                    cache.get("render_distant"),
                    means_list,
                    scales_list,
                    quats_list,
                    opacities_list,
                    colors_list,
                )
            if show_rigid:
                rigid_pack = self._cached_rigid_world_render_pack(cache, frame_idx=rigid_frame_idx)
                used_cache_rigid = self._append_render_pack(
                    rigid_pack,
                    means_list,
                    scales_list,
                    quats_list,
                    opacities_list,
                    colors_list,
                )

        if show_bg and not used_cache_bg and node_state_bg is not None and int(node_state_bg.means.shape[0]) > 0:
            means_list.append(node_state_bg.means)
            scales_list.append(torch.exp(node_state_bg.scales_log))
            quats_list.append(node_state_bg.quats)
            opacities_list.append(torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1))
            colors_list.append(torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1))

        if show_distant and not used_cache_distant and node_state_distant is not None and int(node_state_distant.means.shape[0]) > 0:
            means_list.append(node_state_distant.means)
            scales_list.append(torch.exp(node_state_distant.scales_log))
            quats_list.append(node_state_distant.quats)
            opacities_list.append(torch.sigmoid(node_state_distant.opacity_logit).squeeze(-1))
            colors_list.append(torch.cat([node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1))

        if show_rigid and not used_cache_rigid and node_state_rigid is not None and int(node_state_rigid.means.shape[0]) > 0:
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
        means, scales, quats, opacities, colors = self._sanitize_render_tensors(
            means,
            scales,
            quats,
            opacities,
            colors,
        )
        if int(means.shape[0]) == 0:
            w, h = img_wh
            return np.zeros((h, w, 3), dtype=np.uint8)
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
        viewmats = torch.linalg.inv(c2w)[None, ...]
        Ks = k[None, ...]

        def _rasterize(mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
            render, alpha, _ = _gsplat_rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=int(w),
                height=int(h),
                sh_degree=sh_degree,
                packed=False,
                rasterize_mode=mode,
            )
            return render, alpha

        if self.viewer_rasterize_mode == "auto":
            render_colors, render_alphas = _rasterize("classic")
            alpha_max = float(render_alphas.detach().max().item()) if int(render_alphas.numel()) > 0 else 0.0
            if (not torch.isfinite(render_colors).all().item()) or alpha_max <= 1.0e-8:
                render_colors, _ = _rasterize("antialiased")
        else:
            render_colors, _ = _rasterize(self.viewer_rasterize_mode)
        return self._to_numpy_uint8(render_colors[0])
