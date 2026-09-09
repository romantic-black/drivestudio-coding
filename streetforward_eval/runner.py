from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import torch

from .batch_builder import build_update_batch_from_refs, make_refs_for_frames
from .episode_builder import TestEpisodeSpec
from .protocols import TestProtocolSpec
from .stage5_6_runtime import (
    build_stage5_6_eval_train_batch,
    iter_block_visit_order,
    nearby_offsets_adjacent_non_input,
    run_stage5_6_update_step,
    stage5_6_runtime_policy,
)

ImageRef = Tuple[int, int]


def convert_batch_to_minimal_format(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from tools.train_minimal_streetforward_stage1_1 import (
        convert_batch_to_minimal_format as _convert_batch_to_minimal_format,
    )

    return _convert_batch_to_minimal_format(*args, **kwargs)


class SnapshotWriterLike(Protocol):
    def write_iteration(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        render_rows: List[Dict[str, Any]],
        psnr_by_view: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> str: ...


class MetricAccumulatorLike(Protocol):
    def add_iteration_rows(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        render_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]: ...

    def finalize_episode(self, spec: TestEpisodeSpec) -> Dict[str, Any]: ...


@dataclass
class RunnerRuntimeConfig:
    mode: Optional[str] = None
    no_grad: bool = True
    amp: bool = True
    reset_state_per_episode: bool = True
    update_node_state: bool = True
    update_hidden_state: bool = True
    update_view_transient: bool = True
    update_step_norm_ema: bool = True
    history_record_on_input_exit: bool = True
    history_record_each_step: bool = False
    block_order: str = "block_major"
    step_major_switch_interval_steps: int = 1
    reset_policy: str = "episode_end"
    target_frame_policy: str = "all_observed"
    max_target_frames_including_source: Optional[int] = None
    update_camera_ids: Optional[List[int]] = None
    stage5_6_enable_nearby_feedback: bool = False
    stage5_6_nearby_policy: str = "adjacent_non_input"
    stage5_6_nearby_role_name: str = "near_random"
    stage5_6_allow_partial_nearby: bool = True
    sky_compose_for_metrics: bool = False
    sky_reset_state_per_episode: bool = True


class StreetForwardBatchEvalRunner:
    def __init__(
        self,
        *,
        model: Any,
        dataset: Any,
        protocol: TestProtocolSpec,
        writer: SnapshotWriterLike,
        metric_acc: MetricAccumulatorLike,
        device: torch.device,
        runtime_cfg: RunnerRuntimeConfig,
        sky_branch: Optional[Any] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.protocol = protocol
        self.writer = writer
        self.metric_acc = metric_acc
        self.device = device
        self.runtime_cfg = runtime_cfg
        self.sky_branch = sky_branch
        self._validate_model_interface()

    def _runtime_mode(self) -> str:
        mode = self.runtime_cfg.mode
        if mode is None:
            return "inference_only" if bool(self.runtime_cfg.no_grad) else "segment_finetune_train"
        mode_norm = str(mode).strip()
        if mode_norm in ("inference_only", "segment_finetune_train"):
            return mode_norm
        raise ValueError(
            "runtime.mode must be one of ['inference_only', 'segment_finetune_train'] "
            f"when set, got {mode!r}"
        )

    def _uses_stage5_6_train_batch_path(self) -> bool:
        return bool(self.runtime_cfg.stage5_6_enable_nearby_feedback)

    def _validate_model_interface(self) -> None:
        required: List[str] = []
        if not (hasattr(self.model, "reset_for_segment_eval") or hasattr(self.model, "reset_node_state")):
            raise ValueError(
                "BatchEval requires model.reset_for_segment_eval() or model.reset_node_state()."
            )
        render_supported = hasattr(self.model, "eval_sparse_render_frames") or (
            hasattr(self.model, "_render_scene_views_from_current_state")
            and hasattr(self.dataset, "_assemble_segment_batch_from_image_refs")
        )
        if not render_supported:
            raise ValueError(
                "BatchEval requires model.eval_sparse_render_frames(), or "
                "model._render_scene_views_from_current_state() with "
                "dataset._assemble_segment_batch_from_image_refs()."
            )
        if self._uses_stage5_6_train_batch_path():
            if self._runtime_mode() == "inference_only":
                required.append("inference_step_from_train_batch")
            else:
                required.append("train_step")
        else:
            if bool(self.runtime_cfg.no_grad):
                if not (
                    hasattr(self.model, "eval_sparse_update_step")
                    or hasattr(self.model, "inference_step_from_train_batch")
                ):
                    raise ValueError(
                        "BatchEval no-grad update requires model.eval_sparse_update_step() "
                        "or model.inference_step_from_train_batch()."
                    )
            else:
                required.append("train_step")
        for name in required:
            if not hasattr(self.model, name):
                raise ValueError(
                    f"BatchEval requires model.{name}(). "
                    "Please use production trainer class with sparse eval interfaces."
                )
        need_history_record = bool(self.runtime_cfg.history_record_on_input_exit) or bool(
            self.runtime_cfg.history_record_each_step
        )
        if need_history_record and not (
            hasattr(self.model, "eval_sparse_record_history") or hasattr(self.model, "record_block_history")
        ):
            raise ValueError(
                "history_record_on_input_exit/history_record_each_step is enabled but model lacks "
                "eval_sparse_record_history()/record_block_history()."
            )
        reset_policy = str(self.runtime_cfg.reset_policy).strip()
        if reset_policy not in ("block_end", "episode_end", "never"):
            raise ValueError(
                "runtime.reset_policy must be one of ['block_end', 'episode_end', 'never']"
            )
        if str(self.runtime_cfg.block_order).strip() == "step_major" and reset_policy == "block_end":
            raise ValueError(
                "runtime.block_order=step_major is incompatible with runtime.reset_policy=block_end; "
                "use episode_end or never."
            )
        if self._uses_stage5_6_train_batch_path() and not hasattr(self.dataset, "_assemble_segment_batch_from_image_refs"):
            raise ValueError(
                "Stage5_6 batch eval path requires dataset._assemble_segment_batch_from_image_refs()."
            )

    def _update_camera_ids(self, spec: TestEpisodeSpec) -> List[int]:
        ids = self.runtime_cfg.update_camera_ids
        if ids is None:
            return [int(x) for x in spec.camera_ids]
        if len(ids) == 0:
            raise ValueError("runtime.update_camera_ids must not be empty when provided")
        return [int(x) for x in ids]

    @contextlib.contextmanager
    def _disable_stage5_6_feedback_cache_for_render(self):
        if not self._uses_stage5_6_train_batch_path() or not hasattr(self.model, "stage5_6_cache_enable"):
            yield
            return
        prev = bool(getattr(self.model, "stage5_6_cache_enable"))
        setattr(self.model, "stage5_6_cache_enable", False)
        try:
            yield
        finally:
            setattr(self.model, "stage5_6_cache_enable", prev)

    @contextlib.contextmanager
    def _preserve_model_runtime_for_render(self, batch: Dict[str, Any]):
        if not (
            hasattr(self.model, "_batch_key")
            and hasattr(self.model, "_snapshot_runtime_state")
            and hasattr(self.model, "_restore_runtime_state")
        ):
            yield
            return
        key = self.model._batch_key(batch)
        snap = self.model._snapshot_runtime_state(key)
        try:
            yield
        finally:
            self.model._restore_runtime_state(key, snap)
            for snap_name, cache_name in (
                ("bg", "node_states_bg"),
                ("distant", "node_states_distant"),
                ("rigid", "node_states_rigid"),
                ("sky", "node_states_sky"),
            ):
                if snap.get(snap_name) is not None:
                    continue
                cache = getattr(self.model, cache_name, None)
                if isinstance(cache, dict):
                    cache.pop(key, None)

    def _render_current_state(self, *, spec: TestEpisodeSpec) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]]
        if hasattr(self.model, "_render_scene_views_from_current_state") and hasattr(
            self.dataset, "_assemble_segment_batch_from_image_refs"
        ):
            with torch.no_grad():
                eval_batch = self._build_eval_minimal_batch(spec=spec)
                targets = list(eval_batch.get("targets") or [])
                render_items = [
                    {
                        "view": t["view"],
                        "gt_image": t["gt_image"],
                        "frame_idx": int(t.get("frame_idx", -1)),
                        "cam_idx": int(t.get("cam_idx", -1)),
                    }
                    for t in targets
                ]
                with self._preserve_model_runtime_for_render(eval_batch):
                    if hasattr(self.model, "ensure_runtime_state_from_batch"):
                        self.model.ensure_runtime_state_from_batch(eval_batch)
                    scene_rgb, _scene_alpha = self.model._render_scene_views_from_current_state(eval_batch, render_items)
                rows = []
                for idx, t in enumerate(targets):
                    row: Dict[str, Any] = {
                        "frame_idx": int(t.get("frame_idx", -1)),
                        "cam_idx": int(t.get("cam_idx", -1)),
                        "pred_rgb": scene_rgb[idx].detach(),
                        "gt_image": t["gt_image"].detach() if torch.is_tensor(t.get("gt_image")) else t.get("gt_image"),
                    }
                    if "sky_mask" in t:
                        row["sky_mask"] = t["sky_mask"]
                    if "egocar_mask" in t:
                        row["egocar_mask"] = t["egocar_mask"]
                    rows.append(row)
        else:
            grad_ctx = torch.no_grad() if bool(self.runtime_cfg.no_grad) else contextlib.nullcontext()
            with grad_ctx:
                with self._disable_stage5_6_feedback_cache_for_render():
                    out = self.model.eval_sparse_render_frames(
                        scene_id=int(spec.scene_id),
                        segment_id=int(spec.segment_id),
                        image_refs=[(int(f), int(c)) for f, c in spec.eval_image_refs],
                        camera_ids=[int(x) for x in spec.camera_ids],
                        save_dir=None,
                        amp=bool(self.runtime_cfg.amp),
                    )
            rows = list(out.get("rows", []))
        if self.sky_branch is not None and bool(self.runtime_cfg.sky_compose_for_metrics):
            rows = self._compose_sky_for_render_rows(spec=spec, scene_rows=rows)
        return rows

    def _build_eval_minimal_batch(self, *, spec: TestEpisodeSpec) -> Dict[str, Any]:
        if len(spec.eval_image_refs) == 0:
            raise ValueError("eval_image_refs must not be empty")
        source_ref = (int(spec.eval_image_refs[0][0]), int(spec.eval_image_refs[0][1]))
        raw = self.dataset._assemble_segment_batch_from_image_refs(
            int(spec.scene_id),
            int(spec.segment_id),
            [source_ref],
            [(int(f), int(c)) for f, c in spec.eval_image_refs],
            aux_image_refs=[],
            include_test=False,
            test_image_refs=None,
            enforce_target0_equals_source=False,
            target_ref_purpose="train",
        )
        return convert_batch_to_minimal_format(
            raw,
            device=self.device,
            num_targets=len(spec.eval_image_refs),
            include_source_for_2d=True,
        )

    def _compose_sky_for_render_rows(
        self,
        *,
        spec: TestEpisodeSpec,
        scene_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.sky_branch is None or len(scene_rows) == 0:
            return scene_rows
        if not hasattr(self.dataset, "_assemble_segment_batch_from_image_refs"):
            return scene_rows
        with torch.no_grad():
            eval_batch = self._build_eval_minimal_batch(spec=spec)
            state = self.sky_branch.get_or_init_node_state(eval_batch)
            render_params = self.sky_branch.state_to_render_params(state)
            targets = list(eval_batch.get("targets") or [])
            target_views = [t["view"] for t in targets]
            target_images = [t["gt_image"].to(self.device) for t in targets]
            if len(targets) != len(scene_rows):
                raise ValueError(
                    "SkyBranch metric composition expected one target per render row, got "
                    f"targets={len(targets)} rows={len(scene_rows)}"
                )
            sky_rgb, _sky_alpha = self.sky_branch.render_views(render_params, target_views, target_images)
            out_rows: List[Dict[str, Any]] = []
            for idx, row in enumerate(scene_rows):
                t = targets[idx]
                pred = row.get("pred_rgb")
                if not torch.is_tensor(pred) or t.get("sky_mask") is None:
                    out_rows.append(row)
                    continue
                mask = t["sky_mask"].to(device=pred.device, dtype=pred.dtype)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                if mask.dim() == 3 and int(mask.shape[-1]) != 1 and int(mask.shape[0]) == 1:
                    mask = mask.permute(1, 2, 0).contiguous()
                mask = mask.clamp(0.0, 1.0)
                sky = sky_rgb[idx].to(device=pred.device, dtype=pred.dtype)
                next_row = dict(row)
                next_row["pred_rgb_scene"] = pred
                next_row["pred_rgb_sky"] = sky.detach()
                next_row["pred_rgb"] = pred * (1.0 - mask) + sky * mask
                out_rows.append(next_row)
        return out_rows

    def _record_iteration(
        self,
        *,
        spec: TestEpisodeSpec,
        global_iter: int,
        is_pre_update: bool,
        input_index: Optional[int],
        input_frame_id: Optional[int],
        local_step: int,
        render_rows: List[Dict[str, Any]],
    ) -> None:
        metric_rows = self.metric_acc.add_iteration_rows(
            spec=spec,
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_index=None if input_index is None else int(input_index),
            input_frame_id=None if input_frame_id is None else int(input_frame_id),
            local_step=int(local_step),
            render_rows=render_rows,
        )
        psnr_by_view: Dict[Tuple[int, int], float] = {}
        for row in metric_rows:
            key = (int(row["eval_frame_id"]), int(row["cam_id"]))
            psnr_by_view[key] = float(row["psnr"])
        self.writer.write_iteration(
            spec=spec,
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_index=None if input_index is None else int(input_index),
            input_frame_id=None if input_frame_id is None else int(input_frame_id),
            local_step=int(local_step),
            render_rows=render_rows,
            psnr_by_view=psnr_by_view,
        )

    @staticmethod
    def _iter_block_visit_order(
        *,
        num_blocks: int,
        steps_per_block: int,
        block_order: str,
        step_major_switch_interval_steps: int,
    ) -> List[int]:
        return iter_block_visit_order(
            num_blocks=int(num_blocks),
            steps_per_block=int(steps_per_block),
            block_order=str(block_order),
            step_major_switch_interval_steps=int(step_major_switch_interval_steps),
        )

    def _build_target_frame_order(
        self,
        *,
        spec: TestEpisodeSpec,
        input_frame_ids: List[int],
        current_block_idx: int,
        observed_block_order: List[int],
        visited_blocks: Set[int],
    ) -> List[int]:
        policy = str(self.runtime_cfg.target_frame_policy).strip()
        cur = int(current_block_idx)
        if cur < 0 or cur >= len(input_frame_ids):
            raise ValueError(f"current_block_idx out of range: {cur} vs len(input_frame_ids)={len(input_frame_ids)}")
        source_frame = int(input_frame_ids[cur])
        max_targets = self.runtime_cfg.max_target_frames_including_source
        if max_targets is not None:
            max_targets = int(max_targets)
            if max_targets < 1:
                raise ValueError("runtime.max_target_frames_including_source must be >= 1 when set")

        if policy == "all_observed":
            others = [int(input_frame_ids[b]) for b in observed_block_order if int(b) != int(cur)]
            out = [int(source_frame)] + others
            if max_targets is not None:
                out = out[: int(max_targets)]
            return out

        if policy == "visited_episode_frames":
            limit = len(input_frame_ids) if max_targets is None else int(max_targets)
            if limit < 1:
                limit = 1
            prev_blocks = sorted([int(b) for b in visited_blocks if int(b) < int(cur)], reverse=True)
            next_blocks = sorted([int(b) for b in visited_blocks if int(b) > int(cur)])
            selected_blocks: List[int] = []
            for b in prev_blocks:
                if len(selected_blocks) >= int(limit - 1):
                    break
                selected_blocks.append(int(b))
            for b in next_blocks:
                if len(selected_blocks) >= int(limit - 1):
                    break
                selected_blocks.append(int(b))
            return [int(source_frame)] + [int(input_frame_ids[b]) for b in selected_blocks]

        if policy == "scheduler_v7_block_window":
            if max_targets is None:
                raise ValueError(
                    "runtime.target_frame_policy='scheduler_v7_block_window' requires "
                    "runtime.max_target_frames_including_source."
                )
            limit = int(max_targets)
            source_offset = int(spec.input_offsets[int(cur)])
            if int(source_frame) != int(spec.frame_ids[int(source_offset)]):
                raise ValueError(
                    "scheduler_v7_block_window requires input_offsets to point at input_frame_ids; "
                    f"block={cur} input_offset={source_offset} source_frame={source_frame}"
                )
            end = int(source_offset) + int(limit)
            if end > len(spec.frame_ids):
                raise ValueError(
                    "scheduler_v7_block_window target range exceeds episode frame window: "
                    f"source_offset={source_offset} limit={limit} window_len={len(spec.frame_ids)}"
                )
            return [int(x) for x in spec.frame_ids[int(source_offset) : int(end)]]

        raise ValueError(
            "unsupported runtime.target_frame_policy="
            f"{policy!r}, expected one of "
            "['all_observed', 'visited_episode_frames', 'scheduler_v7_block_window']"
        )

    def _generic_scheduler_aligned_info(
        self,
        *,
        spec: TestEpisodeSpec,
        block_idx: int,
        block_repeat_step: int,
        segment_local_step: int,
        target_frames: List[int],
        source_image_refs: List[ImageRef],
        target_image_refs: List[ImageRef],
        visited_blocks: Set[int],
    ) -> Dict[str, Any]:
        policy = str(self.runtime_cfg.target_frame_policy).strip()
        if policy == "scheduler_v7_block_window":
            scheduler_version = "v7"
        elif policy == "visited_episode_frames":
            scheduler_version = "v8"
        else:
            scheduler_version = "batcheval"
        block_idx_global = int(spec.episode_idx) * int(max(len(spec.input_frame_ids), 1)) + int(block_idx)
        return {
            "epoch_idx": 0,
            "global_step": int(segment_local_step),
            "scene_id": int(spec.scene_id),
            "segment_id": int(spec.segment_id),
            "segment_local_step": int(segment_local_step),
            "segment_local_u": int(segment_local_step),
            "segment_step_budget": int(len(spec.input_frame_ids) * int(self.protocol.steps_per_input)),
            "segment_budget_u": int(len(spec.input_frame_ids) * int(self.protocol.steps_per_input)),
            "block_idx_in_segment": int(block_idx_global),
            "block_idx_global": int(block_idx_global),
            "block_idx_in_episode": int(block_idx),
            "source_frame_idx": int(spec.input_frame_ids[int(block_idx)]),
            "source_keyframe_idx": -1,
            "source_cam_idx": int(source_image_refs[0][1]) if source_image_refs else -1,
            "source_image_ref": tuple(source_image_refs[0]) if source_image_refs else (-1, -1),
            "source_image_refs": [(int(f), int(c)) for f, c in source_image_refs],
            "target_frame_indices": [int(x) for x in target_frames],
            "target_image_refs": [(int(f), int(c)) for f, c in target_image_refs],
            "target_policy": str(policy),
            "visited_block_indices": sorted(int(x) for x in visited_blocks),
            "block_current_source_frame_indices": [int(x) for x in spec.input_frame_ids],
            "U": 1,
            "K_u_nominal": int(self.protocol.steps_per_input),
            "K_u_effective": int(self.protocol.steps_per_input),
            "K_steps_effective": int(self.protocol.steps_per_input),
            "K_steps": int(self.protocol.steps_per_input),
            "R_steps": 0,
            "T_steps": int(self.protocol.steps_per_input),
            "episode_idx_global": int(spec.episode_idx),
            "block_repeat_step": int(block_repeat_step),
            "block_order": str(self.runtime_cfg.block_order),
            "step_major_switch_interval_steps": int(self.runtime_cfg.step_major_switch_interval_steps),
            "episode_step_cursor": int(segment_local_step),
            "scheduler_version": str(scheduler_version),
        }

    @staticmethod
    def _nearby_offsets_adjacent_non_input(spec: TestEpisodeSpec, input_offset: int) -> List[int]:
        return nearby_offsets_adjacent_non_input(spec, int(input_offset))

    @staticmethod
    def _expand_roles(frame_roles: List[str], camera_ids: List[int]) -> List[str]:
        return [str(role) for role in frame_roles for _ in camera_ids]

    def _nearby_frames_for_block(
        self,
        *,
        spec: TestEpisodeSpec,
        input_offset: int,
        existing_target_frames: List[int],
        input_frame_ids: List[int],
    ) -> List[int]:
        policy = str(self.runtime_cfg.stage5_6_nearby_policy).strip()
        if policy != "adjacent_non_input":
            raise ValueError(
                "unsupported stage5_6 nearby_policy="
                f"{policy!r}, expected 'adjacent_non_input'"
            )
        off_to_frame = {int(off): int(frame) for off, frame in zip(spec.frame_offsets, spec.frame_ids)}
        # Skip if a candidate is already any of:
        # - input frame (even if not currently in target list due to max_target cap)
        # - source / visited targets already scheduled for this update
        existing = set(int(x) for x in existing_target_frames) | set(int(x) for x in input_frame_ids)
        out: List[int] = []
        for off in self._nearby_offsets_adjacent_non_input(spec, int(input_offset)):
            if off not in off_to_frame:
                continue
            frame_id = int(off_to_frame[off])
            if frame_id in existing:
                continue
            if frame_id in out:
                continue
            out.append(int(frame_id))
        if len(out) < 2 and not bool(self.runtime_cfg.stage5_6_allow_partial_nearby):
            return []
        return out

    def _build_stage5_6_eval_train_batch(
        self,
        *,
        spec: TestEpisodeSpec,
        block_idx: int,
        block_repeat_step: int,
        segment_local_step: int,
        visited_blocks: Set[int],
    ) -> Dict[str, Any]:
        return build_stage5_6_eval_train_batch(
            dataset=self.dataset,
            spec=spec,
            block_idx=int(block_idx),
            block_repeat_step=int(block_repeat_step),
            segment_local_step=int(segment_local_step),
            visited_blocks=set(int(x) for x in visited_blocks),
            device=self.device,
            update_camera_ids=self.runtime_cfg.update_camera_ids,
            protocol_name=str(self.protocol.name),
            steps_per_input=int(self.protocol.steps_per_input),
            target_frame_policy=str(self.runtime_cfg.target_frame_policy),
            max_target_frames_including_source=self.runtime_cfg.max_target_frames_including_source,
            nearby_policy=str(self.runtime_cfg.stage5_6_nearby_policy),
            nearby_role_name=str(self.runtime_cfg.stage5_6_nearby_role_name),
            allow_partial_nearby=bool(self.runtime_cfg.stage5_6_allow_partial_nearby),
            block_order=str(self.runtime_cfg.block_order),
            step_major_switch_interval_steps=int(self.runtime_cfg.step_major_switch_interval_steps),
            enable_nearby_feedback=bool(self.runtime_cfg.stage5_6_enable_nearby_feedback),
        )

    def _runtime_policy(self, *, do_train: bool):
        return stage5_6_runtime_policy(
            do_train=bool(do_train),
            update_hidden_state=bool(self.runtime_cfg.update_hidden_state),
            update_node_state=bool(self.runtime_cfg.update_node_state),
            force_eval_mode=False,
        )

    def _runtime_policy_for_eval_sparse(self):
        from models.streetforward.minimal_trainer_stage4_3 import RuntimePolicy

        return RuntimePolicy(
            do_backward=False,
            do_optimizer_step=False,
            update_hidden_cache=bool(self.runtime_cfg.update_hidden_state),
            writeback_node_state=bool(self.runtime_cfg.update_node_state),
            reset_node_state_after_block=False,
            force_eval_mode=True,
        )

    def _reset_model_for_episode(self, init_update_batch: Dict[str, Any]) -> None:
        if hasattr(self.model, "reset_for_segment_eval"):
            self.model.reset_for_segment_eval(init_update_batch)
        elif hasattr(self.model, "reset_node_state"):
            self.model.reset_node_state()
        else:
            raise ValueError("model lacks reset_for_segment_eval()/reset_node_state().")

    def _record_model_history(self, update_batch: Dict[str, Any]) -> None:
        if hasattr(self.model, "eval_sparse_record_history"):
            self.model.eval_sparse_record_history(update_batch)
        elif hasattr(self.model, "record_block_history"):
            self.model.record_block_history(update_batch)
        else:
            raise ValueError("model lacks eval_sparse_record_history()/record_block_history().")

    def _run_eval_sparse_update_step(
        self,
        update_batch: Dict[str, Any],
        *,
        local_step: int,
        segment_local_step: int,
    ) -> Dict[str, Any]:
        if hasattr(self.model, "eval_sparse_update_step"):
            return self.model.eval_sparse_update_step(
                update_batch,
                local_iter=int(local_step - 1),
                num_local_iters=int(self.protocol.steps_per_input),
                amp=bool(self.runtime_cfg.amp),
                update_node_state=bool(self.runtime_cfg.update_node_state),
                update_hidden_state=bool(self.runtime_cfg.update_hidden_state),
                update_view_transient=bool(self.runtime_cfg.update_view_transient),
                update_step_norm_ema=bool(self.runtime_cfg.update_step_norm_ema),
            )
        if not hasattr(self.model, "inference_step_from_train_batch"):
            raise ValueError("model lacks eval_sparse_update_step()/inference_step_from_train_batch().")
        with torch.no_grad():
            return self.model.inference_step_from_train_batch(
                update_batch,
                step=None,
                scheduler_node_sync={
                    "U": 1,
                    "segment_local_step": int(segment_local_step),
                    "reset_after_block": False,
                },
                runtime_policy=self._runtime_policy_for_eval_sparse(),
            )

    def _run_stage5_6_update_step(
        self,
        update_batch: Dict[str, Any],
        *,
        segment_local_step: int,
    ) -> Dict[str, Any]:
        return run_stage5_6_update_step(
            model=self.model,
            update_batch=update_batch,
            mode=self._runtime_mode(),
            segment_local_step=int(segment_local_step),
            update_hidden_state=bool(self.runtime_cfg.update_hidden_state),
            update_node_state=bool(self.runtime_cfg.update_node_state),
        )

    def _render_scene_pack_from_current_state(self, update_batch: Dict[str, Any]) -> Any:
        if not hasattr(self.model, "_render_scene_views_from_current_state"):
            raise ValueError("SkyBranch eval requires model._render_scene_views_from_current_state().")
        source_views = list(update_batch.get("source_views") or [])
        source_images = list(update_batch.get("source_images") or [])
        targets = list(update_batch.get("targets") or [])
        if len(source_views) == 0 or len(source_images) == 0 or len(targets) == 0:
            raise ValueError("SkyBranch eval requires source_views/source_images and targets in update batch.")
        source_frame_idx = int(update_batch.get("source_frame_idx", 0))
        source_items = [
            {
                "view": view,
                "gt_image": img,
                "frame_idx": int(source_frame_idx),
                "cam_idx": int(i),
            }
            for i, (view, img) in enumerate(zip(source_views, source_images))
        ]
        target_items = [
            {
                "view": t["view"],
                "gt_image": t["gt_image"],
                "frame_idx": int(t.get("frame_idx", source_frame_idx)),
                "cam_idx": int(t.get("cam_idx", -1)),
            }
            for t in targets
        ]
        with torch.no_grad():
            if hasattr(self.model, "ensure_runtime_state_from_batch"):
                self.model.ensure_runtime_state_from_batch(update_batch)
            source_rgb, source_alpha = self.model._render_scene_views_from_current_state(update_batch, source_items)
            target_rgb, target_alpha = self.model._render_scene_views_from_current_state(update_batch, target_items)
        return SimpleNamespace(
            source_rgb=source_rgb.detach(),
            source_alpha=source_alpha.detach(),
            target_rgb=target_rgb.detach(),
            target_alpha=target_alpha.detach(),
        )

    def _run_sky_update_after_scene_step(self, update_batch: Dict[str, Any]) -> None:
        if self.sky_branch is None:
            return
        scene_pack = self._render_scene_pack_from_current_state(update_batch)
        with torch.no_grad():
            self.sky_branch.forward_scene_batch(update_batch, scene_pack, writeback=True)

    def run_episode(self, spec: TestEpisodeSpec) -> Dict[str, Any]:
        if len(spec.input_frame_ids) == 0:
            raise ValueError(f"episode {spec.episode_uid} has empty input_frame_ids")
        if bool(self.runtime_cfg.reset_state_per_episode):
            init_source_refs = make_refs_for_frames(
                frame_ids=[int(spec.input_frame_ids[0])],
                camera_ids=[int(x) for x in spec.camera_ids],
            )
            init_update_batch = build_update_batch_from_refs(
                dataset=self.dataset,
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                source_image_refs=init_source_refs,
                update_target_image_refs=init_source_refs,
                observed_frame_ids=[int(spec.input_frame_ids[0])],
                camera_ids=[int(x) for x in spec.camera_ids],
                protocol_name=str(self.protocol.name),
                device=self.device,
            )
            self._reset_model_for_episode(init_update_batch)
        if self.sky_branch is not None and bool(self.runtime_cfg.sky_reset_state_per_episode):
            self.sky_branch.reset_runtime_state()

        global_iter = 0
        if bool(self.protocol.save_pre_update):
            pre_rows = self._render_current_state(spec=spec)
            self._record_iteration(
                spec=spec,
                global_iter=int(global_iter),
                is_pre_update=True,
                input_index=None,
                input_frame_id=None,
                local_step=0,
                render_rows=pre_rows,
            )

        block_visit_order = self._iter_block_visit_order(
            num_blocks=int(len(spec.input_frame_ids)),
            steps_per_block=int(self.protocol.steps_per_input),
            block_order=str(self.runtime_cfg.block_order),
            step_major_switch_interval_steps=int(self.runtime_cfg.step_major_switch_interval_steps),
        )
        if len(block_visit_order) == 0:
            return self.metric_acc.finalize_episode(spec)

        local_step_by_block = [0 for _ in range(int(len(spec.input_frame_ids)))]
        observed_block_order: List[int] = []
        observed_block_set: Set[int] = set()
        visited_blocks: Set[int] = set()

        total_visits = int(len(block_visit_order))
        for visit_idx, input_index in enumerate(block_visit_order):
            if int(input_index) not in observed_block_set:
                observed_block_set.add(int(input_index))
                observed_block_order.append(int(input_index))

            input_frame_id = int(spec.input_frame_ids[int(input_index)])
            local_step_by_block[int(input_index)] = int(local_step_by_block[int(input_index)]) + 1
            local_step = int(local_step_by_block[int(input_index)])

            target_frame_order = self._build_target_frame_order(
                spec=spec,
                input_frame_ids=[int(x) for x in spec.input_frame_ids],
                current_block_idx=int(input_index),
                observed_block_order=[int(x) for x in observed_block_order],
                visited_blocks=set(int(x) for x in visited_blocks),
            )
            current_observed_frames = sorted(
                set(int(spec.input_frame_ids[int(b)]) for b in set(int(x) for x in visited_blocks) | {int(input_index)})
            )
            if str(self.runtime_cfg.target_frame_policy).strip() == "scheduler_v7_block_window":
                current_observed_frames = [int(x) for x in target_frame_order]

            global_iter += 1
            if self._uses_stage5_6_train_batch_path():
                update_batch = self._build_stage5_6_eval_train_batch(
                    spec=spec,
                    block_idx=int(input_index),
                    block_repeat_step=int(local_step),
                    segment_local_step=int(global_iter),
                    visited_blocks=set(int(x) for x in visited_blocks),
                )
                self._run_stage5_6_update_step(update_batch, segment_local_step=int(global_iter))
                self._run_sky_update_after_scene_step(update_batch)
            else:
                source_refs = make_refs_for_frames(
                    frame_ids=[int(input_frame_id)],
                    camera_ids=[int(x) for x in spec.camera_ids],
                )
                update_target_refs = make_refs_for_frames(
                    frame_ids=[int(x) for x in target_frame_order],
                    camera_ids=[int(x) for x in spec.camera_ids],
                )
                update_batch = build_update_batch_from_refs(
                    dataset=self.dataset,
                    scene_id=int(spec.scene_id),
                    segment_id=int(spec.segment_id),
                    source_image_refs=source_refs,
                    update_target_image_refs=update_target_refs,
                    observed_frame_ids=[int(x) for x in current_observed_frames],
                    camera_ids=[int(x) for x in spec.camera_ids],
                    protocol_name=str(self.protocol.name),
                    device=self.device,
                    enforce_target0_equals_source=True,
                )
                aligned = self._generic_scheduler_aligned_info(
                    spec=spec,
                    block_idx=int(input_index),
                    block_repeat_step=int(local_step),
                    segment_local_step=int(global_iter),
                    target_frames=[int(x) for x in target_frame_order],
                    source_image_refs=[(int(f), int(c)) for f, c in source_refs],
                    target_image_refs=[(int(f), int(c)) for f, c in update_target_refs],
                    visited_blocks=set(int(x) for x in visited_blocks),
                )
                update_batch["_scheduler_v4_aligned_info"] = dict(aligned)
                update_batch["_scheduler_v7_aligned_info"] = dict(aligned)
                if str(aligned.get("scheduler_version")) == "v8":
                    update_batch["_scheduler_v8_aligned_info"] = dict(aligned)
                if bool(self.runtime_cfg.no_grad):
                    self._run_eval_sparse_update_step(
                        update_batch,
                        local_step=int(local_step),
                        segment_local_step=int(global_iter),
                    )
                else:
                    self.model.train_step(
                        update_batch,
                        step=None,
                        profile_phase_timing=False,
                        sync_cuda_timing=False,
                        scheduler_node_sync={
                            "U": 1,
                            "segment_local_step": int(global_iter),
                            "reset_after_block": False,
                        },
                    )
                self._run_sky_update_after_scene_step(update_batch)

            next_block = int(block_visit_order[int(visit_idx + 1)]) if int(visit_idx + 1) < int(total_visits) else None
            is_block_exit = (next_block is None) or (int(next_block) != int(input_index))

            if bool(self.runtime_cfg.history_record_each_step):
                self._record_model_history(update_batch)
            elif bool(self.runtime_cfg.history_record_on_input_exit) and bool(is_block_exit):
                self._record_model_history(update_batch)

            report_iterations = self.protocol.report_iterations
            should_render = (
                bool(self.protocol.save_each_iter_views)
                or (
                    report_iterations is not None
                    and int(global_iter) in set(int(x) for x in report_iterations)
                )
                or (int(global_iter) == int(total_visits))
            )
            if should_render:
                rows = self._render_current_state(spec=spec)
                self._record_iteration(
                    spec=spec,
                    global_iter=int(global_iter),
                    is_pre_update=False,
                    input_index=int(input_index),
                    input_frame_id=int(input_frame_id),
                    local_step=int(local_step),
                    render_rows=rows,
                )

            reset_policy = str(self.runtime_cfg.reset_policy).strip()
            is_episode_end = int(visit_idx + 1) >= int(total_visits)
            should_reset_now = (
                (reset_policy == "block_end" and bool(is_block_exit))
                or (reset_policy == "episode_end" and bool(is_episode_end))
            )
            if should_reset_now and hasattr(self.model, "reset_node_state"):
                self.model.reset_node_state()
            visited_blocks.add(int(input_index))

        return self.metric_acc.finalize_episode(spec)
