from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import torch

from .batch_builder import build_update_batch_from_refs, make_refs_for_frames
from .episode_builder import TestEpisodeSpec
from .protocols import TestProtocolSpec

ImageRef = Tuple[int, int]


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
    ):
        self.model = model
        self.dataset = dataset
        self.protocol = protocol
        self.writer = writer
        self.metric_acc = metric_acc
        self.device = device
        self.runtime_cfg = runtime_cfg
        self._validate_model_interface()

    def _validate_model_interface(self) -> None:
        required = [
            "reset_for_segment_eval",
            "eval_sparse_render_frames",
        ]
        if bool(self.runtime_cfg.no_grad):
            required.append("eval_sparse_update_step")
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
        if need_history_record and not hasattr(self.model, "eval_sparse_record_history"):
            raise ValueError(
                "history_record_on_input_exit/history_record_each_step is enabled but model lacks "
                "eval_sparse_record_history()."
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

    def _render_current_state(self, *, spec: TestEpisodeSpec) -> List[Dict[str, Any]]:
        grad_ctx = torch.no_grad() if bool(self.runtime_cfg.no_grad) else contextlib.nullcontext()
        with grad_ctx:
            out = self.model.eval_sparse_render_frames(
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                image_refs=[(int(f), int(c)) for f, c in spec.eval_image_refs],
                camera_ids=[int(x) for x in spec.camera_ids],
                save_dir=None,
                amp=bool(self.runtime_cfg.amp),
            )
        return list(out.get("rows", []))

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
            psnr_by_view[key] = float(row["psnr_non_sky"])
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
        if int(num_blocks) < 1:
            return []
        if int(steps_per_block) < 1:
            raise ValueError("steps_per_input must be >= 1")
        order = str(block_order).strip()
        if order == "block_major":
            return [int(b) for b in range(int(num_blocks)) for _ in range(int(steps_per_block))]
        if order == "step_major":
            switch_every = int(step_major_switch_interval_steps)
            if switch_every < 1:
                raise ValueError("step_major_switch_interval_steps must be >= 1")
            out: List[int] = []
            for round_base in range(0, int(steps_per_block), int(switch_every)):
                chunk = int(min(int(switch_every), int(steps_per_block) - int(round_base)))
                for b in range(int(num_blocks)):
                    out.extend([int(b)] * int(chunk))
            return out
        raise ValueError(f"unsupported runtime.block_order={block_order!r}, expected block_major/step_major")

    def _build_target_frame_order(
        self,
        *,
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

        raise ValueError(
            "unsupported runtime.target_frame_policy="
            f"{policy!r}, expected one of ['all_observed', 'visited_episode_frames']"
        )

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
            self.model.reset_for_segment_eval(init_update_batch)

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
                input_frame_ids=[int(x) for x in spec.input_frame_ids],
                current_block_idx=int(input_index),
                observed_block_order=[int(x) for x in observed_block_order],
                visited_blocks=set(int(x) for x in visited_blocks),
            )
            current_observed_frames = sorted(
                set(int(spec.input_frame_ids[int(b)]) for b in set(int(x) for x in visited_blocks) | {int(input_index)})
            )
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
            )

            global_iter += 1
            if bool(self.runtime_cfg.no_grad):
                with torch.no_grad():
                    self.model.eval_sparse_update_step(
                        update_batch,
                        local_iter=int(local_step - 1),
                        num_local_iters=int(self.protocol.steps_per_input),
                        amp=bool(self.runtime_cfg.amp),
                        update_node_state=bool(self.runtime_cfg.update_node_state),
                        update_hidden_state=bool(self.runtime_cfg.update_hidden_state),
                        update_view_transient=bool(self.runtime_cfg.update_view_transient),
                        update_step_norm_ema=bool(self.runtime_cfg.update_step_norm_ema),
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

            next_block = int(block_visit_order[int(visit_idx + 1)]) if int(visit_idx + 1) < int(total_visits) else None
            is_block_exit = (next_block is None) or (int(next_block) != int(input_index))

            if bool(self.runtime_cfg.history_record_each_step):
                self.model.eval_sparse_record_history(update_batch)
            elif bool(self.runtime_cfg.history_record_on_input_exit) and bool(is_block_exit):
                self.model.eval_sparse_record_history(update_batch)

            should_render = bool(self.protocol.save_each_iter_views) or (int(global_iter) == int(total_visits))
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
