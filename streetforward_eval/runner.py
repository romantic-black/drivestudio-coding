from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

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
        if not bool(self.runtime_cfg.no_grad):
            raise NotImplementedError(
                "runtime.no_grad=false is not supported by production eval interfaces; set batch_eval.runtime.no_grad=true."
            )
        self._validate_model_interface()

    def _validate_model_interface(self) -> None:
        required = [
            "reset_for_segment_eval",
            "eval_sparse_update_step",
            "eval_sparse_render_frames",
        ]
        for name in required:
            if not hasattr(self.model, name):
                raise ValueError(
                    f"BatchEval requires model.{name}(). "
                    "Please use production trainer class with sparse eval interfaces."
                )
        if self.runtime_cfg.history_record_on_input_exit and not hasattr(self.model, "eval_sparse_record_history"):
            raise ValueError(
                "history_record_on_input_exit is enabled but model lacks eval_sparse_record_history()."
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
        self.writer.write_iteration(
            spec=spec,
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_index=None if input_index is None else int(input_index),
            input_frame_id=None if input_frame_id is None else int(input_frame_id),
            local_step=int(local_step),
            render_rows=render_rows,
        )
        self.metric_acc.add_iteration_rows(
            spec=spec,
            global_iter=int(global_iter),
            is_pre_update=bool(is_pre_update),
            input_index=None if input_index is None else int(input_index),
            input_frame_id=None if input_frame_id is None else int(input_frame_id),
            local_step=int(local_step),
            render_rows=render_rows,
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

        observed_frames: List[int] = []
        for input_index, input_frame_id_any in enumerate(spec.input_frame_ids):
            input_frame_id = int(input_frame_id_any)
            observed_frames.append(int(input_frame_id))
            source_refs = make_refs_for_frames(
                frame_ids=[int(input_frame_id)],
                camera_ids=[int(x) for x in spec.camera_ids],
            )
            # Keep current source frame first in targets while only using observed frames.
            target_frame_order = [int(input_frame_id)] + [
                int(x) for x in observed_frames if int(x) != int(input_frame_id)
            ]
            update_target_refs = make_refs_for_frames(
                frame_ids=target_frame_order,
                camera_ids=[int(x) for x in spec.camera_ids],
            )

            update_batch = build_update_batch_from_refs(
                dataset=self.dataset,
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                source_image_refs=source_refs,
                update_target_image_refs=update_target_refs,
                observed_frame_ids=[int(x) for x in observed_frames],
                camera_ids=[int(x) for x in spec.camera_ids],
                protocol_name=str(self.protocol.name),
                device=self.device,
            )
            for local_step in range(1, int(self.protocol.steps_per_input) + 1):
                global_iter += 1
                grad_ctx = torch.no_grad() if bool(self.runtime_cfg.no_grad) else contextlib.nullcontext()
                with grad_ctx:
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
                should_render = bool(self.protocol.save_each_iter_views) or (
                    int(input_index) == int(len(spec.input_frame_ids) - 1)
                    and int(local_step) == int(self.protocol.steps_per_input)
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
            if bool(self.runtime_cfg.history_record_on_input_exit):
                self.model.eval_sparse_record_history(update_batch)

        return self.metric_acc.finalize_episode(spec)
