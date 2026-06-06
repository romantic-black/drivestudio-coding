from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .random_window_batch import (
    RANDOM_WINDOW_ASSEMBLY_MODE,
    RANDOM_WINDOW_MODEL_FAMILY,
    RANDOM_WINDOW_SCHEDULER_VERSION,
)
from .resolver import (
    ImageRef,
    IForwardBatchResolver,
    IForwardResolvedBatch,
    IForwardResolvedStep,
    _ref_key,
    _ref_list,
    _require_order_matches,
    _resolve_step_block_clock,
    _tensor_ref_list,
)


class IForwardRandomWindowBatchResolver(IForwardBatchResolver):
    def __init__(self, *, expected_cams_per_step: Optional[int] = None) -> None:
        super().__init__(
            expected_scheduler_version=RANDOM_WINDOW_SCHEDULER_VERSION,
            expected_model_family=RANDOM_WINDOW_MODEL_FAMILY,
            expected_cams_per_step=expected_cams_per_step,
            current_role="current_latest",
            nearby_role="nearby",
        )

    @staticmethod
    def _indices_for_refs(
        *,
        refs: List[ImageRef],
        target_ref_to_index: Dict[ImageRef, int],
        name: str,
        allow_empty: bool = True,
    ) -> Tuple[int, ...]:
        if not refs and not bool(allow_empty):
            raise ValueError(f"IForward random-window requires non-empty {name}.")
        out: List[int] = []
        for ref in refs:
            item = (int(ref[0]), int(ref[1]))
            if item not in target_ref_to_index:
                raise ValueError(f"IForward random-window {name} ref {item} missing from target refs.")
            out.append(int(target_ref_to_index[item]))
        return tuple(out)

    def resolve(self, batch: Dict[str, Any]) -> IForwardResolvedBatch:
        ifwd = self._extract_iforward_meta(batch)
        request_meta = dict(batch.get("request_meta") or {})
        scheduler_version = str(ifwd.get("scheduler_version", request_meta.get("scheduler_version", "")))
        if scheduler_version != RANDOM_WINDOW_SCHEDULER_VERSION:
            raise ValueError(
                f"IForward random-window requires scheduler_version={RANDOM_WINDOW_SCHEDULER_VERSION!r}, got {scheduler_version!r}."
            )
        model_family = str(ifwd.get("model_family", request_meta.get("model_family", RANDOM_WINDOW_MODEL_FAMILY)))
        if model_family != RANDOM_WINDOW_MODEL_FAMILY:
            raise ValueError(f"IForward random-window requires model_family={RANDOM_WINDOW_MODEL_FAMILY!r}.")
        assembly_mode = str(request_meta.get("assembly_mode", ifwd.get("assembly_mode", "")))
        if assembly_mode != RANDOM_WINDOW_ASSEMBLY_MODE:
            raise ValueError(f"IForward random-window requires assembly_mode={RANDOM_WINDOW_ASSEMBLY_MODE}.")

        source_refs = self._source_refs_from_meta(batch, ifwd)
        target_refs, target_roles = self._target_refs_roles_from_meta(batch, ifwd)
        _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source_image_refs")
        _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target_image_refs")

        source_ref_to_index = {tuple(ref): int(idx) for idx, ref in enumerate(source_refs)}
        target_ref_to_index = {tuple(ref): int(idx) for idx, ref in enumerate(target_refs)}
        target_indices_by_role: Dict[str, List[int]] = {}
        for idx, role in enumerate(target_roles):
            target_indices_by_role.setdefault(str(role), []).append(int(idx))

        steps_raw = list(ifwd.get("steps") or [])
        inner_k = int(ifwd.get("inner_K", request_meta.get("inner_K", len(steps_raw))))
        if inner_k != 8:
            raise ValueError(f"IForward random-window expects fixed inner_K=8, got {inner_k}.")
        if len(steps_raw) != inner_k:
            raise ValueError(f"IForward random-window len(steps) must equal inner_K: {len(steps_raw)} vs {inner_k}.")

        resolved_steps: List[IForwardResolvedStep] = []
        for k, raw_step in enumerate(steps_raw):
            if not isinstance(raw_step, Mapping):
                raise ValueError(f"IForward random-window steps[{k}] must be a mapping.")
            step = dict(raw_step)
            step_idx = int(step.get("step_idx", step.get("global_k", k)))
            if step_idx != k:
                raise ValueError(f"IForward random-window steps[{k}].step_idx must equal {k}.")
            refs = tuple(_ref_list(step.get("evidence_refs"), f"steps[{k}].evidence_refs"))
            frames = {int(ref[0]) for ref in refs}
            if len(frames) != 1:
                raise ValueError(f"IForward random-window steps[{k}] must contain exactly one source frame.")
            source_frame_idx = int(step.get("source_frame_idx", next(iter(frames))))
            if frames != {source_frame_idx}:
                raise ValueError(f"IForward random-window steps[{k}] evidence refs do not match source_frame_idx.")
            cams = tuple(sorted(int(ref[1]) for ref in refs))
            if self.expected_cams_per_step is not None:
                expected_cams = tuple(range(int(self.expected_cams_per_step)))
                if cams != expected_cams:
                    raise ValueError(f"IForward random-window expected cams {expected_cams}, got {cams}.")
            repeat_idx = int(step.get("repeat_idx", 0))
            commit = bool(step.get("commit_observation_memory", repeat_idx == 0))
            if commit != (repeat_idx == 0):
                raise ValueError("IForward random-window commit_observation_memory must be true only on repeat_idx=0.")
            if not bool(step.get("update_optimizer_memory", True)):
                raise ValueError("IForward random-window update_optimizer_memory must be true for every repeat.")
            rollout_block_rank = int(step.get("block_pos_in_window", step.get("rollout_block_rank", 0)))
            next_step = dict(steps_raw[k + 1]) if k + 1 < len(steps_raw) and isinstance(steps_raw[k + 1], Mapping) else None
            block_clock = _resolve_step_block_clock(
                step=step,
                next_step=next_step,
                ifwd=ifwd,
                request_meta=request_meta,
                repeat_idx=int(repeat_idx),
                rollout_block_rank=int(rollout_block_rank),
                source_frame_idx=int(source_frame_idx),
            )
            source_indices = []
            for ref in refs:
                if tuple(ref) not in source_ref_to_index:
                    raise ValueError(f"IForward random-window source ref {tuple(ref)} missing from source refs.")
                source_indices.append(int(source_ref_to_index[tuple(ref)]))
            resolved_steps.append(
                IForwardResolvedStep(
                    step_idx=int(step_idx),
                    source_frame_idx=int(source_frame_idx),
                    repeat_idx=int(repeat_idx),
                    rollout_block_rank=int(rollout_block_rank),
                    source_indices=tuple(source_indices),
                    evidence_refs=refs,
                    commit_observation_memory=bool(commit),
                    update_optimizer_memory=True,
                    rollout_pos_code=float(step.get("rollout_pos_code", 0.0)),
                    frame_pos_code=float(step.get("frame_pos_code", 0.0)),
                    repeat_pos_code=float(step.get("repeat_pos_code", 0.0)),
                    block_id=int(block_clock["block_id"]),
                    episode_block_idx=int(block_clock["episode_block_idx"]),
                    repeats_per_block=int(block_clock["repeats_per_block"]),
                    is_block_enter=bool(block_clock["is_block_enter"]),
                    is_block_exit=bool(block_clock["is_block_exit"]),
                    is_frame_exit=bool(block_clock["is_block_exit"]),
                    record_update_norm=bool(step.get("record_update_norm", True)),
                    commit_support_on_exit=bool(step.get("commit_support_on_exit", block_clock["is_block_exit"])),
                    commit_residual_on_exit=bool(step.get("commit_residual_on_exit", block_clock["is_block_exit"])),
                )
            )

        current_latest_refs = _ref_list(ifwd.get("current_latest_refs"), "current_latest_refs", allow_empty=False)
        in_rollout_history_refs = _ref_list(ifwd.get("in_rollout_history_refs"), "in_rollout_history_refs", allow_empty=False)
        short_window_refs = _ref_list(ifwd.get("short_window_history_refs"), "short_window_history_refs", allow_empty=True)
        nearby_refs_list = _ref_list(ifwd.get("nearby_refs"), "nearby_refs", allow_empty=True)
        current_latest_indices = self._indices_for_refs(
            refs=current_latest_refs,
            target_ref_to_index=target_ref_to_index,
            name="current_latest_refs",
            allow_empty=False,
        )
        history_indices = self._indices_for_refs(
            refs=in_rollout_history_refs,
            target_ref_to_index=target_ref_to_index,
            name="in_rollout_history_refs",
            allow_empty=False,
        )
        short_indices = self._indices_for_refs(
            refs=short_window_refs,
            target_ref_to_index=target_ref_to_index,
            name="short_window_history_refs",
            allow_empty=True,
        )
        nearby_indices = self._indices_for_refs(
            refs=nearby_refs_list,
            target_ref_to_index=target_ref_to_index,
            name="nearby_refs",
            allow_empty=True,
        )
        evidence_set = {tuple(ref) for step in resolved_steps for ref in step.evidence_refs}
        if evidence_set & {target_refs[int(idx)] for idx in nearby_indices}:
            raise ValueError("IForward random-window nearby refs leaked into evidence refs.")

        keyed = dict(ifwd.get("source_ref_to_index_keyed") or {})
        for ref in source_refs:
            key = _ref_key(ref)
            if key in keyed and int(keyed[key]) != int(source_ref_to_index[ref]):
                raise ValueError(f"IForward random-window source_ref_to_index_keyed mismatch for ref={ref}.")

        input_frames = tuple(int(x) for x in list(ifwd.get("input_frame_indices") or []))
        if len(input_frames) != 4:
            raise ValueError("IForward random-window requires exactly 4 input_frame_indices.")
        role_tuples = {str(role): tuple(int(x) for x in indices) for role, indices in target_indices_by_role.items()}
        history_commit = tuple(list(history_indices) + list(current_latest_indices))
        return IForwardResolvedBatch(
            raw=batch,
            meta=ifwd,
            scheduler_version=RANDOM_WINDOW_SCHEDULER_VERSION,
            scene_id=int(ifwd.get("scene_id", request_meta.get("scene_id", batch.get("scene_id", -1)))),
            segment_id=int(ifwd.get("segment_id", request_meta.get("segment_id", batch.get("segment_id", -1)))),
            episode_id=int(ifwd.get("episode_id", request_meta.get("episode_id", -1))),
            rollout_id_global=int(ifwd.get("rollout_id_global", request_meta.get("rollout_id_global", -1))),
            rollout_idx_in_episode=int(ifwd.get("rollout_idx_in_episode", request_meta.get("rollout_idx_in_episode", -1))),
            inner_K=int(inner_k),
            steps=tuple(resolved_steps),
            source_refs=tuple(source_refs),
            target_refs=tuple(target_refs),
            target_roles=tuple(target_roles),
            current_target_indices=tuple(current_latest_indices),
            input_frame_indices=tuple(input_frames),
            latest_input_frame_idx=int(input_frames[-1]),
            current_latest_target_indices=tuple(current_latest_indices),
            history_rollout_target_indices=tuple(history_indices),
            nearby_target_indices=tuple(nearby_indices),
            target_indices_by_role=role_tuples,
            reset_scene_state_before_rollout=bool(ifwd.get("reset_scene_state_before_rollout", False)),
            carry_scene_state_after_rollout=bool(ifwd.get("carry_scene_state_after_rollout", True)),
            episode_end_after_rollout=bool(ifwd.get("episode_end_after_rollout", False)),
            detach_graph_after_rollout=bool(ifwd.get("detach_graph_after_rollout", True)),
            rollouts_per_episode=int(ifwd.get("rollouts_per_episode", request_meta.get("rollouts_per_episode", 1))),
            window_start=int(ifwd.get("window_start", request_meta.get("window_start", -1))),
            window_end=int(ifwd.get("window_end", request_meta.get("window_end", -1))),
            window_block_ids=tuple(int(x) for x in list(ifwd.get("window_block_ids", request_meta.get("window_block_ids", [])) or [])),
            window_hash=int(ifwd.get("window_hash", request_meta.get("window_hash", -1))),
            window_revisit_count=int(ifwd.get("window_revisit_count", request_meta.get("window_revisit_count", 0))),
            unique_windows_seen=int(ifwd.get("unique_windows_seen", request_meta.get("unique_windows_seen", 0))),
            is_repeated_window=bool(ifwd.get("is_repeated_window", request_meta.get("is_repeated_window", False))),
            history_commit_target_indices=tuple(history_commit),
            short_window_history_target_indices=tuple(short_indices),
        )


__all__ = ["IForwardRandomWindowBatchResolver"]
