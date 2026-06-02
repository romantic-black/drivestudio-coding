from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ImageRef = Tuple[int, int]

IFORWARD_SCHEDULER_VERSION = "iforward_v1"
IFORWARD_MODEL_FAMILY = "IForward"
IFORWARD_CURRENT_ROLE = "final_current_recon"
IFORWARD_NEARBY_ROLE = "final_nearby_rollout"


@dataclass(frozen=True)
class IForwardResolvedStep:
    step_idx: int
    source_frame_idx: int
    repeat_idx: int
    rollout_block_rank: int
    source_indices: Tuple[int, ...]
    evidence_refs: Tuple[ImageRef, ...]
    commit_observation_memory: bool
    update_optimizer_memory: bool
    rollout_pos_code: float = 0.0
    frame_pos_code: float = 0.0
    repeat_pos_code: float = 0.0


@dataclass(frozen=True)
class IForwardResolvedBatch:
    raw: Dict[str, Any]
    meta: Dict[str, Any]
    scheduler_version: str
    scene_id: int
    segment_id: int
    episode_id: int
    rollout_id_global: int
    rollout_idx_in_episode: int
    inner_K: int
    steps: Tuple[IForwardResolvedStep, ...]
    source_refs: Tuple[ImageRef, ...]
    target_refs: Tuple[ImageRef, ...]
    target_roles: Tuple[str, ...]
    current_target_indices: Tuple[int, ...]
    input_frame_indices: Tuple[int, ...]
    latest_input_frame_idx: int
    current_latest_target_indices: Tuple[int, ...]
    history_rollout_target_indices: Tuple[int, ...]
    nearby_target_indices: Tuple[int, ...]
    target_indices_by_role: Dict[str, Tuple[int, ...]]
    reset_scene_state_before_rollout: bool
    carry_scene_state_after_rollout: bool
    episode_end_after_rollout: bool
    detach_graph_after_rollout: bool

    @property
    def cache_key(self) -> Tuple[int, int, int]:
        return int(self.scene_id), int(self.segment_id), int(self.episode_id)


def _as_ref(raw: Any) -> ImageRef:
    if torch.is_tensor(raw):
        vals = raw.detach().cpu().reshape(-1).tolist()
    else:
        vals = list(raw)
    if len(vals) != 2:
        raise ValueError(f"image ref must have 2 values, got {raw!r}")
    return int(vals[0]), int(vals[1])


def _ref_list(raw: Any, name: str, *, allow_empty: bool = False) -> List[ImageRef]:
    if raw is None:
        if allow_empty:
            return []
        raise ValueError(f"IForward requires {name}.")
    out = [_as_ref(ref) for ref in list(raw)]
    if not allow_empty and len(out) == 0:
        raise ValueError(f"IForward requires non-empty {name}.")
    return out


def _tensor_ref_list(role_batch: Any) -> Optional[List[ImageRef]]:
    if not isinstance(role_batch, dict):
        return None
    frames = role_batch.get("frame_indices")
    cams = role_batch.get("cam_indices")
    if frames is None or cams is None:
        return None
    frame_vals = frames.detach().cpu().reshape(-1).tolist() if hasattr(frames, "detach") else list(frames)
    cam_vals = cams.detach().cpu().reshape(-1).tolist() if hasattr(cams, "detach") else list(cams)
    if len(frame_vals) != len(cam_vals):
        raise ValueError("IForward source/target frame_indices and cam_indices length mismatch.")
    return [(int(f), int(c)) for f, c in zip(frame_vals, cam_vals)]


def _require_order_matches(actual: Optional[List[ImageRef]], expected: Sequence[ImageRef], name: str) -> None:
    if actual is None:
        return
    if len(actual) != len(expected):
        raise ValueError(f"IForward {name} length mismatch: {len(actual)} vs {len(expected)}")
    for idx, (got, want) in enumerate(zip(actual, expected)):
        if tuple(got) != tuple(want):
            raise ValueError(
                f"IForward {name} order/content mismatch at index {idx}: got {tuple(got)} expected {tuple(want)}"
            )


def _ref_key(ref: ImageRef) -> str:
    return f"{int(ref[0])}:{int(ref[1])}"


def _flat_refs(groups: Iterable[Iterable[ImageRef]]) -> List[ImageRef]:
    return [tuple(ref) for group in groups for ref in group]


class IForwardBatchResolver:
    """Resolve and validate the scheduler_iforward batch contract.

    IForward v1 requires one frame with all three cameras per optimizer step.
    The materializer may flatten many rollout frames into `source_*`, but this
    resolver hands each step only its own 3-camera source indices.
    """

    def __init__(
        self,
        *,
        expected_scheduler_version: str = IFORWARD_SCHEDULER_VERSION,
        expected_model_family: str = IFORWARD_MODEL_FAMILY,
        expected_cams_per_step: Optional[int] = 3,
        current_role: str = IFORWARD_CURRENT_ROLE,
        nearby_role: str = IFORWARD_NEARBY_ROLE,
    ) -> None:
        self.expected_scheduler_version = str(expected_scheduler_version)
        self.expected_model_family = str(expected_model_family)
        self.expected_cams_per_step = None if expected_cams_per_step is None else int(expected_cams_per_step)
        self.current_role = str(current_role)
        self.nearby_role = str(nearby_role)

    @staticmethod
    def _extract_iforward_meta(batch: Mapping[str, Any]) -> Dict[str, Any]:
        raw = batch.get("_iforward")
        request_meta = dict(batch.get("request_meta") or {})
        if raw is None:
            raw = request_meta.get("iforward")
        if raw is None:
            raise ValueError("IForward batch requires batch['_iforward'] or request_meta.iforward.")
        if not isinstance(raw, Mapping):
            raise ValueError(f"IForward metadata must be a mapping, got {type(raw)!r}.")
        return dict(raw)

    @staticmethod
    def _source_refs_from_meta(batch: Mapping[str, Any], ifwd: Mapping[str, Any]) -> List[ImageRef]:
        request_meta = dict(batch.get("request_meta") or {})
        source = _ref_list(
            request_meta.get("source_image_refs", ifwd.get("evidence_refs_flat")),
            "source_image_refs/evidence_refs_flat",
        )
        if request_meta.get("source_image_refs") is not None and ifwd.get("evidence_refs_flat") is not None:
            if source != _ref_list(ifwd.get("evidence_refs_flat"), "evidence_refs_flat"):
                raise ValueError("IForward request_meta.source_image_refs disagrees with _iforward.evidence_refs_flat.")
        return source

    @staticmethod
    def _target_refs_roles_from_meta(batch: Mapping[str, Any], ifwd: Mapping[str, Any]) -> Tuple[List[ImageRef], List[str]]:
        request_meta = dict(batch.get("request_meta") or {})
        target_refs = _ref_list(
            request_meta.get("target_image_refs", ifwd.get("target_refs_flat")),
            "target_image_refs/target_refs_flat",
        )
        target_roles = [str(x) for x in list(request_meta.get("target_image_roles", ifwd.get("target_roles_flat")) or [])]
        if len(target_refs) != len(target_roles):
            raise ValueError("IForward target_image_refs and target_image_roles length mismatch.")
        if request_meta.get("target_image_refs") is not None and ifwd.get("target_refs_flat") is not None:
            if target_refs != _ref_list(ifwd.get("target_refs_flat"), "target_refs_flat"):
                raise ValueError("IForward request_meta.target_image_refs disagrees with _iforward.target_refs_flat.")
        if request_meta.get("target_image_roles") is not None and ifwd.get("target_roles_flat") is not None:
            roles_ifwd = [str(x) for x in list(ifwd.get("target_roles_flat") or [])]
            if target_roles != roles_ifwd:
                raise ValueError("IForward request_meta.target_image_roles disagrees with _iforward.target_roles_flat.")
        return target_refs, target_roles

    def _resolve_source_indices(
        self,
        *,
        step_idx: int,
        step: Mapping[str, Any],
        source_ref_to_index: Dict[ImageRef, int],
    ) -> Tuple[int, ...]:
        refs = tuple(_ref_list(step.get("evidence_refs"), f"steps[{step_idx}].evidence_refs"))
        raw_indices = step.get("source_indices")
        if raw_indices is not None:
            indices = tuple(int(x) for x in list(raw_indices))
            if len(indices) != len(refs):
                raise ValueError(
                    f"IForward steps[{step_idx}].source_indices length mismatch: {len(indices)} vs {len(refs)}"
                )
            expected = tuple(int(source_ref_to_index[tuple(ref)]) for ref in refs)
            if tuple(indices) != expected:
                raise ValueError(
                    f"IForward steps[{step_idx}].source_indices do not match evidence_refs order: "
                    f"got {indices} expected {expected}"
                )
            return indices
        out: List[int] = []
        for ref in refs:
            if ref not in source_ref_to_index:
                raise ValueError(f"IForward evidence ref {tuple(ref)} missing from source_image_refs.")
            out.append(int(source_ref_to_index[ref]))
        return tuple(out)

    def resolve(self, batch: Dict[str, Any]) -> IForwardResolvedBatch:
        ifwd = self._extract_iforward_meta(batch)
        request_meta = dict(batch.get("request_meta") or {})
        scheduler_version = str(ifwd.get("scheduler_version", request_meta.get("scheduler_version", "")))
        if scheduler_version != self.expected_scheduler_version:
            raise ValueError(
                f"IForward requires scheduler_version={self.expected_scheduler_version!r}, got {scheduler_version!r}."
            )
        model_family = str(ifwd.get("model_family", request_meta.get("model_family", self.expected_model_family)))
        if model_family != self.expected_model_family:
            raise ValueError(f"IForward requires model_family={self.expected_model_family!r}, got {model_family!r}.")
        assembly_mode = str(request_meta.get("assembly_mode", "image_ref_iforward_v1"))
        if assembly_mode != "image_ref_iforward_v1":
            raise ValueError("IForward requires request_meta.assembly_mode=image_ref_iforward_v1.")

        source_refs = self._source_refs_from_meta(batch, ifwd)
        target_refs, target_roles = self._target_refs_roles_from_meta(batch, ifwd)
        _require_order_matches(_tensor_ref_list(batch.get("source")), source_refs, "source_image_refs")
        _require_order_matches(_tensor_ref_list(batch.get("target")), target_refs, "target_image_refs")

        source_ref_to_index = {tuple(ref): int(idx) for idx, ref in enumerate(source_refs)}
        target_indices_by_role: Dict[str, List[int]] = {}
        for idx, role in enumerate(target_roles):
            target_indices_by_role.setdefault(str(role), []).append(int(idx))

        steps_raw = list(ifwd.get("steps") or [])
        inner_k = int(ifwd.get("inner_K", request_meta.get("inner_K", len(steps_raw))))
        if inner_k < 1:
            raise ValueError("IForward inner_K must be >= 1.")
        if len(steps_raw) != inner_k:
            raise ValueError(f"IForward len(steps) must equal inner_K: {len(steps_raw)} vs {inner_k}.")

        resolved_steps: List[IForwardResolvedStep] = []
        for k, step_raw in enumerate(steps_raw):
            if not isinstance(step_raw, Mapping):
                raise ValueError(f"IForward steps[{k}] must be a mapping.")
            step = dict(step_raw)
            step_idx = int(step.get("step_idx", k))
            if step_idx != k:
                raise ValueError(f"IForward steps[{k}].step_idx must equal {k}, got {step_idx}.")
            refs = tuple(_ref_list(step.get("evidence_refs"), f"steps[{k}].evidence_refs"))
            frames = {int(ref[0]) for ref in refs}
            if len(frames) != 1:
                raise ValueError(f"IForward steps[{k}] must contain exactly one source frame.")
            source_frame_idx = int(step.get("source_frame_idx", next(iter(frames))))
            if frames != {source_frame_idx}:
                raise ValueError(f"IForward steps[{k}].evidence_refs do not match source_frame_idx.")
            cams = tuple(sorted(int(ref[1]) for ref in refs))
            if self.expected_cams_per_step is not None:
                expected_cams = tuple(range(int(self.expected_cams_per_step)))
                if cams != expected_cams:
                    raise ValueError(
                        f"IForward v1 requires one frame with cams {expected_cams} per step, got cams={cams}."
                    )
            if bool(step.get("detach_before_step", False)) or bool(step.get("detach_after_step", False)):
                raise ValueError("IForward forbids detach inside rollout.")
            if bool(step.get("allow_step_render_loss", False)) or list(step.get("step_loss_refs") or []):
                raise ValueError("IForward v1 requires rollout-final render loss only.")
            repeat_idx = int(step.get("repeat_idx", 0))
            commit = bool(step.get("commit_observation_memory", repeat_idx == 0))
            if commit != (repeat_idx == 0):
                raise ValueError("IForward commit_observation_memory must be true only on repeat_idx=0.")
            if not bool(step.get("update_optimizer_memory", True)):
                raise ValueError("IForward update_optimizer_memory must be true for every repeat.")
            source_indices = self._resolve_source_indices(
                step_idx=k,
                step=step,
                source_ref_to_index=source_ref_to_index,
            )
            resolved_steps.append(
                IForwardResolvedStep(
                    step_idx=step_idx,
                    source_frame_idx=source_frame_idx,
                    repeat_idx=repeat_idx,
                    rollout_block_rank=int(step.get("rollout_block_rank", 0)),
                    source_indices=tuple(source_indices),
                    evidence_refs=refs,
                    commit_observation_memory=commit,
                    update_optimizer_memory=bool(step.get("update_optimizer_memory", True)),
                    rollout_pos_code=float(step.get("rollout_pos_code", 0.0)),
                    frame_pos_code=float(step.get("frame_pos_code", 0.0)),
                    repeat_pos_code=float(step.get("repeat_pos_code", 0.0)),
                )
            )

        evidence_refs = set(_flat_refs(step.evidence_refs for step in resolved_steps))
        nearby_indices = tuple(int(x) for x in target_indices_by_role.get(self.nearby_role, []))
        nearby_refs = {target_refs[int(idx)] for idx in nearby_indices}
        if evidence_refs & nearby_refs:
            raise ValueError("IForward nearby target refs leaked into evidence refs.")

        current_indices = tuple(int(x) for x in target_indices_by_role.get(self.current_role, []))
        if len(current_indices) == 0:
            raise ValueError(f"IForward final supervision missing role {self.current_role!r}.")

        final_supervision = dict(ifwd.get("final_supervision") or {})
        input_frame_indices = tuple(
            int(x) for x in list(ifwd.get("input_frame_indices") or final_supervision.get("current_input_frames") or [])
        )
        expected_input_frames = set(input_frame_indices)
        if expected_input_frames:
            actual_current_frames = {int(target_refs[idx][0]) for idx in current_indices}
            missing = sorted(expected_input_frames - actual_current_frames)
            if missing:
                raise ValueError(f"IForward current supervision missing input frames: {missing[:8]}.")
        if not input_frame_indices:
            ordered_frames = []
            seen_frames = set()
            for step in resolved_steps:
                frame = int(step.source_frame_idx)
                if frame not in seen_frames:
                    ordered_frames.append(frame)
                    seen_frames.add(frame)
            input_frame_indices = tuple(ordered_frames)
        if not input_frame_indices:
            raise ValueError("IForward requires non-empty input_frame_indices.")
        latest_input_frame_idx = int(input_frame_indices[-1])
        history_frames = set(int(x) for x in input_frame_indices[:-1])
        current_latest_indices = tuple(
            int(idx) for idx in current_indices if int(target_refs[int(idx)][0]) == latest_input_frame_idx
        )
        history_rollout_indices = tuple(
            int(idx) for idx in current_indices if int(target_refs[int(idx)][0]) in history_frames
        )
        if not current_latest_indices:
            raise ValueError(
                f"IForward current supervision missing latest input frame {int(latest_input_frame_idx)}."
            )

        keyed = dict(ifwd.get("source_ref_to_index_keyed") or {})
        for ref in source_refs:
            key = _ref_key(ref)
            if key in keyed and int(keyed[key]) != int(source_ref_to_index[ref]):
                raise ValueError(f"IForward source_ref_to_index_keyed mismatch for ref={ref}.")

        role_tuples = {str(role): tuple(int(x) for x in indices) for role, indices in target_indices_by_role.items()}
        return IForwardResolvedBatch(
            raw=batch,
            meta=ifwd,
            scheduler_version=scheduler_version,
            scene_id=int(ifwd.get("scene_id", request_meta.get("scene_id", batch.get("scene_id", -1)))),
            segment_id=int(ifwd.get("segment_id", request_meta.get("segment_id", batch.get("segment_id", -1)))),
            episode_id=int(ifwd.get("episode_id", request_meta.get("episode_id", request_meta.get("episode_idx_global", -1)))),
            rollout_id_global=int(ifwd.get("rollout_id_global", request_meta.get("rollout_id_global", -1))),
            rollout_idx_in_episode=int(ifwd.get("rollout_idx_in_episode", request_meta.get("rollout_idx_in_episode", -1))),
            inner_K=inner_k,
            steps=tuple(resolved_steps),
            source_refs=tuple(source_refs),
            target_refs=tuple(target_refs),
            target_roles=tuple(target_roles),
            current_target_indices=current_indices,
            input_frame_indices=input_frame_indices,
            latest_input_frame_idx=latest_input_frame_idx,
            current_latest_target_indices=current_latest_indices,
            history_rollout_target_indices=history_rollout_indices,
            nearby_target_indices=nearby_indices,
            target_indices_by_role=role_tuples,
            reset_scene_state_before_rollout=bool(ifwd.get("reset_scene_state_before_rollout", False)),
            carry_scene_state_after_rollout=bool(ifwd.get("carry_scene_state_after_rollout", True)),
            episode_end_after_rollout=bool(ifwd.get("episode_end_after_rollout", ifwd.get("discard_scene_state_after_rollout", False))),
            detach_graph_after_rollout=bool(ifwd.get("detach_graph_after_rollout", True)),
        )
