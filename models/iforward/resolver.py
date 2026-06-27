from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ImageRef = Tuple[int, int]

IFORWARD_SCHEDULER_VERSION = "iforward_v1"
IFORWARD_V3_SCHEDULER_VERSION = "iforward_v3_random_window"
IFORWARD_V4_SCHEDULER_VERSION = "iforward_v4_coverage_ordered"
IFORWARD_STAGE2_1_SCHEDULER_VERSION = "iforward_stage2_1_parent_temporal"
IFORWARD_SEQUENCE10_SCHEDULER_VERSION = "iforward_sequence10_v1"
IFORWARD_STAGE2_2_SCHEDULER_VERSION = "iforward_stage2_2_stream10_rawframe"
IFORWARD_STAGE2_3_SCHEDULER_VERSION = "iforward_2_3_scheduler_v3_optimizer_mamba"
IFORWARD_MODEL_FAMILY = "IForward"
IFORWARD_CURRENT_ROLE = "final_current_recon"
IFORWARD_HISTORY_ROLE = "final_history_replay"
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
    block_id: int = -1
    episode_block_idx: int = -1
    repeats_per_block: int = 0
    is_block_enter: bool = False
    is_block_exit: bool = False
    is_frame_exit: bool = False
    episode_visit_idx: int = -1
    rollout_visit_idx: int = -1
    optimizer_step_idx_in_episode: int = -1
    record_update_norm: bool = True
    commit_support_on_exit: bool = False
    commit_residual_on_exit: bool = False
    window_hash: int = -1
    window_revisit_count: int = 0
    block_visit_count_before: int = 0
    block_visit_count_after: int = 0
    sequence_pos: int = -1
    visit_kind: str = ""
    frame_gap: int = 0
    temporal_read: bool = True
    temporal_commit: bool = False
    physical_time_advance: bool = False
    scheduler_phase: str = ""
    timestamp_us: int = 0
    timestamp_sec: float = 0.0
    delta_t_sec: float = 0.0
    visit_order_gap: int = 0
    physical_frame_gap_abs: int = 0
    previous_visit_sequence_pos: int = -1
    ego_delta_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_delta_yaw: float = 0.0
    visit_memory_mask: bool = True
    repair_no_commit: bool = False
    repeat_budget: int = 1
    visit_count_for_frame: int = 0
    is_first_visit_of_frame: bool = False
    is_last_update_of_episode: bool = False
    global_update_idx_in_episode: int = -1
    optimizer_memory_read: bool = True
    optimizer_memory_write: bool = True
    time_since_same_frame_visit: float = 0.0
    source_keyframe_idx: int = -1
    validation_render_only: bool = False


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
    rollouts_per_episode: int = 1
    window_start: int = -1
    window_end: int = -1
    window_block_ids: Tuple[int, ...] = ()
    window_hash: int = -1
    window_revisit_count: int = 0
    unique_windows_seen: int = 0
    is_repeated_window: bool = False
    history_commit_target_indices: Tuple[int, ...] = ()
    short_window_history_target_indices: Tuple[int, ...] = ()

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


def _step_get_int(step: Mapping[str, Any], keys: Sequence[str], default: Optional[int] = None) -> Optional[int]:
    for key in keys:
        if key in step and step.get(key) is not None:
            return int(step.get(key))
    return None if default is None else int(default)


def _step_get_bool(step: Mapping[str, Any], keys: Sequence[str]) -> Optional[bool]:
    for key in keys:
        if key in step and step.get(key) is not None:
            return bool(step.get(key))
    return None


def _resolve_step_block_clock(
    *,
    step: Mapping[str, Any],
    next_step: Optional[Mapping[str, Any]],
    ifwd: Mapping[str, Any],
    request_meta: Mapping[str, Any],
    repeat_idx: int,
    rollout_block_rank: int,
    source_frame_idx: int,
) -> Dict[str, Any]:
    block_id = int(
        _step_get_int(
            step,
            ("block_id", "episode_block_idx", "block_pos_in_window", "rollout_block_rank"),
            default=int(rollout_block_rank),
        )
    )
    episode_block_idx = int(_step_get_int(step, ("episode_block_idx", "block_id", "block_pos_in_window"), default=block_id))
    repeats_raw = _step_get_int(step, ("repeats_per_block",), default=None)
    if repeats_raw is None:
        repeats_raw = _step_get_int(ifwd, ("repeats_per_block",), default=None)
    if repeats_raw is None:
        repeats_raw = _step_get_int(request_meta, ("repeats_per_block",), default=0)
    repeats_per_block = int(max(int(repeats_raw or 0), 0))

    enter = _step_get_bool(step, ("is_block_enter",))
    is_block_enter = bool(int(repeat_idx) == 0) if enter is None else bool(enter)

    exit_flag = _step_get_bool(step, ("is_block_exit", "is_frame_exit"))
    if exit_flag is not None:
        is_block_exit = bool(exit_flag)
    elif repeats_per_block > 0:
        is_block_exit = bool(int(repeat_idx) == int(repeats_per_block) - 1)
    elif next_step is None:
        is_block_exit = True
    else:
        next_block_id = _step_get_int(
            next_step,
            ("block_id", "episode_block_idx", "block_pos_in_window", "rollout_block_rank"),
            default=None,
        )
        if next_block_id is not None:
            is_block_exit = bool(int(next_block_id) != int(block_id))
        else:
            next_source_frame_idx = int(next_step.get("source_frame_idx", source_frame_idx))
            is_block_exit = bool(int(next_source_frame_idx) != int(source_frame_idx))

    return {
        "block_id": int(block_id),
        "episode_block_idx": int(episode_block_idx),
        "repeats_per_block": int(repeats_per_block),
        "is_block_enter": bool(is_block_enter),
        "is_block_exit": bool(is_block_exit),
    }


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
        history_role: str = IFORWARD_HISTORY_ROLE,
        nearby_role: str = IFORWARD_NEARBY_ROLE,
    ) -> None:
        self.expected_scheduler_version = str(expected_scheduler_version)
        self.expected_model_family = str(expected_model_family)
        self.expected_cams_per_step = None if expected_cams_per_step is None else int(expected_cams_per_step)
        self.current_role = str(current_role)
        self.history_role = str(history_role)
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
        allowed_versions = {
            self.expected_scheduler_version,
            IFORWARD_V3_SCHEDULER_VERSION,
            IFORWARD_V4_SCHEDULER_VERSION,
            IFORWARD_STAGE2_1_SCHEDULER_VERSION,
            IFORWARD_SEQUENCE10_SCHEDULER_VERSION,
            IFORWARD_STAGE2_2_SCHEDULER_VERSION,
            IFORWARD_STAGE2_3_SCHEDULER_VERSION,
        }
        if scheduler_version not in allowed_versions:
            raise ValueError(
                f"IForward requires scheduler_version in {sorted(allowed_versions)!r}, got {scheduler_version!r}."
            )
        is_v3 = scheduler_version == IFORWARD_V3_SCHEDULER_VERSION
        is_v4 = scheduler_version == IFORWARD_V4_SCHEDULER_VERSION
        is_stage2_1 = scheduler_version == IFORWARD_STAGE2_1_SCHEDULER_VERSION
        is_stage2_2 = scheduler_version == IFORWARD_STAGE2_2_SCHEDULER_VERSION
        is_stage2_3 = scheduler_version == IFORWARD_STAGE2_3_SCHEDULER_VERSION
        is_sequence10 = scheduler_version == IFORWARD_SEQUENCE10_SCHEDULER_VERSION
        is_explicit_iforward = bool(is_v3 or is_v4 or is_stage2_1 or is_stage2_2 or is_stage2_3 or is_sequence10)
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
            no_commit_visit = str(step.get("visit_kind", "")) in {"bootstrap", "repair", "stress", "final_all"}
            stage23_bootstrap_no_commit = bool(is_stage2_3 and str(step.get("visit_kind", "")) == "bootstrap")
            stage23_final_all_no_commit = bool(is_stage2_3 and bool(step.get("validation_render_only", False)))
            expected_commit = False if (
                (bool(is_sequence10 or is_stage2_2) and bool(no_commit_visit))
                or stage23_bootstrap_no_commit
                or stage23_final_all_no_commit
            ) else (int(repeat_idx) == 0)
            if commit != bool(expected_commit):
                raise ValueError("IForward commit_observation_memory must be true only on repeat_idx=0.")
            rollout_block_rank = int(step.get("rollout_block_rank", 0))
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
            if bool(is_sequence10):
                expected_update = bool(step.get("update_optimizer_memory", False))
            elif bool(is_stage2_3):
                expected_update = bool(step.get("optimizer_memory_write", step.get("update_optimizer_memory", False)))
            elif bool(is_stage2_2) and bool(no_commit_visit):
                expected_update = False
            else:
                expected_update = bool(block_clock["is_block_exit"]) if bool(is_stage2_1 or is_stage2_2) else True
            if bool(step.get("update_optimizer_memory", True)) != bool(expected_update):
                if bool(is_stage2_3):
                    raise ValueError("IForward Stage2_3 update_optimizer_memory must match optimizer_memory_write.")
                if bool(is_stage2_1 or is_stage2_2):
                    raise ValueError("IForward Stage2 update_optimizer_memory must be true only on block exit.")
                raise ValueError("IForward update_optimizer_memory must be true for every repeat.")
            is_frame_exit = bool(step.get("is_frame_exit", block_clock["is_block_exit"]))
            episode_visit_idx = int(step.get("episode_visit_idx", -1))
            rollout_visit_idx = int(step.get("rollout_visit_idx", step.get("rollout_block_rank", rollout_block_rank)))
            optimizer_step_idx = int(step.get("optimizer_step_idx_in_episode", -1))
            if is_explicit_iforward:
                missing = [
                    name
                    for name in (
                        "block_id",
                        "episode_block_idx",
                        "is_block_enter",
                        "is_block_exit",
                        "is_frame_exit",
                        "episode_visit_idx",
                        "optimizer_step_idx_in_episode",
                    )
                    if name not in step
                ]
                if missing:
                    raise ValueError(f"IForward {scheduler_version} step requires explicit fields: {missing}")
                if episode_visit_idx < 0 or optimizer_step_idx < 0:
                    raise ValueError(f"IForward {scheduler_version} requires non-negative visit and optimizer clocks.")
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
                    rollout_block_rank=int(rollout_block_rank),
                    source_indices=tuple(source_indices),
                    evidence_refs=refs,
                    commit_observation_memory=commit,
                    update_optimizer_memory=bool(step.get("update_optimizer_memory", True)),
                    rollout_pos_code=float(step.get("rollout_pos_code", 0.0)),
                    frame_pos_code=float(step.get("frame_pos_code", 0.0)),
                    repeat_pos_code=float(step.get("repeat_pos_code", 0.0)),
                    block_id=int(block_clock["block_id"]),
                    episode_block_idx=int(block_clock["episode_block_idx"]),
                    repeats_per_block=int(block_clock["repeats_per_block"]),
                    is_block_enter=bool(block_clock["is_block_enter"]),
                    is_block_exit=bool(block_clock["is_block_exit"]),
                    is_frame_exit=bool(is_frame_exit),
                    episode_visit_idx=int(episode_visit_idx),
                    rollout_visit_idx=int(rollout_visit_idx),
                    optimizer_step_idx_in_episode=int(optimizer_step_idx),
                    record_update_norm=bool(step.get("record_update_norm", True)),
                    commit_support_on_exit=bool(step.get("commit_support_on_exit", block_clock["is_block_exit"])),
                    commit_residual_on_exit=bool(step.get("commit_residual_on_exit", block_clock["is_block_exit"])),
                    window_hash=int(step.get("window_hash", ifwd.get("window_hash", request_meta.get("window_hash", -1)))),
                    window_revisit_count=int(
                        step.get("window_revisit_count", ifwd.get("window_revisit_count", request_meta.get("window_revisit_count", 0)))
                    ),
                    block_visit_count_before=int(step.get("block_visit_count_before", 0)),
                    block_visit_count_after=int(step.get("block_visit_count_after", 0)),
                    sequence_pos=int(step.get("sequence_pos", -1)),
                    visit_kind=str(step.get("visit_kind", "")),
                    frame_gap=int(step.get("frame_gap", 0)),
                    temporal_read=bool(step.get("temporal_read", True)),
                    temporal_commit=bool(step.get("temporal_commit", False)),
                    physical_time_advance=bool(step.get("physical_time_advance", False)),
                    scheduler_phase=str(step.get("scheduler_phase", "")),
                    timestamp_us=int(step.get("timestamp_us", 0)),
                    timestamp_sec=float(step.get("timestamp_sec", float(step.get("timestamp_us", 0)) / 1.0e6)),
                    delta_t_sec=float(step.get("delta_t_sec", 0.0)),
                    visit_order_gap=int(step.get("visit_order_gap", 0)),
                    physical_frame_gap_abs=int(step.get("physical_frame_gap_abs", abs(int(step.get("frame_gap", 0))))),
                    previous_visit_sequence_pos=int(step.get("previous_visit_sequence_pos", -1)),
                    ego_delta_translation=tuple(
                        float(x)
                        for x in (
                            list(step.get("ego_delta_translation", (0.0, 0.0, 0.0))) + [0.0, 0.0, 0.0]
                        )[:3]
                    ),
                    ego_delta_yaw=float(step.get("ego_delta_yaw", 0.0)),
                    visit_memory_mask=bool(step.get("visit_memory_mask", True)),
                    repair_no_commit=bool(step.get("repair_no_commit", False)),
                    repeat_budget=int(step.get("repeat_budget", block_clock["repeats_per_block"] or 1)),
                    visit_count_for_frame=int(step.get("visit_count_for_frame", 0)),
                    is_first_visit_of_frame=bool(step.get("is_first_visit_of_frame", False)),
                    is_last_update_of_episode=bool(step.get("is_last_update_of_episode", False)),
                    global_update_idx_in_episode=int(
                        step.get("global_update_idx_in_episode", step.get("optimizer_step_idx_in_episode", optimizer_step_idx))
                    ),
                    optimizer_memory_read=bool(step.get("optimizer_memory_read", step.get("temporal_read", True))),
                    optimizer_memory_write=bool(
                        step.get("optimizer_memory_write", step.get("temporal_commit", step.get("update_optimizer_memory", True)))
                    ),
                    time_since_same_frame_visit=float(step.get("time_since_same_frame_visit", 0.0)),
                    source_keyframe_idx=int(step.get("source_keyframe_idx", source_frame_idx)),
                    validation_render_only=bool(step.get("validation_render_only", False)),
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
            int(x)
            for x in list(
                ifwd.get("input_frame_indices")
                or final_supervision.get("current_frames")
                or final_supervision.get("current_input_frames")
                or []
            )
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
        if is_explicit_iforward:
            history_rollout_indices = tuple(int(x) for x in target_indices_by_role.get(self.history_role, []))
            current_latest_indices = tuple(int(x) for x in current_indices)
            current_refs = {target_refs[int(idx)] for idx in current_indices}
            history_refs = {target_refs[int(idx)] for idx in history_rollout_indices}
            if current_refs & history_refs:
                raise ValueError(f"IForward {scheduler_version} history refs must be disjoint from current refs.")
        else:
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
            rollouts_per_episode=int(ifwd.get("rollouts_per_episode", request_meta.get("rollouts_per_episode", 1))),
            window_start=int(ifwd.get("window_start", request_meta.get("window_start", -1))),
            window_end=int(ifwd.get("window_end", request_meta.get("window_end", -1))),
            window_block_ids=tuple(int(x) for x in list(ifwd.get("window_block_ids", request_meta.get("window_block_ids", [])) or [])),
            window_hash=int(ifwd.get("window_hash", request_meta.get("window_hash", -1))),
            window_revisit_count=int(ifwd.get("window_revisit_count", request_meta.get("window_revisit_count", 0))),
            unique_windows_seen=int(ifwd.get("unique_windows_seen", request_meta.get("unique_windows_seen", 0))),
            is_repeated_window=bool(ifwd.get("is_repeated_window", request_meta.get("is_repeated_window", False))),
            history_commit_target_indices=tuple(() if is_explicit_iforward else current_indices),
        )
