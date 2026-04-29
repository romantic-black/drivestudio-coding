from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Protocol, Tuple

ImageRef = Tuple[int, int]


class SegmentIndexLike(Protocol):
    frame_indices: List[int]
    train_frame_set: FrozenSet[int]
    num_cams: int


class EvalSparseDatasetLike(Protocol):
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...


@dataclass(frozen=True)
class EvalSparseEpisodeSpec:
    scene_id: int
    segment_id: int
    episode_idx: int
    window_start_pos: int
    frames20: List[int]
    input_offsets: List[int]
    input_frames: List[int]
    eval_frames: List[int]
    camera_ids: List[int]
    eval_image_refs: List[ImageRef]


@dataclass(frozen=True)
class EvalSparseStepSpec:
    episode_idx: int
    step_idx: int
    source_offset: int
    source_frame: int
    target_frames: List[int]
    source_image_refs: List[ImageRef]
    target_image_refs: List[ImageRef]
    target_frame_roles: List[str]
    target_image_roles: List[str]


def _make_refs(frames: List[int], camera_ids: List[int]) -> List[ImageRef]:
    return [(int(f), int(c)) for f in frames for c in camera_ids]


def _select_window_starts(
    *,
    num_frames: int,
    sequence_length: int,
    policy: str,
    stride: int,
) -> List[int]:
    if num_frames < sequence_length:
        return []

    if policy == "middle_per_segment":
        return [(num_frames - sequence_length) // 2]

    if policy == "all_nonoverlap":
        return list(range(0, num_frames - sequence_length + 1, sequence_length))

    if policy == "stride":
        step = int(stride)
        if step < 1:
            raise ValueError("eval_sparse_scheduler.episode_selection.stride must be >= 1 for policy=stride")
        return list(range(0, num_frames - sequence_length + 1, step))

    raise ValueError(f"Unsupported eval sparse window_policy={policy!r}")


def build_eval_sparse_episode_specs(
    *,
    dataset: EvalSparseDatasetLike,
    scene_ids: List[int],
    sequence_length: int,
    input_offsets: List[int],
    camera_ids: List[int],
    window_policy: str,
    stride: int,
    max_episodes_per_scene: Optional[int],
    max_total_episodes: Optional[int],
) -> List[EvalSparseEpisodeSpec]:
    specs: List[EvalSparseEpisodeSpec] = []

    if int(sequence_length) != 20:
        raise ValueError("Primary eval_sparse_scheduler protocol expects sequence_length=20.")

    canonical_offsets = [0, 5, 10, 15]
    norm_offsets = [int(x) for x in input_offsets]
    if norm_offsets != canonical_offsets:
        raise ValueError("Primary eval_sparse_scheduler protocol expects input_offsets=[0,5,10,15].")

    for scene_id in [int(x) for x in scene_ids]:
        scene_specs: List[EvalSparseEpisodeSpec] = []
        for segment_id in sorted(int(x) for x in dataset.list_segment_ids(scene_id)):
            sidx = dataset.get_segment_index(scene_id, segment_id)
            all_frames = [int(x) for x in sorted(sidx.frame_indices)]
            train_frame_set_any = getattr(sidx, "train_frame_set", None)
            if train_frame_set_any is not None:
                train_frame_set = set(int(x) for x in train_frame_set_any)
                frames = [int(x) for x in all_frames if int(x) in train_frame_set]
            else:
                frames = all_frames
            if len(frames) < int(sequence_length):
                continue

            starts = _select_window_starts(
                num_frames=len(frames),
                sequence_length=int(sequence_length),
                policy=str(window_policy),
                stride=int(stride),
            )
            for st in starts:
                frames20 = frames[int(st) : int(st) + int(sequence_length)]
                if len(frames20) != int(sequence_length):
                    continue

                for cam_id in [int(x) for x in camera_ids]:
                    if cam_id < 0 or cam_id >= int(sidx.num_cams):
                        raise ValueError(
                            f"camera_id={cam_id} out of range for "
                            f"scene={scene_id}, segment={segment_id}, num_cams={sidx.num_cams}"
                        )

                input_frames = [int(frames20[int(o)]) for o in norm_offsets]
                eval_frames = [int(x) for x in frames20]
                cams = [int(x) for x in camera_ids]
                scene_specs.append(
                    EvalSparseEpisodeSpec(
                        scene_id=int(scene_id),
                        segment_id=int(segment_id),
                        episode_idx=-1,
                        window_start_pos=int(st),
                        frames20=eval_frames,
                        input_offsets=list(norm_offsets),
                        input_frames=input_frames,
                        eval_frames=eval_frames,
                        camera_ids=cams,
                        eval_image_refs=_make_refs(eval_frames, cams),
                    )
                )

        if max_episodes_per_scene is not None:
            scene_specs = scene_specs[: int(max_episodes_per_scene)]
        specs.extend(scene_specs)

        if max_total_episodes is not None and len(specs) >= int(max_total_episodes):
            specs = specs[: int(max_total_episodes)]
            break

    out: List[EvalSparseEpisodeSpec] = []
    for idx, spec in enumerate(specs):
        out.append(
            EvalSparseEpisodeSpec(
                scene_id=int(spec.scene_id),
                segment_id=int(spec.segment_id),
                episode_idx=int(idx),
                window_start_pos=int(spec.window_start_pos),
                frames20=[int(x) for x in spec.frames20],
                input_offsets=[int(x) for x in spec.input_offsets],
                input_frames=[int(x) for x in spec.input_frames],
                eval_frames=[int(x) for x in spec.eval_frames],
                camera_ids=[int(x) for x in spec.camera_ids],
                eval_image_refs=[(int(x[0]), int(x[1])) for x in spec.eval_image_refs],
            )
        )
    return out


def build_eval_sparse_steps(
    *,
    episode: EvalSparseEpisodeSpec,
    total_target_frames: int,
    include_source_frame: bool,
    history_order: str = "recent_first",
) -> List[EvalSparseStepSpec]:
    if int(total_target_frames) < 1:
        raise ValueError("eval_sparse_scheduler.update.total_target_frames must be >= 1")
    if not bool(include_source_frame):
        raise ValueError("eval_sparse_scheduler requires include_source_frame=true")

    visited_input_frames: List[int] = []
    steps: List[EvalSparseStepSpec] = []
    cams = [int(x) for x in episode.camera_ids]

    for step_idx, source_frame_any in enumerate(episode.input_frames):
        source_frame = int(source_frame_any)
        if history_order == "recent_first":
            hist = list(reversed(visited_input_frames))
        elif history_order == "oldest_first":
            hist = list(visited_input_frames)
        else:
            raise ValueError(f"Unsupported history_order={history_order!r}")

        target_frames = [int(source_frame)] + [int(x) for x in hist]
        target_frames = target_frames[: int(total_target_frames)]

        frame_roles = ["source"] + ["visited" for _ in target_frames[1:]]
        source_refs = _make_refs([int(source_frame)], cams)
        target_refs = _make_refs(target_frames, cams)

        image_roles: List[str] = []
        for role in frame_roles:
            image_roles.extend([str(role) for _ in cams])

        steps.append(
            EvalSparseStepSpec(
                episode_idx=int(episode.episode_idx),
                step_idx=int(step_idx),
                source_offset=int(episode.input_offsets[step_idx]),
                source_frame=int(source_frame),
                target_frames=[int(x) for x in target_frames],
                source_image_refs=source_refs,
                target_image_refs=target_refs,
                target_frame_roles=[str(x) for x in frame_roles],
                target_image_roles=[str(x) for x in image_roles],
            )
        )
        visited_input_frames.append(int(source_frame))

    return steps

