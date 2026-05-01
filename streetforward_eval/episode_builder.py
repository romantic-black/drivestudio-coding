from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol, Tuple

from .protocols import TestProtocolSpec, resolve_eval_offsets

ImageRef = Tuple[int, int]


class SegmentIndexLike(Protocol):
    frame_indices: List[int]
    train_frame_set: set[int]
    num_cams: int


class EvalDatasetLike(Protocol):
    def list_segment_ids(self, scene_id: int) -> List[int]: ...

    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...


@dataclass(frozen=True)
class TestEpisodeSpec:
    exp_name: str
    scene_id: int
    segment_id: int
    episode_idx: int

    sequence_start_pos: int
    frame_offsets: List[int]
    frame_ids: List[int]

    input_offsets: List[int]
    eval_offsets: List[int]
    input_frame_ids: List[int]
    eval_frame_ids: List[int]

    camera_ids: List[int]
    camera_names: List[str]

    input_image_refs: List[ImageRef]
    eval_image_refs: List[ImageRef]

    episode_uid: str


def _make_refs(frames: List[int], camera_ids: List[int]) -> List[ImageRef]:
    return [(int(f), int(cam)) for f in frames for cam in camera_ids]


def _window_starts(
    *,
    num_frames: int,
    sequence_length: int,
    window_policy: str,
    stride: int,
    require_full_window: bool,
) -> List[int]:
    if int(sequence_length) < 1:
        raise ValueError("sequence_length must be >= 1")
    if int(num_frames) == 0:
        return []
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")

    if str(window_policy) == "middle":
        if require_full_window and int(num_frames) < int(sequence_length):
            return []
        start = max(0, (int(num_frames) - int(sequence_length)) // 2)
        return [int(start)]

    if str(window_policy) == "sliding":
        if require_full_window:
            if int(num_frames) < int(sequence_length):
                return []
            return list(range(0, int(num_frames) - int(sequence_length) + 1, int(stride)))
        return list(range(0, int(num_frames), int(stride)))

    raise ValueError(
        f"unsupported window_policy={window_policy!r}, expected one of ['sliding','middle']"
    )


def _filter_segment_ids(segment_ids: List[int], segment_policy: str) -> List[int]:
    out = [int(x) for x in segment_ids]
    if str(segment_policy) == "all":
        return sorted(out)
    if str(segment_policy) == "first":
        out = sorted(out)
        return out[:1]
    raise ValueError(f"unsupported segment_policy={segment_policy!r}, expected one of ['all','first']")


def build_test_episode_specs(
    *,
    dataset: EvalDatasetLike,
    scene_ids: List[int],
    protocol: TestProtocolSpec,
    segment_policy: str,
    window_policy: str,
    stride: int,
    require_full_window: bool,
    max_episodes_per_scene: Optional[int],
    max_total_episodes: Optional[int],
) -> List[TestEpisodeSpec]:
    specs: List[TestEpisodeSpec] = []
    eval_offsets = resolve_eval_offsets(protocol.eval_offsets, sequence_length=int(protocol.sequence_length))
    episode_idx = 0

    for scene_id_any in scene_ids:
        scene_id = int(scene_id_any)
        scene_count = 0
        raw_segment_ids = [int(x) for x in dataset.list_segment_ids(scene_id)]
        segment_ids = _filter_segment_ids(raw_segment_ids, segment_policy=str(segment_policy))
        for segment_id in segment_ids:
            sidx = dataset.get_segment_index(scene_id, segment_id)
            num_cams = int(sidx.num_cams)
            for cam_id in protocol.camera_ids:
                if int(cam_id) < 0 or int(cam_id) >= int(num_cams):
                    raise ValueError(
                        f"camera id out of range for scene={scene_id}, segment={segment_id}: "
                        f"cam_id={cam_id}, num_cams={num_cams}"
                    )

            all_frames = [int(x) for x in sorted(sidx.frame_indices)]
            train_frame_set_any = getattr(sidx, "train_frame_set", None)
            if train_frame_set_any is not None:
                train_set = set(int(x) for x in train_frame_set_any)
                frames = [int(f) for f in all_frames if int(f) in train_set]
            else:
                frames = all_frames
            if len(frames) == 0:
                continue

            starts = _window_starts(
                num_frames=len(frames),
                sequence_length=int(protocol.sequence_length),
                window_policy=str(window_policy),
                stride=int(stride),
                require_full_window=bool(require_full_window),
            )
            for start_pos in starts:
                end_pos = int(start_pos) + int(protocol.sequence_length)
                window_frames = [int(x) for x in frames[int(start_pos) : int(end_pos)]]
                if len(window_frames) < int(protocol.sequence_length) and bool(require_full_window):
                    continue

                frame_offsets = list(range(len(window_frames)))
                if len(window_frames) == 0:
                    continue

                input_offsets: List[int] = []
                for off_any in protocol.input_offsets:
                    off = int(off_any)
                    if off < 0 or off >= len(window_frames):
                        raise ValueError(
                            f"input offset {off} out of range for window len={len(window_frames)} "
                            f"(scene={scene_id}, segment={segment_id}, start={start_pos})"
                        )
                    input_offsets.append(int(off))

                mapped_eval_offsets: List[int] = []
                for off in eval_offsets:
                    if int(off) < 0 or int(off) >= len(window_frames):
                        raise ValueError(
                            f"eval offset {off} out of range for window len={len(window_frames)} "
                            f"(scene={scene_id}, segment={segment_id}, start={start_pos})"
                        )
                    mapped_eval_offsets.append(int(off))

                input_frame_ids = [int(window_frames[o]) for o in input_offsets]
                eval_frame_ids = [int(window_frames[o]) for o in mapped_eval_offsets]
                input_image_refs = _make_refs(input_frame_ids, protocol.camera_ids)
                eval_image_refs = _make_refs(eval_frame_ids, protocol.camera_ids)

                uid = (
                    f"scene{int(scene_id):03d}_seg{int(segment_id):03d}_start{int(start_pos):06d}"
                )
                specs.append(
                    TestEpisodeSpec(
                        exp_name=str(protocol.name),
                        scene_id=int(scene_id),
                        segment_id=int(segment_id),
                        episode_idx=int(episode_idx),
                        sequence_start_pos=int(start_pos),
                        frame_offsets=[int(x) for x in frame_offsets],
                        frame_ids=[int(x) for x in window_frames],
                        input_offsets=[int(x) for x in input_offsets],
                        eval_offsets=[int(x) for x in mapped_eval_offsets],
                        input_frame_ids=[int(x) for x in input_frame_ids],
                        eval_frame_ids=[int(x) for x in eval_frame_ids],
                        camera_ids=[int(x) for x in protocol.camera_ids],
                        camera_names=[str(x) for x in protocol.camera_names],
                        input_image_refs=[(int(f), int(c)) for (f, c) in input_image_refs],
                        eval_image_refs=[(int(f), int(c)) for (f, c) in eval_image_refs],
                        episode_uid=uid,
                    )
                )
                episode_idx += 1
                scene_count += 1
                if max_episodes_per_scene is not None and int(scene_count) >= int(max_episodes_per_scene):
                    break
                if max_total_episodes is not None and len(specs) >= int(max_total_episodes):
                    return specs[: int(max_total_episodes)]
            if max_episodes_per_scene is not None and int(scene_count) >= int(max_episodes_per_scene):
                break

    return specs


def classify_eval_frame(offset: int, input_offsets: List[int]) -> Literal["input", "pre_input", "interp", "extrap"]:
    x = int(offset)
    in_set = set(int(v) for v in input_offsets)
    if x in in_set:
        return "input"
    lo = min(in_set)
    hi = max(in_set)
    if x < lo:
        return "pre_input"
    if x > hi:
        return "extrap"
    return "interp"
