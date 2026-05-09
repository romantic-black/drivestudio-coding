from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


class SegmentIndexLike(Protocol):
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    num_cams: int


class ValidationDatasetLike(Protocol):
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...


@dataclass(frozen=True)
class ValidationEpisodeSpecV7:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    frame_chain: List[int]
    block_windows: List[List[int]]
    eval_image_refs: List[Tuple[int, int]]
    num_cams: int


def build_segment_episode_starts(num_keyframes: int, blocks_per_episode: int, window_keyframes: int) -> List[int]:
    if num_keyframes < window_keyframes:
        return []
    starts = list(range(0, num_keyframes - window_keyframes + 1, blocks_per_episode))
    tail = int(num_keyframes - window_keyframes)
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _middle_frame(frames: List[int]) -> int:
    if len(frames) == 0:
        raise ValueError("keyframe_to_frames entry must not be empty")
    return int(frames[len(frames) // 2])


@dataclass(frozen=True)
class ValidationStartAtV7:
    scene_id: int
    segment_id: int
    frame_id: Optional[int] = None
    sequence_start_pos: Optional[int] = None


def _start_at_from_mapping(start_at: Optional[Dict[str, Any]]) -> Optional[ValidationStartAtV7]:
    if start_at is None:
        return None
    if not isinstance(start_at, dict):
        raise ValueError(f"batch_eval.dataset.start_at must be a mapping, got {type(start_at).__name__}")
    if start_at.get("scene_id") is None:
        raise ValueError("batch_eval.dataset.start_at.scene_id is required")
    if start_at.get("segment_id") is None:
        raise ValueError("batch_eval.dataset.start_at.segment_id is required")
    frame_id = start_at.get("frame_id")
    sequence_start_pos = start_at.get("sequence_start_pos")
    if frame_id is not None and sequence_start_pos is not None:
        raise ValueError(
            "batch_eval.dataset.start_at can specify either frame_id or sequence_start_pos, not both"
        )
    return ValidationStartAtV7(
        scene_id=int(start_at["scene_id"]),
        segment_id=int(start_at["segment_id"]),
        frame_id=None if frame_id is None else int(frame_id),
        sequence_start_pos=None if sequence_start_pos is None else int(sequence_start_pos),
    )


def _start_keyframe_pos_from_frame_id(
    *,
    sidx: SegmentIndexLike,
    frame_id: int,
    scene_id: int,
    segment_id: int,
) -> int:
    frame_id = int(frame_id)
    for pos, keyframe_idx in enumerate([int(x) for x in sidx.keyframe_indices]):
        frames = [int(x) for x in list(sidx.keyframe_to_frames[int(keyframe_idx)])]
        if frame_id in set(frames):
            return int(pos)
    preview: List[Any] = []
    for keyframe_idx in [int(x) for x in sidx.keyframe_indices[:5]]:
        preview.extend([int(x) for x in list(sidx.keyframe_to_frames[int(keyframe_idx)])[:3]])
    raise ValueError(
        "batch_eval.dataset.start_at.frame_id is not in keyframe_to_frames: "
        f"scene={int(scene_id)} segment={int(segment_id)} frame_id={frame_id} "
        f"keyframes={len(sidx.keyframe_indices)} frame_preview={preview}"
    )


def _choose_start_keyframe_pos(
    *,
    starts: List[int],
    start_at: Optional[ValidationStartAtV7],
    sidx: SegmentIndexLike,
    scene_id: int,
    segment_id: int,
    episode_window_keyframes: int,
) -> int:
    if (
        start_at is None
        or int(scene_id) != int(start_at.scene_id)
        or int(segment_id) != int(start_at.segment_id)
        or (start_at.frame_id is None and start_at.sequence_start_pos is None)
    ):
        return int(starts[len(starts) // 2])  # policy=middle

    if start_at.frame_id is not None:
        chosen_start = _start_keyframe_pos_from_frame_id(
            sidx=sidx,
            frame_id=int(start_at.frame_id),
            scene_id=int(scene_id),
            segment_id=int(segment_id),
        )
    else:
        chosen_start = int(start_at.sequence_start_pos)
    if int(chosen_start) < 0:
        raise ValueError(f"batch_eval.dataset.start_at sequence start must be >= 0, got {chosen_start}")
    max_start = int(len(sidx.keyframe_indices) - int(episode_window_keyframes))
    if int(chosen_start) > int(max_start):
        raise ValueError(
            "batch_eval.dataset.start_at does not have enough keyframes for a full scheduler_v7 window: "
            f"scene={int(scene_id)} segment={int(segment_id)} start_pos={int(chosen_start)} "
            f"num_keyframes={len(sidx.keyframe_indices)} window_keyframes={int(episode_window_keyframes)}"
        )
    return int(chosen_start)


def build_validation_episode_specs_v7(
    *,
    dataset: ValidationDatasetLike,
    eval_scene_ids: List[int],
    blocks_per_episode: int,
    total_target_frames: int,
    min_window_keyframes: int | None = None,
    start_at: Optional[Dict[str, Any]] = None,
) -> List[ValidationEpisodeSpecV7]:
    if blocks_per_episode < 1:
        raise ValueError("blocks_per_episode must be >= 1")
    if total_target_frames != 3:
        raise ValueError("validation_v7 expects scheduler_v7.episode.total_target_frames=3")
    episode_window_keyframes = int(blocks_per_episode + int(total_target_frames) - 1)
    if min_window_keyframes is not None:
        episode_window_keyframes = max(int(episode_window_keyframes), int(min_window_keyframes))
    out: List[ValidationEpisodeSpecV7] = []
    start_spec = _start_at_from_mapping(start_at)
    if start_spec is not None and int(start_spec.scene_id) not in set(int(x) for x in eval_scene_ids):
        raise ValueError(
            "batch_eval.dataset.start_at.scene_id must be included in batch_eval.dataset.scene_ids: "
            f"start_at.scene_id={int(start_spec.scene_id)} scene_ids={[int(x) for x in eval_scene_ids]}"
        )
    reached_start_scene = start_spec is None
    for scene_id in [int(x) for x in eval_scene_ids]:
        if start_spec is not None:
            if int(scene_id) == int(start_spec.scene_id):
                reached_start_scene = True
            elif not reached_start_scene:
                continue
        seg_ids = [int(x) for x in dataset.list_segment_ids(int(scene_id))]
        seg_ids.sort()
        if start_spec is not None and int(scene_id) == int(start_spec.scene_id):
            if int(start_spec.segment_id) not in set(int(x) for x in seg_ids):
                raise ValueError(
                    "batch_eval.dataset.start_at.segment_id is not available: "
                    f"scene={int(scene_id)} start_at.segment_id={int(start_spec.segment_id)} "
                    f"available_segments={seg_ids}"
                )
        for segment_id in seg_ids:
            if start_spec is not None:
                if not reached_start_scene:
                    continue
                if int(scene_id) == int(start_spec.scene_id) and int(segment_id) < int(start_spec.segment_id):
                    continue
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            starts = build_segment_episode_starts(
                num_keyframes=len(sidx.keyframe_indices),
                blocks_per_episode=int(blocks_per_episode),
                window_keyframes=int(episode_window_keyframes),
            )
            if len(starts) == 0:
                continue
            chosen_start = _choose_start_keyframe_pos(
                starts=[int(x) for x in starts],
                start_at=start_spec,
                sidx=sidx,
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                episode_window_keyframes=int(episode_window_keyframes),
            )
            kfs = [int(x) for x in sidx.keyframe_indices]
            kf_window = kfs[chosen_start : chosen_start + episode_window_keyframes]
            frame_chain = [_middle_frame(list(sidx.keyframe_to_frames[int(kf)])) for kf in kf_window]
            if (
                start_spec is not None
                and int(scene_id) == int(start_spec.scene_id)
                and int(segment_id) == int(start_spec.segment_id)
                and start_spec.frame_id is not None
            ):
                frame_chain[0] = int(start_spec.frame_id)
            block_windows = [
                [int(x) for x in frame_chain[b : b + int(total_target_frames)]]
                for b in range(int(blocks_per_episode))
            ]
            eval_image_refs: List[Tuple[int, int]] = []
            for frame_idx in frame_chain:
                for cam_idx in range(int(sidx.num_cams)):
                    eval_image_refs.append((int(frame_idx), int(cam_idx)))
            out.append(
                ValidationEpisodeSpecV7(
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    episode_start_keyframe_pos=int(chosen_start),
                    frame_chain=[int(x) for x in frame_chain],
                    block_windows=[[int(x) for x in w] for w in block_windows],
                    eval_image_refs=eval_image_refs,
                    num_cams=int(sidx.num_cams),
                )
            )
    return out
