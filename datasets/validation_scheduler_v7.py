from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple


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


def build_validation_episode_specs_v7(
    *,
    dataset: ValidationDatasetLike,
    eval_scene_ids: List[int],
    blocks_per_episode: int,
    total_target_frames: int,
) -> List[ValidationEpisodeSpecV7]:
    if blocks_per_episode < 1:
        raise ValueError("blocks_per_episode must be >= 1")
    if total_target_frames != 3:
        raise ValueError("validation_v7 expects scheduler_v7.episode.total_target_frames=3")
    episode_window_keyframes = int(blocks_per_episode + 2)
    out: List[ValidationEpisodeSpecV7] = []
    for scene_id in [int(x) for x in eval_scene_ids]:
        seg_ids = [int(x) for x in dataset.list_segment_ids(int(scene_id))]
        seg_ids.sort()
        for segment_id in seg_ids:
            sidx = dataset.get_segment_index(int(scene_id), int(segment_id))
            starts = build_segment_episode_starts(
                num_keyframes=len(sidx.keyframe_indices),
                blocks_per_episode=int(blocks_per_episode),
                window_keyframes=int(episode_window_keyframes),
            )
            if len(starts) == 0:
                continue
            chosen_start = int(starts[len(starts) // 2])  # policy=middle
            kfs = [int(x) for x in sidx.keyframe_indices]
            kf_window = kfs[chosen_start : chosen_start + episode_window_keyframes]
            frame_chain = [_middle_frame(list(sidx.keyframe_to_frames[int(kf)])) for kf in kf_window]
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

