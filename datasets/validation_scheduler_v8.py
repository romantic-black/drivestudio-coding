from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Protocol, Tuple


class SegmentIndexLike(Protocol):
    keyframe_indices: List[int]
    keyframe_to_frames: Dict[int, List[int]]
    num_cams: int


class ValidationDatasetLike(Protocol):
    def list_segment_ids(self, scene_id: int) -> List[int]: ...
    def get_segment_index(self, scene_id: int, segment_id: int) -> SegmentIndexLike: ...


@dataclass(frozen=True)
class ValidationEpisodeSpecV8:
    scene_id: int
    segment_id: int
    episode_start_keyframe_pos: int
    frame_chain: List[int]
    block_visit_order: List[int]
    visit_target_windows: List[List[int]]
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


def _iter_episode_block_indices(
    *,
    blocks_per_episode: int,
    steps_per_block: int,
    block_order: str,
    step_major_switch_interval_steps: int = 1,
) -> List[int]:
    if block_order == "block_major":
        return [
            int(b)
            for b in range(int(blocks_per_episode))
            for _ in range(int(steps_per_block))
        ]
    if block_order == "step_major":
        switch_every = int(step_major_switch_interval_steps)
        if switch_every < 1:
            raise ValueError("step_major_switch_interval_steps must be >= 1")
        out: List[int] = []
        for round_base in range(0, int(steps_per_block), int(switch_every)):
            chunk = int(min(int(switch_every), int(steps_per_block) - int(round_base)))
            for b in range(int(blocks_per_episode)):
                out.extend([int(b)] * int(chunk))
        return out
    raise ValueError(f"unsupported block_order={block_order!r}")


def build_visit_target_windows_v8(
    *,
    frame_chain: List[int],
    block_visit_order: List[int],
    max_target_frames: int,
    history_target_policy: str = "nearest_visited",
) -> List[List[int]]:
    if history_target_policy not in ("nearest_visited", "random_visited"):
        raise ValueError(
            "history_target_policy must be one of ['nearest_visited', 'random_visited']"
        )
    visited: set[int] = set()
    out: List[List[int]] = []
    for bcur in block_visit_order:
        source = int(frame_chain[int(bcur)])
        candidates = sorted(int(b) for b in visited if int(b) != int(bcur))
        prev_blocks = sorted([b for b in visited if b < int(bcur)], reverse=True)
        next_blocks = sorted([b for b in visited if b > int(bcur)])
        selected: List[int] = []
        max_history_frames = max(int(max_target_frames) - 1, 0)
        if history_target_policy == "random_visited":
            selected = [int(x) for x in random.sample(candidates, min(max_history_frames, len(candidates)))]
        else:
            for b in prev_blocks:
                if len(selected) >= max_history_frames:
                    break
                selected.append(int(b))
            for b in next_blocks:
                if len(selected) >= max_history_frames:
                    break
                selected.append(int(b))
        out.append([int(source)] + [int(frame_chain[b]) for b in selected])
        visited.add(int(bcur))
    return out


def build_validation_episode_specs_v8(
    *,
    dataset: ValidationDatasetLike,
    eval_scene_ids: List[int],
    blocks_per_episode: int,
    total_target_frames: int,
    steps_per_block: int = 1,
    block_order: str = "block_major",
    step_major_switch_interval_steps: int = 1,
    history_target_policy: str = "nearest_visited",
) -> List[ValidationEpisodeSpecV8]:
    if blocks_per_episode < 1:
        raise ValueError("blocks_per_episode must be >= 1")
    if total_target_frames < 1:
        raise ValueError("total_target_frames must be >= 1")
    if total_target_frames > blocks_per_episode:
        raise ValueError("validation_v8 requires total_target_frames <= blocks_per_episode")
    if steps_per_block < 1:
        raise ValueError("steps_per_block must be >= 1")
    if block_order not in ("block_major", "step_major"):
        raise ValueError("block_order must be one of ['block_major', 'step_major']")
    if step_major_switch_interval_steps < 1:
        raise ValueError("step_major_switch_interval_steps must be >= 1")
    if history_target_policy not in ("nearest_visited", "random_visited"):
        raise ValueError(
            "history_target_policy must be one of ['nearest_visited', 'random_visited']"
        )

    episode_window_keyframes = int(blocks_per_episode)
    out: List[ValidationEpisodeSpecV8] = []
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
            chosen_start = int(starts[len(starts) // 2])
            kfs = [int(x) for x in sidx.keyframe_indices]
            kf_window = kfs[chosen_start : chosen_start + episode_window_keyframes]
            frame_chain = [_middle_frame(list(sidx.keyframe_to_frames[int(kf)])) for kf in kf_window]
            block_visit_order = _iter_episode_block_indices(
                blocks_per_episode=int(blocks_per_episode),
                steps_per_block=int(steps_per_block),
                block_order=str(block_order),
                step_major_switch_interval_steps=int(step_major_switch_interval_steps),
            )
            visit_target_windows = build_visit_target_windows_v8(
                frame_chain=[int(x) for x in frame_chain],
                block_visit_order=[int(x) for x in block_visit_order],
                max_target_frames=int(total_target_frames),
                history_target_policy=str(history_target_policy),
            )
            eval_image_refs: List[Tuple[int, int]] = []
            for frame_idx in frame_chain:
                for cam_idx in range(int(sidx.num_cams)):
                    eval_image_refs.append((int(frame_idx), int(cam_idx)))
            out.append(
                ValidationEpisodeSpecV8(
                    scene_id=int(scene_id),
                    segment_id=int(segment_id),
                    episode_start_keyframe_pos=int(chosen_start),
                    frame_chain=[int(x) for x in frame_chain],
                    block_visit_order=[int(x) for x in block_visit_order],
                    visit_target_windows=[[int(y) for y in w] for w in visit_target_windows],
                    eval_image_refs=eval_image_refs,
                    num_cams=int(sidx.num_cams),
                )
            )
    return out
