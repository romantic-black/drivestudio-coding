from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from datasets.multi_scene_dataset_v3 import (
    EvalRequestV3,
    MultiSceneDatasetV3,
    TrainSchedulerV4,
    representative_frame_for_keyframe,
)


class TestSchedulerV4:
    """
    Scheduler for formal evaluation.
    - adapt_supervised: delegates to TrainSchedulerV4 when provided.
    - inference_only: emits source + full eval refs per segment, no training/block semantics.
    """

    def __init__(
        self,
        *,
        dataset: MultiSceneDatasetV3,
        mode: str,
        eval_scene_ids: List[int],
        min_test_views_per_segment: int,
        max_segments_per_scene: int = 0,
        deterministic: bool = True,
        seed: int = 123,
        source_protocol: str = "middle_keyframe_middle_frame_cam0",
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        adapt_scheduler: Optional[TrainSchedulerV4] = None,
    ) -> None:
        m = str(mode)
        if m not in ("adapt_supervised", "inference_only", "both"):
            raise ValueError(f"Unsupported TestSchedulerV4 mode={mode!r}")
        self.dataset = dataset
        self.mode = m
        self.eval_scene_ids = [int(x) for x in eval_scene_ids]
        self.min_test_views_per_segment = int(min_test_views_per_segment)
        self.max_segments_per_scene = int(max_segments_per_scene)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.source_protocol = str(source_protocol)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None
        self._events: List[Dict[str, Any]] = []
        self._adapt_scheduler = adapt_scheduler

        if self.min_test_views_per_segment < 1:
            raise ValueError("min_test_views_per_segment must be >= 1")
        if self.source_protocol not in ("first_train_frame_cam0", "middle_keyframe_middle_frame_cam0"):
            raise ValueError(
                "source_protocol must be one of ['first_train_frame_cam0', 'middle_keyframe_middle_frame_cam0']"
            )
        if len(self.eval_scene_ids) == 0:
            raise ValueError("eval_scene_ids must be non-empty")
        if self.mode in ("adapt_supervised", "both") and self._adapt_scheduler is None:
            raise ValueError("adapt_scheduler is required when mode includes adapt_supervised")

        if self.deterministic:
            random.seed(self.seed)

        self._segment_plan: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        self._plan_idx = 0
        self._eval_ref_cursor = 0
        self._build_segment_plan()

    def _build_segment_plan(self) -> None:
        scene_ids = list(self.eval_scene_ids)
        if self.fixed_scene_id is not None:
            scene_ids = [self.fixed_scene_id]
        for sid in scene_ids:
            scene_data = self.dataset.get_scene(int(sid))
            if scene_data is None:
                raise ValueError(f"Scene {sid} cannot be loaded")
            nseg = len(scene_data["segments"])
            seg_ids = [self.fixed_segment_id] if self.fixed_segment_id is not None else list(range(nseg))
            kept = 0
            for seg in seg_ids:
                if seg is None:
                    continue
                if int(seg) < 0 or int(seg) >= nseg:
                    raise ValueError(f"segment_id={seg} out of range for scene_id={sid}")
                refs = self.dataset.resolve_test_image_refs_deterministic(int(sid), int(seg))
                if len(refs) < self.min_test_views_per_segment:
                    continue
                self._segment_plan.append((int(sid), int(seg), [tuple(r) for r in refs]))
                kept += 1
                if self.max_segments_per_scene > 0 and kept >= self.max_segments_per_scene:
                    break
        if len(self._segment_plan) == 0:
            raise ValueError("TestSchedulerV4 has no eligible (scene, segment) pairs")

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._events)
        self._events.clear()
        return out

    def _source_ref_for_segment(self, scene_id: int, segment_id: int) -> Tuple[int, int]:
        sidx = self.dataset.get_segment_index(scene_id, segment_id)
        if self.source_protocol == "first_train_frame_cam0":
            frame = int(sidx.frame_indices[0])
            return (frame, 0)
        # Default: middle keyframe's representative frame, then cam0.
        if len(sidx.keyframe_indices) > 0:
            kf_mid = int(sidx.keyframe_indices[len(sidx.keyframe_indices) // 2])
            frame = int(representative_frame_for_keyframe(sidx, kf_mid))
        else:
            frame = int(sidx.frame_indices[len(sidx.frame_indices) // 2])
        return (frame, 0)

    def next_adapt_batch(self) -> Dict[str, Any]:
        if self._adapt_scheduler is None:
            raise ValueError("TestSchedulerV4.next_adapt_batch requires adapt_scheduler")
        return self._adapt_scheduler.next_batch()

    def get_block_test_refs(self, scene_id: int, segment_id: int) -> List[Tuple[int, int]]:
        return self.dataset.resolve_test_image_refs_deterministic(int(scene_id), int(segment_id))

    def next_eval_batch(self) -> Dict[str, Any]:
        if self._plan_idx >= len(self._segment_plan):
            raise StopIteration("All eval segments are exhausted")
        scene_id, segment_id, refs = self._segment_plan[self._plan_idx]
        if self._eval_ref_cursor == 0:
            self._events.append(
                {
                    "type": "eval_begin",
                    "scene_id": int(scene_id),
                    "segment_id": int(segment_id),
                    "num_eval_views": int(len(refs)),
                }
            )
        start = int(self._eval_ref_cursor)
        end = int(len(refs))
        chunk = refs[start:end]
        source_ref = self._source_ref_for_segment(scene_id, segment_id)
        req = EvalRequestV3(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            source_image_ref=tuple(source_ref),
            eval_image_refs=[tuple(r) for r in chunk],
        )
        batch = self.dataset.get_segment_eval_batch_from_image_refs(req)
        self._eval_ref_cursor = end
        self._events.append(
            {
                "type": "eval_view_chunk_end",
                "scene_id": int(scene_id),
                "segment_id": int(segment_id),
                "num_eval_views_done": int(end),
                "num_eval_views_total": int(len(refs)),
            }
        )
        if self._eval_ref_cursor >= len(refs):
            self._events.append(
                {
                    "type": "segment_eval_end",
                    "scene_id": int(scene_id),
                    "segment_id": int(segment_id),
                    "num_eval_views_total": int(len(refs)),
                }
            )
            self._plan_idx += 1
            self._eval_ref_cursor = 0
        return batch
