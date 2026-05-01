from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets.multi_scene_dataset_v4 import BatchRequestV4, MultiSceneDatasetV4

ImageRef = Tuple[int, int]


@dataclass
class DemoBlockState:
    scene_id: int
    segment_id: int
    episode_idx: int
    episode_start_keyframe_pos: int
    block_idx_in_episode: int
    frame_chain: List[int]
    keyframe_window: List[int]
    visited_block_indices: Set[int]
    entered_block_indices: Set[int]
    updated_block_counts: Dict[int, int]
    current_source_frame_idx: int
    current_target_frame_indices: List[int]
    current_target_frame_roles: List[str]
    source_image_refs: List[ImageRef]
    target_image_refs: List[ImageRef]
    near_random_frame_indices_by_block: Dict[int, List[int]]
    near_random_attempted_blocks: int
    near_random_sampled_blocks: int
    near_random_skipped_blocks: int
    near_random_candidate_frames_sum: float


class Stage5DemoScheduler:
    def __init__(self, *, dataset: MultiSceneDatasetV4, cfg: Any, seed: int = 0) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        demo_cfg = cfg.get("demo") or {}
        scheduler_cfg = demo_cfg.get("scheduler") or {}
        episode_cfg = scheduler_cfg.get("episode") or {}
        traversal_cfg = scheduler_cfg.get("traversal") or {}
        target_cfg = scheduler_cfg.get("target") or {}
        event_cfg = scheduler_cfg.get("events") or {}

        self.blocks_per_episode = int(episode_cfg.get("blocks_per_episode", 3))
        self.total_target_frames = int(episode_cfg.get("total_target_frames", 3))
        self.frame_within_keyframe_policy = str(episode_cfg.get("frame_within_keyframe_policy", "first"))
        self.episode_start_policy = str(episode_cfg.get("episode_start_policy", "first"))
        self.include_test = False

        self.scene_order = str(traversal_cfg.get("scene_order", "ascending"))
        self.segment_order = str(traversal_cfg.get("segment_order", "ascending"))
        self.wrap_scene = bool(traversal_cfg.get("wrap_scene", True))
        self.wrap_segment = bool(traversal_cfg.get("wrap_segment", True))
        self.wrap_block = bool(traversal_cfg.get("wrap_block", True))

        self.include_source_frame = bool(target_cfg.get("include_source_frame", True))
        self.use_entered_blocks_for_target = bool(target_cfg.get("use_entered_blocks_for_target", True))
        self.max_target_frames = int(target_cfg.get("max_target_frames", self.total_target_frames))

        self.emit_manual_block_events = bool(event_cfg.get("emit_manual_block_events", True))
        self.emit_scope_change_events = bool(event_cfg.get("emit_scope_change_events", True))
        sv8_cfg = cfg.get("scheduler_v8") or {}
        near_random_cfg = scheduler_cfg.get("near_random_supervision")
        if near_random_cfg is None:
            near_random_cfg = sv8_cfg.get("near_random_supervision") or {}
        self.near_random_cfg = near_random_cfg or {}
        self.near_random_enable = bool(self._cfg_get(self.near_random_cfg, "enable", False))
        self.near_random_frames_per_block = int(self._cfg_get(self.near_random_cfg, "frames_per_block", 1))
        self.near_random_same_keyframe_only = bool(self._cfg_get(self.near_random_cfg, "same_keyframe_only", True))
        self.near_random_insufficient_policy = str(self._cfg_get(self.near_random_cfg, "insufficient_policy", "skip"))
        self.near_random_exclude_source = bool(self._cfg_get(self.near_random_cfg, "exclude_source_frame", True))
        self.near_random_exclude_existing = bool(self._cfg_get(self.near_random_cfg, "exclude_existing_target_frames", True))
        self.near_random_sample_once_per_block = bool(self._cfg_get(self.near_random_cfg, "sample_once_per_block", True))
        self.near_random_camera_policy = str(self._cfg_get(self.near_random_cfg, "camera_policy", "all_cams"))
        self.near_random_role_name = str(self._cfg_get(self.near_random_cfg, "role_name", "near_random"))
        if self.blocks_per_episode < 1:
            raise ValueError("demo.scheduler.episode.blocks_per_episode must be >= 1")
        if self.total_target_frames < 1:
            raise ValueError("demo.scheduler.episode.total_target_frames must be >= 1")
        if self.max_target_frames < 1:
            raise ValueError("demo.scheduler.target.max_target_frames must be >= 1")
        if self.max_target_frames > self.blocks_per_episode:
            raise ValueError(
                "demo scheduler does not use future frames; "
                "demo.scheduler.target.max_target_frames must be <= demo.scheduler.episode.blocks_per_episode"
            )
        if not self.include_source_frame:
            raise ValueError("demo.scheduler.target.include_source_frame must be true")
        if self.near_random_enable:
            if self.near_random_frames_per_block < 1:
                raise ValueError("near_random_supervision.frames_per_block must be >= 1")
            if not self.near_random_same_keyframe_only:
                raise ValueError("v1 only supports near_random_supervision.same_keyframe_only=true")
            if self.near_random_insufficient_policy != "skip":
                raise ValueError("v1 only supports near_random_supervision.insufficient_policy=skip")
            if self.near_random_camera_policy != "all_cams":
                raise ValueError("v1 only supports near_random_supervision.camera_policy=all_cams")

        self._scene_ids = self._ordered_scene_ids()
        if len(self._scene_ids) == 0:
            raise ValueError("Stage5DemoScheduler requires at least one training scene")

        initial_scene_id = scheduler_cfg.get("initial_scene_id")
        initial_segment_id = scheduler_cfg.get("initial_segment_id")
        self.scene_id, self.segment_id = self._resolve_initial_scope(initial_scene_id, initial_segment_id)

        self._events: List[Dict[str, Any]] = []
        self._segment_local_step = 0
        self._global_model_update_step = 0
        self._block_idx_global = 0
        self._block_nav_count = 0
        self._block_uid = 0
        self._episode_idx_global = 0
        self._episode_resample_cursor_by_scope: Dict[Tuple[int, int], int] = {}
        self._block_state: Optional[DemoBlockState] = None
        self._build_episode_state(scene_id=self.scene_id, segment_id=self.segment_id, reason="init")

    @staticmethod
    def _cfg_get(node: Any, key: str, default: Any) -> Any:
        if node is None:
            return default
        if isinstance(node, dict):
            return node.get(key, default)
        if hasattr(node, "get"):
            out = node.get(key, default)
            return default if out is None else out
        if hasattr(node, key):
            out = getattr(node, key)
            return default if out is None else out
        return default

    def _ordered_scene_ids(self) -> List[int]:
        ids = [int(x) for x in self.dataset.list_training_scene_ids()]
        if str(self.scene_order) not in ("ascending", "descending"):
            raise ValueError("demo.scheduler.traversal.scene_order must be ascending or descending")
        ids = sorted(ids)
        if str(self.scene_order) == "descending":
            ids.reverse()
        return ids

    def _ordered_segment_ids(self, scene_id: int) -> List[int]:
        ids = [int(x) for x in self.dataset.list_segment_ids(int(scene_id))]
        if str(self.segment_order) not in ("ascending", "descending"):
            raise ValueError("demo.scheduler.traversal.segment_order must be ascending or descending")
        ids = sorted(ids)
        if str(self.segment_order) == "descending":
            ids.reverse()
        return ids

    def _resolve_initial_scope(self, initial_scene_id: Any, initial_segment_id: Any) -> Tuple[int, int]:
        if initial_scene_id is None:
            scene_id = int(self._scene_ids[0])
        else:
            scene_id = int(initial_scene_id)
            if scene_id not in set(self._scene_ids):
                raise ValueError(f"demo.scheduler.initial_scene_id={scene_id} is not in training scenes {self._scene_ids}")
        seg_ids = self._ordered_segment_ids(scene_id)
        if len(seg_ids) == 0:
            raise ValueError(f"scene_id={scene_id} has no registered segments")
        if initial_segment_id is None:
            segment_id = int(seg_ids[0])
        else:
            segment_id = int(initial_segment_id)
            if segment_id not in set(seg_ids):
                raise ValueError(
                    f"demo.scheduler.initial_segment_id={segment_id} is invalid for scene_id={scene_id}, valid={seg_ids}"
                )
        return int(scene_id), int(segment_id)

    def _choose_frame_for_keyframe(self, frame_indices: List[int]) -> int:
        frames = [int(x) for x in frame_indices]
        if len(frames) == 0:
            raise ValueError("keyframe has no frame indices")
        policy = str(self.frame_within_keyframe_policy)
        if policy == "first":
            return int(frames[0])
        if policy in ("middle", "middle_frame"):
            return int(frames[len(frames) // 2])
        if policy == "random_once_per_episode":
            return int(frames[self._rng.randrange(len(frames))])
        raise ValueError(
            "demo.scheduler.episode.frame_within_keyframe_policy must be one of: "
            "first, middle, random_once_per_episode"
        )

    @staticmethod
    def _sample_no_replace(rng: random.Random, candidates: List[int], k: int) -> List[int]:
        if int(k) < 0:
            raise ValueError("k must be >= 0")
        if int(k) > len(candidates):
            raise ValueError(f"cannot sample {int(k)} without replacement from {len(candidates)} candidates")
        return [int(x) for x in rng.sample(list(candidates), int(k))]

    def _episode_start_keyframe_pos(self, num_keyframes: int) -> int:
        max_start = self._max_episode_start_pos(num_keyframes)
        policy = str(self.episode_start_policy)
        if policy == "first":
            return 0
        if policy == "random":
            return int(self._rng.randrange(max_start + 1))
        raise ValueError("demo.scheduler.episode.episode_start_policy must be one of: first, random")

    def _max_episode_start_pos(self, num_keyframes: int) -> int:
        if num_keyframes < int(self.blocks_per_episode):
            raise ValueError(
                f"segment keyframes={num_keyframes} < blocks_per_episode={self.blocks_per_episode}, cannot build episode"
            )
        return int(num_keyframes - self.blocks_per_episode)

    def _next_manual_episode_start_pos(self, scene_id: int, segment_id: int, num_keyframes: int) -> int:
        max_start = self._max_episode_start_pos(num_keyframes)
        if max_start <= 0:
            start = 0
        else:
            scope = (int(scene_id), int(segment_id))
            prev = int(self._episode_resample_cursor_by_scope.get(scope, -1))
            start = int((prev + 1) % (max_start + 1))
        self._episode_resample_cursor_by_scope[(int(scene_id), int(segment_id))] = int(start)
        return int(start)

    @staticmethod
    def _frame_targets_to_image_refs(num_cams: int, frame_indices: List[int]) -> List[ImageRef]:
        refs: List[ImageRef] = []
        for frame_idx in frame_indices:
            for cam_idx in range(int(num_cams)):
                refs.append((int(frame_idx), int(cam_idx)))
        return refs

    def _build_target_frames_for_block(
        self,
        *,
        frame_chain: List[int],
        block_idx: int,
        visited_block_indices: Set[int],
        max_target_frames: int,
    ) -> List[int]:
        if not bool(self.include_source_frame):
            raise ValueError("demo.scheduler.target.include_source_frame must be true")
        source_frame = int(frame_chain[int(block_idx)])
        candidates = [int(b) for b in visited_block_indices if int(b) != int(block_idx)]
        prev_blocks = sorted([b for b in candidates if b < int(block_idx)], reverse=True)
        next_blocks = sorted([b for b in candidates if b > int(block_idx)])
        selected_blocks: List[int] = []
        for b in prev_blocks:
            if len(selected_blocks) >= int(max_target_frames) - 1:
                break
            selected_blocks.append(int(b))
        for b in next_blocks:
            if len(selected_blocks) >= int(max_target_frames) - 1:
                break
            selected_blocks.append(int(b))
        return [int(source_frame)] + [int(frame_chain[b]) for b in selected_blocks]

    def _target_visited_indices(self, st: DemoBlockState) -> Set[int]:
        if self.use_entered_blocks_for_target:
            return set(int(x) for x in st.entered_block_indices)
        return {int(k) for k, v in st.updated_block_counts.items() if int(v) > 0}

    def _refresh_block_materialization(self, st: DemoBlockState) -> None:
        source_frame = int(st.frame_chain[int(st.block_idx_in_episode)])
        source_image_refs = self._frame_targets_to_image_refs(self._num_cams, [source_frame])
        visited = self._target_visited_indices(st)
        base_target_frames = self._build_target_frames_for_block(
            frame_chain=st.frame_chain,
            block_idx=int(st.block_idx_in_episode),
            visited_block_indices=visited,
            max_target_frames=int(self.max_target_frames),
        )
        base_roles = ["source"] + ["visited" for _ in base_target_frames[1:]]
        near_random_frames: List[int] = []
        if self.near_random_enable:
            sidx = self.dataset.get_segment_index(int(st.scene_id), int(st.segment_id))
            source_keyframe_idx = int(sidx.frame_to_keyframe[int(source_frame)])
            bidx = int(st.block_idx_in_episode)
            if self.near_random_sample_once_per_block and bidx in st.near_random_frame_indices_by_block:
                near_random_frames = [int(x) for x in st.near_random_frame_indices_by_block[bidx]]
            else:
                near_random_frames, num_candidates = self._sample_near_random_frames_for_block(
                    sidx=sidx,
                    source_keyframe_idx=int(source_keyframe_idx),
                    source_frame=int(source_frame),
                    existing_target_frames=[int(x) for x in base_target_frames],
                    num_frames=int(self.near_random_frames_per_block),
                )
                st.near_random_frame_indices_by_block[bidx] = [int(x) for x in near_random_frames]
                st.near_random_attempted_blocks = int(st.near_random_attempted_blocks) + 1
                st.near_random_candidate_frames_sum = float(st.near_random_candidate_frames_sum) + float(num_candidates)
                if len(near_random_frames) > 0:
                    st.near_random_sampled_blocks = int(st.near_random_sampled_blocks) + 1
                else:
                    st.near_random_skipped_blocks = int(st.near_random_skipped_blocks) + 1
        target_frames = [int(x) for x in base_target_frames] + [int(x) for x in near_random_frames]
        target_roles = [str(x) for x in base_roles] + [str(self.near_random_role_name) for _ in near_random_frames]
        target_image_refs = self._frame_targets_to_image_refs(self._num_cams, target_frames)

        st.current_source_frame_idx = int(source_frame)
        st.current_target_frame_indices = [int(x) for x in target_frames]
        st.current_target_frame_roles = [str(x) for x in target_roles]
        st.source_image_refs = [(int(x[0]), int(x[1])) for x in source_image_refs]
        st.target_image_refs = [(int(x[0]), int(x[1])) for x in target_image_refs]
        st.visited_block_indices.add(int(st.block_idx_in_episode))

    def _sample_near_random_frames_for_block(
        self,
        *,
        sidx: Any,
        source_keyframe_idx: int,
        source_frame: int,
        existing_target_frames: List[int],
        num_frames: int,
    ) -> Tuple[List[int], int]:
        if int(num_frames) <= 0:
            return [], 0
        if not self.near_random_same_keyframe_only:
            raise ValueError("v1 only supports same_keyframe_only=true")
        frames = [int(x) for x in list(sidx.keyframe_to_frames[int(source_keyframe_idx)])]
        existing = set(int(x) for x in existing_target_frames)
        candidates: List[int] = []
        for f in frames:
            if self.near_random_exclude_source and int(f) == int(source_frame):
                continue
            if self.near_random_exclude_existing and int(f) in existing:
                continue
            candidates.append(int(f))
        if len(candidates) == 0:
            return [], 0
        if len(candidates) < int(num_frames):
            if self.near_random_insufficient_policy != "skip":
                raise ValueError(f"unsupported near_random insufficient_policy={self.near_random_insufficient_policy!r}")
            return [], int(len(candidates))
        return [int(x) for x in self._sample_no_replace(self._rng, candidates, int(num_frames))], int(len(candidates))

    def _build_episode_state(
        self,
        *,
        scene_id: int,
        segment_id: int,
        reason: str,
        episode_start_keyframe_pos_override: Optional[int] = None,
    ) -> None:
        sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
        keyframes = [int(x) for x in sidx.keyframe_indices]
        if episode_start_keyframe_pos_override is None:
            kf_start = self._episode_start_keyframe_pos(len(keyframes))
        else:
            kf_start = int(episode_start_keyframe_pos_override)
            max_start = self._max_episode_start_pos(len(keyframes))
            if kf_start < 0 or kf_start > max_start:
                raise ValueError(
                    "episode_start_keyframe_pos_override out of range: "
                    f"got {kf_start}, valid=[0, {max_start}]"
                )
        kf_window = [int(x) for x in keyframes[kf_start : kf_start + int(self.blocks_per_episode)]]
        frame_chain = [self._choose_frame_for_keyframe(list(sidx.keyframe_to_frames[int(kf)])) for kf in kf_window]
        self._num_cams = int(sidx.num_cams)
        self._segment_local_step = 0
        st = DemoBlockState(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            episode_idx=int(self._episode_idx_global),
            episode_start_keyframe_pos=int(kf_start),
            block_idx_in_episode=0,
            frame_chain=[int(x) for x in frame_chain],
            keyframe_window=[int(x) for x in kf_window],
            visited_block_indices=set(),
            entered_block_indices={0},
            updated_block_counts={},
            current_source_frame_idx=-1,
            current_target_frame_indices=[],
            current_target_frame_roles=[],
            source_image_refs=[],
            target_image_refs=[],
            near_random_frame_indices_by_block={},
            near_random_attempted_blocks=0,
            near_random_sampled_blocks=0,
            near_random_skipped_blocks=0,
            near_random_candidate_frames_sum=0.0,
        )
        self._refresh_block_materialization(st)
        self._block_state = st
        self.scene_id = int(scene_id)
        self.segment_id = int(segment_id)
        self._episode_resample_cursor_by_scope[(int(scene_id), int(segment_id))] = int(kf_start)
        self._emit_block_enter(reason=str(reason), manual=True, model_update=False, consumed_step=False)
        self._episode_idx_global += 1

    def _emit(self, event: Dict[str, Any], *, force: bool = False) -> None:
        if (not force) and (str(event.get("type", "")).startswith("demo_scope_")) and (not self.emit_scope_change_events):
            return
        out = dict(event)
        out["scheduler_version"] = "demo_v1"
        info = self.get_current_info()
        out.setdefault("scene_id", int(info.get("scene_id", -1)))
        out.setdefault("segment_id", int(info.get("segment_id", -1)))
        out.setdefault("episode_idx_global", int(info.get("episode_idx_global", -1)))
        out.setdefault("block_idx_global", int(info.get("block_idx_global", -1)))
        out.setdefault("demo_block_uid", int(info.get("demo_block_uid", -1)))
        out.setdefault("block_idx_in_episode", int(info.get("block_idx_in_episode", -1)))
        out.setdefault("source_frame_idx", int(info.get("source_frame_idx", -1)))
        out.setdefault("target_frame_indices", [int(x) for x in info.get("target_frame_indices", [])])
        self._events.append(out)

    def _emit_block_enter(self, *, reason: str, manual: bool, model_update: bool, consumed_step: bool) -> None:
        if not self.emit_manual_block_events and bool(manual):
            return
        self._emit(
            {
                "type": "demo_block_enter",
                "reason": str(reason),
                "manual": bool(manual),
                "model_update": bool(model_update),
                "consumed_step": bool(consumed_step),
            }
        )

    def _emit_block_exit(self, *, reason: str, manual: bool, model_update: bool, consumed_step: bool) -> None:
        if not self.emit_manual_block_events and bool(manual):
            return
        self._emit(
            {
                "type": "demo_block_exit",
                "reason": str(reason),
                "manual": bool(manual),
                "model_update": bool(model_update),
                "consumed_step": bool(consumed_step),
            }
        )

    def _move_block(self, delta: int) -> Dict[str, Any]:
        st = self._require_state()
        self._emit_block_exit(reason="manual_block_nav", manual=True, model_update=False, consumed_step=False)
        nxt = int(st.block_idx_in_episode) + int(delta)
        if nxt < 0 or nxt >= int(self.blocks_per_episode):
            if not self.wrap_block:
                raise ValueError("demo.scheduler.traversal.wrap_block=false and block navigation overflowed")
            nxt = nxt % int(self.blocks_per_episode)
        st.block_idx_in_episode = int(nxt)
        st.entered_block_indices.add(int(nxt))
        self._block_idx_global = max(0, int(self._block_idx_global) + (1 if int(delta) > 0 else -1))
        self._block_nav_count += 1
        self._block_uid = int(self._block_nav_count)
        self._refresh_block_materialization(st)
        self._emit_block_enter(reason="manual_block_nav", manual=True, model_update=False, consumed_step=False)
        return self.materialize_current_batch_without_advance()

    def _require_state(self) -> DemoBlockState:
        if self._block_state is None:
            raise ValueError("Stage5DemoScheduler internal state is not initialized")
        return self._block_state

    def _current_segment_ids(self) -> List[int]:
        seg_ids = self._ordered_segment_ids(int(self.scene_id))
        if len(seg_ids) == 0:
            raise ValueError(f"scene_id={self.scene_id} has no segment ids")
        return seg_ids

    def _set_scope(self, *, scene_id: int, segment_id: int, reason: str) -> Dict[str, Any]:
        old_scene = int(self.scene_id)
        old_segment = int(self.segment_id)
        if int(scene_id) not in set(self._scene_ids):
            raise ValueError(f"scene_id={scene_id} is not in training scenes {self._scene_ids}")
        seg_ids = self._ordered_segment_ids(int(scene_id))
        if int(segment_id) not in set(seg_ids):
            raise ValueError(f"segment_id={segment_id} is invalid for scene_id={scene_id}, valid={seg_ids}")
        self.scene_id = int(scene_id)
        self.segment_id = int(segment_id)
        self._block_idx_global = 0
        self._block_nav_count += 1
        self._block_uid = int(self._block_nav_count)
        self._episode_idx_global = 0
        self._build_episode_state(scene_id=int(scene_id), segment_id=int(segment_id), reason=str(reason))
        if self.emit_scope_change_events or str(reason) == "init":
            self._emit(
                {
                    "type": "demo_scope_change",
                    "old_scene_id": int(old_scene),
                    "old_segment_id": int(old_segment),
                    "new_scene_id": int(scene_id),
                    "new_segment_id": int(segment_id),
                    "reason": str(reason),
                    "manual": True,
                    "model_update": False,
                    "consumed_step": False,
                }
            )
        return self.materialize_current_batch_without_advance()

    def list_scene_ids(self) -> List[int]:
        return [int(x) for x in self._scene_ids]

    def list_segment_ids(self, scene_id: int) -> List[int]:
        return self._ordered_segment_ids(int(scene_id))

    def set_scope(self, scene_id: int, segment_id: int) -> Dict[str, Any]:
        return self._set_scope(scene_id=int(scene_id), segment_id=int(segment_id), reason="set_scope")

    def set_scene(self, scene_id: int) -> Dict[str, Any]:
        seg_ids = self._ordered_segment_ids(int(scene_id))
        if len(seg_ids) == 0:
            raise ValueError(f"scene_id={scene_id} has no registered segments")
        return self._set_scope(scene_id=int(scene_id), segment_id=int(seg_ids[0]), reason="set_scene")

    def set_segment(self, segment_id: int) -> Dict[str, Any]:
        return self._set_scope(scene_id=int(self.scene_id), segment_id=int(segment_id), reason="set_segment")

    def next_scene(self) -> Dict[str, Any]:
        cur_idx = self._scene_ids.index(int(self.scene_id))
        nxt_idx = int(cur_idx) + 1
        if nxt_idx >= len(self._scene_ids):
            if not self.wrap_scene:
                raise ValueError("demo.scheduler.traversal.wrap_scene=false and next_scene overflowed")
            nxt_idx = 0
        next_scene_id = int(self._scene_ids[nxt_idx])
        next_seg_ids = self._ordered_segment_ids(next_scene_id)
        return self._set_scope(scene_id=next_scene_id, segment_id=int(next_seg_ids[0]), reason="next_scene")

    def prev_scene(self) -> Dict[str, Any]:
        cur_idx = self._scene_ids.index(int(self.scene_id))
        nxt_idx = int(cur_idx) - 1
        if nxt_idx < 0:
            if not self.wrap_scene:
                raise ValueError("demo.scheduler.traversal.wrap_scene=false and prev_scene overflowed")
            nxt_idx = len(self._scene_ids) - 1
        next_scene_id = int(self._scene_ids[nxt_idx])
        next_seg_ids = self._ordered_segment_ids(next_scene_id)
        return self._set_scope(scene_id=next_scene_id, segment_id=int(next_seg_ids[0]), reason="prev_scene")

    def next_segment(self) -> Dict[str, Any]:
        seg_ids = self._current_segment_ids()
        cur_idx = seg_ids.index(int(self.segment_id))
        nxt_idx = int(cur_idx) + 1
        if nxt_idx >= len(seg_ids):
            if not self.wrap_segment:
                raise ValueError("demo.scheduler.traversal.wrap_segment=false and next_segment overflowed")
            nxt_idx = 0
        return self._set_scope(
            scene_id=int(self.scene_id),
            segment_id=int(seg_ids[nxt_idx]),
            reason="next_segment",
        )

    def prev_segment(self) -> Dict[str, Any]:
        seg_ids = self._current_segment_ids()
        cur_idx = seg_ids.index(int(self.segment_id))
        nxt_idx = int(cur_idx) - 1
        if nxt_idx < 0:
            if not self.wrap_segment:
                raise ValueError("demo.scheduler.traversal.wrap_segment=false and prev_segment overflowed")
            nxt_idx = len(seg_ids) - 1
        return self._set_scope(
            scene_id=int(self.scene_id),
            segment_id=int(seg_ids[nxt_idx]),
            reason="prev_segment",
        )

    def mark_current_block_updated(self) -> None:
        st = self._require_state()
        cur = int(st.block_idx_in_episode)
        st.updated_block_counts[cur] = int(st.updated_block_counts.get(cur, 0)) + 1
        self._segment_local_step += 1
        self._global_model_update_step += 1
        self._emit(
            {
                "type": "demo_step",
                "manual": True,
                "model_update": True,
                "consumed_step": True,
                "block_update_count": int(st.updated_block_counts[cur]),
                "segment_local_step": int(self._segment_local_step),
                "model_update_step": int(self._global_model_update_step),
            }
        )

    def next_block(self) -> Dict[str, Any]:
        return self._move_block(+1)

    def prev_block(self) -> Dict[str, Any]:
        return self._move_block(-1)

    def resample_episode(self) -> Dict[str, Any]:
        st = self._require_state()
        scene_id = int(st.scene_id)
        segment_id = int(st.segment_id)
        sidx = self.dataset.get_segment_index(scene_id, segment_id)
        num_keyframes = int(len(sidx.keyframe_indices))
        next_start = self._next_manual_episode_start_pos(scene_id, segment_id, num_keyframes)
        self._emit_block_exit(reason="manual_episode_resample", manual=True, model_update=False, consumed_step=False)
        self._block_idx_global = 0
        self._block_nav_count += 1
        self._block_uid = int(self._block_nav_count)
        self._build_episode_state(
            scene_id=scene_id,
            segment_id=segment_id,
            reason="manual_episode_resample",
            episode_start_keyframe_pos_override=int(next_start),
        )
        self._emit(
            {
                "type": "demo_episode_resample",
                "manual": True,
                "model_update": False,
                "consumed_step": False,
                "episode_start_keyframe_pos": int(next_start),
            }
        )
        return self.materialize_current_batch_without_advance()

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        st = self._require_state()
        req = BatchRequestV4(
            scene_id=int(st.scene_id),
            segment_id=int(st.segment_id),
            source_image_ref=(int(st.current_source_frame_idx), 0),
            source_image_refs=[(int(x[0]), int(x[1])) for x in st.source_image_refs],
            target_image_refs=[(int(x[0]), int(x[1])) for x in st.target_image_refs],
            include_test=bool(self.include_test),
        )
        batch = self.dataset.get_segment_batch_from_image_refs(req, enforce_target0_equals_source=True)
        # Keep request_meta shape close to TrainSchedulerV8 so target-view weighting
        # in train_step does not silently fall back to all-source weighting.
        request_meta = dict(batch.get("request_meta") or {})
        target_frame_roles = [str(x) for x in st.current_target_frame_roles]
        if len(target_frame_roles) != len(st.current_target_frame_indices):
            target_frames = [int(x) for x in st.current_target_frame_indices]
            target_frame_roles = ["source"] + ["visited" for _ in target_frames[1:]]
        target_image_roles: List[str] = []
        for role in target_frame_roles:
            for _ in range(int(self._num_cams)):
                target_image_roles.append(str(role))
        bidx = int(st.block_idx_in_episode)
        near_random_frame_indices = [int(x) for x in st.near_random_frame_indices_by_block.get(bidx, [])]
        attempted = int(st.near_random_attempted_blocks)
        skipped = int(st.near_random_skipped_blocks)
        sampled = int(st.near_random_sampled_blocks)
        candidate_sum = float(st.near_random_candidate_frames_sum)
        request_meta["source_image_refs"] = [(int(x[0]), int(x[1])) for x in st.source_image_refs]
        request_meta["target_image_refs"] = [(int(x[0]), int(x[1])) for x in st.target_image_refs]
        request_meta["target_frame_roles"] = [str(x) for x in target_frame_roles]
        request_meta["target_image_roles"] = [str(x) for x in target_image_roles]
        request_meta["near_random_frame_indices"] = [int(x) for x in near_random_frame_indices]
        request_meta["near_random_supervision_enable"] = bool(self.near_random_enable)
        request_meta["scheduler/near_random/enabled"] = float(1.0 if self.near_random_enable else 0.0)
        request_meta["scheduler/near_random/num_frames"] = float(len(near_random_frame_indices))
        request_meta["scheduler/near_random/skip_ratio"] = float(skipped / max(attempted, 1))
        request_meta["scheduler/near_random/num_candidate_frames_mean"] = float(candidate_sum / max(attempted, 1))
        request_meta["scheduler/near_random/sampled_blocks"] = float(sampled)
        batch["request_meta"] = request_meta
        info = self.get_current_info()
        batch["_scheduler_v4_aligned_info"] = dict(info)
        batch["_scheduler_v7_aligned_info"] = dict(info)
        batch["_scheduler_v8_aligned_info"] = dict(info)
        batch["_scheduler_demo_v1_info"] = dict(info)
        batch["_scheduler_demo_v1_peek"] = True
        return batch

    def pop_events(self) -> List[Dict[str, Any]]:
        out = [dict(x) for x in self._events]
        self._events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        st = self._block_state
        if st is None:
            return {
                "scheduler_version": "demo_v1",
                "scene_id": -1,
                "segment_id": -1,
                "episode_idx_global": -1,
                "block_idx_global": int(self._block_idx_global),
                "demo_block_uid": int(self._block_uid),
                "block_nav_count": int(self._block_nav_count),
                "block_idx_in_episode": -1,
                "segment_local_step": int(self._segment_local_step),
                "source_frame_idx": -1,
                "target_frame_indices": [],
                "target_frame_roles": [],
                "source_image_refs": [],
                "target_image_refs": [],
                "visited_block_indices": [],
                "entered_block_indices": [],
                "updated_block_counts": {},
                "episode_start_keyframe_pos": -1,
                "keyframe_window": [],
                "frame_chain": [],
                "global_step": int(self._global_model_update_step),
            }
        return {
            "scheduler_version": "demo_v1",
            "scene_id": int(st.scene_id),
            "segment_id": int(st.segment_id),
            "episode_idx_global": int(st.episode_idx),
            "block_idx_global": int(self._block_idx_global),
            "demo_block_uid": int(self._block_uid),
            "block_nav_count": int(self._block_nav_count),
            "block_idx_in_episode": int(st.block_idx_in_episode),
            "segment_local_step": int(self._segment_local_step),
            "source_frame_idx": int(st.current_source_frame_idx),
            "target_frame_indices": [int(x) for x in st.current_target_frame_indices],
            "target_frame_roles": [str(x) for x in st.current_target_frame_roles],
            "source_image_refs": [tuple(x) for x in st.source_image_refs],
            "target_image_refs": [tuple(x) for x in st.target_image_refs],
            "visited_block_indices": sorted(int(x) for x in st.visited_block_indices),
            "entered_block_indices": sorted(int(x) for x in st.entered_block_indices),
            "updated_block_counts": {int(k): int(v) for k, v in st.updated_block_counts.items()},
            "episode_start_keyframe_pos": int(st.episode_start_keyframe_pos),
            "keyframe_window": [int(x) for x in st.keyframe_window],
            "frame_chain": [int(x) for x in st.frame_chain],
            "global_step": int(self._global_model_update_step),
        }


def build_stage5_demo_scheduler_from_cfg(cfg: Any, dataset: MultiSceneDatasetV4) -> Stage5DemoScheduler:
    training_cfg = cfg.get("training") or {}
    seed = int(training_cfg.get("seed", 0))
    return Stage5DemoScheduler(dataset=dataset, cfg=cfg, seed=seed)
