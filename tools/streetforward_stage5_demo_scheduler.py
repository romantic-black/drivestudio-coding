from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets.multi_scene_dataset_v4 import BatchRequestV4, MultiSceneDatasetV4
from streetforward_eval.episode_builder import TestEpisodeSpec
from streetforward_eval.protocols import TestProtocolSpec, resolve_eval_offsets
from streetforward_eval.stage5_6_runtime import build_stage5_6_eval_train_batch, iter_block_visit_order

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


class Stage5_6EvalDemoScheduler:
    def __init__(self, *, dataset: MultiSceneDatasetV4, cfg: Any, seed: int = 0, device: Any = None) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.seed = int(seed)
        self.device = device
        demo_cfg = cfg.get("demo") or {}
        scheduler_cfg = demo_cfg.get("scheduler") or {}
        if str(scheduler_cfg.get("type", "")).strip() != "eval_v8_stage5_6":
            raise ValueError("Stage5_6EvalDemoScheduler requires demo.scheduler.type=eval_v8_stage5_6")
        self.is_stage5_6_eval_demo = True

        batch_eval_cfg = cfg.get("batch_eval") or {}
        batch_eval_dataset_cfg = batch_eval_cfg.get("dataset") or {}
        runtime_cfg = batch_eval_cfg.get("runtime") or {}
        history_cfg = batch_eval_cfg.get("history") or {}
        stage5_6_cfg = batch_eval_cfg.get("stage5_6_eval") or {}
        sv8_cfg = cfg.get("scheduler_v8") or {}
        sv8_exec = sv8_cfg.get("execution") if hasattr(sv8_cfg, "get") else None
        sv8_episode = sv8_cfg.get("episode") if hasattr(sv8_cfg, "get") else None

        self.name = str(scheduler_cfg.get("name", "stage5_6_demo_eval"))
        self.sequence_length = int(scheduler_cfg.get("sequence_length", self._cfg_get_first(batch_eval_cfg, ["sequence_length"], 10)))
        self.input_offsets = [int(x) for x in self._as_list(scheduler_cfg.get("input_offsets", [0, 2, 4, 6, 8]))]
        self.eval_offsets_any = scheduler_cfg.get("eval_offsets", "all")
        self.steps_per_input = int(scheduler_cfg.get("steps_per_input", 16))
        self.block_order = str(
            scheduler_cfg.get(
                "block_order",
                (sv8_exec.get("block_order") if sv8_exec is not None and sv8_exec.get("block_order") is not None else "step_major"),
            )
        )
        self.step_major_switch_interval_steps = int(
            scheduler_cfg.get(
                "step_major_switch_interval_steps",
                (
                    sv8_exec.get("step_major_switch_interval_steps")
                    if sv8_exec is not None and sv8_exec.get("step_major_switch_interval_steps") is not None
                    else 4
                ),
            )
        )
        self.max_target_frames_including_source = int(
            scheduler_cfg.get(
                "max_target_frames_including_source",
                (
                    sv8_episode.get("total_target_frames")
                    if sv8_episode is not None and sv8_episode.get("total_target_frames") is not None
                    else 3
                ),
            )
        )
        self.window_policy = str(scheduler_cfg.get("window_policy", batch_eval_dataset_cfg.get("window_policy", "sliding")))
        self.stride = int(scheduler_cfg.get("stride", batch_eval_dataset_cfg.get("stride", 30)))
        self.require_full_window = bool(
            scheduler_cfg.get("require_full_window", batch_eval_dataset_cfg.get("require_full_window", True))
        )
        self.wrap_scene = bool(scheduler_cfg.get("wrap_scene", True))
        self.wrap_segment = bool(scheduler_cfg.get("wrap_segment", True))
        self.wrap_episode = bool(scheduler_cfg.get("wrap_episode", True))
        self.update_node_state = bool(runtime_cfg.get("update_node_state", True))
        self.update_hidden_state = bool(runtime_cfg.get("update_hidden_state", True))
        self.record_each_step = bool(history_cfg.get("record_each_step", False))
        self.record_history_on_input_exit = bool(history_cfg.get("record_support_residual_on_input_exit", True))
        self.nearby_policy = str(stage5_6_cfg.get("nearby_policy", "adjacent_non_input"))
        self.nearby_role_name = str(stage5_6_cfg.get("nearby_role_name", "near_random"))
        self.allow_partial_nearby = bool(stage5_6_cfg.get("allow_partial_nearby", True))

        cameras_cfg = scheduler_cfg.get("cameras") or demo_cfg.get("cameras") or batch_eval_cfg.get("cameras") or {}
        self.camera_ids = [int(x) for x in self._as_list(cameras_cfg.get("ids", [0]))]
        names = cameras_cfg.get("names")
        if names is None:
            names = [f"cam{int(x)}" for x in self.camera_ids]
        self.camera_names = [str(x) for x in self._as_list(names)]
        if len(self.camera_names) != len(self.camera_ids):
            raise ValueError("demo scheduler camera ids/names length mismatch")
        update_cameras = scheduler_cfg.get("update_cameras") or demo_cfg.get("update_cameras") or batch_eval_cfg.get("update_cameras")
        if update_cameras is None:
            self.update_camera_ids: Optional[List[int]] = None
        else:
            self.update_camera_ids = [int(x) for x in self._as_list(update_cameras.get("ids", self.camera_ids))]

        configured_scene_ids = scheduler_cfg.get("scene_ids", batch_eval_dataset_cfg.get("scene_ids"))
        if configured_scene_ids is None:
            configured_scene_ids = cfg.get("data", {}).get("train_scene_ids") if cfg.get("data") is not None else None
        if configured_scene_ids is None:
            self._scene_ids = [int(x) for x in dataset.list_training_scene_ids()]
        else:
            self._scene_ids = [int(x) for x in self._as_list(configured_scene_ids)]
        self._scene_ids = sorted(self._scene_ids)
        if len(self._scene_ids) == 0:
            raise ValueError("Stage5_6 eval demo scheduler requires at least one scene id")

        initial_scene_id = scheduler_cfg.get("initial_scene_id")
        initial_segment_id = scheduler_cfg.get("initial_segment_id")
        self.scene_id, self.segment_id = self._resolve_initial_scope(initial_scene_id, initial_segment_id)
        self.sequence_start_pos = self._resolve_initial_start(scheduler_cfg.get("initial_sequence_start_pos"))

        self._events: List[Dict[str, Any]] = []
        self._episode_idx_global = 0
        self._block_idx_global = 0
        self._visit_cursor = 0
        self._global_model_update_step = 0
        self._local_step_by_block: List[int] = []
        self._visited_blocks: Set[int] = set()
        self._updated_block_counts: Dict[int, int] = {}
        self._last_batch: Optional[Dict[str, Any]] = None
        self._last_info: Dict[str, Any] = {}
        self._spec = self._build_current_spec()
        self._visit_order = iter_block_visit_order(
            num_blocks=len(self._spec.input_frame_ids),
            steps_per_block=int(self.steps_per_input),
            block_order=str(self.block_order),
            step_major_switch_interval_steps=int(self.step_major_switch_interval_steps),
        )
        self._reset_episode_state(reason="init", rebuild_spec=False)

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            from omegaconf import ListConfig

            if isinstance(value, ListConfig):
                return list(value)
        except Exception:
            pass
        return [value]

    @staticmethod
    def _cfg_get_first(node: Any, keys: List[str], default: Any) -> Any:
        cur = node
        for key in keys:
            if cur is None or not hasattr(cur, "get"):
                return default
            cur = cur.get(key)
        return default if cur is None else cur

    def _frames_for_segment(self, scene_id: int, segment_id: int) -> List[int]:
        sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
        all_frames = [int(x) for x in sorted(sidx.frame_indices)]
        train_frame_set = getattr(sidx, "train_frame_set", None)
        if train_frame_set is not None:
            train_set = set(int(x) for x in train_frame_set)
            return [int(f) for f in all_frames if int(f) in train_set]
        return all_frames

    def _window_starts(self, scene_id: int, segment_id: int) -> List[int]:
        frames = self._frames_for_segment(int(scene_id), int(segment_id))
        if len(frames) == 0:
            return []
        if self.require_full_window and len(frames) < int(self.sequence_length):
            return []
        if int(self.stride) < 1:
            raise ValueError("demo.scheduler.stride must be >= 1")
        if self.window_policy == "middle":
            return [max(0, (len(frames) - int(self.sequence_length)) // 2)]
        if self.window_policy == "sliding":
            if self.require_full_window:
                return list(range(0, len(frames) - int(self.sequence_length) + 1, int(self.stride)))
            return list(range(0, len(frames), int(self.stride)))
        raise ValueError("demo.scheduler.window_policy must be one of: sliding, middle")

    def _ordered_segment_ids(self, scene_id: int) -> List[int]:
        return sorted(int(x) for x in self.dataset.list_segment_ids(int(scene_id)))

    def _resolve_initial_scope(self, initial_scene_id: Any, initial_segment_id: Any) -> Tuple[int, int]:
        scene_id = int(self._scene_ids[0] if initial_scene_id is None else initial_scene_id)
        if scene_id not in set(self._scene_ids):
            raise ValueError(f"initial scene_id={scene_id} is not configured in demo.scheduler.scene_ids={self._scene_ids}")
        seg_ids = self._ordered_segment_ids(scene_id)
        if len(seg_ids) == 0:
            raise ValueError(f"scene_id={scene_id} has no segment ids")
        segment_id = int(seg_ids[0] if initial_segment_id is None else initial_segment_id)
        if segment_id not in set(seg_ids):
            raise ValueError(f"segment_id={segment_id} is invalid for scene_id={scene_id}, valid={seg_ids}")
        return int(scene_id), int(segment_id)

    def _resolve_initial_start(self, initial_start: Any) -> int:
        starts = self._window_starts(int(self.scene_id), int(self.segment_id))
        if len(starts) == 0:
            raise ValueError(f"scene={self.scene_id} segment={self.segment_id} has no valid eval windows")
        if initial_start is None:
            return int(starts[0])
        start = int(initial_start)
        if start not in set(starts):
            raise ValueError(f"initial_sequence_start_pos={start} is invalid, valid starts begin {starts[:12]}")
        return int(start)

    def _make_refs(self, frames: List[int], camera_ids: List[int]) -> List[ImageRef]:
        return [(int(f), int(c)) for f in frames for c in camera_ids]

    def _build_current_spec(self) -> TestEpisodeSpec:
        frames = self._frames_for_segment(int(self.scene_id), int(self.segment_id))
        start = int(self.sequence_start_pos)
        window_frames = [int(x) for x in frames[start : start + int(self.sequence_length)]]
        if self.require_full_window and len(window_frames) < int(self.sequence_length):
            raise ValueError(
                f"window start={start} has len={len(window_frames)} < sequence_length={self.sequence_length}"
            )
        frame_offsets = list(range(len(window_frames)))
        eval_offsets = resolve_eval_offsets(self.eval_offsets_any, sequence_length=int(self.sequence_length))
        input_offsets: List[int] = []
        for off in self.input_offsets:
            if int(off) < 0 or int(off) >= len(window_frames):
                raise ValueError(f"input offset={off} out of range for window len={len(window_frames)}")
            input_offsets.append(int(off))
        mapped_eval_offsets: List[int] = []
        for off in eval_offsets:
            if int(off) < 0 or int(off) >= len(window_frames):
                raise ValueError(f"eval offset={off} out of range for window len={len(window_frames)}")
            mapped_eval_offsets.append(int(off))
        input_frame_ids = [int(window_frames[o]) for o in input_offsets]
        eval_frame_ids = [int(window_frames[o]) for o in mapped_eval_offsets]
        uid = f"scene{int(self.scene_id):03d}_seg{int(self.segment_id):03d}_start{int(start):06d}"
        spec = TestEpisodeSpec(
            exp_name=str(self.name),
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            episode_idx=int(self._episode_idx_global),
            sequence_start_pos=int(start),
            frame_offsets=[int(x) for x in frame_offsets],
            frame_ids=[int(x) for x in window_frames],
            input_offsets=[int(x) for x in input_offsets],
            eval_offsets=[int(x) for x in mapped_eval_offsets],
            input_frame_ids=[int(x) for x in input_frame_ids],
            eval_frame_ids=[int(x) for x in eval_frame_ids],
            camera_ids=[int(x) for x in self.camera_ids],
            camera_names=[str(x) for x in self.camera_names],
            input_image_refs=self._make_refs(input_frame_ids, self.camera_ids),
            eval_image_refs=self._make_refs(eval_frame_ids, self.camera_ids),
            episode_uid=uid,
        )
        self._episode_idx_global += 1
        return spec

    def _reset_episode_state(self, *, reason: str, rebuild_spec: bool = True) -> None:
        if rebuild_spec:
            self._spec = self._build_current_spec()
            self._visit_order = iter_block_visit_order(
                num_blocks=len(self._spec.input_frame_ids),
                steps_per_block=int(self.steps_per_input),
                block_order=str(self.block_order),
                step_major_switch_interval_steps=int(self.step_major_switch_interval_steps),
            )
        self._visit_cursor = 0
        self._global_model_update_step = 0
        self._block_idx_global = 0
        self._local_step_by_block = [0 for _ in self._spec.input_frame_ids]
        self._visited_blocks = set()
        self._updated_block_counts = {}
        self._last_batch = None
        self._last_info = self._build_info(reason=reason)
        self._emit({"type": "demo_episode_reset", "reason": str(reason), "manual": True, "model_update": False})

    def _current_block_idx(self) -> int:
        if len(self._visit_order) == 0:
            return -1
        if int(self._visit_cursor) >= len(self._visit_order):
            return int(self._visit_order[-1])
        return int(self._visit_order[int(self._visit_cursor)])

    def _build_info(self, *, reason: str = "") -> Dict[str, Any]:
        block_idx = self._current_block_idx()
        block_repeat_step = 0
        if 0 <= block_idx < len(self._local_step_by_block):
            block_repeat_step = int(self._local_step_by_block[block_idx]) + (0 if self.is_episode_done() else 1)
        rm: Dict[str, Any] = {}
        if isinstance(self._last_batch, dict):
            rm = dict(self._last_batch.get("request_meta") or {})
        aligned = self._last_batch.get("_scheduler_v8_aligned_info", {}) if isinstance(self._last_batch, dict) else {}
        return {
            "scheduler_version": "v8",
            "demo_scheduler_type": "eval_v8_stage5_6",
            "scene_id": int(self.scene_id),
            "segment_id": int(self.segment_id),
            "episode_idx_global": int(self._spec.episode_idx),
            "block_idx_global": int(aligned.get("block_idx_global", self._block_idx_global)),
            "block_idx_in_episode": int(block_idx),
            "block_repeat_step": int(aligned.get("block_repeat_step", block_repeat_step)),
            "segment_local_step": int(self._global_model_update_step),
            "source_frame_idx": int(aligned.get("source_frame_idx", -1)),
            "target_frame_indices": [int(x) for x in aligned.get("target_frame_indices", [])],
            "target_frame_roles": [str(x) for x in rm.get("target_frame_roles", aligned.get("target_frame_roles", []))],
            "near_random_frame_indices": [int(x) for x in rm.get("near_random_frame_indices", [])],
            "source_image_refs": [tuple(x) for x in rm.get("source_image_refs", [])],
            "target_image_refs": [tuple(x) for x in rm.get("target_image_refs", [])],
            "target_image_roles": [str(x) for x in rm.get("target_image_roles", [])],
            "visited_block_indices": sorted(int(x) for x in self._visited_blocks),
            "updated_block_counts": {int(k): int(v) for k, v in self._updated_block_counts.items()},
            "sequence_start_pos": int(self.sequence_start_pos),
            "sequence_length": int(self.sequence_length),
            "input_offsets": [int(x) for x in self._spec.input_offsets],
            "input_frame_ids": [int(x) for x in self._spec.input_frame_ids],
            "visit_cursor": int(self._visit_cursor),
            "visit_total": int(len(self._visit_order)),
            "episode_done": bool(self.is_episode_done()),
            "block_order": str(self.block_order),
            "step_major_switch_interval_steps": int(self.step_major_switch_interval_steps),
            "last_reason": str(reason),
            "global_step": int(self._global_model_update_step),
        }

    def _emit(self, event: Dict[str, Any]) -> None:
        out = dict(event)
        out.setdefault("scheduler_version", "v8")
        out.setdefault("demo_scheduler_type", "eval_v8_stage5_6")
        out.setdefault("scene_id", int(self.scene_id))
        out.setdefault("segment_id", int(self.segment_id))
        out.setdefault("episode_idx_global", int(self._spec.episode_idx))
        out.setdefault("block_idx_in_episode", int(self._current_block_idx()))
        out.setdefault("visit_cursor", int(self._visit_cursor))
        self._events.append(out)

    def is_episode_done(self) -> bool:
        return int(self._visit_cursor) >= int(len(self._visit_order))

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        if self.device is None:
            raise ValueError("Stage5_6 eval demo scheduler requires device for minimal batch materialization")
        if self.is_episode_done() and self._last_batch is not None:
            return self._last_batch
        block_idx = self._current_block_idx()
        if block_idx < 0:
            raise ValueError("cannot materialize empty eval episode")
        block_repeat_step = int(self._local_step_by_block[int(block_idx)]) + 1
        segment_local_step = int(self._global_model_update_step) + 1
        batch = build_stage5_6_eval_train_batch(
            dataset=self.dataset,
            spec=self._spec,
            block_idx=int(block_idx),
            block_repeat_step=int(block_repeat_step),
            segment_local_step=int(segment_local_step),
            visited_blocks=set(int(x) for x in self._visited_blocks),
            device=self.device,
            update_camera_ids=self.update_camera_ids,
            protocol_name=str(self.name),
            steps_per_input=int(self.steps_per_input),
            target_frame_policy="visited_episode_frames",
            max_target_frames_including_source=int(self.max_target_frames_including_source),
            nearby_policy=str(self.nearby_policy),
            nearby_role_name=str(self.nearby_role_name),
            allow_partial_nearby=bool(self.allow_partial_nearby),
            block_order=str(self.block_order),
            step_major_switch_interval_steps=int(self.step_major_switch_interval_steps),
        )
        self._last_batch = batch
        self._last_info = self._build_info(reason="materialize")
        batch["_scheduler_eval_v8_demo_info"] = dict(self._last_info)
        return batch

    def mark_current_block_updated(self) -> None:
        if self.is_episode_done():
            self._emit({"type": "demo_step_ignored", "reason": "episode_done", "manual": True})
            return
        block_idx = int(self._visit_order[int(self._visit_cursor)])
        self._local_step_by_block[block_idx] = int(self._local_step_by_block[block_idx]) + 1
        self._updated_block_counts[block_idx] = int(self._updated_block_counts.get(block_idx, 0)) + 1
        self._visited_blocks.add(int(block_idx))
        self._global_model_update_step += 1
        self._block_idx_global = int(self._spec.episode_idx) * int(max(len(self._spec.input_frame_ids), 1)) + int(block_idx)
        next_cursor = int(self._visit_cursor) + 1
        next_block = int(self._visit_order[next_cursor]) if next_cursor < len(self._visit_order) else None
        block_exit = next_block is None or int(next_block) != int(block_idx)
        self._emit(
            {
                "type": "demo_step",
                "manual": True,
                "model_update": True,
                "consumed_step": True,
                "block_idx_in_episode": int(block_idx),
                "block_repeat_step": int(self._local_step_by_block[block_idx]),
                "segment_local_step": int(self._global_model_update_step),
                "block_exit": bool(block_exit),
            }
        )
        if block_exit:
            self._emit(
                {
                    "type": "demo_block_exit",
                    "manual": True,
                    "model_update": True,
                    "consumed_step": True,
                    "block_idx_in_episode": int(block_idx),
                    "block_repeat_step": int(self._local_step_by_block[block_idx]),
                }
            )
        self._visit_cursor = int(next_cursor)
        if block_exit and not self.is_episode_done():
            self._emit(
                {
                    "type": "demo_block_enter",
                    "manual": True,
                    "model_update": False,
                    "consumed_step": False,
                    "block_idx_in_episode": int(self._current_block_idx()),
                }
            )
        self._last_info = self._build_info(reason="mark_updated")

    def pop_events(self) -> List[Dict[str, Any]]:
        out = [dict(x) for x in self._events]
        self._events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info or self._build_info(reason="get_current_info"))

    def list_scene_ids(self) -> List[int]:
        return [int(x) for x in self._scene_ids]

    def list_segment_ids(self, scene_id: int) -> List[int]:
        return self._ordered_segment_ids(int(scene_id))

    def list_sequence_start_positions(self) -> List[int]:
        return [int(x) for x in self._window_starts(int(self.scene_id), int(self.segment_id))]

    def _set_scope_and_start(self, *, scene_id: int, segment_id: int, sequence_start_pos: int, reason: str) -> Dict[str, Any]:
        if int(scene_id) not in set(self._scene_ids):
            raise ValueError(f"scene_id={scene_id} is not configured")
        seg_ids = self._ordered_segment_ids(int(scene_id))
        if int(segment_id) not in set(seg_ids):
            raise ValueError(f"segment_id={segment_id} is invalid for scene_id={scene_id}, valid={seg_ids}")
        starts = self._window_starts(int(scene_id), int(segment_id))
        if int(sequence_start_pos) not in set(starts):
            raise ValueError(f"sequence_start_pos={sequence_start_pos} is invalid, valid starts begin {starts[:12]}")
        old_scene = int(self.scene_id)
        old_segment = int(self.segment_id)
        old_start = int(self.sequence_start_pos)
        self.scene_id = int(scene_id)
        self.segment_id = int(segment_id)
        self.sequence_start_pos = int(sequence_start_pos)
        self._reset_episode_state(reason=str(reason), rebuild_spec=True)
        self._emit(
            {
                "type": "demo_scope_change",
                "old_scene_id": int(old_scene),
                "old_segment_id": int(old_segment),
                "old_sequence_start_pos": int(old_start),
                "new_scene_id": int(scene_id),
                "new_segment_id": int(segment_id),
                "new_sequence_start_pos": int(sequence_start_pos),
                "reason": str(reason),
                "manual": True,
                "model_update": False,
            }
        )
        return self.materialize_current_batch_without_advance()

    def set_scope(self, scene_id: int, segment_id: int) -> Dict[str, Any]:
        starts = self._window_starts(int(scene_id), int(segment_id))
        if len(starts) == 0:
            raise ValueError(f"scene_id={scene_id} segment_id={segment_id} has no valid starts")
        return self._set_scope_and_start(
            scene_id=int(scene_id),
            segment_id=int(segment_id),
            sequence_start_pos=int(starts[0]),
            reason="set_scope",
        )

    def set_sequence_start_pos(self, sequence_start_pos: int) -> Dict[str, Any]:
        return self._set_scope_and_start(
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            sequence_start_pos=int(sequence_start_pos),
            reason="set_sequence_start_pos",
        )

    def set_scene(self, scene_id: int) -> Dict[str, Any]:
        seg_ids = self._ordered_segment_ids(int(scene_id))
        if len(seg_ids) == 0:
            raise ValueError(f"scene_id={scene_id} has no segments")
        return self.set_scope(int(scene_id), int(seg_ids[0]))

    def set_segment(self, segment_id: int) -> Dict[str, Any]:
        return self.set_scope(int(self.scene_id), int(segment_id))

    def next_scene(self) -> Dict[str, Any]:
        idx = self._scene_ids.index(int(self.scene_id)) + 1
        if idx >= len(self._scene_ids):
            if not self.wrap_scene:
                raise ValueError("next_scene overflow and wrap_scene=false")
            idx = 0
        return self.set_scene(int(self._scene_ids[idx]))

    def prev_scene(self) -> Dict[str, Any]:
        idx = self._scene_ids.index(int(self.scene_id)) - 1
        if idx < 0:
            if not self.wrap_scene:
                raise ValueError("prev_scene overflow and wrap_scene=false")
            idx = len(self._scene_ids) - 1
        return self.set_scene(int(self._scene_ids[idx]))

    def next_segment(self) -> Dict[str, Any]:
        seg_ids = self._ordered_segment_ids(int(self.scene_id))
        idx = seg_ids.index(int(self.segment_id)) + 1
        if idx >= len(seg_ids):
            if not self.wrap_segment:
                raise ValueError("next_segment overflow and wrap_segment=false")
            idx = 0
        return self.set_segment(int(seg_ids[idx]))

    def prev_segment(self) -> Dict[str, Any]:
        seg_ids = self._ordered_segment_ids(int(self.scene_id))
        idx = seg_ids.index(int(self.segment_id)) - 1
        if idx < 0:
            if not self.wrap_segment:
                raise ValueError("prev_segment overflow and wrap_segment=false")
            idx = len(seg_ids) - 1
        return self.set_segment(int(seg_ids[idx]))

    def resample_episode(self) -> Dict[str, Any]:
        starts = self._window_starts(int(self.scene_id), int(self.segment_id))
        if len(starts) == 0:
            raise ValueError("current scope has no valid episode starts")
        cur_idx = starts.index(int(self.sequence_start_pos)) if int(self.sequence_start_pos) in set(starts) else -1
        next_idx = cur_idx + 1
        if next_idx >= len(starts):
            if not self.wrap_episode:
                raise ValueError("resample_episode overflow and wrap_episode=false")
            next_idx = 0
        return self._set_scope_and_start(
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            sequence_start_pos=int(starts[next_idx]),
            reason="manual_episode_resample",
        )


def build_stage5_demo_scheduler_from_cfg(
    cfg: Any,
    dataset: MultiSceneDatasetV4,
    *,
    device: Any = None,
) -> Any:
    training_cfg = cfg.get("training") or {}
    seed = int(training_cfg.get("seed", 0))
    demo_cfg = cfg.get("demo") or {}
    scheduler_cfg = demo_cfg.get("scheduler") or {}
    if str(scheduler_cfg.get("type", "")).strip() == "eval_v8_stage5_6":
        return Stage5_6EvalDemoScheduler(dataset=dataset, cfg=cfg, seed=seed, device=device)
    return Stage5DemoScheduler(dataset=dataset, cfg=cfg, seed=seed)
