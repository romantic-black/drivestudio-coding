from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.iforward.random_window_batch import (
    RANDOM_WINDOW_ASSEMBLY_MODE,
    RANDOM_WINDOW_MODEL_FAMILY,
    RANDOM_WINDOW_SCHEDULER_VERSION,
    IForwardRandomWindowPlan,
    IForwardRandomWindowStep,
)

ImageRef = Tuple[int, int]


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    if hasattr(node, "get"):
        value = node.get(key, default)
        return default if value is None else value
    if hasattr(node, key):
        value = getattr(node, key)
        return default if value is None else value
    return default


def stable_window_hash(scene_id: int, segment_id: int, block_ids: Sequence[int]) -> int:
    text = f"{int(scene_id)}:{int(segment_id)}:" + ",".join(str(int(x)) for x in block_ids)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _dedupe_refs_keep_order(refs: Sequence[ImageRef]) -> List[ImageRef]:
    out: List[ImageRef] = []
    seen = set()
    for ref in refs:
        item = (int(ref[0]), int(ref[1]))
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class IForwardRandomWindowScheduler:
    def __init__(
        self,
        *,
        dataset: Any,
        traversal_cfg: Optional[Any] = None,
        segment_cfg: Optional[Any] = None,
        episode_cfg: Optional[Any] = None,
        rollout_cfg: Optional[Any] = None,
        evidence_cfg: Optional[Any] = None,
        supervision_cfg: Optional[Any] = None,
        memory_cfg: Optional[Any] = None,
        loss_timing_cfg: Optional[Any] = None,
        preload_cfg: Optional[Any] = None,
        include_test: bool = False,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
        seed: Optional[int] = None,
        fail_fast: bool = True,
        fixed_window_starts: Optional[Sequence[int]] = None,
    ) -> None:
        self.dataset = dataset
        if hasattr(self.dataset, "initialize") and not bool(getattr(self.dataset, "_initialized", False)):
            self.dataset.initialize()
        self.traversal_cfg = dict(traversal_cfg or {})
        self.segment_cfg = dict(segment_cfg or {})
        self.episode_cfg = dict(episode_cfg or {})
        self.rollout_cfg = dict(rollout_cfg or {})
        self.evidence_cfg = dict(evidence_cfg or {})
        self.supervision_cfg = dict(supervision_cfg or {})
        self.memory_cfg = dict(memory_cfg or {})
        self.loss_timing_cfg = dict(loss_timing_cfg or {})
        self.preload_cfg = dict(preload_cfg or {})
        self.include_test = bool(include_test)
        self.fixed_scene_id = fixed_scene_id
        self.fixed_segment_id = fixed_segment_id
        if self.fixed_scene_id is None:
            self.fixed_scene_id = _cfg_get(self.traversal_cfg, "fixed_scene_id", None)
        if self.fixed_segment_id is None:
            self.fixed_segment_id = _cfg_get(self.traversal_cfg, "fixed_segment_id", None)
        self.fixed_scene_id = None if self.fixed_scene_id is None else int(self.fixed_scene_id)
        self.fixed_segment_id = None if self.fixed_segment_id is None else int(self.fixed_segment_id)
        seed_raw = seed if seed is not None else _cfg_get(self.traversal_cfg, "seed", 41)
        self.seed = 41 if seed_raw is None else int(seed_raw)
        self.fail_fast = bool(fail_fast)
        self.blocks_per_rollout = int(_cfg_get(self.rollout_cfg, "blocks_per_rollout", 4))
        self.repeats_per_block = int(_cfg_get(self.rollout_cfg, "repeats_per_block", 2))
        self.rollouts_per_episode = int(_cfg_get(self.episode_cfg, "rollouts_per_episode", 8))
        self.min_blocks = int(_cfg_get(self.segment_cfg, "min_blocks", self.blocks_per_rollout))
        self.fixed_window_starts = [int(x) for x in list(fixed_window_starts or [])]
        self._validate_cfg()
        self.rng = random.Random(self.seed)
        self.global_step = 0
        self.epoch_idx = -1
        self._episode_id_next = 0
        self._rollout_id_global = 0
        self._episode_plan: List[Dict[str, Any]] = []
        self._episode_plan_cursor = 0
        self._current_episode: Optional[Dict[str, Any]] = None
        self._pending_events: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {}
        self._rebuild_epoch_plan()

    def _validate_cfg(self) -> None:
        if str(_cfg_get(self.segment_cfg, "source_mode", "keyframes")) != "keyframes":
            raise ValueError("scheduler_iforward_random_window.segment.source_mode must be keyframes")
        if int(self.blocks_per_rollout) != 4:
            raise ValueError("scheduler_iforward_random_window.rollout.blocks_per_rollout must be 4")
        if int(self.repeats_per_block) != 2:
            raise ValueError("scheduler_iforward_random_window.rollout.repeats_per_block must be 2")
        if str(_cfg_get(self.rollout_cfg, "window_policy", "random_with_replacement")) not in {
            "random_with_replacement",
            "fixed_random_with_replacement",
        }:
            raise ValueError("scheduler_iforward_random_window.rollout.window_policy must be random_with_replacement")
        if str(_cfg_get(self.rollout_cfg, "delivery_order", "chronological")) != "chronological":
            raise ValueError("scheduler_iforward_random_window.rollout.delivery_order must be chronological")
        if int(self.rollouts_per_episode) < 1:
            raise ValueError("scheduler_iforward_random_window.episode.rollouts_per_episode must be >= 1")
        if int(self.min_blocks) < int(self.blocks_per_rollout):
            raise ValueError("scheduler_iforward_random_window.segment.min_blocks must be >= blocks_per_rollout")

    def _emit(self, event: Dict[str, Any]) -> None:
        self._pending_events.append(dict(event))

    def pop_events(self) -> List[Dict[str, Any]]:
        out = list(self._pending_events)
        self._pending_events.clear()
        return out

    def get_current_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _scene_ids(self) -> List[int]:
        if self.fixed_scene_id is not None:
            return [int(self.fixed_scene_id)]
        return [int(x) for x in list(self.dataset.list_training_scene_ids())]

    def _segment_ids(self, scene_id: int) -> List[int]:
        if self.fixed_segment_id is not None:
            return [int(self.fixed_segment_id)]
        return [int(x) for x in list(self.dataset.list_segment_ids(int(scene_id)))]

    def _rebuild_epoch_plan(self) -> None:
        self.epoch_idx += 1
        scene_ids = self._scene_ids()
        if str(_cfg_get(self.traversal_cfg, "scene_order", "shuffle_per_epoch")) == "shuffle_per_epoch":
            self.rng.shuffle(scene_ids)
        specs: List[Dict[str, Any]] = []
        for scene_id in scene_ids:
            segment_ids = self._segment_ids(int(scene_id))
            if str(_cfg_get(self.traversal_cfg, "segment_order", "shuffle_per_epoch")) == "shuffle_per_epoch":
                self.rng.shuffle(segment_ids)
            for segment_id in segment_ids:
                sidx = self.dataset.get_segment_index(int(scene_id), int(segment_id))
                keyframes = [int(x) for x in list(getattr(sidx, "keyframe_indices", []) or [])]
                if len(keyframes) < int(self.min_blocks):
                    continue
                specs.append(
                    {
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "keyframes": keyframes,
                    }
                )
        if not specs:
            raise ValueError("scheduler_iforward_random_window found no valid segments with >= 4 keyframe blocks")
        self._episode_plan = specs
        self._episode_plan_cursor = 0

    @staticmethod
    def _keyframe_train_frames(sidx: Any, keyframe_idx: int) -> List[int]:
        train_set = set(int(x) for x in getattr(sidx, "train_frame_set", set()))
        raw = list(dict(getattr(sidx, "keyframe_to_frames", {}) or {}).get(int(keyframe_idx), []) or [])
        frames = [int(x) for x in raw if int(x) in train_set]
        if frames:
            return sorted(set(frames))
        if int(keyframe_idx) in train_set:
            return [int(keyframe_idx)]
        return []

    def _episode_seed(self, scene_id: int, segment_id: int, episode_id: int) -> int:
        return int(self.seed) + int(scene_id) * 10007 + int(segment_id) * 1009 + int(self.epoch_idx) * 9176 + int(episode_id) * 131

    def _start_next_episode(self) -> Dict[str, Any]:
        if self._episode_plan_cursor >= len(self._episode_plan):
            self._rebuild_epoch_plan()
        spec = dict(self._episode_plan[int(self._episode_plan_cursor)])
        self._episode_plan_cursor += 1
        episode_id = int(self._episode_id_next)
        self._episode_id_next += 1
        rng = random.Random(self._episode_seed(int(spec["scene_id"]), int(spec["segment_id"]), episode_id))
        episode = {
            "scene_id": int(spec["scene_id"]),
            "segment_id": int(spec["segment_id"]),
            "episode_id": int(episode_id),
            "keyframes": [int(x) for x in list(spec["keyframes"])],
            "rollout_idx": 0,
            "window_counts": {},
            "short_window_history_refs": [],
            "episode_rng_state": rng.getstate(),
        }
        self._current_episode = episode
        self._emit(
            {
                "type": "episode_begin",
                "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(episode["scene_id"]),
                "segment_id": int(episode["segment_id"]),
                "episode_id": int(episode_id),
                "episode_num_blocks": int(len(episode["keyframes"])),
                "rollouts_per_episode": int(self.rollouts_per_episode),
            }
        )
        return episode

    def _ensure_episode(self) -> Dict[str, Any]:
        if self._current_episode is None:
            return self._start_next_episode()
        return self._current_episode

    def _episode_rng(self, episode: Dict[str, Any]) -> random.Random:
        rng = random.Random()
        rng.setstate(episode["episode_rng_state"])
        return rng

    def _store_episode_rng(self, episode: Dict[str, Any], rng: random.Random) -> None:
        episode["episode_rng_state"] = rng.getstate()

    @staticmethod
    def _refs_for_frames(num_cams: int, frames: Sequence[int]) -> List[ImageRef]:
        return [(int(frame_idx), int(cam_idx)) for frame_idx in frames for cam_idx in range(int(num_cams))]

    def _sample_frame_for_keyframe(self, sidx: Any, keyframe_idx: int, rng: random.Random) -> int:
        candidates = self._keyframe_train_frames(sidx, int(keyframe_idx))
        if not candidates:
            raise ValueError(
                "scheduler_iforward_random_window keyframe has no train frames: "
                f"scene={getattr(sidx, 'scene_id', '?')} segment={getattr(sidx, 'segment_id', '?')} keyframe={int(keyframe_idx)}"
            )
        return int(rng.choice(candidates))

    def _sample_window_start(self, episode: Dict[str, Any], rng: random.Random) -> int:
        rollout_idx = int(episode["rollout_idx"])
        max_start = int(len(episode["keyframes"]) - self.blocks_per_rollout)
        if rollout_idx < len(self.fixed_window_starts):
            start = int(self.fixed_window_starts[rollout_idx])
            if start < 0 or start > max_start:
                raise ValueError(f"fixed random-window validation start out of range: {start} not in [0,{max_start}]")
            return start
        return int(rng.randint(0, max_start))

    def _sample_nearby_refs(
        self,
        *,
        sidx: Any,
        episode: Dict[str, Any],
        window_block_ids: Sequence[int],
        rng: random.Random,
    ) -> Tuple[List[int], List[ImageRef]]:
        nearby_cfg = dict(_cfg_get(self.supervision_cfg, "nearby", {}) or {})
        if not bool(_cfg_get(nearby_cfg, "enable", True)):
            return [], []
        frames_per_rollout = int(_cfg_get(nearby_cfg, "frames_per_rollout", 1))
        if frames_per_rollout <= 0:
            return [], []
        num_cams = int(getattr(sidx, "num_cams", 1))
        keyframes = [int(x) for x in list(episode["keyframes"])]
        input_blocks = set(int(x) for x in window_block_ids)
        candidate_blocks = [idx for idx in range(len(keyframes)) if idx not in input_blocks]
        if not candidate_blocks:
            return [], []
        chosen_blocks = rng.sample(candidate_blocks, k=min(frames_per_rollout, len(candidate_blocks)))
        frames: List[int] = []
        for block_id in chosen_blocks:
            frames.append(self._sample_frame_for_keyframe(sidx, int(keyframes[int(block_id)]), rng))
        refs = self._refs_for_frames(num_cams, frames)
        max_refs = int(_cfg_get(nearby_cfg, "max_refs_per_rollout", len(refs)))
        if max_refs > 0:
            refs = refs[:max_refs]
        return [int(x) for x in frames], refs

    def _build_rollout_plan(self, episode: Dict[str, Any]) -> IForwardRandomWindowPlan:
        rng = self._episode_rng(episode)
        sidx = self.dataset.get_segment_index(int(episode["scene_id"]), int(episode["segment_id"]))
        num_cams = int(getattr(sidx, "num_cams", 1))
        keyframes = [int(x) for x in list(episode["keyframes"])]
        start = self._sample_window_start(episode, rng)
        end = int(start + self.blocks_per_rollout)
        block_ids = list(range(int(start), int(end)))
        window_keyframes = [int(keyframes[int(idx)]) for idx in block_ids]
        input_frames = [self._sample_frame_for_keyframe(sidx, int(kf), rng) for kf in window_keyframes]
        evidence_refs = self._refs_for_frames(num_cams, input_frames)
        current_latest_refs = self._refs_for_frames(num_cams, [input_frames[-1]])
        in_rollout_history_refs = self._refs_for_frames(num_cams, input_frames[:-1])
        short_history_cfg = dict(_cfg_get(self.supervision_cfg, "short_window_history", {}) or {})
        short_history_max = int(_cfg_get(short_history_cfg, "max_entries", 24))
        if short_history_max > 0:
            short_window_history_refs = [
                tuple(ref)
                for ref in list(episode.get("short_window_history_refs", []) or [])[-int(short_history_max) :]
            ]
        else:
            short_window_history_refs = []
        nearby_frames, nearby_refs = self._sample_nearby_refs(
            sidx=sidx,
            episode=episode,
            window_block_ids=block_ids,
            rng=rng,
        )
        self._store_episode_rng(episode, rng)

        target_refs: List[ImageRef] = []
        target_roles: List[str] = []
        for role, refs in (
            ("current_latest", current_latest_refs),
            ("in_rollout_history", in_rollout_history_refs),
            ("short_window_history", short_window_history_refs),
            ("nearby", nearby_refs),
        ):
            for ref in refs:
                target_refs.append(tuple(ref))
                target_roles.append(str(role))
        target_refs = _dedupe_refs_keep_order(target_refs)
        role_by_ref = {}
        role_sources = (
            ("current_latest", current_latest_refs),
            ("in_rollout_history", in_rollout_history_refs),
            ("short_window_history", short_window_history_refs),
            ("nearby", nearby_refs),
        )
        for role, refs in role_sources:
            for ref in refs:
                role_by_ref.setdefault(tuple(ref), str(role))
        target_roles = [role_by_ref[tuple(ref)] for ref in target_refs]

        steps: List[IForwardRandomWindowStep] = []
        inner_k = int(self.blocks_per_rollout * self.repeats_per_block)
        for block_pos, (block_id, keyframe_idx, frame_idx) in enumerate(zip(block_ids, window_keyframes, input_frames)):
            refs = self._refs_for_frames(num_cams, [int(frame_idx)])
            for repeat_idx in range(int(self.repeats_per_block)):
                step_idx = len(steps)
                steps.append(
                    IForwardRandomWindowStep(
                        step_idx=int(step_idx),
                        block_id=int(block_id),
                        block_pos_in_window=int(block_pos),
                        repeat_idx=int(repeat_idx),
                        global_k=int(step_idx),
                        source_frame_idx=int(frame_idx),
                        source_keyframe_idx=int(keyframe_idx),
                        evidence_refs=[tuple(x) for x in refs],
                        commit_observation_memory=bool(int(repeat_idx) == 0),
                        update_optimizer_memory=True,
                        is_frame_exit=bool(int(repeat_idx) == int(self.repeats_per_block) - 1),
                        rollout_pos_code=float(step_idx) / float(max(inner_k - 1, 1)),
                        frame_pos_code=float(block_pos) / float(max(self.blocks_per_rollout - 1, 1)),
                        repeat_pos_code=float(repeat_idx) / float(max(self.repeats_per_block - 1, 1)),
                    )
                )

        window_hash = stable_window_hash(int(episode["scene_id"]), int(episode["segment_id"]), block_ids)
        counts = dict(episode.get("window_counts", {}) or {})
        prev_count = int(counts.get(int(window_hash), 0))
        counts[int(window_hash)] = int(prev_count + 1)
        episode["window_counts"] = counts
        rollout_idx = int(episode["rollout_idx"])
        episode_end = bool(rollout_idx + 1 >= int(self.rollouts_per_episode))
        request_meta = {
            "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
            "model_family": RANDOM_WINDOW_MODEL_FAMILY,
            "assembly_mode": RANDOM_WINDOW_ASSEMBLY_MODE,
            "scene_id": int(episode["scene_id"]),
            "segment_id": int(episode["segment_id"]),
            "episode_id": int(episode["episode_id"]),
            "episode_idx_global": int(episode["episode_id"]),
            "rollout_id_global": int(self._rollout_id_global),
            "rollout_idx_in_episode": int(rollout_idx),
            "rollouts_per_episode": int(self.rollouts_per_episode),
            "window_start": int(start),
            "window_end": int(end),
            "window_block_ids": [int(x) for x in block_ids],
            "window_hash": int(window_hash),
            "window_revisit_count": int(prev_count),
            "unique_windows_seen": int(len(counts)),
            "is_repeated_window": bool(prev_count > 0),
            "blocks_per_rollout": int(self.blocks_per_rollout),
            "repeats_per_block": int(self.repeats_per_block),
            "inner_K": int(inner_k),
            "source_image_refs": [tuple(x) for x in evidence_refs],
            "target_image_refs": [tuple(x) for x in target_refs],
            "target_image_roles": [str(x) for x in target_roles],
        }
        plan = IForwardRandomWindowPlan(
            scheduler_version=RANDOM_WINDOW_SCHEDULER_VERSION,
            model_family=RANDOM_WINDOW_MODEL_FAMILY,
            scene_id=int(episode["scene_id"]),
            segment_id=int(episode["segment_id"]),
            episode_id=int(episode["episode_id"]),
            rollout_id_global=int(self._rollout_id_global),
            rollout_idx_in_episode=int(rollout_idx),
            rollouts_per_episode=int(self.rollouts_per_episode),
            window_start=int(start),
            window_end=int(end),
            window_block_ids=[int(x) for x in block_ids],
            window_keyframe_indices=[int(x) for x in window_keyframes],
            window_frame_indices=[int(x) for x in input_frames],
            window_hash=int(window_hash),
            window_revisit_count=int(prev_count),
            unique_windows_seen=int(len(counts)),
            is_repeated_window=bool(prev_count > 0),
            blocks_per_rollout=int(self.blocks_per_rollout),
            repeats_per_block=int(self.repeats_per_block),
            inner_K=int(inner_k),
            reset_scene_state_before_rollout=bool(rollout_idx == 0),
            carry_scene_state_after_rollout=bool(not episode_end),
            episode_end_after_rollout=bool(episode_end),
            detach_graph_after_rollout=True,
            steps=steps,
            evidence_refs_flat=[tuple(x) for x in evidence_refs],
            target_refs_flat=[tuple(x) for x in target_refs],
            target_roles_flat=[str(x) for x in target_roles],
            current_latest_refs=[tuple(x) for x in current_latest_refs],
            in_rollout_history_refs=[tuple(x) for x in in_rollout_history_refs],
            short_window_history_refs=[tuple(x) for x in short_window_history_refs],
            nearby_refs=[tuple(x) for x in nearby_refs],
            input_frame_indices=[int(x) for x in input_frames],
            input_keyframe_indices=[int(x) for x in window_keyframes],
            nearby_frame_indices=[int(x) for x in nearby_frames],
            request_meta=request_meta,
            leakage_check={
                "nearby_evidence_overlap": int(len(set(nearby_refs) & set(evidence_refs))),
                "nearby_input_frame_overlap": int(len(set(nearby_frames) & set(input_frames))),
            },
        )
        next_short_history = list(short_window_history_refs) + [tuple(x) for x in evidence_refs]
        if short_history_max > 0 and len(next_short_history) > short_history_max:
            next_short_history = next_short_history[-int(short_history_max) :]
        if short_history_max <= 0:
            next_short_history = []
        episode["short_window_history_refs"] = [tuple(x) for x in next_short_history]
        return plan

    def _batch_from_plan(self, plan: IForwardRandomWindowPlan) -> Dict[str, Any]:
        if not hasattr(self.dataset, "_assemble_segment_batch_from_iforward_random_window_request"):
            raise ValueError("dataset must implement _assemble_segment_batch_from_iforward_random_window_request")
        return self.dataset._assemble_segment_batch_from_iforward_random_window_request(
            scene_id=int(plan.scene_id),
            segment_id=int(plan.segment_id),
            plan=plan,
            include_test=bool(self.include_test),
        )

    def next_batch(self) -> Dict[str, Any]:
        episode = self._ensure_episode()
        plan = self._build_rollout_plan(episode)
        batch = self._batch_from_plan(plan)
        self._last_info = {
            "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "scene_id": int(plan.scene_id),
            "segment_id": int(plan.segment_id),
            "episode_id": int(plan.episode_id),
            "episode_idx_global": int(plan.episode_id),
            "rollout_id_global": int(plan.rollout_id_global),
            "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
            "rollouts_per_episode": int(plan.rollouts_per_episode),
            "block_idx_global": int(plan.rollout_id_global),
            "block_idx_in_episode": int(plan.rollout_idx_in_episode),
            "block_idx_in_segment": int(plan.window_start),
            "source_frame_idx": int(plan.window_frame_indices[0]),
            "source_keyframe_idx": int(plan.window_keyframe_indices[0]),
            "source_image_ref": tuple(plan.evidence_refs_flat[0]),
            "target_image_refs": [tuple(x) for x in plan.target_refs_flat],
            "window_start": int(plan.window_start),
            "window_end": int(plan.window_end),
            "window_block_ids": [int(x) for x in plan.window_block_ids],
            "window_hash": int(plan.window_hash),
            "window_revisit_count": int(plan.window_revisit_count),
            "unique_windows_seen": int(plan.unique_windows_seen),
            "is_repeated_window": bool(plan.is_repeated_window),
            "U": int(plan.blocks_per_rollout),
            "T_steps": int(plan.blocks_per_rollout),
            "R_steps": int(plan.repeats_per_block),
            "K_steps": int(plan.inner_K),
            "K_steps_effective": int(plan.inner_K),
            "inner_K": int(plan.inner_K),
            "shape_name": "random_window_b4_r2",
        }
        self._emit(
            {
                "type": "rollout_batch_emitted",
                "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
                "global_step": int(self.global_step),
                "scene_id": int(plan.scene_id),
                "segment_id": int(plan.segment_id),
                "episode_id": int(plan.episode_id),
                "rollout_id_global": int(plan.rollout_id_global),
                "rollout_idx_in_episode": int(plan.rollout_idx_in_episode),
                "window_start": int(plan.window_start),
                "window_hash": int(plan.window_hash),
            }
        )
        episode["rollout_idx"] = int(episode["rollout_idx"]) + 1
        self.global_step += 1
        self._rollout_id_global += 1
        if bool(plan.episode_end_after_rollout):
            self._emit(
                {
                    "type": "episode_end",
                    "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
                    "global_step": int(self.global_step),
                    "scene_id": int(plan.scene_id),
                    "segment_id": int(plan.segment_id),
                    "episode_id": int(plan.episode_id),
                    "rollout_id_global": int(plan.rollout_id_global),
                    "reason": "rollouts_per_episode_reached",
                }
            )
            self._current_episode = None
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_class": type(self).__name__,
            "scheduler_version": RANDOM_WINDOW_SCHEDULER_VERSION,
            "global_step": int(self.global_step),
            "epoch_idx": int(self.epoch_idx),
            "episode_id_next": int(self._episode_id_next),
            "rollout_id_global": int(self._rollout_id_global),
            "episode_plan": copy.deepcopy(self._episode_plan),
            "episode_plan_cursor": int(self._episode_plan_cursor),
            "current_episode": copy.deepcopy(self._current_episode),
            "pending_events": copy.deepcopy(self._pending_events),
            "last_info": copy.deepcopy(self._last_info),
            "rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if str(state.get("scheduler_version", "")) != RANDOM_WINDOW_SCHEDULER_VERSION:
            raise ValueError("random-window scheduler state version mismatch")
        self.global_step = int(state["global_step"])
        self.epoch_idx = int(state["epoch_idx"])
        self._episode_id_next = int(state["episode_id_next"])
        self._rollout_id_global = int(state["rollout_id_global"])
        self._episode_plan = copy.deepcopy(list(state["episode_plan"]))
        self._episode_plan_cursor = int(state["episode_plan_cursor"])
        self._current_episode = copy.deepcopy(state.get("current_episode"))
        self._pending_events = copy.deepcopy(list(state.get("pending_events", [])))
        self._last_info = copy.deepcopy(dict(state.get("last_info", {})))
        self.rng.setstate(copy.deepcopy(state["rng_state"]))


__all__ = [
    "IForwardRandomWindowScheduler",
    "stable_window_hash",
]
