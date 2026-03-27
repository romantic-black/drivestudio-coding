from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from datasets.multi_scene_dataset import MultiSceneDataset

logger = logging.getLogger(__name__)


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


class MultiSceneDatasetV2(MultiSceneDataset):
    """
    Scheduler-v2 dataset facade.

    Reuses MultiSceneDataset loading/caching/batch assembly internals, but moves
    source/target sampling responsibility to external schedulers.
    """

    def get_segment_batch_from_frames(
        self,
        scene_id: int,
        segment_id: int,
        source_frame_idx: int,
        target_frame_indices: List[int],
        include_test: bool = False,
        test_frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if len(target_frame_indices) == 0:
            raise ValueError("target_frame_indices must not be empty")
        if target_frame_indices[0] != source_frame_idx:
            raise ValueError(
                "target_frame_indices[0] must equal source_frame_idx for scheduler v2 semantics"
            )

        scene_data = self._ensure_scene_loaded(int(scene_id))
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segments = scene_data.get("segments", [])
        if int(segment_id) < 0 or int(segment_id) >= len(segments):
            raise ValueError(f"segment_id={segment_id} out of range for scene {scene_id}")
        segment = segments[int(segment_id)]
        segment_frames = set(segment.get("frame_indices", []))
        if int(source_frame_idx) not in segment_frames:
            raise ValueError(
                f"source_frame_idx={source_frame_idx} is not in scene={scene_id} segment={segment_id}"
            )
        for frame_idx in target_frame_indices:
            if int(frame_idx) not in segment_frames:
                raise ValueError(
                    f"target frame {frame_idx} is not in scene={scene_id} segment={segment_id}"
                )

        keyframe_segments = scene_data.get("keyframe_segments", [])
        seg_keyframes = segment.get("keyframe_indices", [])

        def _frame_to_keyframe(frame_idx: int) -> int:
            for keyframe_idx in seg_keyframes:
                if keyframe_idx < 0 or keyframe_idx >= len(keyframe_segments):
                    continue
                if int(frame_idx) in keyframe_segments[keyframe_idx]:
                    return int(keyframe_idx)
            raise ValueError(
                f"frame_idx={frame_idx} cannot be mapped to any keyframe in scene={scene_id} segment={segment_id}"
            )

        source_keyframe_idx = _frame_to_keyframe(int(source_frame_idx))
        target_keyframe_indices = [_frame_to_keyframe(int(fidx)) for fidx in target_frame_indices]
        source_keyframe_indices_patched = [source_keyframe_idx]
        target_keyframe_indices_patched = list(target_keyframe_indices)

        # Build a per-keyframe deterministic frame plan to avoid relying on
        # get_segment_batch() internal call order/count.
        planned_frames_by_kf: Dict[int, List[int]] = {}
        planned_frames_by_kf[source_keyframe_idx] = [int(source_frame_idx)]
        for frame_idx in target_frame_indices:
            kf_idx = _frame_to_keyframe(int(frame_idx))
            planned_frames_by_kf.setdefault(kf_idx, []).append(int(frame_idx))

        planned_cursor_by_kf: Dict[int, int] = {kf: 0 for kf in planned_frames_by_kf}

        orig_select_st = self._select_source_and_target_keyframes
        orig_select_frame = self._select_frame_from_keyframe

        def _patched_select_source_and_target_keyframes(
            segment: Dict[str, Any],
            num_source_keyframes: int,
            num_target_keyframes: int,
        ) -> Tuple[List[int], List[int]]:
            del segment
            if num_source_keyframes != 1:
                raise ValueError(
                    f"Scheduler v2 only supports num_source_keyframes=1, got {num_source_keyframes}"
                )
            if num_target_keyframes != len(target_keyframe_indices_patched):
                raise ValueError(
                    "num_target_keyframes mismatch: "
                    f"expected {len(target_keyframe_indices_patched)}, got {num_target_keyframes}"
                )
            return source_keyframe_indices_patched, target_keyframe_indices_patched

        def _patched_select_frame_from_keyframe(keyframe_segment: List[int]) -> int:
            matched_kf: Optional[int] = None
            for kf_idx in seg_keyframes:
                if kf_idx < 0 or kf_idx >= len(keyframe_segments):
                    continue
                if keyframe_segments[kf_idx] == keyframe_segment:
                    matched_kf = int(kf_idx)
                    break
            if matched_kf is None:
                raise ValueError(
                    "Cannot map keyframe_segment to keyframe index in get_segment_batch_from_frames"
                )
            frames = planned_frames_by_kf.get(matched_kf)
            if not frames:
                raise ValueError(f"No planned frame for keyframe={matched_kf}")
            cursor = planned_cursor_by_kf[matched_kf]
            frame_idx = int(frames[cursor % len(frames)])
            planned_cursor_by_kf[matched_kf] = cursor + 1
            return frame_idx

        try:
            self._select_source_and_target_keyframes = _patched_select_source_and_target_keyframes
            self._select_frame_from_keyframe = _patched_select_frame_from_keyframe
            batch = self.get_segment_batch(
                scene_id=int(scene_id),
                segment_id=int(segment_id),
                include_test=bool(include_test),
            )
            if include_test and test_frame_indices is not None:
                self._overwrite_test_views_from_explicit_frames(
                    batch=batch,
                    scene_id=int(scene_id),
                    test_frame_indices=[int(x) for x in test_frame_indices],
                )
            return batch
        finally:
            self._select_source_and_target_keyframes = orig_select_st
            self._select_frame_from_keyframe = orig_select_frame

    def _overwrite_test_views_from_explicit_frames(
        self,
        batch: Dict[str, Any],
        scene_id: int,
        test_frame_indices: List[int],
    ) -> None:
        scene_data = self._ensure_scene_loaded(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        scene_dataset = scene_data["dataset"]
        num_cams = scene_dataset.num_cams

        test_images = []
        test_extrinsics = []
        test_intrinsics = []
        test_depths = []
        test_frame_idxs = []
        test_cam_idxs = []
        test_sky_masks = []
        has_test_sky_mask = False
        test_egocar_masks = []
        has_test_egocar_mask = False

        for frame_idx in test_frame_indices:
            for cam_idx in range(num_cams):
                frame_data = self.get_frame_data(scene_id, int(frame_idx), int(cam_idx))
                test_images.append(frame_data["image"])
                test_extrinsics.append(frame_data["extrinsic"])
                test_intrinsics.append(frame_data["intrinsic"])
                test_depths.append(frame_data["depth"])
                sky_mask = frame_data.get("sky_mask")
                if sky_mask is not None:
                    has_test_sky_mask = True
                test_sky_masks.append(sky_mask)
                egocar_mask = frame_data.get("egocar_mask")
                if egocar_mask is not None:
                    has_test_egocar_mask = True
                test_egocar_masks.append(egocar_mask)
                test_frame_idxs.append(int(frame_idx))
                test_cam_idxs.append(int(cam_idx))

        if len(test_images) == 0:
            batch.pop("test", None)
            return

        # Reuse parent seg0 transform info already written to batch.
        segment_first_pose = batch.get("segment_first_pose")
        if segment_first_pose is None:
            raise ValueError("batch missing segment_first_pose for deterministic test overwrite")
        world_to_seg0 = segment_first_pose.to(device=self.device, dtype=test_extrinsics[0].dtype).inverse()
        transformed = []
        for ext in test_extrinsics:
            ext_tensor = self._to_4x4_tensor(ext).to(device=self.device, dtype=world_to_seg0.dtype)
            transformed.append(world_to_seg0 @ ext_tensor)

        batch["test"] = {
            "image": torch.stack(test_images, dim=0),
            "extrinsics": torch.stack(transformed, dim=0),
            "intrinsics": torch.stack(test_intrinsics, dim=0),
            "depth": torch.stack(test_depths, dim=0),
            "frame_indices": torch.tensor(test_frame_idxs, dtype=torch.long),
            "cam_indices": torch.tensor(test_cam_idxs, dtype=torch.long),
        }
        if has_test_sky_mask:
            sky_stack = []
            for mask, img in zip(test_sky_masks, test_images):
                if mask is None:
                    h, w = img.shape[:2]
                    sky_stack.append(torch.zeros((h, w), dtype=torch.float32, device=self.device))
                else:
                    sky_stack.append(mask.to(self.device).float())
            batch["test"]["sky_mask"] = torch.stack(sky_stack, dim=0)
        if has_test_egocar_mask:
            ego_stack = []
            for mask, img in zip(test_egocar_masks, test_images):
                if mask is None:
                    h, w = img.shape[:2]
                    ego_stack.append(torch.zeros((h, w), dtype=torch.float32, device=self.device))
                else:
                    ego_stack.append(mask.to(self.device).float())
            batch["test"]["egocar_mask"] = torch.stack(ego_stack, dim=0)

    def create_train_scheduler_v2(
        self,
        alpha_steps_per_keyframe: float,
        min_steps_per_segment: int,
        max_steps_per_segment: int,
        source_hold_steps: int,
        num_target_frames_total: int,
        target_include_source: bool,
        include_test: bool,
        fixed_scene_id: Optional[int] = None,
        fixed_segment_id: Optional[int] = None,
    ) -> "TrainSchedulerV2":
        return TrainSchedulerV2(
            dataset=self,
            alpha_steps_per_keyframe=alpha_steps_per_keyframe,
            min_steps_per_segment=min_steps_per_segment,
            max_steps_per_segment=max_steps_per_segment,
            source_hold_steps=source_hold_steps,
            num_target_frames_total=num_target_frames_total,
            target_include_source=target_include_source,
            include_test=include_test,
            fixed_scene_id=fixed_scene_id,
            fixed_segment_id=fixed_segment_id,
        )

    def build_eval_manifest_v2(
        self,
        scene_ids: Sequence[int],
        num_target_frames_total: int,
        max_test_frames_per_segment: int = 0,
    ) -> List[Dict[str, Any]]:
        if num_target_frames_total < 1:
            raise ValueError("num_target_frames_total must be >= 1")
        if max_test_frames_per_segment < 0:
            raise ValueError("max_test_frames_per_segment must be >= 0")
        manifest: List[Dict[str, Any]] = []
        for scene_id in scene_ids:
            scene_data = self._ensure_scene_loaded(int(scene_id))
            if scene_data is None:
                raise ValueError(f"Eval scene {scene_id} cannot be loaded")
            keyframe_segments = scene_data["keyframe_segments"]
            for segment_id, segment in enumerate(scene_data["segments"]):
                seg_keyframes = list(segment["keyframe_indices"])
                if len(seg_keyframes) == 0:
                    raise ValueError(f"Eval scene={scene_id} segment={segment_id} has no keyframes")
                src_kf = seg_keyframes[len(seg_keyframes) // 2]
                src_frames = keyframe_segments[src_kf]
                if len(src_frames) == 0:
                    raise ValueError(
                        f"Eval scene={scene_id} segment={segment_id} source keyframe has no frames"
                    )
                source_frame = int(src_frames[len(src_frames) // 2])
                other_kfs = [kf for kf in seg_keyframes if kf != src_kf]
                extra_needed = num_target_frames_total - 1
                extra_frames: List[int] = []
                if extra_needed > 0:
                    if len(other_kfs) == 0:
                        raise ValueError(
                            "Eval manifest requires at least one non-source keyframe when "
                            f"num_target_frames_total={num_target_frames_total}"
                        )
                    cursor = 0
                    while len(extra_frames) < extra_needed:
                        kf = other_kfs[cursor % len(other_kfs)]
                        kf_frames = keyframe_segments[kf]
                        if len(kf_frames) == 0:
                            raise ValueError(
                                f"Eval scene={scene_id} segment={segment_id} keyframe={kf} has no frames"
                            )
                        extra_frames.append(int(kf_frames[len(kf_frames) // 2]))
                        cursor += 1
                test_frames = list(segment.get("test_frame_indices", []))
                if max_test_frames_per_segment > 0 and len(test_frames) > max_test_frames_per_segment:
                    stride = len(test_frames) / float(max_test_frames_per_segment)
                    selected_test: List[int] = []
                    for i in range(max_test_frames_per_segment):
                        idx = min(int(round(i * stride)), len(test_frames) - 1)
                        selected_test.append(int(test_frames[idx]))
                    test_frames = selected_test
                manifest.append(
                    {
                        "scene_id": int(scene_id),
                        "segment_id": int(segment_id),
                        "source_frame_idx": source_frame,
                        "target_frame_indices": [source_frame] + extra_frames,
                        "test_frame_indices": [int(x) for x in test_frames],
                    }
                )
        return manifest

    def create_eval_scheduler_v2(self, manifest: List[Dict[str, Any]], include_test: bool) -> "EvalSchedulerV2":
        return EvalSchedulerV2(dataset=self, eval_manifest=manifest, include_test=include_test)


class TrainSchedulerV2:
    def __init__(
        self,
        dataset: MultiSceneDatasetV2,
        alpha_steps_per_keyframe: float,
        min_steps_per_segment: int,
        max_steps_per_segment: int,
        source_hold_steps: int,
        num_target_frames_total: int,
        target_include_source: bool,
        include_test: bool,
        fixed_scene_id: Optional[int],
        fixed_segment_id: Optional[int],
    ) -> None:
        if alpha_steps_per_keyframe <= 0:
            raise ValueError("alpha_steps_per_keyframe must be > 0")
        if min_steps_per_segment < 1:
            raise ValueError("min_steps_per_segment must be >= 1")
        if max_steps_per_segment < min_steps_per_segment:
            raise ValueError("max_steps_per_segment must be >= min_steps_per_segment")
        if source_hold_steps < 1:
            raise ValueError("source_hold_steps must be >= 1")
        if num_target_frames_total < 1:
            raise ValueError("num_target_frames_total must be >= 1")
        if not target_include_source:
            raise ValueError("target_include_source must be true in scheduler v2")

        self.dataset = dataset
        self.alpha_steps_per_keyframe = float(alpha_steps_per_keyframe)
        self.min_steps_per_segment = int(min_steps_per_segment)
        self.max_steps_per_segment = int(max_steps_per_segment)
        self.source_hold_steps = int(source_hold_steps)
        self.num_target_frames_total = int(num_target_frames_total)
        self.include_test = bool(include_test)
        self.fixed_scene_id = int(fixed_scene_id) if fixed_scene_id is not None else None
        self.fixed_segment_id = int(fixed_segment_id) if fixed_segment_id is not None else None

        self.epoch_idx = 0
        self.global_step = 0
        self.epoch_plan: List[Dict[str, int]] = []
        self.plan_cursor = 0
        self.current_segment_state: Optional[Dict[str, int]] = None
        if not self.dataset._initialized:
            self.dataset.initialize()
        self.start_new_epoch()

    def _sample_new_source(self, scene_id: int, segment_id: int) -> Tuple[int, int]:
        scene_data = self.dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment = scene_data["segments"][segment_id]
        source_kf = int(random.choice(segment["keyframe_indices"]))
        frame_candidates = scene_data["keyframe_segments"][source_kf]
        if len(frame_candidates) == 0:
            raise ValueError(f"scene={scene_id} segment={segment_id} keyframe={source_kf} has no frames")
        source_frame = int(random.choice(frame_candidates))
        return source_kf, source_frame

    def _sample_target_frames(self, scene_id: int, segment_id: int, source_kf: int, source_frame: int) -> List[int]:
        scene_data = self.dataset.get_scene(scene_id)
        if scene_data is None:
            raise ValueError(f"Scene {scene_id} cannot be loaded")
        segment = scene_data["segments"][segment_id]
        extra_needed = self.num_target_frames_total - 1
        if extra_needed == 0:
            return [int(source_frame)]
        candidate_kfs = [int(kf) for kf in segment["keyframe_indices"] if int(kf) != int(source_kf)]
        if len(candidate_kfs) == 0:
            raise ValueError(
                "No non-source keyframes available for extra target sampling; "
                f"scene={scene_id} segment={segment_id} source_kf={source_kf}"
            )
        picked_kfs = candidate_kfs[:] if len(candidate_kfs) < extra_needed else random.sample(candidate_kfs, extra_needed)
        while len(picked_kfs) < extra_needed:
            picked_kfs.append(int(random.choice(candidate_kfs)))
        extra_frames: List[int] = []
        for keyframe_idx in picked_kfs:
            frame_candidates = scene_data["keyframe_segments"][keyframe_idx]
            if len(frame_candidates) == 0:
                raise ValueError(
                    f"scene={scene_id} segment={segment_id} keyframe={keyframe_idx} has no frames"
                )
            extra_frames.append(int(random.choice(frame_candidates)))
        return [int(source_frame)] + extra_frames

    def build_epoch_plan(self) -> None:
        if self.fixed_scene_id is not None:
            train_scene_ids = [self.fixed_scene_id]
        else:
            if len(getattr(self.dataset, "scene_training_queue", [])) == 0:
                self.dataset._initialize_training_queue()
            train_scene_ids = list(getattr(self.dataset, "scene_training_queue", []))
            if len(train_scene_ids) == 0:
                raise ValueError("No valid scenes in dataset.scene_training_queue")
            random.shuffle(train_scene_ids)
        plan: List[Dict[str, int]] = []
        for scene_id in train_scene_ids:
            scene_data = self.dataset.get_scene(scene_id)
            if scene_data is None:
                raise ValueError(f"Scene {scene_id} cannot be loaded")
            all_segment_ids = list(range(len(scene_data["segments"])))
            if self.fixed_segment_id is not None:
                if self.fixed_segment_id not in all_segment_ids:
                    raise ValueError(
                        f"fixed_segment_id={self.fixed_segment_id} out of range in scene={scene_id}"
                    )
                segment_ids = [self.fixed_segment_id]
            else:
                segment_ids = all_segment_ids
                random.shuffle(segment_ids)
            for segment_id in segment_ids:
                segment = scene_data["segments"][segment_id]
                num_keyframes = len(segment["keyframe_indices"])
                steps = _clamp_int(
                    int(round(self.alpha_steps_per_keyframe * num_keyframes)),
                    self.min_steps_per_segment,
                    self.max_steps_per_segment,
                )
                plan.append({"scene_id": int(scene_id), "segment_id": int(segment_id), "steps": int(steps)})
        if len(plan) == 0:
            raise ValueError("Epoch plan is empty")
        self.epoch_plan = plan
        self.plan_cursor = 0

    def start_new_epoch(self) -> None:
        self.epoch_idx += 1
        self.build_epoch_plan()
        self.current_segment_state = None

    def _enter_segment(self) -> None:
        item = self.epoch_plan[self.plan_cursor]
        source_kf, source_frame = self._sample_new_source(item["scene_id"], item["segment_id"])
        self.current_segment_state = {
            "scene_id": item["scene_id"],
            "segment_id": item["segment_id"],
            "steps": item["steps"],
            "local_step": 0,
            "source_kf": source_kf,
            "source_frame": source_frame,
            "source_block_step": 0,
        }

    def next_batch(self) -> Dict[str, Any]:
        if self.plan_cursor >= len(self.epoch_plan):
            self.start_new_epoch()
        if self.current_segment_state is None:
            self._enter_segment()
        state = self.current_segment_state
        if state is None:
            raise ValueError("Internal scheduler state is not initialized")
        if state["source_block_step"] >= self.source_hold_steps:
            source_kf, source_frame = self._sample_new_source(state["scene_id"], state["segment_id"])
            state["source_kf"] = source_kf
            state["source_frame"] = source_frame
            state["source_block_step"] = 0

        target_frames = self._sample_target_frames(
            state["scene_id"], state["segment_id"], state["source_kf"], state["source_frame"]
        )
        batch = self.dataset.get_segment_batch_from_frames(
            scene_id=state["scene_id"],
            segment_id=state["segment_id"],
            source_frame_idx=state["source_frame"],
            target_frame_indices=target_frames,
            include_test=self.include_test,
        )
        state["local_step"] += 1
        state["source_block_step"] += 1
        self.global_step += 1
        if state["local_step"] >= state["steps"]:
            self.plan_cursor += 1
            self.current_segment_state = None
        return batch

    def next_batch_in_epoch(self) -> Dict[str, Any]:
        """
        Return next batch only within current epoch.
        Raises StopIteration exactly at epoch boundary.
        """
        if self.plan_cursor >= len(self.epoch_plan):
            raise StopIteration("Current epoch has finished")
        return self.next_batch()

    def has_epoch_ended(self) -> bool:
        return self.plan_cursor >= len(self.epoch_plan)

    def get_current_info(self) -> Dict[str, int]:
        state = self.current_segment_state
        if state is None and self.plan_cursor < len(self.epoch_plan):
            item = self.epoch_plan[self.plan_cursor]
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": int(item["scene_id"]),
                "segment_id": int(item["segment_id"]),
                "segment_local_step": 0,
                "segment_step_budget": int(item["steps"]),
                "source_frame_idx": -1,
                "source_block_step": 0,
            }
        if state is None:
            return {
                "epoch_idx": int(self.epoch_idx),
                "global_step": int(self.global_step),
                "scene_id": -1,
                "segment_id": -1,
                "segment_local_step": 0,
                "segment_step_budget": 0,
                "source_frame_idx": -1,
                "source_block_step": 0,
            }
        return {
            "epoch_idx": int(self.epoch_idx),
            "global_step": int(self.global_step),
            "scene_id": int(state["scene_id"]),
            "segment_id": int(state["segment_id"]),
            "segment_local_step": int(state["local_step"]),
            "segment_step_budget": int(state["steps"]),
            "source_frame_idx": int(state["source_frame"]),
            "source_block_step": int(state["source_block_step"]),
        }


class EvalSchedulerV2:
    def __init__(self, dataset: MultiSceneDatasetV2, eval_manifest: List[Dict[str, Any]], include_test: bool) -> None:
        if len(eval_manifest) == 0:
            raise ValueError("eval_manifest must not be empty")
        self.dataset = dataset
        self.eval_manifest = eval_manifest
        self.include_test = bool(include_test)
        self.cursor = 0
        self._seen_keys = set()
        for item in self.eval_manifest:
            scene_id = int(item["scene_id"])
            segment_id = int(item["segment_id"])
            key = (scene_id, segment_id, int(item["source_frame_idx"]))
            if key in self._seen_keys:
                raise ValueError(f"Duplicate eval manifest entry: {key}")
            self._seen_keys.add(key)
            target = list(item.get("target_frame_indices", []))
            if len(target) == 0:
                raise ValueError(f"Eval manifest entry {key} has empty target_frame_indices")
            if int(target[0]) != int(item["source_frame_idx"]):
                raise ValueError(f"Eval manifest entry {key} violates target[0] == source")

    def reset(self) -> None:
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.eval_manifest)

    def next_batch(self) -> Dict[str, Any]:
        if self.cursor >= len(self.eval_manifest):
            raise StopIteration
        item = self.eval_manifest[self.cursor]
        self.cursor += 1
        return self.dataset.get_segment_batch_from_frames(
            scene_id=int(item["scene_id"]),
            segment_id=int(item["segment_id"]),
            source_frame_idx=int(item["source_frame_idx"]),
            target_frame_indices=[int(x) for x in item["target_frame_indices"]],
            include_test=self.include_test,
            test_frame_indices=[int(x) for x in item.get("test_frame_indices", [])] if self.include_test else None,
        )

