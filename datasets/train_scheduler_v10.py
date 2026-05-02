from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets.train_scheduler_v9 import TrainSchedulerV9


class TrainSchedulerV10(TrainSchedulerV9):
    """
    Stage6_0 scheduler.

    Implementation strategy:
    - Reuse V9 episode/block traversal stability.
    - Remap role names and emit structured request metadata.
    - Keep aligned/compat fields to avoid breaking existing trainer loops.
    """

    @staticmethod
    def _map_role_v9_to_v10(role: str) -> str:
        r = str(role)
        if r == "teacher_preserve":
            return "teacher_anchor"
        if r == "visited":
            return "history_visited"
        if r == "near_random":
            return "probe_near"
        return r

    def _patch_request_meta_to_v10(self, batch: Dict[str, Any]) -> None:
        meta = dict(batch.get("request_meta") or {})
        frame_roles = [self._map_role_v9_to_v10(x) for x in list(meta.get("target_frame_roles") or [])]
        image_roles = [self._map_role_v9_to_v10(x) for x in list(meta.get("target_image_roles") or [])]
        target_frames = [int(x) for x in list(meta.get("target_frame_indices") or [])]
        target_frame_weights = [float(x) for x in list(meta.get("target_frame_loss_base_weights") or [])]
        target_refs = [tuple(x) for x in list(meta.get("target_image_refs") or [])]
        target_image_weights = [float(x) for x in list(meta.get("target_image_loss_base_weights") or [])]
        if len(target_refs) != len(image_roles):
            raise ValueError(f"target_image_refs/target_image_roles mismatch: {len(target_refs)} vs {len(image_roles)}")
        if len(target_refs) != len(target_image_weights):
            raise ValueError(
                f"target_image_refs/target_image_loss_base_weights mismatch: {len(target_refs)} vs {len(target_image_weights)}"
            )
        if len(target_frames) != len(frame_roles):
            raise ValueError(f"target_frame_indices/target_frame_roles mismatch: {len(target_frames)} vs {len(frame_roles)}")
        if len(target_frames) != len(target_frame_weights):
            raise ValueError(
                f"target_frame_indices/target_frame_loss_base_weights mismatch: {len(target_frames)} vs {len(target_frame_weights)}"
            )

        train_roles = {"teacher_source", "student_source", "teacher_anchor", "history_visited"}
        probe_roles = {"probe_near"}
        train_frame_triplets = [
            (int(f), str(r), float(w))
            for f, r, w in zip(target_frames, frame_roles, target_frame_weights)
            if str(r) in train_roles
        ]
        probe_frame_triplets = [
            (int(f), str(r), float(w))
            for f, r, w in zip(target_frames, frame_roles, target_frame_weights)
            if str(r) in probe_roles
        ]
        train_image_triplets = [
            (tuple(ref), str(role), float(weight))
            for ref, role, weight in zip(target_refs, image_roles, target_image_weights)
            if str(role) in train_roles
        ]
        probe_image_triplets = [
            (tuple(ref), str(role), float(weight))
            for ref, role, weight in zip(target_refs, image_roles, target_image_weights)
            if str(role) in probe_roles
        ]
        meta["target_frame_roles"] = frame_roles
        meta["target_image_roles"] = image_roles
        meta["scheduler_version"] = "v10"
        meta["stage6_role"] = str(meta.get("stage5_5_role", "teacher"))
        meta["history_record/observed_record_trigger"] = "teacher_exit" if str(meta["stage6_role"]) == "teacher" else "none"

        source_frame = int(meta.get("stage5_5_source_frame_idx", -1))
        teacher_frame = int(meta.get("stage5_5_teacher_frame_idx", -1))
        near_refs = [
            tuple(x) for x, role in zip(target_refs, image_roles) if str(role) == "probe_near"
        ]
        teacher_refs = [tuple(x) for x in list(meta.get("stage5_5_teacher_image_refs") or [])]
        if len(teacher_refs) == 0:
            teacher_refs = [tuple(x) for x, role in zip(target_refs, image_roles) if str(role) == "teacher_anchor"]
        live_bridge_enable = bool(str(meta["stage6_role"]) == "student")

        meta["scheduler_request_v10"] = {
            "scheduler_version": "v10",
            "stage": "6_0",
            "teacher_obs": {
                "enable": bool(str(meta["stage6_role"]) == "teacher"),
                "frame_idx": int(teacher_frame),
                "image_refs": teacher_refs,
                "record_observed": bool(str(meta["stage6_role"]) == "teacher"),
                "update_cache": bool(str(meta["stage6_role"]) == "teacher"),
            },
            "student_prop": {
                "enable": bool(str(meta["stage6_role"]) == "student"),
                "frame_idx": int(source_frame),
                "requires_teacher_anchor": True,
                "requires_live_bridge": True,
            },
            "live_teacher_bridge": {
                "enable": bool(live_bridge_enable),
                "frame_idx": int(teacher_frame),
                "image_refs": teacher_refs,
                "record_observed": False,
                "update_cache": False,
                "rerun_teacher_2d": bool(live_bridge_enable),
            },
            "teacher_anchor": {
                "enable": bool("teacher_anchor" in frame_roles),
                "frame_idx": int(teacher_frame),
                "image_refs": [tuple(x) for x, role in zip(target_refs, image_roles) if str(role) == "teacher_anchor"],
                "role": "teacher_anchor",
            },
            "history_targets": {
                "enable": bool("history_visited" in frame_roles),
                "frame_indices": [int(x) for x, r in zip(target_frames, frame_roles) if str(r) == "history_visited"],
                "role": "history_visited",
            },
            "probe_targets": {
                "enable": bool(len(near_refs) > 0),
                "image_refs": near_refs,
                "role": "probe_near",
                "log_only": True,
            },
            "train_targets": {
                "frame_indices": [int(x[0]) for x in train_frame_triplets],
                "frame_roles": [str(x[1]) for x in train_frame_triplets],
                "frame_loss_base_weights": [float(x[2]) for x in train_frame_triplets],
                "image_refs": [tuple(x[0]) for x in train_image_triplets],
                "image_roles": [str(x[1]) for x in train_image_triplets],
                "image_loss_base_weights": [float(x[2]) for x in train_image_triplets],
            },
        }
        meta["train_target_frame_indices"] = [int(x[0]) for x in train_frame_triplets]
        meta["train_target_frame_roles"] = [str(x[1]) for x in train_frame_triplets]
        meta["train_target_frame_loss_base_weights"] = [float(x[2]) for x in train_frame_triplets]
        meta["train_target_image_refs"] = [tuple(x[0]) for x in train_image_triplets]
        meta["train_target_image_roles"] = [str(x[1]) for x in train_image_triplets]
        meta["train_target_image_loss_base_weights"] = [float(x[2]) for x in train_image_triplets]
        meta["probe_target_frame_indices"] = [int(x[0]) for x in probe_frame_triplets]
        meta["probe_target_frame_roles"] = [str(x[1]) for x in probe_frame_triplets]
        meta["probe_target_image_refs"] = [tuple(x[0]) for x in probe_image_triplets]
        meta["probe_target_image_roles"] = [str(x[1]) for x in probe_image_triplets]

        meta["scheduler_v10/target_has_teacher_anchor"] = float(1.0 if "teacher_anchor" in frame_roles else 0.0)
        meta["scheduler_v10/target_num_history_visited"] = float(sum(1 for x in frame_roles if str(x) == "history_visited"))
        meta["scheduler_v10/target_num_probe_near"] = float(sum(1 for x in frame_roles if str(x) == "probe_near"))
        meta["scheduler_v10/train_target_num_frames"] = float(len(train_frame_triplets))
        meta["scheduler_v10/probe_target_num_frames"] = float(len(probe_frame_triplets))
        meta["scheduler_v10/live_teacher_bridge_enable"] = float(1.0 if live_bridge_enable else 0.0)
        meta["scheduler/v10_is_compat_v9"] = 1.0

        batch["request_meta"] = meta

    def _patch_aligned_to_v10(self, batch: Dict[str, Any]) -> None:
        aligned = dict(
            batch.get("_scheduler_v9_aligned_info")
            or batch.get("_scheduler_v8_aligned_info")
            or batch.get("_scheduler_v7_aligned_info")
            or batch.get("_scheduler_v4_aligned_info")
            or {}
        )
        aligned["scheduler_version"] = "v10"
        batch["_scheduler_v4_aligned_info"] = dict(aligned)
        batch["_scheduler_v7_aligned_info"] = dict(aligned)
        batch["_scheduler_v8_aligned_info"] = dict(aligned)
        batch["_scheduler_v9_aligned_info"] = dict(aligned)
        batch["_scheduler_v10_aligned_info"] = dict(aligned)

    def materialize_current_batch_without_advance(self) -> Dict[str, Any]:
        batch = super().materialize_current_batch_without_advance()
        self._patch_request_meta_to_v10(batch)
        self._patch_aligned_to_v10(batch)
        return batch

    def next_batch(self) -> Dict[str, Any]:
        batch = super().next_batch()
        self._patch_request_meta_to_v10(batch)
        self._patch_aligned_to_v10(batch)
        return batch

    def get_current_info(self) -> Dict[str, Any]:
        out = dict(super().get_current_info())
        out["scheduler_version"] = "v10"
        return out
