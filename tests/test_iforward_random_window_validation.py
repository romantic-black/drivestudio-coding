from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from datasets.iforward_random_window_validation import (
    fixed_random_window_starts,
    write_random_window_validation_rows,
)
from tools import train_iforward as train_ifwd


class _FakeValidationDataset:
    _initialized = True

    def __init__(self, *, num_keyframes: int = 4, num_cams: int = 3):
        self.num_keyframes = int(num_keyframes)
        self.num_cams = int(num_cams)

    def initialize(self):
        return None

    def list_segment_ids(self, scene_id):
        assert int(scene_id) == 1
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        keyframes = list(range(int(self.num_keyframes)))
        frames = [int(k * 10) for k in keyframes]
        return SimpleNamespace(
            scene_id=1,
            segment_id=0,
            num_cams=int(self.num_cams),
            frame_indices=list(frames),
            train_frame_set=set(frames),
            test_frame_set=set(),
            keyframe_indices=list(keyframes),
            keyframe_to_frames={int(k): [int(k * 10)] for k in keyframes},
            frame_to_keyframe={int(k * 10): int(k) for k in keyframes},
            train_image_refs=tuple((int(f), int(c)) for f in frames for c in range(int(self.num_cams))),
            test_image_refs=tuple(),
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        sidx = self.get_segment_index(int(scene_id), int(segment_id))
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert str(purpose) == "train"
        assert frame_idx in set(sidx.frame_indices)
        assert 0 <= cam_idx < int(sidx.num_cams)

    def _assemble_segment_batch_from_iforward_random_window_request(self, *, scene_id, segment_id, plan, include_test=False):
        _ = include_test
        target_refs = list(plan.target_refs_flat)
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "target": {"image": torch.zeros(len(target_refs), 1, 1, 3)},
            "targets": [{"frame_idx": int(f), "cam_idx": int(c), "gt_image": torch.zeros(1, 1, 3)} for f, c in target_refs],
        }

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        _ = include_test
        target_refs = list(plan.target_refs_flat)
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "target": {"image": torch.zeros(len(target_refs), 1, 1, 3)},
            "targets": [{"frame_idx": int(f), "cam_idx": int(c), "gt_image": torch.zeros(1, 1, 3)} for f, c in target_refs],
        }


class _FakeCarriedState:
    def __init__(self, rollout_idx: int):
        self.rollout_idx = int(rollout_idx)

    def detach_for_next_rollout(self):
        return self


class _FakeValidationOutput:
    def __init__(self, *, rollout_idx: int, ifwd):
        value = float(10 + int(rollout_idx))
        final_supervision = dict(ifwd.get("final_supervision", {}) or {})
        history_count = int(final_supervision.get("history_ref_count", len(ifwd.get("in_rollout_history_refs", []) or [])))
        history_psnr = value + 1.0 if history_count > 0 else float("nan")
        history_valid = 1.0 if history_count > 0 else 0.0
        self.loss = torch.tensor(1.0)
        self.losses = {
            "current": torch.tensor(1.0),
            "current_latest": torch.tensor(1.0),
            "in_rollout_history": torch.tensor(2.0),
            "short_window_history": torch.tensor(3.0),
            "nearby": torch.tensor(4.0),
        }
        self.stats = {
            "window_start": int(ifwd["window_start"]),
            "window_hash": int(ifwd["window_hash"]),
            "is_repeated_window": bool(ifwd["is_repeated_window"]),
            "current_psnr": value,
            "current_latest_psnr": value,
            "history_rollout_psnr": history_psnr,
            "in_rollout_history_psnr": history_psnr,
            "short_window_history_psnr": value + 2.0,
            "nearby_psnr": value + 3.0,
            "current_valid_ratio": 1.0,
            "current_num_refs": float(final_supervision.get("current_ref_count", 0)) if final_supervision else 0.0,
            "in_rollout_history_valid_ratio": history_valid,
            "history_rollout_num_refs": float(history_count),
            "nearby_valid_ratio": 1.0,
            "nearby_num_refs": float(final_supervision.get("nearby_ref_count", 0)) if final_supervision else 0.0,
        }
        self.next_state = _FakeCarriedState(int(rollout_idx))
        self.resolved = SimpleNamespace(
            carry_scene_state_after_rollout=bool(ifwd["carry_scene_state_after_rollout"]),
            episode_end_after_rollout=bool(ifwd.get("episode_end_after_rollout", False)),
        )
        self.pred_rgbs = []
        self.gt_images = []
        self.image_refs = []
        self.image_roles = []


class _FakeValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.carried_flags = []
        self.modes = []
        self.window_hashes_by_mode = {}

    def forward_rollout(self, batch, *, carried_state=None, ablation=None):
        mode = str(ablation or "full")
        ifwd = batch["_iforward"]
        rollout_idx = int(ifwd["rollout_idx_in_episode"])
        self.carried_flags.append(carried_state is not None)
        self.modes.append(mode)
        self.window_hashes_by_mode.setdefault(mode, []).append(int(ifwd["window_hash"]))
        return _FakeValidationOutput(rollout_idx=rollout_idx, ifwd=ifwd)


def _cfg():
    return {
        "data": {"eval_scene_ids": [1]},
        "scheduler_iforward_random_window": {
            "traversal": {"seed": 41},
            "segment": {"source_mode": "keyframes", "min_blocks": 4},
            "episode": {"rollouts_per_episode": 8},
            "rollout": {
                "blocks_per_rollout": 4,
                "repeats_per_block": 2,
                "window_policy": "random_with_replacement",
                "delivery_order": "chronological",
                "detach_graph_after_rollout": True,
            },
            "evidence": {"camera_policy": "all_cams", "mask_policy": "non_sky_non_egocar"},
            "supervision": {
                "current_latest": {"enable": True, "camera_policy": "all_cams"},
                "in_rollout_history": {"enable": True, "camera_policy": "all_cams"},
                "short_window_history": {"enable": True, "max_entries": 24},
                "nearby": {"enable": True, "frames_per_rollout": 1, "camera_policy": "all_cams", "max_refs_per_rollout": 3},
            },
            "memory": {
                "observation_commit_policy": "first_repeat_only",
                "optimizer_memory_update_policy": "every_repeat",
            },
        },
        "iforward_random_window_validation": {
            "enable": True,
            "segments_per_scene": 1,
            "rollouts_per_segment": 8,
            "seed": 20260604,
            "carry_state_across_rollouts": True,
            "reset_state_at_segment_begin": True,
            "tensorboard_images": {"enable": False, "rollout_indices": [0, 1, 3, 7], "max_images_per_role": 2},
        },
    }


def _cfg_v3():
    return {
        "data": {"eval_scene_ids": [1]},
        "scheduler_iforward": {
            "version": "iforward_v3_random_window",
            "traversal": {"seed": 41},
            "episode": {"rollouts_per_episode": 8},
            "rollout": {
                "window_policy": "random_with_replacement",
                "delivery_order": "chronological_inside_window",
                "detach_graph_after_rollout": True,
            },
            "evidence": {"camera_policy": "all_cams", "mask_policy": "non_sky_non_egocar"},
            "supervision": {
                "current": {
                    "enable": True,
                    "role_name": "final_current_recon",
                    "frame_policy": "all_rollout_input_frames",
                    "camera_policy": "all_cams",
                },
                "history_replay": {
                    "enable": True,
                    "role_name": "final_history_replay",
                    "camera_policy": "all_cams",
                    "max_frames_per_rollout": 8,
                },
                "nearby": {
                    "enable": True,
                    "role_name": "final_nearby_rollout",
                    "frames_per_rollout": 1,
                    "camera_policy": "all_cams",
                    "max_refs_per_rollout": 3,
                },
            },
            "memory": {
                "observation_commit_policy": "first_repeat_only",
                "optimizer_memory_update_policy": "every_repeat",
            },
            "loss_timing": {"policy": "rollout_final_only", "intermediate_step_loss": False},
            "leakage_check": {"enable": True, "forbid_test_refs_in_train": True},
        },
        "iforward_validation": {
            "enable": True,
            "segments_per_scene": 1,
            "rollouts_per_segment": 3,
            "tensorboard_images": {"enable": False, "max_images_per_role": 2},
        },
    }


def _cfg_v1_validation_shape():
    return {
        "data": {"eval_scene_ids": [1]},
        "scheduler_iforward": {
            "version": "iforward_v1",
            "traversal": {
                "scene_order": "ascending",
                "segment_order": "ascending",
                "traversal_mode": "scene_round_robin_episode",
                "seed": 41,
            },
            "episode": {
                "source_mode": "keyframes",
                "blocks_per_episode": 8,
                "episode_stride": 8,
                "allow_short_last_episode": False,
                "min_blocks_per_episode": 4,
                "rollouts_per_episode": 1,
                "block_source_frame_policy": "random_within_keyframe_per_rollout",
            },
            "rollout": {
                "block_selection_policy": "random_start_contiguous",
                "delivery_order_policy": "chronological",
                "allow_short_final_rollout": False,
                "min_blocks_per_rollout": 1,
                "avoid_single_block_tail": False,
                "detach_graph_after_rollout": True,
                "shapes": [
                    {"name": "b1_r8", "blocks_per_rollout": 1, "repeats_per_block": 8, "prob": 1.0},
                ],
            },
            "evidence": {"camera_policy": "all_cams", "mask_policy": "non_sky_non_egocar"},
            "supervision": {
                "current": {
                    "enable": True,
                    "role_name": "final_current_recon",
                    "frame_policy": "all_input_frames",
                    "camera_policy": "all_cams",
                },
                "nearby": {"enable": False},
                "history_replay": {"enable": False},
            },
            "memory": {
                "observation_commit_policy": "first_repeat_only",
                "optimizer_memory_update_policy": "every_repeat",
                "reset_policy": "episode_begin",
                "carry_policy": "across_rollouts_until_episode_end",
            },
            "loss_timing": {"policy": "rollout_final_only", "intermediate_step_loss": False},
            "leakage_check": {"enable": True, "forbid_test_refs_in_train": True},
        },
        "iforward_validation": {
            "enable": True,
            "segments_per_scene": 1,
            "rollouts_per_segment": 1,
            "rollout_shapes": [
                {"name": "b1_r4", "blocks_per_rollout": 1, "repeats_per_block": 4, "prob": 1.0},
            ],
            "tensorboard_images": {"enable": False, "max_images_per_role": 2},
        },
    }


def test_fixed_random_window_starts_are_deterministic_and_allow_repeats():
    a = fixed_random_window_starts(num_blocks=4, rollouts=8, seed=20260604, scene_id=1, segment_id=0)
    b = fixed_random_window_starts(num_blocks=4, rollouts=8, seed=20260604, scene_id=1, segment_id=0)
    assert a == b
    assert a == [0] * 8
    assert fixed_random_window_starts(num_blocks=3, rollouts=8, seed=1, scene_id=1, segment_id=0) == []


def test_random_window_validation_carries_state_and_writes_revisit_aggregates():
    rows = []
    model = _FakeValidationModel()
    write_random_window_validation_rows(
        cfg=_cfg(),
        dataset=_FakeValidationDataset(num_keyframes=4),
        model=model,
        device=torch.device("cpu"),
        trigger_step=100,
        trigger_train_episode_counter=5,
        metrics_fh=rows,
        writer=None,
        convert_batch_to_minimal_format=lambda raw, *args, **kwargs: raw,
        write_metrics_history=lambda fh, row: fh.append(dict(row)),
    )

    rollout_rows = [row for row in rows if row["split"] == "iforward_random_window_validation"]
    global_rows = [row for row in rows if row["split"] == "iforward_random_window_validation_global"]
    assert len(rollout_rows) == 8
    assert len(global_rows) == 1
    assert [row["rollout_idx"] for row in rollout_rows] == list(range(8))
    assert [row["window_start"] for row in rollout_rows] == [0] * 8
    assert model.carried_flags == [False] + [True] * 7
    assert rollout_rows[0]["revisit_current_psnr_delta"] != rollout_rows[0]["revisit_current_psnr_delta"]
    assert rollout_rows[1]["revisit_current_psnr_delta"] == pytest.approx(1.0)
    assert global_rows[0]["revisit_current_psnr_delta_mean"] == pytest.approx(1.0)
    assert global_rows[0]["final_rollout_current_latest_psnr"] == pytest.approx(17.0)


def test_iforward_v3_validation_scheduler_uses_fixed_v3_history_shapes():
    ds = _FakeValidationDataset(num_keyframes=8, num_cams=3)
    scheduler = train_ifwd._make_validation_scheduler(_cfg_v3(), ds, 1, 0)

    batches = [scheduler.next_batch()["_iforward"] for _ in range(3)]

    assert [batch["shape_name"] for batch in batches] == ["r8b1", "r4b2", "r2b4"]
    assert [batch["window_start"] for batch in batches] == [0, 2, 4]
    assert [batch["final_supervision"]["history_ref_count"] for batch in batches] == [0, 3, 9]
    for batch in batches[1:]:
        history_refs = {tuple(ref) for ref in batch["final_supervision"]["history_refs"]}
        history_frames = {int(ref[0]) for ref in history_refs}
        assert len(history_refs) == len(history_frames) * ds.num_cams
        for frame_idx in history_frames:
            assert {(frame_idx, cam_idx) for cam_idx in range(ds.num_cams)} <= history_refs
        assert history_refs.isdisjoint({tuple(ref) for ref in batch["final_supervision"]["current_refs"]})


def test_iforward_validation_scheduler_uses_configured_v1_rollout_shape():
    scheduler = train_ifwd._make_validation_scheduler(
        _cfg_v1_validation_shape(),
        _FakeValidationDataset(num_keyframes=4, num_cams=3),
        1,
        0,
    )

    batch = scheduler.next_batch()["_iforward"]

    assert batch["shape_name"] == "b1_r4"
    assert batch["blocks_per_rollout"] == 1
    assert batch["repeats_per_block"] == 4


def test_iforward_v3_validation_carries_state_and_reports_history(monkeypatch):
    rows = []
    model = _FakeValidationModel()
    monkeypatch.setattr(train_ifwd.base, "convert_batch_to_minimal_format", lambda raw, *args, **kwargs: raw)
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    train_ifwd._write_iforward_validation_rows(
        cfg=_cfg_v3(),
        dataset=_FakeValidationDataset(num_keyframes=8, num_cams=3),
        model=model,
        device=torch.device("cpu"),
        trigger_step=100,
        trigger_train_episode_counter=5,
        metrics_fh=rows,
        writer=None,
    )

    rollout_rows = [row for row in rows if row["split"] == "iforward_validation"]
    global_rows = [row for row in rows if row["split"] == "iforward_validation_global"]
    assert len(rollout_rows) == 6
    assert len(global_rows) == 3
    full_rows = [row for row in rollout_rows if row["mode"] == "full_adc"]
    no_adc_rows = [row for row in rollout_rows if row["mode"] == "no_adc"]
    assert [row["rollout_shape"] for row in full_rows] == ["r8b1", "r4b2", "r2b4"]
    assert [row["rollout_shape"] for row in no_adc_rows] == ["r8b1", "r4b2", "r2b4"]
    assert model.carried_flags == [False, True, True, False, True, True]
    assert model.modes == ["full_adc", "full_adc", "full_adc", "no_adc", "no_adc", "no_adc"]
    assert model.window_hashes_by_mode["full_adc"] == model.window_hashes_by_mode["no_adc"]
    assert [row["history_rollout_num_refs"] for row in full_rows] == [0.0, 3.0, 9.0]
    assert full_rows[1]["history_rollout_psnr"] == pytest.approx(12.0)
    assert full_rows[1]["delta_full_minus_noadc_current_psnr"] == pytest.approx(0.0)
    global_by_mode = {row["mode"]: row for row in global_rows}
    assert global_by_mode["full_adc"]["history_rollout_psnr"] == pytest.approx(12.5)
    assert global_by_mode["no_adc"]["history_rollout_psnr"] == pytest.approx(12.5)
    assert global_by_mode["full_minus_no_adc"]["current_psnr"] == pytest.approx(0.0)
