from __future__ import annotations

import dataclasses
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from datasets.iforward_coverage_validation import (
    build_coverage_validation_scheduler,
    write_iforward_coverage_validation_rows,
)


class _FakeCoverageDataset:
    _initialized = True

    def __init__(self, *, num_keyframes: int = 10, num_cams: int = 3):
        self.num_keyframes = int(num_keyframes)
        self.num_cams = int(num_cams)
        self.keyframe_to_frames = {
            int(k): [int(k * 10), int(k * 10 + 1)]
            for k in range(int(num_keyframes))
        }
        self.frames = [int(f) for frames in self.keyframe_to_frames.values() for f in frames]

    def initialize(self):
        return None

    def list_segment_ids(self, scene_id):
        assert int(scene_id) == 1
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        frame_to_keyframe = {}
        for keyframe_idx, frames in self.keyframe_to_frames.items():
            for frame_idx in frames:
                frame_to_keyframe[int(frame_idx)] = int(keyframe_idx)
        return SimpleNamespace(
            scene_id=1,
            segment_id=0,
            num_cams=int(self.num_cams),
            frame_indices=list(self.frames),
            train_frame_set=set(self.frames),
            test_frame_set=set(),
            keyframe_indices=list(range(int(self.num_keyframes))),
            keyframe_to_frames={int(k): [int(x) for x in v] for k, v in self.keyframe_to_frames.items()},
            frame_to_keyframe=frame_to_keyframe,
            train_image_refs=tuple((int(f), int(c)) for f in self.frames for c in range(int(self.num_cams))),
            test_image_refs=tuple(),
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        assert int(scene_id) == 1
        assert int(segment_id) == 0
        assert str(purpose) == "train"
        frame_idx, cam_idx = int(ref[0]), int(ref[1])
        assert frame_idx in set(self.frames)
        assert 0 <= cam_idx < int(self.num_cams)

    def _assemble_segment_batch_from_iforward_request(self, *, scene_id, segment_id, plan, include_test=False):
        _ = include_test
        target_refs = list(plan.target_refs_flat)
        return {
            "scene_id": int(scene_id),
            "segment_id": int(segment_id),
            "request_meta": dict(plan.request_meta),
            "_iforward": dataclasses.asdict(plan),
            "source": {"refs": list(plan.evidence_refs_flat)},
            "target": {
                "refs": list(target_refs),
                "roles": list(plan.target_roles_flat),
                "image": torch.zeros(len(target_refs), 1, 1, 3),
            },
            "targets": [
                {"frame_idx": int(f), "cam_idx": int(c), "gt_image": torch.zeros(1, 1, 3)}
                for f, c in target_refs
            ],
        }


class _FakeState:
    def detach_for_next_rollout(self):
        return self


class _FakeCoverageOutput:
    def __init__(self, *, ifwd):
        self.loss = torch.tensor(1.0)
        self.losses = {
            "current": torch.tensor(1.0),
            "current_latest": torch.tensor(1.0),
            "in_rollout_history": torch.tensor(2.0),
            "history": torch.tensor(2.0),
            "nearby": torch.tensor(3.0),
        }
        self.stats = {
            "current_ssim": 0.9,
            "history_rollout_ssim": 0.8,
            "nearby_ssim": 0.7,
            "eval_recon_all_blocks_ssim": 0.95,
            "eval_nearby_nvs_all_blocks_ssim": 0.75,
        }
        self.next_state = _FakeState()
        self.resolved = SimpleNamespace(
            carry_scene_state_after_rollout=bool(ifwd["carry_scene_state_after_rollout"]),
            episode_end_after_rollout=bool(ifwd["episode_end_after_rollout"]),
        )
        role_map = {
            "final_current_recon": "current_latest",
            "final_history_replay": "history_rollout",
            "final_nearby_rollout": "nearby",
            "eval_recon_all_blocks": "eval_recon_all_blocks",
            "eval_nearby_nvs_all_blocks": "eval_nearby_nvs_all_blocks",
        }
        self.pred_rgbs = []
        self.gt_images = []
        self.image_refs = []
        self.image_roles = []
        for ref, role in zip(ifwd["target_refs_flat"], ifwd["target_roles_flat"]):
            self.image_refs.append((int(ref[0]), int(ref[1])))
            self.image_roles.append(role_map.get(str(role), str(role)))
            self.pred_rgbs.append(torch.zeros(1, 1, 3))
            self.gt_images.append(torch.zeros(1, 1, 3))


class _FakeCoverageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.carried_flags = []
        self.loss_keys_seen = []
        self._state_cache = {}
        self.bridge_reset_count = 0

    def reset_iforward_state_cache(self):
        self._state_cache.clear()

    def _reset_bridge_runtime_node_state(self):
        self.bridge_reset_count += 1

    def forward_rollout(self, batch, *, carried_state=None, ablation=None):
        assert str(ablation) == "full"
        self.carried_flags.append(carried_state is not None)
        self._state_cache[("validation", int(batch["_iforward"]["rollout_id_global"]))] = _FakeState()
        out = _FakeCoverageOutput(ifwd=batch["_iforward"])
        self.loss_keys_seen.append(set(out.losses.keys()))
        return out


def _cfg():
    return {
        "data": {"eval_scene_ids": [1]},
        "iforward_coverage_validation": {
            "enable": True,
            "run_at_train_start": True,
            "interval_steps": 1000,
            "segments_per_scene": 1,
            "max_segments_total": 1,
            "seed": 20260614,
            "episode": {
                "blocks_per_episode": 10,
                "episode_stride": 10,
                "allow_short_last_episode": False,
                "min_blocks_per_episode": 4,
            },
            "rollout": {
                "tail_policy": "circular_fill",
                "start_offset": 0,
                "max_inner_K": 8,
            },
            "shapes": [
                {"name": "r8b1", "blocks_per_rollout": 1, "repeats_per_block": 8},
                {"name": "r4b2", "blocks_per_rollout": 2, "repeats_per_block": 4},
                {"name": "r2b4", "blocks_per_rollout": 4, "repeats_per_block": 2},
            ],
            "target_repeats_per_block": [8, 16, 32],
            "supervision": {
                "current": {"enable": True, "role_name": "final_current_recon", "camera_policy": "all_cams"},
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
                    "add_to_evidence": False,
                },
            },
            "final_eval": {
                "enable": True,
                "attach_to_last_rollout": True,
                "recon_all_blocks": True,
                "nearby_nvs_all_blocks": True,
                "roles_zero_loss": True,
            },
            "tensorboard_images": {"enable": False},
        },
    }


def test_coverage_validation_scheduler_plan_is_deterministic():
    ds = _FakeCoverageDataset(num_keyframes=10, num_cams=2)
    shape = {"name": "r2b4", "blocks_per_rollout": 4, "repeats_per_block": 2}
    sched_a = build_coverage_validation_scheduler(
        cfg=_cfg(),
        dataset=ds,
        scene_id=1,
        segment_id=0,
        shape=shape,
        target_repeats_per_block=8,
    )
    sched_b = build_coverage_validation_scheduler(
        cfg=_cfg(),
        dataset=ds,
        scene_id=1,
        segment_id=0,
        shape=shape,
        target_repeats_per_block=8,
    )
    plans_a = [sched_a.next_batch()["_iforward"] for _ in range(10)]
    plans_b = [sched_b.next_batch()["_iforward"] for _ in range(10)]
    assert [p["window_block_ids"] for p in plans_a] == [p["window_block_ids"] for p in plans_b]
    assert plans_a[-1]["request_meta"]["block_frame_map"] == plans_b[-1]["request_meta"]["block_frame_map"]
    assert plans_a[-1]["request_meta"]["final_eval_recon_frames"] == plans_b[-1]["request_meta"]["final_eval_recon_frames"]
    assert (
        plans_a[-1]["request_meta"]["final_eval_nearby_nvs_frame_map"]
        == plans_b[-1]["request_meta"]["final_eval_nearby_nvs_frame_map"]
    )


def test_coverage_validation_enumerates_shapes_r_and_writes_final_global_rows():
    rows = []
    model = _FakeCoverageModel()
    write_iforward_coverage_validation_rows(
        cfg=_cfg(),
        dataset=_FakeCoverageDataset(num_keyframes=10, num_cams=2),
        model=model,
        device=torch.device("cpu"),
        trigger_step=100,
        trigger_train_episode_counter=5,
        metrics_fh=rows,
        writer=None,
        convert_batch_to_minimal_format=lambda raw, *args, **kwargs: raw,
        write_metrics_history=lambda fh, row: fh.append(dict(row)),
    )
    final_rows = [row for row in rows if row["split"] == "iforward_coverage_validation_final"]
    rollout_rows = [row for row in rows if row["split"] == "iforward_coverage_validation_rollout"]
    global_rows = [row for row in rows if row["split"] == "iforward_coverage_validation_global"]
    assert len(final_rows) == 9
    assert len(rollout_rows) == 210
    assert len(global_rows) == 1
    assert {
        (row["shape_name"], row["target_repeats_per_block"])
        for row in final_rows
    } == {
        ("r8b1", 8),
        ("r8b1", 16),
        ("r8b1", 32),
        ("r4b2", 8),
        ("r4b2", 16),
        ("r4b2", 32),
        ("r2b4", 8),
        ("r2b4", 16),
        ("r2b4", 32),
    }
    sample = final_rows[0]
    for key in (
        "final_recon_psnr_mean",
        "final_nearby_nvs_psnr_mean",
        "forget_last_to_final_drop_p90",
        "forget_best_to_final_drop_max",
        "coverage_exact",
        "coverage_exact_target",
        "coverage_exact_achieved",
        "hgv2_validation_mode",
    ):
        assert key in sample
    assert "final_recon_lpips_mean" not in sample
    assert "final_nearby_nvs_lpips_mean" not in sample
    assert all("current_lpips" not in row for row in rollout_rows)
    assert all("history_lpips" not in row for row in rollout_rows)
    assert all("nearby_lpips" not in row for row in rollout_rows)
    global_row = global_rows[0]
    assert "r8b1_R8_recon_psnr_mean" in global_row
    assert "r4b2_R16_nearby_nvs_psnr_mean" in global_row
    assert "r2b4_R32_forget_p90" in global_row
    assert all("eval_recon_all_blocks" not in keys for keys in model.loss_keys_seen)
    assert all("eval_nearby_nvs_all_blocks" not in keys for keys in model.loss_keys_seen)
    assert any(model.carried_flags)


def test_coverage_validation_preserves_training_state_cache_object_and_contents():
    cfg = copy.deepcopy(_cfg())
    cfg["iforward_coverage_validation"]["shapes"] = [
        {"name": "r8b1", "blocks_per_rollout": 1, "repeats_per_block": 8}
    ]
    cfg["iforward_coverage_validation"]["target_repeats_per_block"] = [8]
    rows = []
    model = _FakeCoverageModel()
    sentinel_state = _FakeState()
    model._state_cache = {("training", 0): sentinel_state}
    before_id = id(model._state_cache)

    write_iforward_coverage_validation_rows(
        cfg=cfg,
        dataset=_FakeCoverageDataset(num_keyframes=10, num_cams=2),
        model=model,
        device=torch.device("cpu"),
        trigger_step=100,
        trigger_train_episode_counter=5,
        metrics_fh=rows,
        writer=None,
        convert_batch_to_minimal_format=lambda raw, *args, **kwargs: raw,
        write_metrics_history=lambda fh, row: fh.append(dict(row)),
    )

    assert id(model._state_cache) == before_id
    assert model._state_cache == {("training", 0): sentinel_state}
    assert all(key[0] != "validation" for key in model._state_cache)
    assert model.bridge_reset_count > 0
