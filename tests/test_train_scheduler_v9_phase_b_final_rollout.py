from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from datasets.train_scheduler_v9 import TrainSchedulerV9
from models.streetforward.stage6_0.v9_role_resolver import resolve_v9_phase_b_batch


class _FakeDataset:
    _initialized = True

    def list_training_scene_ids(self):
        return [1]

    def list_segment_ids(self, scene_id):
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        frames = list(range(40))
        return SimpleNamespace(
            keyframe_indices=frames,
            frame_indices=frames,
            keyframe_to_frames={int(i): [int(i)] for i in frames},
            frame_to_keyframe={int(i): int(i) for i in frames},
            train_frame_set=set(frames),
            num_cams=3,
        )

    def validate_image_ref(self, scene_id, segment_id, ref, purpose="train"):
        return None

    def _assemble_segment_batch_from_v9_request(self, *, scene_id, segment_id, v9_plan, include_test=False):
        meta = dict(v9_plan.request_meta or {})
        source_refs = [tuple(x) for x in meta["source_image_refs"]]
        target_refs = [tuple(x) for x in meta["target_image_refs"]]
        return {
            "request_meta": meta,
            "source_views": [{"frame_idx": int(f), "cam_idx": int(c)} for f, c in source_refs],
            "targets": [{"frame_idx": int(f), "cam_idx": int(c)} for f, c in target_refs],
        }


def _scheduler(
    *,
    blocks: int = 4,
    repeats: int = 4,
    rollouts: int = 4,
    blocks_per_episode: int | None = None,
    current_max_frames=None,
    current_allow_subsample: bool = False,
) -> TrainSchedulerV9:
    return TrainSchedulerV9(
        dataset=_FakeDataset(),
        phase="phase_B_viewset_rollout",
        steps_per_block=1,
        blocks_per_episode=(
            int(blocks_per_episode)
            if blocks_per_episode is not None
            else int(rollouts) * max(int(blocks), 6)
        ),
        include_source_frame=True,
        frame_within_keyframe_policy="random_once_per_episode",
        min_keyframes_required_policy="use_available_if_less_than_window",
        traversal_mode="round_robin_episode_interleave",
        switch_after_episode=True,
        segment_order="ascending",
        scene_order="ascending",
        include_test=False,
        fixed_scene_id=None,
        fixed_segment_id=None,
        emit_preload_hints=False,
        warm_next_block_exact=False,
        warm_next_episode_chain=False,
        block_order="step_major",
        step_major_switch_interval_steps=1,
        target_policy="visited_episode_frames",
        reset_policy="episode_end",
        block_source_frame_policy="random_within_keyframe_per_visit",
        episode_source_mode="keyframes",
        phase_b_cfg={
            "episode": {"reset_vsm_on_episode_end": True, "rollouts_per_episode": int(rollouts)},
            "rollout": {
                "mode": "episode_rollout_grouped_repeat_tbptt",
                "shapes": [
                    {
                        "name": f"b{int(blocks)}_r{int(repeats)}",
                        "blocks_per_rollout": int(blocks),
                        "repeats_per_block": int(repeats),
                        "prob": 1.0,
                    }
                ],
                "max_inner_K": 24,
                "short_rollout": {"enable": True, "policy": "early_stop_episode", "min_blocks": 1},
                "block_selection": {
                    "policy": "next_after_history_or_random_future",
                    "next_prob": 1.0,
                    "random_future_prob": 0.0,
                    "require_chronological_execution": True,
                    "distinct_event_blocks": True,
                },
            },
            "final_supervision": {
                "apply": "rollout_final_only",
                "required_roles": ["final_current_recon"],
                "history": {"sample_policy": "recent_then_random", "max_frames": 3},
                "current": {
                    "sample_policy": "all_or_recent",
                    "frame_policy": "all_trained_current_frames",
                    "allow_subsample": bool(current_allow_subsample),
                    "max_frames": current_max_frames,
                },
                "nvs": {
                    "enable": True,
                    "frames_per_rollout": 1,
                    "forbid_evidence_overlap": True,
                    "required_policy": "required_if_future_available",
                },
            },
            "query_observation": {"enable": False},
            "masks": {
                "vsm_scope": "bg_rigid",
                "evidence_mask": "non_sky_non_egocar",
                "prefix_loss_mask": "non_sky_non_egocar",
                "query_label_mask": "non_sky_non_egocar",
            },
        },
        leakage_check_cfg={"enable": True, "same_scene_segment_required": True},
    )


def test_phase_b_final_rollout_shape_evidence_and_loss_timing():
    batch = _scheduler(blocks=4, repeats=4).next_batch()
    meta = batch["request_meta"]

    assert meta["shape_name"] == "b4_r4"
    assert meta["blocks_per_rollout"] == 4
    assert meta["repeats_per_block"] == 4
    assert meta["inner_K"] == 16
    assert meta["requested_blocks_per_rollout"] == 4
    assert meta["actual_blocks_per_rollout"] == 4
    assert meta["requested_inner_K"] == 16
    assert meta["actual_inner_K"] == 16
    assert meta["effective_shape_name"] == "b4_r4"
    assert not meta["short_rollout"]
    assert [i for i, refs in enumerate(meta["prefix_loss_refs_by_step"]) if refs] == [15]

    for refs in meta["evidence_refs_by_step"]:
        assert len({int(f) for f, _ in refs}) == 1
        assert {int(c) for _, c in refs} == {0, 1, 2}

    rollout = meta["phase_b_rollout"]
    final = meta["phase_b_final_supervision"]
    trained = [int(x) for x in rollout["trained_current_frame_indices"]]
    assert final["current_recon_frames"] == trained
    assert final["supervised_current_frame_count"] == 4
    assert final["expected_current_recon_ref_count"] == 12
    expected_current_refs = {(int(f), int(c)) for f in trained for c in range(3)}
    actual_current_refs = {
        tuple(ref)
        for ref, role in zip(final["refs"], final["roles"])
        if str(role) == "final_current_recon"
    }
    assert actual_current_refs == expected_current_refs


def test_phase_b_final_rollout_shape_b6_r3():
    batch = _scheduler(blocks=6, repeats=3).next_batch()
    meta = batch["request_meta"]
    assert meta["shape_name"] == "b6_r3"
    assert meta["requested_blocks_per_rollout"] == 6
    assert meta["actual_blocks_per_rollout"] == 6
    assert meta["repeats_per_block"] == 3
    assert meta["inner_K"] == 18
    assert meta["phase_b_final_supervision"]["supervised_current_frame_count"] == 6
    assert meta["phase_b_final_supervision"]["expected_current_recon_ref_count"] == 18


def test_phase_b_final_rollout_short_shape_is_explicit():
    batch = _scheduler(blocks=4, repeats=4, rollouts=1, blocks_per_episode=2).next_batch()
    meta = batch["request_meta"]

    assert meta["shape_name"] == "b4_r4"
    assert meta["effective_shape_name"] == "b4_r4_short_b2_r4"
    assert meta["requested_blocks_per_rollout"] == 4
    assert meta["actual_blocks_per_rollout"] == 2
    assert meta["requested_inner_K"] == 16
    assert meta["actual_inner_K"] == 8
    assert meta["inner_K"] == 8
    assert meta["short_rollout"] is True
    assert meta["tbptt"]["is_last_chunk"] is True
    final = meta["phase_b_final_supervision"]
    assert final["trained_current_frames"] == final["current_recon_frames"]
    assert final["supervised_current_frame_count"] == 2
    assert final["expected_current_recon_ref_count"] == 6


def test_phase_b_final_rollout_rejects_current_subsample_by_default():
    with pytest.raises(ValueError, match="max_frames=null"):
        _scheduler(blocks=4, repeats=4, current_max_frames=3)


def test_phase_b_final_rollout_history_and_nvs_invariants():
    scheduler = _scheduler(blocks=4, repeats=4)
    scheduler.next_batch()
    batch = scheduler.next_batch()
    meta = batch["request_meta"]

    hist = set(meta["phase_b_rollout"]["history_frame_indices_before_rollout"])
    prior = set(meta["tbptt"]["prior_written_frames"])
    assert hist <= prior

    evidence = {tuple(ref) for step in meta["evidence_refs_by_step"] for ref in step}
    final = meta["phase_b_final_supervision"]
    for ref, role in zip(final["refs"], final["roles"]):
        if str(role).endswith("_nvs"):
            assert tuple(ref) not in evidence


def test_phase_b_final_rollout_episode_end_after_rollouts():
    scheduler = _scheduler(blocks=4, repeats=4, rollouts=4)
    events = []
    for idx in range(4):
        scheduler.next_batch()
        events.extend(scheduler.pop_events())
        if idx < 3:
            assert not any(event["type"] == "episode_end" for event in events)
    assert sum(1 for event in events if event["type"] == "episode_end") == 1


def test_phase_b_final_rollout_resolver_rejects_nvs_evidence_overlap():
    batch = _scheduler(blocks=4, repeats=4).next_batch()
    resolved = resolve_v9_phase_b_batch(batch)
    assert resolved.final_supervision_step_idx == resolved.inner_K - 1
    assert "final_current_recon" in (resolved.final_target_indices_by_role or {})

    bad = copy.deepcopy(batch)
    meta = bad["request_meta"]
    leaked = tuple(meta["evidence_refs_by_step"][0][0])
    meta["target_image_refs"].append(leaked)
    meta["target_image_roles"].append("final_current_nvs")
    meta["flat_render_loss_refs"].append(leaked)
    meta["prefix_loss_refs_by_step"][-1].append(leaked)
    meta["phase_b_final_supervision"]["refs"].append(leaked)
    meta["phase_b_final_supervision"]["roles"].append("final_current_nvs")
    bad["targets"].append({"frame_idx": int(leaked[0]), "cam_idx": int(leaked[1])})

    with pytest.raises(ValueError, match="NVS refs leaked|NVS refs overlap"):
        resolve_v9_phase_b_batch(bad)


def test_phase_b_final_rollout_resolver_rejects_current_frame_mismatch():
    batch = _scheduler(blocks=4, repeats=4).next_batch()
    bad = copy.deepcopy(batch)
    bad["request_meta"]["phase_b_final_supervision"]["current_recon_frames"] = [999]

    with pytest.raises(ValueError, match="final_current_recon frames must equal"):
        resolve_v9_phase_b_batch(bad)
