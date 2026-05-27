from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from datasets.multi_scene_dataset_v4 import MultiSceneDatasetV4, SegmentIndexV4
from datasets.train_scheduler_long_phase_b import TrainSchedulerLongPhaseB
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.stage6_0 import EventPack, LocalGSState
from models.streetforward.stage6_0.local_gs_state import LocalBranchState
from models.streetforward.stage6_0.phase_b_long import (
    LongStreamingVSM,
    PhaseBOffsetState,
    VSMOffsetDecoder,
    materialize_phase_b_state,
    resolve_long_phase_b_batch,
)


def _branch(n: int, hidden: int = 4, sh_rest_bases: int = 0) -> LocalBranchState:
    return LocalBranchState(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.nn.functional.normalize(torch.ones(n, 4), dim=-1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, int(sh_rest_bases), 3),
        hidden=torch.zeros(n, hidden),
    )


def _long_batch():
    meta = {
        "scheduler_version": "long_v1",
        "scheduler_phase": "6_0_phase_b",
        "assembly_mode": "image_ref_long_v1",
        "inner_K": 4,
        "evidence_refs_by_step": [[(10, 0)], [(10, 1)], [(11, 0)], [(11, 1)]],
        "source_image_refs": [(10, 0), (10, 1), (11, 0), (11, 1)],
        "target_image_refs": [(10, 0), (10, 2), (11, 1), (11, 2)],
        "target_image_roles": [
            "final_history_recon",
            "final_history_nvs",
            "final_current_recon",
            "final_current_nvs",
        ],
        "required_final_roles": ["final_current_recon", "final_current_nvs"],
        "final_history_recon_refs": [(10, 0)],
        "final_history_nvs_refs": [(10, 2)],
        "final_current_recon_refs": [(11, 1)],
        "final_current_nvs_refs": [(11, 2)],
        "shape_name": "r2_a2",
        "repeats_per_anchor": 2,
        "anchors_per_rollout": 2,
        "anchor_frames_chronological": [10, 11],
        "anchor_frames_rollout_order": [10, 11],
        "visits": [
            {
                "step_idx": 0,
                "anchor_id": 0,
                "frame_idx": 10,
                "cam_idx": 0,
                "repeat_idx": 0,
                "rollout_order_rank": 0,
                "chronological_rank": 0,
                "visit_pos_code": 0.0,
                "frame_time_code": 0.0,
                "chronological_rank_code": 0.0,
                "repeat_idx_code": 0.0,
            },
            {
                "step_idx": 1,
                "anchor_id": 0,
                "frame_idx": 10,
                "cam_idx": 1,
                "repeat_idx": 1,
                "rollout_order_rank": 0,
                "chronological_rank": 0,
                "visit_pos_code": 0.333,
                "frame_time_code": 0.0,
                "chronological_rank_code": 0.0,
                "repeat_idx_code": 1.0,
            },
            {
                "step_idx": 2,
                "anchor_id": 1,
                "frame_idx": 11,
                "cam_idx": 0,
                "repeat_idx": 0,
                "rollout_order_rank": 1,
                "chronological_rank": 1,
                "visit_pos_code": 0.667,
                "frame_time_code": 1.0,
                "chronological_rank_code": 1.0,
                "repeat_idx_code": 0.0,
            },
            {
                "step_idx": 3,
                "anchor_id": 1,
                "frame_idx": 11,
                "cam_idx": 1,
                "repeat_idx": 1,
                "rollout_order_rank": 1,
                "chronological_rank": 1,
                "visit_pos_code": 1.0,
                "frame_time_code": 1.0,
                "chronological_rank_code": 1.0,
                "repeat_idx_code": 1.0,
            },
        ],
        "step_frame_indices": [10, 10, 11, 11],
        "step_repeat_indices": [0, 1, 0, 1],
        "query_label_refs": [],
        "prefix_loss_refs_by_step": [[], [], [], []],
        "nearby_loss_refs_by_step": [[], [], [], []],
        "block_loss_refs_by_step": [[], [], [], []],
        "rigid_meta": {"has_stable_ids": False},
        "distant_meta": {"mode": "frozen_render_only"},
        "tbptt": {"enable": False},
    }
    return {
        "request_meta": meta,
        "_scheduler_long_phase_b": dict(meta),
        "source": {"frame_indices": torch.tensor([10, 10, 11, 11]), "cam_indices": torch.tensor([0, 1, 0, 1])},
        "target": {"frame_indices": torch.tensor([10, 10, 11, 11]), "cam_indices": torch.tensor([0, 2, 1, 2])},
        "targets": [
            {"frame_idx": 10, "cam_idx": 0},
            {"frame_idx": 10, "cam_idx": 2},
            {"frame_idx": 11, "cam_idx": 1},
            {"frame_idx": 11, "cam_idx": 2},
        ],
    }


def test_long_phase_b_resolver_maps_final_indices_and_rejects_old_roles():
    resolved = resolve_long_phase_b_batch(_long_batch())
    assert resolved.inner_K == 4
    assert resolved.evidence_source_indices_by_step == [[0], [1], [2], [3]]
    assert resolved.final_history_recon_target_indices == [0]
    assert resolved.final_history_nvs_target_indices == [1]
    assert resolved.final_current_recon_target_indices == [2]
    assert resolved.final_current_nvs_target_indices == [3]
    assert resolved.step_repeat_indices == [0, 1, 0, 1]
    assert resolved.step_anchor_ids == [0, 0, 1, 1]
    assert resolved.visit_time_codes[-1] == (1.0, 1.0, 1.0, 1.0)

    bad = _long_batch()
    bad["request_meta"] = dict(bad["request_meta"], query_label_refs=[(12, 0)])
    with pytest.raises(ValueError, match="empty query_label_refs"):
        resolve_long_phase_b_batch(bad)

    bad = _long_batch()
    bad["request_meta"] = dict(
        bad["request_meta"],
        target_image_roles=["prefix_loss", "final_history_nvs", "final_current_recon", "final_current_nvs"],
    )
    with pytest.raises(ValueError, match="target roles"):
        resolve_long_phase_b_batch(bad)

    bad = _long_batch()
    bad["request_meta"] = dict(
        bad["request_meta"],
        target_image_roles=["final_history", "final_history_nvs", "final_current_recon", "final_current_nvs"],
    )
    with pytest.raises(ValueError, match="coarse"):
        resolve_long_phase_b_batch(bad)

    bad = _long_batch()
    bad["request_meta"] = dict(bad["request_meta"], prefix_loss_refs_by_step=[[(10, 0)], [], [], []])
    with pytest.raises(ValueError, match="empty prefix_loss_refs_by_step"):
        resolve_long_phase_b_batch(bad)

    bad = _long_batch()
    visits = [dict(x) for x in bad["request_meta"]["visits"]]
    visits[0]["cam_idx"] = 2
    bad["request_meta"] = dict(bad["request_meta"], visits=visits)
    with pytest.raises(ValueError, match="cam_idx"):
        resolve_long_phase_b_batch(bad)

    bad = _long_batch()
    bad["request_meta"] = dict(
        bad["request_meta"],
        target_image_roles=[
            "final_history_recon",
            "final_history_nvs",
            "final_current_recon",
            "final_history_nvs",
        ],
        final_current_nvs_refs=[],
    )
    with pytest.raises(ValueError, match="required Long role final_current_nvs is empty"):
        resolve_long_phase_b_batch(bad)


def test_bg_streaming_vsm_and_decoder_are_readout_only():
    base = LocalGSState(bg=_branch(5))
    vsm = LongStreamingVSM(event_dim=48, bg_mem_dim=16, rigid_mem_dim=16, dtype="fp32")
    state = vsm.init_state(base_state=base)
    event = EventPack(
        event_bg=torch.randn(5, 48),
        support_bg=torch.ones(5),
        valid_bg=torch.ones(5, dtype=torch.bool),
        obs_code_bg=torch.zeros(5, 2),
    )
    event = SimpleNamespace(**event.__dict__)
    event.view_code_bg = event.obs_code_bg
    state, read, aux = vsm.write_read(
        state=state,
        event=event,
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta={},
        distant_mode="frozen_render_only",
    )
    assert read.bg.shape == (5, 16)
    assert state.bg_seen.shape == (5, 1)
    assert aux["vsm_bg_write_gate_mean"] > 0.0

    decoder = VSMOffsetDecoder(bg_mem_dim=16, rigid_mem_dim=16)
    delta = decoder(read=read)
    assert delta.bg.means.shape == (5, 3)
    assert "event" not in inspect.signature(decoder.forward).parameters

    zero_event = SimpleNamespace(**event.__dict__)
    zero_event.valid_bg = torch.zeros(5, dtype=torch.bool)
    _, read_zero, _ = vsm.write_read(
        state=vsm.init_state(base_state=base),
        event=zero_event,
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta={},
        distant_mode="frozen_render_only",
    )
    delta_zero = decoder(read=read_zero)
    assert torch.count_nonzero(delta_zero.bg.mask) == 0
    assert torch.count_nonzero(delta_zero.bg.means) == 0

    vsm_soft = LongStreamingVSM(
        event_dim=48,
        bg_mem_dim=16,
        rigid_mem_dim=16,
        dtype="fp32",
        support_fallback_when_no_valid=True,
    )
    _, read_soft, aux_soft = vsm_soft.write_read(
        state=vsm_soft.init_state(base_state=base),
        event=zero_event,
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta={},
        distant_mode="frozen_render_only",
    )
    delta_soft = decoder(read=read_soft)
    assert aux_soft["vsm_bg_support_fallback_used"] == 1.0
    assert aux_soft["vsm_bg_seen_rows"] == 5.0
    assert torch.count_nonzero(delta_soft.bg.mask) == 5


def test_bg_active_sparse_path_preserves_cross_step_gradients():
    torch.manual_seed(7)
    base = LocalGSState(bg=_branch(6))
    vsm = LongStreamingVSM(event_dim=48, bg_mem_dim=16, rigid_mem_dim=16, dtype="fp32")
    decoder = VSMOffsetDecoder(bg_mem_dim=16, rigid_mem_dim=16)
    state = vsm.init_state(base_state=base)

    def make_event(valid_rows):
        valid = torch.zeros(6, dtype=torch.bool)
        valid[torch.tensor(valid_rows, dtype=torch.long)] = True
        event = EventPack(
            event_bg=torch.randn(6, 48),
            support_bg=valid.to(dtype=torch.float32),
            valid_bg=valid,
            obs_code_bg=torch.zeros(6, 2),
        )
        event = SimpleNamespace(**event.__dict__)
        event.view_code_bg = event.obs_code_bg
        return event

    state_a, read_a, aux_a = vsm.write_read(
        state=state,
        event=make_event([0, 2]),
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta={},
        distant_mode="frozen_render_only",
    )
    assert read_a.bg_indices is not None
    assert read_a.bg_indices.tolist() == [0, 2]
    assert read_a.bg.shape == (2, 16)
    assert aux_a["vsm_bg_active_rows"] == 2.0

    state_a.bg_h.retain_grad()
    state_b, read_b, aux_b = vsm.write_read(
        state=state_a,
        event=make_event([2, 3]),
        step_idx=1,
        frame_idx=10,
        repeat_idx=1,
        rigid_meta={},
        distant_mode="frozen_render_only",
    )
    assert read_b.bg_indices is not None
    assert read_b.bg_indices.tolist() == [2, 3]
    assert aux_b["vsm_bg_seen_rows"] == 3.0
    assert torch.count_nonzero(state_b.bg_h[1].detach()) == 0

    delta = decoder(read=read_b)
    assert delta.bg.indices is not None
    offset = PhaseBOffsetState.zeros_like(base_state=base)
    offset = offset.apply(delta, frame_idx=10, rigid_meta={})
    assert torch.count_nonzero(offset.bg_means[[0, 1, 4, 5]].detach()) == 0

    loss = offset.bg_means[2].sum()
    loss.backward()
    assert state_a.bg_h.grad is not None
    assert float(state_a.bg_h.grad[2].abs().sum()) > 0.0
    assert float(state_a.bg_h.grad[0].abs().sum()) == 0.0
    assert sum(float(p.grad.abs().sum()) for p in vsm.parameters() if p.grad is not None) > 0.0
    assert sum(float(p.grad.abs().sum()) for p in decoder.parameters() if p.grad is not None) > 0.0


def test_rigid_stable_memory_unstable_snapshot_and_distant_frozen():
    base = LocalGSState(bg=_branch(3), distant=_branch(2), rigid=_branch(4))
    rigid_meta = {"stable_mask": torch.tensor([True, False, True, False])}
    vsm = LongStreamingVSM(event_dim=48, bg_mem_dim=16, rigid_mem_dim=16, dtype="fp32")
    state = vsm.init_state(base_state=base, rigid_meta=rigid_meta)
    event = EventPack(
        event_bg=torch.randn(3, 48),
        event_distant=torch.randn(2, 48),
        event_rigid=torch.randn(3, 48),
        support_bg=torch.ones(3),
        support_distant=torch.ones(2),
        support_rigid=torch.ones(3),
        valid_bg=torch.ones(3, dtype=torch.bool),
        valid_distant=torch.ones(2, dtype=torch.bool),
        valid_rigid=torch.ones(3, dtype=torch.bool),
        obs_code_bg=torch.zeros(3, 2),
        obs_code_distant=torch.zeros(2, 2),
        obs_code_rigid=torch.zeros(3, 2),
        route=SimpleNamespace(S=torch.tensor([0, 2, 3])),
    )
    event = SimpleNamespace(**event.__dict__)
    event.view_code_bg = event.obs_code_bg
    event.view_code_rigid = event.obs_code_rigid
    state, read, aux = vsm.write_read(
        state=state,
        event=event,
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta=rigid_meta,
        distant_mode="frozen_render_only",
    )
    assert read.rigid is not None and read.rigid.shape == (3, 16)
    assert aux["vsm_rigid_stable_rows"] == 2.0
    assert aux["vsm_rigid_unstable_rows"] == 1.0
    assert read.distant is None

    decoder = VSMOffsetDecoder(bg_mem_dim=16, rigid_mem_dim=16)
    offset = PhaseBOffsetState.zeros_like(base_state=base)
    delta = decoder(read=read)
    offset = offset.apply(delta, frame_idx=10, rigid_meta=rigid_meta)
    assert offset.rigid_frame_snapshots[10].mask[3].item() == pytest.approx(1.0)
    assert delta.distant is None
    mat_10 = materialize_phase_b_state(base_state=base, offset=offset, target_frame_idx=10, rigid_meta=rigid_meta)
    mat_11 = materialize_phase_b_state(base_state=base, offset=offset, target_frame_idx=11, rigid_meta=rigid_meta)
    assert getattr(mat_10, "_phase_b_long_rigid_fallback_rows") == 1
    assert getattr(mat_11, "_phase_b_long_rigid_fallback_rows") == 2


def test_distant_vsm_sparse_appearance_scale_offsets_preserve_geometry():
    base = LocalGSState(bg=_branch(3), distant=_branch(4, sh_rest_bases=3))
    vsm = LongStreamingVSM(
        event_dim=48,
        bg_mem_dim=16,
        rigid_mem_dim=16,
        distant_mem_dim=12,
        dtype="fp32",
        distant_mode="appearance_scale_only",
    )
    state = vsm.init_state(base_state=base, distant_mode="appearance_scale_only")
    assert state.distant_h is not None and state.distant_h.shape == (4, 12)
    event = EventPack(
        event_bg=torch.randn(3, 48),
        event_distant=torch.randn(4, 48),
        support_bg=torch.ones(3),
        support_distant=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        valid_bg=torch.ones(3, dtype=torch.bool),
        valid_distant=torch.tensor([True, False, True, False]),
        obs_code_bg=torch.zeros(3, 2),
        obs_code_distant=torch.zeros(4, 2),
    )
    event = SimpleNamespace(**event.__dict__)
    event.view_code_bg = event.obs_code_bg
    event.view_code_distant = event.obs_code_distant
    state, read, aux = vsm.write_read(
        state=state,
        event=event,
        step_idx=0,
        frame_idx=10,
        repeat_idx=0,
        rigid_meta={},
        distant_mode="appearance_scale_only",
    )
    assert read.distant is not None and read.distant.shape == (2, 12)
    assert read.distant_indices.tolist() == [0, 2]
    assert aux["vsm_distant_active_rows"] == 2.0
    assert state.distant_seen is not None
    assert state.distant_seen[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]

    decoder = VSMOffsetDecoder(
        bg_mem_dim=16,
        rigid_mem_dim=16,
        distant_mem_dim=12,
        distant_mode="appearance_scale_only",
        distant_sh_rest_bases=3,
        distant_sh_rest_update_bases=2,
    )
    delta = decoder(read=read, distant_mode="appearance_scale_only")
    assert delta.distant is not None
    assert delta.distant.indices.tolist() == [0, 2]
    assert delta.distant.scales_log.shape == (2, 3)
    assert delta.distant.opacity_logit.shape == (2, 1)
    assert delta.distant.sh_dc.shape == (2, 3)
    assert delta.distant.sh_rest.shape == (2, 2, 3)

    offset = PhaseBOffsetState.zeros_like(base_state=base)
    offset = offset.apply(delta, frame_idx=10, rigid_meta={})
    assert offset.distant_scales_log is not None
    assert torch.count_nonzero(offset.distant_scales_log[[1, 3]]) == 0
    assert offset.distant_sh_rest is not None
    assert torch.count_nonzero(offset.distant_sh_rest[:, 2:]) == 0

    mat = materialize_phase_b_state(base_state=base, offset=offset, target_frame_idx=10, rigid_meta={})
    assert mat.distant is not None
    assert torch.allclose(mat.distant.means, base.distant.means)
    assert torch.allclose(mat.distant.quats, base.distant.quats)


def _make_sidx() -> SegmentIndexV4:
    frames = [10, 20, 30, 40]
    return SegmentIndexV4(
        scene_id=1,
        segment_id=0,
        num_cams=3,
        frame_indices=frames,
        test_frame_indices=[],
        train_frame_set=frozenset(frames),
        test_frame_set=frozenset(),
        keyframe_indices=[0, 1, 2, 3],
        keyframe_to_frames={0: [10], 1: [20], 2: [30], 3: [40]},
        frame_to_keyframe={10: 0, 20: 1, 30: 2, 40: 3},
        segment_first_frame_idx=10,
        train_image_refs=tuple((f, c) for f in frames for c in range(3)),
        test_image_refs=tuple(),
    )


def test_long_scheduler_emits_long_v1_without_first_repeat_write_flags():
    sidx = _make_sidx()
    ds = MagicMock()
    ds.initialize = MagicMock()
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=sidx)
    ds._assemble_segment_batch_from_image_refs = MagicMock(side_effect=lambda *args, **kwargs: {"request_meta": {}})
    sch = TrainSchedulerLongPhaseB(
        dataset=ds,
        episode_window_cfg={"frames_per_window": 4, "min_frames_required": 2, "rollout_budget_per_episode": 2},
        rollout_shapes_schedule=[
            {
                "start_step": 0,
                "shapes": [{"name": "r2_a2", "repeats_per_anchor": 2, "anchors_per_rollout": 2, "prob": 1.0}],
            }
        ],
        anchor_sampling_cfg={
            "min_temporal_span": 1,
            "max_temporal_span": 40,
            "allow_chronological_order_prob": 1.0,
            "allow_reverse_order_prob": 0.0,
            "allow_random_order_prob": 0.0,
        },
        traversal_cfg={"segment_order": "ascending", "scene_order": "ascending"},
        preload_cfg={"emit_hints": False},
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
        evidence_cfg={
            "cams_per_visit": 1,
            "distinct_cams_per_anchor": 1,
            "reserve_nvs_cams_per_anchor": 1,
            "allow_same_cam_repeat": True,
        },
        final_supervision_cfg={
            "history_anchor_count": 1,
            "final_history_recon": {"cams_per_anchor": 1},
            "final_history_nvs": {"cams_per_anchor": 1},
            "final_current_recon": {"cams": 1},
            "final_current_nvs": {"cams": 1},
            "max_nvs_fallback_ratio": 0.0,
            "required_final_roles": ["final_current_recon", "final_current_nvs"],
        },
    )
    batch = sch.next_batch()
    meta = batch["request_meta"]
    assert meta["scheduler_version"] == "long_v1"
    assert meta["scheduler_phase"] == "6_0_phase_b"
    assert meta["assembly_mode"] == "image_ref_long_v1"
    assert meta["inner_K"] == 4
    assert meta["shape_name"] == "r2_a2"
    assert len(meta["visits"]) == 4
    assert all(len(step) > 0 for step in meta["evidence_refs_by_step"])
    assert set(meta["target_image_roles"]) <= {
        "final_history_recon",
        "final_history_nvs",
        "final_current_recon",
        "final_current_nvs",
    }
    evidence_cams_by_frame = {}
    for step in meta["evidence_refs_by_step"]:
        frame_idx, cam_idx = step[0]
        evidence_cams_by_frame.setdefault(int(frame_idx), set()).add(int(cam_idx))
    assert all(len(cams) == 1 for cams in evidence_cams_by_frame.values())
    assert meta["nvs_fallback_to_evidence_cam_ratio"] == 0.0
    assert "step_memory_write_flags" not in meta
    assert "step_block_indices" not in meta
    events = sch.pop_events()
    assert [ev["type"] for ev in events] == ["segment_begin", "rollout_end"]

    _ = sch.next_batch()
    events = sch.pop_events()
    assert [ev["type"] for ev in events] == ["rollout_end", "episode_end"]


def test_long_trainer_skips_zero_grad_no_support_rollout_but_keeps_real_zero_grad_errors():
    model = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    torch.nn.Module.__init__(model)
    model.config = {"model": {"stage6_0": {"phase_b_long": {"skip_no_support_rollout": True}}}}
    model.stage6_long_vsm = torch.nn.Linear(1, 1)
    model.stage6_long_offset_decoder = torch.nn.Linear(1, 1)

    roles = SimpleNamespace(
        request_meta={
            "scene_id": 117,
            "segment_id": 0,
            "shape_name": "r2_a2",
            "inner_K": 4,
            "target_image_roles": ["final_current_recon", "final_current_nvs"],
            "source_image_refs": [(171, 2), (180, 0)],
            "target_image_refs": [(171, 2), (180, 2)],
        },
        shape_name="r2_a2",
        inner_K=4,
    )
    out = {
        "roles": roles,
        "stats": {},
        "per_step": [
            {
                "vsm_bg_support_max": 0.0,
                "vsm_bg_seen_rows": 0.0,
                "vsm_rigid_support_max": 0.0,
                "vsm_rigid_seen_rows": 0.0,
            }
        ],
    }
    sums = model._stage6_assert_required_group_grads_phase_b_long(out)
    assert sums["phase_b_long/skipped_no_support_rollout"] == 1.0

    out["per_step"][0]["vsm_bg_support_max"] = 0.5
    with pytest.raises(RuntimeError, match="zero gradient"):
        model._stage6_assert_required_group_grads_phase_b_long(out)
