from __future__ import annotations

import torch

from models.streetforward.stage6_0.event_encoder import EventPack
from models.iforward.stage2_2.parent_temporal_keys_v2 import ParentTemporalKeysV2
from models.iforward.stage2_2.parent_temporal_mamba_v2 import ParentTemporalMemoryV2
from models.iforward.stage2_2.parent_temporal_state_v2 import ParentTemporalStateV2


def _memory():
    torch.manual_seed(1)
    return ParentTemporalMemoryV2(event_dim=4, ctx_dim=3, model_dim=5, state_dim=2, conv_kernel=2, motion_embed_dim=2)


def _event():
    return EventPack(
        event_bg=torch.randn(3, 4),
        event_distant=torch.randn(2, 4),
        event_rigid=torch.randn(3, 4),
        support_bg=torch.ones(3, 1),
        support_distant=torch.ones(2, 1),
        support_rigid=torch.tensor([[1.0], [3.0], [1.0]]),
        valid_bg=torch.ones(3, dtype=torch.bool),
        valid_distant=torch.ones(2, dtype=torch.bool),
        valid_rigid=torch.ones(3, dtype=torch.bool),
    )


def _keys():
    return ParentTemporalKeysV2(
        bg=torch.arange(3),
        distant=torch.arange(2) + 100,
        rigid=torch.tensor([200, 200, 201], dtype=torch.long),
    )


def test_stage2_2_temporal_preview_no_mutation_and_unseen_zero():
    mem = _memory()
    event = _event()
    state = ParentTemporalStateV2.empty()
    preview = mem.preview(event=event, state=state, keys=_keys(), timestamp_sec=1.0)
    assert state.bg.dense is None
    assert torch.allclose(preview.event.event_bg, event.event_bg)
    assert preview.aux["iforward/parent_temporal_v2/bg_preview_seen_ratio"] == 0.0


def test_stage2_2_temporal_commit_once_and_timestamp():
    mem = _memory()
    event = _event()
    state, aux = mem.commit(event=event, state=ParentTemporalStateV2.empty(), keys=_keys(), block_id=2, timestamp_sec=1.5)
    assert aux["iforward/parent_temporal_v2/raw_frame_commit"] == 1.0
    assert state.last_committed_block_id == 2
    assert state.last_timestamp_sec == 1.5
    assert state.bg.dense is not None
    assert int(state.bg.dense.seen.sum().item()) == 3
    assert torch.allclose(state.bg.dense.last_timestamp_sec, torch.full((3,), 1.5))


def test_stage2_2_temporal_repair_no_timestamp_update():
    mem = _memory()
    event = _event()
    state, _ = mem.commit(event=event, state=ParentTemporalStateV2.empty(), keys=_keys(), block_id=1, timestamp_sec=1.0)
    skipped, aux = mem.commit(
        event=event,
        state=state,
        keys=_keys(),
        block_id=9,
        timestamp_sec=2.0,
        physical_time_advance=False,
    )
    assert skipped is state
    assert aux["iforward/parent_temporal_v2/commit_skipped_no_time_advance"] == 1.0
    assert skipped.last_timestamp_sec == 1.0


def test_stage2_2_temporal_duplicate_rigid_keys_are_aggregated_and_detach_reset():
    mem = _memory()
    state, _ = mem.commit(event=_event(), state=ParentTemporalStateV2.empty(), keys=_keys(), block_id=3, timestamp_sec=2.0)
    assert state.rigid.keyed is not None
    assert state.rigid.keyed.keys.detach().cpu().tolist() == [200, 201]
    detached = state.detach()
    assert detached is not state
    assert detached.rigid.keyed is not state.rigid.keyed
    assert ParentTemporalStateV2.empty().count_tokens()["parent_temporal_v2_last_committed_block_id"] == -1.0


def test_stage2_2_temporal_per_parent_timestamp_changes_preview():
    mem = _memory()
    event = _event()
    state, _ = mem.commit(event=event, state=ParentTemporalStateV2.empty(), keys=_keys(), block_id=1, timestamp_sec=1.0)
    near = mem.preview(event=event, state=state, keys=_keys(), timestamp_sec=1.5, motion_meta={"visit_kind": "causal_first"})
    far = mem.preview(event=event, state=state, keys=_keys(), timestamp_sec=3.0, motion_meta={"visit_kind": "causal_first"})
    assert (near.event.event_bg - far.event.event_bg).abs().max() > 1.0e-8


def test_stage2_2_temporal_repair_preview_zeroes_physical_motion():
    mem = _memory()
    event = _event()
    state, _ = mem.commit(event=event, state=ParentTemporalStateV2.empty(), keys=_keys(), block_id=1, timestamp_sec=1.0)
    noisy = mem.preview(
        event=event,
        state=state,
        keys=_keys(),
        timestamp_sec=9.0,
        motion_meta={
            "visit_kind": "repair",
            "delta_t_sec": 99.0,
            "frame_gap": 99.0,
            "ego_delta_translation": torch.tensor([[9.0, 8.0, 7.0]]),
            "ego_delta_yaw": 3.14,
        },
    )
    zero = mem.preview(
        event=event,
        state=state,
        keys=_keys(),
        timestamp_sec=9.0,
        motion_meta={
            "visit_kind": "repair",
            "delta_t_sec": 0.0,
            "frame_gap": 0.0,
            "ego_delta_translation": torch.zeros(1, 3),
            "ego_delta_yaw": 0.0,
        },
    )
    assert torch.allclose(noisy.event.event_bg, zero.event.event_bg, atol=1e-6)
    assert torch.allclose(noisy.event.event_rigid, zero.event.event_rigid, atol=1e-6)
