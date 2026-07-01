from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from models.iforward.model import IForwardModel
from models.iforward.stage2_2.parent_temporal_keys_v2 import ParentTemporalKeysV2
from models.iforward.stage2_3 import (
    ParentOptimizerDeltaKVState,
    ParentOptimizerGatedDeltaKV,
    VisitMeta,
)
from models.iforward.stage2_3.parent_optimizer_gated_delta_kv import (
    LowRankGatedDeltaKVCell,
    _empty_delta_dense,
    _ensure_delta_dense,
    _gather_delta_keyed,
    _rms_rows,
)
from models.iforward.stage2_3.optimizer_write_token import OptimizerWriteTokenBuilder
from models.streetforward.stage6_0.event_encoder import EventPack


def _memory() -> ParentOptimizerGatedDeltaKV:
    torch.manual_seed(3)
    return ParentOptimizerGatedDeltaKV(
        event_dim=4,
        ctx_dim=3,
        token_dim=4,
        key_dim=2,
        value_dim=3,
        adapter_hidden_dim=8,
        visit_dim=4,
        state_rms_max=1.5,
    )


def _event(order: torch.Tensor | None = None) -> EventPack:
    bg = torch.randn(3, 4)
    distant = torch.randn(2, 4)
    rigid = torch.randn(3, 4)
    if order is not None:
        rigid = rigid.index_select(0, order)
    return EventPack(
        event_bg=bg,
        event_distant=distant,
        event_rigid=rigid,
        support_bg=torch.ones(3, 1),
        support_distant=torch.ones(2, 1),
        support_rigid=torch.tensor([[1.0], [3.0], [1.0]]),
        valid_bg=torch.ones(3, dtype=torch.bool),
        valid_distant=torch.ones(2, dtype=torch.bool),
        valid_rigid=torch.ones(3, dtype=torch.bool),
    )


def _keys(order: torch.Tensor | None = None) -> ParentTemporalKeysV2:
    rigid = torch.tensor([200, 200, 201], dtype=torch.long)
    if order is not None:
        rigid = rigid.index_select(0, order)
    return ParentTemporalKeysV2(
        bg=torch.arange(3),
        distant=torch.arange(2) + 100,
        rigid=rigid,
    )


def _visit(kind: str = "assimilate", repeat: int = 0) -> VisitMeta:
    return VisitMeta(
        visit_kind=kind,
        frame_id=7,
        keyframe_id=7,
        sequence_pos=1,
        timestamp_us=700000,
        frame_gap_from_previous_visit=1,
        time_since_same_frame_visit=0.0,
        visit_count_for_frame=0,
        repeat_idx=repeat,
        repeat_budget=4,
        global_update_idx_in_episode=repeat,
        is_first_visit_of_frame=repeat == 0,
        is_last_update_of_episode=False,
    )


def test_gated_delta_kv_cell_shape_mask_and_state_clamp() -> None:
    torch.manual_seed(4)
    cell = LowRankGatedDeltaKVCell(event_dim=64, token_dim=64, key_dim=16, value_dim=32, state_rms_max=0.75)
    state = cell.init_state(128, device=torch.device("cpu"), dtype=torch.float32)
    event = torch.randn(128, 64)
    ctx, _ = cell.read(event, state)
    assert tuple(ctx.shape) == (128, 32)
    token = torch.randn(128, 64) * 100.0
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:64] = True
    next_state, aux = cell.write(token, state, mask, visit_meta=_visit("repair"))
    assert tuple(next_state.kv_state.shape) == (128, 16, 32)
    assert torch.count_nonzero(next_state.kv_state[:64]).item() > 0
    assert torch.allclose(next_state.kv_state[64:], state.kv_state[64:])
    assert torch.isfinite(next_state.kv_state).all()
    assert float(_rms_rows(next_state.kv_state, dims=(-2, -1)).max().item()) <= 0.7501
    assert aux["state_rms_max"] <= 0.7501


def test_gated_delta_kv_query_key_rms_flags_are_honored() -> None:
    torch.manual_seed(14)
    event = torch.randn(16, 8)
    token = torch.randn(16, 8)
    mask = torch.ones(16, dtype=torch.bool)
    unit = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=4, query_rms_unit=True, key_rms_unit=True)
    raw = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=4, query_rms_unit=False, key_rms_unit=False)
    raw.load_state_dict(unit.state_dict())
    state_unit = unit.init_state(16, device=torch.device("cpu"), dtype=torch.float32)
    state_raw = raw.init_state(16, device=torch.device("cpu"), dtype=torch.float32)
    _, read_aux_unit = unit.read(event, state_unit)
    _, read_aux_raw = raw.read(event, state_raw)
    _, write_aux_unit = unit.write(token, state_unit, mask)
    _, write_aux_raw = raw.write(token, state_raw, mask)
    assert abs(read_aux_unit["query_rms_mean"] - 0.5) < 1.0e-3
    assert abs(write_aux_unit["key_rms_mean"] - 0.5) < 1.0e-3
    assert abs(read_aux_raw["query_rms_mean"] - read_aux_unit["query_rms_mean"]) > 1.0e-2
    assert abs(write_aux_raw["key_rms_mean"] - write_aux_unit["key_rms_mean"]) > 1.0e-2


def test_optimizer_write_token_include_flags_zero_disabled_inputs() -> None:
    torch.manual_seed(15)
    builder = OptimizerWriteTokenBuilder(
        event_dim=4,
        visit_dim=3,
        token_dim=5,
        hidden_dim=8,
        include_spatial_event=False,
        include_parent_event=False,
        include_delta_summary=False,
        include_visit_embedding=False,
    )
    spatial_a = torch.randn(6, 4)
    spatial_b = torch.randn(6, 4) * 7.0
    fused_a = torch.randn(6, 4)
    fused_b = torch.randn(6, 4) * 5.0
    visit_a = torch.randn(1, 3)
    visit_b = torch.randn(1, 3) * 9.0
    support = torch.ones(6, 1)
    valid = torch.ones(6, dtype=torch.bool)
    delta_a = SimpleNamespace(summary7=torch.randn(6, 7))
    delta_b = SimpleNamespace(summary7=torch.randn(6, 7) * 11.0)
    out_a = builder.branch(spatial=spatial_a, fused=fused_a, support=support, valid=valid, visit=visit_a, delta_branch=delta_a)
    out_b = builder.branch(spatial=spatial_b, fused=fused_b, support=support, valid=valid, visit=visit_b, delta_branch=delta_b)
    assert out_a is not None
    assert out_b is not None
    assert torch.allclose(out_a, out_b, atol=1.0e-6, rtol=1.0e-6)


def test_delta_kv_dense_grow_preserves_existing_rows() -> None:
    cell = LowRankGatedDeltaKVCell(event_dim=4, token_dim=4, key_dim=2, value_dim=3)
    state = _empty_delta_dense(cell, rows=3, device=torch.device("cpu"), dtype=torch.float32)
    state.kv_state[1] = 2.0
    grown = _ensure_delta_dense(cell, state, rows=5, device=torch.device("cpu"), dtype=torch.float32)
    assert tuple(grown.kv_state.shape) == (5, 2, 3)
    assert torch.allclose(grown.kv_state[:3], state.kv_state)
    assert torch.count_nonzero(grown.kv_state[3:]).item() == 0


def test_gated_delta_kv_preview_unseen_zero_and_repair_writes() -> None:
    mem = _memory()
    event = _event()
    state = ParentOptimizerDeltaKVState.empty()
    preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair", 0))
    assert state.bg.dense is None
    assert torch.allclose(preview.event.event_bg, event.event_bg)
    assert preview.aux["iforward/parent_optimizer_gdkv/bg_preview_seen_ratio"] == 0.0
    for repeat in range(2):
        preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair", repeat))
        state, aux = mem.write(
            spatial_event=event,
            fused_event=preview.event,
            state=state,
            keys=_keys(),
            visit_meta=_visit("repair", repeat),
        )
        assert aux["iforward/parent_optimizer_gdkv/write"] == 1.0
        assert aux["iforward/parent_optimizer_mamba/write"] == 1.0
    assert state.global_update_step == 2
    assert state.bg.dense is not None
    assert int(state.bg.dense.update_count.sum().item()) == 6
    assert state.rigid.keyed is not None
    assert state.rigid.keyed.keys.detach().cpu().tolist() == [200, 201]


def test_gated_delta_kv_keyed_gather_scatter_repeated_keys() -> None:
    mem = _memory()
    event = _event()
    state, _ = mem.write(spatial_event=event, state=ParentOptimizerDeltaKVState.empty(), keys=_keys(), visit_meta=_visit())
    assert state.rigid.keyed is not None
    keys = torch.tensor([200, 201, 202, 200], dtype=torch.long)
    gathered, meta = _gather_delta_keyed(mem.cells["rigid"], state.rigid.keyed, keys, device=torch.device("cpu"), dtype=torch.float32)
    assert gathered.seen.detach().cpu().tolist() == [True, True, False, True]
    assert meta["update_count"].detach().cpu().tolist() == [1, 1, 0, 1]


def test_gated_delta_kv_repair_order_smoke_is_finite_and_bounded() -> None:
    torch.manual_seed(8)
    order = torch.tensor([2, 0, 1], dtype=torch.long)
    mem_a = _memory()
    mem_b = _memory()
    event_a = _event()
    event_b = _event(order=order)
    state_a, _ = mem_a.write(spatial_event=event_a, state=ParentOptimizerDeltaKVState.empty(), keys=_keys(), visit_meta=_visit("repair"))
    state_b, _ = mem_b.write(
        spatial_event=event_b,
        state=ParentOptimizerDeltaKVState.empty(),
        keys=_keys(order=order),
        visit_meta=_visit("repair"),
    )
    assert state_a.rigid.keyed is not None
    assert state_b.rigid.keyed is not None
    for state in (state_a, state_b):
        rms = _rms_rows(state.rigid.keyed.kv_state, dims=(-2, -1))
        assert torch.isfinite(state.rigid.keyed.kv_state).all()
        assert float(rms.max().item()) <= 1.5001


def test_stage3_1_config_and_legacy_ablation_alias() -> None:
    cfg = OmegaConf.load("configs/iforward/iforward_stage3_1_lowrank_gated_delta_kv.yaml")
    assert cfg.model.iforward.version == "stage3_1_lowrank_gated_delta_kv_lift"
    assert cfg.model.feat_2d_channels == 32
    assert cfg.model.feature_extractor.residual_unet.feat_channels == 16
    assert cfg.model.feature_extractor.dino.out_channels == 16
    assert cfg.model.struct_decoder.feat_2d_channels == 32
    assert cfg.model.iforward.lifting.context_dim == 32
    assert cfg.model.iforward.parent_spatial.context_dim == 32
    assert cfg.model.stage6_0.struct_event_decoder.feat_2d_dim == 32
    assert cfg.model.iforward.parent_optimizer_memory.type == "lowrank_gated_delta_kv"
    assert cfg.model.iforward.parent_optimizer_memory.gated_delta_kv.K == 16
    assert cfg.model.iforward.parent_optimizer_memory.gated_delta_kv.V == 32
    ParentOptimizerGatedDeltaKV(
        event_dim=4,
        ctx_dim=3,
        token_dim=4,
        key_dim=2,
        value_dim=3,
        decay_min=cfg.model.iforward.parent_optimizer_memory.gated_delta_kv.decay_min,
    )
    assert cfg.model.stage6_0.posterior_updater.branch_scope.distant.update_scales is True
    assert float(cfg.model.stage6_0.posterior_updater.appearance_detail.attribute_gates.distant.scales) > 0.0
    model = object.__new__(IForwardModel)
    model.is_stage2_3_optimizer_mamba = True
    assert IForwardModel._normalize_ablation_name(model, "shuffle_memory") == "mamba_shuffle_state"
    assert IForwardModel._normalize_ablation_name(model, "read_only") == "mamba_read_only"
