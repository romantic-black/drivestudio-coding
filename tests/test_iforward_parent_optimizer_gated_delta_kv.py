from __future__ import annotations

from types import SimpleNamespace

import pytest
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


def _constant_key_residual_cell(*, cleanup_enable: bool = False) -> LowRankGatedDeltaKVCell:
    cell = LowRankGatedDeltaKVCell(
        event_dim=2,
        token_dim=2,
        key_dim=2,
        value_dim=2,
        value_rms_max=100.0,
        state_rms_max=100.0,
        update_rule="balanced_residual_delta_v1",
        alpha_init=1.0,
        surprise_gating=False,
        min_alpha_on_unseen=1.0,
        decay_min=1.0,
        cleanup_enable=cleanup_enable,
        cleanup_key="current_key",
        cleanup_max=0.2,
        cleanup_init=0.2,
        cleanup_by_kind={"repair": 1.0, "default": 1.0},
    )
    with torch.no_grad():
        cell.key_proj.weight.zero_()
        cell.key_proj.bias.copy_(torch.tensor([1.0, -1.0]))
        cell.value_proj.weight.copy_(torch.eye(2))
        cell.value_proj.bias.zero_()
    return cell


def _read_write_address(cell: LowRankGatedDeltaKVCell, token: torch.Tensor, state) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, key, value, _ = cell._write_inputs(token, state, visit_meta=None)
    old = torch.einsum("nkv,nk->nv", state.kv_state.float(), key)
    return old, value


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


def test_gated_delta_kv_residual_delta_fixed_point_stops_repeated_write() -> None:
    cell = _constant_key_residual_cell()
    state = cell.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
    token = torch.tensor([[0.75, -0.25]], dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.bool)
    old0, value = _read_write_address(cell, token, state)
    initial_residual = float(torch.norm(value - old0).item())
    residuals = []
    aux = {}
    for _ in range(4):
        state, aux = cell.write(token, state, mask, visit_meta=_visit("repair"))
        old, value = _read_write_address(cell, token, state)
        residuals.append(float(torch.norm(value - old).item()))
    assert residuals[-1] < initial_residual * 1.0e-3
    assert torch.allclose(old, value, atol=1.0e-4, rtol=1.0e-4)
    assert aux["state_clamp_ratio"] == 0.0
    assert aux["residual_rms_mean"] < initial_residual


def test_gated_delta_kv_residual_delta_replaces_conflicting_value() -> None:
    cell = _constant_key_residual_cell()
    state = cell.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.bool)
    token_a = torch.tensor([[0.6, -0.2]], dtype=torch.float32)
    token_b = torch.tensor([[-0.4, 0.9]], dtype=torch.float32)
    state, _ = cell.write(token_a, state, mask, visit_meta=_visit("repair"))
    old_a, value_a = _read_write_address(cell, token_a, state)
    assert torch.allclose(old_a, value_a, atol=1.0e-4, rtol=1.0e-4)
    state, aux = cell.write(token_b, state, mask, visit_meta=_visit("repair", 1))
    old_b, value_b = _read_write_address(cell, token_b, state)
    dist_to_b = torch.norm(old_b - value_b)
    dist_to_a = torch.norm(old_b - value_a)
    assert float(dist_to_b.item()) < 1.0e-4
    assert float(dist_to_b.item()) < float(dist_to_a.item())
    assert aux["state_clamp_ratio"] == 0.0


def test_gated_delta_kv_v2b_cleanup_fixed_point_stays_bounded() -> None:
    cell = _constant_key_residual_cell(cleanup_enable=True)
    state = cell.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
    token = torch.tensor([[0.75, -0.25]], dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.bool)
    old0, value = _read_write_address(cell, token, state)
    initial_residual = float(torch.norm(value - old0).item())
    residuals = []
    aux = {}
    for repeat in range(4):
        state, aux = cell.write(token, state, mask, visit_meta=_visit("repair", repeat))
        old, value = _read_write_address(cell, token, state)
        residuals.append(float(torch.norm(value - old).item()))
    assert residuals[-1] < initial_residual * 1.0e-3
    assert torch.allclose(old, value, atol=1.0e-4, rtol=1.0e-4)
    assert aux["state_clamp_ratio"] == 0.0
    assert aux["cleanup_mean"] > 0.0
    assert aux["cleanup_old_rms_mean"] > 0.0
    assert aux["cleanup_key_rms_mean"] > 0.0


def test_gated_delta_kv_v2b_cleanup_replaces_stale_value_without_saturation() -> None:
    cell = _constant_key_residual_cell(cleanup_enable=True)
    state = cell.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.bool)
    token_a = torch.tensor([[0.6, -0.2]], dtype=torch.float32)
    token_b = torch.tensor([[-0.4, 0.9]], dtype=torch.float32)
    state, _ = cell.write(token_a, state, mask, visit_meta=_visit("repair"))
    old_a, value_a = _read_write_address(cell, token_a, state)
    assert torch.allclose(old_a, value_a, atol=1.0e-4, rtol=1.0e-4)
    state, aux = cell.write(token_b, state, mask, visit_meta=_visit("repair", 1))
    old_b, value_b = _read_write_address(cell, token_b, state)
    assert torch.norm(old_b - value_b).item() < 1.0e-4
    assert torch.norm(old_b - value_b).item() < torch.norm(old_b - value_a).item()
    assert aux["state_clamp_ratio"] == 0.0
    assert aux["post_state_rms_max"] < 100.0
    assert aux["cleanup_mean"] > 0.0


def test_gated_delta_kv_v2b_old_state_dict_loads_with_new_cleanup_params_missing() -> None:
    new_cell = LowRankGatedDeltaKVCell(
        event_dim=4,
        token_dim=4,
        key_dim=2,
        value_dim=3,
        cleanup_enable=True,
    )
    old_style_state = {
        key: value
        for key, value in new_cell.state_dict().items()
        if "alpha_proj" not in key and "cleanup_key_proj" not in key and "cleanup_proj" not in key
    }
    missing, unexpected = new_cell.load_state_dict(old_style_state, strict=False)
    assert unexpected == []
    assert sorted(missing) == [
        "alpha_proj.bias",
        "alpha_proj.weight",
        "cleanup_key_proj.bias",
        "cleanup_key_proj.weight",
        "cleanup_proj.bias",
        "cleanup_proj.weight",
    ]
    assert torch.isfinite(new_cell.cleanup_key_proj.weight).all()
    assert torch.isfinite(new_cell.cleanup_proj.bias).all()


def test_gated_delta_kv_cell_keeps_state_fp32_from_half_state() -> None:
    torch.manual_seed(40)
    cell = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=5)
    state = cell.init_state(6, device=torch.device("cpu"), dtype=torch.float16)
    event = torch.randn(6, 8)
    token = torch.randn(6, 8)
    mask = torch.ones(6, dtype=torch.bool)
    ctx, read_aux = cell.read(event, state)
    next_state, write_aux = cell.write(token, state, mask, visit_meta=_visit("assimilate"))
    assert ctx.dtype is torch.float32
    assert next_state.kv_state.dtype is torch.float32
    assert read_aux["state_dtype_id"] == 0.0
    assert write_aux["state_dtype_id"] == 0.0


def test_gated_delta_kv_cell_stores_bf16_state_with_fp32_compute() -> None:
    torch.manual_seed(42)
    cell = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=5, state_dtype="bf16")
    state = cell.init_state(6, device=torch.device("cpu"), dtype=torch.bfloat16)
    event = torch.randn(6, 8)
    token = torch.randn(6, 8)
    mask = torch.ones(6, dtype=torch.bool)
    ctx, read_aux = cell.read(event, state)
    next_state, write_aux = cell.write(token, state, mask, visit_meta=_visit("assimilate"))
    assert ctx.dtype is torch.float32
    assert next_state.kv_state.dtype is torch.bfloat16
    assert torch.isfinite(next_state.kv_state.float()).all()
    assert read_aux["state_dtype_id"] == 2.0
    assert write_aux["state_dtype_id"] == 2.0


def test_gated_delta_kv_cell_can_skip_heavy_aux_stats() -> None:
    torch.manual_seed(43)
    cell = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=5, state_dtype="bf16")
    state = cell.init_state(6, device=torch.device("cpu"), dtype=torch.bfloat16)
    event = torch.randn(6, 8)
    token = torch.randn(6, 8)
    mask = torch.ones(6, dtype=torch.bool)
    ctx, read_aux = cell.read(event, state, emit_aux_stats=False)
    next_state, write_aux = cell.write(token, state, mask, visit_meta=_visit("assimilate"), emit_aux_stats=False)
    assert ctx.dtype is torch.float32
    assert next_state.kv_state.dtype is torch.bfloat16
    assert read_aux == {"state_dtype_id": 2.0}
    assert write_aux == {"state_dtype_id": 2.0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast is required")
def test_gated_delta_kv_cell_keeps_state_fp32_under_cuda_autocast() -> None:
    torch.manual_seed(41)
    cell = LowRankGatedDeltaKVCell(event_dim=8, token_dim=8, key_dim=4, value_dim=5).cuda()
    state = cell.init_state(6, device=torch.device("cuda"), dtype=torch.float32)
    event = torch.randn(6, 8, device="cuda")
    token = torch.randn(6, 8, device="cuda")
    mask = torch.ones(6, device="cuda", dtype=torch.bool)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        ctx, _ = cell.read(event, state)
        next_state, _ = cell.write(token, state, mask, visit_meta=_visit("assimilate"))
    assert ctx.dtype is torch.float32
    assert next_state.kv_state.dtype is torch.float32


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
    assert state.bg.dense.kv_state.dtype is torch.float32
    assert int(state.bg.dense.update_count.sum().item()) == 6
    assert state.rigid.keyed is not None
    assert state.rigid.keyed.kv_state.dtype is torch.float32
    assert state.rigid.keyed.keys.detach().cpu().tolist() == [200, 201]


def test_gated_delta_kv_parent_optimizer_stores_bf16_state() -> None:
    mem = ParentOptimizerGatedDeltaKV(
        event_dim=4,
        ctx_dim=3,
        token_dim=4,
        key_dim=2,
        value_dim=3,
        adapter_hidden_dim=8,
        visit_dim=4,
        state_dtype="bf16",
    )
    event = _event()
    state, write_aux = mem.write(
        spatial_event=event,
        state=ParentOptimizerDeltaKVState.empty(),
        keys=_keys(),
        visit_meta=_visit("assimilate"),
    )
    assert write_aux["iforward/parent_optimizer_gdkv/state_dtype_id"] == 2.0
    assert state.bg.dense is not None
    assert state.bg.dense.kv_state.dtype is torch.bfloat16
    assert state.distant.dense is not None
    assert state.distant.dense.kv_state.dtype is torch.bfloat16
    assert state.rigid.keyed is not None
    assert state.rigid.keyed.kv_state.dtype is torch.bfloat16
    assert state.rigid.keyed.keys.dtype is torch.long
    assert state.rigid.keyed.update_count.dtype is torch.long
    preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair"))
    assert preview.aux["iforward/parent_optimizer_gdkv/state_dtype_id"] == 2.0
    assert preview.aux["iforward/parent_optimizer_gdkv/bg_state_dtype_id"] == 2.0
    assert preview.event.event_bg.dtype is torch.float32


def test_gated_delta_kv_parent_optimizer_can_skip_heavy_aux_stats() -> None:
    mem = ParentOptimizerGatedDeltaKV(
        event_dim=4,
        ctx_dim=3,
        token_dim=4,
        key_dim=2,
        value_dim=3,
        adapter_hidden_dim=8,
        visit_dim=4,
        state_dtype="bf16",
    )
    event = _event()
    state, write_aux = mem.write(
        spatial_event=event,
        state=ParentOptimizerDeltaKVState.empty(),
        keys=_keys(),
        visit_meta=_visit("assimilate"),
        emit_aux_stats=False,
    )
    assert write_aux["iforward/parent_optimizer_gdkv/write"] == 1.0
    assert write_aux["iforward/parent_optimizer_gdkv/state_dtype_id"] == 2.0
    assert "iforward/parent_optimizer_gdkv/bg_state_rms_max" not in write_aux
    preview = mem.preview(event=event, state=state, keys=_keys(), visit_meta=_visit("repair"), emit_aux_stats=False)
    assert preview.aux["iforward/parent_optimizer_gdkv/read"] == 1.0
    assert preview.aux["iforward/parent_optimizer_gdkv/state_dtype_id"] == 2.0
    assert "iforward/parent_optimizer_gdkv/bg_preview_seen_ratio" not in preview.aux
    assert "iforward/parent_optimizer_gdkv/bg_ctx_rms_max" not in preview.aux


def test_gdkv_aux_interval_sampling_rule() -> None:
    assert IForwardModel._should_emit_gdkv_aux_stats({}, global_step=7) is True
    assert IForwardModel._should_emit_gdkv_aux_stats({"gdkv_aux_interval": 0}, global_step=0) is False
    assert IForwardModel._should_emit_gdkv_aux_stats({"gdkv_aux_interval": 100}, global_step=0) is True
    assert IForwardModel._should_emit_gdkv_aux_stats({"gdkv_aux_interval": 100}, global_step=99) is False
    assert IForwardModel._should_emit_gdkv_aux_stats({"gdkv_aux_interval": 100}, global_step=100) is True


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
    assert cfg.model.iforward.parent_optimizer_memory.gated_delta_kv.get("update_rule", "gdn2_legacy") == "gdn2_legacy"
    assert cfg.model.iforward.parent_optimizer_memory.gated_delta_kv.get("cleanup_enable", False) is False
    assert cfg.model.iforward.debug.gdkv_aux_interval == 100
    assert cfg.training.amp.storage.features_2d_cache_dtype == "fp32"
    assert cfg.training.amp.storage.parent_context_cache_dtype == "fp32"
    assert cfg.training.amp.memory.gdkv_compute_amp is False
    assert cfg.training.amp.memory.gdkv_state_dtype == "fp32"
    assert cfg.training.amp.stage3.child_detail_output_dtype == "fp32"
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
    assert IForwardModel._normalize_ablation_name(model, "shuffle_rw_state") == "mamba_shuffle_read_write_state"
    assert IForwardModel._normalize_ablation_name(model, "wrong_parent_key_fixed") == "mamba_wrong_parent_key_fixed"


def test_stage3_2_distributional_config_keeps_gdkv_model_and_enables_scheduler() -> None:
    cfg = OmegaConf.load("configs/iforward/iforward_stage3_2_distributional_episode_gdkv.yaml")
    assert cfg.output_name == "iforward_stage3_2_distributional_episode_gdkv"
    assert cfg.scheduler_stage3_2.enable is True
    assert cfg.scheduler_stage3_2.version == "stage3_2_distributional_episode_v1"
    assert cfg.scheduler_stage3_2.inherit_from == "scheduler_stage3_0"
    assert cfg.scheduler_stage3_0.version == "stage3_0_optimizer_sequence_v1"
    assert cfg.model.stage6_0.local_rollout.source == "scheduler_stage3_2"
    assert cfg.model.iforward.version == "stage3_1_lowrank_gated_delta_kv_lift"
    assert cfg.model.iforward.parent_optimizer_memory.type == "lowrank_gated_delta_kv"
    gdkv_cfg = cfg.model.iforward.parent_optimizer_memory.gated_delta_kv
    assert gdkv_cfg.update_rule == "balanced_residual_delta_v1"
    assert gdkv_cfg.alpha_mode == "value_channel"
    assert gdkv_cfg.alpha_max == 1.0
    assert gdkv_cfg.alpha_init == 0.10
    assert gdkv_cfg.surprise_gating is True
    assert gdkv_cfg.surprise_target_rms == 1.0
    assert gdkv_cfg.min_alpha_on_unseen == 0.5
    assert gdkv_cfg.cleanup_enable is True
    assert gdkv_cfg.cleanup_key == "learned"
    assert gdkv_cfg.cleanup_max == 0.2
    assert gdkv_cfg.cleanup_init == 0.02
    assert gdkv_cfg.cleanup_by_kind.bootstrap == 0.0
    assert gdkv_cfg.cleanup_by_kind.assimilate == 0.05
    assert gdkv_cfg.cleanup_by_kind.assimilation == 0.05
    assert gdkv_cfg.cleanup_by_kind.repeat_stability == 0.05
    assert gdkv_cfg.cleanup_by_kind.repair == 0.10
    assert gdkv_cfg.cleanup_by_kind.stress == 0.10
    assert gdkv_cfg.decay_min.repair == 0.995
    assert gdkv_cfg.decay_min.repeat_stability == 0.995
    assert cfg.model.iforward.debug.gdkv_aux_interval == 100
    assert cfg.model.iforward.debug.forward_memory_aux_interval == 1000
    assert cfg.model.iforward.repair_training.stage3_2_train_2d_policy_override is True
    assert cfg.model.stage6_0.render_loss_target_chunk_size == 12
    assert cfg.training.amp.storage.features_2d_cache_dtype == "fp32"
    assert cfg.training.amp.storage.parent_context_cache_dtype == "fp32"
    assert cfg.training.amp.memory.gdkv_compute_amp is False
    assert cfg.training.amp.memory.gdkv_state_dtype == "fp32"
    assert cfg.training.amp.stage3.child_detail_output_dtype == "fp32"
    assert list(cfg.model.iforward.repair_training.kinds) == ["repair"]
    assert cfg.scheduler_stage3_2.episode_recipe.train_2d_policy.high_block_repair == "frozen_no_grad"
    assert cfg.scheduler_stage3_0.repair.last_update_write is True
    assert cfg.scheduler_stage3_2.distributions.high_block_repair.last_update_write is True
    assert cfg.scheduler_stage3_2.curriculum[1].weights.repeat_refine == 0.30
    assert cfg.scheduler_stage3_2.curriculum[1].weights.shuffled_coverage == 0.50
    assert cfg.scheduler_stage3_2.curriculum[1].weights.high_block_repair == 0.20
    assert cfg.scheduler_stage3_2.curriculum[2].weights.repeat_refine == 0.22
    assert cfg.scheduler_stage3_2.curriculum[2].weights.shuffled_coverage == 0.56
    assert cfg.scheduler_stage3_2.curriculum[2].weights.high_block_repair == 0.22
    assert cfg.scheduler_stage3_2.curriculum[2].sequence_target_frames == 20
    assert cfg.scheduler_stage3_2.curriculum[2].max_k.train_2d.shuffled_coverage == 8
    assert cfg.scheduler_stage3_2.curriculum[2].max_k.frozen_2d.high_block_repair == 12
    assert cfg.optimizer.lr.parent_temporal_adapter == 2.0e-4
    assert cfg.optimizer.lr.parent_temporal_mamba == 1.5e-4
    assert cfg.optimizer.lr.parent_ptv3 == 1.5e-4
    assert cfg.optimizer.lr.parent_token_builder == 1.5e-4
    assert cfg.training.grad_clip.max_norm == 1.5
    assert cfg.scheduler_stage3_0_validation.enable is False
    assert cfg.iforward_validation_v4.enable is True
    assert cfg.iforward_validation_v4.interval_steps == 10000
    assert cfg.iforward_validation_v4.max_entries_debug == 1
    assert cfg.iforward_validation_v4.frame_sets[1].name == "seq20"
    assert cfg.iforward_validation_v4.frame_sets[1].target_frames == 20
    assert cfg.iforward_validation_v4.repair_permutations == 3
    assert cfg.iforward_validation_v4.protocols.repeat_stability is False
    assert cfg.iforward_validation_v4.report.image_policy == "first_plan_only"
    assert "memory_shuffle_read_write_state" in list(cfg.iforward_validation_v4.memory_ablation)
    assert "memory_freeze_after_prefill" in list(cfg.iforward_validation_v4.memory_ablation)
    assert "memory_wrong_parent_key_fixed" in list(cfg.iforward_validation_v4.memory_ablation)
    assert cfg.iforward_demo.default_recipe == "repair_showcase_20"
