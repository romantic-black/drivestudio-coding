from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.iforward.stage3_0 import (
    GatherConfig,
    ParentContextFusion,
    ParentQueryBuilder,
    SparseGatherLift,
    build_projected_meta_anchor_stats,
    center_child_detail_by_parent,
    support_center_sparse_gather,
)
from models.iforward.stage3_0.sparse_grid_sample import (
    chunked_sparse_grid_sample,
    normalize_uv_for_grid_sample,
    prepare_value_nchw,
    sparse_grid_sample_prepared,
)
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0 import LocalGSState


def _quat(n: int, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    out = torch.zeros((int(n), 4), device=device)
    out[:, 0] = 1.0
    return out


def _node_bg(n: int, *, device: torch.device = torch.device("cpu"), sh_bases: int = 2) -> NodeStateBackground:
    means = torch.arange(int(n), device=device, dtype=torch.float32).reshape(-1, 1).repeat(1, 3) * 0.1
    return NodeStateBackground(
        means=means,
        scales_log=torch.full((n, 3), -2.0, device=device),
        quats=_quat(n, device=device),
        opacity_logit=torch.full((n, 1), -1.5, device=device),
        sh_dc=torch.linspace(0.0, 1.0, max(n * 3, 1), device=device).reshape(n, 3),
        sh_rest=torch.zeros((n, int(sh_bases), 3), device=device),
    )


def test_stage3_config_reads_top_level_lifting_and_forbids_legacy_fwhr() -> None:
    cfg = OmegaConf.load("configs/iforward/iforward_stage3_0_full_sparse_gather_lift.yaml")
    assert cfg.model.iforward.version == "stage3_0_full_sparse_gather_lift"
    assert cfg.model.iforward.repair_training.enable is True
    assert list(cfg.model.iforward.repair_training.kinds) == ["repair"]
    assert cfg.model.iforward.repair_training.freeze_2d_frontend is True
    assert cfg.model.iforward.repair_training.no_grad_2d_forward is True
    assert cfg.model.iforward.lifting.type == "full_sparse_gather"
    assert cfg.model.iforward.lifting.scalar_anchor_backend == "cuda_scalar_anchor"
    assert cfg.model.iforward.lifting.scalar_anchor.anchor_mode == "auto"
    assert cfg.model.iforward.lifting.scalar_anchor.count_pairs is False
    assert cfg.model.iforward.lifting.gather_aux_interval == 100
    assert cfg.model.iforward.lifting.memory_aux_interval == 100
    assert cfg.model.iforward.debug.forward_memory_aux_interval == 100
    assert cfg.model.iforward.lifting.return_stage3_debug_tensors is False
    assert cfg.model.feature_extractor.dino.cache.level == "backbone_intermediate"
    assert cfg.model.iforward.lifting.parent.type == "legacy_direct_lift"
    assert cfg.model.iforward.lifting.parent_query.obs2d_lift_dim == 16
    assert cfg.model.iforward.lifting.parent_query.use_obs2d_lift is False
    assert cfg.model.iforward.lifting.parent_query.use_dino_native_lift is False
    assert cfg.model.iforward.lifting.parent_context.use_dino_native_fusion is False
    assert cfg.model.iforward.lifting.dino_native.enable is False
    assert cfg.model.iforward.lifting.dino_native.out_channels == 16
    assert cfg.model.iforward.lifting.parent_gather.fixed_center_chunk_size == 65536
    assert cfg.model.iforward.lifting.parent_gather.num_taps == 5
    assert cfg.model.iforward.lifting.child_gather.type == "support_center"
    assert cfg.model.iforward.lifting.child_gather.num_taps == 1
    assert cfg.model.iforward.lifting.child_gather.fixed_center_chunk_size == 65536
    assert cfg.model.iforward.lifting.child_gather.detach_child_detail is False
    assert cfg.model.iforward.lifting.child_gather.train_child_detail_every_n == 1
    assert cfg.scheduler_v3.assimilation.max_inner_k == 8
    assert dict(cfg.scheduler_v3.assimilation.repeat_pairs) == {"4,4": 1.0}
    assert cfg.scheduler_v3.repair.max_inner_k == 16
    assert "lifting" not in cfg.model.iforward.biggs

    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    valid = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    valid.scheduler_v9.enable = True
    runtime._validate_stage6_0_phase_a_config(valid)

    bad = OmegaConf.create(OmegaConf.to_container(valid, resolve=False))
    bad.model.iforward.biggs.lifting = {"type": "fwhr"}
    with pytest.raises(ValueError, match="forbids legacy"):
        runtime._validate_stage6_0_phase_a_config(bad)

    projected_meta = OmegaConf.create(OmegaConf.to_container(valid, resolve=False))
    projected_meta.model.iforward.lifting.scalar_anchor_backend = "projected_meta"
    runtime._validate_stage6_0_phase_a_config(projected_meta)

    bad_backend = OmegaConf.create(OmegaConf.to_container(valid, resolve=False))
    bad_backend.model.iforward.lifting.scalar_anchor_backend = "bad_backend"
    with pytest.raises(ValueError, match="unsupported"):
        runtime._validate_stage6_0_phase_a_config(bad_backend)


def test_stage3_full_train_config_uses_30k_assimilation_30k_repair_schedule() -> None:
    cfg = OmegaConf.load("configs/iforward/iforward_stage3_0_full_train_30k_assim_30k_repair.yaml")
    assert cfg.output_name == "iforward_stage3_0_full_train_30k_assim_30k_repair"
    for legacy_key in [
        "scheduler_v3",
        "scheduler_stage2_2",
        "scheduler_v9",
        "validation_v8",
        "validation_v9",
        "iforward_validation",
        "iforward_coverage_validation",
        "iforward_sequence10_validation",
        "iforward_stage2_2_validation",
        "validation_v3",
    ]:
        assert legacy_key not in cfg
    assert cfg.scheduler_stage3_0.version == "stage3_0_optimizer_sequence_v1"
    assert cfg.scheduler_stage3_0.bootstrap.end_step == 0
    assert cfg.scheduler_stage3_0.assimilation.start_step == 0
    assert "frames_per_rollout" not in cfg.scheduler_stage3_0.assimilation
    assert cfg.scheduler_stage3_0.assimilation.max_inner_k == 8
    assert dict(cfg.scheduler_stage3_0.assimilation.rollout_options) == {
        "B1R8": 0.1,
        "B1R6": 0.1,
        "B1R4": 0.1,
        "B1R2": 0.1,
        "B2R4": 0.2,
    }
    assert dict(cfg.scheduler_stage3_0.assimilation.repeat_pairs) == {"2,6": 0.1, "6,2": 0.1, "3,5": 0.1, "5,3": 0.1}
    assert cfg.scheduler_stage3_0.repair.start_step == 30000
    assert cfg.scheduler_stage3_0.repair.max_inner_k == 16
    assert "rounds" not in cfg.scheduler_stage3_0.repair
    assert dict(cfg.scheduler_stage3_0.repair.round_distribution) == {2: 0.25, 4: 0.25, 6: 0.25, 8: 0.25}
    assert dict(cfg.scheduler_stage3_0.repair.rollout_options) == {
        "B6R1": 0.15,
        "B4R2": 0.15,
        "B8R1": 0.15,
        "B4R3": 0.15,
        "B6R2": 0.1,
        "B12R1": 0.1,
        "B8R2": 0.1,
        "B16R1": 0.1,
    }
    schedule = list(cfg.scheduler_stage3_0.sequence.frame_count_schedule)
    assert len(schedule) == 2
    assert dict(schedule[0]) == {"start_step": 0, "target_frames": 10, "min_frames": 10, "allow_short": False}
    assert dict(schedule[1]) == {"start_step": 30000, "target_frames": 24, "min_frames": 8, "allow_short": True}
    assert cfg.scheduler_stage3_0_validation.enable is True
    assert cfg.model.iforward.version == "stage3_0_scalar_anchor_child_support_parent_legacy"
    assert cfg.model.iforward.repair_training.start_step == 30000
    assert list(cfg.model.iforward.repair_training.kinds) == ["repair"]
    assert cfg.model.stage6_0.local_rollout.source == "scheduler_stage3_0"
    assert cfg.model.iforward.lifting.parent.type == "legacy_direct_lift"
    assert cfg.model.iforward.lifting.child_gather.type == "support_center"
    distant_scope = cfg.model.stage6_0.posterior_updater.branch_scope.distant
    assert distant_scope.update_means is True
    assert distant_scope.update_scales is True
    assert distant_scope.update_quat is False
    distant_detail_gates = cfg.model.stage6_0.posterior_updater.appearance_detail.attribute_gates.distant
    assert distant_detail_gates.means > 0.0
    assert distant_detail_gates.scales > 0.0
    distant_clamps = cfg.model.stage6_0.posterior_updater.branch_clamps.distant
    assert distant_clamps.means_max_step_m > 0.0
    assert distant_clamps.scales_log_max_step > 0.0
    assert distant_clamps.quat_axis_angle_max_step_rad == 0.0
    assert "parent_query" not in cfg.model.iforward.lifting
    assert "parent_context" not in cfg.model.iforward.lifting
    assert "dino_native" not in cfg.model.iforward.lifting
    assert "parent_gather" not in cfg.model.iforward.lifting
    assert "regularization" not in cfg.model.iforward.lifting


def test_stage3_repair_training_start_step_and_kind_gate() -> None:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.iforward_repair_training_cfg = {
        "enable": True,
        "start_step": 30000,
        "kinds": ["assimilate", "repair"],
        "freeze_2d_frontend": True,
    }
    assert runtime._repair_training_enabled_for_visit({"global_step": 29999, "visit_kind": "assimilate"}) is False
    assert runtime._repair_training_enabled_for_visit({"global_step": 30000, "visit_kind": "assimilate"}) is True
    assert runtime._repair_training_enabled_for_visit({"global_step": 30000, "visit_kind": "repair"}) is True
    assert runtime._repair_training_enabled_for_visit({"global_step": 30000, "visit_kind": "bootstrap"}) is False


def test_projected_meta_anchor_filters_masks_aggregates_and_detaches() -> None:
    means2d = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [4.5, 1.0],
            [3.0, 0.0],
            [0.0, 0.0],
        ],
        requires_grad=True,
    )
    meta = {
        "means2d": means2d,
        "gaussian_ids": torch.tensor([0, 1, 1, 2, 2], dtype=torch.long),
        "camera_ids": torch.tensor([0, 0, 1, 1, 0], dtype=torch.long),
        "opacities": torch.tensor([0.2, 0.5, 0.7, 0.6, 0.1]),
        "depths": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        "radii": torch.tensor([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [0.0, 0.0]]),
        "conics": torch.ones((5, 3)),
    }
    pair_mask = torch.ones((2, 4, 4), dtype=torch.bool)
    pair_mask[0, 2, 2] = False
    child_to_parent = torch.tensor([0, 0, 1], dtype=torch.long)

    anchor = build_projected_meta_anchor_stats(
        meta=meta,
        child_to_parent=child_to_parent,
        num_children=3,
        num_parents=2,
        num_views=2,
        image_height=4,
        image_width=4,
        source_pair_valid_mask=pair_mask,
        detach_geometry=True,
    )
    assert tuple(anchor.child_uv.shape) == (3, 2, 2)
    assert torch.allclose(anchor.child_support[0], torch.tensor([0.2, 0.0]))
    assert torch.allclose(anchor.child_support[1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(anchor.child_support[2], torch.tensor([0.0, 0.6]))
    assert torch.allclose(anchor.parent_support_total, torch.tensor([0.2, 0.6]))
    assert bool(anchor.parent_valid[0, 0].item())
    assert bool(anchor.parent_valid[1, 1].item())
    assert not anchor.child_uv.requires_grad

    anchor_grad = build_projected_meta_anchor_stats(
        meta=meta,
        child_to_parent=child_to_parent,
        num_children=3,
        num_parents=2,
        num_views=2,
        image_height=4,
        image_width=4,
        source_pair_valid_mask=pair_mask,
        detach_geometry=False,
    )
    assert anchor_grad.child_uv.requires_grad


def test_chunked_sparse_grid_sample_matches_grid_sample_and_backprops() -> None:
    value_map = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3, 1).requires_grad_()
    uv = torch.tensor([[[[1.0, 1.0]]], [[[0.0, 2.0]]]])
    sampled, inbound = chunked_sparse_grid_sample(value_map, uv, image_height=3, image_width=3, chunk_size=1)
    prepared = prepare_value_nchw(value_map)
    sampled_prepared, inbound_prepared = sparse_grid_sample_prepared(
        prepared,
        uv,
        image_height=3,
        image_width=3,
        chunk_size=1,
    )
    grid = normalize_uv_for_grid_sample(uv, image_height=3, image_width=3).permute(1, 0, 2, 3).reshape(1, 2, 1, 2)
    expected = F.grid_sample(
        value_map.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(1, 1, 2, 1).permute(2, 0, 3, 1)
    assert torch.allclose(sampled, expected)
    assert torch.allclose(sampled_prepared, expected)
    assert bool(inbound.all().item())
    assert torch.equal(inbound, inbound_prepared)
    sampled.sum().backward()
    assert value_map.grad is not None
    assert float(value_map.grad.abs().sum().item()) > 0.0


class _RaiseIfCalled(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("module should not be called in fixed-center fast path")


class _CountingChildQuery(nn.Module):
    def __init__(self, query_dim: int) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.calls: list[int] = []

    def forward(self, *, child_params, **_kwargs):
        n = int(child_params["means"].shape[0])
        self.calls.append(n)
        return child_params["means"].new_zeros((n, int(self.query_dim)))


def test_sparse_gather_center_tap_matches_sampler_and_chunk_sizes() -> None:
    value_map = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3, 1)
    anchor_uv = torch.tensor([[[1.0, 1.0]], [[0.0, 2.0]]])
    support = torch.ones((2, 1), dtype=torch.float32)
    valid = torch.ones((2, 1), dtype=torch.bool)
    depth = torch.ones((2, 1), dtype=torch.float32)
    radius = torch.ones((2, 1), dtype=torch.float32)
    query = torch.zeros((2, 4), dtype=torch.float32)
    cfg = GatherConfig(
        query_dim=4,
        num_taps=5,
        chunk_size=1,
        use_geometry_pe=False,
        fixed_center_steps=10,
        train_weights_steps=20,
        offset_warmup_steps=30,
    )
    gather = SparseGatherLift(value_dim=1, config=cfg)
    out, conf, _aux, _reg = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        query=query,
        image_height=3,
        image_width=3,
        global_step=0,
    )
    expected, _ = chunked_sparse_grid_sample(
        value_map,
        anchor_uv[:, :, None, :],
        image_height=3,
        image_width=3,
        chunk_size=16,
    )
    assert torch.allclose(out, expected[:, 0, 0, :], atol=1.0e-6)
    assert torch.allclose(conf, torch.ones_like(conf))

    gather_large_chunk = SparseGatherLift(value_dim=1, config=GatherConfig(**{**cfg.__dict__, "chunk_size": 16}))
    out_large, _conf_large, _aux_large, _reg_large = gather_large_chunk(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        query=query,
        image_height=3,
        image_width=3,
        global_step=0,
    )
    assert torch.allclose(out, out_large, atol=1.0e-6)


def test_sparse_gather_fixed_center_fast_path_skips_query_and_head() -> None:
    value_map = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3, 1)
    anchor_uv = torch.tensor([[[1.0, 1.0]], [[0.0, 2.0]], [[1.0, 1.0]]])
    support = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)
    valid = torch.tensor([[True], [True], [False]])
    depth = torch.ones((3, 1), dtype=torch.float32)
    radius = torch.ones((3, 1), dtype=torch.float32)
    cfg = GatherConfig(
        query_dim=4,
        num_taps=5,
        chunk_size=2,
        use_geometry_pe=False,
        fixed_center_steps=10,
        fixed_center_fast_path=True,
    )
    gather = SparseGatherLift(value_dim=1, config=cfg)
    gather.head = _RaiseIfCalled()
    gather.view_logit_head = _RaiseIfCalled()
    gather.tap_logit_head = _RaiseIfCalled()
    gather.offset_head = _RaiseIfCalled()
    gather.gate_head = _RaiseIfCalled()
    out, conf, aux, reg = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        image_height=3,
        image_width=3,
        query=None,
        global_step=0,
    )
    expected, _ = chunked_sparse_grid_sample(
        value_map,
        anchor_uv[:, :, None, :],
        image_height=3,
        image_width=3,
        chunk_size=16,
    )
    assert torch.allclose(out[:2], expected[:2, 0, 0, :], atol=1.0e-6)
    assert torch.allclose(out[2], torch.zeros_like(out[2]))
    assert torch.allclose(conf[:2], torch.ones_like(conf[:2]))
    assert float(conf[2].item()) == pytest.approx(0.0)
    assert aux["iforward/stage3/gather_fixed_fast_path_enabled"] == 1.0
    assert torch.allclose(reg["offset_l2"], torch.zeros_like(reg["offset_l2"]))

    with pytest.raises(ValueError, match="query is required"):
        gather(
            value_map=value_map,
            anchor_uv=anchor_uv,
            support=support,
            valid=valid,
            depth=depth,
            radius=radius,
            image_height=3,
            image_width=3,
            query=None,
            global_step=11,
        )


def test_support_center_gather_matches_sparse_gather_fixed_center() -> None:
    value_map = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    anchor_uv = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]], [[0.0, 0.0]]])
    support = torch.tensor([[2.0], [1.0], [0.0]], dtype=torch.float32)
    valid = torch.tensor([[True], [True], [False]])
    depth = torch.ones((3, 1), dtype=torch.float32)
    radius = torch.ones((3, 1), dtype=torch.float32)
    cfg = GatherConfig(query_dim=4, num_taps=5, use_geometry_pe=False, fixed_center_steps=10, backend="pytorch")
    gather = SparseGatherLift(value_dim=1, config=cfg)
    out_ref, conf_ref, _aux_ref, _reg_ref = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        image_height=4,
        image_width=4,
        query=None,
        global_step=0,
    )
    out, conf, aux = support_center_sparse_gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        image_height=4,
        image_width=4,
        backend="pytorch",
        chunk_size=2,
    )
    assert torch.allclose(out, out_ref, atol=1.0e-6)
    assert torch.allclose(conf, conf_ref, atol=1.0e-6)
    assert aux["iforward/stage3/support_center_fixed_support_center_enabled"] == 1.0


def test_fixed_center_uses_dedicated_chunk_size() -> None:
    value_map = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4, 1)
    anchor_uv = torch.stack([torch.tensor([[float(i % 4), float(i % 5)]]) for i in range(5)], dim=0)
    support = torch.ones((5, 1), dtype=torch.float32)
    valid = torch.ones((5, 1), dtype=torch.bool)
    depth = torch.ones((5, 1), dtype=torch.float32)
    radius = torch.ones((5, 1), dtype=torch.float32)
    gather = SparseGatherLift(
        value_dim=1,
        config=GatherConfig(
            query_dim=4,
            num_taps=5,
            chunk_size=1,
            fixed_center_chunk_size=2,
            use_geometry_pe=False,
            fixed_center_steps=10,
            fixed_center_fast_path=True,
            backend="pytorch",
        ),
    )
    calls: list[int] = []
    orig = gather._sample_weighted_sum

    def _count_rows(**kwargs):
        calls.append(int(kwargs["sample_uv"].shape[0]))
        return orig(**kwargs)

    gather._sample_weighted_sum = _count_rows  # type: ignore[method-assign]
    out, _conf, _aux, _reg = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        image_height=5,
        image_width=4,
        query=None,
        global_step=0,
    )
    assert tuple(out.shape) == (5, 1)
    assert calls == [2, 2, 1]
    assert gather.effective_chunk_size(11, rows=5) == 1


def test_sparse_gather_heavy_aux_can_be_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    value_map = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3, 1)
    anchor_uv = torch.tensor([[[1.0, 1.0]], [[0.0, 2.0]]])
    support = torch.ones((2, 1), dtype=torch.float32)
    valid = torch.ones((2, 1), dtype=torch.bool)
    depth = torch.ones((2, 1), dtype=torch.float32)
    radius = torch.ones((2, 1), dtype=torch.float32)
    gather = SparseGatherLift(
        value_dim=1,
        config=GatherConfig(query_dim=4, num_taps=5, use_geometry_pe=False, fixed_center_steps=10, backend="pytorch"),
    )

    def _raise_entropy(*_args, **_kwargs):
        raise AssertionError("heavy aux entropy should not be computed")

    monkeypatch.setattr(gather, "_entropy", _raise_entropy)
    _out, _conf, aux, _reg = gather(
        value_map=value_map,
        anchor_uv=anchor_uv,
        support=support,
        valid=valid,
        depth=depth,
        radius=radius,
        image_height=3,
        image_width=3,
        query=None,
        global_step=0,
        emit_heavy_aux=False,
    )
    assert aux["iforward/stage3/gather_heavy_aux_enabled"] == 0.0
    assert "iforward/stage3/gather_view_entropy" not in aux
    assert aux["iforward/stage3/gather_fixed_fast_path_enabled"] == 1.0


def test_sparse_gather_zero_init_center_bias_and_schedule() -> None:
    cfg = GatherConfig(
        query_dim=4,
        num_taps=5,
        center_tap_bias=3.0,
        fixed_center_steps=5,
        train_weights_steps=10,
        offset_warmup_steps=20,
        offset_scale_start=0.25,
    )
    gather = SparseGatherLift(value_dim=2, config=cfg)
    assert torch.allclose(gather.offset_head.weight, torch.zeros_like(gather.offset_head.weight))
    assert torch.allclose(gather.offset_head.bias, torch.zeros_like(gather.offset_head.bias))
    assert torch.allclose(gather.tap_logit_head.weight, torch.zeros_like(gather.tap_logit_head.weight))
    assert float(gather.tap_logit_head.bias[gather.center_tap].item()) == pytest.approx(3.0)
    assert gather._offset_scale_factor(0) == (True, 0.0)
    assert gather._offset_scale_factor(7) == (False, 0.0)
    assert gather._offset_scale_factor(15) == (False, pytest.approx(0.625))
    assert gather._offset_scale_factor(20) == (False, 1.0)


def test_parent_query_accepts_obs2d_and_dino_lifts_and_context_fusion_preserves_dim() -> None:
    params = {
        "means": torch.zeros((3, 3), dtype=torch.float32),
        "scales_log": torch.zeros((3, 3), dtype=torch.float32),
        "quats": _quat(3),
        "opacity_logit": torch.zeros((3, 1), dtype=torch.float32),
    }
    query = ParentQueryBuilder(query_dim=12, extra_input_dim=32)
    out = query(
        params=params,
        support_total=torch.ones((3,), dtype=torch.float32),
        branch_id=0,
        optimizer_prior=torch.zeros((3, 4), dtype=torch.float32),
        obs2d_lift=torch.ones((3, 16), dtype=torch.float32),
        dino_lift=torch.full((3, 16), 2.0, dtype=torch.float32),
    )
    assert tuple(out.shape) == (3, 12)
    with pytest.raises(ValueError, match="extra dim mismatch"):
        query(
            params=params,
            support_total=torch.ones((3,), dtype=torch.float32),
            branch_id=0,
            optimizer_prior=torch.zeros((3, 4), dtype=torch.float32),
            obs2d_lift=torch.ones((3, 16), dtype=torch.float32),
            dino_lift=None,
        )
    fusion = ParentContextFusion(context_dim=48, dino_dim=16, hidden_dim=8)
    context = torch.randn((3, 48), dtype=torch.float32)
    dino = torch.randn((3, 16), dtype=torch.float32)
    fused = fusion(context, dino)
    assert tuple(fused.shape) == (3, 48)
    assert torch.allclose(fused, context, atol=1.0e-6)


def test_child_detail_weighted_parent_centering_is_zero_mean() -> None:
    detail = torch.tensor([[1.0, 0.0], [3.0, 0.0], [10.0, 2.0]])
    child_to_parent = torch.tensor([0, 0, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 3.0, 2.0])
    centered, err = center_child_detail_by_parent(
        detail,
        child_to_parent=child_to_parent,
        weights=weights,
        num_parents=2,
    )
    assert float(err.item()) == pytest.approx(0.0, abs=1.0e-6)
    for parent in (0, 1):
        rows = child_to_parent == parent
        mean = (centered[rows] * weights[rows, None]).sum(dim=0) / weights[rows].sum()
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1.0e-6)


def test_stage3_child_support_center_is_observe_ready_and_skips_query() -> None:
    device = torch.device("cpu")
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.stage3_0_enabled = True
    runtime.stage3_0_global_step = 1
    runtime.stage3_0_lifting_cfg = {
        "child_gather": {
            "type": "support_center",
            "valid_row_filter": True,
            "center_by_parent": True,
            "backend": "pytorch",
            "fixed_center_chunk_size": 2,
        },
        "scalar_anchor": {"support_threshold": {"child": 1.0e-4}},
    }
    runtime.stage3_child_query = _RaiseIfCalled()
    runtime.stage3_child_gather = _RaiseIfCalled()
    num_children = 5
    num_parents = 2
    child_to_parent = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)
    anchor = SimpleNamespace(
        child_support_total=torch.ones((num_children,), device=device),
        child_valid=torch.ones((num_children, 1), dtype=torch.bool, device=device),
        child_uv=torch.tensor([[[1.0, 1.0]], [[1.0, 2.0]], [[2.0, 1.0]], [[2.0, 2.0]], [[1.5, 1.5]]], device=device),
        child_support=torch.ones((num_children, 1), device=device),
        child_depth=torch.ones((num_children, 1), device=device),
        child_radius=torch.ones((num_children, 1), device=device),
    )
    parent_state = _node_bg(num_parents, device=device, sh_bases=3)
    measurement = {
        "stage3_anchor_stats": anchor,
        "stage3_detail_2d": torch.arange(4 * 4 * 2, dtype=torch.float32, device=device)
        .reshape(1, 4, 4, 2)
        .requires_grad_(),
        "stage3_image_height": 4,
        "stage3_image_width": 4,
        "stage3_child_to_parent_global": child_to_parent,
        "num_bg": num_children,
        "num_distant": 0,
        "num_rigid_S": 0,
        "parent_params_bg": runtime._stage3_0_branch_params(parent_state),
        "assign_bg": SimpleNamespace(child_to_parent=child_to_parent, num_parents=num_parents),
        "stage3_gather_reg_terms": {},
    }
    local_state = LocalGSState.from_node_states(bg=_node_bg(num_children, device=device, sh_bases=3), distant=None, rigid=None, hidden_dim=3)
    runtime._stage3_0_gather_child_detail(local_state=local_state, measurement=measurement)
    assert measurement["iforward/stage3/child_bg_num_chunks"] == 3.0
    assert measurement["iforward/stage3/child_support_center_enabled"] == 1.0
    assert measurement["iforward/stage3/child_event_dependency_removed"] == 1.0
    assert measurement["iforward/stage3/child_learned_path_enabled"] == 0.0
    assert measurement["iforward/stage3/child_bg_fixed_support_center_enabled"] == 1.0
    assert bool(measurement["child_detail_valid_bg"].all().item())
    assert tuple(measurement["child_detail_bg"].shape) == (num_children, 2)
    assert measurement["child_detail_bg"].requires_grad


def test_stage3_child_detail_detach_ablation_and_train_interval() -> None:
    device = torch.device("cpu")
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.stage3_0_enabled = True
    runtime.stage3_0_global_step = 3
    runtime.stage3_0_lifting_cfg = {
        "child_gather": {
            "type": "support_center",
            "center_by_parent": False,
            "backend": "pytorch",
            "fixed_center_chunk_size": 8,
            "train_child_detail_every_n": 4,
        },
        "scalar_anchor": {"support_threshold": {"child": 1.0e-4}},
    }
    num_children = 2
    child_to_parent = torch.tensor([0, 0], dtype=torch.long, device=device)
    anchor = SimpleNamespace(
        child_support_total=torch.ones((num_children,), device=device),
        child_valid=torch.ones((num_children, 1), dtype=torch.bool, device=device),
        child_uv=torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]], device=device),
        child_support=torch.ones((num_children, 1), device=device),
    )
    detail_2d = torch.arange(4 * 4 * 2, dtype=torch.float32, device=device).reshape(1, 4, 4, 2).requires_grad_()
    measurement = {
        "stage3_anchor_stats": anchor,
        "stage3_detail_2d": detail_2d,
        "stage3_image_height": 4,
        "stage3_image_width": 4,
        "stage3_child_to_parent_global": child_to_parent,
        "num_bg": num_children,
        "num_distant": 0,
        "num_rigid_S": 0,
        "assign_bg": SimpleNamespace(child_to_parent=child_to_parent, num_parents=1),
        "stage3_gather_reg_terms": {},
    }
    local_state = LocalGSState.from_node_states(bg=_node_bg(num_children, device=device, sh_bases=3), distant=None, rigid=None, hidden_dim=3)
    runtime._stage3_0_gather_child_detail(local_state=local_state, measurement=measurement)
    assert measurement["iforward/stage3/child_detail_detached"] == 1.0
    assert not measurement["child_detail_bg"].requires_grad

    runtime.stage3_0_global_step = 4
    measurement["stage3_detail_2d"] = detail_2d
    runtime._stage3_0_gather_child_detail(local_state=local_state, measurement=measurement)
    assert measurement["iforward/stage3/child_detail_detached"] == 0.0
    assert measurement["child_detail_bg"].requires_grad

    runtime.stage3_0_lifting_cfg["child_gather"]["detach_child_detail"] = True
    runtime._stage3_0_gather_child_detail(local_state=local_state, measurement=measurement)
    assert measurement["iforward/stage3/child_detail_detached"] == 1.0
    assert not measurement["child_detail_bg"].requires_grad


def _run_stage3_observe_smoke(
    device: torch.device,
    *,
    return_debug_tensors: bool = False,
    visit_meta: dict[str, object] | None = None,
    repair_training_cfg: dict[str, object] | None = None,
    source_features_require_grad: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.device = device
    runtime.sh_degree = 1
    runtime.stage2_0_biggs_assignment_cfg = {
        "builder": "vectorized_sort_segment",
        "sort_children": "none",
        "mass_init": "uniform",
        "bg": {"voxel_size": 10.0, "max_children_per_parent": 2, "max_parent_radius": 20.0},
    }
    runtime.stage2_0_biggs_projector_cfg = {"min_scale": 0.01, "max_scale_bg": 1.0}
    runtime.stage2_0_biggs_parent_state_cfg = {"mode": "none"}
    runtime.stage2_0_biggs_observe_cfg = {"parent_scene_for_cnn": False}
    runtime.stage2_0_biggs_lifting_cfg = {}
    runtime.stage2_0_biggs_return_debug_stats = False
    runtime.iforward_repair_training_cfg = repair_training_cfg or {}
    runtime.stage3_0_enabled = True
    runtime.stage3_0_global_step = 0
    runtime.stage3_0_lifting_cfg = {
        "type": "full_sparse_gather",
        "scalar_anchor_backend": "projected_meta",
        "context_dim": 4,
        "detail_dim": 2,
        "detach_geometry": True,
        "return_stage3_debug_tensors": bool(return_debug_tensors),
        "parent": {"type": "legacy_direct_lift"},
        "scalar_anchor": {"support_threshold": {"bg": 1.0e-4}},
    }
    runtime.stage3_parent_lifting_type = "legacy_direct_lift"
    runtime.stage3_parent_query = _RaiseIfCalled().to(device)
    runtime.stage3_parent_gather = _RaiseIfCalled().to(device)
    runtime.stage3_parent_context_fusion = _RaiseIfCalled().to(device)
    runtime._stage5_4_obs_code_all = None
    runtime._mem_debug = lambda *args, **kwargs: None
    runtime._source_subset = lambda batch, indices: (
        [object()],
        [torch.zeros((3, 4, 4), device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
        [torch.zeros((1, 4, 4), dtype=torch.bool, device=device)],
    )
    def _render_source_scene_only_for_cnn(**_kwargs):
        features = torch.arange(4 * 4 * 4, dtype=torch.float32, device=device).reshape(1, 4, 4, 4)
        detail = torch.ones((1, 4, 4, 2), dtype=torch.float32, device=device)
        if bool(source_features_require_grad):
            features = features.detach().clone().requires_grad_(True) * 1.0
            detail = detail.detach().clone().requires_grad_(True) * 1.0
        return {
            "features_2d": features,
            "fwhr_detail_2d": detail,
            "source_pair_valid_mask": torch.ones((1, 4, 4), dtype=torch.bool, device=device),
        }

    runtime._render_source_scene_only_for_cnn = _render_source_scene_only_for_cnn

    def _meta_builder(**kwargs):
        n = int(kwargs["gaussians"]["means"].shape[0])
        uv = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]], device=device)[:n]
        return {
            "means2d": uv,
            "gaussian_ids": torch.arange(n, dtype=torch.long, device=device),
            "camera_ids": torch.zeros((n,), dtype=torch.long, device=device),
            "opacities": torch.ones((n,), dtype=torch.float32, device=device),
            "depths": torch.ones((n,), dtype=torch.float32, device=device),
            "radii": torch.ones((n,), dtype=torch.float32, device=device),
            "conics": torch.ones((n, 3), dtype=torch.float32, device=device),
        }, {"packed_rows": float(n)}

    runtime.alpha_t_extractor_v4 = SimpleNamespace(_build_multi_camera_meta_from_views=_meta_builder)

    backproject_calls = []

    def _legacy_parent_backproject(**kwargs):
        backproject_calls.append(kwargs)
        assert int(kwargs["gaussians_scene"]["means"].shape[0]) == 2
        assert tuple(kwargs["features_2d"].shape) == (1, 4, 4, 4)
        return torch.ones((2, 4), dtype=torch.float32, device=device), torch.ones((2,), dtype=torch.float32, device=device)

    runtime._backproject_scene_features_multi_camera = _legacy_parent_backproject
    local_state = LocalGSState.from_node_states(bg=_node_bg(4, device=device, sh_bases=3), distant=None, rigid=None, hidden_dim=3)
    measurement = runtime._observe_stage2_0_biggs_measurement(
        local_state=local_state,
        batch={"scene_id": 1, "segment_id": 1},
        source_indices=[0],
        source_frame_idx=0,
        biggs_state=None,
        visit_meta=visit_meta or {"global_step": 0},
    )
    assert measurement["biggs_mode"] == "stage3_sparse_gather_event_decode"
    assert measurement["iforward/fwhr/enabled"] == 0.0
    assert measurement["iforward/stage3/enabled"] == 1.0
    assert measurement["iforward/stage3/parent_legacy_direct_lift_enabled"] == 1.0
    assert measurement["iforward/stage3/parent_sparse_gather_enabled"] == 0.0
    assert len(backproject_calls) == 1
    assert tuple(measurement["parent_feat_2d_bg"].shape) == (2, 4)
    assert measurement["parent_obs_bg"] is None
    if bool(return_debug_tensors):
        assert measurement["stage3_anchor_stats"] is not None
        assert measurement["stage3_context_2d"] is not None
        assert measurement["stage3_detail_2d"] is not None
        assert measurement["stage3_child_to_parent_global"] is not None
        assert measurement["stage3_image_height"] == 4
        assert measurement["stage3_image_width"] == 4
    else:
        assert "stage3_anchor_stats" not in measurement
        assert "stage3_context_2d" not in measurement
        assert "stage3_detail_2d" not in measurement
        assert "stage3_child_to_parent_global" not in measurement
        assert "stage3_image_height" not in measurement
        assert "stage3_image_width" not in measurement
    assert measurement["child_detail_bg"] is not None
    assert measurement["child_detail_valid_bg"] is not None
    assert measurement["iforward/stage3/child_event_dependency_removed"] == 1.0
    return measurement, backproject_calls


def test_stage3_observe_uses_legacy_parent_backproject() -> None:
    _run_stage3_observe_smoke(torch.device("cpu"))


def test_stage3_observe_can_return_debug_tensors_when_enabled() -> None:
    _run_stage3_observe_smoke(torch.device("cpu"), return_debug_tensors=True)


def test_stage3_repair_training_detaches_2d_observe_features() -> None:
    normal, normal_calls = _run_stage3_observe_smoke(
        torch.device("cpu"),
        visit_meta={"global_step": 0, "visit_kind": "bootstrap"},
        repair_training_cfg={
            "enable": True,
            "kinds": ["repair"],
            "freeze_2d_frontend": True,
            "no_grad_2d_forward": True,
        },
        source_features_require_grad=True,
    )
    assert normal["iforward/repair_training/enabled"] == 0.0
    assert bool(normal_calls[0]["features_2d"].requires_grad)

    repair, repair_calls = _run_stage3_observe_smoke(
        torch.device("cpu"),
        visit_meta={"global_step": 0, "visit_kind": "repair"},
        repair_training_cfg={
            "enable": True,
            "kinds": ["repair"],
            "freeze_2d_frontend": True,
            "no_grad_2d_forward": True,
        },
        source_features_require_grad=True,
    )
    assert repair["iforward/repair_training/enabled"] == 1.0
    assert repair["iforward/repair_training/freeze_2d_frontend"] == 1.0
    assert repair["iforward/repair_training/no_grad_2d_forward"] == 1.0
    assert repair["iforward/repair_training/features_2d_requires_grad"] == 0.0
    assert repair["iforward/repair_training/detail_2d_requires_grad"] == 0.0
    assert not bool(repair_calls[0]["features_2d"].requires_grad)
    assert repair["parent_feat_2d_bg"].requires_grad is False
    assert repair["child_detail_bg"].requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stage3_observe_cuda_smoke() -> None:
    _run_stage3_observe_smoke(torch.device("cuda"))
