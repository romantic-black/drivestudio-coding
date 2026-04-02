from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.streetforward.minimal_trainer_stage4_1 import MinimalStreetForwardStage4_1
from models.streetforward.minimal_trainer_stage4_0 import MinimalStreetForwardStage4_0
from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid


def _make_bg_state(device: torch.device, n: int = 3) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3, device=device),
        scales_log=torch.zeros(n, 3, device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1, device=device),
        sh_dc=torch.zeros(n, 3, device=device),
        sh_rest=torch.zeros(n, 3, 3, device=device),
    )


def _make_distant_state(device: torch.device, n: int = 2) -> NodeStateDistant:
    return NodeStateDistant(
        means=torch.zeros(n, 3, device=device),
        scales_log=torch.zeros(n, 3, device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1, device=device),
        sh_dc=torch.zeros(n, 3, device=device),
        sh_rest=torch.zeros(n, 3, 3, device=device),
    )


def _make_rigid_state(device: torch.device, n: int = 2) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.zeros(n, 3, device=device),
        scales_log=torch.zeros(n, 3, device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1, device=device),
        sh_dc=torch.zeros(n, 3, device=device),
        sh_rest=torch.zeros(n, 3, 3, device=device),
        point_ids=torch.zeros(n, 1, dtype=torch.long, device=device),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device),
        instances_trans=torch.zeros(1, 1, 3, device=device),
        instances_fv=torch.ones(1, 1, dtype=torch.bool, device=device),
        instance_ids=[0],
        frame_ids=[0],
        cur_frame=0,
    )


def test_init_fast_fail_for_new_bg_distant_keys(monkeypatch):
    monkeypatch.setattr(MinimalStreetForwardStage4_1, "__init__", lambda self, config, device, **kwargs: None)
    cfg_missing = SimpleNamespace(model={"branches": {"bg": {}, "distant": {}, "rigid": {}}})
    with pytest.raises(ValueError):
        MinimalStreetForwardStage4_2(config=cfg_missing, device=torch.device("cpu"))


def test_build_any_target_mask_static_selective_flag():
    trainer = MinimalStreetForwardStage4_2.__new__(MinimalStreetForwardStage4_2)
    dev = torch.device("cpu")
    m0 = trainer._build_any_target_mask_static(4, False, dev)
    m1 = trainer._build_any_target_mask_static(4, True, dev)
    assert m0.tolist() == [True, True, True, True]
    assert m1.tolist() == [True, True, True, True]


def test_one_pass_backprojection_split_ranges():
    trainer = MinimalStreetForwardStage4_2.__new__(MinimalStreetForwardStage4_2)
    trainer.device = torch.device("cpu")
    trainer.feature_backprojector = SimpleNamespace(eps=1e-8)
    trainer._prepare_gaussians_bg_distant = lambda bg, distant: (
        {
            "means": torch.zeros(5, 3),
            "scales": torch.ones(5, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(5, 1),
            "opacities": torch.ones(5),
            "colors": torch.zeros(5, 4, 3),
        },
        3,
        2,
    )
    trainer._transform_rigid_to_world = lambda node_state, means, frame_idx, point_ids_subset=None: means
    trainer._transform_rigid_quats_to_world = lambda node_state, quats, frame_idx, point_ids_subset=None: quats

    def _fake_backproject(**kwargs):
        n = kwargs["gaussians"]["means"].shape[0]
        feat = torch.arange(n * 2, dtype=torch.float32).view(n, 2)
        acc = torch.arange(n, dtype=torch.float32)
        return feat, acc

    trainer._compute_2d_features_for_gaussians = _fake_backproject
    bg = _make_bg_state(torch.device("cpu"), n=3)
    distant = _make_distant_state(torch.device("cpu"), n=2)
    rigid = _make_rigid_state(torch.device("cpu"), n=2)
    out = trainer._compute_2d_features_all_branches_once(
        node_state_bg=bg,
        node_state_distant=distant,
        node_state_rigid=rigid,
        source_frame_idx=0,
        rigid_idx_S=torch.tensor([0, 1], dtype=torch.long),
        source_views=[object()],
        source_images=[torch.zeros(4, 4, 3)],
        height=4,
        width=4,
    )
    assert out["feat_2d_bg"].shape[0] == 3
    assert out["feat_2d_distant"].shape[0] == 2
    assert out["feat_2d_rigid_S"].shape[0] == 2
    assert out["acc_w_bg"].shape[0] == 3
    assert out["acc_w_distant"].shape[0] == 2
    assert out["acc_w_rigid_S"].shape[0] == 2


def test_update_node_state_bg_subset_only_selected():
    trainer = MinimalStreetForwardStage4_2.__new__(MinimalStreetForwardStage4_2)
    trainer.bbx_min = torch.tensor([-1.0, -1.0, -1.0])
    trainer.bbx_max = torch.tensor([1.0, 1.0, 1.0])
    bg = _make_bg_state(torch.device("cpu"), n=3)
    rp = {
        "means_r": torch.tensor([[0.5, 0.0, 0.0], [0.8, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        "scales_log_r": torch.ones(3, 3),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        "opacity_logit_r": torch.ones(3, 1),
        "sh_dc_r": torch.ones(3, 3),
        "sh_rest_r": torch.ones(3, 3, 3),
    }
    trainer._update_node_state_bg_subset(bg, rp, torch.tensor([1], dtype=torch.long))
    assert torch.allclose(bg.means[0], torch.zeros(3))
    assert torch.allclose(bg.means[1], torch.tensor([0.8, 0.0, 0.0]))
    assert torch.allclose(bg.means[2], torch.zeros(3))


def test_update_node_state_distant_subset_writes_means_without_segment_aabb_clamp():
    trainer = MinimalStreetForwardStage4_2.__new__(MinimalStreetForwardStage4_2)
    trainer.input_aabb_min = torch.tensor([-1.0, -1.0, -1.0])
    trainer.input_aabb_max = torch.tensor([1.0, 1.0, 1.0])
    distant = _make_distant_state(torch.device("cpu"), n=2)
    rp = {
        "means_r": torch.tensor([[5.0, 0.0, 0.0], [0.5, 2.0, 0.0]]),
        "scales_log_r": torch.ones(2, 3),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "opacity_logit_r": torch.ones(2, 1),
        "sh_dc_r": torch.ones(2, 3),
        "sh_rest_r": torch.ones(2, 3, 3),
    }
    trainer._update_node_state_distant_subset(distant, rp, torch.tensor([0, 1], dtype=torch.long))
    assert torch.allclose(distant.means[0], torch.tensor([5.0, 0.0, 0.0]))
    assert torch.allclose(distant.means[1], torch.tensor([0.5, 2.0, 0.0]))


def test_one_pass_backprojection_reports_real_count():
    trainer = MinimalStreetForwardStage4_2.__new__(MinimalStreetForwardStage4_2)
    trainer.device = torch.device("cpu")
    trainer.feature_backprojector = SimpleNamespace(eps=1e-8)
    trainer._prepare_gaussians_bg_distant = lambda bg, distant: (
        {
            "means": torch.zeros(1, 3),
            "scales": torch.ones(1, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "opacities": torch.ones(1),
            "colors": torch.zeros(1, 4, 3),
        },
        1,
        0,
    )
    trainer._compute_2d_features_for_gaussians = lambda **kwargs: (torch.zeros(1, 2), torch.ones(1))
    out = trainer._compute_2d_features_all_branches_once(
        node_state_bg=_make_bg_state(torch.device("cpu"), n=1),
        node_state_distant=None,
        node_state_rigid=None,
        source_frame_idx=0,
        rigid_idx_S=torch.zeros(0, dtype=torch.long),
        source_views=[object()],
        source_images=[torch.zeros(2, 2, 3)],
        height=2,
        width=2,
    )
    assert out["src_backproject_pass_count"] == 1


def test_stage4_0_switches_to_fused_v2_path():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    trainer.device = torch.device("cpu")
    trainer.use_fused_cuda_backproject_v2 = True
    trainer._perf_acc = {}

    class _DummyRgbExtractor:
        def render_rgb_only(self, *args, **kwargs):
            return [torch.zeros(2, 2, 3)], {"num_views": 1}

    class _DummyFusedExtractor:
        def __init__(self):
            self.called = False

        def render_and_backproject_streaming_fused(self, **kwargs):
            self.called = True
            return torch.zeros(1, 2), {"pairs_total": 3, "pairs_after_threshold": 2}

    trainer.alpha_t_extractor = _DummyRgbExtractor()
    fused = _DummyFusedExtractor()
    trainer.alpha_t_extractor_v2 = fused
    trainer.feature_backprojector = SimpleNamespace(eps=1e-8)
    trainer.image_feature_extractor = lambda x: torch.zeros(1, 2, 2, 2)

    feat, acc = trainer._compute_2d_features_for_gaussians(
        gaussians={
            "means": torch.zeros(1, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scales": torch.ones(1, 3),
            "opacities": torch.ones(1),
            "colors": torch.zeros(1, 4, 3),
        },
        source_views=[object()],
        source_images=[torch.zeros(2, 2, 3)],
        height=2,
        width=2,
        return_accumulated_weights=False,
    )
    assert fused.called is True
    assert feat.shape == (1, 2)
    assert acc is None


def test_stage4_0_fused_training_path_keeps_feature_grad():
    trainer = MinimalStreetForwardStage4_0.__new__(MinimalStreetForwardStage4_0)
    trainer.device = torch.device("cpu")
    trainer.use_fused_cuda_backproject_v2 = True
    trainer._perf_acc = {}

    class _DummyRgbExtractor:
        def render_rgb_only(self, *args, **kwargs):
            return [torch.zeros(2, 2, 3)], {"num_views": 1}

    class _DummyFusedExtractor:
        def render_and_backproject_streaming_fused(self, **kwargs):
            feat = kwargs["features_2d"]
            out = feat.mean(dim=(1, 2)).sum(dim=0, keepdim=True)
            return out, {"pairs_total": 1, "pairs_after_threshold": 1}

    trainer.alpha_t_extractor = _DummyRgbExtractor()
    trainer.alpha_t_extractor_v2 = _DummyFusedExtractor()
    trainer.feature_backprojector = SimpleNamespace(eps=1e-8)
    scale = torch.tensor(1.0, requires_grad=True)
    trainer.image_feature_extractor = (
        lambda x: torch.ones(x.shape[0], x.shape[1], x.shape[2], 2, dtype=x.dtype) * scale
    )

    feat, _ = trainer._compute_2d_features_for_gaussians(
        gaussians={
            "means": torch.zeros(1, 3),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scales": torch.ones(1, 3),
            "opacities": torch.ones(1),
            "colors": torch.zeros(1, 4, 3),
        },
        source_views=[object()],
        source_images=[torch.zeros(2, 2, 3)],
        height=2,
        width=2,
        return_accumulated_weights=False,
    )
    loss = feat.sum()
    loss.backward()
    assert scale.grad is not None

