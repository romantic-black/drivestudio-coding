from __future__ import annotations

import torch

from models.streetforward.minimal_trainer_stage2_1 import _create_proxy_params
from models.streetforward.minimal_trainer_stage4_0 import (
    _backward_to_render_params_bg_rigid_distant,
    merge_debug_stats_as_perf_floats,
    spatial_hw_from_image_tensor,
)
from models.streetforward.minimal_trainer_stage4_1 import MinimalStreetForwardStage4_1
from models.streetforward.node_states import NodeStateRigid


def _make_rigid_state(device: torch.device) -> NodeStateRigid:
    return NodeStateRigid(
        means=torch.tensor([[1.0, 0.0, 0.0]], device=device),
        scales_log=torch.zeros((1, 3), device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacity_logit=torch.zeros((1, 1), device=device),
        sh_dc=torch.zeros((1, 3), device=device),
        sh_rest=torch.zeros((1, 3, 3), device=device),
        point_ids=torch.tensor([[0]], dtype=torch.long, device=device),
        instances_quats=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device),
        instances_trans=torch.tensor([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], device=device),
        instances_fv=torch.tensor([[True], [True]], dtype=torch.bool, device=device),
        instance_ids=[0],
        frame_ids=[7, 8],
        cur_frame=7,
    )


def test_spatial_hw_from_image_tensor_hwc_vs_chw():
    hwc = torch.zeros(10, 20, 3)
    assert spatial_hw_from_image_tensor(hwc) == (10, 20)
    chw = torch.zeros(3, 10, 20)
    assert spatial_hw_from_image_tensor(chw) == (10, 20)


def test_merge_debug_stats_flattens_per_view_lists():
    out: dict = {}
    merge_debug_stats_as_perf_floats(
        out,
        "2d_bp_",
        {
            "pairs_total": 5,
            "render_packed_ms_per_view": [1.0, 2.0, 3.0],
        },
    )
    assert out["2d_bp_pairs_total"] == 5.0
    assert out["2d_bp_render_packed_ms_per_view_sum"] == 6.0
    assert out["2d_bp_render_packed_ms_per_view_mean"] == 2.0
    assert out["2d_bp_render_packed_ms_per_view_len"] == 3.0


def test_global_to_subset_rows():
    device = torch.device("cpu")
    trainer = MinimalStreetForwardStage4_1.__new__(MinimalStreetForwardStage4_1)
    trainer.device = device
    U = torch.tensor([2, 5, 9], dtype=torch.long)
    g = torch.tensor([5, 9], dtype=torch.long)
    rows = trainer._global_to_subset_rows(g, U, n_rigid=10, device=device)
    assert rows.tolist() == [1, 2]


def test_feat_valid_from_support_matches_design():
    device = torch.device("cpu")
    N = 3
    S = torch.tensor([0, 2], dtype=torch.long)
    mask_src = torch.zeros(N, dtype=torch.bool, device=device)
    mask_src[S] = True
    acc_w = torch.tensor([0.5, 1.0], device=device)
    tau = 0.75
    mask_feat = torch.zeros(N, dtype=torch.bool, device=device)
    mask_feat[S] = acc_w > tau
    mask_tgt = torch.tensor([True, True, True], device=device)
    mask_update = mask_feat & mask_tgt
    assert mask_feat.tolist() == [False, False, True]
    assert mask_update[2].item() is True


def test_validate_requires_sky_mask_per_target():
    device = torch.device("cpu")
    trainer = MinimalStreetForwardStage4_1.__new__(MinimalStreetForwardStage4_1)
    trainer.device = device
    eye = torch.eye(4)
    src_v = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0)})()
    img = torch.zeros(4, 5, 3)
    targets = [
        {
            "frame_idx": 7,
            "view": src_v,
            "gt_image": img,
            "viewdirs": torch.zeros(4, 5, 3),
        }
    ]
    try:
        trainer._validate_stage4_1_batch(
            {"source_views": [src_v], "source_images": [img], "source_frame_idx": 7},
            targets,
            None,
        )
    except ValueError as e:
        assert "sky_mask" in str(e).lower()
        return
    raise AssertionError("Expected ValueError for missing sky_mask.")


def test_validate_multi_frame_fails_missing_rigid_pose():
    device = torch.device("cpu")
    trainer = MinimalStreetForwardStage4_1.__new__(MinimalStreetForwardStage4_1)
    trainer.device = device
    state = _make_rigid_state(device)
    eye = torch.eye(4)
    src_v = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0)})()
    img = torch.zeros(4, 5, 3)
    sm = torch.ones(4, 5)
    vd = torch.zeros(4, 5, 3)
    targets = [
        {"frame_idx": 99, "view": src_v, "gt_image": img, "sky_mask": sm, "viewdirs": vd},
    ]
    try:
        trainer._validate_stage4_1_batch(
            {"source_views": [src_v], "source_images": [img], "source_frame_idx": 7},
            targets,
            state,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown frame_idx in rigid.")


def test_validate_requires_source_frame_target_coverage_for_multi_src():
    device = torch.device("cpu")
    trainer = MinimalStreetForwardStage4_1.__new__(MinimalStreetForwardStage4_1)
    trainer.device = device
    eye = torch.eye(4)
    src_v0 = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 0})()
    src_v1 = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 1})()
    img = torch.zeros(4, 5, 3)
    sm = torch.ones(4, 5)
    vd = torch.zeros(4, 5, 3)
    targets = [
        {"frame_idx": 7, "view": src_v0, "gt_image": img, "sky_mask": sm, "viewdirs": vd},
    ]
    try:
        trainer._validate_stage4_1_batch(
            {"source_views": [src_v0, src_v1], "source_images": [img, img], "source_frame_idx": 7},
            targets,
            None,
        )
    except ValueError as e:
        assert "coverage" in str(e).lower()
        return
    raise AssertionError("Expected ValueError for insufficient source-frame target coverage.")


def test_validate_prefers_cam_idx_match_over_pose_match():
    device = torch.device("cpu")
    trainer = MinimalStreetForwardStage4_1.__new__(MinimalStreetForwardStage4_1)
    trainer.device = device
    eye = torch.eye(4)
    # source view poses differ from target poses; cam_idx matching should still pass
    src_v0 = type("View", (), {"camtoworlds": eye.clone(), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 0})()
    src_v1 = type("View", (), {"camtoworlds": (eye * 2.0), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 1})()
    tgt_v0 = type("View", (), {"camtoworlds": (eye * 3.0), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 0})()
    tgt_v1 = type("View", (), {"camtoworlds": (eye * 4.0), "Ks": torch.eye(3).unsqueeze(0), "cam_idx": 1})()
    img = torch.zeros(4, 5, 3)
    sm = torch.ones(4, 5)
    vd = torch.zeros(4, 5, 3)
    targets = [
        {"frame_idx": 7, "view": tgt_v0, "gt_image": img, "sky_mask": sm, "viewdirs": vd},
        {"frame_idx": 7, "view": tgt_v1, "gt_image": img, "sky_mask": sm, "viewdirs": vd},
    ]
    trainer._validate_stage4_1_batch(
        {"source_views": [src_v0, src_v1], "source_images": [img, img], "source_frame_idx": 7},
        targets,
        None,
    )


def test_backward_accepts_rigid_world_proxy_pairs():
    device = torch.device("cpu")
    means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    bg_rp = {
        "means_r": means,
        "scales_r": torch.ones(2, 3, requires_grad=True),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True),
        "opacities_r": torch.ones(2, requires_grad=True),
        "colors_r": torch.zeros(2, 1, 3, requires_grad=True),
    }
    bg_p = _create_proxy_params(bg_rp)
    r1 = {
        "means_r": torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True),
        "scales_r": torch.ones(1, 3, requires_grad=True),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True),
        "opacities_r": torch.ones(1, requires_grad=True),
        "colors_r": torch.zeros(1, 1, 3, requires_grad=True),
    }
    p1 = _create_proxy_params(r1)
    r2 = {
        "means_r": torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True),
        "scales_r": torch.ones(1, 3, requires_grad=True),
        "quats_r": torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True),
        "opacities_r": torch.ones(1, requires_grad=True),
        "colors_r": torch.zeros(1, 1, 3, requires_grad=True),
    }
    p2 = _create_proxy_params(r2)
    loss = bg_p["means_p"].sum() + p1["means_p"].sum() + p2["means_p"].sum()
    loss.backward()
    _backward_to_render_params_bg_rigid_distant(
        bg_rp,
        bg_p,
        None,
        None,
        None,
        None,
        rigid_world_proxy_pairs=[(r1, p1), (r2, p2)],
    )
    assert bg_rp["means_r"].grad is not None
    assert r1["means_r"].grad is not None
    assert r2["means_r"].grad is not None
