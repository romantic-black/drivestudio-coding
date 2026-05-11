from __future__ import annotations

import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

sys.modules.setdefault("open3d", types.SimpleNamespace())
sys.modules.setdefault("pytorch3d", types.SimpleNamespace())
sys.modules.setdefault(
    "pytorch3d.ops",
    types.SimpleNamespace(knn_points=lambda *args, **kwargs: None),
)
sys.modules.setdefault(
    "pytorch3d.transforms",
    types.SimpleNamespace(
        matrix_to_quaternion=lambda x: torch.zeros((*x.shape[:-2], 4), device=x.device, dtype=x.dtype),
        quaternion_to_matrix=lambda x: torch.eye(3, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 3, 3),
        axis_angle_to_matrix=lambda x: torch.eye(3, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 3, 3),
    ),
)
_nvdiffrast_mod = types.ModuleType("nvdiffrast")
_nvdiffrast_torch_mod = types.ModuleType("nvdiffrast.torch")
setattr(_nvdiffrast_mod, "torch", _nvdiffrast_torch_mod)
sys.modules.setdefault("nvdiffrast", _nvdiffrast_mod)
sys.modules.setdefault("nvdiffrast.torch", _nvdiffrast_torch_mod)

import models.streetforward.minimal_trainer_stage4_5 as stage4_5_mod
import models.streetforward.minimal_trainer_stage5_6 as stage5_6_mod
from models.feature_extractors.alpha_t_extractor import _get_viewmat as _no_flip_viewmat
from models.streetforward.math_utils import get_viewmat as _yz_flip_viewmat
from models.streetforward.minimal_trainer_stage4_2 import MinimalStreetForwardStage4_2
from models.streetforward.minimal_trainer_stage4_5 import MinimalStreetForwardStage4_5
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage4_6 import RigidRoute
from models.streetforward.minimal_trainer_stage5_6 import (
    MinimalStreetForwardStage5_6,
    Stage5_6FrameFlattenFuser,
)


def test_stage5_6_render_viewmat_matches_stage5_4_no_axis_flip():
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])

    actual = stage5_6_mod.get_viewmat(c2w)
    expected = _no_flip_viewmat(c2w)
    flipped = _yz_flip_viewmat(c2w)

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, flipped)


def test_stage5_6_fast_fail_when_bridge_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        MinimalStreetForwardStage5_4,
        "_validate_stage5_3_config",
        lambda self, config: None,
    )
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    cfg = {
        "model": {"stage": "5_6"},
        "feature_splat_uncertainty": {
            "bridge": {"enable": True},
            "head": {"predict_rgb_residual": False},
        },
    }
    with pytest.raises(ValueError, match="bridge.enable=true is not supported"):
        stage._validate_stage5_3_config(cfg)


def test_stage5_6_fast_fail_when_rgb_residual_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        MinimalStreetForwardStage5_4,
        "_validate_stage5_3_config",
        lambda self, config: None,
    )
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    cfg = {
        "model": {"stage": "5_6"},
        "feature_splat_uncertainty": {
            "bridge": {"enable": False},
            "head": {"predict_rgb_residual": True},
        },
    }
    with pytest.raises(ValueError, match="predict_rgb_residual=true is not supported"):
        stage._validate_stage5_3_config(cfg)


def test_stage5_6_allows_disabled_feedback_without_near_random_scheduler(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        MinimalStreetForwardStage5_4,
        "_validate_stage5_3_config",
        lambda self, config: None,
    )
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    cfg = {
        "model": {"stage": "5_6"},
        "feature_splat_uncertainty": {
            "bridge": {"enable": False},
            "head": {"predict_rgb_residual": False},
        },
        "nearby_error_feedback": {
            "enable": False,
            "target_role": "near_random",
            "error_pred": {"enable": False, "target_role": "near_random"},
            "cache": {"mode": "frame_bank", "store_age": False},
            "fusion": {"type": "flatten_frame_slots", "input_feedback_age": False},
        },
        "scheduler_v8": {"near_random_supervision": {"enable": False}},
    }
    stage._validate_stage5_3_config(cfg)


def test_stage4_5_parent_init_allows_stage5_6_v4_direct_path(monkeypatch: pytest.MonkeyPatch):
    def _super_init(self, config, device, **kwargs):
        _ = (config, kwargs)
        self.device = device
        self.renderer = object()
        self.sh_degree = 1

    monkeypatch.setattr(MinimalStreetForwardStage4_2, "__init__", _super_init)
    monkeypatch.setattr(stage4_5_mod, "AlphaTWeightExtractorV3", lambda **kwargs: types.SimpleNamespace(**kwargs))
    cfg = OmegaConf.create(
        {
            "model": {
                "stage": "5_6",
                "use_fused_cuda_backproject_v4": True,
                "fused_cuda_backproject_v4_force_fallback": False,
                "use_fused_cuda_backproject_v3": True,
            },
            "losses": {
                "photometric": {"exclude_sky_region": True},
                "mask": {"require_sky_mask": True},
            },
            "logging": {"offset_monitor": {"enable": True, "near_radius_m": 5.0}},
        }
    )
    MinimalStreetForwardStage4_5(cfg, torch.device("cpu"))


def test_stage5_6_cache_ready_warmup_gate():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.stage5_6_pred_error_only_steps = 7000
    assert stage._cache_ready(6999) is False
    assert stage._cache_ready(7000) is True


def test_stage5_6_fusion_scale_ramps_after_pred_error_warmup():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.stage5_6_cache_enable = True
    stage.stage5_6_fusion_enabled = True
    stage.stage5_6_fusion_start_step = 7000
    stage.stage5_6_fusion_warmup_steps = 3000
    stage.stage5_6_fusion_start_scale = 0.0
    stage.stage5_6_fusion_end_scale = 1.0
    assert stage._fusion_scale(6999) == pytest.approx(0.0)
    assert stage._fusion_scale(7000) == pytest.approx(0.0)
    assert stage._fusion_scale(8500) == pytest.approx(0.5)
    assert stage._fusion_scale(10000) == pytest.approx(1.0)


def test_stage5_6_write_cache_uses_lifted_error_feedback_not_source_features():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.stage5_6_cache_enable = True
    stage.stage5_6_pred_error_only_steps = 7000
    stage.stage5_6_cache_keep_only_current_scope = True
    stage._stage5_6_frame_cache = {}
    stage._current_loss_step = lambda batch: 7000
    stage._batch_key = lambda batch: (1, 2)
    lifted_feat = torch.full((3, 4), 2.0)
    source_feat = torch.full((3, 4), 9.0)
    bg_pack = {
        "error": torch.full((3, 1), 0.25),
        "feat": lifted_feat,
        "support": torch.ones((3, 1)),
        "valid": torch.ones((3, 1)),
        "age": torch.zeros((3, 1)),
    }
    batch = {
        "scene_id": 1,
        "segment_id": 2,
        "source_frame_idx": 5,
        "request_meta": {"episode_idx_global": 3, "block_idx_global": 4},
    }
    stage._write_cache(
        batch,
        {"_cache_key": (1, 2), "_feat_2d_bg": source_feat},
        {
            "cache_write": [
                {
                    "nearby_frame_idx": 8,
                    "bg": bg_pack,
                    "distant": None,
                    "rigid": None,
                    "write_node_ratio": 1.0,
                }
            ]
        },
    )
    entry = stage._stage5_6_frame_cache[(1, 2, 3, 4)][8]
    assert torch.equal(entry["bg"]["feat"], lifted_feat)
    assert not torch.equal(entry["bg"]["feat"], source_feat)


def test_stage5_6_read_cache_consumes_same_block_cache_once():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.device = torch.device("cpu")
    stage.stage5_6_cache_enable = True
    stage.stage5_6_pred_error_only_steps = 7000
    stage.stage5_6_cache_max_age = 1
    stage.stage5_6_cache_keep_only_current_scope = True
    stage.stage5_6_fusion_num_slots = 1
    stage._fusion_scale = lambda step: 1.0
    scope_key = (1, 2, 3, 4)
    bg_pack = {
        "error": torch.full((3, 1), 0.25),
        "feat": torch.full((3, 4), 2.0),
        "support": torch.ones((3, 1)),
        "valid": torch.ones((3, 1)),
    }
    stage._stage5_6_frame_cache = {
        scope_key: {
            8: {
                "step": 7000,
                "bg": bg_pack,
                "distant": None,
                "rigid": None,
            }
        }
    }
    empty_long = torch.empty(0, dtype=torch.long)
    empty_bool = torch.empty(0, dtype=torch.bool)
    empty_xyz = torch.empty(0, 3)
    empty_quat = torch.empty(0, 4)
    route = RigidRoute(
        S=empty_long,
        S_in=empty_long,
        S_out=empty_long,
        inside_mask_S=empty_bool,
        route_inside_global=empty_bool,
        means_world_S=empty_xyz,
        quats_world_S=empty_quat,
    )

    out = stage._read_cache(
        scope_key,
        route,
        torch.float32,
        current_step=7001,
        frame_indices=[8],
    )

    assert out is not None
    assert torch.equal(out["bg"][0]["feat"], bg_pack["feat"])
    assert scope_key not in stage._stage5_6_frame_cache
    assert stage._read_cache(scope_key, route, torch.float32, current_step=7001, frame_indices=[8]) is None


def test_stage5_6_write_cache_without_new_payload_clears_current_scope():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.stage5_6_cache_enable = True
    stage.stage5_6_pred_error_only_steps = 7000
    stage.stage5_6_cache_keep_only_current_scope = True
    stage._current_loss_step = lambda batch: 7001
    stage._batch_key = lambda batch: (1, 2)
    scope_key = (1, 2, 3, 4)
    stage._stage5_6_frame_cache = {scope_key: {8: {"step": 7000}}}
    batch = {"request_meta": {"episode_idx_global": 3, "block_idx_global": 4}}

    stage._write_cache(batch, {}, {})

    assert scope_key not in stage._stage5_6_frame_cache


def test_stage5_6_collects_feedback_targets_from_near_random_target_roles():
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.stage5_6_error_target_role = "near_random"
    batch = {
        "targets": [
            {"frame_idx": 4, "cam_idx": 0},
            {"frame_idx": 8, "cam_idx": 0},
            {"frame_idx": 8, "cam_idx": 1},
            {"frame_idx": 12, "cam_idx": 0},
        ],
        "request_meta": {
            "target_image_roles": ["source", "near_random", "near_random", "visited"],
        },
    }

    out = stage._collect_feedback_targets(batch, max_targets=8, require_aux_if_requested=False)
    assert [(x["frame_idx"], x["cam_idx"], x["target_index"], x["batch_target_index"]) for x in out] == [
        (8, 0, 0, 1),
        (8, 1, 1, 2),
    ]
    assert all(x["role"] == "near_random" for x in out)


def test_stage5_6_near_random_target_role_requires_scheduler_near_random(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        MinimalStreetForwardStage5_4,
        "_validate_stage5_3_config",
        lambda self, config: None,
    )
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    cfg = {
        "model": {"stage": "5_6"},
        "nearby_error_feedback": {"target_role": "near_random"},
        "scheduler_v8": {
            "aux_feature_splat_targets": {"enable": True},
            "near_random_supervision": {"enable": True},
        },
    }
    with pytest.raises(ValueError, match="aux_feature_splat_targets.enable=false"):
        stage._validate_stage5_3_config(cfg)


def test_stage5_6_nearby_direct_loss_requests_retain_graph_for_proxy_backward(monkeypatch: pytest.MonkeyPatch):
    def _super_forward(self, batch):
        _ = (self, batch)
        return {
            "loss": torch.zeros((), requires_grad=True),
            "proxies": {"dummy": torch.zeros((), requires_grad=True)},
        }

    nearby_loss = torch.ones((), requires_grad=True)
    monkeypatch.setattr(MinimalStreetForwardStage5_4, "forward", _super_forward)
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage.training = True
    stage.stage5_6_nearby_enabled = True
    stage.stage5_6_error_enabled = False
    stage.stage5_6_cache_enable = False
    stage.stage5_6_fusion_enabled = False
    stage._stage5_6_frame_cache = {}
    stage._compute_nearby_direct_loss = lambda batch, out: {"loss": nearby_loss, "processed": 1.0}
    stage._log_nearby_pack = lambda out, nearby: None
    stage._write_cache = lambda batch, out, err: None
    stage._log_feedback_state = lambda out, step, err: None
    stage._current_loss_step = lambda batch: 0
    stage._batch_key = lambda batch: (1, 2)

    out = stage.forward({"request_meta": {"episode_idx_global": 3, "block_idx_global": 4}})
    assert out["_retain_graph_for_proxy_backward"] is True


def test_stage5_6_train_step_preserves_debug_images_after_parent_result_filter(monkeypatch: pytest.MonkeyPatch):
    def _super_train_step(
        self,
        batch,
        step=None,
        profile_phase_timing=False,
        sync_cuda_timing=False,
        scheduler_node_sync=None,
    ):
        _ = (batch, step, profile_phase_timing, sync_cuda_timing, scheduler_node_sync)
        self._stage5_6_last_nearby_debug_images = [{"pred": torch.zeros((1, 1, 3))}]
        self._stage5_6_last_error_debug_images = [{"render": torch.zeros((1, 1, 3))}]
        return {"loss": 0.0}

    monkeypatch.setattr(MinimalStreetForwardStage5_4, "train_step", _super_train_step)
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)

    out = stage.train_step({})
    assert len(out["_stage5_6_nearby_debug_images"]) == 1
    assert len(out["_stage5_6_error_debug_images"]) == 1


def test_stage5_6_train_step_forwards_runtime_policy(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def _super_train_step(
        self,
        batch,
        step=None,
        profile_phase_timing=False,
        sync_cuda_timing=False,
        scheduler_node_sync=None,
        runtime_policy=None,
    ):
        _ = (self, batch, step, profile_phase_timing, sync_cuda_timing, scheduler_node_sync)
        seen["runtime_policy"] = runtime_policy
        return {"loss": 0.0}

    monkeypatch.setattr(MinimalStreetForwardStage5_4, "train_step", _super_train_step)
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    policy = object()

    _ = stage.train_step({}, runtime_policy=policy)
    assert seen["runtime_policy"] is policy


def test_batcheval_segment_finetune_freezes_error_and_feedback_by_default():
    from tools.eval_streetforward_benchmark import _configure_segment_finetune_optimizer

    class _ImageFeatureExtractor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual = torch.nn.Linear(2, 2)
            self.fusion = torch.nn.Linear(2, 2)
            self.dino_adapter = torch.nn.Module()
            self.dino_adapter.backbone = torch.nn.Linear(2, 2)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sparse_conv = torch.nn.Linear(2, 2)
            self.image_feature_extractor = _ImageFeatureExtractor()
            self.struct_decoder = torch.nn.Linear(2, 2)
            self.mlp_params_embed = torch.nn.Linear(2, 2)
            self.param_embed_norm = torch.nn.LayerNorm(2)
            self.gru_update = torch.nn.Linear(2, 2)
            self.gru_candidate = torch.nn.Linear(2, 2)
            self.stage5_6_error_head = torch.nn.Linear(2, 2)
            self.err_splat_proj_bg = torch.nn.Linear(2, 2)
            self.stage5_6_bg_fuser = torch.nn.Linear(2, 2)
            self.current_obs_struct_embed = torch.nn.Linear(2, 2)
            self.mlp_offset_pos = torch.nn.Linear(2, 2)
            self.mlp_conv_sky = torch.nn.Linear(2, 2)
            self.sky_model = torch.nn.Linear(2, 2)

    model = _TinyModel()
    _configure_segment_finetune_optimizer(
        model,
        finetune_cfg={"lr": 1.0e-5, "weight_decay": 0.0, "train_feedback_fuser": False},
        start_step=123,
    )

    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    assert flags["struct_decoder.weight"] is True
    assert flags["sparse_conv.weight"] is False
    assert flags["mlp_params_embed.weight"] is False
    assert flags["param_embed_norm.weight"] is False
    assert flags["gru_update.weight"] is True
    assert flags["gru_candidate.weight"] is True
    assert flags["mlp_offset_pos.weight"] is True
    assert flags["image_feature_extractor.residual.weight"] is True
    assert flags["image_feature_extractor.fusion.weight"] is True
    assert flags["current_obs_struct_embed.weight"] is True
    assert flags["stage5_6_error_head.weight"] is False
    assert flags["err_splat_proj_bg.weight"] is False
    assert flags["stage5_6_bg_fuser.weight"] is False
    assert flags["image_feature_extractor.dino_adapter.backbone.weight"] is False
    assert flags["mlp_conv_sky.weight"] is False
    assert flags["sky_model.weight"] is False
    assert model.optimizer.global_step == 123

    _configure_segment_finetune_optimizer(
        model,
        finetune_cfg={"lr": 1.0e-5, "weight_decay": 0.0, "train_feedback_fuser": True},
        start_step=124,
    )
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    assert flags["stage5_6_bg_fuser.weight"] is True
    assert flags["stage5_6_error_head.weight"] is False
    assert flags["sparse_conv.weight"] is False
    assert model.optimizer.global_step == 124

    _configure_segment_finetune_optimizer(
        model,
        finetune_cfg={"lr": 1.0e-5, "weight_decay": 0.0, "freeze_dino": False},
        start_step=125,
    )
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    assert flags["image_feature_extractor.dino_adapter.backbone.weight"] is True
    assert flags["stage5_6_error_head.weight"] is False


def test_stage5_6_reset_node_state_clears_cache(monkeypatch: pytest.MonkeyPatch):
    def _super_reset(self):
        self._super_reset_called = True

    monkeypatch.setattr(MinimalStreetForwardStage5_4, "reset_node_state", _super_reset)
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage._stage5_6_frame_cache = {(1, 2, 3, 4): {8: {"step": 1}}}
    stage._stage5_6_active_cache = {"bg": None}
    stage._stage5_6_last_fused_features = {"bg": torch.zeros((1, 1))}
    stage.reset_node_state()
    assert stage._super_reset_called is True
    assert stage._stage5_6_frame_cache == {}
    assert stage._stage5_6_active_cache is None
    assert stage._stage5_6_last_fused_features == {}


def test_stage5_6_record_block_history_clears_cache(monkeypatch: pytest.MonkeyPatch):
    def _super_record(self, batch, event=None):
        _ = (batch, event)
        return {"super_called": 1.0}

    monkeypatch.setattr(MinimalStreetForwardStage5_4, "record_block_history", _super_record)
    stage = MinimalStreetForwardStage5_6.__new__(MinimalStreetForwardStage5_6)
    stage._stage5_6_frame_cache = {(3, 4, 5, 6): {8: {"step": 5}}}
    stage._stage5_6_active_cache = {"bg": None}
    stage._stage5_6_last_fused_features = {"bg": torch.zeros((1, 1))}
    out = stage.record_block_history(batch={}, event={"dummy": 1})
    assert out == {"super_called": 1.0}
    assert stage._stage5_6_frame_cache == {}
    assert stage._stage5_6_active_cache is None
    assert stage._stage5_6_last_fused_features == {}


def test_stage5_6_fuser_zero_init_is_identity():
    fuser = Stage5_6FrameFlattenFuser(feat_dim=8, feedback_dim=4, num_slots=2, hidden_dim=16)
    feat = torch.randn((5, 8), dtype=torch.float32)
    feedback = {
        "feat": torch.randn((5, 2, 4), dtype=torch.float32),
        "error": torch.rand((5, 2, 1), dtype=torch.float32),
        "support": torch.ones((5, 2, 1), dtype=torch.float32),
        "valid": torch.ones((5, 2, 1), dtype=torch.float32),
    }
    out = fuser(feat, feedback)
    assert torch.allclose(out, feat, atol=0.0, rtol=0.0)


def test_stage5_6_image_debug_saver_exports_near_random_error_maps(tmp_path):
    from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import _save_stage5_6_debug_images

    result = {
        "_stage5_6_error_debug_images": [
            {
                "target_index": 0,
                "frame_idx": 8,
                "cam_idx": 0,
                "role": "near_random",
                "render": torch.full((4, 5, 3), 0.25),
                "pred_error": torch.full((4, 5), 0.1),
                "actual_error": torch.full((4, 5), 0.2),
            }
        ],
    }
    _save_stage5_6_debug_images(
        step=67,
        result=result,
        raw_batch={"scene_folder_name": "004"},
        log_dir=str(tmp_path),
        block_idx_global=1,
        scene_id_fallback=4,
        pixel_camera_ids=[0],
    )
    stem = "step000067_b000001_sc004_near_random0_f00008_c0_nuscam0"
    assert not (tmp_path / "images" / "train" / f"{stem}_pred.png").exists()
    assert not (tmp_path / "images" / "train" / f"{stem}_gt.png").exists()
    assert (tmp_path / "images" / "error" / f"{stem}_render.png").exists()
    assert (tmp_path / "images" / "error" / f"{stem}_error.png").exists()
    assert (tmp_path / "images" / "error" / f"{stem}_actual_error.png").exists()


def test_stage5_6_image_debug_due_matches_block_interval():
    from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import _stage5_6_debug_images_due

    assert _stage5_6_debug_images_due(
        image_trigger_mode="block_end",
        step=0,
        scheduler_info={"block_idx_global": 0},
        step_events=[{"type": "block_end", "block_idx_global": 1}],
        image_trigger_interval_steps=4800,
        image_interval_blocks_equiv=300,
    )
    assert not _stage5_6_debug_images_due(
        image_trigger_mode="block_end",
        step=0,
        scheduler_info={"block_idx_global": 1},
        step_events=[{"type": "block_end", "block_idx_global": 2}],
        image_trigger_interval_steps=4800,
        image_interval_blocks_equiv=300,
    )
    assert _stage5_6_debug_images_due(
        image_trigger_mode="block_end",
        step=0,
        scheduler_info={"block_idx_global": 300},
        step_events=[{"type": "block_end", "block_idx_global": 301}],
        image_trigger_interval_steps=4800,
        image_interval_blocks_equiv=300,
    )
