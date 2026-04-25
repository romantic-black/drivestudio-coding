from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.streetforward.minimal_trainer_stage4_6 import MinimalStreetForwardStage4_6, RigidRoute
from models.streetforward.minimal_trainer_stage5_2 import MinimalStreetForwardStage5_2


def _empty_history(num_rows: int) -> dict[str, torch.Tensor]:
    z = torch.zeros((num_rows, 1), dtype=torch.float32)
    return {
        "support_ema": z.clone(),
        "error_ema": z.clone(),
        "update_norm_ema": z.clone(),
        "initialized": z.clone(),
    }


def test_should_apply_step_update_norm_ema_respects_apply_in_eval_switch():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)

    trainer.training = False
    trainer.stage5_2_history_update_apply_in_eval = False
    assert trainer._should_apply_step_update_norm_ema() is False

    trainer.training = True
    trainer.stage5_2_history_update_apply_in_eval = False
    assert trainer._should_apply_step_update_norm_ema() is True

    trainer.training = False
    trainer.stage5_2_history_update_apply_in_eval = True
    assert trainer._should_apply_step_update_norm_ema() is True


def test_forward_skips_update_norm_ema_in_eval_by_default(monkeypatch):
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.training = False
    trainer.stage5_2_history_update_apply_in_eval = False

    calls = {"n": 0}

    def _fake_parent_forward(self, batch):
        _ = batch
        return {"_cache_key": (0, 0)}

    def _fake_apply(self, out):
        _ = out
        calls["n"] += 1

    monkeypatch.setattr(MinimalStreetForwardStage4_6, "forward", _fake_parent_forward)
    monkeypatch.setattr(MinimalStreetForwardStage5_2, "_apply_step_update_norm_ema_from_out", _fake_apply)

    out = trainer.forward({})
    assert out["_cache_key"] == (0, 0)
    assert calls["n"] == 0


def test_forward_applies_update_norm_ema_in_eval_when_enabled(monkeypatch):
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.training = False
    trainer.stage5_2_history_update_apply_in_eval = True

    calls = {"n": 0}

    def _fake_parent_forward(self, batch):
        _ = batch
        return {"_cache_key": (0, 0)}

    def _fake_apply(self, out):
        _ = out
        calls["n"] += 1

    monkeypatch.setattr(MinimalStreetForwardStage4_6, "forward", _fake_parent_forward)
    monkeypatch.setattr(MinimalStreetForwardStage5_2, "_apply_step_update_norm_ema_from_out", _fake_apply)

    trainer.forward({})
    assert calls["n"] == 1


def test_apply_step_update_norm_ema_updates_written_rows_only():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.stage5_2_update_norm_beta = 0.5
    history = _empty_history(3)

    trainer._apply_step_update_norm_ema(
        history=history,
        update_norm_cur=torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float32),
    )

    assert torch.allclose(history["update_norm_ema"], torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32))


def test_commit_block_support_to_history_uses_block_mean_and_visibility_beta():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.device = torch.device("cpu")
    trainer.stage5_2_support_beta_visible = 0.75
    trainer.stage5_2_support_beta_invisible = 0.90
    trainer.stage5_2_bg_visible_min = 0.75
    trainer.stage5_2_distant_visible_min = 0.5
    trainer.stage5_2_rigid_visible_min = 0.5

    key = (1, 2)
    hist_bg = _empty_history(2)
    hist_dist = _empty_history(0)
    hist_rigid = _empty_history(0)

    trainer.stage5_2_block_support_bg = {
        key: {
            "sum": torch.tensor([[2.0], [1.0]], dtype=torch.float32),
            "count": torch.tensor([[2.0], [2.0]], dtype=torch.float32),
        }
    }
    trainer.stage5_2_block_support_distant = {}
    trainer.stage5_2_block_support_rigid = {}

    trainer._commit_block_support_to_history(
        key=key,
        history_bg=hist_bg,
        history_distant=hist_dist,
        history_rigid=hist_rigid,
    )

    expected_support = torch.tensor([[0.25], [0.05]], dtype=torch.float32)
    assert torch.allclose(hist_bg["support_ema"], expected_support)
    assert torch.allclose(hist_bg["initialized"], torch.tensor([[1.0], [0.0]], dtype=torch.float32))


def test_apply_residual_history_update_visible_only():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.stage5_2_error_beta = 0.5
    history = _empty_history(2)

    trainer._apply_residual_history_update(
        history=history,
        error_cur=torch.tensor([[2.0], [4.0]], dtype=torch.float32),
        visible_mask=torch.tensor([True, False]),
    )

    assert torch.allclose(history["error_ema"], torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    assert torch.allclose(history["initialized"], torch.tensor([[1.0], [0.0]], dtype=torch.float32))


def test_accumulate_support_before_update_maps_rigid_rows_to_global_history_rows():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.device = torch.device("cpu")
    trainer.stage5_2_block_support_bg = {}
    trainer.stage5_2_block_support_distant = {}
    trainer.stage5_2_block_support_rigid = {}

    route = RigidRoute(
        S=torch.tensor([1, 3], dtype=torch.long),
        S_in=torch.tensor([1], dtype=torch.long),
        S_out=torch.tensor([3], dtype=torch.long),
        inside_mask_S=torch.tensor([True, False]),
        route_inside_global=torch.zeros((5,), dtype=torch.bool),
        means_world_S=torch.zeros((2, 3), dtype=torch.float32),
        quats_world_S=torch.zeros((2, 4), dtype=torch.float32),
    )

    trainer._accumulate_support_before_update(
        key=(0, 0),
        num_bg_total=1,
        num_distant_total=0,
        num_rigid_total=5,
        route=route,
        acc_w_bg=torch.tensor([1.0], dtype=torch.float32),
        acc_w_distant=None,
        acc_w_rigid_S=torch.tensor([0.2, 0.4], dtype=torch.float32),
    )

    bg_acc = trainer.stage5_2_block_support_bg[(0, 0)]
    assert torch.allclose(bg_acc["sum"], torch.log1p(torch.tensor([[1.0]], dtype=torch.float32)))
    assert torch.allclose(bg_acc["count"], torch.ones((1, 1), dtype=torch.float32))

    rigid_acc = trainer.stage5_2_block_support_rigid[(0, 0)]
    expected_sum = torch.zeros((5, 1), dtype=torch.float32)
    expected_sum[1] = torch.log1p(torch.tensor([0.2]))
    expected_sum[3] = torch.log1p(torch.tensor([0.4]))
    expected_cnt = torch.zeros((5, 1), dtype=torch.float32)
    expected_cnt[1] = 1.0
    expected_cnt[3] = 1.0
    assert torch.allclose(rigid_acc["sum"], expected_sum)
    assert torch.allclose(rigid_acc["count"], expected_cnt)


def test_build_record_targets_rejects_non_source_record_views():
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer.stage5_2_record_views = "target_image_refs"
    with pytest.raises(ValueError, match="record_views=source_image_refs"):
        trainer._build_record_targets({})


def test_validate_stage5_2_config_rejects_legacy_flat_history_keys():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml")))
    cfg.model.history_memory.support_beta_visible = 0.75
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    with pytest.raises(ValueError, match="no longer supports flat history_memory keys"):
        trainer._validate_stage5_2_config(cfg)


def test_validate_stage5_2_config_requires_feature_extractor_block():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml")))
    del cfg.model["feature_extractor"]
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    with pytest.raises(ValueError, match=r"Missing required config: model\.feature_extractor"):
        trainer._validate_stage5_2_config(cfg)


def test_validate_stage5_2_config_requires_dinov2_unet_fusion_type():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml")))
    cfg.model.feature_extractor.type = "image_feature_extractor"
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    with pytest.raises(ValueError, match="requires model.feature_extractor.type='dinov2_unet_fusion'"):
        trainer._validate_stage5_2_config(cfg)


def test_validate_stage5_2_config_allows_missing_dino_freeze_key():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml")))
    del cfg.model.feature_extractor.dino["freeze"]
    trainer = MinimalStreetForwardStage5_2.__new__(MinimalStreetForwardStage5_2)
    trainer._validate_stage5_2_config(cfg)
