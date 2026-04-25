from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_stage5_2_v8_config_restored_to_legacy_baseline():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_2_multi_scene_v8.yaml")))

    assert str(cfg.model.stage) == "5_2"
    assert int(cfg.model.feat_2d_channels) == 32
    assert int(cfg.model.struct_decoder.feat_2d_channels) == 32

    assert "feature_extractor" not in cfg.model
    assert str(cfg.model.history_memory.record_views) == "source_image_refs"
    assert float(cfg.model.history_memory.support_beta_visible) == 0.75
    assert float(cfg.model.history_memory.support_beta_invisible) == 0.90
    assert float(cfg.model.history_memory.error_beta) == 0.75
    assert float(cfg.model.history_memory.update_norm_beta) == 0.85

    assert int(cfg.model.struct_decoder.near.channels) == 64
    assert int(cfg.model.struct_decoder.near.xcpe.num_layers) == 1
    assert float(cfg.model.struct_decoder.near.xcpe.residual_scale_init) == 1.0e-3


def test_stage5_3_v8_config_contains_fusion_memory_switch_and_phase1_xcpe():
    cfg = OmegaConf.load(str(Path("configs/minimal_streetforward_stage5_3_multi_scene_v8.yaml")))

    assert str(cfg.model.stage) == "5_3"
    assert int(cfg.model.feat_2d_channels) == 48
    assert int(cfg.model.struct_decoder.feat_2d_channels) == 48

    assert str(cfg.model.feature_extractor.type) == "dinov2_unet_fusion"
    assert str(cfg.model.feature_extractor.dino.model_name) == "vit_base_patch14_reg4_dinov2"
    assert bool(cfg.model.feature_extractor.dino.freeze) is True
    assert int(cfg.model.feature_extractor.fusion.out_channels) == 48

    assert bool(cfg.model.history_memory["update"].apply_in_eval) is False
    assert str(cfg.model.history_memory.record_views) == "source_image_refs"

    assert int(cfg.model.struct_decoder.near.channels) == 96
    assert int(cfg.model.struct_decoder.near.xcpe.num_layers) == 2
    assert float(cfg.model.struct_decoder.near.xcpe.residual_scale_init) == 5.0e-3
