"""Shared validation for StreetForward asset export CLIs (full training YAML only)."""

from __future__ import annotations

from omegaconf import OmegaConf


def require_full_training_config_for_asset_export(cfg) -> None:
    """
    Asset export builds MultiSceneDatasetV3 and needs dataset:, data:, etc.
    Snippet-only YAMLs (e.g. tools/streetforward_assets_data_snippet.yaml) are not sufficient.
    """
    if OmegaConf.select(cfg, "dataset") is None:
        raise ValueError(
            "导出资产需要完整训练配置（顶层必须有 `dataset:`）。当前 --config_file 不是完整训练 YAML。"
            "建议直接使用 configs/minimal_streetforward_stage4_4_multi_scene_v5.yaml，"
            "或 tools/streetforward_assets_data_snippet.yaml（该文件已包含 dataset + data + data.assets）。"
        )
    if OmegaConf.select(cfg, "data") is None:
        raise ValueError(
            "导出资产需要完整训练配置：顶层缺少 `data:`。"
        )
