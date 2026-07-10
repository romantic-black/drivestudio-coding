from __future__ import annotations

import json
from types import SimpleNamespace

from omegaconf import OmegaConf

import tools.train_iforward as train_ifwd


def _cfg(tmp_path):
    cfg = OmegaConf.create(
        {
            "output_name": "iforward_stage3_0_full_train_30k_assim_30k_repair",
            "scheduler_stage3_0": {
                "enable": True,
                "version": "stage3_0_optimizer_sequence_v1",
                "index_dir": "/tmp/index",
                "index_fingerprint": "fp123",
            },
            "scheduler_stage3_0_validation": {
                "enable": True,
                "run_at_train_start": True,
            },
            "model": {
                "iforward": {
                    "version": "stage3_0_scalar_anchor_child_support_parent_legacy",
                    "lifting": {
                        "type": "full_sparse_gather",
                        "parent": {"type": "legacy_direct_lift"},
                        "child_gather": {"type": "support_center"},
                    },
                }
            },
        }
    )
    cfg.log_dir = str(tmp_path)
    return cfg


def test_iforward_run_start_hook_writes_manifest_snapshot_and_first_row(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output_name: manifest-test\n", encoding="utf-8")
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    train_ifwd._iforward_run_start_hook(
        cfg=_cfg(tmp_path),
        args=SimpleNamespace(config_file=str(config_path)),
        metrics_fh=rows,
        resume_checkpoint="/tmp/resume.pt",
        init_checkpoint="/tmp/init.pt",
        checkpoint_prefix="iforward_stage3",
    )

    assert rows
    assert rows[0]["split"] == "run_manifest"
    assert rows[0]["schema_version"] == "iforward_run_manifest_v1"
    assert rows[0]["scheduler_key"] == "scheduler_stage3_0"
    assert rows[0]["scheduler_version"] == "stage3_0_optimizer_sequence_v1"
    assert rows[0]["index_fingerprint"] == "fp123"
    assert rows[0]["validation_key"] == "scheduler_stage3_0_validation"
    assert rows[0]["iforward_version"] == "stage3_0_scalar_anchor_child_support_parent_legacy"
    assert rows[0]["parent_lifting_type"] == "legacy_direct_lift"
    assert rows[0]["child_gather_type"] == "support_center"
    assert rows[0]["resume_checkpoint"] == "/tmp/resume.pt"
    assert rows[0]["init_checkpoint"] == "/tmp/init.pt"
    assert rows[0]["local_gs_state_schema_version"] == 2
    assert rows[0]["uncertainty_state_version"] == "appearance_logvar_v1"
    assert rows[0]["uncertainty_raster_version"] == "detached_moments_v1"

    manifest_path = tmp_path / "run_manifest.json"
    snapshot_path = tmp_path / "config_snapshot.yaml"
    assert manifest_path.exists()
    assert snapshot_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_snapshot_sha256"] == rows[0]["config_snapshot_sha256"]
    assert "stage3_0_scalar_anchor_child_support_parent_legacy" in snapshot_path.read_text(encoding="utf-8")


def test_iforward_run_manifest_prefers_enabled_validation_v4(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output_name: manifest-test\n", encoding="utf-8")
    cfg = _cfg(tmp_path)
    cfg.scheduler_stage3_0_validation.enable = False
    cfg.iforward_validation_v4 = {"enable": True, "run_at_train_start": False}
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    train_ifwd._iforward_run_start_hook(
        cfg=cfg,
        args=SimpleNamespace(config_file=str(config_path)),
        metrics_fh=rows,
        resume_checkpoint="",
        init_checkpoint="",
        checkpoint_prefix="iforward_stage3",
    )

    assert rows[0]["validation_key"] == "iforward_validation_v4"
    assert rows[0]["validation_enable"] is True


def test_iforward_v2_manifest_records_semantic_schema_versions(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("output_name: manifest-v2\n", encoding="utf-8")
    cfg = _cfg(tmp_path)
    cfg.model.iforward.version = "stage3_3_uncertainty_v2"
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))
    train_ifwd._iforward_run_start_hook(
        cfg=cfg,
        args=SimpleNamespace(config_file=str(config_path)),
        metrics_fh=rows,
        resume_checkpoint="",
        init_checkpoint="",
        checkpoint_prefix="iforward_stage3_3",
    )
    assert rows[0]["uncertainty_state_version"] == "appearance_logvar_v1"
    assert rows[0]["uncertainty_updater_version"] == "state_conditioned_target_v2"
    assert rows[0]["uncertainty_raster_version"] == "detached_moments_aleatoric_loss_v2"
    assert rows[0]["uncertainty_loss_version"] == "decoupled_warmup_v2"
