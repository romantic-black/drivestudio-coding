from __future__ import annotations

import pytest

import tools.train_iforward as train_ifwd


def _cfg():
    return {
        "scheduler_stage3_0": {
            "enable": True,
            "version": "stage3_0_optimizer_sequence_v1",
            "index_dir": "/tmp/index",
            "index_fingerprint": "fp123",
        },
        "scheduler_stage3_0_validation": {
            "enable": True,
            "run_at_train_start": True,
            "max_entries": 1,
            "protocols": ["Assimilation-Causal"],
            "modes": ["full"],
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


def _val_cfg():
    return train_ifwd.stage2_3_validation_cfg(_cfg())


def _run_wrapper(rows):
    return train_ifwd._run_stage2_3_validation_with_status(
        cfg=_cfg(),
        dataset=object(),
        model=object(),
        device="cpu",
        trigger_step=7,
        trigger="train_start",
        val_cfg=_val_cfg(),
        metrics_fh=rows,
        writer=None,
        convert_batch_to_minimal_format=None,
    )


def test_stage3_validation_status_success(monkeypatch):
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    def fake_run_stage2_3_validation(**kwargs):
        status_writer = kwargs["status_writer"]
        status_writer({"status": "manifest_built", "max_entries": 1, "planned_protocol_count": 1})
        status_writer({"status": "protocol_start", "entry_idx": 0, "protocol": "Assimilation-Causal", "mode": "full"})
        status_writer(
            {
                "status": "protocol_done",
                "entry_idx": 0,
                "protocol": "Assimilation-Causal",
                "mode": "full",
                "rows_emitted": 1,
            }
        )
        return [{"split": "iforward_stage2_3_validation", "protocol": "Assimilation-Causal", "mode": "full"}]

    monkeypatch.setattr(train_ifwd, "run_stage2_3_validation", fake_run_stage2_3_validation)

    out = _run_wrapper(rows)

    assert out
    status_rows = [row for row in rows if row["split"] == "iforward_stage3_0_validation_status"]
    assert [row["status"] for row in status_rows] == [
        "start",
        "manifest_built",
        "protocol_start",
        "protocol_done",
        "done",
    ]
    assert rows[-1]["split"] == "iforward_stage2_3_validation"


def test_stage3_validation_status_empty(monkeypatch):
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))
    monkeypatch.setattr(train_ifwd, "run_stage2_3_validation", lambda **kwargs: [])

    out = _run_wrapper(rows)

    assert out == []
    status_rows = [row for row in rows if row["split"] == "iforward_stage3_0_validation_status"]
    assert status_rows[-1]["status"] == "empty"
    assert status_rows[-1]["num_rows"] == 0


def test_stage3_validation_status_failed(monkeypatch):
    rows = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    def fake_run_stage2_3_validation(**kwargs):
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(train_ifwd, "run_stage2_3_validation", fake_run_stage2_3_validation)

    with pytest.raises(RuntimeError, match="validation exploded"):
        _run_wrapper(rows)

    status_rows = [row for row in rows if row["split"] == "iforward_stage3_0_validation_status"]
    assert status_rows[-1]["status"] == "failed"
    assert status_rows[-1]["exception_type"] == "RuntimeError"
    assert "validation exploded" in status_rows[-1]["exception_tail"]

