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
        "completed",
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


def test_stage3_step_end_hook_runs_validation_v4_when_due(monkeypatch):
    cfg = _cfg()
    cfg["scheduler_stage3_0_validation"]["enable"] = False
    cfg["iforward_validation_v4"] = {"enable": True, "interval_steps": 1, "run_at_train_start": False}
    calls = []
    monkeypatch.setattr(train_ifwd, "_run_validation_v4_with_status", lambda **kwargs: calls.append(dict(kwargs)) or [])

    train_ifwd._iforward_step_end_hook(
        cfg=cfg,
        dataset=object(),
        model=object(),
        device="cpu",
        trigger_step=0,
        metrics_fh=[],
    )

    assert len(calls) == 1
    assert calls[0]["trigger"] == "interval"
    assert calls[0]["trigger_step"] == 0


def test_validation_v4_image_policy_first_plan_only(tmp_path):
    cfg = _cfg()
    cfg["log_dir"] = str(tmp_path)
    cfg["iforward_validation_v4"] = {
        "enable": True,
        "report": {"images": True, "image_policy": "first_plan_only"},
    }

    assert train_ifwd._validation_v4_record_images(cfg, plan_idx=0) is True
    assert train_ifwd._validation_v4_record_images(cfg, plan_idx=1) is False

    cfg["iforward_validation_v4"]["report"]["image_policy"] = "none"
    assert train_ifwd._validation_v4_record_images(cfg, plan_idx=0) is False


def test_validation_v4_with_status_writes_compact_rows(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg["log_dir"] = str(tmp_path)
    cfg["iforward_validation_v4"] = {
        "enable": True,
        "interval_steps": 1,
        "max_entries_debug": 1,
        "repair_permutations": 3,
        "memory_ablation": ["full"],
        "report": {"images": True, "image_policy": "first_plan_only"},
    }
    rows = []
    image_flags = []
    monkeypatch.setattr(train_ifwd.base, "_write_metrics_history", lambda fh, row: fh.append(dict(row)))

    class Episode:
        protocol_name = "proto"

    class Plan:
        def __init__(self, plan_id):
            self.plan_id = plan_id
            self.episode = Episode()

    class Trace:
        events = [object(), object()]
        summary = {"current_psnr/mean": 21.0, "history_rollout_psnr/mean": 20.0, "loss/mean": 0.1}

    class Recorder:
        def __init__(self, output_dir, *, record_images=True):
            image_flags.append(bool(record_images))

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return Trace()

    monkeypatch.setattr(train_ifwd, "build_validation_v4_plans", lambda **kwargs: [Plan("p0"), Plan("p1")])
    monkeypatch.setattr(train_ifwd, "_make_validation_v4_scheduler", lambda cfg, dataset: object())
    monkeypatch.setattr(train_ifwd, "Stage3SchedulerAdapter", lambda scheduler: object())
    monkeypatch.setattr(train_ifwd, "TraceRecorder", Recorder)
    monkeypatch.setattr(train_ifwd, "IForwardRunner", Runner)
    monkeypatch.setattr(train_ifwd, "export_html_report", lambda trace, output_dir, title: str(tmp_path / "index.html"))

    out = train_ifwd._run_validation_v4_with_status(
        cfg=cfg,
        dataset=object(),
        model=object(),
        device="cpu",
        trigger_step=0,
        trigger="interval",
        metrics_fh=rows,
    )

    assert len(out) == 2
    assert image_flags == [True, False]
    assert any(row["split"] == "iforward_validation_v4_global" and row["status"] == "completed" for row in rows)
    status_rows = [row for row in rows if row["split"] == "iforward_validation_v4_status"]
    assert status_rows[0]["status"] == "start"
    assert status_rows[-1]["status"] == "completed"


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
