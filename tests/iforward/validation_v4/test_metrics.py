from __future__ import annotations

from types import SimpleNamespace

from models.iforward.validation_v4.metrics import summarize_event_traces, summarize_legacy_rows


def test_validation_v4_metrics_summarize_retention_repair_and_memory():
    rows = [
        {"protocol": "memory_ablation/seq10/full", "mode": "full", "current_psnr": 23.0, "history_rollout_psnr": 21.0},
        {"protocol": "memory_ablation/seq10/memory_off", "mode": "memory_off", "current_psnr": 22.0, "history_rollout_psnr": 19.5},
        {"protocol": "repair_before_after/seq10", "mode": "full", "scheduler_phase": "repair", "current_psnr": 24.0, "history_rollout_psnr": 20.0},
        {"protocol": "repair_before_after/seq10", "mode": "full", "scheduler_phase": "repair", "current_psnr": 22.0, "history_rollout_psnr": 19.0},
    ]
    summary = summarize_legacy_rows(rows)

    assert summary["num_rows"] == 4
    assert summary["memory_ablation"]["memory_gain_retention"] == 1.5
    repair = [row for row in summary["protocols"] if row["protocol"] == "repair_before_after/seq10"][0]
    assert repair["repair_mean"] == 23.0
    assert repair["repair_worst"] == 22.0


def test_validation_v4_groups_uncertainty_by_distribution_phase_role_and_branch():
    event = SimpleNamespace(
        protocol="repair_before_after/seq10",
        memory_mode="full",
        event_kind="update",
        scheduler_phase="repair",
        metadata={"iforward_stage3_2": {"distribution_type": "high_block_repair"}},
        metrics={
            "current/error_uncertainty_pearson": 0.4,
            "current/error_uncertainty_spearman": 0.5,
            "current/ause": 0.2,
            "current/risk_coverage_20": 0.01,
            "current/calibration/within/error_uncertainty_spearman": 0.6,
            "current/calibration/disagreement/ause": 0.3,
            "uncertainty/bg/sigma_mean": 0.08,
            "uncertainty/bg/clamp_min_ratio": 0.0,
        },
    )
    summary = summarize_event_traces([event])
    calibration = summary["uncertainty_calibration"][0]
    state = summary["uncertainty_state"][0]
    assert calibration["distribution_type"] == "high_block_repair"
    assert calibration["scheduler_phase"] == "repair"
    assert calibration["role"] == "repair"
    assert calibration["error_uncertainty_spearman"] == 0.5
    assert calibration["calibration/within/error_uncertainty_spearman"] == 0.6
    assert calibration["calibration/disagreement/ause"] == 0.3
    assert state["branch"] == "bg"
    assert state["sigma_mean"] == 0.08
