from __future__ import annotations

from models.iforward.validation_v4.metrics import summarize_legacy_rows


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
