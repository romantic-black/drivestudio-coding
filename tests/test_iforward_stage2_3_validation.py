from __future__ import annotations

from types import SimpleNamespace

import torch

from datasets.iforward_stage2_3.resolver import Stage23BatchResolver
from datasets.iforward_stage2_3.validation_runner import (
    run_stage2_3_validation,
    run_stage2_3_validation_manifest_only,
    stage2_3_validation_cfg,
)
from tests.test_iforward_stage2_3_scheduler import _Dataset


def _cfg():
    return {
        "scheduler_v3": {
            "time": {"allow_synthetic_timestamp": True},
            "bootstrap": {"end_step": 0},
            "sequence": {
                "min_frames": 8,
                "max_frames": 10,
                "min_unique_keyframes": 3,
                "min_frame_span": 8,
                "max_frame_span": 30,
            },
            "assimilation": {"repeat_pairs": {"4,4": 1.0}},
            "repair": {"enable": False},
        },
        "validation_v3": {
            "max_entries": 1,
            "protocols": [
                "Assimilation-Causal",
                "Assimilation-Causal-FinalAll",
                "Repair-B6R1",
                "Repair-B8R1",
                "Repair-B6R2",
                "Repair-B10",
                "Repeat Stability",
                "Order Robustness",
            ],
            "repeat_stability_repeats": [4, 8],
            "order_robustness_permutations": 2,
        },
    }


class _State:
    def __init__(self, value=0):
        self.value = int(value)

    def detach_for_next_rollout(self):
        return _State(self.value)


class _FakeModel:
    def __init__(self):
        self.training = True
        self.calls = []
        self.resolver = Stage23BatchResolver()

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def forward_rollout(self, batch, carried_state=None, ablation="full"):
        resolved = self.resolver.resolve(batch)
        self.calls.append(
            {
                "phase": str(resolved.meta.get("scheduler_phase", "")),
                "positions": list(resolved.meta.get("rollout_positions", [])),
                "ablation": str(ablation),
                "carried": carried_state is not None,
            }
        )
        idx = len(self.calls)
        current = torch.tensor(1.0 / float(idx + 1), dtype=torch.float32)
        return SimpleNamespace(
            loss=current,
            losses={
                "current": current,
                "in_rollout_history": current * 0.5,
                "history_damage": current * 0.25,
            },
            stats={
                "current_psnr": float(20.0 + idx),
                "history_rollout_psnr": float(18.0 + idx),
                "stage2_3/best_damage_loss": float(current.item()),
                "stage2_3/best_damage_p90": float(current.item() * 0.5),
                "stage2_3/best_damage_max": float(current.item()),
                "stage2_3/bank_valid_count": float(idx),
                "stage2_3/bank_update_count": float(len(resolved.meta.get("rollout_positions", []))),
            },
            pred_rgbs=[torch.zeros(4, 4, 3)],
            gt_images=[torch.ones(4, 4, 3)],
            image_roles=["final_current_recon"],
            resolved=resolved,
            next_state=_State(idx),
        )


class _FakeWriter:
    def __init__(self):
        self.images = []

    def add_image(self, tag, image, step):
        self.images.append((str(tag), tuple(image.shape), int(step)))


def test_stage2_3_validation_manifest_smoke():
    rows = run_stage2_3_validation_manifest_only(cfg=_cfg(), dataset=_Dataset(), max_entries=2)
    assert len(rows) == 2
    assert all(row["scheduler_phase"] == "assimilation" for row in rows)
    assert all("rollout_positions" in row for row in rows)


def test_stage2_3_validation_mamba_ablation_modes_are_stage2_3_names():
    val = stage2_3_validation_cfg(
        {
            "validation_v3": {
                "protocols": {"mamba_ablation": ["off", "read_only", "read_write", "shuffled", "freeze_write"]}
            }
        }
    )
    assert val["protocols"] == ["Mamba Ablation"]
    assert val["mamba_ablation_modes"] == [
        "mamba_off",
        "mamba_read_only",
        "mamba_read_write",
        "mamba_shuffle_state",
        "mamba_freeze_write",
    ]
    val2 = stage2_3_validation_cfg({"validation_v3": {"protocols": {"assimilation": True}}})
    assert "Assimilation-Causal-FinalAll" in val2["protocols"]


def test_stage2_3_validation_runner_calls_fake_model_and_protocols():
    model = _FakeModel()
    writer = _FakeWriter()
    rows = run_stage2_3_validation(cfg=_cfg(), dataset=_Dataset(), model=model, device="cpu", trigger_step=7, writer=writer)
    assert rows
    assert rows[0]["split"] == "iforward_stage2_3_validation"
    assert rows[0]["trigger_step"] == 7
    assert model.calls
    protocols = {str(row.get("protocol", "")) for row in rows}
    assert {
        "Assimilation-Causal-FinalAll",
        "Repair-B6R1",
        "Repair-B8R1",
        "Repair-B6R2",
        "Repair-B10",
        "Repeat Stability",
        "Order Robustness",
    } <= protocols
    kinds = {str(row.get("validation_rollout_kind", "")) for row in rows}
    assert "final_all" in kinds
    assert "final_all_summary" in kinds
    assert "retention_curve_point" in kinds
    assert "retention_curve_summary" in kinds
    assert "repeat_stability_summary" in kinds
    assert "order_robustness_summary" in kinds
    assert any(kind.startswith("repair_b6r1") for kind in kinds)
    repair_b6r1 = [
        row
        for row in rows
        if row.get("protocol") == "Repair-B6R1" and row.get("validation_rollout_kind") == "repair_b6r1"
    ]
    assert len(repair_b6r1) == 2
    assert len({tuple(row.get("repair_positions", [])) for row in repair_b6r1}) == 2
    final_all = [row for row in rows if row.get("validation_rollout_kind") == "final_all"]
    assert final_all
    assert final_all[0]["scheduler_phase"] == "final_all"
    assert len(final_all[0]["final_all_positions"]) >= 8
    assert any(call["phase"] == "repair" for call in model.calls)
    assert any(call["phase"] == "final_all" for call in model.calls)
    assert writer.images
    assert all(step == 7 for _, _, step in writer.images)
