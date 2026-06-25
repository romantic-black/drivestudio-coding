from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from datasets.iforward_stage2_2.index_builder import build_stage2_2_index_from_dataset
from datasets.iforward_stage2_2.validation_manifest import build_stage2_2_validation_manifest
from datasets.iforward_stage2_2.validation_runner import run_stage2_2_validation, run_stage2_2_validation_manifest_only
from tests.test_iforward_stage2_2_index import _Dataset


class _ValidationDataset(_Dataset):
    def _assemble_segment_batch_from_iforward_stage2_2_request(self, *, scene_id, segment_id, plan, include_test=False):
        return {"_iforward": plan.__dict__, "request_meta": dict(plan.request_meta)}


class _FakeModel:
    training = True

    def __init__(self):
        self.calls = 0

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = bool(mode)

    def forward_rollout(self, batch, carried_state=None, ablation="full"):
        self.calls += 1
        next_state = SimpleNamespace(detach_for_next_rollout=lambda: SimpleNamespace(detach_for_next_rollout=lambda: None))
        return SimpleNamespace(
            loss=torch.tensor(1.0),
            losses={
                "current": torch.tensor(0.5),
                "in_rollout_history": torch.tensor(0.25),
                "history_damage": torch.tensor(0.0),
            },
            stats={
                "current_psnr": 20.0,
                "history_rollout_psnr": 19.0,
                "stage2_2/best_damage_loss": 0.0,
                "stage2_2/best_damage_p90": 0.0,
                "stage2_2/best_damage_max": 0.0,
                "stage2_2/bank_valid_count": 2.0,
                "stage2_2/bank_update_count": 2.0,
            },
            resolved=SimpleNamespace(
                meta=dict(batch.get("_iforward", {})),
                carry_scene_state_after_rollout=bool(batch.get("_iforward", {}).get("carry_scene_state_after_rollout", False)),
                episode_end_after_rollout=bool(batch.get("_iforward", {}).get("episode_end_after_rollout", False)),
            ),
            next_state=next_state,
        )


def test_stage2_2_validation_manifest_coverage(tmp_path):
    index = build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(40)), cfg={}, output_dir=tmp_path / "idx")
    manifest = build_stage2_2_validation_manifest(index=index, max_entries=8)
    protocols = {entry["protocol"] for entry in manifest["entries"]}
    assert {"S10-D1-Causal", "S10-D2-Causal", "S10-I123-Causal", "S10-D1-Repair", "Repeat Stability"} <= protocols


def test_stage2_2_validation_manifest_empty_failfast():
    index = build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(19)), cfg={})
    empty = index.__class__(
        metadata=index.metadata,
        segments=index.segments,
        frames=index.frames,
        windows=index.windows[:0],
        bootstrap_frames=index.bootstrap_frames,
        scene_table=index.scene_table,
    )
    with pytest.raises(ValueError, match="manifest is empty"):
        build_stage2_2_validation_manifest(index=empty)


def test_stage2_2_validation_runner_manifest_roundtrip(tmp_path):
    build_stage2_2_index_from_dataset(dataset=_Dataset(frames=range(40)), cfg={}, output_dir=tmp_path / "idx")
    cfg = {
        "scheduler_stage2_2": {"index_dir": str(tmp_path / "idx")},
        "iforward_stage2_2_validation": {"enable": True, "manifest_path": str(tmp_path / "manifest.json"), "max_entries": 3},
    }
    entries = run_stage2_2_validation_manifest_only(cfg=cfg)
    assert len(entries) >= 7
    counts = {}
    for entry in entries:
        counts[entry["protocol"]] = counts.get(entry["protocol"], 0) + 1
    assert all(count <= 3 for count in counts.values())
    with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["entries"]


def test_stage2_2_validation_runner_executes_model(tmp_path):
    ds = _ValidationDataset(frames=range(40))
    index = build_stage2_2_index_from_dataset(dataset=ds, cfg={}, output_dir=tmp_path / "idx")
    cfg = {
        "scheduler_stage2_2": {"index_dir": str(tmp_path / "idx"), "index_fingerprint": index.fingerprint},
        "iforward_stage2_2_validation": {
            "enable": True,
            "max_entries": 1,
            "protocols": ["S10-D1-Causal", "S10-D1-Repair", "Repeat Stability"],
            "modes": ["full"],
        },
    }
    model = _FakeModel()
    rows = run_stage2_2_validation(cfg=cfg, dataset=ds, model=model, device=torch.device("cpu"), trigger_step=7)
    assert model.calls < len(rows)
    assert {row["protocol"] for row in rows} == {"S10-D1-Causal", "S10-D1-Repair", "Repeat Stability"}
    assert any(row.get("scheduler_phase") == "repair" for row in rows)
    assert any(row.get("validation_rollout_kind") == "S10-D1-Causal-FinalAll10" for row in rows)
    assert any(row.get("validation_rollout_kind") == "repeat_stability_summary" for row in rows)
    assert any(int(row.get("repeat_stability_repeats", 0)) == 32 for row in rows)


def test_stage2_2_validation_order_robustness_runs_permutations(tmp_path):
    ds = _ValidationDataset(frames=range(40))
    index = build_stage2_2_index_from_dataset(dataset=ds, cfg={}, output_dir=tmp_path / "idx")
    cfg = {
        "scheduler_stage2_2": {"index_dir": str(tmp_path / "idx"), "index_fingerprint": index.fingerprint},
        "iforward_stage2_2_validation": {
            "enable": True,
            "max_entries": 1,
            "protocols": ["Order Robustness"],
            "modes": ["full"],
            "order_robustness_permutations": 2,
        },
    }
    rows = run_stage2_2_validation(cfg=cfg, dataset=ds, model=_FakeModel(), device=torch.device("cpu"), trigger_step=7)
    assert sum(row.get("validation_rollout_kind") == "order_robustness_repair" for row in rows) == 2
    assert any(row.get("validation_rollout_kind") == "order_robustness_summary" for row in rows)
