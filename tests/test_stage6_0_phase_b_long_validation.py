from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from datasets.validation_long_phase_b import (
    build_validation_plan_long_phase_b,
    materialize_validation_long_phase_b_batch,
)
from models.streetforward import validation_long_phase_b_runner as vlr
from models.streetforward.stage6_0.phase_b_long.types import LongVSMReadPack


def _cfg():
    return {
        "data": {"eval_scene_ids": [1]},
        "validation_long_phase_b": {
            "enable": True,
            "version": "long_v1",
            "phase": "6_0_phase_b",
            "segment": {
                "policy": "first",
                "fixed_segment_ids": [],
                "max_segments_per_scene": 1,
                "max_frames_per_segment": 6,
                "target_frame_stride": 1,
            },
            "evidence": {
                "interval_T_values": [2],
                "repeats_per_evidence_frame": 2,
                "evidence_cams_per_frame": 1,
                "cam_policy": "fixed_round_robin",
            },
            "order": {"primary": "chronological", "extra_orders": ["reverse", "random_seeded"]},
            "render_eval": {
                "reconstruction": {"enable": True},
                "nvs_same_frame": {"enable": True, "heldout_cams_per_evidence_frame": 2},
                "temporal_nvs": {"enable": True, "eval_non_evidence_frames": True, "cams_per_non_evidence_frame": 1},
                "segment_all": {"max_render_refs": 64},
            },
        },
    }


class _Dataset:
    def __init__(self):
        self.calls = []

    def list_segment_ids(self, scene_id):
        assert int(scene_id) == 1
        return [0]

    def get_segment_index(self, scene_id, segment_id):
        assert (int(scene_id), int(segment_id)) == (1, 0)
        return SimpleNamespace(scene_id=1, segment_id=0, frame_indices=[0, 1, 2, 3, 4, 5], num_cams=3)

    def _assemble_segment_batch_from_image_refs(self, scene_id, segment_id, source_refs, target_refs, **kwargs):
        self.calls.append((scene_id, segment_id, list(source_refs), list(target_refs), dict(kwargs)))
        return {"request_meta": {}}


def test_validation_long_plan_uses_t_stride_orders_and_split_roles():
    ds = _Dataset()
    plan = build_validation_plan_long_phase_b(dataset=ds, cfg=_cfg(), eval_scene_ids=[1])
    assert plan.interval_T_values == [2]
    assert plan.orders == ["chronological", "reverse", "random_seeded"]
    assert len(plan.specs) == 3

    spec = next(x for x in plan.specs if x.order == "chronological")
    assert spec.evidence_frames == [0, 2, 4]
    assert len(spec.visits) == 6
    assert spec.request_meta["inner_K"] == 6
    assert all(len(step) == 1 for step in spec.evidence_refs_by_step)
    evidence_set = {tuple(step[0]) for step in spec.evidence_refs_by_step}
    evidence_cams_by_frame = {}
    for frame_idx, cam_idx in evidence_set:
        evidence_cams_by_frame.setdefault(int(frame_idx), set()).add(int(cam_idx))
    assert all(len(cams) == 1 for cams in evidence_cams_by_frame.values())
    assert set(spec.target_image_roles) <= {
        "final_history_recon",
        "final_history_nvs",
        "final_current_recon",
        "final_current_nvs",
    }
    assert "final_current_nvs" not in spec.target_image_roles
    assert spec.validation_buckets["reconstruction"]
    assert spec.validation_buckets["nvs_same_frame"]
    assert len(spec.validation_buckets["nvs_same_frame"]) == 6
    assert not (set(spec.validation_buckets["nvs_same_frame"]) & evidence_set)
    assert spec.validation_buckets["temporal_nvs"]
    assert spec.validation_buckets["segment_all"]
    assert spec.request_meta["required_final_roles"] == []
    assert "step_block_indices" not in spec.request_meta

    batch = materialize_validation_long_phase_b_batch(ds, spec)
    meta = batch["request_meta"]
    assert meta["validation_version"] == "long_v1"
    assert meta["validation_interval_T"] == 2
    assert ds.calls and ds.calls[-1][4]["enforce_target0_equals_source"] is False


def test_validation_long_bucket_cap_filters_materialized_buckets():
    cfg = _cfg()
    cfg["validation_long_phase_b"]["order"] = {"primary": "chronological", "extra_orders": []}
    cfg["validation_long_phase_b"]["render_eval"]["segment_all"]["max_render_refs"] = 2
    ds = _Dataset()
    plan = build_validation_plan_long_phase_b(dataset=ds, cfg=cfg, eval_scene_ids=[1])
    spec = plan.specs[0]
    target_set = set(spec.target_image_refs)
    for refs in spec.validation_buckets.values():
        assert set(refs) <= target_set
    dropped = spec.request_meta["validation_bucket_dropped_counts"]
    assert dropped["segment_all"] > 0
    assert sum(v for k, v in dropped.items() if k != "segment_all") > 0


def test_validation_long_runner_reports_lpips_and_vsm_ablation(monkeypatch):
    batch = {
        "request_meta": {
            "validation_interval_T": 2,
            "target_image_refs": [(0, 0), (0, 1), (1, 0)],
            "validation_buckets": {
                "reconstruction": [(0, 0)],
                "nvs_same_frame": [(0, 1)],
                "temporal_nvs": [(1, 0)],
                "segment_all": [(0, 0), (0, 1), (1, 0)],
            },
            "validation_bucket_requested_counts": {"segment_all": 3},
            "validation_bucket_materialized_counts": {"segment_all": 3},
            "validation_bucket_dropped_counts": {"segment_all": 0},
        },
        "targets": [{}, {}, {}],
    }
    seen_ablations = []

    def fake_inference(model, batch_arg, *, ablation):
        seen_ablations.append(ablation)
        return {
            "roles": SimpleNamespace(rigid_meta={}, step_chronological_ranks=[]),
            "base_state": object(),
            "offset": SimpleNamespace(rigid_frame_snapshots={}),
        }

    def fake_render(model, *, target_indices, lpips_model, **kwargs):
        n = len(list(target_indices))
        return {
            "num_refs": float(n),
            "psnr": 10.0 + float(n),
            "l1": 0.1,
            "ssim": 0.9,
            "lpips": 0.2,
            "rigid_fallback_rows": 0.0,
        }, lpips_model

    monkeypatch.setattr(vlr, "run_long_phase_b_inference", fake_inference)
    monkeypatch.setattr(vlr, "_render_metrics_for_indices", fake_render)

    out = vlr.validate_long_phase_b(object(), batch, ablations=("normal", "zero_vsm", "shuffle_vsm"))
    assert seen_ablations == ["normal", "zero_vsm", "shuffle_vsm"]
    assert out["val_long/reconstruction_lpips"] == 0.2
    assert out["val_long/zero_vsm/temporal_nvs_lpips"] == 0.2
    assert out["val_long/shuffle_vsm/segment_all_lpips"] == 0.2
    assert "val_long/zero_vsm_gain/segment_all_psnr" in out
    assert out["val_long/materialized_segment_all_refs"] == 3.0


def test_direct_validate_long_phase_b_default_ablations_match_config_names(monkeypatch):
    batch = {
        "request_meta": {
            "validation_interval_T": 2,
            "target_image_refs": [(0, 0)],
            "validation_buckets": {
                "reconstruction": [(0, 0)],
                "nvs_same_frame": [],
                "temporal_nvs": [],
                "segment_all": [(0, 0)],
            },
        },
        "targets": [{}],
    }
    seen_ablations = []

    def fake_inference(model, batch_arg, *, ablation):
        seen_ablations.append(ablation)
        return {
            "roles": SimpleNamespace(rigid_meta={}, step_chronological_ranks=[]),
            "base_state": object(),
            "offset": SimpleNamespace(rigid_frame_snapshots={}),
        }

    def fake_render(model, *, target_indices, lpips_model, **kwargs):
        return {
            "num_refs": float(len(list(target_indices))),
            "psnr": 10.0,
            "l1": 0.1,
            "ssim": 0.9,
            "lpips": 0.2,
            "rigid_fallback_rows": 0.0,
        }, lpips_model

    monkeypatch.setattr(vlr, "run_long_phase_b_inference", fake_inference)
    monkeypatch.setattr(vlr, "_render_metrics_for_indices", fake_render)

    vlr.validate_long_phase_b(object(), batch)
    assert tuple(seen_ablations) == (
        "normal",
        "zero_vsm",
        "zero_read_keep_seen",
        "shuffle_vsm",
        "zero_delta",
    )
    assert vlr.DEFAULT_LONG_VSM_ABLATIONS == tuple(seen_ablations)


def test_validation_zero_read_can_zero_seen_or_keep_seen():
    read = LongVSMReadPack(
        bg=torch.ones(2, 4),
        seen_bg=torch.full((2, 1), 3.0),
        rigid=torch.ones(1, 5),
        rigid_indices=torch.tensor([1]),
        rigid_seen=torch.full((1, 1), 2.0),
        rigid_stable_mask=torch.tensor([True]),
        distant=torch.ones(1, 3),
        distant_indices=torch.tensor([0]),
        distant_seen=torch.full((1, 1), 4.0),
    )

    zero_seen = vlr._zero_read(read, zero_seen=True)
    assert torch.count_nonzero(zero_seen.bg) == 0
    assert torch.count_nonzero(zero_seen.seen_bg) == 0
    assert zero_seen.rigid is not None and torch.count_nonzero(zero_seen.rigid) == 0
    assert zero_seen.rigid_seen is not None and torch.count_nonzero(zero_seen.rigid_seen) == 0
    assert zero_seen.distant is not None and torch.count_nonzero(zero_seen.distant) == 0
    assert zero_seen.distant_seen is not None and torch.count_nonzero(zero_seen.distant_seen) == 0

    keep_seen = vlr._zero_read(read, zero_seen=False)
    assert torch.count_nonzero(keep_seen.bg) == 0
    assert torch.equal(keep_seen.seen_bg, read.seen_bg)
    assert keep_seen.rigid_seen is not None and torch.equal(keep_seen.rigid_seen, read.rigid_seen)
    assert keep_seen.distant_seen is not None and torch.equal(keep_seen.distant_seen, read.distant_seen)


def test_validation_cell_vsm_init_passes_batch_for_persistent_aabb():
    source = inspect.getsource(vlr.run_long_phase_b_inference)
    assert 'stage6_phase_b_long_vsm_type", "streaming_selective_ssm"' in source
    assert 'init_kwargs["batch"] = batch' in source
