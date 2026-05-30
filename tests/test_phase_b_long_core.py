from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from datasets.train_scheduler_long_phase_b import TrainSchedulerLongPhaseB
from models.streetforward.stage6_0.phase_b_long.resolver import resolve_long_phase_b_batch
from streetforward_core.data.assemblers.phase_b_long_batch_assembler import PhaseBLongBatchAssembler
from streetforward_core.data.resolvers.phase_b_long_resolver import PhaseBLongBatchResolver
from streetforward_core.protocols.phase_b_long import (
    PHASE_B_LONG_NAME,
    PHASE_B_LONG_PROTOCOL_VERSION,
    PHASE_B_LONG_SCHEDULER_VERSION,
    LongVisit,
    PhaseBLongRolloutPlan,
)
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.roles import LongRole
from streetforward_core.protocols.validators import validate_phase_b_long_plan
from streetforward_core.recipes.phase_b_long_recipe import PhaseBLongForwardOutput
from streetforward_core.train.phase_b_long_runner import PhaseBLongTrainRunner
from streetforward_core.train.stage6_phase_b_long_trainer import Stage6PhaseBLongFacadeTrainer
import tools.train_minimal_streetforward_stage4_3_multi_scene_v4 as train_base
from tools.train_minimal_streetforward_stage6_0_multi_scene_v9 import (
    _scheduler_long_phase_b_enabled,
    _validation_long_phase_b_enabled,
    build_train_scheduler_stage6_from_cfg,
    checkpoint_prefix_stage6_from_cfg,
)


def _plan(**overrides):
    values = {
        "protocol_version": PHASE_B_LONG_PROTOCOL_VERSION,
        "scheduler_version": PHASE_B_LONG_SCHEDULER_VERSION,
        "phase": PHASE_B_LONG_NAME,
        "scene_id": 1,
        "segment_id": 0,
        "episode_window_id": 7,
        "rollout_id_in_episode": 0,
        "shape_name": "r2_a2",
        "repeats_per_anchor": 2,
        "anchors_per_rollout": 2,
        "inner_K": 4,
        "anchor_frames_chronological": (10, 11),
        "anchor_frames_rollout_order": (10, 11),
        "visits": (
            LongVisit(0, 0, 10, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0),
            LongVisit(1, 0, 10, 1, 1, 0, 0, 0.333, 0.0, 0.0, 1.0),
            LongVisit(2, 1, 11, 0, 0, 1, 1, 0.667, 1.0, 1.0, 0.0),
            LongVisit(3, 1, 11, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0),
        ),
        "evidence_refs_by_step": (
            (ImageRef(10, 0),),
            (ImageRef(10, 1),),
            (ImageRef(11, 0),),
            (ImageRef(11, 1),),
        ),
        "final_history_recon_refs": (ImageRef(10, 0),),
        "final_history_nvs_refs": (ImageRef(10, 2),),
        "final_current_recon_refs": (ImageRef(11, 1),),
        "final_current_nvs_refs": (ImageRef(11, 2),),
        "source_image_refs": (ImageRef(10, 0), ImageRef(10, 1), ImageRef(11, 0), ImageRef(11, 1)),
        "target_image_refs": (ImageRef(10, 0), ImageRef(10, 2), ImageRef(11, 1), ImageRef(11, 2)),
        "target_image_roles": (
            LongRole.FINAL_HISTORY_RECON,
            LongRole.FINAL_HISTORY_NVS,
            LongRole.FINAL_CURRENT_RECON,
            LongRole.FINAL_CURRENT_NVS,
        ),
        "rigid_meta": {"has_stable_ids": False},
        "distant_meta": {"mode": "frozen_render_only"},
        "tbptt": {"enable": False},
        "meta": {
            "required_final_roles": ["final_current_recon", "final_current_nvs"],
            "max_nvs_fallback_ratio": 0.25,
            "nvs_fallback_to_evidence_cam_ratio": 0.0,
        },
    }
    values.update(overrides)
    return PhaseBLongRolloutPlan(**values)


class _Dataset:
    def __init__(self):
        self.calls = []

    def _assemble_segment_batch_from_image_refs(self, scene_id, segment_id, source_refs, target_refs, **kwargs):
        self.calls.append((scene_id, segment_id, list(source_refs), list(target_refs), dict(kwargs)))
        return {
            "request_meta": {},
            "source": {
                "frame_indices": torch.tensor([r[0] for r in source_refs]),
                "cam_indices": torch.tensor([r[1] for r in source_refs]),
            },
            "target": {
                "frame_indices": torch.tensor([r[0] for r in target_refs]),
                "cam_indices": torch.tensor([r[1] for r in target_refs]),
            },
        }


def test_phase_b_long_plan_validator_and_resolver_from_rollout_plan():
    plan = _plan()
    validate_phase_b_long_plan(plan)
    batch = PhaseBLongBatchAssembler(_Dataset()).materialize(plan)
    resolved = PhaseBLongBatchResolver().resolve(batch)
    assert resolved.inner_K == 4
    assert resolved.evidence_source_indices_by_step == ((0,), (1,), (2,), (3,))
    assert resolved.final_current_recon_target_indices == (2,)
    assert resolved.final_current_nvs_target_indices == (3,)


def test_core_and_legacy_long_resolver_parity():
    batch = PhaseBLongBatchAssembler(_Dataset()).materialize(_plan())
    core = PhaseBLongBatchResolver().resolve(batch)
    legacy = resolve_long_phase_b_batch(batch)
    assert core.inner_K == legacy.inner_K
    assert core.evidence_source_indices_by_step == tuple(tuple(x) for x in legacy.evidence_source_indices_by_step)
    assert core.final_history_recon_target_indices == tuple(legacy.final_history_recon_target_indices)
    assert core.final_history_nvs_target_indices == tuple(legacy.final_history_nvs_target_indices)
    assert core.final_current_recon_target_indices == tuple(legacy.final_current_recon_target_indices)
    assert core.final_current_nvs_target_indices == tuple(legacy.final_current_nvs_target_indices)
    assert [visit.__dict__ for visit in core.visits] == [visit.__dict__ for visit in legacy.visits]


def test_phase_b_long_resolver_rejects_request_meta_drift():
    batch = PhaseBLongBatchAssembler(_Dataset()).materialize(_plan())
    batch["request_meta"] = dict(batch["request_meta"])
    batch["request_meta"]["target_image_roles"] = [
        "final_history_recon",
        "final_history_nvs",
        "final_current_recon",
        "final_history_nvs",
    ]
    with pytest.raises(ValueError, match="target_image_roles mismatch"):
        PhaseBLongBatchResolver().resolve(batch)


def test_phase_b_long_validator_rejects_legacy_roles_and_nvs_overlap():
    with pytest.raises(ValueError, match="empty prefix_loss_refs_by_step"):
        validate_phase_b_long_plan(
            _plan(meta={**_plan().meta, "prefix_loss_refs_by_step": [[(10, 0)], [], [], []]})
        )
    with pytest.raises(ValueError, match="NVS/evidence overlap"):
        validate_phase_b_long_plan(
            _plan(
                final_current_nvs_refs=(ImageRef(11, 1),),
                target_image_refs=(ImageRef(10, 0), ImageRef(10, 2), ImageRef(11, 1), ImageRef(11, 1)),
                meta={**_plan().meta, "max_nvs_fallback_ratio": 0.0},
            )
        )


def _sidx():
    frames = [10, 20, 30, 40]
    return SimpleNamespace(
        scene_id=1,
        segment_id=0,
        num_cams=3,
        frame_indices=frames,
        keyframe_to_frames={idx: [frame] for idx, frame in enumerate(frames)},
        frame_to_keyframe={frame: idx for idx, frame in enumerate(frames)},
    )


def test_long_scheduler_emits_rollout_plan_and_advances_global_step_once():
    ds = _Dataset()
    ds._initialized = True
    ds.list_training_scene_ids = MagicMock(return_value=[1])
    ds.list_segment_ids = MagicMock(return_value=[0])
    ds.get_segment_index = MagicMock(return_value=_sidx())
    sch = TrainSchedulerLongPhaseB(
        dataset=ds,
        episode_window_cfg={"frames_per_window": 4, "min_frames_required": 2, "rollout_budget_per_episode": 2},
        rollout_shapes=[{"name": "r2_a2", "repeats_per_anchor": 2, "anchors_per_rollout": 2, "prob": 1.0}],
        anchor_sampling_cfg={
            "min_temporal_span": 1,
            "max_temporal_span": 40,
            "allow_chronological_order_prob": 1.0,
            "allow_reverse_order_prob": 0.0,
            "allow_random_order_prob": 0.0,
        },
        traversal_cfg={"segment_order": "ascending", "scene_order": "ascending"},
        evidence_cfg={"cams_per_visit": 1, "reserve_nvs_cams_per_anchor": 1, "allow_same_cam_repeat": True},
        final_supervision_cfg={
            "history_anchor_count": 1,
            "final_history_recon": {"cams_per_anchor": 1},
            "final_history_nvs": {"cams_per_anchor": 1},
            "final_current_recon": {"cams": 1},
            "final_current_nvs": {"cams": 1},
            "max_nvs_fallback_ratio": 0.0,
            "required_final_roles": ["final_current_recon", "final_current_nvs"],
        },
        include_test=False,
        fixed_scene_id=1,
        fixed_segment_id=0,
    )
    batch = sch.next_batch()
    assert sch.global_step == 1
    assert batch["rollout_plan"].scheduler_version == "long_v1"
    assert batch["request_meta"]["scheduler_version"] == "long_v1"
    assert ds.calls and ds.calls[0][4]["enforce_target0_equals_source"] is False


def test_stage6_entry_builds_long_scheduler_from_config():
    cfg = OmegaConf.create(
        {
            "data": {"train_scene_ids": [1]},
            "scheduler_long_phase_b": {
                "enable": True,
                "version": "long_v1",
                "phase": "6_0_phase_b",
                "episode_window": {"frames_per_window": 4},
                "rollout_shapes": [{"name": "r1_a1", "repeats_per_anchor": 1, "anchors_per_rollout": 1}],
                "anchor_sampling": {},
                "evidence": {"cams_per_visit": 1},
                "final_supervision": {},
                "traversal": {"fixed_scene_id": 1, "fixed_segment_id": 0},
            },
            "validation_long_phase_b": {"enable": True},
            "multi_scene": {"include_test": False},
        }
    )

    class Dataset:
        def create_train_scheduler_long_phase_b(self, **kwargs):
            return ("long", kwargs)

    out = build_train_scheduler_stage6_from_cfg(cfg, Dataset())
    assert _scheduler_long_phase_b_enabled(cfg) is True
    assert _validation_long_phase_b_enabled(cfg) is True
    assert out[0] == "long"
    assert out[1]["fixed_scene_id"] == 1
    assert out[1]["fixed_segment_id"] == 0


def test_phase_b_checkpoint_prefix_is_phase_b_long():
    assert (
        checkpoint_prefix_stage6_from_cfg(OmegaConf.create({"model": {"phase": "6_0_phase_b"}}))
        == "minimal_sf_stage6_0_phase_b_long_v1"
    )
    assert (
        checkpoint_prefix_stage6_from_cfg(OmegaConf.create({"model": {"phase": "phase_A_block_local_unroll"}}))
        == "minimal_sf_stage6_0_phase_a_v9"
    )


def test_phase_b_long_runner_logs_timing_metadata_and_only_endpoint_k_metrics():
    weight = torch.nn.Parameter(torch.tensor(1.0))

    class Optimizer:
        def zero_grad(self, set_to_none=True):
            if weight.grad is not None:
                weight.grad = None

        def step(self):
            with torch.no_grad():
                weight.sub_(0.01 * weight.grad)

    class Runtime:
        optimizer = Optimizer()
        node_states_bg = {}
        node_states_distant = {}
        node_states_rigid = {}

        def train(self):
            pass

        def _stage6_assert_required_group_grads_phase_b_long(self, legacy):
            return {
                "grad/stage6_long_vsm_sum": 1.0,
                "grad/stage6_long_offset_decoder_sum": 1.0,
            }

        def _stage6_compute_and_check_grad_norm(self):
            return weight.grad.detach().abs()

    class Recipe(torch.nn.Module):
        def forward(self, batch):
            roles = SimpleNamespace(
                inner_K=4,
                request_meta={
                    "global_step": 11,
                    "rollout_id": 22,
                    "rollout_id_in_episode": 2,
                    "episode_window_id": 7,
                    "rollout_budget_per_episode": 4,
                    "shape_name": "r2_a2",
                },
            )
            legacy = {
                "loss": weight * 2.0,
                "roles": roles,
                "stats": {"phase_b_long/custom_stat": 3.0},
                "per_step": [
                    {"k": 0, "metric": 1.0},
                    {"k": 1, "metric": 2.0},
                    {"k": 2, "metric": 3.0},
                    {"k": 3, "metric": 4.0},
                ],
                "num_targets": 3,
                "num_source_views": 4,
                "pred_rgbs": [],
                "gt_images": [],
                "node_state_bg": SimpleNamespace(means=torch.zeros(2, 3)),
                "node_state_distant": None,
                "node_state_rigid": None,
            }
            return PhaseBLongForwardOutput(loss=legacy["loss"], legacy=legacy)

    logs = PhaseBLongTrainRunner(runtime=Runtime(), recipe=Recipe()).train_step(
        {"global_step": 9},
        step=10,
        profile_phase_timing=True,
        sync_cuda_timing=False,
    )
    assert logs["train_iter"] == 10
    assert logs["rollout_step"] == 11
    assert logs["rollout_id"] == 22
    assert logs["rollout_id_in_episode"] == 2
    assert logs["episode_window_id"] == 7
    assert logs["shape_name"] == "r2_a2"
    assert logs["visit_count"] == 4
    assert logs["forward_ms"] > 0.0
    assert logs["backward_ms"] > 0.0
    assert logs["grad_check_ms"] >= 0.0
    assert logs["optimizer_ms"] >= 0.0
    assert logs["phase_b_long/k0/metric"] == 1.0
    assert logs["phase_b_long/k3/metric"] == 4.0
    assert "phase_b_long/k1/metric" not in logs
    assert "phase_b_long/k2/metric" not in logs


def test_phase_b_long_facade_train_step_passes_timing_arguments():
    class Runner:
        def train_step(self, **kwargs):
            self.kwargs = kwargs
            return {"ok": True}

    trainer = Stage6PhaseBLongFacadeTrainer.__new__(Stage6PhaseBLongFacadeTrainer)
    trainer.runner = Runner()
    batch = {"x": 1}
    out = trainer.train_step(
        batch,
        step=5,
        profile_phase_timing=True,
        sync_cuda_timing=True,
        scheduler_node_sync={"reset_after_block": False},
        runtime_policy=object(),
    )
    assert out == {"ok": True}
    assert "global_step" not in batch
    assert trainer.runner.kwargs["batch"]["global_step"] == 5
    assert trainer.runner.kwargs["step"] == 5
    assert trainer.runner.kwargs["profile_phase_timing"] is True
    assert trainer.runner.kwargs["sync_cuda_timing"] is True
    assert trainer.runner.kwargs["scheduler_node_sync"] == {"reset_after_block": False}


def test_save_train_monitor_triplets_allows_phase_b_long_non_block_indices(tmp_path, monkeypatch):
    calls = []

    def fake_save(step, pred, gt, out_dir, *, view_suffix, save_error):
        calls.append(
            {
                "step": int(step),
                "out_dir": str(out_dir),
                "view_suffix": str(view_suffix),
                "save_error": bool(save_error),
                "pred_requires_grad": bool(pred.requires_grad),
                "gt_requires_grad": bool(gt.requires_grad),
            }
        )

    monkeypatch.setattr(train_base, "_save_image_triplet", fake_save)
    raw_batch = {
        "scene_folder_name": "014",
        "target": {
            "frame_indices": torch.tensor([123]),
            "cam_indices": torch.tensor([1]),
        },
    }
    train_base._save_train_monitor_triplets(
        step=17,
        pred_rgbs=[torch.zeros(2, 2, 3)],
        gt_images=[torch.ones(2, 2, 3)],
        raw_batch=raw_batch,
        log_dir=str(tmp_path),
        block_idx_global=-1,
        scene_id_fallback=14,
        pixel_camera_ids=[0, 1, 2],
    )
    assert len(calls) == 1
    assert calls[0]["view_suffix"].startswith("b000017_sc014_v0_f00123_c1")
