from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from datasets.train_scheduler_v9 import StepPlanV9, ViewSetRolloutBatchV9
from models.streetforward.minimal_trainer_stage5_4 import MinimalStreetForwardStage5_4
from models.streetforward.minimal_trainer_stage6_0 import MinimalStreetForwardStage6_0
from models.streetforward.node_states import NodeStateBackground
from models.streetforward.stage6_0.posterior_updater import BranchDelta, DeltaPack
from streetforward_core.data.resolvers.phase_a_resolver import PhaseABatchResolver
from streetforward_core.data.schedulers.legacy_v9_phase_a_adapter import (
    convert_v9_phase_a_plan,
    legacy_v9_batch_to_phase_a_plan,
)
from streetforward_core.legacy.stage6_facade import Stage6LegacyFacade
from streetforward_core.protocols.refs import ImageRef
from streetforward_core.protocols.rollout import PHASE_A_NAME, PHASE_A_PROTOCOL_VERSION, PhaseALocalUnrollPlan, RolloutStep
from streetforward_core.protocols.validators import validate_phase_a_plan
from streetforward_core.recipes.phase_a_recipe import PhaseARecipe
from streetforward_core.train.runner import PhaseATrainRunner
from streetforward_core.train.stage6_phase_a_trainer import Stage6PhaseATrainer


def _plan() -> PhaseALocalUnrollPlan:
    return PhaseALocalUnrollPlan(
        protocol_version=PHASE_A_PROTOCOL_VERSION,
        phase=PHASE_A_NAME,
        scene_id=1,
        segment_id=2,
        episode_id=3,
        num_cams=2,
        inner_K=2,
        steps=(
            RolloutStep(
                step_idx=0,
                evidence_refs=(ImageRef(10, 0), ImageRef(10, 1)),
                block_loss_refs=(ImageRef(10, 0),),
            ),
            RolloutStep(
                step_idx=1,
                evidence_refs=(ImageRef(10, 0), ImageRef(10, 1)),
                block_loss_refs=(ImageRef(10, 0),),
                nearby_loss_refs=(ImageRef(11, 0),),
            ),
        ),
        source_keyframe_idx=4,
        block_idx=5,
    )


def _batch(*, rollout_plan=True):
    meta = {
        "scheduler_version": "v9",
        "scheduler_phase": PHASE_A_NAME,
        "assembly_mode": "image_ref_v9",
        "scene_id": 1,
        "segment_id": 2,
        "episode_id": 3,
        "num_cams": 2,
        "inner_K": 2,
        "evidence_refs_by_step": [[(10, 0), (10, 1)], [(10, 0), (10, 1)]],
        "block_loss_refs_by_step": [[(10, 0)], [(10, 0)]],
        "nearby_loss_refs_by_step": [[], [(11, 0)]],
        "prefix_loss_refs_by_step": [[], []],
        "query_label_refs": [],
        "aux_loss_refs": [],
        "source_image_refs": [(10, 0), (10, 1)],
        "target_image_refs": [(10, 0), (11, 0)],
        "target_image_roles": ["block_loss", "nearby_loss"],
    }
    batch = {
        "scene_id": 1,
        "segment_id": 2,
        "request_meta": meta,
        "_scheduler_v9": {
            "scheduler_version": "v9",
            "phase": PHASE_A_NAME,
            "scene_id": 1,
            "segment_id": 2,
            "episode_id": 3,
            "num_cams": 2,
            "inner_K": 2,
            "steps": [
                {"step_idx": 0, "source_keyframe_idx": 4, "block_idx": 5},
                {"step_idx": 1, "source_keyframe_idx": 4, "block_idx": 5},
            ],
        },
        "source_views": [object(), object()],
        "targets": [{"gt_image": torch.zeros(1, 1, 3)}, {"gt_image": torch.zeros(1, 1, 3)}],
    }
    if rollout_plan:
        batch["rollout_plan"] = _plan()
    return batch


def _node_state(n=2) -> NodeStateBackground:
    return NodeStateBackground(
        means=torch.zeros(n, 3),
        scales_log=torch.zeros(n, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        sh_dc=torch.zeros(n, 3),
        sh_rest=torch.zeros(n, 3, 3),
    )


def _delta(local_state, value):
    n = int(local_state.bg.means.shape[0])
    ref = local_state.bg.means
    zeros3 = ref.new_zeros(n, 3)
    return DeltaPack(
        bg=BranchDelta(
            means=ref.new_ones(n, 3) * value,
            scales_log=zeros3,
            quat_axis_angle=zeros3,
            opacity_logit=ref.new_zeros(n, 1),
            sh=ref.new_zeros(n, 12),
            hidden=ref.new_zeros(n, int(local_state.bg.hidden.shape[1])),
            confidence=ref.new_zeros(n, 1),
            noop=ref.new_zeros(n, 1),
        )
    )


def test_phase_a_plan_validation_rejects_bad_shapes_and_leakage():
    validate_phase_a_plan(_plan())
    bad_len = PhaseALocalUnrollPlan(**{**_plan().__dict__, "inner_K": 3})
    with pytest.raises(ValueError, match="len\\(steps\\)"):
        validate_phase_a_plan(bad_len)
    bad_empty = PhaseALocalUnrollPlan(
        **{
            **_plan().__dict__,
            "steps": (
                RolloutStep(step_idx=0, evidence_refs=(), block_loss_refs=()),
                _plan().steps[1],
            ),
        }
    )
    with pytest.raises(ValueError, match="requires evidence_refs"):
        validate_phase_a_plan(bad_empty)
    bad_leak = PhaseALocalUnrollPlan(
        **{
            **_plan().__dict__,
            "steps": (
                _plan().steps[0],
                RolloutStep(
                    step_idx=1,
                    evidence_refs=(ImageRef(10, 0),),
                    block_loss_refs=(ImageRef(10, 0),),
                    nearby_loss_refs=(ImageRef(10, 0),),
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="leaked"):
        validate_phase_a_plan(bad_leak)
    assert sorted([ImageRef(2, 1), ImageRef(1, 2)]) == [ImageRef(1, 2), ImageRef(2, 1)]


def test_v9_phase_a_adapter_converts_plan():
    step0 = StepPlanV9(
        step_idx=0,
        source_keyframe_idx=4,
        source_frame_idx=10,
        block_idx=5,
        evidence_refs=[(10, 0), (10, 1)],
        block_loss_refs=[(10, 0)],
        nearby_loss_refs=[],
        prefix_loss_refs=[],
        query_label_refs=[],
        aux_loss_refs=[],
        evidence_frame_indices=[10],
        loss_frame_indices=[10],
        nearby_frame_indices=[],
        query_frame_indices=[],
    )
    step1 = StepPlanV9(
        **{**step0.__dict__, "step_idx": 1, "nearby_loss_refs": [(11, 0)], "nearby_frame_indices": [11]}
    )
    v9 = ViewSetRolloutBatchV9(
        scheduler_version="v9",
        phase=PHASE_A_NAME,
        scene_id=1,
        segment_id=2,
        episode_id=3,
        episode_start_keyframe_pos=9,
        keyframe_window=[4, 5],
        frame_chain=[10, 11],
        num_cams=2,
        inner_K=2,
        steps=[step0, step1],
        evidence_refs_by_step=[step0.evidence_refs, step1.evidence_refs],
        block_loss_refs_by_step=[step0.block_loss_refs, step1.block_loss_refs],
        nearby_loss_refs_by_step=[step0.nearby_loss_refs, step1.nearby_loss_refs],
        prefix_loss_refs_by_step=[[], []],
        query_label_refs=[],
        aux_loss_refs=[],
    )
    plan = convert_v9_phase_a_plan(v9)
    assert plan.protocol_version == PHASE_A_PROTOCOL_VERSION
    assert plan.source_keyframe_idx == 4
    assert plan.block_idx == 5
    assert plan.steps[1].nearby_loss_refs == (ImageRef(11, 0),)


def test_phase_a_resolver_supports_rollout_plan_and_legacy_fallback():
    resolved = PhaseABatchResolver().resolve(_batch(rollout_plan=True))
    assert resolved.inner_K == 2
    assert resolved.evidence_source_indices_by_step == ((0, 1), (0, 1))
    assert resolved.block_target_indices_by_step == ((0,), (0,))
    assert resolved.nearby_target_indices_by_step == ((), (1,))

    fallback = PhaseABatchResolver().resolve(_batch(rollout_plan=False))
    assert fallback.plan.source_keyframe_idx == 4
    assert fallback.nearby_target_indices_by_step == ((), (1,))

    bad = _batch(rollout_plan=True)
    bad["request_meta"]["target_image_roles"] = ["nearby_loss", "nearby_loss"]
    with pytest.raises(ValueError, match="mapped to target role"):
        PhaseABatchResolver().resolve(bad)

    bad_prefix = _batch(rollout_plan=False)
    bad_prefix["request_meta"]["prefix_loss_refs_by_step"][0] = [(12, 0)]
    with pytest.raises(ValueError, match="prefix_loss"):
        legacy_v9_batch_to_phase_a_plan(bad_prefix)


class _FakeFacade:
    hidden_dim = 2
    block_weight = 1.0
    step_gamma = 1.0
    block_mask_policy = "none"
    nearby_mask_policy = "none"

    def nearby_weight(self, *, global_step, k, K):
        return 0.5 if int(k) == int(K) - 1 else 0.0

    def delta_regularization_kwargs(self):
        return {
            "weight": 0.0,
            "opacity_delta_l2_weight": 0.0,
            "sh_delta_l2_weight": 0.0,
            "scale_barrier_weight": 0.0,
            "scale_log_min": -10.0,
            "scale_log_max": 4.0,
        }


class _FakeNodeProvider:
    def __init__(self, bg):
        self.bg = bg

    def get_or_init(self, batch):
        _ = batch
        return self.bg, None, None


class _FakeMeasurement:
    def observe(self, *, local_state, batch, source_indices, source_frame_idx):
        _ = local_state, batch
        return {"source_indices": list(source_indices), "source_frame_idx": int(source_frame_idx)}


class _FakeEventUpdate:
    def __init__(self, param):
        self.param = param

    def encode_and_update(self, *, local_state, measurement):
        _ = measurement
        delta = _delta(local_state, self.param)
        return local_state.apply_delta(delta), delta, {}


class _FakeRenderer:
    def render_loss(self, *, local_state, batch, target_indices, mask_policy, pred_rgbs_out=None, gt_images_out=None):
        _ = batch, mask_policy
        if not target_indices:
            return local_state.bg.means.new_tensor(0.0), {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
        loss = local_state.bg.means.pow(2).mean()
        if pred_rgbs_out is not None:
            pred_rgbs_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        if gt_images_out is not None:
            gt_images_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        return loss, {
            "num_refs": float(len(target_indices)),
            "num_metric_refs": float(len(target_indices)),
            "metric_valid": 1.0,
            "valid_ratio": 1.0,
            "skipped_no_valid_pixels": 0.0,
            "psnr": 10.0,
            "l1": float(loss.detach().item()),
            "ssim": 0.0,
        }


def _fake_recipe(bg, param):
    return PhaseARecipe(
        facade=_FakeFacade(),
        node_state_provider=_FakeNodeProvider(bg),
        measurement=_FakeMeasurement(),
        event_update=_FakeEventUpdate(param),
        renderer=_FakeRenderer(),
    )


def test_phase_a_recipe_forward_is_pure_and_finite():
    bg = _node_state()
    param = torch.tensor(0.1, requires_grad=True)
    recipe = _fake_recipe(bg, param)
    out = recipe(_batch(rollout_plan=True))
    assert torch.isfinite(out.loss)
    assert len(out.per_step) == 2
    assert out.per_step[0]["loss_nearby"] == 0.0
    assert out.per_step[1]["loss_nearby"] > 0.0
    assert len(out.pred_rgbs) == 2
    assert torch.allclose(bg.means, torch.zeros_like(bg.means))


def test_phase_a_runner_steps_optimizer_logs_aliases_and_detached_writeback():
    class Runtime:
        def __init__(self):
            self.param = nn.Parameter(torch.tensor(0.1))
            self.optimizer = torch.optim.SGD([self.param], lr=0.01)
            self.stage6_writeback_policy = "block_end_detached"
            self.node_states_bg = {}
            self.node_states_distant = {}
            self.node_states_rigid = {}
            self.reset_called = False

        def train(self, mode=True):
            self.training = bool(mode)

        def reset_node_state(self):
            self.reset_called = True

        def _stage6_assert_required_group_grads(self, out):
            assert out["roles"].inner_K == 2
            assert self.param.grad is not None
            return {"grad/stage6_posterior_updater_sum": float(self.param.grad.detach().abs().item())}

        def _stage6_compute_and_check_grad_norm(self):
            return self.param.grad.detach().abs()

    bg = _node_state()
    runtime = Runtime()
    runner = PhaseATrainRunner(runtime=runtime, recipe=_fake_recipe(bg, runtime.param))
    logs = runner.train_step(_batch(rollout_plan=True), step=10, scheduler_node_sync={"reset_after_block": True})
    assert "phaseA/loss_total" in logs
    assert "phase_a/loss_total" in logs
    assert logs["phase_a/k1/loss_nearby"] > 0.0
    assert logs["grad/posterior_updater"] > 0.0
    assert logs["state/num_bg"] == 2
    assert runtime.reset_called is True
    assert runtime.param.grad is None
    for value in bg.__dict__.values():
        if torch.is_tensor(value):
            assert value.requires_grad is False
    assert not torch.allclose(bg.means, torch.zeros_like(bg.means))


def test_phase_a_facade_recipe_matches_legacy_fake_forward():
    runtime = MinimalStreetForwardStage6_0.__new__(MinimalStreetForwardStage6_0)
    nn.Module.__init__(runtime)
    runtime.stage6_hidden_dim = 2
    runtime.stage6_block_weight = 1.0
    runtime.stage6_step_gamma = 1.0
    runtime.stage6_block_mask_policy = "none"
    runtime.stage6_nearby_mask_policy = "none"
    runtime.stage6_delta_l2_weight = 0.0
    runtime.stage6_opacity_delta_l2_weight = 0.0
    runtime.stage6_sh_delta_l2_weight = 0.0
    runtime.stage6_scale_barrier_weight = 0.0
    runtime.stage6_scale_log_min = -10.0
    runtime.stage6_scale_log_max = 4.0
    runtime.stage6_nearby_enable = True
    runtime.stage6_nearby_final_step_only = True
    runtime.stage6_nearby_weight = 0.5
    runtime.stage6_nearby_warmup_steps = 1
    bg = _node_state()

    def get_or_init(batch):
        _ = batch
        return bg, None, None

    def observe(*, local_state, batch, source_indices, source_frame_idx):
        _ = local_state, batch, source_indices
        return {"source_frame_idx": int(source_frame_idx)}

    def encode_and_update(*, local_state, measurement):
        _ = measurement
        delta = _delta(local_state, 0.1)
        return local_state.apply_delta(delta), delta, {}

    def build_event(*, local_state, measurement):
        _ = local_state
        return {"measurement": measurement}

    def apply_event_update(*, local_state, event, ctx_vsm=None):
        _ = ctx_vsm
        return encode_and_update(local_state=local_state, measurement=event["measurement"])

    def render_loss(*, local_state, batch, target_indices, mask_policy, pred_rgbs_out=None, gt_images_out=None):
        _ = batch, mask_policy
        if not target_indices:
            return local_state.bg.means.new_tensor(0.0), {
                "num_refs": 0.0,
                "num_metric_refs": 0.0,
                "metric_valid": 0.0,
                "valid_ratio": 0.0,
                "skipped_no_valid_pixels": 0.0,
            }
        loss = local_state.bg.means[:, 0].mean()
        if pred_rgbs_out is not None:
            pred_rgbs_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        if gt_images_out is not None:
            gt_images_out.append(local_state.bg.means.new_zeros(1, 1, 3))
        return loss, {
            "num_refs": float(len(target_indices)),
            "num_metric_refs": float(len(target_indices)),
            "metric_valid": 1.0,
            "valid_ratio": 1.0,
            "skipped_no_valid_pixels": 0.0,
            "psnr": 12.0,
            "l1": float(loss.detach().item()),
            "ssim": 0.0,
        }

    runtime._get_or_init_node_states_bg_rigid_distant = get_or_init
    runtime._observe_v4_measurement = observe
    runtime._encode_and_update = encode_and_update
    runtime._build_stage6_event_from_measurement = build_event
    runtime._apply_event_update = apply_event_update
    runtime._render_loss_for_indices = render_loss

    batch = _batch(rollout_plan=True)
    batch["global_step"] = 10
    legacy = MinimalStreetForwardStage6_0._forward_phase_a(runtime, batch)
    recipe = PhaseARecipe(facade=Stage6LegacyFacade(runtime))(batch).to_legacy_dict()
    assert torch.allclose(legacy["loss"], recipe["loss"])
    assert legacy["per_step"] == recipe["per_step"]
    assert len(legacy["pred_rgbs"]) == len(recipe["pred_rgbs"])
    assert len(legacy["gt_images"]) == len(recipe["gt_images"])


def test_stage6_phase_a_trainer_is_not_stage5_subclass():
    assert not issubclass(Stage6PhaseATrainer, MinimalStreetForwardStage5_4)
