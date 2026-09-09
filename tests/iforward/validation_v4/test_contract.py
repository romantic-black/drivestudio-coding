from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.iforward.runtime.event import EpisodeSpec, UpdateEvent
from models.iforward.runtime.plan import EpisodePlan
from models.iforward.runtime.trace import EpisodeTrace, EventTrace
from models.iforward.validation_v4.contract import (
    assert_validation_contract,
    build_validation_contract,
    write_validation_contract,
)
from models.iforward.versions import IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION


CODEC_SCHEMA = "legacy17d_plus_geometry8d_residual_v1"


def _cfg(*, stage34: bool = True):
    version = (
        IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION
        if stage34
        else "stage3_3_observation_feedback"
    )
    return {
        "model": {
            "iforward": {
                "version": version,
                "training_variant": version,
                "parent_spatial": {
                    "param_codec": {
                        "schema": CODEC_SCHEMA if stage34 else "legacy17d_v1",
                    }
                },
            }
        }
    }


def _plan() -> EpisodePlan:
    episode = EpisodeSpec(
        scene_id=131,
        segment_id=1,
        sequence_id=0,
        frame_ids=(0, 1),
        frame_positions=(0, 1),
        cam_ids=(0, 0),
        protocol_name="assimilation_timeline/seq10/entry0",
    )
    rollout = SimpleNamespace(steps=[SimpleNamespace(), SimpleNamespace()], actual_inner_K=2)
    event = UpdateEvent(
        event_id="update0",
        kind="observe_update",
        rollout_plan=rollout,
    )
    return EpisodePlan(
        plan_id="plan0",
        version="iforward_episode_plan_v1",
        episode=episode,
        events=(event,),
        expected_outputs=("trace.jsonl", "summary.json"),
    )


def _trace(*, nan: bool = False, legacy: bool = False) -> EpisodeTrace:
    metrics = {
        "loss": float("nan") if nan else 1.0,
        "iforward/stage3_4/model_update_count": 1.0,
        "iforward/parent_optimizer_gdkv/global_update_step": 1.0,
        "feedback/functional_parent/grad_active": 0.0,
        "feedback/functional_parent/forward_only": 1.0,
    }
    if legacy:
        metrics["iforward/biggs/exact_refresh_count"] = 0.0
    event = EventTrace(
        plan_id="plan0",
        event_id="update0",
        event_kind="observe_update",
        event_idx=0,
        protocol="assimilation_timeline/seq10/entry0",
        memory_mode="full",
        metrics=metrics,
    )
    return EpisodeTrace(
        plan_id="plan0",
        protocol="assimilation_timeline/seq10/entry0",
        scene_id=131,
        segment_id=1,
        events=[event],
        summary={"loss/mean": metrics["loss"]},
    )


def _artifacts(tmp_path):
    plan_dir = tmp_path / "0000_plan0"
    plan_dir.mkdir()
    for name in ("plan.json", "trace.jsonl", "summary.json", "html_summary.json", "index.html"):
        (plan_dir / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    return plan_dir


def _checkpoint():
    return {
        "iforward_version": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
        "training_variant": IFORWARD_STAGE3_4_FUNCTIONAL_PARENTGS_LIFT_VERSION,
        "parent_codec_schema": CODEC_SCHEMA,
    }


def test_stage34_validation_contract_accepts_complete_graph_free_trace(tmp_path):
    plan = _plan()
    trace = _trace()
    plan_dir = _artifacts(tmp_path)
    contract = build_validation_contract(
        output_dir=tmp_path,
        cfg=_cfg(),
        plans=[plan],
        traces=[trace],
        plan_dirs=[plan_dir],
        parameter_versions_before={"weight": 2},
        parameter_versions_after={"weight": 2},
        checkpoint_payload=_checkpoint(),
    )

    assert contract["status"] == "passed"
    assert contract["checks"]["k2_update_ancestor_observed"]["passed"] is True
    assert contract["checks"]["causal_localgs_and_gdkv_advanced"]["passed"] is True
    assert contract["checks"]["functional_parent_grad_inactive"]["passed"] is True
    path = write_validation_contract(contract, tmp_path)
    assert path.endswith("validation_contract.json")
    assert_validation_contract(contract)


def test_stage34_validation_contract_reports_all_isolation_failures(tmp_path):
    plan = _plan()
    trace = _trace(nan=True, legacy=True)
    trace.events[0].metrics["feedback/functional_parent/grad_active"] = 1.0
    trace.events[0].metrics["feedback/functional_parent/forward_only"] = 0.0
    plan_dir = _artifacts(tmp_path)
    contract = build_validation_contract(
        output_dir=tmp_path,
        cfg=_cfg(),
        plans=[plan],
        traces=[trace],
        plan_dirs=[plan_dir],
        parameter_versions_before={"weight": 2},
        parameter_versions_after={"weight": 3},
        checkpoint_payload=_checkpoint(),
    )

    assert {
        "all_trace_numbers_finite",
        "model_parameters_unchanged",
        "functional_parent_grad_inactive",
        "functional_parent_forward_only",
        "legacy_parent_runtime_metrics_absent",
    } <= set(contract["failures"])
    with pytest.raises(RuntimeError, match="validation contract failed"):
        assert_validation_contract(contract)


def test_validation_contract_remains_compatible_with_pre_stage34_runs(tmp_path):
    plan = _plan()
    trace = _trace()
    trace.events[0].metrics = {"loss": 1.0}
    plan_dir = _artifacts(tmp_path)
    contract = build_validation_contract(
        output_dir=tmp_path,
        cfg=_cfg(stage34=False),
        plans=[plan],
        traces=[trace],
        plan_dirs=[plan_dir],
        parameter_versions_before={},
        parameter_versions_after={},
    )

    assert contract["status"] == "passed"
    assert contract["stage3_4_required"] is False
    assert "functional_parent_grad_inactive" not in contract["checks"]


def test_stage34_contract_does_not_count_gdkv_write_skipped_as_state_advance(tmp_path):
    plan = _plan()
    trace = _trace()
    trace.events[0].metrics.pop("iforward/parent_optimizer_gdkv/global_update_step")
    trace.events[0].metrics["iforward/parent_optimizer_gdkv/write_skipped"] = 1.0
    plan_dir = _artifacts(tmp_path)
    contract = build_validation_contract(
        output_dir=tmp_path,
        cfg=_cfg(),
        plans=[plan],
        traces=[trace],
        plan_dirs=[plan_dir],
        parameter_versions_before={},
        parameter_versions_after={},
        checkpoint_payload=_checkpoint(),
    )

    assert contract["checks"]["causal_localgs_and_gdkv_advanced"]["passed"] is False
