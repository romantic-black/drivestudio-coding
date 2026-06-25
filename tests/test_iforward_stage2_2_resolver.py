from __future__ import annotations

import copy

import pytest

from datasets.iforward_stage2_2.resolver import Stage22BatchResolver
from tests.test_iforward_stage2_2_scheduler import _scheduler


def test_stage2_2_resolver_metadata_masks_and_timestamp_monotonic():
    batch = _scheduler(weights={"D1": 1.0}).next_batch()
    resolved = Stage22BatchResolver().resolve(batch)
    assert resolved.scheduler_version == "iforward_stage2_2_stream10_rawframe"
    assert resolved.inner_K == 8
    assert [s.sequence_pos for s in resolved.steps if s.repeat_idx == 0] == [0, 1]
    assert resolved.steps[0].timestamp_us < resolved.steps[4].timestamp_us
    assert all(s.visit_memory_mask for s in resolved.steps)


def test_stage2_2_resolver_leakage_asserts():
    batch = _scheduler().next_batch()
    bad = copy.deepcopy(batch)
    current_ref = bad["_iforward"]["final_supervision"]["current_refs"][0]
    bad["_iforward"]["target_refs_flat"].append(tuple(current_ref))
    bad["_iforward"]["target_roles_flat"].append("final_history_replay")
    bad["_iforward"]["final_supervision"]["refs"].append(tuple(current_ref))
    bad["_iforward"]["final_supervision"]["roles"].append("final_history_replay")
    with pytest.raises(ValueError, match="history refs must be disjoint"):
        Stage22BatchResolver().resolve(bad)


def test_stage2_2_resolver_repair_no_commit():
    sched = _scheduler(repair_start=0, repair_prob=1.0, seed=4, repair_blocks=8)
    batch = None
    for _ in range(6):
        batch = sched.next_batch()
    assert batch is not None
    resolved = Stage22BatchResolver().resolve(batch)
    assert resolved.inner_K == 8
    assert len({int(s.sequence_pos) for s in resolved.steps}) == 8
    assert all(s.repair_no_commit for s in resolved.steps)
    bad = copy.deepcopy(batch)
    bad["_iforward"]["steps"][0]["temporal_commit"] = True
    with pytest.raises(ValueError, match="repair must not commit temporal"):
        Stage22BatchResolver().resolve(bad)


def test_stage2_2_resolver_rejects_nonmonotonic_causal_timestamp():
    batch = _scheduler().next_batch()
    bad = copy.deepcopy(batch)
    bad["_iforward"]["steps"][4]["timestamp_us"] = bad["_iforward"]["steps"][0]["timestamp_us"]
    with pytest.raises(ValueError, match="timestamps"):
        Stage22BatchResolver().resolve(bad)


def test_stage2_2_resolver_accepts_stress_no_commit():
    batch = _scheduler().next_batch()
    stress = copy.deepcopy(batch)
    stress["_iforward"]["scheduler_phase"] = "stress"
    stress["_iforward"]["rollout_phase"] = "repeat_stability"
    for step in stress["_iforward"]["steps"]:
        step["scheduler_phase"] = "stress"
        step["visit_kind"] = "stress"
        step["temporal_read"] = True
        step["temporal_commit"] = False
        step["physical_time_advance"] = False
        step["commit_observation_memory"] = False
        step["update_optimizer_memory"] = False
        step["frame_gap"] = 0
        step["delta_t_sec"] = 0.0
    resolved = Stage22BatchResolver().resolve(stress)
    assert {step.visit_kind for step in resolved.steps} == {"stress"}
    assert not any(step.temporal_commit for step in resolved.steps)
