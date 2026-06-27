from __future__ import annotations

import pytest

from datasets.iforward_stage2_3.resolver import Stage23BatchResolver
from tests.test_iforward_stage2_3_scheduler import _scheduler


def test_stage2_3_resolver_accepts_scheduler_batch():
    sched = _scheduler(seed=2)
    resolved = Stage23BatchResolver().resolve(sched.next_batch())
    assert resolved.scheduler_version == "iforward_2_3_scheduler_v3_optimizer_mamba"
    assert resolved.steps[0].repeat_budget >= 1
    assert resolved.steps[0].optimizer_memory_write is True


def test_stage2_3_resolver_rejects_bootstrap_mamba_write():
    sched = _scheduler(bootstrap_end=1, seed=2)
    batch = sched.next_batch()
    batch["_iforward"]["steps"][0]["optimizer_memory_write"] = True
    batch["_iforward"]["steps"][0]["update_optimizer_memory"] = True
    batch["request_meta"]["iforward"]["steps"][0]["optimizer_memory_write"] = True
    batch["request_meta"]["iforward"]["steps"][0]["update_optimizer_memory"] = True
    with pytest.raises(ValueError, match="bootstrap"):
        Stage23BatchResolver().resolve(batch)


def test_stage2_3_resolver_rejects_assimilation_missing_write():
    sched = _scheduler(seed=3)
    batch = sched.next_batch()
    batch["_iforward"]["steps"][0]["optimizer_memory_write"] = False
    batch["_iforward"]["steps"][0]["update_optimizer_memory"] = False
    batch["request_meta"]["iforward"]["steps"][0]["optimizer_memory_write"] = False
    batch["request_meta"]["iforward"]["steps"][0]["update_optimizer_memory"] = False
    with pytest.raises(ValueError, match="assimilation"):
        Stage23BatchResolver().resolve(batch)


def test_stage2_3_resolver_rejects_missing_rollout_positions_metadata():
    sched = _scheduler(seed=4)
    batch = sched.next_batch()
    batch["_iforward"].pop("rollout_positions", None)
    batch["request_meta"]["iforward"].pop("rollout_positions", None)
    with pytest.raises(ValueError, match="rollout_positions"):
        Stage23BatchResolver().resolve(batch)
