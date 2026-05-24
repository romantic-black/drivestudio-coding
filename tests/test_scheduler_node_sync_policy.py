from __future__ import annotations

import pytest

from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _build_scheduler_node_sync
from tools.train_minimal_streetforward_stage4_3_multi_scene_v4 import _build_scheduler_node_sync_v9_fallback


def _base_cfg(reset_policy: str | None = None) -> dict:
    mns = {"sync_with_scheduler": True}
    if reset_policy is not None:
        mns["reset_policy"] = reset_policy
    return {"scheduler_v3": {"model_node_state": mns}}


def test_scheduler_node_sync_default_block_major_uses_block_end() -> None:
    sync = _build_scheduler_node_sync(
        _base_cfg(),
        scheduler_info={"U": 1, "segment_local_step": 10, "block_order": "block_major"},
        step_events=[{"type": "block_end"}],
    )
    assert sync is not None
    assert bool(sync["reset_after_block"]) is True
    assert str(sync["reset_policy"]) == "block_end"


def test_scheduler_node_sync_default_step_major_uses_episode_end() -> None:
    sync = _build_scheduler_node_sync(
        _base_cfg(),
        scheduler_info={"U": 1, "segment_local_step": 10, "block_order": "step_major"},
        step_events=[{"type": "episode_end"}],
    )
    assert sync is not None
    assert bool(sync["reset_after_block"]) is True
    assert str(sync["reset_policy"]) == "episode_end"


def test_scheduler_node_sync_step_major_block_end_policy_is_error() -> None:
    with pytest.raises(ValueError, match="step_major with reset_policy=block_end"):
        _build_scheduler_node_sync(
            _base_cfg(reset_policy="block_end"),
            scheduler_info={"U": 1, "segment_local_step": 10, "block_order": "step_major"},
            step_events=[{"type": "block_end"}],
        )


def test_scheduler_v9_node_sync_resets_on_episode_end() -> None:
    sync = _build_scheduler_node_sync_v9_fallback(
        {
            "scheduler_v9": {
                "enable": True,
                "execution": {"reset_policy": "episode_end"},
            }
        },
        scheduler_info={"U": 1, "segment_local_step": 8, "block_order": "step_major"},
        step_events=[{"type": "episode_end"}],
    )
    assert sync is not None
    assert bool(sync["reset_after_block"]) is True
    assert str(sync["reset_policy"]) == "episode_end"
    assert str(sync["scheduler_version"]) == "v9"


def test_scheduler_v9_node_sync_never_policy_does_not_reset() -> None:
    sync = _build_scheduler_node_sync_v9_fallback(
        {
            "scheduler_v9": {
                "enable": True,
                "execution": {"reset_policy": "never"},
            }
        },
        scheduler_info={"U": 1, "segment_local_step": 8, "block_order": "step_major"},
        step_events=[{"type": "episode_end"}],
    )
    assert sync is not None
    assert bool(sync["reset_after_block"]) is False
