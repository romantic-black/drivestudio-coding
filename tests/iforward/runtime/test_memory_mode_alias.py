from __future__ import annotations

import pytest

from models.iforward.runtime.event import memory_mode_to_forward_ablation, normalize_memory_mode


def test_memory_mode_aliases_normalize_to_runtime_names():
    assert normalize_memory_mode("full") == "full"
    assert normalize_memory_mode("mamba_off") == "memory_off"
    assert normalize_memory_mode("read_only") == "memory_read_only"
    assert normalize_memory_mode("shuffle_memory") == "memory_shuffle_state"
    assert normalize_memory_mode("mamba_freeze_write") == "memory_freeze_write"
    assert normalize_memory_mode("shuffle_rw_state") == "memory_shuffle_read_write_state"
    assert normalize_memory_mode("freeze_after_prefill") == "memory_freeze_after_prefill"
    assert normalize_memory_mode("wrong_parent_key_fixed") == "memory_wrong_parent_key_fixed"


def test_memory_modes_map_to_existing_forward_ablation_names():
    assert memory_mode_to_forward_ablation("full") == "full"
    assert memory_mode_to_forward_ablation("memory_off") == "mamba_off"
    assert memory_mode_to_forward_ablation("memory_read_write") == "mamba_read_write"
    assert memory_mode_to_forward_ablation("memory_shuffle_state") == "mamba_shuffle_state"
    assert memory_mode_to_forward_ablation("memory_shuffle_read_write_state") == "mamba_shuffle_read_write_state"
    assert memory_mode_to_forward_ablation("memory_freeze_after_prefill") == "mamba_freeze_write"
    assert memory_mode_to_forward_ablation("memory_wrong_parent_key_fixed") == "mamba_wrong_parent_key_fixed"


def test_invalid_memory_mode_fails_fast():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_memory_mode("memory_mystery")
