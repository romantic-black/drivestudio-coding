from __future__ import annotations

import pytest

from models.iforward.runtime.event import memory_mode_to_forward_ablation, normalize_memory_mode


def test_memory_mode_aliases_normalize_to_runtime_names():
    assert normalize_memory_mode("full") == "full"
    assert normalize_memory_mode("mamba_off") == "memory_off"
    assert normalize_memory_mode("read_only") == "memory_read_only"
    assert normalize_memory_mode("shuffle_memory") == "memory_shuffle_state"
    assert normalize_memory_mode("mamba_freeze_write") == "memory_freeze_write"


def test_memory_modes_map_to_existing_forward_ablation_names():
    assert memory_mode_to_forward_ablation("full") == "full"
    assert memory_mode_to_forward_ablation("memory_off") == "mamba_off"
    assert memory_mode_to_forward_ablation("memory_read_write") == "mamba_read_write"
    assert memory_mode_to_forward_ablation("memory_shuffle_state") == "mamba_shuffle_state"


def test_invalid_memory_mode_fails_fast():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_memory_mode("memory_mystery")
