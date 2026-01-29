"""
Golden baseline regression test for StreetForwardTrainer.

Requires: opencv-python (cv2), and when running full replay: torchsparse, gsplat,
and other project deps. Use the project conda env (e.g. drivestudio-new) to run:
  pytest tests/test_streetforward_golden_baseline.py -v -s

Uses CUDA baseline by default when CUDA is available and the cuda baseline file exists.
Batch cache is taken from baseline meta when present, or from STREETFORWARD_BATCH_CACHE.
"""
import os
from pathlib import Path

import pytest
import torch

from tools.record_streetforward_golden_baseline import run_recording
from utils.streetforward_baseline import compare_step, load_baseline, load_config

try:
    import cv2  # noqa: F401
    _cv2_available = True
except ImportError:
    _cv2_available = False


def _default_baseline_path() -> Path:
    # Prefer CUDA baseline when CUDA is available so device is cuda, not cpu.
    cuda_path = Path("docs/trainers/golden/streetforward_golden_multi_scene_seed42_steps8_cuda.json")
    cpu_path = Path("docs/trainers/golden/streetforward_golden_multi_scene_seed42_steps8_cpu.json")
    if torch.cuda.is_available() and cuda_path.exists():
        return cuda_path
    return cpu_path


def test_streetforward_golden_baseline(tmp_path):
    if not _cv2_available:
        pytest.skip("cv2 is required to replay the baseline but is not installed.")

    baseline_path = Path(os.getenv("STREETFORWARD_GOLDEN_BASELINE", _default_baseline_path()))
    if not baseline_path.exists():
        pytest.skip(f"Golden baseline not found at {baseline_path}. Run tools/record_streetforward_golden_baseline.py to generate it.")

    baseline = load_baseline(baseline_path)
    meta = baseline.get("meta", {})

    config_path = meta.get("config_path") or os.getenv("STREETFORWARD_GOLDEN_CONFIG", "configs/streetforward/multi_scene.yaml")
    config_path = Path(config_path)
    if not config_path.exists():
        pytest.skip(f"Config path {config_path} does not exist.")

    cfg = load_config(str(config_path))
    data_root = getattr(cfg.data, "data_root", None)
    if data_root and not Path(data_root).exists():
        pytest.skip(f"Data root {data_root} not found; baseline cannot be replayed.")

    device_str = meta.get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("Baseline was recorded on CUDA but CUDA is not available.")
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    scheduler_kwargs = meta.get("scheduler", {})
    batch_cache_env = os.getenv("STREETFORWARD_BATCH_CACHE", None)
    batch_cache_path = Path(batch_cache_env) if batch_cache_env else None
    if batch_cache_path is None and meta.get("batch_cache_path"):
        candidate = Path(meta["batch_cache_path"])
        if candidate.exists():
            batch_cache_path = candidate
    if batch_cache_path is not None and not batch_cache_path.exists():
        pytest.skip(f"Batch cache not found at {batch_cache_path}. Generate it or unset STREETFORWARD_BATCH_CACHE.")

    current_path = tmp_path / "current_baseline.json"
    recorded_path = run_recording(
        config_path=str(config_path),
        steps=int(meta.get("num_steps", 1)),
        seed=int(meta.get("seed", 42)),
        device=device,
        batches_per_segment=int(scheduler_kwargs.get("batches_per_segment", 2)),
        segment_order=scheduler_kwargs.get("segment_order", "sequential"),
        scene_order=scheduler_kwargs.get("scene_order", "sequential"),
        shuffle_segments=bool(scheduler_kwargs.get("shuffle_segments", False)),
        preload_next_scene=bool(scheduler_kwargs.get("preload_next_scene", False)),
        output=current_path,
        batch_cache_path=batch_cache_path,
        harvest_if_missing=False,
        plan_scenes=2,
        plan_segments=2,
        plan_batches=2,
    )

    current = load_baseline(recorded_path)
    assert len(current["per_step"]) == len(baseline["per_step"]), "Step count mismatch between baseline and replay."

    for b_step, c_step in zip(baseline["per_step"], current["per_step"]):
        ok, msg = compare_step(b_step, c_step)
        assert ok, msg
