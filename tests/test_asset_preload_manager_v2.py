from __future__ import annotations

import time
from threading import Lock

from datasets.asset_preload_manager_v2 import (
    PRELOAD_TASK_SEGMENT_STATIC,
    AssetPreloadManagerV2,
    parse_preload_cfg_v2,
)


class _DummyDataset:
    def __init__(self) -> None:
        self.calls = []
        self.lock = Lock()

    def _append(self, x):
        with self.lock:
            self.calls.append(x)

    def _preload_worker_scene_meta(self, scene_id, segment_id, meta):
        self._append(("scene_meta", scene_id, segment_id, dict(meta)))

    def _preload_worker_segment_static(self, scene_id, segment_id, meta):
        self._append(("segment_static", scene_id, segment_id, dict(meta)))

    def _preload_worker_view_meta(self, scene_id, segment_id, image_ref, meta):
        self._append(("view_meta", scene_id, segment_id, tuple(image_ref), dict(meta)))

    def _preload_worker_view_pack(self, scene_id, segment_id, image_ref, meta):
        self._append(("view_pack", scene_id, segment_id, tuple(image_ref), dict(meta)))


def _cfg_dict():
    return {
        "enable": True,
        "num_workers": 1,
        "max_pending_tasks": 8,
        "dedupe_tasks": True,
        "drop_stale_hints": True,
        "warm_scene_meta": True,
        "warm_segment_static": True,
        "warm_next_block_exact": True,
        "warm_test_refs": True,
        "warm_episode_source_superset": True,
        "enable_view_pack_cache": True,
        "stats_log_interval_steps": 0,
    }


def test_parse_preload_cfg_v2_requires_keys():
    cfg = _cfg_dict()
    cfg.pop("warm_scene_meta")
    try:
        parse_preload_cfg_v2(cfg)
        raise AssertionError("expected parse_preload_cfg_v2 to fail on missing key")
    except ValueError as exc:
        assert "missing keys" in str(exc)


def test_asset_preload_manager_v2_dedupes_identical_tasks():
    ds = _DummyDataset()
    mgr = AssetPreloadManagerV2(ds, parse_preload_cfg_v2(_cfg_dict()))
    mgr.start()
    mgr.submit_segment_static(-2, 1, 0, meta={"scope": "x"})
    mgr.submit_segment_static(-2, 1, 0, meta={"scope": "x"})
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if len(ds.calls) >= 1:
            break
        time.sleep(0.01)
    mgr.stop()
    segment_calls = [x for x in ds.calls if x[0] == "segment_static"]
    assert len(segment_calls) == 1


def test_asset_preload_manager_v2_high_priority_eviction():
    cfg = _cfg_dict()
    cfg["max_pending_tasks"] = 1
    ds = _DummyDataset()
    mgr = AssetPreloadManagerV2(ds, parse_preload_cfg_v2(cfg))
    # do not start worker yet; inspect heap behavior first
    mgr.submit_segment_static(10, 1, 0, meta={"kind": "low"})
    mgr.submit_segment_static(0, 1, 1, meta={"kind": "high"})
    assert len(mgr._heap) == 1
    assert int(mgr._heap[0][4]) == PRELOAD_TASK_SEGMENT_STATIC
    assert int(mgr._heap[0][3]) == 1

