"""
Background preload worker for MultiSceneDatasetV3 (image-ref view packs).

Single-threaded worker; priority queue; dedupe and stale-hint dropping.
Stale hints use dataset.set_preload_active_scope() (scheduler), not last batch fetch.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from datasets.multi_scene_dataset_v3 import MultiSceneDatasetV3

logger = logging.getLogger(__name__)

ImageRef = Tuple[int, int]

PRIORITY_NEXT_BLOCK_EXACT = 0
PRIORITY_TEST_REFS = 1
PRIORITY_EPISODE_SUPERSET = 10


@dataclass
class LoadedViewPack:
    image: Tensor
    extrinsic: Tensor
    intrinsic: Tensor
    depth: Tensor
    sky_mask: Optional[Tensor]
    viewdirs: Optional[Tensor]
    egocar_mask: Optional[Tensor]
    frame_idx: int
    cam_idx: int


def _tensor_to_cpu_for_cache(t: Tensor, *, pin_memory: bool) -> Tensor:
    y = t.detach().cpu()
    if pin_memory:
        try:
            y = y.pin_memory()
        except RuntimeError:
            pass
    return y


def _optional_tensor_to_cpu_for_cache(x: Optional[Tensor], *, pin_memory: bool) -> Optional[Tensor]:
    if x is None:
        return None
    return _tensor_to_cpu_for_cache(x, pin_memory=pin_memory)


def dict_to_loaded_view_pack(pack: Dict[str, Any], *, pin_memory: bool) -> LoadedViewPack:
    return LoadedViewPack(
        image=_tensor_to_cpu_for_cache(pack["image"], pin_memory=pin_memory),
        extrinsic=_tensor_to_cpu_for_cache(pack["extrinsic"], pin_memory=pin_memory),
        intrinsic=_tensor_to_cpu_for_cache(pack["intrinsic"], pin_memory=pin_memory),
        depth=_tensor_to_cpu_for_cache(pack["depth"], pin_memory=pin_memory),
        sky_mask=_optional_tensor_to_cpu_for_cache(pack.get("sky_mask"), pin_memory=pin_memory),
        viewdirs=_optional_tensor_to_cpu_for_cache(pack.get("viewdirs"), pin_memory=pin_memory),
        egocar_mask=_optional_tensor_to_cpu_for_cache(pack.get("egocar_mask"), pin_memory=pin_memory),
        frame_idx=int(pack["frame_idx"]),
        cam_idx=int(pack["cam_idx"]),
    )


def loaded_view_pack_to_device(pack: LoadedViewPack, device: torch.device) -> Dict[str, Any]:
    def mv(t: Optional[Tensor]) -> Optional[Tensor]:
        if t is None:
            return None
        x = t.to(device=device, non_blocking=True)
        # CPU cache may share storage with returned tensors; clone so callers cannot corrupt the cache.
        if device.type == "cpu":
            x = x.clone()
        return x

    return {
        "image": mv(pack.image),
        "extrinsic": mv(pack.extrinsic),
        "intrinsic": mv(pack.intrinsic),
        "depth": mv(pack.depth),
        "sky_mask": mv(pack.sky_mask),
        "viewdirs": mv(pack.viewdirs),
        "egocar_mask": mv(pack.egocar_mask),
        "frame_idx": pack.frame_idx,
        "cam_idx": pack.cam_idx,
    }


@dataclass(frozen=True)
class PreloadRuntimeConfig:
    enable: bool
    num_workers: int
    max_pending_tasks: int
    enable_view_pack_cache: bool
    view_cache_max_items_total: int
    view_cache_max_items_per_scene: int
    view_cache_device: str
    drop_stale_hints: bool
    dedupe_tasks: bool
    warm_next_block_exact: bool
    warm_test_refs: bool
    warm_episode_source_superset: bool


_REQUIRED_PRELOAD_KEYS = (
    "enable",
    "num_workers",
    "max_pending_tasks",
    "enable_view_pack_cache",
    "view_cache_max_items_total",
    "view_cache_max_items_per_scene",
    "view_cache_device",
    "drop_stale_hints",
    "dedupe_tasks",
    "warm_next_block_exact",
    "warm_test_refs",
    "warm_episode_source_superset",
)


def parse_preload_cfg(raw: Optional[Dict[str, Any]]) -> Optional[PreloadRuntimeConfig]:
    if raw is None:
        return None
    if not bool(raw.get("enable")):
        return None
    missing = [k for k in _REQUIRED_PRELOAD_KEYS if k not in raw]
    if missing:
        raise ValueError(f"data.preload.enable is true but missing keys: {missing}")
    vw = str(raw["view_cache_device"])
    if vw not in ("cpu", "cpu_pinned"):
        raise ValueError(f"data.preload.view_cache_device must be 'cpu' or 'cpu_pinned', got {vw!r}")
    nw = int(raw["num_workers"])
    if nw != 1:
        raise ValueError(f"data.preload.num_workers must be 1 for now, got {nw}")
    return PreloadRuntimeConfig(
        enable=True,
        num_workers=nw,
        max_pending_tasks=int(raw["max_pending_tasks"]),
        enable_view_pack_cache=bool(raw["enable_view_pack_cache"]),
        view_cache_max_items_total=int(raw["view_cache_max_items_total"]),
        view_cache_max_items_per_scene=int(raw["view_cache_max_items_per_scene"]),
        view_cache_device=vw,
        drop_stale_hints=bool(raw["drop_stale_hints"]),
        dedupe_tasks=bool(raw["dedupe_tasks"]),
        warm_next_block_exact=bool(raw["warm_next_block_exact"]),
        warm_test_refs=bool(raw["warm_test_refs"]),
        warm_episode_source_superset=bool(raw["warm_episode_source_superset"]),
    )


def pin_memory_from_cfg(cfg: PreloadRuntimeConfig) -> bool:
    return cfg.view_cache_device == "cpu_pinned"


class DatasetPreloadManager:
    def __init__(
        self,
        dataset: "MultiSceneDatasetV3",
        cfg: PreloadRuntimeConfig,
    ) -> None:
        self._dataset = dataset
        self._cfg = cfg
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._heap: List[Tuple[int, int, int, int, ImageRef]] = []
        self._seq = itertools.count()
        self._pending_dedupe: Set[Tuple[int, int, int, int]] = set()
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_dropped_stale": 0,
            "tasks_dropped_queue_full": 0,
            "tasks_evicted_for_admission": 0,
            "tasks_dropped_scene_unloading": 0,
            "views_loaded": 0,
            "cache_hits_worker": 0,
            "tasks_failed": 0,
            "total_latency_ms": 0.0,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DatasetPreloadWorker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def pop_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
            self._stats = {
                "tasks_completed": 0,
                "tasks_dropped_stale": 0,
                "tasks_dropped_queue_full": 0,
                "tasks_evicted_for_admission": 0,
                "tasks_dropped_scene_unloading": 0,
                "views_loaded": 0,
                "cache_hits_worker": 0,
                "tasks_failed": 0,
                "total_latency_ms": 0.0,
            }
        return out

    def submit_image_ref(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
    ) -> None:
        key = (int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
        if self._dataset._preload_view_key_is_cached(key):
            return
        with self._lock:
            if self._cfg.dedupe_tasks and key in self._pending_dedupe:
                return
            if len(self._heap) >= self._cfg.max_pending_tasks:
                worst_pr = max(x[0] for x in self._heap)
                if priority >= worst_pr:
                    self._bump_stat("tasks_dropped_queue_full", 1)
                    return
                victim_idx = next(i for i, x in enumerate(self._heap) if x[0] == worst_pr)
                victim = self._heap.pop(victim_idx)
                heapq.heapify(self._heap)
                vk = (victim[2], victim[3], victim[4][0], victim[4][1])
                self._pending_dedupe.discard(vk)
                self._bump_stat("tasks_evicted_for_admission", 1)
            seq = next(self._seq)
            heapq.heappush(self._heap, (priority, seq, key[0], key[1], image_ref))
            self._pending_dedupe.add(key)

    def clear_pending_for_scene(self, scene_id: int) -> None:
        with self._lock:
            kept: List[Tuple[int, int, int, int, ImageRef]] = []
            for item in self._heap:
                pr, sq, sc, sg, ref = item
                if int(sc) == int(scene_id):
                    key = (int(sc), int(sg), int(ref[0]), int(ref[1]))
                    self._pending_dedupe.discard(key)
                    continue
                kept.append(item)
            self._heap = kept
            heapq.heapify(self._heap)

    def _bump_stat(self, name: str, delta: float) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + delta

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not self._heap:
                    item = None
                else:
                    item = heapq.heappop(self._heap)
            if item is None:
                time.sleep(0.01)
                continue
            pr, seq, sc, sg, ref = item
            key = (int(sc), int(sg), int(ref[0]), int(ref[1]))
            with self._lock:
                self._pending_dedupe.discard(key)

            self._dataset._preload_begin_scene_work(int(sc))
            try:
                skip = False
                if self._cfg.drop_stale_hints:
                    active_sc = getattr(self._dataset, "_preload_active_scene_id", None)
                    active_sg = getattr(self._dataset, "_preload_active_segment_id", None)
                    if active_sc is not None and active_sg is not None:
                        if int(sc) != int(active_sc) or int(sg) != int(active_sg):
                            self._bump_stat("tasks_dropped_stale", 1)
                            skip = True
                if not skip and self._dataset._preload_should_abort_for_unload(int(sc)):
                    self._bump_stat("tasks_dropped_scene_unloading", 1)
                    skip = True

                if skip:
                    continue

                t0 = time.perf_counter()
                status = "failed"
                try:
                    status = self._dataset._preload_worker_load_view_pack(int(sc), int(sg), ref)
                except Exception as exc:
                    logger.debug("preload worker load failed: %s", exc, exc_info=True)
                    status = "failed"
                dt_ms = (time.perf_counter() - t0) * 1000.0

                if status == "loaded":
                    self._bump_stat("tasks_completed", 1)
                    self._bump_stat("views_loaded", 1)
                    self._bump_stat("total_latency_ms", dt_ms)
                elif status == "cache_hit":
                    self._bump_stat("tasks_completed", 1)
                    self._bump_stat("cache_hits_worker", 1)
                    self._bump_stat("total_latency_ms", dt_ms)
                elif status == "failed":
                    self._bump_stat("tasks_failed", 1)
                else:
                    self._bump_stat("tasks_completed", 1)
            finally:
                self._dataset._preload_end_scene_work(int(sc))
