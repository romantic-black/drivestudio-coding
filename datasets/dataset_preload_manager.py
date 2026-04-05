"""
Background preload worker for MultiSceneDatasetV3 (segment-static + image-ref view packs + overlap pairs).

Single-threaded worker; priority queue; dedupe and stale-hint dropping.
Stale hints use dataset.set_preload_active_scope() / clear_preload_active_scope().
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

if TYPE_CHECKING:
    from datasets.multi_scene_dataset_v3 import MultiSceneDatasetV3

logger = logging.getLogger(__name__)

ImageRef = Tuple[int, int]

PRELOAD_TASK_VIEW = 0
PRELOAD_TASK_SEGMENT_STATIC = 1
PRELOAD_TASK_OVERLAP_PAIR = 2

PRIORITY_SEGMENT_STATIC = -2
PRIORITY_NEXT_BLOCK_EXACT = 0
PRIORITY_TEST_REFS = 1
PRIORITY_EPISODE_SUPERSET = 10

HeapItem = Tuple[int, int, int, int, int, ImageRef, Dict[str, Any]]


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


DedupeKey = Union[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int, str, int]]


def _dedupe_key_from_heap_item(item: HeapItem) -> DedupeKey:
    _pr, _sq, sc, sg, kind, ref, meta = item
    if int(kind) == PRELOAD_TASK_SEGMENT_STATIC:
        return (int(sc), int(sg), -1, -1)
    if int(kind) == PRELOAD_TASK_OVERLAP_PAIR:
        tr = meta["tgt_rep"]
        return (
            int(sc),
            int(sg),
            int(ref[0]),
            int(ref[1]),
            int(tr[0]),
            int(tr[1]),
            str(meta["mode"]),
            int(meta["point_sample_size"]),
        )
    return (int(sc), int(sg), int(ref[0]), int(ref[1]))


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
    warm_segment_static: bool
    warm_segment_pointcloud: bool
    warm_next_block_exact: bool
    warm_test_refs: bool
    warm_episode_source_superset: bool
    warm_overlap_pairs_episode_superset: bool
    warm_overlap_pairs_next_block_exact: bool
    stats_log_interval_steps: int


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
    "warm_segment_static",
    "warm_segment_pointcloud",
    "warm_next_block_exact",
    "warm_test_refs",
    "warm_episode_source_superset",
    "warm_overlap_pairs_episode_superset",
    "warm_overlap_pairs_next_block_exact",
    "stats_log_interval_steps",
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
        warm_segment_static=bool(raw["warm_segment_static"]),
        warm_segment_pointcloud=bool(raw["warm_segment_pointcloud"]),
        warm_next_block_exact=bool(raw["warm_next_block_exact"]),
        warm_test_refs=bool(raw["warm_test_refs"]),
        warm_episode_source_superset=bool(raw["warm_episode_source_superset"]),
        warm_overlap_pairs_episode_superset=bool(raw["warm_overlap_pairs_episode_superset"]),
        warm_overlap_pairs_next_block_exact=bool(raw["warm_overlap_pairs_next_block_exact"]),
        stats_log_interval_steps=int(raw["stats_log_interval_steps"]),
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
        self._heap: List[HeapItem] = []
        self._seq = itertools.count()
        self._queued_dedupe: Set[DedupeKey] = set()
        self._inflight_dedupe: Set[DedupeKey] = set()
        self._lock = threading.Condition(threading.Lock())
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_dropped_stale": 0,
            "tasks_dropped_queue_full": 0,
            "tasks_evicted_for_admission": 0,
            "tasks_dropped_scene_unloading": 0,
            "views_loaded": 0,
            "cache_hits_worker": 0,
            "segment_static_completed": 0,
            "tasks_failed": 0,
            "total_latency_ms": 0.0,
            "overlap_pairs_loaded": 0,
            "overlap_pair_cache_hits_worker": 0,
            "overlap_pairs_failed": 0,
            "overlap_pair_total_latency_ms": 0.0,
        }

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                return
            self._thread = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DatasetPreloadWorker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("preload worker did not stop within %.2fs; keeping thread handle", timeout)
            return
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
                "segment_static_completed": 0,
                "tasks_failed": 0,
                "total_latency_ms": 0.0,
                "overlap_pairs_loaded": 0,
                "overlap_pair_cache_hits_worker": 0,
                "overlap_pairs_failed": 0,
                "overlap_pair_total_latency_ms": 0.0,
            }
        return out

    def submit_image_ref(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        image_ref: ImageRef,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = (int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
        if self._dataset._preload_view_key_is_cached(key):
            return
        m = dict(meta) if meta is not None else {}
        with self._lock:
            if self._cfg.dedupe_tasks and (key in self._queued_dedupe or key in self._inflight_dedupe):
                return
            if len(self._heap) >= self._cfg.max_pending_tasks:
                worst_pr = max(x[0] for x in self._heap)
                if priority >= worst_pr:
                    self._bump_stat("tasks_dropped_queue_full", 1)
                    return
                victim_idx = next(i for i, x in enumerate(self._heap) if x[0] == worst_pr)
                victim = self._heap.pop(victim_idx)
                heapq.heapify(self._heap)
                vk = _dedupe_key_from_heap_item(victim)
                self._queued_dedupe.discard(vk)
                self._bump_stat("tasks_evicted_for_admission", 1)
            seq = next(self._seq)
            heapq.heappush(
                self._heap,
                (priority, seq, key[0], key[1], PRELOAD_TASK_VIEW, image_ref, m),
            )
            self._queued_dedupe.add(key)
            self._lock.notify()

    def submit_segment_static(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        *,
        meta: Dict[str, Any],
    ) -> None:
        key = (int(scene_id), int(segment_id), -1, -1)
        if self._dataset._preload_segment_static_redundant(int(scene_id), int(segment_id)):
            return
        with self._lock:
            if self._cfg.dedupe_tasks and (key in self._queued_dedupe or key in self._inflight_dedupe):
                return
            if len(self._heap) >= self._cfg.max_pending_tasks:
                worst_pr = max(x[0] for x in self._heap)
                if priority >= worst_pr:
                    self._bump_stat("tasks_dropped_queue_full", 1)
                    return
                victim_idx = next(i for i, x in enumerate(self._heap) if x[0] == worst_pr)
                victim = self._heap.pop(victim_idx)
                heapq.heapify(self._heap)
                vk = _dedupe_key_from_heap_item(victim)
                self._queued_dedupe.discard(vk)
                self._bump_stat("tasks_evicted_for_admission", 1)
            seq = next(self._seq)
            heapq.heappush(
                self._heap,
                (
                    priority,
                    seq,
                    key[0],
                    key[1],
                    PRELOAD_TASK_SEGMENT_STATIC,
                    (-1, -1),
                    dict(meta),
                ),
            )
            self._queued_dedupe.add(key)
            self._lock.notify()

    def submit_overlap_pair(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        src_rep_image_ref: ImageRef,
        tgt_rep_image_ref: ImageRef,
        *,
        mode: str,
        point_sample_size: int,
        meta: Dict[str, Any],
    ) -> None:
        sc, sg = int(scene_id), int(segment_id)
        src_t = (int(src_rep_image_ref[0]), int(src_rep_image_ref[1]))
        tgt_t = (int(tgt_rep_image_ref[0]), int(tgt_rep_image_ref[1]))
        m = dict(meta)
        m["tgt_rep"] = tgt_t
        m["mode"] = str(mode)
        m["point_sample_size"] = int(point_sample_size)
        dkey: DedupeKey = (sc, sg, src_t[0], src_t[1], tgt_t[0], tgt_t[1], str(mode), int(point_sample_size))
        if self._dataset.is_pair_score_cached(sc, sg, src_t, tgt_t, str(mode), int(point_sample_size)):
            return
        with self._lock:
            if self._cfg.dedupe_tasks and (dkey in self._queued_dedupe or dkey in self._inflight_dedupe):
                return
            if len(self._heap) >= self._cfg.max_pending_tasks:
                worst_pr = max(x[0] for x in self._heap)
                if priority >= worst_pr:
                    self._bump_stat("tasks_dropped_queue_full", 1)
                    return
                victim_idx = next(i for i, x in enumerate(self._heap) if x[0] == worst_pr)
                victim = self._heap.pop(victim_idx)
                heapq.heapify(self._heap)
                vk = _dedupe_key_from_heap_item(victim)
                self._queued_dedupe.discard(vk)
                self._bump_stat("tasks_evicted_for_admission", 1)
            seq = next(self._seq)
            heapq.heappush(
                self._heap,
                (priority, seq, sc, sg, PRELOAD_TASK_OVERLAP_PAIR, src_t, m),
            )
            self._queued_dedupe.add(dkey)
            self._lock.notify()

    def clear_pending_for_scene(self, scene_id: int) -> None:
        with self._lock:
            kept: List[HeapItem] = []
            for item in self._heap:
                pr, sq, sc, sg, kind, ref, meta = item
                if int(sc) == int(scene_id):
                    self._queued_dedupe.discard(_dedupe_key_from_heap_item(item))
                    continue
                kept.append(item)
            self._heap = kept
            heapq.heapify(self._heap)

    def _bump_stat(self, name: str, delta: float) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + delta

    def _should_skip_stale(self, scene_id: int, segment_id: int) -> bool:
        if not self._cfg.drop_stale_hints:
            return False
        active_sc = getattr(self._dataset, "_preload_active_scene_id", None)
        active_sg = getattr(self._dataset, "_preload_active_segment_id", None)
        if active_sc is None or active_sg is None:
            return True
        return int(scene_id) != int(active_sc) or int(segment_id) != int(active_sg)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._heap and not self._stop.is_set():
                    self._lock.wait(timeout=0.1)
                if not self._heap:
                    continue
                item = heapq.heappop(self._heap)
            dkey = _dedupe_key_from_heap_item(item)
            with self._lock:
                self._queued_dedupe.discard(dkey)
                self._inflight_dedupe.add(dkey)
            _pr, _sq, sc, sg, kind, ref, meta = item
            try:
                self._dataset._preload_begin_scene_work(int(sc))
                try:
                    skip = False
                    if self._should_skip_stale(int(sc), int(sg)):
                        self._bump_stat("tasks_dropped_stale", 1)
                        skip = True
                    elif self._dataset._preload_should_abort_for_unload(int(sc)):
                        self._bump_stat("tasks_dropped_scene_unloading", 1)
                        skip = True

                    if not skip:
                        t0 = time.perf_counter()
                        status = "failed"
                        try:
                            if int(kind) == PRELOAD_TASK_SEGMENT_STATIC:
                                status = self._dataset._preload_worker_segment_static(int(sc), int(sg), meta)
                            elif int(kind) == PRELOAD_TASK_OVERLAP_PAIR:
                                tm = meta
                                status = self._dataset._preload_worker_overlap_pair(
                                    int(sc),
                                    int(sg),
                                    ref,
                                    tm["tgt_rep"],
                                    mode=str(tm["mode"]),
                                    point_sample_size=int(tm["point_sample_size"]),
                                    meta=tm,
                                )
                            else:
                                status = self._dataset._preload_worker_load_view_pack(int(sc), int(sg), ref)
                        except Exception as exc:
                            logger.debug("preload worker load failed: %s", exc, exc_info=True)
                            status = "failed"
                        dt_ms = (time.perf_counter() - t0) * 1000.0

                        if int(kind) == PRELOAD_TASK_SEGMENT_STATIC:
                            if status in ("loaded", "cache_hit"):
                                self._bump_stat("tasks_completed", 1)
                                self._bump_stat("segment_static_completed", 1)
                                self._bump_stat("total_latency_ms", dt_ms)
                            elif status == "failed":
                                self._bump_stat("tasks_failed", 1)
                            elif status == "skipped":
                                pass
                            else:
                                self._bump_stat("tasks_completed", 1)
                                self._bump_stat("total_latency_ms", dt_ms)
                        elif int(kind) == PRELOAD_TASK_OVERLAP_PAIR:
                            if status == "loaded":
                                self._bump_stat("tasks_completed", 1)
                                self._bump_stat("overlap_pairs_loaded", 1)
                                self._bump_stat("overlap_pair_total_latency_ms", dt_ms)
                            elif status == "cache_hit":
                                self._bump_stat("tasks_completed", 1)
                                self._bump_stat("overlap_pair_cache_hits_worker", 1)
                                self._bump_stat("overlap_pair_total_latency_ms", dt_ms)
                            elif status == "failed":
                                self._bump_stat("tasks_failed", 1)
                                self._bump_stat("overlap_pairs_failed", 1)
                            elif status == "skipped":
                                pass
                            else:
                                self._bump_stat("tasks_completed", 1)
                                self._bump_stat("overlap_pair_total_latency_ms", dt_ms)
                        elif status == "loaded":
                            self._bump_stat("tasks_completed", 1)
                            self._bump_stat("views_loaded", 1)
                            self._bump_stat("total_latency_ms", dt_ms)
                        elif status == "cache_hit":
                            self._bump_stat("tasks_completed", 1)
                            self._bump_stat("cache_hits_worker", 1)
                            self._bump_stat("total_latency_ms", dt_ms)
                        elif status == "failed":
                            self._bump_stat("tasks_failed", 1)
                        elif status == "skipped":
                            pass
                        else:
                            self._bump_stat("tasks_completed", 1)
                finally:
                    self._dataset._preload_end_scene_work(int(sc))
            finally:
                with self._lock:
                    self._inflight_dedupe.discard(dkey)
