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
    from datasets.multi_scene_dataset_v4 import ImageRef, MultiSceneDatasetV4

PRELOAD_TASK_SCENE_META = 0
PRELOAD_TASK_SEGMENT_STATIC = 1
PRELOAD_TASK_VIEW_META = 2
PRELOAD_TASK_VIEW_PACK = 3

PRIORITY_SEGMENT_STATIC = -2
PRIORITY_NEXT_BLOCK_EXACT = 0
PRIORITY_TEST_REFS = 1
PRIORITY_EPISODE_SUPERSET = 10

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedViewPackV2:
    image: Tensor
    extrinsic: Tensor
    intrinsic: Tensor
    depth: Tensor
    sky_mask: Optional[Tensor]
    viewdirs: Optional[Tensor]
    dynamic_mask: Optional[Tensor]
    egocar_mask: Optional[Tensor]
    frame_idx: int
    cam_idx: int


def _tensor_to_cpu(t: Tensor) -> Tensor:
    return t.detach().cpu()


def _optional_to_cpu(t: Optional[Tensor]) -> Optional[Tensor]:
    if t is None:
        return None
    return _tensor_to_cpu(t)


def dict_to_loaded_view_pack_v2(pack: Dict[str, Any]) -> LoadedViewPackV2:
    return LoadedViewPackV2(
        image=_tensor_to_cpu(pack["image"]),
        extrinsic=_tensor_to_cpu(pack["extrinsic"]),
        intrinsic=_tensor_to_cpu(pack["intrinsic"]),
        depth=_tensor_to_cpu(pack["depth"]),
        sky_mask=_optional_to_cpu(pack.get("sky_mask")),
        viewdirs=_optional_to_cpu(pack.get("viewdirs")),
        dynamic_mask=_optional_to_cpu(pack.get("dynamic_mask")),
        egocar_mask=_optional_to_cpu(pack.get("egocar_mask")),
        frame_idx=int(pack["frame_idx"]),
        cam_idx=int(pack["cam_idx"]),
    )


def loaded_view_pack_to_device_v2(pack: LoadedViewPackV2, device: torch.device) -> Dict[str, Any]:
    def mv(x: Optional[Tensor]) -> Optional[Tensor]:
        if x is None:
            return None
        y = x.to(device=device, non_blocking=True)
        if device.type == "cpu":
            y = y.clone()
        return y

    return {
        "image": mv(pack.image),
        "extrinsic": mv(pack.extrinsic),
        "intrinsic": mv(pack.intrinsic),
        "depth": mv(pack.depth),
        "sky_mask": mv(pack.sky_mask),
        "viewdirs": mv(pack.viewdirs),
        "dynamic_mask": mv(pack.dynamic_mask),
        "egocar_mask": mv(pack.egocar_mask),
        "frame_idx": int(pack.frame_idx),
        "cam_idx": int(pack.cam_idx),
    }


@dataclass(frozen=True)
class PreloadRuntimeConfigV2:
    enable: bool
    num_workers: int
    max_pending_tasks: int
    dedupe_tasks: bool
    drop_stale_hints: bool
    warm_scene_meta: bool
    warm_segment_static: bool
    warm_next_block_exact: bool
    warm_test_refs: bool
    warm_episode_source_superset: bool
    warm_episode_chain_exact: bool
    enable_view_pack_cache: bool
    stats_log_interval_steps: int


_REQUIRED_PRELOAD_KEYS_V2 = (
    "enable",
    "num_workers",
    "max_pending_tasks",
    "dedupe_tasks",
    "drop_stale_hints",
    "warm_scene_meta",
    "warm_segment_static",
    "warm_next_block_exact",
    "warm_test_refs",
    "warm_episode_source_superset",
    "enable_view_pack_cache",
    "stats_log_interval_steps",
)


def coerce_preload_cfg_dict_v2(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize ``data.preload`` (dict or OmegaConf) into a plain dict for parse_preload_cfg_v2.

    Returns None when preload is absent, or ``enable`` is false. When ``enable`` is true, all
    V2 keys must be present or ValueError is raised.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        d: Dict[str, Any] = dict(raw)
    else:
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(raw, DictConfig):
                d = OmegaConf.to_container(raw, resolve=True)
            else:
                raise TypeError(f"data.preload must be dict or DictConfig, got {type(raw)}")
        except ImportError as e:
            raise TypeError(f"data.preload must be a dict when OmegaConf is unavailable, got {type(raw)}") from e
    if not bool(d.get("enable", False)):
        return None
    missing = [k for k in _REQUIRED_PRELOAD_KEYS_V2 if k not in d]
    if missing:
        raise ValueError(f"data.preload.enable is true but missing keys for V2: {missing}")
    return d


def parse_preload_cfg_v2(raw: Optional[Dict[str, Any]]) -> Optional[PreloadRuntimeConfigV2]:
    if raw is None or not bool(raw.get("enable")):
        return None
    missing = [k for k in _REQUIRED_PRELOAD_KEYS_V2 if k not in raw]
    if missing:
        raise ValueError(f"data.preload.enable is true but missing keys for V2: {missing}")
    num_workers = int(raw["num_workers"])
    if num_workers != 1:
        raise ValueError(f"data.preload.num_workers must be 1, got {num_workers}")
    return PreloadRuntimeConfigV2(
        enable=True,
        num_workers=num_workers,
        max_pending_tasks=int(raw["max_pending_tasks"]),
        dedupe_tasks=bool(raw["dedupe_tasks"]),
        drop_stale_hints=bool(raw["drop_stale_hints"]),
        warm_scene_meta=bool(raw["warm_scene_meta"]),
        warm_segment_static=bool(raw["warm_segment_static"]),
        warm_next_block_exact=bool(raw["warm_next_block_exact"]),
        warm_test_refs=bool(raw["warm_test_refs"]),
        warm_episode_source_superset=bool(raw["warm_episode_source_superset"]),
        warm_episode_chain_exact=bool(raw.get("warm_episode_chain_exact", raw["warm_episode_source_superset"])),
        enable_view_pack_cache=bool(raw["enable_view_pack_cache"]),
        stats_log_interval_steps=int(raw["stats_log_interval_steps"]),
    )


HeapItem = Tuple[int, int, int, int, int, Tuple[int, int], Dict[str, Any]]
DedupeKey = Tuple[int, int, int, int, int]


class AssetPreloadManagerV2:
    def __init__(self, dataset: "MultiSceneDatasetV4", cfg: PreloadRuntimeConfigV2) -> None:
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
            "tasks_failed": 0,
            "tasks_dropped_stale": 0,
            "tasks_dropped_queue_full": 0,
            "tasks_evicted_for_admission": 0,
            "total_latency_ms": 0.0,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AssetPreloadWorkerV2", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                self._thread = None

    def pop_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            out = dict(self._stats)
            self._stats = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "tasks_dropped_stale": 0,
                "tasks_dropped_queue_full": 0,
                "tasks_evicted_for_admission": 0,
                "total_latency_ms": 0.0,
            }
        return out

    def _bump_stat(self, name: str, delta: float) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0.0) + delta

    def _push(
        self,
        *,
        priority: int,
        task_kind: int,
        scene_id: int,
        segment_id: int,
        image_ref: Tuple[int, int],
        meta: Dict[str, Any],
    ) -> None:
        key = (int(task_kind), int(scene_id), int(segment_id), int(image_ref[0]), int(image_ref[1]))
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
                vk = (
                    int(victim[4]),
                    int(victim[2]),
                    int(victim[3]),
                    int(victim[5][0]),
                    int(victim[5][1]),
                )
                self._queued_dedupe.discard(vk)
                self._bump_stat("tasks_evicted_for_admission", 1)
            seq = next(self._seq)
            heapq.heappush(
                self._heap,
                (
                    int(priority),
                    int(seq),
                    int(scene_id),
                    int(segment_id),
                    int(task_kind),
                    (int(image_ref[0]), int(image_ref[1])),
                    dict(meta),
                ),
            )
            self._queued_dedupe.add(key)
            self._lock.notify()

    def submit_scene_meta(self, priority: int, scene_id: int, segment_id: int, *, meta: Dict[str, Any]) -> None:
        self._push(
            priority=priority,
            task_kind=PRELOAD_TASK_SCENE_META,
            scene_id=scene_id,
            segment_id=segment_id,
            image_ref=(-1, -1),
            meta=meta,
        )

    def submit_segment_static(
        self, priority: int, scene_id: int, segment_id: int, *, meta: Dict[str, Any]
    ) -> None:
        self._push(
            priority=priority,
            task_kind=PRELOAD_TASK_SEGMENT_STATIC,
            scene_id=scene_id,
            segment_id=segment_id,
            image_ref=(-1, -1),
            meta=meta,
        )

    def submit_view_meta(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        image_ref: "ImageRef",
        *,
        meta: Dict[str, Any],
    ) -> None:
        self._push(
            priority=priority,
            task_kind=PRELOAD_TASK_VIEW_META,
            scene_id=scene_id,
            segment_id=segment_id,
            image_ref=(int(image_ref[0]), int(image_ref[1])),
            meta=meta,
        )

    def submit_view_pack(
        self,
        priority: int,
        scene_id: int,
        segment_id: int,
        image_ref: "ImageRef",
        *,
        meta: Dict[str, Any],
    ) -> None:
        self._push(
            priority=priority,
            task_kind=PRELOAD_TASK_VIEW_PACK,
            scene_id=scene_id,
            segment_id=segment_id,
            image_ref=(int(image_ref[0]), int(image_ref[1])),
            meta=meta,
        )

    def submit_preload_hint(
        self,
        *,
        hint: Dict[str, Any],
        hint_scope: str,
        include_test: bool,
    ) -> None:
        scene_id = int(hint["scene_id"])
        segment_id = int(hint["segment_id"])
        refs = [tuple(r) for r in hint.get("future_image_refs", [])]
        if self._cfg.warm_scene_meta:
            self.submit_scene_meta(PRIORITY_SEGMENT_STATIC, scene_id, segment_id, meta={"hint_scope": hint_scope})
        if self._cfg.warm_segment_static:
            self.submit_segment_static(
                PRIORITY_SEGMENT_STATIC,
                scene_id,
                segment_id,
                meta={"hint_scope": hint_scope},
            )
        if hint_scope == "next_block_exact":
            if self._cfg.warm_next_block_exact:
                for ref in refs:
                    self.submit_view_meta(PRIORITY_NEXT_BLOCK_EXACT, scene_id, segment_id, ref, meta={})
                    if self._cfg.enable_view_pack_cache:
                        self.submit_view_pack(PRIORITY_NEXT_BLOCK_EXACT, scene_id, segment_id, ref, meta={})
        elif hint_scope == "episode_source_superset":
            if self._cfg.warm_episode_source_superset:
                for ref in refs:
                    self.submit_view_meta(PRIORITY_EPISODE_SUPERSET, scene_id, segment_id, ref, meta={})
        elif hint_scope == "episode_chain_exact":
            if self._cfg.warm_episode_chain_exact:
                for ref in refs:
                    self.submit_view_meta(PRIORITY_EPISODE_SUPERSET, scene_id, segment_id, ref, meta={})
                    if self._cfg.enable_view_pack_cache:
                        self.submit_view_pack(PRIORITY_EPISODE_SUPERSET, scene_id, segment_id, ref, meta={})
        if include_test and self._cfg.warm_test_refs:
            sidx = self._dataset.get_segment_index(scene_id, segment_id)
            for ref in self._dataset._resolve_test_image_refs(sidx):
                self.submit_view_meta(PRIORITY_TEST_REFS, scene_id, segment_id, ref, meta={"purpose": "test"})

    def _should_skip_stale(self, scene_id: int, segment_id: int) -> bool:
        if not self._cfg.drop_stale_hints:
            return False
        active_sc = getattr(self._dataset, "_preload_active_scene_id", None)
        active_sg = getattr(self._dataset, "_preload_active_segment_id", None)
        training_sc = getattr(self._dataset, "_preload_training_scene_id", None)
        training_sg = getattr(self._dataset, "_preload_training_segment_id", None)
        active = None
        training = None
        if active_sc is not None and active_sg is not None:
            active = (int(active_sc), int(active_sg))
        if training_sc is not None and training_sg is not None:
            training = (int(training_sc), int(training_sg))
        if active is None and training is None:
            return False
        if active is not None and (int(scene_id), int(segment_id)) == active:
            return False
        if training is not None and (int(scene_id), int(segment_id)) == training:
            return False
        return True

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._heap and not self._stop.is_set():
                    self._lock.wait(timeout=0.2)
                if self._stop.is_set() and not self._heap:
                    return
                if not self._heap:
                    continue
                item = heapq.heappop(self._heap)
                pr, seq, scene_id, segment_id, task_kind, image_ref, meta = item
                _ = (pr, seq)
                dkey = (
                    int(task_kind),
                    int(scene_id),
                    int(segment_id),
                    int(image_ref[0]),
                    int(image_ref[1]),
                )
                self._queued_dedupe.discard(dkey)
                self._inflight_dedupe.add(dkey)
            if self._should_skip_stale(int(scene_id), int(segment_id)):
                self._bump_stat("tasks_dropped_stale", 1)
                with self._lock:
                    self._inflight_dedupe.discard(dkey)
                continue
            t0 = time.perf_counter()
            ok = True
            try:
                if int(task_kind) == PRELOAD_TASK_SCENE_META:
                    self._dataset._preload_worker_scene_meta(scene_id, segment_id, meta)
                elif int(task_kind) == PRELOAD_TASK_SEGMENT_STATIC:
                    self._dataset._preload_worker_segment_static(scene_id, segment_id, meta)
                elif int(task_kind) == PRELOAD_TASK_VIEW_META:
                    self._dataset._preload_worker_view_meta(scene_id, segment_id, image_ref, meta)
                elif int(task_kind) == PRELOAD_TASK_VIEW_PACK:
                    self._dataset._preload_worker_view_pack(scene_id, segment_id, image_ref, meta)
                else:
                    raise ValueError(f"unknown preload task kind={task_kind}")
            except Exception:
                ok = False
                logger.exception(
                    "Asset preload task failed (task_kind=%s scene_id=%s segment_id=%s image_ref=%s meta=%s)",
                    int(task_kind),
                    int(scene_id),
                    int(segment_id),
                    tuple(image_ref),
                    dict(meta or {}),
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._bump_stat("tasks_completed", 1)
            self._bump_stat("total_latency_ms", elapsed_ms)
            if not ok:
                self._bump_stat("tasks_failed", 1)
            with self._lock:
                self._inflight_dedupe.discard(dkey)
