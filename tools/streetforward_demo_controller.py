from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from models.streetforward.minimal_trainer_stage4_3 import MinimalStreetForwardStage4_3
from tools.train_minimal_streetforward_stage1_1 import convert_batch_to_minimal_format
from tools.train_minimal_streetforward_stage4_1_one_segment_v3 import _build_scheduler_node_sync


@dataclass
class StreetForwardDisplayState:
    current_snapshot: Optional[Dict[str, Any]] = None
    current_scheduler_info: Optional[Dict[str, Any]] = None
    last_events: List[Dict[str, Any]] = None
    last_metrics: Dict[str, float] = None
    last_raw_batch: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.last_events is None:
            self.last_events = []
        if self.last_metrics is None:
            self.last_metrics = {}


class StreetForwardDemoController:
    def __init__(
        self,
        *,
        cfg: Any,
        dataset: Any,
        scheduler: Any,
        trainer: MinimalStreetForwardStage4_3,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.scheduler = scheduler
        self.trainer = trainer
        self.device = device
        self.busy = False
        self.auto_refresh_after_block = True
        self.display = StreetForwardDisplayState()

    def _get_segment_aabb_tensor(self) -> Optional[torch.Tensor]:
        aabb = getattr(self.dataset, "segment_aabb", None)
        if aabb is None:
            return None
        if torch.is_tensor(aabb):
            return aabb.detach().to("cpu")
        return torch.tensor(aabb, dtype=torch.float32)

    def _to_scalar_float(self, x: Any) -> Optional[float]:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if torch.is_tensor(x) and x.numel() == 1:
            return float(x.detach().item())
        return None

    def _reduce_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(metrics_list) == 0:
            return {}
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for row in metrics_list:
            for k, v in row.items():
                vv = self._to_scalar_float(v)
                if vv is None:
                    continue
                sums[k] = sums.get(k, 0.0) + vv
                counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / max(counts[k], 1) for k in sums.keys()}

    def _batch_to_minimal(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        tgt = raw_batch.get("target")
        if not isinstance(tgt, dict) or tgt.get("image") is None:
            raise ValueError("scheduler batch must contain target.image")
        num_target_views = int(tgt["image"].shape[0])
        return convert_batch_to_minimal_format(
            raw_batch,
            self.device,
            num_targets=num_target_views,
            include_source_for_2d=True,
            view_selection=None,
        )

    def _snapshot_from_raw_batch(
        self,
        raw_batch: Dict[str, Any],
        *,
        scheduler_meta: Optional[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        minimal_batch = self._batch_to_minimal(raw_batch)
        meta = dict(scheduler_meta or {})
        if metrics is not None:
            for key in (
                "num_bg_update",
                "num_distant_update",
                "num_sky_update",
                "num_rigid_update",
                "src_backproject_pass_count",
            ):
                if key in metrics:
                    meta[key] = metrics[key]

        return self.trainer.export_viewer_snapshot(
            minimal_batch,
            scheduler_meta=meta,
            segment_aabb=self._get_segment_aabb_tensor(),
            include_hidden=False,
            allow_hidden_cache_update=False,
            allow_node_state_writeback=False,
            rigid_export_frame_idx=int(minimal_batch.get("source_frame_idx", 0)),
        )

    def peek_scheduler_info(self) -> Dict[str, Any]:
        info = self.scheduler.get_current_info()
        self.display.current_scheduler_info = dict(info)
        return dict(info)

    def prime_first_snapshot(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        raw_batch = self.scheduler.materialize_current_batch_without_advance()
        scheduler_info = raw_batch.get("_scheduler_v4_aligned_info") or self.scheduler.get_current_info()
        events = self.scheduler.pop_events()
        snapshot = self._snapshot_from_raw_batch(raw_batch, scheduler_meta=scheduler_info)
        self.display.current_snapshot = snapshot
        self.display.current_scheduler_info = dict(scheduler_info)
        self.display.last_events = list(events)
        self.display.last_raw_batch = raw_batch
        return snapshot

    def build_or_refresh_snapshot(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        raw_batch = self.scheduler.materialize_current_batch_without_advance()
        scheduler_info = raw_batch.get("_scheduler_v4_aligned_info") or self.scheduler.get_current_info()
        snapshot = self._snapshot_from_raw_batch(raw_batch, scheduler_meta=scheduler_info)
        self.display.current_snapshot = snapshot
        self.display.current_scheduler_info = dict(scheduler_info)
        self.display.last_raw_batch = raw_batch
        return snapshot

    def train_next_block(self, num_blocks: int = 1) -> Dict[str, Any]:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.busy:
            raise ValueError("controller is busy")

        self.busy = True
        collected_events: List[Dict[str, Any]] = []
        collected_metrics: List[Dict[str, Any]] = []
        block_end_count = 0
        last_raw_batch: Optional[Dict[str, Any]] = None
        last_scheduler_info: Optional[Dict[str, Any]] = None
        try:
            while True:
                raw_batch = self.scheduler.next_batch()
                scheduler_info = raw_batch.get("_scheduler_v4_aligned_info") or self.scheduler.get_current_info()
                step_events = self.scheduler.pop_events()
                scheduler_node_sync = _build_scheduler_node_sync(self.cfg, scheduler_info, step_events)

                minimal_batch = self._batch_to_minimal(raw_batch)
                metrics = self.trainer.train_step(
                    minimal_batch,
                    scheduler_node_sync=scheduler_node_sync,
                )
                collected_metrics.append(metrics)
                collected_events.extend(step_events)

                block_end_count += sum(1 for ev in step_events if ev.get("type") == "block_end")
                last_raw_batch = raw_batch
                last_scheduler_info = scheduler_info
                if block_end_count >= num_blocks:
                    break

            if last_raw_batch is None or last_scheduler_info is None:
                raise ValueError("train_next_block ended without a processed batch")

            reduced = self._reduce_metrics(collected_metrics)
            snapshot = self._snapshot_from_raw_batch(
                last_raw_batch,
                scheduler_meta=last_scheduler_info,
                metrics=reduced,
            )
            self.display.current_snapshot = snapshot
            self.display.current_scheduler_info = dict(last_scheduler_info)
            self.display.last_events = list(collected_events)
            self.display.last_metrics = dict(reduced)
            self.display.last_raw_batch = last_raw_batch
            return snapshot
        finally:
            self.busy = False

    def reset_runtime_to_segment_init(self) -> Dict[str, Any]:
        if self.busy:
            raise ValueError("controller is busy")
        self.trainer.reset_node_state()
        return self.build_or_refresh_snapshot()

    def export_current_snapshot(self, path: str) -> str:
        snap = self.display.current_snapshot
        if snap is None:
            raise ValueError("No snapshot available; call prime_first_snapshot() first")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snap, out_path)
        return str(out_path)

