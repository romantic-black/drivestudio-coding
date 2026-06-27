from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn


VISIT_KIND_TO_ID: Dict[str, int] = {
    "bootstrap": 0,
    "assimilate": 1,
    "assimilation": 1,
    "repair": 2,
    "repeat_stability": 3,
    "stress": 3,
}


@dataclass(frozen=True)
class VisitMeta:
    visit_kind: str
    frame_id: int
    keyframe_id: int
    sequence_pos: int
    timestamp_us: int
    frame_gap_from_previous_visit: int
    time_since_same_frame_visit: float
    visit_count_for_frame: int
    repeat_idx: int
    repeat_budget: int
    global_update_idx_in_episode: int
    is_first_visit_of_frame: bool
    is_last_update_of_episode: bool

    @classmethod
    def from_step(cls, step: object) -> "VisitMeta":
        return cls(
            visit_kind=str(getattr(step, "visit_kind", "")),
            frame_id=int(getattr(step, "source_frame_idx", -1)),
            keyframe_id=int(getattr(step, "source_keyframe_idx", getattr(step, "source_frame_idx", -1))),
            sequence_pos=int(getattr(step, "sequence_pos", -1)),
            timestamp_us=int(getattr(step, "timestamp_us", 0)),
            frame_gap_from_previous_visit=int(getattr(step, "frame_gap", 0)),
            time_since_same_frame_visit=float(getattr(step, "time_since_same_frame_visit", 0.0)),
            visit_count_for_frame=int(getattr(step, "visit_count_for_frame", 0)),
            repeat_idx=int(getattr(step, "repeat_idx", 0)),
            repeat_budget=int(getattr(step, "repeat_budget", getattr(step, "repeats_per_block", 1))),
            global_update_idx_in_episode=int(
                getattr(step, "global_update_idx_in_episode", getattr(step, "optimizer_step_idx_in_episode", 0))
            ),
            is_first_visit_of_frame=bool(getattr(step, "is_first_visit_of_frame", int(getattr(step, "repeat_idx", 0)) == 0)),
            is_last_update_of_episode=bool(getattr(step, "is_last_update_of_episode", False)),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "VisitMeta":
        class _Step:
            pass

        step = _Step()
        for key, value in dict(raw).items():
            setattr(step, str(key), value)
        return cls.from_step(step)


def _clamp_index(value: int, max_value: int) -> int:
    return int(max(0, min(int(value), int(max_value))))


class OptimizerVisitEmbedding(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int = 32,
        repeat_idx_max: int = 16,
        repeat_budget_max: int = 16,
        visit_count_max: int = 32,
        sequence_pos_max: int = 32,
        frame_gap_max: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.repeat_idx_max = int(repeat_idx_max)
        self.repeat_budget_max = int(repeat_budget_max)
        self.visit_count_max = int(visit_count_max)
        self.sequence_pos_max = int(sequence_pos_max)
        self.frame_gap_max = int(frame_gap_max)
        self.kind_embed = nn.Embedding(4, 8)
        self.repeat_idx_embed = nn.Embedding(self.repeat_idx_max + 1, 8)
        self.repeat_budget_embed = nn.Embedding(self.repeat_budget_max + 1, 8)
        self.visit_count_embed = nn.Embedding(self.visit_count_max + 1, 8)
        self.sequence_pos_embed = nn.Embedding(self.sequence_pos_max + 1, 8)
        self.frame_gap_embed = nn.Embedding(self.frame_gap_max + 1, 8)
        self.global_mlp = nn.Sequential(
            nn.Linear(8, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 8),
        )
        self.out = nn.Sequential(
            nn.Linear(56, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def _global_features(self, value: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.tensor(float(max(int(value), 0)), device=device, dtype=dtype)
        freqs = torch.tensor([1.0, 0.5, 0.25, 0.125], device=device, dtype=dtype)
        phase = idx * freqs * (2.0 * math.pi / 1024.0)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=0).reshape(1, 8)

    def forward(
        self,
        meta: Optional[VisitMeta | Mapping[str, object] | object],
        *,
        ref: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if meta is None:
            meta_obj = VisitMeta(
                visit_kind="bootstrap",
                frame_id=-1,
                keyframe_id=-1,
                sequence_pos=0,
                timestamp_us=0,
                frame_gap_from_previous_visit=0,
                time_since_same_frame_visit=0.0,
                visit_count_for_frame=0,
                repeat_idx=0,
                repeat_budget=1,
                global_update_idx_in_episode=0,
                is_first_visit_of_frame=True,
                is_last_update_of_episode=True,
            )
        elif isinstance(meta, VisitMeta):
            meta_obj = meta
        elif isinstance(meta, Mapping):
            meta_obj = VisitMeta.from_mapping(meta)
        else:
            meta_obj = VisitMeta.from_step(meta)
        device = ref.device
        dtype = ref.dtype
        kind_id = VISIT_KIND_TO_ID.get(str(meta_obj.visit_kind), 0)
        pieces = [
            self.kind_embed(torch.tensor([kind_id], device=device, dtype=torch.long)),
            self.repeat_idx_embed(torch.tensor([_clamp_index(meta_obj.repeat_idx, self.repeat_idx_max)], device=device)),
            self.repeat_budget_embed(
                torch.tensor([_clamp_index(meta_obj.repeat_budget, self.repeat_budget_max)], device=device)
            ),
            self.visit_count_embed(
                torch.tensor([_clamp_index(meta_obj.visit_count_for_frame, self.visit_count_max)], device=device)
            ),
            self.sequence_pos_embed(torch.tensor([_clamp_index(meta_obj.sequence_pos, self.sequence_pos_max)], device=device)),
            self.frame_gap_embed(
                torch.tensor([_clamp_index(abs(int(meta_obj.frame_gap_from_previous_visit)), self.frame_gap_max)], device=device)
            ),
            self.global_mlp(self._global_features(meta_obj.global_update_idx_in_episode, device=device, dtype=dtype)),
        ]
        out = self.out(torch.cat([p.to(dtype=dtype) for p in pieces], dim=-1))
        return out.expand(int(rows), int(out.shape[-1]))


__all__ = ["OptimizerVisitEmbedding", "VISIT_KIND_TO_ID", "VisitMeta"]
