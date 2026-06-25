from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class HistoryEntryV2:
    sequence_pos: int
    last_loss: torch.Tensor
    best_loss: torch.Tensor
    last_psnr: torch.Tensor
    best_psnr: torch.Tensor
    seen: bool
    last_visit_rollout: int

    def detach(self) -> "HistoryEntryV2":
        return HistoryEntryV2(
            sequence_pos=int(self.sequence_pos),
            last_loss=self.last_loss.detach().clone(),
            best_loss=self.best_loss.detach().clone(),
            last_psnr=self.last_psnr.detach().clone(),
            best_psnr=self.best_psnr.detach().clone(),
            seen=bool(self.seen),
            last_visit_rollout=int(self.last_visit_rollout),
        )


@dataclass
class EpisodeHistoryBankV2:
    entries: Dict[int, HistoryEntryV2] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, device: Optional[torch.device] = None) -> "EpisodeHistoryBankV2":
        _ = device
        return cls(entries={})

    def detach(self) -> "EpisodeHistoryBankV2":
        return EpisodeHistoryBankV2(entries={int(k): v.detach() for k, v in self.entries.items()})

    def update(
        self,
        *,
        sequence_pos: int,
        loss: torch.Tensor,
        psnr: torch.Tensor,
        rollout_id: int,
    ) -> "EpisodeHistoryBankV2":
        pos = int(sequence_pos)
        prev = self.entries.get(pos)
        loss_t = loss.reshape(()).detach()
        psnr_t = psnr.reshape(()).detach()
        if prev is None or bool(loss_t < prev.best_loss.to(device=loss_t.device, dtype=loss_t.dtype)):
            best_loss = loss_t
            best_psnr = psnr_t
        else:
            best_loss = prev.best_loss.to(device=loss_t.device, dtype=loss_t.dtype)
            best_psnr = prev.best_psnr.to(device=psnr_t.device, dtype=psnr_t.dtype)
        entries = dict(self.entries)
        entries[pos] = HistoryEntryV2(
            sequence_pos=pos,
            last_loss=loss_t,
            best_loss=best_loss,
            last_psnr=psnr_t,
            best_psnr=best_psnr,
            seen=True,
            last_visit_rollout=int(rollout_id),
        )
        return EpisodeHistoryBankV2(entries=entries)

    def all_seen(self, positions: Iterable[int]) -> bool:
        return all(bool(self.entries.get(int(pos), None) and self.entries[int(pos)].seen) for pos in positions)

    def count_tokens(self) -> Dict[str, float]:
        return {
            "stage2_2_history_bank_seen": float(sum(1 for item in self.entries.values() if bool(item.seen))),
            "stage2_2_history_bank_capacity": float(len(self.entries)),
        }


def history_damage_hinge_v2(
    *,
    repair_losses: torch.Tensor,
    bank: EpisodeHistoryBankV2,
    positions: Iterable[int],
    valid: Optional[torch.Tensor] = None,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = repair_losses.reshape(-1)
    pos_list = [int(x) for x in positions]
    if int(losses.numel()) == 0 or not pos_list:
        ref = losses if int(losses.numel()) else torch.zeros((), dtype=torch.float32)
        return ref.sum() * 0.0, {"stage2_2/best_damage_num_pos": 0.0}
    terms: List[torch.Tensor] = []
    valid_mask = None if valid is None else valid.to(device=losses.device, dtype=torch.bool).reshape(-1)
    for idx, pos in enumerate(pos_list[: int(losses.numel())]):
        if valid_mask is not None and (idx >= int(valid_mask.numel()) or not bool(valid_mask[int(idx)].detach().item())):
            continue
        entry = bank.entries.get(int(pos))
        if entry is None or not bool(entry.seen):
            continue
        best = entry.best_loss.to(device=losses.device, dtype=losses.dtype)
        terms.append(torch.relu(losses[int(idx)] - best - float(margin)))
    if not terms:
        return losses.sum() * 0.0, {"stage2_2/best_damage_num_pos": 0.0}
    stacked = torch.stack(terms)
    out = stacked.mean()
    detached = stacked.detach().float()
    return out, {
        "stage2_2/best_damage_num_pos": float(len(terms)),
        "stage2_2/best_damage_mean": float(detached.mean().item()),
        "stage2_2/best_damage_p90": float(torch.quantile(detached, 0.9).item()),
        "stage2_2/best_damage_max": float(detached.max().item()),
    }


__all__ = ["EpisodeHistoryBankV2", "HistoryEntryV2", "history_damage_hinge_v2"]
