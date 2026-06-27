from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass(frozen=True)
class HistoryEntryV3:
    best_loss: torch.Tensor
    best_psnr: torch.Tensor
    last_loss: torch.Tensor
    visit_count: int = 0
    last_update_step: int = -1
    seen: bool = False

    def detach(self) -> "HistoryEntryV3":
        return HistoryEntryV3(
            best_loss=self.best_loss.detach().clone(),
            best_psnr=self.best_psnr.detach().clone(),
            last_loss=self.last_loss.detach().clone(),
            visit_count=int(self.visit_count),
            last_update_step=int(self.last_update_step),
            seen=bool(self.seen),
        )


@dataclass(frozen=True)
class EpisodeHistoryBankV3:
    entries: Dict[int, HistoryEntryV3] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, device: Optional[torch.device] = None) -> "EpisodeHistoryBankV3":
        _ = device
        return cls(entries={})

    def detach(self) -> "EpisodeHistoryBankV3":
        return EpisodeHistoryBankV3(entries={int(k): v.detach() for k, v in self.entries.items()})

    def update(self, *, sequence_pos: int, loss: torch.Tensor, psnr: torch.Tensor, rollout_id: int) -> "EpisodeHistoryBankV3":
        pos = int(sequence_pos)
        old = self.entries.get(pos)
        loss_v = loss.reshape(()).detach()
        psnr_v = psnr.reshape(()).detach()
        if old is None or not bool(old.seen):
            entry = HistoryEntryV3(
                best_loss=loss_v,
                best_psnr=psnr_v,
                last_loss=loss_v,
                visit_count=1,
                last_update_step=int(rollout_id),
                seen=True,
            )
        else:
            better = bool((loss_v < old.best_loss.to(device=loss_v.device, dtype=loss_v.dtype)).detach().item())
            entry = HistoryEntryV3(
                best_loss=loss_v if better else old.best_loss,
                best_psnr=psnr_v if better else old.best_psnr,
                last_loss=loss_v,
                visit_count=int(old.visit_count) + 1,
                last_update_step=int(rollout_id),
                seen=True,
            )
        entries = dict(self.entries)
        entries[pos] = entry
        return EpisodeHistoryBankV3(entries=entries)

    def count_tokens(self) -> Dict[str, float]:
        return {
            "stage2_3_history_bank_seen": float(sum(1 for item in self.entries.values() if bool(item.seen))),
            "stage2_3_history_bank_capacity": float(len(self.entries)),
            "stage2_3_history_bank_updates": float(sum(int(item.visit_count) for item in self.entries.values())),
        }


def history_damage_hinge_v3(
    *,
    repair_losses: torch.Tensor,
    bank: EpisodeHistoryBankV3,
    positions: Iterable[int],
    valid: torch.Tensor,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = repair_losses.reshape(-1)
    valid_v = valid.reshape(-1).to(device=losses.device, dtype=torch.bool)
    terms = []
    for raw_pos in positions:
        pos = int(raw_pos)
        if pos < 0 or pos >= int(losses.numel()) or pos >= int(valid_v.numel()) or not bool(valid_v[pos].detach().item()):
            continue
        entry = bank.entries.get(pos)
        if entry is None or not bool(entry.seen):
            continue
        before = entry.best_loss.to(device=losses.device, dtype=losses.dtype)
        terms.append(torch.relu(losses[pos] - before - float(margin)))
    if not terms:
        return losses.sum() * 0.0, {"stage2_3/best_damage_num_pos": 0.0}
    stacked = torch.stack(terms)
    detached = stacked.detach().float()
    return stacked.mean(), {
        "stage2_3/best_damage_num_pos": float(len(terms)),
        "stage2_3/best_damage_mean": float(detached.mean().item()),
        "stage2_3/best_damage_p90": float(torch.quantile(detached, 0.9).item()),
        "stage2_3/best_damage_max": float(detached.max().item()),
    }


__all__ = ["EpisodeHistoryBankV3", "HistoryEntryV3", "history_damage_hinge_v3"]
