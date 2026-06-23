from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass
class Sequence10HistoryBank:
    best_loss: torch.Tensor
    best_psnr: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def empty(cls, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> "Sequence10HistoryBank":
        dev = device if device is not None else torch.device("cpu")
        return cls(
            best_loss=torch.full((10,), float("inf"), device=dev, dtype=dtype),
            best_psnr=torch.full((10,), float("-inf"), device=dev, dtype=dtype),
            valid=torch.zeros((10,), device=dev, dtype=torch.bool),
        )

    def detach(self) -> "Sequence10HistoryBank":
        return Sequence10HistoryBank(
            best_loss=self.best_loss.detach().clone(),
            best_psnr=self.best_psnr.detach().clone(),
            valid=self.valid.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "Sequence10HistoryBank":
        out_dtype = dtype if dtype is not None else self.best_loss.dtype
        return Sequence10HistoryBank(
            best_loss=self.best_loss.to(device=device, dtype=out_dtype),
            best_psnr=self.best_psnr.to(device=device, dtype=out_dtype),
            valid=self.valid.to(device=device),
        )

    def update(self, per_pos_loss: Dict[int, torch.Tensor], per_pos_psnr: Optional[Dict[int, torch.Tensor]] = None) -> None:
        psnr_map = per_pos_psnr or {}
        with torch.no_grad():
            for raw_pos, raw_loss in per_pos_loss.items():
                pos = int(raw_pos)
                if pos < 0 or pos >= 10:
                    raise ValueError(f"Sequence10HistoryBank position must be in [0, 9], got {pos}")
                loss = torch.as_tensor(raw_loss, device=self.best_loss.device, dtype=self.best_loss.dtype).detach()
                psnr = torch.as_tensor(
                    psnr_map.get(pos, torch.full_like(loss, float("-inf"))),
                    device=self.best_psnr.device,
                    dtype=self.best_psnr.dtype,
                ).detach()
                if (not bool(self.valid[pos])) or bool(loss < self.best_loss[pos]):
                    self.best_loss[pos] = loss
                    self.best_psnr[pos] = psnr
                    self.valid[pos] = True

    def before_for_positions(self, positions: Iterable[int], *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor([int(x) for x in positions], device=self.best_loss.device, dtype=torch.long)
        if int(idx.numel()) == 0:
            return (
                torch.zeros((0,), device=device, dtype=dtype),
                torch.zeros((0,), device=device, dtype=torch.bool),
            )
        if bool(((idx < 0) | (idx >= 10)).any().item()):
            raise ValueError("Sequence10HistoryBank positions must be in [0, 9]")
        return (
            self.best_loss.index_select(0, idx).to(device=device, dtype=dtype).detach(),
            self.valid.index_select(0, idx).to(device=device).detach(),
        )


def sequence10_damage_hinge_from_bank(
    *,
    after_loss: torch.Tensor,
    before_loss: torch.Tensor,
    valid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    after = after_loss.reshape(-1)
    before = before_loss.reshape(-1).to(device=after.device, dtype=after.dtype).detach()
    mask = valid.reshape(-1).to(device=after.device, dtype=torch.bool)
    if int(after.numel()) != int(before.numel()) or int(after.numel()) != int(mask.numel()):
        raise ValueError("sequence10 damage hinge tensors must have matching length")
    if int(after.numel()) == 0 or not bool(mask.any().item()):
        return after.new_tensor(0.0)
    damage = torch.relu(after[mask] - before[mask] - float(margin))
    return damage.mean()


__all__ = ["Sequence10HistoryBank", "sequence10_damage_hinge_from_bank"]
