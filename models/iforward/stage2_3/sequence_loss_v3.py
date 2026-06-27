from __future__ import annotations

from typing import Iterable, Optional

import torch


def role_normalized_loss_v3(losses: Iterable[torch.Tensor], *, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    vals = [x.reshape(()) for x in losses if torch.is_tensor(x)]
    if vals:
        return torch.stack(vals).mean()
    if ref is not None:
        return ref.sum() * 0.0
    return torch.tensor(0.0)


__all__ = ["role_normalized_loss_v3"]
