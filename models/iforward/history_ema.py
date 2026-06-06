from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from models.streetforward.stage6_0 import EventPack, LocalGSState
from models.streetforward.stage6_0.posterior_updater import DeltaPack


def _col(
    value: Optional[torch.Tensor],
    *,
    n: int,
    ref: torch.Tensor,
    default: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out_dtype = dtype or ref.dtype
    if value is None:
        return ref.new_full((int(n), 1), float(default), dtype=out_dtype)
    out = value.to(device=ref.device, dtype=out_dtype)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 column tensor row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) != 1:
        out = out.reshape(int(n), -1).mean(dim=-1, keepdim=True)
    return out


def _bool_col(value: Optional[torch.Tensor], *, n: int, ref: torch.Tensor, default: bool = True) -> torch.Tensor:
    if value is None:
        return torch.full((int(n), 1), bool(default), device=ref.device, dtype=torch.bool)
    out = value.to(device=ref.device)
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    if out.dim() != 2 or int(out.shape[0]) != int(n):
        raise ValueError(f"IForward v3 bool tensor row mismatch: got {tuple(out.shape)}, expected rows={int(n)}")
    if int(out.shape[1]) != 1:
        out = out.reshape(int(n), -1).any(dim=-1, keepdim=True)
    return out.to(dtype=torch.bool)


@dataclass
class IForwardResidualPack:
    error_bg: torch.Tensor
    support_bg: torch.Tensor
    error_distant: Optional[torch.Tensor] = None
    support_distant: Optional[torch.Tensor] = None
    error_rigid: Optional[torch.Tensor] = None
    support_rigid: Optional[torch.Tensor] = None


@dataclass
class IForwardHistoryBranchEMA:
    support_fast: torch.Tensor
    error_fast: torch.Tensor
    update_norm_fast: torch.Tensor
    support_slow: torch.Tensor
    error_slow: torch.Tensor
    update_norm_slow: torch.Tensor
    initialized: torch.Tensor
    block_support_sum: torch.Tensor
    block_present_count: torch.Tensor
    block_visible_count: torch.Tensor

    @classmethod
    def empty(cls, *, num_rows: int, ref: torch.Tensor) -> "IForwardHistoryBranchEMA":
        z = ref.detach().new_zeros((int(num_rows), 1), dtype=torch.float32)
        return cls(
            support_fast=z.clone(),
            error_fast=z.clone(),
            update_norm_fast=z.clone(),
            support_slow=z.clone(),
            error_slow=z.clone(),
            update_norm_slow=z.clone(),
            initialized=z.clone(),
            block_support_sum=z.clone(),
            block_present_count=z.clone(),
            block_visible_count=z.clone(),
        )

    def detach(self) -> "IForwardHistoryBranchEMA":
        return IForwardHistoryBranchEMA(
            support_fast=self.support_fast.detach().clone(),
            error_fast=self.error_fast.detach().clone(),
            update_norm_fast=self.update_norm_fast.detach().clone(),
            support_slow=self.support_slow.detach().clone(),
            error_slow=self.error_slow.detach().clone(),
            update_norm_slow=self.update_norm_slow.detach().clone(),
            initialized=self.initialized.detach().clone(),
            block_support_sum=self.block_support_sum.detach().clone(),
            block_present_count=self.block_present_count.detach().clone(),
            block_visible_count=self.block_visible_count.detach().clone(),
        )

    @property
    def num_rows(self) -> int:
        return int(self.support_fast.shape[0])

    @property
    def block_support_count(self) -> torch.Tensor:
        return self.block_visible_count

    def select(self, rows: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if rows is None:
            return {
                "support_fast": self.support_fast,
                "error_fast": self.error_fast,
                "update_norm_fast": self.update_norm_fast,
                "support_slow": self.support_slow,
                "error_slow": self.error_slow,
                "update_norm_slow": self.update_norm_slow,
                "initialized": self.initialized,
            }
        idx = rows.to(device=self.support_fast.device, dtype=torch.long)
        return {
            "support_fast": self.support_fast[idx],
            "error_fast": self.error_fast[idx],
            "update_norm_fast": self.update_norm_fast[idx],
            "support_slow": self.support_slow[idx],
            "error_slow": self.error_slow[idx],
            "update_norm_slow": self.update_norm_slow[idx],
            "initialized": self.initialized[idx],
        }

    def record_support_snapshot(self, support: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Dict[str, float]:
        n = self.num_rows
        support_col = _col(support, n=n, ref=self.support_fast, default=0.0, dtype=torch.float32).detach()
        valid_col = _bool_col(valid, n=n, ref=self.support_fast, default=True)
        self.block_support_sum = self.block_support_sum + torch.log1p(support_col.clamp_min(0.0))
        self.block_present_count = self.block_present_count + torch.ones_like(self.block_present_count)
        self.block_visible_count = self.block_visible_count + valid_col.to(dtype=self.block_visible_count.dtype)
        return {
            "support_snapshot_rows": float(n),
            "support_snapshot_present_rows": float(n),
            "support_snapshot_visible_rows": float(valid_col.sum().item()) if valid_col.numel() else 0.0,
            "support_snapshot_valid_ratio": self._ratio(valid_col),
        }

    def record_indexed_support_snapshot(
        self,
        *,
        rows: torch.Tensor,
        support: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        idx = rows.to(device=self.support_fast.device, dtype=torch.long).reshape(-1)
        n = int(idx.numel())
        if n == 0:
            return {"support_snapshot_rows": 0.0, "support_snapshot_valid_ratio": 0.0}
        support_col = _col(support, n=n, ref=self.support_fast, default=0.0, dtype=torch.float32).detach()
        valid_col = _bool_col(valid, n=n, ref=self.support_fast, default=True)
        add_support = self.block_support_sum.new_zeros(self.block_support_sum.shape)
        add_present = self.block_present_count.new_zeros(self.block_present_count.shape)
        add_visible = self.block_visible_count.new_zeros(self.block_visible_count.shape)
        add_support.index_add_(0, idx, torch.log1p(support_col.clamp_min(0.0)))
        add_present.index_add_(0, idx, torch.ones((n, 1), device=idx.device, dtype=add_present.dtype))
        add_visible.index_add_(0, idx, valid_col.to(dtype=add_visible.dtype))
        self.block_support_sum = self.block_support_sum + add_support
        self.block_present_count = self.block_present_count + add_present
        self.block_visible_count = self.block_visible_count + add_visible
        return {
            "support_snapshot_rows": float(n),
            "support_snapshot_present_rows": float(n),
            "support_snapshot_visible_rows": float(valid_col.sum().item()) if valid_col.numel() else 0.0,
            "support_snapshot_valid_ratio": self._ratio(valid_col),
        }

    def commit_support(
        self,
        *,
        fast_beta_visible: float,
        fast_beta_invisible: float,
        slow_beta_visible: float,
        slow_beta_invisible: float,
        support_min: float,
    ) -> Dict[str, float]:
        has_present = self.block_present_count > 0
        support_cur = self.block_support_sum / self.block_present_count.clamp_min(1.0)
        visible = (support_cur > float(support_min)) & (self.block_visible_count > 0) & has_present
        invisible = has_present & ~visible
        support_cur = torch.where(has_present, support_cur, torch.zeros_like(support_cur))
        vis_f = visible.to(dtype=self.support_fast.dtype)
        self.support_fast = torch.where(
            visible,
            float(fast_beta_visible) * self.support_fast + (1.0 - float(fast_beta_visible)) * support_cur,
            torch.where(
                invisible,
                float(fast_beta_invisible) * self.support_fast,
                self.support_fast,
            ),
        )
        self.support_slow = torch.where(
            visible,
            float(slow_beta_visible) * self.support_slow + (1.0 - float(slow_beta_visible)) * support_cur,
            torch.where(
                invisible,
                float(slow_beta_invisible) * self.support_slow,
                self.support_slow,
            ),
        )
        self.initialized = torch.maximum(self.initialized, vis_f)
        self.block_support_sum = torch.zeros_like(self.block_support_sum)
        self.block_present_count = torch.zeros_like(self.block_present_count)
        self.block_visible_count = torch.zeros_like(self.block_visible_count)
        return {
            "support_commit_rows": float(has_present.sum().item()) if has_present.numel() else 0.0,
            "support_present_ratio": self._ratio(has_present),
            "support_visible_ratio": self._ratio(visible),
            "support_invisible_ratio": self._ratio(invisible),
        }

    def apply_residual(
        self,
        *,
        error_cur: torch.Tensor,
        support_cur: torch.Tensor,
        fast_beta: float,
        slow_beta: float,
        support_min: float,
    ) -> Dict[str, float]:
        n = self.num_rows
        error = _col(error_cur, n=n, ref=self.error_fast, default=0.0, dtype=torch.float32).detach()
        support = _col(support_cur, n=n, ref=self.error_fast, default=0.0, dtype=torch.float32).detach()
        visible = support > float(support_min)
        vis_f = visible.to(dtype=self.error_fast.dtype)
        self.error_fast = torch.where(
            visible,
            float(fast_beta) * self.error_fast + (1.0 - float(fast_beta)) * error,
            self.error_fast,
        )
        self.error_slow = torch.where(
            visible,
            float(slow_beta) * self.error_slow + (1.0 - float(slow_beta)) * error,
            self.error_slow,
        )
        self.initialized = torch.maximum(self.initialized, vis_f)
        return {"residual_visible_ratio": self._ratio(visible)}

    def apply_update_norm(self, update_norm_cur: torch.Tensor, *, fast_beta: float, slow_beta: float) -> Dict[str, float]:
        n = self.num_rows
        cur = _col(update_norm_cur, n=n, ref=self.update_norm_fast, default=0.0, dtype=torch.float32).detach()
        written = cur > 0
        self.update_norm_fast = torch.where(
            written,
            float(fast_beta) * self.update_norm_fast + (1.0 - float(fast_beta)) * cur,
            self.update_norm_fast,
        )
        self.update_norm_slow = torch.where(
            written,
            float(slow_beta) * self.update_norm_slow + (1.0 - float(slow_beta)) * cur,
            self.update_norm_slow,
        )
        return {
            "update_written_ratio": self._ratio(written),
            "update_norm_mean": float(cur.mean().item()) if cur.numel() else 0.0,
        }

    def stats(self, prefix: str) -> Dict[str, float]:
        def mean(x: torch.Tensor) -> float:
            return float(x.detach().mean().item()) if int(x.numel()) > 0 else 0.0

        return {
            f"{prefix}/support_fast_mean": mean(self.support_fast),
            f"{prefix}/support_slow_mean": mean(self.support_slow),
            f"{prefix}/error_fast_mean": mean(self.error_fast),
            f"{prefix}/error_slow_mean": mean(self.error_slow),
            f"{prefix}/update_norm_fast_mean": mean(self.update_norm_fast),
            f"{prefix}/update_norm_slow_mean": mean(self.update_norm_slow),
            f"{prefix}/initialized_ratio": mean((self.initialized > 0).to(dtype=self.initialized.dtype)),
            f"{prefix}/pending_support_rows": float((self.block_present_count > 0).sum().item())
            if self.block_present_count.numel()
            else 0.0,
            f"{prefix}/pending_present_rows": float((self.block_present_count > 0).sum().item())
            if self.block_present_count.numel()
            else 0.0,
            f"{prefix}/pending_visible_rows": float((self.block_visible_count > 0).sum().item())
            if self.block_visible_count.numel()
            else 0.0,
        }

    @staticmethod
    def _ratio(mask: torch.Tensor) -> float:
        return float(mask.detach().to(dtype=torch.float32).mean().item()) if mask.numel() else 0.0


@dataclass
class IForwardHistoryEMAState:
    bg: IForwardHistoryBranchEMA
    distant: Optional[IForwardHistoryBranchEMA]
    rigid: Optional[IForwardHistoryBranchEMA]

    @classmethod
    def from_local_state(cls, local_state: LocalGSState) -> "IForwardHistoryEMAState":
        ref = local_state.bg.means
        return cls(
            bg=IForwardHistoryBranchEMA.empty(num_rows=int(local_state.bg.means.shape[0]), ref=ref),
            distant=(
                IForwardHistoryBranchEMA.empty(num_rows=int(local_state.distant.means.shape[0]), ref=ref)
                if local_state.distant is not None
                else None
            ),
            rigid=(
                IForwardHistoryBranchEMA.empty(num_rows=int(local_state.rigid.means.shape[0]), ref=ref)
                if local_state.rigid is not None
                else None
            ),
        )

    def detach(self) -> "IForwardHistoryEMAState":
        return IForwardHistoryEMAState(
            bg=self.bg.detach(),
            distant=None if self.distant is None else self.distant.detach(),
            rigid=None if self.rigid is None else self.rigid.detach(),
        )

    def record_block_support_snapshot(self, *, event: EventPack, local_state: LocalGSState) -> Dict[str, float]:
        aux: Dict[str, float] = {}
        aux.update({f"v3/history/bg_{k}": v for k, v in self.bg.record_support_snapshot(event.support_bg, event.valid_bg).items()})
        if self.distant is not None and local_state.distant is not None:
            aux.update(
                {
                    f"v3/history/distant_{k}": v
                    for k, v in self.distant.record_support_snapshot(event.support_distant, event.valid_distant).items()
                }
            )
        if self.rigid is not None and local_state.rigid is not None:
            route = getattr(event, "route", None)
            rows = getattr(route, "S", None) if route is not None else None
            if rows is not None:
                aux.update(
                    {
                        f"v3/history/rigid_{k}": v
                        for k, v in self.rigid.record_indexed_support_snapshot(
                            rows=rows,
                            support=event.support_rigid,
                            valid=event.valid_rigid,
                        ).items()
                    }
                )
        return aux

    def commit_block_support(
        self,
        *,
        support_betas: Dict[str, float],
        support_min: Dict[str, float],
    ) -> Dict[str, float]:
        aux: Dict[str, float] = {}
        kwargs = {
            "fast_beta_visible": float(support_betas["fast_beta_visible"]),
            "fast_beta_invisible": float(support_betas["fast_beta_invisible"]),
            "slow_beta_visible": float(support_betas["slow_beta_visible"]),
            "slow_beta_invisible": float(support_betas["slow_beta_invisible"]),
        }
        aux.update(
            {
                f"v3/history/bg_{k}": v
                for k, v in self.bg.commit_support(support_min=float(support_min.get("bg", 0.0)), **kwargs).items()
            }
        )
        if self.distant is not None:
            aux.update(
                {
                    f"v3/history/distant_{k}": v
                    for k, v in self.distant.commit_support(
                        support_min=float(support_min.get("distant", 0.0)),
                        **kwargs,
                    ).items()
                }
            )
        if self.rigid is not None:
            aux.update(
                {
                    f"v3/history/rigid_{k}": v
                    for k, v in self.rigid.commit_support(
                        support_min=float(support_min.get("rigid", 0.0)),
                        **kwargs,
                    ).items()
                }
            )
        return aux

    def record_update_norm(
        self,
        *,
        delta: DeltaPack,
        update_betas: Dict[str, float],
    ) -> Dict[str, float]:
        aux: Dict[str, float] = {}
        fast = float(update_betas["fast_beta"])
        slow = float(update_betas["slow_beta"])
        aux.update(
            {
                f"v3/history/bg_{k}": v
                for k, v in self.bg.apply_update_norm(delta.bg.means.detach().norm(dim=-1, keepdim=True), fast_beta=fast, slow_beta=slow).items()
            }
        )
        if self.distant is not None and delta.distant is not None:
            aux.update(
                {
                    f"v3/history/distant_{k}": v
                    for k, v in self.distant.apply_update_norm(
                        delta.distant.means.detach().norm(dim=-1, keepdim=True),
                        fast_beta=fast,
                        slow_beta=slow,
                    ).items()
                }
            )
        if self.rigid is not None and delta.rigid is not None:
            aux.update(
                {
                    f"v3/history/rigid_{k}": v
                    for k, v in self.rigid.apply_update_norm(
                        delta.rigid.means.detach().norm(dim=-1, keepdim=True),
                        fast_beta=fast,
                        slow_beta=slow,
                    ).items()
                }
            )
        return aux

    def commit_residual(
        self,
        pack: IForwardResidualPack,
        *,
        residual_betas: Dict[str, float],
        support_min: Dict[str, float],
    ) -> Dict[str, float]:
        aux: Dict[str, float] = {}
        fast = float(residual_betas["fast_beta"])
        slow = float(residual_betas["slow_beta"])
        aux.update(
            {
                f"v3/history/bg_{k}": v
                for k, v in self.bg.apply_residual(
                    error_cur=pack.error_bg,
                    support_cur=pack.support_bg,
                    fast_beta=fast,
                    slow_beta=slow,
                    support_min=float(support_min.get("bg", 0.0)),
                ).items()
            }
        )
        if self.distant is not None and pack.error_distant is not None and pack.support_distant is not None:
            aux.update(
                {
                    f"v3/history/distant_{k}": v
                    for k, v in self.distant.apply_residual(
                        error_cur=pack.error_distant,
                        support_cur=pack.support_distant,
                        fast_beta=fast,
                        slow_beta=slow,
                        support_min=float(support_min.get("distant", 0.0)),
                    ).items()
                }
            )
        if self.rigid is not None and pack.error_rigid is not None and pack.support_rigid is not None:
            aux.update(
                {
                    f"v3/history/rigid_{k}": v
                    for k, v in self.rigid.apply_residual(
                        error_cur=pack.error_rigid,
                        support_cur=pack.support_rigid,
                        fast_beta=fast,
                        slow_beta=slow,
                        support_min=float(support_min.get("rigid", 0.0)),
                    ).items()
                }
            )
        return aux

    def stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(self.bg.stats("v3/history/bg"))
        if self.distant is not None:
            out.update(self.distant.stats("v3/history/distant"))
        if self.rigid is not None:
            out.update(self.rigid.stats("v3/history/rigid"))
        return out

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in ("bg", "distant", "rigid"):
            branch = getattr(self, name)
            if branch is None:
                out[f"{name}_history_initialized"] = 0.0
                out[f"{name}_history_capacity"] = 0.0
                out[f"{name}_history_initialized_ratio"] = 0.0
                continue
            initialized = (branch.initialized.detach() > 0).to(dtype=torch.bool)
            capacity = int(initialized.numel())
            seen = int(initialized.sum().item()) if capacity > 0 else 0
            out[f"{name}_history_initialized"] = float(seen)
            out[f"{name}_history_capacity"] = float(capacity)
            out[f"{name}_history_initialized_ratio"] = float(seen) / float(max(capacity, 1))
        return out


__all__ = [
    "IForwardHistoryBranchEMA",
    "IForwardHistoryEMAState",
    "IForwardResidualPack",
]
