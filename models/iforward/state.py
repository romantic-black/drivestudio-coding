from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid
from models.streetforward.stage6_0.local_gs_state import LocalBranchState, LocalGSState


@dataclass
class KeyedMambaState:
    keys: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor

    def detach(self) -> "KeyedMambaState":
        return KeyedMambaState(
            keys=self.keys.detach().clone(),
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "KeyedMambaState":
        return KeyedMambaState(
            keys=self.keys.to(device=device),
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
        )


@dataclass
class DenseMambaState:
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seen: torch.Tensor

    def detach(self) -> "DenseMambaState":
        return DenseMambaState(
            conv_state=self.conv_state.detach().clone(),
            ssm_state=self.ssm_state.detach().clone(),
            seen=self.seen.detach().clone(),
        )

    def to(self, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> "DenseMambaState":
        return DenseMambaState(
            conv_state=self.conv_state.to(device=device, dtype=dtype or self.conv_state.dtype),
            ssm_state=self.ssm_state.to(device=device, dtype=dtype or self.ssm_state.dtype),
            seen=self.seen.to(device=device),
        )


@dataclass
class BranchMemoryState:
    point: Optional[KeyedMambaState] = None
    cell: Optional[KeyedMambaState] = None
    global_token: Optional[KeyedMambaState] = None
    dense_point: Optional[DenseMambaState] = None

    def detach(self) -> "BranchMemoryState":
        return BranchMemoryState(
            point=None if self.point is None else self.point.detach(),
            cell=None if self.cell is None else self.cell.detach(),
            global_token=None if self.global_token is None else self.global_token.detach(),
            dense_point=None if self.dense_point is None else self.dense_point.detach(),
        )


@dataclass
class IForwardMemoryState:
    bg: BranchMemoryState
    distant: BranchMemoryState
    rigid: BranchMemoryState

    @classmethod
    def empty(cls) -> "IForwardMemoryState":
        return cls(bg=BranchMemoryState(), distant=BranchMemoryState(), rigid=BranchMemoryState())

    def detach(self) -> "IForwardMemoryState":
        return IForwardMemoryState(
            bg=self.bg.detach(),
            distant=self.distant.detach(),
            rigid=self.rigid.detach(),
        )

    def count_tokens(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for branch_name in ("bg", "distant", "rigid"):
            branch = getattr(self, branch_name)
            for name in ("point", "cell", "global_token"):
                state = getattr(branch, name)
                if name == "point" and branch.dense_point is not None:
                    seen = branch.dense_point.seen.detach().to(dtype=torch.bool)
                else:
                    seen = state.seen.detach().to(dtype=torch.bool) if state is not None else None
                capacity = int(seen.numel()) if seen is not None else 0
                seen_count = int(seen.sum().item()) if seen is not None and capacity > 0 else 0
                out[f"{branch_name}_{name}"] = float(seen_count)
                out[f"{branch_name}_{name}_seen"] = float(seen_count)
                out[f"{branch_name}_{name}_capacity"] = float(capacity)
                out[f"{branch_name}_{name}_seen_ratio"] = float(seen_count) / float(max(capacity, 1))
        return out


def _detach_target_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {k: _detach_target_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_target_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_target_value(v) for v in value)
    return value


@dataclass
class IForwardShortMemoryEntry:
    frame_idx: int
    step_idx: int
    branch: str
    point_keys: torch.Tensor
    cell_keys: torch.Tensor
    global_keys: torch.Tensor
    event: torch.Tensor
    ctx: torch.Tensor
    support: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None

    def detach(self) -> "IForwardShortMemoryEntry":
        return IForwardShortMemoryEntry(
            frame_idx=int(self.frame_idx),
            step_idx=int(self.step_idx),
            branch=str(self.branch),
            point_keys=self.point_keys.detach().clone().cpu(),
            cell_keys=self.cell_keys.detach().clone().cpu(),
            global_keys=self.global_keys.detach().clone().cpu(),
            event=self.event.detach().clone().cpu(),
            ctx=self.ctx.detach().clone().cpu(),
            support=None if self.support is None else self.support.detach().clone().cpu(),
            valid=None if self.valid is None else self.valid.detach().clone().cpu(),
        )


@dataclass
class IForwardShortWindowHistory:
    entries: List[Dict[str, Any]]
    max_entries: int = 24
    memory_entries: List[IForwardShortMemoryEntry] = field(default_factory=list)
    max_memory_entries: int = 8

    @classmethod
    def empty(
        cls,
        *,
        max_entries: int = 24,
        max_memory_entries: int = 8,
    ) -> "IForwardShortWindowHistory":
        return cls(entries=[], max_entries=int(max_entries), memory_entries=[], max_memory_entries=int(max_memory_entries))

    def detach(self) -> "IForwardShortWindowHistory":
        return IForwardShortWindowHistory(
            entries=[_detach_target_value(item) for item in self.entries],
            max_entries=int(self.max_entries),
            memory_entries=[item.detach() for item in list(self.memory_entries or [])],
            max_memory_entries=int(self.max_memory_entries),
        )

    def commit_targets(self, batch: Dict[str, Any], target_indices: Tuple[int, ...]) -> "IForwardShortWindowHistory":
        targets = list(batch.get("targets") or [])
        new_entries = [_detach_target_value(targets[int(idx)]) for idx in target_indices if int(idx) < len(targets)]
        entries = list(self.entries) + new_entries
        if int(self.max_entries) > 0 and len(entries) > int(self.max_entries):
            entries = entries[-int(self.max_entries) :]
        return IForwardShortWindowHistory(
            entries=entries,
            max_entries=int(self.max_entries),
            memory_entries=list(self.memory_entries or []),
            max_memory_entries=int(self.max_memory_entries),
        )

    def commit_memory_entries(
        self,
        entries: List[IForwardShortMemoryEntry],
        *,
        detach: bool = True,
    ) -> "IForwardShortWindowHistory":
        new_entries = [item.detach() for item in entries] if bool(detach) else list(entries)
        memory_entries = list(self.memory_entries or []) + new_entries
        if int(self.max_memory_entries) > 0 and len(memory_entries) > int(self.max_memory_entries):
            memory_entries = memory_entries[-int(self.max_memory_entries) :]
        return IForwardShortWindowHistory(
            entries=list(self.entries),
            max_entries=int(self.max_entries),
            memory_entries=memory_entries,
            max_memory_entries=int(self.max_memory_entries),
        )

    @staticmethod
    def _read_from_entries(
        entries: List[IForwardShortMemoryEntry],
        *,
        query_keys: torch.Tensor,
        key_name: str,
        out: torch.Tensor,
        hit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(query_keys.numel()) == 0:
            return out, hit
        unresolved = ~hit
        if not bool(unresolved.any().item()):
            return out, hit
        query_cpu = query_keys.detach().cpu().tolist()
        for entry in reversed(entries):
            entry_keys = getattr(entry, key_name).detach().cpu().tolist()
            if not entry_keys:
                continue
            index = {int(key): int(i) for i, key in enumerate(entry_keys)}
            rows = []
            src = []
            unresolved_cpu = unresolved.detach().cpu().tolist()
            for row, is_unresolved in enumerate(unresolved_cpu):
                if not is_unresolved:
                    continue
                found = index.get(int(query_cpu[row]))
                if found is not None:
                    rows.append(int(row))
                    src.append(int(found))
            if not rows:
                continue
            row_t = torch.tensor(rows, device=out.device, dtype=torch.long)
            src_t = torch.tensor(src, device=out.device, dtype=torch.long)
            out[row_t] = entry.ctx.to(device=out.device, dtype=out.dtype)[src_t]
            hit[row_t] = True
            unresolved = ~hit
            if not bool(unresolved.any().item()):
                break
        return out, hit

    @staticmethod
    def _read_row_aligned_recent_context(
        entries: List[IForwardShortMemoryEntry],
        *,
        branch: str,
        point_keys: torch.Tensor,
        ref: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        if str(branch) not in {"bg", "distant"}:
            return None
        n = int(ref.shape[0])
        if n == 0 or int(point_keys.numel()) != n:
            return None
        for entry in reversed(entries):
            if str(entry.branch) != str(branch):
                continue
            if entry.ctx.dim() != 2:
                continue
            if int(entry.ctx.shape[0]) != n or int(entry.ctx.shape[1]) != int(ref.shape[1]):
                continue
            if int(entry.point_keys.numel()) != n:
                continue
            return entry.ctx.to(device=ref.device, dtype=ref.dtype), 1.0
        return None

    def read_context(
        self,
        *,
        branch: str,
        point_keys: torch.Tensor,
        cell_keys: torch.Tensor,
        global_keys: torch.Tensor,
        ref: torch.Tensor,
        drop: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        out = ref.new_zeros((int(ref.shape[0]), int(ref.shape[1])))
        if bool(drop) or int(ref.shape[0]) == 0:
            return out, 0.0
        branch_entries = [item for item in list(self.memory_entries or []) if str(item.branch) == str(branch)]
        if not branch_entries:
            return out, 0.0
        row_aligned = self._read_row_aligned_recent_context(
            branch_entries,
            branch=branch,
            point_keys=point_keys,
            ref=ref,
        )
        if row_aligned is not None:
            return row_aligned
        hit = torch.zeros((int(ref.shape[0]),), device=ref.device, dtype=torch.bool)
        out, hit = self._read_from_entries(branch_entries, query_keys=point_keys, key_name="point_keys", out=out, hit=hit)
        out, hit = self._read_from_entries(branch_entries, query_keys=cell_keys, key_name="cell_keys", out=out, hit=hit)
        out, hit = self._read_from_entries(branch_entries, query_keys=global_keys, key_name="global_keys", out=out, hit=hit)
        hit_ratio = float(hit.float().mean().item()) if hit.numel() else 0.0
        return out, hit_ratio

    def as_batch(self, ref_batch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(ref_batch)
        out["targets"] = [dict(item) for item in self.entries]
        return out


def detach_local_branch(branch: Optional[LocalBranchState]) -> Optional[LocalBranchState]:
    if branch is None:
        return None
    return replace(
        branch,
        means=branch.means.detach().clone(),
        scales_log=branch.scales_log.detach().clone(),
        quats=branch.quats.detach().clone(),
        opacity_logit=branch.opacity_logit.detach().clone(),
        sh_dc=branch.sh_dc.detach().clone(),
        sh_rest=branch.sh_rest.detach().clone(),
        hidden=branch.hidden.detach().clone(),
    )


def detach_local_gs_state(local_state: LocalGSState) -> LocalGSState:
    return LocalGSState(
        bg=detach_local_branch(local_state.bg),
        distant=detach_local_branch(local_state.distant),
        rigid=detach_local_branch(local_state.rigid),
        rigid_template=local_state.rigid_template.detach_clone() if local_state.rigid_template is not None else None,
    )


@dataclass
class IForwardState:
    local_gs: LocalGSState
    memory: IForwardMemoryState
    history: IForwardShortWindowHistory
    scene_id: int
    segment_id: int
    episode_id: int
    node_state_bg: Optional[NodeStateBackground] = None
    node_state_distant: Optional[NodeStateDistant] = None
    node_state_rigid: Optional[NodeStateRigid] = None

    @property
    def cache_key(self) -> Tuple[int, int, int]:
        return int(self.scene_id), int(self.segment_id), int(self.episode_id)

    def detach_for_next_rollout(self) -> "IForwardState":
        return IForwardState(
            local_gs=detach_local_gs_state(self.local_gs),
            memory=self.memory.detach(),
            history=self.history.detach(),
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            episode_id=int(self.episode_id),
            node_state_bg=None,
            node_state_distant=None,
            node_state_rigid=None,
        )
