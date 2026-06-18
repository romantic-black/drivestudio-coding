from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch


def _to_optional(x: Optional[torch.Tensor], *, device: torch.device) -> Optional[torch.Tensor]:
    return None if x is None else x.to(device=device)


def _detach_optional(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.detach().clone().cpu()


@dataclass
class BigGSBranchAssignment:
    branch: str
    child_to_parent: torch.Tensor
    child_order: torch.Tensor
    parent_start: torch.Tensor
    parent_count: torch.Tensor
    child_mass: torch.Tensor
    num_children: int
    num_parents: int
    object_id: Optional[torch.Tensor] = None
    parent_object_id: Optional[torch.Tensor] = None

    def to(self, *, device: torch.device) -> "BigGSBranchAssignment":
        return BigGSBranchAssignment(
            branch=str(self.branch),
            child_to_parent=self.child_to_parent.to(device=device),
            child_order=self.child_order.to(device=device),
            parent_start=self.parent_start.to(device=device),
            parent_count=self.parent_count.to(device=device),
            child_mass=self.child_mass.to(device=device),
            num_children=int(self.num_children),
            num_parents=int(self.num_parents),
            object_id=_to_optional(self.object_id, device=device),
            parent_object_id=_to_optional(self.parent_object_id, device=device),
        )

    def detach(self) -> "BigGSBranchAssignment":
        return BigGSBranchAssignment(
            branch=str(self.branch),
            child_to_parent=self.child_to_parent.detach().clone().cpu(),
            child_order=self.child_order.detach().clone().cpu(),
            parent_start=self.parent_start.detach().clone().cpu(),
            parent_count=self.parent_count.detach().clone().cpu(),
            child_mass=self.child_mass.detach().clone().cpu(),
            num_children=int(self.num_children),
            num_parents=int(self.num_parents),
            object_id=_detach_optional(self.object_id),
            parent_object_id=_detach_optional(self.parent_object_id),
        )

    @property
    def counts(self) -> Tuple[int, int]:
        return int(self.num_children), int(self.num_parents)


@dataclass
class BigGSRigidActiveAssignment:
    fine_S: torch.Tensor
    child_to_active_parent_S: torch.Tensor
    active_parent_global: torch.Tensor
    active_parent_count: torch.Tensor
    active_parent_start: torch.Tensor
    active_child_order_S: torch.Tensor
    child_mass_S: torch.Tensor
    parent_inside_mask: torch.Tensor
    child_inside_mask_S: torch.Tensor

    def to(self, *, device: torch.device) -> "BigGSRigidActiveAssignment":
        return BigGSRigidActiveAssignment(
            fine_S=self.fine_S.to(device=device),
            child_to_active_parent_S=self.child_to_active_parent_S.to(device=device),
            active_parent_global=self.active_parent_global.to(device=device),
            active_parent_count=self.active_parent_count.to(device=device),
            active_parent_start=self.active_parent_start.to(device=device),
            active_child_order_S=self.active_child_order_S.to(device=device),
            child_mass_S=self.child_mass_S.to(device=device),
            parent_inside_mask=self.parent_inside_mask.to(device=device),
            child_inside_mask_S=self.child_inside_mask_S.to(device=device),
        )

    def detach(self) -> "BigGSRigidActiveAssignment":
        return BigGSRigidActiveAssignment(
            fine_S=self.fine_S.detach().clone().cpu(),
            child_to_active_parent_S=self.child_to_active_parent_S.detach().clone().cpu(),
            active_parent_global=self.active_parent_global.detach().clone().cpu(),
            active_parent_count=self.active_parent_count.detach().clone().cpu(),
            active_parent_start=self.active_parent_start.detach().clone().cpu(),
            active_child_order_S=self.active_child_order_S.detach().clone().cpu(),
            child_mass_S=self.child_mass_S.detach().clone().cpu(),
            parent_inside_mask=self.parent_inside_mask.detach().clone().cpu(),
            child_inside_mask_S=self.child_inside_mask_S.detach().clone().cpu(),
        )


@dataclass
class IForwardBigGSState:
    bg: Optional[BigGSBranchAssignment] = None
    distant: Optional[BigGSBranchAssignment] = None
    rigid: Optional[BigGSBranchAssignment] = None
    scene_id: int = -1
    segment_id: int = -1
    episode_id: int = -1

    def to(self, *, device: torch.device) -> "IForwardBigGSState":
        return IForwardBigGSState(
            bg=None if self.bg is None else self.bg.to(device=device),
            distant=None if self.distant is None else self.distant.to(device=device),
            rigid=None if self.rigid is None else self.rigid.to(device=device),
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            episode_id=int(self.episode_id),
        )

    def detach(self) -> "IForwardBigGSState":
        return IForwardBigGSState(
            bg=None if self.bg is None else self.bg.detach(),
            distant=None if self.distant is None else self.distant.detach(),
            rigid=None if self.rigid is None else self.rigid.detach(),
            scene_id=int(self.scene_id),
            segment_id=int(self.segment_id),
            episode_id=int(self.episode_id),
        )

    def with_replaced(self, **kwargs) -> "IForwardBigGSState":
        return replace(self, **kwargs)


__all__ = [
    "BigGSBranchAssignment",
    "BigGSRigidActiveAssignment",
    "IForwardBigGSState",
]
