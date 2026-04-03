from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class NodeStateBackground:
    """静态背景的节点状态，存储分离的高斯参数（世界坐标系）。"""

    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateBackground":
        return NodeStateBackground(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


@dataclass
class NodeStateRigid:
    """动态物体的节点状态，存储分离的高斯参数（局部坐标系）。"""

    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    point_ids: torch.Tensor
    instances_quats: torch.Tensor
    instances_trans: torch.Tensor
    instances_fv: torch.Tensor
    instance_ids: List[int]
    frame_ids: List[int]
    cur_frame: int

    def detach_clone(self) -> "NodeStateRigid":
        return NodeStateRigid(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
            point_ids=self.point_ids.detach().clone(),
            instances_quats=self.instances_quats.detach().clone(),
            instances_trans=self.instances_trans.detach().clone(),
            instances_fv=self.instances_fv.detach().clone(),
            instance_ids=list(self.instance_ids),
            frame_ids=list(self.frame_ids),
            cur_frame=int(self.cur_frame),
        )


@dataclass
class NodeStateDistant:
    """背景静态点的状态（crop_aabb 外、input_aabb 内）。"""

    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateDistant":
        return NodeStateDistant(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


@dataclass
class NodeStateSky:
    """Hemisphere sky shell Gaussians (world coordinates)."""

    means: torch.Tensor
    scales_log: torch.Tensor
    quats: torch.Tensor
    opacity_logit: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor

    def detach_clone(self) -> "NodeStateSky":
        return NodeStateSky(
            means=self.means.detach().clone(),
            scales_log=self.scales_log.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity_logit=self.opacity_logit.detach().clone(),
            sh_dc=self.sh_dc.detach().clone(),
            sh_rest=self.sh_rest.detach().clone(),
        )


NodeState = NodeStateBackground

__all__ = [
    "NodeStateBackground",
    "NodeStateRigid",
    "NodeStateDistant",
    "NodeStateSky",
    "NodeState",
]
