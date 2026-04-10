"""
Minimal StreetForward Stage 4.0: Stage 3.3 + rigid node branch (MVP).

Stage 4.0 constraints:
- rigid branch has independent init/limits/eta/mlp config
- rigid uses 2D-only features (use_3d_feat must be false)
- src=target simplification (single target = source frame/view)
- rigid pointcloud is local coords, transformed to world by dynamic_info poses
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _normalize_quat,
    _pairwise_neighbor_distances,
    _quat_multiply,
    _quat_to_rotmat,
    _sh_to_rgb,
)
from models.streetforward.minimal_trainer_stage3_2d import _create_proxy_params
from models.streetforward.minimal_trainer_stage3_3 import MinimalStreetForwardStage3_3
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant, NodeStateRigid

logger = logging.getLogger(__name__)


def spatial_hw_from_image_tensor(img: torch.Tensor) -> Tuple[int, int]:
    """
    Return (H, W) for a single image or a batch of identical layouts.

    - 3D HWC: [H, W, C]
    - 3D CHW (RGB): [3, H, W]
    - 4D NHWC: [N, H, W, C]
    - 4D NCHW: [N, C, H, W] when C is small (<= 4), e.g. RGB or single-channel CHW
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"spatial_hw_from_image_tensor expected torch.Tensor, got {type(img)}")
    d = int(img.dim())
    if d == 3:
        if int(img.shape[0]) == 3:
            return int(img.shape[1]), int(img.shape[2])
        return int(img.shape[0]), int(img.shape[1])
    if d == 4:
        c1 = int(img.shape[1])
        c2 = int(img.shape[2])
        c3 = int(img.shape[3])
        if c1 <= 4 and c2 > c1 and c3 > c1:
            return c2, c3
        return c1, c2
    raise ValueError(f"spatial_hw_from_image_tensor expects dim 3 or 4, got {d}")


def merge_debug_stats_as_perf_floats(dest: Dict[str, float], prefix: str, raw: Dict[str, Any]) -> None:
    """
    Copy debug stats into dest with string keys suitable for _perf_acc (all float values).

    Scalar values are stored as dest[prefix + k]. List[float|int] values are expanded into
    dest[prefix + k + _sum/_mean/_max/_min/_len] so callers never call float(list).
    """
    for k, v in raw.items():
        key = f"{prefix}{k}"
        if isinstance(v, (bool, int, float)):
            dest[key] = float(v)
            continue
        if isinstance(v, list):
            if len(v) == 0:
                dest[f"{key}_len"] = 0.0
                continue
            if not all(isinstance(x, (bool, int, float)) for x in v):
                continue
            xs = [float(x) for x in v]
            dest[f"{key}_sum"] = float(sum(xs))
            dest[f"{key}_mean"] = float(sum(xs) / len(xs))
            dest[f"{key}_max"] = float(max(xs))
            dest[f"{key}_min"] = float(min(xs))
            dest[f"{key}_len"] = float(len(xs))


def _append_backward_pair(
    render_tensors: List[torch.Tensor],
    grad_tensors: List[torch.Tensor],
    render_tensor: torch.Tensor,
    grad_tensor: torch.Tensor,
) -> None:
    """Only pair proxy grads with render tensors that participate in autograd (e.g. skip frozen sky means/quats)."""
    if render_tensor.requires_grad:
        render_tensors.append(render_tensor)
        grad_tensors.append(grad_tensor)


def _merge_params_bg_rigid_distant(
    proxies_bg: Dict[str, torch.Tensor],
    proxies_rigid_world: Optional[Dict[str, torch.Tensor]],
    proxies_distant: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    chunks = [proxies_bg]
    if proxies_rigid_world is not None:
        chunks.append(proxies_rigid_world)
    if proxies_distant is not None:
        chunks.append(proxies_distant)
    means = torch.cat([p["means_p"] for p in chunks], dim=0)
    quats = torch.cat([p["quats_p"] for p in chunks], dim=0)
    scales = torch.cat([p["scales_p"] for p in chunks], dim=0)
    opacities = torch.cat([p["opacities_p"] for p in chunks], dim=0)
    colors = torch.cat([p["colors_p"] for p in chunks], dim=0)
    return {
        "means_r": means,
        "scales_r": scales,
        "quats_r": quats,
        "opacities_r": opacities,
        "colors_r": colors,
    }


def _backward_to_render_params_bg_rigid_distant(
    render_params_bg: Dict[str, torch.Tensor],
    proxies_bg: Dict[str, torch.Tensor],
    render_params_rigid_world: Optional[Dict[str, torch.Tensor]],
    proxies_rigid_world: Optional[Dict[str, torch.Tensor]],
    render_params_distant: Optional[Dict[str, torch.Tensor]],
    proxies_distant: Optional[Dict[str, torch.Tensor]],
    rigid_world_proxy_pairs: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
) -> None:
    def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
        return t.grad if t.grad is not None else torch.zeros_like(t)

    render_tensors: List[torch.Tensor] = []
    grad_tensors: List[torch.Tensor] = []
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["means_r"], _grad_or_zero(proxies_bg["means_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["scales_r"], _grad_or_zero(proxies_bg["scales_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["quats_r"], _grad_or_zero(proxies_bg["quats_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["opacities_r"], _grad_or_zero(proxies_bg["opacities_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["colors_r"], _grad_or_zero(proxies_bg["colors_p"]))
    if rigid_world_proxy_pairs is not None:
        if render_params_rigid_world is not None or proxies_rigid_world is not None:
            raise ValueError("Use either rigid_world_proxy_pairs or single rigid world/proxy pair, not both.")
        for rp_w, px_w in rigid_world_proxy_pairs:
            _append_backward_pair(render_tensors, grad_tensors, rp_w["means_r"], _grad_or_zero(px_w["means_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["scales_r"], _grad_or_zero(px_w["scales_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["quats_r"], _grad_or_zero(px_w["quats_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["opacities_r"], _grad_or_zero(px_w["opacities_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["colors_r"], _grad_or_zero(px_w["colors_p"]))
    elif render_params_rigid_world is not None and proxies_rigid_world is not None:
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["means_r"], _grad_or_zero(proxies_rigid_world["means_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["scales_r"], _grad_or_zero(proxies_rigid_world["scales_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["quats_r"], _grad_or_zero(proxies_rigid_world["quats_p"])
        )
        _append_backward_pair(
            render_tensors,
            grad_tensors,
            render_params_rigid_world["opacities_r"],
            _grad_or_zero(proxies_rigid_world["opacities_p"]),
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["colors_r"], _grad_or_zero(proxies_rigid_world["colors_p"])
        )
    if render_params_distant is not None and proxies_distant is not None:
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["means_r"], _grad_or_zero(proxies_distant["means_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["scales_r"], _grad_or_zero(proxies_distant["scales_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["quats_r"], _grad_or_zero(proxies_distant["quats_p"])
        )
        _append_backward_pair(
            render_tensors,
            grad_tensors,
            render_params_distant["opacities_r"],
            _grad_or_zero(proxies_distant["opacities_p"]),
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["colors_r"], _grad_or_zero(proxies_distant["colors_p"])
        )
    if render_tensors:
        torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)


def _merge_params_sky_only(proxies_sky: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "means_r": proxies_sky["means_p"],
        "scales_r": proxies_sky["scales_p"],
        "quats_r": proxies_sky["quats_p"],
        "opacities_r": proxies_sky["opacities_p"],
        "colors_r": proxies_sky["colors_p"],
    }


def _tensor_merge_sky_only(render_params_sky: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "means_r": render_params_sky["means_r"],
        "scales_r": render_params_sky["scales_r"],
        "quats_r": render_params_sky["quats_r"],
        "opacities_r": render_params_sky["opacities_r"],
        "colors_r": render_params_sky["colors_r"],
    }


def _backward_to_render_params_bg_rigid_distant_sky(
    render_params_bg: Dict[str, torch.Tensor],
    proxies_bg: Dict[str, torch.Tensor],
    render_params_rigid_world: Optional[Dict[str, torch.Tensor]],
    proxies_rigid_world: Optional[Dict[str, torch.Tensor]],
    render_params_distant: Optional[Dict[str, torch.Tensor]],
    proxies_distant: Optional[Dict[str, torch.Tensor]],
    render_params_sky: Optional[Dict[str, torch.Tensor]],
    proxies_sky: Optional[Dict[str, torch.Tensor]],
    rigid_world_proxy_pairs: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
) -> None:
    def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
        return t.grad if t.grad is not None else torch.zeros_like(t)

    render_tensors: List[torch.Tensor] = []
    grad_tensors: List[torch.Tensor] = []
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["means_r"], _grad_or_zero(proxies_bg["means_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["scales_r"], _grad_or_zero(proxies_bg["scales_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["quats_r"], _grad_or_zero(proxies_bg["quats_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["opacities_r"], _grad_or_zero(proxies_bg["opacities_p"]))
    _append_backward_pair(render_tensors, grad_tensors, render_params_bg["colors_r"], _grad_or_zero(proxies_bg["colors_p"]))
    if rigid_world_proxy_pairs is not None:
        if render_params_rigid_world is not None or proxies_rigid_world is not None:
            raise ValueError("Use either rigid_world_proxy_pairs or single rigid world/proxy pair, not both.")
        for rp_w, px_w in rigid_world_proxy_pairs:
            _append_backward_pair(render_tensors, grad_tensors, rp_w["means_r"], _grad_or_zero(px_w["means_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["scales_r"], _grad_or_zero(px_w["scales_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["quats_r"], _grad_or_zero(px_w["quats_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["opacities_r"], _grad_or_zero(px_w["opacities_p"]))
            _append_backward_pair(render_tensors, grad_tensors, rp_w["colors_r"], _grad_or_zero(px_w["colors_p"]))
    elif render_params_rigid_world is not None and proxies_rigid_world is not None:
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["means_r"], _grad_or_zero(proxies_rigid_world["means_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["scales_r"], _grad_or_zero(proxies_rigid_world["scales_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["quats_r"], _grad_or_zero(proxies_rigid_world["quats_p"])
        )
        _append_backward_pair(
            render_tensors,
            grad_tensors,
            render_params_rigid_world["opacities_r"],
            _grad_or_zero(proxies_rigid_world["opacities_p"]),
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_rigid_world["colors_r"], _grad_or_zero(proxies_rigid_world["colors_p"])
        )
    if render_params_distant is not None and proxies_distant is not None:
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["means_r"], _grad_or_zero(proxies_distant["means_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["scales_r"], _grad_or_zero(proxies_distant["scales_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["quats_r"], _grad_or_zero(proxies_distant["quats_p"])
        )
        _append_backward_pair(
            render_tensors,
            grad_tensors,
            render_params_distant["opacities_r"],
            _grad_or_zero(proxies_distant["opacities_p"]),
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_distant["colors_r"], _grad_or_zero(proxies_distant["colors_p"])
        )
    if render_params_sky is not None and proxies_sky is not None:
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_sky["means_r"], _grad_or_zero(proxies_sky["means_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_sky["scales_r"], _grad_or_zero(proxies_sky["scales_p"])
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_sky["quats_r"], _grad_or_zero(proxies_sky["quats_p"])
        )
        _append_backward_pair(
            render_tensors,
            grad_tensors,
            render_params_sky["opacities_r"],
            _grad_or_zero(proxies_sky["opacities_p"]),
        )
        _append_backward_pair(
            render_tensors, grad_tensors, render_params_sky["colors_r"], _grad_or_zero(proxies_sky["colors_p"])
        )
    if render_tensors:
        torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)


class MinimalStreetForwardStage4_0(MinimalStreetForwardStage3_3):
    """Stage 4.0 trainer built on top of Stage 3.3 with rigid branch."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        branches = self._require_key(config.model, "branches", "model")
        rigid = self._require_key(branches, "rigid", "model.branches")
        self.rigid_cfg = self._parse_branch_cfg(rigid, "rigid")
        if bool(self.rigid_cfg["mlp"]["use_3d_feat"]):
            raise ValueError("Stage4.0 requires model.branches.rigid.mlp.use_3d_feat=false")
        if not bool(self.rigid_cfg["mlp"]["use_2d_feat"]):
            raise ValueError("Stage4.0 requires model.branches.rigid.mlp.use_2d_feat=true")

        self.h_cache_rigid: Dict[Tuple[int, int], torch.Tensor] = {}
        self.node_states_rigid: Dict[Tuple[int, int], Optional[NodeStateRigid]] = {}
        self.rigid_freeze_means = bool(self.rigid_cfg["freeze_means"])
        self.rigid_freeze_quat = bool(self.rigid_cfg["mlp"]["freeze_quat"])

        feat_2d_dim = int(self.config.model.get("feat_2d_channels"))
        if feat_2d_dim <= 0:
            raise ValueError("Stage4.0 requires model.feat_2d_channels > 0 for rigid 2D-only branch.")
        self.rigid_feat_in_dim = feat_2d_dim
        self.rigid_feat_proj = nn.Linear(self.rigid_feat_in_dim, self.fused_in_dim).to(device)

        num_sh = _num_sh_bases(self.sh_degree)
        self.mlp_offset_pos_rigid = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(device)
        self.mlp_conv_rigid = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        ).to(device)
        self.mlp_opacity_rigid = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        self.gaussion_decoder_rigid = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)
        for m in (
            self.mlp_offset_pos_rigid,
            self.mlp_conv_rigid,
            self.mlp_opacity_rigid,
            self.gaussion_decoder_rigid,
        ):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(config.optimizer.get("lr")),
            eps=float(config.optimizer.get("eps")),
            weight_decay=float(config.optimizer.get("weight_decay")),
        )
        self._perf_acc: Dict[str, float] = {}

    def _assert_src_target_consistent(self, batch: Dict, targets: List[Dict]) -> None:
        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        if not source_views or not source_images:
            raise ValueError("Stage4.0 requires source_views/source_images.")
        if len(source_views) != len(targets):
            raise ValueError(
                f"Stage4.0 requires source/target one-to-one alignment. "
                f"Got len(source_views)={len(source_views)} vs len(targets)={len(targets)}."
            )
        for i, target in enumerate(targets):
            if "sky_mask" not in target or target["sky_mask"] is None:
                raise ValueError(f"Stage4.0 requires targets[{i}].sky_mask for mask supervision.")
            if "viewdirs" not in target or target["viewdirs"] is None:
                raise ValueError(f"Stage4.0 requires targets[{i}].viewdirs for sky branch.")
            src_view = source_views[i]
            tgt_view = target["view"]
            src_c2w = src_view.camtoworlds if hasattr(src_view, "camtoworlds") else src_view["camtoworlds"]
            tgt_c2w = tgt_view.camtoworlds if hasattr(tgt_view, "camtoworlds") else tgt_view["camtoworlds"]
            src_c2w = src_c2w if src_c2w.dim() == 2 else src_c2w[0]
            tgt_c2w = tgt_c2w if tgt_c2w.dim() == 2 else tgt_c2w[0]
            if not torch.allclose(src_c2w.to(self.device), tgt_c2w.to(self.device), atol=1e-4, rtol=1e-4):
                raise ValueError(f"Stage4.0 source/target camera mismatch at index {i}.")
            src_frame = int(batch.get("source_frame_idx", target.get("frame_idx", 0)))
            tgt_frame = int(target.get("frame_idx", src_frame))
            if src_frame != tgt_frame:
                raise ValueError(
                    f"Stage4.0 source/target frame mismatch at index {i}: source_frame_idx={src_frame}, target.frame_idx={tgt_frame}"
                )

    def _resolve_rigid_frame_idx(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> Optional[int]:
        if not node_state_rigid.frame_ids:
            return int(frame_idx)
        if frame_idx in node_state_rigid.frame_ids:
            return node_state_rigid.frame_ids.index(frame_idx)
        return None

    def _extend_rigid_frames(self, node_state_rigid: NodeStateRigid, dynamic_info: Dict) -> NodeStateRigid:
        if not dynamic_info:
            return node_state_rigid
        existing_frame_ids = set(node_state_rigid.frame_ids)
        candidate_frame_ids = [int(fid) for fid in dynamic_info.keys()]
        new_frame_ids = [fid for fid in candidate_frame_ids if fid not in existing_frame_ids]
        if not new_frame_ids:
            return node_state_rigid

        new_frame_ids = sorted(new_frame_ids)
        num_new_frames = len(new_frame_ids)
        num_instances = node_state_rigid.instances_quats.shape[1]
        device = node_state_rigid.instances_quats.device

        new_quats = torch.zeros((num_new_frames, num_instances, 4), device=device)
        new_trans = torch.zeros((num_new_frames, num_instances, 3), device=device)
        new_fv = torch.zeros((num_new_frames, num_instances), dtype=torch.bool, device=device)
        new_quats[..., 0] = 1.0

        if node_state_rigid.instance_ids:
            instance_id_map = {int(ins_id): idx for idx, ins_id in enumerate(node_state_rigid.instance_ids)}
        else:
            instance_id_map = {int(idx): idx for idx in range(num_instances)}

        for frame_slot, frame_id in enumerate(new_frame_ids):
            frame_info = dynamic_info.get(frame_id)
            if frame_info is None:
                frame_info = dynamic_info.get(str(frame_id))
            if not frame_info:
                continue
            instances = frame_info.get("instances", {})
            if not isinstance(instances, dict):
                continue
            for instance_id, instance_pose in instances.items():
                ins_id = int(instance_id)
                if ins_id not in instance_id_map:
                    continue
                ins_slot = instance_id_map[ins_id]
                new_quats[frame_slot, ins_slot] = torch.tensor(instance_pose["quat"], device=device)
                new_trans[frame_slot, ins_slot] = torch.tensor(instance_pose["trans"], device=device)
                new_fv[frame_slot, ins_slot] = True

        node_state_rigid.instances_quats = torch.cat([node_state_rigid.instances_quats, new_quats], dim=0)
        node_state_rigid.instances_trans = torch.cat([node_state_rigid.instances_trans, new_trans], dim=0)
        node_state_rigid.instances_fv = torch.cat([node_state_rigid.instances_fv, new_fv], dim=0)
        node_state_rigid.frame_ids.extend(new_frame_ids)
        return node_state_rigid

    def _rigid_instance_valid_mask(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved is None:
            raise ValueError(f"Rigid frame_idx={frame_idx} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}")
        return node_state_rigid.instances_fv[resolved].bool()

    def _rigid_point_valid_mask(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> torch.Tensor:
        ins_valid = self._rigid_instance_valid_mask(node_state_rigid, frame_idx)
        point_ids = node_state_rigid.point_ids[..., 0]
        return ins_valid[point_ids]

    def _transform_rigid_to_world(
        self,
        node_state_rigid: NodeStateRigid,
        means_local: torch.Tensor,
        frame_idx: int,
        point_ids_subset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved is None:
            raise ValueError(f"Rigid frame_idx={frame_idx} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}")
        quats_cur = node_state_rigid.instances_quats[resolved]
        trans_cur = node_state_rigid.instances_trans[resolved]
        rot_cur = _quat_to_rotmat(quats_cur)
        point_ids = node_state_rigid.point_ids[..., 0] if point_ids_subset is None else point_ids_subset
        rot_pts = rot_cur[point_ids.long()]
        trans_pts = trans_cur[point_ids.long()]
        return torch.bmm(rot_pts, means_local.unsqueeze(-1)).squeeze(-1) + trans_pts

    def _transform_rigid_quats_to_world(
        self,
        node_state_rigid: NodeStateRigid,
        quats_local: torch.Tensor,
        frame_idx: int,
        point_ids_subset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        resolved = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved is None:
            raise ValueError(f"Rigid frame_idx={frame_idx} missing in dynamic_info frame_ids={node_state_rigid.frame_ids}")
        quats_cur = node_state_rigid.instances_quats[resolved]
        point_ids = node_state_rigid.point_ids[..., 0] if point_ids_subset is None else point_ids_subset
        quats_pts = quats_cur[point_ids.long()]
        return _normalize_quat(_quat_multiply(quats_pts, quats_local))

    def _init_rigid_node_state_from_pcd(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        point_ids: torch.Tensor,
        dynamic_info: Dict,
        frame_ids: List[int],
        instance_id_map: Dict[int, int],
        instance_ids: List[int],
    ) -> NodeStateRigid:
        if points.shape[0] == 0:
            raise ValueError("Empty rigid pointcloud.")
        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        if colors_tensor.dim() == 1:
            colors_tensor = colors_tensor.unsqueeze(-1).expand(-1, 3)
        elif colors_tensor.shape[1] != 3:
            colors_tensor = colors_tensor[:, :3]
        from models.streetforward.math_utils import _rgb_to_sh

        scales_log = self._compute_initial_scales_by_cfg(means, self.rigid_cfg["init"])
        quats = torch.zeros((means.shape[0], 4), device=self.device, dtype=means.dtype)
        quats[:, 0] = 1.0
        opacity_logit = torch.logit(
            torch.full((means.shape[0], 1), float(self.rigid_cfg["init"]["opacity_init"]), device=self.device)
        )
        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_tensor)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        num_frames = len(frame_ids)
        num_instances = len(instance_id_map)
        instances_quats = torch.zeros(num_frames, num_instances, 4, device=self.device)
        instances_trans = torch.zeros(num_frames, num_instances, 3, device=self.device)
        instances_fv = torch.zeros(num_frames, num_instances, dtype=torch.bool, device=self.device)
        instances_quats[..., 0] = 1.0
        frame_id_map = {fid: i for i, fid in enumerate(frame_ids)}
        for frame_id, frame_info in dynamic_info.items():
            fid = int(frame_id)
            if fid not in frame_id_map:
                continue
            frame_slot = frame_id_map[fid]
            instances = frame_info.get("instances", {})
            if not isinstance(instances, dict):
                continue
            for instance_id, instance_pose in instances.items():
                ins = int(instance_id)
                if ins not in instance_id_map:
                    continue
                ins_slot = instance_id_map[ins]
                instances_quats[frame_slot, ins_slot] = torch.tensor(instance_pose["quat"], device=self.device)
                instances_trans[frame_slot, ins_slot] = torch.tensor(instance_pose["trans"], device=self.device)
                instances_fv[frame_slot, ins_slot] = True

        return NodeStateRigid(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
            point_ids=point_ids.detach().clone(),
            instances_quats=instances_quats.detach().clone(),
            instances_trans=instances_trans.detach().clone(),
            instances_fv=instances_fv.detach().clone(),
            instance_ids=list(instance_ids),
            frame_ids=list(frame_ids),
            cur_frame=0,
        )

    def _get_or_init_node_states_bg_rigid_distant(
        self, batch: Dict
    ) -> Tuple[NodeStateBackground, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        key = self._batch_key(batch)
        if key in self.node_states_bg and key in self.node_states_distant and key in self.node_states_rigid:
            node_state_rigid = self.node_states_rigid[key]
            dynamic_info = batch.get("dynamic_info")
            if node_state_rigid is not None and dynamic_info:
                node_state_rigid = self._extend_rigid_frames(node_state_rigid, dynamic_info)
                self.node_states_rigid[key] = node_state_rigid
            return self.node_states_bg[key], node_state_rigid, self.node_states_distant[key]

        node_state_bg, node_state_distant = super()._get_or_init_node_states_bg_distant(batch)
        node_state_rigid: Optional[NodeStateRigid] = None

        pointcloud = batch["pointcloud"]
        dynamic = pointcloud.get("dynamic") if isinstance(pointcloud, dict) else None
        if dynamic:
            dynamic_points = []
            dynamic_colors = []
            point_ids = []
            instance_ids = sorted(int(ins_id) for ins_id in dynamic.keys())
            instance_id_map = {ins_id: idx for idx, ins_id in enumerate(instance_ids)}
            for ins_id in instance_ids:
                instance_pcd = dynamic[ins_id]
                if instance_pcd is None or len(instance_pcd) == 0:
                    continue
                n_points = instance_pcd.shape[0]
                dynamic_points.append(instance_pcd[:, :3].astype(np.float32))
                dynamic_colors.append(instance_pcd[:, 3:6].astype(np.float32))
                point_ids.extend([instance_id_map[ins_id]] * n_points)
            if dynamic_points:
                dynamic_info = batch.get("dynamic_info")
                if not dynamic_info:
                    raise ValueError("Stage4.0 requires batch.dynamic_info when dynamic pointcloud exists.")
                d_points = np.concatenate(dynamic_points, axis=0)
                d_colors = np.concatenate(dynamic_colors, axis=0)
                point_ids_tensor = torch.tensor(point_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
                frame_ids = sorted(int(fid) for fid in dynamic_info.keys())
                node_state_rigid = self._init_rigid_node_state_from_pcd(
                    points=d_points,
                    colors=d_colors,
                    point_ids=point_ids_tensor,
                    dynamic_info=dynamic_info,
                    frame_ids=frame_ids,
                    instance_id_map=instance_id_map,
                    instance_ids=instance_ids,
                )

        self.node_states_rigid[key] = node_state_rigid
        return node_state_bg, node_state_rigid, node_state_distant

    def _predict_offsets_gru_rigid(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        head_rms_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if feat is None or feat.numel() == 0:
            num_points = params_for_embed["means"].shape[0]
            device = params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            identity_quat = torch.zeros(num_points, 4, device=device, dtype=dtype)
            identity_quat[:, 0] = 1.0
            num_sh = _num_sh_bases(self.sh_degree)
            return {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": identity_quat,
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(num_points, 3 * num_sh, device=device, dtype=dtype),
            }, h_old
        param_vec = self._normalize_params_for_embed(params_for_embed)
        param_embed = self.param_embed_norm(self.mlp_params_embed(param_vec))
        x = torch.cat([feat, param_embed], dim=-1)
        hx = torch.cat([h_old, x], dim=-1)
        z = torch.sigmoid(self.gru_update(hx))
        if self.gru_reset is not None:
            r = torch.sigmoid(self.gru_reset(hx))
            h_cand = torch.tanh(self.gru_candidate(torch.cat([r * h_old, x], dim=-1)))
        else:
            h_cand = torch.tanh(self.gru_candidate(hx))
        h_new = (1.0 - z) * h_old + z * h_cand
        head_input = self.gru_to_head(h_new)
        head_input = self._apply_gru_head_rms(head_input, head_rms_mask)
        offsets = self._predict_offsets_with_heads(
            head_input,
            limits=self.rigid_cfg["limits"],
            mlp_offset_pos=self.mlp_offset_pos_rigid,
            mlp_conv=self.mlp_conv_rigid,
            mlp_opacity=self.mlp_opacity_rigid,
            gaussion_decoder=self.gaussion_decoder_rigid,
            freeze_quat=self.rigid_freeze_quat,
        )
        return offsets, h_new

    def _compute_2d_features_for_gaussians(
        self,
        gaussians: Dict[str, torch.Tensor],
        source_views: List,
        source_images: List[torch.Tensor],
        height: int,
        width: int,
        return_accumulated_weights: bool = False,
        backprojector_override=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_gaussians = int(gaussians["means"].shape[0])
        if num_gaussians == 0:
            return None, None
        stats: Dict[str, float] = {}
        if torch.cuda.is_available():
            stats["cuda_mem_alloc_before"] = float(torch.cuda.memory_allocated())
            stats["cuda_mem_reserved_before"] = float(torch.cuda.memory_reserved())
        with torch.no_grad():
            render_rgb_out = self.alpha_t_extractor.render_rgb_only(
                gaussians, source_views, height, width, return_debug_stats=True
            )
        if isinstance(render_rgb_out, tuple):
            rendered_rgbs, rgb_stats = render_rgb_out
            stats.update({f"2d_rgb_{k}": float(v) for k, v in rgb_stats.items()})
        else:
            rendered_rgbs = render_rgb_out
        image_batch = torch.stack([img.to(self.device) for img in source_images], dim=0)
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)
        rendered_batch = torch.stack(rendered_rgbs, dim=0).detach()
        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        multi = torch.cat([image_batch, rendered_batch], dim=-1)
        features_2d = self.image_feature_extractor(multi)
        use_fused_v2 = bool(getattr(self, "use_fused_cuda_backproject_v2", False))
        backprojector_impl = backprojector_override if backprojector_override is not None else self.feature_backprojector
        if use_fused_v2:
            back_out = self.alpha_t_extractor_v2.render_and_backproject_streaming_fused(
                gaussians=gaussians,
                cameras=source_views,
                features_2d=features_2d,
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                backprojector=backprojector_impl,
                return_accumulated_weights=return_accumulated_weights,
                return_debug_stats=True,
            )
        else:
            back_out = self.alpha_t_extractor.render_and_backproject_streaming(
                gaussians=gaussians,
                cameras=source_views,
                features_2d=features_2d,
                height=height,
                width=width,
                num_gaussians=num_gaussians,
                backprojector=backprojector_impl,
                return_accumulated_weights=return_accumulated_weights,
                return_debug_stats=True,
            )
        if return_accumulated_weights:
            feat_2d_all, acc_w, bp_stats = back_out
            stats.update({f"2d_bp_{k}": float(v) for k, v in bp_stats.items()})
            if torch.cuda.is_available():
                stats["cuda_mem_alloc_after"] = float(torch.cuda.memory_allocated())
                stats["cuda_mem_reserved_after"] = float(torch.cuda.memory_reserved())
                stats["cuda_mem_alloc_delta"] = float(stats["cuda_mem_alloc_after"] - stats["cuda_mem_alloc_before"])
                stats["cuda_mem_reserved_delta"] = float(
                    stats["cuda_mem_reserved_after"] - stats["cuda_mem_reserved_before"]
                )
            for k, v in stats.items():
                self._perf_acc[k] = float(self._perf_acc.get(k, 0.0) + float(v))
            self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 1.0)
            return feat_2d_all, acc_w
        feat_2d_all, bp_stats = back_out
        stats.update({f"2d_bp_{k}": float(v) for k, v in bp_stats.items()})
        if torch.cuda.is_available():
            stats["cuda_mem_alloc_after"] = float(torch.cuda.memory_allocated())
            stats["cuda_mem_reserved_after"] = float(torch.cuda.memory_reserved())
            stats["cuda_mem_alloc_delta"] = float(stats["cuda_mem_alloc_after"] - stats["cuda_mem_alloc_before"])
            stats["cuda_mem_reserved_delta"] = float(
                stats["cuda_mem_reserved_after"] - stats["cuda_mem_reserved_before"]
            )
        for k, v in stats.items():
            self._perf_acc[k] = float(self._perf_acc.get(k, 0.0) + float(v))
        self._perf_acc["2d_call_count"] = float(self._perf_acc.get("2d_call_count", 0.0) + 1.0)
        return feat_2d_all, None

    def _render_params_from_offsets_rigid_local(
        self, node_state_rigid: NodeStateRigid, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        num_points = int(node_state_rigid.means.shape[0])
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)
        if self.rigid_freeze_means:
            means_r = node_state_rigid.means + offsets["offset_pos"] * 0.0
        else:
            means_r = node_state_rigid.means + self.rigid_cfg["eta"]["means"] * offsets["offset_pos"]
        scales_log_r = node_state_rigid.scales_log + self.rigid_cfg["eta"]["scales"] * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state_rigid.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state_rigid.opacity_logit + self.rigid_cfg["eta"]["opacity"] * offsets["offset_opacity"]
        sh_dc_r = node_state_rigid.sh_dc + self.rigid_cfg["eta"]["sh_dc"] * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state_rigid.sh_rest + self.rigid_cfg["eta"]["sh_rest"] * sh_rest_offset
        scales_r = torch.exp(scales_log_r)
        opacities_r = torch.sigmoid(opacity_logit_r).squeeze(-1)
        colors_r = torch.cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)
        return {
            "means_r": means_r,
            "scales_log_r": scales_log_r,
            "quats_r": quats_r,
            "opacity_logit_r": opacity_logit_r,
            "sh_dc_r": sh_dc_r,
            "sh_rest_r": sh_rest_r,
            "scales_r": scales_r,
            "opacities_r": opacities_r,
            "colors_r": colors_r,
        }

    def _rigid_local_to_world_render_params(
        self,
        node_state_rigid: NodeStateRigid,
        render_params_rigid_local: Dict[str, torch.Tensor],
        frame_idx: int,
        point_ids_subset: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        means_world = self._transform_rigid_to_world(
            node_state_rigid, render_params_rigid_local["means_r"], frame_idx, point_ids_subset=point_ids_subset
        )
        quats_world = self._transform_rigid_quats_to_world(
            node_state_rigid, render_params_rigid_local["quats_r"], frame_idx, point_ids_subset=point_ids_subset
        )
        return {
            **render_params_rigid_local,
            "means_r": means_world,
            "quats_r": quats_world,
        }

    def _update_node_state_rigid_local(
        self,
        node_state_rigid: NodeStateRigid,
        render_params_rigid_local: Dict[str, torch.Tensor],
        valid_idx: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if valid_idx.numel() == 0:
                return
            node_state_rigid.means[valid_idx] = render_params_rigid_local["means_r"].detach()
            node_state_rigid.scales_log[valid_idx] = render_params_rigid_local["scales_log_r"].detach()
            node_state_rigid.quats[valid_idx] = render_params_rigid_local["quats_r"].detach()
            node_state_rigid.opacity_logit[valid_idx] = render_params_rigid_local["opacity_logit_r"].detach()
            node_state_rigid.sh_dc[valid_idx] = render_params_rigid_local["sh_dc_r"].detach()
            node_state_rigid.sh_rest[valid_idx] = render_params_rigid_local["sh_rest_r"].detach()

    def forward(self, batch: Dict) -> Dict[str, Any]:
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage4.0 requires non-empty batch['targets'].")
        self._assert_src_target_consistent(batch, targets)
        node_state_bg, node_state_rigid, node_state_distant = self._get_or_init_node_states_bg_rigid_distant(batch)
        key = self._batch_key(batch)
        frame_idx = int(targets[0].get("frame_idx", batch.get("source_frame_idx", 0)))

        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        sample_img = source_images[0]
        height, width = spatial_hw_from_image_tensor(sample_img)

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)
        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)

        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(node_state_bg, node_state_distant)
        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)

        rigid_valid_idx = None
        rigid_point_ids_subset = None
        feat_rigid_input = None
        rigid_valid_total = 0
        rigid_total = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        if node_state_rigid is not None:
            rigid_valid_mask = self._rigid_point_valid_mask(node_state_rigid, frame_idx)
            rigid_valid_idx = torch.nonzero(rigid_valid_mask, as_tuple=False).squeeze(1)
            rigid_valid_total = int(rigid_valid_idx.numel())
            if rigid_total > 0 and rigid_valid_total == 0:
                logger.warning(
                    "No valid rigid points at source frame_idx=%d (rigid_total=%d). "
                    "This step will skip rigid branch and train bg+distant only.",
                    frame_idx,
                    rigid_total,
                )
            if rigid_valid_total > 0:
                rigid_point_ids_subset = node_state_rigid.point_ids[rigid_valid_idx, 0]
        if node_state_rigid is not None and rigid_valid_idx is not None and rigid_valid_idx.numel() > 0:
            means_local_valid = node_state_rigid.means[rigid_valid_idx]
            quats_local_valid = node_state_rigid.quats[rigid_valid_idx]
            gaussians_rigid = {
                "means": self._transform_rigid_to_world(
                    node_state_rigid, means_local_valid, frame_idx, point_ids_subset=rigid_point_ids_subset
                ),
                "quats": self._transform_rigid_quats_to_world(
                    node_state_rigid, quats_local_valid, frame_idx, point_ids_subset=rigid_point_ids_subset
                ),
                "scales": torch.exp(node_state_rigid.scales_log[rigid_valid_idx]),
                "opacities": torch.sigmoid(node_state_rigid.opacity_logit[rigid_valid_idx]).squeeze(-1),
                "colors": torch.cat([node_state_rigid.sh_dc[rigid_valid_idx, None, :], node_state_rigid.sh_rest[rigid_valid_idx]], dim=1),
            }
            feat_2d_rigid, _ = self._compute_2d_features_for_gaussians(
                gaussians=gaussians_rigid,
                source_views=source_views,
                source_images=source_images,
                height=height,
                width=width,
            )
            if feat_2d_rigid is not None:
                if int(feat_2d_rigid.shape[-1]) != int(self.rigid_feat_in_dim):
                    raise ValueError(
                        f"Rigid 2D feature dim mismatch: got {feat_2d_rigid.shape[-1]}, "
                        f"expected {self.rigid_feat_in_dim}"
                    )
                feat_rigid_input = self.rigid_feat_proj(feat_2d_rigid)

        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            feat_distant_input = self.distant_feat_proj(feat_2d_distant)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg")
        offsets_bg, h_new_bg = self._predict_offsets_gru(feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None)
        render_params_bg = self._render_params_from_offsets_bg(node_state_bg, offsets_bg)

        render_params_rigid_local = None
        render_params_rigid_world = None
        h_new_rigid = None
        if (
            node_state_rigid is not None
            and rigid_valid_idx is not None
            and rigid_valid_idx.numel() > 0
            and feat_rigid_input is not None
            and feat_rigid_input.numel() > 0
        ):
            # Align rigid param embedding with source/world semantics.
            class _RigidEmbedState:
                pass

            rigid_embed_state = _RigidEmbedState()
            rigid_embed_state.means = self._transform_rigid_to_world(
                node_state_rigid, node_state_rigid.means[rigid_valid_idx], frame_idx, point_ids_subset=rigid_point_ids_subset
            )
            rigid_embed_state.quats = self._transform_rigid_quats_to_world(
                node_state_rigid, node_state_rigid.quats[rigid_valid_idx], frame_idx, point_ids_subset=rigid_point_ids_subset
            )
            rigid_embed_state.scales_log = node_state_rigid.scales_log[rigid_valid_idx]
            rigid_embed_state.opacity_logit = node_state_rigid.opacity_logit[rigid_valid_idx]
            rigid_embed_state.sh_dc = node_state_rigid.sh_dc[rigid_valid_idx]
            rigid_embed_state.sh_rest = node_state_rigid.sh_rest[rigid_valid_idx]
            params_rigid = self._build_params_for_embed(rigid_embed_state, coord_space="world")
            h_old_rigid = self._get_or_init_hidden(
                self.h_cache_rigid, key, node_state_rigid.means.shape[0], node_state_rigid, "rigid"
            )
            h_old_rigid_valid = h_old_rigid[rigid_valid_idx]
            rigid_head_rms_mask = rigid_valid_mask[rigid_valid_idx].to(
                dtype=feat_rigid_input.dtype, device=feat_rigid_input.device
            )
            offsets_rigid, h_new_rigid_valid = self._predict_offsets_gru_rigid(
                feat_rigid_input,
                params_rigid,
                h_old_rigid_valid,
                head_rms_mask=rigid_head_rms_mask,
            )
            render_params_rigid_local = self._render_params_from_offsets_rigid_local(
                NodeStateRigid(
                    means=node_state_rigid.means[rigid_valid_idx],
                    scales_log=node_state_rigid.scales_log[rigid_valid_idx],
                    quats=node_state_rigid.quats[rigid_valid_idx],
                    opacity_logit=node_state_rigid.opacity_logit[rigid_valid_idx],
                    sh_dc=node_state_rigid.sh_dc[rigid_valid_idx],
                    sh_rest=node_state_rigid.sh_rest[rigid_valid_idx],
                    point_ids=node_state_rigid.point_ids[rigid_valid_idx],
                    instances_quats=node_state_rigid.instances_quats,
                    instances_trans=node_state_rigid.instances_trans,
                    instances_fv=node_state_rigid.instances_fv,
                    instance_ids=node_state_rigid.instance_ids,
                    frame_ids=node_state_rigid.frame_ids,
                    cur_frame=node_state_rigid.cur_frame,
                ),
                offsets_rigid,
            )
            render_params_rigid_world = self._rigid_local_to_world_render_params(
                node_state_rigid, render_params_rigid_local, frame_idx, point_ids_subset=rigid_point_ids_subset
            )
            h_new_rigid = h_old_rigid.clone()
            h_new_rigid[rigid_valid_idx] = h_new_rigid_valid

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant, key, node_state_distant.means.shape[0], node_state_distant, "distant"
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru_distant(feat_distant_input, params_distant, h_old_distant)
            render_params_distant = self._render_params_from_offsets_distant(node_state_distant, offsets_distant)

        merged_render = {
            "means_r": render_params_bg["means_r"],
            "scales_r": render_params_bg["scales_r"],
            "quats_r": render_params_bg["quats_r"],
            "opacities_r": render_params_bg["opacities_r"],
            "colors_r": render_params_bg["colors_r"],
        }
        if render_params_rigid_world is not None:
            merged_render = {
                "means_r": torch.cat([merged_render["means_r"], render_params_rigid_world["means_r"]], dim=0),
                "scales_r": torch.cat([merged_render["scales_r"], render_params_rigid_world["scales_r"]], dim=0),
                "quats_r": torch.cat([merged_render["quats_r"], render_params_rigid_world["quats_r"]], dim=0),
                "opacities_r": torch.cat([merged_render["opacities_r"], render_params_rigid_world["opacities_r"]], dim=0),
                "colors_r": torch.cat([merged_render["colors_r"], render_params_rigid_world["colors_r"]], dim=0),
            }
        if render_params_distant is not None:
            merged_render = {
                "means_r": torch.cat([merged_render["means_r"], render_params_distant["means_r"]], dim=0),
                "scales_r": torch.cat([merged_render["scales_r"], render_params_distant["scales_r"]], dim=0),
                "quats_r": torch.cat([merged_render["quats_r"], render_params_distant["quats_r"]], dim=0),
                "opacities_r": torch.cat([merged_render["opacities_r"], render_params_distant["opacities_r"]], dim=0),
                "colors_r": torch.cat([merged_render["colors_r"], render_params_distant["colors_r"]], dim=0),
            }

        if not self.training:
            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []
            multi_result = self._render_multi_view(merged_render, targets)
            if multi_result is not None:
                pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
                acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
                pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
                for i, target in enumerate(targets):
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    pred_rgbs.append(pred_stack[i])
                    gt_images.append(gt_image)
            else:
                for target in targets:
                    view = target["view"]
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    h, w = gt_image.shape[0], gt_image.shape[1]
                    pred_rgb, acc = self._render_single_view(merged_render, view, h, w)
                    pred_rgb = self._composite_sky(pred_rgb, acc, target)
                    pred_rgbs.append(pred_rgb)
                    gt_images.append(gt_image)
            return {
                "loss": torch.tensor(0.0, device=self.device),
                "render_params": render_params_bg,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_render_params_rigid_world": render_params_rigid_world,
                "_render_params_rigid_local": render_params_rigid_local,
                "_node_state_bg": node_state_bg,
                "_node_state_distant": node_state_distant,
                "_node_state_rigid": node_state_rigid,
                "_h_new_bg": h_new_bg,
                "_h_new_distant": h_new_distant,
                "_h_new_rigid": h_new_rigid,
                "_rigid_valid_idx": rigid_valid_idx,
                "_num_rigid_valid_src": rigid_valid_total,
                "_num_rigid_total": rigid_total,
                "_cache_key": key,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_rigid_world = _create_proxy_params(render_params_rigid_world) if render_params_rigid_world is not None else None
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        merged_for_render = _merge_params_bg_rigid_distant(proxies_bg, proxies_rigid_world, proxies_distant)

        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        opacities: List[torch.Tensor] = []
        multi_result = self._render_multi_view(merged_for_render, targets)
        if multi_result is not None:
            pred_stack = torch.stack([multi_result[i][0] for i in range(len(targets))], dim=0)
            acc_stack = torch.stack([multi_result[i][1] for i in range(len(targets))], dim=0)
            pred_stack = self._composite_sky_batched(pred_stack, acc_stack, targets)
            for i, target in enumerate(targets):
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                pred_rgbs.append(pred_stack[i])
                gt_images.append(gt_image)
                opacities.append(acc_stack[i])
        else:
            for target in targets:
                view = target["view"]
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                h, w = gt_image.shape[0], gt_image.shape[1]
                pred_rgb, acc = self._render_single_view(merged_for_render, view, h, w)
                pred_rgb = self._composite_sky(pred_rgb, acc, target)
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                opacities.append(acc.squeeze(-1) if acc.dim() == 3 and acc.shape[-1] == 1 else acc)

        from models.streetforward.metrics import compute_ssim_loss_masked

        loss_l1_list: List[torch.Tensor] = []
        loss_ssim_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        loss_entropy_list: List[torch.Tensor] = []
        loss_total_list: List[torch.Tensor] = []
        for i, target in enumerate(targets):
            pred_rgb = pred_rgbs[i]
            gt_image = gt_images[i]
            opacity = opacities[i].to(self.device).float()
            if opacity.dim() == 3 and opacity.shape[-1] == 1:
                opacity = opacity.squeeze(-1)
            h, w = gt_image.shape[0], gt_image.shape[1]
            valid_loss_mask = self._valid_loss_mask_from_target(target, height=h, width=w)
            l1_i = self.loss_w_l1 * torch.mean(torch.abs((pred_rgb - gt_image) * valid_loss_mask.unsqueeze(-1)))
            ssim_i = self.loss_w_ssim * compute_ssim_loss_masked(
                pred_rgb, gt_image, valid_mask=valid_loss_mask, sky_mask=None, data_range=1.0
            )
            sm = target["sky_mask"].to(self.device).float()
            if sm.dim() == 3:
                sm = sm.squeeze(-1)
            gt_occupied = (1.0 - sm) * valid_loss_mask
            pred_occupied = opacity.clamp(0.0, 1.0) * valid_loss_mask
            mask_i = self.loss_w_mask * self._mask_bce(pred_occupied, gt_occupied, valid_loss_mask)
            p = opacity.clamp(1e-6, 1.0 - 1e-6)
            entropy_i = self.loss_w_opacity_entropy * self._masked_mean(-p * torch.log(p), valid_loss_mask)
            total_i = l1_i + ssim_i + mask_i + entropy_i
            loss_l1_list.append(l1_i)
            loss_ssim_list.append(ssim_i)
            loss_mask_list.append(mask_i)
            loss_entropy_list.append(entropy_i)
            loss_total_list.append(total_i)

        l1_i = torch.stack(loss_l1_list).mean()
        ssim_i = torch.stack(loss_ssim_list).mean()
        mask_i = torch.stack(loss_mask_list).mean()
        entropy_i = torch.stack(loss_entropy_list).mean()
        loss = torch.stack(loss_total_list).mean()

        return {
            "loss": loss,
            "loss_l1": l1_i,
            "loss_ssim": ssim_i,
            "loss_mask": mask_i,
            "loss_opacity_entropy": entropy_i,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_proxies_distant": proxies_distant,
            "_proxies_rigid_world": proxies_rigid_world,
            "_render_params_distant": render_params_distant,
            "_render_params_rigid_world": render_params_rigid_world,
            "_render_params_rigid_local": render_params_rigid_local,
            "_node_state_bg": node_state_bg,
            "_node_state_distant": node_state_distant,
            "_node_state_rigid": node_state_rigid,
            "_h_new_bg": h_new_bg,
            "_h_new_distant": h_new_distant,
            "_h_new_rigid": h_new_rigid,
            "_rigid_valid_idx": rigid_valid_idx,
            "_num_rigid_valid_src": rigid_valid_total,
            "_num_rigid_total": rigid_total,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
        }

    def train_step(self, batch: Dict, step: Optional[int] = None) -> Dict[str, Any]:
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        if torch.is_tensor(out.get("loss")):
            out["loss"].backward()
        if out.get("proxies") is not None:
            _backward_to_render_params_bg_rigid_distant(
                out["render_params"],
                out["proxies"],
                out.get("_render_params_rigid_world"),
                out.get("_proxies_rigid_world"),
                out.get("_render_params_distant"),
                out.get("_proxies_distant"),
                rigid_world_proxy_pairs=out.get("_rigid_world_proxy_pairs"),
            )
        self.optimizer.step()
        if "_cache_key" in out:
            key = out["_cache_key"]
            if out.get("_h_new_bg") is not None:
                self.h_cache_bg[key] = out["_h_new_bg"].detach()
            if out.get("_h_new_distant") is not None:
                self.h_cache_distant[key] = out["_h_new_distant"].detach()
            if out.get("_h_new_rigid") is not None:
                self.h_cache_rigid[key] = out["_h_new_rigid"].detach()

        if self.update_node_state_interval > 0 and step is not None and step % self.update_node_state_interval == 0:
            if "_node_state_bg" in out:
                self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
            if out.get("_node_state_distant") is not None and out.get("_render_params_distant") is not None:
                self._update_node_state_distant(out["_node_state_distant"], out["_render_params_distant"])
            if out.get("_node_state_rigid") is not None and out.get("_render_params_rigid_local") is not None:
                valid_idx = out.get("_rigid_writeback_idx", out.get("_rigid_valid_idx"))
                if valid_idx is None:
                    raise ValueError("Internal error: missing _rigid_writeback_idx/_rigid_valid_idx for rigid node update.")
                self._update_node_state_rigid_local(out["_node_state_rigid"], out["_render_params_rigid_local"], valid_idx)
            if self.reset_node_state_interval > 0 and step % self.reset_node_state_interval == 0:
                self.reset_node_state()

        num_gaussians_bg = int(out["_node_state_bg"].means.shape[0])
        node_state_distant = out.get("_node_state_distant")
        node_state_rigid = out.get("_node_state_rigid")
        num_gaussians_distant = int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        num_gaussians_rigid = int(node_state_rigid.means.shape[0]) if node_state_rigid is not None else 0
        num_rigid_valid_src = int(out.get("_num_rigid_valid_src", 0))
        num_rigid_total = int(out.get("_num_rigid_total", num_gaussians_rigid))
        return {
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": num_gaussians_bg,
            "num_gaussians_distant": num_gaussians_distant,
            "num_gaussians_rigid": num_gaussians_rigid,
            "num_rigid_valid_src": num_rigid_valid_src,
            "num_rigid_invalid_src": int(max(num_rigid_total - num_rigid_valid_src, 0)),
            "rigid_valid_ratio": float(num_rigid_valid_src / max(num_rigid_total, 1)),
            "num_targets": len(batch.get("targets", [])),
            "num_source_views": len(batch.get("source_views", [])),
        }

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self.node_states_rigid.clear()
        self.h_cache_rigid.clear()


__all__ = [
    "MinimalStreetForwardStage4_0",
    "merge_debug_stats_as_perf_floats",
    "spatial_hw_from_image_tensor",
]
