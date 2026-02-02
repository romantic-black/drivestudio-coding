from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.streetforward.math_utils import (
    _pairwise_neighbor_distances,
    _random_quat_tensor,
    _rgb_to_sh,
    _num_sh_bases,
    _quat_to_rotmat,
    _quat_multiply,
    _quat_conjugate,
    _normalize_quat,
)
from models.streetforward.node_states import NodeState, NodeStateBackground, NodeStateRigid, NodeStateDistant

logger = logging.getLogger(__name__)

@dataclass
class RigidMasks:
    """Container for precomputed rigid visibility masks used during an iteration."""
    mask_src_rigid: Optional[torch.Tensor] = None
    mask_tgt_rigid: List[torch.Tensor] = field(default_factory=list)
    mask_any_tgt_rigid: Optional[torch.Tensor] = None
    mask_update_rigid: Optional[torch.Tensor] = None
    idx_tgt_rigid: List[torch.Tensor] = field(default_factory=list)
    idx_src_rigid: Optional[torch.Tensor] = None


class NodeStateMixin:
    def _compute_initial_scales(self, means: torch.Tensor) -> torch.Tensor:
        """
        基于 k-NN 距离计算初始尺度（对数域）。
        
        Args:
            means: 点位置，形状 [N, 3]
            
        Returns:
            初始尺度对数，形状 [N, 3]
            
        方法：计算每个点到 k 个最近邻的平均距离，取对数作为初始尺度。
        使用 clamp 确保距离不小于 1e-3，避免对数域中的数值问题。
        """
        distances = _pairwise_neighbor_distances(means, k=3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        return torch.log(torch.clamp(avg_dist, min=1e-3).repeat(1, 3))

    def _init_node_state_from_arrays(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        state_cls,
    ):
        """
        从点和颜色数组初始化节点状态。
        """
        if len(points) == 0:
            raise ValueError("Empty point cloud provided for node state initialization.")

        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor

        initial_scales = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        return state_cls(
            means=means.detach().clone(),
            scales_log=initial_scales.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )

    def _init_node_from_pointcloud(
        self,
        scene_id: int,
        segment_id: int,
        pointcloud,
    ) -> NodeState:
        """
        从点云初始化静态背景的 NodeState。
        
        Args:
            scene_id: 场景ID
            segment_id: 片段ID
            pointcloud: 点云数据，可以是字典格式 {"background": [N, 6]} 或对象格式（需有 points 和 colors 属性）
            
        Returns:
            初始化的 NodeStateBackground
            
        处理流程：
        1. 提取点坐标和颜色（如果是字典格式，从 "background" 键获取）
        2. 将颜色归一化到 [0, 1] 范围（如果值域是 [0, 255]）
        3. 计算初始尺度（基于 k-NN 距离）
        4. 生成随机四元数
        5. 初始化不透明度为 logit(0.1)
        6. 将 RGB 转换为 SH DC 分量
        7. 初始化 SH rest 分量为零
        8. 所有参数初始化为分离状态
        """
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3]
            if background.shape[1] >= 6:
                colors = background[:, 3:]
                if colors.max() > 1.0 + 1e-3:
                    colors = colors / 255.0
            else:
                colors = np.zeros_like(points)
        else:
            points = np.asarray(pointcloud.points)  # type: ignore[attr-defined]
            colors = np.asarray(pointcloud.colors)  # type: ignore[attr-defined]
            if colors.max() > 1.0 + 1e-3:
                colors = colors / 255.0

        if len(points) == 0:
            raise ValueError(f"Empty point cloud for scene {scene_id}, segment {segment_id}")

        node_state = self._init_node_state_from_arrays(points, colors, NodeStateBackground)
        self.node_states[(scene_id, segment_id)] = node_state
        return node_state

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
        """
        从点云初始化动态物体的 NodeStateRigid。
        
        Args:
            points: 点坐标数组，形状 [N_rigid, 3]，局部坐标系
            colors: 颜色数组，形状 [N_rigid, 3]
            point_ids: 每个点属于哪个实例，形状 [N_rigid, 1]
            dynamic_info: 动态物体信息字典，包含各帧的实例位姿
            frame_ids: 帧ID列表
            instance_id_map: 实例ID到索引的映射
            instance_ids: 实例ID列表
            
        Returns:
            初始化的 NodeStateRigid
            
        处理流程：
        1. 将点坐标和颜色转换为张量，归一化颜色到 [0, 1]
        2. 计算初始尺度（基于 k-NN 距离）
        3. 生成随机四元数（局部旋转）
        4. 初始化不透明度为 logit(0.1)
        5. 将 RGB 转换为 SH DC 分量
        6. 初始化 SH rest 分量为零
        7. 从 dynamic_info 初始化 instances_quats、instances_trans 和 instances_fv
        8. 所有参数初始化为分离状态
        """
        means = torch.tensor(points, dtype=torch.float32, device=self.device)
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        colors_rgb = colors_tensor
        scales_log = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))

        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_rgb)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)

        num_frames = len(frame_ids)
        num_instances = len(instance_id_map)
        instances_quats = torch.zeros(num_frames, num_instances, 4, device=self.device)
        instances_trans = torch.zeros(num_frames, num_instances, 3, device=self.device)
        instances_fv = torch.zeros(num_frames, num_instances, dtype=torch.bool, device=self.device)
        instances_quats[..., 0] = 1.0

        frame_id_map = {fid: idx for idx, fid in enumerate(frame_ids)}
        for frame_id, frame_info in dynamic_info.items():
            frame_idx = int(frame_id)
            if frame_idx not in frame_id_map:
                continue
            frame_slot = frame_id_map[frame_idx]
            instances = frame_info.get("instances", {})
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        # Skip unmatched dynamic instances to tolerate annotation/pointcloud drift
                        continue
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=self.device)
                    trans = torch.tensor(instance_pose["trans"], device=self.device)
                    instances_quats[frame_slot, ins_slot] = quat
                    instances_trans[frame_slot, ins_slot] = trans
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

    def _get_or_init_node_states(
        self, batch: Dict
    ) -> Tuple[Tuple[int, int], NodeState, Optional[NodeStateRigid], Optional[NodeStateDistant]]:
        """
        获取或初始化双 NodeState（Background + RigidNodes）。
        
        Args:
            batch: 批次数据字典，需包含：
                - "scene_id": 场景ID
                - "segment_id": 片段ID
                - "pointcloud": 点云数据（字典格式包含 "background" 和可选的 "dynamic"）
                - "dynamic_info": 动态物体信息（可选）
                
        Returns:
            (key, node_state_bg, node_state_rigid, node_state_distant) 元组：
                - key: (scene_id, segment_id) 元组
                - node_state_bg: NodeStateBackground（静态背景）
                - node_state_rigid: NodeStateRigid 或 None（动态物体，如果存在）
                - node_state_distant: NodeStateDistant 或 None（背景远景，如果启用）
                
        处理流程：
        1. 如果 NodeState 已存在，直接返回（支持动态扩展帧信息）
        2. 如果 NodeState 不存在（新段开始），清空所有缓存以释放显存，然后从点云初始化
        3. 如果点云包含动态物体，会同时初始化 NodeStateRigid
        
        注意：当遇到新的 (scene_id, segment_id) 时，会自动清空之前的 node_states 缓存，
        只保留当前段的状态，以节省显存。这对于顺序训练多个段的场景特别有用。
        """
        scene_id = batch["scene_id"]
        if isinstance(scene_id, torch.Tensor):
            scene_id = int(scene_id.item())
        segment_id = batch["segment_id"]
        if isinstance(segment_id, torch.Tensor):
            segment_id = int(segment_id.item())
        key = (scene_id, segment_id)
        if key in self.node_states:
            node_state_rigid = self.node_states_rigid.get(key)
            node_state_distant = self.node_states_distant.get(key)
            dynamic_info = batch.get("dynamic_info")
            if node_state_rigid is not None and dynamic_info:
                node_state_rigid = self._extend_rigid_frames(node_state_rigid, dynamic_info)
                self.node_states_rigid[key] = node_state_rigid
            return key, self.node_states[key], node_state_rigid, node_state_distant
        
        # 如果 key 不存在，说明已经开始下一个段的训练，清空之前的缓存以释放显存
        if len(self.node_states) > 0:
            logger.debug(f"Clearing node_states cache before initializing new segment {key}. Previous cache had {len(self.node_states)} entries.")
            self.node_states.clear()
            self.node_states_rigid.clear()
            self.node_states_distant.clear()
            # 强制垃圾回收以释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        pointcloud = batch["pointcloud"]
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            if background.shape[1] >= 6:
                colors = background[:, 3:6].astype(np.float32)
            else:
                colors = np.zeros_like(points, dtype=np.float32)
        else:
            points = np.asarray(getattr(pointcloud, "points", np.zeros((0, 3))), dtype=np.float32)
            raw_colors = getattr(pointcloud, "colors", None)
            if raw_colors is not None:
                colors = np.asarray(raw_colors, dtype=np.float32)
                if colors.ndim == 1:
                    colors = np.expand_dims(colors, axis=0)
                if colors.shape[0] != points.shape[0]:
                    colors = np.zeros_like(points, dtype=np.float32)
            else:
                colors = np.zeros_like(points, dtype=np.float32)
        # 过滤到 input_aabb 范围内
        input_min = self.input_aabb_min.cpu().numpy()
        input_max = self.input_aabb_max.cpu().numpy()
        if points.size > 0:
            inside_mask = (
                (points >= input_min)
                & (points <= input_max)
            ).all(axis=1)
            points = points[inside_mask]
            colors = colors[inside_mask]

        crop_min = self.bbx_min.cpu().numpy()
        crop_max = self.bbx_max.cpu().numpy()
        in_crop_mask = (
            (points >= crop_min)
            & (points <= crop_max)
        ).all(axis=1)
        fg_points = points[in_crop_mask]
        fg_colors = colors[in_crop_mask]
        distant_points = points[~in_crop_mask]
        distant_colors = colors[~in_crop_mask]

        node_state_bg = self._init_node_state_from_arrays(fg_points, fg_colors, NodeStateBackground)
        node_state_distant: Optional[NodeStateDistant] = None
        if len(distant_points) > 0:
            node_state_distant = self._init_node_state_from_arrays(
                distant_points.astype(np.float32),
                distant_colors.astype(np.float32),
                NodeStateDistant,
            )
        self.node_states[(scene_id, segment_id)] = node_state_bg

        node_state_rigid: Optional[NodeStateRigid] = None
        if isinstance(pointcloud, dict) and pointcloud.get("dynamic"):
            dynamic_points = []
            dynamic_colors = []
            point_ids = []
            instance_ids = sorted(int(ins_id) for ins_id in pointcloud["dynamic"].keys())
            instance_id_map = {ins_id: idx for idx, ins_id in enumerate(instance_ids)}
            for ins_id in instance_ids:
                instance_pcd = pointcloud["dynamic"][ins_id]
                if instance_pcd is None or len(instance_pcd) == 0:
                    continue
                n_points = instance_pcd.shape[0]
                dynamic_points.append(instance_pcd[:, :3])
                dynamic_colors.append(instance_pcd[:, 3:6])
                point_ids.extend([instance_id_map[ins_id]] * n_points)

            if dynamic_points:
                dynamic_points = np.concatenate(dynamic_points, axis=0)
                dynamic_colors = np.concatenate(dynamic_colors, axis=0)
                point_ids_tensor = torch.tensor(point_ids, dtype=torch.long, device=self.device).unsqueeze(-1)
                dynamic_info = batch.get("dynamic_info")
                if not dynamic_info:
                    raise ValueError("dynamic_info is required when dynamic pointclouds are provided.")
                frame_ids = sorted(int(fid) for fid in dynamic_info.keys())
                # Filter dynamic_info to only instances present in pointcloud
                filtered_dynamic_info = {}
                for fid, finfo in dynamic_info.items():
                    inst = finfo.get("instances", {})
                    if not isinstance(inst, dict):
                        continue
                    filtered = {iid: pose for iid, pose in inst.items() if int(iid) in instance_id_map}
                    if filtered:
                        filtered_dynamic_info[int(fid)] = {"instances": filtered}
                if not filtered_dynamic_info:
                    dynamic_info = None
                else:
                    dynamic_info = filtered_dynamic_info
                node_state_rigid = self._init_rigid_node_state_from_pcd(
                    points=dynamic_points,
                    colors=dynamic_colors,
                    point_ids=point_ids_tensor,
                    dynamic_info=dynamic_info,
                    frame_ids=frame_ids,
                    instance_id_map=instance_id_map,
                    instance_ids=instance_ids,
                )

        self.node_states_rigid[key] = node_state_rigid
        self.node_states_distant[key] = node_state_distant
        return key, node_state_bg, node_state_rigid, node_state_distant

    def _extend_rigid_frames(self, node_state_rigid: NodeStateRigid, dynamic_info: Dict) -> NodeStateRigid:
        """
        扩展 RigidNodes 的帧信息，添加新的帧数据。
        
        Args:
            node_state_rigid: 现有的 RigidNodes
            dynamic_info: 动态物体信息字典，包含新帧的实例位姿
            
        Returns:
            扩展后的 RigidNodes
        
        如果 dynamic_info 中包含新的帧ID，会将这些帧的实例位姿添加到 instances_* 张量中。
        """
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
            if isinstance(instances, dict):
                for instance_id, instance_pose in instances.items():
                    ins_id = int(instance_id)
                    if ins_id not in instance_id_map:
                        raise ValueError(
                            f"Instance ID {ins_id} from dynamic_info not found in existing instance_ids. "
                            f"Existing instance IDs: {sorted(instance_id_map.keys())}"
                        )
                    ins_slot = instance_id_map[ins_id]
                    quat = torch.tensor(instance_pose["quat"], device=device)
                    trans = torch.tensor(instance_pose["trans"], device=device)
                    new_quats[frame_slot, ins_slot] = quat
                    new_trans[frame_slot, ins_slot] = trans
                    new_fv[frame_slot, ins_slot] = True

        node_state_rigid.instances_quats = torch.cat([node_state_rigid.instances_quats, new_quats], dim=0)
        node_state_rigid.instances_trans = torch.cat([node_state_rigid.instances_trans, new_trans], dim=0)
        node_state_rigid.instances_fv = torch.cat([node_state_rigid.instances_fv, new_fv], dim=0)
        node_state_rigid.frame_ids.extend(new_frame_ids)
        return node_state_rigid

    def _resolve_rigid_frame_idx(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> Optional[int]:
        """
        将 frame_idx（frame ID）解析为 frame_ids 列表中的索引。
        
        Args:
            frame_idx: 场景全局 frame ID（不是索引）
            
        Returns:
            frame_ids 列表中的索引，如果找不到则返回 None
        """
        if not node_state_rigid.frame_ids:
            # 如果没有 frame_ids，假设 frame_idx 就是索引
            return int(frame_idx)
        
        # 首先检查 frame_idx 是否是 frame ID
        if frame_idx in node_state_rigid.frame_ids:
            return node_state_rigid.frame_ids.index(frame_idx)
        
        # 如果找不到，返回 None（而不是抛出错误）
        return None

    def _precompute_rigid_masks(
        self,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
        targets: List[Dict],
    ) -> RigidMasks:
        """
        预计算 rigid 节点在 source/target 帧的可见性与索引，用于 gate offsets 与渲染子集。
        """
        if node_state_rigid is None:
            return RigidMasks()

        pose_valid_src = self._per_point_pose_valid(node_state_rigid, source_frame_idx)
        visible_src = self._visible_mask_from_instances_fv(node_state_rigid, source_frame_idx)
        mask_src_rigid = pose_valid_src & visible_src

        pose_valid_tgt = []
        visible_tgt = []
        for tgt in targets:
            frame_idx = int(tgt.get("frame_idx", source_frame_idx))
            pose_valid_tgt.append(self._per_point_pose_valid(node_state_rigid, frame_idx))
            visible_tgt.append(self._visible_mask_from_instances_fv(node_state_rigid, frame_idx))

        mask_tgt_rigid = [pv & vis for pv, vis in zip(pose_valid_tgt, visible_tgt)]
        mask_any_tgt_rigid = torch.zeros_like(mask_src_rigid)
        for m in mask_tgt_rigid:
            mask_any_tgt_rigid |= m

        mask_update_rigid = mask_src_rigid & mask_any_tgt_rigid
        idx_tgt_rigid = [torch.nonzero(m, as_tuple=False).squeeze(1) for m in mask_tgt_rigid]
        idx_src_rigid = torch.nonzero(mask_src_rigid, as_tuple=False).squeeze(1)

        return RigidMasks(
            mask_src_rigid=mask_src_rigid,
            mask_tgt_rigid=mask_tgt_rigid,
            mask_any_tgt_rigid=mask_any_tgt_rigid,
            mask_update_rigid=mask_update_rigid,
            idx_tgt_rigid=idx_tgt_rigid,
            idx_src_rigid=idx_src_rigid,
        )

    def _per_point_pose_valid(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> torch.Tensor:
        """
        计算每个 rigid 点在该帧是否有有效的位姿。
        
        Args:
            node_state_rigid: Rigid node state
            frame_idx: 帧索引（场景全局 frame_idx）
            
        Returns:
            [Nr] bool tensor，True 表示该点在该帧有有效位姿
        """
        resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved_frame_idx is None:
            return torch.zeros(node_state_rigid.means.shape[0], dtype=torch.bool, device=self.device)
        
        # 获取该帧的实例可见性（作为 pose_valid 的代理）
        visibility = node_state_rigid.instances_fv[resolved_frame_idx]  # [num_instances]
        
        # 扩展到每个点
        point_ids = node_state_rigid.point_ids[..., 0]  # [Nr]
        pose_valid = visibility[point_ids]  # [Nr]
        
        return pose_valid.bool()

    def _visible_mask_from_instances_fv(self, node_state_rigid: NodeStateRigid, frame_idx: int) -> torch.Tensor:
        """
        使用 instances_fv 计算可见性 mask。
        
        Args:
            node_state_rigid: Rigid node state
            frame_idx: 帧索引（场景全局 frame_idx）
            
        Returns:
            [Nr] bool tensor，True 表示该点在该帧可见
        """
        resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved_frame_idx is None:
            return torch.zeros(node_state_rigid.means.shape[0], dtype=torch.bool, device=self.device)
        
        # 获取该帧的实例可见性
        visibility = node_state_rigid.instances_fv[resolved_frame_idx]  # [num_instances]
        
        # 扩展到每个点
        point_ids = node_state_rigid.point_ids[..., 0]  # [Nr]
        visible = visibility[point_ids]  # [Nr]
        
        return visible.bool()

    def _update_node_states(
        self,
        render_params_bg: Dict[str, torch.Tensor],
        render_params_rigid: Optional[Dict[str, torch.Tensor]],
        render_params_distant: Optional[Dict[str, torch.Tensor]],
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
    ) -> None:
        """
        将渲染参数写回 NodeState，并在必要时进行 clamp。
        """
        with torch.no_grad():
            means_clamped = torch.clamp(render_params_bg["means_r"].detach(), min=self.bbx_min, max=self.bbx_max)
            node_state_bg.means.copy_(means_clamped)
            node_state_bg.scales_log.copy_(render_params_bg["scales_log_r"].detach())
            node_state_bg.quats.copy_(render_params_bg["quats_r"].detach())
            node_state_bg.opacity_logit.copy_(render_params_bg["opacity_logit_r"].detach())
            node_state_bg.sh_dc.copy_(render_params_bg["sh_dc_r"].detach())
            node_state_bg.sh_rest.copy_(render_params_bg["sh_rest_r"].detach())

            if node_state_rigid is not None and render_params_rigid is not None:
                node_state_rigid.means.copy_(render_params_rigid["means_r"].detach())
                node_state_rigid.scales_log.copy_(render_params_rigid["scales_log_r"].detach())
                node_state_rigid.quats.copy_(render_params_rigid["quats_r"].detach())
                node_state_rigid.opacity_logit.copy_(render_params_rigid["opacity_logit_r"].detach())
                node_state_rigid.sh_dc.copy_(render_params_rigid["sh_dc_r"].detach())
                node_state_rigid.sh_rest.copy_(render_params_rigid["sh_rest_r"].detach())

            if node_state_distant is not None and render_params_distant is not None:
                means_distant = torch.clamp(
                    render_params_distant["means_r"].detach(),
                    min=self.input_aabb_min,
                    max=self.input_aabb_max,
                )
                node_state_distant.means.copy_(means_distant)
                node_state_distant.scales_log.copy_(render_params_distant["scales_log_r"].detach())
                node_state_distant.quats.copy_(render_params_distant["quats_r"].detach())
                node_state_distant.opacity_logit.copy_(render_params_distant["opacity_logit_r"].detach())
                node_state_distant.sh_dc.copy_(render_params_distant["sh_dc_r"].detach())
                node_state_distant.sh_rest.copy_(render_params_distant["sh_rest_r"].detach())

    def _transform_rigid_to_world(
        self, node_state_rigid: NodeStateRigid, means_local: torch.Tensor, point_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        将动态物体的局部坐标位置变换到世界坐标。
        
        Args:
            node_state_rigid: Rigid node state，包含实例位姿信息
            means_local: 局部坐标的位置，形状 [N_rigid, 3]（可微）
            point_indices: 可选的索引，用于指定 means_local 对应的点索引。如果为 None，假设 means_local 对应所有点。
            
        Returns:
            世界坐标的位置，形状 [N_rigid, 3]（可微）
        
        变换公式：means_world = R * means_local + t
        其中 R 和 t 从 node_state_rigid.instances_* 中获取，根据 cur_frame 和 point_ids 选择。
        
        关键点：
        - 保持梯度连接，不使用 detach，让 PyTorch 自动处理梯度反向传播
        - 使用当前帧（cur_frame）的实例位姿进行变换
        - 如果提供了 point_indices，使用对应的 point_ids 子集来索引旋转和平移
        """
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        if frame_idx is None:
            # 如果没有有效的帧索引，返回零向量
            return torch.zeros_like(means_local)
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        trans_cur_frame = node_state_rigid.instances_trans[frame_idx]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)
        
        # 如果提供了 point_indices，使用对应的 point_ids 子集；否则使用完整的 point_ids
        if point_indices is not None:
            point_ids_subset = node_state_rigid.point_ids[point_indices, 0]  # [N_subset]
        else:
            point_ids_subset = node_state_rigid.point_ids[..., 0]  # [N_full]
        
        rot_per_pts = rot_cur_frame[point_ids_subset]
        trans_per_pts = trans_cur_frame[point_ids_subset]
        means_world = torch.bmm(rot_per_pts, means_local.unsqueeze(-1)).squeeze(-1) + trans_per_pts
        return means_world

    def _transform_rigid_quats_to_world(
        self, node_state_rigid: NodeStateRigid, quats_local: torch.Tensor, point_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        将动态物体的局部坐标旋转变换到世界坐标。
        
        Args:
            node_state_rigid: Rigid node state，包含实例旋转信息
            quats_local: 局部坐标的四元数，形状 [N_rigid, 4]（可微）
            point_indices: 可选的索引，用于指定 quats_local 对应的点索引。如果为 None，假设 quats_local 对应所有点。
            
        Returns:
            世界坐标的四元数，形状 [N_rigid, 4]（可微）
        
        变换公式：quats_world = normalize(quats_instance * quats_local)
        使用四元数乘法组合实例旋转和局部旋转。
        """
        frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, node_state_rigid.cur_frame)
        if frame_idx is None:
            # 如果没有有效的帧索引，返回单位四元数
            return quats_local
        quats_cur_frame = node_state_rigid.instances_quats[frame_idx]
        
        # 如果提供了 point_indices，使用对应的 point_ids 子集；否则使用完整的 point_ids
        if point_indices is not None:
            point_ids_subset = node_state_rigid.point_ids[point_indices, 0]  # [N_subset]
        else:
            point_ids_subset = node_state_rigid.point_ids[..., 0]  # [N_full]
        
        quats_per_pts = quats_cur_frame[point_ids_subset]
        return _normalize_quat(_quat_multiply(quats_per_pts, quats_local))
