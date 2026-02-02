from __future__ import annotations

from typing import Dict, Optional

import torch

from models.streetforward.logging_utils import _debug_log
from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _quat_to_rotmat,
    _normalize_quat,
    _quat_multiply,
    _quat_conjugate,
)
from models.streetforward.node_states import NodeStateRigid, NodeState



class OffsetsMixin:
    def _mask_rigid_offsets(
        self, offsets: Dict[str, torch.Tensor], visible_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        使用可见性掩码屏蔽动态物体的偏移量。
        
        Args:
            offsets: 偏移量字典
            visible_mask: 可见性掩码，形状 [N_rigid]，bool 类型，可选
            
        Returns:
            屏蔽后的偏移量字典
        
        对于不可见的点，将偏移量置零（位置、尺度、不透明度、SH）或设为单位四元数（旋转）。
        """
        if visible_mask is None or visible_mask.numel() == 0:
            return offsets
        mask = visible_mask.to(offsets["offset_pos"].device)
        mask_vec = mask.unsqueeze(-1).float()
        offset_quat = offsets["offset_quat"]
        identity_quat = torch.zeros_like(offset_quat)
        identity_quat[..., 0] = 1.0
        masked_offsets = {
            "offset_pos": offsets["offset_pos"] * mask_vec,
            "offset_scales": offsets["offset_scales"] * mask_vec,
            "offset_quat": torch.where(mask.unsqueeze(-1), offset_quat, identity_quat),
            "offset_opacity": offsets["offset_opacity"] * mask_vec,
            "offset_sh": offsets["offset_sh"] * mask_vec,
        }
        return masked_offsets

    def _predict_offsets(self, feat_3d_crop: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从 3D 特征预测 Gaussian 参数的偏移量。
        
        Args:
            feat_3d_crop: 3D 特征，形状 [N, outdim]（默认 outdim=32）
            
        Returns:
            偏移量字典，包含：
                - "offset_pos": 位置偏移，形状 [N, 3]，范围 [-offset_max, offset_max]
                - "offset_scales": 尺度对数偏移，形状 [N, 3]，范围 [-scale_max, scale_max]
                - "offset_quat": 四元数偏移，形状 [N, 4]，wxyz 格式（从轴角转换）
                - "offset_opacity": 不透明度对数偏移，形状 [N, 1]，范围 [-opacity_max, opacity_max]
                - "offset_sh": SH系数偏移，形状 [N, 3*num_sh]，包含DC和rest分量
        
        处理流程：
        1. 位置偏移：mlp_offset_pos → tanh → offset_max 缩放
        2. 尺度与旋转：mlp_conv → 分离尺度和轴角 → 分别 tanh 限制 → 轴角转四元数
        3. 不透明度偏移：mlp_opacity → tanh → opacity_max 缩放
        4. SH系数偏移：gaussion_decoder → 分离DC和rest → 分别 tanh 限制 → 合并
        
        注意：静态和动态使用相同的 MLP 网络预测偏移量。
        """
        # #region agent log
        if torch.cuda.is_available():
            _debug_log(
                "streetforward.py:_predict_offsets",
                "Start predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "feat_3d_crop_size_mb": feat_3d_crop.numel() * 4 / 1024**2,
                    "feat_3d_crop_shape": list(feat_3d_crop.shape),
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        # Position offset with tanh clamping
        offset_pos = self.offset_max * torch.tanh(self.mlp_offset_pos(feat_3d_crop))
        
        # Scale and rotation offsets
        scales_and_omega = self.mlp_conv(feat_3d_crop)
        offset_scales_raw, offset_omega_raw = scales_and_omega.split([3, 3], dim=-1)
        offset_scales = self.scale_max * torch.tanh(offset_scales_raw)
        offset_omega = self.omega_max * torch.tanh(offset_omega_raw)
        offset_quat = _axis_angle_to_quat(offset_omega)
        
        # Opacity offset with tanh clamping
        offset_opacity = self.opacity_max * torch.tanh(self.mlp_opacity(feat_3d_crop))
        
        # SH offsets with separate DC and rest
        sh_raw = self.gaussion_decoder(feat_3d_crop)
        sh_dc_raw = sh_raw[:, :3]
        sh_rest_raw = sh_raw[:, 3:]
        offset_sh_dc = self.sh_dc_max * torch.tanh(sh_dc_raw)
        offset_sh_rest = self.sh_rest_max * torch.tanh(sh_rest_raw)
        offset_sh = torch.cat([offset_sh_dc, offset_sh_rest], dim=-1)
        
        # #region agent log
        if torch.cuda.is_available():
            total_offset_size = (
                offset_pos.numel() * 4 + offset_scales.numel() * 4 + offset_quat.numel() * 4 +
                offset_opacity.numel() * 4 + offset_sh.numel() * 4
            ) / 1024**2
            _debug_log(
                "streetforward.py:_predict_offsets",
                "After predicting offsets",
                {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "total_offset_size_mb": total_offset_size,
                    "offset_shapes": {
                        "offset_pos": list(offset_pos.shape),
                        "offset_scales": list(offset_scales.shape),
                        "offset_quat": list(offset_quat.shape),
                        "offset_opacity": list(offset_opacity.shape),
                        "offset_sh": list(offset_sh.shape),
                    },
                },
                hypothesis_id="H5",
            )
        # #endregion
        
        return {
            "offset_pos": offset_pos,
            "offset_scales": offset_scales,
            "offset_quat": offset_quat,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
        }

    def _render_params_from_offsets(
        self, node_state: NodeState, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        从 NodeState 和偏移量计算渲染参数。
        
        Args:
            node_state: NodeState（Background 或 RigidNodes），所有参数都是分离的
            offsets: 偏移量字典（可微）
            
        Returns:
            渲染参数字典，包含：
                - "means_r": 渲染用的位置，形状 [N, 3]（可微，未clamp）
                - "scales_log_r": 渲染用的尺度对数，形状 [N, 3]（可微）
                - "scales_r": 渲染用的尺度，形状 [N, 3]（exp(scales_log_r)）
                - "quats_r": 渲染用的四元数，形状 [N, 4]（归一化，可微）
                - "opacity_logit_r": 渲染用的不透明度对数，形状 [N, 1]（可微）
                - "opacities_r": 渲染用的不透明度，形状 [N]（sigmoid(opacity_logit_r)）
                - "sh_dc_r": 渲染用的SH DC分量，形状 [N, 3]（可微）
                - "sh_rest_r": 渲染用的SH高阶分量，形状 [N, num_sh-1, 3]（可微）
                - "colors_r": 完整的SH系数，形状 [N, num_sh, 3]（用于渲染）
        
        关键点：
        - 应用步长因子（eta）控制偏移量幅度
        - means_r 不在此处进行 clamp，以保持梯度流
        - 使用四元数乘法组合旋转
        - 静态背景的渲染参数是世界坐标，动态物体的是局部坐标
        """
        num_points = node_state.means.shape[0]
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_flat = offsets["offset_sh"][:, 3:]
        sh_rest_offset = sh_rest_flat.view(num_points, num_sh - 1, 3)

        # Apply offsets with step size factors (eta)
        # Note: means_r is not clamped here to preserve gradient flow
        means_r = node_state.means + self.eta_means * offsets["offset_pos"]
        scales_log_r = node_state.scales_log + self.eta_scales * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state.opacity_logit + self.eta_opacity * offsets["offset_opacity"]
        sh_dc_r = node_state.sh_dc + self.eta_sh_dc * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state.sh_rest + self.eta_sh_rest * sh_rest_offset

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

    def _transform_offsets_world_to_local(
        self, node_state_rigid: NodeStateRigid, offsets_world: Dict[str, torch.Tensor], frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        将世界坐标的偏移量变换到局部坐标系。
        
        关键：offsets 是向量，变换方式与位置不同（只需要旋转，不需要平移）。
        
        Args:
            node_state_rigid: Rigid node state
            offsets_world: 世界坐标的偏移量字典
            frame_idx: 当前帧的 frame ID
            
        Returns:
            局部坐标的偏移量字典
        """
        resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, frame_idx)
        if resolved_frame_idx is None:
            # 没有对应的帧姿态时，返回全零/单位四元数，避免索引错误
            zero_like_pos = torch.zeros_like(offsets_world["offset_pos"])
            zero_like_scales = torch.zeros_like(offsets_world["offset_scales"])
            zero_like_opacity = torch.zeros_like(offsets_world["offset_opacity"])
            zero_like_sh = torch.zeros_like(offsets_world["offset_sh"])
            identity_quat = torch.zeros_like(offsets_world["offset_quat"])
            identity_quat[..., 0] = 1.0
            return {
                "offset_pos": zero_like_pos,
                "offset_scales": zero_like_scales,
                "offset_quat": identity_quat,
                "offset_opacity": zero_like_opacity,
                "offset_sh": zero_like_sh,
            }
        
        # 获取当前帧的旋转矩阵
        quats_cur_frame = node_state_rigid.instances_quats[resolved_frame_idx]  # [num_instances, 4]
        rot_cur_frame = _quat_to_rotmat(quats_cur_frame)  # [num_instances, 3, 3]
        rot_per_pts = rot_cur_frame[node_state_rigid.point_ids[..., 0]]  # [N_rigid, 3, 3]
        
        # 将世界坐标的位置偏移量变换到局部坐标
        # 对于向量（偏移量），只需要旋转，不需要平移
        offset_pos_world = offsets_world["offset_pos"]  # [N_rigid, 3]
        offset_pos_local = torch.bmm(
            rot_per_pts.transpose(-2, -1),  # R^T: [N_rigid, 3, 3]
            offset_pos_world.unsqueeze(-1)  # [N_rigid, 3, 1]
        ).squeeze(-1)  # [N_rigid, 3]
        
        # 将世界坐标的旋转增量转换到局部坐标：q_local = q_inst^{-1} * q_world * q_inst
        offset_quat_world = offsets_world["offset_quat"]
        quats_per_pts = _normalize_quat(node_state_rigid.instances_quats[resolved_frame_idx][node_state_rigid.point_ids[..., 0]])
        quats_inv = _quat_conjugate(quats_per_pts)
        offset_quat = _normalize_quat(_quat_multiply(_quat_multiply(quats_inv, offset_quat_world), quats_per_pts))
        
        # 其他偏移量（scales, opacity, sh）是标量或颜色，不需要坐标变换
        return {
            "offset_pos": offset_pos_local,
            "offset_scales": offsets_world["offset_scales"],  # 尺度不变
            "offset_quat": offset_quat,
            "offset_opacity": offsets_world["offset_opacity"],  # 不变
            "offset_sh": offsets_world["offset_sh"],  # 不变
        }
