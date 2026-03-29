from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.math_utils import (
    _axis_angle_to_quat,
    _num_sh_bases,
    _quat_to_rotmat,
    _normalize_quat,
    _quat_multiply,
    _quat_conjugate,
)
from models.streetforward.node_states import NodeStateRigid, NodeState


def _quat_to_rot6d(quats: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (wxyz) to 6D rotation representation (first two columns of R)."""
    rot = _quat_to_rotmat(_normalize_quat(quats))
    rot6d = rot[..., :3, :2].reshape(quats.shape[:-1] + (6,))
    return rot6d


class OffsetsMixin:
    def _apply_gru_head_rms(
        self,
        head_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "gru_head_rms") or self.gru_head_rms is None:
            return head_input
        normed = self.gru_head_rms(head_input)
        if mask is None:
            return normed
        m = mask.to(device=head_input.device, dtype=head_input.dtype).unsqueeze(-1)
        return m * normed + (1.0 - m) * head_input

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
        
        return {
            "offset_pos": offset_pos,
            "offset_scales": offset_scales,
            "offset_quat": offset_quat,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
        }

    # --- GRU-style offset prediction helpers ---
    def _normalize_params_for_embed(
        self,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Normalize and compress NodeState params into a fixed-length vector for embedding.

        Returns:
            param_vec: [N, 17] tensor = means(3) + rot6d(6) + scales_norm(3) +
                       opacity_norm(1) + sh_dc(3) + sh_rest_energy(1)
        """
        means = params["means"]  # assumed world coords for bg/distant; world-transformed for rigid
        scales_log = params["scales_log"]
        quats = params["quats"]
        opacity_logit = params["opacity_logit"]
        sh_dc = params["sh_dc"]
        sh_rest = params["sh_rest"]

        # means -> [-1,1] using bbox
        bbx_min = self.bbx_min.to(means.device)
        bbx_max = self.bbx_max.to(means.device)
        denom = (bbx_max - bbx_min).clamp(min=1e-6)
        means_norm = (means - bbx_min) / denom * 2.0 - 1.0

        # scales log clamp + layer norm
        scales_clamped = scales_log.clamp(-10.0, 10.0)
        scales_norm = F.layer_norm(scales_clamped, scales_clamped.shape[1:])

        # rotation 6d
        rot6d = _quat_to_rot6d(quats)

        # opacity logit normalize (tanh keeps within [-1,1])
        opacity_norm = torch.tanh(opacity_logit)

        # sh_dc keep raw; sh_rest energy scalar
        sh_rest_energy = torch.linalg.norm(sh_rest.reshape(sh_rest.shape[0], -1), dim=-1, keepdim=True)

        param_vec = torch.cat(
            [
                means_norm,
                rot6d,
                scales_norm,
                opacity_norm,
                sh_dc,
                sh_rest_energy,
            ],
            dim=-1,
        )
        return param_vec

    def _build_params_for_embed(
        self,
        node_state: NodeState,
        coord_space: str = "world",
        frame_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Gather parameters for embedding. For rigid, coord_space='world' transforms to world frame.
        """
        params = {
            "means": node_state.means,
            "scales_log": node_state.scales_log,
            "quats": node_state.quats,
            "opacity_logit": node_state.opacity_logit,
            "sh_dc": node_state.sh_dc,
            "sh_rest": node_state.sh_rest,
        }

        if isinstance(node_state, NodeStateRigid) and coord_space == "world":
            assert frame_idx is not None, "frame_idx is required for rigid world transform"
            params["means"] = self._transform_rigid_to_world(node_state, node_state.means, frame_idx=frame_idx)
            params["quats"] = self._transform_rigid_quats_to_world(node_state, node_state.quats, frame_idx=frame_idx)

        return params

    def _predict_offsets_gru(
        self,
        feat: torch.Tensor,
        params_for_embed: Dict[str, torch.Tensor],
        h_old: torch.Tensor,
        mask_update_rigid: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        GRU-style fusion of features + params to predict offsets and updated hidden state.

        Args:
            feat: [N, C_fused]
            params_for_embed: dict with means/quats/scales_log/opacity_logit/sh_dc/sh_rest
            h_old: [N, H]
            mask_update_rigid: optional [N] bool for rigid gating
        """
        if feat is None or feat.numel() == 0:
            # Empty case: no features → return "no-change" offsets.
            # offset_quat must be identity [1,0,0,0] so quat_multiply(q, identity)=q; zeros would be invalid.
            num_points = params_for_embed["means"].shape[0]
            device = self.device if hasattr(self, "device") else params_for_embed["means"].device
            dtype = params_for_embed["means"].dtype
            identity_quat = torch.zeros(num_points, 4, device=device, dtype=dtype)
            identity_quat[..., 0] = 1.0
            zero_offsets = {
                "offset_pos": torch.zeros_like(params_for_embed["means"]),
                "offset_scales": torch.zeros_like(params_for_embed["scales_log"]),
                "offset_quat": identity_quat.clone(),  # identity = no rotation change
                "offset_opacity": torch.zeros_like(params_for_embed["opacity_logit"]),
                "offset_sh": torch.zeros(
                    num_points,
                    self.gaussion_decoder[-1].out_features if hasattr(self.gaussion_decoder[-1], "out_features") else _num_sh_bases(self.sh_degree) * 3,
                    device=device,
                    dtype=dtype,
                ),
            }
            h_new = h_old
            if mask_update_rigid is not None:
                gate = mask_update_rigid.unsqueeze(-1).float().detach()
                identity_quat = torch.zeros(num_points, 4, device=device, dtype=dtype)
                identity_quat[..., 0] = 1.0
                for k in zero_offsets:
                    if k == "offset_quat":
                        zero_offsets[k] = torch.where(
                            gate.expand_as(zero_offsets[k]).bool(), zero_offsets[k], identity_quat
                        )
                    else:
                        zero_offsets[k] = zero_offsets[k] * gate
                h_new = h_old * (1 - gate) + h_new * gate
            return zero_offsets, h_new

        param_vec = self._normalize_params_for_embed(params_for_embed)
        param_embed = self.mlp_params_embed(param_vec)
        param_embed = self.param_embed_norm(param_embed)

        x = torch.cat([feat, param_embed], dim=-1)
        hx = torch.cat([h_old, x], dim=-1)

        z = torch.sigmoid(self.gru_update(hx))
        if self.gru_reset is not None:
            r = torch.sigmoid(self.gru_reset(hx))
            h_cand = torch.tanh(self.gru_candidate(torch.cat([r * h_old, x], dim=-1)))
        else:
            h_cand = torch.tanh(self.gru_candidate(hx))
        h_new = (1.0 - z) * h_old + z * h_cand

        # Project to offset head input dim if needed
        head_input = self.gru_to_head(h_new)
        head_input = self._apply_gru_head_rms(head_input, mask_update_rigid)
        offsets = self._predict_offsets(head_input)

        if mask_update_rigid is not None:
            gate = mask_update_rigid.to(offsets["offset_pos"].dtype).unsqueeze(-1).detach()
            identity_quat = torch.zeros_like(offsets["offset_quat"])
            identity_quat[..., 0] = 1.0
            for k in offsets:
                if k == "offset_quat":
                    offsets[k] = torch.where(
                        gate.expand_as(offsets[k]).bool(), offsets[k], identity_quat
                    )
                else:
                    offsets[k] = offsets[k] * gate
            h_new = h_old * (1 - gate) + h_new * gate

        return offsets, h_new

    def _render_params_from_offsets(
        self,
        node_state: NodeState,
        offsets: Dict[str, torch.Tensor],
        node_type: Literal["bg", "rigid", "distant"] = "bg",
    ) -> Dict[str, torch.Tensor]:
        """
        从 NodeState 和偏移量计算渲染参数。
        
        Args:
            node_state: NodeState（Background 或 RigidNodes），所有参数都是分离的
            offsets: 偏移量字典（可微）
            node_type: 节点类型，控制分节点 eta（默认 "bg"）
            
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

        eta_means = self._get_eta_for_node(node_type, "means")
        eta_scales = self._get_eta_for_node(node_type, "scales")
        eta_opacity = self._get_eta_for_node(node_type, "opacity")
        eta_sh_dc = self._get_eta_for_node(node_type, "sh_dc")
        eta_sh_rest = self._get_eta_for_node(node_type, "sh_rest")

        # Apply offsets with step size factors (eta)
        # Note: means_r is not clamped here to preserve gradient flow
        means_r = node_state.means + eta_means * offsets["offset_pos"]
        scales_log_r = node_state.scales_log + eta_scales * offsets["offset_scales"]
        quats_r = _normalize_quat(_quat_multiply(node_state.quats, offsets["offset_quat"]))
        opacity_logit_r = node_state.opacity_logit + eta_opacity * offsets["offset_opacity"]
        sh_dc_r = node_state.sh_dc + eta_sh_dc * offsets["offset_sh"][:, :3]
        sh_rest_r = node_state.sh_rest + eta_sh_rest * sh_rest_offset

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

    def _predict_and_gate_offsets(
        self,
        feat_bg_input: torch.Tensor,
        feat_rigid_input: torch.Tensor,
        feat_distant_input: Optional[torch.Tensor],
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeState],
        mask_update_rigid: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        预测偏移量并对动态物体应用 gate。
        """
        offsets_bg = self._predict_offsets(feat_bg_input)

        offsets_rigid_world = None
        if node_state_rigid is not None and feat_rigid_input is not None and feat_rigid_input.numel() > 0:
            offsets_rigid_world = self._predict_offsets(feat_rigid_input)
            if mask_update_rigid is not None:
                gate = mask_update_rigid.to(offsets_rigid_world["offset_pos"].dtype).unsqueeze(-1).detach()
                offsets_rigid_world["offset_pos"] = offsets_rigid_world["offset_pos"] * gate
                offsets_rigid_world["offset_scales"] = offsets_rigid_world["offset_scales"] * gate
                offsets_rigid_world["offset_quat"] = offsets_rigid_world["offset_quat"] * gate
                offsets_rigid_world["offset_opacity"] = offsets_rigid_world["offset_opacity"] * gate
                offsets_rigid_world["offset_sh"] = offsets_rigid_world["offset_sh"] * gate
            else:
                raise ValueError("mask_update_rigid is not provided")

        offsets_distant = None
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            offsets_distant = self._predict_offsets(feat_distant_input)

        return offsets_bg, offsets_rigid_world, offsets_distant

    def _get_eta_for_node(self, node_type: str, key: str) -> float:
        """
        Fetch per-node eta with fallback to global defaults.
        """
        attr_name = f"eta_{key}"
        default = getattr(self, attr_name, 1.0)
        by_node = getattr(self, "eta_by_node", None) or {}
        node_cfg = by_node.get(node_type, {}) if isinstance(by_node, dict) else {}
        value = None
        if hasattr(node_cfg, "get"):
            value = node_cfg.get(attr_name, None)
        else:
            try:
                value = node_cfg[attr_name]
            except Exception:
                value = None
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _compute_render_params_for_inner_iter(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeState],
        offsets_bg: Dict[str, torch.Tensor],
        offsets_rigid_world: Optional[Dict[str, torch.Tensor]],
        offsets_distant: Optional[Dict[str, torch.Tensor]],
        source_frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        训练内一次迭代的渲染参数计算，处理 rigid 世界→局部变换。
        """
        render_params_bg = self._render_params_from_offsets(node_state_bg, offsets_bg, node_type="bg")

        render_params_rigid = None
        if node_state_rigid is not None and offsets_rigid_world is not None:
            offsets_rigid_local = self._transform_offsets_world_to_local(
                node_state_rigid, offsets_rigid_world, source_frame_idx
            )
            render_params_rigid = self._render_params_from_offsets(
                node_state_rigid, offsets_rigid_local, node_type="rigid"
            )

        render_params_distant = None
        if node_state_distant is not None and offsets_distant is not None:
            render_params_distant = self._render_params_from_offsets(
                node_state_distant, offsets_distant, node_type="distant"
            )

        return render_params_bg, render_params_rigid, render_params_distant
