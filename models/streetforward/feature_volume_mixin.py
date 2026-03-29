from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.math_utils import _num_sh_bases, _sh_to_rgb
from models.streetforward.node_states import NodeState, NodeStateRigid, NodeStateDistant



class FeatureVolumeMixin:
    def _build_3d_feature_volume(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
        mask_src_rigid: Optional[torch.Tensor] = None,
        idx_src_rigid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        构建 3D 特征体积，为静态背景和动态物体提取特征。
        
        这是训练流程中的核心步骤，详细流程请参考 docs/trainers/StreetForward_Flow.md。
        
        Args:
            node_state_bg: 静态背景的 NodeState（世界坐标系）
            node_state_rigid: 动态物体的 NodeStateRigid（局部坐标系），可选
            source_frame_idx: Source 帧的 frame ID（场景全局 frame_idx）
            
        Returns:
            (feat_3d_crop_bg, feat_3d_crop_rigid, rigid_visible_mask, rigid_in_crop_mask) 元组：
                - feat_3d_crop_bg: 静态背景点的3D特征，形状 [N_bg, outdim]
                - feat_3d_crop_rigid: 动态物体点的3D特征，形状 [N_rigid, outdim]
                - rigid_visible_mask: 动态物体可见性掩码，形状 [N_rigid]，可选
                - rigid_in_crop_mask: 动态物体是否在 crop_aabb 内的掩码，形状 [N_rigid]，可选
        
        处理流程：
        1. 设置 RigidNodes.cur_frame = source_frame_idx
        2. 获取静态背景点云（世界坐标）
        3. 变换动态物体到 source 帧的世界坐标
        4. 合并静态和动态点云
        5. 构建统一的 3D 特征体积（稀疏张量 → 稀疏卷积 → 密集体积）
        6. 分别为静态和动态点插值特征
        7. 删除密集体积以释放内存
        """
        rigid_visible_mask = None
        rigid_in_crop_mask = None
        if node_state_rigid is not None:
            node_state_rigid.cur_frame = source_frame_idx
            resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, source_frame_idx)
            if resolved_frame_idx is None:
                # 帧索引无法解析时，保守返回全 False 的可见性，避免索引错误
                rigid_visible_mask = torch.zeros(
                    node_state_rigid.means.shape[0],
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                visibility = node_state_rigid.instances_fv[resolved_frame_idx]
                rigid_visible_mask = visibility[node_state_rigid.point_ids[..., 0]].bool()

        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)

        means_rigid_world_all = torch.empty(0, 3, device=self.device)
        anchor_rgb_rigid_all = torch.empty(0, 3, device=self.device)
        if node_state_rigid is not None:
            # 1. 变换到 source 帧的世界坐标（使用原始局部坐标）
            means_rigid_world_all = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            anchor_rgb_rigid_all = _sh_to_rgb(node_state_rigid.sh_dc)
            
            # 2. 检查是否在 crop_aabb (bbx) 内
            rigid_in_crop_mask = torch.all(
                (means_rigid_world_all >= self.bbx_min) & (means_rigid_world_all <= self.bbx_max),
                dim=-1,
            )
        else:
            rigid_in_crop_mask = None

        # 3. 只将 source 帧可见且在 crop 内的点加入稀疏张量（使用 mask_src_rigid）
        means_list = [means_bg]
        rgb_list = [anchor_rgb_bg]
        if node_state_rigid is not None and means_rigid_world_all.numel() > 0:
            # 默认使用可见且在 crop 内的点；如果传入了 mask_src_rigid 则以其为准
            effective_mask = None
            if mask_src_rigid is not None and idx_src_rigid is not None and len(idx_src_rigid) > 0:
                effective_mask = mask_src_rigid & rigid_in_crop_mask
            elif rigid_visible_mask is not None:
                effective_mask = rigid_visible_mask
                if rigid_in_crop_mask is not None:
                    effective_mask = effective_mask & rigid_in_crop_mask
            elif rigid_in_crop_mask is not None:
                effective_mask = rigid_in_crop_mask

            if effective_mask is not None and effective_mask.any():
                means_list.append(means_rigid_world_all[effective_mask])
                rgb_list.append(anchor_rgb_rigid_all[effective_mask])

        means_all = torch.cat(means_list, dim=0)
        anchor_rgb_all = torch.cat(rgb_list, dim=0)

        # Sparse path (construct_sparse_tensor, sparse_conv) runs in FP32 when AMP enabled
        # (numpy/torchsparse may not support FP16)
        use_amp = getattr(self, "use_amp", False) and torch.cuda.is_available()
        from contextlib import nullcontext
        sparse_ctx = torch.cuda.amp.autocast(enabled=False) if use_amp else nullcontext()
        with sparse_ctx:
            sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
                raw_coords=means_all.clone(),
                feats=anchor_rgb_all,
                Bbx_max=self.bbx_max,
                Bbx_min=self.bbx_min,
                voxel_size=self.voxel_size,
                device=self.device,
            )
            feat_3d = self.sparse_conv(sparse_feat)
        
        dense_volume = self.sparse_to_dense_volume(
            sparse_tensor=feat_3d,
            coords=valid_coords,
            vol_dim=vol_dim,
        ).unsqueeze(dim=0)
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)  # [1, C, D, H, W]
        if hasattr(self, "_record_volume_stats"):
            try:
                self._record_volume_stats(vol_dim=vol_dim, feat_dim=dense_volume.shape[1])
            except Exception:
                # Stats are best-effort; do not break forward if logging is misconfigured
                pass

        grid_coords_bg = self.get_grid_coords(means_bg, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop_bg = self.interpolate_features(grid_coords_bg, dense_volume)

        if node_state_rigid is not None and means_rigid_world_all.shape[0] > 0:
            feat_dim = feat_3d_crop_bg.shape[1]
            feat_3d_crop_rigid = torch.zeros(
                means_rigid_world_all.shape[0],
                feat_dim,
                device=self.device,
            )
            if rigid_in_crop_mask is not None and rigid_in_crop_mask.any():
                means_in_crop = means_rigid_world_all[rigid_in_crop_mask]
                grid_coords_rigid_in_crop = self.get_grid_coords(
                    means_in_crop, self.bbx_min, vol_dim, self.voxel_size
                )
                feat_3d_rigid_in_crop = self.interpolate_features(
                    grid_coords_rigid_in_crop, dense_volume
                )
                feat_3d_crop_rigid[rigid_in_crop_mask] = feat_3d_rigid_in_crop
            if rigid_visible_mask is not None:
                feat_3d_crop_rigid = feat_3d_crop_rigid * rigid_visible_mask[:, None].float()
        else:
            feat_3d_crop_rigid = torch.empty(0, feat_3d_crop_bg.shape[1], device=self.device)

        del dense_volume
        
        return feat_3d_crop_bg, feat_3d_crop_rigid, rigid_visible_mask, rigid_in_crop_mask

    def _record_volume_stats(self, vol_dim: torch.Tensor, feat_dim: int) -> None:
        """
        Cache volume statistics for sentinel logging and safety checks.
        """
        vol_dim_tensor = torch.as_tensor(vol_dim).detach().cpu().long()
        self._last_vol_dim = vol_dim_tensor
        vol_prod = int(vol_dim_tensor.prod().item())
        self._last_vol_dim_prod = vol_prod
        dense_elements = vol_prod * int(feat_dim)
        self._last_dense_elements_est = dense_elements

        max_allowed = getattr(self, "sentinel_max_dense_elements", None)
        if max_allowed is None:
            max_allowed = getattr(self, "sentinel_max_vol_elements", None)
        if max_allowed is not None and dense_elements > max_allowed:
            raise RuntimeError(
                f"dense volume elements ({dense_elements}) exceed limit ({max_allowed}); "
                "consider increasing voxel_size or tightening bounding boxes."
            )

    def _compute_and_fuse_features(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        source_frame_idx: int,
        rigid_visible_mask: Optional[torch.Tensor],
        feat_bg: torch.Tensor,
        feat_rigid: torch.Tensor,
        source_views: List,
        source_images: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算 2D 特征并与 3D 特征融合，返回融合后的输入及原始 2D 特征。
        """
        feat_bg_input = feat_bg
        feat_rigid_input = feat_rigid
        feat_distant_input = None
        feat_2d_bg = None
        feat_2d_rigid = None
        feat_2d_distant = None

        if self.use_2d_features:
            feat_2d_bg, feat_2d_rigid, feat_2d_distant = self._compute_2d_features_all(
                node_state_bg=node_state_bg,
                node_state_rigid=node_state_rigid,
                node_state_distant=node_state_distant,
                source_views=source_views,
                source_images=source_images,
                source_frame_idx=source_frame_idx,
                rigid_visible_mask=rigid_visible_mask,
            )

            if feat_2d_bg is not None and feat_bg.shape[0] == feat_2d_bg.shape[0]:
                vis_bg = torch.ones(feat_bg.shape[0], device=self.device)
                feat_bg_input = self._fuse_features(feat_bg, feat_2d_bg, vis_bg)

            if (
                node_state_rigid is not None
                and feat_rigid.shape[0] > 0
                and feat_2d_rigid is not None
                and feat_2d_rigid.shape[0] == feat_rigid.shape[0]
            ):
                vis_rigid = rigid_visible_mask.float() if rigid_visible_mask is not None else torch.ones(feat_rigid.shape[0], device=self.device)
                feat_rigid_input = self._fuse_features(feat_rigid, feat_2d_rigid, vis_rigid)

            if node_state_distant is not None and feat_2d_distant is not None:
                zeros_3d = torch.zeros(feat_2d_distant.shape[0], self.feat_3d_dim, device=self.device)
                vis_distant = torch.ones(feat_2d_distant.shape[0], device=self.device)
                feat_distant_input = self._fuse_features(zeros_3d, feat_2d_distant, vis_distant)

        return (
            feat_bg_input,
            feat_rigid_input,
            feat_distant_input,
            feat_2d_bg,
            feat_2d_rigid,
            feat_2d_distant,
        )

    def _prepare_gaussians_for_source(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """
        合并静态与动态高斯参数（动态先变换到 source 帧），用于 2D 特征反投影。
        返回合并后的高斯字典以及静态/动态数量。
        """
        num_sh = _num_sh_bases(self.sh_degree)
        means_bg = node_state_bg.means
        quats_bg = node_state_bg.quats
        scales_bg = torch.exp(node_state_bg.scales_log)
        opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
        colors_bg = torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)

        means_rigid_world = torch.empty(0, 3, device=self.device)
        quats_rigid_world = torch.empty(0, 4, device=self.device)
        scales_rigid = torch.empty(0, 3, device=self.device)
        opacities_rigid = torch.empty(0, device=self.device)
        colors_rigid = torch.zeros(0, num_sh, 3, device=self.device)
        if node_state_rigid is not None and node_state_rigid.means.numel() > 0:
            node_state_rigid.cur_frame = source_frame_idx
            means_rigid_world = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, node_state_rigid.quats)
            scales_rigid = torch.exp(node_state_rigid.scales_log)
            opacities_rigid = torch.sigmoid(node_state_rigid.opacity_logit).squeeze(-1)
            colors_rigid = torch.cat([node_state_rigid.sh_dc[:, None, :], node_state_rigid.sh_rest], dim=1)

        gaussians = {
            "means": torch.cat([means_bg, means_rigid_world], dim=0),
            "quats": torch.cat([quats_bg, quats_rigid_world], dim=0),
            "scales": torch.cat([scales_bg, scales_rigid], dim=0),
            "opacities": torch.cat([opacities_bg, opacities_rigid], dim=0),
            "colors": torch.cat([colors_bg, colors_rigid], dim=0),
        }
        return gaussians, means_bg.shape[0], means_rigid_world.shape[0]

    def _prepare_all_gaussians(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        source_frame_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], int, int, int]:
        """
        合并三类点（前景、动态、背景远景）用于 2D 特征计算。
        """
        num_sh = _num_sh_bases(self.sh_degree)

        means_bg = node_state_bg.means
        quats_bg = node_state_bg.quats
        scales_bg = torch.exp(node_state_bg.scales_log)
        opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
        colors_bg = torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)
        num_bg = means_bg.shape[0]

        means_rigid_world = torch.empty(0, 3, device=self.device)
        quats_rigid_world = torch.empty(0, 4, device=self.device)
        scales_rigid = torch.empty(0, 3, device=self.device)
        opacities_rigid = torch.empty(0, device=self.device)
        colors_rigid = torch.zeros(0, num_sh, 3, device=self.device)
        num_rigid = 0
        if node_state_rigid is not None and node_state_rigid.means.numel() > 0:
            node_state_rigid.cur_frame = source_frame_idx
            # 解析帧索引以确保使用正确的帧位姿
            resolved_frame_idx = self._resolve_rigid_frame_idx(node_state_rigid, source_frame_idx)
            means_rigid_world = self._transform_rigid_to_world(node_state_rigid, node_state_rigid.means)
            quats_rigid_world = self._transform_rigid_quats_to_world(node_state_rigid, node_state_rigid.quats)
            scales_rigid = torch.exp(node_state_rigid.scales_log)
            opacities_rigid = torch.sigmoid(node_state_rigid.opacity_logit).squeeze(-1)
            colors_rigid = torch.cat([node_state_rigid.sh_dc[:, None, :], node_state_rigid.sh_rest], dim=1)
            num_rigid = means_rigid_world.shape[0]

        means_distant = torch.empty(0, 3, device=self.device)
        quats_distant = torch.empty(0, 4, device=self.device)
        scales_distant = torch.empty(0, 3, device=self.device)
        opacities_distant = torch.empty(0, device=self.device)
        colors_distant = torch.zeros(0, num_sh, 3, device=self.device)
        num_distant = 0
        if node_state_distant is not None and node_state_distant.means.numel() > 0:
            means_distant = node_state_distant.means
            quats_distant = node_state_distant.quats
            scales_distant = torch.exp(node_state_distant.scales_log)
            opacities_distant = torch.sigmoid(node_state_distant.opacity_logit).squeeze(-1)
            colors_distant = torch.cat([node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1)
            num_distant = means_distant.shape[0]

        gaussians = {
            "means": torch.cat([means_bg, means_rigid_world, means_distant], dim=0),
            "quats": torch.cat([quats_bg, quats_rigid_world, quats_distant], dim=0),
            "scales": torch.cat([scales_bg, scales_rigid, scales_distant], dim=0),
            "opacities": torch.cat([opacities_bg, opacities_rigid, opacities_distant], dim=0),
            "colors": torch.cat([colors_bg, colors_rigid, colors_distant], dim=0),
        }
        return gaussians, num_bg, num_rigid, num_distant

    def _compute_2d_features(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        source_views: List,
        source_images: List[torch.Tensor],
        source_frame_idx: int,
        rigid_visible_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        双轮渲染：先渲染 RGB 供 CNN 使用，再流式渲染提取权重并反投影。
        """
        if (
            not self.use_2d_features
            or self.image_feature_extractor is None
            or self.alpha_t_extractor is None
            or self.feature_backprojector is None
        ):
            return None, None
        if source_images is None or len(source_images) == 0 or source_views is None or len(source_views) == 0:
            return None, None

        imgs = [img.to(self.device) for img in source_images if img is not None]
        if len(imgs) == 0:
            return None, None
        sample_img = imgs[0]
        if sample_img.dim() == 3 and sample_img.shape[-1] == 3:
            height, width = sample_img.shape[0], sample_img.shape[1]
        elif sample_img.dim() == 3 and sample_img.shape[0] == 3:
            height, width = sample_img.shape[1], sample_img.shape[2]
        else:
            height, width = sample_img.shape[-2], sample_img.shape[-1]
        image_batch = torch.stack(imgs, dim=0)

        # Step 1: Prepare Gaussians
        gaussians, num_bg, num_rigid = self._prepare_gaussians_for_source(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            source_frame_idx=source_frame_idx,
        )
        
        # Phase 1: Render RGB only (meta discarded immediately)
        # Important: rendered_batch should not have gradients to avoid:
        # 1. Memory/computation graph explosion
        # 2. Unwanted gradient paths from 2D CNN back to renderer/Gaussian state
        # The 2D CNN is only a conditioning feature extractor, should not backprop to rendering
        with torch.no_grad():
            rendered_rgbs = self.alpha_t_extractor.render_rgb_only(
                gaussians, source_views, height, width
            )

        # Convert images to [V, H, W, 3] format if needed
        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)

        rendered_batch = torch.stack(rendered_rgbs, dim=0)  # [V, H, W, 3]
        del rendered_rgbs
        
        # Ensure rendered_batch is detached (defensive check)
        rendered_batch = rendered_batch.detach()
        assert not rendered_batch.requires_grad, \
            "rendered_batch should not require gradients - this would cause gradient graph explosion"

        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        multi_channel_input = torch.cat([image_batch, rendered_batch], dim=-1)  # [V, H, W, 6]
        del rendered_batch, image_batch

        # Phase 2: CNN forward then streaming backprojection
        features_2d = self.image_feature_extractor(multi_channel_input)  # [V, H_feat, W_feat, C]
        del multi_channel_input

        # Important: reuse the same gaussians for both passes to keep RGB/weights aligned.
        feat_2d_all = self.alpha_t_extractor.render_and_backproject_streaming(
            gaussians=gaussians,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=num_bg + num_rigid,
            backprojector=self.feature_backprojector,
        )

        feat_2d_bg = feat_2d_all[:num_bg]
        feat_2d_rigid = feat_2d_all[num_bg:]
        if rigid_visible_mask is not None and feat_2d_rigid.shape[0] == rigid_visible_mask.shape[0]:
            feat_2d_rigid = feat_2d_rigid * rigid_visible_mask.float().unsqueeze(-1)
        return feat_2d_bg, feat_2d_rigid

    def _compute_2d_features_all(
        self,
        node_state_bg: NodeState,
        node_state_rigid: Optional[NodeStateRigid],
        node_state_distant: Optional[NodeStateDistant],
        source_views: List,
        source_images: List[torch.Tensor],
        source_frame_idx: int,
        rigid_visible_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        计算所有点（前景+动态+背景远景）的 2D 特征。
        """
        if (
            not self.use_2d_features
            or self.image_feature_extractor is None
            or self.alpha_t_extractor is None
            or self.feature_backprojector is None
        ):
            return None, None, None
        if source_images is None or len(source_images) == 0 or source_views is None or len(source_views) == 0:
            return None, None, None

        imgs = [img.to(self.device) for img in source_images if img is not None]
        if len(imgs) == 0:
            return None, None, None
        sample_img = imgs[0]
        if sample_img.dim() == 3 and sample_img.shape[-1] == 3:
            height, width = sample_img.shape[0], sample_img.shape[1]
        elif sample_img.dim() == 3 and sample_img.shape[0] == 3:
            height, width = sample_img.shape[1], sample_img.shape[2]
        else:
            height, width = sample_img.shape[-2], sample_img.shape[-1]
        image_batch = torch.stack(imgs, dim=0)

        gaussians_all, num_bg, num_rigid, num_distant = self._prepare_all_gaussians(
            node_state_bg=node_state_bg,
            node_state_rigid=node_state_rigid,
            node_state_distant=node_state_distant,
            source_frame_idx=source_frame_idx,
        )
        total_points = num_bg + num_rigid + num_distant
        if total_points == 0:
            return None, None, None

        # Phase 1: Render RGB only (meta discarded immediately)
        # Important: rendered_batch should not have gradients to avoid:
        # 1. Memory/computation graph explosion
        # 2. Unwanted gradient paths from 2D CNN back to renderer/Gaussian state
        # The 2D CNN is only a conditioning feature extractor, should not backprop to rendering
        with torch.no_grad():
            rendered_rgbs = self.alpha_t_extractor.render_rgb_only(
                gaussians_all, source_views, height, width
            )

        if image_batch.dim() == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)

        rendered_batch = torch.stack(rendered_rgbs, dim=0)
        del rendered_rgbs
        
        # Ensure rendered_batch is detached (defensive check)
        rendered_batch = rendered_batch.detach()
        assert not rendered_batch.requires_grad, \
            "rendered_batch should not require gradients - this would cause gradient graph explosion"

        if rendered_batch.shape[1:3] != image_batch.shape[1:3]:
            rendered_batch = F.interpolate(
                rendered_batch.permute(0, 3, 1, 2),
                size=(image_batch.shape[1], image_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        multi_channel_input = torch.cat([image_batch, rendered_batch], dim=-1)
        del rendered_batch, image_batch

        features_2d = self.image_feature_extractor(multi_channel_input)
        del multi_channel_input

        feat_2d_all = self.alpha_t_extractor.render_and_backproject_streaming(
            gaussians=gaussians_all,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=total_points,
            backprojector=self.feature_backprojector,
        )

        feat_2d_bg = feat_2d_all[:num_bg] if num_bg > 0 else None
        feat_2d_rigid = feat_2d_all[num_bg:num_bg + num_rigid] if num_rigid > 0 else None
        feat_2d_distant = feat_2d_all[num_bg + num_rigid:] if num_distant > 0 else None

        if feat_2d_rigid is not None and rigid_visible_mask is not None and feat_2d_rigid.shape[0] == rigid_visible_mask.shape[0]:
            feat_2d_rigid = feat_2d_rigid * rigid_visible_mask.float().unsqueeze(-1)

        return feat_2d_bg, feat_2d_rigid, feat_2d_distant

    def _fuse_features(
        self,
        feat_3d: torch.Tensor,
        feat_2d: Optional[torch.Tensor],
        visibility: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        融合 2D/3D 特征与可见性标量。
        """
        if not self.use_2d_features or feat_2d is None or self.feature_fusion is None:
            return feat_3d
        if visibility is None:
            visibility = torch.ones(feat_3d.shape[0], device=feat_3d.device)
        fused = self.feature_fusion.fuse(feat_3d, feat_2d, visibility)
        if hasattr(self, "feat_fused_rms") and self.feat_fused_rms is not None:
            fused = self.feat_fused_rms(fused)
        return fused

    def get_grid_coords(
        self, position_w: torch.Tensor, bbx_min: torch.Tensor, vol_dim, voxel_size: float
    ) -> torch.Tensor:
        """
        将世界坐标转换为体积网格的归一化坐标（用于 grid_sample）。
        
        Args:
            position_w: 世界坐标位置，形状 [N, 3]
            bbx_min: 边界框最小值，形状 [3]
            vol_dim: 体积维度，[D, H, W] 格式（可以是 list、tuple 或 Tensor）
            voxel_size: 体素大小（米）
            
        Returns:
            归一化网格坐标，形状 [N, 3]，格式 [x_norm, y_norm, z_norm]，值域 [-1, 1]
        
        处理流程：
        1. 将坐标相对于边界框原点：pts = position_w - bbx_min
        2. 转换为体素索引（浮点数）：index = pts / voxel_size
        3. 归一化到 [-1, 1] 范围：norm = 2.0 * (index / (vol_dim - 1)) - 1.0
        4. 堆叠为 [x_norm, y_norm, z_norm] 格式（grid_sample 要求）
        
        注意：
        - grid_sample (5D) 期望坐标顺序为 [x, y, z]，对应 [W, H, D] 维度
        - 由于 dense_volume 是 [1, C, Z, Y, X] = [B, C, D, H, W]，其中 D=Z, H=Y, W=X
        - 因此 grid 坐标必须是 [x_norm, y_norm, z_norm] 对应 [W, H, D]
        - 使用 align_corners=True，所以 index 0 映射到 -1.0，index (N-1) 映射到 1.0
        """
        # Clamp positions to bbox range to match construct_sparse_tensor behavior
        # This ensures coordinates are within the volume bounds
        # Use self.bbx_max directly instead of recalculating from vol_dim
        bbx_max = self.bbx_max.to(position_w.device)
        position_w_clamped = torch.clamp(position_w, min=bbx_min, max=bbx_max)
        
        pts = position_w_clamped - bbx_min.to(position_w.device)
        x_index = pts[..., 0] / voxel_size
        y_index = pts[..., 1] / voxel_size
        z_index = pts[..., 2] / voxel_size
        
        # Convert vol_dim to torch.Tensor if it's a list, tuple, or numpy array
        # construct_sparse_tensor may return Python list from nerfstudio implementation
        if isinstance(vol_dim, (list, tuple)):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        elif not isinstance(vol_dim, torch.Tensor):
            vol_dim = torch.tensor(vol_dim, device=position_w.device, dtype=torch.float32)
        else:
            vol_dim = vol_dim.to(position_w.device).float()
        
        # vol_dim is [X, Y, Z] from construct_sparse_tensor (world coordinates)
        # sparse_to_dense_volume creates dense volume as [X, Y, Z, C]
        # After unsqueeze: [1, X, Y, Z, C]
        # After permute(0, 4, 3, 2, 1): [1, C, Z, Y, X] = [1, C, D, H, W]
        # where D=Z, H=Y, W=X
        # 
        # PyTorch grid_sample (5D) for input [B, C, D, H, W] expects grid coordinates
        # in the format [x, y, z] corresponding to [W, H, D] dimensions.
        # This is the standard convention: the last dimension of grid is [x, y, z]
        # which maps to [width, height, depth] of the input volume.
        #
        # Therefore, we must return [x_norm, y_norm, z_norm] to match [W, H, D].
        # For align_corners=True: index 0 maps to -1.0, index (N-1) maps to 1.0
        # Therefore, we use (vol_dim - 1) as denominator to ensure correct boundary mapping
        den_x = torch.clamp(vol_dim[0] - 1.0, min=1.0)
        den_y = torch.clamp(vol_dim[1] - 1.0, min=1.0)
        den_z = torch.clamp(vol_dim[2] - 1.0, min=1.0)
        x_norm = 2.0 * (x_index / den_x) - 1.0  # X -> W
        y_norm = 2.0 * (y_index / den_y) - 1.0  # Y -> H
        z_norm = 2.0 * (z_index / den_z) - 1.0  # Z -> D
        # grid_sample (5D) expects coordinates in [x, y, z] order for [B, C, D, H, W] input
        # This corresponds to [W, H, D] = [X, Y, Z]
        grid_coords = torch.stack([x_norm, y_norm, z_norm], dim=-1)
        
        return grid_coords

    def interpolate_features(self, grid_coords: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
        """
        从 3D 特征体积中插值提取每个点的特征。
        
        Args:
            grid_coords: 归一化网格坐标，形状 [N, 3]，格式 [x_norm, y_norm, z_norm]
            feature_volume: 特征体积，形状 [1, C, D, H, W]（经过 permute 后，其中 D=Z, H=Y, W=X）
            
        Returns:
            每个点的特征，形状 [N, C]
        
        使用三线性插值（grid_sample 在 3D 中）从体积中提取特征。
        grid_sample 期望输入格式为 [B, C, D, H, W]，坐标格式为 [B, D_out, H_out, W_out, 3]。
        我们扩展 grid_coords 为 [1, 1, 1, N, 3] 以匹配要求。
        """
        grid_coords_expanded = grid_coords[None, None, None, ...]
        feature = torch.nn.functional.grid_sample(
            feature_volume,
            grid_coords_expanded,
            mode="bilinear",  # 在3D中实际是三线性插值
            align_corners=True,
            padding_mode="zeros",
        )
        return feature[0, :, 0, 0, :].T  # [1, C, 1, 1, N] → [C, N] → [N, C]
