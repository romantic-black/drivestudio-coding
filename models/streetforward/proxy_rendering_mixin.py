from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.logging_utils import _debug_log
from models.streetforward.math_utils import _num_sh_bases, _sh_to_rgb, get_viewmat

if TYPE_CHECKING:
    from models.streetforward.node_states import NodeState



class ProxyRenderingMixin:
    def _create_proxy_params(self, render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        创建代理参数，用于多视角梯度累积。
        
        Args:
            render_params: 渲染参数字典（可微）
            
        Returns:
            代理参数字典，包含：
                - "means_p": 代理位置（分离但可微）
                - "scales_p": 代理尺度（分离但可微）
                - "quats_p": 代理四元数（分离但可微）
                - "opacities_p": 代理不透明度（分离但可微）
                - "colors_p": 代理颜色（分离但可微）
        
        操作：proxy = render_param.detach().requires_grad_(True)
        
        关键点：
        - 代理参数从渲染参数中分离（detach），但重新启用梯度
        - 这样可以在多个视角上累积梯度，然后一次性反向传播到渲染参数
        - 所有 target 帧共享同一组代理参数
        """
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Creating proxy params",
            {
                "num_points": render_params["means_r"].shape[0],
                "requires_grad_before": render_params["means_r"].requires_grad,
            },
            hypothesis_id="H1",
        )
        # #endregion
        proxies = {
            "means_p": render_params["means_r"].detach().requires_grad_(True),
            "scales_p": render_params["scales_r"].detach().requires_grad_(True),
            "quats_p": render_params["quats_r"].detach().requires_grad_(True),
            "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
            "colors_p": render_params["colors_r"].detach().requires_grad_(True),
        }
        # #region agent log
        _debug_log(
            "streetforward.py:_create_proxy_params",
            "Proxy params created",
            {
                "requires_grad_after": proxies["means_p"].requires_grad,
                "is_leaf": proxies["means_p"].is_leaf,
                "grad_fn": str(proxies["means_p"].grad_fn),
            },
            hypothesis_id="H1",
        )
        # #endregion
        return proxies

    def _merge_all_params(
        self,
        proxies_bg: Dict[str, torch.Tensor],
        proxies_rigid: Optional[Dict[str, torch.Tensor]],
        proxies_distant: Optional[Dict[str, torch.Tensor]],
        means_rigid_world: torch.Tensor,
        quats_rigid_world: torch.Tensor,
        opacities_rigid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        合并前景、动态和背景远景的渲染参数。
        """
        means_list = [proxies_bg["means_p"]]
        quats_list = [proxies_bg["quats_p"]]
        scales_list = [proxies_bg["scales_p"]]
        opacities_list = [proxies_bg["opacities_p"]]
        colors_list = [proxies_bg["colors_p"]]

        if proxies_rigid is not None and means_rigid_world.numel() > 0:
            means_list.append(means_rigid_world)
            quats_list.append(quats_rigid_world)
            scales_list.append(proxies_rigid["scales_p"])
            opacities_list.append(opacities_rigid if opacities_rigid is not None else proxies_rigid["opacities_p"])
            colors_list.append(proxies_rigid["colors_p"])

        if proxies_distant is not None:
            means_list.append(proxies_distant["means_p"])
            quats_list.append(proxies_distant["quats_p"])
            scales_list.append(proxies_distant["scales_p"])
            opacities_list.append(proxies_distant["opacities_p"])
            colors_list.append(proxies_distant["colors_p"])

        return (
            torch.cat(means_list, dim=0),
            torch.cat(quats_list, dim=0),
            torch.cat(scales_list, dim=0),
            torch.cat(opacities_list, dim=0),
            torch.cat(colors_list, dim=0),
        )

    def compute_loss(self, pred_rgb: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
        """
        计算 L2 损失（均方误差）。
        
        Args:
            pred_rgb: 预测的RGB图像，形状 [H, W, 3]
            gt_image: 真实图像，形状 [H, W, 3]
            
        Returns:
            标量损失值：mean((pred_rgb - gt_image)²)
        """
        return torch.mean((pred_rgb - gt_image) ** 2)

    def _compute_render_params(self, node_state: NodeState) -> Dict[str, torch.Tensor]:
        """
        共享的前向传播：从节点状态计算渲染参数。
        
        Args:
            node_state: NodeState（Background 或 RigidNodes）
            
        Returns:
            渲染参数字典
        
        这是评估时使用的简化流程：
        1. 构建 3D 特征体积（只使用单个 NodeState）
        2. 预测偏移量
        3. 计算渲染参数
        
        注意：此方法不处理动态物体的坐标变换，适用于静态背景或已变换到目标帧的动态物体。
        """
        means_s = node_state.means
        anchor_rgb = _sh_to_rgb(node_state.sh_dc)

        sparse_feat, vol_dim, valid_coords = self.construct_sparse_tensor(
            raw_coords=means_s.clone(),
            feats=anchor_rgb,
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
        dense_volume = dense_volume.permute(0, 4, 3, 2, 1)
        grid_coords = self.get_grid_coords(means_s, self.bbx_min, vol_dim, self.voxel_size)
        feat_3d_crop = self.interpolate_features(grid_coords, dense_volume)
        del dense_volume

        offsets = self._predict_offsets(feat_3d_crop)
        render_params = self._render_params_from_offsets(node_state, offsets)
        return render_params

    def _render_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染单个视角，返回 RGB 图像和 alpha 通道。
        
        Args:
            render_params: 渲染参数字典，可以是代理参数（"means_p"等）或渲染参数（"means_r"等）
            view: 相机视角对象，需有 camtoworlds 和 Ks/K 属性
            height: 图像高度
            width: 图像宽度
            
        Returns:
            (rgb, acc) 元组：
                - rgb: RGB 图像，形状 [H, W, 3]
                - acc: 累积不透明度，形状 [H, W]
        
        使用 gsplat 渲染器进行高斯点渲染。
        """
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        viewmat = get_viewmat(c2w)
        k_mat = None
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=self.device).unsqueeze(0)
        
        # Ensure Ks is [1,3,3] format
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)

        means_key = "means_p" if "means_p" in render_params else "means_r"
        scales_key = "scales_p" if "scales_p" in render_params else "scales_r"
        quats_key = "quats_p" if "quats_p" in render_params else "quats_r"
        opacities_key = "opacities_p" if "opacities_p" in render_params else "opacities_r"
        colors_key = "colors_p" if "colors_p" in render_params else "colors_r"

        render, alpha, _ = self.renderer(
            means=render_params[means_key],
            quats=render_params[quats_key],
            scales=render_params[scales_key],
            opacities=render_params[opacities_key],
            colors=render_params[colors_key],
            viewmats=viewmat,
            Ks=k_mat,
            width=width,
            height=height,
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=self.sh_degree,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
        )

        rgb = render[:, ..., :3].squeeze(0)
        acc = alpha.squeeze(0)
        return rgb, acc
