from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.streetforward.math_utils import _num_sh_bases, _sh_to_rgb, get_viewmat
from models.streetforward.node_state_mixin import RigidMasks

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
        proxies = {
            "means_p": render_params["means_r"].detach().requires_grad_(True),
            "scales_p": render_params["scales_r"].detach().requires_grad_(True),
            "quats_p": render_params["quats_r"].detach().requires_grad_(True),
            "opacities_p": render_params["opacities_r"].detach().requires_grad_(True),
            "colors_p": render_params["colors_r"].detach().requires_grad_(True),
        }
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

    def _merge_params_with_rigid_subset(
        self,
        proxies_bg: Dict[str, torch.Tensor],
        proxies_distant: Optional[Dict[str, torch.Tensor]],
        rigid_subset: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        合并参数，支持只渲染可见的 rigid 子集。
        """
        means_list = [proxies_bg["means_p"]]
        quats_list = [proxies_bg["quats_p"]]
        scales_list = [proxies_bg["scales_p"]]
        opacities_list = [proxies_bg["opacities_p"]]
        colors_list = [proxies_bg["colors_p"]]

        if rigid_subset["means"].numel() > 0:
            means_list.append(rigid_subset["means"])
            quats_list.append(rigid_subset["quats"])
            scales_list.append(rigid_subset["scales"])
            opacities_list.append(rigid_subset["opacities"])
            colors_list.append(rigid_subset["colors"])

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

    def compute_loss(
        self,
        pred_rgb: torch.Tensor,
        gt_image: torch.Tensor,
        sky_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 L1 损失，可选天空区域遮挡。
        
        Args:
            pred_rgb: 预测的RGB图像，形状 [H, W, 3]
            gt_image: 真实图像，形状 [H, W, 3]
            sky_mask: 天空掩码，形状 [H, W] 或 [H, W, 1]，1 表示有效区域，0 表示天空（可选）
            
        Returns:
            标量损失值：mean(|pred_rgb - gt_image|)（若提供 sky_mask，则仅在有效区域计算）
        """
        diff = torch.abs(pred_rgb - gt_image)

        if sky_mask is not None:
            mask_2d = sky_mask
            if mask_2d.dim() == 3:
                mask_2d = mask_2d.squeeze(-1)
            mask_2d = mask_2d.to(diff.device).float()
            valid_pixels = mask_2d.sum()
            if valid_pixels > 0:
                diff = diff * mask_2d.unsqueeze(-1)
                return diff.sum() / (valid_pixels * diff.shape[-1])
            # All pixels are marked sky; ignore this view for loss
            return diff.sum() * 0.0

        return diff.mean()

    def _render_targets_and_accumulate_loss(
        self,
        targets: List[Dict],
        proxies_bg: Dict[str, torch.Tensor],
        proxies_rigid: Optional[Dict[str, torch.Tensor]],
        proxies_distant: Optional[Dict[str, torch.Tensor]],
        node_state_rigid,
        masks: RigidMasks,
    ) -> Tuple[float, List[Dict]]:
        """
        遍历 target 视角，渲染并累积梯度到代理参数。
        """
        outputs: List[Dict] = []
        total_loss_val = 0.0
        view_count = max(len(targets), 1)
        num_sh = _num_sh_bases(self.sh_degree)

        for view_idx, target in enumerate(targets):
            view = target["view"]
            gt_img = target["gt_image"]
            target_frame_idx = int(target.get("frame_idx", 0))
            height, width = gt_img.shape[0], gt_img.shape[1]

            means_rigid_world = torch.empty(0, 3, device=self.device)
            quats_rigid_world = torch.empty(0, 4, device=self.device)
            rigid_scales_subset = torch.empty(0, 3, device=self.device)
            opacities_rigid = None
            rigid_sh_subset = torch.empty(0, num_sh, 3, device=self.device)

            if proxies_rigid is not None and node_state_rigid is not None:
                node_state_rigid.cur_frame = target_frame_idx
                if view_idx < len(masks.idx_tgt_rigid) and masks.idx_tgt_rigid[view_idx].numel() > 0:
                    idx = masks.idx_tgt_rigid[view_idx]
                    rigid_means_local_subset = proxies_rigid["means_p"][idx]
                    rigid_quats_local_subset = proxies_rigid["quats_p"][idx]
                    rigid_scales_subset = proxies_rigid["scales_p"][idx]
                    opacities_rigid = proxies_rigid["opacities_p"][idx]
                    rigid_sh_subset = proxies_rigid["colors_p"][idx]

                    means_rigid_world = self._transform_rigid_to_world(
                        node_state_rigid, rigid_means_local_subset, point_indices=idx
                    )
                    quats_rigid_world = self._transform_rigid_quats_to_world(
                        node_state_rigid, rigid_quats_local_subset, point_indices=idx
                    )

            rigid_subset = {
                "means": means_rigid_world,
                "quats": quats_rigid_world,
                "scales": rigid_scales_subset,
                "opacities": opacities_rigid if opacities_rigid is not None else torch.empty(0, device=self.device),
                "colors": rigid_sh_subset,
            }

            if proxies_rigid is not None and rigid_subset["means"].numel() > 0:
                merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = self._merge_params_with_rigid_subset(
                    proxies_bg=proxies_bg,
                    proxies_distant=proxies_distant,
                    rigid_subset=rigid_subset,
                )
            else:
                merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = self._merge_all_params(
                    proxies_bg=proxies_bg,
                    proxies_rigid=proxies_rigid,
                    proxies_distant=proxies_distant,
                    means_rigid_world=means_rigid_world,
                    quats_rigid_world=quats_rigid_world,
                    opacities_rigid=opacities_rigid,
                )

            merged_params = {
                "means_p": merged_means,
                "scales_p": merged_scales,
                "quats_p": merged_quats,
                "opacities_p": merged_opacities,
                "colors_p": merged_colors,
            }

            rgb, acc = self._render_single_view(merged_params, view, height, width)
            sky_mask = target.get("sky_mask")
            loss = self.compute_loss(rgb, gt_img, sky_mask=sky_mask) / view_count
            total_loss_val += float(loss.detach())
            grad_scaler = getattr(self, "grad_scaler", None)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if masks.mask_any_tgt_rigid is not None and proxies_rigid is not None and proxies_rigid["means_p"].grad is not None:
                grad_means_not_rendered = proxies_rigid["means_p"].grad[~masks.mask_any_tgt_rigid]
                if grad_means_not_rendered.numel() > 0:
                    max_grad_not_rendered = grad_means_not_rendered.abs().max().item()
                    if max_grad_not_rendered > 1e-6:
                        import warnings
                        warnings.warn(
                            f"[Sanity Check B] Gradients for points not rendered in any target should be 0, "
                            f"but max abs value is {max_grad_not_rendered:.2e}. This may indicate a bug."
                        )

            if view_idx < len(masks.idx_tgt_rigid):
                num_visible = len(masks.idx_tgt_rigid[view_idx])
                num_total = node_state_rigid.means.shape[0] if node_state_rigid is not None else 0
                if num_total > 0 and num_visible == 0:
                    import warnings
                    warnings.warn(
                        f"[Sanity Check C] Target {view_idx} has no visible rigid points ({num_visible}/{num_total}). "
                        f"This may indicate a visibility issue."
                    )

            if getattr(self, "log_images", False):
                outputs.append(
                    {
                        "rgb": rgb.detach().cpu(),
                        "acc": acc.detach().cpu(),
                        "loss": loss.detach().item(),
                    }
                )
            else:
                outputs.append({"loss": loss.detach().item()})

        return total_loss_val, outputs

    def _backward_to_render_params(
        self,
        render_params_bg: Dict[str, torch.Tensor],
        render_params_rigid: Optional[Dict[str, torch.Tensor]],
        render_params_distant: Optional[Dict[str, torch.Tensor]],
        proxies_bg: Dict[str, torch.Tensor],
        proxies_rigid: Optional[Dict[str, torch.Tensor]],
        proxies_distant: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        将 proxy 梯度反传到渲染参数。
        """
        grad_report: Dict[str, float] = {}
        grad_warned = getattr(self, "_proxy_grad_warned", set())
        strict = bool(getattr(self, "_strict_proxy_grad_active", False))
        warn_on_none = bool(getattr(self, "proxy_grad_warn_on_none", False))
        alert_on_nan = bool(getattr(self, "sentinel_alert_on_nan", False) or getattr(self, "_strict_checks_active", False))

        def _grad_or_zero(proxy_tensor: torch.Tensor, name: str) -> torch.Tensor:
            grad = proxy_tensor.grad
            if grad is None:
                if strict:
                    raise RuntimeError(f"Proxy gradient for {name} is None in strict mode.")
                if warn_on_none and name not in grad_warned:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Proxy gradient for {name} is None; using zeros for backward.")
                    grad_warned.add(name)
                grad_report[name] = 0.0
                return torch.zeros_like(proxy_tensor)
            if alert_on_nan and not torch.isfinite(grad).all():
                raise RuntimeError(f"Proxy gradient for {name} contains NaN/Inf.")
            grad_report[name] = float(grad.norm().detach())
            return grad

        render_tensors = [
            render_params_bg["means_r"],
            render_params_bg["scales_r"],
            render_params_bg["quats_r"],
            render_params_bg["opacities_r"],
            render_params_bg["colors_r"],
        ]
        grad_tensors = [
            _grad_or_zero(proxies_bg["means_p"], "bg.means"),
            _grad_or_zero(proxies_bg["scales_p"], "bg.scales"),
            _grad_or_zero(proxies_bg["quats_p"], "bg.quats"),
            _grad_or_zero(proxies_bg["opacities_p"], "bg.opacities"),
            _grad_or_zero(proxies_bg["colors_p"], "bg.colors"),
        ]

        if render_params_rigid is not None and proxies_rigid is not None:
            render_tensors += [
                render_params_rigid["means_r"],
                render_params_rigid["scales_r"],
                render_params_rigid["quats_r"],
                render_params_rigid["opacities_r"],
                render_params_rigid["colors_r"],
            ]
            grad_tensors += [
                _grad_or_zero(proxies_rigid["means_p"], "rigid.means"),
                _grad_or_zero(proxies_rigid["scales_p"], "rigid.scales"),
                _grad_or_zero(proxies_rigid["quats_p"], "rigid.quats"),
                _grad_or_zero(proxies_rigid["opacities_p"], "rigid.opacities"),
                _grad_or_zero(proxies_rigid["colors_p"], "rigid.colors"),
            ]

        if render_params_distant is not None and proxies_distant is not None:
            render_tensors += [
                render_params_distant["means_r"],
                render_params_distant["scales_r"],
                render_params_distant["quats_r"],
                render_params_distant["opacities_r"],
                render_params_distant["colors_r"],
            ]
            grad_tensors += [
                _grad_or_zero(proxies_distant["means_p"], "distant.means"),
                _grad_or_zero(proxies_distant["scales_p"], "distant.scales"),
                _grad_or_zero(proxies_distant["quats_p"], "distant.quats"),
                _grad_or_zero(proxies_distant["opacities_p"], "distant.opacities"),
                _grad_or_zero(proxies_distant["colors_p"], "distant.colors"),
            ]

        self._proxy_grad_warned = grad_warned
        self._last_proxy_grad_norms = grad_report
        torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)
        return grad_report

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
