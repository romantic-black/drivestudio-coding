"""
Minimal StreetForward Stage 3: 2D branch (mandatory) + bg + distant (mandatory).

Extends Stage 2.1: adds source views for 2D feature extraction and fusion;
splits pointcloud into NodeStateBackground + NodeStateDistant; both participate in
3D/2D features, GRU, proxies, merge render, and backward.
See docs/trainers/Minimal_StreetForward_Next_Steps_Stage3_2D_Branch.md and the plan.
"""

from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractors import (
    AlphaTWeightExtractor,
    FeatureBackprojector,
    FeatureFusion,
    ImageFeatureExtractor,
)
from models.streetforward.math_utils import _sh_to_rgb
from models.streetforward.metrics import compute_l1_loss_masked
from models.streetforward.minimal_trainer_stage2_1 import (
    _backward_to_render_params_bg,
    _create_proxy_params,
    _proxy_params_to_render_dict,
)
from models.streetforward.minimal_trainer_stage2_1 import MinimalStreetForwardStage2_1
from models.streetforward.node_states import NodeStateBackground, NodeStateDistant

logger = logging.getLogger(__name__)


def _merge_params_bg_distant(
    proxies_bg: Dict[str, torch.Tensor],
    proxies_distant: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Merge bg and distant proxy params for rendering (no rigid)."""
    if proxies_distant is None or proxies_bg["means_p"].shape[0] == 0 and proxies_distant["means_p"].shape[0] == 0:
        return _proxy_params_to_render_dict(proxies_bg)
    means = torch.cat([proxies_bg["means_p"], proxies_distant["means_p"]], dim=0)
    quats = torch.cat([proxies_bg["quats_p"], proxies_distant["quats_p"]], dim=0)
    scales = torch.cat([proxies_bg["scales_p"], proxies_distant["scales_p"]], dim=0)
    opacities = torch.cat([proxies_bg["opacities_p"], proxies_distant["opacities_p"]], dim=0)
    colors = torch.cat([proxies_bg["colors_p"], proxies_distant["colors_p"]], dim=0)
    return {
        "means_r": means,
        "scales_r": scales,
        "quats_r": quats,
        "opacities_r": opacities,
        "colors_r": colors,
    }


def _backward_to_render_params_distant(
    render_params_distant: Dict[str, torch.Tensor],
    proxies_distant: Dict[str, torch.Tensor],
) -> None:
    """Push proxy gradients to render_params_distant (kept for API compatibility)."""

    def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
        g = t.grad
        return g if g is not None else torch.zeros_like(t)

    render_tensors = [
        render_params_distant["means_r"],
        render_params_distant["scales_r"],
        render_params_distant["quats_r"],
        render_params_distant["opacities_r"],
        render_params_distant["colors_r"],
    ]
    grad_tensors = [
        _grad_or_zero(proxies_distant["means_p"]),
        _grad_or_zero(proxies_distant["scales_p"]),
        _grad_or_zero(proxies_distant["quats_p"]),
        _grad_or_zero(proxies_distant["opacities_p"]),
        _grad_or_zero(proxies_distant["colors_p"]),
    ]
    torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)


def _backward_to_render_params_bg_distant(
    render_params_bg: Dict[str, torch.Tensor],
    proxies_bg: Dict[str, torch.Tensor],
    render_params_distant: Optional[Dict[str, torch.Tensor]],
    proxies_distant: Optional[Dict[str, torch.Tensor]],
) -> None:
    """Single autograd.backward from proxies (bg + distant) to corresponding render params."""

    def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
        g = t.grad
        return g if g is not None else torch.zeros_like(t)

    render_tensors = [
        render_params_bg["means_r"],
        render_params_bg["scales_r"],
        render_params_bg["quats_r"],
        render_params_bg["opacities_r"],
        render_params_bg["colors_r"],
    ]
    grad_tensors = [
        _grad_or_zero(proxies_bg["means_p"]),
        _grad_or_zero(proxies_bg["scales_p"]),
        _grad_or_zero(proxies_bg["quats_p"]),
        _grad_or_zero(proxies_bg["opacities_p"]),
        _grad_or_zero(proxies_bg["colors_p"]),
    ]

    if render_params_distant is not None and proxies_distant is not None:
        render_tensors.extend(
            [
                render_params_distant["means_r"],
                render_params_distant["scales_r"],
                render_params_distant["quats_r"],
                render_params_distant["opacities_r"],
                render_params_distant["colors_r"],
            ]
        )
        grad_tensors.extend(
            [
                _grad_or_zero(proxies_distant["means_p"]),
                _grad_or_zero(proxies_distant["scales_p"]),
                _grad_or_zero(proxies_distant["quats_p"]),
                _grad_or_zero(proxies_distant["opacities_p"]),
                _grad_or_zero(proxies_distant["colors_p"]),
            ]
        )

    torch.autograd.backward(tensors=render_tensors, grad_tensors=grad_tensors)


class MinimalStreetForwardStage3_2d(MinimalStreetForwardStage2_1):
    """
    Stage 3: 2D branch (mandatory) + bg + distant (mandatory).
    Requires source_views/source_images in batch; pointcloud split into bg + distant by bbx/input_aabb.
    """

    def __init__(
        self,
        config,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(config, device, **kwargs)
        model_cfg = config.model

        feat_2d_channels = int(model_cfg.get("feat_2d_channels", 16))
        feat_2d_downscale = int(model_cfg.get("feat_2d_downscale", 1))
        outdim = self.feat_3d_dim
        self.fused_in_dim = outdim + feat_2d_channels

        # Parent's mlp_params_embed was built with config param_embed_dim; use it for GRU input size.
        param_embed_out_dim = self.param_embed_dim
        self.offset_gru_hidden_dim = self.fused_in_dim
        gru_in_dim = self.fused_in_dim + param_embed_out_dim

        self.image_feature_extractor = ImageFeatureExtractor(
            in_channels=6,
            feat_channels=feat_2d_channels,
            feature_downscale=feat_2d_downscale,
        ).to(device)
        self.alpha_t_extractor = AlphaTWeightExtractor(
            renderer=self.renderer,
            sh_degree=self.sh_degree,
            tile_size=16,
        )
        self.feature_backprojector = FeatureBackprojector()
        self.feature_fusion = FeatureFusion(use_visibility=False)

        self.gru_update = nn.Linear(
            gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim
        ).to(device)
        self.gru_candidate = nn.Linear(
            gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim
        ).to(device)
        self.gru_reset = nn.Linear(
            gru_in_dim + self.offset_gru_hidden_dim, self.offset_gru_hidden_dim
        ).to(device)
        self.gru_to_head = nn.Identity()

        from models.streetforward.math_utils import _num_sh_bases
        num_sh = _num_sh_bases(self.sh_degree)
        self.mlp_offset_pos = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).to(device)
        self.mlp_conv = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        ).to(device)
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)
        self.gaussion_decoder = nn.Sequential(
            nn.Linear(self.fused_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * num_sh),
        ).to(device)
        for m in (self.mlp_offset_pos, self.mlp_conv, self.mlp_opacity, self.gaussion_decoder):
            if isinstance(m, nn.Sequential) and len(m) > 0:
                last = m[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

        self.node_states_distant: Dict[Tuple[int, int], Optional[NodeStateDistant]] = {}
        self.h_cache_distant: Dict[Tuple[int, int], torch.Tensor] = {}

        # Optional per-region background point limits for minimal trainer (read from dataset.pointcloud if present).
        pc_cfg = getattr(config.dataset, "pointcloud", None)
        self.near_max_points: Optional[int] = None
        self.distant_max_points: Optional[int] = None
        if pc_cfg is not None:
            try:
                near_val = pc_cfg.get("near_max_points") if hasattr(pc_cfg, "get") else getattr(
                    pc_cfg, "near_max_points", None
                )
            except Exception:
                near_val = None
            try:
                distant_val = pc_cfg.get("distant_max_points") if hasattr(pc_cfg, "get") else getattr(
                    pc_cfg, "distant_max_points", None
                )
            except Exception:
                distant_val = None
            if near_val is not None:
                self.near_max_points = int(near_val)
            if distant_val is not None:
                self.distant_max_points = int(distant_val)

        self.optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=float(config.optimizer.get("lr", 1e-3)),
            eps=float(config.optimizer.get("eps", 1e-15)),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
        )

    def _init_node_state_from_arrays(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        state_cls: type,
    ):
        if len(points) == 0:
            raise ValueError("Empty point cloud for node state.")
        means = torch.from_numpy(points).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        if colors_tensor.numel() > 0 and colors_tensor.max() > 1.0 + 1e-3:
            colors_tensor = colors_tensor / 255.0
        if colors_tensor.dim() == 1:
            colors_tensor = colors_tensor.unsqueeze(-1).expand(-1, 3)
        elif colors_tensor.shape[1] != 3:
            colors_tensor = colors_tensor[:, :3]
        from models.streetforward.math_utils import _num_sh_bases, _rgb_to_sh
        from models.streetforward.minimal_trainer_stage1_1 import _random_quat_tensor

        scales_log = self._compute_initial_scales(means)
        quats = _random_quat_tensor(means.shape[0], device=self.device)
        opacity_logit = torch.logit(torch.full((means.shape[0], 1), 0.1, device=self.device))
        num_sh = _num_sh_bases(self.sh_degree)
        sh_dc = _rgb_to_sh(colors_tensor)
        sh_rest = torch.zeros((means.shape[0], num_sh - 1, 3), device=self.device)
        return state_cls(
            means=means.detach().clone(),
            scales_log=scales_log.detach().clone(),
            quats=quats.detach().clone(),
            opacity_logit=opacity_logit.detach().clone(),
            sh_dc=sh_dc.detach().clone(),
            sh_rest=sh_rest.detach().clone(),
        )

    def _get_or_init_node_states_bg_distant(
        self, batch: Dict
    ) -> Tuple[NodeStateBackground, Optional[NodeStateDistant]]:
        key = self._batch_key(batch)
        if key in self.node_states_bg and key in self.node_states_distant:
            return self.node_states_bg[key], self.node_states_distant.get(key)
        scene_id = batch["scene_id"]
        segment_id = batch["segment_id"]
        if torch.is_tensor(scene_id):
            scene_id = int(scene_id.item())
        if torch.is_tensor(segment_id):
            segment_id = int(segment_id.item()) if segment_id.numel() == 1 else int(segment_id[0].item())
        pointcloud = batch["pointcloud"]
        if isinstance(pointcloud, dict):
            background = pointcloud.get("background", np.zeros((0, 6), dtype=np.float32))
            points = background[:, :3].astype(np.float32)
            colors = (
                background[:, 3:6].astype(np.float32)
                if background.shape[1] >= 6
                else np.zeros_like(points, dtype=np.float32)
            )
        else:
            points = np.asarray(getattr(pointcloud, "points", np.zeros((0, 3))), dtype=np.float32)
            raw_colors = getattr(pointcloud, "colors", None)
            colors = (
                np.asarray(raw_colors, dtype=np.float32)
                if raw_colors is not None and np.asarray(raw_colors).size > 0
                else np.zeros_like(points, dtype=np.float32)
            )
        if len(points) == 0:
            raise ValueError(
                f"Empty point cloud for scene_id={scene_id}, segment_id={segment_id}."
            )
        crop_min = self.bbx_min.cpu().numpy()
        crop_max = self.bbx_max.cpu().numpy()
        in_crop = ((points >= crop_min) & (points <= crop_max)).all(axis=1)
        fg_points = points[in_crop]
        fg_colors = colors[in_crop]
        distant_points = points[~in_crop]
        distant_colors = colors[~in_crop]

        def _limit_region(
            pts: np.ndarray,
            cols: np.ndarray,
            max_points: Optional[int],
        ) -> Tuple[np.ndarray, np.ndarray]:
            if max_points is None or max_points <= 0 or len(pts) <= max_points:
                return pts, cols
            step = max(1, len(pts) // max_points)
            idx = np.arange(0, len(pts), dtype=int)[::step]
            if len(idx) > max_points:
                idx = idx[:max_points]
            return pts[idx], cols[idx]

        fg_points, fg_colors = _limit_region(
            fg_points, fg_colors, self.near_max_points
        )
        distant_points, distant_colors = _limit_region(
            distant_points, distant_colors, self.distant_max_points
        )
        if len(fg_points) == 0:
            raise ValueError(
                f"No points inside segment_aabb for scene_id={scene_id}, segment_id={segment_id}."
            )
        node_state_bg = self._init_node_state_from_arrays(fg_points, fg_colors, NodeStateBackground)
        node_state_distant: Optional[NodeStateDistant] = None
        if len(distant_points) > 0:
            node_state_distant = self._init_node_state_from_arrays(
                distant_points.astype(np.float32),
                distant_colors.astype(np.float32),
                NodeStateDistant,
            )
        self.node_states_bg[key] = node_state_bg
        self.node_states_distant[key] = node_state_distant
        return node_state_bg, node_state_distant

    def _prepare_gaussians_bg_distant(
        self,
        node_state_bg: NodeStateBackground,
        node_state_distant: Optional[NodeStateDistant],
    ) -> Tuple[Dict[str, torch.Tensor], int, int]:
        from models.streetforward.math_utils import _num_sh_bases

        num_sh = _num_sh_bases(self.sh_degree)
        means_bg = node_state_bg.means
        quats_bg = node_state_bg.quats
        scales_bg = torch.exp(node_state_bg.scales_log)
        opacities_bg = torch.sigmoid(node_state_bg.opacity_logit).squeeze(-1)
        colors_bg = torch.cat([node_state_bg.sh_dc[:, None, :], node_state_bg.sh_rest], dim=1)
        num_bg = means_bg.shape[0]
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
            colors_distant = torch.cat(
                [node_state_distant.sh_dc[:, None, :], node_state_distant.sh_rest], dim=1
            )
            num_distant = means_distant.shape[0]
        gaussians = {
            "means": torch.cat([means_bg, means_distant], dim=0),
            "quats": torch.cat([quats_bg, quats_distant], dim=0),
            "scales": torch.cat([scales_bg, scales_distant], dim=0),
            "opacities": torch.cat([opacities_bg, opacities_distant], dim=0),
            "colors": torch.cat([colors_bg, colors_distant], dim=0),
        }
        return gaussians, num_bg, num_distant

    def _compute_2d_features_bg_distant(
        self,
        gaussians: Dict[str, torch.Tensor],
        num_bg: int,
        num_distant: int,
        source_views: List,
        source_images: List[torch.Tensor],
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        total = num_bg + num_distant
        if total == 0:
            return None, None
        with torch.no_grad():
            rendered_rgbs = self.alpha_t_extractor.render_rgb_only(
                gaussians, source_views, height, width
            )
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
        feat_2d_all = self.alpha_t_extractor.render_and_backproject_streaming(
            gaussians=gaussians,
            cameras=source_views,
            features_2d=features_2d,
            height=height,
            width=width,
            num_gaussians=total,
            backprojector=self.feature_backprojector,
        )
        feat_2d_bg = feat_2d_all[:num_bg] if num_bg > 0 else None
        feat_2d_distant = feat_2d_all[num_bg:] if num_distant > 0 else None
        return feat_2d_bg, feat_2d_distant

    def _fuse_features(
        self, feat_3d: torch.Tensor, feat_2d: Optional[torch.Tensor], visibility: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if feat_2d is None:
            return feat_3d
        if visibility is None:
            visibility = torch.ones(feat_3d.shape[0], device=feat_3d.device)
        return self.feature_fusion.fuse(feat_3d, feat_2d, visibility)

    def _render_params_from_offsets_distant(
        self, node_state_distant: NodeStateDistant, offsets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self._render_params_from_offsets(node_state_distant, offsets)

    def _update_node_state_distant(
        self,
        node_state_distant: NodeStateDistant,
        render_params: Dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            node_state_distant.means.copy_(render_params["means_r"].detach())
            node_state_distant.scales_log.copy_(render_params["scales_log_r"].detach())
            node_state_distant.quats.copy_(render_params["quats_r"].detach())
            node_state_distant.opacity_logit.copy_(render_params["opacity_logit_r"].detach())
            node_state_distant.sh_dc.copy_(render_params["sh_dc_r"].detach())
            node_state_distant.sh_rest.copy_(render_params["sh_rest_r"].detach())

    def _render_multi_view(
        self,
        render_params: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Batched multi-view render, following Stage 2.2: render all views in one gsplat call
        when they share the same (H, W); otherwise return None to signal fallback.
        """
        if not targets:
            return None
        viewmats_list: List[torch.Tensor] = []
        Ks_list: List[torch.Tensor] = []
        heights: List[int] = []
        widths: List[int] = []
        for target in targets:
            view = target["view"]
            gt_image = target["gt_image"]
            if gt_image.dim() == 4:
                gt_image = gt_image.squeeze(0)
            h, w = int(gt_image.shape[0]), int(gt_image.shape[1])
            heights.append(h)
            widths.append(w)
            c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            viewmat = torch.linalg.inv(c2w)
            if viewmat.dim() == 2:
                viewmat = viewmat.unsqueeze(0)
            viewmats_list.append(viewmat)
            if hasattr(view, "Ks"):
                k_mat = view.Ks[0:1]
            elif hasattr(view, "K"):
                k_mat = view.K
            else:
                k_mat = torch.eye(3, device=self.device).unsqueeze(0)
            if k_mat.dim() == 2:
                k_mat = k_mat.unsqueeze(0)
            Ks_list.append(k_mat)
        h0, w0 = heights[0], widths[0]
        if any(h != h0 or w != w0 for h, w in zip(heights, widths)):
            return None
        viewmats = torch.cat(viewmats_list, dim=0)
        Ks = torch.cat(Ks_list, dim=0)
        render, alpha, _ = self.renderer(
            means=render_params["means_r"],
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=viewmats,
            Ks=Ks,
            width=w0,
            height=h0,
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
        C = viewmats.shape[0]
        result: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c in range(C):
            rgb = render[c, ..., :3]
            acc = alpha[c, ..., 0]
            result.append((rgb, acc))
        return result

    def forward(self, batch: Dict) -> Dict[str, Any]:
        if "pointcloud" not in batch:
            raise ValueError("Batch must contain 'pointcloud'.")
        targets = batch["targets"]
        if not targets:
            raise ValueError("Stage 3 requires at least one target.")
        source_views = batch.get("source_views")
        source_images = batch.get("source_images")
        if not source_views or not source_images:
            raise ValueError(
                "Stage 3 requires source_views and source_images. "
                "Use convert_batch_to_minimal_format(..., include_source_for_2d=True)."
            )

        node_state_bg, node_state_distant = self._get_or_init_node_states_bg_distant(batch)
        key = self._batch_key(batch)
        means_bg = node_state_bg.means
        anchor_rgb_bg = _sh_to_rgb(node_state_bg.sh_dc)

        feat_3d_crop_bg = self._build_3d_features(means_bg, anchor_rgb_bg)
        sample_img = source_images[0]
        height = int(sample_img.shape[0] if sample_img.dim() == 3 else sample_img.shape[1])
        width = int(sample_img.shape[1] if sample_img.dim() == 3 else sample_img.shape[2])
        gaussians_all, num_bg, num_distant = self._prepare_gaussians_bg_distant(
            node_state_bg, node_state_distant
        )
        feat_2d_bg, feat_2d_distant = self._compute_2d_features_bg_distant(
            gaussians_all, num_bg, num_distant, source_views, source_images, height, width
        )
        vis_bg = torch.ones(num_bg, device=self.device)
        feat_bg_input = self._fuse_features(feat_3d_crop_bg, feat_2d_bg, vis_bg)
        feat_distant_input = None
        if num_distant > 0 and feat_2d_distant is not None:
            zeros_3d = torch.zeros(
                num_distant, self.feat_3d_dim, device=self.device, dtype=feat_2d_distant.dtype
            )
            vis_d = torch.ones(num_distant, device=self.device)
            feat_distant_input = self._fuse_features(zeros_3d, feat_2d_distant, vis_d)

        params_bg = self._build_params_for_embed(node_state_bg, coord_space="world")
        h_old_bg = self._get_or_init_hidden(
            self.h_cache_bg, key, node_state_bg.means.shape[0], node_state_bg, "bg"
        )
        offsets_bg, h_new_bg = self._predict_offsets_gru(
            feat_bg_input, params_bg, h_old_bg, mask_update_rigid=None
        )
        render_params_bg = self._render_params_from_offsets(node_state_bg, offsets_bg)

        render_params_distant = None
        h_new_distant = None
        if node_state_distant is not None and feat_distant_input is not None and feat_distant_input.numel() > 0:
            params_distant = self._build_params_for_embed(node_state_distant, coord_space="world")
            h_old_distant = self._get_or_init_hidden(
                self.h_cache_distant,
                key,
                node_state_distant.means.shape[0],
                node_state_distant,
                "distant",
            )
            offsets_distant, h_new_distant = self._predict_offsets_gru(
                feat_distant_input, params_distant, h_old_distant, mask_update_rigid=None
            )
            render_params_distant = self._render_params_from_offsets_distant(
                node_state_distant, offsets_distant
            )

        if not self.training:
            if render_params_distant is not None:
                merged = {
                    "means_r": torch.cat([render_params_bg["means_r"], render_params_distant["means_r"]]),
                    "scales_r": torch.cat([render_params_bg["scales_r"], render_params_distant["scales_r"]]),
                    "quats_r": torch.cat([render_params_bg["quats_r"], render_params_distant["quats_r"]]),
                    "opacities_r": torch.cat([render_params_bg["opacities_r"], render_params_distant["opacities_r"]]),
                    "colors_r": torch.cat([render_params_bg["colors_r"], render_params_distant["colors_r"]]),
                }
            else:
                merged = {
                    "means_r": render_params_bg["means_r"],
                    "scales_r": render_params_bg["scales_r"],
                    "quats_r": render_params_bg["quats_r"],
                    "opacities_r": render_params_bg["opacities_r"],
                    "colors_r": render_params_bg["colors_r"],
                }

            pred_rgbs: List[torch.Tensor] = []
            gt_images: List[torch.Tensor] = []

            # Eval path: try batched multi-view render first, fallback to per-view.
            multi_result = self._render_multi_view(merged, targets)
            if multi_result is not None:
                for i, target in enumerate(targets):
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    pred_rgb = multi_result[i][0]
                    pred_rgbs.append(pred_rgb)
                    gt_images.append(gt_image)
            else:
                for target in targets:
                    view = target["view"]
                    gt_image = target["gt_image"]
                    if gt_image.dim() == 4:
                        gt_image = gt_image.squeeze(0)
                    h, w = gt_image.shape[0], gt_image.shape[1]
                    pred_rgb, _ = self._render_single_view(merged, view, h, w)
                    pred_rgbs.append(pred_rgb)
                    gt_images.append(gt_image)

            loss = torch.tensor(0.0, device=self.device)
            return {
                "loss": loss,
                "render_params": render_params_bg,
                "proxies": None,
                "_node_state_bg": node_state_bg,
                "_h_new_bg": h_new_bg,
                "_cache_key": key,
                "pred_rgbs": pred_rgbs,
                "gt_images": gt_images,
                "pred_rgb": pred_rgbs[0],
                "gt_image": gt_images[0],
                "_render_params_distant": render_params_distant,
                "_h_new_distant": h_new_distant,
                "_node_state_distant": node_state_distant,
            }

        proxies_bg = _create_proxy_params(render_params_bg)
        proxies_distant = _create_proxy_params(render_params_distant) if render_params_distant is not None else None
        merged_for_render = _merge_params_bg_distant(proxies_bg, proxies_distant)

        pred_rgbs: List[torch.Tensor] = []
        gt_images: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []

        # Training: try batched multi-view render (like Stage 2.2), fallback to per-view.
        multi_result = self._render_multi_view(merged_for_render, targets)

        if multi_result is not None:
            for i, target in enumerate(targets):
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                pred_rgb = multi_result[i][0]
                loss_i = compute_l1_loss_masked(
                    pred_rgb, gt_image, None, sky_mask=target.get("sky_mask")
                )
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(loss_i)
        else:
            for target in targets:
                view = target["view"]
                gt_image = target["gt_image"]
                if gt_image.dim() == 4:
                    gt_image = gt_image.squeeze(0)
                height, width = gt_image.shape[0], gt_image.shape[1]
                pred_rgb, _ = self._render_single_view(merged_for_render, view, height, width)
                loss_i = compute_l1_loss_masked(
                    pred_rgb, gt_image, None, sky_mask=target.get("sky_mask")
                )
                pred_rgbs.append(pred_rgb)
                gt_images.append(gt_image)
                losses.append(loss_i)

        loss = torch.stack(losses).mean() if losses else torch.tensor(
            0.0, device=render_params_bg["means_r"].device, dtype=render_params_bg["means_r"].dtype
        )
        return {
            "loss": loss,
            "render_params": render_params_bg,
            "proxies": proxies_bg,
            "_node_state_bg": node_state_bg,
            "_h_new_bg": h_new_bg,
            "_cache_key": key,
            "pred_rgbs": pred_rgbs,
            "gt_images": gt_images,
            "pred_rgb": pred_rgbs[0],
            "gt_image": gt_images[0],
            "_render_params_distant": render_params_distant,
            "_proxies_distant": proxies_distant,
            "_h_new_distant": h_new_distant,
            "_node_state_distant": node_state_distant,
        }

    def train_step(self, batch: Dict, step: Optional[int] = None) -> Dict[str, Any]:
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(batch)
        # region agent log
        try:
            log_payload = {
                "sessionId": "45949f",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "minimal_trainer_stage3_2d.py:680",
                "message": "train_step forward output summary",
                "data": {
                    "has_loss": bool(torch.is_tensor(out.get("loss"))),
                    "has_proxies_bg": out.get("proxies") is not None,
                    "has_render_params_bg": out.get("render_params") is not None,
                    "has_proxies_distant": out.get("_proxies_distant") is not None,
                    "has_render_params_distant": out.get("_render_params_distant") is not None,
                },
                "timestamp": int(time.time() * 1000),
            }
            with open("/root/drivestudio-coding/.cursor/debug-45949f.log", "a") as _f:
                _f.write(json.dumps(log_payload) + "\n")
        except Exception:
            pass
        # endregion agent log
        # First backprop from scalar loss to proxy params (and other learnable params).
        if torch.is_tensor(out.get("loss")):
            out["loss"].backward()
        # region agent log
        try:
            log_payload = {
                "sessionId": "45949f",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "minimal_trainer_stage3_2d.py:685",
                "message": "before proxy backward calls",
                "data": {
                    "has_proxies_bg": out.get("proxies") is not None,
                    "has_render_params_bg": out.get("render_params") is not None,
                    "has_proxies_distant": out.get("_proxies_distant") is not None,
                    "has_render_params_distant": out.get("_render_params_distant") is not None,
                },
                "timestamp": int(time.time() * 1000),
            }
            with open("/root/drivestudio-coding/.cursor/debug-45949f.log", "a") as _f:
                _f.write(json.dumps(log_payload) + "\n")
        except Exception:
            pass
        # endregion agent log
        if out.get("proxies") is not None:
            _backward_to_render_params_bg_distant(
                out["render_params"],
                out["proxies"],
                out.get("_render_params_distant"),
                out.get("_proxies_distant"),
            )
        self.optimizer.step()
        if "_h_new_bg" in out and "_cache_key" in out:
            self.h_cache_bg[out["_cache_key"]] = out["_h_new_bg"].detach()
        if out.get("_h_new_distant") is not None and "_cache_key" in out:
            self.h_cache_distant[out["_cache_key"]] = out["_h_new_distant"].detach()
        if (
            self.update_node_state_interval > 0
            and step is not None
            and step % self.update_node_state_interval == 0
        ):
            if "_node_state_bg" in out:
                self._update_node_state_bg(out["_node_state_bg"], out["render_params"])
            if out.get("_node_state_distant") is not None and out.get("_render_params_distant") is not None:
                self._update_node_state_distant(out["_node_state_distant"], out["_render_params_distant"])
        num_gaussians_bg = int(out["_node_state_bg"].means.shape[0])
        node_state_distant = out.get("_node_state_distant")
        num_gaussians_distant = (
            int(node_state_distant.means.shape[0]) if node_state_distant is not None else 0
        )
        num_targets = len(batch.get("targets", []))
        num_source_views = len(batch.get("source_views", []))
        return {
            "loss": out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"],
            "pred_rgbs": out["pred_rgbs"],
            "gt_images": out["gt_images"],
            "pred_rgb": out["pred_rgb"],
            "gt_image": out["gt_image"],
            "num_gaussians_bg": num_gaussians_bg,
            "num_gaussians_distant": num_gaussians_distant,
            "num_targets": num_targets,
            "num_source_views": num_source_views,
        }

    def reset_node_state(self) -> None:
        super().reset_node_state()
        self.node_states_distant.clear()
        if hasattr(self, "h_cache_distant"):
            self.h_cache_distant.clear()


__all__ = [
    "MinimalStreetForwardStage3_2d",
    "_merge_params_bg_distant",
    "_backward_to_render_params_distant",
]
