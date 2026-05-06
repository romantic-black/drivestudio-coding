from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from gsplat.rendering import rasterization as _gsplat_rasterization
except ImportError:  # pragma: no cover
    _gsplat_rasterization = None

from models.feature_extractors import AlphaTWeightExtractorV3, FeatureBackprojector
from models.streetforward.math_utils import _num_sh_bases, _rgb_to_sh
from models.streetforward.node_states import NodeStateSky
from models.streetforward.sky_shell_init import SKY_UP_MULTISCENE, fibonacci_shell_means

from .scene_render_provider import SceneRenderPack
from .sky_loss import skybranch_loss
from .sky_render_utils import (
    build_sky_regions,
    ensure_hw1,
    ensure_hwc3,
    get_cfg,
    resolve_view_intrinsics,
    rotation_only_viewmat_from_view,
    squeeze_mask,
    stack_hwc3,
    stack_hw1,
)


class _ResBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class SkyFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 8, hidden_dim: int = 64, output_dim: int = 64, num_blocks: int = 3) -> None:
        super().__init__()
        blocks: List[nn.Module] = [nn.Conv2d(in_channels, hidden_dim, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(int(num_blocks), 1)):
            blocks.append(_ResBlock2d(hidden_dim))
        blocks.append(nn.Conv2d(hidden_dim, output_dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        *,
        source_rgb: torch.Tensor,
        current_sky_rgb: torch.Tensor,
        sky_mask: torch.Tensor,
        scene_alpha: torch.Tensor,
    ) -> torch.Tensor:
        if source_rgb.dim() != 4 or int(source_rgb.shape[-1]) != 3:
            raise ValueError(f"source_rgb must be [V,H,W,3], got {tuple(source_rgb.shape)}")
        current_sky_rgb = current_sky_rgb.to(device=source_rgb.device, dtype=source_rgb.dtype)
        sky_mask = sky_mask.to(device=source_rgb.device, dtype=source_rgb.dtype)
        scene_alpha = scene_alpha.to(device=source_rgb.device, dtype=source_rgb.dtype)
        if sky_mask.dim() == 3:
            sky_mask = sky_mask.unsqueeze(-1)
        if scene_alpha.dim() == 3:
            scene_alpha = scene_alpha.unsqueeze(-1)
        trans = (1.0 - scene_alpha.detach()).clamp(0.0, 1.0)
        x = torch.cat(
            [
                source_rgb,
                current_sky_rgb.detach(),
                sky_mask.clamp(0.0, 1.0),
                trans,
            ],
            dim=-1,
        )
        feat = self.net(x.permute(0, 3, 1, 2).contiguous())
        return feat.permute(0, 2, 3, 1).contiguous()


@dataclass
class SkyBranchForwardOutput:
    loss: torch.Tensor
    logs: Dict[str, torch.Tensor]
    render_params_sky: Dict[str, torch.Tensor]
    node_state_sky: NodeStateSky
    h_new_sky: torch.Tensor
    cache_key: Tuple[int, int]
    source_pair_valid_mask: torch.Tensor
    support: torch.Tensor
    sky_rgb: torch.Tensor
    sky_alpha: torch.Tensor
    comp_rgb: torch.Tensor


class SkyBranchV0(nn.Module):
    def __init__(
        self,
        config: Any,
        device: torch.device,
        *,
        renderer: Optional[Any] = None,
        alpha_t_extractor: Optional[Any] = None,
        backprojector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        sky_cfg = get_cfg(config, "sky", {}) or {}
        branch_cfg = get_cfg(config, "sky_branch", {}) or {}
        init_cfg = get_cfg(branch_cfg, "init", {}) or {}
        lifting_cfg = get_cfg(branch_cfg, "lifting", {}) or {}
        feat_cfg = get_cfg(branch_cfg, "feature_extractor", {}) or {}

        self.sh_degree = int(get_cfg(branch_cfg, "sh_degree", get_cfg(config, "sh_degree", 2)))
        self.sky_resolution = int(get_cfg(sky_cfg, "resolution", 32))
        self.sky_radius = float(get_cfg(sky_cfg, "radius", 80.0))
        self.sky_center_local = torch.tensor(list(get_cfg(sky_cfg, "center_local", [0.0, 0.0, 0.0])), device=device, dtype=torch.float32)
        self.sky_hemisphere = bool(get_cfg(sky_cfg, "hemisphere", True))
        self.sky_hemisphere_up = tuple(float(x) for x in get_cfg(sky_cfg, "hemisphere_up", SKY_UP_MULTISCENE))
        self.opacity_init = float(get_cfg(init_cfg, "opacity_init", 0.7))
        scale_init = get_cfg(init_cfg, "scale_init", {}) or {}
        self.scale_init_log = float(get_cfg(scale_init, "isotropic_log_value", -1.5))
        self.feature_dim = int(get_cfg(branch_cfg, "feature_dim", 64))
        self.hidden_dim = int(get_cfg(branch_cfg, "hidden_dim", 64))
        self.support_min = float(get_cfg(lifting_cfg, "support_min", 1.0e-4))
        self.weight_threshold = float(get_cfg(lifting_cfg, "weight_threshold", 1.0e-5))
        self.sky_core_erode_kernel = int(get_cfg(lifting_cfg, "sky_core_erode_kernel", 5))
        self.use_sky_core_mask = bool(get_cfg(lifting_cfg, "use_sky_core_mask", True))
        eta_cfg = get_cfg(branch_cfg, "eta", {}) or {}
        self.eta_scales = float(get_cfg(eta_cfg, "scales", 0.03))
        self.eta_opacity = float(get_cfg(eta_cfg, "opacity", 0.20))
        self.eta_sh_dc = float(get_cfg(eta_cfg, "sh_dc", 0.05))
        self.eta_sh_rest = float(get_cfg(eta_cfg, "sh_rest", 0.02))

        self.renderer = renderer or _gsplat_rasterization
        if self.renderer is None and alpha_t_extractor is None:
            raise ImportError("gsplat is required unless alpha_t_extractor/renderer are injected.")
        self.alpha_t_extractor = alpha_t_extractor or AlphaTWeightExtractorV3(self.renderer, self.sh_degree, tile_size=16)
        self.feature_backprojector = backprojector or FeatureBackprojector(weight_threshold=self.weight_threshold)

        self.feature_extractor = SkyFeatureExtractor(
            in_channels=int(get_cfg(feat_cfg, "in_channels", 8)),
            hidden_dim=int(get_cfg(feat_cfg, "hidden_dim", 64)),
            output_dim=int(get_cfg(feat_cfg, "output_dim", self.feature_dim)),
            num_blocks=int(get_cfg(feat_cfg, "num_blocks", 3)),
        ).to(device)
        self.param_embed = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.feature_dim),
        ).to(device)
        self.gru = nn.GRUCell(self.feature_dim * 2, self.hidden_dim).to(device)
        num_sh = _num_sh_bases(self.sh_degree)
        self.scale_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 3)).to(device)
        self.opacity_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)).to(device)
        self.sh_head = nn.Sequential(nn.Linear(self.hidden_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 3 * num_sh)).to(device)

        self.node_states_sky: Dict[Tuple[int, int], NodeStateSky] = {}
        self.h_cache_sky: Dict[Tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def batch_key(batch: Dict[str, Any]) -> Tuple[int, int]:
        scene_id = batch["scene_id"]
        segment_id = batch["segment_id"]
        if torch.is_tensor(scene_id):
            scene_id = int(scene_id.item())
        if torch.is_tensor(segment_id):
            segment_id = int(segment_id.item()) if segment_id.numel() == 1 else int(segment_id[0].item())
        return int(scene_id), int(segment_id)

    def get_or_init_node_state(self, batch: Dict[str, Any]) -> NodeStateSky:
        key = self.batch_key(batch)
        if key in self.node_states_sky:
            return self.node_states_sky[key]
        means = fibonacci_shell_means(
            self.sky_resolution,
            self.sky_radius,
            self.sky_center_local.to(self.device),
            hemisphere=self.sky_hemisphere,
            device=self.device,
            dtype=torch.float32,
            up=self.sky_hemisphere_up,
        )
        n = int(means.shape[0])
        scales_log = torch.full((n, 3), self.scale_init_log, device=self.device, dtype=means.dtype)
        quats = torch.zeros(n, 4, device=self.device, dtype=means.dtype)
        quats[:, 0] = 1.0
        opacity_logit = torch.logit(torch.full((n, 1), self.opacity_init, device=self.device, dtype=means.dtype))
        sh_dc = _rgb_to_sh(torch.full((n, 3), 0.5, device=self.device, dtype=means.dtype))
        sh_rest = torch.zeros(n, _num_sh_bases(self.sh_degree) - 1, 3, device=self.device, dtype=means.dtype)
        state = NodeStateSky(means=means, scales_log=scales_log, quats=quats, opacity_logit=opacity_logit, sh_dc=sh_dc, sh_rest=sh_rest)
        self.node_states_sky[key] = state
        return state

    def get_or_init_hidden(self, key: Tuple[int, int], state: NodeStateSky) -> torch.Tensor:
        if key not in self.h_cache_sky or int(self.h_cache_sky[key].shape[0]) != int(state.means.shape[0]):
            self.h_cache_sky[key] = torch.zeros(state.means.shape[0], self.hidden_dim, device=self.device, dtype=state.means.dtype)
        return self.h_cache_sky[key]

    def state_to_render_params(self, state: NodeStateSky) -> Dict[str, torch.Tensor]:
        colors = torch.cat([state.sh_dc[:, None, :], state.sh_rest], dim=1)
        return {
            "means_r": state.means,
            "scales_log_r": state.scales_log,
            "scales_r": torch.exp(state.scales_log),
            "quats_r": state.quats,
            "opacity_logit_r": state.opacity_logit,
            "opacities_r": torch.sigmoid(state.opacity_logit).squeeze(-1),
            "sh_dc_r": state.sh_dc,
            "sh_rest_r": state.sh_rest,
            "colors_r": colors,
        }

    @staticmethod
    def render_params_to_gaussians(render_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "means": render_params["means_r"],
            "quats": render_params["quats_r"],
            "scales": render_params["scales_r"],
            "opacities": render_params["opacities_r"],
            "colors": render_params["colors_r"],
        }

    def render_params_from_offsets(
        self,
        state: NodeStateSky,
        *,
        offset_scales: torch.Tensor,
        offset_opacity: torch.Tensor,
        offset_sh: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_sh = _num_sh_bases(self.sh_degree)
        sh_rest_offset = offset_sh[:, 3:].reshape(state.means.shape[0], num_sh - 1, 3)
        scales_log_r = state.scales_log + self.eta_scales * offset_scales
        opacity_logit_r = state.opacity_logit + self.eta_opacity * offset_opacity
        sh_dc_r = state.sh_dc + self.eta_sh_dc * offset_sh[:, :3]
        sh_rest_r = state.sh_rest + self.eta_sh_rest * sh_rest_offset
        colors_r = torch.cat([sh_dc_r[:, None, :], sh_rest_r], dim=1)
        return {
            "means_r": state.means,
            "scales_log_r": scales_log_r,
            "scales_r": torch.exp(scales_log_r),
            "quats_r": state.quats,
            "opacity_logit_r": opacity_logit_r,
            "opacities_r": torch.sigmoid(opacity_logit_r).squeeze(-1),
            "sh_dc_r": sh_dc_r,
            "sh_rest_r": sh_rest_r,
            "colors_r": colors_r,
        }

    def render_sky_single_view(
        self,
        render_params: Dict[str, torch.Tensor],
        view: Any,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = render_params["means_r"]
        dtype = means.dtype
        device = means.device
        render, alpha, _ = self.renderer(
            means=means,
            quats=render_params["quats_r"],
            scales=render_params["scales_r"],
            opacities=render_params["opacities_r"],
            colors=render_params["colors_r"],
            viewmats=rotation_only_viewmat_from_view(view).to(device=device, dtype=dtype),
            Ks=resolve_view_intrinsics(view, device=device, dtype=dtype),
            width=int(width),
            height=int(height),
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
        alpha_map = alpha.squeeze(0)
        if alpha_map.dim() == 2:
            alpha_map = alpha_map.unsqueeze(-1)
        return rgb, alpha_map

    def render_views(
        self,
        render_params: Dict[str, torch.Tensor],
        views: List[Any],
        image_refs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(views) != len(image_refs):
            raise ValueError(f"views/image_refs length mismatch: {len(views)} vs {len(image_refs)}")
        rgbs: List[torch.Tensor] = []
        alphas: List[torch.Tensor] = []
        ref_hw: Optional[Tuple[int, int]] = None
        for i, (view, img) in enumerate(zip(views, image_refs)):
            hw = tuple(ensure_hwc3(img, name=f"image_refs[{i}]").shape[:2])
            if ref_hw is None:
                ref_hw = hw
            elif hw != ref_hw:
                raise ValueError(f"SkyBranchV0 P0 requires same source/target H/W within render call, got {hw} vs {ref_hw}.")
            rgb, alpha = self.render_sky_single_view(render_params, view, int(hw[0]), int(hw[1]))
            rgbs.append(rgb)
            alphas.append(ensure_hw1(alpha, name=f"sky_alpha[{i}]"))
        return torch.stack(rgbs, dim=0), torch.stack(alphas, dim=0)

    def build_source_masks(self, batch: Dict[str, Any], source_images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        masks_raw = batch.get("source_sky_masks")
        if masks_raw is None:
            masks_raw = batch.get("source_sky_mask")
        if masks_raw is None:
            raise ValueError("SkyBranchV0 requires source_sky_masks/source_sky_mask.")
        masks = [squeeze_mask(m.to(self.device), name=f"source_sky_masks[{i}]") for i, m in enumerate(masks_raw)]
        sky_mask = torch.stack(masks, dim=0).float().clamp(0.0, 1.0)
        valid = torch.ones_like(sky_mask, dtype=torch.bool)
        ego_raw = batch.get("source_egocar_masks")
        if ego_raw is None:
            ego_raw = batch.get("source_egocar_mask")
        if ego_raw is not None:
            if len(ego_raw) != len(source_images):
                raise ValueError("source_egocar_masks length must match source_images.")
            ego = torch.stack([squeeze_mask(m.to(self.device), name=f"source_egocar_masks[{i}]") for i, m in enumerate(ego_raw)], dim=0)
            valid = valid & (ego <= 0.5)
        return sky_mask, valid

    def build_lifting_mask(self, sky_mask_vhw: torch.Tensor, valid_vhw: torch.Tensor) -> torch.Tensor:
        sky_core, _, _ = build_sky_regions(sky_mask_vhw, erode_kernel=self.sky_core_erode_kernel)
        sky_for_lift = sky_core if self.use_sky_core_mask else sky_mask_vhw
        raw = (valid_vhw.bool() & (sky_for_lift > 0.5)).bool()
        fallback = (valid_vhw.bool() & (sky_mask_vhw > 0.5)).bool()
        flat_raw = raw.reshape(raw.shape[0], -1)
        flat_fallback = fallback.reshape(fallback.shape[0], -1)
        use_fallback = flat_raw.sum(dim=1) == 0
        if bool(use_fallback.any().item()):
            raw = raw.clone()
            raw[use_fallback] = flat_fallback[use_fallback].reshape(raw[use_fallback].shape)
        if raw.dim() != 3 or raw.dtype != torch.bool:
            raise ValueError("source_pair_valid_mask must be [V,H,W] bool.")
        return raw

    def lift_features_to_sky_nodes(
        self,
        *,
        render_params: Dict[str, torch.Tensor],
        source_views: List[Any],
        feat2d: torch.Tensor,
        source_pair_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source_pair_valid_mask.dim() != 3 or source_pair_valid_mask.dtype != torch.bool:
            raise ValueError(f"source_pair_valid_mask must be [V,H,W] bool, got {tuple(source_pair_valid_mask.shape)} {source_pair_valid_mask.dtype}")
        height, width = int(source_pair_valid_mask.shape[1]), int(source_pair_valid_mask.shape[2])
        kwargs = dict(
            gaussians=self.render_params_to_gaussians(render_params),
            cameras=source_views,
            features_2d=feat2d,
            height=height,
            width=width,
            num_gaussians=int(render_params["means_r"].shape[0]),
            backprojector=self.feature_backprojector,
            source_pair_valid_mask=source_pair_valid_mask,
            return_accumulated_weights=True,
        )
        if hasattr(self.alpha_t_extractor, "render_and_backproject_streaming_fused_multi_camera"):
            return self.alpha_t_extractor.render_and_backproject_streaming_fused_multi_camera(**kwargs)
        return self.alpha_t_extractor.render_and_backproject_streaming(**kwargs)

    def update_sky_state(
        self,
        state: NodeStateSky,
        node_feat: torch.Tensor,
        support: torch.Tensor,
        h_old: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        direction = torch.nn.functional.normalize(state.means - self.sky_center_local.to(state.means.device), dim=-1)
        param_vec = torch.cat([direction, state.scales_log, state.opacity_logit, state.sh_dc], dim=-1)
        x = torch.cat([node_feat, self.param_embed(param_vec)], dim=-1)
        h_candidate = self.gru(x, h_old)
        update_mask = (support.reshape(-1) > self.support_min).to(device=state.means.device)
        gate = update_mask.to(dtype=h_old.dtype).unsqueeze(-1)
        h_new = h_old * (1.0 - gate) + h_candidate * gate
        offset_scales = self.scale_head(h_new) * gate
        offset_opacity = self.opacity_head(h_new) * gate
        offset_sh = self.sh_head(h_new) * gate
        render_params = self.render_params_from_offsets(
            state,
            offset_scales=offset_scales,
            offset_opacity=offset_opacity,
            offset_sh=offset_sh,
        )
        offsets = {
            "offset_scales": offset_scales,
            "offset_opacity": offset_opacity,
            "offset_sh": offset_sh,
            "update_mask": update_mask,
        }
        return render_params, h_new, offsets

    def forward_scene_batch(
        self,
        batch: Dict[str, Any],
        scene_pack: SceneRenderPack,
        *,
        writeback: bool = False,
    ) -> SkyBranchForwardOutput:
        key = self.batch_key(batch)
        state = self.get_or_init_node_state(batch)
        h_old = self.get_or_init_hidden(key, state)
        source_views = list(batch.get("source_views") or [])
        source_images = [ensure_hwc3(x.to(self.device), name=f"source_images[{i}]") for i, x in enumerate(batch.get("source_images") or [])]
        targets = list(batch.get("targets") or [])
        target_views = [t["view"] for t in targets]
        target_images = [ensure_hwc3(t["gt_image"].to(self.device), name=f"targets[{i}].gt_image") for i, t in enumerate(targets)]
        target_sky_mask = torch.stack([squeeze_mask(t["sky_mask"].to(self.device), name=f"targets[{i}].sky_mask") for i, t in enumerate(targets)], dim=0)
        target_valid = []
        for i, t in enumerate(targets):
            h, w = target_images[i].shape[:2]
            valid = torch.ones(h, w, dtype=torch.float32, device=self.device)
            ego = t.get("egocar_mask")
            if ego is not None:
                valid = valid * (1.0 - squeeze_mask(ego.to(self.device), name=f"targets[{i}].egocar_mask")).clamp(0.0, 1.0)
            target_valid.append(valid)
        target_valid_mask = torch.stack(target_valid, dim=0)

        current_params = self.state_to_render_params(state)
        sky_src_rgb, _ = self.render_views(current_params, source_views, source_images)
        source_sky_mask, source_valid = self.build_source_masks(batch, source_images)
        sky_feat2d = self.feature_extractor(
            source_rgb=stack_hwc3(source_images, name="source_images"),
            current_sky_rgb=sky_src_rgb,
            sky_mask=source_sky_mask,
            scene_alpha=scene_pack.source_alpha.detach().to(self.device),
        )
        source_pair_valid_mask = self.build_lifting_mask(source_sky_mask, source_valid)
        node_feat, support = self.lift_features_to_sky_nodes(
            render_params=current_params,
            source_views=source_views,
            feat2d=sky_feat2d,
            source_pair_valid_mask=source_pair_valid_mask,
        )
        render_params_sky, h_new, offsets = self.update_sky_state(state, node_feat, support, h_old)
        sky_tgt_rgb, sky_tgt_alpha = self.render_views(render_params_sky, target_views, target_images)
        scene_rgb = scene_pack.target_rgb.detach().to(self.device)
        scene_alpha = scene_pack.target_alpha.detach().to(self.device)
        comp_rgb = scene_rgb + (1.0 - scene_alpha.clamp(0.0, 1.0)) * sky_tgt_rgb
        gt_rgb = stack_hwc3(target_images, name="target_images").to(self.device)
        loss_cfg = get_cfg(self.config, "loss", {}) or {}
        if get_cfg(loss_cfg, "sky_core_erode_kernel", None) is None:
            try:
                loss_cfg.sky_core_erode_kernel = self.sky_core_erode_kernel
            except Exception:
                pass
        loss, logs = skybranch_loss(
            comp_rgb=comp_rgb,
            sky_rgb=sky_tgt_rgb,
            sky_alpha=sky_tgt_alpha,
            gt_rgb=gt_rgb,
            sky_mask=target_sky_mask,
            valid_mask=target_valid_mask,
            scene_alpha=scene_alpha,
            cfg=loss_cfg,
        )
        logs = dict(logs)
        update_mask = offsets["update_mask"].detach()
        logs["sky_support_mean"] = support.detach().float().mean()
        logs["sky_support_ratio"] = (support.detach().reshape(-1) > self.support_min).float().mean()
        logs["sky_updated_node_ratio"] = update_mask.float().mean()
        logs["sky_offset_scale_mean"] = offsets["offset_scales"].detach().abs().mean()
        logs["sky_offset_scale_max"] = offsets["offset_scales"].detach().abs().max()
        logs["sky_offset_opacity_mean"] = offsets["offset_opacity"].detach().abs().mean()
        logs["sky_offset_opacity_max"] = offsets["offset_opacity"].detach().abs().max()
        logs["sky_offset_sh_mean"] = offsets["offset_sh"].detach().abs().mean()
        logs["sky_offset_sh_max"] = offsets["offset_sh"].detach().abs().max()
        out = SkyBranchForwardOutput(
            loss=loss,
            logs=logs,
            render_params_sky=render_params_sky,
            node_state_sky=state,
            h_new_sky=h_new,
            cache_key=key,
            source_pair_valid_mask=source_pair_valid_mask,
            support=support,
            sky_rgb=sky_tgt_rgb,
            sky_alpha=sky_tgt_alpha,
            comp_rgb=comp_rgb,
        )
        if writeback:
            self.commit_forward_output(out)
        return out

    def commit_forward_output(self, out: SkyBranchForwardOutput) -> None:
        with torch.no_grad():
            state = out.node_state_sky
            rp = out.render_params_sky
            state.scales_log.copy_(rp["scales_log_r"].detach())
            state.opacity_logit.copy_(rp["opacity_logit_r"].detach())
            state.sh_dc.copy_(rp["sh_dc_r"].detach())
            state.sh_rest.copy_(rp["sh_rest_r"].detach())
            self.h_cache_sky[out.cache_key] = out.h_new_sky.detach()

    def reset_runtime_state(self) -> None:
        self.node_states_sky.clear()
        self.h_cache_sky.clear()

    @staticmethod
    def _state_to_cpu_dict(state: NodeStateSky) -> Dict[str, torch.Tensor]:
        return {
            "means": state.means.detach().cpu(),
            "scales_log": state.scales_log.detach().cpu(),
            "quats": state.quats.detach().cpu(),
            "opacity_logit": state.opacity_logit.detach().cpu(),
            "sh_dc": state.sh_dc.detach().cpu(),
            "sh_rest": state.sh_rest.detach().cpu(),
        }

    def runtime_state_dict(self) -> Dict[str, Any]:
        return {
            "node_states_sky": {
                f"scene_{k[0]}_segment_{k[1]}": self._state_to_cpu_dict(v)
                for k, v in self.node_states_sky.items()
            },
            "h_cache_sky": {
                f"scene_{k[0]}_segment_{k[1]}": v.detach().cpu()
                for k, v in self.h_cache_sky.items()
            },
        }

    @staticmethod
    def _parse_runtime_key(key: Any) -> Tuple[int, int]:
        if isinstance(key, tuple) and len(key) == 2:
            return int(key[0]), int(key[1])
        text = str(key)
        prefix_scene = "scene_"
        mid = "_segment_"
        if not text.startswith(prefix_scene) or mid not in text:
            raise ValueError(f"Invalid SkyBranch runtime key: {key!r}")
        scene_part, segment_part = text[len(prefix_scene):].split(mid, 1)
        return int(scene_part), int(segment_part)

    def load_runtime_state_dict(self, payload: Dict[str, Any]) -> None:
        self.reset_runtime_state()
        node_payload = payload.get("node_states_sky", {}) or {}
        hidden_payload = payload.get("h_cache_sky", {}) or {}
        for key_raw, state_raw in node_payload.items():
            key = self._parse_runtime_key(key_raw)
            state = NodeStateSky(
                means=state_raw["means"].to(self.device),
                scales_log=state_raw["scales_log"].to(self.device),
                quats=state_raw["quats"].to(self.device),
                opacity_logit=state_raw["opacity_logit"].to(self.device),
                sh_dc=state_raw["sh_dc"].to(self.device),
                sh_rest=state_raw["sh_rest"].to(self.device),
            )
            self.node_states_sky[key] = state
        for key_raw, h in hidden_payload.items():
            self.h_cache_sky[self._parse_runtime_key(key_raw)] = h.to(self.device)
