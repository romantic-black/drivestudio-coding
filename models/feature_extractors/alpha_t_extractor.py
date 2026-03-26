"""
AlphaTWeightExtractor computes sparse alpha-transmittance weights for 2D backprojection.

The extractor renders Gaussians in packed mode to obtain intersection metadata and then
calls `rasterize_to_indices_in_range` to recover `(gaussian_id, pixel_id, weight)` tuples.
Weights are detached by design to avoid coupling gradients through the rasterizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
except Exception:  # pragma: no cover - fall back when gsplat is not available
    rasterize_to_indices_in_range = None


def _get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert camera-to-world to gsplat-style view matrix.
    """
    if camera_to_world.dim() == 2:
        camera_to_world = camera_to_world.unsqueeze(0)
    r = camera_to_world[:, :3, :3]
    t = camera_to_world[:, :3, 3:4]
    r = r * torch.tensor([[[1, -1, -1]]], device=r.device, dtype=r.dtype)
    r_inv = r.transpose(1, 2)
    t_inv = -torch.bmm(r_inv, t)
    viewmat = torch.zeros(r.shape[0], 4, 4, device=r.device, dtype=r.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = r_inv
    viewmat[:, :3, 3:4] = t_inv
    return viewmat


class AlphaTWeightExtractor:
    """
    Render Gaussians to obtain alpha*T weights per pixel.
    """

    def __init__(self, renderer, sh_degree: int, tile_size: int = 16) -> None:
        self.renderer = renderer
        self.sh_degree = sh_degree
        self.tile_size = tile_size
        if rasterize_to_indices_in_range is None:
            raise ImportError("gsplat with rasterize_to_indices_in_range is required for AlphaTWeightExtractor")

    @staticmethod
    def _resolve_intrinsics(view) -> torch.Tensor:
        if hasattr(view, "Ks"):
            k_mat = view.Ks
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=view.camtoworlds.device).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)
        return k_mat

    def render_meta(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        height: int,
        width: int,
        return_rgb: bool = False,
    ) -> tuple[List[Dict[str, torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Render each source view in packed mode to collect metadata for weight extraction.
        
        Args:
            gaussians: Gaussian parameters dictionary
            cameras: List of camera views
            height: Image height
            width: Image width
            return_rgb: If True, also return rendered RGB images for CNN guidance
            
        Returns:
            meta_list: List of metadata dictionaries
            rendered_rgbs: Optional list of rendered RGB images [H, W, 3] (if return_rgb=True)
        """
        meta_list: List[Dict[str, torch.Tensor]] = []
        rendered_rgbs: List[torch.Tensor] = [] if return_rgb else None
        with torch.no_grad():
            for cam in cameras:
                cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
                viewmat = _get_viewmat(cam_ctw)
                k_mat = self._resolve_intrinsics(cam)
                render_colors, _, meta = self.renderer(
                    means=gaussians["means"],
                    quats=gaussians["quats"],
                    scales=gaussians["scales"],
                    opacities=gaussians["opacities"],
                    colors=gaussians["colors"],
                    viewmats=viewmat,
                    Ks=k_mat,
                    width=width,
                    height=height,
                    tile_size=self.tile_size,
                    packed=True,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=self.sh_degree,
                    sparse_grad=False,
                    absgrad=True,
                    rasterize_mode="classic",
                )
                meta_list.append(meta)
                
                if return_rgb:
                    # render_colors shape: [..., C, H, W, 3] for packed mode
                    # Extract RGB image: [H, W, 3]
                    if render_colors.dim() == 5:
                        # [1, 1, H, W, 3] -> [H, W, 3]
                        rgb = render_colors.squeeze(0).squeeze(0)
                    elif render_colors.dim() == 4:
                        # [1, H, W, 3] -> [H, W, 3]
                        rgb = render_colors.squeeze(0)
                    else:
                        # [H, W, 3]
                        rgb = render_colors
                    # Clamp to [0, 1] and detach
                    rgb = torch.clamp(rgb, 0.0, 1.0).detach()
                    rendered_rgbs.append(rgb)
        
        if return_rgb:
            return meta_list, rendered_rgbs
        return meta_list, None

    def extract_weights(
        self,
        meta_list: List[Dict[str, torch.Tensor]],
        height: int,
        width: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert meta dictionaries into sparse weight tuples for each view.
        """
        weight_info: List[Dict[str, torch.Tensor]] = []
        for meta in meta_list:
            if meta is None:
                weight_info.append(
                    {
                        "gaussian_ids": torch.empty(0, dtype=torch.long),
                        "pixel_ids": torch.empty(0, dtype=torch.long),
                        "weights": torch.empty(0),
                    }
                )
                continue
            device = meta["means2d"].device
            transmittances = torch.ones((height, width), device=device, dtype=meta["means2d"].dtype)
            # Fix isect_offsets shape: in packed mode, means2d has shape [nnz, 2], so image_dims = []
            # rasterize_to_indices_in_range expects isect_offsets.shape == image_dims + (tile_height, tile_width)
            # But meta["isect_offsets"] has shape [batch_dims..., C, tile_height, tile_width] or similar
            # We need to extract the correct slice to match image_dims
            image_dims = meta["means2d"].shape[:-2]  # Should be () for packed mode
            isect_offsets_raw = meta["isect_offsets"]
            # If isect_offsets has more dimensions than expected, remove the extra batch/camera dimensions
            # Expected shape: image_dims + (tile_height, tile_width)
            # Actual shape from renderer: [batch_dims..., C, tile_height, tile_width]
            # In packed mode with single batch/camera: [1, tile_height, tile_width] or [1, 1, tile_height, tile_width]
            if len(isect_offsets_raw.shape) > len(image_dims) + 2:
                # Remove extra leading dimensions to match image_dims
                n_dims_to_remove = len(isect_offsets_raw.shape) - len(image_dims) - 2
                for _ in range(n_dims_to_remove):
                    if isect_offsets_raw.shape[0] == 1:
                        isect_offsets_raw = isect_offsets_raw.squeeze(0)
                    else:
                        break  # Only squeeze dimensions of size 1
            isect_offsets_fixed = isect_offsets_raw
            gaussian_ids, pixel_ids, _, weights = rasterize_to_indices_in_range(
                range_start=0,
                range_end=int(1e9),
                transmittances=transmittances,
                means2d=meta["means2d"],
                conics=meta["conics"],
                opacities=meta["opacities"],
                image_width=width,
                image_height=height,
                tile_size=int(meta.get("tile_size", 16)),
                isect_offsets=isect_offsets_fixed,
                flatten_ids=meta["flatten_ids"],
                return_weights=True,
            )

            weight_info.append(
                {
                    "gaussian_ids": gaussian_ids.to(device),
                    "pixel_ids": pixel_ids.to(device),
                    "weights": weights.to(device),
                }
            )
        return weight_info

    def _extract_rgb(self, render_colors: torch.Tensor) -> torch.Tensor:
        """
        From packed renderer output, extract an RGB image tensor with shape [H, W, 3].
        """
        if render_colors.dim() == 5:
            rgb = render_colors.squeeze(0).squeeze(0)
        elif render_colors.dim() == 4:
            rgb = render_colors.squeeze(0)
        else:
            rgb = render_colors
        return torch.clamp(rgb, 0.0, 1.0).detach()

    def render_rgb_only(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        height: int,
        width: int,
    ) -> List[torch.Tensor]:
        """
        First render pass: collect RGB only and release meta immediately.
        """
        if not cameras:
            return []

        # Batched multi-view render when possible (one gsplat call, packed=False).
        rendered_rgbs: List[torch.Tensor] = []
        with torch.no_grad():
            cam0 = cameras[0]
            cam0_ctw = cam0.camtoworlds if hasattr(cam0, "camtoworlds") else cam0["camtoworlds"]
            viewmat0 = _get_viewmat(cam0_ctw)
            k_mat0 = self._resolve_intrinsics(cam0)

            viewmats_list = [viewmat0]
            Ks_list = [k_mat0]
            for cam in cameras[1:]:
                cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
                viewmat = _get_viewmat(cam_ctw)
                k_mat = self._resolve_intrinsics(cam)
                viewmats_list.append(viewmat)
                Ks_list.append(k_mat)

            viewmats = torch.cat(viewmats_list, dim=0)
            Ks = torch.cat(Ks_list, dim=0)

            render_colors, _, _ = self.renderer(
                means=gaussians["means"],
                quats=gaussians["quats"],
                scales=gaussians["scales"],
                opacities=gaussians["opacities"],
                colors=gaussians["colors"],
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                tile_size=self.tile_size,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=self.sh_degree,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode="classic",
            )

            # render_colors shape: [C, H, W, 3] or [1, C, H, W, 3]
            if render_colors.dim() == 5:
                render_colors = render_colors.squeeze(0)
            for c in range(render_colors.shape[0]):
                rgb = render_colors[c]
                rgb = torch.clamp(rgb, 0.0, 1.0).detach()
                rendered_rgbs.append(rgb)

        return rendered_rgbs

    def extract_single_weight(
        self,
        meta: Dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract sparse weight tuples from a single packed-meta output.
        """
        if meta is None:
            return {
                "gaussian_ids": torch.empty(0, dtype=torch.long),
                "pixel_ids": torch.empty(0, dtype=torch.long),
                "weights": torch.empty(0),
            }

        device = meta["means2d"].device
        transmittances = torch.ones((height, width), device=device, dtype=meta["means2d"].dtype)

        image_dims = meta["means2d"].shape[:-2]
        isect_offsets_raw = meta["isect_offsets"]
        if len(isect_offsets_raw.shape) > len(image_dims) + 2:
            n_dims_to_remove = len(isect_offsets_raw.shape) - len(image_dims) - 2
            for _ in range(n_dims_to_remove):
                if isect_offsets_raw.shape[0] == 1:
                    isect_offsets_raw = isect_offsets_raw.squeeze(0)
                else:
                    break

        gaussian_ids, pixel_ids, _, weights = rasterize_to_indices_in_range(
            range_start=0,
            range_end=int(1e9),
            transmittances=transmittances,
            means2d=meta["means2d"],
            conics=meta["conics"],
            opacities=meta["opacities"],
            image_width=width,
            image_height=height,
            tile_size=int(meta.get("tile_size", 16)),
            isect_offsets=isect_offsets_raw,
            flatten_ids=meta["flatten_ids"],
            return_weights=True,
        )

        return {
            "gaussian_ids": gaussian_ids.to(device),
            "pixel_ids": pixel_ids.to(device),
            "weights": weights.to(device),
        }

    def render_and_backproject_streaming(
        self,
        gaussians: Dict[str, torch.Tensor],
        cameras: List,
        features_2d: torch.Tensor,
        height: int,
        width: int,
        num_gaussians: int,
        backprojector: "FeatureBackprojector",
        return_accumulated_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Second render pass: stream per-view weights and accumulate backprojection.

        When return_accumulated_weights is True, also returns per-Gaussian **support**
        accumulated weights over views, shape [N]. Support weights are defined as the
        *unfiltered* sum of (T * alpha) over gaussian-pixel pairs, and therefore do NOT
        depend on FeatureBackprojector.weight_threshold (which may be used as a feature
        aggregation optimization).
        """
        device = features_2d.device
        channels = features_2d.shape[-1]
        eps = getattr(backprojector, "eps", 1e-8)

        accumulated_feat = torch.zeros(num_gaussians, channels, device=device)
        accumulated_weight_feature = torch.zeros(num_gaussians, device=device)
        accumulated_weight_support = (
            torch.zeros(num_gaussians, device=device) if return_accumulated_weights else None
        )

        for i, cam in enumerate(cameras):
            cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
            viewmat = _get_viewmat(cam_ctw)
            k_mat = self._resolve_intrinsics(cam)
            with torch.no_grad():
                _, _, meta = self.renderer(
                    means=gaussians["means"],
                    quats=gaussians["quats"],
                    scales=gaussians["scales"],
                    opacities=gaussians["opacities"],
                    colors=gaussians["colors"],
                    viewmats=viewmat,
                    Ks=k_mat,
                    width=width,
                    height=height,
                    tile_size=self.tile_size,
                    packed=True,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=self.sh_degree,
                    sparse_grad=False,
                    absgrad=True,
                    rasterize_mode="classic",
                )

                weight_info = self.extract_single_weight(meta, height, width)
                del meta

            if return_accumulated_weights:
                feat_sum, weight_sum_feature, weight_sum_support = backprojector.backproject_single_view(
                    features_2d[i],
                    weight_info,
                    height,
                    width,
                    num_gaussians,
                    return_support_weight=True,
                )
            else:
                feat_sum, weight_sum_feature = backprojector.backproject_single_view(
                    features_2d[i],
                    weight_info,
                    height,
                    width,
                    num_gaussians,
                )
            del weight_info

            accumulated_feat += feat_sum
            accumulated_weight_feature += weight_sum_feature
            if return_accumulated_weights:
                if accumulated_weight_support is None:
                    raise RuntimeError("Internal error: accumulated_weight_support is None.")
                accumulated_weight_support += weight_sum_support
                del weight_sum_support
            del feat_sum, weight_sum_feature

        feat_out = accumulated_feat / (accumulated_weight_feature.unsqueeze(-1) + eps)
        if return_accumulated_weights:
            if accumulated_weight_support is None:
                raise RuntimeError("Internal error: accumulated_weight_support is None.")
            return feat_out, accumulated_weight_support
        return feat_out
