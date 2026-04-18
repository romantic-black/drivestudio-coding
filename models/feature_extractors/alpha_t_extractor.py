"""
AlphaTWeightExtractor computes sparse alpha-transmittance weights for 2D backprojection.

The extractor renders Gaussians in packed mode to obtain intersection metadata and then
calls `rasterize_to_indices_in_range` to recover `(gaussian_id, pixel_id, weight)` tuples.
Weights are detached by design to avoid coupling gradients through the rasterizer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

try:
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
except Exception:  # pragma: no cover - fall back when gsplat is not available
    rasterize_to_indices_in_range = None

if TYPE_CHECKING:
    from models.feature_extractors.feature_2d_backprojector import FeatureBackprojector


def _get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert camera-to-world to gsplat-style view matrix (no axis flip).
    """
    if camera_to_world.dim() == 2:
        camera_to_world = camera_to_world.unsqueeze(0)
    viewmat = torch.linalg.inv(camera_to_world)
    if viewmat.dim() == 2:
        viewmat = viewmat.unsqueeze(0)
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
            gaussian_ids_local, pixel_ids, _, weights = rasterize_to_indices_in_range(
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
            if "gaussian_ids" not in meta:
                raise ValueError("Packed render meta missing gaussian_ids; cannot remap local ids to global ids.")
            packed_to_global = meta["gaussian_ids"]
            if packed_to_global is None:
                raise ValueError("Packed render meta gaussian_ids is None; cannot remap local ids to global ids.")
            if gaussian_ids_local.numel() > 0:
                local_min = int(gaussian_ids_local.min().item())
                local_max = int(gaussian_ids_local.max().item())
                if local_min < 0 or local_max >= int(packed_to_global.numel()):
                    raise ValueError(
                        f"Local gaussian id out of range: [{local_min}, {local_max}] vs mapping size {packed_to_global.numel()}."
                    )
                gaussian_ids = packed_to_global[gaussian_ids_local.long()].long()
            else:
                gaussian_ids = gaussian_ids_local.long()

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
        return_acc: bool = False,
        viewmats_override: Optional[torch.Tensor] = None,
        return_debug_stats: bool = False,
    ) -> Union[
        List[torch.Tensor],
        Tuple[List[torch.Tensor], List[torch.Tensor]],
        Tuple[List[torch.Tensor], Dict[str, float]],
        Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, float]],
    ]:
        """
        First render pass: collect RGB only and release meta immediately.
        """
        if not cameras:
            return []

        # Batched multi-view render when possible (one gsplat call, packed=False).
        t_start = time.perf_counter()
        rendered_rgbs: List[torch.Tensor] = []
        rendered_accs: List[torch.Tensor] = []
        with torch.no_grad():
            if viewmats_override is not None:
                if viewmats_override.dim() == 2:
                    viewmats_override = viewmats_override.unsqueeze(0)
                if int(viewmats_override.shape[0]) != int(len(cameras)):
                    raise ValueError(
                        "viewmats_override first dim must match len(cameras), "
                        f"got {viewmats_override.shape[0]} vs {len(cameras)}."
                    )
                viewmats_list = [viewmats_override]
            else:
                cam0 = cameras[0]
                cam0_ctw = cam0.camtoworlds if hasattr(cam0, "camtoworlds") else cam0["camtoworlds"]
                viewmat0 = _get_viewmat(cam0_ctw)
                viewmats_list = [viewmat0]
            k_mat0 = self._resolve_intrinsics(cameras[0])

            Ks_list = [k_mat0]
            for cam in cameras[1:]:
                k_mat = self._resolve_intrinsics(cam)
                if viewmats_override is None:
                    cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
                    viewmat = _get_viewmat(cam_ctw)
                    viewmats_list.append(viewmat)
                Ks_list.append(k_mat)

            viewmats = torch.cat(viewmats_list, dim=0)
            Ks = torch.cat(Ks_list, dim=0)

            render_colors, render_alphas, _ = self.renderer(
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
                if return_acc:
                    acc = render_alphas[c]
                    rendered_accs.append(acc.detach())

        if return_debug_stats:
            stats = {
                "render_rgb_only_ms": float((time.perf_counter() - t_start) * 1000.0),
                "num_views": int(len(cameras)),
                "num_gaussians": int(gaussians["means"].shape[0]),
            }
            if return_acc:
                return rendered_rgbs, rendered_accs, stats
            return rendered_rgbs, stats
        if return_acc:
            return rendered_rgbs, rendered_accs
        return rendered_rgbs

    def extract_single_weight(
        self,
        meta: Dict[str, torch.Tensor],
        height: int,
        width: int,
        pair_valid_mask: Optional[torch.Tensor] = None,
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

        try:
            gaussian_ids_local, pixel_ids, _, weights = rasterize_to_indices_in_range(
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
                pair_valid_mask=pair_valid_mask,
            )
        except TypeError:
            gaussian_ids_local, pixel_ids, _, weights = rasterize_to_indices_in_range(
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
            if pair_valid_mask is not None and pixel_ids.numel() > 0:
                valid_flat = pair_valid_mask.to(device=pixel_ids.device).reshape(-1)
                if valid_flat.dtype != torch.bool:
                    valid_flat = valid_flat > 0.5
                keep = valid_flat[pixel_ids.long()]
                gaussian_ids_local = gaussian_ids_local[keep]
                pixel_ids = pixel_ids[keep]
                weights = weights[keep]
        if "gaussian_ids" not in meta:
            raise ValueError("Packed render meta missing gaussian_ids; cannot remap local ids to global ids.")
        packed_to_global = meta["gaussian_ids"]
        if packed_to_global is None:
            raise ValueError("Packed render meta gaussian_ids is None; cannot remap local ids to global ids.")
        if not torch.is_tensor(packed_to_global):
            raise TypeError("Packed render meta gaussian_ids must be a tensor.")
        if packed_to_global.dtype not in (
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.int8,
        ):
            raise TypeError(f"Packed render meta gaussian_ids must be integer tensor, got {packed_to_global.dtype}.")
        if packed_to_global.device != gaussian_ids_local.device:
            raise ValueError("Packed gaussian_ids device mismatch with rasterized local ids.")
        if gaussian_ids_local.numel() > 0:
            local_min = int(gaussian_ids_local.min().item())
            local_max = int(gaussian_ids_local.max().item())
            if local_min < 0 or local_max >= int(packed_to_global.numel()):
                raise ValueError(
                    f"Local gaussian id out of range: [{local_min}, {local_max}] vs mapping size {packed_to_global.numel()}."
                )
            gaussian_ids = packed_to_global[gaussian_ids_local.long()].long()
            global_min = int(gaussian_ids.min().item())
            if global_min < 0:
                raise ValueError(f"Remapped global gaussian id is negative: min={global_min}.")
        else:
            gaussian_ids = gaussian_ids_local.long()

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
        source_pair_valid_mask: Optional[torch.Tensor] = None,
        return_accumulated_weights: bool = False,
        return_debug_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        """
        Second render pass: stream per-view weights and accumulate backprojection.

        When return_accumulated_weights is True, also returns per-Gaussian **support**
        accumulated weights over views, shape [N]. Support weights are defined as the
        *unfiltered* sum of (T * alpha) over gaussian-pixel pairs, and therefore do NOT
        depend on FeatureBackprojector.weight_threshold (which may be used as a feature
        aggregation optimization).
        """
        t_total_start = time.perf_counter()
        device = features_2d.device
        channels = features_2d.shape[-1]
        eps = getattr(backprojector, "eps", 1e-8)

        accumulated_feat = torch.zeros(num_gaussians, channels, device=device)
        accumulated_weight_feature = torch.zeros(num_gaussians, device=device)
        accumulated_weight_support = (
            torch.zeros(num_gaussians, device=device) if return_accumulated_weights else None
        )

        stats = {
            "render_packed_total_ms": 0.0,
            "extract_weight_total_ms": 0.0,
            "backproject_total_ms": 0.0,
            "pairs_total": 0,
            "pairs_after_threshold": 0,
            "num_views": int(len(cameras)),
            "num_gaussians": int(num_gaussians),
            "pairs_after_mask": 0,
            "masked_pixel_count": 0,
            "valid_pixel_count": 0,
            "source_pair_valid_ratio": 1.0,
        }
        pair_valid_masks: Optional[List[torch.Tensor]] = None
        if source_pair_valid_mask is not None:
            if source_pair_valid_mask.dim() != 3:
                raise ValueError(
                    "source_pair_valid_mask must have shape [V, H, W], "
                    f"got {tuple(source_pair_valid_mask.shape)}."
                )
            if int(source_pair_valid_mask.shape[0]) != int(len(cameras)):
                raise ValueError(
                    f"source_pair_valid_mask.shape[0] ({source_pair_valid_mask.shape[0]}) "
                    f"must equal len(cameras) ({len(cameras)})."
                )
            if int(source_pair_valid_mask.shape[1]) != int(height) or int(source_pair_valid_mask.shape[2]) != int(width):
                raise ValueError(
                    "source_pair_valid_mask spatial shape mismatch with source render size: "
                    f"expected ({height}, {width}), got ({source_pair_valid_mask.shape[1]}, {source_pair_valid_mask.shape[2]})."
                )
            m = source_pair_valid_mask.to(device=device)
            if m.dtype != torch.bool:
                m = m > 0.5
            pair_valid_masks = [m[i].contiguous() for i in range(int(m.shape[0]))]
            valid_pixel_count = int(m.sum().item())
            total_pixel_count = int(m.numel())
            masked_pixel_count = int(total_pixel_count - valid_pixel_count)
            stats["valid_pixel_count"] = valid_pixel_count
            stats["masked_pixel_count"] = masked_pixel_count
            stats["source_pair_valid_ratio"] = float(valid_pixel_count / max(total_pixel_count, 1))

        for i, cam in enumerate(cameras):
            cam_ctw = cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"]
            viewmat = _get_viewmat(cam_ctw)
            k_mat = self._resolve_intrinsics(cam)
            t_render = time.perf_counter()
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
            stats["render_packed_total_ms"] += float((time.perf_counter() - t_render) * 1000.0)

            t_extract = time.perf_counter()
            pair_mask_i = pair_valid_masks[i] if pair_valid_masks is not None else None
            weight_info = self.extract_single_weight(meta, height, width, pair_valid_mask=None)
            stats["extract_weight_total_ms"] += float((time.perf_counter() - t_extract) * 1000.0)
            pairs_total_now = int(weight_info["gaussian_ids"].numel())
            stats["pairs_total"] += pairs_total_now
            if pair_mask_i is not None and pairs_total_now > 0:
                valid_flat = pair_mask_i.reshape(-1)
                keep = valid_flat[weight_info["pixel_ids"].long()]
                weight_info = {
                    "gaussian_ids": weight_info["gaussian_ids"][keep],
                    "pixel_ids": weight_info["pixel_ids"][keep],
                    "weights": weight_info["weights"][keep],
                }
            pairs_after_mask_now = int(weight_info["gaussian_ids"].numel())
            stats["pairs_after_mask"] += pairs_after_mask_now
            del meta

            t_bp = time.perf_counter()
            if return_accumulated_weights:
                feat_sum, weight_sum_feature, weight_sum_support, bp_stats = backprojector.backproject_single_view(
                    features_2d[i],
                    weight_info,
                    height,
                    width,
                    num_gaussians,
                    return_support_weight=True,
                    return_debug_stats=True,
                )
            else:
                feat_sum, weight_sum_feature, bp_stats = backprojector.backproject_single_view(
                    features_2d[i],
                    weight_info,
                    height,
                    width,
                    num_gaussians,
                    return_debug_stats=True,
                )
            stats["backproject_total_ms"] += float((time.perf_counter() - t_bp) * 1000.0)
            stats["pairs_after_threshold"] += int(bp_stats.get("pairs_after_threshold", 0))
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
        stats["streaming_total_ms"] = float((time.perf_counter() - t_total_start) * 1000.0)
        if return_accumulated_weights:
            if accumulated_weight_support is None:
                raise RuntimeError("Internal error: accumulated_weight_support is None.")
            if return_debug_stats:
                return feat_out, accumulated_weight_support, stats
            return feat_out, accumulated_weight_support
        if return_debug_stats:
            return feat_out, stats
        return feat_out
