"""
AlphaTWeightExtractor computes sparse alpha-transmittance weights for 2D backprojection.

The extractor renders Gaussians in packed mode to obtain intersection metadata and then
calls `rasterize_to_indices_in_range` to recover `(gaussian_id, pixel_id, weight)` tuples.
Weights are detached by design to avoid coupling gradients through the rasterizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Render each source view in packed mode to collect metadata for weight extraction.
        """
        meta_list: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for cam in cameras:
                viewmat = _get_viewmat(cam.camtoworlds if hasattr(cam, "camtoworlds") else cam["camtoworlds"])
                k_mat = self._resolve_intrinsics(cam)
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
                meta_list.append(meta)
        return meta_list

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
            try:
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
                    isect_offsets=meta["isect_offsets"],
                    flatten_ids=meta["flatten_ids"],
                    return_weights=True,
                )
            except ValueError:
                gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
                    range_start=0,
                    range_end=int(1e9),
                    transmittances=transmittances,
                    means2d=meta["means2d"],
                    conics=meta["conics"],
                    opacities=meta["opacities"],
                    image_width=width,
                    image_height=height,
                    tile_size=int(meta.get("tile_size", 16)),
                    isect_offsets=meta["isect_offsets"],
                    flatten_ids=meta["flatten_ids"],
                    return_weights=False,
                )
                weights = torch.zeros_like(gaussian_ids, dtype=transmittances.dtype)

            weight_info.append(
                {
                    "gaussian_ids": gaussian_ids.to(device),
                    "pixel_ids": pixel_ids.to(device),
                    "weights": weights.to(device),
                }
            )
        return weight_info
