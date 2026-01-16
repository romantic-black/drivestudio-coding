from typing import Callable, List, Optional, Tuple

import torch


def _get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
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
    Extract per-pixel top-K Gaussian indices and alpha*T weights from low-res rendering.
    Aligned with docs/trainers/StreetForward_2DFeat_Design.md: render once per view at
    feature resolution, gather per-pixel gaussians on GPU, compute alpha*T in the same
    front-to-back order, and return stop-grad weights for 2D feature backprojection.
    """

    def __init__(self, renderer: Callable, top_k: int = 8, device: Optional[torch.device] = None):
        self.renderer = renderer
        self.top_k = top_k
        self.device = device

    def _prepare_view(self, view, target_height: int, target_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c2w = view.camtoworlds if hasattr(view, "camtoworlds") else view["camtoworlds"]
        if hasattr(view, "Ks"):
            k_mat = view.Ks[0:1]
        elif hasattr(view, "K"):
            k_mat = view.K
        else:
            k_mat = torch.eye(3, device=c2w.device).unsqueeze(0)
        if k_mat.dim() == 2:
            k_mat = k_mat.unsqueeze(0)

        # Scale intrinsics to match the target feature resolution
        orig_h = getattr(view, "H", None) if hasattr(view, "H") else None
        orig_w = getattr(view, "W", None) if hasattr(view, "W") else None
        if orig_h is None and isinstance(view, dict):
            orig_h = view.get("H")
        if orig_w is None and isinstance(view, dict):
            orig_w = view.get("W")
        if orig_h is not None and orig_w is not None and orig_h > 0 and orig_w > 0:
            scale_h = float(target_height) / float(orig_h)
            scale_w = float(target_width) / float(orig_w)
            k_scaled = k_mat.clone()
            k_scaled[:, 0, 0] *= scale_w
            k_scaled[:, 1, 1] *= scale_h
            k_scaled[:, 0, 2] *= scale_w
            k_scaled[:, 1, 2] *= scale_h
            k_mat = k_scaled

        viewmat = _get_viewmat(c2w)
        if self.device is not None:
            viewmat = viewmat.to(self.device)
            k_mat = k_mat.to(self.device)
        return viewmat, k_mat

    def _compute_alpha_topk_for_view(
        self,
        meta: dict,
        view_idx: int,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from gsplat import rasterize_to_indices_in_range

        device = self.device or meta["means2d"].device
        flatten_ids = meta.get("flatten_ids")
        offsets = meta.get("isect_offsets")
        conics = meta.get("conics")
        means2d = meta.get("means2d")
        opacities = meta.get("opacities")
        tile_size = int(meta.get("tile_size", 16))

        if (
            flatten_ids is None
            or offsets is None
            or conics is None
            or means2d is None
            or opacities is None
        ):
            idx_map = torch.full((height, width, self.top_k), -1, device=device, dtype=torch.int32)
            w_map = torch.zeros((height, width, self.top_k), device=device, dtype=torch.float32)
            return idx_map, w_map

        means2d = means2d.to(device).detach()
        conics = conics.to(device).detach()
        opacities = opacities.to(device).detach()
        flatten_ids = flatten_ids.to(device)
        offsets_view = offsets[view_idx : view_idx + 1].to(device)
        means2d_view = means2d[view_idx : view_idx + 1]
        conics_view = conics[view_idx : view_idx + 1]
        opacities_view = opacities[view_idx : view_idx + 1]

        # Seed rasterize_to_indices_in_range with transparent rays for a full sweep.
        transmittances = torch.ones((1, height, width), device=device, dtype=means2d.dtype)

        idx_map = torch.full((height, width, self.top_k), -1, device=device, dtype=torch.int32)
        w_map = torch.zeros((height, width, self.top_k), device=device, dtype=torch.float32)

        gauss_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
            range_start=0,
            range_end=int(1e10),
            transmittances=transmittances,
            means2d=means2d_view,
            conics=conics_view,
            opacities=opacities_view,
            image_width=width,
            image_height=height,
            tile_size=tile_size,
            isect_offsets=offsets_view,
            flatten_ids=flatten_ids,
        )

        if gauss_ids.numel() == 0:
            return idx_map, w_map

        # All rays are packed front-to-back; build alpha and weights in parallel.
        pixel_x = (pixel_ids % width).to(torch.float32) + 0.5
        pixel_y = (pixel_ids // width).to(torch.float32) + 0.5
        pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)

        means_sel = means2d_view[image_ids, gauss_ids]
        conics_sel = conics_view[image_ids, gauss_ids]
        opacities_sel = opacities_view[image_ids, gauss_ids]

        deltas = pixel_coords - means_sel
        sigmas = (
            0.5 * (conics_sel[:, 0] * deltas[:, 0] ** 2 + conics_sel[:, 2] * deltas[:, 1] ** 2)
            + conics_sel[:, 1] * deltas[:, 0] * deltas[:, 1]
        )
        alphas = torch.clamp(opacities_sel * torch.exp(-sigmas), max=0.999)

        ray_indices = image_ids * (height * width) + pixel_ids
        ray_trans_start = transmittances.view(-1)[ray_indices]

        one_minus_alpha = torch.clamp_min(1.0 - alphas, 1e-6)
        log_oma = torch.log(one_minus_alpha)
        cum_log = torch.cumsum(log_oma, dim=0)
        cum_log_shifted = cum_log - log_oma

        unique_rays, counts = torch.unique_consecutive(ray_indices, return_counts=True)
        start_indices = torch.cat(
            [torch.tensor([0], device=device, dtype=counts.dtype), torch.cumsum(counts, dim=0)[:-1]]
        )
        start_prefix_vals = torch.zeros_like(start_indices, dtype=log_oma.dtype)
        if start_indices.numel() > 1:
            start_prefix_vals[1:] = cum_log[start_indices[1:] - 1]
        start_prefix_per_elem = torch.repeat_interleave(start_prefix_vals, counts)

        log_trans_before = cum_log_shifted - start_prefix_per_elem
        trans_before = torch.exp(log_trans_before) * ray_trans_start
        weights = alphas * trans_before

        num_rays = unique_rays.numel()
        max_count = int(counts.max().item())
        k = min(self.top_k, max_count)
        ray_ids_for_elem = torch.repeat_interleave(torch.arange(num_rays, device=device, dtype=torch.int64), counts)
        local_indices = torch.arange(weights.shape[0], device=device, dtype=torch.int64) - torch.repeat_interleave(
            start_indices.to(torch.int64), counts
        )

        weights_padded = torch.full((num_rays, max_count), float("-inf"), device=device, dtype=torch.float32)
        ids_padded = torch.full((num_rays, max_count), -1, device=device, dtype=torch.int32)
        weights_padded[ray_ids_for_elem, local_indices] = weights
        ids_padded[ray_ids_for_elem, local_indices] = gauss_ids.to(torch.int32)

        topk_vals, topk_idx = torch.topk(weights_padded, k=k, dim=1)
        topk_ids = torch.gather(ids_padded, 1, topk_idx)
        topk_vals = torch.where(torch.isfinite(topk_vals), topk_vals, torch.zeros_like(topk_vals))

        pixel_flat = unique_rays % (height * width)
        ys = pixel_flat // width
        xs = pixel_flat % width
        idx_map[ys, xs, :k] = topk_ids
        w_map[ys, xs, :k] = topk_vals

        return idx_map, w_map

    def extract_alpha_t_weights(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        views: List,
        height: int,
        width: int,
        sh_degree: int = 1,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        gaussian_indices: List[torch.Tensor] = []
        alpha_weights: List[torch.Tensor] = []

        for view_idx, view in enumerate(views):
            viewmat, K = self._prepare_view(
                view=view,
                target_height=height,
                target_width=width,
            )

            with torch.no_grad():
                _, _, meta = self.renderer(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    tile_size=16,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=sh_degree,
                    sparse_grad=False,
                    absgrad=False,
                    rasterize_mode="classic",
                    return_transmittances=False,
                )
            idx_map, w_map = self._compute_alpha_topk_for_view(
                meta=meta,
                view_idx=view_idx,
                height=height,
                width=width,
            )

            gaussian_indices.append(idx_map)
            alpha_weights.append(w_map.detach())

        return gaussian_indices, alpha_weights
