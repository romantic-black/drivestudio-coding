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

        offsets_view = offsets[view_idx].reshape(-1).to(device)
        flatten_ids = flatten_ids.to(device)
        conics = conics.to(device).detach()
        means2d = means2d.to(device).detach()
        opacities = opacities.to(device).detach()

        tile_height = meta.get("tile_height", int(height / tile_size + 0.999))
        tile_width = meta.get("tile_width", int(width / tile_size + 0.999))
        idx_map = torch.full((height, width, self.top_k), -1, device=device, dtype=torch.int32)
        w_map = torch.zeros((height, width, self.top_k), device=device, dtype=torch.float32)
        total_tiles = offsets_view.numel()
        n_isects = flatten_ids.numel()

        for tile_idx in range(total_tiles):
            y_tile = tile_idx // tile_width
            x_tile = tile_idx % tile_width
            start = int(offsets_view[tile_idx].item())
            end = int(offsets_view[tile_idx + 1].item()) if tile_idx + 1 < total_tiles else n_isects
            if end <= start:
                continue
            gauss_ids = flatten_ids[start:end]
            if gauss_ids.numel() == 0:
                continue

            y_start = y_tile * tile_size
            x_start = x_tile * tile_size
            y_end = min((y_tile + 1) * tile_size, height)
            x_end = min((x_tile + 1) * tile_size, width)
            if y_start >= height or x_start >= width:
                continue
            grid_y = torch.arange(y_start, y_end, device=device, dtype=torch.float32).view(-1, 1)
            grid_x = torch.arange(x_start, x_end, device=device, dtype=torch.float32).view(1, -1)
            grid_y = grid_y + 0.5
            grid_x = grid_x + 0.5
            tile_h = y_end - y_start
            tile_w = x_end - x_start
            T_map = torch.ones((tile_h, tile_w), device=device, dtype=torch.float32)
            contribs = []
            contrib_ids = []

            for gid in gauss_ids:
                gid_int = int(gid.item())
                mean = means2d[view_idx, gid_int]
                conic = conics[view_idx, gid_int]
                opacity = opacities[view_idx, gid_int]
                dx = grid_x - mean[0]
                dy = grid_y - mean[1]
                sigma_term = conic[0] * dx * dx + conic[1] * dx * dy + conic[2] * dy * dy
                weight = torch.exp(-0.5 * sigma_term) * opacity
                contrib = weight * T_map
                T_map = T_map * (1.0 - weight)
                contribs.append(contrib.reshape(-1))
                contrib_ids.append(torch.full_like(contrib.reshape(-1), gid_int, dtype=torch.int32))

            if len(contribs) == 0:
                continue

            contrib_stack = torch.stack(contribs, dim=-1)
            id_stack = torch.stack(contrib_ids, dim=-1)
            k = min(self.top_k, contrib_stack.shape[-1])
            topk_vals, topk_idx = torch.topk(contrib_stack, k=k, dim=-1)
            topk_ids = torch.gather(id_stack, -1, topk_idx)
            topk_vals = topk_vals.view(tile_h, tile_w, k)
            topk_ids = topk_ids.view(tile_h, tile_w, k)
            w_map[y_start:y_end, x_start:x_end, :k] = topk_vals
            idx_map[y_start:y_end, x_start:x_end, :k] = topk_ids

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

        for view in views:
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
                    absgrad=True,
                    rasterize_mode="classic",
                )
            idx_map, w_map = self._compute_alpha_topk_for_view(
                meta=meta,
                view_idx=0,
                height=height,
                width=width,
            )
            gaussian_indices.append(idx_map)
            alpha_weights.append(w_map.detach())

        return gaussian_indices, [w.detach() for w in alpha_weights]
