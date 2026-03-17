"""
Sky model as a learnable cubemap sampled with nvdiffrast dr.texture.

World-space viewdirs (from get_rays) are transformed with to_opengl to match
OpenGL cube convention; no reuse of models.modules.EnvLight class.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import nvdiffrast.torch as dr
except ImportError:
    dr = None


class SkyCubemap(nn.Module):
    """
    Learnable 6-face cubemap for sky rendering. Uses dr.texture with boundary_mode='cube'.
    Input viewdirs are in segment world frame (x=right, -y=up, z=front); transformed
    by to_opengl to match nvdiffrast/OpenGL convention.
    """

    # Same as EnvLight: world (x,y,z) -> (x, z, -y) for cube sampling
    TO_OPENGL = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

    def __init__(
        self,
        resolution: int = 1024,
        init_value: float = 0.5,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.init_value = init_value
        self._device = device or torch.device("cuda")
        self.register_buffer(
            "to_opengl",
            torch.tensor(
                self.TO_OPENGL,
                dtype=torch.float32,
                device=self._device,
            ),
        )
        self.base = nn.Parameter(
            init_value * torch.ones(6, resolution, resolution, 3, device=self._device),
        )

    def forward(self, image_infos: dict) -> torch.Tensor:
        viewdirs = image_infos["viewdirs"]
        orig_shape = tuple(viewdirs.shape)
        if viewdirs.device != self.to_opengl.device:
            viewdirs = viewdirs.to(self.to_opengl.device)
        if viewdirs.dim() == 3:
            # (H, W, 3) -> (1, H, W, 3)
            viewdirs_bhwc = viewdirs.unsqueeze(0)
        elif viewdirs.dim() == 4:
            # (B, H, W, 3)
            viewdirs_bhwc = viewdirs
        else:
            raise ValueError(f"viewdirs must have shape (H,W,3) or (B,H,W,3), got {orig_shape}")
        if viewdirs_bhwc.shape[-1] != 3:
            raise ValueError(f"viewdirs last dim must be 3, got {orig_shape}")
        # world -> cube direction: (x, y, z) -> (x, z, -y)
        cube_dir = viewdirs_bhwc.reshape(-1, 3) @ self.to_opengl.T
        cube_dir = cube_dir.reshape(*viewdirs_bhwc.shape).contiguous()
        if dr is None:
            raise ImportError("nvdiffrast is required for SkyCubemap. Install nvdiffrast.")
        light = dr.texture(
            self.base[None, ...],
            cube_dir,
            filter_mode="linear",
            boundary_mode="cube",
        )
        light = torch.sigmoid(light)
        # Return shape matches input prefix.
        if len(orig_shape) == 3:
            return light.squeeze(0)
        return light


__all__ = ["SkyCubemap"]
