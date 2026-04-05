"""
Sky shell: Fibonacci sphere/hemisphere on segment AABB ground-face center.

Coordinate system is **only** the MultiSceneDataset convention (see ``docs/dataloader/MultiSceneDataset_Usage.md``):
x = left(-) / right(+); y = up(-) / down(+); z = back(-) / front(+).
Sky direction is **-Y** (negative Y). Ground / bottom of the segment AABB is the **y_max** face (larger y = more down).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

# Unit vector toward sky in MultiSceneDataset world coordinates (y increases downward).
SKY_UP_MULTISCENE: Tuple[float, float, float] = (0.0, -1.0, 0.0)


def sky_base_from_aabb(bbx_min: torch.Tensor, bbx_max: torch.Tensor) -> torch.Tensor:
    """
    Ground-face center: mid X, **y_max** (road / bottom), mid Z.

    ``segment_aabb`` uses the same axis semantics as MultiSceneDataset (y positive = down).
    """
    return torch.stack(
        [
            0.5 * (bbx_min[0] + bbx_max[0]),
            bbx_max[1],
            0.5 * (bbx_min[2] + bbx_max[2]),
        ]
    )


def _fibonacci_sphere_directions(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """n directions on unit sphere (Fibonacci lattice)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    i = torch.arange(n, device=device, dtype=dtype)
    z = 1.0 - (2.0 * i + 1.0) / float(n)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    x = torch.cos(theta) * r
    y = torch.sin(theta) * r
    dirs = torch.stack([x, y, z], dim=-1)
    return torch.nn.functional.normalize(dirs, dim=-1, eps=1e-8)


def fibonacci_shell_means(
    resolution: int,
    radius: float,
    sky_origin: torch.Tensor,
    *,
    hemisphere: bool,
    device: torch.device,
    dtype: torch.dtype,
    up: Tuple[float, float, float] = SKY_UP_MULTISCENE,
) -> torch.Tensor:
    """
    Exactly N = resolution**2 points: full Fibonacci sphere, or hemisphere (dir·up >= 0).

    ``up`` is the **outward** reference for the kept half-space: we keep unit directions with ``dot(dir, up) >= 0``.
    Default ``up`` is ``SKY_UP_MULTISCENE`` ``(0,-1,0)`` (physical sky / smaller MultiScene ``y``).
    For the **complementary** half of the Y cut (``dir_y >= 0`` from the sphere center), use ``up=(0,+1,0)``.
    For a **rear-facing** half-space in MultiScene (``z`` back negative / front positive), try ``up=(0,0,-1)`` or ``(0,0,1)``.
    """
    n_target = int(resolution) ** 2
    if n_target < 1:
        raise ValueError("resolution must be >= 1")
    up_t = torch.tensor(up, device=device, dtype=dtype)
    up_t = torch.nn.functional.normalize(up_t, dim=0, eps=1e-8)
    r = float(radius)
    origin = sky_origin.to(device=device, dtype=dtype).view(1, 3)

    if not hemisphere:
        dirs = _fibonacci_sphere_directions(n_target, device, dtype)
        return origin + (r * dirs)

    n_try = max(n_target * 4, n_target + 64)
    max_expand = 256 * n_target
    hem_dirs: torch.Tensor
    while True:
        dirs = _fibonacci_sphere_directions(n_try, device, dtype)
        dots = (dirs * up_t.unsqueeze(0)).sum(dim=-1)
        mask = dots >= 0.0
        hem = dirs[mask]
        if hem.shape[0] >= n_target:
            hem_dirs = hem[:n_target]
            break
        if n_try >= max_expand:
            raise RuntimeError(
                f"fibonacci_shell_means: could not collect {n_target} hemisphere dirs (got {hem.shape[0]} at n_try={n_try})."
            )
        n_try = min(n_try * 2, max_expand)

    return origin + (r * hem_dirs)


def fibonacci_hemisphere_means(
    resolution: int,
    radius: float,
    sky_origin: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    up: Tuple[float, float, float] = SKY_UP_MULTISCENE,
) -> torch.Tensor:
    """Backward-compatible alias: hemisphere shell only."""
    return fibonacci_shell_means(
        resolution,
        radius,
        sky_origin,
        hemisphere=True,
        device=device,
        dtype=dtype,
        up=up,
    )
