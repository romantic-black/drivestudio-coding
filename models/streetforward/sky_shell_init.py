"""
Sky shell utilities for MultiSceneDataset world coordinates.

Coordinate system:
x = left(-) / right(+)
y = up(-) / down(+)
z = back(-) / front(+)

Physical sky direction is -Y.
Ground / bottom of the segment AABB is the y_max face.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Unit vector toward sky in MultiSceneDataset world coordinates.
SKY_UP_MULTISCENE: Tuple[float, float, float] = (0.0, -1.0, 0.0)


def _as_vec3(x: torch.Tensor, *, name: str, device=None, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.numel() != 3:
        raise ValueError(f"{name} must contain exactly 3 elements, got shape={tuple(t.shape)}")
    return t.reshape(3)


def sky_base_from_aabb(bbx_min: torch.Tensor, bbx_max: torch.Tensor) -> torch.Tensor:
    """
    Ground-face center: mid X, y_max (road / bottom), mid Z.
    """
    bbx_min = _as_vec3(bbx_min, name="bbx_min")
    bbx_max = _as_vec3(bbx_max, name="bbx_max")
    if not torch.all(bbx_min < bbx_max):
        raise ValueError("Expected bbx_min < bbx_max elementwise.")

    return torch.stack(
        [
            0.5 * (bbx_min[0] + bbx_max[0]),
            bbx_max[1],
            0.5 * (bbx_min[2] + bbx_max[2]),
        ],
        dim=0,
    )


def sky_base_from_reference(
    *,
    reference_origin: Optional[torch.Tensor],
    bbx_min: torch.Tensor,
    bbx_max: torch.Tensor,
    use_ground_y: bool = True,
) -> torch.Tensor:
    """
    Prefer a reference origin (e.g. source ego/camera rig center) when available.
    Fallback to AABB-based anchor when reference_origin is None.

    If use_ground_y=True, x/z come from reference_origin and y comes from AABB y_max.
    This keeps the shell anchored near the camera rig horizontally while still sitting on
    the segment's ground-face height convention.
    """
    bbx_min = _as_vec3(bbx_min, name="bbx_min")
    bbx_max = _as_vec3(bbx_max, name="bbx_max")
    if not torch.all(bbx_min < bbx_max):
        raise ValueError("Expected bbx_min < bbx_max elementwise.")

    if reference_origin is None:
        return sky_base_from_aabb(bbx_min, bbx_max)

    ref = _as_vec3(
        reference_origin,
        name="reference_origin",
        device=bbx_min.device,
        dtype=bbx_min.dtype,
    )

    if use_ground_y:
        return torch.stack([ref[0], bbx_max[1], ref[2]], dim=0)
    return ref


def _fibonacci_sphere_directions(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return n approximately uniform directions on the unit sphere."""
    if n < 1:
        raise ValueError("n must be >= 1")

    i = torch.arange(n, device=device, dtype=dtype)
    z = 1.0 - (2.0 * i + 1.0) / float(n)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))

    # Standard golden angle form.
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * i

    x = torch.cos(theta) * r
    y = torch.sin(theta) * r
    dirs = torch.stack([x, y, z], dim=-1)
    return F.normalize(dirs, dim=-1, eps=1e-8)


def _even_subsample_rows(x: torch.Tensor, n_target: int) -> torch.Tensor:
    """
    Deterministically select n_target rows approximately evenly from x.
    """
    n = int(x.shape[0])
    if n < n_target:
        raise ValueError(f"Cannot subsample {n_target} rows from only {n} rows.")
    if n == n_target:
        return x

    idx = torch.linspace(
        0,
        n - 1,
        steps=n_target,
        device=x.device,
        dtype=torch.float32,
    ).round().long()
    return x[idx]


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
    Return exactly N = resolution**2 shell points centered at sky_origin.

    - Full sphere: use all N directions.
    - Hemisphere: keep directions with dot(dir, up) >= 0, then evenly subsample to N.
    """
    n_target = int(resolution) ** 2
    if n_target < 1:
        raise ValueError("resolution must be >= 1")

    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")

    origin = _as_vec3(sky_origin, name="sky_origin", device=device, dtype=dtype).view(1, 3)
    up_t = _as_vec3(torch.tensor(up), name="up", device=device, dtype=dtype)
    up_norm = torch.linalg.norm(up_t)
    if float(up_norm.item()) <= 1e-12:
        raise ValueError("up must be non-zero.")
    up_t = up_t / up_norm

    r = float(radius)

    if not hemisphere:
        dirs = _fibonacci_sphere_directions(n_target, device=device, dtype=dtype)
        return origin + r * dirs

    # Oversample, then keep the requested half-space.
    n_try = max(n_target * 4, n_target + 64)
    max_expand = max(256 * n_target, n_target + 64)

    while True:
        dirs = _fibonacci_sphere_directions(n_try, device=device, dtype=dtype)
        dots = (dirs * up_t.view(1, 3)).sum(dim=-1)
        hem = dirs[dots >= 0.0]

        if int(hem.shape[0]) >= n_target:
            hem_dirs = _even_subsample_rows(hem, n_target)
            return origin + r * hem_dirs

        if n_try >= max_expand:
            raise RuntimeError(
                f"fibonacci_shell_means: could not collect {n_target} hemisphere dirs "
                f"(got {hem.shape[0]} at n_try={n_try})."
            )
        n_try = min(n_try * 2, max_expand)


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
        resolution=resolution,
        radius=radius,
        sky_origin=sky_origin,
        hemisphere=True,
        device=device,
        dtype=dtype,
        up=up,
    )