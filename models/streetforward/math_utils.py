from __future__ import annotations

import torch

try:
    from sklearn.neighbors import NearestNeighbors

    _sklearn_available = True
except ImportError:  # pragma: no cover - optional dependency
    _sklearn_available = False


def _num_sh_bases(degree: int) -> int:
    """Number of spherical harmonics bases for given degree."""
    return (degree + 1) ** 2


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors in [0,1] to SH DC components."""
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert SH DC components back to RGB in [0,1]."""
    c0 = 0.28209479177387814
    return sh * c0 + 0.5


def _random_quat_tensor(num: int, device: torch.device) -> torch.Tensor:
    """Generate random unit quaternions (wxyz)."""
    u = torch.rand(num, device=device)
    v = torch.rand(num, device=device)
    w = torch.rand(num, device=device)
    x = torch.sqrt(1 - u) * torch.sin(2 * torch.pi * v)
    y = torch.sqrt(1 - u) * torch.cos(2 * torch.pi * v)
    z = torch.sqrt(u) * torch.sin(2 * torch.pi * w)
    ww = torch.sqrt(u) * torch.cos(2 * torch.pi * w)
    return torch.stack([ww, x, y, z], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication q1 * q2 (wxyz)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix with normalization."""
    q = _normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _axis_angle_to_quat(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Axis-angle (xyz) to quaternion (wxyz) using branchless sinc structure."""
    theta = torch.norm(omega, dim=-1, keepdim=True)
    half_theta = theta * 0.5
    sinc_half = torch.sin(half_theta) / (theta + eps)
    xyz = omega * sinc_half
    w = torch.cos(half_theta)
    return torch.cat([w, xyz], dim=-1)


def get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """
    Convert camera-to-world matrix to view matrix (gsplat convention).

    Mirrors the original implementation: flips Y/Z axes and transposes rotation.
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


def _pairwise_neighbor_distances(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute k-NN distances efficiently using sklearn's NearestNeighbors."""
    if not _sklearn_available:
        raise ImportError("sklearn is required for k-NN search. Please install scikit-learn.")

    if points.is_cuda:
        points_np = points.cpu().numpy().astype("float32")
    else:
        points_np = points.numpy().astype("float32")

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn_model.fit(points_np)
    distances, _ = nn_model.kneighbors(points_np)
    distances = distances[:, 1:]
    return torch.from_numpy(distances.astype("float32")).to(points.device)


__all__ = [
    "_num_sh_bases",
    "_rgb_to_sh",
    "_sh_to_rgb",
    "_random_quat_tensor",
    "_quat_multiply",
    "_quat_conjugate",
    "_normalize_quat",
    "_quat_to_rotmat",
    "_axis_angle_to_quat",
    "get_viewmat",
    "_pairwise_neighbor_distances",
]
