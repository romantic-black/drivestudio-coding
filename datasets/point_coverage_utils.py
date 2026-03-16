"""
Point coverage mask utilities: project 3D points to image plane and build binary masks
for loss masking (only supervise pixels covered by initial point cloud).
World coordinates = segment-first-frame (seg0). Extrinsics = camera_to_world (c2w).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) to 3x3 rotation matrix. q: [4] or [N, 4]."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z + 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
    row1 = np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
    row2 = np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def project_points_to_image(
    points_xyz: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world (seg0) points to image pixel coordinates.

    Args:
        points_xyz: [N, 3] world coordinates.
        c2w: [4, 4] camera-to-world (extrinsics).
        K: [3, 3] intrinsics (fx, fy, cx, cy).
        height: image height.
        width: image width.

    Returns:
        u: [N] pixel x (integer).
        v: [N] pixel y (integer).
        valid: [N] bool, True if in front of camera and inside [0, width) x [0, height).
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim == 1:
        points_xyz = points_xyz[np.newaxis, :]
    N = points_xyz.shape[0]
    c2w = np.asarray(c2w, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    ones = np.ones((N, 1), dtype=np.float64)
    homo = np.concatenate([points_xyz, ones], axis=1)
    p_cam = (w2c[:3, :] @ homo.T).T
    z_cam = p_cam[:, 2]
    x_cam = p_cam[:, 0]
    y_cam = p_cam[:, 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)
    valid = (
        (z_cam > 1e-6)
        & (u_int >= 0)
        & (u_int < width)
        & (v_int >= 0)
        & (v_int < height)
    )
    return u_int, v_int, valid


def _transform_dynamic_local_to_world(
    points_local: np.ndarray,
    quat: List[float],
    trans: List[float],
) -> np.ndarray:
    """Transform points from instance local to world (seg0). quat wxyz, trans [x,y,z]."""
    R = _quat_to_rotmat_np(np.array(quat, dtype=np.float64))
    if R.ndim == 3:
        R = R[0]
    t = np.array(trans, dtype=np.float64)
    return (points_local @ R.T) + t


def build_point_coverage_masks(
    pointcloud: Dict,
    dynamic_info: Optional[Dict],
    input_aabb_min: np.ndarray,
    input_aabb_max: np.ndarray,
    target_frame_idxs: Union[np.ndarray, List[int]],
    target_extrinsics: np.ndarray,
    target_intrinsics: np.ndarray,
    target_shapes: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Build one binary coverage mask [H, W] per target view (1 = at least one point projects there).

    Args:
        pointcloud: dict with "background" [N, 6] (x,y,z,r,g,b seg0), optional "dynamic" {instance_id: [M, 6]} (local).
        dynamic_info: Dict[frame_idx, {"instances": {instance_id: {"quat": [4], "trans": [3]}}}], or None.
        input_aabb_min, input_aabb_max: [3] seg0 AABB for filtering background.
        target_frame_idxs: length V, frame index per target view.
        target_extrinsics: [V, 4, 4] camera_to_world per view.
        target_intrinsics: [V, 4, 4] or [V, 3, 3] intrinsics per view.
        target_shapes: list of (height, width) per view.

    Returns:
        List of [H, W] float32 masks (1.0 = valid, 0.0 = no coverage).
    """
    input_min = np.asarray(input_aabb_min, dtype=np.float64).ravel()[:3]
    input_max = np.asarray(input_aabb_max, dtype=np.float64).ravel()[:3]
    bg = pointcloud.get("background")
    if bg is None or not hasattr(bg, "shape") or bg.shape[0] == 0:
        background_xyz = np.zeros((0, 3), dtype=np.float64)
    else:
        bg = np.asarray(bg, dtype=np.float64)
        background_xyz = bg[:, :3].copy()
        inside = (
            (background_xyz >= input_min)
            & (background_xyz <= input_max)
        ).all(axis=1)
        background_xyz = background_xyz[inside]

    dynamic_pcd = pointcloud.get("dynamic")
    if not isinstance(dynamic_pcd, dict):
        dynamic_pcd = {}
    instance_ids_in_pcd = sorted(int(k) for k in dynamic_pcd.keys())

    V = len(target_shapes)
    if isinstance(target_frame_idxs, (list, tuple)):
        target_frame_idxs = np.array(target_frame_idxs, dtype=np.int64)
    target_extrinsics = np.asarray(target_extrinsics, dtype=np.float64)
    target_intrinsics = np.asarray(target_intrinsics, dtype=np.float64)
    if target_intrinsics.shape[-1] == 4:
        K_per_view = target_intrinsics[:, :3, :3]
    else:
        K_per_view = target_intrinsics

    masks = []
    for i in range(V):
        h, w = target_shapes[i]
        c2w = target_extrinsics[i]
        K = K_per_view[i]
        frame_idx = int(target_frame_idxs[i])

        xyz_list = [background_xyz]

        if dynamic_info and instance_ids_in_pcd:
            frame_instances = dynamic_info.get(frame_idx) or dynamic_info.get(str(frame_idx))
            if isinstance(frame_instances, dict):
                inst_dict = frame_instances.get("instances", frame_instances)
                if isinstance(inst_dict, dict):
                    for ins_id in instance_ids_in_pcd:
                        if ins_id not in inst_dict:
                            continue
                        pose = inst_dict[ins_id]
                        if not isinstance(pose, dict):
                            continue
                        quat = pose.get("quat")
                        trans = pose.get("trans")
                        if quat is None or trans is None:
                            continue
                        pcd_local = dynamic_pcd.get(ins_id)
                        if pcd_local is None or len(pcd_local) == 0:
                            continue
                        pcd_local = np.asarray(pcd_local, dtype=np.float64)
                        pts_local = pcd_local[:, :3]
                        pts_world = _transform_dynamic_local_to_world(pts_local, quat, trans)
                        xyz_list.append(pts_world)

        all_xyz = np.concatenate(xyz_list, axis=0) if len(xyz_list) > 1 else xyz_list[0]
        if all_xyz.size == 0:
            masks.append(np.zeros((h, w), dtype=np.float32))
            continue

        u_int, v_int, valid = project_points_to_image(all_xyz, c2w, K, h, w)
        u_int = u_int[valid]
        v_int = v_int[valid]
        mask = np.zeros((h, w), dtype=np.float32)
        if u_int.size > 0:
            np.maximum.at(mask, (v_int, u_int), 1.0)
        masks.append(mask)

    return masks
