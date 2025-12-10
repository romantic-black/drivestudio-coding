"""
Utility functions extracted from EVolSplat.
"""
import torch
import torch.nn.functional as F


def interpolate_features(grid_coords, feature_volume):
    """Interpolate features from a 3D volume using trilinear interpolation.
    
    Args:
        grid_coords: Grid coordinates [N, 3] in [-1, 1] range
        feature_volume: Feature volume [B, C, H, W, D]
        
    Returns:
        Interpolated features [C, N, 1, 1]
    """
    grid_coords = grid_coords[None, None, None, ...]
    feature = F.grid_sample(
        feature_volume,
        grid_coords,
        mode="bilinear",
        align_corners=True,
    )
    return feature


def get_grid_coords(position_w, bbx_min, vol_dim, voxel_size=0.1):
    """Convert world coordinates to grid coordinates for volume interpolation.
    
    Args:
        position_w: World coordinates [N, 3]
        bbx_min: Bounding box minimum [3]
        vol_dim: Volume dimensions [H, W, D]
        voxel_size: Voxel size
        
    Returns:
        Grid coordinates [N, 3] in [-1, 1] range
    """
    bounding_min = bbx_min
    pts = position_w - bounding_min.to(position_w.device)
    x_index = pts[..., 0] / voxel_size
    y_index = pts[..., 1] / voxel_size
    z_index = pts[..., 2] / voxel_size

    dhw = torch.stack([x_index, y_index, z_index], dim=1)

    # Normalize the point coordinates to [-1, 1]
    dhw[..., 0] = dhw[..., 0] / vol_dim[0] * 2 - 1
    dhw[..., 1] = dhw[..., 1] / vol_dim[1] * 2 - 1
    dhw[..., 2] = dhw[..., 2] / vol_dim[2] * 2 - 1
    grid_coords = dhw[..., [2, 1, 0]]  # Convert to [D, H, W] order
    return grid_coords

