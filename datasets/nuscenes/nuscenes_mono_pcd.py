"""
NuScenes单目点云生成器

从深度图生成点云，支持多种稀疏度级别、天空过滤、深度一致性检查等功能。
"""

import os
import sys
import numpy as np
import cv2
import open3d as o3d
from typing import Literal, List, Tuple, Optional
from copy import deepcopy

# 添加depth_utils路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
depth_utils_path = os.path.join(
    project_root, 
    'third_party/EVolSplat/preprocess/metric3d/mono/tools'
)
if depth_utils_path not in sys.path:
    sys.path.insert(0, depth_utils_path)

try:
    from depth_utils import process_depth_for_use, load_depth_with_metadata
except ImportError as e:
    raise ImportError(
        f"Failed to import depth_utils: {e}. "
        f"Make sure depth_utils.py is at: {depth_utils_path}"
    )

# Default bounding box for nuScenes
X_MIN, X_MAX = -20, 20
Y_MIN, Y_MAX = -20, 4.8
Z_MIN, Z_MAX = -20, 70

# OpenCV to Dataset coordinate transformation
OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class NuScenesMonoPCDGenerator:
    """Point cloud generator for nuScenes dataset from monocular depth maps."""
    
    def __init__(
        self,
        sparsity: Literal['Drop90', 'Drop80', 'Drop50', 'Drop25', 'full'] = 'full',
        save_dir: Optional[str] = None,
        frame_start: int = 0,
        filter_sky: bool = True,
        depth_consistency: bool = True,
        use_bbx: bool = True,
        bbx_min: Optional[np.ndarray] = None,
        bbx_max: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the point cloud generator.
        
        Args:
            sparsity: Sparsity level for frame filtering
            save_dir: Save directory name (None uses sparsity value)
            frame_start: Starting frame index
            filter_sky: Whether to filter sky regions
            depth_consistency: Whether to perform depth consistency check
            use_bbx: Whether to use bounding box cropping
            bbx_min: Minimum bounding box coordinates [x_min, y_min, z_min]. If None, uses default.
            bbx_max: Maximum bounding box coordinates [x_max, y_max, z_max]. If None, uses default.
        """
        self.sparsity = sparsity
        self.save_dir = save_dir if save_dir is not None else sparsity
        self.frame_start = frame_start
        self.filter_sky = filter_sky
        self.depth_consistency = depth_consistency
        self.use_bbx = use_bbx
        
        # Custom bounding box or use defaults
        if bbx_min is not None and bbx_max is not None:
            self.bbx_min = np.array(bbx_min)
            self.bbx_max = np.array(bbx_max)
        else:
            self.bbx_min = None
            self.bbx_max = None
        
        # Will be set during processing
        self.dir_name = None
        self.depth_dir = None
        self.H = None
        self.W = None
        self.camera_front_start = None
        self.c2w = []
        self.intri = []
    
    def get_bbx(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box for point cloud filtering."""
        if self.bbx_min is not None and self.bbx_max is not None:
            return self.bbx_min.copy(), self.bbx_max.copy()
        else:
            # Use default bounding box
            return np.array([X_MIN, Y_MIN, Z_MIN]), np.array([X_MAX, Y_MAX, Z_MAX])
    
    def crop_pointcloud(
        self,
        bbx_min: np.ndarray,
        bbx_max: np.ndarray,
        points: np.ndarray,
        color: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop point cloud to bounding box."""
        mask = (
            (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) &
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) &
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2] + 50)  # Extended Z for background
        )
        return points[mask], color[mask]
    
    def split_pointcloud(
        self,
        bbx_min: np.ndarray,
        bbx_max: np.ndarray,
        points: np.ndarray,
        color: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split point cloud into inside and outside bounding box."""
        mask = (
            (points[:, 0] > bbx_min[0]) & (points[:, 0] < bbx_max[0]) &
            (points[:, 1] > bbx_min[1]) & (points[:, 1] < bbx_max[1]) &
            (points[:, 2] > bbx_min[2]) & (points[:, 2] < bbx_max[2])
        )
        inside_pnt, inside_rgb = points[mask], color[mask]
        outside_pnt, outside_rgb = points[~mask], color[~mask]
        return inside_pnt, inside_rgb, outside_pnt, outside_rgb
    
    def depth_consistency_check(
        self,
        depth_files: List[str],
        H: int,
        W: int
    ) -> List[np.ndarray]:
        """Check depth consistency between consecutive frames."""
        depth_masks = []
        
        for i, file_name in enumerate(depth_files):
            depth_file = os.path.join(self.depth_dir, file_name)
            try:
                depth, metadata = process_depth_for_use(depth_file, target_shape=(H, W))
            except Exception as e:
                print(f"Warning: Failed to load depth {file_name}: {e}")
                depth_masks.append(np.ones((H, W), dtype=np.bool_))
                continue
            
            # Assume the first depth frame is correct
            if i == 0:
                self._last_depth = deepcopy(depth)
                depth_masks.append(np.ones((H, W), dtype=np.bool_))
                continue
            
            try:
                c2w = self.c2w[i]
                last_c2w = self.c2w[i-1]
                K = self.intri[i]
            except IndexError as e:
                depth_masks.append(np.ones((H, W), dtype=np.bool_))
                continue
            
            # Unproject pointcloud
            x = np.arange(0, depth.shape[1])
            y = np.arange(0, depth.shape[0])
            xx, yy = np.meshgrid(x, y)
            pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)
            
            # Unproject depth to pointcloud
            cx, cy = K[0, 2], K[1, 2]
            fx, fy = K[0, 0], K[1, 1]
            
            x_cam = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
            y_cam = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
            z_cam = depth.reshape(-1)
            coordinates = np.stack([x_cam, y_cam, z_cam], axis=1)
            
            depth_mask = self.depth_projection_check(
                coordinates=coordinates,
                pixels=pixels,
                last_c2w=last_c2w,
                c2w=c2w,
                last_depth=self._last_depth,
                depth=depth,
                K=K
            )
            depth_masks.append(depth_mask)
            
            # Update status
            self._last_depth = deepcopy(depth)
        
        return depth_masks
    
    def depth_projection_check(
        self,
        coordinates: np.ndarray,
        pixels: np.ndarray,
        last_c2w: np.ndarray,
        c2w: np.ndarray,
        last_depth: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """Check depth consistency by projecting current points to previous frame."""
        H, W = last_depth.shape[:2]
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        
        trans_mat = np.dot(np.linalg.inv(last_c2w), c2w)
        coordinates_homo = np.column_stack((coordinates.reshape(-1, 3), np.ones(len(coordinates))))
        last_coordinates = np.dot(trans_mat, coordinates_homo.T).T
        
        # Project to previous frame
        last_x = (fx * last_coordinates[:, 0] + cx * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_y = (fy * last_coordinates[:, 1] + cy * last_coordinates[:, 2]) / last_coordinates[:, 2]
        last_pixels = np.vstack((last_x, last_y)).T.reshape(-1, 2).astype(np.int32)
        
        # Swap pixel coordinates (row, col) <-> (x, y)
        pixels_swapped = pixels.copy()
        pixels_swapped[:, [0, 1]] = pixels_swapped[:, [1, 0]]
        last_pixels_swapped = last_pixels.copy()
        last_pixels_swapped[:, [0, 1]] = last_pixels_swapped[:, [1, 0]]
        
        depth_mask = np.ones(depth.shape[0] * depth.shape[1], dtype=np.bool_)
        
        # Reprojection location must be in image plane [0,H] [0,W]
        valid_mask_00 = (last_pixels_swapped[:, 0] < H) & (last_pixels_swapped[:, 1] < W)
        valid_mask_01 = (last_pixels_swapped[:, 0] > 0) & (last_pixels_swapped[:, 1] > 0)
        valid_mask = valid_mask_00 & valid_mask_01
        
        depth_diff = np.abs(
            depth[pixels_swapped[valid_mask, 0], pixels_swapped[valid_mask, 1]] -
            last_depth[last_pixels_swapped[valid_mask, 0], last_pixels_swapped[valid_mask, 1]]
        )
        depth_mask[valid_mask] = depth_diff < depth_diff.mean()
        depth_mask = depth_mask.reshape(*depth.shape)
        
        return depth_mask


def get_image_dimensions(scene_dir: str) -> Tuple[int, int]:
    """
    Get image dimensions from scene directory.
    
    Args:
        scene_dir: Scene directory path
        
    Returns:
        (H, W): Image height and width
    """
    import imageio.v2 as imageio
    
    images_dir = os.path.join(scene_dir, 'images')
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.endswith('.jpg') or f.endswith('.png')]
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Read first image to get dimensions
    first_image_path = os.path.join(images_dir, sorted(image_files)[0])
    img = imageio.imread(first_image_path)
    H, W = img.shape[0], img.shape[1]
    
    return H, W

