"""
Projector component extracted from EVolSplat for 2D image feature extraction.
"""
import torch
import torch.nn.functional as F


class Projector:
    """Projector for sampling 2D image features from 3D points."""

    def __init__(self):
        print("Init the Projector in OpenGL system")

    def inbound(self, pixel_locations, h, w):
        """
        Check if the pixel locations are in valid range.
        
        Args:
            pixel_locations: [..., 2]
            h: height
            w: width
            
        Returns:
            mask: bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def normalize(self, pixel_locations, h, w):
        """Normalize pixel locations to [-1, 1] range.
        
        Args:
            pixel_locations: [n_views, n_points, 2]
            h: height
            w: width
            
        Returns:
            normalized_pixel_locations: [n_views, n_points, 2]
        """
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(
            pixel_locations.device
        )[None, None, :]
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras, train_intrinsics):
        """
        Project 3D points into cameras.
        
        Args:
            xyz: [..., 3] OpenCV coordinates
            train_cameras: [n_views, 4, 4] OpenGL camera-to-world
            train_intrinsics: [n_views, 4, 4] camera intrinsics
            
        Returns:
            pixel_locations: [n_views, ..., 2]
            mask: [n_views, ...] visibility mask
            depth: [n_views, ...] depth values
        """
        original_shape = xyz.shape[:1]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_cameras = train_cameras * torch.tensor(
            [1, -1, -1, 1], device="cuda"
        )
        train_poses = train_cameras.reshape(-1, 4, 4)  # [n_views, 4, 4]

        xyz_h = torch.cat(
            [xyz, torch.ones_like(xyz[..., :1])], dim=-1
        )  # [n_points, 4]
        projections = (
            train_intrinsics.bmm(torch.inverse(train_poses))
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))
        )  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = (
            projections[..., :2]
            / torch.clamp(projections[..., 2:3], min=1e-8)
        )  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0  # A point is invalid if behind the camera

        depth = projections[..., 2].reshape((num_views,) + original_shape)
        return (
            pixel_locations.reshape((num_views,) + original_shape + (2,)),
            mask.reshape((num_views,) + original_shape),
            depth,
        )

    def compute(
        self, xyz, train_imgs, train_cameras, train_intrinsics, cam_idx=0
    ):
        """
        Compute RGB features for background points.
        
        Args:
            xyz: [n_samples, 3]
            train_imgs: [n_views, c, h, w]
            train_cameras: [n_views, 4, 4], in OpenGL
            train_intrinsics: [n_views, 4, 4]
            
        Returns:
            rgb: [n_samples, n_views, c]
            projection_mask: [n_samples] bool mask
        """
        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # Compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, _ = self.compute_projections(
            xyz, train_cameras, train_intrinsics.clone()
        )
        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(
            dim=1
        )  # [n_views, 1, n_points, 2]

        # RGB sampling
        rgbs_sampled = F.grid_sample(
            train_imgs, normalized_pixel_locations, align_corners=False
        )
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(
            dim=0
        )  # [n_points, n_views, 3]

        # Mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 0)[
            ..., None
        ]  # [n_rays, n_samples, n_views, 1]
        rgb = rgb_sampled.masked_fill(mask == 0, 0)

        projection_mask = mask[..., :].sum(dim=1) > 0
        return rgb[projection_mask.squeeze()], projection_mask.squeeze()

    def sample_within_window(
        self,
        xyz,
        train_imgs,
        train_cameras,
        train_intrinsics,
        source_depth=None,
        local_radius=2,
        depth_delta=0.2,
    ):
        """
        Sample features within a local window around projected points.
        
        Args:
            xyz: [n_samples, 3]
            train_imgs: [n_views, c, h, w]
            train_cameras: [n_views, 4, 4], in OpenGL
            train_intrinsics: [n_views, 4, 4]
            source_depth: [n_views, h, w] for occlusion-aware IBR
            local_radius: Radius of the local window
            
        Returns:
            rgb_feat_sampled: [n_samples, n_views, local_h*local_w, 3]
            valid_mask: [n_samples, n_views, local_h*local_w]
            visibility_map: [n_samples, n_views, local_h*local_w, 1]
        """
        n_views, _, _ = train_cameras.shape
        n_samples = xyz.shape[0]

        local_h = 2 * local_radius + 1
        local_w = 2 * local_radius + 1
        window_grid = self.generate_window_grid(
            -local_radius,
            local_radius,
            -local_radius,
            local_radius,
            local_h,
            local_w,
            device=xyz.device,
        )  # [2R+1, 2R+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_views, 1, 1)

        xyz = xyz.detach()
        h, w = train_imgs.shape[2:]

        # Sample within the window size
        pixel_locations, mask_in_front, project_depth = (
            self.compute_projections(xyz, train_cameras, train_intrinsics.clone())
        )

        # Occlusion-aware check for IBR
        if source_depth is not None:
            source_depth = source_depth.unsqueeze(-1).permute(0, 3, 1, 2).cuda()
            depths_sampled = F.grid_sample(
                source_depth,
                self.normalize(pixel_locations, h, w).unsqueeze(dim=1),
                align_corners=False,
            )
            depths_sampled = depths_sampled.squeeze()
            retrived_depth = depths_sampled.masked_fill(mask_in_front == 0, 0)
            projected_depth = project_depth * mask_in_front

            # Use depth priors to distinguish the Occlusion Region
            visibility_map = projected_depth - retrived_depth
            visibility_map = (
                visibility_map.unsqueeze(-1)
                .repeat(1, 1, local_h * local_w)
                .reshape(n_views, n_samples, -1)
            )
        else:
            visibility_map = torch.ones_like(project_depth)

        pixel_locations = (
            pixel_locations.unsqueeze(dim=2) + window_grid.unsqueeze(dim=1)
        )
        pixel_locations = pixel_locations.reshape(
            n_views, -1, 2
        )  # [N_view, N_points, 2]

        # Broadcast the mask
        mask_in_front = (
            mask_in_front.unsqueeze(-1)
            .repeat(1, 1, local_h * local_w)
            .reshape(n_views, -1)
        )
        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [n_views, n_points, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(
            dim=1
        )  # [n_views, 1, n_points, 2]

        # RGB sampling
        rgbs_sampled = F.grid_sample(
            train_imgs, normalized_pixel_locations, align_corners=False
        )
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1).squeeze(
            dim=0
        )  # [n_points, n_views, 3]

        # Mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 0)[
            ..., None
        ]  # [n_samples, n_views, local_h*local_w, 1]
        rgb = rgb_sampled.masked_fill(mask == 0, 0)

        return (
            rgb.reshape(n_samples, n_views, local_w * local_h, 3),
            mask.reshape(n_samples, n_views, local_w * local_h),
            visibility_map.permute(1, 0, 2).unsqueeze(-1),
        )

    def generate_window_grid(
        self, h_min, h_max, w_min, w_max, len_h, len_w, device=None
    ):
        """Generate a window grid for local sampling."""
        assert device is not None

        x, y = torch.meshgrid(
            [
                torch.linspace(w_min, w_max, len_w, device=device),
                torch.linspace(h_min, h_max, len_h, device=device),
            ],
        )
        grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

        return grid

