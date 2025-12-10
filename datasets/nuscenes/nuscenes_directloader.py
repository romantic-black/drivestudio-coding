import logging
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.nuscenes import LidarPointCloud, NuScenes

from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import CameraData, ScenePixelSource
from datasets.dataset_meta import DATASETS_CONFIG
from datasets.nuscenes.nuscenes_sourceloader import (
    OBJECT_CLASS_NODE_MAPPING,
    OPENCV2DATASET,
    AVAILABLE_CAM_LIST,
)

logger = logging.getLogger()


def _collect_keyframe_tokens(nusc: NuScenes, scene_data: dict) -> List[str]:
    """Return the list of keyframe sample tokens for a scene."""
    tokens: List[str] = []
    sample_token = scene_data["first_sample_token"]
    while True:
        tokens.append(sample_token)
        sample = nusc.get("sample", sample_token)
        if sample_token == scene_data["last_sample_token"] or sample["next"] == "":
            break
        sample_token = sample["next"]
    return tokens


def _sensor_to_world(nusc: NuScenes, sample_data_token: str) -> np.ndarray:
    """Compute the sensor-to-world transform for a sample_data token."""
    sample_data = nusc.get("sample_data", sample_data_token)
    calibrated = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

    sensor_to_ego = np.eye(4)
    sensor_to_ego[:3, :3] = Quaternion(calibrated["rotation"]).rotation_matrix
    sensor_to_ego[:3, 3] = np.array(calibrated["translation"])

    ego_to_world = np.eye(4)
    ego_to_world[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
    ego_to_world[:3, 3] = np.array(ego_pose["translation"])
    return ego_to_world @ sensor_to_ego


def _get_reference_pose_inv(
    nusc: NuScenes,
    scene_data: dict,
    start_idx: int,
    cam_id: int = AVAILABLE_CAM_LIST[0],
) -> np.ndarray:
    """Use the first selected camera (default front) as the world origin."""
    keyframe_tokens = _collect_keyframe_tokens(nusc, scene_data)
    if start_idx >= len(keyframe_tokens):
        raise ValueError(f"start_idx {start_idx} exceeds number of keyframes {len(keyframe_tokens)}")
    sample = nusc.get("sample", keyframe_tokens[start_idx])
    cam_name = DATASETS_CONFIG["nuscenes"][cam_id]["camera_name"]
    cam_token = sample["data"][cam_name]
    reference_pose = _sensor_to_world(nusc, cam_token)
    return np.linalg.inv(reference_pose)


def _scale_intrinsics(intrinsic: np.ndarray, load_size: Sequence[int], original_size: Sequence[int]) -> np.ndarray:
    """Scale intrinsics according to resize."""
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    fx *= load_size[1] / original_size[1]
    fy *= load_size[0] / original_size[0]
    cx *= load_size[1] / original_size[1]
    cy *= load_size[0] / original_size[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


class NuScenesDirectCameraData(CameraData):
    """CameraData that pulls images and calibrations directly from NuScenes."""

    def __init__(
        self,
        nusc: NuScenes,
        sample_data_tokens: Sequence[str],
        reference_inv: np.ndarray,
        scene_data: dict,
        **kwargs,
    ):
        self.nusc = nusc
        self.sample_data_tokens = list(sample_data_tokens)
        self.reference_inv = reference_inv
        self.scene_data = scene_data
        super().__init__(**kwargs)

    def create_all_filelist(self):
        # Store sample_data tokens instead of file paths.
        self.img_tokens = np.array(self.sample_data_tokens)

    def load_calibrations(self):
        cam_to_worlds, intrinsics, distortions = [], [], []
        for token in self.img_tokens:
            cam_to_world = _sensor_to_world(self.nusc, token)
            # Align to the first camera pose
            cam_to_world = self.reference_inv @ cam_to_world
            # Convert from OpenCV to dataset coordinates if needed
            cam_to_world = cam_to_world @ OPENCV2DATASET
            cam_to_worlds.append(cam_to_world)

            calib = self.nusc.get(
                "calibrated_sensor", self.nusc.get("sample_data", token)["calibrated_sensor_token"]
            )
            intrinsic = np.array(calib["camera_intrinsic"], dtype=np.float32)
            intrinsic = _scale_intrinsics(intrinsic, self.load_size, self.original_size)
            intrinsics.append(intrinsic)
            distortions.append(np.zeros(5, dtype=np.float32))

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

    def load_images(self):
        images = []
        for ix, token in tqdm(
            enumerate(self.img_tokens),
            desc=f"Loading images for {self.cam_name}",
            dynamic_ncols=True,
            total=len(self.img_tokens),
        ):
            img_path, _, _ = self.nusc.get_sample_data(token)
            rgb = Image.open(img_path).convert("RGB")
            rgb = rgb.resize((self.load_size[1], self.load_size[0]), Image.BILINEAR)
            if self.undistort:
                rgb = np.array(rgb)
                rgb = cv2.undistort(
                    rgb, self.intrinsics[ix].numpy(), self.distortions[ix].numpy()
                )
                rgb = Image.fromarray(rgb)
            images.append(np.array(rgb))
        self.images = torch.from_numpy(np.stack(images, axis=0)).float() / 255.0


class NuScenesDirectPixelSource(ScenePixelSource):
    """Pixel source that reads NuScenes data directly via the devkit."""

    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        nusc: NuScenes,
        scene_idx: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.nusc = nusc
        self.scene_idx = scene_idx
        self.scene_data = self.nusc.get("scene", self.nusc.scene[self.scene_idx]["token"])
        self.sample_tokens: List[str] = []
        self.reference_inv: np.ndarray = None
        self.camera_data: Dict[int, CameraData] = {}
        self.load_data()

    def load_cameras(self):
        all_sample_tokens = _collect_keyframe_tokens(self.nusc, self.scene_data)
        end_idx = len(all_sample_tokens) if self.end_timestep == -1 else min(self.end_timestep, len(all_sample_tokens))
        self.sample_tokens = all_sample_tokens[self.start_timestep:end_idx]
        assert (
            len(self.sample_tokens) > 0
        ), f"No samples selected for scene {self.scene_idx} with range {self.start_timestep}:{end_idx}"

        # Use the front camera of the first selected frame as world origin
        self.reference_inv = _get_reference_pose_inv(
            self.nusc, self.scene_data, self.start_timestep, cam_id=AVAILABLE_CAM_LIST[0]
        )

        # Register shared timestamps (integer indices matching the selected frames)
        frame_count = len(self.sample_tokens)
        self._timesteps = torch.arange(self.start_timestep, self.start_timestep + frame_count)
        self.register_normalized_timestamps()

        # Prepare sample_data tokens per camera
        cam_tokens: Dict[int, List[str]] = {cam_id: [] for cam_id in self.camera_list}
        for sample_token in self.sample_tokens:
            sample = self.nusc.get("sample", sample_token)
            for cam_id in self.camera_list:
                cam_name = DATASETS_CONFIG[self.dataset_name][cam_id]["camera_name"]
                cam_tokens[cam_id].append(sample["data"][cam_name])

        if self.data_cfg.load_dynamic_mask or self.data_cfg.load_sky_mask:
            logger.warning("NuScenesDirectPixelSource does not support dynamic/sky masks; skipping them.")

        for idx, cam_id in enumerate(self.camera_list):
            logger.info(f"[Pixel] Loading camera {cam_id}")
            camera = NuScenesDirectCameraData(
                dataset_name=self.dataset_name,
                data_path=self.nusc.dataroot,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.start_timestep + frame_count,
                load_dynamic_mask=False,
                load_sky_mask=False,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                nusc=self.nusc,
                sample_data_tokens=cam_tokens[cam_id],
                reference_inv=self.reference_inv,
                scene_data=self.scene_data,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"[Pixel] Camera {camera.cam_name} loaded with {len(camera)} frames.")
            self.camera_data[cam_id] = camera

    def load_objects(self):
        if self.data_cfg.load_smpl:
            logger.warning("SMPL data is not available in direct NuScenes loader; skipping.")

        instances_info_raw: Dict[str, Dict] = {}

        for frame_idx, sample_token in enumerate(self.sample_tokens):
            sample = self.nusc.get("sample", sample_token)
            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                class_name = ann["category_name"]
                if class_name not in OBJECT_CLASS_NODE_MAPPING:
                    continue

                inst_token = ann["instance_token"]
                if inst_token not in instances_info_raw:
                    instances_info_raw[inst_token] = {
                        "id": inst_token,
                        "class_name": class_name,
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        },
                    }

                o2w = np.eye(4)
                o2w[:3, :3] = Quaternion(ann["rotation"]).rotation_matrix
                o2w[:3, 3] = np.array(ann["translation"])
                aligned_o2w = self.reference_inv @ o2w
                lwh = [ann["size"][1], ann["size"][0], ann["size"][2]]

                instances_info_raw[inst_token]["frame_annotations"]["frame_idx"].append(frame_idx)
                instances_info_raw[inst_token]["frame_annotations"]["obj_to_world"].append(aligned_o2w)
                instances_info_raw[inst_token]["frame_annotations"]["box_size"].append(lwh)

        num_frames = len(self.sample_tokens)
        num_instances = len(instances_info_raw)
        instances_pose = torch.zeros((num_frames, num_instances, 4, 4), dtype=torch.float32)
        instances_size = torch.zeros((num_instances, 3), dtype=torch.float32)
        per_frame_instance_mask = torch.zeros((num_frames, num_instances), dtype=torch.bool)
        instances_true_id = torch.arange(num_instances, dtype=torch.long)
        instances_model_types = torch.ones(num_instances, dtype=torch.long) * -1
        counts = torch.zeros((num_instances, 1), dtype=torch.float32)

        for new_id, (inst_token, info) in enumerate(instances_info_raw.items()):
            instances_model_types[new_id] = OBJECT_CLASS_NODE_MAPPING[info["class_name"]]
            for frame_idx, pose, box_size in zip(
                info["frame_annotations"]["frame_idx"],
                info["frame_annotations"]["obj_to_world"],
                info["frame_annotations"]["box_size"],
            ):
                instances_pose[frame_idx, new_id] = torch.from_numpy(np.asarray(pose, dtype=np.float32))
                instances_size[new_id] += torch.tensor(box_size, dtype=torch.float32)
                per_frame_instance_mask[frame_idx, new_id] = True
                counts[new_id] += 1

        counts = torch.clamp(counts, min=1.0)
        instances_size = instances_size / counts

        self.instances_pose = instances_pose
        self.instances_size = instances_size
        self.per_frame_instance_mask = per_frame_instance_mask
        self.instances_true_id = instances_true_id
        self.instances_model_types = instances_model_types


class NuScenesDirectLiDARSource(SceneLidarSource):
    """LiDAR source that directly reads NuScenes raw point clouds."""

    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        nusc: NuScenes,
        scene_data: dict,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.nusc = nusc
        self.scene_data = scene_data
        self.reference_inv = _get_reference_pose_inv(
            self.nusc, self.scene_data, self.start_timestep, cam_id=AVAILABLE_CAM_LIST[0]
        )
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        all_sample_tokens = _collect_keyframe_tokens(self.nusc, self.scene_data)
        end_idx = len(all_sample_tokens) if self.end_timestep == -1 else min(self.end_timestep, len(all_sample_tokens))
        self.lidar_tokens = all_sample_tokens[self.start_timestep:end_idx]

    def load_calibrations(self):
        lidar_to_worlds = []
        for token in self.lidar_tokens:
            lidar_to_world = _sensor_to_world(self.nusc, token)
            lidar_to_world = self.reference_inv @ lidar_to_world
            lidar_to_worlds.append(lidar_to_world)
        self.lidar_to_worlds = torch.from_numpy(np.stack(lidar_to_worlds, axis=0)).float()

    def load_lidar(self):
        origins, directions, ranges, timesteps = [], [], [], []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0

        for idx, token in enumerate(
            tqdm(self.lidar_tokens, desc="Loading lidar", dynamic_ncols=True)
        ):
            lidar_path, _, _ = self.nusc.get_sample_data(token)
            pc = LidarPointCloud.from_file(lidar_path)
            lidar_points = torch.from_numpy(pc.points[:3, :].T).float()

            original_length = len(lidar_points)
            accumulated_num_original_rays += original_length

            lidar_origins = torch.zeros_like(lidar_points)
            lidar_origins = (
                self.lidar_to_worlds[idx][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[idx][:3, 3:4]
            ).T
            lidar_points_world = (
                self.lidar_to_worlds[idx][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[idx][:3, 3:4]
            ).T
            lidar_directions = lidar_points_world - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / (lidar_ranges + 1e-8)
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * (
                self.start_timestep + idx
            )
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)

        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()
