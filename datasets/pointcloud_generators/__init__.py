"""RGB point cloud generator package."""

from .base import RGBPointCloudGenerator
from .monocular import MonocularRGBPointCloudGenerator
from .lidar import LiDARRGBPointCloudGenerator

__all__ = [
    "RGBPointCloudGenerator",
    "MonocularRGBPointCloudGenerator",
    "LiDARRGBPointCloudGenerator",
]
