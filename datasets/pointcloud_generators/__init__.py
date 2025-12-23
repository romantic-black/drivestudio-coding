"""
RGB Point Cloud Generator Module

This module provides point cloud generation functionality for MultiSceneDataset.
"""

from .rgb_pointcloud_generator import (
    RGBPointCloudGenerator,
    MonocularRGBPointCloudGenerator,
)

__all__ = [
    'RGBPointCloudGenerator',
    'MonocularRGBPointCloudGenerator',
]

