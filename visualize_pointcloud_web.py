#!/usr/bin/env python3
"""
Lightweight Flask + Three.js point cloud viewer for NuScenes-generated PLY files.
Features:
- Recursively scans scene directories containing both images/ and depth/ (NuScenes preprocess output)
- Lists available point clouds across common sparsity subfolders (Drop90/80/50/25/full or custom save_dir)
- Loads PLY via Open3D, normalizes colors (handles 0-1 float and 0-255 uint8), and supports random sampling
- Returns JSON payload consumable by the front-end Three.js viewer

Usage (default):
    python visualize_pointcloud_web.py

Environment variables:
    NUSCENES_DATA_ROOT: optional override for scene root (e.g., /root/autodl-tmp/nuScenes/preprocess/trainval)
    PORT: server port (default: 5000)
"""

import os
import sys
import json
import random
from typing import Dict, List, Optional, Tuple

# Dependency check for Flask (adds install hint instead of hard crash)
try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    sys.stderr.write(
        "Flask is required for the web viewer.\n"
        "Install with: pip install Flask\n"
    )
    sys.exit(1)

import numpy as np
import open3d as o3d

APP = Flask(__name__, template_folder="templates")

DEFAULT_DATA_ROOTS = [
    "/root/autodl-tmp/nuScenes/preprocess/trainval",
]

SPARSITY_FOLDERS = ["Drop90", "Drop80", "Drop50", "Drop25", "full"]
# Skip heavy subdirectories inside a scene to speed up scanning
SKIP_DIRS = {
    "depth",
    "dynamic_masks",
    "extrinsics",
    "images",
    "instances",
    "intrinsics",
    "lidar",
    "lidar_pose",
    "semantic",
    "sky_masks",
}


def is_scene_dir(path: str) -> bool:
    """Check if directory looks like a processed NuScenes scene."""
    if not os.path.isdir(path):
        return False
    has_images = os.path.isdir(os.path.join(path, "images"))
    has_depth = os.path.isdir(os.path.join(path, "depth"))
    return has_images and has_depth


def discover_scene_roots() -> List[str]:
    """Find candidate scene roots."""
    roots: List[str] = []
    for candidate in DEFAULT_DATA_ROOTS:
        if candidate and os.path.isdir(candidate):
            roots.append(os.path.abspath(candidate))
    return roots


def scan_scenes(root: str, max_depth: int = 3) -> List[str]:
    """
    Recursively scan for scene directories containing images/ and depth/.
    Limit traversal depth to avoid slow walks.
    """
    scenes: List[str] = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"[WARNING] Root directory does not exist: {root}")
        return scenes
    
    for current_root, dirs, _ in os.walk(root):
        # Skip heavy subdirectories to avoid deep traversal inside a scene
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        # Calculate depth relative to root
        rel_path = os.path.relpath(current_root, root)
        if rel_path == '.':
            depth_level = 0
        else:
            depth_level = len(rel_path.split(os.sep))
        
        if depth_level > max_depth:
            # Prune deep traversal
            dirs[:] = []
            continue
        
        if is_scene_dir(current_root):
            scenes.append(current_root)
            # Do not descend further inside a detected scene
            dirs[:] = []
    
    return sorted(scenes)


def list_pointclouds(scene_dir: str) -> List[Dict[str, str]]:
    """List available PLY files under a scene directory across common save dirs."""
    pcs: List[Dict[str, str]] = []
    if not os.path.isdir(scene_dir):
        return pcs

    # Check direct .ply in scene root
    root_plys = [f for f in os.listdir(scene_dir) if f.endswith(".ply")]
    for f in root_plys:
        pcs.append(
            {
                "path": os.path.abspath(os.path.join(scene_dir, f)),
                "save_dir": ".",
                "file": f,
            }
        )

    # Check known sparsity subfolders or any subdir containing ply
    for sub in os.listdir(scene_dir):
        sub_path = os.path.join(scene_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        if sub not in SPARSITY_FOLDERS and sub.startswith("."):
            continue
        ply_files = [f for f in os.listdir(sub_path) if f.endswith(".ply")]
        for f in ply_files:
            pcs.append(
                {
                    "path": os.path.abspath(os.path.join(sub_path, f)),
                    "save_dir": sub,
                    "file": f,
                }
            )
    return sorted(pcs, key=lambda x: (x["save_dir"], x["file"]))


def normalize_colors(colors: np.ndarray) -> np.ndarray:
    """
    Normalize Open3D color array to uint8 0-255.
    Handles both float [0,1] and uint8 [0,255].
    """
    if colors.size == 0:
        return colors
    if colors.dtype != np.uint8:
        # Assume float; detect scale
        if colors.max() <= 1.0:
            colors = (colors * 255.0).clip(0, 255)
        colors = colors.astype(np.uint8)
    return colors


def load_ply(path: str, sample: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load PLY and optionally random sample points."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PLY not found: {path}")

    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

    colors = normalize_colors(colors)

    total = points.shape[0]
    if sample is not None and sample > 0 and total > sample:
        idx = np.random.choice(total, sample, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Convert colors to 0-1 float for Three.js (expects 0-1)
    colors_float = colors.astype(np.float32) / 255.0
    return points.astype(np.float32), colors_float


@APP.route("/")
def index():
    return render_template("pointcloud_viewer.html")


@APP.route("/api/list_scenes", methods=["GET"])
def api_list_scenes():
    try:
        roots = discover_scene_roots()
        print(f"[DEBUG] Discovered roots: {roots}")  # Debug log
        all_scenes: List[str] = []
        for r in roots:
            if os.path.isdir(r):
                scenes = scan_scenes(r)
                print(f"[DEBUG] Scanned {r}, found {len(scenes)} scenes")  # Debug log
                all_scenes.extend(scenes)
        # Remove duplicates while preserving order
        seen = set()
        unique_scenes = []
        for s in all_scenes:
            if s not in seen:
                seen.add(s)
                unique_scenes.append(s)
        print(f"[DEBUG] Total unique scenes: {len(unique_scenes)}")  # Debug log
        return jsonify({"roots": roots, "scenes": unique_scenes})
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to list scenes: {e}\n{traceback.format_exc()}")  # Debug log
        return jsonify({"error": f"Failed to list scenes: {str(e)}"}), 500


@APP.route("/api/list_pointclouds", methods=["GET"])
def api_list_pointclouds():
    scene_dir = request.args.get("scene_dir", "").strip()
    if not scene_dir:
        return jsonify({"error": "scene_dir is required"}), 400
    if not os.path.isdir(scene_dir):
        return jsonify({"error": f"Scene directory not found: {scene_dir}"}), 404
    try:
        pcs = list_pointclouds(scene_dir)
        return jsonify({"count": len(pcs), "pointclouds": pcs})
    except Exception as e:
        return jsonify({"error": f"Failed to list point clouds: {str(e)}"}), 500


@APP.route("/api/load_pointcloud", methods=["GET"])
def api_load_pointcloud():
    path = request.args.get("path", "").strip()
    sample_str = request.args.get("sample", "").strip()
    sample_n = None
    if sample_str:
        try:
            sample_n = int(sample_str)
        except ValueError:
            return jsonify({"error": "sample must be int"}), 400

    if not path:
        return jsonify({"error": "path is required"}), 400

    try:
        points, colors = load_ply(path, sample=sample_n)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load PLY: {e}"}), 500

    return jsonify(
        {
            "num_points": len(points),
            "points": points.tolist(),
            "colors": colors.tolist(),
            "path": path,
        }
    )


def main():
    port = int(os.environ.get("PORT", "5000"))
    APP.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()

