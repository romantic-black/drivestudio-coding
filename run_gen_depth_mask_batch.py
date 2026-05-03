#!/usr/bin/env python3
"""
批量运行 gen_nuscenes_depth_mask.py，为多场景、多相机生成深度图与 mask。

Waymo 等预处理目录布局与 NuScenes 分支一致（images/、intrinsics/）时，
Metric3D 请使用 --dataset nuscenes（见 mono/utils/custom_data.py）。
"""

import argparse
import os
import subprocess
import sys


def log(msg, level="info"):
    prefix = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "error": "[ERROR]",
        "warning": "[WARNING]",
    }.get(level, "[INFO]")
    print(f"{prefix} {msg}")


def _parse_cam_ids(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("cam_ids is empty")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run gen_nuscenes_depth_mask.py per scene and camera."
    )
    parser.add_argument(
        "--base-dir",
        default="/root/autodl-tmp/waymo",
        help="Root directory containing one subdirectory per scene.",
    )
    parser.add_argument(
        "--cam-ids",
        default="0,1,2",
        help="Comma-separated camera ids (e.g. 0,1,2). Each gets a separate subprocess call.",
    )
    parser.add_argument(
        "--dataset",
        default="nuscenes",
        help="Passed to gen_nuscenes_depth_mask --dataset (Waymo preprocess layout -> nuscenes).",
    )
    parser.add_argument(
        "--scene-filter-file",
        default=None,
        help="Optional: text file with scene folder names (skip first line header); "
        "same format as data/nuscenes_filtered_scenes.txt.",
    )
    parser.add_argument("--gen_depth", action="store_true")
    parser.add_argument("--gen_semantic", action="store_true")
    parser.add_argument("--gen_sky_mask", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir
    cam_ids = _parse_cam_ids(args.cam_ids)

    if not os.path.exists(base_dir):
        log(f"Error: Base directory not found: {base_dir}", "error")
        sys.exit(1)

    filtered_scene_ids = set()
    if args.scene_filter_file:
        if not os.path.isfile(args.scene_filter_file):
            log(f"Error: scene filter file not found: {args.scene_filter_file}", "error")
            sys.exit(1)
        with open(args.scene_filter_file, "r") as f:
            filtered_scene_ids = {
                str(int(line.strip())).zfill(3)
                for line in f.readlines()[1:]
                if line.strip()
            }
        log(f"Loaded {len(filtered_scene_ids)} scene IDs from {args.scene_filter_file}", "info")

    all_scene_dirs = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    if filtered_scene_ids:
        scene_dirs = [d for d in all_scene_dirs if d in filtered_scene_ids]
        skipped = len(all_scene_dirs) - len(scene_dirs)
        if skipped > 0:
            log(f"Filtered out {skipped} scenes not in the filter list", "info")
    else:
        scene_dirs = all_scene_dirs

    if len(scene_dirs) == 0:
        log(f"Error: No scene directories under {base_dir} (after filter)", "error")
        sys.exit(1)

    log(f"Found {len(scene_dirs)} scenes, cam_ids={cam_ids}", "info")

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "third_party",
        "EVolSplat",
        "preprocess",
        "gen_nuscenes_depth_mask.py",
    )
    if not os.path.exists(script_path):
        log(f"Error: Script not found: {script_path}", "error")
        sys.exit(1)

    gen_depth = args.gen_depth
    gen_semantic = args.gen_semantic
    gen_sky_mask = args.gen_sky_mask
    if not (gen_depth or gen_semantic or gen_sky_mask):
        log("Warning: No task specified. Use --gen_depth, --gen_semantic, or --gen_sky_mask", "warning")
        log("Running all tasks by default...", "warning")
        gen_depth = True
        gen_semantic = True
        gen_sky_mask = True

    depth_gpu_id = os.getenv("DEPTH_GPU_ID", "0")
    semantic_gpu_id = os.getenv("SEMANTIC_GPU_ID", "0")

    failed_jobs = []
    successful_jobs = []

    total_jobs = len(scene_dirs) * len(cam_ids)
    job_idx = 0
    for scene_dir_name in scene_dirs:
        scene_path = os.path.join(base_dir, scene_dir_name)
        images_dir = os.path.join(scene_path, "images")
        if not os.path.exists(images_dir):
            log(f"Skipping {scene_dir_name}: no images directory", "warning")
            continue
        for cam_id in cam_ids:
            job_idx += 1
            cmd = [
                sys.executable,
                script_path,
                "--scene_dir",
                scene_path,
                "--cam_id",
                str(cam_id),
                "--dataset",
                args.dataset,
            ]
            if gen_depth:
                cmd.extend(["--gen_depth", "--depth_gpu_id", depth_gpu_id])
            if gen_semantic:
                cmd.extend(["--gen_semantic", "--semantic_gpu_id", semantic_gpu_id])
            if gen_sky_mask:
                cmd.append("--gen_sky_mask")

            log(
                f"Processing {scene_dir_name} cam={cam_id} ({job_idx}/{total_jobs})...",
                "info",
            )
            env = os.environ.copy()
            if "METRIC3D_PATH" not in env:
                env["METRIC3D_PATH"] = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "third_party",
                    "EVolSplat",
                    "preprocess",
                    "metric3d",
                )
            try:
                result = subprocess.run(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired:
                failed_jobs.append((scene_dir_name, cam_id, "Timeout"))
                log(f"{scene_dir_name} cam={cam_id} timed out", "error")
                continue
            except Exception as e:
                failed_jobs.append((scene_dir_name, cam_id, str(e)))
                log(f"{scene_dir_name} cam={cam_id} error: {e}", "error")
                continue

            if result.returncode == 0:
                successful_jobs.append((scene_dir_name, cam_id))
                log(f"{scene_dir_name} cam={cam_id} completed", "success")
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                failed_jobs.append((scene_dir_name, cam_id, error_msg or ""))
                log(
                    f"{scene_dir_name} cam={cam_id} failed rc={result.returncode}",
                    "error",
                )
                if error_msg:
                    err = error_msg[-500:] if len(error_msg) > 500 else error_msg
                    log(f"Error output:\n{err}", "error")

    log(f"\nCompleted jobs: {len(successful_jobs)}", "success")
    if failed_jobs:
        log(f"Failed jobs: {len(failed_jobs)}", "error")
        for scene, cam, err in failed_jobs[:10]:
            log(f"  - {scene} cam={cam}: {(err or '')[:100]}", "error")
    else:
        log("All jobs processed successfully!", "success")


if __name__ == "__main__":
    main()
