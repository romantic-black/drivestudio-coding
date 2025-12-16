#!/usr/bin/env python3
"""
演示 show_dirs 中的深度图内容
"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def demo_show_dirs():
    # 查找 show_dirs 目录
    show_dirs_base = "/root/drivestudio-coding/third_party/EVolSplat/preprocess/metric3d/show_dirs"
    
    if not os.path.exists(show_dirs_base):
        print(f"Error: show_dirs not found at {show_dirs_base}")
        return
    
    # 查找最新的运行目录
    config_dirs = [d for d in os.listdir(show_dirs_base) if os.path.isdir(os.path.join(show_dirs_base, d))]
    if not config_dirs:
        print("No config directories found in show_dirs")
        return
    
    print(f"Found config directories: {config_dirs}")
    
    # 使用第一个配置目录（通常是 vit.raft5.giant2）
    config_dir = config_dirs[0]
    config_path = os.path.join(show_dirs_base, config_dir)
    
    # 查找最新的时间戳目录
    timestamp_dirs = sorted([d for d in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, d))])
    if not timestamp_dirs:
        print(f"No timestamp directories found in {config_path}")
        return
    
    latest_dir = timestamp_dirs[-1]
    latest_path = os.path.join(config_path, latest_dir)
    
    print(f"\nLatest run directory: {latest_dir}")
    print(f"Full path: {latest_path}")
    
    # 查找所有 .npy 文件
    npy_files = sorted([f for f in os.listdir(latest_path) if f.endswith('.npy')])
    log_files = [f for f in os.listdir(latest_path) if f.endswith('.log')]
    
    print(f"\nFound {len(npy_files)} depth map files (.npy)")
    if log_files:
        print(f"Found {len(log_files)} log files")
    
    if not npy_files:
        print("No depth map files found!")
        return
    
    # 分析第一个深度图文件
    sample_file = npy_files[0]
    sample_path = os.path.join(latest_path, sample_file)
    
    print(f"\nAnalyzing sample file: {sample_file}")
    depth = np.load(sample_path)
    
    print(f"  Shape: {depth.shape}")
    print(f"  Dtype: {depth.dtype}")
    
    # 检查 NaN 和 Inf
    nan_count = np.isnan(depth).sum()
    inf_count = np.isinf(depth).sum()
    valid_mask = np.isfinite(depth) & (depth > 0)
    valid_count = valid_mask.sum()
    
    print(f"  NaN pixels: {nan_count} ({nan_count/depth.size*100:.2f}%)")
    print(f"  Inf pixels: {inf_count} ({inf_count/depth.size*100:.2f}%)")
    print(f"  Valid pixels (finite and > 0): {valid_count} ({valid_count/depth.size*100:.2f}%)")
    
    if valid_count > 0:
        valid_depth = depth[valid_mask]
        print(f"  Valid depth range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}] meters")
        print(f"  Valid depth mean: {valid_depth.mean():.2f} meters")
        print(f"  Valid depth median: {np.median(valid_depth):.2f} meters")
    else:
        print("  WARNING: No valid depth values found!")
    
    # 尝试可视化（如果有有效值）
    if valid_count > 0:
        # 创建可视化
        depth_vis = depth.copy()
        depth_vis[~valid_mask] = 0
        
        # 归一化到 0-255 用于显示
        if depth_vis.max() > depth_vis.min():
            depth_normalized = ((depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
        
        # 应用 colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # 保存可视化
        vis_output = os.path.join(latest_path, f"{os.path.splitext(sample_file)[0]}_vis.png")
        cv2.imwrite(vis_output, depth_colored)
        print(f"\nSaved visualization to: {vis_output}")
        
        # 显示统计信息
        print(f"\nVisualization saved successfully!")
        print(f"  Output file: {vis_output}")
    else:
        print("\nCannot create visualization: no valid depth values")
    
    # 列出所有文件
    print(f"\nAll files in {latest_dir}:")
    for f in sorted(os.listdir(latest_path)):
        fpath = os.path.join(latest_path, f)
        size = os.path.getsize(fpath)
        print(f"  {f:30s} {size:>10,} bytes")

if __name__ == "__main__":
    demo_show_dirs()

