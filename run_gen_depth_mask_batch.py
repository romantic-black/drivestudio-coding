#!/usr/bin/env python3
"""
批量运行 gen_nuscenes_depth_mask.py 脚本为所有场景生成深度图和mask
"""

import os
import sys
import subprocess
from pathlib import Path

def log(msg, level="info"):
    """简单的日志输出"""
    prefix = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "error": "[ERROR]",
        "warning": "[WARNING]"
    }.get(level, "[INFO]")
    print(f"{prefix} {msg}")

def main():
    # 预处理后的数据目录
    base_dir = "/root/autodl-tmp/nuScenes/preprocess/trainval"
    
    if not os.path.exists(base_dir):
        log(f"Error: Base directory not found: {base_dir}", "error")
        sys.exit(1)
    
    # 读取过滤后的场景ID列表（与run_nuscenes_preprocess.py保持一致）
    filtered_scenes_file = '/root/drivestudio-coding/data/nuscenes_filtered_scenes.txt'
    filtered_scene_ids = set()
    if os.path.exists(filtered_scenes_file):
        with open(filtered_scenes_file, 'r') as f:
            # 跳过第一行标题，读取场景ID并格式化为3位数字（与预处理脚本一致）
            filtered_scene_ids = {str(int(line.strip())).zfill(3) for line in f.readlines()[1:] if line.strip()}
        log(f"Loaded {len(filtered_scene_ids)} filtered scene IDs from {filtered_scenes_file}", "info")
    else:
        log(f"Warning: Filtered scenes file not found: {filtered_scenes_file}", "warning")
        log("Will process all scenes in the directory", "warning")
    
    # 获取所有场景目录，并过滤（如果过滤列表存在）
    all_scene_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    if filtered_scene_ids:
        # 只处理过滤列表中的场景
        scene_dirs = [d for d in all_scene_dirs if d in filtered_scene_ids]
        skipped = len(all_scene_dirs) - len(scene_dirs)
        if skipped > 0:
            log(f"Filtered out {skipped} scenes not in the filtered list", "info")
    else:
        scene_dirs = all_scene_dirs
    
    if len(scene_dirs) == 0:
        log(f"Error: No scene directories found in {base_dir} (after filtering)", "error")
        sys.exit(1)
    
    log(f"Found {len(scene_dirs)} scenes to process", "info")
    
    # 脚本路径
    script_path = "/root/drivestudio-coding/third_party/EVolSplat/preprocess/gen_nuscenes_depth_mask.py"
    
    if not os.path.exists(script_path):
        log(f"Error: Script not found: {script_path}", "error")
        sys.exit(1)
    
    # 处理参数
    gen_depth = '--gen_depth' in sys.argv
    gen_semantic = '--gen_semantic' in sys.argv
    gen_sky_mask = '--gen_sky_mask' in sys.argv
    
    if not (gen_depth or gen_semantic or gen_sky_mask):
        log("Warning: No task specified. Use --gen_depth, --gen_semantic, or --gen_sky_mask", "warning")
        log("Running all tasks by default...", "warning")
        gen_depth = True
        gen_semantic = True
        gen_sky_mask = True
    
    # GPU IDs (可以从环境变量获取，或使用默认值)
    depth_gpu_id = os.getenv('DEPTH_GPU_ID', '0')
    semantic_gpu_id = os.getenv('SEMANTIC_GPU_ID', '0')
    
    # 处理每个场景
    failed_scenes = []
    successful_scenes = []
    
    total = len(scene_dirs)
    for idx, scene_dir_name in enumerate(scene_dirs, 1):
        scene_path = os.path.join(base_dir, scene_dir_name)
        
        # 检查场景目录是否有 images 目录
        images_dir = os.path.join(scene_path, 'images')
        if not os.path.exists(images_dir):
            log(f"Skipping {scene_dir_name}: no images directory", "warning")
            continue
        
        # 构建命令
        cmd = ['python', script_path, '--scene_dir', scene_path]
        
        if gen_depth:
            cmd.extend(['--gen_depth', '--depth_gpu_id', depth_gpu_id])
        if gen_semantic:
            cmd.extend(['--gen_semantic', '--semantic_gpu_id', semantic_gpu_id])
        if gen_sky_mask:
            cmd.append('--gen_sky_mask')
        
        # 运行命令
        log(f"Processing scene {scene_dir_name} ({idx}/{total})...", "info")
        
        try:
            result = subprocess.run(
                cmd,
                cwd='/root/drivestudio-coding',
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per scene
            )
            
            if result.returncode == 0:
                successful_scenes.append(scene_dir_name)
                log(f"Scene {scene_dir_name} completed", "success")
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                failed_scenes.append((scene_dir_name, error_msg))
                log(f"Scene {scene_dir_name} failed with return code {result.returncode}", "error")
                if error_msg:
                    # 显示更多错误信息（最多500字符）
                    error_display = error_msg[-500:] if len(error_msg) > 500 else error_msg
                    log(f"Error output:\n{error_display}", "error")
                if result.stdout:
                    # 也显示标准输出（可能包含有用信息）
                    stdout_display = result.stdout[-300:] if len(result.stdout) > 300 else result.stdout
                    log(f"Stdout:\n{stdout_display}", "warning")
        except subprocess.TimeoutExpired:
            failed_scenes.append((scene_dir_name, "Timeout"))
            log(f"Scene {scene_dir_name} timed out", "error")
        except Exception as e:
            failed_scenes.append((scene_dir_name, str(e)))
            log(f"Scene {scene_dir_name} error: {e}", "error")
    
    # 总结
    log(f"\nCompleted: {len(successful_scenes)} scenes", "success")
    if failed_scenes:
        log(f"Failed: {len(failed_scenes)} scenes", "error")
        for scene, error in failed_scenes[:10]:  # 只显示前10个错误
            log(f"  - {scene}: {error[:100]}", "error")
    else:
        log("All scenes processed successfully!", "success")

if __name__ == "__main__":
    main()

