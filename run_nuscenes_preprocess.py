#!/usr/bin/env python3
"""
Run NuScenes preprocessing for filtered scenes.
"""

import subprocess
import sys
import os

def main():
    # Read filtered scene IDs
    with open('/root/drivestudio-coding/data/nuscenes_filtered_scenes.txt', 'r') as f:
        scene_ids = [int(line.strip()) for line in f.readlines()[1:]]
    
    print(f"Total scenes to process: {len(scene_ids)}")
    
    # Prepare command
    cmd = [
        'python', 'datasets/preprocess.py',
        '--data_root', '/root/autodl-tmp/nuScenes/raw',
        '--target_dir', '/root/autodl-tmp/nuScenes/preprocess',
        '--dataset', 'nuscenes',
        '--split', 'v1.0-trainval',
        '--interpolate_N', '4',
        '--workers', '2',
        '--process_keys', 'images', 'lidar', 'calib', 'objects', 'dynamic_masks',
        '--scene_ids'
    ] + [str(sid) for sid in scene_ids]
    
    print(f"Running preprocessing for {len(scene_ids)} scenes...")
    print(f"Command: {' '.join(cmd[:15])} ... (truncated)")
    
    # Run the command with PYTHONPATH set
    env = os.environ.copy()
    env['PYTHONPATH'] = '/root/drivestudio-coding'
    result = subprocess.run(cmd, cwd='/root/drivestudio-coding', env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Preprocessing completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # 显示最后500字符
    else:
        print(f"Preprocessing failed with exit code {result.returncode}")
        if result.stderr:
            print("Error output:", result.stderr[-1000:])  # 显示最后1000字符
        if result.stdout:
            print("Standard output:", result.stdout[-1000:])  # 显示最后1000字符
        sys.exit(1)

if __name__ == "__main__":
    main()

