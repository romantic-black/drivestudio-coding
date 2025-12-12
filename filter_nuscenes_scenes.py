#!/usr/bin/env python3
"""
Filter NuScenes scenes by weather and time of day.
This script filters scenes that are:
- Clear weather (no rain, no snow, no fog)
- Daytime (6 AM to 8 PM local time)
"""

import os
from datetime import datetime
from nuscenes.nuscenes import NuScenes

def filter_scenes(data_root, split='v1.0-trainval'):
    """Filter scenes by weather and time of day."""
    nusc = NuScenes(version=split, dataroot=data_root, verbose=True)
    
    selected_scene_ids = []
    
    print(f"Total scenes: {len(nusc.scene)}")
    print("Filtering scenes for clear weather and daytime...")
    
    for scene_idx, scene in enumerate(nusc.scene):
        # Get scene metadata
        scene_token = scene['token']
        scene_record = nusc.get('scene', scene_token)
        
        # Get log information
        log_token = scene_record['log_token']
        log_record = nusc.get('log', log_token)
        location = log_record.get('location', '').lower()
        
        # Get scene description
        description = scene_record.get('description', '').lower()
        
        # Check weather conditions from description
        # Exclude scenes with explicit weather mentions
        weather_ok = True
        weather_keywords = ['rain', 'rainy', 'snow', 'snowy', 'fog', 'foggy', 'wet']
        for keyword in weather_keywords:
            if keyword in description:
                weather_ok = False
                break
        
        # Check time of day using timestamp
        # NuScenes timestamps are in microseconds since epoch
        first_sample_token = scene_record['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        timestamp_us = first_sample['timestamp']
        
        # Convert to datetime (UTC)
        timestamp_s = timestamp_us / 1e6
        dt_utc = datetime.utcfromtimestamp(timestamp_s)
        
        # NuScenes data was collected in Singapore (UTC+8) and Boston (UTC-5)
        # For simplicity, we'll check if hour is between 6 and 20 (8 PM) in UTC
        # This roughly corresponds to daytime in both locations
        # Singapore: UTC+8, so UTC 22:00-12:00 = local 6:00-20:00
        # Boston: UTC-5, so UTC 11:00-01:00 = local 6:00-20:00
        # We'll use a wider range: UTC 2:00-18:00 covers daytime in both locations
        hour_utc = dt_utc.hour
        time_ok = (hour_utc >= 2 and hour_utc < 18)
        
        # Additional check: exclude if description explicitly mentions night
        if 'night' in description or 'nighttime' in description:
            time_ok = False
        
        if weather_ok and time_ok:
            selected_scene_ids.append(scene_idx)
            if len(selected_scene_ids) <= 10:  # Print first 10 for verification
                print(f"  Scene {scene_idx}: {description[:60] if description else 'N/A'} (UTC hour: {hour_utc})")
    
    print(f"\nSelected {len(selected_scene_ids)} scenes out of {len(nusc.scene)} total scenes")
    return selected_scene_ids

if __name__ == "__main__":
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/nuScenes/raw"
    split = sys.argv[2] if len(sys.argv) > 2 else "v1.0-trainval"
    
    scene_ids = filter_scenes(data_root, split)
    
    # Save scene IDs to a file
    output_file = "/root/drivestudio-coding/data/nuscenes_filtered_scenes.txt"
    with open(output_file, 'w') as f:
        f.write("scene_id\n")
        for scene_id in scene_ids:
            f.write(f"{scene_id}\n")
    
    print(f"\nFiltered scene IDs saved to: {output_file}")
    print(f"Scene IDs: {scene_ids[:20]}..." if len(scene_ids) > 20 else f"Scene IDs: {scene_ids}")

