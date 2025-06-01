#!/usr/bin/env python3
"""Batch process all videos for the calibration challenge"""

import os
import subprocess
import numpy as np
from app import process_video_with_visualization
import time

def process_single_video(video_num, video_dir, output_dir):
    """Process a single video and save predictions."""
    if video_num < 5:
        video_path = f"labeled/{video_num}.hevc"
        ground_truth_path = f"labeled/{video_num}.txt"
    else:
        video_path = f"unlabeled/{video_num}.hevc"
        ground_truth_path = None
    
    output_path = f"{output_dir}/{video_num}.txt"
    
    start_time = time.time()
    print(f"Processing video {video_num}: {video_path}...")
    
    try:
        predictions = process_video_with_visualization(
            video_path,
            output_video_path=None,
            output_data_path=output_path,
            layout='grid',
            show_live=False,
            save_video=False,
            ground_truth_path=ground_truth_path
        )
        
        elapsed = time.time() - start_time
        
        if predictions is not None and len(predictions) > 0:
            print(f"  Video {video_num}: {len(predictions)} frames in {elapsed:.1f}s")
            print(f"    Avg pitch: {np.degrees(np.mean(predictions[:, 0])):+.2f}°, yaw: {np.degrees(np.mean(predictions[:, 1])):+.2f}°")
            return True
        else:
            print(f"  Video {video_num}: Failed to process")
            return False
    except Exception as e:
        print(f"  Video {video_num}: Error - {e}")
        return False

def main():
    # Create output directory
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Comma.ai Calibration Challenge - Batch Processing")
    print("="*60)
    
    # Process all videos sequentially
    success_count = 0
    total_start = time.time()
    
    # Process labeled videos (0-4)
    print("\nProcessing labeled videos...")
    for i in range(5):
        if process_single_video(i, "labeled", output_dir):
            success_count += 1
    
    # Process unlabeled videos (5-9)
    print("\nProcessing unlabeled videos...")
    for i in range(5, 10):
        if process_single_video(i, "unlabeled", output_dir):
            success_count += 1
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/10 videos in {total_elapsed:.1f}s")
    
    # Run evaluation if we processed the labeled videos
    if success_count >= 5:
        print(f"\n{'='*60}")
        print("Running evaluation on labeled videos...")
        try:
            result = subprocess.run(
                ["python", "eval.py", output_dir],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            
            # Also check if we can run on unlabeled
            unlabeled_complete = all(os.path.exists(f"{output_dir}/{i}.txt") for i in range(5, 10))
            if unlabeled_complete:
                print("\nAll predictions complete! Files ready for submission:")
                for i in range(5, 10):
                    print(f"  {output_dir}/{i}.txt")
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")

if __name__ == "__main__":
    main()