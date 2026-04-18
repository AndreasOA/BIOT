#!/usr/bin/env python3

import os
import glob
from datetime import datetime

def list_checkpoints():
    """List all available checkpoints in wandb_checkpoints directory"""
    checkpoint_dir = "wandb_checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found!")
        return
    
    print("Available Checkpoints:")
    print("=" * 80)
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "**", "*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    for i, ckpt_path in enumerate(checkpoint_files, 1):
        # Get file info
        rel_path = os.path.relpath(ckpt_path)
        file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
        
        # Extract run name and epoch from path
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            run_name = parts[1]
            filename = parts[-1]
            
            # Try to extract epoch from filename
            epoch_info = ""
            if "epoch=" in filename:
                try:
                    epoch_part = filename.split("epoch=")[1].split("-")[0]
                    epoch_info = f" (Epoch {epoch_part})"
                except:
                    pass
        else:
            run_name = "Unknown"
            epoch_info = ""
        
        print(f"{i:2d}. {rel_path}")
        print(f"    Run: {run_name}{epoch_info}")
        print(f"    Size: {file_size:.1f} MB")
        print(f"    Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Show example usage
    print("=" * 80)
    print("Example Usage:")
    print()
    print("To evaluate a checkpoint, use the evaluate_model.py script:")
    print()
    
    if checkpoint_files:
        example_path = os.path.relpath(checkpoint_files[0])
        print(f"python evaluate_model.py \\")
        print(f"    --checkpoint_path \"{example_path}\" \\")
        print(f"    --dataset TUEV \\")
        print(f"    --batch_size 512 \\")
        print(f"    --in_channels 12 \\")
        print(f"    --n_classes 1 \\")
        print(f"    --resampling_rate 200 \\")
        print(f"    --secondsBeforeEvent 2 \\")
        print(f"    --secondsAfterEvent 2")
        print()
        print("Adjust the parameters based on your model configuration.")


if __name__ == "__main__":
    list_checkpoints() 