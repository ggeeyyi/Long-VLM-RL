#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import argparse


def find_model_directories(base_path):
    """Find all global_step directories containing actor subdirectories."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: Directory {base_path} does not exist")
        return []
    
    model_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('global_step_'):
            actor_path = item / 'actor'
            if actor_path.exists() and actor_path.is_dir():
                model_dirs.append(actor_path)
    
    return sorted(model_dirs)


def merge_model(actor_path, dry_run=False):
    """Merge a single model using the model_merger.py script."""
    print(f"Merging model: {actor_path}")
    
    command = [
        sys.executable,  # Use the same Python interpreter
        "scripts/model_merger.py",
        "--local_dir", str(actor_path)
    ]
    
    if dry_run:
        print(f"Would run: {' '.join(command)}")
        return True
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ Successfully merged: {actor_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to merge {actor_path}")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge all models in specified checkpoint directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually running")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue merging other models if one fails")
    args = parser.parse_args()
    
    # Define the target directories
    target_dirs = [
        "checkpoints/easy_r1/qwen2_5_vl_3b_7b_lmms_scienceqa",
        "checkpoints/easy_r1/qwen2_5_vl_3b_32b_lmms_scienceqa"
    ]
    
    all_model_dirs = []
    for target_dir in target_dirs:
        print(f"\nScanning directory: {target_dir}")
        model_dirs = find_model_directories(target_dir)
        if model_dirs:
            print(f"Found {len(model_dirs)} model directories:")
            for model_dir in model_dirs:
                print(f"  - {model_dir}")
            all_model_dirs.extend(model_dirs)
        else:
            print(f"No model directories found in {target_dir}")
    
    if not all_model_dirs:
        print("\nNo model directories found to merge!")
        return
    
    print(f"\nTotal models to merge: {len(all_model_dirs)}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
    
    # Merge all models
    successful = 0
    failed = 0
    
    for model_dir in all_model_dirs:
        success = merge_model(model_dir, dry_run=args.dry_run)
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print(f"\nStopping due to error. Use --continue-on-error to continue despite failures.")
                break
    
    print(f"\n=== Summary ===")
    print(f"Successfully merged: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(all_model_dirs)}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
