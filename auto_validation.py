#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import argparse


def find_merged_models(base_path):
    """Find all merged model directories (huggingface folders) in global_step directories."""
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: Directory {base_path} does not exist")
        return []
    
    merged_models = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('global_step_'):
            huggingface_path = item / 'actor' / 'huggingface'
            if huggingface_path.exists() and huggingface_path.is_dir():
                merged_models.append(huggingface_path)
    
    return sorted(merged_models)


def validate_model(model_path, dry_run=False):
    """Validate a single model using the validation shell script."""
    print(f"Validating model: {model_path}")
    
    # Use the same model path for both actor and rollout
    rollout_model_path = model_path
    
    command = [
        "bash",
        "shell_scripts/val_only/qwen2_5_vl_3b_scienceqa.sh",
        str(model_path),      # ACTOR_MODEL_PATH
        str(rollout_model_path)  # ROLLOUT_MODEL_PATH
    ]
    
    if dry_run:
        print(f"Would run: {' '.join(command)}")
        return True
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✓ Successfully validated: {model_path}")
            return True
        else:
            print(f"✗ Validation failed for {model_path} (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ Error validating {model_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate all merged models in specified checkpoint directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually running")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue validating other models if one fails")
    parser.add_argument("--filter-steps", type=str, nargs="+", help="Only validate specific global steps (e.g., --filter-steps 39 35)")
    args = parser.parse_args()
    
    # Define the target directories
    target_dirs = [
        "checkpoints/easy_r1/qwen2_5_vl_3b_7b_lmms_scienceqa",
        "checkpoints/easy_r1/qwen2_5_vl_3b_32b_lmms_scienceqa"
    ]
    
    all_merged_models = []
    for target_dir in target_dirs:
        print(f"\nScanning directory: {target_dir}")
        merged_models = find_merged_models(target_dir)
        if merged_models:
            print(f"Found {len(merged_models)} merged model directories:")
            for model_dir in merged_models:
                # Extract global step number for filtering
                step_match = model_dir.parent.parent.name  # global_step_XX
                if args.filter_steps:
                    step_num = step_match.replace('global_step_', '')
                    if step_num not in args.filter_steps:
                        print(f"  - {model_dir} [SKIPPED - not in filter]")
                        continue
                
                print(f"  - {model_dir}")
                all_merged_models.append(model_dir)
        else:
            print(f"No merged model directories found in {target_dir}")
    
    if not all_merged_models:
        print("\nNo merged model directories found to validate!")
        return
    
    print(f"\nTotal models to validate: {len(all_merged_models)}")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
    
    # Validate all models
    successful = 0
    failed = 0
    
    for model_dir in all_merged_models:
        print(f"\n{'='*80}")
        success = validate_model(model_dir, dry_run=args.dry_run)
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print(f"\nStopping due to error. Use --continue-on-error to continue despite failures.")
                break
    
    print(f"\n{'='*80}")
    print(f"=== Validation Summary ===")
    print(f"Successfully validated: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(all_merged_models)}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
