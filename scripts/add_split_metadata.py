#!/usr/bin/env python3
"""
Add split metadata to graded datasets to preserve train/test boundaries when merging.

This script applies the same splitting logic that was used during dataset generation
to assign 'split' metadata to each entry in the dataset.

Usage:
    # Single dataset
    python add_split_metadata.py --dataset_path /path/to/dataset.jsonl
    
    # Multiple datasets with custom parameters
    python add_split_metadata.py --dataset_paths /path/to/model_A/dataset.jsonl /path/to/model_B/dataset.jsonl --train_end_p 0.7 --eval_start_p 0.75 --calibration_p 0.05 --seed 42
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamics_project.tuning_models.utils.grading_dataset import add_split_metadata_to_dataset


def main():
    parser = argparse.ArgumentParser(description="Add split metadata to graded datasets")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_path", type=str, help="Path to a single dataset file")
    group.add_argument("--dataset_paths", nargs="+", help="Paths to multiple dataset files")
    
    # Split parameters - these must match what was used during dataset generation
    parser.add_argument("--train_end_p", type=float, default=0.7, 
                       help="Training set end proportion (default: 0.7)")
    parser.add_argument("--eval_start_p", type=float, default=0.75,
                       help="Evaluation set start proportion (default: 0.75)")
    parser.add_argument("--calibration_p", type=float, default=0.05,
                       help="Calibration set proportion (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("--disable_preshuffle", action="store_true",
                       help="Disable pre-shuffling (default: False)")
    
    args = parser.parse_args()
    
    if args.disable_preshuffle:
        input("Unexpected behavior: disable_preshuffle. Are you sure?")

    # Determine which datasets to process
    if args.dataset_path:
        dataset_paths = [args.dataset_path]
    else:
        dataset_paths = args.dataset_paths
    
    # Validate that all files exist
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    print(f"Processing {len(dataset_paths)} dataset(s) with parameters:")
    print(f"  train_end_p: {args.train_end_p}")
    print(f"  eval_start_p: {args.eval_start_p}")  
    print(f"  calibration_p: {args.calibration_p}")
    print(f"  seed: {args.seed}")
    print(f"  disable_preshuffle: {args.disable_preshuffle}")
    print()
    
    # Process each dataset
    for dataset_path in dataset_paths:
        print(f"Processing {dataset_path}")
        import shutil
        backup_path = dataset_path + ".backup"
        print(f"Creating backup {backup_path}")
        shutil.copy2(dataset_path, backup_path)
        try:
            add_split_metadata_to_dataset(
                dataset_path=dataset_path,
                eval_start_p=args.eval_start_p,
                train_end_p=args.train_end_p,
                calibration_p=args.calibration_p,
                seed=args.seed,
                disable_preshuffle=args.disable_preshuffle
            )
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            sys.exit(1)
    
    print(f"\nSuccessfully added split metadata to {len(dataset_paths)} dataset(s)")
    print("These datasets can now be safely merged while preserving train/test boundaries.")


if __name__ == "__main__":
    main()
