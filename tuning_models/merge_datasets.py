#!/usr/bin/env python3
"""
Script to merge multiple graded datasets created by generate_dataset_pt2.
Automatically reads model metadata from model_used_for_dataset_generation.txt files.
"""

import argparse
import json
import os
from tuning_models.utils.grading_dataset import merge_graded_datasets

def main():
    parser = argparse.ArgumentParser(description="Merge multiple graded datasets with model metadata")
    parser.add_argument("--dataset_paths", nargs='+', required=True,
                       help="Paths to graded dataset JSONL files to merge")
    parser.add_argument("--output_path", required=True,
                       help="Path to save the merged dataset")
    parser.add_argument("--no_model_metadata", action="store_true",
                       help="Skip reading model metadata files")
    
    args = parser.parse_args()
    
    # Merge the datasets
    merge_graded_datasets(
        dataset_paths=args.dataset_paths,
        output_path=args.output_path,
        include_model_metadata=not args.no_model_metadata
    )
    
    print(f"Successfully merged {len(args.dataset_paths)} datasets into {args.output_path}")

if __name__ == "__main__":
    main()
