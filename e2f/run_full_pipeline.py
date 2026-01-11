"""
run_full_pipeline.py
Automates the full event dataset workflow: auto script, remove first images, unify, split, for train/test and new event types.
"""
import argparse
import sys
import os
from scripts.automated_script import process_all_operations
from scripts.remove_first_images_v2 import process_folders
from scripts.unify_datasets import unify_datasets_from_new_structure
from scripts.combine_and_split_datasets import reorganize_and_split_dataset_consistently

def generate_event_folder_name(desired_interval, use_time_lookup, high_low_ratio, gamma, r0=None):
    base = f"adjusted_event_frames_{desired_interval}ms"
    if use_time_lookup:
        base += f"_with_time_lookup_htl_{high_low_ratio}_gamma_{gamma}"
        if r0 is not None:
            base += f"_rational_r0_{r0}"
    return base

def run_pipeline(dataset_dir, output_adjusted_base_dir, output_unified_base_dir, output_split_base_dir, modes, steps, desired_interval, rgb_delta_time, use_time_lookup, high_low_ratio, gamma, operation, skip_existing_folder=False, r0=None):
    event_folder_name = generate_event_folder_name(desired_interval, use_time_lookup, high_low_ratio, gamma, r0=r0)
    for mode in modes:
        print(f"\n--- Processing {mode} set ---")
        # Step 1: Run auto script
        if 'auto' in steps:
            print(f"[auto] Generating event frames, interval {desired_interval}ms...")
            if operation == "event_only":
                from scripts.automated_script import add_event_frames
                add_event_frames(f"{dataset_dir}/{mode}", f"{output_adjusted_base_dir}/{mode}", desired_interval, mode, rgb_delta_time, use_time_lookup, high_low_ratio, gamma, event_folder_name, r0=r0)
            else:
                process_all_operations(f"{dataset_dir}/{mode}", f"{output_adjusted_base_dir}/{mode}", desired_interval, mode, rgb_delta_time, use_time_lookup, high_low_ratio, gamma, event_folder_name, r0=r0)
        # Step 2: Remove first images
        if 'remove_first' in steps:
            print(f"[remove_first] Removing first images in {output_adjusted_base_dir}/{mode}...")
            process_folders(
                os.path.join(output_adjusted_base_dir, mode),
                event_folder_base=event_folder_name
            )
        # Step 3: Unify datasets
        if 'unify' in steps:
            print(f"[unify] Unifying dataset in {output_unified_base_dir}/{mode}...")
            unify_datasets_from_new_structure(
                os.path.join(output_adjusted_base_dir, mode),
                os.path.join(output_unified_base_dir, mode),
                event_ms=desired_interval,
                event_folder_name=event_folder_name
            )
    # Step 4: Split datasets
    if 'split' in steps:
        print(f"[split] Splitting unified dataset...")
        from scripts.split_dataset import split_and_copy_dataset
        original_dataset_dir = os.path.join(output_unified_base_dir)
        target_dir = os.path.join(output_split_base_dir)
        split_and_copy_dataset(
            original_dataset_dir=original_dataset_dir,
            target_dir=target_dir,
            event_folder_name=event_folder_name,
            train_ratio=0.8,
            skip_existing_folder=skip_existing_folder
        )
    print(f"--- split set done ---\n")

def main():

    var_config = {
        "argv_override": True,
        "dataset_dir": "/media/CezeriDrive_12TB/GSS/Ozkan_4090_2/DSEC_original_datasets/",
        "output_adjusted_base_dir": "adjusted_dataset",
        "output_unified_base_dir": "unified_dataset",
        "output_split_base_dir": "final_dataset",
        "modes": ["train", "test"],
        # "steps": ["split"],
        "steps": ["auto", "remove_first", "unify", "split"],
        "operation": "event_only", # all or event_only
        "desired_interval": 50,
        "rgb_delta_time": 50,
        "use_time_lookup": True,
        "high_low_ratio": 10.0,
        "gamma": 2.5,
        "skip_existing_folder": True,
        "r0": 0.3 # for rational length calculation give value, or None to ignore and use normal power function
    }

    if not var_config.get("argv_override", False):
        parser = argparse.ArgumentParser(description="Run full event dataset pipeline.")
        parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
        parser.add_argument('--output_adjusted_base_dir', type=str, required=True, help='Path to the output adjusted base directory.')
        parser.add_argument('--output_unified_base_dir', type=str, required=True, help='Path to the output unified base directory.')
        parser.add_argument('--output_split_base_dir', type=str, required=True, help='Path to the output split base directory.')
        parser.add_argument('--modes', nargs='+', default=['train', 'test'], help='Dataset modes to process.')
        parser.add_argument('--steps', nargs='+', default=['auto', 'remove_first', 'unify', 'split'], choices=['auto', 'remove_first', 'unify', 'split'], help='Pipeline steps to run.')
        parser.add_argument('--operation', type=str, choices=['event_only', 'all'], default='event_only', help='Choose whether to process only event frames or perform all operations.')
        parser.add_argument('--desired_interval', type=int, default=50, help='Desired interval (in ms) for event frame processing.')
        parser.add_argument('--rgb_delta_time', type=float, default=None, help='Time window (in milliseconds) for matching frames.')
        parser.add_argument('--use_time_lookup', action='store_true', help='Use time lookup for frame matching.')
        parser.add_argument('--high_low_ratio', type=float, default=10.0, help='High to low frame ratio.')
        parser.add_argument('--gamma', type=float, default=2.0, help='Gamma value for power function.')
        parser.add_argument('--skip_existing_folder', action='store_true', help='Skip existing folders during splitting.')
        args = parser.parse_args()
        run_pipeline(
            args.dataset_dir,
            args.output_adjusted_base_dir,
            args.output_unified_base_dir,
            args.output_split_base_dir,
            args.modes,
            args.steps,
            args.desired_interval,
            args.rgb_delta_time,
            args.use_time_lookup,
            args.high_low_ratio,
            args.gamma,
            args.operation,
            args.skip_existing_folder
    )
    
    else:
        run_pipeline(
            var_config["dataset_dir"],
            var_config["output_adjusted_base_dir"],
            var_config["output_unified_base_dir"],
            var_config["output_split_base_dir"],
            var_config["modes"],
            var_config["steps"],
            var_config["desired_interval"],
            var_config["rgb_delta_time"],
            var_config["use_time_lookup"],
            var_config["high_low_ratio"],
            var_config["gamma"],
            var_config["operation"],
            var_config["skip_existing_folder"],
            r0=var_config["r0"]
        )

if __name__ == "__main__":
    main()
