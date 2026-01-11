import os
import shutil
import subprocess
from tqdm import tqdm
import argparse
from pathlib import Path
from .utils.eventreader import EventReader
import hdf5plugin
import cv2
from .utils.functions import *

# python e2f/automated_script.py --output_base_dir test_dataset_02_13/rgb_mask/ --mode test --operation all --desired_interval 50 --rgb_delta_time 50

def add_event_frames(dataset_dir, output_base_dir, desired_interval=50, mode="train", rgb_delta_time=None, use_time_lookup=False, high_low_ratio=10.0, gamma=2.0, event_folder_name=None, r0=None):
    segmentation_dir = os.path.join(dataset_dir, f"{mode}_semantic_segmentation")
    # Ensure segmentation directory exists
    if not os.path.exists(segmentation_dir):
        raise FileNotFoundError(f"Segmentation directory '{segmentation_dir}' does not exist.")

    folders = [f for f in os.listdir(segmentation_dir) if os.path.isdir(os.path.join(segmentation_dir, f))]
    
    events_dir = os.path.join(dataset_dir, f"{mode}_events")
    if event_folder_name is None:
        adjusted_event_base_name = f"adjusted_event_frames_{desired_interval}ms"
        if use_time_lookup:
            adjusted_event_base_name += f"_with_time_lookup_htl_{high_low_ratio}_gamma_{gamma}"
    else:
        adjusted_event_base_name = event_folder_name

    adjusted_event_base = os.path.join(output_base_dir, adjusted_event_base_name)
    
    if not os.path.exists(events_dir):
        raise FileNotFoundError(f"Events directory '{events_dir}' does not exist.")

    height = 480
    width = 640

    frame_shape = (int(height), int(width))

    for folder in folders:
        event_file = os.path.join(events_dir, folder, "events", "left", "events.h5")
        adjusted_event_dir = os.path.join(adjusted_event_base, folder)

        calibration_file = os.path.join(dataset_dir, f"{mode}_calibration", folder, "calibration", "cam_to_cam.yaml")
        
        event_to_frame(
            event_filepath=Path(event_file),
            event_delta_time_ms=desired_interval,
            high_low_ratio=high_low_ratio,
            rgb_delta_time_ms=rgb_delta_time,
            use_time_lookup=use_time_lookup,
            use_calibration=True,
            output_path=Path(adjusted_event_dir),
            calibration_file=calibration_file,
            frame_shape=frame_shape,
            gamma=gamma,
            r0=r0
        )
        
        print(f"Running Crop Event Frames")
        crop_images_to_640x440(adjusted_event_dir)
        print(f"Event frames processed for folder: {folder}\n")


def process_all_operations(dataset_dir, output_base_dir, desired_interval=50, mode="train", rgb_delta_time=None, use_time_lookup=False, high_low_ratio=10.0, gamma=2.0, event_folder_name=None, r0=None):
    segmentation_dir = os.path.join(dataset_dir, f"{mode}_semantic_segmentation")
    calibration_dir = os.path.join(dataset_dir, f"{mode}_calibration")
    images_dir = os.path.join(dataset_dir, f"{mode}_images")

    # Ensure segmentation directory exists
    if not os.path.exists(segmentation_dir):
        raise FileNotFoundError(f"Segmentation directory '{segmentation_dir}' does not exist.")

    folders = [f for f in os.listdir(segmentation_dir) if os.path.isdir(os.path.join(segmentation_dir, f))]

    for folder in folders:
        # Paths for inputs and outputs
        calibration_file = os.path.join(calibration_dir, folder, "calibration", "cam_to_cam.yaml")
        image_dir = os.path.join(images_dir, folder, "images", "left", "rectified")
        mask11_dir = os.path.join(segmentation_dir, folder, "11classes")
        mask19_dir = os.path.join(segmentation_dir, folder, "19classes")
        adjusted_rgb_dir = os.path.join(output_base_dir, "adjusted_rgbs", folder)
        adjusted_segmentation_dir = os.path.join(output_base_dir, "segmentation", folder)
        adjusted_mask11_dir = os.path.join(adjusted_segmentation_dir, "11classes")
        adjusted_mask19_dir = os.path.join(adjusted_segmentation_dir, "19classes")

        # Ensure output directories exist
        os.makedirs(adjusted_rgb_dir, exist_ok=True)
        os.makedirs(adjusted_mask11_dir, exist_ok=True)
        os.makedirs(adjusted_mask19_dir, exist_ok=True)

        # Step 1: Adjust RGB
        rectify_rgb_images(
            rgb_folder_path=Path(image_dir), 
            output_path=Path(adjusted_rgb_dir), 
            calibration_file=calibration_file, 
            high_res=False, 
        )

        # Step 2: Crop Adjusted RGB
        print(f"Running Crop RGB Images:")
        crop_images_to_640x440(adjusted_rgb_dir)
        
        # Step 3: Copy 11classes and 19classes masks
        print(f"Copying 11classes and 19classes masks for folder: {folder}")
        with tqdm(total=len(os.listdir(mask11_dir)), desc=f"Copying 11classes for {folder}", unit="file") as pbar11:
            for file_name in os.listdir(mask11_dir):
                src_path = os.path.join(mask11_dir, file_name)
                dst_path = os.path.join(adjusted_mask11_dir, file_name)
                shutil.copy(src_path, dst_path)
                pbar11.update(1)

        with tqdm(total=len(os.listdir(mask19_dir)), desc=f"Copying 19classes for {folder}", unit="file") as pbar19:
            for file_name in os.listdir(mask19_dir):
                src_path = os.path.join(mask19_dir, file_name)
                dst_path = os.path.join(adjusted_mask19_dir, file_name)
                shutil.copy(src_path, dst_path)
                pbar19.update(1)

        print(f"Processing completed for folder: {folder}\n")

    # Step 4: Add Event Frames
    add_event_frames(dataset_dir, output_base_dir, desired_interval, mode, rgb_delta_time, use_time_lookup, high_low_ratio, gamma, event_folder_name=event_folder_name, r0=r0)


def main():
    parser = argparse.ArgumentParser(description="Automate dataset processing operations.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Path to the output base directory.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Process train or test dataset.")
    parser.add_argument("--operation", type=str, choices=["event_only", "all"], default="event_only",
                        help="Choose whether to process only event frames or perform all operations.")
    parser.add_argument("--desired_interval", type=int, default=50, help="Desired interval (in ms) for event frame processing.")
    parser.add_argument("--rgb_delta_time", type=float, default=None, help="Time window (in milliseconds) for matching frames.")
    parser.add_argument("--use_time_lookup", action="store_true", help="Use time lookup for frame matching.")
    parser.add_argument("--high_low_ratio", type=float, default=10.0, help="High to low frame ratio.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma value for power function.")
    
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    base_dir = None
    if args.rgb_delta_time != "50":
        # base_dir = os.path.join(args.output_base_dir, f"adjusted_{args.mode}_{args.desired_interval}ms")
        base_dir = os.path.join(args.output_base_dir)
    else:
        # base_dir = os.path.join(args.output_base_dir, f"adjusted_{args.mode}_3D_{args.desired_interval}ms")
        base_dir = os.path.join(args.output_base_dir)
    
    
    if args.operation == "event_only":
        print("Processing only event frames...")
        add_event_frames(dataset_dir, base_dir, args.desired_interval, args.mode, args.rgb_delta_time, args.use_time_lookup, args.high_low_ratio, args.gamma)
    elif args.operation == "all":
        print("Processing all operations...")
        process_all_operations(dataset_dir, base_dir, args.desired_interval, args.mode, args.rgb_delta_time, args.use_time_lookup, args.high_low_ratio, args.gamma)


if __name__ == "__main__":
    main()

# sample run code: 

# python automated_script.py --dataset_dir=/media/CezeriDrive_12TB/GSS/Ozkan_4090_2/DSEC_original_datasets/train --output_base_dir=adjusted_dataset/train --mode=train --operation=all --desired_interval=50 --rgb_delta_time=50 --use_time_lookup --high_low_ratio=10.0 --gamma=2.0

# python automated_script.py --dataset_dir=/media/CezeriDrive_12TB/GSS/Ozkan_4090_2/DSEC_original_datasets/test --output_base_dir=adjusted_dataset/test --mode=test --operation=all --desired_interval=50 --rgb_delta_time=50 --use_time_lookup --high_low_ratio=10.0 --gamma=2.0