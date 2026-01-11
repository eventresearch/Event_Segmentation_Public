import os
import shutil
from tqdm import tqdm

def unify_datasets_from_new_structure(parent_dir, output_dir, event_ms=50, copy_all_event_dirs=False, event_folder_name=None, skip_existing_folder=False):
    """
    Unify datasets from a new folder structure into a single directory with renamed files.

    Args:
        parent_dir (str): Path to the parent directory containing event, rgb, and segmentation subfolders.
        output_dir (str): Path to the output directory where unified dataset will be stored.
        event_ms (int): Milliseconds used to specify event frame subdirectory.
        copy_all_event_dirs (bool): If True, copy all subfolders in parent_dir with 'event' in their name to the event output dir.

    Returns:
        None
    """
    # Define input subdirectories
    if copy_all_event_dirs:
        event_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                     if os.path.isdir(os.path.join(parent_dir, d)) and 'event' in d.lower()]
    else:
        if event_folder_name is not None:
            event_dirs = [os.path.join(parent_dir, event_folder_name)]
        else:
            raise ValueError("event_folder_name must be provided if copy_all_event_dirs is False.")

    sub_dirs = {
        "event": event_dirs,
        "rgb": os.path.join(parent_dir, "adjusted_rgbs"),
        "segmentation": os.path.join(parent_dir, "segmentation"),
    }

    # Define output subdirectories
    output_dirs = {
        "rgb": os.path.join(output_dir, "rgb"),
        "mask11": os.path.join(output_dir, "mask11"),
        "mask19": os.path.join(output_dir, "mask19"),
    }

    # For event output dirs, handle differently if copying all event dirs
    skipped_folders = set()
    if copy_all_event_dirs:
        event_output_dirs = {}
        for event_dir in sub_dirs["event"]:
            event_dir_name = os.path.basename(event_dir.rstrip('/'))
            out_dir = os.path.join(output_dir, f"event_frame_{event_ms}ms_{event_dir_name}")
            if skip_existing_folder and os.path.exists(out_dir):
                skipped_folders.add(out_dir)
                continue
            os.makedirs(out_dir, exist_ok=True)
            event_output_dirs[event_dir] = out_dir
    else:
        if event_folder_name is not None:
            event_output_dir = os.path.join(output_dir, event_folder_name)
        else:
            event_output_dir = os.path.join(output_dir, f"event_frame_{event_ms}ms")
        if skip_existing_folder and os.path.exists(event_output_dir):
            skipped_folders.add(event_output_dir)
        else:
            os.makedirs(event_output_dir, exist_ok=True)

    # Create other output directories
    for path in output_dirs.values():
        if skip_existing_folder and os.path.exists(path):
            skipped_folders.add(path)
            continue
        os.makedirs(path, exist_ok=True)

    # Process event files
    for event_dir in sub_dirs["event"]:
        if not os.path.exists(event_dir):
            print(f"Warning: {event_dir} does not exist. Skipping.")
            continue
        if copy_all_event_dirs:
            dst_dir = event_output_dirs[event_dir]
        else:
            dst_dir = event_output_dir
        if dst_dir in skipped_folders:
            continue
        for location in os.listdir(event_dir):
            location_path = os.path.join(event_dir, location)
            if not os.path.isdir(location_path):
                continue
            files = os.listdir(location_path)
            with tqdm(total=len(files), desc=f"Processing EVENT in {location} ({os.path.basename(event_dir)})", unit="file") as pbar:
                for file_name in files:
                    dst_name = f"{location}_{file_name}"
                    src_path = os.path.join(location_path, file_name)
                    dst_path = os.path.join(dst_dir, dst_name)
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)
                    pbar.update(1)

    # Process RGB files
    rgb_dir = sub_dirs["rgb"]
    dst_dir = output_dirs["rgb"]
    if dst_dir in skipped_folders:
        pass
    elif not os.path.exists(rgb_dir):
        print(f"Warning: {rgb_dir} does not exist. Skipping.")
    else:
        for location in os.listdir(rgb_dir):
            location_path = os.path.join(rgb_dir, location)
            if not os.path.isdir(location_path):
                continue
            files = os.listdir(location_path)
            with tqdm(total=len(files), desc=f"Processing RGB in {location}", unit="file") as pbar:
                for file_name in files:
                    dst_name = f"{location}_{file_name}"
                    src_path = os.path.join(location_path, file_name)
                    dst_path = os.path.join(dst_dir, dst_name)
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)
                    pbar.update(1)

    # Process segmentation masks (11classes, 19classes)
    seg_dir = sub_dirs["segmentation"]
    for mask_key in ["mask11", "mask19"]:
        dst_dir = output_dirs[mask_key]
        if dst_dir in skipped_folders:
            continue
    if not os.path.exists(seg_dir):
        print(f"Warning: {seg_dir} does not exist. Skipping.")
    else:
        for location in os.listdir(seg_dir):
            location_path = os.path.join(seg_dir, location)
            if not os.path.isdir(location_path):
                continue

            # Handle 11classes and 19classes directories
            for mask_type, mask_key in [("11classes", "mask11"), ("19classes", "mask19")]:
                mask_dir = os.path.join(location_path, mask_type)
                if not os.path.exists(mask_dir):
                    print(f"Warning: {mask_dir} does not exist. Skipping.")
                    continue

                dst_dir = output_dirs[mask_key]
                if dst_dir in skipped_folders:
                    continue
                files = os.listdir(mask_dir)
                with tqdm(total=len(files), desc=f"Processing {mask_type.upper()} in {location}", unit="file") as pbar:
                    for file_name in files:
                        dst_name = f"{location}_{file_name}"
                        src_path = os.path.join(mask_dir, file_name)
                        dst_path = os.path.join(dst_dir, dst_name)
                        if not os.path.exists(dst_path):
                            shutil.copy(src_path, dst_path)
                        pbar.update(1)

    print(f"Unified dataset created at: {output_dir}")

if __name__ == "__main__":
    # Example usage
    parent_dir = "adjusted_dataset/test"  # Parent directory containing the new structure
    output_dir = "unified_dataset/test"  # Output directory for unified dataset
    unify_datasets_from_new_structure(parent_dir, output_dir, event_ms=50, copy_all_event_dirs=True)
