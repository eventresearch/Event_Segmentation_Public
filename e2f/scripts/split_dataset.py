
import os
import shutil
from tqdm import tqdm
import glob

def split_and_copy_dataset(original_dataset_dir, target_dir, event_folder_name=None, train_ratio=0.8, skip_existing_folder=False):
    # --- CONFIGURABLE EVENT FRAME TYPE ---
    EVENT_FRAME_TYPE = event_folder_name  # e.g., 'event_frame_50ms' or None for all

    # Dynamically build subfolders for all event_frame_* types in the dataset
    all_event_types = set()
    for split in ['train', 'test']:
        split_dir = os.path.join(original_dataset_dir, split)
        if os.path.exists(split_dir):
            for d in os.listdir(split_dir):
                if 'event_frame' in d and (EVENT_FRAME_TYPE is None or d == EVENT_FRAME_TYPE):
                    all_event_types.add(d)

    subfolders = [
        'images/train', 'images/val', 'images/test',
        'masks/train', 'masks/val', 'masks/test',
        'mask19/train', 'mask19/val', 'mask19/test',
    ]
    for event_type in all_event_types:
        subfolders.extend([
            f'{event_type}/train',
            f'{event_type}/val',
            f'{event_type}/test',
        ])

    # Create the directory structure and track skipped folders
    skipped_folders = set()
    for subfolder in subfolders:
        subfolder_path = os.path.join(target_dir, subfolder)
        if skip_existing_folder and os.path.exists(subfolder_path):
            skipped_folders.add(subfolder_path)
            continue
        os.makedirs(subfolder_path, exist_ok=True)

    # Helper function to split dataset into train and val sets
    def split_dataset(files, train_ratio=0.8):
        train_size = int(len(files) * train_ratio)
        train_files = files[:train_size]
        val_files = files[train_size:]
        return train_files, val_files

    # Copy RGB images
    for split in ['train', 'test']:
        image_files = sorted(os.listdir(os.path.join(original_dataset_dir, split, "rgb")))
        train_folder = os.path.join(target_dir, "images/train")
        val_folder = os.path.join(target_dir, "images/val")
        test_folder = os.path.join(target_dir, "images/test")
        if split == 'train':
            train_images, val_images = split_dataset(image_files, train_ratio)
            if train_folder not in skipped_folders:
                with tqdm(total=len(train_images), desc=f"RGB train", unit="file") as pbar:
                    for image in train_images:
                        shutil.copy(os.path.join(original_dataset_dir, split, "rgb", image), os.path.join(train_folder, image))
                        pbar.update(1)
            if val_folder not in skipped_folders:
                with tqdm(total=len(val_images), desc=f"RGB val", unit="file") as pbar:
                    for image in val_images:
                        shutil.copy(os.path.join(original_dataset_dir, split, "rgb", image), os.path.join(val_folder, image))
                        pbar.update(1)
        else:
            if test_folder not in skipped_folders:
                with tqdm(total=len(image_files), desc=f"RGB test", unit="file") as pbar:
                    for image in image_files:
                        shutil.copy(os.path.join(original_dataset_dir, split, "rgb", image), os.path.join(test_folder, image))
                        pbar.update(1)

    # Copy mask11 files
    for split in ['train', 'test']:
        mask11_files = sorted(os.listdir(os.path.join(original_dataset_dir, split, "mask11")))
        train_folder = os.path.join(target_dir, "masks/train")
        val_folder = os.path.join(target_dir, "masks/val")
        test_folder = os.path.join(target_dir, "masks/test")
        if split == 'train':
            train_mask11, val_mask11 = split_dataset(mask11_files, train_ratio)
            if train_folder not in skipped_folders:
                with tqdm(total=len(train_mask11), desc=f"mask11 train", unit="file") as pbar:
                    for mask in train_mask11:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask11", mask), os.path.join(train_folder, mask))
                        pbar.update(1)
            if val_folder not in skipped_folders:
                with tqdm(total=len(val_mask11), desc=f"mask11 val", unit="file") as pbar:
                    for mask in val_mask11:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask11", mask), os.path.join(val_folder, mask))
                        pbar.update(1)
        else:
            if test_folder not in skipped_folders:
                with tqdm(total=len(mask11_files), desc=f"mask11 test", unit="file") as pbar:
                    for mask in mask11_files:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask11", mask), os.path.join(test_folder, mask))
                        pbar.update(1)

    # Copy mask19 files
    for split in ['train', 'test']:
        mask19_files = sorted(os.listdir(os.path.join(original_dataset_dir, split, "mask19")))
        train_folder = os.path.join(target_dir, "mask19/train")
        val_folder = os.path.join(target_dir, "mask19/val")
        test_folder = os.path.join(target_dir, "mask19/test")
        if split == 'train':
            train_mask19, val_mask19 = split_dataset(mask19_files, train_ratio)
            if train_folder not in skipped_folders:
                with tqdm(total=len(train_mask19), desc=f"mask19 train", unit="file") as pbar:
                    for mask in train_mask19:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask19", mask), os.path.join(train_folder, mask))
                        pbar.update(1)
            if val_folder not in skipped_folders:
                with tqdm(total=len(val_mask19), desc=f"mask19 val", unit="file") as pbar:
                    for mask in val_mask19:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask19", mask), os.path.join(val_folder, mask))
                        pbar.update(1)
        else:
            if test_folder not in skipped_folders:
                with tqdm(total=len(mask19_files), desc=f"mask19 test", unit="file") as pbar:
                    for mask in mask19_files:
                        shutil.copy(os.path.join(original_dataset_dir, split, "mask19", mask), os.path.join(test_folder, mask))
                        pbar.update(1)

    # Copy event frames from selected event_frame_* directories
    for split in ['train', 'test']:
        split_dir = os.path.join(original_dataset_dir, split)
        event_dirs = [d for d in os.listdir(split_dir)
                      if 'event_frame' in d and (EVENT_FRAME_TYPE is None or d == EVENT_FRAME_TYPE)]
        for event_dir in event_dirs:
            # Use full target paths for checks (consistent with skipped_folders storing full paths)
            train_folder = os.path.join(target_dir, event_dir, 'train')
            val_folder = os.path.join(target_dir, event_dir, 'val')
            test_folder = os.path.join(target_dir, event_dir, 'test')

            if train_folder in skipped_folders and val_folder in skipped_folders and test_folder in skipped_folders:
                continue

            event_dir_path = os.path.join(split_dir, event_dir)
            event_files = sorted(os.listdir(event_dir_path))
            if split == 'train':
                # compute split once to avoid undefined variables when one of the folders is skipped
                train_event, val_event = split_dataset(event_files, train_ratio)
                if train_folder not in skipped_folders:
                    with tqdm(total=len(train_event), desc=f"{event_dir} train", unit="file") as pbar:
                        for event in train_event:
                            shutil.copy(os.path.join(event_dir_path, event), os.path.join(train_folder, event))
                            pbar.update(1)
                if val_folder not in skipped_folders:
                    with tqdm(total=len(val_event), desc=f"{event_dir} val", unit="file") as pbar:
                        for event in val_event:
                            shutil.copy(os.path.join(event_dir_path, event), os.path.join(val_folder, event))
                            pbar.update(1)
            else:
                if test_folder not in skipped_folders:
                    with tqdm(total=len(event_files), desc=f"{event_dir} test", unit="file") as pbar:
                        for event in event_files:
                            shutil.copy(os.path.join(event_dir_path, event), os.path.join(test_folder, event))
                            pbar.update(1)

    print(f"Dataset has been reorganized successfully at {target_dir}.")
