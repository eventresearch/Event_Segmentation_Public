import os
import random
import shutil
from tqdm import tqdm

def reorganize_and_split_dataset_consistently(base_folder, subfolders, output_folder, train_ratio=0.8, seed=42):
    """
    Split 'train' into 'train' and 'val' consistently across all subfolders and reorganize the dataset structure.

    Args:
        base_folder (str): Base folder containing 'train' and 'test' folders.
        subfolders (list): List of subfolder names to process (e.g., ["rgb", "mask11", "mask19"]).
        output_folder (str): Output folder to save the reorganized and split datasets.
        train_ratio (float): Proportion of the train set to retain as training data.
        seed (int): Random seed for reproducibility.
    """
    # Ensure reproducibility
    random.seed(seed)

    # Input folder paths
    train_base = os.path.join(base_folder, "train")
    test_base = os.path.join(base_folder, "test")

    # Validate base folders exist
    if not os.path.exists(train_base):
        print(f"Error: Train folder '{train_base}' does not exist!")
        return
    
    if not os.path.exists(test_base):
        print(f"Warning: Test folder '{test_base}' does not exist. Will skip test data.")

    # Collect filenames from the first subfolder
    representative_subfolder = subfolders[0]
    subfolder_train = os.path.join(train_base, representative_subfolder)

    if not os.path.exists(subfolder_train):
        print(f"Error: Representative subfolder '{subfolder_train}' does not exist!")
        return

    # Collect and shuffle filenames (only files, not directories)
    all_filenames = sorted([f for f in os.listdir(subfolder_train) 
                           if os.path.isfile(os.path.join(subfolder_train, f)) and not f.startswith('.')])
    
    if len(all_filenames) == 0:
        print(f"Error: No files found in '{subfolder_train}'!")
        return
    
    random.shuffle(all_filenames)

    # Split into train and val
    train_size = int(len(all_filenames) * train_ratio)
    train_split = all_filenames[:train_size]
    val_split = all_filenames[train_size:]

    print(f"\n{'='*60}")
    print(f"Dataset Split Summary:")
    print(f"{'='*60}")
    print(f"Total files: {len(all_filenames)}")
    print(f"Training files: {len(train_split)} ({len(train_split)/len(all_filenames)*100:.1f}%)")
    print(f"Validation files: {len(val_split)} ({len(val_split)/len(all_filenames)*100:.1f}%)")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")

    # Process all subfolders consistently
    for subfolder in subfolders:
        subfolder_train = os.path.join(train_base, subfolder)
        subfolder_test = os.path.join(test_base, subfolder)

        output_train_folder = os.path.join(output_folder, subfolder, "train")
        output_val_folder = os.path.join(output_folder, subfolder, "val")
        output_test_folder = os.path.join(output_folder, subfolder, "test")

        # Create output directories
        os.makedirs(output_train_folder, exist_ok=True)
        os.makedirs(output_val_folder, exist_ok=True)
        os.makedirs(output_test_folder, exist_ok=True)

        # Counters for statistics
        train_copied = 0
        val_copied = 0
        test_copied = 0

        # Copy training files
        for filename in tqdm(train_split, desc=f"{subfolder:20s} train", unit="file"):
            src_path = os.path.join(subfolder_train, filename)
            dst_path = os.path.join(output_train_folder, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                train_copied += 1
            else:
                print(f"\n  Warning: {src_path} does not exist. Skipping.")

        # Copy validation files
        for filename in tqdm(val_split, desc=f"{subfolder:20s} val", unit="file"):
            src_path = os.path.join(subfolder_train, filename)
            dst_path = os.path.join(output_val_folder, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                val_copied += 1
            else:
                print(f"\n  Warning: {src_path} does not exist. Skipping.")

        # Copy test files (unchanged)
        if os.path.exists(subfolder_test):
            test_files = [f for f in os.listdir(subfolder_test) 
                         if os.path.isfile(os.path.join(subfolder_test, f)) and not f.startswith('.')]
            for filename in tqdm(test_files, desc=f"{subfolder:20s} test", unit="file"):
                src_path = os.path.join(subfolder_test, filename)
                dst_path = os.path.join(output_test_folder, filename)
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    test_copied += 1
                else:
                    print(f"\n  Warning: {src_path} does not exist. Skipping.")
        
        # Print summary for this subfolder
        print(f"  ✓ {subfolder}: {train_copied} train, {val_copied} val, {test_copied} test")
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset reorganization completed successfully!")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Example usage
    reorganize_and_split_dataset_consistently(
        base_folder="adjusted_dataset",
        subfolders=["rgb", "mask11", "mask19", "event_frame_50ms"],
        output_folder="unified_dataset_split_v2",
        train_ratio=0.8
    )
