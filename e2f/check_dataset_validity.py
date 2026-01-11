import os


DATASET_DIR = "final_dataset"
SPLITS = ["train", "test", "val"]  # Add or remove splits as needed

# List all subfolders in the dataset directory

def get_subfolders(dataset_dir):
    return [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]


# Recursively get all files in a subfolder

def get_all_files_recursive(subfolder_path):
    file_list = []
    for root, _, files in os.walk(subfolder_path):
        for f in files:
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, subfolder_path)
            file_list.append((rel_path, abs_path))
    return file_list

# Check if a file is empty
def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0


def check_split(split, subfolders):
    print(f"\nChecking split: {split}")
    split_file_sets = {}
    empty_files = {}
    for subfolder in subfolders:
        split_path = os.path.join(DATASET_DIR, subfolder, split)
        if not os.path.exists(split_path):
            print(f"  Split '{split}' missing in subfolder '{subfolder}'")
            split_file_sets[subfolder] = set()
            empty_files[subfolder] = []
            continue
        files = get_all_files_recursive(split_path)
        rel_files = set([rel for rel, absf in files])
        split_file_sets[subfolder] = rel_files
        empty_files[subfolder] = [rel for rel, absf in files if is_file_empty(absf)]

    # Compare file sets
    all_file_sets = list(split_file_sets.values())
    if len(all_file_sets) > 1 and not all(s == all_file_sets[0] for s in all_file_sets):
        print("  File structure mismatch detected:")
        # Find union and intersection
        union = set().union(*all_file_sets)
        for subfolder in subfolders:
            missing = union - split_file_sets[subfolder]
            extra = split_file_sets[subfolder] - union
            if missing:
                print(f"    {subfolder} is missing {len(missing)} files:")
                for f in sorted(missing):
                    print(f"      {f}")
            if extra:
                print(f"    {subfolder} has {len(extra)} extra files:")
                for f in sorted(extra):
                    print(f"      {f}")
    else:
        print(f"  All subfolders have perfectly matching file trees for split '{split}' ({len(all_file_sets[0])} files).")

    # Report empty files
    any_empty = False
    for subfolder, files in empty_files.items():
        if files:
            any_empty = True
            print(f"  Empty files in {subfolder}/{split}:")
            for f in files:
                print(f"    {f}")
    if not any_empty:
        print(f"  No empty files detected in split '{split}'.")

def main():
    subfolders = get_subfolders(DATASET_DIR)
    print(f"Checking subfolders: {subfolders}")
    for split in SPLITS:
        check_split(split, subfolders)

if __name__ == "__main__":
    main()
