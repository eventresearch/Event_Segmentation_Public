import os

def get_file_count(folder_path, extension=".png"):
    """ Get the number of files with a specific extension in a given folder. """
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return 0

    return len([f for f in os.listdir(folder_path) if f.endswith(extension)])

def remove_first_image(folder_path):
    """ Remove the first PNG file in a given folder if it exists. """
    if not os.path.exists(folder_path):
        print(f"Warning: The folder path {folder_path} does not exist.")
        return

    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    if png_files:
        first_file = os.path.join(folder_path, png_files[0])
        os.remove(first_file)
        print(f"Removed first image: {first_file}")
    else:
        print(f"Warning: No PNG files found in {folder_path}.")

def rename_images_in_folder(folder_path):
    """ Rename all images in a folder to start from 000000.png, 000001.png, etc. """
    if not os.path.exists(folder_path):
        print(f"Warning: The folder path {folder_path} does not exist.")
        return

    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

    if not image_files:
        print(f"Warning: No PNG files found to rename in {folder_path}.")
        return

    for index, filename in enumerate(image_files):
        old_path = os.path.join(folder_path, filename)
        new_name = f"{index:06d}.png"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

def list_subfolders(parent_folder):
    """ List all direct subfolders inside a parent folder. """
    if not os.path.exists(parent_folder):
        print(f"Warning: {parent_folder} does not exist.")
        return []
    
    return [os.path.join(parent_folder, subfolder) for subfolder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, subfolder))]

def process_folders(base_folder, event_folder_base, rgb_folder_base="adjusted_rgbs", mask_folder_base="segmentation"):
    """
    Process RGB, mask, and event folders independently.

    - Lists `adjusted_rgbs`, `segmentation`, and `adjusted_event_frames_10ms` from the base folder.
    - Lists subfolders inside each of them.
    - Removes first images from RGB and mask subfolders.
    - Leaves event folders untouched.

    Args:
    - base_folder (str): Path to the dataset.
    """
    if not os.path.exists(base_folder):
        print(f"Error: The base folder {base_folder} does not exist.")
        return

    # Define main folders
    event_folder_main = os.path.join(base_folder, event_folder_base)
    rgb_folder_main = os.path.join(base_folder, rgb_folder_base)
    mask_folder_main = os.path.join(base_folder, mask_folder_base)

    # Get subfolders (scene folders) inside each main folder
    event_subfolders = list_subfolders(event_folder_main)
    rgb_subfolders = list_subfolders(rgb_folder_main)
    mask_subfolders = list_subfolders(mask_folder_main)

    print(f"\nFound {len(rgb_subfolders)} RGB subfolders, {len(mask_subfolders)} mask subfolders, {len(event_subfolders)} event subfolders (untouched).")

    # Process RGB folders
    for rgb_folder in rgb_subfolders:
        num_rgb = get_file_count(rgb_folder)
        corresponding_event_folder = os.path.join(event_folder_main, os.path.basename(rgb_folder))
        num_event = get_file_count(corresponding_event_folder)

        if num_event == 0 or num_rgb == 0:
            print(f"Skipping {rgb_folder}: No valid files in RGB or Event.")
            continue

        print(f"\nProcessing RGB folder: {rgb_folder} | RGB count: {num_rgb}, Event count: {num_event}")

        # If event files are not a multiple of RGB, remove first RGB file
        if num_event % num_rgb != 0:
            remove_first_image(rgb_folder)

            num_rgb_after = get_file_count(rgb_folder)  # Recount after deletion
            print(f"Updated RGB count: {num_rgb_after}")

            if (num_rgb_after - 1) > 0 and num_event % (num_rgb_after - 1) == 0:
                remove_first_image(rgb_folder)
                print(f"Removed another RGB file to match event count.")

            rename_images_in_folder(rgb_folder)
        else:
            print(f"No changes needed in RGB folder {rgb_folder} .")

    # Process Mask folders separately
    for mask_folder in mask_subfolders:
        scene_name = os.path.basename(mask_folder)  # Get scene folder name
        corresponding_event_folder = os.path.join(event_folder_main, scene_name)
        num_event = get_file_count(corresponding_event_folder)

        if num_event == 0:
            print(f"Skipping mask processing for {mask_folder}: No corresponding event files.")
            continue

        print(f"\nProcessing Mask folder: {mask_folder}")

        # List subfolders (e.g., 11classes, 19classes) inside this scene’s mask folder
        nested_mask_folders = list_subfolders(mask_folder)

        for nested_folder in nested_mask_folders:
            num_mask = get_file_count(nested_folder)

            if num_mask == 0:
                print(f"Skipping {nested_folder}: No mask files found.")
                continue

            print(f"Checking mask folder: {nested_folder} | Mask count: {num_mask}, Event count: {num_event}")

            # If event files are not a multiple of mask files, remove first mask file
            if num_event % num_mask != 0:
                remove_first_image(nested_folder)

                num_mask_after = get_file_count(nested_folder)  # Recount after deletion
                print(f"Updated mask count in {nested_folder}: {num_mask_after}")

                if (num_mask_after - 1) > 0 and num_event % (num_mask_after - 1) == 0:
                    remove_first_image(nested_folder)
                    print(f"Removed another mask file in {nested_folder} to match event count.")

                rename_images_in_folder(nested_folder)
            else:
                print(f"No changes needed in mask folder: {nested_folder}")

    print("\nProcessing complete!")

if __name__ == "__main__":
    # Define base dataset path
    base_path = "adjusted_dataset/train"
    process_folders(base_path, event_folder_base="adjusted_event_frames_50ms")