import os
import numpy as np
from PIL import Image

def colorize_mask_save(mask_path, save_path):
    """
    Colorize a segmentation mask for 19 classes using a specific colormap and save it as an image.

    Args:
    - mask (np.ndarray): Segmentation mask of shape [H, W] with class indices (0-18).
    - save_path (str): Path to save the colorized mask.

    Returns:
    - None
    """
    # Define the colormap for 19 classes
    colormap = [
        (128, 64, 128),   # Class 0
        (244, 35, 232),   # Class 1
        (70, 70, 70),     # Class 2
        (102, 102, 156),  # Class 3
        (190, 153, 153),  # Class 4
        (153, 153, 153),  # Class 5
        (250, 170, 30),   # Class 6
        (220, 220, 0),    # Class 7
        (107, 142, 35),   # Class 8
        (152, 251, 152),  # Class 9
        (70, 130, 180),   # Class 10
        (220, 20, 60),    # Class 11
        (255, 0, 0),      # Class 12
        (0, 0, 142),      # Class 13
        (0, 0, 70),       # Class 14
        (0, 60, 100),     # Class 15
        (0, 80, 100),     # Class 16
        (0, 0, 230),      # Class 17
        (119, 11, 32)     # Class 18
    ]
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Load your grayscale mask
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask).astype(np.uint8)
    
    # Create an RGB image with the same height and width as the mask
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply the colormap to each class index
    for class_idx, color in enumerate(colormap):
        colorized_mask[mask == class_idx] = color

    # Convert to PIL Image and save
    colorized_image = Image.fromarray(colorized_mask, mode="RGB")
    colorized_image.save(save_path)
    print(f"Colorized mask saved to {save_path}")

mask_folder = "/media/CezeriDrive_12TB/GSS/Ozkan_4090_2/e2f/seg_maps_gray"
save_folder = "/media/CezeriDrive_12TB/GSS/Ozkan_4090_2/e2f/segmented_colorized_11classes_new"

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

colorize_folder = True
if colorize_folder:
    for mask_filename in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_filename)
        save_path = os.path.join(save_folder, mask_filename)
        colorize_mask_save(mask_path, save_path)