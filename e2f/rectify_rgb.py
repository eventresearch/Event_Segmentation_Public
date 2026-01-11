from pathlib import Path
# from tqdm import tqdm
import os
import cv2
from scripts.utils.functions import rectify_rgb_images

if __name__ == '__main__':
    rgb_folder_path = Path("rectified")
    
    output_path = Path("rectified_calibrated")
    
    calibration_file = Path("cam_to_cam.yaml")

    if not rgb_folder_path.exists():
        print(f"Error: Input folder '{rgb_folder_path}' does not exist.")
    else:
        print(f"Rectifying images from '{rgb_folder_path}' to '{output_path}'...")
        rectify_rgb_images(
            rgb_folder_path=rgb_folder_path, 
            output_path=output_path, 
            calibration_file=calibration_file, 
            high_res=True, 
            isotropic=True, 
            fps=20
        )
        print("Done.")
