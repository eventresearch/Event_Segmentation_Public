# e2f: Event-to-Frame Dataset Processing Toolkit

## Overview
This toolkit provides a complete pipeline for processing event-based datasets (such as DSEC). It enables the generation of various event representations (histograms, voxel grids, time surfaces), RGB rectification, and dataset organization (splitting, unifying).

## Installation

```bash
git clone https://github.com/eventresearch/e2f.git
cd e2f
conda create -n e2f python=3.8
conda activate e2f
pip install -r requirements.txt
python setup.py install
```

---

## 🚀 One-Click Pipeline: `run_full_pipeline.py`

The **easiest way** to process a dataset is using the `run_full_pipeline.py` script. It automates the entire workflow:
1.  **Auto Processing**: Generates event frames and rectifies RGBs.
2.  **Cleanup**: Removes first images (often corrupted/black in some datasets).
3.  **Unifiction**: Merges different processed folders into a unified structure.
4.  **Splitting**: Splits the unified dataset into Train/Val/Test.

### Usage
```bash
python run_full_pipeline.py --dataset_dir <path_to_input> --output_base_dir <path_to_output> --steps auto remove_first unify split
```

### Key Arguments
-   `--dataset_dir`: Root directory of your raw dataset.
-   `--steps`: List of steps to run (`auto`, `remove_first`, `unify`, `split`).
-   `--operation`: `event_only` (just events) or `all` (events + RGB + masks).
-   `--desired_interval`: Time window for events (ms), e.g., `50`.
-   `--high_low_ratio`: Parameters for advanced event representations.

---

## 🛠️ Standalone Utilities

These scripts are useful for quick, targeted operations without running the full pipeline.

### 1. Events to Frame Conversion (`events_to_frame.py`)
Quickly convert an `events.h5` file into a folder of event frames using the configuration hardcoded or passed in the script.
*   **Input**: `events.h5` (in current dir)
*   **Requires**: `cam_to_cam.yaml` (in current dir)
*   **Usage**: Run from the directory containing your data.
    ```bash
    python events_to_frame.py
    ```

### 2. RGB Rectification (`rectify_rgb.py`)
Rectifies all images in a folder named `rectified` (default) using `cam_to_cam.yaml`.
*   **Usage**: Run from directory with `rectified/` folder and calibration file.
    ```bash
    python rectify_rgb.py
    ```

### 3. Dataset Validity Check (`check_dataset_validity.py`)
Verifies that your processed dataset has consistent file counts across all subfolders and splits (Train/Test/Val).
*   **Usage**:
    ```bash
    python check_dataset_validity.py
    ```
    (Note: You may need to edit `DATASET_DIR` variable inside the script to point to your target).

---

## 🔧 Advanced / Manual Tools (in `scripts/`)

For granular control, you can use the scripts in `scripts/` directly.

-   **`scripts/automated_script.py`**: The core logic provider for processing.
-   **`scripts/remove_first_images_v2.py`**: Removes the first image in sequences to fix synchronization issues.
-   **`scripts/unify_datasets.py`**: Standardizes folder structures.
-   **`scripts/split_dataset.py`**: Creates Train/Val/Test splits.
-   **`scripts/images_to_video.py`**: Converts a folder of images to MP4.
    ```bash
    python scripts/images_to_video.py <image_folder> <output.mp4>
    ```
-   **`scripts/colorize-mask.py`**: Colorizes 19-class or 11-class segmentation masks for visualization.

---

## Utility Functions

See `scripts/utils/functions.py` for reusable core functions:
-   YAML loading & Camera Matrix conversion
-   Rectification logic
-   Event reading and slicing algorithms
