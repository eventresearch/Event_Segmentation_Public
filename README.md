# Event-Based Segmentation Framework

This repository provides a comprehensive framework for semantic segmentation using event-based cameras. It supports multiple state-of-the-art architectures and includes a dedicated toolkit (`e2f`) for processing event data into various representations.

---

## Table of Contents


- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Base Segmentation Class](#base-segmentation-class)
- [Adding a New Dataset Class](#adding-a-new-dataset-class)
- [Creating a New Model](#creating-a-new-model)
- [Modifying the Configuration File](#modifying-the-configuration-file)
- [Configuration Flags](#configuration-flags)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)

---

## Repository Structure

The repository is organized into two main components:

1.  **`event_seg/`**: The core segmentation framework containing:
    *   **Models**: Implementations of U-Net, DeepLabV3+, SegFormer, SwinFormer, Dual Encoders, and Autoencoders.
    *   **Training & Inference**: Scripts for training (`train.py`) and evaluation (`infer.py`).
    *   **Dataloading**: Flexible dataset classes for handling RGB, Event, and Mask data.

2.  **`e2f/` (Event-to-Frame)**: A toolkit for processing event data.
    *   **Data Processing**: Tools to convert raw event streams into frame representations (e.g., histograms, voxel grids).
    *   **Rectification**: Scripts to rectify RGB and Event data.
    *   **Utilities**: Helper scripts for dataset unification, splitting, and visualization.

**For details on event data processing, please refer to the [e2f README](e2f/README.md).**


---

## Installation

Ensure you have the required dependencies installed before using this repository.

```bash
pip install -r requirements.txt
```

Ensure you have PyTorch installed with CUDA support if using a GPU.

---

## Usage

### Dataset Structure

Your dataset should be structured as follows:

```
dataset_root/
│── images/        # RGB images
│── mask/          # Ground truth segmentation masks
│── event_frames/  # Event camera frames
│── autoencoder_output/ # Precomputed features for autoencoders
```

---

## Base Segmentation Class

The base dataset class is located in `base_class.py` and provides foundational functionality for handling different input sources.

```python
class BaseSegmentationDataset(Dataset):
    def __init__(self, config):
        self.image_dir = config.get('image_dir')
        self.mask_dir = config.get('mask_dir')
        self.event_dir = config.get('event_dir')
        self.autoencoder_dir = config.get('autoencoder_dir')
```

---

## Adding a New Dataset Class

To extend functionality, create a new dataset class inheriting from `BaseSegmentationDataset`. Example:

```python
class CustomSegmentationDataset(BaseSegmentationDataset):
    def __getitem__(self, idx):
        img = self._load_image(idx)
        mask = self._load_mask(idx)
        return img, mask
```

Then, register the new dataset in `dataset.py`:

```python
dataset_mapping = {
    "custom_model": CustomSegmentationDataset,
}
```

---

## Creating a New Model

New models can be added by extending `BaseUNet` in `model.py`. Example:

```python
class CustomUNet(BaseUNet):
    def __init__(self, config):
        super().__init__(config)
        self.custom_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

Then, register it in `model.py`:

```python
model_registry = {
    "custom_model": CustomUNet,
}
```

---

## Modifying the Configuration File

Modify `model_config.yaml` to set up your model:

```yaml
training:
  model: "auto_encoder_dual_encoder_concat"
  epochs: 40
  batch_size: 1
  learning_rate: 0.0001
  loss: "FocalDice"
  tensorboard_save_log: true
  enable_inference_during_training: true
  inference_interval: 1
  dont_use_validation: false
```

---

## Configuration Flags

The `model_config.yaml` file contains several important flags:

- **`tensorboard_save_log`**: Enables TensorBoard logging.
- **`is_autoencoder`**: Determines if the model is an autoencoder. If set to `true`, the output channels will be forced to 1.
- **`use_rgb`**: Whether to use RGB frames as input.
- **`use_event`**: Whether to use event camera frames as input.
- **`save_masks`**: Saves predicted segmentation masks.
- **`use_gt_masks`**: If true, ground truth masks will be used for evaluation.
- **`event_ms`**: Specifies the time resolution of event frames.
- **`num_workers`**: Determines the number of workers for data loading.
- **`bilinear`**: If false, uses transposed convolutions instead of bilinear upsampling.
- **`base_model`**: Defines the base model for certain architectures, such as dual encoders.
- **`fusion_type`**: Used for models that combine multiple inputs (e.g., RGB and event frames), determines if concatenation or addition is used.

### Model-Specific Parameters

Each model has its own specific parameters defined in `model_specific_params`. These override general parameters where necessary:

- **Autoencoders** (e.g., `denoise_autoencoder`) enforce `num_of_in_channels=1` and use a specialized loss function (`BCE`).
- **Dual Encoder Models** (e.g., `dual_encoder_attention_add`) specify `base_model` and `fusion_type`.
- **Newly Created Models** should be added to this section to ensure `use_rgb` and `use_event` flags are correctly assigned.

Example configuration snippet:

```yaml
general:
  num_of_out_classes: 11
  num_of_in_channels: 3
  time_steps: 1
  event_channels: 1
  bilinear: false
  num_of_mask_classes: 11
  is_event_scapes: false
  tensorboard_save_log: true
  
model_specific_params:
  custom_model:
    use_rgb: true
    use_event: true
  # These flags should be set appropriately based on the model type.
  # For example, if the model is an autoencoder, `use_rgb` should be `false`.
  # If the base model is different, ensure the `base_model` flag is correctly assigned.
```

The same configuration file can be used for both **training** and **inference**.

---

## Training the Model

To train a model, run the `train.py` script:

```bash
python train.py 
```

Training logs will be saved in TensorBoard:

```bash
tensorboard --logdir=segmentation_results/runs/training
```

---

## Running Inference

To run inference on test images:

```bash
python infer.py 
```

The results will be saved in the folder specified in the `inference: output_folder` field of the configuration file.

---

## Conclusion

This repository provides a flexible and modular approach to event-based segmentation. Extend it by adding custom datasets and models, modify the configuration, and run training or inference seamlessly.
