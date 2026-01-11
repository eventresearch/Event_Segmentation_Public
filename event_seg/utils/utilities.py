"""
Utility functions for semantic segmentation with multi-modal inputs.

This module provides essential utility functions for:
- Dataset configuration and preprocessing
- TensorBoard logging and visualization
- Model checkpoint management
- Event camera data processing and classification
- Mask colorization and visualization
- IoU metric calculation (standard and weighted)
- Loss history tracking and plotting

Key functionality:
    - Event frame classification: Convert event camera data to class-based representations
    - Image/mask preprocessing: Load, normalize, and prepare inputs for models
    - TensorBoard integration: Log multi-modal inputs (RGB, events, autoencoder) with predictions
    - Checkpoint management: Save/load model states with optimizer and training metadata
    - Evaluation metrics: Calculate IoU and weighted IoU for segmentation quality assessment

Dependencies:
    - PyTorch: Model operations and tensor manipulation
    - PIL/cv2: Image loading and processing
    - NumPy: Numerical operations
    - TensorBoard: Training visualization
"""

from typing import Dict, Any, Tuple, Optional, Union, List
import cv2
import numpy as np
import torch
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import wandb

def create_dataset_config(
    base_config: Dict[str, Any], 
    image_dir: Optional[str], 
    mask_dir: Optional[str], 
    event_dir: Optional[str], 
    autoencoder_dir: Optional[str],
    is_train: bool = False
) -> Dict[str, Any]:
    """
    Generates a dataset configuration dictionary based on the base configuration.

    Args:
        base_config: Base configuration settings containing model and dataset parameters
        image_dir: Path to the image dataset directory (can be None if not using RGB)
        mask_dir: Path to the mask dataset directory (can be None for inference)
        event_dir: Path to the event dataset directory (can be None if not using events)
        autoencoder_dir: Path to the autoencoder features directory (can be None if not using)
        is_train: Whether this is for training (enables augmentation)

    Returns:
        Dataset configuration dictionary with all necessary parameters for dataset initialization
    """
    # Force augmentation off if not training
    enable_aug = base_config.get("apply_augmentation", False) if is_train else False
    aug_prob = base_config.get("augmentation_probability", 0.0) if is_train else 0.0
    
    return {
        "model_type": base_config.get("model_type"),
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "event_dir": event_dir,
        "autoencoder_dir": autoencoder_dir,
        "time_steps": base_config.get("time_steps"),
        "edge_method": base_config.get("edge_method"),
        "num_of_out_classes": base_config.get("num_of_out_classes"),
        "num_of_mask_classes": base_config.get("num_of_mask_classes"),
        "is_event_scapes": base_config.get("is_event_scapes"),
        "convert_to_binary": base_config.get("convert_to_binary", False),
        "binary_threshold": base_config.get("binary_threshold", 128),
        "morp_iterations": base_config.get("morp_iterations", 1),
        "apply_morphology": base_config.get("apply_morphology", False),
 
        "apply_augmentation": enable_aug,
        "augmentation_probability": aug_prob,
        "aug_flip_prob": base_config.get("aug_flip_prob"),
        "aug_scale_limit": base_config.get("aug_scale_limit"),
        "aug_rotate_limit": base_config.get("aug_rotate_limit"),
        "aug_shift_limit": base_config.get("aug_shift_limit"),
        "aug_affine_prob": base_config.get("aug_affine_prob"),
        "aug_rgb_prob": base_config.get("aug_rgb_prob"),   
    }

def get_tensorboard_writer(
    base_log_dir: str = "runs", 
    trial_timestamp: Optional[str] = None, 
    datetime: str = datetime.now().strftime("%Y%m%d_%H%M%S")
) -> Tuple[SummaryWriter, str]:
    """
    Create a unique log directory for each training trial.

    Args:
        base_log_dir: The base directory for TensorBoard logs (default: "runs")
        trial_timestamp: A custom timestamp for the trial (optional, auto-generated if None)
        datetime: Datetime string for unique trial naming

    Returns:
        Tuple containing:
            - SummaryWriter: A TensorBoard SummaryWriter instance for logging
            - str: Path to the log directory
    """
    trial_name = f"trial_{trial_timestamp}" if trial_timestamp else f"trial_{datetime}"
    log_dir = f"{base_log_dir}/{trial_name}"
    print(f"TensorBoard logs will be saved to: {log_dir}")
    return SummaryWriter(log_dir=log_dir), log_dir

def log_to_tensorboard(
    writer: SummaryWriter, 
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    outputs: torch.Tensor, 
    global_step: int, 
    prefix: str = "Train", 
    grid_size: int = 10, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs combined grids (RGB, Event, GT, and Predictions) to TensorBoard for segmentation.
    For autoencoders, logs Event, GT, and Predictions.

    Args:
        writer: TensorBoard SummaryWriter object for logging
        inputs: Input tensor(s) - RGB+Event for segmentation, Event for autoencoder [B, C, H, W]
        targets: Ground truth tensor - masks for segmentation, edge maps for autoencoders [B, H, W]
        outputs: Model prediction tensor [B, num_classes, H, W] or [B, C, H, W]
        global_step: Current step for logging (typically epoch number)
        prefix: Prefix to distinguish between Train and Validation logs (default: "Train")
        grid_size: Number of samples to include in the grid (default: 10)
        config: Configuration dict containing model type flags (is_3D, is_autoencoder, etc.)
        
    Note:
        Handles multiple model types based on config flags:
        - Standard segmentation: RGB + events + GT + predictions
        - 3D models: Temporal event stacks + GT + predictions  
        - Autoencoders: Event input + GT + reconstructed output
        - Triple input: RGB + events + autoencoder features + GT + predictions
    """
    if config is None:
        config = {}
    
    # Limit the number of samples to `grid_size`
    inputs = inputs[:grid_size]
    targets = targets[:grid_size]
    outputs = outputs[:grid_size]

    combined_grids = []
    is_3D = config.get("is_3D", False)
    is_autoencoder = config.get("is_autoencoder")
    is_double_channel_autoencoder = config.get("is_double_channel_autoencoder", False)
    
    use_rgb = config.get("use_rgb")
    use_event = config.get("use_event")
    use_autoencoder = config.get("use_autoencoder")
    
    use_gt_masks = config.get("use_gt_masks")
    
    if is_double_channel_autoencoder and not is_3D: # "double_channel_autoencoder":
        for i in range(len(inputs)):
            event_frame = inputs[i]  # Event frame (single-channel)
            gt_map = targets[i].unsqueeze(0)  # Ground truth edge map
            pred_map = outputs[i]  # Predicted edge map

            # Stack Event, GT, and Prediction vertically
            combined = torch.cat([event_frame, gt_map, pred_map], dim=2)  # Combine horizontally
            combined_grids.append(combined)
    elif is_double_channel_autoencoder and is_3D: # "3D_double_channel_autoencoder":
        for i in range(len(inputs)):  # Loop over the batch
            # Create a grid for the temporal event stack
            event_stack = inputs[i].squeeze(0)  # Remove the channel dimension: [1, D, H, W] -> [D, H, W]
            event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2).float()  # Combine horizontally  [1, H, D*W]
            writer.add_image(f"{prefix}/EventFrames_Sample_{i}", event_grid, global_step)
            
            # Combine GT and Prediction horizontally
            gt_map = targets[i].unsqueeze(0)  # Ground truth [1, H, W]
            pred_map = outputs[i]  # Predicted edge map [1, H, W]
            combined = torch.cat([gt_map, pred_map], dim=2).float()  # Combine horizontally [1, H, 2*W]
            combined_grids.append(combined)     
    elif is_autoencoder and not is_3D: #"denoise_autoencoder":
        # Autoencoder: Combine Event, GT, and Prediction
        for i in range(len(inputs)):
            event_frame = inputs[i]  # Event frame (single-channel)
            gt_map = targets[i]  # Ground truth edge map
            pred_map = outputs[i]  # Predicted edge map

            # Stack Event, GT, and Prediction vertically
            combined = torch.cat([event_frame, gt_map, pred_map], dim=2)  # Combine horizontally
            combined_grids.append(combined)
        
    elif is_autoencoder and is_3D: # "3D_denoise_autoencoder":  # 3D Denoise Model
        for i in range(len(inputs)):  # Loop over the batch
            # Create a grid for the temporal event stack
            event_stack = inputs[i].squeeze(0)  # Remove the channel dimension: [1, D, H, W] -> [D, H, W]
            event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2)  # Combine horizontally  [1, H, D*W]
            writer.add_image(f"{prefix}/EventFrames_Sample_{i}", event_grid, global_step)
            
            # Combine GT and Prediction horizontally
            gt_map = targets[i]  # Ground truth [1, H, W]
            pred_map = outputs[i]  # Predicted edge map [1, H, W]
            combined = torch.cat([gt_map, pred_map], dim=2)  # Combine horizontally [1, H, 2*W]
            combined_grids.append(combined)
    else:
        # Segmentation: Combine RGB, Event, Colorized GT, and Colorized Predictions
        rgb_images = inputs[:, :3]  # Extract RGB channels
        event_frames = inputs[:, 3:] if inputs.size(1) > 3 else None  # Extract event frames if available
        autoencoder_features = None
        if (inputs.size(1) > 4) and use_event and use_autoencoder: # triple input, rgb+event+autoencoder
            event_frames = inputs[:, 3].unsqueeze(1)
            autoencoder_features = inputs[:, 4].unsqueeze(1)
        for i in range(len(rgb_images)): 
            # Colorize GT and Predictions 
            colorized_pred = torch.tensor(np.array(colorize_mask_save(torch.argmax(outputs[i], dim=0).cpu(), save=False, print_message=False)).transpose(2, 0, 1))
            colorized_pred = colorized_pred.float() / 255.0  # Normalize to [0, 1]
            if use_gt_masks:
                colorized_gt = torch.tensor(np.array(colorize_mask_save(targets[i].cpu(), save=False, print_message=False)).transpose(2, 0, 1))
                colorized_gt = colorized_gt.float() / 255.0  # Normalize to [0, 1]
            else:   
                colorized_gt = torch.zeros_like(targets[i])  # Dummy tensor if no GT
                
            if autoencoder_features is not None: # triple case
                event_frame_3d = event_frames[i].repeat(3, 1, 1)  # Expand 1D event to 3D
                autoencoder_features_3d = autoencoder_features[i].repeat(3, 1, 1)  # Expand 1D to 3D
                combined = torch.cat([
                    rgb_images[i],  # RGB frame
                    event_frame_3d,  # Expanded Event frame
                    autoencoder_features_3d,  # Expanded autoencoder feature
                    colorized_gt,  # Colorized GT mask
                    colorized_pred  # Colorized Prediction
                ], dim=2)  # Combine horizontally"
            # Combine RGB, Event (if present), Colorized GT, and Predictions
            elif event_frames is None:
                combined = torch.cat([
                    rgb_images[i],  # RGB frame
                    colorized_gt,  # Colorized GT mask
                    colorized_pred  # Colorized Prediction
                ], dim=2)  # Combine horizontally
            elif (event_frames is not None) and (not is_3D) and (event_frames[i].size(0) > 1): # for 3 channel event + rgb
                combined = torch.cat([
                    rgb_images[i],  # RGB frame
                    event_frames[i],  # RGB event frame
                    colorized_gt,  # Colorized GT mask
                    colorized_pred  # Colorized Prediction
                ], dim=2)  # Combine horizontally
            elif event_frames is not None and not is_3D:
                event_frame_3d = event_frames[i].repeat(3, 1, 1)  # Expand 1D event to 3D
                combined = torch.cat([
                    rgb_images[i],  # RGB frame
                    event_frame_3d,  # Expanded Event frame
                    colorized_gt,  # Colorized GT mask
                    colorized_pred  # Colorized Prediction
                ], dim=2)  # Combine horizontally
                    
            elif event_frames is not None and is_3D: # for 3D segmentation
                # Create a grid for the temporal event stack
                
                event_stack = event_frames[i] #.squeeze(0)  # Remove the channel dimension: [1, D, H, W] -> [D, H, W]
                event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2)  # Combine horizontally  [1, H, D*W]
                writer.add_image(f"{prefix}/EventFrames_Sample_{i}", event_grid, global_step)
                
                combined = torch.cat([
                    rgb_images[i],  # RGB frame
                    colorized_gt,  # Colorized GT mask
                    colorized_pred  # Colorized Prediction
                ], dim=2)  # Combine horizontally
            
            combined_grids.append(combined)

    # Create a grid for all combined samples
    grid = make_grid(combined_grids, nrow=1, normalize=True, scale_each=True)

    # Log the grid to TensorBoard
    writer.add_image(f"{prefix}/Combined_Grid", grid, global_step=global_step)

def log_to_wandb(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    outputs: torch.Tensor, 
    step: int,
    epoch: int,
    prefix: str = "Train", 
    grid_size: int = 10, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs combined grids (RGB, Event, GT, and Predictions) to Weights & Biases for segmentation.
    For autoencoders, logs Event, GT, and Predictions.

    Args:
        inputs: Input tensor(s) - RGB+Event for segmentation, Event for autoencoder [B, C, H, W]
        targets: Ground truth tensor - masks for segmentation, edge maps for autoencoders [B, H, W]
        outputs: Model prediction tensor [B, num_classes, H, W] or [B, C, H, W]
        step: Current monotonic step for logging (for internal API correctness)
        epoch: Current epoch number (for dashboard visualization via custom metric)
        prefix: Prefix to distinguish between Train and Validation logs (default: "Train")
        grid_size: Number of samples to include in the grid (default: 10)
        config: Configuration dict containing model type flags (is_3D, is_autoencoder, etc.)
        
    Note:
        Handles multiple model types based on config flags:
        - Standard segmentation: RGB + events + GT + predictions
        - 3D models: Temporal event stacks + GT + predictions  
        - Autoencoders: Event input + GT + reconstructed output
        - Triple input: RGB + events + autoencoder features + GT + predictions
    """
    if config is None:
        config = {}
    
    # Limit the number of samples to `grid_size`
    inputs = inputs[:grid_size]
    targets = targets[:grid_size]
    outputs = outputs[:grid_size]

    combined_grids = []
    is_3D = config.get("is_3D", False)
    is_autoencoder = config.get("is_autoencoder")
    is_double_channel_autoencoder = config.get("is_double_channel_autoencoder", False)
    
    use_rgb = config.get("use_rgb")
    use_event = config.get("use_event")
    use_autoencoder = config.get("use_autoencoder")
    
    use_gt_masks = config.get("use_gt_masks")
    
    if is_double_channel_autoencoder and not is_3D:
        for i in range(len(inputs)):
            event_frame = inputs[i]
            gt_map = targets[i].unsqueeze(0)
            pred_map = outputs[i]
            combined = torch.cat([event_frame, gt_map, pred_map], dim=2)
            combined_grids.append(combined)
    elif is_double_channel_autoencoder and is_3D:
        for i in range(len(inputs)):
            event_stack = inputs[i].squeeze(0)
            event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2).float()
            wandb.log({f"{prefix}/EventFrames_Sample_{i}": wandb.Image(event_grid.cpu().numpy().transpose(1, 2, 0)), "epoch": epoch}, step=step, commit=False)
            
            gt_map = targets[i].unsqueeze(0)
            pred_map = outputs[i]
            combined = torch.cat([gt_map, pred_map], dim=2).float()
            combined_grids.append(combined)
    elif is_autoencoder and not is_3D:
        for i in range(len(inputs)):
            event_frame = inputs[i]
            gt_map = targets[i]
            pred_map = outputs[i]
            combined = torch.cat([event_frame, gt_map, pred_map], dim=2)
            combined_grids.append(combined)
    elif is_autoencoder and is_3D:
        for i in range(len(inputs)):
            event_stack = inputs[i].squeeze(0)
            event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2)
            wandb.log({f"{prefix}/EventFrames_Sample_{i}": wandb.Image(event_grid.cpu().numpy().transpose(1, 2, 0)), "epoch": epoch}, step=step, commit=False)
            
            gt_map = targets[i]
            pred_map = outputs[i]
            combined = torch.cat([gt_map, pred_map], dim=2)
            combined_grids.append(combined)
    else:
        # Segmentation: Combine RGB, Event, Colorized GT, and Colorized Predictions
        rgb_images = inputs[:, :3]
        event_frames = inputs[:, 3:] if inputs.size(1) > 3 else None
        autoencoder_features = None
        if (inputs.size(1) > 4) and use_event and use_autoencoder:
            event_frames = inputs[:, 3].unsqueeze(1)
            autoencoder_features = inputs[:, 4].unsqueeze(1)
        for i in range(len(rgb_images)):
            colorized_pred = torch.tensor(np.array(colorize_mask_save(torch.argmax(outputs[i], dim=0).cpu(), save=False, print_message=False)).transpose(2, 0, 1))
            colorized_pred = colorized_pred.float() / 255.0
            if use_gt_masks:
                colorized_gt = torch.tensor(np.array(colorize_mask_save(targets[i].cpu(), save=False, print_message=False)).transpose(2, 0, 1))
                colorized_gt = colorized_gt.float() / 255.0
            else:
                colorized_gt = torch.zeros_like(targets[i])
                
            if autoencoder_features is not None:
                event_frame_3d = event_frames[i].repeat(3, 1, 1)
                autoencoder_features_3d = autoencoder_features[i].repeat(3, 1, 1)
                combined = torch.cat([
                    rgb_images[i],
                    event_frame_3d,
                    autoencoder_features_3d,
                    colorized_gt,
                    colorized_pred
                ], dim=2)
            elif event_frames is None:
                combined = torch.cat([
                    rgb_images[i],
                    colorized_gt,
                    colorized_pred
                ], dim=2)
            elif (event_frames is not None) and (not is_3D) and (event_frames[i].size(0) > 1):
                combined = torch.cat([
                    rgb_images[i],
                    event_frames[i],
                    colorized_gt,
                    colorized_pred
                ], dim=2)
            elif event_frames is not None and not is_3D:
                event_frame_3d = event_frames[i].repeat(3, 1, 1)
                combined = torch.cat([
                    rgb_images[i],
                    event_frame_3d,
                    colorized_gt,
                    colorized_pred
                ], dim=2)
            elif event_frames is not None and is_3D:
                event_stack = event_frames[i]
                event_grid = torch.cat([event_stack[t].unsqueeze(0) for t in range(event_stack.size(0))], dim=2)
                wandb.log({f"{prefix}/EventFrames_Sample_{i}": wandb.Image(event_grid.cpu().numpy().transpose(1, 2, 0)), "epoch": epoch}, step=step, commit=False)
                
                combined = torch.cat([
                    rgb_images[i],
                    colorized_gt,
                    colorized_pred
                ], dim=2)
            
            combined_grids.append(combined)

    # Create a grid for all combined samples
    grid = make_grid(combined_grids, nrow=1, normalize=True, scale_each=True)

    # Log the grid to Weights & Biases with proper step and epoch
    wandb.log({f"{prefix}/Combined_Grid": wandb.Image(grid.cpu().numpy().transpose(1, 2, 0)), "epoch": epoch}, step=step, commit=True)

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    save_path: str, 
    best_val_loss: Optional[float] = None, 
    dataset: Optional[str] = None
) -> None:
    """
    Save the training checkpoint with model and optimizer states.

    Args:
        model: The PyTorch model being trained
        optimizer: The optimizer used during training
        epoch: The current epoch number
        save_path: Path to save the checkpoint file (.pth)
        best_val_loss: Best validation loss achieved so far (optional, for tracking)
        dataset: Dataset name/identifier for metadata (optional)
        
    Note:
        Saves model state_dict, optimizer state_dict, epoch, and optional metadata
        to enable resuming training from checkpoints.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    if dataset:
        checkpoint["dataset"] = dataset
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    load_path: Optional[str] = None, 
    device: str = "cuda"
) -> Union[Tuple[int, Optional[float]], torch.nn.Module]:
    """
    Load the checkpoint for training or inference.

    Args:
        model: The PyTorch model to load the state into
        optimizer: The optimizer to load the state into (optional, used for resuming training)
        load_path: Path to the checkpoint file (.pth)
        device: Device for loading the model - "cuda", "cpu", or specific GPU (default: "cuda")

    Returns:
        If optimizer is provided (training mode):
            Tuple[int, Optional[float]]: (start_epoch, best_val_loss) for resuming training
        If optimizer is not provided (inference mode):
            torch.nn.Module: The model with loaded weights
            
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist at load_path
        KeyError: If checkpoint is missing required 'model_state_dict' key
    """
    if not load_path or not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model weights loaded from {load_path}")
    else:
        raise KeyError("Checkpoint is missing 'model_state_dict'. Ensure you save the correct keys in your checkpoint.")

    if optimizer:
        # Load optimizer state for training
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state loaded.")
        else:
            print("Optimizer state not found in checkpoint. Proceeding without it.")

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))  # Defaults to infinity if not saved
        print(f"Resuming training from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
        return start_epoch, best_val_loss

    epoch = checkpoint.get("epoch", 0)
    return model, epoch


# Constants for logging synchronization
# Constants for logging synchronization
LOGGING_BATCH_LIMIT = 10      # Max number of batches to log images for per phase
# Automatically calculated offsets and multiplier
LOGGING_VAL_OFFSET = LOGGING_BATCH_LIMIT      # Offset for validation steps (10)
LOGGING_SUMMARY_OFFSET = LOGGING_BATCH_LIMIT * 2 # Offset for end-of-epoch summary (20) - Logged BEFORE inference
LOGGING_INFER_OFFSET = LOGGING_SUMMARY_OFFSET + 1 # Offset for inference steps (21) - Logged AFTER summary
LOGGING_STEP_MULTIPLIER = LOGGING_BATCH_LIMIT * 4 # Total steps reserved per epoch (40)

def get_device(device_id: int = 0) -> torch.device:
    """
    Determines and returns the appropriate PyTorch device (GPU or CPU).

    Args:
        device_id: The ID of the GPU to use if multiple are available. Defaults to 0.

    Returns:
        A torch.device object representing the selected device.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def classify_event_frame(
    event_path: str, 
    expand_dims: bool = True, 
    background_color: int = 255
) -> torch.Tensor:
    """
    Classify the event frame into two classes based on pixel intensity.
    
    Converts event camera grayscale images into binary classification:
    - Action (non-background pixels) -> 1
    - Background (background_color pixels) -> 0

    Args:
        event_path: Path to the event frame image file
        expand_dims: Whether to add a channel dimension [H, W] -> [1, H, W] (default: True)
        background_color: Pixel value considered as background (default: 255 for white)

    Returns:
        Binary classification mask as torch.Tensor [1, H, W] if expand_dims=True, else [H, W]
        Values are 0 (background) or 1 (action/event)
        
    Note:
        EventScapes uses 255 (white) as background, DSEC uses 127 (gray).
        Set background_color accordingly based on your dataset.
    """
    # Read the event frame
    event_frame = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the event frame is read correctly
    if event_frame is None:
        raise FileNotFoundError(f"Event frame not found at path: {event_path}")

    # Convert the grayscale image to a binary mask
    # If pixel intensity is 255 (white), set it to 0 (background)
    # Otherwise, set it to 1 (action region)
    classified_frame = np.where(event_frame == background_color, 0, 1).astype(np.uint8)

    # Convert to PyTorch tensor
    classified_frame = torch.tensor(classified_frame, dtype=torch.float32)

    # Expand dimensions for model input (e.g., [1, H, W])
    if expand_dims:
        classified_frame = classified_frame.unsqueeze(0)

    return classified_frame

def classify_event_frame_old(event_path: str, expand_dims: bool = True) -> torch.Tensor:
    """
    [DEPRECATED] Classify event frame using HSV color segmentation.
    
    Classify the event frame into two classes based on color:
    - Action (Red/Blue pixels) -> 1
    - Background (Black pixels) -> 0
    
    Args:
        event_path: Path to the event frame image file
        expand_dims: Whether to add a channel dimension [H, W] -> [1, H, W] (default: True)
    
    Returns:
        Binary classification mask as torch.Tensor [1, H, W] if expand_dims=True, else [H, W]
        
    Note:
        This is the old method using HSV color segmentation for red/blue event detection.
        Use classify_event_frame() instead for better performance with grayscale events.
    """
    # Read the event frame
    event_frame = cv2.imread(event_path)

    # Ensure the event frame is read correctly
    if event_frame is None:
        raise FileNotFoundError(f"Event frame not found at path: {event_path}")

    # Convert to HSV for better color segmentation
    hsv_frame = cv2.cvtColor(event_frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and blue in HSV
    lower_red = np.array([0, 50, 50])   # Adjust thresholds if needed
    upper_red = np.array([10, 255, 255])
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create masks for red and blue colors
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Combine red and blue masks to detect action regions
    action_mask = cv2.bitwise_or(red_mask, blue_mask)

    # Create a binary classification mask
    # Action regions (red/blue pixels) -> 1
    # Background (black pixels) -> 0
    classified_frame = (action_mask > 0).astype(np.uint8)

    # Convert to PyTorch tensor for model compatibility
    classified_frame = torch.tensor(classified_frame, dtype=torch.float32)

    # Expand dimensions for model input (e.g., [1, H, W])
    if expand_dims:
        classified_frame = classified_frame.unsqueeze(0)

    # Move tensor to the specified device (e.g., GPU
    return classified_frame

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an RGB image for model input.
    
    Performs: BGR->RGB conversion, normalization to [0,1], and channel reordering to CHW format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as torch.Tensor [3, H, W] normalized to [0, 1] range
        
    Note:
        Does not add batch dimension - use torch.unsqueeze(0) if needed for inference.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Convert to CHW format
    
    return torch.tensor(img, dtype=torch.float32)

def preprocess_mask(mask_path: str, num_of_classes: Optional[int] = None) -> torch.Tensor:
    """
    Load and preprocess a segmentation mask.
    
    Args:
        mask_path: Path to the mask image file
        num_of_classes: Number of segmentation classes (required)
        
    Returns:
        Preprocessed mask as torch.Tensor [H, W] with class indices
        
    Raises:
        FileNotFoundError: If mask file doesn't exist
        ValueError: If num_of_classes is not specified
    """
    # Load the mask as RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at path: {mask_path}")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    if num_of_classes is None:
        raise ValueError("Number of classes must be specified for mask preprocessing.")
    # Generate the full mapping for expected classes
    full_color_map = {(i, i, i): i for i in range(num_of_classes)}

    # Convert RGB mask to class indices
    mask_indices = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, class_idx in full_color_map.items():
        mask_indices[(mask == rgb).all(axis=-1)] = class_idx

    # Convert to tensor
    return torch.tensor(mask_indices, dtype=torch.long)

# Save the predicted mask as an image
def save_mask_as_image(mask, save_path, print_message=True):
    """
    Save a mask with class indices as a grayscale image.

    Args:
    - mask (np.ndarray): Mask with class indices (0 to num_of_out_classes - 1).
    - save_path (str): Path to save the image.

    Returns:
    - None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure the mask is uint8 (Pillow requires this format)
    mask_uint8 = mask.astype(np.uint8)
    
    # Save the mask as a grayscale image
    mask_image = Image.fromarray(mask_uint8, mode="L")  # Grayscale image
    mask_image.save(save_path)
    if print_message:
        print(f"Predicted mask saved to {save_path}")


# Save the original RGB image
def save_rgb_image(image: Union[torch.Tensor, np.ndarray], save_path: str) -> None:
    """
    Save RGB image to disk.
    
    Args:
        image: RGB image as torch.Tensor [1, 3, H, W] or [3, H, W], or np.ndarray [H, W, 3]
        save_path: Path to save the image file
        
    Note:
        Automatically converts tensors to numpy, scales [0,1] to [0,255], and handles
        channel/dimension reordering.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert from Tensor or NumPy array to uint8 format
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC
    image_uint8 = (image * 255).astype(np.uint8)  # Scale to 0-255
    rgb_image = Image.fromarray(image_uint8, mode="RGB")
    rgb_image.save(save_path)
    print(f"Input RGB image saved to {save_path}")
    
def save_event_frame(event_frame: Union[torch.Tensor, np.ndarray], save_path: str) -> None:
    """
    Save the processed event frame (binary mask) as an image.

    Args:
    - event_frame (torch.Tensor or np.ndarray): The binary event frame with shape [1, H, W].
    - save_path (str): Path to save the event frame image.

    Returns:
    - None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert torch.Tensor to np.ndarray if necessary
    if isinstance(event_frame, torch.Tensor):
        event_frame = event_frame.squeeze(0).cpu().numpy()  # Remove channel dimension and convert to numpy

    # Scale binary values to 0-255 for image saving
    event_frame_uint8 = (event_frame * 255).astype(np.uint8)

    # Convert to PIL Image and save
    event_image = Image.fromarray(event_frame_uint8, mode="L")  # Grayscale image
    event_image.save(save_path)
    print(f"Event frame saved to {save_path}")

def colorize_mask_save(
    mask: Union[torch.Tensor, np.ndarray], 
    save_path: Optional[str] = None, 
    print_message: bool = True, 
    save: bool = True, 
    class_num: int = 11
) -> Image.Image:
    """
    Colorize a segmentation mask using class-specific colors and optionally save it.

    Args:
        mask: Segmentation mask of shape [H, W] with class indices (0 to class_num-1)
        save_path: Path to save the colorized mask image (required if save=True)
        print_message: Whether to print save confirmation message (default: True)
        save: Whether to save the image to disk (default: True)
        class_num: Number of classes in the segmentation (default: 11)

    Returns:
        Colorized mask as PIL Image [H, W, 3] with RGB colors
        
    Note:
        Uses predefined color palettes for 11-class and 19-class segmentation.
        Each class is mapped to a unique RGB color for visualization.
    """
    # Define the colormap for 19 classes
    colormap_19 = [
        (128, 64, 128),   # Class 0 road
        (244, 35, 232),   # Class 1 sidewalk
        (70, 70, 70),     # Class 2 building
        (102, 102, 156),  # Class 3 wall
        (190, 153, 153),  # Class 4 fence
        (153, 153, 153),  # Class 5 pole
        (250, 170, 30),   # Class 6 traffic light
        (220, 220, 0),    # Class 7 traffic sign
        (107, 142, 35),   # Class 8 vegetation
        (152, 251, 152),  # Class 9 terrain
        (70, 130, 180),   # Class 10 sky
        (220, 20, 60),    # Class 11 person
        (255, 0, 0),      # Class 12 rider
        (0, 0, 142),      # Class 13 car
        (0, 0, 70),       # Class 14 truck
        (0, 60, 100),     # Class 15 bus
        (0, 80, 100),     # Class 16 train
        (0, 0, 230),      # Class 17 motorcycle
        (119, 11, 32)     # Class 18 bicycle
    ]
    colormap_11 = [
        (70, 130, 180),  # Class 0 sky
        (70, 70, 70),   # Class 1 building
        (190, 153, 153),     # Class 2  fence
        (220, 20, 60),  # Class 3 person rider (255, 0, 0)
        (153, 153, 153),  # Class 4 pole
        (128, 64, 128),  # Class 5 road
        (244, 35, 232),   # Class 6 sidewalk
        (107, 142, 35),    # Class 7  vegetation terrain (152, 251, 152)
        (0, 0, 142),   # Class 8  car truck bus train motorcycle bicycle
        (102, 102, 156),  # Class 9  wall 
        (220, 220, 0),   # Class 10 traffic sign ,traffic light (250, 170, 30)
    ]
    if save:
        # Ensure the directory exists
        if save_path is None:
            raise ValueError("save_path must be specified when saving the colorized mask.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create an RGB image with the same height and width as the mask
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply the colormap to each class index
    if class_num == 11:
        colormap = colormap_11
    else:
        colormap = colormap_19
        
    for class_idx, color in enumerate(colormap):
        colorized_mask[mask == class_idx] = color

    # Convert to PIL Image and save
    colorized_image = Image.fromarray(colorized_mask, mode="RGB")
    if save:
        colorized_image.save(save_path)
        if print_message:
            print(f"Colorized mask saved to {save_path}")
    return colorized_image
    
def save_loss_history(train_loss, val_loss, history_path):
    """
    Append the latest training and validation loss to the loss history JSON file.

    Args:
    - train_loss (float): Latest training loss.
    - val_loss (float): Latest validation loss.
    - history_path (str): Path to save the loss history.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    # Load existing history if it exists
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {"train_loss": [], "val_loss": []}
    
    # Append the latest losses
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    
    # Save the updated history to the file
    with open(history_path, "w") as f:
        json.dump(history, f)

def load_and_plot_loss_history(history_path: str) -> None:
    """
    Load loss history from a JSON file and plot it.

    Args:
        history_path: Path to the JSON file with loss history
        
    Note:
        Displays a matplotlib plot showing training and validation loss curves.
    """
    # Load loss history
    with open(history_path, "r") as f:
        history = json.load(f)

    train_loss_history = history["train_loss"]
    val_loss_history = history["val_loss"]

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_iou(
    predicted_mask: np.ndarray, 
    ground_truth_mask: np.ndarray, 
    num_of_out_classes: int
) -> Tuple[List[float], float]:
    """
    Calculate the Intersection over Union (IoU) for predicted and ground truth masks.

    Args:
        predicted_mask: Predicted mask of shape [H, W] with class indices (0 to num_classes-1)
        ground_truth_mask: Ground truth mask of shape [H, W] with class indices
        num_of_out_classes: Total number of segmentation classes

    Returns:
        Tuple containing:
            - iou_per_class (List[float]): IoU score for each class (length: num_of_out_classes)
            - mean_iou (float): Mean IoU across all classes that appear in the masks
            
    Note:
        Classes not present in either mask are excluded from mean IoU calculation
        to avoid division by zero and provide meaningful metrics.
    """
    
    # Convert tensors to NumPy arrays if necessary
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()
        
    iou_per_class = []
    
    for cls in range(num_of_out_classes):
        # Create binary masks for the current class
        pred_class = (predicted_mask == cls).astype(np.uint8)
        gt_class = (ground_truth_mask == cls).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.sum(pred_class & gt_class)
        union = np.sum(pred_class | gt_class)

        # Avoid division by zero
        if union == 0:
            iou = float('nan')  # Class not present in either prediction or ground truth
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    # Calculate mean IoU, excluding NaN values (classes not present in both masks)
    mean_iou = np.nanmean(iou_per_class)

    return iou_per_class, mean_iou

def calculate_weighted_iou(
    predicted_mask: np.ndarray, 
    ground_truth_mask: np.ndarray, 
    num_of_out_classes: int
) -> Tuple[List[float], float]:
    """
    Calculate the Weighted Intersection over Union (Weighted IoU) for segmentation masks.
    
    Weights each class's IoU by its frequency in the ground truth to give more importance
    to classes that occupy larger areas in the image.

    Args:
        predicted_mask: Predicted mask of shape [H, W] with class indices
        ground_truth_mask: Ground truth mask of shape [H, W] with class indices
        num_of_out_classes: Total number of segmentation classes

    Returns:
        Tuple containing:
            - iou_per_class (List[float]): IoU score for each class
            - weighted_mean_iou (float): Weighted mean IoU, weighted by class frequency in GT
            
    Note:
        Weighted IoU is useful for imbalanced datasets where some classes are much more
        frequent than others. Each class IoU is weighted by (class pixel count / total pixels).
    """
    # Convert tensors to NumPy arrays if necessary
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()
        
    iou_per_class = []
    class_weights = []

    # Compute IoU and weights for each class
    for cls in range(num_of_out_classes):
        # Create binary masks for the current class
        pred_class = (predicted_mask == cls).astype(np.uint8)
        gt_class = (ground_truth_mask == cls).astype(np.uint8)

        # Calculate intersection and union
        intersection = np.sum(pred_class & gt_class)
        union = np.sum(pred_class | gt_class)

        # Avoid division by zero
        if union == 0:
            iou = float('nan')  # Class not present in either prediction or ground truth
        else:
            iou = intersection / union

        # Append IoU
        iou_per_class.append(iou)

        # Calculate weight for the class based on ground truth
        class_weight = np.sum(gt_class) / ground_truth_mask.size
        class_weights.append(class_weight)

    # Normalize class weights
    total_weight = sum(class_weights)
    normalized_class_weights = [
        weight / total_weight if total_weight > 0 else 0 for weight in class_weights
    ]

    # Calculate weighted mean IoU
    weighted_mean_iou = sum(
        iou * weight if not np.isnan(iou) else 0
        for iou, weight in zip(iou_per_class, normalized_class_weights)
    )

    return iou_per_class, weighted_mean_iou



def extract_info_from_path(path):
    """
    Extract various details from the given save file path.

    Args:
    - path (str): The file path to parse.

    Returns:
    - info (dict): Extracted details as key-value pairs.
    """
    info = {}
    if path is None:
        raise ValueError("Path must be specified to extract information.")
    # Extract dataset name
    match = re.search(r'segmentation_model_weights/([^/]+)', path)
    if match:
        info['dataset'] = match.group(1)
    
    # Extract model type
    match = re.search(r'/([^/]+)_segmentation/', path)
    if match:
        info['model_type'] = match.group(1)
        
    # Extract epochs
    match = re.search(r'(\d+)epochs', path)
    if match:
        info['epochs'] = int(match.group(1))
    
    # Extract number of colors/classes
    match = re.search(r'_(\d+)color', path)
    if match:
        info['colors'] = int(match.group(1))
    
    # Extract event_ms
    match = re.search(r'_event_frame_(\d+)ms', path)
    if match:
        info['event_ms'] = int(match.group(1))
    
    # Extract loss type
    match = re.search(r'_([a-zA-Z]+)loss_', path)
    if match:
        info['loss'] = match.group(1)
    
    # Extract fusion type
    match = re.search(r'_([a-zA-Z]+)_fusion', path)
    if match:
        info['fusion'] = match.group(1)
    
    # Extract date and time
    match = re.search(r'_(\d{8}_\d{6})', path)
    if match:
        try:
            info['date_time'] = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            info['date_time'] = None
    
    return info


def print_model_info(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """
    Print comprehensive model information including architecture details and parameter count.
    
    Args:
        model: PyTorch model instance
        config: Configuration dictionary with model settings
    """
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    # Model type and architecture
    model_type = config.get('model_type', 'Unknown')
    print(f"Model Type: {model_type}")
    print(f"Model Class: {model.__class__.__name__}")
    
    # Encoder information for SMP-based models
    encoder_name = config.get('encoder_name')
    if encoder_name:
        print(f"Encoder Backbone: {encoder_name}")
        encoder_weights = config.get('encoder_weights', 'None')
        print(f"Pretrained Weights: {encoder_weights}")
    
    # Fusion information for dual-encoder models
    fusion_type = config.get('fusion_type')
    if fusion_type:
        print(f"Fusion Strategy: {fusion_type}")
    
    # Input/output configuration (only the essential info)
    num_classes = config.get('num_of_out_classes', 'Not specified')
    print(f"Number of Output Classes: {num_classes}")
    
    # Additional model-specific info
    use_rgb = config.get('use_rgb', False)
    use_event = config.get('use_event', False)
    use_autoencoder = config.get('use_autoencoder', False)
    
    # Event-related settings
    event_ms = config.get('event_ms')
    if event_ms and use_event:
        print(f"Event Accumulation Time: {event_ms}ms")
        
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nParameter Count:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable Parameters: {non_trainable_params:,}")
    print(f"  Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    
    input_modalities = []
    if use_rgb:
        input_modalities.append('RGB')
    if use_event:
        input_modalities.append('Events')
    if use_autoencoder:
        input_modalities.append('Autoencoder')
    
    if input_modalities:
        print(f"Input Modalities: {', '.join(input_modalities)}")
    
    print("="*80 + "\n")


def filter_non_empty_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter config to only include sections that have sub-elements.
    
    This removes top-level config entries that are not dictionaries or are empty,
    cleaning up the config before saving to YAML.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Filtered config containing only non-empty dictionary sections
        
    Example:
        >>> config = {'paths': {'model': 'path.pth'}, 'epochs': 50, 'empty': {}}
        >>> filtered = filter_non_empty_sections(config)
        >>> filtered
        {'paths': {'model': 'path.pth'}}
    """
    return {k: v for k, v in config.items() if isinstance(v, dict) and v}