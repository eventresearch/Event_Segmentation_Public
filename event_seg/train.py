"""
Training script for multi-modal semantic segmentation models.

This script orchestrates the complete training pipeline:
- Configuration loading and resolution with hierarchical parameter merging
- Dataset creation for RGB, events, and autoencoder features
- Model instantiation based on config (SMP UNet, Segformer, DeepLab, etc.)
- Training loop with validation and optional inference
- Checkpoint saving and loss history tracking

Configuration hierarchy (later overrides earlier):
    general → model_specific_params → training → dataset → paths

Usage:
    python train.py --config path/to/config.yaml

Example:
    python train.py --config event_seg/model_config.yaml
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import get_dataset
from models.model import get_model
from train.train_model import train_model
from train.loss import get_loss_function
from utils.utilities import create_dataset_config, print_model_info
from datetime import datetime
import numpy as np
import random
import argparse
import yaml

import os
os.environ['WANDB_API_KEY'] = 'YOUR_WANDB_API_KEY_HERE'

def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value to use for all random number generators
        
    Note:
        Sets seeds for Python random, NumPy, PyTorch CPU and CUDA.
        Also enables deterministic CUDA operations (may impact performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary with all parameters
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_model_params(config: Dict[str, Any], type: str = "training") -> Dict[str, Any]:
    """
    Resolve and merge configuration parameters with hierarchical precedence.
    
    This function implements the core configuration resolution logic by merging
    parameters from multiple config sections in a specific order. Later sections
    override earlier ones to allow model-specific customization while maintaining
    shared defaults.
    
    Merge order (each step overrides previous):
        1. general: Shared defaults for all models
        2. model_specific_params[model_name]: Model-specific overrides
        3. training: Training-specific parameters
        4. dataset: Dataset configuration
        5. paths: File paths and directories
    
    Args:
        config: Full configuration dictionary from YAML file containing all sections
        type: Config section to use for model selection - "training" or "inference"
        
    Returns:
        Resolved configuration dictionary with:
            - All merged parameters from the hierarchy
            - model_type: Normalized base model name (e.g., "smp_unet")
            - Updated inference config with model name and event_ms
            - Real loss function if base_loss is specified
            
    Example:
        >>> config = load_config('model_config.yaml')
        >>> resolved = resolve_model_params(config, type='training')
        >>> resolved['model_type']  # 'smp_unet'
        >>> resolved['encoder_name']  # 'resnet34' from model_specific_params
        
    Note:
        The merge order is critical:
        - general: Provides sensible defaults (e.g., num_classes=11)
        - model_specific_params: Allows per-model customization (e.g., encoder_name)
        - training: Runtime parameters (e.g., batch_size, epochs)
        - dataset/paths: Environment-specific settings
        
        This design allows a single config file to define multiple models while
        sharing common parameters and avoiding duplication.
    """
    # Step 1: Get the model name from the specified config section (training or inference)
    model_name = config[type]["model"]
    
    # Step 2: Get model-specific parameters for this model (e.g., encoder_name for SMP models)
    # If model has no specific params, returns empty dict
    model_specific = config.get("model_specific_params", {}).get(model_name, {})
    
    # Step 3: Determine the base model type
    # model_specific_params can specify a different base_model (e.g., "smp_dual_encoder_unet" uses "smp_unet" base)
    # If not specified, the model_name itself is the model_type
    model_type = model_specific.get("base_model", model_name)
    
    # Step 4: Merge all config sections in order of precedence (left to right, later overrides earlier)
    # This is the CRITICAL merge order:
    # general (defaults) → model_specific (model overrides) → training (runtime) → dataset → paths
    resolved_config = {
        **config.get("general", {}),      # Start with general defaults
        **model_specific,                  # Override with model-specific params
        **config.get("training", {}),      # Override with training params
        **config.get("dataset", {}),       # Override with dataset params
        **config.get("paths", {})          # Override with path params
    }
    
    # Step 5: Set the normalized model_type for factory functions
    resolved_config["model_type"] = model_type
    
    # Step 6: Sync inference config with training config
    config["inference"]["model"] = model_name
    config["inference"]["event_ms"] = config["training"]["event_ms"]
    config["inference"]["use_custom_time_surface"] = config["training"].get("use_custom_time_surface", False)
    config["inference"]["high_to_low_ratio"] = config["training"].get("high_to_low_ratio", 10.0)
    config["inference"]["r0"] = config["training"].get("r0")
    config["inference"]["gamma"] = config["training"].get("gamma", 2.0)

    # Step 7: Handle base_loss override for compound losses
    # Some models define a base_loss (e.g., "dice") with additional params
    # This ensures the actual loss function name is set correctly
    real_loss = resolved_config.get("base_loss", None)
    if real_loss:    
        resolved_config["loss"] = real_loss  # Use the base loss name
    
    # Step 8: Combine resolved config with original config for complete access
    combined_config = {**resolved_config, **config}
    return combined_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument("--config", type=str, default="event_seg/model_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = args.config  # Load config dynamically
    config = load_config(config_path)
    resolved_config = resolve_model_params(config, type="training")
    # dataset_config = config["dataset"]
    # paths_config = config["paths"]
    # Hyperparameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = resolved_config.get("seed", 42)
    set_random_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    inference_interval  = resolved_config.get("inference_interval")
    
    input_folder =  resolved_config.get("input_folder")  # Path to the input folder
    dataset = input_folder.split("/")[-1]
    if dataset == "unknown":
        raise ValueError("Dataset is not recognized. Please provide a valid dataset folder.")
    epochs = resolved_config.get("epochs", 5)  # Number of epochs
    batch_size = resolved_config.get("batch_size", 1)  # Batch size
    learning_rate = resolved_config.get("learning_rate", 1e-4)  # Learning rate
    event_ms = resolved_config.get("event_ms")
    save_dir = f"{resolved_config.get('save_dir')}/{dataset}"
    result_dir = f"{resolved_config.get('result_dir')}/{dataset}"
    loss = resolved_config.get("loss")
    criterion = get_loss_function(loss, resolved_config)
    model_type = resolved_config.get("model_type")
    enable_inference_during_training = resolved_config.get("enable_inference_during_training")
    dont_use_validation = resolved_config.get("dont_use_validation")
    
    # Custom folder names
    use_rgb = resolved_config.get("use_rgb")
    use_event = resolved_config.get("use_event")
    use_autoencoder = resolved_config.get("use_autoencoder")
    is_3D = resolved_config.get("is_3D")
    
    def get_path(input_folder, folder_name, subfolder, flag):
        return f"{input_folder}/{folder_name}/{subfolder}" if flag else None 
    
    RGB_FOLDER_NAME = resolved_config.get("rgb_folder") # Default is "images"
    MASK_FOLDER_NAME = resolved_config.get("mask_folder")  # Default is "mask11"
    EVENT_FOLDER_NAME = resolved_config.get("event_folder")  # Default is "event_frames_{event_ms}ms"
    AUTOENCODER_FOLDER_NAME = resolved_config.get("autoencoder_folder")  # Default is "autoencoder_output"
    EVENT_FOLDER_NAME = EVENT_FOLDER_NAME.format(event_ms=event_ms)  if EVENT_FOLDER_NAME else None  # Default is "event_frames_{event_ms}ms"
    # Add custom time surface suffix if enabled
    if resolved_config.get("use_custom_time_surface", False) and EVENT_FOLDER_NAME:
        htl = resolved_config.get("high_to_low_ratio", 10.0)
        r0 = resolved_config.get("r0")
        gamma = resolved_config.get("gamma", 2.0)
        EVENT_FOLDER_NAME += f"_with_time_lookup_htl_{htl}_gamma_{gamma}"
        EVENT_FOLDER_NAME += f"_rational_r0_{r0}" if r0 else ""
    AUTOENCODER_FOLDER_NAME = AUTOENCODER_FOLDER_NAME.format(event_ms=event_ms)  if AUTOENCODER_FOLDER_NAME else None  # Default is "autoencoder_output"
    
    resolved_config["event_folder"] = EVENT_FOLDER_NAME # Update the config 
    resolved_config["autoencoder_folder"] = AUTOENCODER_FOLDER_NAME # Update the config
    
    # train paths
    train_folder_name = resolved_config.get("train_folder_name", "train")
    val_folder_name = resolved_config.get("val_folder_name", "val")
    test_folder_name = resolved_config.get("test_folder_name", "test")

    TRAIN_MASK_DIR = get_path(input_folder,MASK_FOLDER_NAME,train_folder_name,True)
    VAL_MASK_DIR = get_path(input_folder,MASK_FOLDER_NAME,val_folder_name,True)
    
    TRAIN_IMAGE_DIR = get_path(input_folder,RGB_FOLDER_NAME,train_folder_name,use_rgb)
    VAL_IMAGE_DIR = get_path(input_folder,RGB_FOLDER_NAME,val_folder_name,use_rgb)
    
    TRAIN_EVENT_DIR = get_path(input_folder,EVENT_FOLDER_NAME,train_folder_name,use_event)
    VAL_EVENT_DIR = get_path(input_folder,EVENT_FOLDER_NAME,val_folder_name,use_event)
    
    TRAIN_AUTOENCODER_DIR = get_path(input_folder,AUTOENCODER_FOLDER_NAME,train_folder_name,use_autoencoder)
    VAL_AUTOENCODER_DIR = get_path(input_folder,AUTOENCODER_FOLDER_NAME,val_folder_name,use_autoencoder)

    # if the enable_inference_during_training is True, then test dataset is required, but even if it is False, since the paths are dummy, it is not a problem
    TEST_MASK_DIR = get_path(input_folder,MASK_FOLDER_NAME,test_folder_name, True)
    TEST_IMAGE_DIR = get_path(input_folder,RGB_FOLDER_NAME,test_folder_name,use_rgb)
    TEST_EVENT_DIR = get_path(input_folder,EVENT_FOLDER_NAME,test_folder_name,use_event)
    TEST_AUTOENCODER_DIR = get_path(input_folder,AUTOENCODER_FOLDER_NAME,test_folder_name,use_autoencoder)

    
    num_of_classes = resolved_config.get("num_of_out_classes")  # Number of classes
    fusion_type = resolved_config.get("fusion_type", None)
    model_path = resolved_config.get("model_path", None)
    history_path = resolved_config.get("history_path", None)
    # Save paths
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_PATH = (
        f"{save_dir}/{model_type}_segmentation/{epochs}epochs_{num_of_classes}color"
        f"{'_event_frame_' + str(event_ms) + 'ms' if (use_event or use_autoencoder) else ''}"
        f"{'_custom_time_surface_htl' + str(resolved_config.get('high_to_low_ratio')) + '_gamma' + str(resolved_config.get('gamma')) + (('_rational_r0' + str(resolved_config.get('r0'))) if resolved_config.get('r0') else '') if resolved_config.get('use_custom_time_surface', False) else ''}" 
        f"_{loss}loss"
        f"{'_' + fusion_type + '_fusion' if fusion_type else ''}"
        f"_{datetime_str}/best_model.pth"
    )
    if model_path:
        SAVE_PATH = model_path
        
    HISTORY_PATH = (
        f"{result_dir}/loss/loss_history_{model_type}_{epochs}epochs_{num_of_classes}color"
        f"{'_event_frame_' + str(event_ms) + 'ms' if (use_event or use_autoencoder) else ''}"
        f"{'_custom_time_surface_htl' + str(resolved_config.get('high_to_low_ratio')) + '_gamma' + str(resolved_config.get('gamma')) + (('_rational_r0' + str(resolved_config.get('r0'))) if resolved_config.get('r0') else '') if resolved_config.get('use_custom_time_surface', False) else ''}" 
        f"_{loss}loss"
        f"{'_' + fusion_type + '_fusion' if fusion_type else ''}"
        f"_{datetime_str}.json"
    )
    
    if history_path:
        HISTORY_PATH = history_path
        
    resolved_config["model_path"] = SAVE_PATH
    resolved_config["history_path"] = HISTORY_PATH
    
    # Create dataset configurations using the helper function
    train_dataset_config = create_dataset_config(resolved_config, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TRAIN_EVENT_DIR, TRAIN_AUTOENCODER_DIR, is_train=True)
    val_dataset_config = create_dataset_config(resolved_config, VAL_IMAGE_DIR, VAL_MASK_DIR, VAL_EVENT_DIR, VAL_AUTOENCODER_DIR, is_train=False)
    test_dataset_config = create_dataset_config(resolved_config, TEST_IMAGE_DIR, TEST_MASK_DIR, TEST_EVENT_DIR, TEST_AUTOENCODER_DIR, is_train=False)
    
    # Load datasets
    train_dataset = get_dataset(train_dataset_config)    
    val_dataset = get_dataset(val_dataset_config)
    
    if dont_use_validation:
        train_dataset = train_dataset + val_dataset

    test_dataset = None
    test_loader = None
    if enable_inference_during_training:
        test_dataset = get_dataset(test_dataset_config)

    use_consecutive_loss = resolved_config.get("use_consecutive_loss")
    train_shuffle = False if use_consecutive_loss else True

    num_workers = resolved_config.get("num_workers")
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, generator=generator, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    if enable_inference_during_training:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    # Instantiate the model
    model = get_model(resolved_config).to(DEVICE)
    
    # Print comprehensive model information
    print_model_info(model, resolved_config)
    
    # Optimize model execution with torch.compile (only for PyTorch 2.0+)
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model)  # Improves execution speed
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        dataset=input_folder,
        enable_inference_during_training=enable_inference_during_training,
        test_loader=test_loader,
        inference_interval=inference_interval,
        config=resolved_config,
        datetime=datetime_str
    )
