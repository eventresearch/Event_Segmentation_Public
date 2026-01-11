"""
Inference script for multi-modal semantic segmentation models.

This script performs model inference on test datasets:
- Loads trained model checkpoints
- Runs inference on test data (RGB, events, autoencoder features)
- Calculates IoU metrics and accuracy
- Optionally saves predicted segmentation masks

Configuration hierarchy (same as training):
    general → model_specific_params → inference → dataset → paths

Usage:
    python infer.py --config path/to/config.yaml

Example:
    python infer.py --config unet_results/runs/training/trial_xyz/training_config.yaml
"""

from typing import Dict, Any
from infer.infer_model import infer_model
import torch
import json
import os
from datetime import datetime
import argparse    
import re
from utils.utilities import extract_info_from_path
from dataset.dataset import get_dataset
from models.model import get_model
from torch.utils.data import DataLoader
from torchsummary import summary
import yaml
from utils.utilities import create_dataset_config, load_checkpoint, print_model_info, log_to_wandb, log_to_tensorboard
import wandb

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

def resolve_model_params(config: Dict[str, Any], type: str = "inference") -> Dict[str, Any]:
    """
    Resolve and merge configuration parameters for inference.
    
    Similar to training config resolution but uses 'inference' section instead of 'training'.
    Merge order: general → model_specific_params → inference → dataset → paths
    
    Args:
        config: Full configuration dictionary from YAML file
        type: Config section to use - typically "inference"
        
    Returns:
        Resolved configuration dictionary with all merged parameters
        
    Note:
        Also pulls loss function from training config for metric calculation.
        Uses same hierarchical merge strategy as training for consistency.
    """
    # Step 1: Get model name from inference section
    model_name = config[type]["model"]
    
    # Step 2: Get model-specific parameters (e.g., encoder settings for SMP models)
    model_specific = config.get("model_specific_params", {}).get(model_name, {})
    
    # Step 3: Determine base model type
    model_type = model_specific.get("base_model", model_name)
    
    # Step 4: Merge config sections in hierarchical order
    # general → model_specific → inference → dataset → paths
    resolved_config = {
        **config.get("general", {}),       # General defaults
        **model_specific,                   # Model-specific overrides
        **config.get("inference", {}),      # Inference-specific params
        **config.get("dataset", {}),        # Dataset config
        **config.get("paths", {})           # Path config
    }
    
    # Step 5: Set normalized model_type
    resolved_config["model_type"] = model_type
    
    # Step 6: Import loss function from training config (needed for some metrics)
    resolved_config["loss"] = config.get("training", {}).get("loss", None)
    
    # Step 7: Handle base_loss override if specified
    real_loss = resolved_config.get("base_loss", None)
    if real_loss:    
        resolved_config["loss"] = real_loss
    
    # Step 8: Combine and return complete config
    combined_config = {**resolved_config, **config}
    return combined_config

if __name__ == "__main__":
    # config_path = "event_seg/model_config.yaml"
    # config_path = "unet_results/runs/training/auto_encoder_dual_encoder_unet/trial_20250131_100101/training_config.yaml"

    parser = argparse.ArgumentParser(description="Infer U-Net Model")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Please provide a valid config file path")

    config_path = args.config  # Load config dynamically

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    resolved_config = resolve_model_params(config, type="inference")
    
    input_folder =  resolved_config.get("input_folder")  # Path to the input folder
    batch_size = resolved_config.get("batch_size", 1)  # Batch size
    model_type = resolved_config.get("model_type")
    output_folder = resolved_config.get("output_folder")
    checkpoint_path = resolved_config.get("model_path")
    metrics_path = resolved_config.get("metric_path")
    num_of_workers = resolved_config.get("num_workers")
    save_masks = resolved_config.get("save_masks")   
    
    use_rgb = resolved_config.get("use_rgb")
    use_event = resolved_config.get("use_event")
    use_autoencoder = resolved_config.get("use_autoencoder")
    is_3D = resolved_config.get("is_3D")
    
    use_gt_masks = resolved_config.get("use_gt_masks")
    event_ms = resolved_config.get("event_ms")

    # Extract event_ms and loss_function from the checkpoint path
    model_info = extract_info_from_path(checkpoint_path)
    # event_ms = model_info["event_ms"] if use_event else None
    loss_function = model_info["loss"]
    train_date = model_info["date_time"]
    train_dataset = model_info["dataset"]

    # event_ms = checkpoint_path.split("_")[-3].split("ms")[0] if "event" in checkpoint_path else None
    # event_ms = extract_ms_from_path(checkpoint_path) if "event" in checkpoint_path else None
    # loss_function = extract_loss_from_path(checkpoint_path) 
    
    # Custom folder names
    def get_path(input_folder, folder_name, subfolder, flag):
        return f"{input_folder}/{folder_name}/{subfolder}" if flag else None 
    
    RGB_FOLDER_NAME = resolved_config.get("rgb_folder") # Default is "images"
    MASK_FOLDER_NAME = resolved_config.get("mask_folder")  # Default is "mask11"
    EVENT_FOLDER_NAME = resolved_config.get("event_folder")  # Default is "event_frames_{event_ms}ms"
    EVENT_FOLDER_NAME = EVENT_FOLDER_NAME.format(event_ms=event_ms)  if EVENT_FOLDER_NAME else None  # Default is "event_frames_{event_ms}ms""
    # Add custom time surface suffix if enabled
    if resolved_config.get("use_custom_time_surface", False) and EVENT_FOLDER_NAME:
        htl = resolved_config.get("high_to_low_ratio", 10.0)
        gamma = resolved_config.get("gamma", 2.0)
        r0 = resolved_config.get("r0")
        EVENT_FOLDER_NAME += f"_with_time_lookup_htl_{htl}_gamma_{gamma}"
        EVENT_FOLDER_NAME += f"_rational_r0_{r0}" if r0 else ""
    AUTOENCODER_FOLDER_NAME = resolved_config.get("autoencoder_folder").format(event_ms=event_ms)  # Default is "autoencoder_output"
    
    resolved_config["event_folder"] = EVENT_FOLDER_NAME
    resolved_config["autoencoder_folder"] = AUTOENCODER_FOLDER_NAME
    
    test_folder_name = resolved_config.get("test_folder_name", "test")
    # test paths
    TEST_MASK_DIR = get_path(input_folder,MASK_FOLDER_NAME,test_folder_name, use_gt_masks)
    TEST_IMAGE_DIR = get_path(input_folder,RGB_FOLDER_NAME,test_folder_name,use_rgb)
    TEST_EVENT_DIR = get_path(input_folder,EVENT_FOLDER_NAME,test_folder_name,use_event)
    TEST_AUTOENCODER_DIR = get_path(input_folder,AUTOENCODER_FOLDER_NAME,test_folder_name,use_autoencoder)
    
    # # Dataset and DataLoader
    num_of_mask_classes = resolved_config.get("num_of_mask_classes")  # Number of classes
    # time_steps = resolved_config.get("time_steps")  # Number of time steps
    # edge_method = resolved_config.get("edge_method")  # Edge detection method
    # test_dataset_config = {
    #     "model_type": model_type,
    #     "image_dir": TEST_IMAGE_DIR,
    #     "mask_dir": TEST_MASK_DIR,
    #     "event_dir": TEST_EVENT_DIR,
    #     "autoencoder_dir": TEST_AUTOENCODER_DIR,
    #     "time_steps": time_steps,
    #     "edge_method": edge_method,
    #     "num_of_mask_classes": num_of_mask_classes,
    # }
    test_dataset_config = create_dataset_config(resolved_config, TEST_IMAGE_DIR, TEST_MASK_DIR, TEST_EVENT_DIR, TEST_AUTOENCODER_DIR, is_train=False)

    dataset = get_dataset(test_dataset_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_of_workers, drop_last=True)
        
    tensorboard_save_log = config.get("tensorboard_save_log", False)
    tensorboard_save_activations = config.get("tensorboard_save_activations", False)
    
    # Initialize Weights & Biases
    wandb_save_log = resolved_config.get("wandb_save_log", False)
    if wandb_save_log:
        try:
            wandb_entity = resolved_config.get("wandb_entity", None)
            wandb_project = resolved_config.get("wandb_project", "event-segmentation")
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_ms = resolved_config.get("event_ms")
            use_event = resolved_config.get("use_event")
            use_autoencoder = resolved_config.get("use_autoencoder")
            custom_time_surface = resolved_config.get("use_custom_time_surface", False)
            htl = resolved_config.get("high_to_low_ratio")
            r0 = resolved_config.get("r0")
            gamma = resolved_config.get("gamma")
            fusion_type = resolved_config.get("fusion_type", None)
            loss = resolved_config.get("loss")
            exp_suffix = f"{'_event_frame_' + str(event_ms) + 'ms' if (use_event or use_autoencoder) else ''}"
            if custom_time_surface:
                exp_suffix += f"_custom_time_surface_htl{htl}_gamma{gamma}"
                exp_suffix += f"_rational_r0{r0}" if r0 is not None else ""
            exp_suffix += f"_{loss}loss"
            if fusion_type:
                exp_suffix += f"_{fusion_type}_fusion"
            wandb_run_name = resolved_config.get("wandb_run_name", None)
            if wandb_run_name is None:
                wandb_run_name = f"Inference_{resolved_config.get('model', 'model')}{exp_suffix}_{current_timestamp}"
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_run_name,
                config=resolved_config,
                resume="allow",
                id=f"infer_{current_timestamp}"
            )
            # Define metric steps
            wandb.define_metric("epoch")
            wandb.define_metric("inference_mean_iou", step_metric="epoch")
            wandb.define_metric("inference_weighted_mean_iou", step_metric="epoch")
            wandb.define_metric("inference_loss", step_metric="epoch")
            print(f"✓ Weights & Biases initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Weights & Biases: {e}")
            wandb_save_log = False
            
    resolved_config["wandb_save_log"] = wandb_save_log

    # Load the provided model
    try:
        if not checkpoint_path:
            raise ValueError("Neither a preloaded model nor a valid checkpoint path is provided.")
        print("Loading model from checkpoint...")
        model = get_model(resolved_config).to(DEVICE)
        
        # Print comprehensive model information
        print_model_info(model, resolved_config)
        
        # if torch.__version__ >= "2.0.0":
        #     model = torch.compile(model)
        model, epoch = load_checkpoint(model, load_path=checkpoint_path, device=DEVICE)
        # Apply torch.compile for faster inference (only on PyTorch 2.0+)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Run inference 
    is_autoencoder = resolved_config.get("is_autoencoder", False)
    
    results = infer_model(
        dataloader=dataloader,
        device=DEVICE,
        model=model,
        config=resolved_config
    ) 
    if not is_autoencoder:

        average_weighted_mean_iou, average_classwise_iou, general_mean_iou, avg_infer_loss = results
        # Calculate General Mean IoU
        # valid_class_iou = [iou for iou in average_classwise_iou if iou > 0]
        # general_mean_iou = sum(valid_class_iou) / len(valid_class_iou) if valid_class_iou else 0.0

        # Print Metrics
        print(f"Average Weighted Mean IoU: {average_weighted_mean_iou:.4f}")
        print(f"General Mean IoU: {general_mean_iou:.4f}")
        print("Average Classwise IoU:")
        for cls, avg_iou in enumerate(average_classwise_iou):
            print(f"Class {cls}: {avg_iou:.4f}")

        # Save Metrics with Timestamp, Model Type, and Event MS (if applicable)
        metrics = {
            "train_timestamp": train_date.strftime("%Y-%m-%d %H:%M:%S"),
            "current_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_dataset": train_dataset,
            "epochs": epoch,
            "best_val_epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "num_of_out_classes": num_of_mask_classes,
            "loss_function": loss_function,
            # "fusion_type": fusion_type if fusion_type else None,
            "event_ms": event_ms if event_ms else None,  # Include event_ms only for attention models
            "average_weighted_mean_iou": average_weighted_mean_iou,
            "general_mean_iou": general_mean_iou,
            "average_classwise_iou": average_classwise_iou,
        }

        # Append to JSON File
        try:
            # Load existing metrics if the file exists
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    existing_metrics = json.load(f)
            else:
                existing_metrics = []

            # Append new metrics
            existing_metrics.append(metrics)

            # Save back to the file
            with open(metrics_path, "w") as f:
                json.dump(existing_metrics, f, indent=4)

            print(f"Metrics appended to {metrics_path}")

        except Exception as e:
            print(f"Error saving metrics: {e}")
    else:
        avg_infer_loss = results
        print(f"Average Inference Loss: {avg_infer_loss:.4f}")
        
    if wandb_save_log:
        wandb_log_dict = {
            "epoch": epoch,
            "inference_loss": avg_infer_loss
        }
        if not is_autoencoder:
            wandb_log_dict.update({
                "inference_mean_iou": general_mean_iou,
                "inference_weighted_mean_iou": average_weighted_mean_iou,
            })
        wandb.log(wandb_log_dict, commit=True)
        wandb.finish()