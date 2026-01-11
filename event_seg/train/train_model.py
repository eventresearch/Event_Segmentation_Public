"""
Training module for semantic segmentation models.

This module provides the main training loop for various segmentation architectures
including standard U-Net, dual-encoder models, autoencoders, and 3D models.

Key Features:
    - Multi-modal training (RGB, events, autoencoder features, triple-input)
    - Checkpoint management with resume capability
    - TensorBoard logging with visualization
    - Validation during training with IoU metrics
    - Optional inference during training at specified intervals
    - Consecutive frame loss for temporal consistency (with EMA and spike detection)
    - Configurable checkpoint saving strategies (loss-based or IoU-based)

Supported Model Types:
    - Standard segmentation: RGB images only
    - Event-based: RGB + event frames
    - Autoencoder: Event frames → edge maps
    - Triple-input: RGB + events + autoencoder features
    - 3D models: Temporal sequences with 3D convolutions

Training Configuration:
    Training behavior is controlled via config dictionary containing:
    - Model architecture and parameters
    - Loss function and optimizer settings
    - Dataset paths and preprocessing options
    - Checkpoint and logging paths
    - TensorBoard and inference settings
"""

import os
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.utilities import (
    get_device,
    filter_non_empty_sections,
    save_checkpoint,
    save_loss_history,
    log_to_tensorboard,
    log_to_wandb,
    LOGGING_BATCH_LIMIT,
    LOGGING_STEP_MULTIPLIER,
    LOGGING_SUMMARY_OFFSET,
    LOGGING_INFER_OFFSET,
    load_checkpoint,
    get_tensorboard_writer
)
from infer.infer_model import infer_model
from train.loss import get_loss_function
import re
import yaml
import json

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    dataset: str = "",
    enable_inference_during_training: bool = False,
    test_loader: Optional[DataLoader] = None,
    inference_interval: int = 1,
    config: Optional[Dict[str, Any]] = None,
    datetime: str = datetime.now().strftime("%Y%m%d_%H%M%S")
) -> None:
    """
    Main training loop for semantic segmentation models.
    
    This function handles the complete training pipeline including:
    - Multi-modal input support (RGB, events, autoencoder features)
    - Checkpoint saving and resumption
    - Validation with IoU metrics
    - TensorBoard logging with visualizations
    - Optional inference during training
    - Consecutive frame loss for temporal consistency
    
    The training loop automatically detects model type from the batch structure:
    - 2 elements: Standard RGB segmentation (images, masks)
    - 3 elements: Dual-input (images, masks, events/autoencoder)
    - 4 elements: Triple-input (images, masks, events, autoencoder)
    
    Args:
        model: PyTorch model to train (nn.Module)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Total number of training epochs
        criterion: Loss function (from get_loss_function())
        optimizer: PyTorch optimizer (Adam, SGD, etc.)
        device: Device to train on (cuda/cpu)
        dataset: Dataset name for logging (default: "")
        enable_inference_during_training: Run inference on test set during training
        test_loader: DataLoader for test set (required if enable_inference_during_training=True)
        inference_interval: Run inference every N epochs (default: 1)
        config: Configuration dictionary containing:
            - input_folder: Dataset path
            - model_path: Checkpoint save path
            - history_path: Loss history save path
            - tensorboard_save_log: Enable TensorBoard logging
            - is_autoencoder: Training autoencoder model
            - is_double_channel_autoencoder: 2-channel autoencoder output
            - is_3D: Using 3D convolutions for temporal modeling
            - save_checkpoint_with_mean_iou: Save based on IoU instead of loss
            - save_interim_checkpoints: Save checkpoints at intervals
            - save_interim_checkpoint_interval: Epochs between interim saves
            - use_consecutive_loss: Enable temporal consistency loss
            - consecutive_loss_warmup: Warmup steps before applying consecutive loss
            - warmup_epochs: Warmup epochs before consecutive loss
            - lambda_consecutive: Weight for consecutive loss
            - ema_decay: Decay factor for EMA smoothing
            - consecutive_loss_reset_factor: Reset threshold for spike detection
        datetime: Timestamp for this training run (default: current time)
        
    Returns:
        None (saves checkpoints and logs to disk)
        
    Raises:
        ValueError: If batch structure doesn't match expected format
        
    Example:
        >>> # Standard training
        >>> config = load_config('config.yaml')
        >>> model = get_model(config)
        >>> criterion = get_loss_function('FocalCEDice', config)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> train_model(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     epochs=100,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     device=device,
        ...     config=config
        ... )
        
    Note:
        - Automatically resumes from checkpoint if model_path exists
        - Saves best model based on validation loss (or IoU if configured)
        - Logs first 10 batches of each epoch to TensorBoard
        - Consecutive loss uses LPIPS for perceptual similarity
        - EMA smoothing prevents sudden spikes from affecting training
    """
    dataset = config.get("input_folder")
    save_path = config.get("model_path")
    history_path = config.get("history_path")
    tensorboard_save_log = config.get("tensorboard_save_log", False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    best_val_loss = float("inf")
    best_infer_loss = float("inf")
    best_val_mean_iou = float("-inf")
    # train_loss_history = []
    # val_loss_history = []
    is_autoencoder = config.get("is_autoencoder", False)
    is_double_channel_autoencoder = config.get("is_double_channel_autoencoder", False)
    if is_double_channel_autoencoder:
        print("using double channel autoencoder")
    is_3D = config.get("is_3D")
    save_checkpoint_with_mean_iou = config.get("save_checkpoint_with_mean_iou", False)
    start_epoch = 0
    trial_timestamp = None
    if os.path.exists(save_path):
        print(f"Checkpoint found at {save_path}. Resuming training...")
        start_epoch, best_metric = load_checkpoint(model, optimizer, save_path, device)
        best_val_loss = best_metric
        best_val_mean_iou = best_metric
        print(f"Resuming from epoch {start_epoch}")

    save_interim_checkpoints = config.get("save_interim_checkpoints", False)
    save_interim_checkpoint_interval = config.get("save_interim_checkpoint_interval", 5)
    if save_interim_checkpoint_interval is not None and save_interim_checkpoint_interval < 1:
        save_interim_checkpoint_interval = 1
        print("Invalid save_interim_checkpoint_interval. Defaulting to 1.")
    save_folder = None
    if save_interim_checkpoints:
        save_folder = os.path.dirname(save_path)
    
    match = re.search(r'(\d{8}_\d{6})/[^/]+\.pth$', save_path)
    if match:
        trial_timestamp = match.group(1)  # Extract the timestamp part
        print(f"Trial timestamp: {trial_timestamp}")
    else:
        print("No timestamp found in the save path.")
    
    # TensorBoard writer
    # Get the parent directory of history_path
    history_dir = os.path.dirname(os.path.dirname(history_path))
    history_parent_name = os.path.dirname(history_dir)
    history_name = os.path.basename(history_path)
    
    save_parent_dir = os.path.dirname(os.path.dirname(save_path))
    save_base_name = os.path.basename(save_parent_dir)
    
    # Compose experiment name suffix for logs
    event_ms = config.get("event_ms")
    use_event = config.get("use_event")
    use_autoencoder = config.get("use_autoencoder")
    custom_time_surface = config.get("use_custom_time_surface", False)
    htl = config.get("high_to_low_ratio")
    r0 = config.get("r0")
    gamma = config.get("gamma")
    fusion_type = config.get("fusion_type", None)
    loss = config.get("loss")
    exp_suffix = f"{'_event_frame_' + str(event_ms) + 'ms' if (use_event or use_autoencoder) else ''}"
    if custom_time_surface:
        exp_suffix += f"_custom_time_surface_htl{htl}_gamma{gamma}"
        exp_suffix += f"_rational_r0{r0}" if r0 is not None else ""
    exp_suffix += f"_{loss}loss"
    if fusion_type:
        exp_suffix += f"_{fusion_type}_fusion"
    base_log_dir = f"{history_parent_name}/{save_base_name}{exp_suffix}"
    logdir = None
    
    # Append 'tensorboard_logs' to the parent directory with the current timestamp
    # log_dir = os.path.join(parent_dir, "tensorboard_logs_train", save_base_name, datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = None  
    if tensorboard_save_log:
        if trial_timestamp is None:
            writer, logdir = get_tensorboard_writer(base_log_dir=base_log_dir, datetime=datetime)
        else:
            writer, logdir = get_tensorboard_writer(base_log_dir=base_log_dir, trial_timestamp=trial_timestamp, datetime=datetime)
    
    # Initialize Weights & Biases
    wandb_save_log = config.get("wandb_save_log", False)
    if wandb_save_log:
        try:
            wandb_entity = config.get("wandb_entity", None)  # Team/entity name
            wandb_project = config.get("wandb_project", "event-segmentation")
            wandb_run_name = config.get("wandb_run_name", None)
            if wandb_run_name is None:
                wandb_run_name = f"{config.get('model', 'model')}{exp_suffix}_{trial_timestamp if trial_timestamp else datetime}"
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_run_name,
                config=config,
                resume="allow",
                id=trial_timestamp if trial_timestamp else None
            )
            # Define epoch as the step metric for all training metrics
            wandb.define_metric("epoch")
            wandb.define_metric("step")
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("val_loss", step_metric="epoch")
            wandb.define_metric("val_mean_iou", step_metric="epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            wandb.define_metric("inference_mean_iou", step_metric="epoch")
            wandb.define_metric("inference_weighted_mean_iou", step_metric="epoch")
            wandb.define_metric("inference_loss", step_metric="epoch")
             # Magic fix: This forces all image logs to use 'epoch' as their x-axis
            wandb.define_metric("Train/*", step_metric="epoch") 
            wandb.define_metric("Validation/*", step_metric="epoch")
            wandb.define_metric("Inference/*", step_metric="epoch")
            
            print(f"✓ Weights & Biases initialized successfully")
            if wandb_entity:
                print(f"  Entity: {wandb_entity}")
            print(f"  Project: {wandb_project}")
            print(f"  Run: {wandb_run_name}")
            # Only watch model for basic tracking (no gradients to save space)
            # Set log=None to disable gradient/parameter logging which is storage-heavy
            # This matches TensorBoard behavior which doesn't log gradients by default
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Weights & Biases: {e}")
            print(f"  Continuing with TensorBoard only. To fix:")
            print(f"  1. Check if project '{wandb_project}' exists in your W&B account")
            print(f"  2. Run: wandb login --relogin")
            print(f"  3. Or set wandb_save_log: false in config to disable W&B")
            wandb_save_log = False  # Disable wandb logging for this run

    history_path = os.path.join(logdir, history_name)
    save_config_path = f"{logdir}/training_config.yaml"
    
    config["paths"]["model_path"] = save_path
    config["paths"]["history_path"] = history_path
    
    config["model_path"] = save_path
    config["history_path"] = history_path

    with open(save_config_path, "w") as f:
        # Filter the config to only include sections that have sub-elements
        filtered_config = filter_non_empty_sections(config)
        yaml.dump(filtered_config, f, default_flow_style=False, sort_keys=False)
        print(f"Config file saved to {save_config_path}")
    
    config["tensorboard_save_activations"] = False
    config["tensorboard_save_log"] = tensorboard_save_log
    config["wandb_save_log"] = wandb_save_log  # Update config with actual wandb status
    config["save_masks"] = False
    config["use_gt_masks"] = enable_inference_during_training
     
    # Log model graph
    # example_images = torch.randn(1, 3, 256, 256).to(device)
    # if use_event:
    #     example_event_frames = torch.randn(1, 1, 256, 256).to(device)
    #     writer.add_graph(model, (example_images, example_event_frames))
    # else:
    #     writer.add_graph(model, example_images)

    use_consecutive_loss = config.get("use_consecutive_loss", False)    # === Update for Consecutive Loss with Warm-Up and EMA ===
    if use_consecutive_loss:
        print("Using Consecutive Loss with Warm-Up and EMA.")
    # Remove any global prev_pred variables; instead, reset the stored previous prediction at each epoch.
    # Define hyperparameters (these can be set via config)
    warmup_steps = config.get("consecutive_loss_warmup", 100)  # e.g., skip first 100 batches
    warmup_epochs = config.get("warmup_epochs", 2)  # e.g., skip first 2 epochs
    lambda_consecutive = config.get("lambda_consecutive", 0.1)  # scaling factor for consecutive loss
    beta = config.get("ema_decay", 0.9)  # decay factor for EMA
    reset_factor = config.get("consecutive_loss_reset_factor", 2.0)  # if current loss > reset_factor * EMA, reset

    start_time = time.time()
    if use_consecutive_loss:
        lpips_loss_fn = get_loss_function("LPIPS")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        # Reset the stored previous prediction at the start of each epoch.
        prev_pred_train = None
        ema_consecutive_loss = None
        
        # Training Loop
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(train_loader):    
                optimizer.zero_grad()  # Reset gradients before processing the batch
                
                # Initialize variables to ensure they're always in scope for logging
                inputs, targets, outputs = None, None, None
                                
                # === AUTOENCODER MODE: Event frames → Edge maps ===
                if is_autoencoder:
                    # Autoencoder learns to extract edge maps from event frames
                    # Used for preprocessing or as auxiliary task
                    event_frames, edge_maps, _ = batch  # Ignore orig_size during training
                    event_frames, edge_maps = event_frames.to(device), edge_maps.to(device)
                    outputs = model(event_frames)
                    loss = criterion(outputs, edge_maps)
                    
                    # Assign for TensorBoard logging
                    inputs = event_frames
                    targets = edge_maps
                    
                    # Post-process outputs for visualization
                    if not is_double_channel_autoencoder:
                        # Single-channel: Apply sigmoid to get [0, 1] range
                        outputs = torch.sigmoid(outputs)
                    elif is_double_channel_autoencoder:
                        # Two-channel: Get class predictions via argmax
                        outputs = torch.argmax(outputs, dim=1, keepdim=True)
                    else: 
                        raise ValueError("Invalid Autoencoder type.")
                        
                # === SEGMENTATION MODE: Multi-modal input support ===
                else:
                    # All datasets return original size as last element for proper inference
                    # Format: (data_elements..., orig_size)
                    *batch_data, _ = batch  # Unpack data, ignore orig_size during training
                    
                    # Detect input modality based on number of elements
                    # This allows flexible model architectures without explicit flags
                    
                    if len(batch_data) == 2:
                        # === Standard RGB Segmentation ===
                        # Dataset: SegmentationDataset
                        # Model: Single-encoder U-Net, DeepLab, etc.
                        images, masks = batch_data
                        images, masks = images.to(device), masks.to(device)
                        outputs = model(images)
                        inputs = images
                        
                    elif len(batch_data) == 3:
                        # === Dual-Input Segmentation ===
                        # Dataset: EventSegmentationDataset or AutoencoderSegmentationDataset
                        # Model: Dual-encoder architecture with fusion
                        # extra_data can be:
                        #   - Event frames (2-channel or 3-channel)
                        #   - Autoencoder-extracted features
                        images, masks, extra_data = batch_data
                        images, masks, extra_data = images.to(device), masks.to(device), extra_data.to(device)
                        outputs = model(images, extra_data)
                        
                        # Special handling for 3D models
                        if is_3D:
                            # Remove temporal dimension for visualization
                            # 3D input: (B, C, T, H, W) → (B, C, H, W) for logging
                            extra_data = extra_data.squeeze(1)
                            
                        # Concatenate for TensorBoard visualization
                        inputs = torch.cat((images, extra_data), dim=1)
                        
                    elif len(batch_data) == 4:
                        # === Triple-Input Segmentation ===
                        # Dataset: TripleInputDataset
                        # Model: Triple-encoder with multi-modal fusion
                        # Combines: RGB + events + autoencoder features
                        images, masks, events, autoencoder_features = batch_data
                        images, masks = images.to(device), masks.to(device)
                        events, autoencoder_features = events.to(device), autoencoder_features.to(device)
                        outputs = model(images, events, autoencoder_features)
                        
                        # Concatenate all modalities for visualization
                        inputs = torch.cat((images, events, autoencoder_features), dim=1)
                    else:
                        raise ValueError(f"Unexpected batch structure with {len(batch_data)} data elements")
                    
                    # Compute segmentation loss
                    loss = criterion(outputs, masks)
                    targets = masks

                # === CONSECUTIVE LOSS: Temporal Consistency ===
                # Encourages smooth predictions across consecutive frames in video/event sequences
                # Uses LPIPS perceptual loss to penalize sudden changes between frames
                # Implements EMA smoothing and spike detection to handle scene changes
                if use_consecutive_loss and epoch >= warmup_epochs:
                    # Skip warm-up epochs to let model stabilize first
                    
                    if batch_idx < warmup_steps:
                        # Warm-up period: Just store the current prediction
                        # Don't apply consecutive loss yet to avoid unstable gradients
                        prev_pred_train = outputs.detach()
                    else:
                        # Compute perceptual similarity between consecutive predictions
                        # LPIPS uses pretrained VGG features to measure perceptual distance
                        current_consecutive_loss = lpips_loss_fn(prev_pred_train, outputs)
                        
                        # Update Exponential Moving Average of consecutive loss
                        # EMA provides a smoothed baseline for detecting anomalies
                        if ema_consecutive_loss is None:
                            # Initialize EMA with first loss value
                            ema_consecutive_loss = current_consecutive_loss.detach()
                        else:
                            # Update: EMA = β * EMA + (1-β) * current_loss
                            ema_consecutive_loss = beta * ema_consecutive_loss + (1 - beta) * current_consecutive_loss.detach()

                        # Spike Detection: Check for scene changes or video cuts
                        # If loss suddenly spikes above threshold, likely a scene change
                        if current_consecutive_loss.item() > reset_factor * ema_consecutive_loss.item():
                            # Scene change detected! Reset to avoid penalizing valid discontinuities
                            print("Resetting EMA due to sudden spike in consecutive loss (scene change detected).")
                            prev_pred_train = outputs.detach()
                        else:
                            # Normal case: Add weighted consecutive loss to encourage smoothness
                            loss += lambda_consecutive * current_consecutive_loss
                            
                            # Update previous prediction using EMA for temporal smoothing
                            # This creates a smoothed reference prediction
                            prev_pred_train = beta * prev_pred_train + (1 - beta) * outputs.detach()
                # === End Consecutive Loss ===

                loss.backward()
                optimizer.step()   
                train_loss += loss.item()
                
                pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1)})
                pbar.update(1)
                
                # Log the first 10 batches for each epoch to TensorBoard and wandb
                # For TensorBoard: keep logging as before (epoch for epoch-level, batch for batch-level)
                if tensorboard_save_log:
                    if batch_idx < LOGGING_BATCH_LIMIT:  # Log only the first N batches
                        log_step = epoch * LOGGING_STEP_MULTIPLIER + batch_idx
                        log_to_tensorboard(
                            writer=writer,
                            inputs=inputs.cpu(),
                            targets=targets.cpu(),
                            outputs=outputs.cpu(),
                            global_step=log_step,  # use strictly increasing step to avoid overwriting
                            prefix=f"Train/Batch_{batch_idx + 1}",
                            grid_size=LOGGING_BATCH_LIMIT,
                            config=config
                        )
                # For wandb: log first N batches per epoch, step=epoch, unique keys
                if wandb_save_log:
                    if batch_idx < LOGGING_BATCH_LIMIT:  # Log only the first 10 batches with unique keys
                        # Use monotonic step for WandB API compliance
                        # Logic: 40 logs per epoch (10 batches + 10 val + 10 infer + summary).
                        # Batch 0-9 -> Step 0-9 (+ offset)
                        log_step = epoch * LOGGING_STEP_MULTIPLIER + batch_idx
                        log_to_wandb(
                            inputs=inputs.cpu(),
                            targets=targets.cpu(),
                            outputs=outputs.cpu(),
                            step=log_step,
                            epoch=epoch,
                            prefix=f"Train/Batch_{batch_idx + 1}",
                            grid_size=LOGGING_BATCH_LIMIT,
                            config=config
                        )
                                                        
        # Calculate Average Loss
        avg_train_loss = train_loss / len(train_loader)
        # avg_val_loss = val_loss / len(val_loader)
        avg_val_loss = None
        result_val = infer_model(step=epoch,dataloader=val_loader,device=device,model=model,config=config,writer=writer, inference_prefix="Validation")
        if is_autoencoder:
            avg_val_loss = result_val
        else:
            avg_val_loss = result_val[-1]
            val_mean_iou = result_val[-2]
        # train_loss_history.append(avg_train_loss)
        # val_loss_history.append(avg_val_loss)

        # Log to TensorBoard
        summary_step = epoch * LOGGING_STEP_MULTIPLIER + LOGGING_SUMMARY_OFFSET # ensure summary is always after batches
        if tensorboard_save_log:
            writer.add_scalar("Losses/Train Loss", avg_train_loss, epoch)
            writer.add_scalar("Losses/Validation Loss", avg_val_loss, epoch)
            if not is_autoencoder:
                writer.add_scalar("Validation/Mean_IoU", val_mean_iou, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
            writer.flush() # Ensure logs are written to disk
        # writer.add_scalars("Loss/Epoch", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch)
        # for i, param_group in enumerate(optimizer.param_groups):
        #     writer.add_scalar(f"Learning_Rate/Group_{i}", param_group['lr'], epoch)
        # writer.add_scalars("Losses", {"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss}, epoch)
        
        # Log to Weights & Biases
        if wandb_save_log:
            wandb_log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            if not is_autoencoder:
                wandb_log_dict["val_mean_iou"] = val_mean_iou
            
            # Log with monotonic step, but because we defined step_metric="epoch", 
            # these metrics will be aligned with the epoch in the UI.
            wandb.log(wandb_log_dict, step=summary_step, commit=True)  # commit=True forces immediate upload


        # Print Epoch Summary
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        remaining_time = (epochs - epoch - 1) * epoch_time
        tqdm.write(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Elapsed Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s, ETA: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s"
        )

        # Scheduler Step
        # scheduler.step(avg_val_loss)

        # Save Checkpoint if Validation Loss Improves
        checpoint_save_count = 0
        if save_interim_checkpoints and ((epoch + 1) % save_interim_checkpoint_interval == 0):        
            os.makedirs(save_folder, exist_ok=True)
            interim_save_path = f"{save_folder}/checkpoint_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, interim_save_path, best_val_loss=best_val_loss, dataset=dataset)
            tqdm.write(f"Interim Checkpoint saved at {interim_save_path} with Val Loss: {avg_val_loss:.4f}")
            checpoint_save_count += 1
            
        if save_checkpoint_with_mean_iou and (not is_autoencoder):
            if val_mean_iou > best_val_mean_iou:
                best_val_mean_iou = val_mean_iou
                save_checkpoint(model, optimizer, epoch, save_path, best_val_loss=best_val_mean_iou, dataset=dataset)
                tqdm.write(f"Checkpoint saved at {save_path} with Val IoU: {val_mean_iou:.4f}")
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, save_path, best_val_loss=best_val_loss, dataset=dataset)
                tqdm.write(f"Checkpoint saved at {save_path} with Val Loss: {avg_val_loss:.4f}")

        # Save Loss History
        if history_path:
            save_loss_history(avg_train_loss, avg_val_loss, history_path)
            tqdm.write(f"Loss history saved to {history_path}")

        if enable_inference_during_training and (epoch + 1) % inference_interval == 0:
            try:
                # Run Inference
                print(f"Inference at Epoch {epoch}")
                results = infer_model(
                    step=epoch,
                    dataloader=test_loader,
                    device=device,
                    model=model,
                    config=config,
                    writer=writer
                )
                avg_infer_loss = None
                if not is_autoencoder: 
                    # Log Inference Metrics
                    average_weighted_mean_iou, average_classwise_iou, general_mean_iou, avg_infer_loss = results
                    print(f"Mean IoU: {general_mean_iou:.4f} at Epoch {epoch}")
                    if tensorboard_save_log:
                        writer.add_scalar("Inference/Mean_IoU", general_mean_iou, global_step=epoch)
                        writer.flush()
                    if wandb_save_log:
                        wandb.log({
                            "epoch": epoch,
                            "inference_mean_iou": general_mean_iou,
                            "inference_weighted_mean_iou": average_weighted_mean_iou,
                            "inference_loss": avg_infer_loss
                        }, step=epoch * LOGGING_STEP_MULTIPLIER + LOGGING_INFER_OFFSET + LOGGING_BATCH_LIMIT, commit=True) # use a step strictly after inference images (e.g. 21 + 10 = 31)
                    
                    inference_metrics = {
                        "epoch": epoch,
                        "average_weighted_mean_iou": average_weighted_mean_iou,
                        "general_mean_iou": general_mean_iou,
                        "average_classwise_iou": average_classwise_iou,
                    }
                    metrics_path = f"{logdir}/inference_metrics.json"
                    # Load existing metrics if the file exists
                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as f:
                            existing_metrics = json.load(f)
                    else:
                        existing_metrics = []

                    # Append new metrics
                    existing_metrics.append(inference_metrics)
                    # Save back to the file
                    with open(metrics_path, "w") as f:
                        json.dump(existing_metrics, f, indent=4)

                    print(f"Metrics appended to {metrics_path}")
                else:
                    avg_infer_loss = results
                
                # if avg_infer_loss < best_infer_loss:
                #     best_infer_loss = avg_infer_loss
                #     save_checkpoint(model, optimizer, epoch, save_path, best_val_loss=best_infer_loss, dataset=dataset)
                #     tqdm.write(f"Checkpoint saved at {save_path} with Val Loss: {avg_infer_loss:.4f}")

            except Exception as e:
                print(f"Error during inference: {e}")
    if tensorboard_save_log:
        writer.close()
    if wandb_save_log:
        wandb.finish()
