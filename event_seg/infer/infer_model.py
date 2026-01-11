"""
Inference module for semantic segmentation models.

This module provides inference functionality for various segmentation architectures
with support for multi-modal inputs, IoU metric computation, and visualization.

Key Features:
    - Multi-modal inference (RGB, events, autoencoder features, triple-input)
    - Automatic padding removal using original image sizes
    - IoU metric calculation with class-wise and weighted averaging
    - Mask saving in colorized and grayscale formats
    - TensorBoard logging with activation visualization
    - Support for autoencoder edge extraction models
    - Loss computation during inference for validation
    - Batch processing with progress tracking

Supported Model Types:
    - Standard segmentation: RGB images only
    - Event-based: RGB + event frames
    - Autoencoder: Event frames → edge maps
    - Triple-input: RGB + events + autoencoder features
    - 3D models: Temporal sequences

Inference Modes:
    1. Validation during training (compute loss + IoU, no mask saving)
    2. Full inference with mask saving (colorized + grayscale outputs)
    3. TensorBoard logging mode (visualizations + activations)
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils.utilities import (
    get_tensorboard_writer, 
    colorize_mask_save, 
    save_mask_as_image, 
    calculate_weighted_iou, 
    log_to_tensorboard, 
    log_to_wandb,
    LOGGING_BATCH_LIMIT,
    LOGGING_STEP_MULTIPLIER,
    LOGGING_VAL_OFFSET,
    LOGGING_INFER_OFFSET
)
from torchsummary import summary
from train.loss import get_loss_function

def save_segmentation_masks(
    predicted_masks: np.ndarray,
    image_files: List[str],
    start_idx: int,
    colorized_output_dir: str,
    grayscale_output_dir: str,
    color_file_prefix: str,
    gray_file_prefix: str
) -> None:
    """
    Save predicted segmentation masks in colorized and grayscale formats.
    
    Args:
        predicted_masks: Numpy array of predicted masks with shape (B, H, W)
        image_files: List of original image filenames from dataset
        start_idx: Starting index for mapping batch to filenames
        colorized_output_dir: Directory to save colorized masks
        grayscale_output_dir: Directory to save grayscale masks
        color_file_prefix: Prefix for colorized mask filenames
        gray_file_prefix: Prefix for grayscale mask filenames
        
    Note:
        - Colorized masks use color palette for visualization
        - Grayscale masks contain raw class indices
        - Skips files if index exceeds image_files length
    """
    for i, predicted_mask in enumerate(predicted_masks):
        file_index = start_idx + i  # Ensure correct mapping of batch images
        if file_index >= len(image_files):
            print(f"Warning: Index {file_index} out of range for image files.")
            continue
        
        file_name = image_files[file_index]  # Filename from dataset
        color_path = os.path.join(colorized_output_dir, f"{color_file_prefix}{file_name}")
        gray_path = os.path.join(grayscale_output_dir, f"{gray_file_prefix}{file_name}")
        
        colorize_mask_save(predicted_mask, color_path, print_message=False)
        save_mask_as_image(predicted_mask, gray_path, print_message=False)


def compute_iou_metrics(
    predicted_masks: np.ndarray,
    ground_truth_masks: torch.Tensor,
    num_of_out_classes: int,
    total_weighted_mean_iou: float,
    classwise_iou_sums: Optional[List[float]],
    classwise_counts: Optional[List[int]],
    total_images: int
) -> Tuple[float, float, List[float], float, List[float], List[int]]:
    """
    Compute IoU metrics with class-wise and weighted averaging.
    
    Calculates three types of IoU metrics:
    1. Class-wise IoU: Average IoU for each semantic class
    2. Weighted Mean IoU: IoU weighted by class frequency in ground truth
    3. General Mean IoU: Simple average across all classes
    
    Args:
        predicted_masks: Predicted class indices with shape (B, H, W)
        ground_truth_masks: Ground truth masks as torch.Tensor
        num_of_out_classes: Number of semantic classes
        total_weighted_mean_iou: Accumulated weighted IoU from previous batches
        classwise_iou_sums: List of accumulated IoU per class (initialized if None)
        classwise_counts: List of sample counts per class (initialized if None)
        total_images: Total number of images processed so far
        
    Returns:
        Tuple containing:
        - general_mean_iou: Average IoU across all classes
        - avg_weighted_mean_iou: Weighted average IoU
        - avg_classwise_iou: List of average IoU per class
        - total_weighted_mean_iou: Updated accumulated weighted IoU
        - classwise_iou_sums: Updated per-class IoU sums
        - classwise_counts: Updated per-class sample counts
        
    Note:
        - Handles NaN values by excluding them from averages
        - Class-wise metrics help identify which classes are challenging
        - Weighted IoU accounts for class imbalance in the dataset
    """
    if classwise_iou_sums is None:
        classwise_iou_sums = [0.0] * num_of_out_classes
    if classwise_counts is None:
        classwise_counts = [0] * num_of_out_classes
        
    for i, predicted_mask in enumerate(predicted_masks):
        ground_truth_mask = ground_truth_masks[i].cpu().numpy()
        iou_per_class, weighted_mean_iou = calculate_weighted_iou(predicted_mask, ground_truth_mask, num_of_out_classes)
        
        total_weighted_mean_iou += weighted_mean_iou

        for cls, iou in enumerate(iou_per_class):
            if not torch.isnan(torch.tensor(iou)):
                classwise_iou_sums[cls] += iou
                classwise_counts[cls] += 1

    avg_classwise_iou = [
        classwise_iou_sums[cls] / classwise_counts[cls] if classwise_counts[cls] > 0 else 0.0
        for cls in range(num_of_out_classes)
    ]
    avg_weighted_mean_iou = total_weighted_mean_iou / total_images if total_images > 0 else 0.0
    
    # Compute general mean IoU
    valid_classwise_iou = [iou for iou in avg_classwise_iou if iou >= 0]
    general_mean_iou = sum(valid_classwise_iou) / len(valid_classwise_iou) if valid_classwise_iou else 0.0  
    # return general_mean_iou, avg_weighted_mean_iou, avg_classwise_iou

    return general_mean_iou, avg_weighted_mean_iou, avg_classwise_iou, total_weighted_mean_iou, classwise_iou_sums, classwise_counts

def get_layer_names(model: nn.Module) -> List[str]:
    """
    Extract all layer names from a PyTorch model for activation visualization.

    Args:
        model: PyTorch model to inspect

    Returns:
        List of layer names as strings (e.g., 'encoder.layer1', 'decoder.up1')
        
    Example:
        >>> layer_names = get_layer_names(model)
        >>> print(layer_names[:5])
        ['encoder', 'encoder.conv1', 'encoder.bn1', 'encoder.relu', 'encoder.maxpool']
    """
    return [name for name, _ in model.named_modules()]



def register_hooks(model: nn.Module, layer_names: List[str]) -> Dict[str, List[torch.Tensor]]:
    """
    Register forward hooks on specified model layers for activation capture.
    
    This function attaches hooks to layers to capture their activations during
    forward pass. Useful for visualization and debugging model behavior.
    
    Args:
        model: PyTorch model to attach hooks to
        layer_names: List of layer names to monitor (from get_layer_names())

    Returns:
        Dictionary mapping layer names to lists of activation tensors
        Format: {layer_name: [activation_tensor1, activation_tensor2, ...]}
        
    Note:
        - Handles dict outputs from models like DeepLabV3 (extracts 'out' key)
        - Activations are detached and moved to CPU to save GPU memory
        - Each forward pass appends to the activation list
        
    Example:
        >>> layer_names = ['encoder.layer1', 'decoder.up1']
        >>> activations = register_hooks(model, layer_names)
        >>> output = model(input)  # Forward pass
        >>> print(activations['encoder.layer1'][0].shape)  # First activation
        torch.Size([1, 64, 128, 128])
    """
    activations = {}  # To store activations

    def hook_fn(name):
        def hook(module, input, output):
            # Handle the case where output is a dictionary (e.g., from DeepLabV3)
            if isinstance(output, dict):
                # Most segmentation models use 'out' key for the main output
                if 'out' in output:
                    activations[name].append(output['out'].detach().cpu())
                else:
                    # Take the first value if 'out' key is not present
                    first_key = next(iter(output))
                    activations[name].append(output[first_key].detach().cpu())
            else:
                activations[name].append(output.detach().cpu())
        return hook

    for name, layer in model.named_modules():
        # print(name)  # Print all layer names
        if name in layer_names:  # Match the layer names to hook
            activations[name] = []  # Initialize a list to store activations
            layer.register_forward_hook(hook_fn(name))  # Pass the name to the hook function

    return activations

def infer_model(
    dataloader: DataLoader,
    device: str = "cuda",
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    step: int = 0,
    writer: Optional[SummaryWriter] = None,
    inference_prefix: str = "Inference",
) -> Optional[Tuple[float, List[float], float, float]]:
    """
    Run inference on segmentation models with multi-modal support and metrics.
    
    This function handles the complete inference pipeline including:
    - Multi-modal input processing (RGB, events, autoencoder features)
    - Automatic padding removal using original image sizes
    - IoU metric calculation for validation
    - Mask saving in colorized and grayscale formats
    - TensorBoard logging with visualizations
    - Activation visualization for model debugging
    
    The function automatically detects model type from batch structure:
    - 2 elements: Standard RGB segmentation (images, masks)
    - 3 elements: Dual-input (images, masks, events/autoencoder)
    - 4 elements: Triple-input (images, masks, events, autoencoder)
    
    Args:
        dataloader: DataLoader containing test/validation data
        device: Device for inference ('cuda' or 'cpu')
        model: Pretrained PyTorch model (must be in eval mode)
        config: Configuration dictionary containing:
            - model_type: Architecture name for logging
            - output_folder: Base directory for saving masks
            - use_rgb: Whether model uses RGB images
            - use_event: Whether model uses event frames
            - use_gt_masks: Whether to compute IoU metrics
            - is_3D: Using 3D convolutions
            - time_steps: Number of temporal frames (for 3D)
            - is_autoencoder: Running autoencoder model
            - is_double_channel_autoencoder: 2-channel autoencoder output
            - save_masks: Save predictions to disk
            - num_of_mask_classes: Number of semantic classes
            - tensorboard_save_log: Enable TensorBoard logging
            - tensorboard_save_activations: Save layer activations
            - loss: Loss function name
            - torch_summary: Print model summary
            - inference.result_dir: Directory for inference results
            - event_folder: Event frame folder name (optional)
        step: Current training epoch (for TensorBoard logging)
        writer: TensorBoard SummaryWriter (created if None and logging enabled)
        inference_prefix: Prefix for TensorBoard logs (e.g., "Validation", "Inference")
        
    Returns:
        For segmentation models with use_gt_masks=True:
            Tuple of (average_weighted_mean_iou, average_classwise_iou, 
                     general_mean_iou, avg_inference_loss)
        For autoencoder models:
            avg_inference_loss (float)
        If use_gt_masks=False:
            None
            
    Raises:
        ValueError: If batch structure doesn't match expected format
        
    Example:
        >>> # Validation during training
        >>> config = {'use_gt_masks': True, 'save_masks': False, ...}
        >>> results = infer_model(
        ...     dataloader=val_loader,
        ...     device='cuda',
        ...     model=model,
        ...     config=config,
        ...     step=epoch,
        ...     writer=tb_writer,
        ...     inference_prefix='Validation'
        ... )
        >>> mean_iou = results[2]
        
        >>> # Full inference with mask saving
        >>> config = {'use_gt_masks': True, 'save_masks': True, ...}
        >>> results = infer_model(
        ...     dataloader=test_loader,
        ...     device='cuda',
        ...     model=model,
        ...     config=config
        ... )
        
    Note:
        - Automatically crops predictions to original image size (removes padding)
        - Logs first 10 batches to TensorBoard if enabled
        - Creates colorized and grayscale mask directories if save_masks=True
        - Computes loss even during inference for validation monitoring
        - For autoencoders: outputs are single-channel edge maps
        - For segmentation: outputs are class predictions (argmax of logits)
    """
    # model = model
    model_type = config.get("model_type")
    output_folder = config.get("output_folder")
    # writer = None
    use_rgb = config.get("use_rgb")
    use_event = config.get("use_event")
    use_gt_masks = config.get("use_gt_masks", False)
    
    is_3D = config.get("is_3D")
    
    time_steps = config.get("time_steps")
    
    is_autoencoder = config.get("is_autoencoder", False)
    save_masks = config.get("save_masks")
    num_of_mask_classes = config.get("num_of_mask_classes")
    
    tensorboard_save_log = config.get("tensorboard_save_log", False)
    tensorboard_save_activations = config.get("tensorboard_save_activations", False)
    wandb_save_log = config.get("wandb_save_log", False)
    
    if tensorboard_save_log:
        log_dir=f"{config['inference'].get('result_dir')}/{model_type}"
        if writer is None:
            writer, _ = get_tensorboard_writer(base_log_dir=log_dir)

    model.eval()

    loss = config.get("loss")
    criterion = get_loss_function(loss, config)
    # use_consecutive_loss = config.get("use_consecutive_loss", False)
    is_double_channel_autoencoder = config.get("is_double_channel_autoencoder", False)
    
    # print(summary(model))
    torch_summary = config.get("torch_summary", False)
    if torch_summary:
        print(summary(model=model, input_size=dataloader.dataset[0][0].shape))
    if save_masks:
        # Name output directories and create them
        colorized_output_dir = os.path.join(output_folder, "colorized", f"{model_type}")
        grayscale_output_dir = os.path.join(output_folder, "grayscale", f"{model_type}")
        denoise_output_dir = os.path.join(output_folder, "smoothed", f"{model_type}")
        
        if use_event:
            event_folder_name = config.get("event_folder")
            colorized_output_dir += f"_{event_folder_name}"
            grayscale_output_dir += f"_{event_folder_name}"
        
        if is_autoencoder:
            os.makedirs(denoise_output_dir, exist_ok=True)
        else:
            os.makedirs(colorized_output_dir, exist_ok=True)
            os.makedirs(grayscale_output_dir, exist_ok=True)  

    # Initialize IoU metrics accumulation variables
    total_weighted_mean_iou = 0.0
    classwise_iou_sums = None
    classwise_counts = None
    total_images = 0
    infer_loss = 0.0
    # prev_pred_infer = None
    # Inference loop
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Processing Batches", unit="batch") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if is_autoencoder:
                    # if not save_masks and tensorboard_save_log and batch_idx >= 10:
                    #     print("Stopping inference: No mask saving, TensorBoard logging active, and batch index exceeds 10.")
                    #     break
                    if not save_masks and not tensorboard_save_log:
                        print("Stopping inference: Neither mask saving nor TensorBoard logging is enabled.")
                        break
                    
                inputs, targets, outputs = None, None, None
                if tensorboard_save_activations:
                    layer_names = get_layer_names(model)
                    activations = register_hooks(model, layer_names)

                if is_autoencoder:  # === AUTOENCODER MODE: Event frames → Edge maps ===
                    event_frames, edge_maps, orig_sizes = batch
                    event_frames, edge_maps = event_frames.to(device), edge_maps.to(device)
                    
                    # === Extract Original Dimensions for Padding Removal ===
                    # DataLoader may collate orig_sizes in two formats:
                    # Format 1: [tensor([h1,h2,...]), tensor([w1,w2,...])] - separate height/width tensors
                    # Format 2: tensor([[h1,w1], [h2,w2], ...]) - paired dimensions
                    # We need the first sample's (h, w) to crop ALL samples to the same size
                    if isinstance(orig_sizes, list) and len(orig_sizes) == 2:
                        # Format 1: Separate height and width tensors
                        orig_h, orig_w = orig_sizes[0][0].item(), orig_sizes[1][0].item()
                    else:
                        # Format 2: Paired dimensions tensor
                        orig_h, orig_w = orig_sizes[0][0].item(), orig_sizes[0][1].item()
                    
                    outputs = model(event_frames)
                    
                    # === Crop to Original Size (Remove Padding) ===
                    # All images were padded to multiples of 32 for model compatibility
                    # Now remove padding to get back to original dimensions for accurate metrics
                    event_frames_cropped = event_frames[..., :orig_h, :orig_w]
                    edge_maps_cropped = edge_maps[..., :orig_h, :orig_w]
                    outputs_cropped = outputs[..., :orig_h, :orig_w]
                    
                    loss = criterion(outputs_cropped, edge_maps_cropped)  # Loss on original size only
                    inputs = event_frames_cropped
                    targets = edge_maps_cropped
                    inputs_for_log = inputs  # For TensorBoard visualization
                    outputs_for_log = outputs_cropped
                    
                    # === Post-process Autoencoder Outputs ===
                    if not is_double_channel_autoencoder:
                        # Single-channel: Sigmoid to get [0, 1] probability map
                        outputs = torch.sigmoid(outputs_cropped).cpu()
                    elif is_double_channel_autoencoder:
                        # Two-channel: Argmax to get binary edge/no-edge classification
                        outputs = torch.argmax(outputs_cropped, dim=1, keepdim=True).cpu()
                    else: 
                        raise ValueError("Invalid Autoencoder type.")
                else:
                    # === SEGMENTATION MODE: Multi-modal Input Support ===
                    # All datasets return original size as last element for proper cropping
                    *batch_data, orig_sizes = batch
                    
                    # === Extract Original Dimensions (Same Logic as Autoencoder) ===
                    if isinstance(orig_sizes, list) and len(orig_sizes) == 2:
                        # DataLoader collated as list of tensors
                        orig_h, orig_w = orig_sizes[0][0].item(), orig_sizes[1][0].item()
                    else:
                        # Fallback: assume tensor of shape [batch_size, 2]
                        orig_h, orig_w = orig_sizes[0][0].item(), orig_sizes[0][1].item()
                    
                    # === Detect Input Modality and Run Inference ===
                    # Unpack based on number of elements (excluding orig_size which was already extracted)
                    # This allows automatic model selection without explicit configuration flags
                    
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
                        # Model: Dual-encoder with fusion
                        # extra_data can be event frames or autoencoder features
                        images, masks, extra_data = batch_data
                        images, masks, extra_data = images.to(device), masks.to(device), extra_data.to(device)
                        outputs = model(images, extra_data)
                        
                        # Special handling for 3D models
                        if is_3D:
                            # Remove temporal dimension: (B, C, T, H, W) → (B, C, H, W)
                            extra_data = extra_data.squeeze(1)
                            
                        # Concatenate modalities for TensorBoard visualization
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
                        
                        # Concatenate all three modalities for visualization
                        inputs = torch.cat((images, events, autoencoder_features), dim=1)
                    else:
                        raise ValueError(f"Unexpected batch structure with {len(batch_data)} data elements")
                    
                    # === Crop to Original Size (Remove Padding) ===
                    # Critical: Must crop BEFORE computing loss and metrics to ensure accuracy
                    # Padding affects IoU calculation and loss values
                    masks_cropped = masks[..., :orig_h, :orig_w]
                    outputs_cropped = outputs[..., :orig_h, :orig_w]
                    inputs_cropped = inputs[..., :orig_h, :orig_w]
                    
                    # Compute loss on cropped regions for accurate validation metrics
                    loss = criterion(outputs_cropped, masks_cropped)
                    
                    # Prepare cropped versions for TensorBoard logging
                    targets = masks_cropped
                    outputs_for_log = outputs_cropped
                    inputs_for_log = inputs_cropped
                    
                    # === Get Final Predictions ===
                    # Convert logits to class predictions via argmax
                    pred = torch.argmax(outputs, dim=1)  # (B, C, H, W) → (B, H, W)
                    # Crop predictions to original size
                    pred = pred[..., :orig_h, :orig_w]
                    predicted_masks = pred.cpu().numpy()  # Move to CPU for IoU calculation

                # # Apply consecutive loss if enabled
                # if use_consecutive_loss:
                #     if prev_pred_infer is not None:
                #         consecutive_loss = criterion(prev_pred_infer, outputs)  # Compute LPIPS between consecutive outputs
                #         loss += consecutive_loss  # Add to total loss
                        
                #         # Update previous prediction for next iteration
                #         prev_pred_infer = outputs.detach()

                infer_loss += loss.item()

                if batch_idx < LOGGING_BATCH_LIMIT:  # Log only the first N batches
                    # Calculate monotonic step for this batch
                    # Training batches use indices 0-9 (step * 40 + [0-9])
                    # Validation batches use indices 10-19 (step * 40 + 10 + [0-9])
                    # Inference batches use indices 20-29 (step * 40 + 20 + [0-9])
                    
                    log_step_offset = LOGGING_VAL_OFFSET # Default to validation offset
                    if "Inference" in inference_prefix:
                        log_step_offset = LOGGING_INFER_OFFSET
                    
                    log_step = step * LOGGING_STEP_MULTIPLIER + log_step_offset + batch_idx

                    if tensorboard_save_log:
                        # All models now use cropped versions for logging
                        log_to_tensorboard(writer=writer,inputs=inputs_for_log.cpu(),targets=targets.cpu(),outputs=outputs_for_log.cpu(), global_step=log_step, prefix=f"{inference_prefix}/Batch_{batch_idx + 1}",grid_size=LOGGING_BATCH_LIMIT, config=config)   
                
                    if wandb_save_log:
                        # All models now use cropped versions for logging
                        log_to_wandb(inputs=inputs_for_log.cpu(),targets=targets.cpu(),outputs=outputs_for_log.cpu(), step=log_step, epoch=step, prefix=f"{inference_prefix}/Batch_{batch_idx + 1}",grid_size=LOGGING_BATCH_LIMIT, config=config)
                    
                if tensorboard_save_activations and tensorboard_save_log:
                    # Visualize activations
                    for name, acts in activations.items():
                        for idx, act in enumerate(acts):  # Acts is a list of tensors for this layer
                            act = act[0]  # Now shape is (C, H, W)
                            act_min, act_max = act.min(), act.max()
                            if act_max > act_min:
                                act = (act - act_min) / (act_max - act_min)
                            else:
                                act = torch.zeros_like(act)
                            # Create a grid of feature maps
                            grid = vutils.make_grid(act.unsqueeze(1), nrow=8, normalize=True, scale_each=True)  # Add a channel dim
                            writer.add_image(f"{inference_prefix}/Activations/{name}_Sample_{idx}_Grid", grid, global_step=total_images)
                
                if not is_autoencoder:
                    # Predictions for segmentation models
                    if save_masks:
                        save_segmentation_masks(predicted_masks, dataloader.dataset.image_files, total_images, colorized_output_dir, grayscale_output_dir, "colorized_", "grayscale_" )
                    total_images += len(predicted_masks)
                    # Calculate IoU if ground truth exists
                    if use_gt_masks:
                        # Crop ground truth masks to original size for accurate IoU calculation
                        masks_cropped_cpu = masks_cropped.cpu()
                        # general_mean_iou, average_weighted_mean_iou, average_classwise_iou = compute_iou_metrics(predicted_masks, masks_cropped_cpu, num_of_mask_classes)
                        general_mean_iou, average_weighted_mean_iou, average_classwise_iou, total_weighted_mean_iou, classwise_iou_sums, classwise_counts = compute_iou_metrics(predicted_masks, masks_cropped_cpu, num_of_mask_classes, total_weighted_mean_iou, classwise_iou_sums, classwise_counts, total_images)
                # for autoencoder
                if save_masks and is_autoencoder:
                    # Handle batch size appropriately
                    batch_size = outputs.size(0)
                    for i in range(batch_size):
                        try:
                            # Get correct file index, accounting for previous batches
                            file_index = batch_idx * dataloader.batch_size + i
                            
                            # Get the filename safely, handling potential index errors
                            if file_index < len(dataloader.dataset.event_files):
                                file_name = dataloader.dataset.event_files[file_index*time_steps]
                            else:
                                # If for some reason we've run out of filenames, generate a placeholder
                                file_name = f"output_{file_index}.png"
                            
                            # Extract and process the current output in the batch
                            output = outputs[i]
                            output_image = (output.squeeze() * 255).cpu().numpy().astype(np.uint8)
                            
                            # Create save path and save the image
                            save_path = os.path.join(denoise_output_dir, file_name)
                            save_mask_as_image(output_image, save_path, print_message=False)
                        except Exception as e:
                            print(f"Error saving output {i} from batch {batch_idx}: {e}")
                    
                    # Update total images processed (for metrics)
                    total_images += batch_size
                        
                pbar.set_postfix({f"{inference_prefix} Loss": infer_loss / (pbar.n + 1)})
                pbar.update(1)


    # Calculate Average Loss
    avg_infer_loss = infer_loss / len(dataloader)

    if tensorboard_save_log:
        writer.add_scalar(f"Losses/{inference_prefix} Loss", avg_infer_loss, step)
        writer.close()
    # Return all metrics for segmentation models
    if use_gt_masks:
        if not is_autoencoder: 
            return average_weighted_mean_iou, average_classwise_iou, general_mean_iou, avg_infer_loss
        else:
            return avg_infer_loss
    return None  # No metrics when use_gt_masks is False
