"""
Dataset factory for multi-modal semantic segmentation.

This module provides the factory function to instantiate the correct dataset class
based on the model type. It maps model architecture names to their corresponding
dataset classes (RGB-only, RGB+events, RGB+autoencoder, etc.).

The factory pattern allows centralized dataset selection and easy extension for
new model types.
"""

from typing import Dict, Any
from dataset.classes import *

def get_dataset(config: Dict[str, Any]):
    """
    Factory function to initialize the dataset class based on model type.
    
    This function implements a registry pattern that maps model architecture names
    to their corresponding dataset classes. It supports various input modalities:
    - RGB-only segmentation
    - RGB + 1-channel event data
    - RGB + 3-channel RGB event data
    - RGB + autoencoder features
    - Triple input (RGB + events + autoencoder)
    - Event-only edge detection
    - 3D temporal models

    Args:
        config: Configuration dictionary containing:
            - model_type (str): Model architecture name (e.g., "smp_unet", "deeplab")
            - image_dir (str): Path to RGB images directory (or None if not used)
            - mask_dir (str): Path to segmentation masks directory
            - event_dir (str, optional): Path to event frames directory
            - autoencoder_dir (str, optional): Path to autoencoder features directory
            - time_steps (int, optional): Number of temporal time steps for 3D models
            - edge_method (str, optional): Edge detection method ('canny', 'dog', 'log')
            - num_of_out_classes (int): Number of segmentation classes
            - num_of_mask_classes (int): Number of classes in masks
            - is_event_scapes (bool): Whether using EventScapes dataset (affects background color)

    Returns:
        Initialized PyTorch Dataset object of the appropriate class
        
    Raises:
        ValueError: If model_type is not in the registry or dataset initialization fails
        
    Example:
        >>> config = {
        ...     'model_type': 'smp_dual_encoder_unet',
        ...     'image_dir': 'data/images',
        ...     'mask_dir': 'data/masks',
        ...     'event_dir': 'data/events',
        ...     'num_of_out_classes': 11
        ... }
        >>> dataset = get_dataset(config)
        >>> type(dataset).__name__
        'EventSegmentationDataset'
        
    Note:
        To add support for a new model type:
        1. Add the model_type key to dataset_mapping
        2. Map it to the appropriate dataset class
        3. Ensure the dataset class exists in classes.py
        
        The registry design allows a single entry point for all dataset creation,
        making it easy to maintain and extend.
    """
    # Mapping model types to dataset classes
    dataset_mapping = {
        "smp_unet": SegmentationDataset,
        "custom_smp_unet": SegmentationDataset,
        "smp_dual_encoder_unet": EventSegmentationDataset,
        "smp_generic": SegmentationDataset,
        "smp_event_generic": EventSegmentationDataset,
        "smp_dual_encoder_unet_3_channel": Event3ChannelSegmentationDataset,
        "smp_segformer": SegmentationDataset,
        "basic": SegmentationDataset,
        "deeplab": SegmentationDataset,
        "swinformer": SegmentationDataset,
        "deeplab_event": EventSegmentationDataset,
        "deeplab_event_3channel": Event3ChannelSegmentationDataset,
        "deeplab_triple_input": TripleInputDataset,
        "deeplab_event_v2": EventSegmentationDataset,
        "deeplab_4_channel": EventSegmentationDataset,
        "deeplab_mid_fusion_deep": EventSegmentationDataset,
        "deeplab_mid_fusion_shallow": EventSegmentationDataset,
        "attentionv1": EventSegmentationDataset,
        "attentionv2": EventSegmentationDataset,
        "attentionv3": EventSegmentationDataset,
        "attentionv4": EventSegmentationDataset,
        "m_attentionv1": EventSegmentationDataset,
        "dual_encoder": EventSegmentationDataset,
        "dual_encoder_attention": EventSegmentationDataset,
        "dual_deeplab": EventSegmentationDataset,
        "dual_deeplab_3D": EventSegmentation3DDataset,
        "deeplab_transformer_attention": EventSegmentationDataset,
        "dual_encoder_3D": EventSegmentation3DDataset,
        "attention_auto_encoder_v1": AutoencoderSegmentationDataset,
        "attention_auto_encoder_v2": AutoencoderSegmentationDataset,
        "attention_auto_encoder_v3": AutoencoderSegmentationDataset,
        "attention_auto_encoder_v4": AutoencoderSegmentationDataset,
        "auto_encoder_dual_encoder": AutoencoderSegmentationDataset,
        "deeplab_autoencoder_seg": AutoencoderSegmentationDataset,
        "deeplab_autoencoder_seg_morp": AutoencoderSegmentationDatasetMorphology,
        "deeplab_autoencoder_seg_v2": AutoencoderSegmentationDataset,
        "deeplab_autoencoder": EdgeEventDataset,
        "denoise_autoencoder": EdgeEventDataset,
        "denoise_autoencoder_double_channel": EdgeEventDatasetDoubleChannel,
        "concatenated_autoencoder": EdgeEventDataset,
        "denoise_autoencoder_3D": EdgeEvent3DDataset,
        "denoise_autoencoder_3D_double_channel": EdgeEvent3DDatasetDoubleChannel,
    }
    model_type = config.get("model_type")
    if model_type not in dataset_mapping:
        raise ValueError(f"Invalid dataset type: {model_type}. Available types: {list(dataset_mapping.keys())}")

    # Get the corresponding dataset class
    dataset_class = dataset_mapping[model_type]

    # Initialize the dataset
    try:
        return dataset_class(config)
    except TypeError as e:
        raise ValueError(f"Error initializing dataset for model type '{model_type}': {e}")
