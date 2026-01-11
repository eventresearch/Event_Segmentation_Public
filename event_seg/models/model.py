"""
Model factory for multi-modal semantic segmentation architectures.

This module provides the factory function to instantiate the correct model class
based on the model type. It supports various architectures:
- SMP models (UNet, Segformer) with pretrained encoders
- DeepLab variants with different fusion strategies
- Custom dual-encoder architectures
- Attention-based models
- Autoencoder-based denoising models
- 3D spatiotemporal models

The factory pattern centralizes model selection and simplifies adding new architectures.
"""

from typing import Dict, Any
import torch.nn as nn
from .base_model import BaseUNet
from .attention_models import UNetWithEventAttentionv1, UNetWithEventAttentionv2, UNetWithEventAttentionv3, UNetWithEventAttentionv4
from .dual_encoder import BaseDualEncoderUNet, DualEncoderWithAttentionUNet, DualEncoder3DUNet
from .attention_auto_encoder import Attention_Autoencoder_Unet_v1, Attention_Autoencoder_Unet_v2, Attention_Autoencoder_Unet_v3, Attention_Autoencoder_Unet_v4
from .denoise_autoencoder import AutoencoderUNet, AutoencoderUNet3D, ConcatenatedAutoencoder
from .deeplab import DeepLabWrapper, DeepLabEventIntegratedWrapper, DualDeepLabFusionModel, GatedCrossModalTransformerFusionModel, DeepLabEventIntegratedWrapperv2, DeepLabEventIntegratedWrapper3D, EarlyFusionDeepLab, ShallowMidFusionDeepLab, DeepMidFusionDeepLab, DeepLabTripleEventAutoencoderIntegratedWrapper
from .swinformer import SwinSegmentationWrapper
from .smp_unet import SMPUNet, SMPDualEncoderUNet, CustomSMPUNet
from .smp_segformer import SMPSegformer
from .smp_wrapper import SMPWrapper, SMPEventWrapper

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to initialize the model architecture based on model type.
    
    This function implements a registry pattern that maps model architecture names
    to their corresponding model classes. It supports various segmentation architectures:
    
    Model Categories:
        - SMP models: Modern architectures with pretrained encoders (ResNet, EfficientNet, etc.)
        - DeepLab: Various DeepLab v3+ fusion strategies
        - Dual-encoder: Custom architectures with separate RGB and event encoders
        - Attention: Attention-based fusion mechanisms
        - Autoencoder: Edge detection and denoising models
        - 3D: Spatiotemporal models for temporal event stacks
    
    Args:
        config: Configuration dictionary containing:
            - model_type (str): Model architecture name (e.g., "smp_unet", "deeplab")
            - num_of_out_classes (int): Number of output segmentation classes
            - encoder_name (str, optional): Backbone encoder for SMP models (e.g., "resnet34")
            - encoder_weights (str, optional): Pretrained weights ("imagenet", "ssl", etc.)
            - fusion_type (str, optional): Fusion strategy ("concat", "add", etc.)
            - event_channels (int, optional): Number of event input channels
            - time_steps (int, optional): Temporal depth for 3D models
            
    Returns:
        Initialized PyTorch nn.Module model ready for training or inference
        
    Raises:
        ValueError: If model_type is not in registry or initialization fails
        
    Example:
        >>> config = {
        ...     'model_type': 'smp_unet',
        ...     'encoder_name': 'resnet34',
        ...     'encoder_weights': 'imagenet',
        ...     'num_of_out_classes': 11
        ... }
        >>> model = get_model(config)
        >>> type(model).__name__
        'SMPUNet'
        
    Note:
        To add a new model architecture:
        1. Implement the model class inheriting from nn.Module
        2. Add import statement at the top of this file
        3. Add model_type -> ModelClass mapping to model_registry
        4. Ensure the model class accepts config dict in __init__
        
        Registry keys can map to the same class with different config parameters,
        allowing variants of the same architecture (e.g., different fusion strategies).
    """
    model_registry = {
        "smp_unet": SMPUNet,
        "smp_generic": SMPWrapper,
        "smp_event_generic": SMPEventWrapper,
        "custom_smp_unet": CustomSMPUNet,
        "smp_dual_encoder_unet": SMPDualEncoderUNet,
        "smp_dual_encoder_unet_3_channel": SMPDualEncoderUNet,
        "smp_segformer": SMPSegformer,
        "basic": BaseUNet,
        "deeplab": DeepLabWrapper,
        "deeplab_event": DeepLabEventIntegratedWrapper,
        "deeplab_event_3channel": DeepLabEventIntegratedWrapper,
        "deeplab_triple_input": DeepLabTripleEventAutoencoderIntegratedWrapper,
        "deeplab_event_v2": DeepLabEventIntegratedWrapperv2,
        "deeplab_4_channel": EarlyFusionDeepLab,
        "deeplab_mid_fusion_deep": DeepMidFusionDeepLab,
        "deeplab_mid_fusion_shallow": ShallowMidFusionDeepLab,
        "dual_deeplab": DualDeepLabFusionModel,
        "dual_deeplab_3D": DeepLabEventIntegratedWrapper3D,
        "deeplab_transformer_attention": GatedCrossModalTransformerFusionModel,
        "swinformer": SwinSegmentationWrapper,
        "attentionv1": UNetWithEventAttentionv1,
        "attentionv2": UNetWithEventAttentionv2,
        "attentionv3": UNetWithEventAttentionv3,
        "attentionv4": UNetWithEventAttentionv4,
        "dual_encoder": BaseDualEncoderUNet,
        "dual_encoder_attention": DualEncoderWithAttentionUNet,
        "dual_encoder_3D": DualEncoder3DUNet,
        "attention_auto_encoder_v1": Attention_Autoencoder_Unet_v1,
        "attention_auto_encoder_v2": Attention_Autoencoder_Unet_v2,
        "attention_auto_encoder_v3": Attention_Autoencoder_Unet_v3,
        "attention_auto_encoder_v4": Attention_Autoencoder_Unet_v4,
        "auto_encoder_dual_encoder": BaseDualEncoderUNet,
        "deeplab_autoencoder_seg": DeepLabEventIntegratedWrapper,
        "deeplab_autoencoder_seg_v2": DeepLabEventIntegratedWrapperv2,
        "deeplab_autoencoder_seg_morp": DeepLabEventIntegratedWrapper,
        "deeplab_autoencoder": DeepLabWrapper,
        "denoise_autoencoder": AutoencoderUNet,
        "denoise_autoencoder_double_channel": AutoencoderUNet,
        "concatenated_autoencoder": ConcatenatedAutoencoder,
        "denoise_autoencoder_3D": AutoencoderUNet3D,
        "denoise_autoencoder_3D_double_channel": AutoencoderUNet3D,
    }

    model_type = config.get('model_type')
    if model_type not in model_registry:
        raise ValueError(f"Invalid model type: {model_type}. Available types: {list(model_registry.keys())}")

    model_class = model_registry[model_type]
    # Initialize the dataset
    try:
        return model_class(config)
    except TypeError as e:
        raise ValueError(f"Error initializing model class for model type '{model_type}': {e}")