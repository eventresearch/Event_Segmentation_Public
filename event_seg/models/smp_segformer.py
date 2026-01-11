"""
Segformer model wrapper using Segmentation Models PyTorch (SMP).

This module provides a lightweight wrapper around the SMP Segformer implementation,
which is a transformer-based segmentation architecture offering excellent
performance with fewer parameters than traditional CNNs.

Architecture:
    Segformer uses a hierarchical transformer encoder (MiT - Mix Transformer)
    combined with a lightweight All-MLP decoder. Key features:
    - Multi-scale feature extraction via hierarchical transformers
    - Mix-FFN for positional encoding without interpolation
    - All-MLP decoder for efficiency
    - No need for complex decoder structures
    
Available Encoders:
    - mit_b0: Lightweight (3.7M params, 512 embedding dim)
    - mit_b1: Small (13.7M params, 512 embedding dim)
    - mit_b2: Medium (27.4M params, 768 embedding dim)  
    - mit_b3: Base (47.3M params, 1024 embedding dim)
    - mit_b4: Large (64.1M params, 1024 embedding dim)
    - mit_b5: Extra Large (84.7M params, 1024 embedding dim)

Configuration:
    Required config keys:
        - num_of_in_channels (int): Input channels (default: 3)
        - num_of_out_classes (int): Output segmentation classes (default: 1)
    
    Optional config keys:
        - encoder_name (str): MiT encoder variant (default: 'mit_b0')
        - encoder_weights (str): Pretrained weights ('imagenet' or None, default: 'imagenet')

Reference:
    Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation 
    with Transformers" NeurIPS 2021
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SMPSegformer(nn.Module):
    """
    Segformer model for semantic segmentation using transformer encoders.
    
    This wrapper provides easy configuration of Segformer architectures via
    the config dictionary, with support for pretrained ImageNet weights.
    
    Attributes:
        model: SMP Segformer instance with configured encoder and decoder
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Segformer model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        self.model = smp.Segformer(
            encoder_name=config.get('encoder_name', 'mit_b0'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('num_of_in_channels', 3),
            classes=config.get('num_of_out_classes', 1),
            activation=None  # Return logits, apply softmax/sigmoid in loss
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Segformer.
        
        Args:
            x: Input tensor with shape (B, C, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Returns logits (no activation applied)
            - Input resolution should be divisible by 32 for optimal performance
            - Transformer attention works on patch-based representations
        """
        return self.model(x)
