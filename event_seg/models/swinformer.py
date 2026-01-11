"""
Swin Transformer wrapper for semantic segmentation.

This module provides a wrapper around the Swin Transformer backbone
for semantic segmentation tasks. Swin Transformer is a hierarchical
vision transformer that uses shifted windows for efficient attention.

Architecture:
    - Backbone: Swin Transformer (tiny/small/base/large variants)
    - Feature extraction: Multi-scale features from intermediate stages
    - Decoder: Simple upsampling decoder with conv layers
    - Output: Per-pixel class predictions

Key Features:
    - Efficient attention via shifted windows
    - Hierarchical feature maps like CNNs
    - Can handle arbitrary image sizes
    - Pretrained on ImageNet (optional)

Available Backbones (via timm):
    - swin_tiny_patch4_window7_224: 28M params
    - swin_small_patch4_window7_224: 50M params
    - swin_base_patch4_window7_224: 88M params
    - swin_large_patch4_window7_224: 197M params

Configuration:
    Required config keys:
        - num_of_out_classes (int): Number of segmentation classes (default: 21)
    
    Optional config keys:
        - backbone_name (str): Swin model variant (default: 'swin_tiny_patch4_window7_224')
        - img_size (tuple): Expected input resolution (default: (440, 640))

Reference:
    Liu et al. "Swin Transformer: Hierarchical Vision Transformer using 
    Shifted Windows" ICCV 2021
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinSegmentationWrapper(nn.Module):
    """
    Swin Transformer-based segmentation model.
    
    Uses Swin Transformer as a feature extractor and adds a simple
    decoder for pixel-wise predictions.
    
    Architecture:
        Input → Swin Backbone → Multi-scale features → Decoder → Logits
    
    Attributes:
        backbone: Swin Transformer feature extractor
        decoder: Simple upsampling decoder (conv + ReLU + conv)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Swin segmentation wrapper.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(SwinSegmentationWrapper, self).__init__()
        num_of_out_classes = config.get('num_of_out_classes', 21)
        backbone_name = config.get('backbone_name', 'swin_tiny_patch4_window7_224')
        
        # Create Swin backbone in features_only mode for intermediate features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3),  # Extract features from stages 1, 2, 3
            img_size=(440, 640)     # Expected input resolution
        )
        
        # Get channel count from last feature stage
        last_stage_channels = self.backbone.feature_info[-1]['num_chs']
        
        # Simple decoder: conv → ReLU → 1x1 conv → upsample
        self.decoder = nn.Sequential(
            nn.Conv2d(last_stage_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_of_out_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, event: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Swin segmentation model.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W)
            event: Unused, for interface compatibility with event models
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Extracts multi-scale features from Swin backbone
            - Uses last feature stage for decoding
            - Upsamples to match input resolution
            - Transposes features from (B, H, W, C) to (B, C, H, W)
        """
        # Extract multi-scale features from backbone
        features = self.backbone(x)
        last_feature = features[-1]  # Use deepest feature map
        
        # Transpose from NHWC to NCHW format
        last_feature = last_feature.permute(0, 3, 1, 2)
        
        # Decode features to segmentation logits
        logits = self.decoder(last_feature)
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        
        return logits
