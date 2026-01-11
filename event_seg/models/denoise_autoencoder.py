"""
Autoencoder models for event frame denoising and edge extraction.

This module provides U-Net-based autoencoders for processing event camera data:
- AutoencoderUNet: Standard 2D U-Net autoencoder for denoising
- AutoencoderUNet3D: 3D U-Net for temporal event sequences
- ConcatenatedAutoencoder: Two-stage autoencoder cascade

Use Cases:
    - Event frame denoising and smoothing
    - Edge map extraction from event streams
    - Feature extraction for downstream segmentation
    - Temporal consistency in event sequences (3D variant)

Typical Pipeline:
    Events → AutoencoderUNet → Edge maps → Segmentation model

Configuration:
    Required config keys:
        - num_of_in_channels (int): Input channels (typically 1 for events)
        - num_of_out_classes (int): Output channels (1 for binary edges, 2 for multi-channel)
    
    Optional config keys:
        - bilinear (bool): Use bilinear interpolation (default: False)
        - dropout_rate (float): Dropout probability (default: 0.0)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseUNet
from .model_parts import DoubleConv3D, Down3D, Up3D, OutConv3D

class AutoencoderUNet(BaseUNet):
    """
    Standard 2D U-Net autoencoder for event frame denoising.
    
    Identical architecture to BaseUNet but used for autoencoding tasks
    (e.g., event frame → edge map) rather than segmentation.
    
    Architecture:
        Encoder: 5 levels (64→128→256→512→1024 channels)
        Decoder: 4 levels with skip connections
        Output: Single-channel or multi-channel edge maps
    
    Attributes:
        Inherits all attributes from BaseUNet
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize autoencoder U-Net.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(AutoencoderUNet, self).__init__(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input event frame with shape (B, C, H, W)
        
        Returns:
            Denoised/edge-extracted output with shape (B, num_out, H, W)
            
        Note:
            - Typically used with BCE or MSE loss
            - Output should be compared to ground truth edge maps
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class AutoencoderUNet3D(nn.Module):
    """
    3D U-Net autoencoder for temporal event sequences.
    
    Processes event sequences with temporal dimension using 3D convolutions,
    then collapses the temporal dimension to produce 2D output.
    
    Architecture:
        Input: (B, C, D, H, W) where D is temporal depth
        Encoder: 5 levels with 3D convolutions
        Decoder: 4 levels with 3D skip connections
        Temporal pooling: Collapse D dimension via adaptive averaging
        Output: (B, C, H, W) - 2D edge maps
    
    Use Cases:
        - Temporal smoothing of event sequences
        - Extracting temporally-consistent edges
        - Leveraging multiple frames for better denoising
    
    Attributes:
        num_of_in_channels: Input channels (typically 1 for events)
        num_of_out_classes: Output channels
        bilinear: Whether to use bilinear interpolation
        dropout_rate: Dropout probability
        inc, down1-4: 3D encoder blocks
        up1-4: 3D decoder blocks with skip connections
        outc: Final 3D output convolution
        temporal_pooling: Adaptive pooling to collapse temporal dimension
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize 3D autoencoder U-Net.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(AutoencoderUNet3D, self).__init__()
        self.num_of_in_channels = config.get('num_of_in_channels', 1)
        self.num_of_out_classes = config.get('num_of_out_classes', 1)
        self.bilinear = config.get('bilinear', False)
        self.dropout_rate = config.get("dropout_rate", 0.0)

        factor = 2 if self.bilinear else 1
        # 3D encoder with progressive downsampling
        self.inc = DoubleConv3D(self.num_of_in_channels, 64, dropout_rate=self.dropout_rate)
        self.down1 = Down3D(64, 128, dropout_rate=self.dropout_rate)
        self.down2 = Down3D(128, 256, dropout_rate=self.dropout_rate)
        self.down3 = Down3D(256, 512, dropout_rate=self.dropout_rate)
        self.down4 = Down3D(512, 1024 // factor, dropout_rate=self.dropout_rate)

        # 3D decoder with skip connections
        self.up1 = Up3D(1024, 512 // factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up2 = Up3D(512, 256 // factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up3 = Up3D(256, 128 // factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up4 = Up3D(128, 64, self.bilinear, dropout_rate=self.dropout_rate)

        self.outc = OutConv3D(64, self.num_of_out_classes)

        # Temporal pooling: collapse D dimension to 1
        self.temporal_pooling = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D autoencoder.
        
        Args:
            x: Input event sequence with shape (B, C, D, H, W)
                - D: temporal depth (number of frames)
        
        Returns:
            2D edge map with shape (B, C, H, W)
            
        Note:
            - Processes temporal sequences with 3D convolutions
            - Collapses temporal dimension via adaptive pooling
            - Preserves spatial resolution through skip connections
        """
        # 3D Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck

        # 3D Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)  # [B, C, D, H, W]
        x = self.temporal_pooling(x)  # [B, C, 1, H, W]
        return x.squeeze(2)  # Remove temporal dim: [B, C, H, W]
    
class ConcatenatedAutoencoder(nn.Module):
    """
    Two-stage cascaded autoencoder for progressive denoising.
    
    Applies two autoencoder stages in sequence, where the output of
    the first autoencoder becomes the input to the second.
    
    Architecture:
        Input → AutoencoderUNet1 → Intermediate → AutoencoderUNet2 → Output
    
    Use Cases:
        - Progressive denoising with two refinement stages
        - Learning hierarchical edge representations
        - Improving edge quality through iteration
    
    Attributes:
        autoencoder1: First autoencoder stage
        autoencoder2: Second autoencoder stage
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize concatenated two-stage autoencoder.
        
        Args:
            config: Configuration dictionary (shared by both stages)
        """
        super(ConcatenatedAutoencoder, self).__init__()

        # First autoencoder stage
        self.autoencoder1 = AutoencoderUNet(config)

        # Second autoencoder stage
        self.autoencoder2 = AutoencoderUNet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through two-stage autoencoder.
        
        Args:
            x: Input event frame with shape (B, C, H, W)
        
        Returns:
            Twice-refined output with shape (B, C, H, W)
            
        Note:
            - First stage produces intermediate denoised output
            - Second stage refines the intermediate result
            - Can be trained end-to-end or stage-by-stage
        """
        # First stage: initial denoising
        intermediate_output = self.autoencoder1(x)

        # Second stage: refinement
        final_output = self.autoencoder2(intermediate_output)

        return final_output