"""
Base U-Net model implementation for semantic segmentation.

This module implements the classic U-Net architecture with configurable channels,
dropout, and interpolation modes. The U-Net follows an encoder-decoder structure
with skip connections for preserving spatial information.

Architecture:
    Encoder (Contracting Path):
        - Initial: 64 channels
        - Down1: 128 channels
        - Down2: 256 channels
        - Down3: 512 channels
        - Down4 (Bottleneck): 1024 channels (or 512 if bilinear)
    
    Decoder (Expanding Path):
        - Up1: 512 channels
        - Up2: 256 channels
        - Up3: 128 channels
        - Up4: 64 channels
        - Output: num_of_out_classes channels
        
    Each encoder block: Conv → ReLU → Conv → ReLU → MaxPool → Dropout
    Each decoder block: Upsample → Concat → Conv → ReLU → Conv → ReLU → Dropout

Configuration:
    Required config keys:
        - num_of_in_channels (int): Number of input channels (3 for RGB)
        - num_of_out_classes (int): Number of segmentation classes
    
    Optional config keys:
        - bilinear (bool): Use bilinear interpolation for upsampling (default: False)
                          If False, uses transposed convolutions
        - dropout_rate (float): Dropout probability after each conv block (default: 0.0)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .model_parts import DoubleConv, Down, Up, OutConv

class BaseUNet(nn.Module):
    """
    Classic U-Net architecture for semantic segmentation.
    
    This is the standard U-Net implementation with 5 encoder levels and 4 decoder
    levels, using skip connections to preserve spatial details during upsampling.
    
    Attributes:
        num_of_in_channels: Number of input channels
        num_of_out_classes: Number of output segmentation classes
        bilinear: Whether to use bilinear interpolation (vs transposed conv)
        dropout_rate: Dropout probability for regularization
        inc: Initial double convolution layer
        down1-4: Encoder downsampling blocks
        up1-4: Decoder upsampling blocks
        outc: Final output convolution
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Base U-Net model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(BaseUNet, self).__init__()
        self.num_of_in_channels = config.get('num_of_in_channels')
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.bilinear = config.get('bilinear', False)
        self.dropout_rate = config.get('dropout_rate', 0.0)

        # Encoder: Progressive downsampling with channel expansion
        self.inc = DoubleConv(self.num_of_in_channels, 64, dropout_rate=self.dropout_rate)
        self.down1 = Down(64, 128, dropout_rate=self.dropout_rate)
        self.down2 = Down(128, 256, dropout_rate=self.dropout_rate)
        self.down3 = Down(256, 512, dropout_rate=self.dropout_rate)
        
        # Bottleneck: Reduce channels by 2 if using bilinear upsampling
        self.factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // self.factor, dropout_rate=self.dropout_rate)

        # Decoder: Progressive upsampling with skip connections
        self.up1 = Up(1024, 512 // self.factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up2 = Up(512, 256 // self.factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up3 = Up(256, 128 // self.factor, self.bilinear, dropout_rate=self.dropout_rate)
        self.up4 = Up(128, 64, self.bilinear, dropout_rate=self.dropout_rate)

        # Output: 1x1 convolution to produce segmentation map
        self.outc = OutConv(64, self.num_of_out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor with shape (B, C_in, H, W)
        
        Returns:
            Segmentation logits with shape (B, C_out, H, W)
            
        Note:
            - Encoder outputs are preserved for skip connections
            - Input is automatically padded to multiples of 32 by dataset
            - Returns logits (not softmax probabilities)
        """
        # Encoder with skip connections
        x1 = self.inc(x)        # 64 channels
        x2 = self.down1(x1)     # 128 channels
        x3 = self.down2(x2)     # 256 channels
        x4 = self.down3(x3)     # 512 channels
        x5 = self.down4(x4)     # 1024/512 channels (bottleneck)
        
        # Decoder with skip connections from encoder
        x = self.up1(x5, x4)    # Upsample + concat with x4
        x = self.up2(x, x3)     # Upsample + concat with x3
        x = self.up3(x, x2)     # Upsample + concat with x2
        x = self.up4(x, x1)     # Upsample + concat with x1
        
        # Output segmentation map
        logits = self.outc(x)
        return logits

