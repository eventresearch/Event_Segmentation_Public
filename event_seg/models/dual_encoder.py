"""
Custom dual-encoder U-Net implementations for multi-modal segmentation.

This module provides custom-built (non-SMP) dual-encoder architectures with
various fusion strategies and attention mechanisms.

Architectures:
    - BaseDualEncoderUNet: Basic dual-encoder with concat/add fusion
    - DualEncoderWithAttentionUNet: Adds attention mechanism for feature refinement
    - TripleEncoderUNet: Three encoders (RGB + Events + Autoencoder features)

Key Features:
    - Custom encoder/decoder blocks with configurable dropout
    - Multiple fusion strategies (concat, add)
    - Optional bilinear upsampling
    - Attention-based feature refinement
    - Triple-input support for complex multi-modal scenarios
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .model_parts import *


class BaseDualEncoderUNet(nn.Module):
    """
    Basic dual-encoder U-Net with custom conv blocks.
    
    Unlike SMPDualEncoderUNet which uses pretrained backbones, this
    implementation uses custom convolutional blocks trained from scratch.
    Useful when pretrained weights are not beneficial or when you need
    more control over the architecture.
    
    Architecture:
        - Two parallel encoders (RGB and Event)
        - Feature fusion at each scale (concat or add)
        - Single decoder with skip connections
        - Configurable dropout for regularization
    
    Attributes:
        num_of_in_channels: RGB input channels (typically 3)
        num_of_out_classes: Output segmentation classes
        bilinear: Use bilinear upsampling (True) vs transposed conv (False)
        fusion_type: 'concat' or 'add' for feature fusion
        dropout_rate: Dropout probability for regularization
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize base dual-encoder U-Net.
        
        Args:
            config: Configuration dictionary containing:
                - num_of_in_channels (int): RGB input channels
                - num_of_out_classes (int): Output classes
                - bilinear (bool): Use bilinear upsampling (default: False)
                - fusion_type (str): 'concat' or 'add' (default: 'concat')
                - dropout_rate (float): Dropout probability (default: 0.0)
                
        Raises:
            AssertionError: If fusion_type not in ['concat', 'add']
        """
        super(BaseDualEncoderUNet, self).__init__()
        self.num_of_in_channels = config.get('num_of_in_channels')
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.bilinear = config.get('bilinear')
        self.fusion_type = config.get('fusion_type')
        self.dropout_rate = config.get("dropout_rate", 0.0)  # ✅ Get dropout from config

        assert self.fusion_type in ["concat", "add"], "fusion_type must be 'concat' or 'add'"
        print(f"Using {self.fusion_type} fusion")

        self.factor = 2 if self.bilinear else 1

        # Adjust decoder input channels for concat
        self.bottleneck_channels = (1024 * 2 // self.factor) if self.fusion_type == "concat" else (1024 // self.factor)
        self.skip_channels = lambda x: (x * 2 // self.factor) if self.fusion_type == "concat" else (x // self.factor)

        self.rgb_inc = DoubleConv(self.num_of_in_channels, 64, dropout_rate=self.dropout_rate)
        self.rgb_down1 = Down(64, 128, dropout_rate=self.dropout_rate)
        self.rgb_down2 = Down(128, 256, dropout_rate=self.dropout_rate)
        self.rgb_down3 = Down(256, 512, dropout_rate=self.dropout_rate)
        self.rgb_down4 = Down(512, 1024 // self.factor, dropout_rate=self.dropout_rate)

        self.event_inc = DoubleConv(1, 64, dropout_rate=self.dropout_rate)
        self.event_down1 = Down(64, 128, dropout_rate=self.dropout_rate)
        self.event_down2 = Down(128, 256, dropout_rate=self.dropout_rate)
        self.event_down3 = Down(256, 512, dropout_rate=self.dropout_rate)
        self.event_down4 = Down(512, 1024 // self.factor, dropout_rate=self.dropout_rate)

        # Decoder
        self.up1 = Up(self.bottleneck_channels, self.skip_channels(512), self.bilinear, dropout_rate=self.dropout_rate)
        self.up2 = Up(self.skip_channels(512), self.skip_channels(256), self.bilinear, dropout_rate=self.dropout_rate)
        self.up3 = Up(self.skip_channels(256), self.skip_channels(128), self.bilinear, dropout_rate=self.dropout_rate)
        self.up4 = Up(self.skip_channels(128), 64, self.bilinear, dropout_rate=self.dropout_rate)

        self.outc = OutConv(64, self.num_of_out_classes)

    def forward(self, rgb, event):
        # RGB Encoder
        rgb_x1 = self.rgb_inc(rgb)
        rgb_x2 = self.rgb_down1(rgb_x1)
        rgb_x3 = self.rgb_down2(rgb_x2)
        rgb_x4 = self.rgb_down3(rgb_x3)
        rgb_x5 = self.rgb_down4(rgb_x4)

        # Event Encoder
        event_x1 = self.event_inc(event)
        event_x2 = self.event_down1(event_x1)
        event_x3 = self.event_down2(event_x2)
        event_x4 = self.event_down3(event_x3)
        event_x5 = self.event_down4(event_x4)

        # Fusion at bottleneck
        x5 = torch.cat([rgb_x5, event_x5], dim=1) if self.fusion_type == "concat" else (rgb_x5 + event_x5)


        # Decoder with Skip Connection Fusion
        x = self.up1(x5, torch.cat([rgb_x4, event_x4], dim=1) if self.fusion_type == "concat" else (rgb_x4 + event_x4))
        x = self.up2(x, torch.cat([rgb_x3, event_x3], dim=1) if self.fusion_type == "concat" else (rgb_x3 + event_x3))
        x = self.up3(x, torch.cat([rgb_x2, event_x2], dim=1) if self.fusion_type == "concat" else (rgb_x2 + event_x2))
        x = self.up4(x, torch.cat([rgb_x1, event_x1], dim=1) if self.fusion_type == "concat" else (rgb_x1 + event_x1))

        logits = self.outc(x)
        return logits

class DualEncoderWithAttentionUNet(BaseDualEncoderUNet):
    def __init__(self, config):
        super(DualEncoderWithAttentionUNet, self).__init__(config)
        # Attention Module
        self.outc = OutConv(128, self.num_of_out_classes)
        self.attention = AttentionModule(64, config.get("dropout_rate", 0.0))  # ✅ Pass dropout

    def forward(self, rgb, event):
        x = super().forward(rgb, event)  # Use the BaseDualEncoderUNet forward function

        # Apply attention at the final stage
        attention = self.attention(x, event)
        x_att = x * attention  # Element-wise multiplication with attention weights
        x_ct = torch.cat([x_att, x], dim=1)  # Concatenate attention-applied features and original
        logits = self.outc(x_ct)
        
        return logits  
        
# class DualEncoder3DUNet(BaseDualEncoderUNet):
#     def __init__(self, config):
#         super(DualEncoder3DUNet, self).__init__(config)
#         self.event_channels = config.get('event_channels', 1)
        
#         # Event Encoder (3D Convolutions)
#         self.event_inc = DoubleConv3D(self.event_channels, 64, dropout_rate=self.dropout_rate)
#         self.event_down1 = Down3D(64, 128, dropout_rate=self.dropout_rate)
#         self.event_down2 = Down3D(128, 256, dropout_rate=self.dropout_rate)
#         self.event_down3 = Down3D(256, 512, dropout_rate=self.dropout_rate)
#         self.event_down4 = Down3D(512, 1024 // self.factor, dropout_rate=self.dropout_rate)
        
#         # Temporal pooling to collapse the depth dimension (D) of event branch outputs.
#         # This pooling layer will be applied to the bottleneck (and optionally to skip connections).
#         self.temporal_pooling = nn.AdaptiveAvgPool3d((1, None, None))  # Collapse D to 1

#     def forward(self, rgb, event):
#         # RGB Encoder
#         rgb_x1 = self.rgb_inc(rgb)
#         rgb_x2 = self.rgb_down1(rgb_x1)
#         rgb_x3 = self.rgb_down2(rgb_x2)
#         rgb_x4 = self.rgb_down3(rgb_x3)
#         rgb_x5 = self.rgb_down4(rgb_x4)

#         # Event Encoder (3D Convolutions)
#         event_x1 = self.event_inc(event)
#         event_x2 = self.event_down1(event_x1)
#         event_x3 = self.event_down2(event_x2)
#         event_x4 = self.event_down3(event_x3)
#         event_x5 = self.event_down4(event_x4).squeeze(2)  # Reduce temporal dimension

#         # Fusion at bottleneck
#         if self.fusion_type == "concat":
#             x5 = torch.cat([rgb_x5, event_x5], dim=1)
#         else:
#             x5 = rgb_x5 + event_x5

#         # Decoder with Skip Connection Fusion
#         x = self.up1(
#             x5,
#             torch.cat([rgb_x4, event_x4.squeeze(2)], dim=1) if self.fusion_type == "concat" else (rgb_x4 + event_x4.squeeze(2))
#         )
#         x = self.up2(
#             x,
#             torch.cat([rgb_x3, event_x3.squeeze(2)], dim=1) if self.fusion_type == "concat" else (rgb_x3 + event_x3.squeeze(2))
#         )
#         x = self.up3(
#             x,
#             torch.cat([rgb_x2, event_x2.squeeze(2)], dim=1) if self.fusion_type == "concat" else (rgb_x2 + event_x2.squeeze(2))
#         )
#         x = self.up4(
#             x,
#             torch.cat([rgb_x1, event_x1.squeeze(2)], dim=1) if self.fusion_type == "concat" else (rgb_x1 + event_x1.squeeze(2))
#         )

#         # Output logits
#         logits = self.outc(x)
#         return logits
    

class DualEncoder3DUNet(nn.Module):
    def __init__(self, config):
        super(DualEncoder3DUNet, self).__init__()
        self.num_of_in_channels = config.get('num_of_in_channels')
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.bilinear = config.get('bilinear')
        self.fusion_type = config.get('fusion_type')
        self.dropout_rate = config.get("dropout_rate", 0.0)  # ✅ Get dropout from config
        self.event_channels = config.get('event_channels', 1)  # Set to match your event data (e.g. 1)
        

        assert self.fusion_type in ["concat", "add"], "fusion_type must be 'concat' or 'add'"
        print(f"Using {self.fusion_type} fusion")

        self.factor = 2 if self.bilinear else 1

        # Adjust decoder input channels for concat
        self.bottleneck_channels = (1024 * 2 // self.factor) if self.fusion_type == "concat" else (1024 // self.factor)
        self.skip_channels = lambda x: (x * 2 // self.factor) if self.fusion_type == "concat" else (x // self.factor)

        self.rgb_inc = DoubleConv(self.num_of_in_channels, 64, dropout_rate=self.dropout_rate)
        self.rgb_down1 = Down(64, 128, dropout_rate=self.dropout_rate)
        self.rgb_down2 = Down(128, 256, dropout_rate=self.dropout_rate)
        self.rgb_down3 = Down(256, 512, dropout_rate=self.dropout_rate)
        self.rgb_down4 = Down(512, 1024 // self.factor, dropout_rate=self.dropout_rate)

        self.event_inc = DoubleConv3D(self.event_channels, 64, dropout_rate=self.dropout_rate)
        self.event_down1 = Down3D(64, 128, dropout_rate=self.dropout_rate)
        self.event_down2 = Down3D(128, 256, dropout_rate=self.dropout_rate)
        self.event_down3 = Down3D(256, 512, dropout_rate=self.dropout_rate)
        self.event_down4 = Down3D(512, 1024 // self.factor, dropout_rate=self.dropout_rate)
        
        # Pool the temporal dimension from the event branch (assumes event features are [B, C, T, H, W])
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))  # Collapse T to 1

        # Decoder
        self.up1 = Up(self.bottleneck_channels, self.skip_channels(512), self.bilinear, dropout_rate=self.dropout_rate)
        self.up2 = Up(self.skip_channels(512), self.skip_channels(256), self.bilinear, dropout_rate=self.dropout_rate)
        self.up3 = Up(self.skip_channels(256), self.skip_channels(128), self.bilinear, dropout_rate=self.dropout_rate)
        self.up4 = Up(self.skip_channels(128), 64, self.bilinear, dropout_rate=self.dropout_rate)

        self.outc = OutConv(64, self.num_of_out_classes)

    def forward(self, rgb, event):
        # RGB Encoder
        rgb_x1 = self.rgb_inc(rgb)
        rgb_x2 = self.rgb_down1(rgb_x1)
        rgb_x3 = self.rgb_down2(rgb_x2)
        rgb_x4 = self.rgb_down3(rgb_x3)
        rgb_x5 = self.rgb_down4(rgb_x4)

        # Event Encoder
        event_x1 = self.event_inc(event)
        event_x2 = self.event_down1(event_x1)
        event_x3 = self.event_down2(event_x2)
        event_x4 = self.event_down3(event_x3)
        event_x5 = self.event_down4(event_x4)
        
        # Collapse the temporal dimension
        event_x5 = self.temporal_pool(event_x5)  # [B, 1024//factor, 1, H/16, W/16]
        event_x5 = event_x5.squeeze(2)           # [B, 1024//factor, H/16, W/16]
        event_x4 = self.temporal_pool(event_x4)  
        event_x4 = event_x4.squeeze(2)           
        event_x3 = self.temporal_pool(event_x3)  
        event_x3 = event_x3.squeeze(2)           
        event_x2 = self.temporal_pool(event_x2)  
        event_x2 = event_x2.squeeze(2)          
        event_x1 = self.temporal_pool(event_x1)  
        event_x1 = event_x1.squeeze(2)            
        
        # Fusion at bottleneck
        x5 = torch.cat([rgb_x5, event_x5], dim=1) if self.fusion_type == "concat" else (rgb_x5 + event_x5)

        # Decoder with Skip Connection Fusion
        x = self.up1(x5, torch.cat([rgb_x4, event_x4], dim=1) if self.fusion_type == "concat" else (rgb_x4 + event_x4))
        x = self.up2(x, torch.cat([rgb_x3, event_x3], dim=1) if self.fusion_type == "concat" else (rgb_x3 + event_x3))
        x = self.up3(x, torch.cat([rgb_x2, event_x2], dim=1) if self.fusion_type == "concat" else (rgb_x2 + event_x2))
        x = self.up4(x, torch.cat([rgb_x1, event_x1], dim=1) if self.fusion_type == "concat" else (rgb_x1 + event_x1))

        logits = self.outc(x)
        return logits

class AttentionModule(nn.Module):
    """Attention module with optional dropout"""
    def __init__(self, in_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # ✅ Dropout after activation
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, event):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)  # ✅ Dropout applied here
        attention = self.sigmoid(self.conv2(x)) * event
        return attention