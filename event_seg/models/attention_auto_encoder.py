"""
U-Net models with attention for autoencoder feature integration.

This module provides U-Net architectures that integrate autoencoder-extracted
edge features through various fusion strategies. The autoencoder typically
extracts edge maps from event data, which are then fused with RGB features.

Fusion Variants:
    v1: Attention-based modulation with concatenation
        - Uses attention module to compute event-based weights
        - Concatenates attended + original features
        
    v2: Direct channel expansion and concatenation
        - Repeats single-channel event to 64 channels
        - Simple concatenation without learned processing
        
    v3: Learned event feature extraction
        - Processes events through conv layers
        - Extracts 64-channel features for fusion
        
    v4: Early fusion via input concatenation
        - Concatenates event as additional input channel
        - Processes through entire U-Net together

Use Cases:
    - Edge-aware segmentation using autoencoder edge maps
    - Integrating geometric priors from event cameras
    - Multi-stage pipelines (autoencoder → segmentation)

Configuration:
    Required config keys:
        - num_of_in_channels (int): RGB input channels (usually 3)
        - num_of_out_classes (int): Number of segmentation classes
    
    Optional config keys:
        - event_channels (int): Event/autoencoder feature channels (default: 1)
        - dropout_rate (float): Dropout probability (default: 0.0)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from .base_model import BaseUNet, DoubleConv, OutConv

class Attention_Autoencoder_Unet_v1(BaseUNet):
    """
    U-Net with attention-based autoencoder feature integration.
    
    This variant uses an attention module to compute spatially-varying weights
    from autoencoder edge features, then concatenates attended and original
    RGB features before the output layer.
    
    Architecture:
        RGB path: Standard U-Net encoder-decoder
        Attention: Computes weights from RGB + autoencoder features
        Fusion: Concatenate (attended features, original features)
    
    Attributes:
        outc: Modified output conv accepting 128 channels
        attention: Attention module for feature modulation
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize attention-based autoencoder U-Net.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(Attention_Autoencoder_Unet_v1, self).__init__(config)
        self.outc = (OutConv(128, self.num_of_out_classes))

        self.attention = AttentionModule(64, config.get("dropout_rate", 0.0))  # Pass dropout rate

    def forward(self, x, event):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        attention = self.attention(x, event)
        x_att = x * attention
        x_ct = torch.cat([x_att, x],dim=1)
        logits = self.outc(x_ct)
        return logits        
        
class AttentionModule(nn.Module):
    """Attention module with optional dropout"""
    def __init__(self, in_channels, dropout_rate=0.0):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()  # Dropout after activation
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, event):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)  # Dropout applied here
        attention = self.sigmoid(self.conv2(x)) * event
        return attention
    
class Attention_Autoencoder_Unet_v2(BaseUNet):
    def __init__(self, config):
        super(Attention_Autoencoder_Unet_v2, self).__init__(config)
        self.outc = (OutConv(128, self.num_of_out_classes))
    def forward(self, x, event):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Expand event frame to 64 channels
        event_expanded = event.repeat(1, 64, 1, 1)
        x_ct = torch.cat([x, event_expanded], dim=1)  # Concatenate with the output
        logits = self.outc(x_ct)
        return logits
    
class Attention_Autoencoder_Unet_v3(BaseUNet):
    def __init__(self, config):
        super(Attention_Autoencoder_Unet_v3, self).__init__(config)
        self.outc = (OutConv(128, self.num_of_out_classes))
        self.event_channels = config.get('event_channels', 1)
        self.event_processor = nn.Sequential(
            nn.Conv2d(self.event_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get("dropout_rate", 0.0)) if config.get("dropout_rate", 0.0) > 0 else nn.Identity()  # Dropout
        )
    def forward(self, x, event):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Process the event frame
        event_features = self.event_processor(event)
        x_ct = torch.cat([x, event_features], dim=1)
        logits = self.outc(x_ct)
        return logits
    
class Attention_Autoencoder_Unet_v4(BaseUNet):
    def __init__(self, config):
        super(Attention_Autoencoder_Unet_v4, self).__init__(config)
        self.num_of_in_channels += 1  # Add 1 for the event channel
        self.inc = DoubleConv(self.num_of_in_channels, 64, dropout_rate=config.get("dropout_rate", 0.0))

    def forward(self, x, event):
        # Concatenate event as the last channel of the input
        x = torch.cat([x, event], dim=1)

        # Forward pass through U-Net
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