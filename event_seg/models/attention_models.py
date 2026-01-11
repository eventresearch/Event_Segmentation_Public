"""
U-Net models with event-based attention mechanisms.

This module implements various U-Net architectures that incorporate event camera
data through different attention mechanisms. Event cameras capture temporal changes
asynchronously, providing complementary information to standard RGB frames.

Attention Variants:
    v1: Late fusion with multiplicative attention
        - Processes RGB and events separately
        - Applies event features as attention map after decoder
        - Simple element-wise multiplication
        
    v2: Learnable attention gates
        - Concatenates RGB and event features
        - Learns attention weights via conv layers + sigmoid
        - More flexible than v1 with trainable gating
        
    v3: Early fusion via concatenation
        - Concatenates RGB + events at input
        - Processes combined features through encoder
        - Simpler but loses RGB pretraining benefits
        
    v4: Dual-path with skip connection fusion
        - Separate encoders for RGB and events
        - Fuses at skip connections and decoder
        - Best for leveraging both modalities

Use Cases:
    - Event-based segmentation for high-speed scenes
    - Low-light segmentation where events provide additional signal
    - Temporal consistency in video segmentation
    - Complementing RGB with motion information

Configuration:
    Required config keys:
        - num_of_in_channels (int): RGB input channels (usually 3)
        - num_of_out_classes (int): Number of segmentation classes
        - event_channels (int): Event frame channels (1, 3, or more)
    
    Optional config keys:
        - dropout_rate (float): Dropout probability (default: 0.0)
        - bilinear (bool): Use bilinear interpolation (default: False)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseUNet, DoubleConv, OutConv

class UNetWithEventAttentionv1(BaseUNet):
    """
    U-Net with late fusion multiplicative event attention.
    
    This variant applies event features as an attention map after the decoder,
    using simple element-wise multiplication to modulate RGB features.
    
    Architecture:
        RGB path: Standard U-Net encoder-decoder
        Event path: 1x1 conv to match decoder channels
        Fusion: Element-wise multiplication at decoder output
    
    Attributes:
        event_channels: Number of event input channels
        event_transform: 1x1 conv + ReLU + Dropout for event processing
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize U-Net with event attention v1.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(UNetWithEventAttentionv1, self).__init__(config)
        self.event_channels = config.get('event_channels')
        self.dropout_rate = config.get("dropout_rate", 0.0)
        # Transform event features to match decoder output channels (64)
        self.event_transform = nn.Sequential(
            nn.Conv2d(self.event_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor, event_attention: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with event attention.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W)
            event_attention: Event frame tensor with shape (B, event_channels, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Event features are applied after full decoder pass
            - Uses element-wise multiplication for attention
            - Event features automatically match spatial dimensions
        """
        # Standard U-Net encoder-decoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Apply event attention via multiplication
        event_input = self.event_transform(event_attention)
        x_with_attention = x * event_input  # Element-wise attention
        
        logits = self.outc(x_with_attention)
        return logits
    
class UNetWithEventAttentionv2(BaseUNet):
    """
    U-Net with learnable attention gates for event fusion.
    
    This variant learns optimal attention weights by processing concatenated
    RGB and event features through convolutional layers with sigmoid activation.
    
    Architecture:
        RGB path: Standard U-Net encoder-decoder
        Event path: 1x1 conv to match decoder channels
        Attention gate: Conv → ReLU → Conv → Sigmoid
        Fusion: Learned attention weights * RGB features
    
    Attributes:
        event_channels: Number of event input channels
        event_transform: 1x1 conv for event feature extraction
        attention_gate: Learnable gating mechanism for fusion
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize U-Net with event attention v2.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(UNetWithEventAttentionv2, self).__init__(config)
        self.event_channels = config.get('event_channels', 1)
        self.dropout_rate = config.get("dropout_rate", 0.0)
        # Transform event features to 64 channels
        self.event_transform = nn.Sequential(
            nn.Conv2d(self.event_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
        # Learnable attention gate: processes concat(RGB, events) → attention weights
        self.attention_gate = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1),  # 128 → 64
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=1),  # 64 → 64
            nn.Sigmoid(),  # Output attention weights in [0, 1]
        )
        
    def forward(self, x: torch.Tensor, event_attention: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with learnable attention gates.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W)
            event_attention: Event frame tensor with shape (B, event_channels, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Resizes event features to match RGB spatial dimensions
            - Concatenates RGB and event features
            - Learns optimal attention weights via gating network
            - Applies learned weights to RGB features
        """
        # Standard U-Net encoder-decoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Resize event features to match RGB decoder output
        event_attention_resized = F.interpolate(
            event_attention, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        event_features = self.event_transform(event_attention_resized)
        
        # Learn attention weights from combined features
        combined_features = torch.cat([x, event_features], dim=1)  # Concat along channel dim
        attention_weights = self.attention_gate(combined_features)  # Learned weights
        x_with_attention = x * attention_weights  # Apply attention
        
        logits = self.outc(x_with_attention)
        return logits
    
class UNetWithEventAttentionv3(BaseUNet):
    """
    U-Net with early fusion of RGB and event frames.
    
    This variant concatenates RGB and event frames at the input, processing
    them together through the entire encoder-decoder pipeline.
    
    Architecture:
        Input: Concatenated RGB + events
        Encoder-Decoder: Standard U-Net on combined input
        Fusion: Early concatenation before first conv
    
    Advantages:
        - Simplest fusion approach
        - Allows full interaction between modalities
    
    Disadvantages:
        - Loses RGB pretraining benefits
        - Event noise affects entire network
    
    Attributes:
        event_channels: Number of event input channels
        inc: Modified initial conv accepting RGB + event channels
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize U-Net with event attention v3.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(UNetWithEventAttentionv3, self).__init__(config)
        self.event_channels = config.get('event_channels', 1)
        self.dropout_rate = config.get("dropout_rate", 0.0)
        # Modify initial conv to accept RGB + event channels
        self.inc = DoubleConv(
            self.num_of_in_channels + self.event_channels, 64, dropout_rate=self.dropout_rate
        )

    def forward(self, x: torch.Tensor, event_frame: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W)
            event_frame: Event frame tensor with shape (B, event_channels, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Concatenates RGB + events before processing
            - Events resized to match RGB spatial dimensions
            - Combined features processed through entire network
        """
        # Concatenate RGB and events at input
        event_frame_resized = F.interpolate(
            event_frame, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x_combined = torch.cat([x, event_frame_resized], dim=1)  # Early fusion
        
        # Process combined features through U-Net
        x1 = self.inc(x_combined)
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
    

class UNetWithEventAttentionv4(BaseUNet):
    """
    U-Net with attention module and skip connection fusion.
    
    This variant uses a dedicated attention module to compute event-based
    attention, then concatenates attended and original features before output.
    
    Architecture:
        RGB path: Standard U-Net encoder-decoder
        Attention module: Computes event-modulated attention weights
        Fusion: Concatenate (attended features, original features)
    
    Attributes:
        outc: Modified output conv accepting 128 channels (64*2)
        attention: Attention module for event-based gating
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize U-Net with event attention v4.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(UNetWithEventAttentionv4, self).__init__(config)
        self.outc = OutConv(128, self.num_of_out_classes)  # Accept 128 channels (64*2)
        self.dropout_rate = config.get("dropout_rate", 0.0)
        self.attention = AttentionModule(64, self.dropout_rate)

        
    def forward(self, x: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention module fusion.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W)
            event: Event frame tensor with shape (B, event_channels, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Computes event-modulated attention weights
            - Applies attention to RGB features
            - Concatenates attended and original features
            - Provides skip connection for stable training
        """
        # Standard U-Net encoder-decoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Compute event-based attention and apply
        attention = self.attention(x, event)
        x_att = x * attention  # Attended features
        
        # Concatenate attended and original features for robustness
        x_ct = torch.cat([x_att, x], dim=1)  # 64 + 64 = 128 channels
        logits = self.outc(x_ct)
        return logits    
        
class AttentionModule(nn.Module):
    """
    Attention module for computing event-modulated attention weights.
    
    This module processes RGB features and modulates them with event information
    to produce spatial attention weights.
    
    Attributes:
        conv1: First convolution for feature extraction
        conv2: Second convolution producing single-channel attention map
        relu: ReLU activation
        dropout: Optional dropout for regularization
        sigmoid: Sigmoid to normalize attention weights to [0, 1]
    """
    
    def __init__(self, in_channels: int, dropout_rate: float = 0.0) -> None:
        """
        Initialize attention module.
        
        Args:
            in_channels: Number of input feature channels
            dropout_rate: Dropout probability (default: 0.0)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Compute event-modulated attention weights.
        
        Args:
            x: RGB features with shape (B, C, H, W)
            event: Event frame with shape (B, 1, H, W)
        
        Returns:
            Attention weights with shape (B, 1, H, W)
            
        Note:
            - Processes RGB features through conv layers
            - Modulates with event information
            - Returns normalized attention in [0, 1]
        """
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        attention = self.sigmoid(self.conv2(x)) * event  # Event modulation
        return attention