"""
U-Net models using Segmentation Models PyTorch (SMP) library.

This module provides standard U-Net and dual-encoder U-Net implementations
using pretrained encoders from the SMP library.

Supported Architectures:
    - SMPUNet: Standard single-encoder U-Net with pretrained backbone
    - SMPDualEncoderUNet: Dual-encoder for multi-modal fusion (RGB + Events)

Supported Encoders:
    ResNet family: resnet18, resnet34, resnet50, resnet101, resnet152
    EfficientNet: efficientnet-b0 through efficientnet-b7
    MobileNet: mobilenet_v2
    DenseNet: densenet121, densenet169, densenet201
    VGG: vgg11, vgg13, vgg16, vgg19
    And many more from SMP library
"""

from typing import Dict, Any, List, Tuple
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import initialization as smp_init
import torch
import torch.nn as nn

class SMPUNet(nn.Module):
    """
    Standard U-Net with pretrained encoder from SMP library.
    
    This is a wrapper around segmentation_models_pytorch.Unet that provides
    a consistent interface with other models in this codebase.
    
    Architecture:
        - Encoder: Pretrained backbone (ResNet, EfficientNet, etc.)
        - Decoder: U-Net style decoder with skip connections
        - Head: Segmentation head with configurable output channels
    
    Attributes:
        model: SMP U-Net instance with specified configuration
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize standard U-Net with pretrained encoder.
        
        Args:
            config: Configuration dictionary containing:
                - encoder_name (str): Backbone architecture (default: 'resnet34')
                - encoder_weights (str): Pretrained weights ('imagenet' or None, default: 'imagenet')
                - num_of_in_channels (int): Input channels (default: 3 for RGB)
                - num_of_out_classes (int): Output classes (default: 1 for binary)
                
        Example:
            >>> config = {
            ...     'encoder_name': 'resnet50',
            ...     'encoder_weights': 'imagenet',
            ...     'num_of_in_channels': 3,
            ...     'num_of_out_classes': 11
            ... }
            >>> model = SMPUNet(config)
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=config.get('num_of_in_channels', 3),
            classes=config.get('num_of_out_classes', 1),
            activation=None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor with shape (B, C, H, W)
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
        """
        return self.model(x)


# Custom U-Net implementation using SMP encoder/decoder, similar to SMPDualEncoderUNet but for single input
class CustomSMPUNet(nn.Module):
    """
    Custom U-Net with explicit encoder, decoder, and segmentation head using SMP components.
    Mirrors the structure of SMPDualEncoderUNet but for a single input modality.
    
    This implementation matches the exact initialization scheme of smp.Unet to ensure
    identical behavior and performance. Uses kaiming_uniform for decoder and xavier_uniform
    for segmentation head, matching SMP's initialization.py.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize custom U-Net with SMP encoder/decoder.
        Args:
            config: dict with keys:
                - encoder_name (str): Backbone architecture (default: 'resnet34')
                - encoder_weights (str): Pretrained weights (default: 'imagenet')
                - num_of_in_channels (int): Input channels (default: 3)
                - num_of_out_classes (int): Output classes (default: 1)
        """
        super().__init__()
        self.encoder_name = config.get('encoder_name', 'resnet34')
        self.encoder_weights = config.get('encoder_weights', 'imagenet')
        self.num_of_in_channels = config.get('num_of_in_channels', 3)
        self.num_of_out_classes = config.get('num_of_out_classes', 1)

        # Encoder
        self.encoder = smp.encoders.get_encoder(
            name=self.encoder_name,
            in_channels=self.num_of_in_channels,
            depth=5,
            weights=self.encoder_weights
        )
        self.encoder_channels = self.encoder.out_channels

        # Decoder - set center=True for VGG encoders, False otherwise (same as smp.Unet)
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=True if self.encoder_name.startswith("vgg") else False,
            attention_type=None
        )

        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=self.num_of_out_classes,
            activation=None,
            kernel_size=3,
        )
        
        # Initialize decoder and segmentation head using SMP's API
        smp_init.initialize_decoder(self.decoder)
        smp_init.initialize_head(self.segmentation_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for custom U-Net.
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Step 1: Extract multi-scale features from encoder
        features = self.encoder(x)
        # Step 2: Decode features
        decoder_output = self.decoder(*features)
        # Step 3: Segmentation head
        logits = self.segmentation_head(decoder_output)
        return logits


class SMPDualEncoderUNet(nn.Module):
    """
    Dual-encoder U-Net for multi-modal segmentation (RGB + Events).
    
    This architecture uses two separate encoders to extract features from RGB images
    and event data independently, then fuses them at multiple levels before decoding.
    
    Structure mirrors CustomSMPUNet but with dual encoders and feature fusion.
    Uses SMP's initialization API for consistent weight initialization.
    
    Architecture:
        1. RGB Encoder: Pretrained backbone (ResNet, EfficientNet, etc.)
        2. Event Encoder: Same architecture but trained from scratch
        3. Fusion: Concatenation or addition at each encoder level
        4. Decoder: Single U-Net decoder processing fused features
        5. Segmentation Head: Generates class logits
    
    Fusion Strategies:
        - 'concat': Concatenate features along channel dimension (doubles channels)
        - 'add': Element-wise addition (preserves channel count)
    
    Typical Use Cases:
        - Event camera + RGB fusion for robust segmentation
        - Multi-modal learning with different input modalities
        - Temporal event data integration with spatial RGB
    
    Attributes:
        encoder_name: Backbone architecture name
        encoder_weights: Pretrained weights for RGB encoder
        num_of_in_channels: RGB input channels (typically 3)
        event_channels: Event data channels (typically 1)
        num_of_out_classes: Output segmentation classes
        fusion_type: Feature fusion method ('concat' or 'add')
        rgb_encoder: Pretrained encoder for RGB images
        event_encoder: Encoder for event data (no pretraining)
        decoder: U-Net decoder for fused features
        segmentation_head: Final classification layer
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize dual-encoder U-Net for multi-modal fusion.
        
        Args:
            config: Configuration dictionary containing:
                - encoder_name (str): Backbone architecture (default: 'resnet34')
                - encoder_weights (str): RGB encoder pretrained weights (default: 'imagenet')
                - num_of_in_channels (int): RGB input channels (default: 3)
                - event_channels (int): Event input channels (default: 1)
                - num_of_out_classes (int): Output segmentation classes (default: 1)
                - fusion_type (str): Fusion strategy - 'concat' or 'add' (default: 'concat')
                
        Raises:
            AssertionError: If fusion_type is not 'concat' or 'add'
            
        Example:
            >>> config = {
            ...     'encoder_name': 'resnet50',
            ...     'encoder_weights': 'imagenet',
            ...     'num_of_in_channels': 3,
            ...     'event_channels': 1,
            ...     'num_of_out_classes': 11,
            ...     'fusion_type': 'concat'
            ... }
            >>> model = SMPDualEncoderUNet(config)
        """
        super().__init__()
        
        self.encoder_name = config.get('encoder_name', 'resnet34')
        self.encoder_weights = config.get('encoder_weights', 'imagenet')
        self.num_of_in_channels = config.get('num_of_in_channels', 3)
        self.event_channels = config.get('event_channels', 1)
        self.num_of_out_classes = config.get('num_of_out_classes', 1)
        self.fusion_type = config.get('fusion_type', 'concat')
        
        assert self.fusion_type in ["concat", "add"], "fusion_type must be 'concat' or 'add'"
        print(f"SMPDualEncoderUNet: Using {self.fusion_type} fusion with {self.encoder_name} backbone")
        
        # RGB Encoder - pretrained on ImageNet
        self.rgb_encoder = smp.encoders.get_encoder(
            name=self.encoder_name,
            in_channels=self.num_of_in_channels,
            depth=5,
            weights=self.encoder_weights
        )
        
        # Event Encoder - same architecture, trained from scratch
        self.event_encoder = smp.encoders.get_encoder(
            name=self.encoder_name,
            in_channels=self.event_channels,
            depth=5,
            weights=None  # No pretrained weights for event encoder
        )
        
        # Get encoder output channels
        self.encoder_channels = self.rgb_encoder.out_channels
        
        # Calculate decoder input channels based on fusion strategy
        if self.fusion_type == "concat":
            decoder_channels = [ch * 2 for ch in self.encoder_channels]
        else:  # add
            decoder_channels = self.encoder_channels
        
        # Decoder - set center=True for VGG encoders, False otherwise (same as CustomSMPUNet)
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=decoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=True if self.encoder_name.startswith("vgg") else False,
            attention_type=None
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=self.num_of_out_classes,
            activation=None,
            kernel_size=3,
        )
        
        # Initialize decoder and segmentation head using SMP's API (same as CustomSMPUNet)
        smp_init.initialize_decoder(self.decoder)
        smp_init.initialize_head(self.segmentation_head)
        
    def fuse_features(
        self, 
        rgb_features: List[torch.Tensor], 
        event_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Fuse multi-modal features from RGB and event encoders.
        
        This method combines features extracted at multiple encoder levels,
        enabling the decoder to leverage information from both modalities.
        
        Args:
            rgb_features: List of RGB encoder outputs at different scales
            event_features: List of event encoder outputs at different scales
                
        Returns:
            List of fused features for decoder input
        """
        if self.fusion_type == "concat":
            return [torch.cat([rgb_feat, event_feat], dim=1) 
                    for rgb_feat, event_feat in zip(rgb_features, event_features)]
        else:  # add
            return [rgb_feat + event_feat 
                    for rgb_feat, event_feat in zip(rgb_features, event_features)]
    
    def forward(self, rgb: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dual-encoder multi-modal segmentation.
        
        Args:
            rgb: RGB image tensor (B, C_rgb, H, W)
            event: Event data tensor (B, C_event, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Extract multi-scale features from both encoders
        rgb_features = self.rgb_encoder(rgb)
        event_features = self.event_encoder(event)
        
        # Fuse features at each encoder level
        fused_features = self.fuse_features(rgb_features, event_features)
        
        # Decode fused features
        decoder_output = self.decoder(*fused_features)
        
        # Generate segmentation logits
        logits = self.segmentation_head(decoder_output)
        
        return logits
