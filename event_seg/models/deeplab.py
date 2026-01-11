"""
DeepLabV3 model wrappers for semantic segmentation.

This module provides various DeepLabV3-based architectures with different fusion
strategies for integrating RGB images with event camera data and autoencoder features.

DeepLabV3 Architecture:
    - Backbone: ResNet50 with atrous convolutions
    - ASPP (Atrous Spatial Pyramid Pooling) module
    - Output stride: 16 (downsample by 16x)
    - Pretrained on COCO + VOC datasets (optional)

Available Models:
    1. DeepLabWrapper: Standard DeepLabV3 for RGB-only segmentation
    2. DeepLabEventIntegratedWrapper: Late fusion with event attention
    3. DeepLabEventIntegratedWrapperv2: Event fusion without sigmoid gating
    4. DualDeepLabFusionModel: Dual DeepLab branches for RGB + events
    5. DeepLabEventIntegratedWrapper3D: 3D convolutions for temporal events
    6. GatedCrossModalTransformerFusionModel: Transformer-based fusion
    7. EarlyFusionDeepLab: Early concatenation of RGB + events
    8. DeepMidFusionDeepLab: Deep event branch with mid-level fusion
    9. ShallowMidFusionDeepLab: Shallow event branch with mid-level fusion
    10. DeepLabTripleEventAutoencoderIntegratedWrapper: Triple-input fusion

Fusion Strategies:
    - Early Fusion: Concatenate inputs before backbone
    - Mid-Level Fusion: Fuse at backbone feature level
    - Late Fusion: Fuse at decoder output level
    - Attention-based: Use event data to modulate RGB features
    - Transformer: Cross-modal attention for feature interaction

Configuration:
    Required config keys:
        - num_of_out_classes (int): Number of segmentation classes
    
    Optional config keys:
        - pretrained (bool): Use COCO+VOC pretrained weights (default: False)
        - event_channels (int): Event input channels (default: 1)
        - autoencoder_channels (int): Autoencoder feature channels (default: 1)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn.functional as F
from torchvision import models

class DeepLabWrapper(nn.Module):
    """
    Standard DeepLabV3 wrapper for RGB image segmentation.
    
    This is the baseline DeepLabV3 model without event integration,
    useful for comparing against multi-modal variants.
    
    Attributes:
        num_of_out_classes: Number of segmentation classes
        pretrained: Whether to use pretrained weights
        deeplab: DeepLabV3-ResNet50 model
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize DeepLabV3 wrapper.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(DeepLabWrapper, self).__init__()
        # Retrieve the number of classes from the configuration
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            # Initialize with pretrained weights, then replace classifier
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
    
    def forward(self, x: torch.Tensor, event: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through DeepLabV3.
        
        Args:
            x: RGB input tensor with shape (B, 3, H, W) or (B, 1, H, W)
            event: Unused, for interface compatibility with event models
        
        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
            
        Note:
            - Automatically converts grayscale to 3-channel
            - Returns only 'out' key from DeepLabV3 dict output
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        # DeepLabV3 returns a dictionary with keys "out" and "aux"
        output = self.deeplab(x)
        logits = output['out']  # Main output
        return logits

class DeepLabEventIntegratedWrapper(nn.Module):
    """
    DeepLabV3 with event attention-based late fusion.
    
    This model processes RGB through DeepLabV3 and events through a lightweight
    branch, then applies event features as an attention mask to modulate the
    RGB segmentation logits.
    
    Architecture:
        RGB → DeepLabV3 → img_logits
        Event → Conv layers → event_attention
        Fusion: img_logits * sigmoid(event_attention)
    
    Attributes:
        deeplab: DeepLabV3 model for RGB processing
        event_branch: Lightweight CNN for event processing
        fusion_conv: Refinement layers for fused features
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize DeepLab with event integration.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(DeepLabEventIntegratedWrapper, self).__init__()
        # Number of segmentation classes
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            # Initialize the model with pretrained weights but default classifier
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            # Replace the classifier with a new one that has the correct number of classes
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            # When training from scratch, we can directly specify the number of classes
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Event branch: processes event input and outputs a tensor with shape [B, num_of_out_classes, H, W]
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv2d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Fusion layers: after modulating the image branch with event attention,
        # we concatenate the original and modulated logits and refine them.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.num_of_out_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
    
    def forward(self, x, event=None):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Get image segmentation logits from DeepLabV3
        output = self.deeplab(x)
        img_logits = output['out']  # Shape: [B, num_of_out_classes, H, W]
        
        if event is not None:
            # Process event data through the event branch
            event_out = self.event_branch(event)  # Shape: [B, num_of_out_classes, H, W]
            # Use the event branch output as an attention mask (normalize between 0 and 1)
            attn = torch.sigmoid(event_out)
            # Multiply the image logits by the attention mask to modulate them
            modulated_img = img_logits * attn
            # Concatenate the original image logits and the modulated logits along the channel dimension
            fused_features = torch.cat([img_logits, modulated_img], dim=1)  # Shape: [B, 2*num_of_out_classes, H, W]
            # Refine the fused features with additional convolutional layers
            final_logits = self.fusion_conv(fused_features)
        else:
            raise ValueError("Event cannot be None")
        
        return final_logits

class DeepLabEventIntegratedWrapperv2(nn.Module):
    def __init__(self, config):
        super(DeepLabEventIntegratedWrapperv2, self).__init__()
        # Number of segmentation classes
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            # Initialize the model with pretrained weights but default classifier
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            # Replace the classifier with a new one that has the correct number of classes
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            # When training from scratch, we can directly specify the number of classes
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Event branch: processes event input and outputs a tensor with shape [B, num_of_out_classes, H, W]
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv2d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Fusion layers: after modulating the image branch with event attention,
        # we concatenate the original and modulated logits and refine them.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.num_of_out_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
    
    def forward(self, x, event=None):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Get image segmentation logits from DeepLabV3
        output = self.deeplab(x)
        img_logits = output['out']  # Shape: [B, num_of_out_classes, H, W]
        
        if event is not None:
            # Process event data through the event branch
            event_out = self.event_branch(event)  # Shape: [B, num_of_out_classes, H, W]
            # Use the event branch output as an attention mask (normalize between 0 and 1)
            # attn = torch.sigmoid(event_out)
            # Multiply the image logits by the attention mask to modulate them
            # modulated_img = img_logits * attn
            # Concatenate the original image logits and the modulated logits along the channel dimension
            fused_features = torch.cat([img_logits, event_out], dim=1)  # Shape: [B, 2*num_of_out_classes, H, W]
            # Refine the fused features with additional convolutional layers
            final_logits = self.fusion_conv(fused_features)
        else:
            final_logits = img_logits
        
        return final_logits


class DualDeepLabFusionModel(nn.Module):
    """
    A segmentation model that uses DeepLabV3 for both image and event branches.
    
    Both branches produce segmentation logits for a given number of classes.
    If the event input is single-channel, it is repeated to form a 3-channel input.
    The outputs of the two DeepLab networks are fused via concatenation followed by a fusion module.
    """
    def __init__(self, config):
        super(DualDeepLabFusionModel, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes', 21)
        self.pretrained = config.get('pretrained', False)
        
        # -----------------------------
        # Main Image Branch using DeepLabV3
        # -----------------------------
        if self.pretrained:
            print("Using pretrained DeepLabV3 model for image branch")
            self.deeplab_img = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab_img.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model for image branch from scratch")
            self.deeplab_img = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # -----------------------------
        # Event Branch also using DeepLabV3
        # -----------------------------
        # Note: Event input may have a different number of channels (e.g. 1),
    # but DeepLabV3 expects 3 channels. We will replicate the event channels if needed.
        print("Training DeepLabV3 model for event branch from scratch")
        self.deeplab_event = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # -----------------------------
        # Fusion Module: Fuse the outputs of both branches.
        # -----------------------------
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.num_of_out_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
        
    def forward(self, x, event):
        # Process image branch:
        # If image is grayscale, replicate channels.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        img_logits = self.deeplab_img(x)['out']  # [B, num_of_out_classes, H, W]
        
        # If event input is not 3-channel, replicate channels.
        if event.shape[1] != 3:
            event = event.repeat(1, 3, 1, 1)
        event_logits = self.deeplab_event(event)['out']  # [B, num_of_out_classes, H, W]
        
        # Fuse the two outputs:
        fused = torch.cat([img_logits, event_logits], dim=1)  # [B, 2*num_of_out_classes, H, W]
        final_logits = self.fusion_conv(fused)
        
        return final_logits
    

class DeepLabEventIntegratedWrapper3D(nn.Module):
    def __init__(self, config):
        super(DeepLabEventIntegratedWrapper3D, self).__init__()
        # Number of segmentation classes
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Event branch using 3D convolutions:
        # Expected event input shape: [B, event_channels, T, H, W]
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv3d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Fusion layers: after modulating the image branch with event attention,
        # we concatenate the original and modulated logits and refine them.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.num_of_out_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
    
    def forward(self, x, event=None):
        # Ensure image input has 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Get image segmentation logits from DeepLabV3
        output = self.deeplab(x)
        img_logits = output['out']  # [B, num_of_out_classes, H, W]
        
        if event is not None:
            # Process event data through the 3D event branch.
            # Expected event input shape: [B, event_channels, T, H, W]
            event_out = self.event_branch(event)  # Shape: [B, num_of_out_classes, T, H, W]
            # Collapse the temporal dimension (T) by averaging
            event_out = torch.mean(event_out, dim=2)  # Now [B, num_of_out_classes, H, W]
            # Use sigmoid to obtain an attention mask
            attn = torch.sigmoid(event_out)
            # Modulate the image logits with the attention mask
            modulated_img = img_logits * attn
            # Fuse the original and modulated logits along the channel dimension
            fused_features = torch.cat([img_logits, modulated_img], dim=1)  # [B, 2*num_of_out_classes, H, W]
            # Refine the fused features
            final_logits = self.fusion_conv(fused_features)
        else:
            final_logits = img_logits
        
        return final_logits

class GatedCrossModalTransformerFusionModel(nn.Module):
    """
    A segmentation model that uses DeepLabV3 for both image and event branches,
    and fuses their outputs via a novel gated cross-modal transformer fusion module.
    
    The fusion module first computes gating maps to weight each modality's features
    and then applies a lightweight transformer (using multi-head self-attention)
    to capture cross-modal interactions and long-range dependencies.
    """
    def __init__(self, config):
        super(GatedCrossModalTransformerFusionModel, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes', 21)
        self.pretrained = config.get('pretrained', False)
        
        # Image branch: DeepLabV3 for RGB images.
        if self.pretrained:
            print("Using pretrained DeepLabV3 for image branch")
            self.deeplab_img = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab_img.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 for image branch from scratch")
            self.deeplab_img = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Event branch: DeepLabV3 for event input.
        print("Training DeepLabV3 for event branch from scratch")
        self.deeplab_event = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Gating modules to learn per-pixel importance for each branch.
        self.gate_img = nn.Conv2d(self.num_of_out_classes, self.num_of_out_classes, kernel_size=1)
        self.gate_event = nn.Conv2d(self.num_of_out_classes, self.num_of_out_classes, kernel_size=1)
        
        # A lightweight transformer block for fusion.
        # We use nn.MultiheadAttention which operates on a sequence of tokens.
        # Here, each spatial location (per channel) is treated as a token.
        # self.fusion_transformer = nn.MultiheadAttention(embed_dim=self.num_of_out_classes, num_heads=4)
        
        # A small convolutional refinement block after the transformer.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.num_of_out_classes, self.num_of_out_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_of_out_classes),
            nn.ReLU(inplace=True)
        )
        # Project the initial logits to a higher dimension (e.g., 64)
        self.proj_in = nn.Conv2d(self.num_of_out_classes, 64, kernel_size=1)
        # Now create the transformer with embed_dim=64
        self.fusion_transformer = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        # And after the transformer, project back to self.num_of_out_classes
        self.proj_out = nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        
    def forward(self, x, event):
        # Process the image branch. If input is grayscale, repeat channels.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat_img = self.deeplab_img(x)['out']  # shape: [B, num_of_out_classes, H, W]
        
        # Process the event branch. Replicate channels if necessary.
        if event.shape[1] != 3:
            event = event.repeat(1, 3, 1, 1)
        feat_event = self.deeplab_event(event)['out']  # shape: [B, num_of_out_classes, H, W]
        
        # Compute gating maps for each modality.
        gate_img = torch.sigmoid(self.gate_img(feat_img))  # [B, num_of_out_classes, H, W]
        gate_event = torch.sigmoid(self.gate_event(feat_event))  # [B, num_of_out_classes, H, W]
        
        # Weight each branch's features.
        feat_img_weighted = feat_img * gate_img
        feat_event_weighted = feat_event * gate_event
        
        # Simple fusion: sum the weighted features.
        fused_feat = feat_img_weighted + feat_event_weighted  # [B, num_of_out_classes, H, W]
        # Project to a new dimension:
        proj_feat = self.proj_in(fused_feat)  # [B, 64, H, W]
        
        # Prepare for transformer: flatten spatial dimensions.
        B, C, H, W = proj_feat.shape
        tokens = proj_feat.view(B, C, H * W).permute(2, 0, 1)  # shape: [H*W, B, C]
        
        # Apply multi-head self-attention (treating all tokens equally).
        attn_output, _ = self.fusion_transformer(tokens, tokens, tokens)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        
        # Residual connection to preserve original fused features.
        fused_proj  = proj_feat  + attn_output
        
        # Refine with convolutional block.
        final_logits = self.proj_out(fused_proj)
    
        return final_logits

class EarlyFusionDeepLab(nn.Module):
    def __init__(self, config):
        super(EarlyFusionDeepLab, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        # Load the pretrained model
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            # Initialize the model with pretrained weights but default classifier
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            # Replace the classifier with a new one that has the correct number of classes
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            # When training from scratch, we can directly specify the number of classes
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Modify the first conv layer to accept 4 channels (RGB + Event)
        original_conv = self.deeplab.backbone.conv1
        self.deeplab.backbone.conv1 = nn.Conv2d(4, original_conv.out_channels,
                                                  kernel_size=original_conv.kernel_size,
                                                  stride=original_conv.stride,
                                                  padding=original_conv.padding,
                                                  bias=original_conv.bias is not None)
        # Initialize the new weights: copy RGB weights and initialize the event channel as the mean
        with torch.no_grad():
            self.deeplab.backbone.conv1.weight[:, :3] = original_conv.weight
            self.deeplab.backbone.conv1.weight[:, 3:] = original_conv.weight.mean(dim=1, keepdim=True)
        
        # Replace the classifier to match the number of classes
        self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
    
    def forward(self, x, event):
        # Normalize event to [0, 1] and concatenate with image
        if x.shape[1] == 3 and event.shape[1] == 1:
            event = event  # Ensure proper normalization if needed
            x = torch.cat([x, event], dim=1)
        output = self.deeplab(x)
        return output['out']

class DeepMidFusionDeepLab(nn.Module):
    def __init__(self, config):
        super(DeepMidFusionDeepLab, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        # Load DeepLabV3 backbone
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        backbone_output_channels = 2048  # From ResNet50
        
        # Deeper event branch: add more convolutional layers to capture richer features.
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv2d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, backbone_output_channels, kernel_size=1)
        )
        
        # Fusion module: concatenate the image and event features and refine.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(backbone_output_channels * 2, backbone_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head.
        self.classifier = DeepLabHead(backbone_output_channels, self.num_of_out_classes)
    
    def forward(self, x, event):
        # Image branch: obtain deep features from the backbone.
        image_features = self.deeplab.backbone(x)['out']  # Expected: [B, 2048, H, W]
        
        # Event branch: process the event data through the deeper branch.
        event_features = self.event_branch(event)  # Expected: [B, 2048, H_e, W_e]
        
        # Ensure spatial dimensions match.
        if image_features.shape[2:] != event_features.shape[2:]:
            event_features = F.adaptive_avg_pool2d(event_features, image_features.shape[2:])
        
        # Fuse features.
        fused_features = torch.cat([image_features, event_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Final segmentation prediction.
        logits = self.classifier(fused_features)
        return logits

class ShallowMidFusionDeepLab(nn.Module):
    def __init__(self, config):
        super(ShallowMidFusionDeepLab, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        # Load DeepLabV3 backbone
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        backbone_output_channels = 2048  # From ResNet50
        
        # Shallow event branch: fewer layers, mapping from 1 channel to 2048 channels.
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv2d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, backbone_output_channels, kernel_size=1)
        )
        
        # Fusion: Concatenate along the channel dimension then refine.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(backbone_output_channels * 2, backbone_output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(backbone_output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        self.classifier = DeepLabHead(backbone_output_channels, self.num_of_out_classes)
    
    def forward(self, x, event):
        # Image branch: extract deep features from the backbone.
        image_features = self.deeplab.backbone(x)['out']  # Expected: [B, 2048, H, W]
        
        # Event branch: process event data.
        event_features = self.event_branch(event)  # Expected: [B, 2048, H_e, W_e]
        
        # If needed, adjust spatial dimensions (here we assume they match; otherwise, use adaptive pooling)
        if image_features.shape[2:] != event_features.shape[2:]:
            event_features = F.adaptive_avg_pool2d(event_features, image_features.shape[2:])
        
        # Fuse the two feature maps.
        fused_features = torch.cat([image_features, event_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Final segmentation logits.
        logits = self.classifier(fused_features)
        return logits


class DeepLabTripleEventAutoencoderIntegratedWrapper(nn.Module):
    def __init__(self, config):
        super(DeepLabTripleEventAutoencoderIntegratedWrapper, self).__init__()
        self.num_of_out_classes = config.get('num_of_out_classes')
        self.pretrained = config.get('pretrained', False)
        
        # Initialize DeepLabV3 backbone for RGB images.
        if self.pretrained:
            print("Using pretrained DeepLabV3 model")
            self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            self.deeplab.classifier = DeepLabHead(2048, self.num_of_out_classes)
        else:
            print("Training DeepLabV3 model from scratch")
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=self.num_of_out_classes)
        
        # Event branch: processes event input.
        event_channels = config.get('event_channels', 1)
        self.event_branch = nn.Sequential(
            nn.Conv2d(event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Autoencoder branch: processes autoencoder features.
        autoencoder_channels = config.get('autoencoder_channels', 1)
        self.autoencoder_branch = nn.Sequential(
            nn.Conv2d(autoencoder_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Fusion block: Fuse the original image logits and the two modulated versions.
        # We concatenate along the channel dimension (3 * num_of_out_classes).
        fusion_in_channels = self.num_of_out_classes * 3
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
    
    def forward(self, x, event, autoencoder):
        # Process RGB image. If grayscale, replicate channels.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        output = self.deeplab(x)
        img_logits = output['out']  # Shape: [B, num_of_out_classes, H, W]
        
        # Process event data if provided.
        if event is not None:
            event_out = self.event_branch(event)  # [B, num_of_out_classes, H, W]
            event_attn = torch.sigmoid(event_out)
            modulated_event = img_logits * event_attn
        else:
            raise ValueError("Event input is required.")
        
        # Process autoencoder features if provided.
        if autoencoder is not None:
            auto_out = self.autoencoder_branch(autoencoder)  # [B, num_of_out_classes, H, W]
            auto_attn = torch.sigmoid(auto_out)
            modulated_auto = img_logits * auto_attn
        else:
            raise ValueError("Autoencoder features are required.")
        
        # Concatenate the original logits, event-modulated, and autoencoder-modulated logits.
        fused_features = torch.cat([img_logits, modulated_event, modulated_auto], dim=1)  # [B, 3*num_of_out_classes, H, W]
        
        # Refine fused features to produce final segmentation logits.
        final_logits = self.fusion_conv(fused_features)
        return final_logits