
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import initialization as smp_init

class SMPWrapper(nn.Module):
    """
    Generic wrapper for ANY Segmentation Models PyTorch (SMP) architecture.
    
    This class allows instantiation of any SMP model (Unet, DeepLabV3, DeepLabV3+, 
    PSPNet, FPN, etc.) by specifying the 'architecture' name in the config.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        self.arch_name = config.get('architecture', 'Unet')
        self.encoder_name = config.get('encoder_name', 'resnet34')
        self.encoder_weights = config.get('encoder_weights', 'imagenet')
        if self.encoder_weights == "None":
            self.encoder_weights = None
        self.in_channels = config.get('num_of_in_channels', 3)
        self.classes = config.get('num_of_out_classes', 1)
        self.activation = config.get('activation', None)
        
        # Get the model class from smp
        if not hasattr(smp, self.arch_name):
            raise ValueError(f"SMP architecture '{self.arch_name}' not found. Available: Unet, DeepLabV3, DeepLabV3Plus, PSPNet, FPN, etc.")
        
        model_class = getattr(smp, self.arch_name)
        
        self.extra_params = config.get('extra_params', {})
        
        # Initialize the model
        self.model = model_class(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=self.activation,
            **self.extra_params
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SMPEventWrapper(nn.Module):
    """
    Generic wrapper for ANY SMP architecture with an added Event Branch (Late Fusion).
    
    This preserves the powerful pretrained backbone for RGB while creating a 
    lightweight, effective fusion for event data.
    
    Fusion Strategy (Late Fusion / Sigmoid Attention):
    1. RGB Image -> SMP Model -> RGB Logits
    2. Event Frames -> Simple CNN -> Event Features
    3. Attention Mask = Sigmoid(Event Features)
    4. Modulated RGB = RGB Logits * Attention Mask
    5. Fused = Concat(RGB Logits, Modulated RGB)
    6. Output = Fusion Conv(Fused)
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        self.num_of_out_classes = config.get('num_of_out_classes', 1)
        self.event_channels = config.get('event_channels', 1)
        
        # Initialize the SMP model (RGB Backbone)
        self.smp_model = SMPWrapper(config)
        
        print(f"initialized SMPEventWrapper with {config.get('architecture')} and encoder {config.get('encoder_name')}")

        # Lightweight Event Branch
        self.event_branch = nn.Sequential(
            nn.Conv2d(self.event_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.num_of_out_classes, kernel_size=1)
        )
        
        # Fusion Layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2 * self.num_of_out_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_of_out_classes, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor, event: Optional[torch.Tensor] = None) -> torch.Tensor:
        # RGB Forward Pass
        rgb_logits = self.smp_model(x)
        
        if event is None:            
            raise ValueError("Event cannot be None")

            
        # Event Forward Pass
        event_features = self.event_branch(event)
        
        # Attention Mechanism
        attention_mask = torch.sigmoid(event_features)
        
        # Modulate RGB logits with Event Attention
        modulated_rgb = rgb_logits * attention_mask
        
        # Concatenate: [Original RGB, Modulated RGB]
        # We assume event_features provides "where to look" (attention), 
        # so we keep the original RGB predictions but enriched.
        # Alternatively, we could concat [rgb_logits, event_features] directly.
        # But per user request "like we do in deeplab event", we follow that pattern.
        # Existing DeepLabEventIntegratedWrapper uses: cat([img_logits, modulated_img])
        
        fused = torch.cat([rgb_logits, modulated_rgb], dim=1)
        
        # Final Refinement
        output = self.fusion_conv(fused)
        
        return output
