"""
Loss functions for semantic segmentation training.

This module provides various loss functions optimized for segmentation tasks:
- Cross-Entropy (CE): Standard classification loss
- Binary Cross-Entropy (BCE): For binary segmentation
- Dice Loss: Optimizes IoU directly
- Focal Loss: Handles class imbalance
- Hybrid losses: Combinations of multiple loss functions
- LPIPS Loss: Perceptual loss for temporal consistency

The get_loss_function factory allows easy selection and configuration of losses
via the config system.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # Import LPIPS
import segmentation_models_pytorch as smp

def get_loss_function(loss_type: str, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Factory function to initialize loss functions based on loss type.
    
    This function implements a registry pattern for loss function selection,
    supporting various loss functions optimized for segmentation tasks.
    
    Supported Loss Functions:
        - CE: Cross-Entropy loss (standard for multi-class segmentation)
        - BCE: Binary Cross-Entropy (for edge detection, binary segmentation)
        - Dice: Dice loss (directly optimizes IoU metric)
        - Focal: Focal loss (handles class imbalance by down-weighting easy examples)
        - FocalDice: Weighted combination of Focal and Dice losses
        - FocalCE: Weighted combination of Focal and Cross-Entropy
        - FocalCEDice: Triple combination for balanced optimization
        - LPIPS: Perceptual loss using pretrained VGG/AlexNet features
        - HybridLPIPS: Combines LPIPS with another loss for temporal consistency
    
    Args:
        loss_type: Name of the loss function (e.g., "Dice", "Focal", "FocalCE")
        config: Optional configuration dictionary containing loss-specific parameters:
            - alpha (float): Focal loss alpha parameter (default: 0.8)
            - gamma (float): Focal loss gamma parameter (default: 2)
            - weight_dice (float): Weight for Dice component in hybrid losses
            - weight_focal (float): Weight for Focal component in hybrid losses
            - weight_ce (float): Weight for CE component in hybrid losses
            - other_loss (str): Secondary loss type for HybridLPIPS (default: "BCE")
            - other_loss_weight (float): Weight for secondary loss in HybridLPIPS
            - lpips_weight (float): Weight for LPIPS component
            
    Returns:
        Initialized PyTorch loss function (nn.Module)
        
    Raises:
        ValueError: If loss_type is not supported
        
    Example:
        >>> # Simple Dice loss
        >>> criterion = get_loss_function("Dice")
        
        >>> # Focal loss with custom parameters
        >>> config = {'alpha': 0.75, 'gamma': 2.5}
        >>> criterion = get_loss_function("Focal", config)
        
        >>> # Hybrid loss with custom weights
        >>> config = {'weight_ce': 0.4, 'weight_focal': 0.3, 'weight_dice': 0.3}
        >>> criterion = get_loss_function("FocalCEDice", config)
        
    Note:
        - Focal loss is effective for highly imbalanced datasets
        - Dice loss directly optimizes the IoU metric
        - Hybrid losses combine benefits of multiple loss functions
        - LPIPS is useful for temporal consistency in video/event sequences
        
        To add a new loss function:
        1. Implement the loss class inheriting from nn.Module
        2. Add elif branch in this factory function
        3. Document the loss and its parameters
    """


    if loss_type == "CE":
        return nn.CrossEntropyLoss()
    elif loss_type == "BCE":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "Dice":
        return DiceLoss()
    elif loss_type == "Jaccard":
        return smp.losses.JaccardLoss(mode="multiclass")
    elif loss_type == "Focal":
        return FocalLoss(alpha=config.get("alpha", 0.8), gamma=config.get("gamma", 2))
    elif loss_type == "FocalDice":
        return FocalDiceLoss(weight_dice=config.get("weight_dice", 0.5), weight_focal=config.get("weight_focal", 0.5), alpha=config.get("alpha", 0.8), gamma=config.get("gamma", 2))
    elif loss_type == "FocalCE":
        return FocalCELoss(
            weight_ce=config.get("weight_ce", 0.5),
            weight_focal=config.get("weight_focal", 0.5),
            alpha=config.get("alpha", 0.8),
            gamma=config.get("gamma", 2)
        )
    elif loss_type == "FocalCEDice":
        return FocalCEDiceLoss(
            weight_ce=config.get("weight_ce", 0.33),
            weight_focal=config.get("weight_focal", 0.33),
            weight_dice=config.get("weight_dice", 0.34),
            alpha=config.get("alpha", 0.8),
            gamma=config.get("gamma", 2)
        )
    elif loss_type == "LPIPS":
        return LPIPSLoss(net_type="alex")
    elif loss_type == "HybridLPIPS":
        hybrid_other_loss = config.get("other_loss", "BCE")
        other_loss_weight = config.get("other_loss_weight", 0.5)
        lpips_weight = config.get("lpips_weight", 0.5)
        return HybridLPIPSLoss(net_type="alex", weight_lpips=lpips_weight, weight_other_loss=other_loss_weight, other_loss=hybrid_other_loss)  # Default weights
    else:
        raise ValueError("Invalid loss type. Choose from 'Dice', 'Focal', 'HybridLPIPS'.")


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation tasks.
    
    Dice loss directly optimizes the Dice coefficient (F1-score), which is
    equivalent to IoU for binary tasks. It's particularly effective for
    imbalanced datasets where small objects matter.
    
    Formula: Loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    
    Attributes:
        smooth: Small value to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss between predictions and targets.
        
        Args:
            outputs: Predicted logits with shape (B, C, H, W) where:
                - B: batch size
                - C: number of classes
                - H, W: height and width
            targets: Ground truth class indices with shape (B, H, W)

        Returns:
            Scalar Dice loss value (averaged over batch and classes)
            
        Note:
            - Applies softmax internally to convert logits to probabilities
            - Converts targets to one-hot encoding for per-class computation
            - Returns 1 - Dice coefficient (lower is better)
        """
        # Apply softmax to the output logits to get class probabilities
        outputs = torch.softmax(outputs, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=outputs.size(1)).permute(0, 3, 1, 2).float()

        # Calculate the intersection and union
        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))  # Sum over height and width
        union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        # Calculate Dice score
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss
        dice_loss = 1 - dice_score.mean()  # Average over batch
        return dice_loss

        dice_loss = 1 - dice_score.mean()  # Average over batch
        return dice_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation.
    
    Focal loss reshapes standard cross-entropy loss to down-weight easy examples
    and focus training on hard negatives. This is especially useful for datasets
    with severe class imbalance (e.g., small objects on large backgrounds).
    
    Formula: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.
    
    Attributes:
        alpha: Balancing factor for positive/negative examples (default: 0.8)
        gamma: Focusing parameter (higher = more focus on hard examples, default: 2)
    
    Note:
        - gamma=0 reduces to standard cross-entropy loss
        - gamma=2 is the recommended default from the original paper
        - alpha helps balance class frequency (typically 0.25-0.75)
    """
    def __init__(self, alpha: float = 0.8, gamma: float = 2) -> None:
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class (0 < alpha < 1)
            gamma: Exponent for modulating loss focus (gamma >= 0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss between predictions and targets.
        
        Args:
            outputs: Predicted logits (before softmax) with shape (B, C, H, W)
            targets: Ground truth class indices with shape (B, H, W)

        Returns:
            Scalar focal loss value (averaged over all pixels)
            
        Note:
            - Computes cross-entropy loss per pixel first
            - Applies focal modulation: (1 - p_t)^gamma
            - Scales by alpha for class balance
        """
        # Compute Cross-Entropy Loss for each pixel
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')

        # Calculate the probability of the correct class (softmax)
        pt = torch.exp(-ce_loss)  # Probability for the true class

        # Compute the Focal Loss
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        # Average the focal loss
        focal_loss = focal_loss.mean()  # Mean over batch
        return focal_loss


class FocalDiceLoss(nn.Module):
    """
    Hybrid loss combining Focal Loss and Dice Loss.
    
    This loss function combines the benefits of both:
    - Focal Loss: Handles class imbalance by focusing on hard examples
    - Dice Loss: Directly optimizes IoU metric for segmentation
    
    Formula: Loss = w_dice * DiceLoss + w_focal * FocalLoss
    
    Use this when:
    - Dataset has significant class imbalance
    - Want to optimize both classification and segmentation quality
    - Need balance between pixel-wise accuracy and object-level metrics
    
    Attributes:
        dice_loss: DiceLoss instance
        focal_loss: FocalLoss instance
        weight_dice: Weight for Dice component (default: 0.5)
        weight_focal: Weight for Focal component (default: 0.5)
    """
    def __init__(
        self,
        weight_dice: float = 0.5,
        weight_focal: float = 0.5,
        alpha: float = 0.8,
        gamma: float = 2,
        smooth: float = 1e-6
    ) -> None:
        """
        Initialize FocalDice hybrid loss.
        
        Args:
            weight_dice: Weight for Dice loss component (typically 0.3-0.7)
            weight_focal: Weight for Focal loss component (typically 0.3-0.7)
            alpha: Focal loss alpha parameter (see FocalLoss)
            gamma: Focal loss gamma parameter (see FocalLoss)
            smooth: Dice loss smoothing factor
        """
        super(FocalDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted combination of Focal and Dice losses.
        
        Args:
            outputs: Predicted logits with shape (B, C, H, W)
            targets: Ground truth class indices with shape (B, H, W)

        Returns:
            Weighted sum of Dice and Focal losses
        """
        # Calculate the individual losses
        dice_loss = self.dice_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        
        # Combine the two losses
        combined_loss = self.weight_dice * dice_loss + self.weight_focal * focal_loss
        return combined_loss

class FocalCELoss(nn.Module):
    """
    Hybrid loss combining Focal Loss and Cross-Entropy Loss.
    
    Combines standard Cross-Entropy with Focal Loss for handling both
    standard classification and class imbalance simultaneously.
    
    Formula: Loss = w_ce * CrossEntropyLoss + w_focal * FocalLoss
    
    Use this when:
    - Need both standard gradient flow (CE) and hard example mining (Focal)
    - Want smoother training than pure Focal loss
    - Dataset has moderate class imbalance
    
    Attributes:
        ce_loss: Standard CrossEntropyLoss instance
        focal_loss: FocalLoss instance
        weight_ce: Weight for CE component (default: 0.5)
        weight_focal: Weight for Focal component (default: 0.5)
    """
    def __init__(
        self,
        weight_ce: float = 0.5,
        weight_focal: float = 0.5,
        alpha: float = 0.8,
        gamma: float = 2
    ) -> None:
        """
        Initialize FocalCE hybrid loss.
        
        Args:
            weight_ce: Weight for Cross-Entropy component
            weight_focal: Weight for Focal component
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super(FocalCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weight_ce = weight_ce
        self.weight_focal = weight_focal

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted combination of CE and Focal losses.
        
        Args:
            outputs: Predicted logits with shape (B, C, H, W)
            targets: Ground truth class indices with shape (B, H, W)

        Returns:
            Weighted sum of CE and Focal losses
        """
        ce = self.ce_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        return self.weight_ce * ce + self.weight_focal * focal


class FocalCEDiceLoss(nn.Module):
    """
    Triple hybrid loss combining Focal, Cross-Entropy, and Dice losses.
    
    This is the most comprehensive loss function, combining three complementary
    objectives for robust segmentation:
    - Cross-Entropy: Standard pixel-wise classification
    - Focal Loss: Hard example mining for class imbalance
    - Dice Loss: Direct IoU optimization
    
    Formula: Loss = w_ce * CE + w_focal * Focal + w_dice * Dice
    
    Use this when:
    - Need maximum robustness across various scenarios
    - Dataset has complex characteristics (imbalance + small objects)
    - Want to optimize both pixel accuracy and segmentation quality
    - Training is unstable with simpler loss functions
    
    Attributes:
        ce_loss: CrossEntropyLoss instance
        focal_loss: FocalLoss instance
        dice_loss: DiceLoss instance
        weight_ce: Weight for CE component (default: 0.33)
        weight_focal: Weight for Focal component (default: 0.33)
        weight_dice: Weight for Dice component (default: 0.34)
    """
    def __init__(
        self,
        weight_ce: float = 0.33,
        weight_focal: float = 0.33,
        weight_dice: float = 0.34,
        alpha: float = 0.8,
        gamma: float = 2,
        smooth: float = 1e-6
    ) -> None:
        """
        Initialize FocalCEDice triple hybrid loss.
        
        Args:
            weight_ce: Weight for Cross-Entropy component
            weight_focal: Weight for Focal component
            weight_dice: Weight for Dice component
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            smooth: Dice loss smoothing factor
            
        Note:
            Default weights sum to 1.0 for balanced contribution
        """
        super(FocalCEDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight_ce = weight_ce
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted combination of CE, Focal, and Dice losses.
        
        Args:
            outputs: Predicted logits with shape (B, C, H, W)
            targets: Ground truth class indices with shape (B, H, W)

        Returns:
            Weighted sum of all three loss components
        """
        ce = self.ce_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return self.weight_ce * ce + self.weight_focal * focal + self.weight_dice * dice

class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.
    
    LPIPS uses pretrained deep features (VGG/AlexNet) to compute perceptual
    similarity between images. Unlike pixel-wise losses, LPIPS captures
    high-level semantic and structural differences.
    
    Use this for:
    - Temporal consistency in event/video segmentation
    - Ensuring perceptually similar predictions across frames
    - Complementing pixel-wise losses for better visual quality
    
    Attributes:
        lpips_loss: Pretrained LPIPS model (VGG or AlexNet based)
        
    Note:
        - Requires 3-channel input (automatically replicates single-channel)
        - Expects normalized inputs in [-1, 1] range
        - GPU-accelerated when available
    """
    def __init__(self, net_type: str = 'alex') -> None:
        """
        Initialize LPIPS loss with pretrained network.
        
        Args:
            net_type: Backbone network type ('alex' or 'vgg')
                - 'alex': AlexNet-based (faster, slightly less accurate)
                - 'vgg': VGG-based (slower, more accurate)
        """
        super(LPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net_type).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate LPIPS perceptual loss between predictions and targets.
        
        Args:
            pred: Predicted outputs with shape (B, C, H, W)
                - If C=1, automatically replicated to 3 channels
                - Values should be in [0, 1] range
            target: Ground truth with shape (B, C, H, W)
                - If C=1, automatically replicated to 3 channels
                - Values should be in [0, 1] range

        Returns:
            Scalar LPIPS loss (averaged over batch)
            
        Note:
            Automatically handles:
            - Single-channel to 3-channel conversion
            - [0, 1] to [-1, 1] normalization for LPIPS
        """
        # Assume pred and target are continuous outputs in [0, 1].
        # If the output is single channel, replicate to 3 channels.
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] as required by LPIPS.
        pred_norm = pred * 2 - 1
        target_norm = target * 2 - 1
        
        loss_lpips = self.lpips_loss(pred_norm, target_norm).mean()
        return loss_lpips
    
class HybridLPIPSLoss(nn.Module):
    """
    Hybrid loss combining LPIPS with another segmentation loss.
    
    This loss combines perceptual similarity (LPIPS) with traditional
    segmentation losses (CE, BCE, Dice, etc.) for temporal consistency
    and segmentation accuracy.
    
    Formula: Loss = w_other * OtherLoss + w_lpips * LPIPS
    
    Use this for:
    - Event-based segmentation with temporal consistency requirements
    - Video segmentation where frames should look similar
    - Combining perceptual quality with pixel-wise accuracy
    
    Typical configurations:
    - HybridLPIPS + BCE: For binary edge/object segmentation
    - HybridLPIPS + Dice: For multi-class with IoU optimization
    - HybridLPIPS + FocalCE: For imbalanced temporal sequences
    
    Attributes:
        lpips_loss: Pretrained LPIPS model
        other_loss: Primary segmentation loss (CE, BCE, Dice, etc.)
        weight_lpips: Weight for LPIPS component (default: 0.5)
        weight_other_loss: Weight for segmentation component (default: 0.5)
    """
    def __init__(
        self,
        net_type: str = 'alex',
        weight_lpips: float = 0.5,
        weight_other_loss: float = 0.5,
        other_loss: Optional[str] = None
    ) -> None:
        """
        Initialize HybridLPIPS loss.
        
        Args:
            net_type: LPIPS backbone ('alex' or 'vgg')
            weight_lpips: Weight for perceptual loss component
            weight_other_loss: Weight for segmentation loss component
            other_loss: Name of segmentation loss (e.g., "BCE", "Dice", "FocalCE")
                Uses get_loss_function() to initialize
                
        Example:
            >>> # Binary segmentation with temporal consistency
            >>> loss = HybridLPIPSLoss(
            ...     net_type='alex',
            ...     weight_lpips=0.3,
            ...     weight_other_loss=0.7,
            ...     other_loss='BCE'
            ... )
        """
        super(HybridLPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net_type).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.other_loss = get_loss_function(other_loss, config=None)
        self.weight_lpips = weight_lpips
        self.weight_other_loss = weight_other_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate hybrid loss combining LPIPS and segmentation loss.
        
        Args:
            pred: Predicted outputs with shape (B, C, H, W)
            target: Ground truth with shape (B, H, W) or (B, C, H, W)
                - Shape depends on the other_loss requirements

        Returns:
            Weighted sum of segmentation loss and LPIPS loss
            
        Note:
            - Other loss is computed first on original predictions
            - LPIPS is computed on continuous outputs
            - Automatically handles channel conversion for LPIPS
        """
        other_loss = self.other_loss(pred, target)
        
        # For LPIPS, use the continuous outputs.
        # If output is single channel, replicate to 3 channels.
        lpips_pred = pred
        lpips_target = target
        if lpips_pred.shape[1] == 1:
            lpips_pred = lpips_pred.repeat(1, 3, 1, 1)
        if lpips_target.shape[1] == 1:
            lpips_target = lpips_target.repeat(1, 3, 1, 1)
        
        pred_norm = lpips_pred * 2 - 1
        target_norm = lpips_target * 2 - 1
        
        loss_lpips = self.lpips_loss(pred_norm, target_norm).mean()
        total_loss = self.weight_other_loss * other_loss + self.weight_lpips * loss_lpips
        return total_loss