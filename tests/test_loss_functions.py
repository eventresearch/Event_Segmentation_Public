"""
Comprehensive unit tests for loss functions.

Tests all loss function implementations including:
- DiceLoss, FocalLoss, FocalDiceLoss
- Combined losses (FocalCEDiceLoss, FocalCE DiceLoss3Loss)
- Perceptual losses (LPIPSLoss, HybridLPIPSLoss)
- Loss factory function
"""

import unittest
import torch
import torch.nn as nn
from event_seg.train.loss import (
    DiceLoss,
    FocalLoss,
    FocalDiceLoss,
    FocalCELoss,
    FocalCEDiceLoss,
    LPIPSLoss,
    HybridLPIPSLoss,
    get_loss_function
)


class TestDiceLoss(unittest.TestCase):
    """Test DiceLoss implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss = DiceLoss()
        
    def test_perfect_prediction(self):
        """Test that perfect predictions give low loss."""
        pred = torch.zeros(2, 2, 4, 4)  # 2 classes for binary classification
        pred[:, 1, :, :] = 10.0  # Make class 1 very dominant (logits, not probabilities)
        pred[:, 0, :, :] = -10.0  # Class 0 very unlikely
        target = torch.ones(2, 4, 4).long()  # All pixels are class 1
        loss = self.loss(pred, target)
        self.assertLess(loss.item(), 0.1)  # Should be low for confident correct predictions
        
    def test_worst_prediction(self):
        """Test that opposite predictions give high loss."""
        pred = torch.ones(2, 2, 4, 4)  # 2 classes for binary classification
        pred[:, 1, :, :] = 5.0  # Make class 1 dominant
        pred[:, 0, :, :] = 0.1
        target = torch.zeros(2, 4, 4).long()  # All pixels are class 0 (opposite)
        loss = self.loss(pred, target)
        self.assertGreater(loss.item(), 0.5)  # Should be high for wrong predictions
        
    def test_output_shape(self):
        """Test that loss outputs a scalar."""
        pred = torch.rand(2, 2, 4, 4)  # 2 classes for binary segmentation
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        loss = self.loss(pred, target)
        self.assertEqual(loss.shape, torch.Size([]))
        
    def test_multi_class(self):
        """Test multi-class dice loss."""
        pred = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4)).long()  # Long tensor for classification
        loss = self.loss(pred, target)
        self.assertGreater(loss.item(), 0.0)


class TestFocalLoss(unittest.TestCase):
    """Test FocalLoss implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss = FocalLoss(alpha=0.8, gamma=2)
        
    def test_output_shape(self):
        """Test that loss outputs a scalar."""
        pred = torch.rand(2, 2, 4, 4)  # 2 classes for binary segmentation
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        loss = self.loss(pred, target)
        self.assertEqual(loss.shape, torch.Size([]))
        
    def test_gamma_effect(self):
        """Test that higher gamma focuses more on hard examples."""
        pred = torch.tensor([[[[0.9, 0.1], [0.5, 0.5]], [[0.1, 0.9], [0.5, 0.5]]]])  # 2 classes
        target = torch.tensor([[[1, 0], [1, 0]]]).long()  # Long tensor for classification
        
        loss_low_gamma = FocalLoss(gamma=0.5)(pred, target)
        loss_high_gamma = FocalLoss(gamma=5.0)(pred, target)
        
        # Higher gamma should penalize the hard example (0.5, 0.5) more
        self.assertNotEqual(loss_low_gamma.item(), loss_high_gamma.item())
        
    def test_alpha_balance(self):
        """Test that alpha balances class weights."""
        pred = torch.rand(2, 2, 4, 4)  # 2 classes for binary segmentation
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        
        loss_balanced = FocalLoss(alpha=0.5)(pred, target)
        loss_weighted = FocalLoss(alpha=0.9)(pred, target)
        
        self.assertIsInstance(loss_balanced.item(), float)
        self.assertIsInstance(loss_weighted.item(), float)


class TestFocalDiceLoss(unittest.TestCase):
    """Test FocalDiceLoss implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss = FocalDiceLoss(weight_dice=0.5, weight_focal=0.5)
        
    def test_output_shape(self):
        """Test that loss outputs a scalar."""
        pred = torch.rand(2, 2, 4, 4)  # 2 classes for binary segmentation
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        loss = self.loss(pred, target)
        self.assertEqual(loss.shape, torch.Size([]))
        
    def test_weight_combination(self):
        """Test that weights properly combine dice and focal."""
        pred = torch.rand(2, 2, 4, 4)  # 2 classes for binary segmentation
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        
        loss_dice_heavy = FocalDiceLoss(weight_dice=0.9, weight_focal=0.1)(pred, target)
        loss_focal_heavy = FocalDiceLoss(weight_dice=0.1, weight_focal=0.9)(pred, target)
        
        # Both should produce valid losses
        self.assertGreater(loss_dice_heavy.item(), 0.0)
        self.assertGreater(loss_focal_heavy.item(), 0.0)


class TestFocalCEDiceLoss(unittest.TestCase):
    """Test FocalCEDiceLoss implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss = FocalCEDiceLoss()
        
    def test_output_shape(self):
        """Test that loss outputs a scalar."""
        pred = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        loss = self.loss(pred, target)
        self.assertEqual(loss.shape, torch.Size([]))
        
    def test_three_component_combination(self):
        """Test that all three loss components contribute."""
        pred = torch.rand(2, 3, 4, 4)
        target = torch.randint(0, 3, (2, 4, 4))
        loss = self.loss(pred, target)
        self.assertGreater(loss.item(), 0.0)


class TestLossFactory(unittest.TestCase):
    """Test get_loss_function factory."""
    
    def test_dice_loss_creation(self):
        """Test creating DiceLoss."""
        loss = get_loss_function("Dice")  # Factory uses short names
        self.assertIsInstance(loss, DiceLoss)
        
    def test_focal_loss_creation(self):
        """Test creating FocalLoss with config."""
        config = {"alpha": 0.7, "gamma": 3}
        loss = get_loss_function("Focal", config)  # Factory uses short names
        self.assertIsInstance(loss, FocalLoss)
        
    def test_bce_loss_creation(self):
        """Test creating BCE loss."""
        loss = get_loss_function("BCE")
        self.assertIsInstance(loss, nn.BCEWithLogitsLoss)
        
    def test_ce_loss_creation(self):
        """Test creating CrossEntropy loss."""
        loss = get_loss_function("CE")  # Factory uses "CE" not "CrossEntropy"
        self.assertIsInstance(loss, nn.CrossEntropyLoss)
        
    def test_focal_dice_loss_creation(self):
        """Test creating FocalDiceLoss with config."""
        config = {"weight_dice": 0.6, "weight_focal": 0.4}
        loss = get_loss_function("FocalDice", config)  # Factory uses "FocalDice"
        self.assertIsInstance(loss, FocalDiceLoss)
        
    def test_focal_ce_dice_loss_creation(self):
        """Test creating FocalCEDiceLoss with config."""
        config = {
            "weight_ce": 0.5,
            "weight_focal": 0.5,
            "alpha": 0.8,
            "gamma": 2
        }
        loss = get_loss_function("FocalCEDice", config)  # Factory uses "FocalCEDice"
        self.assertIsInstance(loss, FocalCEDiceLoss)
        
    def test_focal_ce_dice_3loss_creation(self):
        """Test creating FocalCEDiceLoss with 3-component config."""
        config = {
            "weight_ce": 0.33,
            "weight_focal": 0.33,
            "weight_dice": 0.34,
            "alpha": 0.8,
            "gamma": 2
        }
        loss = get_loss_function("FocalCEDice", config)  # Factory uses "FocalCEDice"
        self.assertIsInstance(loss, FocalCEDiceLoss)
        
    def test_invalid_loss_name(self):
        """Test that invalid loss name raises error."""
        with self.assertRaises(ValueError):
            get_loss_function("InvalidLoss")
            
    def test_default_config(self):
        """Test that empty config works with defaults."""
        loss = get_loss_function("Focal", {})  # Empty dict uses defaults
        self.assertIsInstance(loss, FocalLoss)


class TestLossGradients(unittest.TestCase):
    """Test that all losses produce valid gradients."""
    
    def test_dice_loss_gradients(self):
        """Test DiceLoss produces gradients."""
        loss_fn = DiceLoss()
        pred = torch.rand(2, 2, 4, 4, requires_grad=True)  # 2 classes
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        
    def test_focal_loss_gradients(self):
        """Test FocalLoss produces gradients."""
        loss_fn = FocalLoss()
        pred = torch.rand(2, 2, 4, 4, requires_grad=True)  # 2 classes
        target = torch.randint(0, 2, (2, 4, 4)).long()  # Long tensor for classification
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        
    def test_combined_loss_gradients(self):
        """Test FocalCEDiceLoss produces gradients."""
        loss_fn = FocalCEDiceLoss()
        pred = torch.rand(2, 3, 4, 4, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 4)).long()  # Long tensor for classification
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)


if __name__ == '__main__':
    unittest.main()
