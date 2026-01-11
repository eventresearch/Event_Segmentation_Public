"""
Comprehensive unit tests for model architectures.

Tests model instantiation, forward passes, and output shapes for:
- Base U-Net models
- SMP-based models (U-Net, Segformer)
- Dual encoder models
- Attention models
- Model factory function
"""

import unittest
import torch
import torch.nn as nn
from event_seg.models.model import get_model
from event_seg.models.base_model import BaseUNet
from event_seg.models.smp_unet import SMPUNet, SMPDualEncoderUNet
from event_seg.models.smp_segformer import SMPSegformer


class TestBaseUNet(unittest.TestCase):
    """Test BaseUNet model."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.1
        }
        
    def test_model_creation(self):
        """Test model instantiation."""
        model = BaseUNet(self.config)
        self.assertIsInstance(model, nn.Module)
        
    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        model = BaseUNet(self.config)
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        # Output should be [batch, classes, H, W]
        self.assertEqual(output.shape, (2, 1, 64, 64))
        
    def test_multi_class_output(self):
        """Test multi-class segmentation output."""
        config = self.config.copy()
        config['num_of_out_classes'] = 5
        model = BaseUNet(config)
        
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 5, 64, 64))
        
    def test_different_input_channels(self):
        """Test with different input channels."""
        config = self.config.copy()
        config['num_of_in_channels'] = 1
        model = BaseUNet(config)
        
        x = torch.randn(2, 1, 64, 64)
        output = model(x)
        
        self.assertEqual(output.shape[1], 1)
        
    def test_dropout_application(self):
        """Test that dropout is applied."""
        config = self.config.copy()
        config['dropout_rate'] = 0.5
        model = BaseUNet(config)
        
        # Model should have dropout layers
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        self.assertTrue(has_dropout)


class TestSMPUNet(unittest.TestCase):
    """Test SMPUNet model."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'encoder_name': 'resnet18',
            'encoder_weights': None,  # Don't download weights in tests
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
        }
        
    def test_model_creation(self):
        """Test model instantiation."""
        model = SMPUNet(self.config)
        self.assertIsInstance(model, nn.Module)
        
    def test_forward_pass(self):
        """Test forward pass."""
        model = SMPUNet(self.config)
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        
        # SMP models handle size automatically
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1)
        
    def test_different_encoders(self):
        """Test with different encoder backbones."""
        encoders = ['resnet18', 'resnet34']
        
        for encoder_name in encoders:
            config = self.config.copy()
            config['encoder_name'] = encoder_name
            model = SMPUNet(config)
            
            x = torch.randn(1, 3, 128, 128)
            output = model(x)
            
            self.assertEqual(output.shape[0], 1)


class TestSMPDualEncoderUNet(unittest.TestCase):
    """Test SMPDualEncoderUNet model."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'event_channels': 1,
            'num_of_out_classes': 1,
            'fusion_type': 'concat'
        }
        
    def test_model_creation(self):
        """Test model instantiation."""
        model = SMPDualEncoderUNet(self.config)
        self.assertIsInstance(model, nn.Module)
        
    def test_dual_input_forward(self):
        """Test forward pass with dual inputs."""
        model = SMPDualEncoderUNet(self.config)
        rgb = torch.randn(2, 3, 128, 128)
        event = torch.randn(2, 1, 128, 128)
        
        output = model(rgb, event)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1)
        
    def test_fusion_types(self):
        """Test different fusion strategies."""
        fusion_types = ['concat', 'add']  # Only test supported fusion types
        
        for fusion_type in fusion_types:
            config = self.config.copy()
            config['fusion_type'] = fusion_type
            model = SMPDualEncoderUNet(config)
            
            rgb = torch.randn(1, 3, 128, 128)
            event = torch.randn(1, 1, 128, 128)
            
            output = model(rgb, event)
            self.assertEqual(output.shape[0], 1)


class TestSMPSegformer(unittest.TestCase):
    """Test SMPSegformer model."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'encoder_name': 'mit_b0',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'num_of_out_classes': 11,
        }
        
    def test_model_creation(self):
        """Test model instantiation."""
        model = SMPSegformer(self.config)
        self.assertIsInstance(model, nn.Module)
        
    def test_forward_pass(self):
        """Test forward pass."""
        model = SMPSegformer(self.config)
        x = torch.randn(2, 3, 128, 128)
        output = model(x)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 11)


class TestModelFactory(unittest.TestCase):
    """Test get_model factory function."""
    
    def test_basic_unet_creation(self):
        """Test creating basic UNet through factory."""
        config = {
            'model_type': 'basic',
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.0
        }
        model = get_model(config)
        self.assertIsInstance(model, nn.Module)
        
    def test_smp_unet_creation(self):
        """Test creating SMPUNet through factory."""
        config = {
            'model_type': 'smp_unet',
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'num_of_out_classes': 1
        }
        model = get_model(config)
        self.assertIsInstance(model, SMPUNet)
        
    def test_smp_segformer_creation(self):
        """Test creating SMPSegformer through factory."""
        config = {
            'model_type': 'smp_segformer',
            'encoder_name': 'mit_b0',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'num_of_out_classes': 11
        }
        model = get_model(config)
        self.assertIsInstance(model, SMPSegformer)
        
    def test_smp_dual_encoder_creation(self):
        """Test creating dual encoder through factory."""
        config = {
            'model_type': 'smp_dual_encoder_unet',
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'event_channels': 1,
            'num_of_out_classes': 1,
            'fusion_type': 'concat'
        }
        model = get_model(config)
        self.assertIsInstance(model, SMPDualEncoderUNet)
        
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        config = {
            'model_type': 'invalid_model',
            'num_of_in_channels': 3,
            'num_of_out_classes': 1
        }
        with self.assertRaises(ValueError):
            get_model(config)


class TestModelGradients(unittest.TestCase):
    """Test that models produce valid gradients."""
    
    def test_base_unet_gradients(self):
        """Test BaseUNet produces gradients."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.0
        }
        model = BaseUNet(config)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        self.assertIsNotNone(x.grad)
        
    def test_smp_unet_gradients(self):
        """Test SMPUNet produces gradients."""
        config = {
            'encoder_name': 'resnet18',
            'encoder_weights': None,
            'num_of_in_channels': 3,
            'num_of_out_classes': 1
        }
        model = SMPUNet(config)
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestModelModes(unittest.TestCase):
    """Test model training/eval modes."""
    
    def test_training_mode(self):
        """Test model in training mode."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.5
        }
        model = BaseUNet(config)
        model.train()
        
        self.assertTrue(model.training)
        
    def test_eval_mode(self):
        """Test model in eval mode."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.5
        }
        model = BaseUNet(config)
        model.eval()
        
        self.assertFalse(model.training)
        
    def test_dropout_behavior(self):
        """Test dropout behavior in train vs eval mode."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.5
        }
        model = BaseUNet(config)
        x = torch.randn(2, 3, 64, 64)
        
        # Training mode - outputs should vary due to dropout
        model.train()
        out1 = model(x)
        out2 = model(x)
        
        # Eval mode - outputs should be deterministic
        model.eval()
        out3 = model(x)
        out4 = model(x)
        
        # Eval outputs should be identical
        self.assertTrue(torch.allclose(out3, out4))


class TestModelOutputRanges(unittest.TestCase):
    """Test model output value ranges."""
    
    def test_output_finite(self):
        """Test that model outputs are finite."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.0
        }
        model = BaseUNet(config)
        model.eval()
        
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        # No NaN or Inf values
        self.assertTrue(torch.isfinite(output).all())
        
    def test_output_reasonable_range(self):
        """Test that outputs are in reasonable range."""
        config = {
            'num_of_in_channels': 3,
            'num_of_out_classes': 1,
            'bilinear': True,
            'dropout_rate': 0.0
        }
        model = BaseUNet(config)
        model.eval()
        
        x = torch.randn(2, 3, 64, 64)
        output = model(x)
        
        # Outputs should be in reasonable range (not exploding)
        self.assertLess(output.abs().max().item(), 1e6)


if __name__ == '__main__':
    unittest.main()
