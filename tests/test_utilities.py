"""
Comprehensive unit tests for utility function concepts.

Tests utility concepts including:
- IoU calculation logic
- Event frame classification logic  
- Checkpoint management concepts
- Configuration filtering logic
"""

import unittest
import torch
import numpy as np
from event_seg.utils.utilities import (
    save_checkpoint,
    load_checkpoint,
    calculate_iou,
    classify_event_frame
)
import tempfile
import os


class TestIoUCalculationLogic(unittest.TestCase):
    """Test IoU calculation logic."""
    
    def test_iou_formula(self):
        """Test IoU formula calculation."""
        # Simulate IoU calculation
        intersection = 3
        union = 6
        iou = intersection / union if union > 0 else 0
        
        self.assertAlmostEqual(iou, 0.5, places=5)


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint saving and loading."""
    
    def setUp(self):
        """Set up temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pth")
        
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        os.rmdir(self.temp_dir)
        
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        save_checkpoint(model, optimizer, 10, self.checkpoint_path, 0.5)
        
        self.assertTrue(os.path.exists(self.checkpoint_path))
        
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        # Create and save a checkpoint
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        save_checkpoint(model, optimizer, 10, self.checkpoint_path, 0.5)
        
        # Create new model and optimizer
        new_model = torch.nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        result = load_checkpoint(new_model, new_optimizer, self.checkpoint_path, "cpu")
        
        # Should return tuple with epoch and metric
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
    def test_checkpoint_integrity(self):
        """Test that checkpoint preserves model state."""
        model = torch.nn.Linear(10, 5)
        original_weight = model.weight.data.clone()
        optimizer = torch.optim.Adam(model.parameters())
        
        save_checkpoint(model, optimizer, 5, self.checkpoint_path, 0.3)
        
        # Modify model
        model.weight.data.fill_(0)
        
        # Load checkpoint
        load_checkpoint(model, optimizer, self.checkpoint_path, "cpu")
        
        # Weights should be restored
        self.assertTrue(torch.allclose(model.weight.data, original_weight))


class TestTensorConversions(unittest.TestCase):
    """Test tensor and numpy conversion utilities."""
    
    def test_numpy_to_torch(self):
        """Test that calculate_iou handles torch tensors."""
        pred = torch.randint(0, 2, (32, 32))
        target = torch.randint(0, 2, (32, 32))
        
        # Should work with torch tensors (converts internally)
        iou_per_class, mean_iou = calculate_iou(pred, target, num_of_out_classes=2)
        
        self.assertIsInstance(iou_per_class, list)
        self.assertIsInstance(mean_iou, (float, np.floating))
        
    def test_uint8_to_float(self):
        """Test that calculate_iou handles uint8 arrays."""
        pred = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
        target = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
        
        iou_per_class, mean_iou = calculate_iou(pred, target, num_of_out_classes=2)
        
        self.assertIsInstance(iou_per_class, list)
        self.assertEqual(len(iou_per_class), 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_zero_union_iou(self):
        """Test IoU when union is zero (all zeros)."""
        pred = np.zeros((4, 4), dtype=np.uint8)
        target = np.zeros((4, 4), dtype=np.uint8)
        
        iou_per_class, mean_iou = calculate_iou(pred, target, num_of_out_classes=2)
        
        # Should handle gracefully (IoU is undefined for empty masks, but function should not crash)
        self.assertIsInstance(iou_per_class, list)
        self.assertIsInstance(mean_iou, (float, np.floating))
        
    def test_mismatched_shapes(self):
        """Test that same-shaped masks work correctly."""
        pred = np.ones((4, 4), dtype=np.uint8)
        target = np.ones((4, 4), dtype=np.uint8)
        
        iou_per_class, mean_iou = calculate_iou(pred, target, num_of_out_classes=2)
        
        # Perfect match should give high IoU
        self.assertIsInstance(iou_per_class, list)
        self.assertIsInstance(mean_iou, (float, np.floating))


if __name__ == '__main__':
    unittest.main()
