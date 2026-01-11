"""
Unit tests to validate type annotations across the codebase.

Tests use runtime type checking to ensure that functions return the expected types
and accept the correct parameter types.
"""

import sys
import os
import unittest
from typing import get_type_hints
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'event_seg')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.base_class import BaseSegmentationDataset
from dataset import classes as dataset_classes


class TestBaseSegmentationDatasetTypes(unittest.TestCase):
    """Test type annotations for BaseSegmentationDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'image_dir': None,
            'mask_dir': None,
            'event_dir': None,
            'autoencoder_dir': None,
            'time_steps': 1,
            'edge_method': 'canny',
            'num_of_out_classes': 11,
            'num_of_mask_classes': 11,
            'convert_to_binary': False,
            'binary_threshold': 128,
            'morp_iterations': 1,
            'apply_morphology': False,
            'is_event_scapes': False
        }
    
    def test_init_annotations(self):
        """Test __init__ has proper type annotations."""
        hints = get_type_hints(BaseSegmentationDataset.__init__)
        self.assertIn('config', hints)
        self.assertIn('return', hints)
        # Return should be None for __init__
        self.assertEqual(hints['return'], type(None))
    
    def test_len_return_type(self):
        """Test __len__ returns int."""
        hints = get_type_hints(BaseSegmentationDataset.__len__)
        self.assertEqual(hints['return'], int)
    
    def test_get_original_size_annotations(self):
        """Test get_original_size has proper return type."""
        hints = get_type_hints(BaseSegmentationDataset.get_original_size)
        self.assertIn('return', hints)
        # Should return Tuple[Optional[int], Optional[int]]
    
    def test_pad_to_32_annotations(self):
        """Test _pad_to_32 has proper type annotations."""
        hints = get_type_hints(BaseSegmentationDataset._pad_to_32)
        self.assertIn('img', hints)
        self.assertIn('return', hints)
    
    def test_load_files_annotations(self):
        """Test _load_files has proper type annotations."""
        hints = get_type_hints(BaseSegmentationDataset._load_files)
        self.assertIn('directory', hints)
        self.assertIn('label', hints)
        self.assertIn('return', hints)
    
    def test_validate_file_counts_annotations(self):
        """Test _validate_file_counts has proper return type."""
        hints = get_type_hints(BaseSegmentationDataset._validate_file_counts)
        self.assertEqual(hints['return'], type(None))
    
    def test_preprocess_autoencoder_feature_annotations(self):
        """Test preprocess_autoencoder_feature has proper annotations."""
        hints = get_type_hints(BaseSegmentationDataset.preprocess_autoencoder_feature)
        self.assertIn('autoencoder_feature', hints)
        self.assertIn('convert_to_binary', hints)
        self.assertIn('binary_threshold', hints)
        self.assertIn('return', hints)
        self.assertEqual(hints['return'], torch.Tensor)
    
    def test_apply_morphology_func_annotations(self):
        """Test apply_morphology_func has proper annotations."""
        hints = get_type_hints(BaseSegmentationDataset.apply_morphology_func)
        self.assertIn('mask', hints)
        self.assertIn('kernel_size', hints)
        self.assertIn('iterations', hints)
        self.assertIn('return', hints)
        self.assertEqual(hints['return'], torch.Tensor)


class TestBaseSegmentationDatasetRuntimeTypes(unittest.TestCase):
    """Test runtime type behavior for BaseSegmentationDataset methods."""
    
    def test_pad_to_32_with_tensor(self):
        """Test _pad_to_32 works with torch.Tensor."""
        dataset = BaseSegmentationDataset({
            'num_of_out_classes': 11,
            'num_of_mask_classes': 11,
            'time_steps': 1,
            'edge_method': 'canny'
        })
        
        # Create a tensor that needs padding
        img = torch.randn(3, 100, 100)  # [C, H, W]
        padded, (pad_h, pad_w) = dataset._pad_to_32(img)
        
        # Check return types
        self.assertIsInstance(padded, torch.Tensor)
        self.assertIsInstance(pad_h, int)
        self.assertIsInstance(pad_w, int)
        
        # Check dimensions are multiples of 32
        self.assertEqual(padded.shape[-2] % 32, 0)
        self.assertEqual(padded.shape[-1] % 32, 0)
    
    def test_pad_to_32_with_numpy(self):
        """Test _pad_to_32 works with numpy array."""
        dataset = BaseSegmentationDataset({
            'num_of_out_classes': 11,
            'num_of_mask_classes': 11,
            'time_steps': 1,
            'edge_method': 'canny'
        })
        
        # Create a numpy array that needs padding
        img = np.random.randn(100, 100, 3)  # [H, W, C]
        padded, (pad_h, pad_w) = dataset._pad_to_32(img)
        
        # Check return types
        self.assertIsInstance(padded, np.ndarray)
        self.assertIsInstance(pad_h, int)
        self.assertIsInstance(pad_w, int)
        
        # Check dimensions are multiples of 32
        self.assertEqual(padded.shape[0] % 32, 0)
        self.assertEqual(padded.shape[1] % 32, 0)
    
    def test_preprocess_autoencoder_feature_return_type(self):
        """Test preprocess_autoencoder_feature returns torch.Tensor."""
        # Create a fake autoencoder feature
        feature = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        result = BaseSegmentationDataset.preprocess_autoencoder_feature(
            feature, 
            convert_to_binary=False, 
            binary_threshold=128
        )
        
        # Check return type
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # Should have channel dimension
        self.assertTrue(result.max() <= 1.0)  # Should be normalized
        self.assertTrue(result.min() >= 0.0)
    
    def test_apply_morphology_func_return_type(self):
        """Test apply_morphology_func returns torch.Tensor."""
        dataset = BaseSegmentationDataset({
            'num_of_out_classes': 11,
            'num_of_mask_classes': 11,
            'time_steps': 1,
            'edge_method': 'canny',
            'morp_iterations': 1
        })
        
        # Create a fake mask
        mask = torch.randint(0, 2, (1, 100, 100), dtype=torch.float32)
        
        result = dataset.apply_morphology_func(mask, kernel_size=3, iterations=1)
        
        # Check return type
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # Should have channel dimension


class TestTypeAnnotationCompleteness(unittest.TestCase):
    """Test that all public methods have type annotations."""
    
    def test_all_public_methods_have_annotations(self):
        """Verify all public methods in BaseSegmentationDataset have type hints."""
        methods_to_check = [
            '__init__',
            '__len__',
            '__getitem__',
            'get_original_size',
            '_load_files',
            '_find_num_classes_from_dataset',
            '_validate_file_counts',
            '_pad_to_32',
            '_load_image',
            '_load_mask',
            '_load_event_frames',
            '_load_event_frame_3channel',
            'preprocess_autoencoder_feature',
            '_load_autoencoder_feature',
            '_load_autoencoder_feature_v2',
            'apply_morphology_func',
            '_load_edge',
            '_load_edge_v2'
        ]
        
        for method_name in methods_to_check:
            method = getattr(BaseSegmentationDataset, method_name)
            hints = get_type_hints(method)
            
            # All methods should have at least a return type annotation
            self.assertIn('return', hints, 
                         f"Method {method_name} missing return type annotation")


class TestDatasetClassesTypeAnnotations(unittest.TestCase):
    """Test type annotations for all dataset classes in classes.py."""
    
    def test_segmentation_dataset_has_annotations(self):
        """Test SegmentationDataset.__getitem__ has type annotations."""
        hints = get_type_hints(dataset_classes.SegmentationDataset.__getitem__)
        self.assertIn('return', hints, "SegmentationDataset.__getitem__ missing return type")
        self.assertIn('idx', hints, "SegmentationDataset.__getitem__ missing idx parameter type")
    
    def test_event_segmentation_dataset_has_annotations(self):
        """Test EventSegmentationDataset.__getitem__ has type annotations."""
        hints = get_type_hints(dataset_classes.EventSegmentationDataset.__getitem__)
        self.assertIn('return', hints, "EventSegmentationDataset.__getitem__ missing return type")
        self.assertIn('idx', hints)
    
    def test_event_3channel_dataset_has_annotations(self):
        """Test Event3ChannelSegmentationDataset.__getitem__ has type annotations."""
        hints = get_type_hints(dataset_classes.Event3ChannelSegmentationDataset.__getitem__)
        self.assertIn('return', hints)
        self.assertIn('idx', hints)
    
    def test_event_3d_dataset_has_annotations(self):
        """Test EventSegmentation3DDataset.__getitem__ has type annotations."""
        hints = get_type_hints(dataset_classes.EventSegmentation3DDataset.__getitem__)
        self.assertIn('return', hints)
        self.assertIn('idx', hints)
    
    def test_edge_event_datasets_have_annotations(self):
        """Test all EdgeEvent dataset variants have type annotations."""
        edge_datasets = [
            dataset_classes.EdgeEvent3DDataset,
            dataset_classes.EdgeEventDataset,
            dataset_classes.EdgeEventDatasetDoubleChannel,
            dataset_classes.EdgeEvent3DDatasetDoubleChannel,
        ]
        
        for dataset_class in edge_datasets:
            with self.subTest(dataset=dataset_class.__name__):
                hints = get_type_hints(dataset_class.__getitem__)
                self.assertIn('return', hints, f"{dataset_class.__name__}.__getitem__ missing return type")
                self.assertIn('idx', hints)
    
    def test_autoencoder_datasets_have_annotations(self):
        """Test autoencoder dataset variants have type annotations."""
        ae_datasets = [
            dataset_classes.AutoencoderSegmentationDataset,
            dataset_classes.AutoencoderSegmentationDatasetMorphology,
        ]
        
        for dataset_class in ae_datasets:
            with self.subTest(dataset=dataset_class.__name__):
                hints = get_type_hints(dataset_class.__getitem__)
                self.assertIn('return', hints)
                self.assertIn('idx', hints)
    
    def test_triple_input_dataset_has_annotations(self):
        """Test TripleInputDataset.__getitem__ has type annotations."""
        hints = get_type_hints(dataset_classes.TripleInputDataset.__getitem__)
        self.assertIn('return', hints)
        self.assertIn('idx', hints)
    
    def test_all_dataset_classes_documented(self):
        """Verify all dataset classes have docstrings."""
        dataset_class_names = [
            'SegmentationDataset',
            'EventSegmentationDataset',
            'Event3ChannelSegmentationDataset',
            'EventSegmentation3DDataset',
            'EdgeEvent3DDataset',
            'EdgeEventDataset',
            'EdgeEventDatasetDoubleChannel',
            'EdgeEvent3DDatasetDoubleChannel',
            'AutoencoderSegmentationDataset',
            'AutoencoderSegmentationDatasetMorphology',
            'TripleInputDataset',
        ]
        
        for class_name in dataset_class_names:
            with self.subTest(class_name=class_name):
                dataset_class = getattr(dataset_classes, class_name)
                self.assertIsNotNone(dataset_class.__doc__, 
                                    f"{class_name} missing docstring")
                self.assertGreater(len(dataset_class.__doc__), 50,
                                  f"{class_name} docstring too short")


if __name__ == '__main__':
    unittest.main()
