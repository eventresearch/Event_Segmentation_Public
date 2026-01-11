"""
Concrete dataset implementations for multi-modal semantic segmentation.

This module provides specialized dataset classes that combine different input modalities:
- RGB images
- Event camera data (1-channel classified or 3-channel RGB representation)
- Autoencoder features (with optional morphological operations)
- Edge maps (Canny, DoG, or LoG)

All classes inherit from BaseSegmentationDataset and implement the __getitem__ method
to return modality-specific combinations of inputs, ground truth, and metadata.

Common return patterns:
    - Standard: (img, mask, orig_size)
    - With events: (img, mask, event, orig_size)
    - With autoencoder: (img, mask, autoencoder_feature, orig_size)
    - Triple input: (img, mask, event, autoencoder_feature, orig_size)
    - Edge detection: (event, edge, orig_size)

All datasets handle missing ground truth by returning zero tensors as placeholders.
"""

from typing import Tuple, Optional
import torch
import numpy as np
import cv2

from .augmentation_utils import apply_augmentations
import torch
import numpy as np
import cv2

from .base_class import *

# Dataset Classes
class SegmentationDataset(BaseSegmentationDataset):
    """
    Standard RGB-only semantic segmentation dataset.
    
    Loads RGB images and their corresponding segmentation masks. This is the
    simplest dataset configuration for standard semantic segmentation tasks.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded to multiples of 32
            - mask (torch.Tensor): Segmentation mask [H, W] with class indices, or zero tensor if missing
            - orig_size (Tuple[int, int]): Original image dimensions (height, width) before padding
    
    Example:
        >>> dataset = SegmentationDataset(config={'image_dir': 'imgs/', 'mask_dir': 'masks/', ...})
        >>> img, mask, orig_size = dataset[0]
        >>> img.shape  # (3, 480, 640) - padded to 32
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)

        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        mask = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Apply augmentation (automatically handles conversions)
        augmented = apply_augmentations(img, mask=mask, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug = augmented['image'], augmented['mask']

        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, orig_size
    
class EventSegmentationDataset(BaseSegmentationDataset):
    """
    RGB + 1-channel event camera segmentation dataset.
    
    Combines RGB images with event camera data that has been classified into a 
    single-channel representation using the classify_event_frame() function. Event
    frames are stacked temporally across time_steps.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - classified_event (torch.Tensor): Classified event frames [time_steps, H, W], padded
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'event_dir': 'events/', 'time_steps': 5, ...}
        >>> dataset = EventSegmentationDataset(config)
        >>> img, mask, event, orig_size = dataset[0]
        >>> event.shape  # (5, 480, 640) - 5 time steps
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)

        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        mask = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load and classify event camera frame
        classified_event = self._load_event_frames(idx)

        # Apply augmentation (automatically handles conversions)
        augmented = apply_augmentations(img, mask=mask, event=classified_event, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug, event_aug_tensor = augmented['image'], augmented['mask'], augmented['event']

        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, event_aug_tensor, orig_size
    
class Event3ChannelSegmentationDataset(BaseSegmentationDataset):
    """
    RGB + 3-channel RGB event camera segmentation dataset.
    
    Combines RGB images with event camera data represented as 3-channel RGB images.
    Unlike EventSegmentationDataset which uses classified single-channel events, this
    uses the raw RGB representation of event data for models that work better with
    RGB-like inputs.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - event_frame (torch.Tensor): RGB event frame [3, H, W], normalized and padded
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'event_dir': 'events/', ...}
        >>> dataset = Event3ChannelSegmentationDataset(config)
        >>> img, mask, event_rgb, orig_size = dataset[0]
        >>> event_rgb.shape  # (3, 480, 640) - RGB format
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # # Load and preprocess RGB image
        img = self._load_image(idx)
        
        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        mask  = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load event frame as 3-channel image
        event_frame = self._load_event_frame_3channel(idx)
        
        # Apply augmentation (automatically handles conversions)
        augmented = apply_augmentations(img, mask=mask, event=event_frame, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug, event_aug = augmented['image'], augmented['mask'], augmented['event']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, event_aug, orig_size
    
class EventSegmentation3DDataset(BaseSegmentationDataset):
    """
    RGB + 3D event stack segmentation dataset for spatiotemporal models.
    
    Similar to EventSegmentationDataset but adds an extra channel dimension to the
    event stack for 3D CNNs or spatiotemporal models. Event frames are stacked as
    [1, time_steps, H, W] to match 3D convolution input format.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - event_stack (torch.Tensor): 3D event stack [1, time_steps, H, W] for 3D convs
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        The event stack dimension order [1, D, H, W] is designed for 3D convolutions
        where D is the temporal depth (time_steps).
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'event_dir': 'events/', 'time_steps': 10, ...}
        >>> dataset = EventSegmentation3DDataset(config)
        >>> img, mask, event_3d, orig_size = dataset[0]
        >>> event_3d.shape  # (1, 10, 480, 640) - ready for 3D conv
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)
        
        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        mask  = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load and classify event camera frame
        # Reorder dimensions: [D, H, W] -> [1, D, H, W] (add channel) -> [B, C, D, H, W]
        event_stack = self._load_event_frames(idx).unsqueeze(0)
        
        # Apply augmentation (automatically handles conversions)
        augmented = apply_augmentations(img, mask=mask, event=event_stack, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug, event_aug = augmented['image'], augmented['mask'], augmented['event']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, event_aug, orig_size

class EdgeEvent3DDataset(BaseSegmentationDataset):
    """
    3D event stack + edge map dataset for edge-based segmentation.
    
    Used for models that predict edge maps from event camera data. The 3D event
    stack provides temporal context while edge maps serve as the ground truth.
    Useful for boundary detection and edge-aware segmentation tasks.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - event_stack (torch.Tensor): 3D event stack [1, time_steps, H, W]
            - edge (torch.Tensor): Edge map [1, H, W] or zero tensor if missing
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        Edge maps are extracted using the method specified in config['edge_method']:
        'canny', 'dog' (Difference of Gaussians), or 'log' (Laplacian of Gaussian).
    
    Example:
        >>> config = {'event_dir': 'events/', 'mask_dir': 'masks/', 'edge_method': 'canny', ...}
        >>> dataset = EdgeEvent3DDataset(config)
        >>> events, edges, orig_size = dataset[0]
        >>> edges.shape  # (1, 480, 640) - binary edge map
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and classify event camera frame
        event_stack = self._load_event_frames(idx)
        
        # Load and preprocess segmentation edeges (handle missing GT)
        edge = None
        if self.mask_dir and idx < len(self.mask_files):
            edge = self._load_edge(idx)
        edge  = edge if edge is not None else torch.zeros_like(event_stack[0])  # Dummy tensor if no GT
        
        
        # event_stack is [T, H, W]
        # Reorder dimensions: [T, H, W] -> [1, T, H, W] (add channel) -> [B, C, D, H, W]
        event_stack_unsqueezed = event_stack.unsqueeze(0)
        
        # Augmentation
        # Create a dummy image for geometric reference since we don't have one
        dummy_img = torch.zeros((3, event_stack.shape[-2], event_stack.shape[-1]), dtype=torch.uint8)
        
        augmented = apply_augmentations(dummy_img, mask=None, event=event_stack, edge=edge, aug_prob=self.aug_prob, aug_params=self.aug_params)
        event_aug, edge_aug = augmented['event'], augmented['edge']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return event_aug, edge_aug, orig_size
    
class EdgeEventDataset(BaseSegmentationDataset):
    """
    Temporal event stack + edge map dataset for 2D edge prediction.
    
    Similar to EdgeEvent3DDataset but without the extra channel dimension, providing
    event frames as [time_steps, H, W] for 2D+time models or recurrent architectures.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - event (torch.Tensor): Temporal event stack [time_steps, H, W]
            - edge (torch.Tensor): Edge map [1, H, W] or zero tensor if missing
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Example:
        >>> config = {'event_dir': 'events/', 'mask_dir': 'masks/', 'time_steps': 5, ...}
        >>> dataset = EdgeEventDataset(config)
        >>> events, edges, orig_size = dataset[0]
        >>> events.shape  # (5, 480, 640) - temporal stack
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and classify event camera frame
        event = self._load_event_frames(idx)
        
        # Load and preprocess segmentation edges (handle missing GT)
        edge = None
        if self.mask_dir and idx < len(self.mask_files):
            edge = self._load_edge(idx)
        edge  = edge if edge is not None else torch.zeros_like(event)  # Dummy tensor if no GT
        
        # Augmentation
        dummy_img = torch.zeros((3, event.shape[-2], event.shape[-1]), dtype=torch.uint8)
        
        augmented = apply_augmentations(dummy_img, mask=None, event=event, edge=edge, aug_prob=self.aug_prob, aug_params=self.aug_params)
        event_aug, edge_aug = augmented['event'], augmented['edge']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return event_aug, edge_aug, orig_size

class EdgeEventDatasetDoubleChannel(BaseSegmentationDataset):
    """
    Event + edge dataset with optional morphological edge enhancement.
    
    Similar to EdgeEventDataset but applies optional morphological operations (dilation)
    to edge maps for thicker/more visible boundaries. The edge map is returned without
    the channel dimension [H, W] and cast to long for classification loss functions.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - event (torch.Tensor): Temporal event stack [time_steps, H, W]
            - edge (torch.Tensor): Edge map [H, W] as long tensor (no channel dim)
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        If config['apply_morphology'] is True, edges are dilated using a 3x3 kernel
        for config['morp_iterations'] iterations to create thicker boundaries.
    
    Example:
        >>> config = {'event_dir': 'events/', 'mask_dir': 'masks/', 
        ...           'apply_morphology': True, 'morp_iterations': 2, ...}
        >>> dataset = EdgeEventDatasetDoubleChannel(config)
        >>> events, edges, orig_size = dataset[0]
        >>> edges.dtype  # torch.int64 (long) for loss functions
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and classify event camera frame
        event = self._load_event_frames(idx)
            
        # Load and preprocess segmentation edges (handle missing GT)
        edge = None
        if self.mask_dir and idx < len(self.mask_files):
            edge = self._load_edge(idx)
        edge  = edge if edge is not None else torch.zeros_like(event)  # Dummy tensor if no GT
        if self.apply_morphology:
            # Create a square kernel for dilation.
            kernel = np.ones((3, 3), np.uint8)
            # Convert edge tensor to a NumPy array.
            # Assumes edge has shape [1, H, W]; squeeze the channel.
            edge_np = edge.squeeze(0).cpu().numpy().astype(np.uint8)
            # Apply dilation.
            edge_dilated_np = cv2.dilate(edge_np, kernel, iterations=self.morp_iterations)
            # Convert back to tensor (and add back the channel dimension if desired).
            edge = torch.tensor(edge_dilated_np, dtype=torch.long) #.unsqueeze(0)
        else:
            # If not applying morphology, simply cast the edge to long after squeezing.
            edge = edge.type(torch.long).squeeze(0)
        
        # Augmentation
        dummy_img = torch.zeros((3, event.shape[-2], event.shape[-1]), dtype=torch.uint8)
        
        augmented = apply_augmentations(dummy_img, mask=None, event=event, edge=edge, aug_prob=self.aug_prob, aug_params=self.aug_params)
        event_aug, edge_aug = augmented['event'], augmented['edge']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return event_aug, edge_aug, orig_size

class EdgeEvent3DDatasetDoubleChannel(BaseSegmentationDataset):
    """
    3D event stack + morphologically enhanced edge map dataset.
    
    Combines 3D event representation with optional morphological edge enhancement.
    The 3D event stack [1, time_steps, H, W] provides spatiotemporal context while
    edges can be thickened via dilation for better visibility.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - event_stack (torch.Tensor): 3D event stack [1, time_steps, H, W]
            - edge (torch.Tensor): Edge map [H, W] as long tensor (no channel dim)
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        If config['apply_morphology'] is True, edges are dilated using a 3x3 kernel
        for config['morp_iterations'] iterations. Edge tensor is squeezed to [H, W]
        and cast to long dtype for compatibility with classification losses.
    
    Example:
        >>> config = {'event_dir': 'events/', 'mask_dir': 'masks/', 'time_steps': 10,
        ...           'apply_morphology': True, 'morp_iterations': 1, ...}
        >>> dataset = EdgeEvent3DDatasetDoubleChannel(config)
        >>> events_3d, edges, orig_size = dataset[0]
        >>> events_3d.shape  # (1, 10, 480, 640)
        >>> edges.shape      # (480, 640) - no channel dimension
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and classify event camera frame
        event_stack = self._load_event_frames(idx)
            
        # Load and preprocess segmentation edges (handle missing GT)
        edge = None
        if self.mask_dir and idx < len(self.mask_files):
            edge = self._load_edge(idx)
        edge  = edge if edge is not None else torch.zeros_like(event_stack[0])  # Dummy tensor if no GT
        if self.apply_morphology:
            # Create a square kernel for dilation.
            kernel = np.ones((3, 3), np.uint8)
            # Convert edge tensor to a NumPy array.
            # Assumes edge has shape [1, H, W]; squeeze the channel.
            edge_np = edge.squeeze(0).cpu().numpy().astype(np.uint8)
            # Apply dilation.
            edge_dilated_np = cv2.dilate(edge_np, kernel, iterations=self.morp_iterations)
            # Convert back to tensor (and add back the channel dimension if desired).
            edge = torch.tensor(edge_dilated_np, dtype=torch.long) #.unsqueeze(0)
        else:
            # If not applying morphology, simply cast the edge to long after squeezing.
            edge = edge.type(torch.long).squeeze(0)
        
        # Augmentation
        # Edge handling: wait, morphology logic is already applied above
        dummy_img = torch.zeros((3, event_stack.shape[-2], event_stack.shape[-1]), dtype=torch.uint8)
        
        augmented = apply_augmentations(dummy_img, mask=None, event=event_stack, edge=edge, aug_prob=self.aug_prob, aug_params=self.aug_params)
        
        # Restore: [H, W, T] -> [T, H, W] -> [1, T, H, W]
        event_aug = augmented['event']
        edge_aug = augmented['edge']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return event_aug, edge_aug, orig_size
    
class AutoencoderSegmentationDataset(BaseSegmentationDataset):
    """
    RGB + autoencoder feature fusion dataset.
    
    Combines RGB images with pre-computed autoencoder features for dual-encoder
    architectures. Autoencoder features provide learned representations that can
    complement RGB information for improved segmentation performance.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - autoencoder_feature (torch.Tensor): Autoencoder feature map [1, H, W], padded
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        Autoencoder features are loaded from grayscale images, normalized to [0, 1],
        and optionally binarized based on config['convert_to_binary'] and
        config['binary_threshold'] settings.
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'mask_dir': 'masks/', 
        ...           'autoencoder_dir': 'features/', ...}
        >>> dataset = AutoencoderSegmentationDataset(config)
        >>> img, mask, ae_feat, orig_size = dataset[0]
        >>> ae_feat.shape  # (1, 480, 640) - single channel feature
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)
        
        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        
        mask  = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load and preprocess autoencoder feature
        autoencoder_feature = self._load_autoencoder_feature(idx)
        
        augmented = apply_augmentations(img, mask=mask, ae=autoencoder_feature, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug, ae_aug = augmented['image'], augmented['mask'], augmented['ae']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, ae_aug, orig_size
    
class AutoencoderSegmentationDatasetMorphology(BaseSegmentationDataset):
    """
    RGB + morphologically enhanced autoencoder feature dataset.
    
    Similar to AutoencoderSegmentationDataset but applies morphological operations
    to autoencoder features if config['apply_morphology'] is enabled. Uses the v2
    loading method which supports optional morphological transformations.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - autoencoder_feature (torch.Tensor): Enhanced autoencoder feature [1, H, W], padded
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        Uses _load_autoencoder_feature_v2() which applies morphological operations
        (dilation/erosion) based on config['apply_morphology'] and config['morp_iterations'].
        This can help smooth or enhance features before feeding to the model.
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'autoencoder_dir': 'features/',
        ...           'apply_morphology': True, 'morp_iterations': 2, ...}
        >>> dataset = AutoencoderSegmentationDatasetMorphology(config)
        >>> img, mask, ae_feat_enhanced, orig_size = dataset[0]
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)
        
        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        
        mask  = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load and preprocess autoencoder feature
        autoencoder_feature = self._load_autoencoder_feature_v2(idx)
        
        augmented = apply_augmentations(img, mask=mask, ae=autoencoder_feature, aug_prob=self.aug_prob, aug_params=self.aug_params)
        img_aug, mask_aug, ae_aug = augmented['image'], augmented['mask'], augmented['ae']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, ae_aug, orig_size
    
class TripleInputDataset(BaseSegmentationDataset):
    """
    RGB + event + autoencoder feature triple-input dataset.
    
    Combines three complementary modalities for maximum information fusion:
    1. RGB images - standard visual information
    2. Event camera data - temporal dynamics and motion
    3. Autoencoder features - learned representations
    
    Designed for advanced multi-encoder architectures that can leverage all three
    input streams for improved segmentation performance.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]: 
            A tuple containing:
            - img (torch.Tensor): RGB image [3, H, W], normalized and padded
            - mask (torch.Tensor): Segmentation mask [H, W] or zero tensor if missing
            - classified_event (torch.Tensor): Classified event frames [time_steps, H, W], padded
            - autoencoder_feature (torch.Tensor): Autoencoder feature map [1, H, W], padded
            - orig_size (Tuple[int, int]): Original dimensions (height, width) before padding
    
    Note:
        All three modalities are independently loaded and padded to multiples of 32
        to ensure compatibility with encoder-decoder architectures. This dataset is
        particularly useful for exploring multi-modal fusion strategies.
    
    Example:
        >>> config = {'image_dir': 'imgs/', 'event_dir': 'events/', 
        ...           'autoencoder_dir': 'features/', 'mask_dir': 'masks/', ...}
        >>> dataset = TripleInputDataset(config)
        >>> img, mask, events, ae_feat, orig_size = dataset[0]
        >>> print(img.shape, events.shape, ae_feat.shape)
        # (3, 480, 640), (5, 480, 640), (1, 480, 640)
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        # Load and preprocess RGB image
        img = self._load_image(idx)
        
        # Load and preprocess segmentation mask (handle missing GT)
        mask = None
        if self.mask_dir and idx < len(self.mask_files):
            mask = self._load_mask(idx)
        
        mask  = mask if mask is not None else torch.zeros_like(img[0])  # Dummy tensor if no GT

        # Load and preprocess autoencoder feature
        autoencoder_feature = self._load_autoencoder_feature(idx)
        
        # Load and classify event camera frame
        classified_event = self._load_event_frames(idx)
        
        augmented = apply_augmentations(img, mask=mask, event=classified_event, ae=autoencoder_feature, aug_prob=self.aug_prob, aug_params=self.aug_params)
        
        img_aug, mask_aug = augmented['image'], augmented['mask']
        event_aug, ae_aug = augmented['event'], augmented['ae']
        
        # Return original size for unpadding during inference
        orig_size = (self._orig_h, self._orig_w)
        return img_aug, mask_aug, event_aug, ae_aug, orig_size