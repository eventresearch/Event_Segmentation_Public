import albumentations as A
import numpy as np
import cv2
import torch
import random
import functools

# --- Configuration ---

# Define which keys should use Nearest Neighbor interpolation (masks, discrete features)
# All others default to Linear interpolation (images, continuous features)
MASK_KEYS = {'mask', 'edge', 'event'} 

# Define all possible keys that might be passed to the augmentation pipeline
# This superset allows us to build the pipeline once and reuse it.
ALL_POSSIBLE_KEYS = {'image', 'mask', 'event', 'edge', 'ae'}


# --- Helpers for Explicit Formatting ---

def to_numpy_image(tensor):
    """
    Explicitly converts a Tensor (C, H, W) to Numpy (H, W, C).
    Args:
        tensor (torch.Tensor): Shape (C, H, W)
    Returns:
        numpy.ndarray: Shape (H, W, C)
    """
    if not isinstance(tensor, torch.Tensor):
        # Assume already numpy (H, W, C)
        return tensor
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)

def to_numpy_mask(tensor):
    """
    Explicitly converts a Tensor (H, W) or (1, H, W) to Numpy (H, W).
    Args:
        tensor (torch.Tensor): Shape (H, W) or (1, H, W)
    Returns:
        numpy.ndarray: Shape (H, W)
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    np_mask = tensor.detach().cpu().numpy()
    if np_mask.ndim == 3 and np_mask.shape[0] == 1:
        np_mask = np_mask[0] # Squeeze channel dim if present
    return np_mask

def from_numpy_image(np_img, template_tensor=None):
    """
    Explicitly converts Numpy (H, W, C) back to Tensor (C, H, W).
    """
    if np_img is None:
        return None
    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(np_img.transpose(2, 0, 1))
    
    # Restore Dtype/Device from template if available, else default to float32
    if template_tensor is not None:
        tensor = tensor.to(dtype=template_tensor.dtype, device=template_tensor.device)
    else:
        tensor = tensor.float()
    return tensor

def from_numpy_mask(np_mask, template_tensor=None):
    """
    Explicitly converts Numpy (H, W) back to Tensor (H, W).
    """
    if np_mask is None:
        return None
    # (H, W) -> (H, W)
    tensor = torch.from_numpy(np_mask)
    
    if template_tensor is not None:
        tensor = tensor.to(dtype=template_tensor.dtype, device=template_tensor.device)
    else:
        tensor = tensor.long()
    return tensor

def from_numpy_generic(np_arr, template_tensor=None, is_3d_input=False):
    """
    Generic restorer.
    If input was 3D (C, H, W), we expect (H, W, C) numpy -> convert back.
    If input was 2D (H, W), we expect (H, W) numpy -> keep.
    """
    if np_arr is None:
        return None
    
    if is_3d_input and np_arr.ndim == 3:
        # Restore (H,W,C) -> (C,H,W)
        tensor = torch.from_numpy(np_arr.transpose(2, 0, 1))
    else:
        # Keep as is (H,W)
        tensor = torch.from_numpy(np_arr)
        
    if template_tensor is not None:
        tensor = tensor.to(dtype=template_tensor.dtype, device=template_tensor.device)
    return tensor


# --- Global Pipeline Construction ---

@functools.lru_cache(maxsize=16)
def get_global_geometric_pipeline(
    flip_prob=0.5, 
    scale_limit=0.1, 
    rotate_limit=15, 
    shift_limit=0.0625,
    affine_prob=0.2
):
    """
    Creates the shared geometric pipeline ONCE (cached by arguments).
    Includes targets for all possible keys.
    """
    additional_targets = {}
    for key in ALL_POSSIBLE_KEYS:
        if key == 'image': continue
        # Decide interpolation mode
        target_type = 'mask' if key in MASK_KEYS else 'image'
        additional_targets[key] = target_type
        
    return A.Compose([
        A.HorizontalFlip(p=flip_prob),
        # Affine combines Shift, Scale, Rotate
        # scale: (1-limit, 1+limit)
        # translate: (-limit, +limit)
        # rotate: (-limit, +limit)
        A.Affine(
            scale=(1.0 - scale_limit, 1.0 + scale_limit), 
            translate_percent=(-shift_limit, shift_limit), 
            rotate=(-rotate_limit, rotate_limit), 
            p=affine_prob, # Probability of applying Affine transform itself
            fill=0, 
            border_mode=cv2.BORDER_CONSTANT
        ),
    ], additional_targets=additional_targets)

def get_rgb_pixel_pipeline(p=0.2):
    return A.Compose([
        A.RandomBrightnessContrast(p=p),
    ])

def apply_augmentations(img, mask=None, aug_prob=1.0, aug_params=None, **kwargs):
    """
    Apply synchronized augmentations with configurable parameters.
    
    Args:
        img (torch.Tensor): (C, H, W)
        mask (torch.Tensor, optional): (H, W)
        aug_prob (float): Probability of applying augmentation. Default 1.0.
        aug_params (dict, optional): Dictionary of augmentation parameters:
            - flip_prob (float): Probability of horizontal flip (default 0.5)
            - scale_limit (float): Scale factor range (default 0.1)
            - rotate_limit (int): Rotation range in degrees (default 15)
            - shift_limit (float): Translation limit (default 0.0625)
            - affine_prob (float): Probability of applying affine transform (default 0.2)
            - rgb_prob (float): Probability of applying RGB pixel transforms (default 0.2)
        **kwargs: Additional tensors (C, H, W) or (H, W).
    """
    
    # 0. Probability Gate
    if aug_prob < 1.0 and random.random() > aug_prob:
        return {'image': img, 'mask': mask, **kwargs}
        
    # Set default params if not provided
    if aug_params is None:
        aug_params = {}
        
    flip_prob = aug_params.get('flip_prob', 0.5)
    scale_limit = aug_params.get('scale_limit', 0.1)
    rotate_limit = aug_params.get('rotate_limit', 15)
    shift_limit = aug_params.get('shift_limit', 0.0625)
    affine_prob = aug_params.get('affine_prob', 0.2)
    rgb_prob = aug_params.get('rgb_prob', 0.2)

    # 1. Validation & Conversion
    # Checks for invalid keys
    for k in kwargs:
        if k not in ALL_POSSIBLE_KEYS:
            raise ValueError(f"Key '{k}' is not in ALL_POSSIBLE_KEYS {ALL_POSSIBLE_KEYS}. "
                             "Please update augmentation_utils.py if you need new modalities.")

    # Image: (C, H, W) -> (H, W, C)
    data = {}
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for 'img', got {type(img)}")
    data['image'] = to_numpy_image(img)
    
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for 'mask', got {type(mask)}")
        # Mask: (H, W) -> (H, W)
        data['mask'] = to_numpy_mask(mask)
        
    # Handle kwargs
    # We need to track which inputs were 3D to restore them correctly
    is_3d_map = {} 
    
    for key, val in kwargs.items():
        if val is None:
            continue
            
        if not isinstance(val, torch.Tensor):
             raise TypeError(f"Expected torch.Tensor for key '{key}', got {type(val)}")
            
        if val.ndim == 3:
            # Assume (C, H, W) -> (H, W, C)
            data[key] = to_numpy_image(val)
            is_3d_map[key] = True
        else:
            # Assume (H, W) or simple numpy
            data[key] = to_numpy_mask(val)
            is_3d_map[key] = False
            
    # 2. Apply Geometric Augmentation (Global Pipeline)
    # Allows generic keys even if not present in this specific call
    pipeline = get_global_geometric_pipeline(
        flip_prob=flip_prob,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        shift_limit=shift_limit,
        affine_prob=affine_prob
    )
    res = pipeline(**data)
    
    # 3. Apply Pixel Augmentation (RGB Only)
    pixel_aug = get_rgb_pixel_pipeline(p=rgb_prob)
    res['image'] = pixel_aug(image=res['image'])['image']
    
    # 4. Restore Outputs (Explicit Restoration)
    final_results = {}
    
    # Image: (H, W, C) -> (C, H, W)
    final_results['image'] = from_numpy_image(res['image'], template_tensor=img)
    
    if mask is not None:
         # Mask: (H, W) -> (H, W)
        final_results['mask'] = from_numpy_mask(res['mask'], template_tensor=mask)
        
    for key, val in kwargs.items():
        if val is None:
            final_results[key] = None
            continue
            
        final_results[key] = from_numpy_generic(
            res[key], 
            template_tensor=val, 
            is_3d_input=is_3d_map[key]
        )
            
    return final_results

