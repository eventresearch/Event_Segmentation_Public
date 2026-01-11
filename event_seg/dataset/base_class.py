"""
Base dataset module for semantic segmentation tasks with multi-modal inputs.

This module provides the foundation for loading and preprocessing RGB images,
segmentation masks, event camera frames, and autoencoder features. All inputs
are automatically padded to multiples of 32 for compatibility with standard
encoder-decoder architectures.
"""

import os
from typing import List, Optional, Tuple, Dict, Any, Union
from torch.utils.data import Dataset
from utils.utilities import classify_event_frame, preprocess_image, preprocess_mask
import torch
import cv2
import numpy as np
from tqdm import tqdm

class BaseSegmentationDataset(Dataset):
    """
    Base dataset class for semantic segmentation with multi-modal inputs.
    
    Handles loading and preprocessing of RGB images, segmentation masks, event frames,
    and autoencoder features. Automatically pads all inputs to multiples of 32.
    
    Args:
        config: Configuration dictionary with the following keys:
            - image_dir (Optional[str]): Path to directory containing RGB images
            - mask_dir (Optional[str]): Path to directory containing segmentation masks
            - event_dir (Optional[str]): Path to directory containing event frames
            - autoencoder_dir (Optional[str]): Path to autoencoder output features
            - time_steps (int): Number of temporal frames for 3D/sequential data
            - edge_method (str): Edge detection method ('canny', 'dog', 'log')
            - num_of_out_classes (int): Number of output segmentation classes
            - num_of_mask_classes (Optional[int]): Number of unique mask color classes
            - convert_to_binary (bool): Whether to convert autoencoder features to binary
            - binary_threshold (int): Threshold value (0-255) for binarization
            - morp_iterations (int): Number of morphological dilation iterations
            - apply_morphology (bool): Whether to apply morphological operations
            - is_event_scapes (bool): Whether using EventScape dataset format
            
    Attributes:
        _orig_h (int): Original image height before padding
        _orig_w (int): Original image width before padding  
        _last_pad (Tuple[int, int]): Last applied padding (height, width)
        _last_shape (Tuple[int, int]): Last processed shape after padding
        image_files (List[str]): List of image filenames
        mask_files (List[str]): List of mask filenames
        event_files (List[str]): List of event frame filenames
        autoencoder_files (List[str]): List of autoencoder output filenames
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the base segmentation dataset.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.image_dir: Optional[str] = config.get('image_dir')
        self.mask_dir: Optional[str] = config.get('mask_dir')
        self.event_dir: Optional[str] = config.get('event_dir')
        self.autoencoder_dir: Optional[str] = config.get('autoencoder_dir')
        self.time_steps: int = config.get('time_steps')
        self.edge_method: str = config.get('edge_method')
        self.num_of_out_classes: int = config.get('num_of_out_classes')
        self.convert_to_binary: bool = config.get('convert_to_binary', False)
        self.binary_threshold: int = config.get('binary_threshold', 128)
        self.morp_iterations: int = config.get('morp_iterations', 1)
        self.apply_morphology: bool = config.get('apply_morphology', False)
        
        self.is_event_scapes: bool = config.get('is_event_scapes', False)

        # Check if directories exist and handle missing cases gracefully
        self.image_files: List[str] = self._load_files(self.image_dir, label="image_dir")
        self.mask_files: List[str] = self._load_files(self.mask_dir, label="mask_dir")
        self.event_files: List[str] = self._load_files(self.event_dir, label="event_dir")
        self.autoencoder_files: List[str] = self._load_files(self.autoencoder_dir, label="autoencoder_dir")
        
        self.mask_classes: int = config.get('num_of_mask_classes')
        if self.mask_classes is None:
            self.mask_classes = self._find_num_classes_from_dataset()
        
        # Augmentation Configuration
        # If 'augmentation' flag is False, effectively disable it by setting prob to 0.0
        # This allows easy toggling in config while keeping dataset logic clean.
        self.apply_aug: bool = config.get('apply_augmentation', False)
        base_prob: float = config.get('augmentation_probability')
        self.aug_prob: float = base_prob if self.apply_aug else 0.0
        
        # Specific Augmentation Parameters
        self.aug_params = {
            'flip_prob': config.get('aug_flip_prob', 0.5),
            'scale_limit': config.get('aug_scale_limit', 0.1),
            'rotate_limit': config.get('aug_rotate_limit', 15),
            'shift_limit': config.get('aug_shift_limit', 0.0625),
            'affine_prob': config.get('aug_affine_prob', 0.2),
            'rgb_prob': config.get('aug_rgb_prob', 0.2)
        }
        
        self._validate_file_counts()
        
    def _load_files(self, directory: Optional[str], label: str) -> List[str]:
        """
        Load filenames from a directory if it exists.
        
        Args:
            directory: Path to directory containing files
            label: Label for logging/debugging purposes
            
        Returns:
            Sorted list of filenames, or empty list if directory doesn't exist
        """
        if directory is None:
            return []

        if not os.path.exists(directory):
            print(f"Warning: Optional directory '{label}' is missing: {directory}")
            return []

        return sorted(os.listdir(directory))

    def _find_num_classes_from_dataset(self, sample_ratio: float = 0.1, max_files: int = 500) -> int:
        """
        Efficiently estimate the number of unique segmentation classes by sampling masks.
        
        Args:
            sample_ratio: Fraction of masks to sample (0.0 to 1.0)
            max_files: Maximum number of files to sample
            
        Returns:
            Estimated number of unique segmentation classes
        """
        unique_colors = set()
        
        total_files = len(self.mask_files)
        sample_size = min(max(int(total_files * sample_ratio), 1), max_files)  # Limit to `max_files`
        
        print(f"\n🔍 Scanning {sample_size}/{total_files} masks to estimate the number of segmentation classes...")

        sampled_mask_files = np.random.choice(self.mask_files, sample_size, replace=False)
        
        for mask_file in tqdm(sampled_mask_files, desc="Processing Sampled Masks", unit="mask", leave=True):
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                continue  # Skip corrupted files

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            unique_colors.update(map(tuple, np.unique(mask.reshape(-1, 3), axis=0)))

        num_of_out_classes = len(unique_colors)
        print(f"✅ Estimated {num_of_out_classes} unique classes from {sample_size} sampled masks.\n")
        
        return num_of_out_classes

    def _validate_file_counts(self) -> None:
        """
        Validate that file counts match expected relationships.
        
        Raises:
            ValueError: If file counts don't match expected relationships
        """
        # Validate image and mask file counts
        if self.image_files and self.mask_files:
            if len(self.image_files) != len(self.mask_files):
                raise ValueError(f"Mismatch in image and mask file counts: {len(self.image_files)} images vs {len(self.mask_files)} masks.")

        # Validate event files if event_dir exists and is not empty
        if self.event_files and self.image_files:
            expected_event_count = len(self.image_files) * self.time_steps
            if len(self.event_files) != expected_event_count:
                raise ValueError(
                    f"Event file count mismatch: {len(self.event_files)} event files vs expected {expected_event_count} "
                    f"(based on {len(self.image_files)} images and {self.time_steps} time steps)."
                )

        # Validate autoencoder files if autoencoder_dir exists and is not empty
        if self.autoencoder_files and self.image_files:
            if len(self.autoencoder_files) != len(self.image_files):
                raise ValueError(
                    f"Autoencoder file count mismatch: {len(self.autoencoder_files)} autoencoder outputs vs {len(self.image_files)} images."
                )

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples based on available file lists
        """
        if self.image_files not in [None, []]:        
           return len(self.image_files)
        elif self.mask_files not in [None, []]:
            return len(self.mask_files)
        elif self.event_files not in [None, []]:
            return len(self.event_files) // self.time_steps
        elif self.autoencoder_files not in [None, []]:
            return len(self.autoencoder_files)
        return 0  # Default case when no files are available

    def get_original_size(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the original image size before padding was applied.
        
        Returns:
            Tuple of (height, width) or (None, None) if not yet set
        """
        if hasattr(self, '_orig_h') and hasattr(self, '_orig_w'):
            return self._orig_h, self._orig_w
        return None, None

    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Subclass-specific return format (must be implemented by subclasses)
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        # Placeholder for specific dataset implementations
        raise NotImplementedError("This method should be implemented by subclasses.")


    def _pad_to_32(self, img: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Tuple[int, int]]:
        """
        Pad image/tensor to make dimensions multiples of 32.
        
        Required for encoder-decoder architectures that use downsampling/upsampling
        with factors of 2. Padding to 32 ensures compatibility with 5 levels of
        2x downsampling (2^5 = 32).
        
        Args:
            img: Input image as torch.Tensor [C, H, W] or numpy array [H, W, C]
            
        Returns:
            Tuple containing:
                - Padded image/tensor with dims divisible by 32
                - Tuple of (pad_height, pad_width) that was applied
        """
        # img: torch.Tensor [C, H, W] or numpy [H, W, C]
        import torch.nn.functional as F
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            h, w = img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h == 0 and pad_w == 0:
            return img, (0, 0)
        if isinstance(img, torch.Tensor):
            padded = F.pad(img, (0, pad_w, 0, pad_h))
        else:
            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        return padded, (pad_h, pad_w)

    def _load_image(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess an RGB image with padding.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Preprocessed and padded image tensor of shape [C, H', W']
            where H' and W' are padded to multiples of 32
        """
        path = os.path.join(self.image_dir, self.image_files[idx])
        img = preprocess_image(path)
        # Store original shape before padding
        if isinstance(img, torch.Tensor):
            self._orig_h, self._orig_w = img.shape[-2:]
        else:
            self._orig_h, self._orig_w = img.shape[:2]
        
        img, (pad_h, pad_w) = self._pad_to_32(img)
        self._last_pad = (pad_h, pad_w)
        self._last_shape = img.shape[-2:] if isinstance(img, torch.Tensor) else img.shape[:2]
        return img

    def _load_mask(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a segmentation mask with padding.
        
        Args:
            idx: Index of the mask to load
            
        Returns:
            Preprocessed and padded mask tensor of shape [H', W']
            where H' and W' are padded to multiples of 32
        """
        path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = preprocess_mask(path, self.mask_classes)
        mask, _ = self._pad_to_32(mask)
        return mask

    def _load_event_frames(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess 1-channel event frames as a temporal stack.
        
        Loads multiple consecutive event frames based on time_steps configuration,
        classifies them into binary action/background, and stacks them temporally.
        
        Args:
            idx: Base index for the sample (multiplied by time_steps for actual frame indices)
            
        Returns:
            Stacked and padded event frames of shape [time_steps, H', W']
            where H' and W' are padded to multiples of 32
            
        Raises:
            ValueError: If event_dir or time_steps is not defined
        """
        if self.event_dir is None or self.time_steps is None:
            raise ValueError("Event directory or time_steps is not defined. Ensure both are set for this dataset.")
        
        # EventScapes uses black (0) background, DSEC uses white (255)
        background_color = 0 if self.is_event_scapes else 255
        
        start_idx = idx * self.time_steps
        end_idx = start_idx + self.time_steps
        event_stack = []
        for i in range(start_idx, end_idx):
            event_path = os.path.join(self.event_dir, self.event_files[i])
            event_frame = classify_event_frame(event_path, expand_dims=False, background_color=background_color)
            event_stack.append(event_frame)
        
        event_stack_tensor = torch.stack(event_stack, dim=0)  # [time_steps, H, W]
        
        # Store original shape before any potential padding for event frames
        if not hasattr(self, '_orig_h') or not hasattr(self, '_orig_w'):
            self._orig_h, self._orig_w = event_stack_tensor.shape[-2:]
        
        # Apply padding to event frames as well (for autoencoders)
        event_stack_tensor, _ = self._pad_to_32(event_stack_tensor)
        
        return event_stack_tensor

    def _load_event_frame_3channel(self, idx: int) -> torch.Tensor:
        """
        Load a 3-channel RGB event frame and apply padding.
        
        For datasets where event frames are stored as RGB images rather than
        binary action/background classification.
        
        Args:
            idx: Index of the event frame to load
            
        Returns:
            Preprocessed and padded 3-channel event frame of shape [3, H', W']
            where H' and W' are padded to multiples of 32
            
        Raises:
            ValueError: If event_dir is not defined
        """
        if self.event_dir is None:
            raise ValueError("Event directory is not defined. Ensure it is set for this dataset.")
        
        event_path = os.path.join(self.event_dir, self.event_files[idx])
        event_frame = preprocess_image(event_path)
        
        # Store original shape before padding if not already set
        if not hasattr(self, '_orig_h') or not hasattr(self, '_orig_w'):
            self._orig_h, self._orig_w = event_frame.shape[-2:]
        
        # Apply padding to event frame to match RGB and mask dimensions
        event_frame, _ = self._pad_to_32(event_frame)
        
        return event_frame

    # def _load_autoencoder_feature(self, idx):
    #     if self.autoencoder_dir is None:
    #         raise ValueError("Autoencoder directory is not defined. Ensure it is set for this dataset.")
        
    #     autoencoder_path = os.path.join(self.autoencoder_dir, self.autoencoder_files[idx])
    #     autoencoder_feature = cv2.imread(autoencoder_path, cv2.IMREAD_GRAYSCALE)
    #     if autoencoder_feature is None:
    #         raise FileNotFoundError(f"Autoencoder output not found at path: {autoencoder_path}")
    #     autoencoder_feature = autoencoder_feature / 255.0  # Normalize
    #     autoencoder_feature = torch.tensor(autoencoder_feature, dtype=torch.float32)
    #     return autoencoder_feature.clone().detach().unsqueeze(0)  # Add channel dim, and to fix warning used clone and detach
    #     # return torch.tensor(autoencoder_feature, dtype=torch.float32).unsqueeze(0)  # Add channel dim
    @staticmethod
    def preprocess_autoencoder_feature(
        autoencoder_feature: Union[np.ndarray, torch.Tensor],
        convert_to_binary: bool = False,
        binary_threshold: int = 128
    ) -> torch.Tensor:
        """
        Optimized preprocessing for autoencoder output features.
        
        Normalizes features to [0, 1] range and optionally applies binary thresholding.
        
        Args:
            autoencoder_feature: Input feature map as numpy array or tensor
            convert_to_binary: Whether to threshold the feature to binary values
            binary_threshold: Threshold value in range 0-255 for binarization
            
        Returns:
            Processed autoencoder feature tensor of shape [1, H, W], normalized to [0, 1]
        """
        if isinstance(autoencoder_feature, np.ndarray):
            autoencoder_feature = torch.tensor(autoencoder_feature, dtype=torch.float32)

        # Fast check if already in [0,1] range (avoid expensive .max())
        scale_factor = 1.0 if autoencoder_feature.max() > 1 else (255.0 / binary_threshold)

        # Normalize & apply threshold in a single step
        autoencoder_feature = autoencoder_feature / (scale_factor * 255.0)
        
        if convert_to_binary:
            autoencoder_feature = (autoencoder_feature > (binary_threshold / 255.0)).float()

        return autoencoder_feature.unsqueeze(0)  # Add channel dimension
    
    def _load_autoencoder_feature(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess autoencoder output feature with padding.
        
        Args:
            idx: Index of the autoencoder feature to load
            
        Returns:
            Preprocessed and padded autoencoder feature of shape [1, H', W']
            where H' and W' are padded to multiples of 32
            
        Raises:
            ValueError: If autoencoder_dir is not defined
            FileNotFoundError: If autoencoder output file doesn't exist
        """
        if self.autoencoder_dir is None:
            raise ValueError("Autoencoder directory is not defined. Ensure it is set for this dataset.")
        
        autoencoder_path = os.path.join(self.autoencoder_dir, self.autoencoder_files[idx])
        autoencoder_feature = cv2.imread(autoencoder_path, cv2.IMREAD_GRAYSCALE)
        if autoencoder_feature is None:
            raise FileNotFoundError(f"Autoencoder output not found at path: {autoencoder_path}")
        # Apply preprocessing function (now includes normalization)
        autoencoder_feature = self.preprocess_autoencoder_feature(
            autoencoder_feature,
            convert_to_binary=self.convert_to_binary,  # Config-based flag
            binary_threshold=self.binary_threshold   # Config-based threshold
        )
        # Apply padding to autoencoder feature to match RGB and mask dimensions
        autoencoder_feature, _ = self._pad_to_32(autoencoder_feature)
        return autoencoder_feature


    def apply_morphology_func(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        kernel_size: int = 3,
        iterations: int = 1
    ) -> torch.Tensor:
        """
        Apply morphological dilation to thicken mask boundaries.
        
        Used to expand autoencoder features or edge maps for better visibility
        and coverage in segmentation tasks.
        
        Args:
            mask: Single-channel mask tensor or numpy array
            kernel_size: Size of the square structuring element (default: 3)
            iterations: Number of times dilation is applied (default: 1)
            
        Returns:
            Dilated mask tensor of shape [1, H, W]
        """
        # Convert the mask to a NumPy array (removing extra channel dimensions if any)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = mask

        # Create a square structuring element (kernel)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Apply dilation to thicken boundaries
        dilated_np = cv2.dilate(mask_np, kernel, iterations=iterations)
        # Convert back to a tensor and add the channel dimension
        dilated_tensor = torch.tensor(dilated_np, dtype=mask.dtype).unsqueeze(0)
        return dilated_tensor

    def _load_autoencoder_feature_v2(self, idx: int, kernel_size: int = 3) -> torch.Tensor:
        """
        Load autoencoder feature with morphological processing and padding.
        
        Extends _load_autoencoder_feature by applying morphological dilation
        to thicken boundaries before padding.
        
        Args:
            idx: Index of the autoencoder feature to load
            kernel_size: Size of the kernel for morphology (default: 3)
        
        Returns:
            Processed and padded autoencoder feature of shape [1, H', W']
            with morphological dilation applied
            
        Raises:
            ValueError: If autoencoder_dir is not defined
            FileNotFoundError: If autoencoder output file doesn't exist
        """
        if self.autoencoder_dir is None:
            raise ValueError("Autoencoder directory is not defined. Ensure it is set for this dataset.")
        
        autoencoder_path = os.path.join(self.autoencoder_dir, self.autoencoder_files[idx])
        autoencoder_feature = cv2.imread(autoencoder_path, cv2.IMREAD_GRAYSCALE)
        if autoencoder_feature is None:
            raise FileNotFoundError(f"Autoencoder output not found at path: {autoencoder_path}")
        
        # Normalize the feature to the range [0, 1]
        autoencoder_feature = self.preprocess_autoencoder_feature(
            autoencoder_feature,
            convert_to_binary=self.convert_to_binary,  # Config-based flag
            binary_threshold=self.binary_threshold   # Config-based threshold
        )
        
        # Apply morphological dilation using the helper method added to BaseSegmentationDataset
        autoencoder_feature = self.apply_morphology_func(autoencoder_feature, kernel_size, iterations=self.morp_iterations)
        # Apply padding to autoencoder feature to match RGB and mask dimensions
        autoencoder_feature, _ = self._pad_to_32(autoencoder_feature)
        return autoencoder_feature


    def _load_edge(self, idx: int) -> torch.Tensor:
        """
        Extract edge map from segmentation mask using edge detection.
        
        Converts color mask to grayscale, applies specified edge detection method
        (Canny, DoG, or LoG), and returns normalized edge map with padding.
        
        Args:
            idx: Index of the mask to process for edge extraction
            
        Returns:
            Normalized and padded edge map tensor of shape [1, H', W']
            where values are in range [0, 1] and H', W' are padded to multiples of 32
            
        Raises:
            FileNotFoundError: If mask file doesn't exist
            ValueError: If edge_method is invalid
        """
        # Load mask directly for edge detection
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at path: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # Convert to grayscale for edge detection
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Scale mask to the range [0, 255] if needed
        if mask.max() <= 1:
            mask = (mask * 255).astype("uint8")
        elif mask.max() <= 255:
            mask = mask.astype("uint8")
        
        edges = None
        # Generate edge map based on the specified method
        if self.edge_method == "canny":
            edges = cv2.Canny(mask, 0, 0)
        elif self.edge_method == "dog":  # Difference of Gaussians
            blur1 = cv2.GaussianBlur(mask, (3, 3), 1.0)
            blur2 = cv2.GaussianBlur(mask, (3, 3), 2.0)
            edges = cv2.subtract(blur1, blur2)
        elif self.edge_method == "log":  # Laplacian of Gaussian
            blur = cv2.GaussianBlur(mask, (5, 5), 1.0)
            edges = cv2.Laplacian(blur, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)
        else:
            raise ValueError(f"Invalid edge detection method: {self.edge_method}. Use 'canny', 'dog', or 'log'.")

        # Normalize and convert to tensor
        edges_tensor = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0
        # Apply padding to edge maps as well (for autoencoders)
        edges_tensor, _ = self._pad_to_32(edges_tensor)
        return edges_tensor
        
    
    def _load_edge_v2(self, idx: int) -> torch.Tensor:
        """
        Legacy edge extraction method (older version).
        
        Loads preprocessed mask first, then applies edge detection. This is an older
        implementation kept for backward compatibility. Prefer using _load_edge() for new code.
        
        Args:
            idx: Index of the mask to process for edge extraction
            
        Returns:
            Normalized edge map tensor of shape [1, H, W] (NOT padded)
            
        Raises:
            ValueError: If edge_method is invalid
        """
        mask = self._load_mask(idx).cpu().numpy()  # Convert to NumPy for OpenCV operations

        # Scale mask to the range [0, 255]
        if 1 < mask.max() <= self.num_of_out_classes:
            mask = (mask * (255 / self.num_of_out_classes)).astype("uint8")
        elif mask.max() <= 1:
            mask = (mask * 255).astype("uint8")
        elif mask.max() <= 255:
            mask = mask.astype("uint8")

        # Generate edge map based on the specified method
        if self.edge_method == "canny":
            edges = cv2.Canny(mask, 0, 0)
        elif self.edge_method == "dog":
            blur1 = cv2.GaussianBlur(mask, (3, 3), 1.0)
            blur2 = cv2.GaussianBlur(mask, (3, 3), 2.0)
            edges = cv2.subtract(blur1, blur2)
        elif self.edge_method == "log":
            blur = cv2.GaussianBlur(mask, (5, 5), 1.0)
            edges = cv2.Laplacian(blur, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)
        else:
            raise ValueError(f"Invalid edge detection method: {self.edge_method}. Use 'canny', 'dog', or 'log'.")

        # Normalize and convert to tensor
        edges = torch.tensor(edges, dtype=torch.float32).unsqueeze(0) / 255.0
        return edges