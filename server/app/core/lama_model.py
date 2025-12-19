"""
Standalone LaMa (Large Mask Inpainting) model implementation.

Extracted from IOPaint to avoid dependency conflicts.
Model file should be located at /checkpoints/big-lama.pt
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import torch
import cv2

logger = logging.getLogger(__name__)


def norm_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: Image array (uint8 or float)
    
    Returns:
        Normalized image array (float32, range [0, 1])
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


class LaMa:
    """
    LaMa (Large Mask Inpainting) model for structural guidance generation.
    
    This is a standalone implementation extracted from IOPaint to avoid
    dependency conflicts with huggingface-hub.
    """
    
    name = "lama"
    pad_mod = 8
    is_erase_model = True
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize LaMa model.
        
        Args:
            device: Torch device (default: cuda if available, else cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.init_model(self.device)
    
    def init_model(self, device: torch.device, **kwargs) -> None:
        """
        Load LaMa model from checkpoint file.
        
        Args:
            device: Torch device to load model on
            **kwargs: Additional arguments (ignored)
        """
        # Model path in Modal Volume
        model_path = "/checkpoints/big-lama.pt"
        
        # Also check alternative paths
        candidate_paths = [
            model_path,
            os.path.join("/checkpoints", "big-lama.pt"),
            os.path.join("/checkpoints", "big-lama", "big-lama.pt"),
        ]
        
        found_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        
        if not found_path:
            raise FileNotFoundError(
                f"LaMa model not found at expected paths: {candidate_paths}. "
                "Please ensure big-lama.pt is available in Modal Volume at /checkpoints/"
            )
        
        logger.info(f"🎯 Loading LaMa model from {found_path} on {device}")
        try:
            self.model = torch.jit.load(found_path, map_location=device).eval()
            self.device = device
            logger.info("✅ LaMa model loaded successfully")
        except Exception as exc:
            logger.error(f"❌ Failed to load LaMa model: {exc}")
            raise RuntimeError(f"Failed to load LaMa model: {exc}") from exc
    
    @staticmethod
    def is_downloaded() -> bool:
        """
        Check if LaMa model file exists.
        
        Returns:
            True if model file exists, False otherwise
        """
        candidate_paths = [
            "/checkpoints/big-lama.pt",
            os.path.join("/checkpoints", "big-lama.pt"),
            os.path.join("/checkpoints", "big-lama", "big-lama.pt"),
        ]
        return any(os.path.exists(p) for p in candidate_paths)
    
    def forward(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        config: Optional[dict] = None
    ) -> np.ndarray:
        """
        Run LaMa inpainting inference.
        
        Args:
            image: Input image as [H, W, C] RGB numpy array (uint8)
            mask: Input mask as [H, W] numpy array (uint8, 0-255)
            config: Optional config dict (ignored for now)
        
        Returns:
            Inpainted image as [H, W, C] RGB numpy array (uint8)
        """
        if self.model is None:
            raise RuntimeError("LaMa model not initialized. Call init_model() first.")
        
        # Normalize inputs to [0, 1]
        image_norm = norm_img(image)
        mask_norm = norm_img(mask)
        
        # Binarize mask: > 0 -> 1, else 0
        mask_norm = (mask_norm > 0).astype(np.float32)
        
        # Convert to torch tensors and add batch dimension
        # Image: [H, W, C] -> [1, C, H, W]
        # Mask: [H, W] -> [1, 1, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            inpainted_image = self.model(image_tensor, mask_tensor)
        
        # Convert back to numpy: [1, C, H, W] -> [H, W, C]
        result = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Clip to [0, 1] and convert to uint8 [0, 255]
        result = np.clip(result, 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
        
        # Model outputs RGB, so return as-is
        return result
    
    def _calculate_square_size(self, width: int, height: int) -> int:
        """
        Calculate target square size for LaMa processing.
        
        Takes max(width, height), rounds up to nearest multiple of 8,
        and caps at 2048 to avoid memory issues.
        
        Args:
            width: Original image width
            height: Original image height
        
        Returns:
            Target square size (multiple of 8, max 2048)
        """
        max_dim = max(width, height)
        
        # Round up to nearest multiple of 8 (pad_mod requirement)
        if max_dim % self.pad_mod != 0:
            max_dim = ((max_dim // self.pad_mod) + 1) * self.pad_mod
        
        # Cap at 2048 (LaMa works up to ~2K)
        return min(max_dim, 2048)
    
    def _pad_to_square(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        target_size: int
    ) -> tuple:
        """
        Pad image and mask to square size with centered content.
        
        Args:
            image: Input image [H, W, C] (uint8)
            mask: Input mask [H, W] (uint8)
            target_size: Target square dimension (must be multiple of 8)
        
        Returns:
            (padded_image, padded_mask, padding_info)
            padding_info: {top, bottom, left, right, original_h, original_w}
        """
        h, w = image.shape[:2]
        
        # Calculate padding to center the image
        pad_h = target_size - h
        pad_w = target_size - w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Pad with black (0, 0, 0) for image
        padded_image = np.pad(
            image, 
            ((top, bottom), (left, right), (0, 0)), 
            mode='constant', 
            constant_values=0
        )
        
        # Pad with black (0) for mask
        padded_mask = np.pad(
            mask,
            ((top, bottom), (left, right)),
            mode='constant',
            constant_values=0
        )
        
        padding_info = {
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'original_h': h,
            'original_w': w
        }
        
        logger.debug(
            f"Padded {w}x{h} to {target_size}x{target_size} "
            f"(T:{top}, B:{bottom}, L:{left}, R:{right})"
        )
        
        return padded_image, padded_mask, padding_info
    
    def _crop_from_padded(
        self, 
        result: np.ndarray, 
        padding_info: dict
    ) -> np.ndarray:
        """
        Crop result back to original size after padding.
        
        Args:
            result: Padded result image [H_pad, W_pad, C]
            padding_info: Dict with padding coordinates
        
        Returns:
            Cropped image [H_orig, W_orig, C]
        """
        top = padding_info['top']
        left = padding_info['left']
        h = padding_info['original_h']
        w = padding_info['original_w']
        
        cropped = result[top:top+h, left:left+w]
        logger.debug(f"Cropped from padded to original {w}x{h}")
        
        return cropped
    
    def forward_with_preprocessing(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: Optional[dict] = None
    ) -> np.ndarray:
        """
        Run LaMa with automatic preprocessing/postprocessing for any image size.
        
        This method handles non-square images by:
        1. Padding to square (maintaining 1:1 ratio required by LaMa)
        2. Ensuring dimensions are multiples of 8
        3. Capping at 2048x2048 (LaMa generalizes well up to ~2K)
        4. Running inference
        5. Cropping back to original size
        
        LaMa requires 1:1 aspect ratio but generalizes surprisingly well to much
        higher resolutions (~2K) than it saw during training (256x256).
        
        Args:
            image: Input RGB image [H, W, C] (uint8)
            mask: Input mask [H, W] (uint8, 0=keep, 255=inpaint)
            config: Optional config (unused)
        
        Returns:
            Inpainted image [H, W, C] (uint8) at original size
        """
        original_h, original_w = image.shape[:2]
        
        # Calculate target square size (multiple of 8, max 2048)
        target_size = self._calculate_square_size(original_w, original_h)
        
        # Check if padding is needed
        if original_h == target_size and original_w == target_size:
            # Already square and correct size - no preprocessing needed
            logger.info(
                f"🟩 LaMa: Image already square {target_size}x{target_size}, "
                "no padding needed"
            )
            return self.forward(image, mask, config)
        
        # Pad to square
        padded_image, padded_mask, padding_info = self._pad_to_square(
            image, mask, target_size
        )
        logger.info(
            f"🔲 LaMa: Padded {original_w}x{original_h} → {target_size}x{target_size} "
            f"(1:1 ratio required)"
        )
        
        # Run LaMa inference on padded square image
        result = self.forward(padded_image, padded_mask, config)
        
        # Crop back to original size
        result = self._crop_from_padded(result, padding_info)
        
        logger.info(f"✅ LaMa: Cropped back to original {original_w}x{original_h}")
        return result
    
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Alias for forward() to allow calling model directly.
        
        Args:
            image: Input image as [H, W, C] RGB numpy array (uint8)
            mask: Input mask as [H, W] numpy array (uint8, 0-255)
        
        Returns:
            Inpainted image as [H, W, C] RGB numpy array (uint8)
        """
        return self.forward(image, mask)

