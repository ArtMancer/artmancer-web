"""
Condition Builder Service
Builds conditional images with correct order for each task type.
"""
from __future__ import annotations

import logging
import hashlib
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from fastapi import HTTPException

from ..services.image_processing import (
    prepare_mask_conditionals,
    generate_canny_image,
    CV2_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Cache for conditionals (60s TTL)
_conditional_cache: Dict[str, Tuple[List[Image.Image], float]] = {}
CACHE_TTL_SECONDS = 60


def _generate_cache_key(original: Image.Image, mask: Optional[Image.Image], task: str, **kwargs) -> str:
    """Generate cache key from images and task parameters."""
    # Create hash from image data and parameters
    hash_input = f"{task}_{original.size}_{original.mode}"
    
    if mask:
        hash_input += f"_{mask.size}_{mask.mode}"
    
    # Add additional parameters
    for key, value in sorted(kwargs.items()):
        if value is not None:
            hash_input += f"_{key}_{value}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def _get_cached_conditionals(cache_key: str) -> Optional[List[Image.Image]]:
    """Get cached conditionals if still valid."""
    if cache_key in _conditional_cache:
        conditionals, timestamp = _conditional_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            logger.debug(f"✅ Using cached conditionals for key: {cache_key[:8]}...")
            return conditionals
        else:
            # Expired, remove from cache
            del _conditional_cache[cache_key]
    
    return None


def _cache_conditionals(cache_key: str, conditionals: List[Image.Image]) -> None:
    """Cache conditionals with timestamp."""
    _conditional_cache[cache_key] = (conditionals, time.time())
    logger.debug(f"💾 Cached conditionals for key: {cache_key[:8]}...")


def build_insertion_conditionals(
    original: Image.Image,
    mask: Image.Image,
    ref_img: Image.Image,
    use_cache: bool = True,
) -> List[Image.Image]:
    """
    Build conditionals for insertion task.
    
    Conditional order (MANDATORY): [ref_img, mask, masked_bg]
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        ref_img: Reference image (object to insert)
        use_cache: Whether to use caching
    
    Returns:
        List of conditional images: [ref_img_resized, mask_rgb, masked_bg]
    """
    # Generate cache key
    cache_key = None
    if use_cache:
        cache_key = _generate_cache_key(original, mask, "insertion", ref_img_size=ref_img.size)
        cached = _get_cached_conditionals(cache_key)
        if cached:
            return cached
    
    # Ensure all images are RGB
    original = original.convert("RGB")
    mask = mask.convert("RGB")
    ref_img = ref_img.convert("RGB")
    
    # Resize mask to match original if needed
    if mask.size != original.size:
        logger.info(f"🔄 Resizing mask from {mask.size} to {original.size}")
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    
    # Resize ref_img to match original size
    if ref_img.size != original.size:
        logger.info(f"🔄 Resizing ref_img from {ref_img.size} to {original.size}")
        ref_img = ref_img.resize(original.size, Image.Resampling.LANCZOS)
    
    # Prepare mask conditionals (mask_rgb, masked_bg, masked_object, mae)
    # For insertion, we only need mask_rgb and masked_bg
    mask_rgb, masked_bg, _, _ = prepare_mask_conditionals(original, mask, include_mae=False)
    
    # Build conditionals in correct order: [ref_img, mask, masked_bg]
    conditionals = [ref_img, mask_rgb, masked_bg]
    
    # Cache result
    if use_cache and cache_key:
        _cache_conditionals(cache_key, conditionals)
    
    logger.info("✅ Built insertion conditionals: [ref_img, mask, masked_bg]")
    return conditionals


def build_removal_conditionals(
    original: Image.Image,
    mask: Image.Image,
    enable_mae: bool = True,
    use_cache: bool = True,
) -> List[Image.Image]:
    """
    Build conditionals for removal task.
    
    Conditional order (MANDATORY): [original, mask, mae]
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        enable_mae: Whether to generate MAE (default: True, uses Telea)
        use_cache: Whether to use caching
    
    Returns:
        List of conditional images: [original, mask_rgb, mae]
    
    Raises:
        HTTPException: If OpenCV is not available and MAE is enabled
    """
    # Validate MAE requirement
    if enable_mae and not CV2_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "OpenCV library (cv2) is not installed. "
                "MAE image generation requires OpenCV. "
                "Please install 'opencv-python-headless' package."
            )
        )
    
    # Generate cache key
    cache_key = None
    if use_cache:
        cache_key = _generate_cache_key(original, mask, "removal", enable_mae=enable_mae)
        cached = _get_cached_conditionals(cache_key)
        if cached:
            return cached
    
    # Ensure all images are RGB
    original = original.convert("RGB")
    mask_original_mode = mask.mode
    mask = mask.convert("RGB")
    logger.info(f"🎭 [Removal Conditionals] Mask converted from {mask_original_mode} to RGB: size={mask.size}")
    
    # Resize mask to match original if needed (auto-resize, don't fail)
    if mask.size != original.size:
        logger.info(f"🔄 [Removal Conditionals] Resizing mask from {mask.size} to {original.size}")
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    
    # Verify mask content before processing
    mask_array = np.array(mask.convert("RGB"))
    mask_min = np.min(mask_array)
    mask_max = np.max(mask_array)
    mask_mean = np.mean(mask_array)
    logger.info(f"🎭 [Removal Conditionals] Mask pixel stats: min={mask_min}, max={mask_max}, mean={mask_mean:.2f}")
    
    # Prepare mask conditionals (mask_rgb, masked_bg, masked_object, mae)
    # For removal, we need mask_rgb and mae
    mask_rgb, _, _, mae = prepare_mask_conditionals(original, mask, include_mae=enable_mae)
    
    # Verify mask_rgb after processing
    mask_rgb_array = np.array(mask_rgb.convert("RGB"))
    mask_rgb_min = np.min(mask_rgb_array)
    mask_rgb_max = np.max(mask_rgb_array)
    mask_rgb_mean = np.mean(mask_rgb_array)
    logger.info(f"🎭 [Removal Conditionals] Mask RGB pixel stats: min={mask_rgb_min}, max={mask_rgb_max}, mean={mask_rgb_mean:.2f}")
    logger.info(f"🎭 [Removal Conditionals] Mask RGB size: {mask_rgb.size}, mode: {mask_rgb.mode}")
    
    # Build conditionals in correct order: [original, mask, mae]
    # Note: Do NOT use masked_bg for removal
    conditionals = [original, mask_rgb, mae]
    
    # Log final conditionals info
    logger.info(f"🎭 [Removal Conditionals] Final conditionals: [0] original={original.size}, [1] mask_rgb={mask_rgb.size}, [2] mae={mae.size}")
    
    # Cache result
    if use_cache and cache_key:
        _cache_conditionals(cache_key, conditionals)
    
    logger.info(f"✅ Built removal conditionals: [original, mask, mae] (MAE enabled: {enable_mae})")
    return conditionals


def build_white_balance_conditionals(
    input_image: Image.Image,
    canny_thresholds: Tuple[int, int] = (100, 200),
    use_cache: bool = True,
) -> List[Image.Image]:
    """
    Build conditionals for white balance task.
    
    Conditional order (MANDATORY): [input, canny]
    
    Args:
        input_image: Input RGB image
        canny_thresholds: (low, high) thresholds for Canny edge detection
        use_cache: Whether to use caching
    
    Returns:
        List of conditional images: [input, canny]
    
    Raises:
        HTTPException: If OpenCV is not available (required for Canny)
    """
    # Validate OpenCV requirement
    if not CV2_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "OpenCV library (cv2) is not installed. "
                "Canny edge detection requires OpenCV. "
                "Please install 'opencv-python-headless' package."
            )
        )
    
    # Generate cache key
    cache_key = None
    if use_cache:
        cache_key = _generate_cache_key(input_image, None, "white_balance", canny_thresholds=canny_thresholds)
        cached = _get_cached_conditionals(cache_key)
        if cached:
            return cached
    
    # Ensure image is RGB
    input_image = input_image.convert("RGB")
    
    # Generate Canny edge image (RGB output, not grayscale)
    canny = generate_canny_image(input_image, canny_thresholds[0], canny_thresholds[1])
    
    # Build conditionals in correct order: [input, canny]
    conditionals = [input_image, canny]
    
    # Cache result
    if use_cache and cache_key:
        _cache_conditionals(cache_key, conditionals)
    
    logger.info(f"✅ Built white_balance conditionals: [input, canny] (thresholds: {canny_thresholds})")
    return conditionals

