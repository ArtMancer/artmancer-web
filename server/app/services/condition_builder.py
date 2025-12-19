"""
Condition Builder Service

Builds conditional images with correct order for each task type.
Implements caching to avoid redundant image processing operations.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from fastapi import HTTPException

from ..services.image_processing import (
    prepare_mask_conditionals,
    generate_canny_image,
    dilate_mask_for_mae,
    CV2_AVAILABLE,
    LAMA_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Lazy import for shadow hint function
try:
    from ..services.image_processing import add_shadow_hint_for_inpainting
    SHADOW_HINT_AVAILABLE = True
except ImportError:
    SHADOW_HINT_AVAILABLE = False
    logger.warning(
        "⚠️ Shadow hint function not available. "
        "Shadows will not be enhanced in insertion pipeline."
    )

# Cache for conditionals (60s TTL)
_conditional_cache: Dict[str, Tuple[List[Image.Image], float]] = {}
CACHE_TTL_SECONDS = 60


def _generate_cache_key(
    original: Image.Image,
    mask: Optional[Image.Image],
    task: str,
    **kwargs
) -> str:
    """
    Generate cache key from images and task parameters.
    
    Args:
        original: Original image
        mask: Optional mask image
        task: Task type string
        **kwargs: Additional parameters to include in cache key
    
    Returns:
        MD5 hash string as cache key
    """
    hash_input = f"{task}_{original.size}_{original.mode}"
    
    if mask:
        hash_input += f"_{mask.size}_{mask.mode}"
    
    # Add additional parameters (sorted for consistency)
    for key, value in sorted(kwargs.items()):
        if value is not None:
            hash_input += f"_{key}_{value}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def _get_cached_conditionals(cache_key: str) -> Optional[List[Image.Image]]:
    """
    Get cached conditionals if still valid.
    
    Args:
        cache_key: Cache key to lookup
    
    Returns:
        Cached conditionals list, or None if not found or expired
    """
    if cache_key not in _conditional_cache:
        return None
    
    conditionals, timestamp = _conditional_cache[cache_key]
    if time.time() - timestamp < CACHE_TTL_SECONDS:
        logger.debug(f"✅ Using cached conditionals for key: {cache_key[:8]}...")
        return conditionals
    
    # Expired, remove from cache
    del _conditional_cache[cache_key]
    return None


def _cache_conditionals(cache_key: str, conditionals: List[Image.Image]) -> None:
    """
    Cache conditionals with timestamp.
    
    Args:
        cache_key: Cache key
        conditionals: List of conditional images to cache
    """
    _conditional_cache[cache_key] = (conditionals, time.time())
    logger.debug(f"💾 Cached conditionals for key: {cache_key[:8]}...")


def _ensure_rgb_image(image: Image.Image) -> Image.Image:
    """
    Ensure image is in RGB mode.
    
    Args:
        image: PIL Image
    
    Returns:
        Image in RGB mode
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _resize_to_match(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image to match target size if needed.
    
    Args:
        image: PIL Image to resize
        target_size: Target size (width, height)
    
    Returns:
        Resized image (or original if already matches)
    """
    if image.size != target_size:
        logger.info(f"🔄 Resizing image from {image.size} to {target_size}")
        return image.resize(target_size, Image.Resampling.LANCZOS)
    return image


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
        cache_key = _generate_cache_key(
            original, mask, "insertion", ref_img_size=ref_img.size
        )
        cached = _get_cached_conditionals(cache_key)
        if cached:
            return cached
    
    # Ensure all images are RGB
    original = _ensure_rgb_image(original)
    mask = _ensure_rgb_image(mask)
    ref_img = _ensure_rgb_image(ref_img)
    
    # Resize images to match original size
    mask = _resize_to_match(mask, original.size)
    ref_img = _resize_to_match(ref_img, original.size)
    
    # Prepare mask conditionals (mask_rgb, masked_bg, masked_object, mae)
    # For insertion, we only need mask_rgb and masked_bg
    mask_rgb, masked_bg, _, _ = prepare_mask_conditionals(
        original, mask, include_mae=False
    )
    
    # Apply shadow hint for realistic shadow generation
    # This darkens the background around the object and expands the mask
    # to give the model context for generating realistic cast shadows
    if SHADOW_HINT_AVAILABLE:
        try:
            # Convert to numpy arrays for shadow hint processing
            original_array = np.array(original.convert("RGB"), dtype=np.uint8)
            mask_array = np.array(mask.convert("L"), dtype=np.uint8)
            
            # Apply shadow hint to original image (full background)
            original_with_shadow, expanded_mask = add_shadow_hint_for_inpainting(
                composite_image=original_array,
                object_mask=mask_array,
                shadow_offset=(3, 5),    # Small offset for subtle shadow hint
                shadow_opacity=0.4,      # Subtle hint (0.3-0.6 recommended)
                blur_radius=21,          # Soft natural shadow
            )
            
            # Convert expanded mask back to PIL
            expanded_mask_pil = Image.fromarray(expanded_mask, mode="L")
            
            # Update masked_bg: Apply shadow hint to background regions
            # Keep empty (black) regions at mask, but darken shadow areas
            original_shadow_pil = Image.fromarray(original_with_shadow, mode="RGB")
            mask_array_float = np.array(mask.convert("L"), dtype=np.float32) / 255.0
            mask_stack = np.repeat(mask_array_float[..., None], 3, axis=2)
            
            # Convert original_with_shadow to numpy
            original_shadow_array = np.array(original_shadow_pil.convert("RGB"), dtype=np.float32) / 255.0
            
            # Combine: use original_with_shadow for background, keep black for mask region
            masked_bg_with_shadow = original_shadow_array * (1.0 - mask_stack)
            masked_bg = Image.fromarray(
                np.uint8(np.clip(masked_bg_with_shadow * 255, 0, 255)),
                mode="RGB"
            )
            
            # Use expanded mask for generation (gives model more context)
            mask_rgb = expanded_mask_pil.convert("RGB")
            
            logger.info(
                "✨ Applied shadow hint to insertion conditionals: "
                f"darkened background, expanded mask"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to apply shadow hint: {e}. "
                "Continuing without shadow enhancement."
            )
            # Continue with original mask_rgb and masked_bg
    else:
        logger.debug("Shadow hint not available, skipping shadow enhancement")
    
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
    request_id: Optional[str] = None,
    mask_tool_type: Optional[str] = None,
    enable_mae_refinement: bool = True,
) -> Tuple[List[Image.Image], Optional[Image.Image]]:
    """
    Build conditionals for removal task.
    
    Conditional order (MANDATORY): [masked_bg, mask_rgb, mae]
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        enable_mae: Whether to generate MAE (default: True, uses LaMa)
        use_cache: Whether to use caching
        request_id: Request id for cancellation tracking
        mask_tool_type: Mask creation tool type ("brush" or "box"). 
                       If "brush" and enable_mae=True, mask will be dilated 15-25px before MAE generation.
        enable_mae_refinement: Whether to refine LaMa output with Stable Diffusion Inpainting
                              (default: True, improves texture quality)
    
    Returns:
        Tuple of:
        - List of conditional images: [masked_bg, mask_rgb, mae]
        - Optional dilated mask for MAE (if brush mask and MAE enabled), None otherwise
    
    Raises:
        HTTPException: If LaMa is not available and MAE is enabled
    """
    # Validate LaMa requirement
    if enable_mae and not LAMA_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail=(
                "LaMa library is not available. "
                "MAE image generation requires LaMa (simple-lama-inpainting). "
                "Please ensure 'simple-lama-inpainting' package is installed."
            )
        )
    
    # Generate cache key (include mask_tool_type for proper caching)
    cache_key = None
    if use_cache:
        cache_key = _generate_cache_key(
            original,
            mask,
            "removal",
            enable_mae=enable_mae,
            mask_tool_type=mask_tool_type,
        )
        cached = _get_cached_conditionals(cache_key)
        if cached:
            # For cached results, we don't have mask_for_mae, return None
            return cached, None
    
    # Ensure all images are RGB
    original = _ensure_rgb_image(original)
    mask_original_mode = mask.mode
    mask = _ensure_rgb_image(mask)
    logger.info(
        f"🎭 [Removal Conditionals] Mask converted from {mask_original_mode} to RGB: "
        f"size={mask.size}"
    )
    
    # Resize mask to match original if needed
    mask = _resize_to_match(mask, original.size)
    mask_gray = mask.convert("L")

    # BiRefNet refine disabled - not needed for current use case
    # if refine_mask_with_birefnet:
    #     try:
    #         refined_mask = refine_mask_with_birefnet_fn(...)
    #         mask_gray = refined_mask
    #     except Exception as exc:
    #         logger.warning("BiRefNet refine failed: %s", exc)
    
    # Use mask as RGB
    mask = mask_gray.convert("RGB")
    
    # Verify mask content before processing
    mask_array = np.array(mask.convert("RGB"))
    mask_min = np.min(mask_array)
    mask_max = np.max(mask_array)
    mask_mean = np.mean(mask_array)
    logger.info(
        f"🎭 [Removal Conditionals] Mask pixel stats: "
        f"min={mask_min}, max={mask_max}, mean={mask_mean:.2f}"
    )
    
    # For brush masks with MAE enabled: dilate mask before MAE generation
    # This gives LaMa more context around the object boundary
    mask_for_mae = mask
    logger.debug(
        f"🔍 [Removal Conditionals] Mask tool type: {mask_tool_type}, "
        f"enable_mae: {enable_mae}"
    )
    if mask_tool_type == "brush" and enable_mae:
        try:
            # Dilate mask 35px for MAE preprocessing (increased from 20px for more visible expansion)
            mask_for_mae = dilate_mask_for_mae(mask, dilation_pixels=35)
            logger.info(
                f"🔍 [Removal Conditionals] Dilated brush mask for MAE: "
                f"35px expansion applied (original={mask.size}, dilated={mask_for_mae.size})"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to dilate mask for MAE: {e}. "
                "Using original mask for MAE generation."
            )
            mask_for_mae = mask
    else:
        logger.debug(
            f"🔍 [Removal Conditionals] Skipping dilation: "
            f"mask_tool_type={mask_tool_type} (expected 'brush'), enable_mae={enable_mae}"
        )
    
    # Prepare mask conditionals (mask_rgb, masked_bg, masked_object, mae)
    # For removal, chúng ta cần masked_bg, mask_rgb, mae
    # Note: Use original mask for mask_rgb and masked_bg, but use dilated mask_for_mae for MAE
    mask_rgb, masked_bg, _, _ = prepare_mask_conditionals(
        original, mask, include_mae=False
    )
    
    # Generate MAE separately with potentially dilated mask
    if enable_mae:
        from ..services.image_processing import generate_mae_image
        mae = generate_mae_image(original, mask_for_mae, enable_refinement=enable_mae_refinement)
        if mask_tool_type == "brush":
            logger.info(
                f"🧩 Generated MAE image with dilated brush mask (35px expansion), "
                f"refinement={'enabled' if enable_mae_refinement else 'disabled'}"
            )
        else:
            logger.info(
                f"🧩 Generated MAE image (no dilation for box mask), "
                f"refinement={'enabled' if enable_mae_refinement else 'disabled'}"
            )
    else:
        mae = masked_bg
    
    # Verify mask_rgb after processing
    mask_rgb_array = np.array(mask_rgb.convert("RGB"))
    mask_rgb_min = np.min(mask_rgb_array)
    mask_rgb_max = np.max(mask_rgb_array)
    mask_rgb_mean = np.mean(mask_rgb_array)
    logger.info(
        f"🎭 [Removal Conditionals] Mask RGB pixel stats: "
        f"min={mask_rgb_min}, max={mask_rgb_max}, mean={mask_rgb_mean:.2f}"
    )
    logger.info(
        f"🎭 [Removal Conditionals] Mask RGB size: {mask_rgb.size}, mode: {mask_rgb.mode}"
    )
    
    # Build conditionals in correct order: [masked_bg, mask, mae]
    conditionals = [masked_bg, mask_rgb, mae]
    
    # Log final conditionals info
    logger.info(
        f"🎭 [Removal Conditionals] Final conditionals: "
        f"[0] masked_bg={masked_bg.size}, [1] mask_rgb={mask_rgb.size}, [2] mae={mae.size}"
    )
    
    # Cache result
    if use_cache and cache_key:
        _cache_conditionals(cache_key, conditionals)
    
    logger.info(
        f"✅ Built removal conditionals: [masked_bg, mask, mae] (MAE enabled: {enable_mae})"
    )
    
    # Return conditionals and dilated mask for MAE (if available)
    # Only return dilated mask if it's different from original mask (i.e., dilation was applied)
    if mask_tool_type == "brush" and enable_mae and mask_for_mae != mask:
        logger.info(
            f"🔍 [Removal Conditionals] Returning dilated mask for MAE: "
            f"original_size={mask.size}, dilated_size={mask_for_mae.size}"
        )
        mask_for_mae_dilated = mask_for_mae
    else:
        logger.debug(
            f"🔍 [Removal Conditionals] No dilated mask to return: "
            f"mask_tool_type={mask_tool_type}, enable_mae={enable_mae}, "
            f"mask_changed={mask_for_mae != mask}"
        )
        mask_for_mae_dilated = None
    return conditionals, mask_for_mae_dilated


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
        cache_key = _generate_cache_key(
            input_image, None, "white_balance", canny_thresholds=canny_thresholds
        )
        cached = _get_cached_conditionals(cache_key)
        if cached:
            return cached
    
    # Ensure image is RGB
    input_image = _ensure_rgb_image(input_image)
    
    # Generate Canny edge image (RGB output, not grayscale)
    canny = generate_canny_image(input_image, canny_thresholds[0], canny_thresholds[1])
    
    # Build conditionals in correct order: [input, canny]
    conditionals = [input_image, canny]
    
    # Cache result
    if use_cache and cache_key:
        _cache_conditionals(cache_key, conditionals)
    
    logger.info(
        f"✅ Built white_balance conditionals: [input, canny] "
        f"(thresholds: {canny_thresholds})"
    )
    return conditionals
