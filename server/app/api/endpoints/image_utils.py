"""
Image utility endpoints for image processing operations.

Provides endpoints for object extraction from images using masks.
"""

import base64
import io
import logging
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for shadow hint function
try:
    from app.services.image_processing import add_shadow_hint_for_inpainting
    SHADOW_HINT_AVAILABLE = True
except ImportError:
    SHADOW_HINT_AVAILABLE = False
    logger.warning(
        "⚠️ Shadow hint function not available. "
        "Shadow preview will not work."
    )

router = APIRouter(prefix="/api/image-utils", tags=["image-utils"])


class ExtractObjectRequest(BaseModel):
    """Request model for object extraction."""
    image: str  # Base64 encoded image
    mask: str   # Base64 encoded mask (white = object, black = background)


class ExtractObjectResponse(BaseModel):
    """Response model for object extraction."""
    success: bool
    extracted_image: Optional[str] = None  # Base64 encoded PNG with transparent bg
    error: Optional[str] = None


class AddShadowPreviewRequest(BaseModel):
    """Request model for shadow preview."""
    extracted_image: str  # Base64 encoded extracted image (object on white background)
    mask: Optional[str] = None  # Optional: Base64 encoded mask (if not provided, will extract from image)
    shadow_offset_x: int = 3  # Optional: Shadow offset X (default: 3)
    shadow_offset_y: int = 5  # Optional: Shadow offset Y (default: 5)
    shadow_opacity: float = 0.4  # Optional: Shadow opacity 0.0-1.0 (default: 0.4)
    blur_radius: int = 21  # Optional: Blur radius (default: 21)


class AddShadowPreviewResponse(BaseModel):
    """Response model for shadow preview."""
    success: bool
    preview_image: Optional[str] = None  # Base64 encoded PNG with shadow preview
    error: Optional[str] = None


def _decode_image(image_b64: str) -> Image.Image:
    """
    Decode base64 image to PIL Image.
    
    Args:
        image_b64: Base64 encoded image
    
    Returns:
        PIL Image in RGBA mode
    
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        image_data = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_data)).convert("RGBA")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e


def _decode_mask(mask_b64: str) -> Image.Image:
    """
    Decode base64 mask to PIL Image.
    
    Args:
        mask_b64: Base64 encoded mask
    
    Returns:
        PIL Image in grayscale (L) mode
    
    Raises:
        ValueError: If mask cannot be decoded
    """
    try:
        mask_data = base64.b64decode(mask_b64)
        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        return mask
    except Exception as e:
        raise ValueError(f"Failed to decode mask: {e}") from e


def _find_bounding_box(mask: Image.Image) -> Tuple[int, int, int, int]:
    """
    Find bounding box of white pixels in mask.
    
    Args:
        mask: PIL Image in grayscale mode
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max)
    
    Raises:
        ValueError: If no white pixels found
    """
    mask_array = np.array(mask, dtype=np.uint8)
    white_pixels = np.where(mask_array > 128)
    
    if len(white_pixels[0]) == 0:
        raise ValueError("No object found in mask (no white pixels)")
    
    y_min, y_max = int(np.min(white_pixels[0])), int(np.max(white_pixels[0])) + 1
    x_min, x_max = int(np.min(white_pixels[1])), int(np.max(white_pixels[1])) + 1
    
    return x_min, y_min, x_max, y_max


def _calculate_target_size(cropped_width: int, cropped_height: int) -> int:
    """
    Calculate target size for resized object (512x512 or 1024x1024).
    
    Args:
        cropped_width: Width of cropped object
        cropped_height: Height of cropped object
    
    Returns:
        Target size (512 or 1024)
    """
    max_dimension = max(cropped_width, cropped_height)
    return 512 if max_dimension <= 512 else 1024


def _resize_with_aspect_ratio(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image to fit within target_size while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        target_size: Maximum dimension size
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _create_result_image(
    resized_object: Image.Image,
    resized_mask: Image.Image,
    target_size: int
) -> Image.Image:
    """
    Create final result image with object centered on white background.
    
    Args:
        resized_object: Resized object image (RGBA)
        resized_mask: Resized mask (L)
        target_size: Target canvas size
    
    Returns:
        Final result image (RGB) with object on white background
    """
    # Create white background (RGB, not RGBA, not transparent, not black)
    result = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
    
    # Ensure object is in RGBA mode for alpha compositing
    if resized_object.mode != "RGBA":
        resized_object = resized_object.convert("RGBA")
    
    # Create alpha channel from mask
    alpha = resized_mask.convert("L")
    resized_object.putalpha(alpha)
    
    # Calculate position to center the object
    paste_x = (target_size - resized_object.width) // 2
    paste_y = (target_size - resized_object.height) // 2
    
    # Paste object onto white background
    result.paste(resized_object, (paste_x, paste_y), resized_object)
    
    return result


def _encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image to encode
    
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_object_mask_from_image(image: Image.Image) -> Image.Image:
    """
    Extract object mask from extracted image (object on white background).
    
    Finds all non-white pixels to create a binary mask of the object.
    
    Args:
        image: PIL Image with object on white background (RGB or RGBA)
    
    Returns:
        Binary mask (L mode) where white (255) = object, black (0) = background
    
    Raises:
        ValueError: If no object found (all pixels are white)
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.uint8)
    
    # Find non-white pixels (not exactly 255, 255, 255)
    # Use threshold to account for slight variations
    white_threshold = 250
    is_white = np.all(img_array >= white_threshold, axis=2)
    
    # Create binary mask: 255 for object (non-white), 0 for background (white)
    mask_array = np.where(is_white, 0, 255).astype(np.uint8)
    
    # Check if mask has any object pixels
    if np.all(mask_array == 0):
        raise ValueError("No object found in extracted image (all pixels are white)")
    
    return Image.fromarray(mask_array, mode="L")


def _composite_on_light_background(image: Image.Image, background_color: Tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    """
    Composite image on light gray background for shadow preview.
    
    Args:
        image: PIL Image to composite (RGB or RGBA)
        background_color: RGB tuple for background color (default: #F5F5F5)
    
    Returns:
        Composite image (RGB) with object on light background
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Create light gray background
    width, height = image.size
    background = Image.new("RGB", (width, height), color=background_color)
    
    # If image has alpha channel, composite it
    if image.mode == "RGBA":
        background.paste(image, (0, 0), image)
    else:
        # For RGB, just paste directly
        background.paste(image, (0, 0))
    
    return background


@router.post("/extract-object", response_model=ExtractObjectResponse)
async def extract_object_endpoint(request: ExtractObjectRequest):
    """
    Extract object from image using mask.
    
    The mask should be:
    - White (255) = object to keep
    - Black (0) = background to remove
    
    Returns:
        PNG image with object resized to 1:1 ratio (512x512 or 1024x1024)
        on white background (not transparent, not black)
    
    Raises:
        HTTPException: If extraction fails
    """
    try:
        # Decode images
        image = _decode_image(request.image)
        mask = _decode_mask(request.mask)
        
        # Resize mask if needed (use NEAREST for mask - faster and preserves binary nature)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        logger.info(f"Extracting object: image={image.size}, mask={mask.size}")
        
        # Find bounding box and crop
        x_min, y_min, x_max, y_max = _find_bounding_box(mask)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        # Calculate target size and resize
        target_size = _calculate_target_size(
            cropped_image.width, cropped_image.height
        )
        resized_object = _resize_with_aspect_ratio(cropped_image, target_size)
        resized_mask = _resize_with_aspect_ratio(cropped_mask, target_size)
        
        # Create final result
        result = _create_result_image(resized_object, resized_mask, target_size)
        
        # Encode as base64
        result_base64 = _encode_image_to_base64(result)
        
        logger.info(
            f"Object extracted successfully: {cropped_image.size} -> "
            f"{target_size}x{target_size} on white background"
        )
        
        return ExtractObjectResponse(
            success=True,
            extracted_image=result_base64
        )
        
    except ValueError as e:
        logger.error(f"Validation error extracting object: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error extracting object: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract object: {str(e)}"
        ) from e


@router.post("/add-shadow-preview", response_model=AddShadowPreviewResponse)
async def add_shadow_preview_endpoint(request: AddShadowPreviewRequest):
    """
    Add shadow preview to extracted object image.
    
    The extracted_image should be an object on white background (from extract-object endpoint).
    This function:
    1. Extracts object mask from the image (finds non-white pixels)
    2. Composites image on light gray background (#F5F5F5) for shadow visibility
    3. Applies shadow hint using add_shadow_hint_for_inpainting
    4. Returns preview image with shadow
    
    Args:
        request: AddShadowPreviewRequest with extracted_image and optional parameters
    
    Returns:
        PNG image with shadow preview on light gray background
    
    Raises:
        HTTPException: If shadow preview fails
    """
    if not SHADOW_HINT_AVAILABLE:
        logger.error("Shadow hint function not available (OpenCV may not be installed)")
        return AddShadowPreviewResponse(
            success=False,
            error="Shadow preview not available. OpenCV is required but not installed."
        )
    
    try:
        # Decode extracted image
        extracted_image = _decode_image(request.extracted_image)
        
        # Get or extract mask
        if request.mask:
            # Use provided mask
            mask = _decode_mask(request.mask)
            # Resize mask if needed
            if mask.size != extracted_image.size:
                mask = mask.resize(extracted_image.size, Image.Resampling.NEAREST)
        else:
            # Extract mask from image (find non-white pixels)
            logger.info("Extracting object mask from image...")
            mask = _extract_object_mask_from_image(extracted_image)
        
        logger.info(f"Adding shadow preview: image={extracted_image.size}, mask={mask.size}")
        
        # Composite extracted image on light gray background
        composite_image = _composite_on_light_background(extracted_image)
        
        # Convert to numpy arrays for shadow hint processing
        composite_array = np.array(composite_image.convert("RGB"), dtype=np.uint8)
        mask_array = np.array(mask.convert("L"), dtype=np.uint8)
        
        # Apply shadow hint
        preview_with_shadow, _ = add_shadow_hint_for_inpainting(
            composite_image=composite_array,
            object_mask=mask_array,
            shadow_offset=(request.shadow_offset_x, request.shadow_offset_y),
            shadow_opacity=request.shadow_opacity,
            blur_radius=request.blur_radius,
            floating_height=0,  # Object grounded
            ground_bias=True,  # Gradient from top to bottom
        )
        
        # Convert back to PIL Image
        preview_image = Image.fromarray(preview_with_shadow, mode="RGB")
        
        # Encode as base64
        preview_base64 = _encode_image_to_base64(preview_image)
        
        logger.info(
            f"Shadow preview added successfully: "
            f"offset=({request.shadow_offset_x}, {request.shadow_offset_y}), "
            f"opacity={request.shadow_opacity}, blur={request.blur_radius}"
        )
        
        return AddShadowPreviewResponse(
            success=True,
            preview_image=preview_base64
        )
        
    except ValueError as e:
        logger.error(f"Validation error adding shadow preview: {e}")
        return AddShadowPreviewResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Error adding shadow preview: {e}", exc_info=True)
        return AddShadowPreviewResponse(
            success=False,
            error=f"Failed to add shadow preview: {str(e)}"
        )
