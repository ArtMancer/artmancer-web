from __future__ import annotations

import base64
import io
import logging
from typing import Tuple

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Optional import for cv2 (opencv-python)
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ opencv-python not available, MAE image generation will use fallback method")

logger = logging.getLogger(__name__)


def base64_to_image(data: str) -> Image.Image:
    """
    Convert base64 encoded image data to PIL Image.
    
    Supports common image formats: PNG, JPEG, JPG, WEBP, BMP, GIF, TIFF.
    
    Args:
        data: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        PIL Image in RGB mode
    
    Raises:
        ValueError: If data is invalid or image format is not supported
    """
    if not data or not data.strip():
        raise ValueError("Base64 image data cannot be empty or None")
    
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    
    if not data or not data.strip():
        raise ValueError("Base64 image data is empty after removing data URL prefix")
    
    try:
        image_bytes = base64.b64decode(data)
        if not image_bytes:
            raise ValueError("Decoded image bytes are empty")
        
        # Validate image format by checking magic bytes
        supported_formats = {
            b'\x89PNG': 'PNG',
            b'\xff\xd8\xff': 'JPEG',
            b'RIFF': 'WEBP',  # WEBP starts with RIFF
            b'BM': 'BMP',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'II*\x00': 'TIFF',  # Little-endian TIFF
            b'MM\x00*': 'TIFF',  # Big-endian TIFF
        }
        
        # Check if image format is supported
        format_detected = False
        for magic_bytes, format_name in supported_formats.items():
            if image_bytes.startswith(magic_bytes):
                format_detected = True
                logger.debug(f"📸 Detected image format: {format_name}")
                break
        
        if not format_detected:
            # Try to open anyway (PIL might handle it)
            logger.warning("⚠️ Image format not recognized from magic bytes, attempting to open with PIL")
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Verify it's a valid image format
        if image.format not in ['PNG', 'JPEG', 'JPG', 'WEBP', 'BMP', 'GIF', 'TIFF']:
            logger.warning(f"⚠️ Uncommon image format detected: {image.format}, converting to RGB")
        
        # Convert to RGB mode (required for model input)
        # If image has transparency (RGBA, LA, P with transparency), composite with white background
        if image.mode in ("RGBA", "LA", "P"):
            # Create white background
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            # Composite image on white background
            if image.mode == "P":
                image = image.convert("RGBA")
            image = Image.alpha_composite(
                white_bg.convert("RGBA"), 
                image.convert("RGBA")
            ).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        logger.debug(f"✅ Image loaded: {image.format}, size: {image.size}, mode: {image.mode}")
        return image
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid base64 image data or unsupported format: {exc}") from exc


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def resize_with_aspect_ratio_pad(
    image: Image.Image, 
    target_size: Tuple[int, int], 
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize image while preserving aspect ratio, then pad to target size.
    
    This prevents distortion when reference image has different aspect ratio
    than the original image.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        background_color: RGB color for padding (default: black)
    
    Returns:
        Resized and padded image with target size
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate scale factor to fit within target size (preserve aspect ratio)
    scale = min(target_width / original_width, target_height / original_height)
    
    # Resize with aspect ratio preserved
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create canvas with target size and background color
    canvas = Image.new('RGB', target_size, background_color)
    
    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Paste resized image onto canvas
    canvas.paste(resized, (x_offset, y_offset))
    
    logger.info(f"🔄 Resized reference image from {image.size} to {target_size} with aspect ratio preserved (padded to center)")
    
    return canvas


def generate_mae_image(
    original: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Generate MAE (Masked Autoencoder) conditional image.
    
    Uses OpenCV inpainting (Telea algorithm) to create a preview of the inpainted
    region, providing better context representation for the model.
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
    
    Returns:
        MAE image: RGB image with inpainted masked region
    """
    # Resize mask to match original image size if needed
    if mask.size != original.size:
        logger.info(f"🔄 Resizing mask from {mask.size} to {original.size} for MAE")
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    
    # Convert PIL images to numpy arrays
    original_array = np.array(original, dtype=np.uint8)
    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.uint8)
    
    # Invert mask for OpenCV inpainting (OpenCV expects mask where region to inpaint is non-zero)
    # Our mask is white where object is, so we need to invert it
    mask_inverted = 255 - mask_array
    
    # Use OpenCV inpainting with Telea algorithm (fast and effective)
    # cv2.INPAINT_TELEA is faster than cv2.INPAINT_NS (Navier-Stokes)
    if CV2_AVAILABLE:
        try:
            mae_array = cv2.inpaint(original_array, mask_inverted, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            mae_image = Image.fromarray(mae_array, mode="RGB")
            logger.info("✅ Generated MAE image using OpenCV Telea inpainting")
            return mae_image
        except Exception as e:
            logger.warning(f"⚠️ OpenCV inpainting failed: {e}, falling back to mean fill")
            # Fall through to mean fill fallback
    else:
        logger.info("📝 OpenCV not available, using mean fill for MAE image")
    
    # Fallback: fill masked region with mean color of background
    mask_bool = mask_array > 128  # Threshold to get binary mask
    
    # Calculate mean color of background (where mask is NOT)
    background_pixels = original_array[~mask_bool]
    if len(background_pixels) > 0:
        mean_color = np.mean(background_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        # If no background pixels, use gray
        mean_color = np.array([128, 128, 128], dtype=np.uint8)
    
    # Create MAE image: original where mask is NOT, mean color where mask is
    mae_array = original_array.copy()
    mae_array[mask_bool] = mean_color
    mae_image = Image.fromarray(mae_array, mode="RGB")
    logger.info("✅ Generated MAE image using mean color fill fallback")
    
    return mae_image


def prepare_mask_conditionals(
    original: Image.Image,
    mask: Image.Image,
    include_mae: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Create conditional images for the model: mask, background, object, mae.
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        include_mae: Whether to include MAE image (default: True)
    
    Returns:
        - mask_rgb: RGB mask image
        - background_rgb: Background image (black where mask is, original elsewhere)
        - object_rgb: Object image (original where mask is, black elsewhere)
        - mae_image: MAE image (inpainted preview) if include_mae=True, otherwise same as background
    """

    # Resize mask to match original image size if needed
    if mask.size != original.size:
        logger.info(f"🔄 Resizing mask from {mask.size} to {original.size}")
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)

    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)

    original_array = np.array(original, dtype=np.float32) / 255.0
    if mask_array.shape != original_array.shape[:2]:
        raise ValueError("Mask and original image must have the same spatial size")

    mask_stack = np.repeat(mask_array[..., None], 3, axis=2)

    # Create object image: keep original where mask is, black (0,0,0) elsewhere
    object_rgb = original_array * mask_stack
    object_img = Image.fromarray(np.uint8(object_rgb * 255), mode="RGB")

    # Create background image: keep original where mask is NOT, black (0,0,0) where mask is
    background_rgb = original_array * (1.0 - mask_stack)
    background_img = Image.fromarray(np.uint8(background_rgb * 255), mode="RGB")

    # Mask RGB (for model input, keep as RGB)
    mask_rgb = mask.convert("RGB")

    # Generate MAE image if requested
    if include_mae:
        mae_image = generate_mae_image(original, mask)
        logger.info("🧩 Generated conditional images (mask/background/object/mae)")
    else:
        # Fallback: use background image as MAE (backward compatibility)
        mae_image = background_img
        logger.info("🧩 Generated conditional images (mask/background/object) - MAE disabled")

    return mask_rgb, background_img, object_img, mae_image

