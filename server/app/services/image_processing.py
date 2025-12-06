from __future__ import annotations

import base64
import io
import logging
from typing import Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Optional import for cv2 (OpenCV) – required for MAE, NO fallback
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.error(
        "❌ OpenCV library (cv2) is not installed. "
        "MAE image generation requires OpenCV. "
        "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package in the runtime environment."
    )

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
    
    # OpenCV inpainting expects mask where region to inpaint is non-zero (white)
    # Our mask is already white where object/edit area is, so use it directly (no inversion needed)
    mask_for_inpaint = mask_array
    
    # MAE requires OpenCV; raise clear error if missing
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV library (cv2) is not installed in the runtime environment. "
            "Cannot generate MAE image. "
            "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package, "
            "or disable MAE in conditional preparation logic."
        )
    
    # Use OpenCV inpainting with Telea algorithm (fast and effective)
    # cv2.INPAINT_TELEA is faster than cv2.INPAINT_NS (Navier-Stokes)
    try:
        mae_array = cv2.inpaint(
            original_array,
            mask_for_inpaint,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        mae_image = Image.fromarray(mae_array, mode="RGB")
        logger.info("✅ Generated MAE image using OpenCV Telea inpainting")
        return mae_image
    except Exception as exc:
        logger.error("❌ OpenCV inpainting failed for MAE image: %s", exc)
        raise RuntimeError(
            f"OpenCV inpainting failed while generating MAE image: {exc}"
        ) from exc


def generate_canny_image(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """
    Generate Canny edge detection image.
    
    Uses OpenCV Canny algorithm to extract edges from input image.
    
    Args:
        image: Input RGB image
        low_threshold: Lower threshold for edge detection (default: 100)
        high_threshold: Upper threshold for edge detection (default: 200)
    
    Returns:
        Canny edge image: RGB image with white edges on black background
    """
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV library (cv2) is not installed in the runtime environment. "
            "Cannot generate Canny edge image. "
            "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package."
        )
    
    try:
        # Convert to numpy array
        image_array = np.array(image, dtype=np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert to RGB (white edges on black background)
        canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        canny_image = Image.fromarray(canny_rgb, mode="RGB")
        
        logger.info("✅ Generated Canny edge image")
        return canny_image
    except Exception as exc:
        logger.error("❌ Canny edge detection failed: %s", exc)
        raise RuntimeError(
            f"Canny edge detection failed: {exc}"
        ) from exc


def prepare_mask_conditionals(
    original: Image.Image,
    mask: Image.Image,
    include_mae: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Create conditional images for the model: mask, masked_bg, masked_object, mae.
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        include_mae: Whether to include MAE image (default: True)
    
    Returns:
        - mask_rgb: RGB mask image
        - masked_bg: Background with mask area removed (original - mask region = black)
        - masked_object: Object extracted from mask area (NOT used for insertion - use ref_img instead)
        - mae_image: MAE inpainted preview if include_mae=True, otherwise same as masked_bg
    
    Note:
        For insertion task, do NOT use masked_object. Use ref_img (reference image) from frontend instead.
        masked_object is only kept for backward compatibility with removal task debugging.
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

    # Create masked_object: keep original where mask is, black (0,0,0) elsewhere
    # NOTE: This is NOT used for insertion - use ref_img from frontend instead
    masked_object_array = original_array * mask_stack
    masked_object = Image.fromarray(np.uint8(masked_object_array * 255), mode="RGB")

    # Create masked_bg: keep original where mask is NOT, black (0,0,0) where mask is
    # This is the background with the mask area removed
    masked_bg_array = original_array * (1.0 - mask_stack)
    masked_bg = Image.fromarray(np.uint8(masked_bg_array * 255), mode="RGB")

    # Mask RGB (for model input, keep as RGB)
    mask_rgb = mask.convert("RGB")

    # Generate MAE image if requested
    if include_mae:
        mae_image = generate_mae_image(original, mask)
        logger.info("🧩 Generated conditional images (mask/masked_bg/masked_object/mae)")
    else:
        # Fallback: use masked_bg as MAE (backward compatibility)
        mae_image = masked_bg
        logger.info("🧩 Generated conditional images (mask/masked_bg/masked_object) - MAE disabled")

    return mask_rgb, masked_bg, masked_object, mae_image

