"""
Rembg service for background removal.
Uses rembg library with U2-Net model for automatic background removal.
"""
import io
import logging
from PIL import Image
from rembg import remove

logger = logging.getLogger(__name__)

# Maximum image size for processing (to optimize RAM and speed)
MAX_IMAGE_SIZE = 1024


def remove_background(image_bytes: bytes) -> bytes:
    """
    Remove background from an image using rembg.
    
    Args:
        image_bytes: Raw image bytes (any format supported by PIL)
        
    Returns:
        bytes: PNG image bytes with transparent background
        
    Raises:
        ValueError: If image_bytes is empty or invalid
        Exception: If rembg processing fails
    """
    if not image_bytes:
        raise ValueError("Image bytes cannot be empty")
    
    try:
        # 1. Convert bytes to PIL Image
        input_image = Image.open(io.BytesIO(image_bytes))
        original_size = input_image.size
        logger.info(f"Processing image: {original_size[0]}x{original_size[1]}")
        
        # 2. Resize if too large (to optimize RAM and processing speed)
        if max(original_size) > MAX_IMAGE_SIZE:
            # Calculate new size maintaining aspect ratio
            ratio = MAX_IMAGE_SIZE / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to: {new_size[0]}x{new_size[1]}")
        
        # 3. Process with rembg
        # alpha_matting=False: Faster processing, good enough for most cases
        # Set alpha_matting=True if you need extremely smooth edges (slower)
        output_image = remove(
            input_image,
            alpha_matting=False,
        )
        
        # 4. Convert PIL Image to PNG bytes
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        result_bytes = output_buffer.getvalue()
        logger.info(f"Background removal successful. Output size: {len(result_bytes)} bytes")
        
        return result_bytes
        
    except Exception as e:
        logger.error(f"Error removing background: {str(e)}", exc_info=True)
        raise Exception(f"Failed to remove background: {str(e)}")

