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
        #
        # Đưa ảnh (sau khi resize nếu có) về PNG bytes rồi truyền vào rembg.
        # rembg.remove(...) sẽ trả về bytes, phù hợp với type hint (không còn gọi .save trên bytes).
        buffer = io.BytesIO()
        input_image.save(buffer, format="PNG")
        buffer.seek(0)

        result = remove(
            buffer.getvalue(),
            alpha_matting=False,
        )

        # Chuẩn hoá kết quả về bytes để khớp type hint
        if isinstance(result, bytes):
            result_bytes = result
        elif isinstance(result, Image.Image):
            out_buf = io.BytesIO()
            result.save(out_buf, format="PNG")
            out_buf.seek(0)
            result_bytes = out_buf.getvalue()
        else:
            # rembg cũng có thể trả về ndarray; chuyển sang PNG bytes
            img = Image.fromarray(result)
            out_buf = io.BytesIO()
            img.save(out_buf, format="PNG")
            out_buf.seek(0)
            result_bytes = out_buf.getvalue()

        logger.info("Background removal successful. Output size: %d bytes", len(result_bytes))

        return result_bytes
        
    except Exception as e:
        logger.error(f"Error removing background: {str(e)}", exc_info=True)
        raise Exception(f"Failed to remove background: {str(e)}")

