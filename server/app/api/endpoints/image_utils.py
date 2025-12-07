"""
Image utility endpoints for image processing operations.
"""
import base64
import io
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

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
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        # Decode base64 mask
        mask_data = base64.b64decode(request.mask)
        mask = Image.open(io.BytesIO(mask_data)).convert("L")  # Grayscale
        
        # Resize mask if needed (use NEAREST for mask - faster and preserves binary nature)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        logger.info(f"Extracting object: image={image.size}, mask={mask.size}")
        
        # Convert mask to numpy array to find bounding box
        mask_array = np.array(mask, dtype=np.uint8)
        
        # Find bounding box of white pixels (object region)
        white_pixels = np.where(mask_array > 128)
        if len(white_pixels[0]) == 0:
            raise ValueError("No object found in mask (no white pixels)")
        
        y_min, y_max = int(np.min(white_pixels[0])), int(np.max(white_pixels[0])) + 1
        x_min, x_max = int(np.min(white_pixels[1])), int(np.max(white_pixels[1])) + 1
        
        # Crop object from image and mask
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        # Determine target size (1:1 ratio, 512x512 or 1024x1024, not too high)
        # Use the larger dimension of cropped object to determine scale
        cropped_width, cropped_height = cropped_image.size
        max_dimension = max(cropped_width, cropped_height)
        
        # Choose target size: 512x512 if max_dimension <= 512, else 1024x1024
        if max_dimension <= 512:
            target_size = 512
        else:
            target_size = 1024
        
        # Resize cropped object to target size (1:1 ratio) while maintaining aspect ratio
        # Calculate scale to fit within target_size while preserving aspect ratio
        scale = min(target_size / cropped_width, target_size / cropped_height)
        new_width = int(cropped_width * scale)
        new_height = int(cropped_height * scale)
        
        # Resize object image
        resized_object = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create white background (RGB, not RGBA, not transparent, not black)
        result = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
        
        # Calculate position to center the object on white background
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        
        # Paste object onto white background using mask for transparency
        # Convert resized object back to RGBA for alpha compositing
        if resized_object.mode != "RGBA":
            resized_object = resized_object.convert("RGBA")
        
        # Resize mask to match resized object size
        resized_mask = cropped_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create alpha channel from mask
        alpha = resized_mask.convert("L")
        resized_object.putalpha(alpha)
        
        # Paste object onto white background
        result.paste(resized_object, (paste_x, paste_y), resized_object)
        
        # Save to buffer
        buffer = io.BytesIO()
        result.save(buffer, format="PNG", optimize=False)
        buffer.seek(0)
        
        # Encode as base64
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Object extracted successfully: {cropped_image.size} -> {target_size}x{target_size} on white background")
        
        return ExtractObjectResponse(
            success=True,
            extracted_image=result_base64
        )
        
    except Exception as e:
        logger.error(f"Error extracting object: {e}", exc_info=True)
        return ExtractObjectResponse(
            success=False,
            error=str(e)
        )

