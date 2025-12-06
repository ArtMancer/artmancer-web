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
        PNG image with transparent background where mask is black
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
        
        # Convert to numpy arrays for faster processing
        image_array = np.array(image.convert("RGBA"), dtype=np.uint8)
        mask_array = np.array(mask.convert("L"), dtype=np.uint8)
        
        # Apply mask as alpha channel using numpy (much faster than PIL split/merge)
        # White in mask = opaque, Black in mask = transparent
        result_array = image_array.copy()
        result_array[:, :, 3] = mask_array  # Set alpha channel to mask
        
        # Convert back to PIL Image
        result = Image.fromarray(result_array, mode="RGBA")
        
        # Save to buffer with optimization
        buffer = io.BytesIO()
        # Use optimize=False for faster encoding (slight size trade-off)
        result.save(buffer, format="PNG", optimize=False)
        buffer.seek(0)
        
        # Encode as base64
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        logger.info(f"Object extracted successfully")
        
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

