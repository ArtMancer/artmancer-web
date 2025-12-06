"""
Image utility endpoints for image processing operations.
"""
import asyncio
import base64
import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

from app.services.rembg_service import remove_background

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


@router.post("/remove-bg")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    """
    Remove background from an uploaded image.
    
    Args:
        file: Uploaded image file (any format supported by PIL)
        
    Returns:
        Response: PNG image with transparent background
        
    Raises:
        HTTPException: If file is not an image or processing fails
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image. Supported formats: PNG, JPEG, JPG, WEBP"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        logger.info(f"Processing background removal for file: {file.filename} ({len(file_content)} bytes)")
        
        # Process removal (run CPU/GPU intensive work in thread pool)
        processed_image = await asyncio.to_thread(remove_background, file_content)
        
        logger.info(f"Background removal successful. Returning {len(processed_image)} bytes")
        
        # Return as image/png
        return Response(content=processed_image, media_type="image/png")
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error removing background: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error removing background: {str(e)}"
        )


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
        
        # Resize mask if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        
        logger.info(f"Extracting object: image={image.size}, mask={mask.size}")
        
        # Apply mask as alpha channel
        # White in mask = opaque, Black in mask = transparent
        r, g, b, _ = image.split()
        result = Image.merge("RGBA", (r, g, b, mask))
        
        # Save to buffer
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
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

