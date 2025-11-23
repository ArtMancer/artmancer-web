"""
Image utility endpoints for image processing operations.
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from app.services.rembg_service import remove_background

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/image-utils", tags=["image-utils"])


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
        
        # Process removal
        processed_image = remove_background(file_content)
        
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

