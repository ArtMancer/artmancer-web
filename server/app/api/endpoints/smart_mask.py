"""
Smart mask generation endpoint using FastSAM.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.fastsam_service import generate_smart_mask
from app.services.image_cache import cache_image, get_cached_image, cleanup_expired_images

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/smart-mask", tags=["smart-mask"])


class SmartMaskRequest(BaseModel):
    """Request model for smart mask generation."""
    image: Optional[str] = None  # Base64 encoded image (only on first request)
    image_id: Optional[str] = None  # Cached image ID (for subsequent requests)
    bbox: Optional[List[float]] = None  # [x_min, y_min, x_max, y_max]
    points: Optional[List[List[float]]] = None  # [[x, y], ...]
    dilate_amount: int = 10  # Dilation amount in pixels
    use_blur: bool = False  # Apply Gaussian blur for soft edges


class SmartMaskResponse(BaseModel):
    """Response model for smart mask generation."""
    success: bool
    mask_base64: str  # Base64 encoded mask image
    image_id: Optional[str] = None  # Image ID if image was cached
    error: Optional[str] = None


@router.post("", response_model=SmartMaskResponse)
async def generate_smart_mask_endpoint(request: SmartMaskRequest):
    """
    Generate a smart mask using FastSAM.
    
    Accepts either:
    - image (base64): First request, image will be cached
    - image_id: Subsequent requests using cached image
    
    Requires either:
    - bbox: [x_min, y_min, x_max, y_max]
    - points: [[x, y], ...] (takes priority over bbox)
    """
    try:
        # Cleanup expired images periodically
        cleanup_expired_images()
        
        # Get or cache image
        image_path = None
        image_id = request.image_id
        
        if request.image:
            # Cache the image and get image_id
            try:
                image_id = cache_image(request.image)
                logger.info(f"Cached image with ID: {image_id}")
            except Exception as e:
                logger.error(f"Failed to cache image: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to cache image: {str(e)}")
        
        if image_id:
            image_path = get_cached_image(image_id)
            if not image_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Image ID not found or expired: {image_id}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'image' or 'image_id' must be provided"
            )
        
        # Validate prompts
        if not request.points and not request.bbox:
            raise HTTPException(
                status_code=400,
                detail="Either 'points' or 'bbox' must be provided"
            )
        
        # Validate bbox format
        if request.bbox and len(request.bbox) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox must have 4 values: [x_min, y_min, x_max, y_max]"
            )
        
        # Validate points format
        if request.points:
            for point in request.points:
                if not isinstance(point, list) or len(point) != 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Each point must be [x, y]"
                    )
        
        # Generate mask
        try:
            mask_base64 = generate_smart_mask(
                image_path=image_path,
                bbox_coords=request.bbox,
                points=request.points,
                dilate_amount=request.dilate_amount,
                use_blur=request.use_blur,
            )
            
            return SmartMaskResponse(
                success=True,
                mask_base64=mask_base64,
                image_id=image_id,
            )
        except ValueError as e:
            logger.error(f"FastSAM error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in mask generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Mask generation failed: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

