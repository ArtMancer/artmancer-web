"""
Smart mask generation endpoint using FastSAM.
"""
import asyncio
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.mask_segmentation_service import generate_smart_mask, set_cancelled, clear_cancellation
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
    # Auto-detect mode: if True and no bbox/points provided, auto-detect main object
    auto_detect: bool = False
    # Guidance mask for auto-detect (white = area to focus on)
    mask: Optional[str] = None  # Base64 encoded guidance mask


class SmartMaskResponse(BaseModel):
    """Response model for smart mask generation."""
    success: bool
    mask_base64: str  # Base64 encoded mask image
    image_id: Optional[str] = None  # Image ID if image was cached
    request_id: Optional[str] = None  # Request ID for cancellation tracking
    error: Optional[str] = None


class AutoDetectRequest(BaseModel):
    """Request model for auto object detection."""
    image: str  # Base64 encoded image
    mask: Optional[str] = None  # Optional guidance mask (white = area to focus on)
    auto_detect: bool = True  # If True, auto-detect main object; if False, use mask as guide


class AutoDetectResponse(BaseModel):
    """Response model for auto object detection."""
    success: bool
    mask: Optional[str] = None  # Base64 encoded mask (white = detected object)
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
    - auto_detect=True: Auto-detect main object (uses mask as guidance if provided)
    """
    try:
        # Cleanup expired images periodically (run in thread pool)
        await asyncio.to_thread(cleanup_expired_images)
        
        # Get or cache image
        image_path = None
        image_id = request.image_id
        
        if request.image:
            # Cache the image and get image_id (run in thread pool)
            try:
                image_id = await asyncio.to_thread(cache_image, request.image)
                logger.info(f"Cached image with ID: {image_id}")
            except Exception as e:
                logger.error(f"Failed to cache image: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to cache image: {str(e)}")
        
        if image_id:
            image_path = await asyncio.to_thread(get_cached_image, image_id)
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
        
        # Handle auto-detect mode
        bbox_to_use = request.bbox
        points_to_use = request.points
        
        if request.auto_detect and not request.points and not request.bbox:
            # Auto-detect: process mask guidance or use default bbox
            import base64
            import io
            import numpy as np
            from PIL import Image
            
            # Get image for size calculation
            from app.services.image_processing import base64_to_image
            if request.image:
                image = base64_to_image(request.image)
            else:
                # Load from cached path
                image = Image.open(image_path).convert("RGB")
            
            # If mask provided, find centroid to use as prompt
            if request.mask:
                try:
                    mask_data = base64.b64decode(request.mask)
                    mask = Image.open(io.BytesIO(mask_data)).convert("L")
                    
                    # Resize mask if needed
                    if mask.size != image.size:
                        mask = mask.resize(image.size, Image.Resampling.NEAREST)
                    
                    # Find centroid of white area
                    mask_array = np.array(mask)
                    white_pixels = np.where(mask_array > 100)
                    
                    if len(white_pixels[0]) > 0:
                        center_y = float(np.mean(white_pixels[0]))
                        center_x = float(np.mean(white_pixels[1]))
                        points_to_use = [[center_x, center_y]]
                        logger.info(f"Auto-detect: using mask centroid as prompt: {points_to_use}")
                except Exception as e:
                    logger.warning(f"Failed to process guidance mask: {e}")
            
            # If no points from mask, use default bbox covering most of image
            if not points_to_use:
                w, h = image.size
                bbox_to_use = [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
                logger.info(f"Auto-detect: using default bbox {bbox_to_use}")
                # Set dilate_amount to 5 for auto-detect (softer)
                if request.dilate_amount == 10:
                    request.dilate_amount = 5
        else:
            # Manual mode: validate prompts
            if not request.points and not request.bbox:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'points', 'bbox', or 'auto_detect=True' must be provided"
                )
        
        # Validate bbox format
        if bbox_to_use and len(bbox_to_use) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox must have 4 values: [x_min, y_min, x_max, y_max]"
            )
        
        # Validate points format
        if points_to_use:
            for point in points_to_use:
                if not isinstance(point, list) or len(point) != 2:
                    raise HTTPException(
                        status_code=400,
                        detail="Each point must be [x, y]"
                    )
        
        # Generate unique request_id for cancellation tracking
        request_id = str(uuid.uuid4())
        
        # Generate mask (run CPU/GPU intensive work in thread pool)
        try:
            mask_base64 = await asyncio.to_thread(
                generate_smart_mask,
                image_path=image_path,
                bbox_coords=bbox_to_use,
                points=points_to_use,
                dilate_amount=request.dilate_amount,
                use_blur=request.use_blur,
                request_id=request_id,  # Pass request_id for cancellation
            )
            
            # Clear cancellation flag on success
            clear_cancellation(request_id)
            
            return SmartMaskResponse(
                success=True,
                mask_base64=mask_base64,
                image_id=image_id,
                request_id=request_id,  # Include request_id for cancellation
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


@router.post("/detect", response_model=AutoDetectResponse)
async def auto_detect_object_endpoint(request: AutoDetectRequest):
    """
    Auto-detect main object in image using FastSAM.
    
    DEPRECATED: Use /smart-mask with auto_detect=True instead.
    This endpoint is kept for backward compatibility.
    
    If mask is provided, uses it as guidance to focus detection on masked area.
    Otherwise, auto-detects the main/largest object in the image.
    
    Returns:
        Binary mask where white = detected object
    """
    # Convert AutoDetectRequest to SmartMaskRequest format
    smart_mask_request = SmartMaskRequest(
        image=request.image,
        image_id=None,
        bbox=None,
        points=None,
        dilate_amount=5,
        use_blur=False,
        auto_detect=True,
        mask=request.mask,
    )
    
    # Call the main endpoint
    try:
        result = await generate_smart_mask_endpoint(smart_mask_request)
        # Convert SmartMaskResponse to AutoDetectResponse format
        return AutoDetectResponse(
            success=result.success,
            mask=result.mask_base64 if result.success else None,
            error=result.error,
        )
    except HTTPException as e:
        return AutoDetectResponse(
            success=False,
            error=str(e.detail) if isinstance(e.detail, str) else str(e.detail.get("error", str(e.detail))) if isinstance(e.detail, dict) else "Unknown error"
        )
    except Exception as e:
        logger.error(f"Auto-detect failed: {e}", exc_info=True)
        return AutoDetectResponse(
            success=False,
            error=str(e)
        )


@router.post("/cancel/{request_id}")
def cancel_smart_mask(request_id: str):
    """
    Cancel a FastSAM segmentation request.
    
    Args:
        request_id: Request identifier (generated by generate_smart_mask_endpoint)
    
    Returns:
        Success message
    """
    try:
        set_cancelled(request_id)
        return {
            "success": True,
            "message": f"Smart mask request {request_id} marked for cancellation",
            "request_id": request_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel smart mask: {str(e)}"
        )

