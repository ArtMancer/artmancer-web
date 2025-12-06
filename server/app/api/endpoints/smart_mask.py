"""
Smart mask generation endpoint using FastSAM.
"""
import asyncio
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
        
        # Generate mask (run CPU/GPU intensive work in thread pool)
        try:
            mask_base64 = await asyncio.to_thread(
                generate_smart_mask,
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


@router.post("/detect", response_model=AutoDetectResponse)
async def auto_detect_object_endpoint(request: AutoDetectRequest):
    """
    Auto-detect main object in image using FastSAM.
    
    If mask is provided, uses it as guidance to focus detection on masked area.
    Otherwise, auto-detects the main/largest object in the image.
    
    Returns:
        Binary mask where white = detected object
    """
    import base64
    import io
    import numpy as np
    from PIL import Image
    
    try:
        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        logger.info(f"Auto-detecting object in image: {image.size}")
        
        # If mask provided, find centroid to use as prompt
        points = None
        if request.mask and not request.auto_detect:
            mask_data = base64.b64decode(request.mask)
            mask = Image.open(io.BytesIO(mask_data)).convert("L")
            
            # Resize mask if needed
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            
            # Find centroid of white area
            mask_array = np.array(mask)
            white_pixels = np.where(mask_array > 100)
            
            if len(white_pixels[0]) > 0:
                center_y = int(np.mean(white_pixels[0]))
                center_x = int(np.mean(white_pixels[1]))
                points = [[center_x, center_y]]
                logger.info(f"Using mask centroid as prompt: {points}")
        
        # Cache image for FastSAM
        image_id = await asyncio.to_thread(cache_image, request.image)
        image_path = await asyncio.to_thread(get_cached_image, image_id)
        
        if not image_path:
            return AutoDetectResponse(
                success=False,
                error="Failed to cache image"
            )
        
        # Generate mask
        if points:
            # Use points as prompt
            mask_base64 = await asyncio.to_thread(
                generate_smart_mask,
                image_path=image_path,
                points=points,
                dilate_amount=5,
            )
        else:
            # Auto-detect: use center point as initial prompt, or bbox for whole image
            # Use a bbox covering most of the image to get the main object
            w, h = image.size
            bbox = [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
            
            mask_base64 = await asyncio.to_thread(
                generate_smart_mask,
                image_path=image_path,
                bbox_coords=bbox,
                dilate_amount=5,
            )
        
        return AutoDetectResponse(
            success=True,
            mask=mask_base64
        )
        
    except Exception as e:
        logger.error(f"Auto-detect failed: {e}", exc_info=True)
        return AutoDetectResponse(
            success=False,
            error=str(e)
        )

