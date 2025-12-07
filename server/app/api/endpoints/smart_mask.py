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
    border_adjustment: int = 0  # Border adjustment in pixels (negative = shrink, positive = grow)
    use_blur: bool = False  # Apply Gaussian blur for soft edges
    # Auto-detect mode: if True and no bbox/points provided, auto-detect main object
    auto_detect: bool = False
    # Guidance mask for auto-detect (white = area to focus on)
    mask: Optional[str] = None  # Base64 encoded guidance mask
    # Model type: "segmentation" (FastSAM) or "birefnet" (default: "segmentation")
    # Note: BiRefNet only supports bbox (not points) and requires cropping
    model_type: str = "segmentation"  # "segmentation" (FastSAM) or "birefnet"


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
    border_adjustment: int = 0  # Border adjustment in pixels (negative = shrink, positive = grow)
    model_type: str = "segmentation"  # "segmentation" (FastSAM) or "birefnet" (default: "segmentation")


class AutoDetectResponse(BaseModel):
    """Response model for auto object detection."""
    success: bool
    mask: Optional[str] = None  # Base64 encoded mask (white = detected object)
    error: Optional[str] = None


@router.post("", response_model=SmartMaskResponse)
async def generate_smart_mask_endpoint(request: SmartMaskRequest):
    """
    Generate a smart mask using FastSAM or BiRefNet.
    
    Accepts either:
    - image (base64): First request, image will be cached
    - image_id: Subsequent requests using cached image
    
    Requires either:
    - bbox: [x_min, y_min, x_max, y_max]
    - points: [[x, y], ...] (takes priority over bbox)
    - auto_detect=True: Auto-detect main object (uses mask as guidance if provided)
    
    Model types:
    - "segmentation" (default): FastSAM - supports points and bbox
    - "birefnet": BiRefNet - only supports bbox, requires cropping
    """
    import time
    start_time = time.time()
    try:
        logger.info(f"📥 [SmartMask] Received request with model_type={request.model_type}")
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
        
        # Convert mask to points if mask is provided and no points/bbox given
        # This is needed for BiRefNet with brush input (mask guidance)
        mask_converted_to_points = False
        if request.mask and not request.points and not request.bbox:
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
            
            # Convert mask to points (centroid) for brush guidance
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
                    mask_converted_to_points = True
                    logger.info(f"Converted mask to points (centroid): {points_to_use}")
                else:
                    # Mask is empty or has no white pixels - fallback to auto_detect with mask guidance
                    logger.warning("Mask provided but has no white pixels, falling back to auto_detect with mask guidance")
                    request.auto_detect = True
            except Exception as e:
                logger.warning(f"Failed to process guidance mask: {e}, falling back to auto_detect")
                request.auto_detect = True
            
        # Only use auto_detect if mask conversion didn't succeed and no points/bbox were provided
        if request.auto_detect and not mask_converted_to_points and not request.points and not request.bbox:
            # Auto-detect: use default bbox if no mask was provided
            if not points_to_use:
                from app.services.image_processing import base64_to_image
                if request.image:
                    image = base64_to_image(request.image)
                else:
                    from PIL import Image
                    image = Image.open(image_path).convert("RGB")
                
                w, h = image.size
                bbox_to_use = [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
                logger.info(f"Auto-detect: using default bbox {bbox_to_use}")
                # Set border_adjustment to 5 for auto-detect (softer)
                if request.border_adjustment == 0:
                    request.border_adjustment = 5
        else:
            # Manual mode: validate prompts
            if not points_to_use and not bbox_to_use:
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
        
        # Validate model_type and inputs (support both old "fastsam" and new "segmentation")
        model_type = request.model_type.lower() if request.model_type else "segmentation"
        if model_type == "fastsam":
            model_type = "segmentation"  # Map old name to new name
        if model_type == "birefnet":
            # BiRefNet supports points (brush) - it will run FastSAM first to get bbox
            # Only require bbox if no points are provided
            if not points_to_use or len(points_to_use) == 0:
                if not bbox_to_use or len(bbox_to_use) != 4:
                    raise HTTPException(
                        status_code=400,
                        detail="BiRefNet requires either points (brush strokes) or bbox [x_min, y_min, x_max, y_max]"
                    )
        
        # Generate mask (run CPU/GPU intensive work in thread pool)
        try:
            mask_start_time = time.time()
            logger.info(f"🔄 [SmartMask] Starting mask generation with model_type={model_type}")
            mask_base64 = await asyncio.to_thread(
                generate_smart_mask,
                image_path=image_path,
                bbox_coords=bbox_to_use,
                points=points_to_use,
                border_adjustment=request.border_adjustment,
                use_blur=request.use_blur,
                request_id=request_id,  # Pass request_id for cancellation
                model_type=model_type,  # Pass model_type
            )
            mask_duration = time.time() - mask_start_time
            logger.info(f"✅ [SmartMask] Mask generation completed in {mask_duration:.2f}s")
            
            # Clear cancellation flag on success
            clear_cancellation(request_id)
            
            total_duration = time.time() - start_time
            logger.info(f"✅ [SmartMask] Total request time: {total_duration:.2f}s")
            
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
        border_adjustment=request.border_adjustment if request.border_adjustment != 0 else 5,  # Use request value or default to 5
        use_blur=False,
        auto_detect=True,
        mask=request.mask,
        model_type=request.model_type,  # Use model_type from request
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

