"""
Smart mask generation endpoint using FastSAM or BiRefNet.
"""
import asyncio
import logging
import uuid
import base64
import io
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image

from app.services.mask_segmentation_service import generate_smart_mask, set_cancelled, clear_cancellation
from app.services.image_cache import cache_image, get_cached_image, cleanup_expired_images
from app.services.image_processing import base64_to_image

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
    auto_detect: bool = False  # Auto-detect mode: if True and no bbox/points provided, auto-detect main object
    mask: Optional[str] = None  # Base64 encoded guidance mask (white = area to focus on)
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
    model_type: str = "segmentation"  # "segmentation" (FastSAM) or "birefnet"


class AutoDetectResponse(BaseModel):
    """Response model for auto object detection."""
    success: bool
    mask: Optional[str] = None  # Base64 encoded mask (white = detected object)
    error: Optional[str] = None


def _convert_mask_to_points(mask_b64: str, image: Image.Image) -> Optional[List[List[float]]]:
    """
    Convert guidance mask to centroid point for brush guidance.
    
    Args:
        mask_b64: Base64 encoded mask image
        image: PIL Image for size reference
    
    Returns:
        List of points [[x, y]] or None if conversion fails
    """
    try:
        mask_data = base64.b64decode(mask_b64)
        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        
        # Resize mask if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        # Find centroid of white area
        mask_array = np.array(mask)
        white_pixels = np.where(mask_array > 100)
        
        if len(white_pixels[0]) == 0:
            return None
        
        center_y = float(np.mean(white_pixels[0]))
        center_x = float(np.mean(white_pixels[1]))
        return [[center_x, center_y]]
    except Exception as e:
        logger.warning(f"Failed to convert mask to points: {e}")
        return None


def _get_image_from_request(request: SmartMaskRequest, image_path: Optional[str]) -> Image.Image:
    """
    Get PIL Image from request (either from base64 or cached path).
    
    Args:
        request: SmartMaskRequest
        image_path: Optional cached image path
    
    Returns:
        PIL Image in RGB mode
    """
    if request.image:
        return base64_to_image(request.image)
    elif image_path:
        return Image.open(image_path).convert("RGB")
    else:
        raise ValueError("No image source available")


def _validate_bbox(bbox: Optional[List[float]]) -> None:
    """Validate bbox format."""
    if bbox and len(bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="bbox must have 4 values: [x_min, y_min, x_max, y_max]"
        )


def _validate_points(points: Optional[List[List[float]]]) -> None:
    """Validate points format."""
    if points:
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each point must be [x, y]"
                )


def _normalize_model_type(model_type: Optional[str]) -> str:
    """
    Normalize model type string (support legacy "fastsam" name).
    
    Args:
        model_type: Model type string
    
    Returns:
        Normalized model type ("segmentation" or "birefnet")
    """
    if not model_type:
        return "segmentation"
    
    normalized = model_type.lower()
    if normalized == "fastsam":
        return "segmentation"
    return normalized


def _prepare_mask_inputs(
    request: SmartMaskRequest,
    image_path: Optional[str]
) -> Tuple[Optional[List[float]], Optional[List[List[float]]], str]:
    """
    Prepare bbox and points inputs from request, handling mask conversion and auto-detect.
    
    Args:
        request: SmartMaskRequest
        image_path: Optional cached image path
    
    Returns:
        Tuple of (bbox_to_use, points_to_use, normalized_model_type)
    """
    bbox_to_use = request.bbox
    points_to_use = request.points
    model_type = _normalize_model_type(request.model_type)
    
    # Convert mask to points if mask is provided and no points/bbox given
    # This is needed for BiRefNet with brush input (mask guidance)
    mask_converted_to_points = False
    if request.mask and not request.points and not request.bbox:
        image = _get_image_from_request(request, image_path)
        points = _convert_mask_to_points(request.mask, image)
        
        if points:
            points_to_use = points
            mask_converted_to_points = True
            logger.info(f"Converted mask to points (centroid): {points_to_use}")
        else:
            # Mask is empty or has no white pixels - fallback to auto_detect with mask guidance
            logger.warning("Mask provided but has no white pixels, falling back to auto_detect")
            request.auto_detect = True
    
    # Handle auto-detect mode
    if request.auto_detect and not mask_converted_to_points and not request.points and not request.bbox:
        if not points_to_use:
            image = _get_image_from_request(request, image_path)
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
    
    # Validate formats
    _validate_bbox(bbox_to_use)
    _validate_points(points_to_use)
    
    # Validate model_type-specific requirements
    if model_type == "birefnet":
        # BiRefNet supports points (brush) - it will run FastSAM first to get bbox
        # Only require bbox if no points are provided
        if not points_to_use or len(points_to_use) == 0:
            if not bbox_to_use or len(bbox_to_use) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="BiRefNet requires either points (brush strokes) or bbox [x_min, y_min, x_max, y_max]"
                )
    
    return bbox_to_use, points_to_use, model_type


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
        
        # Prepare mask inputs (handles mask conversion, auto-detect, validation)
        bbox_to_use, points_to_use, model_type = _prepare_mask_inputs(request, image_path)
        
        # Generate unique request_id for cancellation tracking
        request_id = str(uuid.uuid4())
        
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
                request_id=request_id,
                model_type=model_type,
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
                request_id=request_id,
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
        border_adjustment=request.border_adjustment if request.border_adjustment != 0 else 5,
        use_blur=False,
        auto_detect=True,
        mask=request.mask,
        model_type=request.model_type,
    )
    
    # Call the main endpoint
    try:
        result = await generate_smart_mask_endpoint(smart_mask_request)
        return AutoDetectResponse(
            success=result.success,
            mask=result.mask_base64 if result.success else None,
            error=result.error,
        )
    except HTTPException as e:
        error_msg = str(e.detail)
        if isinstance(e.detail, dict):
            error_msg = str(e.detail.get("error", error_msg))
        return AutoDetectResponse(success=False, error=error_msg)
    except Exception as e:
        logger.error(f"Auto-detect failed: {e}", exc_info=True)
        return AutoDetectResponse(success=False, error=str(e))


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
