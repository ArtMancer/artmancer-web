"""
Mask Segmentation and Reference Object Extraction Service
Combines mask generation (FastSAM) and reference object extraction functionality.

This service handles:
1. Mask generation/segmentation using FastSAM (for both main image and reference image)
2. Reference object extraction using Mask R (for two-source mask workflow)
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING, Any
import numpy as np
from PIL import Image

# psutil and skimage are imported lazily where needed to avoid requiring them in services that don't need them

if TYPE_CHECKING:
    from ultralytics import FastSAM  # type: ignore

logger = logging.getLogger(__name__)

# ============================================================================
# FastSAM Model Management (for mask generation/segmentation)
# ============================================================================

# Global model instance (lazy loaded)
_fastsam_model: Optional[Any] = None
_model_path = "FastSAM-s.pt"  # FastSAM-s: lighter, faster (~40MB)

# Cancellation flags for segmentation requests (shared across instances)
# Key: request_id (string), Value: True if cancelled
_cancellation_flags: dict[str, bool] = {}


def _check_cancellation(request_id: str) -> bool:
    """Check if segmentation request is cancelled."""
    return _cancellation_flags.get(request_id, False)


def set_cancelled(request_id: str) -> None:
    """Mark segmentation request as cancelled."""
    _cancellation_flags[request_id] = True


def clear_cancellation(request_id: str) -> None:
    """Clear cancellation flag for request."""
    _cancellation_flags.pop(request_id, None)

# Modal Volume paths (if running in Modal environment)
VOL_MOUNT_PATH = "/checkpoints"
VOL_FASTSAM_PATH = f"{VOL_MOUNT_PATH}/fastsam"  # Load trực tiếp từ Volume (không copy)


def _get_fastsam_model_path() -> str:
    """
    Determine the best path to load FastSAM model from.
    Priority: Volume (load trực tiếp) → Default (auto-download)
    
    Load trực tiếp từ Volume (không copy) để giảm cold-start, giống Qwen checkpoints.
    
    Returns:
        Path to FastSAM model file
    """
    # Check if running in Modal environment (Volume exists)
    vol_fastsam_file = f"{VOL_FASTSAM_PATH}/FastSAM-s.pt"
    if os.path.exists(vol_fastsam_file):
        # Load trực tiếp từ Volume (không copy để giảm cold-start)
        logger.info(f"🚀 FastSAM-s.pt loaded directly from Volume: {vol_fastsam_file}")
        return vol_fastsam_file
    else:
        # Not in Modal or Volume not set up - use default (will auto-download)
        logger.info("⚠️ FastSAM-s.pt not in Volume, will use default path (auto-download if needed)")
        return _model_path


def get_fastsam_model():
    """
    Get or load the FastSAM model (lazy loading).
    Model is only loaded on first request.
    
    Strategy:
    1. Check Modal Volume for cached model
    2. Copy to local SSD for faster I/O
    3. Fallback to default path (auto-download from GitHub)
    
    Returns:
        FastSAM model instance
    """
    # Lazy import to avoid dependency in HeavyService
    try:
        from ultralytics import FastSAM  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ultralytics package is required for FastSAM. "
            "Install it with: pip install ultralytics"
        ) from e
    
    global _fastsam_model
    
    if _fastsam_model is None:
        # Check available memory (lazy import psutil)
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
        except ImportError:
            # psutil not available, skip memory check
            logger.warning("psutil not available, skipping memory check")
            available_gb = float('inf')
        
        if available_gb < 2:
            logger.warning(
                f"Low available memory: {available_gb:.2f}GB. "
                "FastSAM may run slowly or fail."
            )
        
        # Get model path (check Volume first)
        model_path = _get_fastsam_model_path()
        
        logger.info(f"Loading FastSAM-s model from: {model_path} (this may take a moment on first use)...")
        try:
            # FastSAM-s will auto-download if not present (~40MB)
            _fastsam_model = FastSAM(model_path)
            logger.info("FastSAM-s model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FastSAM model: {e}")
            raise RuntimeError(f"Failed to load FastSAM model: {e}")
    
    return _fastsam_model


def generate_smart_mask(
    image_path: str,
    bbox_coords: Optional[List[float]] = None,
    points: Optional[List[List[float]]] = None,
    dilate_amount: int = 10,
    use_blur: bool = False,
    request_id: Optional[str] = None,  # Request ID for cancellation tracking
) -> str:
    """
    Generate a smart mask using FastSAM (segmentation).
    
    This function can be used for:
    - Generating Mask A (placement mask) on main input image
    - Generating Mask R (object isolation mask) on reference image
    
    Args:
        image_path: Path to the input image
        bbox_coords: Bounding box [x_min, y_min, x_max, y_max] (optional)
        points: List of points [[x, y], ...] (optional, takes priority over bbox)
        dilate_amount: Amount to dilate the mask (pixels)
        use_blur: Whether to apply Gaussian blur for soft edges
        
    Returns:
        Base64 encoded mask image (PNG format)
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check cancellation before starting
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled before model loading: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Load model
    model = get_fastsam_model()
    
    # Check cancellation after model loading
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled after model loading: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Read image using PIL (no need for cv2)
    try:
        pil_image = Image.open(image_path)
        # Convert to RGB if needed (handles RGBA, P, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image_rgb = np.array(pil_image, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to read image: {image_path} - {e}")
    
    height, width = image_rgb.shape[:2]
    
    # Prepare prompts for FastSAM
    # FastSAM expects prompts in format: points or boxes
    results = None
    
    if points and len(points) > 0:
        # Use point-based prompts (more accurate)
        # FastSAM expects points as list of lists: [[x, y], ...]
        # and labels as list: [1, ...] (1 = positive point, 0 = negative point)
        point_list = [[float(p[0]), float(p[1])] for p in points]
        label_list = [1] * len(points)  # All positive points
        
        logger.info(f"Using {len(points)} points for mask generation")
        results = model(image_rgb, points=point_list, labels=label_list)
        
    elif bbox_coords and len(bbox_coords) == 4:
        # Use bounding box prompt
        # FastSAM expects bboxes as flat list: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox_coords
        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        
        # Use flat list format as per Ultralytics FastSAM API
        bboxes = [x_min, y_min, x_max, y_max]
        
        logger.info(f"Using bbox [{x_min}, {y_min}, {x_max}, {y_max}] for mask generation")
        results = model(image_rgb, bboxes=bboxes)
    
    else:
        raise ValueError("Either bbox_coords or points must be provided")
    
    # Check cancellation after mask generation
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled after mask generation: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    if not results or len(results) == 0:
        raise ValueError("FastSAM did not generate any masks")
    
    # Get the first (most confident) mask
    # FastSAM returns results in a specific format
    masks = results[0].masks
    if masks is None or len(masks.data) == 0:
        raise ValueError("No masks found in FastSAM results")
    
    # Get the first mask
    mask_data = masks.data[0].cpu().numpy()
    
    # Log mask shape for debugging
    if mask_data.shape != (height, width):
        logger.warning(
            f"Mask shape {mask_data.shape} does not match image size ({height}, {width}). "
            "Resizing mask to match image dimensions."
        )
        # Resize mask using PIL (nearest neighbor for binary mask)
        mask_pil = Image.fromarray((mask_data * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        mask_data = np.array(mask_pil, dtype=np.float32) / 255.0
    
    # Convert to uint8 binary mask
    mask_binary = (mask_data > 0.5).astype(np.uint8) * 255
    
    # Apply morphological operations for smoother, softer mask (similar to manual stroke)
    # Manual stroke uses: lineCap='round', lineJoin='round', imageSmoothingQuality='high'
    try:
        from skimage.morphology import (
            binary_dilation, binary_closing, binary_opening
        )
        # Try to use new API (disk footprint) or fallback to deprecated API
        try:
            from skimage.morphology import disk
        except ImportError:
            # Fallback to deprecated API
            from skimage.morphology.footprints import disk
    except ImportError:
        raise RuntimeError(
            "scikit-image is required for mask processing. "
            "Install it with: pip install scikit-image"
        )
    
    # Check cancellation before mask processing
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled before mask processing: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Step 1: Apply opening (erosion + dilation) to remove small noise and smooth edges
    # This makes the mask cleaner and less jagged (similar to imageSmoothingEnabled)
    # Use larger radius for smoother initial cleanup
    opening_footprint = disk(radius=3)  # Increased from 2 to 3 for smoother edges
    
    mask_bool = (mask_binary > 128).astype(bool)
    mask_opened = binary_opening(mask_bool, footprint=opening_footprint)
    mask_binary = mask_opened.astype(np.uint8) * 255
    
    # Check cancellation after opening
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled during mask processing: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Step 2: Apply dilation if requested (using disk footprint for round, soft edges)
    # Similar to brush with round lineCap and lineJoin in manual stroke
    if dilate_amount > 0:
        # Use disk footprint instead of rectangle for softer, rounder edges
        # This mimics the round brush used in manual stroke (lineCap='round')
        dilation_radius = max(1, dilate_amount // 2)  # Convert pixel amount to radius
        footprint = disk(radius=dilation_radius)
        
        # Apply dilation with disk footprint (creates round, soft expansion)
        mask_bool = (mask_binary > 128).astype(bool)
        mask_dilated = binary_dilation(mask_bool, footprint=footprint)
        mask_binary = mask_dilated.astype(np.uint8) * 255
        
        logger.info(f"🎨 Applied dilation with disk footprint (radius={dilation_radius}) for soft edges")
    
    # Step 3: Apply median filter to remove noise and smooth edges (before blur)
    # This helps create smoother outlines by removing small irregularities
    try:
        from skimage.filters.rank import median
        from skimage.morphology import disk as median_disk
        # Use small kernel for gentle smoothing without losing detail
        # Convert to uint8 if needed
        mask_binary_uint8 = mask_binary.astype(np.uint8)
        # Apply median filter with small disk footprint
        mask_binary = median(mask_binary_uint8, median_disk(radius=1))
        logger.debug("Applied median filter for noise reduction")
    except (ImportError, AttributeError) as e:
        # Fallback if median filter not available (shouldn't happen, but safe)
        logger.debug(f"Median filter not available, skipping: {e}")
    
    # Step 4: Apply Gaussian blur for soft, feathered edges (similar to imageSmoothingQuality='high')
    # Always apply blur for FastSAM masks to make them softer and more natural
    # This mimics the smooth rendering of manual stroke
    try:
        from skimage.filters import gaussian
    except ImportError:
        raise RuntimeError(
            "scikit-image is required for Gaussian blur. "
            "Install it with: pip install scikit-image"
        )
    
    # Apply stronger Gaussian blur for softer edges (similar to high-quality smoothing)
    # Use adaptive sigma based on image size for better quality
    # Increased base sigma for smoother outlines
    base_sigma = max(3.0, min(width, height) / 200.0)  # Increased from 2.0 and /256 to /200
    if use_blur:
        # User requested blur, use stronger sigma
        blur_sigma = max(4.0, base_sigma * 1.5)  # Increased from 3.0
    else:
        blur_sigma = base_sigma
    
    # Convert to float for blur, then back to uint8
    mask_float = mask_binary.astype(np.float32) / 255.0
    mask_blurred = gaussian(mask_float, sigma=blur_sigma, preserve_range=True)
    
    # Step 5: Apply gentle closing to smooth out any remaining jagged edges
    # This creates a more cohesive, natural-looking mask
    # Use larger closing footprint for smoother result
    closing_footprint = disk(radius=2)  # Increased from 1 to 2
    
    # Use adaptive threshold after blur (lower threshold preserves more detail while smoothing)
    threshold = 0.25  # Lowered from 0.3 for smoother edges
    mask_bool = (mask_blurred > threshold).astype(bool)
    mask_closed = binary_closing(mask_bool, footprint=closing_footprint)
    
    # Step 6: Final blur pass for extra smoothness (like multiple smoothing passes)
    # Use stronger final blur for ultra-smooth outlines
    mask_closed_float = mask_closed.astype(np.float32)
    final_blur_sigma = blur_sigma * 0.7  # Increased from 0.5 to 0.7 for smoother result
    mask_final = gaussian(mask_closed_float, sigma=final_blur_sigma, preserve_range=True)
    
    # Step 7: Optional gentle erosion to refine edges (makes outline cleaner)
    # This helps remove any remaining artifacts from blur
    try:
        from skimage.morphology import binary_erosion
        erosion_footprint = disk(radius=1)
        mask_bool_final = (mask_final > 0.4).astype(bool)  # Slightly higher threshold
        mask_eroded = binary_erosion(mask_bool_final, footprint=erosion_footprint)
        # Convert back to float and apply final light blur
        mask_eroded_float = mask_eroded.astype(np.float32)
        mask_final = gaussian(mask_eroded_float, sigma=blur_sigma * 0.3, preserve_range=True)
        logger.debug("Applied final erosion and light blur for edge refinement")
    except Exception as e:
        logger.debug(f"Skipping final erosion: {e}")
    
    # Check cancellation before final encoding
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ FastSAM segmentation cancelled before final encoding: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Convert back to uint8 with proper scaling
    mask_binary = (mask_final * 255).astype(np.uint8)
    
    logger.info(f"🎨 Applied enhanced smooth mask processing: opening(r=3), dilation, median filter, Gaussian blur (σ={blur_sigma:.2f}), closing(r=2), final blur (σ={final_blur_sigma:.2f})")
    
    # Convert to PIL Image and encode as base64
    mask_image = Image.fromarray(mask_binary, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return mask_base64


# ============================================================================
# Reference Object Extraction (for two-source mask workflow)
# ============================================================================

def extract_object_with_mask(
    reference_image: Image.Image,
    mask_R: Image.Image,
) -> Image.Image:
    """
    Extract object from reference image using Mask R.
    
    Applies mask to isolate object, removes background (sets to black).
    Preserves the natural shape and proportions defined by Mask R.
    
    This is used in the two-source mask workflow:
    - Mask R: Object isolation mask on reference image (defines WHAT to insert)
    - This function extracts the object from reference image using Mask R
    
    Args:
        reference_image: Full reference image containing the object
        mask_R: Mask image (white where object is, black elsewhere)
    
    Returns:
        Isolated object image with black background (preserves Mask R shape)
    """
    # Resize mask to match reference image size if needed
    if mask_R.size != reference_image.size:
        logger.info(f"🔄 Resizing mask_R from {mask_R.size} to {reference_image.size}")
        mask_R = mask_R.resize(reference_image.size, Image.Resampling.LANCZOS)
    
    # Convert mask to grayscale and normalize
    mask_gray = mask_R.convert("L")
    mask_array = np.array(mask_gray, dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)
    
    # Convert reference image to numpy array
    reference_array = np.array(reference_image.convert("RGB"), dtype=np.float32) / 255.0
    
    # Ensure mask and image have matching spatial dimensions
    if mask_array.shape != reference_array.shape[:2]:
        raise ValueError(
            f"Mask R and reference image must have the same spatial size. "
            f"Got mask: {mask_array.shape[:2]}, image: {reference_array.shape[:2]}"
        )
    
    # Stack mask to 3 channels for element-wise multiplication
    mask_stack = np.repeat(mask_array[..., None], 3, axis=2)
    
    # Extract object: keep reference where mask is, black (0,0,0) elsewhere
    extracted_array = reference_array * mask_stack
    
    # Convert back to PIL Image
    extracted_image = Image.fromarray(
        np.uint8(extracted_array * 255),
        mode="RGB"
    )
    
    logger.info(
        f"✅ Extracted object using Mask R: {extracted_image.size}, "
        f"mask coverage: {np.mean(mask_array) * 100:.1f}%"
    )
    
    return extracted_image


def prepare_reference_conditionals(
    reference_image: Image.Image,
    mask_R: Image.Image,
) -> Tuple[Image.Image, Image.Image]:
    """
    Prepare reference conditionals for model input.
    
    Extracts object using Mask R and creates masked reference object
    (object on black background). Both are returned for conditioning.
    
    This is used in the two-source mask workflow for insertion tasks.
    
    Args:
        reference_image: Full reference image
        mask_R: Object isolation mask (Mask R)
    
    Returns:
        Tuple of (extracted_object, masked_reference_object)
        - extracted_object: Object isolated with black background
        - masked_reference_object: Same as extracted_object (for consistency)
    """
    # Extract object using Mask R
    extracted_object = extract_object_with_mask(reference_image, mask_R)
    
    # For now, masked_reference_object is the same as extracted_object
    # This can be extended later if different processing is needed
    masked_reference_object = extracted_object
    
    logger.info(
        "🧩 Prepared reference conditionals: extracted_object and masked_reference_object "
        f"(both preserve Mask R shape: {mask_R.size})"
    )
    
    return extracted_object, masked_reference_object

