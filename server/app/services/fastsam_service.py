"""
FastSAM service for smart mask generation.
Uses FastSAM-s model (smallest, fastest) with lazy loading.
"""
import base64
import io
import logging
import psutil
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
from ultralytics import FastSAM  # type: ignore
from skimage.morphology import binary_dilation
from skimage.morphology.footprints import rectangle
from skimage.filters import gaussian

logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_fastsam_model: Optional[FastSAM] = None
_model_path = "FastSAM-s.pt"  # Will auto-download on first use


def get_model() -> FastSAM:
    """
    Get or load the FastSAM model (lazy loading).
    Model is only loaded on first request.
    """
    global _fastsam_model
    
    if _fastsam_model is None:
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        if available_gb < 2:
            logger.warning(
                f"Low available memory: {available_gb:.2f}GB. "
                "FastSAM may run slowly or fail."
            )
        
        logger.info("Loading FastSAM-s model (this may take a moment on first use)...")
        try:
            # FastSAM-s will auto-download if not present
            _fastsam_model = FastSAM(_model_path)
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
) -> str:
    """
    Generate a smart mask using FastSAM.
    
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
    
    # Load model
    model = get_model()
    
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
    
    # Apply dilation if requested (using scikit-image)
    if dilate_amount > 0:
        # Create binary mask for dilation (0 or 1)
        mask_bool = (mask_binary > 128).astype(bool)
        # Create structuring element (square kernel using rectangle)
        kernel_size = dilate_amount * 2 + 1
        footprint = rectangle(kernel_size, kernel_size)
        # Apply binary dilation (scikit-image uses 'footprint' parameter)
        mask_dilated = binary_dilation(mask_bool, footprint=footprint)
        # Convert back to uint8
        mask_binary = mask_dilated.astype(np.uint8) * 255
    
    # Apply Gaussian blur for soft edges if requested (using scikit-image)
    if use_blur:
        # Convert to float for blur, apply gaussian filter, convert back
        mask_float = mask_binary.astype(np.float32) / 255.0
        mask_blurred = gaussian(mask_float, sigma=3.0)  # sigma=3 approximates (21,21) kernel
        mask_binary = (mask_blurred * 255).astype(np.uint8)
    
    # Convert to PIL Image and encode as base64
    mask_image = Image.fromarray(mask_binary, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    mask_image.save(buffer, format='PNG')
    mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return mask_base64

