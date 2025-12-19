"""
Mask Segmentation and Reference Object Extraction Service

This service handles:
1. Mask generation/segmentation using FastSAM or BiRefNet
   (for both main image and reference image)
2. Reference object extraction using Mask R (for two-source mask workflow)

Architecture:
- FastSAM: Fast, lightweight segmentation model (supports points and bbox)
- BiRefNet: Advanced segmentation model (only supports bbox, requires cropping)
- Both models support cancellation via request_id
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
from PIL import Image

# psutil and skimage are imported lazily where needed

logger = logging.getLogger(__name__)

# ============================================================================
# Model Management (FastSAM and BiRefNet for mask generation/segmentation)
# ============================================================================

# Global model instances (lazy loaded)
_fastsam_model: Optional[Any] = None
_birefnet_model: Optional[Any] = None
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
VOL_SEGMENTATION_PATH = f"{VOL_MOUNT_PATH}/segmentation"  # Load trực tiếp từ Volume (không copy)
VOL_BIREFNET_PATH = f"{VOL_MOUNT_PATH}/birefnet"  # BiRefNet model cache (load trực tiếp từ Volume, không copy)


def _get_segmentation_model_path() -> str:
    """
    Determine the best path to load segmentation model (FastSAM) from.
    Priority: Volume (load trực tiếp) → Default (auto-download)
    
    Load trực tiếp từ Volume (không copy) để giảm cold-start, giống Qwen checkpoints.
    
    Returns:
        Path to segmentation model file
    """
    vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
    if os.path.exists(vol_segmentation_file):
        return vol_segmentation_file
    return _model_path


def get_segmentation_model() -> Any:
    """
    Get or load the segmentation model (FastSAM) (lazy loading).
    Model is only loaded on first request.
    
    Strategy:
    1. Check Modal Volume for cached model
    2. Load trực tiếp từ Volume (không copy) để giảm cold-start
    3. Fallback to default path (auto-download from GitHub)
    
    Returns:
        Segmentation model instance (FastSAM)
    
    Raises:
        RuntimeError: If model loading fails
    """
    # Lazy import to avoid dependency in HeavyService
    try:
        from ultralytics import FastSAM  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ultralytics package is required for segmentation model. "
            "Install it with: pip install ultralytics"
        ) from e
    
    global _fastsam_model
    
    if _fastsam_model is None:
        model_path = _get_segmentation_model_path()
        try:
            _fastsam_model = FastSAM(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation model: {e}")
    
    return _fastsam_model


def _get_birefnet_model_path() -> str:
    """
    Determine the best path to load BiRefNet model from.
    Priority: Volume (load trực tiếp) → Default (auto-download from HuggingFace)
    
    Load trực tiếp từ Volume (không copy) để giảm cold-start, giống Qwen checkpoints.
    
    Returns:
        Path to BiRefNet model directory or repo_id
    """
    if os.path.exists(VOL_BIREFNET_PATH) and os.listdir(VOL_BIREFNET_PATH):
        return VOL_BIREFNET_PATH
    return 'zhengpeng7/BiRefNet'


def get_birefnet_model() -> Any:
    """
    Get or load the BiRefNet model (lazy loading).
    Model is only loaded on first request.
    
    Strategy:
    1. Check Modal Volume for cached model
    2. Load trực tiếp từ Volume (không copy) để giảm cold-start
    3. Fallback to default path (auto-download from HuggingFace)
    
    Returns:
        BiRefNet model instance
    
    Raises:
        RuntimeError: If model loading fails
    """
    global _birefnet_model
    
    if _birefnet_model is None:
        try:
            from transformers import AutoModelForImageSegmentation
        except ImportError as e:
            raise RuntimeError(
                "transformers package is required for BiRefNet. "
                "Install it with: pip install transformers"
            ) from e
        
        model_path = _get_birefnet_model_path()
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Suppress FutureWarning from timm (used by BiRefNet)
            # These warnings are from dependency, not our code
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=".*timm.*",
                )
                _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                )
            
            if device == "cuda" and _birefnet_model is not None:
                _birefnet_model = _birefnet_model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load BiRefNet model: {e}")
    
    return _birefnet_model


# ============================================================================
# Helper Functions for BiRefNet Processing
# ============================================================================

def _calculate_birefnet_bbox_from_points(
    points: List[List[float]],
    image_rgb: np.ndarray,
    height: int,
    width: int
) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box for BiRefNet from brush stroke points using FastSAM.
    
    Args:
        points: List of points [[x, y], ...]
        image_rgb: Image as numpy array
        height: Image height
        width: Image width
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) with 10% padding
    
    Raises:
        ValueError: If FastSAM fails to generate mask or mask is empty
    """
    fastsam_model = get_segmentation_model()
    point_list = [[float(p[0]), float(p[1])] for p in points]
    fastsam_results = fastsam_model(image_rgb, points=point_list, labels=[1] * len(points))
    
    if not fastsam_results or not fastsam_results[0].masks or len(fastsam_results[0].masks.data) == 0:
        raise ValueError("FastSAM did not generate any masks from brush strokes")
    
    fastsam_mask_data = fastsam_results[0].masks.data[0].cpu().numpy()
    if fastsam_mask_data.shape != (height, width):
        mask_pil = Image.fromarray((fastsam_mask_data * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        fastsam_mask_data = np.array(mask_pil, dtype=np.float32) / 255.0
    
    fastsam_mask_binary = (fastsam_mask_data > 0.5).astype(np.uint8) * 255
    white_pixels = np.where(fastsam_mask_binary > 128)
    
    if len(white_pixels[0]) == 0:
        raise ValueError("FastSAM mask is empty, cannot calculate bbox for BiRefNet")
    
    y_min_mask, y_max_mask = int(np.min(white_pixels[0])), int(np.max(white_pixels[0]))
    x_min_mask, x_max_mask = int(np.min(white_pixels[1])), int(np.max(white_pixels[1]))
    
    # Add 10% padding
    padding_x = int((x_max_mask - x_min_mask) * 0.1)
    padding_y = int((y_max_mask - y_min_mask) * 0.1)
    
    return (
        max(0, x_min_mask - padding_x),
        max(0, y_min_mask - padding_y),
        min(width, x_max_mask + padding_x),
        min(height, y_max_mask + padding_y)
    )


def _normalize_bbox(
    bbox_coords: List[float],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    """
    Normalize bounding box coordinates to valid image bounds.
    
    Args:
        bbox_coords: Bounding box [x_min, y_min, x_max, y_max]
        width: Image width
        height: Image height
    
    Returns:
        Tuple of normalized (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = bbox_coords
    return (
        max(0, min(int(x_min), width - 1)),
        max(0, min(int(y_min), height - 1)),
        max(int(x_min) + 1, min(int(x_max), width)),
        max(int(y_min) + 1, min(int(y_max), height))
    )


def _extract_birefnet_mask_tensor(outputs: Any) -> Any:
    """
    Extract mask tensor from BiRefNet model output.
    
    Args:
        outputs: BiRefNet model output (can be various formats)
    
    Returns:
        Mask tensor (2D)
    
    Raises:
        RuntimeError: If mask cannot be extracted from output
    
    Note:
        BiRefNet outputs can have different structures depending on model version.
        This function handles all known formats.
    """
    import torch  # Lazy import
    
    if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
        return outputs.pred_masks[0, 0]
    elif hasattr(outputs, 'masks') and outputs.masks is not None:
        return outputs.masks[0, 0]
    elif hasattr(outputs, 'logits') and outputs.logits is not None:
        return torch.sigmoid(outputs.logits[0, 0])
    elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        first_output = outputs[0]
        if isinstance(first_output, torch.Tensor) and first_output.dim() >= 2:
            return first_output[0, 0]
        return torch.tensor(first_output)
    elif isinstance(outputs, dict) and len(outputs) > 0:
        first_value = list(outputs.values())[0]
        if isinstance(first_value, torch.Tensor) and first_value.dim() >= 2:
            return first_value[0, 0]
        return torch.tensor(first_value)
    else:
        raise RuntimeError(
            f"Could not extract mask from BiRefNet output. Output type: {type(outputs)}"
        )


def _process_birefnet_mask(
    mask_tensor: Any,
    cropped_height: int,
    cropped_width: int,
    height: int,
    width: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int
) -> np.ndarray:
    """
    Process BiRefNet mask: convert to numpy, invert, resize, and paste into full image.
    
    Args:
        mask_tensor: Mask tensor from BiRefNet
        cropped_height: Height of cropped region
        cropped_width: Width of cropped region
        height: Full image height
        width: Full image width
        x_min, y_min, x_max, y_max: Bounding box coordinates
    
    Returns:
        Full-size mask as numpy array (uint8)
    
    Note:
        BiRefNet returns background mask, so we invert it to get object mask.
    """
    import torch  # Lazy import
    
    # Convert tensor to numpy
    mask_data = (
        mask_tensor.cpu().numpy()
        if isinstance(mask_tensor, torch.Tensor)
        else np.array(mask_tensor)
    )
    
    # Normalize to 0-255 range
    mask_data = (
        (mask_data * 255).astype(np.uint8)
        if mask_data.max() <= 1.0
        else mask_data.astype(np.uint8)
    )
    
    # Invert: BiRefNet returns background mask, we need object mask
    mask_data = 255 - mask_data
    
    # Ensure 2D shape
    if len(mask_data.shape) != 2:
        if len(mask_data.shape) == 3:
            mask_data = mask_data[0]
        else:
            raise ValueError(
                f"Unexpected mask_data shape: {mask_data.shape}, expected 2D array"
            )
    
    # Resize to cropped region size if needed
    if mask_data.shape != (cropped_height, cropped_width):
        mask_pil = Image.fromarray(mask_data, mode='L')
        mask_pil = mask_pil.resize(
            (cropped_width, cropped_height),
            Image.Resampling.LANCZOS
        )
        mask_data = np.array(mask_pil, dtype=np.uint8)
    
    # Paste into full-size mask
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = mask_data
    
    return full_mask


def _process_fastsam_mask(
    results: Any,
    height: int,
    width: int
) -> np.ndarray:
    """
    Process FastSAM mask: extract and normalize to image size.
    
    Args:
        results: FastSAM results object
        height: Image height
        width: Image width
    
    Returns:
        Binary mask as numpy array (uint8, 0 or 255)
    
    Raises:
        ValueError: If no masks generated
    """
    if not results or not results[0].masks or len(results[0].masks.data) == 0:
        raise ValueError("FastSAM did not generate any masks")
    
    mask_data = results[0].masks.data[0].cpu().numpy()
    
    # Resize if needed
    if mask_data.shape != (height, width):
        mask_pil = Image.fromarray((mask_data * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
        mask_data = np.array(mask_pil, dtype=np.float32) / 255.0
    
    return (mask_data > 0.5).astype(np.uint8) * 255


def _apply_mask_post_processing(
    mask_binary: np.ndarray,
    border_adjustment: int,
    use_blur: bool,
    width: int,
    height: int,
    request_id: Optional[str],
    skip_fastsam_processing: bool = False
) -> np.ndarray:
    """
    Apply post-processing to mask: morphology operations, border adjustment, and blur.
    
    Args:
        mask_binary: Binary mask (uint8, 0 or 255)
        border_adjustment: Border adjustment in pixels (negative = shrink, positive = grow)
        use_blur: Whether to apply Gaussian blur
        width: Image width
        height: Image height
        request_id: Optional request ID for cancellation checking
        skip_fastsam_processing: If True, skip FastSAM-specific processing
    
    Returns:
        Processed mask as numpy array (float32, 0.0-1.0 range)
    
    Note:
        - FastSAM processing: opening, border adjustment, median filter, closing, erosion
        - BiRefNet processing: only blur if requested (simpler, already high quality)
    """
    from skimage.filters import gaussian
    
    # FastSAM-specific processing (morphology operations)
    if not skip_fastsam_processing:
        from skimage.morphology import (
            binary_dilation, binary_erosion, binary_closing, binary_opening
        )
        try:
            from skimage.morphology import disk
        except ImportError:
            from skimage.morphology.footprints import disk
        
        if request_id and _check_cancellation(request_id):
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        # Opening: remove small noise
        opening_footprint = disk(radius=3)
        mask_bool = (mask_binary > 128).astype(np.bool_)  # type: ignore
        mask_opened = binary_opening(mask_bool, footprint=opening_footprint)
        mask_binary = mask_opened.astype(np.uint8) * 255
        
        if request_id and _check_cancellation(request_id):
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        # Border adjustment: dilation (grow) or erosion (shrink)
        if border_adjustment != 0:
            adjustment_radius = max(1, abs(border_adjustment) // 2)
            footprint = disk(radius=adjustment_radius)
            mask_bool = (mask_binary > 128).astype(np.bool_)
            if border_adjustment > 0:
                mask_adjusted = binary_dilation(mask_bool, footprint=footprint)
            else:
                mask_adjusted = binary_erosion(mask_bool, footprint=footprint)
            mask_binary = mask_adjusted.astype(np.uint8) * 255
        
        # Median filter: smooth edges
        try:
            from skimage.filters.rank import median
            from skimage.morphology import disk as median_disk
            mask_binary = median(mask_binary.astype(np.uint8), median_disk(radius=1))
        except (ImportError, AttributeError):
            pass
    
    # Blur processing
    base_sigma = max(3.0, min(width, height) / 200.0)
    
    if skip_fastsam_processing:
        # BiRefNet: simpler processing (only blur if requested)
        if use_blur:
            blur_sigma = max(4.0, base_sigma * 1.5)
            mask_float = mask_binary.astype(np.float32) / 255.0
            return gaussian(mask_float, sigma=blur_sigma, preserve_range=True)
        else:
            return mask_binary.astype(np.float32) / 255.0
    else:
        # FastSAM: more complex processing (blur, closing, erosion)
        blur_sigma = max(4.0, base_sigma * 1.5) if use_blur else base_sigma
        mask_float = mask_binary.astype(np.float32) / 255.0
        mask_blurred = gaussian(mask_float, sigma=blur_sigma, preserve_range=True)
        
        # Closing: fill small holes
        closing_footprint = disk(radius=2)
        mask_bool = (mask_blurred > 0.25).astype(np.bool_)  # type: ignore
        mask_closed = binary_closing(mask_bool, footprint=closing_footprint)
        
        mask_closed_float = mask_closed.astype(np.float32)
        final_blur_sigma = blur_sigma * 0.7
        mask_final = gaussian(mask_closed_float, sigma=final_blur_sigma, preserve_range=True)
        
        # Erosion: refine edges
        try:
            from skimage.morphology import binary_erosion
            erosion_footprint = disk(radius=1)
            mask_bool_final = (mask_final > 0.4).astype(np.bool_)  # type: ignore
            mask_eroded = binary_erosion(mask_bool_final, footprint=erosion_footprint)
            mask_eroded_float = mask_eroded.astype(np.float32)
            return gaussian(mask_eroded_float, sigma=blur_sigma * 0.3, preserve_range=True)
        except Exception:
            return mask_final


# ============================================================================
# Helper: BBox từ mask
# ============================================================================
def _calculate_bbox_from_mask(
    mask: Image.Image,
    padding_percent: float = 0.1,
) -> Tuple[int, int, int, int]:
    """
    Tính bounding box của vùng trắng trong mask.

    Args:
        mask: Mask (trắng = đối tượng, đen = nền)
        padding_percent: Padding thêm theo tỉ lệ bbox (mặc định 10%)

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.uint8)
    white_pixels = np.argwhere(mask_array > 0)
    if white_pixels.size == 0:
        raise ValueError("Mask rỗng, không tính được bbox")

    y_min, x_min = np.min(white_pixels, axis=0)
    y_max, x_max = np.max(white_pixels, axis=0)

    padding_x = int((x_max - x_min) * padding_percent)
    padding_y = int((y_max - y_min) * padding_percent)

    width, height = mask.size
    x_min_p = max(0, x_min - padding_x)
    y_min_p = max(0, y_min - padding_y)
    x_max_p = min(width, x_max + padding_x)
    y_max_p = min(height, y_max + padding_y)

    return x_min_p, y_min_p, x_max_p, y_max_p


# ============================================================================
# Helper: Refine mask với BiRefNet
# ============================================================================
def refine_mask_with_birefnet(
    original_image: Image.Image,
    existing_mask: Image.Image,
    border_adjustment: int = 0,
    use_blur: bool = False,
    request_id: Optional[str] = None,
) -> Image.Image:
    """
    Refine mask hiện có bằng BiRefNet.

    Quy trình:
      1. Tính bbox từ mask hiện tại
      2. Chạy BiRefNet với bbox
      3. Trả về mask refine (PIL)
    """
    # Tính bbox; nếu mask rỗng sẽ raise, caller sẽ fallback
    x_min, y_min, x_max, y_max = _calculate_bbox_from_mask(existing_mask)
    bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]

    # Lưu original tạm (generate_smart_mask nhận path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        original_image.save(tmp.name, format="PNG")
        image_path = tmp.name

    try:
        mask_b64 = generate_smart_mask(
            image_path=image_path,
            bbox_coords=bbox,
            points=None,
            border_adjustment=border_adjustment,
            use_blur=use_blur,
            request_id=request_id,
            model_type="birefnet",
        )
        mask_bytes = base64.b64decode(mask_b64)
        refined_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        return refined_mask
    finally:
        try:
            os.remove(image_path)
        except Exception:
            logger.warning("Không thể xoá file tạm của BiRefNet", exc_info=True)


# ============================================================================
# Main Mask Generation Function
# ============================================================================

def generate_smart_mask(
    image_path: str,
    bbox_coords: Optional[List[float]] = None,
    points: Optional[List[List[float]]] = None,
    border_adjustment: int = 0,
    use_blur: bool = False,
    request_id: Optional[str] = None,
    model_type: str = "segmentation",
) -> str:
    """
    Generate a smart mask using FastSAM or BiRefNet (segmentation).
    
    This function can be used for:
    - Generating Mask A (placement mask) on main input image
    - Generating Mask R (object isolation mask) on reference image
    
    Args:
        image_path: Path to the input image
        bbox_coords: Bounding box [x_min, y_min, x_max, y_max]
            (required for BiRefNet when no points, optional for segmentation)
        points: List of points [[x, y], ...]
            (optional for segmentation, not supported directly for BiRefNet)
        border_adjustment: Border adjustment in pixels
            (negative = shrink, positive = grow, 0 = no adjustment)
        use_blur: Whether to apply Gaussian blur for soft edges
        request_id: Request ID for cancellation tracking
        model_type: Model to use - "segmentation" (FastSAM) or "birefnet"
            (default: "segmentation")
    
    Returns:
        Base64 encoded mask image (PNG format)
    
    Raises:
        FileNotFoundError: If image not found
        ValueError: If inputs are invalid or model fails
        RuntimeError: If model loading fails
    
    Note:
        - BiRefNet only supports bbox (not points) and requires cropping the region first
        - BiRefNet detects the entire region, so it's better for single objects in cropped areas
        - If points provided for BiRefNet, FastSAM is used first to calculate bbox
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate and normalize model_type
    model_type = model_type.lower()
    if model_type == "fastsam":
        model_type = "segmentation"  # Map old name to new name
    if model_type not in ["segmentation", "birefnet"]:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Must be 'segmentation' (FastSAM) or 'birefnet'"
        )
    
    # BiRefNet validation: if points provided, we'll use FastSAM first, then BiRefNet
    # If no points, bbox is required
    if model_type == "birefnet":
        if not points or len(points) == 0:
            if not bbox_coords or len(bbox_coords) != 4:
                raise ValueError(
                    "BiRefNet requires bbox_coords [x_min, y_min, x_max, y_max] "
                    "when no brush strokes are provided"
                )
    
    # Check cancellation before starting
    if request_id and _check_cancellation(request_id):
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Load model
    model = get_birefnet_model() if model_type == "birefnet" else get_segmentation_model()
    
    if request_id and _check_cancellation(request_id):
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Load and prepare image
    pil_image = Image.open(image_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    image_rgb = np.array(pil_image, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    
    # Process based on model type
    if model_type == "birefnet":
        # BiRefNet processing: requires cropping region first
        if points and len(points) > 0:
            # Use FastSAM to calculate bbox from points
            x_min, y_min, x_max, y_max = _calculate_birefnet_bbox_from_points(
                points, image_rgb, height, width
            )
        else:
            # Use provided bbox (already validated above)
            if bbox_coords is None or len(bbox_coords) != 4:
                raise ValueError(
                    "BiRefNet requires bbox_coords [x_min, y_min, x_max, y_max] "
                    "when no brush strokes are provided"
                )
            # Type assertion: bbox_coords is guaranteed to be List[float] here
            assert bbox_coords is not None  # For type checker
            x_min, y_min, x_max, y_max = _normalize_bbox(bbox_coords, width, height)
        
        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        cropped_height, cropped_width = cropped_image.size[1], cropped_image.size[0]
        
        if request_id and _check_cancellation(request_id):
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        # Prepare image tensor for BiRefNet
        import torch  # Lazy import
        import torchvision.transforms as transforms
        
        processed_image = cropped_image.resize((1024, 1024), Image.Resampling.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(processed_image).unsqueeze(0)  # type: ignore
        
        if model is None:
            raise RuntimeError("BiRefNet model is None")
        
        device = next(model.parameters()).device  # type: ignore
        model_dtype = next(model.parameters()).dtype  # type: ignore
        image_tensor = image_tensor.to(device=device, dtype=model_dtype)
        
        # Run inference
        model.eval()  # type: ignore
        with torch.no_grad():
            try:
                outputs = model(image_tensor)  # type: ignore
            except TypeError:
                outputs = model(pixel_values=image_tensor)  # type: ignore
        
        if outputs is None:
            raise RuntimeError("BiRefNet model returned None output")
        
        # Extract and process mask
        mask_tensor = _extract_birefnet_mask_tensor(outputs)
        mask_binary = _process_birefnet_mask(
            mask_tensor, cropped_height, cropped_width,
            height, width, x_min, y_min, x_max, y_max
        )
        skip_fastsam_processing = True
    
    else:
        # FastSAM processing
        skip_fastsam_processing = False
        if model is None:
            raise RuntimeError("FastSAM model is None")
        
        if points and len(points) > 0:
            point_list = [[float(p[0]), float(p[1])] for p in points]
            results = model(image_rgb, points=point_list, labels=[1] * len(points))  # type: ignore
        elif bbox_coords and len(bbox_coords) == 4:
            # Type assertion: bbox_coords is guaranteed to be List[float] here
            assert bbox_coords is not None  # For type checker
            x_min, y_min, x_max, y_max = _normalize_bbox(bbox_coords, width, height)
            results = model(image_rgb, bboxes=[x_min, y_min, x_max, y_max])  # type: ignore
        else:
            raise ValueError("Either bbox_coords or points must be provided")
        
        if request_id and _check_cancellation(request_id):
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        mask_binary = _process_fastsam_mask(results, height, width)
    
    # Apply post-processing (morphology, border adjustment, blur)
    if request_id and _check_cancellation(request_id):
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    mask_final = _apply_mask_post_processing(
        mask_binary, border_adjustment, use_blur,
        width, height, request_id, skip_fastsam_processing
    )
    
    if request_id and _check_cancellation(request_id):
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Convert to uint8 and create PIL Image
    mask_binary = (mask_final * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_binary, mode='L')
    
    # Encode as base64
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
    
    Extracts object, resizes to 1:1 ratio (512x512 or 1024x1024),
    and places on white background (not transparent, not black).
    The mask is kept at original size (not resized).
    
    This is used in the two-source mask workflow:
    - Mask R: Object isolation mask on reference image (defines WHAT to insert)
    - This function extracts the object from reference image using Mask R
    
    Args:
        reference_image: Full reference image containing the object
        mask_R: Mask image (white where object is, black elsewhere)
            NOTE: mask_R is NOT resized - it stays at original size
    
    Returns:
        Isolated object image resized to 1:1 ratio (512x512 or 1024x1024)
        on white background (RGB, not transparent, not black)
    
    Raises:
        ValueError: If no object found in mask
    """
    if mask_R.size != reference_image.size:
        mask_R = mask_R.resize(reference_image.size, Image.Resampling.LANCZOS)
    
    # Convert mask to numpy array to find bounding box
    mask_array = np.array(mask_R.convert("L"), dtype=np.uint8)
    
    # Find bounding box of white pixels (object region)
    white_pixels = np.where(mask_array > 128)
    if len(white_pixels[0]) == 0:
        raise ValueError("No object found in mask (no white pixels)")
    
    y_min, y_max = int(np.min(white_pixels[0])), int(np.max(white_pixels[0])) + 1
    x_min, x_max = int(np.min(white_pixels[1])), int(np.max(white_pixels[1])) + 1
    
    # Crop object from image and mask
    cropped_image = reference_image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask_R.crop((x_min, y_min, x_max, y_max))
    
    # Determine target size (1:1 ratio, 512x512 or 1024x1024)
    cropped_width, cropped_height = cropped_image.size
    max_dimension = max(cropped_width, cropped_height)
    target_size = 512 if max_dimension <= 512 else 1024
    
    # Resize cropped object to target size (1:1 ratio) while maintaining aspect ratio
    scale = min(target_size / cropped_width, target_size / cropped_height)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)
    
    resized_object = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create white background (RGB, not RGBA, not transparent, not black)
    result = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
    
    # Calculate position to center the object on white background
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste object onto white background using mask for transparency
    if resized_object.mode != "RGBA":
        resized_object = resized_object.convert("RGBA")
    
    # Resize mask to match resized object size (for alpha compositing only)
    resized_mask = cropped_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create alpha channel from mask
    alpha = resized_mask.convert("L")
    resized_object.putalpha(alpha)
    
    result.paste(resized_object, (paste_x, paste_y), resized_object)
    return result


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
    masked_reference_object = extracted_object
    return extracted_object, masked_reference_object
