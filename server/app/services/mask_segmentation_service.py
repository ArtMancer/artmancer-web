"""
Mask Segmentation and Reference Object Extraction Service
Combines mask generation (FastSAM/BiRefNet) and reference object extraction functionality.

This service handles:
1. Mask generation/segmentation using FastSAM or BiRefNet (for both main image and reference image)
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
    pass

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
    # Check if running in Modal environment (Volume exists)
    vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
    if os.path.exists(vol_segmentation_file):
        # Load trực tiếp từ Volume (không copy để giảm cold-start)
        logger.info(f"🚀 FastSAM-s.pt loaded directly from Volume: {vol_segmentation_file}")
        return vol_segmentation_file
    else:
        # Not in Modal or Volume not set up - use default (will auto-download)
        logger.info("⚠️ FastSAM-s.pt not in Volume, will use default path (auto-download if needed)")
        return _model_path


def get_segmentation_model():
    """
    Get or load the segmentation model (FastSAM) (lazy loading).
    Model is only loaded on first request.
    
    Strategy:
    1. Check Modal Volume for cached model
    2. Load trực tiếp từ Volume (không copy) để giảm cold-start
    3. Fallback to default path (auto-download from GitHub)
    
    Returns:
        Segmentation model instance (FastSAM)
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
                "Segmentation model may run slowly or fail."
            )
        
        # Get model path (check Volume first)
        model_path = _get_segmentation_model_path()
        
        logger.info(f"Loading segmentation model (FastSAM-s) from: {model_path} (this may take a moment on first use)...")
        try:
            # FastSAM-s will auto-download if not present (~40MB)
            _fastsam_model = FastSAM(model_path)
            logger.info("Segmentation model (FastSAM-s) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
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
    # Check if running in Modal environment (Volume exists)
    if os.path.exists(VOL_BIREFNET_PATH) and os.listdir(VOL_BIREFNET_PATH):
        # Load trực tiếp từ Volume (không copy để giảm cold-start)
        logger.info(f"🚀 BiRefNet model loaded directly from Volume: {VOL_BIREFNET_PATH}")
        return VOL_BIREFNET_PATH
    else:
        # Not in Modal or Volume not set up - use default (will auto-download from HuggingFace)
        logger.info("⚠️ BiRefNet model not in Volume, will use default path (auto-download from HuggingFace if needed)")
        return 'zhengpeng7/BiRefNet'


def get_birefnet_model():
    """
    Get or load the BiRefNet model (lazy loading).
    Model is only loaded on first request.
    
    Strategy:
    1. Check Modal Volume for cached model
    2. Load trực tiếp từ Volume (không copy) để giảm cold-start
    3. Fallback to default path (auto-download from HuggingFace)
    
    Returns:
        BiRefNet model instance
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
        
        # Get model path (check Volume first)
        model_path = _get_birefnet_model_path()
        
        logger.info(f"Loading BiRefNet model from: {model_path} (this may take a moment on first use)...")
        try:
            import torch
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"BiRefNet will be loaded on device: {device}")
            
            _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use half precision on GPU
            )
            
            # Move model to GPU if available
            if device == "cuda" and _birefnet_model is not None:
                _birefnet_model = _birefnet_model.to(device)
                logger.info("BiRefNet model moved to GPU")
            
            if _birefnet_model is None:
                raise RuntimeError("BiRefNet model failed to load")
            
            logger.info("BiRefNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BiRefNet model: {e}")
            raise RuntimeError(f"Failed to load BiRefNet model: {e}")
    
    return _birefnet_model


def generate_smart_mask(
    image_path: str,
    bbox_coords: Optional[List[float]] = None,
    points: Optional[List[List[float]]] = None,
    border_adjustment: int = 0,
    use_blur: bool = False,
    request_id: Optional[str] = None,  # Request ID for cancellation tracking
    model_type: str = "segmentation",  # "segmentation" (FastSAM) or "birefnet"
) -> str:
    """
    Generate a smart mask using FastSAM or BiRefNet (segmentation).
    
    This function can be used for:
    - Generating Mask A (placement mask) on main input image
    - Generating Mask R (object isolation mask) on reference image
    
    Args:
        image_path: Path to the input image
        bbox_coords: Bounding box [x_min, y_min, x_max, y_max] (required for BiRefNet, optional for segmentation)
        points: List of points [[x, y], ...] (optional for segmentation, not supported for BiRefNet)
        border_adjustment: Border adjustment in pixels (negative = shrink, positive = grow, 0 = no adjustment)
        use_blur: Whether to apply Gaussian blur for soft edges
        request_id: Request ID for cancellation tracking
        model_type: Model to use - "segmentation" (FastSAM) or "birefnet" (default: "segmentation")
        
    Returns:
        Base64 encoded mask image (PNG format)
    
    Note:
        - BiRefNet only supports bbox (not points) and requires cropping the region first
        - BiRefNet detects the entire region, so it's better for single objects in cropped areas
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate model_type (support both old "fastsam" and new "segmentation" for backward compatibility)
    model_type = model_type.lower()
    if model_type == "fastsam":
        model_type = "segmentation"  # Map old name to new name
    if model_type not in ["segmentation", "birefnet"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'segmentation' (FastSAM) or 'birefnet'")
    
    # BiRefNet validation: if points provided, we'll use FastSAM first, then BiRefNet
    # If no points, bbox is required
    if model_type == "birefnet":
        if not points or len(points) == 0:
            # No brush strokes - bbox is required
            if not bbox_coords or len(bbox_coords) != 4:
                raise ValueError("BiRefNet requires bbox_coords [x_min, y_min, x_max, y_max] when no brush strokes are provided")
    
    # Check cancellation before starting
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ Segmentation cancelled before model loading: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Load model based on model_type
    if model_type == "birefnet":
        model = get_birefnet_model()
    else:
        model = get_segmentation_model()
    
    # Type narrowing: model should not be None after loading
    if model is None:
        raise RuntimeError(f"Failed to load {model_type} model")
    
    # Check cancellation after model loading
    if request_id and _check_cancellation(request_id):
        logger.warning(f"⚠️ Segmentation cancelled after model loading: {request_id}")
        raise ValueError(f"Segmentation request {request_id} was cancelled")
    
    # Read image using PIL
    try:
        pil_image = Image.open(image_path)
        # Convert to RGB if needed (handles RGBA, P, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        original_size = pil_image.size  # Store original size
        image_rgb = np.array(pil_image, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to read image: {image_path} - {e}")
    
    height, width = image_rgb.shape[:2]
    
    # Handle BiRefNet: if brush (points) provided, run FastSAM first, then BiRefNet
    if model_type == "birefnet":
        # If points (brush) provided, run FastSAM first to get initial mask
        if points and len(points) > 0:
            logger.info("BiRefNet with brush: Running FastSAM first to get initial mask from brush strokes")
            
            # Load FastSAM model
            fastsam_model = get_segmentation_model()
            
            # Run FastSAM with points
            point_list = [[float(p[0]), float(p[1])] for p in points]
            label_list = [1] * len(points)  # All positive points
            
            logger.info(f"FastSAM: Using {len(points)} points for initial mask generation")
            fastsam_results = fastsam_model(image_rgb, points=point_list, labels=label_list)
            
            if not fastsam_results or len(fastsam_results) == 0:
                raise ValueError("FastSAM did not generate any masks from brush strokes")
            
            # Get the first (most confident) mask from FastSAM
            fastsam_masks = fastsam_results[0].masks
            if fastsam_masks is None or len(fastsam_masks.data) == 0:
                raise ValueError("No masks found in FastSAM results")
            
            # Get FastSAM mask
            fastsam_mask_data = fastsam_masks.data[0].cpu().numpy()
            
            # Resize if needed
            if fastsam_mask_data.shape != (height, width):
                mask_pil = Image.fromarray((fastsam_mask_data * 255).astype(np.uint8), mode='L')
                mask_pil = mask_pil.resize((width, height), Image.Resampling.NEAREST)
                fastsam_mask_data = np.array(mask_pil, dtype=np.float32) / 255.0
            
            # Convert to binary mask
            fastsam_mask_binary = (fastsam_mask_data > 0.5).astype(np.uint8) * 255
            
            # Calculate bbox from FastSAM mask (find bounding box of white pixels)
            white_pixels = np.where(fastsam_mask_binary > 128)
            if len(white_pixels[0]) == 0:
                raise ValueError("FastSAM mask is empty, cannot calculate bbox for BiRefNet")
            
            y_min_mask = int(np.min(white_pixels[0]))
            y_max_mask = int(np.max(white_pixels[0]))
            x_min_mask = int(np.min(white_pixels[1]))
            x_max_mask = int(np.max(white_pixels[1]))
            
            # Add padding around mask bbox (10% on each side)
            padding_x = int((x_max_mask - x_min_mask) * 0.1)
            padding_y = int((y_max_mask - y_min_mask) * 0.1)
            x_min = max(0, x_min_mask - padding_x)
            y_min = max(0, y_min_mask - padding_y)
            x_max = min(width, x_max_mask + padding_x)
            y_max = min(height, y_max_mask + padding_y)
            
            logger.info(f"BiRefNet: Calculated bbox from FastSAM mask: [{x_min}, {y_min}, {x_max}, {y_max}]")
        else:
            # Use provided bbox
            # bbox_coords is guaranteed to be non-None and valid at this point (validated above)
            assert bbox_coords is not None, "bbox_coords must be provided for BiRefNet"
            x_min, y_min, x_max, y_max = bbox_coords
            # Ensure coordinates are within image bounds
            x_min = max(0, min(int(x_min), width - 1))
            y_min = max(0, min(int(y_min), height - 1))
            x_max = max(x_min + 1, min(int(x_max), width))
            y_max = max(y_min + 1, min(int(y_max), height))
            
            logger.info(f"BiRefNet: Using provided bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
        
        logger.info(f"BiRefNet: Cropping region [{x_min}, {y_min}, {x_max}, {y_max}] from image {width}x{height}")
        
        # Crop the image
        cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
        cropped_array = np.array(cropped_image, dtype=np.uint8)
        cropped_height, cropped_width = cropped_array.shape[:2]
        
        # Check cancellation before BiRefNet inference
        if request_id and _check_cancellation(request_id):
            logger.warning(f"⚠️ BiRefNet segmentation cancelled before inference: {request_id}")
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        # Run BiRefNet on cropped image
        try:
            import torch
            import torchvision.transforms as transforms
            
            # BiRefNet typically expects images of size 1024x1024 or similar
            # Process image manually (BiRefNet doesn't have standard image processor)
            # Resize to model's expected input size (usually 1024x1024)
            target_size = 1024
            processed_image = cropped_image.resize((target_size, target_size), Image.Resampling.BILINEAR)
            
            # Convert PIL to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Convert PIL Image to tensor, then add batch dimension
            # transform() returns torch.Tensor, not PIL Image
            tensor_result = transform(processed_image)
            if not isinstance(tensor_result, torch.Tensor):
                raise RuntimeError(f"Expected torch.Tensor from transform, got {type(tensor_result)}")
            image_tensor = tensor_result.unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
            
            # Move to same device as model and match dtype
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            image_tensor = image_tensor.to(device=device, dtype=model_dtype)
            logger.info(f"BiRefNet: Input tensor shape={image_tensor.shape}, dtype={image_tensor.dtype}, device={device}")
            
            # Run inference
            model.eval()
            outputs = None
            with torch.no_grad():
                # BiRefNet may accept image directly or as a dict
                try:
                    outputs = model(image_tensor)
                except TypeError:
                    # Try with pixel_values key (some models expect this)
                    try:
                        outputs = model(pixel_values=image_tensor)
                    except Exception as e:
                        logger.error(f"BiRefNet model call failed: {e}")
                        raise
            
            if outputs is None:
                raise RuntimeError("BiRefNet model returned None output")
            
            # Get mask from BiRefNet output
            # BiRefNet typically returns mask in pred_masks or similar attribute
            mask_tensor: torch.Tensor
            if hasattr(outputs, 'pred_masks') and outputs.pred_masks is not None:
                mask_tensor = outputs.pred_masks[0, 0]  # Get first mask: [batch, channel, H, W] -> [H, W]
            elif hasattr(outputs, 'masks') and outputs.masks is not None:
                mask_tensor = outputs.masks[0, 0]
            elif hasattr(outputs, 'logits') and outputs.logits is not None:
                # Apply sigmoid if logits
                mask_tensor = torch.sigmoid(outputs.logits[0, 0])
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Try to get first output
                first_output = outputs[0]
                if isinstance(first_output, torch.Tensor):
                    if first_output.dim() >= 2:
                        mask_tensor = first_output[0, 0] if first_output.dim() >= 2 else first_output[0]
                    else:
                        mask_tensor = first_output
                else:
                    mask_tensor = torch.tensor(first_output)
            elif isinstance(outputs, dict) and len(outputs) > 0:
                # Get first value from dict
                first_value = list(outputs.values())[0]
                if isinstance(first_value, torch.Tensor):
                    if first_value.dim() >= 2:
                        mask_tensor = first_value[0, 0] if first_value.dim() >= 2 else first_value[0]
                    else:
                        mask_tensor = first_value
                else:
                    mask_tensor = torch.tensor(first_value)
            else:
                raise RuntimeError(f"Could not extract mask from BiRefNet output. Output type: {type(outputs)}, attributes: {dir(outputs) if hasattr(outputs, '__dict__') else 'N/A'}")
            
            # Convert to numpy
            if isinstance(mask_tensor, torch.Tensor):
                mask_data = mask_tensor.cpu().numpy()
            else:
                mask_data = np.array(mask_tensor)
            
            # Normalize to 0-255
            if mask_data.max() <= 1.0:
                mask_data = (mask_data * 255).astype(np.uint8)
            else:
                mask_data = mask_data.astype(np.uint8)
            
            # Invert mask: BiRefNet returns background mask, we need foreground mask
            # White (255) should represent the object, Black (0) should represent background
            # BiRefNet typically returns background as white, so we invert it
            logger.info("BiRefNet: Inverting mask (BiRefNet returns background mask, we need foreground mask)")
            mask_data = 255 - mask_data
            
            # Resize mask to match cropped image size if needed
            # mask_data shape should be (H, W) after extraction
            if len(mask_data.shape) == 2:
                if mask_data.shape != (cropped_height, cropped_width):
                    mask_pil = Image.fromarray(mask_data, mode='L')
                    mask_pil = mask_pil.resize((cropped_width, cropped_height), Image.Resampling.LANCZOS)
                    mask_data = np.array(mask_pil, dtype=np.uint8)
            else:
                logger.warning(f"Unexpected mask_data shape: {mask_data.shape}, expected 2D array")
                # Try to extract 2D array
                if len(mask_data.shape) == 3:
                    mask_data = mask_data[0]  # Take first channel
                mask_pil = Image.fromarray(mask_data, mode='L')
                mask_pil = mask_pil.resize((cropped_width, cropped_height), Image.Resampling.LANCZOS)
                mask_data = np.array(mask_pil, dtype=np.uint8)
            
            # Create full-size mask and place cropped mask at bbox position
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = mask_data
            
            logger.info(f"BiRefNet: Detected mask in cropped region, placed in full image at [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # Use the full_mask for further processing
            mask_binary = full_mask
            
            # Skip FastSAM post-processing for BiRefNet (mask is already processed)
            # Apply only border adjustment and blur if requested
            skip_fastsam_processing = True
            
        except Exception as e:
            logger.error(f"BiRefNet inference failed: {e}")
            raise RuntimeError(f"BiRefNet inference failed: {e}")
    
    else:
        # FastSAM path (original logic)
        skip_fastsam_processing = False
        results = None
        
        if points and len(points) > 0:
            # Use point-based prompts (more accurate)
            point_list = [[float(p[0]), float(p[1])] for p in points]
            label_list = [1] * len(points)  # All positive points
            
            logger.info(f"FastSAM: Using {len(points)} points for mask generation")
            results = model(image_rgb, points=point_list, labels=label_list)
            
        elif bbox_coords and len(bbox_coords) == 4:
            # Use bounding box prompt
            x_min, y_min, x_max, y_max = bbox_coords
            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(x_min + 1, min(x_max, width))
            y_max = max(y_min + 1, min(y_max, height))
            
            # Use flat list format as per Ultralytics FastSAM API
            bboxes = [x_min, y_min, x_max, y_max]
            
            logger.info(f"FastSAM: Using bbox [{x_min}, {y_min}, {x_max}, {y_max}] for mask generation")
            results = model(image_rgb, bboxes=bboxes)
        
        else:
            raise ValueError("Either bbox_coords or points must be provided")
    
    # Process FastSAM results (only if not using BiRefNet)
    if model_type != "birefnet":
        # Check cancellation after mask generation
        if request_id and _check_cancellation(request_id):
            logger.warning(f"⚠️ FastSAM segmentation cancelled after mask generation: {request_id}")
            raise ValueError(f"Segmentation request {request_id} was cancelled")
        
        if not results or len(results) == 0:
            raise ValueError("FastSAM did not generate any masks")
        
        # Get the first (most confident) mask
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
    
    # Apply morphological operations for smoother, softer mask (only for FastSAM)
    # BiRefNet masks are already well-processed, so skip heavy morphological operations
    if not skip_fastsam_processing:
        # Manual stroke uses: lineCap='round', lineJoin='round', imageSmoothingQuality='high'
        try:
            from skimage.morphology import (
                binary_dilation, binary_erosion, binary_closing, binary_opening
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
        
        # Step 2: Apply border adjustment (shrink or grow) if requested
        # Negative values = shrink (erosion), positive values = grow (dilation)
        if border_adjustment != 0:
            # Calculate radius from pixel amount
            adjustment_radius = max(1, abs(border_adjustment) // 2)
            footprint = disk(radius=adjustment_radius)
            
            mask_bool = (mask_binary > 128).astype(bool)
            
            if border_adjustment > 0:
                # Grow: apply dilation (expand mask)
                mask_adjusted = binary_dilation(mask_bool, footprint=footprint)
                logger.info(f"🎨 Applied dilation (grow) with disk footprint (radius={adjustment_radius}, adjustment={border_adjustment}px) for soft edges")
            else:
                # Shrink: apply erosion (contract mask)
                mask_adjusted = binary_erosion(mask_bool, footprint=footprint)
                logger.info(f"🎨 Applied erosion (shrink) with disk footprint (radius={adjustment_radius}, adjustment={border_adjustment}px) to remove excess border")
            
            mask_binary = mask_adjusted.astype(np.uint8) * 255
    
    # Step 3: Apply median filter to remove noise and smooth edges (before blur) - only for FastSAM
    if not skip_fastsam_processing:
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
    
    # Step 4: Apply Gaussian blur for soft, feathered edges
    # For FastSAM: always apply blur to make masks softer
    # For BiRefNet: only apply blur if requested
    try:
        from skimage.filters import gaussian
    except ImportError:
        raise RuntimeError(
            "scikit-image is required for Gaussian blur. "
            "Install it with: pip install scikit-image"
        )
    
    # Apply blur based on model type
    base_sigma = max(3.0, min(width, height) / 200.0)
    if skip_fastsam_processing:
        # BiRefNet: only blur if requested
        if use_blur:
            blur_sigma = max(4.0, base_sigma * 1.5)
            mask_float = mask_binary.astype(np.float32) / 255.0
            mask_blurred = gaussian(mask_float, sigma=blur_sigma, preserve_range=True)
            mask_final = mask_blurred
        else:
            # No blur for BiRefNet if not requested
            mask_final = mask_binary.astype(np.float32) / 255.0
    else:
        # FastSAM: always apply blur
        if use_blur:
            blur_sigma = max(4.0, base_sigma * 1.5)
        else:
            blur_sigma = base_sigma
        
        # Convert to float for blur
        mask_float = mask_binary.astype(np.float32) / 255.0
        mask_blurred = gaussian(mask_float, sigma=blur_sigma, preserve_range=True)
        
        # Step 5: Apply gentle closing to smooth out any remaining jagged edges (FastSAM only)
        closing_footprint = disk(radius=2)
        threshold = 0.25
        mask_bool = (mask_blurred > threshold).astype(bool)
        mask_closed = binary_closing(mask_bool, footprint=closing_footprint)
        
        # Step 6: Final blur pass for extra smoothness (FastSAM only)
        mask_closed_float = mask_closed.astype(np.float32)
        final_blur_sigma = blur_sigma * 0.7
        mask_final = gaussian(mask_closed_float, sigma=final_blur_sigma, preserve_range=True)
        
        # Step 7: Optional gentle erosion to refine edges (FastSAM only)
        try:
            from skimage.morphology import binary_erosion
            erosion_footprint = disk(radius=1)
            mask_bool_final = (mask_final > 0.4).astype(bool)
            mask_eroded = binary_erosion(mask_bool_final, footprint=erosion_footprint)
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
    
    if skip_fastsam_processing:
        logger.info(f"🎨 BiRefNet mask processing: border_adjustment={border_adjustment}, blur={'enabled' if use_blur else 'disabled'}")
    else:
        blur_sigma_str = f"{blur_sigma:.2f}" if 'blur_sigma' in locals() else "N/A"
        final_blur_sigma_str = f"{final_blur_sigma:.2f}" if 'final_blur_sigma' in locals() else "N/A"
        logger.info(f"🎨 FastSAM mask processing: opening(r=3), dilation, median filter, Gaussian blur (σ={blur_sigma_str}), closing(r=2), final blur (σ={final_blur_sigma_str})")
    
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
    """
    # Resize mask to match reference image size if needed (for cropping only)
    if mask_R.size != reference_image.size:
        logger.info(f"🔄 Resizing mask_R from {mask_R.size} to {reference_image.size} for cropping")
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
    
    # Determine target size (1:1 ratio, 512x512 or 1024x1024, not too high)
    # Use the larger dimension of cropped object to determine scale
    cropped_width, cropped_height = cropped_image.size
    max_dimension = max(cropped_width, cropped_height)
    
    # Choose target size: 512x512 if max_dimension <= 512, else 1024x1024
    if max_dimension <= 512:
        target_size = 512
    else:
        target_size = 1024
    
    # Resize cropped object to target size (1:1 ratio) while maintaining aspect ratio
    # Calculate scale to fit within target_size while preserving aspect ratio
    scale = min(target_size / cropped_width, target_size / cropped_height)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)
    
    # Resize object image
    resized_object = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create white background (RGB, not RGBA, not transparent, not black)
    result = Image.new("RGB", (target_size, target_size), color=(255, 255, 255))
    
    # Calculate position to center the object on white background
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste object onto white background using mask for transparency
    # Convert resized object back to RGBA for alpha compositing
    if resized_object.mode != "RGBA":
        resized_object = resized_object.convert("RGBA")
    
    # Resize mask to match resized object size (for alpha compositing only)
    resized_mask = cropped_mask.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create alpha channel from mask
    alpha = resized_mask.convert("L")
    resized_object.putalpha(alpha)
    
    # Paste object onto white background
    result.paste(resized_object, (paste_x, paste_y), resized_object)
    
    logger.info(
        f"✅ Extracted object using Mask R: {cropped_image.size} -> {target_size}x{target_size} on white background "
        f"(mask coverage: {np.mean(mask_array > 128) * 100:.1f}%)"
    )
    
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
    
    # For now, masked_reference_object is the same as extracted_object
    # This can be extended later if different processing is needed
    masked_reference_object = extracted_object
    
    logger.info(
        "🧩 Prepared reference conditionals: extracted_object and masked_reference_object "
        f"(both preserve Mask R shape: {mask_R.size})"
    )
    
    return extracted_object, masked_reference_object

