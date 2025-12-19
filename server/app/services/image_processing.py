"""
Image Processing Service

This module provides utilities for image manipulation and conversion:
- Base64 encoding/decoding
- Image resizing with aspect ratio preservation
- LaMa (Large Mask Inpainting) for structural guidance generation
- Canny edge detection
- Mask conditional preparation

All functions work with PIL Image objects and support RGB mode conversion.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Global cache for LaMa model (lazy loading)
_inpainting_model: Optional["LaMa"] = None  # type: ignore[name-defined]

# Global cache for Stable Diffusion Inpainting pipeline (for LaMa refinement)
_refinement_pipeline: Optional[Any] = None

# Import standalone LaMa implementation
try:
    from app.core.lama_model import LaMa

    LAMA_AVAILABLE = True
except ImportError as e:
    LAMA_AVAILABLE = False
    logger.warning(
        f"⚠️ LaMa model is not available: {e}. "
        "Please ensure LaMa module is available."
    )

# Optional import for cv2 (OpenCV) – required for Canny edge detection
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning(
        "⚠️ OpenCV library (cv2) is not installed. "
        "Canny edge detection requires OpenCV. "
        "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package."
    )


def base64_to_image(data: str) -> Image.Image:
    """
    Convert base64 encoded image data to PIL Image.
    
    Supports common image formats: PNG, JPEG, JPG, WEBP, BMP, GIF, TIFF.
    
    Args:
        data: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        PIL Image in RGB mode
    
    Raises:
        ValueError: If data is invalid or image format is not supported
    """
    if not data or not data.strip():
        raise ValueError("Base64 image data cannot be empty or None")
    
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    
    if not data or not data.strip():
        raise ValueError("Base64 image data is empty after removing data URL prefix")
    
    try:
        image_bytes = base64.b64decode(data)
        if not image_bytes:
            raise ValueError("Decoded image bytes are empty")
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Verify it's a valid image format
        if image.format not in ['PNG', 'JPEG', 'JPG', 'WEBP', 'BMP', 'GIF', 'TIFF']:
            raise ValueError(f"Unsupported image format: {image.format}. Supported formats: PNG, JPEG, WEBP, BMP, GIF, TIFF")
        
        # Convert to RGB mode (required for model input)
        # If image has transparency (RGBA, LA, P with transparency), composite with white background
        if image.mode in ("RGBA", "LA", "P"):
            # Create white background
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            # Composite image on white background
            if image.mode == "P":
                image = image.convert("RGBA")
            image = Image.alpha_composite(
                white_bg.convert("RGBA"), 
                image.convert("RGBA")
            ).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid base64 image data or unsupported format: {exc}") from exc


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image to encode
        format: Image format for encoding (default: "PNG")
            Supported formats: PNG, JPEG, WEBP, etc.
    
    Returns:
        Base64 encoded image string (without data URL prefix)
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def resize_with_aspect_ratio_pad(
    image: Image.Image, 
    target_size: Tuple[int, int], 
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize image while preserving aspect ratio, then pad to target size.
    
    This prevents distortion when reference image has different aspect ratio
    than the original image.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        background_color: RGB color for padding (default: black)
    
    Returns:
        Resized and padded image with target size
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # Calculate scale factor to fit within target size (preserve aspect ratio)
    scale = min(target_width / original_width, target_height / original_height)
    
    # Resize with aspect ratio preserved
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create canvas with target size and background color
    canvas = Image.new('RGB', target_size, background_color)
    
    # Calculate position to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Paste resized image onto canvas
    canvas.paste(resized, (x_offset, y_offset))
    
    return canvas


# ============================================================================
# LaMa Implementation
# ============================================================================

def _get_inpainting_model() -> "LaMa":  # type: ignore[name-defined]
    """
    Lazily load standalone LaMa model from Modal Volume.

    - Model file should be at /checkpoints/big-lama.pt
    - Chooses CUDA if available, otherwise CPU.
    """
    global _inpainting_model

    if not LAMA_AVAILABLE:
        raise RuntimeError(
            "LaMa model is not available. Please ensure LaMa module is available."
        )

    if _inpainting_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("🎨 Loading standalone LaMa model on %s...", device)
        try:
            _inpainting_model = LaMa(device=device)
            logger.info("✅ LaMa model loaded successfully")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("❌ Failed to load LaMa model: %s", exc)
            raise RuntimeError(f"Failed to load LaMa model: {exc}") from exc

    return _inpainting_model




def generate_mae_image(
    original: Image.Image,
    mask: Image.Image,
    enable_refinement: bool = True,
    refinement_strength: float = 0.35,
    refinement_steps: int = 30,
) -> Image.Image:
    """
    Generate structural guidance image using standalone LaMa model, optionally refined with SD Inpainting.

    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere) - region to inpaint
        enable_refinement: Whether to refine LaMa output with Stable Diffusion Inpainting
                          (default: True, fixes blurry texture artifacts)
        refinement_strength: Denoising strength for refinement (0.0-1.0, default: 0.35)
        refinement_steps: Number of inference steps for refinement (default: 30)

    Returns:
        LaMa inpainting result (optionally refined): RGB image with inpainted background structure
    """
    logger.info(
        f"🎨 [generate_mae_image] Starting MAE generation: "
        f"size={original.size}, enable_refinement={enable_refinement}, "
        f"refinement_strength={refinement_strength}, refinement_steps={refinement_steps}"
    )
    # Ensure mask matches original size
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)

    original = original.convert("RGB")
    mask = mask.convert("L")

    model = _get_inpainting_model()

    img_np = np.array(original)
    mask_np = np.array(mask)
    mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)

    try:
        # Use preprocessing to handle any image size (pads to square 1:1 ratio)
        # LaMa requires square images but generalizes well to ~2K resolution
        result_np = model.forward_with_preprocessing(img_np, mask_np)
    except Exception as exc:
        logger.error("❌ LaMa inference failed: %s", exc)
        raise RuntimeError(f"LaMa inference failed: {exc}") from exc

    coarse_image = Image.fromarray(result_np.astype(np.uint8), "RGB")
    
    # Optionally refine with Stable Diffusion Inpainting to fix blurry textures
    if enable_refinement:
        logger.info(
            f"🔧 LaMa refinement enabled: strength={refinement_strength}, steps={refinement_steps}. "
            "Attempting to refine with Stable Diffusion Inpainting..."
        )
        try:
            logger.info("🎨 [generate_mae_image] Calling refine_lama_output...")
            refined_image = refine_lama_output(
                coarse_image=coarse_image,
                mask_image=mask,
                prompt="clean background, high resolution, 8k, realistic texture",
                strength=refinement_strength,
                num_inference_steps=refinement_steps,
            )
            logger.info("✅ [generate_mae_image] LaMa output refined with Stable Diffusion Inpainting")
            return refined_image
        except Exception as exc:
            logger.warning(
                f"⚠️ LaMa refinement failed: {exc}. "
                "Returning coarse LaMa output without refinement."
            )
            import traceback
            logger.error(f"❌ [generate_mae_image] Refinement error traceback: {traceback.format_exc()}")
            return coarse_image
    else:
        logger.info("⏭️  LaMa refinement disabled. Returning coarse LaMa output.")
    
    return coarse_image


def _get_refinement_pipeline() -> Optional[Any]:
    """
    Lazily load Stable Diffusion Inpainting pipeline for LaMa refinement from Volume.
    
    Priority: Volume → HuggingFace (fallback)
    - First checks /checkpoints/stable-diffusion-inpainting/ (Modal Volume)
    - Falls back to HuggingFace if not found in Volume
    
    Returns:
        Loaded StableDiffusionInpaintPipeline or None if not available
    
    Note:
        Pipeline is cached globally to avoid reloading on every call.
        Returns None if diffusers is not installed or pipeline loading fails.
    """
    global _refinement_pipeline
    
    if _refinement_pipeline is not None:
        return _refinement_pipeline
    
    # Check if already attempted and failed (avoid repeated attempts)
    if hasattr(_get_refinement_pipeline, '_load_failed'):
        return None
    
    try:
        from diffusers import StableDiffusionInpaintPipeline  # type: ignore
        from ..core.pipeline import get_device
        import os
        
        device = get_device()
        
        # Check Modal Volume first (load trực tiếp từ Volume, không copy)
        VOL_MOUNT_PATH = "/checkpoints"
        VOL_REFINEMENT_PATH = f"{VOL_MOUNT_PATH}/stable-diffusion-inpainting"
        
        # Check if model exists in Volume (diffusers format requires model_index.json)
        model_index_path = os.path.join(VOL_REFINEMENT_PATH, "model_index.json")
        if os.path.exists(VOL_REFINEMENT_PATH) and os.path.isdir(VOL_REFINEMENT_PATH) and os.path.exists(model_index_path):
            model_path = VOL_REFINEMENT_PATH
            logger.info(f"🎯 Loading refinement pipeline from Volume: {model_path}")
        else:
            # Fallback to HuggingFace
            model_path = "runwayml/stable-diffusion-inpainting"
            if os.path.exists(VOL_REFINEMENT_PATH):
                logger.warning(
                    f"⚠️ Model directory found at {VOL_REFINEMENT_PATH} but missing model_index.json. "
                    f"Falling back to HuggingFace: {model_path}"
                )
            else:
                logger.info(
                    f"🎯 Refinement model not found in Volume at {VOL_REFINEMENT_PATH}. "
                    f"Loading from HuggingFace: {model_path}"
                )
        
        logger.info(f"🎨 Loading Stable Diffusion Inpainting pipeline for LaMa refinement on {device}...")
        logger.info(f"📦 Model path: {model_path}, local_files_only: {model_path == VOL_REFINEMENT_PATH}")
        
        # Load pipeline (from Volume or HuggingFace)
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False,
            local_files_only=(model_path == VOL_REFINEMENT_PATH),  # Use local files if loading from Volume
        )
        logger.info("✅ Pipeline loaded from pretrained, moving to device...")
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)  # Disable progress bar
        _refinement_pipeline = pipeline
        
        logger.info(f"✅ Stable Diffusion Inpainting pipeline loaded for refinement on {device}")
        return _refinement_pipeline
    except ImportError as e:
        logger.warning(
            f"⚠️ Stable Diffusion Inpainting pipeline not available: {e}. "
            "LaMa refinement will be skipped. Install 'diffusers' package."
        )
        _get_refinement_pipeline._load_failed = True  # type: ignore
        return None
    except Exception as exc:
        logger.error(f"❌ Failed to load refinement pipeline: {exc}")
        import traceback
        logger.error(f"❌ Pipeline loading traceback: {traceback.format_exc()}")
        _get_refinement_pipeline._load_failed = True  # type: ignore
        return None


def refine_lama_output(
    coarse_image: Image.Image,
    mask_image: Image.Image,
    prompt: str = "clean background, high resolution, 8k, realistic texture",
    strength: float = 0.35,
    num_inference_steps: int = 30,
    pipeline: Optional[Any] = None,
) -> Image.Image:
    """
    Refine LaMa output using Stable Diffusion Inpainting to fix blurry texture artifacts.
    
    Uses Image-to-Image Inpainting technique with low denoising strength to preserve
    LaMa's structural guidance while adding realistic texture details.
    
    Args:
        coarse_image: LaMa output image (coarse inpainting result)
        mask_image: Mask image (white where object is, black elsewhere) - region to refine
        prompt: Text prompt for refinement (default: focuses on texture quality)
        strength: Denoising strength (0.0-1.0). Lower values preserve more LaMa structure.
                 Default 0.35 adds ~35% noise, allowing texture hallucination without
                 changing geometry.
        num_inference_steps: Number of diffusion steps (default: 30, or 4 if using LCM/Turbo)
        pipeline: Optional pre-loaded pipeline. If None, will load/cache pipeline automatically.
    
    Returns:
        Refined image with sharp, realistic textures in the inpainted region
    
    Note:
        - Requires diffusers library with StableDiffusionInpaintPipeline
        - Pipeline is cached globally to avoid reloading
        - If pipeline is not available, returns coarse_image unchanged
        - Low strength (0.35) preserves LaMa structure while adding texture details
    """
    # Ensure images match size
    if mask_image.size != coarse_image.size:
        mask_image = mask_image.resize(coarse_image.size, Image.Resampling.LANCZOS)
    
    coarse_image = coarse_image.convert("RGB")
    mask_image = mask_image.convert("L")
    
    # Get or use provided pipeline
    logger.info("🔍 [refine_lama_output] Getting refinement pipeline...")
    if pipeline is None:
        pipeline = _get_refinement_pipeline()
    
    if pipeline is None:
        logger.warning(
            "⚠️ [refine_lama_output] Refinement pipeline not available. Returning coarse LaMa output unchanged. "
            "Check if Stable Diffusion Inpainting model is downloaded to Volume."
        )
        return coarse_image
    
    logger.info("✅ [refine_lama_output] Refinement pipeline loaded successfully")
    
    try:
        # Prepare mask for inpainting (white = inpaint, black = keep)
        # LaMa mask: white = object (to remove), black = background
        # SD Inpainting mask: white = inpaint, black = keep
        # So we can use the mask directly (white areas will be refined)
        mask_array = np.array(mask_image, dtype=np.uint8)
        # Ensure mask is binary (0 or 255)
        mask_binary = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        mask_for_inpaint = Image.fromarray(mask_binary, mode="L")
        
        logger.info(
            f"🔧 Refining LaMa output: strength={strength}, steps={num_inference_steps}, "
            f"size={coarse_image.size}, mask_coverage={np.mean(mask_binary > 127) * 100:.1f}%"
        )
        
        # Run inpainting refinement
        logger.info("🎨 [refine_lama_output] Calling SD Inpainting pipeline...")
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image=coarse_image,
                mask_image=mask_for_inpaint,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,  # Standard CFG scale for SD
            )
        logger.info("✅ [refine_lama_output] SD Inpainting pipeline call completed")
        
        # Extract result image
        logger.info("🔍 [refine_lama_output] Extracting result image...")
        if hasattr(result, "images"):
            refined_image = result.images[0] if isinstance(result.images, list) else result.images
            logger.info(f"✅ [refine_lama_output] Extracted from result.images, type: {type(refined_image)}")
        elif isinstance(result, (list, tuple)):
            refined_image = result[0]
            logger.info(f"✅ [refine_lama_output] Extracted from result tuple/list, type: {type(refined_image)}")
        else:
            refined_image = result
            logger.info(f"✅ [refine_lama_output] Using result directly, type: {type(refined_image)}")
        
        # Ensure output matches input size
        if refined_image.size != coarse_image.size:
            logger.info(f"🔄 [refine_lama_output] Resizing refined image from {refined_image.size} to {coarse_image.size}")
            refined_image = refined_image.resize(coarse_image.size, Image.Resampling.LANCZOS)
        
        logger.info(f"✅ [refine_lama_output] LaMa output refined successfully, final size: {refined_image.size}")
        return refined_image.convert("RGB")
        
    except Exception as exc:
        logger.error(f"❌ [refine_lama_output] LaMa refinement failed: {exc}")
        import traceback
        logger.error(f"❌ [refine_lama_output] Refinement traceback: {traceback.format_exc()}")
        logger.warning("Returning coarse LaMa output unchanged due to refinement error")
        return coarse_image


def generate_canny_image(
    image: Image.Image,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    use_auto_threshold: bool = True,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    auto_threshold_sigma: float = 0.33,
) -> Image.Image:
    """
    Generate Canny edge detection image with improved preprocessing.
    
    Uses OpenCV Canny algorithm with bilateral filtering for noise reduction
    and automatic threshold calculation based on image median.
    
    Args:
        image: Input RGB image
        low_threshold: Lower threshold for edge detection (ignored if use_auto_threshold=True)
        high_threshold: Upper threshold for edge detection (ignored if use_auto_threshold=True)
        use_auto_threshold: If True, automatically calculate thresholds from image median
        bilateral_d: Diameter of pixel neighborhood for bilateral filter (default: 9)
        bilateral_sigma_color: Filter sigma in color space (default: 75.0)
        bilateral_sigma_space: Filter sigma in coordinate space (default: 75.0)
        auto_threshold_sigma: Sigma multiplier for auto threshold calculation (default: 0.33)
    
    Returns:
        Canny edge image: RGB image with white edges on black background
    """
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV library (cv2) is not installed in the runtime environment. "
            "Cannot generate Canny edge image. "
            "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package."
        )
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)  # type: ignore[attr-defined]
    
    # 1. Khử nhiễu nhưng giữ cạnh (Quan trọng)
    # d=9, sigmaColor=75, sigmaSpace=75 là thông số an toàn
    blurred = cv2.bilateralFilter(  # type: ignore[attr-defined]
        gray, 
        bilateral_d, 
        bilateral_sigma_color, 
        bilateral_sigma_space
    )
    
    # 2. Tính ngưỡng tự động (Auto Threshold)
    if use_auto_threshold:
        sigma = auto_threshold_sigma
        v = float(np.median(blurred))
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    else:
        lower = low_threshold if low_threshold is not None else 100
        upper = high_threshold if high_threshold is not None else 200
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower, upper)  # type: ignore[attr-defined]
    
    # Convert to RGB (white edges on black background)
    canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # type: ignore[attr-defined]
    canny_image = Image.fromarray(canny_rgb, mode="RGB")
    
    logger.info(
        f"✅ Generated Canny edge image "
        f"(thresholds: [{lower}, {upper}], "
        f"auto={use_auto_threshold}, "
        f"bilateral=[d={bilateral_d}, σ_color={bilateral_sigma_color}, σ_space={bilateral_sigma_space}])"
    )
    return canny_image


def dilate_mask_for_mae(
    mask: Image.Image,
    dilation_pixels: int = 20,
) -> Image.Image:
    """
    Dilate mask for MAE preprocessing (brush masks only).
    
    Expands the mask by the specified number of pixels using morphological dilation
    with a circular kernel. This gives LaMa more context around the object boundary
    for better inpainting results.
    
    Args:
        mask: Mask image (white where object is, black elsewhere) - PIL Image
        dilation_pixels: Number of pixels to dilate (default: 35, range: 15-50)
    
    Returns:
        Dilated mask as PIL Image (same mode as input)
    
    Note:
        - Requires CV2_AVAILABLE = True (OpenCV must be installed)
        - Uses circular kernel (MORPH_ELLIPSE) for natural expansion
        - Kernel size is approximately 2 * dilation_pixels + 1
        - Only applied to brush masks for removal task MAE preprocessing
    """
    if not CV2_AVAILABLE:
        logger.warning(
            "⚠️ OpenCV not available. Cannot dilate mask. "
            "Returning original mask."
        )
        return mask
    
    # Clamp dilation_pixels to valid range (15-50)
    dilation_pixels = max(15, min(50, dilation_pixels))
    
    # Ensure mask is grayscale
    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.uint8)
    
    # Binarize mask: > 127 -> 255, else 0
    mask_binary = np.where(mask_array > 127, 255, 0).astype(np.uint8)
    
    # Calculate kernel size: approximately 2 * dilation_pixels + 1
    # This ensures the dilation expands by roughly dilation_pixels in each direction
    kernel_size = 2 * dilation_pixels + 1
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create circular kernel for natural expansion
    kernel = cv2.getStructuringElement(  # type: ignore[attr-defined]
        cv2.MORPH_ELLIPSE,  # type: ignore[attr-defined]
        (kernel_size, kernel_size)
    )
    
    # Apply dilation
    dilated_mask = cv2.dilate(mask_binary, kernel, iterations=1)  # type: ignore[attr-defined]
    
    # Convert back to PIL Image (preserve original mode if not L)
    if mask.mode == "L":
        dilated_image = Image.fromarray(dilated_mask, mode="L")
    else:
        # If original was RGB, convert back to RGB
        dilated_image = Image.fromarray(dilated_mask, mode="L").convert(mask.mode)
    
    logger.info(
        f"🔍 Dilated mask for MAE: {dilation_pixels}px expansion "
        f"(kernel_size={kernel_size}x{kernel_size})"
    )
    
    return dilated_image


def prepare_mask_conditionals(
    original: Image.Image,
    mask: Image.Image,
    include_mae: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Create conditional images for the model: mask, masked_bg, masked_object, mae.
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
        include_mae: Whether to include MAE image (default: True)
    
    Returns:
        - mask_rgb: RGB mask image
        - masked_bg: Background with mask area removed (original - mask region = black)
        - masked_object: Object extracted from mask area (NOT used for insertion - use ref_img instead)
        - mae_image: MAE inpainted preview if include_mae=True, otherwise same as masked_bg
    
    Note:
        For insertion task, do NOT use masked_object. Use ref_img (reference image) from frontend instead.
        masked_object is only kept for backward compatibility with removal task debugging.
    """

    # Ensure mask matches original size
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)

    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)

    original_array = np.array(original, dtype=np.float32) / 255.0
    if mask_array.shape != original_array.shape[:2]:
        raise ValueError("Mask and original image must have the same spatial size")

    mask_stack = np.repeat(mask_array[..., None], 3, axis=2)

    # Create masked_object: keep original where mask is, black (0,0,0) elsewhere
    # NOTE: This is NOT used for insertion - use ref_img from frontend instead
    masked_object_array = original_array * mask_stack
    masked_object = Image.fromarray(np.uint8(masked_object_array * 255), mode="RGB")

    # Create masked_bg: keep original where mask is NOT, black (0,0,0) where mask is
    # This is the background with the mask area removed
    masked_bg_array = original_array * (1.0 - mask_stack)
    masked_bg = Image.fromarray(np.uint8(masked_bg_array * 255), mode="RGB")

    # Mask RGB (for model input, keep as RGB)
    mask_rgb = mask.convert("RGB")

    # Generate MAE image if requested
    if include_mae:
        mae_image = generate_mae_image(original, mask)
        logger.info("🧩 Generated conditional images (mask/masked_bg/masked_object/mae)")
    else:
        mae_image = masked_bg
        logger.info("🧩 Generated conditional images (mask/masked_bg/masked_object) - MAE disabled")

    return mask_rgb, masked_bg, masked_object, mae_image


def add_shadow_hint_for_inpainting(
    composite_image: np.ndarray,
    object_mask: np.ndarray,
    shadow_offset: tuple[int, int] = (3, 5),
    shadow_opacity: float = 0.4,
    blur_radius: int = 21,
    floating_height: int = 0,
    ground_bias: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add shadow hint to composite image for GenAI inpainting to generate realistic cast shadows.
    
    This function creates a "fake" shadow by darkening the background around the object,
    then expands the mask to include the shadow region. This gives the inpainting model
    enough context to generate realistic shadows that blend with the background.
    
    Supports complex geometric scenarios:
    - Floating objects (e.g., drones): Use floating_height > 0
    - Partial contact objects (e.g., mushrooms, people with arms out): Use ground_bias=True
    
    Args:
        composite_image: Input image as numpy array (H, W, C), dtype uint8, range [0, 255]
        object_mask: Binary mask as numpy array (H, W), dtype uint8, 255=object, 0=background
        shadow_offset: (x, y) base offset for shadow direction in pixels. 
                      Positive x = right, positive y = down.
                      Default (3, 5) creates subtle shadow hint (smaller offset = less noticeable).
        shadow_opacity: Shadow darkness factor (0.0-1.0). Higher = darker shadow.
                       Default 0.4 gives subtle hint without being too obvious.
        blur_radius: Base Gaussian blur radius for soft shadow edges (must be odd number).
                    Default 21 creates soft, natural-looking shadow falloff.
                    Automatically increased for floating objects.
        floating_height: Height of object above ground in pixels. 
                       0 = grounded (default), > 0 = floating in air.
                       When > 0, shadow is shifted down more and blur is increased.
        ground_bias: If True, applies vertical gradient to suppress shadow at top
                    while keeping strong shadow at bottom. Useful for objects with
                    wide tops that touch ground (e.g., mushrooms, people with arms out).
                    Default True.
    
    Returns:
        tuple containing:
        - modified_image: Image with shadow hint darkened (np.ndarray, uint8)
        - expanded_mask: Mask expanded to include shadow region (np.ndarray, uint8)
                       Covers both object and shadow area (even if disconnected for floating objects)
    
    Algorithm:
        1. Calculate bounding box of object for gradient mask
        2. Shift object_mask by shadow_offset + floating_height to create shadow_mask
        3. Apply ground_bias gradient if enabled (fade shadow at top, strong at bottom)
        4. Increase blur_radius proportionally for floating objects
        5. Blur shadow_mask with Gaussian filter for soft edges
        6. Darken composite_image in shadow region (excluding object area)
        7. Create final_mask = object_mask OR shadow_mask (covers both even if disconnected)
        8. Dilate final_mask slightly for seamless blending context
    
    Example:
        >>> composite = np.array(Image.open("composite.png"))
        >>> mask = np.array(Image.open("mask.png").convert("L"))
        >>> # Grounded object with wide top (mushroom)
        >>> modified, expanded = add_shadow_hint_for_inpainting(
        ...     composite, mask, 
        ...     shadow_offset=(3, 5),  # Small offset for subtle hint
        ...     floating_height=0,
        ...     ground_bias=True  # Suppress shadow at top
        ... )
        >>> # Floating object (drone)
        >>> modified, expanded = add_shadow_hint_for_inpainting(
        ...     composite, mask,
        ...     shadow_offset=(3, 5),  # Small offset for subtle hint
        ...     floating_height=50,  # 50px above ground
        ...     ground_bias=False    # No gradient needed
        ... )
    
    Note:
        - Requires CV2_AVAILABLE = True (OpenCV must be installed)
        - blur_radius must be odd; function will make it odd if even
        - Shadow only affects background pixels, never the object itself
        - Final mask is dilated by ~5-10px for better model context
        - For floating objects, shadow may be disconnected from object in final mask
    """
    if not CV2_AVAILABLE:
        logger.warning(
            "⚠️ OpenCV not available. Cannot add shadow hint. "
            "Returning original image and mask."
        )
        return composite_image.copy(), object_mask.copy()
    
    # Ensure blur_radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1
    
    # Ensure object_mask is binary (0 or 255)
    object_mask_binary = (object_mask > 127).astype(np.uint8) * 255
    
    # Calculate bounding box of object for gradient mask (if ground_bias is enabled)
    bbox = None
    if ground_bias:
        # Find bounding box of object
        coords = np.column_stack(np.where(object_mask_binary > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (x_min, y_min, x_max, y_max)
    
    # Step 1: Create shadow mask by shifting object mask
    # Add floating_height to Y offset (floating objects cast shadows further down)
    offset_x, offset_y_base = shadow_offset
    offset_y = offset_y_base + floating_height
    
    # Increase blur_radius proportionally for floating objects (shadows get softer with distance)
    if floating_height > 0:
        # Increase blur by ~30% per 50px of floating height
        blur_multiplier = 1.0 + (floating_height / 50.0) * 0.3
        adjusted_blur_radius = int(blur_radius * blur_multiplier)
        # Ensure adjusted blur is odd
        if adjusted_blur_radius % 2 == 0:
            adjusted_blur_radius += 1
    else:
        adjusted_blur_radius = blur_radius
    
    height, width = object_mask_binary.shape
    
    # Create translation matrix for affine transform
    translation_matrix = np.array([[1.0, 0.0, float(offset_x)], 
                                     [0.0, 1.0, float(offset_y)]], dtype=np.float32)
    shadow_mask = cv2.warpAffine(  # type: ignore[call-overload]
        object_mask_binary,
        translation_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0,)
    )
    
    # Step 1.5: Apply ground_bias gradient if enabled
    # This creates a vertical gradient that fades shadow at top, keeps it strong at bottom
    if ground_bias and bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        
        if bbox_height > 0 and bbox_width > 0:
            # Create vertical gradient: 0 (black) at top, 1 (white) at bottom
            # Gradient is created in bounding box coordinates
            gradient = np.linspace(0.0, 1.0, bbox_height, dtype=np.float32)
            gradient_2d = np.tile(gradient[:, np.newaxis], (1, bbox_width))
            
            # Create full-size gradient mask (same size as shadow_mask)
            gradient_mask = np.zeros((height, width), dtype=np.float32)
            
            # Calculate shifted bbox for shadow (apply same offset)
            shadow_x_min = max(0, min(width - 1, x_min + offset_x))
            shadow_y_min = max(0, min(height - 1, y_min + offset_y))
            shadow_x_max = max(shadow_x_min + 1, min(width, shadow_x_min + bbox_width))
            shadow_y_max = max(shadow_y_min + 1, min(height, shadow_y_min + bbox_height))
            
            # Crop gradient to fit within image bounds
            grad_h = shadow_y_max - shadow_y_min
            grad_w = shadow_x_max - shadow_x_min
            if grad_h > 0 and grad_w > 0:
                # Resize gradient to match shadow bbox size
                gradient_cropped = cv2.resize(  # type: ignore[attr-defined]
                    gradient_2d,
                    (grad_w, grad_h),
                    interpolation=cv2.INTER_LINEAR
                )
                gradient_mask[shadow_y_min:shadow_y_max, shadow_x_min:shadow_x_max] = gradient_cropped
            
            # Multiply shadow_mask by gradient (suppress shadow at top, keep at bottom)
            shadow_mask = (shadow_mask.astype(np.float32) * gradient_mask).astype(np.uint8)
    
    # Step 2: Blur shadow mask for soft edges
    shadow_mask_blurred = cv2.GaussianBlur(
        shadow_mask,
        (adjusted_blur_radius, adjusted_blur_radius),
        0
    )
    
    # Convert to float [0, 1] for alpha blending
    shadow_alpha = shadow_mask_blurred.astype(np.float32) / 255.0
    
    # Step 3: Darken composite image in shadow region (but NOT in object region)
    # Create object protection mask: 1 where object is, 0 elsewhere
    object_protection = (object_mask_binary > 127).astype(np.float32)
    
    # Apply shadow only to non-object pixels
    # shadow_factor = 1 - (shadow_alpha * shadow_opacity) for background
    # shadow_factor = 1 for object pixels (no darkening)
    shadow_factor = 1.0 - (shadow_alpha * shadow_opacity * (1.0 - object_protection))
    
    # Apply darkening to each channel
    modified_image = composite_image.astype(np.float32)
    for c in range(3):  # RGB channels
        modified_image[:, :, c] = modified_image[:, :, c] * shadow_factor
    
    # Clip and convert back to uint8
    modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)
    
    # Step 4: Create expanded mask = object_mask OR shadow_mask
    # This covers both object and shadow area, even if disconnected (floating objects)
    final_mask = cv2.bitwise_or(object_mask_binary, shadow_mask)
    
    # Step 5: Dilate final mask slightly for better blending context
    # Use 7x7 kernel for ~5-10px expansion
    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    logger.info(
        f"✨ Added shadow hint: offset={shadow_offset}, "
        f"opacity={shadow_opacity}, blur={adjusted_blur_radius}px "
        f"(base={blur_radius}px), floating_height={floating_height}px, "
        f"ground_bias={ground_bias}, expansion={kernel_size}x{kernel_size}"
    )
    
    return modified_image, expanded_mask

