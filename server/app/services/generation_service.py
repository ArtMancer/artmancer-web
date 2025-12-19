"""
Generation service for image generation using Qwen pipeline.

This service handles:
- Image generation for insertion, removal, and white-balance tasks
- Pipeline loading and management
- Progress tracking and cancellation
- Debug session management
"""

from __future__ import annotations

import logging
import time
import hashlib
import io
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, cast, Callable

from PIL import Image
import numpy as np

# torch is imported lazily where needed to avoid requiring it in services that don't need it

from ..core.config import settings
from ..core.pipeline import get_device, get_device_info, load_pipeline
from ..services.image_processing import (
    base64_to_image,
    image_to_base64,
    prepare_mask_conditionals,
    resize_with_aspect_ratio_pad,
)
from ..services.mask_segmentation_service import prepare_reference_conditionals
from ..services.reference_guided_insertion import execute_insertion_pipeline
from ..services.condition_builder import (
    build_insertion_conditionals,
    build_removal_conditionals,
    build_white_balance_conditionals,
)
from ..services.debug_service import debug_service, DEBUG_ENABLED, DebugSession
from ..models.schemas import GenerationRequest

logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Exception raised when a generation task is cancelled."""
    def __init__(self, task_id: str, message: str = "Task was cancelled"):
        self.task_id = task_id
        self.message = message
        super().__init__(self.message)


try:
    # Use FastAPI's HTTPException if available (when running via FastAPI)
    from fastapi import HTTPException  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without fastapi
    class HTTPException(Exception):  # type: ignore[override]
        """Minimal HTTP-like exception for use in core service when FastAPI is not available."""

        def __init__(self, status_code: int, detail: str) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)


# ============================================================================
# Helper Functions (extracted for reusability and clarity)
# ============================================================================

def _round_to_multiple_of_8(value: int) -> int:
    """
    Round up to nearest multiple of 8.
    
    Args:
        value: Integer value to round
    
    Returns:
        Rounded value (multiple of 8)
    
    Note:
        Stable Diffusion requires width and height to be divisible by 8.
    """
    return ((value + 7) // 8) * 8


def _calculate_output_dimensions(
    original: Image.Image,
    requested_width: Optional[int],
    requested_height: Optional[int]
) -> Tuple[int, int]:
    """
    Calculate output dimensions preserving aspect ratio and rounding to multiple of 8.
    
    Args:
        original: Original image to preserve aspect ratio from
        requested_width: Optional requested width
        requested_height: Optional requested height
    
    Returns:
        Tuple of (output_width, output_height) both divisible by 8
    
    Note:
        - If both dimensions provided, uses them (after rounding)
        - If only one dimension provided, calculates the other to maintain aspect ratio
        - If neither provided, uses original dimensions (after rounding)
        - Always rounds to nearest multiple of 8 while preserving aspect ratio as much as possible
    """
    # Use original dimensions as default
    output_width = requested_width if requested_width and requested_width != original.width else original.width
    output_height = requested_height if requested_height and requested_height != original.height else original.height
    
    # If only one dimension provided, calculate the other to maintain aspect ratio
    if requested_width and not requested_height:
        aspect_ratio = original.width / original.height
        output_height = int(requested_width / aspect_ratio)
    elif requested_height and not requested_width:
        aspect_ratio = original.width / original.height
        output_width = int(requested_height * aspect_ratio)
    
    # Round to multiple of 8 while preserving aspect ratio
    aspect_ratio = output_width / output_height
    original_ratio = original.width / original.height
    
    # Try rounding width first, then calculate height
    rounded_width = _round_to_multiple_of_8(output_width)
    height_from_width = _round_to_multiple_of_8(int(rounded_width / aspect_ratio))
    
    # Try rounding height first, then calculate width
    rounded_height = _round_to_multiple_of_8(output_height)
    width_from_height = _round_to_multiple_of_8(int(rounded_height * aspect_ratio))
    
    # Choose the option that better preserves the original aspect ratio
    width_first_ratio = rounded_width / height_from_width
    height_first_ratio = width_from_height / rounded_height
    
    width_first_diff = abs(width_first_ratio - original_ratio)
    height_first_diff = abs(height_first_ratio - original_ratio)
    
    if height_first_diff < width_first_diff:
        return width_from_height, rounded_height
    else:
        return rounded_width, height_from_width


def _is_qwen_pipeline(pipeline: Any) -> bool:
    """
    Check if pipeline is a QwenImageEditPlusPipeline.
    
    Args:
        pipeline: Pipeline instance to check
    
    Returns:
        True if pipeline is QwenImageEditPlusPipeline, False otherwise
    """
    try:
        from diffusers import QwenImageEditPlusPipeline  # type: ignore
        return isinstance(pipeline, QwenImageEditPlusPipeline)
    except (ImportError, AttributeError):
        try:
            return "QwenImageEditPlus" in pipeline.__class__.__name__
        except Exception:
            return False


def _extract_generated_image(result: Any) -> Image.Image:
    """
    Extract generated image from pipeline result.
    
    Args:
        result: Pipeline result (can be various formats)
    
    Returns:
        Generated PIL Image
    
    Note:
        Handles different result formats:
        - result.images[0] (DiffusionPipeline format)
        - result[0] (list/tuple format)
        - result (direct Image format)
    """
    if hasattr(result, "images"):
        return result.images[0] if isinstance(result.images, list) else result.images
    elif isinstance(result, (list, tuple)):
        return result[0]
    else:
        return result


def _calculate_image_difference(original: Image.Image, generated: Image.Image) -> Optional[Dict[str, float]]:
    """
    Calculate pixel-level difference between original and generated images.
    
    Args:
        original: Original image
        generated: Generated image
    
    Returns:
        Dictionary with difference metrics, or None if calculation fails
    
    Note:
        Used for verification that generation actually changed the image.
    """
    try:
        original_array = np.array(original.convert("RGB"))
        generated_array = np.array(generated.convert("RGB"))
        
        if original_array.shape != generated_array.shape:
            logger.warning(
                f"Generated image shape {generated_array.shape} != original {original_array.shape}"
            )
            return None
        
        pixel_diff = np.abs(original_array.astype(np.int16) - generated_array.astype(np.int16))
        max_diff = float(np.max(pixel_diff))
        mean_diff = float(np.mean(pixel_diff))
        diff_percentage = (np.count_nonzero(pixel_diff) / pixel_diff.size) * 100
        
        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "diff_percentage": diff_percentage,
        }
    except Exception as e:
        logger.debug(f"Failed to calculate image difference: {e}")
        return None


# ============================================================================
# GenerationService Class
# ============================================================================

class GenerationService:
    """
    Service for image generation using Qwen pipeline.
    
    Handles:
    - Task type determination (insertion, removal, white-balance)
    - Pipeline loading and management
    - Image preprocessing and conditional building
    - Generation execution with progress tracking
    - Cancellation support
    """
    
    # Class-level cancellation flags for generation tasks
    # Used by async generation worker (qwen_worker) to check cancellation
    _cancellation_flags: Dict[str, bool] = {}
    
    def __init__(self) -> None:
        # Pipeline is now managed centrally by qwen_loader, no need to cache here
        pass
    
    @classmethod
    def _check_cancellation(cls, task_id: str) -> bool:
        """Check if task is cancelled."""
        return cls._cancellation_flags.get(task_id, False)
    
    @classmethod
    def set_cancelled(cls, task_id: str) -> None:
        """Mark task as cancelled."""
        cls._cancellation_flags[task_id] = True
    
    @classmethod
    def clear_cancellation(cls, task_id: str) -> None:
        """Clear cancellation flag for task."""
        cls._cancellation_flags.pop(task_id, None)

    def _ensure_pipeline(
        self,
        task_type: str = "insertion",
        enable_flowmatch_scheduler: Optional[bool] = None,
        on_loading_progress: Optional[Callable[[str, float]]] = None,
    ) -> Any:
        """
        Get or load pipeline for the specified task type with optional optimization flags.
        
        Args:
            task_type: "insertion", "removal", or "white-balance"
            enable_flowmatch_scheduler: Override config setting (None = use config)
            on_loading_progress: Optional callback for loading progress
        
        Returns:
            Loaded DiffusionPipeline (single instance shared across all tasks)
        """
        # Pipeline is managed centrally, just load/switch adapter as needed
        return load_pipeline(
            task_type=task_type,
            enable_flowmatch_scheduler=enable_flowmatch_scheduler,
            on_loading_progress=on_loading_progress,
        )

    def _resolve_quality(self, override: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolve quality preset to (quality_label, preset_type).
        
        Args:
            override: Optional quality override string
        
        Returns:
            Tuple of (quality_label, preset_type) where preset_type is "1:1" or "keep"
        
        Raises:
            ValueError: If override is invalid
        """
        presets = settings.input_quality_presets
        quality_label = settings.input_quality
        preset_type = settings.input_quality_preset

        if override:
            normalized = override.strip().lower()
            if normalized in presets:
                quality_label = normalized
                preset_type = presets[normalized]
            else:
                raise ValueError(
                    f"Unknown input quality override '{override}'. "
                    f"Valid options: {list(presets.keys())}"
                )
        return quality_label, preset_type

    def _apply_input_quality(
        self,
        original: Image.Image,
        mask: Optional[Image.Image] = None,
        reference: Optional[Image.Image] = None,
        quality_override: Optional[str] = None,
    ) -> Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]:
        """
        Apply input quality preset: resize to 1:1 aspect ratio or keep original.
        
        Args:
            original: Original image
            mask: Optional mask image
            reference: Optional reference image
            quality_override: Optional quality override
        
        Returns:
            Tuple of (processed_original, processed_mask, processed_reference)
        """
        quality_label, preset_type = self._resolve_quality(quality_override)

        if preset_type == "keep":
            return original, mask, reference

        if preset_type == "1:1":
            max_dim = max(original.width, original.height)
            target_size = (
                512 if max_dim <= 640 else
                768 if max_dim <= 896 else
                1024 if max_dim <= 1280 else
                1536 if max_dim <= 1792 else
                2048
            )
            new_size = (target_size, target_size)
            original = resize_with_aspect_ratio_pad(original, new_size, background_color=(0, 0, 0))
            if mask is not None:
                mask = resize_with_aspect_ratio_pad(mask, new_size, background_color=(0, 0, 0))
            if reference is not None:
                reference = resize_with_aspect_ratio_pad(reference, new_size, background_color=(0, 0, 0))
            return original, mask, reference

        raise ValueError(f"Unknown preset_type: {preset_type}. Expected 'keep' or '1:1'")

    def _prepare_images(
        self, request: GenerationRequest, task_type: str = "removal"
    ) -> Tuple[Image.Image, Image.Image, list[Image.Image], Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
        """
        Prepare images for generation (decode, resize, build conditionals).
        
        Args:
            request: Generation request
            task_type: Task type ("insertion", "removal", "white-balance")
        
        Returns:
            Tuple of (original, mask, conditional_images, reference_image, mask_A_for_compositing, mask_mae_dilated)
        
        Raises:
            HTTPException: If required inputs are missing
        """
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")

        original = base64_to_image(request.input_image)
        
        # New Qwen flow:
        # - conditional_images[0]: mask
        # - conditional_images[1:]: additional conditions (background masked, canny, mae, ...)
        cond_list = list(request.conditional_images or [])
        extra_cond_b64s: list[str] = []

        if task_type == "white-balance":
            # White-balance: conditionals are [input_image, canny_edge]
            # No mask needed for white-balance
            original, _, _ = self._apply_input_quality(
                original, quality_override=request.input_quality
            )
            conditional_images = build_white_balance_conditionals(
                original,
                canny_thresholds=(100, 200),
                use_cache=True,
            )
            mask = Image.new("RGB", original.size, (255, 255, 255))
            mask_mae_dilated = None
        else:
            # insertion / removal: must have at least 1 condition as mask
            if not cond_list:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "conditional_images must include at least one image "
                        "(mask) for insertion/removal tasks."
                    ),
                )
            mask_b64 = cond_list[0]
            extra_cond_b64s = cond_list[1:]
            
            mask = base64_to_image(mask_b64)
            original, mask, _ = self._apply_input_quality(
                original, mask, None, quality_override=request.input_quality
            )
            
            if mask is None:
                raise RuntimeError("Mask image missing after preprocessing")

            # For insertion: handle two-source mask workflow if reference_mask_R is provided
            # Mask A (placement): mask from conditional_images[0] (already decoded above)
            # Mask R (object shape): reference_mask_R from request
            ref_img = None
            
            if task_type == "insertion":
                if request.reference_mask_R and request.reference_image:
                    reference_image = base64_to_image(request.reference_image)
                    mask_R = base64_to_image(request.reference_mask_R)
                    extracted_object, _ = prepare_reference_conditionals(reference_image, mask_R)
                    ref_img = extracted_object
                    if ref_img.size != original.size:
                        ref_img = ref_img.resize(original.size, Image.Resampling.LANCZOS)
                elif request.reference_image:
                    ref_img = base64_to_image(request.reference_image)
                    if ref_img.size != original.size:
                        ref_img = ref_img.resize(original.size, Image.Resampling.LANCZOS)
            
            if task_type == "insertion" and ref_img is not None:
                conditional_images = build_insertion_conditionals(
                    original=original,
                    mask=mask,
                    ref_img=ref_img,
                    use_cache=True,
                )
                mask_mae_dilated = None
            elif task_type == "removal":
                # Get enable_mae_refinement from request, default to True if not provided
                request_mae_refinement = getattr(request, 'enable_mae_refinement', None)
                enable_mae_refinement = (
                    request_mae_refinement 
                    if request_mae_refinement is not None
                    else True
                )
                logger.info(
                    f"🔍 [Removal Task] MAE refinement setting: "
                    f"request.enable_mae_refinement={request_mae_refinement}, "
                    f"final enable_mae_refinement={enable_mae_refinement}"
                )
                conditional_images, mask_mae_dilated = build_removal_conditionals(
                    original=original,
                    mask=mask,
                    enable_mae=True,
                    use_cache=True,
                    request_id=None,
                    mask_tool_type=request.mask_tool_type,
                    enable_mae_refinement=enable_mae_refinement,
                )
            else:
                mask_cond, masked_bg, _, _ = prepare_mask_conditionals(
                    original, mask, include_mae=False
                )
                conditional_images = [mask_cond, masked_bg]
                mask_mae_dilated = None
        
        # Add extra conditional images if provided
        if extra_cond_b64s:
            for img_b64 in extra_cond_b64s:
                extra_img = base64_to_image(img_b64)
                if extra_img.size != original.size:
                    extra_img = extra_img.resize(original.size, Image.Resampling.LANCZOS)
                conditional_images.append(extra_img)
        
        # Return ref_img and mask_A for post-processing if using two-source mask workflow
        # mask_A is the placement mask (from conditional_images[0])
        use_two_source_masks_for_return = (
            task_type == "insertion" 
            and request.reference_mask_R is not None 
            and request.reference_image is not None
        )
        mask_A_for_compositing = mask if use_two_source_masks_for_return else None
        
        return (
            original,
            mask,
            conditional_images,
            ref_img if task_type == "insertion" else None,
            mask_A_for_compositing,
            mask_mae_dilated
        )

    def _build_params(
        self,
        request: GenerationRequest,
        original: Image.Image,
        mask: Image.Image,
        conditional_images: list[Image.Image],
        task_type: str = "removal",
    ) -> Dict[str, Any]:
        """
        Build parameters for StableDiffusionInpaintPipeline.
        
        Args:
            request: Generation request
            original: Original image
            mask: Mask image
            conditional_images: List of conditional images
            task_type: Task type (for documentation)
        
        Returns:
            Dictionary of pipeline parameters
        """
        output_width, output_height = _calculate_output_dimensions(
            original, request.width, request.height
        )
        
        params: Dict[str, Any] = {
            "prompt": request.prompt,
            "image": original,
            "mask_image": mask,
            "conditional_images": conditional_images,
            "num_inference_steps": request.num_inference_steps or 10,
            "guidance_scale": request.guidance_scale or 4.0,
            "width": output_width,
            "height": output_height,
        }

        if request.seed is not None:
            import torch  # Lazy import
            device = get_device()
            params["generator"] = torch.Generator(device=device).manual_seed(request.seed)

        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt

        if request.true_cfg_scale is not None:
            params["true_cfg_scale"] = request.true_cfg_scale

        return params

    def _build_qwen_params(
        self,
        request: GenerationRequest,
        original: Image.Image,
        conditional_images: list[Image.Image],
        task_type: str,
        progress_callback: Optional[Callable[[int, int, int], None]],
        task_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build parameters for QwenImageEditPlusPipeline.
        
        Args:
            request: Generation request
            original: Original image
            conditional_images: List of conditional images
            task_type: Task type (affects default true_cfg_scale)
            progress_callback: Optional progress callback
            task_id: Optional task ID for cancellation tracking
        
        Returns:
            Dictionary of Qwen pipeline parameters
        
        Note:
            - guidance_scale is ineffective for Qwen. Use true_cfg_scale instead.
            - negative_prompt is required (even empty string) to enable classifier-free guidance.
        """
        # Note: guidance_scale is ineffective for Qwen. Use true_cfg_scale instead.
        # negative_prompt is required (even empty string) to enable classifier-free guidance
        true_cfg = request.true_cfg_scale or request.guidance_scale or (
            3.3 if task_type == "white-balance" else 4.0
        )
        
        qwen_params: Dict[str, Any] = {
            "image": conditional_images,
            "prompt": request.prompt or "",
            "num_inference_steps": request.num_inference_steps or 10,
            "true_cfg_scale": true_cfg,
            "height": original.height,
            "width": original.width,
            # Always provide negative_prompt (even if empty) to enable classifier-free guidance
            "negative_prompt": request.negative_prompt if request.negative_prompt else " ",
        }
        
        if request.seed is not None:
            import torch  # Lazy import
            device = get_device()
            qwen_params["generator"] = torch.Generator(device=device).manual_seed(request.seed)
        
        # Add progress callback if provided
        if progress_callback is not None:
            num_steps = qwen_params["num_inference_steps"]
            task_id_for_cancel = (
                task_id or
                getattr(request, 'request_id', None) or
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            
            def callback_on_step_end(
                pipeline_instance: Any,
                step: int,
                timestep: int,
                callback_kwargs: dict
            ) -> dict:
                """
                Callback function for pipeline progress tracking with cancellation check.
                
                Args:
                    pipeline_instance: Pipeline instance (unused)
                    step: Current step number
                    timestep: Current timestep (unused)
                    callback_kwargs: Callback kwargs to return
                
                Returns:
                    Callback kwargs dict
                
                Raises:
                    CancelledError: If task is cancelled
                """
                try:
                    # Check cancellation flag
                    if self._check_cancellation(task_id_for_cancel):
                        logger.warning(f"⚠️ Generation cancelled at step {step}/{num_steps}")
                        raise CancelledError(
                            task_id_for_cancel,
                            f"Generation cancelled at step {step}"
                        )
                    
                    progress_callback(step, timestep, num_steps)
                except CancelledError:
                    raise  # Re-raise cancellation error
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
                return callback_kwargs
            
            qwen_params["callback_on_step_end"] = callback_on_step_end
        
        return qwen_params

    def _initialize_generation_metrics(self) -> Dict[str, Any]:
        """
        Initialize research metrics and record initial memory state.
        
        Returns:
            Dictionary with initialized metrics structure
        """
        research_metrics: Dict[str, Any] = {
            "timing": {},
            "memory": {},
            "image_stats": {},
            "mask_stats": {},
            "device_info": {},
        }
        
        try:
            import torch  # Lazy import
            if torch.cuda.is_available():
                research_metrics["memory"]["before_generation"] = {
                    "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
                }
            research_metrics["device_info"] = get_device_info()
        except Exception as e:
            logger.warning(f"Failed to record initial memory state: {e}")
        
        return research_metrics

    def _determine_task_type(self, request: GenerationRequest) -> str:
        """
        Determine and validate task type from request.
        
        Args:
            request: Generation request
        
        Returns:
            Task type string ("insertion", "removal", or "white-balance")
        
        Raises:
            HTTPException: If task_type is invalid
        """
        if request.task_type:
            task_type = request.task_type
            valid_task_types = {"insertion", "removal", "white-balance"}
            if task_type not in valid_task_types:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid task_type: {task_type}. "
                        f"Must be one of: {', '.join(valid_task_types)}"
                    )
                )
        else:
            task_type = "removal"
        
        return task_type

    def _verify_generated_image(
        self,
        original: Image.Image,
        generated: Image.Image,
        task_type: str,
        conditionals_count: int = 0
    ) -> None:
        """
        Verify that generated image is different from original.
        
        Args:
            original: Original image
            generated: Generated image
            task_type: Task type (for logging)
            conditionals_count: Number of conditionals used (for logging)
        
        Note:
            Logs warnings if images are too similar (hash match, pixel match, or <1% different).
            This helps detect cases where generation didn't actually change the image.
        """
        try:
            # Check hash equality (fast check)
            original_bytes = io.BytesIO()
            original.save(original_bytes, format="PNG")
            original_hash = hashlib.md5(original_bytes.getvalue()).hexdigest()
            
            generated_bytes = io.BytesIO()
            generated.save(generated_bytes, format="PNG")
            generated_hash = hashlib.md5(generated_bytes.getvalue()).hexdigest()
            
            if original_hash == generated_hash:
                logger.warning(
                    f"Generated image identical to original (hash: {original_hash[:8]}...)"
                )
                return
            
            # Check pixel-level difference
            diff_metrics = _calculate_image_difference(original, generated)
            if diff_metrics is None:
                return
            
            if diff_metrics["max_diff"] == 0 and diff_metrics["mean_diff"] == 0:
                logger.warning("Generated image pixels identical to original (0 difference)")
            elif diff_metrics["diff_percentage"] < 1.0:
                logger.warning(
                    f"Generated image very similar to original "
                    f"({diff_metrics['diff_percentage']:.2f}% different)"
                )
        except Exception as e:
            logger.debug(f"Failed to verify generated image: {e}")

    def _execute_reference_guided_insertion(
        self,
        request: GenerationRequest,
        pipeline_callable: Any,
        debug_session: Any,
        research_metrics: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        task_id: Optional[str] = None
    ) -> Tuple[Image.Image, Dict[str, Any], Optional[Image.Image], list, Image.Image]:
        """
        Execute reference-guided insertion workflow.
        
        Args:
            request: Generation request
            pipeline_callable: Pipeline callable function
            debug_session: Debug session for saving images
            research_metrics: Research metrics dictionary
            progress_callback: Optional progress callback
            task_id: Optional task ID for cancellation
        
        Returns:
            Tuple of (generated, params, reference_image, conditionals_for_debug, positioned_mask_R)
        
        Raises:
            HTTPException: If required inputs are missing
        """
        original = base64_to_image(request.input_image)
        if not request.conditional_images:
            raise HTTPException(
                status_code=400,
                detail="conditional_images[0] (main_mask_A) is required"
            )
        if not request.reference_image:
            raise HTTPException(status_code=400, detail="reference_image is required")
        if not request.reference_mask_R:
            raise HTTPException(status_code=400, detail="reference_mask_R is required")
        
        main_mask_A = base64_to_image(request.conditional_images[0])
        reference_image = base64_to_image(request.reference_image)
        reference_mask_R = base64_to_image(request.reference_mask_R)
        
        # Apply input quality
        original, main_mask_A_processed, _ = self._apply_input_quality(
            original, main_mask_A, None, quality_override=request.input_quality
        )
        if main_mask_A_processed is None:
            raise RuntimeError("main_mask_A is None after input quality processing")
        main_mask_A = main_mask_A_processed
        
        generator = None
        if request.seed is not None:
            import torch  # Lazy import
            generator = torch.Generator(device=get_device()).manual_seed(request.seed)
        
        debug_session.save_image(original, "input_image", "01", "Original base image")
        debug_session.save_image(main_mask_A, "main_mask_A", "02", "Main mask A")
        debug_session.save_image(reference_image, "reference_image", "03", "Reference image")
        debug_session.save_image(reference_mask_R, "reference_mask_R", "04", "Reference mask R")
        
        # Execute pipeline
        inference_start = time.time()
        generated, positioned_mask_R = execute_insertion_pipeline(
            base_image=original,
            main_mask_A=main_mask_A,
            reference_image=reference_image,
            reference_mask_R=reference_mask_R,
            pipeline=pipeline_callable,
            prompt=request.prompt or "",
            num_inference_steps=request.num_inference_steps or 10,
            true_cfg_scale=request.true_cfg_scale or request.guidance_scale or 4.0,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            generator=generator,
            debug_session=debug_session,
            progress_callback=progress_callback,
        )
        inference_time = time.time() - inference_start
        research_metrics["timing"]["inference_seconds"] = round(inference_time, 3)
        
        self._verify_generated_image(original, generated, "insertion (reference-guided)", 0)
        
        research_metrics["image_stats"]["output"] = {
            "size": list(generated.size),
            "mode": generated.mode,
            "width": generated.width,
            "height": generated.height,
            "aspect_ratio": round(generated.width / generated.height, 3),
        }
        debug_session.save_image(generated, "generated_output", "09", "Final output")
        
        true_cfg = request.true_cfg_scale or request.guidance_scale or 4.0
        params = {
            "prompt": request.prompt or "",
            "num_inference_steps": request.num_inference_steps or 10,
            "guidance_scale": true_cfg,
            "true_cfg_scale": true_cfg,
            "width": original.width,
            "height": original.height,
            "pipeline_type": "reference_guided_insertion",
            "two_source_masks": True,
        }
        
        from ..services.mask_segmentation_service import extract_object_with_mask
        masked_R_image = extract_object_with_mask(reference_image, reference_mask_R)
        debug_session.save_image(masked_R_image, "masked_R", "07", "Masked R")
        
        # Return conditionals for debug info: [original, mask_A, masked_R, reference_image, positioned_mask_R]
        # This matches what frontend expects for insertion debug display
        conditionals_for_debug = [
            original,          # 1. Original input image
            main_mask_A,       # 2. Mask A (placement mask on original)
            masked_R_image,    # 3. Masked R (extracted object from reference)
            reference_image,   # 4. Reference image (original)
            positioned_mask_R, # 5. Positioned mask R (reference mask R pasted into main mask A)
        ]
        
        return generated, params, reference_image, conditionals_for_debug, positioned_mask_R

    def _execute_standard_workflow(
        self,
        request: GenerationRequest,
        task_type: str,
        pipeline: Any,
        pipeline_callable: Any,
        debug_session: Any,
        research_metrics: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        task_id: Optional[str] = None
    ) -> Tuple[Image.Image, Dict[str, Any], Optional[Image.Image], list]:
        """
        Execute standard workflow (removal/insertion/white-balance).
        
        Args:
            request: Generation request
            task_type: Task type ("insertion", "removal", "white-balance")
            pipeline: Pipeline instance (for type detection)
            pipeline_callable: Pipeline callable function
            debug_session: Debug session for saving images
            research_metrics: Research metrics dictionary
            progress_callback: Optional progress callback
            task_id: Optional task ID for cancellation
        
        Returns:
            Tuple of (generated, params, reference, conditionals)
        """
        # Prepare images
        prep_start = time.time()
        original, mask, conditionals, reference, mask_A_for_compositing, mask_mae_dilated = self._prepare_images(
            request, task_type
        )
        prep_time = time.time() - prep_start
        research_metrics["timing"]["preprocessing_seconds"] = round(prep_time, 3)
        
        # Calculate mask statistics
        if mask is not None:
            try:
                mask_array = np.array(mask.convert("L"))
                white_pixels = np.sum(mask_array > 128)
                total_pixels = mask_array.size
                coverage = (white_pixels / total_pixels) * 100
                research_metrics["mask_stats"] = {
                    "coverage_percent": round(coverage, 2),
                    "white_pixels": int(white_pixels),
                    "total_pixels": int(total_pixels),
                    "mask_size": list(mask.size),
                }
            except Exception as e:
                logger.warning(f"Failed to calculate mask stats: {e}")
        
        # Save input image statistics
        research_metrics["image_stats"]["input"] = {
            "size": list(original.size),
            "mode": original.mode,
            "width": original.width,
            "height": original.height,
            "aspect_ratio": round(original.width / original.height, 3),
        }
        
        # Save debug images
        debug_session.save_image(original, "input_image", "01", "Original input image")
        
        # Save conditional images based on task type
        if task_type == "white-balance":
            if len(conditionals) > 0:
                debug_session.save_image(conditionals[0], "wb_input", "02", "White-balance input image")
            if len(conditionals) > 1:
                debug_session.save_image(conditionals[1], "canny_edge", "03", "Canny edge detection")
        else:
            debug_session.save_image(mask, "mask_image", "02", "Mask image")
            if len(conditionals) > 1:
                # Ẩn masked_bg cho removal task (không lưu vào debug session)
                if task_type != "removal":
                    debug_session.save_image(
                        conditionals[1],
                        "masked_bg",
                        "03",
                        "Masked background (original with mask area removed)"
                    )
            if len(conditionals) > 2:
                if task_type == "removal":
                    debug_session.save_image(conditionals[2], "mae_output", "05", "MAE inpainted preview")
                else:
                    debug_session.save_image(
                        conditionals[2],
                        "ref_img",
                        "05",
                        "Reference image (from frontend upload)"
                    )
        
        if reference is not None:
            debug_session.save_image(reference, "reference_image", "04", "Reference image (optional)")
        
        inference_start = time.time()
        
        import torch  # Lazy import
        with torch.no_grad():
            is_qwen = _is_qwen_pipeline(pipeline)
            
            if is_qwen:
                image_list = conditionals if conditionals else []
                qwen_params = self._build_qwen_params(
                    request, original, image_list, task_type, progress_callback, task_id
                )
                
                result = pipeline_callable(**qwen_params)
                generated = _extract_generated_image(result)
                self._verify_generated_image(original, generated, task_type, len(image_list))
                
                params = {
                    "prompt": qwen_params.get("prompt", request.prompt or ""),
                    "num_inference_steps": qwen_params["num_inference_steps"],
                    "guidance_scale": qwen_params.get("true_cfg_scale", request.guidance_scale or 4.0),
                    "true_cfg_scale": qwen_params.get("true_cfg_scale"),
                    "width": qwen_params["width"],
                    "height": qwen_params["height"],
                }
            else:
                # For standard StableDiffusionInpaintPipeline
                params = self._build_params(request, original, mask, conditionals, task_type=task_type)
                result = pipeline_callable(**params)
                generated = _extract_generated_image(result)
                self._verify_generated_image(original, generated, task_type, len(conditionals))
            
            # Ensure generated image matches original size
            if generated.size != original.size:
                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            
            debug_session.save_image(generated, "generated_output", "07", "Final generated output")
        
        inference_time = time.time() - inference_start
        research_metrics["timing"]["inference_seconds"] = round(inference_time, 3)
        
        # Store mask_mae_dilated in research_metrics for access in generate()
        research_metrics["mask_mae_dilated"] = mask_mae_dilated
        
        return generated, params, reference, conditionals

    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        on_pipeline_loaded: Optional[Callable[[], None]] = None,
        on_loading_progress: Optional[Callable[[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Generate image using Qwen pipeline.
        
        Uses sync def because:
        - Runs on Heavy Worker (A100) where PyTorch/GPU tasks are blocking
        - Modal automatically wraps sync functions in thread pool
        - No benefit from async for CPU/GPU bound tasks (Python GIL)
        
        Args:
            request: Generation request
            progress_callback: Optional callback function(step: int, timestep: int, total_steps: int)
            on_pipeline_loaded: Optional callback when pipeline is loaded
            on_loading_progress: Optional callback for loading progress
        
        Returns:
            Dictionary with generation results including:
            - success: bool
            - generated_pil: PIL.Image.Image object (not base64 for memory efficiency)
            - generation_time: float
            - model_used: str
            - parameters_used: dict
            - request_id: str
            - debug_info: dict
        
        Raises:
            HTTPException: If generation fails
            CancelledError: If task is cancelled
        """
        start = time.time()
        
        # Generate task_id for cancellation tracking
        task_id = (
            getattr(request, 'request_id', None) or
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        
        # Create debug session only if explicitly enabled
        # Check both request.enable_debug and global DEBUG_ENABLED
        enable_debug = getattr(request, 'enable_debug', False) or DEBUG_ENABLED
        debug_session = debug_service.create_session() if enable_debug else DebugSession(enabled=False)
        research_metrics = self._initialize_generation_metrics()

        try:
            if self._check_cancellation(task_id):
                raise CancelledError(task_id, "Task was cancelled before generation started")
            
            task_type = self._determine_task_type(request)
            
            if task_type == "removal":
                request.prompt = "remove object"
            
            optimization_flags = {
                "enable_flowmatch_scheduler": request.enable_flowmatch_scheduler,
            }
            
            # Load pipeline and record timing
            pipeline_load_start = time.time()
            pipeline = self._ensure_pipeline(
                task_type=task_type,
                on_loading_progress=on_loading_progress,
                **optimization_flags
            )
            pipeline_callable = cast(Any, pipeline)
            pipeline_load_time = time.time() - pipeline_load_start
            research_metrics["timing"]["pipeline_load_seconds"] = round(pipeline_load_time, 3)
            
            if on_pipeline_loaded:
                on_pipeline_loaded()
            
            from ..core.qwen_loader import _current_adapter, _loaded_adapters
            debug_session.save_lora_info({
                "task_type": task_type,
                "loaded_adapters": list(_loaded_adapters),
                "current_adapter": _current_adapter,
            })
            
            # Check if using reference-guided insertion pipeline
            use_reference_guided_insertion = (
                task_type == "insertion"
                and request.reference_mask_R is not None
                and request.reference_image is not None
            )
            
            positioned_mask_R = None  # Only set for reference-guided insertion
            if use_reference_guided_insertion:
                # Execute reference-guided insertion workflow
                original = base64_to_image(request.input_image)
                generated, params, reference, conditionals, positioned_mask_R = (
                    self._execute_reference_guided_insertion(
                        request, pipeline_callable, debug_session, research_metrics,
                        progress_callback, task_id
                    )
                )
            else:
                # Execute standard workflow
                original = base64_to_image(request.input_image)
                generated, params, reference, conditionals = self._execute_standard_workflow(
                    request, task_type, pipeline, pipeline_callable, debug_session,
                    research_metrics, progress_callback, task_id
                )

            generation_time = time.time() - start
            
            # Record final memory state
            try:
                import torch  # Lazy import
                if torch.cuda.is_available():
                    research_metrics["memory"]["after_generation"] = {
                        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
                    }
                    # Calculate memory delta
                    if "before_generation" in research_metrics["memory"]:
                        before = research_metrics["memory"]["before_generation"]
                        after = research_metrics["memory"]["after_generation"]
                        research_metrics["memory"]["delta"] = {
                            "allocated_delta_mb": round(
                                after["allocated_mb"] - before["allocated_mb"], 2
                            ),
                            "reserved_delta_mb": round(
                                after["reserved_mb"] - before["reserved_mb"], 2
                            ),
                        }
            except Exception as e:
                logger.warning(f"Failed to record final memory state: {e}")
            
            # Save output image statistics
            research_metrics["image_stats"]["output"] = {
                "size": list(generated.size),
                "mode": generated.mode,
                "width": generated.width,
                "height": generated.height,
                "aspect_ratio": round(generated.width / generated.height, 3),
            }
            
            # Complete timing breakdown
            research_metrics["timing"]["total_seconds"] = round(generation_time, 3)
            research_metrics["timing"]["other_seconds"] = round(
                generation_time
                - research_metrics["timing"].get("pipeline_load_seconds", 0)
                - research_metrics["timing"].get("preprocessing_seconds", 0)
                - research_metrics["timing"].get("inference_seconds", 0),
                3
            )
            
            debug_session.save_research_metrics(research_metrics)
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.clear_cancellation(task_id)
            
            metadata = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": params["num_inference_steps"],
                "guidance_scale": params["guidance_scale"],
                "true_cfg_scale": params.get("true_cfg_scale"),
                "seed": request.seed,
                "image_size": f"{params['width']}x{params['height']}",
                "generation_time": round(generation_time, 2),
                "timestamp": datetime.now().isoformat(),
                "has_reference_image": reference is not None,
                "task_type": task_type,
            }
            
            actual_prompt = params.get("prompt", request.prompt)
            
            # Save generation parameters to debug session
            debug_session.save_parameters({
                "task_type": task_type,
                "prompt": actual_prompt,  # Use actual prompt used in generation
                "original_prompt": request.prompt,  # Also save original prompt from request
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": params["num_inference_steps"],
                "guidance_scale": params["guidance_scale"],
                "true_cfg_scale": params.get("true_cfg_scale"),
                "seed": request.seed,
                "width": params["width"],
                "height": params["height"],
                "input_quality": request.input_quality,
                "generation_time": round(generation_time, 2),
            })
            
            debug_session.finalize(success=True)
            
            # Build debug_info
            from ..core.qwen_loader import _current_adapter, _loaded_adapters
            
            # Determine conditional labels based on task type.
            # IMPORTANT: These labels must match the actual conditional order
            # returned from the condition_builder helpers:
            # - build_white_balance_conditionals  → [input, canny]
            # - build_removal_conditionals        → [masked_bg, mask, mae]
            # - build_insertion_conditionals      → [ref_img, mask, masked_bg]
            if task_type == "white-balance":
                conditional_labels = ["input", "canny"]
                max_conditionals = 2
            elif task_type == "removal":
                # For removal task debug display:
                # - Thêm original background vào đầu (chỉ cho debug, không ảnh hưởng pipeline)
                # - Bỏ masked_bg khỏi debug display (ẩn)
                # - Pipeline vẫn nhận [masked_bg, mask, mae] như bình thường
                conditional_labels = ["original", "mask", "mae"]
                max_conditionals = 3
            elif use_reference_guided_insertion:
                conditional_labels = [
                    "original", "mask_A", "masked_R", "reference_image", "positioned_mask_R"
                ]
                max_conditionals = 5
            else:
                # Standard insertion (non reference-guided):
                # conditionals are [ref_img, mask, masked_bg]
                conditional_labels = ["ref_img", "mask", "masked_bg"]
                max_conditionals = 3
            
            # Build conditional images for debug display
            if task_type == "removal":
                # Cho removal task: thêm original vào đầu, bỏ masked_bg (index 0)
                # conditionals = [masked_bg, mask, mae] từ pipeline
                # debug display = [original, mask, mae]
                if len(conditionals) >= 3:
                    conditional_images_b64 = [
                        image_to_base64(original),  # Thêm original background
                        image_to_base64(conditionals[1]),  # mask (index 1)
                        image_to_base64(conditionals[2]),  # mae (index 2)
                    ]
                else:
                    # Fallback nếu không đủ conditionals
                    conditional_images_b64 = [
                        image_to_base64(cond_img)
                        for cond_img in conditionals[:max_conditionals]
                    ]
            else:
                conditional_images_b64 = [
                    image_to_base64(cond_img)
                    for cond_img in conditionals[:max_conditionals]
                ]
            
            debug_info: Dict[str, Any] = {
                "conditional_images": conditional_images_b64,
                "conditional_labels": conditional_labels[:len(conditional_images_b64)],
                "input_image_size": f"{original.width}x{original.height}",
                "output_image_size": f"{generated.width}x{generated.height}",
                "lora_adapter": _current_adapter,
                "loaded_adapters": list(_loaded_adapters) if _loaded_adapters else [],
                "original_image": request.input_image,
                "mask_A": request.conditional_images[0] if request.conditional_images else None,
                "reference_image": request.reference_image,
                "reference_mask_R": request.reference_mask_R,
            }
            
            # Add dilated mask for MAE (removal task with brush mask only)
            if task_type == "removal":
                mask_mae_dilated = research_metrics.get("mask_mae_dilated")
                # Check if mask_mae_dilated exists and is not None (it's a PIL Image)
                if mask_mae_dilated is not None and hasattr(mask_mae_dilated, 'size'):
                    logger.info(
                        f"🔍 [Debug Info] Adding dilated mask for MAE to debug_info: "
                        f"size={mask_mae_dilated.size}, mode={mask_mae_dilated.mode}, "
                        f"mask_tool_type={request.mask_tool_type}"
                    )
                    debug_info["mask_mae_dilated"] = image_to_base64(mask_mae_dilated)
                    logger.info("✅ [Debug Info] mask_mae_dilated added to debug_info")
                else:
                    logger.debug(
                        f"🔍 [Debug Info] No dilated mask for MAE: "
                        f"mask_tool_type={request.mask_tool_type}, "
                        f"mask_mae_dilated={mask_mae_dilated}, "
                        f"has_size_attr={hasattr(mask_mae_dilated, 'size') if mask_mae_dilated is not None else False}"
                    )
            
            if use_reference_guided_insertion and positioned_mask_R is not None:
                mask_R_for_debug = (
                    positioned_mask_R.convert("L")
                    if positioned_mask_R.mode != "L"
                    else positioned_mask_R
                )
                debug_info["positioned_mask_R"] = image_to_base64(mask_R_for_debug)
            
            if DEBUG_ENABLED:
                session_path = debug_session.get_session_path()
                if session_path:
                    debug_info["session_name"] = session_path.name
                    debug_info["debug_path"] = str(session_path)
                    debug_info["debug_note"] = (
                        "Debug session files are stored in H200 worker container "
                        "and may not be accessible from Job Manager Service"
                    )
            
            # Build response
            # Return PIL Image object instead of base64 for memory efficiency
            # modal_app.py will encode to base64 only when needed for job_state_dictio
            response: Dict[str, Any] = {
                "success": True,
                "generated_pil": generated,  # PIL Image object (not base64)
                "generation_time": round(generation_time, 2),
                "model_used": f"qwen_local_{task_type}",
                "parameters_used": metadata,
                "request_id": request_id,  # Include request_id for visualization access
                "debug_info": debug_info,
            }
            
            # Include debug path if enabled
            if DEBUG_ENABLED and debug_session.get_session_path():
                response["debug_path"] = str(debug_session.get_session_path())
            
            return response
        
        except HTTPException:
            debug_session.log_lora("HTTPException raised")
            debug_session.finalize(success=False, error="HTTPException")
            raise
        except ValueError as exc:
            debug_session.log_lora(f"ValueError: {exc}")
            debug_session.finalize(success=False, error=f"ValueError: {exc}")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            debug_session.log_lora(f"RuntimeError: {exc}")
            debug_session.finalize(success=False, error=f"RuntimeError: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error during generation")
            debug_session.log_lora(f"Unexpected error: {exc}")
            debug_session.finalize(success=False, error=f"Unexpected error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
