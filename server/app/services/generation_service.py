from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, cast, Callable

import numpy as np
from PIL import Image

# torch is imported lazily where needed to avoid requiring it in services that don't need it (e.g., ImageUtilsService)

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
from ..services.debug_service import debug_service, DEBUG_ENABLED
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


class GenerationService:
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
    ):
        """
        Get or load pipeline for the specified task type with optional optimization flags.
        
        Args:
            task_type: "insertion", "removal", or "white-balance"
            enable_flowmatch_scheduler: Override config setting (None = use config)
        
        Returns:
            Loaded DiffusionPipeline (single instance shared across all tasks)
        """
        # Pipeline is managed centrally, just load/switch adapter as needed
        return load_pipeline(
            task_type=task_type,
            enable_flowmatch_scheduler=enable_flowmatch_scheduler,
            on_loading_progress=on_loading_progress,
        )

    def _resolve_quality(
        self, override: str | None = None
    ) -> Tuple[str, str]:
        """Resolve quality preset to (quality_label, preset_type).
        
        Returns:
            Tuple of (quality_label, preset_type) where preset_type is "1:1" or "keep"
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
                logger.warning(
                    "⚠️ Unknown input quality override '%s'. Falling back to '%s'.",
                    override,
                    quality_label,
                )
        return quality_label, preset_type

    def _apply_input_quality(
        self,
        original: Image.Image,
        mask: Image.Image | None = None,
        reference: Image.Image | None = None,
        quality_override: str | None = None,
    ) -> tuple[Image.Image, Image.Image | None, Image.Image | None]:
        """Apply input quality preset: resize to 1:1 aspect ratio or keep original."""
        quality_label, preset_type = self._resolve_quality(quality_override)

        # If preset is "keep", keep original dimensions
        if preset_type == "keep":
            max_dim = max(original.width, original.height)
            if (
                quality_label == "original"
                and max_dim >= settings.input_quality_warning_px
            ):
                logger.warning(
                    "⚠️ Input quality 'original' selected for large image %dx%d. "
                    "This may consume significant GPU memory.",
                    original.width,
                    original.height,
                )
            return original, mask, reference

        # If preset is "1:1", resize to square aspect ratio with padding (preserve aspect ratio)
        if preset_type == "1:1":
            # Determine target size based on max dimension of original image
            max_dim = max(original.width, original.height)
            
            # Choose appropriate square size: 512, 768, 1024, 1536, 2048
            if max_dim <= 640:
                target_size = 512
            elif max_dim <= 896:
                target_size = 768
            elif max_dim <= 1280:
                target_size = 1024
            elif max_dim <= 1792:
                target_size = 1536
            else:
                target_size = 2048
            
            new_size = (target_size, target_size)
            
            logger.info(
                "🎚️ Applying input quality '%s' (1:1 aspect ratio with padding): %dx%d -> %dx%d",
                quality_label,
                original.width,
                original.height,
                target_size,
                target_size,
            )
            
            # Resize with aspect ratio preserved + padding to square
            original = resize_with_aspect_ratio_pad(original, new_size, background_color=(0, 0, 0))
            if mask is not None:
                mask = resize_with_aspect_ratio_pad(mask, new_size, background_color=(0, 0, 0))
            if reference is not None:
                reference = resize_with_aspect_ratio_pad(reference, new_size, background_color=(0, 0, 0))
            
            return original, mask, reference

        # Fallback: keep original
        return original, mask, reference

    def _prepare_images(
        self, request: GenerationRequest, task_type: str = "removal"
    ) -> tuple[Image.Image, Image.Image, list[Image.Image], Image.Image | None, Image.Image | None]:
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
            
            # Use condition_builder for white-balance (ensures correct order: [input, canny])
            canny_thresholds = (100, 200)  # Default thresholds
            conditional_images = build_white_balance_conditionals(
                original,
                canny_thresholds=canny_thresholds,
                use_cache=True,
            )
            mask = Image.new("RGB", original.size, (255, 255, 255))  # Dummy mask for compatibility
            
            logger.info(
                "🔍 [Backend Debug] White-balance conditionals: input + canny (via condition_builder)"
            )
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
            
            # Validate base64 mask
            try:
                mask = base64_to_image(mask_b64)
                logger.info(f"🎭 [Mask Debug] Decoded mask from request: size={mask.size}, mode={mask.mode}, format={getattr(mask, 'format', 'unknown')}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"Invalid base64 mask data: {str(e)}") from e

            # Apply input_quality for insert/remove
            original, mask, _ = self._apply_input_quality(
                original, mask, None, quality_override=request.input_quality
            )
            
            if mask is None:
                raise RuntimeError("Mask image missing after preprocessing")
            
            # Log mask info after preprocessing
            logger.info(f"🎭 [Mask Debug] Mask after preprocessing: size={mask.size}, mode={mask.mode}")
            
            # Verify mask is valid (not empty, has content)
            if mask.mode == "RGBA":
                # Check if mask has any non-transparent pixels
                mask_array = np.array(mask)
                has_content = np.any(mask_array[:, :, 3] > 0)  # Check alpha channel
                logger.info(f"🎭 [Mask Debug] RGBA mask has content: {has_content}")
            elif mask.mode in ["RGB", "L"]:
                # Check if mask has any non-black pixels
                mask_array = np.array(mask.convert("RGB"))
                has_content = np.any(mask_array > 0)
                logger.info(f"🎭 [Mask Debug] RGB/L mask has content: {has_content}, pixel range: [{np.min(mask_array)}, {np.max(mask_array)}]")

            # For insertion: handle two-source mask workflow if reference_mask_R is provided
            # Mask A (placement): mask from conditional_images[0] (already decoded above)
            # Mask R (object shape): reference_mask_R from request
            ref_img = None
            
            if task_type == "insertion":
                logger.info(
                    "🔍 [Insertion Debug] request.reference_image present: %s (length: %d bytes), "
                    "request.reference_mask_R present: %s",
                    bool(request.reference_image),
                    len(request.reference_image) if request.reference_image else 0,
                    bool(request.reference_mask_R)
                )
                
                # Check if using two-source mask workflow
                if request.reference_mask_R and request.reference_image:
                    logger.info("🎯 [Two-Source Mask] Using Mask A (placement) + Mask R (object shape) workflow")
                    
                    # Decode reference image and Mask R
                    reference_image = base64_to_image(request.reference_image)
                    mask_R = base64_to_image(request.reference_mask_R)
                    
                    logger.info(
                        "📥 [Two-Source Mask] Reference image: %s, Mask R: %s",
                        reference_image.size,
                        mask_R.size
                    )
                    
                    # Extract object using Mask R (preserves Mask R shape)
                    extracted_object, masked_reference_object = prepare_reference_conditionals(
                        reference_image, mask_R
                    )
                    
                    # Use extracted object as ref_img (preserves Mask R shape)
                    ref_img = extracted_object
                    
                    # Resize extracted object to match original size if needed
                    if ref_img.size != original.size:
                        logger.info(
                            "🔄 Resizing extracted object from %s to %s (preserving Mask R shape)",
                            ref_img.size,
                            original.size,
                        )
                        ref_img = ref_img.resize(
                            original.size, Image.Resampling.LANCZOS
                        )
                    
                    logger.info(
                        "✅ [Two-Source Mask] Extracted object using Mask R: %s (shape preserved)",
                        ref_img.size
                    )
                elif request.reference_image:
                    # Legacy workflow: use reference image directly (no Mask R)
                    logger.info("📥 [Legacy Insertion] Using reference image directly (no Mask R)")
                    ref_img = base64_to_image(request.reference_image)
                    if ref_img.size != original.size:
                        logger.info(
                            "🔄 Resizing ref_img from %s to %s for insertion (keeping original content)",
                            ref_img.size,
                            original.size,
                        )
                        ref_img = ref_img.resize(
                            original.size, Image.Resampling.LANCZOS
                        )
                else:
                    # No ref_img provided for insertion - skip ref_img conditional
                    logger.warning("⚠️ No ref_img provided for insertion task - ref_img conditional will be omitted")
                    logger.warning("   → _masked_object is NOT used as fallback - insertion requires ref_img from frontend")
            
            # Build conditional_images using condition_builder (ensures correct order)
            if task_type == "insertion" and ref_img is not None:
                # Insertion: [ref_img, mask, masked_bg] - use condition_builder
                conditional_images = build_insertion_conditionals(
                    original=original,
                    mask=mask,
                    ref_img=ref_img,
                    use_cache=True,
                )
                logger.info("✅ [Insertion] Conditional images order: [ref_img, mask, masked_bg] (via condition_builder)")
            elif task_type == "removal":
                # Removal: [original, mask, mae] - use condition_builder
                enable_mae = True  # MAE is required for removal
                conditional_images = build_removal_conditionals(
                    original=original,
                    mask=mask,
                    enable_mae=enable_mae,
                    use_cache=True,
                )
                logger.info("✅ [Removal] Conditional images order: [original, mask, mae] (via condition_builder)")
            else:
                # Fallback (insertion without ref_img): use prepare_mask_conditionals for backward compatibility
                logger.warning("⚠️ [Insertion] No ref_img, using fallback: [mask, masked_bg]")
                mask_cond, masked_bg, _, _ = prepare_mask_conditionals(
                    original, mask, include_mae=False
                )
                conditional_images = [mask_cond, masked_bg]
        
        # Append additional conditions from client (any base64 images)
        if extra_cond_b64s:
            extra_images: list[Image.Image] = []
            for idx, img_b64 in enumerate(extra_cond_b64s):
                try:
                    extra_img = base64_to_image(img_b64)
                    if extra_img.size != original.size:
                        logger.info(
                            "🔄 Resizing extra conditional image %d from %s to %s",
                            idx,
                            extra_img.size,
                            original.size,
                        )
                        extra_img = extra_img.resize(
                            original.size, Image.Resampling.LANCZOS
                        )
                    extra_images.append(extra_img)
                except Exception as exc:
                    logger.warning(
                        "⚠️ Failed to decode extra conditional image %d: %s", idx, exc
                    )
            if extra_images:
                logger.info(
                    "📎 Appended %d extra conditional images from request",
                    len(extra_images),
                )
                conditional_images.extend(extra_images)
        
        # Log conditional images based on task type
        if task_type == "white-balance":
            cond_desc = "input/canny"
        elif task_type == "removal":
            cond_desc = "mask/masked_bg/mae"
        else:
            cond_desc = "mask/masked_bg/ref_img"
        logger.info(
            "🔍 [Backend Debug] Conditional images count: %d (task: %s - %s)",
            len(conditional_images),
            task_type,
            cond_desc,
        )
        # Return ref_img and mask_A for post-processing if using two-source mask workflow
        # mask_A is the placement mask (from conditional_images[0])
        # Check if using two-source masks: insertion task with both reference_image and reference_mask_R
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
            mask_A_for_compositing
        )

    def _build_params(
        self,
        request: GenerationRequest,
        original: Image.Image,
        mask: Image.Image,
        conditional_images: list[Image.Image],
        task_type: str = "removal",
    ) -> Dict[str, Any]:
        # Use original image dimensions as default to preserve aspect ratio
        # Only use custom width/height if explicitly provided and different from original
        output_width = request.width if request.width and request.width != original.width else original.width
        output_height = request.height if request.height and request.height != original.height else original.height
        
        # Ensure dimensions match original aspect ratio to avoid distortion
        # If only one dimension is provided, calculate the other to maintain aspect ratio
        if request.width and not request.height:
            aspect_ratio = original.width / original.height
            output_height = int(request.width / aspect_ratio)
        elif request.height and not request.width:
            aspect_ratio = original.width / original.height
            output_width = int(request.height * aspect_ratio)
        
        # Stable Diffusion requires width and height to be divisible by 8
        # Round up to nearest multiple of 8 while preserving aspect ratio as much as possible
        def round_to_multiple_of_8(value: int) -> int:
            """Round up to nearest multiple of 8"""
            return ((value + 7) // 8) * 8
        
        # Calculate aspect ratio
        aspect_ratio = output_width / output_height
        
        # Try rounding width first, then calculate height
        rounded_width = round_to_multiple_of_8(output_width)
        height_from_width = round_to_multiple_of_8(int(rounded_width / aspect_ratio))
        
        # Try rounding height first, then calculate width
        rounded_height = round_to_multiple_of_8(output_height)
        width_from_height = round_to_multiple_of_8(int(rounded_height * aspect_ratio))
        
        # Choose the option that better preserves the original aspect ratio
        original_ratio = original.width / original.height
        width_first_ratio = rounded_width / height_from_width
        height_first_ratio = width_from_height / rounded_height
        
        width_first_diff = abs(width_first_ratio - original_ratio)
        height_first_diff = abs(height_first_ratio - original_ratio)
        
        if height_first_diff < width_first_diff:
            output_width = width_from_height
            output_height = rounded_height
        else:
            output_width = rounded_width
            output_height = height_from_width
        
        # Use prompt directly (no composition)
        final_prompt = request.prompt
        logger.info(f"📝 Using prompt: {final_prompt[:100]}...")
        
        params: Dict[str, Any] = {
            "prompt": final_prompt,
            "image": original,
            "mask_image": mask,
            "conditional_images": conditional_images,
            "num_inference_steps": request.num_inference_steps or 10,
            "guidance_scale": request.guidance_scale or 1.0,
            "width": output_width,
            "height": output_height,
        }
        
        logger.info(f"📐 Output dimensions: {output_width}x{output_height} (original: {original.width}x{original.height})")

        if request.seed is not None:
            import torch  # Lazy import to avoid requiring torch in services that don't need it
            device = get_device()
            params["generator"] = torch.Generator(device=device).manual_seed(
                request.seed
            )

        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt

        if request.true_cfg_scale is not None:
            params["true_cfg_scale"] = request.true_cfg_scale

        return params

    def _initialize_generation_metrics(self) -> Dict[str, Any]:
        """Initialize research metrics and record initial memory state."""
        research_metrics = {
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
        """Determine and validate task type from request."""
        if request.task_type:
            task_type = request.task_type
            valid_task_types = {"insertion", "removal", "white-balance"}
            if task_type not in valid_task_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid task_type: {task_type}. Must be one of: {', '.join(valid_task_types)}"
                )
        else:
            task_type = "removal"
        
        logger.info("🎯 Task type (resolved): %s", task_type)
        return task_type

    def _verify_generated_image(self, original: Image.Image, generated: Image.Image, task_type: str, conditionals_count: int = 0) -> None:
        """Verify that generated image is different from original."""
        try:
            import hashlib
            import io
            import numpy as np
            
            original_bytes = io.BytesIO()
            original.save(original_bytes, format="PNG")
            original_hash = hashlib.md5(original_bytes.getvalue()).hexdigest()
            
            generated_bytes = io.BytesIO()
            generated.save(generated_bytes, format="PNG")
            generated_hash = hashlib.md5(generated_bytes.getvalue()).hexdigest()
            
            original_array = np.array(original.convert("RGB"))
            generated_array = np.array(generated.convert("RGB"))
            
            if original_array.shape != generated_array.shape:
                logger.warning(f"⚠️ [WARNING] Generated image shape {generated_array.shape} != original shape {original_array.shape}")
            else:
                pixel_diff = np.abs(original_array.astype(np.int16) - generated_array.astype(np.int16))
                max_diff = np.max(pixel_diff)
                mean_diff = np.mean(pixel_diff)
                diff_percentage = (np.count_nonzero(pixel_diff) / pixel_diff.size) * 100
                
                logger.info("🔍 Generated image comparison:")
                logger.info(f"   Max pixel difference: {max_diff}")
                logger.info(f"   Mean pixel difference: {mean_diff:.2f}")
                logger.info(f"   Different pixels: {diff_percentage:.2f}%")
                
                if original_hash == generated_hash:
                    logger.warning("⚠️ [WARNING] Generated image is IDENTICAL to original image (same hash)!")
                    logger.warning("   This indicates the model returned the original instead of generating.")
                    logger.warning(f"   Original hash: {original_hash}")
                    logger.warning(f"   Generated hash: {generated_hash}")
                    logger.warning(f"   Task type: {task_type}")
                    logger.warning(f"   Conditional images count: {conditionals_count}")
                elif max_diff == 0 and mean_diff == 0:
                    logger.warning("⚠️ [WARNING] Generated image pixels are IDENTICAL to original (0 difference)!")
                    logger.warning("   This may indicate the model returned the original instead of generating.")
                elif diff_percentage < 1.0:
                    logger.warning(f"⚠️ [WARNING] Generated image is very similar to original ({diff_percentage:.2f}% different pixels)")
                    logger.warning("   This may indicate the model did not generate properly.")
                else:
                    logger.info(f"✅ Generated image is different from original (hash: {generated_hash[:8]}... vs {original_hash[:8]}...)")
                    logger.info(f"   {diff_percentage:.2f}% of pixels are different (max diff: {max_diff})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to verify generated image difference: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _execute_reference_guided_insertion(
        self,
        request: GenerationRequest,
        pipeline_callable: Any,
        debug_session: Any,
        research_metrics: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        task_id: Optional[str] = None
    ) -> Tuple[Image.Image, Dict[str, Any], Optional[Image.Image], list, Image.Image]:
        """Execute reference-guided insertion workflow."""
        logger.info("🎯 [Reference-Guided Insertion] Using new pipeline: reference_mask_R positioned inside main_mask_A")
        
        # Load inputs
        original = base64_to_image(request.input_image)
        if not request.conditional_images or len(request.conditional_images) == 0:
            raise HTTPException(
                status_code=400,
                detail="conditional_images[0] (main_mask_A) is required for reference-guided insertion"
            )
        if not request.reference_image:
            raise HTTPException(
                status_code=400,
                detail="reference_image is required for reference-guided insertion"
            )
        if not request.reference_mask_R:
            raise HTTPException(
                status_code=400,
                detail="reference_mask_R is required for reference-guided insertion"
            )
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
        
        # Prepare generator
        generator = None
        if request.seed is not None:
            import torch
            device = get_device()
            generator = torch.Generator(device=device).manual_seed(request.seed)
        
        # Save debug images
        debug_session.save_image(original, "input_image", "01", "Original base image")
        debug_session.save_image(main_mask_A, "main_mask_A", "02", "Main mask A (placement region)")
        debug_session.save_image(reference_image, "reference_image", "03", "Reference image (appearance source)")
        debug_session.save_image(reference_mask_R, "reference_mask_R", "04", "Reference mask R (object shape)")
        
        # Log pipeline info
        debug_session.log_lora("=" * 80)
        debug_session.log_lora("REFERENCE-GUIDED INSERTION PIPELINE")
        debug_session.log_lora("=" * 80)
        debug_session.log_lora(f"Base image size: {original.size}")
        debug_session.log_lora(f"Main mask A size: {main_mask_A.size} (placement region)")
        debug_session.log_lora(f"Reference image size: {reference_image.size} (appearance)")
        debug_session.log_lora(f"Reference mask R size: {reference_mask_R.size} (object shape)")
        debug_session.log_lora(f"Prompt: {request.prompt or ''}")
        debug_session.log_lora(f"Num inference steps: {request.num_inference_steps or 10}")
        debug_session.log_lora(f"CFG scale: {request.true_cfg_scale or request.guidance_scale or 4.0}")
        debug_session.log_lora(f"Seed: {request.seed}")
        debug_session.log_lora(f"Negative prompt: {request.negative_prompt or 'None'}")
        
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
        debug_session.log_lora(f"Inference completed in {inference_time:.3f}s")
        
        # Verify generated image
        self._verify_generated_image(original, generated, "insertion (reference-guided)", 0)
        
        # Save output stats
        research_metrics["image_stats"]["output"] = {
            "size": list(generated.size),
            "mode": generated.mode,
            "width": generated.width,
            "height": generated.height,
            "aspect_ratio": round(generated.width / generated.height, 3),
        }
        debug_session.save_image(generated, "generated_output", "09", "Final composited output (reference-guided insertion)")
        
        # Build params for metadata
        params = {
            "prompt": request.prompt or "",
            "num_inference_steps": request.num_inference_steps or 10,
            "guidance_scale": request.true_cfg_scale or request.guidance_scale or 4.0,
            "true_cfg_scale": request.true_cfg_scale or request.guidance_scale or 4.0,
            "width": original.width,
            "height": original.height,
            "pipeline_type": "reference_guided_insertion",
            "two_source_masks": True,
        }
        
        debug_session.log_lora("=" * 80)
        debug_session.log_lora("PIPELINE COMPLETED SUCCESSFULLY")
        debug_session.log_lora("=" * 80)
        
        # Create masked R image (extracted object from reference) for debug info
        from ..services.mask_segmentation_service import extract_object_with_mask
        masked_R_image = extract_object_with_mask(reference_image, reference_mask_R)
        debug_session.save_image(masked_R_image, "masked_R", "07", "Masked R (extracted object from reference)")
        
        # Return conditionals for debug info: [original, mask_A, masked_R, reference_image, positioned_mask_R]
        # This matches what frontend expects for insertion debug display
        conditionals_for_debug = [
            original,      # 1. Original input image
            main_mask_A,  # 2. Mask A (placement mask on original)
            masked_R_image,  # 3. Masked R (extracted object from reference)
            reference_image,  # 4. Reference image (original)
            positioned_mask_R,  # 5. Positioned mask R (reference mask R pasted into main mask A)
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
        """Execute standard workflow (removal/insertion/white-balance)."""
        # Prepare images
        prep_start = time.time()
        original, mask, conditionals, reference, mask_A_for_compositing = self._prepare_images(request, task_type)
        prep_time = time.time() - prep_start
        research_metrics["timing"]["preprocessing_seconds"] = round(prep_time, 3)
        
        # Calculate mask statistics
        if mask is not None:
            try:
                import numpy as np
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
                debug_session.save_image(conditionals[1], "masked_bg", "03", "Masked background (original with mask area removed)")
            if len(conditionals) > 2:
                if task_type == "removal":
                    debug_session.save_image(conditionals[2], "mae_output", "05", "MAE inpainted preview")
                else:
                    debug_session.save_image(conditionals[2], "ref_img", "05", "Reference image (from frontend upload)")
        
        if reference is not None:
            debug_session.save_image(reference, "reference_image", "04", "Reference image (optional)")
        
        debug_session.log_lora(f"Input image size: {original.size}")
        debug_session.log_lora(f"Number of conditional images: {len(conditionals)}")
        debug_session.log_lora(f"Has reference image: {reference is not None}")
        
        # Record inference start time
        inference_start = time.time()
        
        import torch
        with torch.no_grad():
            logger.info("🎯 Using Qwen pipeline for task: %s", task_type)
            
            # Check if using QwenImageEditPlusPipeline
            is_qwen_pipeline = False
            try:
                from diffusers import QwenImageEditPlusPipeline # type: ignore
                is_qwen_pipeline = isinstance(pipeline, QwenImageEditPlusPipeline)
            except (ImportError, AttributeError):
                try:
                    pipeline_class_name = pipeline.__class__.__name__
                    is_qwen_pipeline = "QwenImageEditPlus" in pipeline_class_name
                except Exception:
                    is_qwen_pipeline = False
            
            if is_qwen_pipeline:
                logger.info("🎯 Using QwenImageEditPlusPipeline interface")
                image_list = conditionals if conditionals else []
                
                # Log conditional images order
                if task_type == "insertion":
                    expected_order = ["ref_img", "mask", "masked_bg"]
                    logger.info(f"🔍 [Insertion] Conditional images order (expected): {expected_order}")
                    logger.info(f"🔍 [Insertion] Conditional images count: {len(image_list)}")
                    if len(image_list) >= 3:
                        logger.info(f"   [0] ref_img: {image_list[0].size} (mode: {image_list[0].mode})")
                        logger.info(f"   [1] mask: {image_list[1].size} (mode: {image_list[1].mode})")
                        logger.info(f"   [2] masked_bg: {image_list[2].size} (mode: {image_list[2].mode})")
                elif task_type == "removal":
                    expected_order = ["original", "mask", "mae"]
                    logger.info(f"🔍 [Removal] Conditional images order (expected): {expected_order}")
                    logger.info(f"🔍 [Removal] Conditional images count: {len(image_list)}")
                    if len(image_list) >= 3:
                        logger.info(f"   [0] original: {image_list[0].size} (mode: {image_list[0].mode})")
                        logger.info(f"   [1] mask: {image_list[1].size} (mode: {image_list[1].mode})")
                        logger.info(f"   [2] mae: {image_list[2].size} (mode: {image_list[2].mode})")
                else:
                    logger.info(f"🔍 [White-balance] Conditional images count: {len(image_list)}")
                
                # Build params for QwenImageEditPlusPipeline
                # Note: guidance_scale is ineffective for Qwen. Use true_cfg_scale instead.
                # negative_prompt is required (even empty string) to enable classifier-free guidance
                true_cfg = request.true_cfg_scale or request.guidance_scale or (3.3 if task_type == "white-balance" else 4.0)
                qwen_params = {
                    "image": image_list,
                    "prompt": request.prompt or "",
                    "num_inference_steps": request.num_inference_steps or 10,
                    "true_cfg_scale": true_cfg,
                    "height": original.height,
                    "width": original.width,
                    # Always provide negative_prompt (even if empty) to enable classifier-free guidance
                    "negative_prompt": request.negative_prompt if request.negative_prompt else " ",
                }
                
                if request.seed is not None:
                    device = get_device()
                    qwen_params["generator"] = torch.Generator(device=device).manual_seed(request.seed)
                
                # Add progress callback if provided
                if progress_callback is not None:
                    num_steps = qwen_params["num_inference_steps"]
                    # Use task_id for cancellation check
                    task_id_for_cancel = task_id or getattr(request, 'request_id', None) or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    def callback_on_step_end(pipeline_instance, step: int, timestep: int, callback_kwargs: dict):
                        """Callback function for pipeline progress tracking with cancellation check."""
                        try:
                            # Check cancellation flag
                            if self._check_cancellation(task_id_for_cancel):
                                logger.warning(f"⚠️ Generation cancelled at step {step}/{num_steps}")
                                raise CancelledError(task_id_for_cancel, f"Generation cancelled at step {step}")
                            
                            progress_callback(step, timestep, num_steps)
                        except CancelledError:
                            raise  # Re-raise cancellation error
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                        return callback_kwargs
                    
                    qwen_params["callback_on_step_end"] = callback_on_step_end
                
                logger.info(f"📥 Calling QwenImageEditPlusPipeline with {len(image_list)} conditional images for task={task_type}")
                result = pipeline_callable(**qwen_params)
                
                if hasattr(result, "images"):
                    generated = result.images[0] if isinstance(result.images, list) else result.images
                else:
                    generated = result[0] if isinstance(result, (list, tuple)) else result
                
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
                generated = (
                    result.images[0]
                    if hasattr(result, "images")
                    else result[0]
                    if isinstance(result, list)
                    else result
                )
                self._verify_generated_image(original, generated, task_type, len(conditionals))
            
            # Ensure generated image has same size as original
            if generated.size != original.size:
                logger.info(f"🔄 Resizing generated image from {generated.size} to {original.size} to match input")
                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            
            debug_session.save_image(generated, "generated_output", "07", "Final generated output")
        
        # Record inference end time
        inference_time = time.time() - inference_start
        research_metrics["timing"]["inference_seconds"] = round(inference_time, 3)
        debug_session.log_lora(f"Inference completed in {inference_time:.3f}s")
        
        return generated, params, reference, conditionals

    def generate(self, request: GenerationRequest, progress_callback: Optional[Callable[[int, int, int], None]] = None, on_pipeline_loaded: Optional[Callable[[], None]] = None, on_loading_progress: Optional[Callable[[str, float]]] = None) -> Dict[str, Any]:
        """
        Generate image using Qwen pipeline.
        
        Uses sync def because:
        - Runs on Heavy Worker (A100) where PyTorch/GPU tasks are blocking
        - Modal automatically wraps sync functions in thread pool
        - No benefit from async for CPU/GPU bound tasks (Python GIL)
        
        Args:
            request: Generation request
            progress_callback: Optional callback function(step: int, timestep: int, total_steps: int) to track progress
        """
        start = time.time()
        
        # Generate task_id for cancellation tracking (use request_id if available, otherwise generate new)
        task_id = getattr(request, 'request_id', None) or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Create debug session
        debug_session = debug_service.create_session()
        debug_session.log_lora("=" * 80)
        debug_session.log_lora("GENERATION REQUEST STARTED")
        debug_session.log_lora("=" * 80)
        
        # Initialize research metrics
        research_metrics = self._initialize_generation_metrics()

        try:
            # Check cancellation before starting
            if self._check_cancellation(task_id):
                raise CancelledError(task_id, "Task was cancelled before generation started")
            # Determine task type
            task_type = self._determine_task_type(request)
            debug_session.log_lora(f"Task type: {task_type}")
            
            # Extract optimization flags from request (will override config if provided)
            optimization_flags = {
                "enable_flowmatch_scheduler": request.enable_flowmatch_scheduler,
            }
            
            # Log optimization flags from request
            if any(flag is not None for flag in optimization_flags.values()):
                logger.info(
                    "ℹ️ Optimization flags provided in request (will override config): "
                    f"flowmatch={request.enable_flowmatch_scheduler}"
                )
            
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
            debug_session.log_lora(f"Pipeline loaded in {pipeline_load_time:.3f}s")
            logger.info(f"✅ Pipeline loaded in {pipeline_load_time:.3f}s, notifying callback...")
            
            # Notify that pipeline is loaded (this should update status from loading_pipeline to processing)
            if on_pipeline_loaded:
                logger.info("📢 Calling on_pipeline_loaded callback...")
                on_pipeline_loaded()
                logger.info("✅ on_pipeline_loaded callback completed")
            else:
                logger.debug("ℹ️ on_pipeline_loaded callback not provided (optional)")
            
            # Log LoRA info from qwen_loader
            from ..core.qwen_loader import _current_adapter, _loaded_adapters
            debug_session.log_lora(f"Loaded adapters: {list(_loaded_adapters)}")
            debug_session.log_lora(f"Current adapter: {_current_adapter}")
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
                generated, params, reference, conditionals, positioned_mask_R = self._execute_reference_guided_insertion(
                    request, pipeline_callable, debug_session, research_metrics, progress_callback, task_id
                )
                
            else:
                # Execute standard workflow
                original = base64_to_image(request.input_image)
                generated, params, reference, conditionals = self._execute_standard_workflow(
                    request, task_type, pipeline, pipeline_callable, debug_session, research_metrics, progress_callback, task_id
                )

            generation_time = time.time() - start
            logger.info("✅ Image generated in %.2fs", generation_time)
            debug_session.log_lora(f"Generation completed in {generation_time:.2f}s")
            
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
                            "allocated_delta_mb": round(after["allocated_mb"] - before["allocated_mb"], 2),
                            "reserved_delta_mb": round(after["reserved_mb"] - before["reserved_mb"], 2),
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
                generation_time - research_metrics["timing"].get("pipeline_load_seconds", 0) - 
                research_metrics["timing"].get("preprocessing_seconds", 0) - 
                research_metrics["timing"].get("inference_seconds", 0), 3
            )
            
            # Save research metrics to debug session
            debug_session.save_research_metrics(research_metrics)
            debug_session.log_lora(f"Research metrics saved: {len(research_metrics)} categories")

            # Save visualization
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Clear cancellation flag on successful completion
            self.clear_cancellation(task_id)
            
            # Build metadata
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

            # Use conditional images for visualization (RGB format)
            conditional_images_for_viz = list(conditionals)
            # If reference image was added, include it in visualization
            if reference and len(conditional_images_for_viz) == 3:
                conditional_images_for_viz = conditional_images_for_viz + [reference]
            
            # Get the actual prompt used (final_prompt from params if available, otherwise request.prompt)
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
            
            # Finalize debug session
            debug_session.log_lora("=" * 80)
            debug_session.log_lora("GENERATION REQUEST COMPLETED SUCCESSFULLY")
            debug_session.log_lora("=" * 80)
            debug_session.finalize(success=True)
            
            # Build debug_info with conditional images
            # Always create debug_info to ensure frontend gets updated info for each generation
            # This ensures debug panel shows correct info even when switching between tasks
            debug_info = None
            try:
                # Convert conditional images to base64 for frontend debug
                conditional_images_b64 = []
                # Labels depend on task_type:
                # - White-balance: input, canny
                # - Removal: original, mask, mae
                # - Insertion (reference-guided): original, mask_A, masked_R, reference_image
                # - Insertion (standard): ref_img, mask, masked_bg
                if task_type == "white-balance":
                    conditional_labels = ["input", "canny"]
                    max_conditionals = 2
                elif task_type == "removal":
                    conditional_labels = ["original", "mask", "mae"]
                    max_conditionals = 3
                elif use_reference_guided_insertion:
                    # Reference-guided insertion: [original, mask_A, masked_R, reference_image, positioned_mask_R]
                    conditional_labels = ["original", "mask_A", "masked_R", "reference_image", "positioned_mask_R"]
                    max_conditionals = 5
                else:
                    # Standard insertion: [ref_img, mask, masked_bg]
                    conditional_labels = ["ref_img", "mask", "masked_bg"]
                    max_conditionals = 3
                
                for i, cond_img in enumerate(conditionals[:max_conditionals]):
                    try:
                        cond_b64 = image_to_base64(cond_img)
                        conditional_images_b64.append(cond_b64)
                    except Exception as e:
                        logger.warning(f"Failed to encode conditional image {i}: {e}")
                        # Continue with other images even if one fails
                
                # Get LoRA info
                from ..core.qwen_loader import _current_adapter, _loaded_adapters
                
                # Always create debug_info, even if some images failed to encode
                # This ensures frontend gets updated info for each generation
                debug_info = {
                    "conditional_images": conditional_images_b64,
                    "conditional_labels": conditional_labels[:len(conditional_images_b64)],
                    "input_image_size": f"{original.width}x{original.height}",
                    "output_image_size": f"{generated.width}x{generated.height}",
                    "lora_adapter": _current_adapter,
                    "loaded_adapters": list(_loaded_adapters) if _loaded_adapters else [],
                    # Additional debug images for download/display
                    "original_image": request.input_image,
                    "mask_A": request.conditional_images[0] if request.conditional_images else None,
                    "reference_image": request.reference_image,
                    "reference_mask_R": request.reference_mask_R,
                }
                
                # Add positioned_mask_R for reference-guided insertion
                if use_reference_guided_insertion and positioned_mask_R is not None:
                    try:
                        # Ensure mask R is binary (L mode) for debug visualization
                        # Convert to L mode if it's RGB (from prepare_condition_images)
                        mask_R_for_debug = positioned_mask_R
                        if mask_R_for_debug.mode != "L":
                            # Convert RGB back to L (grayscale) for proper binary mask display
                            mask_R_for_debug = mask_R_for_debug.convert("L")
                        positioned_mask_R_b64 = image_to_base64(mask_R_for_debug)
                        debug_info["positioned_mask_R"] = positioned_mask_R_b64
                    except Exception as e:
                        logger.warning(f"Failed to encode positioned_mask_R: {e}")
                
                # Add session_name and debug_path to debug_info for download functionality
                if DEBUG_ENABLED:
                    session_path = debug_session.get_session_path()
                    if session_path:
                        # Extract session name from path (e.g., "debug_output/20241206_123456_abc12345" -> "20241206_123456_abc12345")
                        session_name = session_path.name if hasattr(session_path, 'name') else str(session_path).split('/')[-1]
                        debug_info["session_name"] = session_name
                        debug_info["debug_path"] = str(session_path)
                    # Also try to get session_name directly from debug_session if available
                    elif hasattr(debug_session, 'session_name') and debug_session.session_name:
                        debug_info["session_name"] = debug_session.session_name
                        if hasattr(debug_session, 'session_dir') and debug_session.session_dir:
                            debug_info["debug_path"] = str(debug_session.session_dir)
            except Exception as e:
                # Even if building debug_info fails, try to create minimal info
                logger.warning(f"Failed to build debug_info: {e}")
                try:
                    from ..core.qwen_loader import _current_adapter, _loaded_adapters
                    debug_info = {
                        "conditional_images": [],
                        "conditional_labels": [],
                        "input_image_size": f"{original.width}x{original.height}",
                        "output_image_size": f"{generated.width}x{generated.height}",
                        "lora_adapter": _current_adapter,
                        "loaded_adapters": list(_loaded_adapters) if _loaded_adapters else [],
                        "original_image": request.input_image,
                        "mask_A": request.conditional_images[0] if request.conditional_images else None,
                        "reference_image": request.reference_image,
                        "reference_mask_R": request.reference_mask_R,
                    }
                    # Add session_name and debug_path if available
                    if DEBUG_ENABLED:
                        session_path = debug_session.get_session_path()
                        if session_path:
                            session_name = session_path.name if hasattr(session_path, 'name') else str(session_path).split('/')[-1]
                            debug_info["session_name"] = session_name
                            debug_info["debug_path"] = str(session_path)
                        elif hasattr(debug_session, 'session_name') and debug_session.session_name:
                            debug_info["session_name"] = debug_session.session_name
                            if hasattr(debug_session, 'session_dir') and debug_session.session_dir:
                                debug_info["debug_path"] = str(debug_session.session_dir)
                except Exception:
                    # Last resort: create empty debug_info
                    debug_info = {
                        "conditional_images": [],
                        "conditional_labels": [],
                        "input_image_size": f"{original.width}x{original.height}",
                        "output_image_size": f"{generated.width}x{generated.height}",
                        "original_image": request.input_image,
                        "mask_A": request.conditional_images[0] if request.conditional_images else None,
                        "reference_image": request.reference_image,
                        "reference_mask_R": request.reference_mask_R,
                    }
                    # Try to add session info even in fallback
                    try:
                        if DEBUG_ENABLED:
                            session_path = debug_session.get_session_path()
                            if session_path:
                                session_name = session_path.name if hasattr(session_path, 'name') else str(session_path).split('/')[-1]
                                debug_info["session_name"] = session_name
                                debug_info["debug_path"] = str(session_path)
                            elif hasattr(debug_session, 'session_name') and debug_session.session_name:
                                debug_info["session_name"] = debug_session.session_name
                                if hasattr(debug_session, 'session_dir') and debug_session.session_dir:
                                    debug_info["debug_path"] = str(debug_session.session_dir)
                    except Exception:
                        pass  # Ignore if can't add session info
            
            # Add debug session path to response
            response = {
                "success": True,
                "image": image_to_base64(generated),
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

