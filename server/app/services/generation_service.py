from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Tuple, cast

import torch
from PIL import Image

from ..core.config import settings
from ..core.pipeline import get_device, load_pipeline
from ..services.image_processing import (
    base64_to_image,
    image_to_base64,
    prepare_mask_conditionals,
    resize_with_aspect_ratio_pad,
)
from ..services.prompt_composer import compose_prompt
from ..services.debug_service import debug_service, DEBUG_ENABLED
from ..models.schemas import GenerationRequest

logger = logging.getLogger(__name__)


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
    def __init__(self) -> None:
        # Pipeline is now managed centrally by qwen_loader, no need to cache here
        pass

    async def _ensure_pipeline(self, task_type: str = "insertion"):
        """
        Get or load pipeline for the specified task type.
        
        Args:
            task_type: "insertion", "removal", or "white-balance"
        
        Returns:
            Loaded DiffusionPipeline (single instance shared across all tasks)
        """
        # Pipeline is managed centrally, just load/switch adapter as needed
        return await load_pipeline(task_type=task_type)

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
    ) -> tuple[Image.Image, Image.Image, list[Image.Image], Image.Image | None]:
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")

        original = base64_to_image(request.input_image)
        
        # New Qwen flow:
        
        # - conditional_images[0]: mask
        # - conditional_images[1:]: additional conditions (background masked, canny, mae, ...)
        cond_list = list(request.conditional_images or [])
        extra_cond_b64s: list[str] = []

        if task_type == "white-balance":
            # White-balance: mask may not be sent, in which case use full white mask.
            original, _, _ = self._apply_input_quality(
                original, quality_override=request.input_quality
            )
            if cond_list:
                mask_b64 = cond_list[0]
                extra_cond_b64s = cond_list[1:]
                mask = base64_to_image(mask_b64)
            else:
                mask = Image.new("RGB", original.size, (255, 255, 255))
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

            # Apply input_quality for insert/remove
            original, mask, _ = self._apply_input_quality(
                original, mask, None, quality_override=request.input_quality
        )
        
        if mask is None:
            raise RuntimeError("Mask image missing after preprocessing")

        # From mask build basic conditions: mask/background/object/mae
        mask_cond, background_rgb, obj_rgb, mae_image = prepare_mask_conditionals(
            original, mask
        )
        conditional_images: list[Image.Image] = [
            mask_cond,
            background_rgb,
            obj_rgb,
            mae_image,
        ]
        
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
        
        logger.info(
            "🔍 [Backend Debug] Conditional images count: %d (>=4: base conditionals + optional extras)",
            len(conditional_images),
        )
        # reference is no longer used in the new Qwen flow
        return original, mask, conditional_images, None

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
        
        # Apply prompt composition for insertion/removal; white-balance uses raw/simple prompt
        final_prompt = request.prompt
        if task_type in ("insertion", "removal") and (request.angle or request.background_preset):
            # Use prompt composition when angle or background_preset is provided
            final_prompt = compose_prompt(
                base_prompt=request.prompt,
                angle=request.angle,
                background_preset=request.background_preset,
                style_notes=None,  # Can be extended later if needed
            )
            logger.info(f"📝 Composed prompt: {final_prompt[:100]}...")
        
        params: Dict[str, Any] = {
            "prompt": final_prompt,
            "image": original,
            "mask_image": mask,
            "conditional_images": conditional_images,
            "num_inference_steps": request.num_inference_steps or 20,
            "guidance_scale": request.guidance_scale or 1.0,
            "width": output_width,
            "height": output_height,
        }
        
        logger.info(f"📐 Output dimensions: {output_width}x{output_height} (original: {original.width}x{original.height})")

        if request.seed is not None:
            device = get_device()
            params["generator"] = torch.Generator(device=device).manual_seed(
                request.seed
            )

        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt

        if request.true_cfg_scale is not None:
            params["true_cfg_scale"] = request.true_cfg_scale

        return params

    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start = time.time()
        
        # Create debug session
        debug_session = debug_service.create_session()
        debug_session.log_lora("=" * 80)
        debug_session.log_lora("GENERATION REQUEST STARTED")
        debug_session.log_lora("=" * 80)

        try:
            # Determine task type: use request.task_type if provided, otherwise auto-detect
            if request.task_type:
                task_type = request.task_type
                # Validate task_type
                valid_task_types = {"insertion", "removal", "white-balance"}
                if task_type not in valid_task_types:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid task_type: {task_type}. Must be one of: {', '.join(valid_task_types)}"
                    )
            else:
                # No task_type: default to removal
                task_type = "removal"
            logger.info("🎯 Task type (resolved): %s", task_type)
            debug_session.log_lora(f"Task type: {task_type}")
            
            # Log low-end optimization flags from request (for future reference)
            # Note: Currently pipeline uses config settings, not request overrides
            # To change optimizations, update config and restart or clear cache
            if any([
                request.enable_4bit_text_encoder is not None,
                request.enable_cpu_offload is not None,
                request.enable_memory_optimizations is not None,
                request.enable_flowmatch_scheduler is not None,
            ]):
                logger.info(
                    "ℹ️ Low-end optimization flags provided in request: "
                    f"4bit={request.enable_4bit_text_encoder}, "
                    f"cpu_offload={request.enable_cpu_offload}, "
                    f"mem_opt={request.enable_memory_optimizations}, "
                    f"flowmatch={request.enable_flowmatch_scheduler}. "
                    "Note: Pipeline uses config settings. To change, update config and restart."
                )
            
            pipeline = await self._ensure_pipeline(task_type=task_type)
            pipeline_callable = cast(Any, pipeline)
            
            # Log LoRA info from qwen_loader
            from ..core.qwen_loader import _current_adapter, _loaded_adapters
            debug_session.log_lora(f"Loaded adapters: {list(_loaded_adapters)}")
            debug_session.log_lora(f"Current adapter: {_current_adapter}")
            debug_session.save_lora_info({
                "task_type": task_type,
                "loaded_adapters": list(_loaded_adapters),
                "current_adapter": _current_adapter,
            })
            original, mask, conditionals, reference = self._prepare_images(request, task_type)
            
            # Save debug images
            debug_session.save_image(original, "input_image", "01", "Original input image")
            debug_session.save_image(mask, "mask_image", "02", "Mask image (first conditional)")
            
            # Save conditional images
            if len(conditionals) > 1:
                debug_session.save_image(conditionals[1], "mask_background", "03", "Mask background (original with mask applied)")
            if len(conditionals) > 2:
                debug_session.save_image(conditionals[2], "canny_edge", "05", "Canny edge detection")
            if len(conditionals) > 3:
                debug_session.save_image(conditionals[3], "mae_output", "06", "MAE feature map")
            
            if reference is not None:
                debug_session.save_image(reference, "reference_image", "04", "Reference image (optional)")
            
            debug_session.log_lora(f"Input image size: {original.size}")
            debug_session.log_lora(f"Number of conditional images: {len(conditionals)}")
            debug_session.log_lora(f"Has reference image: {reference is not None}")
            
            with torch.no_grad():
                # All tasks (insert/remove/white-balance) use Qwen; differences only in params/conditionals.
                # Check if current pipeline is QwenImageEditPlusPipeline.
                logger.info("🎯 Using Qwen pipeline for task: %s", task_type)
                
                # Check if using QwenImageEditPlusPipeline for all task types
                is_qwen_pipeline = False
                try:
                    from diffusers import QwenImageEditPlusPipeline # type: ignore
                    is_qwen_pipeline = isinstance(pipeline, QwenImageEditPlusPipeline)
                except (ImportError, AttributeError):
                    # QwenImageEditPlusPipeline not available, use standard pipeline
                    # Also check by class name as fallback
                    try:
                        pipeline_class_name = pipeline.__class__.__name__
                        is_qwen_pipeline = "QwenImageEditPlus" in pipeline_class_name
                    except Exception:
                        is_qwen_pipeline = False
                
                if is_qwen_pipeline:
                    # Use QwenImageEditPlusPipeline interface
                    logger.info("🎯 Using QwenImageEditPlusPipeline interface")
                    
                    # QwenImageEditPlusPipeline expects:
                    # - image: list of PIL Images (conditional images)
                    # - prompt: str
                    # - negative_prompt: optional str
                    # - true_cfg_scale: float (guidance scale)
                    # - num_inference_steps: int
                    # - height, width: int
                    # - prompt_embeds, prompt_embeds_mask: optional (will be encoded if not provided)
                    
                    # Prepare image list (conditional images)
                    image_list = conditionals if conditionals else []
                    
                    # Build params for QwenImageEditPlusPipeline
                    qwen_params = {
                        "image": image_list,
                        "prompt": request.prompt or "",
                        "num_inference_steps": request.num_inference_steps or 20,
                        "true_cfg_scale": request.true_cfg_scale or request.guidance_scale or (3.3 if task_type == "white-balance" else 4.0),
                        "height": original.height,
                        "width": original.width,
                    }
                    
                    if request.negative_prompt:
                        qwen_params["negative_prompt"] = request.negative_prompt
                    
                    if request.seed is not None:
                        device = get_device()
                        qwen_params["generator"] = torch.Generator(device=device).manual_seed(request.seed)
                    
                    logger.info(f"📥 Calling QwenImageEditPlusPipeline with {len(image_list)} conditional images for task={task_type}")
                    # Run blocking inference in thread pool
                    result = await asyncio.to_thread(pipeline_callable, **qwen_params)
                    
                    # QwenImageEditPlusPipeline returns QwenImagePipelineOutput with images attribute
                    if hasattr(result, "images"):
                        generated = result.images[0] if isinstance(result.images, list) else result.images
                    else:
                        generated = result[0] if isinstance(result, (list, tuple)) else result
                    
                    # Convert qwen_params to params format for metadata
                    params = {
                        "num_inference_steps": qwen_params["num_inference_steps"],
                        "guidance_scale": qwen_params.get("true_cfg_scale", request.guidance_scale or 4.0),
                        "true_cfg_scale": qwen_params.get("true_cfg_scale"),
                        "width": qwen_params["width"],
                        "height": qwen_params["height"],
                    }
                else:
                    # For standard StableDiffusionInpaintPipeline, use existing params
                    params = self._build_params(request, original, mask, conditionals, task_type=task_type)
                    # Run blocking inference in thread pool
                    result = await asyncio.to_thread(pipeline_callable, **params)
                    generated = (
                        result.images[0]
                        if hasattr(result, "images")
                        else result[0]
                        if isinstance(result, list)
                        else result
                    )
                
                # Ensure generated image has same size as original
                # Model might output different size due to internal constraints (e.g., multiples of 64)
                if generated.size != original.size:
                    logger.info(f"🔄 Resizing generated image from {generated.size} to {original.size} to match input")
                    generated = generated.resize(original.size, Image.Resampling.LANCZOS)
                
                # Save generated image for debug
                debug_session.save_image(generated, "generated_output", "07", "Final generated output")

            generation_time = time.time() - start
            logger.info("✅ Image generated in %.2fs", generation_time)
            debug_session.log_lora(f"Generation completed in {generation_time:.2f}s")

            # Save visualization
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
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
            
            # Save generation parameters to debug session
            debug_session.save_parameters({
                "task_type": task_type,
                "prompt": request.prompt,
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
            
            # Add debug session path to response
            response = {
                "success": True,
                "image": image_to_base64(generated),
                "generation_time": round(generation_time, 2),
                "model_used": f"qwen_local_{task_type}",
                "parameters_used": metadata,
                "request_id": request_id,  # Include request_id for visualization access
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

