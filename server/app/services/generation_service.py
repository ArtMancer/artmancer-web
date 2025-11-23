from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict

import torch
from fastapi import HTTPException
from PIL import Image

from ..core.pipeline import get_device, load_pipeline
from ..services.image_processing import (
    base64_to_image,
    image_to_base64,
    prepare_mask_conditionals,
)
from ..services.prompt_composer import compose_prompt
from ..services.visualization import create_visualization_service
from ..models.schemas import GenerationRequest

logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(self) -> None:
        self._pipeline_insertion = None
        self._pipeline_removal = None
        # Initialize visualization service
        from ..core.config import settings
        self._visualization = create_visualization_service(
            output_dir=settings.visualization_dir if settings.visualization_dir else None,
            enabled=settings.enable_visualization,
        )

    def _ensure_pipeline(self, task_type: str = "insertion"):
        """
        Get or load pipeline for the specified task type.
        
        Args:
            task_type: "insertion", "removal", or "white-balance"
        
        Returns:
            Loaded DiffusionPipeline
        """
        if task_type == "removal":
            if self._pipeline_removal is None:
                self._pipeline_removal = load_pipeline(task_type="removal")
            return self._pipeline_removal
        elif task_type == "white-balance":
            # White balance uses a different pipeline (Pix2Pix from HuggingFace)
            return load_pipeline(task_type="white-balance")
        else:  # insertion (default)
            if self._pipeline_insertion is None:
                self._pipeline_insertion = load_pipeline(task_type="insertion")
            return self._pipeline_insertion

    def _prepare_images(
        self, request: GenerationRequest, task_type: str = "removal"
    ) -> tuple[Image.Image, Image.Image, list[Image.Image], Image.Image | None]:
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")
        # For white-balance, mask is not required
        if task_type != "white-balance" and not request.mask_image:
            raise HTTPException(status_code=400, detail="mask_image is required")

        original = base64_to_image(request.input_image)
        
        # For white-balance, mask is not needed
        if task_type == "white-balance":
            # Return dummy mask and empty conditionals for white-balance
            mask = Image.new("RGB", original.size, (0, 0, 0))
            return original, mask, [], None
        
        # For other tasks, mask is required
        mask = base64_to_image(request.mask_image)
        
        # Resize mask to match original if needed (handled in prepare_mask_conditionals)
        # Returns: mask_rgb, background_rgb, object_rgb, mae_image (all RGB, no alpha channel)
        mask_cond, background_rgb, obj_rgb, mae_image = prepare_mask_conditionals(original, mask)
        
        # All conditional images are already RGB format (black/white, no transparency)
        # For QwenImageEditPlus: [mask, background, object, mae] for removal
        conditional_images = [mask_cond, background_rgb, obj_rgb, mae_image]
        
        # Add reference image if provided (for object insertion)
        reference = None
        logger.info("🔍 [Backend Debug] Checking for reference image:")
        logger.info("   - request.reference_image is None: %s", request.reference_image is None)
        if request.reference_image:
            logger.info("   - request.reference_image length: %d", len(request.reference_image))
            logger.info("   - request.reference_image is empty string: %s", request.reference_image.strip() == "")
            logger.info("   - request.reference_image first 50 chars: %s", request.reference_image[:50] if len(request.reference_image) > 50 else request.reference_image)
        
        if request.reference_image and request.reference_image.strip() != "":
            logger.info("🔍 [Backend Debug] Processing reference image for insertion")
            reference = base64_to_image(request.reference_image)
            # Resize reference to match original image size
            if reference.size != original.size:
                logger.info(f"🔄 Resizing reference image from {reference.size} to {original.size}")
                reference = reference.resize(original.size, Image.Resampling.LANCZOS)
            conditional_images.append(reference)
            logger.info("✅ Reference image added for object insertion")
        else:
            logger.info("🔍 [Backend Debug] No reference image provided - using removal model")
        
        logger.info(f"🔍 [Backend Debug] Conditional images count: {len(conditional_images)} (expected 4 for removal, 5 for insertion with MAE)")
        return original, mask, conditional_images, reference

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
        
        # Apply prompt composition for insertion/removal tasks only
        # White balance tasks skip prompt composition (they use simple prompts or empty)
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
            "num_inference_steps": request.num_inference_steps or 40,
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

    def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        start = time.time()

        try:
            # Determine task type: use request.task_type if provided, otherwise auto-detect
            if request.task_type:
                task_type = request.task_type
                logger.info("🎯 Using task_type from request: %s", task_type)
            else:
                # Auto-detect based on whether reference_image is provided
                has_reference = request.reference_image is not None and request.reference_image.strip() != ""
                task_type = "insertion" if has_reference else "removal"
                logger.info("🔍 [Backend Debug] Task type auto-detection:")
                logger.info("   - reference_image is None: %s", request.reference_image is None)
                logger.info("   - reference_image provided: %s", has_reference)
                logger.info("   - reference_image length: %s", len(request.reference_image) if request.reference_image else 0)
                if request.reference_image:
                    logger.info("   - reference_image first 100 chars: %s", request.reference_image[:100])
                logger.info("   - Determined task_type: %s", task_type)
            
            logger.info("🎯 Task type: %s", task_type)
            
            pipeline = self._ensure_pipeline(task_type=task_type)
            original, mask, conditionals, reference = self._prepare_images(request, task_type)
            
            with torch.no_grad():
                # For white-balance task, use DiffusionPipeline (Pix2Pix)
                if task_type == "white-balance":
                    logger.info("🎨 Using Pix2Pix pipeline for white balance")
                    
                    # Prompt is optional for white balance, can be empty string
                    prompt = request.prompt if request.prompt else ""
                    
                    # Default parameters for Pix2Pix white balance
                    num_inference_steps = request.num_inference_steps or 20
                    image_guidance_scale = 1.5  # Default from reference code
                    guidance_scale = 0  # Default from reference code (0 means no text guidance)
                    
                    # Get device for generator
                    device_obj = get_device()
                    
                    # Create generator with seed if provided
                    generator = None
                    if request.seed is not None:
                        generator = torch.Generator(device=device_obj).manual_seed(request.seed)
                    else:
                        generator = torch.Generator(device=device_obj).manual_seed(0)  # Default seed
                    
                    # Pix2Pix model needs both image and prompt
                    # According to reference: pipe(prompt, image=image, num_inference_steps=..., image_guidance_scale=..., guidance_scale=..., generator=...)
                    # The model loaded might be StableDiffusionInstructPix2PixPipeline which requires image parameter
                    try:
                        result = pipeline(
                            prompt=prompt,
                            image=original,
                            num_inference_steps=num_inference_steps,
                            image_guidance_scale=image_guidance_scale,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        )
                        generated = result.images[0]
                        
                        # Ensure generated image has same size as original
                        if generated.size != original.size:
                            logger.info(f"🔄 Resizing generated image from {generated.size} to {original.size} to match input")
                            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
                    except (TypeError, ValueError) as e:
                        # If parameters are not accepted, try simpler call
                        logger.warning(f"Pipeline call failed with full parameters, trying simpler call: {e}")
                        try:
                            result = pipeline(prompt=prompt, image=original)
                            generated = result.images[0]
                            # Ensure generated image has same size as original
                            if generated.size != original.size:
                                logger.info(f"🔄 Resizing generated image from {generated.size} to {original.size} to match input")
                                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
                        except (TypeError, ValueError) as e2:
                            # Last resort: try prompt only (for text-to-image models)
                            logger.warning(f"Pipeline doesn't accept image parameter, trying prompt only: {e2}")
                            result = pipeline(prompt=prompt)
                            generated = result.images[0]
                            # For text-to-image, resize to match original
                            if generated.size != original.size:
                                logger.info(f"🔄 Resizing generated image from {generated.size} to {original.size} to match input")
                                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
                else:
                    # For other tasks (insertion/removal), check if using QwenImageEditPlusPipeline
                    is_qwen_pipeline = False
                    try:
                        from diffusers import QwenImageEditPlusPipeline
                        is_qwen_pipeline = isinstance(pipeline, QwenImageEditPlusPipeline)
                    except (ImportError, AttributeError):
                        # QwenImageEditPlusPipeline not available, use standard pipeline
                        # Also check by class name as fallback
                        try:
                            pipeline_class_name = pipeline.__class__.__name__
                            is_qwen_pipeline = "QwenImageEditPlus" in pipeline_class_name
                        except:
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
                            "prompt": request.prompt,
                            "num_inference_steps": request.num_inference_steps or 40,
                            "true_cfg_scale": request.true_cfg_scale or request.guidance_scale or 4.0,
                            "height": original.height,
                            "width": original.width,
                        }
                        
                        if request.negative_prompt:
                            qwen_params["negative_prompt"] = request.negative_prompt
                        
                        if request.seed is not None:
                            device = get_device()
                            qwen_params["generator"] = torch.Generator(device=device).manual_seed(request.seed)
                        
                        logger.info(f"📥 Calling QwenImageEditPlusPipeline with {len(image_list)} conditional images")
                        result = pipeline(**qwen_params)
                        
                        # QwenImageEditPlusPipeline returns QwenImagePipelineOutput with images attribute
                        if hasattr(result, "images"):
                            generated = result.images[0] if isinstance(result.images, list) else result.images
                        else:
                            generated = result[0] if isinstance(result, (list, tuple)) else result
                    else:
                        # For standard StableDiffusionInpaintPipeline, use existing params
                        params = self._build_params(request, original, mask, conditionals, task_type=task_type)
                        result = pipeline(**params)  # type: ignore[operator]
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

            generation_time = time.time() - start
            logger.info("✅ Image generated in %.2fs", generation_time)

            # Save visualization
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Build metadata - handle white-balance case where params might not exist
            if task_type == "white-balance":
                metadata = {
                    "prompt": prompt,  # Use the prompt we used for generation
                    "negative_prompt": request.negative_prompt,
                    "num_inference_steps": request.num_inference_steps or 40,
                    "guidance_scale": request.guidance_scale or 1.0,
                    "true_cfg_scale": request.true_cfg_scale,
                    "seed": request.seed,
                    "image_size": f"{original.width}x{original.height}",
                    "generation_time": round(generation_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "has_reference_image": reference is not None,
                    "task_type": "white-balance",
                }
            else:
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
                }
            # Use conditional images for visualization (RGB format)
            # For white-balance, conditionals are empty, so skip them
            conditional_images_for_viz = list(conditionals) if task_type != "white-balance" else []
            # If reference image was added, include it in visualization
            if reference and len(conditional_images_for_viz) == 3:
                conditional_images_for_viz = conditional_images_for_viz + [reference]
            
            self._visualization.save_generation_visualization(
                request_id=request_id,
                original=original,
                mask=mask if task_type != "white-balance" else None,  # No mask for white-balance
                conditional_images=conditional_images_for_viz,
                output=generated,
                reference_image=reference,
                metadata=metadata,
            )

            return {
                "success": True,
                "image": image_to_base64(generated),
                "generation_time": round(generation_time, 2),
                "model_used": f"qwen_local_{task_type}",
                "parameters_used": metadata,
                "request_id": request_id,  # Include request_id for visualization access
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error during generation")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

