from __future__ import annotations

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
)
from ..services.prompt_composer import compose_prompt
from ..models.schemas import GenerationRequest

logger = logging.getLogger(__name__)


try:
    # Dùng HTTPException của FastAPI nếu có (khi chạy qua FastAPI/Modal)
    from fastapi import HTTPException  # type: ignore
except Exception:  # pragma: no cover - fallback cho môi trường không có fastapi
    class HTTPException(Exception):  # type: ignore[override]
        """Minimal HTTP-like exception để dùng trong core service khi không có FastAPI."""

        def __init__(self, status_code: int, detail: str) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)


class GenerationService:
    def __init__(self) -> None:
        self._pipeline_insertion = None
        self._pipeline_removal = None

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
            # White-balance cũng dùng Qwen pipeline (một biến thể cấu hình nhẹ hơn).
            if self._pipeline_insertion is None:
                self._pipeline_insertion = load_pipeline(task_type="insertion")
            return self._pipeline_insertion
        else:  # insertion (default)
            if self._pipeline_insertion is None:
                self._pipeline_insertion = load_pipeline(task_type="insertion")
            return self._pipeline_insertion

    def _resolve_quality(
        self, override: str | None = None
    ) -> Tuple[str, float]:
        levels = settings.input_quality_levels
        quality_label = settings.input_quality
        scale = settings.input_quality_scale

        if override:
            normalized = override.strip().lower()
            if normalized in levels:
                quality_label = normalized
                scale = levels[normalized]
            else:
                logger.warning(
                    "⚠️ Unknown input quality override '%s'. Falling back to '%s'.",
                    override,
                    quality_label,
                )
        return quality_label, scale

    def _apply_input_quality(
        self,
        original: Image.Image,
        mask: Image.Image | None = None,
        reference: Image.Image | None = None,
        quality_override: str | None = None,
    ) -> tuple[Image.Image, Image.Image | None, Image.Image | None]:
        """Downscale inputs according to configured quality preset to save VRAM."""
        quality_label, scale = self._resolve_quality(quality_override)

        if scale >= 0.999:
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

        new_width = max(64, int(original.width * scale))
        new_height = max(64, int(original.height * scale))
        new_size = (new_width, new_height)

        logger.info(
            "🎚️ Applying input quality '%s' (scale %.4f): %dx%d -> %dx%d",
            quality_label,
            scale,
            original.width,
            original.height,
            new_width,
            new_height,
        )

        original = original.resize(new_size, Image.Resampling.LANCZOS)
        if mask is not None:
            mask = mask.resize(new_size, Image.Resampling.NEAREST)
        if reference is not None:
            reference = reference.resize(new_size, Image.Resampling.LANCZOS)

        return original, mask, reference

    def _prepare_images(
        self, request: GenerationRequest, task_type: str = "removal"
    ) -> tuple[Image.Image, Image.Image, list[Image.Image], Image.Image | None]:
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")

        original = base64_to_image(request.input_image)
        
        # Qwen flow mới:
        # - conditional_images[0]: mask
        # - conditional_images[1:]: các condition bổ sung (background masked, canny, mae, ...)
        cond_list = list(request.conditional_images or [])
        extra_cond_b64s: list[str] = []

        if task_type == "white-balance":
            # White-balance: có thể không gửi mask, khi đó dùng full white mask.
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
            # insertion / removal: bắt buộc phải có ít nhất 1 condition làm mask
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

            # Áp dụng input_quality cho insert/remove
            original, mask, _ = self._apply_input_quality(
                original, mask, None, quality_override=request.input_quality
        )
        
        if mask is None:
            raise RuntimeError("Mask image missing after preprocessing")

        # Từ mask build các condition cơ bản: mask/background/object/mae
        mask_cond, background_rgb, obj_rgb, mae_image = prepare_mask_conditionals(
            original, mask
        )
        conditional_images: list[Image.Image] = [
            mask_cond,
            background_rgb,
            obj_rgb,
            mae_image,
        ]
        
        # Append các condition bổ sung từ client (ảnh base64 bất kỳ)
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
        # reference không còn dùng trong Qwen flow mới
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
        
        # Apply prompt composition cho insertion/removal; white-balance dùng prompt raw/đơn giản
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
                # Map task_type từ API format sang internal
                if task_type == "object-removal":
                    task_type = "removal"
                elif task_type == "object-insert":
                    task_type = "insertion"
            else:
                # Không có task_type: mặc định removal
                task_type = "removal"
            logger.info("🎯 Task type (resolved): %s", task_type)
            
            pipeline = self._ensure_pipeline(task_type=task_type)
            pipeline_callable = cast(Any, pipeline)
            original, mask, conditionals, reference = self._prepare_images(request, task_type)
            
            with torch.no_grad():
                # Tất cả task (insert/remove/white-balance) đều dùng Qwen; khác biệt chỉ ở params/conditionals.
                # Kiểm tra xem pipeline hiện tại có phải QwenImageEditPlusPipeline.
                if task_type == "white-balance":
                    logger.info("🎨 Using Qwen pipeline for white-balance task")
                # For other tasks (insertion/removal/white-balance), check if using QwenImageEditPlusPipeline
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
                        
                        # Build params cho QwenImageEditPlusPipeline
                        qwen_params = {
                            "image": image_list,
                            "prompt": request.prompt or "",
                            "num_inference_steps": request.num_inference_steps or (20 if task_type == "white-balance" else 40),
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
                        result = pipeline_callable(**qwen_params)
                        
                        # QwenImageEditPlusPipeline returns QwenImagePipelineOutput with images attribute
                        if hasattr(result, "images"):
                            generated = result.images[0] if isinstance(result.images, list) else result.images
                        else:
                            generated = result[0] if isinstance(result, (list, tuple)) else result
                    else:
                        # For standard StableDiffusionInpaintPipeline, use existing params
                        params = self._build_params(request, original, mask, conditionals, task_type=task_type)
                    result = pipeline_callable(**params)
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

