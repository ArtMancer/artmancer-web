"""
Reference-Guided Object Insertion Pipeline

Implements object insertion where reference_mask_R provides both appearance 
(via reference_image) and shape (via mask boundary), while main_mask_A only 
defines placement location. Generation occurs ONLY inside the positioned 
reference_mask_R region, not the full main_mask_A.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_and_validate_inputs(
    base_image: Image.Image,
    main_mask_A: Image.Image,
    reference_image: Image.Image,
    reference_mask_R: Image.Image,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    Load and validate all inputs for reference-guided insertion.
    
    Args:
        base_image: Base/background image
        main_mask_A: User-drawn placement mask (defines WHERE to insert)
        reference_image: Reference image (provides appearance)
        reference_mask_R: Reference mask (provides shape, MUST be provided)
    
    Returns:
        Tuple of validated (base_image, main_mask_A, reference_image, reference_mask_R)
    
    Raises:
        ValueError: If any input is invalid or missing
    """
    # Validate base_image
    if base_image is None:
        raise ValueError("base_image is required")
    if base_image.mode != "RGB":
        base_image = base_image.convert("RGB")
    
    # Validate main_mask_A
    if main_mask_A is None:
        raise ValueError("main_mask_A is required")
    if main_mask_A.size != base_image.size:
        logger.info(f"🔄 Resizing main_mask_A from {main_mask_A.size} to {base_image.size}")
        main_mask_A = main_mask_A.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Validate reference_image
    if reference_image is None:
        raise ValueError("reference_image is required")
    if reference_image.mode != "RGB":
        reference_image = reference_image.convert("RGB")
    
    # Validate reference_mask_R (MUST be provided, no auto-segmentation)
    if reference_mask_R is None:
        raise ValueError(
            "reference_mask_R is required for reference-guided insertion. "
            "Auto-segmentation is not supported. Please provide reference_mask_R."
        )
    if reference_mask_R.size != reference_image.size:
        logger.info(f"🔄 Resizing reference_mask_R from {reference_mask_R.size} to {reference_image.size}")
        reference_mask_R = reference_mask_R.resize(reference_image.size, Image.Resampling.LANCZOS)
    
    logger.info(
        f"✅ Validated inputs: base_image={base_image.size}, "
        f"main_mask_A={main_mask_A.size}, reference_image={reference_image.size}, "
        f"reference_mask_R={reference_mask_R.size}"
    )
    
    return base_image, main_mask_A, reference_image, reference_mask_R


def calculate_mask_bounding_box(mask: Image.Image) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box of white region in mask.
    
    Args:
        mask: Mask image (white where region is, black elsewhere)
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) bounding box coordinates
    """
    mask_array = np.array(mask.convert("L"))
    white_pixels = np.where(mask_array > 128)
    
    if len(white_pixels[0]) == 0:
        # No white pixels, return full image bounds
        return (0, 0, mask.width, mask.height)
    
    y_min, y_max = int(np.min(white_pixels[0])), int(np.max(white_pixels[0])) + 1
    x_min, x_max = int(np.min(white_pixels[1])), int(np.max(white_pixels[1])) + 1
    
    return (x_min, y_min, x_max, y_max)


def position_reference_mask_in_main_mask(
    base_image: Image.Image,
    main_mask_A: Image.Image,
    reference_mask_R: Image.Image,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Position reference_mask_R inside main_mask_A region.
    
    Resizes reference_mask_R to fit inside main_mask_A (preserving aspect ratio)
    and centers it within the main_mask_A bounding box.
    
    Args:
        base_image: Base image (for output size)
        main_mask_A: Placement mask (defines WHERE to insert)
        reference_mask_R: Reference mask (defines WHAT shape to insert)
    
    Returns:
        Tuple of (positioned_mask_R, transformation_info)
        - positioned_mask_R: Reference mask resized and positioned (same size as base_image)
        - transformation_info: Dict with scale, position, bounding boxes
    """
    # Ensure masks match base_image size
    if main_mask_A.size != base_image.size:
        logger.info(f"🔄 Resizing main_mask_A from {main_mask_A.size} to {base_image.size}")
        main_mask_A = main_mask_A.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Calculate bounding boxes
    x_min_A, y_min_A, x_max_A, y_max_A = calculate_mask_bounding_box(main_mask_A)
    x_min_R, y_min_R, x_max_R, y_max_R = calculate_mask_bounding_box(reference_mask_R)
    
    # Calculate dimensions
    width_A = x_max_A - x_min_A
    height_A = y_max_A - y_min_A
    width_R = x_max_R - x_min_R
    height_R = y_max_R - y_min_R
    
    # Validate mask areas
    if width_A == 0 or height_A == 0:
        raise ValueError("main_mask_A has zero area - cannot position reference mask")
    if width_R == 0 or height_R == 0:
        raise ValueError("reference_mask_R has zero area - cannot position")
    
    # Calculate scale factor to fit reference_mask_R inside main_mask_A
    # Preserve aspect ratio - use the smaller scale factor
    scale_x = width_A / width_R
    scale_y = height_A / height_R
    scale = min(scale_x, scale_y)  # Preserve aspect ratio
    
    # Calculate new reference_mask_R size
    new_width = int(width_R * scale)
    new_height = int(height_R * scale)
    
    # Resize reference_mask_R (preserving aspect ratio)
    resized_mask_R = reference_mask_R.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )
    
    # Calculate center position within main_mask_A bounding box
    center_x_A = x_min_A + width_A // 2
    center_y_A = y_min_A + height_A // 2
    
    # Calculate paste position (center the resized mask)
    paste_x = center_x_A - new_width // 2
    paste_y = center_y_A - new_height // 2
    
    # Ensure paste position is within image bounds
    paste_x = max(0, min(paste_x, base_image.width - new_width))
    paste_y = max(0, min(paste_y, base_image.height - new_height))
    
    # Create positioned_mask_R canvas (same size as base_image)
    positioned_mask_R = Image.new("L", base_image.size, 0)  # Black background
    
    # Paste resized reference_mask_R at calculated position
    # Convert to L mode for proper alpha handling
    if resized_mask_R.mode != "L":
        resized_mask_R = resized_mask_R.convert("L")
    
    positioned_mask_R.paste(resized_mask_R, (paste_x, paste_y))
    
    # Store transformation info
    transformation_info = {
        "scale": scale,
        "paste_position": (paste_x, paste_y),
        "resized_size": (new_width, new_height),
        "main_mask_bbox": (x_min_A, y_min_A, x_max_A, y_max_A),
        "reference_mask_bbox": (x_min_R, y_min_R, x_max_R, y_max_R),
    }
    
    logger.info(
        f"✅ Positioned reference_mask_R: "
        f"original size {reference_mask_R.size} -> "
        f"resized {new_width}x{new_height} at ({paste_x}, {paste_y}), "
        f"scale factor {scale:.3f} (preserved aspect ratio)"
    )
    
    return positioned_mask_R, transformation_info


def create_masked_background(
    base_image: Image.Image,
    positioned_mask_R: Image.Image,
) -> Image.Image:
    """
    Create masked background with empty regions at positioned_mask_R location.
    
    Copies base_image and creates empty (black) regions where positioned_mask_R is.
    This provides the generation context: model will fill these empty regions.
    
    Args:
        base_image: Base/background image
        positioned_mask_R: Reference mask positioned inside main_mask_A
    
    Returns:
        masked_bg: Base image with empty regions at positioned_mask_R location
    """
    # Ensure positioned_mask_R matches base_image size
    if positioned_mask_R.size != base_image.size:
        logger.warning(
            f"⚠️ positioned_mask_R size {positioned_mask_R.size} != base_image size {base_image.size}, "
            "resizing positioned_mask_R"
        )
        positioned_mask_R = positioned_mask_R.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    base_array = np.array(base_image.convert("RGB"), dtype=np.float32) / 255.0
    mask_array = np.array(positioned_mask_R.convert("L"), dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)
    
    # Stack mask to 3 channels
    mask_stack = np.repeat(mask_array[..., None], 3, axis=2)
    
    # Create masked_bg: keep base_image where mask is NOT, black (empty) where mask is
    # This creates empty regions for generation
    masked_bg_array = base_array * (1.0 - mask_stack)
    
    # Convert back to PIL Image
    masked_bg = Image.fromarray(
        np.uint8(np.clip(masked_bg_array * 255, 0, 255)),
        mode="RGB"
    )
    
    mask_coverage = np.mean(mask_array) * 100
    logger.info(
        f"✅ Created masked_bg: base_image with empty regions at positioned_mask_R "
        f"({mask_coverage:.1f}% coverage)"
    )
    
    return masked_bg


def prepare_condition_images(
    reference_image: Image.Image,
    positioned_mask_R: Image.Image,
    masked_bg: Image.Image,
    base_image: Image.Image,
) -> List[Image.Image]:
    """
    Prepare conditional images for Qwen Image Edit pipeline.
    
    All images must be the same size as base_image.
    Order: [reference_image, positioned_mask_R, masked_bg]
    
    Args:
        reference_image: Full reference image (for appearance learning)
        positioned_mask_R: Reference mask positioned inside main_mask_A (for generation boundary)
        masked_bg: Base image with empty regions at positioned_mask_R location
        base_image: Base image (for size reference)
    
    Returns:
        List of conditional images in correct order for Qwen pipeline
    """
    target_size = base_image.size
    
    # Resize reference_image to match base_image size
    if reference_image.size != target_size:
        logger.info(f"🔄 Resizing reference_image from {reference_image.size} to {target_size}")
        reference_image = reference_image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Ensure positioned_mask_R is RGB (Qwen expects RGB masks)
    if positioned_mask_R.mode != "RGB":
        positioned_mask_R = positioned_mask_R.convert("RGB")
    
    # Ensure masked_bg matches size
    if masked_bg.size != target_size:
        logger.warning(f"⚠️ masked_bg size {masked_bg.size} != target size {target_size}, resizing")
        masked_bg = masked_bg.resize(target_size, Image.Resampling.LANCZOS)
    
    # Prepare condition images in correct order for Qwen
    conditional_images = [
        reference_image,      # [0] Full reference image (appearance)
        positioned_mask_R,    # [1] Positioned mask (generation boundary)
        masked_bg,            # [2] Masked background (context with empty regions)
    ]
    
    logger.info(
        f"✅ Prepared condition images: "
        f"[reference_image={reference_image.size}, "
        f"positioned_mask_R={positioned_mask_R.size}, "
        f"masked_bg={masked_bg.size}]"
    )
    
    return conditional_images


def composite_final_result(
    generated_image: Image.Image,
    base_image: Image.Image,
    positioned_mask_R: Image.Image,
) -> Image.Image:
    """
    Composite generated content into base image using positioned_mask_R region.
    
    Keeps base_image everywhere except positioned_mask_R region.
    In positioned_mask_R region, uses generated content.
    
    Args:
        generated_image: Generated image (contains content only in positioned_mask_R region)
        base_image: Original base/background image
        positioned_mask_R: Reference mask positioned inside main_mask_A (generation boundary)
    
    Returns:
        Final composited image
    """
    # Ensure all images have same size
    if generated_image.size != base_image.size:
        logger.warning(
            f"⚠️ Generated image size {generated_image.size} != base_image size {base_image.size}, "
            "resizing generated image"
        )
        generated_image = generated_image.resize(base_image.size, Image.Resampling.LANCZOS)
    
    if positioned_mask_R.size != base_image.size:
        logger.info(f"🔄 Resizing positioned_mask_R from {positioned_mask_R.size} to {base_image.size}")
        positioned_mask_R = positioned_mask_R.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    base_array = np.array(base_image.convert("RGB"), dtype=np.float32) / 255.0
    generated_array = np.array(generated_image.convert("RGB"), dtype=np.float32) / 255.0
    mask_array = np.array(positioned_mask_R.convert("L"), dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)
    
    # Stack mask to 3 channels
    mask_stack = np.repeat(mask_array[..., None], 3, axis=2)
    
    # Composite: use generated content where mask is, base_image elsewhere
    composited_array = base_array * (1.0 - mask_stack) + generated_array * mask_stack
    
    # Convert back to PIL Image
    composited_image = Image.fromarray(
        np.uint8(np.clip(composited_array * 255, 0, 255)),
        mode="RGB"
    )
    
    mask_coverage = np.mean(mask_array) * 100
    logger.info(
        f"✅ Composited final result: generated content in positioned_mask_R region "
        f"({mask_coverage:.1f}% coverage), base_image preserved elsewhere"
    )
    
    return composited_image


def execute_insertion_pipeline(
    base_image: Image.Image,
    main_mask_A: Image.Image,
    reference_image: Image.Image,
    reference_mask_R: Image.Image,
    pipeline: Any,
    prompt: str,
    num_inference_steps: int = 10,
    true_cfg_scale: float = 4.0,
    negative_prompt: str | None = None,
    seed: int | None = None,
    generator: Any | None = None,
    debug_session: Any | None = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Image.Image:
    """
    Execute complete reference-guided insertion pipeline.
    
    Main orchestrator that:
    1. Validates inputs
    2. Positions reference_mask_R inside main_mask_A
    3. Creates masked background
    4. Prepares condition images
    5. Runs Qwen generation with positioned_mask_R as generation mask
    6. Composites final result
    
    Args:
        base_image: Base/background image
        main_mask_A: User-drawn placement mask
        reference_image: Reference image (appearance)
        reference_mask_R: Reference mask (shape, MUST be provided)
        pipeline: Qwen Image Edit pipeline instance
        prompt: Generation prompt
        num_inference_steps: Number of inference steps
        true_cfg_scale: CFG scale for Qwen
        negative_prompt: Negative prompt (optional)
        seed: Random seed (optional)
        generator: Torch generator (optional, created from seed if not provided)
        debug_session: DebugSession instance (optional, for saving debug images)
    
    Returns:
        Final composited image with generated object inserted
    """
    logger.info("🚀 Starting reference-guided insertion pipeline")
    
    try:
        # Step 1: Load and validate inputs
        base_image, main_mask_A, reference_image, reference_mask_R = load_and_validate_inputs(
            base_image, main_mask_A, reference_image, reference_mask_R
        )
        
        # Step 2: Position reference_mask_R inside main_mask_A
        positioned_mask_R, transformation_info = position_reference_mask_in_main_mask(
            base_image, main_mask_A, reference_mask_R
        )
        logger.info(f"📐 Transformation: {transformation_info}")
        
        # Save debug images for mask positioning
        if debug_session is not None:
            debug_session.save_image(
                positioned_mask_R, 
                "positioned_mask_R", 
                "05", 
                f"Positioned mask R (scale={transformation_info['scale']:.3f}, pos={transformation_info['paste_position']})"
            )
            debug_session.log_lora(
                f"Mask positioning: scale={transformation_info['scale']:.3f}, "
                f"resized_size={transformation_info['resized_size']}, "
                f"paste_position={transformation_info['paste_position']}, "
                f"main_mask_bbox={transformation_info['main_mask_bbox']}, "
                f"reference_mask_bbox={transformation_info['reference_mask_bbox']}"
            )
        
        # Step 3: Create masked background
        masked_bg = create_masked_background(base_image, positioned_mask_R)
        
        # Save debug image for masked background
        if debug_session is not None:
            mask_coverage = np.mean(np.array(positioned_mask_R) > 128) * 100
            debug_session.save_image(
                masked_bg,
                "masked_bg",
                "06",
                f"Masked background (empty regions at positioned_mask_R, {mask_coverage:.1f}% coverage)"
            )
        
        # Step 4: Prepare condition images
        conditional_images = prepare_condition_images(
            reference_image, positioned_mask_R, masked_bg, base_image
        )
        
        # Save debug images for conditional images
        if debug_session is not None:
            debug_session.save_image(
                conditional_images[0],
                "conditional_reference_image",
                "07a",
                "Conditional image [0]: Reference image (appearance)"
            )
            debug_session.save_image(
                conditional_images[1],
                "conditional_positioned_mask",
                "07b",
                "Conditional image [1]: Positioned mask R (generation boundary)"
            )
            debug_session.save_image(
                conditional_images[2],
                "conditional_masked_bg",
                "07c",
                "Conditional image [2]: Masked background (context with empty regions)"
            )
            debug_session.log_lora(
                f"Conditional images prepared: {len(conditional_images)} images "
                f"[reference_image={conditional_images[0].size}, "
                f"positioned_mask_R={conditional_images[1].size}, "
                f"masked_bg={conditional_images[2].size}]"
            )
        
        # Step 5: Run Qwen generation
        # Use positioned_mask_R as generation mask (NOT main_mask_A)
        # Convert positioned_mask_R to RGB for Qwen
        generation_mask = positioned_mask_R.convert("RGB")
        
        logger.info(
            f"🎨 Running Qwen generation: "
            f"prompt='{prompt[:50]}...', steps={num_inference_steps}, "
            f"cfg_scale={true_cfg_scale}, mask_coverage={np.mean(np.array(positioned_mask_R) > 128) * 100:.1f}%"
        )
        
        # Prepare generator if not provided
        if generator is None and seed is not None:
            import torch
            from ..core.pipeline import get_device
            device = get_device()
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Call Qwen pipeline
        # Qwen expects: image (list of conditionals), prompt
        # The positioned_mask_R in conditional_images[1] acts as the generation boundary
        # Qwen will generate only inside the white regions of positioned_mask_R
        # Note: guidance_scale is ineffective for Qwen. Use true_cfg_scale instead.
        # negative_prompt is required (even empty string) to enable classifier-free guidance
        qwen_params = {
            "image": conditional_images,  # [reference_image, positioned_mask_R, masked_bg]
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "height": base_image.height,
            "width": base_image.width,
            # Always provide negative_prompt (even if empty) to enable classifier-free guidance
            "negative_prompt": negative_prompt if negative_prompt else " ",
        }
        
        if generator is not None:
            qwen_params["generator"] = generator
        
        # Add progress callback if provided
        if progress_callback is not None:
            def callback_on_step_end(pipeline_instance, step: int, timestep: int, callback_kwargs: dict):
                """Callback function for pipeline progress tracking."""
                try:
                    progress_callback(step, timestep, num_inference_steps)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
                return callback_kwargs
            
            qwen_params["callback_on_step_end"] = callback_on_step_end
        
        logger.info(
            f"📥 Calling Qwen pipeline with {len(conditional_images)} conditional images: "
            f"[reference_image, positioned_mask_R, masked_bg]"
        )
        
        # Run inference
        result = pipeline(**qwen_params)
        
        # Extract generated image
        if hasattr(result, "images"):
            generated_image = result.images[0] if isinstance(result.images, list) else result.images
        else:
            generated_image = result[0] if isinstance(result, (list, tuple)) else result
        
        # Ensure generated image matches base_image size
        if generated_image.size != base_image.size:
            logger.info(f"🔄 Resizing generated image from {generated_image.size} to {base_image.size}")
            generated_image = generated_image.resize(base_image.size, Image.Resampling.LANCZOS)
        
        # Save debug image for generated image (before compositing)
        if debug_session is not None:
            debug_session.save_image(
                generated_image,
                "generated_before_composite",
                "08",
                "Generated image from Qwen (before compositing)"
            )
            debug_session.log_lora(
                f"Qwen generation completed: "
                f"prompt='{prompt[:100]}...', "
                f"steps={num_inference_steps}, "
                f"cfg_scale={true_cfg_scale}, "
                f"seed={seed}, "
                f"generated_size={generated_image.size}"
            )
        
        # Step 6: Composite final result
        final_image = composite_final_result(generated_image, base_image, positioned_mask_R)
        
        logger.info("✅ Reference-guided insertion pipeline completed successfully")
        
        return final_image
        
    except Exception as e:
        logger.error(f"❌ Reference-guided insertion pipeline failed: {e}", exc_info=True)
        raise

