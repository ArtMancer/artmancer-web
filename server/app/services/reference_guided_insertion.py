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

# Lazy import scipy for connected component detection
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, multi-region mask detection will use fallback method")


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


def detect_mask_regions(
    mask: Image.Image,
    min_region_size: int = 100,
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Detect all connected components (regions) in mask.
    
    Finds all separate mask regions using connected component analysis.
    Filters out very small regions (noise).
    
    Args:
        mask: Mask image (white where region is, black elsewhere)
        min_region_size: Minimum number of pixels for a region to be considered (default: 100)
    
    Returns:
        List of tuples: [(region_mask, bbox), ...]
        - region_mask: PIL Image containing only this region (same size as input mask)
        - bbox: Tuple of (x_min, y_min, x_max, y_max) bounding box coordinates
    """
    # Convert mask to binary array
    mask_array = np.array(mask.convert("L"))
    binary_mask = (mask_array > 128).astype(np.uint8)
    
    # Find connected components
    if SCIPY_AVAILABLE:
        # Use scipy.ndimage.label for efficient connected component detection
        labeled_array, num_features = ndimage.label(binary_mask)  # type: ignore
    else:
        # Fallback: simple flood fill approach (less efficient but works without scipy)
        logger.warning("Using fallback connected component detection (scipy not available)")
        labeled_array = np.zeros_like(binary_mask, dtype=np.int32)
        num_features = 0
        visited = np.zeros_like(binary_mask, dtype=bool)
        
        def flood_fill(y, x, label):
            """Simple flood fill to label connected components."""
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if (cy < 0 or cy >= binary_mask.shape[0] or 
                    cx < 0 or cx >= binary_mask.shape[1] or 
                    visited[cy, cx] or binary_mask[cy, cx] == 0):
                    continue
                visited[cy, cx] = True
                labeled_array[cy, cx] = label
                stack.append((cy-1, cx))
                stack.append((cy+1, cx))
                stack.append((cy, cx-1))
                stack.append((cy, cx+1))
        
        for y in range(binary_mask.shape[0]):
            for x in range(binary_mask.shape[1]):
                if binary_mask[y, x] > 0 and not visited[y, x]:
                    num_features += 1
                    flood_fill(y, x, num_features)
    
    if num_features == 0:
        logger.warning("No mask regions detected")
        return []
    
    logger.info(f"🔍 Detected {num_features} connected component(s) in mask")
    
    # Extract each region
    regions = []
    for label_id in range(1, num_features + 1):
        # Create binary mask for this region
        region_binary = (labeled_array == label_id).astype(np.uint8) * 255
        
        # Check region size
        region_size = np.sum(region_binary > 0)
        if region_size < min_region_size:
            logger.debug(f"   Skipping region {label_id}: too small ({region_size} pixels < {min_region_size})")
            continue
        
        # Calculate bounding box
        region_pixels = np.where(region_binary > 0)
        if len(region_pixels[0]) == 0:
            continue
        
        y_min, y_max = int(np.min(region_pixels[0])), int(np.max(region_pixels[0])) + 1
        x_min, x_max = int(np.min(region_pixels[1])), int(np.max(region_pixels[1])) + 1
        bbox = (x_min, y_min, x_max, y_max)
        
        # Create PIL Image for this region (same size as original mask)
        region_mask = Image.fromarray(region_binary, mode="L")
        
        regions.append((region_mask, bbox))
        logger.debug(f"   Region {label_id}: bbox={bbox}, size={region_size} pixels")
    
    logger.info(f"✅ Found {len(regions)} valid mask region(s) (filtered from {num_features} components)")
    return regions


def position_reference_mask_in_single_region(
    base_image: Image.Image,
    region_mask: Image.Image,
    reference_mask_R: Image.Image,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Position reference_mask_R inside a single mask region.
    
    This is the core positioning logic used for both single and multi-region cases.
    
    Args:
        base_image: Base image (for output size)
        region_mask: Single mask region (defines WHERE to insert for this region)
        reference_mask_R: Reference mask (defines WHAT shape to insert)
    
    Returns:
        Tuple of (positioned_mask_R, transformation_info)
        - positioned_mask_R: Reference mask resized and positioned for this region (same size as base_image)
        - transformation_info: Dict with scale, position, bounding boxes
    """
    # Calculate bounding boxes
    x_min_A, y_min_A, x_max_A, y_max_A = calculate_mask_bounding_box(region_mask)
    x_min_R, y_min_R, x_max_R, y_max_R = calculate_mask_bounding_box(reference_mask_R)
    
    # Calculate dimensions of bounding boxes
    width_A = x_max_A - x_min_A
    height_A = y_max_A - y_min_A
    width_R_bbox = x_max_R - x_min_R
    height_R_bbox = y_max_R - y_min_R
    
    # Validate mask areas
    if width_A == 0 or height_A == 0:
        raise ValueError("region_mask has zero area - cannot position reference mask")
    if width_R_bbox == 0 or height_R_bbox == 0:
        raise ValueError("reference_mask_R has zero area - cannot position")
    
    # Calculate scale factor to fit reference_mask_R bounding box inside region_mask bounding box
    # Preserve aspect ratio - use the smaller scale factor
    scale_x = width_A / width_R_bbox
    scale_y = height_A / height_R_bbox
    scale = min(scale_x, scale_y)  # Preserve aspect ratio
    
    # Calculate new size for the entire reference_mask_R image
    # Scale is applied to the full image size, not just the bounding box
    original_mask_R_width = reference_mask_R.width
    original_mask_R_height = reference_mask_R.height
    new_width = int(original_mask_R_width * scale)
    new_height = int(original_mask_R_height * scale)
    
    # Resize the entire reference_mask_R image (preserving aspect ratio)
    resized_mask_R = reference_mask_R.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )
    
    # Recalculate bounding box of resized mask R
    x_min_R_resized, y_min_R_resized, x_max_R_resized, y_max_R_resized = calculate_mask_bounding_box(resized_mask_R)
    width_R_resized = x_max_R_resized - x_min_R_resized
    height_R_resized = y_max_R_resized - y_min_R_resized
    
    # Calculate center position within region_mask bounding box (in base_image coordinates)
    center_x_A = x_min_A + width_A // 2
    center_y_A = y_min_A + height_A // 2
    
    # Calculate center of bounding box in resized mask R (in resized_mask_R coordinates)
    center_x_R_resized = x_min_R_resized + width_R_resized // 2
    center_y_R_resized = y_min_R_resized + height_R_resized // 2
    
    # Calculate paste position to align centers
    paste_x = center_x_A - center_x_R_resized
    paste_y = center_y_A - center_y_R_resized
    
    # Create positioned_mask_R canvas (same size as base_image)
    positioned_mask_R = Image.new("L", base_image.size, 0)  # Black background
    
    # Convert resized mask to L mode for proper alpha handling
    if resized_mask_R.mode != "L":
        resized_mask_R = resized_mask_R.convert("L")
    
    # Paste resized reference_mask_R at calculated position
    # Ensure paste position is within image bounds
    paste_x = max(0, min(paste_x, base_image.width - resized_mask_R.width))
    paste_y = max(0, min(paste_y, base_image.height - resized_mask_R.height))
    
    positioned_mask_R.paste(resized_mask_R, (paste_x, paste_y))
    
    # Store transformation info
    transformation_info = {
        "scale": scale,
        "paste_position": (paste_x, paste_y),
        "resized_size": (new_width, new_height),
        "resized_bbox": (x_min_R_resized, y_min_R_resized, x_max_R_resized, y_max_R_resized),
        "region_bbox": (x_min_A, y_min_A, x_max_A, y_max_A),
        "reference_mask_bbox": (x_min_R, y_min_R, x_max_R, y_max_R),
        "original_mask_R_size": (original_mask_R_width, original_mask_R_height),
    }
    
    return positioned_mask_R, transformation_info


def position_reference_mask_in_multiple_regions(
    base_image: Image.Image,
    main_mask_A: Image.Image,
    reference_mask_R: Image.Image,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Position reference_mask_R inside multiple mask regions.
    
    Detects all connected components in main_mask_A, scales and positions 
    reference_mask_R into each region independently, then combines all 
    positioned masks into a single mask.
    
    Args:
        base_image: Base image (for output size)
        main_mask_A: Placement mask (may contain multiple disconnected regions)
        reference_mask_R: Reference mask (defines WHAT shape to insert)
    
    Returns:
        Tuple of (combined_positioned_mask_R, transformation_infos)
        - combined_positioned_mask_R: Combined mask with reference_mask_R positioned in all regions
        - transformation_infos: List of transformation info dicts, one for each region
    """
    # Ensure main_mask_A matches base_image size
    if main_mask_A.size != base_image.size:
        logger.info(f"🔄 Resizing main_mask_A from {main_mask_A.size} to {base_image.size}")
        main_mask_A = main_mask_A.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Detect all mask regions
    regions = detect_mask_regions(main_mask_A)
    
    if len(regions) == 0:
        raise ValueError("No valid mask regions detected in main_mask_A")
    
    logger.info(f"🎯 Processing {len(regions)} mask region(s) for multi-region insertion")
    
    # Position reference_mask_R in each region
    positioned_masks = []
    transformation_infos = []
    
    for idx, (region_mask, region_bbox) in enumerate(regions, 1):
        logger.info(f"   Processing region {idx}/{len(regions)}: bbox={region_bbox}")
        
        try:
            positioned_mask, transform_info = position_reference_mask_in_single_region(
                base_image, region_mask, reference_mask_R
            )
            positioned_masks.append(positioned_mask)
            transform_info["region_index"] = idx
            transform_info["region_bbox"] = region_bbox
            transformation_infos.append(transform_info)
            
            logger.info(
                f"   ✅ Region {idx}: positioned at {transform_info['paste_position']}, "
                f"scale={transform_info['scale']:.3f}"
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to position in region {idx}: {e}")
            continue
    
    if len(positioned_masks) == 0:
        raise ValueError("Failed to position reference_mask_R in any region")
    
    # Combine all positioned masks using OR operation (union)
    combined_mask_array = np.zeros((base_image.height, base_image.width), dtype=np.uint8)
    
    for positioned_mask in positioned_masks:
        mask_array = np.array(positioned_mask.convert("L"))
        # OR operation: any pixel with value > 0 becomes 255
        combined_mask_array = np.maximum(combined_mask_array, mask_array)
    
    # Convert to PIL Image
    combined_positioned_mask_R = Image.fromarray(combined_mask_array, mode="L")
    
    total_coverage = np.mean(combined_mask_array > 0) * 100
    logger.info(
        f"✅ Combined {len(positioned_masks)} positioned mask(s): "
        f"total coverage {total_coverage:.1f}%"
    )
    
    return combined_positioned_mask_R, transformation_infos


def position_reference_mask_in_main_mask(
    base_image: Image.Image,
    main_mask_A: Image.Image,
    reference_mask_R: Image.Image,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Position reference_mask_R inside main_mask_A region(s).
    
    Automatically detects if main_mask_A contains multiple disconnected regions.
    If multiple regions found, positions reference_mask_R in each region separately.
    If single region, uses optimized single-region positioning.
    
    Args:
        base_image: Base image (for output size)
        main_mask_A: Placement mask (defines WHERE to insert, may contain multiple regions)
        reference_mask_R: Reference mask (defines WHAT shape to insert)
    
    Returns:
        Tuple of (positioned_mask_R, transformation_info)
        - positioned_mask_R: Reference mask resized and positioned (same size as base_image)
        - transformation_info: Dict with scale, position, bounding boxes
          For single region: single dict
          For multiple regions: dict with 'regions' key containing list of transformation infos
    """
    # Ensure masks match base_image size
    if main_mask_A.size != base_image.size:
        logger.info(f"🔄 Resizing main_mask_A from {main_mask_A.size} to {base_image.size}")
        main_mask_A = main_mask_A.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Detect mask regions
    regions = detect_mask_regions(main_mask_A)
    
    if len(regions) == 0:
        raise ValueError("No valid mask regions detected in main_mask_A")
    elif len(regions) == 1:
        # Single region: use optimized single-region positioning
        region_mask, _ = regions[0]
        positioned_mask_R, transformation_info = position_reference_mask_in_single_region(
            base_image, region_mask, reference_mask_R
        )
        # Add main_mask_bbox for backward compatibility
        transformation_info["main_mask_bbox"] = transformation_info.get("region_bbox", (0, 0, base_image.width, base_image.height))
        
        logger.info(
            f"✅ Positioned reference_mask_R (single region): "
            f"scale={transformation_info['scale']:.3f}, "
            f"position={transformation_info['paste_position']}"
        )
        
        return positioned_mask_R, transformation_info
    else:
        # Multiple regions: use multi-region positioning
        logger.info(f"🎯 Multiple regions detected ({len(regions)}), using multi-region positioning")
        positioned_mask_R, transformation_infos = position_reference_mask_in_multiple_regions(
            base_image, main_mask_A, reference_mask_R
        )
        
        # Create combined transformation info for backward compatibility
        transformation_info = {
            "num_regions": len(transformation_infos),
            "regions": transformation_infos,
            # For backward compatibility, include first region's info at top level
            "scale": transformation_infos[0]["scale"] if transformation_infos else 1.0,
            "paste_position": transformation_infos[0]["paste_position"] if transformation_infos else (0, 0),
        }
        
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
) -> Tuple[Image.Image, Image.Image]:
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
        Tuple of (final_composited_image, positioned_mask_R)
        - final_composited_image: Final composited image with generated object inserted
        - positioned_mask_R: Reference mask R after being positioned inside main_mask_A
    """
    logger.info("🚀 Starting reference-guided insertion pipeline")
    
    try:
        # Step 1: Load and validate inputs
        base_image, main_mask_A, reference_image, reference_mask_R = load_and_validate_inputs(
            base_image, main_mask_A, reference_image, reference_mask_R
        )
        
        # Step 2: Position reference_mask_R inside main_mask_A
        # This automatically detects multiple regions and handles them appropriately
        positioned_mask_R, transformation_info = position_reference_mask_in_main_mask(
            base_image, main_mask_A, reference_mask_R
        )
        
        # Log transformation info (handle both single and multi-region cases)
        if "num_regions" in transformation_info:
            num_regions = transformation_info["num_regions"]
            logger.info(f"📐 Multi-region transformation: {num_regions} region(s)")
            for idx, region_info in enumerate(transformation_info["regions"], 1):
                logger.info(
                    f"   Region {idx}: scale={region_info['scale']:.3f}, "
                    f"pos={region_info['paste_position']}, bbox={region_info['region_bbox']}"
                )
        else:
            logger.info(
                f"📐 Single-region transformation: "
                f"scale={transformation_info['scale']:.3f}, "
                f"pos={transformation_info['paste_position']}"
            )
        
        # Save debug images for mask positioning
        if debug_session is not None:
            if "num_regions" in transformation_info:
                num_regions = transformation_info["num_regions"]
                debug_session.save_image(
                    positioned_mask_R, 
                    "positioned_mask_R", 
                    "05", 
                    f"Positioned mask R (combined from {num_regions} region(s))"
                )
                debug_session.log_lora(
                    f"Multi-region mask positioning: {num_regions} region(s)"
                )
                for idx, region_info in enumerate(transformation_info["regions"], 1):
                    debug_session.log_lora(
                        f"   Region {idx}: scale={region_info['scale']:.3f}, "
                        f"resized_size={region_info['resized_size']}, "
                        f"paste_position={region_info['paste_position']}, "
                        f"region_bbox={region_info['region_bbox']}, "
                        f"reference_mask_bbox={region_info['reference_mask_bbox']}"
                    )
            else:
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
                    f"main_mask_bbox={transformation_info.get('main_mask_bbox', 'N/A')}, "
                    f"reference_mask_bbox={transformation_info.get('reference_mask_bbox', 'N/A')}"
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
        # positioned_mask_R is already in conditional_images[1] as RGB
        
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
        
        return final_image, positioned_mask_R
        
    except Exception as e:
        logger.error(f"❌ Reference-guided insertion pipeline failed: {e}", exc_info=True)
        raise

