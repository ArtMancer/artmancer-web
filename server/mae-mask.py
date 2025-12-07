# MAE
def generate_mae_image(
    original: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Generate MAE (Masked Autoencoder) conditional image.
    
    Uses OpenCV inpainting (Telea algorithm) to create a preview of the inpainted
    region, providing better context representation for the model.
    
    Args:
        original: Original RGB image
        mask: Mask image (white where object is, black elsewhere)
    
    Returns:
        MAE image: RGB image with inpainted masked region
    """
    # Resize mask to match original image size if needed
    if mask.size != original.size:
        logger.info(f"🔄 Resizing mask from {mask.size} to {original.size} for MAE")
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    
    # Convert PIL images to numpy arrays
    original_array = np.array(original, dtype=np.uint8)
    mask_gray = mask.convert("L")
    mask_array = np.array(mask_gray, dtype=np.uint8)
    
    # OpenCV inpainting expects mask where region to inpaint is non-zero (white)
    # Our mask is already white where object/edit area is, so use it directly (no inversion needed)
    mask_for_inpaint = mask_array
    
    # MAE requires OpenCV; raise clear error if missing
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV library (cv2) is not installed in the runtime environment. "
            "Cannot generate MAE image. "
            "Please install 'opencv-python-headless' (recommended) or 'opencv-python' package, "
            "or disable MAE in conditional preparation logic."
        )
    
    # Use OpenCV inpainting with Telea algorithm (fast and effective)
    # cv2.INPAINT_TELEA is faster than cv2.INPAINT_NS (Navier-Stokes)
    try:
        mae_array = cv2.inpaint(
            original_array,
            mask_for_inpaint,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )
        mae_image = Image.fromarray(mae_array, mode="RGB")
        logger.info("✅ Generated MAE image using OpenCV Telea inpainting")
        return mae_image
    except Exception as exc:
        logger.error("❌ OpenCV inpainting failed for MAE image: %s", exc)
        raise RuntimeError(
            f"OpenCV inpainting failed while generating MAE image: {exc}"
        ) from exc
        
        
        

# Mask
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

    # Resize mask to match original image size if needed
    if mask.size != original.size:
        logger.info(f"🔄 Resizing mask from {mask.size} to {original.size}")
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
        # Fallback: use masked_bg as MAE (backward compatibility)
        mae_image = masked_bg
        logger.info("🧩 Generated conditional images (mask/masked_bg/masked_object) - MAE disabled")

    return mask_rgb, masked_bg, masked_object, mae_image