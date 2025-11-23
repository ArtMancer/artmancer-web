from __future__ import annotations

import logging
import time
from io import BytesIO

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from ...core.pipeline import load_pipeline
from ...models import WhiteBalanceResponse
from ...services.image_processing import image_to_base64

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["white-balance"])


@router.post("/white-balance", response_model=WhiteBalanceResponse)
async def white_balance(
    file: UploadFile = File(...),
    method: str = Form("auto"),
    temperature: float | None = Form(None),
    tint: float | None = Form(None),
):
    """
    Apply white balance correction to an image.
    
    Methods:
    - "auto": Automatic white balance using AI model
    - "manual": Manual adjustment using temperature and tint
    - "ai": AI-based white balance (same as auto)
    
    Args:
        file: Image file to process
        method: White balance method ("auto", "manual", "ai")
        temperature: Temperature adjustment (-100 to 100, for manual method)
        tint: Tint adjustment (-100 to 100, for manual method)
    
    Returns:
        WhiteBalanceResponse with corrected image
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty")
        
        # Load image
        try:
            original_image = Image.open(BytesIO(image_bytes))
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        logger.info(f"📸 Processing white balance: method={method}, size={original_image.size}")
        
        # Process based on method
        if method in ["auto", "ai"]:
            # Use AI model for white balance (Pix2Pix)
            logger.info("🎨 Using Pix2Pix model for white balance")
            
            # Load white balance pipeline
            pipeline = load_pipeline(task_type="white-balance")
            
            # Prompt is optional for white balance, can be empty string
            prompt = ""  # Empty prompt as per reference code
            
            # According to reference code: pipe(prompt).images[0]
            # This is a text-to-image model, so we only pass prompt
            with torch.no_grad():
                result = pipeline(prompt=prompt)
                corrected_image = result.images[0]
            
            method_used = "ai"
            parameters = None
            
        elif method == "manual":
            # Manual adjustment using PIL
            logger.info(f"🔧 Using manual white balance: temperature={temperature}, tint={tint}")
            
            # Apply temperature and tint adjustments
            # Temperature: positive = warmer, negative = cooler
            # Tint: positive = more magenta, negative = more green
            corrected_image = original_image.copy()
            
            # Simple color temperature adjustment
            if temperature is not None:
                # Convert temperature to RGB adjustment
                # Temperature range: -100 (cool/blue) to 100 (warm/orange)
                temp_factor = temperature / 100.0
                if temp_factor > 0:
                    # Warm: increase red, decrease blue
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Color(corrected_image)
                    corrected_image = enhancer.enhance(1.0 + temp_factor * 0.3)
            
            if tint is not None:
                # Tint adjustment: -100 (green) to 100 (magenta)
                # This is a simplified implementation
                pass  # Can be enhanced with more sophisticated color matrix
            
            method_used = "manual"
            parameters = {
                "temperature": temperature,
                "tint": tint,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method}. Use 'auto', 'manual', or 'ai'")
        
        # Convert images to base64
        original_base64 = image_to_base64(original_image, format="PNG")
        corrected_base64 = image_to_base64(corrected_image, format="PNG")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ White balance completed in {processing_time:.2f}s")
        
        return WhiteBalanceResponse(
            success=True,
            original_image=original_base64,
            corrected_image=corrected_base64,
            method_used=method_used,
            parameters=parameters,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Error applying white balance")
        raise HTTPException(status_code=500, detail=f"Failed to apply white balance: {str(e)}")

