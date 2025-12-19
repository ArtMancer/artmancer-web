from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of the desired edit")
    input_image: str = Field(..., description="Base64 encoded original image")
    # Qwen: only uses additional condition images (mask + other conditions) as base64
    conditional_images: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of base64-encoded conditional images for Qwen. "
            "By convention, conditional_images[0] is the mask; "
            "subsequent items are additional condition images."
        ),
    )
    # Reference image (for object-insert task - used as object conditional)
    reference_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded reference image (required for insertion task, used as object conditional)"
    )
    # Reference mask R (for two-source mask workflow - isolates object in reference image)
    reference_mask_R: Optional[str] = Field(
        default=None,
        description="Base64 encoded mask image isolating object in reference image (Mask R - defines object shape for insertion)"
    )
    width: Optional[int] = Field(None, ge=256, le=2048)
    height: Optional[int] = Field(None, ge=256, le=2048)
    num_inference_steps: Optional[int] = Field(10, ge=1, le=100)
    guidance_scale: Optional[float] = Field(4.0, ge=0.5, le=20)
    true_cfg_scale: Optional[float] = Field(4.0, ge=0.5, le=15)
    negative_prompt: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    task_type: Optional[Literal["insertion", "removal", "white-balance"]] = Field(
        default=None, 
        description="Task type: 'insertion', 'removal', or 'white-balance' (auto-detected if not provided)"
    )
    refine_mask_with_birefnet: Optional[bool] = Field(
        default=True,
        description="Refine mask using BiRefNet for removal task (default: True)"
    )
    angle: Optional[str] = Field(default=None, description="Angle macro label for prompt composition (e.g., 'wide-angle', 'top-down'). Only used for insert/remove tasks, ignored for white-balance.")
    background_preset: Optional[str] = Field(default=None, description="Background preset name for prompt composition (e.g., 'marble-surface', 'white-background'). Only used for insert/remove tasks, ignored for white-balance.")
    input_quality: Optional[Literal["resized", "original"]] = Field(
        default=None,
        description="Optional override for input quality preset ('resized' for 1:1 square, 'original' to keep original size).",
    )
    enable_flowmatch_scheduler: Optional[bool] = Field(
        default=None,
        description="Use FlowMatchEulerDiscreteScheduler instead of default. Overrides config if provided."
    )
    mask_tool_type: Optional[Literal["brush", "box"]] = Field(
        default=None,
        description="Mask creation tool type: 'brush' or 'box'. Used for MAE preprocessing in removal task."
    )
    enable_mae_refinement: Optional[bool] = Field(
        default=True,
        description="Enable Stable Diffusion Inpainting refinement for LaMa output (default: True). Improves texture quality in removal task."
    )
    enable_debug: Optional[bool] = Field(
        default=False,
        description="Enable debug mode to save debug images and logs (default: False). Only enable when needed for debugging."
    )


class DebugInfo(BaseModel):
    """Debug information for generation request."""
    conditional_images: Optional[List[str]] = Field(
        default=None, 
        description="Base64 encoded conditional images used for generation (mask, background, object, mae)"
    )
    conditional_labels: Optional[List[str]] = Field(
        default=None,
        description="Labels for each conditional image"
    )
    input_image_size: Optional[str] = Field(default=None, description="Original input image size")
    output_image_size: Optional[str] = Field(default=None, description="Final output image size")
    lora_adapter: Optional[str] = Field(default=None, description="LoRA adapter used")
    loaded_adapters: Optional[List[str]] = Field(default=None, description="All loaded LoRA adapters")
    positioned_mask_R: Optional[str] = Field(
        default=None,
        description="Base64 encoded positioned mask R (reference mask R after being pasted into main mask A, only for reference-guided insertion)"
    )
    # Additional debug images for downloads
    original_image: Optional[str] = Field(default=None, description="Base64 encoded original input image")
    mask_A: Optional[str] = Field(default=None, description="Base64 encoded mask A (conditional_images[0])")
    reference_image: Optional[str] = Field(default=None, description="Base64 encoded reference image")
    reference_mask_R: Optional[str] = Field(default=None, description="Base64 encoded reference mask R")
    mask_mae_dilated: Optional[str] = Field(
        default=None,
        description="Base64 encoded dilated mask used for MAE generation (removal task with brush mask only)"
    )


class GenerationResponse(BaseModel):
    success: bool
    image: str
    generation_time: float
    model_used: str
    parameters_used: Dict[str, str | float | int | None]
    request_id: Optional[str] = Field(default=None, description="Request ID for accessing visualization images")
    debug_info: Optional[DebugInfo] = Field(default=None, description="Debug information (conditional images, parameters)")


class ModelSettings(BaseModel):
    name: str
    description: str
    max_width: int
    max_height: int
    supports_mask: bool = True
    supports_multi_image: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    device_info: Dict[str, str | int | float | bool]


class WhiteBalanceRequest(BaseModel):
    """Request for white balance correction."""
    method: str = Field(default="auto", description="White balance method: 'auto', 'manual', or 'ai'")
    temperature: Optional[float] = Field(default=None, description="Temperature adjustment (-100 to 100, for manual method)")
    tint: Optional[float] = Field(default=None, description="Tint adjustment (-100 to 100, for manual method)")


class WhiteBalanceResponse(BaseModel):
    """Response from white balance correction."""
    success: bool
    original_image: str = Field(..., description="Base64 encoded original image")
    corrected_image: str = Field(..., description="Base64 encoded corrected image")
    method_used: str = Field(..., description="Method that was used: 'auto', 'manual', or 'ai'")
    parameters: Optional[Dict[str, float]] = Field(default=None, description="Parameters used for correction")

