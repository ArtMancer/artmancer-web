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
    guidance_scale: Optional[float] = Field(1.0, ge=0.5, le=20)
    true_cfg_scale: Optional[float] = Field(4.0, ge=0.5, le=15)
    negative_prompt: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    task_type: Optional[Literal["insertion", "removal", "white-balance"]] = Field(
        default=None, 
        description="Task type: 'insertion', 'removal', or 'white-balance' (auto-detected if not provided)"
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


class EvaluationImagePair(BaseModel):
    """Pair of images for evaluation: original and target."""
    original_image: str = Field(..., description="Base64 encoded original image")
    target_image: str = Field(..., description="Base64 encoded target/reference image")
    prompt: Optional[str] = Field(default=None, description="Prompt used for this image pair")
    filename: Optional[str] = Field(default=None, description="Original filename for matching pairs")


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


class EvaluationRequest(BaseModel):
    """Request for image evaluation."""
    # Task type (determines which model to use)
    task_type: Optional[str] = Field(
        default="object-removal", 
        description="Task type: 'object-insert' or 'object-removal' (determines which model to use)"
    )
    
    # Single image evaluation
    original_image: Optional[str] = Field(
        default=None, description="Base64 encoded original image (for single image evaluation)"
    )
    target_image: Optional[str] = Field(
        default=None, description="Base64 encoded target image (for single image evaluation)"
    )
    prompt: Optional[str] = Field(
        default=None, description="Prompt used for single image evaluation"
    )
    
    # Batch evaluation (multiple image pairs)
    image_pairs: Optional[List[EvaluationImagePair]] = Field(
        default=None, description="List of image pairs for batch evaluation"
    )
    
    # Reference image (for object-insert task)
    reference_image: Optional[str] = Field(
        default=None, description="Base64 encoded reference image (required for object-insert task)"
    )
    
    # Conditional images (optional, can have multiple)
    conditional_images: Optional[List[str]] = Field(
        default=None, description="List of base64 encoded conditional images (mask, background, etc.)"
    )
    
    # Input image (for conditional evaluation)
    input_image: Optional[str] = Field(
        default=None, description="Base64 encoded input image (for conditional evaluation)"
    )
    
    # Storage optimization: skip saving visualization for batch evaluation to save disk space
    save_visualization: Optional[bool] = Field(
        default=True, description="Whether to save visualization images (set False for batch to save storage)"
    )


class EvaluationMetrics(BaseModel):
    """Evaluation metrics scores."""
    # Image quality metrics
    psnr: Optional[float] = Field(default=None, description="Peak Signal-to-Noise Ratio [dB] - Higher is better (>30 good, >40 excellent)")
    ssim: Optional[float] = Field(default=None, description="Structural Similarity Index [0,1] - Higher is better (>0.90 good, >0.95 excellent)")
    lpips: Optional[float] = Field(default=None, description="Learned Perceptual Image Patch Similarity [0,1] - Lower is better (<0.10 good, <0.05 excellent)")
    de00: Optional[float] = Field(default=None, description="Delta E 2000 (color difference) - Lower is better (<1.0 imperceptible, 1-2 barely noticeable)")
    
    # Metadata
    evaluation_time: Optional[float] = Field(default=None, description="Time taken for evaluation in seconds")


class EvaluationResult(BaseModel):
    """Result for a single image pair evaluation."""
    filename: Optional[str] = Field(default=None, description="Filename if provided")
    metrics: EvaluationMetrics
    success: bool = Field(default=True, description="Whether evaluation was successful")
    error: Optional[str] = Field(default=None, description="Error message if evaluation failed")
    # Model and task information
    evaluation_task_type: Optional[str] = Field(default=None, description="Task type selected in evaluation UI")
    generation_task_type: Optional[str] = Field(default=None, description="Task type used for generation")
    model_file_used: Optional[str] = Field(default=None, description="Model file path used for generation")
    actual_model_used: Optional[str] = Field(default=None, description="Actual model name used for generation")


class EvaluationResponse(BaseModel):
    """Response for evaluation request."""
    success: bool
    results: List[EvaluationResult] = Field(..., description="List of evaluation results")
    total_pairs: int = Field(..., description="Total number of image pairs evaluated")
    successful_evaluations: int = Field(..., description="Number of successful evaluations")
    failed_evaluations: int = Field(..., description="Number of failed evaluations")
    total_evaluation_time: float = Field(..., description="Total time taken for all evaluations")


class BenchmarkRequest(BaseModel):
    """Request for benchmark evaluation."""
    input_path: str = Field(..., description="Path to folder, ZIP file, or Parquet file")
    task_type: str = Field(
        default="object-removal",
        description="Task type: 'object-removal', 'object-insert', or 'white-balance'"
    )
    prompt: str = Field(
        default="remove the object",
        description="Prompt for image generation"
    )
    sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of samples to process (0 = all images)"
    )
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    guidance_scale: Optional[float] = Field(default=None, ge=0.5, le=20)
    true_cfg_scale: Optional[float] = Field(default=None, ge=0.5, le=15)
    negative_prompt: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    input_quality: Optional[str] = Field(
        default=None,
        description="Input quality preset: 'super_low', 'low', 'medium', 'high', 'original'"
    )


class BenchmarkResult(BaseModel):
    """Result for a single benchmark image."""
    image_id: str = Field(..., description="Image ID (filename without extension)")
    filename: str = Field(..., description="Original filename")
    success: bool = Field(..., description="Whether generation was successful")
    psnr: Optional[float] = Field(default=None, description="PSNR score [dB]")
    ssim: Optional[float] = Field(default=None, description="SSIM score [0,1]")
    lpips: Optional[float] = Field(default=None, description="LPIPS score [0,1]")
    clip_score: Optional[float] = Field(default=None, description="CLIP-S score [0,1]")
    delta_e: Optional[float] = Field(default=None, description="Delta E 2000 score")
    generation_time: float = Field(default=0.0, description="Generation time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BenchmarkSummary(BaseModel):
    """Summary statistics for benchmark results."""
    total_images: int
    successful: int
    failed: int
    psnr: Optional[Dict[str, float]] = None
    ssim: Optional[Dict[str, float]] = None
    lpips: Optional[Dict[str, float]] = None
    clip_score: Optional[Dict[str, float]] = None
    delta_e: Optional[Dict[str, float]] = None
    generation_time: Optional[Dict[str, float]] = None


class BenchmarkResponse(BaseModel):
    """Response for benchmark request."""
    success: bool
    results: List[BenchmarkResult] = Field(..., description="Per-image results")
    summary: BenchmarkSummary = Field(..., description="Summary statistics")
    total_time: float = Field(..., description="Total benchmark time in seconds")
    exported_files: Optional[Dict[str, str]] = Field(
        default=None,
        description="Paths to exported files (CSV, JSON, LaTeX)"
    )

