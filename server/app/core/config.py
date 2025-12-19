from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]

# Check for Modal Volume mount point first
# Modal volumes are mounted at /checkpoints by default
MODAL_VOLUME_PATH = Path("/checkpoints")
if MODAL_VOLUME_PATH.exists() and MODAL_VOLUME_PATH.is_dir():
    # Use Modal Volume if available (Modal deployment)
    # Volume is mounted at /checkpoints, checkpoints are stored directly there
    CHECKPOINTS_DIR = MODAL_VOLUME_PATH
# Check for RunPod network volume mount point
# RunPod mounts volumes at /runpod-volume/ by default, but can be configured
elif (RUNPOD_VOLUME_PATH := Path("/runpod-volume")).exists() and RUNPOD_VOLUME_PATH.is_dir():
    # Use network volume if available (RunPod deployment)
    CHECKPOINTS_DIR = RUNPOD_VOLUME_PATH / "checkpoints"
else:
    # Fallback to local checkpoints directory (local development)
    CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Default paths for 3 Qwen checkpoints (can be overridden by environment variables)
DEFAULT_MODEL_FILE_INSERTION = CHECKPOINTS_DIR / "insertion_cp.safetensors"
DEFAULT_MODEL_FILE_REMOVAL = CHECKPOINTS_DIR / "removal_cp.safetensors"
DEFAULT_MODEL_FILE_WHITE_BALANCE = CHECKPOINTS_DIR / "wb_cp.safetensors"

INPUT_QUALITY_PRESETS: Dict[str, str] = {
    "resized": "1:1",     # Resize về aspect ratio vuông (512x512, 1024x1024,...)
    "original": "keep",   # Giữ nguyên kích thước và aspect ratio gốc
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "ArtMancer API"
    description: str = "AI-powered image editing with local Qwen models"
    version: str = "2.1.0"

    # Model file paths (Qwen LoRA checkpoints)
    model_file_insertion: str = Field(default=str(DEFAULT_MODEL_FILE_INSERTION), alias="MODEL_FILE_INSERTION")
    model_file_removal: str = Field(default=str(DEFAULT_MODEL_FILE_REMOVAL), alias="MODEL_FILE_REMOVAL")
    model_file_white_balance: str = Field(
        default=str(DEFAULT_MODEL_FILE_WHITE_BALANCE), alias="MODEL_FILE_WHITE_BALANCE"
    )

    # Input quality + resize policy
    input_quality: str = Field(default="resized", alias="INPUT_QUALITY")
    input_quality_warning_px: int = Field(
        default=2048, alias="INPUT_QUALITY_WARNING_PX", ge=512, le=8192
    )

    enable_flowmatch_scheduler: bool = Field(
        default=False, alias="ENABLE_FLOWMATCH_SCHEDULER",
        description="Use FlowMatchEulerDiscreteScheduler instead of default"
    )
    scheduler_shift: float = Field(
        default=3.0, alias="SCHEDULER_SHIFT", ge=0.0, le=10.0,
        description="Shift parameter for FlowMatch scheduler"
    )
    
    # Adapter loading optimization
    enable_memory_aware_unloading: bool = Field(
        default=False, alias="ENABLE_MEMORY_AWARE_UNLOADING",
        description="Enable memory-aware adapter unloading (unload least-recently-used adapter when GPU memory is low). Default: False for H200 80GB."
    )
    gpu_memory_unload_threshold_mb: int = Field(
        default=50000, alias="GPU_MEMORY_UNLOAD_THRESHOLD_MB", ge=10000,
        description="Unload adapters if free GPU memory falls below this threshold (MB). Default: 50000 (50GB) for H200."
    )
    
    # MAE (Masked Autoencoder) configuration
    mae_model_size: str = Field(
        default="large", alias="MAE_MODEL_SIZE",
        description="ViT-MAE model size: 'base', 'large', or 'huge'. Larger models provide better reconstruction but use more memory. Default: 'large'."
    )
    mae_reconstruction_strength: float = Field(
        default=1.5, alias="MAE_RECONSTRUCTION_STRENGTH", ge=0.0, le=2.0,
        description="Strength of MAE reconstruction blending (0.0 = original only, 1.0 = full MAE, 2.0 = enhanced MAE). Default: 1.5 (enhanced mode)."
    )
    mae_use_enhanced_blending: bool = Field(
        default=True, alias="MAE_USE_ENHANCED_BLENDING",
        description="Use enhanced blending with edge-aware smoothing for better MAE reconstruction. Default: True."
    )
    # Custom MAE decoder configuration
    mae_decoder_dim: int = Field(
        default=512, alias="MAE_DECODER_DIM", ge=128, le=2048,
        description="Decoder dimension for custom MAE. Default: 512."
    )
    mae_decoder_depth: int = Field(
        default=1, alias="MAE_DECODER_DEPTH", ge=1, le=12,
        description="Decoder depth (number of transformer blocks). Default: 1."
    )
    mae_decoder_heads: int = Field(
        default=8, alias="MAE_DECODER_HEADS", ge=1, le=32,
        description="Number of attention heads in decoder. Default: 8."
    )
    mae_decoder_dim_head: int = Field(
        default=64, alias="MAE_DECODER_DIM_HEAD", ge=32, le=256,
        description="Dimension per attention head in decoder. Default: 64."
    )
    mae_masking_ratio: float = Field(
        default=0.75, alias="MAE_MASKING_RATIO", ge=0.1, le=0.95,
        description="Masking ratio for MAE (fraction of patches to mask). Default: 0.75."
    )
    mae_use_user_mask_guidance: bool = Field(
        default=True, alias="MAE_USE_USER_MASK_GUIDANCE",
        description="Guide masking by user mask (prioritize masking patches in user mask region). Default: True."
    )

    @field_validator("input_quality", mode="before")
    @classmethod
    def validate_input_quality(cls, value: str) -> str:
        """Normalize and validate input quality level."""
        if not value:
            return "resized"
        normalized = value.strip().lower()
        if normalized not in INPUT_QUALITY_PRESETS:
            raise ValueError(
                f"Invalid INPUT_QUALITY '{value}'. "
                f"Valid options: {', '.join(INPUT_QUALITY_PRESETS.keys())}"
            )
        return normalized
    
    @field_validator("mae_model_size", mode="before")
    @classmethod
    def validate_mae_model_size(cls, value: str) -> str:
        """Normalize and validate MAE model size."""
        if not value:
            return "large"
        normalized = value.strip().lower()
        if normalized not in ["base", "large", "huge"]:
            raise ValueError(
                f"Invalid MAE_MODEL_SIZE '{value}'. "
                "Valid options: 'base', 'large', 'huge'"
            )
        return normalized

    @property
    def input_quality_preset(self) -> str:
        """Return preset type for configured input quality."""
        return INPUT_QUALITY_PRESETS.get(self.input_quality, INPUT_QUALITY_PRESETS["resized"])

    @property
    def input_quality_presets(self) -> Dict[str, str]:
        """Expose available input quality presets (for diagnostics)."""
        return INPUT_QUALITY_PRESETS


settings = Settings()

