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

    # Low-end optimization settings (for GPU 12GB or lower)
    enable_4bit_text_encoder: bool = Field(
        default=False, alias="ENABLE_4BIT_TEXT_ENCODER",
        description="Enable 4-bit quantization for text encoder (saves ~4GB VRAM)"
    )
    enable_cpu_offload: bool = Field(
        default=False, alias="ENABLE_CPU_OFFLOAD",
        description="Enable CPU offload for transformer and VAE (saves VRAM, slower)"
    )
    enable_memory_optimizations: bool = Field(
        default=False, alias="ENABLE_MEMORY_OPTIMIZATIONS",
        description="Enable memory optimizations (safetensors, low_cpu_mem_usage, TF32)"
    )
    enable_flowmatch_scheduler: bool = Field(
        default=False, alias="ENABLE_FLOWMATCH_SCHEDULER",
        description="Use FlowMatchEulerDiscreteScheduler instead of default"
    )
    scheduler_shift: float = Field(
        default=3.0, alias="SCHEDULER_SHIFT", ge=0.0, le=10.0,
        description="Shift parameter for FlowMatch scheduler"
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

    @property
    def input_quality_preset(self) -> str:
        """Return preset type for configured input quality."""
        return INPUT_QUALITY_PRESETS.get(self.input_quality, INPUT_QUALITY_PRESETS["resized"])

    @property
    def input_quality_presets(self) -> Dict[str, str]:
        """Expose available input quality presets (for diagnostics)."""
        return INPUT_QUALITY_PRESETS


settings = Settings()

