from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Đường dẫn mặc định cho 3 checkpoint Qwen (có thể override bằng biến môi trường)
DEFAULT_MODEL_FILE_INSERTION = CHECKPOINTS_DIR / "insertion_cp.safetensors"
DEFAULT_MODEL_FILE_REMOVAL = CHECKPOINTS_DIR / "removal_cp.safetensors"
DEFAULT_MODEL_FILE_WHITE_BALANCE = CHECKPOINTS_DIR / "wb_cp.safetensors"

INPUT_QUALITY_LEVELS: Dict[str, float] = {
    "super_low": 1 / 16,
    "low": 1 / 8,
    "medium": 1 / 4,
    "high": 1 / 2,
    "original": 1.0,
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
    input_quality: str = Field(default="high", alias="INPUT_QUALITY")
    input_quality_warning_px: int = Field(
        default=2048, alias="INPUT_QUALITY_WARNING_PX", ge=512, le=8192
    )

    @field_validator("input_quality", mode="before")
    @classmethod
    def validate_input_quality(cls, value: str) -> str:
        """Normalize and validate input quality level."""
        if not value:
            return "high"
        normalized = value.strip().lower()
        if normalized not in INPUT_QUALITY_LEVELS:
            raise ValueError(
                f"Invalid INPUT_QUALITY '{value}'. "
                f"Valid options: {', '.join(INPUT_QUALITY_LEVELS.keys())}"
            )
        return normalized

    @property
    def input_quality_scale(self) -> float:
        """Return scale factor for configured input quality."""
        return INPUT_QUALITY_LEVELS.get(self.input_quality, INPUT_QUALITY_LEVELS["high"])

    @property
    def input_quality_levels(self) -> Dict[str, float]:
        """Expose available input quality presets (for diagnostics)."""
        return INPUT_QUALITY_LEVELS


settings = Settings()

