from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FILE_INSERTION = BASE_DIR / "qwen_2509_object_insertion_512_000002750.safetensors"
DEFAULT_MODEL_FILE_REMOVAL = BASE_DIR / "qwen2509_object_removal_512_000002500.safetensors"
DEFAULT_MODEL_FILE_WHITE_BALANCE = "ArtMancer/Pix2Pix_wb"

INPUT_QUALITY_LEVELS: Dict[str, float] = {
    "super_low": 1 / 16,
    "low": 1 / 8,
    "medium": 1 / 4,
    "high": 1 / 2,
    "original": 1.0,
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ArtMancer API"
    description: str = "AI-powered image editing with local Qwen models"
    version: str = "2.1.0"
    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8003, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    # CORS configuration
    allowed_origins_raw: str = Field(default="http://localhost:3000", alias="ALLOWED_ORIGINS")
    # Model file paths
    model_file_insertion: str = Field(default=str(DEFAULT_MODEL_FILE_INSERTION), alias="MODEL_FILE_INSERTION")
    model_file_removal: str = Field(default=str(DEFAULT_MODEL_FILE_REMOVAL), alias="MODEL_FILE_REMOVAL")
    model_file_white_balance: Optional[str] = Field(
        default=DEFAULT_MODEL_FILE_WHITE_BALANCE, alias="MODEL_FILE_WHITE_BALANCE"
    )
    enable_xformers: bool = Field(default=True, alias="ENABLE_XFORMERS")
    attention_slicing: bool = Field(default=True, alias="ATTENTION_SLICING")
    # VAE slicing for memory optimization
    enable_vae_slicing: bool = Field(default=True, alias="ENABLE_VAE_SLICING")
    vae_slicing_tile_size: int = Field(default=512, alias="VAE_SLICING_TILE_SIZE", ge=256, le=1024)
    enable_vae_slicing_white_balance: bool = Field(default=True, alias="ENABLE_VAE_SLICING_WHITE_BALANCE")
    # Visualization settings
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    visualization_dir: str = Field(default="", alias="VISUALIZATION_DIR")  # Empty = auto (visualizations/)
    # Model preloading: if True, load models on startup (faster first request, slower startup)
    preload_models: bool = Field(default=False, alias="PRELOAD_MODELS")
    input_quality: str = Field(default="high", alias="INPUT_QUALITY")
    input_quality_warning_px: int = Field(
        default=2048, alias="INPUT_QUALITY_WARNING_PX", ge=512, le=8192
    )
    _allowed_origins: List[str] | None = None

    @model_validator(mode="after")
    def parse_allowed_origins(self) -> "Settings":
        """Parse allowed_origins from comma-separated string or JSON array."""
        value = self.allowed_origins_raw
        if not value:
            self._allowed_origins = ["http://localhost:3000"]
            return self
        # Try JSON first
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                self._allowed_origins = parsed
                return self
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back to comma-separated string
        self._allowed_origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        return self

    @property
    def allowed_origins(self) -> List[str]:
        """Get parsed allowed origins list."""
        if self._allowed_origins is None:
            # Fallback if validator didn't run
            return ["http://localhost:3000"]
        return self._allowed_origins

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

