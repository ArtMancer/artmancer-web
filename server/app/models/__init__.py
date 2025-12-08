"""
Models package for ArtMancer backend.

This package contains Pydantic models for request/response validation:
- schemas: Main API request/response models
- unified_schemas: Unified request schema for multiple task types

All models use Pydantic BaseModel for automatic validation and serialization.
"""

from .schemas import (
    DebugInfo,
    EvaluationImagePair,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    EvaluationMetrics,
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    ModelSettings,
    WhiteBalanceRequest,
    WhiteBalanceResponse,
)

__all__ = [
    "DebugInfo",
    "EvaluationImagePair",
    "EvaluationRequest",
    "EvaluationResponse",
    "EvaluationResult",
    "EvaluationMetrics",
    "GenerationRequest",
    "GenerationResponse",
    "HealthResponse",
    "ModelSettings",
    "WhiteBalanceRequest",
    "WhiteBalanceResponse",
]

