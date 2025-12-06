from __future__ import annotations

from fastapi import APIRouter

from ...models import HealthResponse

router = APIRouter(prefix="/api", tags=["system"])


def _get_device_info_safe():
    """Get device info without requiring heavy dependencies."""
    try:
        from ...core.pipeline import get_device_info
        return get_device_info()
    except ImportError:
        # Light service doesn't have full pipeline dependencies
        import torch
        return {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
        }


def _is_pipeline_loaded_safe():
    """Check if pipeline is loaded without requiring heavy dependencies."""
    try:
        from ...core.pipeline import is_pipeline_loaded
        return is_pipeline_loaded()
    except ImportError:
        # Light service doesn't have pipeline
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device_info = _get_device_info_safe()
    pipeline_loaded = _is_pipeline_loaded_safe()
    
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline_loaded,
        device=device_info.get("device", "cpu"),
        device_info=device_info,
    )


@router.get("/")
async def root():
    """Root endpoint for testing."""
    return {"message": "ArtMancer API is running", "version": "2.1.0"}


@router.post("/clear-cache")
async def clear_cache():
    """Clear pipeline cache (only works on Heavy service)."""
    try:
        from ...core.pipeline import clear_pipeline_cache
        clear_pipeline_cache()
        return {"success": True, "message": "Cache cleared"}
    except ImportError:
        return {"success": False, "message": "Cache clear not available on this service"}

