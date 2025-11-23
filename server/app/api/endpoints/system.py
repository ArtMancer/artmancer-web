from __future__ import annotations

from fastapi import APIRouter

from ...core.pipeline import clear_pipeline_cache, get_device_info
from ...models import HealthResponse

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from ...core.pipeline import is_pipeline_loaded
    
    device_info = get_device_info()
    # Check if any pipeline is loaded without forcing a load
    pipeline_loaded = is_pipeline_loaded()
    
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
    clear_pipeline_cache()
    return {"success": True, "message": "Cache cleared"}

