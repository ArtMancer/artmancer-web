from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import routers
from app.api.endpoints import generation, system, smart_mask, image_utils, debug
from app.core.pipeline import is_pipeline_loaded
from app.services.generation_service import GenerationService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArtMancer API",
    description="AI-powered image editing with Qwen models",
    version="2.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(generation.router)
app.include_router(system.router)
app.include_router(smart_mask.router)
app.include_router(image_utils.router)
app.include_router(debug.router)

# Global service instance
_service: GenerationService | None = None


def _get_service() -> GenerationService:
    """Initialize and cache GenerationService for RunPod worker."""
    global _service
    if _service is None:
        logger.info("🔧 Initializing GenerationService inside RunPod worker")
        _service = GenerationService()
        # Preload pipeline for insertion task to avoid cold start delay
        try:
            logger.info("🔄 Preloading Qwen pipeline for insertion task...")
            # Use asyncio.run to call async function from sync context
            asyncio.run(_service._ensure_pipeline(task_type="insertion"))
            logger.info("✅ Qwen pipeline preloaded and cached")
        except Exception as e:
            logger.warning("⚠️ Failed to preload pipeline (will load on first request): %s", e)
    return _service


# Bắt buộc: Health check endpoint cho RunPod Load Balancer
@app.get("/ping", response_model=None)
async def ping() -> Response:
    """
    Health check endpoint required by RunPod Load Balancer.
    
    Returns:
        - 200: Worker healthy and ready
        - 204: Worker initializing (cold start)
        - 500: Worker unhealthy
    """
    try:
        if is_pipeline_loaded():
            return Response(content='{"status": "healthy"}', media_type="application/json", status_code=200)
        else:
            return Response(status_code=204)  # 204 No Content - Initializing
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(status_code=500)  # Unhealthy


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint for GET requests."""
    return {
        "success": True,
        "message": "ArtMancer RunPod endpoint is online. Use POST /api/generate for generation.",
        "version": "2.1.0",
    }


@app.post("/")
async def generate_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main generation endpoint:
    
    - Input JSON: matches GenerationRequest schema
    - Output JSON: matches GenerationResponse schema
    """
    from pydantic import ValidationError
    from app.models.schemas import GenerationRequest
    
    try:
        payload = body or {}
        if not isinstance(payload, dict):
            raise TypeError("Request body must be a JSON object.")

        request = GenerationRequest.model_validate(payload)
        service = _get_service()
        result = await service.generate(request)

        # Ensure success is always present
        if "success" not in result:
            result["success"] = True

        return result
    except ValidationError as exc:
        logger.warning("Validation error in RunPod generate endpoint: %s", exc)
        return {
            "success": False,
            "error_type": "validation_error",
            "errors": exc.errors(),
        }
    except Exception as exc:
        logger.exception("Unexpected error in RunPod generate endpoint")
        return {
            "success": False,
            "error_type": "runtime_error",
            "error": str(exc),
        }


@app.on_event("startup")
async def startup_event():
    """Preload pipeline on startup to reduce cold start time."""
    logger.info("🚀 Starting ArtMancer API on RunPod...")
    try:
        # Preload pipeline in background
        _get_service()
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload pipeline on startup: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

