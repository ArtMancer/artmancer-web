from __future__ import annotations

from fastapi import APIRouter

from ...models import GenerationRequest, GenerationResponse
from ...services.generation_service import GenerationService

router = APIRouter(prefix="/api", tags=["generation"])
service = GenerationService()


@router.post("/generate", response_model=GenerationResponse)
def generate_image(request: GenerationRequest):
    """
    Generate image endpoint.
    
    Uses sync def because:
    - This runs on Heavy Worker (A100) where PyTorch/GPU tasks are blocking
    - Modal automatically wraps sync functions in thread pool
    - No benefit from async for CPU/GPU bound tasks (Python GIL)
    """
    return service.generate(request)

