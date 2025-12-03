from __future__ import annotations

from fastapi import APIRouter

from ...models import GenerationRequest, GenerationResponse
from ...services.generation_service import GenerationService

router = APIRouter(prefix="/api", tags=["generation"])
service = GenerationService()


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    return await service.generate(request)

