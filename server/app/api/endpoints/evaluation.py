from __future__ import annotations

from fastapi import APIRouter

from ...models import EvaluationRequest, EvaluationResponse
from ...services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api", tags=["evaluation"])
service = EvaluationService()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_images(request: EvaluationRequest):
    """
    Evaluate image quality and similarity metrics.
    
    Supports:
    - Single image pair evaluation
    - Batch evaluation with multiple image pairs
    - Conditional images (optional)
    """
    return service.evaluate(request)

