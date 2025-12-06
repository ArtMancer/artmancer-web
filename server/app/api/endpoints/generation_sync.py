"""
Synchronous generation endpoints for direct image generation.
Used by QwenService (A100).
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...models import GenerationRequest, GenerationResponse
from ...services.generation_service import GenerationService

router = APIRouter(prefix="/api/generate", tags=["generation-sync"])
service = GenerationService()


@router.post("", response_model=GenerationResponse)
def generate_image(request: GenerationRequest):
    """
    Generate image endpoint (synchronous, backward compatible).
    
    Uses sync def because:
    - This runs on QwenService (A100) where PyTorch/GPU tasks are blocking
    - Modal automatically wraps sync functions in thread pool
    - No benefit from async for CPU/GPU bound tasks (Python GIL)
    """
    return service.generate(request)


@router.post("/cancel/{task_id}")
def cancel_generation(task_id: str) -> Dict[str, Any]:
    """
    Cancel a synchronous generation task.
    
    Args:
        task_id: Task identifier (request_id from generation request)
    
    Returns:
        Success message
    """
    try:
        GenerationService.set_cancelled(task_id)
        return {
            "success": True,
            "message": f"Generation task {task_id} marked for cancellation",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel generation: {str(e)}"
        )

