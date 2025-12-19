"""
Synchronous and async generation endpoints.

This module provides endpoints for image generation:
- Synchronous generation (blocking, for local development)
- Async generation with job tracking (for Modal deployment)
- Progress streaming via Server-Sent Events (SSE)

Note: In Modal environment, all requests automatically use async mode.
"""

from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from ...models import GenerationRequest
from ...services.generation_service import GenerationService
from ._helpers import (
    get_modal_imports,
    get_job_state_dictio,
    create_job_state,
    update_job_state_error,
    generate_sse_events,
)

router = APIRouter(prefix="/api", tags=["generation"])
service = GenerationService()


@router.post("/generate")
def generate_image(request: GenerationRequest):
    """
    Generate image endpoint.
    
    If running in Modal environment, automatically uses async mode with progress tracking.
    Returns task_id immediately and generation runs in background.
    Use /api/generate/stream/{task_id} for real-time progress updates.
    
    For backward compatibility, if not in Modal environment, runs synchronously.
    """
    worker, job_dictio, is_modal = get_modal_imports()
    
    # If Modal is available, use async mode with progress tracking
    if is_modal and worker is not None:
        task_id = str(uuid.uuid4())
        payload = request.model_dump()
        
        try:
            create_job_state(task_id)
            
            # Spawn A100 worker function asynchronously
            try:
                worker.spawn(task_id, payload)
            except Exception as e:
                update_job_state_error(task_id, str(e))
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to spawn generation job: {str(e)}"
                )
            
            return {
                "task_id": task_id,
                "status": "queued",
                "message": "Generation job submitted successfully. Use /api/generate/stream/{task_id} for progress updates."
            }
        except HTTPException:
            raise
        except Exception as e:
            # No fallback to sync mode - async generation is required
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create generation job: {str(e)}"
            ) from e
    else:
        # Not in Modal environment - async generation is required
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment). Please use /api/generate/async endpoint."
        )


@router.post("/generate/async")
def submit_generation_job(request: GenerationRequest) -> Dict[str, Any]:
    """
    Submit async generation job.
    
    Returns task_id immediately, generation runs in background on A100.
    Use /api/generate/status/{task_id} to poll status.
    """
    worker, job_dictio, is_modal = get_modal_imports()
    
    if not is_modal or worker is None:
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment)"
        )
    
    task_id = str(uuid.uuid4())
    payload = request.model_dump()
    
    try:
        create_job_state(task_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    # Spawn A100 worker function asynchronously
    try:
        worker.spawn(task_id, payload)
    except Exception as e:
        update_job_state_error(task_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn generation job: {str(e)}"
        )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Generation job submitted successfully"
    }


@router.get("/generate/status/{task_id}")
def get_generation_status(task_id: str) -> Dict[str, Any]:
    """
    Get generation job status.
    
    Returns current status, progress, current_step, total_steps, and error if any.
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_dictio[task_id]
    return {
        "task_id": task_id,
        "status": state.get("status", "unknown"),
        "progress": state.get("progress", 0.0),
        "current_step": state.get("current_step"),
        "total_steps": state.get("total_steps"),
        "error": state.get("error"),
    }


@router.get("/generate/result/{task_id}")
def get_generation_result(task_id: str) -> Dict[str, Any]:
    """
    Get generation job result.
    
    Returns result only if status is "done".
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_dictio[task_id]
    status = state.get("status", "unknown")
    
    if status == "error":
        raise HTTPException(
            status_code=500,
            detail=state.get("error", "Generation failed")
        )
    
    if status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed yet. Current status: {status}"
        )
    
    result = state.get("result")
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Result not available"
        )
    
    return {
        "task_id": task_id,
        "status": "done",
        "image": result,
        "debug_info": state.get("debug_info"),
    }


@router.get("/generate/stream/{task_id}")
async def stream_generation_progress(task_id: str):
    """
    Stream generation progress using Server-Sent Events (SSE).
    
    Returns real-time progress updates including:
    - status: queued, processing, done, error
    - progress: 0.0 to 1.0
    - current_step: current inference step
    - total_steps: total inference steps
    - error: error message if any
    """
    return StreamingResponse(
        generate_sse_events(task_id, include_heartbeat=False),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
