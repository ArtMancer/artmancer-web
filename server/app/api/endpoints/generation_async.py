"""
Async generation endpoints for job-based image generation.
Used by JobManagerService (CPU-only).
"""
from __future__ import annotations

import uuid
import time
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...models import GenerationRequest

router = APIRouter(prefix="/api/generate", tags=["generation-async"])

# Import run_generation function from modal_app
# This will be available when running in Modal environment
# Use lazy import to avoid circular dependencies
MODAL_ENV = False
run_generation = None
job_state_dictio = None

def _get_modal_imports():
    """Lazy import of Modal components."""
    global MODAL_ENV, run_generation, job_state_dictio
    if run_generation is None:
        try:
            import sys
            if 'modal_app' not in sys.modules:
                import modal_app
            else:
                modal_app = sys.modules['modal_app']
            run_generation = getattr(modal_app, 'run_generation', None)
            job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
            MODAL_ENV = run_generation is not None and job_state_dictio is not None
        except (ImportError, AttributeError):
            MODAL_ENV = False
            run_generation = None
            job_state_dictio = None
    return run_generation, job_state_dictio


@router.post("/async")
def submit_generation_job(request: GenerationRequest) -> Dict[str, Any]:
    """
    Submit async generation job.
    
    Returns task_id immediately, generation runs in background on A100.
    Use /api/generate/status/{task_id} to poll status.
    """
    run_gen, job_dictio = _get_modal_imports()
    if not MODAL_ENV or run_gen is None or job_dictio is None:
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment)"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Convert request to dict for payload
    payload = request.model_dump()
    
    # Initialize job state
    try:
        job_dictio[task_id] = {
            "status": "queued",
            "progress": 0.0,
            "error": None,
            "result": None,
            "debug_info": None,
            "created_at": time.time(),  # Timestamp for cleanup
        }
        
        # Add to tracking list
        tracking_key = "__job_tracking_list__"
        try:
            job_ids = job_dictio.get(tracking_key, [])
            if task_id not in job_ids:
                job_ids.append(task_id)
                job_dictio[tracking_key] = job_ids
        except Exception:
            pass  # Tracking list might not be available
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    # Spawn A100 worker function asynchronously
    try:
        run_gen.spawn(task_id, payload)
    except Exception as e:
        # Update state to error
        job_dictio[task_id] = {
            "status": "error",
            "progress": 0.0,
            "error": str(e),
            "result": None,
            "debug_info": None,
            "created_at": job_dictio.get(task_id, {}).get("created_at", time.time()),
        }
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn generation job: {str(e)}"
        )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Generation job submitted successfully"
    }


@router.get("/status/{task_id}")
def get_generation_status(task_id: str) -> Dict[str, Any]:
    """
    Get generation job status.
    
    Returns current status, progress, and error if any.
    """
    _, job_dictio = _get_modal_imports()
    if job_dictio is None:
        raise HTTPException(
            status_code=500,
            detail="Job state storage not available"
        )
    
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
        "error": state.get("error"),
    }


@router.get("/result/{task_id}")
def get_generation_result(task_id: str) -> Dict[str, Any]:
    """
    Get generation job result.
    
    Returns result only if status is "done".
    """
    _, job_dictio = _get_modal_imports()
    if job_dictio is None:
        raise HTTPException(
            status_code=500,
            detail="Job state storage not available"
        )
    
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


@router.post("/async/cancel/{task_id}")
def cancel_async_generation(task_id: str) -> Dict[str, Any]:
    """
    Cancel an async generation job.
    
    Sets cancellation flag in job_state_dictio, which will be checked
    by the A100 worker during generation.
    
    Args:
        task_id: Task identifier from async job submission
    
    Returns:
        Success message
    """
    _, job_dictio = _get_modal_imports()
    if job_dictio is None:
        raise HTTPException(
            status_code=500,
            detail="Job state storage not available"
        )
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    # Update job state to mark as cancelled
    state = job_dictio.get(task_id, {})
    state["cancelled"] = True
    state["status"] = "cancelled"
    job_dictio[task_id] = state
    
    return {
        "success": True,
        "message": f"Async generation task {task_id} marked for cancellation",
        "task_id": task_id
    }

