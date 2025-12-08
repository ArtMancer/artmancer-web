"""
Async generation endpoints for job-based image generation.
Used by JobManagerService (CPU-only).
"""
from __future__ import annotations

import asyncio
import json
import uuid
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
from pathlib import Path
import base64

from ...models import GenerationRequest

router = APIRouter(prefix="/api/generate", tags=["generation-async"])

# Import qwen_worker function from modal_app
# This will be available when running in Modal environment
# Use lazy import to avoid circular dependencies
MODAL_ENV = False
qwen_worker = None
job_state_dictio = None
volume = None
VOL_MOUNT_PATH = "/checkpoints"  # fallback default

def _get_modal_imports():
    """Lazy import of Modal components."""
    global MODAL_ENV, qwen_worker, job_state_dictio, volume, VOL_MOUNT_PATH
    if qwen_worker is None:
        try:
            import sys
            if 'modal_app' not in sys.modules:
                import modal_app
            else:
                modal_app = sys.modules['modal_app']
            qwen_worker = getattr(modal_app, 'qwen_worker', None)
            job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
            volume = getattr(modal_app, 'volume', None)
            VOL_MOUNT_PATH = getattr(modal_app, 'VOL_MOUNT_PATH', VOL_MOUNT_PATH)
            MODAL_ENV = qwen_worker is not None and job_state_dictio is not None
        except (ImportError, AttributeError):
            MODAL_ENV = False
            qwen_worker = None
            job_state_dictio = None
            volume = None
    return qwen_worker, job_state_dictio


@router.post("/async")
def submit_generation_job(request: GenerationRequest) -> Dict[str, Any]:
    """
    Submit async generation job.
    
    Returns task_id immediately, generation runs in background on A100.
    Use /api/generate/status/{task_id} to poll status.
    """
    worker, job_dictio = _get_modal_imports()
    if not MODAL_ENV or worker is None or job_dictio is None:
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
        worker.spawn(task_id, payload)
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


# New: Multi-step sequential generation (sync) for reporting
@router.post("/multi-steps")
def generate_multi_steps(request: GenerationRequest) -> Dict[str, Any]:
    """
    Run generation sequentially for multiple inference steps (default: 5, 10, 25).
    Executes on GPU via qwen_worker.call (synchronous). Saves each image to Volume and returns base64 + path.
    """
    worker, job_dictio = _get_modal_imports()
    if not MODAL_ENV or worker is None:
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment)"
        )

    steps: List[int] = [5, 10, 25]
    base_request_id = str(uuid.uuid4())
    results: List[Dict[str, Any]] = []

    # Ensure volume mount path exists
    save_root = Path(VOL_MOUNT_PATH) / "reports" / base_request_id
    save_root.mkdir(parents=True, exist_ok=True)

    for step in steps:
        # Override num_inference_steps per call
        payload = request.model_copy(update={"num_inference_steps": step}).model_dump()
        task_id = f"{base_request_id}_s{step}"

        try:
            # Call GPU worker synchronously
            result = worker.call(task_id, payload)
            image_b64 = result.get("image")
            if not image_b64:
                raise ValueError("Worker returned no image")

            # Save to volume
            save_path = save_root / f"step_{step}.png"
            try:
                raw = base64.b64decode(image_b64)
                with open(save_path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                # Continue even if save fails; report error in path field
                save_path = None
                result["save_error"] = str(e)

            results.append(
                {
                    "step": step,
                    "image": image_b64,
                    "path": str(save_path) if save_path else None,
                    "request_id": result.get("request_id"),
                    "debug_info": result.get("debug_info"),
                }
            )
        except Exception as e:
            results.append(
                {
                    "step": step,
                    "error": str(e),
                }
            )

    return {
        "base_request_id": base_request_id,
        "steps": steps,
        "results": results,
    }

@router.get("/stream/{task_id}")
async def stream_generation_progress(task_id: str):
    """
    Stream generation progress using Server-Sent Events (SSE).
    
    Returns real-time progress updates including:
    - status: queued, processing, done, error
    - progress: 0.0 to 1.0
    - current_step: current inference step
    - total_steps: total inference steps
    - error: error message if any
    - loading_message: loading stage message
    """
    async def event_generator():
        try:
            _, job_dictio = _get_modal_imports()
            if job_dictio is None:
                yield f"data: {json.dumps({'error': 'Job state storage not available'})}\n\n"
                return
            
            if task_id not in job_dictio:
                yield f"data: {json.dumps({'error': f'Task {task_id} not found'})}\n\n"
                return
            
            last_step = -1
            last_status = None
            last_loading_message = None
            first_message = True
            
            while True:
                state = job_dictio.get(task_id)
                if not state:
                    yield f"data: {json.dumps({'error': f'Task {task_id} not found'})}\n\n"
                    break
                
                status = state.get("status", "unknown")
                current_step = state.get("current_step")
                total_steps = state.get("total_steps")
                progress = state.get("progress", 0.0)
                error = state.get("error")
                loading_message = state.get("loading_message")
                
                # Send initial state immediately, then only on changes
                # Also send if loading_message changes (for smooth progress updates)
                should_send = (
                    first_message or 
                    status != last_status or 
                    (current_step is not None and current_step != last_step) or
                    (loading_message is not None and loading_message != last_loading_message)
                )
                
                if should_send:
                    event_data = {
                        "task_id": task_id,
                        "status": status,
                        "progress": progress,
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "error": error,
                        "loading_message": loading_message,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    last_step = current_step if current_step is not None else last_step
                    last_status = status
                    last_loading_message = loading_message
                    first_message = False
                
                # Stop if done or error
                if status in ("done", "error", "cancelled"):
                    break
                
                # Poll every 200ms for updates
                await asyncio.sleep(0.2)
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

