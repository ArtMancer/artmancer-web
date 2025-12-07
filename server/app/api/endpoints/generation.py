from __future__ import annotations

import uuid
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from ...models import GenerationRequest, GenerationResponse
from ...services.generation_service import GenerationService

router = APIRouter(prefix="/api", tags=["generation"])
service = GenerationService()

# Import qwen_worker function from modal_app
# This will be available when running in Modal environment
# Use lazy import to avoid circular dependencies
MODAL_ENV = False
qwen_worker = None

def _get_qwen_worker():
    """Lazy import of qwen_worker function."""
    global MODAL_ENV, qwen_worker
    if qwen_worker is None:
        try:
            import sys
            import importlib
            # Import modal_app module
            if 'modal_app' not in sys.modules:
                import modal_app
            else:
                modal_app = sys.modules['modal_app']
            qwen_worker = getattr(modal_app, 'qwen_worker', None)
            MODAL_ENV = qwen_worker is not None
        except (ImportError, AttributeError):
            MODAL_ENV = False
            qwen_worker = None
    return qwen_worker


@router.post("/generate")
def generate_image(request: GenerationRequest):
    """
    Generate image endpoint.
    
    If running in Modal environment, automatically uses async mode with progress tracking.
    Returns task_id immediately and generation runs in background.
    Use /api/generate/stream/{task_id} for real-time progress updates.
    
    For backward compatibility, if not in Modal environment, runs synchronously.
    """
    worker = _get_qwen_worker()
    
    # If Modal is available, use async mode with progress tracking
    if MODAL_ENV and worker is not None:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Convert request to dict for payload
        payload = request.model_dump()
        
        # Initialize job state
        try:
            import sys
            if 'modal_app' not in sys.modules:
                import modal_app
            else:
                modal_app = sys.modules['modal_app']
            job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
            if job_state_dictio is None:
                # Fallback to sync mode if job state not available
                return service.generate(request)
            
            import time
            job_state_dictio[task_id] = {
                "status": "queued",
                "progress": 0.0,
                "current_step": None,
                "total_steps": None,
                "error": None,
                "result": None,
                "debug_info": None,
                "created_at": time.time(),
            }
            
            # Add to tracking list
            tracking_key = "__job_tracking_list__"
            try:
                job_ids = job_state_dictio.get(tracking_key, [])
                if task_id not in job_ids:
                    job_ids.append(task_id)
                    job_state_dictio[tracking_key] = job_ids
            except Exception:
                pass
            
            # Spawn A100 worker function asynchronously
            try:
                worker.spawn(task_id, payload)
            except Exception as e:
                import time
                job_state_dictio[task_id] = {
                    "status": "error",
                    "progress": 0.0,
                    "error": str(e),
                    "result": None,
                    "debug_info": None,
                    "created_at": job_state_dictio.get(task_id, {}).get("created_at", time.time()),
                }
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to spawn generation job: {str(e)}"
                )
            
            # Return task_id for async mode
            return {
                "task_id": task_id,
                "status": "queued",
                "message": "Generation job submitted successfully. Use /api/generate/stream/{task_id} for progress updates."
            }
        except (ImportError, AttributeError):
            # Modal not available - async generation is required
            raise HTTPException(
                status_code=501,
                detail="Async generation not available (not running in Modal environment). Please use /api/generate/async endpoint."
            )
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
    worker = _get_qwen_worker()
    if not MODAL_ENV or worker is None:
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
        import sys
        if 'modal_app' not in sys.modules:
            import modal_app
        else:
            modal_app = sys.modules['modal_app']
        job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
        if job_state_dictio is None:
            raise HTTPException(
                status_code=500,
                detail="Job state storage not available"
            )
        import time
        job_state_dictio[task_id] = {
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
            job_ids = job_state_dictio.get(tracking_key, [])
            if task_id not in job_ids:
                job_ids.append(task_id)
                job_state_dictio[tracking_key] = job_ids
        except Exception:
            pass  # Tracking list might not be available
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    # Spawn A100 worker function asynchronously
    try:
        worker.spawn(task_id, payload)
    except Exception as e:
        # Update state to error
        import time
        job_state_dictio[task_id] = {
            "status": "error",
            "progress": 0.0,
            "error": str(e),
            "result": None,
            "debug_info": None,
            "created_at": job_state_dictio.get(task_id, {}).get("created_at", time.time()),  # Preserve or set timestamp
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


@router.get("/generate/status/{task_id}")
def get_generation_status(task_id: str) -> Dict[str, Any]:
    """
    Get generation job status.
    
    Returns current status, progress, current_step, total_steps, and error if any.
    """
    try:
        import sys
        if 'modal_app' not in sys.modules:
            import modal_app
        else:
            modal_app = sys.modules['modal_app']
        job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
        if job_state_dictio is None:
            raise HTTPException(
                status_code=500,
                detail="Job state storage not available"
            )
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    if task_id not in job_state_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_state_dictio[task_id]
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
    try:
        import sys
        if 'modal_app' not in sys.modules:
            import modal_app
        else:
            modal_app = sys.modules['modal_app']
        job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
        if job_state_dictio is None:
            raise HTTPException(
                status_code=500,
                detail="Job state storage not available"
            )
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    if task_id not in job_state_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_state_dictio[task_id]
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
    async def event_generator():
        try:
            import sys
            if 'modal_app' not in sys.modules:
                import modal_app
            else:
                modal_app = sys.modules['modal_app']
            job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
            if job_state_dictio is None:
                yield f"data: {json.dumps({'error': 'Job state storage not available'})}\n\n"
                return
            
            if task_id not in job_state_dictio:
                yield f"data: {json.dumps({'error': f'Task {task_id} not found'})}\n\n"
                return
            
            last_step = -1
            last_status = None
            last_loading_message = None
            first_message = True
            
            while True:
                state = job_state_dictio.get(task_id)
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

